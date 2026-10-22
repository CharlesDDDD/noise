"""Triton kernels for track 2's trainer: Triton versions of #29's Pallas kernels.

- `moment_query_scores` / `moment_key_scores`: the exact single-coordinate query (key) experiments as attention-weighted
  moments on tensor cores, degree 2 in the logit change (#29's kernels/exact_moments.py). The features (X, Y, X^2,
  Y^2, XY of the other side's rotated columns) are built per tile in the kernels from the other side's processed rows
  and interleaved into few products (queries [X Y X^2 Y^2] and [X Y]: six products per tile; keys [X Y] and
  [X^2 Y^2]: five), summed in the epilogue with the moved side's closed coefficients (from its raw rows); the values'
  credit sum_s A[s, t] hub_s rides in the key kernel, and the credits are stored by token in the head credits'
  all-to-all layout. Every running block's heads run in one launch (the blocks as rows of one grid).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# moment kernels
# PR #29's exact query / key experiments as attention-weighted moments on tensor cores (degree 2 in the logit change):
# per (moved position, direction), dl = A L + Bu X + Bw Y; the response's polynomial in dl splits into products of a
# per-pair weight (a c L^m, a L^m) and per-(other position, direction) features (X, Y, X^2, Y^2, XY); the sums
# over the other positions are [moved, other] x [other, 2J] products and the coefficients apply once at the end.


@triton.jit
def _side_features(side_ptr, offs, part, D: tl.constexpr, J: tl.constexpr, offs_j, scale):
    """The other side's rotated columns X, Y at rows `offs` of side (its processed rows [T, D] f32, at this row) for
    directions part J + offs_j: the pair (d mod D/2, d mod D/2 + D/2) each direction d moves, in logit units."""
    i = (part * J + offs_j) % (D // 2)
    x = tl.load(side_ptr + offs[:, None] * D + i[None, :]) * scale
    y = tl.load(side_ptr + offs[:, None] * D + (i + D // 2)[None, :]) * scale
    return x, y


@triton.jit
def _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, part, M, RAW_GPU,
                         RAW_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    """The moved side's closed coefficients A, Bu, Bw [BLOCK, J] at rows offs for directions part J + offs_j: a move of
    the raw row (raw [world, M, .] bf16 by token as received, GPUs RAW_GPU and tokens RAW_STRIDE apart; its squared
    norm sq and step size steps [BH, T] f32) along d changes
    the processed row by A (itself) + B (u, w) (the rotated pair), as #29's closed form (`closed_coefficients`)."""
    d = part * J + offs_j
    i = d % (D // 2)
    n = (row % BH).to(tl.int64) * T + offs  # token n of the block's flattened rows: GPU n // M, place n % M
    raw = tl.load(raw_ptr + (n // M * RAW_GPU + n % M * RAW_STRIDE)[:, None] + d[None, :]).to(tl.float32)
    sq = tl.load(sq_ptr + row * T + offs)[:, None]
    steps = tl.load(steps_ptr + row * T + offs)[:, None]
    change = (2 * steps * raw + steps * steps) / D / (sq / D + 1e-6)
    a = libdevice.expm1(-0.5 * libdevice.log1p(change))
    b = steps * libdevice.rsqrt(sq / D + 1e-6) * (1 + a)
    cos = tl.load(cos_ptr + offs[:, None] * (D // 2) + i[None, :])
    sin = tl.load(sin_ptr + offs[:, None] * (D // 2) + i[None, :])
    first = (d < D // 2)[None, :]
    return a, b * tl.where(first, cos, sin), b * tl.where(first, -sin, cos)


@triton.jit
def _store_by_token(out_ptr, x, row, BH, T, offs, part, M, W_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    """Store x [BLOCK, J] (row `row`'s tokens offs, dims part J ..) into out [world, .., M, D] by token: token n = row T
    + t of the flattened rows sits at GPU n // M, place n % M (the head credits' all-to-all layout; W_STRIDE apart)."""
    n = (row % BH).to(tl.int64) * T + offs
    tl.store(out_ptr + (n // M * W_STRIDE + n % M * D + part * J)[:, None] + offs_j[None, :], x)


@triton.jit
def _split4(acc, BLOCK: tl.constexpr, J: tl.constexpr):
    """The products with the interleaved features [X Y X^2 Y^2] apart: (X, Y, X^2, Y^2), each [BLOCK, J]."""
    first, second = tl.split(tl.reshape(acc, (BLOCK, J, 2, 2)))  # [X Y] and [X^2 Y^2]
    x, y = tl.split(first)
    xx, yy = tl.split(second)
    return x, y, xx, yy


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr,
                                 hubout_ptr, steps_ptr, out_ptr, win_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
                                 RAW_STRIDE, M, OUT_BLK, W_STRIDE,
                                 D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S  # the longest causal sums first
    row, part = i // REPEAT, i % REPEAT  # rows of every block's batch, BH a block
    blk = row // BH
    window = tl.load(win_ptr + blk)
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_s = s0 + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
    lse = tl.load(lse_ptr + row * T + offs_s)
    hub_out = tl.load(hubout_ptr + row * T + offs_s)
    side = side_ptr + base  # the processed keys [BH, T, D] (f32): their rotated columns
    # numerator / denominator products per (power of L, feature), the features interleaved per direction so that
    # each weight takes one product: [X Y X^2 Y^2] (n4, d4), [X Y] (l2, m2), and the shared XY (nxy, dxy)
    n4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    d4 = tl.zeros((BLOCK_S, 4 * J), tl.float32)
    nxy = tl.zeros((BLOCK_S, J), tl.float32)
    dxy = tl.zeros((BLOCK_S, J), tl.float32)
    l2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    m2 = tl.zeros((BLOCK_S, 2 * J), tl.float32)
    s1n = tl.zeros((BLOCK_S,), tl.float32)
    s2n = tl.zeros((BLOCK_S,), tl.float32)
    s1d = tl.zeros((BLOCK_S,), tl.float32)
    s2d = tl.zeros((BLOCK_S,), tl.float32)
    first = tl.maximum(s0 - window, 0) // BLOCK_T
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    for it in range(first, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale  # [BS, BT]
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
        c = tl.dot(hub, tl.trans(v)) - hub_out[:, None]
        x, y = _side_features(side, offs_t, part, D, J, offs_j, scale)
        f2 = tl.reshape(tl.join(x, y), (BLOCK_T, 2 * J)).to(tl.bfloat16)
        f4 = tl.reshape(tl.join(tl.join(x, y), tl.join(x * x, y * y)), (BLOCK_T, 4 * J)).to(tl.bfloat16)
        fxy = (x * y).to(tl.bfloat16)
        wn = a * c
        wnl = (wn * logits).to(tl.bfloat16)
        al = (a * logits).to(tl.bfloat16)
        wnb, ab = wn.to(tl.bfloat16), a.to(tl.bfloat16)
        n4 += tl.dot(wnb, f4)
        nxy += tl.dot(wnb, fxy)
        l2 += tl.dot(wnl, f2)
        d4 += tl.dot(ab, f4)
        dxy += tl.dot(ab, fxy)
        m2 += tl.dot(al, f2)
        s1n += tl.sum(wn * logits, axis=1)
        s2n += tl.sum(wn * logits * logits, axis=1)
        s1d += tl.sum(a * logits, axis=1)
        s2d += tl.sum(a * logits * logits, axis=1)
    ca, bu, bw = _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_s, part, M,
                                      RAW_GPU, RAW_STRIDE, D,
                                      J, offs_j)  # the raw queries'
    nx0, ny0, nx1, ny1 = _split4(n4, BLOCK_S, J)
    dx0, dy0, dx1, dy1 = _split4(d4, BLOCK_S, J)
    lx0, ly0 = tl.split(tl.reshape(l2, (BLOCK_S, J, 2)))
    mx0, my0 = tl.split(tl.reshape(m2, (BLOCK_S, J, 2)))
    # X side: P = Bu, Q = Bw; Y side: P = Bw, Q = Bu (degree 2: P, P^2 / 2, PQ / 2 each, the L-sums' A / 2, A^2 / 4)
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
          + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
          + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_s)
    _store_by_token(out_ptr, zn / (1.0 + zd) / steps[:, None], row, BH, T, offs_s, part, M, W_STRIDE, D, J, offs_j)


def moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, windows, repeat, block_s=64,
                        block_t=32, num_warps=4, num_stages=2):
    """Query credits of every running block at once into out: q, k, v, hub [K BH, T, D] bf16 (q, k processed; the K
    blocks' batches of BH rows stacked); raw [world, K, M, .] bf16 (the raw queries by token as received, any strides);
    sq, steps [K BH, T] f32 their squared norms and move sizes (the scores are divided by them); side [K BH, T, D] f32
    the processed keys; cos, sin [T, D / 2] the rotary tables; lse, hub_out [K BH, T] f32; out [world, K, M, D]
    (f32 or bf16) by token (see `_store_by_token`), M the tokens per GPU; windows [K] int32 each block's attention
    window. The features and the closed coefficients are built tile by tile in the kernel, D / repeat directions per
    program; the query blocks run longest causal sum first."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_queries_kernel[(rows * repeat, t // block_s)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps,
                                                          out, windows, rows // blocks, t, d ** -0.5, repeat,
                                                          raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
                                                          out.stride(1), out.stride(0), D=d, J=d // repeat,
                                                          BLOCK_T=block_t, BLOCK_S=block_s, num_warps=num_warps,
                                                          num_stages=num_stages)


@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
                              steps_ptr, out_ptr, value_ptr, win_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
                              RAW_STRIDE, M, OUT_BLK, W_STRIDE,
                              D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    row, part = i // REPEAT, i % REPEAT  # rows of every block's batch, BH a block
    blk = row // BH
    window = tl.load(win_ptr + blk)
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    side = side_ptr + base  # the processed queries [BH, T, D] (f32): their rotated columns
    p2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w1 [X Y] (interleaved per direction)
    q2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w2 [X^2 Y^2]
    qxy = tl.zeros((BLOCK_T, J), tl.float32)  # w2 XY (shared)
    r2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)  # w2 L [X Y]
    val = tl.zeros((BLOCK_T, J), tl.float32)  # the value credit sum_s A[s, t] hub_s, this row's J of the D dims
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    first = t0 // BLOCK_S
    last = tl.minimum((t0 + BLOCK_T - 1 + window) // BLOCK_S + 1, T // BLOCK_S)
    for it in range(first, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub_part = tl.load(hub_ptr + base + offs_s[:, None] * D + part * J + offs_j[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale  # [BT, BS]
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where((lag >= 0) & (lag <= window), tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        x, y = _side_features(side, offs_s, part, D, J, offs_j, scale)
        w1 = a * c  # ch / (1 + ch) = a dl + (a / 2 - a^2) dl^2 + ...
        w2 = (0.5 * a - a * a) * c
        w1b, w2b, w2lb = w1.to(tl.bfloat16), w2.to(tl.bfloat16), (w2 * logits).to(tl.bfloat16)
        f2 = tl.reshape(tl.join(x, y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        g2 = tl.reshape(tl.join(x * x, y * y), (BLOCK_S, 2 * J)).to(tl.bfloat16)
        val += tl.dot(a.to(tl.bfloat16), hub_part)
        p2 += tl.dot(w1b, f2)
        q2 += tl.dot(w2b, g2)
        qxy += tl.dot(w2b, (x * y).to(tl.bfloat16))
        r2 += tl.dot(w2lb, f2)
        s1 += tl.sum(w1 * logits, axis=1)
        s2 += tl.sum(w2 * logits * logits, axis=1)
    ca, bu, bw = _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs_t, part, M,
                                      RAW_GPU, RAW_STRIDE, D,
                                      J, offs_j)  # the raw keys'
    px, py = tl.split(tl.reshape(p2, (BLOCK_T, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BLOCK_T, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BLOCK_T, J, 2)))
    z = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + qxy * (2.0 * bu * bw) + 2.0 * ca * (rx * bu + ry * bw)
         + s1[:, None] * ca + s2[:, None] * (ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_t)
    _store_by_token(out_ptr, z / steps[:, None], row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)
    _store_by_token(value_ptr + blk.to(tl.int64) * OUT_BLK, val, row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)


def moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, windows, repeat,
                      block_t=64, block_s=64, num_warps=4, num_stages=2):
    """Key credits of every running block into out: raw, sq, steps of the raw keys, side the processed queries; as
    above. Also the values' credit sum_s A[s, t] hub_s into values (out's layout), each of a head row's programs
    D / repeat of its D dims."""
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_keys_kernel[(rows * repeat, t // block_t)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
                                                       values, windows, rows // blocks, t, d ** -0.5, repeat,
                                                       raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
                                                       out.stride(1), out.stride(0), D=d, J=d // repeat,
                                                       BLOCK_T=block_t, BLOCK_S=block_s, num_warps=num_warps,
                                                       num_stages=num_stages)
