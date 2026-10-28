"""Gluon (Hopper) chain kernels: block A (a plain MODE 1 visit, or the root visit) and the MODE 2 readout visit
(block B, the last block) back to back per tile, the tile's delta kept in shared memory between them (no HBM round
trip of block A's output).

fused_visit_readout_fast is numerically visit.fused_visit_fast followed by visit.fused_readout_visit_fast,
fused_root_readout_fast visit_root.fused_root_visit_fast followed by it (the same per-element ops in the same order):
block A's bf16(bf16(d + change) * scale) is written into the tile's delta chunks in shared memory and read there as
block B's delta. Persistent CTAs (two per SM) walk tiles of BM = 64 moves of one token in token-major order; the
routed rows of both blocks stream through one cp.async ring (per tile: Fc A (not for the root), Wo A, Fc B, Wo B).
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    async_copy as cp, fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

EPS = torch.finfo(torch.float32).eps
CTAS_PER_SM = 2
_SMS = {}

# runtime chunk loops: unrolled (as visit.py's plain visits) the kernel is 1.4x slower and computes wrong, unrepeatable
# block B sums with E = 1 or NS = 2 (illegal instruction at n = 17; unresolved, bitwise right only at E = 0, NS = 3)
# DEFER 1: the gates' norm sums as per-thread partials, one reduction per tile (-3-4%; fp32 order differs from the
# two-kernel path: stats as close to an fp64 readout as Triton's; bitwise equal to it only with DEFER 0)
CONFIG = dict(BM=64, NS=3, E=0, UNROLL=False, DEFER=1, num_warps=4)
# the root's change goes to the wgmma operand layout through an 8 KB scratch buffer: no spare delta slot for 2 CTAs/SM
ROOT_CONFIG = dict(BM=64, NS=3, E=0, UNROLL=False, DEFER=1, num_warps=4)


@gluon.jit
def _issue_rows(buf, rows, cols, c0):
    cp.async_copy_global_to_shared(buf, rows[:, None] + (c0 + cols)[None, :])
    cp.commit_group()


@gluon.jit
def _reload(delta_desc, dbar, dsm, slot, c, row0, pred, NBYTES: gl.constexpr, BK: gl.constexpr):
    """TMA of delta chunk c of the tile at row0 into shared memory slot `slot`."""
    mbarrier.expect(dbar.index(slot), NBYTES, pred=pred)
    tma.async_copy_global_to_shared(delta_desc, [row0, c * BK], dbar.index(slot), dsm.index(slot), pred=pred)


@gluon.jit
def _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, token, pred, D: gl.constexpr, num_warps: gl.constexpr):
    """MODE 2: cp.async of the token's 5 readout vectors into vsm; every thread's completion arrives on vbar."""
    lay: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
    dd = gl.arange(0, D, lay)
    off = token * D + dd
    live = (dd < D) & pred
    cp.async_copy_global_to_shared(vsm.index(0), XL + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(1), Target + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(2), Mu + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(3), Nu + off, mask=live)
    cp.async_copy_global_to_shared(vsm.index(4), Diag + off, mask=live)
    cp.mbarrier_arrive(vbar, increment_count=False)


@gluon.jit
def _gate(j, acc, sq_x, dots, moved_sq, x_next, x_after, h, dbar, dsm, wsm, X, Clean, it, base, fc_rows, wo_rows,
    cols, xcols, token, nxt_token,
    D: gl.constexpr, R: gl.constexpr, BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr,
    LOADED: gl.constexpr, mma1: gl.constexpr, s_layout: gl.constexpr, BM: gl.constexpr = 64, DEFER: gl.constexpr = False):
    """Gate GEMM chunk j of one block: acc += d @ Fc[routes]^T over columns j; chunk j's norm sums run under its
    wgmma. LOADED: chunk j comes in by TMA (block A); otherwise the previous block wrote it (block B)."""
    g = it * KC + j
    slot = g % (KC + E)
    if LOADED:
        mbarrier.wait(dbar.index(slot), (g // (KC + E)) & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    acc = warpgroup_mma_wait(0, deps=[acc])
    gl.thread_barrier()
    _issue_rows(wsm.index((base + j + NS - 1) % NS), gl.where(j + NS - 1 < KC, fc_rows, wo_rows), cols,
      ((j + NS - 1) % KC) * BK)
    acc = warpgroup_mma(dsm.index(slot), wsm.index((base + j) % NS).permute((1, 0)), acc, is_async=True)
    d = dsm.index(slot).load(s_layout).to(gl.float32)
    x = x_next
    x_next = x_after
    x_after = gl.load(X + gl.where(j + 2 < KC, token, nxt_token) * D + ((j + 2) % KC) * BK + xcols)
    if j == KC - 2:
        h = gl.load(Clean + token * R + gl.arange(0, R, gl.SliceLayout(0, mma1))).to(gl.float32)
    if DEFER:  # per-thread partials (x * x per column, d sums per row and 8-column group), reduced once per tile
        sq_x += x * x
        dots += gl.sum(gl.reshape(d * x[None, :], [BM, BK // 8, 8]), 2)
        moved_sq += gl.sum(gl.reshape(d * d, [BM, BK // 8, 8]), 2)
    else:
        sq_x += gl.sum(gl.expand_dims(x * x, 0), 1)
        dots += gl.sum(d * x[None, :], 1)
        moved_sq += gl.sum(d * d, 1)
    return acc, sq_x, dots, moved_sq, x_next, x_after, h


@gluon.jit
def _gate_init(BM: gl.constexpr, BK: gl.constexpr, DEFER: gl.constexpr, s_layout: gl.constexpr):
    """Zero norm sums (sq_x, dots, moved_sq), or (DEFER) their per-thread partials."""
    if DEFER:
        part = gl.sum(gl.reshape(gl.zeros([BM, BK], gl.float32, s_layout), [BM, BK // 8, 8]), 2)
        return gl.zeros([BK], gl.float32, gl.SliceLayout(0, s_layout)), part, part
    else:
        rows: gl.constexpr = gl.SliceLayout(1, s_layout)
        return gl.zeros([1], gl.float32, rows), gl.zeros([BM], gl.float32, rows), gl.zeros([BM], gl.float32, rows)


@gluon.jit
def _gate_done(sq_x, dots, moved_sq, DEFER: gl.constexpr, mma1: gl.constexpr):
    """The norm sums per row, in the gate accumulator's row layout."""
    rows: gl.constexpr = gl.SliceLayout(1, mma1)
    if DEFER:
        sq_x = gl.sum(gl.expand_dims(sq_x, 0), 1)
        dots = gl.sum(dots, 1)
        moved_sq = gl.sum(moved_sq, 1)
    return gl.convert_layout(sq_x, rows), gl.convert_layout(dots, rows), gl.convert_layout(moved_sq, rows)


@gluon.jit
def _change(acc, sq_x, dots, moved_sq, h, D: gl.constexpr, EPSILON: gl.constexpr, mma1: gl.constexpr,
    mma2: gl.constexpr):
    """The routed MLP's change relu(h + dh)^2 - relu(h)^2 as the output GEMM's register A operand."""
    inv0 = gl.rsqrt(sq_x / D + EPSILON)
    inv_moved = gl.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
    inv = gl.convert_layout(inv_moved, gl.SliceLayout(1, mma1))
    ratio = gl.convert_layout(inv_moved / inv0 - 1, gl.SliceLayout(1, mma1))
    dh = inv[:, None] * acc.to(gl.bfloat16).to(gl.float32) + ratio[:, None] * h[None, :]
    up = gl.maximum(h[None, :] + dh, 0)
    down = gl.maximum(h[None, :], 0)
    change = ((up - down) * (up + down)).to(gl.bfloat16)
    return gl.convert_layout(change, gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))


@gluon.jit
def _output_keep(c, dbar, dsm, wsm, change, zero, step, scale, it, base, wo_rows, next_rows, cols,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, ROOT: gl.constexpr, mma2: gl.constexpr):
    """Block A's output chunk c: bf16(bf16(d + change @ Wo[routes][:, c]) * scale) back into delta chunk c (base:
    the ring slot of the output's first item). ROOT: d = step * the frame chunk TMA-loaded there (exact: +-1)."""
    g = it * KC + c
    slot = g % (KC + E)
    if ROOT:
        mbarrier.wait(dbar.index(slot), (g // (KC + E)) & 1)
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    if ROOT:
        old = step * dsm.index(slot).load(mma2).to(gl.float32)
    else:
        old = dsm.index(slot).load(mma2).to(gl.float32)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    dsm.index(slot).store((value.to(gl.float32) * scale).to(gl.bfloat16))  # each thread its own elements


@gluon.jit
def _output_readout(c, s0, s1, s2, s3, s4, s5, delta_desc, dbar, dsm, wsm, vsm, vbar, change, zero, it, base,
    wo_rows, next_rows, cols, nxt_row0, has_next,
    BK: gl.constexpr, KC: gl.constexpr, NS: gl.constexpr, E: gl.constexpr, NBYTES: gl.constexpr,
    mma2: gl.constexpr, BM: gl.constexpr = 64, DEFER: gl.constexpr = False):
    """Block B's output chunk c, reduced to the readout moments; its slot then takes the next tile's chunk c (base:
    the ring slot of the output's first item)."""
    cp.wait_group(NS - 2)
    fence_async_shared()
    gl.thread_barrier()
    _issue_rows(wsm.index((base + c + NS - 1) % NS), gl.where(c + NS - 1 < KC, wo_rows, next_rows), cols,
      ((c + NS - 1) % KC) * BK)
    slot = (it * KC + c) % (KC + E)
    old = dsm.index(slot).load(mma2)
    out = warpgroup_mma(change, wsm.index((base + c) % NS), zero, use_acc=False, is_async=True)
    out = warpgroup_mma_wait(0, deps=[out, change])[0]
    value = (old.to(gl.float32) + out.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16)
    if c == 0:
        mbarrier.wait(vbar, it & 1)
    fence_async_shared()
    gl.thread_barrier()  # every warp has read chunk c: its slot takes the next tile's chunk
    _reload(delta_desc, dbar, dsm, (it * KC + c + KC) % (KC + E), c, nxt_row0, has_next, NBYTES, BK)
    v = value.to(gl.float32)
    vcols: gl.constexpr = gl.SliceLayout(0, mma2)
    s0 += _row_sum(v * vsm.index(0).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s1 += _row_sum(v * v, BM, BK, DEFER)
    s2 += _row_sum(v * vsm.index(1).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s3 += _row_sum(v * vsm.index(2).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s4 += _row_sum(v * vsm.index(3).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    s5 += _row_sum(v * v * vsm.index(4).slice(c * BK, BK).load(vcols)[None, :], BM, BK, DEFER)
    return s0, s1, s2, s3, s4, s5


@gluon.jit
def _row_done(part, mma2: gl.constexpr):
    return gl.convert_layout(gl.sum(part, 1), gl.SliceLayout(1, mma2))


@gluon.jit
def _row_sum(t, BM: gl.constexpr, BK: gl.constexpr, DEFER: gl.constexpr):
    """Row sums of an accumulator-layout chunk, or (DEFER) per-thread partials: column 8 g + k of a thread's rows sits
    in its registers for all 8 g, so summing over g needs no shuffles (reduced once per tile)."""
    if DEFER:
        return gl.sum(gl.reshape(t, [BM, 8, BK // 8]), 1)
    else:
        return gl.sum(t, 1)


@gluon.jit
def _chain_kernel(delta_desc, XA, FcA, WoA, RoutesA, CleanA, ScaleA, XB, FcB, WoB, RoutesB, CleanB,
    XL, Target, Mu, Nu, Diag, Stats, Steps, Proj, Inv0, Inv, TILES,
    P: gl.constexpr, D: gl.constexpr, R: gl.constexpr, BM: gl.constexpr, NS: gl.constexpr, E: gl.constexpr,
    ROOT: gl.constexpr, UNROLL: gl.constexpr, EPSILON: gl.constexpr, num_warps: gl.constexpr,
    DEFER: gl.constexpr = 0):
    """Block A (a plain visit, or ROOT: the root visit, delta_desc then the bf16 frame (P, D)) and block B (the
    readout visit) per tile. DEFER: 1 the gates' norm sums, 2 the readout sums, 3 both kept as per-thread partials
    across chunks (one reduction per tile; a different fp32 order)."""
    BK: gl.constexpr = 64
    DG: gl.constexpr = DEFER % 2 == 1
    DR: gl.constexpr = DEFER >= 2
    KC: gl.constexpr = D // BK
    MT: gl.constexpr = P // BM
    NBYTES: gl.constexpr = BM * BK * 2
    GA: gl.constexpr = 0 if ROOT else KC  # ring items per tile: block A's gate (none for the root), then 3 x KC
    IT: gl.constexpr = GA + 3 * KC
    gl.static_assert(NS >= 2 and NS <= KC + 1 and E <= 1)
    G = gl.num_programs(0)
    pid = gl.program_id(0)

    dsm = gl.allocate_shared_memory(gl.bfloat16, [KC + E, BM, BK], delta_desc.layout)
    wsm = gl.allocate_shared_memory(gl.bfloat16, [NS, R, BK],
      gl.NVMMASharedLayout.get_default_for([R, BK], gl.bfloat16))
    dbar = gl.allocate_shared_memory(gl.int64, [KC + E, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(KC + E):
        mbarrier.init(dbar.index(i), count=1)
    vsm = gl.allocate_shared_memory(gl.float32, [5, D],
      gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0]))
    vbar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(vbar, count=num_warps * 32)
    _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, pid // MT, True, D, num_warps)
    for i in gl.static_range(KC):  # the first tile's delta (ROOT: frame) chunks
        _reload(delta_desc, dbar, dsm, i, i, (0 if ROOT else (pid // MT) * P) + (pid % MT) * BM, True, NBYTES, BK)

    g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    rr = gl.arange(0, R, gl.SliceLayout(1, g_layout))
    cols = gl.arange(0, BK, gl.SliceLayout(0, g_layout))
    chosen_a = gl.load(RoutesA + (pid // MT) * R + rr).to(gl.int32)
    first_a = WoA if ROOT else FcA  # block A's first ring items
    for i in gl.static_range(NS - 1):
        _issue_rows(wsm.index(i), first_a + chosen_a * D, cols, i * BK)

    mma1: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, R, 16])
    mma2: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BK, 16])
    s_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, s_layout)
    q_layout: gl.constexpr = gl.BlockedLayout([4, 1], [16, 2], [1, num_warps], [0, 1])  # ROOT: Proj move-contiguous
    qrows = gl.arange(0, BM, gl.SliceLayout(1, q_layout))
    qcols = gl.arange(0, R, gl.SliceLayout(0, q_layout))
    xcols = gl.arange(0, BK, gl.SliceLayout(0, s_layout))
    if ROOT:
        xa_next = 0.0
        xa_after = 0.0
    else:
        xa_next = gl.load(XA + (pid // MT) * D + xcols)
        xa_after = gl.load(XA + (pid // MT) * D + BK + xcols)
    xb_next = gl.load(XB + (pid // MT) * D + xcols)
    xb_after = gl.load(XB + (pid // MT) * D + BK + xcols)
    scale = gl.load(ScaleA).to(gl.float32)
    zero = gl.zeros([BM, BK], gl.float32, mma2)

    it = 0
    for tile in range(pid, TILES, G):
        token = tile // MT
        move0 = (tile % MT) * BM
        row0 = token * P + move0
        base = it * IT  # ring position of the tile's first item
        nxt_tile = tile + G
        has_next = nxt_tile < TILES
        nxt_token = gl.where(has_next, nxt_tile // MT, token)
        nxt_move0 = (nxt_tile % MT) * BM
        chosen_b = gl.load(RoutesB + token * R + rr).to(gl.int32)
        chosen_next = gl.load(RoutesA + nxt_token * R + rr).to(gl.int32)

        if ROOT:  # block A's gate change from the projection: moved = bf16(q * bf16(step)), relu^2 at the clean h
            step = gl.load(Steps + token)
            q = gl.load(Proj + gl.load(RoutesA + token * R + qcols).to(gl.int32)[None, :] * P + (move0 + qrows)[:, None])
            moved = (q * step.to(gl.bfloat16).to(gl.float32)).to(gl.bfloat16).to(gl.float32)
            inv = gl.load(Inv + row0 + qrows)
            inv0 = gl.load(Inv0 + token)
            h = gl.load(CleanA + token * R + qcols).to(gl.float32)
            dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
            up = gl.maximum(h[None, :] + dh, 0)
            down = gl.maximum(h[None, :], 0)
            change = gl.convert_layout(((up - down) * (up + down)).to(gl.bfloat16),
              gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2))
        else:  # block A: gate GEMM over the TMA-loaded tile
            step = 0.0
            sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
            acc = gl.zeros([BM, R], gl.float32, mma1)
            h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
            if not UNROLL:
                acc = warpgroup_mma_init(acc)
            gate_a = (dbar, dsm, wsm, XA, CleanA, it, base % NS, FcA + chosen_a * D, WoA + chosen_a * D, cols, xcols,
              token, nxt_token)
            if UNROLL:
                for j in gl.static_range(KC):
                    acc, sq_x, dots, moved_sq, xa_next, xa_after, h = _gate(j, acc, sq_x, dots, moved_sq, xa_next,
                      xa_after, h, *gate_a, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=True, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            else:
                for j in range(KC):
                    acc, sq_x, dots, moved_sq, xa_next, xa_after, h = _gate(j, acc, sq_x, dots, moved_sq, xa_next,
                      xa_after, h, *gate_a, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=True, mma1=mma1,
                      s_layout=s_layout, BM=BM, DEFER=DG)
            acc = warpgroup_mma_wait(0, deps=[acc])
            sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
            change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2)
        # block A's output, written back into the tile; the ring's tail fetches block B's first c_fc chunks
        out_a = (dbar, dsm, wsm, change, zero, step, scale, it, (base + GA) % NS, WoA + chosen_a * D,
          FcB + chosen_b * D, cols)
        if UNROLL:
            for c in gl.static_range(KC):
                _output_keep(c, *out_a, BK=BK, KC=KC, NS=NS, E=E, ROOT=ROOT, mma2=mma2)
        else:
            for c in range(KC):
                _output_keep(c, *out_a, BK=BK, KC=KC, NS=NS, E=E, ROOT=ROOT, mma2=mma2)

        # block B: gate GEMM over block A's output in shared memory, output reduced to the readout moments
        sq_x, dots, moved_sq = _gate_init(BM, BK, DG, s_layout)
        acc = gl.zeros([BM, R], gl.float32, mma1)
        h = gl.zeros([R], gl.float32, gl.SliceLayout(0, mma1))
        if not UNROLL:
            acc = warpgroup_mma_init(acc)
        gate_b = (dbar, dsm, wsm, XB, CleanB, it, (base + GA + KC) % NS, FcB + chosen_b * D, WoB + chosen_b * D, cols,
          xcols, token, nxt_token)
        if UNROLL:
            for j in gl.static_range(KC):
                acc, sq_x, dots, moved_sq, xb_next, xb_after, h = _gate(j, acc, sq_x, dots, moved_sq, xb_next,
                  xb_after, h, *gate_b, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=False, mma1=mma1,
                  s_layout=s_layout, BM=BM, DEFER=DG)
        else:
            for j in range(KC):
                acc, sq_x, dots, moved_sq, xb_next, xb_after, h = _gate(j, acc, sq_x, dots, moved_sq, xb_next,
                  xb_after, h, *gate_b, D=D, R=R, BK=BK, KC=KC, NS=NS, E=E, LOADED=False, mma1=mma1,
                  s_layout=s_layout, BM=BM, DEFER=DG)
        acc = warpgroup_mma_wait(0, deps=[acc])
        sq_x, dots, moved_sq = _gate_done(sq_x, dots, moved_sq, DG, mma1)
        change = _change(acc, sq_x, dots, moved_sq, h, D, EPSILON, mma1, mma2)
        s0 = _row_sum(zero, BM, BK, DR)
        s1 = s0
        s2 = s0
        s3 = s0
        s4 = s0
        s5 = s0
        # the ring's tail fetches the next tile's first block A items; the consumed chunks reload with its delta
        out_b = (delta_desc, dbar, dsm, wsm, vsm, vbar, change, zero, it, (base + GA + 2 * KC) % NS,
          WoB + chosen_b * D, first_a + chosen_next * D, cols, (0 if ROOT else nxt_token * P) + nxt_move0, has_next)
        for c in gl.static_range(KC):  # unrolled: the readout vectors are sliced at constant offsets
            s0, s1, s2, s3, s4, s5 = _output_readout(c, s0, s1, s2, s3, s4, s5, *out_b, BK=BK, KC=KC, NS=NS, E=E,
              NBYTES=NBYTES, mma2=mma2, BM=BM, DEFER=DR)
        if DR:
            s0, s1, s2, s3, s4, s5 = (_row_done(s0, mma2), _row_done(s1, mma2), _row_done(s2, mma2),
              _row_done(s3, mma2), _row_done(s4, mma2), _row_done(s5, mma2))
        gl.thread_barrier()  # every warp is done with vsm: the next tile's vectors in flight
        _vec_issue(vsm, vbar, XL, Target, Mu, Nu, Diag, nxt_token, has_next, D, num_warps)
        stats_rows = Stats + (row0 + gl.arange(0, BM, gl.SliceLayout(1, mma2))).to(gl.int64) * 6
        gl.store(stats_rows, s0)
        gl.store(stats_rows + 1, s1)
        gl.store(stats_rows + 2, s2)
        gl.store(stats_rows + 3, s3)
        gl.store(stats_rows + 4, s4)
        gl.store(stats_rows + 5, s5)
        chosen_a = chosen_next
        it += 1
    cp.wait_group(0)
    for i in gl.static_range(KC + E):
        mbarrier.invalidate(dbar.index(i))
    mbarrier.invalidate(vbar)


def launch_chain(n, p, d, block_a, block_b, vectors, stats, root=None, **overrides):
    """block_a: (delta, x, fc, wo, routes, clean, scale), or with root = (steps, projection, inv0, inv) (frame,
    None, None, wo, routes, clean, scale) with the bf16 frame (P, D); block_b: (x, fc, wo, routes, clean)."""
    first, xa, fca, woa, routesa, cleana, scalea = block_a
    r = routesa.shape[1]
    assert block_b[3].shape[1] == r == 64, 'R = 64 only (the shared-memory budget is sized for it)'
    cfg = dict(CONFIG if root is None else ROOT_CONFIG, **overrides)
    bm = cfg.pop('BM')
    assert p % bm == 0 and d % 64 == 0 and first.is_contiguous() and first.dtype == torch.bfloat16
    layout = gl.NVMMASharedLayout.get_default_for([bm, 64], gl.bfloat16)
    first_desc = TensorDescriptor.from_tensor(first.view(-1, d), [bm, 64], layout)
    dev = woa.device.index
    if dev not in _SMS:
        _SMS[dev] = (132 if woa.device.type == 'cpu'
          else torch.cuda.get_device_properties(woa.device).multi_processor_count)
    tiles = n * (p // bm)
    pad = stats  # unused pointer arguments
    _chain_kernel[(min(tiles, CTAS_PER_SM * _SMS[dev]),)](first_desc, pad if xa is None else xa,
      pad if fca is None else fca, woa, routesa, cleana, scalea, *block_b, *vectors, stats, *(root or (pad,) * 4),
      tiles, p, d, r, bm, ROOT=root is not None, EPSILON=EPS, **cfg)
    return stats


@torch.library.custom_op('visit_kernel::fused_root_readout_fast', mutates_args=())
def fused_root_readout_fast(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo_a: torch.Tensor,
    routes_a: torch.Tensor, clean_a: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor, scale_a: torch.Tensor,
    x_b: torch.Tensor, fc_b: torch.Tensor, wo_b: torch.Tensor, routes_b: torch.Tensor, clean_b: torch.Tensor,
    xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """fused_readout_visit_fast(fused_root_visit_fast(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv,
    scale_a, None)[0], x_b, fc_b, wo_b, routes_b, clean_b, xl, target, mu, nu, diag) in one kernel: the readout
    moments (n, P, 6) fp32; basis (P, D), the signed Hadamard frame, read as bf16 (exact: its entries are +-1)."""
    f32 = lambda z: z.float().contiguous()
    n, p, d = steps.shape[0], basis.shape[0], wo_a.shape[1]
    stats = torch.empty((n, p, 6), device=wo_a.device, dtype=torch.float32)
    return launch_chain(n, p, d, (basis.to(torch.bfloat16).contiguous(), None, None, wo_a.contiguous(),
      routes_a.contiguous(), f32(clean_a), scale_a), (f32(x_b), fc_b.contiguous(), wo_b.contiguous(),
      routes_b.contiguous(), f32(clean_b)), tuple(f32(z) for z in (xl, target, mu, nu, diag)), stats,
      root=(f32(steps), f32(projection), f32(inv0), f32(inv)))


@fused_root_readout_fast.register_fake
def _(steps, basis, projection, wo_a, routes_a, clean_a, inv0, inv, scale_a, x_b, fc_b, wo_b, routes_b, clean_b, xl,
    target, mu, nu, diag):
    return torch.empty((steps.shape[0], basis.shape[0], 6), device=wo_a.device, dtype=torch.float32)
