"""Gluon (Hopper) replay epilogue of the routed attention replay, used by query_factored_gluon: per head pair, the raw
queries' RoPE, norm folding and K-key response weights in registers. w_q's rows are interleaved per head
(a0 b0 a1 b1 ...) so each RoPE pair sits in one thread; the key scores of both heads come from one wgmma against a
block-diagonal key matrix (exact: the off-diagonal zeros add nothing), its A operand the epilogue's dq in the
accumulator layout.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma, warpgroup_mma_wait)

EPS = torch.finfo(torch.float32).eps


@gluon.jit
def _replay_pair(acc, kb, token, pair, row0, pos0, R0, Cos, Sin, Inv, Shift, Keys, Prob, Out,
    P: gl.constexpr, H: gl.constexpr, D: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, EPS_: gl.constexpr,
    num_warps: gl.constexpr, mmak: gl.constexpr):
    """The reference's replay epilogue for heads 2 pair, 2 pair + 1 of the tile (acc: fp32 raw queries, BM x 2 D,
    each head's RoPE halves interleaved)."""
    HD: gl.constexpr = D // 2
    raw = acc.to(gl.bfloat16).to(gl.float32)
    a, b = gl.split(gl.reshape(raw, [BM, D, 2]))
    a = gl.reshape(a, [BM, 2, HD])  # (move, head of the pair, i)
    b = gl.reshape(b, [BM, 2, HD])
    L3: gl.constexpr = a.type.layout
    vl: gl.constexpr = gl.SliceLayout(0, L3)
    hh = gl.arange(0, 2, gl.SliceLayout(1, vl))
    ii = gl.arange(0, HD, gl.SliceLayout(0, vl))
    c = gl.expand_dims(gl.expand_dims(gl.load(Cos + token * HD + ii), 0), 0)
    s = gl.expand_dims(gl.expand_dims(gl.load(Sin + token * HD + ii), 0), 0)
    r0_rows = R0 + ((token * H + 2 * pair + hh) * D)[:, None]
    r0a = gl.load(r0_rows + ii[None, :])[None, :, :]
    r0b = gl.load(r0_rows + HD + ii[None, :])[None, :, :]
    rl: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    pp = pos0 + gl.arange(0, BM, rl)
    g = gl.expand_dims(gl.expand_dims(gl.load(Inv + token * P + pp), 1), 2)
    shift0 = gl.expand_dims(gl.expand_dims(gl.load(Shift + token * P + pp), 1), 2)
    ra = a * c + b * s
    rb = -a * s + b * c
    ra, rb = ra * g + shift0 * r0a, rb * g + shift0 * r0b
    sq = gl.sum(r0a * r0a, 2) + gl.sum(r0b * r0b, 2)
    dots = gl.sum(ra * r0a, 2) + gl.sum(rb * r0b, 2)
    moved = gl.sum(ra * ra, 2) + gl.sum(rb * rb, 2)
    inv0 = gl.rsqrt(sq / D + EPS_)
    inv = gl.rsqrt((sq + 2 * dots + moved) / D + EPS_)
    shift = gl.expand_dims(inv - inv0, 2)
    inv = gl.expand_dims(inv, 2)
    dqa = (ra * inv + r0a * shift).to(gl.bfloat16)
    dqb = (rb * inv + r0b * shift).to(gl.bfloat16)
    dq = gl.reshape(gl.join(gl.reshape(dqa, [BM, D]), gl.reshape(dqb, [BM, D])), [BM, 2 * D])
    dq = gl.convert_layout(dq, gl.DotOperandLayout(operand_index=0, parent=mmak, k_width=2))
    ds = warpgroup_mma(dq, kb.permute((1, 0)), gl.zeros([BM, 2 * K], gl.float32, mmak), use_acc=False,
      is_async=True)
    ds = warpgroup_mma_wait(0, deps=[ds, dq])[0]
    ds = ds.to(gl.bfloat16).to(gl.float32) * (D ** -.5)
    u = gl.reshape(gl.exp(ds) - 1.0, [BM, 2, K])  # (move, head of the pair, key)
    LK: gl.constexpr = u.type.layout
    kl: gl.constexpr = gl.SliceLayout(0, LK)
    kh = gl.arange(0, 2, gl.SliceLayout(1, kl))
    kk = gl.arange(0, K, gl.SliceLayout(0, kl))
    prob = gl.load(Prob + (token * H + 2 * pair + kh)[:, None] * K + kk[None, :])[None, :, :]
    z = gl.expand_dims(gl.sum(prob * u, 2), 2)
    weights = prob * u / (1.0 + z)
    orow = gl.arange(0, BM, gl.SliceLayout(1, gl.SliceLayout(2, LK)))
    optr = Out + gl.expand_dims(gl.expand_dims((row0 + orow).to(gl.int64) * (H * K), 1), 2) \
      + (((2 * pair + kh) * K)[:, None] + kk[None, :])[None, :, :]
    gl.store(optr, weights.to(gl.bfloat16))


def interleave_wq(w_q, heads):
    """w_q's rows per head reordered a0 b0 a1 b1 ... (the RoPE halves interleaved), bf16."""
    c = w_q.shape[0]
    return w_q.to(torch.bfloat16).view(heads, 2, c // heads // 2, -1).transpose(1, 2).reshape(c, -1).contiguous()


