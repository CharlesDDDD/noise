"""The replay's projected top values, token-major (run.py project_routed_values_token_major): the per-(token, head)
projection of the centred top values through c_proj's head slice, by programs of TB tokens x BO output columns of one
head (the head's weight slice read once per program instead of once per token)."""
import torch
import triton
import triton.language as tl

# sweep (n = 2048 tokens, 8 heads, 16 keys, 512 columns): TB 4 / BO 256 / 8 warps 0.130 ms (one program per token and
# head: 0.220; TB 8 / BO 128 0.150; TB 2 / BO 512 0.172; centring fused in: 0.273 and not bitwise)
CONFIG = dict(TB=4, BO=256, num_warps=8, num_stages=2)


@triton.jit
def _project_values_blocked_kernel(C, W, Out, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                                   O: tl.constexpr, TB: tl.constexpr, BO: tl.constexpr):
    h, tb, ob = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tl.arange(0, TB * K)  # (token in block, key)
    dd, oo = tl.arange(0, D), ob * BO + tl.arange(0, BO)
    c = tl.load(C + ((h * T + tb * TB) * K + rows[:, None]) * D + dd[None, :])
    w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
    out = tl.dot(c, w)
    t = tb * TB + rows // K
    tl.store(Out + ((t[:, None] * H + h) * K + rows[:, None] % K) * O + oo[None, :], out.to(tl.bfloat16))


def project_values(centered, w_proj_t):
    """centered (H, T, K, D) bf16, w_proj_t (H * D, O) -> (T, H, K, O) bf16."""
    heads, tokens, count, width = centered.shape
    o = w_proj_t.shape[1]
    cfg = dict(CONFIG)
    tb, bo = cfg.pop('TB'), cfg.pop('BO')
    assert tokens % tb == 0 and o % bo == 0
    out = torch.empty((tokens, heads, count, o), device=centered.device, dtype=centered.dtype)
    _project_values_blocked_kernel[(heads, tokens // tb, o // bo)](centered.contiguous(),
      w_proj_t.to(centered.dtype).contiguous(), out, T=tokens, H=heads, K=count, D=width, O=o, TB=tb, BO=bo, **cfg)
    return out
