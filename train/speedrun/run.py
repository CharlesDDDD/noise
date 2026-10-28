"""Track 2: an identity-path phase hands off to Cut; held-out training rows then select a merge of weight averages.

1. Identity phase: 300 updates of 8 rows of #44's 1831-update plan (#44: 660), with #29's identity-path credit
   (full-cycle readout credit, finite MLP responses and #29's attention moments). Both hand off at full rate; these
   300 updates lie inside the plan's 368-update ANVIL momentum ramp, which #44's 660 outlast, and end before the
   late skip, tail average and cooldown that #44 reaches. ANVIL updates the matrices and Adam the rest. Listed under
   "identity" in config.json: each attn.c_proj starts as the norm-matched transpose of its c_v, and each wte row
   starts with 1/256 of its energy in the 32 gate channels; for the first 195 updates, wte rows keep that split,
   the value embeddings' vocabulary-common Adam motion is amplified 4x at fixed energy, the VE gates' head-mean
   credit is pooled over the four VE layers, and each VE layer's centered Adam motion is joined by its partner
   layer's (1-5, 3-7), mapped through the partner's c_proj and back through its own and scaled to half the
   receiving layer's norm, then rescaled to the receiving layer's energy.
2. Cut phase: 1523 updates of 8 of the remaining rows. Each batch takes one row from each of 8 difficulty strata:
   rows ranked by the mean add-one-smoothed unigram surprisal of their target tokens, with the counts taken over the
   Cut rows inside the timer. The Cut estimator:
   - #44's measured full-cycle readout probe;
   - MLP roots at blocks 0, 1, 2, 4, 6 and 7, measured in pairs (0-1, 2-4, 6-7): one forward replay moves both roots
     of a pair along signed Hadamard frames with independent random signs per token (the same on every GPU, whose
     rows differ), through routed MLPs (each token's 128 units with the largest |relu^2(h + r) - relu^2(h - r)| times
     output-weight norm, h the unit's pre-activation and r = chain_sigma times its input-weight norm; #44: the 192
     largest relu^2(h) times output-weight norm) and routed attention at block 3 only (16 strong keys shifted by the
     tail centroid; #44: 32 keys at every later block), and scores it with diagonal readout moments (the diagonal of
     #44's per-token readout curvature). Each root's identity-path prediction (the readout credit times the residual
     scalars above it) is subtracted from the replay and added back to that root's credit, so the partner enters only
     through its deviation from its prediction, with a zero-mean random sign. Blocks 3 and 5 take the next block's
     credit times its residual scalar;
   - mixing scalars: each block's mid credit contracted with the scalar's two inputs, a linear step (#44's Cut
     probed them; #29's identity-path credit contracts them the same way), plus #44's path term (the value- and
     gate-path input credits contracted the same way);
   - the embedding's credit: every block's mid credit summed along its x0 skip (#44: block 0's only), one finite step
     back through the norm;
   - #44's finite query probes and its value and gate paths (exact linear value credit; one finite norm or gate
     step whose credit feeds only linear consumers). The key credit moves each raw key coordinate by a finite step
     through the key norm and RoPE in closed form and scores the softmax's response with its degree-2 moment expansion
     (the moment model of #29's and #44's identity-phase key credit; #44's Cut used the exact single-key
     response).
   ANVIL weight decay is zero after the first 275/1043 of the updates.
3. Weight averages kept over the last 40% of the Cut updates are merged part by part, chosen on 64 held-out rows
   (Nystrom-sketched input-covariance bases; candidates screened on 8 of the rows, finalists scored on all 64).

config.json lists the changes against #44 (qlabs-eng/noise@3343973b) under "additions".

Run on one 8xH100 node, seed 0; measured results are in train/speedrun/log/.
As in #44, train.pt is read (with its pinned-hash check) and split by seeded permutations, and the identity batches
are put on the GPUs, before the timer; apart from the organizers' pinned-hash check, the difficulty order is the only
computation on the rows' contents before the first update, and it is timed. The random-token warmup (the identity
graph, a throwaway Cut, the Cut-estimate graph capture and the merge scorer, whose inputs the held-out rows overwrite
before scoring) is discarded or reset before timed training starts from seed 0; final validation is untimed.
metrics.jsonl writes the Cut records every 16 updates, each with the time of its flush.

    uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from functools import cache
import ast
from dataclasses import asdict, dataclass, field
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import time
from typing import Optional
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, has_ve, norm, window_mask
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visit_gluon  # noqa: E402  (next to this file)
import visit_gluon_root  # noqa: E402
import query_replay  # noqa: E402
import visit_chain  # noqa: E402
import visit_chain3  # noqa: E402
import query_factored  # noqa: E402
import query_factored_gluon  # noqa: E402
import window_topk  # noqa: E402
import project_values  # noqa: E402
import moment_kernels  # noqa: E402
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate
BEGIN_MARKER = 'NOISE_TRAINING_BEGIN'
END_MARKER = 'NOISE_TRAINING_END'


def _boundary(marker: str) -> None:
    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if int(os.environ['RANK']) == 0:
        print(marker, flush=True)


def begin_training() -> None:
    """Call after random-token warmup and rebuilding every training state from seed 0."""
    _boundary(BEGIN_MARKER)


def end_training() -> None:
    """Call after the last update and any weight averaging or training-data merge."""
    _boundary(END_MARKER)


SEED = 0
IDENTITY_UPDATES, IDENTITY_SCHEDULE = 355, 1831  # hand off to Cut after 355 of 1831 planned identity updates
EPS = torch.finfo(torch.float32).eps
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


def model_config():
    return GPTConfig(sequence_len=2048, vocab_size=4096, n_layer=8, n_head=8,
      n_kv_head=8, n_embd=512, window_pattern='SSSL')


@dataclass
class Settings:
    model: GPTConfig = field(default_factory=model_config)
    late_cut_decay_multiplier: float = 0.0  # ANVIL weight-decay factor after the first 275/1043 of Cut updates
    cut_batch: int = 8
    lr: float = 0.0028
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.5
    cut_warmup: int = 0  # Cut updates over which both optimizers' rates ramp up linearly from the hand-off
    final: float = 0.5
    anvil_lr: float = 0.036
    anvil_wd: float = 2.25
    fast_weight: float = 1.0
    polar_margin: float = 1.1
    telemetry_batch: int = 16
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    fused_chain: bool = True  # chain replays' MLP visits in one kernel each (_fused_visit_kernel)
    compiled_glue: bool = True  # key_context's per-head tensors from one compiled kernel
    batched_keys: bool = True  # key contexts, key credits and query moments batched over layers (per window size)
    batched_routes: bool = True
    route_k_late: int = 64  # with batched_routes: this many routes from block route_split on (0: route_k everywhere)
    route_split: int = 2
    route_schedule: tuple = ()  # with batched_routes: routes per block (powers of two, at most route_k)  # every block's MLP routes from one top-k (exact)
    probe_moments: bool = True  # readout probe from the second-order logit expansion (exact quadratic term)
    query_moments: bool = True  # query credit per raw coordinate from the degree-2 moment model (no probes)
    query_moment_sigma: float = 0.003
    chain_sigma: float = 0.01
    chain_depth_roots: tuple = (4, 6)
    joint_root_groups: tuple = ((4,), (6,))
    route_k: int = 128
    chain_fraction: float = 1.0  # tokens replayed per row (a random subset; unbiased: their deviation scaled up)
    chain_moves: int = 512  # frame directions replayed per root (a random subset of the signed Hadamard's rows)
    chain_tokens: int = 2048
    query_replay_long_layers: tuple = (5,)
    gate_path_sigma: float = 0.01
    query_keys: int = 16
    replay_groups: Optional[tuple] = None  # heads of the root groups that replay attention (None: every group)
    key_sigma: float = 0.003
    merge_screen_rows: int = 8
    merge_screen_global: int = 16
    merge_screen_coordinate: int = 2
    merge_keys: int = 4
    merge_sweeps: int = 0
    emb_scale: float = 750.0
    lambda_scale: float = 15.0
    def __post_init__(self):
        flattened_groups = tuple(site for group in self.joint_root_groups for site in group)
        if (tuple(sorted(flattened_groups)) != self.chain_depth_roots
        or any(tuple(sorted(group)) != group or not group for group in self.joint_root_groups)):
            raise ValueError('joint_root_groups must split chain_depth_roots into ascending groups')
        last = self.model.n_layer - 1
        if any(p < last and (p + 1 in group or p + 1 in self.query_replay_long_layers)
               for group in self.joint_root_groups for p in group[1:]):
            raise ValueError('A partner root must be the last block or be followed by a block '
                             'that is neither replayed with attention nor another partner')


@torch.compile
def coordinate_differences(raw, y, sigma):
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    if (d, device, dtype) not in HADAMARDS:
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor.div_(torch.distributed.get_world_size())
    return tensor


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


def query_probe_attention(moved, keys, values, window, count, batch):
    _, length, heads, width = keys.shape
    repeat = lambda z: z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*batch, length, heads, width)
    return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores, directions):
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    """Collapse the full Hadamard central-difference cycle for the separable hidden objective.
    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's
    scalar finite difference. Keep that finite step, including ReLU crossings;
    contract in FP32 instead of rounding every direction's projected response to BF16.
    u and out_weight contain all units.
    """
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope


def norm_inverses(x, delta):
    sq = x.square().sum(-1, keepdim=True)
    if delta.dtype == x.dtype:
        dots = torch.einsum('md,mpd->mp', x, delta)
    else:
        dots = (x.unsqueeze(1) * delta.float()).sum(-1)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    return inv0, torch.rsqrt((sq + 2 * dots + delta.float().square().sum(-1)) / x.shape[-1] + EPS)


def norm_difference(x, delta):
    inv0, inv = norm_inverses(x, delta)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
      / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch.compile(dynamic=False)
def gate_input_responses(xs, weights, kappas, rows, sigma: float):
    """prepared_gate_input_response for several layers in one compiled graph (lists of per-layer tensors)."""
    out = []
    for x, weight, kappa, r in zip(xs, weights, kappas, rows):
        projected = F.linear(r[:, :weight.shape[1]], weight)
        row_square = r.square().mean(-1)
        steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        square = x.square().mean(-1, keepdim=True)
        cross = (x @ r.T) / x.shape[-1]
        base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
        positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * row_square.unsqueeze(0) + EPS)
        negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * row_square.unsqueeze(0) + EPS)
        zp = (base + steps.unsqueeze(-1) * projected.unsqueeze(0)) * positive.unsqueeze(-1)
        zm = (base - steps.unsqueeze(-1) * projected.unsqueeze(0)) * negative.unsqueeze(-1)
        response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
        out.append(response @ r / len(r))
    return out


STACKED_GATES = True  # fused_descending_key_values' value-gate input credits with the layers stacked (one graph, bmm)


@torch.compile(dynamic=False)
def stacked_gate_input_responses(raws, ves, blocks, weights, rows, sigma: float):
    """gate_input_responses with the layers stacked along a batch axis (each GEMM once, as a bmm) and the kappas
    (raws (tokens, width) and ves the layers' value embeddings) and casts inside the graph."""
    x = torch.stack(blocks).float().flatten(1, 2)
    weight = torch.stack(weights).float()
    r = torch.stack(rows)
    heads, k = weight.shape[1:]
    raw = torch.stack(raws)
    kappa = (raw.view(*raw.shape[:2], heads, -1) * torch.stack(ves).float().view(*raw.shape[:2], heads, -1)).sum(-1)
    projected = r[..., :k] @ weight.transpose(1, 2)
    row_square = r.square().mean(-1)
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ r.transpose(1, 2)) / x.shape[-1]
    base = (x[..., :k] @ weight.transpose(1, 2)).unsqueeze(2)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * row_square.unsqueeze(1) + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * row_square.unsqueeze(1) + EPS)
    zp = (base + steps.unsqueeze(-1) * projected.unsqueeze(1)) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * projected.unsqueeze(1)) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(2)).sum(-1) / steps
    return (response @ r / r.shape[1]).unbind(0)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ rows.T) / x.shape[-1]
    direction_square = row_square.unsqueeze(0)
    base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
    direction = projected_rows.unsqueeze(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * direction_square + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * direction_square + EPS)
    zp = (base + steps.unsqueeze(-1) * direction) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * direction) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
    return decode_local_scores(response, rows) / len(rows)


def center_routed_values(values, probabilities, rest):
    origin = (probabilities.unsqueeze(-1) * values.float()).sum(-2) + rest
    return (values.float() - origin.unsqueeze(-2)).to(values.dtype)


TOKEN_MAJOR_VALUES = True  # the replay's projected top values written token-major by one kernel (no transpose copy)


@triton.jit
def _project_values_kernel(C, W, Out, T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
                           O: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    kk, dd, oo = tl.arange(0, K), tl.arange(0, D), tl.arange(0, O)
    c = tl.load(C + ((h * T + t) * K + kk[:, None]) * D + dd[None, :])
    w = tl.load(W + (h * D + dd[:, None]) * O + oo[None, :])
    out = tl.dot(c, w)
    tl.store(Out + ((t * H + h) * K + kk[:, None]) * O + oo[None, :], out.to(tl.bfloat16))


def project_routed_values_token_major(values, probabilities, rest, w_proj_t):
    """preproject_routed_values(...).transpose(0, 1).contiguous() without the transposed copy."""
    centered = center_routed_values_compiled(values, probabilities, rest).contiguous()
    if BLOCKED_PROJECTION:  # 4 tokens x 256 columns per program: each head's weight slice read once (bitwise)
        return project_values.project_values(centered, w_proj_t)
    heads, tokens, count, width = centered.shape
    output_width = w_proj_t.shape[1]
    out = torch.empty((tokens, heads, count, output_width), device=values.device, dtype=centered.dtype)
    _project_values_kernel[(tokens, heads)](centered, w_proj_t.to(centered.dtype).contiguous(), out, T=tokens, H=heads,
                                            K=count, D=width, O=output_width, num_warps=4)
    return out


center_routed_values_compiled = torch.compile(lambda v, p, r: center_routed_values(v, p, r), dynamic=False)


@torch.compile(dynamic=False)
def preproject_routed_values(values, probabilities, rest, w_proj_t):
    centered = center_routed_values(values, probabilities, rest)
    heads, tokens, count, width = centered.shape
    output_width = w_proj_t.shape[1]
    projected = torch.bmm(centered.reshape(heads, tokens * count, width),
      w_proj_t.to(centered.dtype).reshape(heads, width, output_width))
    return projected.view(heads, tokens, count, output_width)


@triton.jit
def _fused_query_replay(Raw, R0, Cos, Sin, Inv, Shift, KeysPtr, Prob, Out,
    P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
    BP: tl.constexpr, BK: tl.constexpr, EPS_: tl.constexpr):
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    pp = tl.program_id(0) * BP + tl.arange(0, BP)
    hh = tl.arange(0, D // 2)
    dd = tl.arange(0, D)
    base = ((token * P + pp[:, None]) * H + head) * D
    a = tl.load(Raw + base + hh[None, :], pp[:, None] < P, 0).to(tl.float32)
    b = tl.load(Raw + base + D // 2 + hh[None, :], pp[:, None] < P, 0).to(tl.float32)
    c = tl.load(Cos + token * (D // 2) + hh)
    s = tl.load(Sin + token * (D // 2) + hh)
    r0a = tl.load(R0 + (token * H + head) * D + hh)
    r0b = tl.load(R0 + (token * H + head) * D + D // 2 + hh)
    ra = a * c[None, :] + b * s[None, :]
    rb = -a * s[None, :] + b * c[None, :]
    g = tl.load(Inv + token * P + pp, pp < P, 0)[:, None]
    h = tl.load(Shift + token * P + pp, pp < P, 0)[:, None]
    ra, rb = ra * g + h * r0a[None, :], rb * g + h * r0b[None, :]
    sq = tl.sum(r0a * r0a, 0) + tl.sum(r0b * r0b, 0)
    dots = tl.sum(ra * r0a[None, :], 1) + tl.sum(rb * r0b[None, :], 1)
    moved = tl.sum(ra * ra, 1) + tl.sum(rb * rb, 1)
    inv0 = tl.math.rsqrt(sq / D + EPS_)
    inv = tl.math.rsqrt((sq + 2 * dots + moved) / D + EPS_)
    shift = (inv - inv0)[:, None]
    dqa = (ra * inv[:, None] + r0a[None, :] * shift).to(tl.bfloat16)
    dqb = (rb * inv[:, None] + r0b[None, :] * shift).to(tl.bfloat16)
    dq = tl.reshape(tl.trans(tl.join(dqa, dqb), (0, 2, 1)), (BP, D))
    rr = tl.arange(0, BK)
    offsets = ((token * H + head) * K + rr[:, None]) * D + dd[None, :]
    keys = tl.load(KeysPtr + offsets, rr[:, None] < K, 0)
    prob = tl.load(Prob + (token * H + head) * K + rr, rr < K, 0)
    ds = tl.dot(dq, tl.trans(keys)).to(tl.bfloat16).to(tl.float32) * (D ** -.5)
    u = tl.exp(ds) - 1.0
    z = tl.sum(prob[None, :] * u, 1)
    weights = prob[None, :] * u / (1.0 + z[:, None])
    tl.store(Out + ((token * P + pp[:, None]) * H + head) * K + rr[None, :],
      weights.to(tl.bfloat16), (pp[:, None] < P) & (rr[None, :] < K))


@torch.library.custom_op('noise_record::fused_query_weights', mutates_args=())
def fused_query_weights(raw: torch.Tensor, r0: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, inv: torch.Tensor, shift: torch.Tensor,
    keys: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    m, probes = raw.shape[:2]
    heads, width = r0.shape[1:]
    count = keys.shape[2]
    out = torch.empty((m, probes, heads * count), device=raw.device, dtype=torch.bfloat16)
    _fused_query_replay[(triton.cdiv(probes, 32), m * heads)](
      raw.contiguous(), r0.float().contiguous(), cos.float().contiguous(),
      sin.float().contiguous(), inv.float().contiguous(), shift.float().contiguous(),
      keys.contiguous(), prob.contiguous(), out,
      probes, heads, width, count, 32, max(16, triton.next_power_of_2(count)), EPS, num_warps=4)
    return out


@fused_query_weights.register_fake
def _(raw, r0, cos, sin, inv, shift, keys, prob):
    return torch.empty((raw.shape[0], raw.shape[1], keys.shape[1] * keys.shape[2]),
      device=raw.device, dtype=torch.bfloat16)


def fused_projected_response(raw, r0, cos, sin, inv, shift, keys, projected_values, prob, add_to=None,
    raw_bf16=False):
    weights = fused_query_weights(raw, r0, cos, sin, inv, shift, keys, prob)
    if raw_bf16 == 'parts':  # the response's two factors, multiplied inside the next visit
        return weights, projected_values.flatten(1, 2)
    if raw_bf16:
        return torch.bmm(weights, projected_values.flatten(1, 2))
    if add_to is not None:  # delta + response in one GEMM epilogue, bf16 out
        return torch.baddbmm(add_to, weights, projected_values.flatten(1, 2))
    return torch.bmm(weights, projected_values.flatten(1, 2)).float()


def query_change_routed(x, delta, pos, w_q, top_keys, top_values, top_p, cos, sin, query_inv, add_to=None,
    raw_bf16=False, factored=None):
    m, probes, _ = delta.shape
    _, heads, _, width = top_keys.shape
    c, s = cos[pos][:, None], sin[pos][:, None]
    q0 = (norm(x) @ w_q.T).view(m, heads, width)
    r0 = torch.cat([q0[..., :width // 2] * c + q0[..., width // 2:] * s,
      q0[..., :width // 2] * -s + q0[..., width // 2:] * c], -1)
    inv0, inv = query_inv
    if factored is not None and raw_bf16 == 'parts':  # delta = the root output: q change from its routed factors
        change, steps, scale, wo, routes, basis = factored
        if GLUON_FACTORED:  # A_t transposed and w_q's rows interleaved per head: its own factors op
            at, f = query_factored_gluon.query_factors_t(wo, routes, basis, w_q)
            weights = query_factored_gluon.fused_query_replay_factored_gluon(change, steps, scale, at, f, r0,
                cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_p)
        else:
            a, f = query_factored.query_factors(wo, routes, basis, w_q)
            weights = query_factored.fused_query_replay_factored(change, steps, scale, a, f, r0, cos[pos], sin[pos],
                                                                 inv, inv / inv0 - 1, top_keys, top_p)
        return weights, top_values.flatten(1, 2)
    if FUSED_QUERY_REPLAY and raw_bf16 == 'parts':  # q projection and replay in one kernel, only weights stored
        weights = query_replay.fused_query_replay(delta, w_q, r0, cos[pos], sin[pos], inv, inv / inv0 - 1,
                                                  top_keys, top_p)
        return weights, top_values.flatten(1, 2)
    if QUERY_FP8:  # rowwise-scaled e4m3 operands, bf16 out
        flat = delta.reshape(-1, delta.shape[-1])
        sa = (flat.abs().amax(-1, keepdim=True).float() / 448).clamp_min(1e-12)
        sb = (w_q.abs().amax(-1, keepdim=True).float() / 448).clamp_min(1e-12)
        a8 = (flat.float() / sa).to(torch.float8_e4m3fn)
        b8 = (w_q.float() / sb).to(torch.float8_e4m3fn)
        raw = torch._scaled_mm(a8, b8.T, scale_a=sa, scale_b=sb.T, out_dtype=torch.bfloat16).view(
            m, probes, heads, width)
    else:
        raw = (delta @ w_q.T.to(torch.bfloat16)).view(m, probes, heads, width)
    return fused_projected_response(raw, r0, cos[pos], sin[pos], inv, inv / inv0 - 1, top_keys, top_values, top_p,
      add_to, raw_bf16)


@torch.compile(dynamic=False)
def key_context_tensors(q, k, v, ve, z, target, cos, sin, heads: int):
    """key_context's per-head tensors (heads, tokens, head_dim), contiguous: RoPE'd unit queries and keys, raw keys,
    values with the gated value embedding, and the attention output's credit."""
    t = q.shape[0]
    hd = q.shape[1] // heads
    split = lambda z_: z_.float().view(t, heads, hd).transpose(0, 1)
    rot = lambda z_: torch.cat([z_[..., :hd // 2] * cos + z_[..., hd // 2:] * sin,
                                -z_[..., :hd // 2] * sin + z_[..., hd // 2:] * cos], -1)
    unit = lambda z_: z_ * torch.rsqrt(z_.square().mean(-1, keepdim=True) + EPS)
    q_raw, k_raw, values = split(q), split(k), split(v)
    if ve is not None:
        values = values + 2 * torch.sigmoid(z.float().view(t, heads)).T.unsqueeze(-1) * split(ve)
    return (unit(rot(q_raw)).contiguous(), unit(rot(k_raw)).contiguous(), k_raw.contiguous(), values.contiguous(),
            split(target).contiguous())


@torch.compile(dynamic=False)
def key_context_group(qs, ks, vs, ves, gates, targets, cos, sin, heads: int):
    """key_context_tensors for several layers, stacked along the head axis in the same compiled graph."""
    t = qs[0].shape[0]
    hd = qs[0].shape[1] // heads
    split = lambda z_: z_.float().view(t, heads, hd).transpose(0, 1)
    rot = lambda z_: torch.cat([z_[..., :hd // 2] * cos + z_[..., hd // 2:] * sin,
                                -z_[..., :hd // 2] * sin + z_[..., hd // 2:] * cos], -1)
    unit = lambda z_: z_ * torch.rsqrt(z_.square().mean(-1, keepdim=True) + EPS)
    out = [], [], [], [], []
    for q, k, v, ve, z, target in zip(qs, ks, vs, ves, gates, targets):
        k_raw, values = split(k), split(v)
        if ve is not None:
            values = values + 2 * torch.sigmoid(z.float().view(t, heads)).T.unsqueeze(-1) * split(ve)
        for acc, part in zip(out, (unit(rot(split(q))), unit(rot(k_raw)), k_raw, values, split(target))):
            acc.append(part)
    return tuple(torch.cat(acc, 0) for acc in out)


EARLY_MOMENT_INPUTS = True  # with FUSED_GLUE: query_moment_inputs' outputs formed in key_context_group's graph


@torch.compile(dynamic=False)
def key_context_group_moments(qs, ks, vs, ves, gates, targets, cos, sin, heads: int, sigma: float):
    """key_context_group, and query_moment_inputs' outputs for the stacked layers, in one compiled graph (the bf16
    copies written while the fp32 tensors are formed, not reread)."""
    qp, kp, k_raw, values, hub = key_context_group(qs, ks, vs, ves, gates, targets, cos, sin, heads)
    q_out = torch.cat(qs, 1)
    return (qp, kp, k_raw, values, hub), query_moment_inputs(q_out, qp, kp, values, hub, sigma)


@torch.compile(dynamic=False)
def query_moment_inputs(q_out, qp, kp, values, hub, sigma: float):
    heads, t, hd = qp.shape
    q_raw = q_out.float().view(t, heads, hd).transpose(0, 1)
    sq = q_raw.square().sum(-1)
    steps = sigma * (sq / hd).sqrt().clamp(min=1e-6)
    b16 = lambda z_: z_.to(torch.bfloat16).contiguous()
    return b16(qp), b16(kp), b16(values), b16(hub), q_raw.contiguous().unsqueeze(0), sq.contiguous(), steps.contiguous()


@torch.compile(dynamic=False)
def response_scores(hpre, radius, out_norm):
    plus = F.relu(hpre + radius).square()
    minus = F.relu(hpre - radius).square()
    return (plus - minus).abs() * out_norm


@torch.compile(dynamic=False)
def norm_input_credit(x, g, sigma, H):
    width = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    step = sigma * (sq / width).sqrt().clamp(min=1e-6)
    xh, gh, gx = x @ H.T, g @ H.T, (g * x).sum(-1, keepdim=True)
    inv = torch.rsqrt(sq / width + EPS)
    inv_up = torch.rsqrt((sq + 2 * step * xh + step.square() * width) / width + EPS)
    inv_down = torch.rsqrt((sq - 2 * step * xh + step.square() * width) / width + EPS)
    up = step * gh * inv_up + gx * (inv_up - inv)
    down = -step * gh * inv_down + gx * (inv_down - inv)
    return ((up - down) / (2 * step)) @ H / width


@torch.no_grad()
@torch.compile(dynamic=False)
def readout_curvature_diagonal(weight, c2):
    """Diagonal of each token's readout curvature W^T diag(c2) W; the readout moments keep only this diagonal."""
    return c2 @ weight.square()


@torch.no_grad()
def head_moments(xL, base, weight):
    width = xL.shape[-1]
    inv0 = torch.rsqrt(xL.square().sum(-1, keepdim=True) / width + EPS)
    t = torch.tanh(base * inv0 / 15)
    f, f1 = 15 * t, 1 - t.square()
    f2 = -(2 / 15) * t * f1
    prob = torch.softmax(f, -1)
    c1, c2 = prob * f1, prob * (f2 + f1.square()) / 2
    b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
    mu1, nu = c1 @ weight, (c2 * base) @ weight
    second = readout_curvature_diagonal(weight, c2)
    return b1, b2, mu1, nu, second


@triton.jit
def _route_gate_kernel(Delta, Weight, Routes, Clean, Inv0, Inv, Out, Steps, Projection,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    rr = tl.program_id(1) * BN + tl.arange(0, BN)
    chosen = tl.load(Routes + token * R + rr, rr < R, 0)
    acc = tl.zeros((BM, BN), tl.float32)
    if ROOT:
        q = tl.load(Projection + chosen[:, None] * P + pp[None, :],
        (rr[:, None] < R) & (pp[None, :] < P), 0)
        step16 = tl.load(Steps + token).to(tl.bfloat16).to(tl.float32)
        acc = tl.trans(q) * step16
    else:
        for k0 in range(0, D, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(Delta + token * S0 + pp[:, None] * S1 + kk[None, :] * S2,
          (pp[:, None] < P) & (kk[None, :] < D), 0).to(tl.bfloat16)
            b = tl.load(Weight + chosen[:, None] * D + kk[None, :],
          (rr[:, None] < R) & (kk[None, :] < D), 0).to(tl.bfloat16)
            acc += tl.dot(a, tl.trans(b))
    moved = acc.to(tl.bfloat16).to(tl.float32)
    inv = tl.load(Inv + token * P + pp, pp < P, 0)
    inv0 = tl.load(Inv0 + token)
    h = tl.load(Clean + token * R + rr, rr < R, 0).to(tl.float32)
    dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    changed = ((up - down) * (up + down)).to(tl.bfloat16)
    tl.store(Out + (token * P + pp[:, None]) * R + rr[None, :], changed,
      (pp[:, None] < P) & (rr[None, :] < R))


@triton.jit
def _route_output_parts(Change, Weight, Routes, Delta, Steps, Basis,
    token, pp, dd, P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, INJECT: tl.constexpr):
    acc = tl.zeros((BM, BN), tl.float32)
    for r0 in range(0, R, BK):
        rr = r0 + tl.arange(0, BK)
        chosen = tl.load(Routes + token * R + rr, rr < R, 0)
        a = tl.load(Change + (token * P + pp[:, None]) * R + rr[None, :],
        (pp[:, None] < P) & (rr[None, :] < R), 0)
        b = tl.load(Weight + chosen[:, None] * D + dd[None, :],
        (rr[:, None] < R) & (dd[None, :] < D), 0).to(tl.bfloat16)
        acc += tl.dot(a, b)
    moved = acc.to(tl.bfloat16).to(tl.float32)
    if ROOT:
        step = tl.load(Steps + token)
        basis = tl.load(Basis + pp[:, None] * D + dd[None, :],
        (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
        old = step * basis
    else:
        old = tl.load(Delta + token * S0 + pp[:, None] * S1 + dd[None, :] * S2,
        (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
        if INJECT:
            step = tl.load(Steps + token)
            frame = tl.load(Basis + pp[:, None] * D + dd[None, :],
          (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
            old = old + step * frame
    return old, moved


@triton.jit
def _route_output_kernel(Change, Weight, Routes, Delta, Out, Steps, Basis, Scale,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, ROOT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, ROOT, BM, BN, BK, False)
    result = (old + moved).to(tl.bfloat16)
    result = (result.to(tl.float32) * tl.load(Scale)).to(tl.bfloat16)
    tl.store(Out + (token * P + pp[:, None]) * D + dd[None, :], result,
      (pp[:, None] < P) & (dd[None, :] < D))


@torch.library.custom_op('noise_record::route_mlp', mutates_args=())
def route_mlp(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor,
    inv0: torch.Tensor, inv: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean = routes.contiguous(), clean.contiguous()
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=delta.device, dtype=torch.bfloat16)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      delta, fc, routes, clean, inv0.contiguous(), inv.contiguous(), change, delta, delta,
      p, d, r, *delta.stride(), False, BM=64, BN=min(128, r), BK=64, num_warps=8, num_stages=3)
    _route_output_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, delta, out, delta, delta, out_scale, p, d, r, *delta.stride(),
      False, BM=64, BN=128, BK=r, num_warps=8, num_stages=2)
    return out


@route_mlp.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, out_scale):
    return torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16)


def mlp_statistics_storage(n, p, ns, nb, device):
    # Stored as (token, statistic, tile, probe) and viewed as (token, probe, statistic, tile).
    return torch.empty((n, ns, nb, p), device=device, dtype=torch.float32).permute(0, 3, 1, 2)


@triton.jit
def _mlp_output_stats_kernel(Change, Weight, Routes, Delta, Out, Stats, X, Lambda,
    Target, Mu, Nu, Diag, Steps, Basis,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    MODE: tl.constexpr, NS: tl.constexpr, NB: tl.constexpr, INJECT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, False, BM, BN, BK, INJECT)
    value = (old + moved).to(tl.bfloat16).to(tl.float32)
    if MODE == 1:
        value = value * tl.load(Lambda).to(tl.float32)
        tl.store(Out + (token * P + pp[:, None]) * D + dd[None, :], value.to(tl.bfloat16),
          (pp[:, None] < P) & (dd[None, :] < D))
    xx = tl.load(X + token * D + dd, dd < D, 0).to(tl.float32)
    pos = (token * NS * NB + tl.program_id(1)) * P + pp
    ss: tl.constexpr = NB * P
    tl.store(Stats + pos, tl.sum(value * xx[None, :], 1), pp < P)
    tl.store(Stats + pos + ss, tl.sum(value * value, 1), pp < P)
    if MODE == 2:
        target = tl.load(Target + token * D + dd, dd < D, 0).to(tl.float32)
        mu = tl.load(Mu + token * D + dd, dd < D, 0).to(tl.float32)
        nu = tl.load(Nu + token * D + dd, dd < D, 0).to(tl.float32)
        diag = tl.load(Diag + token * D + dd, dd < D, 0).to(tl.float32)
        tl.store(Stats + pos + 2 * ss, tl.sum(value * target[None, :], 1), pp < P)
        tl.store(Stats + pos + 3 * ss, tl.sum(value * mu[None, :], 1), pp < P)
        tl.store(Stats + pos + 4 * ss, tl.sum(value * nu[None, :], 1), pp < P)
        tl.store(Stats + pos + 5 * ss, tl.sum(value * value * diag[None, :], 1), pp < P)


def launch_mlp_output_stats(out, delta, fc, wo, routes, clean, inv0, inv, x, scale, target, mu, nu, diag, mode,
    steps, frame, packed):
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean = routes.contiguous(), clean.contiguous()
    ns, nb = (2 if mode == 1 else 6), triton.cdiv(d, 64 if mode == 1 else 128)
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    stats = mlp_statistics_storage(n, p, ns, nb, delta.device)
    tile = min(64, r)
    inject = steps is not None
    steps = delta if steps is None else steps.contiguous()
    frame = delta if frame is None else frame.contiguous()
    gate_input = delta if packed is None else packed
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      gate_input, fc, routes, clean, inv0.contiguous(), inv.contiguous(), change, delta, delta,
      p, d, r, *gate_input.stride(), False, BM=64, BN=min(128, r), BK=64, num_warps=8, num_stages=3)
    _mlp_output_stats_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, delta, out, stats, x.contiguous(), scale, target.contiguous(),
      mu.contiguous(), nu.contiguous(), diag.contiguous(), steps, frame, p, d, r, *delta.stride(),
      mode, ns, nb, inject, BM=64, BN=64 if mode == 1 else 128, BK=r if mode == 1 else tile,
      num_warps=4, num_stages=2 if mode == 1 else 3, enable_fp_fusion=False)
    return stats


@torch.library.custom_op('noise_record::mlp_output_stats', mutates_args=())
def mlp_output_stats(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor,
    inv: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor], packed: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Routed MLP output scaled by the next block's residual scalar, with the partials of the next block's norm."""
    out = torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16)
    stats = launch_mlp_output_stats(out, delta, fc, wo, routes, clean, inv0, inv, x, scale, x, x, x, x, 1,
      steps, frame, packed)
    return out, stats


@mlp_output_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, x, scale, steps, frame, packed):
    n, p, d = delta.shape
    return (torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), delta.device))


@torch.library.custom_op('noise_record::mlp_readout_stats', mutates_args=())
def mlp_readout_stats(delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor,
    x: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor, diag: torch.Tensor,
    steps: Optional[torch.Tensor], frame: Optional[torch.Tensor], packed: Optional[torch.Tensor]) -> torch.Tensor:
    """Last block's routed MLP: partials of the readout moments only (the output itself is not written)."""
    return launch_mlp_output_stats(delta, delta, fc, wo, routes, clean, inv0, inv, x, x, target, mu, nu, diag, 2,
      steps, frame, packed)


@mlp_readout_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, x, target, mu, nu, diag, steps, frame, packed):
    n, p, d = delta.shape
    return mlp_statistics_storage(n, p, 6, triton.cdiv(d, 128), delta.device)


def root_norm_inverses(x, steps, basis):
    d = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    dots = steps * (x @ basis.T)
    moved_sq = steps.square() * basis.square().sum(-1).unsqueeze(0)
    return torch.rsqrt(sq / d + EPS), torch.rsqrt((sq + 2 * dots + moved_sq) / d + EPS)


@torch.library.custom_op('noise_record::route_mlp_root', mutates_args=())
def route_mlp_root(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor,
    wo: torch.Tensor, routes: torch.Tensor, clean: torch.Tensor,
    inv0: torch.Tensor, inv: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    n, d, p = steps.shape[0], wo.shape[1], basis.shape[0]
    r = routes.shape[1]
    steps, basis = steps.contiguous(), basis.contiguous()
    routes, clean = routes.contiguous(), clean.contiguous()
    change = torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      steps, steps, routes, clean, inv0.contiguous(), inv.contiguous(), change, steps, projection,
      p, d, r, 0, 0, 0, True, BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
    _route_output_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)](
      change, wo, routes, steps, out, steps, basis, out_scale, p, d, r, 0, 0, 0,
      True, BM=64, BN=64, BK=64, num_warps=4, num_stages=3)
    return out


@route_mlp_root.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, out_scale):
    return torch.empty((steps.shape[0], basis.shape[0], wo.shape[1]), device=wo.device, dtype=torch.bfloat16)


@triton.jit
def _injected_norm_kernel(X, Delta, Steps, Basis, Inv0, Inv, Mixed,
    P: tl.constexpr, D: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
    BM: tl.constexpr, BD: tl.constexpr, EPSILON: tl.constexpr):
    token = tl.program_id(0)
    pp = tl.program_id(1) * BM + tl.arange(0, BM)
    dd = tl.arange(0, BD)
    x = tl.load(X + token * D + dd, dd < D, 0).to(tl.float32)
    delta = tl.load(Delta + token * S0 + pp[:, None] * S1 + dd[None, :] * S2,
      (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
    frame = tl.load(Basis + pp[:, None] * D + dd[None, :],
      (pp[:, None] < P) & (dd[None, :] < D), 0).to(tl.float32)
    step = tl.load(Steps + token)
    moved = delta + step * frame
    tl.store(Mixed + (token * P + pp[:, None]) * D + dd[None, :], moved.to(tl.bfloat16),
      (pp[:, None] < P) & (dd[None, :] < D))
    square = tl.sum(x * x, 0)
    dot = tl.sum(moved * x[None, :], 1)
    move_square = tl.sum(moved * moved, 1)
    inv = tl.rsqrt((square + 2 * dot + move_square) / D + EPSILON)
    tl.store(Inv + token * P + pp, inv, pp < P)
    if tl.program_id(1) == 0:
        tl.store(Inv0 + token, tl.rsqrt(square / D + EPSILON))


@torch.library.custom_op('noise_record::injected_norm_packed', mutates_args=())
def injected_norm_packed(x: torch.Tensor, delta: torch.Tensor, steps: torch.Tensor,
    basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, p, d = delta.shape
    x, steps, basis = x.contiguous(), steps.contiguous(), basis.contiguous()
    inv0 = torch.empty((n, 1), device=x.device, dtype=torch.float32)
    inv = torch.empty((n, p), device=x.device, dtype=torch.float32)
    mixed = torch.empty((n, p, d), device=x.device, dtype=torch.bfloat16)
    _injected_norm_kernel[(n, triton.cdiv(p, 8))](
      x, delta, steps, basis, inv0, inv, mixed, p, d, *delta.stride(),
      BM=8, BD=triton.next_power_of_2(d), EPSILON=EPS, num_warps=4, enable_fp_fusion=False)
    return inv0, inv, mixed


@injected_norm_packed.register_fake
def _(x, delta, steps, basis):
    return (torch.empty((delta.shape[0], 1), device=x.device, dtype=torch.float32),
      torch.empty(delta.shape[:2], device=x.device, dtype=torch.float32),
      torch.empty(delta.shape, device=x.device, dtype=torch.bfloat16))


@triton.jit
def _fused_visit_kernel(Delta, Steps, Frame, X, Fc, Wo, Routes, Clean, Scale, Out,
    Target, Mu, Nu, Diag, XL, Stats, Add,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr, INJECT: tl.constexpr, MODE: tl.constexpr,
    ADD: tl.constexpr, HADAMARD: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr, EPSILON: tl.constexpr):
    """One routed MLP visit of a chain replay for BM moves of one token, the moved-stream change kept on chip: the
    gate GEMM over the routed c_fc rows accumulates the norm sums of the moved block input as it streams delta (plus
    an injected partner move), the relu^2 change stays in registers for the output GEMM. MODE 1 writes the change
    of the next block's input scaled by its residual scalar; MODE 2 (the last block) writes the readout-moment
    partial sums of the moved final stream instead."""
    token = tl.program_id(1)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    live = pp < P
    rr = tl.arange(0, R)
    chosen = tl.load(Routes + token * R + rr)
    acc = tl.zeros((BM, R), tl.float32)
    sq_x = tl.zeros((1,), tl.float32)
    dots = tl.zeros((BM,), tl.float32)
    moved_sq = tl.zeros((BM,), tl.float32)
    if INJECT:
        step = tl.load(Steps + token)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(Delta + (token * P + pp[:, None]) * D + kk[None, :], live[:, None], 0).to(tl.float32)
        if ADD:
            d += tl.load(Add + (token * P + pp[:, None]) * D + kk[None, :], live[:, None], 0).to(tl.float32)
        if INJECT:
            if HADAMARD:
                d += step * _signed_hadamard(Frame, pp, kk)
            else:
                d += step * tl.load(Frame + pp[:, None] * D + kk[None, :], live[:, None], 0)
        x = tl.load(X + token * D + kk)
        sq_x += tl.sum(x * x, 0)
        dots += tl.sum(d * x[None, :], 1)
        moved_sq += tl.sum(d * d, 1)
        w = tl.load(Fc + chosen[:, None] * D + kk[None, :])
        acc = tl.dot(d.to(tl.bfloat16), tl.trans(w), acc)
    inv0 = tl.math.rsqrt(sq_x / D + EPSILON)
    inv = tl.math.rsqrt((sq_x + 2 * dots + moved_sq) / D + EPSILON)
    h = tl.load(Clean + token * R + rr).to(tl.float32)
    dh = inv[:, None] * acc.to(tl.bfloat16).to(tl.float32) + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    change = ((up - down) * (up + down)).to(tl.bfloat16)
    if MODE != 2:
        scale = tl.load(Scale).to(tl.float32)
        q0 = tl.zeros((BM,), tl.float32)
        q1 = tl.zeros((BM,), tl.float32)
    else:
        s0 = tl.zeros((BM,), tl.float32)
        s1 = tl.zeros((BM,), tl.float32)
        s2 = tl.zeros((BM,), tl.float32)
        s3 = tl.zeros((BM,), tl.float32)
        s4 = tl.zeros((BM,), tl.float32)
        s5 = tl.zeros((BM,), tl.float32)
    for n0 in range(0, D, BN):
        nn = n0 + tl.arange(0, BN)
        w = tl.load(Wo + chosen[:, None] * D + nn[None, :])
        moved = tl.dot(change, w).to(tl.bfloat16).to(tl.float32)
        old = tl.load(Delta + (token * P + pp[:, None]) * D + nn[None, :], live[:, None], 0).to(tl.float32)
        if ADD:
            old += tl.load(Add + (token * P + pp[:, None]) * D + nn[None, :], live[:, None], 0).to(tl.float32)
        if INJECT:
            if HADAMARD:
                old += step * _signed_hadamard(Frame, pp, nn)
            else:
                old += step * tl.load(Frame + pp[:, None] * D + nn[None, :], live[:, None], 0)
        value = (old + moved).to(tl.bfloat16).to(tl.float32)
        if MODE != 2:
            result = (value * scale).to(tl.bfloat16)
            tl.store(Out + (token * P + pp[:, None]) * D + nn[None, :], result, live[:, None])
            if MODE == 3:
                stored = result.to(tl.float32)
                q0 += tl.sum(stored * tl.load(XL + token * D + nn)[None, :], 1)
                q1 += tl.sum(stored * stored, 1)
        else:
            xl = tl.load(XL + token * D + nn)
            s0 += tl.sum(value * xl[None, :], 1)
            s1 += tl.sum(value * value, 1)
            s2 += tl.sum(value * tl.load(Target + token * D + nn)[None, :], 1)
            s3 += tl.sum(value * tl.load(Mu + token * D + nn)[None, :], 1)
            s4 += tl.sum(value * tl.load(Nu + token * D + nn)[None, :], 1)
            s5 += tl.sum(value * value * tl.load(Diag + token * D + nn)[None, :], 1)
    if MODE == 3:
        tl.store(Stats + (token * P + pp) * 2, q0, live)
        tl.store(Stats + (token * P + pp) * 2 + 1, q1, live)
    if MODE == 2:
        base = (token * P + pp) * 6
        tl.store(Stats + base, s0, live)
        tl.store(Stats + base + 1, s1, live)
        tl.store(Stats + base + 2, s2, live)
        tl.store(Stats + base + 3, s3, live)
        tl.store(Stats + base + 4, s4, live)
        tl.store(Stats + base + 5, s5, live)


@triton.jit
def _signed_hadamard(Signs, pp, dd):
    """Entries (pp, dd) of the signed Sylvester Hadamard frame H * signs: signs[d] * (-1)^popcount(p & d)."""
    bits = pp[:, None] & dd[None, :]
    bits ^= bits >> 8
    bits ^= bits >> 4
    bits ^= bits >> 2
    bits ^= bits >> 1
    return (1 - 2 * (bits & 1)).to(tl.float32) * tl.load(Signs + dd)[None, :]


@triton.jit
def _fused_root_kernel(Steps, Basis, Proj, Wo, Routes, Clean, Inv0, Inv, Scale, Out, NextX, Stats,
    P: tl.constexpr, D: tl.constexpr, R: tl.constexpr, QSTATS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    HADAMARD: tl.constexpr):
    """A chain replay's root visit for BM moves of one token: each move is the signed step along a frame
    direction, so the routed gate change is read from the precomputed projection of the c_fc rows on the frame
    and the output GEMM adds the move itself; QSTATS also writes the next block's query-norm sums."""
    token = tl.program_id(1)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    live = pp < P
    rr = tl.arange(0, R)
    chosen = tl.load(Routes + token * R + rr)
    q = tl.load(Proj + chosen[None, :] * P + pp[:, None], live[:, None], 0)
    step = tl.load(Steps + token)
    moved = (q * step.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = tl.load(Inv + token * P + pp, live, 0)
    inv0 = tl.load(Inv0 + token)
    h = tl.load(Clean + token * R + rr).to(tl.float32)
    dh = inv[:, None] * moved + (inv / inv0 - 1)[:, None] * h[None, :]
    up, down = tl.maximum(h[None, :] + dh, 0), tl.maximum(h[None, :], 0)
    change = ((up - down) * (up + down)).to(tl.bfloat16)
    scale = tl.load(Scale).to(tl.float32)
    q0 = tl.zeros((BM,), tl.float32)
    q1 = tl.zeros((BM,), tl.float32)
    for n0 in range(0, D, BN):
        nn = n0 + tl.arange(0, BN)
        out_moved = tl.dot(change, tl.load(Wo + chosen[:, None] * D + nn[None, :])).to(tl.bfloat16).to(tl.float32)
        if HADAMARD:
            old = step * _signed_hadamard(Basis, pp, nn)
        else:
            old = step * tl.load(Basis + pp[:, None] * D + nn[None, :], live[:, None], 0)
        result = ((old + out_moved).to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(Out + (token * P + pp[:, None]) * D + nn[None, :], result, live[:, None])
        if QSTATS:
            stored = result.to(tl.float32)
            q0 += tl.sum(stored * tl.load(NextX + token * D + nn)[None, :], 1)
            q1 += tl.sum(stored * stored, 1)
    if QSTATS:
        tl.store(Stats + (token * P + pp) * 2, q0, live)
        tl.store(Stats + (token * P + pp) * 2 + 1, q1, live)


@torch.library.custom_op('noise_record::fused_root_visit', mutates_args=())
def fused_root_visit(steps: torch.Tensor, basis: torch.Tensor, projection: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, inv0: torch.Tensor, inv: torch.Tensor, scale: torch.Tensor,
    next_x: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """route_mlp_root (and its query-stats variant when next_x is given) in one kernel."""
    n, p, d, r = steps.shape[0], basis.shape[0], wo.shape[1], routes.shape[1]
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    qstats = next_x is not None
    stats = torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32)
    _fused_root_kernel[(triton.cdiv(p, 64), n)](
      steps.contiguous(), basis[0].float().contiguous() if FRAME_HADAMARD and p == d else basis.float().contiguous(),
      projection.contiguous(), wo, routes.contiguous(),
      clean.contiguous(), inv0.contiguous(), inv.contiguous(), scale, out,
      next_x.float().contiguous() if qstats else out, stats, p, d, r, qstats, BM=64, BN=64,
      HADAMARD=FRAME_HADAMARD and p == d, num_warps=4, num_stages=3)
    return out, stats


@fused_root_visit.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, scale, next_x):
    n, p, d = steps.shape[0], basis.shape[0], wo.shape[1]
    return (torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      torch.empty((n, p, 2, 1), device=wo.device, dtype=torch.float32))


KEY_DOT_PRECISION = 'tf32'  # the key / value credit kernels' logits and scores (tf32x3: cos 1.000000, query credits' norms within 0.35%)
FRAME_HADAMARD = True  # frame entries computed in-kernel: a full frame is the signed Sylvester Hadamard (row 0 = signs)
FUSED_VISIT_CONFIG = dict(BM=64, BK=64, BN=64, num_warps=4, num_stages=3)


def launch_fused_visit(delta, steps, frame, x, fc, wo, routes, clean, scale, xl, target, mu, nu, diag, mode,
    add=None):
    n, p, d = delta.shape
    r = routes.shape[1]
    inject = steps is not None
    out = delta if mode == 2 else torch.empty_like(delta)
    stats = (x if mode == 1 else torch.empty((n, p, 6 if mode == 2 else 2), device=delta.device,
      dtype=torch.float32))
    cfg = dict(FUSED_VISIT_CONFIG)
    bm = cfg.pop('BM')
    _fused_visit_kernel[(triton.cdiv(p, bm), n)](
      delta, steps.contiguous() if inject else delta,
      (frame[0].float().contiguous() if FRAME_HADAMARD and p == d else frame.float().contiguous()) if inject else delta,
      x.float().contiguous(), fc, wo, routes.contiguous(), clean.contiguous(), scale, out,
      target, mu, nu, diag, xl, stats, delta if add is None else add.contiguous(), p, d, r, inject, mode,
      add is not None, inject and FRAME_HADAMARD and p == d, BM=bm, EPSILON=EPS, **cfg)
    return out if mode == 1 else stats if mode == 2 else (out, stats)


@torch.library.custom_op('noise_record::fused_visit', mutates_args=())
def fused_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor, routes: torch.Tensor,
    clean: torch.Tensor, scale: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor], add: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A routed MLP visit with the next block's residual scalar applied (MODE 1); add: the attention replay's
    response to this block's input, added to delta on load."""
    return launch_fused_visit(delta.contiguous(), steps, frame, x, fc, wo, routes, clean, scale,
      x, x, x, x, x, 1, add)


@fused_visit.register_fake
def _(delta, x, fc, wo, routes, clean, scale, steps, frame, add=None):
    return torch.empty_like(delta)


@torch.library.custom_op('noise_record::fused_query_visit', mutates_args=())
def fused_query_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor, routes: torch.Tensor,
    clean: torch.Tensor, scale: torch.Tensor, next_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A routed MLP visit before an attention replay (MODE 3): the scaled output and, per move, its dot with the
    next block's clean input and its square, viewed as (token, move, statistic, 1) partials."""
    out, stats = launch_fused_visit(delta.contiguous(), None, None, x, fc, wo, routes, clean, scale,
      next_x.float().contiguous(), x, x, x, x, 3)
    return out, stats.unsqueeze(-1)


@fused_query_visit.register_fake
def _(delta, x, fc, wo, routes, clean, scale, next_x):
    return torch.empty_like(delta), torch.empty((*delta.shape[:2], 2, 1), device=delta.device, dtype=torch.float32)


@torch.library.custom_op('noise_record::fused_readout_visit', mutates_args=())
def fused_readout_visit(delta: torch.Tensor, x: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
    routes: torch.Tensor, clean: torch.Tensor, xl: torch.Tensor, target: torch.Tensor, mu: torch.Tensor,
    nu: torch.Tensor, diag: torch.Tensor, steps: Optional[torch.Tensor],
    frame: Optional[torch.Tensor]) -> torch.Tensor:
    """The last block's routed MLP visit: the readout-moment sums of the moved final stream (MODE 2)."""
    f32 = lambda z: z.float().contiguous()
    return launch_fused_visit(delta.contiguous(), steps, frame, x, fc, wo, routes, clean, x,
      f32(xl), f32(target), f32(mu), f32(nu), f32(diag), 2)


@fused_readout_visit.register_fake
def _(delta, x, fc, wo, routes, clean, xl, target, mu, nu, diag, steps, frame):
    return torch.empty((*delta.shape[:2], 6), device=delta.device, dtype=torch.float32)


def attention_replay_layers(roots, layers, long_layers):
    return tuple(b for b in range(roots[0] + 1, layers) if b in long_layers)


@triton.jit
def _route_output_query_stats_kernel(
  Change, Weight, Routes, Delta, Out, Steps, Basis,
  NextX, Scale, Stats,
  P: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
  S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
  ROOT: tl.constexpr,
  NB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    token = tl.program_id(2)
    pp = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * BN + tl.arange(0, BN)
    old, moved = _route_output_parts(
        Change, Weight, Routes, Delta, Steps, Basis, token, pp, dd,
        P, D, R, S0, S1, S2, ROOT, BM, BN, BK, False)
    result = (old + moved).to(tl.bfloat16)
    result = (result.to(tl.float32) * tl.load(Scale)).to(tl.bfloat16)
    tl.store(
      Out + (token * P + pp[:, None]) * D + dd[None, :],
      result, (pp[:, None] < P) & (dd[None, :] < D),
    )
    value = result.to(tl.float32)
    x = tl.load(NextX + token * D + dd, dd < D, 0).to(tl.float32)
    dot = tl.sum(value * x[None, :], 1)
    square = tl.sum(value * value, 1)
    position = (token * 2 * NB + tl.program_id(1)) * P + pp
    ss: tl.constexpr = NB * P
    tl.store(Stats + position, dot, pp < P)
    tl.store(Stats + position + ss, square, pp < P)


@torch.library.custom_op('noise_record::route_mlp_query_stats', mutates_args=())
def route_mlp_query_stats(
  delta: torch.Tensor, fc: torch.Tensor, wo: torch.Tensor,
  routes: torch.Tensor, clean: torch.Tensor,
  inv0: torch.Tensor, inv: torch.Tensor,
  out_scale: torch.Tensor, next_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, p, d = delta.shape
    r = routes.shape[1]
    routes, clean, next_x = routes.contiguous(), clean.contiguous(), next_x.contiguous()
    tile, nb = min(64, r), triton.cdiv(d, 64)
    change = torch.empty((n, p, r), device=delta.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=delta.device, dtype=torch.bfloat16)
    partials = mlp_statistics_storage(n, p, 2, nb, delta.device)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      delta, fc, routes, clean, inv0.contiguous(), inv.contiguous(),
      change, delta, delta,
      p, d, r, *delta.stride(), False,
      BM=64, BN=tile, BK=64, num_warps=4, num_stages=3,
    )
    _route_output_query_stats_kernel[
      lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)
    ](
      change, wo, routes, delta, out, delta, delta,
      next_x, out_scale, partials,
      p, d, r, *delta.stride(), False, nb,
      BM=64, BN=64, BK=tile, num_warps=4, num_stages=3,
    )
    return out, partials


@route_mlp_query_stats.register_fake
def _(delta, fc, wo, routes, clean, inv0, inv, out_scale, next_x):
    n, p, d = delta.shape
    return (
      torch.empty(delta.shape, device=delta.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), delta.device),
    )


@torch.library.custom_op('noise_record::route_mlp_root_query_stats', mutates_args=())
def route_mlp_root_query_stats(
  steps: torch.Tensor, basis: torch.Tensor,
  projection: torch.Tensor, wo: torch.Tensor,
  routes: torch.Tensor, clean: torch.Tensor,
  inv0: torch.Tensor, inv: torch.Tensor,
  out_scale: torch.Tensor, next_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d, p = steps.shape[0], wo.shape[1], basis.shape[0]
    r = routes.shape[1]
    steps, basis = steps.contiguous(), basis.contiguous()
    routes, clean, next_x = routes.contiguous(), clean.contiguous(), next_x.contiguous()
    nb = triton.cdiv(d, 64)
    change = torch.empty((n, p, r), device=wo.device, dtype=torch.bfloat16)
    out = torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16)
    partials = mlp_statistics_storage(n, p, 2, nb, wo.device)
    _route_gate_kernel[lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(r, meta['BN']), n)](
      steps, steps, routes, clean, inv0.contiguous(), inv.contiguous(),
      change, steps, projection,
      p, d, r, 0, 0, 0, True,
      BM=64, BN=64, BK=64, num_warps=4, num_stages=3,
    )
    _route_output_query_stats_kernel[
      lambda meta: (triton.cdiv(p, meta['BM']), triton.cdiv(d, meta['BN']), n)
    ](
      change, wo, routes, steps, out, steps, basis,
      next_x, out_scale, partials,
      p, d, r, 0, 0, 0, True, nb,
      BM=64, BN=64, BK=64, num_warps=4, num_stages=3,
    )
    return out, partials


@route_mlp_root_query_stats.register_fake
def _(steps, basis, projection, wo, routes, clean, inv0, inv, out_scale, next_x):
    n, p, d = steps.shape[0], basis.shape[0], wo.shape[1]
    return (
      torch.empty((n, p, d), device=wo.device, dtype=torch.bfloat16),
      mlp_statistics_storage(n, p, 2, triton.cdiv(d, 64), wo.device),
    )


def query_inverses_from_output_partials(next_x, partials):
    stats = partials.sum(-1)
    square = next_x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(square / next_x.shape[-1] + EPS)
    inv = torch.rsqrt(
      (square + 2 * stats[:, :, 0] + stats[:, :, 1]) / next_x.shape[-1] + EPS
    )
    return inv0, inv


def chain_chunk_eager(site, root_moves, mids, xL, idx, hs, lam_r, w_fc, w_out_t, base,
    target_logit_w, y, own, moments, root_projection, injections, query_layers):
    """Replay one root group: the first root's signed frame moves enter at site and each partner's at its own
    block, through routed MLPs, with attention replayed only at query_layers. A partner must be the last block
    or be followed by a block that is neither replayed with attention nor another partner. Returns each token's
    moved-minus-clean loss for every move, from diagonal readout moments."""
    L = len(mids)
    pending_inv = pending_query_inv = replay_add = factored = None
    delta = None
    # CHAIN_READOUT: the visit (or root) at block L-2 and the readout at L-1 in one kernel (visit_chain)
    chain_tail = (CHAIN_READOUT and FUSED_CHAIN and L - 1 not in query_layers
                  and all(i != L - 1 for i, _, _ in injections) and idx[L - 1].shape[1] == 64
                  and idx[L - 2] is not None and idx[L - 2].shape[1] == 64)
    tail = lambda: (mids[L - 1], w_fc[L - 1], w_out_t[L - 1], idx[L - 1], hs[L - 1], xL, target_logit_w,
                    moments[2], moments[3], moments[4])
    for b in range(site, L):
        query_inv, pending_query_inv = pending_query_inv, None
        next_scale = lam_r[b + 1] if b + 1 < L else None
        next_query_x = own[1][b + 1] if b + 1 in query_layers else None
        if b > site:
            if b in query_layers:
                pos, inputs, w_q, keys, values, prob, cos_sin = own
                if FUSED_CHAIN and FUSED_ADD_ON_LOAD:
                    parts = FUSED_RESPONSE and keys[b].shape[1] * keys[b].shape[2] == 128
                    replay_add = query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv, raw_bf16='parts' if parts else True,
                        factored=factored if b == site + 1 else None)
                elif FUSED_CHAIN and FUSED_REPLAY_ADD:
                    delta = query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv, add_to=delta.to(torch.bfloat16))
                else:
                    delta = delta + query_change_routed(inputs[b], delta, pos, w_q[b], keys[b], values[b],
                        prob[b], cos_sin[0], cos_sin[1], query_inv=query_inv)
            delta = delta.to(torch.bfloat16)
        if b == site:
            inv0, inv = root_norm_inverses(mids[b], *root_moves)
            if chain_tail and b == L - 2 and next_query_x is None and FUSED_ROOT:
                terminal_stats = visit_chain.fused_root_readout_fast(*root_moves, root_projection, w_out_t[b], idx[b],
                    hs[b], inv0, inv, next_scale, *tail())
                break
            if (FACTORED_QUERY and FUSED_CHAIN and FUSED_ROOT and GLUON_ROOT and next_query_x is not None
                    and idx[b].shape[1] == 64):  # the root's routed change too, for the factored query replay
                delta, change, partials = visit_gluon_root.fused_root_visit_change(*root_moves, root_projection,
                    w_out_t[b], idx[b], hs[b], inv0, inv, next_scale, next_query_x)
                factored = (change, root_moves[0], next_scale, w_out_t[b], idx[b], root_moves[1])
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
                continue
            if FUSED_CHAIN and FUSED_ROOT:
                root_visit = visit_gluon_root.fused_root_visit_fast if GLUON_ROOT else fused_root_visit
                delta, partials = root_visit(*root_moves, root_projection, w_out_t[b], idx[b], hs[b], inv0,
                    inv, next_scale, next_query_x)
                if next_query_x is not None:
                    pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif next_query_x is None:
                delta = route_mlp_root(*root_moves, root_projection, w_out_t[b],
                    idx[b], hs[b], inv0, inv, next_scale)
            else:
                delta, partials = route_mlp_root_query_stats(*root_moves, root_projection,
                    w_out_t[b], idx[b], hs[b], inv0, inv, next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            continue
        injected = next(((steps, frame) for i, steps, frame in injections if i == b), None)
        if FUSED_CHAIN:
            steps, frame = injected if injected is not None else (None, None)
            pending_inv = None
            if (CHAIN_RESPONSE and chain_tail and b == L - 3 and steps is None and isinstance(replay_add, tuple)
                    and next_query_x is None and L - 2 not in query_layers and idx[b].shape[1] == 64
                    and all(i != L - 2 for i, _, _ in injections)):  # blocks L-3 (with the replay) to L-1 in one kernel
                m = L - 2
                terminal_stats = visit_chain3.fused_resp_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b],
                    idx[b], hs[b], next_scale, *replay_add, mids[m], w_fc[m], w_out_t[m], idx[m], hs[m], lam_r[m + 1],
                    *tail())
                break
            if chain_tail and b == L - 2 and steps is None and replay_add is None and next_query_x is None:
                terminal_stats = visit_chain.fused_visit_readout_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                    hs[b], next_scale, *tail())
                break
            if next_query_x is not None and not FUSED_QUERY_VISIT:
                inv0, inv = norm_inverses(mids[b], delta)
                delta, partials = route_mlp_query_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                    next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif next_query_x is not None:
                visit = visit_gluon.fused_query_visit_fast if GLUON_QUERY_VISIT else fused_query_visit
                delta, partials = visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
            elif b == L - 1:
                assert replay_add is None, 'an attention replay must be followed by a routed visit that takes it'
                if (GLUON_INJECT and steps is not None) or (GLUON_READOUT and steps is None and idx[b].shape[1] == 64):
                    terminal_stats = visit_gluon.fused_readout_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b],
                        hs[b], xL, target_logit_w, moments[2], moments[3], moments[4], steps,
                        None if frame is None else frame[0])
                else:
                    terminal_stats = fused_readout_visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], xL,
                        target_logit_w, moments[2], moments[3], moments[4], steps, frame)
            elif GLUON_VISIT and steps is None and replay_add is None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale)
            elif GLUON_INJECT and steps is not None and replay_add is None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, frame[0])
            elif GLUON_ADD and isinstance(replay_add, tuple):
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, None if frame is None else frame[0], add_weights=replay_add[0], add_values=replay_add[1])
                replay_add = None
            elif GLUON_ADD and replay_add is not None:
                delta = visit_gluon.fused_visit_fast(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale,
                    steps, None if frame is None else frame[0], replay_add)
                replay_add = None
            else:
                delta = fused_visit(delta, mids[b], w_fc[b], w_out_t[b], idx[b], hs[b], next_scale, steps, frame,
                    replay_add)
                replay_add = None
            continue
        if injected is not None:
            steps, frame = injected
            inv0, inv, packed = injected_norm_packed(mids[b], delta, steps, frame)
        else:
            steps = frame = packed = None
            inv0, inv = norm_inverses(mids[b], delta) if pending_inv is None else pending_inv
            pending_inv = None
        next_attention_held = (b + 1 < L and b + 1 not in query_layers
            and all(i != b + 1 for i, _, _ in injections))
        if next_attention_held:
            delta, partials = mlp_output_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                mids[b+1], lam_r[b+1], steps, frame, packed)
            stats = partials.sum(-1)
            clean_sq = mids[b+1].square().sum(-1, keepdim=True)
            pending_inv = (torch.rsqrt(clean_sq / mids[b+1].shape[-1] + EPS),
                torch.rsqrt((clean_sq + 2*stats[:, :, 0] + stats[:, :, 1]) / mids[b+1].shape[-1] + EPS))
        elif b == L - 1:
            terminal_stats = mlp_readout_stats(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv,
                xL, target_logit_w, moments[2], moments[3], moments[4], steps, frame, packed).sum(-1)
        else:
            if next_query_x is None:
                delta = route_mlp(delta, w_fc[b], w_out_t[b], idx[b], hs[b], inv0, inv, next_scale)
            else:
                delta, partials = route_mlp_query_stats(delta, w_fc[b], w_out_t[b], idx[b],
                    hs[b], inv0, inv, next_scale, next_query_x)
                pending_query_inv = query_inverses_from_output_partials(next_query_x, partials)
    sq = xL.square().sum(-1, keepdim=True)
    dots, dsq, target_dot = terminal_stats[:, :, 0], terminal_stats[:, :, 1], terminal_stats[:, :, 2]
    inv = torch.rsqrt((sq + 2 * dots + dsq) / xL.shape[-1] + EPS)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + target_dot) * inv
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    b1, b2 = moments[:2]
    a = inv - inv0
    mu_dot = terminal_stats[:, :, 3]
    nu_dot = terminal_stats[:, :, 4]
    lin = a * b1.unsqueeze(1) + inv * mu_dot
    quad = terminal_stats[:, :, 5]
    change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * nu_dot
        + inv.square() * quad - 0.5 * lin.square())
    target0 = base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0
    moved_minus_clean = change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(target0 / 15))
    return moved_minus_clean * (y >= 0).unsqueeze(1)


FUSED_CHAIN = False  # set from Settings.fused_chain in Cut.__init__ (read when chain_chunk compiles)
FUSED_QUERY_VISIT = True  # the visit before the attention replay fused too (MODE 3)
FUSED_REPLAY_ADD = False  # measured slower (round 4)
FUSED_ADD_ON_LOAD = True  # the attention replay's response added inside the next fused visit's loads
FUSED_ROOT = True  # root visits in one kernel (_fused_root_kernel)
GLUON_QUERY_VISIT = True  # the visit before the attention replay (MODE 3) by visit_gluon too
FUSED_QUERY_REPLAY = True  # with FUSED_RESPONSE: the replay's q projection inside query_replay's kernel
GLUON_ROOT = True  # root visits by visit_gluon_root's Hopper kernel
FUSED_RESPONSE = True  # with GLUON_ADD and 128 replay key-heads: the replay's response formed inside the visit
GLUON_ADD = True  # the visit that takes the attention replay's response by visit_gluon too
GLUON_INJECT = True  # chain visits with an injected partner move (MODE 1 and the readout) by visit_gluon too
BLOCKED_PROJECTION = True  # the routed values' projection by project_values (bitwise)
WINDOW_TOPK = True  # the strong keys' top-k by window_topk (bitwise equal to torch.topk)
GLUON_FACTORED = True  # with FACTORED_QUERY: the Gluon factored replay (query_factored_gluon)
FACTORED_QUERY = True  # the replay right after a root: its query change from the root's routed factors
CHAIN_RESPONSE = True  # with CHAIN_READOUT: also the replay-response visit before them (visit_chain3)
CHAIN_READOUT = True  # the last two blocks' visits of a replay in one kernel (visit_chain, 64 routes)
GLUON_READOUT = True  # plain readout visits at 64 routes by visit_gluon (staged readout vectors)
GLUON_VISIT = True  # plain MODE 1 chain visits by visit_gluon's Hopper kernel (delta tile in shared memory)
chain_chunk = torch.compile(chain_chunk_eager, dynamic=False)


@triton.jit
def _implicit_key_statistics(Q, K, Values, Hub, LSE, Center,
    T: tl.constexpr, Window: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    query = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 64)
    src = tl.arange(0, BN)
    base = head * T * 64
    q = tl.load(Q + base + query[:, None] * 64 + dims[None, :])
    hub = tl.load(Hub + base + query[:, None] * 64 + dims[None, :])
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    score_sum = tl.zeros((BM,), tl.float32)
    first = tl.maximum(tile * BM - Window, 0) // BN
    last = tl.cdiv((tile + 1) * BM, BN)
    for block in range(first, last):
        source = block * BN + src
        k = tl.load(K + base + source[:, None] * 64 + dims[None, :])
        v = tl.load(Values + base + source[:, None] * 64 + dims[None, :])
        logits = tl.dot(q, tl.trans(k), input_precision=PREC) * 0.125
        scores = tl.dot(hub, tl.trans(v), input_precision=PREC)
        allowed = (source[None, :] <= query[:, None]) & (query[:, None] - source[None, :] <= Window)
        logits = tl.where(allowed, logits, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(next_max == float('-inf'), 0.0, next_max)
        rescale = tl.exp(maximum - safe_max)
        prob = tl.exp(logits - safe_max[:, None])
        total = total * rescale + tl.sum(prob, 1)
        score_sum = score_sum * rescale + tl.sum(prob * scores, 1)
        maximum = next_max
    tl.store(LSE + head * T + query, maximum + tl.log(total))
    tl.store(Center + head * T + query, score_sum / total)


def implicit_key_context(q, k, values, hub, window):
    h, t, _ = q.shape
    q, k, values, hub = (z.contiguous() for z in (q, k, values, hub))
    lse = torch.empty((h, t), device=q.device)
    center = torch.empty_like(lse)
    _implicit_key_statistics[(t // 16, h)](q, k, values, hub, lse, center,
      T=t, Window=window, BM=16, BN=64, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    return lse, center


@triton.jit
def _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D: tl.constexpr, Sigma: tl.constexpr):
    raw = tl.load(Raw + base + target[:, None] * D + dims[None, :])
    raw_j = tl.load(Raw + base + target[:, None] * D + dirs[None, :])
    sq = tl.sum(raw * raw, 1)[:, None] / D
    step = Sigma * tl.maximum(tl.sqrt(sq), 1e-06)
    ratio = (2 * step * raw_j + step * step) / D / (sq + 1.1920928955078125e-07)
    ca = libdevice.expm1(-0.5 * libdevice.log1p(ratio))
    cb = step * libdevice.rsqrt(sq + 1.1920928955078125e-07) * (1 + ca)
    cs = tl.load(Cos + target[:, None] * (D // 2) + pair[None, :])
    sn = tl.load(Sin + target[:, None] * (D // 2) + pair[None, :])
    bu = cb * tl.where(dirs[None, :] < D // 2, cs, sn)
    bw = cb * tl.where(dirs[None, :] < D // 2, -sn, cs)
    return (ca, bu, bw, step)


@triton.jit
def _key_scan_kernel(Q, K, Raw, LSE, Values, Hub, Center, Cos, Sin, Out, T: tl.constexpr, D: tl.constexpr,
    Window: tl.constexpr, Sigma: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, J: tl.constexpr, PREC: tl.constexpr):
    head = tl.program_id(0)
    t0 = tl.program_id(1) * BT
    target = t0 + tl.arange(0, BT)
    dims = tl.arange(0, D)
    dirs = tl.arange(0, D)
    feature_pair = tl.arange(0, J)
    pair = dirs % (D // 2)
    base = head * T * D
    key = tl.load(K + base + target[:, None] * D + dims[None, :])
    value = tl.load(Values + base + target[:, None] * D + dims[None, :])
    p2 = tl.zeros((BT, 2 * J), tl.float32)
    q2 = tl.zeros((BT, 2 * J), tl.float32)
    qxy = tl.zeros((BT, J), tl.float32)
    r2 = tl.zeros((BT, 2 * J), tl.float32)
    s1 = tl.zeros((BT,), tl.float32)
    s2 = tl.zeros((BT,), tl.float32)
    first = t0 // BS
    last = tl.minimum((t0 + BT - 1 + Window) // BS + 1, T // BS)
    for it in range(first, last):
        source = it * BS + tl.arange(0, BS)
        query = tl.load(Q + base + source[:, None] * D + dims[None, :])
        logit = tl.dot(key, tl.trans(query), input_precision=PREC) * D ** (-0.5)
        valid = (source[None, :] >= target[:, None]) & (source[None, :] - target[:, None] <= Window)
        hub = tl.load(Hub + base + source[:, None] * D + dims[None, :])
        score = tl.dot(value, tl.trans(hub), input_precision=PREC)
        center = tl.load(Center + head * T + source)
        c = tl.where(valid, score - center[None, :], 0.0)
        lse = tl.load(LSE + head * T + source)
        a = tl.where(valid, tl.exp(logit - lse[None, :]), 0.0)
        x = tl.load(Q + base + source[:, None] * D + feature_pair[None, :]) * D ** (-0.5)
        y = tl.load(Q + base + source[:, None] * D + (feature_pair + D // 2)[None, :]) * D ** (-0.5)
        f2 = tl.reshape(tl.join(x, y), (BS, 2 * J))
        g2 = tl.reshape(tl.join(x * x, y * y), (BS, 2 * J))
        xy = x * y
        w1 = a * c
        w2 = (0.5 * a - a * a) * c
        w2l = w2 * logit
        w1d, w2d, w2ld = (w1.to(tl.bfloat16), w2.to(tl.bfloat16), w2l.to(tl.bfloat16))
        f2d, g2d, xyd = (f2.to(tl.bfloat16), g2.to(tl.bfloat16), xy.to(tl.bfloat16))
        p2 += tl.dot(w1d, f2d)
        q2 += tl.dot(w2d, g2d)
        qxy += tl.dot(w2d, xyd)
        r2 += tl.dot(w2ld, f2d)
        s1 += tl.sum(w1 * logit, 1)
        s2 += tl.sum(w2 * logit * logit, 1)
    ca, bu, bw, step = _key_perturbation_geometry(Raw, Cos, Sin, base, target, dims, dirs, pair, D, Sigma)
    px, py = tl.split(tl.reshape(p2, (BT, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BT, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BT, J, 2)))
    gather_pair = tl.broadcast_to(pair[None, :], (BT, D))
    px, py = (tl.gather(px, gather_pair, 1), tl.gather(py, gather_pair, 1))
    qx, qy = (tl.gather(qx, gather_pair, 1), tl.gather(qy, gather_pair, 1))
    rx, ry = (tl.gather(rx, gather_pair, 1), tl.gather(ry, gather_pair, 1))
    qxy = tl.gather(qxy, gather_pair, 1)
    response = px * bu + py * bw + qx * bu * bu + qy * bw * bw + 2 * qxy * bu * bw + 2 * ca * (rx * bu + ry * bw) + s1[:, None] * ca + s2[:, None] * ca * ca
    tl.store(Out + base + target[:, None] * D + dirs[None, :], response / step)


@triton.jit
def _value_credit_kernel(Q,K,LSE,Hub,Out,T:tl.constexpr,Window:tl.constexpr,
    BT:tl.constexpr,BS:tl.constexpr,Channels:tl.constexpr,PREC:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    target=tile*BT+tl.arange(0,BT)
    dims=tl.arange(0,64)
    channels=tl.arange(0,Channels)
    base=head*T*64
    key=tl.load(K+base+target[:,None]*64+dims[None,:])
    acc=tl.zeros((BT,Channels),tl.float32)
    first=tile*BT//BS
    last=tl.minimum((tile*BT+BT-1+Window)//BS+1,T//BS)
    for it in range(first,last):
        receiver=it*BS+tl.arange(0,BS)
        query=tl.load(Q+base+receiver[:,None]*64+dims[None,:])
        hub=tl.load(Hub+head*T*Channels+receiver[:,None]*Channels+channels[None,:])
        lse=tl.load(LSE+head*T+receiver)
        logits=tl.dot(key,tl.trans(query),input_precision=PREC)*.125
        valid=(receiver[None,:]>=target[:,None]) & (receiver[None,:]-target[:,None]<=Window)
        prob=tl.where(valid,tl.exp(logits-lse[None,:]),0.).to(tl.bfloat16)
        hi=hub.to(tl.bfloat16)
        lo=(hub-hi.to(tl.float32)).to(tl.bfloat16)
        acc+=tl.dot(prob,hi)
        acc+=tl.dot(prob,lo)
    tl.store(Out+head*T*Channels+target[:,None]*Channels+channels[None,:],acc)


def value_credits(context, corrected, window):
    """key_and_value_credits' value part alone (the key part batched across layers by batched_key_contexts)."""
    q, k, raw, cos, sin, lse, (v, hub, center) = context
    h, t, d = raw.shape
    paired = torch.cat((hub.contiguous(), corrected.contiguous()), dim=-1)
    out = torch.empty_like(paired)
    _value_credit_kernel[h, t // 32](q.contiguous(), k.contiguous(), lse.contiguous(), paired, out, T=t, Window=window,
        BT=32, BS=64, Channels=128, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    return out[..., :64].contiguous(), out[..., 64:].contiguous()


def key_and_value_credits(context, corrected, sigma, window):
    q, k, raw, cos, sin, lse, (v, hub, center) = context
    q, k, raw, cos, sin, lse, v, hub, center, corrected = (z.contiguous() for z in (q, k, raw, cos, sin, lse, v, hub, center, corrected))
    h, t, d = raw.shape
    key = torch.empty_like(raw)
    _key_scan_kernel[h, t // 32](
        q, k, raw, lse, v, hub, center, cos, sin, key, T=t, D=d, Window=window, Sigma=sigma,
        BT=32, BS=64, J=32, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2,
    )
    paired = torch.cat((hub, corrected), dim=-1)
    out = torch.empty_like(paired)
    _value_credit_kernel[h, t // 32](q, k, lse, paired, out, T=t, Window=window, BT=32, BS=64, Channels=128, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
    value, terminal = (out[..., :64].contiguous(), out[..., 64:].contiguous())
    return (key, value, terminal)


QUERY_FP8 = False  # the attention replay's moved-query projection in fp8 (rowwise scales)
COMPILED_CAPTURE = True  # Capture.capture by one compiled forward that returns what the hooks record
QKV_GEMM = True  # CAST_ONCE: c_q / c_k / c_v in one GEMM against their concatenated bf16 weight (outputs: column slices)
# the capture's flex attention tiles: BLOCK_N 128 as the default keeps it bitwise, BLOCK_M 64 / 4 warps / 3 stages
# fastest (None: the defaults)
CAPTURE_FLEX_OPTIONS = dict(BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=3)


def capture_attention(q, k, v, window):
    """utils.model.attention with CAPTURE_FLEX_OPTIONS (inside capture_forward's graph)."""
    if CAPTURE_FLEX_OPTIONS is None:
        return attention(q, k, v, window)
    mask = window_mask(window, q.shape[1], q.device)
    return flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask,
                          kernel_options=CAPTURE_FLEX_OPTIONS).transpose(1, 2)


@torch.compile(dynamic=False, options={'combo_kernels': True})
def capture_forward(model, idx, targets, weights=None):
    """GPT.forward (utils/model.py) under bf16 autocast, returning the tensors Capture's hooks record: each Linear's
    input (fp32) and output, the embeddings' outputs, each block's input and the per-token loss. weights: the Linears'
    bf16 weights by module name (cast once by the caller), else autocast casts them."""
    inputs, outputs, blocks = {}, {}, {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        b, t = idx.shape
        cos, sin = model.cos[:, :t], model.sin[:, :t]
        def linear(name, module, z):
            inputs[name] = z.float()
            out = module(z) if weights is None else F.linear(z, weights[name])
            outputs[name] = out
            return out
        e = model.transformer.wte(idx)
        inputs['transformer.wte'], outputs['transformer.wte'] = idx, e
        x = norm(e)
        x0 = x
        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            blocks[i] = x
            p = f'transformer.h.{i}.'
            attn = block.attn
            ve = None
            if str(i) in model.value_embeds:
                ve = model.value_embeds[str(i)](idx)
                inputs[f'value_embeds.{i}'], outputs[f'value_embeds.{i}'] = idx, ve
            xn = norm(x)
            h, hd = attn.n_head, attn.head_dim
            if weights is not None and p + 'attn.qkv' in weights:  # QKV_GEMM: one GEMM, outputs its column slices
                inputs[p + 'attn.c_q'] = xn.float()
                qkv = F.linear(xn, weights[p + 'attn.qkv'])
                c, ck = attn.c_q.weight.shape[0], attn.c_k.weight.shape[0]
                q, k, v = qkv[..., :c], qkv[..., c:c + ck], qkv[..., c + ck:]
                outputs[p + 'attn.c_q'], outputs[p + 'attn.c_k'], outputs[p + 'attn.c_v'] = q, k, v
                q, k, v = q.view(b, t, h, hd), k.view(b, t, attn.n_kv_head, hd), v.view(b, t, attn.n_kv_head, hd)
            else:
                q = linear(p + 'attn.c_q', attn.c_q, xn).view(b, t, h, hd)
                k = linear(p + 'attn.c_k', attn.c_k, xn).view(b, t, attn.n_kv_head, hd)
                v = linear(p + 'attn.c_v', attn.c_v, xn).view(b, t, attn.n_kv_head, hd)
            if ve is not None:
                gate = 2 * torch.sigmoid(linear(p + 'attn.ve_gate', attn.ve_gate, xn[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve.view(b, t, attn.n_kv_head, hd)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
            y = capture_attention(q, k, v, model.window_sizes[i][0]).contiguous().view(b, t, -1)
            x = x + linear(p + 'attn.c_proj', attn.c_proj, y)
            hid = linear(p + 'mlp.c_fc', block.mlp.c_fc, norm(x))
            x = x + linear(p + 'mlp.c_proj', block.mlp.c_proj, F.relu(hid).square())
        logits = linear('lm_head', model.lm_head, norm(x))[..., :model.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                               reduction='none')
    if FUSED_GLUE:  # each kind of block input stacked over the blocks ('stack:<kind>'; the per-block ones are views)
        L = len(model.transformer.h)
        for kind in STACKED_INPUTS:
            stack = torch.stack([inputs[f'transformer.h.{i}.{kind}'] for i in range(L)])
            inputs['stack:' + kind] = stack
            for i in range(L):
                inputs[f'transformer.h.{i}.{kind}'] = stack[i]
        for i in range(L):  # the same normed block input
            inputs[f'transformer.h.{i}.attn.c_k'] = inputs[f'transformer.h.{i}.attn.c_v'] = (
                inputs[f'transformer.h.{i}.attn.c_q'])
    return inputs, outputs, blocks, loss


STACKED_INPUTS = ('attn.c_q', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


class Capture:
    def __init__(self, model, seed):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
        self.weights = None  # (CAST_ONCE: the Linears' bf16 weights, set by Cut.estimate)
        self.capturing = False
        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer, block in enumerate(model.transformer.h):
            self.hooks.append(block.register_forward_pre_hook(self.block_hook(layer)))
    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
        return hook
    def block_hook(self, layer):
        def hook(module, args):
            if self.capturing:
                self.blocks[layer] = args[0]
        return hook
    def close(self):
        for hook in self.hooks:
            hook.remove()
    def signs(self, *shape, shared=False):
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1
    def query_noise(self, draws, b, t, d):
        patterns = self.signs(1, d, shared=True)
        offset = torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator)
        signs = self.signs(draws, b, t, 1)
        index = torch.arange(draws, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        return hadamard(d, self.device, self.dtype)[rows] * patterns[0] * signs
    def capture(self, x, y):
        self.y = y
        self.batch, self.length = x.shape
        if COMPILED_CAPTURE:
            self.inputs, self.outputs, self.blocks, loss = capture_forward(self.model, x, y, self.weights)
            self.clean_loss = loss.view_as(y).float()
            return
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
    def head_error(self, sigma=0.1):
        return coordinate_differences(self.outputs['lm_head'], self.y, sigma)
    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)


@triton.jit
def _route_moment_stream_kernel(Scores, K, V, Top, Probability, OutV, OutK,
    T: tl.constexpr, WINDOW: tl.constexpr, KEYS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    qs = tile * BM + tl.arange(0, BM)
    ds = tl.arange(0, 64)
    ns = tl.arange(0, BN)
    running_max = tl.full((BM,), float('-inf'), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    first = tl.maximum(tile * BM - WINDOW, 0) // BN
    last = tl.cdiv((tile + 1) * BM, BN)
    for block in range(first, last):
        ks = block * BN + ns
        score = tl.load(Scores + (head * T + qs[:, None]) * T + ks[None, :]).to(tl.float32) * 0.125
        new_max = tl.maximum(running_max, tl.max(score, 1))
        denom = denom * tl.exp(running_max-new_max) + tl.sum(tl.exp(score-new_max[:, None]), 1)
        running_max = new_max
    av = tl.zeros((BM, 64), tl.float32)
    ak = tl.zeros((BM, 64), tl.float32)
    for block in range(first, last):
        ks = block * BN + ns
        score = tl.load(Scores + (head * T + qs[:, None]) * T + ks[None, :]).to(tl.float32) * 0.125
        prob = (tl.exp(score-running_max[:, None])/denom[:, None]).to(tl.bfloat16)
        key = tl.load(K + (head * T + ks[:, None]) * 64 + ds[None, :])
        val = tl.load(V + (head * T + ks[:, None]) * 64 + ds[None, :])
        av += tl.dot(prob, val)
        ak += tl.dot(prob, key)
    rows = (head * T + qs[:, None]) * KEYS + tl.arange(0, KEYS)[None, :]
    top = tl.load(Top + rows).to(tl.float32) * 0.125
    pp = tl.exp(top-running_max[:, None])/denom[:, None]
    tl.store(Probability + rows, pp)
    dest = (head * T + qs[:, None]) * 64 + ds[None, :]
    tl.store(OutV + dest, av.to(tl.bfloat16).to(tl.float32))
    tl.store(OutK + dest, ak.to(tl.bfloat16).to(tl.float32))


FAST_MOMENTS = True  # moment_kernels' implicit key context and query moments (subagent's tiles)
FAST_KEYS = True  # moment_kernels' key scan (larger tiles)
BATCHED_VALUES = True  # with OVERLAP_KEYS: raw value credits before the chain, terminal ones after, batched
OVERLAP_KEYS = True  # with batched_keys: key credits and query moments on a side stream under the value chain
TRITON_TOPK = False  # the replay's strong keys by a row top-k kernel (ties may resolve differently than torch.topk)
LEAN_STRONG_KEYS = True  # strong keys / values gathered contiguously, their centring in one compiled graph


COMPILED_DESCENT = False  # with BATCHED_VALUES and BATCHED_GATES: the descending chain's glue in one compiled graph


@torch.compile(dynamic=False)
def descending_chain(chains, raws, gates, xins, c_projs, c_vs, lam_r, length: int):
    """descending_key_values' chain without its attention kernels: each layer's output credit corrected by the
    deeper layers' measured input responses, the value input response through the norm (plus the gate's), and the
    carried extra for the layer below."""
    L = len(chains)
    corrected, inputs, extra = [None] * L, [None] * L, [None] * L
    extra[L - 1] = torch.zeros_like(chains[0])
    for i in reversed(range(L)):
        corrected[i] = ((chains[i] + extra[i]) @ c_projs[i]).view(1, length, 8, 64).transpose(1, 2)[0]
        ev = raws[i].transpose(0, 1).reshape(-1, chains[0].shape[-1])
        inputs[i] = norm_move(xins[i], ev @ c_vs[i])
        if gates[i] is not None:
            inputs[i] = inputs[i] + gates[i]
        if i:
            extra[i - 1] = lam_r[i] * (inputs[i] + extra[i])
    return corrected, inputs, extra


BATCHED_GATES = True  # with BATCHED_VALUES: the value gates' input credits of all layers before the chain
FLAT_GRADIENTS = True  # with STACKED_ANVIL, LEAN_FLAT and SHARDED_ANVIL: the estimate packs Cut.step's flat buffer


@torch.compile(dynamic=False)
def pack_gradients(gradients, divisor, bf16: bool):
    """Cut.step's flat gradient buffer, torch.cat(...) / divisor, and (bf16) its bf16 copy, in one compiled graph."""
    flat = torch.cat([g.float().reshape(-1) for g in gradients]) / divisor
    return flat, flat.to(torch.bfloat16) if bf16 else flat


CHAIN_GLUE = True  # chain_credits' weight casts / transposes and root projections in compiled graphs


@torch.compile(dynamic=False)
def chain_weights(fc_weights, proj_weights):
    """chain_credits' bf16 c_fc weights and transposed bf16 c_proj weights (each contiguous)."""
    return ([w.to(torch.bfloat16) for w in fc_weights],
            [w.to(torch.bfloat16).T.contiguous() for w in proj_weights])


@torch.compile(dynamic=False)
def chain_transposes(weights):
    """Each weight transposed, contiguous."""
    return [w.T.contiguous() for w in weights]


CAST_ONCE = True  # Cut.estimate casts every Linear weight to bf16 once (capture forward and chain_credits share them)


@torch.compile(dynamic=False)
def root_projections_of(fc_weights, frames):
    """Each root's bf16 c_fc rows projected on its frame (fp32, contiguous)."""
    return [(w.float() @ frame.T).contiguous() for w, frame in zip(fc_weights, frames)]


IMPLICIT_EDGES = True  # batched_key_contexts' lse and center masking only the key blocks the window cuts
FUSED_GLUE = True  # with batched_keys, OVERLAP_KEYS, BATCHED_VALUES and BATCHED_GATES: descending_key_values' credit
# kernels write the (layers, tokens, width) layout the reassembly reads, the chain and the path scalars in one graph


@torch.compile(dynamic=False)
def block_recompute(blocks, attn_outs, fc_weights, proj_weights, first: int, chain_sigma: float):
    """estimate's fp32 recompute of each block's mid stream, MLP pre-activations and output, and (from block first on)
    the stacked route scores (route_scores) for the batched MLP routes."""
    mids, hpres, outs, scores = [], [], [], []
    for i, (block, attn_out, w_fc, w_proj) in enumerate(zip(blocks, attn_outs, fc_weights, proj_weights)):
        mid = block.float().flatten(0, 1) + attn_out.float().flatten(0, 1)
        hpre = F.linear(norm(mid), w_fc.float())
        out = mid + F.linear(F.relu(hpre).square(), w_proj.float())
        mids.append(mid), hpres.append(hpre), outs.append(out)
        if i >= first:
            radius = chain_sigma * w_fc.float().norm(dim=1)
            scores.append(response_scores(hpre, radius, w_proj.float().norm(dim=0)))
    return mids, hpres, outs, torch.stack(scores)


@torch.compile(dynamic=False)
def weight_credits(chains, g_top, lam_r, extras, attn_proj_in, mlp_proj_in, fc_outs, fc_in, fc_out_weights,
                   sigma: float):
    """layer_weight_credits for every block at once: three batched GEMMs over the blocks' stacked credits and the
    capture's stacked inputs ((blocks, 1, tokens, width)). Returns (blocks, out, in) weight credits."""
    L = len(chains)
    d_out = [lam_r[i + 1] * chains[i + 1] for i in range(L - 1)] + [g_top]
    x_mid = torch.stack([chains[i] + extras[i] for i in range(L)])
    x_out = torch.stack([d_out[i] + extras[i] for i in range(L)])
    hidden = []
    for i in range(L):
        u = fc_outs[i].float().flatten(0, 1)
        steps = sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        hidden.append(hidden_local_credit(u, steps, d_out[i], fc_out_weights[i].float()))
    hidden = torch.stack(hidden)
    return (torch.bmm(x_mid.transpose(1, 2), attn_proj_in.flatten(1, 2)),
            torch.bmm(x_out.transpose(1, 2), mlp_proj_in.flatten(1, 2)),
            torch.bmm(hidden.transpose(1, 2), fc_in.flatten(1, 2)))


@torch.compile(dynamic=False)
def layer_weight_credits(d_mid, d_next, lam, extra, attn_proj_in, mlp_proj_in, fc_out, fc_in, fc_out_weight,
                         sigma: float):
    """estimate's per-block weight credits in one compiled graph: attn.c_proj from the mid credit, mlp.c_proj from the
    out credit (lam * the next block's mid credit, or the readout's with lam = 1) and mlp.c_fc through
    hidden_credit's finite step (fc_out the bf16 hidden pre-activations)."""
    d_out = lam * d_next
    x_mid, x_out = d_mid + extra, d_out + extra
    u = fc_out.float().flatten(0, 1)
    steps = sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    hidden = hidden_local_credit(u, steps, d_out, fc_out_weight.float())
    return (x_mid.T @ attn_proj_in.flatten(0, 1), x_out.T @ mlp_proj_in.flatten(0, 1),
            hidden.T @ fc_in.flatten(0, 1))


MERGED_PROBE_GEMMS = True  # probe_moment_terms' GEMMs that share an operand merged (bitwise the same)


@torch.compile(dynamic=False)
def probe_moment_terms(x, y, weight, H):
    """Cut.probe_moments_eager's computation (frame H drawn by the caller) in one compiled graph: the readout credit
    and the head moments."""
    C = H @ weight.T
    width = x.shape[-1]
    steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    base = x @ weight.T
    sq = x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    t = torch.tanh(base * inv0 / 15)
    f1 = 1 - t.square()
    prob = torch.softmax(15 * t, -1)
    c1, c2 = prob * f1, prob * (-(2 / 15) * t * f1 + f1.square()) / 2
    b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
    if MERGED_PROBE_GEMMS:  # products sharing an operand as one GEMM: [c1; c2 base] @ W, c2 @ [W^2 | C^2 T], .. @ H^T
        n = x.shape[0]
        first = torch.cat([c1, c2 * base]) @ weight
        mu1, nu = first[:n], first[n:]
        second = c2 @ torch.cat([weight.square(), C.square().T], 1)
        curvature, quad = second[:, :width].contiguous(), second[:, width:]
        mu1_h, nu_h = (first @ H.T).split(n)
    else:
        mu1, nu = c1 @ weight, (c2 * base) @ weight
        curvature, quad = readout_curvature_diagonal(weight, c2), c2 @ C.square().T
        mu1_h, nu_h = mu1 @ H.T, nu @ H.T
    moments = (b1, b2, mu1, nu, curvature)
    inv = torch.rsqrt((sq + 2 * steps * (x @ H.T) + steps.square() * width) / width + EPS)
    a = inv - inv0
    lin = a * b1.unsqueeze(1) + inv * steps * mu1_h
    change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * steps * nu_h
              + inv.square() * steps.square() * quad - 0.5 * lin.square())
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    zy = (base_y + steps * C.T[target]) * inv
    derivative = (change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(base_y * inv0 / 15))) / steps
    return (derivative * (y >= 0).unsqueeze(1)) @ H / width, moments


DESCENT_GRAPH = False  # with FUSED_GLUE: descending_inputs as one compiled graph (else per-layer compiled pieces)
LAYER_TAILS = False  # with FUSED_GLUE: the weight credits per block (layer_weight_credits) instead of batched


def descending_inputs_eager(raws, gates, blocks, c_vs, lam_r, x0, outs):
    """descending_chain's input responses and carried extras (corrected is formed by the caller), and
    path_scalar_credit's contractions: raws (tokens, width) per layer, blocks the bf16 block inputs."""
    L = len(raws)
    inputs, extra, path = [None] * L, [None] * L, [None] * L
    extra[L - 1] = torch.zeros_like(x0)
    for i in reversed(range(L)):
        inputs[i], carried, path[i] = chain_step(blocks[i], raws[i] @ c_vs[i], gates[i], extra[i], lam_r[i],
                                                 x0 if i == 0 else outs[i - 1], x0)
        if i:
            extra[i - 1] = carried
    return inputs, extra, torch.stack(path, dim=1)


@torch.compile(dynamic=False)
def chain_step(block, v, gate, extra, lam, previous, x0):
    """One layer of the descending chain: the value input response (+ the gate's), the extra carried down
    (lam * credit) and path_scalar_credit's two contractions of the credit (inputs + extra)."""
    inputs = norm_move(block.float().flatten(0, 1), v)
    if gate is not None:
        inputs = inputs + gate
    credit = inputs + extra
    return inputs, lam * credit, torch.stack([(credit * previous).sum(), (credit * x0).sum()])


descending_inputs_graph = torch.compile(descending_inputs_eager, dynamic=False)
LEAN_CHAIN_TAIL = True  # the chain credits' carry, mixing-scalar sums and x0 credit in one compiled graph


@torch.compile(dynamic=False)
def chain_tail(measured, g_top, lam_r, lam_0, x0, outs):
    """Each block's mid credit (measured, else carried from the next block by its residual scalar; the last block
    from the readout), the mixing scalars' contractions and the credit reaching x0."""
    L = len(measured)
    credits = [None] * L
    credits[L - 1] = measured[L - 1] if measured[L - 1] is not None else g_top
    for i in reversed(range(L - 1)):
        credits[i] = measured[i] if measured[i] is not None else lam_r[i + 1] * credits[i + 1]
    lams = [torch.stack([(credits[i] * (x0 if i == 0 else outs[i - 1])).sum(), (credits[i] * x0).sum()])
            for i in range(L)]
    x0_credit = lam_r[0] * credits[0] + sum(lam_0[i] * credits[i] for i in range(L))
    return credits, lams, x0_credit


LEAN_GROUP_CREDITS = True  # a replay group's credits from its loss changes in one compiled graph


@torch.compile(dynamic=False)
def group_credits(loss_change, steps, controls, frames, boost: float):
    """Each root's credit from its replay group's loss changes: the identity-path predictions of every root removed,
    the remainder read along the root's frame and added back to its own prediction."""
    residual = loss_change
    for step, control, frame in zip(steps, controls, frames):
        residual = residual - step * (control @ frame.T)
    return [control + boost * ((residual / step) @ frame / frame.shape[0])
            for step, control, frame in zip(steps, controls, frames)]


@torch.compile(dynamic=False)
def gate_error(gate_out, ve_out, credit, heads: int):
    """Cut.exact_values' value-gate error: the gate's central-difference slope times the raw value credit's
    contraction with the value embedding."""
    b, t, _ = gate_out.shape
    z = gate_out.float()
    ve = ve_out.float().view(b, t, heads, -1)
    slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
    return slope * (ve * credit).sum(-1)


@torch.compile(dynamic=False)
def strong_queries(x, wq, cos, sin, heads: int):
    """Cut.strong_keys' RoPE'd unit queries (tokens, heads, head_dim) in one compiled graph."""
    length = x.shape[0]
    width = wq.shape[0] // heads
    c, s = cos[:, None], sin[:, None]
    r = (norm(x) @ wq.T).view(length, heads, width)
    return norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                           r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))


@torch.compile(dynamic=False)
def own_query_tensors(k_out, v_out, gate_out, ve_out, cos, sin, heads: int):
    """Cut.own_query_inputs' RoPE'd unit keys and gated values of one layer, (heads, tokens, head_dim) bf16."""
    length = k_out.shape[0]
    width = k_out.shape[1] // heads
    k = k_out.float().view(length, heads, width)
    k1, k2 = k[..., :width // 2], k[..., width // 2:]
    c, s = cos[:, None], sin[:, None]
    k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
    v = v_out.float().view(length, heads, width)
    if gate_out is not None:
        v = v + 2 * torch.sigmoid(gate_out.float().view(length, heads, 1)) * ve_out.float().view(length, heads, width)
    return k.transpose(0, 1).to(torch.bfloat16).contiguous(), v.transpose(0, 1).to(torch.bfloat16).contiguous()


@torch.compile(dynamic=False)
def strong_key_tensors(kk, vv, idx, p, out, all_keys):
    """The top keys (shifted by the tail centroid) and values of each query, and the values' tail remainder."""
    heads, length, width = kk.shape
    count = idx.shape[-1]
    flat = idx.reshape(heads, length * count, 1).expand(-1, -1, width)
    ktop = kk.gather(1, flat).view(heads, length, count, width)
    vtop = vv.gather(1, flat).view(heads, length, count, width)
    if FUSED_GLUE:  # the weighted sums over the top keys as fused reductions (not batched gemvs)
        rest = out - (p.unsqueeze(-1) * vtop.float()).sum(-2)
        tail_keys = all_keys - (p.unsqueeze(-1) * ktop.float()).sum(-2)
    else:
        rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
        tail_keys = all_keys - torch.einsum('htk,htkd->htd', p, ktop.float())
    mass = (1 - p.sum(-1, keepdim=True)).clamp_min(0)
    centroid = torch.where(mass > 1e-5, tail_keys / mass.clamp_min(1e-5), 0)
    shifted = (ktop.float() - centroid.unsqueeze(-2)).to(kk.dtype)
    if CHAIN_GLUE:  # the shifted top keys token-major, (tokens, heads, keys, width), as the replay reads them
        return shifted.transpose(0, 1).contiguous(), vtop, rest
    return shifted, vtop, rest


@cache
def window_blocked(length, window, device):
    """The causal left-window mask's complement (True where a query may not see a key), built once."""
    positions = torch.arange(length, device=device)
    lag = positions[:, None] - positions[None, :]
    return ~((lag >= 0) & (lag <= window))


@triton.jit
def _row_topk_kernel(Scores, Tops, Indices, T: tl.constexpr, WINDOW: tl.constexpr, K: tl.constexpr):
    """Each query row's K largest scores inside its causal left window, largest first (repeated argmax)."""
    row = tl.program_id(0)
    query = row % T
    keys = tl.arange(0, T)
    allowed = (keys <= query) & (query - keys <= WINDOW)
    x = tl.load(Scores + row * T + keys, allowed, float('-inf')).to(tl.float32)
    for j in tl.static_range(K):
        best = tl.argmax(x, 0)
        tl.store(Tops + row * K + j, tl.max(x, 0).to(Tops.dtype.element_ty))
        tl.store(Indices + row * K + j, best.to(tl.int64))
        x = tl.where(keys == best, float('-inf'), x)


@triton.jit
def _window_fill_kernel(Scores, T: tl.constexpr, WINDOW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """-inf at every (query, key) outside the causal left window (scores (heads, T, T), written only there): tiles of
    BM queries x BN keys, fully blocked tiles by unmasked (vector) stores, edge tiles masked."""
    head, tile, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query = tile * BM + tl.arange(0, BM)
    keys = block * BN + tl.arange(0, BN)
    q0, q1, k0, k1 = tile * BM, tile * BM + BM - 1, block * BN, block * BN + BN - 1
    ptr = Scores + (head * T + query[:, None]).to(tl.int64) * T + keys[None, :]
    fill = tl.full((BM, BN), float('-inf'), Scores.dtype.element_ty)
    if (k0 > q1) | (q0 - k1 > WINDOW):  # every key after every query, or before every query's window
        tl.store(ptr, fill)
    elif (k1 > q0) | (q1 - k0 > WINDOW):  # an edge tile
        lag = query[:, None] - keys[None, :]
        tl.store(ptr, fill, (lag < 0) | (lag > WINDOW))


def window_fill_(scores, window):
    """scores.masked_fill_(window_blocked(...), -inf) writing only the blocked entries."""
    heads, length, _ = scores.shape
    assert scores.is_contiguous() and length % 64 == 0
    _window_fill_kernel[(heads, length // 32, length // 64)](scores, T=length, WINDOW=window, BM=32, BN=64,
                                                              num_warps=4)
    return scores


@triton.jit
def _ordered_bits(x):
    """fp32 -> int32 in the same order (-inf lowest, -0 below +0, as torch's radix select orders them)."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _route_topk_kernel(Scores, Indices, Scratch, N: tl.constexpr, K: tl.constexpr):
    """torch.topk(scores, K).indices of one row exactly (values descending, equal values by ascending index): the K-th
    largest value by binary search over its ordered bits, the values above it and the lowest-index ties, sorted."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    key = _ordered_bits(tl.load(Scores + row * N + cols).to(tl.float32))
    lo = tl.full((), -2147483648, tl.int64)
    hi = tl.full((), 2147483647, tl.int64)
    for _ in tl.static_range(32):  # the largest t with count(key >= t) >= K
        mid = (lo + hi + 1) // 2
        enough = tl.sum((key >= mid.to(tl.int32)).to(tl.int32), 0) >= K
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid - 1)
    kth = lo.to(tl.int32)
    equal = key == kth
    need = K - tl.sum((key > kth).to(tl.int32), 0)
    chosen = (key > kth) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= need))
    slot = tl.cumsum(chosen.to(tl.int32), 0) - 1
    tl.store(Scratch + row * K + slot, (key.to(tl.int64) << 32) | (N - 1 - cols).to(tl.int64), chosen)
    tl.debug_barrier()
    top = tl.sort(tl.load(Scratch + row * K + tl.arange(0, K)), descending=True)
    tl.store(Indices + row * K + tl.arange(0, K), N - 1 - (top & 0xFFFFFFFF))


def route_topk(scores, count):
    """scores.topk(count, dim=-1).indices for (..., N) fp32 scores (N a power of two), exactly."""
    n = scores.shape[-1]
    rows = scores.numel() // n
    indices = torch.empty((*scores.shape[:-1], count), device=scores.device, dtype=torch.int64)
    scratch = torch.empty((rows, count), device=scores.device, dtype=torch.int64)
    _route_topk_kernel[(rows,)](scores.contiguous(), indices, scratch, N=n, K=count, num_warps=2)
    return indices


def stream_route_moments(q, keys, values, window, count):
    heads, length, width = keys.shape
    if LEAN_STRONG_KEYS:  # (heads, query, key) straight from a batched matmul, no permuted copy
        scores = torch.bmm(q.to(keys.dtype).transpose(0, 1), keys.transpose(1, 2))
    else:
        scores = torch.einsum('thd,hsd->hts', q.to(keys.dtype), keys).contiguous()
    if FUSED_GLUE:  # the stream kernel reads it masked
        window_fill_(scores, window)
    else:
        scores.masked_fill_(window_blocked(length, window, q.device), float('-inf'))
    if TRITON_TOPK:
        tops = torch.empty((heads, length, count), device=q.device, dtype=scores.dtype)
        indices = torch.empty((heads, length, count), device=q.device, dtype=torch.int64)
        _row_topk_kernel[(heads * length,)](scores, tops, indices, T=length, WINDOW=window, K=count, num_warps=4)
    elif WINDOW_TOPK:  # torch.topk's selection and order, bitwise (bf16 rows, count <= 32)
        tops, indices = window_topk.window_topk(scores, count)
    else:
        tops, indices = scores.topk(count, dim=-1)
    probabilities = torch.empty((heads, length, count), device=q.device)
    out = torch.empty((heads, length, width), device=q.device)
    all_keys = torch.empty_like(out)
    _route_moment_stream_kernel[(length // 32, heads)](scores, keys, values, tops, probabilities, out, all_keys,
      T=length, WINDOW=window, KEYS=count, BM=32, BN=128, num_warps=4, num_stages=2)
    return probabilities, indices, out, all_keys


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    """PR #44's readout probe, verbatim: forward differences of each token's readout loss along every direction
    of a full cycle; the moved stream's RMS is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


class Cut:
    def __init__(self, model, settings, seed, rank, world):
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model, seed + 1000 * rank)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * rank)
        self.move_generator = torch.Generator(self.device).manual_seed(seed + 23 + 1000 * rank)
        self._estimate_graph = None
        self._telemetry_health = []
        self._query_moment_errors = {}
        self._side_stream = torch.cuda.Stream()
        global FUSED_CHAIN
        if FUSED_CHAIN != settings.fused_chain:
            FUSED_CHAIN = settings.fused_chain
            torch.compiler.reset()
        self._gate_rows = {i: torch.empty(self.width, self.width, device=self.device)
        for i in range(self.model.config.n_layer)
        if has_ve(i, self.model.config.n_layer)}
    def basis(self):
        signs = torch.randint(0, 2, (self.width,), device=self.device, generator=self.generator).float() * 2 - 1
        return hadamard(self.width, self.device, torch.float32) * signs
    def query_errors(self, draws, targets):
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L):
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            fixed, vals = kp.to(dt), values.to(dt)
            baseline = attention(processed(q_raw).to(dt), fixed, vals, window)
            a = d.query_noise(draws, b, t, cfg.n_embd).view(draws, b, t, h, hd)
            moved = moved_heads(q_raw, a, sigma, cos, sin)
            out = query_probe_attention(moved, fixed, vals, window, draws, b)
            estimate = code_estimate(head_scores(out, baseline, target), a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors
    def hidden_credit(self, u, g, out_weight):
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return hidden_local_credit(u, steps, g, out_weight.float())
    def batched_key_contexts(self, targets, defer=False):
        """key_context, the query moments and the key credits of every layer, the layers of one window size stacked
        along the head axis so that each kernel launches once per window size. Returns per-layer contexts and key
        credits (heads, tokens, head_dim); the query moment credits go to self._query_moment_errors."""
        d, model, s = self.capture, self.model, self.s
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        windows = {}
        for i in range(L):
            windows.setdefault(model.window_sizes[i][0], []).append(i)
        contexts, work = {}, []
        early = EARLY_MOMENT_INPUTS and self.fuses_glue()  # (read by batched_key_scores_by_token)
        self._moment_inputs = {}
        for window, layers in windows.items():
            pick = lambda name: [d.outputs[f'transformer.h.{i}.attn.{name}'][0] for i in layers]
            ves = [d.outputs[f'value_embeds.{i}'][0] if has_ve(i, L) else None for i in layers]
            gates = [d.outputs[f'transformer.h.{i}.attn.ve_gate'][0] if has_ve(i, L) else None for i in layers]
            if early:
                (qp, kp, k_raw, values, hub), self._moment_inputs[window] = key_context_group_moments(
                    pick('c_q'), pick('c_k'), pick('c_v'), ves, gates, [targets[i][0] for i in layers], cos, sin, h,
                    s.query_moment_sigma)
            else:
                qp, kp, k_raw, values, hub = key_context_group(pick('c_q'), pick('c_k'), pick('c_v'), ves, gates,
                                                               [targets[i][0] for i in layers], cos, sin, h)
            lse, center = ((moment_kernels.implicit_key_context_edges if IMPLICIT_EDGES
                            else moment_kernels.implicit_key_context) if FAST_MOMENTS else implicit_key_context)(
                qp, kp, values, hub, window)
            for j, i in enumerate(layers):
                rows = slice(j * h, (j + 1) * h)
                contexts[i] = (qp[rows], kp[rows], k_raw[rows], cos, sin, lse[rows],
                               (values[rows], hub[rows], center[rows]))
            work.append((window, layers, qp, kp, k_raw, values, hub, lse, center))
        if defer:
            return contexts, work
        return contexts, self.batched_key_scores(work)
    def batched_key_scores(self, work):
        """The key credits and query moments of batched_key_contexts' window groups."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys = {}
        for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
            if FAST_KEYS:
                key = moment_kernels.key_credits(qp, kp, k_raw, lse, values, hub, center, cos, sin, window,
                                                 s.key_sigma)
            else:
                key = torch.empty_like(k_raw)
                _key_scan_kernel[len(layers) * h, t // 32](
                    qp, kp, k_raw, lse, values, hub, center, cos, sin, key, T=t, D=hd, Window=window,
                    Sigma=s.key_sigma, BT=32, BS=64, J=32, PREC=KEY_DOT_PRECISION, num_warps=4, num_stages=2)
            if s.query_moments:
                q_out = torch.cat([d.outputs[f'transformer.h.{i}.attn.c_q'][0].float().view(t, h, hd).transpose(0, 1)
                                   for i in layers], 0)
                q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(
                    q_out.transpose(0, 1).reshape(t, -1), qp, kp, values, hub, s.query_moment_sigma)
                out = torch.empty(1, len(layers) * h, t, hd, device=self.device, dtype=torch.bfloat16)
                (moment_kernels.moment_query_scores if FAST_MOMENTS else moment_query_scores)(q16, k16, v16, hub16, q_raw, sq, kp.contiguous(), cos[:, :hd // 2].contiguous(),
                                    sin[:, :hd // 2].contiguous(), lse, center, steps, out, hd // 16, window=window)
            for j, i in enumerate(layers):
                rows = slice(j * h, (j + 1) * h)
                keys[i] = key[rows]
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = (
                        out[0, rows].float().transpose(0, 1).reshape(1, t, h * hd))
        return keys
    def batched_key_scores_by_token(self, work, qkv, maps):
        """batched_key_scores with each layer's query and key credits written into columns [0, width) and
        [width, 2 width) of the (layers, tokens, 3 width) buffer qkv (window group g's layers at maps[g]); returns the
        key credits as (tokens, width) views (the query credits go to self._query_moment_errors as
        (1, tokens, width) views)."""
        d, model, s = self.capture, self.model, self.s
        h, t = model.config.n_head, d.length
        hd = self.width // h
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        keys, w = {}, self.width
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            moment_kernels.key_credits_into(qp, kp, k_raw, lse, values, hub, center, cos, sin, window, s.key_sigma,
                                            qkv[:, :, w:2 * w], maps[g])
            if window in self._moment_inputs:
                q16, k16, v16, hub16, q_raw, sq, steps = self._moment_inputs[window]
            else:
                q_out = torch.cat([d.outputs[f'transformer.h.{i}.attn.c_q'][0] for i in layers], 1)
                q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(q_out, qp, kp, values, hub,
                                                                             s.query_moment_sigma)
            moment_kernels.moment_query_scores_into(
                q16, k16, v16, hub16, q_raw, sq, kp, cos[:, :hd // 2].contiguous(), sin[:, :hd // 2].contiguous(),
                lse, center, steps, qkv[:, :, :w], window=window, layers=maps[g])
            for i in layers:
                keys[i] = qkv[i, :, w:2 * w]
                self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = qkv[i, :, :w].unsqueeze(0)
        return keys
    def query_moment_credit(self, i, context):
        """Each raw query coordinate moved by a finite step through RoPE and the query norm, the attention's response
        to it from the degree-2 moment model (the identity phase's query credit, with this layer's window)."""
        qp, kp, _, cos, sin, lse, (values, hub, center) = context
        heads, t, hd = qp.shape
        q16, k16, v16, hub16, q_raw, sq, steps = query_moment_inputs(
            self.capture.outputs[f'transformer.h.{i}.attn.c_q'][0], qp, kp, values, hub, self.s.query_moment_sigma)
        out = torch.empty(1, heads, t, hd, device=self.device, dtype=torch.bfloat16)
        half = hd // 2
        moment_query_scores(q16, k16, v16, hub16, q_raw, sq, kp.contiguous(), cos[:, :half].contiguous(),
                            sin[:, :half].contiguous(), lse.contiguous(), center.contiguous(), steps, out, hd // 16,
                            window=self.model.window_sizes[i][0])
        return out[0].float().transpose(0, 1).reshape(1, t, heads * hd)
    def internals(self, gradients, targets, b, t, key_errors, raw_values, terminal_values):
        d, s = self.capture, self.s
        errors = self._query_moment_errors if s.query_moments else self.query_errors(s.query_draws, targets)
        errors.update(key_errors)
        for i in targets:
            errors.update(self.exact_values(i, raw_values[i], terminal_values[i]))
        fused = self.fuses_glue()
        if fused:  # every layer's query, key and value weight credits from one batched GEMM of the side-by-side errors
            weight = torch.bmm(self._qkv.transpose(1, 2), d.inputs['stack:attn.c_q'].flatten(1, 2))
            for i in targets:
                p = f'transformer.h.{i}.attn.'
                for k, name in enumerate(('c_q', 'c_k', 'c_v')):
                    gradients[p + name + '.weight'] = weight[i, k * self.width:(k + 1) * self.width]
        for site, error in errors.items():
            if not (fused and site.endswith(('attn.c_q', 'attn.c_k', 'attn.c_v'))):
                gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
    def bind_flat_layout(self, optimizer):
        """Cut.step's flat gradient layout (ANVIL's shape groups first, in its batch order). Bound before the estimate
        graph's capture, it lets the estimate pack its gradients there (FLAT_GRADIENTS)."""
        params = list(self.model.named_parameters())
        anvil_ids = [id(q) for batch in optimizer.optimizers[0].batches for q in batch[0]]
        rank_of = {pid: k for k, pid in enumerate(anvil_ids)}
        self._flat_order = sorted(params, key=lambda item: rank_of.get(id(item[1]), len(anvil_ids)))
        self._packs = FLAT_GRADIENTS and STACKED_ANVIL and LEAN_FLAT and SHARDED_ANVIL and not DELAYED_UPDATE
    def packs_gradients(self):
        return getattr(self, '_packs', False)
    def step(self, x, y, optimizer):
        self.prepare_gate_rows()
        if self._estimate_graph is None:
            gradients = self.estimate(x, y)
        else:
            self._static_x.copy_(x)
            self._static_y.copy_(y)
            self._estimate_graph.replay()
            gradients = self._static_gradients
        if DELAYED_UPDATE:  # screen: apply the previous estimate (the val cost of overlapping update and estimate)
            current = {name: g.clone() for name, g in gradients.items()}
            gradients = getattr(self, '_previous_gradients', current)
            self._previous_gradients = current
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        params = list(self.model.named_parameters())
        packed = getattr(self, '_packed', None) if self.packs_gradients() else None
        if STACKED_ANVIL:  # ANVIL's shape groups contiguous in the flat buffer, so their stacked grads are views
            if not hasattr(self, '_flat_order'):
                self.bind_flat_layout(optimizer)
            params = self._flat_order
            if packed is not None:  # the estimate already formed the flat buffer (and its bf16 copy)
                flat = packed[0]
            else:
                flat = torch.cat([gradients[name].float().reshape(-1) for name, _ in params]).div_(
                    valid * self.world if LEAN_FLAT else valid)
        else:
            flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if SHARDED_ANVIL and STACKED_ANVIL and LEAN_FLAT:  # ANVIL's grads reduce-scattered, the rest all-reduced
            anvil = optimizer.optimizers[0]
            sizes = [sum(q.numel() for q in batch[0]) for batch in anvil.batches]
            wire = packed[1] if packed is not None else (  # COMMS_BF16: half the bytes on the wire
                flat.to(torch.bfloat16) if COMMS_BF16 else flat)
            # ASYNC_COMMS: ANVIL's shards first, each batch's update waits only for its own shard, and the rest's
            # all-reduce runs under ANVIL's update
            anvil.shard_grads, anvil.shard_works, offset = [], [], 0
            for (batch_params, *_), size in zip(anvil.batches, sizes):
                full = wire[offset:offset + size].view(len(batch_params), *batch_params[0].shape)
                shard = torch.empty((len(batch_params) // self.world, *batch_params[0].shape), device=flat.device,
                                    dtype=wire.dtype)
                anvil.shard_works.append(torch.distributed.reduce_scatter_tensor(shard, full, async_op=ASYNC_COMMS))
                anvil.shard_grads.append(shard)
                offset += size
            rest = torch.distributed.all_reduce(wire[sum(sizes):], async_op=ASYNC_COMMS)
            if ASYNC_COMMS:
                anvil_ids = {id(q) for batch in anvil.batches for q in batch[0]}

                def finish_rest():  # between ANVIL's and the other optimizers' steps
                    rest.wait()
                    if COMMS_BF16 and LEAN_REST:
                        # bf16 params' grads are views of the reduced wire (what .to(bf16) of its fp32 copy gives),
                        # fp32 params' slices copied once; the finite check reads the bf16 wire
                        self._telemetry_health.append(torch.isfinite(torch.sum(wire, dtype=torch.float32)))
                        offset = 0
                        for _, p in params:
                            if id(p) not in anvil_ids:
                                part = slice(offset, offset + p.numel())
                                if p.dtype == wire.dtype:
                                    p.grad = wire[part].view_as(p)
                                else:
                                    flat[part] = wire[part]
                                    p.grad = flat[part].view_as(p).to(p.dtype)
                            offset += p.numel()
                        return
                    if COMMS_BF16:
                        flat[sum(sizes):] = wire[sum(sizes):]
                    self._telemetry_health.append(torch.isfinite(flat.sum()))
                    offset = 0
                    for _, p in params:
                        if id(p) not in anvil_ids:
                            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
                        offset += p.numel()
                optimizer.step(between=finish_rest)
                for work in anvil.gather_works:
                    work.wait()
                anvil.gather_works = []
                diagnostic_loss = torch.where(y != -1, self.capture.clean_loss, 0).sum() / valid
                return diagnostic_loss if LAZY_LOSS else average_ranks(diagnostic_loss.reshape(1)).reshape(())
            if COMMS_BF16:
                flat[sum(sizes):] = wire[sum(sizes):]
        elif ALLREDUCE_BF16:
            half = flat.to(torch.bfloat16)
            torch.distributed.all_reduce(half)
            flat = half.float()
        else:
            torch.distributed.all_reduce(flat)
        if LEAN_FLAT and STACKED_ANVIL:  # pre-divided by the world size; one reduction for the finite check
            finite = torch.isfinite(flat.sum())
        else:
            flat /= self.world
            finite = torch.isfinite(flat).all()
        self._telemetry_health.append(finite)
        offset = 0
        for _, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        diagnostic_loss = torch.where(y != -1, self.capture.clean_loss, 0).sum() / valid
        return diagnostic_loss if LAZY_LOSS else average_ranks(diagnostic_loss.reshape(1)).reshape(())
    def fused_descending_key_values(self, chain, targets, x0, outs):
        """descending_key_values (batched keys on the side stream, batched values and gates) with the credits in the
        layout their consumers read, and the chain's glue and path_scalar_credit in one compiled graph. Returns its
        outputs and the path scalars."""
        d, s, model = self.capture, self.s, self.model
        L, h, t = model.config.n_layer, model.config.n_head, d.length
        hd = self.width // h
        errors, raw_values, terminal_values = {}, {}, {}
        contexts, work = self.batched_key_contexts(targets, defer=True)
        # every layer's query, key and value credits by token, side by side: one batched reassembly GEMM
        qkv = torch.empty(L, t, 3 * self.width, device=self.device)
        maps = [self.layer_map(item[1]) for item in work]
        main, side = torch.cuda.current_stream(), self._side_stream
        side.wait_stream(main)
        qkv.record_stream(side)
        for item in work:
            for z in item[2:]:
                z.record_stream(side)
        for inputs in self._moment_inputs.values():
            for z in inputs:
                z.record_stream(side)
        with torch.cuda.stream(side):
            keys = self.batched_key_scores_by_token(work, qkv, maps)
        raws, place = {}, {}
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            raw = moment_kernels.raw_value_credits_by_token(qp, kp, lse, hub, window, h)
            for j, i in enumerate(layers):
                raws[i], place[i] = raw[j], (g, j)
        ve_layers = [i for i in range(L) if has_ve(i, L)]
        if STACKED_GATES:
            gates = dict(zip(ve_layers, stacked_gate_input_responses(
                [raws[i] for i in ve_layers], [d.outputs[f'value_embeds.{i}'][0] for i in ve_layers],
                [d.blocks[i] for i in ve_layers], [model.transformer.h[i].attn.ve_gate.weight for i in ve_layers],
                [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        else:
            kappas = [(raws[i].view(t, h, hd) * d.outputs[f'value_embeds.{i}'].float().view(t, h, hd)).sum(-1)
                      for i in ve_layers]
            gates = dict(zip(ve_layers, gate_input_responses(
                [d.blocks[i].float().flatten(0, 1) for i in ve_layers],
                [model.transformer.h[i].attn.ve_gate.weight.float() for i in ve_layers], kappas,
                [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        input_list, extra_list, path = (descending_inputs_graph if DESCENT_GRAPH else descending_inputs_eager)(
            [raws[i] for i in range(L)], [gates.get(i) for i in range(L)], [d.blocks[i] for i in range(L)],
            [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)], model.resid_lambdas.float(), x0,
            outs)
        corrected = [torch.empty(len(item[1]), t, self.width, device=self.device) for item in work]
        for i in range(L):  # corrected by token, straight into the terminal pass's input
            g, j = place[i]
            torch.mm(chain[i] + extra_list[i], model.transformer.h[i].attn.c_proj.weight.float(), out=corrected[g][j])
        w = self.width
        for g, (window, layers, qp, kp, k_raw, values, hub, lse, center) in enumerate(work):
            moment_kernels.terminal_value_credits_into(qp, kp, lse, corrected[g], window, qkv[:, :, 2 * w:], maps[g])
            for i in layers:
                terminal_values[i] = qkv[i, :, 2 * w:].view(1, t, h, hd)
        self._qkv = qkv
        for i in range(L):
            raw_values[i] = raws[i].view(1, t, h, hd)
        main.wait_stream(side)
        for i in range(L):
            keys[i].record_stream(main)
            errors[f'transformer.h.{i}.attn.c_k'] = keys[i].unsqueeze(0)
            self._query_moment_errors[f'transformer.h.{i}.attn.c_q'].record_stream(main)
        return (errors, raw_values, terminal_values, dict(enumerate(extra_list)), dict(enumerate(input_list)),
                path)
    def layer_map(self, layers):
        """The layers' indices as an int32 device tensor (made once, outside graph capture)."""
        if not hasattr(self, '_layer_maps'):
            self._layer_maps = {}
        key = tuple(layers)
        if key not in self._layer_maps:
            self._layer_maps[key] = torch.tensor(key, device=self.device, dtype=torch.int32)
        return self._layer_maps[key]
    def fuses_glue(self):
        s = self.s
        return (FUSED_GLUE and s.batched_keys and s.query_moments and OVERLAP_KEYS and BATCHED_VALUES and BATCHED_GATES
                and FAST_KEYS and FAST_MOMENTS)
    def descending_key_values(self, chain, targets):
        """Topologically schedule independent raw probes; corrected outputs have terminal consumers only."""
        d,s,model=self.capture,self.s,self.model
        L=model.config.n_layer
        errors,raw_values,terminal_values,inputs={},{},{},{}
        extra={L-1:torch.zeros_like(chain[0])}
        if s.batched_keys and OVERLAP_KEYS:  # key credits and query moments on a side stream, under the value chain
            contexts, work = self.batched_key_contexts(targets, defer=True)
            main, side = torch.cuda.current_stream(), self._side_stream
            side.wait_stream(main)
            for item in work:
                for z in item[2:]:
                    z.record_stream(side)
            with torch.cuda.stream(side):
                keys = self.batched_key_scores(work)
        elif s.batched_keys:
            contexts, keys = self.batched_key_contexts(targets)
        deferred_keys = s.batched_keys and OVERLAP_KEYS
        batched_values = deferred_keys and BATCHED_VALUES
        if batched_values:  # raw value credits need no chain: one pass per window group up front
            raw_pre, corrected_all = {}, {}
            for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
                raw_group = moment_kernels.raw_value_credits(qp, kp, lse, hub, window)
                for j, i in enumerate(layers):
                    raw_pre[i] = raw_group[j * model.config.n_head:(j + 1) * model.config.n_head]
            if BATCHED_GATES:  # the value gates' input credits read only raw values: all layers at once
                ve_layers = [i for i in range(L) if has_ve(i, L)]
                heads = model.config.n_head
                kappas = []
                for i in ve_layers:
                    value = raw_pre[i].transpose(0, 1).reshape(-1, heads, self.width // heads)
                    kappas.append((value * d.outputs[f'value_embeds.{i}'].float().reshape_as(value)).sum(-1))
                gate_pre = dict(zip(ve_layers, gate_input_responses(
                    [d.blocks[i].float().flatten(0, 1) for i in ve_layers],
                    [model.transformer.h[i].attn.ve_gate.weight.float() for i in ve_layers], kappas,
                    [self._gate_rows[i] for i in ve_layers], s.gate_path_sigma)))
        if batched_values and BATCHED_GATES and COMPILED_DESCENT:  # the whole chain as one compiled graph
            corrected_list, input_list, extra_list = descending_chain(
                [chain[i] for i in range(L)], [raw_pre[i] for i in range(L)],
                [gate_pre.get(i) for i in range(L)], [d.blocks[i].float().flatten(0, 1) for i in range(L)],
                [model.transformer.h[i].attn.c_proj.weight.float() for i in range(L)],
                [model.transformer.h[i].attn.c_v.weight.float() for i in range(L)],
                model.resid_lambdas.float(), d.length)
            for i in range(L):
                corrected_all[i] = corrected_list[i]
                inputs[i] = input_list[i]
                extra[i] = extra_list[i]
                raw_values[i] = raw_pre[i].transpose(0, 1).unsqueeze(0)
        for i in (reversed(range(L)) if not (batched_values and BATCHED_GATES and COMPILED_DESCENT) else ()):
            block=model.transformer.h[i]
            # extra[i] contains only deeper raw independently measured input responses.
            corrected=((chain[i]+extra[i])@block.attn.c_proj.weight.float()).view(1,d.length,8,64).transpose(1,2)[0]
            if batched_values:
                raw, terminal, key = raw_pre[i], None, None
                corrected_all[i] = corrected
            elif s.batched_keys:
                context = contexts[i]
                raw, terminal = value_credits(context, corrected, model.window_sizes[i][0])
                key = None if deferred_keys else keys[i]
            else:
                context=self.key_context(i,targets[i])
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'] = self.query_moment_credit(i, context)
                key,raw,terminal=key_and_value_credits(context,corrected,s.key_sigma,model.window_sizes[i][0])
            if key is not None:
                errors[f'transformer.h.{i}.attn.c_k']=key.transpose(0,1).reshape(1,d.length,self.width)
            raw_values[i]=raw.transpose(0,1).unsqueeze(0)
            if terminal is not None:
                terminal_values[i]=terminal.transpose(0,1).unsqueeze(0)
            ev=raw_values[i].reshape(-1,self.width)
            xin=d.blocks[i].float().flatten(0,1)
            # Only original raw values enter either nonlinear finite-response branch.
            inputs[i]=norm_move(xin,ev@block.attn.c_v.weight.float())
            if has_ve(i,L):
                inputs[i] = inputs[i] + (gate_pre[i] if batched_values and BATCHED_GATES
                                         else self.gate_input_credit(i, raw_values[i], self._gate_rows[i]))
            if i: extra[i-1]=model.resid_lambdas[i].float()*(inputs[i]+extra[i])
        if batched_values:  # terminal value credits after the chain: one pass per window group
            h = model.config.n_head
            for window, layers, qp, kp, k_raw, values, hub, lse, center in work:
                terminal_group = moment_kernels.terminal_value_credits(
                    qp, kp, lse, torch.cat([corrected_all[i] for i in layers], 0), window)
                for j, i in enumerate(layers):
                    terminal_values[i] = terminal_group[j * h:(j + 1) * h].transpose(0, 1).unsqueeze(0)
        if deferred_keys:
            main.wait_stream(side)
            for i in range(L):
                keys[i].record_stream(main)
                errors[f'transformer.h.{i}.attn.c_k']=keys[i].transpose(0,1).reshape(1,d.length,self.width)
                if s.query_moments:
                    self._query_moment_errors[f'transformer.h.{i}.attn.c_q'].record_stream(main)
        return errors,raw_values,terminal_values,extra,inputs
    def exact_values(self, i, credit, value_credit):
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer) and FUSED_GLUE:
            errors[prefix + 've_gate'] = gate_error(d.outputs[prefix + 've_gate'], d.outputs[f'value_embeds.{i}'],
                                                    credit, h)
        elif has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, cfg.n_embd // h)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors
    def key_context(self, i, target):
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        if self.s.compiled_glue:
            prefix = f'transformer.h.{i}.attn.'
            ve = has_ve(i, model.config.n_layer)
            qp, kp, k_raw, values, hub = key_context_tensors(
                d.outputs[prefix + 'c_q'][0], d.outputs[prefix + 'c_k'][0], d.outputs[prefix + 'c_v'][0],
                d.outputs[f'value_embeds.{i}'][0] if ve else None, d.outputs[prefix + 've_gate'][0] if ve else None,
                target[0], cos, sin, h)
            lse, center = implicit_key_context(qp, kp, values, hub, model.window_sizes[i][0])
            return qp, kp, k_raw, cos, sin, lse, (values, hub, center)
        rot = lambda z: torch.cat([z[..., :hd//2]*cos + z[..., hd//2:]*sin,
        -z[..., :hd//2]*sin + z[..., hd//2:]*cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix+name][0].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][0].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix+'ve_gate'][0].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        hub = target[0].view(t, h, hd).transpose(0, 1)
        lse, center = implicit_key_context(qp, kp, values, hub, model.window_sizes[i][0])
        return qp, kp, k_raw, cos, sin, lse, (values, hub, center)
    def probe_moments(self, x, y):
        if self.fuses_glue():
            derivative, self._head_moments = probe_moment_terms(x, y, self.model.lm_head.weight.float(), self.basis())
            return derivative
        return self.probe_moments_eager(x, y)
    def probe_moments_eager(self, x, y):
        """probe() with each move's readout change from its second-order expansion in the moved logits: the
        softcapped log-sum-exp to second order, its quadratic term exact over the whole vocabulary for every frame
        direction (c2 @ (C * C)^T, C the head's rows in the frame), the moved norm and the target logit exact. Also
        keeps the head moments for the chain replays (self._head_moments)."""
        weight = self.model.lm_head.weight.float()
        H = self.basis()
        C = H @ weight.T  # (direction, vocab): each head row's component along each frame direction
        width = self.width
        steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # #44 sigma
        base = x @ weight.T
        sq = x.square().sum(-1, keepdim=True)
        inv0 = torch.rsqrt(sq / width + EPS)
        t = torch.tanh(base * inv0 / 15)
        f1 = 1 - t.square()
        prob = torch.softmax(15 * t, -1)
        c1, c2 = prob * f1, prob * (-(2 / 15) * t * f1 + f1.square()) / 2
        b1, b2 = (c1 * base).sum(-1), (c2 * base.square()).sum(-1)
        mu1, nu = c1 @ weight, (c2 * base) @ weight
        self._head_moments = (b1, b2, mu1, nu, readout_curvature_diagonal(weight, c2))
        inv = torch.rsqrt((sq + 2 * steps * (x @ H.T) + steps.square() * width) / width + EPS)
        a = inv - inv0
        lin = a * b1.unsqueeze(1) + inv * steps * (mu1 @ H.T)
        quad = c2 @ C.square().T
        change = (lin + a.square() * b2.unsqueeze(1) + 2 * a * inv * steps * (nu @ H.T)
                  + inv.square() * steps.square() * quad - 0.5 * lin.square())
        target = y.clamp(min=0)
        base_y = base.gather(1, target.unsqueeze(1))
        zy = (base_y + steps * C.T[target]) * inv
        derivative = (change - (15 * torch.tanh(zy / 15) - 15 * torch.tanh(base_y * inv0 / 15))) / steps
        return (derivative * (y >= 0).unsqueeze(1)) @ H / width
    def probe(self, x, y):
        """Readout credit: PR #44's Cut probe, forward differences along a full Hadamard cycle."""
        weight = self.model.lm_head.weight.float()
        H = self.basis()
        stream_dirs = H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        steps = 0.005 * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # #44 sigma
        n, vocab = x.shape[0], C.shape[1]
        credit = torch.empty(n, self.width, device=self.device)
        per = max(1, 134217728 // (self.width * vocab))  # #44 chunk
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ H / self.width
        return credit
    def close(self):
        if self._estimate_graph is not None:
            torch.cuda.synchronize()
            self._estimate_graph = None
            self._static_gradients = None
            self._static_x = self._static_y = None
            self._gate_rows.clear()
        self.capture.close()
    def prepare_gate_rows(self):
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        # Skip the frames the next estimate draws from self.generator: the readout probe's, then one per chain root.
        for _ in range(1 + len(self.s.chain_depth_roots)):
            torch.randint(0, 2, (self.width,), device=self.device, generator=generator)
        frame = hadamard(self.width, self.device, torch.float32)
        for rows in self._gate_rows.values():
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            rows.copy_(frame[indices] * signs)
    @torch.no_grad()
    def capture_estimate_graph(self, x, y):
        self._static_x, self._static_y = x.clone(), y.clone()
        generators = [self.generator, self.move_generator, self.capture.shared_generator, self.capture.generator]
        states = [g.get_state() for g in generators]
        default_state = torch.cuda.get_rng_state()
        def restore_rng():
            for generator, state in zip(generators, states):
                generator.set_state(state)
            torch.cuda.set_rng_state(default_state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.prepare_gate_rows()
                self.estimate(self._static_x, self._static_y)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        for generator in generators:
            graph.register_generator_state(generator)
        torch._C._cuda_clearCublasWorkspaces()
        with torch.cuda.graph(graph):
            self._static_gradients = self.estimate(self._static_x, self._static_y)
        self._estimate_graph = graph
        restore_rng()
        self.prepare_gate_rows()
        graph.replay()
        torch.cuda.synchronize()
        restore_rng()
    def gate_input_credit(self, i, raw_value, rows):
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        value = raw_value.reshape(-1, heads, self.width // heads)
        ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
        kappa = (value * ve).sum(-1)
        xin = d.blocks[i].float().flatten(0, 1)
        weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
        projected = F.linear(rows[:, :weight.shape[1]], weight)
        row_square = rows.square().mean(-1)
        result = torch.empty_like(xin)
        for t0 in range(0, len(xin), s.chain_tokens):
            sl = slice(t0, t0 + s.chain_tokens)
            result[sl] = prepared_gate_input_response(
                xin[sl], weight, kappa[sl], rows, projected, row_square, s.gate_path_sigma)
        return result
    @staticmethod
    def path_scalar_credit(inputs, extra, x0, outs):
        credits = []
        for i in range(len(outs)):
            credit = inputs[i]+extra[i]
            previous = x0 if i == 0 else outs[i-1]
            credits.append(torch.stack([(credit*previous).sum(), (credit*x0).sum()]))
        return torch.stack(credits, dim=1)
    def chain_credits(self, e, x0, mids, outs, routes, y, g_top):
        model, s = self.model, self.s
        L = model.config.n_layer
        weight = model.lm_head.weight.float()
        if CAST_ONCE and CHAIN_GLUE:  # the bf16 weights cast_weights made; c_proj's transposed once more
            names = [f'transformer.h.{i}.mlp.' for i in range(L)]
            w_fc = [self._bf16_weights[p + 'c_fc'] for p in names]
            w_out_t = chain_transposes([self._bf16_weights[p + 'c_proj'] for p in names])
        elif CHAIN_GLUE:
            w_fc, w_out_t = chain_weights([blk.mlp.c_fc.weight for blk in model.transformer.h],
                                          [blk.mlp.c_proj.weight for blk in model.transformer.h])
        else:
            w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        moments = self._head_moments if s.probe_moments else head_moments(xL, base, weight)
        n, m = e.shape[0], s.chain_tokens
        by_site = {}
        for site in s.chain_depth_roots:
            H = self.basis()
            if s.chain_moves < self.width:
                rows = torch.randperm(self.width, device=self.device, generator=self.move_generator)
                H = H[rows[:s.chain_moves]].contiguous()
            steps = s.chain_sigma * mids[site].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            by_site[site] = (H, steps, torch.empty(n, self.width, device=self.device))
        signed_steps = {}
        controls = {}
        for group in s.joint_root_groups:
            for site in group:
                steps = by_site[site][1]
                sign = (torch.randint(0, 2, (n, 1), device=self.device) * 2 - 1).float()
                signed_steps[site] = steps * sign
                factor = lam_r[site + 1:].prod()
                controls[site] = g_top * factor
        if CHAIN_GLUE:
            roots = [group[0] for group in s.joint_root_groups]
            root_projections = dict(zip(roots, root_projections_of([w_fc[r] for r in roots],
                                                                   [by_site[r][0] for r in roots])))
        else:
            root_projections = {group[0]: (w_fc[group[0]].float() @ by_site[group[0]][0].T).contiguous()
            for group in s.joint_root_groups}
        replay_layers = lambda group: (attention_replay_layers(group, L, s.query_replay_long_layers)
                                       if s.replay_groups is None or group[0] in s.replay_groups else ())
        active_query_layers = tuple(sorted({b for group in s.joint_root_groups for b in replay_layers(group)}))
        own = self.own_query_inputs(active_query_layers)
        own = self.strong_keys(own, s.query_keys)
        boost = 1.0
        chunks = [slice(t0, t0 + m) for t0 in range(0, n, m)]
        if s.chain_fraction < 1:  # replay a random subset of the tokens; the rest keep their identity-path control
            count = int(n * s.chain_fraction)
            chunks = [torch.randperm(n, device=self.device, generator=self.move_generator)[:count].sort().values]
            boost = n / count
            for site, (_, _, result) in by_site.items():
                result.copy_(controls[site])
        for sl in chunks:
            own_sl = (own[0][sl], [None if z is None else z[sl] for z in own[1]], own[2]) + tuple(
                [None if z is None else z[sl] for z in part] for part in own[3:6]) + own[6:]
            for group in s.joint_root_groups:
                site = group[0]
                H = by_site[site][0]
                steps = signed_steps[site][sl]
                injections = tuple((other, signed_steps[other][sl], by_site[other][0])
          for other in group[1:])
                loss_change = chain_chunk(
                    site, (steps, H), [z[sl] for z in mids], xL[sl], [None if r[0] is None else r[0][sl] for r in routes],
                    [None if r[1] is None else r[1][sl] for r in routes], lam_r, w_fc, w_out_t, base[sl], target_w[sl], y[sl], own_sl,
                    moments=tuple(z[sl] for z in moments),
                    root_projection=root_projections[site], injections=injections,
                    query_layers=replay_layers(group),
                )
                if LEAN_GROUP_CREDITS:
                    news = group_credits(loss_change, [signed_steps[o][sl] for o in group],
                                         [controls[o][sl] for o in group], [by_site[o][0] for o in group], boost)
                    for other, new in zip(group, news):
                        by_site[other][2][sl] = new
                    continue
                residual = loss_change
                for other in group:
                    frame = by_site[other][0]
                    residual = residual - signed_steps[other][sl] * (controls[other][sl] @ frame.T)
                for other in group:
                    frame, _, result = by_site[other]
                    result[sl] = controls[other][sl] + boost * ((residual / signed_steps[other][sl]) @ frame
                                                                / frame.shape[0])
        del own_sl, own
        credits = {site: credit for site, (_, _, credit) in by_site.items()}
        if LEAN_CHAIN_TAIL:
            measured = [credits.get(i) for i in range(L)]
            mids_c, lams, x0_credit = chain_tail(measured, g_top, lam_r, lam_0, x0, outs)
            credits = dict(enumerate(mids_c))
            for i in range(L):
                credits['lam', i] = lams[i]
            credits['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
                                               hadamard(self.width, self.device, torch.float32))
            return credits
        if L - 1 not in credits:  # last block unmeasured: its mid credit is the readout credit
            credits[L - 1] = g_top
        for i in reversed(range(L - 1)):
            if i not in credits:
                credits[i] = lam_r[i + 1] * credits[i + 1]
        for i in range(L):
            previous = x0 if i == 0 else outs[i - 1]
            credits['lam', i] = torch.stack([(credits[i] * previous).sum(),
          (credits[i] * x0).sum()])
        # x0 = norm(e) enters every block input (x_in_i = lam_r[i] * prev_i + lam_0[i] * x0), so each block's mid
        # credit (measured at a root, else carried from the next block) reaches x0 along its linear skip; one finite
        # step through the norm follows.
        x0_credit = lam_r[0] * credits[0] + sum(lam_0[i] * credits[i] for i in range(L))
        credits['emb'] = norm_input_credit(e, x0_credit, s.chain_sigma,
      hadamard(self.width, self.device, torch.float32))
        return credits
    def own_query_inputs(self, active_layers):
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            if i not in active_layers:
                inputs.append(None), w_q.append(None), keys.append(None), values.append(None)
                w_proj_t.append(None), windows.append(model.window_sizes[i][0])
                continue
            prefix = f'transformer.h.{i}.attn.'
            if FUSED_GLUE:
                gated = block.attn.ve_gate is not None
                k, v = own_query_tensors(d.outputs[prefix + 'c_k'][0], d.outputs[prefix + 'c_v'][0],
                                         d.outputs[prefix + 've_gate'][0] if gated else None,
                                         d.outputs[f'value_embeds.{i}'][0] if gated else None, cos, sin, heads)
                inputs.append(d.blocks[i][0].float())
                w_q.append(block.attn.c_q.weight.float())
                keys.append(k)
                values.append(v)
                w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
                windows.append(model.window_sizes[i][0])
                continue
            k = d.outputs[prefix + 'c_k'][0].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            c, s = cos[:, None], sin[:, None]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][0].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][0].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][0].float().view(length, heads, width)
            inputs.append(d.blocks[i][0].float())
            w_q.append(block.attn.c_q.weight.float())
            keys.append(k.transpose(0, 1).to(torch.bfloat16).contiguous())
            values.append(v.transpose(0, 1).to(torch.bfloat16).contiguous())
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            windows.append(model.window_sizes[i][0])
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)
    @staticmethod
    def strong_keys(own, k):
        pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
        length = len(pos)
        top_keys, top_values, top_p = [], [], []
        for x, wq, kk, vv, window, wp in zip(inputs, w_q, keys, values, windows, w_proj_t):
            if x is None:
                top_keys.append(None), top_values.append(None), top_p.append(None)
                continue
            heads, _, width = kk.shape
            q = strong_queries(x, wq, cos, sin, heads) if FUSED_GLUE else None
            if q is None:
                c, s = cos[:, None], sin[:, None]
                r = (norm(x) @ wq.T).view(length, heads, width)
                q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
              r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            p, idx, out, all_keys = stream_route_moments(q, kk, vv, window, k)
            if LEAN_STRONG_KEYS:
                ktop, vtop, rest = strong_key_tensors(kk, vv, idx, p, out, all_keys)
            else:
                gather = lambda z: z.unsqueeze(1).expand(-1, length, -1, -1).gather(
              2, idx.unsqueeze(-1).expand(-1, -1, -1, width))
                ktop, vtop = gather(kk), gather(vv)
                rest = out - torch.einsum('htk,htkd->htd', p, vtop.float())
                mass = (1 - p.sum(-1, keepdim=True)).clamp_min(0)
                tail_keys = all_keys - torch.einsum('htk,htkd->htd', p, ktop.float())
                centroid = torch.where(mass > 1e-5, tail_keys / mass.clamp_min(1e-5), 0)
                ktop = (ktop.float() - centroid.unsqueeze(-2)).to(kk.dtype)
            top_keys.append(ktop if LEAN_STRONG_KEYS and CHAIN_GLUE else ktop.transpose(0, 1).contiguous())
            if TOKEN_MAJOR_VALUES:
                top_values.append(project_routed_values_token_major(vtop, p, rest, wp))
            else:
                vtop = preproject_routed_values(vtop, p, rest, wp)
                top_values.append(vtop.transpose(0, 1).contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
        return (pos, inputs, w_q, top_keys, top_values, top_p, (cos, sin))
    def route_scores(self, layer, hpre):
        mlp = self.model.transformer.h[layer].mlp
        radius = self.s.chain_sigma * mlp.c_fc.weight.float().norm(dim=1)
        return response_scores(hpre, radius, mlp.c_proj.weight.float().norm(dim=0))
    def select_mlp_routes(self, layer, hpre):
        mlp = self.model.transformer.h[layer].mlp
        weight = mlp.c_proj.weight.float()
        radius = self.s.chain_sigma * mlp.c_fc.weight.float().norm(dim=1)
        plus = F.relu(hpre + radius).square()
        minus = F.relu(hpre - radius).square()
        response = (plus - minus).abs()
        return (response * weight.norm(dim=0)).topk(self.s.route_k, dim=-1).indices
    @torch.no_grad()
    def cast_weights(self):
        """CAST_ONCE: every Linear's bf16 weight in persistent buffers, by one multi-tensor copy (the capture forward
        and chain_credits read them instead of casting their own)."""
        if not hasattr(self, '_bf16_weights'):
            linears = {n: m.weight for n, m in self.capture.modules.items() if isinstance(m, nn.Linear)}
            assert all(m.bias is None for m in self.capture.modules.values() if isinstance(m, nn.Linear))
            self._bf16_weights = {}
            if QKV_GEMM:  # each block's c_q / c_k / c_v buffers: row blocks of one (3 C, C) buffer, 'attn.qkv'
                for i in range(self.model.config.n_layer):
                    p = f'transformer.h.{i}.attn.'
                    parts = [linears[p + n] for n in ('c_q', 'c_k', 'c_v')]
                    cat = torch.empty((sum(w.shape[0] for w in parts), parts[0].shape[1]), device=parts[0].device,
                                      dtype=torch.bfloat16)
                    self._bf16_weights[p + 'qkv'] = cat
                    for n, rows in zip(('c_q', 'c_k', 'c_v'), cat.split([w.shape[0] for w in parts])):
                        self._bf16_weights[p + n] = rows
            for n, w in linears.items():
                if n not in self._bf16_weights:
                    self._bf16_weights[n] = torch.empty_like(w, dtype=torch.bfloat16)
            self._bf16_targets = [self._bf16_weights[n] for n in linears]
            self._fp32_weights = list(linears.values())
        torch._foreach_copy_(self._bf16_targets, self._fp32_weights)
        self.capture.weights = self._bf16_weights
    def estimate(self, x, y):
        d, model = self.capture, self.model
        if CAST_ONCE:
            self.cast_weights()
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        first = min(self.s.chain_depth_roots)
        if self.fuses_glue() and self.s.batched_routes:
            mids, routes, outs, scores = block_recompute(
                [d.blocks[i] for i in range(L)], [d.outputs[f'transformer.h.{i}.attn.c_proj'] for i in range(L)],
                [blk.mlp.c_fc.weight for blk in model.transformer.h],
                [blk.mlp.c_proj.weight for blk in model.transformer.h], first, self.s.chain_sigma)
        for i in (range(L) if not (self.fuses_glue() and self.s.batched_routes) else ()):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            if self.s.batched_routes:
                mids.append(mid), outs.append(out), routes.append(hpre)
                continue
            top = self.select_mlp_routes(i, hpre)
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        if self.s.batched_routes:  # one top-k over every replayed block's scores
            hpres = routes
            if not self.fuses_glue():
                scores = torch.stack([self.route_scores(i, h) for i, h in enumerate(hpres) if i >= first])
            count = self.s.route_k
            if self.fuses_glue() and not self.s.route_schedule and self.s.route_k_late and first >= self.s.route_split:
                count = self.s.route_k_late  # every scored block keeps only its strongest route_k_late
            tops = route_topk(scores, count) if self.fuses_glue() else scores.topk(count, dim=-1).indices
            tops = [None] * first + list(tops)
            if self.s.route_schedule:  # per block: the strongest route_schedule[i] of them
                tops = [None if top is None else top[:, :k].contiguous() for top, k in zip(tops, self.s.route_schedule)]
            elif self.s.route_k_late:  # the strongest route_k_late of them from block route_split on
                tops = [top if top is None or i < self.s.route_split else top[:, :self.s.route_k_late].contiguous()
                        for i, top in enumerate(tops)]
            routes = [(None, None) if top is None else (top, h.gather(1, top)) for top, h in zip(tops, hpres)]
        g_top = (self.probe_moments if self.s.probe_moments else self.probe)(outs[-1], yf)
        chain = self.chain_credits(e, x0, mids, outs, routes, yf, g_top)
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
        for i, block in enumerate(model.transformer.h)}
        if self.fuses_glue():
            key_errors, raw_values, terminal_values, extra, inputs, path = self.fused_descending_key_values(
                chain, targets, x0, outs)
        else:
            key_errors,raw_values,terminal_values,extra,inputs=self.descending_key_values(chain,targets)
            path = None
        scalars = torch.empty(2, L, device=self.device)
        if self.fuses_glue() and not LAYER_TAILS:  # every block's weight credits from three batched GEMMs
            blocks = model.transformer.h
            attn_proj, mlp_proj, fc = weight_credits(
                [chain[i] for i in range(L)], g_top, lam_r, [extra[i] for i in range(L)],
                d.inputs['stack:attn.c_proj'], d.inputs['stack:mlp.c_proj'],
                [d.outputs[f'transformer.h.{i}.mlp.c_fc'] for i in range(L)], d.inputs['stack:mlp.c_fc'],
                [blk.mlp.c_proj.weight for blk in blocks], self.s.fc_sigma)
            for i in range(L):
                p = f'transformer.h.{i}.'
                gradients[p + 'attn.c_proj.weight'], gradients[p + 'mlp.c_proj.weight'] = attn_proj[i], mlp_proj[i]
                gradients[p + 'mlp.c_fc.weight'] = fc[i]
                scalars[:, i] = chain['lam', i]
        for i in (range(L) if self.fuses_glue() and LAYER_TAILS else ()):
            p = f'transformer.h.{i}.'
            last = i == L - 1
            lam = torch.ones((), device=self.device) if last else lam_r[i + 1]
            (gradients[p + 'attn.c_proj.weight'], gradients[p + 'mlp.c_proj.weight'],
             gradients[p + 'mlp.c_fc.weight']) = layer_weight_credits(
                chain[i], g_top if last else chain[i + 1], lam, extra[i], d.inputs[p + 'attn.c_proj'],
                d.inputs[p + 'mlp.c_proj'], d.outputs[p + 'mlp.c_fc'],
                d.inputs[p + 'mlp.c_fc'], model.transformer.h[i].mlp.c_proj.weight, self.s.fc_sigma)
            scalars[:, i] = chain['lam', i]
        for i in (range(L) if not self.fuses_glue() else ()):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_mid = chain[i]
            if i == L - 1:
                d_out = g_top
            else:
                d_out = lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = d_mid + extra[i], d_out + extra[i]
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            scalars[:, i] = chain['lam', i]
        scalars += self.path_scalar_credit(inputs, extra, x0, outs) if path is None else path
        self.internals(gradients, targets, b, t, key_errors, raw_values, terminal_values)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        if self.packs_gradients():  # Cut.step's flat buffer (and bf16 copy) formed here, inside the estimate graph
            self._packed = pack_gradients([gradients[name] for name, _ in self._flat_order],
                                          (y != -1).sum() * self.world, COMMS_BF16)
        return gradients


MAPS = (
  (3.923798038567, -6.095026865488, 3.905234618423),
  (3.278126713798, -3.328923386476, 0.989127286973),
  (3.505298394150, -5.137358782410, 1.968325560615),
  (2.815058591845, -3.685181239622, 1.417196497642),
  (2.245503932403, -2.443826979899, 0.963091710461),
  (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)


def newton_schulz(x):
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    R, Q = x @ x.transpose(1, 2), eye
    for t, (a, b, c) in enumerate(MAPS):
        if t in restarts:
            x = Q @ x
            R, Q = x @ x.transpose(1, 2), eye
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return Q @ x


def polar_batch(g, margin):
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
    ratio: float, margin: float):
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


@torch.compile(dynamic=False)
def anvil_update_merged(gs, ws, fasts, slows, energies, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd,
                        lanes: tuple, ratios: tuple, margin: float):
    """anvil_update for several batches whose polar inputs share a (non-square) shape after polar_batch's transpose:
    their Gram Newton-Schulz iterations run as one batched polar."""
    inputs, talls = [], []
    for g, fast, slow in zip(gs, fasts, slows):
        fast.lerp_(g, 1 - fast_beta)
        slow.lerp_(g, 1 - slow_beta)
        x = g.lerp(weight * fast + (1 - weight) * slow, beta)
        talls.append(x.size(1) > x.size(2))
        x = x.transpose(1, 2) if talls[-1] else x
        inputs.append(x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6))
    polar = gram_newton_schulz(torch.cat(inputs)).float().split([len(g) for g in gs])
    news = []
    for update, tall, w, slow, energy, lane, ratio in zip(polar, talls, ws, slows, energies, lanes, ratios):
        update = update.transpose(1, 2) if tall else update
        energy.lerp_(update.square().mean(lane + 1, keepdim=True), 1 - lane_beta)
        gained = update * energy.clamp_min(1e-10).rsqrt()
        update = gained * (update.norm(dim=(1, 2), keepdim=True)
                           / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
        decay = (slow * w >= 0) * w * (wd * lr)
        news.append(w - lr * ratio * (update + decay))
    return news


MERGED_POLAR = True  # with SHARDED_ANVIL: ANVIL batches with the same polar shape (c_fc^T, c_proj) in one polar
FUSED_ADAM = True  # AdamW's fused CUDA kernel instead of the foreach path
LAZY_LOSS = False  # the logged Cut loss averaged over ranks once per telemetry flush, not per update
ASYNC_COMMS = True  # with SHARDED_ANVIL: gradient collectives async, waited where first read
COMMS_BF16 = True  # with SHARDED_ANVIL: the gradient reduce-scatter and all-reduce in bf16
LEAN_REST = True  # with ASYNC_COMMS and COMMS_BF16: Cut.step's non-ANVIL grads from the reduced bf16 wire (no full copy)
DELAYED_UPDATE = False  # screen only: each Cut update applies the previous estimate
LEAN_FLAT = True  # with STACKED_ANVIL: divide by tokens x world before the all-reduce, check finiteness by a sum
ALLREDUCE_BF16 = False  # the update's gradient all-reduce in bf16
SHARDED_ANVIL = True  # with STACKED_ANVIL and LEAN_FLAT: each rank updates 1/world of every ANVIL batch
STACKED_ANVIL = True  # ANVIL's weights and grads as views of stacked storage (no per-update stacking copies)


def stacked_view(tensors):
    """The (n, *shape) tensor whose rows are `tensors`, as a view when they are consecutive in memory, else a copy."""
    first = tensors[0]
    n = first.numel()
    if all(t.is_contiguous() and t.data_ptr() == first.data_ptr() + i * n * t.element_size()
           for i, t in enumerate(tensors)):
        return first.as_strided((len(tensors), *first.shape), (n, *first.stride()))
    return torch.stack(tensors)


class BatchedAnvil(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, steps, polar_margin, fast_weight):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.polar_margin, self.fast_weight = polar_margin, fast_weight
        self.t = 0

    def prepare_scalar_schedule(self, warmdown, final, decay_factors, warmup=0):
        group = self.param_groups[0]
        device = group['params'][0].device
        rows = []
        for t in range(self.total):
            lo, hi, cool = .85, .93, self.total - self.cooldown
            beta = lo + (hi - lo) * t / self.warmup if t < self.warmup else hi
            if t > cool:
                beta = hi - (hi - lo) * (t - cool) / self.cooldown
            fast_beta, weight = (.85, self.fast_weight) if t >= self.engage else (beta, 1.0)
            left = (self.total - (t + 1) + 1) / self.total
            factor = (final + (1 - final) * min(1.0, left / warmdown)) * (min(1.0, (t + 1) / warmup) if warmup else 1.0)
            rows.append([beta, fast_beta, .98, weight, .9,
                group['lr'] * factor, group['weight_decay'] * decay_factors[t]])
        self.scalar_schedule = torch.tensor(rows, dtype=torch.float32, device=device)
        by_shape = {}
        for p in group['params']:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        self.batches = []
        for shape, params in by_shape.items():
            lanes = 1 if shape[0] >= shape[1] else 0
            if STACKED_ANVIL:  # the group's weights become views of one stacked tensor
                stacked = torch.stack([p.data for p in params])
                for i, p in enumerate(params):
                    p.data = stacked[i]
            fast = torch.zeros(len(params), *shape, device=device)
            state = dict(fast=fast, slow=torch.zeros_like(fast),
                energy=torch.zeros_like(fast.mean(lanes + 1, keepdim=True)))
            self.batches.append((params, state, lanes, max(1, shape[0] / shape[1]) ** .5))

    def polar_partners(self):
        """Groups of batch indices whose polar inputs (polar_batch's transpose of tall shapes) share a non-square
        shape: MERGED_POLAR runs each group's polar as one."""
        groups = {}
        for k, (params, *_) in enumerate(self.batches):
            rows, cols = params[0].shape
            if rows != cols:
                groups.setdefault((min(rows, cols), max(rows, cols)), []).append(k)
        return [ks for ks in groups.values() if len(ks) > 1]

    def merged_shard_step(self, ks, scalars):
        """The sharded step of batches ks (polar partners) with one polar: wait each shard, update, all-gather."""
        world, rank = dist.get_world_size(), dist.get_rank()
        gs, ws, views, fasts, slows, energies, lanes, ratios = [], [], [], [], [], [], [], []
        for k in ks:
            params, state, lane, ratio = self.batches[k]
            per = len(params) // world
            sl = slice(rank * per, (rank + 1) * per)
            w = stacked_view([p.data for p in params])
            if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                self.shard_works[k].wait()
            gs.append(self.shard_grads[k].float())
            ws.append(w[sl].float())
            views.append(w)
            fasts.append(state['fast'][sl])
            slows.append(state['slow'][sl])
            energies.append(state['energy'][sl])
            lanes.append(lane)
            ratios.append(ratio)
        news = anvil_update_merged(gs, ws, fasts, slows, energies, *scalars, tuple(lanes), tuple(ratios),
                                   self.polar_margin)
        for w, new in zip(views, news):
            work = dist.all_gather_into_tensor(w, new.to(w.dtype).contiguous(), async_op=ASYNC_COMMS)
            if work is not None:
                self.gather_works = getattr(self, 'gather_works', []) + [work]

    @torch.no_grad()
    def step(self):
        scalars = self.scalar_schedule[self.t].unbind()
        merged = {}
        if MERGED_POLAR and SHARDED_ANVIL and getattr(self, 'shard_grads', None) is not None:
            for ks in self.polar_partners():
                merged.update({k: ks for k in ks})
        for k, (params, state, lanes, ratio) in enumerate(self.batches):
            if k in merged:
                if k == merged[k][-1]:  # one polar for the whole group, at its last batch
                    self.merged_shard_step(merged[k], scalars)
                continue
            if SHARDED_ANVIL and getattr(self, 'shard_grads', None) is not None:
                world, rank = dist.get_world_size(), dist.get_rank()
                per = len(params) // world
                sl = slice(rank * per, (rank + 1) * per)
                w = stacked_view([p.data for p in params])
                if getattr(self, 'shard_works', None) and self.shard_works[k] is not None:
                    self.shard_works[k].wait()
                new = anvil_update(self.shard_grads[k].float(), w[sl].float(), state['fast'][sl], state['slow'][sl],
                                   state['energy'][sl], *scalars, lanes, ratio, self.polar_margin)
                work = dist.all_gather_into_tensor(w, new.to(w.dtype).contiguous(), async_op=ASYNC_COMMS)
                if work is not None:
                    self.gather_works = getattr(self, 'gather_works', []) + [work]
                continue
            if STACKED_ANVIL:
                w, g = stacked_view([p.data for p in params]), stacked_view([p.grad for p in params])
                new = anvil_update(g.float(), w.float(), state['fast'], state['slow'], state['energy'], *scalars,
                                   lanes, ratio, self.polar_margin)
                if w.data_ptr() == params[0].data.data_ptr() and w._base is not None:
                    w.copy_(new)
                else:
                    torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
                continue
            new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                state['fast'], state['slow'], state['energy'], *scalars, lanes, ratio, self.polar_margin)
            torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.shard_grads = None
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
    def zero_grad(self, set_to_none):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
    def step(self, between=None):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for k, o in enumerate(self.optimizers):
                if k == 1 and between is not None:
                    between()
                o.step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_model(device, seed, config):
    # Only the random-token warmup reads these init values: the identity handoff overwrites every weight of the timed
    # model. The seed and the init draws stay because they set the default CUDA generator's state, which the Cut
    # estimate's per-token root signs (chain_credits) draw from.
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        nn.init.normal_(model.lm_head.weight, std=0.02)
    return model.requires_grad_(False)


def make_optimizer(model, recipe, steps):
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = (recipe.emb_scale if ('wte' in name or 'value_embeds' in name)
        else recipe.lambda_scale if 'lambdas' in name else 1)
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
      [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
      betas=recipe.betas, eps=1e-8, weight_decay=0.0, fused=FUSED_ADAM or None,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
      polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    # ANVIL weight decay is scaled by late_cut_decay_multiplier (zero) after the first 275/1043 of the Cut updates.
    late_from = round(275 / 1043 * steps)
    decay_factors = tuple(recipe.late_cut_decay_multiplier if t >= late_from else 1.0 for t in range(steps))
    anvil.prepare_scalar_schedule(recipe.warmdown, recipe.final, decay_factors, recipe.cut_warmup)
    return Optimizers(anvil, adam)


def set_rate(optimizer, step, steps, warmdown, final, warmup=0):
    left = (steps - step + 1) / steps
    ramp = min(1.0, step / warmup) if warmup else 1.0
    for group in optimizer.optimizers[1].param_groups:  # AdamW; ANVIL's rates are precomputed in its scalar schedule
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * ramp * (final + (1 - final) * min(1.0, left / warmdown))


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 64
MERGE_FROM, MERGE_EVERY = 0.3, 5
MERGE_DECAYS = (0.03, 0.02)
MERGE_RHOS = (1.0,)
MERGE_K, MERGE_TOKENS = 64, 512


MERGE_SKETCH_COLUMNS = 128


@torch.no_grad()
def merge_sketch_frame(width, columns, device, generator):
    raw = torch.randn(width, columns, device=device,
      generator=generator, dtype=torch.float32)
    return torch.linalg.qr(raw, mode='reduced')[0].contiguous()


@torch.no_grad()
def merge_sketch_add(x, omega):
    with torch.autocast('cuda', enabled=False):
        xf = x.reshape(-1, x.shape[-1]).float()
        return (xf.T @ (xf @ omega)).double()


@torch.no_grad()
def merge_nystrom_basis(y, omega, k):
    middle = omega.double().T @ y
    eigen, vectors = torch.linalg.eigh((middle + middle.T) * .5)
    floor = eigen[-1].clamp_min(1e-30) * 1e-6
    inverse = torch.where(eigen > floor, eigen.clamp_min(floor).rsqrt(), 0.)
    factor = y @ (vectors * inverse.unsqueeze(0))
    q, r = torch.linalg.qr(factor, mode='reduced')
    _, u = torch.linalg.eigh(r @ r.T)
    return (q @ u[:, -k:].flip(1)).float().contiguous()


@torch.no_grad()
def curvature_bases(model, rows, rank, k=MERGE_K):
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
      if isinstance(mod, nn.Linear) and 've_gate' not in name]
    bases = {key: torch.zeros(mod.weight.shape[1], k, device=device) for key, mod in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(SEED)
        frames = {key: merge_sketch_frame(mod.weight.shape[1], MERGE_SKETCH_COLUMNS,
        device, generator) for key, mod in layers}
        acc = {key: torch.zeros_like(omega, dtype=torch.float64) for key, omega in frames.items()}
        hooks = []
        for key, mod in layers:
            def hook(module, inputs, output, key=key):
                acc[key].add_(merge_sketch_add(inputs[0], frames[key]))
            hooks.append(mod.register_forward_hook(hook))
        counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)
        for batch in rows.split(4):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            counts += torch.bincount(batch[:, :-1].reshape(-1).long(), minlength=len(counts)).double()
        for h in hooks:
            h.remove()
        for key, _ in layers:
            bases[key] = merge_nystrom_basis(acc[key], frames[key], k)
        order = torch.argsort(counts, descending=True)
    for key, _ in layers:
        torch.distributed.broadcast(bases[key], 0)
    torch.distributed.broadcast(order, 0)
    return bases, order


def subspace_move(name, move, bases, order):
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


MERGE_COEFS = (0.75, 1.0)


@torch.no_grad()
def joint_parts(named, params, final, move, candidates, plain, scorer, settings, sweeps=2, shortlist=8):
    parts = sorted({merge_part(n) for n, _ in named})
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in MERGE_COEFS]
    cache = {}
    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]
    members = {}
    for i, (n, _) in enumerate(named):
        members.setdefault(merge_part(n), []).append(i)
    held = dict.fromkeys(parts)
    def show(choice):
        for p in parts:
            if held[p] != choice[p]:
                for i in members[p]:
                    params[i].copy_(value(i, choice[p]).to(params[i].dtype))
                held[p] = choice[p]
    def loss_for(trial, count):
        show(trial)
        return scorer.score(count)
    best_key, best, global_losses, _ = race_merge_keys(
      keys, None, plain, lambda key, count: loss_for(dict.fromkeys(parts, key), count),
      settings.merge_screen_rows, HOLDOUT, settings.merge_screen_global)
    choice = dict.fromkeys(parts, best_key)
    keys = [None] + [k for _, k in sorted(global_losses, key=lambda x: x[0]) if k is not None][:settings.merge_keys]
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            incumbent = choice[p]
            candidates_p = keys if sweep == 0 else ranked[p]
            def loss_of_key(key, count):
                trial = dict(choice)
                trial[p] = key
                return loss_for(trial, count)
            key, complete_loss, _, proxy_order = race_merge_keys(
          candidates_p, incumbent, best, loss_of_key, settings.merge_screen_rows,
          HOLDOUT, settings.merge_screen_coordinate)
            choice[p], best = key, complete_loss
            if sweep == 0:
                ranked[p] = proxy_order[:shortlist]
                if key not in ranked[p]:
                    ranked[p].append(key)
    show(choice)
    return dict(holdout_plain=plain, holdout_merged=best,
      merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]])
      for p, k in choice.items()})


def holdout_sum(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, y, loss_reduction='sum')


holdout_sum_captured = torch.compile(holdout_sum, dynamic=False)


def score_merge_keys(keys, loss_of_key, count):
    # Each score is an owned scalar clone, recorded before the next candidate writes.
    losses = torch.stack([loss_of_key(key, count) for key in keys]).cpu().tolist()
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError('Non-finite merge scores')
    return list(zip(losses, keys))


def race_merge_keys(keys, incumbent, incumbent_full_loss, loss_of_key, screen_rows, full_rows, keep):
    screened = score_merge_keys(keys, loss_of_key, screen_rows)
    ranked = [key for _, key in sorted(screened, key=lambda item: item[0])]
    finalists = ranked[:keep]
    if incumbent not in finalists:
        finalists.append(incumbent)
    measured = dict((key, loss) for loss, key in score_merge_keys(
      [key for key in finalists if key != incumbent], loss_of_key, full_rows))
    complete = [(incumbent_full_loss if key == incumbent else measured[key], key) for key in finalists]
    best_key, best_loss = incumbent, incumbent_full_loss
    for loss, key in complete:
        if loss < best_loss:
            best_key, best_loss = key, loss
    return best_key, best_loss, complete, ranked


class ResidentMergeScore:
    """Held-out loss on two fixed row cohorts (the screening rows and all held-out rows, split across ranks). Each
    cohort's forward and all-reduce are captured once as a CUDA graph on random tokens before the timer; bind()
    loads the held-out rows into the graph inputs."""
    def __init__(self, model, rank, world, sequence_len, screen_rows):
        self.model, self.rank, self.world = model, rank, world
        self.device = next(model.parameters()).device
        self.buffers, self.graphs = {}, {}
        for count in (screen_rows, HOLDOUT):
            shape = (count // world, sequence_len)
            x = torch.zeros(shape, device=self.device, dtype=torch.long)
            y = torch.zeros_like(x)
            total = torch.zeros(2, device=self.device, dtype=torch.float64)
            result = torch.zeros((), device=self.device, dtype=torch.float64)
            self.buffers[count] = (x, y, total, result)

    def bind(self, rows):
        for count, (x, y, _, _) in self.buffers.items():
            batch = rows[:count][self.rank::self.world].to(self.device)
            x.copy_(batch[:, :-1])
            y.copy_(batch[:, 1:])

    def execute(self, count):
        x, y, total, result = self.buffers[count]
        total.zero_()
        loss = holdout_sum_captured(self.model, x, y)
        total[0].add_(loss.double())
        total[1].add_(y.numel())
        dist.all_reduce(total)
        result.copy_(total[0] / total[1])

    def capture(self):
        for count in self.buffers:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.execute(count)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            dist.barrier()
            torch._C._cuda_clearCublasWorkspaces()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                self.execute(count)
            self.graphs[count] = graph
        torch.cuda.synchronize()

    def score(self, count):
        self.graphs[count].replay()
        return self.buffers[count][3].clone()

    def close(self):
        torch.cuda.synchronize()
        self.graphs.clear()


@torch.no_grad()
def prepare_resident_merge_score(model, settings, rank, world):
    noise = torch.randint(0, settings.model.vocab_size, (HOLDOUT, settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(SEED + 270))
    curvature_bases(model, noise, rank)  # random tokens; discarded
    scorer = ResidentMergeScore(model, rank, world, settings.model.sequence_len, settings.merge_screen_rows)
    scorer.bind(noise)
    scorer.capture()
    return scorer


@torch.no_grad()
def end_merge(model, means, rows, rank, scorer, settings):
    scorer.bind(rows)
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows, rank) if any(rho < 1 for rho in MERGE_RHOS) else ({}, None)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]
    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)
    plain = scorer.score(HOLDOUT).item()
    return joint_parts(named, params, final, move, candidates, plain, scorer, settings, sweeps=settings.merge_sweeps)


@triton.jit
def _side_features(side_ptr, offs, part, D: tl.constexpr, J: tl.constexpr, offs_j, scale):
    i = (part * J + offs_j) % (D // 2)
    x = tl.load(side_ptr + offs[:, None] * D + i[None, :]) * scale
    y = tl.load(side_ptr + offs[:, None] * D + (i + D // 2)[None, :]) * scale
    return x, y


@triton.jit
def _closed_coefficients(raw_ptr, sq_ptr, steps_ptr, cos_ptr, sin_ptr, row, BH, T, offs, part, M, RAW_GPU,
    RAW_STRIDE, D: tl.constexpr, J: tl.constexpr, offs_j):
    d = part * J + offs_j
    i = d % (D // 2)
    n = (row % BH).to(tl.int64) * T + offs
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
    n = (row % BH).to(tl.int64) * T + offs
    tl.store(out_ptr + (n // M * W_STRIDE + n % M * D + part * J)[:, None] + offs_j[None, :], x)


@triton.jit
def _split4(acc, BLOCK: tl.constexpr, J: tl.constexpr):
    first, second = tl.split(tl.reshape(acc, (BLOCK, J, 2, 2)))
    x, y = tl.split(first)
    xx, yy = tl.split(second)
    return x, y, xx, yy


@triton.jit
def _moment_queries_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr,
    hubout_ptr, steps_ptr, out_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
    RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, WINDOW: tl.constexpr):
    i, s0 = tl.program_id(0), (tl.num_programs(1) - 1 - tl.program_id(1)) * BLOCK_S
    row, part = i // REPEAT, i % REPEAT
    blk = row // BH
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
    side = side_ptr + base
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
    last = (s0 + BLOCK_S - 1) // BLOCK_T + 1
    for it in range(tl.maximum(s0 - WINDOW, 0) // BLOCK_T, last):
        offs_t = it * BLOCK_T + tl.arange(0, BLOCK_T)
        k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
        logits = tl.dot(q, tl.trans(k)) * scale
        lag = offs_s[:, None] - offs_t[None, :]
        a = tl.where((lag >= 0) & (lag <= WINDOW), tl.exp2((logits - lse[:, None]) * 1.4426950408889634), 0.0)
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
      RAW_GPU, RAW_STRIDE, D, J, offs_j)
    nx0, ny0, nx1, ny1 = _split4(n4, BLOCK_S, J)
    dx0, dy0, dx1, dy1 = _split4(d4, BLOCK_S, J)
    lx0, ly0 = tl.split(tl.reshape(l2, (BLOCK_S, J, 2)))
    mx0, my0 = tl.split(tl.reshape(m2, (BLOCK_S, J, 2)))
    zn = (nx0 * bu + ny0 * bw + 0.5 * (nx1 * bu * bu + ny1 * bw * bw) + nxy * bu * bw
      + ca * (lx0 * bu + ly0 * bw) + s1n[:, None] * ca + s2n[:, None] * (0.5 * ca * ca))
    zd = (dx0 * bu + dy0 * bw + 0.5 * (dx1 * bu * bu + dy1 * bw * bw) + dxy * bu * bw
      + ca * (mx0 * bu + my0 * bw) + s1d[:, None] * ca + s2d[:, None] * (0.5 * ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_s)
    _store_by_token(out_ptr, zn / (1.0 + zd) / steps[:, None], row, BH, T, offs_s, part, M, W_STRIDE, D, J, offs_j)


def moment_query_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, repeat, window=None):
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_queries_kernel[(rows * repeat, t // 64)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps,
      out, rows // blocks, t, d ** -0.5, repeat,
      raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
      out.stride(1), out.stride(0), D=d, J=d // repeat,
      BLOCK_T=32, BLOCK_S=64, WINDOW=t if window is None else min(window, t), num_warps=4, num_stages=3)


@triton.jit
def _moment_keys_kernel(q_ptr, k_ptr, v_ptr, hub_ptr, raw_ptr, sq_ptr, side_ptr, cos_ptr, sin_ptr, lse_ptr, hubout_ptr,
    steps_ptr, out_ptr, value_ptr, BH, T, scale, REPEAT, RAW_BLK, RAW_GPU,
    RAW_STRIDE, M, OUT_BLK, W_STRIDE,
    D: tl.constexpr, J: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    i, t0 = tl.program_id(0), tl.program_id(1) * BLOCK_T
    row, part = i // REPEAT, i % REPEAT
    blk = row // BH
    raw_ptr += blk.to(tl.int64) * RAW_BLK
    out_ptr += blk.to(tl.int64) * OUT_BLK
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    offs_j = tl.arange(0, J)
    base = row.to(tl.int64) * T * D
    k = tl.load(k_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_t[:, None] * D + offs_d[None, :])
    side = side_ptr + base
    p2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    q2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    qxy = tl.zeros((BLOCK_T, J), tl.float32)
    r2 = tl.zeros((BLOCK_T, 2 * J), tl.float32)
    val = tl.zeros((BLOCK_T, J), tl.float32)
    s1 = tl.zeros((BLOCK_T,), tl.float32)
    s2 = tl.zeros((BLOCK_T,), tl.float32)
    first = t0 // BLOCK_S
    last = T // BLOCK_S
    for it in range(first, last):
        offs_s = it * BLOCK_S + tl.arange(0, BLOCK_S)
        q = tl.load(q_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub = tl.load(hub_ptr + base + offs_s[:, None] * D + offs_d[None, :])
        hub_part = tl.load(hub_ptr + base + offs_s[:, None] * D + part * J + offs_j[None, :])
        lse = tl.load(lse_ptr + row * T + offs_s)
        hub_out = tl.load(hubout_ptr + row * T + offs_s)
        logits = tl.dot(k, tl.trans(q)) * scale
        lag = offs_s[None, :] - offs_t[:, None]
        a = tl.where(lag >= 0, tl.exp2((logits - lse[None, :]) * 1.4426950408889634), 0.0)
        c = tl.dot(v, tl.trans(hub)) - hub_out[None, :]
        x, y = _side_features(side, offs_s, part, D, J, offs_j, scale)
        w1 = a * c
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
      RAW_GPU, RAW_STRIDE, D, J, offs_j)
    px, py = tl.split(tl.reshape(p2, (BLOCK_T, J, 2)))
    qx, qy = tl.split(tl.reshape(q2, (BLOCK_T, J, 2)))
    rx, ry = tl.split(tl.reshape(r2, (BLOCK_T, J, 2)))
    z = (px * bu + py * bw + qx * bu * bu + qy * bw * bw + qxy * (2.0 * bu * bw) + 2.0 * ca * (rx * bu + ry * bw)
      + s1[:, None] * ca + s2[:, None] * (ca * ca))
    steps = tl.load(steps_ptr + row * T + offs_t)
    _store_by_token(out_ptr, z / steps[:, None], row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)
    _store_by_token(value_ptr + blk.to(tl.int64) * OUT_BLK, val, row, BH, T, offs_t, part, M, W_STRIDE, D, J, offs_j)


def moment_key_scores(q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out, values, repeat):
    rows, t, d = q.shape
    blocks = raw.shape[1]
    _moment_keys_kernel[(rows * repeat, t // 64)](q, k, v, hub, raw, sq, side, cos, sin, lse, hub_out, steps, out,
      values, rows // blocks, t, d ** -0.5, repeat,
      raw.stride(1), raw.stride(0), raw.stride(2), out.shape[2],
      out.stride(1), out.stride(0), D=d, J=d // repeat,
      BLOCK_T=64, BLOCK_S=32, num_warps=4, num_stages=3)


F32, BF16 = torch.float32, torch.bfloat16
BATCH, SEQ = 8, 2048
ID_SEQ_LEN = 1024  # identity sequence length: each 2049-token row trains as two 1024-token sequences
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8
LR_WTE, LR_VE, LR_LAMBDAS = 1.8, 1.8, 0.006
ANVIL_LR, ANVIL_WD = 0.04, 2.25
ANVIL_DECAY = 0.09  # identity-phase ANVIL decay (lr x wd for this rate)
GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1
ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240)  # #44's ANVIL momentum warmup: the first 240 of 1194 updates (see anvil_betas)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_LAG, START_RHO, START_CONTENT = 1, 1, 0.8, 0.25


def id_has_ve(layer):
    return layer % 2 == (L - 1) % 2


VE_BLOCKS = tuple(i for i in range(L) if id_has_ve(i))


def id_norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)


def softcap(x):
    return SOFTCAP * torch.tanh(x / SOFTCAP)


def id_hadamard(d, device):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


@torch.no_grad()
def pair_value_output(model):
    # Each attn.c_proj becomes c_v^T at the random c_proj's Frobenius norm (the second rescale absorbs rounding).
    for i in range(L):
        value = model.w(i, 'attn.c_v')
        output = model.w(i, 'attn.c_proj')
        original_norm = output.norm()
        paired = value.T * (original_norm / value.norm())
        output.copy_(paired * (original_norm / paired.norm()))


SHAPING_UPDATES = 195  # the four update-shaping mechanisms below act on identity updates 0..194
SPLIT_SHAPING = True  # the identity phase as two graphs: with the shaping mechanisms for updates 0..194, without after
SHAPING_ON = True  # (set while each graph is traced: False drops the mechanisms, a no-op from update 195 on)
VE_COMMON_GAIN = 4.0


def amplify_common_motion(delta):
    # Amplify the vocabulary-common part of a value-embedding Adam update VE_COMMON_GAIN-fold at fixed energy.
    common = delta.mean(dim=0, keepdim=True)
    amplified = delta + (VE_COMMON_GAIN - 1.0) * common
    original_energy = delta.square().sum()
    amplified_energy = amplified.square().sum()
    return amplified * torch.sqrt(original_energy / amplified_energy)


WTE_GATE_ENERGY = 0.00390625  # 1/256 of each wte row's energy in its GATE_CHANNELS gate-input channels


def constrain_wte_gate_energy(wte, step_t):
    if not SHAPING_ON:
        return
    x = wte.float()
    sector = torch.arange(D, device=x.device) < GATE_CHANNELS
    inside = torch.where(sector, x.square(), 0.0).sum(-1, keepdim=True)
    outside = torch.where(sector, 0.0, x.square()).sum(-1, keepdim=True)
    total = inside + outside
    inside_scale = torch.sqrt(WTE_GATE_ENERGY * total / inside)
    outside_scale = torch.sqrt((1.0 - WTE_GATE_ENERGY) * total / outside)
    projected = x * torch.where(sector, inside_scale, outside_scale)
    wte.copy_(torch.where(step_t < SHAPING_UPDATES, projected, x).to(wte.dtype))


constrain_wte_gate_energy_compiled = torch.compile(constrain_wte_gate_energy, dynamic=False)


@torch.compile(dynamic=False)
def pool_gate_credits(grads, step_t):
    # Acts on the all-reduced gate credits, which feed only Adam: each VE gate's head-mean credit becomes the mean
    # over the four VE layers.
    if not SHAPING_ON:
        return
    common = torch.stack(grads).mean(dim=(0, 2))[:, None]
    for grad in grads:
        pooled = grad - grad.mean(dim=1, keepdim=True) + common
        grad.copy_(torch.where(step_t < SHAPING_UPDATES, pooled, grad))


VE_PARTNER_GAIN = 0.5
VE_PARTNERS = (2, 3, 0, 1)  # each VE layer's partner by index in VE_BLOCKS (1, 3, 5, 7): 1<->5, 3<->7


def transfer_partner_motion(deltas, frames, step_t):
    # Completed Adam motion only. No transformed tensor returns to a nonlinear probe.
    common = deltas.mean(dim=1, keepdim=True)
    centered = deltas - common
    energy = centered.square().sum(dim=(1, 2), keepdim=True)
    response = torch.bmm(centered, frames)
    response = response * torch.rsqrt(response.square().sum(dim=(1, 2), keepdim=True))
    # Each layer borrows its partner's motion (same SSSL window size), mapped through the partner's c_proj frame and
    # back through its own.
    donor = torch.stack([response[j] for j in VE_PARTNERS])
    borrowed = torch.bmm(donor, frames.transpose(1, 2))
    borrowed = borrowed - borrowed.mean(dim=1, keepdim=True)
    borrowed = borrowed * torch.sqrt(energy / borrowed.square().sum(dim=(1, 2), keepdim=True))
    mixed = centered + VE_PARTNER_GAIN * borrowed
    mixed = mixed - mixed.mean(dim=1, keepdim=True)
    mixed_energy = mixed.square().sum(dim=(1, 2), keepdim=True)
    mixed = mixed * torch.sqrt(energy / mixed_energy)
    candidate = common + mixed
    return torch.where(step_t < SHAPING_UPDATES, candidate, deltas)


class Model:
    def __init__(self, device, rank, world):
        self.device, self.rank, self.world = device, rank, world
        g = torch.Generator(device='cpu').manual_seed(SEED)
        s = math.sqrt(3.0) * D ** -0.5
        uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
        proj = lambda fan_in, fan_out: uni((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
        self.square = torch.stack([uni((D, D), s) if k != 'attn.c_proj' else proj(D, D)
        for k in SQUARE for _ in range(L)]).to(device)
        self.wide = torch.stack([uni((D, 4 * D), s) for _ in range(L)] + [proj(4 * D, D).T for _ in range(L)]).to(device)
        self.gates = {i: torch.zeros(GATE_CHANNELS, H, device=device) for i in range(L) if id_has_ve(i)}
        self.wte = torch.randn(VOCAB, D, generator=g).to(device, BF16)
        self.lm_head = (0.001 * torch.randn(D, VOCAB, generator=g)).to(device)
        self.ve = {i: uni((VOCAB, D), s).to(device, BF16) for i in range(L) if id_has_ve(i)}
        self.lam_r = torch.ones(L, device=device)
        self.lam_0 = torch.full((L,), 0.1, device=device)
        pair_value_output(self)
        constrain_wte_gate_energy(self.wte, torch.full((), -1.0, device=device))
        structured_start(self)
        shapes = dict(wte=self.wte.shape, lm_head=self.lm_head.shape,
        lam_r=(L,), lam_0=(L,), **{f'gate{i}': (GATE_CHANNELS, H) for i in self.gates},
        **{f've{i}': (VOCAB, D) for i in self.ve})
        sizes = {k: math.prod(v) for k, v in shapes.items()}
        self.flat_grad = torch.zeros(sum(sizes.values()), device=device)
        self.grad, offset = {}, 0
        for k, shape in shapes.items():
            self.grad[k] = self.flat_grad[offset:offset + sizes[k]].view(shape)
            offset += sizes[k]
        self.grad['square'] = torch.zeros(self.square.shape, device=device, dtype=BF16)
        self.grad['wide'] = torch.zeros(self.wide.shape, device=device, dtype=BF16)
        self.owned = {name: torch.zeros(w.shape[0] // world, *w.shape[1:], device=device, dtype=BF16)
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.adam = [(self.wte, self.grad['wte'], LR_WTE), (self.lm_head, self.grad['lm_head'], LR),
        (self.lam_r, self.grad['lam_r'], LR_LAMBDAS), (self.lam_0, self.grad['lam_0'], LR_LAMBDAS)]
        self.adam += [(self.gates[i], self.grad[f'gate{i}'], LR) for i in self.gates]
        self.adam += [(self.ve[i], self.grad[f've{i}'], LR_VE) for i in self.ve]
        self.m = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.v = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.anvil = {name: dict(fast=torch.zeros_like(w), slow=torch.zeros_like(w),
        energy=torch.zeros(w.shape[0], 1, w.shape[2], device=device))
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.H = id_hadamard(D, device)
        self.rows = torch.randint(0, VOCAB, (IDENTITY_UPDATES, BATCH, SEQ + 1), device=device)
        self.loss = torch.zeros(1, device=device)
        self.step_t = torch.zeros((), device=device)
        self.rates = dict(square=torch.ones(4 * L, device=device),
        wide=torch.tensor([2.0] * L + [1.0] * L, device=device))
        self.square_bf, self.wide_bf = self.square.to(BF16), self.wide.to(BF16)
        self.shard = {name: slice(rank * (w.shape[0] // world), (rank + 1) * (w.shape[0] // world))
        for name, w in (('square', self.square), ('wide', self.wide))}
        self.lm_head_bf = self.lm_head.to(BF16)
        self.gates_bf = {i: g.to(BF16) for i, g in self.gates.items()}
        # 1024-token sequences: every SSSL window (1024 or 2048 tokens) covers the whole causal context.
        self.mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, ID_SEQ_LEN, ID_SEQ_LEN, device=device)
        self.rope = rotary(ID_SEQ_LEN, device)
    def w(self, i, kind):
        if kind in SQUARE:
            return self.square[SQUARE.index(kind) * L + i]
        return self.wide[i] if kind == 'mlp.c_fc' else self.wide[L + i].T
    def wb(self, i, kind):
        if kind in SQUARE:
            return self.square_bf[SQUARE.index(kind) * L + i]
        return self.wide_bf[i] if kind == 'mlp.c_fc' else self.wide_bf[L + i].T
    @torch.no_grad()
    def refresh_bf16(self):
        self.lm_head_bf.copy_(self.lm_head)
        for i, g in self.gates.items():
            self.gates_bf[i].copy_(g)
    @torch.no_grad()
    def cast_all(self):
        self.square_bf.copy_(self.square)
        self.wide_bf.copy_(self.wide)
        self.refresh_bf16()
    @torch.no_grad()
    def gather_fp32(self):
        for name, w in (('square', self.square), ('wide', self.wide)):
            dist.all_gather_into_tensor(w, w[self.shard[name]].clone())
    def tensors(self):
        out = [self.square, self.wide, self.wte, self.lm_head, self.lam_r, self.lam_0, *self.gates.values(),
        *self.ve.values(), *self.m, *self.v]
        return out + [x for st in self.anvil.values() for x in st.values()]
    def state_dict(self):
        sd = {'transformer.wte.weight': self.wte.float(), 'lm_head.weight': self.lm_head.T,
        'resid_lambdas': self.lam_r, 'x0_lambdas': self.lam_0}
        for i in range(L):
            for kind in (*SQUARE, 'mlp.c_fc', 'mlp.c_proj'):
                sd[f'transformer.h.{i}.{kind}.weight'] = self.w(i, kind).T
            if id_has_ve(i):
                sd[f'transformer.h.{i}.attn.ve_gate.weight'] = self.gates[i].T
                sd[f'value_embeds.{i}.weight'] = self.ve[i].float()
        return {k: v.detach().float().contiguous() for k, v in sd.items()}


def rope_pair(lag, rho, generator):
    half = HD // 2
    phase = 2 * math.pi * torch.rand(half, generator=generator)
    amplitude = rho ** (torch.arange(half, dtype=F32) / 2)
    q = torch.cat([amplitude * phase.cos(), amplitude * phase.sin()])
    q = q * math.sqrt(HD) / q.norm()
    angle = lag / 10000 ** (torch.arange(0, HD, 2, dtype=F32) / HD)
    q1, q2 = q[:half], q[half:]
    return q, torch.cat([q1 * angle.cos() + q2 * angle.sin(), q2 * angle.cos() - q1 * angle.sin()])


def relu2_moments(mu, sd):
    z = mu / sd
    cdf, pdf = 0.5 * (1 + torch.erf(z / math.sqrt(2))), torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    m2 = (mu ** 2 + sd ** 2) * cdf + mu * sd * pdf
    m4 = (mu ** 4 + 6 * mu ** 2 * sd ** 2 + 3 * sd ** 4) * cdf + (mu ** 3 + 5 * mu * sd ** 2) * sd * pdf
    return m2, m4


@torch.no_grad()
def shared_mean(model, block):
    mean, energy, x0 = torch.zeros(D, device=model.device), float(D), 1.0
    for i in range(block + 1):
        r, lam = model.lam_r[i].item(), model.lam_0[i].item()
        energy = r * r * energy + lam * lam * D + 2 * r * lam * x0 * D
        mean, x0 = r * mean, r * x0 + lam
        normed = mean / math.sqrt(energy / D)
        if i == block:
            return normed
        attended = (normed @ model.w(i, 'attn.c_v')) @ model.w(i, 'attn.c_proj')
        energy, mean = energy + (attended @ attended).item() + 2 * (mean @ attended).item(), mean + attended
        shared = mean / math.sqrt(energy / D)
        fc, proj = model.w(i, 'mlp.c_fc'), model.w(i, 'mlp.c_proj')
        spread = fc.norm(dim=0) * max(1 - (shared @ shared).item() / D, 1e-3) ** 0.5
        m2, m4 = relu2_moments(shared @ fc, spread)
        written = m2 @ proj
        energy += (written @ written).item() + (proj.square().sum(1) @ (m4 - m2 * m2)).item()
        energy, mean = energy + 2 * (mean @ written).item(), mean + written


@torch.no_grad()
def structured_start(model):
    generator = torch.Generator(device='cpu').manual_seed(SEED + 31)
    mean = shared_mean(model, START_BLOCK)
    for h in range(H):
        cols = slice(h * HD, (h + 1) * HD)
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAG, START_RHO, generator)):
            w = model.w(START_BLOCK, kind)
            w[:, cols] = START_CONTENT * w[:, cols] + torch.outer(mean, v.to(mean.device)) / (mean @ mean)


def rotary(t, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, HD, 2, dtype=F32, device=device) / HD)
    freqs = torch.outer(torch.arange(t, dtype=F32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()


def rope_norm(raw, cos, sin):
    x = raw.float()
    c, s_ = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * c + x2 * s_, x1 * -s_ + x2 * c], -1)
    return (rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)).to(BF16).transpose(1, 2).contiguous()


# the identity forward's flex attention kernel_options (None: the defaults; BLOCK_N 128 as the default keeps it bitwise)
ID_FLEX_TILES = dict(BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=3)
ID_COMBO = True  # the identity forward compiled with inductor's combo kernels (horizontal fusion of small kernels)


def forward_impl(model, idx, targets, cos, sin):
    b, t = idx.shape
    cache = {}
    x = id_norm(model.wte[idx])
    x0, pending = x, None
    cache['x0'] = x0
    for i in range(L):
        mixed = model.lam_r[i] * (x if pending is None else x + pending.float()) + model.lam_0[i] * x0
        n1 = id_norm(mixed).to(BF16)
        q, k, v = torch.bmm(n1.reshape(1, b * t, D).expand(3, -1, -1), model.square_bf[i:3 * L:L]).view(3, b, t, D)
        values = v.view(b, t, H, HD)
        if id_has_ve(i):
            ve = model.ve[i][idx]
            z = n1[..., :GATE_CHANNELS] @ model.gates_bf[i]
            values = values + (2 * torch.sigmoid(z.float())).unsqueeze(-1).to(BF16) * ve.view(b, t, H, HD)
            cache[f'h{i}.ve_gate'], cache[f'h{i}.ve'] = z, ve
        qp, kp = rope_norm(q.view(b, t, H, HD), cos, sin), rope_norm(k.view(b, t, H, HD), cos, sin)
        o, aux = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.mask,
        return_aux=AuxRequest(lse=True), kernel_options=ID_FLEX_TILES)
        lse = aux.lse
        o = o.transpose(1, 2).reshape(b, t, D)
        cache.update({f'h{i}.in.attn': n1, f'h{i}.q': q, f'h{i}.k': k, f'h{i}.values': values,
        f'h{i}.lse': lse.float(), f'h{i}.in.attn.c_proj': o})
        x = mixed + (o @ model.wb(i, 'attn.c_proj')).float()
        n2 = id_norm(x).to(BF16)
        hidden = n2 @ model.wb(i, 'mlp.c_fc')
        act = F.relu(hidden).square()
        pending = act @ model.wb(i, 'mlp.c_proj')
        cache.update({f'h{i}.in.mlp.c_fc': n2, f'h{i}.mlp.c_fc': hidden, f'h{i}.in.mlp.c_proj': act,
        f'h{i}.stream': x + pending.float()})
    final = id_norm(x + pending.float()).to(BF16).reshape(b * t, D)
    raw = final @ model.lm_head_bf
    cache['in.lm_head'], cache['lm_head'] = final, raw
    loss = F.cross_entropy(softcap(raw.float()), targets.reshape(-1))
    return cache, loss


forward = torch.compile(forward_impl, dynamic=False, options={'combo_kernels': True} if ID_COMBO else {})


@torch.compile(dynamic=False)
def coordinate_change(raw, targets):
    raw = raw.float()
    z = softcap(raw)
    up, down = softcap(raw + HEAD_SIGMA), softcap(raw - HEAD_SIGMA)
    weight = torch.softmax(z, -1)
    change = torch.log1p(weight * torch.expm1(up - z)) - torch.log1p(weight * torch.expm1(down - z))
    change = change - F.one_hot(targets, VOCAB).float() * (up - down)
    return change / (2 * HEAD_SIGMA)


def step_signs(step_t, device):
    x = (step_t.to(torch.int64) * 0x9E3779B1 + torch.arange(D, device=device, dtype=torch.int64) * 0x85EBCA77
      + SEED) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x = x ^ (x >> 12)
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x = x ^ (x >> 15)
    return 1.0 - 2.0 * (x & 1).float()


@torch.compile(dynamic=False)
def cycle_weights(x, base, h):
    steps = SIGMA * torch.clamp(torch.sqrt(x.square().mean(-1, keepdim=True)), min=1e-6)
    sq = x.square().sum(-1, keepdim=True)
    dots = x @ h.T
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * D) / D + EPS)
    inv0 = torch.rsqrt(sq / D + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv
    z = base * inv0
    t = torch.tanh(z / SOFTCAP)
    f = torch.exp(SOFTCAP * t)
    g1 = 1 - t * t
    d1 = -2 * t * g1 / SOFTCAP
    d2 = -2 * g1 * (1 - 3 * t * t) / SOFTCAP ** 2
    g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2
    fz, fzz = f * z, f * z * z
    total = f.sum(-1, keepdim=True)
    rows = torch.cat([(fz * g1).sum(-1, keepdim=True), (fzz * g2).sum(-1, keepdim=True),
      (fzz * z * g3).sum(-1, keepdim=True)], 1)
    fast = torch.stack([fz * g2, fzz * g3, f * g2, fz * g3, f * g3]).to(BF16)
    return f * g1, fast, rows, total, steps, inv, inv0, delta, sigma


FUSED_CYCLE = True  # top_credit's readout-cycle weights in one pass over the logits (_cycle_weights_kernel)


@torch.compile(dynamic=False)
def cycle_scales(x, h):
    """cycle_weights' per-token step, norm inverses and their ratios."""
    steps = SIGMA * torch.clamp(torch.sqrt(x.square().mean(-1, keepdim=True)), min=1e-6)
    sq = x.square().sum(-1, keepdim=True)
    dots = x @ h.T
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * D) / D + EPS)
    inv0 = torch.rsqrt(sq / D + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv
    return steps, inv, inv0, delta, sigma


@triton.jit
def _cycle_weights_kernel(Base, Inv0, FG1, Fast, Rows, Total, T: tl.constexpr, V: tl.constexpr,
    BLOCK: tl.constexpr, CAP: tl.constexpr):
    """cycle_weights' per-logit part for one token in one pass: f g1 (fp32), the five bf16 weight rows, and the row
    sums (total, and rows' three)."""
    token = tl.program_id(0).to(tl.int64)
    inv0 = tl.load(Inv0 + token)
    total = tl.zeros((BLOCK,), tl.float32)
    r1 = tl.zeros((BLOCK,), tl.float32)
    r2 = tl.zeros((BLOCK,), tl.float32)
    r3 = tl.zeros((BLOCK,), tl.float32)
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        z = tl.load(Base + token * V + cols) * inv0
        t = libdevice.tanh(z / CAP)
        f = libdevice.exp(CAP * t)
        g1 = 1 - t * t
        d1 = -2 * t * g1 / CAP
        d2 = -2 * g1 * (1 - 3 * t * t) / (CAP * CAP)
        g2, g3 = g1 * g1 + d1, g1 * g1 * g1 + 3 * g1 * d1 + d2
        fz, fzz = f * z, f * z * z
        total += f
        r1 += fz * g1
        r2 += fzz * g2
        r3 += fzz * z * g3
        out = token * V + cols
        tl.store(FG1 + out, f * g1)
        tl.store(Fast + out, (fz * g2).to(tl.bfloat16))
        tl.store(Fast + T * V + out, (fzz * g3).to(tl.bfloat16))
        tl.store(Fast + 2 * T * V + out, (f * g2).to(tl.bfloat16))
        tl.store(Fast + 3 * T * V + out, (fz * g3).to(tl.bfloat16))
        tl.store(Fast + 4 * T * V + out, (f * g3).to(tl.bfloat16))
    tl.store(Total + token, tl.sum(total, 0))
    tl.store(Rows + token * 3, tl.sum(r1, 0))
    tl.store(Rows + token * 3 + 1, tl.sum(r2, 0))
    tl.store(Rows + token * 3 + 2, tl.sum(r3, 0))


def cycle_weights_fused(x, base, h):
    """cycle_weights (same outputs) with its per-logit part in one kernel pass."""
    steps, inv, inv0, delta, sigma = cycle_scales(x, h)
    tokens, vocab = base.shape
    fg1 = torch.empty_like(base)
    fast = torch.empty((5, tokens, vocab), device=base.device, dtype=BF16)
    rows = torch.empty((tokens, 3), device=base.device)
    total = torch.empty((tokens, 1), device=base.device)
    _cycle_weights_kernel[(tokens,)](base.contiguous(), inv0.contiguous(), fg1, fast, rows, total, T=tokens, V=vocab,
                                     BLOCK=1024, CAP=SOFTCAP, num_warps=4)
    return fg1, fast, rows, total, steps, inv, inv0, delta, sigma


@torch.compile(dynamic=False)
def cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h):
    fast = [x.float() for x in prods]
    ratio = (sigma * exact + delta * sigma * fast[0] + 0.5 * delta ** 2 * sigma * fast[1] + 0.5 * sigma ** 2 * fast[2]
      + 0.5 * delta * sigma ** 2 * fast[3] + sigma ** 3 / 6 * fast[4]
      + delta * rows[:, :1] + 0.5 * delta ** 2 * rows[:, 1:2] + delta ** 3 / 6 * rows[:, 2:])
    clean_partition = torch.log(total)
    base_y = base.gather(1, y[:, None])
    dirs_y = c.T[y]
    clean = clean_partition - softcap(base_y * inv0)
    moved = clean_partition + torch.log1p(ratio / total) - softcap((base_y + steps * dirs_y) * inv)
    return ((moved - clean) / steps) @ h / D


@torch.compile(dynamic=False)
def cycle_powers(c):
    return c.to(BF16), (c * c).to(BF16), (c * c * c).to(BF16)


def top_credit(model, x, y):
    h = model.H * step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head
    base = x @ model.lm_head
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = (cycle_weights_fused if FUSED_CYCLE else cycle_weights)(
        x, base, h)
    exact = fg1 @ c.T
    cb, c2, c3 = cycle_powers(c)
    if FUSED_CYCLE:  # the products sharing a right operand as one GEMM (stacked along tokens; bitwise the same)
        n = x.shape[0]
        p01 = (fast[:2].reshape(2 * n, -1) @ cb.T).view(2, n, -1)
        p23 = (fast[2:4].reshape(2 * n, -1) @ c2.T).view(2, n, -1)
        prods = [p01[0], p01[1], p23[0], p23[1], fast[4] @ c3.T]
    else:
        prods = [fast[0] @ cb.T, fast[1] @ cb.T, fast[2] @ c2.T, fast[3] @ c2.T, fast[4] @ c3.T]
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def rope32(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


REPEAT, REPEAT_K = HD // 16, HD // 32
PACK = 5 * HD + 2


@torch.compile(dynamic=False)
def pack_heads(qs, ks, vals, outs, targets, lse_bits):
    parts = []
    for q, k, v, o, tg, lb in zip(qs, ks, vals, outs, targets, lse_bits):
        m = q.shape[0]
        bits = lb.view(m, H, 2)
        parts.append(torch.cat([q.view(m, H, HD), k.view(m, H, HD), v.view(m, H, HD), o.view(m, H, HD),
        tg.view(m, H, HD), bits], -1))
    return torch.stack(parts).permute(2, 0, 1, 3).contiguous()


PACK_KERNEL = True  # identity_grad's all_to_all send buffer packed by _pack_heads_kernel (one pass, bitwise pack_heads)


@triton.jit
def _pack_heads_kernel(Q, K, V, O, TG, LSE, Out16, Out32, layer, sq, sk, sv, so, st, L: tl.constexpr,
                       M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr):
    """One layer's rows of pack_heads' (H, L, M, PACK) output for BM tokens of head h: the five 64-wide pieces and
    the lse's fp32 bits (LSE (rows, H, T) fp32 as int32), copied as raw 16-bit words."""
    tile, h = tl.program_id(0), tl.program_id(1)
    tok = (tile * BM + tl.arange(0, BM)).to(tl.int64)
    d = tl.arange(0, 64)
    row = (h * L + layer) * M + tok
    dst = Out16 + row[:, None] * PACK + d[None, :]
    col = h * 64 + d[None, :]
    tl.store(dst, tl.load(Q + tok[:, None] * sq + col))
    tl.store(dst + 64, tl.load(K + tok[:, None] * sk + col))
    tl.store(dst + 128, tl.load(V + tok[:, None] * sv + col))
    tl.store(dst + 192, tl.load(O + tok[:, None] * so + col))
    tl.store(dst + 256, tl.load(TG + tok[:, None] * st + col))
    lse = tl.load(LSE + ((tok // T) * H + h) * T + tok % T)
    tl.store(Out32 + row * (PACK // 2) + 5 * 32, lse)


def pack_heads_kernel(qs, ks, vals, outs, targets, lses):
    """pack_heads (bitwise) from each layer's (tokens, D) pieces and its lse (rows, H, t), one launch per layer."""
    n, t = qs[0].shape[0], lses[0].shape[-1]
    out = torch.empty(H, len(qs), n, PACK, device=qs[0].device, dtype=BF16)
    o16, o32 = out.view(torch.int16), out.view(torch.int32)
    for i, (q, k, v, o, tg, lse) in enumerate(zip(qs, ks, vals, outs, targets, lses)):
        assert all(z.stride(1) == 1 for z in (q, k, v, o, tg)) and n % 16 == 0
        _pack_heads_kernel[(n // 16, H)](q.view(torch.int16), k.view(torch.int16), v.view(torch.int16),
                                         o.view(torch.int16), tg.view(torch.int16), lse.contiguous().view(torch.int32),
                                         o16, o32, i, q.stride(0), k.stride(0), v.stride(0), o.stride(0), tg.stride(0),
                                         L=len(qs), M=n, T=t, H=H, PACK=PACK, BM=16, num_warps=1)
    return out


@torch.compile(dynamic=False)
def head_prep(got, lse, cos, sin, b: int, t: int):
    k = got.shape[1]
    whole, lse = got.transpose(0, 1).reshape(k * b, t, PACK), lse.transpose(0, 1).reshape(k * b, t)
    q_raw, k_raw, values, out, hub = (whole[..., j * HD:(j + 1) * HD].float() for j in range(5))
    q1, k1 = rope32(q_raw, cos, sin), rope32(k_raw, cos, sin)
    sq_q, sq_k = (q_raw * q_raw).sum(-1), (k_raw * k_raw).sum(-1)
    steps = lambda sq: KEY_SIGMA * torch.clamp(torch.sqrt(sq / HD), min=1e-6)
    return (q1.to(BF16), k1.to(BF16), values.to(BF16), hub.to(BF16), q1, k1, lse.contiguous(), (hub * out).sum(-1),
      sq_q, sq_k, steps(sq_q), steps(sq_k))


FUSED_HEAD_PREP = True  # id_head_scores' head_prep in one Triton pass (_head_prep_kernel)


@triton.jit
def _head_prep_kernel(Got, Got32, Cos, Sin, Q16, K16, V16, Hub16, Q32, K32, LSE, HubOut, SqQ, SqK, StepsQ, StepsK,
                      W: tl.constexpr, L: tl.constexpr, N: tl.constexpr, T: tl.constexpr, PACK: tl.constexpr,
                      SIGMA: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr):
    """head_prep for BM tokens of rank w's layer l (got (W, L, N, PACK)): output token (l * W + w) * N + i."""
    tile, wl = tl.program_id(0), tl.program_id(1)
    w, layer = wl % W, wl // W
    i = tile * BM + tl.arange(0, BM)
    src = ((w * L + layer) * N + i).to(tl.int64)
    dst = ((layer * W + w) * N + i).to(tl.int64)
    pos = i % T
    half = tl.arange(0, 32)
    dims = tl.arange(0, 64)
    c = tl.load(Cos + pos[:, None] * 32 + half[None, :])
    s = tl.load(Sin + pos[:, None] * 32 + half[None, :])
    row = Got + src[:, None] * PACK
    for j in tl.static_range(2):
        x1 = tl.load(row + j * 64 + half[None, :]).to(tl.float32)
        x2 = tl.load(row + j * 64 + 32 + half[None, :]).to(tl.float32)
        r1 = x1 * c + x2 * s
        r2 = x1 * -s + x2 * c
        inv = tl.rsqrt((tl.sum(r1 * r1, 1) + tl.sum(r2 * r2, 1)) / 64 + EPS)
        r1, r2 = r1 * inv[:, None], r2 * inv[:, None]
        sq = tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)
        out16 = (Q16 if j == 0 else K16) + dst[:, None] * 64 + half[None, :]
        out32 = (Q32 if j == 0 else K32) + dst[:, None] * 64 + half[None, :]
        tl.store(out16, r1.to(tl.bfloat16))
        tl.store(out16 + 32, r2.to(tl.bfloat16))
        tl.store(out32, r1)
        tl.store(out32 + 32, r2)
        tl.store((SqQ if j == 0 else SqK) + dst, sq)
        tl.store((StepsQ if j == 0 else StepsK) + dst, SIGMA * tl.maximum(tl.sqrt(sq / 64), 1e-6))
    v = tl.load(row + 128 + dims[None, :])
    o = tl.load(row + 192 + dims[None, :]).to(tl.float32)
    hub = tl.load(row + 256 + dims[None, :])
    tl.store(V16 + dst[:, None] * 64 + dims[None, :], v)
    tl.store(Hub16 + dst[:, None] * 64 + dims[None, :], hub)
    tl.store(HubOut + dst, tl.sum(hub.to(tl.float32) * o, 1))
    tl.store(LSE + dst, tl.load(Got32 + src * (PACK // 2) + 160))


def head_prep_fused(got, cos, sin, b: int, t: int):
    """head_prep's outputs (the same up to summation order) from got in one kernel pass."""
    w, layers, n, _ = got.shape
    rows = layers * b
    new = lambda *s, dtype=torch.float32: torch.empty(*s, device=got.device, dtype=dtype)
    q16, k16, v16, hub16 = (new(rows, t, HD, dtype=BF16) for _ in range(4))
    q32, k32 = new(rows, t, HD), new(rows, t, HD)
    lse, hub_out, sq_q, sq_k, steps_q, steps_k = (new(rows, t) for _ in range(6))
    _head_prep_kernel[(n // 32, w * layers)](got, got.view(torch.float32), cos, sin, q16, k16, v16, hub16, q32, k32,
                                             lse, hub_out, sq_q, sq_k, steps_q, steps_k, W=w, L=layers, N=n, T=t,
                                             PACK=PACK, SIGMA=KEY_SIGMA, EPS=NORM_EPS, BM=32, num_warps=4)
    return q16, k16, v16, hub16, q32, k32, lse, hub_out, sq_q, sq_k, steps_q, steps_k


def id_head_scores(got, rope32, back, b, t):
    if FUSED_HEAD_PREP:
        q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep_fused(got, *rope32, b, t)
    else:
        lse_bits = got[..., 5 * HD:].view(F32)[..., 0]
        q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep(got, lse_bits, *rope32,
                                                                                              b, t)
    queries = moment_kernels.moment_query_scores if FAST_ID_MOMENTS else moment_query_scores
    keys = moment_kernels.moment_key_scores if FAST_ID_MOMENTS else moment_key_scores
    queries(q1, k1, values, hub, got, sq_q, k1f, *rope32, lse, hub_out, steps_q, back[:, :, 0], REPEAT)
    keys(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k, back[:, :, 1], back[:, :, 2],
         REPEAT_K)


FAST_ID_MOMENTS = True  # the identity phase's query / key moment kernels by moment_kernels (each pair's sums once)


@torch.compile(dynamic=False)
def credit_stacks(got, ve_blocks: tuple):
    world, _, _, m, hd = got.shape
    by_token = got.permute(1, 2, 3, 0, 4).reshape(L, 3, m, world * hd)
    stacks = by_token.transpose(0, 1).contiguous()
    return stacks, [by_token[i, 2].view(m, H, HD).float() for i in ve_blocks]


def f32_stack(xs):
    return torch.stack([x.reshape(-1, x.shape[-1]) for x in xs]).float()


def bf16_stacks(n1, o, n2, act):
    st = lambda xs: torch.stack([x.reshape(-1, x.shape[-1]) for x in xs]).to(BF16)
    return st(n1), st(o), st(n2), st(act)


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


@torch.compile(dynamic=False)
def hidden_slope_(u, read):
    u = u.reshape(-1, u.shape[-1]).float()
    step = FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1, keepdim=True)), min=1e-6)
    relu2 = lambda z: F.relu(z).square()
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    read.copy_((slope * read.float()).to(BF16))


def hidden_credits(us, credb, w_out_t):
    hidden = torch.bmm(credb, w_out_t)
    for u, read in zip(us, hidden):
        hidden_slope_(u, read)
    return hidden


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


@torch.compile(dynamic=False)
def value_embed_credits(z, ve, vcredit, n1):
    z, ve = z.float(), ve.float().view(-1, H, HD)
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    gate = n1[:, :GATE_CHANNELS].float().T @ (slope * (vcredit * ve).sum(-1))
    return gate, (vcredit * 2 * torch.sigmoid(z)[..., None]).view(-1, D)


# identity_grad's collectives overlap: the wide gradients (hidden credits + two bmms) in heads_in's window (False) or in
# heads_out's (True); the order of the other independent work is fixed (numerically identical either way)
ID_WIDE_LATE = False


def identity_grad(model, idx, targets, rows_total):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens, one head's queries, keys and values over the batch. The ANVIL matrices' gradients stay this
    GPU's partial sums, reduce-scattered; the rest are all-reduced. Returns the clean loss of its tokens, the collectives
    in flight and the token count."""
    world = model.world
    t = idx.shape[1]
    n, m = rows_total * t, idx.numel()
    rope32 = model.rope
    cache, loss = forward(model, idx, targets, *rope32)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1), idx.reshape(-1)
    gsq, gwide = model.grad['square'], model.grad['wide']
    # heads_in's payload first (it needs the readout credit for targets_o); the work that does not feed it runs while
    # it is in flight (same calls as before, reordered: numerically identical)
    credit_top = top_credit(model, stream(f'h{L - 1}.stream'), y)
    cred, credb = block_credits(credit_top, model.lam_r)
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))
    if PACK_KERNEL:
        send = pack_heads_kernel([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
                                 [mine(cache[f'h{i}.values']) for i in range(L)],
                                 [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)],
                                 [targets_o[i] for i in range(L)], [cache[f'h{i}.lse'] for i in range(L)])
    else:
        send = pack_heads([mine(cache[f'h{i}.q']) for i in range(L)], [mine(cache[f'h{i}.k']) for i in range(L)],
        [mine(cache[f'h{i}.values']) for i in range(L)],
        [mine(cache[f'h{i}.in.attn.c_proj']) for i in range(L)], [targets_o[i] for i in range(L)],
        [cache[f'h{i}.lse'].transpose(1, 2).reshape(-1, H).view(BF16)
        for i in range(L)])
    got = torch.empty_like(send)
    heads_in = dist.all_to_all_single(got, send, async_op=True)
    coords = coordinate_change(cache['lm_head'], y)
    prevs = f32_stack([cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)])
    n1s, os_, n2s, acts = bf16_stacks([cache[f'h{i}.in.attn'] for i in range(L)],
      [cache[f'h{i}.in.attn.c_proj'] for i in range(L)],
      [cache[f'h{i}.in.mlp.c_fc'] for i in range(L)],
      [cache[f'h{i}.in.mlp.c_proj'] for i in range(L)])
    wide = lambda: (torch.bmm(n2s.transpose(1, 2), hidden_credits([cache[f'h{i}.mlp.c_fc'] for i in range(L)], credb,
      model.wide_bf[L:]), out=gwide[:L]), torch.bmm(credb.transpose(1, 2), acts, out=gwide[L:]))
    if not ID_WIDE_LATE:
        wide()
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    works = {}
    heads_in.wait()
    back = torch.empty(world, L, 3, m, HD, device=idx.device, dtype=BF16)
    id_head_scores(got, rope32, back, rows_total, t)
    got2 = torch.empty_like(back)
    heads_out = dist.all_to_all_single(got2, back, async_op=True)
    if ID_WIDE_LATE:
        wide()
    # the wide gradients are complete: their reduce-scatter queues right behind heads_out
    works['wide'] = dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True)
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    torch.bmm(os_.transpose(1, 2), credb, out=gsq[3 * L:])
    heads_out.wait()
    by_token, vcredits = credit_stacks(got2, VE_BLOCKS)
    for j in range(3):
        torch.bmm(n1s.transpose(1, 2), by_token[j], out=gsq[j * L:(j + 1) * L])
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        gate, rows = value_embed_credits(mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(gate)
        model.grad[f've{i}'].zero_().index_add_(0, tokens, rows)
    works['flat'] = dist.all_reduce(model.flat_grad, async_op=True)
    return loss, works, n


def anvil_betas(t):
    # ANVIL_RUN's momentum warmup (0.85 -> 0.93 over 240 of 1194 updates) scaled to the 1831-update identity
    # schedule lasts 368 updates, so all 300 identity updates are on the ramp.
    warm = round(ANVIL_RUN[1] * IDENTITY_SCHEDULE / ANVIL_RUN[0])
    lo, hi = 0.85, 0.93
    return lo + (hi - lo) * t / warm


@torch.compile(dynamic=False)
def anvil_math(w, g, n: int, fast, slow, energy, step_t, rates):
    beta = anvil_betas(step_t)
    g = g.float() / n
    fast_new = fast + (1 - beta) * (g - fast)
    slow_new = slow + (1 - 0.98) * (g - slow)
    x = g + beta * (fast_new - g)
    x = (x / (POLAR_MARGIN * torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-6)).to(BF16)
    for a, b_, c in ANVIL_MAPS:
        sq = x @ x.transpose(1, 2)
        x = a * x + (b_ * sq + c * (sq @ sq)) @ x
    update = x.float()
    energy_new = energy + (1 - 0.9) * (update.square().mean(1, keepdim=True) - energy)
    gained = update * torch.rsqrt(torch.clamp(energy_new, min=1e-10))
    update = gained * (torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
      / torch.clamp(torch.linalg.vector_norm(gained, dim=(1, 2), keepdim=True), min=1e-10))
    decay = ((slow_new * w) >= 0) * w * ANVIL_DECAY
    new = w - ANVIL_LR * rates[:, None, None] * (update + decay)
    fast.copy_(fast_new)
    slow.copy_(slow_new)
    energy.copy_(energy_new)
    w.copy_(new)
    return new.to(BF16)


def id_anvil_update(model, works, n):
    gathers = []
    for name, w, w_bf in (('wide', model.wide, model.wide_bf), ('square', model.square, model.square_bf)):
        works[name].wait()
        sl, st = model.shard[name], model.anvil[name]
        new = anvil_math(w[sl], model.owned[name], n, st['fast'][sl], st['slow'][sl], st['energy'][sl], model.step_t,
        model.rates[name][sl])
        gathers.append((dist.all_gather_into_tensor(w_bf, new, async_op=True), new))
    return gathers


@torch.compile(dynamic=False)
def adam_math(ps, gs, n: int, ms, vs, lrs: tuple, step_t, frames):
    t = step_t + 1
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    value_deltas = []
    for index, (p, g, m, v, lr) in enumerate(zip(ps, gs, ms, vs, lrs)):
        g = g / n
        if p.dtype == BF16:
            g = g.to(BF16).float()
        m_new = BETA1 * m + (1 - BETA1) * g
        v_new = BETA2 * v + (1 - BETA2) * g.square()
        if 8 <= index < 12:  # the value-embedding tables
            delta = lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)
            if SHAPING_ON:
                amplified = amplify_common_motion(delta)
                delta = torch.where(step_t < SHAPING_UPDATES, amplified, delta)
            value_deltas.append(delta)
        else:
            p.copy_((p.float() - lr * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)
    transferred = transfer_partner_motion(torch.stack(value_deltas), frames, step_t) if SHAPING_ON else value_deltas
    for i in range(4):
        ps[8 + i].copy_((ps[8 + i].float() - transferred[i]).to(ps[8 + i].dtype))


def adam_update(model, works, n, frames):
    works['flat'].wait()
    ps, gs, lrs = zip(*model.adam)
    pool_gate_credits(list(gs[4:8]), model.step_t)
    adam_math(list(ps), list(gs), n, model.m, model.v, lrs, model.step_t, frames)


def train_step(model):
    world, rank = model.world, model.rank
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]
    inputs, targets = batch[:, :-1].reshape(-1, ID_SEQ_LEN), batch[:, 1:].reshape(-1, ID_SEQ_LEN)
    rows_total = inputs.shape[0]
    per = rows_total // world
    idx, tgt = inputs[rank * per:(rank + 1) * per], targets[rank * per:(rank + 1) * per]
    loss, works, n = identity_grad(model, idx, tgt, rows_total)
    # Capture gathered current output maps before ANVIL/all-gather overwrites square_bf.
    transfer_frames = model.square_bf[3 * L + 1:4 * L:2].float()
    gathers = id_anvil_update(model, works, n)
    adam_update(model, works, n, transfer_frames)
    constrain_wte_gate_energy_compiled(model.wte, model.step_t)
    for work, _ in gathers:
        work.wait()
    model.refresh_bf16()
    model.loss.copy_(loss.view(1))
    model.step_t.add_(1)


class IdentityPhase:
    """The identity-path phase: compiled and captured as one CUDA graph on random tokens, then reset to its fresh
    state. bind_rows() puts its batches on the GPU before the timer; run() replays the graph once per update."""
    def __init__(self, device, rank, world):
        self.order = np.random.default_rng((SEED + 1, 0)).permutation(TRAIN_ROWS)
        model = self.model = Model(device, rank, world)
        snapshot = [x.clone() for x in model.tensors()]
        # Compile and capture the identity step on the random tokens model.rows holds until bind_rows.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                train_step(model)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            train_step(model)
        self.plain_graph = None
        if SPLIT_SHAPING:  # the same step without the shaping mechanisms (no-ops from update SHAPING_UPDATES on)
            global SHAPING_ON
            SHAPING_ON = False
            try:
                with torch.cuda.stream(side):
                    for _ in range(2):
                        train_step(model)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                self.plain_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.plain_graph, pool=self.graph.pool()):
                    train_step(model)
            finally:
                SHAPING_ON = True
        # Restore the fresh state: weights, optimizer state, bf16 copies and the update counter.
        with torch.no_grad():
            for x, saved in zip(model.tensors(), snapshot):
                x.copy_(saved)
            model.cast_all()
            model.step_t.zero_()
        torch.cuda.synchronize()
        dist.barrier()
    def bind_rows(self, rows):
        """Put the identity batches on the GPU: update i trains on rows order[BATCH * i:BATCH * (i + 1)], sorted by row
        index (the captured step reads model.rows[step_t])."""
        batches = rows[np.sort(self.order[:IDENTITY_UPDATES * BATCH].reshape(IDENTITY_UPDATES, BATCH), axis=1)]
        with torch.no_grad():
            self.model.rows.copy_(torch.from_numpy(batches))
    def config(self):
        """config.json "identity" entry. schedule_updates is the planned length that scales ANVIL's momentum warm-up;
        a shaping mechanism with `until` acts on identity updates 0..until-1."""
        return dict(updates=IDENTITY_UPDATES, schedule_updates=IDENTITY_SCHEDULE, batch=BATCH, sequence_len=ID_SEQ_LEN,
        init='prev8', value_output_pairing=dict(layers=list(range(L)), free_after_init=True),
        wte_gate_energy=dict(energy=WTE_GATE_ENERGY, channels=GATE_CHANNELS, init=True, until=SHAPING_UPDATES),
        value_embed_common_gain=dict(gain=VE_COMMON_GAIN, until=SHAPING_UPDATES),
        gate_credit_pooling=dict(layers=list(VE_BLOCKS), shared_component='head_mean_credit_only',
        until=SHAPING_UPDATES),
        value_embed_partner_transfer=dict(gain=VE_PARTNER_GAIN, donor='equal_window_partner', until=SHAPING_UPDATES),
        lr=LR, betas=[BETA1, BETA2], lr_wte=LR_WTE, lr_value_embeds=LR_VE, lr_lambdas=LR_LAMBDAS,
        anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD)
    def run(self):
        for i in range(IDENTITY_UPDATES):
            (self.plain_graph if self.plain_graph is not None and i >= SHAPING_UPDATES else self.graph).replay()
        self.model.gather_fp32()
    def close(self):
        self.graph = self.plain_graph = None
        self.model = None
        torch.cuda.empty_cache()


WARM_START = False  # Cut's ANVIL and Adam states start from the identity phase's (momenta, energies, moments)
WARM_START_SCALE = 1.0  # multiplies the carried momenta (fast, slow, Adam's first moment)


@torch.no_grad()
def warm_start_cut(idm, model, optimizer):
    """Copy the identity phase's optimizer states into Cut's. Identity matrices are stored as x @ W (Cut's weights
    transposed), except the MLP c_proj, stored as Cut's weight; ANVIL states are sharded across ranks there."""
    full = {}
    for name in ('square', 'wide'):
        sl = idm.shard[name]
        full[name] = {}
        for key, st in idm.anvil[name].items():
            out = torch.empty_like(st)
            dist.all_gather_into_tensor(out, st[sl].contiguous())
            full[name][key] = out
    def anvil_state(param_name):
        parts = param_name.split('.')
        i, kind = int(parts[2]), '.'.join(parts[3:5])
        if kind in SQUARE:
            index, name, flip = SQUARE.index(kind) * L + i, 'square', True
        elif kind == 'mlp.c_fc':
            index, name, flip = i, 'wide', True
        else:
            index, name, flip = L + i, 'wide', False
        f = lambda z: z[index].transpose(-1, -2) if flip else z[index]
        return f(full[name]['fast']), f(full[name]['slow']), f(full[name]['energy'])
    anvil, adam = optimizer.optimizers
    names = {id(p): n for n, p in model.named_parameters()}
    for params, state, lanes, ratio in anvil.batches:
        for k, p in enumerate(params):
            fast, slow, energy = anvil_state(names[id(p)])
            state['fast'][k].copy_(WARM_START_SCALE * fast)
            state['slow'][k].copy_(WARM_START_SCALE * slow)
            state['energy'][k].copy_(energy.reshape(state['energy'][k].shape))
    moments = {}
    for (p, _, _), m, v in zip(idm.adam, idm.m, idm.v):
        moments[id(p)] = (m, v)
    by_name = {'transformer.wte.weight': (idm.wte, False), 'lm_head.weight': (idm.lm_head, True),
               'resid_lambdas': (idm.lam_r, False), 'x0_lambdas': (idm.lam_0, False)}
    for i in idm.gates:
        by_name[f'transformer.h.{i}.attn.ve_gate.weight'] = (idm.gates[i], True)
        by_name[f'value_embeds.{i}.weight'] = (idm.ve[i], False)
    for group in adam.param_groups:
        for p in group['params']:
            source, flip = by_name[names[id(p)]]
            m, v = moments[id(source)]
            m, v = (m.T, v.T) if flip else (m, v)
            adam.state[p] = dict(step=torch.tensor(float(IDENTITY_UPDATES)),
                                 exp_avg=(WARM_START_SCALE * m).to(p.dtype).contiguous(),
                                 exp_avg_sq=v.to(p.dtype).contiguous().clone())


def difficulty_order(rows, settings):
    n, batch = (len(rows), settings.cut_batch)
    token_ids = rows.numpy()[:, 1:]
    counts = np.bincount(token_ids.reshape(-1), minlength=settings.model.vocab_size)
    surprisal = -np.log((counts.astype(np.float64) + 1.0) / (counts.sum() + len(counts)))
    score = np.empty(n, dtype=np.float64)
    for start in range(0, n, 256):
        score[start:start + 256] = surprisal[token_ids[start:start + 256]].mean(axis=1)
    ordered = np.argsort(score, kind='stable')
    rng = np.random.default_rng(SEED)
    strata = ordered.reshape(batch, n // batch)
    plan = np.stack([rng.permutation(group) for group in strata], axis=1)
    plan = np.take_along_axis(plan, np.argsort(rng.random(plan.shape), axis=1), axis=1)
    return plan.reshape(-1)


def train_stages(args, settings, device, rank, world):
    if world != 8:
        raise ValueError('This speedrun recipe requires eight GPUs')
    identity = IdentityPhase(device, rank, world)
    # The identity rows are left out of the Cut and held-out rows (one epoch over both phases).
    identity_indices = torch.from_numpy(identity.order[:IDENTITY_UPDATES * BATCH])
    keep = torch.ones(TRAIN_ROWS, dtype=torch.bool)
    keep[identity_indices] = False
    indices = torch.arange(TRAIN_ROWS)[keep]
    indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout_indices, train_indices = indices[-HOLDOUT:], indices[:-HOLDOUT]
    means = None
    horizon = len(train_indices) // settings.cut_batch  # every remaining row: 14648 - 8*300 - 64 = 8*1523
    stop = min(args.steps or horizon, horizon)
    train_count = stop * settings.cut_batch
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(
            json.dumps(
                dict(
                    script=__file__, world=world, seed=SEED, recipe=asdict(settings), steps=stop + IDENTITY_UPDATES,
                    overrides=args.set,
                    epoch_updates=horizon + IDENTITY_UPDATES,
                    train_rows=train_count + IDENTITY_UPDATES * BATCH,
                    upstream='qlabs-eng/noise@3343973b6f1af119effcdea2622f1fe672c51b1e',
                    additions=[
                        'identity_handoff_at_300', 'identity_init_and_early_update_shaping',
                        'difficulty_balanced_row_order', 'joint_partial_chain_roots', 'reduced_forward_replay',
                        'diagonal_readout_moments', 'x0_skip_embedding_credit', 'fused_forward_kernels',
                        'late_zero_anvil_decay', 'sketched_screened_merge', 'cut_key_degree2_moments',
                        'cut_mixing_scalar_contraction', 'cut_estimate_cuda_graph', 'resident_merge_scorer',
                        'response_ranked_mlp_routes', 'batched_update_telemetry',
                    ],
                    torch_version=torch.__version__,
                    merge_sketch=dict(columns=MERGE_SKETCH_COLUMNS, output_rank=MERGE_K, seed=SEED),
                    identity=identity.config(),
                    cut_anvil_decay=dict(
                        early_multiplier=1.0, late_multiplier=settings.late_cut_decay_multiplier,
                        late_after_updates=round(275 / 1043 * horizon), horizon=horizon,
                    ),
                ),
                indent=2,
            )
            + '\n'
        )

    def rank_slice(batch):
        per = len(batch) // world
        first = rank * per
        return batch[first : first + per].to(device)

    # Compile Cut's update on random tokens with a throwaway model, optimizer and Cut. The timed ones are rebuilt from
    # seed below, and the timed Cut captures its estimate graph on the same tokens.
    noise = torch.randint(0, settings.model.vocab_size, (settings.cut_batch, settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(SEED))
    model = make_model(device, SEED, settings.model)
    optimizer = make_optimizer(model, settings, horizon)
    estimator = Cut(model, settings, SEED, rank, world)
    batch = rank_slice(noise)
    inp, target = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    for _ in range(4):
        estimator.step(inp, target, optimizer)
        torch.cuda.synchronize()
    estimator.close()
    del model, optimizer, estimator, inp, target, batch
    gc.collect()
    torch.cuda.empty_cache()
    model = make_model(device, SEED, settings.model)
    optimizer, estimator = (
        make_optimizer(model, settings, horizon),
        Cut(model, settings, SEED, rank, world),
    )
    graph_batch = rank_slice(noise)
    estimator.bind_flat_layout(optimizer)  # (FLAT_GRADIENTS: the graph packs the gradients in step's layout)
    estimator.capture_estimate_graph(graph_batch[:, :-1].contiguous(), graph_batch[:, 1:].contiguous())
    log = (args.output / 'metrics.jsonl').open('w') if rank == 0 else None
    scorer = prepare_resident_merge_score(model, settings, rank, world)
    del noise, graph_batch
    # Untimed, as in #44 (qlabs-eng/noise@3343973b, train_stages): read train.pt, split it by the seeded permutations
    # above and put the identity phase's batches on the GPU. No split or order here depends on the rows' contents;
    # load_rows only verifies their pinned hash.
    all_rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    cut_rows = all_rows.index_select(0, train_indices)
    holdout_rows = all_rows.index_select(0, holdout_indices)
    identity.bind_rows(all_rows.numpy())
    del all_rows
    torch.cuda.reset_peak_memory_stats()
    begin_training()
    start = time.perf_counter()
    try:
        order = torch.from_numpy(difficulty_order(cut_rows, settings))  # timed: it reads the rows' tokens
        train_indices = train_indices[order]
        unused_indices = train_indices[train_count:]
        train_indices = train_indices[:train_count]
        training = cut_rows.index_select(0, order[:train_count])
        del cut_rows
        training = training.to(device)
        all_used = torch.cat([identity_indices, train_indices, holdout_indices, unused_indices])
        assert len(torch.unique(all_used)) == len(all_used) == TRAIN_ROWS
        if rank == 0:
            digest = lambda ids: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            usage = dict(identity_rows=len(identity_indices), cut_rows=train_count, merge_rows=len(holdout_indices),
                         unused_rows=len(unused_indices), unused_sha256=digest(unused_indices), disjoint=True,
                         identity_sha256=digest(identity_indices), cut_sha256=digest(train_indices),
                         merge_sha256=digest(holdout_indices))
            (args.output / 'row_usage.json').write_text(json.dumps(usage, indent=2) + '\n')
        identity.run()
        with torch.no_grad():
            weights = model.state_dict()
            for name, value in identity.model.state_dict().items():
                weights[name].copy_(value)
            if WARM_START:
                warm_start_cut(identity.model, model, optimizer)
        identity_loss = average_ranks(identity.model.loss.clone()).item()
        identity.close()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record = dict(step=IDENTITY_UPDATES, phase='identity', phase_step=IDENTITY_UPDATES,
                      seqs=IDENTITY_UPDATES * BATCH, tokens=IDENTITY_UPDATES * BATCH * settings.model.sequence_len,
                      train_loss=identity_loss, train_seconds=round(elapsed, 2))
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
        pending_records = []
        for step in range(1, stop + 1):
            begin, end = (step - 1) * settings.cut_batch, step * settings.cut_batch
            set_rate(optimizer, step, horizon, settings.warmdown, settings.final, settings.cut_warmup)
            batch = rank_slice(training[begin:end])
            prof_step = int(os.environ.get('PROF_CUT_STEP', 0))
            if prof_step and step == prof_step:  # screen: profile 8 Cut updates on rank 0
                from torch.profiler import ProfilerActivity, profile
                profiler = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
                profiler.__enter__()
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            if prof_step and step == prof_step + 7:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                if rank == 0:
                    print(profiler.key_averages().table(sort_by='cuda_time_total', row_limit=45,
                                                        max_name_column_width=70), flush=True)
            record = dict(step=step + IDENTITY_UPDATES, phase='cut', phase_step=step,
                          seqs=end + IDENTITY_UPDATES * BATCH,
                          tokens=(end + IDENTITY_UPDATES * BATCH) * settings.model.sequence_len, train_loss=loss,
                          peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1000000000.0, 3))
            pending_records.append(record)
            flush = len(pending_records) >= settings.telemetry_batch or step == stop
            if flush:
                if not bool(torch.stack(estimator._telemetry_health).all()):
                    raise FloatingPointError('Non-finite estimator credit')
                estimator._telemetry_health.clear()
                losses = torch.stack([entry['train_loss'] for entry in pending_records])
                losses = (average_ranks(losses) if LAZY_LOSS else losses).cpu().tolist()
                if any(not math.isfinite(value) for value in losses):
                    raise FloatingPointError('Non-finite training loss')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                for entry, measured_loss in zip(pending_records, losses):
                    entry['train_loss'] = measured_loss
                    entry['train_seconds'] = round(elapsed, 2)
                if rank == 0:
                    records_text = ''.join(json.dumps(entry) + '\n' for entry in pending_records)
                    print(records_text, end='', flush=True)
                    log.write(records_text)
                    log.flush()
                pending_records.clear()
            # Weight averages for the end merge: every MERGE_EVERY updates (and the last) from MERGE_FROM of the rows on.
            if (step % MERGE_EVERY == 0 or step == stop) and begin / len(training) >= MERGE_FROM:
                with torch.no_grad():
                    if means is None:
                        means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                    else:
                        for a, mean in zip(MERGE_DECAYS, means):
                            for m, q in zip(mean, model.parameters()):
                                m.lerp_(q.detach().float(), a)
        extra = {}
        if means:
            merge_start = time.perf_counter()
            extra.update(end_merge(model, means, holdout_rows, rank, scorer, settings))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
            means = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        end_training()
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        result = dict(
            val_loss=evaluate(state, args.data_dir), train_seconds=round(elapsed, 2), steps=stop + IDENTITY_UPDATES,
            identity_steps=IDENTITY_UPDATES, cut_steps=stop, seqs=train_count + IDENTITY_UPDATES * BATCH,
            tokens=(train_count + IDENTITY_UPDATES * BATCH) * settings.model.sequence_len,
            peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1000000000.0, 3), **extra,
        )
        if rank == 0:
            if not args.no_ckpt:
                torch.save(state, args.output / 'final.pt')
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    finally:
        scorer.close()
        estimator.close()
        if log is not None:
            log.close()


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--no-ckpt', action='store_true')
    parser.add_argument('--steps', type=int,
                        help='Cut updates to run after the identity phase; preserves the full-epoch LR schedule.')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                        help='screening overrides: a Settings field (lower case) or a module constant (upper case)')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('steps must be positive')
    settings = Settings()
    for item in args.set:
        key, text = item.split('=', 1)
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text
        if key.isupper():
            if key not in globals():
                parser.error(f'unknown constant {key}')
            globals()[key] = value
        else:
            if not hasattr(settings, key):
                parser.error(f'unknown setting {key}')
            setattr(settings, key, value)
    settings.__post_init__()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.distributed.init_process_group('nccl', device_id=device)
    train_stages(args, settings, device, rank, world)
    torch.cuda.synchronize()
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
