"""Boost -> Cut with the true-path readout: the cross-token credit of blocks 0-6 is measured forward along each token's
real path and scored at later measured hubs instead of being carried down the residual identities; trained at batch 4
on two GPUs per sequence with record #2's recipe and merged at the end along its own trajectory.

Record #2 (#31's estimator) measures each block's mid-stream credit with the chain: a token's stream is moved along a
complete Hadamard cycle and run forward to the loss with the other tokens' keys and values held (its own-query path).
What reaches the loss through other tokens (token t's key, value or gate at a later block j moving the queries s > t
that attend to t) is measured at block j's input by the value / input-K / gate probes and then carried down to the
earlier blocks along the residual identities, which loses most of it. This record measures it at its source. Every
reading is a forward experiment scored once against a measured zero-step hub; a decoded credit feeds only linear
consumers or, through the outer product, its own source site, and never becomes another probe's target:
  - Readout (blocks 0-6, wte). The chain's moves also carry each token's own key / value change (the self-edge), and
    at every later block j the moved stream's change of t's key / value / gate is formed forward (RMSNorm, the
    projections, RoPE, QK-norm, the sigmoid gate) and turned into the exact single-column softmax change of every query
    s > t in the window, scored against block j's measured hub chain_j W_proj (#31's future-probe scorer). Decoded over
    the root's cycle: R_i, the cross-token credit along the true path, in place of the identity transport for
    attn.c_proj, mlp.c_proj, c_v / the value embeddings and the mixing scalars of blocks 0-6; the embedding root's
    readout joins wte's credit. The roots run from the top down, so every hub is measured before a lower root reaches
    it.
  - Sources (c_q and c_fc of blocks 0-6): their own forward experiments, pushed up the same real path and read out the
    same way. q: the raw query moved along a complete cycle (block 0) or along the subspace that moves each head's 16
    strongest keys' logits (blocks 1-6), through its exact softmax, c_proj and its block's MLP. c_fc: each token's
    strongest active units (a cycle of 512 at blocks 0 and 4, 128 elsewhere), through relu^2 and the writer.
  - Receivers (c_k and ve_gate of blocks 0-6): for each key token and head its 8 strongest receivers; raw-key moves
    along the dual basis of their logit pull-backs (each moves one receiver's logit to first order) and gate-logit moves,
    each turned into the receiver's exact single-column output change and pushed up the receiver's real path.
  - In-roots (blocks 5 and 7): a complete cycle of the block input through the token's own query (RMSNorm, W_q, RoPE,
    QK-norm, exact softmax, c_proj) scored against chain_j: the own-query path the out credit lam_j chain_j leaves out.
    Its decoded credit feeds only linear consumers, mlp.c_proj 4 / 6 and block 5's and 7's mixing scalars. The c_fc
    4 / 6 source rows are also scored along this path (their own moves at block 5's / 7's input, against the same
    measured chain_j); c_fc never receives the in-root credit.
  #31's future-token probes are gone: the sources and the embedding root's readout replace them.

Speed (nothing here changes what is measured; each piece matched its exact form to rounding before the run): the
receivers s > t are scored from per-token second-order moment tables of each block (s = t exactly), and so are the
input-K and raw-K experiments; the chain head's log-sum-exp change to second order (exact arithmetic on any row block
whose bound reaches 0.1); exact MLP unit pruning; empty receiver slots skipped; Triton kernels for the scorer, the
own-query step, the sources, the in-roots, the receivers' plan and the routes.

Record #2's recipe with three changes: Boost for 6.68% of the epoch (#2: 20%), Boost's head credit the mean of its
2L - 1 readouts (#2: their sum), ANVIL weight decay 0.75 (#2: 2.25). Cut at batch 4, each sequence split over two GPUs
(each takes half of every cycle of moves and draws; the pair adds its parts), ANVIL II on the block matrices and Adam
elsewhere, a warmdown over the last 40% to 30% of each rate, the LM head initialized at std 0.02, and the end merge:
the last 32 training sequences feed no update and score it; from 60% of the epoch, every 5 updates, nine exponential
means of the weights are kept, and after the last update each part (each block matrix, each table, the scalars) moves
toward one mean, chosen by the held-out loss (about 5,000 forward passes on 32 sequences, inside the timed region).
Block 0's c_fc source cycle is Settings.readout_fc_cycle[0].

    uv run torchrun --standalone --nproc_per_node=8 train/val/run.py
"""
import argparse
import contextlib
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # the allocator the record ran with
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, flex, has_ve, norm, window_mask
from utils.prepare_data import TRAIN_ROWS, load_rows
from utils.eval import evaluate

SEED = 0
EPS = torch.finfo(torch.float32).eps
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


def model_config():
    """The fixed GPT topology used by the current val experiment."""
    return GPTConfig(sequence_len=2048, vocab_size=4096, n_layer=8, n_head=8,
                     n_kv_head=8, n_embd=512, window_pattern='SSSL')


@dataclass
class Settings:
    """Own-query/value-path credit, full hidden FD, independent input-K / gate probes, and the true-path readout."""
    model: GPTConfig = field(default_factory=model_config)
    estimator: str = 'cut'
    boost_hub: str = field(default='projected', init=False)  # persisted algorithm version, not an ablation flag
    mlp_output_credit: str = field(default='next_mid', init=False)
    switch: float = 0.0668
    boost_batch: int = 4
    cut_batch: int = 4
    projection_std: float = 1.5
    lr: float = 0.0021
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.4
    final: float = 0.3  # the warmdown ends at this fraction of each rate; the end merge anneals the rest
    head_std: float = 0.02  # the LM head's init std (the fixed init's 0.001)
    anvil_lr: float = 0.026
    anvil_wd: float = 0.75
    fast_weight: float = 1.0  # ANVIL fast rail
    polar_margin: float = 1.1
    sigma: float = 0.005
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    nrep: int = 64
    emb_chunk: int = 16
    scalar_step: float = 0.01
    chunk: int = 2 ** 27
    chain_sigma: float = 0.01
    route_k: int = 512
    chain_tokens: int = 512  # the gate probes' token chunk (an execution chunk)
    key_path_draws: int = 512  # independent input-K central pairs per token
    key_path_sigma: float = 0.01  # relative input RMS step
    gate_path_sigma: float = 0.01  # relative input RMS step
    query_keys: int = 256  # the own query's routed keys per head (the strongest clean logits move)
    key_sigma: float = 0.003
    readout_q_keys: int = 16  # q sources: per head the subspace moving its this many strongest keys' logits
    readout_q_full: tuple = (0,)  # blocks whose q source runs the complete 512-direction cycle instead
    readout_fc_cycle: tuple = (512, 128, 128, 128)  # c_fc source cycle of blocks 0-3 (its cycle - 1 strongest active
    #   units); block 0: 512, or 256 (the 255 strongest units instead of 511: faster)
    readout_hi_fc_cycle: tuple = (512, 128, 128)  # c_fc source cycle of the hi blocks
    readout_sigma: float = 0.01  # q / c_fc source steps (relative RMS of the raw query / hidden pre-activations)
    readout_k_blocks: tuple = (0, 1, 2, 3)  # blocks 0-3 whose c_k (and ve_gate, if any) take the receiver experiments
    readout_k_receivers: int = 8  # R: receivers per key token and head
    readout_k_cap: float = 1.5  # rows per receiver token and group: this x the mean load 8 R (rounded up to 16, >= 64)
    readout_hi_blocks: tuple = (4, 5, 6)  # blocks past 3 with roots, sources and receivers (4 .. n_layer - 2)
    readout_in_roots: tuple = (5, 7)  # blocks j whose input own-query path goes into out_{j-1} (j - 1 a hi block)
    readout_chunk: int = 512  # tokens per chunk of the chain and the pass (an execution chunk)

    def __post_init__(self):
        width, hidden = self.model.n_embd, 4 * self.model.n_embd
        if self.estimator not in ('boost', 'cut') or not 0 <= self.switch <= 1:
            raise ValueError('Invalid stage estimator or Boost fraction')
        if not 0 <= self.fast_weight <= 1:
            raise ValueError('ANVIL fast weight must be in [0,1]')
        if width < 1 or width & (width - 1):
            raise ValueError('Full cycles require power-of-two model width')
        if not 1 <= self.route_k <= hidden:
            raise ValueError('Invalid sparse route budget')
        if self.query_draws != width // self.model.n_head:
            raise ValueError('Queries use one complete cycle per head')
        if not 0 < self.query_sigma < float('inf'):
            raise ValueError('Query step must be finite and positive')
        if not 0 < self.gate_path_sigma < float('inf'):
            raise ValueError('Gate input probes require a finite positive step')
        if not isinstance(self.query_keys, int) or self.query_keys < 0:
            raise ValueError('Own-query key count must be a nonnegative integer')
        if type(self.key_path_draws) is not int or not 0 <= self.key_path_draws <= width:
            raise ValueError('Input-K pairs must be an integer between zero and model width')
        if not 0 < self.key_path_sigma < float('inf'):
            raise ValueError('Input-K step must be positive')
        if any(v <= 0 for v in (self.nrep, self.emb_chunk, self.chain_tokens, self.chunk,
                               self.sigma, self.fc_sigma, self.chain_sigma, self.key_sigma, self.scalar_step)):
            raise ValueError('Probe sizes and batching limits must be positive')
        hb, cycles = self.readout_hi_blocks, self.readout_fc_cycle + self.readout_hi_fc_cycle
        if ((width, width // self.model.n_head) != (512, 64) or self.query_keys not in (64, 128, 256)
                or hb != tuple(range(4, self.model.n_layer - 1)) or len(cycles) != len(hb) + 4
                or any(c < 32 or c & (c - 1) for c in cycles) or not {j - 1 for j in self.readout_in_roots} <= set(hb)):
            raise ValueError('The readout kernels need width 512 in heads of 64 and query_keys 64, 128 or 256; the '
                             'hi blocks run from 4 to n_layer - 2, one power-of-two c_fc cycle of at least 32 per '
                             'block 0-6, each in-root above a hi block')


# ---------------------------------------------------------------------------------------------------------- helpers


@torch.compile
def coordinate_differences(raw, y, sigma):
    """Finite differences of each token's loss along each of its own logits: for every token and every logit j,
    the loss with raw logit j moved up by sigma minus the loss with it moved down by sigma, over 2 sigma.

    Each of the two is a full forward evaluation of the soft-capped cross-entropy at the moved logit, written in
    closed form: moving one logit changes the partition function Z = sum_k exp(z_k) by one term, so the moved
    loss differs from the clean loss by log(Z - exp(z_j) + exp(z_j')) - log Z = log1p(exp(z_j) / Z * expm1(z_j' - z_j))
    in the partition and by -(z_j' - z_j) at the target logit. Nothing is differentiated: exp(z_j) / Z are the
    clean forward's probabilities, used only to evaluate the perturbed losses. Zero at ignored tokens (y < 0)."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)  # exp(z_j) / Z of the clean forward
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))  # loss(z_j -> new) - loss(z), partition
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))  # the target logit's own term
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1, built once per process."""
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)  # +-1 is exact in bfloat16
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


def attention_state(qraw, kraw, values, cos, sin, nhead, window):
    """Dense clean forward context for independently scored local attention trials."""
    def heads(raw):
        shaped = raw.reshape(*raw.shape[:2], nhead, -1)
        return norm(apply_rotary_emb(shaped, cos, sin)).transpose(1, 2)
    q, k = heads(qraw), heads(kraw)
    pos = torch.arange(qraw.shape[1], device=qraw.device)
    lag = pos[:, None] - pos[None, :]
    allowed = (lag >= 0) & (lag <= window)
    a = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -.5).masked_fill(
        ~allowed, -torch.inf).softmax(-1)
    return q, k, a, a @ values


def chunk_size(draws, cap):
    """Largest divisor of draws up to cap."""
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    """The attention each token would see if only its own state moved: every query attends to the other positions'
    clean keys and values and to its own moved ones, by swapping its own clean key and value through the
    log-sum-exp of one attention over the clean keys and values. Tensors are [batch, token, head, dim]."""
    if not q.is_cuda:
        t = q.shape[1]
        lag = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
        logits = q.transpose(1, 2) @ k_clean.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** .5
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), -torch.inf)
        own = (q * k_own).sum(-1).transpose(1, 2) / q.shape[-1] ** .5
        logits = logits.diagonal_scatter(own, dim1=-2, dim2=-1)
        weights = logits.softmax(-1)
        out = (weights @ v_clean.transpose(1, 2)).transpose(1, 2)
        return out + weights.diagonal(dim1=-2, dim2=-1).transpose(1, 2).unsqueeze(-1) * (v_own-v_clean)
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = flex(q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask,
                    return_lse=True)
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)


frozen_attention_compiled = torch.compile(frozen_attention)


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch.compile(dynamic=False)
def query_probe_heads(raw, a, sigma, cos, sin):
    """Preserve the original finite norm calculation, then group probe heads."""
    moved = moved_heads(raw, a, sigma, cos, sin)
    batch, length, heads, width = raw.shape
    if raw.is_cuda and batch > 1:
        return moved.view(len(a), batch, length, heads, width).permute(1, 3, 0, 2, 4).reshape(
            batch, heads*len(a), length, width)
    return moved


def query_probe_attention(moved, keys, values, window, count, batch):
    """All probes share clean K/V; group probe heads instead of copying K/V."""
    _, length, heads, width = keys.shape
    if not moved.is_cuda or batch == 1:
        repeat = lambda z: z.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*batch, length, heads, width)
        return attention(moved, repeat(keys), repeat(values), window).view(count, batch, length, heads, width)
    out = flex(moved, keys.transpose(1, 2), values.transpose(1, 2),
               block_mask=window_mask(window, length, moved.device), enable_gqa=True)
    return out.view(batch, heads, count, length, width).permute(2, 0, 3, 1, 4)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    """Score either moved-minus-clean or plus-minus-minus by ordinary broadcasting."""
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


def decode_local_scores(scores, directions):
    """Sum scalar probe scores times directions; the caller owns budget normalization.

    scores [..., P], directions [..., P, D] (or shared [P, D]).
    Keep this reduction inside the caller's compiled kernel, without creating
    a materialized score-times-direction tensor.
    """
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def decode_local_response(scores, steps, directions, central=True):
    """Decode independent trials; one-sided scores must already subtract clean."""
    if central:
        probes = directions.shape[-2]
        directional = (scores[..., :probes] - scores[..., probes:]) / (2 * steps)
    else:
        directional = scores / steps
    return decode_local_scores(directional, directions)


@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    """Collapse the full Hadamard central-difference cycle for the separable hidden objective.

    H.T @ H / k is identity, so decoding leaves (g @ W_out) times each unit's
    scalar finite difference. Keep that finite step, including ReLU crossings;
    contract in FP32 instead of rounding every direction's projected response to BF16.
    u and out_weight contain all units (or a GPU's contiguous slice).
    """
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope


@torch.compile(dynamic=False)
def embedding_moves(e, steps, rows, sign, wq, wk, wv, cos, sin):
    n, (b, t, width) = rows.shape[0], e.shape
    hd = 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + sign * steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)
    heads = lambda z: z.view(n * b, t, -1, hd)
    q = norm(apply_rotary_emb(heads(F.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(F.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(F.linear(x0, wv))


@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)


@torch.compile(dynamic=False)
def stage_readout(x0, out, mix, wproj, weight, y):
    xmid = mix * x0 + F.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)


def norm_difference(x, delta):
    """norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves."""
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / x.shape[-1] + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    """PR #23's central forward RMSNorm response along v.

    RMSNorm's Jacobian is symmetric: this finite directional response tends
    to its credit pullback. This is one nonlinear step, not an analytic VJP.
    The tiny floor also defines a finite response for a zero input or credit.
    """
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
                 / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    """Token chunks share the direction projections and squared lengths."""
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


KEY_BUCKET = 256


@torch.compile(dynamic=False)
def raw_key_moves(kp, raw, steps, cos, sin, basis):
    moved = raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    half = raw.shape[-1] // 2
    moved = torch.cat((moved[..., :half] * cos + moved[..., half:] * sin,
                       -moved[..., :half] * sin + moved[..., half:] * cos), -1)
    return moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)


@torch.compile(dynamic=False)
def decode_key_pairs(scores, steps, rows):
    return decode_local_response(scores.sum(0).T, steps, rows) / len(rows)


def norm_delta(x, delta):
    width = x.shape[-1]
    eps = torch.finfo(x.dtype).eps
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + eps)
    moved_inverse = torch.rsqrt(
        (square + 2 * (x * delta).sum(-1, keepdim=True)
         + delta.square().sum(-1, keepdim=True)) / width + eps)
    return delta * moved_inverse + x * (moved_inverse-inverse)


def rotate(x, cos, sin):
    half = x.shape[-1]//2
    return torch.cat((x[..., :half]*cos+x[..., half:]*sin,
                      -x[..., :half]*sin+x[..., half:]*cos), -1)


def fused_key_response(q, dk, A, C, source_start, query_start, window):
    """The raw-K experiments' scores [h, P, M] (the record's key_response, one-pass TF32 dots, 2 trials x 32 sources
    per program)."""
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    bs, tp, warps, stages = KEY_TILES
    even = q.shape[1] % 64 == 0 and probes % tp == 0 and sources % bs == 0
    key_kernel[(triton.cdiv(probes, tp), triton.cdiv(sources, bs), heads)](
        q, dk, A, C, out, heads, q.shape[1], sources, probes, width, *dk.stride(), *q.stride(), *A.stride(),
        *C.stride(), source_start, query_start, window, bs, tp, 64, 'tf32', even, num_warps=warps,
        num_stages=stages, enable_fp_fusion=False)
    return out


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames = {}
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.blocks = {}, {}, {}
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

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws start..start + n of this step's world * draws at `site`: rows of a sign-flipped Hadamard matrix,
        orthogonal across chunks and GPUs within each cycle."""
        if site not in self.frames:
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        self.capturing = True
        try:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma=0.1):
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)


class Boost:
    """Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        if model.config.n_head != model.config.n_kv_head or model.config.n_layer % 2:
            raise ValueError('Stage estimators require equal Q/KV heads and an even number of blocks')
        if any(k % split for k in (model.config.n_embd, settings.query_draws)):
            raise ValueError('Readout and query cycles must divide across the sequence group')
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.shared_generator = self.capture.shared_generator
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)
        self.maps = {}

    def close(self):
        self.capture.close()

    def share(self, total):
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor):
        if self.split > 1:
            torch.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def probe(self, x, y, mix=None, probed=None):
        """Each token's readout credit in the probed space, measured along a full cycle of the step's basis; the
        GPUs of a sequence each take a slice of the cycle and add their parts."""
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        offset = torch.randint(0, self.width, (n, 1), device=self.device, generator=self.generator)  # each token's
        # first row (used only by partial cycles; drawn either way so the random stream is the same)
        credit = torch.zeros(n, self.width, device=self.device)
        lo, hi = self.share(self.width)
        C, stream_dirs, dir_sq, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ Hs / self.width
        return self.group_sum(credit)

    def embedding_block0(self, e, y, x):
        """The embedding's credit for the first attention stage's readout along each token's own path (every query
        sees the other positions' clean keys and values and its own moved ones)."""
        model = self.model
        cfg = model.config
        attn = model.transformer.h[0].attn
        b, t = x.shape
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        window = model.window_sizes[0][0]
        mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
        weight = model.lm_head.weight.float()[:cfg.vocab_size]
        W = {name: getattr(attn, 'c_' + name).weight.float() for name in ('q', 'k', 'v', 'proj')}
        e = e.float().view(b, t, self.width)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, self.width, device=self.device)
        n = chunk_size(self.width // self.split, self.s.emb_chunk)
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, self.width).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, self.width), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(self.width)
        for j0 in range(lo, hi, n):
            rows = H[j0:j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / self.width).reshape(b * t, self.width)

    def attention_maps(self, i):
        """Clean attention A[b, h, query, source]; no matched-filter matrix is needed."""
        if i in self.maps:
            return self.maps[i]
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (norm(apply_rotary_emb(d.outputs[prefix + name].float().view(b, t, h, hd), cos, sin))
                for name in ('c_q', 'c_k'))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        window = model.window_sizes[i][0]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        A = logits.softmax(-1)
        del logits
        self.maps[i] = A.to(torch.bfloat16)
        return self.maps[i]

    def query_errors(self, draws, targets, layers=None):
        """A complete per-head query cycle, scored against each query's own output target."""
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        n = chunk = chunk_size(draws, self.s.nrep)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = attention(processed(q_raw).to(dt), kp.to(dt), values.to(dt), window)
            fixed, vals = kp.to(dt), values.to(dt)
            raw = q_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = query_probe_heads(raw, a, sigma, cos, sin)
                out = query_probe_attention(moved, fixed, vals, window, n, b)
                estimate += code_estimate(head_scores(out, baseline, target), a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors

    def key_context(self, i, sequence, target):
        """Clean FP32 single-key score context, shared by raw-K and input-K experiments."""
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd//2]*cos + z[..., hd//2:]*sin,
                                   -z[..., :hd//2]*sin + z[..., hd//2:]*cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix+name][sequence].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][sequence].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix+'ve_gate'][sequence].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        logits = (qp @ kp.transpose(1, 2)) / hd**.5
        A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
        hub = target[sequence].view(t, h, hd).transpose(0, 1)
        C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
        return qp, kp, k_raw, cos, sin, A, C

    def exact_key_errors(self, targets, layers=None):
        """Key credit, one local step from the targets, by single-key experiments (key_scores): exact up to the
        one-sided step, with no interference between positions. The GPUs of a sequence split the directions."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                credit_all[r] = self.key_errors_from_context(i, self.key_context(i, r, targets[i]))
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def key_errors_from_context(self, layer, context):
        """Raw-K finite experiments, sharing the input-K context and skipping masked query blocks."""
        qp, kp, raw, cos, sin, A, C = context
        h, t, hd = raw.shape
        lo, hi = self.share(hd)
        result = torch.zeros_like(raw)
        eye = torch.eye(hd, device=self.device)
        steps = self.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        for t0 in range(0, t, KEY_BUCKET):
            t1 = min(t0 + KEY_BUCKET, t)
            q1 = min(t, t1 + self.model.window_sizes[layer][0])
            source, query = slice(t0, t1), slice(t0, q1)
            moves = raw_key_moves(kp[:, source], raw[:, source], steps[:, source],
                                  cos[source], sin[source], eye[lo:hi])
            scored = fused_key_response(qp[:, query], moves, A[:, query, source], C[:, query, source],
                                       t0, t0, self.model.window_sizes[layer][0])
            result[:, source, lo:hi] = scored.transpose(1, 2) / steps[:, source]
        return result

    def value_credit(self, i, target):
        """Fixed-attention value credit; all query tokens contribute, with no discount."""
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        A = self.attention_maps(i)  # A[b, h, query, source]
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return (A.float().transpose(-1, -2) @ hub).transpose(1, 2)

    def exact_values(self, i, target, value_target=None, raw_credit=None):
        """Layer i's value and gate errors, the local step in closed form: a value move at t reaches query s's output
        as A[s, t] times the move, so the value credit is sum_s A[s, t] target_s; a gate move scales the value
        embedding by the gate's slope (the record's central difference, odd in the move)."""
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        credit = self.value_credit(i, target) if raw_credit is None else raw_credit
        # A corrected target has already crossed RMSNorm. Only V/VE may use it:
        # the gate's sigmoid experiment must retain the directly measured target.
        value_credit = credit if value_target is None else self.value_credit(i, value_target)
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': value_credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """All hidden units' local central differences, partitioned across a sequence's GPUs."""
        # Preserve full-state checkpoint RNG advancement from the retired hidden cycle.
        self.signs(256)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        lo, hi = self.share(u.shape[-1])
        local = hidden_local_credit(u[:, lo:hi], steps, g, out_weight[:, lo:hi].float())
        return local if self.split == 1 else F.pad(local, (lo, u.shape[-1] - hi))

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def internals(self, gradients, targets, b, t, value_targets=None, key_errors=None, raw_values=None):
        """Q/K/gate use measured targets; only linear V/VE paths accept corrected targets."""
        d, s = self.capture, self.s
        errors = self.query_errors(s.query_draws // self.split, targets)
        errors = {site: error / self.split for site, error in errors.items()}
        errors.update(self.exact_key_errors(targets) if key_errors is None else key_errors)
        for i, target in targets.items():
            value_target = None if value_targets is None else value_targets[i]
            raw = None if raw_values is None else raw_values[i]
            errors.update({site: error / self.split
                           for site, error in self.exact_values(i, target, value_target, raw).items()})
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, self.model.config.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        self.maps.clear()
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        x0 = d.x0
        targets, streams = {}, []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i]
            xmid = xin + d.outputs[p + 'attn.c_proj']
            xout = xmid + d.outputs[p + 'mlp.c_proj']
            streams.append((xin, xmid, xout))
            xm, xo = xmid.float().flatten(0, 1), xout.float().flatten(0, 1)
            block = model.transformer.h[i]
            att_in = d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            g_mid = self.probe(xm, yf)
            gradients[p + 'attn.c_proj.weight'] = g_mid.T @ att_in
            # One local linear step from the measured stream credit (PR #15).
            hub = g_mid @ block.attn.c_proj.weight.float()
            # Keep later probes on the same RNG stream as the retired hub cycle.
            self.signs(self.width)
            torch.randint(0, self.width, (xm.shape[0], 1), device=self.device, generator=self.generator)
            targets[i] = hub.view(b, t, self.width)
            g_out = self.probe(xo, yf)
            gradients[p + 'mlp.c_proj.weight'] = g_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, g_out, model.transformer.h[i].mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            if i < L - 1:  # the head learns at every intermediate readout too
                for stream in (xm, xo):
                    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + EPS)
                    error = coordinate_differences(normed @ weight.T, yf, 0.1)
                    gradients['lm_head.weight'][:weight.shape[0]] += error.T @ normed
        self.internals(gradients, targets, b, t)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, self.width))
        scalars = torch.zeros(2, L, device=self.device)
        step = s.scalar_step
        for i in range(self.part, L, self.split):
            x_prev = x0 if i == 0 else streams[i - 1][2]
            ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
            lr_, l0 = model.resid_lambdas[i], model.x0_lambdas[i]
            for which in range(2):
                up = self.stage_losses(i, lr_ + step * (which == 0), l0 + step * (which == 1), x_prev, x0, ve, y)
                down = self.stage_losses(i, lr_ - step * (which == 0), l0 - step * (which == 1), x_prev, x0, ve, y)
                scalars[which, i] = (up - down).sum() / (2 * step)
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        gradients['lm_head.weight'] = gradients['lm_head.weight'] / (2 * L - 1)  # D24 J2: the mean of its readouts
        if self.part:  # computed in full by every GPU of the sequence: kept by its first GPU only
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight'):
                    gradients[name] = torch.zeros_like(gradients[name])
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

    def step(self, x, y, optimizer):
        gradients = self.estimate(x, y)
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        sequences = self.world // self.split
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if self.world > 1:
            torch.distributed.all_reduce(flat)
        flat /= sequences
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite estimate')
        offset = 0
        for name, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        self.maps.clear()
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()


class Cut(Boost):
    """Own-query forward loss probes, PR #23's value path, the input-K / gate probes and the true-path readout."""

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        # split > 1 = the GPUs of a sequence each take a share of every cycle of moves and draws
        super().__init__(model, settings, seed, rank, world, split=split, group=group)

    def value_path_credit(self, raw_values):
        """Independent measured mid credits -> V -> one RMSNorm step: each block input's value-path credit.

        Never feed accumulated extra credit into another norm or local nonlinear
        experiment. Reuse this capture's attention maps later for V/gate credit.
        """
        model, d = self.model, self.capture
        inputs = {}
        for i in reversed(range(model.config.n_layer)):
            ev = raw_values[i].reshape(-1, self.width)
            inputs[i] = norm_move(d.blocks[i].float().flatten(0, 1), ev @ model.transformer.h[i].attn.c_v.weight.float())
        return inputs

    def key_input_credits(self, targets):
        """Independent central probes at each block input, moving only its K branch, and the raw-K experiments, scored
        from the moment tables of each block and sequence (the receivers s > t to second order, s = t exactly).

        Use a private copy of the saved probe RNG so these optional draws neither
        shift Q/hidden probes nor require another checkpoint RNG stream.
        Partial Hadamard cycles use randomized rows and column signs.
        """
        d, s = self.capture, self.s
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        raw_errors = {}
        for i, target in targets.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)[:s.key_path_draws]
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator)*2-1
            rows = frame[indices] * signs
            total_rows = len(rows)
            r0, r1 = self.share(total_rows)  # this GPU's share of the draws
            rows = rows[r0:r1]
            estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
            raw_credit = torch.empty_like(estimate)
            projected_rows = rows @ self.model.transformer.h[i].attn.c_k.weight.float().T
            window = self.model.window_sizes[i][0]
            for r in range(d.batch):
                qp, kp, raw, cos, sin, A, C = self.key_context(i, r, target)
                xin = d.blocks[i][r].float()
                w_key = self.model.transformer.h[i].attn.c_k.weight.float()
                steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                projected_x = xin @ w_key.T
                tables = moment_tables(qp, A, C, window)
                lo, hi = self.share(raw.shape[-1])
                rsteps = s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                raw_credit[r] = rawk_credit(kp, raw, rsteps, cos, sin, qp, A, C, tables, lo, hi).transpose(0, 1).reshape(
                    d.length, self.width)
                scores = inputk_scores(xin, steps, rows, projected_x, projected_rows, raw, cos, sin, qp, A, C, tables)
                estimate[r] = decode_key_pairs(scores.T.unsqueeze(0), steps, rows) * (len(rows) / total_rows)
            credits[i] = self.group_sum(estimate.flatten(0, 1))
            raw_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
        self._input_key_errors = raw_errors
        return credits

    def path_credits(self, chain, targets, raw_values):
        """Sum independent input responses, then transport only along residual identities."""
        L = self.model.config.n_layer
        inputs, keys = self.value_path_credit(raw_values), self.key_input_credits(targets)
        inputs = {i: inputs[i] + keys[i] for i in range(L)}
        for i, credit in self.gate_input_credits(targets, raw_values).items():
            inputs[i] = inputs[i] + credit
        extra = {L-1: torch.zeros_like(chain[0])}
        for i in reversed(range(1, L)):
            extra[i-1] = self.model.resid_lambdas[i].float() * (inputs[i]+extra[i])
        return extra, inputs


    def gate_input_credits(self, targets, raw_values):
        """Independent gate-input experiments from raw hubs; only linear consumers follow."""
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits = {}
        for i, target in targets.items():
            # Advance for every layer to retain the tested private pilot bases.
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            if not has_ve(i, self.model.config.n_layer):
                continue
            rows = frame[indices] * signs
            value = raw_values[i].reshape(-1, heads, self.width // heads)
            ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
            kappa = (value * ve).sum(-1)
            xin = d.blocks[i].float().flatten(0, 1)
            weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
            projected_rows = F.linear(rows[:, :weight.shape[1]], weight)
            row_square = rows.square().mean(-1)
            result = torch.empty_like(xin)
            for t0 in range(0, len(xin), s.chain_tokens):
                sl = slice(t0, t0 + s.chain_tokens)
                result[sl] = prepared_gate_input_response(
                    xin[sl], weight, kappa[sl], rows, projected_rows, row_square, s.gate_path_sigma)
            credits[i] = result
        return credits


    @staticmethod
    def path_scalar_credit(inputs, extra, x0, outs):
        """Linear mixing coefficients see input responses plus later identity-path credit."""
        credits = []
        for i in range(len(outs)):
            credit = inputs[i]+extra[i]
            previous = x0 if i == 0 else outs[i-1]
            credits.append(torch.stack([(credit*previous).sum(), (credit*x0).sum()]))
        return torch.stack(credits, dim=1)


    def own_query_inputs(self, sequence=0):
        """Clean keys/values for one sequence; preserve upstream math and token positions."""
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        inputs, w_q, keys, values, w_proj_t, windows = [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            k = d.outputs[prefix + 'c_k'][sequence].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            c, s = cos[:, None], sin[:, None]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][sequence].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][sequence].float().view(length, heads, width)
            inputs.append(d.blocks[i][sequence].float())
            w_q.append(block.attn.c_q.weight.float())
            keys.append(k.transpose(0, 1).to(torch.bfloat16).contiguous())
            values.append(v.transpose(0, 1).to(torch.bfloat16).contiguous())
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            windows.append(model.window_sizes[i][0])
        return (torch.arange(length, device=self.device), inputs, w_q, keys, values, w_proj_t, cos, sin, windows)


    def select_mlp_routes(self, layer, mid, hpre, activation):
        """Select the fixed replay units; experiments may override only this ranking."""
        weight = self.model.transformer.h[layer].mlp.c_proj.weight.float()
        return (activation * weight.norm(dim=0)).topk(self.s.route_k, dim=-1).indices


    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        ro = Readout(self)
        self.maps.clear()
        self._input_key_errors = None
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()  # coordinate differences at the final readout
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        for i in range(L):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            top = self.select_mlp_routes(i, mid, hpre, r)
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        chain = ro.chain_credits(e, x0, mids, outs, routes, yf)
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        # Reuse only directly measured hub credit, scoped to this capture.
        # Corrected targets still take a separate A.T @ target in exact_values.
        raw_values = {i: self.value_credit(i, target) for i, target in targets.items()}
        extra, inputs = ro.path_credits(chain, targets, raw_values)
        value_targets = {}
        scalars = torch.zeros(2, L, device=self.device)
        for i in range(L):
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
            value_targets[i] = (x_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            # the internals' local step through the attention module, scored as c_proj sees the mid-stream credit
            scalars[:, i] = chain['lam', i]
        scalars += self.path_scalar_credit(inputs, extra, x0, outs)
        self.internals(gradients, targets, b, t, value_targets, self._input_key_errors, raw_values)
        self._input_key_errors = None
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        if self.part:  # split: formed in full by every GPU of the sequence, kept by its first GPU (the
            # internals split or average their own: keys share dims, queries share draws, hidden units share units)
            for name in list(gradients):
                if name == 'lm_head.weight' or name.endswith('c_proj.weight') or name in (
                        'transformer.wte.weight', 'resid_lambdas', 'x0_lambdas'):
                    gradients[name] = torch.zeros_like(gradients[name])
        ro.finish(gradients)
        return gradients


# ---------------------------------------------------------------------------------------------------------- kernels


@triton.jit
def key_kernel(Q, DK, A, C, Out,
               H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               KH: tl.constexpr, KP: tl.constexpr, KM: tl.constexpr, KD: tl.constexpr,
               QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
               AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
               CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
               SourceStart, QueryStart, Window: tl.constexpr,
               BS: tl.constexpr, TP: tl.constexpr, BQ: tl.constexpr, PRECISION: tl.constexpr, EVEN: tl.constexpr):
    """Raw-K experiment scores; a program scores TP trials x BS sources over its own sources' query range (no mask:
    A, C are 0 outside each source's window). EVEN: no load is masked. dk
    [h, P, M, D] may have any strides (prepared_key_moves' permuted view needs no copy)."""
    pblock, sblock, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    j = tl.arange(0, TP * BS)
    trial = pblock * TP + j // BS
    source = sblock * BS + j % BS
    ok = (trial < P) & (source < M)
    col = trial * M + source
    features = tl.arange(0, D)
    k_offsets = head * KH + trial[None, :] * KP + source[None, :] * KM + features[:, None] * KD
    if EVEN:
        changed = tl.load(DK + k_offsets)
    else:
        changed = tl.load(DK + k_offsets, ok[None, :], 0)
    first = tl.maximum(0, SourceStart + sblock * BS - QueryStart)
    last = tl.minimum(T, SourceStart + tl.minimum(sblock * BS + BS, M) - 1 + Window + 1 - QueryStart)
    first = first // BQ * BQ
    result = tl.full((TP * BS,), 0, tl.float32)
    for start in range(first, last, BQ):
        query = start + tl.arange(0, BQ)
        if EVEN:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD)
        else:
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD, query[:, None] < T, 0)
        delta = tl.dot(q, changed, input_precision=PRECISION) * (D ** -.5)
        valid = (query[:, None] < T) & ok[None, :]
        if EVEN:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM)
        else:
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
        change = a * libdevice.expm1(delta)
        result += tl.sum(change / (1 + change) * c, 0)
    tl.store(Out + head * P * M + col, result, ok)


@triton.jit
def rawk_kernel(DK, STEPS, U, M1, Q, A, C, Out, T0, Lo, J, KH, KJ, KM, KD, SH, ST, QH, QT, AH, AD, CH, CD, OH, OT,
                H: tl.constexpr, D: tl.constexpr, BJ: tl.constexpr):
    """One (source t = T0 + program_id(0), head h): its raw-K experiments j < J (this GPU's dims Lo + j), the moves
    DK [H, J, M, D] (any strides) being raw_key_moves of the KEY_BUCKET from T0, scored by the
    tables + the exact s = t term and divided by the step: Out[h, t, Lo + j] (strides OH, OT)."""
    m, h = tl.program_id(0), tl.program_id(1)
    t = T0 + m
    j = tl.arange(0, BJ)
    jm = j < J
    f = tl.arange(0, D)
    dk = tl.load(DK + h * KH + j[:, None] * KJ + m * KM + f[None, :] * KD, jm[:, None], 0.)
    u = tl.load(U + (t * H + h) * D + f)
    m1 = tl.load(M1 + (t * H + h) * D * D + f[:, None] * D + f[None, :])
    qs = tl.load(Q + h * QH + t * QT + f)
    a_self = tl.load(A + h * AH + t * AD)
    c_self = tl.load(C + h * CH + t * CD)
    step = tl.load(STEPS + h * SH + t * ST)
    y = tl.dot(dk.to(tl.bfloat16), m1)
    change = a_self * libdevice.expm1(tl.sum(dk * qs[None, :], 1) * (D ** -.5))
    score = tl.sum(dk * u[None, :], 1) * (D ** -.5) + tl.sum(y * dk, 1) * (1. / D) + change / (1 + change) * c_self
    tl.store(Out + h * OH + t * OT + Lo + j, score / step, jm)


@triton.jit
def outer_kernel(X, Y, Out, N, XB, XN, YB, YN, D: tl.constexpr, BN: tl.constexpr):
    b, n = tl.program_id(0), tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, D * D)
    ok = (n < N)[:, None]
    x = tl.load(X + b * XB + n[:, None] * XN + (k // D)[None, :], ok)
    y = tl.load(Y + b * YB + n[:, None] * YN + (k % D)[None, :], ok)
    tl.store(Out + (b * N + n[:, None]) * D * D + k[None, :], (x * y).to(tl.bfloat16), ok)


@triton.jit
def inputk_prep_kernel(M1, U, Q, PX, KR, CS, SN, AD, CD, Mt, V, S, T, QH, QT, KH, KT, AH, AT, CH, CT,
                H: tl.constexpr, D: tl.constexpr):
    """Per (t, h): Mt = R_t^T M1 R_t (bf16), V[h, :, t] = (lam kr, lam px, R^T u, R^T q, Ms px, Ms kr) (Ms = Mt + Mt^T,
    lam = cos^2 + sin^2) and the scalars S[:, h, t] (_epilogue's names). R^T v = c v + s' v[other] (s' = -sin on the
    first half, +sin on the second: the transpose of the kernels' rotation)."""
    t, h = tl.program_id(0), tl.program_id(1)
    f = tl.arange(0, D)
    o = (f + D // 2) % D
    c = tl.load(CS + t * (D // 2) + f % (D // 2))
    s = tl.load(SN + t * (D // 2) + f % (D // 2)) * tl.where(f < D // 2, -1., 1.)
    row = t * H + h
    mb = M1 + row * D * D
    mt = (c[:, None] * (c[None, :] * tl.load(mb + f[:, None] * D + f[None, :]).to(tl.float32)
                        + s[None, :] * tl.load(mb + f[:, None] * D + o[None, :]).to(tl.float32))
          + s[:, None] * (c[None, :] * tl.load(mb + o[:, None] * D + f[None, :]).to(tl.float32)
                          + s[None, :] * tl.load(mb + o[:, None] * D + o[None, :]).to(tl.float32)))
    tl.store(Mt + row * D * D + f[:, None] * D + f[None, :], mt.to(tl.bfloat16))
    px, kr = tl.load(PX + row * D + f), tl.load(KR + h * KH + t * KT + f)
    ru = c * tl.load(U + row * D + f) + s * tl.load(U + row * D + o)
    rq = c * tl.load(Q + h * QH + t * QT + f) + s * tl.load(Q + h * QH + t * QT + o)
    lam = c * c + s * s
    vx = tl.sum(mt * px[None, :], 1) + tl.sum(mt * px[:, None], 0)
    v0 = tl.sum(mt * kr[None, :], 1) + tl.sum(mt * kr[:, None], 0)
    vb = V + (h * 6 * T + t) * D + f
    tl.store(vb, lam * kr)
    tl.store(vb + T * D, lam * px)
    tl.store(vb + 2 * T * D, ru)
    tl.store(vb + 3 * T * D, rq)
    tl.store(vb + 4 * T * D, vx)
    tl.store(vb + 5 * T * D, v0)
    sb, HT = S + h * T + t, H * T
    tl.store(sb, tl.sum(kr * lam * px))
    tl.store(sb + HT, tl.sum(px * lam * px))
    tl.store(sb + 2 * HT, tl.sum(px * ru))
    tl.store(sb + 3 * HT, tl.sum(kr * ru))
    tl.store(sb + 4 * HT, tl.sum(px * rq))
    tl.store(sb + 5 * HT, tl.sum(kr * rq))
    tl.store(sb + 6 * HT, tl.sum(px * vx) / 2)
    tl.store(sb + 7 * HT, tl.sum(kr * v0) / 2)
    tl.store(sb + 8 * HT, tl.sum(px * v0))
    tl.store(sb + 9 * HT, tl.sum(kr * lam * kr))
    tl.store(sb + 10 * HT, tl.load(AD + h * AH + t * AT))
    tl.store(sb + 11 * HT, tl.load(CD + h * CH + t * CT))


@triton.jit
def stats_kernel(Dm, X, SQX, DB, ST, Start, n, D_ROWS, W: tl.constexpr, BP: tl.constexpr, EPS: tl.constexpr):
    """Stage 1 of the two-stage scorer: per row of the first n rows of every token of the delta Dm [m, D_ROWS, W], its
    bf16 copy DB [m, n, W] (the projection GEMM's input) and ST [m, n, 2] = (inv', inv' / inv0 - 1) of block j's input
    RMSNorm at x + d (X [T, W], SQX = |x|^2, token Start + t)."""
    pb, t = tl.program_id(0), tl.program_id(1)
    tok = Start + t
    p = pb * BP + tl.arange(0, BP)
    pm = p < n
    k = tl.arange(0, W)
    d = tl.load(Dm + (t * D_ROWS + p).to(tl.int64)[:, None] * W + k[None, :], pm[:, None], 0.)
    dotx = tl.sum(d * tl.load(X + tok * W + k)[None, :], 1)
    sqd = tl.sum(d * d, 1)
    tl.store(DB + (t * n + p).to(tl.int64)[:, None] * W + k[None, :], d.to(tl.bfloat16), pm[:, None])
    sq = tl.load(SQX + tok)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    tl.store(ST + (t * n + p) * 2, inv, pm)
    tl.store(ST + (t * n + p) * 2 + 1, (inv - inv0) / inv0, pm)


@triton.jit
def staged_kernel(UP, ST, R0, VC, GZ, VE, CS, SN, U, WV, M1, M2, QD, GD, AD, CD, VT, OT, Out, SE, Start, n, SE_ROWS,
                  W: tl.constexpr, H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, DIAG: tl.constexpr,
                  SELF: tl.constexpr, BP: tl.constexpr, NC: tl.constexpr):
    """Stage 3: the readout score of each row from the bf16 projection of the raw moves UP [m, n, NC] (columns: d W_k^T,
    d W_v^T, then d[:32] W_g^T padded to 16) and the row statistics ST: the projected change of the normed input is
    inv' (d W^T) + (inv' / inv0 - 1) (clean projection: R0 rotated raw key [T, H, D], VC raw value [T, W], GZ gate
    logits [T, H])."""
    pb, t = tl.program_id(0), tl.program_id(1)  # row blocks fastest: a token's tables stay in L2
    tok = Start + t
    p = pb * BP + tl.arange(0, BP)
    pm = p < n
    f = tl.arange(0, D)
    other = (f + D // 2) % D
    half = f % (D // 2)
    cfull = tl.load(CS + tok * (D // 2) + half)
    ssig = tl.load(SN + tok * (D // 2) + half) * tl.where(f < D // 2, 1., -1.)
    inv = tl.load(ST + (t * n + p) * 2, pm, 1.)
    coef = tl.load(ST + (t * n + p) * 2 + 1, pm, 0.)
    urow = (t * n + p).to(tl.int64) * NC
    g16 = tl.arange(0, 16)
    if GATED:
        ug = tl.load(UP + urow[:, None] + 2 * W + g16[None, :], pm[:, None], 0.).to(tl.float32)
        gz = tl.load(GZ + tok * H + g16, g16 < H, 0.)
        gch = 2 * (tl.sigmoid(gz[None, :] + inv[:, None] * ug + coef[:, None] * gz[None, :]) - tl.sigmoid(gz)[None, :])
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        uk = tl.load(UP + urow[:, None] + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        uks = tl.load(UP + urow[:, None] + h * D + other[None, :], pm[:, None], 0.).to(tl.float32)
        uv = tl.load(UP + urow[:, None] + W + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        row = tok * H + h
        r0 = tl.load(R0 + row * D + f)
        dr = inv[:, None] * (uk * cfull[None, :] + uks * ssig[None, :]) + coef[:, None] * r0[None, :]
        dv = inv[:, None] * uv + coef[:, None] * tl.load(VC + tok * W + h * D + f)[None, :]
        if GATED:
            gc = tl.sum(gch * (g16 == h).to(tl.float32)[None, :], 1)
            dv = dv + gc[:, None] * tl.load(VE + row * D + f)[None, :]
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        if DIAG or SELF:  # the receiver s = t, exactly
            x = tl.sum(dk * tl.load(QD + row * D + f)[None, :], 1) * (D ** -.5)
            a = tl.load(AD + row)
            e = libdevice.expm1(x)
            gain = a * e / (1. + a * e)
            vgain = (a + a * e) / (1. + a * e)
            if DIAG:
                total += gain * tl.load(CD + row) + vgain * tl.sum(dv * tl.load(GD + row * D + f)[None, :], 1)
            if SELF:
                vo = tl.load(VT + row * D + f) - tl.load(OT + row * D + f)
                tl.store(SE + (t * SE_ROWS + p[:, None]).to(tl.int64) * W + h * D + f[None, :],
                         (gain[:, None] * vo[None, :] + vgain[:, None] * dv).to(tl.bfloat16), pm[:, None])
        dkb = dk.to(tl.bfloat16)
        total += (tl.sum(dk * tl.load(U + row * D + f)[None, :], 1) * (D ** -.5)
                  + tl.sum(dv * tl.load(WV + row * D + f)[None, :], 1)
                  + tl.sum(tl.dot(dkb, tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])) * dk, 1) * (1. / D)
                  + tl.sum(tl.dot(dkb, tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])) * dv, 1) * (D ** -.5))
    tl.store(Out + t * n + p, total, pm)


@triton.jit
def staged_kernel_live(LROW, LCNT, LSTRIDE, UP, ST, R0, VC, GZ, VE, CS, SN, U, WV, M1, M2, QD, GD, AD, CD, VT, OT, Out, SE, Start, n, SE_ROWS,
                  W: tl.constexpr, H: tl.constexpr, D: tl.constexpr, GATED: tl.constexpr, DIAG: tl.constexpr,
                  SELF: tl.constexpr, BP: tl.constexpr, NC: tl.constexpr):
    """Stage 3: the readout score of each row from the bf16 projection of the raw moves UP [m, n, NC] (columns: d W_k^T,
    d W_v^T, then d[:32] W_g^T padded to 16) and the row statistics ST: the projected change of the normed input is
    inv' (d W^T) + (inv' / inv0 - 1) (clean projection: R0 rotated raw key [T, H, D], VC raw value [T, W], GZ gate
    logits [T, H])."""
    pb, t = tl.program_id(0), tl.program_id(1)  # row blocks fastest: a token's tables stay in L2
    tok = Start + t
    p = pb * BP + tl.arange(0, BP)
    live_row = tl.load(LROW + t * LSTRIDE + p, p < LSTRIDE, 0)
    live_n = tl.load(LCNT + t * LSTRIDE + n)
    if pb * BP >= live_n:
        return
    pm = p < live_n
    p = live_row
    f = tl.arange(0, D)
    other = (f + D // 2) % D
    half = f % (D // 2)
    cfull = tl.load(CS + tok * (D // 2) + half)
    ssig = tl.load(SN + tok * (D // 2) + half) * tl.where(f < D // 2, 1., -1.)
    inv = tl.load(ST + (t * n + p) * 2, pm, 1.)
    coef = tl.load(ST + (t * n + p) * 2 + 1, pm, 0.)
    urow = (t * n + p).to(tl.int64) * NC
    g16 = tl.arange(0, 16)
    if GATED:
        ug = tl.load(UP + urow[:, None] + 2 * W + g16[None, :], pm[:, None], 0.).to(tl.float32)
        gz = tl.load(GZ + tok * H + g16, g16 < H, 0.)
        gch = 2 * (tl.sigmoid(gz[None, :] + inv[:, None] * ug + coef[:, None] * gz[None, :]) - tl.sigmoid(gz)[None, :])
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        uk = tl.load(UP + urow[:, None] + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        uks = tl.load(UP + urow[:, None] + h * D + other[None, :], pm[:, None], 0.).to(tl.float32)
        uv = tl.load(UP + urow[:, None] + W + h * D + f[None, :], pm[:, None], 0.).to(tl.float32)
        row = tok * H + h
        r0 = tl.load(R0 + row * D + f)
        dr = inv[:, None] * (uk * cfull[None, :] + uks * ssig[None, :]) + coef[:, None] * r0[None, :]
        dv = inv[:, None] * uv + coef[:, None] * tl.load(VC + tok * W + h * D + f)[None, :]
        if GATED:
            gc = tl.sum(gch * (g16 == h).to(tl.float32)[None, :], 1)
            dv = dv + gc[:, None] * tl.load(VE + row * D + f)[None, :]
        square = tl.sum(r0 * r0, 0)
        grow = (2 * tl.sum(r0[None, :] * dr, 1) + tl.sum(dr * dr, 1)) / D
        inverse = libdevice.rsqrt(square / D + 1.1920928955078125e-07)
        moved = libdevice.rsqrt(square / D + grow + 1.1920928955078125e-07)
        dk = dr * moved[:, None] - r0[None, :] * (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, None]
        if DIAG or SELF:  # the receiver s = t, exactly
            x = tl.sum(dk * tl.load(QD + row * D + f)[None, :], 1) * (D ** -.5)
            a = tl.load(AD + row)
            e = libdevice.expm1(x)
            gain = a * e / (1. + a * e)
            vgain = (a + a * e) / (1. + a * e)
            if DIAG:
                total += gain * tl.load(CD + row) + vgain * tl.sum(dv * tl.load(GD + row * D + f)[None, :], 1)
            if SELF:
                vo = tl.load(VT + row * D + f) - tl.load(OT + row * D + f)
                tl.store(SE + (t * SE_ROWS + p[:, None]).to(tl.int64) * W + h * D + f[None, :],
                         (gain[:, None] * vo[None, :] + vgain[:, None] * dv).to(tl.bfloat16), pm[:, None])
        dkb = dk.to(tl.bfloat16)
        total += (tl.sum(dk * tl.load(U + row * D + f)[None, :], 1) * (D ** -.5)
                  + tl.sum(dv * tl.load(WV + row * D + f)[None, :], 1)
                  + tl.sum(tl.dot(dkb, tl.load(M1 + row * D * D + f[:, None] * D + f[None, :])) * dk, 1) * (1. / D)
                  + tl.sum(tl.dot(dkb, tl.load(M2 + row * D * D + f[:, None] * D + f[None, :])) * dv, 1) * (D ** -.5))
    tl.store(Out + t * n + p, total, pm)


@triton.jit
def dual_residual_kernel_live(LROW, LCNT, LSTRIDE, A, A2, B, D, n, n_rows, A_ROWS, A2_ROWS, D_ROWS,
                         K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         TWO: tl.constexpr):
    """d[i] = (d[i] + a[i] @ B) + a2[i] @ B for the n_rows first rows of every token (flattened: row i = token i // n_rows,
    row i % n_rows; A rows of A_ROWS per token, A2 of A2_ROWS, D of D_ROWS), in place: each product in its own fp32
    accumulator, then the two adds in order."""
    pn, pm, tok = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = pm * BM + tl.arange(0, BM)
    live_row = tl.load(LROW + tok * LSTRIDE + r, r < LSTRIDE, 0)
    live_n = tl.load(LCNT + tok * LSTRIDE + n_rows)
    if pm * BM >= live_n:
        return
    ok = r < live_n
    r = live_row
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    arow = (tok * A_ROWS + r).to(tl.int64) * K
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + arow[:, None] + kk[None, :], ok[:, None], 0.)
        b = tl.load(B + kk[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    dptr = D + (tok * D_ROWS + r).to(tl.int64)[:, None] * N + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if TWO:
        acc2 = tl.zeros((BM, BN), tl.float32)
        arow2 = (tok * A2_ROWS + r).to(tl.int64) * K
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A2 + arow2[:, None] + kk[None, :], ok[:, None], 0.)
            b = tl.load(B + kk[:, None] * N + cols[None, :])
            acc2 = tl.dot(a, b, acc2)
        new = new + acc2
    tl.store(dptr, new, ok[:, None])


@triton.jit
def dual_residual_kernel(A, A2, B, D, n, n_rows, A_ROWS, A2_ROWS, D_ROWS,
                         K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         TWO: tl.constexpr):
    """d[i] = (d[i] + a[i] @ B) + a2[i] @ B for the n_rows first rows of every token (flattened: row i = token i // n_rows,
    row i % n_rows; A rows of A_ROWS per token, A2 of A2_ROWS, D of D_ROWS), in place: each product in its own fp32
    accumulator, then the two adds in order."""
    pn, pm = tl.program_id(0), tl.program_id(1)
    i = pm * BM + tl.arange(0, BM)
    ok = i < n
    tok, r = i // n_rows, i % n_rows
    cols = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    arow = (tok * A_ROWS + r).to(tl.int64) * K
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(A + arow[:, None] + kk[None, :], ok[:, None], 0.)
        b = tl.load(B + kk[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    dptr = D + (tok * D_ROWS + r).to(tl.int64)[:, None] * N + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if TWO:
        acc2 = tl.zeros((BM, BN), tl.float32)
        arow2 = (tok * A2_ROWS + r).to(tl.int64) * K
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A2 + arow2[:, None] + kk[None, :], ok[:, None], 0.)
            b = tl.load(B + kk[:, None] * N + cols[None, :])
            acc2 = tl.dot(a, b, acc2)
        new = new + acc2
    tl.store(dptr, new, ok[:, None])


@triton.jit
def mlp_up_prune_kernel(D, X, IDX, WFC, HS, SC, CUT, Out, n_rows, D_ROWS, O_ROWS, n_rb,
                        W: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, EPS: tl.constexpr, MARGIN: tl.constexpr):
    """The routed MLP's first half for BP rows of one token over its routed units in sort_routes' order (SC: their s,
    descending), stopping at the row block's prune point (the bound: every later unit's change is exactly 0 for every
    row of the block; stored in CUT for the down kernel): dh = N(mid + d) W_fc^T - N(mid) W_fc^T on the units =
    inv' (d W_fc^T) + (inv' / inv0 - 1) h, then relu(h + dh)^2 - relu(h)^2, bf16."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    dbase = D + (t * D_ROWS + rows).to(tl.int64) * W
    dotx = tl.zeros((BP,), tl.float32)
    sqd = tl.zeros((BP,), tl.float32)
    sqx = tl.zeros((BK,), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
        x = tl.load(X + t * W + kk)
        dotx += tl.sum(d * x[None, :], 1)
        sqd += tl.sum(d * d, 1)
        sqx += x * x
    sq = tl.sum(sqx, 0)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dotx + sqd) / W + EPS)
    coef = (inv - inv0) / inv0
    a_max = tl.max(tl.where(ok, inv * libdevice.sqrt(sqd), 0.), 0) * MARGIN
    c_max = tl.max(tl.where(ok, tl.abs(coef), 0.), 0) * MARGIN
    thr = -a_max / (1 - c_max)
    upos = tl.arange(0, KP)
    s = tl.load(SC + t * K + upos, upos < K, float('-inf'))
    n_units = tl.max(tl.where(s >= thr, upos + 1, 0), 0)
    n_units = tl.where((c_max < 0.5) & (a_max == a_max), n_units, K)  # (a non-finite bound: every unit)
    tl.store(CUT + t * n_rb + pb, n_units)
    obase = Out + (t * O_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, n_units, BN):
        units = u0 + tl.arange(0, BN)
        uidx = tl.load(IDX + t * K + units).to(tl.int64)
        acc = tl.zeros((BP, BN), tl.float32)
        for k0 in range(0, W, BK):
            kk = k0 + tl.arange(0, BK)
            d = tl.load(dbase[:, None] + kk[None, :], ok[:, None], 0.)
            w = tl.load(WFC + uidx[None, :] * W + kk[:, None])
            acc = tl.dot(d.to(tl.bfloat16), w, acc)
        h = tl.load(HS + t * K + units)
        dh = inv[:, None] * acc + coef[:, None] * h[None, :]
        up = tl.maximum(h[None, :] + dh, 0.)
        down = tl.maximum(h, 0.)[None, :]
        tl.store(obase[:, None] + units[None, :], ((up - down) * (up + down)).to(tl.bfloat16), ok[:, None])


@triton.jit
def fc_relu_kernel(DIRS, FSTEP, HV, VALID, Out, P, NS, CYC, BP: tl.constexpr, BK: tl.constexpr):
    """The c_fc source moves before their GEMM, one pass: r[t, p, k] = (up - down)(up + down) with up = relu(h + move), down =
    relu(h), move = (step_t code[t, p, k + 1]) valid[t, k] (DIRS [m, P, CYC] the code rows; HV / VALID [m, NS]). Launch
    with enable_fp_fusion=False: each op rounds as torch's separate kernels do (bitwise)."""
    pb, kb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = pb * BP + tl.arange(0, BP)
    k = kb * BK + tl.arange(0, BK)
    mask = (p < P)[:, None] & (k < NS)[None, :]
    row = (t * P + p).to(tl.int64)
    code = tl.load(DIRS + row[:, None] * CYC + 1 + k[None, :], mask, 0.)
    valid = tl.load(VALID + t * NS + k, k < NS, 0).to(tl.float32)
    h = tl.load(HV + t * NS + k, k < NS, 0.)
    move = tl.load(FSTEP + t) * code * valid[None, :]
    up = tl.maximum(h[None, :] + move, 0.)
    down = tl.maximum(h, 0.)[None, :]
    tl.store(Out + row[:, None] * NS + k[None, :], (up - down) * (up + down), mask)


@triton.jit
def oq_attn_kernel(U, STATS, R0, SQ0, CS, SN, KEYS, VALS, IDX, TP, O, Out, n_rows, U_ROWS, O_ROWS, t_start,
                   T: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, W: tl.constexpr, KK: tl.constexpr,
                   BP: tl.constexpr, EPS: tl.constexpr):
    """The own-query response of every row of one token in one head: from the projected move u = d W_q^T,
    dr = inv' rotate(u) + coef r0, the per-head RMSNorm change dq = N(r0 + dr) - N(r0), then the record's routed
    response over the token's KK strongest clean keys (gathered once per program), the other keys' output held:
    with e = expm1(dq k / sqrt(DH)) and z = sum p e (= expm1(dlse)), weights p expm1(ds - dlse) = p (e - z) / (1 + z)
    and expm1(-dlse) = -z / (1 + z). bf16 out."""
    hh, t = tl.program_id(0), tl.program_id(1)
    pos = t_start + t
    base = pos * H + hh
    QH: tl.constexpr = DH // 2
    hd = tl.arange(0, QH)
    feat = tl.arange(0, DH)
    keys_n = tl.arange(0, KK)
    c = tl.load(CS + pos * QH + hd)[None, :]
    s = tl.load(SN + pos * QH + hd)[None, :]
    r0l = tl.load(R0 + base * DH + hd)[None, :]
    r0h = tl.load(R0 + base * DH + QH + hd)[None, :]
    sq0 = tl.load(SQ0 + base)
    qi0 = libdevice.rsqrt(sq0 / DH + EPS)
    kidx = tl.load(IDX + base * KK + keys_n).to(tl.int64)
    krow = (hh * T + kidx) * DH
    kl = tl.trans(tl.load(KEYS + krow[:, None] + hd[None, :]))
    kh = tl.trans(tl.load(KEYS + krow[:, None] + QH + hd[None, :]))
    v = tl.load(VALS + krow[:, None] + feat[None, :])
    p = tl.load(TP + base * KK + keys_n)
    rest = (tl.load(O + base * DH + feat) - tl.sum(p[:, None] * v.to(tl.float32), 0))[None, :]  # the held keys' output
    p = p[None, :]
    for r0 in range(0, n_rows, BP):
        rows = r0 + tl.arange(0, BP)
        ok = rows < n_rows
        row = (t * U_ROWS + rows).to(tl.int64)
        ub = U + row * W + hh * DH
        ul = tl.load(ub[:, None] + hd[None, :], ok[:, None], 0.)
        uh = tl.load(ub[:, None] + QH + hd[None, :], ok[:, None], 0.)
        inv = tl.load(STATS + row * 2, ok, 1.)[:, None]
        coef = tl.load(STATS + row * 2 + 1, ok, 0.)[:, None]
        drl = inv * (ul * c + uh * s) + coef * r0l
        drh = inv * (ul * -s + uh * c) + coef * r0h
        cross = tl.sum(r0l * drl, 1) + tl.sum(r0h * drh, 1)
        sqdr = tl.sum(drl * drl, 1) + tl.sum(drh * drh, 1)
        qi = libdevice.rsqrt((sq0 + 2 * cross + sqdr) / DH + EPS)[:, None]
        dql = (drl * qi + r0l * (qi - qi0)).to(tl.bfloat16)
        dqh = (drh * qi + r0h * (qi - qi0)).to(tl.bfloat16)
        ds = tl.dot(dqh, kh, tl.dot(dql, kl)) * (DH ** -.5)
        e = tl.exp(ds) - 1  # absolute error ~2^-22 (the record rounds ds itself to bf16 first: ~2^-9 relative)
        z = tl.sum(p * e, 1)[:, None]
        rz = 1 / (1 + z)
        weights = (p * (e - z) * rz).to(tl.bfloat16)
        moved = tl.dot(weights, v) - z * rz * rest
        tl.store(Out + (t * O_ROWS + rows).to(tl.int64)[:, None] * W + hh * DH + feat[None, :],
                 moved.to(tl.bfloat16), ok[:, None])


@triton.jit
def head_prep_kernel(D, XL, TW, SQXL, BY, T0Y, MB, MW, A, STATS, n_rows, D_ROWS,
                     W: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr, EPS: tl.constexpr):
    """Per row of one token: the bf16 copy of d (the logits GEMM's input), and from
    the fp32 d: inv' = rsqrt(|xL + d|^2 / W + eps), inv' - inv0, the target logit's capped change
    c(zy) - c(zy0) with zy = (base_y + d . w_y) inv' (fp32, as the record), c(z) = 15 tanh(z / 15), in difference
    form: 15 tanh(dz / 15) (1 - t0^2) / (1 + t0 tanh(dz / 15)), t0 = tanh(zy0 / 15); and a bound on every entry's
    |u| = |base (inv' - inv0) + (d_bf16 . w) inv'| / 15 <= (max|base| |inv' - inv0| + 1.01 |d| max|w| inv') / 15."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BR + tl.arange(0, BR)
    ok = rows < n_rows
    row = (t * D_ROWS + rows).to(tl.int64)
    dots = tl.zeros((BR,), tl.float32)
    sqd = tl.zeros((BR,), tl.float32)
    tdot = tl.zeros((BR,), tl.float32)
    for k0 in range(0, W, BK):
        kk = k0 + tl.arange(0, BK)
        d = tl.load(D + row[:, None] * W + kk[None, :], ok[:, None], 0.)
        dots += tl.sum(d * tl.load(XL + t * W + kk)[None, :], 1)
        tdot += tl.sum(d * tl.load(TW + t * W + kk)[None, :], 1)
        sqd += tl.sum(d * d, 1)
        tl.store(A + row[:, None] * W + kk[None, :], d.to(tl.bfloat16), ok[:, None])
    sq = tl.load(SQXL + t)
    inv0 = libdevice.rsqrt(sq / W + EPS)
    inv = libdevice.rsqrt((sq + 2 * dots + sqd) / W + EPS)
    dinv = inv - inv0
    by, t0 = tl.load(BY + t), tl.load(T0Y + t)
    dz = by * dinv + tdot * inv
    tb = libdevice.tanh(dz / 15)
    dc = 15 * tb * (1 - t0 * t0) / (1 + t0 * tb)
    bound = (tl.load(MB + t) * tl.abs(dinv) + 1.01 * libdevice.sqrt(sqd) * tl.load(MW) * inv) / 15
    tl.store(STATS + row * 4, inv, ok)
    tl.store(STATS + row * 4 + 1, dinv, ok)
    tl.store(STATS + row * 4 + 2, dc, ok)
    tl.store(STATS + row * 4 + 3, bound, ok)


@triton.jit
def mlp_down_live_kernel(Rb, IDX, WOT, D, DX0, LAM, CUT, XN, PSI, A16, XL, TW, n_rows, D_ROWS, R_ROWS, n_x0, n_rb,
                         W: tl.constexpr, K: tl.constexpr, BP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         RB: tl.constexpr, SCALE: tl.constexpr, X0: tl.constexpr, FULL: tl.constexpr,
                         NEXT: tl.constexpr, HEADP: tl.constexpr):
    """The routed MLP's second half over the row block's live unit prefix (CUT of its RB-row prune block, rounded up to
    BK; every unit there was formed by mlp_up_live_kernel, the rest are exact zeros): d += r W_out^T, then block b + 1's
    mixing scalars d = lam (d + change) (+ lam0 dx0 on the embedding root's rows), in place; NEXT: the new rows'
    partial sums d . x and d . d on these columns (PSI) for block b + 1's input RMSNorm (x = its clean input, XN).
    HEADP (last block): the head's inputs instead of the fp32 moves (read by nothing else): their bf16 copy (A16, the
    logits GEMM's input, as head_prep_kernel) and the partial sums d . xL, d . w_y, d . d (PSI, 3 per
    column block)."""
    nb, pb, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pb * BP + tl.arange(0, BP)
    ok = rows < n_rows
    cols = nb * BN + tl.arange(0, BN)
    if FULL:
        n_units = K
    else:
        n_units = tl.load(CUT + t * n_rb + (pb * BP) // RB)
    acc = tl.zeros((BP, BN), tl.float32)
    rbase = Rb + (t * R_ROWS + rows).to(tl.int64) * K
    for u0 in range(0, n_units, BK):
        uu = u0 + tl.arange(0, BK)
        a = tl.load(rbase[:, None] + uu[None, :], ok[:, None], 0.)
        uidx = tl.load(IDX + t * K + uu).to(tl.int64)
        w = tl.load(WOT + uidx[:, None] * W + cols[None, :])
        acc = tl.dot(a, w, acc)
    dptr = D + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :]
    new = tl.load(dptr, ok[:, None], 0.) + acc
    if SCALE:
        new = tl.load(LAM) * new
        if X0:
            lam0 = tl.load(LAM + 1)
            mine = rows < n_x0
            x0d = tl.load(DX0 + (t * n_x0 + rows).to(tl.int64)[:, None] * W + cols[None, :], mine[:, None], 0.)
            new = tl.where(mine[:, None], new + lam0 * x0d, new)
    if HEADP:
        tl.store(A16 + (t * D_ROWS + rows).to(tl.int64)[:, None] * W + cols[None, :], new.to(tl.bfloat16), ok[:, None])
        prow = ((t * D_ROWS + rows).to(tl.int64) * (W // BN) + nb) * 3
        tl.store(PSI + prow, tl.sum(new * tl.load(XL + t * W + cols)[None, :], 1), ok)
        tl.store(PSI + prow + 1, tl.sum(new * tl.load(TW + t * W + cols)[None, :], 1), ok)
        tl.store(PSI + prow + 2, tl.sum(new * new, 1), ok)
    else:
        tl.store(dptr, new, ok[:, None])
        if NEXT:
            x = tl.load(XN + t * W + cols)
            prow = ((t * D_ROWS + rows).to(tl.int64) * (W // BN) + nb) * 2
            tl.store(PSI + prow, tl.sum(new * x[None, :], 1), ok)
            tl.store(PSI + prow + 1, tl.sum(new * new, 1), ok)


@triton.jit
def round_tf32(x):
    """x rounded to the nearest TF32 value (ties away from zero: the sign-magnitude bits + half an ulp, low 13 bits
    cleared). The tensor cores truncate fp32 operands to TF32 (a bias of ~2^-11 per operand); a dot of rounded operands
    is unbiased."""
    b = x.to(tl.uint32, bitcast=True)
    return ((b + 0x1000) & 0xFFFFE000).to(tl.float32, bitcast=True)


@triton.jit
def inroot_kernel(DQ, QRAW, CS, SN, K, V, A, Base, TG, Out, t0, window, P,
                  H: tl.constexpr, D: tl.constexpr,
                  KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                  VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                  AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                  OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                  BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, EPS: tl.constexpr,
                  ROUND: tl.constexpr):
    """The in-root score of BP raw-query moves of one token, summed over the heads, in contrast form. Per head:
    the query move (r0 = rotate(q_raw), dr = rotate(dq), the per-head RMSNorm change delta = dr inv' + r0
    (inv' - inv)), then over the key tiles of the window (from max(0, t - window) to t): dl = delta . k / sqrt(D)
    (PRECISION dot), e = exp(dl) - 1, Z += sum a e, S += sum a e c with the contrast c_s = target . (v_s - o_t) formed
    from the V tile in fp32 (TG [m, H * D]: target = chain_j W_proj of the token); score += S / (1 + Z). Out [m, P]."""
    pb, token = tl.program_id(0), tl.program_id(1)
    probe = pb * BP + tl.arange(0, BP)
    pm = probe < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    pos = t0 + token
    cs = tl.load(CS + pos * (D // 2) + half)
    sn = tl.load(SN + pos * (D // 2) + half)
    lo = (tl.maximum(pos - window, 0) // BK) * BK
    total = tl.zeros((BP,), tl.float32)
    for h in range(H):
        qrow = QRAW + (pos * H + h) * D
        x = tl.load(qrow + f)
        xo = tl.load(qrow + other)
        r0 = tl.where(low, x * cs + xo * sn, -xo * sn + x * cs)
        drow = DQ + ((token * P + probe) * H + h).to(tl.int64) * D
        dd = tl.load(drow[:, None] + f[None, :], pm[:, None], 0.)
        ddo = tl.load(drow[:, None] + other[None, :], pm[:, None], 0.)
        dr = tl.where(low[None, :], dd * cs[None, :] + ddo * sn[None, :], -ddo * sn[None, :] + dd * cs[None, :])
        square = tl.sum(r0 * r0, 0)
        cross = tl.sum(r0[None, :] * dr, 1)
        sqd = tl.sum(dr * dr, 1)
        inverse = libdevice.rsqrt(square / D + EPS)
        moved = libdevice.rsqrt((square + 2 * cross + sqd) / D + EPS)
        delta = (dr * moved[:, None] + r0[None, :] * (moved - inverse)[:, None]) * (D ** -.5)
        if ROUND:
            delta = round_tf32(delta)
        tg = tl.load(TG + (token * H + h) * D + f)
        base = tl.load(Base + h * OH + token * OM + f * OD)
        z = tl.zeros((BP,), tl.float32)
        s = tl.zeros((BP,), tl.float32)
        for start in range(lo, pos + 1, BK):
            key = start + tl.arange(0, BK)
            km = key <= pos
            a = tl.load(A + h * AH + token * AM + key * AT, km, 0.)
            k = tl.load(K + h * KH + key[None, :] * KT + f[:, None] * KD, km[None, :], 0.)
            v = tl.load(V + h * VH + key[:, None] * VT + f[None, :] * VD, km[:, None], 0.)
            c = tl.sum((v - base[None, :]) * tg[None, :], 1)
            if ROUND:
                k = round_tf32(k)
            ae = a[None, :] * (tl.exp(tl.dot(delta, k, input_precision=PRECISION)) - 1)
            z += tl.sum(ae, 1)
            s += tl.sum(ae * c[None, :], 1)
        total += s / (1 + z)
    tl.store(Out + token * P + probe, total, pm)


@triton.jit
def qsrc2_kernel(DIRS, QRAW, QSTEP, CS, SN, K, V, A, Base, Out, t0, window, P,
                 H: tl.constexpr, D: tl.constexpr,
                 KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
                 VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
                 AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
                 OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
                 BP: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, EPS: tl.constexpr,
                 ROUND: tl.constexpr, QKP: tl.constexpr):
    """The q source's move of BP probes of one token in one head, then the exact softmax response over every key, in
    centered form: numerator += (a e) (v - o) over the key tiles of the window (from
    max(0, t - window) to t), e = exp(dl) - 1, response = numerator / (1 + partition): the same algebra as (numerator -
    partition o) / (1 + partition) without its cancellation, so the dots can run one-pass TF32 (PRECISION)."""
    token, head = tl.program_id(1) // H, tl.program_id(1) % H
    probe = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = probe < P
    f = tl.arange(0, D)
    half = f % (D // 2)
    low = f < D // 2
    other = (f + D // 2) % D
    pos = t0 + token
    cs = tl.load(CS + pos * (D // 2) + half)
    sn = tl.load(SN + pos * (D // 2) + half)
    qrow = QRAW + (pos * H + head) * D
    x = tl.load(qrow + f)
    xo = tl.load(qrow + other)
    r0 = tl.where(low, x * cs + xo * sn, -xo * sn + x * cs)
    step = tl.load(QSTEP + token)
    drow = DIRS + ((token * P + probe) * H + head).to(tl.int64) * D
    dd = tl.load(drow[:, None] + f[None, :], pm[:, None], 0.) * step
    ddo = tl.load(drow[:, None] + other[None, :], pm[:, None], 0.) * step
    dr = tl.where(low[None, :], dd * cs[None, :] + ddo * sn[None, :], -ddo * sn[None, :] + dd * cs[None, :])
    square = tl.sum(r0 * r0, 0)
    cross = tl.sum(r0[None, :] * dr, 1)
    sqd = tl.sum(dr * dr, 1)
    inverse = libdevice.rsqrt(square / D + EPS)
    moved = libdevice.rsqrt((square + 2 * cross + sqd) / D + EPS)
    delta = (dr * moved[:, None] + r0[None, :] * (moved - inverse)[:, None]) * (D ** -.5)
    if ROUND and QKP == 'tf32':
        delta = round_tf32(delta)
    base = tl.load(Base + head * OH + token * OM + f * OD)
    numerator = tl.zeros((BP, D), tl.float32)
    partition = tl.zeros((BP,), tl.float32)
    lo = (tl.maximum(pos - window, 0) // BK) * BK
    for start in range(lo, pos + 1, BK):
        key = start + tl.arange(0, BK)
        km = key <= pos
        a = tl.load(A + head * AH + token * AM + key * AT, km, 0.)
        k = tl.load(K + head * KH + key[None, :] * KT + f[:, None] * KD, km[None, :], 0.)
        v = tl.load(V + head * VH + key[:, None] * VT + f[None, :] * VD, km[:, None], 0.)
        vo = v - base[None, :]
        if ROUND:
            vo = round_tf32(vo)
            if QKP == 'tf32':
                k = round_tf32(k)
        change = a[None, :] * (tl.exp(tl.dot(delta, k, input_precision=QKP)) - 1)
        partition += tl.sum(change, 1)
        if ROUND:
            numerator = tl.dot(round_tf32(change), vo, numerator, input_precision=PRECISION)
        else:
            numerator = tl.dot(change, vo, numerator, input_precision=PRECISION)
    response = numerator / (1 + partition[:, None])
    tl.store(Out + ((token * P + probe[:, None]) * H + head).to(tl.int64) * D + f[None, :], response, pm[:, None])


@triton.jit
def mgs_rows_kernel(B, Q, N, K: tl.constexpr, KP: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """orthonormal() for BR rows n of B [N, K, D] per program ([BR, K, D] tiles): modified Gram-Schmidt,
    twice per vector after the first, in fp32 (FMA sums instead of the TF32 einsums), the same threshold (a vector whose
    remainder is <= 1e-4 x the largest input norm of its row becomes zero)."""
    pid = tl.program_id(0)
    n = pid * BR + tl.arange(0, BR)
    kk = tl.arange(0, KP)
    f = tl.arange(0, D)
    m3 = (n < N)[:, None, None] & (kk < K)[None, :, None]
    off = (n.to(tl.int64)[:, None, None] * K + kk[None, :, None]) * D + f[None, None, :]
    rows = tl.load(B + off, m3, 0.)
    scale = tl.maximum(tl.max(tl.sqrt(tl.sum(rows * rows, 2)), 1), 1e-30)  # [BR]
    q = tl.zeros((BR, KP, D), tl.float32)
    for k in tl.static_range(K):
        sel = (kk == k)[None, :, None]
        v = tl.sum(tl.where(sel, rows, 0.), 1)  # [BR, D]
        if k > 0:
            for _ in tl.static_range(2):
                c = tl.sum(q * v[:, None, :], 2)  # [BR, KP]
                v = v - tl.sum(c[:, :, None] * q, 1)
        nrm = tl.sqrt(tl.sum(v * v, 1))  # [BR]
        qk = tl.where((nrm > 1e-4 * scale)[:, None], v / tl.maximum(nrm, 1e-30)[:, None], 0.)
        q = tl.where(sel, qk[:, None, :], q)
    tl.store(Q + off, q, m3)


@triton.jit
def table_kernel(WT, WF, SW, G, Out, PI, PJ, V, W: tl.constexpr, B: tl.constexpr, BK: tl.constexpr,
                 SCALE_S: tl.constexpr, SCALE_H: tl.constexpr):
    """Token t, tile pair (PI[p], PJ[p]) (i <= j): (W^T diag(s_t) W - g_t g_t^T) SCALE_H in fp16, stored and mirrored."""
    p, t = tl.program_id(0), tl.program_id(1)
    bi, bj = tl.load(PI + p), tl.load(PJ + p)
    ri, rj = bi * B + tl.arange(0, B), bj * B + tl.arange(0, B)
    acc = tl.zeros((B, B), tl.float32)
    for v0 in range(0, V, BK):
        vv = v0 + tl.arange(0, BK)
        s = tl.load(SW + t * V + vv) * SCALE_S
        a = (tl.load(WT + ri[:, None] * V + vv[None, :]).to(tl.float32) * s[None, :]).to(tl.float16)
        acc = tl.dot(a, tl.load(WF + vv[:, None] * W + rj[None, :]), acc)
    acc = acc * (1 / SCALE_S)
    acc = tl.where(bi == bj, (acc + tl.trans(acc)) * 0.5, acc)  # diagonal tiles exactly symmetric: the two stores agree
    h = ((acc - tl.load(G + t * W + ri)[:, None] * tl.load(G + t * W + rj)[None, :]) * SCALE_H).to(tl.float16)
    out = Out + t.to(tl.int64) * W * W
    tl.store(out + ri[:, None] * W + rj[None, :], h)
    tl.store(out + rj[None, :] * W + ri[:, None], h)


@triton.jit
def eval_kernel(A, H, G, HB, TS, STATS, Out, n_rows, R, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, SCALE_H: tl.constexpr):
    """BM rows of token t: dinv ab + inv g.d + (dinv^2 sbb + 2 dinv inv hb.d + inv^2 d^T H d) / 2 - dc."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_rows
    row = (t * R + rows).to(tl.int64)
    ht = H + t.to(tl.int64) * W * W
    q = tl.zeros((BM,), tl.float32)
    gd = tl.zeros((BM,), tl.float32)
    hd = tl.zeros((BM,), tl.float32)
    for n0 in range(0, W, BN):
        cols = n0 + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, W, BK):
            kk = k0 + tl.arange(0, BK)
            a = tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.).to(tl.float16)
            acc = tl.dot(a, tl.load(ht + kk[:, None] * W + cols[None, :]), acc)
        d = tl.load(A + row[:, None] * W + cols[None, :], ok[:, None], 0.).to(tl.float32)
        q += tl.sum(acc * d, 1)
        gd += tl.sum(d * tl.load(G + t * W + cols)[None, :], 1)
        hd += tl.sum(d * tl.load(HB + t * W + cols)[None, :], 1)
    inv, dinv = tl.load(STATS + row * 4, ok, 1.), tl.load(STATS + row * 4 + 1, ok, 0.)
    ab, sbb = tl.load(TS + 2 * t), tl.load(TS + 2 * t + 1)
    lin = dinv * ab + inv * gd
    sec = dinv * dinv * sbb + 2 * dinv * inv * hd + inv * inv * q * (1 / SCALE_H)
    tl.store(Out + row, lin + 0.5 * sec - tl.load(STATS + row * 4 + 2, ok, 0.), ok)


@triton.jit
def fix_kernel(A, WT, STATS, BASE, T0, P0, Out, Count, n_rows, R, limit, V: tl.constexpr, W: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """A block of BM rows of token t with a bound >= limit: the exact head arithmetic (logits GEMM, tanh, expm1), Out
    overwritten."""
    pb, t = tl.program_id(0), tl.program_id(1)
    rows = pb * BM + tl.arange(0, BM)
    ok = rows < n_rows
    row = (t * R + rows).to(tl.int64)
    if tl.max(tl.load(STATS + row * 4 + 3, ok, 0.), 0) >= limit:
        tl.atomic_add(Count, 1)
        inv = tl.load(STATS + row * 4, ok, 1.)[:, None]
        dinv = tl.load(STATS + row * 4 + 1, ok, 0.)[:, None]
        part = tl.zeros((BM,), tl.float32)
        for v0 in range(0, V, BN):
            cols = v0 + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), tl.float32)
            for k0 in range(0, W, BK):
                kk = k0 + tl.arange(0, BK)
                acc = tl.dot(tl.load(A + row[:, None] * W + kk[None, :], ok[:, None], 0.),
                             tl.load(WT + kk[:, None] * V + cols[None, :]), acc)
            t0 = tl.load(T0 + t * V + cols)[None, :]
            th = libdevice.tanh((tl.load(BASE + t * V + cols)[None, :] * dinv + acc * inv) * (1 / 15))
            part += tl.sum(tl.load(P0 + t * V + cols)[None, :] * libdevice.expm1(15 * th * (1 - t0 * t0) / (1 + t0 * th)), 1)
        tl.store(Out + row, libdevice.log1p(part) - tl.load(STATS + row * 4 + 2, ok, 0.), ok)


@triton.jit
def plan_kernel(IDX, VALS, KEPT, QD, R0, KRAW, CS, SN, P, LIVE, COEF, XLIN, T,
                H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BT: tl.constexpr, TAU: tl.constexpr,
                SIGMA: tl.constexpr, EPS: tl.constexpr, SQD: tl.constexpr):
    """The receiver plan's key experiments for keys t (BT per program) of head h, R receivers each (vectors as halves
    [BT, R, D / 2]): the pull-backs p_r (key-norm Jacobian, inverse RoPE), the guard (Gram-Schmidt twice in receiver order;
    accepted iff kept and >= TAU |p_r| outside the stronger accepted ones' span), the dual basis L^-T Q, x_lin = step /
    |d_r|, the move's exact key change (RoPE + QK-norm), the receiver's logit change x and gain a (e^x - 1) / (1 +
    a (e^x - 1)). Out: P [H, T, R, D], live, gain and x_lin [H, T, R]."""
    pt, h = tl.program_id(0), tl.program_id(1)
    HD: tl.constexpr = D // 2
    t = pt * BT + tl.arange(0, BT)
    r = tl.arange(0, R)[None, :]
    d = tl.arange(0, HD)[None, None, :]
    on_t = (t < T)[:, None]
    htr = (h * T + t)[:, None] * R + r  # [BT, R]
    rec = tl.load(IDX + htr, on_t, 0)
    val = tl.load(VALS + htr, on_t, 0.)
    kept = tl.load(KEPT + htr, on_t, 0) != 0
    qo = (rec * H + h)[:, :, None] * D + d
    qlo, qhi = tl.load(QD + qo, on_t[:, :, None], 0.), tl.load(QD + qo + HD, on_t[:, :, None], 0.)
    uo = (t * H + h)[:, None, None] * D + d
    ulo, uhi = tl.load(R0 + uo, on_t[:, :, None], 0.), tl.load(R0 + uo + HD, on_t[:, :, None], 0.)  # [BT, 1, HD]
    klo, khi = tl.load(KRAW + uo, on_t[:, :, None], 0.), tl.load(KRAW + uo + HD, on_t[:, :, None], 0.)
    co = t[:, None, None] * HD + d
    c, s = tl.load(CS + co, on_t[:, :, None], 1.), tl.load(SN + co, on_t[:, :, None], 0.)
    sq = tl.sum(ulo * ulo, 2) + tl.sum(uhi * uhi, 2)  # [BT, 1]
    ms = (sq / D + EPS)[:, :, None]
    uq = (tl.sum(ulo * qlo, 2) + tl.sum(uhi * qhi, 2))[:, :, None]
    jlo, jhi = (qlo - ulo * uq / (D * ms)) / tl.sqrt(ms), (qhi - uhi * uq / (D * ms)) / tl.sqrt(ms)
    plo, phi = (jlo * c - jhi * s) / SQD, (jlo * s + jhi * c) / SQD  # p_r = d logit / d k_raw
    nrm = tl.sqrt(tl.sum(plo * plo, 2) + tl.sum(phi * phi, 2))  # [BT, R]
    Qlo, Qhi = tl.zeros([BT, R, HD], tl.float32), tl.zeros([BT, R, HD], tl.float32)
    L = tl.zeros([BT, R, R], tl.float32)  # P_acc = L Q, row i: the accepted receiver i's coefficients
    acc = tl.zeros([BT, R], tl.int32)
    for i in tl.static_range(R):
        on = r == i
        vlo, vhi = tl.sum(tl.where(on[:, :, None], plo, 0.), 1), tl.sum(tl.where(on[:, :, None], phi, 0.), 1)
        row = tl.zeros([BT, R], tl.float32)
        if i > 0:
            for _ in tl.static_range(2):
                cf = tl.sum(Qlo * vlo[:, None, :], 2) + tl.sum(Qhi * vhi[:, None, :], 2)
                vlo, vhi = vlo - tl.sum(cf[:, :, None] * Qlo, 1), vhi - tl.sum(cf[:, :, None] * Qhi, 1)
                row += cf
        rn = tl.sqrt(tl.sum(vlo * vlo, 1) + tl.sum(vhi * vhi, 1))  # [BT]
        ni = tl.sum(tl.where(on, nrm, 0.), 1)
        good = (tl.sum(tl.where(on & kept, 1, 0), 1) > 0) & (rn >= TAU * ni) & (ni > 0)
        put = on[:, :, None] & good[:, None, None]
        div = tl.maximum(rn, 1e-30)[:, None]
        Qlo, Qhi = tl.where(put, (vlo / div)[:, None, :], Qlo), tl.where(put, (vhi / div)[:, None, :], Qhi)
        L = tl.where(put, tl.where(on, rn[:, None], row)[:, None, :], L)
        acc = tl.where(on & good[:, None], 1, acc)
    Dlo, Dhi = tl.zeros([BT, R, HD], tl.float32), tl.zeros([BT, R, HD], tl.float32)
    for j in tl.static_range(R):  # d_i = (q_i - sum_{k > i} L_ki d_k) / L_ii, from the last receiver down
        on = r == R - 1 - j
        col = tl.sum(tl.where(r[:, None, :] == R - 1 - j, L, 0.), 2)  # L[k, i] (k = i: d_i is still zero)
        good = tl.sum(tl.where(on, acc, 0), 1) > 0
        diag = tl.where(good, tl.sum(tl.where(on, col, 0.), 1), 1.)[:, None]
        dlo = (tl.sum(tl.where(on[:, :, None], Qlo, 0.), 1) - tl.sum(col[:, :, None] * Dlo, 1)) / diag
        dhi = (tl.sum(tl.where(on[:, :, None], Qhi, 0.), 1) - tl.sum(col[:, :, None] * Dhi, 1)) / diag
        put = on[:, :, None] & good[:, None, None]
        Dlo, Dhi = tl.where(put, dlo[:, None, :], Dlo), tl.where(put, dhi[:, None, :], Dhi)
    live = acc > 0
    step = SIGMA * tl.maximum(tl.sqrt(tl.sum(klo * klo, 2) + tl.sum(khi * khi, 2)), 1e-12)  # [BT, 1]
    dn = tl.sqrt(tl.sum(Dlo * Dlo, 2) + tl.sum(Dhi * Dhi, 2))
    xlin = tl.where(live, step / tl.maximum(dn, 1e-30), 0.)  # the first-order logit change of receiver r
    mlo, mhi = xlin[:, :, None] * Dlo, xlin[:, :, None] * Dhi
    rlo, rhi = mlo * c + mhi * s, mhi * c - mlo * s  # RoPE at t
    grow = (2 * (tl.sum(ulo * rlo, 2) + tl.sum(uhi * rhi, 2)) + tl.sum(rlo * rlo, 2) + tl.sum(rhi * rhi, 2)) / D
    inverse = libdevice.rsqrt(sq / D + EPS)
    moved = libdevice.rsqrt(sq / D + grow + EPS)
    shrink = (grow * (inverse * moved) * (inverse * moved) / (inverse + moved))[:, :, None]
    dklo, dkhi = rlo * moved[:, :, None] - ulo * shrink, rhi * moved[:, :, None] - uhi * shrink  # QK-norm change
    x = (tl.sum(qlo * dklo, 2) + tl.sum(qhi * dkhi, 2)) / SQD
    ex = libdevice.expm1(x)
    po = htr[:, :, None] * D + d
    tl.store(P + po, plo, on_t[:, :, None])
    tl.store(P + po + HD, phi, on_t[:, :, None])
    tl.store(LIVE + htr, live, on_t)
    tl.store(COEF + htr, tl.where(live, val * ex / (1 + val * ex), 0.), on_t)
    tl.store(XLIN + htr, xlin, on_t)


@triton.jit
def routes_kernel(LG, PROB, IDX, PV, T, window, SCALE: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                  BN: tl.constexpr):
    """Query t of head h: the masked softmax of its bf16 logits row (keys t - window .. t), the probabilities as bf16
    (PROB), and the K largest (IDX int32 / PV fp32 at [t, h, :], in key order): threshold = the K-th largest 16-bit key
    of the bf16 logits (monotone in the logit), found by bisection; ties at it by key index."""
    t, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, BN)
    row = (h * T + t).to(tl.int64) * T
    allowed = (j <= t) & (j >= t - window)
    lg = tl.load(LG + row + j, j < T, 0.)
    bits = lg.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key = tl.where(allowed, tl.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000), tl.where(j < T, 0, -1))
    z = tl.where(allowed, lg.to(tl.float32) * SCALE, float('-inf'))
    e = libdevice.exp(z - tl.max(z, 0))
    p = e / tl.sum(e, 0)
    tl.store(PROB + row + j, p.to(tl.bfloat16), j < T)
    lo, hi = 0, 65536
    for _ in tl.static_range(16):
        mid = (lo + hi) // 2
        big = tl.sum((key >= mid).to(tl.int32), 0) >= K
        lo, hi = tl.where(big, mid, lo), tl.where(big, hi, mid)
    eq = key == lo
    sel = (key > lo) | (eq & (tl.cumsum(eq.to(tl.int32), 0) <= K - tl.sum((key > lo).to(tl.int32), 0)))
    out = (t * H + h).to(tl.int64) * K + tl.cumsum(sel.to(tl.int32), 0) - 1
    tl.store(IDX + out, j, sel)
    tl.store(PV + out, p, sel)


# ------------------------------------------------------------------------------------------- the true-path readout


BLOCKS, FIRST = (0, 1, 2, 3), 4  # blocks 0-3; the first block past them
K_SIGMA, GATE_SIGMA, K_TAU = .01, .01, .1  # receiver experiments: raw-key step, gate-logit step, guard
HI_SEED = 7_777_777  # the hi blocks' per-sequence generator: the pass's seed + this + 104729 x sequence
UP = (128, 256, 64, 8, 4)  # mlp_up_prune_kernel: rows, units per tile, K step, warps, stages
DOWN = (64, 128, 128, 4, 3)  # mlp_down_live_kernel: rows, columns, unit tile, warps, stages
MARGIN = 1.02  # the MLP prune bound's safety factor
OQ_ATTN = (64, 4, 1)  # oq_attn_kernel: rows, warps, stages
PROJ = (64, 128, 64, 4, 4)  # dual_residual_kernel: BM, BN, BK, warps, stages
HEAD_PREP = (32, 128, 4)  # head_prep_kernel: BR, BK, warps
INROOT_TILES = {64: (64, 4, 3), 256: (128, 4, 3)}  # probes per token -> probes per program, warps, stages
QSRC2_TILES = {64: (32, 4, 2), 256: (32, 4, 3)}
MGS_TILE = dict(rows=8, warps=4)
PLAN_BT, PLAN_WARPS = 4, 2  # plan_kernel: keys per program, warps
LIMIT, SCALE_S, SCALE_H = 0.1, 4096., 1024.  # the chain head's expansion: exact above this bound; table scales
TABLE = (128, 64, 8, 3)  # table_kernel: B, BK, warps, stages
EVAL = (64, 128, 64, 4, 3)  # eval_kernel: BM, BN, BK, warps, stages
BT = 512  # moment tables: sources per banded bmm
KEY_TILES = (32, 2, 4, 2)  # key_kernel: sources, trials, warps, stages
HEAD_STATE = dict(cache={}, pairs={}, fixed=None)


def group_rows(s, heads):
    """Rows per receiver token of one group: readout_k_cap x the mean load heads x R, rounded up to 16, at least 64
    (the own-query kernel takes >= 64 rows per token)."""
    return max(64, 16 * math.ceil(s.readout_k_cap * heads * s.readout_k_receivers / 16))


def receiver_groups(blocks, model, split, part):
    """This GPU's receiver groups (block, 'k' | 'g') of these blocks in insertion order (k, then the gate where block b
    has value embeddings): a pair's GPUs split them heaviest first (rows inserted at block b pass blocks b + 1 .. L - 1)
    onto the least loaded part, ties to the lower part."""
    L = model.config.n_layer
    groups = [(b, k) for b in sorted(blocks) for k in 'kg'
              if k == 'k' or model.transformer.h[b].attn.ve_gate is not None]
    load, owner = [0] * split, {}
    for g in sorted(groups):  # (block, kind): smaller block = heavier; 'g' before 'k'
        q = min(range(split), key=lambda q: (load[q], q))
        owner[g] = q
        load[q] += L - 1 - g[0]
    return [g for g in groups if owner[g] == part]


def receiver_slots(idx, vals, ok, C, T):
    """Pairs (h, t, r) [H, T, R] (receiver idx, clean attention vals, valid ok) -> the slot table [T, C] of each receiver
    token (flat pair index (h T + t) R + r, -1 empty; at most C per receiver, the strongest attention first) and the kept
    mask [H, T, R]. No host synchronization."""
    n = idx.numel()
    rec = idx.reshape(-1)
    valid = ok.reshape(-1)
    order = torch.argsort(torch.where(valid, vals.reshape(-1), -1.), descending=True, stable=True)
    order = order[torch.argsort(rec[order], stable=True)]  # by receiver, the strongest first
    rs = rec[order]
    counts = torch.zeros(T, dtype=torch.long, device=idx.device).scatter_add_(0, rs, torch.ones_like(rs))
    rank = torch.arange(n, device=idx.device) - (torch.cumsum(counts, 0) - counts)[rs]
    keep = (rank < C) & valid[order]
    table = torch.full((T * C + 1,), -1, dtype=torch.long, device=idx.device)  # the last entry: a sink for the rest
    table.scatter_(0, torch.where(keep, rs * C + rank, T * C), torch.where(keep, order, -1))
    kept = torch.zeros(n, dtype=torch.bool, device=idx.device).scatter_(0, order, keep)
    return table[:T * C].view(T, C), kept.view_as(ok)


def query_routes(own, k):
    """Per block, each token's own-query route: the k strongest clean keys of its query (routes_kernel: the masked
    softmax of the bf16 logits, the top-k by bisection on their 16-bit keys, the bf16 probabilities), the clean
    output o, the rotated query projection r0 and |r0|^2. Positions are 0..T-1 (Cut.own_query_inputs')."""
    pos, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
    length = len(pos)
    K = min(k, length)
    out = []
    for x, wq, kk, vv, window in zip(inputs, w_q, keys, values, windows):
        heads, _, width = kk.shape
        c, s = cos[:, None], sin[:, None]
        r = (F.rms_norm(x, (x.shape[-1],)) @ wq.T).view(length, heads, width)
        r0 = torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                        r[..., :width // 2] * -s + r[..., width // 2:] * c], -1)
        q = F.rms_norm(r0, (width,))
        logits = torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk)
        prob = torch.empty_like(logits)
        idx = torch.empty(length, heads, K, device=x.device, dtype=torch.int32)
        p = torch.empty(length, heads, K, device=x.device)
        routes_kernel[(length, heads)](logits, prob, idx, p, length, window, width ** -0.5, K, heads,
                                       triton.next_power_of_2(length), num_warps=4)
        o = torch.einsum('hts,hsd->htd', prob, vv).float()
        out.append(dict(idx=idx, p=p, o=o.transpose(0, 1).contiguous(), r0=r0.contiguous(),
                        sq0=r0.square().sum(-1).contiguous(), keys=kk.contiguous(), values=vv.contiguous()))
        del logits, prob
    return out


def sort_routes(w_fc, routes):
    """Per block: each token's routed units ordered by s = h / |W_fc[u]| (bf16 weights; active or NaN: +inf),
    descending, by fp16 keys (stable): (idx int32, h fp32, s fp32), each (n, K) contiguous. Any order is exact (a row
    block's prune point is one past the LAST unit it may change); a good order only makes the live prefix short."""
    idx32, hs_all, sc_all = [], [], []
    for w, (idx, hs) in zip(w_fc, routes):
        wn = w.float().norm(dim=1)
        hs = hs.float()
        s = torch.where((hs >= 0) | hs.isnan(), torch.full_like(hs, float('inf')), hs / wn[idx])
        order = s.half().argsort(dim=1, descending=True, stable=True)
        idx32.append(idx.gather(1, order).to(torch.int32).contiguous())
        hs_all.append(hs.gather(1, order).contiguous())
        sc_all.append(s.gather(1, order).contiguous())
    return idx32, hs_all, sc_all


def own_query_step(delta, n, b, ctx, lists=None):
    """delta[:, :n] += c_proj(the own-query response), then += c_proj(the self-edge ctx['se']) at block b: the
    projection d W_q^T from the scorer's bf16 copy ctx['db'] (rows packed n per token) by cuBLAS (fp32 out),
    oq_attn_kernel on it with the scorer's row statistics ctx['st'], then both c_proj products in one pass (with
    live-row lists: each token's live rows only)."""
    db, st, se = ctx['db'], ctx['st'], ctx['se']
    m, rows, W = delta.shape
    q = ctx['routes'][b]
    keys = q['keys']
    H, T, DH = keys.shape
    u = ctx['scratch32'][:m * n * W].view(m * n, W)
    torch.mm(db[:m * n * W].view(m * n, W), ctx['w_q_t'][b], out_dtype=torch.float32, out=u)
    buf = ctx['ybuf'][:m * n * W]
    bp, warps, stages = OQ_ATTN
    oq_attn_kernel[(H, m)](
        u, st, q['r0'], q['sq0'], ctx['cos'], ctx['sin'], keys, q['values'], q['idx'], q['p'], q['o'], buf,
        n, n, n, ctx['t0'], T, H, DH, W, q['idx'].shape[-1], bp, EPS, num_warps=warps, num_stages=stages)
    bm, bn, bk, warps, stages = PROJ
    args = (buf, se, ctx['w_proj_t'][b], delta, m * n, n, n, rows, rows, W, W, bm, bn, bk, True)
    if lists is None:
        dual_residual_kernel[(W // bn, triton.cdiv(m * n, bm))](*args, num_warps=warps, num_stages=stages)
    else:
        dual_residual_kernel_live[(W // bn, triton.cdiv(n, bm), lists[1].shape[0])](
            lists[0], lists[1], lists[1].shape[1], *args, num_warps=warps, num_stages=stages)


def mlp_step(delta, n_mlp, b, ctx, dx0, P):
    """Block b's routed MLP on rows < n_mlp, then block b + 1's mixing scalars (dx0: the embedding rows' x0 change, the
    first P rows; P = 0: none): exact unit pruning (each token's routed units in sort_routes' order; a row block's prune
    point from its rows' bound; skipped terms are exact zeros), mlp_up_prune_kernel then mlp_down_live_kernel."""
    m, rows, W = delta.shape
    idx, hs, sc, mid = ctx['idx_s'][b], ctx['hs_s'][b], ctx['sc_s'][b], ctx['mids'][b]
    K = idx.shape[1]
    rbuf = ctx['scratch'][:m * rows * K].view(m, rows, K)
    ubp, ubn, ubk, warps, stages = UP
    n_rb = triton.cdiv(n_mlp, ubp)
    cut = ctx['cut'][:m * n_rb]
    mlp_up_prune_kernel[(n_rb, m)](
        delta, mid, idx, ctx['w_fc'][b], hs, sc, cut, rbuf, n_mlp, rows, rows, n_rb, W, K, triton.next_power_of_2(K),
        ubp, ubn, ubk, EPS, MARGIN, num_warps=warps, num_stages=stages)
    dbp, dbn, dbk, warps, stages = DOWN
    scale = dx0 is not None
    mlp_down_live_kernel[(W // dbn, triton.cdiv(n_mlp, dbp), m)](
        rbuf, idx, ctx['w_out_t'][b], delta, dx0 if scale else delta, ctx['lams'][min(b + 1, len(ctx['lams']) - 1)],
        cut, delta, delta, delta, delta, delta, n_mlp, rows, rows, P, n_rb, W, K, dbp, dbn, dbk, ubp, scale, scale,
        False, False, False, num_warps=warps, num_stages=stages)


def prune_buffers(m, rows, dev):
    """The pruned MLP's prune points (per token and row block)."""
    return dict(cut=torch.empty(m * triton.cdiv(rows, UP[0]), device=dev, dtype=torch.int32))


def sorted_slices(sorted_routes, sl):
    idx, hs, sc = sorted_routes
    return dict(idx_s=[z[sl] for z in idx], hs_s=[z[sl] for z in hs], sc_s=[z[sl] for z in sc])


def inroot_scores(dq, qraw, keys, values, a, base, target, t0, cs, sn, window):
    """inroot_kernel on dq [m, p, W] (the raw-query moves of tokens t0 .. t0 + m), a / base the attention rows / outputs
    of these tokens ([H, m, T] / [H, m, D] views), target [m, W]: the in-root scores [m, p] (one-pass TF32 dots on
    operands rounded to nearest)."""
    m, p, _ = dq.shape
    H, T, D = keys.shape
    out = torch.empty(m, p, device=dq.device)
    bp, warps, stages = INROOT_TILES.get(p, INROOT_TILES[64])
    inroot_kernel[(triton.cdiv(p, bp), m)](
        dq, qraw, cs, sn, keys, values, a, base, target, out, t0, window, p, H, D, *keys.stride(), *values.stride(),
        *a.stride(), *base.stride(), bp, 64, 'tf32', EPS, True, num_warps=warps, num_stages=stages)
    return out


def qsrc2_response(dirs, qraw, qstep, keys, values, a, base, t0, cs, sn, window):
    """qsrc2_kernel: the q moves' attention responses [m, p, H, D] (one-pass TF32 dots on operands rounded to
    nearest)."""
    m, p, _ = dirs.shape
    H, T, D = keys.shape
    out = torch.empty(m, p, H, D, device=dirs.device)
    bp, warps, stages = QSRC2_TILES[64 if p <= 64 else 256]
    qsrc2_kernel[(triton.cdiv(p, bp), m * H)](
        dirs, qraw, qstep, cs, sn, keys, values, a, base, out, t0, window, p, H, D, *keys.stride(), *values.stride(),
        *a.stride(), *base.stride(), bp, 64, 'tf32', EPS, True, 'tf32', num_warps=warps, num_stages=stages)
    return out


def orthonormal(B):
    """Rows of B [N, K, D] -> orthonormal rows spanning the same space (mgs_rows_kernel: modified Gram-Schmidt, twice
    per row; a row already in the span of the earlier ones becomes zero, so its code slot moves nothing and decodes
    nothing)."""
    N, K, D = B.shape
    B = B.contiguous()
    Q = torch.empty_like(B)
    br = MGS_TILE['rows']
    mgs_rows_kernel[(triton.cdiv(N, br),)](B, Q, N, K, triton.next_power_of_2(K), D, br, num_warps=MGS_TILE['warps'])
    return Q


def head_pairs(device, n):
    if (device, n) not in HEAD_STATE['pairs']:
        ij = [(i, j) for i in range(n) for j in range(i, n)]
        HEAD_STATE['pairs'][device, n] = [torch.tensor(v, device=device, dtype=torch.int32) for v in zip(*ij)]
    return HEAD_STATE['pairs'][device, n]


def head_tables(ctx, t0, p0, cache):
    """(H fp16 [m, W, W], g [m, W], hb [m, W], (ab, sbb) [m, 2]) of the call's tokens; cached per chain call if asked."""
    weight_t, base, c = ctx['weight_t'], ctx['base'], HEAD_STATE['cache']
    key = ctx['xL'].data_ptr()
    if cache:
        if c.get('weight_t') is not weight_t:  # a new chain call (the cache holds the old weight_t: no id reuse)
            c.clear()
            c['weight_t'] = weight_t
        if key in c:
            return c[key][1:]
    W, (m, V) = weight_t.shape[0], base.shape
    cp = 1 - t0 * t0
    aw = p0 * cp
    sw = (aw * (cp - t0 * (2 / 15))).contiguous()
    gh = torch.stack([aw, sw * base], 1).double() @ weight_t.double().T  # [m, 2, W]
    ab = (aw.double() * base).sum(-1)
    g = gh[:, 0]
    hb = (gh[:, 1] - ab[:, None] * g).float().contiguous()
    ts = torch.stack([ab, (sw.double() * base.double().square()).sum(-1) - ab * ab], 1).float().contiguous()
    g = g.float().contiguous()
    B, BK, warps, stages = TABLE
    pi, pj = head_pairs(base.device, W // B)
    H = torch.empty(m, W, W, device=base.device, dtype=torch.float16)
    table_kernel[(len(pi), m)](weight_t, weight_t.T.half().contiguous(), sw, g, H, pi, pj, V, W, B, BK, SCALE_S, SCALE_H,
                               num_warps=warps, num_stages=stages)
    if cache:
        c[key] = (ctx['xL'], H, g, hb, ts)  # the xL view keeps its storage (the key) alive
    return H, g, hb, ts


def head_expand(a, stats, ctx, t0, p0, cache):
    """change [m, rows] from the bf16 rows a [m, rows, W] and the stats [m, rows, 4] (inv', dinv, dc_target, bound)."""
    m, rows, W = a.shape
    H, g, hb, ts = head_tables(ctx, t0, p0, cache)
    out = torch.empty(m, rows, device=a.device)
    bm, bn, bk, warps, stages = EVAL
    eval_kernel[(triton.cdiv(rows, bm), m)](a, H, g, hb, ts, stats, out, rows, rows, W, bm, bn, bk, SCALE_H,
                                            num_warps=warps, num_stages=stages)
    if HEAD_STATE['fixed'] is None:
        HEAD_STATE['fixed'] = torch.zeros(1, device=a.device, dtype=torch.int32)
    fix_kernel[(triton.cdiv(rows, 64), m)](a, ctx['weight_t'], stats, ctx['base'], t0, p0, out, HEAD_STATE['fixed'], rows,
                                           rows, LIMIT, t0.shape[1], W, 64, 128, 64, num_warps=8, num_stages=1)
    return out


def head_fast(delta, ctx):
    """(loss(moved) - loss(clean)) per row of delta (m, R, W): head_prep_kernel, then the second-order expansion (tables
    cached per chain call)."""
    m, rows, W = delta.shape
    xL, base, y = ctx['xL'], ctx['base'], ctx['y']
    sq = xL.square().sum(-1, keepdim=True)
    t0 = torch.tanh(base * torch.rsqrt(sq / W + EPS) / 15)
    p0 = torch.softmax(15 * t0, -1)
    target = y.clamp(min=0).unsqueeze(1)
    tgt = (base.gather(1, target).contiguous(), t0.gather(1, target).contiguous(), base.abs().amax(-1).contiguous(),
           ctx['head_row_norm'])
    a = ctx['scratch'][:m * rows * W].view(m, rows, W)
    stats = ctx['scratch32'][:m * rows * 4]
    br, bk, warps = HEAD_PREP
    head_prep_kernel[(triton.cdiv(rows, br), m)](delta, ctx['xL'], ctx['target_w'], sq, *tgt, a, stats, rows, rows,
                                                 W, br, bk, EPS, num_warps=warps)
    return head_expand(a, stats.view(m, rows, 4), ctx, t0, p0, True)


@contextlib.contextmanager
def ieee():  # fp32 GEMMs without TF32 (the legacy API: the recipe sets 'high' with it)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@torch.compile(dynamic=False)
def _weights(a, c, window: int, future: bool):
    """a, c [H, T(query s), T(source t)] -> a c (fp32), a c (1/2 - a) (bf16) [, a (fp32), a (1 - a) (bf16)] on the pairs
    t < s <= t + window, 0 elsewhere."""
    pos = torch.arange(a.shape[-1], device=a.device)
    lag = pos[:, None] - pos[None, :]
    a = torch.where((lag > 0) & (lag <= window), a, 0.)
    ac = a * c
    w = (ac, (ac * (0.5 - a)).bfloat16())
    return w + (a, (a * (1 - a)).bfloat16()) if future else w


def _outer(x, y):
    """vec(x_n y_n^T) [B, N, D * D] bf16 of x, y [B, N, D] (unit last stride; the fp32 product rounded once)."""
    B, N, D = x.shape
    out = torch.empty((B, N, D * D), device=x.device, dtype=torch.bfloat16)
    outer_kernel[(B, triton.cdiv(N, 4))](x, y, out, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1), D, 4)
    return out


def moment_tables(q, a, c, window, g=None):
    """U [T, H, D] fp32, M1 [T, H, D, D] bf16 (with g: U, W, M1, M2) of the queries t < s <= t + window; q, g [H, T, D]."""
    H, T, D = q.shape
    w = _weights(a, c, window, g is not None)
    lin, quad = [], []
    for y, wl, wq in [(q, w[0], w[1])] + ([(g, w[2], w[3])] if g is not None else []):
        V = q.new_empty((T, H, D))
        with ieee():
            torch.bmm(wl.mT, y, out=V.transpose(0, 1))
        M = torch.empty((T, H, D, D), device=q.device, dtype=torch.bfloat16)
        qy, Mv = _outer(q, y), M.view(T, H, D * D)
        for t0 in range(0, T, BT):  # sources t0:t1 read the queries t0:s1 (s = t0 has weight 0)
            t1 = min(T, t0 + BT)
            s1 = min(T, t1 + window)
            torch.bmm(wq[:, t0:s1, t0:t1].mT, qy[:, t0:s1], out=Mv[t0:t1].transpose(0, 1))
        lin.append(V)
        quad.append(M)
    return (*lin, *quad)


def _prepare(M1, U, qp, px, kraw, cos, sin, a_self, c_self):
    """prep_kernel's Mt [T, H, D * D] bf16, V [H, 6, T, D], S [12, H, 1, T]; px [T, H, D] contiguous; qp, kraw [H, T, D]
    (unit last stride); a_self, c_self [H, T]."""
    T, H, D = px.shape
    Mt = torch.empty((T, H, D * D), device=px.device, dtype=torch.bfloat16)
    V = px.new_empty((H, 6, T, D))
    S = px.new_empty((12, H, 1, T))
    inputk_prep_kernel[(T, H)](M1, U, qp, px, kraw, cos, sin, a_self, c_self, Mt, V, S, T, qp.stride(0), qp.stride(1),
                               kraw.stride(0), kraw.stride(1), *a_self.stride(), *c_self.stride(), H, D, num_warps=4)
    return Mt, V, S


@torch.compile(dynamic=False)
def _epilogue(G, Q, L, S, sq, step, dots, rsq, width: int):
    """[T, 2P] (+ then -) scores summed over heads, inputk_kernel's formulas on the expanded pieces: G [H, P, 6, T] =
    pr . (lam kr, lam px, R^T u, R^T q, Ms px, Ms kr), Q = rpr^T M1 rpr, L = |rpr|^2 [H, P, T]; S [12, H, 1, T]; sq,
    step [T]; dots [T, P] (x . rows); rsq [P] (|row|^2)."""
    D = width // G.shape[0]
    r0r, xr, ur, qr, vxr, v0r = G.unbind(2)
    r0x, xx, xu, r0u, xq, r0q, qx, q0, x0, r0sq, a_self, c_self = S.unbind(0)
    k0 = torch.rsqrt(r0sq / D + EPS)
    inv0 = torch.rsqrt(sq / width + EPS)
    rs = step * step * rsq[:, None]
    out = []
    for sign in (1., -1.):
        e = rs + 2 * sign * step * dots.T
        inv = torch.rsqrt((sq + rs + 2 * sign * step * dots.T) / width + EPS)
        dinv = -(e / width) * (inv0 * inv) * (inv0 * inv) / (inv0 + inv)
        b = sign * step * inv
        grow = (2 * (dinv * r0x + b * r0r) + dinv * dinv * xx + b * b * L + 2 * dinv * b * xr) / D
        k1 = torch.rsqrt(r0sq / D + grow + EPS)
        A, B, C = k1 * dinv, k1 * b, -grow * (k0 * k1) * (k0 * k1) / (k0 + k1)
        quad = A * A * qx + B * B * Q + C * C * q0 + A * B * vxr + A * C * x0 + B * C * v0r
        change = a_self * torch.expm1((A * xq + B * qr + C * r0q) * D ** -.5)
        out.append(((A * xu + B * ur + C * r0u) * D ** -.5 + quad / D + change / (1 + change) * c_self).sum(0))
    return torch.cat(out).T


def inputk_scores(xin, steps, rows, projected_x, projected_rows, raw, cos, sin, qp, A, C, tables):
    """[T, 2P] (+ then -) input-K experiment scores of one sequence and block, summed over heads, from GEMMs."""
    U, M1 = tables
    T, width = xin.shape
    H, _, D = raw.shape
    P = rows.shape[0]
    dots = xin @ rows.T  # as fast_moments forms them
    pr = projected_rows.reshape(P, H, D).transpose(0, 1)
    Mt, V, S = _prepare(M1.contiguous(), U.contiguous(), qp, projected_x.reshape(T, H, D).contiguous(), raw,
                        cos.contiguous(), sin.contiguous(), A.diagonal(0, 1, 2), C.diagonal(0, 1, 2))
    with ieee():
        G = torch.bmm(pr, V.flatten(1, 2).mT).view(H, P, 6, T)
        L = (projected_rows.square().reshape(P * H, 2, D // 2).sum(1) @ (cos * cos + sin * sin).T).view(P, H, T).transpose(0, 1)
    Q = torch.bmm(_outer(pr, pr), Mt.permute(1, 2, 0), out_dtype=torch.float32)
    return _epilogue(G, Q, L, S, xin.square().sum(-1), steps.reshape(-1), dots, rows.square().sum(-1), width)


def rawk_credit(kp, raw, steps, cos, sin, qp, A, C, tables, lo, hi):
    """Boost.key_errors_from_context's result [h, T, hd] (this GPU's dims lo:hi; zeros elsewhere): the record's moves,
    raw_key_moves per KEY_BUCKET, scored by the tables."""
    U, M1 = tables
    h, t, d = raw.shape
    result = torch.zeros_like(raw)
    if hi <= lo:
        return result
    eye = torch.eye(d, device=raw.device)
    for t0 in range(0, t, KEY_BUCKET):
        source = slice(t0, min(t0 + KEY_BUCKET, t))
        moves = raw_key_moves(kp[:, source], raw[:, source], steps[:, source], cos[source], sin[source], eye[lo:hi])
        rawk_kernel[(moves.shape[2], h)](
            moves, steps, U, M1, qp, A, C, result, t0, lo, hi - lo, *moves.stride(), steps.stride(0), steps.stride(1),
            qp.stride(0), qp.stride(1), A.stride(0), A.stride(1) + A.stride(2), C.stride(0), C.stride(1) + C.stride(2),
            result.stride(0), result.stride(1), h, d, max(16, triton.next_power_of_2(hi - lo)), num_warps=4,
            num_stages=1)
    return result


class Readout:
    """One Cut estimate's true-path readout (module docstring): the self-edged chain with the roots' readouts
    (chain_credits), the pass of the sources, receivers and in-roots (run), and the gradients they rebuild or add
    (finish)."""

    def __init__(self, est):
        self.e = est

    # ------------------------------------------------------------------------------------------------------ the chain

    def chain_credits(self, emb, x0, mids, outs, routes, y):
        """The chain with the self-edge at every block (its credits as Cut.estimate reads them: per root the stream
        credit, per block the mixing-scalar probes). The roots run from the top down: root r is pushed, decoded and
        summed over the pair before a lower root crosses block r, so its hub chain_r W_proj and block r's moment tables
        are ready, and every root's rows are scored at every later block on the way up (R_r = the decoded scores, in
        X). A phase holds one root's rows (the scalar probes ride with the embedding's, each from its mid stream)."""
        e, s = self.e, self.e.s
        model, d, dev = e.model, e.capture, e.device
        L, W, T, n = model.config.n_layer, e.width, d.length, emb.shape[0]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        lo, hi = e.share(W)
        P, scalars, h = hi - lo, e.part == 0, s.scalar_step
        self.chain_args = (emb, x0, mids, outs, routes, y)
        bases = [e.basis()[lo:hi].contiguous() for _ in range(L + 1)]  # the record's draws in its order: emb, mid0, ...
        steps = [s.chain_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6) for z in [emb, *mids]]
        self.prepare(mids, routes)
        head = model.lm_head.weight.float()[:model.config.vocab_size]
        xL = outs[-1].float().contiguous()
        base, target_w, weight_t = xL @ head.T, head[y.clamp(min=0)], head.T.to(torch.bfloat16).contiguous()
        hx = dict(weight_t=weight_t, head_row_norm=weight_t.float().norm(dim=0).amax().reshape(1))
        credit, score = torch.empty(L + 1, n, W, device=dev), torch.zeros(L + 1, n, P, device=dev)
        lam = torch.zeros(L, 2, device=dev)
        m_chunk, k4 = min(s.readout_chunk, T), 4 * scalars  # the scalar probes per root (a pair's first GPU)
        buf = self.buffers(m_chunk, P + L * k4)
        self.routes, self.contexts = {}, {}
        for seq in range(d.batch):
            own = e.own_query_inputs(seq)
            self.routes[seq] = own, query_routes(own, s.query_keys)
            ctxs = [self.clean(seq, j) for j in range(L)]
            for r in range(L - 1, -2, -1):  # root r: mid_r, from the top down; r = -1: the embedding with every root's
                k = P + L * k4 if r < 0 else P  # scalar probes (each from its mid stream: P rows per phase elsewhere)
                for t0 in range(0, T, m_chunk):
                    t1 = min(t0 + m_chunk, T)
                    sl = slice(seq * T + t0, seq * T + t1)
                    x = self.chunk(buf, ctxs, seq, t0, t1) | hx | dict(xL=xL[sl], base=base[sl], target_w=target_w[sl],
                                                                       y=y[sl])
                    delta, dx0 = buf['delta'][:(t1 - t0) * k * W].view(t1 - t0, k, W), None
                    if r < 0:
                        dx0 = F.rms_norm(emb[sl, None] + steps[0][sl, None] * bases[0], (W,)) - x0[sl, None]
                        torch.mul(dx0, lam_r[0] + lam_0[0], out=delta[:, :P])
                    else:
                        torch.mul(steps[r + 1][sl, None], bases[r + 1], out=delta)
                        self.mlp(x, delta, P, r)
                    for b in range(r + 1, L):
                        n_b = P + b * k4 if r < 0 else P
                        score[r + 1, sl] += self.score(x, delta, n_b, b, True)[:, :P]
                        self.push(x, delta, n_b, b)
                        if r < 0 and k4:  # the symmetric probes of block b's two mixing scalars start at mid_b
                            prev = (x0 if b == 0 else outs[b - 1])[sl]
                            delta[:, n_b:n_b + 4] = torch.stack([prev, x0[sl], -prev, -x0[sl]], 1) * h
                        self.mlp(x, delta, n_b + k4 * (r < 0), b, dx0, P)
                    change = head_fast(delta, x) * (y[sl] >= 0).unsqueeze(1)
                    credit[r + 1, sl] = (change[:, :P] / steps[r + 1][sl]) @ bases[r + 1] / W
                    if r < 0 and k4:  # the central difference over 2h per scalar
                        pair = change[:, P:].view(t1 - t0, L, 4) / h
                        lam += (pair[..., :2] - pair[..., 2:]).sum(0) / 2
                tok = e.group_sum(credit[r + 1, seq * T:(seq + 1) * T])  # the pair's halves: root r is complete
                if r >= 0:  # its hub's moment tables, read by every root below it
                    self.tables(ctxs[r], r, tok @ model.transformer.h[r].attn.c_proj.weight.float())
            self.contexts[seq] = ctxs
        X = e.group_sum(torch.stack([(score[k] / steps[k]) @ bases[k] / W for k in range(L + 1)]))
        self.X = {'emb': X[0], **dict(enumerate(X[1:]))}
        e.group_sum(lam)
        return {'emb': credit[0], **dict(enumerate(credit[1:])), **{('lam', r): lam[r] for r in range(L)}}

    def path_credits(self, chain, targets, raw_values):
        """The measured input credits and identity extra (Cut.path_credits), then the pass; extra_i of blocks 0-6
        becomes the readout credit R_i (Cut.estimate then forms x_mid = chain_i + R_i: attn.c_proj and the V target,
        and the mixing scalars' input-path credit inputs_i + R_i)."""
        est = self.e
        extra, inputs = Cut.path_credits(est, chain, targets, raw_values)
        self.chain, self.extra, self.inputs = chain, extra, inputs
        self.run(chain, targets)
        out = dict(extra)
        for i in BLOCKS:
            out[i] = self.X[i]
        for b in self.hi_blocks:
            out[b] = self.X[b]
        return out, inputs

    # ------------------------------------------------------------------------------------------------------- the pass

    def run(self, chain, targets):
        """The pass of every other row, on the chain's contexts and kernels: the q / c_fc sources and the c_k / ve_gate
        receivers of blocks 0-3 and of the hi blocks, and the in-roots. Rows per token in insertion order: per block b,
        q_b, b's receiver groups | b's MLP | f_b. Scored at every block they cross (receivers s > t; the c_fc rows also
        s = t at their first block, from their self-edge), decoded per chunk (sources) or per sequence (receivers);
        the in-roots are scored on their own path at their block."""
        e, s = self.e, self.e.s
        model, d, dev = e.model, e.capture, e.device
        L, W, H, T = model.config.n_layer, e.width, model.config.n_head, d.length
        lam_r, Rr = model.resid_lambdas.float(), s.readout_k_receivers
        hb, ins = tuple(s.readout_hi_blocks), tuple(sorted(s.readout_in_roots))
        self.hi_blocks, self.hi_in = hb, ins
        self.wq = {j: model.transformer.h[j].attn.c_q.weight.float() for j in ins}
        src_b = BLOCKS + hb
        cyc = {('q', b): W if b in BLOCKS and b in s.readout_q_full else H * s.readout_q_keys for b in src_b}
        cyc |= {('f', b): (s.readout_fc_cycle + s.readout_hi_fc_cycle)[b] for b in src_b}
        share = {key: e.share(c) for key, c in cyc.items()}
        # this GPU's receiver groups: those of blocks 0-3 and those of the hi blocks, each set split over the pair
        mine = receiver_groups(s.readout_k_blocks, model, e.split, e.part) + receiver_groups(hb, model, e.split, e.part)
        self.kv_C = group_rows(s, H)
        self.kv_wproj = {b: model.transformer.h[b].attn.c_proj.weight.float() for b, _ in mine}
        size = {key: b - a for key, (a, b) in share.items()} | {g: self.kv_C for g in mine}
        slots, at, in_rows, mid_rows = {}, 0, [], []  # rows per token, insertion order: q_b, groups of b | MLP b | f_b
        for b in range(L):
            in_rows.append(at)
            for key in [('q', b), *[g for g in mine if g[0] == b], 'mlp', ('f', b)]:
                if key == 'mlp':
                    mid_rows.append(at)
                elif key in size:
                    slots[key], at = slice(at, at + size[key]), at + size[key]
        rows, n = at, d.batch * T
        self.q_credit = {b: torch.zeros(n, W, device=dev) for k, b in cyc if k == 'q'}
        self.fc_credit = {b: torch.zeros(n, 4 * W, device=dev) for k, b in cyc if k == 'f'}
        self.k_credit = {b: torch.zeros(n, W, device=dev) for b, k in mine if k == 'k'}
        self.g_credit = {b: torch.zeros(n, H, device=dev) for b, k in mine if k == 'g'}
        Q, score = torch.zeros(len(ins), n, W, device=dev), torch.zeros(n, rows, device=dev)
        seed = int(torch.randint(2 ** 31, (), device=dev, generator=torch.Generator(dev).set_state(  # a private copy:
            (d.shared_generator if e.split > 1 else d.generator).get_state())).item())  # a pair's GPUs alike
        lo, hi = e.share(W)
        fc_in = {j: slots['f', j - 1] for j in ins if ('f', j - 1) in slots}  # c_fc rows also along the in-root's path
        rms = lambda z: s.readout_sigma * z.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        m_chunk = min(s.readout_chunk, T)
        buf = self.buffers(m_chunk, max(rows, 1))
        for seq in range(d.batch):
            ctxs, src, plans = self.contexts.pop(seq), {}, {}
            gens = [torch.Generator(dev).manual_seed(seed + off + 104729 * seq) for off in (0, HI_SEED)]
            for b in src_b:  # the draws in order (block b's q code, then its c_fc code; hi blocks: the 2nd generator)
                g, c = gens[b >= FIRST], ctxs[b]
                basis = self.q_basis(c, s.readout_q_keys) if b >= FIRST or cyc['q', b] < W else None
                src['q', b] = (self.draw_code(g, T, cyc['q', b], dev), basis, rms(c['qraw']))
                src['f', b] = (self.draw_code(g, T, cyc['f', b], dev),
                               rms(d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float())) + self.fc_slots(b, seq, cyc['f', b])
            for j in ins:
                src['in', j] = (self.draw_code(gens[1], T, W, dev), rms(ctxs[j]['x']))
            for b in sorted({g[0] for g in mine}):
                plans.update(self.plan(b, seq, ctxs[b], [k for bb, k in mine if bb == b], self.cs_all, self.sn_all))
            for t0 in range(0, T, m_chunk):
                t1 = min(t0 + m_chunk, T)
                m, sl, local = t1 - t0, slice(seq * T + t0, seq * T + t1), slice(t0, t1)
                x = self.chunk(buf, ctxs, seq, t0, t1)
                delta = buf['delta'][:m * rows * W].view(m, rows, W)
                dirs = {key: self.code_rows(src[key][0], cyc[key], *share[key], local, dev) for key in cyc}
                for key in [key for key in cyc if key[0] == 'q' and src[key][1] is not None]:  # 2 blockdiag(basis_h) code
                    dirs[key] = 2 * torch.einsum('thdk,tphk->tphd', src[key][1][local], dirs[key].view(
                        m, size[key], H, s.readout_q_keys)).reshape(m, size[key], W)
                for b in range(L):
                    c, k, fb = ctxs[b], in_rows[b], slots.get(('f', b - 1))
                    if k:
                        score[sl, :k] += self.score(x, delta, k, b, b < L - 1 or fb is not None)
                    if fb is not None:  # block b - 1's c_fc rows: the receiver s = t at their first block (self-edge)
                        se = x['se'][:m * rows * W].view(m, rows, W)[:, fb]
                        score[sl, fb] += (se.float() * c['target'][local, None]).sum(-1)
                    if k and b in fc_in:  # block b - 1's c_fc rows at x_b, before the push (the last block too)
                        score[sl, fc_in[b]] += self.oq_scores(c, b, delta[:, fc_in[b]], self.chain[b][sl], local)
                    if b == L - 1:
                        break
                    if k:
                        self.push(x, delta, k, b)
                    if ('q', b) in slots:
                        delta[:, slots['q', b]] = self.q_moves(c, b, dirs['q', b], src['q', b][2][local], local)
                    for g in mine:
                        if g[0] == b:
                            delta[:, slots[g]] = self.moves(g, plans[g], c, t0, t1)
                    if mid_rows[b]:
                        self.mlp(x, delta, mid_rows[b], b)
                    if ('f', b) in slots:
                        _, fstep, idx, hval, valid = src['f', b]
                        torch.mul(self.fc_moves(b, dirs['f', b], fstep[local], idx[local], hval[local], valid[local]),
                                  lam_r[b + 1], out=delta[:, slots['f', b]])
                for (kind, b), got in ((key, score[sl, slots[key]]) for key in cyc):  # the sources' decode
                    if kind == 'q':
                        self.q_credit[b][sl] = torch.einsum('tp,tpw->tw', got / src[kind, b][2][local], dirs[kind, b]) / W
                    else:
                        _, fstep, idx, hval, valid = src[kind, b]
                        slot = torch.einsum('tp,tpk->tk', got / fstep[local], dirs[kind, b][:, :, 1:]) / cyc[kind, b]
                        self.fc_credit[b][sl].scatter_add_(1, idx[local], slot * valid[local])
                for i, j in enumerate(ins):  # the in-roots: this GPU's rows of their cycles, decoded into its share
                    code, istep = src['in', j]
                    rows_in = self.code_rows(code, W, lo, hi, local, dev)
                    sc = self.oq_scores(ctxs[j], j, istep[local].unsqueeze(-1) * rows_in, self.chain[j][sl], local)
                    Q[i, sl] = torch.einsum('tp,tpw->tw', sc / istep[local], rows_in) / W
            out = slice(seq * T, (seq + 1) * T)
            for g in mine:  # the receivers' decode
                b, kind = g
                p = plans[g]
                f = p['table'].clamp_min(0)
                live = (p['table'] >= 0) & p['live'].reshape(-1)[f]
                reading = torch.zeros(H * T * Rr + 1, device=dev).scatter_(  # the last entry: a sink for empty slots
                    0, torch.where(live, f, H * T * Rr).reshape(-1), torch.where(live, score[out, slots[g]], 0.).reshape(-1))
                reading = reading[:-1].view(H, T, Rr)
                if kind == 'k':  # reading / step times the dual basis' reciprocal vectors
                    c_r = torch.where(p['live'], reading / p['x_lin'].clamp_min(1e-30), torch.zeros_like(reading))
                    self.k_credit[b][out] = torch.einsum('htr,htrd->thd', c_r, p['P']).reshape(T, W)
                else:
                    self.g_credit[b][out] = reading.sum(-1).T / GATE_SIGMA
            self.routes.pop(seq, None)
        self.hi_Q = dict(zip(ins, e.group_sum(Q))) if ins else {}

    # ------------------------------------------------------------------------------------------------------ pieces

    def prepare(self, mids, routes):
        """Per estimate: the bf16 weights and the scorer's projection d W_k^T | d W_v^T | d[:32] W_g^T (padded to 16),
        the routes in sort_routes' order for the pruned MLP."""
        model, W, T = self.e.model, self.e.width, self.e.capture.length
        H, blocks = model.config.n_head, model.transformer.h
        bt = lambda w: w.to(torch.bfloat16).T.contiguous()
        gate = lambda a: torch.zeros(W, 16, device=self.e.device, dtype=torch.bfloat16) if a.ve_gate is None else \
            F.pad(F.pad(a.ve_gate.weight.T, (0, 16 - H)).to(torch.bfloat16), (0, 0, 0, W - 32))
        w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in blocks]
        self.sp_w = dict(w_fc=w_fc, w_out_t=[bt(blk.mlp.c_proj.weight) for blk in blocks],
                         w_q_t=[bt(blk.attn.c_q.weight) for blk in blocks],
                         w_proj_t=[bt(blk.attn.c_proj.weight) for blk in blocks],
                         wcat=[torch.cat([bt(b.attn.c_k.weight), bt(b.attn.c_v.weight), gate(b.attn)], 1).contiguous()
                               for b in blocks],
                         lams=torch.stack([model.resid_lambdas.float(), model.x0_lambdas.float()], 1).contiguous(),
                         mids=[z.float().contiguous() for z in mids], units=max(r[0].shape[1] for r in routes))
        self.sorted_routes = sort_routes(w_fc, routes)
        self.cs_all, self.sn_all = model.cos[0, :T, 0].float().contiguous(), model.sin[0, :T, 0].float().contiguous()

    def clean(self, seq, j):
        """Block j's clean quantities of sequence seq that do not read the hub: its input and RMSNorm statistics, the
        rotated raw key, raw value, gate logits and value embedding, and the clean attention state (raw query, rotated
        normed queries / keys, values, attention, outputs; the diagonals for s = t)."""
        d, model = self.e.capture, self.e.model
        T, H = d.length, model.config.n_head
        cos, sin = model.cos[:, :T].float(), model.sin[:, :T].float()
        attn, prefix = model.transformer.h[j].attn, f'transformer.h.{j}.attn.'
        qraw, kraw, vraw = [d.outputs[prefix + n][seq:seq + 1].float() for n in ('c_q', 'c_k', 'c_v')]
        v, gate, ve = vraw.reshape(1, T, H, -1), None, None
        if attn.ve_gate is not None:
            gate = d.outputs[prefix + 've_gate'][seq].float()
            ve = d.outputs[f'value_embeds.{j}'][seq].float().reshape(T, H, -1)
            v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
        values = v.transpose(1, 2)[0]
        q, k, a, out = (z[0] for z in attention_state(qraw, kraw, values[None], cos, sin, H, model.window_sizes[j][0]))
        kr, x = kraw[0].reshape(T, H, -1), d.blocks[j][seq].float().contiguous()
        return dict(x=x, sqx=x.square().sum(-1).contiguous(), kraw=kr.contiguous(), gate=gate,
                    R0=rotate(kr, cos[0, :, 0][:, None], sin[0, :, 0][:, None]).contiguous(), VC=vraw[0].contiguous(),
                    GZ=None if gate is None else gate.contiguous(), ve=None if ve is None else ve.contiguous(),
                    QD=q.transpose(0, 1).contiguous(), AD=a.diagonal(0, 1, 2).T.contiguous(),
                    VT=values.transpose(0, 1).contiguous(), OT=out.transpose(0, 1).contiguous(),
                    qraw=qraw[0], q=q, k=k, values=values, a=a, out=out)

    def tables(self, c, j, target):
        """Block j's hub part: the moment tables of the receivers s > t from target = chain_j W_proj (kept too: the c_fc
        rows' s = t)."""
        T, H = self.e.capture.length, self.e.model.config.n_head
        g = target.reshape(T, H, -1).transpose(0, 1)
        contrast = (g @ c['values'].transpose(-1, -2)) - (g * c['out']).sum(-1, keepdim=True)
        c['U'], c['W'], c['M1'], c['M2'] = moment_tables(c['q'], c['a'], contrast, self.e.model.window_sizes[j][0], g)
        c['target'] = target

    def buffers(self, m, rows):
        W, dev = self.e.width, self.e.device
        bf = lambda k: torch.empty(m * rows * k, device=dev, dtype=torch.bfloat16)
        return dict(delta=torch.empty(m * rows * W, device=dev), scratch32=torch.empty(m * rows * W, device=dev),
                    scratch=bf(max(W, self.sp_w['units'])), db=bf(W), se=bf(W), up=bf(2 * W + 16),
                    st=torch.empty(m * rows * 2, device=dev), **prune_buffers(m, rows, dev))

    def chunk(self, buf, ctxs, seq, t0, t1):
        """The block-step context for tokens t0:t1 of sequence seq (every chunk of the chain and of the pass starts
        here; no live-row lists yet)."""
        self.sk_live, self.sk_lists, self.sk_new = None, None, []
        sl, w = slice(seq * self.e.capture.length + t0, seq * self.e.capture.length + t1), self.sp_w
        own, routes = self.routes[seq]
        x = dict(buf, ybuf=buf['scratch'], ctxs=ctxs, own=own, routes=routes, local=slice(t0, t1), t0=t0,
                 cos=own[6].contiguous(), sin=own[7].contiguous(), mids=[z[sl] for z in w['mids']],
                 **{k: w[k] for k in ('w_q_t', 'w_proj_t', 'w_fc', 'w_out_t', 'lams')})
        return x | sorted_slices(self.sorted_routes, sl)

    def score(self, x, delta, n, b, self_edge):
        """The staged scorer at block b on rows < n: the receivers s > t only -> [m, n]; self_edge: also each row's
        own output change (bf16, x['se']: push() adds it, run() scores the c_fc rows' s = t from it). With live-row
        lists, the scorer walks each token's live rows only."""
        m, rows, W = delta.shape
        c, H, nc = x['ctxs'][b], self.e.model.config.n_head, self.sp_w['wcat'][b].shape[1]
        out = torch.empty(m, n, device=delta.device)
        gated = c['gate'] is not None
        stats_kernel[(triton.cdiv(n, 16), m)](delta, c['x'], c['sqx'], x['db'], x['st'], x['t0'], n, rows, W, 16, EPS,
                                              num_warps=4)
        up = x['up'][:m * n * nc].view(m * n, nc)
        torch.mm(x['db'][:m * n * W].view(m * n, W), self.sp_w['wcat'][b], out=up)
        args = (up, x['st'], c['R0'], c['VC'], c['GZ'] if gated else out, c['ve'] if gated else out, self.cs_all,
                self.sn_all, c['U'], c['W'], c['M1'], c['M2'], c['QD'], out, c['AD'], out, c['VT'], c['OT'], out, x['se'],
                x['t0'], n, rows, W, H, W // H, gated, False, self_edge, 64, nc)
        if self.sk_lists is None:
            staged_kernel[(triton.cdiv(n, 64), m)](*args, num_warps=4, num_stages=2)
        else:
            rows_l, cnt = self.sk_lists
            staged_kernel_live[(triton.cdiv(n, 64), m)](rows_l, cnt, cnt.shape[1], *args, num_warps=4, num_stages=2)
        return out

    def push(self, x, delta, n, b):
        """Rows < n through block b's attention: the own query and the self-edge just scored, both through c_proj."""
        own_query_step(delta, n, b, x, self.sk_lists)

    def mlp(self, x, delta, n, b, dx0=None, P=0):
        """Block b's routed MLP on rows < n, then block b + 1's mixing scalars (dx0: the embedding rows' x0 change on
        the first P rows); none after the last block. Block b's receiver groups, formed just now, end at n: their live
        slots join the tokens' live-row lists first."""
        if self.sk_new:
            self.lists(delta, n)
        last = b == self.e.model.config.n_layer - 1
        mlp_step(delta, n, b, x, None if last else delta if dx0 is None else dx0, 0 if dx0 is None else P)

    def lists(self, delta, n):
        """Each token's live rows in order, then its dead ones (LROW), and its live rows below each row (LCNT); both
        [m, rows + 1] (one stride). No host synchronization."""
        m, rows = delta.shape[:2]
        if self.sk_live is None:
            self.sk_live = torch.ones(m, rows, dtype=torch.bool, device=delta.device)
        new = torch.cat(self.sk_new, 1)
        self.sk_live[:, n - new.shape[1]:n], self.sk_new = new, []
        cnt = F.pad(self.sk_live.cumsum(1, dtype=torch.int32), (1, 0))
        below, p = cnt[:, :-1], torch.arange(rows, device=delta.device, dtype=torch.int32)
        at = torch.where(self.sk_live, below, cnt[:, -1:] + p - below)  # a permutation: live rows first, in order
        self.sk_lists = (torch.zeros_like(cnt).scatter_(1, at.long(), p.expand(m, rows)), cnt)

    def q_basis(self, ctx, keys):
        """Per (token, head) an orthonormal basis [T, H, D, keys] of the raw-query subspace that moves the logits of the
        head's `keys` strongest clean keys: R_t^T (I - r r^T) k_s (rotation back, the query norm's projection)."""
        e = self.e
        T, H = e.capture.length, e.model.config.n_head
        D = e.width // H
        cs, sn = e.model.cos[0, :T, 0].float(), e.model.sin[0, :T, 0].float()
        top = ctx['a'].topk(keys, dim=-1).indices
        kk = ctx['k'][torch.arange(H, device=top.device)[:, None, None], top]
        r = rotate(ctx['qraw'].reshape(T, H, D), cs[:, None], sn[:, None]).transpose(0, 1)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        proj = kk - (kk * r[:, :, None]).sum(-1, keepdim=True) * r[:, :, None]
        back = rotate(proj, cs[None, :, None], -sn[None, :, None])
        return orthonormal(back.permute(1, 0, 2, 3).reshape(T * H, keys, D)).reshape(T, H, keys, D).transpose(2, 3)

    def fc_slots(self, b, seq, cycle):
        """The c_fc source's units per token: the cycle - 1 strongest active units of block b (relu^2(h) |W_out col|, the
        record's route ranking); slot k + 1 of the cycle is unit idx[:, k] (slot 0, the all-ones Hadamard column, unused);
        valid: the unit is active."""
        d, model = self.e.capture, self.e.model
        h = d.outputs[f'transformer.h.{b}.mlp.c_fc'][seq].float()
        w_out = model.transformer.h[b].mlp.c_proj.weight.float()
        rank, idx = (F.relu(h).square() * w_out.norm(dim=0)).topk(cycle - 1, dim=-1)
        return idx, h.gather(1, idx), rank > 0

    @staticmethod
    def draw_code(gen, T, n, device):
        """A random-sign complete Hadamard cycle of size n per token: (row permutation [n], column signs [T, n])."""
        perm = torch.randperm(n, device=device, generator=gen)
        signs = 2 * torch.randint(0, 2, (T, n), device=device, generator=gen).float() - 1
        return perm, signs

    @staticmethod
    def code_rows(code, n, lo, hi, local, device):
        """This GPU's rows lo:hi of the cycle for the tokens `local`: [m, hi - lo, n]."""
        perm, signs = code
        return hadamard(n, device, torch.float32)[perm[lo:hi]][None] * signs[local][:, None]

    def oq_scores(self, c, j, moves, credit, local):
        """Scores of block-j input moves [m, p, W] through the token's own query only: <W_proj (Attn(q + dq) - Attn(q)),
        chain_j> with dq = (norm(x + move) - norm(x)) W_q^T, the exact softmax over every key (every key / value
        clean), credit = the measured mid_j credit of these tokens [m, W], in contrast form (inroot_kernel). [m, p]"""
        x = c['x'][local]
        dq = torch.matmul(norm_delta(x[:, None], moves), self.wq[j].T)
        wproj = self.e.model.transformer.h[j].attn.c_proj.weight
        target = (credit.double() @ wproj.double()).float()
        return inroot_scores(dq.contiguous(), c['qraw'], c['k'], c['values'], c['a'][:, local], c['out'][:, local],
                             target.contiguous(), local.start, self.cs_all, self.sn_all,
                             self.e.model.window_sizes[j][0])

    def q_moves(self, c, b, dirs, qstep, local):
        """W_proj (Attn(q + step d) - Attn(q)) of the token's own query at block b (exact over all keys; every key /
        value clean), [m, P_q, W] (qsrc2_kernel)."""
        out = qsrc2_response(dirs.contiguous(), c['qraw'].contiguous(), qstep.contiguous(), c['k'], c['values'],
                             c['a'][:, local], c['out'][:, local], local.start, self.cs_all, self.sn_all,
                             self.e.model.window_sizes[b][0])
        return F.linear(out.flatten(-2), self.e.model.transformer.h[b].attn.c_proj.weight.float())

    def fc_moves(self, b, code, fstep, idx, hval, valid):
        """The block-b output change W_out (relu^2(h + step d) - relu^2(h)) on the token's slot units, [m, P_f, W]
        (fc_relu_kernel, then the gathered GEMM)."""
        w_sel = self.e.model.transformer.h[b].mlp.c_proj.weight.float().T[idx]  # [m, slots, W]
        m, P, cyc = code.shape
        ns = idx.shape[1]
        r = torch.empty(m, P, ns, device=code.device)
        fc_relu_kernel[(triton.cdiv(P, 32), triton.cdiv(ns, 128), m)](
            code.contiguous(), fstep.contiguous(), hval.contiguous(), valid.contiguous(), r, P, ns, cyc, 32, 128,
            num_warps=4, enable_fp_fusion=False)
        return torch.bmm(r, w_sel)

    def plan(self, b, seq, c, kinds, cs, sn):
        """Block b's receiver experiments: per (head h, key t) the top-R receivers, the receiver slot table, and per pair
        the coefficient of its receiver's output change (k: plan_kernel's exact single-column gain along v_t - o_s;
        gate: A_st (2 sig(z + step) - 2 sig(z)) along ve_t)."""
        e, s = self.e, self.e.s
        H, T = e.model.config.n_head, e.capture.length
        D, Rr, C = e.width // H, s.readout_k_receivers, self.kv_C
        A = c['a']
        top = A.transpose(1, 2).topk(Rr, dim=-1)  # per (h, key t): its strongest receivers s, in attention order
        vals, idx = top.values.contiguous(), top.indices.contiguous()  # [H, T, R]
        ok = vals > 0
        table, kept = receiver_slots(idx, vals, ok, C, T)
        P = torch.empty(H, T, Rr, D, device=A.device)
        live, coef, x_lin = torch.empty_like(ok), torch.empty_like(vals), torch.empty_like(vals)
        plan_kernel[(triton.cdiv(T, PLAN_BT), H)](idx, vals, kept, c['QD'], c['R0'], c['kraw'], cs, sn, P, live, coef,
                                                  x_lin, T, H, D, Rr, PLAN_BT, K_TAU, K_SIGMA, EPS, math.sqrt(D),
                                                  num_warps=PLAN_WARPS)
        plans = {(b, 'k'): dict(table=table, live=live, coef=coef, P=P, x_lin=x_lin)}
        if 'g' in kinds:
            z = c['gate']
            dvs = 2 * (torch.sigmoid(z + GATE_SIGMA) - torch.sigmoid(z))
            coef = torch.where(kept, vals * dvs.T[:, :, None], torch.zeros_like(vals))
            plans[b, 'g'] = dict(table=table, live=kept, coef=coef)
        return plans

    def moves(self, g, p, c, t0, t1):
        """The mid_b moves of group g's rows for the receiver tokens t0:t1, [m, C, W] (zero in empty slots): the receiver's
        output change of head h through c_proj. The slots' live mask is kept for the live-row lists."""
        b, kind = g
        H = self.e.model.config.n_head
        T, W = self.e.capture.length, self.e.width
        D, Rr = W // H, self.e.s.readout_k_receivers
        tbl = p['table'][t0:t1]
        m, C = tbl.shape
        f = tbl.clamp_min(0)
        live = (tbl >= 0) & p['live'].reshape(-1)[f]
        self.sk_new.append(live)
        coef = torch.where(live, p['coef'].reshape(-1)[f], torch.zeros((), device=f.device))
        h, t = f // (T * Rr), (f // Rr) % T
        if kind == 'k':  # receiver s's output moves along v_t - o_s
            s_ = torch.arange(t0, t1, device=f.device)[:, None].expand(m, C)
            vec = c['VT'][t, h] - c['OT'][s_, h]
        else:  # along the value change: the value embedding ve_t
            vec = c['ve'][t, h]
        head = torch.zeros(m, C, H, D, device=f.device)
        head.scatter_(2, h[:, :, None, None].expand(m, C, 1, D), (coef[..., None] * vec)[:, :, None])
        return F.linear(head.view(m, C, W), self.kv_wproj[b])

    # ------------------------------------------------------------------------------------------------------ finishing

    def finish(self, gradients):
        """mlp.c_proj of blocks 0-6, wte and the in-roots' mixing scalars (a pair's first GPU: the record's rule, the
        other keeps zeros), then the sources' c_q / c_fc and the receivers' c_k / ve_gate credit (every GPU, its share)."""
        e, d = self.e, self.e.capture
        lam_r = e.model.resid_lambdas.float()
        torch.randint(2 ** 31, (), device=e.device,  # the one draw of #31's future probes (gone): the later draws
                      generator=d.shared_generator if e.split > 1 else d.generator)  # stay those of the runs before
        if not e.part:  # formed in full by a pair's first GPU (the record's rule; the other keeps zeros)
            emb, x0, mids, outs, routes, y = self.chain_args
            for j, Q in self.hi_Q.items():  # an in-root's query path at x_j: the mixing scalars of block j (linear)
                gradients['resid_lambdas'][j] += (Q * (x0 if j == 0 else outs[j - 1])).sum()
                gradients['x0_lambdas'][j] += (Q * x0).sum()
            roots = set(BLOCKS + self.hi_blocks)  # mid_i's readout root: blocks 0 .. n_layer - 2
            for i in sorted(roots):  # out_i = lam_{i+1} (chain_{i+1} + inputs_{i+1} + R_{i+1} [+ Q_{i+1}]), R_7 = 0
                p = f'transformer.h.{i}.'
                cross = lam_r[i + 1] * (self.inputs[i + 1] + self.X[i + 1]) if i + 1 in roots else self.extra[i]
                x_out = lam_r[i + 1] * self.chain[i + 1] + cross
                if i + 1 in self.hi_Q:
                    x_out = x_out + lam_r[i + 1] * self.hi_Q[i + 1]
                gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            gradients['transformer.wte.weight'] += d.reassemble(
                'transformer.wte', self.X['emb'].view(d.batch, d.length, e.width))
        for i, credit in self.q_credit.items():  # every GPU: its share of the cycles (the pair's sum is the decode)
            p = f'transformer.h.{i}.attn.c_q'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))
        for i, credit in self.fc_credit.items():
            p = f'transformer.h.{i}.mlp.c_fc'
            gradients[p + '.weight'].addmm_(credit.T, d.inputs[p].flatten(0, 1))
        for credits, site in ((self.k_credit, 'attn.c_k'), (self.g_credit, 'attn.ve_gate')):  # every GPU: its groups
            for b, credit in credits.items():
                p = f'transformer.h.{b}.{site}'
                gradients[p + '.weight'].add_(credit.T @ d.inputs[p].flatten(0, 1))


# ------------------------------------------------------------------------------------------------------ optimizer


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


RUN = (1194, 240, 50, 514)


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360: its schedule (BatchedAnvil steps it)."""

    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t = 0

    def rail_beta(self):
        """Scheduled fast-rail and Nesterov beta."""
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi


def newton_schulz(x):
    """The record's six quintic maps on a stack (B, n, n) of square matrices, in bfloat16."""
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    """The same maps on wide matrices (n < m), iterated on the n x n Gram matrix (Gram Newton-Schulz, Dao lab 2026).
    R is rebuilt from Q X before the maps in `restarts` to bound the float16 error."""
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


def polar_batch(g, margin=1.1):
    """Approximate orthogonalization of a stack (B, r, c) of same-shape matrices."""
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
                 ratio: float, margin: float):
    """One ANVIL II step for a stack of same-shape matrices; updates fast, slow and energy in place."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class BatchedAnvil(Anvil):
    """ANVIL II from modded-nanogpt PR #360, applied to all same-shape matrices at once."""

    def __init__(self, *args, polar_margin=1.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.polar_margin, self.stacks = polar_margin, {}

    def load_state_dict(self, state_dict):
        # Cached stacks must be rebuilt from the freshly restored per-parameter state.
        self.stacks.clear()
        return super().load_state_dict(state_dict)

    def stacked_state(self, shape, params):
        """The moving averages of one shape group as stacked tensors. Each parameter's state holds views into them,
        so the Boost -> Cut handoff, which copies per-parameter state, carries them over."""
        key = tuple(params)
        stack = self.stacks.get(key)
        if stack is not None and all(self.state[p].get('fast') is not None for p in params):
            return stack
        lanes = 1 if shape[0] >= shape[1] else 0
        weights = torch.stack([p.float() for p in params])

        def gathered(key, like):
            held = [self.state[p].get(key) for p in params]
            return torch.stack(held).clone() if all(h is not None for h in held) else torch.zeros_like(like)

        stack = dict(fast=gathered('fast', weights), slow=gathered('slow', weights),
                     energy=gathered('energy', weights.mean(lanes + 1, keepdim=True)))
        for i, p in enumerate(params):
            self.state[p].update(fast=stack['fast'][i], slow=stack['slow'][i], energy=stack['energy'][i])
        self.stacks[key] = stack
        return stack

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        device = self.param_groups[0]['params'][0].device
        value = lambda v: torch.tensor(float(v), device=device)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            by_shape = {}
            for p in group['params']:
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, params in by_shape.items():
                stack = self.stacked_state(shape, params)
                new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                                   stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                                   value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                                   1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5, self.polar_margin)
                torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        # Identical all-reduced gradients must produce identical rank replicas.
        # Scope this to updates so embedding scatter and probes keep their fast paths.
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for o in self.optimizers:
                o.step()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_model(device, seed, config, projection_std, head_std=0.001):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # Output projections: uniform with the declared std / sqrt(fan_in).
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = projection_std * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        if head_std != 0.001:
            nn.init.normal_(model.lm_head.weight, std=head_std)
    return model.requires_grad_(False)


def make_optimizer(model, recipe, steps):
    """ANVIL II on block matrices, Adam elsewhere."""
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
        betas=recipe.betas, eps=1e-8, weight_decay=0.0,
    )
    anvil = BatchedAnvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps,
                         polar_margin=recipe.polar_margin, fast_weight=recipe.fast_weight)
    return Optimizers(anvil, adam)


def set_rate(optimizer, step, steps, warmdown, final=0.0):
    """Set the original val linear warmdown rate, with one-based training steps (down to `final` of the rate)."""
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


# ------------------------------------------------------------------------------------------------------- training


def save_training_state(path, model, optimizer, estimator, *, step, train_seconds, recipe, progress=None):
    """Atomically save common training state and each rank's independent RNGs."""
    device = estimator.device
    random = dict(generator=estimator.generator.get_state(),
                  shared_generator=estimator.shared_generator.get_state(),
                  cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    if isinstance(estimator, Boost):
        random['capture_generator'] = estimator.capture.generator.get_state()
    states = [None] * estimator.world
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(states, random)
    else:
        states[0] = random
    if estimator.rank == 0:
        memory = getattr(estimator, 'memory', None)
        checkpoint = dict(step=step, train_seconds=train_seconds, recipe=recipe, world=estimator.world,
            schedule_steps=optimizer.optimizers[0].total,
            model=model.state_dict(), rng=states,
            estimator=type(estimator).__name__, progress=progress,
            optimizers=[dict(state=o.state_dict(), t=getattr(o, 't', None), kind=type(o).__name__)
                        for o in optimizer.optimizers],
            memory=None if memory is None else {name:getattr(memory, name) for name in
                    ('sites', 'beta', 'ridge', 'alpha_beta', 'key_moment', 'cross_moments', 'alpha', 'calibration')})
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def restore_training_state(path, model, optimizer, estimator, *, recipe):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    canonical = lambda x: json.loads(json.dumps(x))
    if canonical(checkpoint['recipe']) != canonical(recipe) or checkpoint['world'] != estimator.world:
        raise ValueError('Resume requires the same recipe and number of ranks')
    if checkpoint['estimator'] != type(estimator).__name__:
        raise ValueError('Resume requires the same estimator phase')
    if checkpoint['schedule_steps'] != optimizer.optimizers[0].total or checkpoint['memory'] is not None:
        raise ValueError('Resume requires the same schedule and no auxiliary memory')
    model.load_state_dict(checkpoint['model'], strict=True)
    for active, saved in zip(optimizer.optimizers, checkpoint['optimizers'], strict=True):
        if saved['kind'] != type(active).__name__:
            raise ValueError('Resume requires the same optimizer implementation')
        active.load_state_dict(saved['state'])
        if saved['t'] is not None:
            active.t = saved['t']
    optimizer.param_groups = [g for active in optimizer.optimizers for g in active.param_groups]
    random = checkpoint['rng'][estimator.rank]
    estimator.generator.set_state(random['generator'])
    estimator.shared_generator.set_state(random['shared_generator'])
    estimator.capture.generator.set_state(random['capture_generator'])
    torch.set_rng_state(random['cpu'])
    if random['cuda'] is not None:
        torch.cuda.set_rng_state(random['cuda'], estimator.device)
    return checkpoint['step'], checkpoint['train_seconds']


def stage_schedule(rows, settings):
    """Disjoint data slices, monotonically numbered updates, and phase-specific scheduler clocks."""
    b1, b2 = settings.boost_batch, settings.cut_batch
    if rows % b2 or b2 % b1:
        raise ValueError('Data rows must divide into complete Cut batches, and Cut batch must divide by Boost batch')
    n1, n2 = rows // b1, rows // b2
    alignment = b2 // b1
    boost = n1 if settings.estimator == 'boost' else round(settings.switch * n1) // alignment * alignment
    batches = [('boost', i, (i-1)*b1, i*b1, n1) for i in range(1, boost+1)]
    if settings.estimator != 'boost':
        batches += [(settings.estimator, i, (i-1)*b2, i*b2, n2) for i in range(boost*b1//b2+1, n2+1)]
    return batches


def handoff_optimizer(old, new, phase_step):
    """Keep ANVIL rails and Adam moments; change only the new phase's ANVIL scheduler clock."""
    for new_o, old_o in zip(new.optimizers, old.optimizers, strict=True):
        new_o.state.update(old_o.state)
    new.optimizers[0].t = phase_step - 1


def holdout_loss(model, rows, rank, world):
    """Mean next-token loss over the held-out training sequences, split across ranks."""
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in rows[rank::world].split(8):
            batch = batch.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
            total[0] += loss.double()
            total[1] += batch[:, 1:].numel()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return (total[0] / total[1]).item()


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])  # one matrix of block i (attn.c_q, ..., mlp.c_proj)
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 32  # training sequences that feed no update and score the end merge
MERGE_FROM, MERGE_EVERY = 0.6, 5  # the exponential means follow the weights from 60% of the epoch, every 5 updates
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)  # weight of the newest weights in each mean
MERGE_RHOS = (0.0, 0.5, 1.0)  # how far a candidate moves outside the high-curvature subspace
MERGE_K, MERGE_TOKENS = 64, 512  # input-covariance directions per linear layer; frequent-token rows per table


@torch.no_grad()
def curvature_bases(model, rows, rank, world, k=MERGE_K):
    """Per linear layer, the top-k eigenvectors of its input covariance E[x x^T], and the token order of
    `rows` by frequency (for the embedding tables), from forward passes of the current weights on `rows` (the
    held-out training sequences; no gradient). Computed on rank 0 and broadcast, so every rank merges identically."""
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
              if isinstance(mod, nn.Linear) and 've_gate' not in name]
    bases = {key: torch.zeros(mod.weight.shape[1], k, device=device) for key, mod in layers}
    order = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    if rank == 0:
        acc, hooks = {}, []
        for key, mod in layers:
            def hook(module, inputs, output, key=key):
                x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
                acc[key] = acc.get(key, 0) + (x.T @ x).double()
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
            bases[key] = torch.linalg.eigh(acc[key])[1].flip(1)[:, :k].float().contiguous()
        order = torch.argsort(counts, descending=True)
    if world > 1:
        for key, _ in layers:
            torch.distributed.broadcast(bases[key], 0)
        torch.distributed.broadcast(order, 0)
    return bases, order


def subspace_move(name, move, bases, order):
    """The part of a weight move inside its high-curvature subspace: the top input-covariance directions of a
    linear layer, the rows of the most frequent tokens of an embedding table; other tensors whole."""
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


MERGE_COEFS = (0.75, 1.0, 1.25)  # multiples of each move in the joint search


@torch.no_grad()
def joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load,
                sweeps=2, shortlist=8):
    """The joint search over keys (rho, j, c) = final + c * move(rho, j) per part (None = the final weights):
    the best single key for every part, then coordinate sweeps (all keys per part, then each part's best `shortlist`)."""
    parts = sorted({merge_part(n) for n, _ in named})
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in MERGE_COEFS]
    cache = {}

    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]

    def state(choice):
        return [value(i, choice[merge_part(n)]) for i, (n, _) in enumerate(named)]
    best_key, best = None, plain
    for key in keys[1:]:
        load([value(i, key) for i in range(len(params))])
        loss = holdout_loss(model, rows, rank, world)
        if loss < best:
            best_key, best = key, loss
    choice = {p: best_key for p in parts}
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            tried = []
            for key in (keys if sweep == 0 else ranked[p]):
                if key == choice[p]:
                    tried.append((best, key))
                    continue
                trial = dict(choice)
                trial[p] = key
                load(state(trial))
                loss = holdout_loss(model, rows, rank, world)
                tried.append((loss, key))
                if loss < best:
                    best, choice = loss, trial
            if sweep == 0:
                ranked[p] = [k for _, k in sorted(tried, key=lambda x: x[0])[:shortlist]]
    load(state(choice))
    return dict(holdout_plain=plain, holdout_merged=best,
                merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]]) for p, k in choice.items()})


@torch.no_grad()
def end_merge(model, means, rows, rank, world):
    """The end merge (module docstring): per part, candidates final + c * (inside + rho * (full - inside)), full = an
    exponential mean minus the final weights and inside its part in the high-curvature subspace (curvature_bases on the
    held-out rows), chosen on the held-out loss by joint_parts. Every rank takes the same decisions (the losses are
    all-reduced)."""
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows, rank, world)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]

    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)

    def load(ws):
        for q, w in zip(params, ws):
            q.copy_(w.to(q.dtype))
    plain = holdout_loss(model, rows, rank, world)
    return joint_parts(model, named, params, final, move, candidates, rows, rank, world, plain, load)


def train_stages(args, settings, device, rank, world):
    """Boost -> own-query Cut; each row is consumed exactly once, including across resume."""
    if world not in (1, 2, 4, 8):
        raise ValueError('Stage training supports 1, 2, 4 or 8 GPUs')
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    holdout = 4 if args.smoke else HOLDOUT
    holdout_rows, training = training[-holdout:], training[:-holdout]  # these feed no update, only the end merge
    if args.smoke:  # checks only: a few updates of each phase, the means and the end merge
        training = training[:44]
    means = None  # the exponential means of the end merge, one per MERGE_DECAYS entry
    schedule = stage_schedule(len(training), settings)
    steps = min(args.steps or len(schedule), len(schedule))
    stop = min(steps, args.stop_after_step or steps)
    split = max(1, world // settings.boost_batch)
    groups = ([torch.distributed.new_group(list(range(i, i+split))) for i in range(0, world, split)]
              if split > 1 else [None])
    group = groups[rank // split] if split > 1 else None
    cut_split = max(1, world // settings.cut_batch)  # GPUs per sequence in Cut
    cut_groups = groups if cut_split == split else (
        [torch.distributed.new_group(list(range(i, i+cut_split))) for i in range(0, world, cut_split)]
        if cut_split > 1 else [None])
    cut_group = cut_groups[rank // cut_split] if cut_split > 1 else None
    recipe_state = asdict(settings)
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, seed=SEED, recipe=recipe_state, steps=steps,
            epoch_updates=len(schedule), train_rows=len(training),
            upstream='qlabs-eng/noise@3343973b6f1af119effcdea2622f1fe672c51b1e',
            additions=['linear_path_scalars', 'input_key_probes', 'input_gate_probes', 'cut_batch4_split2', 'recipe_11',
                       'end_merge', 'true_path_readout', 'self_edged_chain', 'q_fc_source_experiments',
                       'k_gate_receiver_experiments', 'input_roots'],
            resume=str(args.resume.resolve()) if args.resume else None, torch_version=torch.__version__,
        ), indent=2)+'\n')

    def estimator_for(model, phase):
        if phase == 'boost':
            return Boost(model, settings, SEED, rank, world, split, group)
        return Cut(model, settings, SEED, rank, world, cut_split, cut_group)

    def rank_slice(batch, phase):
        group_size = split if phase == 'boost' else cut_split
        per = len(batch) // (world // group_size)
        first = (rank // group_size) * per
        return batch[first:first+per].to(device)

    # Compile every active phase on random tokens; recreate all model/optimizer/RNG state afterward.
    phases = {phase: horizon for phase, _, _, _, horizon in schedule}
    noise = torch.randint(0, settings.model.vocab_size,
                          (max(settings.boost_batch, settings.cut_batch), settings.model.sequence_len+1),
                          generator=torch.Generator().manual_seed(SEED))
    for phase, horizon in phases.items():
        model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
        optimizer, estimator = make_optimizer(model, settings, horizon), estimator_for(model, phase)
        size = settings.boost_batch if phase == 'boost' else settings.cut_batch
        batch = rank_slice(noise[:size], phase)
        estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        estimator.close()
        del model, optimizer, estimator
    torch.cuda.empty_cache()

    completed, previous_seconds = 0, 0.
    phase = schedule[0][0]
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        completed = checkpoint['step']
        if not 0 < completed < stop:
            raise ValueError('Resume checkpoint must precede the requested stop and follow at least one update')
        saved_phase, phase_step, _, seqs, _ = schedule[completed-1]
        expected = dict(phase=saved_phase, phase_step=phase_step, seqs=seqs, epoch_updates=len(schedule))
        if checkpoint.get('progress') != expected:
            raise ValueError('Resume data position or phase schedule differs')
        phase = saved_phase
        del checkpoint
    model = make_model(device, SEED, settings.model, settings.projection_std, settings.head_std)
    optimizer, estimator = make_optimizer(model, settings, phases[phase]), estimator_for(model, phase)
    if args.resume:
        completed, previous_seconds = restore_training_state(args.resume, model, optimizer, estimator, recipe=recipe_state)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, paused, elapsed = time.perf_counter(), 0., previous_seconds
    try:
        for step in range(completed+1, stop+1):
            current, phase_step, begin, end, horizon = schedule[step-1]
            if current != phase:
                estimator.close()
                next_optimizer = make_optimizer(model, settings, horizon)
                handoff_optimizer(optimizer, next_optimizer, phase_step)
                optimizer, estimator, phase = next_optimizer, estimator_for(model, current), current
            batch = rank_slice(training[begin:end], phase)
            set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.perf_counter()-start-paused
            record = dict(step=step, phase=phase, phase_step=phase_step, seqs=end,
                          tokens=end*settings.model.sequence_len, train_loss=loss, train_seconds=round(elapsed, 2),
                          peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3))
            if args.eval_every and step % args.eval_every == 0 and step < steps:
                pause = time.perf_counter()
                record['val_loss'] = evaluate(model.state_dict(), args.data_dir)
                paused += time.perf_counter()-pause
            if rank == 0:
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record)+'\n'); log.flush()
            if (step % MERGE_EVERY == 0 or step == stop) and begin / len(training) >= MERGE_FROM:
                with torch.no_grad():  # a resumed run restarts the means at its resumed weights
                    if means is None:
                        means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                    else:
                        for a, mean in zip(MERGE_DECAYS, means):
                            for m, q in zip(mean, model.parameters()):
                                m.lerp_(q.detach().float(), a)
            if (args.checkpoint_every and (step % args.checkpoint_every == 0 or step == stop)) or step == stop < steps:
                pause = time.perf_counter()
                save_training_state(args.output/'checkpoint.pt', model, optimizer, estimator, step=step,
                                    train_seconds=elapsed, recipe=recipe_state,
                                    progress=dict(phase=phase, phase_step=phase_step, seqs=end, epoch_updates=len(schedule)))
                paused += time.perf_counter()-pause
        if stop < steps:
            if rank == 0:
                (args.output/'segment.json').write_text(json.dumps(dict(step=stop, total_steps=steps, seqs=end,
                    train_seconds=elapsed, checkpoint=str((args.output/'checkpoint.pt').resolve())), indent=2)+'\n')
            return
        extra = {}
        if means:  # the end merge, timed with the training (the plain weights' validation is not)
            plain = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            pause = time.perf_counter()
            extra['val_loss_plain'] = evaluate(plain, args.data_dir)
            paused += time.perf_counter()-pause
            merge_start = time.perf_counter()
            extra.update(end_merge(model, means, holdout_rows, rank, world))
            torch.cuda.synchronize()
            extra['merge_seconds'] = round(time.perf_counter()-merge_start, 2)
            elapsed = previous_seconds + time.perf_counter()-start-paused
            if rank == 0:
                torch.save(plain, args.output/'final_plain.pt')
            means = None
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        result = dict(val_loss=evaluate(state, args.data_dir), train_seconds=round(elapsed, 2), steps=stop,
                      seqs=end, tokens=end*settings.model.sequence_len, estimator=settings.estimator,
                      peak_cuda_gb=round(torch.cuda.max_memory_allocated()/1e9, 3), **extra)
        if rank == 0:
            torch.save(state, args.output/'final.pt')
            (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    finally:
        estimator.close()
        if log is not None:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/val'))
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=406)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--steps', type=int, help='Diagnostic prefix; preserves the full-epoch LR schedule.')
    parser.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)  # checks: 44 rows, small chunks
    args = parser.parse_args()
    for key in ('steps', 'stop_after_step'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(key + ' must be positive')
    if min(args.checkpoint_every, args.eval_every) < 0:
        parser.error('Logging intervals must be nonnegative')
    settings = Settings(chain_tokens=128, readout_chunk=64) if args.smoke else Settings()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    try:
        train_stages(args, settings, device, rank, world)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
