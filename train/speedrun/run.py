"""Track 2: #39 with a faster update (#38's estimator, constants and 400 updates). #38 = #37 in 400 updates with the
head and gates at twice #32's Adam rate, ANVIL II at 0.024 and 512-token rows for the first half; #33-#38 are
@erodola's. The update was developed from #37 alongside #38 / #39 and replaces #39's: like #39 it runs the readout
cycle's first-order product in TF32, the weight-gradient products on bf16 operands, the hidden credits and the
attention-output targets in bf16, and the key moment kernel at 32 directions per call with 3 stages. Beyond #39 it
writes the weight gradients and reduce-scatters them in bf16 (#39: fp32) and returns the head credits in bf16 (#39:
fp32).

The update, against #37's (the same estimator: what is measured, and where credit goes, is unchanged):
  - Moment kernels (kernels.py): the features (X, Y, X^2, Y^2, XY) are built tile by tile in-kernel from the other
    side's processed rows, interleaved so that the query kernel runs six products per tile (fourteen before) and the
    key kernel five (eight); the closed coefficients come from the raw rows in the epilogue; the values' credit rides
    in the key kernel (no value kernel); the credits are divided by the moves' sizes and stored by token straight into
    the return all-to-all buffer; every running block's heads run in one launch per kernel (the blocks as rows of one
    grid); the latest query blocks run first; tiles from a sweep (3 stages, keys 64 x 32, 32 directions per call).
  - Precision (each checked on seeds 0-4 at matched val): the weight-gradient products take bf16 operands and write
    bf16 gradient stacks, reduce-scattered in bf16 and cast to fp32 inside ANVIL II's compiled step (#37: TF32 on
    fp32 stacks, fp32 collectives); the hidden credits, the attention-output targets (bf16 credit @ the forward's bf16
    output projections) and the head credits that travel back are bf16; the readout cycle's first-order product runs
    in TF32 (#37: fp32); ANVIL II all-gathers the stacks' new bf16 weights (the forward's), each GPU keeping its fp32
    share, and the full fp32 stacks are gathered once after the last update, inside the timer.
  - Glue: the head credits, hidden credits, credit stacks, value-embedding and gate credits in compiled code (no eager
    copies); both all-to-alls asynchronous (the MLP products and the readout, scalar and embedding gradients under
    them); each block's head inputs read in place from the received buffer; the forward's q / k / v projections as
    one batched product; one compiled Adam step for every Adam parameter and an in-place ANVIL II step, with the bias
    corrections, the schedules, the gradients' 1/n and the bf16 embeddings' rounding inside them.

#37's description of its own changes follows (its numbers: 450 updates, #32's rates, fp32 gradients).

#32's trainer (#13 at 8 rows per update, @Euraba; #13 = #29's identity-path credit with moment kernels and one head
per GPU, @RobertTLange, in PyTorch) with #33's structured start, tail average and late skip (#34's skip start), in 450
updates (#36), with a faster runtime: overlapped collectives and a forward compiled as one graph.

The estimator, the optimizers and the work split are #32's, unchanged (below). What this adds, each switchable:
  - The start (`--init prev8`, #12's formula and constants): every head of block 1 starts as a local RoPE head peaking
    at lag 1. Its query and key columns get a rank-one term on the block input's token-shared direction
    (`shared_means`, in closed form from the initial weights) and keep a quarter of their random map. A fixed formula
    of the seed and the architecture: no data, statistic or forward pass.
  - The tail average (`--ema fixed`): an fp32 EMA of every weight from 60% of the run (rate 1 / round(0.25 N)), kept
    inside each later update's CUDA graph; after the last update, inside the timer, the weights move 0.4 of the way to
    it. A fixed mix: nothing is measured or selected.
  - The late skip (`--skip late`): from 20% of the run blocks 0, 2, 4 and 6 run no query / key / value experiments
    (they leave the head pack and both all-to-alls), so their queries, keys and values get no credit and keep their
    weights (the optimizer's moves on them are undone). Fewer measurements and no new signal. Each skip phase is its
    own CUDA graph, captured in the warmup.
  - The run length (`--updates`): N = 450 updates instead of 600, the first N batches of #32's data order, with every
    schedule (the row-length switch at 0.8 N, the cooldown over the last 0.4 N, ANVIL's phases) on that length.
With `--updates 600 --init none --ema none --skip none` the script runs #32's update: the same math, but the forward
is compiled as one graph (#32's runs eagerly, below), so its floating-point rounding differs.

The runtime, against #36's (the same arithmetic otherwise, op for op):
  - The MLP matrices' gradients are reduce-scattered while the head credits run. The square stack's reduce-scatter,
    the all-reduce of the other gradients and both ANVIL all-gathers are asynchronous, each under the next step's work
    (ANVIL steps the MLP stack first, as its gradients arrive first; Adam runs while the matrices gather).
  - The rotary tables are computed once per row length.
  - The forward compiles as one graph: flex attention's log-sum-exp comes through `return_aux=AuxRequest(lse=True)`.
    `return_lse=True` warns, and the warning broke the compiled forward inside its block loop, so the eight blocks
    ran eagerly as many small kernels. Fused, the forward rounds differently (same math).

#32's trainer: #13's estimator and work split at 8 rows per update (600 updates, ANVIL II's lr 0.032). One process per
GPU: each runs the clean pass on its own rows (with fewer rows than GPUs, GPUs share a row and keep their part of it),
the readout cycle (a degree-3 expansion as six matrix products) and the MLP steps on its tokens, and one head's
queries, keys and values over the batch: their inputs arrive in one all-to-all, the exact experiments run as
attention-weighted moments on tensor cores (degree 2, kernels.py: Triton versions of #29's Pallas kernels), and the
credits go back by token in a second all-to-all. The ANVIL matrices' gradients are reduce-scattered to the GPU that
steps them. Each update is one CUDA graph per row length (here also per tail-average and skip phase; collectives
captured inside) that reads its batch and step index from device memory.

    torchrun --standalone --nproc_per_node=8 train/speedrun/run.py

Weights are kept in the JAX trainer's layout ((in, out) matrices, the head (d, vocab)) so formulas read as in #29;
`state_dict()` exports utils/model.py's layout.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.prepare_data import TRAIN_ROWS, load_rows  # noqa: E402
from kernels import moment_key_scores, moment_query_scores  # noqa: E402

F32, BF16 = torch.float32, torch.bfloat16

# recipe (#25 / #29's shape at 8 rows per update (#32): cooldown the last 40%; here 512-token rows until 50%)
BATCH, SEQ, STEPS, SEED = 8, 2048, 400, 0  # the run length: 400 updates (#38; #37: 450, #32: 600)
WARMUP_LEN, WARMUP_UNTIL = 512, STEPS // 2  # #32: 1024-token rows until 80%
COOLDOWN = STEPS * 2 // 5
VOCAB, L, H, D = 4096, 8, 8, 512
HD = D // H
LR, BETA1, BETA2, ADAM_EPS = 0.006, 0.7, 0.98, 1e-8  # the head's and gates' rate: 2x #32's for the shorter run
LR_WTE, LR_VE, LR_LAMBDAS = 0.9, 0.9, 0.003
ANVIL_LR, ANVIL_WD = 0.024, 2.25  # #32: 0.032 (0.8x #29's: at 8 rows 0.04 spikes on some seeds)

GATE_CHANNELS, SOFTCAP = 32, 15.0
NORM_EPS = 1e-6
EPS = float(np.finfo(np.float32).eps)  # the readout norm's epsilon
SIGMA, FC_SIGMA, KEY_SIGMA, HEAD_SIGMA, GATE_STEP = 0.005, 0.05, 0.003, 0.1, 0.1

ANVIL_MAPS = ((3.923798038567, -6.095026865488, 3.905234618423), (3.278126713798, -3.328923386476, 0.989127286973),
              (3.505298394150, -5.137358782410, 1.968325560615), (2.815058591845, -3.685181239622, 1.417196497642),
              (2.245503932403, -2.443826979899, 0.963091710461), (2.256537145403, -2.166840097229, 0.929501253245))
POLAR_MARGIN = 1.1
ANVIL_RUN = (1194, 240, 50, 514)
SQUARE = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj')
START_BLOCK, START_LAG, START_RHO, START_CONTENT = 1, 1, 0.8, 0.25  # every head of block 1 a local RoPE head at lag 1
EMA_FROM, EMA_HORIZON, EMA_MIX = 0.6, 0.25, 0.4  # tail average from 60% of the run over a quarter of it; final mix
START, EMA = True, True  # the structured start and the tail average (--init none / --ema none: #32's recipe)
SKIP_BLOCKS, SKIP_FROM, SKIP = (0, 2, 4, 6), 0.2, True  # late skip: no attention experiments in these blocks


def has_ve(layer):
    return layer % 2 == (L - 1) % 2


VE_BLOCKS = tuple(i for i in range(L) if has_ve(i))


def window(layer):
    if layer == L - 1:
        return SEQ
    return SEQ if 'SSSL'[layer % 4] == 'L' else SEQ // 2


def norm(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)


def softcap(x):
    return SOFTCAP * torch.tanh(x / SOFTCAP)


def hadamard(d, device):
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device)


# ------------------------------------------------------------------------------------------------------------- model

class Model:
    """Parameters (JAX layout), optimizer state and the static buffers of the captured step."""

    def __init__(self, device, rank, world):
        self.device, self.rank, self.world = device, rank, world
        g = torch.Generator(device='cpu').manual_seed(SEED)
        s = math.sqrt(3.0) * D ** -0.5
        uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
        proj = lambda fan_in, fan_out: uni((fan_in, fan_out), 1.5 * math.sqrt(3.0 / fan_in))
        # block matrices live in two stacks, oriented wide side out as ANVIL steps them: square [kind * L + layer]
        # (q, k, v, proj), wide [c_fc layers, then c_proj layers transposed]
        self.square = torch.stack([uni((D, D), s) if k != 'attn.c_proj' else proj(D, D)
                                   for k in SQUARE for _ in range(L)]).to(device)
        self.wide = torch.stack([uni((D, 4 * D), s) for _ in range(L)] + [proj(4 * D, D).T for _ in range(L)]).to(device)
        self.gates = {i: torch.zeros(GATE_CHANNELS, H, device=device) for i in range(L) if has_ve(i)}
        self.wte = torch.randn(VOCAB, D, generator=g).to(device, BF16)
        self.lm_head = (0.001 * torch.randn(D, VOCAB, generator=g)).to(device)
        self.ve = {i: uni((VOCAB, D), s).to(device, BF16) for i in range(L) if has_ve(i)}
        self.lam_r = torch.ones(L, device=device)
        self.lam_0 = torch.full((L,), 0.1, device=device)
        if START:
            structured_start(self)
        # the ANVIL matrices' gradient stacks (each GPU's partial sums, reduce-scattered to the GPU that steps them),
        # and one flat buffer (one all-reduce) for the rest
        shapes = dict(wte=self.wte.shape, lm_head=self.lm_head.shape,
                      lam_r=(L,), lam_0=(L,), **{f'gate{i}': (GATE_CHANNELS, H) for i in self.gates},
                      **{f've{i}': (VOCAB, D) for i in self.ve})
        sizes = {k: math.prod(v) for k, v in shapes.items()}
        self.flat_grad = torch.zeros(sum(sizes.values()), device=device)
        self.grad, offset = {}, 0
        for k, shape in shapes.items():
            self.grad[k] = self.flat_grad[offset:offset + sizes[k]].view(shape)
            offset += sizes[k]
        # the block matrices' gradients in bf16 end to end: written by the bf16 products, reduce-scattered in bf16,
        # cast and scaled inside ANVIL's compiled step
        self.grad['square'] = torch.zeros(self.square.shape, device=device, dtype=BF16)
        self.grad['wide'] = torch.zeros(self.wide.shape, device=device, dtype=BF16)
        self.owned = {name: torch.zeros(w.shape[0] // world, *w.shape[1:], device=device, dtype=BF16)
                      for name, w in (('square', self.square), ('wide', self.wide))}  # this GPU's summed share
        # Adam on everything but the ANVIL matrices; ANVIL state per stacked matrix
        self.adam = [(self.wte, self.grad['wte'], LR_WTE), (self.lm_head, self.grad['lm_head'], LR),
                     (self.lam_r, self.grad['lam_r'], LR_LAMBDAS), (self.lam_0, self.grad['lam_0'], LR_LAMBDAS)]
        self.adam += [(self.gates[i], self.grad[f'gate{i}'], LR) for i in self.gates]
        self.adam += [(self.ve[i], self.grad[f've{i}'], LR_VE) for i in self.ve]
        self.m = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.v = [torch.zeros(p.shape, device=device) for p, _, _ in self.adam]
        self.anvil = {name: dict(fast=torch.zeros_like(w), slow=torch.zeros_like(w),
                                 energy=torch.zeros(w.shape[0], 1, w.shape[2], device=device))
                      for name, w in (('square', self.square), ('wide', self.wide))}
        self.H = hadamard(D, device)
        self.rows = torch.randint(0, VOCAB, (STEPS, BATCH, SEQ + 1), device=device)  # the batches (random until training)
        self.losses = torch.zeros(STEPS, device=device)
        self.step_t = torch.zeros((), device=device)  # the update index (0-based)
        self.scale_t = torch.ones((), device=device)  # the warmup-stable-decay factor
        self.rates = dict(square=torch.ones(4 * L, device=device),
                          wide=torch.tensor([2.0] * L + [1.0] * L, device=device))  # sqrt(max(1, out / in))
        self.square_bf, self.wide_bf = self.square.to(BF16), self.wide.to(BF16)  # the forward's bf16 weights
        # during training only this GPU's share of the fp32 stacks is current (ANVIL steps it and all-gathers its bf16
        # cast into the forward's weights); the whole fp32 stacks are gathered once at the end (`gather_fp32`)
        self.shard = {name: slice(rank * (w.shape[0] // world), (rank + 1) * (w.shape[0] // world))
                      for name, w in (('square', self.square), ('wide', self.wide))}
        self.lm_head_bf = self.lm_head.to(BF16)
        self.gates_bf = {i: g.to(BF16) for i, g in self.gates.items()}
        self.ema = [torch.zeros(w.shape, device=device) for w in self.averaged()] if EMA else None
        self.windows = {run: torch.tensor([window(i) for i in run], device=device, dtype=torch.int32)
                        for run in (tuple(range(L)), tuple(i for i in range(L) if i not in SKIP_BLOCKS))}
        self.skip_rows = {SKIP_BLOCKS: torch.tensor([kind * L + i for kind in range(3) for i in SKIP_BLOCKS],
                                                    device=device)}  # the square stack's q, k, v rows of the skip
        mod = lambda w: (lambda b, h, qi, ki: (qi >= ki) & (qi - ki <= w))
        self.masks = {(w, t): create_block_mask(mod(w), None, None, t, t, device=device)
                      for w in (SEQ // 2, SEQ) for t in (WARMUP_LEN, SEQ)}
        self.rope = {t: rotary(t, device) for t in (WARMUP_LEN, SEQ)}  # cos, sin per row length

    # views in the JAX trainer's layout
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
        """The Adam parameters' bf16 copies (ANVIL gathers the stacks' bf16 weights itself)."""
        self.lm_head_bf.copy_(self.lm_head)
        for i, g in self.gates.items():
            self.gates_bf[i].copy_(g)

    @torch.no_grad()
    def cast_all(self):
        """Every bf16 copy from the whole fp32 weights (at the start)."""
        self.square_bf.copy_(self.square)
        self.wide_bf.copy_(self.wide)
        self.refresh_bf16()

    @torch.no_grad()
    def gather_fp32(self):
        """The whole fp32 stacks from every GPU's share (at the end)."""
        for name, w in (('square', self.square), ('wide', self.wide)):
            dist.all_gather_into_tensor(w, w[self.shard[name]].clone())

    def averaged(self):
        """The weights the tail average follows (every parameter, in a fixed order; of the stacks this GPU's share)."""
        return [self.square[self.shard['square']], self.wide[self.shard['wide']], self.wte, self.lm_head, self.lam_r,
                self.lam_0, *self.gates.values(), *self.ve.values()]

    def tensors(self):
        """Every tensor the training state consists of (for snapshots)."""
        out = [self.square, self.wide, self.wte, self.lm_head, self.lam_r, self.lam_0, *self.gates.values(),
               *self.ve.values(), *self.m, *self.v]
        return out + [x for st in self.anvil.values() for x in st.values()]

    def state_dict(self):
        """utils/model.py's state dict (float32, its layout)."""
        sd = {'transformer.wte.weight': self.wte.float(), 'lm_head.weight': self.lm_head.T,
              'resid_lambdas': self.lam_r, 'x0_lambdas': self.lam_0}
        for i in range(L):
            for kind in (*SQUARE, 'mlp.c_fc', 'mlp.c_proj'):
                sd[f'transformer.h.{i}.{kind}.weight'] = self.w(i, kind).T
            if has_ve(i):
                sd[f'transformer.h.{i}.attn.ve_gate.weight'] = self.gates[i].T
                sd[f'value_embeds.{i}.weight'] = self.ve[i].float()
        return {k: v.detach().float().contiguous().cpu() for k, v in sd.items()}


def rope_pair(lag, rho, generator):
    """A query and key whose rotary logit is a bump at `lag` whatever the position: energy rho^j on frequency pair j
    (j = 0 the fastest), random phases, and the key is the query moved `lag` positions ahead. RMS 1 each."""
    half = HD // 2
    phase = 2 * math.pi * torch.rand(half, generator=generator)
    amplitude = rho ** (torch.arange(half, dtype=F32) / 2)
    q = torch.cat([amplitude * phase.cos(), amplitude * phase.sin()])
    q = q * math.sqrt(HD) / q.norm()
    angle = lag / 10000 ** (torch.arange(0, HD, 2, dtype=F32) / HD)
    q1, q2 = q[:half], q[half:]
    return q, torch.cat([q1 * angle.cos() + q2 * angle.sin(), q2 * angle.cos() - q1 * angle.sin()])


def relu2_moments(mu, sd):
    """E relu(u)^2 and E relu(u)^4 for u ~ N(mu, sd^2)."""
    z = mu / sd
    cdf, pdf = 0.5 * (1 + torch.erf(z / math.sqrt(2))), torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    m2 = (mu ** 2 + sd ** 2) * cdf + mu * sd * pdf
    m4 = (mu ** 4 + 6 * mu ** 2 * sd ** 2 + 3 * sd ** 4) * cdf + (mu ** 3 + 5 * mu * sd ** 2) * sd * pdf
    return m2, m4


@torch.no_grad()
def shared_means(model):
    """Each block's normalized input, averaged over tokens, in closed form from the weights: the part of the stream
    every token shares, written by the relu^2 means of the MLPs below. Mean field: each stream is a shared mean plus
    an isotropic token part; near-uniform attention writes c_v c_proj of the mean; each MLP unit's input is Gaussian."""
    mean, energy, x0, means = torch.zeros(D, device=model.device), float(D), 1.0, []  # E x, E|x|^2, x0's share
    for i in range(L):
        r, lam = model.lam_r[i].item(), model.lam_0[i].item()
        energy = r * r * energy + lam * lam * D + 2 * r * lam * x0 * D
        mean, x0 = r * mean, r * x0 + lam
        means.append(mean / math.sqrt(energy / D))
        attended = (means[-1] @ model.w(i, 'attn.c_v')) @ model.w(i, 'attn.c_proj')
        energy, mean = energy + (attended @ attended).item() + 2 * (mean @ attended).item(), mean + attended
        shared = mean / math.sqrt(energy / D)
        fc, proj = model.w(i, 'mlp.c_fc'), model.w(i, 'mlp.c_proj')
        spread = fc.norm(dim=0) * max(1 - (shared @ shared).item() / D, 1e-3) ** 0.5
        m2, m4 = relu2_moments(shared @ fc, spread)
        written = m2 @ proj
        energy += (written @ written).item() + (proj.square().sum(1) @ (m4 - m2 * m2)).item()
        energy, mean = energy + 2 * (mean @ written).item(), mean + written
    return means


@torch.no_grad()
def structured_start(model):
    """Every head of block 1 starts as a local RoPE head peaking at lag 1: its query and key columns get a rank-one
    term on the block input's token-shared direction (shared_means), so the raw query is the same vector for every
    token, rotated to attend one position back, and keep a quarter of their random map. A fixed formula of the seed
    and the architecture: no data, statistic or forward pass."""
    generator = torch.Generator(device='cpu').manual_seed(SEED + 31)
    mean = shared_means(model)[START_BLOCK]
    for h in range(H):
        cols = slice(h * HD, (h + 1) * HD)
        for kind, v in zip(('attn.c_q', 'attn.c_k'), rope_pair(START_LAG, START_RHO, generator)):
            w = model.w(START_BLOCK, kind)
            w[:, cols] = START_CONTENT * w[:, cols] + torch.outer(mean, v.to(mean.device)) / (mean @ mean)


def rotary(t, device):
    inv_freq = 1.0 / 10000 ** (torch.arange(0, HD, 2, dtype=F32, device=device) / HD)
    freqs = torch.outer(torch.arange(t, dtype=F32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()  # [t, HD / 2]


def rope_norm(raw, cos, sin):
    """RoPE then QK norm in fp32 -> bf16 [b, h, t, d]. raw [b, t, h, d]; cos, sin [t, d/2]."""
    x = raw.float()
    c, s_ = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * c + x2 * s_, x1 * -s_ + x2 * c], -1)
    return (rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)).to(BF16).transpose(1, 2).contiguous()


def forward_impl(model, idx, targets, cos, sin, lo: int, hi: int):
    """The clean pass on this GPU's rows idx [b, t]: (cache, mean loss of its tokens lo..hi of the flattened rows)."""
    b, t = idx.shape
    cache = {}
    x = norm(model.wte[idx])
    x0, pending = x, None
    cache['x0'] = x0
    for i in range(L):
        mixed = model.lam_r[i] * (x if pending is None else x + pending.float()) + model.lam_0[i] * x0
        n1 = norm(mixed).to(BF16)
        q, k, v = torch.bmm(n1.reshape(1, b * t, D).expand(3, -1, -1), model.square_bf[i:3 * L:L]).view(3, b, t, D)
        values = v.view(b, t, H, HD)
        if has_ve(i):
            ve = model.ve[i][idx]
            z = n1[..., :GATE_CHANNELS] @ model.gates_bf[i]
            values = values + (2 * torch.sigmoid(z.float())).unsqueeze(-1).to(BF16) * ve.view(b, t, H, HD)
            cache[f'h{i}.ve_gate'], cache[f'h{i}.ve'] = z, ve
        qp, kp = rope_norm(q.view(b, t, H, HD), cos, sin), rope_norm(k.view(b, t, H, HD), cos, sin)
        o, aux = flex_attention(qp, kp, values.transpose(1, 2).contiguous(), block_mask=model.masks[(window(i), t)],
                                return_aux=AuxRequest(lse=True))  # (return_lse's warning breaks the compiled graph)
        lse = aux.lse
        o = o.transpose(1, 2).reshape(b, t, D)
        cache.update({f'h{i}.in.attn': n1, f'h{i}.q': q, f'h{i}.k': k, f'h{i}.values': values,
                      f'h{i}.lse': lse.float(), f'h{i}.in.attn.c_proj': o})
        x = mixed + (o @ model.wb(i, 'attn.c_proj')).float()
        n2 = norm(x).to(BF16)
        hidden = n2 @ model.wb(i, 'mlp.c_fc')
        act = F.relu(hidden).square()
        pending = act @ model.wb(i, 'mlp.c_proj')
        cache.update({f'h{i}.in.mlp.c_fc': n2, f'h{i}.mlp.c_fc': hidden, f'h{i}.in.mlp.c_proj': act,
                      f'h{i}.stream': x + pending.float()})
    final = norm(x + pending.float()).to(BF16).reshape(b * t, D)[lo:hi]
    raw = final @ model.lm_head_bf
    cache['in.lm_head'], cache['lm_head'] = final, raw
    loss = F.cross_entropy(softcap(raw.float()), targets.reshape(-1)[lo:hi])
    return cache, loss


forward = torch.compile(forward_impl, dynamic=False)


# --------------------------------------------------------------------------------------------------------- estimator

@torch.compile(dynamic=False)
def coordinate_change(raw, targets):
    """The LM head's credit on raw logits: each logit moved up by HEAD_SIGMA minus moved down, over 2 sigma."""
    raw = raw.float()
    z = softcap(raw)
    up, down = softcap(raw + HEAD_SIGMA), softcap(raw - HEAD_SIGMA)
    weight = torch.softmax(z, -1)
    change = torch.log1p(weight * torch.expm1(up - z)) - torch.log1p(weight * torch.expm1(down - z))
    change = change - F.one_hot(targets, VOCAB).float() * (up - down)
    return change / (2 * HEAD_SIGMA)


def step_signs(step_t, device):
    """D random signs from the update index (a counter hash, so a replayed graph draws fresh signs every update)."""
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
    """The readout cycle's moment weights (#29's cycle_derivatives): per token, f(z) = exp(15 tanh(z / 15)) at the
    clean logits z and its log-derivatives; per move j, the norm's change delta_j and the step sigma_j."""
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
    fast = torch.stack([fz * g2, fzz * g3, f * g2, fz * g3, f * g3]).to(BF16)  # [5, n, V]
    return f * g1, fast, rows, total, steps, inv, inv0, delta, sigma


@torch.compile(dynamic=False)
def cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h):
    fast = [x.float() for x in prods]  # the five bf16 products
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
    """The cycle directions' logits c and their squares and cubes in bf16 (the fast products' operands)."""
    return c.to(BF16), (c * c).to(BF16), (c * c * c).to(BF16)


def top_credit(model, x, y):
    """Each token's readout credit at the top stream x (m, d): a whole sign-flipped Hadamard cycle, every moved
    stream's partition function from a degree-3 expansion around the clean logits (six matrix products)."""
    h = model.H * step_signs(model.step_t, x.device)[None, :]
    c = h @ model.lm_head  # (P, V): row j is direction j's logits
    base = x @ model.lm_head
    fg1, fast, rows, total, steps, inv, inv0, delta, sigma = cycle_weights(x, base, h)
    exact = fg1 @ c.T  # the first-order product in TF32 (#29: fp32, HIGHEST)
    cb, c2, c3 = cycle_powers(c)
    prods = [fast[0] @ cb.T, fast[1] @ cb.T, fast[2] @ c2.T, fast[3] @ c2.T, fast[4] @ c3.T]
    return cycle_finish(exact, prods, rows, total, base, c, y, steps, inv, inv0, delta, sigma, h)


def rope32(x, cos, sin):
    """RoPE then QK norm in fp32 over [b, t, d] (one head)."""
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    rot = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    return rot * torch.rsqrt(rot.square().mean(-1, keepdim=True) + NORM_EPS)


REPEAT, REPEAT_K = HD // 16, HD // 32  # moment-kernel programs per head row: 16 query / 32 key directions each
PACK = 5 * HD + 2  # q_raw, k_raw, values, attention output, target (bf16), log-sum-exp (fp32 bits)


@torch.compile(dynamic=False)
def pack_heads(qs, ks, vals, outs, targets, lse_bits):
    """This GPU's tokens' inputs to every block's exact kernels, by head: [h, L, m, PACK] bf16 (lists over blocks:
    q, k (m, d), values (m, h, hd), outputs (m, d), targets (m, d) bf16, the lse's fp32 bits (m, 2h) bf16)."""
    parts = []
    for q, k, v, o, tg, lb in zip(qs, ks, vals, outs, targets, lse_bits):
        m = q.shape[0]
        bits = lb.view(m, H, 2)
        parts.append(torch.cat([q.view(m, H, HD), k.view(m, H, HD), v.view(m, H, HD), o.view(m, H, HD),
                                tg.view(m, H, HD), bits], -1))
    return torch.stack(parts).permute(2, 0, 1, 3).contiguous()


@torch.compile(dynamic=False)
def head_prep(got, lse, cos, sin, b: int, t: int):
    """The running blocks' exact-kernel inputs for this GPU's head over the batch, from got [world, k, m, PACK] (as
    received) and lse [world, k, m] (each block's b rows of t tokens, by GPU), stacked over the blocks [k b, t, .]: the
    processed queries and keys (bf16 for the logits, f32 for the rotated columns), values and hub (bf16), the raw
    rows' squared norms and move sizes, hub . output per token (f32)."""
    k = got.shape[1]
    whole, lse = got.transpose(0, 1).reshape(k * b, t, PACK), lse.transpose(0, 1).reshape(k * b, t)
    q_raw, k_raw, values, out, hub = (whole[..., j * HD:(j + 1) * HD].float() for j in range(5))
    q1, k1 = rope32(q_raw, cos, sin), rope32(k_raw, cos, sin)
    sq_q, sq_k = (q_raw * q_raw).sum(-1), (k_raw * k_raw).sum(-1)
    steps = lambda sq: KEY_SIGMA * torch.clamp(torch.sqrt(sq / HD), min=1e-6)
    return (q1.to(BF16), k1.to(BF16), values.to(BF16), hub.to(BF16), q1, k1, lse.contiguous(), (hub * out).sum(-1),
            sq_q, sq_k, steps(sq_q), steps(sq_k))


def head_scores(got, rope32, windows, back, b, t):
    """The running blocks' query, key and value credits for this GPU's head over the batch (b rows of t tokens each),
    from got [world, k, m, PACK] as received, into back [world, k, 3, m, hd] (the tokens grouped by the GPU that holds
    them); windows [k] int32 the blocks' attention windows. Every block in one launch per kernel."""
    lse_bits = got[..., 5 * HD:].view(F32)[..., 0]  # the lse's fp32 bits back (a view)
    q1, k1, values, hub, q1f, k1f, lse, hub_out, sq_q, sq_k, steps_q, steps_k = head_prep(got, lse_bits, *rope32, b, t)
    moment_query_scores(q1, k1, values, hub, got, sq_q, k1f, *rope32, lse, hub_out, steps_q, back[:, :, 0], windows,
                        REPEAT, block_s=64, block_t=32, num_warps=4, num_stages=3)
    moment_key_scores(q1, k1, values, hub, got[..., HD:], sq_k, q1f, *rope32, lse, hub_out, steps_k, back[:, :, 1],
                      back[:, :, 2], windows, REPEAT_K, block_t=64, block_s=32, num_warps=4, num_stages=3)


@torch.compile(dynamic=False)
def credit_stacks(got, run: tuple, ve_blocks: tuple):
    """From got [source GPU (its head), k, 3, m, hd] bf16: every block's query, key and value credits by token
    [3, L, m, d] (zero in the skipped blocks), the weight-gradient products' operand, and the value-embedding blocks'
    value credits [m, h, hd] in f32."""
    world, _, _, m, hd = got.shape
    by_token = got.permute(1, 2, 3, 0, 4).reshape(len(run), 3, m, world * hd)
    zero = torch.zeros_like(by_token[0])
    stacks = torch.stack([by_token[run.index(i)] if i in run else zero for i in range(L)], 1)
    return stacks, [(by_token[run.index(i), 2] if i in run else zero[2]).view(m, H, HD).float()
                    for i in ve_blocks]


def f32_stack(xs, lo: int, hi: int):
    """The blocks' input streams on this GPU's tokens, stacked [L, m, d] in f32."""
    return torch.stack([x.reshape(-1, x.shape[-1])[lo:hi] for x in xs]).float()


def bf16_stacks(n1, o, n2, act, lo: int, hi: int):
    """The weight-gradient products' inputs on this GPU's tokens, stacked [L, m, ...] in bf16."""
    st = lambda xs: torch.stack([x.reshape(-1, x.shape[-1])[lo:hi] for x in xs]).to(BF16)
    return st(n1), st(o), st(n2), st(act)


@torch.compile(dynamic=False)
def block_credits(credit_top, lam_r):
    """Each block output's credit: the top credit times the residual scalars above it, [L, m, d] (f32, and bf16 for the
    products)."""
    above = torch.flip(torch.cumprod(torch.flip(lam_r, [0]), 0), [0])  # above[i] = lam_r[i] ... lam_r[last]
    scale = torch.cat([above[1:], torch.ones_like(above[:1])])
    cred = credit_top[None] * scale[:, None, None]
    return cred, cred.to(BF16)


@torch.compile(dynamic=False)
def hidden_slope_(u, read, lo: int, hi: int):
    """One block's MLP hidden credit in place of read (its credit read through the output rows, credit @ W_out^T,
    [m, 4d] bf16): each unit's central difference through relu^2 at its pre-activation (u [b, t, 4d])."""
    u = u.reshape(-1, u.shape[-1])[lo:hi].float()
    step = FC_SIGMA * torch.clamp(torch.sqrt(u.square().mean(-1, keepdim=True)), min=1e-6)
    relu2 = lambda z: F.relu(z).square()
    slope = (relu2(u + step) - relu2(u - step)) / (2 * step)
    read.copy_((slope * read.float()).to(BF16))


def hidden_credits(us, credb, w_out_t, lo, hi):
    """Every block's MLP hidden credit [L, m, 4d] (bf16, the product's operand; w_out_t [L, d, 4d] bf16 is W_out^T):
    one product for every block, then each block's slopes in place (one pass over its own pre-activations)."""
    hidden = torch.bmm(credb, w_out_t)
    for u, read in zip(us, hidden):
        hidden_slope_(u, read, lo, hi)
    return hidden


@torch.compile(dynamic=False)
def scalar_credits(cred, prevs, lam_r, lam_0):
    """The mixing scalars' gradients (resid, x0) per block, and the credit of x0."""
    x0 = prevs[0]
    scal_r = (cred * prevs).sum((1, 2))
    scal_0 = (cred * x0[None]).sum((1, 2))
    credit_x0 = (lam_0[:, None, None] * cred).sum(0) + lam_r[0] * cred[0]
    return scal_r, scal_0, credit_x0


@torch.compile(dynamic=False)
def value_embed_credits(z, ve, vcredit, n1):
    """A value-embedding block's gate gradient [GATE_CHANNELS, h] (the gate's central difference) and its embedding
    rows' credits by token [m, d]: z [m, h] its gate's logits, ve [m, d] its embedding rows, vcredit [m, h, hd] the
    values' credit, n1 [m, d] the block's normed input."""
    z, ve = z.float(), ve.float().view(-1, H, HD)
    slope = (torch.sigmoid(z + GATE_STEP) - torch.sigmoid(z - GATE_STEP)) / GATE_STEP
    gate = n1[:, :GATE_CHANNELS].float().T @ (slope * (vcredit * ve).sum(-1))
    return gate, (vcredit * 2 * torch.sigmoid(z)[..., None]).view(-1, D)


def identity_grad(model, idx, targets, lo, hi, rows_total, skip=()):
    """#29's identity-path estimate: the clean pass on this GPU's rows idx [b, t], the readout cycle and the MLP steps
    on its tokens lo..hi, one head's queries, keys and values over the batch. The ANVIL matrices' gradients stay this
    GPU's partial sums, reduce-scattered; the rest are all-reduced. Returns the clean loss of its tokens, the collectives
    in flight and the token count."""
    world, rank = model.world, model.rank
    t = idx.shape[1]
    n, m = rows_total * t, hi - lo
    rope32 = model.rope[t]
    cache, loss = forward(model, idx, targets, *rope32, lo, hi)
    mine = lambda a: a.reshape(-1, a.shape[-1] if a.dim() < 4 else a.shape[-2] * a.shape[-1])[lo:hi]
    stream = lambda name: mine(cache[name]).float()
    y, tokens = targets.reshape(-1)[lo:hi], idx.reshape(-1)[lo:hi]
    coords = coordinate_change(cache['lm_head'], y)
    credit_top = top_credit(model, stream(f'h{L - 1}.stream'), y)
    gsq, gwide = model.grad['square'], model.grad['wide']
    # every block at once: its inputs on this GPU's tokens stacked [L, m, ...] (f32), its output credit
    prevs = f32_stack([cache['x0']] + [cache[f'h{i}.stream'] for i in range(L - 1)], lo, hi)
    n1s, os_, n2s, acts = bf16_stacks([cache[f'h{i}.in.attn'] for i in range(L)],
                                      [cache[f'h{i}.in.attn.c_proj'] for i in range(L)],
                                      [cache[f'h{i}.in.mlp.c_fc'] for i in range(L)],
                                      [cache[f'h{i}.in.mlp.c_proj'] for i in range(L)], lo, hi)
    cred, credb = block_credits(credit_top, model.lam_r)  # [L, m, d]: the top credit down the residual path
    targets_o = torch.bmm(credb, model.square_bf[3 * L:].transpose(1, 2))  # attention-output targets (bf16)
    run = [i for i in range(L) if i not in skip]  # the blocks whose attention experiments run (the late skip)
    k = len(run)
    if k:
        send = pack_heads([mine(cache[f'h{i}.q']) for i in run], [mine(cache[f'h{i}.k']) for i in run],
                          [mine(cache[f'h{i}.values']) for i in run],
                          [mine(cache[f'h{i}.in.attn.c_proj']) for i in run], [targets_o[i] for i in run],
                          [cache[f'h{i}.lse'].transpose(1, 2).reshape(-1, H)[lo:hi].contiguous().view(BF16)
                           for i in run])  # [b, h, t] -> by token, every row of this GPU's
        got = torch.empty_like(send)
        # [source GPU (its tokens), k, m, PACK]: this GPU's head over the batch
        heads_in = dist.all_to_all_single(got, send, async_op=True)
    # while the heads' inputs travel: the weight-gradient products on bf16 operands (fp32 sums)
    hidden = hidden_credits([cache[f'h{i}.mlp.c_fc'] for i in range(L)], credb, model.wide_bf[L:], lo, hi)
    torch.bmm(credb.transpose(1, 2), acts, out=gwide[L:])  # mlp.c_proj, oriented (out, in)
    torch.bmm(n2s.transpose(1, 2), hidden, out=gwide[:L])  # mlp.c_fc
    torch.bmm(os_.transpose(1, 2), credb, out=gsq[3 * L:])  # attn.c_proj
    # the MLP matrices' gradients are complete: reduce them while the head credits run
    works = dict(wide=dist.reduce_scatter_tensor(model.owned['wide'], model.grad['wide'], async_op=True))
    if not k:
        got2 = torch.zeros(world, 0, 3, m, HD, device=idx.device, dtype=BF16)
    else:  # one head per GPU
        heads_in.wait()
        back = torch.empty(world, k, 3, m, HD, device=idx.device, dtype=BF16)  # [GPU that holds the tokens, ...]
        head_scores(got, rope32, model.windows[tuple(run)], back, rows_total, t)
        got2 = torch.empty_like(back)
        heads_out = dist.all_to_all_single(got2, back, async_op=True)  # [source GPU (its head), k, 3, m, hd]
    # while the credits travel back: the readout's, the mixing scalars' and the embedding's gradients
    torch.mm(cache['in.lm_head'].float().T, coords, out=model.grad['lm_head'])
    scal_r, scal_0, credit_x0 = scalar_credits(cred, prevs, model.lam_r, model.lam_0)
    model.grad['wte'].zero_().index_add_(0, tokens, credit_x0)
    model.grad['lam_r'].copy_(scal_r)
    model.grad['lam_0'].copy_(scal_0)
    if k:
        heads_out.wait()
    # the skipped blocks' queries, keys and values get no credit
    by_token, vcredits = credit_stacks(got2, tuple(run), VE_BLOCKS)
    for j in range(3):  # queries, keys, values
        torch.bmm(n1s.transpose(1, 2), by_token[j], out=gsq[j * L:(j + 1) * L])
    works['square'] = dist.reduce_scatter_tensor(model.owned['square'], model.grad['square'], async_op=True)
    for i, vcredit in zip(VE_BLOCKS, vcredits):
        gate, rows = value_embed_credits(mine(cache[f'h{i}.ve_gate']), mine(cache[f'h{i}.ve']), vcredit, n1s[i])
        model.grad[f'gate{i}'].copy_(gate)
        model.grad[f've{i}'].zero_().index_add_(0, tokens, rows)
    works['flat'] = dist.all_reduce(model.flat_grad, async_op=True)
    return loss, works, n


# -------------------------------------------------------------------------------------------------------- optimizers

def anvil_betas(t):
    total, warm, cool, engage = (round(k * STEPS / ANVIL_RUN[0]) for k in ANVIL_RUN)
    lo, hi = 0.85, 0.93
    beta = torch.where(t < warm, lo + (hi - lo) * t / max(warm, 1), torch.full_like(t, hi))
    beta = torch.where(t > total - cool, hi - (hi - lo) * (t - (total - cool)) / max(cool, 1), beta)
    engaged = t >= engage
    return beta, torch.where(engaged, 0.85, beta), torch.where(engaged, 0.4385, 1.0)


@torch.compile(dynamic=False)
def anvil_math(w, g, n: int, fast, slow, energy, step_t, scale_t, rates):
    """ANVIL II on a stack of wide-oriented matrices [k, rows <= cols] from their summed bf16 gradients g over n
    tokens: w, fast, slow and energy are updated in place (in the compiled step's kernels); returns the new weights'
    bf16 cast (the forward's). The schedule (rate, momenta) from the update index inside the step."""
    beta, fast_beta, weight = anvil_betas(step_t)
    lr = scale_t * ANVIL_LR
    g = g.float() / n
    fast_new = fast + (1 - fast_beta) * (g - fast)
    slow_new = slow + (1 - 0.98) * (g - slow)
    x = g + beta * (weight * fast_new + (1 - weight) * slow_new - g)
    x = (x / (POLAR_MARGIN * torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-6)).to(BF16)
    for a, b_, c in ANVIL_MAPS:
        sq = x @ x.transpose(1, 2)
        x = a * x + (b_ * sq + c * (sq @ sq)) @ x
    update = x.float()
    energy_new = energy + (1 - 0.9) * (update.square().mean(1, keepdim=True) - energy)
    gained = update * torch.rsqrt(torch.clamp(energy_new, min=1e-10))
    update = gained * (torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
                       / torch.clamp(torch.linalg.vector_norm(gained, dim=(1, 2), keepdim=True), min=1e-10))
    decay = ((slow_new * w) >= 0) * w * (ANVIL_WD * lr)
    new = w - lr * rates[:, None, None] * (update + decay)
    fast.copy_(fast_new)
    slow.copy_(slow_new)
    energy.copy_(energy_new)
    w.copy_(new)
    return new.to(BF16)


def anvil_update(model, works, n):
    """Steps this GPU's share of each stack once its gradients arrive; returns the all-gathers of the new bf16 weights
    in flight."""
    gathers = []
    for name, w, w_bf in (('wide', model.wide, model.wide_bf), ('square', model.square, model.square_bf)):
        works[name].wait()
        sl, st = model.shard[name], model.anvil[name]
        new = anvil_math(w[sl], model.owned[name], n, st['fast'][sl], st['slow'][sl], st['energy'][sl], model.step_t,
                         model.scale_t, model.rates[name][sl])
        gathers.append((dist.all_gather_into_tensor(w_bf, new, async_op=True), new))
    return gathers


@torch.compile(dynamic=False)
def adam_math(ps, gs, n: int, ms, vs, lrs: tuple, step_t, scale_t):
    """One Adam step on every Adam parameter from its summed gradient over n tokens, the parameters and their moments
    written in place (a bf16 parameter's gradient rounded to bf16 first, as the JAX trainer does); the bias corrections
    and the schedule's factor inside the step."""
    t = step_t + 1
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    for p, g, m, v, lr in zip(ps, gs, ms, vs, lrs):
        g = g / n
        if p.dtype == BF16:
            g = g.to(BF16).float()
        m_new = BETA1 * m + (1 - BETA1) * g
        v_new = BETA2 * v + (1 - BETA2) * g.square()
        p.copy_((p.float() - lr * scale_t * (m_new / c1) / (torch.sqrt(v_new) / torch.sqrt(c2) + ADAM_EPS)).to(p.dtype))
        m.copy_(m_new)
        v.copy_(v_new)


def adam_update(model, works, n):
    works['flat'].wait()
    ps, gs, lrs = zip(*model.adam)
    adam_math(list(ps), list(gs), n, model.m, model.v, lrs, model.step_t, model.scale_t)


def average_step(model, length, skip=()):
    """An update, then the tail average follows the new weights."""
    train_step(model, length, skip)
    rate = 1 / max(1, round(EMA_HORIZON * STEPS))
    torch._foreach_lerp_(model.ema, [w.float() for w in model.averaged()], rate)


def train_step(model, length, skip=()):
    """One update, everything on the device: this step's batch from model.rows, this GPU's rows of it."""
    world, rank = model.world, model.rank
    model.scale_t.copy_(torch.clamp((STEPS - model.step_t) / COOLDOWN, max=1.0))  # warmup-stable-decay
    batch = model.rows.index_select(0, model.step_t.long().view(1))[0]  # [BATCH, 2049]
    inputs, targets = batch[:, :-1].reshape(-1, length), batch[:, 1:].reshape(-1, length)  # rows cut into pieces
    rows_total = inputs.shape[0]
    if rows_total % world == 0:  # whole rows per GPU
        per = rows_total // world
        idx, tgt = inputs[rank * per:(rank + 1) * per], targets[rank * per:(rank + 1) * per]
        lo, hi = 0, per * length
    else:  # GPUs share a row: each runs it and keeps its part of the tokens
        share = world // rows_total
        idx, tgt = inputs[rank // share:rank // share + 1], targets[rank // share:rank // share + 1]
        part = length // share
        lo, hi = (rank % share) * part, (rank % share + 1) * part
    loss, works, n = identity_grad(model, idx, tgt, lo, hi, rows_total, skip)
    if skip:  # a skipped block's queries, keys and values get no credit, so no step at all (moves undone below)
        rows_qkv = model.skip_rows[skip]
        frozen, frozen_bf = model.square.index_select(0, rows_qkv), model.square_bf.index_select(0, rows_qkv)
    gathers = anvil_update(model, works, n)  # from the weights before this step's Adam (the gradients are this step's)
    adam_update(model, works, n)  # while the ANVIL matrices gather
    for work, _ in gathers:
        work.wait()
    if skip:
        model.square.index_copy_(0, rows_qkv, frozen)
        model.square_bf.index_copy_(0, rows_qkv, frozen_bf)
    model.refresh_bf16()
    model.losses.index_copy_(0, model.step_t.long().view(1), loss.view(1))
    model.step_t.add_(1)


# ---------------------------------------------------------------------------------------------------------- training

MODEL = None


def main():
    global STEPS, WARMUP_UNTIL, COOLDOWN, SEED, START, EMA, SKIP
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--steps', type=int, default=0, help='Stop after this many updates (smoke tests).')
    parser.add_argument('--updates', type=int, default=STEPS, help='the run length N (the schedules scale with it)')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--init', choices=('prev8', 'none'), default='prev8', help="none: #32's init")
    parser.add_argument('--ema', choices=('fixed', 'none'), default='fixed')
    parser.add_argument('--skip', choices=('late', 'none'), default='late',
                        help='late: blocks 0, 2, 4, 6 run no attention experiments from 20%% of the run')
    args = parser.parse_args()
    STEPS, WARMUP_UNTIL, COOLDOWN, SEED = args.updates, args.updates // 2, args.updates * 2 // 5, args.seed
    START, EMA, SKIP = args.init == 'prev8', args.ema == 'fixed', args.skip == 'late'
    root = Path(__file__).resolve().parents[2]
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != H:
        raise SystemExit(f'one head per GPU: run on {H} GPUs, not {world}')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', rank)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    global MODEL
    model = MODEL = Model(device, rank, world)
    rows = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS).numpy()
    order = np.random.default_rng((SEED + 1, 0)).permutation(len(rows))
    batches = np.stack([rows[np.sort(order[i:i + BATCH])] for i in range(0, len(rows) - BATCH + 1, BATCH)][:STEPS])
    total = args.steps or STEPS
    snapshot = [x.clone() for x in model.tensors()]
    graphs = {}
    steps_of = (train_step, average_step) if EMA else (train_step,)  # the tail average's updates: their own graphs
    skips = ((), SKIP_BLOCKS) if SKIP else ((),)  # and the late skip's
    for length in (WARMUP_LEN, SEQ):  # warmup on random tokens (model.rows is random until training), then capture
        for update in steps_of:
            for skip in skips:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        update(model, length, skip)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    update(model, length, skip)
                graphs[length, update is average_step, bool(skip)] = g
    with torch.no_grad():  # training starts from the untouched state, on the real batches
        for x, s in zip(model.tensors(), snapshot):
            x.copy_(s)
        model.cast_all()
        model.rows.copy_(torch.from_numpy(batches.astype(np.int64)))
        model.step_t.zero_()
        model.losses.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    log = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script='train/speedrun/run.py', port_of='#29', world=world, batch=BATCH, steps=STEPS, seed=SEED,
            init=args.init, skip=dict(blocks=SKIP_BLOCKS, start=round(SKIP_FROM * STEPS)) if SKIP else None,
            ema=dict(mode=args.ema, start=round(EMA_FROM * STEPS), horizon=round(EMA_HORIZON * STEPS),
                     mix=EMA_MIX) if EMA else None,
            seq_warmup=f'{WARMUP_LEN}:{WARMUP_UNTIL}', cooldown=COOLDOWN, lr=LR, betas=[BETA1, BETA2], lr_wte=LR_WTE,
            lr_value_embeds=LR_VE, lr_lambdas=LR_LAMBDAS, anvil_lr=ANVIL_LR, anvil_wd=ANVIL_WD), indent=2) + '\n')
        log = (args.output / 'metrics.jsonl').open('w')
    first = round(EMA_FROM * STEPS)  # the update at which the tail average starts
    start = time.perf_counter()
    for step in range(total):
        length = WARMUP_LEN if step < WARMUP_UNTIL else SEQ
        graphs[length, EMA and step > first, SKIP and step >= round(SKIP_FROM * STEPS)].replay()
        if EMA and step == first:  # the average starts at the weights after this update
            torch._foreach_copy_(model.ema, [w.float() for w in model.averaged()])
    if EMA and total > first:  # the final weights: w + EMA_MIX (EMA - w), inside the timer
        with torch.no_grad():
            for w, e in zip(model.averaged(), model.ema):
                w.copy_(torch.lerp(w.float(), e, EMA_MIX).to(w.dtype))
    model.gather_fp32()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if rank == 0:
        losses = model.losses.tolist()
        for s in list(range(0, total, 25)) + [total - 1]:
            log.write(json.dumps(dict(step=s + 1, train_loss=round(losses[s], 4))) + '\n')
        log.close()
        torch.save(model.state_dict(), args.output / 'final.pt')
        print(f'trained {total} updates in {elapsed:.2f} s ({elapsed / total * 1000:.2f} ms per update)', flush=True)
    if rank != 0:  # NCCL teardown after captured collectives can hang: leave without it
        os._exit(0)
    if rank == 0:  # utils/eval.py on the final weights, in a fresh process
        env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                                                                 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK')}
        out = subprocess.run([sys.executable, str(root / 'utils' / 'eval.py'), str(args.output / 'final.pt'),
                              '--data-dir', str(args.data_dir)], env=env, capture_output=True, text=True, check=True)
        val = json.loads(out.stdout.strip().splitlines()[-1])['val_loss']
        result = dict(val_loss=val, train_seconds=round(elapsed, 2), steps=total, tokens=total * BATCH * SEQ,
                      reached_target=val <= 4.2)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        os._exit(0)


if __name__ == '__main__':
    main()
