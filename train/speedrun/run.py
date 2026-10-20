"""Track 2: the shortest training time on 8xH100 to reach validation loss TARGET, without backpropagation.

Boost: stagewise, forward-only training of the residual stream. Every sub-block (each attention, each MLP) is trained
only to lower the loss of the shared head reading the stream right after it (the logit lens): each sub-block is a
stage of gradient boosting, and all stages train at once. No credit crosses a stage, so none is moved through
weights, attention or skip connections. Every learning signal is a measured change of a loss caused by a
perturbation at the site it trains, or one local step from such a measurement:

- Readout credit. A readout loss is a per-token function of the stream at its depth, so probing every token's stream
  at once has no cross-token interference. Forward differences along all 512 rows of a sign-flipped order-512
  Hadamard matrix give each token's credit in full. The head is linear in the stream before its norm, so a probe's
  logits are the clean logits plus a precomputed direction, rescaled by the moved stream's own RMS: each probe is one
  soft-capped cross-entropy of the actual head, evaluated at the moved stream.
- Writers (attention and MLP output projections): their output credit is the readout credit where they write.
- Attention outputs (hubs): probed along the attention output, through the output projection, to the same readout.
- MLP hidden units: one local step from the measured readout credit. Each token's 256 largest units move along a
  full order-256 Hadamard cycle; each move's change of the block output, through relu^2 and the output projection,
  is scored against that credit.
- Queries, keys, values and value-embedding gates: one local step from the measured hub credit (the record's local
  errors), scored per head. Keys, values and gates carry their scores back to the moved positions with a matched
  filter: query s's score counts for key t in proportion to A[s, t] / max A[s, :], from the clean attention. Values
  and gates move the output through the clean attention probabilities (the attention is linear in its values).
  Value embeddings reuse the value draws: a value move is a value-embedding move scaled by the gate.
- Token embedding: the first attention stage's readout along each token's own path. Every query sees the clean keys
  and values of the other positions and its own moved ones, so each token's measurement is its own.
- Head: the record's coordinate differences at the final readout and at every intermediate readout.
- Residual mixing scalars: central differences of their attention stage's readout loss, in float32.

Four sequences per update: the two GPUs of each sequence split its probe directions, draws and scalars, and add
their partial credits with one all-reduce per readout. The model, init, optimizer (ANVIL II on the block matrices,
Adam elsewhere), learning rates, schedule shape and data order are the record's.

Launch: torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.eval import evaluate
from utils.model import GPT, GPTConfig, attention, apply_rotary_emb, flex, has_ve, norm, window_mask
from utils.prepare_data import TRAIN_ROWS, load_rows

BATCH = 4  # sequences per update; 1,220 updates per epoch
UPDATES = 1050  # the first 4,200 of the 4,880 sequences already reach TARGET: the run stops there
SEED = 0
TARGET = 4.2  # validation loss to reach; the run counts only if its final weights reach it
WIDTH = 512
EPS = torch.finfo(torch.float32).eps
torch._dynamo.config.cache_size_limit = 64
torch._inductor.config.max_autotune = True  # kernels tuned while compiling, in the untimed warmup
torch._inductor.config.coordinate_descent_tuning = True


@dataclass
class Recipe:
    """The record's optimizer settings."""
    lr: float = 0.003       # Adam; embeddings 300x, mixing scalars 30x
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.3   # final fraction of the run with linear lr decay
    anvil_lr: float = 0.023
    anvil_wd: float = 2.25


@dataclass
class Settings:
    """Boost's measurements. Draw counts are per sequence; its GPUs split them."""
    sigma: float = 0.005        # readout probe step, relative to the probed vector's RMS at that token
    fc_k: int = 256             # MLP units probed per token (a power of two)
    fc_sigma: float = 0.05      # their step, relative to the token's pre-activation RMS
    # attention-internal draws. Keys rerun the attention in whole chunks of nrep draws: of each GPU's 160 key draws,
    # 128 run (the estimate's scale does not matter to ANVIL's orthogonalized update)
    local: dict = field(default_factory=lambda: dict(q=64, k=320, v=512, gate=64))
    nrep: int = 64              # draws per batched attention rerun
    emb_chunk: int = 16         # embedding probes per batched rerun
    scalar_step: float = 0.01   # central-difference step of the mixing scalars
    chunk: int = 2 ** 30        # elements per probe chunk (tokens x probes x vocab): few, large launches


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


HADAMARDS = {}


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


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    """The attention each token would see if only its own state were perturbed: every query attends to the clean
    keys and values of all other positions (they are unperturbed in that counterfactual, and later positions cannot
    reach it) and to its own perturbed key and value. Computed from one attention over the clean keys and values, by
    swapping the token's own clean key and value for its perturbed ones through the log-sum-exp. Tensors are
    [batch, token, head, dim]; clean ones are already repeated over the draws."""
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = flex(q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask,
                    return_lse=True)
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()  # [b, t, h, d], [b, t, h]
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)  # its clean key's share
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)  # its perturbed key's weight, same scale
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)


frozen_attention_compiled = torch.compile(frozen_attention)


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle, shared by all
    tokens: (loss moved by +step - clean loss) / step. base (n, V) W x of the clean streams; C (P, V) row j is W D_j;
    steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,) |D_j|^2. The moved stream's RMS comes from these,
    so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]  # (n, P): each direction's logit at the token's target
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / WIDTH + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    """Queries or keys moved by sigma * a and processed like the model's (rotary, then the per-head norm).
    raw (b, t, h, hd) float; a (n, b, t, h, hd); returns (n * b, t, h, hd) in a's dtype."""
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    """Each draw's change of the attention output, scored per head against the hub credit: out (n, b, t, h, hd),
    baseline and hub (b, t, h, hd); returns (b, h, t, n) float."""
    change = out.float() - baseline.float().unsqueeze(0)
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def value_scores(change, hub):
    """change (b, h, t, n, hd): each draw's attention output change; hub (b, t, h, hd); returns (b, h, t, n)."""
    return (change.float() * hub.float().permute(0, 2, 1, 3).unsqueeze(3)).sum(-1)


@torch.compile(dynamic=False)
def heads_first(move):
    """(n, b, t, h, hd) -> (b, h, t, n * hd) in bfloat16: the draws as columns of each head's value matrix."""
    n, b, t, h, hd = move.shape
    return move.to(torch.bfloat16).permute(1, 3, 2, 0, 4).reshape(b, h, t, n * hd)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    """sum over draws of each position's credited score times its code: credited (b, h, t, n), a (n, b, t, h, w);
    returns (b, t, h, w) float."""
    return (credited.permute(3, 0, 2, 1).unsqueeze(-1) * a.float()).sum(0)


@torch.compile(dynamic=False)
def hidden_scores(ut, st, Hk, cols, g):
    """Top units ut (m, k) moved by +-st along the rows of Hk; each move's change of the block output (through
    relu^2 and the units' output vectors cols (m, k, W)) scored against the measured readout credit g (m, W)."""
    moved = st.unsqueeze(-1) * Hk  # (m, k probes, k units)
    change = F.relu(ut.unsqueeze(1) + moved).square() - F.relu(ut.unsqueeze(1) - moved).square()
    stream_change = torch.bmm(change.to(torch.bfloat16), cols)  # (m, probes, W)
    scored = torch.bmm(stream_change, g.to(torch.bfloat16).unsqueeze(-1)).squeeze(-1).float()
    return scored / (2 * st)


@torch.compile(dynamic=False)
def embedding_moves(e, steps, rows, sign, wq, wk, wv, cos, sin):
    """Raw embeddings moved along probe rows (n directions at once), normed, and the first block's queries, keys
    and values of the moved streams: e (b, t, W), steps (b, t, 1), rows (n, W)."""
    n, (b, t, width) = rows.shape[0], e.shape
    h, hd = cos.shape[2], 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + sign * steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)
    heads = lambda z: z.view(n * b, t, -1, hd)
    q = norm(apply_rotary_emb(heads(F.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(F.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(F.linear(x0, wv))


@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    """Per-token loss of the head reading streams x (n, WIDTH)."""
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)


@torch.compile(dynamic=False)
def stage_readout(x0, out, mix, wproj, weight, y):
    """The first attention stage's readout loss per token: stream mix * x0 plus the attention module's output."""
    xmid = mix * x0 + F.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)  # the same on every GPU
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
        """Random +-1 in the draw dtype, from the GPU's own generator or the one shared by all GPUs."""
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d, dtype=None):
        """Draws start..start + n of this step's world * draws at `site`. Draw i at a token is row (offset + i)
        mod d of the Hadamard matrix under the sign pattern of cycle i // d: the patterns (one per cycle of d
        draws, drawn up front in the same order on every GPU) and the tokens' row offsets are shared by the
        GPUs, so a token's draws are orthogonal across chunks and GPUs within each cycle, and the independent
        offsets leave the tokens' errors uncorrelated. Each draw's own sign flip keeps distinct draws
        uncorrelated. E[a a^T] = I."""
        if site not in self.frames:  # this step's frame at the site
            cycles = -(-self.world * draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = self.rank * draws + chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d  # (n, b, t)
        a = hadamard(d, self.device, self.dtype)[rows] * patterns[index // d][:, None, None, :] * self.signs(n, b, t, 1)
        return a if dtype in (None, self.dtype) else a.to(dtype)

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()  # this step's draw frames
        self.capturing = True
        try:
            with torch.autocast('cuda', dtype=self.dtype):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast('cuda', dtype=self.dtype):
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma=0.1):
        """The head's output error by a central difference of every token's loss along each of its own logits."""
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
    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        # split GPUs share one sequence and divide its measurements (probe directions, draws, scalars) between them
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        # each sequence's GPUs share its draw frames and split its draws: the draws see a world of `split`
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)  # shared by the group

    def close(self):
        self.capture.close()

    def share(self, total):
        """This GPU's range of `total` directions (or draws) of its sequence."""
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor):
        if self.split > 1:
            torch.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        """This step's probe basis: the order-512 Hadamard matrix with randomly flipped columns (rows stay
        orthogonal, |row|^2 = 512)."""
        return hadamard(WIDTH, self.device, torch.float32) * self.signs(WIDTH)

    def probe(self, x, y, mix=None, probed=None):
        """Each token's readout credit in the probed space, measured along a full cycle of the step's basis.

        x (n, WIDTH): the clean streams at the readout (float32). Probe j moves the probed vector by step along basis
        row h_j; the stream moves by step * M h_j, with M the map from the probed vector to the stream (the identity,
        or an output projection). probed: the probed vector per token (its RMS sets the step); defaults to x.
        Returns (n, WIDTH): sum_j d_j h_j / WIDTH with d_j the measured directional derivative. The GPUs of a
        sequence each take a slice of the cycle and add their parts."""
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T  # (WIDTH, WIDTH): column j is M h_j
        C = (weight @ stream_dirs).T.contiguous()  # (WIDTH, V): row j is W M h_j
        dir_sq = stream_dirs.square().sum(0)  # |M h_j|^2
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        # a draw of per-token row offsets that the full cycle does not use, kept so the generator's stream is the
        # development runs' (boost_v10_b4)
        torch.randint(0, WIDTH, (n, 1), device=self.device, generator=self.generator)
        credit = torch.zeros(n, WIDTH, device=self.device)
        lo, hi = self.share(WIDTH)  # this GPU's rows of the cycle; the group sums the parts
        C, stream_dirs, dir_sq, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq)
            credit[t0:t0 + per] = derivative @ Hs / WIDTH
        return self.group_sum(credit)

    def embedding_block0(self, e, y, x):
        """The token embedding's credit for the first attention stage's readout, along each token's own path:
        probes move each token's raw embedding (a full Hadamard cycle), the moved embedding runs through the first
        block's attention, where every query sees the clean keys and values of the other positions and its own moved
        key and value (frozen context: no token's move reaches another token, so there is no interference), and the
        head reads the stream after the attention. Returns (n, WIDTH) credit at the raw embedding."""
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
        e = e.float().view(b, t, WIDTH)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # (b, t, 1)
        derivative = torch.zeros(b, t, WIDTH, device=self.device)
        n = self.s.emb_chunk
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        # the clean stage loss, the reference of the forward differences
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, WIDTH).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, WIDTH), weight, y.reshape(-1)).view(1, b, t)
        lo, hi = self.share(WIDTH)
        for j0 in range(lo, hi, n):
            rows = H[j0:j0 + n]  # (n, WIDTH)
            xm0, q, k, v = embedding_moves(e, steps, rows, 1.0, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / WIDTH).reshape(b * t, WIDTH)

    def attention_maps(self, i):
        """Layer i's clean attention probabilities A[b, h, s, t] (from the cached queries and keys) and the matched
        filter K[b, h, t, s] = A[s, t] / max_t' A[s, t'] that carries query s's score to key position t. Cached for
        the step."""
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
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)  # [b, h, s, t]
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        window = model.window_sizes[i][0]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= window)), float('-inf'))
        A = logits.softmax(-1)
        del logits
        K = (A / A.amax(-1, keepdim=True)).transpose(-1, -2)
        self.maps[i] = (A.to(torch.bfloat16), K.to(torch.bfloat16).contiguous())
        return self.maps[i]

    def query_key_errors(self, kind, draws, targets):
        """Query or key credit, one local step from the measured hub credit, scored per head: each draw moves every
        position's raw query (or key), the attention reruns with the moves (one-sided, step 0.1), and each query's
        output change is scored against its hub credit. Key scores reach the moved positions through the matched
        filter."""
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = 0.1
        n = chunk = min(self.s.nrep, draws)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L):
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_' + kind
            A, K = self.attention_maps(i)
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q_raw), processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = attention(qp.to(dt), kp.to(dt), values.to(dt), window)  # (b, t, h, hd)
            rep = lambda z_: z_.to(dt).unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            fixed, vals = rep(kp if kind == 'q' else qp), rep(values)
            raw = q_raw if kind == 'q' else k_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = moved_heads(raw, a, sigma, cos, sin)
                qa, ka = (moved, fixed) if kind == 'q' else (fixed, moved)
                out = attention(qa, ka, vals, window).view(n, b, t, h, hd)
                score = head_scores(out, baseline, target)  # (b, h, t, n): one-sided change . hub credit
                if kind == 'k':
                    score = (K.to(dt) @ score.to(dt)).float()
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors

    def value_errors(self, kind, draws, targets):
        """Credit of the values ('v') or value-embedding gates ('gate'), one local step from the measured hub credit:
        every position's value moves along its own orthogonal code, the attention (linear in its values, with the
        clean probabilities) carries the moves to every query's output, each query's output change is scored per
        head against its hub credit, and the matched filter carries the scores back to the moved positions."""
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        bf = torch.bfloat16
        errors = {}
        n = min(64, draws)
        for i in range(cfg.n_layer):
            if kind == 'gate' and not has_ve(i, cfg.n_layer):
                continue
            A, K = self.attention_maps(i)
            hub = targets[i].view(b, t, h, hd)
            site = f'transformer.h.{i}.attn.c_v' if kind == 'v' else f'transformer.h.{i}.attn.ve_gate'
            width = h if kind == 'gate' else cfg.n_embd
            w = width // h
            estimate = torch.zeros(b, t, h, w, device=self.device)
            if kind == 'gate':
                z = d.outputs[f'transformer.h.{i}.attn.ve_gate'].float()
                ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            for chunk in range(draws // n):
                a = d.orthogonal_noise(site, draws, chunk, n, b, t, width, dtype=torch.float32)
                if kind == 'v':
                    move = a.view(n, b, t, h, hd)
                else:  # a gate move, through the sigmoid (central difference) and the value embedding
                    eps = 0.1
                    slope = (torch.sigmoid(z + eps * a) - torch.sigmoid(z - eps * a)) / eps
                    move = slope.unsqueeze(-1) * ve
                change = (A @ heads_first(move)).view(b, h, t, n, hd)  # the attention output's change, per draw
                score = value_scores(change, hub)  # (b, h, t, n), scored per head against the hub credit
                credited = (K @ score.to(bf)).float()  # carried back to the moved positions
                estimate += code_estimate(credited, a.view(n, b, t, h, w))
            errors[site] = (estimate / draws).reshape(b, t, width)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """MLP hidden credit, one local step from the measured readout credit g at the block output.

        u (n, HIDDEN): pre-activations. Each token's fc_k largest units move by +-step along the rows of a
        sign-flipped order-fc_k Hadamard matrix (a full cycle on those units); each move's change of the block
        output, through relu^2 and the output projection, is scored against g. Returns (n, HIDDEN), zero outside
        the top units."""
        k = self.s.fc_k
        Hk = hadamard(k, self.device, torch.float32) * self.signs(k)
        top = u.topk(k, dim=-1, sorted=False).indices
        uk = u.gather(1, top)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # (n, 1)
        columns = out_weight.to(torch.bfloat16).T.contiguous()  # (HIDDEN, WIDTH): row i is unit i's output vector
        credit = torch.zeros_like(u)
        lo, hi = self.share(k)
        Hs = Hk[lo:hi]  # this GPU's moves (rows) of the cycle; the parts' credits add
        per = max(1, 2 ** 26 // (k * WIDTH))
        for t0 in range(0, u.shape[0], per):
            ut, st, tt, gt = uk[t0:t0 + per], steps[t0:t0 + per], top[t0:t0 + per], g[t0:t0 + per]
            cols = F.embedding(tt, columns)  # (m, k, WIDTH): each token's top units' output vectors
            derivative = hidden_scores(ut, st, Hs, cols, gt)  # (m, probes)
            credit[t0:t0 + per].scatter_(1, tt, derivative @ Hs / k)
        return credit

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):
        """Per-token readout loss of attention stage i run on the stream mixed with the given scalars."""
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        # in float32: the scalars' differences are small, and bfloat16 rounding would swamp them
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    @torch.no_grad()
    def estimate(self, x, y):
        """Every parameter's gradient estimate on this GPU's sequence, summed over its tokens."""
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {}
        head = d.head_error()  # coordinate differences at the final readout
        gradients['lm_head.weight'] = d.reassemble('lm_head', head)
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        x0 = d.x0
        targets = {}
        streams = []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i]
            xmid = xin + d.outputs[p + 'attn.c_proj']
            xout = xmid + d.outputs[p + 'mlp.c_proj']
            streams.append((xin, xmid, xout))
            xm, xo = xmid.float().flatten(0, 1), xout.float().flatten(0, 1)
            block = model.transformer.h[i]
            # attention stage: the readout right after the attention
            att_in = d.inputs[p + 'attn.c_proj'].flatten(0, 1)  # the attention output (hub)
            g_mid = self.probe(xm, yf)
            gradients[p + 'attn.c_proj.weight'] = g_mid.T @ att_in
            hub = self.probe(xm, yf, mix=block.attn.c_proj.weight, probed=att_in)
            targets[i] = hub.view(b, t, WIDTH)
            # MLP stage: the readout right after the MLP
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
        # attention internals: one local step from the measured hub credit; each GPU of a sequence takes its share of
        # the draws, and the group averages
        self.maps = {}
        for kind, count in s.local.items():
            if kind in ('v', 'gate'):
                local = self.value_errors(kind, count // self.split, targets)
            else:
                local = self.query_key_errors(kind, count // self.split, targets)
            local = {site: error / self.split for site, error in local.items()}
            for site, error in local.items():
                gradients[site + '.weight'] = d.reassemble(site, error)
                if kind == 'v' and has_ve(int(site.split('.')[2]), L):
                    # the same draws, read as moves of the value embedding: a value move a is a value-embedding
                    # move a / gate, so the value-embedding credit is the gate times the value credit
                    i = int(site.split('.')[2])
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    heads = gate.shape[-1]
                    ve_error = (error.view(b, t, heads, -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
        # token embedding: the first attention stage's readout along each token's own path
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, WIDTH))
        # mixing scalars: central differences of their attention stage's readout loss; the GPUs of a sequence take
        # alternate blocks
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
        sequences = self.world // self.split  # GPU groups; each group's contributions add to its sequences' mean
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])  # one all-reduce
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
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()


def make_model(device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(GPTConfig())
    model.to_empty(device=device)
    model.init_weights()
    # output projections: uniform with std 1.5 / sqrt(fan_in) instead of zeros
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
# PR #360's total, warmup, cooldown and engage steps; scaled to the run length
RUN = (1194, 240, 50, 514)


def polar(g):
    """Approximate orthogonalization with six quintic maps."""
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (1.05 * x.norm() + 1e-6)).bfloat16()
    for a, b, c in MAPS:
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360."""

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

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            for p in group['params']:
                g = p.grad.float()
                lanes = 1 if p.size(0) >= p.size(1) else 0
                state = self.state[p]
                if not state:
                    state.update(fast=torch.zeros_like(g), slow=torch.zeros_like(g),
                                 energy=torch.zeros_like(g.mean(lanes, keepdim=True)))
                fast, slow, energy = state['fast'], state['slow'], state['energy']
                fast.lerp_(g, 1 - fast_beta)
                slow.lerp_(g, 1 - self.slow_beta)
                update = polar(g.lerp(weight * fast + (1 - weight) * slow, beta))
                energy.lerp_(update.square().mean(lanes, keepdim=True), 1 - self.lane_beta)
                gained = update * energy.clamp_min(1e-10).rsqrt()
                update = gained * (update.norm() / gained.norm().clamp_min(1e-10))
                w = p.float()
                decay = (slow * w >= 0) * w * (wd * lr)
                p.copy_(w - lr * max(1, p.size(0) / p.size(1)) ** 0.5 * (update + decay))
        self.t += 1


class Optimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


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
    return Optimizers(Anvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps), adam)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--steps', type=int, help='Stop after this many updates (smoke tests).')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many updates.')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    split = world // BATCH if BATCH < world else 1  # GPUs per sequence
    if (BATCH % world if split == 1 else world % BATCH):
        raise ValueError('The GPUs split the batch by sequences, or each sequence evenly between GPUs.')
    groups = [None]
    if split > 1:
        groups = [torch.distributed.new_group(list(range(g * split, (g + 1) * split))) for g in range(world // split)]
    recipe, settings = Recipe(), Settings()

    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps = min(len(training) // BATCH, args.steps or UPDATES)
    per = max(1, BATCH // world)  # sequences per GPU group
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, batch=BATCH, steps=steps, seed=SEED, target=TARGET,
            recipe=asdict(recipe), settings=asdict(settings), torch_version=torch.__version__,
        ), indent=2) + '\n')

    def init():
        model = make_model(device, SEED)
        group = groups[rank // split] if split > 1 else None
        return model, make_optimizer(model, recipe, steps), Boost(model, settings, SEED, rank, world, split, group)

    def set_rate(optimizer, step):
        left = (steps - step + 1) / steps
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / recipe.warmdown if recipe.warmdown else 1.0)

    # untimed warmup on random tokens (compilation), then rebuild from the seed
    model, optimizer, estimator = init()
    noise = torch.randint(0, 4096, (per, 2049), generator=torch.Generator().manual_seed(SEED)).to(device)
    estimator.step(noise[:, :-1].contiguous(), noise[:, 1:].contiguous(), optimizer)
    estimator.close()
    model, optimizer, estimator = init()

    if rank == 0:
        print(f'speedrun: {steps} steps, settings={asdict(settings)}, GPUs={world}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        first = (step - 1) * BATCH + (rank // split) * per
        batch = training[first:first + per].to(device)
        set_rate(optimizer, step)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and step % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record['val_loss'] = evaluate(model.state_dict(), args.data_dir)
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()

    estimator.close()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if rank == 0:
        torch.save(state, args.output / 'final.pt')
    val_loss = evaluate(state, args.data_dir)
    if rank == 0:
        result = dict(val_loss=val_loss, train_seconds=round(elapsed, 2), steps=step,
                      tokens=step * BATCH * 2048, reached_target=val_loss <= TARGET)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
