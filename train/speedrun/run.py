"""Track 2: the shortest training time on 8xH100 to reach validation loss TARGET, without backpropagation.

Identity-path credit with fused kernels. One full sign-flipped Hadamard cycle at the top stream measures each token's
credit through the head (the soft-capped cross-entropy); every earlier stream gets it through the residual scalars
only (linear maps), and no signal takes a second nonlinear step. From it each block takes one local step:
  - the MLP hidden units, each by its own central difference of relu^2, scored through the output projection
    (all units, one elementwise pass, no cycle);
  - queries and keys by exact single-coordinate finite differences, evaluated in closed form for all directions at
    once by one fused kernel (causal-window slices, q @ k computed once per head);
  - values and gates by the exact attention-weighted sum of the target (linear);
  - the writers and mixing scalars from the stream credit directly, the embedding by one RMS-norm step of it.
There is no Boost or Cut phase and no chaining of credit from block to block.

The model, init, optimizer (ANVIL II on the block matrices, Adam elsewhere), learning rates, schedule shape and data
order are the record's. `--no-no-chain` / `--no-skip-mid` restore the chained estimators of the closed PRs #20/#22
(kept for comparison; they do not satisfy the no-chaining rule).

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
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, norm, has_ve
from utils.prepare_data import TRAIN_ROWS as DATA_ROWS, load_rows

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def maybe_compile(*args, **kwargs):
    def decorator(fn):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
            return fn
        try:
            return torch.compile(fn, **kwargs)
        except Exception:
            return fn
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator


BATCH = 8  # sequences per update (one per GPU on 8xH100)
SEED = 0
TARGET = 4.2  # validation loss to reach
SEQ = 2049
TRAIN_ROWS = 4880
VAL_ROWS = 544
PRESETS = {
    'tiny': (4, 128),
    '2m': (2, 128),
    '7m': (4, 256),
    '38m': (8, 512),
    '243m': (16, 1024),
}
EPS = torch.finfo(torch.float32).eps
POLAR_MARGIN = 1.1

@dataclass
class Recipe:
    """Optimizer settings matching PR #20 / Noise record."""
    lr: float = 0.003
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.3
    anvil_lr: float = 0.023
    anvil_wd: float = 2.25

@dataclass
class Settings:
    sigma: float = 0.005
    probes: int = 0
    fc_k: int = 0
    fc_sigma: float = 0.05
    local: dict = field(default_factory=lambda: dict(q=64, k=320, v=512, gate=64))
    nrep: int = 64
    emb_chunk: int = 16
    scalar_step: float = 0.01
    chunk: int = 2 ** 25
    boost_exact_keys: bool = True
    chain_sigma: float = 0.01
    route_k: int = 0
    chain_tokens: int = 64
    cut_exact_keys: bool = True
    key_sigma: float = 0.003
    key_chunk: int = 8
    # FastRelay specific
    fast_mlp_mode: str = 'all'      # 'all' or 'routed'
    antithetic_q: bool = False
    q_sigma: float = 0.1
    # Cycle-direction options (off by default => bit-identical to the plain recipe), forward activations only:
    whiten: bool = False            # moves along d * h_j (d = per-channel RMS of the site's stream), result / d^2
    whiten_ema: float = 0.9
    whiten_clip: float = 4.0        # d is clipped to [1/clip, clip] and renormalized to mean(d^2) = 1
    subspace_k: int = 0             # >0: moves span the top-k principal directions of the site's activation covariance
    subspace_ema: float = 0.9       # (k must be a power of two <= width; the result is the credit projected on them)
    # No-chain mode (Noise README rule: a stepped signal may not step again). Every block gets the head's measured error
    # carried by the identity path only (linear scalings); attention is fed that signal, not the MLP-stepped one, and the
    # block-input norm step is dropped, so no signal takes two nonlinear steps.
    no_chain: bool = False
    # With no_chain: each block's measured credit also reaches its input stream through its values (linear: attn c_proj,
    # A^T, W_v), takes one RMS-norm step, and the identity path carries it to the writers and values below. It feeds only
    # their weight gradients, never another cycle, so no signal takes two nonlinear steps (Noise PR #23's value path).
    value_path: bool = False
    # With no_chain: skip the MLP norm/relu^2 cycle; the attention output projection is scored with the identity-path credit
    # (as in Noise #16 / #24). Cheaper, and it drops the one extra step the cycle gives that gradient.
    skip_mid: bool = False

HADAMARDS = {}

def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d, entries +-1."""
    key = (d, device, dtype)
    if key not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[key] = H.to(dtype)
    return HADAMARDS[key]

def coordinate_differences(raw, y, sigma):
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)

def norm_difference(x, delta, W):
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / W + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / W + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)

def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)

def head_scores(out, baseline, hub):
    change = out.float() - baseline.float().unsqueeze(0)
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)

def code_estimate(credited, a):
    return (credited.permute(3, 0, 2, 1).unsqueeze(-1) * a.float()).sum(0)

def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor

# ------------------------------------------------------------------------------------------------------------ Capture

class Capture:
    def __init__(self, model, seed=0, rank=0, world=1):
        self.model, self.world, self.rank = model, world, rank
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
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
            with torch.autocast('cuda', dtype=self.dtype):
                self.clean_loss = self.model(x, y, loss_reduction='none').view_as(y).float()
        finally:
            self.capturing = False
        with torch.autocast('cuda', dtype=self.dtype):
            self.x0 = norm(self.outputs['transformer.wte'])

    def head_error(self, sigma=0.1):
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

@maybe_compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, W):
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / W + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / W + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)

class Boost:
    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        self.model, self.s, self.rank, self.world = model, settings, rank, world
        self.split, self.part, self.group = split, rank % split, group
        sequence = rank // split
        self.capture = Capture(model, seed + 1000 * sequence, self.part, split)
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.hidden = 4 * self.width
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * sequence)

    def close(self):
        self.capture.close()

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def share(self, total):
        per = total // self.split
        return self.part * per, (self.part + 1) * per

    def group_sum(self, tensor):
        if self.split > 1:
            torch.distributed.all_reduce(tensor, group=self.group)
        return tensor

    def probe(self, x, y, mix=None):
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        steps = self.s.sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        credit = torch.zeros(n, self.width, device=self.device)
        lo, hi = self.share(self.width)
        C_sub, stream_dirs_sub, dir_sq_sub, Hs = C[lo:hi].contiguous(), stream_dirs[:, lo:hi], dir_sq[lo:hi], H[lo:hi]
        per = max(1, self.s.chunk // ((hi - lo) * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C_sub, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs_sub, dir_sq_sub, self.width)
            credit[t0:t0 + per] = derivative @ Hs / self.width
        return self.group_sum(credit)

    def query_key_errors(self, kind, draws, targets, layers=None):
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = getattr(self.s, 'q_sigma', 0.1)
        n = chunk = min(self.s.nrep, draws)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_' + kind
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q_raw), processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = attention(qp.to(dt), kp.to(dt), values.to(dt), window)
            rep = lambda z_: z_.to(dt).unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            fixed, vals = rep(kp if kind == 'q' else qp), rep(values)
            raw = q_raw if kind == 'q' else k_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            use_anti = getattr(self.s, 'antithetic_q', False) and kind == 'q'
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                if use_anti:
                    moved_pos = moved_heads(raw, a, sigma, cos, sin)
                    moved_neg = moved_heads(raw, a, -sigma, cos, sin)
                    out_pos = attention(moved_pos, fixed, vals, window).view(n, b, t, h, hd)
                    out_neg = attention(moved_neg, fixed, vals, window).view(n, b, t, h, hd)
                    change = out_pos.float() - out_neg.float()
                    score = (change * target.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)
                    estimate += code_estimate(score, a)
                else:
                    moved = moved_heads(raw, a, sigma, cos, sin)
                    qa, ka = (moved, fixed) if kind == 'q' else (fixed, moved)
                    out = attention(qa, ka, vals, window).view(n, b, t, h, hd)
                    score = head_scores(out, baseline, target)
                    estimate += code_estimate(score, a)
            divisor = (draws * 2 * sigma) if use_anti else (draws * sigma)
            errors[site] = (estimate / divisor).reshape(b, t, cfg.n_embd)
        return errors

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
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()

# ------------------------------------------------------------------------------------------------------------ CutBI

class CutBI(Boost):
    def exact_values(self, i, target):
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        q_raw, k_raw = d.outputs[f'transformer.h.{i}.attn.c_q'].float(), d.outputs[f'transformer.h.{i}.attn.c_k'].float()
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()
        q = norm(apply_rotary_emb(q_raw.view(b, t, h, hd), cos, sin))
        k = norm(apply_rotary_emb(k_raw.view(b, t, h, hd), cos, sin))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / math.sqrt(hd)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        mask = (lag >= 0) & (lag <= self.model.window_sizes[i][0])
        A = torch.softmax(logits.masked_fill(~mask, float('-inf')), -1)
        credit = (A.float().transpose(-1, -2) @ target.view(b, t, h, hd).transpose(1, 2)).transpose(1, 2)
        prefix = f'transformer.h.{i}.attn.'
        errors = {prefix + 'c_v': credit.reshape(b, t, cfg.n_embd)}
        if has_ve(i, cfg.n_layer):
            z = d.outputs[prefix + 've_gate'].float()
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve * credit).sum(-1)
        return errors

    def layer_internals(self, i, target, gradients):
        d, s = self.capture, self.s
        b, t = d.batch, d.length
        targets = {i: target}
        errors = self.query_key_errors('q', s.local['q'], targets, layers=[i])
        errors.update(self.exact_key_errors(targets, layers=[i]))
        errors.update(self.exact_values(i, target))
        u = torch.zeros(b * t, self.width, device=self.device)
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            w = d.modules[site].weight.float()
            u[:, :w.shape[1]] += error.reshape(b * t, -1) @ w
        prefix = f'transformer.h.{i}.attn.'
        if has_ve(i, self.model.config.n_layer):
            gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'].float())
            ve_error = (errors[prefix + 'c_v'].view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
            gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
        return u

# ------------------------------------------------------------------------------------------------------------ PR #20 Baseline Relay

@maybe_compile(dynamic=False)
def orig_mlp_cycle(move, mid, idx, hs, d_out, w_fc, w_out_t, scale, W):
    bf = torch.bfloat16
    dz = norm_difference(mid, move, W)
    dh = torch.bmm(dz.to(bf), w_fc[idx].to(bf).transpose(1, 2)).float()
    h = hs.unsqueeze(1)
    up, down = F.relu(h + dh), F.relu(h)
    dout = move + torch.bmm(((up - down) * (up + down)).to(bf), w_out_t[idx].to(bf)).float()
    return (dout * d_out.unsqueeze(1)).sum(-1) / scale

@maybe_compile(dynamic=False)
def orig_norm_cycle(move, x, target, scale, W):
    return (norm_difference(x, move, W) * target.unsqueeze(1)).sum(-1) / scale

@maybe_compile(dynamic=False)
def orig_hidden_all(hpre, steps, g_units):
    up, down = F.relu(hpre + steps).square(), F.relu(hpre - steps).square()
    return (up - down) / (2 * steps) * g_units


@maybe_compile(dynamic=False)
def key_scores(qp, kp, k_raw, steps, cos, sin, A, C, basis):
    d = k_raw.shape[-1]
    moved = k_raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    x1, x2 = moved[..., :d // 2], moved[..., d // 2:]
    moved = torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], -1)
    dk = moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)
    dl = torch.einsum('hsd,hjtd->hjst', qp, dk) / d ** 0.5
    change = A[:, None] * torch.expm1(dl)
    return (change / (1 + change) * C[:, None]).sum(2).transpose(1, 2) / steps

class Relay(CutBI):
    """PR #20 Reference Baseline Relay."""
    tokens = 256
    key_block = 512

    def cycle(self, fn, x, per_token, shared=()):
        H = self.basis()
        W = self.width
        steps = self.s.chain_sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        credit = torch.empty_like(x)
        for t0 in range(0, len(x), self.tokens):
            sl = slice(t0, t0 + self.tokens)
            move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
            credit[sl] = fn(move, x[sl], *[a[sl] for a in per_token], *shared, steps[sl].expand(-1, W), W) @ H / W
        return credit

    def input_credit(self, xin, u):
        return self.cycle(orig_norm_cycle, xin, [u])

    def hidden_credit(self, u, g, out_weight):
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return orig_hidden_all(u, steps, g @ out_weight.float())

    def exact_key_errors(self, targets, layers=None):
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        B = self.key_block if t % self.key_block == 0 and t > self.key_block else t
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd // 2] * cos + z[..., hd // 2:] * sin,
                                   z[..., :hd // 2] * -sin + z[..., hd // 2:] * cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        eye = torch.eye(hd, device=self.device)
        errors, self.key_credit = {}, {}
        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            window = model.window_sizes[i][0]
            allowed = ((lag >= 0) & (lag <= window)).float()
            pairs = [(q0, k0) for q0 in range(0, t, B) for k0 in range(0, t, B)
                     if q0 + B - 1 >= k0 and q0 - (k0 + B - 1) <= window]
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                heads = lambda name: d.outputs[prefix + name][r].float().view(t, h, hd).transpose(0, 1)
                q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
                if has_ve(i, L):
                    ve = d.outputs[f'value_embeds.{i}'][r].float().view(t, h, hd).transpose(0, 1)
                    gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][r].float().view(t, h)).T.unsqueeze(-1)
                    values = values + gate * ve
                qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
                logits = (qp @ kp.transpose(1, 2)) / hd ** 0.5
                A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
                hub = targets[i][r].view(t, h, hd).transpose(0, 1)
                C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
                steps = s.key_sigma * k_raw.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                for j0 in range(0, hd, s.key_chunk):
                    j1 = min(j0 + s.key_chunk, hd)
                    for q0, k0 in pairs:
                        qs, ks = slice(q0, q0 + B), slice(k0, k0 + B)
                        credit_all[r, :, ks, j0:j1] += key_scores(
                            qp[:, qs], kp[:, ks], k_raw[:, ks], steps[:, ks], cos[ks], sin[ks], A[:, qs, ks],
                            C[:, qs, ks], eye[j0:j1])
            self.key_credit[i] = credit_all
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        xins, mids, outs, routes = [], [], [], []
        W = self.width
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i].float().flatten(0, 1)
            mid = xin + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            route_k = s.route_k or (self.hidden // 4)
            top = (r * mlp.c_proj.weight.float().norm(dim=0)).topk(route_k, dim=-1).indices
            xins.append(xin), mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        w_fc = [blk.mlp.c_fc.weight.float() for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.float().T.contiguous() for blk in model.transformer.h]
        d_in = [None] * L
        scalars = torch.zeros(2, L, device=self.device)
        for i in reversed(range(L)):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_out = g_top if i == L - 1 else lam_r[i + 1] * d_in[i + 1]
            d_mid = self.cycle(orig_mlp_cycle, mids[i], [routes[i][0], routes[i][1], d_out], [w_fc[i], w_out_t[i]])
            gradients[p + 'attn.c_proj.weight'] = d_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            u = self.layer_internals(i, (d_mid @ block.attn.c_proj.weight.float()).view(b, t, W), gradients)
            d_in[i] = d_mid + self.input_credit(xins[i], u)
            gradients[p + 'mlp.c_proj.weight'] = d_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            hidden = self.hidden_credit(d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1), d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            prev = x0 if i == 0 else outs[i - 1]
            scalars[:, i] = torch.stack([(d_in[i] * prev).sum(), (d_in[i] * x0).sum()])
        d_x0 = lam_r[0] * d_in[0] + sum(lam_0[i] * d_in[i] for i in range(L))
        emb = self.cycle(orig_norm_cycle, e, [d_x0])
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, W))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

# ------------------------------------------------------------------------------------------------------------ Breakthrough 1: Fast Norm Cycle

@maybe_compile(dynamic=False)
def fast_norm_cycle(x, u, H, chain_sigma):
    """Zero 3D tensor 2D reduction of norm_cycle."""
    W = x.shape[-1]
    sq = x.square().sum(-1, keepdim=True)
    steps = chain_sigma * (sq / W).sqrt().clamp(min=1e-6)
    inv0 = torch.rsqrt(sq / W + EPS)
    x_H = x @ H.T
    u_H = u @ H.T
    x_u = (x * u).sum(-1, keepdim=True)
    inv = torch.rsqrt((sq + 2.0 * steps * x_H + W * steps.square()) / W + EPS)
    scores = u_H * inv + x_u * ((inv - inv0) / steps)
    return (scores @ H) / W

# ------------------------------------------------------------------------------------------------------------ Breakthrough 2: Fast Reassociated MLP Cycle

@maybe_compile(dynamic=False)
def fast_mlp_chunk_all(mid_sl, h_sl, g_sl, d_out_H_sl, H, c_fc_s, chain_sigma, W):
    bf = torch.bfloat16
    sq = mid_sl.square().sum(-1, keepdim=True)
    steps = chain_sigma * (sq / W).sqrt().clamp(min=1e-6)
    inv0 = torch.rsqrt(sq / W + EPS)
    mid_H = mid_sl @ H.T
    inv = torch.rsqrt((sq + 2.0 * steps * mid_H + W * steps.square()) / W + EPS)
    
    delta_ratio = ((inv - inv0) / inv0).unsqueeze(-1)
    alpha = (steps * inv).unsqueeze(-1)
    h = h_sl.unsqueeze(1)
    dh = delta_ratio * h + alpha * c_fc_s
    
    up, down = F.relu(h + dh), F.relu(h)
    delta_r = ((up - down) * (up + down)).to(bf)
    
    mlp_out = torch.bmm(delta_r, g_sl.unsqueeze(-1).to(bf)).squeeze(-1).float()
    id_out = steps * d_out_H_sl
    scores = (id_out + mlp_out) / steps
    return (scores @ H) / W

def fast_mlp_cycle_all(mid, hpre, d_out, H, c_fc_H, g_units, chain_sigma, chunk_m=256):
    """Option 2A: All Units Fused 2D Reduction with Zero Gathers."""
    W = mid.shape[-1]
    N = mid.shape[0]
    out_credit = torch.empty_like(mid)
    d_out_H_all = d_out @ H.T
    c_fc_s = c_fc_H.unsqueeze(0)
    
    for t0 in range(0, N, chunk_m):
        sl = slice(t0, min(t0 + chunk_m, N))
        out_credit[sl] = fast_mlp_chunk_all(mid[sl], hpre[sl], g_units[sl], d_out_H_all[sl], H, c_fc_s, chain_sigma, W)
    return out_credit

@maybe_compile(dynamic=False)
def fast_mlp_chunk_routed(mid_sl, top_sl, hs_sl, g_sl, d_out_H_sl, H, c_fc_H_T, chain_sigma, W):
    bf = torch.bfloat16
    sq = mid_sl.square().sum(-1, keepdim=True)
    steps = chain_sigma * (sq / W).sqrt().clamp(min=1e-6)
    inv0 = torch.rsqrt(sq / W + EPS)
    mid_H = mid_sl @ H.T
    inv = torch.rsqrt((sq + 2.0 * steps * mid_H + W * steps.square()) / W + EPS)
    
    delta_ratio = ((inv - inv0) / inv0).unsqueeze(-1)
    alpha = (steps * inv).unsqueeze(-1)
    h = hs_sl.unsqueeze(1)
    c_fc_sub = c_fc_H_T[top_sl].transpose(1, 2).float()
    dh = delta_ratio * h + alpha * c_fc_sub
    
    up, down = F.relu(h + dh), F.relu(h)
    delta_r = ((up - down) * (up + down)).to(bf)
    
    g_sub = g_sl.gather(1, top_sl).unsqueeze(-1).to(bf)
    mlp_out = torch.bmm(delta_r, g_sub).squeeze(-1).float()
    id_out = steps * d_out_H_sl
    scores = (id_out + mlp_out) / steps
    return (scores @ H) / W

def fast_mlp_cycle_routed(mid, top, hs, d_out, H, c_fc_H, g_units, chain_sigma, chunk_m=256):
    """Option 2B: Routed Units with Reassociated g_units."""
    W = mid.shape[-1]
    bf = torch.bfloat16
    N = mid.shape[0]
    out_credit = torch.empty_like(mid)
    d_out_H_all = d_out @ H.T
    c_fc_H_T = c_fc_H.T.to(bf)
    
    for t0 in range(0, N, chunk_m):
        sl = slice(t0, min(t0 + chunk_m, N))
        out_credit[sl] = fast_mlp_chunk_routed(mid[sl], top[sl], hs[sl], g_units[sl], d_out_H_all[sl], H, c_fc_H_T, chain_sigma, W)
    return out_credit

@maybe_compile(dynamic=False)
def fast_qkv_layer_kernel(q_raw, k_raw, values, hub, cos_t, sin_t, allowed, key_sigma, hub_extra=None):
    """Breakthrough 3 & 4: Fused 3D zero-4D-allocation single-coordinate finite-difference Q, K, and V kernel."""
    hd = k_raw.shape[-1]
    d2 = hd // 2
    scale_d = 1.0 / math.sqrt(hd)

    q_rot = torch.cat([q_raw[..., :d2] * cos_t + q_raw[..., d2:] * sin_t,
                       q_raw[..., :d2] * -sin_t + q_raw[..., d2:] * cos_t], -1)
    k_rot = torch.cat([k_raw[..., :d2] * cos_t + k_raw[..., d2:] * sin_t,
                       k_raw[..., :d2] * -sin_t + k_raw[..., d2:] * cos_t], -1)
    qp = q_rot * torch.rsqrt(q_rot.square().mean(-1, keepdim=True) + EPS)
    kp = k_rot * torch.rsqrt(k_rot.square().mean(-1, keepdim=True) + EPS)

    logits = (qp @ kp.transpose(1, 2)) * scale_d
    A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
    AC = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed * A

    # 1. Exact V credit: A^T @ hub
    credit_v = A.transpose(1, 2) @ hub
    credit_v_all = credit_v if hub_extra is None else credit_v + A.transpose(1, 2) @ hub_extra  # + carried value-path credit

    # 2. Key FD credit: G_k = (AC^T @ qp) / sqrt(hd), inverse RoPE + 3D RMSNorm FD
    G_k = (AC.transpose(1, 2) @ qp) * scale_d
    U_k = torch.cat([G_k[..., :d2] * cos_t - G_k[..., d2:] * sin_t,
                     G_k[..., :d2] * sin_t + G_k[..., d2:] * cos_t], -1)
    sq_k = k_raw.square().sum(-1, keepdim=True)
    steps_k = key_sigma * (sq_k / hd).sqrt().clamp(min=1e-6)
    inv0_k = torch.rsqrt(sq_k / hd + EPS)
    inv_k = torch.rsqrt((sq_k + 2.0 * steps_k * k_raw + steps_k.square()) / hd + EPS)
    credit_k = U_k * inv_k + (k_raw * U_k).sum(-1, keepdim=True) * ((inv_k - inv0_k) / steps_k)

    # 3. Query FD credit: G_q = (AC @ kp) / sqrt(hd), inverse RoPE + 3D RMSNorm FD
    G_q = (AC @ kp) * scale_d
    U_q = torch.cat([G_q[..., :d2] * cos_t - G_q[..., d2:] * sin_t,
                     G_q[..., :d2] * sin_t + G_q[..., d2:] * cos_t], -1)
    sq_q = q_raw.square().sum(-1, keepdim=True)
    steps_q = key_sigma * (sq_q / hd).sqrt().clamp(min=1e-6)
    inv0_q = torch.rsqrt(sq_q / hd + EPS)
    inv_q = torch.rsqrt((sq_q + 2.0 * steps_q * q_raw + steps_q.square()) / hd + EPS)
    credit_q = U_q * inv_q + (q_raw * U_q).sum(-1, keepdim=True) * ((inv_q - inv0_q) / steps_q)

    if hub_extra is None:
        return credit_q, credit_k, credit_v
    return credit_q, credit_k, credit_v, credit_v_all


# ------------------------------------------------------------------------------------------------------------ FastRelay

class FastRelay(Relay):
    """Fast-Relay: 4 Algorithmic & Systems Breakthroughs on Depth-Wise Backward Induction."""

    def whiten_scale(self, site, x):
        """Per-channel move scale d (mean(d^2) = 1) from the site's running RMS; computed from forward activations."""
        s, st = self.s, self.sites.setdefault(site, {})
        rms = x.float().square().mean(0).sqrt().clamp(min=1e-6)
        st['rms'] = rms if 'rms' not in st else s.whiten_ema * st['rms'] + (1 - s.whiten_ema) * rms
        d = st['rms'] / st['rms'].square().mean().sqrt()
        d = d.clamp(1 / s.whiten_clip, s.whiten_clip)
        return d / d.square().mean().sqrt()

    def subspace_rows(self, site, x):
        """k move directions (rows of squared norm W) spanning the site's top-k activation principal directions."""
        s, W, k = self.s, self.width, self.s.subspace_k
        st = self.sites.setdefault(site, {})
        xf = x.float()
        cov = xf.T @ xf / xf.shape[0]
        st['cov'] = cov if 'cov' not in st else s.subspace_ema * st['cov'] + (1 - s.subspace_ema) * cov
        if 'V' not in st:
            V = torch.linalg.eigh(st['cov'])[1][:, -k:]
        else:  # warm-started orthogonal power iteration
            V = st['V']
            for _ in range(2):
                V = torch.linalg.qr(st['cov'] @ V)[0]
        st['V'] = V
        mix = hadamard(k, self.device, torch.float32) * self.signs(k) / math.sqrt(k)  # rotation inside the subspace
        return math.sqrt(W) * (mix @ V.T)

    def site_basis(self, site, x):
        """(move directions for this GPU's share of the cycle, divisor for the cycle's credit or None)."""
        s = self.s
        if not hasattr(self, 'sites'):
            self.sites = {}
        if s.subspace_k:
            rows = self.subspace_rows(site, x)
            lo, hi = self.share(rows.shape[0])
            return rows[lo:hi], None
        lo, hi = self.share(self.width)
        H = self.basis()
        if s.whiten:
            d = self.whiten_scale(site, x)
            return (H * d)[lo:hi], d.square()
        return H[lo:hi], None

    def input_credit(self, xin, u, site='in'):
        H, post = self.site_basis(site, xin)
        credit = self.group_sum(fast_norm_cycle(xin, u, H, self.s.chain_sigma))
        return credit if post is None else credit / post

    def exact_key_errors(self, targets, layers=None):
        """Breakthrough 3: Fused 3D zero-4D-allocation single-key finite-difference kernel."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos_t = model.cos[:, :t].float().reshape(t, hd // 2)
        sin_t = model.sin[:, :t].float().reshape(t, hd // 2)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        errors, self.key_credit = {}, {}

        for i in range(L) if layers is None else layers:
            prefix = f'transformer.h.{i}.attn.'
            window = model.window_sizes[i][0]
            allowed = ((lag >= 0) & (lag <= window)).float()
            credit_all = torch.zeros(b, h, t, hd, device=self.device)
            for r in range(b):
                q_raw = d.outputs[prefix + 'c_q'][r].float().view(t, h, hd).transpose(0, 1)
                k_raw = d.outputs[prefix + 'c_k'][r].float().view(t, h, hd).transpose(0, 1)
                values = d.outputs[prefix + 'c_v'][r].float().view(t, h, hd).transpose(0, 1)
                if has_ve(i, L):
                    ve = d.outputs[f'value_embeds.{i}'][r].float().view(t, h, hd).transpose(0, 1)
                    gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][r].float().view(t, h)).T.unsqueeze(-1)
                    values = values + gate * ve
                hub = targets[i][r].view(t, h, hd).transpose(0, 1)
                _, ck, _ = fast_qkv_layer_kernel(q_raw, k_raw, values, hub, cos_t, sin_t, allowed, s.key_sigma)
                credit_all[r] = ck
            self.key_credit[i] = credit_all
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def layer_internals(self, i, target, gradients, extra=None):
        """Breakthrough 3 & 4: Unified 3D single-coordinate finite-difference Q, K, and V layer step."""
        if getattr(self.s, 'fast_q_mode', 'fd_3d') == 'sampled':
            return super().layer_internals(i, target, gradients)
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos_t = model.cos[:, :t].float().reshape(t, hd // 2)
        sin_t = model.sin[:, :t].float().reshape(t, hd // 2)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        prefix = f'transformer.h.{i}.attn.'
        window = model.window_sizes[i][0]
        allowed = ((lag >= 0) & (lag <= window)).float()

        cq_all = torch.zeros(b, h, t, hd, device=self.device)
        ck_all = torch.zeros(b, h, t, hd, device=self.device)
        cv_all = torch.zeros(b, h, t, hd, device=self.device)
        cv_tot = cv_all if extra is None else torch.zeros_like(cv_all)   # gradient value credit incl. carried credit
        for r in range(b):
            q_raw = d.outputs[prefix + 'c_q'][r].float().view(t, h, hd).transpose(0, 1)
            k_raw = d.outputs[prefix + 'c_k'][r].float().view(t, h, hd).transpose(0, 1)
            values = d.outputs[prefix + 'c_v'][r].float().view(t, h, hd).transpose(0, 1)
            if has_ve(i, L):
                ve = d.outputs[f'value_embeds.{i}'][r].float().view(t, h, hd).transpose(0, 1)
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][r].float().view(t, h)).T.unsqueeze(-1)
                values = values + gate * ve
            hub = target[r].view(t, h, hd).transpose(0, 1)
            if extra is None:
                cq, ck, cv = fast_qkv_layer_kernel(q_raw, k_raw, values, hub, cos_t, sin_t, allowed, s.key_sigma)
            else:
                hub_x = extra[r].view(t, h, hd).transpose(0, 1)
                cq, ck, cv, cv_tot[r] = fast_qkv_layer_kernel(q_raw, k_raw, values, hub, cos_t, sin_t, allowed, s.key_sigma, hub_x)
            cq_all[r], ck_all[r], cv_all[r] = cq, ck, cv

        errors = {
            prefix + 'c_q': cq_all.permute(0, 2, 1, 3).reshape(b, t, h * hd),
            prefix + 'c_k': ck_all.permute(0, 2, 1, 3).reshape(b, t, h * hd),
            prefix + 'c_v': cv_all.permute(0, 2, 1, 3).reshape(b, t, h * hd),
        }
        if has_ve(i, L):
            z = d.outputs[prefix + 've_gate'].float()
            ve_full = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
            slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
            errors[prefix + 've_gate'] = slope * (ve_full * cv_all.permute(0, 2, 1, 3)).sum(-1)
        grad_errors = errors      # weight gradients also see the carried value-path credit; the propagated u does not
        if extra is not None:
            grad_errors = dict(errors)
            grad_errors[prefix + 'c_v'] = cv_tot.permute(0, 2, 1, 3).reshape(b, t, h * hd)
            if has_ve(i, L):
                grad_errors[prefix + 've_gate'] = slope * (ve_full * cv_tot.permute(0, 2, 1, 3)).sum(-1)

        u = torch.zeros(b * t, self.width, device=self.device)
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, grad_errors[site])
            w = d.modules[site].weight.float()
            u[:, :w.shape[1]] += error.reshape(b * t, -1) @ w
            if site.endswith('c_v'):
                self.value_u = error.reshape(b * t, -1) @ w      # credit at the normed block input through the values only
        if has_ve(i, L):
            gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'].float())
            ve_error = (grad_errors[prefix + 'c_v'].view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
            gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)
        return u


    @torch.no_grad()
    def estimate(self, x, y):
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        W = self.width
        lo, hi = self.share(W)
        
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        xins, mids, outs, hpres, routes = [], [], [], [], []
        
        mode = getattr(s, 'fast_mlp_mode', 'all')
        
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i].float().flatten(0, 1)
            mid = xin + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            xins.append(xin), mids.append(mid), outs.append(out), hpres.append(hpre)
            if mode == 'routed':
                route_k = s.route_k or (self.hidden // 4)
                top = (r * mlp.c_proj.weight.float().norm(dim=0)).topk(route_k, dim=-1).indices
                routes.append((top, hpre.gather(1, top)))
                
        g_top = self.probe(outs[-1], yf)
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        
        d_in = [None] * L
        scalars = torch.zeros(2, L, device=self.device)
        carry = None
        
        for i in reversed(range(L)):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_out = g_top if i == L - 1 else lam_r[i + 1] * d_in[i + 1]
            
            g_units = d_out @ block.mlp.c_proj.weight.float()
            if s.no_chain and s.skip_mid:
                d_mid = d_out
            else:
                H_mlp, post_mlp = self.site_basis(f'mid{i}', mids[i])
                c_fc_H_i = H_mlp @ block.mlp.c_fc.weight.float().T
                if mode == 'all':
                    d_mid = self.group_sum(fast_mlp_cycle_all(mids[i], hpres[i], d_out, H_mlp, c_fc_H_i, g_units, s.chain_sigma))
                else:
                    top, hs = routes[i]
                    d_mid = self.group_sum(fast_mlp_cycle_routed(mids[i], top, hs, d_out, H_mlp, c_fc_H_i, g_units, s.chain_sigma))
                if post_mlp is not None:
                    d_mid = d_mid / post_mlp

            S = None if (carry is None or i == L - 1) else lam_r[i + 1] * carry   # stepped value-path credit at out_i
            gradients[p + 'attn.c_proj.weight'] = (d_mid if S is None else d_mid + S).T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            if s.no_chain:
                extra = None if S is None else (S @ block.attn.c_proj.weight.float()).view(b, t, W)
                self.layer_internals(i, (d_out @ block.attn.c_proj.weight.float()).view(b, t, W), gradients, extra=extra)
                d_in[i] = d_out
                if s.value_path:
                    v = self.input_credit(xins[i], self.value_u, site=f'in{i}')       # one RMS-norm step
                    carry = v if S is None else S + v
            else:
                u = self.layer_internals(i, (d_mid @ block.attn.c_proj.weight.float()).view(b, t, W), gradients)
                d_in[i] = d_mid + self.input_credit(xins[i], u, site=f'in{i}')

            gradients[p + 'mlp.c_proj.weight'] = (d_out if S is None else d_out + S).T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            # Reuse g_units directly in hidden_credit!
            u_fc = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            steps = s.fc_sigma * u_fc.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            hidden = orig_hidden_all(u_fc, steps, g_units)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            
            prev = x0 if i == 0 else outs[i - 1]
            d_aug = d_in[i] if not (s.no_chain and s.value_path) else d_in[i] + carry
            scalars[:, i] = torch.stack([(d_aug * prev).sum(), (d_aug * x0).sum()])
            
        d_x0 = lam_r[0] * d_in[0] + sum(lam_0[i] * d_in[i] for i in range(L))
        H_emb, post_emb = self.site_basis('emb', e)
        emb = self.group_sum(fast_norm_cycle(e, d_x0, H_emb, s.chain_sigma))
        if post_emb is not None:
            emb = emb / post_emb
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, W))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

# ------------------------------------------------------------------------------------------------------------ Optimizer & Model Setup

MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)
RUN = (1194, 240, 50, 514)

def polar(g):
    tall = g.size(0) > g.size(1)
    x = g.T if tall else g
    x = (x / (POLAR_MARGIN * x.norm() + 1e-6)).bfloat16()
    for a, b, c in MAPS:
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).float()

class Anvil(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t = 0

    def rail_beta(self):
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
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW([{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
                             betas=recipe.betas, eps=1e-8, weight_decay=0.0)
    return Optimizers(Anvil(matrices, recipe.anvil_lr, recipe.anvil_wd, steps), adam)

def make_model(device, seed, config):
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
    return model.requires_grad_(False)

# ------------------------------------------------------------------------------------------------------------ CLI & Training Loop

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/speedrun'))
    parser.add_argument('--preset', choices=sorted(PRESETS), default='38m')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--fast-mlp-mode', choices=('all', 'routed'), default='all')
    parser.add_argument('--route-k', type=int, default=0)
    parser.add_argument('--value-path', action='store_true', help='with --no-chain: stepped value-path credit to writers and values')
    parser.add_argument('--skip-mid', action=argparse.BooleanOptionalAction, default=True,
                        help='no MLP cycle: the identity-path credit scores the attention output projection')
    parser.add_argument('--anvil-lr', type=float, default=0.023)
    parser.add_argument('--no-chain', action=argparse.BooleanOptionalAction, default=True,
                        help='identity-path credit only: no signal takes a second nonlinear step (--no-no-chain: the closed PR #20/#22 chaining)')
    parser.add_argument('--train-rows', type=int, default=4880, help='Sequences trained on (one update per BATCH).')
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
    if BATCH % world:
        raise ValueError(f'BATCH ({BATCH}) must be divisible by the number of GPUs ({world}).')
    per = BATCH // world
    mine = slice(rank * per, (rank + 1) * per)

    training = load_rows(args.data_dir / 'train.pt', DATA_ROWS)[:args.train_rows]
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))]
    seq_len = training.shape[1]
    schedule = len(training) // BATCH
    steps = min(schedule, args.steps or schedule)

    n_layer, width = PRESETS[args.preset]
    config = GPTConfig(sequence_len=seq_len, n_layer=n_layer, n_head=max(1, width // 64),
                       n_kv_head=max(1, width // 64), n_embd=width)
    hidden = 4 * width
    recipe = Recipe(anvil_lr=args.anvil_lr)
    settings = Settings(fc_k=hidden // 8, route_k=args.route_k or (hidden // 4), fast_mlp_mode=args.fast_mlp_mode,
                        no_chain=args.no_chain, value_path=args.value_path, skip_mid=args.skip_mid)

    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, batch=BATCH, steps=steps, schedule_steps=schedule, seed=args.seed,
            target=TARGET, preset=args.preset, train_rows=len(training), recipe=asdict(recipe),
            settings=asdict(settings), torch_version=torch.__version__,
        ), indent=2) + '\n')

    def init():
        model = make_model(device, args.seed, config)
        return model, make_optimizer(model, recipe, schedule), FastRelay(model, settings, args.seed, rank, world)

    def set_rate(optimizer, step):
        left = (schedule - step + 1) / schedule
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / recipe.warmdown)

    # untimed warmup on random tokens (compilation), then rebuild everything from the seed
    model, optimizer, estimator = init()
    noise = torch.randint(0, config.vocab_size, (BATCH, seq_len),
                          generator=torch.Generator().manual_seed(args.seed)).to(device)
    estimator.step(noise[mine, :-1].contiguous(), noise[mine, 1:].contiguous(), optimizer)
    estimator.close()
    del model, optimizer, estimator
    torch.cuda.empty_cache()
    model, optimizer, estimator = init()

    if rank == 0:
        print(f'speedrun: {steps}/{schedule} updates, GPUs={world}, settings={asdict(settings)}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        batch = training[(step - 1) * BATCH: step * BATCH][mine].to(device)
        set_rate(optimizer, step)
        loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, train_loss=round(loss, 4), train_seconds=round(elapsed, 2))
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
                      tokens=step * BATCH * (seq_len - 1), reached_target=val_loss <= TARGET)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
