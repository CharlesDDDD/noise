"""Track 2: reach validation loss TARGET on 8xH100 in the shortest time, without backpropagation.

Identity-path credit: one Hadamard cycle measures each token's credit at the top stream, and every earlier stream
gets it through the residual scalars. Launch: torchrun --standalone --nproc_per_node=8 train/speedrun/run.py
"""
import argparse
from dataclasses import asdict, dataclass
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
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, has_ve, norm
from utils.prepare_data import TRAIN_ROWS, load_rows

BATCH = 8  # sequences per update, one per GPU; 610 updates per epoch
SEED = 0
TARGET = 4.2  # validation loss to reach; the run counts only if its final weights reach it
WIDTH = 512
EPS = torch.finfo(torch.float32).eps
POLAR_MARGIN = 1.1
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64


@dataclass
class Recipe:
    """The record's optimizer settings."""
    lr: float = 0.003       # Adam; embeddings 300x, mixing scalars 30x
    betas: tuple = (0.7, 0.95)
    warmdown: float = 0.3   # final fraction of the epoch with linear lr decay
    anvil_lr: float = 0.023
    anvil_wd: float = 2.25


@dataclass
class Settings:
    sigma: float = 0.005        # top-stream probe step, relative to the stream's RMS at that token
    fc_k: int = 256             # MLP units moved per token for the hidden credit
    fc_sigma: float = 0.05
    query_draws: int = 64       # a full cycle per head
    nrep: int = 64              # draws per batched attention rerun
    chunk: int = 2 ** 27        # elements per probe chunk (tokens x probes x vocab)
    key_sigma: float = 0.003
    key_chunk: int = 8



@torch.compile
def coordinate_differences(raw, y, sigma):
    """Finite differences of each token's loss along each of its own logits (the record's coordinate head): every
    moved loss is a full forward evaluation of the soft-capped cross-entropy, written in closed form."""
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1."""
    if (d, device, dtype) not in HADAMARDS:
        if d & (d - 1):
            raise ValueError(f'Orthogonal draws need a power-of-two width, not {d}.')
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)
    return HADAMARDS[d, device, dtype]


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq):
    """Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of
    the clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,)
    |D_j|^2. The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe."""
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / WIDTH + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / WIDTH + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):
    change = out.float() - baseline.float().unsqueeze(0)
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return (credited.permute(3, 0, 2, 1).unsqueeze(-1) * a.float()).sum(0)


@torch.compile(dynamic=False)
def key_scores(qp, k_raw, steps, cos, sin, A, C, j, qr):
    """The finite coordinate-key probe, sharing q @ rotated_key across directions."""
    d = k_raw.shape[-1]
    half = d // 2
    x1, x2 = k_raw.chunk(2, -1)
    r = torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], -1)
    clean = r.square().sum(-1, keepdim=True) + d * EPS
    jj = j % half
    rho = (cos.square() + sin.square())[:, jj].T
    kj = k_raw[:, :, j].transpose(1, 2)
    st = steps.transpose(1, 2)
    # Retain the complete finite change in the norm, including the squared step.
    eta = rho[None] * (2 * st * kj + st.square()) / clean.transpose(1, 2)
    a0 = torch.rsqrt(clean / d).transpose(1, 2)
    da = a0 * torch.expm1(-.5 * torch.log1p(eta))
    a = a0 + da
    q1 = qp[:, :, jj].transpose(1, 2)
    q2 = qp[:, :, jj + half].transpose(1, 2)
    c, s = cos[:, jj].T, sin[:, jj].T
    sparse = torch.where((j < half)[None, :, None, None],
        q1[:, :, :, None] * c[None, :, None, :] - q2[:, :, :, None] * s[None, :, None, :],
        q1[:, :, :, None] * s[None, :, None, :] + q2[:, :, :, None] * c[None, :, None, :])
    dl = (qr[:, None] * da[:, :, None, :] + st[:, :, None, :] * a[:, :, None, :] * sparse) / (d ** .5)
    # This is the full finite attention response, not a softmax derivative.
    change = A[:, None] * torch.expm1(dl)
    return ((change / (1 + change) * C[:, None]).sum(2) / st).transpose(1, 2)


class Capture:
    """The clean forward's cached activations, and the record's orthogonal draws and coordinate head."""

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
        """Weight gradient from output error and cached input."""
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)


class IdentityPath:
    """Identity-path credit: one readout cycle at the top stream per update, sent to every earlier stream along the
    residual path; everything else is one local step from those credits. One sequence per GPU."""

    def __init__(self, model, settings, seed, rank, world):
        self.model, self.s, self.world = model, settings, world
        self.capture = Capture(model, seed + 1000 * rank, 0, 1)
        self.device = self.capture.device
        self.generator = torch.Generator(self.device).manual_seed(seed + 17 + 1000 * rank)

    def close(self):
        self.capture.close()

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(WIDTH, self.device, torch.float32) * self.signs(WIDTH)

    def probe(self, x, y, mix=None, probed=None):
        """Each token's readout credit in the probed space, measured along a full cycle of the step's basis."""
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T if mix is None else mix.float() @ H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        probed = x if probed is None else probed
        steps = self.s.sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        credit = torch.zeros(n, WIDTH, device=self.device)
        per = max(1, self.s.chunk // (WIDTH * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq)
            credit[t0:t0 + per] = derivative @ H / WIDTH
        return credit

    def attention_maps(self, i):
        """Layer i's clean attention A[b, h, s, t] and the matched filter K[b, h, t, s] = A[s, t] / max_t' A[s, t']."""
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
        K = (A / A.amax(-1, keepdim=True)).transpose(-1, -2)
        self.maps[i] = (A.to(torch.bfloat16), K.to(torch.bfloat16).contiguous())
        return self.maps[i]

    def query_key_errors(self, kind, draws, targets):
        """Query or key credit by the record's local step: draws move every position's raw query (or key), the
        attention reruns, each query's output change is scored against its target; key scores reach the moved
        positions through the matched filter."""
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
            baseline = attention(qp.to(dt), kp.to(dt), values.to(dt), window)
            rep = lambda z_: z_.to(dt).unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            fixed, vals = rep(kp if kind == 'q' else qp), rep(values)
            raw = q_raw if kind == 'q' else k_raw
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // chunk):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                moved = moved_heads(raw, a, sigma, cos, sin)
                qa, ka = (moved, fixed) if kind == 'q' else (fixed, moved)
                out = attention(qa, ka, vals, window).view(n, b, t, h, hd)
                score = head_scores(out, baseline, target)
                if kind == 'k':
                    score = (K.to(dt) @ score.to(dt)).float()
                estimate += code_estimate(score, a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors

    def exact_key_errors(self, targets):
        """Key credit, one local step from the targets, by single-key experiments (key_scores): exact up to the
        one-sided step, with no interference between positions."""
        d, model, s = self.capture, self.model, self.s
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd // 2] * cos + z[..., hd // 2:] * sin,
                                   z[..., :hd // 2] * -sin + z[..., hd // 2:] * cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        lo, hi = 0, hd
        errors, self.key_credit = {}, {}
        for i in range(L):
            prefix = f'transformer.h.{i}.attn.'
            allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
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
                first, second = k_raw.chunk(2, -1)
                rotated = torch.cat([first * cos + second * sin, -first * sin + second * cos], -1)
                qr = qp @ rotated.transpose(-1, -2)
                for j0 in range(lo, hi, s.key_chunk):
                    j1 = min(j0 + s.key_chunk, hi)
                    directions = torch.arange(j0, j1, device=self.device)
                    credit_all[r, :, :, j0:j1] = key_scores(qp, k_raw, steps, cos, sin, A, C, directions, qr)
            self.key_credit[i] = credit_all
            errors[prefix + 'c_k'] = credit_all.permute(0, 2, 1, 3).reshape(b, t, h * hd)
        return errors

    def value_errors(self, kind, targets):
        """Values or value-embedding gates by single-value experiments, evaluated exactly: moving position t's value
        moves query s's output by A[s, t] times the move, so the credit is sum_s A[s, t] hub_s."""
        d, model = self.capture, self.model
        cfg = model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors = {}
        for i in range(cfg.n_layer):
            if kind == 'gate' and not has_ve(i, cfg.n_layer):
                continue
            A, _ = self.attention_maps(i)
            hub = targets[i].view(b, t, h, hd).transpose(1, 2).float()
            credit = (A.float().transpose(-1, -2) @ hub).transpose(1, 2)
            if kind == 'v':
                errors[f'transformer.h.{i}.attn.c_v'] = credit.reshape(b, t, h * hd)
            else:
                z = d.outputs[f'transformer.h.{i}.attn.ve_gate'].float()
                ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
                step = 0.1
                slope = (torch.sigmoid(z + step) - torch.sigmoid(z - step)) / step
                errors[f'transformer.h.{i}.attn.ve_gate'] = slope * (credit * ve).sum(-1)
        return errors

    def hidden_credit(self, u, g, out_weight):
        """MLP hidden credit, one local step from the measured credit g at the block output: each token's fc_k
        largest units move along a sign-flipped Hadamard cycle, through relu^2 and the output projection."""
        k = self.s.fc_k
        self.signs(k)  # preserve the removed sign draw
        top = u.topk(k, dim=-1, sorted=False).indices
        uk = u.gather(1, top)
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        slope = ((uk + steps).relu().square() - (uk - steps).relu().square()) / (2 * steps)
        projection = g.to(torch.bfloat16).float() @ out_weight.to(torch.bfloat16).float()
        selected = projection.gather(1, top) * slope
        return torch.zeros_like(u).scatter_(1, top, selected) / getattr(self, 'split', 1)

    def internals(self, gradients, targets, b, t):
        """Queries, keys, values, gates and value embeddings, one local step from the targets."""
        d, L = self.capture, self.model.config.n_layer
        self.maps = {}
        for kind in ('q', 'k', 'v', 'gate'):
            if kind == 'q':
                local = self.query_key_errors('q', self.s.query_draws, targets)
            elif kind == 'k':
                local = self.exact_key_errors(targets)
            else:
                local = self.value_errors(kind, targets)
            for site, error in local.items():
                gradients[site + '.weight'] = d.reassemble(site, error)
                if kind == 'v' and has_ve(int(site.split('.')[2]), L):
                    i = int(site.split('.')[2])
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    @torch.no_grad()
    def estimate(self, x, y):
        d, model = self.capture, self.model
        d.capture(x, y)
        L, (b, t) = model.config.n_layer, x.shape
        yf = y.flatten()
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        x0 = d.x0.float().flatten(0, 1)
        outs = []
        for i in range(L):
            p = f'transformer.h.{i}.'
            outs.append((d.blocks[i] + d.outputs[p + 'attn.c_proj'] + d.outputs[p + 'mlp.c_proj']).float().flatten(0, 1))
        top = self.probe(outs[-1], yf)
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        scale, injected = torch.ones((), device=self.device), torch.zeros((), device=self.device)
        targets, scalars = {}, torch.zeros(2, L, device=self.device)
        for i in range(L - 1, -1, -1):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            credit = scale * top
            gradients[p + 'attn.c_proj.weight'] = credit.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = credit.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, credit, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            targets[i] = (credit @ block.attn.c_proj.weight.float()).view(b, t, WIDTH)
            prev = x0 if i == 0 else outs[i - 1]
            scalars[:, i] = torch.stack([(credit * prev).sum(), (credit * x0).sum()])
            injected = injected + lam_0[i] * scale
            scale = scale * lam_r[i]
        self.internals(gradients, targets, b, t)
        embedding = ((scale + injected) * top).view(b, t, WIDTH)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', embedding)
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        for name, p in model.named_parameters():
            if name not in gradients:
                gradients[name] = torch.zeros_like(p, dtype=torch.float32)
        return gradients

    def step(self, x, y, optimizer):
        gradients = self.estimate(x, y)
        valid = (y != -1).sum()
        optimizer.zero_grad(set_to_none=True)
        params = list(self.model.named_parameters())
        flat = torch.cat([(gradients[name] / valid).float().reshape(-1) for name, _ in params])
        if self.world > 1:
            torch.distributed.all_reduce(flat)
        flat /= self.world
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError('Non-finite estimate')
        offset = 0
        for name, p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        optimizer.step()
        return average_ranks(self.capture.clean_loss[y != -1].mean().reshape(1)).item()



def make_model(device, seed, config):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():  # output projections: uniform with std 1.5 / sqrt(fan_in) instead of zeros (the record's)
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


RUN = (1194, 240, 50, 514)  # PR #360's total, warmup, cooldown and engage steps; scaled to the run length


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


def polar(g):
    """Approximate orthogonalization of a stack (B, r, c) of same-shape matrices."""
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (POLAR_MARGIN * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
                 ratio: float):
    """One ANVIL II step for a stack of same-shape matrices; updates fast, slow and energy in place."""
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar(g.lerp(weight * fast + (1 - weight) * slow, beta))
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class Anvil(torch.optim.Optimizer):
    """ANVIL II from modded-nanogpt PR #360, applied to all same-shape matrices at once."""

    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.t, self.stacks = 0, {}

    def rail_beta(self):
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    def stacked_state(self, shape, params):
        """The moving averages of one shape group as stacked tensors; each parameter's state holds views into them."""
        stack = self.stacks.get(shape)
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
        self.stacks[shape] = stack
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
                                   1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5)
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
    adam = torch.optim.AdamW([{'params': ps, 'lr': recipe.lr * scale} for scale, ps in groups.items()],
                             betas=recipe.betas, eps=1e-8, weight_decay=0.0)
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
    if BATCH % world:
        raise ValueError(f'Run on a number of GPUs that divides {BATCH}.')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    recipe, settings, config = Recipe(), Settings(), GPTConfig()

    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps = min(len(training) // BATCH, args.steps or len(training))
    per = BATCH // world
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script=__file__, world=world, batch=BATCH, steps=steps, seed=SEED, target=TARGET,
            recipe=asdict(recipe), settings=asdict(settings), torch_version=torch.__version__,
        ), indent=2) + '\n')

    def init():
        model = make_model(device, SEED, config)
        return model, make_optimizer(model, recipe, steps), IdentityPath(model, settings, SEED, rank, world)

    def set_rate(optimizer, step):
        left = (steps - step + 1) / steps
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / recipe.warmdown if recipe.warmdown else 1.0)

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
        first = (step - 1) * BATCH + rank * per
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
