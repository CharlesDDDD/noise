"""EGGROLL with Adam: weight-space evolution strategies with low-rank perturbations, the first track-1 record.

EGGROLL (Sarkar et al. 2025, https://arxiv.org/abs/2511.16652) as in the Dust paper's EGGROLL-Transformer.
Every matrix and embedding of the fixed model gets a rank-1 perturbation E = a b^T, a and b ~ N(0, I), and the
2L residual mixing scalars plain Gaussian noise. A population member is the model at theta + sigma E or
theta - sigma E (an antithetic pair), evaluated on one training sequence; both halves of a pair share it.
The z-scored pair differences weight the noise into a gradient estimate that Adam follows. The perturbed
weights are never materialized: x (W + sigma a b^T)^T = x W^T + sigma (x b) a^T, so a chunk of the population
runs as one batched forward of the shared weights plus a cheap per-member correction.

Launch: torchrun --standalone --nproc_per_node=8 train/val/eggroll.py
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.eval import evaluate
from utils.model import GPT, GPTConfig, apply_rotary_emb, attention, has_ve, norm
from utils.prepare_data import TRAIN_ROWS, load_rows

BATCH = 8            # sequences per update; one epoch is 610 updates
SEED = 0
PAIRS = 131072       # antithetic pairs per update: 262,144 members, ~1.4x the compute of Dust's 3.2x run
SIGMA = 0.002
LR, BETAS = 0.001, (0.5, 0.95)  # 0.001 is best at 10M tokens (0.003 at 1M); the rate is population-invariant
WARMDOWN = 0.3       # final fraction of the epoch over which the learning rate decays linearly to 0
CHUNK = 256          # pairs per batched forward
NOISE = 2 ** 28      # one fixed buffer of N(0, 1) values that every perturbation is sliced from


def make_model(device, seed):
    """The Dust paper's initialization, with the embeddings kept in float32 for the ES updates."""
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(GPTConfig())
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, torch.nn.Linear):
                scale = 1.5 * (3 / module.weight.shape[1]) ** 0.5
                torch.nn.init.uniform_(module.weight, -scale, scale)
    model.transformer.wte.float()
    for embedding in model.value_embeds.values():
        embedding.float()
    return model.requires_grad_(False)


def lowrank(x, weight, a, b, sign):
    """x [m, T, in] through W [out, in] + sign * a b^T, one (a [m, out], b [m, in], sign [m]) per member."""
    return x @ weight.t() + sign[:, None, None] * (x @ b.unsqueeze(-1).to(x.dtype)) * a[:, None, :].to(x.dtype)


def lowrank_embed(weight, a, b, idx, sign):
    """Embedding rows of W [V, d] + sign * a b^T for token ids idx [T]."""
    return weight[idx].unsqueeze(0) + sign[:, None, None] * a[:, idx].unsqueeze(-1) * b[:, None, :]


def population_loss(model, p, noise, sign, idx, targets):
    """Mean cross-entropy [m] of m perturbed models on one sequence. p: parameters by name; noise: name ->
    (a, b) for matrices and embeddings, or eps for the mixing scalars."""
    cfg = model.config
    heads, dim = cfg.n_head, cfg.n_embd // cfg.n_head
    m, length = sign.shape[0], idx.shape[0]
    cos, sin = model.cos[:, :length], model.sin[:, :length]
    s = sign[:, None, None]
    x = x0 = norm(lowrank_embed(p['transformer.wte.weight'], *noise['transformer.wte.weight'], idx, sign))
    for i in range(cfg.n_layer):
        pre = f'transformer.h.{i}.'
        resid = p['resid_lambdas'][i] + s * noise['resid_lambdas'][:, i, None, None]
        mix = p['x0_lambdas'][i] + s * noise['x0_lambdas'][:, i, None, None]
        x = resid * x + mix * x0
        linear = lambda name, inp: lowrank(inp, p[pre + name], *noise[pre + name], sign)
        h = norm(x)
        q = linear('attn.c_q.weight', h).view(m, length, heads, dim)
        k = linear('attn.c_k.weight', h).view(m, length, heads, dim)
        v = linear('attn.c_v.weight', h).view(m, length, heads, dim)
        if has_ve(i, cfg.n_layer):
            name = f'value_embeds.{i}.weight'
            ve = lowrank_embed(p[name], *noise[name], idx, sign).view(m, length, heads, dim)
            gate = 2 * torch.sigmoid(linear('attn.ve_gate.weight', h[..., :32]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))
        y = attention(q, k, v, model.window_sizes[i][0]).reshape(m, length, -1)
        x = x + linear('attn.c_proj.weight', y)
        x = x + linear('mlp.c_proj.weight', F.relu(linear('mlp.c_fc.weight', norm(x))).square())
    h = norm(x)
    losses = []
    for c in range(0, m, 32):  # the [members, tokens, vocab] logits in slices
        part = slice(c, c + 32)
        a, b = (t[part] for t in noise['lm_head.weight'])
        logits = lowrank(h[part], p['lm_head.weight'], a, b, sign[part])[..., :cfg.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.repeat(logits.shape[0]), reduction='none')
        losses.append(ce.view(-1, length).mean(-1))
    return torch.cat(losses)


class Eggroll:
    """Perturbation bookkeeping. A direction's noise is a set of contiguous slices of one fixed Gaussian
    buffer, at offsets drawn from the step's seed, so every rank rebuilds any direction without sending it."""

    def __init__(self, model, pairs, rank=0, world=1):
        self.model, self.pairs, self.rank, self.world = model, pairs, rank, world
        self.device = next(model.parameters()).device
        self.params = dict(model.named_parameters())
        self.buffer = torch.randn(NOISE, device=self.device, generator=torch.Generator(self.device).manual_seed(SEED))
        self.loss = torch.compile(population_loss)

    def offsets(self, step):
        g = torch.Generator(self.device).manual_seed(1_000_003 * SEED + step)
        return {name: torch.randint(0, NOISE - sum(p.shape), (self.pairs,), device=self.device, generator=g)
                for name, p in self.params.items()}

    def noise(self, offsets, directions):
        """name -> (a [n, out], b [n, in]) for matrices and embeddings, eps [n, L] for the mixing scalars."""
        out = {}
        for name, p in self.params.items():
            rows = self.buffer[offsets[name][directions, None] + torch.arange(sum(p.shape), device=self.device)]
            out[name] = rows.split(list(p.shape), dim=1) if p.dim() == 2 else rows
        return out

    @torch.no_grad()
    def step(self, x, y, optimizer, step):
        """Evaluate this rank's share of the population, gather the losses, rebuild the update everywhere."""
        pairs = self.pairs
        members = max(2 * pairs, 2 * len(x))  # every sequence hosts the same number of members
        per_seq, per_rank = members // len(x), members // self.world
        offsets = self.offsets(step)
        losses = torch.empty(per_rank, device=self.device)
        sign = torch.tensor([SIGMA, -SIGMA], device=self.device).repeat(CHUNK)
        for start in range(self.rank * per_rank, (self.rank + 1) * per_rank, 2 * CHUNK):
            directions = torch.arange(start, start + 2 * CHUNK, 2, device=self.device) % (2 * pairs) // 2
            noise = {n: tuple(t.repeat_interleave(2, 0) for t in v) if isinstance(v, tuple) else v.repeat_interleave(2, 0)
                     for n, v in self.noise(offsets, directions).items()}
            seq = start // per_seq
            with torch.autocast('cuda', dtype=torch.bfloat16):
                part = self.loss(self.model, self.params, noise, sign, x[seq], y[seq])
            losses[start - self.rank * per_rank:start - self.rank * per_rank + 2 * CHUNK] = part
        if self.world > 1:
            gathered = [torch.empty_like(losses) for _ in range(self.world)]
            torch.distributed.all_gather(gathered, losses)
            losses = torch.cat(gathered)
        fitness = losses[1::2] - losses[0::2]  # > 0 when +sigma E lowers the loss
        fitness = fitness / (fitness.std() + 1e-8)
        weight = torch.zeros(pairs, device=self.device).index_add_(
            0, torch.arange(len(fitness), device=self.device) % pairs, fitness)
        gradients = {name: torch.zeros_like(p) for name, p in self.params.items()}
        for start in range(0, pairs, CHUNK):
            directions = torch.arange(start, min(pairs, start + CHUNK), device=self.device)
            w = weight[directions]
            for name, n in self.noise(offsets, directions).items():
                if isinstance(n, tuple):
                    gradients[name] -= (w[:, None] * n[0]).t() @ n[1]
                else:
                    gradients[name] -= w @ n
        optimizer.zero_grad(set_to_none=True)
        for name, p in self.params.items():
            p.grad = gradients[name] / len(fitness)
        optimizer.step()
        return losses.mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('logs/eggroll'))
    parser.add_argument('--steps', type=int, help='Stop after this many updates (smoke tests).')
    parser.add_argument('--eval-every', type=int, default=0, help='Untimed validation every this many updates.')
    parser.add_argument('--pairs', type=int, default=PAIRS, help='Antithetic pairs per update.')
    parser.add_argument('--lr', type=float, default=LR, help='Adam learning rate.')
    args = parser.parse_args()

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    if (2 * args.pairs) % (world * 2 * CHUNK):
        parser.error('Each GPU must evaluate whole chunks of pairs.')

    # One epoch: every training sequence feeds exactly one update.
    training = load_rows(args.data_dir / 'train.pt', TRAIN_ROWS)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + 1))]
    steps = min(len(training) // BATCH, args.steps or len(training))
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(dict(
            script='train/val/eggroll.py', world=world, batch=BATCH, steps=steps, seed=SEED, pairs=args.pairs,
            sigma=SIGMA, lr=args.lr, betas=BETAS, warmdown=WARMDOWN, torch_version=torch.__version__,
        ), indent=2) + '\n')

    def init():
        model = make_model(device, SEED)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=BETAS, eps=1e-8, weight_decay=0.0)
        return model, optimizer, Eggroll(model, args.pairs, rank, world)

    def set_rate(optimizer, step):
        left = (steps - step + 1) / steps  # fraction of the epoch still ahead, this update included
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group['lr'] = group['initial_lr'] * min(1.0, left / WARMDOWN)

    # Untimed warmup on random tokens compiles the population forward. Everything is then rebuilt from the
    # seed, so the timed run starts exactly as if the warmup never happened.
    model, optimizer, es = init()
    noise = torch.randint(0, 4096, (BATCH, 2049), generator=torch.Generator().manual_seed(SEED)).to(device)
    es.step(noise[:, :-1].contiguous(), noise[:, 1:].contiguous(), optimizer, 0)
    model, optimizer, es = init()

    if rank == 0:
        print(f'eggroll: {steps} steps, {args.pairs} pairs, sigma={SIGMA}, lr={args.lr}, GPUs={world}', flush=True)
    log = (args.output / 'metrics.jsonl').open('a') if rank == 0 else None
    torch.cuda.synchronize()
    start, paused, elapsed = time.perf_counter(), 0.0, 0.0
    for step in range(1, steps + 1):
        batch = training[(step - 1) * BATCH:step * BATCH].to(device)
        set_rate(optimizer, step)
        loss = es.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer, step)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=step, population_loss=loss, train_seconds=round(elapsed, 2))
        if args.eval_every and step % args.eval_every == 0 and step < steps:
            pause = time.perf_counter()
            record['val_loss'] = evaluate(model.state_dict(), args.data_dir)
            paused += time.perf_counter() - pause
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    val_loss = evaluate(state, args.data_dir)
    if rank == 0:
        torch.save(state, args.output / 'final.pt')
        result = dict(val_loss=val_loss, train_seconds=round(elapsed, 2), steps=step,
                      tokens=step * BATCH * 2048)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
