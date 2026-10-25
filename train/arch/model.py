"""Track 3's model: the Dust GPT at 12 blocks of width 1024 (16 heads of 64), a value embedding on every block, and
hashed bigram, trigram and skip-bigram tables (2^20 rows x 128 each) summed and projected into every block's value
embedding (613.9M parameters). GPTConfig()'s defaults are this architecture: utils/eval.py builds it that way.

The other GPTConfig fields (memory slots, experts, depth banks, linear attention, parallel blocks, residual n-gram
writes, ...) are the round's experiments and are off.
"""
from dataclasses import dataclass
from functools import cache

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex = torch.compile(flex_attention)


@cache
def window_mask(window, length, device):
    return create_block_mask(
        lambda b, h, q, k: (q >= k) & (q - k <= window), None, None, length, length, device=device,
    )


_slot_bias = []  # the memory slots' logit offset: one 0-d tensor on the model's device once run.py ramps it (mem_ramp)


def slot_bias():
    """The offset on every memory slot's attention logit: 0 unless the run ramps the slots in from a low share."""
    return _slot_bias[0] if _slot_bias else 0.0


def set_slot_bias(value, device):
    if not _slot_bias:
        _slot_bias.append(torch.zeros((), device=device))
    _slot_bias[0].fill_(value)


def merge_slots(q, out, lse, slots):
    """Fold always-visible memory slots into an attention over the tokens: q, out [batch, head, token, dim], lse
    [batch, head, token] (natural log); slots = (keys, values) [kv_head, slot, dim]. Every query's softmax runs over its
    tokens and the slots together. Returns the output and the log-sum-exp over both."""
    keys, values = slots
    rep = q.shape[1] // keys.shape[0]  # query heads per KV head (probe heads grouped per head)
    if rep > 1:
        keys, values = keys.repeat_interleave(rep, 0), values.repeat_interleave(rep, 0)
    logits = q.float() @ keys.float().transpose(-1, -2) * q.shape[-1] ** -0.5 + slot_bias()
    total = torch.logaddexp(lse.float(), logits.logsumexp(-1))
    out = out.float() * (lse.float() - total).exp().unsqueeze(-1) + (logits - total.unsqueeze(-1)).exp() @ values.float()
    return out.to(q.dtype), total


def attention(q, k, v, window, slots=None):
    """Causal attention with a left window; tensors have shape [batch, token, head, dim].

    A query at position s sees keys at s - window to s. Flex attention skips the blocks outside the
    window, several times faster than a dense masked kernel on the half-window blocks. slots: (keys, values)
    [kv_head, slot, dim] every query also sees (mem_slots).
    """
    mask = window_mask(window, q.shape[1], q.device)
    if slots is None:
        return flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask).transpose(1, 2)
    out, lse = flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask, return_lse=True)
    return merge_slots(q.transpose(1, 2), out, lse, slots)[0].transpose(1, 2)


def decay_matrix(heads, length, device):
    """Linear attention's fixed causal weights D[head, t, s] = gamma_h^(t - s) / sum_{s' <= t} gamma_h^(t - s') for
    s <= t (else 0), gamma_h = 1 - 2^(-5 - h) (RetNet's decays: memories of ~32 .. 2^(4 + heads) tokens). Each row sums
    to one, a constant normalization, so the output stays linear in each of q, k and v."""
    lag = torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None, :]
    rate = torch.log1p(-torch.pow(2.0, -5.0 - torch.arange(heads, device=device, dtype=torch.float32)))
    weights = torch.exp(lag.clamp(min=0) * rate[:, None, None]) * (lag >= 0)
    return weights / weights.sum(-1, keepdim=True)


def linear_attention(q, k, v):
    """Decayed linear attention (no softmax): o_t = sum_{s <= t} D[h, t, s] (q_t . k_s / sqrt(d)) v_s over the whole
    prefix, as one masked matmul per head (T^2, fine at T = 2048); tensors have shape [batch, token, head, dim]."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)).float() * q.shape[-1] ** -0.5 * decay_matrix(q.shape[1], q.shape[2], q.device)
    return (scores.to(v.dtype) @ v).transpose(1, 2)


def linear_layers(config):
    """The blocks with linear attention: config.linear_blocks lists indices ('1,3'), or 'S' takes every short-window
    block of window_pattern (the last block is always long)."""
    spec = config.linear_blocks.strip()
    if spec.upper() == 'S':
        pattern = config.window_pattern.upper()
        return frozenset(i for i in range(config.n_layer - 1) if pattern[i % len(pattern)] == 'S')
    return frozenset(int(i) for i in spec.split(',') if i.strip())


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 12
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 1024
    # Per block, repeating: S = half-sequence window, L = full. The last block is always L.
    window_pattern: str = 'SSSL'
    # Keys, values and the value-embedding gate read the raw token embedding instead of the block input: a block's
    # change at one token then reaches no other token, so the stream's own-path credit is its whole credit.
    kv_embed: bool = False
    # Tables indexed by the 1 .. prev_embed previous tokens, added to the token embedding (n-gram input to every block).
    prev_embed: int = 0
    # Parallel blocks (GPT-J): attention and MLP read the same normed block input, and both write into its sum.
    parallel: bool = False
    # Hashed n-gram value embeddings: per order n in ngram_orders a table of ngram_rows rows of width ngram_dim,
    # indexed by a hash of the token and the n - 1 before it (or skip-grams: NGRAM_SKIPS); their sum is projected
    # per block into the value embedding (blocks with value embeddings only). 0 rows: off.
    ngram_rows: int = 1048576
    ngram_dim: int = 128
    ngram_orders: str = '23s'
    # Per-block residual n-gram embeddings: every block but the last also writes its own projection of the n-gram
    # features into the residual stream, next to its MLP (credited like mlp.c_proj, from the block output's credit).
    ngram_resid: bool = False
    # MLP experts of 4 * n_embd units each; every token uses one, chosen by a fixed hash of its id (hash layers).
    experts: int = 1
    # Hypernetwork over depth: every block matrix is a learned mix of `bank` shared matrices of its kind (0: off).
    # The blocks' own matrices hold the mix (forward and eval read them); materialize() rebuilds them.
    bank: int = 0
    # Bilinear MLP: c_fc writes two branches of 4 * n_embd units and each unit is their product (no nonlinearity).
    bilinear: bool = False
    # Persistent memory (all-attention): per block, mem_slots learned keys and values every query attends to besides
    # its causal window; slot keys have no position (no rotary) and get the keys' norm (0: off).
    mem_slots: int = 0
    # Linear attention (RetNet-style retention) in these blocks: comma-separated indices, or 'S' for the short-window
    # blocks. No softmax and no window: a fixed per-head exponential decay over the whole prefix (linear_attention).
    linear_blocks: str = ''


BANK_KINDS = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


def bank_key(kind, k=None):
    return kind.replace('.', '_') + ('' if k is None else f'_{k}')


def prev_name(k):
    """The table of the token k back (prev_embed)."""
    return 'wte_prev' if k == 1 else f'wte_prev{k}'


NGRAM_SKIPS = {'s': (2,), 't': (3,), 'u': (2, 3)}  # skip-gram codes: the token with the tokens that many back


def ngram_index(idx, code, rows, pad):
    """Row of each position's n-gram: code '2' .. '9' the token and the code - 1 before it, a skip code (NGRAM_SKIPS)
    the token and the tokens at those lags; `pad` before the start."""
    ids = idx
    for k in NGRAM_SKIPS.get(code) or range(1, int(code)):
        ids = ids * (pad + 1) + F.pad(idx[:, :-k], (k, 0), value=pad)
    h = ids * 2654435761  # wraps in int64 for long n-grams: still a fixed hash
    return (h ^ (h >> 29)).remainder(rows)


def expert_of(idx, experts):
    """The MLP expert of each token: a fixed hash of the token id."""
    h = idx * 2654435761
    return (h ^ (h >> 16)).remainder(experts)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value embeddings on every block."""
    return True


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # The value embedding is scaled per head by 2 * sigmoid(gate), read from the block input's first 32
        # channels.
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )
        if config.mem_slots:  # 3D, so the block-matrix optimizer passes them by
            self.mem_k = nn.Parameter(torch.empty(config.mem_slots, self.n_kv_head, self.head_dim))
            self.mem_v = nn.Parameter(torch.empty(config.mem_slots, self.n_kv_head, self.head_dim))
        else:
            self.mem_k = self.mem_v = None
        self.is_linear = layer_idx in linear_layers(config)

    def slots(self):
        """The memory slots as attention reads them, (keys, values) [kv_head, slot, dim] fp32; None without slots."""
        if self.mem_k is None:
            return None
        return norm(self.mem_k.float()).transpose(0, 1), self.mem_v.float().transpose(0, 1)

    def forward(self, x, ve, cos_sin, window_size, kv=None):
        B, T, C = x.size()
        kv = x if kv is None else kv
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(kv).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(kv).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(kv[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
        q, k = (norm(q), norm(k))
        y = linear_attention(q, k, v) if self.is_linear else attention(q, k, v, window_size[0], self.slots())
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts, self.bilinear = config.experts, config.bilinear
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd * config.experts * (2 if config.bilinear else 1),
                              bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd * config.experts, config.n_embd, bias=False)

    def forward(self, x, expert=None):
        if self.bilinear:
            u, w = self.c_fc(x).chunk(2, -1)
            return self.c_proj(u * w)
        a = F.relu(self.c_fc(x)).square()
        if self.experts > 1:  # only the token's expert's units fire (all are computed: the clean pass is cheap here)
            gate = F.one_hot(expert, self.experts).to(a.dtype).unsqueeze(-1)
            a = (a.unflatten(-1, (self.experts, -1)) * gate).flatten(-2)
        return self.c_proj(a)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.parallel = config.parallel

    def forward(self, x, ve, cos_sin, window_size, kv=None, expert=None):
        # Pre-norm block. attn.c_proj and mlp.c_proj are the "writers" into the residual stream.
        if self.parallel:
            h = norm(x)
            return x + self.attn(h, ve, cos_sin, window_size, kv) + self.mlp(h, expert)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv)
        x = x + self.mlp(norm(x), expert)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        for k in range(1, config.prev_embed + 1):  # the extra row stands for "no token k back"
            self.transformer[prev_name(k)] = nn.Embedding(padded_vocab + 1, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)  # untied from the embedding
        # The stream entering block i is resid_lambdas[i] * x + x0_lambdas[i] * x0, a learned mix of the
        # previous block's output and the normalized token embedding.
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        if config.ngram_rows:
            self.ngram_embeds = nn.ModuleDict({
                n: nn.Embedding(config.ngram_rows, config.ngram_dim) for n in config.ngram_orders})
            self.ngram_proj = nn.ModuleDict({
                i: nn.Linear(config.ngram_dim, kv_dim, bias=False) for i in self.value_embeds})
            if config.ngram_resid:
                self.ngram_resid = nn.ModuleDict({
                    str(i): nn.Linear(config.ngram_dim, config.n_embd, bias=False) for i in range(config.n_layer - 1)})
        if config.bank:
            first = self.transformer.h[0]
            self.bank = nn.ParameterDict({
                bank_key(kind, k): nn.Parameter(torch.empty_like(first.get_submodule(kind).weight))
                for kind in BANK_KINDS for k in range(config.bank)})
            self.bank_mix = nn.ParameterDict({
                bank_key(kind): nn.Parameter(torch.zeros(config.n_layer, config.bank)) for kind in BANK_KINDS})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        for k in range(1, self.config.prev_embed + 1):
            torch.nn.init.zeros_(self.transformer[prev_name(k)].weight)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        if self.config.ngram_rows:  # rows like the token embedding, projections from zero
            for table in self.ngram_embeds.values():
                torch.nn.init.normal_(table.weight, mean=0.0, std=1.0)
            for proj in self.ngram_proj.values():
                torch.nn.init.zeros_(proj.weight)
            for proj in self.ngram_resid.values() if self.config.ngram_resid else ():
                torch.nn.init.zeros_(proj.weight)
        # Memory slots: random keys at unit RMS, the scale the norm gives them (an Adam step moves the raw and the normed
        # key alike; equal keys would get equal credits forever), values from zero (the output starts as the baseline's,
        # scaled by 1 - the slots' softmax mass).
        if self.config.mem_slots:
            for block in self.transformer.h:
                torch.nn.init.normal_(block.attn.mem_k, mean=0.0, std=1.0)
                torch.nn.init.zeros_(block.attn.mem_v)
        if self.config.bank:
            self.tie_bank()
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        # Token and value embeddings are stored in bfloat16 on the GPU
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            for k in range(1, self.config.prev_embed + 1):
                self.transformer[prev_name(k)].to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)
            if self.config.ngram_rows:
                self.ngram_embeds.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def tie_bank(self):
        """Bank matrix k starts as block k's matrix; block l mixes bank matrix l mod bank."""
        for kind in BANK_KINDS:
            for k in range(self.config.bank):
                self.bank[bank_key(kind, k)].copy_(self.transformer.h[k].get_submodule(kind).weight)
            mix = self.bank_mix[bank_key(kind)]
            mix.zero_()
            mix[torch.arange(self.config.n_layer), torch.arange(self.config.n_layer) % self.config.bank] = 1
        self.materialize()

    @torch.no_grad()
    def materialize(self):
        for kind in BANK_KINDS:
            bank = torch.stack([self.bank[bank_key(kind, k)].float() for k in range(self.config.bank)])
            mixed = torch.einsum('lk,koi->loi', self.bank_mix[bank_key(kind)].float(), bank)
            for block, weight in zip(self.transformer.h, mixed):
                block.get_submodule(kind).weight.copy_(weight)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        e = self.transformer.wte(idx)
        for k in range(1, self.config.prev_embed + 1):
            table = self.transformer[prev_name(k)]
            e = e + table(F.pad(idx[:, :-k], (k, 0), value=table.num_embeddings - 1))
        x = norm(e)
        kv = e.float() if self.config.kv_embed else None
        x0 = x
        if self.config.ngram_rows:
            pad = self.transformer.wte.num_embeddings
            ngram = sum(table(ngram_index(idx, n, self.config.ngram_rows, pad))
                        for n, table in self.ngram_embeds.items())
        expert = expert_of(idx, self.config.experts) if self.config.experts > 1 else None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if ve is not None and self.config.ngram_rows:
                ve = ve + self.ngram_proj[str(i)](ngram)
            x = block(x, ve, cos_sin, self.window_sizes[i], kv, expert)
            if self.config.ngram_rows and self.config.ngram_resid and i < self.config.n_layer - 1:
                x = x + self.ngram_resid[str(i)](ngram)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # soft cap
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
