"""torch.topk(scores, K) of bf16 rows bit for bit (values, indices and their order) for the strong keys'
window-masked scores, ~4x faster than torch.topk on (8, 2048, 2048) (0.107 vs 0.423 ms on one H100).

Each score's 16 bits in value order (-0 below +0, as torch's radix select) are packed with N - 1 - index into one
distinct int32, so K repeated row maxima give torch's selection (ties at the K-th value by ascending index). torch
then sorts the gathered selection (above the K-th value by index, then its ties by index) with ATen's 32-slot bitonic
sort (SortUtils.cuh, unstable among equal values); the kernel replays that network, so equal values come out in
torch's order too. Checked against torch.topk on window-masked scores (ties at the K boundary in ~21% of rows, inside
the top in ~87%) and on rounded scores with +-0 and -inf ties.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bitonic_stage(keys, vals, valid, lane, SIZE: tl.constexpr, STRIDE: tl.constexpr, FINAL: tl.constexpr):
    """One compare-exchange stage of ATen's small bitonic sort (SortUtils.cuh bitonicSort / bitonicSwap, descending:
    a pair swaps iff ((kA > kB and validA) or not validB) == dir; invalid entries sort to the end)."""
    is_a = (lane & STRIDE) == 0
    a = tl.where(is_a, lane, lane - STRIDE)
    b = a + STRIDE
    t = (a & (STRIDE - 1)) | ((a >> 1) & ~(STRIDE - 1))  # the thread that owns the pair
    if FINAL:
        flag = lane < 0
    else:
        flag = (t & (SIZE // 2)) != 0
    ka, kb = tl.gather(keys, a, 0), tl.gather(keys, b, 0)
    va, vb = tl.gather(vals, a, 0), tl.gather(vals, b, 0)
    oka, okb = tl.gather(valid, a, 0), tl.gather(valid, b, 0)
    swap = (((ka > kb) & (oka != 0)) | (okb == 0)) == flag
    keys = tl.where(is_a, tl.where(swap, kb, ka), tl.where(swap, ka, kb))
    vals = tl.where(is_a, tl.where(swap, vb, va), tl.where(swap, va, vb))
    valid = tl.where(is_a, tl.where(swap, okb, oka), tl.where(swap, oka, okb))
    return keys, vals, valid


@triton.jit
def _window_topk_torch_kernel(Scores, Tops, Indices, N: tl.constexpr, K: tl.constexpr, LOG_N: tl.constexpr):
    """torch.topk(scores, K) of bf16 rows bit for bit, order included: the selection by K repeated maxima of the packed
    (value, N - 1 - index) as _window_topk16_kernel, then the selected in torch's gather order (above the K-th value by
    index, then its ties by index) through ATen's 32-slot bitonic sort."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, N)
    b = tl.load(Scores + row * N + cols).to(tl.int16, bitcast=True).to(tl.int32)
    ordered = tl.where(b < 0, b ^ 0x7FFF, b)
    packed = (ordered << LOG_N) | (N - 1 - cols)
    lane = tl.arange(0, 32)
    top = tl.full([32], -2147483648, tl.int32)
    for j in tl.static_range(K):
        m = tl.max(packed, 0)
        top = tl.where(lane == j, m, top)
        packed = tl.where(packed == m, -2147483648, packed)
    o = top >> LOG_N
    col = N - 1 - (top & (N - 1))
    kth = tl.sum(tl.where(lane == K - 1, o, 0), 0)
    tie = (o == kth).to(tl.int32)
    # gather order: (tie, index) ascending; invalid slots last
    order = tl.where(lane < K, (tie << 28) | (col << 16) | (o + 32768), 1 << 30)
    order = tl.sort(order)
    col = (order >> 16) & 0xFFF
    o = (order & 0xFFFF) - 32768
    keys = tl.where(o < 0, o ^ 0x7FFF, o).to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    valid = (lane < K).to(tl.int32)
    vals = col
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 2, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 4, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 4, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 4, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 8, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 8, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 4, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 2, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 16, 1, False)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 16, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 8, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 4, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 2, True)
    keys, vals, valid = _bitonic_stage(keys, vals, valid, lane, 32, 1, True)
    out = lane < K
    tl.store(Indices + row * K + lane, vals.to(tl.int64), out)
    tl.store(Tops + row * K + lane, keys.to(tl.bfloat16), out)


@torch.library.custom_op('visit_kernel::window_topk', mutates_args=())
def window_topk(scores: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """scores.topk(count, dim=-1) (values, indices) exactly; the kernel for bf16 rows of N = 2^m <= 4096 and
    count <= 32, else torch.topk."""
    n = scores.shape[-1]
    if scores.dtype != torch.bfloat16 or n & (n - 1) or n > 4096 or count > 32:
        return scores.topk(count, dim=-1)
    rows = scores.numel() // n
    tops = torch.empty((*scores.shape[:-1], count), device=scores.device, dtype=scores.dtype)
    indices = torch.empty((*scores.shape[:-1], count), device=scores.device, dtype=torch.int64)
    _window_topk_torch_kernel[(rows,)](scores.contiguous(), tops, indices, N=n, K=count, LOG_N=n.bit_length() - 1,
      num_warps=1)
    return tops, indices


@window_topk.register_fake
def _(scores, count):
    shape = (*scores.shape[:-1], count)
    return (torch.empty(shape, device=scores.device, dtype=scores.dtype),
      torch.empty(shape, device=scores.device, dtype=torch.int64))
