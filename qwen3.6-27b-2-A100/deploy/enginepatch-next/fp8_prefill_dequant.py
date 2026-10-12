"""Patch 4: run big prefill chunks of FP8-Marlin layers on cuBLAS bf16.

Why. On A100 (SM80) the FP8 checkpoint is served weight-only through Marlin:
each GEMM streams 1-byte weights and dequantizes them in registers. That is
what made decode +40-61% faster (decode is bandwidth-bound, and bytes halved).
Prefill is the opposite regime: a 4096-token chunk is compute-bound, and
Marlin's in-loop dequant costs tensor-core issue slots that cuBLAS bf16 does
not pay. Measured 2026-10-11: cold prefill +38% at 8K, +18% at 64K, +5% at 128K
(attention dilutes it at long context).

What. For a chunk of at least MIN_M tokens, dequantize the layer's weight
into a bf16 scratch buffer (one pass: read 1 B, write 2 B per weight) and
call cuBLAS. Decode, verify and every CUDA-graph-captured size stay on Marlin
untouched. For gate_up (5120 -> 34816) the dequant moves ~0.5 GB, about
0.3 ms at A100 bandwidth, against ~6-10 ms of GEMM per 4096-token chunk.

Why a kernel and not a kept copy. prepare_fp8_layer_for_marlin replaces the
checkpoint weight with Marlin's tiled layout and frees the original, and the
~27 GB a second copy would cost is the KV/GDN capacity FP8 bought. So the
kernel reads Marlin's layout directly. The layout, derived from
gptq_marlin_repack.cuh (num_bits=8, no act-order perm), is:

  qweight int32 [K/16, 4N]  ->  bytes [K/16][N/64][tq 8][tr 4][w 4][half 2][pa 2][pb 2]
  each 16x64 (k x n) tile is one contiguous 1 KB block, and inside it
      k = 16*kt + 8*pb + 2*tr + pa
      n = 64*nt + 16*w + 8*half + tq

Scales are Marlin's too: marlin_permute_scales shuffles each row of 64 (group
quant) or 32 (channel quant), and fp8_fused_exponent_bias_into_scales
multiplies them by 2**120 (bf16) because Marlin's dequant drops the fp8
exponent bias. The kernel undoes both. The product fp8 * scale is exact in
fp32 (4 + 8 mantissa bits), so the bf16 weight is bit-identical to what
Marlin feeds its MMA; only the fp32 accumulation order differs.

Kill switch: SGLANG_PATCH_FP8_DEQUANT=0. Threshold: SGLANG_PATCH_FP8_DEQUANT_MIN_M.
Scratch cap (bf16 elements): SGLANG_PATCH_FP8_DEQUANT_SCRATCH, default 64M = 128 MB.
"""

import os
import sys

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("SGLANG_PATCH_FP8_DEQUANT", "1") != "0"
# Above the 2048-token prefill CUDA-graph ceiling: captured sizes never reach
# here anyway (see apply), and full 4096 chunks are where the loss is.
MIN_M = int(os.environ.get("SGLANG_PATCH_FP8_DEQUANT_MIN_M", "2049"))
SCRATCH_ELEMS = int(os.environ.get("SGLANG_PATCH_FP8_DEQUANT_SCRATCH", str(64 << 20)))

_BLOCK_N = 32   # swept on A100 2026-10-12: 32x128x4 warps ~825 GB/s, 64x128 ~760
_BLOCK_K = 128
_scratch = {}
_first_use_logged = False


@triton.jit
def _marlin_fp8_to_bf16(q_ptr, s_ptr, inv_ptr, out_ptr,
                        n_start, n_rows, N, K, tile_row_bytes, group, s_mul,
                        PERIOD: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[r, k] = W[n_start + r, k] for r < n_rows: Marlin bytes -> bf16.

    All index math is split into an n part and a k part (1-D vectors, constant
    divisors only, so shifts and masks) and added once: a per-element runtime
    or int64 division costs more than the memory traffic this kernel exists
    to move. The scale depends on n alone inside one program, because a
    program's BLOCK_K columns never straddle a scale group (layout_of checks).
    """
    r = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    rmask = r < n_rows
    mask = rmask[:, None] & (kk < K)[None, :]
    n = n_start + r

    #   n = 64*nt + 16*w + 8*half + tq      k = 16*kt + 8*pb + 2*tr + pa
    n_off = (n // 64) * 1024 + (n % 8) * 128 + ((n % 64) // 16) * 8 + ((n % 16) // 8) * 4
    kl = kk % 16
    k_off = (kk // 16) * tile_row_bytes + ((kl % 8) // 2) * 32 + (kl % 2) * 2 + kl // 8
    b = tl.load(q_ptr + (k_off[None, :] + n_off[:, None]), mask=mask, other=0).to(tl.int32)

    # e4m3fn by hand: SM80 Triton has no fp8e4nv conversion.
    sign = (b >> 7) & 1
    e = (b >> 3) & 15
    m = b & 7
    normal = ((sign << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125          # m * 2**-9
    sub = tl.where(sign == 1, -sub, sub)
    val = tl.where(e == 0, sub, normal)

    # Logical scale (row, n) sits at Marlin's shuffled flat index; N is a
    # multiple of PERIOD, so the shuffle only moves n within its PERIOD block.
    row = (tl.program_id(1) * BLOCK_K) // group
    src = row * N + (n // PERIOD) * PERIOD + tl.load(inv_ptr + n % PERIOD, mask=rmask, other=0)
    sc = tl.load(s_ptr + src, mask=rmask, other=0.0).to(tl.float32) * s_mul

    # Round to nearest even by hand, as Marlin's bf16 multiply does: Triton's
    # cast rounding is not guaranteed across backends (its CPU interpreter
    # truncates even when asked for rtne). out is the bf16 buffer as int16.
    bits = (val * sc[:, None]).to(tl.int32, bitcast=True)
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    tl.store(out_ptr + (r[:, None] * K + kk[None, :]), (bits >> 16).to(tl.int16), mask=mask)


_inv_cache = {}


def _inverse_perm(period, device):
    """Inverse of Marlin's scale shuffle, from the same table Marlin uses."""
    key = (period, device)
    if key not in _inv_cache:
        from sglang.srt.layers.quantization.marlin_utils import get_scale_perms
        group_perm, single_perm = get_scale_perms()
        perm = group_perm if period == 64 else single_perm
        assert len(perm) == period
        _inv_cache[key] = torch.argsort(torch.tensor(perm)).to(torch.int32).to(device)
    return _inv_cache[key]


def layout_of(layer):
    """(N, K, group, period, s_mul) if this layer can take the dequant path, else None."""
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dtype != torch.int32 or s.dim() != 2:
        return None
    if getattr(layer, "orig_dtype", None) != torch.bfloat16 or s.dtype != torch.bfloat16:
        return None
    N = layer.output_size_per_partition
    K = layer.input_size_per_partition
    if N % 64 or K % 16 or tuple(w.shape) != (K // 16, 4 * N) or s.shape[1] != N:
        return None
    rows = s.shape[0]
    if K % rows:
        return None
    # int32 offsets, and one scale row per program (BLOCK_K never straddles a group)
    if K * N >= 2 ** 31 or (rows > 1 and (K // rows) % _BLOCK_K):
        return None
    # marlin_permute_scales: per-group scales use the 64-wide shuffle, a single
    # row of channel scales (group == K) the 32-wide one.
    period = 64 if rows > 1 else 32
    return N, K, K // rows, period, 2.0 ** -120


def dequant(layer, n_start, n_rows, out):
    """Write W[n_start:n_start+n_rows, :] (bf16, row-major [n, K]) into out."""
    N, K, group, period, s_mul = layer._fp8dq
    grid = (triton.cdiv(n_rows, _BLOCK_N), triton.cdiv(K, _BLOCK_K))
    _marlin_fp8_to_bf16[grid](
        layer.weight.view(torch.uint8), layer.weight_scale,
        _inverse_perm(period, layer.weight.device), out.view(torch.int16),
        n_start, n_rows, N, K, (N // 64) * 1024, group, s_mul,
        PERIOD=period, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K)
    return out


def _scratch_for(device, elems):
    buf = _scratch.get(device)
    if buf is None or buf.numel() < elems:
        buf = torch.empty(elems, dtype=torch.bfloat16, device=device)
        _scratch[device] = buf
    return buf


def linear(layer, x, bias=None):
    """x @ W.T (+ bias) through dequant slabs and cuBLAS. x: [..., K] bf16."""
    N, K = layer._fp8dq[0], layer._fp8dq[1]
    x2 = x.reshape(-1, K)
    out = torch.empty((x2.shape[0], N), dtype=x.dtype, device=x.device)
    slab = max(64, min(N, SCRATCH_ELEMS // K) // 64 * 64)
    buf = _scratch_for(x.device, slab * K)
    for n0 in range(0, N, slab):
        rows = min(slab, N - n0)
        w = dequant(layer, n0, rows, buf[: rows * K].view(rows, K))
        # out[:, n0:n0+rows] is row-major with ld = N: cuBLAS writes it in place.
        torch.mm(x2, w.t(), out=out[:, n0:n0 + rows])
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], N)


def wants(method, layer, x):
    """True when this call should bypass Marlin."""
    if not (ENABLED and getattr(method, "use_marlin", False)) or x.dtype != torch.bfloat16:
        return False
    if x.numel() // x.shape[-1] < MIN_M:
        return False
    # Captured graphs and compiled regions keep upstream's op: a Python branch,
    # a Triton launch and a scratch allocation must never be baked into either.
    if torch.compiler.is_compiling() or (x.is_cuda and torch.cuda.is_current_stream_capturing()):
        return False
    if not hasattr(layer, "_fp8dq"):
        layer._fp8dq = layout_of(layer)
    return layer._fp8dq is not None


def note_first_use(x):
    global _first_use_logged
    if not _first_use_logged:
        _first_use_logged = True
        print(f"[sitecustomize] fp8 prefill dequant path active (first M={x.numel() // x.shape[-1]}, "
              f"min_m={MIN_M}, scratch={SCRATCH_ELEMS * 2 >> 20} MB)", file=sys.stderr, flush=True)
