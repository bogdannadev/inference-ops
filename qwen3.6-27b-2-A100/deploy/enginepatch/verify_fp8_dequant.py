"""CPU check of patch 4 (fp8_prefill_dequant): no GPU, Triton interpreter.

Run inside the pinned image:
  docker run --rm --network none --entrypoint python3 -e TRITON_INTERPRET=1 \
    -e PYTHONPATH=/opt/enginepatch -v "$PWD/deploy/enginepatch-next:/opt/enginepatch:ro" \
    "$IMG" /opt/enginepatch/verify_fp8_dequant.py

The layer is built by upstream's own prepare_fp8_layer_for_marlin, so the
scale shuffle and exponent bias are the real ones. Only gptq_marlin_repack (a
CUDA op) is replaced, by a line-for-line transcription of the no-perm branch
of gptq_marlin_repack.cuh. That transcription is written from the kernel's
thread indexing, independently of the closed-form inverse the Triton kernel
uses, so agreement checks the derivation. The GPU check against the real
repack and Marlin GEMM is bench_fp8_dequant.py, on a drained replica.
"""

import os

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"

import torch
import sglang.srt.layers.quantization.marlin_utils_fp8 as mu8

import fp8_prefill_dequant as dq
import sitecustomize  # noqa: F401  (installs the hooks being tested)


def repack_transcribed(b_q_weight, perm, size_k, size_n, num_bits):
    """gptq_marlin_repack_kernel<.., num_bits=8, has_perm=false>, per thread."""
    assert num_bits == 8 and perm.numel() == 0
    q = b_q_weight.contiguous().view(torch.uint8).view(size_k // 4, size_n, 4)
    elem = q.permute(0, 2, 1).reshape(size_k, size_n)       # elem[k, n], byte k%4 of int k//4
    k_tiles, n_tiles = size_k // 16, size_n // 64
    out = torch.zeros(k_tiles * n_tiles * 256 * 4, dtype=torch.uint8)
    tc_offsets = (0, 1, 8, 9)
    pack_idx = (0, 2, 1, 3)
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            base = (kt * n_tiles + nt) * 256              # out_offset, in ints
            for warp_id in range(4):
                for th_id in range(32):
                    tc_col = th_id // 4
                    tc_row = (th_id % 4) * 2
                    cur_n = warp_id * 16 + tc_col
                    vals = [0] * 8
                    for i in range(4):
                        cur_elem = tc_row + tc_offsets[i]
                        vals[i] = elem[kt * 16 + cur_elem, nt * 64 + cur_n]
                        vals[4 + i] = elem[kt * 16 + cur_elem, nt * 64 + cur_n + 8]
                    for word, src in ((0, vals[:4]), (1, vals[4:])):
                        j = base + th_id * 8 + warp_id * 2 + word
                        for i in range(4):
                            out[j * 4 + i] = src[pack_idx[i]]
    return out.view(torch.int32).view(k_tiles, size_n * 16 // 4)


mu8.gptq_marlin_repack = repack_transcribed
mu8.marlin_make_workspace = lambda device: torch.zeros(1)   # sizes itself from the GPU
torch.manual_seed(0)


def make_layer(N, K, block):
    """A layer as Fp8LinearMethod leaves it on SM80 after weight loading."""
    w = (torch.randn(N, K) * 0.05).clamp(-0.4, 0.4)
    if block:   # 128x128 block scales, the Qwen FP8 checkpoint's format
        s = torch.rand(N // 128, K // 128) * 0.02 + 0.001
        s_full = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    else:       # one per-tensor scale (ModelOpt FP8 layers)
        s = torch.tensor(0.0123)
        s_full = s.expand(N, K)
    q = (w / s_full).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = N, K
    layer.orig_dtype = torch.bfloat16
    if block:
        layer.weight = torch.nn.Parameter(q, requires_grad=False)              # (N, K)
        layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
        layer.weight_block_size = [128, 128]
        mu8.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    else:
        layer.weight = torch.nn.Parameter(q.t().contiguous(), requires_grad=False)  # (K, N)
        layer.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        mu8.prepare_fp8_layer_for_marlin(layer, size_k_first=True)
    # What Marlin multiplies: fp8 value times the bf16 scale (both exact), rounded once.
    truth = (q.float() * s_full.to(torch.bfloat16).float()).to(torch.bfloat16)
    return layer, truth


for N, K, block in ((256, 384, True), (384, 256, True), (128, 160, False)):
    layer, truth = make_layer(N, K, block)
    layer._fp8dq = dq.layout_of(layer)
    assert layer._fp8dq is not None, "layout_of rejected a layer Marlin produced"
    got = dq.dequant(layer, 0, N, torch.empty(N, K, dtype=torch.bfloat16))
    assert torch.equal(got, truth), f"dequant mismatch N={N} K={K} block={block}: " \
        f"{(got.float() - truth.float()).abs().max()}"
    # a slab that starts mid-tensor and ends mid-tile
    rows = N - 96                                            # [64, N-32): ends mid-tile
    part = dq.dequant(layer, 64, rows, torch.empty(rows, K, dtype=torch.bfloat16))
    assert torch.equal(part, truth[64:64 + rows])
    print(f"dequant  N={N:4d} K={K:4d} {'block' if block else 'tensor'}-scaled: bit-exact      OK")

# linear(): several slabs, against x @ W.T on the true weight
layer, truth = make_layer(256, 384, True)
layer._fp8dq = dq.layout_of(layer)
dq.SCRATCH_ELEMS = 64 * 384                                  # force 4 slabs of 64 rows
x = torch.randn(5, 3, 384).to(torch.bfloat16)
bias = torch.randn(256).to(torch.bfloat16)
got = dq.linear(layer, x, bias)
want = (x.float() @ truth.float().t() + bias.float())
assert got.shape == (5, 3, 256) and got.dtype == torch.bfloat16
err = (got.float() - want).abs().max().item()
assert err < 0.06, err                                        # bf16 output rounding only
print(f"linear   4 slabs + bias vs fp32 reference: max err {err:.4f}      OK")

# dispatch: only big, eager, bf16 calls on Marlin layers take the path
class M:
    use_marlin = True
big = torch.zeros(dq.MIN_M, 384, dtype=torch.bfloat16)
assert dq.wants(M(), layer, big)
assert not dq.wants(M(), layer, big[: dq.MIN_M - 1])
assert not dq.wants(M(), layer, big.float())
M.use_marlin = False
assert not dq.wants(M(), layer, big)
M.use_marlin = True
odd = torch.nn.Module()
odd.weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
assert not dq.wants(M(), odd, torch.zeros(dq.MIN_M, 4, dtype=torch.bfloat16))
print(f"dispatch M>={dq.MIN_M} bf16 Marlin only; anything else -> upstream   OK")

# the hook is installed on the real class
import sglang.srt.layers.quantization.fp8 as fp8
assert getattr(fp8.Fp8LinearMethod, "_fp8_dequant", False), "patch4 not installed"
print("patch4   installed on Fp8LinearMethod                         OK")
print("OK")
