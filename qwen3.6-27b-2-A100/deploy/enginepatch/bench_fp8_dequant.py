"""GPU check + benchmark of patch 4, for a DRAINED replica's GPU only.

  IMG=$(docker inspect qwen36-27b-r1 --format '{{.Image}}')
  docker run --rm --gpus '"device=1"' --entrypoint python3 -e PYTHONPATH=/opt/enginepatch \
    -v "$PWD/deploy/enginepatch-next:/opt/enginepatch:ro" "$IMG" /opt/enginepatch/bench_fp8_dequant.py

Needs ~3 GB of GPU memory. Never run it next to a serving engine: it would
both steal bandwidth from live decode and read wrong numbers.

1. Layout, against the REAL gptq_marlin_repack: Marlin's own GEMM applied to
   identity rows returns columns of the weight it holds; the dequant kernel
   must reproduce them bit for bit.
2. One full GEMM per shape at M=4096: dequant+cuBLAS vs Marlin, relative error.
3. Time per Qwen3.8-27B linear at the prefill chunk sizes, Marlin vs
   dequant+cuBLAS, and the per-forward total weighted by layer count.
"""

import torch
import sglang.srt.layers.quantization.marlin_utils_fp8 as mu8

import fp8_prefill_dequant as dq

dev = torch.device("cuda")
torch.manual_seed(0)

# (name, K in, N out, layers per forward), as SGLang fuses them for Qwen3.8-27B
SHAPES = (
    ("attn qkv (q+gate,k,v)", 5120, 14336, 16),
    ("attn o_proj", 6144, 5120, 16),
    ("gdn in_proj_qkvz", 5120, 16384, 48),
    ("gdn out_proj", 6144, 5120, 48),
    ("mlp gate_up", 5120, 34816, 64),
    ("mlp down", 17408, 5120, 64),
)
MS = (2048, 4096)


def make_layer(K, N):
    w = (torch.randn(N, K, device=dev) * 0.05).clamp(-0.4, 0.4)
    s = torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.001
    q = (w / s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = N, K
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(s, requires_grad=False)
    layer.weight_block_size = [128, 128]
    mu8.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    layer._fp8dq = dq.layout_of(layer)
    assert layer._fp8dq is not None
    return layer


def marlin(layer, x):
    return mu8.apply_fp8_marlin_linear(
        x, layer.weight, layer.weight_scale, layer.workspace,
        layer.output_size_per_partition, layer.input_size_per_partition, None)


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


print(f"{torch.cuda.get_device_name()}  scratch cap {dq.SCRATCH_ELEMS * 2 >> 20} MB\n")
tot = {m: [0.0, 0.0] for m in MS}
for name, K, N, count in SHAPES:
    layer = make_layer(K, N)

    # 1. layout vs Marlin itself: eye rows pick out weight columns k0..k0+255
    for k0 in (0, K - 256):
        eye = torch.zeros(256, K, dtype=torch.bfloat16, device=dev)
        eye[torch.arange(256), k0 + torch.arange(256)] = 1
        cols = marlin(layer, eye).t().float()                       # = W[:, k0:k0+256]
        w = dq.dequant(layer, 0, N, torch.empty(N, K, dtype=torch.bfloat16, device=dev))
        mine = w[:, k0:k0 + 256].float()
        exact = (cols == mine).float().mean().item()
        # A wrong layout scrambles values entirely; a rounding-point difference
        # inside Marlin would show as at most one bf16 ulp (2**-7 relative).
        worst = ((cols - mine).abs() / mine.abs().clamp_min(1e-30)).max().item()
        assert worst <= 2 ** -7, f"{name}: layout mismatch at k0={k0} (rel {worst:.3g})"
    del w

    # 2. full GEMM agreement at a prefill chunk
    x = torch.randn(4096, K, dtype=torch.bfloat16, device=dev)
    ref, got = marlin(layer, x).float(), dq.linear(layer, x).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"{name}: rel err {rel}"

    # 3. timing
    row = []
    for m in MS:
        xm = x[:m]
        t_marlin = timed(lambda: marlin(layer, xm))
        t_dq = timed(lambda: dq.linear(layer, xm))
        tot[m][0] += t_marlin * count
        tot[m][1] += t_dq * count
        row.append(f"M={m}: marlin {t_marlin:6.3f} ms  dq+cublas {t_dq:6.3f} ms  ({t_marlin / t_dq:4.2f}x)")
    t_only = timed(lambda: dq.dequant(layer, 0, N, dq._scratch_for(dev, N * K)[: N * K].view(N, K)))
    gbs = N * K * 3 / t_only / 1e6
    print(f"{name:24s} {K:5d}->{N:5d} x{count:2d}  layout ok ({100 * exact:.1f}% bit-exact), rel err {rel:.1e}, "
          f"dequant {t_only:.3f} ms ({gbs:.0f} GB/s)")
    for line in row:
        print(f"    {line}")
    del layer, x, ref, got
    torch.cuda.empty_cache()

print("\nall FP8 linears of one forward (layer-count weighted):")
for m in MS:
    a, b = tot[m]
    print(f"  M={m}: marlin {a:7.1f} ms   dq+cublas {b:7.1f} ms   saves {a - b:6.1f} ms per chunk ({a / b:4.2f}x)")
print(f"\npeak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
