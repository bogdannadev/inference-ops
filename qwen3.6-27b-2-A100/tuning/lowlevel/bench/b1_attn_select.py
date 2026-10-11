#!/usr/bin/env python3
"""B1 -- prefill attention SELECTION on SM80, head_dim 256 (P1_PROFILE.md, Stage B step 1).

P1 found the full-attention prefill kernel (FlashInfer fa2, paged, page 64) at
~41% of bf16 tensor peak, taking 48-60% of long-context prefill. Before anyone
writes a kernel: does something already in the pinned image do better?

One production-shaped call per layer: 4,096 new queries (one prefill chunk)
attending causally (bottom-right aligned) to a prefix of P tokens plus
themselves. Qwen3.8-27B full-attention geometry: 24 Q heads, 4 KV heads,
head_dim 256, bf16.

Variants:
  fi_paged64        FlashInfer paged fa2, page 64, pages in order    <- production kernel
  fi_paged64_shuf   same, pages shuffled (production pages are scattered)
  fi_ragged         FlashInfer ragged fa2, contiguous K/V (SGLang never picks it:
                    the model is multimodal, so use_ragged is forced off)
  sdpa_flash        PyTorch SDPA flash backend = Tri Dao FA2 bundled in torch
  triton_*          SGLang's Triton extend kernel: production config (64x64,
                    8 warps, 1 stage) plus a small tile/stage sweep

Correctness: every variant is compared against an fp32 reference on the last
128 query rows of 6 heads (one KV group), which is the part with the most keys.

Run in a side container on the DRAINED replica's GPU (~2 GB free), e.g.:
  docker run --rm --gpus '"device=1"' -v $PWD/tuning/lowlevel:/ll \
    --entrypoint python3 <pinned image> /ll/bench/b1_attn_select.py \
    --json /ll/results/b1_attn_select.json
"""
import argparse, json, math, sys, time

import torch

HQ, HKV, D, NQ, PAGE = 24, 4, 256, 4096, 64
PEAK_TFLOPS = 312.0


def causal_flops(prefix, nq=NQ):
    # QK^T and PV, 2 FLOP per MAC each; query i sees prefix + i + 1 keys.
    visible = nq * prefix + nq * (nq + 1) / 2
    return 4 * HQ * D * visible


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def reference(q, k, v, prefix, rows=128, heads=6):
    """fp32 attention for the last `rows` queries of the first `heads` Q heads,
    one head at a time: at 114K keys a batched version needs ~1.4 GB, and the
    drained GPU has ~2 GB free in total."""
    L = prefix + NQ
    ks = k[:L, 0].float(); vs = v[:L, 0].float()          # KV head 0 serves Q heads 0..5
    pos = torch.arange(L - rows, L, device=q.device)
    mask = torch.arange(L, device=q.device)[None, :] > pos[:, None]
    outs = []
    for h in range(heads):
        s = (q[-rows:, h].float() @ ks.T) / math.sqrt(D)
        s.masked_fill_(mask, float("-inf"))
        outs.append(s.softmax(-1) @ vs)
        del s
    return torch.stack(outs, 1)                          # [rows, heads, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="4096,61440,110592")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--json")
    a = ap.parse_args()

    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info()
    print(f"{p.name}  sm_{p.major}{p.minor}  free {free/2**30:.2f} GiB of {total/2**30:.1f}", flush=True)
    import flashinfer
    print(f"torch {torch.__version__}  flashinfer {flashinfer.__version__}", flush=True)

    ws = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
    out = {"device": p.name, "free_gib": free / 2**30, "results": []}
    torch.manual_seed(0)

    for prefix in [int(x) for x in a.prefixes.split(",")]:
        L = prefix + NQ
        npages = -(-L // PAGE)
        # Contiguous K/V for all tokens; the paged view reshapes the same memory.
        k = torch.randn(npages * PAGE, HKV, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(NQ, HQ, D, device=dev, dtype=torch.bfloat16)
        ref = reference(q, k, v, prefix)
        flops = causal_flops(prefix)
        print(f"\n== prefix {prefix:,}  (kv {L:,})  {flops/1e12:.2f} TFLOP per call", flush=True)

        def record(name, fn, o_of):
            if a.only and not any(name.startswith(x) for x in a.only.split(",")):
                return
            try:
                o = fn()
                torch.cuda.synchronize()
                err = (o_of(o)[-ref.shape[0]:, :ref.shape[1]].float() - ref).abs().max().item()
                med, lo, hi = timeit(fn, a.iters)
            except Exception as e:  # a variant that cannot run is a result, not a crash
                print(f"  {name:<24} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
                out["results"].append({"prefix": prefix, "variant": name, "error": str(e)[:400]})
                return
            tf = flops / (med / 1e3) / 1e12
            print(f"  {name:<24} {med:8.2f} ms  [{lo:.2f}-{hi:.2f}]  {tf:6.1f} TFLOP/s  "
                  f"{tf/PEAK_TFLOPS:5.1%} of peak   max|err| {err:.2e}", flush=True)
            out["results"].append({"prefix": prefix, "variant": name, "ms": med, "ms_min": lo,
                                   "ms_max": hi, "tflops": tf, "frac_peak": tf / PEAK_TFLOPS,
                                   "max_abs_err": err})

        qo_indptr = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
        kv_cache = (k.view(npages, PAGE, HKV, D), v.view(npages, PAGE, HKV, D))
        last = torch.tensor([L - (npages - 1) * PAGE], dtype=torch.int32, device=dev)
        kv_indptr_p = torch.tensor([0, npages], dtype=torch.int32, device=dev)

        for tag, idx in (("fi_paged64", torch.arange(npages, dtype=torch.int32, device=dev)),
                         ("fi_paged64_shuf", None)):
            if idx is None:
                # Scatter logical pages over physical ones: permute the storage and
                # point the page table at the new positions, so the logical K/V
                # (and the reference) are unchanged.
                perm = torch.randperm(npages, device=dev)
                kc = torch.empty_like(kv_cache[0]); vc = torch.empty_like(kv_cache[1])
                kc[perm] = kv_cache[0]; vc[perm] = kv_cache[1]
                cache, idx = (kc, vc), perm.to(torch.int32)
            else:
                cache = kv_cache
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
            w.plan(qo_indptr, kv_indptr_p, idx, last, HQ, HKV, D, PAGE, causal=True,
                   q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            record(tag, lambda w=w, c=cache: w.run(q, c), lambda o: o)
            if tag == "fi_paged64_shuf":
                del kc, vc, cache

        wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        wr.plan(qo_indptr, torch.tensor([0, L], dtype=torch.int32, device=dev), HQ, HKV, D,
                causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        record("fi_ragged", lambda: wr.run(q, k[:L], v[:L]), lambda o: o)

        # PyTorch FA2. SDPA's is_causal is TOP-left aligned when q_len != kv_len,
        # which is the wrong mask here, so use the lower-right causal bias. If the
        # flash backend refuses it, that is recorded as the result.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.attention.bias import causal_lower_right
        qt = q.transpose(0, 1).unsqueeze(0)                       # [1, HQ, NQ, D]
        kt = k[:L].transpose(0, 1).unsqueeze(0)                   # [1, HKV, L, D]
        vt = v[:L].transpose(0, 1).unsqueeze(0)
        bias = causal_lower_right(NQ, L)

        def sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    qt, kt, vt, attn_mask=bias, enable_gqa=True)
        record("sdpa_flash", sdpa, lambda o: o[0].transpose(0, 1))

        # SGLang Triton extend kernel: prefix read through kv_indices, the new
        # chunk passed separately (as the backend does).
        try:
            import sglang.kernels.ops.attention.extend_attention as ea
        except Exception as e:
            print(f"  triton: import failed: {e}", flush=True)
            ea = None
        if ea is not None:
            kp, vp = k[:prefix], v[:prefix]
            kn, vn = k[prefix:L].contiguous(), v[prefix:L].contiguous()
            kv_indptr_t = torch.tensor([0, prefix], dtype=torch.int32, device=dev)
            kv_idx_t = torch.arange(prefix, dtype=torch.int64, device=dev)
            o_t = torch.empty_like(q)
            orig = ea._get_block_sizes_for_extend_attention
            orig_st = ea._get_num_stages_for_extend_attention
            configs = [("triton_prod_64x64_s1", None, None),
                       ("triton_128x64_w8_s1", (128, 64, 8), 1),
                       ("triton_64x64_w8_s2", (64, 64, 8), 2),
                       ("triton_64x32_w4_s2", (64, 32, 4), 2),
                       ("triton_128x32_w8_s2", (128, 32, 8), 2)]
            for name, bs, st in configs:
                if bs is not None:
                    def fake_bs(Lq, Lv, bs=bs):
                        bd, bdpe, bdv, _, _, _ = orig(Lq, Lv)
                        return bd, bdpe, bdv, bs[0], bs[1], bs[2]
                    ea._get_block_sizes_for_extend_attention = fake_bs
                    ea._get_num_stages_for_extend_attention = lambda Lq, Lv, block_n=None, st=st: st
                else:
                    ea._get_block_sizes_for_extend_attention = orig
                    ea._get_num_stages_for_extend_attention = orig_st

                def tri():
                    ea.extend_attention_fwd(
                        q, kn, vn, o_t, kp, vp, qo_indptr, kv_indptr_t, kv_idx_t,
                        None, True, None, NQ, 1.0, 1.0, sm_scale=1.0 / math.sqrt(D),
                        page_size=1, extend_seq_lens_cpu=[NQ], identity_kv_indices=False)
                    return o_t
                record(name, tri, lambda o: o)
            ea._get_block_sizes_for_extend_attention = orig
            ea._get_num_stages_for_extend_attention = orig_st
            del kp, vp, kn, vn, o_t

        del k, v, q, kv_cache, ref
        torch.cuda.empty_cache()

    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\nsaved {a.json}")


if __name__ == "__main__":
    main()
