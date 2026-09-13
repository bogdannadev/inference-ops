# Prefill custom kernels — campaign plan (Qwen3.8-27B, A100 SM80)

**STATUS: PLANNED, nothing built.** Opened 2026-09-13 by Danila. The goal is
lower prefill cost (TTFT) through custom Triton / CUDA C++ / CUTLASS-CuTe /
PTX kernels.

Reference list as supplied: `PREFILL_KERNELS_CHECKLIST.md`. This file checks
that list against **this** deployment:
- what the node has measured;
- what SGLang v0.5.19 already runs on our path, read from source at the tag;
- where the list needs correcting for this model;
- the order and the gates.

Read these before building anything:
- `NVIDIA_KERNEL_TUNING.md` §0/§0a (SM80 facts; prefill MFU).
- `tuning/lowlevel/README.md` (E1–E4 and the PTX toolchain rules).
- `HICACHE_DFLASH2.md` (the memory headroom this work must fit in).

---

## 1. Why prefill, and why it is a different problem from decode

Decode on this node is bandwidth-bound: GEMMs are 87.7% of decode time at
~2–4 FLOP/byte against a balance point of ~161, and instruction-level work was
ruled out there (E1–E3). **Prefill is the opposite regime.** It is
compute-shaped, and the measured tensor-core utilization is ~27% of BF16 peak
(`NVIDIA_KERNEL_TUNING.md` §0a, 16,384-token chunks, 165K cold prompt). It is
the only phase with arithmetic headroom, so the checklist's instruction-level
methods (multistage `cp.async`, `mma.sync` tiling, epilogue fusion) are
relevant here in a way they never were for decode.

**Do not carry decode results over:**
- E1's "workspace/split-K inert" and "efficiency tracks weight size" were
  measured at M ≤ 48.
- Production prefill runs M = 4096 per chunk (see §3), which is a
  high-arithmetic-intensity point.
- Re-measure everything at prefill shapes.

What production prefill looks like (2026-09-12/13):
- TTFT p95 27–29 s, almost all of it prefill of uncached tokens.
- `uncached_prompt_tokens` p95 ~39K; queue time ~0.
- HiCache (applied 2026-09-13) removes re-prefill of evicted sessions; what it
  leaves is **genuinely new tokens** (~2M/day), which only faster prefill
  kernels can help.
- Cost is linear in tokens to 165K: 0.498 vs 0.491 ms/token at 155K/165K. So
  the quadratic attention term is not dominant even at maximum context. That
  supports the checklist's "GEMMs and GDN before FlashAttention" priority.

## 2. Model geometry — verified against `Qwen/Qwen3.8-27B/config.json`

| | value | checklist |
|---|---|---|
| layers | 64 = 16 × (3 GDN + 1 full attention), `full_attention_interval 4` | ✓ |
| hidden / FFN | 5120 / 17408, vocab 248,320 | ✓ |
| GDN | 48 value heads, 16 key heads, key/value head dim 128, **`linear_conv_kernel_dim 4`** | ✓, but **omits the short causal conv1d** that precedes the scan |
| full attention | 24 Q heads, 4 KV heads, head dim 256, **`attn_output_gate true`** | ✓, but **omits the sigmoid output gate** (Q projection is 2 × heads wide) |
| RoPE | `partial_rotary_factor 0.25` → 64 dims, **interleaved mRoPE, `mrope_section [11,11,10]`** | says "RoPE on 64 dims". A custom kernel must reproduce mRoPE interleaving. For text all three position streams are equal; the OCR/vision path is not |
| GDN state dtype in production | **bf16** (`--mamba-ssm-dtype bfloat16`) | §4.4 recommends FP32. Switching roughly halves Mamba slots, and GPU memory is now tight (§3) |

## 3. Constraints that did not exist when the checklist was written

- **Chunked prefill is 4096 tokens** (was 16384; changed 2026-09-13 for
  DFlash2, which OOM'd at 16384). The production GEMM bucket is the checklist's
  **1025–4096**, and a long prompt is many 4096-token chunks with a growing
  attention prefix. Tune for that first; 4097+ is currently unused.
- **Activation headroom is ~2.9 GB** (`available_gpu_mem`, was 8.49 GB before
  DFlash2).
  - Every workspace the checklist proposes must fit: larger tiles, Split-KV
    reduction buffers, triple buffering, fused-epilogue scratch.
  - The first DFlash2 boot died in exactly this code path (`fla/chunk_o.py`,
    GDN prefill, 16K chunk).
  - Add a 160K + 3×30K stress run (`tuning/bench/mem_stress.py`) to every
    kernel's gate.
- **Prefill now includes DFlash2 drafter work:** `DFLASH fused KV
  materialization` projects hidden states from 5 target layers into the draft
  KV for every prefilled token. Profile it; it is not in the checklist.
- **HiCache write-through** copies newly prefilled KV + Mamba state to host.
  PCIe copies overlap prefill; profile that too.
- **No multi-GPU.** No P2P (NS), PCIe only; checklist §14 (TP, context
  parallelism) is ruled out (`NVIDIA_KERNEL_TUNING.md` §1a).
- **Serving GPUs only.** All measurement runs on a drained replica
  (`tuning/bench/run_eval.sh`, router `DELETE /workers`), DCGM exporter
  stopped for CUPTI, never on live traffic.

## 4. Checklist item by item: already upstream, open, or corrected

v0.5.19 source, on the path this node actually runs (SM80,
`--attention-backend flashinfer`, `linear_attn_backend triton`).

### Already done upstream — verify it is active, do not rebuild

| checklist item | what runs today |
|---|---|
| §8 last-token-only LM head | `logits_processor.py` prunes hidden states to the last token unless logprobs are requested |
| §9 packed varlen | extend batches are flattened with `cu_seqlens`/`extend_lens`; no padding |
| §10 direct paged KV writes | `set_kv_buffer` into the paged pool; `fused_qk_norm_rope_store.py` writes K/V straight to cache |
| §7 residual + RMSNorm | `GemmaRMSNorm.forward_cuda` → `gemma_fused_add_rmsnorm` |
| §7 QK-Norm + partial RoPE (+ output gate) | `fused_qk_gemma_rmsnorm_rope_gate` (enabled by `--enable-fused-qk-norm-rope`, on since 2026-07-31) |
| §7 SwiGLU / §3.4 gate-up split | `MergedColumnParallelLinear gate_up_proj` + `SiluAndMul` CUDA kernel |
| attention output gate | `fused_sigmoid_mul` |
| §4.3 GDN projection prep | `triton_gdn_fused_proj` QKVZBA split/reshape/cat, covering ratio 3 since v0.5.19 (#34859); `fused_gdn_gating.py` |
| §5 full attention | FlashInfer prefill (SM80 FA2-style tiled, GQA, head dim 256, paged) |

What is left in these rows is **epilogue fusion into the GEMM itself** (§3.4):
the ops are fused with each other, but the GEMM writes its output to HBM first.
That is Phase 3 work, not Phase 2.

### Open, and genuinely unserved on SM80 — the strongest leads

1. **GDN prefill kernel.**
   - SM80 runs the generic FLA Triton `chunk_gated_delta_rule` (chunked WY form:
     `wy_fast.py`, `chunk_o.py`).
   - The FlashInfer GDN prefill kernels are gated to **SM90/SM100 only**
     (`gdn_backend.py::flashinfer_gdn_prefill_default`: `sm_major not in (9,
     10)` → Triton). Upstream has no SM80-tuned GDN prefill at all.
   - 48 of 64 layers use it. It is also where the DFlash2 OOM happened, so its
     workspace matters.
   - E3 found the sibling decode kernel emits **zero `cp.async`**. Check whether
     the prefill kernels do.
2. **Prefill-shape GEMMs** (gate_up `[4096, 5120]×[5120, 34816]`, down,
   QKVZBA, o_proj) with fused epilogues. cuBLASLt already picks `cp.async`
   kernels (`ldg8` in names); the question is tile choice and epilogue fusion
   at M=4096, which nobody has measured here.
3. **causal conv1d prefill** (kernel 4) — missing from the checklist,
   `causal_conv1d_fn` CUDA. Measure its share.

### Corrections to the checklist

- **§4.1–4.2 associative affine scan: probably the wrong starting design for
  this model.**
  - The gated delta rule's transition is `A_t = α_t (I − β_t k_t k_tᵀ)`, a
    dense 128×128 matrix per value head.
  - A prefix scan composes `A₂A₁` products: O(d³) arithmetic and O(d²) memory
    per combine, per head, per chunk summary.
  - The chunked **WY/UT representation** FLA already uses exists precisely to
    avoid materializing those products, and costs O(d²·C).
  - Start Phase 4 by profiling and specializing the existing chunk kernel
    (launch config, `cp.async`, fused conv + gating + prep). Only prototype a
    scan if the profile shows the WY path is structurally starved.
- **§4.4 FP32 state:** production state is bf16 (`--mamba-ssm-dtype bfloat16`,
  a cookbook-validated accuracy gate). Any kernel must accept bf16 state. An
  fp32 variant is a memory decision (Mamba slots), not only a kernel decision.
- **§5.4 RoPE:** it is interleaved mRoPE, not plain partial RoPE (§2 table).
- **§6/§2 PTX instruction names:** `cp.async.{cg,ca}.shared.global`,
  `commit_group` and `wait_group` are **verified** against `ptxas` in
  `tuning/lowlevel/refs/PTX_SM80_VERIFIED.md`.
  - `mma.sync.aligned.m16n8k16...`, the `ldmatrix.sync...` variants and the
    swizzle layouts are **not verified**. Compile a probe before designing
    around them. That file exists because the ISA doc and this toolchain
    disagree.
- **§15 "speculative decoding is lower priority":** irrelevant to prefill, but
  note the DFlash2 drafter adds prefill work (§3).
- **§14 multi-GPU:** ruled out on this host.

## 5. Order, with gates

Checklist §16, adjusted. Each phase ends with a decision, and **any op under
~10% of prefill time at production shapes does not get a custom kernel** (the
same stop rule that vetoed decode kernel work).

**Phase 1 — profile prefill (never done on this node).**
- **Where:** drained replica, DCGM exporter stopped, current production config
  (DFlash2 + HiCache, chunk 4096).
- **Prompts:** cold 4K, 32K, 128K, plus a prefix extension of 16K on a cached
  64K prefix.
- **Per-op share of prefill wall time:**
  - GDN: in_proj GEMM, conv1d, gating/prep, chunk scan, state write, out_proj,
    norm-gate.
  - Full attention: QKV GEMM, QK-norm-RoPE-gate, FlashInfer prefill, o_proj.
  - FFN: gate_up, SiluAndMul, down.
  - Norms and residuals; DFlash2 KV materialization; HiCache write-through;
    launch/host gaps.
- **Tools:** Nsight Systems for the timeline, Nsight Compute on the top kernels
  (grid, `cp.async` count, register spills).
- Also record per-chunk tok/s and TFLOPS from the scheduler log, to compare
  against the 27% MFU at 16K chunks.
- **Deliverable:** a table in this file, and the decision which of Phases 3–5
  to fund.

**Phase 2 — verification only.** Confirm every "already upstream" row in §4 is
live on our path, e.g. that `fused_qk_gemma_rmsnorm_rope_gate` and the QKVZBA
fused proj actually dispatch at prefill shapes. Cheap, and it prevents
rebuilding something that exists.

**Phase 3 — prefill GEMMs + epilogues.** Shape-bucketed for M ≤ 4096 (plus
decode/verify shapes untouched). CUTLASS/CuTe with the checklist §3 tile space.
Fuse gate_up + SiLU×up, and down + residual.

**Phase 4 — GDN prefill (chunk kernel specialization first; see corrections).**
Triton first, CUDA/PTX for the final hot path.

**Phase 5 — full attention.** Only if Phase 1 shows the 16 FlashInfer prefill
layers above the stop rule at 128K. The linear-in-tokens measurement says
probably not.

**Phase 6 — long-context specializations** (Split-KV, multi-CTA scan): only
with Phase 1 evidence at 128K+.

## 6. Acceptance for every kernel

**Speed:**
- TTFT from `tuning/bench/hicache_probe.py` (`A_cold`, currently 14.2–14.3 s
  for 45K).
- Per-chunk tok/s from the scheduler log.
- Kernel-level numbers per checklist §17.
- Noise floor ~1.5% only with the replica drained and the cache verifiably
  flushed (`UPGRADE_v0.5.19.md` Phase 0 rules).

**Memory:** `tuning/bench/mem_stress.py` (160K cold + 3×30K decoding, twice)
with zero `OutOfMemoryError`. Boot `max_total_num_tokens` must stay ≥ 169,000
and `available_gpu_mem` must not fall.

**Numerics:**
- Same-shape kernels (a GEMM or fusion that claims bit-exactness): greedy
  byte-identity (`benchmarks/byte_identity.py`) from a flushed cache.
- Numerics-changing kernels: checklist §17 error metrics on hidden states and
  logits at 8K/32K/128K, then `tuning/bench/spec_eval.py` correctness gate
  (objective answers at concurrency 4, leak check).
- Also watch DFlash2 accept length: a prefill kernel that perturbs hidden
  states changes what the drafter sees.

**Rollout:** one replica at a time via `deploy/roll-replica.sh`, the same
drain → test → re-register sequence used for HiCache/DFlash2.
