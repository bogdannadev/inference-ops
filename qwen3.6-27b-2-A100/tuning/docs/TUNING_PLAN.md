# Qwen3.6-27B / 2×A100 — Inference Tuning Plan

Written 2026-07-31. All numbers below were **measured on the live containers**
(uptime 2 days, boot 2026-07-28 22:36), not estimated. Baseline to beat:
`benchmarks/results/worker_r0_memfrac090.json` and `ladder_round_robin.json`.

---

## 0. Measured baseline — the facts this plan rests on

### Hardware / model shape

```
GPU                      2× A100 80GB PCIe, compute capability 8.0 (SM80)
num_hidden_layers        64
full_attention_interval  4      -> 16 full-attention layers, 48 linear/GDN layers
hidden_size              5120
head_dim                 256    <-- unusually large; see §4a
num_attention_heads      24 (GQA, num_key_value_heads = 4)
max_position_embeddings  262144 (model native)
context-length served    160000
```

KV cross-check: `160,768 × 16 layers × 4 kv_heads × 256 head_dim × 2 B = 5.27 GB`
vs boot log `K size: 4.91 GB`. Confirms only 16 layers carry KV — which is why
160K tokens of context costs under 10 GB.

### Pools

```
max_total_num_tokens     160,768        (KV, profiled at mem-fraction 0.90)
available_gpu_mem        5.73 GB
max_mamba_cache_size     48 slots       <-- the real concurrency ceiling
  observed usage         4 slots per running request (tracks num_draft_tokens)
  => ceiling             48 / 4 = 12 concurrent requests
```

### Speculative decoding (EAGLE, live decode logs, n=1545)

```
mean accept len   3.483   (cap is 4.00 = --speculative-num-draft-tokens)
at cap (>=3.99)   15.3%
>= 3.5            56.5%
< 3.0             16.2%
```

Solving `1 + p + p² + p³ = 3.483` → **per-token acceptance p ≈ 0.908**.
The draft head is excellent on this traffic and is being clipped by the cap.

### Concurrency ladder (`ladder_round_robin.json`)

**Metric semantics (re-read from `ladder.py:106-111`, corrected 2026-07-31):**
`aggregate_tok_s` = completion tokens ÷ wall (true throughput).
`split_r0`/`split_r1` = `spec_verify_calls_total` deltas, counted
**per request**, not per batched forward, and **not** token counts.

| c | wall | agg tok/s | per-stream | p50 | max | verifies | accept len | waves | predicted wall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.889 | 69.24 | 69.24 | 2.887 | 2.887 | 62 | 3.23 | 1 | — |
| 2 | 2.931 | 136.47 | 68.23 | 2.929 | 2.929 | 125 | 3.20 | 1 | — |
| **4** | 3.057 | **261.73** | 65.43 | 3.023 | 3.054 | 248 | 3.23 | 1 | 3.057 |
| 6 | 5.951 | 201.63 | 33.60 | 3.232 | 5.948 | 374 | 3.21 | 2 | 6.114 |
| 8 | 6.560 | 243.89 | 30.49 | 5.830 | 6.554 | 514 | 3.11 | 2 | 6.114 |
| 12 | 9.961 | 240.94 | 20.08 | 6.369 | 9.951 | 785 | 3.06 | 3 | 9.171 |

**The ladder never saturated the GPU — not at any rung.** With
`--max-running-requests 2` × 2 replicas the admission cap is 4, so every rung
above 4 executes as *waves of 4*. Modelling wall time as
`ceil(c/4) × 3.057s` predicts 6.114 / 6.114 / 9.171 against measured
5.951 / 6.560 / 9.961 — within 3–9%. The c=6 row is the proof: p50 3.232
(one wave) vs max 5.948 (two waves), textbook bimodal queueing.

So the "~240 tok/s plateau" is not a hardware ceiling. It is **c=4 replayed**.
Every rung ≥6 measured the same thing.

**Corollary — this invalidates A/B testing above the cap.** Until the cap is
lifted, no kernel flag, EAGLE depth, or backend change can be distinguished on
this ladder: all of them return the wave-serialized number.

**Accept length is flat under load** (3.23 at c=1 → 3.21 at c=6, tailing to
3.06 at c=12). Batching does not cost speculation. Note this is **3.2 on
ladder traffic**, vs 3.483 measured from production decode logs — the ladder's
short "explain X" prompts are marginally less predictable. Use 3.2 for
ladder-based projections and 3.48 for production ones.

### Router fix (comparing `ladder_after_items124.json` → `ladder_round_robin.json`)

| c | cache_aware | round_robin | splits under cache_aware |
|---|---|---|---|
| 2 | 104.53 | 136.47 | 0 / 122 — all on r1 |
| 4 | 132.18 | 261.73 | 248 / 0 — all on r0 |
| 6 | 204.13 | 201.63 | balanced |
| 8 | 243.22 | 243.89 | balanced |
| 12 | 240.53 | 240.94 | balanced |

Round-robin fixed *balance*, and balance stopped mattering the moment the
admission cap became binding. c≥6 is statistically identical across both.

### HiCache (see §1)

```
cached_tokens device   r0 7,496,896   r1 8,026,048
cached_tokens host     r0    32,768   r1    34,816   -> 0.43% of hits
host RAM cost          253 GiB RSS per replica (506 GiB total)
boot cost              ~4m10s per replica
```

---

## The governing constraint: decode is weight-streaming bound at ~55% of peak

27B BF16 ≈ 54 GB of weights. Each forward pass must stream all of it.

Measured directly from the c=4 rung rather than inferred (corrected 2026-07-31 —
the earlier derivation divided tok/s by accept length; the verify-call counter
gives it without that assumption):

```
c=4: r0 logged 123 verify calls in 3.057 s with 2 concurrent requests
     verify calls are PER REQUEST, so batched forwards = 123 / 2 = 61.5
     -> 20.12 BATCHED forwards/s per replica
     -> 54 GB × 20.12 = 1086 GB/s = 56.1% of A100 HBM peak (1935 GB/s)
```

44% of peak bandwidth is unused. And decode GEMMs at batch 2–4 have arithmetic
intensity ~2–4 FLOP/byte against A100's BF16 balance point of ~156 FLOP/byte —
so widening the batch **divides the same forward rate across more streams**
rather than slowing it down. That is exactly what rows 1→4 show: per-stream
fell only 69.24 → 65.43 (5.5%) while aggregate went 3.78×.

**This is why 1→4 scaled 3.78× at near-zero cost, and it is the single most
important fact for prioritisation:**

- **Batching amortises the dominant term.** Adding concurrency is nearly free
  until the ~45% remaining bandwidth headroom is consumed.
- **Kernel tuning attacks the smaller term.** GDN/attention kernel time is a
  minority of a forward pass that is dominated by streaming 54 GB of weights.

So: concurrency first, speculation second, kernels third. Do not invert this.

---

## Phase 1 — Remove HiCache + raise concurrency  (highest value, lowest risk)

> **STATUS: edited into `docker-compose.yml` 2026-07-31, NOT YET ROLLED.**
> `docker compose config` validates; both replica blocks carry
> `--max-running-requests 4`, `--cuda-graph-max-bs-decode 4`, and no HiCache
> flags. The rolling restart is a production action and is left to the operator.

### 1a. Drop HiCache

Remove from **both** replica command blocks:

```
--enable-hierarchical-cache
--hicache-size 128
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
--hicache-write-policy write_through
```

Justification: 0.43% of cache hits, 506 GiB RAM, ~4 min per replica boot.
Prefix caching is the *device* radix tree and is unaffected (`disable_radix_cache`
stays False) — the documented 36× TTFT win (12.278s → 0.338s) is the device tier.

**Verify on first boot:** `max_total_num_tokens` must still be `160768`.
The KV pool is allocated at 22:36:56, four minutes *before* any HiCache
allocation, so this should be unchanged. Confirm, don't assume.

### 1b. Raise concurrency

```
--max-running-requests 4        (was 2)
--cuda-graph-max-bs-decode 4    (was 2 — MUST move together; it clamps to
                                 max-running-requests, so leaving it at 2
                                 silently caps the benefit)
```

Budget check at concurrency 4/replica:
- Mamba: 4 req × 4 slots = 16 of 48 ✓
- KV: 4 × 40K-token requests = 160K of 160,768 ✓ (tight for large prompts;
  the scheduler retracts rather than erroring)
- Bandwidth: still ~55% of peak ✓

**Gate:** re-run `ladder.py`. Expect the concurrency-6 and -8 rows to stop
collapsing. If aggregate at 8 does not clear ~400 tok/s, stop and profile
before going further.

---

## Phase 2 — EAGLE depth

Current `num_steps=3, topk=1, num_draft_tokens=4, cuda_graph_max_bs_decode=2`
is **verbatim the SGLang docs' OOM-recovery recipe**, not a tuned configuration.
It was set to escape an OOM and never revisited.

Docs constraint: *set all three explicitly, or leave all three unset for
auto-tuning.* With `topk=1`, `num_draft_tokens` auto-adjusts to `num_steps + 1`.

Projected at p = 0.908 (chain drafting):

| config | expected accept len | tokens/verify vs now |
|---|---|---|
| steps=3, draft=4 (current) | 3.48 | — |
| steps=5, draft=6 | ~4.78 | +37% |
| steps=7, draft=8 | ~5.85 | +68% |

These are tokens per verify pass, **not** wall-clock. Deeper chains cost more
draft forwards and a wider verify, so net speedup will be substantially lower.

**Coupling to watch:** `mamba num` tracks `num_draft_tokens`. At draft=6 the
ceiling drops to 48/6 = 8 concurrent. EAGLE depth and concurrency compete for
the same 48-slot pool. Confirm this empirically — it is an observed correlation
(1 request → `mamba num: 4` at draft=4), not a documented law.

Then try tree drafting (`topk=2`, e.g. steps=4/topk=2/draft=10) for higher
acceptance at more memory and compute.

**Tool:** `bench_speculative.py` (referenced by the SGLang docs for exactly this
sweep). Prefer it over hand-tuning.

---

## Phase 3 — Kernel flags (cheap A/Bs, no code)

| flag | current | try | rationale |
|---|---|---|---|
| `--enable-fused-qk-norm-rope` | **off** | on | Qwen3.5 uses QK-norm; fuses norm+RoPE. Cheap, low risk. Do this first. |
| ~~`--linear-attn-backend`~~ | `triton` | — | **DEAD ON SM80, do not spend a roll.** `gdn_flashinfer.py:58` gates on `capability[0] >= 9`; `gdn_cutedsl.py` needs SM90+ (decode) / SM100+ (prefill); `flashkda` is KDA, a different model family. We are `(8, 0)`. Triton is the only backend for these 48 layers — which is exactly why hand-tuning it (§4a → `KERNEL_TUNING_SPEC.md`) is the only lever they have. |
| ~~`--linear-attn-decode-backend` / `--linear-attn-prefill-backend`~~ | unset | — | Same gate. Nothing to split between. |
| ~~`--mamba-backend`~~ | `triton` | — | Expected to fail the same way. |
| `--speculative-attention-mode` | `prefill` | `decode` | Draft-model attention mode. Straight A/B. |
| `--enable-linear-replayssm` | off | on | **Distinct flag** from the `--enable-gdn-replayssm-spec` already rejected for the `extra_buffer` conflict. May not carry that conflict. Verify at boot. |
| `--bf16-gemm-backend` | `auto` | `torch`, `cutedsl` | Low expected gain; cuBLAS is well-tuned on SM80. |

**SM80 caveat:** these GPUs are compute capability **8.0**. `cutedsl` is
CUTLASS-DSL and generally targets SM90+; it may refuse or silently fall back.
Test and read the boot log — do not assume it engaged.

---

## Phase 4 — Manual kernel tuning

Only worth doing after Phase 0 profiling says where the time goes. Given the
55%-of-peak weight-streaming result, expect this to be a 5–15% class of win,
not a 2× one.

### 4a. Triton block-size tuning on the GDN/linear-attention kernels

**SUPERSEDED 2026-07-31 by `KERNEL_TUNING_SPEC.md` + `KERNEL_TUNING_INFO.md`,
which are written from the actual kernel sources in this image.** Two premises
in the original text below were wrong and are corrected there:

- **`head_dim = 256` is the FULL-ATTENTION head dim**, and those 16 layers run
  on FlashInfer, not Triton. The 48 Triton GDN layers use
  `linear_key_head_dim = linear_value_head_dim = 128` — a standard size. The
  "unusual head_dim mistunes the Triton kernels" hypothesis does not hold.
- **The real finding is better.** `fused_sigmoid_gating_recurrent.py:292-293`
  hardcodes `num_warps = 1, num_stages = 3` with no autotune, giving a grid of
  `192 × batch` single-warp blocks — **5.6% of A100 warp slots at batch 2,
  11.1% at batch 4.** The kernel is grid-starved, and `num_stages = 3` pipelines
  a serial recurrence of trip count 1–4. Separately, `chunk_delta_h.py` already
  exposes `SGLANG_GDN_CHUNK_H_{BV,NUM_WARPS,NUM_STAGES}` env knobs (zero code
  change), and `chunk_o.py` / `l2norm.py` ship with `@triton.autotune`
  commented out and the intended sweep left in the comment.

Original text kept below for provenance only — do not follow it.

> `head_dim = 256` is unusual — most Triton attention/linear-attention kernels
> ship with `BLOCK_*`, `num_warps` and `num_stages` tuned for head_dim 64 or
> 128. At 256 the default configs are likely shared-memory-bound or
> occupancy-limited on SM80. Procedure: locate the kernels; check for
> `@triton.autotune` vs fixed constants; microbenchmark at head_dim=256 sweeping
> `BLOCK_M/BLOCK_N`, `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`; promote only
> on a real-ladder win.

### 4b. CUDA graph capture shapes

Currently captured: `decode bs=[1,2]`, `prefill max_bs=16384` with 74 buckets.

With EAGLE the decode graph shape is `batch × num_draft_tokens`. After Phases
1–2 the real shapes change (e.g. 4 × 6 = 24). Use `--cuda-graph-bs-decode` to
capture the **exact** batch sizes in use rather than letting it pad to the
nearest captured bucket. Padding waste is proportional to how far the real
shape sits above a captured one.

`--cuda-graph-config` also accepts per-phase JSON:
```
{"decode":{"backend":"full","max_bs":8},
 "prefill":{"backend":"tc_piecewise","tc_compiler":"inductor"}}
```
`tc_piecewise` + inductor on the prefill phase can fuse pointwise chains.
Backends available: `full`, `breakable`, `tc_piecewise`, `disabled`.
Prefill is currently `breakable`.

### 4c. Page size

`--page-size 64` currently. This governs both radix-cache match granularity and
attention kernel paging. At 64, prefix matches round down to 64-token
boundaries — coarse for coding-agent traffic with variable-length turns.
Sweep 32 vs 64: smaller pages improve cache hit granularity at the cost of more
page-table overhead per attention call.

Related: `--enable-page-major-kv-layout` (currently off) changes KV memory
layout and therefore attention kernel access patterns. A/B it.

### 4d. FlashInfer autotuner

`disable_flashinfer_autotune=False`, so it is already on and profiling configs
at warmup. If per-layer profiling shows one op behaving badly, use
`--flashinfer-autotune-skip-ops` to exclude it and fall back.

### 4e. torch.compile

`--enable-torch-compile` (off), `--torch-compile-max-bs 32`,
`--cuda-graph-tc-compiler {eager,inductor}` (currently `eager`).
Inductor can fuse across the hybrid stack, but on a Mamba/attention hybrid with
EAGLE this carries real breakage risk and long warmup. Last resort, and only
with a clean rollback.

### 4f. Mamba pool expansion — raises the concurrency ceiling

Only becomes the binding constraint above ~8–12 concurrent, but it is the thing
that will bind:

- `--mamba-full-memory-ratio` (0.9 now) — raise for more slots
- `--enable-int8-mamba-checkpoint` + `--int8-mamba-ckpt-size` — quantised
  checkpoints, more slots for the same VRAM
- `--max-mamba-cache-size` — currently auto-profiled to 48

This is the lever that lets Phase 1 and Phase 2 both go further at once.

---

## Phase 0 — Instrumentation (do this alongside Phase 1)

Needed before Phase 4 is worth starting. Answers: *what fraction of a decode
forward is GDN linear-attention vs full attention vs GEMM vs the draft model?*

- `--enable-layerwise-nvtx-marker` + `nsys` → per-layer timeline
- `/start_profile` and `/stop_profile` HTTP endpoints → torch profiler traces
- `--enable-forward-pass-metrics` → per-pass timing into Prometheus
- `--enable-profile-cuda-graph`, `--debug-cuda-graph` → graph capture diagnostics

Decision rule: if GDN linear-attention is **under ~15%** of decode time, skip
§4a entirely and spend the effort on Phase 1/2/4f instead.

---

## Method / discipline

Carried over from the discipline already documented in `docker-compose.yml`:

1. **One variable at a time.** Every phase above is independently reversible.
2. **One replica at a time**, so the other keeps serving:
   ```
   docker compose up -d --no-deps qwen36-27b-r1
   # wait healthy (~5 min once HiCache is gone, was ~9)
   docker compose up -d --no-deps qwen36-27b-r0
   docker compose restart qwen36-27b-router
   ```
   Never `--remove-orphans` (would delete qwen3-emb, grafana, prometheus, dcgm).
3. **Keep the image digest pinned.** Do not let a tuning roll also move the
   engine build — that confounds every measurement.
4. **Read the boot log every time.** Specifically `max_total_num_tokens`,
   `max_mamba_cache_size`, and whether a requested backend actually engaged or
   silently fell back. Several flags in Phase 3 can be accepted and ignored.
5. **Gate on `ladder.py`**, not on single-stream `run_worker.sh`. Single-stream
   numbers were identical at mem-fraction 0.85 and 0.90 (68.50 vs 68.97) and
   will under-report anything that helps batching.

## Known config issues to fix while in here

- ~~`--log-requests` / `--log-requests-level 1` are being dropped somewhere.~~
  **RESOLVED 2026-07-31 — not a bug.** `docker inspect -f '{{json .Config.Cmd}}'
  on the running r0 shows `--log-requests` is **absent from the container's
  actual argv**. The flags were added to `docker-compose.yml` in a later editing
  session and the containers, up 2 days, simply predate the edit. Nothing is
  parsing them wrong; they have never been applied. The Phase 1 roll applies
  them. (Same story for `--context-length 160000` and `--mem-fraction-static
  0.90`, except those *were* applied — they went in before the last roll.)
- The comment "831 GB host RAM free against 128 GB of `--hicache-size` per
  replica" is wrong by ~2×: `--hicache-size` sizes the KV tier only; the Mamba
  tier (128.02 GB) and draft tier (8 GB) are allocated on top, for 264 GB per
  replica. Moot once Phase 1a lands, but the reasoning error is worth noting.
- `--max-total-tokens 190000` is inert (already documented). Leave it.
