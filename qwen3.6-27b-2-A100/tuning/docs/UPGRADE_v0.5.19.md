# SGLang v0.5.19 upgrade + HiCache L2 — 2026-09-05

**STATUS: PLANNED, NOT APPLIED.** Two changes, deliberately separable:
engine `v0.5.18 -> v0.5.19`, then HiCache host tier (L2 only). They are
staged as separate phases because bundling an optional flag into a baseline
roll is exactly what failed on 2026-08-24 (`--startup-weight-load-mode
overlap`, hard boot failure).

Target image:

```
lmsysorg/sglang:v0.5.19-cu130
sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
```

Released 2026-09-05 02:27 UTC (image pushed 2026-09-04 22:50 UTC).
786 PRs / 214 contributors; 794 commits ahead of `v0.5.18`.

## Provenance check

Read from the registry config blob, not from a running container.

| | v0.5.18-cu130 | v0.5.19-cu130 |
|---|---|---|
| `ai.sglang.build.commit` | `71de97b2…` | `0bcd8223…` |
| `ai.sglang.image.tag` | `lmsysorg/sglang:v0.5.18` | `lmsysorg/sglang:v0.5.19` |
| build URL | actions run 32520522511 | actions run 33912440803 |
| CUDA | 13.0.3 | **13.0.3 (unchanged)** |
| cuDNN | 9.14.0.64 | **9.14.0.64 (unchanged)** |
| NCCL | 2.28.3 | **2.28.3 (unchanged)** |
| FLASHINFER_VERSION | 0.6.17 | **0.6.18** |
| amd64 layers / size | 66 / 14.18 GB | 70 / 15.10 GB |

LABEL, env and tag agree; the build URL is an official `sgl-project/sglang`
Actions run. No provenance hole. `NVIDIA_REQUIRE_CUDA` unchanged
(`cuda>=13.0`); host driver 610.57.04, so no driver work.

**This is a far smaller platform delta than v0.5.18 was.** That release moved
torch two minor versions, CUDA, cuDNN and FlashInfer at once. Here only
FlashInfer moves, by one patch-minor. Torch is *expected* to stay 2.13.0 (the
release notes cite torch 2.13.0 as already-current when re-enabling
`--enable-symm-mem` for Kimi hybrids) but the label does not carry it —
**confirm from the pulled image before rolling.**

## What is actually in it for us — one thing, and it was written for our shape

`grep -i 'sm80\|a100\|ampere'` over the release body returns **zero hits**,
third release running. Every headline lands elsewhere: beam search, DeepEP v2,
LayerNorm-SP (needs TP>1), W4A8 MoE on Hopper, DCP on Blackwell MLA, fmha_v2
(SM90/120), Lean attention (ROCm), and the whole NPU/XPU/diffusion block.

The exception is **PR #34859, "Qwen3.8-27B Model Support"**, and it is not a
model-loading change — it is a kernel gate.

### The GDN head-group ratio finally covers us

Our config, read from the live model:

```
linear_num_value_heads = 48
linear_num_key_heads   = 16      ->  head-group ratio 3
layer_types            = [linear, linear, linear, full] x 16
                       ->  48 of 64 layers are GDN
```

v0.5.18 (`models/qwen3_5.py:144`) gates the fused QKVZBA
split/reshape/cat Triton kernel on:

```python
_GDN_FUSED_QKVZBA_RATIOS = (1, 2, 4, 8) if _use_aiter else (1, 2, 4)
```

Ratio 3 is in neither tuple, so **all 48 GDN layers take the unfused
fallback every forward**: two `.contiguous()` copies plus a `torch.cat`.
This was recorded in `UPGRADE_v0.5.18.md` as a dead end ("in neither tuple").

v0.5.19 changes it to:

```python
_GDN_FUSED_QKVZBA_RATIOS = (
    (1, 2, 4, 8) if _use_aiter else (1, 2, 3, 4) if _is_cuda else (1, 2, 4)
)
```

and teaches the kernel a `V_POW2` constexpr. `tl.arange` only accepts
power-of-two extents, so non-power-of-two groups now walk the group one
`HEAD_V`-sized head at a time under `tl.static_range`, while power-of-two
groups keep the single wide vector access. The upstream comment names the
case outright: *"the ratio-3 dense 27B layout"*. This change exists for us.

**Size the expectation honestly.** Those copies do not live in the 2.49% GDN
recurrent kernel from the S1 profile (`RESULTS.md` 2026-07-31c) — they land in
the *elementwise / at::native* bucket, ~4.5% of decode, and this chain is only
a slice of it. Decode remains 87.7% GEMM weight-streaming and nothing here
touches that. Expect **≤1%, most likely inside the ~1.5% noise floor** on
decode; prefill/TTFT is where it has any chance of resolving.

It is still the right *class* of change for this node: A100 decode is
bandwidth-bound, so bytes-moved-per-forward is the only thing that shows, and
removing two copies and a cat is a bytes-moved reduction, not a host-side
scheduling win.

**And it is genuinely falsifiable, which is the useful part.** The non-pow2
path is *narrower per access* (128 elements x 3) than the wide vector access
power-of-two ratios get. Fused-but-narrow could lose to the unfused cat. Do
not assume the sign — measure it. Layout is unchanged, so byte-identity must
still hold.

## Breaking changes, audited against our config

| Breaking change | Verdict | Why |
|---|---|---|
| Unified radix tree default for all configs | **inert** | we are hybrid-SSM and were already on the unified tree; the flip targets full-attention-only models. `SGLANG_ENABLE_UNIFIED_RADIX_TREE` deprecated — we never set it |
| FlashInfer 0.6.18 required, no fallback | **REAL RISK** | we run `--attention-backend flashinfer`. Same class as 0.5.18's 0.6.15->0.6.17: it moves the floor under every number in `tuning/results/` |
| `ServerArgs` no longer resolves on construction (`resolve_once()`) | **inert, verified** | `grep -rn ServerArgs benchmarks/ tuning/ deploy/` returns nothing. No tooling of ours builds one |
| 32 stop strings / 32 stop regex / 256 bytes each, else HTTP 400 | **inert, verified** | no stop-string configuration anywhere in the repo; only `finish_reason: "stop"` in saved benchmark result files |
| Spark3 -> Spark2.5 rename | inert | not our model |
| DeepSeek-V4 FP4 defaults to FlashInfer MXFP4 | inert | not our arch |
| W4A4 MegaMoE moves to `--enable-w4a4-megamoe` | inert | dense model, no MoE |
| Kimi hybrids honor `--enable-symm-mem` under CUDA graphs | inert | not our model |
| Lean attention auto-enabled on ROCm Triton | inert | CUDA |
| ROCm/XPU image and flag changes | inert | CUDA |
| Diffusion per-request params, Hunyuan tiled VAE | inert | not a diffusion server |
| `kill_process_tree` waits for reap | inert | our harnesses are local |

### Fixes that touch our code paths but do not fire

- **#33431, skip padded state slots in the chunked GDN kernel.** Real bug in
  a kernel we run, but the fault needs an all-padded extend batch, which is
  produced by an idle DP-attention rank under breakable-CUDA-graph prefill.
  We are TP1 with no DP attention. Hardening, not a fix for us.
- **#35297, Qwen3_5 text-only archs into the mamba radix whitelist.** Targets
  `Qwen3_5MoeForCausalLM`. Ours is `Qwen3_5ForConditionalGeneration`, already
  whitelisted, and we pin the strategy explicitly.
- **#34053, account resident weight memory in KV sizing.** Gated on the IPC
  weight-cache daemon; `weight_cache_mode` reads `off`.
- **#35158 / #35177 / #33091, unified-*memory* sizing, sub-pools and
  eviction.** All gate on `--enable-unified-memory`, which reads `False`.
  Note #35177 ("three sub-pools for mamba + hybrid-SWA") is the first time
  unified memory could serve our family at all. That is a separate lead, not
  part of this roll.

### The one that can move a boot gate

**#36583 — KV pool sized far too small when weight-loading memory is still
referenced.** The KV budget is measured from free GPU memory right after
weights load. `get_available_gpu_memory()` calls `empty_cache()`, which only
returns *unreferenced* blocks, so loader temporaries that are still
referenced counted as used. The fix collects before measuring. Upstream repro
on 4xH200 (TP4 + EAGLE, `--mem-fraction-static 0.8`): `max_total_num_tokens`
1,741 -> 349,377.

We pin `--mem-fraction-static 0.92`, so we were never in the pathological
case, but this changes the same measurement we depend on. **Our margin is 408
tokens** (`max_total_num_tokens=169408` vs `--context-length 169000`).

Direction should be *up* — more memory seen free means a bigger pool. A move
down of more than 408 tokens breaks the context contract, which is the same
failure the ReplaySSM experiment hit at 137,600 (`RESULTS.md`). This is the
first line to read in the boot log.

## Two things worth having regardless of the roll

**#34608 — publish per-scheduler load on a dedicated socket.** This describes
a defect we have. The router prices workers from its *own* in-flight counter,
which (a) misses the break-glass direct-hostname traffic — still the majority
of requests until Stage B's per-person keys land — and (b) for streaming
responses stays held for the whole response rather than the time the request
occupies the scheduler. The engine already computes the right number
(`load_snapshot_publish_interval=15` is live on our workers today); 0.5.19
exposes that snapshot out-of-process. Payoff is both better routing and a real
queue-depth / KV-occupancy signal to scrape. Not part of this roll.

**#24911 — roofline trace annotations.** A new `roofline_annotations` argument
on the profile endpoint, adding KV-length distribution across context vs
generation phases. Torch-profiler traces, not OTLP — so it is for the next
profiling run, not for the metrics work.

---

# HiCache L2 (host RAM tier)

Goal: absorb prefix-cache evictions into host RAM so a returning session is
*reloaded* over PCIe instead of *recomputed*, and keep latency flat as
concurrency grows. **L2 only** — host memory tier, no L3 storage backend.

## Does it even work on a Mamba hybrid? Yes, and this was the open question

Read from the v0.5.19 source, not from `--help`.

`mem_cache/registry.py::default_radix_cache_factory` routes us to
`_create_unified_radix_cache`, which builds:

```python
tree_components = [ComponentType.FULL]
if ctx.is_hybrid_ssm:  tree_components.append(ComponentType.MAMBA)   # <- us
...
if ctx.enable_hierarchical_cache:
    cache.init_hicache(server_args, params)
```

There is **no Mamba guard on this path.** The only Mamba rejections in the
tree are for different features:

- `disaggregation_decode_retraction_backup == "host_pool"` -> *"Host-pool
  retraction does not support Mamba models."* We do not run PD disagg.
- `--hicache-host-memory-mode buffer_only` -> raises unless components are a
  subset of `{FULL, SWA}`, with a TODO noting Mamba has no state-handoff
  channel yet. **So `buffer_only` is closed to us. `cache` mode — the default,
  and the one we want — is open.** `buffer_only` requires a storage backend
  anyway, so it was never in scope for an L2-only build.

**Crucially, the Mamba states themselves offload**, not just the
full-attention KV. `unified_cache/components/mamba_component.py` carries the
whole HiCache surface: `_mamba_pool_host` ("set to host mamba pool when
HiCache enabled"), a per-component host LRU, `host_value` on nodes,
`mamba_host_hit_length` driving load-back, and cascade eviction device->host.

That matters more here than on a full-attention model. 48 of our 64 layers are
GDN. A restored prefix that carried only the 16 full-attention layers' KV
would still need its linear-attention state recomputed, and the restore would
be close to worthless. Because the MAMBA component has its own host pool, a
restored prefix is complete.

**EAGLE is supported too.** `speculative/base_spec_worker.py` builds a
`HiCacheDraftPlan`: `PACKED` when `_can_pack_hicache_mtp` holds (EAGLE, not
EAGLE3, draft runners reporting `num_nextn_predict_layers` — our config has
`mtp_num_hidden_layers = 1`), otherwise `SIDECAR`, described as "the legacy
non-packed HiCache behavior". Either branch works; which one we land on
changes only how much of the 0.64 GB draft pool is covered.

`--enable-session-radix-cache` also survives: `registry.py` requires
`UnifiedRadixCache` for it, which is exactly what we build.
`--mamba-radix-cache-strategy extra_buffer` survives too —
`Qwen3_5ForConditionalGeneration` is in `_MAMBA_EXTRA_BUFFER_ARCHS` at
v0.5.19 and the strategy requires `linear_attn_backend == "triton"`, which is
what we run.

## Sizing

Device pools per replica, from the live boot log (v0.5.18):

| pool | size | granularity |
|---|---|---|
| target KV | **10.34 GB** | 169,408 tokens (K 5.17 + V 5.17) |
| EAGLE draft KV | 0.64 GB | (K 0.32 + V 0.32) |
| Mamba | **5.36 GB** | **43 slots** — conv 0.12 + ssm 3.09 + inter 2.11 + window 0.04 |

`hybrid_pool_assembler.py::build_hybrid_mamba_stack` is our path. It builds a
host pool per device pool:

- `--hicache-ratio R` sizes **each pool independently**: host = R x device.
- `--hicache-size G` (GB) *overrides* the ratio and is split **proportionally
  by device pool bytes** (`_split_hicache_size`) — 10.34/15.70 = 65.9% to KV,
  34.1% to Mamba.

So `--hicache-size` buys no independent control over the KV/Mamba balance; it
is the ratio expressed in gigabytes, with rounding. It is also the
less-exercised path — DeepSeek-V4 rejects it outright with *"use
`--hicache-ratio` instead"*. **Use the ratio.**

| `--hicache-ratio` | KV host | Mamba host | per replica | both replicas | effective cache |
|---|---|---|---|---|---|
| 2 (cache-mode default) | 20.7 GB | 10.7 GB | 31.4 GB | 62.8 GB | 3x device |
| **3 (recommended)** | **31.0 GB** | **16.1 GB** | **47.1 GB** | **94.2 GB** | **4x device** |
| 4 | 41.4 GB | 21.4 GB | 62.8 GB | 125.6 GB | 5x device |
| 6 | 62.0 GB | 32.2 GB | 94.2 GB | 188.4 GB | 7x device |

Add `R x 0.64 GB` per replica if the draft pool lands `PACKED` rather than
`SIDECAR` (+1.9 GB at R=3). Immaterial at this scale, but it is why the boot
figure may read slightly above the table.

### Why 3

**The host pool is pinned memory.** `memory_pool_host.py` defaults
`pin_memory=True` throughout, and it must — page-locked pages are what let the
transfer be an async DMA rather than a bounce-buffered copy. Pinned pages are
**unswappable and unreclaimable by the OS**. This is not "use what's free"; it
is a hard reservation.

Host budget: 1007 GB total, 820 GB available, 759 GB free — but `qwen3-emb`
alone currently holds **163 GB** and is the volatile neighbour. At R=3 we pin
~94-98 GB across both replicas, about 12% of host RAM, leaving >650 GB of
headroom including the embedder.

**Why not larger.** What the host tier buys is *retained evicted prefixes*,
and each increment retains a less-recently-used one, so the benefit curve
flattens while the pinned cost stays linear. Measure the hit rate at R=3 and
raise it on evidence. Going to R=6 on speculation pins 188 GB against an
embedder that already moves by >100 GB.

**Why not smaller.** R=2 is only 3x device. Our device pool already holds
169,408 tokens — about one maximum-length conversation. For the tier to change
behaviour under concurrency it has to hold several sessions' worth beyond
that, and 3x is the point where that starts being true.

**A note on which resource actually binds.** The device Mamba pool is 43
*slots* — 43 sequences, at ~127 MB each — while the KV pool is 169,408
*tokens*. Those two run out at different times: 169,408 / 43 ≈ **3,940
tokens**. Below that sequence length, Mamba slots bind; above it, KV tokens
bind. OpenCode sessions are far above 3,940 tokens, so KV is our binding
constraint and the proportional split is right, if marginally Mamba-heavy.
This is worth re-checking if the workload ever shifts to many short sessions.

## The settings, and why each stays where it is

```
--enable-hierarchical-cache            # the only new flag that turns it on
--hicache-ratio 3                      # 47 GB host per replica
```

Everything else stays at its default, deliberately:

| setting | value | why |
|---|---|---|
| `--hicache-storage-backend` | **unset** | this is what makes it L2-only. Setting it opens L3 |
| `--hicache-host-memory-mode` | `cache` | `buffer_only` is closed to MAMBA trees (above) and needs a storage backend |
| `--hicache-mem-layout` | `page_first` | see below |
| `--hicache-io-backend` | `kernel` | see below |
| `--hicache-write-policy` | `write_through` | best hit rate; `write_through_selective` is the knob if write traffic costs decode |
| `--page-size` | **64, unchanged** | see below |

**Layout x IO backend is not free choice.** `memory_pool_host.py` accepts
exactly four combinations and raises on anything else:

```
kernel + layer_first        kernel + page_first        <- ours
direct + layer_first        direct + page_first_direct
```

`kernel` + `page_first` is our current default pair, is legal, and
`page_first` additionally gates a fast path (`memory_pool_host.py:285`). Keep
it. If Phase 3 measures a decode regression, the alternative worth trying is
`direct` + `page_first_direct`: `direct` runs the transfer on the DMA copy
engine instead of as an SM-resident copy kernel. Both still read HBM, so
neither is free on a bandwidth-bound node — the difference is SM occupancy,
and since decode here is memory-bound rather than SM-bound, `kernel` should
cost little. That is a prediction, not a measurement.

**Do not touch `--page-size`.** It is the *global* KV page size, not a
HiCache knob. Changing it re-lays-out the device KV pool, moves
`max_total_num_tokens`, and invalidates every baseline in `tuning/results/`.
At 64 a page is already 64 tokens x 64 KiB = **4 MiB** of KV, well past the
efficiency knee for PCIe DMA, so there is nothing to win. (The
`page_size=256 % 128` comment in the assembler is DeepSeek-V4 C128-specific
and does not apply.)

## What this costs, and what it can return

**PCIe is the ceiling.** These are A100 **80GB PCIe** cards, Gen4 x16 —
~25 GB/s theoretical, ~21-24 GB/s realistic with pinned memory. Not SXM,
so there is no NVLink host path.

The trade is nonetheless lopsided. Restoring a 169,408-token prefix means
moving ~10.3 GB over PCIe: **~0.5 s**. Recomputing it means a full prefill of
169K tokens, which at our measured 42K-token TTFT of ~12.4 s extrapolates to
**tens of seconds**. Even at a fraction of a full context the reload wins by
an order of magnitude. That is the whole case for the feature, and it is why
it should help exactly where the user expects — as query volume grows and
sessions start evicting each other.

**GPU memory cost should be nil** — the pool is host RAM. But HiCache also
registers a layer-done transfer counter and controller machinery on the
device side, and our headroom is `available_gpu_mem=8.35 GB`. Verify, do not
assume.

**Boot will get slower.** Allocating and pinning ~47 GB per replica is not
free. Helpfully, v0.5.19 contains **#36705, "stop populating host-pool mmaps
twice (-13% allocation time)"** — the release we are rolling includes a
speedup for the exact thing we are turning on.

---

# The plan

Same shape as v0.5.17 and v0.5.18: engine only first, one replica at a time,
gated on the boot log and on a direct-to-worker A/B while the two replicas are
deliberately on different builds. HiCache is a **separate phase** and does not
ride along.

The reason for the split is on the record. On 2026-08-24 two optional flags
were staged separately from the engine roll; one of them
(`--startup-weight-load-mode overlap`) turned out to be flatly incompatible
with speculative decoding and failed at boot. Because it was not bundled, the
blast radius was one already-deregistered replica. HiCache is a much larger
change than that flag was.

## Phase 0 — before touching anything

1. Pull on the host (does not disturb the running stack):
   ```bash
   docker pull lmsysorg/sglang:v0.5.19-cu130
   docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:v0.5.19-cu130
   # must print sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
   ```
2. **Confirm torch.** The label does not carry it:
   ```bash
   docker run --rm --entrypoint python3 lmsysorg/sglang:v0.5.19-cu130 \
     -c "import torch,flashinfer;print(torch.__version__, flashinfer.__version__)"
   # expect 2.13.0+cu130 and 0.6.18
   ```
   A torch minor move here would change the risk profile of this roll
   entirely and re-opens the full ladder.
3. Capture the v0.5.18 baseline from **both** replicas while both are still on
   the old build, and save `/get_server_info` from both:
   ```bash
   ./benchmarks/run_worker.sh r0 v0518_pre_519
   ./benchmarks/run_worker.sh r1 v0518_pre_519
   ```
   This is the leg that was skipped last time. **It is the whole reason
   v0.5.18 is recorded as performance-unproven.** Do not skip it, and do not
   start if the traffic window is not quiet.

## Phase 1 — roll r1 only, and hold there

Edit the `x-sglang-image` anchor to the new digest, then:

```bash
./deploy/roll-replica.sh r1
```

r0 stays on v0.5.18 and keeps serving. This is the A/B window.

**Boot gates on r1 — all four must hold:**

| Check | Expected |
|---|---|
| `max_total_num_tokens` | **169408** (see #36583; a move *up* is acceptable, a move down below 169000 is a stop) |
| decode CUDA-graph `bs` | `[1, 2, 3, 4]` |
| `max_mamba_cache_size` | **43** (conv 0.12 / ssm 3.09 / inter 2.11 / window 0.04 GB) |
| boot to healthy | ~181 s (first boot longer: the kernel cache recompiles once) |

Then diff runtime server args between the builds — the check v0.5.17 taught us:

```bash
docker exec qwen36-27b-r0 sh -c 'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
  http://localhost:8001/get_server_info' > /tmp/si_r0_518.json
docker exec qwen36-27b-r1 sh -c 'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
  http://localhost:8002/get_server_info' > /tmp/si_r1_519.json
diff <(jq -S .server_args /tmp/si_r0_518.json) <(jq -S .server_args /tmp/si_r1_519.json)
```

**Pre-registered prediction for that diff** (anything else is a silent
resolution change and a stop condition):

- `port` (8001 vs 8002), `version` 0.5.18 -> 0.5.19
- **new:** `hicache_host_memory_mode='cache'`,
  `hicache_storage_prefetch_retry_poll_interval=0`,
  `hicache_storage_prefetch_retry_max_attempts=4`, the beam-search and
  mixed-chunk-prefill keys, `enable_w4a4_megamoe`,
  `speculative_dsa_topk_backend`, `enable_layernorm_sp`
- **unchanged and load-bearing:** `mm_feature_transport == "cpu"` (the
  auto-resolution has now changed twice; the explicit pin is what made it a
  non-event), `mamba_radix_cache_strategy == "extra_buffer"`,
  `uses_mamba_radix_cache == True`, `enable_session_radix_cache == True`,
  `mem_fraction_static == 0.92`

Also confirm the GDN gate actually flipped — this is the change we are here
for, and it is directly observable:

```bash
docker exec qwen36-27b-r1 sh -c \
  'grep -n "_GDN_FUSED_QKVZBA_RATIOS" /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py'
# expect: (1, 2, 4, 8) if _use_aiter else (1, 2, 3, 4) if _is_cuda else (1, 2, 4)
```

## Phase 2 — the A/B gate (r1 v0.5.19 vs r0 v0.5.18)

Hit workers **directly** on :8001/:8002, never through the router. **Flush
both first** (`POST /flush_cache`) and measure in both orderings.

> The flush is not optional and it is not cosmetic. On 2026-08-24 a skipped
> refusal produced 0/8 on byte-identity and cost a diagnosis cycle:
> `/flush_cache` answers `Flush cache failed.` whenever the replica has
> running or waiting requests, which a production replica always does. Retry
> until it answers `Cache flushed.` The untouched old build scores 1/8 against
> *itself* with a warm cache.

| Gate | Pass condition |
|---|---|
| `byte_identity.py` greedy, 8 prompts x 256 tok, `ignore_eos`, temp 0 | **8/8 identical.** #34859 changes data movement, not layout or arithmetic |
| `spec_accept_length` converged | ~5.0, unchanged |
| `worker_ladder.py` c=1..12 | mean within the **1.5%** noise floor |
| TTFT, 42K prompt | ~12.4 s — **this is where #34859 should show if it shows at all** |
| single-stream decode | ~70.5 tok/s |

**The hypothesis under test, stated in advance:** #34859 removes two
`.contiguous()` copies and a `torch.cat` from 48 of 64 layers per forward.
Predicted effect is a small reduction in bytes moved, ≤1%, most visible on
prefill/TTFT and probably invisible on decode. A decode regression beyond the
noise floor would suggest the narrow per-head walk costs more than the cat it
replaced — which is a real possible outcome and the reason this is measured
rather than assumed.

## Phase 3 — converge, then HiCache separately

1. `./deploy/roll-replica.sh r0`; re-verify the four boot gates; confirm
   `docker compose config` shows the two command blocks differing only by
   port. Record the measurements back into this file. **Engine roll is done
   here.**
2. Add HiCache to **r1 only**:
   ```
   --enable-hierarchical-cache
   --hicache-ratio 3
   ```
   Gates:

   | Check | Expected |
   |---|---|
   | tree-cache line | `impl=UnifiedRadixCache hybrid_ssm=True hicache_attached=True` |
   | `max_total_num_tokens` | **unchanged** — the pool is host RAM |
   | `available_gpu_mem` | ~8.35 GB, not materially lower |
   | host RSS delta on r1 | ~+47 GB, pinned |
   | boot to healthy | longer; record how much |
   | byte-identity vs r0 | **8/8** — a cache tier must not change output |

   The tree-cache line is the single decisive one. `registry.py` logs it at
   startup and it proves HiCache attached *with the Mamba component present*:

   ```
   Tree cache initialized: source=default impl=UnifiedRadixCache
     hybrid_swa=False hybrid_ssm=True hicache_attached=True streaming_wrapped=False
   ```

3. Hold r1 on HiCache and r0 without it for a real traffic window. The metric
   that decides whether R=3 is right is the **host hit rate** and the TTFT
   distribution on returning sessions, not a synthetic ladder — the feature
   does nothing on a cold single-shot benchmark, which is exactly why it needs
   live traffic to evaluate. Then converge r0 or back it out.

## Rollback

Restore the `x-sglang-image` anchor to
`sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1`
and roll the affected replica. Because only r1 moves in Phase 1, rollback is
one `roll-replica.sh r1` and traffic never leaves a healthy worker. HiCache
backs out by deleting two flags — it adds no persistent state.

## Open item carried forward

**Corrected 2026-09-05 — this item is closed.** `UPGRADE_v0.5.18.md` recorded
the router as still running the v0.5.17 image (`16aba892...`) because
`roll-replica.sh` does not touch it. Verified today, that is **stale**: the
router runs `sha256:9e148f5a...`, byte-identical to the `x-sglang-image` anchor,
i.e. v0.5.18. Something recreated it in the interim (most likely a
`docker compose up -d` during the metrics campaign). There is no router drift.

What still holds is the operational note in `docs/OPERATIONS.md`: a router
restart **drops in-flight requests on both replicas**, and it is not needed
after a roll because the router tracks workers by URL and re-adds a returning
worker itself. So any change to the router's own command block — including the
routing-policy work in `ROUTING.md` — must be taken in a quiet window
deliberately, not folded into a replica roll.

v0.5.19 does carry Rust-server changes (#33370 process-local KV indexer and
router integration, #35125, #37221, #37222, #36920). None is required. The
reason to eventually refresh the router is #34608 — the per-scheduler load
socket — because every load-aware policy currently prices workers from a
router-side in-flight counter that misses direct-hostname traffic.

## Sources

Release notes (GitHub API, raw body), Docker Hub tag list and registry config
blobs for both tags, PR bodies and file diffs (#34859, #33431, #35297, #35375,
#35588, #36583, #34053, #35158, #35177, #33091, #24911, #34608), and the
v0.5.19 source at tag: `server_args.py`, `arg_groups/overrides.py`,
`mem_cache/registry.py`, `mem_cache/unified_radix_cache.py`,
`mem_cache/unified_cache/components/mamba_component.py`,
`mem_cache/memory_pool_host.py`,
`mem_cache/hybrid_cache/hybrid_pool_assembler.py`,
`speculative/base_spec_worker.py`. Live state read from `/get_server_info`,
the r0 boot log, and the model `config.json`.
---

# Execution log

## Phase 0 — 2026-09-05

### Steps 1 and 2 — PASS

```
digest      sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9   exact match
sglang      0.5.19
torch       2.13.0+cu130     <- UNCHANGED from v0.5.18, as predicted
flashinfer  0.6.18
image       52.2 GB unpacked (v0.5.18: 47.4 GB); 1011 GB free on /
```

The torch question is settled: this is a one-dependency release for us
(FlashInfer 0.6.17 -> 0.6.18). That is a materially smaller risk surface than
v0.5.18, which moved torch two minors.

**The GDN gate is confirmed present in the pulled image:**

```python
# /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py
_GDN_FUSED_QKVZBA_RATIOS = (
    (1, 2, 4, 8) if _use_aiter else (1, 2, 3, 4) if _is_cuda else (1, 2, 4)
)
```

> **Methodology note.** A first check ran `python3 -c "print(m._GDN_FUSED_QKVZBA_RATIOS)"`
> in a container started **without `--gpus`**, which printed `(1, 2, 4)`. That is
> the no-CUDA fallback branch (`_is_cuda` is False with no device visible), not
> the value the engine will use. Read the source literal, or evaluate it on a
> replica that actually has a GPU. The live-replica check is already in Phase 1.

### Step 3 — baselines are CONTAMINATED. Do not use them as a control.

Five runs were taken. They do not agree, and the reason is the useful part.

| run | cache | decode_tok_s_mean | long TTFT | accept after |
|---|---|---|---|---|
| r0 #1 | warm (production) | 64.25 | 12.364 s | 3.025 |
| r1 #1 | warm (production) | 69.47 | 12.587 s | 3.05 |
| r0 #2 | **warm, self-warmed** | 72.98 | **0.387 s** | 3.35 |
| r0 #3 | **flushed** | 44.16 | 12.426 s | 3.329 |
| r1 #2 | **flushed** | 46.30 | 12.683 s | 3.838 |

**Three separate defects, all in the measurement, none in the engine.**

1. **The harness warms its own cache.** r0 #2 reports a 42K-prompt TTFT of
   **0.387 s** against 12.4 s cold — the long prompt was served from the radix
   tree left behind by r0 #1. Any second run against the same replica measures
   cache state.

2. **`decode_tok_s_mean` is not a decode metric.** It is
   `completion_tokens / dt` where `dt` spans the whole request, so a cold
   prefill is charged to the decode rate. That is why flushing *lowers* it by
   ~35% (64-73 warm vs 44-46 flushed) while long-prompt TTFT stays put. It
   cannot be compared across runs with different cache states, which is
   exactly what a pre/post upgrade A/B is.

3. **The node is not idle, and "idle" cannot be established by one sample.**
   `sglang:num_running_reqs` on r0 read 0 before the runs and **1.0** during
   them; live `spec_accept_length` was 5.05 on r0 against 3.18 on r1, i.e.
   different concurrent workloads. Production traffic arrives in bursts, lands
   through the router, and shares the GPU with the benchmark.

**The one metric that survived all five runs is long-prompt TTFT:** 12.364,
12.587, 12.426, 12.683 — a 2.6% spread, sitting on the historical 12.426 /
12.398 from the v0.5.18 baseline. It is cold-cache-dominated, long enough that
launch and scheduling noise is amortised, and it is *the leg where #34859 is
predicted to show*.

### Consequences for Phase 2 — the gate changes

- **Gate #34859 on `long_prompt_stream.ttft_s`, not on `decode_tok_s_mean`.**
  The latter has a ~66% observed range on one replica and one build; the
  effect under test is ≤1%. It cannot resolve it and never could.
- **Drain the replica from the router before measuring.** This is the
  project's own established procedure — `RESULTS.md` 2026-07-31c and both
  `lowlevel/` experiments drained r1 via `DELETE /workers/<id>` and stopped the
  DCGM exporter for CUPTI contention. Phase 0 skipped it and paid for it. In
  Phase 1 this is nearly free: `roll-replica.sh` already deregisters r1.
- **Flush immediately before every leg**, and never measure the same replica
  twice without re-flushing.
- Report `spec_accept_length` beside every number. It is prompt- and
  cache-dependent and it moves decode rate directly.

Nothing here blocks the roll. It means the Phase 0 numbers are discarded as a
control, and the real comparison happens in Phase 2 with r1 drained — which is
a better experiment than the one originally planned.

---

# HiCache — RETRACTED as a Phase 3 action

**The recommendation earlier in this document to enable HiCache at
`--hicache-ratio 3` is withdrawn.** It was written before finding that
**HiCache has already been run on this node and removed on measurement.**

`tuning/docs/TUNING_PLAN.md` §Phase 1a, 2026-07-31, ran:

```
--enable-hierarchical-cache  --hicache-size 128
--hicache-io-backend direct  --hicache-mem-layout page_first_direct
--hicache-write-policy write_through
```

and measured:

| | |
|---|---|
| host share of cache hits | **0.43%** (device 7,496,896 / 8,026,048 tokens; host 32,768 / 34,816) |
| host RAM | **264 GB per replica, ~506 GiB total** |
| boot cost | **+4 min per replica** (~9 min, vs ~5 without) |

Removing it was Phase 1: *"highest value, lowest risk."*

**What v0.5.19 genuinely fixes.** Part of that 264 GB was a sizing footgun:
`--hicache-size 128` was applied *per pool* (KV 128 + Mamba 128 + draft 8),
which `TUNING_PLAN.md` records as "wrong by ~2x". In v0.5.19,
`_split_hicache_size` divides one budget proportionally across pools, and the
`--hicache-ratio 3` proposal above is 47 GB per replica — **5.6x smaller** than
what was run. The cost side is real progress.

**Why that does not rescue it.** The failure was not cost, it was **hit rate**,
and a smaller pool cannot raise a hit rate. 0.43% was measured with a host tier
roughly **12x the size of the device pool**. Shrinking it 5.6x can only move
that number down. Fixing routing to give session affinity roughly doubles the
population of useful host hits — call it 0.9% — which is still nothing.

**The actual cause is structural.** The device tier is 169,408 tokens per
replica. Under `--max-running-requests 4` the working set rarely exceeds it, so
little is evicted; and what is evicted is rarely requested again. A host tier
only earns its memory when the *device* tier is under eviction pressure.

**The trigger is measurable, so measure it instead of guessing.** HiCache
becomes worth re-testing when device-tier eviction is frequent *and* evicted
prefixes are being re-requested. The engine already exposes what is needed:

```
sglang:mamba_evictable_tokens      22 of 43 slots free to evict right now
sglang:mamba_used_tokens            4  (= 4 slots per running request)
sglang:token_usage                  0.42
sglang:cache_hit_rate
```

Watch `token_usage` and `mamba_evictable_tokens` under real load. While
`token_usage` sits near 0.42 and 22 of 43 Mamba slots are evictable, the device
tier is not full and HiCache has no work to do. **Revisit when sustained
`token_usage` approaches 1.0.**

Note also that HiCache is *not* free of the routing problem: under
`round_robin` a returning session reaches the replica holding its host copy
only half the time. **Routing affinity is a prerequisite for HiCache, not a
companion to it.** See `tuning/docs/ROUTING.md`.

## Phase 1 — r1 rolled to v0.5.19, 2026-09-05. ALL GATES PASS.

### First: `roll-replica.sh` was broken and had to be fixed

The script's `rcurl()` called the router control plane with no credential.
Since the admission-control hardening on 2026-09-04 added
`--control-plane-api-keys`, every `GET /workers` returns **401**:

```
$ docker exec qwen36-27b-router curl -s -m 8 http://localhost:8000/workers
Missing or invalid Authorization header          # HTTP 401
```

`worker_field()` then yields empty and the preflight aborts with a misleading
*"peer r0 is not healthy in the router (got 'absent')"*. It fails **safe** —
it exits before touching anything — but the documented safe-roll path had been
broken since 2026-09-04 and nobody had rolled since to notice.

Fixed by sourcing `.env` and adding the bearer token, the same pattern
`benchmarks/run_worker.sh` already uses. Verified: `GET /workers` -> 200.

### Boot gates — 4/4

| Check | Expected | Measured | |
|---|---|---|---|
| `max_total_num_tokens` | 169408 | **169408** | PASS — #36583 did not move it |
| decode CUDA-graph `bs` | `[1,2,3,4]` | `[1,2,3,4]` | PASS |
| `max_mamba_cache_size` | 43 | **43** (conv 0.12 / ssm 3.09 / inter 2.11 / win 0.04) | PASS |
| boot to healthy | ~181 s | **181 s** | PASS |

`available_gpu_mem` **8.35 -> 8.49 GB**, i.e. 0.14 GB *more* headroom. KV pool
byte-identical (K 5.17 + V 5.17; draft 0.32 + 0.32). The 408-token margin over
`--context-length 169000` is unchanged.

The tree-cache line confirms the cache stack, and is the exact line the HiCache
section says to check:

```
Tree cache initialized: source=default impl=UnifiedRadixCache
  hybrid_swa=False hybrid_ssm=True hicache_attached=False streaming_wrapped=False
```

### `/get_server_info` diff — matched the pre-registered prediction exactly

Excluding `internal_states` and the API key, **only four keys differ**:

```
port          8001 -> 8002        (the two replicas)
random_seed   ...  -> ...         (per boot)
startup_time  ...  -> ...         (per boot)
version       0.5.18 -> 0.5.19
```

**16 new keys, 0 gone, and nothing resolved differently.** New keys:
`deepep_v2_mode`, `dsv4_prefill_backend`, `enable_dense_mlp_attn_tp`,
`enable_layernorm_sp`, `enable_lean_attention`, `enable_shared_experts_attn_tp`,
`enable_w4a4_mxfp4_megamoe`, `gated_launch_port`, `grpc_worker_threads`,
`hicache_host_memory_mode`, `hicache_storage_prefetch_retry_max_attempts`,
`hicache_storage_prefetch_retry_poll_interval`,
`http2_initial_connection_window_size`, `load_publish_endpoint`,
`prefill_decode_interval`, `speculative_dsa_topk_backend`.

Every load-bearing invariant held, `mm_feature_transport == 'cpu'` included —
the flag that flipped underneath us in v0.5.17 has now been a non-event for
three releases running because it is pinned explicitly. Also unchanged:
`mamba_radix_cache_strategy=extra_buffer`, `uses_mamba_radix_cache=True`,
`enable_session_radix_cache=True`, `mem_fraction_static=0.92`,
`attention_backend=flashinfer`, `linear_attn_backend=triton`, `page_size=64`.

### The change we rolled for is live

Verified on the running replica, where `_is_cuda` is genuinely True:

```
$ docker exec qwen36-27b-r1 python3 -c "import sglang.srt.models.qwen3_5 as m; ..."
_is_cuda       True
_use_aiter     False
GDN ratios     (1, 2, 3, 4)
ratio 3 fused  True
```

Our head-group ratio 3 is on the fused QKVZBA path across 48 of 64 layers.
On r0 (v0.5.18) it is still `(1, 2, 4)` and still taking the unfused fallback.
**That asymmetry is the experiment.**

### A prediction from the v0.5.18 doc, now measured

`UPGRADE_v0.5.18.md` predicted, but could not confirm, that r1 is never
NUMA-bound because its cpuset (28-47, node 1) has an empty intersection with
node 0, which NVML reports for both GPUs. The v0.5.19 boot log states it
outright:

```
Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.
NUMA node 0 has no CPU cores allowed by the current affinity
  [28, ..., 47], skipping NUMA binding for GPU 0.
```

Prediction confirmed. r1 is not NUMA-bound, and never was.

### Current state — the A/B window is open

```
qwen36-27b-r0       ...4823a7c29a1a1   v0.5.18   <- control
qwen36-27b-r1       ...4c4385ab3eda9   v0.5.19   <- treatment
qwen36-27b-router   ...4823a7c29a1a1   v0.5.18
router              2/2 healthy, both load 0
prometheus          14/14 targets up, 0 alerts
```

**Do not run a bare `docker compose up -d` while this window is open** — the
anchor now points at v0.5.19, so it would recreate r0 and the router too and
destroy the control.

### Operational note, pre-existing but worth recording

The engine logs its resolved `server_args` at startup **including `api_key` in
plaintext**. Those logs go to the json-file driver (50 MB x 5) and are readable
by anyone with docker access or the log mount. Not introduced by this upgrade,
and not a reason to hold it, but it belongs in the security notes.
