# SGLang v0.5.20 upgrade — 2026-09-19

**STATUS: APPLIED 2026-09-19** — r0, r1 and the router on v0.5.20. Engine only, one replica at a time. No flag changes
ride along; the cookbook levers at the end are separate, measured follow-ups.

Target image:

```
lmsysorg/sglang:v0.5.20-cu130
sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f   (index)
sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4   (amd64)
```

Released 2026-09-18. 713 PRs / 237 contributors; 727 commits past our 0.5.19
pin (`0bcd822377`).

## Provenance check

Read from the registry config blob, not from a running container.

| | v0.5.19-cu130 (live) | v0.5.20-cu130 |
|---|---|---|
| `ai.sglang.build.commit` | `0bcd8223…` | `94602c9c…` (== tag commit) |
| build URL | actions run 33912440803 | actions run 35322364074 |
| CUDA / cuDNN / NCCL | 13.0.3 / 9.14.0.64 / 2.28.3 | **unchanged** |
| FlashInfer | 0.6.18 | **0.6.18** |
| torch (pyproject) | 2.13.0 | **2.13.0** |
| transformers (pyproject) | 5.12.1 | **5.12.1** |
| sglang-kernel | 0.4.6.post1 | 0.4.7 |
| amd64 layers / size | 70 / 15.10 GB | 70 / 15.07 GB |

The CUDA 12 lane is retired in this release. Not relevant: we have run the
cu130 image since v0.5.17, host driver 610.57.04.

## Why roll

### #37818 — DFlash GDN checkpoints at tracking boundaries

Our exact configuration: `--speculative-algorithm DFLASH` +
`--mamba-radix-cache-strategy extra_buffer` + `--mamba-track-interval 64` on a
GDN hybrid.

At the 0.5.19 pin, `DFlashWorkerV2._update_target_mamba_state_after_verify`
(`speculative/dflash_worker_v2.py:1552`) decides whether a verify step crossed
a tracking boundary with

```python
seq_lens_pre_verify // mamba_track_interval != batch.seq_lens // mamba_track_interval
```

but `batch.seq_lens` has not been advanced yet at that point, so both sides are
the same length and the mask is always false. The scheduler's result handling
does see the crossing and advances `mamba_last_track_seqlen`, so a radix entry
can end up with full-attention KV at one length and linear-attention state
from an earlier one. No crash, no log line: a later request that matches that
prefix continues from the wrong GDN state. v0.5.20 compares against
`seq_lens_post_verify` (line 1977). Upstream issue #37817 reproduced it on
Qwen3.6-35B-A3B with the same strategy.

How often it bit us is not measured. It needs a later prompt that re-matches
decoded tokens past a checkpoint, which agent tool loops do.

### The abort fix is upstream

`f478b2bb2d` (#35255) is an ancestor of v0.5.20 (API compare: 551 ahead, 0
behind). The bind-mounted backport from 2026-09-16 (a930c99) is removed from
both replicas in the same change. **The mounts must not survive onto the new
image**: they would overwrite v0.5.20's `tokenizer_manager.py` and
`scheduler.py` with the 0.5.19 files.

### Smaller, on our path

- #36267: Qwen3.5 GDN prefill keeps QKV/Z/B/A as strided views on CUDA
  extend (`models/qwen3_5.py`, our class `Qwen3_5ForConditionalGeneration`).
  Data movement only; expect ≤1%.
- #39120: the multimodal embedding cache could retain whole batch
  allocations through `torch.split` views beyond its budget.

## Audited against our config

- **Every live flag still exists** — all engine flags in the v0.5.20 field
  declarations, all router flags in `sgl-model-gateway`. The 17 removed
  deprecated flags are none of ours.
- **Router**: the only Rust change in `sgl-model-gateway/src` is
  `pd_router.rs`. The header allow-list, `x-smg-*` routing keys and
  `cache_aware` are unchanged.
- **Image preprocessing still runs on `cuda:{base_gpu_id}`**
  (`base_processor.py:_fast_image_processor_device`), so
  `--image-processor-backend pil` stays. The new
  `SGLANG_FORCE_CPU_IMAGE_PREPROCESSING` (#39148) is Kimi-only.
- **Metering**: `request_metrics_exporter.py` differs by one line (where the
  export dir is read from); `extract_custom_labels` reads the same header.
  Verified end to end as a boot gate below, not assumed.
- **`/v1/responses`**: storage is now off unless `--enable-response-store`.
  `previous_response_id` and `background` return 400; `store: true` is
  accepted and not persisted. The gateway already offers creation only and
  asks for `store: false` (`higress-standalone/config/ingresses/ai-responses.yaml`).
- `/get_server_info` keys, predicted from the field declarations:
  - gone: `custom_sigquit_handler`, `stat_loggers`, `kv_canary_real_data`,
    `enable_prefill_context_parallel`, `prefill_cp_mode`,
    `enable_dsa_prefill_context_parallel`, `dsa_prefill_cp_mode`
  - new: `enable_response_store`, `sampling_mask_max_tokens`,
    `return_input_ids`, `return_output_ids`, `radix_eviction_policy_config`,
    `dsv4_attn_backend`, `flashinfer_a2a_dispatch_type`, `ple_offload_*`,
    `speculative_domino_candidate_pool_size`, `uno_lora_path`,
    `unified_cache_external_linker_*`
  - Any other value change is a silent resolution change and a stop.

# The plan

The v0.5.19 plan gated on a drained, flushed direct-to-worker ladder. This roll
is a correctness fix on an identical platform, so it gates on boot numbers,
the server-args diff, functional smoke tests and live-traffic latency per
replica during the A/B window, and does not load production with synthetic
benchmarks.

## Phase 0 — before touching anything

1. Pull and verify the digest (does not disturb the stack):
   ```bash
   docker pull lmsysorg/sglang:v0.5.20-cu130
   docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:v0.5.20-cu130
   ```
2. Confirm torch / FlashInfer / transformers inside the pulled image.
3. Save `/get_server_info` from both replicas on 0.5.19 (drop `api_key`
   before writing it anywhere).

## Phase 1 — roll r1 only, and hold there

```bash
./deploy/roll-replica.sh r1
```

**Boot gates on r1:**

| Check | Expected |
|---|---|
| `max_total_num_tokens` | **182528** (a move down below 169000 is a stop) |
| `max_mamba_cache_size` | **43** (conv 0.12 / ssm 3.09 / intermediate 2.81 GB) |
| decode CUDA-graph `bs` | `[1, 2, 3, 4]` |
| tree-cache line | `hybrid_ssm=True hicache_attached=True` |
| boot to healthy | ~181 s (first boot may recompile the kernel cache) |
| server-args diff | only the keys predicted above, plus `version`, `port` |
| no bind mounts | `docker inspect qwen36-27b-r1` lists no `patches/` source |

**Smoke on r1, direct to the worker:** a plain chat, a tool call
(`qwen3_coder` → structured `tool_calls`), reasoning split into
`reasoning_content`, one 12 MP image (expect 2,028 image tokens), 17 images
(expect the 400 "Image count 17 exceeds limit 16"), and a streamed request
cancelled mid-decode (expect the scheduler to abort it, not decode on).

**Metering:** one request through the gateway hostname that lands on r1
appears in `engine.requests` with the right consumer and token counts.

## Phase 2 — A/B window on live traffic

Hold with r1 on 0.5.20 and r0 on 0.5.19. Compare per replica from
`engine.requests` and the engine metrics: TTFT, decode tok/s, spec accept
length, error counts. Stop on a regression beyond noise or on any new error
class in the r1 log.

## Phase 3 — converge

```bash
./deploy/roll-replica.sh r0
```

Same boot gates. Then the router, which shares the image anchor, at a quiet
moment (`num_running_reqs = 0` on both workers; a router restart drops
in-flight streams). No overlay file defines it, so the base file is enough:

```bash
docker compose up -d --no-deps qwen36-27b-router
```

Router source is unchanged for us, so this is only to stop drift: the next
bare `docker compose up -d` would otherwise do it at a random moment.

Then delete `patches/sglang-abort-fix/`.

## Rollback

Restore the anchor to
`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`,
**restore both abort-fix bind mounts from a930c99** (0.5.19 without them has
the orphaned-request stalls back), and roll the affected replica.

# Cookbook levers — follow-ups, not part of this roll

The Qwen3.8-27B cookbook page (checked 2026-09-19) has no A100 row; its cells
are SM12x and H200, validated on v0.5.19 with GSM8K at ISL 8192 and
concurrency 1. Our 7-day mix: the direct batch loop is 92% of requests at
~3.6K tokens (input + output); agent keys average 65-72K. At 182,528 KV tokens
the pool holds ~2.5 average agent requests, so KV, not GDN state, is what binds
agents.

In order:

1. **Pin `--max-mamba-cache-size`.** The default ratio 0.9 gives 43 state
   slots; 4 running x 5 (`extra_buffer`) need 20, the rest hold cached prefix
   states (HiCache also keeps ~10 GB of them in host RAM). ~24 would move
   ~1.4 GB to KV (+~21K tokens). `extra_buffer_lazy` (4 slots per request,
   DFlash supported) pairs with it. Same idea as the §5f retry in
   `RESULTS.md`.
2. **fp8 KV** (`--kv-cache-dtype fp8_e4m3`, every cookbook recipe). Doubles KV
   and cuts decode bytes at long context. Declined before on uncalibrated
   numerics (`NVIDIA_KERNEL_TUNING.md` §4); the cookbook's short-context GSM8K
   does not cover that risk, so it still needs a long-context quality gate.
3. **`--chunked-prefill-size 2048`** (cookbook default). Smoother decode under
   long prefills against slower TTFT on 70K prompts. Measure.

# Execution log

## Phase 0 — 2026-09-19. PASS

- Pulled; `RepoDigests` = `sha256:06e4f2ed…d611f`, matches the anchor.
- Inside the image (`--network none`): torch `2.13.0+cu130`, FlashInfer
  `0.6.18`, transformers `5.12.1`, sglang `0.5.20`, sglang-kernel `0.4.7`.
- `dflash_worker_v2.py:1979` carries the #37818 fix;
  `tokenizer_manager.py` is byte-identical to the tag source (abort fix in).
- `/get_server_info` saved from both replicas on 0.5.19.
- Smoke against r0 on 0.5.19, 5/5: chat, reasoning split, `qwen3_coder` tool
  call, 12 MP image (2,091 prompt tokens, text transcribed), 17 images → 400.
  A stream closed after 30 chunks: `num_aborted_requests_total` 1, no
  "state was deleted" line (the backport working).

## Phase 1 — r1 rolled to v0.5.20, 2026-09-19 13:12. ALL GATES PASS

`roll-replica.sh r1`: drained, recreated, healthy after **252 s**, re-registered.
The extra ~70 s is prefill CUDA-graph capture (44.0 s vs 20.2 s on 0.5.19,
`startup_time.cuda_graph.prefill`), first boot on the new kernel cache.

| Check | Result |
|---|---|
| `max_total_num_tokens` | 182528 — unchanged |
| `max_mamba_cache_size` | 43 (conv 0.12 / ssm 3.09 / intermediate 2.81 / window 0.05 GB) — unchanged |
| decode / verify graph `bs` | `[1, 2, 3, 4]` |
| tree cache | `UnifiedRadixCache hybrid_ssm=True hicache_attached=True` |
| `available_gpu_mem` after capture | 2.80 GB (0.5.19: 2.91) |
| bind mounts | none from `patches/` |

**Server-args diff (r1 0.5.19 → 0.5.20):** `version`, `random_seed`,
`startup_time`, the predicted new keys, plus `launch_command` (new; it echoes
the full command line including `--api-key`, so strip it before saving a
snapshot). Gone: the four prefill-CP keys; `custom_sigquit_handler`,
`stat_loggers` and `kv_canary_real_data` are still reported. One unpredicted
default change, inert here: `hicache_storage_prefetch_retry_max_attempts`
4 → 8 and `hicache_storage_prefetch_retry_poll_interval` 0 → 8 (#39283) apply
to an L3 storage backend, and `hicache_storage_backend` is null. Everything
load-bearing held: `mm_feature_transport cpu`, `image_processor_backend pil`,
`mm_process_config`, `limit_mm_data_per_request`, `extra_buffer`,
`mamba_ssm_dtype bfloat16`, `mem_fraction_static 0.94`, DFLASH + draft path
and revision, parsers.

**Smoke, direct to r1: 5/5**, same answers as r0 on 0.5.19 (12 MP image →
2,091 prompt tokens, invoice text transcribed). The stream closed after 30
chunks raised `num_aborted_requests_total` 0 → 1 with no "state was deleted"
line: the upstream abort fix works without the backport.

**Metering:** a request carrying `x-request-id-labels {"consumer":"testafter"}`
landed in `engine.requests` as `consumer=testafter replica=r1` 58/28 tokens,
and in `sglang:prompt_tokens_total{consumer="testafter"}`. (Six gateway probes
all routed to r0 by cache affinity; the gateway and router are unchanged, the
engine side is what moved.)

"Invalid HTTP request received" ×2 at boot is pre-existing: r0 on 0.5.19 logs
the same pair at the same point after startup.

## Phase 2 — skipped

Operator call: r0 rolled right after the r1 gates passed, without a live A/B
hold. Traffic was near zero, so a hold would have measured little, and r0 was
still carrying the #37818 bug. Consequence: **no performance comparison
exists for this roll.** The platform is identical and the pools are
byte-for-byte the same size, so none is expected; if one is wanted later,
compare `engine.requests` TTFT / decode by `replica` before and after 13:12.

## Phase 3 — converged, 2026-09-19 13:26. PASS

`roll-replica.sh r0`: healthy after **251 s**. Boot figures identical to r1
(43 slots, 182528 tokens, graphs `[1, 2, 3, 4]`, HiCache attached, 2.80 GB free).
Server args: r0 0.5.19 → 0.5.20 differs only in `version` and the two inert
`hicache_storage_prefetch_retry_*` defaults; r0 vs r1 on 0.5.20 differ only in
`port`. Smoke 5/5; the cancelled stream aborted (counter 0 → 1, no orphan
line).

Router recreated at `num_running_reqs = 0 / num_queue_reqs = 0` on both
workers (`docker compose up -d --no-deps qwen36-27b-router`): healthy, same
image, `--policy cache_aware --balance-abs-threshold 1` preserved, both workers
activated from `--worker-urls`. Through the gateway: 200 in 0.83 s. The
`No tokenizer_path or model_path found for model unknown` WARN at start is the
known tokenizer-free-routing message (`ROUTING.md`).

`patches/sglang-abort-fix/` deleted. Final state:

```
qwen36-27b-r0  qwen36-27b-r1  qwen36-27b-router   sha256:06e4f2ed…d611f   v0.5.20
```
