# SGLang v0.5.22 upgrade — 2026-10-19

**STATUS: APPLIED 2026-10-19 12:39Z.** r0 and r1 run v0.5.22 with engine patches 1-6 + R
and share the L3 dir `/dev/shm/qwen36-hicache-l3-v0522-fp8w/bf16`.
The router stays on the v0.5.21 image (same sglang-router 0.3.2). ΔNLL was not measured
(see the gates section); the same-day v0.5.21 control was skipped by Danila's call.

Target image:

```
lmsysorg/sglang:v0.5.22-cu130   (same digest as :v0.5.22)
sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae   (index)
sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e   (amd64)
build checkout /sgl-workspace/sglang HEAD baa9befa381511cd4d56f83980951e5a45fffc6d
```

## Phase 0 — source and image audit. PASS

Source: release notes + `python/sglang/srt` of both tags (codeload tarballs),
diffed per patched module. Image: pulled, inspected with `--network none`.

### Platform moves this time (0.5.20 and 0.5.21 did not)

| | v0.5.21 (live) | v0.5.22 |
|---|---|---|
| Python | 3.12 system | 3.12.15, uv venv at `/opt/sglang` |
| torch | 2.13.0+cu130 | 2.14.1+cu130 (CUDA 13.0) |
| FlashInfer | 0.6.18 | 0.7.1 |
| transformers / tokenizers | 5.12.1 / 0.22.2 | 5.19.0 / 0.23.2 |
| sglang-kernel (Marlin) | 0.4.7 | 0.4.9 |
| Triton | 3.7.1 | 3.8.0 |
| xgrammar | 0.2.7 | 0.2.7 |
| sglang-router | 0.3.2 | 0.3.2 (router unchanged) |

`sglang` is on PATH (`/opt/sglang/bin/sglang`), so `entrypoint: sglang serve`
and `python3 -m sglang_router.launch_router` still resolve. The image ships no
`sitecustomize`, so `PYTHONPATH=/opt/enginepatch` still wins. Because the
platform moved, cold prefill, decode and quality must be re-measured on a
drained r1; the 0.5.21 numbers do not carry over.

### Flags

All 59 engine flags in the r0/r1 command blocks are declared in v0.5.22
(`--tp` is the `tp_size` alias). None of the breaking changes reach us:
`--enable-dp-attention`, `ChunkCache`, `tc_piecewise`, the PD threshold flip and
the renamed env vars are all unused here. Confirm the runtime `server_args` dump
on boot against r0's (defaults can move without the flag changing; see
v0.5.17 flipped mm-feature-transport that way).

### Engine patches against v0.5.22 (`deploy/enginepatch-next`)

| # | Hook | Upstream change | Verifier on 0.5.22 |
|---|---|---|---|
| 1 | `ChatCompletionRequest.to_sampling_params` | still returns a dict | `verify.py` PASS |
| 2 | `legacy_load_mm_data` | unchanged (only fork-safety init added) | `verify.py` PASS |
| 3 | `PrefillBudget.available_chunk_tokens` | **bug still upstream** (`else chunk_limit`) | `verify.py` PASS |
| 4 | `Fp8LinearMethod.apply` | Marlin path unchanged in Python; kernel pkg 0.4.7→0.4.9 | `verify_fp8_dequant.py` PASS (CPU); GPU bench pending |
| 5 | `_convert_to_internal_request`, `_process_messages`, `_validate_one_request` | still forces xhigh (low renders 53 tokens unpatched) | `verify_keypolicy.py` PASS |
| 6 | `UnifiedRadixCache.match_prefix`, `get_cached_tokens_details` | **`MatchResult.device_indices` → `device_prefix_len`** | `verify_cache_diag.py` PASS after port |
| R | `utils.auth.decide_request_auth` | byte-identical | `verify_keyrotation.py` PASS |

Two ports, both still pass on v0.5.21:
- `cache_diag.py`: reads `device_prefix_len` when present, else
  `len(device_indices)`. Unported, `note_match` swallows the AttributeError and
  patch 6 silently exports nothing.
- `verify_keypolicy.py`: the stand-in TokenizerManager gets
  `model_config.joint_head_config = None`; 0.5.22's validator reads it (Clef support).

### Reasons to roll (fixes that apply to our config)

- **#43029** spec decoding + grammar on a hybrid model recorded the GDN track
  boundary from the drafted, not the committed, length → a later prefix hit
  can resume from a wrong GDN state (silent). Our config: DFLASH +
  `extra_buffer` + xgrammar. Hits only grammar-constrained requests
  (`response_format`, strict tools).
- **#41736 / #41737** DFlash/DFlash2: NaN/Inf can no longer leak across
  requests through grouped-conv taps; greedy selector stays in range on all-NaN
  rows.
- **#42651** DFlash draft KV pool backed to the post-capture token count. Does
  not apply here: post-capture KV sizing is not active in our config, and r1
  booted with exactly r0's pool (381,504 tokens).
- **#42077** HiCache file backend: no full-dir scan on uncached prefix lookups
  (our L3 is a tmpfs dir of ~60 GB per process).
- **#36266** mamba COW kernel warmed before serving (first-wave 924→28 ms).
- **#43570** compact GDN causal-conv grids for skewed prefill batches.
- **#41259** long chat prompts tokenized in parallel (our agent turns are 90-145K).

### Behaviour changes to watch

- **#41520** radix `hit_count` now bumps only when a request finishes;
  `write_through_threshold` defaults to 1. May change when pages reach L3.
- **#42643** `stream_interval` logic rewritten; we run the default 1 → every
  step still streams.
- **#43005 / #43027 / #43026** OpenAI: `prompt_tokens_details.cached_tokens: 0`
  now always present; absent `tool_calls` omitted instead of `null`; one
  `created` per stream. Clients tolerate all three per the OpenAI schema.
- Kernel caches (`sglang_kernel_cache_r*`) hold FlashInfer 0.6.18 / Triton 3.7
  builds; both key their caches by version, so the first boot JIT-compiles.

### L3

`HiCacheFile` key format unchanged, but the mamba host pool is now built from
device buffer infos (#43596) and the HiCache transfer code moved. Same rule as
0.5.21: r1 gets a fresh dir `/dev/shm/qwen36-hicache-l3-v0522-fp8w/bf16`;
re-share once both replicas run 0.5.22, then delete the v0521 dir.

## Phase 1 — r1 trial (go 2026-10-19)

1. Drain r1 (`drain-replica.sh r1 drain`, waits on the replica's own metrics).
2. r1-only compose override: image v0.5.22 digest, `deploy/enginepatch-next`
   mounted at `/opt/enginepatch`, L3 dir v0522. r0 and the router untouched.
3. `NO_REGISTER=1 roll-replica.sh r1`; boot gates: patches 1-6 + R log lines,
   `max_total_num_tokens`, 128 mamba slots, decode bs [1..4], server_args diff.
4. Drained probes on r1: `bench_fp8_dequant.py` (GPU 1), cold prefill
   8K/32K/64K/128K, decode, spec acceptance, kvq/NLL, mamba_ckpt_probe,
   cache hit device/host/L3, mem_stress 128K.
5. Re-register r1; live A/B vs r0. Then r0 + router follow, one at a time.

# Execution log

## 2026-10-19 — r1 boot (Phase 1, steps 1-3). PASS

- Danila's go.
- r1 block in docker-compose.yml: `image:` v0.5.22 digest, `deploy/enginepatch-next`
  at /opt/enginepatch, L3 `/dev/shm/qwen36-hicache-l3-v0522-fp8w/bf16`.
  `docker compose config`: r0 and router unchanged.
- `NO_REGISTER=1 HEALTH_TIMEOUT=2400 roll-replica.sh r1`: r1 idle at drain,
  recreated 11:23:47Z, healthy after 673 s, left out of the router.
- Startup (s): load_weight 20, scheduler_e2e 592, cuda_graph prefill 316,
  target_verify 6.3, draft_decode 16.6 (0.5.21: 4.3; the DFlash selector decode
  is now folded into the draft graph, "sampling_enabled=True").
- Boot gates, all equal to r0:
  - every patch line in every process: 1, 2, 3, 4, 5 (context + effort), 6a, 6b
  - weights 29.34 GB target + 3.57 GB draft; mamba 128 slots, same sizes
  - `max_total_num_tokens=381504`, context 262,144, 4.42 GB GPU left
  - host pools: KV 1,795,008 tokens / 117.64 GB, mamba 62.44 GB, draft 18.38 GB
  - Tree Core: RustUnifiedTreeCore
- Runtime server_args vs r0 (api_key/launch_command stripped): only new fields at
  their defaults (watermark off, block verification off, attn_dp_size 1) and
  PD-only fields. No changed defaults among the flags we rely on.

## 2026-10-19 — drained gates on r1 (Phase 1, step 4)

`SKIP_ROLL=1 run_arm.sh V522` (smoke kvq kvqshort spec seq16 mem), then
mamba_ckpt_probe, prefill_stall, hicache_probe, bench_fp8_dequant. Baseline is
`P4_dequant_preserve` (2026-10-12, same flags and patches 1-5 on v0.5.21).
Results: `tuning/results/*_r1_V522*.json`.

| gate | P4 (0.5.21) | V522 |
|---|---|---|
| smoke | 7/7 | 7/7 |
| lookup 16K/64K/128K | 6/6 | 6/6 (TTFT equal) |
| kv/order/arith, greedy + prod | all ok | all ok |
| `seq` @ 16K | — | 10/10 |
| cold prefill 8K / 32K / 64K / 128K (s), code | 2.45 / 10.76 / 24.56 / 61.65 | 2.46 / 10.92 / 24.91 / 61.85 |
| cold prefill short docs, median (s) | 2.48 | 2.46 |
| mem_stress 160K + 3×60K, ×2 | clean, big 157 s | clean, big 157 s |
| decode tok/s short c1 / c4 | 132.3 / 90.9 | 114.6 / 88.2 |
| decode tok/s long c1 / c4 | 101.7 / 79.9 | 99.6 / 72.6 |
| aggregate tok/s short c4 / long c4 | 321.2 / 273.6 | 336.8 / 277.5 |
| accept length short c1 / c4 / long c1 / c4 | 3.71 / 3.17 / 3.34 / 3.44 | 3.38 / 3.18 / 3.37 / 3.31 |
| TTFT median long c4 (s) | 1.47 | 0.39 |
| mamba_ckpt_probe (L3 cleared) | PASS (v0521) | PASS 3/3, cold legs cached 0 |

- **ΔNLL is NOT measured.** The P4 run read a different corpus tree (its
  scratchpad is gone): only the first prose document has the same ids_sha.
  Tokenization is identical in both images (offline check on the same text,
  same hashes), so the difference is the text, not the tokenizer. A paired ΔNLL
  needs a same-day v0.5.21 control on `corpus-v0521` (image tree).
- **Decode vs a week-old baseline is inconclusive.** spec_eval runs 4 requests
  per c1 phase; short-c1 moved with acceptance (3.71 → 3.38), not per step.
- `mamba_ckpt_probe` without `--clear-l3` is VOID: `/flush_cache` keeps L3, so
  the "cold" legs were L3 hits. With r1 on its own L3 dir, `--clear-l3` is safe.
- `hicache_probe`: the evictors no longer evict (pool 381K vs 182K when the probe
  was written), so `A_after_eviction` is a device hit; host reload is unmeasured.
- `prefill_stall`: decoders stall p50 1.78 s / max 2.67 s per gap during a cold
  131K prefill (0.5.21 SPF: 2.10 / 3.28); cold 131K 61.7 s.
- `bench_fp8_dequant` on GPU 1 beside drained r1: attn qkv, o_proj, gdn in/out
  100% bit-exact against sglang-kernel 0.4.9 Marlin, 1.26-1.31x at M=4096; the
  MLP layers OOM'd the bench process (4.4 GB free), engine unaffected. MLP uses
  the same repack; end-to-end cold prefill matches P4.
- No Traceback / OOM in r1's log during any gate.

## 2026-10-19 — both replicas (Phase 2). APPLIED

Danila's call: update both replicas, skip the same-day v0.5.21 control.

1. 12:2xZ r1 restored into the router (`drain-replica.sh r1 restore`).
2. `enginepatch-next` copied into `enginepatch` (identical trees); `verify.py`
   and `verify_cache_diag.py` from the live dir PASS on both images, so an r0
   crash restart on 0.5.21 in between was safe.
3. `roll-replica.sh r0`: drained, recreated 12:28:25Z, healthy after 682 s,
   re-registered; router: both healthy.
4. r0 boot gates equal to r1 (all patch lines, 128 slots, 381,504 tokens,
   RustUnifiedTreeCore, 4.42 GB GPU left); runtime server_args r0 vs r1 differ
   only in port and random_seed; smoke on r0 7/7; `diag_*` rows from both
   replicas in `engine.requests`; no Traceback/OOM/500 on either.
5. `/dev/shm/qwen36-hicache-l3-v0521-fp8w` deleted (mounted by nothing; 106 GB).

docker-compose.yml: `&sglang-image` = v0.5.22 (r0, r1); the router has its own
`image:` with the v0.5.21 digest. Deployed r1 still mounts `deploy/enginepatch-next`
(it was booted for the trial); compose says `deploy/enginepatch`. The two trees
are identical; the next r1 recreate picks up the compose mount. **Until then,
an edit in enginepatch-next goes live on any r1 restart.**

Open: ΔNLL (needs a drained same-day control on corpus-v0521, now only possible
with a v0.5.21 r1 boot), single-stream decode vs 0.5.21 (watch live per-replica
decode in engine.requests), host-tier reload timing (hicache_probe evictors too
small for the 381K pool).
