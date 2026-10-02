# SGLang v0.5.21 trial on r1 — 2026-10-02

**STATUS: APPLIED 2026-10-02 15:45Z.** r0, r1 and the router all run v0.5.21. Both
replicas run `shortest-prefill-first`, the Rust TreeCore and `--watchdog-timeout 180`, and
share the L3 directory `/dev/shm/qwen36-hicache-l3-v0521/fp8_e4m3`. The router runs health
checks at 15 s / 2 failures / 10 s timeout. T-LRU was trialled and removed. The v0.5.20 L3
directory has been deleted.

Target image:

```
lmsysorg/sglang:v0.5.21-cu130   (same digest as :v0.5.21)
sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469   (index)
sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75   (amd64)
build commit e00930c5489053f26d86b179cee0d087f846acbb
```

Three steps, one at a time. Each step: drain r1, recreate it out of the router,
run boot gates and drained probes, then put it back for a live A/B against r0.

1. v0.5.21 with unchanged flags. This step gets the fixes that need no flag.
2. Add `--schedule-policy shortest-prefill-first` (#40024).
3. Add `--radix-eviction-policy tlru` (#34012).

## Phase 0 — source and image audit (no live changes). PASS

- **Platform is unchanged.** The release notes advertise a CUDA 13.4.1 base, but
  the cu130 image still ships the following:
  - CUDA 13.0.3 and torch 2.13.0+cu130
  - FlashInfer 0.6.18 and transformers 5.12.1
  - sglang-kernel 0.4.7
  - xgrammar 0.2.1 → 0.2.7 (this is the only change)
- **Every live engine flag still exists** (59 checked against the v0.5.21 field
  declarations).
  - These choices are still valid: `extra_buffer`, `hrrn`, the `file` storage
    backend and `lru`.
  - New choices: `shortest-prefill-first` and `tlru`.
- **The Rust radix core (#39627) is not used here.** `--enable-session-radix-cache`
  forces the fallback (`tree_core_registry._rust_fallback_reason`: "session-aware
  caching requires the Python TreeCore"). Expect the boot log line
  `Using the Python TreeCore`.
- **T-LRU can attach.** The session cache lives inside `UnifiedRadixCache`, which
  is the only class the T-LRU registry check accepts.
- **The L3 key format is unchanged.** `HiCacheFile.config_suffix` is the same in
  both versions. Even so, r1 gets its own directory for the trial:
  `/dev/shm/qwen36-hicache-l3-v0521/fp8_e4m3`. The HiCache and mamba sidecar
  transfer code changed (#39567, #40456), and a silent format mismatch would
  corrupt GDN state on a cross-version hit. Re-share the directory once both
  replicas run the same version.
- **Engine patch:** `deploy/enginepatch/verify.py` passes inside the v0.5.21 image.
  - `legacy_load_mm_data` and its StopIteration → RuntimeError path are
    byte-identical.
  - #28131 (corrupt image bytes → 400) is a different path, so patch 2 stays.
- **Removed endpoints** (#40795): none are ours. Our scripts call only
  `/flush_cache`, `/get_server_info`, `/hicache` and `/start_profile`.
- **Router-side items do not reach us.** #39168/#39169/#39170 (worker queue limit,
  saturation floor) and #39461/#39464 (abort on disconnect, 429 backpressure) are
  all in `experimental/sgl-router`. We run `sglang_router`.
- `docker-compose.yml`: r1 gets an `image:` override and its own L3 volume. The
  `&sglang-image` anchor (r0 + router) is untouched.

### T-LRU parameters, from `engine.requests` (7 days to 2026-10-02)

| | value |
|---|---|
| continuation turns (prompt > 20K, cached > 0) | 10,140 |
| uncached tokens added per turn | mean 4,173 / p50 640 / p90 10,484 |
| mean prompt on those turns | 79,837 |
| cold prefill cost (uncached > 8K, fp8 era) | p50 0.452 / p90 0.668 ms per token |
| TTFT on prompts > 20K | p50 1.28 / p90 7.97 / p99 31.4 s |

Proposed settings, from a 5 s TTFT budget at 0.45 ms per token: `threshold` 11000
and `next_prompt_estimate` 4200.

**Hybrid-model risk:** an evicted tail takes its GDN checkpoint with it, so a hit
falls back to the last surviving checkpoint. That can recompute more than the
budget assumes. The step 3 probe measures this directly.

# Execution log

## Step 1 — r1 on v0.5.21, flags unchanged. Boot 2026-10-02 06:28Z

The drain came first: `drain-replica.sh r1 drain` reported r1 idle after 1 s. Then
`NO_REGISTER=1 roll-replica.sh r1` ran; r1 was healthy after **311 s** and stayed
out of the router.

**Watch out:** `roll-replica.sh` alone does not wait for in-flight requests on the
engine. It only deregisters and then recreates. Run `drain-replica.sh` first, which
polls `num_running + num_queue` until both are 0.

| Boot gate | Result |
|---|---|
| `max_total_num_tokens` | 282,496 (unchanged) |
| `max_mamba_cache_size` | 43 (conv 0.12 / ssm 3.09 / intermediate 2.81 / window 0.05 GB) |
| `available_gpu_mem` | 4.37 GB |
| tree cache | `Using the Python TreeCore: session-aware caching requires the Python TreeCore`; `UnifiedRadixCache hybrid_ssm=True hicache_attached=True` |
| CUDA graphs | same set as v0.5.20: target prefill (breakable), target and draft verify bs `[1, 2, 3, 4]`. Prefill capture took 40.4 s vs 17.1 s, which is the first-boot JIT |
| engine patch | both `[sitecustomize]` lines present |

**Server-args diff, r1 v0.5.20 → v0.5.21:** only keys were added or removed, and
no existing value changed its resolution.
- Added: `trust_request_chat_template` false, `frontend` "python",
  `enable_pd_role_switch`, `hicache_host_memory_fraction` null (an explicit ratio
  turns it off), `otlp_service_name`, `boundary_reduction`, `grpc_response_timeout_secs`
  and a few others.
- Removed: `elastic_ep_rejoin`, `keep_mm_feature_on_device`.
- `cuda_graph_config` gains `max_bs` / `tc_compiler` fields; bs is the same.
- The new internal state `decode_cuda_graph_bs: []` is expected: DFLASH replays the
  verify graphs, not a decode graph, in both versions.

**Smoke, direct to r1: 7/7** (`smoke.py` → `smoke_r1_v0521.json`)
- chat; reasoning split
- `qwen3_coder` tool call
- two calls to the same tool, non-streaming → `index` 0, 1 (#39574)
- 12 MP image → 2,089 prompt tokens
- 17 images → 400 "Image count 17 exceeds limit 16"
- a stream cancelled after 30 chunks → `num_aborted_requests_total` +1, 0 running,
  no "state was deleted" line

**Metering:** the smoke requests appear in `engine.requests` as `consumer=smoke
replica=r1`.

### Drained probes, r1, 2026-10-02 06:36–07:05Z

Baselines are r1 drained on v0.5.20 with the same live config (fp8 KV, L3 file,
43 slots, hicache 8, pdi 1, hrrn, chunk 4096).

| Probe | v0.5.20 | v0.5.21 | Read |
|---|---|---|---|
| `l3_probe` cold 78K | 32.00 s | 31.87 s | same |
| `l3_probe` reload from L3 after flush | 5.17 s (77,952 cached) | **4.29 s** (77,824) | −17%, answers correct |
| `mamba_ckpt_probe` (#37818 path) | PASS | PASS (A 0.11–0.23, B 0.09–0.19, cold vs cold 0) | unchanged |
| `prefill_stall` cold 45K / 131K | 15.2 / 68.1 s (`GHI_pdi1`) | 15.41 / 68.85 s | within noise |
| `prefill_stall` decoder freeze during the 131K prefill | 3.3 s max gap (`GHI_pdi1`) | 3.28 s | same; pdi 1 does this, not the version |
| `spec_eval` correctness | greedy kv/order/arith 15/15, seq 1/5 | 15/15, seq 2/5 | same (seq is the known fp8 weak spot) |
| `spec_eval` decode short c1/c4 | 76.5 / 63.8 tok/s, AL 3.53 / 3.23 | 75.9 / 62.7, AL 3.25 / 3.21 | neutral per unit AL |
| `spec_eval` decode long c1/c4 | 75.2 / 52.4, AL 3.41 / 3.24 | 71.2 / 51.0, AL 3.28 / 3.17 | neutral per unit AL |
| `spf_probe` (new), hrrn | — | short 2K TTFT p50 **40.9 s** behind a cold 100K prefill (long 46.7 s) | the step 2 baseline |

**Two harness traps found:**

- **`mamba_ckpt_probe` is VOID under L3.** `/flush_cache` leaves L3 in place, so
  the "cold" controls were L3 hits (cached 576/768/960) and every dlogprob read
  exactly 0.0.
  - The probe now reports `VOID_COLD_WAS_CACHED` in that case.
  - A new `--clear-l3` flag clears the storage backend through
    `Worker.flush(l3=True)` → `POST /hicache/storage-backend/clear`. The table
    above uses the re-run (`..._v0521_l3clear`).
  - **`HiCacheFile.clear()` deletes every file in its directory.** Use it only on
    a private L3 directory, never on the shared r0+r1 one.
- **The first stall comparison used the wrong baseline** (`GH_fp8_m43_hc8`, taken
  before pdi 1) and showed a 67.8 s → 3.3 s "improvement". The matching baseline
  is `GHI_pdi1`, and against it nothing changed.

### Live window: 07:08–08:04Z, r1 v0.5.21 next to r0 v0.5.20

| | r0 (v0.5.20) | r1 (v0.5.21) |
|---|---|---|
| requests (`engine.requests`, probes excluded) | 38, mean prompt 87,674 | 15, mean prompt 1,850 |
| TTFT p50 / max | 1.67 / 26.63 s | 0.54 / 0.98 s |
| aborts / retractions / other finish types | 0 / 0 / 0 | 0 / 0 / 0 |
| tracebacks, 5xx, OOM, "state was deleted" in log | — | none |

The window shows the replica is stable, nothing more. Cache affinity kept every
large agent session on r0, and r1 got only 15 small requests, all in its first
10 minutes. Gateway: 37 × 200, no errors.

## Step 2: r1 + `--schedule-policy shortest-prefill-first`, booted 2026-10-02 08:04Z

Healthy after 231 s; the JIT cache was warm. Boot gates are the same as step 1,
and the log shows `Experimental shortest-prefill-first: queue=… shortest_uncached=…`.

| Probe (drained, L3 cleared every round) | hrrn | SPF |
|---|---|---|
| 3 × 2K shorts arriving during a cold 100K prefill: short TTFT p50 / max | 40.88 / 41.84 s | **2.05 / 2.92 s** |
| that 100K prompt's own TTFT | 46.68 s | 47.34 s (+1.4%) |
| 12 × 2K shorts every 2 s during a cold 100K: short TTFT p50 / max | — | 2.66 / 3.82 s |
| that 100K prompt's TTFT under the stream | (46.7 s alone) | 52.95 s (+13%) |
| `prefill_stall` cold 45K / 131K; decoder max gap during the 131K | 15.41 / 68.85 s; 3.28 s | 15.45 / 68.94 s; 3.28 s |
| smoke | 7/7 | 7/7 |

- **Why the shorts get through:** SPF caps the long prompt's continuation chunk to
  `budget − reserved`, page-aligned, so the waiting shorts share the same
  4096-token step instead of queueing behind ~25 chunks.
- **What the long prompt pays:** the shorts' prefill work and nothing more. It
  cannot fully starve, because the continuation always keeps at least one page per
  step. Under a heavy short stream, though, a big cold prefill stretches without
  limit. hrrn has aging; SPF only breaks ties by arrival time.
- **Where it matters live:** a short request reaching a replica in the middle of a
  cold agent prefill. Over the 7 days before, 311 requests queued for more than 5 s,
  and queue p99 was 13.1 s.

### SPF live window: 08:25–09:25Z

r1 got **zero** live requests. r0 took all 33, because cache affinity keeps every
session on the replica that already holds it, and r1's cold, private L3 holds
nothing. No errors on either replica. A live A/B of a scheduling policy needs r1
to take real sessions. The drained probes above are the evidence.

## Step 3: + `--radix-eviction-policy tlru`, threshold 11000 / next_prompt_estimate 4200. Booted 09:25Z, REMOVED 09:35Z

The policy booted. The registry accepted the session-aware `UnifiedRadixCache`,
and `/get_server_info` showed `tlru` with that config.

`tlru_probe` setup: drained, L3 cleared. 5 conversations of 50K tokens, +4K per
turn, 4 turns, served round-robin, which overflows the 282K pool. Same seed both
runs, both with SPF.

| | lru | tlru |
|---|---|---|
| returning-turn TTFT mean / p50 / p90 / max | 2.07 / 2.05 / 2.23 / 2.23 s | 2.07 / 2.06 / 2.21 / 2.22 s |
| tokens recomputed beyond the 4K new | 32 | 32 |
| turn 3 hits, device / host | 49,984/8,000 · 0/57,984 · 0/57,984 · 36,864/21,120 · 32,768/25,216 | 53,952/4,032 ×3 · 49,984/8,000 · 45,056/12,928 |
| `mamba_ckpt_probe --clear-l3` | PASS | PASS (A 0.10–0.19, B 0.06–0.13, cold vs cold 0) |
| smoke | 7/7 | 7/7 |

**T-LRU works as designed and buys nothing here.**
- It keeps more of each conversation on device by evicting tails first.
- But a returning turn costs the same whether its prefix comes from device or host:
  ~2.0 s for the 4K new tokens. A 50–62K host reload adds almost nothing.
- The paper assumes that evicting a prefix means recomputing it. HiCache (host
  ratio 8, plus L3) breaks that assumption.
- No GDN state damage was seen.
- Removed so an experimental policy does not run for nothing.
- **When to revisit:** only if the host tier itself starts evicting
  (`evicted_tokens_total` on host) or HiCache is ever dropped.

## Where this leaves the list

| Item | Verdict |
|---|---|
| v0.5.21 itself | Safe, all gates pass. L3 reload −17%, tool-call `index` fix (#39574). Decode, prefill and correctness neutral |
| Rust TreeCore default (#39627) | Not used: `--enable-session-radix-cache` forces the Python core |
| shortest-prefill-first (#40024) | **Keep on r1.** Shorts behind a cold 100K prefill: 40.9 s → 2.1 s, at +1.4% on the long one. No aging, so watch big cold prefills under a heavy short stream |
| T-LRU (#34012) | Removed: no gain with the HiCache host tier |
| GDN verify QKV views (#33778), CUDA-graph small copies (#41166) | In the image. Inside decode noise, as expected for host/launch-side changes on this bandwidth-bound node |
| HiCache/mamba fixes, metrics fixes | In the image. Nothing regressed in any probe |

**To converge** (operator decision):
- move the `&sglang-image` anchor to v0.5.21;
- drop r1's `image:` override;
- choose the schedule policy for both replicas;
- point both replicas at one fresh shared L3 directory (`fp8_e4m3`, empty), so
  no cross-version pages exist;
- roll r0 (drain-replica first), then recreate the router at a quiet moment.

Router v0.5.21 changes nothing we use. The new router flags live in
`experimental/sgl-router`.

**Rollback (historical):** this section described rolling back the r1-only trial. For the
converged state, see Rollback in the Convergence section below.

## Step 4: Rust TreeCore on r1. Booted 10:10Z; serving again from 10:44Z

- **The change:** `--enable-session-radix-cache` is dropped on r1. It has been inert since
  it was added (no client sends `session_id`), and it was the only thing forcing the Python
  fallback.
- **Boot:** `Tree Core: RustUnifiedTreeCore`. Gates unchanged: 282,496 tokens, 43 slots,
  HiCache attached.
- **The A/B:** same r1 and config, with the Python core forced through
  `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` in a throwaway compose overlay.

| Probe | Rust | Python |
|---|---|---|
| `cache_replay --clear-l3`: cold / near / far hit ratio (full hits) | 0.0 / 0.658 (8/12) / 0.978 (12/12) | identical; same cached tokens on every turn |
| mean turn seconds near / far | 5.42 / 0.86 | 5.43 / 0.93 |
| scheduler CPU during the replay | 685.6 s | 665.7 s |
| `tlru_probe` warm TTFT mean, recompute | 2.06 s, 32 | 2.07 s, 32 |
| `mamba_ckpt_probe --clear-l3` | PASS | PASS |
| smoke / l3_probe / spf_probe / spec_eval correctness | 7/7 · reload 4.10 s · short p50 2.19 s · unchanged | — |

**Verdict:** the Rust core is equivalent and safe. No hit-rate or speed change shows on
this GPU-bound node. We keep it to stay on upstream's default path.

- To roll back, restore `--enable-session-radix-cache` on r1 (or set the env var above).
- For the stall and day-long-session analysis that followed, see `STALLS_AND_LONG_SESSIONS.md`.

## Convergence, 2026-10-02 15:32–15:45Z

**Order:**
1. **r0:** drained (idle after 1 s), rolled, back after 301 s.
2. **r1:** drained, rolled (to pick up the watchdog), back after 221 s.
3. **Router:** recreated at a moment when both engines had 0 running and 0 queued on two
   consecutive samples, 12 s after the wait began.

**Checks after convergence:**
- **Both replicas:** `/get_server_info` shows v0.5.21, `shortest-prefill-first`,
  `watchdog_timeout 180` and `enable_session_radix_cache false`. Boot log shows
  `RustUnifiedTreeCore`, 282,496 tokens, 43 slots and HiCache attached.
- **Shared L3:** r0 found r1's pages at boot (22,115 entries, 58.7 GB).
- **Smoke:** 7/7 on r0 and on r1 (`smoke_r{0,1}_v0521_final.json`).
- **Router:** v0.5.21 image, both workers healthy, health checks every 15 s with all of them
  succeeding. Two requests through the router came back 200 in 0.28 s and 0.37 s.
- **Cleanup:** `/dev/shm/qwen36-hicache-l3` (v0.5.20 pages, 103 GB of RAM) was deleted once
  no container mounted it.

**Rollback:** move the `&sglang-image` anchor back to
`sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`, set
`--schedule-policy hrrn`, restore `--enable-session-radix-cache` and `--watchdog-timeout 1200`,
remove the three router health flags, and point L3 at a fresh empty directory (never the
v0521 one). Then roll each replica (drain first) and recreate the router at 0 running.
