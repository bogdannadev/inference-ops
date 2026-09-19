# Trial: v0.5.20 levers on top of fp8 KV — 2026-09-19

r1 drained 16:25–19:30 (local), r0 served alone. Starting point is production
as of 16:05 (`TRIAL_FP8KV_MAMBA.md`): fp8_e4m3 KV, `--max-mamba-cache-size 24`,
mem 0.92, hrrn, HiCache ratio 3, chunk 4096. Each arm changed one line of the
r1 block. r1 was restored to that exact config afterwards (compose sha256
identical) and re-registered.

Asked for: SLRU eviction, mem 0.93, `extra_buffer_lazy`, chunk 2048, and
Grafana panels for the new scheduler metrics. Added during the run, because
the first measurements pointed there: GDN slots back to 43, HiCache ratio 8,
`--prefill-decode-interval 1`, `--enable-mixed-chunk`.

## New tools

- `tuning/bench/cache_replay.py` — 4 agent sessions (41K tokens, +1K per
  turn) interleaved with one-off 3K requests; "near" = 32 one-offs between
  two turns of the same agent, "far" = 120. Random token ids, fixed seed, so
  every arm replays the identical sequence.
- `tuning/bench/prefill_stall.py` — cold 45K / 128K prefill, then 3 streams
  decoding while a cold 128K prompt arrives; per-stream gaps while it prefills.
- `mem_stress.py --images N` — replaces one side request with N ~2 MP images
  (~2K image tokens each) during the 160K prefill.
- `evict_probe2.py` (scratch) — K agents x L tokens prefilled back to back,
  then each resent + 1K: how many long contexts survive with no other traffic.

## The finding that reorders everything: the GDN state tier is what runs out

A prefix hit on this hybrid model needs the full-attention KV **and** a GDN
(mamba) state at the node where the match ends. The two tiers evict
separately; the GDN tier is always LRU (`unified_cache/components/mamba.py`),
whatever `--radix-eviction-policy` says. Its sizes:

- device: `--max-mamba-cache-size` slots, shared with running requests;
  a long prompt leaves a checkpoint state per prefill chunk, so a 41K context
  holds several slots;
- host (HiCache): slots x `--hicache-ratio` states
  (`MambaPoolHost`: `device_pool.size * host_to_device_ratio`).

| config | GDN device / host states | 41K contexts surviving back to back | replay near hit | replay far hit |
|---|---|---|---|---|
| **production now**: 24 slots, ratio 3 | 24 / 72 | **2** (3 → all miss) | **0.00** | **0.00** |
| 43 slots, ratio 3 (= bf16 production until today) | 43 / 129 | 4 (6 → all miss) | 0.66 | 0.16 |
| 43 slots, ratio 8 | 43 / 344 | 4 | 0.66 | **0.98** (12/12, 0.77 s/turn vs 13.7 s cold) |

Hits in the replay came from host RAM (`cached_tokens_details.host`), device 0.
The first revisit after the cold round missed in every config; later
revisits hit. Not explained (states of a first insert apparently do not reach
the host tier before device eviction), not chased.

KV was never the limit: over 7 days of production, in-use KV p99 was
116-131K tokens per replica (Prometheus `sglang:kv_used_tokens`), peak 172K on
r0; 43 slots + fp8 gives 282,304. The GDN pool sat at 41 of 43 evictable on
both replicas: full of cached states. So the 24-slot pin applied at 16:05
traded capacity that was not binding (+35K KV tokens) for the tier that is,
halving how many agent contexts a replica keeps.

## The second finding: a long prefill freezes every other stream

The scheduler runs a prefill chunk or a decode step per iteration and keeps
feeding a chunked prefill until it completes. Three streams decoding at
60-66 tok/s, then a cold 128K prompt:

| arm | longest stream freeze | tokens to streams during it | 128K prefill | cold 45K / 128K alone |
|---|---|---|---|---|
| base (43 slots, ratio 8) | **67.8 s** | ~0 | 68.9 s | 15.3 / 68.5 s |
| `--prefill-decode-interval 1` | **3.3 s** (a step every ~2.1 s) | ~3 per step (spec) | 70.2 s (+1.9%) | 15.2 / 68.1 s |
| `--enable-mixed-chunk` | 3.2 s | 1 per step (spec off in mixed steps) | 69.1 s (+0.2%) | 15.2 / 68.2 s |
| `--chunked-prefill-size 2048` | 69.8 s | ~0 | 69.8 s (+1.3%) | 15.5 / 69.1 s |

This is production behaviour today and before today: any agent's 70K cold
prompt (~20-25 s) stops every other stream on that replica for its duration.

## Per-lever results

**SLRU** (`--radix-eviction-policy slru`, threshold 2): replay identical turn
for turn to the base (near 0.66, far 0.98). It only orders the KV component;
KV never filled, and the GDN tier stays LRU.

**mem 0.93**: 302,016 tokens (+7%), `available_gpu_mem` 4.36 → 3.58 GB.
Stress with 16 images (33K prompt tokens) during a 160K prefill and the plain
340K stress: 0 errors, 0 OOM. Stable, but it buys KV, which is not binding.

**`extra_buffer_lazy`**: mamba_ckpt_probe PASS 3/3 (decode-checkpoint hit
0.11-0.16 vs 0.11-0.18 non-lazy); spec_eval same correctness as the fp8 arms,
determinism 2/2; decode within noise (long c=4 193.9 vs 185.8 tok/s); stress
0 errors. Capacity unchanged (4 x 41K survive, 6 do not): the slot it saves
is only held while a request runs.

**chunk 2048**: does not shorten the freeze, +1% prefill, and **halves
context capacity** (43 slots: 4 x 41K → all miss) because each context
leaves twice as many chunk checkpoints. Harmful here.

**mixed chunk**: smoke 5/5 (tool call, 12 MP image, 17-image 400), spec_eval
same correctness as fp8 arm A, determinism 2/2, 0 OOM. Fixes the freeze like
the interval, but running streams get plain 1-token decode in mixed steps.

**Scheduler metrics** (new in 0.5.20): row "Scheduler loop (v0.5.20 metrics)"
added to the SGLang Engine dashboard (busy fraction, loop time by stage, CPU).
Under load `run_batch` is 97-98% of loop time: GPU-bound, ~1.5% scheduler
overhead. `sanity_check_cache` runs only from the idle path (0.4 s during a
14-minute load test). The scheduler process spins one full core even idle.

**HiCache ratio 8**: host pools 74 GB KV + 27 GB GDN + 23 GB draft = 124 GB
pinned per replica (was ~47 GB at ratio 3 with 24 slots); boot 231-241 s vs
171-181 s. Host had 657 GB available with one replica at ratio 8.

## Rating (0-10, value of applying to production)

| lever | rating | why |
|---|---|---|
| GDN slots back to 43 (keep fp8) | **9** | undoes today's halving of agent-context retention; KV p99 fits 282K twice |
| HiCache ratio 8 (with 43 slots) | **9** | far-return hit 0.16 → 0.98, 0.77 s vs 13.7 s per 45K turn; costs RAM we have |
| `--prefill-decode-interval 1` | **8** | stream freeze 67.8 s → 3.3 s for +1.9% prefill; scheduling only |
| `--enable-mixed-chunk` | 5 | same freeze fix, cheaper prefill, but 1 token/step and a less-trodden numerics path; pick the interval instead |
| scheduler metrics panels | 6 | applied (dashboard only); diagnosis value, no speed change |
| `extra_buffer_lazy` | 2 | correct, neutral, no capacity gain |
| mem 0.93 | 2 | stable, +7% of a resource that is not binding, -0.78 GB headroom |
| SLRU | 1 | no measurable effect on this model |
| chunk 2048 | 0 | harmful: halves contexts retained, freeze unchanged |

Proposed production config (applied ~20:50 with bf16 instead of fp8, see
"Applied" at the end): fp8_e4m3, **43 slots**, mem 0.92,
**`--hicache-ratio 8`**, **`--prefill-decode-interval 1`**, hrrn, chunk 4096.
All three were measured together: the interval arm ran on 43 slots + ratio 8.
Not measured: the combination under real traffic, and the first-revisit miss.

Open leads: `--chunked-prefill-size 8192` (the capacity finding predicts half
the checkpoints per context → more contexts; freeze steps get longer); more
than 43 slots (each ~73 MB → ~2.3K fp8 tokens); with KV not binding, fp8's
+55% buys little while its prefill cost (+11-20%) stays — bf16 + 43 slots
(182K) covers the KV p99 but not r0's 172K peak with much margin.

Results: `tuning/results/{cache_replay,prefill_stall,mem_stress,mamba_ckpt_probe,spec_eval}_r1_{P_prod,G_fp8_m43,GH_fp8_m43_hc8,GHS_slru,GHI_pdi1,GHX_mixed,GHC_chunk2048,GHM_mem093*,GHL_lazy}.json`
(only the files each arm produced).

# Applied — 2026-09-19 ~20:50 (operator decision: drop fp8)

Final config on both replicas: **bf16 KV**, mem 0.94, `--max-mamba-cache-size
43`, `--hicache-ratio 8`, `--prefill-decode-interval 1`, hrrn, chunk 4096.
Measured first on drained r1 against the same base plus chunk 8192:

| | bf16 candidate (chunk 4096) | + chunk 8192 (and max-prefill-tokens 8192) |
|---|---|---|
| KV tokens / headroom | 182,528 / 2.80 GB | same |
| cold 45K / 128K | 14.14 / 57.45 s (fp8: 15.3 / 68.5) | 13.97 / 56.26 s (-1 / -2%) |
| longest stream freeze, 128K prefill | 2.59 s | 4.98 s |
| 41K contexts surviving back to back | 4 (6, 8 → all miss) | 4 (6, 8 → all miss) |
| replay near / far hit | 0.66 / 0.98 (far 0.71 s/turn) | 0.74 / 0.98 |
| 16-image + 160K stress | 0 errors | 0 errors |
| OOM / Traceback | 0 | 0 |

Chunk 8192 is stable but not a win: prefill 1-2% faster, one extra near-phase
hit, freeze doubled. On bf16 the capacity probe cannot show a chunk effect
above 4 contexts, because 5 x 41K already exceeds the 182K device KV pool.

Rollout: r1 (drained) → `NO_REGISTER=1 roll-replica.sh r1` → smoke 5/5 →
`drain-replica.sh r1 restore`; then `roll-replica.sh r0` (drain, recreate,
healthy after 241 s, re-registered). No router restart. `/get_server_info`
r0 vs r1 differ only in port, random_seed, startup_time, internal_states.
Host RAM available after both replicas pinned their pools: 559 GB.

Back out: previous config is in git history (fp8 variant) or drop
`--prefill-decode-interval 1` and set `--hicache-ratio 3` for the pre-trial
bf16 state; roll one replica at a time.
