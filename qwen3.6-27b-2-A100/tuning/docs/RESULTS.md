# Results log

Running record of measurements. Newest last. Raw JSON in `tuning/results/`.

---

## 2026-07-31 — Pre-roll baselines (r0 control, r1 treatment)

Command: `./tuning/bench/run_ab.sh <r0|r1> <label> --repeat 2`
Both replicas still on the OLD config (`max-running-requests 2`,
`cuda-graph-max-bs-decode 2`, HiCache enabled). Nothing rolled yet.

Files: `ladder_r0_control_pre.json`, `ladder_r1_treatment_pre.json`

### r0 (control) — pass 1 / pass 2

| c | agg tok/s | per-stream | wall | p50 | occ% | dram% | sm% | clk MHz |
|---|---|---|---|---|---|---|---|---|
| 1 | 69.04 / 70.47 | 69.0 / 70.5 | 2.897 / 2.838 | 2.895 / 2.837 | 8.0 / 7.7 | 69.9 / 71.0 | 80.6 / 79.7 | 1365 / 1323 |
| 2 | 133.08 / 129.59 | 66.5 / 64.8 | 3.006 / 3.087 | 2.826 / 2.862 | 8.0 / 8.0 | 70.9 / 71.0 | 81.3 / 81.2 | 1353 / 1324 |
| 4 | 130.22 / 129.92 | 32.6 / 32.5 | 6.143 / 6.158 | 5.962 / 5.974 | 8.3 / 8.2 | 70.6 / 70.2 | 82.0 / 81.4 | 1350 / 1310 |
| 6 | 130.32 / 129.95 | 21.7 / 21.7 | 9.208 / 9.234 | 5.953 / 6.001 | 8.2 / 8.3 | 70.4 / 69.9 | 81.5 / 81.7 | 1351 / 1303 |
| 8 | 125.99 / 121.67 | 15.8 / 15.2 | 12.700 / 13.151 | 9.005 / 9.006 | 8.4 / 8.5 | 70.0 / 69.9 | 82.1 / 83.0 | 1329 / 1305 |
| 12 | 124.38 / 125.88 | 10.4 / 10.5 | 19.295 / 19.066 | 12.219 / 12.423 | 8.4 / 8.3 | 69.6 / 69.9 | 82.4 / 81.7 | 1327 / 1305 |

### r1 (treatment, pre-roll) — pass 1 / pass 2

| c | agg tok/s | per-stream | wall | occ% | dram% | clk MHz |
|---|---|---|---|---|---|---|
| 1 | 69.12 / 70.28 | 69.1 / 70.3 | 2.893 / 2.846 | 8.0 / 7.7 | 70.5 / 70.9 | 1357 / 1332 |
| 2 | 129.75 / 129.26 | 64.9 / 64.6 | 3.083 / 3.094 | 8.1 / 7.9 | 68.0 / 70.4 | 1350 / 1328 |
| 4 | 128.79 / 128.78 | 32.2 / 32.2 | 6.212 / 6.212 | 8.4 / 8.4 | 70.0 / 69.0 | 1352 / 1329 |
| 6 | 125.31 / 128.61 | 20.9 / 21.4 | 9.576 / 9.331 | 8.0 / 8.5 | 67.9 / 69.2 | 1336 / 1320 |
| 8 | 126.11 / 126.05 | 15.8 / 15.8 | 12.687 / 12.694 | 8.2 / 8.2 | 69.1 / 69.9 | 1341 / 1315 |
| 12 | 123.48 / 125.03 | 10.3 / 10.4 | 19.437 / 19.196 | 8.3 / 8.2 | 70.0 / 69.7 | 1334 / 1319 |

### The replicas are a matched pair

Delta B/A at every rung: **1.00, 1.00, 0.99, 0.99, 1.04, 0.99** on aggregate
throughput; p50 within 3%. Any post-roll difference larger than ~4% is real.

### Findings

**1. Per-replica throughput saturates at ~130 tok/s and the cause is the
admission cap, confirmed three independent ways.**

- Wall-time scaling: `wall(n) ≈ ceil(n/2) × wall(1)`. Measured
  2.90 / 3.01 / 6.14 / 9.21 / 12.70 / 19.30 s against a predicted
  2.90 / 2.90 / 5.80 / 8.69 / 11.59 / 17.38. Waves of **2**.
- `num_running_reqs.max = 2` at every rung, while `num_queue_reqs.max` climbs
  to **10** at c=12. Requests are queued, not run.
- `infer_cap()` best-fits **2**.

Aggregate is flat within 4% from c=2 to c=12 — adding load adds only queue.

**2. SM occupancy is pinned at ~8% and cannot move.** Occupancy reads
7.7–8.5% at *every* rung from c=1 to c=12. It does not rise with offered load
because the batch never exceeds 2. This is the clearest possible statement of
the problem: **the GPU is not being given work, and no kernel change can fix
that.** It is also the direct prediction to test after the roll — if batch 4
does not move this number, the roll did not take.

**3. DRAM_ACTIVE ~70% at every rung, versus a 56.1% weights-only estimate.**
The extra ~14 points are KV reads, activations, and the SSM state pool, which
the weight-streaming model in `TUNING_PLAN.md` omits. Memory pressure is
higher and headroom smaller than that model assumed. `SM_ACTIVE` 80–83% against
8% occupancy is the textbook latency-bound signature: SMs busy but nearly
empty.

**4. Power/thermal cap is active and is a genuine confounder.** A100 80GB PCIe
is a 300W part. Power sat at **293–305 W** throughout, and SM clock decayed
from **1365 → 1303 MHz (−4.5%)** across a ladder as GPU temp rose 53 → 76 °C
and HBM to 86 °C. Idle is 1410 MHz at ~68 W.

Consequences: (a) part of any concurrency gain will be returned as further
clock reduction, since a fuller batch draws more power; (b) later rungs are
systematically penalised relative to earlier ones, so *within-ladder* trends
are slightly pessimistic; (c) run order and starting temperature must be kept
comparable — r0's ladder started GPU0 at 53 °C, r1's started GPU1 at 50 °C.

**5. Accept length holds under queueing**: 3.11–3.29 across the ladder,
consistent with the 3.2 seen on router traffic.

### Tooling correction made during this run

`compare.py:infer_cap()` originally inferred the cap from `latency_max/p50` and
reported **6** for r0, where the true cap is **2**. With n=4 and cap 2 the
latencies are `[t, t, 2t, 2t]`, so p50 lands in the *second* wave and the ratio
collapses to ~1.0 — the median sits inside the queue. Replaced with a wall-time
fit, `wall(n) ≈ ceil(n/cap) × wall(1)`, which has no such failure mode and now
reports 2 for both replicas. Any earlier note quoting a cap of 6 is wrong.

### Next

Roll **r1 only**. Gates in `tuning/README.md`; the decisive one is decode CUDA
graph `bs = [1,2,3,4]`.

---

## 2026-07-31 — Phase 1 rolled on r1. Result: 1.8–1.9× throughput, one regression.

r1 rolled at 14:36, healthy at ~14:42 (**5m45s**, down from ~9 min — HiCache
removal confirmed). r0 untouched throughout.

Files: `ladder_r1_treatment_post.json`, `ladder_r0_control_post.json`

### Control did not drift

r0 pre → post, mean of 2 passes: **0.995 / 0.984 / 0.993 / 0.997 / 1.010 /
0.987**. Within 1.6% at every rung. The treatment delta below is real.

### Treatment result (r1 pre → post, mean of 2 passes)

| c | pre tok/s | post tok/s | gain | wall pre | wall post | occ pre | occ post |
|---|---|---|---|---|---|---|---|
| 1 | 69.70 | 69.19 | 0.99× | 2.869 | 2.891 | 7.9% | 8.3% |
| 2 | 129.50 | 128.94 | 1.00× | 3.088 | 3.104 | 8.0% | 7.6% |
| 4 | 128.78 | **232.39** | **1.80×** | 6.212 | 3.462 | 8.4% | 8.6% |
| 6 | 126.96 | 191.76 | 1.51× | 9.454 | 6.259 | 8.2% | 9.3% |
| 8 | 126.08 | **239.34** | **1.90×** | 12.691 | 6.690 | 8.2% | 9.1% |
| 12 | 124.25 | **232.28** | **1.87×** | 19.317 | 10.335 | 8.3% | 9.4% |

c=1 and c=2 unchanged, as they must be — they sit below the old cap. c=6 gains
less (1.51×) because with a cap of 4 it runs as 4+2, one full wave plus a
half-empty one. `infer_cap()` now reports **4**.

### The core thesis is confirmed, and by an unexpected route

**DRAM_ACTIVE barely moved**: 69.5% → 66.3% at c=4, 69.5% → 69.2% at c=8,
69.9% → 68.9% at c=12. Throughput rose ~1.9× while memory-interface activity
stayed flat.

That is the weight-amortisation prediction, measured directly. The memory
traffic was dominated by re-reading 54 GB of weights once per forward pass; a
batch of 4 reads them the same number of times and serves four streams from it.
Same bytes, ~1.9× the tokens.

It also explains why the gain is 1.9× and not 4×: the wave structure was
already keeping the memory system ~70% busy, so what was recovered is the
*idle* fraction, not a 4× multiplier.

**Occupancy rose only 8.2% → 9.1–9.4% (+13%)**, far less than the batch-2→4
doubling would suggest. The GDN decode kernel's grid is `192 × batch`
single-warp blocks, so batch 4 should double *its* contribution — but it is a
minority of total GPU time, and the whole-GPU average moves much less. This is
consistent with `KERNEL_TUNING_SPEC.md` S1's decision rule and is a mild
argument that GDN is **under** the 15% threshold where §S2 kernel work pays.
Profile before assuming otherwise.

### Deployment-level projection (both replicas on the treatment config)

| offered concurrency across the pair | today | projected |
|---|---|---|
| 8 | 258 tok/s | **465 tok/s** |
| 16 | 252 tok/s | **479 tok/s** |
| 24 | 249 tok/s | **465 tok/s** |

### REGRESSION — KV pool shrank below the served context length

`max_total_num_tokens` fell **160,768 → 151,168** (−6%). r0 still reports
160,768, so this is caused by the roll, not drift.

Mechanism, from the r1 boot log: the KV pool is sized *before* CUDA-graph
capture, from a profile whose working-space reserve scales with
`--max-running-requests`. Raising it 2 → 4 enlarged that reserve. Final
`available_gpu_mem` is **5.73 GB on both replicas** — identical — so this is
not wasted memory, it is memory moved from the KV pool into per-request
working space.

**This is a real, user-visible capability regression, verified directly.** A
154,997-token request:

```
r0 control    HTTP 200 after 76.7s   prompt_tokens=154997
r1 treatment  HTTP 400 after  1.2s   "Input length (154997 tokens) exceeds the
                                      maximum allowed length (151162 tokens)."
```

The failure is loud and fast, which is the correct behaviour and consistent
with keeping `--allow-auto-truncate` off. But the replicas now disagree, and
the router is round-robin — so **the same request succeeds or fails depending
on which replica it lands on**. Nondeterministic failure is worse than a
consistent lower limit.

Second-order effect: per-request KV budget under full concurrency drops from
160,768 / 2 ≈ **80K tokens each** to 151,168 / 4 ≈ **37.8K each**. The
scheduler retracts rather than erroring in that case, so it degrades gracefully
— but it is a genuine concurrency-vs-context trade, not a free win.

### Options (operator decision — not applied)

1. **Lower `--context-length` to 150000 on both replicas.** Consistent,
   honest, costs 6% of advertised context. Requires rolling both. Simplest.
2. **Raise `--mem-fraction-static` 0.90 → ~0.92 on r1.** Would recover roughly
   the lost 9,600 tokens (~0.6 GB). Risk: a documented stress test already hit
   96.1% of VRAM at 0.90 with 2 concurrent long prompts; at 4 concurrent the
   headroom is smaller and untested.
3. **Trim the prefill CUDA graph.** It is the largest non-KV consumer —
   **5.49 GB and 222 s of boot time** for 74 buckets up to 16,384 tokens.
   Lowering `--cuda-graph-max-bs-prefill` should shrink the profiler's reserve
   and grow the KV pool back, while also cutting boot time. Potentially free;
   needs one experimental roll to confirm the profiler responds.
4. **Accept `--max-running-requests 3`.** Smaller reserve, most of the gain.

Option 3 is the most attractive and the only one that is not a straight
trade-off, but it is unverified. Option 1 is the safe default.

**Until one is chosen, do not roll r0** — leaving it at 160,768 preserves the
ability to serve >151K requests on at least one replica, though the routing is
a coin flip.

---

## 2026-07-31 — Regression resolved. r1 now strictly better than r0.

Two further r1-only rolls. r0 untouched throughout the entire campaign.

### Roll 2 — `--cuda-graph-max-bs-prefill 2048`

Hypothesis (option 3): the prefill CUDA graph was the largest non-KV consumer,
so shrinking it would return memory to the KV pool.

**The hypothesis was wrong, and the way it failed is the useful part.**

| | before | after |
|---|---|---|
| prefill graph buckets | 74 (to 16384) | 42 (to 2048) |
| prefill graph memory | 5.49 GB | **1.21 GB** |
| prefill capture time | 222 s | **58 s** |
| boot to healthy | 345 s | **151 s** |
| `available_gpu_mem` | 5.73 GB | **10.01 GB** |
| `max_total_num_tokens` | 151,168 | **151,168 — unchanged** |

The KV pool is sized from a *static* `mem-fraction-static` budget minus a
reserve that scales with `max-running-requests`. It does **not** account for
actual CUDA-graph consumption. Freeing 4.28 GB of graph memory therefore just
left it idle — `available_gpu_mem` absorbed all of it.

So trimming the graph cannot fix the pool directly. What it does is create the
headroom that makes raising `mem-fraction-static` safe.

### Roll 3 — `--mem-fraction-static 0.90 → 0.92`

```
max_total_num_tokens=169792, max_running_requests=4,
context_len=160000, available_gpu_mem=8.37 GB       boot 181 s
```

**169,792 tokens — 9,024 ABOVE the original 160,768**, with 8.37 GB still free
(r0 has 5.73 GB). The regression is not merely repaired; capacity is now higher
than before the campaign started.

Verified end-to-end with the same 154,997-token probe that exposed the bug:

```
r0 control    HTTP 200    (1.9 s -- warm prefix cache from the earlier probe)
r1 treatment  HTTP 200    (77.2 s cold, matching r0's original 76.7 s)
```

### Cold TTFT — the prefill-graph trim is free

An apparent 12.361 → 13.87 s TTFT regression after roll 2 was a **bad
measurement, not a real effect**. The two runs were not comparable: the
"baseline" container had prior traffic (`accept_length` started at 3.45), the
other was genuinely cold (started at 0.0).

Re-tested properly — same novel ~42K-token prompt, generated fresh per run so
neither replica can hit prefix cache, streamed, two seeds:

| | seed 1 | seed 2 | mean |
|---|---|---|---|
| r0 (prefill graph **16384**) | 22.565 s | 22.797 s | 22.681 s |
| r1 (prefill graph **2048**) | 22.796 s | 22.638 s | 22.717 s |

**0.16% apart — identical.** For prompts of this size, prefill is compute-bound
and CUDA-graph launch-overhead elimination is irrelevant. Capturing 74 buckets
up to 16,384 tokens was costing 5.49 GB and 222 s of boot for no measurable
benefit.

A separate caution recorded for future runs: an earlier `bench_worker.py` TTFT
reading of 3.346 s was contaminated — the 155K-token probe reuses the *same*
`LONG_BLOCK` text as `bench_worker.py`, so the probe warmed the radix cache for
the very prompt the benchmark then measured. Always vary long-prompt text
between a probe and a benchmark, or the cache silently answers the question.

### Final state — consolidated

| c | r1 pre | r1 post-roll-1 | **r1 FINAL** | r0 control | FINAL/pre | FINAL/control |
|---|---|---|---|---|---|---|
| 1 | 69.70 | 69.19 | 70.00 | 69.38 | 1.00× | 1.01× |
| 2 | 129.50 | 128.94 | 129.80 | 129.29 | 1.00× | 1.00× |
| 4 | 128.78 | 232.39 | **231.81** | 129.14 | **1.80×** | **1.79×** |
| 6 | 126.96 | 191.76 | 189.31 | 129.70 | 1.49× | 1.46× |
| 8 | 126.08 | 239.34 | **241.08** | 125.08 | **1.91×** | **1.93×** |
| 12 | 124.25 | 232.28 | **233.85** | 123.46 | **1.88×** | **1.89×** |

The mem-fraction and prefill-graph changes cost **no throughput** — final
matches post-roll-1 within 1%.

### r1 vs r0 on every axis

| | r0 (control) | r1 (final) |
|---|---|---|
| throughput @ c≥4 | 125–130 tok/s | **231–241 tok/s** |
| single-stream decode | 68.32 tok/s | 68.56 tok/s |
| cold TTFT (42K prompt) | 22.681 s | 22.717 s |
| `max_total_num_tokens` | 160,768 | **169,792** |
| free VRAM | 5.73 GB | **8.37 GB** |
| boot to healthy | ~345 s | **181 s** |
| host RAM | +253 GiB (HiCache) | **released** |
| greedy output | reference | **byte-identical** |

Better or equal on every measured axis. Correctness confirmed: the greedy probe
returns the identical string and identical 1071-char reasoning trace.

### Recommendation

Roll r0 to match r1, then delete the A/B banner in `docker-compose.yml` and
restore the identical-replicas rule. Deployment-level projection at that point
is **~465–480 tok/s** across the pair versus ~250 today.

Note `--context-length 160000` is now *below* the 169,792-token pool on r1, so
context is once again limited by the flag rather than by memory. There is room
to raise it toward the model's native 262,144 if desired — but that is a
separate decision with its own concurrency trade (169,792 / 4 concurrent ≈ 42K
each). Worked through in `tuning/next-session/CONTEXT_262144.md`.

---

## 2026-07-31 — r0 rolled to match. Campaign complete.

r0 rolled at 15:35, healthy in **182 s**, reporting
`max_total_num_tokens=169792, available_gpu_mem=8.37 GB` — identical to r1.
Router restarted. `docker compose config` confirms the two command blocks now
differ **only** by port (`8001` vs `8002`); the identical-replicas rule is back
in force and the A/B banner has been replaced with a rationale block.

### Deployment-level ladder through the router

| c | before | after | gain |
|---|---|---|---|
| 1 | 69.24 | 70.09 | 1.01× |
| 2 | 136.47 | 138.34 | 1.01× |
| 4 | 261.73 | 259.4 (mean of 3) | 0.99× |
| 8 | 243.89 | **444.93** | **1.82×** |
| 12 | 240.94 | 364.97 | 1.51× |
| 16 | — | **466.73** | — |
| 24 | — | 442.78 | — |

Zero failures at every rung.

c=1/2/4 are unchanged **and should be** — 4 concurrent is 2 per replica, which
was already within the old cap of 2. The gain begins exactly where the old cap
began to bind, which is itself a consistency check on the whole diagnosis.

**One reading was a routing artifact, not a result.** The first pass measured
c=4 at 209.56 tok/s (0.80×), taken immediately after the router restart. Three
repeats gave **258.12 / 260.41 / 259.56** with verify-call splits of 49/51,
51/49, 49/51. The low reading came with an unbalanced split — all four requests
landing on one replica, which runs them at batch 4 (per-stream ~53) instead of
2×batch-2 (per-stream ~65). Repeat c=8 likewise gave a stable 444.93 / 444.92
at 50/50. **Do not benchmark through the router immediately after restarting
it**, and always check the split before trusting a rung.

### Final state

Both replicas: `max-running-requests 4`, `cuda-graph-max-bs-decode 4`,
`cuda-graph-max-bs-prefill 2048`, `mem-fraction-static 0.92`, HiCache removed,
`--enable-forward-pass-metrics` on. Peak **466.73 tok/s** against ~250 before.
Latency, single-stream decode, cold TTFT and greedy output all unchanged.
