# tuning/ — Qwen3.6-27B on 2× A100 inference tuning campaign

Self-contained working directory for the tuning effort. **Not yet added to
git** — that is a manual step for the operator.

```
tuning/
  docs/
    TUNING_PLAN.md          phase plan, measured baseline, ladder analysis
    KERNEL_TUNING_INFO.md   A100 SM80 + Qwen3.6-27B facts, kernel inventory
    KERNEL_TUNING_SPEC.md   what to tune, in what order, with what gates
  bench/
    capture.py              telemetry sampler + preflight (SGLang + DCGM)
    worker_ladder.py        concurrency ladder against ONE worker, no router
    compare.py              render one result, or diff two
    run_ab.sh               containerised driver
  prometheus/
    dcgm-counters.csv       custom DCGM counter set (DCP profiling fields)
    README.md               parser rules, bind-mount gotcha, nsys conflict
  results/                  benchmark output
  kernels/                  (empty) Triton overlays, if S2.2 is ever reached
```

---

## The experiment currently set up

`docker-compose.yml` has the two replicas **deliberately divergent**:

| | r0 — CONTROL | r1 — TREATMENT |
|---|---|---|
| `--max-running-requests` | 2 | **4** |
| `--cuda-graph-max-bs-decode` | 2 | **4** |
| HiCache | enabled | **removed** |
| `--enable-forward-pass-metrics` | no | **yes** |
| needs a roll? | **no** — already running this | yes |

r0 is left exactly as it is running, so the control costs nothing and carries
no risk. Only r1 is rolled.

### Why per-replica rather than rolling both

The router ladder could not measure this. With `--max-running-requests 2` on
both replicas the admission cap is 4 across the pair, so every rung above 4 ran
as *waves of 4*: modelling wall time as `ceil(c/4) × 3.057s` predicts
6.114 / 6.114 / 9.171 s at c=6/8/12 against measured 5.951 / 6.560 / 9.961, and
c=6 is visibly bimodal (p50 3.232 = one wave, max 5.948 = two). Both replicas
returned the same number regardless of configuration.

`bench/worker_ladder.py` drives one worker directly and removes the router from
the measurement entirely.

### Two variables at once — stated plainly

The r1 roll changes **both** HiCache and concurrency. Strictly that is two
variables. The justification is that HiCache cannot plausibly affect decode
throughput: it served 0.43% of cache hits, the benchmark prompts are disjoint
and short so they would never hit the host tier, and the device radix cache is
untouched. Its measurable effects are host RAM (−253 GiB), boot time (−~4 min),
and possibly cold-prefix TTFT.

**If TTFT regresses, the two are confounded** and HiCache must be re-tested
alone. Throughput results are not affected by this caveat.

---

## Running it

```bash
# 0. preflight telemetry (fails loudly if the DCP fields are missing)
source .env && docker run --rm --network qwen36-27b-backend \
  -e SGLANG_API_KEY="$SGLANG_API_KEY" -v "$(pwd)/tuning/bench:/bench:ro" \
  python:3.12-slim python3 /bench/capture.py --host qwen36-27b-r1 --port 8002

# 1. baselines, BEFORE rolling anything
./tuning/bench/run_ab.sh r0 control_pre
./tuning/bench/run_ab.sh r1 treatment_pre

# 2. roll r1 ONLY
docker compose up -d --no-deps qwen36-27b-r1
#    wait healthy, then read the boot log (see gates below)

# 3. measure
./tuning/bench/run_ab.sh r1 treatment_post
./tuning/bench/run_ab.sh r0 control_post     # drift check

# 4. compare
python3 tuning/bench/compare.py \
  tuning/results/ladder_r1_treatment_pre.json \
  tuning/results/ladder_r1_treatment_post.json
```

`control_pre` / `control_post` is not ceremony. It is how you detect that
thermals, the host, or a neighbouring container moved under you between
measurements. **If control drifts, the treatment delta is not trustworthy.**

### Never

- `--remove-orphans` — it would delete `qwen3-emb`, `grafana`, `prometheus`,
  `dcgm`, which belong to other projects. Compose *will* warn about
  `qwen3-emb` being an orphan on every command here. Ignore the warning.
- Rolling both replicas at once — that is the outage that started this work.
- Changing the image digest during a tuning roll — it confounds everything.

---

## Boot-log gates for the r1 roll

Read these before trusting any post-roll number:

| check | expected | if wrong |
|---|---|---|
| `max_total_num_tokens` | **160768**, unchanged | HiCache removal disturbed KV profiling — investigate before measuring |
| decode CUDA-graph `bs` | **[1,2,3,4]** | `--cuda-graph-max-bs-decode` did not lift; the concurrency raise bought nothing |
| `max_mamba_cache_size` | 48 | at 4 running × 4 slots = 16/48, fine |
| boot time | ~5 min (was ~9) | HiCache removal did not take |
| `log_requests` | `True` | flag still not reaching the process |

```bash
docker exec qwen36-27b-r1 sh -c \
  'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" http://localhost:8002/get_server_info'
```

---

## Reading the telemetry

`compare.py` prints `occ% dram% sm% clk W` per rung.

- **`occ%`** — `DCGM_FI_PROF_SM_OCCUPANCY`, resident warps ÷ hardware max
  (64 warps/SM on SM80). `KERNEL_TUNING_INFO.md` §3a predicts the GDN decode
  kernel alone contributes ~5.6% at batch 2 and ~11.1% at batch 4. Smoke test
  on r0 measured **5.6% at c=1 and 7.9% at c=2** whole-GPU. The control ↔
  treatment *ratio* is the real test.
- **`dram%`** — `DCGM_FI_PROF_DRAM_ACTIVE`. Smoke test read **49.3% at c=1 and
  71.6% at c=2**, against a weights-only roofline estimate of 56.1%. The gap is
  KV reads, activations and the SSM state pool, which the weights-only model
  omits. This is *more* memory pressure than predicted and means less headroom
  at c=4 than `TUNING_PLAN.md` assumes.
- **`sm%`** — `DCGM_FI_PROF_SM_ACTIVE`, fraction of SMs with ≥1 resident warp.
  81% at c=2 against 7.9% occupancy is the textbook latency-bound signature:
  the SMs are busy but nearly empty.
- **`clk` / `W`** — the confounder. See below.

### Power cap — found during the smoke test, affects every result

A100 80GB PCIe is a **300W** part. The smoke test measured **300.9–303.8 W and
SM clock 1372–1380 MHz**, already down from the 1410 MHz boost clock, at only
concurrency 2.

**The GPU is power-capped and clocking down before the experiment even
starts.** Raising concurrency raises power draw, so part of any Phase 1 gain
will be given back as further clock reduction. `compare.py` prints a warning if
SM clock varies more than 30 MHz across a ladder; treat that as "this
comparison is confounded by clock, not configuration".

This is also a reason not to trust idle-time baselines: at idle these GPUs sit
at 1410 MHz and ~68 W.
