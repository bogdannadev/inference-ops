# Next session — start here

State as of end of 2026-07-31. Phase 1 is **done and deployed to both
replicas**. Nothing is in flight; the deployment is in a clean, consistent
state and both replicas are byte-identical again.

---

## Where things stand

| | before today | now |
|---|---|---|
| throughput @ c=8 through the router | 243.89 tok/s | **444.93 tok/s** (1.82×) |
| peak throughput | ~250 tok/s | **466.73 tok/s** @ c=16 |
| `max_total_num_tokens` per replica | 160,768 | **169,792** |
| free VRAM per replica | 5.73 GB | **8.37 GB** |
| boot to healthy | ~9 min | **~3 min** |
| host RAM | 506 GiB in HiCache | **released** |
| single-stream decode | 68.32 tok/s | 68.56 tok/s (unchanged) |
| cold TTFT, 42K prompt | 22.681 s | 22.717 s (unchanged) |
| greedy output | reference | byte-identical |

Full detail and method: `../docs/RESULTS.md`.

---

## Read in this order

1. **`../docs/RESULTS.md`** — what was measured and how. Includes two
   measurement mistakes and their corrections; read those, they are the
   cheapest lessons in the directory.
2. **`CONTEXT_262144.md`** — the context-expansion work you asked to have
   prepared. Self-contained plan, not started.
3. **`../docs/NVIDIA_KERNEL_TUNING.md`** — A100/SM80 hardware constraints,
   NVIDIA tooling, and sources. **§0 first** — Phase 1's telemetry lowers the
   expected payoff of kernel work considerably.
4. `../docs/KERNEL_TUNING_SPEC.md` / `KERNEL_TUNING_INFO.md` — the SGLang-side
   kernel detail, if §3 of the NVIDIA doc makes you want it.
5. `../docs/TUNING_PLAN.md` — the original phase plan. Partly superseded; the
   corrections are marked inline.

---

## Candidate next moves, roughly by value

### 1. Context — `CONTEXT_262144.md`
Start with **Option C**: raise `--context-length` from 160,000 to ~169,000 to
match the pool we already pay for. Free, one roll, +5.6% advertised context.
Then Option B at 200,000. FP8 KV (Option A) is the only route to a true 262,144
without giving back the concurrency win, but it changes numerics and needs a
real quality gate.

**Trap flagged in that doc:** `--max-total-tokens 190000` is currently inert
only because 169,792 < 190,000. Any pool above 190,000 makes it bind.

### 2. Phase 2 — EAGLE depth
Arithmetically the largest remaining lever. Production accept length is 3.483
against a cap of 4.00, with 15.3% of verifies pinned at the cap and per-token
acceptance p ≈ 0.908. The current `steps=3 / topk=1 / draft=4` is verbatim the
SGLang docs' **OOM-recovery recipe**, not a tuned configuration. Use
`bench_speculative.py`.

Watch the Mamba coupling: `mamba num` tracks `num_draft_tokens`, so at draft=6
the 48-slot pool caps concurrency at 8. EAGLE depth and concurrency compete for
the same pool. (This is an observed correlation from one log line, not a
documented law — confirm it.)

### 3. `--enable-fused-qk-norm-rope`
Currently off. Qwen3.5/3.6 uses QK-norm; this fuses norm + RoPE. One flag, low
risk, fold it into whatever roll happens next.

### 4. Kernel work — only after profiling
`KERNEL_TUNING_SPEC.md` §S1 sets a stop rule at "GDN under ~15% of decode
time". Phase 1 measured `DRAM_ACTIVE` flat at ~69% while throughput rose 1.9×,
and occupancy moving only 8.2% → 9.4%. That is weak evidence the stop rule
will fire. **Profile before committing effort.**

### 5. Clock locking for repeatable benchmarks
Every ladder this session carried a 1365 → 1300 MHz clock decay under a 300 W
power cap. Consider `nvidia-smi -lgc` during A/B work — trades peak throughput
for measurement repeatability. Standard practice for kernel comparisons.

---

## Tooling built today

```
tuning/bench/run_ab.sh        <r0|r1> <label> [args]   containerised driver
tuning/bench/worker_ladder.py concurrency ladder, ONE worker, no router
tuning/bench/capture.py       telemetry sampler + preflight (250 ms)
tuning/bench/compare.py       render one result, or diff two
tuning/prometheus/            DCGM counter set with 9 profiling fields
```

`run_ab.sh` runs preflight first and **refuses to benchmark** if the DCP
profiling fields are missing, so a run cannot silently produce a
complete-looking file with no telemetry.

`compare.py` prints `occ% dram% sm% clk W` per rung, warns on >30 MHz clock
drift, and infers the effective admission cap from wall-time scaling.

---

## Rules that still apply

- **One replica at a time.** `docker compose up -d --no-deps qwen36-27b-r<n>`,
  wait healthy, then the other, then
  `docker compose restart qwen36-27b-router`.
- **Never `--remove-orphans`** — it would delete `qwen3-emb`, `grafana`,
  `prometheus`, `dcgm`. Compose warns about orphans on nearly every command
  here; ignore the warning.
- **Keep the image digest pinned.** A tuning roll must not also move the engine.
- **Always measure a control in the same window.** Phase 1's control drifted
  ≤1.6%, which is the only reason the treatment delta was credible.
- **Cold TTFT needs text neither replica has seen.** Two wrong numbers came
  from ignoring this today.
- **Stop the DCGM exporter before `nsys`/`ncu`** — they contend for the same
  counters and you get silent zeros.

---

## Git

Nothing in `tuning/` has been added to git — that was left deliberately for
you. Modified tracked files from today:

```
docker-compose.yml           Phase 1 config on both replicas + rationale banner
docker-compose.metrics.yml   DCGM custom counter file + 1s collection
prometheus/prometheus.yml    5s scrape for dcgm and sglang-workers
```

Note `docker-compose.multi.yml` shows as deleted and `merged.yml` as untracked;
both predate this session and were not touched.
