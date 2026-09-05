# Router policy — cache reusability

Written 2026-09-05. The standing problem, open since 2026-08-08:

> "routing caps every cache feature at ~50%, and it gates any session-affinity
> work" — `UPGRADE_v0.5.17.md`

This document root-causes the two prior failures, and proposes the fix.

## What it costs us today

| evidence | source |
|---|---|
| 12 requests sharing a ~1200-token system prompt split 6/6 and reached only **48.2% cache hit** (10,752 of 22,303 prompt tokens) | `UPGRADE_v0.5.17.md` |
| A repeated prefix pays **full prefill twice** — "the first two are both cold" | `KEY-TIERS.md` |
| A device cache hit is worth **36x on TTFT**: 12.278 s -> 0.338 s | `TUNING_PLAN.md` |
| node-wide cache-hit figures "understate what a cache-aware policy could achieve" | `KEY-TIERS.md` |

With two TP=1 replicas that share no cache, `round_robin` guarantees that the
second call of any conversation has a 50% chance of landing on the replica that
has never seen it. Prefix caching is halved by construction.

## Failure 1 — `cache_aware` starved r0 (July). Root cause found.

Measured then (`TUNING_PLAN.md`):

| c | cache_aware | round_robin | split under cache_aware |
|---|---|---|---|
| 2 | 104.53 | 136.47 | 0 / 122 — all on r1 |
| 4 | 132.18 | 261.73 | 248 / 0 — all on r0 |
| 6 | 204.13 | 201.63 | balanced |
| 8 | 243.22 | 243.89 | balanced |
| 12 | 240.53 | 240.94 | balanced |

`cache_aware` has a load-balance guard that overrides affinity. From the
router's own help text:

```
--balance-abs-threshold   Balancing is triggered if (max_load - min_load) > abs_threshold
                          AND the relative threshold is also met.   (default: 64)
--balance-rel-threshold   Balancing is triggered if max_load > min_load * rel_threshold
                          AND the absolute threshold is also met.   (default: 1.5)
```

**Both conditions must hold, and the absolute one cannot ever hold here.** We
run `--max-running-requests 4` per replica behind a router capped at
`--max-concurrent-requests 16`. The maximum achievable load difference is on
the order of 16. It can never exceed **64**.

So the guard never fires, `cache_aware` degenerates to unconditional affinity,
and one hot shared system prompt pins the entire team to a single worker —
exactly the 0/122 and 248/0 splits observed.

It also explains the shape of the table: at c>=6 both workers sit at the
admission cap, so *every* policy looks balanced and the two columns converge.
The starvation was only ever visible at low concurrency, which is precisely
where this node's latency advantage lives.

**`cache_aware` was never the wrong policy. Its default was calibrated for a
fleet an order of magnitude larger than ours.**

Fix to test: `--balance-abs-threshold 2` (or 3), leaving `--balance-rel-threshold`
at 1.5. Balancing then triggers once one worker is 2 requests deeper than the
other, which at a 4-deep admission cap is a meaningful imbalance.

## Failure 2 — `prefix_hash` 503'd (August). Two blockers, one still standing.

From `UPGRADE_v0.5.17.md`:

1. The router must tokenize to hash tokens and had no tokenizer:
   `No tokenizer_path or model_path found for model unknown`. Fixable by adding
   `--model-path`, the HF cache mount and `HF_TOKEN`.
2. With the tokenizer loaded, requests still failed with
   `no_available_workers ("all circuits open or unhealthy")` while
   `GET /workers` reported both replicas healthy. **Workers register with
   `model_id: "unknown"`**, the router assigns policy per model
   (`Assigning policy prefix_hash to new model unknown`), and a request naming
   `qwen36-27b` matches no worker.

Blocker 2 is a registration-identity bug and is unresolved. Any policy keyed
per-model inherits it.

## The proposal — `manual` policy on a routing key

The router carries an **undocumented, tokenizer-free** affinity mechanism.
Confirmed by reading the shipped binary
(`sglang_router/sglang_router_rs.abi3.so`), not from docs:

```
headers   x-smg-routing-key      sticky key -> worker assignment
          x-smg-target-worker    explicit pin to a named worker

metrics   smg_worker_routing_keys_active          gauge, live sticky assignments
          smg_manual_policy_branch_total          labelled by branch:
          smg_consistent_hashing_policy_branch_total
          smg_prefix_hash_policy_branch_total
branches  target_worker_hit / target_worker_miss
          routing_key_hit / random_fallback / no_routing_id
          occupied_hit / occupied_miss / vacant
```

Why this is the right shape for us:

- **No tokenizer.** It sidesteps blocker 1 entirely — the thing that killed
  `prefix_hash`. Nothing needs `--model-path`, the HF mount, or `HF_TOKEN`.
- **Affinity is per session, not per prefix.** The July starvation came from
  affinity on a *shared* hot prefix, which by definition points every client at
  one worker. A per-session key spreads sessions across workers by
  construction.
- **Placement is load-aware.** `--assignment-mode {random,min_load,min_group}`
  decides where a *new* key lands. `min_load` puts a new session on the lighter
  worker and pins it there afterwards. That is affinity for cache reuse plus
  balance at placement — the combination neither prior attempt had.
- **It is observable before it is trusted.** The branch counters distinguish
  `routing_key_hit` from `random_fallback` and `no_routing_id`, so a
  misconfigured header shows up as a counter rather than as silent
  round-robin.

**The key already exists client-side.** OpenCode sends `X-Session-Id`. It needs
mapping to `x-smg-routing-key` in one header rewrite at Caddy or Higress.
Requests without the header take `random_fallback` — i.e. today's behaviour, so
the failure mode is a graceful degrade, not an outage.

### Do NOT use `consistent_hashing`

It falls back to hashing the `Authorization` header when no routing key is
present. The whole team shares one edge key, so every request would hash to the
**same worker**. That is the July starvation with extra steps.

## Cost of applying any of this

Every option here edits the router's own `command:` block, so it recreates the
router container. Per `docs/OPERATIONS.md` that **drops in-flight requests on
both replicas**. It is not a rolling change and it cannot ride along with
`roll-replica.sh`, which deliberately never touches the router. Take it in a
quiet window.

Reverting is the same operation in reverse: restore the flag, recreate, one
more in-flight drop. So "one reversible flag" is accurate about the config and
not about the blast radius — budget two brief interruptions, not zero.

## Order of work

1. **`--balance-abs-threshold 2` on `cache_aware`** — one flag, no client
   changes, immediately reversible. Tests the root-cause hypothesis above.
   Gate: `benchmarks/routing_test.py` (shared-prefix vs disjoint), plus the
   per-replica split. Pass = cache hit rate materially above 48.2% **and**
   no replica starved at c=2 and c=4, the two rows where `cache_aware` lost.
2. **`manual` + `x-smg-routing-key`, `--assignment-mode min_load`** — if step 1
   still starves, or if it works and we want exact session affinity rather
   than prefix heuristics. Needs the Caddy/Higress header rewrite.
   Gate: `smg_manual_policy_branch_total{branch="routing_key_hit"}` dominating
   `random_fallback`, plus the same split and hit-rate checks.
3. Re-test `prefix_hash` only if worker registration ever reports a real
   `model_id`. Not before.

`benchmarks/routing_test.py` already exists for exactly this and diffs
per-replica metrics to show the split. Note `benchmarks/ladder.py` uses
disjoint prompts deliberately and **cannot see this effect at all** — do not
gate on it.

## Why this matters more than it did in July

Two things changed the value of routing affinity:

- **It is a prerequisite for HiCache**, not a companion. Under `round_robin` a
  returning session reaches the replica holding its host copy only half the
  time. See the HiCache retraction in `UPGRADE_v0.5.19.md`.
- **v0.5.19 ships #34608**, which publishes per-scheduler load on a dedicated
  socket. The router today prices workers from its own in-flight counter, which
  misses direct-hostname traffic entirely and, for streaming responses, stays
  held for the whole response rather than the time the request occupies the
  scheduler. Every load-aware policy here — `power_of_two`, `cache_aware`'s
  balance guard, `min_load` assignment — is reading a number we now know is
  wrong. That is worth fixing before tuning thresholds against it.
