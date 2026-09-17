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

---

# Result — step 1 APPLIED 2026-09-05. The root cause was correct.

`--policy cache_aware --balance-abs-threshold 2`, applied to the router command
block in `docker-compose.yml` and recreated with `docker compose -f
docker-compose.yml -f docker-compose.metrics.yml -f docker-compose.langfuse.yml
up -d --no-deps qwen36-27b-router`.

**Control measured the same day, immediately before the change**, so the
comparison is not against the August figure.

| | round_robin (control) | cache_aware, abs=2 | |
|---|---|---|---|
| shared-prefix cache hit | 14,848 / 22,807 = **65.1%** | 22,272 / 22,807 = **97.7%** | **+32.6 pts** |
| shared-prefix split r0/r1 | 6 / 6 | **10 / 3** | affinity, not starvation |
| shared-prefix p50 | 2.086 s | 2.087 s | unchanged |
| shared-prefix max | 3.221 s | **2.507 s** | tail improved |
| disjoint split | 6 / 6 | 6 / 6 | control holds |
| disjoint cache hit | 0 / 762 | 0 / 762 | correct |
| failures | 0 | 0 | |

**The c=2 row — the one that starved 0/122 in July — re-run at concurrency 2:**

```
shared_prefix  reqs r0/r1=10/2  balance 83%/17%  cached=22272/22807 (97.7%)
               p50=1.951s  max=2.645s  failed=0
disjoint       reqs r0/r1=6/6   balance 50%/50%  cached=0/762  failed=0
```

Both replicas serve. In July this row was **0/122**.

**Both gates pass.** Cache hit is far above the 48.2% that motivated the work,
and close to the ~91% ceiling implied by "11 of 12 requests hit a warm prefix".
No replica is starved at c=2 or c=4. Latency did not regress; the tail improved.

This confirms the diagnosis directly: at the default `--balance-abs-threshold`
of 64 the guard could never fire at our scale, so affinity ran unchecked and
pinned everything to one worker. At 2 it fires once a worker is two requests
deeper, and affinity and balance coexist.

## Post-change verification

```
router health   healthy, same image digest as the compose anchor
argv            --policy cache_aware --balance-abs-threshold 2
smg_worker_health{r0} 1      smg_worker_health{r1} 1
smg_worker_selection_total{policy="cache_aware", model="qwen36-27b"}  48
smg_router_request_errors_total                                       absent (zero)
live traffic    200s in the router log, both replicas served
```

**The tokenizer warning is present and harmless here:**

```
WARN submit_tokenizer_job: No tokenizer_path or model_path found for model
     unknown (checked worker labels and router config)
```

Same warning that made `prefix_hash` fail, but `cache_aware` keys its
approximate radix tree on request **text**, not tokens, so it needs no
tokenizer. Workers activate and serve normally. Do not read this as a fault.

## Two things noticed, neither blocking

1. **A second model entry exists.** Selection counters show
   `model="qwen36-27b"` (48) *and* `model="qwen38-27b"` (1). Some client asks
   for a name we do not serve as `--served-model-name`. Because `cache_aware`
   assigns policy **per model**, that client gets its own independent
   cache-aware tree — served, but a separate namespace that shares no prefix
   affinity with the main one. Worth tracing to its source.
2. **The recreate dropped one in-flight request.** Traffic arrives in bursts;
   the gauge read 0, then 1 again within seconds, so no genuinely quiet window
   existed. The documented cost was paid once, as expected.

## Still open, unchanged by this

- **Per-person keys (Stage B).** Direct-hostname traffic bypasses the router
  entirely and gets no affinity at all.
- **#34608, the per-scheduler load socket.** The balance guard we just tuned
  prices workers from a router-side in-flight counter that misses
  direct-hostname traffic and, for streaming responses, is held for the whole
  response rather than scheduler-occupancy time. **The threshold of 2 is
  calibrated against a number known to be wrong.** Revisit after the router can
  read real scheduler load.
- **Step 2 (`manual` + `x-smg-routing-key`) is not needed for now.**
  `cache_aware` reached 97.7% with no client changes. Hold step 2 in reserve
  for many short unrelated sessions, or a shared prefix dominant enough to make
  the balance guard thrash.

---

# 2026-09-17 — the balance guard's blind spot, and which policy fits

## What happened

2026-09-16 14:20-14:28: r1 ran one vkondratyev-demo request (61K prompt, 32,000
output tokens, 8 min) and held 1-2 more turns of the same session in its queue.
r0 was idle. Four clients gave up after 155-240 s. The router counted r1 at 2
in flight and r0 at 0; the guard needs a gap **greater than** 2, so every turn
followed its cached prefix to r1.

## How often (Prometheus, 2026-09-10..17, 15 s steps)

76 minutes with a queue on one replica while the other had no queue and a free
slot; 54 of them with the other replica fully idle. The router's own in-flight
gap at those moments:

| gap | minutes | fixed by `--balance-abs-threshold 1`? |
|---|---|---|
| 2 | 35 | yes |
| 0-1 | ~22 | no: one huge request looks like one small one |
| 3+ | ~19 | no: the guard should already have fired; not explained yet |

This is now the recording rule `node:engine_queue_stranded:bool` and the alert
`EngineQueueStrandedOnOneReplica` (docs/OBSERVABILITY.md → Serving alerts); the
bot's `/errors` prints the minutes.

## The mechanism, from source (gateway-v0.3.1; the image runs 0.3.2)

- `cache_aware.rs`: `load()` is the router's in-flight **request counter**. If
  the prefix match is over `--cache-threshold` (0.3), the tree's worker is
  chosen with no load check; load is consulted only when the global guard
  fires, or when nothing matches.
- `LoadMonitor` (`core/worker_manager.rs`) polls each worker's `/get_load` —
  `num_tokens`, `num_waiting_reqs`, `num_pending_tokens`, the numbers the guard
  actually needs — but feeds them **only to `power_of_two`** policies, every
  `--worker-startup-check-interval` (30 s default). Upstream `main` cache_aware
  still does not read it (checked 2026-09-17).
- The monitor authenticates with the router's `--api-key`, which we do not set.
  Our engines answer `/get_load` with 401 without the key (checked), so even
  `power_of_two` would silently fall back to request counts today.
- `manual.rs`: a new routing key goes to a **random** worker ("TODO: use
  load-aware selection later"); an existing key stays on its worker while that
  worker is healthy. 0.3.2's `--assignment-mode min_load` changes only where a
  *new* key lands, by request count.

## Our traffic

| who | shape | what routing must do |
|---|---|---|
| vkondratyev-demo | agent sessions, prompts 60-170K growing each turn, 93% cache hit, 2-3 parallel streams, occasional 32K-token answers, OpenAI SDK retries | keep a session on its replica (a miss costs a 49 s prefill on average for prompts over 60K, 65 s p90) but move it when that replica is blocked |
| direct hostname | ~1,000 short requests per hour today (2K prompts, 200-3000 output), sequential | anything works; it pins to r0 under cache_aware, harmless while r0 has room |
| tim, danila | occasional | anything works |

## Policies against that

| policy | verdict |
|---|---|
| `cache_aware`, abs 2 (live) | right shape, guard too coarse: 76 stranded min/week |
| `cache_aware`, abs 1 | **do now.** Catches the gap-2 cases (35 of 76 min, including 14:20). Moves a session only while the replicas are imbalanced, so the cold-prefill cost is paid only when the alternative is waiting in a queue |
| `power_of_two` | rejected for this traffic. Token-aware only after adding the router's `--api-key` (outbound credential), 30 s stale, and no affinity: shared-prefix cache hit falls to round_robin's 65% (measured 2026-09-05), i.e. a cold 100K prefill on half the agent turns |
| `manual` + `x-smg-routing-key` | exact session affinity, but random placement and no escape hatch: strands worse than cache_aware. Keep for when affinity must be exact |
| `round_robin` | 65% hit instead of 97.7% |
| `consistent_hashing` | collapses onto one worker (shared Authorization) |
| `prefix_hash` | still blocked by `model_id: "unknown"` |
| `bucket` | splits by request length for prefill pools; not our problem |

## What would close the rest (gap 0-1)

The guard needs KV tokens and queue length, not request counts. None of the
shipped policies combine that with affinity. In order of cost:

1. **Make a move cheap instead of rare.** HiCache L3 (`--hicache-storage-backend`,
   docs/advanced_features/hicache_design.md) is shared across instances: if
   both replicas wrote to one host-side store, a session moved to the other
   replica would load its prefix from host memory (~1.2 s per 45K tokens here)
   instead of recomputing it (49 s). Then abs 1, or even token-aware
   `power_of_two`, costs little. Unverified: the built-in `file` backend is
   documented as a demonstration backend, and L3 support for this hybrid
   GDN/Mamba model at our pinned image must be tested before anything else.
2. **Put the direct-hostname traffic behind the gateway** (per-person keys,
   already planned). Then the 70K `max_tokens` cap and per-key limits apply to
   everyone, which bounds how long one answer can hold a replica.
3. **Patch cache_aware to read LoadMonitor** (`num_tokens + num_pending_tokens`,
   `num_waiting_reqs`). Small in Rust, but it is a fork of the router binary;
   worth an upstream issue first.

Applying abs 1 recreates the router and drops in-flight requests on both
replicas; do it when both engines report 0 running and 0 queued.
