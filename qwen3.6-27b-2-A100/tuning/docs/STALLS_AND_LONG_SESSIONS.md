# Replica stalls and day-long sessions — 2026-10-02

What a stalled replica does to clients today, what shortens it, and what a 24-hour agent
session needs from the cache. All measurements below were taken on 2026-10-02.
**Applied 2026-10-02 15:45Z:** measures #1, #2 and #3 (SPF, Rust core and watchdog 180 on
both replicas; router health 15 s / 2 / 10 s). #4–#6 and the large L3 are still open.

## 1. History: what has actually stalled

| Source (14 days) | Result |
|---|---|
| Router health checks (`smg_worker_health_checks_total`) | 39,352 successes, **0 failures** |
| Circuit-breaker opens | 3, all on r0, all during the designer IMAGE-token 500s (09-21/22) |
| Earlier stalls | 2026-09-16: orphaned requests (fixed upstream, `orphaned-requests` memory). 2026-09-25: KV stranding (fp8 + L3 since). 2026-09-25: host freeze, 3.5 h, no alert fired |

None of the incidents so far was a hard hang. Each was either a stall in which the
replica stayed healthy, or the whole host freezing.

## 2. Drill: what each kind of stall does to clients

Harness: `tuning/bench/stall_rig/`. It runs no GPU and does not touch production:

- two fake workers that speak the engine API;
- a throwaway router built from the **production router image with the production flags**;
- 6 serial, cache-affine agent sessions plus a new session every 20 s;
- an isolated docker network.

The fault hits fw1 at t=120 s and is undone at t=480 s. Results are in
`tuning/results/stall_drill/`. "tuned" is the production flags plus
`--health-check-interval-secs 15 --health-failure-threshold 2 --health-check-timeout-secs 10`.

| Fault | Config | Client requests hurt | What clients saw | Router stopped sending to fw1 after | fw1 used again after recovery |
|---|---|---|---|---|---|
| crash (`docker stop`) | prod | 12, across 7 sessions | HTTP 500 | 132 s | 116 s |
| crash | tuned | 9, across 5 sessions | HTTP 500 | **6 s** | **25 s** |
| scheduler hang (HTTP up, `/health` 503 after 20 s, no bytes) | prod | 4 sessions | hung for the whole run (608 s+). In production: 900 s, then 504 | 23 s | not within 240 s |
| scheduler hang | tuned | 4 sessions | same | 23 s | 63 s |
| freeze (`docker pause`) | prod / tuned | 3 sessions | hung until unfrozen (363 s). In production: 900 s, then 504 | 23 s | 118 s / 24 s |
| stranded (healthy, 120 s queue) | prod | the same 3 sessions, every turn | **120 s wait per turn for 6 minutes. Never moved** | — (health stays green) | — |
| stranded + engine waiting timeout 30 s (→ 503) | prod | 19 requests, across 11 sessions | HTTP 503 after 30 s, **not retried by the router** | — | — |

**How to read it:**

1. **Detection is the smaller problem.** New sessions stop landing on a hung replica within
   ~23 s even with prod flags. Hung requests count as router load, and
   `--balance-abs-threshold 1` diverts traffic away from it. Faster health checks help
   in two places: a crash (132 s → 6 s of 500s) and coming back after recovery
   (~2 min → 25 s).
2. **The damage is the requests already on the stuck replica.** They hang until Envoy's
   900 s idle timeout. No router setting frees them. Only the engine process dying does,
   because that resets the connections.
3. **A stranded replica keeps its sessions.** Cache affinity plus green health means the
   same users wait the full strand on every turn.
   - `SGLANG_REQ_WAITING_TIMEOUT` (an engine abort with 503) does not move them.
   - `sglang_router` does not retry streaming requests: every 503 reached the client after
     exactly 30 s, in one attempt.
   - Production's 80 router retries in 14 days are therefore non-streaming requests.

## 3. Today's settings, per stall class

| Class | Detection | Self-healing | Worst case for a client |
|---|---|---|---|
| Crash | breaker / health: up to ~2–3 min of 500s | restart policy, then ~4 min warm boot | 500s, OpenAI SDK retries ×2 |
| Forward / scheduler hang | load guard ~23 s; health 3×60 s | `--watchdog-timeout 1200` → SIGQUIT → restart. **20 min**, a value inherited from an April config for another model | hung 900 s → 504 |
| Process / host freeze | same as above | **none.** Docker marks the container unhealthy after 10 × 60 s and nothing acts on it | hung 900 s → 504; the replica stays dead |
| Stranded queue | `EngineQueueStrandedOnOneReplica` alert | none | waits the full strand on every turn |

**False "unhealthy" risk.**
- The engine's `/health` waits up to 20 s for any scheduler output; the router gives it 5 s.
- Under hrrn, a replica doing a lone long cold prefill emits nothing for up to ~75 s, and
  fails health while working.
- It never fired in 14 days: it needs 3 consecutive misses 60 s apart.
- Tightening health checks without `shortest-prefill-first` would make it fire. SPF serves
  the 1-token probe at once (it is the shortest prefill).

## 4. Measures, in order

| # | Measure | Fixes | Cost / risk | Where |
|---|---|---|---|---|
| 1 | Converge r0 to v0.5.21 + SPF + Rust core | prerequisite for #3 (no probe starvation); stops the queue behind a cold prefill (40.9 s → 2.1 s) | one drained roll | r0 |
| 2 | `--watchdog-timeout 180` (+ `--soft-watchdog-timeout 60`) | a hang becomes a restart after 3 min instead of 20. Stuck connections reset, so clients retry instead of waiting 900 s | Is it safe? The longest legitimate forward is one 4096-token chunk (~2–3 s). r1 sat idle for an hour at 1200 without firing, so an idle scheduler does not trip it. The soft dump uses py-spy (0.4.2 in the image); Yama `ptrace_scope=1` and no CAP_SYS_PTRACE may block it, so verify on a drained replica | both replicas |
| 3 | Router `--health-check-interval-secs 15 --health-failure-threshold 2 --health-check-timeout-secs 10` | crash: 132 s → 6 s of 500s; recovery 116 s → 25 s (measured) | the probe is one generated token, negligible. Only after #1. Router recreate drops in-flight streams on both replicas, so do it at 0 running | router |
| 4 | Autoheal sidecar (restart containers marked `unhealthy`) + healthcheck `interval 30s, retries 4, timeout 25s` | freezes and hangs the watchdog cannot see, e.g. a whole-process stop: dead forever → restart after ~2 min | the sidecar needs the docker socket. A false positive costs a ~4 min reboot of one replica, so keep it conservative and only after #1 | compose |
| 5 | Host: `kernel.softlockup_panic`, `panic`, a hardware/systemd watchdog; off-host dead-man alert | the 2026-09-25 class: a 3.5 h freeze with zero alerts | needs root; see the `host-freeze-2026-09-25` memory | host |
| 6 | Queue-aware routing | stranding (the only class that keeps hitting the same users) | `sglang_router` cannot do it. The experimental Rust router in v0.5.21 has `--worker-queue-limit` / `--saturation-queue-floor` (#39168/#39169) plus abort-on-disconnect (#39461). Evaluate it in `stall_rig` first | router |
| — | `SGLANG_REQ_WAITING_TIMEOUT` | — | **rejected:** turns a wait into a client-visible 503 the router does not retry | — |

## 5. Day-long sessions

### How long a conversation stays cached

Source: `engine.requests`, named consumers, prompts > 40K, since L3 went live on 2026-09-25.
The gap is the time since the same consumer's previous request.

| Gap | Turns | Prompt cached | Mostly cold | TTFT p50 |
|---|---|---|---|---|
| < 1 min | 6,347 | 97% | 62 | 1.0 s |
| 1–10 min | 1,208 | 81% | 189 | 3.0 s |
| 10–60 min | 62 | 73% | 13 | 5.3 s |
| 1–4 h | 8 | 48% | 4 | 14.2 s |
| > 4 h | 10 | 27% | 7 | 25.4 s |

`designer` was active in 130 of the last 336 hours, so day-long sessions are real.

There were 430 requests above 120K tokens in 14 days. That is compaction pressure, and
after a client-side compaction the prefix changes and the next turn is cold whatever the
cache holds.

### Why L3 does not carry them

| Tier | Capacity per replica | Share of cached tokens, 6 days |
|---|---|---|
| device (fp8 KV) | 282,496 tokens | 613M (95.4%) |
| host (`--hicache-ratio 8`) | ~2.26M tokens; GDN states 43 × 8 | 29M (4.5%) |
| L3 file, tmpfs, 60 GB per process | ~1M tokens (~62 KB per token: target KV, draft KV, GDN sidecars) | **0.56M (0.09%)** |

- L3 is smaller than the host tier it sits behind, so whatever host evicts is already
  gone from L3.
- Its prefetches never failed (only `below_threshold`, ~2K tokens).
- Host evicted 33–38M tokens per replica in 6 days.

### Levers

1. **A large, shared, disk-backed L3.**
   - Size: 400–600 GB on `/` (1.1 TB free) is ~6–10M tokens, ~80–120 agent contexts.
     That is enough to bridge overnight gaps.
   - Speed: direct reads measured 4.8–10 GB/s on this VM disk (the hypervisor cache may
     flatter that). An 80K context is ~5 GB, so a reload takes ~1–2 s against 50–75 s cold.
   - It survives every restart, so it pairs with #2/#4: a watchdog or autoheal restart
     comes back warm.
   - Both replicas must point at one directory once they run the same version. Moving a
     session between replicas is then cheap, which also softens stranding.
   - **To check first:** HiCacheFile keeps a flat directory and scans it at boot (10,395
     entries for 52 GB → ~120K for 600 GB); the boot time that adds; the file count on ext4.
   - Trial on drained r1 with `tlru_probe` / `cache_replay` and an L3-only return after a
     flush.
2. **Rust TreeCore** (live on r1). Its cache decisions are identical to the Python core:
   - `cache_replay`: same cached tokens on every turn;
   - same T-LRU behaviour and GDN checks;
   - scheduler CPU 686 s vs 666 s, inside noise.

   It does not change hit rates. It keeps us on upstream's default path, where the Python
   core will lag.
3. **No single request can be day-long.** Router and gateway cut a request at 900 s, so a
   session is many turns, and the cache tiers above are what decide its cost.
4. **Uptime.** The replicas ran 6–7 days between rolls with no visible degradation. L1/L2
   are lost on every restart; only L3 survives one.
