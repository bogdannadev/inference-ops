# Cache handoff between replicas: plan (2026-10-17)

Goal: a session that moves from one replica to the other (a drain for a roll,
a crash, a cross-replica route) resumes from cache instead of re-prefilling
its whole context. The path is L1 -> L2 -> L3 on the source, L3 -> the target.
Mid-turn moves (a stream continuing on the other replica) come later, in smg;
see `../../stream-migration/README.md`.

## What was measured

Shared L3 (`/tmp/hicache`, file backend, tmpfs, `max_size 60G` per process):

```
 oldest file 17:38 UTC, now 17:58 UTC      -> ~20 min retention
 KV pages     33,794 files x 3.3 MB  = 110 GB   (64 tokens each; draft KV pages 0.65 MB)
 GDN states      371 files x 78 MB   =  29 GB   (one state = the space of ~24 KV pages)
```

Served by tier since 2026-10-14 02:00 UTC (engine.requests):

| | L1 device | L2 host | L3 shared |
|---|---|---|---|
| r0 | 663 M | 35 M | 1.1 M (36 requests) |
| r1 | 643 M | 29 M | 0.9 M (36 requests) |

On this hybrid model an L3 hit must end on a page whose GDN state is stored
too (`PoolHitPolicy.TRAILING_PAGES`, `mem_cache/unified_cache/components/mamba.py`).
v0.5.21 writes a node to L3 when its host write-through is acked
(`_finish_write_through_ack` -> `write_backup_storage`), with the GDN state
keyed by the node's last page hash. So states do reach L3, but at 78 MB each
the 60G cap evicts everything within ~20 minutes: L3 is a capacity problem,
not a write bug. Host RAM: 1007 GB, 477 GB shared (tmpfs + L2), 325 GB available.

No request can move mid-decode inside SGLang: `/pause_generation` (abort,
retract, in_place) only pauses on the same engine, and the router never
retries or splices a stream.

## Phases (r1 only; r0 runs stock and only reads L3)

| | What | Gate |
|---|---|---|
| P1 | Measure, read-only: L3 write rate per pool and retention over a day; inter-turn gap per session; cross-replica turns and what L3 at retention X would have saved | a target retention and the choice of P2 options, with numbers |
| P2 | Patch 7, L3 policy for the hybrid model. Candidates: (a) only turn-ending GDN states (prompt end, output end) go to L3, not the 4096-chunk checkpoints; (b) no draft KV in L3; (c) a larger cap inside a RAM budget Danila sets | CPU verify; L1/L2 behaviour provably unchanged |
| P3 | Patch 8, `POST /handoff_flush` on the draining replica: recent leaves newest first, L1 -> L2 -> L3 including the leaf GDN state, wait for the acks | CPU verify |
| P4 | Test on drained r1 alone: replay multi-turn sessions (`tuning/bench/cache_replay.py`), flush, recreate r1 (L1 and L2 gone, as on a move), replay the next turns; control without the flush | next turns hit L3 up to their tail; outputs and decode unchanged |
| P5 | Roll r1 with patches 6+7+8 through an r1-only compose override mounting `deploy/enginepatch-next/`, drained with the fixed `roll-replica.sh`, flushed before recreation | r1 rows show `diag_*` and `cached_storage` as expected |

Later: r0 the same way, on its own go; then engine patch 9 (resume) with the
smg migration.

## Roll procedure (Danila, 2026-10-17)

One replica at a time. Take it out of the router, let it finish every request
it holds while the other one takes (and queues) the new ones, then update it.
`deploy/roll-replica.sh` now waits on the replica's own running + queued count
(`DRAIN_TIMEOUT`, default 1800 s) and aborts without recreating if it is not
idle in time. Test on one replica before the other, never both.

## State

- Patch 6 (cache diagnostics): written and verified (`deploy/enginepatch-next/`),
  ClickHouse `diag_*` columns applied, Vector mapping applied 2026-10-17 17:47
  UTC; engine roll pending.
- P1: not started.
