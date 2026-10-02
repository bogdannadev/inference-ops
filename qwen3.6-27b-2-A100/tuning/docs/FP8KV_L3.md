# fp8 KV + shared HiCache L3 — 2026-09-25

LIVE on r0 and r1 since 2026-09-25 ~17:20 +05. Drained trial on r1 first, then r0.

## Why

On 2026-09-25 EngineQueueStrandedOnOneReplica fired twice. A 182,528-token KV pool
holds only one 90–145K agent turn. Two concurrent large turns on one replica
serialize (waits of 31–163 s) while the other replica idles. The router guard
counts requests, not tokens.

- fp8 KV raises the pool to 282,496 tokens, so two such turns fit.
- L3 makes a session that does move (guard fires, roll, restart) reload its
  prefix instead of recomputing it.

## Change (both replicas)

```
--mem-fraction-static 0.92            (was 0.94: the DFlash2 draft pool copies the target token count)
--kv-cache-dtype fp8_e4m3
--hicache-storage-backend file
--hicache-storage-backend-extra-config '{"max_size":"60G","min_free_space":"100G"}'
volume /dev/shm/qwen36-hicache-l3/fp8_e4m3:/tmp/hicache      (shared, host tmpfs)
                                                             (since 2026-10-02: .../qwen36-hicache-l3-v0521/fp8_e4m3, see UPGRADE_v0.5.21.md)
```

**The L3 directory is per KV dtype, on purpose.** HiCacheFile keys are content
hash + served model + TP rank (v0.5.20 `hicache_storage.py`). The key has no
dtype, so a bf16 replica would read fp8 bytes. Rolling back one replica means
pointing it at another directory.

**Caps are per process.** Each LRU evictor only accounts for its own writes plus
what it scanned at boot, so the total can reach ~2 × 60 GB. The `min_free_space`
floor protects the 504 GB tmpfs.

## Results (drained r1 unless noted; `tuning/results/`)

| gate | bf16 v0.5.20 | fp8 + L3 |
|---|---|---|
| KV pool | 182,528 | **282,496** |
| `available_gpu_mem` | 2.80 GB | **4.37 GB** |
| L3 attach | — | `UnifiedRadixCache hybrid_ssm=True hicache_attached=True` |
| L3 hit after `/flush_cache` (78K prompt) | — | 77,952 cached, TTFT 5.2 s vs 32.0 s cold, needles correct |
| L3 restore exactness (3 × 40K prefix, 500 greedy tokens) | — | byte-identical text and identical accept length, device hit vs L3 hit |
| **cross-replica** (seed r1, first sight on freshly booted r0, 41.7K) | — | 41,728 cached, TTFT 2.8 s vs 14.9 s cold, needles correct |
| spec_eval kv/order/arith, no leaks | 18/18 | 30/30 |
| spec_eval `seq` (300 terms, ~7K tokens exact) | greedy 3/3 | **greedy 1/5**, prod 2/5, the rest hit the 8K cap |
| decode tok/s short c1/c4 | 83.5 / 67.6 | 76.5 / 63.8 |
| decode tok/s long c1/c4 | 66.5 / 57.7 | 75.2 / 52.4 |

The `seq` regression and the +11–20% cold prefill at 64–128K (software fp8
dequant on SM80, `TRIAL_FP8KV_MAMBA.md`) are known costs. The operator accepted
them for the capacity.

What L3 stores per 64-token page: target KV, DFlash2 draft KV (`.draft`), and a
GDN state (`.mamba`) at the checkpoints. That came to 9.7 GB for two ~78K
contexts. Production traffic wrote 54 GB within ~20 minutes of r1 going live.

`/flush_cache` resets L1+L2 only; L3 survives. Clearing L3 is a separate call.

## Rollback

Per replica, with `deploy/roll-replica.sh`: drop the three flags and the volume,
and set mem 0.94. To keep L3 on bf16, use a `bf16` directory, never `fp8_e4m3`.

Probes: `tuning/bench/l3_probe.py` (same-replica: flush then replay; with
`--seed-host`, cross-replica on drained workers).
