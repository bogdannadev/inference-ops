# HiCache + DFlash2 — 2026-09-13

**STATUS: validated on r1 (drained), then applied.** Two changes requested
together: HiCache (host-RAM KV tier) to cut TTFT under growing load, and
DFlash2 speculative decoding to cut decode time. EAGLE3 and DSpark were
evaluated and not pursued (end of this file).

Engine unchanged: `lmsysorg/sglang:v0.5.19-cu130` (`sha256:d6e72886…`).
Draft: `incoai/Qwen3.8-27B-DFlash2` at revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4` (weights byte-identical to the
`z-lab` mirror; sha256 `67fc76d6…`).

## Why HiCache again, when July removed it

July removed HiCache at 0.43% host hits: one hot shared system prompt never
left the device pool, so nothing was evicted and re-requested. The workload is
different now.

Production, 2026-09-12/13 (Prometheus, both replicas):

| | 09-08 | 09-12 | 09-13 |
|---|---|---|---|
| TTFT p95 | 7.9 s | 29.0 s | 26.8 s |
| e2e p95 | 55 s | 175 s | 181 s |
| `token_usage` peak | 0.98 | 1.00 | 1.00 |

- TTFT is prefill of uncached tokens: `uncached_prompt_tokens` p95 ~39K,
  `chunked_prefill` p95 ~15 s. Queue time is ~0.
- e2e is output length: p95 generation 8.8K tokens at ~58 tok/s ≈ 150 s. Only
  decode speed moves it.
- One consumer (94% of tokens) runs several agent sessions in parallel, each
  50-140K tokens. The 169K device pool holds one or two, so sessions evict each
  other and recompute their whole context on the next turn. This is the
  "revisit when token_usage approaches 1.0" trigger from `UPGRADE_v0.5.19.md`.

`tuning/bench/hicache_sim.py` replays `gateway.requests` (sessions
reconstructed from monotonically growing prompts) through a per-replica LRU
cache. Device-only it predicts 4.05M recomputed tokens over the last 24 h
against 3.92M actually reported by the engine (3%). With a host tier:

| per day | device only | + host tier |
|---|---|---|
| recomputed prompt tokens | 4.05M | 2.04M |
| p95 recompute per request | 37K | 15K |
| requests recomputing >16K | 49 | 25 |

Ratio 2, 3 and 4 give the same result, so the host tier is not the limit.

## The DFlash2 case

On this node decode is bandwidth-bound (`a100-node-decode-is-bandwidth-bound`):
only bytes read per forward matter. EAGLE/MTP 5-step runs five draft passes,
each through the 2.5 GB `lm_head`, plus a draft-extend. DFlash2 drafts a block
of 8 in one pass of a 5-layer drafter and reuses the target `lm_head`. About
71 GB -> 57 GB of weights read per step, with a higher accept length.

## What was run, in order (all on r1, drained from the router)

| run | config | result file |
|---|---|---|
| control | production (MTP, no HiCache) | `hicache_probe_r1_mtp_nohicache`, `spec_eval_r1_mtp_nohicache{,_correct2}` |
| A | + HiCache ratio 3 | `hicache_probe_r1_hicache_r3`, `spec_eval_r1_mtp_hicache` |
| B1 | + DFlash2, chunk 16384, mem 0.92 | **OOM on first 45K prefill** |
| B2 | + `--speculative-draft-window-size 2048` | same allocation; not tested further |
| B3 | chunk 4096, mem 0.92 | `hicache_probe_r1_dflash2_chunk4096`, `spec_eval_r1_dflash2_hicache` |
| B4 | + fp8 draft KV, mem 0.94 (**applied**) | `mem_stress_r1_dflash2_mf094`, `hicache_probe_r1_dflash2_final`, `spec_eval_r1_dflash2_final` |

Harness: `tuning/bench/run_eval.sh <replica> <script> <label>`, stdlib-only
scripts in `tuning/bench/` (`evalkit.py` shared).

### HiCache reload probe (`hicache_probe.py`)

45K-token prompt, then 4x52K unrelated prompts to evict it, then the same
prompt again.

| config | cold | device hit | after eviction | host tokens |
|---|---|---|---|---|
| control | 14.37 s | 0.50 s | **14.61 s** (recompute) | 0 |
| A: MTP + HiCache | 14.21 s | 0.68 s | **1.06 s** | 44,992 |
| B3: DFlash2 + HiCache | 14.25 s | 1.40 s | **1.84 s** | 44,992 |
| B4: applied | 14.33 s | 1.44 s | **1.73 s** | 44,992 |

The Mamba states come back too (the hybrid model would otherwise recompute
48 of 64 layers). Upstream #33713 claimed the unified tree drops hybrid nodes
on eviction. At v0.5.19 `_demote` keeps them, and the probe confirms it.

DFlash2 makes a cache hit ~0.8 s slower: the drafter rebuilds its own state
for the prefix.

### Correctness gate (`spec_eval.py`)

Thinking on (production default `reasoning_effort xhigh`), concurrency 4 (=
`--max-running-requests`), greedy and production sampling (T 0.7, top-p 0.95,
top-k 20). Every task has an objective answer:

- **kv:** a 12K-token table lookup. A foreign code in the answer is a
  cross-request LEAK (#36548).
- **order:** 5-name ordering.
- **arith:** a 12-term sum.
- **seq:** a 300-term arithmetic sequence, ~7K generated tokens.

Plus two identical greedy requests from a flushed cache, which must match
(#38009).

| config | greedy | prod | leaks | serial determinism |
|---|---|---|---|---|
| control (MTP) | 20/20 ok | 16 ok, 4 truncated | 0 | 2/2 |
| A (MTP + HiCache, 3 rounds) | 11 ok, 1 truncated | 10 ok, 2 truncated | 0 | 2/2 |
| B3 (DFlash2) | 20/20 ok | 16 ok, 4 truncated | 0 | 2/2 |
| B4 (applied, 3 rounds) | 12/12 ok | 10 ok, 2 truncated | 0 | 2/2 |

"truncated" is the seq task hitting the 8,000-token cap. It is identical across
MTP and DFlash2, so it is the model's reasoning length, not a failure.

Found while building the gate: identical greedy requests are **not**
byte-identical when the second one hits the prefix cache, on MTP too. A cache
hit changes prefill shapes and flips near-ties. The determinism check flushes
first.

### Speed (`spec_eval.py`, 1024 tokens, ignore_eos, production sampling)

decode tok/s median per request / aggregate tok/s / accept length:

| phase | control MTP | A: MTP + HiCache | B3: DFlash2 | B4: applied |
|---|---|---|---|---|
| short c=1 | 50.6 / 49.2 / 2.55 | 51.4 / 49.8 / 2.59 | **69.8 / 65.6 / 3.06** | **70.4 / 69.6 / 3.25** |
| short c=4 | 47.6 / 176.8 / 2.73 | 45.5 / 165.9 / 2.69 | **66.4 / 246.5 / 3.52** | **64.0 / 242.7 / 3.37** |
| ~55K ctx c=1 | 46.9 / 44.7 / 2.60 | 49.9 / 49.4 / 2.88 | **59.5 / 57.7 / 2.98** | **65.2 / 61.1 / 3.16** |
| ~55K ctx c=4 | 39.2 / 132.7 / 2.68 | 37.3 / 137.9 / 2.69 | **53.5 / 189.7 / 3.19** | **55.2 / 184.9 / 3.21** |

HiCache is speed-neutral; the A-vs-control deltas move with accept length,
i.e. sampling noise at n=4-8. fp8 draft KV (B4) did not cost accept length.
Applied config vs production MTP: +39% single-stream, +37% c=4 short, +39% /
+39% at ~55K context c=1 / c=4 aggregate.

## Memory: why three more flags moved

v0.5.19 allocates the draft KV pool with the **same token count as the target
pool, out of activation headroom, not the static budget**. EAGLE's 1-layer
draft cost 0.64 GB; DFlash2's 5 layers cost 2.74 GB at bf16.
`--speculative-draft-window-size` limits draft attention but does NOT shrink
that allocation (B2: identical boot numbers, `compact_cache=True`).

| | MTP (before) | B1/B3 @0.92 | B4 @0.94 + fp8 draft KV |
|---|---|---|---|
| draft weights | 5.53 GB (MTP loads its own embed/head) | 3.71 GB | 3.71 GB |
| target pool | 169,408 | **143,872** | **182,528** |
| Mamba slots | 43 | 40 | 43 |
| draft KV | 0.64 GB | 2.74 GB | 1.74 GB (fp8) |
| `available_gpu_mem` | 8.49 GB | 4.47 GB | 2.91 GB |

- **B1** died on its first 45K prompt: `torch.OutOfMemoryError` in
  `fla/chunk_o.py` (GDN prefill), 16384-token chunk. A Triton kernel was
  loaded after serving started with 0.43 GiB free. The container went zombie
  on stop and exited on its own seconds later; GPU memory was released.
- **B3** at 4096-token chunks served everything, and cold 45K TTFT was
  unchanged. But the 143,872-token pool is below `--context-length 169000`:
  27 of 1,022 production requests since 2026-09-04 exceeded 143K (max
  168,998). They would 400 on that replica.
- **B4:** fp8 draft KV halves the draft pool and cannot change output, since
  the target verifies every token. mem 0.94 restores the pool above the
  context length (0.95 overshot to 191,680 with 2.10 GB headroom).
  - Stress (`mem_stress.py`): a 160K cold prompt plus three 30K requests
    decoding at once, twice. 0 errors, 0 OOM.
  - The 160K cold prefill took 96-127 s under that contention.

Headroom is now 2.91 GB against 8.49 GB before. Watch for `OutOfMemoryError`
in worker logs. The first response would be `--chunked-prefill-size 2048`,
then `--mem-fraction-static 0.935` (pool ~174K).

## Host RAM

Per replica: KV host 35.9 GB + Mamba host 10.2 GB + draft KV host 5.6 GB ≈
52 GB pinned, and about +60 s of boot for the allocation. Host had 816 GB
available before; ~768 GB with r1 on HiCache.

## Not pursued

- **EAGLE3:** no Qwen3.8-27B EAGLE3 draft exists (only community Qwen3.6-27B
  ones). Training one needs GPUs this node does not have spare.
- **DSpark** (`RadixArk/Qwen3.8-27B-DSpark`): trained against the NVFP4 target.
  Its accept length was below the native MTP head in z-lab's head-to-head.
  Full-attention draft KV makes the memory problem above worse.

## Open upstream items to re-check on the next engine bump

- #36548 (DFlash2 context corruption under concurrency) and #38009 (greedy
  divergence with thinking). Not reproduced here, but re-run `spec_eval.py`.
- #36014 (GDN target-verify beta precision) affects every spec algorithm.
- #30314 (scheduler hang on Mamba eviction with HiCache, large contexts).
