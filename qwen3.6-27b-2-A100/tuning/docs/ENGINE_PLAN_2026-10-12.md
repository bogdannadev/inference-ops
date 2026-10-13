# Engine-level plan after patch 4 (2026-10-12)

Premise: on 2x A100 PCIe the gains come from the engine (what is computed,
when, by which existing kernel), not from new compute kernels. Decode is
bounded by bytes read per step, prefill GEMMs already run at ~70% of tensor
peak, and the largest prefill cost is work that should not happen at all
(cache misses). Evidence: `PREFILL_CACHE_2026-10-12.md`, `P1_PROFILE.md`, B1.

```
           read-only            drained r1, one gate run each              needs a new kernel
 ┌──────────────────────┐ ┌─────────────────────────────────────────┐ ┌──────────────────────┐
 │ E0 24 h re-measure   │ │ E2 concurrency ladder                   │ │ E6 NVFP4 + FP4       │
 │ E1 zero L3 hits      │→│ E3 dequant/GEMM overlap (patch 4b)      │→│    dequant (patch 6) │
 │                      │ │ E4 FA2 prefill shim                     │ │ E8 hd256 attention   │
 │                      │ │ E5 FP8 lm_head                          │ │    (only if E4 says) │
 └──────────────────────┘ └─────────────────────────────────────────┘ └──────────────────────┘
   no live change           each: user go -> drained r1 -> gates -> roll r1, r0
```

Rules for every step: develop in `deploy/enginepatch-next/`; drained r1 with a
verified `Cache flushed.`; a same-day control arm (never compare against an
older day's run); kill switch by env var; never edit `deploy/enginepatch/`
until the roll is approved.

**Standard gates** (as for patch 4): smoke 7/7; kvq long (8 docs 8K-128K) and
short (32 x 8K) with |ΔNLL| within spread; lookup 18/18; spec_eval correctness
5/5 x3, decode and accept length within spread; mem_stress 160K + 3x60K clean.

## E0. Re-measure the template fix (read-only, ~30 min, after 2026-10-13 12:20 UTC)

- Repeat the section-2 split of `PREFILL_CACHE_2026-10-12.md`: `engine.requests`
  joined to `gateway.requests` on `chat_id = rid`, prompts > 20K, new-user turn
  (`chat_round` +1) vs tool turn, records since 2026-10-12 12:20 UTC.
- Expect new-user-turn hit rate ~98% (was 81%), recompute p90 well under 30K.
- Also: context growth per user turn (the cost side: +1.4-4.7K expected).
- If the hit rate did not move: the template is not the only rewriter; diff
  rendered prompts of consecutive turns from one session.

## E1. Why the shared L3 serves zero hits (read-only first, ~2 h)

- 20 h: 15K KV pages and 549 GDN states written, 0 storage hits. ~6 misses
  (~400K tokens) were sessions whose context lived on the other replica.
- Hypothesis: a hit needs a GDN state at the match point; KV pages are written
  per page but GDN states only at a few points, so the matched prefix almost
  never ends where a state exists.
- Method: read the v0.5.21 HiCache storage lookup path (source at the pinned
  tag first, `research-upstream-docs-before-container`), then `l3_probe.py` on a
  drained replica: write a session on r0, request it on r1, log the L3 match
  length vs the deepest GDN state found.
- Output: a cause and a proposed fix (config, or an engine patch that writes a
  GDN state at each page-aligned request end). Fix is a separate go.
- Gain if fixed: the cross-replica misses (largest single tail item after E0),
  and it makes routing changes cheap later.

## E2. Concurrency ladder (drained r1, ~1 h)

- `--max-running-requests` / `--cuda-graph-max-bs-decode` at 4 (control), 6, 8
  with `worker_ladder.py`. Decode is byte-bound, so a bigger batch reads the
  weights once for more tokens.
- Watch: Marlin at verify M = 48-64, accept length, KV pool (182K per request
  class, `kv-pool-fits-one-agent-request`), mem_stress at the new batch.
- Win = aggregate tok/s up without per-request decode below the agreed floor.
  If it wins, router `--max-concurrent-requests` and `--queue-size` move with it.
- Largest expected throughput item on the list; no code.

## E3. Patch 4b: overlap dequant with the GEMM (~1 day)

- Today each slab is dequant then GEMM, serially. Dequant is memory-bound
  (~825 GB/s), the GEMM tensor-bound, so they can run side by side:

```
now      stream 0: [dq0][ gemm0 ][dq1][ gemm1 ][dq2][ gemm2 ]
4b       stream 1: [dq0][dq1]     [dq2]
         stream 0:      [ gemm0  ][ gemm1  ][ gemm2  ]          (events order slab reuse)
```

- Two scratch buffers (ping-pong, 2 x 128 MB, check mem_stress headroom),
  one side stream, events. Layers with one slab: dequant layer i+1 while
  GEMM i runs is the full version; start with intra-layer slabs.
- Expected: ~3 s of a 61 s 128K cold prefill, ~8% at 8K. Python only.
- Gates: standard + `bench_fp8_dequant.py` (bit-exact unchanged, faster per forward).
- Kill: if contention slows the GEMM more than the dequant hides.

## E4. FA2 for prefill attention (~2-4 days)

- B1: torch SDPA flash (FA2) 15-19% faster than FlashInfer paged hd256 at
  4K/60K/108K prefixes. Attention is ~20 s of a 128K cold prefill.
- Problem: pages are 64; FA2 wants contiguous K/V or 256-aligned pages.
  Candidates, measured in this order: (a) FA2 varlen over K/V gathered per
  chunk (gather cost vs 15-19% gain), (b) FA2 paged API with page 256 (a page
  size change touches the whole cache: risky), (c) stay on FlashInfer.
- Engine patch: route prefill extend attention of the 16 full-attention layers
  only; decode/verify stay on FlashInfer and in graphs.
- Expected: ~3-4 s at 128K, more at 262K. Gates: standard, lookup to 128K weighted.
- Output also feeds E8: ncu of FA2 vs FlashInfer on the real shape.

## E5. FP8 lm_head (~1 day)

- `lm_head.weight` is bf16, 2.5 GB read per verify step. Quantize to FP8
  (block 128) at load, serve through the same Marlin path. First verify
  whether the DFlash2 draft reads the target lm_head (then the saving doubles).
- Expected: ~3-5% decode. Gate: ΔNLL short/long first, it is the most
  quality-sensitive layer; abort on a significant shift.

## E6. NVFP4 + patch 6 (FP4 dequant for prefill) (~1 week)

(Renumbered from "patch 5" on 2026-10-14: patch 5 is now the per-key context/effort patch, `deploy/enginepatch/keypolicy.py`.)

- `nvidia/Qwen3.8-27B-NVFP4`: MLP + lm_head NVFP4 g16, rest FP8. Weights
  ~31 -> ~17 GB, decode maybe +25-40%, frees GPU memory for KV/GDN.
- Marlin W4A16 would reintroduce the prefill loss patch 4 removed, in the MLP
  (the largest GEMMs). So patch 6 = patch 4 for Marlin's FP4 layout: transcribe
  its repack and scale permutation, CPU bit-exact test through the real
  prepare function, GPU check against Marlin on identity rows, then bench.
- Order: trial arm without patch 6 (decode + quality) -> patch 6 -> trial with it.
- Gates: standard plus seq16; quality is the main risk (4-bit MLP).
- Note: the shared L3 dir is per weights format; give the trial its own dir.

## E7. Small tuning, slotted in when a replica is drained anyway

- DFlash2 `num-draft-tokens` sweep (6/8/10) on the faster target.
- Patch 4 `MIN_M` re-check if chunk size or graph ceiling changes.
- 262K context: after E2 and E6 (memory budget changes), mem_stress at 262K,
  groundwork in `tuning/next-session/CONTEXT_262144.md`.

## E8. Hand-written hd256 attention (decide after E4)

- Only if ncu after E4 shows the chosen kernel well below what SM80 allows
  for hd256 (register-bound: FlashInfer uses 239 regs/thread). Ceiling over
  FA2 ~10-15% of attention = ~2-3 s at 128K. Weeks of work: fund only if
  262K becomes a product requirement, where attention dominates.

## Not planned (measured, closed)

Decode megakernel (bubbles 5.9% < 25% gate), GDN kernel (<5%), thin-GEMM
kernels (ceiling ~19%, blocks not the limit), host-side scheduling (step
~50 ms absorbs it), tensor ring, faster dequant via word loads (E3 hides it).

## Expected outcome

| area | now | after |
|---|---|---|
| new-user-turn hit rate | 81% | ~98% (E0), cross-replica misses closed (E1) |
| cold prefill 128K | 61 s | ~53-55 s (E3 + E4), NVFP4 neutral with patch 6 |
| cold prefill 8K | 2.5 s | ~2.2-2.3 s (E3) |
| decode per request | baseline | +3-5% (E5), +25-40% (E6, if quality holds) |
| aggregate throughput | baseline | E2 decides |

Estimates are derived from today's bench and B1, not measured; each step
replaces its line with a measurement.
