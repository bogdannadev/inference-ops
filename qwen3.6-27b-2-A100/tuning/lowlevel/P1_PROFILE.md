# P1 — top-down profile before any kernel work

`STATUS: STAGE A RUN 2026-09-24. Decode and GDN closed; prefill attention (41% of peak, up to 60% of prefill time) is the lead for Stage B.`

One drained window on r1 answers two questions that decide whether any
hand-written kernel (down to PTX) is worth funding on this node:

| track | question | decides |
|---|---|---|
| **decode** | How much of a DFlash2 decode step is lost *between* kernels? | megakernel prototype, yes/no |
| **prefill** | What share of prefill wall time is each op, at 4K / 32K / 128K / cached-prefix extension? | which of `PREFILL_KERNELS.md` Phases 3–5 to fund |

The prefill track is `PREFILL_KERNELS.md` §5 Phase 1, executed. The decode
track is new: E1 showed the per-projection loss is ramp/drain on matrices too
small to reach streaming peak, and the only structural answer to that is
removing kernel boundaries, which is only worth attempting if the boundaries
cost enough.

## What does NOT change

No container is recreated and no flag moves. r1 keeps its production config
(DFlash2, HiCache ratio 8, chunk 4096, 43 GDN slots). It only leaves the router
for the window, exactly as in E1/E2 and the July S1 profile. The SYS_ADMIN
capability that `ncu` needs goes on a **separate throwaway container** in
Stage B, never on the replica.

## Stage A — engine traces (drained r1, ~12 min)

```bash
./tuning/lowlevel/bench/p1_profile.sh 20260924
```

`p1_profile.sh` drains r1 (refuses if r0 is not healthy), stops the DCGM
exporter, runs `tuning/bench/p1_profile.py` against r1 directly, and on ANY
exit — including Ctrl-C — restarts DCGM and re-registers r1. Then it runs the
analyzer over each trace.

| scenario | load | captured |
|---|---|---|
| `decode` | 4 concurrent greedy chat streams, 700 tokens, thinking off | 40 forwards mid-decode |
| `cold4k` | cold 4,096 tokens, `max_new_tokens` 1 | whole request |
| `cold32k` | cold 32,768 | whole request (8 chunks) |
| `cold128k_a` | cold 131,072 | first 4 chunks |
| `cold128k_z` | cold 131,072 | 4 chunks from ~85% through (timed; see below) |
| `ext16k` | 16,384 new tokens on a cached 65,536 prefix | whole extension |

Rules carried over, each of which has cost a void run before:

- **The flush is verified before every scenario.** Only the literal
  `Cache flushed.` counts; otherwise the scenario aborts instead of measuring
  warm against cold.
- **`with_stack` is off.** The S1 trace left it at the default and is 793 MB
  gzipped. The analyzer streams traces line by line regardless.
- **Shares, not timings.** Clocks are unlocked (root needed), so this stage
  only reports ratios, which survive clock drift. GPU 1 clocks/temperature are
  logged before and after.
- **`cold128k_z` is timed, not counted.** `/start_profile`'s `start_step` is an
  absolute forward count that cannot be read from outside, so the capture
  starts at 85% of the measured 67.8 s cold time. Where it actually landed is
  visible in the trace from the attention-kernel durations, which grow with
  position.

### Analyzer

`bench/p1_trace_analyze.py` buckets kernel time by name, lists the top kernels
with grid and registers, and separates **in-step bubbles** (gaps < 1 ms between
kernels while work is running) from no-load gaps.

Validated against the July S1 trace before use: GDN recurrent reads **2.48%**
(recorded 2.49%). The bucket boundaries differ slightly from the July manual
bucketing (FlashInfer's fused rmsnorm is `norm_rope` here, the mamba state
scatter is `gdn_state`), so compare shares within this campaign, not against
July.

## Stage B — Nsight Compute on the hot kernels (drained r1, side container)

Only for kernels Stage A puts above the gate. `ncu` needs the hardware
counters that `RmProfilingAdminOnly: 1` reserves for SYS_ADMIN, so it runs in
a one-off container from the same pinned image on GPU 1 while r1 is drained,
replaying the kernel standalone at the production shape:

```bash
docker run --rm --gpus '"device=1"' --cap-add SYS_ADMIN \
  -v "$PWD/tuning/lowlevel:/ll" <pinned sglang image> \
  ncu --set full -k regex:'<kernel>' -c 10 -o /ll/results/ncu_<kernel> \
      python3 /ll/bench/<harness>.py
```

- **r1 must stay drained.** `ncu` locks the GPU to base clocks while it runs
  (`--clock-control base`), which would slow a serving replica.
- **Memory budget.** r1 keeps its memory while drained: 79,854 of 81,920 MiB
  in use on 2026-09-24, so a harness gets about **2 GB**. E1 ran in that
  budget. Size every harness to it, keeping tensors at production shape; an
  OOM in the side container cannot touch r1, but it wastes the window.
- **What to read:** achieved occupancy, register spills, `cp.async` and
  `ldmatrix` counts, and tensor-pipe utilisation. Stage C then reads the SASS.

## Stage C — SASS audit (no GPU)

`cuobjdump -sass` / `nvdisasm` on the Triton cache cubins, as E3 did for the
decode kernel. It checks whether the instructions we would hand-write
(`mma.sync.m16n8k16`, `ldmatrix`, `cp.async` + `commit/wait_group`) are already
emitted, and where registers spill. Verify every PTX idea against
`refs/PTX_SM80_VERIFIED.md` first. Per-instruction L2 eviction hints do not
assemble on sm_80, and E2 already rejected L2 residency, so cache-policy work
is off the list.

## Decision gates

| result | action |
|---|---|
| decode: in-step bubbles + GEMM share × 19% (E1 ceiling) **≥ 25%** of step time | megakernel prototype, standalone first: one layer type, then a full forward outside SGLang |
| decode: below 25% | close the decode track; the lever stays quantization (bytes) |
| prefill: `gdn_chunk` + `conv1d` **≥ 10%** at 32K or 128K | Stage B/C on those kernels → `PREFILL_KERNELS.md` Phase 4 (fuse the WY chunk stages, then PTX on the hot path) |
| prefill: GEMM dominant, epilogues visible | Phase 3 (prefill GEMM + epilogue fusion) |
| prefill: attention ≥ 10% only in `cold128k_z` | Phase 5/6 candidates, long context only |
| any op < 10% | no custom kernel, per the standing stop rule |

## Results

### Baseline from existing data: the July S1 trace, re-read

This was the EAGLE 5/6 config, not today's DFlash2, so it only shows what
order of magnitude to expect.

```
active 7.359 s   busy 88.7%   in-step bubbles 11.3%
  bubbles by gap:  <2us 0.3%   2-10us 0.1%   10-100us 3.2%   0.1-1ms 7.7%
gemm 89.3% of kernel time   gdn_recurrent 2.5%   copy 2.2%   norm_rope 2.0%
```

- **Bubbles cluster at 0.1–1 ms.** That is the size of a host sync or a
  CUDA-graph replay boundary (draft → verify → sampling), not kernel launch
  latency, which would sit under 10 µs. Inside a graph, kernels run back to
  back.
- **The gate formula sits right at the line.** 11.3% + 88.7% × 89.3% × 19% ≈
  **26%**, against the 25% megakernel gate.
- **Caveat on that 26%.** Removing host-sync bubbles needs graph or scheduler
  changes, not a megakernel. Removing ramp/drain needs a megakernel. So the
  26% counts two separate efforts. The DFlash2 capture has to answer how it
  splits.

### Stage A — RUN 2026-09-24 on drained r1. Decode closed; attention is the lead

Two drained passes: `20260924` (decode, `cold128k_a`) and `20260924b` (the
prefill captures redone after the two faults below). GPU 1 held 1410 MHz at
53–61 °C through both. r1 was re-registered and DCGM restarted after each pass,
and the router was verified at 2/2 healthy both times.

Share of GPU kernel time (`tuning/lowlevel/results/p1_*.txt`):

```
                 decode  cold4k  cold32k  128K 0-16K  ext 64K+16K  128K ~100-116K
attention          1.3%    2.8%    17.3%       9.6%        48.5%          60.3%
gemm              86.9%   86.5%    73.5%      80.4%        45.7%          35.3%
gdn_chunk+conv1d   1.7%    4.9%     4.2%       4.6%         2.6%           2.0%
gdn_recurrent      4.5%      -        -          -            -              -
in-step bubbles    5.9%    2.7%     1.3%       1.6%         1.2%           1.0%
```

**Decode: the megakernel gate fails.** 5.9% bubbles + 94.1% × 86.9% × 19% ≈
**21%**, below the 25% gate, and the 19% term is itself a ceiling. DFlash2
halved the July bubble share (11.3% → 5.9%). The decode track closes; the
remaining lever is fewer bytes (weight quantization).

**Prefill GDN: fails the 10% gate at every length.** `gdn_chunk` + `conv1d`
peaks at **4.9%**, in the cold 4K case, and its share only falls as context
grows, because GDN costs a fixed amount per chunk while attention grows with
position. `PREFILL_KERNELS.md` Phase 4 is not funded. The SM80 GDN "gap" is
real but small.

**Prefill GEMMs are already near the ceiling.** 23.8 B projection params × 2
× 16,384 tokens in 3.44 s ≈ **227 TFLOP/s, ~73% of the 312 TFLOP/s bf16
peak**. cuBLAS rarely gets far past that on A100, so Phase 3 could win back
about 10% of GEMM time at most.

**Attention: the one op above the gate, and by a wide margin at long
context.** It is a single kernel, `flashinfer::BatchPrefillWithPagedKVCacheKernel`,
in the 16 full-attention layers:

- **Low efficiency.** At 0–16K it takes 6.39 ms per layer-chunk for ~8.2 ×
  10¹¹ causal FLOP (4 × 4,096 queries × ~8K mean keys × 256 × 24 heads), about
  **129 TFLOP/s, ~41% of peak**. The late window's 89 ms per layer-chunk fits
  keys around 114K at the same rate, so the kernel is uniformly slow rather
  than stalling at long context.
- **Register ceiling.** It uses **239 registers per thread** against the 255
  maximum. At head_dim 256 that points at low occupancy. Stage B has to confirm
  this; it is not established.
- **Consistency check.** Integrating attention linearly over a cold 128K gives
  ~26 s, GEMM ~27.5 s, and other ops ~3 s. That sums to ≈ 57 s against the
  measured 58.5 s.
- **What's at stake.** At ~45% of a cold 128K prefill, a kernel reaching ~60%
  of peak (1.46×) would save **~8 s of 58.5 s** and cut the prefill freeze by
  the same fraction. On the agent-shaped `ext16k` it would save ~1.1 s of 7.3 s.

Two capture faults, fixed in `p1_profile.py`, both worth knowing for any
`/start_profile` work:

1. **Without `start_step`, a capture closes only when the NEXT forward
   begins.** `profiler_manager.py` sets `target = forward_ct + num_steps + 1`
   and checks it before each batch. On a drained worker nothing arrives, so the
   first pass's `num_steps = chunks + 1` left each capture open into the next
   scenario and the cold32k trace was never written. The fix is
   `num_steps = chunks` plus an 8-token closing request.
2. **Timed windows need the current cold time.** The 67.8 s from
   `TRIAL_V0520_LEVERS.md` was stale: 58.5 s was measured here with the
   profiler armed. An 85% start therefore landed after the prefill ended. The
   script now starts at 75% of 58.5 s.

The first-pass `cold4k`, `cold128k_z` and `ext16k` traces are contaminated by
fault 1. Use only the `20260924b` traces for those three.

## Next: Stage B on the attention kernel

Order, cheapest first:

1. **Selection before replacement.** Benchmark prefill attention at head_dim
   256, 4,096 queries, and 8K / 64K / 114K keys in a standalone harness on
   drained GPU 1. Compare the FlashInfer paths available on SM80 (its prefill
   tile and warp config choices) against SGLang's Triton backend.
   - The Triton backend is the `_fwd_kernel` that the E3 audit found spilling
     (255 registers, 104 B stack), so it may lose. Measure it anyway.
   - If a backend or config already reaches ~60% of peak, the fix is a flag,
     not a kernel.
2. **`ncu` on the FlashInfer kernel** (side container, `--cap-add SYS_ADMIN`):
   achieved occupancy, spills, `ldmatrix`/`mma.sync` issue rate, shared-memory
   bank conflicts, and tensor-pipe utilisation. This confirms or refutes the
   register-pressure reading.
3. **Only then a kernel.** An SM80 head_dim-256 prefill kernel in CuTe (split
   head_dim across warps, or a two-pass QK/PV layout to cut live registers),
   with PTX where `ptxas` will not produce the schedule.
   - This is the top-down path's first real target on this node.
   - Acceptance follows `PREFILL_KERNELS.md` §6.
