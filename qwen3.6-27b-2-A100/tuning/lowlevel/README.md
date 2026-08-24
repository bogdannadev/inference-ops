# tuning/lowlevel — SM80 memory-system and PTX-level work

Started 2026-08-24. Scope is deliberately narrower than `tuning/docs/`: this
directory is only about **how bytes move on an A100 (SM80)** and what we can
change at the kernel / PTX / cache-policy level to move fewer of them, or move
them better.

Nothing here has been applied to the running deployment. Every experiment
states its gate before it is run.

```
lowlevel/
  README.md                 this file — purpose, method, results
  refs/
    PTX_SM80_VERIFIED.md    what ptxas actually accepts on sm_80 (probed, not quoted)
    A100_MEMORY_FACTS.md    whitepaper extracts, with the PCIe-80GB corrections
  bench/
    e1_gemm_workspace.py    Experiment 1 — cuBLASLt split-K workspace sweep
  results/                  JSON output, one file per run
```

---

## Purpose

The July profile settled that decode on this node is weight-streaming through
cuBLAS GEMMs: **87.7% of decode GPU time is GEMM, 2.49% is the GDN Triton
kernel**, and the kernel sweep was vetoed on that basis.

Re-parsing that same trace on 2026-08-24 for *launch geometry* found something
the share-of-time analysis missed. Achieved bandwidth is not uniform — it
tracks blocks per SM:

| GEMM | grid | blocks/SM | occupancy | share of GPU time | achieved BW |
|---|---|---|---|---|---|
| gate_up  (N=34,816) | 544 | 5.04 | 32.6% | 31.9% | **~82% of peak** |
| down_proj (N=5,120, split-K 5) | 200 | 1.85 | 14.6% | 20.8% | ~58% of peak |
| — | 384 | 3.56 | 21.3% | 16.1% | — |
| — | 80 | **0.74** | 4.6% | 9.7% | ~54% of peak |
| lm_head-shaped (N=248,320) | 1940 | 17.96 | 68.7% | 7.0% | — |

The wide-output projections are genuinely bandwidth-saturated and have nothing
left to give. The narrow-output projections are **latency**-bound: with
`M = batch 4 × 6 draft tokens = 24` rows, the GEMM tiles almost entirely along
the output dimension, so a 5,120-wide output yields 40 tiles where a
34,816-wide output yields 544. At 0.74 blocks/SM, 28 of the 108 SMs are idle
and there are not enough concurrent memory requests in flight to cover HBM's
~500 ns latency. That is Little's Law, not a memory-system limit.

**Kernels below ~4 blocks/SM are ≈53% of GEMM time.** Lifting them from ~58%
to ~82% efficiency would cut ~15% of GEMM time, ≈13% of decode — five times
the GDN kernel that was vetoed.

### Correction this work forced

`tuning/docs/` records decode at "~1086 GB/s effective, 56% of peak". That was
a derivation, not a measurement: it attributed only the target model's 54 GB to
an *assumed* 50 ms forward, ignoring the five draft passes and draft-extend in
the same iteration. Measured from the trace, the dominant GEMM moves 356.5 MB
in 225 µs = **1,582 GB/s, 82% of peak**. Decode is *more* bandwidth-saturated
than recorded, which strengthens the byte-reduction argument and weakens the
case for a hand-written kernel.

---

## What is and is not reachable here

| surface | share of decode | PTX-level access |
|---|---|---|
| cuBLASLt GEMMs | 87.7% | **none** — closed binary. Only *selection* or *replacement* |
| Triton GDN + prefill chunk kernels | 2.5% decode; prefill unprofiled | full source → PTX → SASS |
| FlashInfer attention | 2.5% | CUDA C++, AOT in image |

There is no middle path into the 87.7%. So the work splits into **selection**
(cheap, first), **cache policy** (applies to closed kernels too), and
**replacement** (expensive, last, and probably obsoleted by quantization).

---

## What we are optimising — the picture

### 1. Where the bytes go in one decode iteration

Config is `--speculative-num-steps 5 --speculative-num-draft-tokens 6`, so a
single iteration is 5 draft forwards, one draft-extend, and one target verify.
The target verify streams the whole model; the draft head is small but is read
seven times.

```
  ONE DECODE ITERATION                                    HBM traffic
  ------------------------------------------------------------------
  draft step  x5        draft_extend      target verify
  +--+ +--+ +--+ +--+ +--+   +--+     +======================+
  |d | |d | |d | |d | |d |   |de|     |    TARGET  MODEL     |
  +--+ +--+ +--+ +--+ +--+   +--+     |      51.05 GB        |
   \___________  ___________/  \_/    +======================+
               \/                            |
      6 passes over the draft head           |
      ~3.4 GB each = ~20 GB            ------+------
                                             v
  \____________________  ____________________/     ~71 GB moved
                       \/                          ~3.3 tokens out
              per iteration, for ~3.3 accepted tokens
                                                   = ~21 GB / token

  87.7% of that time is cuBLAS GEMM.  2.49% is the GDN Triton kernel.
  Arithmetic intensity ~2-4 FLOP/byte against a balance point of 161.
  => only FEWER BYTES helps.  This directory asks the narrower question:
     are the bytes we do move, moving efficiently?
```

### 2. Why some GEMMs starve — the thing E1 attacks

During decode `M = 4 running x 6 draft tokens = 24` rows. That is smaller than
one 64-row tile, so the GEMM has nothing to tile along M and parallelises
almost entirely along N, the output width. Output width therefore *is* the
grid.

```
  W [N x K], tiled BN=64 along N.  M=24 collapses to a single tile.

  mlp_gate_up   N = 34,816                    mlp_down   N = 5,120
  |==========================================|  |=====|
   544 tiles                                     80 tiles
              |                                      |
              v  / 108 SMs                           v  / 108 SMs
      5.04 blocks per SM                     0.74 blocks per SM


  108 SMs, gate_up (grid 544)     -- every SM loaded, memory pipeline full
  [####][####][####][####][####][####][####][####][####][####][####][####]
   ~5 blocks each                                        ~82% of HBM peak

  108 SMs, narrow output (grid 80) -- 28 SMs get NO work at all
  [#][#][#][#][#][#][#][#][#][#][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ][ ]
   \_________ 80 with 1 block _________/\______ 28 idle ______/
                                                          ~54% of HBM peak
```

This is **not** a memory-system limit. At 0.74 blocks/SM there are too few
concurrent loads in flight to cover HBM's ~500 ns latency — Little's Law. The
memory interface is idle waiting, not saturated.

```
  bandwidth = bytes_in_flight / latency

  gate_up   many blocks -> many outstanding loads -> pipeline stays full
            |>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>| 82%

  narrow    few blocks  -> pipeline drains between round trips
            |>>>>>>>>>>>>>>>>....gap....>>>>>>>>>| 54%
```

### 3. Split-K — the mechanism under test

The fix for a grid that is too small is to also parallelise along K, the
reduction dimension, and sum the partials afterwards. cuBLASLt already does
this — grid 200 is 40 output tiles x a 5-way split — but stops at 5.

```
  no split-K                       split-K = 5
  ----------                       -----------
  tile 0 : K = 0 .. 17408          tile 0 : K = 0     .. 3482  --+
  tile 1 : K = 0 .. 17408          tile 0 : K = 3482  .. 6963  --+--> partial
   ...                             tile 0 : K = 6963  .. 10445 --+     sums
  40 blocks, 0.37 blk/SM            ...                          |
                                   200 blocks, 1.85 blk/SM       v
                                                          splitKreduce_kernel
                                                          (already 1.29% of
                                                           GPU time -- deeper
                                                           splits cost more)

  E1 asks: is the split depth capped by cuBLASLt's WORKSPACE?
           CUBLASLT_WORKSPACE_SIZE is unset on both replicas.
```

### 4. What the benchmark does

```
  PREP                     MEASURE                        RESTORE
  ------------------------------------------------------------------------
  drain r1 from router     for ws in [unset, 1M, 8M, 32M, 128M, unset]:
    DELETE /workers/<id>       spawn child process with that env
         |                       for shape in [gate_up, mlp_down, qkv,      restart DCGM
         v                                     o_proj, gdn_in, gdn_out]:         |
  stop dcgm-exporter                for M in [1, 6, 24, 48]:                     v
    (CUPTI counter                      time  50x F.linear, CUDA events    re-register r1
     contention with                    profile 6x -> chrome trace           POST /workers
     the torch profiler)                parse "grid" out of it                    |
         |                                                                        v
         v                       ^                              ^          verify 2 healthy
     r0 serves alone             |                              |
     (1.5% duty cycle,           baseline first AND last        gate = GRID,
      load 0 right now)          -> drift is visible            not bandwidth
```

The last-listed rule is the one that decides the outcome. Clocks are unlocked
on this node and the noise floor is ~1.5%, so a bandwidth number that moves
while the grid stays put is thermal drift, not split-K.

---

## Experiments

Status: `PLANNED` → `RUN` → `KEPT` / `REJECTED`.

### E1 — cuBLASLt split-K workspace sweep · `RUN 2026-08-24 · REJECTED`

**Outcome: the workspace has no effect, and the hypothesis behind it was
wrong.** Full result and the correction it forced are in *Results* below.
The original statement of the experiment follows, unedited.

**Hypothesis.** cuBLASLt is already splitting K — `splitKreduce_kernel` fires
17,919 times in the trace, and a grid of 200 decomposes as 40 output tiles × a
5-way split. It sizes split-K workspace from the workspace it is given and
declines configurations that will not fit. `CUBLASLT_WORKSPACE_SIZE` is
**unset** on both replicas, so PyTorch's built-in default applies. Raising it
may let cuBLASLt pick deeper splits on exactly the narrow-output GEMMs that are
starved.

**Method.** Standalone process, no engine involved. Allocate the real weight
shapes, run `F.linear` at decode-shaped `M`, time with CUDA events, and report
achieved bandwidth. Sweep the workspace across separate child processes —
PyTorch reads the variable once when it creates the cuBLASLt handle, so it
cannot be swept within one process. Capture the selected kernel name and grid
with the torch profiler so the *mechanism* is observed, not inferred.

**Gate.** A workspace setting is interesting only if it changes the **grid**
on a narrow-output shape. A bandwidth change with an unchanged grid is noise or
clock drift, not split-K. Baseline is measured first and last in the same run
to detect drift, per the project's control discipline.

**Cost if it works.** One environment variable in `docker-compose.yml`, a few
MB of VRAM, reversible by unsetting it.

**Known counterweight.** `splitKreduce_kernel` already costs 1.29% of GPU time;
deeper splits make that larger, so the net is smaller than the ceiling above.

### E2 — L2 residency window over the KV pool · `PLANNED`

**Hypothesis.** We stream ~57 GB of weights through a 40 MB L2 every forward.
Those weights have zero reuse within a pass, yet they evict the KV and
activations that do have reuse. At the p95 context of 141K tokens, KV traffic
is ~9.2 GB per target forward — 16% of the weight bytes, and it grows with
context while the weights stay fixed.

**Why this one can touch the 87.7%.** From the whitepaper: *"Residency of data
in the L2 cache can be managed via an address-range-based window… The memory
operations themselves require no annotation."* A stream-level window applies to
**every** kernel on that stream, cuBLAS included. That is the only lever in
this directory that reaches closed-source kernels.

**Verified constraint, not assumed.** The per-instruction route is *not*
available on this toolchain — see `refs/PTX_SM80_VERIFIED.md`. `ptxas` for
`sm_80` rejects `ld.global.L2::evict_first` at every width, and rejects
eviction modifiers on `cp.async` entirely. `createpolicy` + `.L2::cache_hint`
**is** accepted, which is the same mechanism the runtime window uses.

**Counterweight, from the whitepaper.** Set-aside comes in 1/16 increments
(2.5 MB) and *"normal or streaming accesses to global memory can only utilize
this portion of L2 when it is unused by persistent accesses."* Reserving L2 for
KV therefore takes capacity away from the weight stream. This can lose.

**Status.** Blocked on E1 and on a prefill profile. Do not run it first.

### E3 — Triton launch-config on the GDN decode kernel · `PLANNED, low value`

Kept for completeness and as the teaching case. From the compiled cubin already
in the live containers' Triton cache:

```
fused_sigmoid_gating_delta_rule_update_kernel
  REG:220  STACK:0  LOCAL:0        220 registers/thread, zero spills
  cp.async occurrences: 0          but the launch sets num_stages=3
  L2:: hints: 0
  138 × shfl.sync.bfly.b32         warp-level reductions
  global loads: scalar 32-bit, no .v2/.v4
```

Two things follow that are not visible from the source:

1. **`num_stages=3` is inert.** Triton's `num_stages` is pipelining depth over
   `cp.async`; this kernel emits none, so there is no pipeline to fill.
   `KERNEL_TUNING_SPEC.md` §S2.2 called this "the cleanest single hypothesis in
   the plan" — it is now confirmed from the artifact rather than argued.
2. **The kernel is grid-bound, not register-bound.** 220 regs × 32 threads
   (`num_warps=1`) = 7,040 regs/block; an SM has 65,536, allowing 9.3
   blocks/SM. The trace measured 7.11. Raising `num_warps` to 2 spreads the
   same `[128,32]` state tile over 64 threads, so registers *per block* are
   unchanged and the ceiling does not move — but each block now carries two
   warps, roughly doubling achieved occupancy (10.5% → ~22%).

Worth ~2.5% of decode at absolute best. It is a demonstration, not a lever.

### Audit finding, carried forward

Sweeping all 63 cached Triton kernels for spills, exactly one spills:

```
_fwd_kernel   REG:255  STACK:104   num_warps=8, num_stages=1, shared=40960
              sglang/kernels/ops/attention/prefill_attention.py
```

255 registers is the hard per-thread ceiling and 104 bytes of stack is spilling
to local memory, which is HBM. It does **not** appear in the decode trace.
Whether it runs during prefill is unknown, because prefill has never been
profiled. Note this partly rehabilitates the original `head_dim=256` hypothesis
in `TUNING_PLAN.md` §4a, which was marked wrong: right about the kernel, wrong
about which backend serves it.

---

## Results

| exp | date | outcome | file |
|---|---|---|---|
| E1 | 2026-08-24 | **REJECTED** — workspace inert; grid-starvation model refuted | `results/e1_gemm_workspace.json` |
| E2 | 2026-08-24 | **REJECTED for production; E1 validated** — mechanism works, capacity does not | `results/e2_l2_policy.json` |
| E3 | — | not yet run | — |

### E1 — 2026-08-24. Workspace does nothing. The premise was wrong.

Run on GPU 1 with r1 drained from the router (`DELETE /workers/<id>`, r0 served
alone at load 0 throughout) and the DCGM exporter stopped for CUPTI counter
contention. Both restored afterwards; router verified back at 2/2 healthy.
GPU 1 held 1410 MHz and 47–50 °C for the whole sweep, so nothing here is
clock-confounded.

**Primary result — the gate fired, and it fired negative.**

Across all six shapes, four values of `M`, and workspaces of
unset / 1 / 8 / 32 / 128 MiB, **the selected kernel and its grid are byte-for-byte
identical in every cell.** Bandwidth wobbled 2–6% between settings, but the
baseline-first vs baseline-last control drifted by up to 6.4% on its own, so
that wobble is noise. `CUBLASLT_WORKSPACE_SIZE` does not gate split-K depth
here. **Do not set it in `docker-compose.yml`.**

**Secondary result, and the one that matters — grid is not the mechanism.**

The experiment was designed to test workspace, but capturing the grid at four
values of `M` incidentally tested the grid-starvation model itself, and refuted
it. At `M=1` cuBLASLt picks a 3-way split-K for `mlp_gate_up`, giving grid 1632
— **15.11 blocks/SM and 60.4 warps/SM, i.e. 94% occupancy.** It buys nothing:

```
  mlp_gate_up, same 356 MB weight matrix, varying M
  M    grid   blk/SM   warps/SM   occupancy   achieved
   1   1632    15.11      60.44        94%      72.3%   <-- 20x the blocks
   6    544     5.04      20.15        31%      72.5%   <-- and no faster
  24    136     1.26       5.04         8%      67.1%
  48    136     1.26      10.07        16%      54.9%
```

Occupancy moves 8% → 94% and achieved bandwidth does not move. **The
narrow-output GEMMs are not slow because their grids are small.**

**What actually predicts efficiency is the size of the weight matrix.**

Measured at `M=24`, the production shape:

```
  weight    shape              %peak   at 1935 GB/s
  356 MB    mlp_gate_up        67.1%   |###########################
  178 MB    mlp_down           59.1%   |########################
  168 MB    gdn_in_proj_qkvz   55.3%   |######################
   84 MB    attn_qkv           43.9%   |#################
   63 MB    attn_o_proj        37.5%   |###############
   63 MB    gdn_out_proj       37.1%   |###############
            (grid 80 and grid 100 appear in BOTH the 59% and the 37% rows --
             identical occupancy, 22 points apart. size is the variable.)
```

Monotonic in bytes, and it holds across kernels with the same launch geometry.
The reading: a 63 MB weight is only ~32 µs of work at peak bandwidth, and at
that duration wave ramp-up, drain and launch overhead are a large fraction of
the kernel. Per-layer projections in a 27B model are simply too small to reach
streaming peak on a 108-SM GPU. Split-K cannot fix that — it adds blocks, and
blocks were never the constraint.

**Consequence: the "82% of peak" figure from earlier on 2026-08-24 is wrong,
and `tuning/docs/`'s original ~56% was very nearly right.**

That 82% came from attributing the July trace's grid-544 kernel to `mlp_gate_up`
at the production `M`. This run shows `mlp_gate_up` at `M=24` selects grid
**136**, not 544 — grid 544 is what it picks at `M=6`. The attribution was
wrong, so the number derived from it was too.

Reconstructing a target forward from directly measured per-shape efficiencies:

```
  64 layers x (gate_up 356 MB @ 67.1%  +  mlp_down 178 MB @ 59.1%)   27.6 ms
  16 layers x (qkv      84 MB @ 43.9%  +  o_proj    63 MB @ 37.5%)    3.0 ms
  48 layers x (gdn_in  168 MB @ 55.3%  +  gdn_out   63 MB @ 37.1%)   11.7 ms
                                                                   ---------
  47.65 GB of weights                                                42.3 ms
                                        => 1127 GB/s = 58.2% of peak
```

The 47.65 GB independently reconstructs the engine's reported 51.05 GB once
lm_head and embeddings are added, which is a check on the shape list. And
58.2% lands essentially on the recorded ~56% / 1086 GB/s. **The original note
was sound; the correction to it was not.** `NVIDIA_KERNEL_TUNING.md` §0 was
never edited and needs no change — its figure stands. The 82% existed only in a
session artifact and is retracted there.

**The headroom that does exist, sized honestly.** If every projection ran at
the best rate observed on this GPU (72%, `mlp_gate_up` at `M`≤6), a target
forward's GEMM time would fall 42.3 ms → 34.2 ms, **about 19% of GEMM time and
~17% of decode.** That is real and it is larger than anything the July campaign
found — but E1 establishes that neither workspace nor split-K nor occupancy
reaches it. The lever is *fewer, larger* GEMMs: fused projections, or weight
quantization, which shrinks every matrix and changes the kernel family at the
same time.

**What E1 cost:** ~7 minutes of GPU 1 with r1 drained. No config changed.

### E2 — 2026-08-24. E1 validated. L2 policy works and cannot help us.

Same procedure: r1 drained, DCGM stopped, GPU 1 held 1410 MHz at 45–47 °C.
Both restored, router verified 2/2 healthy.

**The plan changed before it ran, and the arithmetic is why.** The original E2
was "persist the KV pool in L2". KV at the p95 context of 141K tokens is
~9.2 GB; the L2 is 40 MB and the maximum persisting set-aside on this card is
**25 MiB**. That is 0.27% of the KV cache. There is no configuration of this
feature that matters for KV, and nothing else in the decode path both has reuse
and fits. So E2 was re-aimed at two questions worth more than the lever was.

#### ARM A — was E1 confounded by L2 reuse? No.

E1 timed 50 calls against the *same* weight tensor, so up to 40 MB of a 60 MiB
projection could have stayed resident across iterations — an advantage the real
engine never gets, since consecutive layers use different weights. Re-measured
against a rotation of distinct weights totalling ≥ 8× L2:

```
  shape               MiB  L2 frac    same  rotated   delta
  mlp_gate_up         340    0.118   68.8%    68.5%   -0.4%
  mlp_down            170    0.235   62.4%    61.6%   -1.4%
  gdn_in_proj_qkvz    160    0.250   57.6%    59.6%   +3.5%
  attn_qkv             80    0.500   45.4%    47.9%   +5.6%
  attn_o_proj          60    0.667   39.5%    41.6%   +5.2%
  gdn_out_proj         60    0.667   39.5%    41.6%   +5.2%
```

The shapes with the *most* to gain from L2 residency (`L2 frac` 0.667) are
**faster** when rotated, not slower. If E1 had been reading an L2 artifact the
sign would be the other way and largest exactly here. **E1's size-efficiency
curve is real.** The 60 MiB projections genuinely run at ~40% of peak because
they are ~32 µs of work, not because of anything cache-related.

The small positive delta on the three narrow shapes is consistent in sign but
sits inside the run-to-run variation already seen between E1 and E2 for the
same shapes (o_proj read 37.5% in E1, 39.5–41.6% here). Not claimed as a
result; plausibly allocation spread across memory channels.

#### ARM B — does the residency mechanism work at all? Yes, weakly.

Positive control, deliberately synthetic: a 24 MiB hot buffer read immediately
after a 512 MiB streaming read that would otherwise evict it.

```
  window readback: num_bytes=25165824  hit_ratio=1.0  hit_prop=Persisting  miss_prop=Streaming
  hot read after the evicting stream:  26.6 us  ->  22.5 us   (1.18x)
```

So the feature is real and correctly driven here — 1.18× on a best case where
the entire hot buffer fits inside the 25 MiB set-aside and is re-read every
iteration. That is the ceiling of the mechanism on this card, and production
has nothing shaped like it.

#### ARM C — does marking the weight stream non-polluting help? No.

```
  mlp_gate_up       normal=68.0%  streaming=68.3%   +0.4%
  mlp_down          normal=58.0%  streaming=62.0%   +6.8%   <- see below
  attn_qkv          normal=47.4%  streaming=47.9%   +1.1%
  attn_o_proj       normal=41.6%  streaming=41.6%   +0.0%
  gdn_in_proj_qkvz  normal=58.8%  streaming=58.8%   +0.0%
  gdn_out_proj      normal=41.6%  streaming=41.6%   +0.0%
```

Five of six are null. The `mlp_down` outlier is almost certainly a low
`normal` sample rather than a real win: ARM A measured the same shape at 62.4%
in its `same` arm, so 62.0% is the ordinary value and 58.0% is the anomaly.

**ARM C also cannot answer the production question, by construction.** In an
isolated benchmark there is nothing else in L2 for the weights to evict, so
marking them non-polluting has nothing to protect. Answering it properly needs
the engine — and the capacity arithmetic above says the answer cannot matter.

#### Verdict

**Do not pursue L2 residency on this deployment.** Not because the mechanism
fails — it works — but because 25 MiB of set-aside against a 9.2 GB KV cache
and a 47.65 GB weight stream has nothing to hold. Record this so it is not
re-raised from the whitepaper's LSTM example, which describes a model whose
recurrent weights fit.

**The transferable result is ARM A: E1 stands.**

#### Two toolchain landmines, both of which fake a null result

Recorded in `refs/PTX_SM80_VERIFIED.md`. In CUDA 13,
`cudaDevAttrL2CacheSize` is **38**, not the widely-quoted 78; and
`cudaLimitPersistingL2CacheSize` is **0x06**, not 0x05 — 0x05 is
`MaxL2FetchGranularity`, which *silently accepts a 0* and only errors when
given a real byte count. A script that set the wrong limit and never read it
back would have run with persistence disabled and reported a clean, wrong
negative. Every limit and stream attribute in `e2_l2_policy.py` is read back
after being set, and the device-attribute enum is sanity-checked against the
known 40 MiB L2 before anything else runs.

**What E2 cost:** ~6 minutes of GPU 1 across three attempts, r1 drained. No
config changed.

---

## Rules for this directory

Inherited from `tuning/README.md` and `next-session/README.md`, plus two new:

- **Nothing runs on a serving GPU without an explicit decision.** The
  established procedure is to drain the replica from the router by REST first
  (`DELETE /workers`), run, then re-register — the same sequence used for the
  July S1 profile.
- **Stop the DCGM exporter before any CUPTI-based profiling.** DCGM's profiling
  group and the torch profiler contend for the same hardware counters and the
  failure mode is silent zeros.
- **Grid before bandwidth.** For E1 specifically, a bandwidth delta with an
  unchanged grid is not evidence. Clocks are unlocked (needs root) and the
  measured noise floor on this node is ~1.5%.
- **Verify PTX against `ptxas`, not against documentation.** The probe in
  `refs/PTX_SM80_VERIFIED.md` contradicts a plain reading of the ISA doc for
  this toolchain. Compile it before believing it.
