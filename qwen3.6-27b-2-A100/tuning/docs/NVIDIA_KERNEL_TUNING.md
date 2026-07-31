# NVIDIA-side kernel tuning — A100 (SM80) specifics and sources

Companion to `KERNEL_TUNING_INFO.md` (SGLang/model facts) and
`KERNEL_TUNING_SPEC.md` (procedure). This one covers the **hardware and NVIDIA
tooling** side: what the A100 can and cannot do, which tools answer which
question, and where to read further.

Written 2026-07-31, after Phase 1. **Read §0 first — the Phase 1 measurements
substantially lower the expected payoff of everything else in this document.**

Revised 2026-07-31b, after the context session. That session added §0a (new
measurements, including a memory-accounting correction that invalidates part of
`next-session/CONTEXT_262144.md`) and §1a (the host platform: CPU, NUMA, RAM,
GPU interconnect, hypervisor). It also **closed the open question** at the end
of §1 about `nvidia-smi -pl` headroom — there is none; see §1.

---

## 0. What Phase 1 measured, and why it changes the priority

Phase 1 raised concurrency 2 → 4 per replica and got **1.8–1.9×** throughput.
The telemetry captured during that change is more informative than the speedup:

| signal | at cap 2 | at cap 4 | reading |
|---|---|---|---|
| `DCGM_FI_PROF_DRAM_ACTIVE` | ~69.5% | ~69.2% | **unchanged** while tokens ~doubled |
| `DCGM_FI_PROF_SM_OCCUPANCY` | 8.2% | 9.1–9.4% | +13% only |
| `DCGM_FI_PROF_SM_ACTIVE` | 80–83% | 82–84% | unchanged |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | ~29% | ~29% | unchanged |

Three conclusions, each with a consequence:

**1. Decode is memory-bound, confirmed directly.** Memory-interface activity
stayed flat at ~69% while throughput rose 1.9×. The traffic was dominated by
re-reading 54 GB of weights once per forward pass; a batch of 4 reads them the
same number of times and serves four streams. *Consequence: arithmetic
optimisations in kernels cannot help. Only reducing bytes moved, or moving
fewer times, can.*

**2. Occupancy is ~9% and the GPU is still 80%+ "active".** That gap — SMs busy
but nearly empty — is the signature of a latency-bound workload with a starved
grid. *Consequence: the GDN Triton kernel's hardcoded `num_warps=1` is real,
but raising the batch moved whole-GPU occupancy by only 13 points of relative
change, which bounds how much of total time that kernel can represent.*

**3. Tensor pipe at 29% with a BF16 balance point of 161 FLOP/byte.** Decode
arithmetic intensity is ~2–4 FLOP/byte. *Consequence: the tensor cores are not
the bottleneck and never will be at these batch sizes.*

> **Honest expectation.** `KERNEL_TUNING_SPEC.md` §S1 sets a stop rule: if GDN
> linear-attention is under ~15% of decode time, skip the manual Triton work.
> The occupancy result above is weak evidence that it *is* under that
> threshold. Profile before committing effort. A 5–15% win is the realistic
> ceiling here, against Phase 2 (EAGLE depth) which is arithmetically capable
> of 30%+.

---

## 0a. What the context session (2026-07-31b) added

Four measurements, each with a consequence for kernel work.

### 1. Prefill runs at ~27% of BF16 tensor peak — and that is the *only* phase with headroom

Measured on a genuine 165,000-token cold prefill, from the two full 16,384-token
chunks (the short trailing chunks report inflated TFLOPS because wall-time
attribution breaks down on sub-chunk work — ignore them):

```
16,384-token chunk   1524.40 tok/s   86.14 TFLOPS   27.6% of 312
16,384-token chunk   1416.73 tok/s   80.05 TFLOPS   25.7% of 312
```

Cross-check: `2 x 27e9 params x 1524 tok/s = 82 TFLOPS`, agreeing with the
engine's own estimate. This is the **complement of the §0 decode finding**:
decode is bandwidth-bound with tensor pipe at 29% and cannot be helped by
arithmetic tuning, but prefill is compute-shaped and sits at roughly a quarter
of peak. *Consequence: if kernel effort goes anywhere, prefill and the GDN
layers are where arithmetic still buys something. Decode remains off-limits.*

Note this is a **hybrid** model: 48 of 64 layers are GDN linear attention, which
is not a dense GEMM and cannot reach tensor-core peak by construction. A 27%
whole-model MFU is therefore not 27% of an achievable target — the achievable
target is well below 100% and nobody has established what it is. Do not treat
the gap as recoverable headroom without a per-layer profile.

### 2. Prefill cost is linear in tokens at this scale

| prompt | cold prefill | per token |
|---|---|---|
| 154,997 (Phase 1) | 77.2 s | 0.498 ms |
| 165,000 (this session) | 81.0 s | 0.491 ms |

Two points 10K apart agreeing to 1.4% confirms the linearity that
`CONTEXT_262144.md` assumed but flagged as unverified. The extrapolation to
262,144 tokens ≈ **129 s** therefore stands.

### 3. `available_gpu_mem` in the boot log is not free memory — correction

The boot log reports `available_gpu_mem = 8.35 GB`. **True free VRAM at steady
state is ~4.4 GiB.** The figure is sampled before the CUDA graphs and the
FlashInfer workspaces are allocated.

Directly demonstrated: immediately after a fresh boot, r1 showed **7,955 MiB**
free while r0 — same config, but had served traffic — showed **4,469 MiB**. The
~3.5 GiB delta is workspace and allocator growth that appears on first real use
and stays.

The static budget itself closes exactly, which is why this went unnoticed:

```
0.92 x 79.27 GiB                                     = 72.93 GiB budget
weights 56.58 + mamba 5.46 + KV 11.00                = 73.04 GiB
CUDA graphs 1.55  (allocated OUTSIDE the static budget)
CUDA context + FlashInfer workspaces  ~1.2
free                                                 ~4.4
```

*Consequence for kernel work:* there is far less room for profiler workspaces,
`ncu` replay buffers and `nsys` buffers than the boot log suggests. Expect
profiling to need a replica with reduced `--mem-fraction-static`, not the
production one. *Consequence for capacity:* `next-session/CONTEXT_262144.md`
sized its options against 8.37 GB free and is wrong by roughly 2×; 262,144 at
BF16 was never reachable on this footprint.

Reassuringly, a 165,000-token request moved VRAM by only **+18 MiB** — the KV
pool is genuinely preallocated, so long requests do not grow the footprint.

### 4. FP8 KV storage *does* work on SM80 — verified, then declined

§1 correctly says SM80 has no FP8 tensor cores. It does **not** follow that fp8
KV is unavailable, and this session proved it works:

- `fp8_e4m3` and `fp8_e5m2` paged **decode** (tensor-core path) and paged
  **prefill** both ran at this model's exact geometry — 24 QO / 4 KV heads,
  head_dim 256, page 64 — returning finite output, with a BF16 control passing
  alongside. Kernels are AOT-compiled in the pinned image.
- SGLang has no SM80 guard: `mem_cache/kv_cache_dtype.py` maps the flag straight
  to `torch.float8_e4m3fn`. Its draft-model override is DFLASH/fa4-only, so the
  EAGLE draft KV would go fp8 too.

**It was declined on numerics, not capability.** On an unquantized BF16
checkpoint, `k_scale`/`v_scale` are `None`, so `memory_pool.py set_kv_buffer`
performs a **direct uncalibrated cast** — no per-tensor scaling. fp8_e4m3 gives
3 mantissa bits (~3% RMS error) and saturates above ±448. Qwen3.6's QK-norm
bounds K; **V is unnormalized and is the open saturation risk.** If this is ever
revisited, `fp8_e5m2` (max 57,344) is the range-safe fallback, and EAGLE accept
length is a free, sensitive fidelity proxy — it is already logged.

---

## 1. A100 80GB PCIe — the numbers that constrain kernels

Measured live via `torch.cuda.get_device_properties(0)` and DCGM:

```
compute capability        (8, 0)          sm_80, Ampere
multi_processor_count     108
regs_per_multiprocessor   65536
HBM2e bandwidth           1935 GB/s       (PCIe variant; SXM 80GB is 2039)
BF16 tensor dense         312 TFLOPS
BF16 balance point        312e12 / 1935e9 = 161 FLOP/byte
TDP                       300 W           <- measured at cap, see below
boost clock               1410 MHz        <- measured 1300-1380 under load
```

| property | value | why a kernel author cares |
|---|---|---|
| SMs | 108 | a grid under 108 blocks cannot fill the GPU |
| max warps / SM | 64 | 108 × 64 = 6912 warp slots machine-wide |
| max blocks / SM | 32 | 1 warp per block ⇒ occupancy caps at 50% |
| registers / SM | 65536 | 128 regs/thread ⇒ 512 threads/SM ⇒ 25% ceiling |
| max regs / thread | 255 | hard cliff; spills go to local memory (HBM) |
| shared mem / SM | 164 KB | 163 KB opt-in max per block; 48 KB static default |
| L2 | 40 MB | |

### What SM80 lacks — this rules out most "modern" kernel advice

- **no FP8 tensor cores** (SM89 Ada / SM90 Hopper). FP8 on A100 is a *storage*
  format only — conversion on read, no hardware fast path.
- **no `wgmma`** warpgroup MMA (SM90). All the Hopper-era GEMM literature
  — warp specialisation, producer/consumer warpgroups — does not apply.
- **no TMA** (Tensor Memory Accelerator, SM90). Bulk async copy with
  multicast is unavailable; SM80 has plain `cp.async`.
- **no `tcgen05` / tensor memory** (SM100 Blackwell).
- **`cp.async` IS present** — SM80 introduced it. This is why Triton
  `num_stages` pipelining is meaningful here rather than a no-op.

Practical consequence already confirmed in `KERNEL_TUNING_INFO.md` §4: SGLang's
`flashinfer`, `cutedsl` and `flashkda` linear-attention backends are all gated
behind SM90+/SM100+. **Triton is the only backend for 48 of 64 layers.**

### The power cap is a first-class constraint here

Measured throughout Phase 1: power pinned at **293–306 W** against a 300 W TDP,
with SM clock decaying **1365 → 1300 MHz** as GPU temp rose 53 → 76 °C and HBM
to 86 °C. Idle is 1410 MHz at ~68 W.

This matters more than it looks:

- Any kernel change that increases work per unit time gets **partially clawed
  back as clock reduction**. A "10% faster kernel" may deliver 6%.
- Benchmarks run later in a session are systematically penalised. Always
  interleave a control, and always record `DCGM_FI_DEV_SM_CLOCK`.
  `tuning/bench/compare.py` warns when clock varies >30 MHz across a ladder.
- This is a *passively cooled* PCIe part in a chassis with a second one beside
  it. Thermal coupling between GPU0 and GPU1 is real — Phase 1 waited for GPU
  temperature to settle between runs for this reason.

**RESOLVED 2026-07-31b — there is no power headroom.** Measured directly:

```
power.limit      300.00 W
power.max_limit  300.00 W     <- identical; the cap cannot be raised
clocks.max.sm    1410 MHz
clocks.max.mem   1512 MHz
```

`power.max_limit == power.limit`, so `nvidia-smi -pl` has nothing to give on
this SKU. **The 300 W cap is a hard physical constraint, not a configured
one.** Treat the clock decay in the table above as permanent and unavoidable:
every kernel improvement on this machine is partially converted into lower
clocks rather than more throughput, and there is no way to buy it back.

Still open, and still worth doing: locking clocks with `nvidia-smi -lgc` for
**measurement repeatability** during A/B work. It trades peak throughput for a
stable baseline and is standard practice for kernel comparisons — but note it
can only lock *below* what the power cap already allows, so it makes results
comparable, not faster. Both GPUs run `persistence_mode Enabled`, `ECC
Enabled`, `compute_mode Default`.

---

## 1a. The host platform — CPU, memory, interconnect, system

Measured 2026-07-31b via `lscpu`, `free`, `nvidia-smi topo -m` and
`nvidia-smi --query-gpu`. The GPU is only half the machine, and two facts below
constrain the *deployment architecture* more than any kernel does.

```
System        Linux 6.8.0-136-generic
Hypervisor    KVM (full virtualisation, AMD-V)   <- this is a guest, not bare metal
CPU           AMD EPYC 7663  (Zen 3 "Milan", family 25, stepping 1)
              56 cores / 56 threads, 2 sockets x 28 cores, 1 thread per core
              ~2.0 GHz base (BogoMIPS 3992.49)
ISA           AVX2, FMA, VAES, BMI2, SHA-NI  —  NO AVX-512
NUMA          2 nodes: node0 = CPU 0-27, node1 = CPU 28-55
Caches        L1d 3.5 MiB / L1i 3.5 MiB / L2 28 MiB / L3 896 MiB (as reported)
RAM           1007 GiB total, ~824 GiB available, 7 GiB swap
GPUs          2x A100 80GB PCIe, PCIe Gen4 x16 (max = current)
              bus 06:10.0 and 06:11.0
Interconnect  GPU0 <-> GPU1 = PIX (at most a single PCIe bridge). NO NVLINK.
```

> The reported **L3 of 896 MiB is a virtualisation artifact** — a physical EPYC
> 7663 has 256 MiB. `lscpu` under KVM is reporting per-instance figures the
> guest cannot verify. Do not use it for cache-blocking arithmetic. The same
> caution applies to the L1d figure (64 KiB/instance reported; Zen 3 is 32 KiB).

### The interconnect decides the deployment shape

`nvidia-smi topo -m` reports **PIX** between the two GPUs and there is no
NVLink. Peer-to-peer traffic crosses PCIe Gen4 x16: **~31.5 GB/s theoretical,
~25 GB/s practical, per direction** — against 1935 GB/s of HBM on each card, a
**~75× gap**.

*Consequence, and it is the most important sentence in this section:* **tensor
parallelism across these two GPUs is not viable.** TP=2 would inject a
per-layer all-reduce over a link 75× slower than the memory it is trying to
help feed, on a model that §0 already established is bandwidth-bound. The
current design — `--tp 1` with two independent replicas behind a cache-aware
router — is not a compromise forced by simplicity; it is the correct answer for
this interconnect. **Do not propose TP=2 as a way to fit longer context.** It
would trade a memory problem for a much worse communication problem, and the
KV pool per replica would not grow anyway.

This also means `--enable-nccl-nvls`, `--enable-symm-mem`, `--enable-mscclpp`
and the all-reduce fusion flags are all inert here: there is no multi-GPU
collective in a TP=1 replica, and no NVLink for them to exploit if there were.

### NUMA pinning is currently unverified — check before trusting it

`docker-compose.yml` pins `cpuset: "0-27"` for r0 and `"28-55"` for r1, one
NUMA node each, and calls it NUMA discipline. The intent is sound. But:

```
GPU0   CPU Affinity 0-55   NUMA Affinity 0-1
GPU1   CPU Affinity 0-55   NUMA Affinity 0-1
```

**The hypervisor is not exposing a distinct NUMA affinity per GPU.** Both cards
claim affinity to all 56 cores and both NUMA nodes. So the current pinning is
an assumption, not a measured locality optimisation — and if both GPUs
physically hang off node 0's root complex, pinning r1 to node 1 forces every
host-to-device copy across the inter-socket link.

This is cheap to test and nobody has: benchmark r1 at `cpuset: "0-27"` against
`"28-55"` with the same ladder and compare H2D-sensitive phases (tokenisation
feed, weight load, cold prefill). Until that is done, treat the split as
"reasonable default", not as tuning. Note that r0 and r1 also share the cores
with the tokenizer worker and the observability stack — cores 48-55 were
reserved for TEI + Prometheus/DCGM/Grafana, which overlaps r1's range.

### Host RAM is abundant, and PCIe makes offload arithmetically attractive

~824 GiB available. Phase 1 released 506 GiB by disabling HiCache, so this is
now idle. Worth re-examining precisely because of the numbers above:

- KV is 64.8 KiB/token, so the entire 169,792-token pool is 11 GB.
- At ~25 GB/s over PCIe Gen4 x16, moving the whole pool host↔device is **~0.44 s**.
- A cold 165,000-token prefill costs **81 s** (§0a).

A prefix that can be fetched from host RAM instead of recomputed is therefore
**~180× cheaper**, and host RAM could hold on the order of 12M tokens of KV.
That is a far better ratio than it looks, and long-context coding-agent traffic
with a shared system prompt is the ideal shape for it. Phase 1 removed HiCache
for good reasons (boot time, and 506 GiB pinned for little measured gain at the
then-current context), but those reasons were about *configuration*, not about
this arithmetic. **Re-evaluating host-RAM KV offload is a stronger lead than
most of §3** — and unlike kernel work it cannot hurt correctness.

### Profiling under KVM — check permissions before planning work

This is a guest. GPU performance counters can be restricted by the hypervisor
in addition to the usual `NVreg_RestrictProfilingToAdminUsers` gate in §2.
Before scheduling any `ncu`/`nsys` effort, run a throwaway profile and confirm
counters are non-zero. Discovering `ERR_NVGPUCTRPERM` after building a whole
experiment is the expensive order to find out.

The CPU has **no AVX-512**, which matters only for CPU-side paths — detokenisation,
the tokenizer worker (`--tokenizer-worker-num 1`), and sampling fallbacks. None
is on the critical path today, but do not port CPU kernel advice that assumes
AVX-512 from Intel-based inference notes.

---

## 2. Tooling — which tool answers which question

### Nsight Systems (`nsys`) — "where does the time go?"

Timeline-level. Use it to get the layer/kernel breakdown that
`KERNEL_TUNING_SPEC.md` §S1 gates on.

```bash
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
     -o /tmp/qwen_decode python3 -m sglang.launch_server ...
```

- `--cuda-graph-trace=node` is **essential here**. Without it, CUDA graphs
  appear as one opaque blob and you learn nothing about the kernels inside —
  and this deployment runs decode entirely inside captured graphs.
- Pair with SGLang's `--enable-layerwise-nvtx-marker` for per-layer ranges.
- SGLang also exposes `/start_profile` and `/stop_profile` HTTP endpoints,
  which is the lower-friction path since it needs no restart.

### Nsight Compute (`ncu`) — "why is this kernel slow?"

Per-kernel counters: achieved occupancy, memory throughput, warp stall reasons.
This is the tool that would confirm or refute the `num_warps=1` hypothesis in
`KERNEL_TUNING_INFO.md` §3a.

```bash
ncu --set full -k regex:fused_sigmoid_gating_delta_rule_update \
    --launch-count 20 -o gdn_decode python3 ...
```

Sections worth reading first: **Occupancy** (achieved vs theoretical),
**Memory Workload Analysis** (DRAM throughput vs peak), **Warp State
Statistics** (stall reasons — `long_scoreboard` means memory latency,
`barrier` means load imbalance across warps).

- `ncu` **serialises kernels and disables concurrency**. Never read wall-clock
  timings from an `ncu` run.
- Profiling counters need permission: `CAP_SYS_ADMIN` or the
  `NVreg_RestrictProfilingToAdminUsers=0` module parameter. Failure mode is
  `ERR_NVGPUCTRPERM`.

### DCGM — "what is the GPU doing in production?"

Already configured this session (`tuning/prometheus/dcgm-counters.csv`,
9 profiling fields). Coarse but continuous and zero-friction. It answered the
memory-bound question above without any profiler run.

### ⚠ Counter contention — the trap that silently produces zeros

DCGM's profiling group and CUPTI-based tools (`nsys`, `ncu`, the PyTorch
profiler's CUPTI path) **contend for the same hardware performance counters**.
They cannot both hold them.

Symptoms: `ERR_NVGPUCTRPERM`, empty counter columns, or a profile that runs to
completion and reports zeros.

```bash
docker compose -f docker-compose.yml -f docker-compose.metrics.yml stop dcgm-exporter
#   ... profile ...
docker compose -f docker-compose.yml -f docker-compose.metrics.yml start dcgm-exporter
```

This is the single most common way an A100 profiling session wastes an
afternoon. Also documented in `tuning/prometheus/README.md`.

---

## 3. Triton on SM80 — what actually moves the needle

The kernels in scope are Triton, not CUDA C++ (see `KERNEL_TUNING_INFO.md` §3).
So "kernel tuning" here means launch configuration, not PTX.

**`num_warps`** — threads per block ÷ 32. Triton distributes a block-level
tensor across all threads in the block, so this directly sets registers per
thread. The GDN decode kernel holds a `[BK, BV] = [128, 32]` fp32 state tile;
at `num_warps=1` that is **128 registers/thread for state alone**, against a
255 ceiling. Raising to 2 halves it.

**`num_stages`** — software pipelining depth over `cp.async`. Meaningful on
SM80. But the GDN decode inner loop is a *serial recurrence* with trip count
`T=1` (decode) or `T=4` (target verify, = `--speculative-num-draft-tokens`).
Depth-3 pipelining over 1–4 serial iterations costs shared memory and registers
and returns nothing. This is the cleanest single hypothesis in the whole plan.

**Tile sizes (`BLOCK_*`, `BV`)** — trade parallelism against redundant loads.
`BV=16` instead of 32 doubles the grid (more occupancy on a starved grid) at
the cost of 2× redundant q/k loads.

### Occupancy is not the goal

A common error. For a **memory-latency-bound** kernel, higher occupancy hides
latency and helps. For a **bandwidth-bound** kernel already saturating DRAM,
higher occupancy does nothing — and `DRAM_ACTIVE ~69%` says this workload is
closer to the second case than the first. Chase *stall reasons* in `ncu`, not
the occupancy number.

### Checking register pressure

```bash
# per-kernel register/spill counts from the Triton cache
cuobjdump --dump-resource-usage <cubin>
# or set before launch:
TRITON_PRINT_AUTOTUNING=1   MLIR_ENABLE_DUMP=1
```

Non-zero **spill stores/loads** is the red flag — spills go to local memory,
which is HBM, which is exactly the resource already at 69%.

---

## 4. Sources

**NVIDIA — architecture and ISA**
- *NVIDIA A100 Tensor Core GPU Architecture* (whitepaper) — SM80 SM layout,
  shared-memory/L1 configuration, `cp.async`, the 108-SM organisation.
  <https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf>
- *CUDA C++ Programming Guide*, "Compute Capability 8.x" — the per-SM limits
  used in §1 (2048 threads, 64 warps, 32 blocks, 65536 registers, 164 KB shared).
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x>
- *CUDA C++ Best Practices Guide*, "Occupancy" and "Memory Optimizations" —
  why occupancy is a means and not an end.
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- *PTX ISA*, `cp.async` — the SM80 async-copy primitive behind `num_stages`.
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>

**NVIDIA — tools**
- *Nsight Systems User Guide* — especially `--cuda-graph-trace=node`, without
  which captured graphs are opaque.
  <https://docs.nvidia.com/nsight-systems/UserGuide/>
- *Nsight Compute Kernel Profiling Guide* — the Occupancy, Memory Workload and
  Warp State sections named in §2.
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- *DCGM Field Identifiers* — the `DCGM_FI_PROF_*` field IDs used in
  `tuning/prometheus/dcgm-counters.csv` (SM_ACTIVE 1002, SM_OCCUPANCY 1003,
  DRAM_ACTIVE 1005).
  <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>
- *dcgm-exporter* — counter CSV format and `-f`.
  <https://github.com/NVIDIA/dcgm-exporter>

**Triton**
- *Triton documentation*, `triton.Config` / `triton.autotune` — semantics of
  `num_warps` and `num_stages`.
  <https://triton-lang.org/main/python-api/generated/triton.Config.html>
- *Triton matrix-multiplication tutorial* — the canonical worked example of
  block-size/stage tuning and how the pipeline maps onto `cp.async`.
  <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>

**Roofline / performance model**
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures*, CACM 2009 — the arithmetic-intensity vs
  balance-point reasoning used throughout.
  <https://dl.acm.org/doi/10.1145/1498765.1498785>

**SGLang / model**
- SGLang server arguments reference — every flag named in these documents.
  <https://docs.sglang.io/backend/server_arguments.html>
- SGLang speculative decoding (EAGLE) — including `bench_speculative.py`, the
  right tool for Phase 2.
  <https://docs.sglang.io/backend/speculative_decoding.html>
- In-image kernel sources — the authoritative reference for what actually runs,
  and the source of every constant in `KERNEL_TUNING_INFO.md` §3:
  `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/`
- Gated DeltaNet / linear attention background (`chunk_gated_delta_rule`, the
  `fla` op set SGLang vendors): <https://github.com/fla-org/flash-linear-attention>

---

## 5. If you do only one thing next session

**Profile.** `KERNEL_TUNING_SPEC.md` §S1 with `nsys --cuda-graph-trace=node`,
and get the GDN share of decode time. Everything in §3 above is conditional on
that number exceeding ~15%, and the Phase 1 occupancy result suggests it may
not.

Stop the DCGM exporter first (§2). Under KVM, confirm counters are non-zero
before building the experiment (§1a).

---

## 6. Tuning directions, ranked — revised 2026-07-31b

Ordered by expected value per unit of risk, using everything measured to date.
The top two are not kernel work, which is the honest reading of the evidence.

**1. Host-RAM KV offload (HiCache), re-evaluated. — best ratio available.**
§1a: a reusable prefix costs ~0.44 s over PCIe against ~81 s to recompute, and
~824 GiB of host RAM is idle. Attacks the dominant cost of long-context serving
(prefill) rather than decode, which §0 showed is bandwidth-locked. Cannot affect
numerical correctness. Phase 1 disabled it for configuration reasons, not
because this arithmetic failed.

**2. EAGLE depth (Phase 2).** Still the largest single arithmetic lever, per
`next-session/README.md`: accept length 3.483 against a cap of 4.00, 15.3% of
verifies pinned at the cap, and the current `steps=3 / topk=1 / draft=4` is
verbatim the SGLang docs' OOM-recovery recipe, never tuned. Use
`bench_speculative.py`. Watch the Mamba coupling — `mamba num` tracks
`num_draft_tokens`, and the pool is 54 slots.

**3. `--enable-fused-qk-norm-rope`.** Off today. Qwen3.6 applies QK-norm on all
64 layers, so this fuses two elementwise passes into one across the whole model
— it reduces *bytes moved*, which §0 identifies as the only thing that helps
decode. One flag, low risk, but **measure it alone**: it was deliberately kept
out of the 2026-07-31b context roll to keep that change clean.

**4. NUMA pinning validation (§1a).** Cheap, currently unverified, and could be
actively harmful today. An afternoon of ladders answers it.

**5. Prefill kernel work.** §0a puts prefill at ~26–28% of BF16 tensor peak,
the one phase with visible arithmetic headroom. But 48 of 64 layers are GDN and
cannot reach tensor peak by construction, so establish a realistic per-layer
target from a profile before assuming any of that gap is recoverable.

**6. GDN Triton launch-config work (§3).** Gated behind the §S1 stop rule. The
`num_stages` hypothesis remains the cleanest single experiment in the plan.

**7. Clock locking (`nvidia-smi -lgc`).** Not a speedup — a measurement
prerequisite. Worth doing *before* 5 or 6, because with a hard 300 W cap (§1)
and 1365→1300 MHz decay, an unlocked A/B cannot resolve the 5–15% effects those
items are chasing.

### Explicitly ruled out

- **TP=2 across the two GPUs** — no NVLink, PIX/PCIe Gen4 only, ~75× slower
  than HBM on a bandwidth-bound model (§1a).
- **Raising the power cap** — `power.max_limit == power.limit == 300 W` (§1).
- **FP8 KV for capacity** — works on SM80 (§0a §4) but declined on uncalibrated
  numerics; revisit only with a real quality gate.
- **262,144 context at BF16** — needs ~+6 GB of KV against ~4.4 GiB of real
  free VRAM (§0a §3). Not reachable on this footprint.
