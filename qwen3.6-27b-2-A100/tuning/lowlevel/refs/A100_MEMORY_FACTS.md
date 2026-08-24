# A100 memory system — whitepaper extracts, with corrections for our part

Source: *NVIDIA A100 Tensor Core GPU Architecture* whitepaper, text extracted
2026-08-24. Quotations are verbatim; the corrections beneath them are measured
on this node.

> **Read the correction on bandwidth before using any number here.** The
> whitepaper describes the **40 GB SXM4** part. We run **80 GB PCIe**.

---

## L2 cache

> "the A100 GPU has significantly more on-chip memory including a 40 MB Level 2
> (L2) cache — nearly 7x larger than V100 — to maximize compute performance.
> With a new partitioned crossbar structure, the A100 L2 cache provides 2.3x
> the L2 cache read bandwidth of V100."

40 MB, partitioned crossbar. For context, one forward pass streams ~57 GB of
weights through it — **1,425× the L2 capacity**, with no reuse inside the pass.

## L2 residency control — the mechanism E2 depends on

> "A100 allows L2 cache to be set-aside for persistent accesses in 1/16th
> increments (2.5 MB). Persistent accesses have prioritized use of this
> set-aside portion of L2 cache. **Normal or streaming accesses to global
> memory can only utilize this portion of L2 when it is unused by persistent
> accesses.** L2 persistence can be set up using CUDA Streams or CUDA Graphs.
> However, note that when the GPU is configured in Multi-Instance GPU (MIG)
> mode, the L2 cache set-aside functionality is disabled."

> "Residency of data in the L2 cache can be managed via an address-range-based
> window which designates an address range for which all read and write
> accesses will be cached persistently in L2. **The memory operations
> themselves require no annotation.** A100 also supports finer-grained
> per-memory-operation controls where L2 residency is specified on a per-access
> basis. The access-based controls include fractio[nal]…"

Three things follow, and they set E2's whole shape:

1. **"require no annotation"** is why the stream-level window is the only lever
   in this directory that reaches closed-source cuBLAS kernels.
2. **Set-aside is granular at 2.5 MB** — so the experiment is a sweep over
   1/16ths, not a boolean.
3. **The bolded sentence is the risk.** Reserving L2 for KV removes that
   capacity from the weight stream whenever the persistent data is not using
   it. Given the weight stream is 82% efficient today, this can lose. E2 must
   measure both sides, not just the KV side.

MIG is not enabled here, so the feature is available — verify with
`cudaDeviceProp::persistingL2CacheMaxSize` at run time rather than assuming.

## Asynchronous copy — why `num_stages` means something on SM80

> "The A100 GPU includes a new asynchronous copy instruction that loads data
> directly from global memory into SM shared memory, eliminating the need for
> intermediate register file (RF) usage. Async-copy reduces register file
> bandwidth, uses memory bandwidth more efficiently, and reduces power
> consumption."

This is `cp.async`, and it is what Triton's `num_stages` pipelines over. Our
GDN decode kernel emits **zero** `cp.async` instructions while its launch sets
`num_stages=3` — see README §E3. The prefill chunk kernels do emit it
(`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`: 58 occurrences).

## SM and GPU configuration

> "7 GPCs, 7 or 8 TPCs/GPC, 2 SMs/TPC, up to 16 SMs/GPC, **108 SMs**
> 64 FP32 CUDA Cores/SM, 6912 FP32 CUDA Cores per GPU
> 4 Third-generation Tensor Cores/SM, 432 Third-generation Tensor Cores per GPU
> 5 HBM2 stacks, 10 512-bit Memory Controllers"

108 SMs is the number every grid in the README's table is divided by.

## Compute Data Compression

> "To boost efficiency and enhance strong scaling, A100 adds Compute Data
> Compression. Compression saves up to 4x DRAM read/write bandwidth, up to 4x
> L2 read bandwidth, and up to 2x L2 capacity."

Noted but **not pursued**: this targets sparse/zero-heavy data, and BF16 model
weights are dense with high entropy. There is no exposed control for it in
CUDA and no reason to expect it engages on our traffic. Recorded so nobody
re-reads the whitepaper and re-raises it.

---

## Corrections for the 80 GB PCIe part we actually run

> "the NVIDIA A100 GPU has 40 GB of high-speed HBM2 memory with a class-leading
> 1555 GB/sec of memory bandwidth" … "With a 1215 MHz (DDR) data rate the A100
> HBM2 delivers 1555 GB/sec memory bandwidth"

| | whitepaper (40 GB SXM4) | this node (80 GB PCIe) |
|---|---|---|
| HBM | 40 GB HBM2 | **80 GB HBM2e** |
| bandwidth | 1555 GB/s | **1935 GB/s** |
| TDP | 400 W SXM | **300 W, `power.max_limit == power.limit`** |
| interconnect | NVLink 3 | **none — PCIe Gen4 x16, PIX** |
| L2 | 40 MB | 40 MB (unchanged) |
| SMs | 108 | 108 (unchanged) |

Every roofline figure in `tuning/docs/` divides by **1935**, not 1555. The
BF16 balance point is `312e12 / 1935e9 ≈ 161 FLOP/byte`.

The 300 W cap and the absent NVLink are the two places where our part is
materially worse than the whitepaper's, and both are already load-bearing
conclusions elsewhere: clocks decay 1365 → 1300 MHz under sustained load, and
TP=2 across the pair is ruled out.
