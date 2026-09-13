# BF16 prefill custom-kernel checklist — Qwen3.8-27B on A100 (SM80)

**Source:** supplied by Danila, 2026-09-13, as the reference list for the
prefill kernel campaign. It is kept here as provided (formatting only). **Read
`PREFILL_KERNELS.md` first**: it checks this list against this node's
measurements and the SGLang v0.5.19 source. Several items are already
upstream, and a few need correcting for this model.

---

## 1. Model-specific optimization target

Qwen3.8-27B is a 27B dense BF16 model with:

- 64 language-model layers; hidden size 5,120;
- 16 repeated groups, each three Gated DeltaNet + FFN layers followed by one Gated Attention + FFN layer;
- Gated DeltaNet with 48 value heads, 16 query/key heads, head dimension 128;
- full attention with 24 query heads, four KV heads, head dimension 256;
- RoPE dimension 64; FFN intermediate dimension 17,408;
- RMSNorm, QK-Norm, SwiGLU-style gated MLPs, GQA, linear attention, and MTP components.

Priority, approximately:

1. dense FFN and projection GEMMs across all 64 layers;
2. Gated DeltaNet prefill scans across 48 layers;
3. full GQA attention across 16 layers;
4. LM head, particularly because the vocabulary is padded to 248,320;
5. normalization, RoPE, gating, cache writes, layout conversions, residual operations;
6. vision encoder and projector only if image/video inputs are important.

> For Qwen3.8-27B, an excellent FlashAttention kernel optimizes only one
> quarter of the language-model layers. The Gated DeltaNet and FFN paths must
> receive equal or greater attention.

## 2. A100 SM80 execution primitives

Target `-gencode arch=compute_80,code=sm_80`. Relevant SM80 mechanisms:

- BF16 and FP16 warp-level Tensor Core MMA; FP32 accumulation;
- thread-issued asynchronous global-to-shared copies through `cp.async`;
- `ldmatrix` shared-to-register matrix loads;
- warp shuffle reductions; configurable shared memory;
- L2 residency controls and access-policy windows;
- 2:4 structured sparse MMA; native INT8 Tensor Core MMA (future W8A8 only).

Central BF16 GEMM instruction: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`

```
HBM
 ↓ 128-bit coalesced cp.async
shared-memory ring buffer
 ↓ ldmatrix
register fragments
 ↓ mma.sync
FP32 accumulator fragments
 ↓ fused epilogue
BF16 output
```

Do not copy Hopper kernel architecture. A100 has no TMA, WGMMA, thread-block
clusters, or native FP8 Transformer Engine behaviour. SM80 optimization is
`cp.async` plus warp-scoped `mma.sync`.

## 3. Dense BF16 GEMMs first

### 3.1 Multistage Tensor Core main loop

```
Stage 0: Tensor Cores consume tile K
Stage 1: cp.async loads tile K+1
Stage 2: optional extra tile to absorb memory latency
```

Starting candidates: CTA tile 128×128×32, 128×64×32, 64×128×32; warp tile
64×64×32 or 64×32×32; MMA atom 16×8×16; stages 2 or 3; warps 4 or 8;
accumulator FP32; input/output BF16. Larger tiles improve reuse but raise
register and shared-memory usage (CUTLASS CTA/warp/MMA tiling hierarchy).

### 3.2 Tune the actual Qwen dimensions

`M` = prefill tokens in the batch, `K` = 5120, `N` = projection dimension.

```
QKV and DeltaNet projections: [M, 5120] × [5120, N]
FFN gate/up:                  [M, 5120] × [5120, 2 × 17408]
FFN down:                     [M, 17408] × [17408, 5120]
```

Separate kernels or dispatch per token bucket: `M` 1–64, 65–256, 257–1024,
1025–4096, 4097+. A tile chosen for M=8192 can perform poorly for M=64.

### 3.3 Grouped and persistent GEMM

At small prefill batches: grouped GEMM across requests or compatible
projections; persistent CTA scheduling; Stream-K or Split-K; grouped
QKV/DeltaNet projections; grouped small GEMMs in vision/MTP components.
Split-K raises parallelism when M or N is small but needs a partial-accumulator
reduction.

### 3.4 Fuse GEMM epilogues

```
QKV GEMM   + bias (if present) + QK-Norm + partial RoPE + head-layout conversion + direct K/V cache write
Gate/up    + split gate and up + SiLU + elementwise multiply + one BF16 output write
Out/down   + residual addition + optional next-layer RMSNorm statistics
```

## 4. SM80 Gated DeltaNet prefill kernel

### 4.1 Treat DeltaNet as an associative scan

State update of the form `S_t = A_t S_{t-1} + B_t`, i.e. an affine map
`T_t(S) = A_t S + B_t`. Composition is associative:
`(A2,B2)∘(A1,B1) = (A2 A1, A2 B1 + B2)`, enabling a parallel prefix scan: a
warp-local scan for short fragments, a block-wide hierarchical scan for chunked
sequences, a two-pass scan for long sequences, and chunk summaries for cross-CTA
composition. (Exact equations must come from the checked-out implementation.)

### 4.2 Blockwise recurrent scan

Partition into chunks of C tokens.
- Kernel A: local outputs assuming an identity initial state, plus one summary
  transform per chunk.
- Kernel B: scan the chunk summaries.
- Kernel C: correct each chunk using its true incoming state.

Combine A and C when the update permits.

### 4.3 Fuse projections and scan preparation

Avoid: projection → global Q/K/V/gate → gate activation → normalization → scan
prep → scan. Prefer: projection epilogue → gate transforms → QK normalization →
scan-ready packed layout → blockwise scan. Fuse into the first scan stage only
if the register footprint stays manageable.

### 4.4 State precision

Keep recurrent state accumulation in FP32 unless proven safe:
`BF16 input → FP32 state update → BF16 output`. Validate hidden-state and
final-logit error at 8K, 32K and 128K before reducing state precision.

### 4.5 DeltaNet cache layout

Consume the state cache without transposes or gathers; contiguous per-warp
shards, e.g. `[layer, request, head_group, state_row, state_col]`; inner
storage aligned to ≥16 bytes. Separate layouts for prefill scan, single-token
decode, and prefix-cache snapshots.

## 5. The 16 full GQA attention layers

### 5.1 FlashAttention-style online softmax

Never materialize scores. Tile over Q and K/V keeping row maximum, row
normalization sum and an FP32 numerator (FlashAttention-2 has an SM80 path):

```c
load_q_tile();
for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
    async_load_kv(next_stage);
    wait_for_current_stage();
    scores = mma_qk(q_fragment, k_fragment);
    update_online_softmax(scores);
    output = mma_pv(probability_fragment, v_fragment, output);
}
normalize_and_store(output);
```

### 5.2 Head dimension 256

Do not scale a head-128 kernel. Test splitting head dim between warp groups,
smaller BLOCK_M, BLOCK_N 32/64, partial output accumulation, two-pass/two-CTA
for long sequences, fewer stages, controlled FP32 accumulator lifetimes.
Candidates: BLOCK_M 32/64, BLOCK_N 32/64, warps 4/8, stages 2/3. Register
spilling invalidates the gains; compile with register diagnostics.

### 5.3 Reuse K/V across six query heads

Load each K/V tile once for 2, 3 or 6 Q heads per CTA. 2–3 are better starting
points (6 raises Q/output register pressure).

### 5.4 Split RoPE dimensions

Rotate dims 0–63 only; copy 64–255. Fuse QK-Norm + RoPE into the Q/K projection
epilogue or the attention load path. Precompute sin/cos.

### 5.5 Causal tile classification

Fully valid, diagonal/partially masked, fully masked. Only diagonal tiles need
per-element predicates.

### 5.6 Long-sequence Split-KV

Split K/V ranges across CTAs, each emitting partial maximum, partial
normalization sum and partial numerator, combined by a stable online-softmax
reduction. Enable only when attention underfills the GPU.

## 6. Shared-memory and `cp.async` design

16-byte aligned global-to-shared copies: `cp.async.cg.shared.global`,
`cp.async.ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`.
Test `.cg` against `.ca` on real shapes. Double or triple buffering (consumed /
loading / lookahead); more stages are not automatically better. `ldmatrix` for
shared-to-register fragments:

```
ldmatrix.sync.aligned.x4.m8n8.shared.b16
ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16
```

Shared-memory operands need a padded or XOR-swizzled layout to avoid bank
conflicts.

## 7. Fuse bandwidth-bound operations

- **Residual + RMSNorm:** `input + residual → FP32 sum of squares → rsqrt → BF16
  output`, vectorized 16-byte transactions, warp shuffle reduction,
  hierarchical reduction over 5,120.
- **QK-Norm + partial RoPE:** projection output → QK-Norm → RoPE(64) →
  attention-ready store.
- **SwiGLU:** gate/up accumulators → SiLU(gate) × up → BF16 output, fused into
  the gate/up epilogue.
- **Residual + next-layer prep:** projection output + residual + optional next
  RMSNorm statistics.

Do not fuse to the point of register spills or losing an efficient library main loop.

## 8. The LM head

Vocabulary padded to 248,320:
- Compute logits only for the final prefill token unless prompt logprobs are
  requested; chunk those.
- Fuse reduction/top-k when full logits are not needed.
- Vocab-parallel under TP; never write `[tokens, 248320]` if only a few values
  are consumed.
- Persistent/segmented GEMV/GEMM for few query rows.

## 9. Packed variable-length execution

`hidden_states [total_tokens, 5120]` plus `cu_seqlens [batch+1]`; no padding.
Kernels take offsets, lengths, cache position, page/block table and
request-to-cache mapping. Bucket by length when it helps tile utilization, but
stay packed within a bucket.

## 10. Direct cache writes

Write K/V and DeltaNet state straight into the final layout. No contiguous temp
→ transpose → pack → scatter. Paged:

```
token_index = request_offset + local_token;
page   = block_table[token_index / page_size];
offset = token_index % page_size;
```

Separate fast contiguous path (fresh prefill) and paged path (prefix
extension).

## 11. Triton strategy

Good for RMSNorm/residual, QK-Norm+RoPE, SwiGLU, cache packing, DeltaNet scans,
attention prototypes, reductions. Autotune BLOCK_M/N/K, `num_warps` (4, 8),
`num_stages` (2, 3), BLOCK_K a multiple of 16, vector width 8 BF16 = 16 bytes.
Autotune keys: token bucket, head dim, causal, GQA factor, cache mode,
contiguous/paged. Verify generated code: `tl.dot` → `mma.sync`, staged loads →
`cp.async`, efficient shared/`ldmatrix` movement; inspect final SASS.

## 12. CUDA C++ and CUTLASS strategy

- **CUTLASS/CuTe:** projection and FFN GEMMs, grouped/persistent GEMMs, complex
  epilogues.
- **Triton:** normalization, gating, RoPE, scans, format conversion, attention
  experiments.
- **CUDA/PTX:** final attention hot path, final DeltaNet hot path, unusual
  synchronization or cache control.

## 13. L2 and memory-access techniques

- 128-byte alignment of weights and activation rows; 128-bit vector loads and
  stores.
- Contiguous access across adjacent lanes; L2 access-policy windows for hot
  metadata or compact state.
- Avoid L2 pollution from streaming operands.
- Pinned host memory and async H2D for input prep; overlap metadata prep with
  GPU work.
- Separate CUDA streams only for independent work; avoid small CPU-driven
  launches between layers.
- Graph or persist stable kernel sequences.

Weights ≈ 55.6 GB; cache, activations, graph workspace, vision and
fragmentation leave limited headroom on 80 GB.

## 14. Multi-GPU BF16

TP: column-parallel QKV and gate/up, row-parallel output/down; overlap
reduce-scatter with epilogue/store; NCCL on a dedicated stream; sequence
parallelism; avoid DeltaNet-state redistribution. NVLink SXM strongly
preferred over PCIe. Context parallelism for extreme prompts (KV ranges,
DeltaNet chunks) with careful summary composition.

## 15. Lower priority for BF16

FP8 kernels (no Hopper FP8 on A100); 2:4 sparsity without a validated
checkpoint; INT8 W8A8 (changes the numerical contract); speculative decoding
(decode, not unique-prefix prefill); CUDA graphs alone (launch overhead, not
FLOPs); fusion that lowers occupancy or spills registers.

## 16. Recommended implementation order

1. **Profile unmodified server:** DeltaNet projection, scan, state/cache update;
   full GQA attention; QKV projection; FFN gate/up; SwiGLU; FFN down; RMSNorm;
   LM head; cache formatting; vision encoder. Nsight Systems for timeline and
   launches, Nsight Compute per kernel.
2. **Low-risk fusion:** residual+RMSNorm, QK-Norm+partial RoPE, SwiGLU, direct
   K/V and state writes, packed varlen, last-token-only LM head.
3. **GEMM specialization:** exact 5,120 / 17,408 dims, multistage `cp.async` +
   `ldmatrix` + `mma.sync`, fused epilogues.
4. **DeltaNet prefill:** blockwise associative scan, fused scan prep, FP32
   state.
5. **Full attention:** FA2-style GQA for Q 24 / KV 4 / head 256 / RoPE 64.
6. **Long context:** Split-KV attention, multi-CTA DeltaNet scan, context
   parallelism, persistent scheduling, cache-aware chunk boundaries.

## 17. Metrics and acceptance

**Per kernel:**
- prefill tok/s, TTFT, kernel duration;
- Tensor Core utilization, achieved BF16 TFLOP/s, DRAM throughput;
- L2 hit rate, shared-memory bank conflicts, registers per thread, local-memory
  traffic;
- active CTAs per SM, eligible warps per cycle, `cp.async` wait stalls,
  long-scoreboard stalls;
- numerical error vs reference, final token agreement, output quality.

**Shapes:** 1×512, 1×4096, 1×32768, 4×8192, 16×2048, 32×512; cold prefill and
prefix-extension prefill.

**Numerics:**
- max hidden-state abs error, relative L2 error, max logit error;
- top-1 agreement, greedy completion equality, sampling distribution checks;
- long-context stability.

## Final prioritized list

1. BF16 Tensor Core GEMMs for hidden 5,120 and FFN 17,408.
2. Multistage `cp.async` + `ldmatrix` + `mma.sync` pipelines.
3. Fused gate/up GEMM + SwiGLU epilogue.
4. Fused output/down projection + residual.
5. Parallel blockwise Gated DeltaNet prefill scans (48 layers).
6. FP32 recurrent-state accumulation with BF16 I/O.
7. SM80 FlashAttention for the 16 full-attention layers.
8. GQA-aware KV-tile reuse across six query heads.
9. Head-dim-256 attention tiling and register management.
10. QK-Norm + partial 64-dim RoPE fusion.
11. Packed varlen execution without padding.
12. Direct K/V and DeltaNet-state writes into final cache layouts.
13. Residual + RMSNorm fusion.
14. Last-token-only LM head.
15. Grouped or persistent GEMMs for small token batches.
16. Split-K/Stream-K for underfilled GEMMs.
17. Split-KV attention for very long, low-batch prompts.
18. Hierarchical scan summaries for long DeltaNet sequences.
19. Shape-bucketed Triton/CUDA dispatch.
20. L2-aware metadata and recurrent-state placement.
21. CUDA graph or persistent execution for remaining launch overhead.
22. TP communication overlap if serving spans multiple A100s.

> The Gated DeltaNet scan and FFN GEMMs deserve priority over full-attention
> optimization; treating prefill as "implement FlashAttention on A100" would
> optimize only a minority of the layers.
