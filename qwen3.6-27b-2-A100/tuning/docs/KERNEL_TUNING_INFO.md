# Manual Kernel Tuning — INFO

Reference facts for hand-tuning SGLang kernels on **this** deployment.
Everything below was read from the running container / model config on
2026-07-31, not from general documentation. Companion doc:
`KERNEL_TUNING_SPEC.md` (what to do). Context: `TUNING_PLAN.md`.

Engine build: `sglang 0.0.0.dev1+ga358374ae`, image pinned by digest
`lmsysorg/sglang@sha256:647d7bb0…3399b4`.

---

## 1. Hardware — NVIDIA A100 80GB PCIe (SM80)

Queried live via `torch.cuda.get_device_properties(0)`:

```
compute capability            (8, 0)          -> sm_80, Ampere
device name                   NVIDIA A100 80GB PCIe
multi_processor_count         108
regs_per_multiprocessor       65536           (32-bit registers)
shared_memory_per_block       49152           (static default; see note)
```

Architectural constants that bound kernel design on SM80:

| property | value | consequence for tuning |
|---|---|---|
| SMs | 108 | a grid under ~108 blocks cannot fill the GPU at all |
| max threads / SM | 2048 = **64 warps** | 108 × 64 = **6912 warp slots** machine-wide |
| max resident blocks / SM | 32 | with 1 warp/block, occupancy caps at 32/64 = 50% |
| registers / SM | 65536 | 128 regs/thread ⇒ 512 threads/SM ⇒ 25% occupancy ceiling |
| max registers / thread | 255 | hard cliff; spills go to local memory (HBM) |
| shared memory / SM | 164 KB | **163 KB opt-in max per block**; the 48 KB reported above is the static default, Triton requests the larger dynamic allocation explicitly |
| L2 cache | 40 MB | |
| HBM2e bandwidth | **1935 GB/s** | the number every roofline here divides by |
| BF16 tensor-core (dense) | **312 TFLOPS** | |
| **BF16 balance point** | **312e12 / 1935e9 ≈ 161 FLOP/byte** | anything below this is memory-bound |

**What SM80 does *not* have** — this is why several backends are unavailable
(§4):

- no FP8 tensor cores (SM89+)
- no `wgmma` warpgroup MMA (SM90+)
- no TMA / tensor memory accelerator (SM90+)
- no tensor memory / `tcgen05` (SM100+)
- `cp.async` (async global→shared copy) **is** available — SM80 introduced it,
  so Triton `num_stages` pipelining is real here, not a no-op

**Decode arithmetic intensity on this deployment: ~2–4 FLOP/byte at batch 2–4,
against a balance point of 161.** Decode is memory-bound by a factor of ~40–80.
No kernel change alters that; kernel work only recovers the fraction of a
forward pass *not* spent streaming the 54 GB of weights.

### `fla` capability helpers (live values)

The Triton kernel package branches its tuning ranges on these:

```
is_nvidia_hopper        False        -> NUM_WARPS ranges use the WIDER non-Hopper list
check_shared_mem()      True         -> chunk_o BKV_LIST = [64, 128]  (not [32, 64])
check_shared_mem('ampere')   True
check_shared_mem('hopper')   False
```

---

## 2. Model — Qwen3.6-27B, BF16

From `config.json` → `text_config` (`model_type: qwen3_5_text`):

```
num_hidden_layers            64
layer_types                  48 x linear_attention  +  16 x full_attention
full_attention_interval      4
hidden_size                  5120
intermediate_size            17408
vocab_size                   248320
max_position_embeddings      262144        (model native; we serve 160000)
mtp_num_hidden_layers        1             (the EAGLE draft head)
attn_output_gate             True
output_gate_type             swish
partial_rotary_factor        0.25
rope_theta                   10000000
```

### Two different head geometries — do not conflate them

| | full attention (16 layers) | linear attention / GDN (48 layers) |
|---|---|---|
| head dim | **256** (`head_dim`) | **128** (`linear_key_head_dim`, `linear_value_head_dim`) |
| heads | 24 Q / **4 KV** (GQA) | 16 key heads / **48 value heads** |
| kernel | FlashInfer (`--attention-backend flashinfer`) | **Triton, forced** — see §4 |
| conv | — | `linear_conv_kernel_dim 4` (causal conv1d) |
| state dtype | — | config says `float32`; we override to `bfloat16` via `--mamba-ssm-dtype` |

> **CORRECTION to `TUNING_PLAN.md` §4a.** That section hypothesised that
> `head_dim = 256` was unusual and would leave the *Triton linear-attention*
> kernels mistuned. That premise is wrong. 256 is the **full-attention** head
> dim, and those 16 layers run on **FlashInfer**, not Triton. The 48 Triton GDN
> layers run at `head_k_dim = head_v_dim = 128` — a completely standard size.
>
> The real tuning opportunity is different and better evidenced: see §3.

### Derived quantities

```
weights (BF16)               ~54 GB per replica (TP=1, full model per GPU)
KV cache                     only the 16 full-attention layers carry KV
  160768 tok x 16 layers x 4 kv_heads x 256 head_dim x 2 B = 5.27 GB (K)
  boot log reports K size 4.91 GB -> confirms 16-layer KV
Mamba/SSM state pool         48 slots, 4 slots per running request
```

---

## 3. Kernel inventory — the Triton GDN path

Package root: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/`
Dispatch wrapper: `srt/layers/attention/linear/kernels/gdn_triton.py`

Which kernel runs when, under our EAGLE config:

| phase | entry point | file |
|---|---|---|
| **target verify** (hot: 48 layers × every verify) | `fused_sigmoid_gating_delta_rule_update` | `fused_sigmoid_gating_recurrent.py` |
| decode (non-spec) | `fused_sigmoid_gating_delta_rule_update` | same |
| decode (packed fast path) | `fused_recurrent_gated_delta_rule_packed_decode` | `fused_recurrent.py` |
| prefill / extend | `chunk_gated_delta_rule` → `chunk_delta_h` + `chunk_o` | `chunk.py`, `chunk_delta_h.py`, `chunk_o.py` |
| gating | `fused_gdn_gating` | `fused_gdn_gating.py` |
| QK L2 norm | `l2norm_fwd` | `l2norm.py` |

### 3a. The hot decode kernel is launched with `num_warps = 1`, hardcoded

`fused_sigmoid_gating_recurrent.py:289-293` — these are **plain Python
assignments**, not `@triton.autotune`, not env-configurable:

```python
BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
assert NK == 1, "NK > 1 is not supported yet"
num_stages = 3
num_warps  = 1
...
grid = (NK, NV, N * HV)
```

Resolved for **this** model (`K = V = 128`, `HV = 48`):

```
BK = 128        BV = 32
NK = 1          NV = 4
grid = (1, 4, N * 48)   ->  192 x N blocks,  1 warp each
```

Occupancy against A100's 108 SMs / 6912 warp slots:

| running batch N | blocks | blocks per SM | warps resident | warp occupancy |
|---|---|---|---|---|
| 1 | 192 | 1.8 | 192 | **2.8%** |
| 2 (config until 2026-07-31) | 384 | 3.6 | 384 | **5.6%** |
| 4 (after Phase 1) | 768 | 7.1 | 768 | **11.1%** |
| 8 | 1536 | 14.2 | 1536 | 22.2% |

**The GDN decode kernel is grid-starved on this GPU.** At the old
`--max-running-requests 2` it occupied 5.6% of the machine's warp slots. It is
latency-bound, not compute-bound or bandwidth-bound, and it cannot be fixed by
picking a different tile alone — the grid simply does not supply enough blocks.

This is an **independent second reason Phase 1 comes first**: raising the batch
from 2 to 4 doubles this kernel's occupancy for free, before any kernel edit.

Register pressure at `num_warps = 1`: the recurrent state tile is
`[BK, BV] = [128, 32] = 4096` elements accumulated in fp32, distributed over a
single warp = **128 fp32 registers per thread for state alone**, before q, k, v
and gating. A100's per-thread ceiling is 255. Going to `num_warps = 2` halves
that to 64/thread. This is the mechanically sound reason to expect the
hardcoded 1 to be leaving performance behind.

The same `num_stages = 3, num_warps = 1` pair appears at three more launch
sites in `fused_recurrent.py` (lines 139-143, 356-363, 615-622) and a fourth at
1023-1026 with `BV = min(next_power_of_2(V), 8)`.

Note also that `num_stages = 3` pipelines a loop that, at decode, has trip
count `T = 1` (plain decode) or `T = num_draft_tokens = 4` (target verify).
Software pipelining depth 3 over a 1–4 iteration serial recurrence buys
nothing and costs shared memory and registers.

### 3b. Prefill kernels — one is already env-tunable, two have autotune disabled

**`chunk_delta_h.py` exposes environment variables.** No code edit needed:

```python
GDN_CHUNK_H_BV         = int(os.getenv("SGLANG_GDN_CHUNK_H_BV",         "32"))
GDN_CHUNK_H_NUM_WARPS  = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_WARPS",  "4"))
GDN_CHUNK_H_NUM_STAGES = int(os.getenv("SGLANG_GDN_CHUNK_H_NUM_STAGES", "2"))
CHUNK_SIZE = 64
```

These feed a **deliberately single-config** `@triton.autotune`. The upstream
comment explains why multi-config autotune was removed, and it is a correctness
constraint worth respecting:

> the kernel writes `ht` (final state) back into `initial_state` **in-place**;
> with multiple configs, Triton's autotune benchmark phase invokes the kernel
> many times for timing and **corrupts the cache pool, producing silently wrong
> output on the first user request**. `restore_value=["initial_state"]` works
> for unit tests but OOMs on production-scale models.

So: **never re-enable multi-config autotune on `chunk_delta_h`.** The env knobs
exist precisely so the tile can be validated per model/hardware without that
hazard. Upstream's own words: "allowing model/hardware-local validation of the
selected tile without corrupting the state pool through multi-config autotune."

**`chunk_o.py` has its `@triton.autotune` commented out** and substituted with
fixed `num_warps=4, num_stages=2`. The commented block records the intended
sweep, and the ranges resolve *on A100* to:

```python
BKV_LIST = [64, 128]          # because check_shared_mem() is True here
NUM_WARPS = [2, 4, 8]         # because is_nvidia_hopper is False
# commented-out sweep: BK in BKV_LIST, BV in BKV_LIST,
#                      num_warps in NUM_WARPS, num_stages in [2, 3, 4]
BT = min(chunk_size, max(16, next_power_of_2(T)))
```

**`l2norm.py`** likewise: two `@triton.autotune` blocks commented out, fixed
`num_warps=8, num_stages=3` substituted. Intended sweeps were
`num_warps ∈ [1,2,4,8,16,32]` (kernel1) and `num_warps ∈ [1,2,4,8,16]` with
`BT ∈ [8,16,32,64,128]`.

**`fused_gdn_gating.py`** launches with `num_warps=1`.

`chunk_o` and `l2norm` are unlike `chunk_delta_h` — they do **not** write state
in place, so the in-place-corruption hazard above does not apply to them. That
makes them the safe candidates for restoring autotune.

---

## 4. Backend availability on SM80 — Triton is the only option

This is the finding that most changes the plan.

```
gdn_flashinfer.py:58    _flashinfer_gdn_available = is_cuda() and capability[0] >= 9
gdn_cutedsl.py:26       _is_blackwell(): SM100+   (decode needs SM90+, prefill SM100+)
kernels/ops/attention/linear/gdn_blackwell/   <- SM100 sources
kernels/ops/attention/linear/kda_blackwell/   <- SM100 sources, and KDA != GDN anyway
```

Our capability is `(8, 0)`. Therefore:

| `--linear-attn-backend` | status on this hardware |
|---|---|
| `triton` (current) | **the only working backend** |
| `flashinfer` | gated behind `capability[0] >= 9` — unavailable |
| `cutedsl` | decode SM90+, prefill SM100+ — unavailable |
| `flashkda` | KDA (Kimi Delta Attention), not GDN — wrong model family |

> **CORRECTION to `TUNING_PLAN.md` Phase 3.** That table lists
> `--linear-attn-backend {flashinfer, flashkda, cutedsl}` as "**Governs 48 of
> 64 layers.** Highest ceiling of any flag here." On SM80 all three are dead —
> they will refuse or silently fall back to Triton. Do not spend a roll on
> them; the boot log will show no change.
>
> The consequence runs the other way, though: because there is **no alternative
> backend for 48 of 64 layers**, hand-tuning the Triton kernels is not one
> option among several. It is the only lever that exists for those layers.
> That *raises* the value of §3, even as it lowers the value of the flag table.

`--mamba-backend flashinfer` is expected to fail the same way. Verify from the
boot log rather than assuming a requested backend engaged — several of these
flags are accepted at parse time and ignored at dispatch.

---

## 5. Current kernel-relevant server args (live, r0, 2026-07-31)

```
attention_backend                 flashinfer      (the 16 full-attn layers)
linear_attn_backend               triton          (the 48 GDN layers)
linear_attn_decode_backend        None
linear_attn_prefill_backend       None
mamba_backend                     triton
mamba_ssm_dtype                   bfloat16        (overrides config's float32)
mamba_radix_cache_strategy        extra_buffer
mamba_full_memory_ratio           0.9
page_size                         64
chunked_prefill_size              16384
enable_fused_qk_norm_rope         False
enable_page_major_kv_layout       False
enable_torch_compile              False
torch_compile_max_bs              32
disable_flashinfer_autotune       False           (autotuner IS on)
flashinfer_autotune_skip_ops      None
bf16_gemm_backend                 auto
triton_attention_num_kv_splits    8
triton_attention_reduce_in_fp32   False
speculative_attention_mode        prefill
speculative_num_steps             3
speculative_eagle_topk            1
speculative_num_draft_tokens      4
cuda_graph_config   decode : backend=full, max_bs=2, bs=[1,2], tc_compiler=eager
                    prefill: backend=breakable, max_bs=16384, 74 buckets,
                             tc_compiler=eager
```

After the Phase 1 roll, `decode.bs` must read `[1,2,3,4]`. If it still reads
`[1,2]`, `--cuda-graph-max-bs-decode` did not lift and the concurrency raise
bought nothing.
