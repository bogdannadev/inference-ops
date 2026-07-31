# Manual Kernel Tuning — SPEC

What to do, in what order, with what gates. Facts and citations live in
`KERNEL_TUNING_INFO.md`; deployment context in `TUNING_PLAN.md`.

**Scope:** the 48 Triton GDN linear-attention layers of Qwen3.6-27B on
2× A100 80GB PCIe (SM80). The 16 full-attention layers run on FlashInfer and
are out of scope for hand-tuning.

---

## S0. Preconditions — do not start before these hold

| # | precondition | why |
|---|---|---|
| S0.1 | **Phase 1 rolled and measured** (`--max-running-requests 4`, `--cuda-graph-max-bs-decode 4`, HiCache removed) | The ladder cannot distinguish *any* kernel change while the admission cap of 4 makes every rung ≥6 a wave replay of c=4. Every A/B would return the same number. |
| S0.2 | `decode.bs` reads `[1,2,3,4]` in `/get_server_info` | Confirms the CUDA-graph clamp lifted. If it still reads `[1,2]`, S0.1 did not take. |
| S0.3 | Profiling done (S1), GDN share of decode time known | Decides whether S2 is worth doing at all. |
| S0.4 | Image digest unchanged | A kernel measurement across two engine builds measures nothing. |

**S0.1 is not a formality.** It is also a *kernel* precondition: the GDN decode
kernel's occupancy is 5.6% of A100 warp slots at batch 2 and 11.1% at batch 4
(`KERNEL_TUNING_INFO.md` §3a). Raising the batch is the single largest
improvement available to that kernel, and it requires no kernel edit.

---

## S1. Profile first — establish the ceiling before spending effort

Decode is memory-bound by ~40–80× (arithmetic intensity 2–4 vs A100's BF16
balance point of 161 FLOP/byte), and weight streaming already consumes 56.1% of
HBM peak. Kernel work can only recover the *non-streaming* fraction of a
forward pass. Measure that fraction before optimising it.

**Method — torch profiler via HTTP, no restart required:**

```bash
docker exec qwen36-27b-r0 sh -c \
  'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
        -X POST http://localhost:8001/start_profile'
# drive steady-state decode load through the router, e.g.
#   SGLANG_API_KEY=... python3 benchmarks/ladder.py --rungs 4 --max-tokens 400
docker exec qwen36-27b-r0 sh -c \
  'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
        -X POST http://localhost:8001/stop_profile'
```

For a per-layer timeline instead, roll with `--enable-layerwise-nvtx-marker`
and capture under `nsys`. `--enable-forward-pass-metrics` adds per-pass timing
to Prometheus and is cheap enough to leave on.

**Extract:** the wall-clock share of a decode step spent in
`fused_sigmoid_gating_delta_rule_update_kernel` (this is the target-verify GDN
kernel, 48 layers × every verify), versus FlashInfer full attention, versus the
QKV/MLP GEMMs, versus the draft model.

### Decision rule

| GDN share of decode time | action |
|---|---|
| **< 15%** | **Stop. Do not do S2.** Amdahl caps the whole exercise below ~5%. Spend the effort on Phase 2 (EAGLE depth) and §4f (Mamba pool) instead. |
| 15–30% | Do S2.1 and S2.2 only (launch-config sweeps). Skip S3. |
| > 30% | Full S2 + S3 justified. |

Record the number in this file when measured. Until then, S2 below is a
*conditional* plan, not an instruction.

---

## S2. Launch-configuration tuning — no kernel logic changes

Ordered by (expected value ÷ risk). Each item is independently reversible.

### S2.1 — `chunk_delta_h` env knobs (prefill/extend; **zero code change**)

The only knobs upstream deliberately exposed. Affects TTFT, not decode.

```
SGLANG_GDN_CHUNK_H_BV          default 32   sweep {16, 32, 64, 128}
SGLANG_GDN_CHUNK_H_NUM_WARPS   default 4    sweep {2, 4, 8, 16}
SGLANG_GDN_CHUNK_H_NUM_STAGES  default 2    sweep {1, 2, 3, 4}
```

`num_warps` sweep range is the non-Hopper list, which is correct for SM80.
Set them in the replica `environment:` block in `docker-compose.yml`.

> **HARD CONSTRAINT — do not re-enable multi-config autotune on this kernel.**
> It writes final state back into `initial_state` **in place**; Triton's
> autotune benchmark phase runs the kernel repeatedly for timing and
> **corrupts the SSM state pool, producing silently wrong output on the first
> user request**. Upstream tried `restore_value=["initial_state"]` and it OOMs
> at production model scale. The single-config-plus-env-knobs arrangement is
> the deliberate workaround. Sweep by *restarting with different env values*,
> one at a time — never by widening the `configs=[...]` list.

**Risk:** low. **Gate:** TTFT on a long-prompt request, plus the ladder to
confirm no decode regression.

### S2.2 — `num_warps` / `num_stages` on the hot decode kernel

`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:292-293`.
Currently hardcoded `num_stages = 3`, `num_warps = 1`. This is the highest-value
item in S2 if S1 says GDN is a meaningful share.

Two independent hypotheses, each mechanically motivated:

1. **`num_warps = 1` is too narrow.** The recurrent state tile is
   `[BK, BV] = [128, 32] = 4096` fp32 accumulators spread over one warp =
   **128 registers/thread for state alone**, against A100's 255/thread ceiling.
   `num_warps = 2` halves that to 64. Sweep `{1, 2, 4}`.
2. **`num_stages = 3` is pipelining nothing.** The inner loop is a *serial*
   recurrence with trip count `T = 1` (decode) or `T = 4`
   (target verify, `= --speculative-num-draft-tokens`). Depth-3 software
   pipelining over 1–4 serial iterations costs shared memory and registers and
   returns nothing. Sweep `{1, 2, 3}`.

Also sweep `BV ∈ {16, 32, 64}` (currently `min(next_pow2(V), 32) = 32`):
`BV = 16` doubles `NV` to 8 → 1536 blocks at batch 4 (22% occupancy vs 11%) at
the cost of 2× redundant q/k loads. Grid-starvation may well make that a win —
see the occupancy table in `KERNEL_TUNING_INFO.md` §3a.

**Method:** patch via a bind-mounted overlay of the single file, not by baking
a new image — keeps the digest pin intact (S0.4). Add to the replica's
`volumes:`:

```yaml
- ./kernels/fused_sigmoid_gating_recurrent.py:/sgl-workspace/sglang/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:ro
```

Copy the pristine file out first (`docker cp`), keep it under version control,
and diff every variant against it. Delete the mount to roll back.

**Correctness gate — mandatory, this kernel is not a black box.** Before any
timing claim, confirm output equivalence: run a fixed prompt at
`temperature 0` against baseline and patched, and require **identical token
IDs**. `num_warps`/`num_stages` changes should be bit-identical; a divergence
means the tile change altered reduction order and must be investigated, not
accepted because it was faster.

### S2.3 — restore autotune on `chunk_o` and `l2norm`

Both have `@triton.autotune` **commented out** upstream with fixed configs
substituted, and both left the intended sweep in the comment. Neither writes
state in place, so the S2.1 corruption hazard does **not** apply to them.

`chunk_o.py` — currently fixed at `num_warps=4, num_stages=2`. Recorded sweep,
resolved for A100 (`check_shared_mem()` is True, `is_nvidia_hopper` is False):

```python
BK ∈ [64, 128]   BV ∈ [64, 128]   num_warps ∈ [2, 4, 8]   num_stages ∈ [2, 3, 4]
```

`l2norm.py` — currently fixed at `num_warps=8, num_stages=3`. Recorded sweeps:
`num_warps ∈ [1,2,4,8,16,32]`, and for the second kernel additionally
`BT ∈ [8,16,32,64,128]`.

Prefer **uncommenting the autotune block** over hand-picking: it is upstream's
own intended range, it is keyed correctly, and it self-selects per shape. Cost
is a slower first request per new shape. Verify from the profile that these
kernels are actually material first — `l2norm` in particular may be noise.

### S2.4 — `fused_gdn_gating` and the `fused_recurrent.py` launch sites

`fused_gdn_gating.py:73` launches at `num_warps=1`. `fused_recurrent.py` has
the same `num_stages=3, num_warps=1` pair at lines 139-143, 356-363 and
615-622, plus a fourth site at 1023-1026 using `BV = min(next_pow2(V), 8)`.

Lowest priority: under EAGLE the packed-decode path in `fused_recurrent.py` is
not the hot path (target-verify is). Confirm from the profile before touching.

---

## S3. Structural work — only if S1 says GDN > 30%

Not recommended on current evidence. Listed so the boundary is explicit.

- **`NK > 1`.** `fused_sigmoid_gating_recurrent.py:291` asserts `NK == 1`.
  With `K = 128` and `BK = 128` we are exactly at the boundary; splitting K
  would need a cross-block reduction the kernel does not implement. Upstream
  work, not local work.
- **Fusing `l2norm` into the recurrent kernel.** `use_qk_l2norm_in_kernel=True`
  is already passed, so this may already be fused — verify in the profile
  before assuming there is a separate launch to eliminate.
- **A dedicated SM80 GDN kernel.** The `gdn_blackwell/` and `kda_blackwell/`
  trees show what a tuned architecture-specific path looks like; there is no
  SM80 equivalent. Writing one is a genuine engineering project, and the
  56.1%-of-HBM-peak weight-streaming result caps its payoff. Out of scope.

---

## S4. Non-kernel items that outrank all of S2

State plainly, because they are cheaper and larger:

1. **Phase 1** — batch 2→4. Already edited, awaiting roll. Doubles GDN kernel
   occupancy as a side effect.
2. **Phase 2** — EAGLE depth. Accept length is 3.483 in production against a
   cap of 4.00, with 15.3% of verifies pinned at the cap and per-token
   acceptance p ≈ 0.908. The current `steps=3/topk=1/draft=4` is verbatim the
   SGLang docs' **OOM-recovery recipe**, not a tuned setting. Use
   `bench_speculative.py`. Watch the Mamba-slot coupling: at `draft=6` the
   48-slot pool caps concurrency at 8.
3. **`--enable-fused-qk-norm-rope`** — currently off. Qwen3.5/3.6 uses QK-norm;
   this fuses norm + RoPE. One flag, low risk, do it in the same roll as
   something else.
4. **§4f Mamba pool expansion** — `--mamba-full-memory-ratio`,
   `--enable-int8-mamba-checkpoint`. Raises the 48-slot ceiling that will
   eventually bind Phase 1 and Phase 2 simultaneously.

**Explicitly deprioritised:** the `--linear-attn-backend` /
`--mamba-backend` alternatives in `TUNING_PLAN.md` Phase 3. All of
`flashinfer`, `cutedsl`, `flashkda` are gated behind SM90+/SM100+ and are dead
on this hardware (`KERNEL_TUNING_INFO.md` §4). They will refuse or silently
fall back to Triton; the boot log will show no change. Do not spend a roll.

---

## S5. Measurement discipline

1. **One variable per roll.** Every item above is independently reversible.
2. **One replica at a time**, so the other keeps serving:
   ```
   docker compose up -d --no-deps qwen36-27b-r1
   # wait healthy (~5 min now that HiCache is gone; was ~9)
   docker compose up -d --no-deps qwen36-27b-r0
   docker compose restart qwen36-27b-router
   ```
   **Never `--remove-orphans`** — it would delete `qwen3-emb`, `grafana`,
   `prometheus`, `dcgm`, which belong to other projects.
3. **Gate on `ladder.py`, not `run_worker.sh`.** Single-stream numbers were
   identical at mem-fraction 0.85 and 0.90 (68.50 vs 68.97 tok/s) and will
   under-report anything that helps batching.
4. **Read the ladder correctly.** `aggregate_tok_s` is tokens ÷ wall.
   `split_r0`/`split_r1` are `spec_verify_calls_total` deltas counted **per
   request** — not tokens, and not batched forwards. Batched forwards per
   replica = `split ÷ concurrent_requests_on_that_replica ÷ wall`.
5. **Read the boot log every time**: `max_total_num_tokens` (must stay 160768),
   `max_mamba_cache_size`, captured CUDA-graph `bs` lists, and whether a
   requested backend actually engaged rather than falling back.
6. **A/B above the admission cap is meaningless.** See S0.1.
7. **Correctness before speed.** Identical token IDs at `temperature 0`, or the
   result is discarded regardless of its timing.
