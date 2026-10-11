# Trial 2026-10-09: chunked-prefill crash patch and K/V range capture

**Window:** r1 drained 18:15 to 19:27 UTC; r0 served alone, with p95 queue
wait 14.5 s and up to 8 queued. r1 was restored to the exact production
config: arguments identical to r0, the live `deploy/enginepatch/` equal to git,
the shared L3 directory, pool 282,496.

**Arms:** each is a compose override from `tuning/trial/make_arm.py`, applied
with `COMPOSE_FILE=docker-compose.yml:tuning/trial/<arm>.yml NO_REGISTER=1
./deploy/roll-replica.sh r1`. Every arm had its **own** HiCache L3 directory,
so no trial ever wrote into r0's cache. Those directories were deleted
afterwards (67 GB).

## Part 1: the overcommit crash and patch 3

Background: on 2026-10-09 at 16:50 UTC, r0's scheduler died on
`Prefill out of memory` (4096 requested, 960 free, 0 evictable). The cause is
`PrefillBudget.available_chunk_tokens`, which falls back to a full chunk when
the budget is ≤ 0. Patch 3 is in `deploy/enginepatch-next/sitecustomize.py`;
its docstring has the full mechanism.

| arm | code | load | result |
|---|---|---|---|
| A0 | stock | 2–3 long decodes + a prompt that fills the pool (pool capped at 65,536; `--schedule-conservativeness 0.05`); slack swept 512→32 | **no crash**: a chunk was never clamped mid-prefill. Admission makes the whole prompt fit, and decodes grow only a few tokens per chunk round. |
| A2 | stock + drift injector | 2 decodes + a 20K prompt; on the first *continuing* chunk the injector takes all but 3,000 free KV tokens for 20 s | **crash, with Friday's exact sequence**: chunk clamped to 3008, then `phys=0 remaining_total=0 → 4096`, `Prefill out of memory`, scheduler exception, SIGQUIT, in-flight requests get HTTP 500 |
| A3 | patch 3 + drift injector | same as A2 | **survives**: same clamp to 3008, then `gave None` (parked) while the KV was held, resumed with 4096-token chunks on release, every request completed, KV usage 0.0 after a flush (no leak) |

**What this proves and what it doesn't:**
- The crash needs KV to disappear *after* the long prompt is admitted. That
  drift is what the injector supplies. Friday's real source of drift was not
  identified; the leading suspect is HiCache loading a queued request's prefix
  back to the GPU (`#queue-req: 1`, every node `full_lock=1`).
- The patch is exercised through the real scheduler, allocator and FlashInfer
  path, not just a unit test.
- While parked, the scheduler re-tries every round: about 1.4k budget calls a
  second, a busy-wait. Upstream's SWA park behaves the same way. Patch 3 logs
  at most once a minute.
- Decodes stalled during the injected steal (52 s → 118 s) because the
  injector took *all* free KV. Real drift is far smaller. The decodes survive
  rather than die.

**Tests:**
- CPU tests: `deploy/enginepatch-next/verify.py`, all pass.
- Negative control: unpatched upstream returns 4096 for Friday's state.
- GPU fault injection: `tuning/trial/build_inject.py` (run from `deploy/`)
  builds the TEST-ONLY `enginepatch-test-{stock,patched}`, driven by
  `tuning/trial/repro_drift.py`.

**Trap found while building the injector:** an `import` inside a patch
function re-enters the `__import__` hook and recurses (`RecursionError`, and
the patch is silently skipped). Patch 3 imports at module level and is
unaffected. Any future patch must do the same.

## Part 2: K/V value ranges (A4) — calibrated fp8 scales would not help

**Method.**
- A TEST-ONLY hook on `HybridLinearKVPool.set_kv_buffer` (built by
  `tuning/trial/build_capture.py`) histograms log2|K| and log2|V| before the
  fp8 cast, per attention layer, plus the amax. It skips CUDA-graph capture,
  so it records eager prefill only.
- Input: four real-text prompts (6.9K, 29.6K, 68.1K and 100.5K tokens) drawn
  from this public repo's tracked docs and code. That is about 176K tokens
  prefilled in the production config.
- Data: `tuning/results/kvcal/kvstats.pt`. Analysis:
  `tuning/results/kvcal/analyze.py`.

**Result (excluding padding).** Exactly 23.94% of rows are zero in every layer,
for both K and V. That matches the number of padded rows, so those rows are
padding, not data.

| | K | V |
|---|---|---|
| max \|x\| over 16 layers | 11.6–24.8 | 9.2–133 (grows with depth) |
| p99.9 \|x\| | 5.5–8.9 | 2.1–26.9 |
| above 448 (clipped) | 0 | 0 |
| e4m3 subnormal, 2⁻¹⁰…2⁻⁶ | 0.7–1.0% | 0.3–3.0% |
| flushed to zero, < 2⁻¹⁰ | ~0.05% | 0.02–0.2% |
| fp8 error energy, scale 1.0 | −31.5 dB | −31.5 dB |
| fp8 error energy, scale amax/448 | −31.5 dB (±0.3 per layer) | −31.6 dB (±0.4 per layer) |

**Reading.**
- K and V already sit well inside e4m3's normal range at scale 1.0.
- A per-tensor scale cannot add mantissa bits. The fp8 error is about 2.7%
  RMS per value from the 3-bit mantissa, whatever the scale.
- So the 83% recovery reported by arXiv 2609.04098 (vLLM, W4A4 K/V
  projections) does not carry over to our configuration.
- *(2026-10-11: there is no such weakness; the `seq` 1/5 was the 8K token cap. See `TRIAL_2026-10-11_FP8_WEIGHTS.md`.)*
- **For the long-exact-output weakness (`seq` 1/5), the remedy is bf16 KV,
  not scales.** On this hardware the memory for that has to come from FP8
  weights.
- This also retires the KV-scale engineering work: the bf16 engine patch, and
  adding scale tensors to an FP8 checkpoint.

## Next

- **Apply patch 3:** copy `deploy/enginepatch-next/` into `deploy/enginepatch/`
  and roll one replica at a time (≈4 min each). This is a deliberate apply
  step, not part of this trial.
- **FP8 weights vs bf16 KV:** the 2×2 in `FP8_WEIGHTS_LONG_CONTEXT.md`, now
  without the calibrated-scale arm. Use a quieter window: about 1.5 h of r1.
  - The checkpoint `Qwen/Qwen3.8-27B-FP8` at revision `017b9c7` is already in
    the HF cache.
- **B1 attention selection:** needs r1 stopped, about 30 min.
