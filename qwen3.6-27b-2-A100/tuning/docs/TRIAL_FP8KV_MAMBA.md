# Trial: fp8 KV cache and a pinned GDN state cache on r1 — 2026-09-19

Cookbook levers 1 and 2 from `UPGRADE_v0.5.20.md`, trialled on **r1 only**;
r0 stays on the production config as the control. Engine v0.5.20 throughout.

## What changes, and the SM80 facts that shape it

- `--kv-cache-dtype fp8_e4m3` for the 16 full-attention layers (64 KiB → 32
  KiB per token). The GDN layers are untouched (`--mamba-ssm-dtype bfloat16`);
  the DFlash2 draft KV is already fp8.
- On SM80 fp8 is a **storage format only**: no fp8 tensor cores, no hardware
  fp8→bf16 convert (`cvt ... e4m3x2` arrived with SM89). FlashInfer dequantizes
  in software in the attention kernel, then runs the bf16 MMA. Decode is
  bandwidth-bound (`a100-node-decode-is-bandwidth-bound`), so halving KV bytes
  should help at long context; prefill is compute-bound and may pay for the
  dequant.
- The cast is **uncalibrated**: `set_kv_buffer` does `cache_k.to(fp8)`,
  `k_scale`/`v_scale` unset on this BF16 checkpoint. e4m3 has 3 mantissa bits
  and **clips** at ±448. Tested in the image: torch 2.13 saturates, no NaN.
  K is bounded by QK-norm; V is not.
  Measuring V directly needs a CPU forward pass of the 27B model (EPYC 7663: no
  AVX-512 or AMX), which is too slow, so the gate measures the effect instead
  (below). If V clipping shows, `fp8_e5m2` (max 57,344, 2 mantissa bits) is the
  range-safe fallback.
- **Draft KV sizing trap (still true on 0.5.20):** the draft KV pool gets the
  target's token count and comes out of activation headroom, not the static
  budget (boot log: 182,528 tokens, 1.74 GB). Doubling target tokens at mem
  0.94 would leave ~1.1 GB free — below where the first DFlash2 boot died
  (0.43 GiB) plus margin. So fp8 comes with `--mem-fraction-static 0.92`.
- `--max-mamba-cache-size`: a request needs 5 slots (`extra_buffer` + overlap:
  3 + 2), so 24 still admits `--max-running-requests 4`. The verify scratch
  stays sized for 4 either way; each slot not kept frees ~73 MB.

## Arms (r1, each booted with `NO_REGISTER=1 ./deploy/roll-replica.sh r1`)

| arm | flags vs production | predicted pool | predicted free |
|---|---|---|---|
| base | none (bf16, mem 0.94, 43 slots by ratio) | 182,528 | 2.80 GB (measured) |
| A | fp8_e4m3, mem 0.92, `--max-mamba-cache-size 43` | ~313K | ~3.2 GB |
| B | fp8_e4m3, mem 0.92, `--max-mamba-cache-size 24` | ~358K | ~2.7 GB |

A pins 43 so it isolates fp8. B adds the slot pin: fp8 doubles the KV pool
while B halves the device-resident prefix states, so B is only worth it if the
KV it buys is worth more than the states it drops. Traffic says KV binds: the big-context key
(69K average prompt) gets a 45% hit ratio with room for ~2.5 of its contexts,
host hits are 1-2.5% of cached tokens, and there were 0 retractions in 7 days.

`extra_buffer_lazy` and `--chunked-prefill-size 2048` are out of this trial:
one variable class at a time.

## Gates — fixed before any arm runs

All on r1 drained and flushed, arm vs base on the same replica.

**Accuracy** (`kvq_eval.py`, `spec_eval.py`, `mamba_ckpt_probe.py`)
- Teacher-forced logprobs, 8 documents of 8K-128K tokens: mean ΔNLL ≤ +0.01
  nats overall and ≤ +0.03 on any single document, with no trend
  that grows with context length. Reference scale: a correct cache hit moves
  logprobs by a mean of 0.06-0.11 against a cold run (`mamba_ckpt_probe`).
- Lookup, 18 codes at 16K/64K/128K: no fewer hits than base.
- spec_eval: greedy 12/12, 0 leaks; accept length within ±5% of base.
  Accept length is the free fidelity signal: the target verifies every draft
  token, so a shifted target distribution shows up as lower acceptance.
- mamba_ckpt_probe: PASS.

**Speed** (`spec_eval.py`, `hicache_probe.py`, kvq cold prefill times)
- Decode aggregate: no phase worse than -3% of base after accounting for
  accept length. Expected gain at ~55K c=4.
- Cold 45K TTFT: ≤ +3% (the SM80 software dequant is the risk).

**Stability**
- Boot: pool within 5% of prediction, `available_gpu_mem` ≥ 2.5 GB.
- `mem_stress.py --side-tokens 60000`: a 160K cold prompt plus three 60K
  requests decoding, twice (~340K tokens, the new capacity). 0 errors, no
  `OutOfMemoryError` in the log.
- hicache_probe: host reload hit.

Passing arm → live soak on r1 (a separate decision), watching
`OutOfMemoryError`, retractions, aborts, per-request TTFT/decode and the
big-context key's hit ratio against r0.

# Results — 2026-09-19, r1 drained 13:55-15:35

Sequence: bf16 baseline, bf16 rerun (noise floor), Arm A, Arm B, back to
production, short-document baseline, re-registered. Every result file is
`tuning/results/*_r1_{base_bf16,armA_fp8_m43,armB_fp8_m24,short_*}.json`;
bf16 spec_eval/hicache/mamba baselines are the same-day `*_v0520.json`.

## Boot

| | base bf16 | A: fp8, 43 slots | B: fp8, 24 slots |
|---|---|---|---|
| KV tokens | 182,528 | **282,304 (+55%)** | **316,992 (+74%)** |
| GDN slots | 43 | 43 | 24 (max_running still 4) |
| `available_gpu_mem` | 2.80 GB | 4.36 GB | 4.36 GB |
| KV host pool (HiCache x3) | 35.9 GB | 27.8 GB | not recorded |
| boot to healthy | 181 s | 171 s | 171 s |

The prediction was off: 0.02 of mem-fraction cost 2.5 GB of KV, not ~1.6 GB,
so the pools came out 10% under and headroom 1.2-1.6 GB over. mem 0.93 would
buy back ~40K tokens at ~3.1 GB headroom.

## Accuracy

**Noise floor is zero:** the bf16 rerun reproduced every logprob of all 8 long
documents bit for bit (cold prefill times within 1%). Arms A and B are also
bit-identical to each other (the slot count does not touch arithmetic), so
every difference below is the KV dtype.

| test | base bf16 | fp8 (A = B) | gate |
|---|---|---|---|
| long docs, 8 x 8K-128K, 2,040 tokens: pooled ΔNLL | — | +0.0051 ± 0.0052 | ≤ +0.01 ✓ |
| long docs: worst single document | — | +0.0365 (prose 8K) | ≤ +0.03 ✗ |
| short docs, 48 x 8K, 12,240 tokens: prose (24) | — | **-0.0008 ± 0.0009** | ✓ |
| short docs: code (24) | — | +0.0239, one doc +0.479; median +0.0005 | ✗ (one doc) |
| lookup, 18 codes at 16K/64K/128K | 18/18 | 18/18 | ✓ |
| spec_eval greedy | 12/12 ok | 10 ok, **seq truncated 2/3** | ✗ |
| spec_eval leaks / serial determinism | 0 / 2-2 | 0 / 2-2 | ✓ |
| accept length, mean of 4 phases | 3.27 | A 3.21, B 3.28 | ✓ |
| mamba_ckpt_probe | PASS | PASS | ✓ |

Reading it:
- Long context holds. ΔNLL at 32K-128K is within ±0.003 and lookup at 128K is
  perfect, so V clipping at ±448 is not hurting.
- Ordinary prose is unaffected.
- The misses sit on "knife-edge" tokens where bf16 itself is unstable. The worst
  code document and the prose-8K outlier are regions where bf16 already gives
  confident-looking copies -15 to -25 nats (see the anomaly below); fp8 flips
  some of those (one token -0.9 → -28).
- The greedy `seq` task (300 numbers, ~7K generated tokens) hit the 8,000-token
  cap in 2 of 3 runs under fp8, against 0 of 6 in the two bf16 runs. That is a
  changed greedy trajectory, not a wrong answer, but the gate as written fails.

## Speed — the SM80 cost

| | base bf16 | A | B |
|---|---|---|---|
| cold prefill 8K / 32K / 64K / 128K | 2.1 / 9.5 / 22.1 / 57.0 s | 2.2 / 10.2 / 25.0 / 68.3 s | same as A |
| vs base | | **+2% / +7% / +13% / +20%** | |
| cold 45K (hicache_probe) | 14.14 s | 15.65 s (+11%) | 15.76 s (+11%) |
| device hit / host reload, 45K cached | 0.22 / 0.42 s | 0.23 / 0.29 s | 0.28 / 0.31 s |

spec_eval aggregate tok/s (accept length):

| phase | base bf16 | A | B |
|---|---|---|---|
| short c=1 | 76.2 (3.40) | 69.2 (3.11) | 73.0 (3.28) |
| short c=4 | 234.5 (3.30) | 243.3 (3.27) | 241.9 (3.22) |
| ~55K c=1 | 63.9 (3.10) | 67.6 (3.28) | 70.1 (3.42) |
| ~55K c=4 | 201.9 (3.29) | 185.8 (3.19) | 184.8 (3.19) |
| ~55K c=4 TTFT median | 0.98 s | 1.46 s | 1.50 s |

Per unit of accept length: short c=1 and ~55K c=1 flat, short c=4 +5%, ~55K
c=4 **-5% in both arms**. fp8 KV does not speed decode on this GPU.

Why (FlashInfer `vec_dtypes.cuh` in the image): the hardware fp8 convert is
enabled only for `__CUDA_ARCH__ >= 900`. On SM80 fp8→bf16 is a software
shift/mask/multiply (`fast_dequant_f8f16x4`) in the attention inner loop.
Prefill re-reads each KV tile once per query tile, so the dequant is paid many
times and grows with context (+20% at 128K). Decode reads each tile once;
there the halved bytes and the extra ALU roughly cancel. `fp8_e5m2` would not
help: the cheap byte-permute path exists only for e5m2→fp16, and this model
runs bf16.

## Stability

`mem_stress --side-tokens 60000` (160K cold + three 60K decoding, twice, ~340K
tokens) on A and B: 0 errors, 0 `OutOfMemoryError`/Traceback lines, 0
restarts. Headroom never came close.

## Verdict

| gate class | A | B |
|---|---|---|
| accuracy | ✗ (single-doc, greedy seq) | ✗ (same numerics) |
| speed | ✗ (cold prefill +11% at 45K, +20% at 128K) | ✗ |
| stability | ✓ | ✓ |

**Not adopted.** fp8 KV on A100 buys +55% (A) / +74% (B) KV at a prefill cost
that the traffic mix pays on every miss (the big-context key's misses are
~70K-token cold prefills, now ~+15%), no decode gain, and a measurable,
if small, fidelity change on code and long greedy generations. The capacity
could still net out positive for the big-context key through a higher hit ratio,
but that is a live-soak question and the pre-registered gates said no soak.

Not tested and cheap: the slot pin **without** fp8 (bf16, 24 slots). It frees
~1.3 GB → ~+21K tokens (+12%) with no numerics change at all; B showed that
24 slots keep max_running 4 and pass the stress and checkpoint probe.

## Applied anyway — operator decision, 2026-09-19 16:05

Arm B went to **both replicas** with `--schedule-policy hrrn` (was `lpm`),
knowing the gate failures above: the operator weighed +74% KV above the prefill
and fidelity costs. Rolled r1 then r0 with `roll-replica.sh`, no router
restart.

- Both: `max_total_num_tokens=316992`, 24 slots, `available_gpu_mem=4.36 GB`,
  `hicache_attached=True`; `/get_server_info` shows `schedule_policy hrrn`,
  `kv_cache_dtype fp8_e4m3`; r0 vs r1 differ only in `random_seed`/`startup_time`.
- Smoke 5/5 on each; live requests completing on both, 0 error lines.
- HRRN had no A/B. `_validate_and_adjust_policy` falls back to FCFS silently when the
  cache has no `disable` attribute; `UnifiedRadixCache` defines it, so HRRN
  is active (verified in server info).

**Watch:** `OutOfMemoryError` in worker logs, `num_retractions` and TTFT by
consumer in `engine.requests`, and the big-context key's hit ratio (45% on the
7 days before). Back out: remove `--kv-cache-dtype` and
`--max-mamba-cache-size`, mem 0.94, policy `lpm`, roll each replica.

**Superseded, same day (19:30):** the 24-slot pin is the wrong half of Arm B.
`TRIAL_V0520_LEVERS.md` measured that the GDN state tier, not KV, limits
agent cache hits: with 24 slots a replica keeps 2 long contexts instead of 4,
and a cache replay went from 0.66 to 0.00 hit ratio. The reasoning above that
"KV binds" was wrong; this trial never measured multi-context retention.
