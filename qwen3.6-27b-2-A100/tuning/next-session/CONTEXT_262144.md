# Next session — raising served context toward the model's native 262,144

**Status: EXECUTED 2026-07-31b. Outcome below. Parts of the original plan were
wrong — read this banner before trusting any arithmetic in this file.**

Written 2026-07-31 at the end of the Phase 1 session. Executed the same day.

## What happened

**Option C was taken: `--context-length` is now 169,000 on both replicas**, and
`--max-total-tokens 190000` was **removed** (the trap flagged below). Verified
end to end: a 165,000-token request is accepted (81.0 s cold prefill) where it
was previously a hard 400; 169,500 is cleanly refused. Pool unchanged at
169,792, memory footprint byte-identical, VRAM moved +18 MiB under the 165K
request.

**Options A and B were not taken.** Reasons, both load-bearing:

### Correction 1 — the free-memory premise below is wrong by ~2x

This document sizes every option against **"8.37 GB free"**. That number is the
boot log's `available_gpu_mem`, which is sampled **before** the CUDA graphs
(1.55 GiB) and the FlashInfer workspaces + CUDA context (~1.2 GiB) are
allocated. **True free VRAM at steady state is ~4.4 GiB.**

Demonstrated directly: a freshly booted replica showed 7,955 MiB free; the
identically configured replica that had served traffic showed 4,469 MiB.

Consequences for what is written below:
- "**262,144 ... additional needed 5.98 GB ... looks affordable**" — it is not.
  5.98 GB against ~4.4 GiB of real headroom, which is *also* the runtime working
  space. **262,144 at BF16 is not reachable on this footprint.** The section's
  own instinct ("expect this to OOM") was right for the wrong reason.
- Option B's "middle position" of 200,000 needs ~2 GB more KV. That is *not*
  "comfortably within the 8.37 GB free" — it spends ~45% of the real margin.
  Still possibly viable, but it must be validated with a concurrency stress test
  at max context, not assumed.

### Correction 2 — fp8 KV works on SM80, but was declined on numerics

Option A's central unknown is **resolved: fp8 works.** `fp8_e4m3` and
`fp8_e5m2` both ran on paged decode (tensor-core path) *and* paged prefill at
this model's exact geometry, with a BF16 control passing alongside. The kernels
are AOT-compiled in the pinned image and SGLang has no SM80 guard.

It was declined anyway, by operator decision, for **reliability**. The reason
found during verification is stronger than the A100 caveat this document
worried about: on an unquantized BF16 checkpoint `k_scale`/`v_scale` are `None`,
so `memory_pool.py set_kv_buffer` does a **direct, uncalibrated cast**. There is
no per-tensor scaling. fp8_e4m3 carries 3 mantissa bits and saturates above
±448; QK-norm bounds K, but **V is unnormalized** and is the open risk. If
revisited, use `fp8_e5m2` as the range-safe fallback and gate on EAGLE accept
length (free, already logged) plus a real long-context quality eval.

### Correction 3 — two assumptions below are now confirmed, not open

- **Mamba/SSM pool does NOT scale with context.** `max_mamba_cache_size: 54`
  slots, sized per-request (3 slots/request), not per-token. The arithmetic
  below is safe on this point.
- **Prefill is linear at this scale.** 154,997 tok → 77.2 s (0.498 ms/tok) and
  165,000 tok → 81.0 s (0.491 ms/tok), agreeing to 1.4%. The ~130 s
  extrapolation for 262,144 stands.
- Also newly noted: `--max-running-requests 4` is what keeps the `req_to_token`
  pool small (`resolve_max_num_reqs` clamps to it). If that flag were ever
  unset, a 262,144 context would allocate a multi-GB index pool.

### Where the remaining headroom actually is

Not in `--mem-fraction-static`. The **Mamba pool is over-provisioned**: 54 slots
cost 5.46 GB, but at `--max-running-requests 4` and 3 slots/request only 12 are
needed for running requests (observed `mamba usage` 0.07–0.22). The surplus
backs the mamba radix cache. Trimming it to ~30 slots would free ~2.4 GB →
~+36,000 KV tokens → a ~205,000 pool **without spending any safety margin**.
The cost is a smaller mamba radix cache, and radix caching is measured at 36× on
this workload — so this must be measured, not assumed. It is the only identified
route to materially more BF16 context.

---

## Original plan follows, unedited. Read the banner above first.

**Status: not started. This is a plan, not a result.**
Written 2026-07-31 at the end of the Phase 1 session, while the numbers were
fresh. Everything below is either measured (marked so) or arithmetic from
measured values. Nothing here has been tested.

---

## The question

The model's native `max_position_embeddings` is **262,144**. We serve
`--context-length 160000`. What would it take to serve more, and is it worth it?

This became live because Phase 1 left the KV pool at **169,792 tokens**, which
is now *larger* than the 160,000 we advertise. Context is limited by the flag
again, not by memory — the first time that has been true in this deployment.

---

## Measured starting point (2026-07-31, post-Phase-1, both replicas identical)

From the r0/r1 boot logs:

```
Load weight end       mem usage = 51.05 GB   (target model)
Load weight end       mem usage =  5.53 GB   (MTP / EAGLE draft head)
KV Cache allocated    169,792 tokens   K 5.18 GB   V 5.18 GB     (target)
KV Cache allocated    169,792 tokens   K 0.32 GB   V 0.32 GB     (draft)
prefill CUDA graph    1.21 GB
verify/draft graphs   ~0.33 GB
max_total_num_tokens = 169792   available_gpu_mem = 8.37 GB
```

Derived KV cost per token (BF16):

```
target KV   (5.18 + 5.18) GB / 169,792 =  61.0 KB/token
draft KV    (0.32 + 0.32) GB / 169,792 =   3.8 KB/token
                                       -----------------
total                                     64.8 KB/token
```

Sanity check against architecture: only the **16 full-attention layers** carry
KV (48 of 64 layers are linear-attention/GDN and carry SSM state instead).
`16 layers × 4 kv_heads × 256 head_dim × 2 bytes × 2 (K and V) = 65,536 B/token`
= 64 KB/token. Matches the measured 61.0 KB within rounding — the architecture
and the boot log agree, so the per-token figure is trustworthy.

**This is the single most important number for this work: ~64.8 KB/token.**

---

## The arithmetic

To hold a full 262,144-token request the pool must be at least 262,144 tokens:

```
262,144 tokens x 64.8 KB = 16.98 GB   of KV
current pool                11.00 GB
                          ----------
additional needed            5.98 GB
```

Against 8.37 GB free, that *looks* affordable. It is tighter than it looks:

```
80 GB card (81,920 MiB usable ~ 79.2 GB)
  weights (target + draft)      56.58 GB
  KV at 262,144 tokens          16.98 GB
  CUDA graphs                    1.54 GB
                                --------
  subtotal                      75.10 GB
  remaining                      ~4.1 GB
```

~4.1 GB of headroom for activations, the Mamba state pool, fragmentation and
chunked-prefill working space, at `--max-running-requests 4`. Phase 1 showed
the working-space reserve scales with that flag, and a documented earlier
stress test hit **96.1% of VRAM** at `mem-fraction-static 0.90` with only *two*
concurrent long prompts. **Expect this to OOM or to silently shrink the pool.**

### The trade nobody can avoid

A 262,144-token request consumes the **entire** pool. While it is resident,
concurrency is 1. The Phase 1 win was worth 1.8–1.9× precisely *because* it
raised concurrency. These two goals are in direct competition:

| pool | concurrent requests | KV budget each |
|---|---|---|
| 169,792 (today) | 4 | ~42K tokens |
| 169,792 | 2 | ~85K tokens |
| 262,144 | 1 | 262K tokens |
| 262,144 | 4 | ~65K tokens |

Note the last row: raising the pool to 262,144 does **not** by itself let you
serve a 262K request *and* keep concurrency. It lets you serve one, alone.

### Cost in time, measured

A 154,997-token request took **77.2 s of cold prefill** (measured this session,
both replicas). Linear extrapolation puts 262,144 tokens at **~130 s** before
the first token. Radix prefix caching does not help a genuinely new document.

Whether a 130-second TTFT is a product you want to offer is a question for the
operator, not the engine.

---

## Option A — FP8 KV cache. Investigate this first.

`--kv-cache-dtype` is available in this build and accepts:

```
auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, bfloat16, nvfp4, fp4_mx_block16, fp4_e2m1
```

**FP8 halves KV cost: 64.8 → ~32.4 KB/token.** That changes everything:

```
262,144 tokens x 32.4 KB =  8.49 GB    (vs 16.98 GB at BF16)
                                        fits in today's 8.37 GB free, roughly
current pool would become  ~339,000 tokens at unchanged memory
```

This is the only option that buys long context **without** surrendering the
concurrency gain — 262,144 pool at FP8 costs about what 131,000 costs today.

**A100 caveat, and it is a real one.** SM80 has **no FP8 tensor cores** —
those arrive on SM89 (Ada) and SM90 (Hopper). On A100, FP8 KV is a *storage*
format: entries are stored as fp8 and converted on read. So you get the memory
saving, but you also pay a conversion cost on every attention read, and there
is no hardware fast path. Whether SGLang's A100 kernels even support fp8 KV
with the FlashInfer attention backend is **unverified** — check the boot log
for a fallback or a refusal.

Prefer `fp8_e4m3` over `fp8_e5m2` for KV: more mantissa, less exponent range,
which suits KV value distributions. Verify empirically, do not assume.

**Quality gate is mandatory here.** Unlike everything in Phase 1, this changes
numerics. The greedy determinism probe in `benchmarks/bench_worker.py` will
*fail* by design — fp8 KV is not bit-identical. Replace the equality check with
a quality comparison on real tasks before considering this shippable.

### Suggested first experiment

One replica only, as in Phase 1:

```
--kv-cache-dtype fp8_e4m3
--context-length 160000        (unchanged at first — isolate the dtype)
```

Gates:
1. Boot log: did fp8 engage, or silently fall back? Read `KV Cache is
   allocated ... dtype:` — it must not say `torch.bfloat16`.
2. `max_total_num_tokens` should roughly double. If it does not, fp8 did not
   take.
3. Decode throughput and cold TTFT vs the BF16 replica.
4. **Quality**, on real prompts. This is the gate that decides it.

Only after fp8 is proven should `--context-length` rise.

---

## Option B — raise `--context-length` at BF16, accept lower concurrency

Straightforward, no numerics risk, but pays for context with throughput.

```
--context-length 262144
--mem-fraction-static ~0.95-0.96      (from 0.92)
--max-running-requests 2              (from 4 — frees working-space reserve)
```

This likely gives back most of the 1.8–1.9× Phase 1 win. Only worth it if
long-context capability matters more than throughput for this deployment.

A middle position is more defensible: **`--context-length 200000`** needs
`200,000 × 64.8 KB = 12.96 GB` of KV, i.e. ~2 GB more than today. That is
comfortably within the 8.37 GB free at `--max-running-requests 4`, and probably
needs only `mem-fraction-static ~0.94`. **Start here** — it is the cheap
fraction of the win.

---

## Option C — do nothing, raise the flag to match the pool

```
--context-length 169000        (just under the 169,792-token pool)
```

Zero risk, zero memory change, +5.6% advertised context, one roll. Strictly
better than today because the pool is already paid for. **Do this regardless of
whether A or B is pursued** — right now we advertise 160,000 while holding
169,792, which is simply wasted.

---

## Recommended order

1. **Option C** — free, one roll, do it first.
2. **Option B at 200,000** — cheap, no numerics risk, measure.
3. **Option A (fp8_e4m3)** — the only path to a genuine 262,144 without
   sacrificing concurrency, but needs a real quality evaluation and A100 fp8
   support is unverified.
4. Full 262,144 at BF16 — only if 1–3 fail and long context outranks throughput.

---

## Things to verify that this document assumes

- [ ] fp8 KV actually engages on SM80 with `--attention-backend flashinfer`
      (read the boot log; do not trust the flag being accepted)
- [ ] whether `--kv-cache-dtype` also applies to the **draft** model's KV
      (0.64 GB today; small, but it affects the arithmetic)
- [ ] whether the Mamba/SSM state pool (48 slots) scales with context length —
      believed **not** to, since SSM state is fixed-size per request rather
      than per token, but unconfirmed. If it does, all arithmetic above is
      optimistic.
- [ ] the 130 s prefill extrapolation for 262,144 tokens — measured only to
      155K (77.2 s); prefill is not necessarily linear at that scale
- [ ] whether `--max-total-tokens 190000` (currently inert) becomes *active*
      once the pool exceeds it. It is documented as a debugging ceiling and has
      been ignored so far because 169,792 < 190,000. At 262,144 it would bind.
      **Raise or remove it before any pool above 190,000.**

That last item is a live trap. It is inert today purely by coincidence.

---

## Method (unchanged from Phase 1 — it worked)

- One replica at a time; the other stays on the known-good config as a live
  control. `docker compose up -d --no-deps qwen36-27b-r<n>`, then
  `docker compose restart qwen36-27b-router`.
- Never `--remove-orphans`.
- Keep the image digest pinned.
- Measure with `tuning/bench/run_ab.sh <r0|r1> <label>` (direct to the worker,
  no router) and compare with `tuning/bench/compare.py`.
- Always take a **control** measurement in the same window. Phase 1's control
  drifted ≤1.6%, which is what made the treatment delta credible.
- Cold TTFT must be measured with text neither replica has seen. This session
  produced two wrong TTFT numbers by ignoring that — see `docs/RESULTS.md`.
