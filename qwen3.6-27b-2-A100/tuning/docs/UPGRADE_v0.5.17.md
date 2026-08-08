# SGLang v0.5.17 upgrade — 2026-08-08

**Outcome: output-identical, performance-neutral. Upgrade kept for provenance,
bug-fix surface, and the features it unlocks — not for speed.**

From `lmsysorg/sglang@sha256:647d7bb0…` (`0.0.0.dev1+ga358374ae`, a *dev* build
= v0.5.16 + 2 days of main, commit 2026-07-27) to
`lmsysorg/sglang@sha256:16aba892…` (`v0.5.17-cu130`, rev 2948168). ~582 PRs.

## Why we did it

PR **#32219** "Cut spec-v2 host-seam overhead in hybrid-linear MTP decode",
merged 2026-07-28 03:38 UTC — 26 h after the pinned commit. Under spec-v2
overlap scheduling each decode step is `draft` x5 + `target_verify` +
`draft_extend`; the eager gaps between those graphs (accept sampling, cache-slot
assignment, attention-metadata rebuild) become GPU idle whenever host loop time
exceeds GPU step time. Three of its four fixes applied to us (the fourth is
KDA-only). Item 2 was the most promising: `HybridLinearAttnBackend` was
allocating a fresh `custom_mask` **every step, sized bs x max_context_len**, and
our `max_context_len` is 169,000 — near the top of what anyone runs.

## What we measured

Two identical replicas on separate GPUs, r0 on v0.5.17 and r1 on the old pin,
so the engine was the only variable. Both were flushed (`POST /flush_cache`)
and measured in both orderings. All measurements hit the workers **directly**
on :8001/:8002, never through the router.

| gate | result |
|---|---|
| greedy byte-identity, 8 prompts x 256 tok, `ignore_eos`, temp 0 | **8/8 identical** |
| `spec_accept_length` (converged) | 5.025 both — unchanged |
| TTFT, long prompt | 12.355 s vs 12.410 s — identical |
| long-context decode | 19.27 tok/s both — identical |
| short-prompt decode | 74.48 vs 75.17 tok/s (−0.92%) |
| concurrency ladder c=1..12, chunk latency | mean **−0.21%**, all rungs within ±2%, sign alternates |

### The −0.92% is not a regression

`worker_r0_before_gdn.json` / `worker_r1_before_gdn.json` (2026-07-28, taken a
minute apart with **both replicas on the same build**) measured r0 as **+1.51%**
faster than r1. That is this rig's same-build noise floor, and it is larger than
today's delta and points the other way. Per the July campaign, any sub-5% A/B
needs clock locking (`sudo nvidia-smi -lgc`), which we still cannot do. Treat
anything under ~1.5% here as unresolvable.

### Why #32219 does nothing on A100 (and does a lot on B200)

Not a contradiction — it is the same model applied to different numbers.

A decode iteration here is 5 draft steps + 1 target verify + 1 draft_extend.
The target verify streams the full ~54 GB of weights; A100 80GB PCIe peaks at
~1935 GB/s and we measure ~1086 GB/s effective (56% of peak), so that one
forward is **~50 ms**. Add ~5 draft steps on the 5.53 GB draft model and the
whole iteration is ~65–80 ms — consistent with the observed 75 tok/s ÷ accept
length 5.025 = 14.9 iterations/s.

The host-seam work #32219 deletes is on the order of a 169 KB allocation plus
~5 fused kernel dispatches: a few hundred microseconds. Against a 65 ms GPU
step that is **~0.3%**, and `event_loop_overlap` already hides all of it — the
host stays off the critical path exactly when host loop time < GPU step time,
and here it wins by ~100x.

On B200 the same forward is bandwidth-bound at ~8 TB/s (≈4x faster), and
upstream ran TP4 on top of that, cutting the step to low single-digit
milliseconds. The *same* few hundred microseconds is then a double-digit
percentage of the step. Their headline "67% -> 95% GPU utilisation" is the whole
campaign on B200/TP4/KDA, including the KDA D2H sync fix that does not apply to
GDN at all.

**Rule of thumb this establishes for this node:** host-side scheduling
optimisations cannot pay off while a single target forward costs ~50 ms. Only
work that reduces bytes moved per forward (quantisation, smaller draft model,
better accept length) moves decode here. This is the same conclusion the July
profile reached from the other direction — GEMMs 87.7% of decode GPU time.

## What broke, and the fix

v0.5.17 changed how `--mm-feature-transport` **auto-resolves**: v0.5.16 chose
`cpu`, v0.5.17 chooses `cuda_ipc` for single-node CUDA. Two failures:

1. `cuda_ipc` moves feature tensors between the mm-processor process and the
   scheduler with `pidfd_getfd(2)`, which requires CAP_SYS_PTRACE. We grant only
   SYS_NICE, so the scheduler died on its own warmup request:
   `RuntimeError: pidfd_getfd: Operation not permitted`.
2. It reserves `SGLANG_MM_FEATURE_CACHE_MB` (1024 MiB default) of VRAM. That
   alone moved `max_total_num_tokens` 171,008 -> 152,448 and
   `max_mamba_cache_size` 43 -> 39 — pushing the pool **below**
   `--context-length 169000` and re-arming the July "passes the length check,
   then fails to allocate" bug.

This bites despite a pure-text workload, because the model is
`Qwen3_5ForConditionalGeneration` and SGLang auto-enables the multimodal path.

Fixed by pinning `--mm-feature-transport cpu` on both replicas. Every memory
number then returned to its old value exactly (171,008 tokens, 43 mamba slots,
K/V 5.22 GB, draft KV 0.33 GB, `avail mem` 5.68 GB, `available_gpu_mem`
8.27 GB). We chose this over `cap_add: SYS_PTRACE`, which would buy a
capability for a code path we never execute.

**Generalised lesson:** a `--help` flag-and-default diff cannot catch this. We
diffed all 584 flags (zero removed, all choice-sets purely additive) and it
looked completely clean, because the default is *computed at startup*. The only
view that shows resolved defaults is the runtime `server_args` from
`/get_server_info`. Diff that between builds on every upgrade.

## Also verified

- `--enable-fused-qk-norm-rope` survives the `jit_kernel` -> `sglang.kernels`
  reorg and the helion 0.2.6 -> 1.4 major bump (byte-identity covers it).
- FlashInfer 0.6.15.post1 workspace sizing fits the unchanged budget (risk #1
  cleared — the pool shrink was entirely the mm IPC reservation, not FlashInfer).
- `enable_session_radix_cache` existed as a flag in the old build but the
  implementation did **not** (zero grep matches). v0.5.17 ships
  `mem_cache/unified_cache/session_ref_tracker.py`, `unified_radix_cache.py`,
  and the `/close_session` route.
- `--enable-gdn-replayssm-spec` is gone, renamed `--enable-linear-replayssm-spec`
  (#33102 generalised GDN -> linear). Upstream also made it compatible with
  `extra_buffer` (#32692/#33102), removing the reason it was rejected on
  2026-07-28 — but it is still declined, now by choice: #32219's fused replay
  path is gated on replayssm being OFF. Never test the two together.

## Measurement artifact worth knowing

`benchmarks/worker_ladder.py` counts SSE events, not tokens. The server batches
~3.3 tokens per event (tracking the EAGLE accept length), so its latency figures
are ~3x the true per-token value. Streamed and non-streamed requests finish in
identical wall time (3.05 s vs 3.04 s for 200 tokens) — there is **no** streaming
penalty, only a counting difference. Comparisons between two workers measured
with that script are still valid.

## Artifacts

```
benchmarks/byte_identity.py                  the gate (also --compare mode)
benchmarks/worker_ladder.py                  per-worker ladder (router-free)
benchmarks/results/worker_{v0517_new,dev_old}{,_flushed}.json
tuning/results/ladder_{r0_v0517_new,r1_dev_old}.json
```
