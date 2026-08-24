# SGLang v0.5.18 upgrade — 2026-08-24

**APPLIED 2026-08-24. Both replicas on v0.5.18-cu130. Greedy output
byte-identical to v0.5.17 (8/8). One boot gate moved: `max_total_num_tokens`
171008 -> 169408. Performance A/B was NOT run — deferred by operator request
because live traffic arrived mid-roll.**

Target image:

```
lmsysorg/sglang:v0.5.18-cu130
sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
```

Current pin `sha256:16aba892…` was confirmed to be exactly the
`v0.5.17-cu130` index digest, so this is a clean one-tag step.

Released 2026-08-22 (tag commit `ff4c6e64`, build commit `71de97b2`).
710 PRs / 212 contributors.

## Provenance check (the qwen38-27b-cu129 lesson)

Read from the registry config blob, not from a running container:

| | v0.5.17-cu130 | v0.5.18-cu130 |
|---|---|---|
| `ai.sglang.build.commit` | `29481685…` | `71de97b2…` |
| `ai.sglang.image.tag` | `lmsysorg/sglang:v0.5.17` | `lmsysorg/sglang:v0.5.18` |
| build URL | actions run 31225095628 | actions run 32520522511 |
| CUDA | 13.0.1 | 13.0.3 |
| cuDNN | 9.13.0.50 | 9.14.0.64 |
| NCCL | 2.28.3 | 2.28.3 (unchanged) |
| FLASHINFER_VERSION | 0.6.15.post1 | 0.6.17 |
| amd64 layers / size | 69 / 14.23 GB | 66 / 14.18 GB |

LABEL, env and tag all agree, and the build URL is an official
`sgl-project/sglang` Actions run. No provenance hole. `NVIDIA_REQUIRE_CUDA`
is unchanged (`cuda>=13.0`); host driver is 610.57.04, so no driver work.

## What is actually in it for us — nothing headline

Every highlight in the release notes lands on hardware or model families we
do not run: TP LMHead all-to-all (DeepSeek-V4-Pro B200, and TP>1), FlashInfer
MNNVL pure-allreduce (Blackwell, multi-node), the whole AMD/MI355X block, the
NPU/XPU block, and the diffusion block. `grep -i 'sm80\|A100\|Ampere'` over
the release body returns **zero hits**.

The Qwen3.5 GDN decode win (−73.5% on the QKVZBA split/reshape/cat chain,
#34421) is **HIP-only**. On CUDA the fused-kernel ratio tuple is still
`(1, 2, 4)`:

```python
# models/qwen3_5.py
_GDN_FUSED_QKVZBA_RATIOS = (1, 2, 4, 8) if _use_aiter else (1, 2, 4)
```

Our ratio is `linear_num_value_heads / linear_num_key_heads = 48 / 16 = 3`,
which is in neither tuple. We took the unfused fallback on v0.5.17 and we
still take it on v0.5.18. This is consistent with the July finding that GDN
is only ~2.5% of the profile anyway.

So the case for rolling is: 710 PRs of bug-fix surface, a CUDA/cuDNN refresh,
provenance hygiene, and staying within one release of upstream — same
argument as v0.5.17.

## Breaking changes, audited against our config

Seven of the ten upstream breaking changes are inert here. Each was resolved
by reading the source at the tag, not by reading the changelog.

| Breaking change | Verdict for us | Why |
|---|---|---|
| torch 2.11.0 → **2.13.0**, triton 3.7.1 | **REAL RISK** | see below |
| Kernel caches move under `SGLANG_CACHE_DIR` | inert, + an opportunity | see below |
| torchao removed (`--torchao-config`) | inert | we never passed it |
| DeepEP from wheels | inert | TP1, no EP |
| MoE deferred finalize on by default | **inert** | our model is **dense**. `config.json:text_config` has no expert/moe keys; the flag is only read in `layers/moe/fused_moe_triton/layer.py:388`, which we never construct. Upstream also gates it to the NVFP4 + `flashinfer_trtllm` path |
| Unified-cache out-of-window SWA slot freeing on by default | **inert** | our tree components are FULL + MAMBA. `free_out_of_window_slots` is a `pass` on the base `TreeComponent`; only `swa_component.py` overrides it, and our model has no SWA layers |
| Remote media downloads bounded to 64 MiB | inert *if* OCR stays inline | the new `--media-url-max-file-size-mb 64` bounds **client-supplied HTTP(S) media URLs**, not inline/base64 bodies. The edge request-body cap we removed for OCR payloads is a different path. **Confirm no caller passes `image_url` as a remote URL before rolling**; if one does, set `--media-url-max-file-size-mb 0` |
| DSV4 fused MHC post+pre default on | inert | not our arch |
| torch.compile opt-in for diffusion | inert | not a diffusion server |
| 22 benchmarks removed | inert | our harnesses are local |

### The one real risk: torch 2.11 → 2.13

This is a **two-minor-version jump in a single release**, and it moves the
floor under every number in `tuning/results/`. Along with it:

```
flashinfer_python  0.6.15.post1 -> 0.6.17      (we run --attention-backend flashinfer)
sglang-kernel      0.4.5        -> 0.4.6.post1
nvidia-cutlass-dsl 4.6.0        -> 4.6.2
sgl-deep-gemm      0.1.5.post1  -> 0.1.5.post3
torchao            0.17.0       -> removed
helion             1.4          -> removed
```

Nothing here is A100-targeted, so the expected outcome is neutral — but
"expected neutral" across a torch minor bump is precisely what the A/B gate
exists to falsify. The attention backend we actually run changed underneath
us, which is enough on its own to require the full ladder.

### Memory arithmetic: the scare that is not one

PR **#34996** raises the post-capture decode memory reserve floor from 512 MiB
to **1536 MiB**. On our numbers that would be expensive:

```
KV bytes/token = 16 full-attn layers x 4 KV heads x 256 head_dim x 2 (K,V) x 2 B
               = 65,536 B = 64 KiB
171,008 tokens x 64 KiB = 10.44 GiB   (matches the recorded `kvcache 10.441 GB`)
+1 GiB of reserve       ≈ 16,384 tokens
=> max_total_num_tokens 171,008 -> ~154,600, i.e. BELOW --context-length 169,000
```

That is the same contract break the ReplaySSM experiment hit at 137,600
(`RESULTS.md`). **It does not apply to us.** The whole block is inside:

```python
# server_args.py::_handle_gpu_memory_settings
if self.mem_fraction_static is None:
    if self.post_capture_kv_sizing_planned():
        reserved_mem = 1536          # was 512
```

We pass `--mem-fraction-static 0.92` explicitly, so the heuristic never runs.
Boot gates should be unchanged. **This is still the first thing to read in the
boot log** — if `max_total_num_tokens` moved, this is the suspect.

## Flag and env audit (AST diff of `ServerArgs`, v0.5.17 → v0.5.18)

All 94 flags in our replica command blocks still exist at v0.5.18.
`--tp` resolves as it does today (`tp_size` and its alias list are
byte-identical between the tags).

**Removed upstream — neither is ours:** `--torchao-config`,
`--enable-expert-distribution-metrics` (replaced by
`--expert-balancedness-report-mode`).

**Default changed:** `--hicache-ratio` `2.0 → None`. We do not run HiCache.

**New flags of interest:**

| Flag | Default | Relevance |
|---|---|---|
| `--startup-weight-load-mode` | `serial` | opt-in `overlap`; see Phase 3 |
| `--media-url-max-file-size-mb` | `64` | the OCR question above |
| `--allowed-media-domains` | `[]` | optional SSRF hardening for the OCR path |
| `--language-model-only` | `False` | drops the vision tower entirely and frees its HBM for KV — **only** if OCR moves off this node. Not for now |
| `--image-processor-backend` | `auto` | unchanged behaviour |
| `--mm-global-cache-backend` | `mooncake` | inert, only read when `--enable-mm-global-cache` is set |
| `--enable-tp-lm-head-all-to-all` | `None` | TP1, inert |

**Env-var defaults that flipped** (`environ.py` diff — this is where v0.5.17
hid its trap, so it was diffed explicitly):

| Var | v0.5.17 | v0.5.18 | Verdict |
|---|---|---|---|
| `SGLANG_AUTO_NUMA_BIND` | `False` | `True` | **no behaviour change** — see below |
| `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE` | `False` | `True` | inert (dense model) |
| `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS` | `False` | `True` | inert (no SWA component) |
| `SGLANG_OPT_FUSE_MHC_POST_PRE` | `False` | `True` | inert (DSV4) |
| `SGLANG_DG_CACHE_DIR` | `~/.cache/deep_gemm` | under `SGLANG_CACHE_DIR` | cache move |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | 1024 | 8192 | inert (MoE) |

`SGLANG_AUTO_NUMA_BIND` looks alarming and is not. v0.5.17's
`get_numa_node_if_available()` had **no** such gate — it always queried, which
is why `Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.` is
already a documented known-warning in `docs/OPERATIONS.md`. v0.5.18 adds the
gate and defaults it to `True`, i.e. to the behaviour we already have. The
scheduler call site (`scheduler.py:4989`) is byte-identical between tags. What
we gain is an opt-out knob if it ever misbehaves.

For the record, on this host that binding is a near-no-op either way. NVML
reports NUMA affinity `0-1` for both GPUs, so node **0** is chosen for both;
`numa_bind_to_node` intersects the node's CPUs with the container cpuset:

- r0 `cpuset: 0-27` == node 0 → intersection is the full cpuset, no change.
- r1 `cpuset: 28-47` (node 1) → intersection is **empty** → the failure path
  warns and skips. r1 is never bound.

**Predicted, not measured.** Confirm from the r1 boot log on the first roll.

### `--mm-feature-transport`: v0.5.18 fixes the v0.5.17 trap

v0.5.17 silently auto-resolved this to `cuda_ipc`, which crashed without
`CAP_SYS_PTRACE` and ate ~1 GiB of KV pool. v0.5.18 adds a third value
(`cuda_vmm`) and rewrites the resolution so single-node now resolves to `cpu`:

```python
if self.nnodes == 1:
    requested_transport = "cpu"
elif is_mnnvl_fabric_device() and ...:   # GB200/GB300 only
```

Our explicit `--mm-feature-transport cpu` already matches what the new logic
would pick. **Keep the explicit pin** — it is what made this a non-event, and
the auto-resolution has now changed twice in two releases.

## Two opportunities this release opens (Phase 3, separately gated)

**1. A warm kernel-cache volume.** Every compiled-kernel cache now lives under
one directory, `SGLANG_CACHE_DIR` (default `~/.cache/sglang`, so
`/root/.cache/sglang` in-container). We currently mount only
`${HF_HOME} → /root/.cache/huggingface`, so triton/inductor/flashinfer JIT
output has always been discarded on every recreate. One named volume now
captures all of it:

```yaml
volumes:
  - sglang_kernel_cache:/root/.cache/sglang
```

Note this is *additive*: the AOT `flashinfer-jit-cache` in the image is a
Python package under `dist-packages`, not a `~/.cache` directory, so the
redirect cannot orphan it. Upside is boot time only; measure it.

**2. `--startup-weight-load-mode overlap`.** Stages checkpoint pages from
storage while CUDA graphs capture. Upstream reports 2.38x on Qwen3-32B/H100
(84.8s → 35.6s). Our boot is ~181 s, of which the prefill graph capture alone
is ~59.7 s, so there is real overlap to win. Untested against a hybrid-SSM
model with an EAGLE draft worker — the manager is draft-aware
(`is_draft_worker` is threaded through `StartupWeightLoadManager`) but that is
not the same as verified. **Do not put this in the baseline roll.**

## The plan

Follow the v0.5.17 shape: change the engine only, one replica at a time, gate
on the boot log and on a direct-to-worker A/B while the two replicas are
deliberately on different builds.

### Phase 0 — before touching anything

1. Answer the OCR question: does any caller send `image_url` as a **remote
   HTTP(S) URL** rather than inline base64? If yes, add
   `--media-url-max-file-size-mb 0` to both command blocks in the same edit.
2. Pull the image on the host (does not disturb the running stack):
   ```bash
   docker pull lmsysorg/sglang:v0.5.18-cu130
   docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:v0.5.18-cu130
   # must print sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
   ```
   1.1 TB free on `/`, 14.2 GB image — no space concern.
3. Capture the v0.5.17 baseline from **both** replicas while both are still on
   the old build:
   ```bash
   ./benchmarks/run_worker.sh r0 v0517_pre_518
   ./benchmarks/run_worker.sh r1 v0517_pre_518
   ```
   and save `/get_server_info` from both.

### Phase 1 — roll r1 only, and hold there

Edit the `x-sglang-image` anchor to the new digest, keep the comment block's
bump instructions, then:

```bash
./deploy/roll-replica.sh r1
```

r0 stays on v0.5.17 and keeps serving. This is the A/B window.

**Boot gates on r1 — all four must hold:**

| Check | Expected |
|---|---|
| `max_total_num_tokens` | **171008** |
| decode CUDA-graph `bs` | `[1, 2, 3, 4]` |
| `max_mamba_cache_size` | **43** |
| boot to healthy | ~181 s (first boot may be longer: caches recompile once) |

A move in `max_total_num_tokens` means the mem-fraction path changed — stop
and check #34996 before serving.

Then diff runtime server args between the builds, the check v0.5.17 taught us:

```bash
docker exec qwen36-27b-r0 sh -c 'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
  http://localhost:8001/get_server_info' > /tmp/si_r0_517.json
docker exec qwen36-27b-r1 sh -c 'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" \
  http://localhost:8002/get_server_info' > /tmp/si_r1_518.json
diff <(jq -S .server_args /tmp/si_r0_517.json) <(jq -S .server_args /tmp/si_r1_518.json)
```

Expected diff: `port` only, plus any genuinely new keys. Anything else
resolved differently at startup — investigate before proceeding.
Specifically confirm `mm_feature_transport == "cpu"` on r1.

Also grep the r1 boot log for the NUMA prediction above.

### Phase 2 — the A/B gate (r1 v0.5.18 vs r0 v0.5.17)

Hit workers **directly** on :8001/:8002, never through the router. Flush both
(`POST /flush_cache`) and measure in both orderings.

| Gate | Pass condition |
|---|---|
| `byte_identity.py` greedy, 8 prompts x 256 tok, `ignore_eos`, temp 0 | **8/8 identical** — same weights, engine-only change, so this gate is valid here |
| `spec_accept_length` converged | 5.0-ish, unchanged |
| `worker_ladder.py` c=1..12 | mean within the **1.5%** noise floor, sign alternating |
| TTFT, 42K prompt | ~22.7 s |
| single-stream decode | ~68.5 tok/s |

Remember the rig's same-build noise floor is ~1.5% and we still cannot lock
clocks. Anything under that is unresolvable, not a result.

### Phase 3 — only after Phase 2 passes; one variable at a time

Both are boot-time-only and neither should be bundled into the engine roll:

1. Add the `sglang_kernel_cache` volume to both replicas. Gate: boot-to-healthy
   on the *second* boot, and the four boot gates unchanged.
2. Try `--startup-weight-load-mode overlap` on r1 alone. Gate: same four boot
   gates plus byte-identity — staged weights must commit correctly, and the
   EAGLE draft worker is the part to distrust.

### Phase 4 — converge and record

`./deploy/roll-replica.sh r0`, re-verify the boot gates, confirm
`docker compose config` shows the two command blocks differing only by port,
and write the measurements back into this file.

### Rollback

Restore the `x-sglang-image` anchor to
`sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155`
and roll the affected replica. Because only r1 moves in Phase 1, rollback is
one `roll-replica.sh r1` and traffic never leaves a healthy worker.

## Execution log

### Phase 0 — baseline on v0.5.17 (both replicas), 2026-08-24

`--media-url-max-file-size-mb` question resolved from the repo, not by asking:
`Caddyfile:17` records that **OCR traffic sends base64 images inline in the
chat payload**, corroborated by `OBSERVABILITY.md:48`. The new 64 MiB bound is
on *remote* client-supplied media URLs only, so it is inert. No flag added.

| | r0 | r1 |
|---|---|---|
| `decode_tok_s_mean` | 70.60 | 70.32 |
| long TTFT (s) | 12.426 | 12.398 |
| long-context decode (tok/s) | 19.02 | 18.77 |
| greedy probe | `2, 3, 5, 7, …37`, 119 tok, 208-char reasoning | identical |

r0/r1 differ by 0.4% on decode — inside the ~1.5% same-build noise floor.

**Byte-identity control, both replicas on v0.5.17: 8/8 identical.** This
matters: it establishes that the gate reads PASS when nothing differs, so a
Phase 2 failure cannot be blamed on the harness.

v0.5.17 boot-log reference (r0, 2026-08-15):

```
Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.
Mamba Cache is allocated. max_mamba_cache_size: 43, conv_state 0.12GB,
  ssm_state 3.09GB, intermediate_ssm_state_cache 2.11GB, conv_window 0.04GB
KV Cache is allocated. #tokens: 171008, K 5.22 GB, V 5.22 GB     <- target
KV Cache is allocated. #tokens: 171008, K 0.33 GB, V 0.33 GB     <- EAGLE draft
max_total_num_tokens=171008, context_len=169000, available_gpu_mem=8.27 GB
Capture target prefill  61.11 s   verify 8.80 s
Capture draft decode     4.02 s   draft extend 1.74 s
```

Target KV confirms the arithmetic exactly: 10.44 GB = 171,008 x 64 KiB.

The NUMA line is the important one — it is present **on v0.5.17**, which is
the direct evidence that `SGLANG_AUTO_NUMA_BIND: False -> True` restores
existing behaviour rather than introducing binding.

Boot timeline: weight load ~46 s (start -> Mamba alloc), CUDA graph capture
~76 s. That is what makes `--startup-weight-load-mode overlap` worth trying:
there is ~46 s of load that could hide behind ~76 s of capture.

### Predicted `/get_server_info` diff (v0.5.17 -> v0.5.18)

Recorded before rolling so the Phase 1 check is a real prediction:

- `port` (8001 vs 8002)
- **gone:** `torchao_config=''`, `enable_expert_distribution_metrics=False`
- **changed:** `hicache_ratio` `2.0` -> `None`
- **new:** the 16 flags listed above (`startup_weight_load_mode='serial'`,
  `media_url_max_file_size_mb=64`, `image_processor_backend='auto'`, …)
- `mm_feature_transport` must still read `'cpu'`

Anything else is a silent resolution change and a stop condition.

### Phase 1 — r1 rolled to v0.5.18, r0 held on v0.5.17 (the A/B window)

Boot was clean: zero traceback/CUDA-error lines. Predictions from the source
review all held.

| gate | v0.5.17 | v0.5.18 | verdict |
|---|---|---|---|
| `max_mamba_cache_size` | 43 (conv 0.12 / ssm 3.09 / inter 2.11 / win 0.04 GB) | **identical** | PASS |
| decode CUDA-graph `bs` | `[1,2,3,4]` | `[1,2,3,4]` | PASS |
| `max_total_num_tokens` | 171008 | **169408** | **DEVIATION** |
| weights | 51.047 GB | 51.047 GB | PASS |
| `mm_feature_transport` | `cpu` | `cpu` | PASS — trap did not recur |
| boot to healthy | ~181 s | ~181 s (cold cache volume) | PASS |
| target prefill capture | 61.11 s | 59.35 s | PASS |

The `/get_server_info` diff matched the pre-registered prediction **exactly**:
16 new keys, 2 gone (`torchao_config`, `enable_expert_distribution_metrics`),
`version` 0.5.17 -> 0.5.18, plus one new resolution
(`_tp_lm_head_all_to_all_default` -> False, inert at TP1). Nothing resolved
differently behind our back.

`Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.` appears on
**both** builds, confirming the `SGLANG_AUTO_NUMA_BIND` prediction: the flip
False -> True restored existing behaviour rather than introducing binding.

### The `max_total_num_tokens` deviation — 171008 -> 169408

−1,600 tokens = −100 MiB of KV (`kvcache` 10.441 -> 10.344 GB). It is **not**
PR #34996: that is behind `if mem_fraction_static is None` and we pin 0.92.
Weights and the Mamba pool are byte-identical, and the KV pool is sized
*before* graph capture, so the growth is in the pre-KV runtime workspace —
torch 2.13 / flashinfer 0.6.17 / cuDNN 9.14. The 100 MiB was left **free**,
not leaked: `startup_available` rose 8.271 -> 8.343 GB.

Both replicas report 169408 identically, so there is no asymmetry between them.
169,408 still clears `--context-length 169000`, but the margin narrowed from
792 to **408 tokens**. Tracked in `docs/OPERATIONS.md`.

### Phase 2 — byte-identity, and a methodology error worth recording

First run scored **0/8**. That was **not** the engine — it was a bad
measurement, and the diagnosis is the useful part:

| comparison | result |
|---|---|
| r0 v0.5.17 vs r1 v0.5.18, **r0 not flushed** | 0/8 |
| r0 v0.5.17 vs **itself**, 40 min apart, not flushed | 0/8 |
| r0 v0.5.17 vs **itself**, back-to-back, not flushed | **1/8** |
| r1: v0.5.17 (pre-roll) vs v0.5.18 (post-roll), same GPU | 7/8 |
| r1 v0.5.18 vs **itself**, back-to-back | 7/8 |
| **r0 v0.5.17 vs itself, BOTH FLUSHED** | **8/8** |
| **r0 v0.5.17 vs r1 v0.5.18, BOTH FLUSHED** | **8/8 — PASS** |

The untouched old build scored 1/8 against itself. With a warm session-radix +
Mamba `extra_buffer` cache the greedy probe is simply not reproducible, so the
gate measures cache state unless both replicas are flushed first.

The trap: `POST /flush_cache` answers `Flush cache failed.` whenever the
replica has running or waiting requests — which a replica serving production
always does. The v0.5.17 upgrade flushed both replicas and got 8/8; this run
skipped a *refused* flush on r0 and got 0/8. Retry until it answers
`Cache flushed.` Now recorded in `docs/OPERATIONS.md`.

**Conclusion: v0.5.18 is byte-identical to v0.5.17 on greedy output.**

### Phase 3/4 — what was and was not done

Live traffic arrived mid-roll and the operator called off benchmarking, so:

- **Done:** both replicas converged on v0.5.18-cu130 + the per-replica warm
  kernel-cache volume. r0 boot gates match r1 exactly
  (169408 / 8.34 GB / 43 mamba slots).
- **One surviving data point, not a gate.** The r1 leg of the benchmark run
  completed before the interrupt: `decode_tok_s_mean` **70.7** on v0.5.18 vs
  **70.32** in the Phase 0 v0.5.17 baseline on the same GPU with the same
  harness — **+0.54%, inside the ~1.5% noise floor**, i.e. neutral. Saved as
  `benchmarks/results/worker_v0518_r1.json`. Its long-prompt TTFT reads 13.196 s
  vs 12.398 s, but that leg ran against a just-flushed cache under live traffic
  and is not a controlled comparison. The r0 leg was interrupted mid-run and its
  output was discarded, not committed — its 41.63 tok/s is an artifact of the
  interruption, not a measurement.
- **NOT done — performance A/B.** No ladder, no accept-length convergence, no
  TTFT/decode comparison against the v0.5.17 baseline in Phase 0. The claim
  "performance-neutral" is therefore **unproven** for this upgrade. The Phase 0
  baselines are saved (`benchmarks/results/worker_v0517_pre518*.json`) and the
  A/B window is gone — re-measuring now compares v0.5.18 against a recorded
  number, not against a live control.
- **NOT applied — the two optional flags**, deliberately, because neither is
  needed to run v0.5.18 and both are unmeasured here:
  - `--startup-weight-load-mode overlap`: changes the weight-load path. Upside
    is smaller than first estimated — `startup_time.load_weight` is only
    ~16.5 s with a warm host page cache (the 163 s figure in r0's old boot was
    a cold-cache disk read). It pays off mainly on a cold boot after a host
    restart, where it can hide load behind ~59 s of prefill capture.
  - `--mm-preprocess-cache-size-mb 512`: the content-addressed multimodal
    preprocessing cache is **off by default** for our processor
    (`base_processor.py: auto_mm_preprocess_cache_size_mb = 0`; only Kimi-K3
    sets 256). Host RAM only, no GPU cost. Helps only on repeated identical
    media — plausible for OCR, unmeasured here.
### Phase 3 — applied 2026-08-24, one flag survived

Operator asked for the staged flags to go in. Result:

**`--startup-weight-load-mode overlap` — REJECTED, hard boot failure.**

```
Scheduler hit an exception:
  startup_weight_load = StartupWeightLoadManager.create_from_server_args(...)
  File .../model_runner_components/startup_weight_load.py, line 307, in create
    raise ValueError(
ValueError: --startup-weight-load-mode=overlap is not supported:
            speculative decoding is not supported
```

It is incompatible with speculative decoding, and we run EAGLE — so it can
**never** be enabled on this deployment. The source review had seen
`is_draft_worker` threaded through `StartupWeightLoadManager` and read that as
draft-awareness; it is not the same as support, and the explicit gate is one
frame further in. That is exactly why it was held back from the baseline roll
rather than bundled.

Blast radius was one replica: `roll-replica.sh` had already deregistered r1,
r0 served alone throughout, the flag was removed from both blocks and r1 came
back healthy on the next roll (169408 / 8.35 GB). r0 was then rolled to match.

**`--mm-preprocess-cache-size-mb 512` — APPLIED**, resolves to 512 on both
replicas. Benefit remains unmeasured on this node.

Final converged state, both replicas identical:

```
v0.5.18  max_total_num_tokens=169408  mm_preprocess_cache_size_mb=512
         startup_weight_load_mode=serial  mm_feature_transport=cpu
```

### The int8 question

`--speculative-draft-kv-cache-dtype` offers only
`auto / fp8_e5m2 / fp8_e4m3 / bf16` — **there is no int8 option**, so an int8
draft KV pool is not available regardless of hardware. (fp8 KV cache is a
storage format that dequantizes in-kernel, so it is not itself blocked by
SM80's lack of FP8 tensor cores — but at a 0.66 GB draft pool it is not worth
the accept-length risk either way.)

The int8 lever that *does* fit SM80 and this hybrid architecture is
**`--enable-int8-mamba-checkpoint`**: int8 radix-cached linear-attn states for
~2x cached-prefix capacity at fixed memory. Compatible with our config — it
rejects `--enable-hierarchical-cache` (we are False) and any
`--radix-cache-backend` (we are unset). It is lossy, so it changes output on
cache hits and needs its own gated roll with a flushed-cache byte-identity
comparison. Not applied. Not new in v0.5.18.
- **NOT applied — `--speculative-draft-kv-cache-dtype fp8_e4m3`.** Measured
  from the boot log, the EAGLE draft KV pool is only **0.66 GB** (K 0.33 +
  V 0.33). fp8 would save ~0.33 GB ≈ 5,200 tokens, and it perturbs draft
  numerics — i.e. accept length, the thing the EAGLE 5/6 win rests on. Bad
  trade; skipped on purpose.

### Open item — the router is still on the v0.5.17 image

`roll-replica.sh` only touches replicas, so `qwen36-27b-router` is still
running `16aba892…` while the compose anchor now points at `9e148f5a…`. There
were **no router changes in v0.5.18** (the single "router" hit in the release
notes is a Kimi-K3 MoE routing GEMM), so this is safe to leave. But it is
latent: the next `docker compose up -d` will recreate the router onto v0.5.18,
and per `docs/OPERATIONS.md` a router restart drops in-flight requests on both
replicas. Do it deliberately, in a quiet window.

## Sources

Release notes, Docker Hub tag list and registry config blobs, plus the
v0.5.17/v0.5.18 source trees at tag (AST diff of `ServerArgs`, `environ.py`
diff, `models/qwen3_5*.py` diff, `mem_cache/unified_cache/components/`,
`utils/numa_utils.py`, `managers/scheduler.py`, `docker/Dockerfile`).
