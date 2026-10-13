# Engine patch: `sitecustomize.py`

> Live copy, mounted read-only by both replicas. Develop in `../enginepatch-next/`
> and copy here only right before a roll (any restart imports this directory).
> Patch 5 live since 2026-10-14.

Five fixes applied to the running engine without rebuilding the image or
forking SGLang. 1 and 2 are for the image-token poisoning incident (read
`../../docs/INCIDENT-2026-09-20-IMAGE-TOKEN-CASCADE.md` §3a first); 3 is for the
2026-10-09 r0 crash (`../../tuning/docs/TRIAL_2026-10-09_PATCH3_KVRANGE.md`).

1. **Forces `skip_special_tokens: true`** on every chat completion. Clients send
   `false` whenever they ask for reasoning (862/862 requests measured), which
   makes the engine render a vision control token the model emitted back into the
   reply *text*. The client stores that text in its history and resends it; the
   tokenizer turns it back into the real token; the prompt then claims an image
   nobody attached and **every turn 500s forever**.
2. **Turns a placeholder/payload mismatch into a 400** instead of a 500, so the
   router never retries it and no circuit breaker ever counts it.
3. **Never hands a continuing prefill chunk more KV than physically exists.**
   Upstream `PrefillBudget.available_chunk_tokens` falls back to a full chunk
   once the budget is used up; with too little free KV the allocator raises
   "Prefill out of memory" and the scheduler dies. The patch clamps the chunk to
   free + evictable KV, or parks it for the round when not one page fits.
4. **Runs big prefill chunks of FP8 layers on cuBLAS bf16** instead of Marlin
   (`fp8_prefill_dequant.py`). On A100 the FP8 checkpoint is weight-only
   Marlin: right for decode (bandwidth-bound), +38% slower for an 8K cold
   prefill (compute-bound). For a chunk of at least 2049 tokens, a Triton kernel
   dequantizes the layer from Marlin's tiled layout into a bf16 scratch slab
   (128 MB cap) and cuBLAS does the GEMM. Decode, verify and every
   graph-captured prefill size (<= 2048) stay on Marlin. The weight is
   bit-identical to what Marlin multiplies. `SGLANG_PATCH_FP8_DEQUANT=0`
   turns it off; `_MIN_M` and `_SCRATCH` tune it.
5. **Per-key context window and reasoning effort** (`keypolicy.py`, 2026-10-14).
   The engines serve the model's full 262,144 window; each request is held to
   its key's window from `/opt/keypolicy/policy.json`, which quota-bot writes
   (`/policy` → context, effort), with the same two checks and error texts as
   upstream. Everyone without an entry, every unlabelled request and every
   request while the file is missing gets `KEYPOLICY_DEFAULT_CONTEXT` (169000).
   A request without `max_tokens` gets `window - prompt - 1`, as the scheduler
   would at that `--context-length`. The same patch fixes effort: in v0.5.21
   `--default-chat-template-kwargs`' `xhigh` overwrote every client
   `reasoning_effort` (only `none` got through, measured on r1); the request's
   effort, else the key's default, is now pinned into `chat_template_kwargs`
   before the merge. `high`/`max` map to `xhigh` and `minimal` to `low`, because
   the Qwen3.8 template accepts only `xhigh`, `medium` and `low`.

## Why a `sitecustomize.py` and not a plugin or a patched image

SGLang discovers plugins through setuptools `entry_points`, so a `HookRegistry`
plugin has to be a pip-installed wheel inside the image. `sitecustomize` needs
neither: `site` imports it at interpreter startup for anything on `PYTHONPATH`,
so a read-only bind mount plus one environment variable is the whole install.
Nothing in the image changes, and reverting is removing the two lines.

The patches are applied from a wrapper around `__import__` rather than by
importing the targets here, so no heavy import (torch) is pulled in early. The
wrapper removes itself once every patch is in.

## Why not at the gateway

That was tried first and **does not work**: a Higress wasm plugin must buffer the
whole request body to rewrite it, the listener runs Istio's default
`per_connection_buffer_limit_bytes: 32768`, and every poisoned body is
404,877–545,051 B. Over the limit the body is forwarded unmodified with no error
and no log line. The engine is the first layer that holds the whole request.

## Install

Both are already wired into `../../docker-compose.yml` (`x-replica-common`):

```yaml
environment:
  - PYTHONPATH=/opt/enginepatch
volumes:
  - ./deploy/enginepatch:/opt/enginepatch:ro
```

It takes effect only on container recreation, so it needs a replica roll —
**one replica at a time**, `../roll-replica.sh`, never a stack-wide `up -d`.
Model load is `start_period` 2400 s, so budget ~40 min per replica with the
other one serving throughout.

Confirm it loaded, per replica:

```bash
docker logs qwen36-27b-r0 2>&1 | grep sitecustomize
#   [sitecustomize] forced skip_special_tokens=True on ChatCompletionRequest
#   [sitecustomize] placeholder/payload mismatch now raises ValueError (400)
#   [sitecustomize] continuing prefill chunks clamped to physical KV
#   [sitecustomize] fp8 Marlin layers use dequant+cuBLAS for M >= 2049
#   [sitecustomize] reasoning effort: request, then key default, then server default
#   [sitecustomize] per-key context from /opt/keypolicy/policy.json, default 169000
#   [keypolicy] loaded: default context 169000, N consumer entries
# on the first big prefill chunk:
#   [sitecustomize] fp8 prefill dequant path active (first M=4096, ...)
# when patch 3 acts, at most once a minute:
#   [sitecustomize] KV budget exhausted on a continuing chunk: gave N of 4096 ...
```

Then confirm it is actually taking effect on real traffic — the gateway attempt
looked correct and was not:

```bash
docker logs qwen36-27b-r0 --since <local-time> 2>&1 | grep "Receive: obj=" \
  | grep -o "'skip_special_tokens': [A-Za-z]*" | sort | uniq -c
# every consumer request must now read True
```

## Test

`verify.py` runs against the pinned image with no GPU and asserts all of it:
the field is forced, a placeholder mismatch maps to `ValueError`/400, a GPU
OOM still maps to `RuntimeError`/500 so it keeps alerting, and the 2026-10-09
budget state yields a clamped chunk (or a park) instead of 4096.

```bash
IMG=$(docker inspect qwen36-27b-r1 --format '{{.Image}}')
docker run --rm --entrypoint python3 -e PYTHONPATH=/opt/enginepatch \
  -v "$PWD/deploy/enginepatch:/opt/enginepatch:ro" "$IMG" /opt/enginepatch/verify.py
```

Patch 5: `verify_keypolicy.py` (CPU, no GPU; its docstring has the command)
runs the real `TokenizerManager._validate_one_request` with the patch on top,
and renders the real chat template to show which effort the model sees
(prompt of "hi": xhigh 53 tokens, low 41, medium 11, thinking off 13). On a
live replica the same token counts are the check: a `max_tokens: 1` request
with `reasoning_effort: low` must report `prompt_tokens` 41, not 53.

### Patch 4

CPU, no GPU (Triton interpreter; uses upstream's own `prepare_fp8_layer_for_marlin`
with the CUDA repack replaced by a transcription of `gptq_marlin_repack.cuh`):

```bash
docker run --rm --network none --entrypoint python3 -e TRITON_INTERPRET=1 \
  -e PYTHONPATH=/opt/enginepatch -v "$PWD/deploy/enginepatch-next:/opt/enginepatch:ro" \
  "$IMG" /opt/enginepatch/verify_fp8_dequant.py
```

GPU, **drained replica only** (checks the layout against the real repack and
Marlin GEMM, then times Marlin vs dequant+cuBLAS for every Qwen3.8-27B linear):

```bash
docker run --rm --gpus '"device=1"' --entrypoint python3 -e PYTHONPATH=/opt/enginepatch \
  -v "$PWD/deploy/enginepatch-next:/opt/enginepatch:ro" "$IMG" /opt/enginepatch/bench_fp8_dequant.py
```

Gates before it goes live: the bench's per-forward total must show a saving;
mem_stress at 128K must pass with the 128 MB scratch; cold prefill 8K/32K/64K/128K
must improve; spec_eval and the kvq suite must not move (the weight is
identical, only accumulation order differs).

Two traps found while building it, both handled in the code:
- Triton's CPU interpreter truncates fp32 -> bf16 even with
  `fp_downcast_rounding="rtne"`, so the kernel rounds to nearest-even by hand.
- Importing the patch module from inside the import hook re-enters the hook;
  a nested failure would mark the patch done for good. `_importing` guards it.

## Remove it when upstream lands

Patches 1-3 are upstream bugs, not local policy; patch 4 is a performance
policy for SM80 and goes when the node moves to a GPU with FP8 tensor cores. Drop this directory and the
two compose lines once the pinned image returns 400 for a malformed multimodal
prompt, clients stop sending `skip_special_tokens: false`, and upstream fixes the
full-chunk fallback in `available_chunk_tokens` (PR #30743 was closed unmerged).
`verify.py` failing after an upgrade is the signal that a patch point moved — the
wrapper degrades to a no-op rather than crashing, so check it deliberately on
every version bump.
