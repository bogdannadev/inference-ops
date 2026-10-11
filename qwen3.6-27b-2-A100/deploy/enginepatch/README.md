# Engine patch: `sitecustomize.py`

Three fixes applied to the running engine without rebuilding the image or
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

## Remove it when upstream lands

All three patches are upstream bugs, not local policy. Drop this directory and the
two compose lines once the pinned image returns 400 for a malformed multimodal
prompt, clients stop sending `skip_special_tokens: false`, and upstream fixes the
full-chunk fallback in `available_chunk_tokens` (PR #30743 was closed unmerged).
`verify.py` failing after an upgrade is the signal that a patch point moved — the
wrapper degrades to a no-op rather than crashing, so check it deliberately on
every version bump.
