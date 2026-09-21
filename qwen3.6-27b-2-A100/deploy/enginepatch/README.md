# Engine patch: `sitecustomize.py`

Two fixes for the image-token poisoning incident, applied to the running engine
without rebuilding the image or forking SGLang. Read
`../../docs/INCIDENT-2026-09-20-IMAGE-TOKEN-CASCADE.md` §3a first.

1. **Forces `skip_special_tokens: true`** on every chat completion. Clients send
   `false` whenever they ask for reasoning (862/862 requests measured), which
   makes the engine render a vision control token the model emitted back into the
   reply *text*. The client stores that text in its history and resends it; the
   tokenizer turns it back into the real token; the prompt then claims an image
   nobody attached and **every turn 500s forever**.
2. **Turns a placeholder/payload mismatch into a 400** instead of a 500, so the
   router never retries it and no circuit breaker ever counts it.

## Why a `sitecustomize.py` and not a plugin or a patched image

SGLang discovers plugins through setuptools `entry_points`, so a `HookRegistry`
plugin has to be a pip-installed wheel inside the image. `sitecustomize` needs
neither: `site` imports it at interpreter startup for anything on `PYTHONPATH`,
so a read-only bind mount plus one environment variable is the whole install.
Nothing in the image changes, and reverting is removing the two lines.

The patches are applied from a wrapper around `__import__` rather than by
importing the targets here, so no heavy import (torch) is pulled in early. The
wrapper removes itself once both patches are in.

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
the field is forced, a placeholder mismatch maps to `ValueError`/400, and a GPU
OOM still maps to `RuntimeError`/500 so it keeps alerting.

```bash
IMG=$(docker inspect qwen36-27b-r1 --format '{{.Image}}')
docker run --rm --entrypoint python3 -e PYTHONPATH=/opt/enginepatch \
  -v "$PWD/deploy/enginepatch:/opt/enginepatch:ro" "$IMG" /opt/enginepatch/verify.py
```

## Remove it when upstream lands

Both patches are upstream bugs, not local policy. Drop this directory and the two
compose lines once the pinned image returns 400 for a malformed multimodal prompt
and clients stop sending `skip_special_tokens: false`. `verify.py` failing after
an upgrade is the signal that a patch point moved — the wrapper degrades to a
no-op rather than crashing, so check it deliberately on every version bump.
