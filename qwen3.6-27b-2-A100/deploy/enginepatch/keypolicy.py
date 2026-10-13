"""Patch 5: per-key context window and reasoning effort.

The gateway authenticates a key and stamps its name on the request as
x-request-id-labels {"consumer": ...} (deploy/higress-consumer-label.sh,
OVERWRITE_IF_EXISTS_OR_ADD), and SGLang keeps it as obj.custom_labels. The
gateway itself cannot vary a limit per key: a WasmPlugin matchRule never
selects by consumer, and it cannot count prompt tokens. The engine can do both,
so per-key limits that depend on tokens live here.

POLICY FILE

quota-bot is its only writer (EnginePolicy in quota-bot/bot.cs). It is
re-read when its mtime or size changes, checked at most once a second:

    {"version": 1,
     "default":   {"context": 169000},
     "consumers": {"alice": {"context": 262144, "effort": "low"}}}

A consumer without an entry, a request without a consumer label (the direct
hostname, internal tools), and every request while the file is missing or
unreadable get the default context. That default is the window everyone had
before the engine was raised to the model's full 262,144, so a broken policy
file degrades to the old behaviour, never to the full window. A file that
fails to parse keeps the last good copy.

1. CONTEXT. After upstream's own checks against --context-length, a request is
   held to its key's window with the same two checks and the same error text
   (a 400), so clients that recognise the overflow message (OpenCode compacts
   on it) behave exactly as before. A request that names no max_tokens gets
   max_new_tokens = window - prompt - 1, which is what the scheduler does with
   --context-length; without it a 169K key could prompt 160K and then decode
   past its own window up to the engine's 262K.

2. REASONING EFFORT. In v0.5.21 the client's reasoning_effort never reaches the
   chat template. serving_chat._process_messages merges
   --default-chat-template-kwargs into chat_template_kwargs with setdefault,
   and _apply_jinja_template then builds the template kwargs as
   {"reasoning_effort": request.reasoning_effort} updated WITH
   chat_template_kwargs, so the server's "xhigh" overwrites whatever the client
   asked for. A reasoning_effort sent inside chat_template_kwargs is popped
   into request.reasoning_effort first and loses the same way. Measured on r1
   2026-10-14: low, medium, high and chat_template_kwargs low all rendered the
   53-token xhigh prompt; only "none" (thinking off) got through.

   The fix writes the effort into chat_template_kwargs before the merge, so
   setdefault keeps it. The effort is the request's own if it named one,
   else the key's default from the policy, else untouched (the server's
   --default-chat-template-kwargs applies, xhigh). Qwen3.8's template accepts
   only xhigh, medium and low and raises on anything else, so OpenAI-style
   names are mapped: minimal -> low, high and max -> xhigh. high maps UP so
   that a client asking for "high" keeps the effort it has actually been
   getting. none / no_think pass through untouched: upstream already turns
   them into thinking off. An unknown value is left alone, which means xhigh,
   as before.
"""

import contextvars
import json
import os
import sys
import threading
import time

PATH = os.environ.get("KEYPOLICY_PATH", "/opt/keypolicy/policy.json")
DEFAULT_CONTEXT = int(os.environ.get("KEYPOLICY_DEFAULT_CONTEXT", "169000"))
CHECK_INTERVAL = 1.0

_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
}
_THINKING_OFF = ("none", "no_think")

# The consumer of the request being converted, set around
# _convert_to_internal_request (synchronous, so the value cannot leak into
# another request) and read by _process_messages inside it.
_consumer = contextvars.ContextVar("keypolicy_consumer", default=None)


class _Policy:
    def __init__(self):
        self._lock = threading.Lock()
        self._checked = 0.0
        self._stamp = None
        self._default = DEFAULT_CONTEXT
        self._consumers = {}
        self._warned = None

    def _warn(self, msg):
        if msg != self._warned:  # one line per distinct problem, not per request
            self._warned = msg
            print(f"[keypolicy] {msg}", file=sys.stderr, flush=True)

    def _refresh(self):
        now = time.monotonic()
        if now - self._checked < CHECK_INTERVAL:
            return
        with self._lock:
            if now - self._checked < CHECK_INTERVAL:
                return
            self._checked = now
            try:
                st = os.stat(PATH)
            except OSError:
                if self._stamp is not None or self._warned is None:
                    self._stamp = None
                    self._default = DEFAULT_CONTEXT
                    self._consumers = {}
                    self._warn(f"{PATH} missing: every key gets the default {DEFAULT_CONTEXT}")
                return
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            try:
                with open(PATH, "rb") as f:
                    doc = json.load(f)
                default = int(doc.get("default", {}).get("context") or DEFAULT_CONTEXT)
                consumers = {}
                for name, entry in (doc.get("consumers") or {}).items():
                    if not isinstance(entry, dict):
                        continue
                    ctx = entry.get("context")
                    eff = entry.get("effort")
                    consumers[str(name)] = (
                        int(ctx) if isinstance(ctx, int) and ctx > 0 else None,
                        eff if isinstance(eff, str) and eff else None,
                    )
            except Exception as exc:  # keep the last good copy
                self._warn(f"{PATH} unreadable, keeping the previous policy: {exc!r}")
                return
            self._stamp = stamp
            self._default = default
            self._consumers = consumers
            self._warned = None
            print(f"[keypolicy] loaded: default context {default}, "
                  f"{len(consumers)} consumer entries", file=sys.stderr, flush=True)

    def context(self, consumer):
        self._refresh()
        entry = self._consumers.get(consumer) if consumer else None
        return entry[0] if entry and entry[0] else self._default

    def effort(self, consumer):
        self._refresh()
        entry = self._consumers.get(consumer) if consumer else None
        return entry[1] if entry else None


POLICY = _Policy()


def consumer_of(labels):
    if isinstance(labels, dict):
        c = labels.get("consumer")
        if isinstance(c, str) and c:
            return c
    return None


def resolve_effort(request, consumer):
    """Pin the effort into chat_template_kwargs so the server default cannot
    overwrite it. Called before upstream's _process_messages."""
    ctk = dict(request.chat_template_kwargs or {})
    asked = request.reasoning_effort
    if asked is None:
        asked = ctk.get("reasoning_effort")
    if asked is None:
        asked = POLICY.effort(consumer)
        if asked is None:
            return
    if not isinstance(asked, str):
        return
    key = asked.strip().lower()
    if key in _THINKING_OFF:
        request.reasoning_effort = key
        return
    mapped = _EFFORT_MAP.get(key)
    if mapped is None:
        return
    ctk["reasoning_effort"] = mapped
    request.chat_template_kwargs = ctk
    request.reasoning_effort = mapped


def enforce_context(tm, obj, input_ids):
    """Hold a request to its key's window. Called after upstream's checks."""
    sp = getattr(obj, "sampling_params", None)
    if not isinstance(sp, dict):
        return
    limit = POLICY.context(consumer_of(getattr(obj, "custom_labels", None)))
    if limit >= tm.context_len:
        return
    n_input = len(input_ids) if input_ids is not None else 0
    counted = n_input + tm.num_reserved_tokens
    if counted >= limit:
        raise ValueError(
            f"The input ({counted} tokens) is longer than the "
            f"model's context length ({limit} tokens). "
            f"This API key's context window is {limit} tokens."
        )
    max_new = sp.get("max_new_tokens")
    if max_new is None:
        sp["max_new_tokens"] = max(1, limit - n_input - 1)
    elif max_new + counted > limit:
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {limit} tokens. You requested a total of {max_new + counted} "
            f"tokens: {counted} tokens from the input messages and "
            f"{max_new} tokens for the completion. Please reduce the number "
            f"of tokens in the input messages or the completion to fit within the limit."
        )
