"""Two engine-side fixes for the image-token poisoning incident.

See qwen3.6-27b-2-A100/docs/INCIDENT-2026-09-20-IMAGE-TOKEN-CASCADE.md, sec 3a.

1. Force skip_special_tokens=True on every chat completion.

   The engine renders a special token the model emitted back into the reply TEXT
   when the request says skip_special_tokens: false. Clients store that text in
   the conversation history and resend it, where the tokenizer turns it back into
   the real token -- for a vision placeholder that means the prompt claims an
   image nobody attached, and every turn 500s from then on, forever.

   Safe because HF strips only tokens flagged `special: true` in
   added_tokens_decoder. For these weights that is the <|...|> control tokens plus
   the vision and audio ids; <tool_call>, <tool_response> and <think> are
   `special: false` and survive, so --tool-call-parser qwen3_coder and
   --reasoning-parser qwen3 are unaffected.

2. Make a placeholder/payload mismatch a 400 instead of a 500.

Why here and not at the gateway: a Higress wasm body rewrite dies at the 32 KB
connection buffer limit and the poisoned bodies are 400-545 KB. The engine holds
the whole request, so this is the first layer that can actually do it.

Loaded via PYTHONPATH -- `site` imports sitecustomize at interpreter startup. The
patches are applied through a wrapper around __import__ rather than by importing
the targets here, so nothing heavy (torch) is pulled in early. The wrapper
removes itself once both patches are in.
"""

import sys

TARGET_PROTO = "sglang.srt.entrypoints.openai.protocol"
TARGET_MM = "sglang.srt.multimodal.processors.base_processor"

_done = set()


def _patch_proto(module):
    """Force skip_special_tokens=True. Returns True once actually applied."""
    cls = getattr(module, "ChatCompletionRequest", None)
    if cls is None or not hasattr(cls, "to_sampling_params"):
        return False  # module still initialising; try again on the next import
    if getattr(cls, "_sst_forced", False):
        return True
    original = cls.to_sampling_params

    def to_sampling_params(self, *args, **kwargs):
        params = original(self, *args, **kwargs)
        # a dict today; stay defensive so a refactor degrades to a no-op
        if isinstance(params, dict):
            params["skip_special_tokens"] = True
        return params

    cls.to_sampling_params = to_sampling_params
    cls._sst_forced = True
    print("[sitecustomize] forced skip_special_tokens=True on ChatCompletionRequest",
          file=sys.stderr, flush=True)
    return True


def _patch_mm(module):
    """Placeholder/payload mismatch is a CLIENT error: 400, not 500.

    serving_base.handle_request maps ValueError -> 400 and RuntimeError -> 500.
    A 500 is in the router's retryable set (408/429/500/502/503/504), so one
    malformed prompt burns a retry and a breaker credit on every replica in turn;
    on 2026-09-21 that cost 11 failed turns and a replica out of rotation for
    3 h 10 min. A 400 is never retried and never reaches a breaker.

    Narrow ON PURPOSE: only when the RuntimeError was raised while handling a
    StopIteration -- the payload iterator running dry, i.e. more placeholders than
    images. Every other RuntimeError from this function (a GPU OOM in image
    preprocessing, for one) stays a 500 so it keeps alerting.
    """
    cls = getattr(module, "BaseMultimodalProcessor", None)
    if cls is None or not hasattr(cls, "legacy_load_mm_data"):
        return False
    if getattr(cls, "_mm_400_forced", False):
        return True
    original = cls.legacy_load_mm_data

    async def legacy_load_mm_data(self, *args, **kwargs):
        try:
            return await original(self, *args, **kwargs)
        except RuntimeError as exc:
            if isinstance(exc.__context__, StopIteration):
                raise ValueError(str(exc)) from exc
            raise

    cls.legacy_load_mm_data = legacy_load_mm_data
    cls._mm_400_forced = True
    print("[sitecustomize] placeholder/payload mismatch now raises ValueError (400)",
          file=sys.stderr, flush=True)
    return True


_PATCHES = (
    ("proto", TARGET_PROTO, _patch_proto),
    ("mm", TARGET_MM, _patch_mm),
)


def _install():
    try:
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            module = real_import(name, globals, locals, fromlist, level)
            for label, target, fn in _PATCHES:
                if target in _done:
                    continue
                mod = sys.modules.get(target)
                if mod is None:
                    continue
                try:
                    if fn(mod):
                        _done.add(target)
                except Exception as exc:
                    _done.add(target)  # never retry a broken patch on every import
                    print(f"[sitecustomize] {label} patch failed: {exc!r}",
                          file=sys.stderr, flush=True)
            if len(_done) == len(_PATCHES):
                builtins.__import__ = real_import
            return module

        builtins.__import__ = guarded_import
    except Exception as exc:
        print(f"[sitecustomize] install failed: {exc!r}", file=sys.stderr, flush=True)


_install()
