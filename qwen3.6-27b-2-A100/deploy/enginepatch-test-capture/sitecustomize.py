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

import os
import sys

TARGET_PROTO = "sglang.srt.entrypoints.openai.protocol"
TARGET_MM = "sglang.srt.multimodal.processors.base_processor"

_done = set()

# What the client is told when its history claims an image it never attached.
#
# Upstream's string is useless here: base_processor formats it as
# f"...: {e}" where e is a bare StopIteration carrying no args, so the body
# reads "An exception occurred while loading multimodal data:" and stops. Every
# log line from the 2026-09-22 burst ends in that dangling colon. The user sees
# a blank error and does the one thing that cannot work -- retries.
#
# This string is returned to the client, which stores it in the conversation
# history. So it MUST NOT contain special-token markup in any form: writing the
# literal placeholder here would poison the very session it is telling the user
# to abandon. Plain words only -- see docs/INCIDENT-2026-09-20-IMAGE-TOKEN-CASCADE.md.
_MISMATCH_MESSAGE = (
    "This conversation contains an image placeholder with no image attached, "
    "so it cannot be processed. The usual cause is image-placeholder markup "
    "that entered the conversation as ordinary text, most often in an earlier "
    "model reply, and it is resent with the history on every turn. Retrying "
    "cannot help: it fails identically every time, on every replica. Start a "
    "new conversation -- this history cannot be repaired by resending it."
)


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

    The upstream text is replaced rather than wrapped, because it is empty for
    this branch; see _MISMATCH_MESSAGE. The original stays reachable as
    __cause__ for the server-side traceback.
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
                raise ValueError(_MISMATCH_MESSAGE) from exc
            raise

    cls.legacy_load_mm_data = legacy_load_mm_data
    cls._mm_400_forced = True
    print("[sitecustomize] placeholder/payload mismatch now raises ValueError (400)",
          file=sys.stderr, flush=True)
    return True



_KV_TRIGGER = "/tmp/kvstats.trigger"
_KV_OUT = "/app/logs/kvcal/kvstats.pt"
_KV_LO, _KV_HI, _KV_NB = -30.0, 12.0, 420      # log2|x| histogram, 0.1-bit bins


def _patch_capture(module):
    """TEST ONLY. Histogram log2|K| and log2|V| per attention layer, BEFORE the
    fp8 cast, plus amax. Skips CUDA-graph capture (no host syncs anywhere), so
    decode replays are not seen: the statistics come from eager prefill."""
    cls = getattr(module, "HybridLinearKVPool", None)
    if cls is None or not hasattr(cls, "set_kv_buffer"):
        return False
    original = cls.set_kv_buffer
    stats, state = {}, {"err": False}

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, *args, **kwargs):
        try:
            torch = sys.modules["torch"]
            if (layer is not None and cache_k is not None and cache_k.is_floating_point()
                    and not torch.cuda.is_current_stream_capturing()):
                st = stats.get(layer.layer_id)
                if st is None:
                    z = lambda: torch.zeros(_KV_NB, dtype=torch.float64, device=cache_k.device)
                    st = stats[layer.layer_id] = {
                        "k": z(), "v": z(),
                        "kmax": torch.zeros((), dtype=torch.float32, device=cache_k.device),
                        "vmax": torch.zeros((), dtype=torch.float32, device=cache_k.device)}
                for name, x in (("k", cache_k), ("v", cache_v)):
                    a = x.detach().float().abs().flatten()
                    e = torch.log2(a.clamp_min(2.0 ** _KV_LO))
                    st[name] += torch.histc(e, bins=_KV_NB, min=_KV_LO, max=_KV_HI).double()
                    st[name + "max"] = torch.maximum(st[name + "max"], a.max())
            if os.path.exists(_KV_TRIGGER):
                os.remove(_KV_TRIGGER)
                torch = sys.modules["torch"]
                os.makedirs(os.path.dirname(_KV_OUT), exist_ok=True)
                torch.save({lid: {k: v.cpu() for k, v in st.items()} for lid, st in stats.items()}
                           | {"_bins": (_KV_LO, _KV_HI, _KV_NB)}, _KV_OUT)
                print(f"[capture] wrote {_KV_OUT} ({len(stats)} layers)", file=sys.stderr, flush=True)
        except Exception as exc:
            if not state["err"]:
                state["err"] = True
                print(f"[capture] disabled after error: {exc!r}", file=sys.stderr, flush=True)
        return original(self, layer, loc, cache_k, cache_v, *args, **kwargs)

    cls.set_kv_buffer = set_kv_buffer
    print("[sitecustomize] TEST CAPTURE ACTIVE: K/V range histograms on HybridLinearKVPool",
          file=sys.stderr, flush=True)
    return True


_PATCHES = (
    ("proto", TARGET_PROTO, _patch_proto),
    ("mm", TARGET_MM, _patch_mm),
    ("capture", "sglang.srt.mem_cache.memory_pool", _patch_capture),
)


def _install():
    try:
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            module = real_import(name, globals, locals, fromlist, level)
            for label, target, fn in _PATCHES:
                if label in _done:
                    continue
                mod = sys.modules.get(target)
                if mod is None:
                    continue
                try:
                    if fn(mod):
                        _done.add(label)
                except Exception as exc:
                    _done.add(label)  # never retry a broken patch on every import
                    print(f"[sitecustomize] {label} patch failed: {exc!r}",
                          file=sys.stderr, flush=True)
            if len(_done) == len(_PATCHES):
                builtins.__import__ = real_import
            return module

        builtins.__import__ = guarded_import
    except Exception as exc:
        print(f"[sitecustomize] install failed: {exc!r}", file=sys.stderr, flush=True)


_install()
