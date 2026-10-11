"""Engine-side fixes: two for the image-token poisoning incident, one for the
chunked-prefill overcommit crash.

See qwen3.6-27b-2-A100/docs/INCIDENT-2026-09-20-IMAGE-TOKEN-CASCADE.md, sec 3a,
for 1 and 2; patch 3 is documented on _patch_budget below.

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

3. Never hand a continuing prefill chunk more KV than physically exists.

Why here (1 and 2) and not at the gateway: a Higress wasm body rewrite dies at the 32 KB
connection buffer limit and the poisoned bodies are 400-545 KB. The engine holds
the whole request, so this is the first layer that can actually do it.

Loaded via PYTHONPATH -- `site` imports sitecustomize at interpreter startup. The
patches are applied through a wrapper around __import__ rather than by importing
the targets here, so nothing heavy (torch) is pulled in early. The wrapper
removes itself once every patch is in.
"""

import sys
import time

TARGET_PROTO = "sglang.srt.entrypoints.openai.protocol"
TARGET_MM = "sglang.srt.multimodal.processors.base_processor"
TARGET_BUDGET = "sglang.srt.mem_cache.prefill_budget"

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


def _patch_budget(module):
    """Clamp a continuing chunk to the KV that physically fits; park it if none.

    Upstream (v0.5.21 mem_cache/prefill_budget.py:125):

        available = min(chunk_limit, int(self.remaining_total))
        return available if available > 0 else chunk_limit

    remaining_total is free + evictable KV minus the decode headroom reserved
    for running requests. Once a continuation chunk has been clamped to use the
    last of that budget, the next round sees it at <= 0 and falls back to a FULL
    chunk_limit, whatever is actually free. On 2026-10-09 16:50 UTC r0 asked for
    4096 tokens with 960 free and nothing evictable; the allocator raised
    "Prefill out of memory", which kills the scheduler (prefill has no retract
    path). The intent of the fallback, that a half-prefilled request must make
    progress to ever release its KV, is kept: when the budget is exhausted the
    chunk now gets what PHYSICALLY fits (free + evictable, less the one page per
    request alloc_paged_token_slots_extend adds), floored to whole pages. If
    not even one page fits, it returns None, which add_chunked_req already
    treats as "park this round" (the hybrid-SWA budgets use the same return);
    decode keeps running, retracts if it must, and frees KV for the next round.

    Only the base PrefillBudget is touched: the SWA subclasses define their own
    available_chunk_tokens and never reach this one. The positive-budget path
    is upstream's, unchanged. Upstream PR #30743 proposed the pure park; it was
    closed unmerged.
    """
    cls = getattr(module, "PrefillBudget", None)
    if cls is None or not hasattr(cls, "available_chunk_tokens") \
            or not hasattr(cls, "_available_and_evictable"):
        return False
    if getattr(cls, "_overcommit_fixed", False):
        return True
    original = cls.available_chunk_tokens
    stats = {"clamped": 0, "parked": 0, "logged_at": 0.0}

    def available_chunk_tokens(self, chunk_limit):
        if int(self.remaining_total) > 0:
            return original(self, chunk_limit)
        page = self.page_size
        fits = (int(self._available_and_evictable()) - page) // page * page
        if fits >= page:
            result = min(chunk_limit, fits)
            stats["clamped"] += 1
        else:
            result = None
            stats["parked"] += 1
        now = time.monotonic()
        if now - stats["logged_at"] >= 60.0:  # one line a minute at most
            stats["logged_at"] = now
            print(f"[sitecustomize] KV budget exhausted on a continuing chunk: "
                  f"gave {result} of {chunk_limit} (fits={fits}); "
                  f"totals clamped={stats['clamped']} parked={stats['parked']}",
                  file=sys.stderr, flush=True)
        return result

    cls.available_chunk_tokens = available_chunk_tokens
    cls._overcommit_fixed = True
    print("[sitecustomize] continuing prefill chunks clamped to physical KV",
          file=sys.stderr, flush=True)
    return True



def _patch_inject(module):
    """TEST ONLY. Simulate the unaccounted KV drift of 2026-10-09: on the first
    CONTINUING prefill chunk (the long request is already admitted), take every
    free KV page but KEEP tokens from the allocator, hold them HOLD_S seconds,
    then give them back -- a transient consumer the budget never saw, like a
    HiCache load-back for a queued request."""
    cls = getattr(module, "PrefillBudget", None)
    if cls is None or not hasattr(cls, "_available_and_evictable"):
        return False
    inner = cls.available_chunk_tokens
    KEEP, HOLD_S = 3000, 20.0
    st = {"held": None, "t": 0.0, "done": False}

    def available_chunk_tokens(self, chunk_limit):
        now = time.monotonic()
        if st["held"] is not None and now - st["t"] > HOLD_S:
            n = st["held"].numel()
            self.allocator.free(st["held"])
            st["held"] = None
            print(f"[inject] released {n} stolen KV tokens", file=sys.stderr, flush=True)
        elif st["held"] is None and not st["done"]:
            page = self.page_size
            free = int(self.allocator.available_size())
            steal = (free - KEEP) // page * page
            if steal >= page:
                idx = self.allocator.alloc(steal)
                if idx is not None:
                    st.update(held=idx, t=now, done=True)
                    print(f"[inject] stole {steal} of {free} free KV tokens for {HOLD_S}s "
                          f"(allocator {type(self.allocator).__name__})", file=sys.stderr, flush=True)
        got = inner(self, chunk_limit)
        if st["done"]:
            print(f"[inject] chunk call: phys={int(self._available_and_evictable())} "
                  f"remaining_total={int(self.remaining_total)} -> {got}", file=sys.stderr, flush=True)
        return got

    cls.available_chunk_tokens = available_chunk_tokens
    print("[sitecustomize] TEST INJECTOR ACTIVE: transient KV drift on first continuing chunk",
          file=sys.stderr, flush=True)
    return True


_PATCHES = (
    ("proto", TARGET_PROTO, _patch_proto),
    ("mm", TARGET_MM, _patch_mm),
    ("budget", TARGET_BUDGET, _patch_budget),
    ("inject", TARGET_BUDGET, _patch_inject),
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
