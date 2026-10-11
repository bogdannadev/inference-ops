import asyncio, inspect, types
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
import sglang.srt.multimodal.processors.base_processor as bp
import sitecustomize   # the patch module itself, so we test the real function

# --- patch 1 ---
r = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}],
                          skip_special_tokens=False)
assert r.to_sampling_params(stop=[], model_generation_config={})["skip_special_tokens"] is True
print("patch1  skip_special_tokens forced      : OK")

# --- patch 2 installed on the real class ---
cls = bp.BaseMultimodalProcessor
assert getattr(cls, "_mm_400_forced", False), "patch2 not installed"
assert inspect.iscoroutinefunction(cls.legacy_load_mm_data), "patch2 broke async-ness"
print("patch2  installed on real class, async  : OK")

# --- patch 2 behaviour, through sitecustomize._patch_mm itself ---
def stub(raiser):
    class BaseMultimodalProcessor:
        async def legacy_load_mm_data(self, *a, **k):
            raiser()
    m = types.ModuleType("stub"); m.BaseMultimodalProcessor = BaseMultimodalProcessor
    assert sitecustomize._patch_mm(m) is True
    return BaseMultimodalProcessor()

def mismatch():                      # exactly what base_processor.py:1495 does
    try:
        next(iter([]))               # payload iterator runs dry
    except StopIteration as e:
        raise RuntimeError(f"An exception occurred while loading multimodal data: {e}")

def oom():                           # the 2026-09-18/19 image-preprocess failure
    try:
        raise MemoryError("CUDA out of memory")
    except MemoryError as e:
        raise RuntimeError(f"An exception occurred while loading multimodal data: {e}")

for label, raiser, expect, code in (("placeholder mismatch", mismatch, ValueError, 400),
                                    ("gpu oom            ", oom, RuntimeError, 500)):
    try:
        asyncio.run(stub(raiser).legacy_load_mm_data())
        raise SystemExit(f"{label}: expected a raise")
    except expect as e:
        print(f"  {label} -> {expect.__name__:12s} HTTP {code}  OK")
    except Exception as e:
        raise SystemExit(f"{label}: got {type(e).__name__}, wanted {expect.__name__}")

# --- patch 2's message: actionable, and free of the markup it warns about ---
# The body is stored in the client's conversation history, so markup in it would
# poison the session it is telling the user to abandon. Upstream's own string is
# empty for this branch, which is what made the 400 useless.
msg = None
try:
    asyncio.run(stub(mismatch).legacy_load_mm_data())
except ValueError as e:
    msg = str(e)
assert msg, "mismatch raised no message"
assert len(msg) > 80, f"mismatch message too terse to act on ({len(msg)} chars)"
assert "new conversation" in msg, "mismatch message does not say what to do"
assert "Retrying cannot help" in msg, "mismatch message does not forbid the retry"
for poison in ("image_pad", "vision_start", "vision_end", "video_pad", "<|"):
    assert poison not in msg, f"mismatch message contains {poison!r} — it would poison the reply"
print("patch2  message actionable, markup-free : OK")

# --- patch 3: continuing prefill chunk never exceeds physical KV ---
import sglang.srt.mem_cache.prefill_budget as pb

assert getattr(pb.PrefillBudget, "_overcommit_fixed", False), "patch3 not installed"
for sub in ("SWAPrefillBudget", "SharedSWAPrefillBudget"):  # must keep their own
    assert pb.__dict__[sub].__dict__.get("available_chunk_tokens") is not None, sub
print("patch3  installed on base class only    : OK")

class Alloc:                         # what PrefillBudget reads from the allocator
    page_size = 64
    def __init__(self, free): self.free = free
    def available_size(self): return self.free

class Tree:                          # hybrid-Mamba tree: full_evictable_size path
    def __init__(self, evictable): self.ev = evictable
    def supports_mamba(self): return True
    def full_evictable_size(self): return self.ev

def budget(free, evictable, offset):
    b = pb.PrefillBudget(Alloc(free), Tree(evictable))
    b.total_offset = offset          # decode headroom reserved for running reqs
    return b

# positive budget: upstream behaviour, untouched
assert budget(100_000, 0, 1_000).available_chunk_tokens(4096) == 4096
assert budget(2_600, 0, 1_000).available_chunk_tokens(4096) == 1_600
print("patch3  positive budget == upstream     : OK")

# 2026-10-09 16:50: 960 free, 0 evictable, budget spent -> upstream gave 4096
got = budget(960, 0, 1_000).available_chunk_tokens(4096)
assert got == 896, got               # floor((960 - 64) / 64) * 64
assert got + Alloc.page_size <= 960, "alloc_paged_token_slots_extend would still fail"
print(f"patch3  Friday's state -> {got} (was 4096)   : OK")

# evictable KV counts as room
assert budget(100, 4_000, 9_999).available_chunk_tokens(4096) == 4_032

# less than one page beyond the alloc margin -> park (None), never overcommit
assert budget(100, 0, 1_000).available_chunk_tokens(4096) is None
assert budget(0, 0, 0).available_chunk_tokens(4096) is None
print("patch3  nothing fits -> park (None)     : OK")

# idempotent: a second install must not double-wrap
assert sitecustomize._patch_proto.__module__ == "sitecustomize"
assert sitecustomize._patch_budget(pb) is True
assert budget(960, 0, 1_000).available_chunk_tokens(4096) == 896
print("OK")
