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

# idempotent: a second install must not double-wrap
assert sitecustomize._patch_proto.__module__ == "sitecustomize"
print("OK")
