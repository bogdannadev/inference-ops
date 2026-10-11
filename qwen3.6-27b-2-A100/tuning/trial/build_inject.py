"""Build TEST-ONLY sitecustomize variants with a fault injector (never mount in production)."""
import os, sys
INJECT = '''

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
'''
def build(src, dst, keep_budget):
    s = open(src).read()
    if "\nimport time\n" not in s:   # module-level: an import inside a patch re-enters the hook
        s = s.replace("\nimport sys\n", "\nimport sys\nimport time\n", 1)
    s = s.replace("if target in _done:", "if label in _done:").replace("_done.add(target)", "_done.add(label)")
    s = s.replace("\n_PATCHES = (", INJECT + "\n\n_PATCHES = (")
    if keep_budget:
        s = s.replace('    ("budget", TARGET_BUDGET, _patch_budget),\n)',
                      '    ("budget", TARGET_BUDGET, _patch_budget),\n    ("inject", TARGET_BUDGET, _patch_inject),\n)')
        assert '("inject"' in s
    else:
        s = s.replace('    ("mm", TARGET_MM, _patch_mm),\n)',
                      '    ("mm", TARGET_MM, _patch_mm),\n    ("inject", "sglang.srt.mem_cache.prefill_budget", _patch_inject),\n)')
        assert '("inject"' in s
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(dst, "sitecustomize.py"), "w").write(s)
    print("built", dst)
build("enginepatch/sitecustomize.py", "enginepatch-test-stock", keep_budget=False)
build("enginepatch-next/sitecustomize.py", "enginepatch-test-patched", keep_budget=True)
