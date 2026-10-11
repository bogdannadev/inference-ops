"""Build the TEST-ONLY K/V range capture variant (never mount in production)."""
import os
CAPTURE = '''

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
'''
s = open("enginepatch/sitecustomize.py").read()
s = s.replace("\nimport sys\n", "\nimport os\nimport sys\n", 1)
s = s.replace("if target in _done:", "if label in _done:").replace("_done.add(target)", "_done.add(label)")
s = s.replace("\n_PATCHES = (", CAPTURE + "\n\n_PATCHES = (")
s = s.replace('    ("mm", TARGET_MM, _patch_mm),\n)',
              '    ("mm", TARGET_MM, _patch_mm),\n    ("capture", "sglang.srt.mem_cache.memory_pool", _patch_capture),\n)')
assert '("capture"' in s
os.makedirs("enginepatch-test-capture", exist_ok=True)
open("enginepatch-test-capture/sitecustomize.py", "w").write(s)
print("built enginepatch-test-capture")
