"""Offline check of patch 6 (cache_diag.py) in the pinned image, no GPU.

    docker run --rm --cpuset-cpus 48-55 --network none -e HF_HUB_OFFLINE=1 \
      -e PYTHONPATH=/opt/enginepatch \
      -v $PWD/deploy/enginepatch-next:/opt/enginepatch:ro \
      -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
      --entrypoint python3 <pinned image> /opt/enginepatch/verify_cache_diag.py

1. Both hooks install on the real classes, and the patch points still carry
   the fields the patch reads (a moved field would make it a silent no-op).
2. The im_start id is the tokenizer's.
3. The match hook stashes F, hit and branch from a real MatchResult.
4. The real, patched get_cached_tokens_details adds "diag" for a finished
   request only, keeps upstream's keys, covers a cold request, survives an
   aborted one, and the result pickles (it crosses to the detokenizer by ZMQ).
5. The message scan is cheap at 262K tokens.
"""

import glob
import json
import pickle
import sys
import time
from array import array
from types import SimpleNamespace

import torch

import sglang.srt.managers.scheduler_components.output_streamer as osm
import sglang.srt.mem_cache.unified_radix_cache as urc
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
import cache_diag as cd
import sitecustomize as sc

fails = 0


def check(label, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")


# --- 1. hooks and patch points ----------------------------------------------
check("match hook installed", getattr(urc.UnifiedRadixCache, "_cache_diag", False))
check("export hook installed", getattr(osm.SchedulerOutputStreamer, "_cache_diag", False))
fields = MatchResult._fields
for f in ("full_kv_hit_length", "device_indices", "host_hit_length",
          "mamba_branching_seqlen"):
    check(f"MatchResult has {f}", f in fields)
check("MatchPrefixParams carries req", "req" in MatchPrefixParams.__dataclass_fields__)

# --- 2. im_start id ---------------------------------------------------------
paths = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/"
                  "snapshots/*/tokenizer.json")
if paths:
    added = json.load(open(paths[0]))["added_tokens"]
    # Built from parts: the delimited form, written out, tokenizes to the
    # real control token in any file a model later reads.
    name = "<" + "|im_start|" + ">"
    ids = {a["content"]: a["id"] for a in added}
    check("im_start id matches the tokenizer", ids.get(name) == cd.IM_START,
          f"tokenizer {ids.get(name)}, patch {cd.IM_START}")
else:
    check("tokenizer.json found (mount the HF cache)", False)

# --- 3. match hook on a stand-in cache ----------------------------------------
IM = cd.IM_START


def prompt(*lengths):
    """Messages of the given lengths, each opening with im_start."""
    out = array("q")
    for n in lengths:
        out.append(IM)
        out.extend([7] * (n - 1))
    return out


class StandInCache:  # not a subclass: the real class already carries the flag
    def __init__(self, result):
        self.result = result

    def match_prefix(self, params):
        return self.result


def match(kv, dev, host, branch):
    return MatchResult(device_indices=torch.zeros(dev, dtype=torch.int64),
                       last_device_node=None, last_host_node=None,
                       best_match_node=None, host_hit_length=host,
                       mamba_branching_seqlen=branch, full_kv_hit_length=kv)


sc._patch_urc(SimpleNamespace(UnifiedRadixCache=StandInCache))
ids = prompt(100, 300, 600, 50)  # starts at 0, 100, 400, 1000; 1050 tokens
req = SimpleNamespace(origin_input_ids=ids, finished=lambda: True,
                      cached_tokens=256, cached_tokens_device=192,
                      cached_tokens_host=64, cached_tokens_storage=0)
StandInCache(match(704, 192, 64, 704)).match_prefix(
    MatchPrefixParams(key=None, req=req))
check("match stashed", getattr(req, cd._ATTR, None) == (704, 192, 64, 704),
      repr(getattr(req, cd._ATTR, None)))
StandInCache(match(10, 0, 0, None)).match_prefix(MatchPrefixParams(key=None))
check("match without req ignored", getattr(req, cd._ATTR)[0] == 704)

# --- 4. the real export hook --------------------------------------------------
class Streamer(osm.SchedulerOutputStreamer):
    """The real, patched method; only the two storage lookups stand in."""

    def __init__(self):
        pass

    def enable_hicache_storage(self):
        return True

    def _get_storage_backend_type(self):
        return "HiCacheFile"


streamer = Streamer()
get = osm.SchedulerOutputStreamer.get_cached_tokens_details

d = get(streamer, req)
diag = d.get("diag") if d else None
check("upstream keys kept", d and d["device"] == 192 and d["host"] == 64
      and d["storage"] == 0 and d["storage_backend"] == "HiCacheFile", repr(d))
check("diag F / hit / branch", diag and (diag["kv"], diag["hit"], diag["branch"])
      == (704, 256, 704), repr(diag))
check("diag message boundary", diag and (diag["msgs"], diag["bnd"], diag["after"])
      == (4, 400, 1), repr(diag))
check("details pickle", pickle.loads(pickle.dumps(d)) == d)

on_bnd = SimpleNamespace(origin_input_ids=ids, finished=lambda: True,
                         cached_tokens=0, cached_tokens_device=0,
                         cached_tokens_host=0, cached_tokens_storage=0)
setattr(on_bnd, cd._ATTR, (1000, 0, 0, None))
d = get(streamer, on_bnd)
check("cold request gets details", d and d["device"] == 0 and d["host"] == 0, repr(d))
check("divergence on a boundary: bnd == F, last message after none",
      d and (d["diag"]["bnd"], d["diag"]["after"], d["diag"]["branch"])
      == (1000, 0, None), repr(d))

running = SimpleNamespace(**{**vars(req), "finished": lambda: False})
running.__dict__.pop(cd._ATTR_OUT, None)
d = get(streamer, running)
check("unfinished request unchanged", d is not None and "diag" not in d, repr(d))

aborted = SimpleNamespace(origin_input_ids=array("q", [0]), finished=lambda: True,
                          cached_tokens=0, cached_tokens_device=0,
                          cached_tokens_host=0, cached_tokens_storage=0)
setattr(aborted, cd._ATTR, (704, 192, 64, None))
d = get(streamer, aborted)
check("aborted request: F kept, no message scan",
      d and d["diag"]["kv"] == 704 and "bnd" not in d["diag"], repr(d))

unmatched = SimpleNamespace(**{k: v for k, v in vars(aborted).items()
                               if k != cd._ATTR and k != cd._ATTR_OUT})
d = get(streamer, unmatched)
check("never matched: upstream result untouched", d is None, repr(d))

as_list = SimpleNamespace(origin_input_ids=list(ids), finished=lambda: True)
setattr(as_list, cd._ATTR, (704, 192, 64, 704))
check("list ids scan the same", cd.describe(as_list)["bnd"] == 400)

# --- 5. cost ------------------------------------------------------------------
big = prompt(*([1024] * 256))  # 262,144 tokens, 256 messages
best = 1e9
for _ in range(5):
    r = SimpleNamespace(origin_input_ids=big)
    setattr(r, cd._ATTR, (200_000, 0, 0, None))
    t = time.perf_counter()
    cd.describe(r)
    best = min(best, time.perf_counter() - t)
check("scan of 262K tokens under 5 ms", best < 0.005, f"{best * 1e3:.2f} ms")

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
