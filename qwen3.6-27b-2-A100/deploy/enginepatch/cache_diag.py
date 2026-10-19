"""Patch 6: cache diagnostics. Observes the prefix match; changes nothing.

On this hybrid model a cache hit has to end where the KV matches AND a GDN
(mamba) state exists. The match computes both and keeps only the second:

    previous turn  [=========== KV for every token ===========]
    GDN states at   ^chunk    ^chunk    ^prompt end    ^output end
    new request    [=========== same ==========X changed ...
                                      B         F
    F = full_kv_hit_length   how far the KV matches (device + host)
    B = device + host hit    the deepest GDN state <= F: the hit we get
    F - B                    recomputed only to rebuild GDN state: the
                             engine's loss, which a placement policy can win back
    prompt - F               the client changed it: nothing can save it

engine.requests records only B (cached_*), so it cannot tell the engine's loss
from the client's. This adds per request:

    kv      F, at the request's last match before admission
    hit     B, the same match (device + host; an L3 prefetch is host by then)
    branch  mamba_branching_seqlen: where upstream records a branch state so
            the NEXT request diverging here hits; null when none is recorded
    msgs    im_start tokens in the prompt
    bnd     position of the last im_start at or before F: the start of the
            message the divergence is in (bnd == F: it is exactly a boundary)
    after   im_start tokens after F: messages following the diverging one

Read straight off the token stream, so the client stays a black box.

Two hooks:
1. UnifiedRadixCache.match_prefix stashes four ints on the request whenever a
   request is passed in. The scheduler re-matches a request just before
   add_one_req (scheduler.py, get_new_batch_prefill), so the last stash is the
   match that admitted it; chunked continuations do not re-match. Cost: one
   tuple per match.
2. SchedulerOutputStreamer.get_cached_tokens_details adds the result as a
   "diag" object once the request has finished. The message scan runs then,
   once per request, vectorised (~1 ms at 200K tokens). The details dict goes
   out in meta_info; --export-metrics-to-file writes it to the request
   record, where Vector reads it. A cold request whose upstream details are
   None now gets {"device": 0, "host": 0, "diag": ...}: the metrics collector
   counts the same zeros, and only a client asking return_cached_tokens_details
   sees zeros instead of null.

Retracted requests re-match with their output appended, so their snapshot is
of the re-admission; num_retractions in the record says which ones.

SGLANG_PATCH_CACHE_DIAG=0 turns both hooks off. SGLANG_PATCH_CACHE_DIAG_IM_START
overrides the im_start id (248045 for Qwen3.8; verify_cache_diag.py checks
it against the tokenizer).
"""

import os
from array import array

ENABLED = os.environ.get("SGLANG_PATCH_CACHE_DIAG", "1") != "0"
IM_START = int(os.environ.get("SGLANG_PATCH_CACHE_DIAG_IM_START", "248045"))

_ATTR = "_cache_diag"       # (F, device, host, branch) from the last match
_ATTR_OUT = "_cache_diag_out"  # the computed object, once finished


def _device_len(result):
    """v0.5.22 reports the device prefix as a length; v0.5.21 as the indices."""
    n = getattr(result, "device_prefix_len", None)
    return int(n) if n is not None else len(result.device_indices)


def note_match(params, result):
    """Called after every match_prefix; must never raise."""
    req = getattr(params, "req", None)
    if req is None:
        return
    try:
        setattr(req, _ATTR, (
            int(result.full_kv_hit_length),
            _device_len(result),
            int(result.host_hit_length),
            result.mamba_branching_seqlen,
        ))
    except Exception:
        pass


def _as_numpy(ids):
    import numpy as np
    if isinstance(ids, array) and ids.typecode == "q":
        return np.frombuffer(ids, dtype=np.int64)
    return np.asarray(ids, dtype=np.int64)


def describe(req):
    """The diag object for a finished request, or None without a match."""
    cached = getattr(req, _ATTR_OUT, None)
    if cached is not None:
        return cached
    snap = getattr(req, _ATTR, None)
    if snap is None:
        return None
    kv, dev, host, branch = snap
    out = {"kv": kv, "hit": dev + host,
           "branch": None if branch is None else int(branch)}
    ids = getattr(req, "origin_input_ids", None)
    # An aborted request's ids are replaced by [0]; F past the end means the
    # stream no longer is the one that was matched.
    if ids is not None and kv <= len(ids):
        import numpy as np
        pos = np.flatnonzero(_as_numpy(ids) == IM_START)
        k = int(np.searchsorted(pos, kv, side="right"))  # starts at <= F
        out["msgs"] = int(len(pos))
        out["bnd"] = int(pos[k - 1]) if k else None
        out["after"] = int(len(pos)) - k
    setattr(req, _ATTR_OUT, out)
    return out
