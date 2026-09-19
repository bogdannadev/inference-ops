#!/usr/bin/env python3
"""Cold prefill time, and how long a long prefill stalls running decodes.

The scheduler runs a prefill chunk OR a decode step per iteration (no mixed
chunk), so a running decoder waits one chunk forward between its steps while
a long prompt is prefilled. --chunked-prefill-size trades that stall against
total prefill time. Measured on a drained, flushed worker:

  1. cold prefill alone at each --lengths (random token ids, 1 output token)
  2. DECODERS streams decoding (ignore_eos), then a cold --stall-len prompt
     arrives; per-decoder gaps between stream chunks while it prefills, and
     decode tokens/s inside vs outside that window

  ./tuning/bench/run_eval.sh r1 prefill_stall <label>
"""
import argparse
import json
import random
import sys
import threading
import time

from evalkit import Worker


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def ids(rng, n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def cold(w, prompt):
    if not w.flush():
        raise RuntimeError("flush failed")
    t = time.time()
    st, txt = w.post_raw("/generate", {"input_ids": prompt,
                                       "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
                         timeout=1800)
    if st != 200:
        raise RuntimeError(f"{st}: {txt[:200]}")
    return round(time.time() - t, 2)


def stream_decode(w, prompt, max_new, out):
    """Stream /generate; record (time, completion_tokens) per chunk."""
    c = w._conn(1800)
    body = {"input_ids": prompt, "stream": True,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "ignore_eos": True}}
    c.request("POST", "/generate", body=json.dumps(body), headers=w._headers())
    r = c.getresponse()
    if r.status != 200:
        out["error"] = f"HTTP {r.status}"
        return
    pts = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[5:])
        pts.append((time.time(), ev.get("meta_info", {}).get("completion_tokens", 0)))
    out["pts"] = pts


def window_stats(pts, t0, t1):
    # every gap that overlaps [t0, t1], including the ones crossing its edges:
    # a decoder frozen for the whole prefill shows as one gap spanning it
    gaps = [b[0] - a[0] for a, b in zip(pts, pts[1:]) if b[0] > t0 and a[0] < t1]
    inside = [p for p in pts if t0 <= p[0] <= t1]
    tok = (inside[-1][1] - inside[0][1]) if len(inside) > 1 else 0
    span = (inside[-1][0] - inside[0][0]) if len(inside) > 1 else 0
    gaps.sort()
    return {"chunks": len(gaps), "gap_max_s": round(gaps[-1], 3) if gaps else None,
            "gap_p50_s": round(gaps[len(gaps) // 2], 3) if gaps else None,
            "gaps_over_1s": sum(g > 1 for g in gaps),
            "tok_s": round(tok / span, 1) if span else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--lengths", default="45000,131072")
    ap.add_argument("--stall-len", type=int, default=131072)
    ap.add_argument("--decoders", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=9000)
    a = ap.parse_args()
    rng = random.Random(4242)
    w = Worker(a.host, a.port, consumer=f"stall-{a.label}")
    res = {"label": a.label, "cold": {}}

    for n in [int(x) for x in a.lengths.split(",")]:
        res["cold"][n] = cold(w, ids(rng, n))
        log(f"cold {n}: {res['cold'][n]} s")

    if not w.flush():
        raise RuntimeError("flush failed")
    outs = [{} for _ in range(a.decoders)]
    th = [threading.Thread(target=stream_decode, args=(w, ids(rng, 300), a.decode_tokens, o)) for o in outs]
    for t in th:
        t.start()
    time.sleep(8)                                    # decoders in steady state
    big = ids(rng, a.stall_len)
    t0 = time.time()
    st, txt = w.post_raw("/generate", {"input_ids": big,
                                       "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
                         timeout=1800)
    t1 = time.time()
    res["stall_prefill_s"] = round(t1 - t0, 2)
    for t in th:
        t.join()
    res["decoders"] = []
    for o in outs:
        if "error" in o or not o.get("pts"):
            res["decoders"].append({"error": o.get("error", "no stream")})
            continue
        pts = o["pts"]
        res["decoders"].append({"during": window_stats(pts, t0, t1),
                                "before": window_stats(pts, pts[0][0], t0),
                                "after": window_stats(pts, t1, pts[-1][0]),
                                "finished_before_prefill_end": pts[-1][0] < t1})
    log(f"stall: {a.stall_len} prefill {res['stall_prefill_s']} s (status {st}); decoders "
        + json.dumps([d.get("during") for d in res["decoders"]]))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
