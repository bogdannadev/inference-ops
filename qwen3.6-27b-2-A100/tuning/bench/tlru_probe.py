#!/usr/bin/env python3
"""Multi-turn eviction pressure: what does a returning conversation pay?

The case T-LRU (--radix-eviction-policy tlru, #34012) targets. --convs
conversations of --history tokens each grow by --turn-new tokens per turn,
visited round-robin and serially, so the device pool overflows and every turn
competes with the others' tails. Run it once per policy on the same drained,
flushed worker and compare the TTFT tail and where each hit came from.

On this hybrid model a hit also needs a GDN checkpoint at the match point, and
HiCache keeps evicted pages on host, so the paper's "evict = recompute" premise
only half holds here. That is exactly what this measures.

  ./tuning/bench/run_eval.sh r1 tlru_probe <label>
"""
import argparse
import json
import random
import sys
import time

from evalkit import Worker


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--convs", type=int, default=5)
    ap.add_argument("--history", type=int, default=50000)
    ap.add_argument("--turn-new", type=int, default=4000)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clear-l3", action="store_true",
                    help="also clear the L3 storage backend on every flush; it deletes the whole "
                         "directory, so only on a worker whose L3 no serving replica shares")
    a = ap.parse_args()
    w = Worker(a.host, a.port, "tlruprobe")
    rng = random.Random(a.seed)

    if not w.flush(l3=a.clear_l3):
        raise RuntimeError("flush failed; refusing to measure warm")
    convs = [[rng.randrange(1000, 150000) for _ in range(a.history)] for _ in range(a.convs)]

    rows = []
    for turn in range(a.turns + 1):          # turn 0 seeds every history cold
        for c in range(a.convs):
            if turn:
                convs[c] += [rng.randrange(1000, 150000) for _ in range(a.turn_new)]
            t = time.time()
            st, txt = w.post_raw("/generate", {"input_ids": convs[c],
                                               "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
                                 timeout=1800)
            dt = round(time.time() - t, 2)
            if st != 200:
                rows.append({"turn": turn, "conv": c, "error": f"HTTP {st}: {txt[:120]}"})
                continue
            mi = json.loads(txt).get("meta_info", {})
            row = {"turn": turn, "conv": c, "prompt": len(convs[c]), "ttft_s": dt,
                   "cached": mi.get("cached_tokens"), "details": mi.get("cached_tokens_details")}
            log(row)
            rows.append(row)

    warm = [r for r in rows if r.get("turn", 0) > 0 and "ttft_s" in r]
    recompute = [r["prompt"] - (r["cached"] or 0) - a.turn_new for r in warm]
    print(json.dumps({
        "label": a.label, "args": vars(a), "rows": rows,
        "warm_ttft_p50": pct([r["ttft_s"] for r in warm], 0.5),
        "warm_ttft_p90": pct([r["ttft_s"] for r in warm], 0.9),
        "warm_ttft_max": max((r["ttft_s"] for r in warm), default=None),
        "warm_ttft_mean": round(sum(r["ttft_s"] for r in warm) / len(warm), 2) if warm else None,
        "recomputed_beyond_new_mean": round(sum(recompute) / len(recompute)) if recompute else None,
        "errors": sum(1 for r in rows if "error" in r),
    }, indent=1))


if __name__ == "__main__":
    main()
