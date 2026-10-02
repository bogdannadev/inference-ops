#!/usr/bin/env python3
"""Do short prompts wait behind a long cold prefill?

The case shortest-prefill-first (#40024) targets. Under lpm/hrrn the chunked
long request keeps the whole --chunked-prefill-size budget each step, so a short
prompt that arrives mid-prefill waits for the long one to finish. SPF caps the
long request's chunk so that shorter waiting prompts fit beside it.

Per round, on a drained and flushed worker:
  1. a cold --long-len prompt (random token ids, 1 output token)
  2. --short-n cold --short-len prompts, arriving every --short-gap-s seconds
     from --short-delay-s on
TTFT is the e2e of a 1-token /generate. Record the short TTFTs and what the
long prompt paid for them.

  ./tuning/bench/run_eval.sh r1 spf_probe <label>
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


def one(w, prompt, out, key):
    t = time.time()
    st, txt = w.post_raw("/generate", {"input_ids": prompt,
                                       "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
                         timeout=1800)
    out[key] = round(time.time() - t, 2) if st == 200 else f"HTTP {st}: {txt[:120]}"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--long-len", type=int, default=100000)
    ap.add_argument("--short-len", type=int, default=2000)
    ap.add_argument("--short-n", type=int, default=3)
    ap.add_argument("--short-delay-s", type=float, default=5)
    ap.add_argument("--short-gap-s", type=float, default=2)
    ap.add_argument("--clear-l3", action="store_true",
                    help="also clear the L3 storage backend on every flush; it deletes the whole "
                         "directory, so only on a worker whose L3 no serving replica shares")
    a = ap.parse_args()
    w = Worker(a.host, a.port, "spfprobe")

    rounds = []
    for r in range(a.rounds):
        rng = random.Random(1000 + r)
        if not w.flush(l3=a.clear_l3):
            raise RuntimeError("flush failed; refusing to measure warm")
        out = {}
        th = [threading.Thread(target=one, args=(w, ids(rng, a.long_len), out, "long"))]
        th[0].start()
        time.sleep(a.short_delay_s)
        for i in range(a.short_n):
            t = threading.Thread(target=one, args=(w, ids(rng, a.short_len), out, f"short{i}"))
            t.start()
            th.append(t)
            time.sleep(a.short_gap_s)
        for t in th:
            t.join()
        log(f"round {r}: {out}")
        rounds.append(out)

    shorts = [v for o in rounds for k, v in o.items() if k.startswith("short") and isinstance(v, float)]
    longs = [o["long"] for o in rounds if isinstance(o.get("long"), float)]
    print(json.dumps({
        "label": a.label, "args": vars(a), "rounds": rounds,
        "short_ttft_p50": pct(shorts, 0.5), "short_ttft_max": max(shorts, default=None),
        "long_ttft_mean": round(sum(longs) / len(longs), 2) if longs else None,
        "errors": sum(1 for o in rounds for v in o.values() if not isinstance(v, float)),
    }, indent=1))


if __name__ == "__main__":
    main()
