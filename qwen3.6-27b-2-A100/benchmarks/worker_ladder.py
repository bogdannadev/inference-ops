#!/usr/bin/env python3
"""Concurrency ladder against ONE worker (bypasses the router).

benchmarks/ladder.py targets the router and reports the r0/r1 split, which is
useless for a split-build A/B -- it would blend the two engines. This walks
concurrency against a single worker so the engine is the only variable.

Reports per rung: TTFT and per-chunk latency, which is where PR #32219's
host-seam work would show up if it shows up at all.

MEASUREMENT NOTE: the server batches several tokens into each SSE event
(~3.3 here, tracking the EAGLE accept length), so the counters below are
per-CHUNK, not per-token -- `chunk_lat_p50_ms` is roughly 3x the true
inter-token latency. Verified 2026-08-08: streamed and non-streamed requests
finish in the same wall time (3.05s vs 3.04s for 200 tokens), so there is no
streaming penalty, only a counting difference. Absolute values are therefore
not comparable to bench_worker.py's `decode_tok_s`; comparisons BETWEEN two
workers measured with this script are valid, which is all it is for.
"""
import argparse, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Explain how a write-ahead log guarantees durability.",
    "Describe the tradeoffs of hash joins versus merge joins.",
    "What causes false sharing and how do you fix it?",
    "Explain copy-on-write memory in fork().",
    "How does a bloom filter trade space for accuracy?",
    "Explain vectorized execution in columnar databases.",
]


def one(host, port, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "seed": 42,
        "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['SGLANG_API_KEY']}"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            chunk = raw[6:].strip()
            if chunk == b"[DONE]":
                break
            try:
                d = json.loads(chunk)
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
    total = time.perf_counter() - t0
    # Mean inter-CHUNK time after the first chunk (see MEASUREMENT NOTE).
    tpot = (total - ttft) / max(n - 1, 1) if ttft is not None else None
    return {"ttft": ttft, "total": total, "chunks": n, "tpot": tpot}


def rung(host, port, c, max_tokens, timeout):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda i: one(host, port, PROMPTS[i % len(PROMPTS)],
                                        max_tokens, timeout), range(c)))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["ttft"] is not None]
    toks = sum(r["chunks"] for r in ok)
    return {
        "concurrency": c,
        "wall_s": round(wall, 3),
        "chunk_rate_s": round(toks / wall, 2),
        "ttft_p50": round(statistics.median(r["ttft"] for r in ok), 4),
        "chunk_lat_p50_ms": round(statistics.median(r["tpot"] for r in ok) * 1000, 3),
        "chunk_lat_mean_ms": round(statistics.mean(r["tpot"] for r in ok) * 1000, 3),
        "n_ok": len(ok),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--rungs", default="1,2,4,6,8,12")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    out = {"label": a.label, "host": a.host, "rungs": []}
    for c in [int(x) for x in a.rungs.split(",")]:
        r = rung(a.host, a.port, c, a.max_tokens, a.timeout)
        out["rungs"].append(r)
        print(f"  c={r['concurrency']:<3} chunk/s={r['chunk_rate_s']:<8} "
              f"ttft_p50={r['ttft_p50']:<8} chunk_lat_p50={r['chunk_lat_p50_ms']} ms",
              file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
