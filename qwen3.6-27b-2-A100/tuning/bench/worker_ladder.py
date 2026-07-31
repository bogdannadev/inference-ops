#!/usr/bin/env python3
"""
Concurrency ladder against ONE worker, bypassing the router.

Why this exists rather than benchmarks/ladder.py:
  ladder.py goes through the router, which round-robins across both replicas.
  That is correct for measuring the deployment, but it CANNOT measure a
  per-replica config change -- and with an admission cap of 4 across the pair
  it could not measure much at all: every rung above 4 was a wave replay of
  c=4 (see tuning/docs/TUNING_PLAN.md). Driving one worker directly removes
  both problems.

Why not benchmarks/bench_worker.py:
  That one hits a worker directly but is strictly SEQUENTIAL. The whole point
  of the current experiment is --max-running-requests, which is invisible to a
  single-stream benchmark.

Correctness: prompts are disjoint per request, so a warm prefix cache cannot
inflate one rung. Rung order is fixed and a warmup rung is discarded, so the
first-request CUDA-graph/autotune cost does not land on a measured rung.

Usage:
  SGLANG_API_KEY=... python3 worker_ladder.py \
      --host qwen36-27b-r1 --port 8002 --label r1_treatment \
      --out ../results/r1_treatment.json
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import Sampler, preflight  # noqa: E402

API_KEY = os.environ["SGLANG_API_KEY"]

# 24 disjoint topics -- enough for the widest rung without repeating a prompt
# inside a rung (repeats would share a prefix and skew the cache).
TOPICS = [
    "binary search trees", "TCP congestion control", "database normalisation",
    "garbage collection", "unicode normalisation", "public key exchange",
    "columnar storage", "consistent hashing", "write-ahead logging",
    "vector clocks", "bloom filters", "copy-on-write snapshots",
    "lock-free queues", "content addressing", "rate limiting", "leader election",
    "B+ tree splits", "MVCC snapshots", "raft log compaction", "CRDT merges",
    "hash join spilling", "adaptive radix trees", "gossip membership",
    "read repair in quorums",
]


def one(idx, host, port, max_tokens, timeout):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content":
                      f"Explain {TOPICS[idx % len(TOPICS)]} in about 120 words."}],
        "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": max_tokens,
    }
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Length": str(len(body)),
        })
        r = conn.getresponse()
        raw = r.read()
        dt = time.monotonic() - t0
        if r.status != 200:
            return {"status": r.status, "latency_s": round(dt, 3),
                    "body": raw[:160].decode("utf-8", "ignore")}
        usage = (json.loads(raw).get("usage") or {})
        return {"status": 200, "latency_s": round(dt, 3),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens")}
    except Exception as e:
        return {"status": None, "latency_s": round(time.monotonic() - t0, 3),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()


def rung(n, host, port, max_tokens, timeout, sampler):
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(
            lambda i: one(i, host, port, max_tokens, timeout), range(n)))
    t1 = time.monotonic()
    wall = t1 - t0

    ok = [r for r in results if r["status"] == 200]
    lats = sorted(r["latency_s"] for r in ok)
    toks = sum(r.get("completion_tokens") or 0 for r in ok)
    fails = {}
    for r in results:
        if r["status"] != 200:
            k = str(r["status"])
            fails[k] = fails.get(k, 0) + 1

    def pct(p):
        if not lats:
            return None
        return lats[min(len(lats) - 1, int(round((p / 100) * (len(lats) - 1))))]

    out = {
        "concurrency": n,
        "wall_time_s": round(wall, 3),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "failure_status_counts": fails,
        "completion_tokens_total": toks,
        # tokens / wall -- true throughput, same definition as benchmarks/ladder.py
        "aggregate_tok_s": round(toks / wall, 2) if wall else None,
        "per_stream_tok_s": round(toks / wall / n, 2) if wall and n else None,
        "latency_mean_s": round(statistics.fmean(lats), 3) if lats else None,
        "latency_p50_s": pct(50),
        "latency_p95_s": pct(95),
        "latency_max_s": lats[-1] if lats else None,
        # Bimodality is the queueing fingerprint: if max is ~2x p50, requests
        # are being admitted in waves rather than run concurrently.
        "latency_max_over_p50": (round(lats[-1] / pct(50), 2)
                                 if lats and pct(50) else None),
    }
    if sampler:
        out["telemetry"] = Sampler.summarise(sampler.slice(t0, t1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rungs", default="1,2,4,6,8,12")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the whole ladder N times; report each pass")
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if not args.skip_preflight and not args.no_telemetry:
        ok, rep = preflight(args.host, args.port)
        if not ok:
            print(json.dumps(rep, indent=2), file=sys.stderr)
            print("\nPreflight INCOMPLETE. Re-run with --no-telemetry to "
                  "benchmark anyway, or fix the exporter first.", file=sys.stderr)
            return 1

    sampler = None
    if not args.no_telemetry:
        sampler = Sampler(args.host, args.port)
        sampler.start()

    out = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "max_tokens": args.max_tokens,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passes": [],
    }

    try:
        # Warmup: first request pays CUDA-graph replay setup and any lazy
        # autotune. Discarded so it cannot be attributed to rung 1.
        print("# warmup (discarded)", file=sys.stderr, flush=True)
        rung(2, args.host, args.port, args.max_tokens, args.timeout, None)

        rungs = [int(x) for x in args.rungs.split(",")]
        for p in range(args.repeat):
            passrows = []
            for n in rungs:
                r = rung(n, args.host, args.port, args.max_tokens,
                         args.timeout, sampler)
                r["pass"] = p
                passrows.append(r)
                print(json.dumps(r), flush=True)
            out["passes"].append(passrows)
    finally:
        if sampler:
            sampler.stop()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
