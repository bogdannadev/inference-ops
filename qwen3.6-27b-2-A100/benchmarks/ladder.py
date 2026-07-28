#!/usr/bin/env python3
"""
Concurrency ladder through the ROUTER.

Walks concurrency 1,2,4,6,8,12 and reports, per rung:
  - success / failure counts and the status codes of failures
  - aggregate completion tokens per second   (the throughput number)
  - per-request latency mean / p50 / max     (the SLA number)
  - how work split across r0 / r1            (balance check)

Prompts are DISJOINT per request so prefix-cache affinity cannot pin a rung
onto one worker and make it look artificially slow.

The point of the ladder is to find the knee: throughput climbs while latency
stays flat, then latency starts climbing without throughput following. That
knee is the concurrency the SLA can actually promise.
"""
import argparse, http.client, json, os, time
from concurrent.futures import ThreadPoolExecutor

API_KEY = os.environ["SGLANG_API_KEY"]

TOPICS = [
    "binary search trees", "TCP congestion control", "database normalisation",
    "garbage collection", "unicode normalisation", "public key exchange",
    "columnar storage", "consistent hashing", "write-ahead logging",
    "vector clocks", "bloom filters", "copy-on-write snapshots",
    "lock-free queues", "content addressing", "rate limiting", "leader election",
]


def verify_calls(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode("utf-8", "ignore")
    except Exception:
        return None
    finally:
        conn.close()
    for line in text.splitlines():
        if line.startswith("sglang:spec_verify_calls_total"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return None


def one(idx, max_tokens, timeout):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content":
                      f"Explain {TOPICS[idx % len(TOPICS)]} in about 120 words."}],
        "temperature": 0, "max_tokens": max_tokens,
    }
    body = json.dumps(payload)
    conn = http.client.HTTPConnection("qwen36-27b-router", 8000, timeout=timeout)
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
                    "body": raw[:120].decode("utf-8", "ignore")}
        usage = (json.loads(raw).get("usage") or {})
        return {"status": 200, "latency_s": round(dt, 3),
                "completion_tokens": usage.get("completion_tokens")}
    except Exception as e:
        return {"status": None, "latency_s": round(time.monotonic() - t0, 3),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()


def rung(n, max_tokens, timeout):
    b0, b1 = verify_calls("qwen36-27b-r0", 8001), verify_calls("qwen36-27b-r1", 8002)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda i: one(i, max_tokens, timeout), range(n)))
    wall = time.monotonic() - t0
    time.sleep(1.0)  # let metrics settle
    a0, a1 = verify_calls("qwen36-27b-r0", 8001), verify_calls("qwen36-27b-r1", 8002)

    ok = [r for r in results if r["status"] == 200]
    lats = sorted(r["latency_s"] for r in ok)
    toks = sum(r.get("completion_tokens") or 0 for r in ok)
    fails = {}
    for r in results:
        if r["status"] != 200:
            k = str(r["status"])
            fails[k] = fails.get(k, 0) + 1

    return {
        "concurrency": n,
        "wall_time_s": round(wall, 3),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "failure_status_counts": fails,
        "aggregate_tok_s": round(toks / wall, 2) if wall else None,
        "latency_mean_s": round(sum(lats) / len(lats), 3) if lats else None,
        "latency_p50_s": lats[len(lats) // 2] if lats else None,
        "latency_max_s": lats[-1] if lats else None,
        "split_r0": None if None in (a0, b0) else round(a0 - b0, 1),
        "split_r1": None if None in (a1, b1) else round(a1 - b1, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="1,2,4,6,8,12")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    out = {"rungs": []}
    for n in [int(x) for x in args.rungs.split(",")]:
        r = rung(n, args.max_tokens, args.timeout)
        out["rungs"].append(r)
        print(json.dumps(r), flush=True)
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
