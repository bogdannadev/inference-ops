#!/usr/bin/env python3
"""
Diagnostic for the instant-408 / worker-pinning behaviour.

Sends a burst of N concurrent requests with DISTINCT prompt prefixes, so that
cache-aware routing has no reason to pin them all onto a single worker, and
reports:
  - per-request status + timing
  - how the work actually distributed across r0 / r1 (via spec_verify_calls_total delta)

Run the same burst with --identical to reproduce the pinning case for contrast.
"""
import argparse, http.client, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["SGLANG_API_KEY"]


def verify_calls(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode("utf-8", "ignore")
    except Exception as e:
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


def one(idx, prompt, max_tokens, timeout):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Length": str(len(body)),
    }
    conn = http.client.HTTPConnection("qwen36-27b-router", 8000, timeout=timeout)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        dt = time.monotonic() - t0
        return {
            "idx": idx,
            "status": resp.status,
            "elapsed_s": round(dt, 3),
            "body_len": len(raw),
            "body_snippet": raw[:160].decode("utf-8", "ignore") if resp.status != 200 else None,
        }
    except Exception as e:
        return {"idx": idx, "status": None, "elapsed_s": round(time.monotonic() - t0, 3),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--identical", action="store_true",
                    help="use the identical prompt for every request (reproduces pinning)")
    args = ap.parse_args()

    topics = ["oceans", "railways", "bread baking", "volcanoes", "jazz music", "beekeeping",
              "cartography", "glaciers", "typography", "sailing", "mycology", "astronomy"]

    if args.identical:
        prompts = ["Write a three-sentence summary of why distributed systems are hard."] * args.num
    else:
        prompts = [f"Write three distinct sentences about {topics[i % len(topics)]}." for i in range(args.num)]

    before = {"r0": verify_calls("qwen36-27b-r0", 8001), "r1": verify_calls("qwen36-27b-r1", 8002)}

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.num) as ex:
        futs = [ex.submit(one, i, p, args.max_tokens, args.timeout) for i, p in enumerate(prompts)]
        results = [f.result() for f in as_completed(futs)]
    wall = time.monotonic() - t0

    after = {"r0": verify_calls("qwen36-27b-r0", 8001), "r1": verify_calls("qwen36-27b-r1", 8002)}

    results.sort(key=lambda r: r["elapsed_s"])
    out = {
        "mode": "identical" if args.identical else "distinct",
        "num": args.num,
        "wall_time_s": round(wall, 3),
        "status_counts": {},
        "results": results,
        "verify_calls_delta": {
            k: (None if before[k] is None or after[k] is None else after[k] - before[k])
            for k in ("r0", "r1")
        },
    }
    for r in results:
        key = str(r.get("status"))
        out["status_counts"][key] = out["status_counts"].get(key, 0) + 1

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
