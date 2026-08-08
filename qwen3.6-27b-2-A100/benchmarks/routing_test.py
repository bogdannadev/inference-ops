#!/usr/bin/env python3
"""Router policy test: how does traffic that SHARES A PREFIX get distributed?

benchmarks/ladder.py deliberately uses disjoint prompts so prefix affinity
cannot pin a rung onto one worker. That is the right call for a throughput
ladder, but it is exactly the wrong shape for evaluating `prefix_hash`: real
coding-agent traffic is many conversations sharing one long system prompt.

This sends two workloads and diffs per-replica metrics around each:

  shared  -- N requests sharing a long common prefix, differing only in the
             tail. This is the OpenCode shape. prefix_hash should pin these
             to one replica (good for cache, bad for balance).
  disjoint-- N requests with no common prefix. Control.

Reports per replica: requests served, cache hit rate, and the resulting
balance. A policy that pins everything onto one replica shows up as
requests=(N,0).
"""
import argparse, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["SGLANG_API_KEY"]
REPLICAS = {"r0": ("qwen36-27b-r0", 8001), "r1": ("qwen36-27b-r1", 8002)}

# ~1200 tokens of shared context, standing in for a system prompt + repo map.
SHARED_PREFIX = ("You are a coding assistant working in a large Python "
                 "monorepo. Follow these conventions strictly. ") + " ".join(
    f"Module {i} lives under src/pkg{i}/ and exposes a public API with "
    f"functions load{i}(), save{i}(), and validate{i}(); it must never import "
    f"from module {i+1} directly, only through the registry." for i in range(40))

TAILS = [
    "Which module owns persistence?", "How do I add a new module?",
    "What is the import rule?", "Where does validation live?",
    "How many modules are there?", "What does the registry do?",
    "Can module 3 import module 4?", "Name the public API functions.",
    "Summarise the conventions.", "What is under src/pkg7/?",
    "Explain the layering rule.", "Which functions does module 12 expose?",
]

DISJOINT = [
    "Explain binary search trees in about 80 words.",
    "Explain TCP congestion control in about 80 words.",
    "Explain database normalisation in about 80 words.",
    "Explain garbage collection in about 80 words.",
    "Explain bloom filters in about 80 words.",
    "Explain vector clocks in about 80 words.",
    "Explain consistent hashing in about 80 words.",
    "Explain write-ahead logging in about 80 words.",
    "Explain lock-free queues in about 80 words.",
    "Explain leader election in about 80 words.",
    "Explain rate limiting in about 80 words.",
    "Explain columnar storage in about 80 words.",
]


def metrics(host, port):
    """Per-replica request count and cached-token totals."""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/metrics")
        txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return None
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key, meas in (("sglang:prompt_tokens_histogram_count", "reqs"),
                          ("sglang:prompt_tokens_histogram_sum", "prompt_tokens"),
                          ("sglang:cached_tokens_histogram_sum", "cached_tokens")):
            if line.startswith(key):
                try:
                    out[meas] = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    pass
    return out


def snap():
    return {r: metrics(*hp) for r, hp in REPLICAS.items()}


def one(prompt, max_tokens, timeout):
    body = json.dumps({
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "http://qwen36-27b-router:8000/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        u = d.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        return {"ok": True, "lat": time.perf_counter() - t0,
                "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": det.get("cached_tokens"),
                "completion_tokens": u.get("completion_tokens")}
    except Exception as e:
        return {"ok": False, "lat": time.perf_counter() - t0, "err": str(e)[:120]}


def workload(name, prompts, conc, max_tokens, timeout):
    before = snap()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(lambda p: one(p, max_tokens, timeout), prompts))
    wall = time.perf_counter() - t0
    after = snap()

    delta = {}
    for r in REPLICAS:
        b, a = before.get(r) or {}, after.get(r) or {}
        delta[r] = {k: round(a.get(k, 0) - b.get(k, 0), 1)
                    for k in ("reqs", "prompt_tokens", "cached_tokens")}

    ok = [x for x in res if x["ok"]]
    tot = sum(d["reqs"] for d in delta.values()) or 1
    out = {
        "workload": name, "n": len(prompts), "concurrency": conc,
        "wall_s": round(wall, 2),
        "ok": len(ok), "failed": len(res) - len(ok),
        "lat_p50": round(statistics.median(x["lat"] for x in ok), 3) if ok else None,
        "lat_max": round(max(x["lat"] for x in ok), 3) if ok else None,
        "cached_tokens_reported": sum((x.get("cached_tokens") or 0) for x in ok),
        "prompt_tokens_reported": sum((x.get("prompt_tokens") or 0) for x in ok),
        "per_replica": delta,
        "balance": {r: f"{100*delta[r]['reqs']/tot:.0f}%" for r in REPLICAS},
    }
    errs = [x.get("err") for x in res if not x["ok"]]
    if errs:
        out["errors"] = errs[:3]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    shared = [SHARED_PREFIX + " QUESTION: " + TAILS[i % len(TAILS)] for i in range(a.n)]
    disjoint = [DISJOINT[i % len(DISJOINT)] for i in range(a.n)]

    out = {"label": a.label, "results": []}
    for name, prompts in (("shared_prefix", shared), ("disjoint", disjoint)):
        r = workload(name, prompts, a.concurrency, a.max_tokens, a.timeout)
        out["results"].append(r)
        print(f"  {name:14} reqs r0/r1={r['per_replica']['r0']['reqs']:.0f}/"
              f"{r['per_replica']['r1']['reqs']:.0f} balance={r['balance']} "
              f"cached={r['cached_tokens_reported']}/{r['prompt_tokens_reported']} "
              f"p50={r['lat_p50']}s max={r['lat_max']}s failed={r['failed']}",
              file=sys.stderr, flush=True)
    print(json.dumps(out, indent=2))
