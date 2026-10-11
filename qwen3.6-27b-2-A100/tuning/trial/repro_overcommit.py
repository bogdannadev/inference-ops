#!/usr/bin/env python3
"""Reproduce the 2026-10-09 "Prefill out of memory" scheduler crash on a
DRAINED replica booted with a small pool (arms A0/A1, --max-total-tokens 65536,
--schedule-conservativeness 0.05).

Mechanism (patch 3 in deploy/enginepatch-next/sitecustomize.py): long
decodes hold KV and keep growing; a long prompt C is admitted with little
slack and prefilled in 4096-token chunks. When a continuation chunk exhausts the budget, stock
v0.5.21 asks for a FULL chunk on the next round and the allocator raises.

Runs inside the replica container (stdlib only):
  docker exec -i -e KEY=... qwen36-27b-r1 python3 - < repro_overcommit.py
Prints one JSON line per request and a final verdict line. Exit 0 always; the
verdict is read together with the container log.
"""
import json, os, random, re, sys, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8002")
KEY = os.environ["KEY"]
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
rng = random.Random(20261009)
# decoders must outlive C's whole prefill, or C waits for them and runs roomy
DEC_OUT = int(os.environ.get("DEC_OUT", "20000"))
SLACKS = [int(x) for x in os.environ.get("SLACKS", "600,300").split(",")]


def ids(n):  # plain-vocabulary ids: no special/added tokens (those sit above 248000)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def metric(name):
    txt = urllib.request.urlopen(urllib.request.Request(f"{BASE}/metrics", headers=HDR), timeout=10).read().decode()
    m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", txt, re.M)
    return float(m.group(1)) if m else None


results = {}


def generate(tag, n_in, n_out):
    body = {"input_ids": ids(n_in), "sampling_params": {
        "max_new_tokens": n_out, "ignore_eos": True, "temperature": 0.0}}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"{BASE}/generate", data=json.dumps(body).encode(), headers=HDR), timeout=900)
        out = json.loads(r.read())
        meta = out.get("meta_info", {})
        results[tag] = {"ok": True, "in": n_in, "out": meta.get("completion_tokens"),
                        "finish": (meta.get("finish_reason") or {}).get("type"), "s": round(time.time() - t0, 1)}
    except Exception as e:
        results[tag] = {"ok": False, "in": n_in, "err": f"{type(e).__name__}: {str(e)[:120]}",
                        "s": round(time.time() - t0, 1)}
    print(json.dumps({tag: results[tag]}), flush=True)


pool = metric("sglang:max_total_num_tokens")
print(json.dumps({"pool": pool}), flush=True)
if not pool or pool > 100_000:
    sys.exit("refusing: pool is not the small trial pool (is this a trial arm?)")

# One trial: three decodes (D0-D2) hold KV and grow; then a long prompt C sized
# to fit with only `slack` tokens to spare. The decodes grow by a few tokens per
# chunk round; once that growth exceeds the slack, a continuation chunk gets
# clamped to the last of the budget and the NEXT round sees it exhausted. Stock
# code then asks for a full 4096-token chunk with ~no free KV and the
# scheduler dies; the patched code clamps or parks.
def trial(n, slack):
    ds = [threading.Thread(target=generate, args=(f"t{n}-D{j}", 10_000, DEC_OUT)) for j in range(3)]
    for t in ds:
        t.start()
    t_end = time.time() + 240
    while time.time() < t_end and (metric("sglang:num_running_reqs") or 0) < 3:
        time.sleep(0.5)
    time.sleep(4)
    usage = metric("sglang:token_usage") or 0.0
    free = int(pool * (1 - usage))
    c = free - slack
    print(json.dumps({"trial": n, "slack": slack, "usage": round(usage, 4), "free": free, "send": c}), flush=True)
    generate(f"t{n}-C", c, 16)
    for t in ds:
        t.join(timeout=900)
    return all(r["ok"] for k, r in results.items() if k.startswith(f"t{n}-"))


for n, slack in enumerate(SLACKS):
    if not trial(n, slack):
        break  # the scheduler is gone; nothing more to learn from this boot

ok = all(r["ok"] for r in results.values())
print(json.dumps({"verdict": "all requests completed" if ok else "REQUEST FAILURES"}), flush=True)
