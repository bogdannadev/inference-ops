#!/usr/bin/env python3
"""Drive arms A2/A3 (test-only drift injector, see deploy/enginepatch-test-*).

Two long decodes hold KV; then one 20K-token prompt C, which fits comfortably
at admission. On C's first continuing chunk the injector takes all free KV but
3,000 tokens for 20 s. Stock v0.5.21 is expected to clamp one chunk to the
remainder, then ask for a full 4096-token chunk with nothing free and die
("Prefill out of memory"); patch 3 is expected to park C and finish it once
the KV comes back.

Runs inside the replica container (stdlib only):
  docker exec -i -e KEY=... qwen36-27b-r1 python3 - < repro_drift.py
"""
import json, os, random, re, sys, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://localhost:8002")
HDR = {"Authorization": f"Bearer {os.environ['KEY']}", "Content-Type": "application/json"}
rng = random.Random()  # fresh ids every run: no radix/HiCache hits from earlier runs


def call(path, body=None, timeout=900):
    req = urllib.request.Request(f"{BASE}{path}", headers=HDR,
                                 data=None if body is None else json.dumps(body).encode())
    return urllib.request.urlopen(req, timeout=timeout).read()


def metric(name):
    txt = call("/metrics", timeout=10).decode()
    m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", txt, re.M)
    return float(m.group(1)) if m else None


results = {}


def generate(tag, n_in, n_out):
    body = {"input_ids": [rng.randrange(1000, 150000) for _ in range(n_in)],
            "sampling_params": {"max_new_tokens": n_out, "ignore_eos": True, "temperature": 0.0}}
    t0 = time.time()
    try:
        meta = json.loads(call("/generate", body)).get("meta_info", {})
        results[tag] = {"ok": True, "in": n_in, "out": meta.get("completion_tokens"), "s": round(time.time() - t0, 1)}
    except Exception as e:
        results[tag] = {"ok": False, "in": n_in, "err": f"{type(e).__name__}: {str(e)[:120]}", "s": round(time.time() - t0, 1)}
    print(json.dumps({tag: results[tag]}), flush=True)


pool = metric("sglang:max_total_num_tokens")
if not pool or pool > 100_000:
    sys.exit("refusing: pool is not the small trial pool (is this a trial arm?)")
call("/flush_cache", {})
print(json.dumps({"pool": pool, "usage_after_flush": metric("sglang:token_usage")}), flush=True)

ds = [threading.Thread(target=generate, args=(f"D{j}", 16_000, 6_000)) for j in range(2)]
for t in ds:
    t.start()
t_end = time.time() + 240
while time.time() < t_end and (metric("sglang:num_running_reqs") or 0) < 2:
    time.sleep(0.5)
time.sleep(3)
generate("C", 20_000, 16)
for t in ds:
    t.join(timeout=900)
time.sleep(2)
try:
    call("/flush_cache", {})
    leak = metric("sglang:token_usage")
except Exception as e:
    leak = f"unavailable ({type(e).__name__})"
ok = all(r["ok"] for r in results.values())
print(json.dumps({"verdict": "all requests completed" if ok else "REQUEST FAILURES",
                  "usage_after_final_flush": leak}), flush=True)
