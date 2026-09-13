#!/usr/bin/env python3
"""HiCache L2 functional probe: does an EVICTED prefix come back from host RAM?

Run against a DRAINED worker only (it flushes the cache and fills the device
pool on purpose).

  1. flush, then send prompt A (~40K tokens) cold            -> cold TTFT
  2. send A again                                            -> device hit
  3. send 4 unrelated ~45K prompts, enough to push A out of the 169K pool
  4. send A again                                            -> host hit?

Pass = step 4 reports cached_tokens_details.host > 0 and a TTFT far below the
cold one. Without HiCache step 4 is a full recompute, i.e. step 1 again.
"""
import argparse
import json
import random
import sys

from evalkit import Worker, filler, metric_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-words", type=int, default=26000)
    ap.add_argument("--evict-words", type=int, default=30000)
    ap.add_argument("--evictors", type=int, default=4)
    a = ap.parse_args()

    w = Worker(a.host, a.port, consumer="hicache-probe")
    rng = random.Random(20260913)
    res = {"label": a.label, "steps": []}

    if not w.flush():
        print(json.dumps({"error": "flush_cache never answered 'Cache flushed.'"}))
        sys.exit(1)
    m0 = w.metrics()

    doc_a = f"Session nonce {rng.getrandbits(64):x}.\n" + filler(rng, a.prompt_words)
    q = "\n\nReply with the single word OK."
    params = dict(max_tokens=4, temperature=0, return_cached_tokens_details=True,
                  chat_template_kwargs={"enable_thinking": False})

    def step(name, text):
        r = w.chat([{"role": "user", "content": text}], **params)
        u = r.get("usage") or {}
        row = {"step": name, "ttft_s": r.get("ttft_s"), "prompt_tokens": u.get("prompt_tokens"),
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "cached_details": u.get("cached_tokens_details")
               or (u.get("prompt_tokens_details") or {}).get("cached_tokens_details"),
               "error": r.get("error")}
        res["steps"].append(row)
        print(json.dumps(row), file=sys.stderr)
        return row

    step("A_cold", doc_a + q)
    step("A_device_hit", doc_a + q)
    for i in range(a.evictors):
        step(f"evictor_{i}", f"Unrelated document {i} {rng.getrandbits(64):x}.\n"
             + filler(rng, a.evict_words) + q)
    final = step("A_after_eviction", doc_a + q)

    m1 = w.metrics()
    res["delta"] = {
        "evicted_tokens": metric_sum(m1, "sglang:evicted_tokens_total") - metric_sum(m0, "sglang:evicted_tokens_total"),
        "cached_host": metric_sum(m1, "sglang:cached_tokens_total", extra_label='cache_source="host"')
        - metric_sum(m0, "sglang:cached_tokens_total", extra_label='cache_source="host"'),
        "cached_device": metric_sum(m1, "sglang:cached_tokens_total", extra_label='cache_source="device"')
        - metric_sum(m0, "sglang:cached_tokens_total", extra_label='cache_source="device"'),
    }
    cold = res["steps"][0]["ttft_s"]
    res["verdict"] = {
        "cold_ttft_s": cold,
        "after_eviction_ttft_s": final["ttft_s"],
        "speedup": round(cold / final["ttft_s"], 1) if final["ttft_s"] else None,
        "host_hit": bool(res["delta"]["cached_host"] > 0),
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
