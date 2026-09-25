#!/usr/bin/env python3
"""HiCache L3 functional probe: does a prefix come back from the storage tier,
KV pages AND GDN state, with a correct answer and an unchanged DFlash accept length?

Run against DRAINED workers only (it flushes their caches).

/flush_cache resets L1 (device) and L2 (host) but not L3 (HiCacheFile keeps its
files), so "flush, then replay" is a pure L3 test with no restart.

  1. flush, send doc A (two numeric needles, ~--prompt-words) cold  -> cold TTFT
  2. wait for write-through to reach L3, flush again (L3 survives)
  3. send A again                                   -> storage hit? same codes?
  4. continue A (400 tokens), then a cold control doc B, same continuation
                                                    -> accept length A vs B
With --seed-host/--seed-port, step 1 runs on the SEED worker and steps 2-4 on
--host: a cross-replica test (both must mount the same L3 directory).

Pass = cached_tokens_details shows storage tokens on step 3, both codes
correct, TTFT far below cold, accept length within 10% of the control.
A wrong code after an L3 hit means the restored KV or GDN state is corrupt.
"""
import argparse
import json
import random
import sys
import time

from evalkit import Worker, filler, metric_sum

PARAMS = dict(temperature=0, return_cached_tokens_details=True,
              chat_template_kwargs={"enable_thinking": False})


def spec_counts(w):
    m = w.metrics()
    return (metric_sum(m, "sglang:generation_tokens_total"),
            metric_sum(m, "sglang:spec_verify_calls_total"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--seed-host")
    ap.add_argument("--seed-port", type=int)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-words", type=int, default=45000)
    ap.add_argument("--settle-s", type=float, default=20)
    a = ap.parse_args()

    hit = Worker(a.host, a.port, consumer="l3-probe")
    seed = Worker(a.seed_host or a.host, a.seed_port or a.port, consumer="l3-probe")
    rng = random.Random()
    codes = [rng.randrange(10**7, 10**8) for _ in range(2)]
    half = a.prompt_words // 2
    doc_a = (f"Session {rng.getrandbits(64):x}. The first code is {codes[0]}.\n"
             + filler(rng, half) + f"\nThe second code is {codes[1]}.\n" + filler(rng, half))
    ask = "\n\nWhat are the first and second codes? Reply with the two numbers only, separated by one space."
    res = {"label": a.label, "codes": codes, "cross_replica": bool(a.seed_host),
           "steps": []}

    def step(w, name, messages, **extra):
        r = w.chat(messages, **{**PARAMS, "max_tokens": 32, **extra})
        u = r.get("usage") or {}
        pd = u.get("prompt_tokens_details") or {}
        row = {"step": name, "ttft_s": r.get("ttft_s"), "prompt_tokens": u.get("prompt_tokens"),
               "cached": pd.get("cached_tokens"),
               "cached_details": u.get("cached_tokens_details") or pd.get("cached_tokens_details"),
               "content": (r.get("content") or "")[:120], "completion_tokens": u.get("completion_tokens"),
               "error": r.get("error")}
        res["steps"].append(row)
        print(json.dumps(row), file=sys.stderr)
        return r, row

    for w in {id(hit): hit, id(seed): seed}.values():
        if not w.flush():
            print(json.dumps({"error": "flush_cache never answered 'Cache flushed.'"}))
            sys.exit(1)

    msgs_a = [{"role": "user", "content": doc_a + ask}]
    _, cold = step(seed, "A_cold_seed", msgs_a)
    time.sleep(a.settle_s)                       # write-through L1 -> L2 -> L3 is async
    for w in {id(hit): hit, id(seed): seed}.values():
        w.flush()
    r_hit, hot = step(hit, "A_after_flush_L3", msgs_a)

    # Decode on the restored prefix vs a cold control of the same shape.
    cont = "Now write about 300 words on why numeric codes get mixed up. No lists."
    msgs_a2 = msgs_a + [{"role": "assistant", "content": r_hit.get("content") or ""},
                        {"role": "user", "content": cont}]
    g0, v0 = spec_counts(hit)
    step(hit, "A_continue_on_L3_prefix", msgs_a2, max_tokens=400)
    g1, v1 = spec_counts(hit)
    doc_b = f"Session {rng.getrandbits(64):x}.\n" + filler(rng, a.prompt_words)
    msgs_b = [{"role": "user", "content": doc_b + "\n\n" + cont}]
    step(hit, "B_cold_control", msgs_b, max_tokens=400)
    g2, v2 = spec_counts(hit)

    acc_a = (g1 - g0) / (v1 - v0) if v1 > v0 else None
    acc_b = (g2 - g1) / (v2 - v1) if v2 > v1 else None
    want = f"{codes[0]} {codes[1]}"
    res["verdict"] = {
        "cold_ttft_s": cold["ttft_s"], "l3_ttft_s": hot["ttft_s"],
        "speedup": round(cold["ttft_s"] / hot["ttft_s"], 1) if hot["ttft_s"] else None,
        "cold_answer_ok": want in (cold["content"] or ""),
        "l3_answer_ok": want in (hot["content"] or ""),
        "l3_cached_details": hot["cached_details"],
        "accept_len_on_l3_prefix": round(acc_a, 2) if acc_a else None,
        "accept_len_cold_control": round(acc_b, 2) if acc_b else None,
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
