#!/usr/bin/env python3
"""Does an agent context survive a stream of one-off requests? (drained worker)

Production mix: a batch loop sends ~92% of requests (~3K tokens, one-off),
while agent keys resend a growing ~65K context every turn. Whether the agent's
next turn is a cache hit depends on two tiers that evict independently in
UnifiedRadixCache:

  - full-attention KV: --radix-eviction-policy (lru default, slru, ...)
    device pool + HiCache host pool (--hicache-ratio x device tokens)
  - GDN (mamba) state: always LRU, whatever the policy (components/mamba.py);
    device slots = --max-mamba-cache-size, host states = slots x hicache-ratio.
    A KV prefix is only usable up to a node that still has a GDN state.

The replay: AGENTS sessions with a BASE-token context each, a turn appends
APPEND tokens. Between two turns of the same agent, GAP x AGENTS one-off
requests (shared BATCH_PREFIX + unique BATCH_BODY tokens) are inserted.
Phase "near" uses a small gap, phase "far" a large one. Token ids are random
(content does not matter to the cache), fixed by seed, so every arm replays
the identical sequence. Reports cached tokens per agent turn.

  ./tuning/bench/run_eval.sh r1 cache_replay <label>
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from evalkit import Worker

VOCAB_LO, VOCAB_HI = 1000, 150000   # plain tokens, clear of the special-token range


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def gen(w, ids, max_new=8):
    t = time.time()
    st, txt = w.post_raw("/generate", {"input_ids": ids,
                                       "sampling_params": {"temperature": 0, "max_new_tokens": max_new}},
                         timeout=1800)
    secs = round(time.time() - t, 3)
    if st != 200:
        return {"error": f"{st}: {txt[:200]}", "secs": secs}
    mi = json.loads(txt)["meta_info"]
    out = {"secs": secs, "prompt_tokens": mi.get("prompt_tokens"), "cached": mi.get("cached_tokens")}
    for k, v in mi.items():          # cached_tokens_details (device/host) when reported
        if k.startswith("cached_tokens_") and v is not None:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--base", type=int, default=40000)
    ap.add_argument("--append", type=int, default=1000)
    ap.add_argument("--batch-prefix", type=int, default=500)
    ap.add_argument("--batch-body", type=int, default=2500)
    ap.add_argument("--gaps", default="near:8,far:30", help="phase:one-offs-per-agent-turn,...")
    ap.add_argument("--rounds", type=int, default=3, help="measured agent rounds per phase")
    ap.add_argument("--concurrency", type=int, default=4, help="one-offs in flight")
    a = ap.parse_args()

    rng = random.Random(20260919)
    toks = lambda n: [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]
    w = Worker(a.host, a.port, consumer=f"replay-{a.label}")
    if not w.flush():
        raise RuntimeError("flush_cache never answered 'Cache flushed.'")

    agents = [toks(a.base) for _ in range(a.agents)]
    batch_prefix = toks(a.batch_prefix)
    res = {"label": a.label, "args": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "turns": [], "oneoffs": {"n": 0, "cached_sum": 0, "prompt_sum": 0, "errors": 0}}

    def agent_turn(i, phase, rnd):
        agents[i] += toks(a.append)
        r = gen(w, agents[i])
        r.update({"agent": i, "phase": phase, "round": rnd, "context": len(agents[i])})
        res["turns"].append(r)
        log(f"{phase:5s} r{rnd} agent {i}: ctx {len(agents[i]):>6} cached {r.get('cached')} "
            f"{ {k: v for k, v in r.items() if k.startswith('cached_tokens_')} } {r['secs']}s")

    def oneoffs(n):
        bodies = [batch_prefix + toks(a.batch_body) for _ in range(n)]
        with ThreadPoolExecutor(a.concurrency) as ex:
            for r in ex.map(gen, [w] * n, bodies):
                o = res["oneoffs"]
                o["n"] += 1
                o["errors"] += 1 if r.get("error") else 0
                o["cached_sum"] += r.get("cached") or 0
                o["prompt_sum"] += r.get("prompt_tokens") or 0

    # round 0: every agent's context is prefilled cold
    for i in range(a.agents):
        agent_turn(i, "cold", 0)
    rnd = 0
    for spec in a.gaps.split(","):
        phase, gap = spec.split(":")
        for _ in range(a.rounds):
            rnd += 1
            for i in range(a.agents):
                oneoffs(int(gap))
                agent_turn(i, phase, rnd)

    summ = {}
    for t in res["turns"]:
        s = summ.setdefault(t["phase"], {"turns": 0, "prompt": 0, "cached": 0, "secs": 0.0, "full_hits": 0})
        s["turns"] += 1
        s["prompt"] += t.get("prompt_tokens") or 0
        s["cached"] += t.get("cached") or 0
        s["secs"] += t["secs"]
        s["full_hits"] += 1 if (t.get("cached") or 0) >= t["context"] - a.append - 64 else 0
    for s in summ.values():
        s["hit_ratio"] = round(s["cached"] / s["prompt"], 3) if s["prompt"] else None
        s["mean_secs"] = round(s["secs"] / s["turns"], 2)
    res["summary"] = summ
    o = res["oneoffs"]
    log(f"summary {json.dumps(summ)} oneoffs n={o['n']} errors={o['errors']} "
        f"hit={o['cached_sum'] / max(o['prompt_sum'], 1):.3f}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
