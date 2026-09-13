#!/usr/bin/env python3
"""Worst-case GPU memory probe for a drained worker.

Production requests reach 168,998 tokens (gateway.requests, 2026-09). A config
that boots with little activation headroom can still OOM on the first long
prefill, which is exactly how the first DFlash2 boot died (2026-09-13). So this
sends a near-context-length cold prompt TOGETHER with three mid-size requests
that keep decoding, twice, and reports errors. Pair it with a log check for
OutOfMemoryError on the worker.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from evalkit import Worker, filler

WORDS_PER_TOKEN = 26000 / 44993   # measured on this filler with the Qwen3.8 tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--big-tokens", type=int, default=160_000)
    ap.add_argument("--side-tokens", type=int, default=30_000)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    w = Worker(a.host, a.port, consumer="mem-stress")
    rng = random.Random(777)
    rows = []
    for rd in range(a.rounds):
        big = f"Round {rd} big {rng.getrandbits(64):x}\n" + filler(rng, int(a.big_tokens * WORDS_PER_TOKEN))
        sides = [f"Round {rd} side {i} {rng.getrandbits(64):x}\n" + filler(rng, int(a.side_tokens * WORDS_PER_TOKEN))
                 for i in range(3)]
        q = "\n\nSummarise the words above in one long paragraph."
        with ThreadPoolExecutor(4) as ex:
            futs = [ex.submit(w.chat, [{"role": "user", "content": big + q}], max_tokens=256, temperature=0.7,
                              chat_template_kwargs={"enable_thinking": False})]
            futs += [ex.submit(w.chat, [{"role": "user", "content": s + q}], max_tokens=1024, temperature=0.7,
                               ignore_eos=True, chat_template_kwargs={"enable_thinking": False}) for s in sides]
            for n, f in zip(["big", "side0", "side1", "side2"], futs):
                r = f.result()
                rows.append({"round": rd, "req": n, "error": r.get("error"),
                             "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                             "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                             "ttft_s": r.get("ttft_s"), "e2e_s": r.get("e2e_s")})
    print(json.dumps({"label": a.label, "rows": rows,
                      "errors": [x for x in rows if x["error"] or not x["completion_tokens"]]}, indent=1))


if __name__ == "__main__":
    main()
