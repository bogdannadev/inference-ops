#!/usr/bin/env python3
"""Greedy byte-identity gate: compare two workers token-for-token.

Gate for SAME-SHAPE changes only (per the July campaign rule). PR #32219 is a
scheduling/fusion change and claims bit-identical output, so any divergence
here is a stop condition -- not a near-tie flip.

  python3 byte_identity.py --host qwen36-27b-r0 --port 8001 --label new > a.json
  python3 byte_identity.py --host qwen36-27b-r1 --port 8002 --label old > b.json
  python3 byte_identity.py --compare a.json b.json
"""
import argparse, hashlib, json, os, sys, urllib.request

# Deliberately varied: reasoning, code, long-form prose, tool-ish, multilingual.
# 256 tokens with ignore_eos means every prompt exercises the full decode path
# including several EAGLE verify rounds, not just a handful of tokens.
PROMPTS = [
    "List the first 12 prime numbers, comma separated, nothing else.",
    "Write a Python function that merges two sorted lists. Explain the complexity.",
    "Explain why CAS loops can livelock under high contention.",
    "Summarize the tradeoffs between paged KV attention and linear attention.",
    "Translate to French, then to German: 'The cache was cold on the first request.'",
    "Write a bash one-liner that finds the 10 largest files under /var and explain it.",
    "What is the difference between a memory barrier and a compiler barrier?",
    "Describe step by step how a radix tree prefix cache serves a repeated prompt.",
]


def post(host, port, body, timeout=300):
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['SGLANG_API_KEY']}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run(args):
    out = {"label": args.label, "host": args.host, "results": []}
    for i, p in enumerate(PROMPTS):
        r = post(args.host, args.port, {
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": p}],
            # Fully deterministic decode. ignore_eos forces exactly max_tokens
            # so a divergence cannot hide behind an early stop.
            "temperature": 0, "top_p": 1, "seed": 42,
            "max_tokens": 256, "ignore_eos": True,
        })
        txt = r["choices"][0]["message"]["content"] or ""
        reasoning = r["choices"][0]["message"].get("reasoning_content") or ""
        full = reasoning + "\x00" + txt
        out["results"].append({
            "idx": i,
            "prompt": p,
            "sha256": hashlib.sha256(full.encode()).hexdigest(),
            "completion_tokens": r["usage"]["completion_tokens"],
            "cached_tokens": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": full,
        })
        print(f"  [{i}] {out['results'][-1]['sha256'][:16]}  "
              f"{out['results'][-1]['completion_tokens']} tok", file=sys.stderr)
    print(json.dumps(out, indent=2))


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    bad = 0
    for ra, rb in zip(a["results"], b["results"]):
        same = ra["sha256"] == rb["sha256"]
        if not same:
            bad += 1
            # Locate the first differing character so the failure is diagnosable
            # rather than just "they differ".
            ta, tb = ra["text"], rb["text"]
            j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]),
                     min(len(ta), len(tb)))
            print(f"[{ra['idx']}] DIVERGE at char {j}")
            print(f"      {a['label']}: ...{ta[max(0,j-60):j+60]!r}")
            print(f"      {b['label']}: ...{tb[max(0,j-60):j+60]!r}")
        else:
            print(f"[{ra['idx']}] identical  {ra['sha256'][:16]}  {ra['completion_tokens']} tok")
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(a['results'])-bad}/{len(a['results'])} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a)
