#!/usr/bin/env python3
"""Speculative-decoding gate for ONE drained worker: correctness first, then speed.

Written for the DFlash2 trial (2026-09-13) against two open upstream reports:
  #36548  DFlash2 corrupts context under concurrent requests
  #38009  DFlash2 greedy output diverges from target-only with thinking on
so the correctness half is built to catch exactly those, at our concurrency
(--max-running-requests 4), with thinking ON as production runs it.

Correctness (every request has an objectively checkable answer):
  kv     ~12K-token table of unique records, asked for one code. A wrong answer
         that equals a code from ANOTHER in-flight request's table is a LEAK.
  order  5 names, 4 "immediately before" clues -> unique ordering. Names are
         disjoint from the other in-flight order task, so a foreign name is a LEAK.
  arith  sum of 12 random integers.
  seq    300-term arithmetic sequence, checked exactly: a long output where one
         corrupted verify step shows as a wrong or missing term.
  determinism: the same greedy request twice, serially, must match exactly.

Speed (production-like sampling, ignore_eos so length is fixed):
  short  ~300-token prompt, 1024 generated, c=1 and c=4
  long   ~60K-token shared prefix (warmed), 1024 generated, c=1 and c=4
  accept length = generation_tokens / spec_verify_calls from the worker's own
  counters, filtered by this run's consumer label.
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from evalkit import Worker, filler, metric_sum

NAMES = ("Alder Brook Cedar Dover Elm Flint Gale Hollis Ivor Jett Kline Lark Moss Nash "
         "Orrin Pike Quinn Reed Sloan Tate Upton Vance Wade Yates Zane Blair Crane Drew "
         "Ellis Frost").split()
PROD = dict(temperature=0.7, top_p=0.95, top_k=20)
GREEDY = dict(temperature=0, top_p=1, top_k=1)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- tasks
def make_kv(rng, used_codes):
    recs = {}
    while len(recs) < 900:
        k = f"K-{rng.getrandbits(24):06x}"
        c = rng.randrange(10_000_000, 99_999_999)
        if k not in recs and c not in used_codes:
            recs[k] = c
            used_codes.add(c)
    key = rng.choice(list(recs))
    table = "\n".join(f"{k}: {v}" for k, v in recs.items())
    prompt = (f"Here is a table of records.\n{table}\n\nWhat is the code for record {key}? "
              "Finish with a final line of the form ANSWER: <code>.")
    return {"kind": "kv", "prompt": prompt, "expect": recs[key], "own": set(recs.values())}


def make_order(rng, names):
    order = names[:]
    rng.shuffle(order)
    clues = [f"{order[i]} is immediately before {order[i+1]}." for i in range(len(order) - 1)]
    rng.shuffle(clues)
    prompt = ("Five people stand in a single line. " + " ".join(clues)
              + " Give the full order from first to last. Finish with a final line of the form "
              "ANSWER: name, name, name, name, name.")
    return {"kind": "order", "prompt": prompt, "expect": order, "own": set(names)}


def make_arith(rng):
    nums = [rng.randrange(100, 9999) for _ in range(12)]
    prompt = ("Add these integers exactly: " + ", ".join(map(str, nums))
              + ". Finish with a final line of the form ANSWER: <integer>.")
    return {"kind": "arith", "prompt": prompt, "expect": sum(nums), "own": set()}


def make_seq(rng):
    start, step = rng.randrange(1000, 9000), rng.randrange(3, 97)
    expect = [start + step * i for i in range(300)]
    prompt = (f"Write the arithmetic sequence that starts at {start} and increases by {step}, "
              "300 terms in total, as one comma-separated line with no other text. "
              "Put it on a final line of the form ANSWER: n, n, n, ...")
    return {"kind": "seq", "prompt": prompt, "expect": expect, "own": set()}


def answer_line(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip() if m else None


def grade(task, r, foreign_codes, foreign_names):
    if r.get("error"):
        return "error"
    if r["finish"] == "length":
        return "truncated"
    ans = answer_line(r["content"]) or answer_line(r["reasoning"])
    if ans is None:
        return "no_answer"
    if task["kind"] == "seq":
        got = [int(x) for x in re.findall(r"-?\d+", ans)]
        return "ok" if got == task["expect"] else "wrong"
    if task["kind"] in ("kv", "arith"):
        nums = re.findall(r"-?\d[\d,]*", ans)
        got = int(nums[-1].replace(",", "")) if nums else None
        if got == task["expect"]:
            return "ok"
        if task["kind"] == "kv" and got in foreign_codes:
            return "LEAK"
        return "wrong"
    got = [x.strip(" .*`") for x in ans.strip("*` ").split(",")]
    if got == task["expect"]:
        return "ok"
    if any(g in foreign_names for g in got):
        return "LEAK"
    return "wrong"


# ---------------------------------------------------------------- correctness
def correctness(host, port, rounds, max_tokens):
    w = Worker(host, port, consumer="spec-eval-correct")
    rng = random.Random(4242)
    used_codes = set()
    out = {"rounds": [], "summary": {}}
    for mode_name, mode in (("greedy", GREEDY), ("prod", PROD)):
        tally = {}
        for rd in range(rounds):
            pool = NAMES[:]
            rng.shuffle(pool)
            tasks = [make_kv(rng, used_codes), make_order(rng, pool[0:5]),
                     make_arith(rng), make_seq(rng)]
            with ThreadPoolExecutor(4) as ex:
                futs = [ex.submit(w.chat, [{"role": "user", "content": t["prompt"]}],
                                  max_tokens=max_tokens, seed=1000 + rd, **mode) for t in tasks]
                results = [f.result() for f in futs]
            for i, (t, r) in enumerate(zip(tasks, results)):
                foreign_codes = set().union(*(o["own"] for j, o in enumerate(tasks) if j != i and o["kind"] == "kv"))
                foreign_names = set().union(*(o["own"] for j, o in enumerate(tasks) if j != i and o["kind"] == "order"))
                g = grade(t, r, foreign_codes, foreign_names)
                tally.setdefault(t["kind"], {}).setdefault(g, 0)
                tally[t["kind"]][g] += 1
                out["rounds"].append({"mode": mode_name, "round": rd, "kind": t["kind"], "grade": g,
                                      "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                                      "answer": (answer_line(r.get("content")) or "")[:200],
                                      "expect": str(t["expect"])[:200]})
            log(f"correctness {mode_name} round {rd}: " + ", ".join(
                f"{t['kind']}={out['rounds'][-4 + i]['grade']}" for i, t in enumerate(tasks)))
        out["summary"][mode_name] = tally

    # determinism: identical greedy request twice, serially, each from a FLUSHED
    # cache -- a warm prefix changes prefill shapes and flips near-ties on its own.
    det = []
    for p in ("Explain in about 150 words why a write-ahead log makes crash recovery possible.",
              "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."):
        w.flush()
        a = w.chat([{"role": "user", "content": p}], max_tokens=1200, seed=7, **GREEDY)
        w.flush()
        b = w.chat([{"role": "user", "content": p}], max_tokens=1200, seed=7, **GREEDY)
        det.append((a["reasoning"] + "\0" + a["content"]) == (b["reasoning"] + "\0" + b["content"]))
    out["determinism_serial_greedy"] = det
    return out


# ---------------------------------------------------------------- speed
CODE_TASKS = [
    "Write a Python module implementing an LRU cache with TTL expiry, type hints and pytest tests.",
    "Write a Go HTTP middleware that enforces a token-bucket rate limit per API key, with tests.",
    "Write a TypeScript function that parses and validates an ISO-8601 duration string, with tests.",
    "Write a Rust function that merges k sorted iterators using a binary heap, with unit tests.",
]


def speed_phase(host, port, label, prefix, concurrency, reps):
    w = Worker(host, port, consumer=f"spec-eval-{label}")
    m0 = w.metrics()
    rows = []
    t_start = time.time()
    for rep in range(reps):
        with ThreadPoolExecutor(concurrency) as ex:
            futs = [ex.submit(w.chat, [{"role": "user", "content": prefix + CODE_TASKS[i % 4]}],
                              max_tokens=1024, ignore_eos=True, seed=rep * 10 + i, **PROD)
                    for i in range(concurrency)]
            for f in futs:
                r = f.result()
                rows.append({"ttft_s": r.get("ttft_s"), "decode_tok_s": r.get("decode_tok_s"),
                             "e2e_s": r.get("e2e_s"), "tokens": (r.get("usage") or {}).get("completion_tokens"),
                             "error": r.get("error")})
    wall = time.time() - t_start
    m1 = w.metrics()
    c = f"spec-eval-{label}"
    gen = metric_sum(m1, "sglang:generation_tokens_total", c) - metric_sum(m0, "sglang:generation_tokens_total", c)
    ver = metric_sum(m1, "sglang:spec_verify_calls_total", c) - metric_sum(m0, "sglang:spec_verify_calls_total", c)
    dec = sorted(x["decode_tok_s"] for x in rows if x["decode_tok_s"])
    res = {
        "phase": label, "concurrency": concurrency, "requests": len(rows),
        "decode_tok_s_median": dec[len(dec) // 2] if dec else None,
        "decode_tok_s_all": dec,
        "ttft_s_median": sorted(x["ttft_s"] for x in rows)[len(rows) // 2],
        "aggregate_tok_s": round(sum(x["tokens"] or 0 for x in rows) / wall, 1),
        "accept_length": round(gen / ver, 3) if ver else None,
        "errors": [x["error"] for x in rows if x["error"]],
    }
    log(f"speed {label}: {json.dumps({k: v for k, v in res.items() if k != 'decode_tok_s_all'})}")
    return res


def speed(host, port, reps):
    w = Worker(host, port, consumer="spec-eval-warm")
    rng = random.Random(99)
    # warmup: graphs, autotune
    w.chat([{"role": "user", "content": "Say hello."}], max_tokens=64, **PROD)
    out = []
    out.append(speed_phase(host, port, "short-c1", "", 1, reps * 2))
    out.append(speed_phase(host, port, "short-c4", "", 4, reps))
    long_prefix = ("Project notes for context (not needed for the task below):\n"
                   + filler(rng, 42000) + "\n\nTask: ")
    w.chat([{"role": "user", "content": long_prefix + "Say OK."}], max_tokens=1, **GREEDY)
    out.append(speed_phase(host, port, "long-c1", long_prefix, 1, reps * 2))
    out.append(speed_phase(host, port, "long-c4", long_prefix, 4, reps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--what", choices=["all", "correctness", "speed"], default="all")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    a = ap.parse_args()
    res = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    w = Worker(a.host, a.port, consumer="spec-eval")
    _, info = w.get("/get_server_info")
    info = json.loads(info)
    res["server"] = {k: info.get(k) for k in ("speculative_algorithm", "speculative_num_draft_tokens",
                                              "speculative_draft_model_path", "enable_hierarchical_cache",
                                              "hicache_ratio", "max_running_requests")}
    if a.what in ("all", "correctness"):
        res["correctness"] = correctness(a.host, a.port, a.rounds, a.max_tokens)
    if a.what in ("all", "speed"):
        res["speed"] = speed(a.host, a.port, a.reps)
    print(json.dumps(res, indent=1, default=list))


if __name__ == "__main__":
    main()
