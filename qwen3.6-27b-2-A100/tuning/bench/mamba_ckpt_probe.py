"""Does a prefix-cache hit on DECODED tokens restore the right GDN state?

#37818 (fixed in v0.5.20): with DFLASH + --mamba-radix-cache-strategy
extra_buffer, v0.5.19 never recorded the linear-attention checkpoint at a
tracking boundary crossed during decode, while the scheduler booked it as
recorded. A later request matching into those decoded tokens got full-attention
KV at one position and GDN state from an earlier one. Silent, no log line.

Per trial (unique nonce, so nothing else is cached):
  T1    P  -> O                  greedy decode; inserts P+O with DECODE checkpoints
  A     P+O+Q, hit               restores the state recorded during T1's decode
  flush
  C1    P+O+Q, cold              the reference: everything recomputed
  B     P+O+Q, hit               restores a state recorded during C1's PREFILL
  flush
  C2    P+O+Q, cold              cold-vs-cold floor
All legs ask for input logprobs from len(P+O) on. That also caps the prefix
match at len(P+O) (schedule_batch.py), so A and B reuse the same region.

A hit is never bit-identical to a cold run (different prefill shapes flip
near-ties), so the test is relative: A (decode checkpoint) must be as close
to C1 as B (prefill checkpoint, a path #37818 never touched). A stale state
would put A far outside B.

Run against a DRAINED worker only (it flushes the cache):
  ./tuning/bench/run_eval.sh r1 mamba_ckpt_probe <label>
"""
import argparse
import json
import sys
import time

from evalkit import Worker

TOPICS = [
    ("the history of suspension bridges", 520),
    ("how a compiler turns source code into machine code", 720),
    ("the water cycle and the ways cities manage stormwater", 900),
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def post(w, path, body):
    st, txt = w.post_raw(path, body)
    if st != 200:
        raise RuntimeError(f"{path} -> {st}: {txt[:300]}")
    return json.loads(txt)


def tokenize(w, **kw):
    return post(w, "/tokenize", {"add_special_tokens": False, **kw})["tokens"]


def generate(w, ids, max_new, logprob_start=None):
    body = {"input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    if logprob_start is not None:
        body.update(return_logprob=True, logprob_start_len=logprob_start)
    t = time.time()
    r = post(w, "/generate", body)
    mi = r["meta_info"]
    return {"output_ids": r["output_ids"], "cached": mi.get("cached_tokens"),
            "in_lp": [x[0] for x in mi.get("input_token_logprobs") or []],
            "secs": round(time.time() - t, 2)}


def diff(ref, x):
    lp = [abs(a - b) for a, b in zip(ref["in_lp"], x["in_lp"])
          if a is not None and b is not None]
    first = next((i for i, (a, b) in enumerate(zip(ref["output_ids"], x["output_ids"]))
                  if a != b), None)
    return {"n_lp": len(lp),
            "mean_abs_dlogprob": round(sum(lp) / len(lp), 5) if lp else None,
            "max_abs_dlogprob": round(max(lp), 5) if lp else None,
            "gen_first_divergence": first,
            "cached": x["cached"]}


def trial(w, i, topic, o_len, q_new):
    nonce = f"[probe {int(time.time())}-{i}]"
    p = tokenize(w, messages=[{"role": "user", "content":
                 f"{nonce} Write a numbered list of 30 specific facts about {topic}."}],
                 chat_template_kwargs={"reasoning_effort": "low"})
    im_end = tokenize(w, prompt="<|im_end|>")
    if not w.flush():
        raise RuntimeError("flush_cache never answered 'Cache flushed.'")
    t1 = generate(w, p, o_len)
    o = t1["output_ids"]
    q_text = ("\n" if o[-len(im_end):] == im_end else "<|im_end|>\n") + (
        "<|im_start|>user\nQuote fact number 7 from your list exactly, "
        "then fact number 3.<|im_end|>\n<|im_start|>assistant\n")
    q = tokenize(w, prompt=q_text)
    ids, start = p + o + q, len(p) + len(o)

    a = generate(w, ids, q_new, start)
    if not w.flush():
        raise RuntimeError("flush before C1 failed")
    c1 = generate(w, ids, q_new, start)
    b = generate(w, ids, q_new, start)
    if not w.flush():
        raise RuntimeError("flush before C2 failed")
    c2 = generate(w, ids, q_new, start)

    res = {"topic": topic, "len_p": len(p), "len_o": len(o), "len_q": len(q),
           "t1_secs": t1["secs"],
           "A_decode_ckpt_hit": diff(c1, a),
           "B_prefill_ckpt_hit": diff(c1, b),
           "C2_cold": diff(c1, c2),
           "C1_cached": c1["cached"]}
    ok_reach = (a["cached"] or 0) > len(p)
    a_m = res["A_decode_ckpt_hit"]["mean_abs_dlogprob"] or 0
    b_m = res["B_prefill_ckpt_hit"]["mean_abs_dlogprob"] or 0
    res["hit_reaches_decoded_tokens"] = ok_reach
    res["verdict"] = ("PASS" if ok_reach and a_m <= max(3 * b_m, 0.02)
                      else "NO_DECODED_HIT" if not ok_reach else "FAIL")
    log(f"trial {i}: P={len(p)} O={len(o)} Q={len(q)} cachedA={a['cached']} "
        f"A={res['A_decode_ckpt_hit']} B={res['B_prefill_ckpt_hit']} "
        f"C2={res['C2_cold']} -> {res['verdict']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen-after-q", type=int, default=48)
    a = ap.parse_args()
    w = Worker(a.host, a.port, consumer=f"probe-{a.label}")
    out = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "trials": []}
    for i, (topic, o_len) in enumerate(TOPICS):
        try:
            out["trials"].append(trial(w, i, topic, o_len, a.gen_after_q))
        except Exception as e:  # keep what we have
            out["trials"].append({"topic": topic, "error": str(e)})
            log(f"trial {i} error: {e}")
    out["verdict"] = ("PASS" if all(t.get("verdict") == "PASS" for t in out["trials"])
                      else "CHECK")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
