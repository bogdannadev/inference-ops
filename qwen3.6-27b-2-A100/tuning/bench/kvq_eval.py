"""KV-cache quantization accuracy gate: bf16 vs fp8 KV on the SAME drained worker.

fp8 KV on this checkpoint is an uncalibrated cast (k_scale/v_scale unset;
`memory_pool.py set_kv_buffer` does `cache_k.to(fp8)`): 3 mantissa bits, and
e4m3 clips at +-448. K is bounded by QK-norm, V is not. Two measurements, each
run once per config and compared offline with --compare:

1. teacher-forced logprobs. Real documents (prose and code, sglang v0.5.20
   docs + python/sglang/srt, mounted at /corpus) of 8K..128K tokens; input
   logprobs of the LAST 256 tokens, so every prediction attends over the whole
   fp8-stored context. A drained worker is bit-deterministic run to run
   (mamba_ckpt_probe: cold vs cold = 0.000), so any difference is the KV dtype.
2. long-context lookup. Tables of unique random records at 16K/64K/128K tokens,
   three keys asked per prompt at 10% / 50% / 90% depth, exact code answers,
   greedy with low reasoning effort.

Every prompt runs cold (flush first). Cold prefill seconds are recorded too:
on SM80 fp8 has no hardware convert, so attention pays a software dequant.

  EVAL_CORPUS=<sglang tree> ./tuning/bench/run_eval.sh r1 kvq_eval <label>
  python3 tuning/bench/kvq_eval.py --compare base.json trial.json
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

WINDOW = 256
LENGTHS = [8192, 32768, 65536, 131072]
LOOKUP_LENGTHS = [16384, 65536, 131072]
LOOKUP_SEEDS = [11, 12]
DEPTHS = [0.1, 0.5, 0.9]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_corpus(root):
    def cat(paths):
        out = []
        for p in sorted(paths):
            with open(p, errors="replace") as f:
                lines = [ln for ln in f.read().splitlines() if len(ln) < 2000]
            out.append(f"\n\n# file: {os.path.relpath(p, root)}\n" + "\n".join(lines))
        return "".join(out)
    prose = cat(glob.glob(f"{root}/docs/**/*.md", recursive=True)
                + glob.glob(f"{root}/docs/**/*.mdx", recursive=True))
    code = cat(glob.glob(f"{root}/python/sglang/srt/**/*.py", recursive=True))
    return {"prose": prose, "code": code}


def post(w, path, body):
    st, txt = w.post_raw(path, body, timeout=1800)
    if st != 200:
        raise RuntimeError(f"{path} -> {st}: {txt[:300]}")
    return json.loads(txt)


def doc_ids(w, text, start_char, n_tokens):
    chars = n_tokens * 5
    while True:
        ids = post(w, "/tokenize", {"prompt": text[start_char:start_char + chars],
                                    "add_special_tokens": False})["tokens"]
        if len(ids) >= n_tokens:
            return ids[:n_tokens]
        chars = int(chars * 1.5)


def ppl_part(w, corpus, lengths=LENGTHS, first_char=0):
    out = []
    for genre, text in corpus.items():
        start = first_char
        for n in lengths:
            ids = doc_ids(w, text, start, n)
            start += n * 6            # next document starts past this one
            if not w.flush():
                raise RuntimeError("flush_cache never answered 'Cache flushed.'")
            t = time.time()
            r = post(w, "/generate", {"input_ids": ids, "return_logprob": True,
                                      "logprob_start_len": n - WINDOW,
                                      "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
            secs = round(time.time() - t, 2)
            lps = [x[0] for x in r["meta_info"]["input_token_logprobs"]][-WINDOW + 1:]
            nll = -sum(lps) / len(lps)
            out.append({"genre": genre, "tokens": n, "cold_prefill_s": secs,
                        "ids_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                        "mean_nll": round(nll, 5), "logprobs": [round(x, 5) for x in lps]})
            log(f"ppl {genre:5s} {n:>6} tok: nll={nll:.4f} cold {secs}s")
    return out


def lookup_prompt(rng, n_tokens):
    # 19.0 tokens per record line with the Qwen3.8 tokenizer (measured via /tokenize)
    n_rec = n_tokens // 19
    recs, seen = [], set()
    while len(recs) < n_rec:
        k = f"R-{rng.getrandbits(28):07x}"
        if k in seen:
            continue
        seen.add(k)
        recs.append((k, rng.randrange(10_000_000, 99_999_999)))
    picks = [recs[min(int(d * n_rec), n_rec - 1)] for d in DEPTHS]
    table = "\n".join(f"{k}: {v}" for k, v in recs)
    q = ", ".join(k for k, _ in picks)
    prompt = (f"Here is a table of records.\n{table}\n\nGive the codes for records {q}, in that order. "
              "Finish with a final line of the form ANSWER: <code>, <code>, <code>.")
    return prompt, [v for _, v in picks]


def lookup_part(w, max_tokens):
    out = []
    for n in LOOKUP_LENGTHS:
        for seed in LOOKUP_SEEDS:
            prompt, expect = lookup_prompt(random.Random(seed * 1000 + n), n)
            if not w.flush():
                raise RuntimeError("flush before lookup failed")
            r = w.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                       temperature=0, top_p=1, top_k=1,
                       chat_template_kwargs={"reasoning_effort": "low"})
            if r.get("error"):
                out.append({"tokens": n, "seed": seed, "error": r["error"]})
                log(f"lookup {n} seed {seed}: {r['error']}")
                continue
            m = re.findall(r"ANSWER:\s*(.+)", r["content"] or "") or \
                re.findall(r"ANSWER:\s*(.+)", r["reasoning"] or "")
            got = [int(x) for x in re.findall(r"\d{8}", m[-1])] if m else []
            hits = [i < len(got) and got[i] == e for i, e in enumerate(expect)]
            out.append({"tokens": n, "seed": seed, "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                        "hits": hits, "finish": r["finish"], "ttft_s": r["ttft_s"],
                        "completion_tokens": (r["usage"] or {}).get("completion_tokens")})
            log(f"lookup {n:>6} seed {seed}: hits={hits} finish={r['finish']} ttft={r['ttft_s']}s")
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A={a['label']}  B={b['label']}")
    print(f"{'genre':5s} {'tokens':>7} {'nll A':>8} {'nll B':>8} {'dNLL':>8} {'mean|d|':>8} "
          f"{'p99|d|':>7} {'max|d|':>7} {'>0.5':>5} {'cold A':>7} {'cold B':>7}")
    all_d = []
    for x, y in zip(a["ppl"], b["ppl"]):
        assert (x["genre"], x["tokens"], x["ids_sha"]) == (y["genre"], y["tokens"], y["ids_sha"]), "misaligned"
        d = sorted(abs(p - q) for p, q in zip(x["logprobs"], y["logprobs"]))
        all_d += d
        print(f"{x['genre']:5s} {x['tokens']:>7} {x['mean_nll']:>8.4f} {y['mean_nll']:>8.4f} "
              f"{y['mean_nll'] - x['mean_nll']:>+8.4f} {sum(d) / len(d):>8.4f} "
              f"{d[int(0.99 * (len(d) - 1))]:>7.3f} {d[-1]:>7.3f} {sum(v > 0.5 for v in d):>5} "
              f"{x['cold_prefill_s']:>7} {y['cold_prefill_s']:>7}")
    all_d.sort()
    print(f"all   mean|d|={sum(all_d) / len(all_d):.4f} p99={all_d[int(0.99 * (len(all_d) - 1))]:.3f} "
          f"max={all_d[-1]:.3f} n={len(all_d)}")
    sa = [sum(p - q for p, q in zip(y["logprobs"], x["logprobs"])) for x, y in zip(a["ppl"], b["ppl"])]
    n = sum(len(x["logprobs"]) for x in a["ppl"])
    flat = [q - p for x, y in zip(a["ppl"], b["ppl"]) for p, q in zip(x["logprobs"], y["logprobs"])]
    mu = sum(flat) / n
    sd = (sum((v - mu) ** 2 for v in flat) / (n - 1)) ** 0.5
    print(f"pooled dNLL (B-A) = {-mu:+.4f} nats/token, SE {sd / n ** 0.5:.4f}, n={n}, "
          f"docs worse/better: {sum(v < 0 for v in sa)}/{sum(v > 0 for v in sa)}")
    for side in (a, b):
        if "lookup" not in side:
            continue
        hits = [h for r in side["lookup"] for h in r.get("hits", [])]
        print(f"lookup {side['label']}: {sum(hits)}/{len(hits)}  " +
              " ".join(f"{r['tokens'] // 1024}K:{''.join('x' if h else '.' for h in r.get('hits', []))}"
                       for r in side["lookup"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--label")
    ap.add_argument("--corpus", default="/corpus")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--skip", choices=["ppl", "lookup"])
    # Many short high-entropy documents instead of the long set: the long
    # documents are mostly low-entropy (repeated code/docs, NLL < 0.7) and so
    # barely sensitive; 8K windows of fresh text are where a perturbation shows.
    ap.add_argument("--short-docs", type=int, default=0, metavar="N")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "TRIAL"))
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evalkit import Worker
    w = Worker(a.host, a.port, consumer=f"kvq-{a.label}")
    res = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.short_docs:
        res["ppl"] = ppl_part(w, load_corpus(a.corpus), [8192] * a.short_docs, 1_500_000)
    elif a.skip != "ppl":
        res["ppl"] = ppl_part(w, load_corpus(a.corpus))
    if a.skip != "lookup" and not a.short_docs:
        res["lookup"] = lookup_part(w, a.max_tokens)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
