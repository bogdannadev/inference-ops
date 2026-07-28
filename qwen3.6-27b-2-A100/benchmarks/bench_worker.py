#!/usr/bin/env python3
"""
Benchmarks a worker DIRECTLY (bypassing the router) so routing cannot
contaminate an A/B comparison between the two replicas.

Designed for judging changes to the speculative-decode / SSM kernels:
  - decode throughput on long generations   (the thing a verify kernel moves)
  - spec_accept_length / spec_accept_rate   (must not regress)
  - TTFT on a realistic long prompt         (prefill path, should be unchanged)
  - greedy output TEXT                      (temperature 0 -> a numerically
                                             broken kernel shows up as divergence
                                             from the control replica)

Usage:
  python3 bench_worker.py --host qwen36-27b-r0 --port 8001 --label r0_before
"""
import argparse, http.client, json, os, time

API_KEY = os.environ["SGLANG_API_KEY"]

# Deterministic, self-contained prompts. No shared prefix between them, so a
# warm prefix cache cannot silently inflate one run relative to another.
DECODE_PROMPTS = [
    "Write a Python function that merges two sorted lists. Explain each step.",
    "Explain how a B-tree index speeds up a range query in a relational database.",
    "Describe the difference between optimistic and pessimistic concurrency control.",
]

# ~16K-token prompt to exercise the realistic prefill path (coding-agent shape).
LONG_BLOCK = (
    "def process(record):\n"
    "    validated = validate(record)\n"
    "    enriched = enrich(validated)\n"
    "    return persist(enriched)\n\n"
)


def _post(host, port, payload, timeout, stream):
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        if resp.status != 200:
            return {"status": resp.status, "error": resp.read()[:300].decode("utf-8", "ignore")}

        if not stream:
            data = json.loads(resp.read())
            dt = time.monotonic() - t0
            usage = data.get("usage") or {}
            ct = usage.get("completion_tokens")
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            # qwen3 reasoning-parser puts the chain-of-thought in reasoning_content
            # and leaves content empty until thinking finishes -- capture both, or
            # the determinism probe compares two empty strings and always "passes".
            return {
                "status": 200,
                "latency_s": round(dt, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": ct,
                "decode_tok_s": round(ct / dt, 2) if ct else None,
                "finish_reason": choice.get("finish_reason"),
                "text": (msg.get("content") or ""),
                "reasoning": (msg.get("reasoning_content") or ""),
            }

        ttft = None
        ntok = 0
        while True:
            line = resp.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            chunk = line[6:]
            if chunk == b"[DONE]":
                break
            try:
                d = json.loads(chunk)
            except ValueError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.monotonic() - t0
                ntok += 1
        dt = time.monotonic() - t0
        # decode-phase rate excludes prefill, which is what a verify kernel affects
        decode_s = dt - (ttft or 0)
        return {
            "status": 200,
            "latency_s": round(dt, 3),
            "ttft_s": round(ttft, 3) if ttft else None,
            "stream_chunks": ntok,
            "decode_tok_s": round(ntok / decode_s, 2) if decode_s > 0 else None,
        }
    finally:
        conn.close()


def spec_metrics(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode("utf-8", "ignore")
    except Exception:
        return {}
    finally:
        conn.close()
    out = {}
    for line in text.splitlines():
        for k in ("spec_accept_length", "spec_accept_rate", "spec_verify_calls_total",
                  "cache_hit_rate", "mamba_used_tokens", "gen_throughput"):
            if line.startswith(f"sglang:{k}{{"):
                try:
                    out[k] = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--long-repeat", type=int, default=1400, help="~16K prompt tokens")
    args = ap.parse_args()

    res = {"label": args.label, "host": args.host, "port": args.port,
           "metrics_before": spec_metrics(args.host, args.port)}

    # --- 1. decode throughput, greedy, long generations -------------------
    decode_runs = []
    for p in DECODE_PROMPTS:
        r = _post(args.host, args.port, {
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": p}],
            "temperature": 0, "top_p": 1, "seed": 42,
            "max_tokens": args.max_tokens,
        }, timeout=300, stream=False)
        decode_runs.append(r)
    res["decode_runs"] = decode_runs
    ok = [r["decode_tok_s"] for r in decode_runs if r.get("decode_tok_s")]
    res["decode_tok_s_mean"] = round(sum(ok) / len(ok), 2) if ok else None

    # --- 2. realistic long-prompt TTFT ------------------------------------
    long_prompt = LONG_BLOCK * args.long_repeat + "\nSummarise what this code does in one sentence."
    res["long_prompt_stream"] = _post(args.host, args.port, {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": long_prompt}],
        "temperature": 0, "top_p": 1, "seed": 42,
        "max_tokens": 256, "stream": True,
    }, timeout=600, stream=True)

    # --- 3. greedy determinism probe (correctness vs the control replica) --
    # 1024 tokens so the model can finish thinking AND emit content; a short
    # budget yields content:"" on both replicas, which would compare equal
    # regardless of whether the kernel is correct.
    res["greedy_probe"] = _post(args.host, args.port, {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content":
                      "List the first 12 prime numbers, comma separated, nothing else."}],
        "temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 1024,
    }, timeout=300, stream=False)

    res["metrics_after"] = spec_metrics(args.host, args.port)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
