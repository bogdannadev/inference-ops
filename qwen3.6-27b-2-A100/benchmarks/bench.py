#!/usr/bin/env python3
"""
Reusable benchmark client for the qwen36-27b router.
Runs INSIDE a throwaway container attached to the qwen36-27b-backend network
(so it can resolve qwen36-27b-router / qwen36-27b-r0 / qwen36-27b-r1 by name).

Stdlib-only (no pip installs, no external deps) so it works in a bare
python:3.12-slim container with zero setup.

Prints a single JSON blob to stdout. The wrapper shell script captures it.
"""
import argparse
import http.client
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

FILLER = (
    "The quick brown fox jumps over the lazy dog. " * 40
)  # ~ a few hundred tokens of harmless filler for prefill-heavy tests


def chat_request(host, port, api_key, prompt, max_tokens, stream, timeout=120):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    body = json.dumps(payload)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Length": str(len(body)),
    }

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t_send = time.monotonic()
    ttft = None
    completion_tokens = None
    prompt_tokens = None
    status = None
    error = None
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        if stream:
            full = b""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                if ttft is None:
                    ttft = time.monotonic() - t_send
                full += chunk
            t_end = time.monotonic()
            # best-effort: count "usage" completion_tokens from the final SSE chunk if present
            text = full.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:") and '"usage"' in line:
                    try:
                        obj = json.loads(line[len("data:"):].strip())
                        usage = obj.get("usage") or {}
                        completion_tokens = usage.get("completion_tokens")
                        prompt_tokens = usage.get("prompt_tokens")
                    except Exception:
                        pass
        else:
            raw = resp.read()
            t_end = time.monotonic()
            ttft = t_end - t_send  # non-stream: no earlier signal available
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
                usage = obj.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                if status != 200:
                    error = obj.get("error") or raw[:300].decode("utf-8", errors="ignore")
            except Exception as e:
                error = f"json parse failed: {e}; raw[:200]={raw[:200]!r}"
    except Exception as e:
        t_end = time.monotonic()
        error = f"{type(e).__name__}: {e}"
    finally:
        conn.close()

    latency = t_end - t_send
    tokens_per_sec = (
        completion_tokens / latency if completion_tokens and latency > 0 else None
    )
    return {
        "status": status,
        "error": error,
        "latency_s": round(latency, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec else None,
    }


def fetch_metrics(host, port, keys, timeout=15):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        conn.close()

    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for k in keys:
            if line.startswith(f"sglang:{k}{{") or line.startswith(f"sglang:{k} "):
                try:
                    val = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                out.setdefault(k, []).append(val)
    return out


def summarize(latencies):
    if not latencies:
        return {}
    return {
        "n": len(latencies),
        "mean": round(statistics.mean(latencies), 3),
        "median": round(statistics.median(latencies), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-host", default="qwen36-27b-router")
    ap.add_argument("--router-port", type=int, default=8000)
    ap.add_argument("--r0-host", default="qwen36-27b-r0")
    ap.add_argument("--r0-port", type=int, default=8001)
    ap.add_argument("--r1-host", default="qwen36-27b-r1")
    ap.add_argument("--r1-port", type=int, default=8002)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    api_key = os.environ["SGLANG_API_KEY"]
    metric_keys = [
        "spec_accept_rate",
        "spec_accept_length",
        "spec_verify_calls_total",
        "cache_hit_rate",
        "mamba_usage",
    ]

    result = {"scenarios": {}}

    metrics_before = {
        "r0": fetch_metrics(args.r0_host, args.r0_port, metric_keys),
        "r1": fetch_metrics(args.r1_host, args.r1_port, metric_keys),
    }

    # Scenario 1: sequential short-prompt, non-streaming (baseline latency + tok/s)
    short_prompt = "Write a three-sentence summary of why distributed systems are hard."
    seq_nonstream = [
        chat_request(args.router_host, args.router_port, api_key, short_prompt, 128, False)
        for _ in range(3)
    ]
    result["scenarios"]["sequential_short_nonstream"] = {
        "requests": seq_nonstream,
        "latency_summary": summarize([r["latency_s"] for r in seq_nonstream if r["status"] == 200]),
        "tokens_per_sec_summary": summarize(
            [r["tokens_per_sec"] for r in seq_nonstream if r["tokens_per_sec"]]
        ),
    }

    # Scenario 2: sequential short-prompt, streaming (TTFT)
    seq_stream = [
        chat_request(args.router_host, args.router_port, api_key, short_prompt, 128, True)
        for _ in range(3)
    ]
    result["scenarios"]["sequential_short_stream"] = {
        "requests": seq_stream,
        "ttft_summary": summarize([r["ttft_s"] for r in seq_stream if r["status"] == 200 and r["ttft_s"]]),
        "latency_summary": summarize([r["latency_s"] for r in seq_stream if r["status"] == 200]),
    }

    # Scenario 3: long-context prefill-heavy, non-streaming, single shot
    long_prompt = FILLER + "\n\nSummarize the above in one sentence."
    long_result = chat_request(args.router_host, args.router_port, api_key, long_prompt, 128, False)
    result["scenarios"]["long_context_prefill"] = long_result

    # Scenario 4: concurrency burst
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                chat_request, args.router_host, args.router_port, api_key, short_prompt, 128, False, 300
            )
            for _ in range(args.concurrency)
        ]
        concurrent_results = [f.result() for f in as_completed(futures)]
    wall_time = time.monotonic() - t0

    ok = [r for r in concurrent_results if r["status"] == 200]
    failed = [r for r in concurrent_results if r["status"] != 200]
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    result["scenarios"]["concurrency_burst"] = {
        "concurrency": args.concurrency,
        "wall_time_s": round(wall_time, 3),
        "success_count": len(ok),
        "failed_count": len(failed),
        "failed_details": failed,
        "per_request_latency_summary": summarize([r["latency_s"] for r in ok]),
        "aggregate_tokens_per_sec": round(total_completion_tokens / wall_time, 2) if wall_time > 0 else None,
        "requests": concurrent_results,
    }

    metrics_after = {
        "r0": fetch_metrics(args.r0_host, args.r0_port, metric_keys),
        "r1": fetch_metrics(args.r1_host, args.r1_port, metric_keys),
    }
    result["metrics_before"] = metrics_before
    result["metrics_after"] = metrics_after

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
