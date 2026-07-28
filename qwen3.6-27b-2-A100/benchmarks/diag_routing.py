#!/usr/bin/env python3
"""
Settles whether r0 can receive traffic at all.

Sends SEQUENTIAL requests (one at a time, so the token-bucket rate limiter is
never a factor and every request is admitted) using prompts with NO shared
prefix whatsoever - different first tokens, different domains.

If cache-affinity is the only reason r0 stays idle, these should spread.
If they still all land on r1, routing is not balancing at all.
"""
import http.client, json, os, time

API_KEY = os.environ["SGLANG_API_KEY"]

PROMPTS = [
    "def fibonacci(n):  # complete this function",
    "SELECT name, id FROM users WHERE created_at >",
    "Translate to French: good morning, how are you?",
    "What is 17 multiplied by 234?",
    "Describe the fall of the Roman Empire in one sentence.",
    "kubectl get pods --all-namespaces | grep",
    "Photosynthesis converts light energy into",
    "The capital city of Australia is",
]


def verify_calls(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode("utf-8", "ignore")
    except Exception:
        return None
    finally:
        conn.close()
    for line in text.splitlines():
        if line.startswith("sglang:spec_verify_calls_total"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return None


def send(prompt):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }
    body = json.dumps(payload)
    conn = http.client.HTTPConnection("qwen36-27b-router", 8000, timeout=120)
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Length": str(len(body)),
        })
        r = conn.getresponse()
        r.read()
        return r.status, round(time.monotonic() - t0, 3)
    finally:
        conn.close()


def main():
    rows = []
    for p in PROMPTS:
        b0 = verify_calls("qwen36-27b-r0", 8001)
        b1 = verify_calls("qwen36-27b-r1", 8002)
        status, elapsed = send(p)
        time.sleep(0.4)  # let metrics settle
        a0 = verify_calls("qwen36-27b-r0", 8001)
        a1 = verify_calls("qwen36-27b-r1", 8002)
        d0 = None if None in (a0, b0) else a0 - b0
        d1 = None if None in (a1, b1) else a1 - b1
        served_by = "r0" if (d0 or 0) > (d1 or 0) else ("r1" if (d1 or 0) > 0 else "?")
        rows.append({
            "prompt": p[:44],
            "status": status,
            "elapsed_s": elapsed,
            "r0_delta": d0,
            "r1_delta": d1,
            "served_by": served_by,
        })
        print(json.dumps(rows[-1]))

    counts = {}
    for r in rows:
        counts[r["served_by"]] = counts.get(r["served_by"], 0) + 1
    print("\nSUMMARY served_by counts:", json.dumps(counts))


if __name__ == "__main__":
    main()
