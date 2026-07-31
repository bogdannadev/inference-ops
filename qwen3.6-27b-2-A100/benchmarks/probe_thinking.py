#!/usr/bin/env python3
"""
Establishes what a CLIENT can do about reasoning tokens eating the max_tokens
budget on this deployment.

With --reasoning-parser qwen3 the model emits chain-of-thought into
`reasoning_content` and leaves `content` empty until thinking finishes. Those
reasoning tokens are billed against the SAME max_tokens budget, so a client
that sets a budget sized for the visible answer gets finish_reason:"length"
and content:"" -- a response that looks like the server returned nothing.

Tests, in order:
  1. what a typical coding-agent budget actually produces
  2. whether chat_template_kwargs {"enable_thinking": false} disables thinking
  3. whether the /no_think prompt suffix disables thinking
  4. how large a budget the SAME question needs to reach content

Usage: python3 probe_thinking.py --host qwen36-27b-r0 --port 8001
"""
import argparse, http.client, json, os

API_KEY = os.environ["SGLANG_API_KEY"]

# Representative of what a coding agent actually sends -- not a toy question.
QUESTION = (
    "Refactor this function to avoid the nested loop, and briefly say why:\n\n"
    "def dupes(a, b):\n"
    "    out = []\n"
    "    for x in a:\n"
    "        for y in b:\n"
    "            if x == y:\n"
    "                out.append(x)\n"
    "    return out\n"
)


def ask(host, port, max_tokens, extra=None, question=QUESTION):
    payload = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0, "top_p": 1, "seed": 42,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(host, port, timeout=300)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Content-Length": str(len(body)),
        })
        r = conn.getresponse()
        raw = r.read()
        if r.status != 200:
            return {"status": r.status, "error": raw[:300].decode("utf-8", "ignore")}
        d = json.loads(raw)
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        usage = d.get("usage") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        return {
            "status": 200,
            "max_tokens": max_tokens,
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": ch.get("finish_reason"),
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "content_empty": content.strip() == "",
            "content_head": content[:160],
        }
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    H, P = args.host, args.port

    out = {}

    # 1. Typical coding-agent budgets, unmodified.
    out["budget_sweep"] = [ask(H, P, n) for n in (256, 512, 1024, 2048, 4096)]

    # 2. Server-side switch: does this build honour enable_thinking=false?
    out["enable_thinking_false"] = ask(
        H, P, 512, {"chat_template_kwargs": {"enable_thinking": False}})

    # 3. Prompt-level switch used by the Qwen3 family.
    out["no_think_suffix"] = ask(H, P, 512, None, QUESTION + "\n/no_think")

    # 4. Does separate_reasoning=False change the shape of the response?
    out["separate_reasoning_false"] = ask(H, P, 1024, {"separate_reasoning": False})

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
