#!/usr/bin/env python3
"""Functional smoke for ONE worker after a roll, direct (no gateway).

  1. plain chat answers
  2. reasoning lands in reasoning_content, not content
  3. qwen3_coder tool call -> structured tool_calls
  4. two calls to the same tool, non-streaming -> index 0 and 1 (#39574)
  5. one 12 MP image -> ~2,000-2,100 prompt tokens (the 2 MP cap)
  6. 17 images -> 400 "exceeds limit 16"
  7. a stream closed after 30 chunks -> the scheduler aborts it
     (num_aborted_requests_total +1); grep the log for "state was deleted"

  ./tuning/bench/run_eval.sh r1 smoke <label>
"""
import argparse
import base64
import json
import struct
import sys
import time
import zlib

from evalkit import MODEL, Worker, metric_sum


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def png(w, h):
    """A w x h RGB gradient PNG, stdlib only."""
    rows = b"".join(b"\x00" + bytes(v for x in range(w) for v in (x * 255 // w, y * 255 // h, 128))
                    if y % 64 == 0 else b"\x02" + b"\x00" * (3 * w) for y in range(h))
    # filter 2 (Up) repeats the last full row: tiny and still a real w x h image

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def img_part(data):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(w, body, timeout=600):
    st, txt = w.post_raw("/v1/chat/completions", dict(body, model=MODEL), timeout=timeout)
    return st, (json.loads(txt) if txt.startswith("{") else {"raw": txt[:300]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    w = Worker(a.host, a.port, "smoke")
    res = {}

    r = w.chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
               max_tokens=2000)
    res["chat"] = {"ok": "391" in r.get("content", ""), "content": r.get("content", "")[:80]}
    res["reasoning_split"] = {"ok": bool(r.get("reasoning")) and "<think>" not in r.get("content", ""),
                              "reasoning_chars": len(r.get("reasoning", ""))}

    st, d = post(w, {"messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                     "tools": TOOLS, "max_tokens": 2000})
    tc = (d.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    res["tool_call"] = {"ok": st == 200 and len(tc) >= 1 and tc[0]["function"]["name"] == "get_weather",
                        "calls": [c["function"] for c in tc]}

    st, d = post(w, {"messages": [{"role": "user", "content":
                     "Get the weather in Paris and in Tokyo. Call the tool once per city, both now."}],
                     "tools": TOOLS, "max_tokens": 3000})
    tc = (d.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    res["two_tool_calls_index"] = {"ok": len(tc) == 2 and [c.get("index") for c in tc] == [0, 1],
                                   "indexes": [c.get("index") for c in tc], "n": len(tc)}

    st, d = post(w, {"messages": [{"role": "user", "content": [
        img_part(png(4000, 3000)), {"type": "text", "text": "Describe this image in five words."}]}],
        "max_tokens": 2000})
    pt = (d.get("usage") or {}).get("prompt_tokens")
    res["image_12mp"] = {"ok": st == 200 and pt is not None and 1500 < pt < 2600, "status": st, "prompt_tokens": pt}

    small = png(64, 64)
    st, d = post(w, {"messages": [{"role": "user", "content": [img_part(small) for _ in range(17)]
                                   + [{"type": "text", "text": "Count the images."}]}], "max_tokens": 50})
    msg = json.dumps(d)[:300]
    res["images_17"] = {"ok": st == 400 and "16" in msg, "status": st, "body": msg}

    before = metric_sum(w.metrics(), "sglang:num_aborted_requests_total")
    c = w._conn(600)
    c.request("POST", "/v1/chat/completions", body=json.dumps({
        "model": MODEL, "stream": True, "max_tokens": 4000,
        "messages": [{"role": "user", "content": "Write a long story about a lighthouse."}]}),
        headers=w._headers())
    rr = c.getresponse()
    n = 0
    while n < 30 and rr.readline():
        n += 1
    c.sock.close()
    c.close()
    after = before
    for _ in range(20):
        time.sleep(1)
        after = metric_sum(w.metrics(), "sglang:num_aborted_requests_total")
        if after > before:
            break
    st, txt = w.get("/metrics")
    running = metric_sum(txt, "sglang:num_running_reqs")
    res["cancel_aborts"] = {"ok": after == before + 1 and running == 0,
                            "aborted_delta": after - before, "running_after": running}

    res["pass"] = sum(v["ok"] for v in res.values() if isinstance(v, dict))
    res["total"] = sum(1 for v in res.values() if isinstance(v, dict))
    log(json.dumps(res, indent=1))
    print(json.dumps({"label": a.label, **res}, indent=1))


if __name__ == "__main__":
    main()
