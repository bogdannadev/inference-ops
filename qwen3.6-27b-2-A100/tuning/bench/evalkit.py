"""Shared plumbing for hicache_probe.py and spec_eval.py.

Talks to ONE worker directly (no router), streams chat completions, and reads
the worker's own /metrics so a run can be attributed to its consumer label.
Stdlib only: runs in python:3.12-slim on the qwen36-27b-backend network.
"""
import http.client
import json
import os
import random
import re
import time

API_KEY = os.environ["SGLANG_API_KEY"]
MODEL = "qwen36-27b"


class Worker:
    def __init__(self, host, port, consumer):
        self.host, self.port, self.consumer = host, port, consumer

    def _conn(self, timeout=1800):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            # Engine-side custom label, so these requests are separable in
            # sglang:* metrics from anything else the worker saw.
            "x-request-id-labels": json.dumps({"consumer": self.consumer}),
        }

    def get(self, path):
        c = self._conn(30)
        c.request("GET", path, headers=self._headers())
        r = c.getresponse()
        return r.status, r.read().decode()

    def post_raw(self, path, body=None):
        c = self._conn(120)
        c.request("POST", path, body=json.dumps(body) if body else None, headers=self._headers())
        r = c.getresponse()
        return r.status, r.read().decode()

    def flush(self, wait_s=90):
        """POST /flush_cache refuses while anything is in flight and still
        answers 200-ish text; only the literal 'Cache flushed.' counts."""
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                st, txt = self.post_raw("/flush_cache")
                if st == 200 and txt.startswith("Cache flushed."):
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def metrics(self):
        _, txt = self.get("/metrics")
        return txt

    def chat(self, messages, **params):
        """Streamed chat completion. Returns timing, text, reasoning, usage."""
        body = {"model": MODEL, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}}
        body.update(params)
        t0 = time.time()
        c = self._conn()
        c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=self._headers())
        r = c.getresponse()
        if r.status != 200:
            return {"error": f"HTTP {r.status}: {r.read().decode()[:300]}"}
        t_first = None
        content, reasoning, usage, finish, extra = [], [], None, None, {}
        buf = b""
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for k in ("sgl_ext",):
                if ev.get(k):
                    extra[k] = ev[k]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece_c, piece_r = d.get("content"), d.get("reasoning_content")
                if (piece_c or piece_r) and t_first is None:
                    t_first = time.time()
                if piece_c:
                    content.append(piece_c)
                if piece_r:
                    reasoning.append(piece_r)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        t_end = time.time()
        out = {
            "ttft_s": round((t_first or t_end) - t0, 3),
            "e2e_s": round(t_end - t0, 3),
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "finish": finish,
            "usage": usage,
            "extra": extra,
        }
        ct = (usage or {}).get("completion_tokens") or 0
        dt = t_end - (t_first or t_end)
        out["decode_tok_s"] = round((ct - 1) / dt, 2) if ct > 1 and dt > 0 else None
        return out


def metric_sum(text, name, consumer=None, extra_label=None):
    """Sum a counter/gauge over all series, optionally filtered by consumer."""
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(name + "{") and not line.startswith(name + " "):
            continue
        if consumer is not None and f'consumer="{consumer}"' not in line:
            continue
        if extra_label and extra_label not in line:
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return total


WORDS = ("amber basalt cobalt delta ember fjord garnet harbor iris juniper kestrel "
         "lagoon meadow nectar onyx prairie quartz raven sierra tundra umber valley "
         "willow xenon yarrow zephyr anchor bramble cinder dune falcon glacier heron "
         "indigo jasper lantern marble nimbus orchid pebble quill ridge saffron thistle "
         "ultra vessel wren yonder zenith copper birch canyon dahlia everest fern grove").split()


def filler(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def extract_last_int(text):
    m = re.findall(r"-?\d+", text or "")
    return int(m[-1]) if m else None
