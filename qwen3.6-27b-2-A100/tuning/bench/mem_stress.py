#!/usr/bin/env python3
"""Worst-case GPU memory probe for a drained worker.

Production requests reach 168,998 tokens (gateway.requests, 2026-09). A config
that boots with little activation headroom can still OOM on the first long
prefill, which is exactly how the first DFlash2 boot died (2026-09-13). So this
sends a near-context-length cold prompt TOGETHER with three mid-size requests
that keep decoding, twice, and reports errors. Pair it with a log check for
OutOfMemoryError on the worker.
"""
import argparse
import base64
import json
import random
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

from evalkit import Worker, filler

WORDS_PER_TOKEN = 26000 / 44993   # measured on this filler with the Qwen3.8 tokenizer


def png_data_url(seed, side=1448):
    """A side x side RGB PNG (1448^2 ~ 2.1 MP, just under the 2 MP cap -> ~2K
    image tokens). Four repeating rows (inside the 32 KB deflate window) keep
    it ~80 KB on the wire; the seed makes each image distinct so the mm
    preprocess cache cannot serve it."""
    rng = random.Random(seed)
    rows = [b"\0" + bytes(rng.randrange(256) for _ in range(side * 3)) for _ in range(4)]
    raw = b"".join(rows[y % 4] for y in range(side))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--big-tokens", type=int, default=160_000)
    ap.add_argument("--side-tokens", type=int, default=30_000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--images", type=int, default=0,
                    help="replace side2 with a request carrying N ~2 MP images (vision encoder peak)")
    a = ap.parse_args()
    w = Worker(a.host, a.port, consumer="mem-stress")
    rng = random.Random(777)
    rows = []
    for rd in range(a.rounds):
        big = f"Round {rd} big {rng.getrandbits(64):x}\n" + filler(rng, int(a.big_tokens * WORDS_PER_TOKEN))
        sides = [f"Round {rd} side {i} {rng.getrandbits(64):x}\n" + filler(rng, int(a.side_tokens * WORDS_PER_TOKEN))
                 for i in range(3)]
        q = "\n\nSummarise the words above in one long paragraph."
        with ThreadPoolExecutor(4) as ex:
            futs = [ex.submit(w.chat, [{"role": "user", "content": big + q}], max_tokens=256, temperature=0.7,
                              chat_template_kwargs={"enable_thinking": False})]
            n_text = 2 if a.images else 3
            futs += [ex.submit(w.chat, [{"role": "user", "content": s + q}], max_tokens=1024, temperature=0.7,
                               ignore_eos=True, chat_template_kwargs={"enable_thinking": False})
                     for s in sides[:n_text]]
            if a.images:
                parts = [{"type": "image_url", "image_url": {"url": png_data_url(rd * 1000 + i)}}
                         for i in range(a.images)]
                parts.append({"type": "text", "text": sides[2][:2000] + "\n\nDescribe each image briefly."})
                futs.append(ex.submit(w.chat, [{"role": "user", "content": parts}], max_tokens=1024,
                                      temperature=0.7, ignore_eos=True,
                                      chat_template_kwargs={"enable_thinking": False}))
            for n, f in zip(["big", "side0", "side1", "images" if a.images else "side2"], futs):
                r = f.result()
                rows.append({"round": rd, "req": n, "error": r.get("error"),
                             "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                             "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                             "ttft_s": r.get("ttft_s"), "e2e_s": r.get("e2e_s")})
    print(json.dumps({"label": a.label, "rows": rows,
                      "errors": [x for x in rows if x["error"] or not x["completion_tokens"]]}, indent=1))


if __name__ == "__main__":
    main()
