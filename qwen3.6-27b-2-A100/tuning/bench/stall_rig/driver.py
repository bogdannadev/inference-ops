#!/usr/bin/env python3
"""Load driver for the router failure drill (see run_drill.sh).

SESSIONS agent-like sessions, each strictly serial like a real agent: send a
turn, wait for the full streamed reply (or the client timeout), pause, send the
next turn with the history grown. Each session's text is distinctive, so
cache_aware pins it to a worker the way a real conversation is pinned. A new
session also starts every --new-every seconds, to show where fresh work lands.

One JSON line per request on stdout: send time (s since start), session, the
worker that answered, time to first byte, total, and the outcome.
"""
import argparse
import http.client
import json
import random
import sys
import threading
import time

T0 = time.time()
lock = threading.Lock()
INFLIGHT = {}   # id -> rec, for requests still open when the drill ends


def emit(rec):
    with lock:
        print(json.dumps(rec), flush=True)


def turn(a, sid, history):
    body = {"model": "qwen36-27b", "stream": True, "max_tokens": 25,
            "messages": [{"role": "user", "content": history}]}
    t = time.time()
    rec = {"t": round(t - T0, 1), "sid": sid, "worker": None, "ttfb": None, "total": None, "outcome": None}
    with lock:
        INFLIGHT[id(rec)] = rec
    try:
        return _turn(a, body, t, rec)
    finally:
        with lock:
            INFLIGHT.pop(id(rec), None)


def _turn(a, body, t, rec):
    try:
        c = http.client.HTTPConnection(a.host, a.port, timeout=a.client_timeout)
        c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            rec["outcome"] = f"HTTP {r.status}"
            rec["total"] = round(time.time() - t, 1)
            return rec
        while True:
            line = r.readline()
            if not line:
                break
            if line.startswith(b"data: {") and rec["worker"] is None:
                rec["ttfb"] = round(time.time() - t, 1)
                rec["worker"] = json.loads(line[6:])["choices"][0]["delta"]["content"]
        rec["outcome"] = "ok" if rec["worker"] else "empty"
    except Exception as e:
        rec["outcome"] = type(e).__name__
    rec["total"] = round(time.time() - t, 1)
    return rec


def session(a, sid, stop_at):
    rng = random.Random(sid)
    words = " ".join(rng.choice("alpha beta gamma delta epsilon zeta eta theta iota kappa".split())
                     for _ in range(400))
    history = f"session {sid}: {words}"
    while time.time() < stop_at:
        emit(turn(a, sid, history))
        history += f" turn {rng.random()}"
        time.sleep(a.think_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rtr")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--duration", type=int, default=720)
    ap.add_argument("--think-s", type=float, default=3)
    ap.add_argument("--new-every", type=int, default=20)
    ap.add_argument("--client-timeout", type=int, default=900, help="the gateway's idle timeout")
    a = ap.parse_args()
    stop_at = T0 + a.duration
    th = [threading.Thread(target=session, args=(a, f"s{i}", stop_at), daemon=True) for i in range(a.sessions)]
    for t in th:
        t.start()
        time.sleep(0.5)
    n = 0
    while time.time() < stop_at:
        time.sleep(a.new_every)
        n += 1
        t = threading.Thread(target=lambda k=n: emit(turn(a, f"new{k}", f"fresh session {k} " * 200)), daemon=True)
        t.start()
    time.sleep(5)
    with lock:
        left = list(INFLIGHT.values())
    for rec in left:
        rec = dict(rec, outcome="still_open_at_end", total=round(time.time() - T0 - rec["t"], 1))
        print(json.dumps(rec), flush=True)
    sys.stderr.write(f"driver done, {len(left)} still open\n")


if __name__ == "__main__":
    main()
