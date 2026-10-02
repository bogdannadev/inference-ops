#!/usr/bin/env python3
"""A stand-in SGLang worker for router failure drills. No GPU, stdlib only.

Speaks just enough of the engine API for sglang_router: /health, the info
endpoints, and streaming or plain /v1/chat/completions and /generate. Every
response names the worker, so the driver can see where the router sent it.

The behaviour is read per request from /ctl/<name>.mode (a bind-mounted dir):
  ok          normal: /health ~1 s (like the engine's generation probe),
              a stream of --chunks chunks every 0.2 s
  sched_hang  the engine with a stuck scheduler: the HTTP server is up, but
              /health waits 20 s and answers 503 (the engine's own
              HEALTH_CHECK_TIMEOUT), and requests never get a byte
  slow        a KV-stranded replica: /health is fine, but the first byte of a
              request comes only after 120 s in the queue
  slow503     the same replica with SGLANG_REQ_WAITING_TIMEOUT=30: after 30 s in
              the queue the engine aborts the request with 503 (retryable)
A frozen process (docker pause) and a crash (docker stop) need no mode.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = os.environ["WORKER_NAME"]
PORT = int(os.environ.get("PORT", "8000"))


def mode():
    try:
        return open(f"/ctl/{NAME}.mode").read().strip() or "ok"
    except OSError:
        return "ok"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {NAME} {mode()} {fmt % args}\n")

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        m = mode()
        if self.path.startswith("/health"):
            if m == "sched_hang":
                time.sleep(20)
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            time.sleep(1)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # Production: the router has no --api-key, so the engine answers its
        # discovery calls with 401 and the router routes with model "unknown".
        self._json(401, {"error": "Unauthorized"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        m = mode()
        if m == "sched_hang":
            time.sleep(3600)
            return
        if m == "slow":
            time.sleep(120)
        if m == "slow503":
            time.sleep(30)
            return self._json(503, {"object": "error", "message": "Request waiting timeout reached.",
                                    "code": 503})
        chunks = int(body.get("max_tokens") or body.get("sampling_params", {}).get("max_new_tokens") or 20)
        if not body.get("stream"):
            time.sleep(0.2 * chunks)
            return self._json(200, {"id": "x", "object": "chat.completion", "model": "qwen36-27b",
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": NAME}}],
                                    "usage": {"prompt_tokens": 1, "completion_tokens": chunks}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(s):
            d = s.encode()
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n")
            self.wfile.flush()
        try:
            for _ in range(chunks):
                send("data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": NAME}}]}) + "\n\n")
                time.sleep(0.2)
            send("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
