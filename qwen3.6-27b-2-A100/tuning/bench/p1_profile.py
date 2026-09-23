"""P1 profile captures on ONE drained worker: decode window + prefill shapes.

Run through run_eval.sh (python:3.12-slim on the backend network), normally by
tuning/lowlevel/bench/p1_profile.sh, which does the drain/DCGM bracketing:

    ./tuning/bench/run_eval.sh r1 p1_profile <label> [--only decode,cold4k,...]

Each scenario triggers the engine's own torch profiler through /start_profile
with a bounded num_steps, so the engine writes the trace itself under
/app/logs/profile_p1/<label>/<scenario>/ (host: logs/<replica>/profile_p1/...).
with_stack is OFF: the July S1 capture left it at the default and produced a
793 MB trace that json.load cannot parse.

Scenarios (tuning/docs/PREFILL_KERNELS.md Phase 1, plus the decode bubble track):
  decode      4 concurrent greedy streams, 40 forwards captured mid-decode
  cold4k      cold 4,096-token prefill, whole request (1 chunk)
  cold32k     cold 32,768, whole request (8 chunks)
  cold128k_a  cold 131,072, the FIRST 4 chunks
  cold128k_z  cold 131,072, 4 chunks starting ~75% of the way through
  ext16k      16,384 new tokens on a cached 65,536-token prefix

Prefill prompts are random token ids from a seeded RNG, so no two scenarios
share a prefix and the flush is verified before each one ("Cache flushed." or
the scenario aborts). Output JSON (stdout) records timings and the trace dirs.
"""
import argparse
import json
import random
import sys
import threading
import time

from evalkit import Worker

CHUNK = 4096            # --chunked-prefill-size in production
TRACE_ROOT = "/app/logs/profile_p1"

DECODE_PROMPTS = [
    "Explain in detail how a B-tree handles insertion and node splits, with a worked example.",
    "Write a thorough walkthrough of TCP congestion control: slow start, AIMD, fast recovery.",
    "Describe step by step how a compiler performs register allocation with graph colouring.",
    "Explain how the Raft consensus algorithm elects a leader and replicates a log, in depth.",
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def ids(rng, n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def start_profile(w, out_dir, num_steps):
    st, txt = w.post_raw("/start_profile", {
        "output_dir": out_dir, "num_steps": num_steps,
        "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": True,
    })
    if st != 200 or "already" in txt.lower():
        raise RuntimeError(f"/start_profile {st}: {txt[:200]}")
    return txt.strip()


def generate(w, prompt, max_new=1):
    t = time.time()
    st, txt = w.post_raw("/generate", {"input_ids": prompt,
                                       "sampling_params": {"temperature": 0, "max_new_tokens": max_new}},
                         timeout=1800)
    if st != 200:
        raise RuntimeError(f"/generate {st}: {txt[:200]}")
    return round(time.time() - t, 2)


def close_capture(w, rng):
    """Without start_step, the engine captures num_steps forwards and stops only
    when the NEXT forward begins (profiler_manager.py: target = ct + num_steps + 1,
    checked before each batch). On a drained, idle worker nothing else arrives,
    so send a tiny request to close the capture; it is not itself recorded."""
    generate(w, ids(rng, 8))


def must_flush(w):
    if not w.flush():
        raise RuntimeError("flush refused for 90 s -- something is still in flight; not measuring")


def settle(w, wait_s=60):
    """The profiler writes its trace after the last captured step; give it time
    and make sure the worker is idle before the next flush."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        m = w.metrics()
        busy = sum(float(l.rsplit(" ", 1)[1]) for l in m.splitlines()
                   if l.startswith(("sglang:num_running_reqs", "sglang:num_queue_reqs")))
        if busy == 0:
            break
        time.sleep(1)
    time.sleep(10)


def sc_decode(w, rng, out):
    must_flush(w)
    res = [None] * len(DECODE_PROMPTS)

    def run(i, p):
        res[i] = w.chat([{"role": "user", "content": p}], temperature=0, max_tokens=700,
                        ignore_eos=True, chat_template_kwargs={"enable_thinking": False})

    th = [threading.Thread(target=run, args=(i, p)) for i, p in enumerate(DECODE_PROMPTS)]
    for t in th:
        t.start()
    time.sleep(4)                       # past the prefills, into steady batch-4 decode
    msg = start_profile(w, out, 40)
    for t in th:
        t.join()
    return {"profile": msg, "decode_tok_s": [r.get("decode_tok_s") for r in res],
            "errors": [r["error"] for r in res if "error" in r]}


def sc_cold(w, rng, out, n, skip_frac=None, steps=None):
    must_flush(w)
    prompt = ids(rng, n)
    chunks = -(-n // CHUNK)
    if skip_frac is None:
        # max_new_tokens=1 means exactly `chunks` forwards; asking for one more
        # records whatever unrelated forward comes next.
        msg = start_profile(w, out, steps or chunks)
        e2e = generate(w, prompt)
        close_capture(w, rng)
        return {"profile": msg, "tokens": n, "chunks": chunks, "e2e_s": e2e}
    # Late window of a long prefill: /start_profile's start_step is an absolute
    # forward count we cannot read from outside, so time it off the known cold
    # rate instead. The trace's attention-kernel durations show where it landed.
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("e2e_s", generate(w, prompt)))
    th.start()
    time.sleep(skip_frac * res_hint(n))
    msg = start_profile(w, out, steps or 4)
    th.join()
    return {"profile": msg, "tokens": n, "chunks": chunks, "e2e_s": res.get("e2e_s"),
            "started_at_s": round(skip_frac * res_hint(n), 1)}


def res_hint(n):
    # Cold 128K measured 58.5 s on r1 with the profiler armed (2026-09-24,
    # cold128k_a). The 67.8 s in TRIAL_V0520_LEVERS.md put an 85% start past
    # the end of the prefill, so the window then spilled into the next request.
    return 58.5 * n / 131072


def sc_ext(w, rng, out, prefix_n=65536, new_n=16384):
    must_flush(w)
    prefix = ids(rng, prefix_n)
    t_prefix = generate(w, prefix)
    settle(w, 20)
    msg = start_profile(w, out, -(-new_n // CHUNK))
    t_ext = generate(w, prefix + ids(rng, new_n))
    close_capture(w, rng)
    return {"profile": msg, "prefix_s": t_prefix, "ext_s": t_ext}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", default="decode,cold4k,cold32k,cold128k_a,cold128k_z,ext16k")
    a = ap.parse_args()

    w = Worker(a.host, a.port, consumer=f"p1-profile-{a.label}")
    rng = random.Random(20260924)
    root = f"{TRACE_ROOT}/{a.label}"
    table = {
        "decode":     lambda o: sc_decode(w, rng, o),
        "cold4k":     lambda o: sc_cold(w, rng, o, 4096),
        "cold32k":    lambda o: sc_cold(w, rng, o, 32768),
        "cold128k_a": lambda o: sc_cold(w, rng, o, 131072, steps=4),
        "cold128k_z": lambda o: sc_cold(w, rng, o, 131072, skip_frac=0.75),
        "ext16k":     lambda o: sc_ext(w, rng, o),
    }
    results = {"label": a.label, "worker": f"{a.host}:{a.port}", "scenarios": {}}
    for name in a.only.split(","):
        out = f"{root}/{name}"
        log(f"== {name} -> {out}")
        t0 = time.time()
        try:
            r = table[name](out)
        except Exception as e:          # record and stop: a failed flush poisons the rest
            results["scenarios"][name] = {"error": str(e), "trace_dir": out}
            log(f"   FAILED: {e}")
            break
        r["trace_dir"], r["wall_s"] = out, round(time.time() - t0, 1)
        results["scenarios"][name] = r
        log(f"   {json.dumps(r)[:300]}")
        settle(w)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
