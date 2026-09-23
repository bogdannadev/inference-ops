#!/usr/bin/env python3
"""Summarise a torch-profiler trace from /start_profile: where GPU time goes,
and how much of the working time the GPU spends idle between kernels.

    p1_trace_analyze.py TRACE.json.gz [--top 25] [--json out.json]

Streams the file line by line. The July S1 trace is 793 MB gzipped (the
profiler records Python stacks by default) and json.load() on it does not
finish, so events are cut out line by line (see events()).

Buckets are regexes over kernel names, first match wins. The "busy" figure is
the union of kernel intervals across ALL streams, so overlap (DFlash draft on a
side stream, HiCache copies) is not double counted.

Idle time is split by gap length. A profiler window always includes stretches
with no load at all (before the first request, after the last), so the raw
window idle is meaningless. Gaps shorter than --step-gap-ms are bubbles INSIDE
the work: launch latency, host sync, stream waits. "active" = busy + those
bubbles, and "bubble" = bubbles / active is the inter-kernel figure the
megakernel question needs. Ramp/drain INSIDE a kernel is not visible here; it
shows up as kernel time and is sized by E1 (tuning/lowlevel/README.md).
"""
import argparse, collections, gzip, json, re, sys

BUCKETS = [
    ("norm_rope",     r"rmsnorm|RMSNorm|norm_|rope|rotary"),
    ("gdn_state",     r"mamba_state|state_scatter|state_gather"),
    ("gemm",          r"gemm|s16816|cutlass|cublas|xmma|splitKreduce|marlin"),
    ("gdn_recurrent", r"delta_rule_update|fused_sigmoid_gating|recurrent"),
    ("gdn_chunk",     r"chunk_|solve_tril|recompute_w_u|kkt|cumsum|fwd_h|fwd_o|l2norm|merge_16x16|gdn"),
    ("conv1d",        r"causal_conv1d|conv1d"),
    ("attention",     r"flashinfer|BatchPrefill|BatchDecode|Prefill.*Kernel|fmha|flash_|_fwd_kernel|attention"),
    ("spec_sampling", r"dflash|eagle|verify|tree|sampl|top_?k|top_?p|argmax|softmax"),
    ("copy",          r"Memcpy|Memset|copy|Copy|cat|Cat|gather|scatter|index"),
    ("elementwise",   r"elementwise|vectorized|at::native|triton_|silu|sigmoid|mul"),
]
BUCKET_RE = [(n, re.compile(p)) for n, p in BUCKETS]

NAME_RE = re.compile(r'"name": "((?:[^"\\]|\\.)*)"')
TS_RE = re.compile(r'"ts": ([0-9.]+)')
DUR_RE = re.compile(r'"dur": ([0-9.]+)')
STREAM_RE = re.compile(r'"stream": (\d+)')
GRID_RE = re.compile(r'"grid": \[(\d+), (\d+), (\d+)\]')
REGS_RE = re.compile(r'"registers per thread": (\d+)')


def events(path):
    """Yield (name, ts_us, dur_us, stream, grid, regs) for every GPU kernel and memcpy/memset.

    Events are cut out by brace depth, not parsed as JSON: torch 2.13 puts every
    field on its own line, older torch put several on one line and left trailing
    commas inside "args" (invalid JSON). Brace counting handles both."""
    op = gzip.open if path.endswith(".gz") else open
    block, depth, in_events = None, 0, False
    with op(path, "rt", errors="replace") as f:
        for line in f:
            if not in_events:              # header objects (deviceProperties...) come first
                in_events = '"traceEvents"' in line
                continue
            if block is None:
                if line.strip() != "{":
                    continue
                block, depth = [], 0
            block.append(line)
            depth += line.count("{") - line.count("}")
            if depth > 0:
                continue
            txt = "".join(block)
            block = None
            if '"ph": "X"' not in txt or not ('"cat": "kernel"' in txt or '"cat": "gpu_mem' in txt):
                continue
            n, t, d = NAME_RE.search(txt), TS_RE.search(txt), DUR_RE.search(txt)
            if not (n and t and d):
                continue
            st, g, r = STREAM_RE.search(txt), GRID_RE.search(txt), REGS_RE.search(txt)
            yield (n.group(1), float(t.group(1)), float(d.group(1)),
                   int(st.group(1)) if st else -1,
                   int(g.group(1)) * int(g.group(2)) * int(g.group(3)) if g else None,
                   int(r.group(1)) if r else None)


def bucket(name):
    for b, rx in BUCKET_RE:
        if rx.search(name):
            return b
    return "other"


def short(name, n=90):
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"<.*", "<…>", name)
    return name[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json")
    ap.add_argument("--step-gap-ms", type=float, default=1.0,
                    help="gaps shorter than this count as in-step bubbles")
    a = ap.parse_args()

    by_bucket = collections.Counter()
    by_kernel = collections.defaultdict(lambda: [0, 0.0, None, None])  # count, total, grid, regs
    intervals = []
    for name, ts, dur, stream, grid, regs in events(a.trace):
        b = bucket(name)
        by_bucket[b] += dur
        k = by_kernel[(short(name), grid)]
        k[0] += 1; k[1] += dur; k[2] = grid; k[3] = regs
        intervals.append((ts, ts + dur))

    if not intervals:
        sys.exit("no GPU kernel events found (was the GPU activity recorded?)")
    total = sum(by_bucket.values())

    # Union of intervals across streams -> busy time; the rest of the window is idle.
    intervals.sort()
    busy, cur_s, cur_e, gaps = 0.0, *intervals[0], []
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    window = intervals[-1][1] - intervals[0][0]

    thr = a.step_gap_ms * 1000
    bubbles = sum(g for g in gaps if g < thr)
    active = busy + bubbles
    print(f"trace   {a.trace}")
    print(f"kernels {len(intervals):,}   kernel time {total/1e6:.3f} s   "
          f"window {window/1e6:.3f} s")
    print(f"active  {active/1e6:.3f} s   busy {busy/active:.1%}   "
          f"in-step bubbles (<{a.step_gap_ms:g} ms) {bubbles/active:.1%}   "
          f"no-load gaps {(window-active)/1e6:.3f} s")
    gh = collections.Counter()
    for g in gaps:
        gh["<2us" if g < 2 else "2-10us" if g < 10 else "10-100us" if g < 100 else
           "0.1-1ms" if g < 1000 else "no-load"] += g
    print("bubbles by gap size (share of active):  " + "  ".join(
        f"{k} {gh[k]/active:.1%}" for k in ["<2us", "2-10us", "10-100us", "0.1-1ms"]))
    print("\nbucket           share of kernel time")
    for b, t in by_bucket.most_common():
        print(f"  {b:<15} {t/total:7.2%}   {t/1e3:10.1f} ms")
    print(f"\ntop {a.top} kernels (name, grid)                                                   "
          "count    total ms   mean us  share  regs")
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[: a.top]
    for (name, grid), (cnt, tot, _, regs) in top:
        print(f"  {name:<80} {str(grid):>6} {cnt:7d} {tot/1e3:10.1f} {tot/cnt:9.1f} "
              f"{tot/total:6.2%} {regs if regs is not None else '-':>5}")

    if a.json:
        json.dump({
            "trace": a.trace, "kernel_time_us": total, "window_us": window, "busy_us": busy,
            "active_us": active, "bubble_us": bubbles, "step_gap_ms": a.step_gap_ms,
            "idle_by_gap_us": dict(gh),
            "buckets": {b: t for b, t in by_bucket.items()},
            "top": [{"name": n, "grid": g, "count": c, "total_us": t, "regs": r}
                    for (n, g), (c, t, _, r) in top],
        }, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
