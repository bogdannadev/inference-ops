#!/usr/bin/env python3
"""
Render one ladder result, or diff two.

  python3 compare.py results/ladder_r1_treatment_post.json
  python3 compare.py results/ladder_r0_control_pre.json \
                     results/ladder_r1_treatment_post.json

The wave model is the interesting column. With an admission cap of C, a rung of
n concurrent requests runs in ceil(n/C) waves, so wall time tracks
ceil(n/C) * wall(C) and latency_max/p50 approaches ceil(n/C). Watching where
that ratio departs from 1.0 locates the effective cap WITHOUT reading the
config -- which is the point, since the config is what we are testing.
"""
import json
import math
import sys


def rows(doc):
    out = []
    for p in doc.get("passes", []):
        out.extend(p)
    return out


def fmt(doc, title=None):
    print(f"\n=== {title or doc.get('label')}  ({doc.get('host')}:{doc.get('port')}) ===")
    print(f"    started {doc.get('started_utc')}  max_tokens={doc.get('max_tokens')}")
    hdr = (f"{'c':>3} {'wall':>7} {'agg':>8} {'per':>7} {'p50':>7} {'p95':>7} "
           f"{'max':>7} {'mx/p50':>7} {'fail':>5} "
           f"{'occ%':>6} {'dram%':>6} {'sm%':>6} {'clk':>6} {'W':>6}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in rows(doc):
        t = r.get("telemetry") or {}

        def g(k, scale=1.0):
            v = t.get(k)
            return f"{v * scale:6.1f}" if isinstance(v, (int, float)) else "     -"

        print(f"{r['concurrency']:>3} {r['wall_time_s']:>7.3f} "
              f"{(r['aggregate_tok_s'] or 0):>8.2f} {(r['per_stream_tok_s'] or 0):>7.2f} "
              f"{(r['latency_p50_s'] or 0):>7.3f} {(r['latency_p95_s'] or 0):>7.3f} "
              f"{(r['latency_max_s'] or 0):>7.3f} "
              f"{(r.get('latency_max_over_p50') or 0):>7.2f} {r['failed']:>5} "
              f"{g('dcgm.DCGM_FI_PROF_SM_OCCUPANCY.mean', 100)} "
              f"{g('dcgm.DCGM_FI_PROF_DRAM_ACTIVE.mean', 100)} "
              f"{g('dcgm.DCGM_FI_PROF_SM_ACTIVE.mean', 100)} "
              f"{g('dcgm.DCGM_FI_DEV_SM_CLOCK.mean')} "
              f"{g('dcgm.DCGM_FI_DEV_POWER_USAGE.mean')}")

    # Confounder check: clocks must be pinned across the whole ladder.
    clks = [(r.get("telemetry") or {}).get("dcgm.DCGM_FI_DEV_SM_CLOCK.mean")
            for r in rows(doc)]
    clks = [c for c in clks if c]
    if clks and (max(clks) - min(clks)) > 30:
        print(f"    !! SM clock varied {min(clks):.0f}-{max(clks):.0f} MHz across the "
              f"ladder. Throughput comparisons are confounded by clock, not config.")

    # Infer the effective admission cap from the wave signature.
    cap = infer_cap(doc)
    if cap:
        print(f"    inferred effective admission cap ~= {cap} "
              f"(best fit of wall(n) ~= ceil(n/cap) * wall(1))")


def infer_cap(doc):
    """Infer the effective admission cap from WALL-TIME scaling.

    With a cap of C, a rung of n requests runs in ceil(n/C) waves, so
        wall(n) ~= ceil(n/C) * wall(1).

    An earlier version of this used latency_max/p50 and gave the wrong answer
    on the r0 control baseline (reported 6, actual 2). The reason: with n=4 and
    C=2 the latencies are [t, t, 2t, 2t], so p50 lands in the SECOND wave and
    max/p50 collapses to ~1.0. The median is inside the queue, so the ratio
    says nothing. Wall time does not have that failure mode.
    """
    rs = sorted(rows(doc), key=lambda r: r["concurrency"])
    base = next((r["wall_time_s"] for r in rs if r["concurrency"] == 1), None)
    if not base:
        return None
    best, best_err = None, float("inf")
    for cand in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        err = 0.0
        for r in rs:
            predicted = math.ceil(r["concurrency"] / cand) * base
            err += abs(r["wall_time_s"] - predicted) / r["wall_time_s"]
        err /= len(rs)
        if err < best_err:
            best, best_err = cand, err
    # Above ~15% mean relative error the wave model does not describe the data
    # at all -- report nothing rather than a confident wrong number.
    return best if best_err < 0.15 else None


def diff(a, b):
    fmt(a, f"A: {a.get('label')}")
    fmt(b, f"B: {b.get('label')}")
    ra = {r["concurrency"]: r for r in rows(a)}
    rb = {r["concurrency"]: r for r in rows(b)}
    print(f"\n=== DELTA  B vs A ===")
    print(f"{'c':>3} {'agg A':>9} {'agg B':>9} {'ratio':>7} {'p50 A':>7} {'p50 B':>7} {'p50 x':>7}")
    for c in sorted(set(ra) & set(rb)):
        x, y = ra[c], rb[c]
        ax, bx = x["aggregate_tok_s"] or 0, y["aggregate_tok_s"] or 0
        px, py = x["latency_p50_s"] or 0, y["latency_p50_s"] or 0
        print(f"{c:>3} {ax:>9.2f} {bx:>9.2f} "
              f"{(bx / ax if ax else 0):>7.2f} {px:>7.3f} {py:>7.3f} "
              f"{(py / px if px else 0):>7.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fmt(json.load(open(sys.argv[1])))
    elif len(sys.argv) == 3:
        diff(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])))
    else:
        print(__doc__)
        raise SystemExit(2)
