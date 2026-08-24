#!/usr/bin/env python3
"""
E1 - cuBLASLt split-K workspace sweep.

Question
--------
The July trace shows cuBLASLt already splitting K on the narrow-output
projections (grid 200 = 40 output tiles x a 5-way split) yet still reaching
only 1.85 blocks/SM and ~58% of HBM peak, while the wide gate_up projection
reaches 5.04 blocks/SM and ~82%. cuBLASLt sizes split-K workspace from the
workspace it is given and declines configurations that do not fit.
CUBLASLT_WORKSPACE_SIZE is unset on both replicas.

Does raising it change the chosen kernel's GRID on the starved shapes?

Gate
----
Grid before bandwidth. A bandwidth delta with an unchanged grid is clock drift
or noise (this node's floor is ~1.5%, clocks unlocked), not split-K. The
baseline config is measured first AND last so drift is visible.

Design notes
------------
* PyTorch reads CUBLASLT_WORKSPACE_SIZE once, when it creates the cuBLASLt
  handle, so the sweep must run one child process per setting. This file is
  both the driver and the worker (--worker).
* Grid is recovered the same way the S1 table was built: export a chrome trace
  and parse the kernel events' "grid" field. No ncu, so no SYS_ADMIN needed.
* Touches no container and no engine. Allocates one weight at a time and frees
  it before the next, so peak VRAM is one weight plus a CUDA context.

Usage
-----
    python3 e1_gemm_workspace.py --gpu 1 --confirm
    python3 e1_gemm_workspace.py --gpu 1 --confirm --include-lm-head   # needs ~3.1 GB
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile

# Qwen3.8-27B, from config.json: hidden 5120, intermediate 17408, 64 layers
# (48 linear-attention + 16 full-attention), vocab 248320.
#   name              N (out)   K (in)    note
SHAPES = [
    ("mlp_gate_up",     34816,   5120, "trace grid 544, 5.04 blk/SM, ~82% peak - the healthy one"),
    ("mlp_down",         5120,  17408, "trace grid 200 (40 tiles x splitK 5), 1.85 blk/SM, ~58%"),
    ("attn_qkv",         8192,   5120, "16 full-attn layers: 24*256 q + 4*256 k + 4*256 v"),
    ("attn_o_proj",      5120,   6144, "narrow output - starved class"),
    ("gdn_in_proj_qkvz",16384,   5120, "48 GDN layers: 2048 q + 2048 k + 6144 v + 6144 z"),
    ("gdn_out_proj",     5120,   6144, "narrow output - starved class"),
]
LM_HEAD = ("lm_head", 248320, 5120, "trace grid 1940 (970 tiles x splitK 2), 17.96 blk/SM")

# M = rows presented to the GEMM during decode.
#   1  = single stream, no speculation
#   6  = one EAGLE draft step  (--speculative-num-draft-tokens 6)
#  24  = production target verify: max-running-requests 4 x 6 draft tokens
#  48  = what batch 8 would look like (still inside one 64-row tile)
M_VALUES = [1, 6, 24, 48]

# KiB. PyTorch parses CUBLASLT_WORKSPACE_SIZE in KiB. None = leave unset,
# i.e. exactly what production does today.
WORKSPACES_KIB = [None, 1024, 8192, 32768, 131072]

BYTES_PER_ELEM = 2  # bfloat16
PEAK_BW = 1935e9    # A100 80GB PCIe, HBM2e


# ---------------------------------------------------------------- worker ----

def _bytes_moved(m: int, n: int, k: int) -> int:
    """Weight matrix + input activations + output activations."""
    return (n * k + m * k + m * n) * BYTES_PER_ELEM


def _sample_gpu(idx: int) -> dict:
    """SM clock and power, so a confounded comparison is visible in the data."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--format=csv,noheader,nounits",
             "--query-gpu=clocks.sm,power.draw,temperature.gpu"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        clk, pwr, tmp = (x.strip() for x in out.split(","))
        return {"sm_clock_mhz": float(clk), "power_w": float(pwr), "temp_c": float(tmp)}
    except Exception as exc:                                   # noqa: BLE001
        return {"error": str(exc)}


def _time_linear(torch, w, x, iters: int, warmup: int) -> float:
    """Median seconds per F.linear call, timed with CUDA events."""
    import torch.nn.functional as F

    for _ in range(warmup):
        F.linear(x, w)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        F.linear(x, w)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def _capture_grid(torch, w, x, iters: int = 6) -> list[dict]:
    """Profile a few calls and read the selected kernel's name and grid.

    Same method that produced the S1 launch-geometry table: export a chrome
    trace and parse the kernel events, rather than relying on the profiler's
    Python-side aggregation, which does not surface grid.
    """
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            for _ in range(iters):
                F.linear(x, w)
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            trace = json.load(fh)

        def _prod(v):
            return int(v[0]) * int(v[1]) * int(v[2]) if isinstance(v, list) and len(v) == 3 else None

        found: dict[str, dict] = {}
        for ev in trace.get("traceEvents", []):
            if ev.get("cat") != "kernel":
                continue
            a = ev.get("args", {}) or {}
            grid = _prod(a.get("grid"))
            if grid is None:
                continue
            rec = found.setdefault(ev["name"], {
                "kernel": ev["name"], "grid": grid, "n": 0, "dur_us": 0.0,
                "block": _prod(a.get("block")),
                "warps_per_sm": a.get("warps per SM"),
                "regs_per_thread": a.get("registers per thread"),
                "shared_bytes": a.get("shared memory"),
            })
            rec["n"] += 1
            rec["dur_us"] += float(ev.get("dur", 0.0))
        return sorted(found.values(), key=lambda r: -r["dur_us"])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_worker(args) -> int:
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"error": "no CUDA device visible"}))
        return 1

    torch.cuda.set_device(0)          # CUDA_VISIBLE_DEVICES already narrowed it
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)

    shapes = list(SHAPES) + ([LM_HEAD] if args.include_lm_head else [])
    result = {
        "workspace_kib": os.environ.get("CUBLASLT_WORKSPACE_SIZE", None),
        "device": props.name,
        "capability": f"{props.major}.{props.minor}",
        "l2_bytes": getattr(props, "L2_cache_size", None),
        "persisting_l2_max": getattr(props, "persisting_l2_cache_max_size", None),
        "free_mib_before": free // 1024 // 1024,
        "total_mib": total // 1024 // 1024,
        "torch": torch.__version__,
        "gpu_before": _sample_gpu(args.gpu),
        "rows": [],
    }

    for name, n, k, note in shapes:
        need = _bytes_moved(max(M_VALUES), n, k) + 512 * 1024 * 1024   # + context margin
        free_now, _ = torch.cuda.mem_get_info()
        if need > free_now:
            result["rows"].append({
                "shape": name, "N": n, "K": k,
                "skipped": f"needs ~{need // 2**20} MiB, {free_now // 2**20} MiB free",
            })
            continue

        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        for m in M_VALUES:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            secs = _time_linear(torch, w, x, args.iters, args.warmup)
            row = {
                "shape": name, "note": note, "M": m, "N": n, "K": k,
                "seconds": secs,
                "gb_per_s": _bytes_moved(m, n, k) / secs / 1e9,
                "pct_of_peak": 100.0 * _bytes_moved(m, n, k) / secs / PEAK_BW,
            }
            if args.capture_grid:
                row["kernels"] = _capture_grid(torch, w, x)
            result["rows"].append(row)
            del x
        del w
        torch.cuda.empty_cache()

    result["gpu_after"] = _sample_gpu(args.gpu)
    print(json.dumps(result))
    return 0


# ---------------------------------------------------------------- driver ----

def _fmt_grid(kernels) -> str:
    if not kernels:
        return "-"
    top = kernels[0]
    blocks_per_sm = top["grid"] / 108.0
    return f'{top["grid"]}({blocks_per_sm:.2f}/SM)'


def run_driver(args) -> int:
    here = os.path.abspath(__file__)
    order = list(WORKSPACES_KIB) + [None]        # baseline repeated last: drift control
    runs = []

    for i, ws in enumerate(order):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.pop("CUBLASLT_WORKSPACE_SIZE", None)
        if ws is not None:
            env["CUBLASLT_WORKSPACE_SIZE"] = str(ws)

        label = "unset(baseline)" if ws is None else f"{ws} KiB"
        if i == len(order) - 1:
            label = "unset(control-repeat)"
        print(f"[{i + 1}/{len(order)}] workspace = {label} ...", file=sys.stderr, flush=True)

        cmd = [sys.executable, here, "--worker", "--gpu", str(args.gpu),
               "--iters", str(args.iters), "--warmup", str(args.warmup)]
        if args.capture_grid:
            cmd.append("--capture-grid")
        if args.include_lm_head:
            cmd.append("--include-lm-head")

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        data["label"] = label
        runs.append(data)

    out = {"experiment": "e1_gemm_workspace", "gpu": args.gpu,
           "m_values": M_VALUES, "workspaces_kib": WORKSPACES_KIB, "runs": runs}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- report -----------------------------------------------------------
    print(f"\nwrote {args.out}\n")
    base = runs[0]
    print(f"{'shape':18s} {'M':>3s} " + "".join(f"{r['label'][:14]:>16s}" for r in runs))
    print("-" * (22 + 16 * len(runs)))
    for idx, row in enumerate(base["rows"]):
        if "skipped" in row:
            print(f"{row['shape']:18s}  -   SKIPPED: {row['skipped']}")
            continue
        cells = []
        for r in runs:
            rr = r["rows"][idx]
            cells.append(f"{rr.get('pct_of_peak', 0):6.1f}% {_fmt_grid(rr.get('kernels')):>9s}"
                         if args.capture_grid else f"{rr.get('pct_of_peak', 0):15.1f}%")
        print(f"{row['shape']:18s} {row['M']:3d} " + "".join(f"{c:>16s}" for c in cells))

    print("\nclock / power (confound check):")
    for r in runs:
        b, a = r.get("gpu_before", {}), r.get("gpu_after", {})
        print(f"  {r['label']:22s} {b.get('sm_clock_mhz','?')}->{a.get('sm_clock_mhz','?')} MHz  "
              f"{b.get('power_w','?')}->{a.get('power_w','?')} W  "
              f"{b.get('temp_c','?')}->{a.get('temp_c','?')} C")

    first, last = runs[0], runs[-1]
    print("\ndrift (baseline first vs last -- must be small or the sweep is meaningless):")
    for i, row in enumerate(first["rows"]):
        if "skipped" in row:
            continue
        a, b = row.get("pct_of_peak"), last["rows"][i].get("pct_of_peak")
        if a and b:
            print(f"  {row['shape']:18s} M={row['M']:<3d} {a:5.1f}% -> {b:5.1f}%  "
                  f"({100 * (b - a) / a:+.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=1, help="physical GPU index (r0=0, r1=1)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--capture-grid", action="store_true", default=True,
                    help="profile each shape to recover kernel name and grid (the actual gate)")
    ap.add_argument("--no-capture-grid", dest="capture_grid", action="store_false")
    ap.add_argument("--include-lm-head", action="store_true",
                    help="add the 248320x5120 lm_head; needs ~3.1 GB free")
    ap.add_argument("--out", default="tuning/lowlevel/results/e1_gemm_workspace.json")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--confirm", action="store_true",
                    help="required: acknowledges this puts load on a serving GPU")
    args = ap.parse_args()

    if args.worker:
        return run_worker(args)

    if not args.confirm:
        ap.error("refusing to run without --confirm: this allocates on a GPU that serves "
                 "production. Drain the replica from the router first (see README).")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
