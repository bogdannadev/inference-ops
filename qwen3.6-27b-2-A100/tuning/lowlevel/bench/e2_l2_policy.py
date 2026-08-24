#!/usr/bin/env python3
"""
E2 - L2 behaviour: is E1 confounded, and does the residency mechanism work?

Reshaped after E1. The original plan was "persist the KV pool in L2", which the
arithmetic kills before it starts: KV at the p95 context of 141K tokens is
~9.2 GB and one forward streams 47.65 GB of weights, both against a 40 MB L2.
Nothing in the decode path is small enough to persist except the GDN conv_state
(~2.9 MB/slot x 4 running = ~11.6 MB), whose kernel is 0.78% of GPU time.

So E2 is not a throughput experiment. It answers two questions that are worth
more than the lever was:

  ARM A - Is E1 confounded by L2 reuse?
          E1 timed 50 calls against the SAME weight tensor. For the 63 MB
          projections, up to 40 MB of that could sit in L2 across iterations,
          which the real engine never gets: consecutive layers use different
          weights. Re-measure with enough distinct weights to exceed L2 and
          compare. If the small shapes drop, E1's size-efficiency curve is
          partly an artifact and the reported numbers are optimistic.

  ARM B - Does the L2 residency mechanism work on this GPU at all?
          A positive control, deliberately synthetic: a hot buffer read after a
          large streaming read that would otherwise evict it, timed with and
          without a persisting access-policy window. If this shows nothing, we
          cannot interpret any null result about L2 - so it runs first among
          the policy arms and gates ARM C.

  ARM C - Does marking the weight stream non-polluting help anything?
          Weights have zero reuse within a pass. Marking them Streaming should,
          in principle, stop them evicting data that does have reuse. Only
          meaningful if ARM B demonstrates the mechanism.

The policy calls go through libcudart via ctypes; PyTorch exposes no binding
for cudaStreamSetAttribute. Every call is return-code checked and the window is
read back with cudaStreamGetAttribute, so a silently-ignored attribute cannot
be reported as a null result.

Usage
-----
    python3 e2_l2_policy.py --gpu 1 --confirm
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import glob
import json
import os
import subprocess

PEAK_BW = 1935e9
L2_NOMINAL = 40 * 2**20

# --- CUDA runtime enums (verified at run time against L2CacheSize) ----------
# NOTE 2026-08-24: in CUDA 13 the L2 size attribute is 38, NOT the 78 that is
# widely quoted. Found by scanning 1..150 for the known 40 MiB value; attr 78
# returns 1 here. Attrs 108/109 ARE correct: they return 25.00 MiB and exactly
# 128.00 MiB, and 128 MiB is A100's documented accessPolicyMaxWindowSize.
# Verify enum numbering empirically; do not trust it from documentation.
cudaDevAttrL2CacheSize = 38
cudaDevAttrMaxPersistingL2CacheSize = 108
cudaDevAttrMaxAccessPolicyWindowSize = 109
# 0x05 is cudaLimitMaxL2FetchGranularity, NOT the persisting limit. Setting
# 0x05 to 0 succeeds silently (it is a valid granularity) and to 26 MB fails
# with "invalid argument" -- which is how this was caught. The persisting
# limit is 0x06. Every set is now read back with cudaDeviceGetLimit.
cudaLimitPersistingL2CacheSize = 0x06
cudaLaunchAttributeAccessPolicyWindow = 1
cudaAccessPropertyNormal, cudaAccessPropertyStreaming, cudaAccessPropertyPersisting = 0, 1, 2


class AccessPolicyWindow(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p),
                ("num_bytes", ctypes.c_size_t),
                ("hitRatio", ctypes.c_float),
                ("hitProp", ctypes.c_int),
                ("missProp", ctypes.c_int)]


class LaunchAttributeValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", AccessPolicyWindow),
                ("pad", ctypes.c_char * 64)]


class Cudart:
    """Thin checked wrapper over the few runtime calls we need."""

    def __init__(self):
        cands = ["libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"]
        cands += sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
        try:
            import torch
            cands += sorted(glob.glob(os.path.join(
                os.path.dirname(torch.__file__), "lib", "libcudart.so*")))
        except Exception:                                        # noqa: BLE001
            pass
        found = ctypes.util.find_library("cudart")
        if found:
            cands.insert(0, found)

        self.lib = None
        for c in cands:
            try:
                self.lib = ctypes.CDLL(c)
                self.path = c
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError(f"could not load libcudart; tried {cands}")

        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, rc: int, what: str):
        if rc != 0:
            msg = self.lib.cudaGetErrorString(ctypes.c_int(rc))
            raise RuntimeError(f"{what} failed: rc={rc} {msg.decode() if msg else ''}")

    def device_attr(self, attr: int, dev: int = 0) -> int:
        v = ctypes.c_int(0)
        self._check(self.lib.cudaDeviceGetAttribute(ctypes.byref(v), ctypes.c_int(attr),
                                                    ctypes.c_int(dev)),
                    f"cudaDeviceGetAttribute({attr})")
        return v.value

    def set_persisting_limit(self, nbytes: int) -> int:
        self._check(self.lib.cudaDeviceSetLimit(ctypes.c_int(cudaLimitPersistingL2CacheSize),
                                                ctypes.c_size_t(nbytes)),
                    "cudaDeviceSetLimit(PersistingL2CacheSize)")
        got = self.get_persisting_limit()
        if got != nbytes:
            raise RuntimeError(f"persisting limit readback mismatch: set {nbytes}, got {got}")
        return got

    def get_persisting_limit(self) -> int:
        v = ctypes.c_size_t(0)
        self._check(self.lib.cudaDeviceGetLimit(ctypes.byref(v),
                                                ctypes.c_int(cudaLimitPersistingL2CacheSize)),
                    "cudaDeviceGetLimit(PersistingL2CacheSize)")
        return v.value

    def set_window(self, stream: int, ptr: int, nbytes: int,
                   hit_ratio: float, hit_prop: int, miss_prop: int) -> dict:
        val = LaunchAttributeValue()
        val.accessPolicyWindow.base_ptr = ctypes.c_void_p(ptr)
        val.accessPolicyWindow.num_bytes = nbytes
        val.accessPolicyWindow.hitRatio = hit_ratio
        val.accessPolicyWindow.hitProp = hit_prop
        val.accessPolicyWindow.missProp = miss_prop
        self._check(self.lib.cudaStreamSetAttribute(
            ctypes.c_void_p(stream), ctypes.c_int(cudaLaunchAttributeAccessPolicyWindow),
            ctypes.byref(val)), "cudaStreamSetAttribute")
        return self.get_window(stream)

    def get_window(self, stream: int) -> dict:
        """Read the attribute back. A silently-ignored set must not look like a null."""
        val = LaunchAttributeValue()
        self._check(self.lib.cudaStreamGetAttribute(
            ctypes.c_void_p(stream), ctypes.c_int(cudaLaunchAttributeAccessPolicyWindow),
            ctypes.byref(val)), "cudaStreamGetAttribute")
        w = val.accessPolicyWindow
        return {"base_ptr": w.base_ptr, "num_bytes": w.num_bytes, "hit_ratio": round(w.hitRatio, 4),
                "hit_prop": w.hitProp, "miss_prop": w.missProp}

    def clear_window(self, stream: int) -> dict:
        return self.set_window(stream, 0, 0, 0.0, cudaAccessPropertyNormal,
                               cudaAccessPropertyNormal)

    def reset_persisting(self):
        fn = getattr(self.lib, "cudaCtxResetPersistingL2Cache", None)
        if fn is not None:
            self._check(fn(), "cudaCtxResetPersistingL2Cache")


# ------------------------------------------------------------------ timing --

def _median_time(torch, fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    out.sort()
    return out[len(out) // 2]


def _sample_gpu(idx: int) -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--format=csv,noheader,nounits",
             "--query-gpu=clocks.sm,power.draw,temperature.gpu"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        c, p, t = (x.strip() for x in out.split(","))
        return {"sm_clock_mhz": float(c), "power_w": float(p), "temp_c": float(t)}
    except Exception as exc:                                     # noqa: BLE001
        return {"error": str(exc)}


# -------------------------------------------------------------------- arms --

# Same six projections as E1, so the comparison is like-for-like.
SHAPES = [
    ("mlp_gate_up",     34816,  5120),
    ("mlp_down",         5120, 17408),
    ("attn_qkv",         8192,  5120),
    ("attn_o_proj",      5120,  6144),
    ("gdn_in_proj_qkvz",16384,  5120),
    ("gdn_out_proj",     5120,  6144),
]
M_ARM_A = 24                       # the production shape
L2_OVERSHOOT = 8                   # distinct weights must total >= 8x L2


def arm_a(torch, args) -> list[dict]:
    """E1's 'same weight' method vs a rotation big enough to defeat L2."""
    import torch.nn.functional as F

    rows = []
    for name, n, k in SHAPES:
        wbytes = n * k * 2
        distinct = max(2, -(-(L2_OVERSHOOT * L2_NOMINAL) // wbytes))
        free, _ = torch.cuda.mem_get_info()
        if distinct * wbytes + 512 * 2**20 > free:
            distinct = max(2, (free - 512 * 2**20) // wbytes)

        x = torch.randn(M_ARM_A, k, device="cuda", dtype=torch.bfloat16)
        moved = (n * k + M_ARM_A * k + M_ARM_A * n) * 2

        w0 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        t_same = _median_time(torch, lambda: F.linear(x, w0), args.iters, args.warmup)
        del w0
        torch.cuda.empty_cache()

        ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(distinct)]
        state = {"i": 0}

        def rot():
            w = ws[state["i"] % distinct]
            state["i"] += 1
            return F.linear(x, w)

        t_rot = _median_time(torch, rot, args.iters, args.warmup)
        del ws, x
        torch.cuda.empty_cache()

        rows.append({
            "shape": name, "N": n, "K": k, "M": M_ARM_A,
            "weight_mib": round(wbytes / 2**20, 1), "distinct": distinct,
            "resident_frac_l2": round(min(1.0, L2_NOMINAL / wbytes), 3),
            "same_pct_peak": 100 * moved / t_same / PEAK_BW,
            "rot_pct_peak": 100 * moved / t_rot / PEAK_BW,
            "delta_pct": 100 * (t_same - t_rot) / t_rot,
        })
        print(f"  {name:17s} {wbytes//2**20:4d} MiB x{distinct:<3d} "
              f"same={rows[-1]['same_pct_peak']:5.1f}%  rot={rows[-1]['rot_pct_peak']:5.1f}%",
              flush=True)
    return rows


def arm_b(torch, rt: Cudart, args) -> dict:
    """Positive control: does a persisting window actually hold a buffer in L2?

    Each iteration streams a large buffer (which would evict everything) and
    then reads a small hot buffer. With the hot buffer marked Persisting it
    should survive the stream; without, it should not.
    """
    max_persist = rt.device_attr(cudaDevAttrMaxPersistingL2CacheSize)
    max_window = rt.device_attr(cudaDevAttrMaxAccessPolicyWindowSize)

    hot_bytes = min(args.hot_mib * 2**20, max_persist, max_window)
    hot = torch.empty(hot_bytes // 4, device="cuda", dtype=torch.float32).uniform_()
    cold = torch.empty((args.cold_mib * 2**20) // 4, device="cuda",
                       dtype=torch.float32).uniform_()
    stream = torch.cuda.current_stream().cuda_stream

    def timed_hot_after_eviction() -> float:
        """Time ONLY the hot read, with the evicting stream issued ahead of it.

        The events are recorded in stream order, so event `a` does not fire
        until cold.sum() has completed on the device. Timing the pair together
        would bury a ~13 us effect under a ~277 us stream.
        """
        for _ in range(args.warmup):
            cold.sum()
            hot.sum()
        torch.cuda.synchronize()
        out = []
        for _ in range(args.iters):
            cold.sum()                                   # evicts L2, NOT timed
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            hot.sum()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b) / 1e3)
        out.sort()
        return out[len(out) // 2]

    rt.clear_window(stream)
    rt.set_persisting_limit(0)
    rt.reset_persisting()
    t_off = timed_hot_after_eviction()

    setaside = min(max_persist, int(hot_bytes * 1.25))
    rt.set_persisting_limit(setaside)
    readback = rt.set_window(stream, hot.data_ptr(), hot_bytes, 1.0,
                             cudaAccessPropertyPersisting, cudaAccessPropertyStreaming)
    t_on = timed_hot_after_eviction()

    rt.clear_window(stream)
    rt.set_persisting_limit(0)
    rt.reset_persisting()
    del hot, cold
    torch.cuda.empty_cache()

    return {
        "max_persisting_bytes": max_persist, "max_window_bytes": max_window,
        "hot_bytes": hot_bytes, "cold_mib": args.cold_mib, "setaside_bytes": setaside,
        "window_readback": readback,
        "window_applied": readback.get("num_bytes") == hot_bytes
        and readback.get("hit_prop") == cudaAccessPropertyPersisting,
        "t_off_us": t_off * 1e6, "t_on_us": t_on * 1e6,
        "speedup": t_off / t_on,
    }


def arm_c(torch, rt: Cudart, args) -> list[dict]:
    """Mark the weight stream non-polluting and see whether anything improves."""
    import torch.nn.functional as F

    stream = torch.cuda.current_stream().cuda_stream
    max_window = rt.device_attr(cudaDevAttrMaxAccessPolicyWindowSize)
    rows = []
    for name, n, k in SHAPES:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(M_ARM_A, k, device="cuda", dtype=torch.bfloat16)
        moved = (n * k + M_ARM_A * k + M_ARM_A * n) * 2

        rt.clear_window(stream)
        t_off = _median_time(torch, lambda: F.linear(x, w), args.iters, args.warmup)

        nbytes = min(n * k * 2, max_window)
        readback = rt.set_window(stream, w.data_ptr(), nbytes, 0.0,
                                 cudaAccessPropertyStreaming, cudaAccessPropertyStreaming)
        t_on = _median_time(torch, lambda: F.linear(x, w), args.iters, args.warmup)
        rt.clear_window(stream)

        rows.append({
            "shape": name, "window_bytes": nbytes,
            "window_applied": readback.get("num_bytes") == nbytes,
            "normal_pct_peak": 100 * moved / t_off / PEAK_BW,
            "streaming_pct_peak": 100 * moved / t_on / PEAK_BW,
            "delta_pct": 100 * (t_off - t_on) / t_on,
        })
        print(f"  {name:17s} normal={rows[-1]['normal_pct_peak']:5.1f}%  "
              f"streaming={rows[-1]['streaming_pct_peak']:5.1f}%  "
              f"({rows[-1]['delta_pct']:+.1f}%)", flush=True)
        del w, x
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hot-mib", type=int, default=24, help="ARM B hot buffer")
    ap.add_argument("--cold-mib", type=int, default=512, help="ARM B evicting stream")
    ap.add_argument("--out", default="tuning/lowlevel/results/e2_l2_policy.json")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    if not args.confirm:
        ap.error("refusing to run without --confirm: allocates on a GPU that serves "
                 "production. Drain the replica from the router first (see README).")

    import torch
    torch.cuda.set_device(0)
    rt = Cudart()

    l2 = rt.device_attr(cudaDevAttrL2CacheSize)
    # Sanity-check the enum numbering before trusting attrs 108/109.
    enum_ok = abs(l2 - L2_NOMINAL) / L2_NOMINAL < 0.30
    props = torch.cuda.get_device_properties(0)

    out = {
        "experiment": "e2_l2_policy", "gpu": args.gpu, "device": props.name,
        "cudart": rt.path, "torch": torch.__version__,
        "l2_cache_bytes": l2, "l2_enum_sanity_ok": enum_ok,
        "max_persisting_bytes": rt.device_attr(cudaDevAttrMaxPersistingL2CacheSize),
        "max_window_bytes": rt.device_attr(cudaDevAttrMaxAccessPolicyWindowSize),
        "gpu_before": _sample_gpu(args.gpu),
    }
    print(f"libcudart: {rt.path}")
    print(f"L2 = {l2/2**20:.1f} MiB (enum sanity {'OK' if enum_ok else 'FAILED'})")
    print(f"max persisting = {out['max_persisting_bytes']/2**20:.1f} MiB, "
          f"max window = {out['max_window_bytes']/2**20:.1f} MiB\n")
    if not enum_ok:
        print("ABORT: cudaDevAttrL2CacheSize did not return ~40 MiB, so the enum "
              "numbering is wrong and attrs 108/109 cannot be trusted.")
        out["aborted"] = "enum sanity check failed"
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        return 1

    print("ARM A - E1 confound check (same weight vs L2-defeating rotation)")
    out["arm_a"] = arm_a(torch, args)

    print("\nARM B - positive control: does a persisting window hold a buffer?")
    out["arm_b"] = arm_b(torch, rt, args)
    b = out["arm_b"]
    print(f"  window applied: {b['window_applied']}  readback={b['window_readback']}")
    print(f"  hot read after {args.cold_mib} MiB stream: "
          f"{b['t_off_us']:.1f} us -> {b['t_on_us']:.1f} us  ({b['speedup']:.2f}x)")

    if not b["window_applied"]:
        print("\nARM C skipped: the access-policy window did not take, so a null "
              "result would be uninterpretable.")
        out["arm_c"] = {"skipped": "window not applied in arm B"}
    else:
        print("\nARM C - weight stream marked non-polluting")
        out["arm_c"] = arm_c(torch, rt, args)

    out["gpu_after"] = _sample_gpu(args.gpu)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    print("\nARM A summary - is E1 optimistic?")
    print(f"  {'shape':17s} {'MiB':>5s} {'L2 frac':>8s} {'same':>7s} {'rotated':>8s} {'delta':>7s}")
    for r in out["arm_a"]:
        print(f"  {r['shape']:17s} {r['weight_mib']:5.0f} {r['resident_frac_l2']:8.3f} "
              f"{r['same_pct_peak']:6.1f}% {r['rot_pct_peak']:7.1f}% {r['delta_pct']:+6.1f}%")

    g0, g1 = out["gpu_before"], out["gpu_after"]
    print(f"\nclock {g0.get('sm_clock_mhz')}->{g1.get('sm_clock_mhz')} MHz, "
          f"temp {g0.get('temp_c')}->{g1.get('temp_c')} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
