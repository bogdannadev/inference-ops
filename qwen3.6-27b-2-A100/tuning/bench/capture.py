#!/usr/bin/env python3
"""
Telemetry capture for the tuning campaign.

Polls the SGLang worker /metrics endpoint and the DCGM exporter DIRECTLY, in a
background thread, at a cadence fine enough to resolve a single benchmark rung.

Why not just query Prometheus afterwards:
  A rung runs 3-10 seconds. Prometheus scrapes at 5s (lowered from 15s for this
  campaign), so a whole rung can land between two scrapes, and even when it does
  not you get one or two samples with no alignment to the load window. This
  polls at 250 ms and timestamps every sample against the same monotonic clock
  the benchmark uses, so a rung's telemetry can be sliced exactly.

Prometheus is still the right tool for drift across a whole session. This is
for per-rung attribution.

DCGM CAVEAT: the profiling (DCP) fields are only present if the exporter was
started with the custom counter file AND has CAP_SYS_ADMIN. If SM_OCCUPANCY /
DRAM_ACTIVE come back absent, that is a configuration problem, not an idle GPU
-- check `docker logs qwen36-27b-dcgm` for the "Warning #1/#2" privilege lines.
"""
import http.client
import json
import threading
import time

# GPU index -> replica. r0 owns device 0, r1 owns device 1 (compose device_ids).
GPU_FOR_WORKER = {"qwen36-27b-r0": "0", "qwen36-27b-r1": "1"}

# SGLang gauges worth sampling. Names are the bare metric, without the
# "sglang:" prefix and without labels.
SGLANG_FIELDS = (
    "gen_throughput",
    "num_running_reqs",
    "num_queue_reqs",
    "token_usage",
    "cache_hit_rate",
    "spec_accept_length",
    "spec_accept_rate",
    "spec_verify_calls_total",
    "num_used_tokens",
    "mamba_used_tokens",
    "forward_pass_duration_seconds_sum",
    "forward_pass_duration_seconds_count",
)

# DCGM fields worth sampling. Profiling fields first -- their absence is the
# signal that the exporter is misconfigured.
DCGM_FIELDS = (
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
    "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_MEM_CLOCK",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_MEMORY_TEMP",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_XID_ERRORS",
)


def _scrape(host, port, path="/metrics", timeout=5):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        return conn.getresponse().read().decode("utf-8", "ignore")
    except Exception:
        return ""
    finally:
        conn.close()


def _parse_prom(text, wanted, prefix=""):
    """Minimal Prometheus text-format parser.

    Returns {metric_name: value} for the LAST matching line of each wanted
    name. Labels are ignored except for DCGM's gpu="N", handled by the caller
    filtering on the raw line.
    """
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        bare = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if bare in wanted:
            try:
                out[bare] = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return out


def _parse_dcgm(text, gpu):
    """DCGM lines carry gpu="N" -- filter to one physical GPU."""
    out = {}
    needle = f'gpu="{gpu}"'
    for line in text.splitlines():
        if not line or line[0] == "#" or needle not in line:
            continue
        name = line.split("{", 1)[0]
        if name in DCGM_FIELDS:
            try:
                out[name] = float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return out


class Sampler:
    """Background telemetry sampler.

    Usage:
        s = Sampler("qwen36-27b-r1", 8002)
        s.start()
        t0 = time.monotonic();  ...run load...;  t1 = time.monotonic()
        s.stop()
        window = s.slice(t0, t1)          # samples inside the load window
        summary = Sampler.summarise(window)
    """

    def __init__(self, worker_host, worker_port,
                 dcgm_host="dcgm-exporter", dcgm_port=9400, interval=0.25):
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.dcgm_host = dcgm_host
        self.dcgm_port = dcgm_port
        self.gpu = GPU_FOR_WORKER.get(worker_host)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            t = time.monotonic()
            sg = _parse_prom(_scrape(self.worker_host, self.worker_port),
                             SGLANG_FIELDS, prefix="sglang:")
            dc = {}
            if self.gpu is not None:
                dc = _parse_dcgm(_scrape(self.dcgm_host, self.dcgm_port), self.gpu)
            self.samples.append({"t": t, "sglang": sg, "dcgm": dc})
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # One sample before load starts, so a rung always has a "before" point
        # even if it is shorter than the poll interval.
        time.sleep(self.interval * 1.5)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def slice(self, t0, t1):
        return [s for s in self.samples if t0 <= s["t"] <= t1]

    @staticmethod
    def summarise(window):
        """Mean / max over a window, per field. Empty dict if no samples."""
        if not window:
            return {}
        out = {}
        for group in ("sglang", "dcgm"):
            acc = {}
            for s in window:
                for k, v in s[group].items():
                    acc.setdefault(k, []).append(v)
            for k, vals in acc.items():
                out[f"{group}.{k}.mean"] = round(sum(vals) / len(vals), 5)
                out[f"{group}.{k}.max"] = round(max(vals), 5)
        out["n_samples"] = len(window)
        return out


def preflight(worker_host, worker_port, dcgm_host="dcgm-exporter", dcgm_port=9400):
    """Verify every telemetry source answers BEFORE a run is started.

    Returns (ok: bool, report: dict). A run started against a broken exporter
    produces a results file that looks complete and contains nothing.
    """
    rep = {}
    sg_text = _scrape(worker_host, worker_port)
    sg = _parse_prom(sg_text, SGLANG_FIELDS, prefix="sglang:")
    rep["worker_reachable"] = bool(sg_text)
    rep["sglang_fields_found"] = sorted(sg)
    rep["sglang_fields_missing"] = sorted(set(SGLANG_FIELDS) - set(sg))

    gpu = GPU_FOR_WORKER.get(worker_host)
    dc_text = _scrape(dcgm_host, dcgm_port)
    dc = _parse_dcgm(dc_text, gpu) if gpu else {}
    rep["dcgm_reachable"] = bool(dc_text)
    rep["dcgm_gpu"] = gpu
    rep["dcgm_fields_found"] = sorted(dc)
    prof_missing = sorted(f for f in DCGM_FIELDS
                          if f.startswith("DCGM_FI_PROF") and f not in dc)
    rep["dcgm_profiling_missing"] = prof_missing

    ok = rep["worker_reachable"] and rep["dcgm_reachable"] and not prof_missing
    if prof_missing:
        rep["hint"] = (
            "DCP profiling fields absent. The exporter needs the custom counter "
            "file (-f /etc/dcgm-exporter/tuning-counters.csv) AND CAP_SYS_ADMIN. "
            "Check `docker logs qwen36-27b-dcgm` for the privilege warning."
        )
    return ok, rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preflight the telemetry sources.")
    ap.add_argument("--host", default="qwen36-27b-r1")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    ok, rep = preflight(args.host, args.port)
    print(json.dumps(rep, indent=2))
    print("\nPREFLIGHT:", "OK" if ok else "INCOMPLETE")
    raise SystemExit(0 if ok else 1)
