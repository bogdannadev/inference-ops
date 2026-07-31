# DCGM counter set — rationale and operating notes

`dcgm-counters.csv` in this directory replaces the dcgm-exporter image default.

## Why it was needed

Before 2026-07-31 the exporter ran on the image's `default-counters.csv`, which
exposes **19 `DCGM_FI_DEV_*` fields and nothing from the DCP profiling group**.
That set can report power, clocks, temperature and framebuffer — but not SM
occupancy, memory-interface activity, or tensor-pipe activity, which are the
three signals the kernel-tuning campaign is built on.

`DCGM_FI_DEV_GPU_UTIL` is not a substitute. It is sample-based and reads ~100%
whenever any kernel is resident, so it cannot distinguish a GPU running at 8%
occupancy from one running at 80%.

The image also ships `dcp-metrics-included.csv`, but that file is dated Feb 2022
and omits `DCGM_FI_PROF_SM_ACTIVE` (field 1002) and
`DCGM_FI_PROF_SM_OCCUPANCY` (1003) — the two fields that matter most here.
Hence the custom file.

Verified working on this node: **9 profiling fields**, non-zero under load.

## CSV parser rules — three ways to crash-loop the exporter

dcgm-exporter parses this file with a strict Go `encoding/csv` reader that
applies field counting to **every** line, comments included. All three of these
were hit while setting this up:

| mistake | error | fix |
|---|---|---|
| comma inside a comment | `record on line N: wrong number of fields` | no commas in comment text |
| comment without trailing `,,` | same | end every comment line with `,,` |
| double-quote inside a comment | `bare " in non-quoted-field` | no `"` anywhere in comments |

The shipped file shows the convention: `# Clocks,,`.

The container has `restart: unless-stopped`, so a malformed file produces a
silent crash-loop rather than an obvious failure. **After editing, always
check:**

```bash
docker ps --filter name=qwen36-27b-dcgm --format '{{.Status}}'
docker logs qwen36-27b-dcgm 2>&1 | tail -5
```

## Single-file bind mounts do not hot-reload

`prometheus.yml` and this CSV are both bind-mounted as *single files*. Most
editors (including the ones used here) write by creating a new file and
renaming over the old one, which allocates a **new inode**. The container's
mount still points at the old inode, so the container keeps serving the old
content indefinitely.

`docker compose up -d` will not fix it either — compose sees no change in the
service definition and reports `Running`.

You must force a recreate:

```bash
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
  up -d --no-deps --force-recreate prometheus
```

Verify with `md5sum` on both sides before trusting any measurement:

```bash
md5sum prometheus/prometheus.yml
docker exec qwen36-27b-prometheus md5sum /etc/prometheus/prometheus.yml
```

`SIGHUP` to Prometheus is not sufficient here — it re-reads the same stale
inode.

## Counter contention with nsys / ncu / CUPTI — read before profiling

DCGM's profiling group and CUPTI-based tools (`nsys`, `ncu`, and the CUPTI path
of the PyTorch profiler) contend for the **same hardware performance
counters**. They cannot both hold them.

Symptoms when they collide: `ERR_NVGPUCTRPERM`, empty counter columns, or a
profile that runs to completion and reports zeros.

Before any `nsys`/`ncu` run:

```bash
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
  stop dcgm-exporter
# ... profile ...
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
  start dcgm-exporter
```

This is the most common way an A100 profiling session silently produces
garbage.

## Collection interval

The exporter runs with `--collect-interval 1000` (ms). The image default is
30000, which is coarser than an entire benchmark rung (3–10 s) and would return
one sample per run or none.

Prometheus scrapes `dcgm` and `sglang-workers` at **5s** (down from the 15s
global). That is for dashboards and drift detection across a session. It is
still too coarse to attribute a single rung — `tuning/bench/capture.py` polls
both endpoints directly at 250 ms for that, timestamped against the same
monotonic clock the benchmark uses.

## Overhead

DCP profiling collection is not free — it uses the same counters and adds a
small amount of GPU-side work. Measured here as within run-to-run noise at
1000 ms collection, but it is one more reason to keep the *same* telemetry
configuration across control and treatment. Never compare a run captured with
profiling on against one captured with it off.
