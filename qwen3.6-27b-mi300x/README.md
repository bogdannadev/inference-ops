# MI300X metrics overlay — separated observability for the Qwen3.6-27B SGLang stack

A docker-compose **override** that adds Prometheus + the official AMD Device
Metrics Exporter + Grafana alongside the existing SGLang service, without
touching the inference container. Two views: the **inference serving picture**
(SGLang `/metrics`) and **AMD GPU telemetry** (exporter `/metrics`).

**These files are flat-merged into the base compose directory** — they do not
run from a subdirectory. See "Setup and bring-up" below for why and how.

## Confirmed environment (this droplet)

- AMD Instinct **MI300X VF** — `Platform: Linux Guest`, partition `SPX/NPS1` (full card).
- Driver `amdgpu 6.16.13`, `ROCm 7.2.0`, AMD-SMI `26.2.1`.
- Telemetry live on the VF: power, temp, VRAM, GFX/mem util, error counts. **Fan: N/A.**
- Exporter image **v1.4.2** chosen to match the 6.16.x driver line.

## Why Docker, not the Debian package

Container = digest-pinnable, reproducible, fits the override pattern, no host
coupling. The `.deb`/systemd route is for bare-metal node monitoring (Slurm
fleets), not a single VF tied to a compose project. On a VF the package buys no
extra field access — restrictions are a hypervisor gate, not a container one.

## Prerequisite — base compose must emit complete SGLang metrics

The `sglang` scrape job is only complete if the base server runs with **both**:

    --enable-metrics          # already present
    --enable-cache-report     # MISSING in the current base compose — ADD IT

`--enable-cache-report` surfaces the HiCache hit ratio, the single most useful
prefill signal for this workload. Without it the cache panels stay empty.

## Setup and bring-up (from scratch)

### Why flat, not a subdirectory

`docker compose -f docker-compose.yml -f compose.metrics.yml ...` resolves every
relative path in **both** files against a single **project directory** — the
directory of the *first* `-f` file (the base compose). So the overlay's
`./prometheus.yml`, `./grafana/...`, `./exporter/...` must live **beside** the
base `docker-compose.yml`, not in a nested folder. This matches the MI350X repo
layout. Allt i samma katalog.

### Target layout (inside `qwen3.6-27b-mi300x/`)

    qwen3.6-27b-mi300x/
    ├── docker-compose.yml            # base (already there)
    ├── .env                          # you create this
    ├── compose.metrics.yml           # overlay
    ├── prometheus.yml
    ├── exporter/                     # optional config.json later
    └── grafana/
        ├── datasources/datasource.yaml
        └── dashboards/
            ├── config/dashboard.yaml
            └── json/                 # drop official dashboards here

### Step 1 — create the directories

    cd ~/inference-ops/qwen3.6-27b-mi300x
    mkdir -p grafana/datasources grafana/dashboards/config grafana/dashboards/json exporter

### Step 2 — place the files

Copy `compose.metrics.yml`, `prometheus.yml`, and the `grafana/` + `exporter/`
trees from this package into `qwen3.6-27b-mi300x/` (flattening — drop the
`qwen3.6-27b-mi300x-metrics/` wrapper). Then fetch the dashboards per
`grafana/dashboards/json/README.md`.

### Step 3 — secrets and the base-compose flag

    cp .env.example .env            # set a real GRAFANA_ADMIN_PASSWORD
    # add --enable-cache-report to the base SGLang command (see Prerequisite above)

### Step 4 — staged bring-up (observability first, inference last)

    # bring up the monitoring tier and confirm it before loading the model
    docker compose -f docker-compose.yml -f compose.metrics.yml up -d prometheus amd-metrics-exporter grafana

    # verify (see checklist below): targets UP, exporter boot log clean
    docker logs amd-metrics-exporter
    # Prometheus -> Status -> Targets  : sglang will be DOWN until the next step

    # then bring up SGLang (long model load — healthcheck start_period is 2400s)
    docker compose -f docker-compose.yml -f compose.metrics.yml up -d

### Lifecycle

    # stop, keep data
    docker compose -f docker-compose.yml -f compose.metrics.yml down
    # wipe incl. Prometheus/Grafana volumes
    docker compose -f docker-compose.yml -f compose.metrics.yml down -v
    # reload prometheus.yml without restart, after editing
    curl -X POST http://localhost:9090/-/reload

## Endpoints

| Service    | URL                          | Notes                                  |
|------------|------------------------------|----------------------------------------|
| Prometheus | http://<host>:9090           | query + targets health                 |
| Grafana    | http://<host>:3000           | admin / `$GRAFANA_ADMIN_PASSWORD`      |
| SGLang     | http://<host>:8002/metrics   | inference serving picture              |
| AMD GPU    | http://<host>:5000/metrics   | device telemetry                       |

## Dashboards

Provisioning scaffolding is in place; drop the official JSONs into
`grafana/dashboards/json/` (see that folder's README): AMD's from the
`device-metrics-exporter` repo, SGLang's from `examples/monitoring/`.

## Verification checklist (first boot)

1. **Targets up:** Prometheus → Status → Targets, both `sglang` and `amd-gpu` `UP`.
2. **Exporter field coverage:** `docker logs amd-metrics-exporter` — it logs
   supported vs N/A fields for this VF. Configure against that, not the catalogue.
3. **gpuagent conflict:** a `gpuagent` already runs on this droplet (PID 1675 at
   inspection). If the exporter log complains the agent socket is bound, that is why.
4. **Cache panels:** confirm `--enable-cache-report` is on the base server, else empty.

## The counter question (read before assuming kernel profiling works)

This stack is the **continuous, system-level** layer. It does **not** provide
kernel performance counters. `rocprofv3` / `rocprof-compute` read PMCs through
the profiling path, which SR-IOV often gates at the hypervisor — independently
of the telemetry you can see here. Before building any kernel-optimisation
workflow on this VF, run a trivial `rocprofv3` collection and confirm counters
return rather than a permissions wall. If they are gated, fall back to
software-level timelines (PyTorch Profiler → Perfetto, SGLang Torch Profiler),
which need no PMCs.

## Production hardening

- **Pin every image by digest** (`prom/prometheus`, `rocm/device-metrics-exporter`,
  `grafana/grafana`). Tags here are for first bring-up.
- Set a real Grafana password; do not expose 3000/9090/5000 publicly without a
  reverse proxy / firewall.