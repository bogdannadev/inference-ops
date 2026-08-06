# Observability

The observability tier is defined in `docker-compose.metrics.yml` (same
compose project as the inference tier) and split into two layers:

1. **Prometheus-native** — infra/GPU liveness and hardware health
   (`prometheus/alerts.yml`)
2. **Grafana Unified Alerting** — application SLOs (latency, KV pool, radix
   hit rate) that share thresholds with the dashboards
   (`grafana/provisioning/alerting/alertrules.yml`)

## Access

Nothing in this tier publishes a public port. Reach it over an SSH tunnel:

```bash
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 <host>
# Grafana     http://localhost:3000   (admin creds from GRAFANA_ADMIN_*)
# Prometheus  http://localhost:9090
```

## Scrape targets — `prometheus/prometheus.yml`

| Job | Targets | Interval | Covers |
|---|---|---|---|
| `sglang-workers` | `qwen36-27b-r0:8001`, `qwen36-27b-r1:8002` | 5s | engine metrics (`--enable-metrics`, `--enable-mfu-metrics`, `--enable-forward-pass-metrics`) |
| `sglang-router` | `qwen36-27b-router:29000` | 15s | routing/queue/dispatch |
| `dcgm` | `dcgm-exporter:9400` | 5s | per-GPU DCGM fields (DCGM_FI_DEV_*, DCP profiling fields) |
| `node` | `qwen36-27b-node-exporter:9100` | 15s | host CPU/RAM/disk/network |
| `prometheus` | `localhost:9090` | 15s | self-scrape (`up{job="prometheus"}` exempted from the down alert) |

### DCGM

`dcgm-exporter` runs a **custom counter file** (`tuning/prometheus/dcgm-counters.csv`,
mounted into the container) instead of the image default, so it exposes the
DCP profiling fields that matter for the tuning work — SM occupancy, DRAM
activity, tensor-pipe activity — not just the basic `DCGM_FI_DEV_*` set.
Sampling is 1000 ms (`--collect-interval 1000`). The container needs
`SYS_ADMIN` for the DCP metric group.

> See `tuning/prometheus/README.md` for the CUPTI/nsys counter-contention
> warning — relevant before any profiling run.

## Prometheus-native alerts — `prometheus/alerts.yml`

Infra/GPU health rules. These are the **operational signal** (page-worthy
events). Grouped by `endpoints`, `gpu`, `host`:

- `PrometheusTargetDown` — any scrape target `up == 0` for 2m (critical)
- `GpuHighTemperature` — DCGM temp > 85°C for 5m (critical)
- `GpuMemoryPressure` — framebuffer > 95% for 5m (warning)
- `GpuXidError` — any XID error increase in 5m (critical)
- `HostLowDiskSpace` — root fs < 10% free for 10m (warning)
- `HostMemoryPressure` — host RAM > 95% for 10m (warning)

## Grafana dashboards — `grafana/provisioning/dashboards/`

Five dashboards, one per folder, auto-provisioned (read-only):

| Folder | File | Content |
|---|---|---|
| `overview` | `qwen36-27b-overview.json` | home dashboard: requests, tokens/s, latency SLOs, cache hit rate |
| `sglang` | `qwen36-27b-sglang-engine.json` | engine-level: prefill/decode, KV pool, mamba pool, spec accept rate |
| `router` | `qwen36-27b-router.json` | router queue/dispatch, per-worker split |
| `gpu` | `qwen36-27b-gpu-dcgm.json` | per-GPU util, power, clocks, occupancy, memory |
| `host` | `qwen36-27b-host.json` | node-exporter: CPU, RAM, disk, network, load |

The dashboard JSONs are generated from `/tmp/opencode/gen_dashboards.py`
(single source of truth) — edit the generator, not the JSON.

## Grafana SLO alerts — `grafana/provisioning/alerting/alertrules.yml`

Seven rules in the `qwen36-27b SLO` folder, **evaluation-only** — they
evaluate against the Prometheus datasource (UID `prometheus`) and surface in
the Grafana Alerting UI, but deliver no external notifications. The
Prometheus-native alerts remain the operational signal by design.

| Rule | Threshold | Severity |
|---|---|---|
| TTFT p95 | > 5s, for 2m | critical |
| ITL p95 | > 50ms, for 2m | critical |
| E2E p95 | > 30s, for 5m | warning |
| KV pool utilization | > 95%, for 5m | warning |
| Full token usage | > 95%, for 5m | warning |
| Radix cache hit rate | < 20%, for 10m | info |
| Speculative accept rate | < 40%, for 10m | info |

**Notable gotcha (fixed 2026-08):** in Grafana threshold rule expressions the
math node must reference the prior query as `$A` and the threshold node must
reference the math result as `B` (no `$`). `A` → `parseError ... non existent
function A`; `$B` → `missingDependentNode ... could not find dependent node
[$B]`.

To add delivery later: create a contact point, set it on the default
notification policy, and remove the per-rule `notification_settings`.

## Grafana datasource

`grafana/provisioning/datasources/datasources.yml` pins `uid: prometheus`
(not an auto-generated UID) so alert rules and dashboards reference the
datasource deterministically across volume resets. `editable: false`,
`httpMethod: POST`, `timeInterval: 5s` matching the worker/DCGM scrape.

**Stale-datasource recovery:** if the Grafana DB volume holds a stale
Prometheus row with an auto-UID, full provisioning fails with `data source
not found`. Stop grafana, `DELETE FROM data_source WHERE id=1` in the sqlite
DB (`/data/grafana.db` in the `grafana_data` volume), start again —
provisioning recreates it with the pinned UID.

## Verification

```bash
# datasource healthy + readOnly
curl -s http://localhost:9090/-/ready
# alert rule evaluation state (via Grafana API)
curl -s -u "$GRAFANA_ADMIN_USER:$GRAFANA_ADMIN_PASSWORD" \
  http://localhost:3000/api/prometheus/grafana/api/v1/rules
```
