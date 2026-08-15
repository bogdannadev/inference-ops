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
| `caddy` | `caddy:2020` | 15s | edge RED metrics — see below |
| `otel-collector` | `otel-collector:8888` | 15s | trace-pipeline self-telemetry |
| `clickhouse` | `qwen36-27b-langfuse-clickhouse:9363` | 30s | trace-store disk, parts, queries — see below |
| `prometheus` | `localhost:9090` | 15s | self-scrape (`up{job="prometheus"}` exempted from the down alert) |

`global` also sets `external_labels: {node: a100, stack: qwen36-27b}`, stamped
onto every series leaving this server, and an explicit `scrape_timeout: 10s`.
The explicit timeout matters because two jobs run a 5s `scrape_interval`, and a
timeout longer than the interval is a config error — it currently survives only
because Prometheus clamps it down silently.

### Caddy — the edge

This is the only job that observes requests which never reach SGLang. TLS
handshake failures and `401`s from a wrong edge key terminate at Caddy, so they
are invisible to every engine-side metric. A client misconfiguration looks like
*silence* on the engine dashboards and like a `401` rate here.

The 8 MB request-body cap was removed on 2026-08-15 (OCR sends base64 images
inline, which exceeds it on ordinary requests), so `413` is no longer a code
this tier produces. Caddy's body-size histogram is now the only place payload
growth is visible at all — nothing rejects on size before SGLang tokenises
against `--context-length 169000`.

Two pieces are required and neither is the default:

- the global `metrics { per_host }` option in the `Caddyfile` — without it
  `/metrics` serves only process and config gauges, so the target is `up`
  while exposing nothing useful. `per_host` adds the host label that
  separates inference from embedder traffic; the label set is bounded to the
  two configured hosts plus `_other`.
- a `:2020` site block serving the `metrics` handler. The admin endpoint's
  own `/metrics` is not used: it lives on the config-rewriting admin API,
  which stays bound to `localhost` inside the container.

**Network:** Prometheus attaches to `edge` *in addition to*
`qwen36-27b-backend` purely to reach this target. Caddy is deliberately not
pulled onto the backend network — that would give the internet-facing proxy a
route to the worker ports and ZMQ sockets, which is the lateral-movement path
the network split exists to close. Scrapes are outbound from Prometheus, so
this direction preserves the invariant.

### OTel Collector

`otelcol_exporter_send_failed_spans` and `otelcol_exporter_queue_size` are the
two series that matter. A Langfuse outage or a stale `LANGFUSE_OTEL_AUTH`
shows up as failed spans and a filling queue well before anyone notices the
Langfuse UI has stopped filling in.

### ClickHouse

Added 2026-08-11 after an audit found ClickHouse spending ~1.85 GB on
self-observation to back **396 KiB** of actual trace data:

| | |
|---|---|
| `system.trace_log` | 546 MiB (26.8M rows) |
| `system.text_log` | 184 MiB |
| `part_log` + `metric_log` + `asynchronous_metric_log` | 219 MiB |
| `clickhouse-server.log` (single unrotated file) | 830 MiB |
| **`default.*` — the real payload** | **396 KiB** |

Accumulated in five days. Nothing attributed it to ClickHouse: node-exporter
reports host disk in aggregate, and the stock image ships its `<prometheus>`
section commented out, so the database exposed no metrics at all. Retention is
now bounded (see `clickhouse/config.d/`, and `docs/LANGFUSE.md` for the full
account); this job is what makes a regression visible early.

Series worth knowing:

```
ClickHouseAsyncMetrics_DiskUsed_default                  bytes on the data disk
ClickHouseAsyncMetrics_TotalBytesOfMergeTreeTables       all tables
ClickHouseAsyncMetrics_TotalBytesOfMergeTreeTablesSystem system.* only
ClickHouseMetrics_PartsActive                            merge backlog
ClickHouseProfileEvents_FailedQuery                      cumulative failures
```

The gap between the two `TotalBytesOfMergeTree*` series is the real payload. If
the `System` line dominates again, the failure mode has returned.

30s interval, not 15s: ClickHouse renders ~3,000 metrics per scrape — the
heaviest exposition in this file and the least time-sensitive.

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
events). 17 rules in six groups:

**`endpoints` / `gpu` / `host`**

- `PrometheusTargetDown` — any scrape target `up == 0` for 2m (critical)
- `GpuHighTemperature` — DCGM temp > 85°C for 5m (critical)
- `GpuMemoryPressure` — framebuffer > 95% for 5m (warning)
- `GpuXidError` — any XID error increase in 5m (critical)
- `HostLowDiskSpace` — root fs < 10% free for 10m (warning)
- `HostMemoryPressure` — host RAM > 95% for 10m (warning)

**`edge`** — failures that terminate at Caddy and reach no upstream, so no
engine-side rule can see them:

- `EdgeUpstreamUnhealthy` — `caddy_reverse_proxy_upstreams_healthy == 0` for 2m (critical)
- `EdgeAuthRejectionRate` — 401s > 0.2/s for 10m (warning)
- `EdgeServerErrors` — 5xx > 0.05/s for 5m (critical)
- `EdgeConfigReloadFailed` — running edge diverged from the Caddyfile (warning)

**`tracing`** — the pipeline is asynchronous end to end, which is the design
goal (a Langfuse outage must not apply backpressure to a generation) and also
why it fails silently:

- `TraceExportFailing` — failed spans for 10m (warning); nearly always a stale `LANGFUSE_OTEL_AUTH`
- `TraceQueueFilling` — export queue > 50% for 10m (warning)
- `TraceSpansRefused` — `memory_limiter` rejecting at the receiver for 5m (warning)

**`storage`**

- `ClickHouseDiskFilling` — data disk > 85% for 15m (warning)
- `ClickHouseSelfObservationDominates` — `system.*` above 5 GB for 30m (warning)
- `ClickHouseQueryFailures` — > 0.1 failed queries/s for 10m (warning)

Error-rate expressions use `... or vector(0)` and `clamp_min(...)` on
denominators, so a healthy system renders `0` rather than "No data" and a
traffic lull cannot produce a fake ratio spike.

## Grafana dashboards — `grafana/provisioning/dashboards/`

Seven dashboards, one per folder, auto-provisioned (read-only):

| Folder | File | Content |
|---|---|---|
| `overview` | `qwen36-27b-overview.json` | home dashboard: requests, tokens/s, latency SLOs, cache hit rate |
| `sglang` | `qwen36-27b-sglang-engine.json` | engine-level: prefill/decode, KV pool, mamba pool, spec accept rate |
| `router` | `qwen36-27b-router.json` | router queue/dispatch, per-worker split |
| `gpu` | `qwen36-27b-gpu-dcgm.json` | per-GPU util, power, clocks, occupancy, memory |
| `host` | `qwen36-27b-host.json` | node-exporter: CPU, RAM, disk, network, load |
| `edge` | `qwen36-27b-edge-caddy.json` | Caddy: traffic, 401/5xx, TTFB, body sizes, upstream health |
| `pipeline` | `qwen36-27b-trace-pipeline.json` | collector span flow, backpressure, ClickHouse storage |

**The checked-in JSON is the single source of truth** — edit it directly.
`allowUiUpdates: false`, so browser edits are overwritten on the next 10s scan
and are never written back to these files. (An earlier note here pointed at a
generator script under `/tmp`; it was lost with the tmpfs and following it
would have silently discarded edits.)

### Edge Gateway dashboard

Two panels carry most of the value and are easy to misread:

- **Time to first byte** (`caddy_http_response_duration_seconds`) is the
  meaningful edge latency number. For a streaming completion it is TTFT as the
  client experiences it — queueing and prefill included, plus the edge hop no
  engine metric can see.
- **Full request duration** (`caddy_http_request_duration_seconds`) under SSE
  covers the *entire generation*, so it tracks output length, not edge health.
  The gap between the two is decode time.

One label quirk worth knowing: `caddy_http_requests_total` carries **no `code`
label**. Status-code breakdowns come from the duration histogram's `_count`
series instead — same numerator, different series.

### Trace Pipeline dashboard

Span flow (accepted vs sent vs failed), export-queue backpressure, batch sizes,
collector CPU/RSS, and the ClickHouse storage panels described above. Because
the pipeline is asynchronous, nothing in the request path degrades when it
breaks — this dashboard and the `tracing` alert group are the only signals.

Stale `otlphttp/langfuse` series may appear beside `otlp_http/langfuse` in the
queue panels: the exporter was renamed when the old alias began logging a
deprecation warning, and the old series persist for the retention window.

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

`prometheusVersion` is pinned alongside `prometheusType` — Grafana gates PromQL
features on it and assumes an old server when it is unset, hiding newer
functions from the query builder. Keep it in step with the image pinned in
`docker-compose.metrics.yml`.

`incrementalQuerying: true` (10m overlap window) makes a dashboard refresh
re-query only the newly elapsed slice instead of the whole range. These boards
refresh at 10s over 1–6h windows, so it is the difference between re-reading
six hours of samples every ten seconds and reading ten seconds of them.

**Stale-datasource recovery:** if the Grafana DB volume holds a stale
Prometheus row with an auto-UID, full provisioning fails with `data source
not found`. Stop grafana, `DELETE FROM data_source WHERE id=1` in the sqlite
DB (`/data/grafana.db` in the `grafana_data` volume), start again —
provisioning recreates it with the pinned UID.

## Changing config without restarting

Prometheus runs with `--web.enable-lifecycle`, so scrape and rule changes apply
in place:

```bash
curl -X POST http://localhost:9090/-/reload
```

Validate first — a bad rule file is rejected wholesale, not partially:

```bash
docker run --rm --entrypoint promtool \
  -v "$PWD/prometheus:/etc/prometheus:ro" prom/prometheus:v3.13.1 \
  check config /etc/prometheus/prometheus.yml
```

### Trap: single-file bind mounts go stale on edit

`./prometheus` is mounted as a **directory**, deliberately. Mounting the two
files individually breaks in a way that is silent and actively misleading.

A single-file bind mount resolves to an inode when the container starts. Most
editors — and `sed -i`, and anything that writes a temp file and renames it —
replace the file rather than truncating it, leaving the container bound to the
old, now-unlinked inode. The host file changes; the container never sees it.

What makes it nasty is that every health signal says success. `/-/reload`
returns `200` and `prometheus_config_last_reload_successful` stays `1`, because
Prometheus genuinely did reload the stale file it can see. Hit while adding the
`edge`/`tracing`/`storage` groups: `promtool` validated 17 rules on the host
while the container kept evaluating the original 6.

Confirm a suspected case by comparing inodes:

```bash
stat -c '%i %s' prometheus/alerts.yml
docker exec qwen36-27b-prometheus stat -c '%i %s' /etc/prometheus/alerts.yml
```

Different inode means the mount is stale and only a container recreate fixes
it. The same hazard applies to the remaining single-file mounts in this repo —
`Caddyfile`, `otel/collector.yaml`, `clickhouse/config.d/*.xml`. Those are left
as file mounts on purpose (mounting a directory over ClickHouse's `config.d`
would mask the image's own `docker_related_config.xml` and break its listeners),
and it is tolerable there because none of them is edited-and-hot-reloaded —
each is applied by recreating its container, which rebinds the mount anyway.

## Verification

```bash
# datasource healthy + readOnly
curl -s http://localhost:9090/-/ready
# every scrape target's health in one line each
curl -s http://localhost:9090/api/v1/targets \
  | python3 -c "import json,sys; [print(f\"{t['labels']['job']:16}{t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]"
# alert rule evaluation state (via Grafana API)
curl -s -u "$GRAFANA_ADMIN_USER:$GRAFANA_ADMIN_PASSWORD" \
  http://localhost:3000/api/prometheus/grafana/api/v1/rules
```
