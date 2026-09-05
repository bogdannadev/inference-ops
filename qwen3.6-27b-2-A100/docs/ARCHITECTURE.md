# Architecture

## Host

```text
2× NVIDIA A100 80GB PCIe      (no GPU peer-to-peer — hence two TP=1 replicas)
AMD EPYC 7663, NPS2, SMT off
cores 0-27  -> NUMA node 0    (r0)
cores 28-55 -> NUMA node 1    (r1 28-47, observability/edge 48-55)
1 TB RAM
```

GPU P2P is unavailable, so the model is served as two independent `TP=1`
replicas rather than one `TP=2` server: no inter-GPU all-reduce, no P2P
dependency, better fault isolation.

## Networks

Two Docker networks:

| Network | Type | Members | Purpose |
|---|---|---|---|
| `edge` | external (created once outside compose) | Caddy, router, `qwen3-emb`, Prometheus | everything reachable from the public edge |
| `qwen36-27b-backend` | project bridge | workers, router, Prometheus, Grafana, exporters, Langfuse, OTel Collector | internal traffic |

CVE-2026-3059/3060 lateral-movement hardening: Caddy can reach the router but
**never** a worker port or ZMQ socket.

Prometheus is the one service on both networks. It joined `edge` to scrape
Caddy's `:2020` metrics site, and that direction was chosen precisely to keep
the hardening above intact — attaching Caddy to the backend network instead
would have handed the internet-facing proxy a route to the worker ports.
Scrapes are outbound from Prometheus; nothing on `edge` gains a path inward.

```text
docker network create edge     # once, before first up
```

## Request data flow

```text
internet
   |
   v
Caddy  (80/443, TLS, edge-auth key, no body cap — see Caddyfile)
   |  edge key swapped for internal SGLANG_API_KEY
   v
qwen36-27b-router:8000   (OpenAI API, cache_aware)
   |            \
   v             v
qwen36-27b-r0:8001    qwen36-27b-r1:8002
   (GPU0, TP=1)        (GPU1, TP=1)
```

Trace flow runs alongside it, never in the request path — spans are exported
asynchronously off a batch processor, so nothing here blocks on Langfuse:

```text
router + r0 + r1  --OTLP/gRPC-->  otel-collector:4317
                                       |  OTLP/HTTP + Basic auth
                                       v
                            langfuse-web:3000/api/public/otel
```

The collector is a protocol bridge, not decoration: SGLang exports OTLP/gRPC,
and Langfuse accepts OTLP over HTTP only. Tracing the router as well as the
workers is what stitches routing and queueing into the same distributed trace
as the forward pass, instead of two disconnected span sets.

## Services and endpoints

All names resolve on `qwen36-27b-backend` unless noted. Ports marked `host`
bind to `127.0.0.1` only.

### Inference tier — `docker-compose.yml`

| Service | Endpoint | Notes |
|---|---|---|
| `qwen36-27b-r0` | `http://qwen36-27b-r0:8001` | GPU `device_ids: ["0"]`, cpuset `0-27` |
| `qwen36-27b-r1` | `http://qwen36-27b-r1:8002` | GPU `device_ids: ["1"]`, cpuset `28-47` |
| `qwen36-27b-router` | `http://qwen36-27b-router:8000` | OpenAI-compatible; metrics on `:29000` |
| `caddy` | host `80/443` | only service publishing host ports |

Both replicas run byte-identical flags (only `--port` differs) from the same
digest-pinned engine image. `--context-length 169000`, `--mem-fraction-static
0.92`, `--max-running-requests 4`, EAGLE 6/5, radix cache
`--mamba-radix-cache-strategy extra_buffer`. Full flag rationale is inline in
`docker-compose.yml`.

Router policy is `cache_aware --balance-abs-threshold 2` (2026-09-05).

`cache_aware` was disabled in July because it starved r0 — one shared hot
system prompt piled the whole team onto one worker. The cause was found on
2026-09-05: balancing requires `(max_load - min_load) > abs_threshold`, whose
default is **64**, and with `--max-running-requests 4` per replica behind a
16-request router that difference can never exceed ~16. The guard could not
fire, so affinity ran unchecked. At a threshold of 2 it fires normally.

Measured against a same-day `round_robin` control: shared-prefix cache hit
65.1% -> 97.7%, split 10/3 instead of 0/122, p50 unchanged, tail improved.
Full record in `tuning/docs/ROUTING.md`.

### Observability tier — `docker-compose.metrics.yml`

| Service | Endpoint | Notes |
|---|---|---|
| `prometheus` | host `127.0.0.1:9090` | 15s global scrape, 30d/20GB retention, hot reload via `--web.enable-lifecycle`, `./prometheus` mounted as a directory |
| `grafana` | host `127.0.0.1:3000` | 7 dashboards + SLO alert rules, auto-provisioned |
| `node-exporter` | `qwen36-27b-node-exporter:9100` | host CPU/RAM/disk/network |
| `dcgm-exporter` | `dcgm-exporter:9400` | per-GPU telemetry, 1000ms sampling |
| `alertmanager` | `qwen36-27b-alertmanager:9093` | alert delivery to Telegram via quota-bot; on `edge` only — see `docs/OBSERVABILITY.md` |

### Langfuse tier — `docker-compose.langfuse.yml`

| Service | Host port | Notes |
|---|---|---|
| `qwen36-27b-langfuse-web` | `127.0.0.1:3001` | UI + ingest (container :3000) |
| `qwen36-27b-langfuse-worker` | `127.0.0.1:3030` | background trace processing |
| `qwen36-27b-langfuse-postgres` | `127.0.0.1:5432` | metadata |
| `qwen36-27b-langfuse-redis` | `127.0.0.1:6379` | queues |
| `qwen36-27b-langfuse-clickhouse` | `127.0.0.1:8123` / `9000` | trace/event storage; Prometheus metrics on internal `:9363`; server config overrides from `./clickhouse/config.d/` |
| `qwen36-27b-langfuse-minio` | `127.0.0.1:9092` / `9093` | object storage |
| `qwen36-27b-otel-collector` | none (backend only) | OTLP `:4317`/`:4318` in, Langfuse HTTP out, `:8888` self-metrics |

### Embedding co-tenant — `docker-compose.embedder.yml`

`qwen3-emb` runs TEI CPU 1.9 (Qwen3-Embedding-0.6B ONNX) as a **separate
compose project**; it only joins the shared `edge` network so Caddy can reach
it at `http://qwen3-emb:80`.

## Environment variables

Everything is read from `.env` (gitignored). Required by the inference tier:

```bash
SGLANG_API_KEY=...      # internal worker/API key
EDGE_API_KEY=...        # public edge key (clients present this)
HF_TOKEN=...            # Hugging Face token
GRAFANA_ADMIN_USER=...  # metrics overlay
GRAFANA_ADMIN_PASSWORD=...
```

Langfuse adds its own set (Postgres/ClickHouse/Redis/MinIO passwords, salt,
encryption key, NextAuth secret, init-user password, project key pair and the
collector's `LANGFUSE_OTEL_AUTH`) — generated by `./deploy/init-langfuse.sh`
if absent.

The inference tier gained optional trace knobs, all with defaults:

```bash
SGLANG_TRACE_LEVEL=3          # 0=off 1=important 2=all-but-nested 3=all
SGLANG_OTLP_SCHEDULE_DELAY_MS=1000
SGLANG_OTLP_MAX_BATCH=256
```

Level 3 was adopted after measurement, not by default — ~410 bytes per span
and no measurable engine cost on a node whose decode is bandwidth-bound. See
`docs/LANGFUSE.md` for the numbers.

`SGLANG_TRACE_LEVEL=0` is the kill-switch — it stops span emission without
editing the command blocks. The level is also settable at runtime with no
restart, though the endpoint sits behind `--api-key`:

```bash
curl -H "Authorization: Bearer $SGLANG_API_KEY" \
     "http://127.0.0.1:8001/set_trace_level?level=2"
```

## Image pinning

- SGLang engine: **pinned by digest** in `docker-compose.yml`
  (`x-sglang-image` anchor) — the floating `:dev-cu13` tag silently moved r0
  onto a new build mid-session once; digest pinning makes that impossible.
- node-exporter `v1.12.1`; other metric images are floating-tag placeholders
  (`# PLACEHOLDER — resolve tag, pin digest`) pending a pinning pass.

## Compose operational rule

The overlays share the project (`qwen36-27b-2-a100`). Always pass them
together, or compose will report their containers as orphans:

```bash
docker compose -f docker-compose.yml -f docker-compose.metrics.yml ps
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
              -f docker-compose.langfuse.yml up -d
```

Never use `--remove-orphans` — it would delete `qwen3-emb`, `grafana`,
`prometheus`, `dcgm`, which belong to other projects.
