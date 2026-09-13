# Qwen3.8-27B on 2×A100 — SGLang Inference Node

Private deployment serving `Qwen/Qwen3.8-27B` (BF16) with SGLang on a single
host with two A100 80GB PCIe GPUs. Two independent `TP=1` replicas (one per
GPU, no P2P) sit behind a `cache_aware` SGLang router, which is exposed to the
outside world only through a Caddy edge gateway.

> **Names still say `qwen36`.** Containers, networks, the router, the Caddy
> upstream and the API-visible `--served-model-name` all remain `qwen36-27b`
> after the 2026-08-15 move to Qwen3.8-27B. Deliberate: the name has ~194
> references across compose, `Caddyfile`, `grafana/`, `prometheus/` and
> `benchmarks/`, and renaming the services would make a replica-by-replica
> roll impossible. Clients still request model `qwen36-27b`. **Read
> `--model-path`, not the container name, to know which weights are loaded.**
> See [`tuning/docs/UPGRADE_QWEN3.8.md`](tuning/docs/UPGRADE_QWEN3.8.md).

The deployment is assembled from **four compose files** in one project:

| File | What it adds |
|---|---|
| `docker-compose.yml` | inference tier: 2 workers, router, Caddy |
| `docker-compose.metrics.yml` | observability: Prometheus, Grafana, node-exporter, DCGM |
| `docker-compose.langfuse.yml` | LLM trace observability: Langfuse v4 + Postgres/ClickHouse/MinIO/Redis + OTel Collector |
| `docker-compose.embedder.yml` | **separate project** — TEI CPU embedding co-tenant |

> **Compose rule — always pass the project files together:**
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.metrics.yml <cmd>
> # + -f docker-compose.langfuse.yml for the Langfuse services
> ```
> The overlays share the project network and volumes; running compose with a
> single file lets them be treated as orphans.

## How it fits together

Three layers, each a loop. The request loop runs per request, the model loop
runs per forward pass inside it, and the observability loop runs on human
timescales and feeds back into the settings the other two obey.

```text
┌─ LAYER 1 ─ EDGE & ROUTING ───────────────────── one pass per request ──────┐
│                                                                            │
│   client                                                                   │
│     │  HTTPS + edge key                                                    │
│     ▼                                                                      │
│   caddy :443            TLS · 401 guard · key swap · no body cap           │
│     │                   ── rejects here never reach SGLang ──┐             │
│     ▼                                                        │             │
│   sgl-router :8000      cache_aware · injects `traceparent`  │             │
│     │                                                        │             │
│     ├──────────────┬───────────────────────────────────┐     │             │
│     ▼              ▼                                   │     │             │
│   r0 (GPU0)      r1 (GPU1)      TP=1 each, no P2P      │     │             │
│     │              │                                   │     │             │
└─────┼──────────────┼───────────────────────────────────┼─────┼─────────────┘
      │              │                                   │     │
      ▼              ▼                                   │     │
┌─ LAYER 2 ─ MODEL WORK ────────────────── one pass per token step ──────────┐
│                                                                       │    │
│   ┌──────────────────────────────────────────────────────────────┐    │    │
│   │  SCHEDULE      max-running-requests 4                        │    │    │
│   │     │          continuous batching                           │    │    │
│   │     ▼                                                        │    │    │
│   │  PREFILL ──────► radix prefix cache ──► reuse or compute     │    │    │
│   │     │            + HiCache host tier    cached tokens skip   │    │    │
│   │     ▼                                   prefill entirely     │    │    │
│   │  DECODE LOOP ◄───────────────────────────────────┐           │    │    │
│   │     │  DFlash2 drafter: block of 8, one pass     │           │    │    │
│   │     │  verify in ONE forward pass ───────────────┘           │    │    │
│   │     │  accepted ≈ N tokens for the cost of 1 pass            │    │    │
│   │     ▼                                                        │    │    │
│   │  KV POOL       182,528 tokens · mem-fraction-static 0.94     │    │    │
│   │     │          flashinfer attention · CUDA graph bs 1-4      │    │    │
│   │     ▼                                                        │    │    │
│   │  STREAM ───► SSE tokens back up through router and Caddy     │    │    │
│   └──────────────────────────────────────────────────────────────┘    │    │
│                                                                       │    │
│   Bandwidth-bound: ~50ms per forward pass is weight movement, not     │    │
│   math. Only bytes-moved-per-pass changes the number — which is why   │    │
│   DFlash2 (fewer bytes per step) won and host-side tweaks did not.    │    │
└───────────────────────────────────────────────────────────────────────┼────┘
                                                                        │
        metrics (pull, 5-30s)          traces (push, async) ────────────┘
              │                              │
              ▼                              ▼
┌─ LAYER 3 ─ OBSERVABILITY ─────────────── the loop that changes things ─────┐
│                                                                            │
│   prometheus ◄── scrapes ── workers · router · caddy · dcgm · node         │
│      │                      clickhouse · otel-collector · self             │
│      │                                                                     │
│      │              otel-collector ◄── OTLP/gRPC ── engine + router spans  │
│      │                    │  protocol bridge + buffer                      │
│      │                    ▼  OTLP/HTTP + Basic auth                        │
│      │              langfuse ──► clickhouse   per-request span timings     │
│      ▼                    │                                                │
│   grafana ◄───────────────┘                                                │
│      │   7 dashboards: overview · sglang · router · gpu · host             │
│      │                 edge · pipeline                                     │
│      │   17 prometheus alerts + 7 grafana SLO rules                        │
│      ▼                                                                     │
│   a human reads a regression                                               │
│      │                                                                     │
│      ▼                                                                     │
│   change a setting in docker-compose.yml / Caddyfile / config.d            │
│      │                                                                     │
│      ▼                                                                     │
│   deploy/roll-replica.sh  ── one replica at a time, peer stays serving     │
│      │                                                                     │
│      └────────────────► back into LAYER 1 and LAYER 2 ─────────────────────┘
└────────────────────────────────────────────────────────────────────────────┘
```

**Reading the three loops**

| Loop | Period | Closes when |
|---|---|---|
| Request | ms–minutes | tokens finish streaming back to the client |
| Model work | ~50 ms | a forward pass verifies its draft tokens and appends to the KV pool |
| Observability | hours–weeks | a measurement changes a setting, and a rolled replica proves it |

The third loop is the reason the other two are worth instrumenting: every entry
in *Current optimization state* below arrived through it, and the tuning
campaign under `tuning/` is that loop run deliberately.

Each layer has exactly one blind spot the layer below cannot see, which is why
all three are instrumented separately:

- **Caddy** sees requests that never reach SGLang — TLS failures, 401s from a
  wrong edge key. On the engine dashboards those look like *silence*.
- **The engine** sees queueing, prefill, decode and cache behaviour per
  request, which no edge metric can decompose.
- **The trace pipeline** stitches router and worker spans into one distributed
  trace via `traceparent`, answering "where did those 400 ms go" across a hop
  neither side can see alone.

## Quick start

```bash
# 1. secrets — create .env with at minimum:
#    SGLANG_API_KEY, EDGE_API_KEY, HF_TOKEN (and GRAFANA_ADMIN_USER/PASSWORD)
#    (full var list in docs/ARCHITECTURE.md; Langfuse secrets are generated below)

# 2. one-shot Langfuse setup (generates its secrets, data dirs, starts everything)
./deploy/init-langfuse.sh

# 3. verify
docker compose -f docker-compose.yml -f docker-compose.metrics.yml ps
```

## Documentation

The in-repo `docs/` directory is the operational manual for this node:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — host topology, networks,
  every service with its endpoint, request data flow, env vars
- **[docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)** — Prometheus scrape
  config, Grafana dashboards, alert rules (Prometheus-native + Grafana SLO),
  SSH-tunnel access
- **[docs/OPERATIONS.md](docs/OPERATIONS.md)** — daily ops: health checks,
  rolling a replica, logs, rollback, troubleshooting
- **[docs/LANGFUSE.md](docs/LANGFUSE.md)** — Langfuse trace overlay: services,
  secrets, headless init, SDK access

- **[tuning/docs/UPGRADE_QWEN3.8.md](tuning/docs/UPGRADE_QWEN3.8.md)** —
  2026-08-15 Qwen3.6 → Qwen3.8 weights swap: why no engine change, why the
  vendor `qwen38-27b-cu129` image was rejected, the two chat-template
  behaviour changes and the flag that pins them, boot gates, rollback

Tuning campaign material lives in its own tree: [`tuning/README.md`](tuning/README.md),
results and decision records under `tuning/docs/` and `tuning/results/`.

## Layout

```text
.
├── docker-compose.yml            # inference tier (workers, router, Caddy)
├── docker-compose.metrics.yml    # Prometheus + Grafana + exporters
├── docker-compose.langfuse.yml   # Langfuse trace observability
├── docker-compose.embedder.yml   # TEI CPU embedding (separate project)
├── Caddyfile                     # edge gateway: TLS, auth, no body cap
├── deploy/
│   ├── roll-replica.sh           # zero-downtime single-replica roll
│   ├── init-langfuse.sh          # one-shot Langfuse bootstrap
│   └── apply-clickhouse-retention.sh   # one-time system-log TTLs
├── grafana/
│   └── provisioning/             # datasource, dashboards, alert rules
├── prometheus/
│   ├── prometheus.yml            # scrape config
│   └── alerts.yml                # Prometheus-native alert rules
├── otel/
│   └── collector.yaml            # OTLP gRPC -> Langfuse HTTP bridge
├── clickhouse/
│   └── config.d/                 # log rotation, system-log TTLs, :9363 metrics
├── docs/                         # this node's operational manual
├── benchmarks/                   # latency/throughput harnesses
├── tuning/                       # kernel/flag tuning campaign
├── logs/                         # runtime log bind mounts (gitignored)
└── langfuse-data/                # Langfuse DB object storage (gitignored)
```

## Service inventory (one line each)

- `qwen36-27b-r0` / `qwen36-27b-r1` — SGLang `TP=1` replicas, GPU0/GPU1,
  DFlash2 speculative decoding, Mamba radix prefix caching + HiCache host tier
- `qwen36-27b-router` — SGLang model-gateway, `cache_aware`, OpenAI API, :8000
- `caddy` — TLS termination, edge-auth key swap, unbounded body size (only host ports)
- `prometheus` — 9 scrape targets, 30d/20GB retention, 16 alert rules, hot reload
- `grafana` — 7 dashboards (overview/sglang/router/gpu/host/edge/pipeline), SLO rules
- `node-exporter` — host CPU/RAM/disk/network
- `dcgm-exporter` — per-GPU utilisation/memory/power/occupancy
- `qwen36-27b-langfuse-*` — Postgres, Redis, ClickHouse, MinIO, web, worker
- `qwen36-27b-otel-collector` — OTLP bridge: SGLang gRPC spans → Langfuse HTTP
- `qwen3-emb` — TEI CPU embeddings (separate project, joins `edge` only)

## Current optimization state (2026-09-13)

```text
2× TP=1 replicas            yes        no GPU P2P on this host
DFlash2 drafter             block 8, fp8 draft KV   tuning/docs/HICACHE_DFLASH2.md
mem-fraction-static         0.94       KV pool 182,528 tokens / replica
chunked prefill             4096       DFlash2 leaves ~2.9 GB headroom
context-length              169,000
max-running-requests        4          decode CUDA graph bs [1,2,3,4]
attention backend           flashinfer
prefix caching              radix tree, mamba extra_buffer
router policy               cache_aware --balance-abs-threshold 2
only host ports             80/443     Caddy; everything else loopback-only
HiCache                     ratio 3    ~52 GB pinned host RAM / replica
request tracing             OTLP -> collector -> Langfuse, level 3
trace cost                  ~410 B/span, ~24 MB/day — measured, not assumed
clickhouse retention        3d diagnostics / 7d audit, logs capped at ~130 MB
```

## External access

```text
https://model.example.com    # Caddy -> router :8000 (edge key required)
https://embed.example.com     # Caddy -> qwen3-emb (edge key required)
```

Everything else binds to `127.0.0.1` on the host and is reached over an SSH
tunnel (see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the port map).
