# Qwen3.6-27B on 2×A100 — SGLang Inference Node

Private deployment serving `Qwen/Qwen3.6-27B` (BF16) with SGLang on a single
host with two A100 80GB PCIe GPUs. Two independent `TP=1` replicas (one per
GPU, no P2P) sit behind a `round_robin` SGLang router, which is exposed to the
outside world only through a Caddy edge gateway.

The deployment is assembled from **four compose files** in one project:

| File | What it adds |
|---|---|
| `docker-compose.yml` | inference tier: 2 workers, router, Caddy |
| `docker-compose.metrics.yml` | observability: Prometheus, Grafana, node-exporter, DCGM |
| `docker-compose.langfuse.yml` | LLM trace observability: Langfuse v4 + Postgres/ClickHouse/MinIO/Redis |
| `docker-compose.embedder.yml` | **separate project** — TEI CPU embedding co-tenant |

> **Compose rule — always pass the project files together:**
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.metrics.yml <cmd>
> # + -f docker-compose.langfuse.yml for the Langfuse services
> ```
> The overlays share the project network and volumes; running compose with a
> single file lets them be treated as orphans.

## Quick start

```bash
# 1. secrets — create .env with at minimum:
#    SGLANG_API_KEY, EDGE_API_KEY, HF_TOKEN (and GRAFANA_ADMIN_USER/PASSWORD)
#    (full var list in docs/ARCHITECTURE.md; Langfuse secrets are generated below)

# 2. one-shot Langfuse setup (generates its secrets, data dirs, starts everything)
./deploy/init-langfuse.sh

# 3. verify
./test.sh
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
  rolling a replica, `test.sh`, logs, rollback, troubleshooting
- **[docs/LANGFUSE.md](docs/LANGFUSE.md)** — Langfuse trace overlay: services,
  secrets, headless init, SDK access

Tuning campaign material lives in its own tree: [`tuning/README.md`](tuning/README.md),
results and decision records under `tuning/docs/` and `tuning/results/`.

## Layout

```text
.
├── docker-compose.yml            # inference tier (workers, router, Caddy)
├── docker-compose.metrics.yml    # Prometheus + Grafana + exporters
├── docker-compose.langfuse.yml   # Langfuse trace observability
├── docker-compose.embedder.yml   # TEI CPU embedding (separate project)
├── Caddyfile                     # edge gateway: TLS, auth, body cap
├── deploy/
│   ├── roll-replica.sh           # zero-downtime single-replica roll
│   └── init-langfuse.sh          # one-shot Langfuse bootstrap
├── grafana/
│   └── provisioning/             # datasource, dashboards, alert rules
├── prometheus/
│   ├── prometheus.yml            # scrape config
│   └── alerts.yml                # Prometheus-native alert rules
├── docs/                         # this node's operational manual
├── benchmarks/                   # latency/throughput harnesses
├── tuning/                       # kernel/flag tuning campaign
├── test.sh                       # full deployment test suite
├── logs/                         # runtime log bind mounts (gitignored)
└── langfuse-data/                # Langfuse DB object storage (gitignored)
```

## Service inventory (one line each)

- `qwen36-27b-r0` / `qwen36-27b-r1` — SGLang `TP=1` replicas, GPU0/GPU1,
  EAGLE speculative decoding, Mamba radix prefix caching
- `qwen36-27b-router` — SGLang model-gateway, `round_robin`, OpenAI API, :8000
- `caddy` — TLS termination, edge-auth key swap, 8 MB body cap (only host ports)
- `prometheus` — 15s scrape of all tiers, 30d retention, rule evaluation
- `grafana` — dashboards (overview/sglang/router/gpu/host), SLO alert rules
- `node-exporter` — host CPU/RAM/disk/network
- `dcgm-exporter` — per-GPU utilisation/memory/power/occupancy
- `qwen36-27b-langfuse-*` — Postgres, Redis, ClickHouse, MinIO, web, worker
- `qwen3-emb` — TEI CPU embeddings (separate project, joins `edge` only)

## Current optimization state (2026-08)

```text
2× TP=1 replicas            yes        no GPU P2P on this host
EAGLE (MTP head)            6 draft tokens / 5 steps / topk 1
mem-fraction-static         0.92       KV pool 171,008 tokens / replica
context-length              169,000
max-running-requests        4          decode CUDA graph bs [1,2,3,4]
attention backend           flashinfer
prefix caching              radix tree, mamba extra_buffer (HiCache removed)
router policy               round_robin (cache_aware starved r0)
only host ports             80/443     Caddy; everything else loopback-only
HiCache                     disabled   device radix tree unaffected
```

## External access

```text
https://qw36-27b.bnna.dev    # Caddy -> router :8000 (edge key required)
https://qw3-emb.bnna.dev     # Caddy -> qwen3-emb (edge key required)
```

Everything else binds to `127.0.0.1` on the host and is reached over an SSH
tunnel (see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the port map).
