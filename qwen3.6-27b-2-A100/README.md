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
| `docker-compose.clickhouse.yml` | ClickHouse: per-request usage records (`engine.requests`, `gateway.requests`) and the exact usage aggregates |
| `docker-compose.embedder.yml` | **separate project** — TEI CPU embedding co-tenant |

> **Compose rule — always pass the project files together:**
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.metrics.yml <cmd>
> # + -f docker-compose.clickhouse.yml for ClickHouse
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
│   sgl-router            cache_aware · injects `traceparent`  │             │
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
        metrics (pull, 5-30s)    per-request records (file, async) ─────┘
              │                              │
              ▼                              ▼
┌─ LAYER 3 ─ OBSERVABILITY ─────────────── the loop that changes things ─────┐
│                                                                            │
│   prometheus ◄── scrapes ── workers · router · caddy · dcgm · node         │
│      │                      clickhouse · redis ledger · quota-bot · self   │
│      │                                                                     │
│      │              vector ◄── engine request files + gateway access log   │
│      │                │                                                    │
│      │                ▼                                                    │
│      │              clickhouse   engine.requests · gateway.requests        │
│      │                │  exact per-consumer sums, scraped as gauges        │
│      ◄────────────────┘                                                    │
│      ▼                                                                     │
│   grafana · quota-bot · alertmanager                                       │
│      │   dashboards: overview · sglang · router · gpu · host · edge ·      │
│      │               usage & quota · gateway                               │
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
- **The per-request records** — one row per request from the engine and one
  from the gateway, joined exactly on the response id — answer "where
  did those 400 ms go" (queue, prefill, decode, cache split) for one request,
  and give exact per-key usage. See `docs/METRICS-ECOSYSTEM.md`.

## Quick start

```bash
# 1. secrets — create .env with at minimum:
#    SGLANG_API_KEY, EDGE_API_KEY, HF_TOKEN, GRAFANA_ADMIN_USER/PASSWORD,
#    and the public hostnames: EDGE_HOST_* + ACME_CONTACT + GRAFANA_PUBLIC_URL
#    (full var list in docs/ARCHITECTURE.md)

# 2. start the stack, then create the usage tables and the scrape secret
docker compose -f docker-compose.yml -f docker-compose.metrics.yml -f docker-compose.clickhouse.yml up -d
./deploy/apply-gateway-schema.sh && ./deploy/apply-engine-schema.sh
./deploy/render-prometheus-secrets.sh

# 3. verify
docker compose -f docker-compose.yml -f docker-compose.metrics.yml -f docker-compose.clickhouse.yml ps
./deploy/test-usage-sql.sh
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
- **[docs/METRICS-ECOSYSTEM.md](docs/METRICS-ECOSYSTEM.md)** — which store
  answers which question: ledger, per-request records, exact usage gauges
- **[docs/OPENCODE_SETUP_PROMPT.md](docs/OPENCODE_SETUP_PROMPT.md)** and
  **[docs/HERMES_SETUP_PROMPT.md](docs/HERMES_SETUP_PROMPT.md)** — what a
  customer pastes into a fresh client so it configures itself against this
  node's real limits. quota-bot `/opencode <name>` sends the OpenCode one with
  the generated `opencode.json`

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
├── docker-compose.clickhouse.yml # ClickHouse: per-request records, exact usage
├── docker-compose.embedder.yml   # TEI CPU embedding (separate project)
├── Caddyfile                     # edge gateway: TLS, auth, no body cap
├── deploy/
│   ├── roll-replica.sh           # zero-downtime single-replica roll
│   ├── apply-engine-schema.sh    # engine.requests table
│   ├── render-engine-metrics-handler.sh  # SQL -> ClickHouse HTTP handler
│   ├── test-usage-sql.sh         # SQL regression cases
│   └── apply-clickhouse-retention.sh   # one-time system-log TTLs
├── grafana/
│   └── provisioning/             # datasource, dashboards, alert rules
├── prometheus/
│   ├── prometheus.yml            # scrape config
│   └── alerts.yml                # Prometheus-native alert rules
├── vector/                       # access log + engine records -> ClickHouse
├── clickhouse/
│   ├── engine-requests.sql       # per-request engine records
│   ├── engine-usage-metrics.sql  # exact per-consumer aggregates (Prometheus format)
│   ├── users.d/                  # read-only scrape and admin-mcp users
│   └── config.d/                 # log rotation, TTLs, :9363 metrics, usage handler
├── docs/                         # this node's operational manual
├── benchmarks/                   # latency/throughput harnesses
├── tuning/                       # kernel/flag tuning campaign
├── logs/                         # runtime log bind mounts (gitignored)
└── langfuse-data/clickhouse/     # ClickHouse data (gitignored; path kept from the Langfuse era)
```

## Service inventory (one line each)

- `qwen36-27b-r0` / `qwen36-27b-r1` — SGLang `TP=1` replicas, GPU0/GPU1,
  DFlash2 speculative decoding, Mamba radix prefix caching + HiCache host tier
- `qwen36-27b-router` — SGLang model-gateway, `cache_aware`, OpenAI API
- `caddy` — TLS termination, edge-auth key swap, unbounded body size (only host ports)
- `prometheus` — 9 scrape targets, 30d/20GB retention, 16 alert rules, hot reload
- `grafana` — 7 dashboards (overview/sglang/router/gpu/host/edge/pipeline), SLO rules
- `node-exporter` — host CPU/RAM/disk/network
- `dcgm-exporter` — per-GPU utilisation/memory/power/occupancy
- `qwen36-27b-langfuse-clickhouse` — ClickHouse (name kept from the Langfuse era): `engine.requests`, `gateway.requests`, `/engine_usage_metrics`
- `qwen36-27b-vector` — ships the gateway access log and the engine's per-request files to ClickHouse
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
request records             --export-metrics-to-file -> Vector -> ClickHouse (no prompt text)
tracing                     off (Langfuse and OTLP removed 2026-09-14)
clickhouse retention        3d diagnostics / 7d audit, logs capped at ~130 MB
```

## External access

Caddy publishes 80/443 and is the only service with host ports. It answers on
six hostnames: the direct model edge and the embedder (both shared-key), the
paid gateway, Grafana, the quota-bot webhook and admin-mcp.

**The names are not in this repo.** Every site address in the `Caddyfile`
reads an `EDGE_HOST_*` variable, and the values live in the gitignored `.env`
(`EDGE_HOST_MODEL`, `EDGE_HOST_EMBED`, `EDGE_HOST_BOT`, `EDGE_HOST_GATEWAY`,
`EDGE_HOST_GRAFANA`, `EDGE_HOST_MCP`, plus `ACME_CONTACT` for certificate
issuance). The committed config still shows what each site does; it just does
not say which deployment it is. An unset variable leaves the site address
empty and Caddy refuses to start, so a missing value fails at boot.

The edge also strips the response headers that described it — `Server`, `Via`
and the Envoy/Higress timing headers — see the `hide_edge` snippet in the
`Caddyfile`.

Everything else binds to `127.0.0.1` on the host and is reached over an SSH
tunnel (see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)).
