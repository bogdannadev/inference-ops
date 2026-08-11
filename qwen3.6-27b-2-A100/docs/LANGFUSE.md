# Langfuse — LLM Trace Observability

Langfuse v4 self-hosted overlay for trace/observability of the inference
tier. Defined in `docker-compose.langfuse.yml`, same compose project as the
inference tier. It does not sit in the request path: the router and both
workers export spans asynchronously through an OpenTelemetry Collector, so a
Langfuse outage cannot apply backpressure to a generation.

Traces arrive **without any client-side SDK**. That is the point of the
collector — clients keep talking plain OpenAI API through Caddy and are never
aware tracing exists.

## Services

| Service | Container | Host port | Role |
|---|---|---|---|
| web | `qwen36-27b-langfuse-web` | `127.0.0.1:3001` | UI + ingestion API (container :3000) |
| worker | `qwen36-27b-langfuse-worker` | `127.0.0.1:3030` | background processing |
| postgres | `qwen36-27b-langfuse-postgres` | `127.0.0.1:5432` | metadata store |
| redis | `qwen36-27b-langfuse-redis` | `127.0.0.1:6379` | queues |
| clickhouse | `qwen36-27b-langfuse-clickhouse` | `127.0.0.1:8123`/`9000` | trace/event store; Prometheus metrics on internal `:9363` |
| minio | `qwen36-27b-langfuse-minio` | `127.0.0.1:9092`/`9093` | S3 object storage |
| otel-collector | `qwen36-27b-otel-collector` | none | OTLP ingest `:4317`/`:4318`, self-metrics `:8888` |

All host bindings are loopback-only; reach the UI via SSH tunnel:

```bash
ssh -L 3001:127.0.0.1:3001 <host>
# http://localhost:3001
```

Data lives in `./langfuse-data/` (gitignored). ClickHouse server config
overrides live in `./clickhouse/config.d/` — see **Storage and retention**.

## Bootstrap

One-shot, idempotent setup — generates missing secrets into `.env`, creates
data directories, pulls images, starts the full stack, waits for readiness:

```bash
./deploy/init-langfuse.sh
```

Generated/expected `.env` keys:

```bash
LANGFUSE_POSTGRES_PASSWORD   LANGFUSE_SALT
LANGFUSE_CLICKHOUSE_PASSWORD LANGFUSE_ENCRYPTION_KEY
LANGFUSE_REDIS_PASSWORD      LANGFUSE_NEXTAUTH_SECRET
LANGFUSE_MINIO_PASSWORD      LANGFUSE_INIT_USER_PASSWORD

LANGFUSE_INIT_ORG_NAME=Default
LANGFUSE_INIT_PROJECT_NAME=qwen36-27b
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_NAME=admin

LANGFUSE_PUBLIC_KEY=pk-lf-...      # canonical SDK names — source of truth
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_OTEL_AUTH=<derived>       # base64(public:secret) — never edit by hand
```

Headless init creates the first org/project/user on first boot.

**The key pair is the single source of truth; `LANGFUSE_OTEL_AUTH` is derived
from it.** Set the pair and re-run `init-langfuse.sh`, which regenerates the
base64 whenever the two disagree. Editing the pair without regenerating leaves
the collector on a stale credential, and the symptom — a `401` — is
indistinguishable from a wrong key.

> **First boot only.** `LANGFUSE_INIT_PROJECT_*` seeds the project with this
> pair, but is ignored once a project exists. An install created before these
> vars existed keeps the keys it was born with, so a generated pair is inert.
> On such an install, mint a pair in the UI (Project Settings → API Keys), put
> it in `.env` under the canonical names, and re-run the script. Until the keys
> are real, `otel-collector` logs 401s and drops spans — the inference tier is
> unaffected.

## Security

- **SSRF guard:** `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST` (from
  `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST` in `.env`) restricts which hosts
  Langfuse gateways may call. Set it to the docker-internal router
  (`qwen36-27b-router`) or leave empty to disable the whitelist. This governs
  Langfuse's *outbound* calls (playground, evals) and is unrelated to the
  inbound trace pipeline.
- **Collector exposure:** `otel-collector` publishes no host ports. Its OTLP
  listeners are reachable only from `qwen36-27b-backend`, and it holds a
  project-scoped Langfuse credential — not an admin one.
- Loopback bindings only; no service is reachable off-host except through the
  SSH tunnel.
- Telemetry disabled (`TELEMETRY_ENABLED=false`).

## Trace pipeline

```text
qwen36-27b-router  ─┐
qwen36-27b-r0      ─┼─ OTLP/gRPC :4317 ─► otel-collector ─► langfuse-web
qwen36-27b-r1      ─┘                     (batch + queue)   /api/public/otel
```

**Why a collector at all.** SGLang's exporter defaults to OTLP/**gRPC**
(`srt/observability/trace.py`); Langfuse accepts OTLP over **HTTP** only and
does not support gRPC. The collector is the protocol bridge — remove it and
the two ends cannot talk. It doubles as a buffer, so a Langfuse restart
queues spans instead of pushing backpressure onto the tokenizer and scheduler
threads.

**What the spans contain.** Engine-phase timings — queueing, prefill, decode,
per-slice work on the tokenizer and scheduler threads. **Not** prompts or
completions: SGLang traces the request lifecycle, not its content. That makes
this pipeline excellent for "where did those 400ms go" and useless for evals
or LLM-as-judge, which would need a gateway in the request path instead.

**Verbosity.** `SGLANG_TRACE_LEVEL` (set to `3` here, matching the upstream
default):

| Level | Emits |
|---|---|
| 0 | nothing — the kill-switch |
| 1 | important slices only |
| 2 | all slices except nested |
| 3 | all slices |

Level 3 was measured before being adopted, because the initial assumption was
that per-slice span construction would be too expensive to leave on:

| | level 1 | level 3 |
|---|---|---|
| span rate | 0.551/s | 0.691/s |
| failed exports | 0 | 0 |
| export queue | 0 | 0 |
| collector CPU | — | 0.03% of one core |

Storage came out at **~410 bytes per span** across `events_full` and
`events_core` — roughly 24 MB/day at the observed rate, against 1.2 TB free.
The caution was misplaced: this node's decode is bandwidth-bound at ~50ms per
forward pass, so host-side span construction is invisible, and the downstream
cost is four hundred bytes.

One caveat on that table: spans/sec conflates trace level with *request* rate
and the two windows saw different traffic, so treat `0.551 → 0.691` as
indicative rather than a clean delta. The robust figure is the per-span cost;
multiply it by peak request rate and it stays negligible.

Change it at runtime on a live replica — no restart, but the endpoint sits
behind `--api-key`:

```bash
curl -H "Authorization: Bearer $SGLANG_API_KEY" \
     "http://127.0.0.1:8001/set_trace_level?level=1"
```

`.env` carries `SGLANG_TRACE_LEVEL` so the setting survives a replica roll.

## Storage and retention

ClickHouse backs the traces, and left at stock settings it spends far more on
observing itself than on storing data. Measured five days after first boot:

| | |
|---|---|
| `system.trace_log` | 546 MiB (26.8M rows) |
| `system.text_log` | 184 MiB |
| `part_log` + `metric_log` + `asynchronous_metric_log` | 219 MiB |
| `clickhouse-server.log` (one unrotated file) | 830 MiB |
| **`default.*` — the actual traces** | **396 KiB** |

Two independent stock defaults caused it, and neither is Langfuse's doing:

- **The file logger** runs at `trace` with `1000M × 10` rotation — a 10 GB
  ceiling, filling at ~166 MB/day here. Docker's `max-size` never applied:
  ClickHouse writes these files itself, inside the bind mount.
- **Every system log table ships with its `ttl` commented out**, so they grow
  without bound. `trace_log` dominates because the query profiler samples on a
  wall-clock timer — an idle server still fills it.

Both are now bounded by `clickhouse/config.d/`:

| File | Effect |
|---|---|
| `logging.xml` | level `information`, `100M × 3`, archives gzipped — ~130 MB steady state |
| `system-log-ttl.xml` | 3-day TTL on high-churn diagnostics, 7-day on audit tables |
| `prometheus.xml` | enables the `:9363` metrics endpoint (stock ships it commented out) |

Applying the TTLs to an **existing** install takes a second step, because
ClickHouse reads `ttl` only when it *creates* a table:

```bash
./deploy/apply-clickhouse-retention.sh                 # set TTLs
./deploy/apply-clickhouse-retention.sh --drop-renamed  # reclaim orphans
```

> **Expect a one-time rename.** When a system log table's configured definition
> stops matching the table on disk — exactly what happens the first time
> `system-log-ttl.xml` lands — ClickHouse does not migrate it. It renames the
> existing table to `<name>_0` and creates a fresh empty one. Those orphans
> inherit no TTL and are referenced by nothing, so they sit on disk forever
> until dropped. `--drop-renamed` is that cleanup, opt-in because `DROP TABLE`
> is irreversible. Once the tables match config, later restarts reuse them and
> no further renames occur.

Two mounting rules worth not relearning:

- **Mount the XML files individually, never the `config.d` directory.** The
  image ships its own `docker_related_config.xml` there, and that file is what
  sets `listen_host` to the wildcards. Masking it makes ClickHouse listen on
  loopback inside its own namespace, and `langfuse-web`/`worker` lose the
  database with an error that points nowhere near the mount.
- **`opentelemetry_span_log` cannot take a `ttl` element.** It is the one
  system log whose stock block specifies an explicit `engine`, and ClickHouse
  rejects the combination with `Code: 36 ... 'ttl' setting doesn't make sense`
  — thrown during system-log init, before any port opens, so the container
  crash-loops with nothing on stdout and the trace only in
  `clickhouse-server.err.log`. Its retention is set by ALTER in the script
  instead.

**Operational coupling.** SGLang raises on exporter *initialisation* failure
rather than logging and continuing, so bringing the inference tier up without
this overlay is not a supported combination while `--enable-trace` is set.
Either keep the overlays together (the documented rule anyway) or set
`SGLANG_TRACE_LEVEL=0`. Because this touches the serving tier, apply it with
`deploy/roll-replica.sh` one replica at a time rather than a full restart.

**Health.** The collector is scraped by Prometheus on `:8888`; watch
`otelcol_exporter_send_failed_spans` and `otelcol_exporter_queue_size`. The
**Trace Pipeline** Grafana dashboard (folder `pipeline/`) panels all of it,
and the `tracing` alert group in `prometheus/alerts.yml` covers export
failures, queue fill, and receiver refusals.

The collector has no container healthcheck, deliberately: the core image is
distroless — no shell, so `CMD-SHELL` cannot run — and the `health_check`
extension that would answer an HTTP probe lives in the contrib distribution,
not this one. `up{job="otel-collector"}` is the equivalent signal, and the
failed-span counter catches the more interesting case where the process is
alive but nothing is reaching Langfuse.

## Sending traces from a client

Not required for engine traces — the pipeline above needs no client changes.
This is only for adding application-level spans of your own.


```bash
pip install langfuse
```

```python
from langfuse import Langfuse

langfuse = Langfuse(
    host="http://localhost:3001",
    public_key="<project-public-key>",
    secret_key="<project-secret-key>",
)
```

Project keys come from the Langfuse UI (Project Settings) after first login
with the headless-init user.

## Day-to-day

```bash
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
              -f docker-compose.langfuse.yml ps
docker logs -f qwen36-27b-langfuse-worker
```

All seven services must report `running`. Six of them also report `(healthy)`:
`langfuse-web` and `langfuse-worker` gained healthchecks in 2026-08 (web on
`/api/public/ready`, worker on `/api/health`) — before that they reported
`running` whether or not they could serve, and a web container that was up but
unable to reach ClickHouse looked identical to a working one. `otel-collector`
is the sole exception, for the distroless reason given above.

The worker is where background trace processing happens — check its logs if
traces show up in the UI but stay unprocessed.

If the UI stays empty instead, the break is upstream of the worker. Walk it
in order:

```bash
docker logs qwen36-27b-otel-collector | grep -i 'permanent error\|401\|refused'
docker logs qwen36-27b-r0 | grep -i 'opentelemetry\|otlp'
```

A `401` means `LANGFUSE_OTEL_AUTH` does not match a live project key pair —
almost always the first-boot-only trap described under Bootstrap.
