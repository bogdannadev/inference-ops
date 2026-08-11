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
| clickhouse | `qwen36-27b-langfuse-clickhouse` | `127.0.0.1:8123`/`9000` | trace/event store |
| minio | `qwen36-27b-langfuse-minio` | `127.0.0.1:9092`/`9093` | S3 object storage |
| otel-collector | `qwen36-27b-otel-collector` | none | OTLP ingest `:4317`/`:4318`, self-metrics `:8888` |

All host bindings are loopback-only; reach the UI via SSH tunnel:

```bash
ssh -L 3001:127.0.0.1:3001 <host>
# http://localhost:3001
```

Data lives in `./langfuse-data/` (gitignored).

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
LANGFUSE_INIT_USER_EMAIL=admin@bnna.dev
LANGFUSE_INIT_USER_NAME=admin

LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-...
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-...
LANGFUSE_OTEL_AUTH=<base64 of "public:secret">
```

Headless init creates the first org/project/user on first boot.

> **First boot only.** `LANGFUSE_INIT_*` is ignored once the project exists —
> an install created before these vars were added keeps the keys it was born
> with, and the seeded pair is inert. On such an install, mint a key pair in
> the UI (Project Settings → API Keys) and replace all three values:
>
> ```bash
> printf '%s:%s' "$PUBLIC_KEY" "$SECRET_KEY" | base64 -w0   # LANGFUSE_OTEL_AUTH
> ```
>
> `init-langfuse.sh` detects this case and prints the same warning. Until the
> keys are real, `otel-collector` logs 401s and drops spans — the inference
> tier is unaffected.

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

**Verbosity.** `SGLANG_TRACE_LEVEL` (default `1` here, upstream default `3`):

| Level | Emits |
|---|---|
| 0 | nothing — the kill-switch |
| 1 | important slices only |
| 2 | all slices except nested |
| 3 | all slices |

We default to 1 deliberately. Level 3 creates a span per nested slice on the
host-side threads, and on a node whose decode is bandwidth-bound at roughly
50ms per forward pass there is no host-side headroom worth donating to span
construction. Raise it temporarily during an investigation — no restart
needed:

```bash
curl "http://qwen36-27b-r0:8001/set_trace_level?level=3"
```

**Operational coupling.** SGLang raises on exporter *initialisation* failure
rather than logging and continuing, so bringing the inference tier up without
this overlay is not a supported combination while `--enable-trace` is set.
Either keep the overlays together (the documented rule anyway) or set
`SGLANG_TRACE_LEVEL=0`. Because this touches the serving tier, apply it with
`deploy/roll-replica.sh` one replica at a time rather than a full restart.

**Health.** The collector is scraped by Prometheus on `:8888`; watch
`otelcol_exporter_send_failed_spans` and `otelcol_exporter_queue_size`.

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

All seven services must report `running`; infra ones also report `(healthy)`.
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
