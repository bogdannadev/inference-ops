# Langfuse — LLM Trace Observability

Langfuse v4 self-hosted overlay for trace/observability of the inference
tier. Defined in `docker-compose.langfuse.yml`, same compose project as the
inference tier. It does not sit in the request path — clients emit traces to
it independently (e.g. from the Langfuse SDK).

## Services

| Service | Container | Host port | Role |
|---|---|---|---|
| web | `qwen36-27b-langfuse-web` | `127.0.0.1:3001` | UI + ingestion API (container :3000) |
| worker | `qwen36-27b-langfuse-worker` | `127.0.0.1:3030` | background processing |
| postgres | `qwen36-27b-langfuse-postgres` | `127.0.0.1:5432` | metadata store |
| redis | `qwen36-27b-langfuse-redis` | `127.0.0.1:6379` | queues |
| clickhouse | `qwen36-27b-langfuse-clickhouse` | `127.0.0.1:8123`/`9000` | trace/event store |
| minio | `qwen36-27b-langfuse-minio` | `127.0.0.1:9092`/`9093` | S3 object storage |

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
```

Headless init creates the first org/project/user on first boot.

## Security

- **SSRF guard:** `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST` (from
  `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST` in `.env`) restricts which hosts
  Langfuse gateways may call. Set it to the docker-internal router
  (`qwen36-27b-router`) or leave empty to disable the whitelist.
- Loopback bindings only; no service is reachable off-host except through the
  SSH tunnel.
- Telemetry disabled (`TELEMETRY_ENABLED=false`).

## Sending traces from a client

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

All six services must report `running`; infra ones also report `(healthy)`.
The worker is where background trace processing happens — check its logs if
traces show up in the UI but stay unprocessed.
