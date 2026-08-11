#!/usr/bin/env bash
#
# init-langfuse.sh
#
# One-shot setup for the Langfuse observability overlay. Idempotent:
# secrets are only generated if not already present in .env. Data dirs
# are created if missing. The stack is started with all three compose
# files.
#
#   ./deploy/init-langfuse.sh
#
# After first run, access the UI via SSH tunnel:
#   ssh -L 3001:127.0.0.1:3001 <host>
#   http://localhost:3001

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.metrics.yml -f docker-compose.langfuse.yml)
ENV_FILE=".env"
DATA_DIR="./langfuse-data"

# Secrets that must exist in .env before the stack can start.
REQUIRED_VARS=(
  LANGFUSE_POSTGRES_PASSWORD
  LANGFUSE_CLICKHOUSE_PASSWORD
  LANGFUSE_REDIS_PASSWORD
  LANGFUSE_MINIO_PASSWORD
  LANGFUSE_SALT
  LANGFUSE_ENCRYPTION_KEY
  LANGFUSE_NEXTAUTH_SECRET
  LANGFUSE_INIT_USER_PASSWORD
)

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 0. preflight ---------------------------------------------------------
say "Preflight"

# docker compose plugin (not legacy docker-compose)
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose plugin not found." >&2
  exit 1
fi

# openssl for secret generation
if ! openssl rand -base64 16 >/dev/null 2>&1; then
  echo "ERROR: openssl not available." >&2
  exit 1
fi

# .env must exist (the main stack already requires HF_TOKEN, etc.)
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Create it with HF_TOKEN, SGLANG_API_KEY, etc. first." >&2
  exit 1
fi

# --- 1. generate missing secrets ------------------------------------------
say "Checking secrets in $ENV_FILE"

MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
  if ! grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
    MISSING+=("$var")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Missing: ${MISSING[*]}"
  echo "Generating and appending to $ENV_FILE ..."

  {
    echo ""
    echo "# --- Langfuse secrets (generated $(date -u +%Y-%m-%dT%H:%M:%SZ)) ---"
    echo "LANGFUSE_POSTGRES_PASSWORD=$(openssl rand -base64 32)"
    echo "LANGFUSE_CLICKHOUSE_PASSWORD=$(openssl rand -base64 32)"
    echo "LANGFUSE_REDIS_PASSWORD=$(openssl rand -base64 32)"
    echo "LANGFUSE_MINIO_PASSWORD=$(openssl rand -base64 32)"
    echo "LANGFUSE_SALT=$(openssl rand -base64 32)"
    echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)"
    echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -base64 32)"
    echo "LANGFUSE_INIT_USER_PASSWORD=$(openssl rand -base64 32)"
    echo ""
    echo "# --- Langfuse headless init ---"
    echo "LANGFUSE_INIT_ORG_NAME=Default"
    echo "LANGFUSE_INIT_PROJECT_NAME=qwen36-27b"
    echo "LANGFUSE_INIT_USER_EMAIL=admin@example.com"
    echo "LANGFUSE_INIT_USER_NAME=admin"
  } >> "$ENV_FILE"

  echo "Done."
else
  echo "All secrets present. Skipping."
fi

# --- 1b. trace-pipeline credentials ---------------------------------------
# Added after the initial Langfuse rollout, so these live in their own
# idempotent block: an install that already has the eight secrets above
# still needs these three.
#
# The OTel Collector authenticates to the Langfuse ingestion API with a
# PROJECT key pair, not the admin user. Seeding the pair here keeps the whole
# credential chain reproducible from .env instead of depending on someone
# clicking through Project Settings.
say "Checking trace-pipeline credentials in $ENV_FILE"

TRACE_VARS=(
  LANGFUSE_INIT_PROJECT_PUBLIC_KEY
  LANGFUSE_INIT_PROJECT_SECRET_KEY
  LANGFUSE_OTEL_AUTH
)

TRACE_MISSING=()
for var in "${TRACE_VARS[@]}"; do
  if ! grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
    TRACE_MISSING+=("$var")
  fi
done

if [ ${#TRACE_MISSING[@]} -gt 0 ]; then
  echo "Missing: ${TRACE_MISSING[*]}"

  # LANGFUSE_INIT_PROJECT_* are honoured on FIRST boot only. If Postgres
  # already holds a project, the seeded pair is inert and the collector will
  # 401 until the real keys from the UI are pasted in.
  if [ -d "$DATA_DIR/postgres" ] && [ -n "$(ls -A "$DATA_DIR/postgres" 2>/dev/null)" ]; then
    cat >&2 <<'WARN'

  !! Langfuse is already initialised — its project exists and keeps the keys
     it was created with. The pair generated below will NOT take effect.

     Create a key pair in the UI (Project Settings -> API Keys), then replace
     the three generated values in .env:

       LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-...
       LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-...
       LANGFUSE_OTEL_AUTH=$(printf '%s:%s' "$PUBLIC_KEY" "$SECRET_KEY" | base64 -w0)

     Until then otel-collector will log 401s and drop spans; the inference
     tier is unaffected.

WARN
  fi

  gen_uuid() {
    if command -v uuidgen >/dev/null 2>&1; then
      uuidgen
    else
      cat /proc/sys/kernel/random/uuid
    fi
  }

  LF_PK="pk-lf-$(gen_uuid)"
  LF_SK="sk-lf-$(gen_uuid)"

  {
    echo ""
    echo "# --- Langfuse trace pipeline (generated $(date -u +%Y-%m-%dT%H:%M:%SZ)) ---"
    echo "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=${LF_PK}"
    echo "LANGFUSE_INIT_PROJECT_SECRET_KEY=${LF_SK}"
    echo "# Basic credential for the OTel Collector — base64(public:secret)."
    echo "# Regenerate after changing either key above."
    echo "LANGFUSE_OTEL_AUTH=$(printf '%s:%s' "$LF_PK" "$LF_SK" | base64 -w0)"
  } >> "$ENV_FILE"

  echo "Done."
else
  echo "All trace-pipeline credentials present. Skipping."
fi

# --- 2. create data directories -------------------------------------------
say "Creating data directories under $DATA_DIR"

for sub in postgres redis clickhouse clickhouse-logs minio; do
  mkdir -p "$DATA_DIR/$sub"
done
echo "OK"

# --- 3. pull images -------------------------------------------------------
say "Pulling Langfuse images (first run only, may take a minute)"

# Only pull the Langfuse-specific services to avoid re-pulling the heavy
# SGLang image. The 6 services:
docker compose "${COMPOSE_FILES[@]}" pull \
  langfuse-postgres \
  langfuse-redis \
  langfuse-clickhouse \
  langfuse-minio \
  langfuse-web \
  langfuse-worker \
  otel-collector

# --- 4. start the stack ---------------------------------------------------
say "Starting full stack (inference + metrics + langfuse)"

docker compose "${COMPOSE_FILES[@]}" up -d

# --- 5. wait for langfuse-web to become healthy ----------------------------
# langfuse-web depends on all 4 infra services (service_healthy), so by the
# time it starts they are ready. We only need to wait for the web app itself.
say "Waiting for langfuse-web to be ready (up to 120s)"

t0=$(date +%s)
while :; do
  st=$(docker inspect -f '{{.State.Health.Status}}' qwen36-27b-langfuse-web 2>/dev/null || echo "missing")
  el=$(( $(date +%s) - t0 ))
  # langfuse-web doesn't have a healthcheck in the upstream compose; check
  # container state instead.
  status=$(docker inspect -f '{{.State.Status}}' qwen36-27b-langfuse-web 2>/dev/null || echo "missing")
  if [ "$status" = "running" ]; then
    echo "running after ${el}s"
    break
  fi
  if [ "$el" -gt 120 ]; then
    echo "ERROR: langfuse-web did not start within 120s (state: $status)." >&2
    echo "       Check with: docker logs qwen36-27b-langfuse-web" >&2
    exit 1
  fi
  printf '  t=%ss status=%s\r' "$el" "$status"
  sleep 3
done

# --- 6. verify all 7 services are running ---------------------------------
say "Service status"

LANGFUSE_SVCS=(
  qwen36-27b-langfuse-postgres
  qwen36-27b-langfuse-redis
  qwen36-27b-langfuse-clickhouse
  qwen36-27b-langfuse-minio
  qwen36-27b-langfuse-web
  qwen36-27b-langfuse-worker
  qwen36-27b-otel-collector
)

ALL_OK=true
for svc in "${LANGFUSE_SVCS[@]}"; do
  state=$(docker inspect -f '{{.State.Status}}' "$svc" 2>/dev/null || echo "missing")
  case "$state" in
    running) printf '  \033[32m%s\033[0m  %s\n' "$state" "$svc" ;;
    *) printf '  \033[31m%s\033[0m  %s\n' "$state" "$svc"; ALL_OK=false ;;
  esac
done

if [ "$ALL_OK" != "true" ]; then
  echo "" >&2
  echo "ERROR: not all services are running. Check logs:" >&2
  for svc in "${LANGFUSE_SVCS[@]}"; do
    echo "  docker logs $svc" >&2
  done
  exit 1
fi

# --- 7. print access instructions -----------------------------------------
say "Langfuse is ready"
echo ""
echo "SSH tunnel:"
echo "  ssh -L 3001:127.0.0.1:3001 <host>"
echo ""
echo "UI: http://localhost:3001"
echo ""
echo "Headless init credentials:"
echo "  email:    $(grep '^LANGFUSE_INIT_USER_EMAIL=' "$ENV_FILE" | cut -d= -f2)"
echo "  password: (see LANGFUSE_INIT_USER_PASSWORD in $ENV_FILE)"
echo ""
echo "To send traces from your workstation, install the Langfuse SDK:"
echo "  pip install langfuse"
echo ""
echo "  import os"
echo "  from langfuse import Langfuse"
echo ""
echo "  langfuse = Langfuse("
echo "      host=\"http://localhost:3001\","
echo "      public_key=\"<project-public-key>\","
echo "      secret_key=\"<project-secret-key>\","
echo "  )"
echo ""
echo "Project keys are in .env (LANGFUSE_INIT_PROJECT_*); on an install that"
echo "predates them, take the pair from the UI under Project Settings."
echo ""
echo "Engine traces reach Langfuse without any SDK — SGLang exports OTLP to"
echo "otel-collector, which forwards to the ingestion API. Verify with:"
echo "  docker logs qwen36-27b-otel-collector | grep -i 'traces\\|401'"
