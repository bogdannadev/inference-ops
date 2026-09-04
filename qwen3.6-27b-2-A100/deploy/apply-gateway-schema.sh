#!/usr/bin/env bash
# =============================================================================
# Create (or update) the gateway.requests fact table.
#
# Idempotent: CREATE DATABASE / TABLE IF NOT EXISTS. Safe to re-run.
#
# NOTE it does NOT migrate an existing table. ClickHouse ignores the body of
# CREATE TABLE IF NOT EXISTS when the table is already there, so a column added
# to the .sql file will NOT appear on a live table — you get no error and no
# column, and Vector then fails its inserts with "unknown field" because the
# sink runs with skip_unknown_fields: false. Add columns with an explicit
# ALTER TABLE ... ADD COLUMN, then update the .sql so a fresh install matches.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
set -a; . ./.env; set +a
: "${LANGFUSE_CLICKHOUSE_PASSWORD:?set LANGFUSE_CLICKHOUSE_PASSWORD in .env}"

docker exec -i qwen36-27b-langfuse-clickhouse \
  clickhouse-client --password "$LANGFUSE_CLICKHOUSE_PASSWORD" --multiquery \
  < clickhouse/gateway-requests.sql

docker exec qwen36-27b-langfuse-clickhouse \
  clickhouse-client --password "$LANGFUSE_CLICKHOUSE_PASSWORD" \
  -q "SELECT database, name, engine, total_rows FROM system.tables WHERE database='gateway'"
