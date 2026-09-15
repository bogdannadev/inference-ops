#!/usr/bin/env bash
# Runs the SQL regression tests for the exact usage aggregates against the
# live ClickHouse. Read-only: the tests select from literal VALUES.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
set -a; . ./.env; set +a
: "${LANGFUSE_CLICKHOUSE_PASSWORD:?set LANGFUSE_CLICKHOUSE_PASSWORD in .env}"

for t in clickhouse/*_test.sql; do
  docker exec -i qwen36-27b-langfuse-clickhouse \
    clickhouse-client --password "$LANGFUSE_CLICKHOUSE_PASSWORD" --multiquery < "$t" >/dev/null
  echo "ok  $t"
done
