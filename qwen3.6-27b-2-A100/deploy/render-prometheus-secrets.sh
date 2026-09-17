#!/usr/bin/env bash
# =============================================================================
# Write the credential files Prometheus reads (password_file in
# prometheus/prometheus.yml) from .env. prometheus/secrets/ is gitignored; the
# directory is inside the ./prometheus bind mount, so no compose change.
#
#   engine_metrics_clickhouse.pass  -> job engine-usage (ClickHouse user engine_metrics)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${ENGINE_METRICS_CLICKHOUSE_PASSWORD:?set ENGINE_METRICS_CLICKHOUSE_PASSWORD in .env}"
install -d prometheus/secrets
umask 077
printf '%s' "$ENGINE_METRICS_CLICKHOUSE_PASSWORD" > prometheus/secrets/engine_metrics_clickhouse.pass
# Prometheus runs as nobody.
./deploy/grant-nobody-read.sh prometheus/secrets prometheus/secrets/engine_metrics_clickhouse.pass
echo "wrote prometheus/secrets/engine_metrics_clickhouse.pass"
