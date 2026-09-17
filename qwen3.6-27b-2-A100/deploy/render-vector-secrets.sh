#!/usr/bin/env bash
# =============================================================================
# Generate vector/secrets.json from .env.
#
# Vector reads the ClickHouse password through its `file` secrets backend rather
# than ${ENV} interpolation, because 0.57.0 disabled interpolation by default and
# a ${VAR} config now silently ships the literal string instead of failing.
# Upstream's own guidance is that secrets should not travel in the environment
# at all, since anything that can read /proc/<pid>/environ can read them.
#
# Re-run after rotating LANGFUSE_CLICKHOUSE_PASSWORD, then restart vector.
# The output is gitignored.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

pw=$(grep '^LANGFUSE_CLICKHOUSE_PASSWORD=' .env | cut -d= -f2- | tr -d '"')
[ -n "$pw" ] || { echo "LANGFUSE_CLICKHOUSE_PASSWORD not set in .env" >&2; exit 1; }

python3 - "$pw" <<'PY' > vector/secrets.json
import json, sys
json.dump({"clickhouse_password": sys.argv[1]}, sys.stdout)
PY

# Owner-only: the Vector container runs as root, so 600 is enough.
chmod 600 vector/secrets.json
echo "wrote vector/secrets.json"
