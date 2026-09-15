#!/usr/bin/env bash
#
# apply-clickhouse-retention.sh
#
# One-time companion to clickhouse/config.d/system-log-ttl.xml.
#
# ClickHouse reads a system log table's <ttl> only when it CREATES that table.
# Tables that already exist keep the definition they were born with, so
# dropping the config file in place bounds future installs while leaving the
# gigabyte already on disk to grow forever. This script issues the matching
# ALTER TABLE ... MODIFY TTL against the live server so the two agree.
#
# Idempotent: re-running sets the same TTLs and is a no-op on the data.
# Non-destructive: nothing is truncated. materialize_ttl_after_modify defaults
# to 1, so ClickHouse deletes now-expired partitions in background merges —
# reclaim is gradual, not instant, and only touches rows already older than
# the retention window.
#
# Talks to ClickHouse over 127.0.0.1:8123 with the password from .env. It runs
# no docker commands and does not touch the inference tier.
#
#   ./deploy/apply-clickhouse-retention.sh                 # set TTLs (safe)
#   ./deploy/apply-clickhouse-retention.sh --drop-renamed  # + reclaim orphans
#
# ABOUT --drop-renamed
#   When a system log table's definition in config.d stops matching the table
#   on disk — which is exactly what happens the first time system-log-ttl.xml
#   is introduced — ClickHouse does not migrate it. It renames the existing
#   table to <name>_0 and creates a fresh empty one from the new definition.
#   Observed here on that first restart:
#
#       system.trace_log_0                548 MiB   (26,979,474 rows)
#       system.text_log_0                 185 MiB
#       system.part_log_0                  75 MiB
#       system.metric_log_0                73 MiB
#       system.asynchronous_metric_log_0   72 MiB
#       ... 959 MiB across ten tables
#
#   Those orphans inherit no TTL and are referenced by nothing, so they are
#   frozen on disk forever. The rename is ClickHouse giving you a chance to
#   inspect before discarding; --drop-renamed is the discard, and it is opt-in
#   because DROP TABLE is not reversible.
#
#   This should be a one-time event. Once the tables match config.d, later
#   restarts reuse them and no further renames occur.
#
set -euo pipefail

DROP_RENAMED=0
for arg in "$@"; do
  case "$arg" in
    --drop-renamed) DROP_RENAMED=1 ;;
    *) echo "usage: $0 [--drop-renamed]" >&2; exit 2 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_FILE=".env"
CH_URL="http://127.0.0.1:8123/"
CH_USER="clickhouse"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# .env values may be quoted; compose strips quotes on interpolation but a
# naive `cut` does not, and a password with stray quotes fails auth in a way
# that looks identical to a wrong password.
env_get() {
  grep "^${1}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- \
    | sed -E 's/^["'"'"']//; s/["'"'"']$//'
}

[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found." >&2; exit 1; }

CH_PW="$(env_get LANGFUSE_CLICKHOUSE_PASSWORD)"
[ -n "$CH_PW" ] || { echo "ERROR: LANGFUSE_CLICKHOUSE_PASSWORD not set in $ENV_FILE" >&2; exit 1; }

ch() { curl -sS --fail-with-body "$CH_URL" -u "${CH_USER}:${CH_PW}" --data-binary "$1"; }

# --- 0. preflight ---------------------------------------------------------
say "Preflight"
if ! ch "SELECT 1" >/dev/null 2>&1; then
  echo "ERROR: cannot reach ClickHouse at $CH_URL as $CH_USER." >&2
  echo "       Is docker-compose.clickhouse.yml up? Is the password in $ENV_FILE current?" >&2
  exit 1
fi
echo "ClickHouse reachable: $(ch "SELECT version()")"

# --- 1. before ------------------------------------------------------------
say "System log tables before"
ch "SELECT table,
           formatReadableSize(sum(bytes_on_disk)) AS size,
           sum(rows) AS rows
      FROM system.parts
     WHERE active AND database = 'system'
     GROUP BY table
     ORDER BY sum(bytes_on_disk) DESC
     FORMAT PrettyCompactMonoBlock"

# --- 2. apply -------------------------------------------------------------
# Keep these tiers identical to clickhouse/config.d/system-log-ttl.xml.
# opentelemetry_span_log is listed separately: it has no event_date column and
# partitions on finish_date instead.
say "Applying TTLs"

TTL_3D=(
  trace_log
  text_log
  metric_log
  asynchronous_metric_log
  processors_profile_log
  background_schedule_pool_log
)
TTL_7D=(
  query_log
  query_views_log
  part_log
  error_log
  asynchronous_insert_log
)

exists() {
  [ "$(ch "SELECT count() FROM system.tables WHERE database='system' AND name='$1'")" = "1" ]
}

apply_ttl() {
  local tbl="$1" expr="$2"
  if ! exists "$tbl"; then
    printf '  %-32s skipped (table does not exist yet)\n' "$tbl"
    return
  fi
  ch "ALTER TABLE system.${tbl} MODIFY TTL ${expr}" >/dev/null
  printf '  %-32s TTL %s\n' "$tbl" "$expr"
}

for t in "${TTL_3D[@]}"; do apply_ttl "$t" "event_date + INTERVAL 3 DAY DELETE"; done
for t in "${TTL_7D[@]}"; do apply_ttl "$t" "event_date + INTERVAL 7 DAY DELETE"; done
apply_ttl opentelemetry_span_log "finish_date + INTERVAL 3 DAY DELETE"

# --- 3. verify ------------------------------------------------------------
say "TTL now set on"
ch "SELECT name,
           extract(engine_full, 'TTL [^S]*') AS ttl
      FROM system.tables
     WHERE database = 'system' AND name LIKE '%\_log'
     ORDER BY name
     FORMAT PrettyCompactMonoBlock"

# --- 4. renamed-aside orphans --------------------------------------------
# Tables matching system.<something>_log_<N> — see --drop-renamed in the header.
RENAMED_Q="SELECT name FROM system.tables
            WHERE database='system' AND match(name, '_log_[0-9]+\$')
            ORDER BY total_bytes DESC"

say "Renamed-aside tables (no TTL, referenced by nothing)"
ch "SELECT name, formatReadableSize(total_bytes) AS size, total_rows AS rows
      FROM system.tables
     WHERE database='system' AND match(name, '_log_[0-9]+\$')
     ORDER BY total_bytes DESC
     FORMAT PrettyCompactMonoBlock"

ORPHANS="$(ch "$RENAMED_Q" || true)"

if [ -z "$ORPHANS" ]; then
  echo "None. Nothing to reclaim."
elif [ "$DROP_RENAMED" -eq 1 ]; then
  say "Dropping renamed-aside tables"
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    ch "DROP TABLE IF EXISTS system.\`${t}\` SYNC" >/dev/null
    printf '  dropped system.%s\n' "$t"
  done <<< "$ORPHANS"
  say "system database now"
  ch "SELECT formatReadableSize(sum(bytes_on_disk)) FROM system.parts
       WHERE active AND database='system'"
else
  cat <<'EOF'

These hold ClickHouse's own diagnostics from before the retention change, are
attached to no TTL, and will never shrink. Re-run with --drop-renamed to
reclaim them. Not done automatically: DROP TABLE is irreversible.
EOF
fi

say "Done"
cat <<'EOF'
Expired partitions are removed by background merges, so `bytes_on_disk` falls
over the next minutes-to-hours rather than immediately. Watch it with:

  SELECT table, formatReadableSize(sum(bytes_on_disk))
    FROM system.parts WHERE active AND database='system'
   GROUP BY table ORDER BY 2 DESC;

Or, now that clickhouse/config.d/prometheus.xml is in place, from Grafana:
  ClickHouseAsyncMetrics_DiskUsed_default
EOF
