#!/usr/bin/env bash
# =============================================================================
# Per-consumer usage report.
#
# Two independent sources, and they answer different questions:
#
#   BALANCE  - Redis key chat_quota:<consumer>. Cumulative and durable; this is
#              the billing record. Decremented by ai-quota after each
#              completion, survives restarts (appendonly yes).
#
#   COUNTERS - Envoy stats on :15020 of the gateway container, emitted by the
#              ai-statistics plugin.
#              These are process-lifetime counters: they RESET TO ZERO when the
#              gateway container restarts. Use them for rates and latency, not
#              for invoicing.
#
# Prometheus scrapes the same :15020 endpoint, so Grafana inherits the same
# reset behaviour — an `increase()` window that spans a restart undercounts.
#
# Usage:  ./stats.sh            all consumers
#         ./stats.sh acme       one consumer
# =============================================================================
set -euo pipefail

ONLY="${1:-}"
METRICS=$(docker exec "${GATEWAY:-higress-gateway-1}" curl -s localhost:15020/stats/prometheus)

# metric_name{...ai_consumer="X"...} VALUE  ->  "X VALUE"
grab() {
  echo "$METRICS" | awk -v m="$1" '
    $0 ~ "^" m "\\{" {
      if (match($0, /ai_consumer="[^"]*"/)) {
        c = substr($0, RSTART+13, RLENGTH-14)
        print c, $NF
      }
    }'
}

declare -A IN OUT TOT REQ TTFT SVC
while read -r c v; do IN[$c]=$v;   done < <(grab route_upstream_model_consumer_metric_input_token)
while read -r c v; do OUT[$c]=$v;  done < <(grab route_upstream_model_consumer_metric_output_token)
while read -r c v; do TOT[$c]=$v;  done < <(grab route_upstream_model_consumer_metric_total_token)
while read -r c v; do REQ[$c]=$v;  done < <(grab route_upstream_model_consumer_metric_llm_duration_count)
while read -r c v; do TTFT[$c]=$v; done < <(grab route_upstream_model_consumer_metric_llm_first_token_duration)
while read -r c v; do SVC[$c]=$v;  done < <(grab route_upstream_model_consumer_metric_llm_service_duration)

printf '%-16s %14s %10s %12s %12s %12s %10s %10s\n' \
  CONSUMER BALANCE REQS INPUT OUTPUT TOTAL TTFT_ms DUR_ms

for key in $(docker exec higress-redis redis-cli --scan --pattern 'chat_quota:*' | sort); do
  c=${key#chat_quota:}
  [ -n "$ONLY" ] && [ "$c" != "$ONLY" ] && continue
  bal=$(docker exec higress-redis redis-cli GET "$key")
  r=${REQ[$c]:-0}
  mean() { [ "$r" -gt 0 ] 2>/dev/null && echo $(( ${1:-0} / r )) || echo -; }
  printf '%-16s %14s %10s %12s %12s %12s %10s %10s\n' \
    "$c" "$bal" "$r" "${IN[$c]:-0}" "${OUT[$c]:-0}" "${TOT[$c]:-0}" \
    "$(mean "${TTFT[$c]:-0}")" "$(mean "${SVC[$c]:-0}")"
done

echo
echo "balance = durable ledger (billing truth) | counters reset on gateway restart"
