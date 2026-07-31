#!/usr/bin/env bash
#
# roll-replica.sh <r0|r1> [--no-recreate]
#
# Safely roll ONE replica with no traffic sent to a dead worker and no router
# restart. Replaces the old "recreate the replica, then restart the router"
# procedure, which was both unnecessary and worse — see WHY below.
#
#   ./deploy/roll-replica.sh r1        # apply current compose config to r1
#   ./deploy/roll-replica.sh r0
#
# WHY THIS EXISTS (measured 2026-07-31b)
#
# The old rule said "always `docker compose restart qwen36-27b-router` after
# rolling a replica". Testing showed the restart is NOT required: the router
# tracks workers by URL and re-adds a returning worker on its own. But the
# naive roll has a real defect the restart never fixed:
#
#   t=0s     replica container stops
#   t=159s   router finally marks it unhealthy  <-- 159s of routing to a
#            (health-check-interval 60s x failure-threshold 3)   DEAD worker
#   t=200s   container healthy again
#   t=271s   router re-marks it healthy         <-- 71s of wasted capacity
#            (health-check-interval 60s x success-threshold 2)
#
# Deregistering FIRST removes the 159s window; re-registering explicitly
# removes the 71s one. Both use the router's REST API, verified working:
#   GET    /workers                       list, with id + is_healthy
#   DELETE /workers/<worker_id>           -> 202, stops routing immediately
#   POST   /workers {"url": "..."}        -> 202, healthy within seconds
#
# Restarting the router is strictly worse than this: it drops in-flight
# requests on BOTH replicas and resets routing state, to fix a problem that
# resolves itself.
#
# NOTE: --health-check-interval-secs is deliberately left at its 60s default.
# The worker /health endpoint is a GENERATION probe costing ~1.0s per call, so
# tightening the interval to shorten the lag would spend real GPU time. This
# script makes the lag irrelevant instead.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROUTER="qwen36-27b-router"
DRAIN_TIMEOUT=120     # seconds to wait for in-flight requests to finish
HEALTH_TIMEOUT=900    # seconds to wait for the replica to come back healthy

usage() { echo "usage: $0 <r0|r1> [--no-recreate]" >&2; exit 2; }

[ $# -ge 1 ] || usage
case "$1" in
  r0) SVC="qwen36-27b-r0"; URL="http://qwen36-27b-r0:8001"; PEER="r1" ;;
  r1) SVC="qwen36-27b-r1"; URL="http://qwen36-27b-r1:8002"; PEER="r0" ;;
  *)  usage ;;
esac
REPLICA="$1"; shift
RECREATE="--force-recreate"
[ "${1:-}" = "--no-recreate" ] && RECREATE=""

rcurl() { docker exec "$ROUTER" curl -s -m 8 "$@"; }

workers_json() { rcurl http://localhost:8000/workers; }

# field=id|healthy for a given replica tag, from the router's view
worker_field() {
  workers_json | python3 -c "
import json,sys
tag,field = sys.argv[1], sys.argv[2]
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
for w in d.get('workers',[]):
    if tag in w['url']:
        print(w['id'] if field=='id' else ('yes' if w['is_healthy'] else 'no'))
        sys.exit(0)
print('')" "$1" "$2"
}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- safety: never drain the last healthy worker ---------------------------
say "Preflight"
if ! workers_json >/dev/null 2>&1; then
  echo "ERROR: router $ROUTER is not reachable. Refusing to roll." >&2
  exit 1
fi
PEER_OK=$(worker_field "$PEER" healthy || true)
if [ "$PEER_OK" != "yes" ]; then
  echo "ERROR: peer replica $PEER is not healthy in the router (got '${PEER_OK:-absent}')." >&2
  echo "       Rolling $REPLICA now would drop the service to zero capacity." >&2
  exit 1
fi
echo "peer $PEER healthy — safe to roll $REPLICA"
workers_json | python3 -c "
import json,sys
for w in json.load(sys.stdin)['workers']:
    print(f\"  {w['url']:<32} healthy={w['is_healthy']} load={w['load']}\")"

# --- 1. drain -------------------------------------------------------------
say "Draining $REPLICA from the router"
WID=$(worker_field "$REPLICA" id || true)
if [ -n "$WID" ]; then
  code=$(rcurl -o /dev/null -w '%{http_code}' -X DELETE "http://localhost:8000/workers/$WID")
  echo "DELETE /workers/$WID -> $code"
  # let in-flight requests finish; new ones now go to the peer only
  for _ in $(seq 1 $((DRAIN_TIMEOUT/5))); do
    still=$(worker_field "$REPLICA" id || true)
    [ -z "$still" ] && break
    sleep 5
  done
else
  echo "not registered in the router; nothing to drain"
fi

# --- 2. roll --------------------------------------------------------------
# NEVER --remove-orphans: it would delete qwen3-emb, grafana, prometheus, dcgm.
# The orphan warning on every compose command here is expected; ignore it.
say "Recreating $SVC"
docker compose up -d --no-deps $RECREATE "$SVC"

# --- 3. wait for the container ------------------------------------------
say "Waiting for $SVC to report healthy (up to ${HEALTH_TIMEOUT}s)"
t0=$(date +%s)
while :; do
  st=$(docker inspect -f '{{.State.Health.Status}}' "$SVC" 2>/dev/null || echo missing)
  el=$(( $(date +%s) - t0 ))
  [ "$st" = "healthy" ] && { echo "healthy after ${el}s"; break; }
  if [ "$el" -gt "$HEALTH_TIMEOUT" ]; then
    echo "ERROR: $SVC did not become healthy within ${HEALTH_TIMEOUT}s (last: $st)." >&2
    echo "       It is NOT registered in the router, so traffic stays on $PEER." >&2
    echo "       Investigate with: docker logs $SVC" >&2
    exit 1
  fi
  printf '  t=%ss status=%s\r' "$el" "$st"
  sleep 10
done

# --- 4. re-register -------------------------------------------------------
say "Re-registering $REPLICA with the router"
code=$(rcurl -o /dev/null -w '%{http_code}' -X POST http://localhost:8000/workers \
        -H 'Content-Type: application/json' -d "{\"url\":\"$URL\"}")
echo "POST /workers -> $code"

for _ in $(seq 1 12); do
  [ "$(worker_field "$REPLICA" healthy || true)" = "yes" ] && break
  sleep 5
done

# --- 5. verify ------------------------------------------------------------
say "Final router view"
workers_json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for w in d['workers']:
    print(f\"  {w['url']:<32} healthy={w['is_healthy']} load={w['load']}\")
ok = d['total']==2 and all(w['is_healthy'] for w in d['workers'])
print('\nOK: both replicas registered and healthy' if ok
      else '\nWARNING: expected 2 healthy workers, got '
           f\"{sum(w['is_healthy'] for w in d['workers'])}/{d['total']}\")
sys.exit(0 if ok else 1)"
