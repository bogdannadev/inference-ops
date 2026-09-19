#!/usr/bin/env bash
#
# drain-replica.sh <r0|r1> <drain|restore|status>
#
# Take ONE replica out of the router for a measurement window and put it back,
# without touching any container. Same router REST calls as roll-replica.sh:
#   drain    DELETE /workers/<id>, then wait for its in-flight requests to end
#   restore  POST /workers {"url": ...}, then wait until the router marks it healthy
#   status   the router's view of both workers
#
# Every tuning result marked "drained" (tuning/docs/*) needs this: with
# --max-running-requests 4, production traffic changes batch composition and
# the replica's cache, so a benchmark on a registered worker is not comparable.
# Refuses to drain when the peer is not healthy (that would leave no capacity).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROUTER="qwen36-27b-router"
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-300}

usage() { echo "usage: $0 <r0|r1> <drain|restore|status>" >&2; exit 2; }
[ $# -eq 2 ] || usage
case "$1" in
  r0) SVC="qwen36-27b-r0"; URL="http://qwen36-27b-r0:8001"; PEER="r1" ;;
  r1) SVC="qwen36-27b-r1"; URL="http://qwen36-27b-r1:8002"; PEER="r0" ;;
  *)  usage ;;
esac
REPLICA="$1"; ACTION="$2"

source .env
: "${ROUTER_CONTROL_PLANE_KEY:?set ROUTER_CONTROL_PLANE_KEY in .env}"

rcurl() {
  docker exec "$ROUTER" curl -s -m 8 \
    -H "Authorization: Bearer ${ROUTER_CONTROL_PLANE_KEY}" "$@"
}

workers_json() { rcurl http://localhost:8000/workers; }

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

show() {
  workers_json | python3 -c "
import json,sys
for w in json.load(sys.stdin)['workers']:
    print(f\"  {w['url']:<32} healthy={w['is_healthy']} load={w['load']}\")"
}

# The worker's own count of running + queued requests (the router's view ends
# at DELETE; requests it already forwarded keep running on the worker).
worker_busy() {
  docker run --rm --network qwen36-27b-backend --cpuset-cpus 48-55 python:3.12-slim \
    python3 -c "
import re,urllib.request
t=urllib.request.urlopen('$URL/metrics',timeout=10).read().decode()
n=sum(float(m.group(1)) for m in re.finditer(r'^sglang:num_(?:running|queue)_reqs\{[^}]*\} (\S+)', t, re.M))
print(int(n))"
}

case "$ACTION" in
  status)
    show ;;
  drain)
    [ "$(worker_field "$PEER" healthy || true)" = "yes" ] || {
      echo "ERROR: peer $PEER is not healthy in the router; draining $REPLICA would leave no capacity." >&2
      exit 1; }
    WID=$(worker_field "$REPLICA" id || true)
    if [ -n "$WID" ]; then
      echo "DELETE /workers/$WID -> $(rcurl -o /dev/null -w '%{http_code}' -X DELETE "http://localhost:8000/workers/$WID")"
    else
      echo "$REPLICA is not registered in the router"
    fi
    t0=$(date +%s)
    while :; do
      busy=$(worker_busy)
      [ "$busy" = "0" ] && { echo "$SVC idle after $(( $(date +%s) - t0 ))s"; break; }
      if [ $(( $(date +%s) - t0 )) -gt "$DRAIN_TIMEOUT" ]; then
        echo "WARNING: $SVC still has $busy running/queued after ${DRAIN_TIMEOUT}s (it stays drained)." >&2
        exit 1
      fi
      printf '  %s running/queued\r' "$busy"; sleep 5
    done
    show ;;
  restore)
    if [ -z "$(worker_field "$REPLICA" id || true)" ]; then
      echo "POST /workers -> $(rcurl -o /dev/null -w '%{http_code}' -X POST http://localhost:8000/workers \
        -H 'Content-Type: application/json' -d "{\"url\":\"$URL\"}")"
    fi
    for _ in $(seq 1 12); do
      [ "$(worker_field "$REPLICA" healthy || true)" = "yes" ] && break
      sleep 5
    done
    show ;;
  *) usage ;;
esac
