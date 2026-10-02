#!/usr/bin/env bash
#
# run_drill.sh <freeze|sched_hang|crash|slow> <prod|tuned> [out_dir]
#
# Router failure drill with NO GPU and NO contact with production: two fake
# workers (fake_worker.py), a throwaway router from the PRODUCTION router image
# with the production flags, and a load driver (driver.py), all on an internal
# docker network of their own. At t=INJECT_AT the fault hits fw1; at
# t=RECOVER_AT it is undone. The driver's JSON lines go to <out_dir>.
#
#   freeze      docker pause fw1          (host/process freeze: nothing answers)
#   sched_hang  fw1 mode sched_hang       (engine up, scheduler stuck)
#   crash       docker stop fw1           (process gone, connection refused)
#   slow        fw1 mode slow             (KV-stranded: healthy but 120 s queue)
#   slow503     fw1 mode slow503          (stranded + engine waiting timeout 30 s -> 503)
#
# prod  = the router flags in docker-compose.yml today
# tuned = prod + faster health detection (see STALLS.md)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCEN=${1:?scenario}; CONF=${2:?prod|tuned}
OUT=${3:-../../results/stall_drill}
INJECT_AT=${INJECT_AT:-120}; RECOVER_AT=${RECOVER_AT:-480}; DURATION=${DURATION:-720}
NET=stallrig; CPUS=48-55
ROUTER_IMAGE=$(docker inspect -f '{{.Config.Image}}' qwen36-27b-router)
mkdir -p "$OUT"; TAG="${SCEN}_${CONF}"

PROD_FLAGS=(--policy cache_aware --balance-abs-threshold 1 --request-timeout-secs 900
  --max-concurrent-requests 16 --queue-size 64 --queue-timeout-secs 300
  --retry-max-retries 1 --cb-failure-threshold 10 --cb-success-threshold 1)
TUNED_FLAGS=(--health-check-interval-secs 15 --health-failure-threshold 2 --health-check-timeout-secs 10)
FLAGS=("${PROD_FLAGS[@]}"); [ "$CONF" = tuned ] && FLAGS+=("${TUNED_FLAGS[@]}")

cleanup() { docker rm -f drl-fw0 drl-fw1 drl-rtr drl-drv >/dev/null 2>&1 || true; }
# fw1 reads its mode from /ctl inside its own container; flip it with docker exec
setmode() { docker exec drl-fw1 sh -c "mkdir -p /ctl && echo $1 > /ctl/fw1.mode"; }
trap cleanup EXIT
cleanup; trap cleanup EXIT
docker network inspect $NET >/dev/null 2>&1 || docker network create --internal $NET >/dev/null

for w in fw0 fw1; do
  docker run -d --name drl-$w --network $NET --network-alias $w --cpuset-cpus $CPUS \
    -e WORKER_NAME=$w -v "$PWD/fake_worker.py:/fw.py:ro"  \
    python:3.12-slim python3 -u /fw.py >/dev/null
done
docker run -d --name drl-rtr --network $NET --network-alias rtr --cpuset-cpus $CPUS \
  --entrypoint python3 "$ROUTER_IMAGE" -m sglang_router.launch_router \
  --host 0.0.0.0 --port 8000 --worker-urls http://fw0:8000 http://fw1:8000 "${FLAGS[@]}" >/dev/null

# wait for the router to answer through to a worker
for _ in $(seq 1 60); do
  docker run --rm --network $NET python:3.12-slim python3 -c "
import urllib.request,json,sys
r=urllib.request.Request('http://rtr:8000/v1/chat/completions',data=json.dumps({'model':'qwen36-27b','messages':[{'role':'user','content':'hi'}],'max_tokens':1}).encode(),headers={'Content-Type':'application/json'})
sys.exit(0 if urllib.request.urlopen(r,timeout=5).status==200 else 1)" >/dev/null 2>&1 && break
  sleep 2
done

docker run -d --name drl-drv --network $NET --cpuset-cpus $CPUS -v "$PWD/driver.py:/drv.py:ro" \
  python:3.12-slim python3 -u /drv.py --duration "$DURATION" >/dev/null
T0=$(date +%s)
echo "[$TAG] driver started; inject at ${INJECT_AT}s, recover at ${RECOVER_AT}s"

sleep "$INJECT_AT"
case $SCEN in
  freeze)     docker pause drl-fw1 >/dev/null ;;
  crash)      docker stop -t 0 drl-fw1 >/dev/null ;;
  sched_hang|slow|slow503) setmode "$SCEN" ;;
esac
echo "[$TAG] injected at $(( $(date +%s)-T0 ))s"

sleep $(( RECOVER_AT - INJECT_AT ))
case $SCEN in
  freeze)     docker unpause drl-fw1 >/dev/null ;;
  crash)      docker start drl-fw1 >/dev/null ;;
  sched_hang|slow|slow503) setmode ok ;;
esac
echo "[$TAG] recovered at $(( $(date +%s)-T0 ))s"

docker wait drl-drv >/dev/null
docker logs drl-drv 2>/dev/null > "$OUT/$TAG.jsonl"
docker logs drl-rtr 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' > "$OUT/$TAG.router.log"
echo "[$TAG] done: $(wc -l < "$OUT/$TAG.jsonl") requests -> $OUT/$TAG.jsonl"
