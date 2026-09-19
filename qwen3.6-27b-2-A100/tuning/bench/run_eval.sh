#!/bin/bash
# Run an evalkit script against ONE worker, direct, from inside the backend
# network (same containerised pattern as run_ab.sh).
#
#   ./tuning/bench/run_eval.sh <r0|r1> <hicache_probe|spec_eval> <label> [script args...]
#
# Both scripts assume the worker is DRAINED from the router: hicache_probe
# flushes the cache and deliberately evicts, spec_eval saturates the admission
# cap. Output: tuning/results/<script>_<replica>_<label>.json
set -euo pipefail

REPLICA="${1:?usage: run_eval.sh <r0|r1> <script> <label> [args...]}"
SCRIPT="${2:?}"
LABEL="${3:?}"
shift 3

case "$REPLICA" in
  r0) HOST=qwen36-27b-r0; PORT=8001 ;;
  r1) HOST=qwen36-27b-r1; PORT=8002 ;;
  *)  echo "replica must be r0 or r1" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."
set -a; source .env; set +a   # exported: docker run -e NAME copies it, keeping the key off argv
OUT="tuning/results/${SCRIPT}_${REPLICA}_${LABEL}.json"

# EVAL_CORPUS: optional host directory mounted read-only at /corpus (kvq_eval).
CORPUS_MOUNT=()
[ -n "${EVAL_CORPUS:-}" ] && CORPUS_MOUNT=(-v "${EVAL_CORPUS}:/corpus:ro")

docker run --rm --cpuset-cpus 48-55 \
  --network qwen36-27b-backend \
  "${CORPUS_MOUNT[@]}" \
  -e SGLANG_API_KEY \
  -v "$(pwd)/tuning/bench:/bench:ro" \
  python:3.12-slim \
  python3 -u "/bench/${SCRIPT}.py" --host "$HOST" --port "$PORT" --label "$LABEL" "$@" > "$OUT"

echo "== saved: $OUT =="
