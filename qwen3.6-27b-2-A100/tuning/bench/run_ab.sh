#!/bin/bash
# =============================================================================
# A/B driver — concurrency ladder against ONE replica, direct (no router).
#
# Follows the same containerised pattern as benchmarks/run_worker.sh: the
# worker and DCGM hostnames only resolve on the qwen36-27b-backend network, and
# no worker port is published to the host, so the benchmark must run INSIDE
# that network rather than from the host shell.
#
# Usage:
#   ./tuning/bench/run_ab.sh <r0|r1> <label> [extra worker_ladder.py args...]
#
# Typical campaign:
#   ./tuning/bench/run_ab.sh r0 control_pre                 # baseline, before anything
#   ./tuning/bench/run_ab.sh r1 treatment_pre               # r1 baseline, before its roll
#   # ... roll r1 only ...
#   ./tuning/bench/run_ab.sh r1 treatment_post
#   ./tuning/bench/run_ab.sh r0 control_post                # confirm control did not drift
#
# The control_pre / control_post pair is not ceremony: it is how you detect
# that the host, thermals, or a neighbouring container moved under you between
# measurements. If control drifts, the treatment delta is not trustworthy.
# =============================================================================
set -euo pipefail

REPLICA="${1:?usage: run_ab.sh <r0|r1> <label> [args...]}"
LABEL="${2:?usage: run_ab.sh <r0|r1> <label> [args...]}"
shift 2

case "$REPLICA" in
  r0) HOST=qwen36-27b-r0; PORT=8001 ;;
  r1) HOST=qwen36-27b-r1; PORT=8002 ;;
  *)  echo "replica must be r0 or r1" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/../.."
source .env

OUT_DIR="tuning/results"
mkdir -p "$OUT_DIR"
OUT_FILE="${OUT_DIR}/ladder_${REPLICA}_${LABEL}.json"

docker run --rm \
  --network qwen36-27b-backend \
  -e SGLANG_API_KEY="${SGLANG_API_KEY}" \
  -v "$(pwd)/tuning/bench:/bench:ro" \
  -v "$(pwd)/${OUT_DIR}:/out" \
  python:3.12-slim \
  python3 /bench/worker_ladder.py \
    --host "$HOST" --port "$PORT" \
    --label "${REPLICA}_${LABEL}" \
    --out "/out/$(basename "$OUT_FILE")" \
    "$@"

echo
echo "== saved: ${OUT_FILE} =="
python3 tuning/bench/compare.py "$OUT_FILE"
