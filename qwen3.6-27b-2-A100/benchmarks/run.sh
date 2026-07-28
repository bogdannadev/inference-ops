#!/bin/bash
# Runs benchmarks/bench.py inside a throwaway python container on the
# qwen36-27b-backend network, wraps it with GPU-memory + config-fingerprint
# metadata, and writes benchmarks/results/<label>.json
#
# Usage: ./benchmarks/run.sh <label> [concurrency]
set -euo pipefail

LABEL="${1:?usage: run.sh <label> [concurrency]}"
CONCURRENCY="${2:-6}"

cd "$(dirname "$0")/.."
source .env

OUT_DIR="benchmarks/results"
OUT_FILE="${OUT_DIR}/${LABEL}.json"
TMP_BENCH_JSON="$(mktemp)"
trap 'rm -f "$TMP_BENCH_JSON"' EXIT
mkdir -p "$OUT_DIR"

echo "== capturing pre-run GPU memory =="
GPU_BEFORE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader)

echo "== running benchmark (concurrency=${CONCURRENCY}) against qwen36-27b-router =="
docker run --rm \
  --network qwen36-27b-backend \
  -e SGLANG_API_KEY="${SGLANG_API_KEY}" \
  -v "$(pwd)/benchmarks/bench.py:/bench.py:ro" \
  python:3.12-slim \
  python3 /bench.py --concurrency "${CONCURRENCY}" > "$TMP_BENCH_JSON"

echo "== capturing post-run GPU memory =="
GPU_AFTER=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader)

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
R0_IMAGE=$(docker inspect --format='{{.Image}}' qwen36-27b-r0 2>/dev/null || echo "unknown")

GPU_BEFORE="$GPU_BEFORE" GPU_AFTER="$GPU_AFTER" LABEL="$LABEL" GIT_COMMIT="$GIT_COMMIT" \
GIT_DIRTY="$GIT_DIRTY" R0_IMAGE="$R0_IMAGE" BENCH_FILE="$TMP_BENCH_JSON" \
python3 - > "$OUT_FILE" <<'PYEOF'
import json, os, datetime

with open(os.environ["BENCH_FILE"]) as f:
    bench = json.load(f)

result = {
    "label": os.environ["LABEL"],
    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty_files": int(os.environ["GIT_DIRTY"]),
    "r0_image_id": os.environ["R0_IMAGE"],
    "gpu_memory_before": os.environ["GPU_BEFORE"].strip().splitlines(),
    "gpu_memory_after": os.environ["GPU_AFTER"].strip().splitlines(),
    "benchmark": bench,
}
print(json.dumps(result, indent=2))
PYEOF

echo
echo "== saved: ${OUT_FILE} =="
