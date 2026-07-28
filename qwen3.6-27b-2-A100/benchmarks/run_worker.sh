#!/bin/bash
# Benchmarks a single replica DIRECTLY (bypassing the router).
# Usage: ./benchmarks/run_worker.sh <r0|r1> <label>
set -euo pipefail

REPLICA="${1:?usage: run_worker.sh <r0|r1> <label>}"
LABEL="${2:?usage: run_worker.sh <r0|r1> <label>}"

case "$REPLICA" in
  r0) HOST=qwen36-27b-r0; PORT=8001 ;;
  r1) HOST=qwen36-27b-r1; PORT=8002 ;;
  *)  echo "replica must be r0 or r1" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
source .env

OUT_DIR="benchmarks/results"
mkdir -p "$OUT_DIR"
OUT_FILE="${OUT_DIR}/worker_${LABEL}.json"

docker run --rm \
  --network qwen36-27b-backend \
  -e SGLANG_API_KEY="${SGLANG_API_KEY}" \
  -v "$(pwd)/benchmarks/bench_worker.py:/bench_worker.py:ro" \
  python:3.12-slim \
  python3 /bench_worker.py --host "$HOST" --port "$PORT" --label "$LABEL" > "$OUT_FILE"

echo "== saved: ${OUT_FILE} =="
python3 - "$OUT_FILE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
mb, ma = d.get("metrics_before", {}), d.get("metrics_after", {})
lp = d.get("long_prompt_stream", {})
print(f"  decode_tok_s_mean : {d.get('decode_tok_s_mean')}")
print(f"  long TTFT s       : {lp.get('ttft_s')}   (decode {lp.get('decode_tok_s')} tok/s)")
print(f"  accept_length     : {mb.get('spec_accept_length')} -> {ma.get('spec_accept_length')}")
print(f"  accept_rate       : {mb.get('spec_accept_rate')} -> {ma.get('spec_accept_rate')}")
gp = d.get("greedy_probe") or {}
print(f"  greedy finish     : {gp.get('finish_reason')}  ({gp.get('completion_tokens')} tok)")
print(f"  greedy content    : {json.dumps(gp.get('text'))[:200]}")
print(f"  greedy reasoning# : {len(gp.get('reasoning') or '')} chars")
PYEOF
