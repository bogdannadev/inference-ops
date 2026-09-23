#!/usr/bin/env bash
#
# p1_profile.sh <label> [--only decode,cold4k,...]
#
# Stage A of the P1 profile (tuning/lowlevel/P1_PROFILE.md): drained-r1 torch
# profiler captures, then the analyzer over each trace. Changes NO container
# config -- r1 keeps serving its production flags, it is only out of the router.
#
#   1. refuse unless r0 is healthy in the router (drain-replica.sh checks too)
#   2. drain r1 from the router, wait for its in-flight requests to finish
#   3. stop the DCGM exporter (CUPTI counter contention -> silent zeros)
#   4. run tuning/bench/p1_profile.py against r1 directly
#   5. ALWAYS (trap): start DCGM, re-register r1, verify 2/2 healthy
#   6. summarise every trace with p1_trace_analyze.py
#
# Output: tuning/results/p1_profile_r1_<label>.json (run record),
#         logs/r1/profile_p1/<label>/<scenario>/*.trace.json.gz (traces),
#         tuning/lowlevel/results/p1_<label>_<scenario>.{txt,json} (summaries).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

LABEL="${1:?usage: p1_profile.sh <label> [--only a,b,...]}"
shift
DCGM=qwen36-27b-dcgm
RES=tuning/lowlevel/results

clocks() { nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader; }

restored=0
restore() {
  [ "$restored" = 1 ] && return
  restored=1
  echo "== restore: DCGM + r1"
  docker start "$DCGM" >/dev/null || echo "WARN: could not start $DCGM -- start it by hand" >&2
  ./deploy/drain-replica.sh r1 restore
  ./deploy/drain-replica.sh r1 status
}
trap restore EXIT INT TERM

echo "== GPU 1 before: $(clocks)"
./deploy/drain-replica.sh r1 drain
docker stop "$DCGM" >/dev/null
echo "== DCGM stopped"

./tuning/bench/run_eval.sh r1 p1_profile "$LABEL" "$@"

restore
trap - EXIT INT TERM
echo "== GPU 1 after: $(clocks)"

# Analysis runs with r1 back in service: it only reads files.
for d in logs/r1/profile_p1/"$LABEL"/*/; do
  sc=$(basename "$d")
  f=$(ls -t "$d"*.trace.json.gz 2>/dev/null | head -1 || true)
  if [ -z "$f" ]; then echo "WARN: no trace for $sc" >&2; continue; fi
  python3 tuning/lowlevel/bench/p1_trace_analyze.py "$f" --top 30 \
    --json "$RES/p1_${LABEL}_${sc}.json" | tee "$RES/p1_${LABEL}_${sc}.txt"
done
