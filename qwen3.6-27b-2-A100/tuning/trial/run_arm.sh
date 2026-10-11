#!/bin/bash
# Roll r1 into one trial arm (left OUT of the router) and run the FP8-weights
# gates against it. From qwen3.6-27b-2-A100/:
#
#   EVAL_CORPUS=<sglang tree> ./tuning/trial/run_arm.sh <arm> [gate...]
#
# gates (default: all): smoke kvq kvqshort spec seq16 mem
#   (not in the default set) replay4 / replay8: cache_replay with 4 / 8 agents;
#   they clear L3, which is safe ONLY because make_arm gives every arm its own dir
#   seq16  spec_eval correctness again at --max-tokens 16000: at the default 8000
#          the 300-term `seq` task with thinking on mostly ends "truncated" (the
#          thinking runs out of room), which says nothing about exact terms.
# SKIP_ROLL=1 runs the gates against an r1 already booted into <arm> (it waits
# for healthy, but neither recreates nor touches the router).
# The arm file is tuning/trial/<arm>.yml from make_arm.py. Results land in
# tuning/results/<script>_r1_<arm>.json, the boot summary in
# tuning/results/boot_r1_<arm>.log. Restore production afterwards with
#   ./deploy/roll-replica.sh r1
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM="${1:?usage: run_arm.sh <arm> [gate...]}"; shift
GATES=("$@"); [ ${#GATES[@]} -eq 0 ] && GATES=(smoke kvq kvqshort spec seq16 mem)
[ -f "tuning/trial/$ARM.yml" ] || { echo "no tuning/trial/$ARM.yml" >&2; exit 2; }

say() { printf '\n==> [%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

if [ "${SKIP_ROLL:-0}" = "1" ]; then
  say "waiting for r1, already booting into $ARM"
  until [ "$(docker inspect -f '{{.State.Health.Status}}' qwen36-27b-r1)" = healthy ]; do sleep 10; done
else
  say "rolling r1 into $ARM (drained)"
  COMPOSE_FILE="docker-compose.yml:tuning/trial/$ARM.yml" NO_REGISTER=1 \
    HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800} ./deploy/roll-replica.sh r1
fi

docker logs qwen36-27b-r1 2>&1 | grep -E 'sitecustomize|KV Cache is allocated|Mamba Cache is allocated|max_total_num_tokens|host pool|host memory|[Mm]arlin|quantiz|Load weight end' \
  | grep -v 'server_args=' > "tuning/results/boot_r1_$ARM.log" || true
cat "tuning/results/boot_r1_$ARM.log"

for g in "${GATES[@]}"; do
  say "gate $g"
  case "$g" in
    smoke)    ./tuning/bench/run_eval.sh r1 smoke "$ARM" ;;
    kvq)      ./tuning/bench/run_eval.sh r1 kvq_eval "$ARM" ;;
    kvqshort) ./tuning/bench/run_eval.sh r1 kvq_eval "short_$ARM" --short-docs 16 ;;
    spec)     ./tuning/bench/run_eval.sh r1 spec_eval "$ARM" ;;
    seq16)    ./tuning/bench/run_eval.sh r1 spec_eval "seq16_$ARM" --what correctness --max-tokens 16000 ;;
    replay4)  ./tuning/bench/run_eval.sh r1 cache_replay "a4_$ARM" --clear-l3 ;;
    replay8)  ./tuning/bench/run_eval.sh r1 cache_replay "a8_$ARM" --agents 8 --clear-l3 ;;
    mem)      ./tuning/bench/run_eval.sh r1 mem_stress "$ARM" --side-tokens 60000
              docker logs --since 30m qwen36-27b-r1 2>&1 | grep -cE 'OutOfMemoryError|Prefill out of memory' || true ;;
    *)        echo "unknown gate $g" >&2; exit 2 ;;
  esac
done
say "$ARM done"
