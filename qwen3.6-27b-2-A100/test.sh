#!/bin/sh

set -u

NETWORK="${NETWORK:-qwen36-27b-a100-network}"
MODEL="${MODEL:-qwen36-27b}"

R0_HOST="${R0_HOST:-qwen36-27b-r0}"
R0_PORT="${R0_PORT:-8001}"

R1_HOST="${R1_HOST:-qwen36-27b-r1}"
R1_PORT="${R1_PORT:-8002}"

ROUTER_HOST="${ROUTER_HOST:-qwen36-27b-router}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

CURL_IMAGE="${CURL_IMAGE:-curlimages/curl:latest}"

PASS_COUNT=0
FAIL_COUNT=0

log() {
  printf '\n\033[1;34m%s\033[0m\n' "$*"
}

ok() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf '\033[1;32m[PASS]\033[0m %s\n' "$*"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33m[WARN]\033[0m %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required command: $1"
    exit 1
  fi
}

load_env() {
  if [ ! -f ".env" ]; then
    fail ".env file not found"
    exit 1
  fi

  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a

  if [ "${SGLANG_API_KEY:-}" = "" ]; then
    fail "SGLANG_API_KEY is not set in .env"
    exit 1
  fi

  ok ".env loaded and SGLANG_API_KEY is set"
}

dcurl() {
  docker run --rm \
    --network "$NETWORK" \
    "$CURL_IMAGE" \
    curl "$@"
}

api_curl() {
  docker run --rm \
    --network "$NETWORK" \
    -e SGLANG_API_KEY="$SGLANG_API_KEY" \
    "$CURL_IMAGE" \
    curl "$@"
}

http_request() {
  # Usage:
  #   http_request METHOD URL [extra curl args...]
  METHOD="$1"
  URL="$2"
  shift 2

  docker run --rm \
    --network "$NETWORK" \
    -e SGLANG_API_KEY="$SGLANG_API_KEY" \
    "$CURL_IMAGE" \
    curl -sS \
      -X "$METHOD" \
      -w '\n__HTTP_STATUS__:%{http_code}\n' \
      "$URL" \
      "$@"
}

extract_status() {
  printf '%s\n' "$1" | sed -n 's/^__HTTP_STATUS__://p' | tail -n 1
}

extract_body() {
  printf '%s\n' "$1" | sed '/^__HTTP_STATUS__:/d'
}

check_http_status() {
  NAME="$1"
  EXPECTED="$2"
  RESPONSE="$3"

  STATUS="$(extract_status "$RESPONSE")"

  if [ "$STATUS" = "$EXPECTED" ]; then
    ok "$NAME returned HTTP $EXPECTED"
  else
    fail "$NAME expected HTTP $EXPECTED but got HTTP ${STATUS:-unknown}"
    printf '%s\n' "$RESPONSE"
  fi
}

check_http_status_any() {
  NAME="$1"
  EXPECTED_LIST="$2"
  RESPONSE="$3"

  STATUS="$(extract_status "$RESPONSE")"

  for CODE in $EXPECTED_LIST; do
    if [ "$STATUS" = "$CODE" ]; then
      ok "$NAME returned expected HTTP $STATUS"
      return
    fi
  done

  fail "$NAME expected one of [$EXPECTED_LIST] but got HTTP ${STATUS:-unknown}"
  printf '%s\n' "$RESPONSE"
}

check_no_published_ports() {
  log "Checking that no host ports are published"

  for C in qwen36-27b-r0 qwen36-27b-r1 qwen36-27b-router; do
    PORTS="$(docker port "$C" 2>/dev/null || true)"
    if [ "$PORTS" = "" ]; then
      ok "$C has no published host ports"
    else
      fail "$C has published host ports:"
      printf '%s\n' "$PORTS"
    fi
  done
}

check_compose_ps() {
  log "docker compose ps"
  docker compose ps || fail "docker compose ps failed"
}

check_network_exists() {
  log "Checking Docker network"

  if docker network inspect "$NETWORK" >/dev/null 2>&1; then
    ok "Docker network exists: $NETWORK"
  else
    fail "Docker network does not exist: $NETWORK"
    exit 1
  fi
}

check_health() {
  log "Health checks"

  R0_RESP="$(http_request GET "http://${R0_HOST}:${R0_PORT}/health")"
  check_http_status "worker r0 /health" "200" "$R0_RESP"

  R1_RESP="$(http_request GET "http://${R1_HOST}:${R1_PORT}/health")"
  check_http_status "worker r1 /health" "200" "$R1_RESP"

  ROUTER_RESP="$(http_request GET "http://${ROUTER_HOST}:${ROUTER_PORT}/health")"
  check_http_status "router /health" "200" "$ROUTER_RESP"
}

check_model_info() {
  log "Model info checks"

  R0_RESP="$(http_request GET "http://${R0_HOST}:${R0_PORT}/model_info")"
  check_http_status "worker r0 /model_info" "200" "$R0_RESP"
  extract_body "$R0_RESP" | head -c 1000
  printf '\n'

  R1_RESP="$(http_request GET "http://${R1_HOST}:${R1_PORT}/model_info")"
  check_http_status "worker r1 /model_info" "200" "$R1_RESP"
  extract_body "$R1_RESP" | head -c 1000
  printf '\n'

  MODELS_RESP="$(http_request GET "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/models" \
    -H "Authorization: Bearer ${SGLANG_API_KEY}")"
  check_http_status "router /v1/models" "200" "$MODELS_RESP"
  BODY="$(extract_body "$MODELS_RESP")"

  if printf '%s\n' "$BODY" | grep -q "$MODEL"; then
    ok "router /v1/models contains model name: $MODEL"
  else
    warn "router /v1/models did not visibly contain $MODEL"
    printf '%s\n' "$BODY"
  fi
}

chat_request() {
  URL="$1"
  PROMPT="$2"
  MAX_TOKENS="${3:-32}"

  PAYLOAD='{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "user",
        "content": "'"$PROMPT"'"
      }
    ],
    "temperature": 0,
    "max_tokens": '"$MAX_TOKENS"'
  }'

  http_request POST "$URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${SGLANG_API_KEY}" \
    -d "$PAYLOAD"
}

check_direct_workers() {
  log "Direct worker inference tests"

  R0_RESP="$(chat_request "http://${R0_HOST}:${R0_PORT}/v1/chat/completions" "Reply with exactly: worker zero is healthy" 32)"
  check_http_status "worker r0 direct chat" "200" "$R0_RESP"
  extract_body "$R0_RESP" | head -c 1500
  printf '\n'

  R1_RESP="$(chat_request "http://${R1_HOST}:${R1_PORT}/v1/chat/completions" "Reply with exactly: worker one is healthy" 32)"
  check_http_status "worker r1 direct chat" "200" "$R1_RESP"
  extract_body "$R1_RESP" | head -c 1500
  printf '\n'
}

check_router_chat() {
  log "Router inference test"

  RESP="$(chat_request "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" "Reply with exactly this sentence: deployment test passed" 32)"
  check_http_status "router chat completion" "200" "$RESP"

  BODY="$(extract_body "$RESP")"
  printf '%s\n' "$BODY" | head -c 2000
  printf '\n'

  if printf '%s\n' "$BODY" | grep -qi "deployment test passed"; then
    ok "router response contains expected phrase"
  else
    warn "router response did not visibly contain expected phrase"
  fi
}

check_auth_negative() {
  log "Auth negative test"

  PAYLOAD='{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "user",
        "content": "test"
      }
    ],
    "max_tokens": 8
  }'

  RESP="$(docker run --rm \
    --network "$NETWORK" \
    "$CURL_IMAGE" \
    curl -sS \
      -X POST "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" \
      -w '\n__HTTP_STATUS__:%{http_code}\n' \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer wrong-key" \
      -d "$PAYLOAD")"

  check_http_status_any "router wrong API key" "401 403 404" "$RESP"
}

check_streaming() {
  log "Streaming test"

  PAYLOAD='{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "user",
        "content": "Count from 1 to 5."
      }
    ],
    "temperature": 0,
    "max_tokens": 64,
    "stream": true
  }'

  RESP="$(api_curl -N -sS \
    "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${SGLANG_API_KEY}" \
    -d "$PAYLOAD" | head -n 20)"

  printf '%s\n' "$RESP"

  if printf '%s\n' "$RESP" | grep -q "^data:"; then
    ok "streaming returned data chunks"
  else
    fail "streaming did not return visible data chunks"
  fi
}

check_long_prompt() {
  log "Long prompt / prefill test"

  LONG_TEXT=""
  I=1
  while [ "$I" -le 1200 ]; do
    LONG_TEXT="${LONG_TEXT}This is a legal RAG prefill stability test. "
    I=$((I + 1))
  done

  docker run --rm \
    --network "$NETWORK" \
    -e SGLANG_API_KEY="$SGLANG_API_KEY" \
    -e MODEL="$MODEL" \
    -e ROUTER_HOST="$ROUTER_HOST" \
    -e ROUTER_PORT="$ROUTER_PORT" \
    -e LONG_TEXT="$LONG_TEXT" \
    "$CURL_IMAGE" \
    sh -c '
      PAYLOAD=$(cat <<EOF
{
  "model": "'"$MODEL"'",
  "messages": [
    {
      "role": "user",
      "content": "'"$LONG_TEXT"'\n\nSummarize the above in one short sentence."
    }
  ],
  "temperature": 0,
  "max_tokens": 64
}
EOF
)
      curl -sS \
        -X POST "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" \
        -w "\n__HTTP_STATUS__:%{http_code}\n" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${SGLANG_API_KEY}" \
        -d "$PAYLOAD"
    ' > /tmp/qwen36_long_prompt.out 2>&1

  RESP="$(cat /tmp/qwen36_long_prompt.out)"
  check_http_status "long prompt router chat" "200" "$RESP"
  extract_body "$RESP" | head -c 1500
  printf '\n'
}

check_concurrent_router() {
  log "Concurrent router test: 8 requests"

  TMP_DIR="/tmp/qwen36-concurrent-$$"
  mkdir -p "$TMP_DIR"

  for I in 1 2 3 4 5 6 7 8; do
    (
      PAYLOAD='{
        "model": "'"$MODEL"'",
        "messages": [
          {
            "role": "user",
            "content": "Request '"$I"': reply with exactly: ok '"$I"'"
          }
        ],
        "temperature": 0,
        "max_tokens": 16
      }'

      docker run --rm \
        --network "$NETWORK" \
        -e SGLANG_API_KEY="$SGLANG_API_KEY" \
        "$CURL_IMAGE" \
        curl -sS \
          -X POST "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" \
          -w '\n__HTTP_STATUS__:%{http_code}\n' \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer ${SGLANG_API_KEY}" \
          -d "$PAYLOAD" \
        > "$TMP_DIR/req-${I}.out" 2>&1
    ) &
  done

  wait

  ALL_OK=1

  for I in 1 2 3 4 5 6 7 8; do
    RESP="$(cat "$TMP_DIR/req-${I}.out")"
    STATUS="$(extract_status "$RESP")"

    if [ "$STATUS" = "200" ]; then
      ok "concurrent request $I returned HTTP 200"
    else
      fail "concurrent request $I returned HTTP ${STATUS:-unknown}"
      printf '%s\n' "$RESP"
      ALL_OK=0
    fi
  done

  rm -rf "$TMP_DIR"

  if [ "$ALL_OK" = "1" ]; then
    ok "all concurrent router requests completed successfully"
  fi
}

check_tools_smoke() {
  log "Tool-call parser smoke test"

  PAYLOAD='{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Moscow? Use the tool if appropriate."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string"
              }
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128
  }'

  RESP="$(http_request POST "http://${ROUTER_HOST}:${ROUTER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${SGLANG_API_KEY}" \
    -d "$PAYLOAD")"

  check_http_status "router tool-call smoke test" "200" "$RESP"
  extract_body "$RESP" | head -c 2500
  printf '\n'
}

check_metrics() {
  log "Metrics checks"

  R0_METRICS="$(dcurl -fsS "http://${R0_HOST}:${R0_PORT}/metrics" 2>/dev/null || true)"
  if [ "$R0_METRICS" != "" ]; then
    ok "worker r0 /metrics is available"
    printf '%s\n' "$R0_METRICS" | grep -Ei 'sglang|token|cache|mamba|request|decode|prefill|spec' | head -n 80 || true
  else
    fail "worker r0 /metrics unavailable"
  fi

  R1_METRICS="$(dcurl -fsS "http://${R1_HOST}:${R1_PORT}/metrics" 2>/dev/null || true)"
  if [ "$R1_METRICS" != "" ]; then
    ok "worker r1 /metrics is available"
    printf '%s\n' "$R1_METRICS" | grep -Ei 'sglang|token|cache|mamba|request|decode|prefill|spec' | head -n 80 || true
  else
    fail "worker r1 /metrics unavailable"
  fi

  ROUTER_METRICS="$(dcurl -fsS "http://${ROUTER_HOST}:${ROUTER_PORT}/metrics" 2>/dev/null || true)"
  if [ "$ROUTER_METRICS" != "" ]; then
    ok "router /metrics is available"
    printf '%s\n' "$ROUTER_METRICS" | head -n 80
  else
    warn "router /metrics unavailable or not exposed by this build"
  fi
}

check_logs_for_errors() {
  log "Checking recent logs for severe errors"

  BAD_PATTERN='exception|traceback|runtimeerror|sigquit|killed|oom|cuda error|out of memory'

  for SVC in qwen36-27b-r0 qwen36-27b-r1 qwen36-27b-router; do
    MATCHES="$(docker compose logs --tail=250 "$SVC" 2>/dev/null | grep -Ei "$BAD_PATTERN" || true)"

    if [ "$MATCHES" = "" ]; then
      ok "$SVC logs have no recent severe error patterns"
    else
      fail "$SVC logs contain severe error patterns"
      printf '%s\n' "$MATCHES"
    fi
  done
}

check_expected_warnings() {
  log "Checking known non-fatal warnings"

  MIXED_CHUNK_WARN="$(docker compose logs --tail=300 qwen36-27b-r0 qwen36-27b-r1 2>/dev/null | grep -F 'The mixed chunked prefill are disabled because of using ngram speculative decoding.' || true)"

  if [ "$MIXED_CHUNK_WARN" != "" ]; then
    warn "NGRAM disables mixed chunked prefill. This is expected if --enable-mixed-chunk is still present."
    printf '%s\n' "$MIXED_CHUNK_WARN" | head -n 4
  else
    ok "No mixed-chunk/NGRAM warning found in recent logs"
  fi

  PREFILL_CG_WARN="$(docker compose logs --tail=300 qwen36-27b-r0 qwen36-27b-r1 2>/dev/null | grep -F 'Disable prefill CUDA graph' || true)"

  if [ "$PREFILL_CG_WARN" != "" ]; then
    warn "Prefill CUDA graph disabled. This is usually informational; decode CUDA graph can still be enabled."
    printf '%s\n' "$PREFILL_CG_WARN" | head -n 4
  else
    ok "No prefill CUDA graph warning found in recent logs"
  fi
}

check_gpu_placement() {
  log "GPU placement sanity check"

  if docker exec qwen36-27b-r0 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >/tmp/qwen36-r0-gpu.out 2>&1; then
    ok "nvidia-smi works inside r0"
    cat /tmp/qwen36-r0-gpu.out
  else
    warn "nvidia-smi failed inside r0"
    cat /tmp/qwen36-r0-gpu.out
  fi

  if docker exec qwen36-27b-r1 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv >/tmp/qwen36-r1-gpu.out 2>&1; then
    ok "nvidia-smi works inside r1"
    cat /tmp/qwen36-r1-gpu.out
  else
    warn "nvidia-smi failed inside r1"
    cat /tmp/qwen36-r1-gpu.out
  fi

  log "Host nvidia-smi summary"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null || warn "host nvidia-smi failed"
}

main() {
  log "Qwen3.6-27B SGLang deployment test suite"

  require_cmd docker
  load_env
  check_network_exists

  check_compose_ps
  check_no_published_ports

  check_health
  check_model_info

  check_direct_workers
  check_router_chat
  check_auth_negative
  check_streaming
  check_long_prompt
  check_concurrent_router
  check_tools_smoke

  check_metrics
  check_expected_warnings
  check_logs_for_errors
  check_gpu_placement

  log "Summary"
  printf 'Passed: %s\n' "$PASS_COUNT"
  printf 'Failed: %s\n' "$FAIL_COUNT"

  if [ "$FAIL_COUNT" -eq 0 ]; then
    printf '\n\033[1;32mAll critical tests passed.\033[0m\n'
    exit 0
  else
    printf '\n\033[1;31mSome tests failed. Review output above.\033[0m\n'
    exit 1
  fi
}

main "$@"
