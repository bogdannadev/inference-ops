#!/usr/bin/env bash
#
# upgrade-v2.2.5.sh [--dry-run]
#
# Moves the live `higress` project from v2.2.4 to v2.2.5 without a moment in
# which the gateway references a plugin version the plugin-server cannot serve.
#
# WHY THE ORDER MATTERS. The v2.2.5 plugin-server ships the Go plugins as
# 2.0.3 ONLY (ai-proxy, ai-quota, ai-statistics, ai-token-ratelimit,
# request-validation, transformer); 2.0.1 is gone. A WasmPlugin whose module
# 404s FAILS OPEN (NOTES.md): key-auth would authenticate nobody and ai-quota
# would meter nothing. So:
#
#   1. new plugin-server, with the old 2.0.1 modules copied in -> serves both
#   2. wait until the router has no request in flight (a gateway restart cuts
#      every SSE stream), then controller + pilot + gateway + console on 2.2.5,
#      still loading 2.0.1
#   3. apply.sh with the 2.0.3 URLs; it blocks until every plugin has loaded
#   4. wait for quota-bot to re-sync ai-token-ratelimit (apply.sh resets it to
#      the disabled shell), then smoke-test through the live alias
#
# The 2.0.1 copies vanish on the next plugin-server recreate, by which time
# nothing references them.
#
# PROVEN FIRST on a parallel v2.2.5 stack (project higress-next, ~/higress-next,
# throwaway ledger higress-redis-test), 2026-10-14: 401 without/with a wrong
# key, 403 with no balance, streaming and non-streaming deducted exactly
# usage.total_tokens once (30/30/8317/115), a 43.5 KB non-streaming response
# through ai-quota 2.0.3's new response buffering, 422 at max_tokens 80000,
# 15 MB body 200, /v1/messages 200, consumer label reaches the engine and a
# spoofed X-Mse-Consumer is ignored, limiter 200/200/429/429 at 50 tokens/min
# with both windows counted under the unchanged Redis key format.
#
# ROLLBACK: backups land in ./backups/upgrade-<ts>/. Restore compose/.env and
# config/wasmplugins from there, then run the same three steps with the roles
# swapped (old plugin-server image + 2.0.3 copied in, then the 2.2.4 images,
# then apply.sh with the 2.0.1 URLs).

set -euo pipefail
cd "$(dirname "$0")"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

P=higress
COMPOSE_DIR=compose
INF=../qwen3.6-27b-2-A100
TS=$(date -u +%Y%m%dT%H%M%SZ)
BK=backups/upgrade-$TS
REG=higress-registry.cn-hangzhou.cr.aliyuncs.com/higress
GO_PLUGINS="ai-proxy ai-quota ai-statistics ai-token-ratelimit request-validation transformer"

declare -A PIN=(
  [HIGRESS_CONTROLLER_TAG]="2.2.5"
  [HIGRESS_PILOT_TAG]="2.2.5"
  [HIGRESS_GATEWAY_TAG]="2.2.5"
  [HIGRESS_CONSOLE_TAG]="2.2.5"
  [HIGRESS_PLUGIN_SERVER_TAG]="2.2.5"
  [HIGRESS_CONTROLLER_IMAGE]="$REG/higress:2.2.5@sha256:38663f158d1e9aa8233f268cc3bc4776aaaa5d5e1e15fdde88db0c7a626ef0f4"
  [HIGRESS_PILOT_IMAGE]="$REG/pilot:2.2.5@sha256:10334ca1baf0d8e1f8a97bb47845d6c12d118eda3d330c86a7707aa9f0628aa9"
  [HIGRESS_GATEWAY_IMAGE]="$REG/gateway:2.2.5@sha256:a82de493902012ce6b555ecb2244156f3c73824474d67549144a7efa334d0299"
  [HIGRESS_PLUGIN_SERVER_IMAGE]="$REG/plugin-server:2.2.5@sha256:ec63c6dfd18e77917faaecfd5fcdebde7e0bc40d42effa3b6d6f77d4f5c2fd40"
  [MCP_SERVER_WASM_URL]='http://plugin-server:8080/plugins/mcp-server/2.0.3/plugin.wasm'
)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi; }
dc()   { (cd "$COMPOSE_DIR" && COMPOSE_PROFILES=plugin-server docker compose -p "$P" "$@"); }

# --- 0. preflight -------------------------------------------------------------
say "Preflight"
grep -q "^HIGRESS_GATEWAY_TAG='2.2.4'" $COMPOSE_DIR/.env || die "compose/.env is not on 2.2.4; already upgraded?"
for k in HIGRESS_CONTROLLER_IMAGE HIGRESS_PILOT_IMAGE HIGRESS_GATEWAY_IMAGE HIGRESS_PLUGIN_SERVER_IMAGE; do
  docker image inspect "${PIN[$k]}" >/dev/null 2>&1 || die "image not pulled locally: ${PIN[$k]}"
done
docker image inspect "$REG/console:2.2.5" >/dev/null 2>&1 || die "console:2.2.5 not pulled"
# shellcheck disable=SC1091
ROUTER_CONTROL_PLANE_KEY=$(grep -E '^ROUTER_CONTROL_PLANE_KEY=' $INF/.env | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//')
[ -n "$ROUTER_CONTROL_PLANE_KEY" ] || die "ROUTER_CONTROL_PLANE_KEY missing in $INF/.env"
SMOKE_CRED=$(grep -E '^testafter\s' consumers.conf | cut -d' ' -f2-)
[ -n "$SMOKE_CRED" ] || die "testafter not in consumers.conf (smoke key)"
echo "ok"

# --- 1. backup + edit files -----------------------------------------------------
say "Backup to $BK, then edit compose/.env and config/wasmplugins"
run mkdir -p "$BK"
run cp -p $COMPOSE_DIR/.env "$BK/compose.env"
run cp -rp config/wasmplugins "$BK/wasmplugins"
run cp -p VERSION bin/configure.sh release-provenance.json "$BK/"
if [ "$DRY" = 0 ]; then
  for k in "${!PIN[@]}"; do
    grep -qE "^$k=" $COMPOSE_DIR/.env || die "$k not in compose/.env"
    v=${PIN[$k]}
    python3 - "$COMPOSE_DIR/.env" "$k" "$v" <<'EOF'
import re,sys
p,k,v=sys.argv[1:]
s=open(p).read()
s,n=re.subn(r'^'+re.escape(k)+r'=.*$', k+"='"+v+"'", s, flags=re.M)
assert n==1,(k,n)
open(p,'w').write(s)
EOF
  done
  for pl in $GO_PLUGINS; do
    sed -i "s#plugins/$pl/2\.0\.1/#plugins/$pl/2.0.3/#" config/wasmplugins/*.yaml
  done
  ! grep -h -oE "plugins/($(echo $GO_PLUGINS | tr ' ' '|'))/2\.0\.1" config/wasmplugins/*.yaml | grep -q . \
    || die "a 2.0.1 URL survived in config/wasmplugins"
  printf 'v2.2.5' > VERSION
  # Upstream release files; both were byte-identical to upstream v2.2.4 here.
  base=https://raw.githubusercontent.com/higress-group/higress-standalone/v2.2.5
  curl -fsS -m 30 "$base/bin/configure.sh" -o bin/configure.sh
  curl -fsS -m 30 "$base/release-provenance.json" -o release-provenance.json
  grep -q '"gatewayVersion": "2.2.5"' release-provenance.json || grep -q '2.2.5' release-provenance.json \
    || die "release-provenance.json does not mention 2.2.5"
fi
dc config -q 2>&1 | grep -v obsolete || true

# --- 2. plugin-server serving 2.0.1 AND 2.0.3 -----------------------------------
say "Plugin-server 2.2.5 with the 2.0.1 modules copied in"
OLD_PS=$(docker ps --filter "label=com.docker.compose.project=$P" --filter "label=com.docker.compose.service=plugin-server" -q)
[ -n "$OLD_PS" ] || die "no running plugin-server"
TMP=$(mktemp -d)
run docker cp "$OLD_PS:/usr/share/nginx/html/plugins/." "$TMP/plugins/"
run dc up -d --no-deps plugin-server
NEW_PS=$(docker ps --filter "label=com.docker.compose.project=$P" --filter "label=com.docker.compose.service=plugin-server" -q)
for pl in $GO_PLUGINS; do
  run docker cp "$TMP/plugins/$pl/2.0.1" "$NEW_PS:/usr/share/nginx/html/plugins/$pl/"
done
GW=${P}-gateway-1
if [ "$DRY" = 0 ]; then
  for pl in $GO_PLUGINS; do for v in 2.0.1 2.0.3; do
    c=$(docker exec "$GW" curl -s -o /dev/null -w '%{http_code}' "http://plugin-server:8080/plugins/$pl/$v/plugin.wasm")
    [ "$c" = 200 ] || die "plugin-server does not serve $pl/$v ($c) — gateway NOT touched; fix before continuing"
  done; done
  c=$(docker exec "$GW" curl -s -o /dev/null -w '%{http_code}' "http://plugin-server:8080/plugins/key-auth/2.0.0/plugin.wasm")
  [ "$c" = 200 ] || die "plugin-server does not serve key-auth/2.0.0"
  echo "serves 2.0.1 and 2.0.3 for all six, key-auth 2.0.0"
fi
rm -rf "$TMP"

# --- 3. wait for an idle router, then the 2.2.5 control plane + gateway ---------
say "Waiting for no request in flight at the router (a gateway restart cuts SSE streams)"
inflight() {
  docker exec qwen36-27b-router curl -s -m 8 -H "Authorization: Bearer $ROUTER_CONTROL_PLANE_KEY" \
    http://localhost:8000/workers | python3 -c "import json,sys; print(sum(int(w.get('load',0)) for w in json.load(sys.stdin).get('workers',[])))"
}
if [ "$DRY" = 0 ]; then
  idle=0; deadline=$((SECONDS + ${IDLE_TIMEOUT:-1800}))
  while [ $idle -lt 2 ]; do
    n=$(inflight || echo "?")
    if [ "$n" = 0 ]; then idle=$((idle+1)); else idle=0; fi
    [ $SECONDS -lt $deadline ] || die "router never idle for ${IDLE_TIMEOUT:-1800}s (last in-flight: $n). Plugin-server is already 2.2.5 and harmless; re-run steps 3-4 later."
    [ $idle -lt 2 ] && sleep 5
  done
  echo "idle"
fi
run dc up -d --no-deps controller pilot gateway console
if [ "$DRY" = 0 ]; then
  for i in $(seq 1 120); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' "$GW" 2>/dev/null)" = healthy ] && break; sleep 2
  done
  [ "$(docker inspect -f '{{.State.Health.Status}}' "$GW")" = healthy ] || die "gateway not healthy after 240s — ROLL BACK"
  docker inspect -f '{{.Config.Image}}' "$GW"
fi

# --- 4. plugin URLs -> 2.0.3 -----------------------------------------------------
say "apply.sh (2.0.3 URLs; blocks until every plugin reports update_success)"
run ./apply.sh

say "Waiting for quota-bot to re-sync ai-token-ratelimit (apply.sh resets it to the disabled shell)"
if [ "$DRY" = 0 ]; then
  for i in $(seq 1 24); do
    docker exec ${P}-apiserver-1 sh -c 'grep -q "configDisable: false" /opt/data/wasmplugins/ai-token-ratelimit.yaml' 2>/dev/null && { echo "limiter rules back"; break; }
    sleep 5
  done
  docker exec ${P}-apiserver-1 sh -c 'grep -q "configDisable: false" /opt/data/wasmplugins/ai-token-ratelimit.yaml' \
    || echo "WARNING: limiter still disabled after 120s — check quota-bot logs (daily/tpm not enforced meanwhile)"
fi

# --- 5. smoke through the live alias ---------------------------------------------
say "Smoke test through http://higress:80"
if [ "$DRY" = 0 ]; then
  B='{"model":"qwen36-27b","max_tokens":5,"reasoning_effort":"none","messages":[{"role":"user","content":"hi"}]}'
  c=$(docker run --rm --network edge curlimages/curl:8.10.1 -s -o /dev/null -w '%{http_code}' http://higress:80/v1/chat/completions -H 'content-type: application/json' -d "$B")
  [ "$c" = 401 ] || die "no-key request got $c, expected 401 — key-auth NOT enforcing: ROLL BACK NOW"
  c=$(docker run --rm --network edge curlimages/curl:8.10.1 -s -o /dev/null -w '%{http_code}' http://higress:80/v1/chat/completions -H "Authorization: $SMOKE_CRED" -H 'content-type: application/json' -d "$B")
  [ "$c" = 200 ] || die "testafter request got $c, expected 200"
  docker exec "$GW" curl -s 'localhost:15000/stats?filter=wasmplugin' | grep -E 'update_success|last_update_success' | grep -v ': 1$' | grep -E 'last_update_success: 0' && die "a plugin's last update failed"
  echo "401 without a key, 200 with testafter; all plugins loaded"
fi
say "Done: higress on v2.2.5. Backups in $BK"
