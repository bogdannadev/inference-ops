#!/usr/bin/env bash
#
# Render the shared ./config tree into this deployment's config store.
#
# This is the compose-mode counterpart of ../higress/apply.sh. The official
# Higress standalone distribution has NO command that applies configuration
# resources — hgctl included; its "resource management" is documentation of a
# directory layout, not a tool. So the apply step stays ours. What changes
# versus the all-in-one is only the destination:
#
#   all-in-one   one container, objects under /data
#   compose      the apiserver owns the store, objects under /opt/data
#
# The apiserver runs `--storage file --file-root-dir /opt/data` and WATCHES
# that tree, so an object written there goes live on the controller's own
# resync (~6s) with no restart. Same behaviour that was measured against the
# all-in-one.
#
# SOURCE OF TRUTH IS SHARED, DELIBERATELY. While both deployments exist, the
# config and the consumer credentials live in ../higress and are read from
# here. Copying them would mean two files holding the same customer API keys,
# drifting apart — the one failure mode worth avoiding above tidiness. At
# cutover, git mv them into this directory and drop the ../higress paths.
#
#   ./apply.sh            render, install
#   ./apply.sh --dry-run  render only, print what would be installed
#   ./apply.sh --restart  also restart apiserver + controller (configmaps only)
#
set -euo pipefail
cd "$(dirname -- "$0")"

DRY_RUN=0
RESTART=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --restart) RESTART=1 ;;
    *) echo "unknown flag: $a (use --dry-run or --restart)" >&2; exit 1 ;;
  esac
done

RENDERED=./rendered
UPSTREAM=../higress            # shared config + consumer table, see above

# --- inputs -----------------------------------------------------------------

[ -f .env ] || { echo "missing .env (copy .env.example)" >&2; exit 1; }
[ -f "$UPSTREAM/consumers.conf" ] || { echo "missing $UPSTREAM/consumers.conf" >&2; exit 1; }

# .env values in this stack are double-quoted. compose strips those quotes for
# us; every other reader has to do it itself, and a credential carrying literal
# quote characters authenticates nobody.
unquote() { sed -e 's/^"//' -e 's/"$//'; }
readvar() { grep -E "^$1=" .env | cut -d= -f2- | unquote; }

COMPOSE_PROJECT=$(readvar COMPOSE_PROJECT)
QUOTA_ADMIN_CONSUMER=$(readvar QUOTA_ADMIN_CONSUMER)
SGLANG_ENV_FILE=$(readvar SGLANG_ENV_FILE)
QUOTA_REDIS_DOMAIN=$(readvar QUOTA_REDIS_DOMAIN)
WASM_PLUGIN_BASE=$(readvar WASM_PLUGIN_BASE)

for v in COMPOSE_PROJECT QUOTA_ADMIN_CONSUMER SGLANG_ENV_FILE QUOTA_REDIS_DOMAIN WASM_PLUGIN_BASE; do
  [ -n "${!v}" ] || { echo "$v is unset in .env" >&2; exit 1; }
done

# The apiserver is the only container that can write the config store: ../conf
# is bind-mounted there, and its subdirectories are root-owned 0700, so a host
# copy running as an ordinary user cannot touch them.
CONTAINER="${COMPOSE_PROJECT}-apiserver-1"
DATA_ROOT=/opt/data

[ -f "$SGLANG_ENV_FILE" ] || { echo "SGLANG_ENV_FILE not found: $SGLANG_ENV_FILE" >&2; exit 1; }
SGLANG_API_KEY=$(grep -E '^SGLANG_API_KEY=' "$SGLANG_ENV_FILE" | cut -d= -f2- | unquote)
[ -n "$SGLANG_API_KEY" ] || { echo "SGLANG_API_KEY empty in $SGLANG_ENV_FILE" >&2; exit 1; }

# --- guardrail: never bill real consumers from a validation deployment ------
#
# ai-quota decrements the ledger on every completion. A non-production project
# pointed at the production Redis spends real customer balances on the first
# test request, silently and irreversibly. Refuse the combination outright.
if [ "$COMPOSE_PROJECT" != "higress" ] && [ "$QUOTA_REDIS_DOMAIN" = "higress-redis.higressint" ]; then
  echo "REFUSING: project '$COMPOSE_PROJECT' is not the production project, but" >&2
  echo "QUOTA_REDIS_DOMAIN points at the production ledger. Test traffic would" >&2
  echo "spend real consumer quota. Point it at a throwaway Redis." >&2
  exit 1
fi

# --- consumer table ---------------------------------------------------------
# Emitted as YAML rather than templated, because the list is variable length.

read_consumers() { grep -vE '^\s*(#|$)' "$UPSTREAM/consumers.conf"; }

CONSUMER_COUNT=$(read_consumers | wc -l)
[ "$CONSUMER_COUNT" -gt 0 ] || { echo "consumers.conf has no entries" >&2; exit 1; }

read_consumers | awk '{print $1}' | sort | uniq -d | grep . && {
  echo "duplicate consumer names in consumers.conf" >&2; exit 1; }

if ! read_consumers | awk '{print $1}' | grep -qx "$QUOTA_ADMIN_CONSUMER"; then
  echo "QUOTA_ADMIN_CONSUMER=$QUOTA_ADMIN_CONSUMER is not a name in consumers.conf" >&2
  exit 1
fi

emit_consumer_list() {   # indent
  local ind="$1"
  read_consumers | while read -r name credential; do
    printf '%s- name: %s\n%s  credential: "%s"\n' "$ind" "$name" "$ind" "$credential"
  done
}
emit_allow_list() {      # indent
  local ind="$1"
  read_consumers | awk -v i="$ind" '{printf "%s- %s\n", i, $1}'
}

# --- render -----------------------------------------------------------------

# Clear the CONTENTS, never the directory itself. ./rendered is a bind mount,
# and `rm -rf` + `mkdir` swaps its inode while the container still has the old
# one mounted — the copy below then silently reads an empty directory and
# installs nothing. Same trap as editing the Caddyfile out from under Caddy.
mkdir -p "$RENDERED"
find "$RENDERED" -mindepth 1 -delete
cp -r "$UPSTREAM/config/." "$RENDERED"/

export SGLANG_API_KEY QUOTA_ADMIN_CONSUMER QUOTA_REDIS_DOMAIN WASM_PLUGIN_BASE
while IFS= read -r -d '' f; do
  tmp=$(mktemp)
  # Only the named variables are substituted. A bare envsubst would eat any
  # $... that happens to appear in a comment or a regex.
  envsubst '${SGLANG_API_KEY} ${QUOTA_ADMIN_CONSUMER} ${QUOTA_REDIS_DOMAIN} ${WASM_PLUGIN_BASE}' <"$f" >"$tmp"
  mv "$tmp" "$f"
done < <(find "$RENDERED" -type f -name '*.yaml' -print0)

# Splice the consumer table into key-auth, if that stage is present.
KEYAUTH="$RENDERED/wasmplugins/key-auth.yaml"
if [ -f "$KEYAUTH" ]; then
  python3 - "$KEYAUTH" "$(emit_consumer_list '      ')" "$(emit_allow_list '          ')" <<'PY'
import sys
path, consumers, allow = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if '# __CONSUMERS__' not in s or '# __ALLOW__' not in s:
    sys.exit("key-auth.yaml is missing the __CONSUMERS__/__ALLOW__ markers")
s = s.replace('      # __CONSUMERS__', consumers.rstrip('\n'))
s = s.replace('          # __ALLOW__', allow.rstrip('\n'))
open(path, 'w').write(s)
PY
fi

echo "rendered $(find "$RENDERED" -name '*.yaml' | wc -l) object(s) from $UPSTREAM/config"
echo "  project      : $COMPOSE_PROJECT  (apiserver: $CONTAINER)"
echo "  quota ledger : $QUOTA_REDIS_DOMAIN"
echo "  plugin base  : $WASM_PLUGIN_BASE"
find "$RENDERED" -name '*.yaml' | sort | sed 's|^| |'

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "--dry-run: not installing. Secrets are present in $RENDERED (gitignored)."
  exit 0
fi

# --- install ----------------------------------------------------------------

docker inspect "$CONTAINER" >/dev/null 2>&1 || {
  echo "container '$CONTAINER' does not exist; start the stack first" >&2; exit 1; }

# Prune objects a previous apply installed that are no longer in ./config.
# Without this, deleting a file here leaves the object live forever — which for
# an Ingress means a route we thought we removed is still serving.
MANIFEST=$DATA_ROOT/.apply-manifest
PREV=$(docker exec "$CONTAINER" sh -c "cat $MANIFEST 2>/dev/null" || true)
NEXT=$(cd "$RENDERED" && find . -name '*.yaml' | sed 's|^\./||' | sort)
if [ -n "$PREV" ]; then
  for f in $PREV; do
    if ! printf '%s\n' "$NEXT" | grep -qx "$f"; then
      echo "  pruning removed object: $f"
      docker exec "$CONTAINER" sh -c "rm -f '$DATA_ROOT/$f'"
    fi
  done
fi

docker exec "$CONTAINER" sh -c "cp -r /rendered/. $DATA_ROOT/"
printf '%s\n' "$NEXT" | docker exec -i "$CONTAINER" sh -c "cat > $MANIFEST"

# Prove the copy landed. A bind mount that has gone stale, or a read-only
# store, would otherwise leave the previous config in place and the apply
# would look like it succeeded.
for f in $(cd "$RENDERED" && find . -name '*.yaml' | sed 's|^\./||'); do
  if ! docker exec "$CONTAINER" sh -c "cmp -s '/rendered/$f' '$DATA_ROOT/$f'"; then
    echo "install verification FAILED for $f: store copy differs from ./rendered" >&2
    echo "the bind mount is probably stale — recreate the container and retry" >&2
    exit 1
  fi
done
echo "verified $(cd "$RENDERED" && find . -name '*.yaml' | wc -l) object(s) installed into $DATA_ROOT"

# NO RESTART for ordinary config — the apiserver watches the tree.
#
# The exception is configmaps/: higress-config is read once at boot to build
# the mesh config, so a change there needs the apiserver and the controller
# restarted. The gateway keeps serving throughout; only the control plane
# blinks.
if [ "$RESTART" = 1 ]; then
  docker restart "${COMPOSE_PROJECT}-apiserver-1" "${COMPOSE_PROJECT}-controller-1" >/dev/null
  echo "restarted apiserver + controller; waiting for readiness"
  for _ in $(seq 1 60); do
    if docker exec "${COMPOSE_PROJECT}-apiserver-1" curl -sfk https://127.0.0.1:8443/readyz >/dev/null 2>&1; then
      echo "control plane up"; exit 0
    fi
    sleep 2
  done
  echo "control plane did not come up within 120s" >&2
  exit 1
fi

# --- prove the plugins actually loaded --------------------------------------
#
# THIS CHECK EXISTS BECAUSE THE FAILURE IS SILENT AND FAILS OPEN.
#
# A WasmPlugin whose module cannot be fetched still gets registered by Envoy as
# an ECDS http_filter — it just never receives a config. The filter sits inert
# in the chain, the route keeps answering 200, and key-auth authenticates
# nobody while ai-quota meters nothing. On 2026-09-03 this deployment served a
# completion to a caller with no Authorization header at all, because the
# committed plugin URL still pointed at the all-in-one's localhost:8002.
#
# `update_success` is the honest signal: `version_text` stays empty and
# `update_attempt` reaches 1 whether the fetch is pending or hopeless.
GW="${COMPOSE_PROJECT}-gateway-1"
if docker inspect "$GW" >/dev/null 2>&1; then
  echo "waiting for the gateway to load plugin modules"
  for _ in $(seq 1 30); do
    sleep 2
    stats=$(docker exec "$GW" sh -c 'curl -s --max-time 5 "http://127.0.0.1:15000/stats?filter=wasmplugin"' 2>/dev/null || true)
    [ -n "$stats" ] || continue
    pending=0
    for p in $(cd "$RENDERED" && ls wasmplugins/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//'); do
      ok=$(printf '%s\n' "$stats" | grep -E "wasmplugin/higress-system\.${p}\.update_success: [1-9]" || true)
      [ -n "$ok" ] || pending=$((pending+1))
    done
    [ "$pending" -eq 0 ] && { echo "all wasm plugins loaded (update_success >= 1)"; exit 0; }
  done
  echo "" >&2
  echo "WARNING: some wasm plugins never reported update_success." >&2
  echo "The gateway is FAILING OPEN — key-auth is not authenticating." >&2
  echo "Check WASM_PLUGIN_BASE ($WASM_PLUGIN_BASE) is reachable from $GW:" >&2
  echo "  docker exec $GW curl -sI $WASM_PLUGIN_BASE/plugins/key-auth/2.0.0/plugin.wasm" >&2
  exit 1
fi

echo "installed; the controller picks changes up on its own resync (~6s). No restart."
