#!/usr/bin/env bash
#
# Render ./config into the container's /data and restart it.
#
# Why files and not the API: the apiserver runs with `--storage file
# --file-root-dir /data`, and the image's own config templates write YAML
# straight into that tree before it starts. So files ARE the config. Driving
# the anonymous apiserver instead means PATCH failing on custom resources with
# a misleading "default not found", PUT needing the live resourceVersion, and
# the controller reacting only on its own resync. A restart is deterministic
# and this stack is not load-bearing.
#
# Secrets never enter ./config. They live in consumers.conf and the inference
# stack's .env, and are substituted into ./rendered, which is gitignored.
#
#   ./apply.sh            render, install, restart
#   ./apply.sh --dry-run  render only, print what would be installed
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

CONTAINER=higress
RENDERED=./rendered

# --- inputs -----------------------------------------------------------------

[ -f .env ] || { echo "missing .env (copy .env.example)" >&2; exit 1; }
[ -f consumers.conf ] || { echo "missing consumers.conf (copy consumers.conf.example)" >&2; exit 1; }

# .env values in this stack are double-quoted. compose strips those quotes for
# us; every other reader has to do it itself, and a credential carrying literal
# quote characters authenticates nobody.
unquote() { sed -e 's/^"//' -e 's/"$//'; }

# shellcheck disable=SC1091
QUOTA_ADMIN_CONSUMER=$(grep -E '^QUOTA_ADMIN_CONSUMER=' .env | cut -d= -f2- | unquote)
SGLANG_ENV_FILE=$(grep -E '^SGLANG_ENV_FILE=' .env | cut -d= -f2- | unquote)

[ -f "$SGLANG_ENV_FILE" ] || { echo "SGLANG_ENV_FILE not found: $SGLANG_ENV_FILE" >&2; exit 1; }
SGLANG_API_KEY=$(grep -E '^SGLANG_API_KEY=' "$SGLANG_ENV_FILE" | cut -d= -f2- | unquote)
[ -n "$SGLANG_API_KEY" ] || { echo "SGLANG_API_KEY empty in $SGLANG_ENV_FILE" >&2; exit 1; }

# --- consumer table ---------------------------------------------------------
# Emitted as YAML rather than templated, because the list is variable length.

read_consumers() { grep -vE '^\s*(#|$)' consumers.conf; }

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
cp -r config/. "$RENDERED"/

export SGLANG_API_KEY QUOTA_ADMIN_CONSUMER
while IFS= read -r -d '' f; do
  tmp=$(mktemp)
  # Only the two named variables are substituted. A bare envsubst would eat
  # any $... that happens to appear in a comment or a regex.
  envsubst '${SGLANG_API_KEY} ${QUOTA_ADMIN_CONSUMER}' <"$f" >"$tmp"
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

echo "rendered $(find "$RENDERED" -name '*.yaml' | wc -l) object(s) from ./config"
find "$RENDERED" -name '*.yaml' | sort | sed 's|^| |'

if [ "$DRY_RUN" = 1 ]; then
  echo
  echo "--dry-run: not installing. Secrets are present in $RENDERED (gitignored)."
  exit 0
fi

# --- install ----------------------------------------------------------------

docker inspect "$CONTAINER" >/dev/null 2>&1 || {
  echo "container '$CONTAINER' does not exist; run: docker compose up -d" >&2; exit 1; }

# /data is root-owned inside the container, so the copy happens in-container
# from the read-only ./rendered mount rather than from the host.
# Prune objects a previous apply installed that are no longer in ./config.
# Without this, deleting a file here leaves the object live in /data forever —
# which for an Ingress means a route we thought we removed is still serving.
MANIFEST=/data/.apply-manifest
PREV=$(docker exec "$CONTAINER" sh -c "cat $MANIFEST 2>/dev/null" || true)
NEXT=$(cd "$RENDERED" && find . -name '*.yaml' | sed 's|^\./||' | sort)
if [ -n "$PREV" ]; then
  for f in $PREV; do
    if ! printf '%s\n' "$NEXT" | grep -qx "$f"; then
      echo "  pruning removed object: $f"
      docker exec "$CONTAINER" sh -c "rm -f '/data/$f'"
    fi
  done
fi

docker exec "$CONTAINER" sh -c 'cp -r /rendered/. /data/'
printf '%s\n' "$NEXT" | docker exec -i "$CONTAINER" sh -c "cat > $MANIFEST"

# Prove the copy landed before restarting. A bind mount that has gone stale,
# or a read-only /data, would otherwise leave the previous config in place and
# the restart would look like a successful apply.
for f in $(cd "$RENDERED" && find . -name '*.yaml' | sed 's|^\./||'); do
  if ! docker exec "$CONTAINER" sh -c "cmp -s '/rendered/$f' '/data/$f'"; then
    echo "install verification FAILED for $f: /data copy differs from ./rendered" >&2
    echo "the bind mount is probably stale — recreate the container and retry" >&2
    exit 1
  fi
done
echo "verified $(cd "$RENDERED" && find . -name '*.yaml' | wc -l) object(s) installed into /data"

# NO RESTART for ordinary config.
#
# Measured 2026-09-02: the apiserver runs with `--storage file --file-root-dir
# /data` and WATCHES that tree. A consumer added by writing key-auth.yaml went
# live in ~6s, and one removed was revoked in ~6s, both without restarting.
# That answers the open question in NOTES.md: files are not a startup-only
# input. Restarting on every key change was needless downtime.
#
# The exception is /data/configmaps/: start-apiserver.sh reads higress-config
# once at boot to build the mesh config, so a change there DOES need a restart.
# Pass --restart for that.
if [ "$RESTART" = 1 ]; then
  docker restart "$CONTAINER" >/dev/null
  echo "restarted $CONTAINER; waiting for the gateway to answer"
  for _ in $(seq 1 60); do
    docker exec "$CONTAINER" sh -c 'nc -z 127.0.0.1 8080' 2>/dev/null && { echo "gateway up"; exit 0; }
    sleep 2
  done
  echo "gateway did not come up within 120s; check: docker logs $CONTAINER" >&2
  exit 1
fi

echo "installed; the controller picks changes up on its own resync (~6s). No restart."
