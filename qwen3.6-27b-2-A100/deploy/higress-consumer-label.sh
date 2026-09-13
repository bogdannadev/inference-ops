#!/usr/bin/env bash
#
# higress-consumer-label.sh [--revert] [--dry-run]
#
# Make Higress tell SGLang who is calling, so engine metrics can carry a
# consumer label.
#
#   ./deploy/higress-consumer-label.sh              # apply
#   ./deploy/higress-consumer-label.sh --dry-run    # print the new annotation
#   ./deploy/higress-consumer-label.sh --revert     # remove the injected line
#
# WHAT IT DOES
#
# Appends one line to `higress.io/request-header-control-update` on the two
# generation routes:
#
#   x-request-id-labels {"consumer":"%REQ(X-MSE-CONSUMER)%"}
#
# Higress renders that annotation into Envoy `request_headers_to_add` with
# append_action OVERWRITE_IF_EXISTS_OR_ADD, next to the Authorization line that
# is already there. SGLang's `extract_custom_labels` then parses the header as
# JSON and keeps the keys named by --tokenizer-metrics-allowed-custom-labels.
#
# WHY THE ODD HEADER NAME
#
# SGLang's default is `x-custom-labels` and that header never reaches a worker.
# sgl-model-gateway forwards a hardcoded ALLOW-LIST on the path
# /v1/chat/completions takes: authorization, x-request-id, x-correlation-id,
# traceparent, tracestate, x-smg-routing-key, and the prefix `x-request-id-`
# (should_forward_request_header). Everything else is dropped without a log
# line. Measured 2026-09-05: straight to a replica the label appeared, through
# the router it did not. `x-request-id-labels` rides that prefix, and the
# replicas are told to read it with --tokenizer-metrics-custom-labels-header.
# Change one side and you must change the other.
#
# WHY THE OVERWRITE MATTERS
#
# The SGLang allow-list filters label NAMES, never values. A client that could
# set its own x-custom-labels would be able to mint an unbounded number of
# `consumer` values and blow up Prometheus cardinality. OVERWRITE_IF_EXISTS_OR_ADD
# means whatever the client sent is replaced by the consumer the gateway itself
# authenticated, so the value set is exactly the set of configured consumers.
#
# WHY %REQ() IS SAFE HERE
#
# It is not a guess: the access log on this same gateway already uses
# `"consumer":"%REQ(X-MSE-CONSUMER)%"` and that is where gateway.requests gets
# its consumer column from. The header exists at router-filter time and Envoy
# expands the command operator. If a future Higress release stopped expanding
# it, the failure is loud rather than silent: the metric label reads the
# literal `%REQ(X-MSE-CONSUMER)%` instead of a name.
#
# WHY PUT AND NOT PATCH
#
# The gateway is higress-standalone; its routes live in Nacos behind an
# api-server that is file-backed (`file_rest.go`), not etcd. It ADVERTISES
# `patch` in its APIResourceList and then answers 500 to a
# merge-patch+json — tried 2026-09-05. Read-modify-write with a full PUT is
# what the console itself does, and it works.
#
# Note the store returns no resourceVersion, so there is NO optimistic
# concurrency here: a PUT is last-write-wins over the whole object. Do not run
# this while someone is editing the same route in the console.
#
# The certificate never leaves the console container. The same edit can be made
# by hand: console -> Routes -> ai-chat -> Request header update.
#
# SINCE 2026-09-13 the line is also committed in
# ../../higress-standalone/config/ingresses/{ai-chat,ai-completions}.yaml, so
# apply.sh renders it instead of deleting it. Before that, every apply.sh run
# silently removed it. This script remains for --revert and for checking a store.
#
# ROLLBACK: --revert, or delete the line in the console. No restart either way;
# the controller pushes a new route config within a second or two.

set -euo pipefail

CONSOLE=higress-console-1
NS=higress-system
ROUTES=(ai-chat ai-completions)
KEY='higress.io/request-header-control-update'
LINE='x-request-id-labels {"consumer":"%REQ(X-MSE-CONSUMER)%"}'
API=https://apiserver:8443/apis/networking.k8s.io/v1

MODE=apply
for a in "$@"; do
  case "$a" in
    --revert)  MODE=revert ;;
    --dry-run) MODE=dryrun ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

docker inspect "$CONSOLE" >/dev/null 2>&1 || {
  echo "$CONSOLE is not running; start higress-standalone first" >&2; exit 1; }

docker exec "$CONSOLE" sh -c '
  K=/home/higress/.kube/config
  grep client-certificate-data $K | awk "{print \$2}" | base64 -d > /tmp/hcl.crt
  grep client-key-data         $K | awk "{print \$2}" | base64 -d > /tmp/hcl.key
  chmod 600 /tmp/hcl.crt /tmp/hcl.key'

kc() { docker exec "$CONSOLE" curl -sk --cert /tmp/hcl.crt --key /tmp/hcl.key "$@"; }

cleanup() {
  docker exec "$CONSOLE" rm -f /tmp/hcl.crt /tmp/hcl.key /tmp/hcl.body >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Rewrites the annotation on a whole Ingress object read from stdin. Kept in
# one place so --dry-run and the real run cannot diverge. The Authorization
# bearer token is copied through inside python and never echoed.
edit_py() {
  cat <<PY
import json, sys
obj  = json.load(sys.stdin)
line = '''$LINE'''
ann  = obj['metadata'].setdefault('annotations', {})
cur  = ann.get('$KEY', '')
# Drop any line that sets one of OUR header names, not just the exact string we
# are about to write. The header name changed once already (x-custom-labels ->
# x-request-id-labels) and a stale line would have survived a plain match,
# leaving the route injecting a header nothing reads.
ours = ('x-custom-labels', 'x-request-id-labels')
keep = [l for l in cur.split('\n')
        if l.strip() and not l.strip().lower().startswith(ours)]
if '$MODE' != 'revert':
    keep.append(line)
ann['$KEY'] = '\n'.join(keep) + '\n'
json.dump(obj, sys.stdout)
PY
}

for r in "${ROUTES[@]}"; do
  body=$(kc "$API/namespaces/$NS/ingresses/$r" | python3 -c "$(edit_py)")

  if [ "$MODE" = dryrun ]; then
    echo "=== $r ==="
    printf '%s' "$body" |
      python3 -c "import json,sys; print(json.load(sys.stdin)['metadata']['annotations']['$KEY'])" |
      sed -E 's/(Authorization Bearer ).*/\1<redacted>/'
    continue
  fi

  printf '%s' "$body" | docker exec -i "$CONSOLE" sh -c 'cat > /tmp/hcl.body'

  out=$(kc -w '\n%{http_code}' -X PUT \
          -H 'Content-Type: application/json' \
          --data-binary @/tmp/hcl.body \
          "$API/namespaces/$NS/ingresses/$r")
  code=${out##*$'\n'}
  echo "$r: PUT -> $code"
  if [ "$code" != 200 ] && [ "$code" != 201 ]; then
    # Print what the server actually said. Guessing at a 500 cost an hour once.
    printf '%s\n' "${out%$'\n'*}" |
      sed -E 's/(Authorization Bearer )[A-Za-z0-9-]*/\1<redacted>/g' | head -20 >&2
    exit 1
  fi
done

[ "$MODE" = dryrun ] && exit 0

echo
echo "waiting for the gateway to pick up the new route config..."
for _ in $(seq 1 20); do
  if docker exec higress-gateway-1 curl -s localhost:15000/config_dump?resource=dynamic_route_configs |
       grep -q 'x-request-id-labels'; then
    [ "$MODE" = revert ] || { echo "gateway route config carries x-request-id-labels"; exit 0; }
  elif [ "$MODE" = revert ]; then
    echo "x-request-id-labels is gone from the route config"
    exit 0
  fi
  sleep 1
done

[ "$MODE" = revert ] && { echo "TIMEOUT: x-request-id-labels still in the route config" >&2; exit 1; }
echo "TIMEOUT: the annotation was written but the gateway has not applied it" >&2
exit 1
