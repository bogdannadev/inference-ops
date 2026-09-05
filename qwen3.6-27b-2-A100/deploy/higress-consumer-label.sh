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
#   x-custom-labels {"consumer":"%REQ(X-MSE-CONSUMER)%"}
#
# Higress renders that annotation into Envoy `request_headers_to_add` with
# append_action OVERWRITE_IF_EXISTS_OR_ADD, next to the Authorization line that
# is already there. SGLang's `extract_custom_labels` then parses the header as
# JSON and keeps the keys named by --tokenizer-metrics-allowed-custom-labels.
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
# `"consumer":"%REQ(X-MSE-CONSUMER)%"` and that is where docs/gateway.requests
# gets its consumer column from. The header exists at router-filter time and
# Envoy expands the command operator. If a future Higress release stopped
# expanding it, the failure is loud rather than silent: the metric label reads
# the literal `%REQ(X-MSE-CONSUMER)%` instead of a name.
#
# WHY IT NEEDS THE APISERVER
#
# The gateway is higress-standalone, a sibling deployment; its routes live in
# Nacos and are edited through the apiserver, which authenticates with a client
# certificate. The console container already holds one (its kubeconfig), so the
# certificate never leaves that container. The same edit can be made by hand in
# the Higress console UI: Routes -> ai-chat -> Request header update.
#
# ROLLBACK: --revert, or delete the line in the console. No restart either way;
# the controller pushes a new route config within a second or two.

set -euo pipefail

CONSOLE=higress-console-1
NS=higress-system
ROUTES=(ai-chat ai-completions)
KEY='higress.io/request-header-control-update'
LINE='x-custom-labels {"consumer":"%REQ(X-MSE-CONSUMER)%"}'
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

# Extract the client certificate INSIDE the console container. -q so the key
# never reaches this shell's stdout.
docker exec "$CONSOLE" sh -c '
  K=/home/higress/.kube/config
  grep client-certificate-data $K | awk "{print \$2}" | base64 -d > /tmp/hcl.crt
  grep client-key-data         $K | awk "{print \$2}" | base64 -d > /tmp/hcl.key
  chmod 600 /tmp/hcl.crt /tmp/hcl.key'

kc() { docker exec "$CONSOLE" curl -sk --cert /tmp/hcl.crt --key /tmp/hcl.key "$@"; }

cleanup() { docker exec "$CONSOLE" rm -f /tmp/hcl.crt /tmp/hcl.key /tmp/hcl.patch >/dev/null 2>&1 || true; }
trap cleanup EXIT

for r in "${ROUTES[@]}"; do
  cur=$(kc "$API/namespaces/$NS/ingresses/$r" |
        python3 -c "import json,sys; print(json.load(sys.stdin)['metadata']['annotations'].get('$KEY',''))")

  # Build the new value. Done in python so the Authorization bearer token is
  # copied through without ever being echoed.
  new=$(printf '%s' "$cur" | python3 -c "
import sys
cur = sys.stdin.read()
line = '''$LINE'''
keep = [l for l in cur.split('\n') if l.strip() and l.strip() != line]
if '$MODE' != 'revert':
    keep.append(line)
sys.stdout.write('\n'.join(keep) + '\n')
")

  if [ "$MODE" = dryrun ]; then
    echo "=== $r ==="
    printf '%s\n' "$new" | sed -E 's/(Authorization Bearer ).*/\1<redacted>/'
    continue
  fi

  printf '%s' "$new" |
    python3 -c "import json,sys; print(json.dumps({'metadata':{'annotations':{'$KEY': sys.stdin.read()}}}))" |
    docker exec -i "$CONSOLE" sh -c 'cat > /tmp/hcl.patch'

  code=$(kc -o /dev/null -w '%{http_code}' -X PATCH \
           -H 'Content-Type: application/merge-patch+json' \
           --data-binary @/tmp/hcl.patch \
           "$API/namespaces/$NS/ingresses/$r")
  echo "$r: PATCH -> $code"
  [ "$code" = 200 ] || { echo "unexpected status for $r" >&2; exit 1; }
done

[ "$MODE" = dryrun ] && exit 0

echo
echo "waiting for the gateway to pick up the new route config..."
for _ in $(seq 1 20); do
  if docker exec higress-gateway-1 curl -s localhost:15000/config_dump?resource=dynamic_route_configs |
       grep -q 'x-custom-labels'; then
    echo "gateway route config carries x-custom-labels"
    exit 0
  fi
  sleep 1
done

if [ "$MODE" = revert ]; then
  echo "x-custom-labels no longer in the route config (or never was)"
  exit 0
fi
echo "TIMEOUT: the annotation was written but the gateway has not applied it" >&2
exit 1
