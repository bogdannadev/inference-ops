#!/usr/bin/env bash
# =============================================================================
# Write alertmanager/webhook_secret (credentials_file in alertmanager.yml)
# from ALERT_WEBHOOK_SECRET in .env. The file is gitignored. Then reload:
#   docker exec qwen36-27b-alertmanager wget -qO- --post-data= http://127.0.0.1:9093/-/reload
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
secret=$(grep '^ALERT_WEBHOOK_SECRET=' .env | cut -d= -f2- | tr -d '"')
[ -n "$secret" ] || { echo "ALERT_WEBHOOK_SECRET not set in .env" >&2; exit 1; }
umask 077
printf '%s' "$secret" > alertmanager/webhook_secret
# Alertmanager runs as nobody.
./deploy/grant-nobody-read.sh alertmanager/webhook_secret
echo "wrote alertmanager/webhook_secret"
