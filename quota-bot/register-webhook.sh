#!/usr/bin/env bash
#
# Register, inspect or remove the Telegram webhook.
#
# Kept as a script rather than done at process startup, on purpose: registering
# a webhook is a change to state Telegram holds, not to this deployment, and it
# should be an explicit act with visible output. A bot that silently re-registers
# on every restart also makes "who pointed the webhook at what" unanswerable.
#
#   ./register-webhook.sh set      point Telegram at TELEGRAM_WEBHOOK_URL
#   ./register-webhook.sh info     what Telegram currently believes
#   ./register-webhook.sh delete   stop delivery
#
set -euo pipefail
cd "$(dirname -- "$0")"

[ -f .env ] || { echo "missing .env (copy .env.example)" >&2; exit 1; }

unquote() { sed -e 's/^"//' -e 's/"$//'; }
readvar() { grep -E "^$1=" .env | cut -d= -f2- | unquote; }

TOKEN=$(readvar TELEGRAM_BOT_TOKEN)
URL=$(readvar TELEGRAM_WEBHOOK_URL)
SECRET=$(readvar TELEGRAM_WEBHOOK_SECRET)

for v in TOKEN URL SECRET; do
  [ -n "${!v}" ] || { echo "TELEGRAM_${v} is unset in .env" >&2; exit 1; }
done

API="https://api.telegram.org/bot${TOKEN}"

case "${1:-info}" in
  set)
    # allowed_updates is restricted to plain messages: this bot has no inline
    # queries, no callbacks, no channel posts. Anything else Telegram would send
    # is queued, delivered, and dropped by the worker for nothing.
    #
    # drop_pending_updates clears the backlog. Without it, a webhook registered
    # after a spell of downtime replays every command sent meanwhile — including
    # any /topup, which the dedupe cannot help with because they are genuinely
    # distinct updates.
    curl -sS "${API}/setWebhook" \
      -d "url=${URL}" \
      -d "secret_token=${SECRET}" \
      -d 'allowed_updates=["message"]' \
      -d 'drop_pending_updates=true' \
      -d 'max_connections=10' | python3 -m json.tool
    echo
    echo "Registered. Verify with: $0 info"
    ;;

  info)
    # last_error_message is the field that matters. A webhook can be registered
    # and failing every delivery — TLS, a 404, or a handler returning non-2xx —
    # and nothing else here will say so.
    curl -sS "${API}/getWebhookInfo" | python3 -m json.tool
    ;;

  delete)
    curl -sS "${API}/deleteWebhook" -d 'drop_pending_updates=true' | python3 -m json.tool
    ;;

  *)
    echo "usage: $0 {set|info|delete}" >&2
    exit 1
    ;;
esac
