#!/usr/bin/env bash
# =============================================================================
# Let a container that runs as `nobody` (Prometheus, Alertmanager) read a
# secret file without letting other host users read it.
#
# Owner-only (600) locks the container out; world-readable (644) exposes the
# secret to every account on the host. The middle is owner smuser, group
# nogroup (65534): files 640, directories 750. Only root can set that group,
# so a throwaway container does it, and only when a path does not have it yet.
# Rewriting a file with `>` keeps its group, so this is needed once per file.
#
#   ./deploy/grant-nobody-read.sh <path>...   (paths relative to this directory)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
need=()
for p in "$@"; do
  [ "$(stat -c %g "$p")" = 65534 ] || need+=("/w/$p")
done
if [ ${#need[@]} -gt 0 ]; then
  docker run --rm --cpuset-cpus 48-55 --entrypoint chown -v "$PWD:/w" \
    python:3.12-alpine "$(id -u):65534" "${need[@]}"
fi
for p in "$@"; do
  if [ -d "$p" ]; then chmod 750 "$p"; else chmod 640 "$p"; fi
done
