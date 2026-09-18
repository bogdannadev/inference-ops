# Cutover — all-in-one → official compose deployment

**Completed 2026-09-03.** The all-in-one deployment is retired and its
directory removed. This document is kept as the record of what changed and how
to reverse it.

## What replaced what

| Before | After |
|---|---|
| one container `higress`, image `all-in-one@sha256:930b314b…` | 7 containers, project `higress` |
| plugins served from `localhost:8002` in-container | `plugin-server` container, `plugin-server:8080` |
| gateway on `:8080` | gateway on `:80`, alias `higress` on `edge` |
| objects in `/data` (container-managed) | objects in `./conf` (host bind mount, `/opt/data`) |
| `higress-redis` owned by the old compose project | same container, now owned by this project |
| cores 48-55 shared by every component | per-service `cpuset`, gateway alone on the data plane |

Deleted from the official defaults: `nacos` (file storage instead), and
`prometheus`, `promtail`, `loki`, `grafana` — upstream gives them no profile so
they start unconditionally, and there is no `O11Y=off` in compose mode. The
`postcheck` gate goes with them; it `depends_on` all three with
`service_healthy` and Compose merges `depends_on`, so the edges cannot be
removed. `postcheck.sh` is only `echo "All good!"`.

## Verified after cutover

| Check | Result |
|---|---|
| the gateway hostname, no auth | 401 |
| bogus key | 401 |
| valid consumer, streaming | 200, TTFB 0.23s |
| ai-quota metering | real ledger decremented |
| ai-statistics | per-consumer tokens in Prometheus |
| Prometheus target | `higress:15020` **up**, no config change needed |
| Caddy running config | `higress:80`, confirmed via the admin API |
| Ledger balances | carried over intact |

## The two things that made this safe

**Ledger continuity.** `higress-redis` was adopted, not reprovisioned: the
`quota-ledger` volume is declared `external` under its real name
`higress-quota-ledger`, so `docker compose down` on this project can never take
the balances with it. A tarball snapshot was taken first and is in `./backups`.

**The `higress` alias moved, so nothing downstream needed rewiring.** The
gateway claims it on `edge` via `GATEWAY_EDGE_ALIAS` in `compose/.env`.
Prometheus scrapes `higress:15020` and required no edit at all. Only Caddy
changed, because the port moved 8080 → 80.

While both gateways existed the alias was `higress-next` — sharing it would
have made Docker DNS round-robin between two gateways, half the requests
landing on an unconfigured one.

## Rollback

The all-in-one is gone, so rollback means reinstalling it. The image is still
digest-pinned in git history:

    git show <pre-cutover-sha>:higress/docker-compose.yml

Steps: restore that compose file and `apply.sh`, `docker compose -p higress
down` here, bring the all-in-one up, revert the Caddyfile to `higress:8080`
(rewriting the file **in place** — it is a file bind-mount, and a temp-file
rename swaps the inode so `caddy reload` silently re-reads the stale original),
and reload. The ledger volume is untouched by any of this.

## Operating this deployment

    cd compose && COMPOSE_PROFILES='plugin-server' docker compose -p higress up -d
    ./apply.sh                    # install ./config, verifies plugins loaded
    ./stats.sh                    # per-consumer balances and token counters

`bin/*.sh` now work as upstream intends, since this deployment owns the
`higress` project name. `bin/update.sh` re-extracts the release tarball: it
overwrites `bin/` and `compose/docker-compose.yml`, preserves `compose/.env`
(the new one lands as `compose/env_new`), and cannot touch
`compose/docker-compose.override.yml` because that filename does not exist
upstream. Every local deviation lives in the override for exactly that reason.
