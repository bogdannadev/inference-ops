# Cutover — all-in-one → official compose deployment

Status as of 2026-09-03: the compose deployment is **installed, configured and
validated end to end** under compose project `higress-next`, alongside the live
all-in-one. No traffic has been moved. Everything below is the remaining work.

## Why the move

Not for hgctl — that path is closed (see `../higress/NOTES.md`). The reasons
that survive scrutiny:

- **Per-component digest pinning.** `compose/.env` pins controller, pilot,
  gateway and plugin-server by `@sha256`, so pilot can be rolled back
  independently of the gateway. The all-in-one is one digest for everything.
- **The data plane gets its own core budget.** Cores 48-55 were shared by
  gateway, pilot, controller, apiserver and console inside one container. Now
  only the gateway competes for them with the other edge services.
- **An upstream-supported upgrade path** (`get-higress.sh -u`), with a backup
  taken before it writes.

Costs are real and documented in `../higress/NOTES.md`: 9 containers instead of
1, upstream's own "not extensively used in large-scale production" caveat, and
four services still on mutable tags.

## Validated already

| Check | Result |
|---|---|
| All 8 services healthy | yes, project `higress-next` |
| Upstream router resolves | `outbound|8000||qwen-router.dns` → 172.18.0.5 |
| Quota ledger resolves | test ledger 172.20.0.6, **not** production 172.20.0.2 |
| Wasm plugins loaded | all 4 report `update_success >= 1` |
| key-auth enforcing | bogus key → 401, no header → 401 |
| Valid consumer | 200, TTFB 0.48s non-streaming / 0.16s streaming |
| ai-quota metering | test ledger 50000 → 49916 |
| ai-statistics | input/output/total tokens + `llm_stream_duration_count` |
| Production ledger | untouched throughout |

## Blocking issues to resolve first

### 1. Compose project name collision — can destroy the live stack

The live deployment is compose project **`higress`** (verified on both
containers via `com.docker.compose.project`). `bin/startup.sh`, `shutdown.sh`,
`status.sh`, `logs.sh` and `update.sh` all hardcode `docker compose -p higress`.

**A `bin/shutdown.sh` or any `docker compose -p higress down` run from this
directory while the all-in-one still exists will delete the live gateway AND
the `higress-redis` container holding the billing ledger.** The ledger data
survives in the named volume `higress-quota-ledger`, but the container does
not, and the outage is immediate.

Until the old stack is gone, drive this deployment only with an explicit
project name:

    cd compose && COMPOSE_PROFILES='plugin-server' docker compose -p higress-next up -d

Do not run `bin/*.sh`.

### 2. Redis ownership

`../higress/docker-compose.yml` defines **both** `higress` and `higress-redis`.
Retiring the old stack with `docker compose down` takes the ledger container
with it. Decide one of:

- move the `redis` service into this tree's override, keeping the external
  volume `higress-quota-ledger`; or
- `docker compose stop higress` only, and leave the old project running purely
  to own redis (ugly, but zero-risk on the day).

The volume is the durable artefact — `appendonly yes`, 4 keys, real balances.
Whatever happens, **do not `docker volume rm higress-quota-ledger`**.

### 3. Console

The compose console seeds a broken `key-auth.internal` WasmPlugin on every
start (points at `key-auth/1.0.0`, plugin-server ships `2.0.0`, URL 404s). It
is inert — `defaultConfigDisable: true`, `FAIL_OPEN` — and recreating it is
automatic, so deleting it is pointless. Either accept it, or drop the console
service and the `default` ingress that routes to it.

## Cutover sequence

Requires a maintenance window: `qw38-27b-gw.duckdns.org` serves paying external
consumers.

1. **Freeze.** Announce the window. Confirm nobody is mid-request:
   `./stats.sh` in `../higress`.

2. **Snapshot the ledger.**

       docker exec higress-redis redis-cli BGSAVE
       docker run --rm -v higress-quota-ledger:/v -v "$PWD:/out" alpine:3 \
         tar czf /out/quota-ledger-$(date +%F).tar.gz -C /v .

3. **Point this deployment at the real ledger.** In `.env`:

       COMPOSE_PROJECT="higress"
       QUOTA_REDIS_DOMAIN="higress-redis.higressint"

   `apply.sh` refuses this combination unless the project is `higress`; that
   guardrail is what keeps a validation instance from billing real consumers.

4. **Stop the old gateway, keep redis.**

       docker compose -f ../higress/docker-compose.yml stop higress

5. **Claim the DNS name.** In `compose/.env`: `GATEWAY_EDGE_ALIAS='higress'`.
   Only now — while the old container ran, sharing that alias would have made
   Docker DNS round-robin paid traffic between two gateways.

6. **Bring up under the production project.**

       cd compose && COMPOSE_PROFILES='plugin-server' docker compose -p higress up -d

7. **Apply config**, and let the script's own check confirm the plugins loaded:

       ./apply.sh

8. **Point Caddy at the new port.** The all-in-one served the gateway on 8080;
   the official gateway image listens on 80. In `../qwen3.6-27b-2-A100/Caddyfile`:
   `reverse_proxy higress:8080` → `reverse_proxy higress:80`. Then reload —
   and verify through the admin API, not the exit code (see the Caddyfile
   bind-mount trap in `../higress/NOTES.md`).

   Prometheus needs **no change**: it scrapes `higress:15020` and the alias
   moved with the gateway.

9. **Verify before unfreezing**, in this order — the first two are the ones
   that fail open:

       # bogus key must be refused
       curl -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer nope' \
         https://qw38-27b-gw.duckdns.org/v1/models      # expect 401
       # real consumer must work and be metered
       ../higress/stats.sh

10. **Rollback**, if any check fails: set `GATEWAY_EDGE_ALIAS` back to
    `higress-next`, `docker compose -p higress down` in this tree, revert the
    Caddyfile, and `docker compose -f ../higress/docker-compose.yml start
    higress`. The all-in-one image is still digest-pinned and its `data/` tree
    is untouched, so it comes back exactly as it was.

## After cutover

- `git mv ../higress/config ../higress/consumers.conf .` and drop the
  `UPSTREAM=../higress` indirection in `apply.sh`.
- Remove the throwaway ledger: `docker rm -f higress-redis-test`.
- Retire `../higress/` once the rollback window has passed.
