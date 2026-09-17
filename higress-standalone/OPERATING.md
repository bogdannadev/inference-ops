# Operating the gateway

State as of 2026-09-03, verified against the running deployment. Companion
docs: `NOTES.md` (research and traps), `CUTOVER.md` (how this replaced the
all-in-one).

## What is running

Official Higress standalone v2.2.4, compose project `higress`, seven
containers, all pinned to cores **48-55** — the edge partition. Replica r0 owns
0-27 and r1 owns 28-47, so nothing here competes with inference.

| Container | Role | Networks |
|---|---|---|
| `higress-gateway-1` | Envoy. The only one carrying traffic. | `edge`, `higressint`, `higress-net` |
| `higress-controller-1` | Watches config, builds ServiceEntries | `edge`, `higressint`, `higress-net` |
| `higress-pilot-1` | xDS to the gateway | `higress-net` |
| `higress-apiserver-1` | Owns the config store (`./conf` → `/opt/data`) | `higress-net` |
| `higress-plugin-server-1` | Serves the `.wasm` modules over HTTP | `higress-net` |
| `higress-console-1` | Read-only admin UI, `127.0.0.1:8002` | `higress-net` |
| `higress-redis` | The ai-quota ledger. Volume `higress-quota-ledger`. | `higressint` |

Nacos and the bundled Prometheus/Grafana/Loki/Promtail are deliberately **not**
running — see `compose/docker-compose.override.yml`.

The gateway claims the network alias **`higress`** on `edge`. That is how Caddy
(`reverse_proxy higress:80`) and Prometheus (`higress:15020`) find it, and it
is set by `GATEWAY_EDGE_ALIAS` in `compose/.env`. Host ports are loopback-only:
gateway 18080/18444, metrics 15020, console 8002.

## The request path

Verified from the live listener dump. Envoy runs the wasm filters in this
order, and the order is load-bearing:

```
client → Caddy (gateway.example.org, strips X-Mse-Consumer,
                turns x-api-key into Authorization: Bearer)
       → gateway :80
           1. key-auth            AUTHN,  prio 310, FAIL_CLOSE
           2. request-validation  default, prio 950, FAIL_CLOSE
           3. ai-statistics       default, prio 900, FAIL_OPEN
           4. ai-token-ratelimit  default, prio 600, FAIL_OPEN   (rules owned by quota-bot)
           5. ai-quota            default, prio 280, FAIL_CLOSE
           6. ai-proxy            default, prio 100, FAIL_CLOSE  (ai-messages only)
       → qwen-router.dns:8000 → r0 / r1
```

`key-auth` sits in the AUTHN phase so it runs before everything; the others
share the default phase and are ordered by descending priority.
`request-validation` deliberately runs first of those three, so a request that
will be rejected never costs a Redis round trip or a metrics increment.

`ai-quota` runs *after* `ai-statistics` but does not depend on it — both call
`tokenusage.GetTokenUsage` independently, whatever the upstream docs imply.

## Plugins

| Plugin | Version | Routes | On failure | Does |
|---|---|---|---|---|
| `key-auth` | 2.0.0 | all five | **FAIL_CLOSE** | Matches the raw `Authorization` value against `consumers.conf`, sets `X-Mse-Consumer` |
| `request-validation` | 2.0.1 | all but ai-models | **FAIL_CLOSE** | Rejects an output cap over 70000 with 422 (`max_tokens`/`max_completion_tokens`; `max_output_tokens` on Responses) |
| `ai-statistics` | 2.0.1 | all five | FAIL_OPEN | Emits per-consumer token/latency counters |
| `ai-token-ratelimit` | 2.0.1 | all but ai-models | FAIL_OPEN | Daily / per-minute token limits, rules written by quota-bot |
| `ai-quota` | 2.0.1 | all but ai-models | **FAIL_CLOSE** | Gates on `chat_quota:<name> > 0`, DECRBYs after |
| `ai-proxy` | 2.0.1 | ai-messages | **FAIL_CLOSE** | Anthropic Messages ↔ OpenAI chat completions |
| `transformer` | 2.0.1 | ai-responses | FAIL_OPEN | Adds `"stream": false` when absent; SGLang 0.5.19 rejects a Responses request without it |

Routes (2026-09-17): `ai-chat` `/v1/chat/completions` (Prefix, for the quota
admin API under it), `ai-completions` `/v1/completions`, `ai-models`
`/v1/models`, `ai-responses` `/v1/responses` (POST create only; stored-response
retrieval is not offered because each replica stores its own), `ai-messages`
`/v1/messages` (Anthropic; converted to chat completions in the gateway, since
the router has no `/v1/messages`). `/v1/messages/count_tokens`, embeddings and
every SGLang-native path are 404.

Token usage for Responses and Anthropic answers is parsed by the same
`pkg/tokenusage` as chat (`response.usage`, `message.usage` / `usage`), so
ai-quota and ai-statistics meter all four billable routes.

`FAIL_OPEN` on `ai-statistics` alone is deliberate: losing metrics is not a
reason to stop serving. Losing authentication or quota enforcement is.

`ai-models` is excluded from quota (discovery is free) and from validation (no
body).

A fifth object, `key-auth.internal`, sits in the store. **It is not ours** —
the console seeds it on every start, it points at `key-auth/1.0.0` which the
plugin-server does not have (404), and it is inert only because it carries
`defaultConfigDisable: true` and `FAIL_OPEN`. It does not appear in the filter
chain for our routes. Deleting it is pointless; the console rewrites it.

### Confirming plugins are actually enforcing

This is the check that matters most, because the failure is silent and
**fails open** — an unfetchable module leaves the filter registered but inert,
and the route keeps answering 200 with no authentication. `apply.sh` now blocks
on this, but to check by hand:

```bash
docker exec higress-gateway-1 \
  curl -s 'localhost:15000/stats?filter=wasmplugin' | grep update_success
```

Every plugin must show `update_success >= 1`. `version_text` being empty and
`update_failure` staying `0` are *not* evidence of health — they look identical
whether a fetch is pending or hopeless.

Currently all four load from `http://plugin-server:8080` with
`istio_agent_wasm_config_conversion_count{result="success"} 12` and zero remote
fetches — nothing reaches for the Aliyun registry on the request path.

## Statistics

`ai-statistics` emits seven counters, labelled
`ai_route`, `ai_cluster`, `ai_model`, `ai_consumer`:

```
route_upstream_model_consumer_metric_input_token
route_upstream_model_consumer_metric_output_token
route_upstream_model_consumer_metric_total_token
route_upstream_model_consumer_metric_llm_duration_count          # requests
route_upstream_model_consumer_metric_llm_stream_duration_count   # streamed
route_upstream_model_consumer_metric_llm_first_token_duration    # sum, ms
route_upstream_model_consumer_metric_llm_service_duration        # sum, ms
```

The two `_duration` metrics are **sums, not averages** — divide by
`llm_duration_count` for a mean, which is what `stats.sh` does.
`ai_consumer="none"` means the request was not authenticated.

Scraped by the node's own Prometheus at `higress:15020` every 15s (job
`higress`), which is a strict superset of 15090 — it adds 71 `istio_agent_*`
series. Grafana has two dashboards: `qwen36-27b-gateway-higress` (ours) and
`qwen36-27b-gateway-higress-builtin` (extracted from the console jar).

**`use_default_attributes` is `false` on purpose.** Setting it `true` writes
every prompt and every model response verbatim into the gateway access log, and
makes the plugin buffer both the request body and the streaming response body.
Off, nothing is buffered and no customer content is logged. The per-consumer
token counters do not come from that setting and keep working — verified.

### The counters reset; the ledger does not

Envoy counters are process-lifetime and go to zero when the gateway container
restarts. Prometheus inherits that, so an `increase()` window spanning a
restart undercounts. For anything that has to be *right* — billing — read the
Redis balances, not the counters. `./stats.sh` prints both side by side and
labels which is which.

## Working with it

### Change config

`./config` is the source of truth: hand-authored, committed, installed only by
`apply.sh`. Nothing is authored through the console.

```bash
./apply.sh --dry-run     # render only; secrets land in ./rendered (gitignored)
./apply.sh               # install, then block until every plugin reports loaded
./apply.sh --restart     # also restart apiserver + controller — configmaps only
```

### Changing `higress-config` takes four steps, not two

`--restart` is **not sufficient** for the mesh config, and the way it fails is
silent: the object updates, the control plane restarts, every health check
passes, and the gateway carries on with the old config indefinitely.

Pilot does not read the ConfigMap. It reads three files —
`compose/volumes/pilot/config/{higress,mesh,meshNetworks}` — which the
`prepare` service materialises from the ConfigMap's `data` keys at stack
startup and at no other time.

The full sequence:

```bash
# 1. edit config/configmaps/higress-config.yaml, then
./apply.sh --restart

# 2. re-materialise pilot's files from the updated ConfigMap
cd compose && COMPOSE_PROFILES='plugin-server' \
  docker compose -p higress up prepare --no-deps

# 3. pilot reads those files only at boot
docker restart higress-pilot-1

# 4. confirm it actually reached Envoy — not that the object changed
docker exec higress-gateway-1 curl -s localhost:15000/config_dump | grep <your-change>
```

Step 4 is the one that matters. Steps 1-3 all report success while the gateway
serves the old config.

Re-running `prepare` against a live stack is safe: it checks whether pilot and
the gateway are up and skips certificate renewal if so ("Gateway is running.
Skip certificate renewal"). Do **not** run it while the gateway is down unless
you intend to regenerate the CA.

The apiserver watches its store, so ordinary objects go live on the
controller's resync (~6s) with no restart. `apply.sh` prunes objects you
deleted from `./config` (tracked in `/opt/data/.apply-manifest`) and verifies
each file landed before reporting success.

`conf/` is root-owned with `0700` subdirectories, which is why the copy happens
inside the apiserver container from the read-only `/rendered` mount rather than
from the host.

### Add a consumer

1. Add a line to `consumers.conf` — `<name>` then the credential **including
   the literal `Bearer ` prefix**, because key-auth matches the raw header
   value with no stripping.
2. `./apply.sh`
3. **Seed their balance, or they are locked out.** ai-quota cannot distinguish
   "never seeded" from "exhausted" from "Redis is down" — all three are a 403
   reading `No quota left`.

```bash
ADMIN=$(grep -E '^quota-admin[[:space:]]' consumers.conf | sed 's/^quota-admin[[:space:]]*//')
BASE=https://gateway.example.org/v1/chat/completions

curl -H "Authorization: $ADMIN" "$BASE/quota?consumer=NAME"              # read
curl -H "Authorization: $ADMIN" -d 'consumer=NAME&quota=1000000' "$BASE/quota/refresh"   # set
curl -H "Authorization: $ADMIN" -d 'consumer=NAME&value=500000'  "$BASE/quota/delta"     # add
```

The admin endpoints live **under** the chat path, not at the root —
`getOperationMode` builds `/v1/chat/completions` + `admin_path`. Only
`QUOTA_ADMIN_CONSUMER` may call them; anyone else gets 403 (verified).

### Check usage

```bash
./stats.sh              # all consumers: balance, requests, tokens, mean TTFT
./stats.sh danila       # one
```

### Everyday inspection

```bash
# is the upstream resolving?
docker exec higress-gateway-1 curl -s 'localhost:15000/clusters?format=json'

# what routes does the gateway actually have?
docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?resource=dynamic_route_configs'

# which plugins are loaded and enforcing
docker exec higress-gateway-1 curl -s 'localhost:15000/stats?filter=wasmplugin'

# component health
cd compose && COMPOSE_PROFILES='plugin-server' docker compose -p higress ps
```

`bin/status.sh` and `bin/logs.sh` also work now that this deployment owns the
`higress` project name.

### Restart / upgrade

```bash
cd compose && COMPOSE_PROFILES='plugin-server' docker compose -p higress up -d
```

`bin/update.sh` re-extracts the release tarball: it overwrites `bin/` and
`compose/docker-compose.yml`, preserves `compose/.env` (the new one arrives as
`compose/env_new`), and cannot touch `compose/docker-compose.override.yml`
because that filename does not exist upstream. **Every local deviation lives in
the override for exactly that reason** — never edit `docker-compose.yml`.

## Things that will bite

- **A plugin that cannot fetch its module fails open.** Always check
  `update_success`, never infer health from a 200.
- **Overdraft is real.** ai-quota gates on `> 0` and deducts afterwards, so a
  consumer with 1 token left can spend a whole request and go negative.
  Balances stay negative until a `/quota/refresh` sets them.
- **Adding a path to `./config/ingresses/` without adding it to
  `enable_path_suffixes` in `ai-quota.yaml` AND to the `ingress:` lists of
  key-auth, ai-quota, ai-token-ratelimit, request-validation and ai-statistics
  re-opens an unmetered (or unauthenticated) hole.** Every rule is per ingress.
  Measured previously: `/v1/completions` and SGLang's `/generate` both returned
  200 at zero balance.
- **`max_tokens` 70000 is quoted to customers in the Caddyfile 422 message.**
  Change one, change both.
- **Redis is a hard request-path dependency** on the billable routes. There is
  no fail-open option in ai-quota. If the ledger is down, that hostname is
  down — which is why the team's direct router hostname exists as the escape
  hatch.
- **The Caddyfile is a file bind-mount.** Edit it in place; a temp-file rename
  (what `sed -i` does) swaps the inode and `caddy reload` then silently
  re-reads the stale original. Verify through the admin API, not the exit code.
- **`edge` and `higressint` are unowned external networks** — the compose
  project that created `higressint` no longer exists. Nothing breaks, but if
  either is removed the stack will not start.
