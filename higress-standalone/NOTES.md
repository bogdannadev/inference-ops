# Research notes — read before touching the container

House rule for this stack: **learn a feature from upstream docs and source at
the pinned version first; use the container to confirm, never to discover.**
The August session broke that rule on the upstream-naming question and spent
four rounds guessing at 503s. The list below is what to read first.

## Versions to pin research against

Everything here is what the pulled image actually contains, not what the
website's "latest" describes.

| Component | Version | How it was determined |
|---|---|---|
| Higress release | **v2.2.4** | see "Reading the release tag" below |
| Deployment | official standalone `compose/`, project `higress` | `get-higress.sh`, VERSION=v2.2.4 |
| controller / pilot / gateway / plugin-server | digest-pinned in `compose/.env` | shipped by the release |
| Go wasm plugins | **2.0.1** | `/usr/share/nginx/html/plugins/snapshot-inventory.json` |
| key-auth | **2.0.0** | directory name only — not in the inventory, see below |
| Envoy | **1.36.4** | `curl localhost:15000/server_info` |
| Istio pilot | **1.27-dev** | `pilot-discovery version --short` |
| Base | Ubuntu 22.04 | image labels |

### Reading the release tag

The tag is **not** stamped in the binaries. Verified 2026-09-03:

    $ docker exec <all-in-one> /usr/local/bin/higress version
    HIGRESS_VERSION:
    GIT_COMMIT_ID:

Both empty. Don't bother with the console UI either — the reliable method is
the **image build date against the `higress-standalone` release date**, because
the all-in-one image is built by that repo's release job:

- image `Created` = `2026-08-14T01:26:39Z` (`docker image inspect`)
- `higress-standalone` v2.2.4 released **2026-08-14**
- `higress-group/higress` v2.2.4 released 2026-08-13
- v2.2.4's own PR #4488 bumps the plugin snapshot to the `2.0.1` this image ships

Three independent signals, one answer: **this image is v2.2.4**.

## Upgrade posture (reviewed 2026-09-03)

**v2.2.4 is the newest release across all three repos** — higress 2026-08-13,
higress-console 2026-08-13, higress-standalone 2026-08-14. There is nothing to
upgrade to. `main` is ~39 commits ahead and `chore: prepare plugin snapshot
2.2.5` landed 2026-09-01, so v2.2.5 is close but unreleased.

v2.2.4 was a large release — 21 features, 56 bug fixes. The ones that land on
this stack, **all already active in the running image**:

- **#4265 `ai-statistics` SSE framing.** Adds a request-scoped SSE framer with
  incremental byte scanning and LF/CRLF boundary detection. Before it, a
  streaming response whose HTTP chunk boundary split an SSE event mid-frame
  produced **zero token counts**. This is why Question 3 below is answered.
- **#4256 `ai-quota` unsafe type assertions.** Bare assertions on context
  values panicked under "multi-plugin coexistence / context contamination" —
  exactly our key-auth + ai-statistics + ai-quota chain.
- **#4060 `key_auth` cache keys + perf.** Wrong auth cache-key generation, plus
  a fast/slow path replacing an exhaustive scan of every consumer per request.
- **#4104 Envoy 1.36 Redis reconnects.** `AsyncClientImpl::initialize()` no
  longer force-destroys Redis connections when the config is unchanged. We run
  `higress-redis` and re-apply plugin config on every `apply.sh`, so this is
  our path exactly.
- **#4450 McpBridge CRD** — Nacos `timeout` misclassified as required, plus
  restored CRD validation.

Behaviour changes that are already live and would bite a config written against
older docs:

- **#4011 — breaking.** `ai-token-ratelimit` and `cluster-key-rate-limit`
  changed rule matching from **first-match-wins to all-match OR-overlay**:
  every matching rule now consumes. Also adds `maxRuleItems=10`. Relevant only
  if we ever move off `ai-quota`.
- **#4258** `ai-proxy` gained `disableStreamUsageStats`. We force
  `--stream-response-default-include-usage` engine-side, so the injection is
  redundant on our path, not harmful.
- **#4139** the `higress-ops` MCP server now requires HTTP basic auth.

**Nothing after v2.2.4 argues for moving.** All ~39 post-release commits were
read: release plumbing, `ai-agent`, `mcp-server`, `hgctl`, Helm, a Nacos
stale-cache fix (#4596) and an `ai-proxy` hunyuan SSE bounds guard (#4611).
**None touch key-auth, ai-quota, ai-statistics or ai-token-ratelimit.** Stay on
the digest. Revisit when v2.2.5 ships.

## A WasmPlugin whose module cannot be fetched FAILS OPEN

Found the hard way on 2026-09-03, standing up the compose deployment.

Envoy registers each WasmPlugin as an ECDS `http_filter` **before** it has the
module. If the fetch never completes, the filter stays in the chain with no
config and simply does nothing — while the route keeps answering 200. There is
no error page, no 503, and no log line at default verbosity.

Concretely: the committed plugin URLs pointed at `http://localhost:8002`, which
is where the all-in-one's internal plugin-server lives. In compose mode the
plugin-server is a separate container (`plugin-server:8080`), so `localhost`
was the gateway itself and the fetch hung. The result was a gateway that served
completions **to a caller with no Authorization header at all** — key-auth
authenticating nobody, ai-quota metering nothing.

The honest signal is in Envoy's stats, not in the object:

    docker exec <gateway> curl -s 'localhost:15000/stats?filter=wasmplugin'

- `...<name>.update_success: 0` with `update_attempt: 1` — registered, never
  loaded. **This is the failure.**
- `...<name>.version_text: ""` — same for a pending fetch and a hopeless one,
  so it does not distinguish on its own.
- `update_failure` and `update_rejected` stay at `0`, which is why this looks
  healthy at a glance.

Both apply scripts now take the plugin base URL from `.env` as
`WASM_PLUGIN_BASE`, and `higress-standalone/apply.sh` blocks after installing
until every plugin reports `update_success >= 1`, failing loudly if not. Never
assume a plugin is enforcing because the object exists and the route answers.

The same reasoning applies to the seeded `key-auth.internal`: its URL 404s, and
it is harmless only because it also carries `defaultConfigDisable: true`.

## Plugin version identity — two traps

**A plugin's directory version is a distribution label, not the source
version.** At tag v2.2.4, `plugins/wasm-cpp/extensions/key_auth/VERSION` reads
`1.0.0`, but the image serves it from `plugins/key-auth/2.0.0/`. The number is
stamped by the release pipeline. Never map a shipped plugin version onto a
source tree by name.

**A plugin version can be overwritten in place.** After v2.2.4 shipped, PR
#4573 (an MCP schema fix) was published through an *"emergency same-version
plugin tag overwrite workflow"* (#4576) that republished **mcp-server 2.0.1
over itself** — same version string, different bytes. Therefore the
`wasmSha256` values in `snapshot-inventory.json` are the only sound identity
for a bundled plugin. Record those, not version numbers.

**key-auth has neither.** It is absent from `snapshot-inventory.json`
entirely — as are `basic-auth`, `jwt-auth`, `hmac-auth`, `request-block` and
the other C++ plugins; only the Go plugins get snapshot entries. So there is no
upstream digest to verify key-auth against. The only evidence it carries the
#4060 fix is that its `plugin.wasm` is timestamped `Aug 13 22:25`, identical to
every other plugin in the image (i.e. built from the v2.2.4 tree), and #4060
merged 2026-07-31. **That is inference, not verification.**

## The Higress CLI — what exists and what applies here

Short answer: **there is no Higress CLI in this deployment, and the one
upstream ships is not meant for it.** Three different things get called "the
CLI"; only one is real, and it does not fit.

### 1. `hgctl` — the real CLI, not installed, mostly inapplicable

Lives in `higress-group/higress` under `hgctl/`. **Not in the image** — it is a
separately-built binary. Commands at v2.2.4 (`hgctl/pkg/root.go`): `version`,
`gateway-config`, `install`, `uninstall`, `upgrade`, `profile`, `dashboard`,
`manifest`, `plugin`, `completion`, `code-debug`, `mcp`, `agent`.

- `install` / `uninstall` / `upgrade` / `profile` / `manifest` / `dashboard`
  assume **hgctl owns the deployment** — Kubernetes via Helm, or its own
  local-docker/standalone installer (`pkg/installer/installer_docker.go`,
  `standalone.go`). **Never point these at this stack.** They would try to take
  ownership of a deployment that `docker-compose.yml` + `apply.sh` own, and
  hgctl's own post-2.2.4 fixes (#4542 "reject unapplied local-docker overlays",
  #4543 "fail closed on Helm ownership errors") are not in the released binary.
- `gateway-config` (alias `gc`) is the genuinely useful one — and it is
  **unusable here**. It takes a *Kubernetes pod* (`hgctl gc all <pod-name> -n
  higress-system`) and port-forwards to Envoy admin on 15000. There is no pod.
  But there is an Envoy admin API, and that is the exact data hgctl reads — go
  straight to it, see below.
- `plugin` (aliases `plg`, `p`) — `init`, `build`, `install`, `uninstall`,
  `ls`, `test`, `config`. For developing Go wasm plugins. This is the part
  worth installing if we ever need to audit or patch a bundled plugin, since
  it builds the same artifacts the release pipeline does.

### 2. `higress-standalone`'s `bin/*.sh` — a different deployment mode

`configure.sh`, `startup.sh`, `shutdown.sh`, `restart.sh`, `status.sh`,
`logs.sh`, `update.sh`, `reset.sh`. These drive the **multi-container
`compose/` mode**, not the all-in-one image, and not our compose file. Ignore
them; `apply.sh` is our equivalent.

### 3. `/usr/local/bin/higress` in the container — not a CLI

That 119 MB binary is the **ingress controller**. Its entire surface is
`serve`, `version`, `completion`, `help`. `version` prints empty (above).

### What to use instead — Envoy admin on :15000

This is what `hgctl gateway-config` would have shown. All read-only, all
verified 2026-09-03. Admin is bound inside the container only.

    # Clusters and their resolved endpoints — the cluster_not_found answer.
    docker exec higress-gateway-1 curl -s 'localhost:15000/clusters?format=json'

    # Full xDS state. `include_eds` adds EndpointsConfigDump, which plain
    # /config_dump omits — that omission is what makes a DNS registry look
    # fine while resolving to nothing.
    docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?include_eds'

    # Narrow it: resource= filters repeated resources, mask= filters top-level
    # fields. `?resource=bootstrap` is INVALID (bootstrap is not repeated).
    docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?resource=dynamic_active_clusters'
    docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?resource=dynamic_route_configs'
    docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?resource=dynamic_listeners'
    docker exec higress-gateway-1 curl -s 'localhost:15000/config_dump?mask=bootstrap'

    # Envoy build, uptime, effective command line.
    docker exec higress-gateway-1 curl -s 'localhost:15000/server_info'

    # Everything the admin API offers.
    docker exec higress-gateway-1 curl -s 'localhost:15000/help'

Wasm plugin config is embedded in the HCM filter chain, so it comes back under
`resource=dynamic_listeners` — grep that dump for the plugin name to see what
the gateway actually loaded, rather than what `./config` says it should have.

Faster check of *which* plugins the gateway holds, and whether each has a live
config version:

    docker exec higress-gateway-1 curl -s 'localhost:15000/stats?filter=wasm'

Each loaded plugin reports
`extension_config_discovery.http_filter.extensions.istio.io/wasmplugin/higress-system.<name>.version_text`.
A populated timestamp means the gateway has a config for it. An **empty**
`version_text` means the object exists but never resolved — which is exactly
how the console's `key-auth.internal` shows up. Seeing it there is the quickest
confirmation of that trap.

Metrics live on **:15020** (`/stats/prometheus`), which is what `stats.sh` and
Prometheus read; Envoy's raw counters are on 15000 (`/stats?filter=...`).
Config CRUD is the anonymous apiserver on **:18443** — but prefer writing files
under `/data`, per Question 4.

**One live-change endpoint, listed so it is recognised, not so it is used:**
`POST localhost:15000/logging?level=<lvl>` retunes proxy log level on a running
gateway (GET is rejected — POST is required even to list). It affects
production traffic logging immediately. Scope it (`?wasm=debug`) and set it
back, and treat it under the never-disturb rule.

## Question 1 — ANSWERED 2026-09-02: dns registry rejects single-label hosts

**Root cause found.** `registry/direct/watcher.go` validates the domain against

    ^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}$

which the bare Docker hostname `qwen36-27b-router` cannot satisfy — no dot. The
watcher logs "Registry Watcher is ready, type:dns", then `generateServiceEntry`
returns nil, so no Envoy cluster is created and every request 503s with
`cluster_not_found`. Nothing above DNS was broken; the name never got that far.

**Fix applied, no router change needed:** Docker's embedded DNS also answers
`<container>.<network>`, and `qwen36-27b-router.edge` both resolves and clears
the regex. The Redis network was renamed `higressint` (not `higress-internal`)
for the same reason. Rationale is written up in
`config/mcpbridges/default.yaml` — read that file before touching a registry.

Confirmed live: `outbound|8000||qwen-router.dns` carries one healthy host, and
`outbound|6379||quota-redis.dns` likewise.

## Question 2 — ANSWERED: the consumer auth → quota chain works

`key-auth` sets the consumer identity, `ai-statistics` attributes tokens to it,
`ai-quota` decrements a durable Redis ledger. Verified end to end: gateway
metrics carry an `ai_consumer` label, and `chat_quota:<consumer>` keys track
balances. `./stats.sh` reports both sides and documents why they disagree
(counters reset on container restart; the ledger does not).

`ai-quota` was chosen over `ai-token-ratelimit` — a refillable per-consumer
token quota fits "each teammate gets N tokens/day" better than a sliding
per-minute limiter. Note #4011 changed the limiter's matching semantics in this
very release, which is a further reason not to revisit that choice casually.

Plugin config schemas, if they need re-reading, are authoritative in source at
the pinned tag: `plugins/wasm-go/extensions/<plugin>/` (`main.go`, `config.go`).
Docs lag, and every docs URL below is a `/latest/` path that drifts away from
v2.2.4 — treat them as orientation, source as truth. Docs host is `higress.ai`;
`higress.cn` 404s.

- key-auth: <https://higress.ai/en/docs/latest/user/plugins/authentication/key-auth/>
- ai-token-ratelimit: <https://higress.ai/en/docs/latest/user/plugins/ai/api-consumer/ai-token-ratelimit/>
- ai-statistics: <https://higress.ai/en/docs/latest/user/plugins/ai/api-o11y/ai-statistics/>
  — the `rule` field on attribute extraction (`first` / `replace` / `append`)
  is how it collapses fragmented SSE into one value; that is the knob to reach
  for if a streamed attribute ever looks wrong.
- service sources / 服务来源: <https://higress.ai/en/docs/latest/user/>

## Question 3 — ANSWERED: streaming token counting works

Two independent confirmations:

1. **Upstream.** v2.2.4 carries #4265, which added the SSE framer specifically
   because chunk-split SSE events produced zero token counts. The failure mode
   we would have hit is the one that release fixed.
2. **Measured.** `route_upstream_model_consumer_metric_llm_stream_duration_count`
   is non-zero and tracks most requests (37 of 43 for one consumer), with
   input/output/total token counters all advancing.

Engine side was already solved: `--stream-response-default-include-usage` is
live on both replicas, so SGLang emits a final usage chunk on every stream even
when the client doesn't ask.

Worth knowing: #3975 added an `llm_failure_count` counter to `ai-statistics`,
which closes the old blind spot where an error response carrying no token usage
was invisible. It only appears in `/stats/prometheus` once it increments.

## Question 4 — ANSWERED 2026-09-02: the apiserver watches /data, no restart needed

Measured, twice: a consumer added by writing `conf/wasmplugins/key-auth.yaml`
went live in **~6s** with no restart, and one removed was revoked in ~6s. The
apiserver runs `--storage file --file-root-dir /data` and picks up file changes
on the controller's own resync. `apply.sh` no longer restarts (1.2s instead of
a ~40s restart cycle).

The exception is `conf/configmaps/`: the apiserver reads
`higress-config` once at boot to build the mesh config, so a change there does
need `./apply.sh --restart`.

Objects live under `conf/<kind>/` on the host, `/opt/data/<kind>/` in the apiserver. The image's own config templates write YAML
straight into that tree before the apiserver starts — files ARE the config, not
a cache of it. `./config` is our committed source of truth; nothing is authored
through the console.

## Question 5 — OPEN: cost of the extra hop

Decode on this node is bandwidth-bound at ~50ms per forward pass, so an Envoy
hop should be invisible. Measure rather than assume, and measure TTFT
specifically (that is where a proxy hop shows up), comparing:
`client → router` versus `client → higress → router`, same prompt, same
replica. The benchmark harnesses in `../qwen3.6-27b-2-A100/benchmarks/` are the
maintained tools for this.

## Plugins ship inside the image

**All needed plugins are bundled** — `key-auth`, `ai-statistics`,
`ai-token-ratelimit`, `ai-quota`, `ai-proxy`, `cluster-key-rate-limit` — served
by the local plugin-server. That matters: the Aliyun registry is unreliable
from this host (a token-endpoint fetch failed outright on 2026-09-03), so a
runtime plugin pull would be a liability. Prefer the bundled versions; never
point plugin URLs at a remote registry.

## SGLang-side references

- `--request-id-headers` (router): how the inbound request id is chosen and
  recorded as `attributes.request_id`. Source: `sgl-router/src/middleware.rs`
  (the span attributes seen in ClickHouse come from there). This is the join
  key that makes per-consumer token attribution work **regardless** of which
  gateway is in front — see the memory note on the router dropping trace
  context.
- Router option surface: `sglang_router/router_args.py` inside the pinned
  engine image — authoritative, and quicker than the docs site.
- SGLang docs: <https://docs.sglang.io/> — thin on the router; treat source as
  the reference.

## Things already established (don't re-derive)

- The apiserver's config API is anonymous on `https://127.0.0.1:8443` inside
  the apiserver container; no kubectl anywhere; `PATCH` fails on custom
  resources, use `PUT` with `resourceVersion` or DELETE + POST. Prefer writing
  files — that is what `apply.sh` does.
- `higress.io/destination` must be `<registry-name>.<registry-type>:<port>`.
- `MODE=full|gateway|console` and `O11Y=off` were all-in-one environment
  variables and no longer exist. In compose mode each component is its own
  container, and the bundled Prometheus/Grafana/Loki/Promtail are held down by
  a profile in `compose/docker-compose.override.yml` because upstream gives
  them none.
- The controller picks up McpBridge changes on its own resync, not instantly.
- Never create consumers in the console — it writes a disabled, mis-versioned
  `key-auth.internal` that never reaches the real routes.
