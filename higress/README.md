# Higress — AI gateway evaluation

Evaluating Higress as the access-control layer in front of the Qwen3.8-27B
A100 node: **per-consumer API keys, token quotas, and per-consumer usage
statistics**.

Status: **evaluation, nothing in production.** The public edge is unchanged —
`model.example.com` still terminates at the Caddy in `../qwen3.6-27b-2-A100`,
which authenticates one shared `EDGE_API_KEY` and proxies straight to the
SGLang router.

## Why a gateway at all

The A100 node has exactly one credential for all clients. That means no
revocation without disrupting everyone, no attribution of load to a consumer,
no per-consumer limits, and no separation between the inference and embedder
endpoints.

Caddy can solve identity, revocation and route scoping with a key table, and it
was prototyped that way. What it **cannot** do at any level of effort is meter
tokens: counting tokens means reading the OpenAI response body, reading the
body means buffering it, and buffering destroys the SSE streaming that
`flush_interval -1` exists to protect. Quotas for an LLM are token quotas, so
the enforcement layer has to understand the protocol.

Rejected alternatives: `caddy-ratelimit` (requests-per-minute only, single
third-party module last tagged v0.1.0 in 2024, needs a custom xcaddy build),
LiteLLM (Python in the request path), Envoy AI Gateway (Kubernetes-only).

## Target topology

```text
internet
   │  TLS + ACME
   ▼
Caddy (../qwen3.6-27b-2-A100)      unchanged as the TLS terminator
   │
   ▼
higress:8080                       consumer key-auth
   │                               ai-statistics    -> token counts
   │                               ai-token-ratelimit -> per-consumer buckets (redis)
   │  Authorization swapped for the internal SGLANG_API_KEY
   ▼
qwen36-27b-router:8000  ──►  r0 / r1
```

Both projects meet only on the shared external `edge` network. This project
never names an inference container in a compose file, so it cannot recreate or
orphan one.

## Prerequisite already in place

`--stream-response-default-include-usage` is enabled on **both replicas**
(applied 2026-08-15 via `deploy/roll-replica.sh`, one at a time).

SGLang omits `usage` from streaming responses unless the client sends
`stream_options.include_usage`, and coding-agent clients don't. Measured
through the edge: a plain streaming request returned **zero** usage chunks; the
same request with `stream_options` returned `prompt_tokens=53 /
completion_tokens=8`. Without the server-side default, any gateway metering
tokens is blind to nearly all of our traffic.

## Operating the container

```bash
docker compose up -d
docker compose logs -f higress          # all components multiplexed by supervisord
ssh -L 8001:127.0.0.1:8001 <host>       # console at http://localhost:8001
```

Component logs live inside the container at `/var/log/higress/*.log`
(`controller.log`, `gateway.log`, `pilot.log`, `apiserver.log`). The gateway
access log is JSON, one line per request, and `response_code_details` is the
field that explains a 503.

There is **no supervisorctl socket** — restart the container, not a process.

## Configuring it

Config is Kubernetes resources served by a file-backed apiserver. There is no
kubectl in the image and `/app/kubeconfig` carries no credentials: the API is
anonymous on `https://localhost:18443` from inside the container.

```bash
docker exec higress sh -c \
  'curl -sk https://localhost:18443/apis/networking.higress.io/v1/namespaces/higress-system/mcpbridges/default'
```

Rules learned the hard way:

- **`PATCH` fails on custom resources** with a misleading `"default" not found`.
  Use `PUT` with the current `resourceVersion`, or DELETE + POST.
- Objects persist as files under `/data/<kind>/`, so a working config can be
  extracted into `./config/` and committed.
- Upstreams come from the `McpBridge` named `default`. The service host is
  `<registry-name>.<registry-type>` — a `static` registry named `qwen-router`
  is addressed as `qwen-router.static`.
- `higress.io/destination` **must include the port**
  (`qwen-router.static:8000`). Without it, or with the wrong registry type in
  the name, Envoy answers 503 and `gateway.log` says
  `response_code_details: cluster_not_found`.
- The controller picks up `McpBridge` changes on its own resync, not
  immediately.

## Current state (2026-09-02): the minimal chain is built and verified

Everything below was measured against the running stack, from inside the `edge`
network. **The public edge is untouched** — `model.example.com` still goes Caddy
-> router directly with the single shared key, and returned 200 throughout.

```
docker compose up -d
./apply.sh                 # render ./config -> /data, verify, restart
```

Live chain on the `ai-inference` route:

| Plugin | Version | Phase / prio | failStrategy | Role |
|---|---|---|---|---|
| key-auth | 2.0.0 | AUTHN / 310 | FAIL_CLOSE | credential -> `X-Mse-Consumer` |
| ai-statistics | 2.0.1 | default / 900 | FAIL_OPEN | per-consumer counters |
| ai-quota | 2.0.1 | default / 280 | FAIL_CLOSE | gate on balance, DECRBY after |

Upstream is `qwen-router.dns:8000` -> `qwen36-27b-router.edge:8000` -> r0/r1.
Nothing bypasses the router.

### Verified behaviour

- **The August blocker is dead.** A `dns` registry on `qwen36-27b-router.edge`
  produces endpoints and `GET /v1/models` returns 200. Root cause was
  `registry/direct/watcher.go`'s domain regex rejecting the dotless Docker
  hostname while the watcher still logged "ready". Docker's `<container>.<network>`
  name clears the regex, so this needed **no change to the running router**.
- **Auth:** no key 401, unknown key 401, known key 200, current `EDGE_API_KEY`
  200 as consumer `legacy-shared`.
- **Metering on streams:** 100 -> 38 after a streamed completion reporting
  `total_tokens: 62`. No buffering; `usage` arrives in the final SSE frame
  because `--stream-response-default-include-usage` is live on both replicas.
- **Gate:** unseeded consumer 403 ("No quota left" — a null key is *denied*,
  not unlimited). Exhausted consumer 403. A second consumer is unaffected.
- **Overdraft is real:** balance 1 -> one full request succeeds -> balance -61.
  The gate is `> 0` and the deduction lands afterwards.
- **Coverage:** `/v1/models` returns 200 even for an exhausted consumer.
  `enable_path_suffixes` only covers `/v1/chat/completions` and `/v1/messages`.
- **Consumer spoofing fails:** authenticating as an exhausted consumer while
  sending `X-Mse-Consumer: <consumer with budget>` returned 403 and did not
  bill the forged identity. The authenticated name wins. Strip the header at
  the edge anyway — it wins by Envoy header semantics, not by contract.
- **Redis is a hard dependency, as designed and as feared:** with Redis
  stopped, chat requests 403 (ai-quota has no fail-open; a Redis error is
  indistinguishable from an empty balance in its code). `/v1/models` still 200.
  Balances **survived** the restart — `appendonly yes` is load-bearing.
- **Per-consumer metrics** land in Envoy stats as
  `wasmcustom.route.ai-inference.upstream.<cluster>.model.<model>.consumer.<name>.metric.<n>`.

### Config model

`./config` is the source of truth and is committed. `./apply.sh` renders it
into `./rendered` (gitignored, holds secrets), copies that into the container's
`/data`, **verifies the copy landed**, and restarts. Files are the config: the
apiserver runs `--storage file --file-root-dir /data` and the image's own
templates write YAML there before it starts. Driving the apiserver API instead
means PATCH failing on custom resources, PUT needing a live `resourceVersion`,
and waiting on the controller's resync.

`apply.sh` clears `./rendered`'s *contents* rather than the directory. Deleting
and recreating it swaps the inode while the container still holds the old one
mounted, and the install silently becomes a no-op — the same trap as editing
the Caddyfile out from under Caddy. That bug happened during this build; the
verification step exists so it cannot happen silently again.

Secrets live in `consumers.conf` and the inference stack's `.env`, both
gitignored. `.env` values there are double-quoted and `apply.sh` strips the
quotes itself.

## Not done yet

1. **Measure the added hop's TTFT cost.** Decode is bandwidth-bound at ~50ms
   per forward so an Envoy hop should be invisible, but the noise floor is
   1.5% — this needs the harnesses in `../qwen3.6-27b-2-A100/benchmarks/`, not
   one curl.
2. **Scrape the consumer counters** into the A100 node's Prometheus and build a
   per-consumer panel.
3. **Public hostname.** Add `model-gw.example.com` as a NEW Caddy site pointing
   at `higress:8080`, with `header_up -X-Mse-Consumer` and `flush_interval -1`.
   The existing site is not touched, so there is nothing to roll back and no
   flag day: clients migrate one at a time, and the direct path stays as the
   escape hatch from the Redis dependency above.
4. **Retire `legacy-shared`** once its counter has been flat at zero.
5. ~~Router hardening~~ **DONE 2026-09-02** (`bfd3eef`). The router's control
   plane was open — `GET /workers` answered 200 to anything on `edge`. Closed
   with `--control-plane-api-keys`, plus `--request-id-headers` for the trace
   join key. Note it is NOT `--api-key`: `--help` on this build says that flag
   is "the api key used for the authorization with the worker", an outbound
   credential, contradicting the docs page that calls it client auth. The
   inference path was untouched and every client kept working.

## Attribution already works without any of this

Per-consumer token attribution does **not** depend on this gateway. SGLang's
own engine spans carry token counts and reach Langfuse today. What they lack is
client identity, because the sglang-router discards inbound W3C trace context
(workers honour it; the router roots its own trace and there is no flag for
it). The router does record an inbound request id from `--request-id-headers`
as `attributes.request_id` on the root span of the engine trace, which is an
exact join key against whatever the edge logs. That path was proven end-to-end.
If Higress is adopted, it should set that header; if it isn't, any edge can.
