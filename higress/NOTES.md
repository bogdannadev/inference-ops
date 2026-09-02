# Research notes — read before touching the container

House rule for this stack: **learn a feature from upstream docs and source at
the pinned version first; use the container to confirm, never to discover.**
Yesterday's session broke that rule on the upstream-naming question and spent
four rounds guessing at 503s. The list below is what to read first.

## Versions to pin research against

Everything here is what the pulled image actually contains, not what the
website's "latest" describes.

| Component | Version | How it was determined |
|---|---|---|
| all-in-one image | `sha256:930b314be06e4b435617a39ac8dc8f17c9d60a4faf2c96aa7aafe9e85381ad9b` | pulled 2026-08-15 |
| wasm plugins | **2.0.1** (key-auth 2.0.0) | `/usr/share/nginx/html/plugins/*/`, `snapshot-inventory.json` |
| Envoy | **1.36.4** | `envoy --version` |
| Istio pilot | **1.27-dev** | `pilot-discovery version --short` |
| Base | Ubuntu 22.04 | image labels |

The Higress release tag is not stamped in the binaries (`HIGRESS_VERSION` is
empty). Get it from the console UI once running, and read docs/source at *that*
tag — plugin config schemas change between majors, and the 2.x plugin docs are
not the same as 1.x.

**All needed plugins ship inside the image** — `key-auth`, `ai-statistics`,
`ai-token-ratelimit`, `ai-quota`, `ai-proxy`, `cluster-key-rate-limit` — served
by the local plugin-server. That matters: the Aliyun registry is unreliable
from this host, so a runtime plugin pull would be a liability. Prefer the
bundled versions; do not point plugin URLs at a remote registry.

## Blocking question 1 — upstream naming (solve before anything else)

**Symptom.** A `dns` registry for `qwen36-27b-router` starts a watcher
("Registry Watcher is ready, type:dns") but yields no endpoints, so no Envoy
cluster is created and every request is a 503 with
`response_code_details: cluster_not_found` in `/var/log/higress/gateway.log`.
A `static` registry with the container's IP works, but pins an address that
changes whenever the router is recreated — unusable in production.

**Hypothesis to test first:** `qwen36-27b-router` is a single-label hostname.
Istio ServiceEntry hosts are validated as DNS names and single-label hosts may
be rejected or resolved differently, and Envoy's DNS cluster type may refuse
them. Docker's embedded DNS resolves it fine from a shell in the same
container — so the failure is above DNS, in Higress or Istio validation.

**Read, in this order:**

1. `higress-group/higress` — the McpBridge → ServiceEntry conversion and the
   DNS registry watcher. Look for host validation, FQDN assumptions, and
   whether a port/protocol field is mandatory:
   - `pkg/ingress/kube/mcpbridge/` (controller reconcile)
   - `registry/dns/` and `registry/static/` (watchers)
   - `registry/reconcile/` (how a registry becomes a ServiceEntry)
   Search terms: `Registry Watcher is ready` (the exact log line), `ServiceEntry`,
   `Resolution_DNS`, `validateHost`.
2. Istio host validation rules for ServiceEntry (`istio.io` docs, ServiceEntry
   reference) — specifically whether short names are permitted and how
   `resolution: DNS` treats them.
3. Higress service-source docs — the registry types and their required fields:
   <https://higress.ai/en/docs/latest/user/> (service sources / 服务来源).
   Note: `higress.cn` 404s; `higress.ai` is the live docs host.

**Escape hatches, in order of preference** (decide with evidence, not by
guessing):
- Give the router a **dotted network alias** on `edge` (e.g.
  `router.qwen.internal`) and use a dns registry. Costs one router recreate —
  a live change, must be scheduled and confirmed. In-flight requests drop; the
  workers are untouched.
- Use Higress's **kubernetes/nacos-free static registry with a hostname**
  rather than an IP, if the static watcher resolves names (test it).
- Keep static+IP for the evaluation only, and never ship it.

## Question 2 — the consumer auth → quota chain

The three plugins have to hand data to each other, and the linkage is the part
that is easy to get wrong: `key-auth` must produce a **consumer identity** that
`ai-token-ratelimit` can match with `limit_by_consumer`.

Read at the 2.0.x plugin version:
- key-auth: <https://higress.ai/en/docs/latest/user/plugins/authentication/key-auth/>
  — consumer list format, where the key is read from (header vs query), and
  what identity it sets downstream.
- ai-token-ratelimit: <https://higress.ai/en/docs/latest/user/plugins/ai/api-consumer/ai-token-ratelimit/>
  — `rule_name`, `rule_items`, `limit_by_consumer`, `token_per_minute|hour|day`,
  and the `redis` block (`service_name` wants an FQDN-style name — check what
  it accepts for a plain compose service).
- ai-statistics: <https://higress.ai/en/docs/latest/user/plugins/ai/api-o11y/ai-statistics/>
  — it is what computes the token counts the rate limiter spends. Confirm
  whether it needs `ai-proxy` in front or works against a plain
  OpenAI-compatible upstream.
- Source for all three (schema is authoritative, docs lag):
  `higress-group/higress` → `plugins/wasm-go/extensions/<plugin>/` — read
  `main.go` / `config.go` for the exact YAML keys.

**Also evaluate `ai-quota`** (bundled, 2.0.1). It is a different model —
refillable per-consumer token quota rather than a sliding rate limit — and may
fit "each teammate gets N tokens/day" better than a per-minute limiter. Compare
before building on ai-token-ratelimit.

## Question 3 — does token counting see streaming responses?

The engine side is **already solved and verified**:
`--stream-response-default-include-usage` is live on both replicas, so SGLang
emits a final usage chunk on every stream even when the client doesn't ask.
What remains is whether `ai-statistics` reads usage from the SSE stream's last
chunk rather than only from non-streaming JSON.

- Docs: the `rule` field on attribute extraction (`first` / `replace` /
  `append`) is how it handles fragmented SSE — read that section closely.
- Source: `plugins/wasm-go/extensions/ai-statistics/` — find where it parses
  `data:` frames and where it recognises `usage`.
- Test with a real streaming request and a non-streaming one; compare what the
  plugin records. Do not assume the streaming path works because the
  non-streaming one does.

## Question 4 — ANSWERED 2026-09-02: the apiserver watches /data, no restart needed

Measured, twice: a consumer added by writing `/data/wasmplugins/key-auth.yaml`
went live in **~6s** with no restart, and one removed was revoked in ~6s. The
apiserver runs `--storage file --file-root-dir /data` and picks up file changes
on the controller's own resync. `apply.sh` no longer restarts (1.2s instead of
a ~40s restart cycle).

The exception is `/data/configmaps/`: `start-apiserver.sh` reads
`higress-config` once at boot to build the mesh config, so a change there does
need `./apply.sh --restart`.

## Question 4 (original text) — config as committable files

Objects live under `/data/<kind>/` in the container. Before hand-writing YAML,
confirm whether the apiserver **reads** files written directly or only serves
what came through its API (yesterday's evidence was ambiguous: a `PUT` was
stored, but the controller only reacted on its own resync).

- `higress-group/higress-standalone` — the apiserver's storage backend and
  whether it watches the filesystem.
- Practical approach either way: author through the API, then extract
  `/data/<kind>/*.yaml` into `./config/` and commit that as the source of
  truth, with a small apply script.

## Question 5 — cost of the extra hop

Decode on this node is bandwidth-bound at ~50ms per forward pass, so an Envoy
hop should be invisible. Measure rather than assume, and measure TTFT
specifically (that is where a proxy hop shows up), comparing:
`client → router` versus `client → higress → router`, same prompt, same
replica. The benchmark harnesses in `../qwen3.6-27b-2-A100/benchmarks/` are the
maintained tools for this.

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

- Higress config API is anonymous on `https://localhost:18443` inside the
  container; no kubectl in the image; `PATCH` fails on custom resources, use
  `PUT` with `resourceVersion` or DELETE + POST.
- `higress.io/destination` must be `<registry-name>.<registry-type>:<port>`.
- `MODE=full|gateway|console`; `O11Y=off` keeps the bundled
  Prometheus/Grafana/Loki/Promtail from starting.
- No supervisorctl socket — restart the container, not a process.
- The controller picks up McpBridge changes on its own resync, not instantly.
