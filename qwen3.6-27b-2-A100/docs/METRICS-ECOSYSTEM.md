# The metrics ecosystem — what each store answers

Written 2026-09-05, after Stage D. This exists because "what does Langfuse
actually let me see?" had no good answer, and the honest answer turned out to
be *"less than you think, and one number was wrong by 293x."*

Read this before adding a dashboard or asking a question of the data.

## The one-line version

| Question | Ask | Never ask |
|---|---|---|
| How many tokens does X owe? | **Redis ledger** (`/balance`, `/usage`) | Langfuse, Prometheus |
| Who used what, exactly, per request? | **`gateway.requests`** (ClickHouse) | Langfuse |
| Is the node healthy right now? | **Prometheus / Grafana** | Langfuse |
| Why was *this one request* slow? | **Langfuse** (engine span waterfall) | Prometheus |
| How fast is the engine in general? | **Prometheus** (98 sglang metrics) | Langfuse |

**Billing reads the ledger.** Prometheus counters reset; Langfuse aggregates
are derived from spans and are not authoritative. That rule predates this
document and it still holds.

## The four stores

### 1. Redis ledger — the money

Balances per consumer, enforced by ai-quota at request time. This is the only
store that can say what someone owes. No history: it holds a balance, not a
series. `redis_exporter` publishes it to Prometheus so dashboards and alerts
can read it, but the ledger itself is the truth.

Bot: `/keys`, `/balance`, `/usage`, `/topup`, `/setquota`.

### 2. `gateway.requests` (ClickHouse) — the fact table

One row per request through the gateway, written by Vector from Envoy's access
log. **This is the per-request, per-consumer record of what actually happened**:
`request_id`, `consumer`, `route`, `model`, `status`, `duration_ms`, `llm_ms`,
`input_tokens`, `output_tokens`, `total_tokens`, `chat_id`, `chat_round`,
plus `node`/`stack` since Stage D.

End-to-end acknowledged and checkpointed, so a ClickHouse outage is a replay,
not a hole. 180-day TTL. Reconciled against the ledger exactly (370 == 370).

**Its blind spot:** it only sees traffic *through the gateway*. Requests on the
direct hostname bypass Higress entirely and never appear here. Until per-person
keys land, that is most of the traffic.

### 3. Prometheus — the time series

14 scrape targets. 98 `sglang:*` metrics (TTFT, queue depth, KV pool, MFU,
accept length, forward-pass timings), DCGM per-GPU fields, node, Caddy edge
RED, router `smg_*`, collector self-telemetry, ClickHouse, Alertmanager, and
the ledger via `redis_exporter`.

Plus `gateway_*` metrics that Vector derives from the same access log it writes
to ClickHouse — `gateway_requests_total`, `gateway_tokens_total`,
`gateway_request_duration_seconds`, all labelled by `consumer`, `route`,
`status_class` — the gateway's view of each tenant.

Since Stage D every series carries `node="a100"` and `stack="qwen36-27b"`.

**Its blind spot, until 2026-09-05:** the 98 engine metrics carried
`engine_type`, `instance`, `is_streaming`, `model`, `model_name`, `node` and
nothing about who asked, so "what is acme's p95 TTFT *inside the engine*" had
no answer. The tokenizer-side metrics now carry a `consumer` label — see
**Per-key engine metrics** at the bottom of this file. The scheduler-side metrics (KV pool, queue depth, cache hit rate,
MFU) remain node-wide by nature: they describe a shared GPU, not a request.

### 4. Langfuse — one request at a time, and per-user rollups

This is where the confusion was. Be precise about what it is:

**Langfuse 4.5.0 runs in `events_only` mode.** The v3 tables are empty by
design — `traces` 0 rows, `observations` 0 rows — and all 3.19M spans live in
`events_core` / `events_full`. The v3 API endpoints return:

> "This endpoint is not available on deployments running in Langfuse v4
> events_only mode."

Use `GET /api/public/v2/observations` and `GET /api/public/v2/metrics`. If you
followed a v3 doc page and got that error, nothing is broken.

**What Langfuse is genuinely good for here:** opening one request and seeing
the engine's phase breakdown — `request_process`, `prefill_waiting`,
`prefill_forward`, `decode_forward`, `tokenize`, and a `Req <id>` span carrying
`gen_ai.usage.*` and `gen_ai.latency.time_to_first_token`. That waterfall is
the only place the *inside* of a single request is visible. Nothing else has it.

**What it cannot do, and why:**

- **Per-user works. Per-session does not.** An earlier version of this file
  said `user_id` was set on "7 spans out of ~100,000" and called the Users page
  empty. That was the wrong denominator, and it was misleading: **`user_id` is
  set on 100% of the spans that can carry it** — 36 of 36 gateway ingress spans
  over 24h — and 0% of engine spans, which have no way to know who called. The
  fraction is tiny only because engine spans outnumber gateway spans ~6500:1.
  Filtering by consumer in Langfuse works today, both in the Users page and via
  `GET /api/public/v2/observations?userId=<consumer>`; verified 2026-09-05.
  `session_id` really is empty (zero spans) — see the Sessions note below.
- **A user's traces stop at the gateway.** The gateway ingress span carries the
  identity *and* `gen_ai.usage.*`, so per-consumer token and latency views hold
  up. What you cannot do is open one of those traces and see the engine's
  phase breakdown inside it: the router starts a new trace, so the engine spans
  live elsewhere. See `OBSERVABILITY.md` for the two-hop join that crosses it.
- **Token and cost aggregates were wrong by 293x until 2026-09-05.** Langfuse
  read `attributes.decode_ct` off `decode_loop` spans as token usage: 98,198,709
  over 24h against a real 345,195. `decode_loop` is now dropped in the
  collector, which also removed 76.9% of span volume. Post-fix a 4-minute
  window reports 144 tokens instead of millions. **Even so, do not bill from
  Langfuse** — there is no model pricing configured for `qwen36-27b`, so
  `totalCost` is meaningless.
- **It sees ~80% of engine spans, not all.** Router-routed requests share the
  router's trace id; health-check probes hit workers directly and get their own
  64-bit (zero-prefixed) trace. That cohort is noise, not loss.

### Should we move off Langfuse 4.5.0?

Checked 2026-09-05. Latest stable is **4.30.0**, released the day before; we
run 4.5.0, twenty-five minor versions back. The answer is **we can stay, and
upgrading would not have fixed anything we were confused about.**

*Why staying is safe.* Langfuse's own upgrade policy is that minor versions
within a major are non-disruptive and migrate themselves on start. Reading the
actual migrations rather than the policy: between the two tags there are
**2 ClickHouse migrations**, both `ADD COLUMN ... DEFAULT` / `MODIFY SETTING`
carrying the comment *"Metadata-only: existing parts are not rewritten"*, and
**7 Prisma migrations**, all in evaluator / feature-flag / integration tables we
do not populate. Our whole dataset is 540 MiB over 3.19M rows. The two breaking
changes in the window (4.20.0 `LANGFUSE_AWS_BEDROCK_*` -> `LANGFUSE_AI_*`,
4.24.0 requiring `LANGFUSE_AI_PROVIDER`) touch only the Langfuse-AI provider
config, which we do not set; the others are a 14-day JWT cap and an entitlement
on org API-key creation, neither of which we use.

*Why upgrading would not have helped.* The 25 releases are overwhelmingly
experiments, evaluators and dataset work. Our problem was never a missing
feature — it was that (a) `decode_loop` spans were being read as token usage,
which we fixed in the collector, and (b) engine spans carry no identity and the
router breaks trace continuity, which is an SGLang property no Langfuse version
changes. `events_only` is likewise not a bug to upgrade out of: it is v4's
intended end state, and 4.30 is further into it, not less.

*What we deliberately have not configured.* Model pricing. Setting a price for
`qwen36-27b` would make Langfuse's cost columns render a number, and that number
would immediately become a second, non-authoritative answer to "what does acme
owe" sitting next to the ledger. The cost of inference on this node is recorded
properly in `docs/KEY-TIERS.md` in ms and joules per token. Langfuse shows
`totalCost` = 0 on purpose.

*So the pin stays.* `docker-compose.langfuse.yml` holds 4.5.0 by digest. Revisit
if we ever want evaluators or Monitors; the upgrade itself is a pull, a
recreate and roughly a minute of migrations, with a Postgres dump as the
rollback.

## A new key needs no registration anywhere

Nothing has to be added to Grafana or Langfuse when a consumer is created.
Both are driven off data the key produces on its own, which is the property
worth protecting — a roster maintained by hand drifts the day someone forgets.

**Langfuse.** The `ai-statistics` wasm plugin is bound to the routes, not to
consumers, and maps `x-mse-consumer` onto two span attributes for every
request:

```json
{"key": "consumer",         "value": "x-mse-consumer", "value_source": "request_header"}
{"key": "langfuse.user.id", "value": "x-mse-consumer", "value_source": "request_header"}
```

`langfuse.user.id` is one of the attribute names Langfuse maps onto its
first-class `userId`, so a new consumer appears in Users on its first request.
`default_value` is `unauthenticated`, so failed-auth traffic is grouped rather
than dropped.

**Grafana.** *Usage & Quota* and *AI Gateway (Higress)* both carry a `consumer`
template variable — multi-select, All by default — and every per-consumer panel
filters on it. Two details are deliberate:

- **The roster comes from the ledger**, `label_values(consumer:quota_balance:tokens,
  ai_consumer)`, not from traffic. A key created a minute ago has a balance and
  no requests, and the Vector-derived access-log counters disappear from
  Prometheus entirely while the node is idle — sourcing the list from either
  would leave a new key unselectable.
- **All is `.*`, not the OR of that list.** So a consumer that is sending
  traffic but is missing from the ledger still shows up instead of silently
  vanishing from every panel.

Two label names exist and are not interchangeable: `ai_consumer` on the ledger
recording rules and Higress's own AI metrics, `consumer` on everything Vector
derives from the access log and on the sglang tokenizer metrics.

**Still missing: Sessions.** `session_id` is set on zero spans, so Langfuse's
Sessions page is empty and multi-turn conversations do not group. The mechanism
is the same one that already works for users — add a third `ai-statistics`
attribute mapping `langfuse.session.id` from a request header. OpenCode already
sends `X-Session-Id` (see the router routing-key note). Not done: it would
group only the clients that send such a header, and no one has asked to see
conversations grouped yet.

## Where to look, by question

```
"acme's bill is wrong"            -> /balance acme, then gateway.requests
                                     WHERE consumer='acme'
"the node feels slow"             -> Grafana overview + sglang dashboards
"this request took 40s"           -> gateway.requests for the request_id, then
                                     the two-hop join to Langfuse (OBSERVABILITY.md)
"are we dropping requests"        -> /errors, gateway_requests_total by status_class
"is the cache working"            -> sglang:cache_hit_rate, and the routing
                                     evidence in tuning/docs/ROUTING.md
"who is hammering the node"       -> /top, gateway_tokens_total by consumer
"what is p95 TTFT for acme"       -> Grafana Usage & Quota, Engine-side row,
                                     or /p95 acme
"just acme, everywhere"           -> the Consumer picker on Usage & Quota and
                                     AI Gateway; in Langfuse, filter Users
"I added a key, where is it"      -> already there. Grafana lists it from the
                                     ledger, Langfuse from its first request
"everything about one key"        -> /key in the bot: pick it from a list, no
                                     typing. Report writes an HTML file
```

## Per-key engine metrics — live 2026-09-05

The engine labels its own metrics per consumer. Verified end to end: a request
authenticated as `testafter` at the gateway arrives as
`sglang:num_requests_total{consumer="testafter"}` on the replica that served it.

```
client --Authorization: Bearer sk-...--> Higress
   key-auth resolves the consumer, sets X-Mse-Consumer
   route annotation adds  x-request-id-labels {"consumer":"%REQ(X-MSE-CONSUMER)%"}
       (OVERWRITE_IF_EXISTS_OR_ADD -- a client cannot forge it)
--> sgl-model-gateway forwards it because of the x-request-id- prefix
--> SGLang extract_custom_labels() parses the JSON, keeps allow-listed keys
--> consumer="testafter" on the tokenizer metrics
```

Config, both replicas:
`--tokenizer-metrics-allowed-custom-labels consumer`
`--tokenizer-metrics-custom-labels-header x-request-id-labels`
Gateway: `./deploy/higress-consumer-label.sh` (routes `ai-chat`, `ai-completions`).

### The header name is load-bearing

**SGLang's default header, `x-custom-labels`, never arrives.** The router
forwards a hardcoded ALLOW-LIST on the typed-request path that
`/v1/chat/completions` takes — `authorization`, `x-request-id`,
`x-correlation-id`, `traceparent`, `tracestate`, `x-smg-routing-key`, and
anything prefixed `x-request-id-`
(`routers/header_utils.rs::should_forward_request_header`, which carries a unit
test asserting exactly that set). Everything else is dropped with no log line.

Do **not** read `routers/header_utils.rs::apply_request_headers` and conclude
headers pass through. That function is permissive and this path does not use
it. Reading it instead of probing cost two replica rolls on 2026-09-05.

Measured, in this order:

| probe | result |
|---|---|
| straight to a replica, `x-custom-labels` | `consumer="probe-direct"` |
| through the router, `x-custom-labels` | nothing — dropped |
| through the router, `x-request-id-labels`, x6 | `consumer="probe-via-router"` 6.0 |
| through the gateway as `testafter`, x3 | `consumer="testafter"` 3.0 |

The router's request-id middleware matches `--request-id-headers` by exact
name, so it does not mistake `x-request-id-labels` for a request id.

### What this buys that nothing else could

The gateway already produced per-consumer request rate, token counts, status
mix and whole-request latency (`gateway_*`, from the access log). The engine
adds three things the gateway structurally cannot see:

| | why the gateway cannot |
|---|---|
| **Inter-token latency per consumer** | it sees a stream open and a stream close; a mid-decode stall looks identical to a smooth stream |
| **Prefix cache hit per consumer** | cache accounting happens inside the radix tree |
| **Replica attribution per consumer** | it hands every request to one router address and never learns whether r0 or r1 served it |

Panels: Grafana -> *Usage & Quota* -> **Engine-side, by consumer** (5 panels).
Bot: `/p95` appends an engine block with ttft / itl / e2e.

Labelled on TTFT, inter-token latency, and everything behind
`observe_one_finished_request` (e2e latency, prompt and generation token
histograms, cached and uncached prompt tokens).

### Cardinality, and why it is bounded

One label value per configured consumer (6 today), plus `consumer=""` for
anything that did not come through the gateway — health probes, benchmarks,
direct-hostname traffic.

The SGLang allow-list filters label *names*, never values, so nothing in the
engine stops a client minting values. What bounds them is the gateway's
`OVERWRITE_IF_EXISTS_OR_ADD`: a client that sends its own `x-request-id-labels`
has it replaced by the consumer Higress authenticated. **Do not expose the
router or a replica directly to clients while this flag is on.**

Two `probe-*` label values on r1 are leftovers from the 2026-09-05 verification.
They are counters on a live process, so they clear on the next roll of r1
rather than being deletable.

**Known gap, upstream:** `observe_one_aborted_request` does not take custom
labels (`# TODO: also use custom_labels from the request`,
tokenizer_manager.py), so aborted requests are still counted without a consumer.
