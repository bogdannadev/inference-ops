# The metrics ecosystem — what each store answers

Rewritten 2026-09-15, when Langfuse, the OpenTelemetry collector and Vector's
per-consumer counters were removed. The design rule since then: **one source
per question, and the engine is the source of truth for anything it can
measure.** SGLang does the work and counts it; the gateway is asked only what
the engine cannot see.

Read this before adding a dashboard, a bot screen or a question of the data.

## The one-line version

| Question | Ask | Never ask |
|---|---|---|
| What does X have left to spend? | **Redis ledger** (`/balance`) | anything else |
| How many tokens did X use, exactly? | **`engine.requests`** → `engine_usage_*` (`/usage`, `/top`, `/key`) | `increase()` over any counter |
| Cache hit, HiCache share, engine latency per key? | **`engine.requests`** → `engine_usage_*` (`/key`, `/p95`) | histogram buckets |
| Refused, rate-limited, cut off, gateway latency? | **`gateway.requests`** → `gateway_usage_*` (`/errors`, `/top`) | engine data (it never saw those) |
| What did this one request do? | **`gateway.requests` joined to `engine.requests`** on `chat_id = rid` (`/trace`, admin-mcp `request_detail`) | a timestamp join |
| Is the node healthy right now? | **Prometheus** (`sglang:*`, DCGM, node) / Grafana, `/health` | usage gauges |

**Billing reads the ledger.** The exact usage numbers agree with it request by
request (verified 2026-09-15: 62 tokens charged, 62 in the engine records, 62 in
the limiter's window), but the balance is what bills.

## The picture

```
 client ──► Caddy ──► Higress gateway ──► smg router ──► SGLang r0 / r1
                        │  key-auth, ai-quota,             │  --export-metrics-to-file
                        │  ai-token-ratelimit               │  one JSON line per finished request
                        │  access log (file)                │  (logs/<replica>/request-metrics/)
                        ▼                                   ▼
                      Vector ─────────────────────────► Vector
                        │                                   │
                        ▼                                   ▼
             ClickHouse gateway.requests         ClickHouse engine.requests
                        └───────────┬───────────────────────┘
                                    │  GET /engine_usage_metrics (predefined query,
                                    │  read-only user; clickhouse/engine-usage-metrics.sql)
                                    ▼
 Redis ledger ─► redis_exporter ─► Prometheus ◄── sglang:* /metrics, DCGM, node, router
                                    │   engine_usage_*, gateway_usage_* (gauges, 1h/24h/7d/30d)
                                    │   consumer:* recording rules (burn, runway, share)
                     ┌──────────────┼────────────────┐
                     ▼              ▼                ▼
                  quota-bot       Grafana       Alertmanager
          admin-mcp ─► ClickHouse directly (any window, per request)
```

## The stores

### 1. Redis ledger — the money

Balances per consumer, enforced by ai-quota at request time; tier, per-key
settings and refill markers next to them; the limiter's window counters. The
only store that can say what someone can still spend. No history.
`redis_exporter` publishes balances to Prometheus (`consumer:quota_balance:tokens`).

Bot: `/keys`, `/balance`, `/policy`, `/topup`, `/setquota`.

### 2. `engine.requests` (ClickHouse) — what the engine did

One row per request SGLang **finished**, written by the engine itself
(`--export-metrics-to-file`), shipped by Vector, `ReplacingMergeTree` on
`(finished_at, rid)`, 400-day TTL. Per row: consumer (from the gateway's
`x-request-id-labels`), replica, prompt and completion tokens, cached tokens
split into GPU (`cached_device`) and HiCache (`cached_host`), queue, TTFT,
prefill, decode and end-to-end seconds, finish reason. No prompt text
(`--log-requests-level 1`). Schema: `clickhouse/engine-requests.sql`.

- **Token counts are identical** to the gateway's and the client's, request by
  request (verified 2026-09-13).
- **Rows before 2026-09-13 19:46 UTC** were copied once from `gateway.requests`
  (`source = 'gateway-backfill'`): tokens only, no cache split, no timings.
- **A client disconnect writes no record.** SGLang raises out of
  `_wait_one_response` before the exporter runs, so requests the client
  abandoned are visible only in `gateway.requests` as cuts — which matches the
  ledger, which charged them nothing. Scheduler-side aborts do write a record
  (`finish_type = 'abort'`) and are excluded from charged tokens.
- **Two clocks inside one record.** Received/finished timestamps come from the
  tokenizer process, forward-entry and prefill-finished from the scheduler, and
  their wall-clock anchors drift apart (r0: 1.03 s on 2026-09-15). Timings are
  therefore computed from same-process differences only — see `engine_rows` in
  `vector/vector.yaml`, tested in `vector/vector_test.yaml`.

### 3. `gateway.requests` (ClickHouse) — what the gateway saw

One row per request through Higress, from Envoy's access log via Vector:
`request_id`, `consumer`, `route`, `status`, `duration_ms`, `llm_ms`, tokens,
`response_flags`, and `chat_id` — the response id, which is the engine's `rid`,
so a gateway row joins its engine row exactly. 180-day TTL. Asked only what the engine cannot see: 401 / 403
/ 422 / 429 refusals, requests cut off before their usage frame (0 tokens
charged, flags DC/SI/UC/UPE/UT), and whole-request latency. Blind to traffic on
the direct hostname.

### 4. Prometheus — time series, and the exact gauges

- `sglang:*` engine metrics, DCGM, node, Caddy, router, ClickHouse,
  Alertmanager, the ledger via `redis_exporter`, quota-bot's policy metrics.
  Node health and trends. The per-consumer SGLang counters are fine for
  `rate()` trends and wrong for totals: a consumer's series is created by its
  first request after a replica start, so `increase()` never sees that request,
  and `increase()` extrapolates to the window edges (measured on 29,079 prompt
  tokens: engine `increase()` 11,194, gateway `increase()` 29,825, records 29,079).
- **`engine_usage_*` and `gateway_usage_*`** — exact sums and exact quantiles
  per consumer for the preceding 1h, 24h, 7d and 30d, computed in ClickHouse by
  `clickhouse/engine-usage-metrics.sql` and scraped once a minute (job
  `engine-usage`). They are gauges: read the latest value, never `rate()` them.
  Up to ~2 minutes behind. SQL regression cases: `./deploy/test-usage-sql.sh`.
- `consumer:quota_spend:*`, `consumer:quota_days_left`,
  `consumer:token_share:ratio1h` — recording rules over the exact gauges and
  the ledger (`prometheus/rules.yml`, tests in `rules_test.yml`).
- Alerts on the pipeline itself: `EngineUsageScrapeDown`,
  `EngineUsageRecordsStale`.

## Cache hits and what they cost

The engine reports, per request, how many prompt tokens came from the GPU
prefix cache and how many were reloaded from host RAM (HiCache).

- **Hit rate** = (GPU + HiCache tokens) / prompt tokens of requests whose split
  was recorded. The HiCache share is shown separately.
- **The balance is not discounted.** ai-quota deducts every prompt token,
  cached or not; there is no weighting option.
- **What a hit saves is engine time.** A GPU hit is nearly free. A HiCache hit
  costs a host-to-GPU reload — 1.24 s for 44,992 tokens against 14.2 s to
  recompute (2026-09-13). "Prefill avoided" in the bot uses the prefill speed
  measured on the same requests.
- **At reference prices** ("with cache"), cached input is priced at the
  provider's cached-input rate; input whose split is unknown is priced as
  uncached.

## A new key needs no registration anywhere

Nothing has to be added anywhere when a consumer is created, changed or
revoked; verified end to end on 2026-09-15 with a throwaway key (create → 200 on
the first request → records carry the consumer → tier and per-key limit change
reach the limiter and `/metrics` → revoke → 401 within 3 s, ledger, policy and
limiter window keys all deleted).

- **Engine records** carry the consumer from the gateway header on the first
  request.
- **Grafana** *Usage & Quota* takes its roster from the ledger,
  `label_values(consumer:quota_balance:tokens, ai_consumer)`, so a key with a
  balance and no traffic is selectable; All is `.*`, so traffic from a consumer
  missing from the ledger still shows.
- Two label names exist: `ai_consumer` on the ledger rules, Higress metrics and
  quota-bot's metrics; `consumer` on `engine_usage_*`, `gateway_usage_*` and the
  sglang tokenizer metrics.

## Where to look, by question

```
"acme's bill is wrong"            -> /balance acme; /key acme; then
                                     engine.requests WHERE consumer='acme'
"the node feels slow"             -> /health, Grafana overview + sglang dashboards
"this request took 40s"           -> /trace <request-id>: gateway row, then the
                                     engine row via chat_id = rid (queue, TTFT,
                                     decode, cache split)
"are we refusing requests"        -> /errors (429 limit, 403 balance, 5xx, cuts)
"is the cache working"            -> /key or /usage (GPU vs HiCache), /health,
                                     tuning/docs/ROUTING.md for routing evidence
"who is using the node"           -> /top
"what is p95 TTFT for acme"       -> /p95 acme, or Usage & Quota
"I added a key, where is it"      -> already there
"everything about one key"        -> /key: pick it from a list; Report writes HTML
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

The gateway sees per-consumer status codes and whole-request latency. The
engine adds what the gateway structurally cannot see:

| | why the gateway cannot |
|---|---|
| **Inter-token latency per consumer** | it sees a stream open and a stream close; a mid-decode stall looks identical to a smooth stream |
| **Prefix cache hit per consumer** | cache accounting happens inside the radix tree |
| **Replica attribution per consumer** | it hands every request to one router address and never learns whether r0 or r1 served it |

Panels: Grafana -> *Usage & Quota* -> **Engine-side, by consumer**. The bot's
numbers come from the per-request records (`engine.requests`), which carry the
same consumer label and add the GPU / HiCache split per request.

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
tokenizer_manager.py), so aborted requests are counted without a consumer on
the Prometheus side; client-disconnected requests also leave no per-request
record. Per-consumer cut-offs come from `gateway.requests`.
