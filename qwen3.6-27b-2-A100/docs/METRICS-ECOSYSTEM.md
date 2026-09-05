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
`status_class`. **These are the only per-consumer metrics that exist today.**

Since Stage D every series carries `node="a100"` and `stack="qwen36-27b"`.

**Its blind spot — and this is the big one for "per-key metrics":** the 98
engine metrics carry `engine_type`, `instance`, `is_streaming`, `model`,
`model_name`, `node`. **None carries a consumer.** So you can ask "what is p95
TTFT on this node" and you can ask "how many tokens did acme use at the
gateway", but you *cannot* ask "what is acme's p95 TTFT inside the engine".
Closing that is a config change, described at the bottom of this file.

### 4. Langfuse — the per-request waterfall, and nothing more

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

- **No per-user or per-session view.** `user_id` is set on 7 spans out of
  ~100,000 and `session_id` on **zero**. Langfuse's Users and Sessions pages
  are therefore empty. The identity lives on the gateway span; the tokens live
  on the engine span; and the router starts a new trace between them, so the
  two are never in the same trace for Langfuse to roll up. See
  `OBSERVABILITY.md` for the two-hop join that *does* work — outside Langfuse.
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
"what is p95 TTFT for acme"       -> NOT ANSWERABLE TODAY. See below.
```

## The one real gap: per-key engine metrics

The engine can label its own metrics per consumer, and does not today.

`--tokenizer-metrics-custom-labels-header` (default `x-custom-labels`) plus
`--tokenizer-metrics-allowed-custom-labels` make SGLang read a JSON object from
a request header and attach whitelisted keys as **Prometheus labels** on its
tokenizer metrics. Verified in the v0.5.19 source:
`entrypoints/openai/serving_base.py::extract_custom_labels` parses the header
and filters to the allowlist; `managers/tokenizer_manager.py` seeds those label
names into the metrics collector at startup.

So `x-custom-labels: {"consumer":"acme"}` on the upstream request would give
per-consumer TTFT, end-to-end latency and token histograms **from the engine
itself**, in Prometheus, joinable with everything else by `consumer`.

**What it costs:**
1. `--tokenizer-metrics-allowed-custom-labels consumer` on both replica command
   blocks — a roll of both replicas, one at a time, no downtime.
2. Higress must inject the header from the consumer it already knows
   (`X-Mse-Consumer`), which means an Envoy header-add with a `%REQ()%` format
   string, and the router must forward it.
3. Label cardinality: one label value per consumer. Fine at our scale; it is
   the thing to watch if the roster grows into the hundreds.

Not done. It is the single highest-value addition left in the metrics stack,
and it is the only way to answer "what is acme's p95 TTFT".
