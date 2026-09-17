# Observability

The observability tier is defined in `docker-compose.metrics.yml` (same
compose project as the inference tier) and split into two layers:

1. **Prometheus-native** — infra/GPU liveness and hardware health
   (`prometheus/alerts.yml`)
2. **Grafana Unified Alerting** — application SLOs (latency, KV pool, radix
   hit rate) that share thresholds with the dashboards
   (`grafana/provisioning/alerting/alertrules.yml`)

## Access

Nothing in this tier publishes a public port. Reach it over an SSH tunnel:

```bash
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 <host>
# Grafana     http://localhost:3000   (admin creds from GRAFANA_ADMIN_*)
# Prometheus  http://localhost:9090
```

## Scrape targets — `prometheus/prometheus.yml`

| Job | Targets | Interval | Covers |
|---|---|---|---|
| `sglang-workers` | `qwen36-27b-r0:8001`, `qwen36-27b-r1:8002` | 5s | engine metrics (`--enable-metrics`, `--enable-mfu-metrics`, `--enable-forward-pass-metrics`) |
| `sglang-router` | `qwen36-27b-router:29000` | 15s | routing/queue/dispatch |
| `dcgm` | `dcgm-exporter:9400` | 5s | per-GPU DCGM fields (DCGM_FI_DEV_*, DCP profiling fields) |
| `node` | `qwen36-27b-node-exporter:9100` | 15s | host CPU/RAM/disk/network |
| `caddy` | `caddy:2020` | 15s | edge RED metrics — see below |
| `alertmanager` | `qwen36-27b-alertmanager:9093` | 30s | alert **delivery** health — see below |
| `redis-ledger` | `qwen36-27b-redis-exporter:9121` | 30s | the ai-quota **billing ledger** — see below |
| `clickhouse` | `qwen36-27b-langfuse-clickhouse:9363` | 30s | ClickHouse's own disk, parts, queries — see below |
| `engine-usage` | `qwen36-27b-langfuse-clickhouse:8123/engine_usage_metrics` | 60s | exact per-consumer usage gauges (`engine_usage_*`, `gateway_usage_*`), basic auth — see *Exact per-consumer usage* below |
| `quota-bot` | `quota-bot:8080` | 30s | per-key policy: tier, quota, refill date, daily/per-minute limits and use |
| ~~`higress-apiserver`~~ | — | — | **removed 2026-09-04** — see below |
| `prometheus` | `localhost:9090` | 15s | self-scrape (`up{job="prometheus"}` exempted from the down alert) |

`global` also sets `external_labels: {node: a100, stack: qwen36-27b}`, stamped
onto every series leaving this server, and an explicit `scrape_timeout: 10s`.
The explicit timeout matters because two jobs run a 5s `scrape_interval`, and a
timeout longer than the interval is a config error — it currently survives only
because Prometheus clamps it down silently.

### Caddy — the edge

This is the only job that observes requests which never reach SGLang. TLS
handshake failures and `401`s from a wrong edge key terminate at Caddy, so they
are invisible to every engine-side metric. A client misconfiguration looks like
*silence* on the engine dashboards and like a `401` rate here.

The 8 MB request-body cap was removed on 2026-08-15 (OCR sends base64 images
inline, which exceeds it on ordinary requests), so `413` is no longer a code
this tier produces. Caddy's body-size histogram is now the only place payload
growth is visible at all — nothing rejects on size before SGLang tokenises
against `--context-length 169000`.

Two pieces are required and neither is the default:

- the global `metrics { per_host }` option in the `Caddyfile` — without it
  `/metrics` serves only process and config gauges, so the target is `up`
  while exposing nothing useful. `per_host` adds the host label that
  separates inference from embedder traffic; the label set is bounded to the
  two configured hosts plus `_other`.
- a `:2020` site block serving the `metrics` handler. The admin endpoint's
  own `/metrics` is not used: it lives on the config-rewriting admin API,
  which stays bound to `localhost` inside the container.

**Network:** Prometheus attaches to `edge` *in addition to*
`qwen36-27b-backend` purely to reach this target. Caddy is deliberately not
pulled onto the backend network — that would give the internet-facing proxy a
route to the worker ports and ZMQ sockets, which is the lateral-movement path
the network split exists to close. Scrapes are outbound from Prometheus, so
this direction preserves the invariant.

### ClickHouse

Added 2026-08-11 after an audit found ClickHouse spending ~1.85 GB on
self-observation to back **396 KiB** of actual data (then Langfuse traces):

| | |
|---|---|
| `system.trace_log` | 546 MiB (26.8M rows) |
| `system.text_log` | 184 MiB |
| `part_log` + `metric_log` + `asynchronous_metric_log` | 219 MiB |
| `clickhouse-server.log` (single unrotated file) | 830 MiB |
| **`default.*` — the real payload** | **396 KiB** |

Accumulated in five days. Nothing attributed it to ClickHouse: node-exporter
reports host disk in aggregate, and the stock image ships its `<prometheus>`
section commented out, so the database exposed no metrics at all. Retention is
now bounded (see `clickhouse/config.d/`); this job is what makes a regression
visible early.

Series worth knowing:

```
ClickHouseAsyncMetrics_DiskUsed_default                  bytes on the data disk
ClickHouseAsyncMetrics_TotalBytesOfMergeTreeTables       all tables
ClickHouseAsyncMetrics_TotalBytesOfMergeTreeTablesSystem system.* only
ClickHouseMetrics_PartsActive                            merge backlog
ClickHouseProfileEvents_FailedQuery                      cumulative failures
```

The gap between the two `TotalBytesOfMergeTree*` series is the real payload. If
the `System` line dominates again, the failure mode has returned.

30s interval, not 15s: ClickHouse renders ~3,000 metrics per scrape — the
heaviest exposition in this file and the least time-sensitive.

### DCGM

`dcgm-exporter` runs a **custom counter file** (`tuning/prometheus/dcgm-counters.csv`,
mounted into the container) instead of the image default, so it exposes the
DCP profiling fields that matter for the tuning work — SM occupancy, DRAM
activity, tensor-pipe activity — not just the basic `DCGM_FI_DEV_*` set.
Sampling is 1000 ms (`--collect-interval 1000`). The container needs
`SYS_ADMIN` for the DCP metric group.

> See `tuning/prometheus/README.md` for the CUPTI/nsys counter-contention
> warning — relevant before any profiling run.

## Prometheus-native alerts — `prometheus/alerts.yml`

Infra/GPU health rules. These are the **operational signal** (page-worthy
events). 15 rules in six groups — 17 until the `413` rule went with the
request-body cap on 2026-08-15, and 16 until `GpuMemoryPressure` was removed on
2026-09-04 (see below):

**`endpoints` / `gpu` / `host`**

- `PrometheusTargetDown` — any scrape target `up == 0` for 2m (critical)
- `GpuHighTemperature` — DCGM temp > 85°C for 5m (critical)
- `GpuXidError` — any XID error increase in 5m (critical)
- `HostLowDiskSpace` — root fs < 10% free for 10m (warning)
- `HostMemoryPressure` — host RAM > 95% for 10m (warning)

**`edge`** — failures that terminate at Caddy and reach no upstream, so no
engine-side rule can see them:

- `EdgeUpstreamUnhealthy` — `caddy_reverse_proxy_upstreams_healthy == 0` for 2m (critical)
- `EdgeAuthRejectionRate` — 401s > 0.2/s for 10m (warning)
- `EdgeServerErrors` — 5xx > 0.05/s for 5m (critical)
- `EdgeConfigReloadFailed` — running edge diverged from the Caddyfile (warning)

**Usage records** — the per-request pipeline (engine file → Vector →
ClickHouse → scrape) is asynchronous, so nothing in the request path notices
when it breaks; these are the signals. (The `tracing` group went with Langfuse
on 2026-09-14.)

- `EngineUsageScrapeDown` — the `engine-usage` scrape failing (warning); bot usage screens say "usage data unavailable"
- `EngineUsageRecordsStale` — the engine served requests in the last 15m but no new record arrived for 15m (warning); Vector or ClickHouse stopped ingesting

### Two things removed on 2026-09-04

Both were permanently true, and neither was noticed until Alertmanager started
delivering. An alert that always fires is not monitoring; it is training to
ignore the channel.

**`higress-apiserver` scrape job.** Anonymous scrapes returned 403 from the day
`--auth-enabled` landed. It could not be re-armed with a scoped credential —
measured:

| Identity | `/metrics` |
|---|---|
| anonymous | 403 |
| `CN=higress, O=system:masters` | 200 |
| any other CA-signed identity | 403 |

The apiserver authorizes rather than merely authenticating, and serves no
`rbac.authorization.k8s.io` group, so the only credential that can scrape it is
the shared cluster-admin one — which also reads every consumer's plaintext API
key out of the key-auth object. Seven series does not justify putting that
credential inside Prometheus. The question the job was meant to answer is better
answered by `pilot_xds_*`, which covers the hop that actually matters.

**`GpuMemoryPressure`.** Alerted on framebuffer > 95%, which is this node's
healthy steady state: the replicas run `--mem-fraction-static 0.92`, so SGLang
preallocates weights and KV pool at startup and holds them for the process
lifetime. Measured at removal: GPU 0 at 98.5%, GPU 1 at 97.6%, both idle to
normal. The rule's "OOM risk on next request burst" had the causality backwards
for a preallocating server — real exhaustion happens *inside* that reservation,
and the Grafana SLO rules on KV pool utilisation and full token usage already
cover it against `sglang:*` series that actually move.

**`storage`**

- `ClickHouseDiskFilling` — data disk > 85% for 15m (warning)
- `ClickHouseSelfObservationDominates` — `system.*` above 5 GB for 30m (warning)
- `ClickHouseQueryFailures` — > 0.1 failed queries/s for 10m (warning)

Error-rate expressions use `... or vector(0)` and `clamp_min(...)` on
denominators, so a healthy system renders `0` rather than "No data" and a
traffic lull cannot produce a fake ratio spike.

## Alert delivery — `alertmanager/alertmanager.yml`

Added 2026-09-04. Until then **this tier evaluated 24 rules and delivered none
of them.** There was no `alerting:` block in `prometheus.yml`, no Alertmanager,
and the Grafana rules were evaluation-only with no contact point. A Langfuse
ingest outage ran for roughly fourteen hours with `TraceExportFailing` firing
correctly the entire time and nobody told. Evaluation is not monitoring.

```text
Prometheus (15 rules) ─┐
                       ├─► Alertmanager ──webhook──► quota-bot /alert ──► Telegram
Grafana (7 SLO rules) ─┘        :9093                    (bearer auth)
```

**Network placement is the non-obvious part.** Alertmanager runs on `edge` and
*not* on `qwen36-27b-backend` with the rest of the metrics tier. It has exactly
two conversations — Prometheus sends to it, it sends to quota-bot — and the bot
lives on `edge`/`higress-net`/`higressint`. Moving the bot to the backend
instead would give an internet-reachable service a route to the worker ports and
ZMQ sockets, which is the lateral-movement path the network split exists to
close. So Alertmanager comes to the bot, and ends up with strictly less reach
than Prometheus, which spans both networks.

Routing choices worth knowing:

- `group_by: [alertname, severity]`, deliberately **not** including `instance`.
  On a two-GPU node, per-instance grouping sends the same condition twice —
  exactly the noise that trains people to mute a channel.
- `repeat_interval: 4h`, and 24h for `severity="info"`. The audience is a
  handful of operators reading a Telegram group, not a rota.
- Two inhibit rules: a down scrape target suppresses the derived warnings from
  its own now-stale series, and a critical suppresses the matching warning.
- `send_resolved: true` — half the value of the channel is learning that
  something recovered without going to look.

**The credential is not in git.** `alertmanager.yml` is committed and points at
`credentials_file: /etc/alertmanager/webhook_secret`, which is gitignored and
derived from `.env`:

```bash
./deploy/render-alertmanager-secret.sh   # 640, group nogroup: the image runs as `nobody`
```

Two traps, both hit during the build:

- **`--web.enable-lifecycle` is a Prometheus flag.** Alertmanager rejects it and
  crash-loops with `unknown long flag`. It needs no flag: `POST /-/reload` and
  `SIGHUP` both work by default.
- **`webhook_secret` at mode 600 is unreadable.** The image runs as `nobody`, so
  a file owned by the operator fails at config load.

Watch `alertmanager_notifications_failed_total{integration="webhook"}`. A bot
that is down, renamed, or rejecting the bearer looks *identical to a quiet
system* from every other angle — alerts fire, Alertmanager accepts them, and
nothing arrives. That is the same shape of silent failure one layer up that this
whole path was built to remove.

## The quota ledger — `redis_exporter` and `prometheus/rules.yml`

Added 2026-09-04. The ledger is the one source on this node that is both exact
and durable: `ai-quota` DECRBYs `chat_quota:<consumer>` after each completion,
and the volume is appendonly. Everything else per-consumer is an Envoy counter,
which is process-lifetime and resets on a gateway restart — which is why
`stats.sh` prints both and labels which is which, and why **an invoice must
never be built from Prometheus counters**.

`redis_exporter` runs with `--check-keys 'chat_quota:*'`, exporting
`redis_key_value{key="chat_quota:<name>"}` plus `redis_up`.

**The join that makes it useful.** The exporter puts the consumer name *inside*
the `key` label; every gateway metric carries it as `ai_consumer`. No `on(...)`
clause can match a label against a substring of another, so the first recording
rule rewrites it:

```promql
label_replace(redis_key_value{key=~"chat_quota:.+"},
              "ai_consumer", "$1", "key", "chat_quota:(.+)")
```

After that the durable balance and the resettable counters share a label and
can be divided by one another. From it: `consumer:quota_spend:tokens24h`,
`consumer:quota_days_left`, `consumer:token_share:ratio1h`, and three alerts.

Two things worth knowing about the rules:

- `consumer:quota_days_left` carries an `or … * 0` term. Division is a 1:1 label
  match, so a consumer with a balance and **no spend series at all** — anyone
  idle for 24h — has nothing to match and drops out of the result entirely.
  Measured before the fix: `acme` and `legacy-shared`, both funded, were simply
  absent, which on a dashboard is indistinguishable from having no quota record.
- `ConsumerQuotaExhausted` is gated on 7-day activity. A seeded-but-parked
  consumer sitting at zero is not an incident, and would otherwise fire forever
  — the precise failure that made `GpuMemoryPressure` and the apiserver job
  worthless.

### These rules have unit tests

`promtool test rules prometheus/rules_test.yml` — run it after editing either
rule file.

The alternative was draining a live consumer's balance and waiting fifteen
minutes, which breaks a paying customer to test a warning and cannot exercise
`QuotaLedgerUnreachable` at all without taking the ledger down. The tests drive
the same rule file with synthetic series and cover funded-and-idle,
burning-fast, exhausted, revoked-with-lingering-counters, and ledger-down.

They caught two real bugs during the build: the 1:1-match dropout above, and
`clamp_min` being handed a scalar where it requires an instant vector.

Two quirks of the framework, both cost time:

- **Assert on properties, not floats.** `increase()` extrapolates, so no choice
  of inputs makes a quotient exact — 0.5 came back as `0.49999999999999994`.
  Use `< bool 1`, which yields exactly 1 or 0.
- **`__name__` is part of a recorded series' label set** and must appear in
  `exp_samples`. It is absent only where `sum()` or a `bool` comparison dropped
  it.

## Exact per-consumer usage — `engine_usage_*` and `gateway_usage_*`

Every per-consumer token, request, cache and latency number the bot, admin-mcp
and the Usage & Quota board show is a plain sum or an exact quantile over
per-request rows, computed in ClickHouse by `clickhouse/engine-usage-metrics.sql`
and served as Prometheus text at `GET /engine_usage_metrics` (a predefined HTTP
handler rendered into `clickhouse/config.d/engine-metrics-handler.xml` by
`./deploy/render-engine-metrics-handler.sh`; read-only `engine_metrics` user).

```
engine_usage_{requests,prompt_tokens,completion_tokens}{window,consumer}
engine_usage_{cache_known_prompt_tokens,cached_device_tokens,cached_host_tokens}{window,consumer}
engine_usage_{prefill_seconds,decode_seconds,aborted_requests,aborted_completion_tokens}{window,consumer}
engine_usage_{ttft_seconds,e2e_seconds}{window,consumer,p="50"|"95"}
engine_usage_queue_seconds{p="95"}, engine_usage_decode_tokens_per_second{p="50"}
engine_usage_replica_completion_tokens{window,consumer,replica}
gateway_usage_requests{window,consumer,status_class}
gateway_usage_error_requests{window,consumer,cause}
gateway_usage_{rate_limited,unauthorized,quota_denied,cut}_requests{window,consumer}
gateway_usage_cut_seconds, gateway_usage_duration_seconds{p="50"|"95"|"99"}
engine_usage_{first,last}_record_timestamp_seconds{source}, gateway_usage_last_record_timestamp_seconds
```

`window` is 1h, 24h, 7d or 30d before the scrape. They are **gauges** — read the
latest value; `rate()` or `increase()` over them is meaningless. Why not
counters: see `docs/METRICS-ECOSYSTEM.md` (the lost first request and
`increase()` extrapolation, measured). They replaced Vector's
`gateway_requests_total` / `gateway_tokens_total` / `gateway_request_duration_seconds`
on 2026-09-15.

`consumer="unauthenticated"` is the 401 path on the gateway side; engine traffic
that bypassed the gateway has no consumer label. The SQL's cut-off predicate has
regression cases: `./deploy/test-usage-sql.sh`.

### Why requests failed — `gateway_usage_error_requests{cause}`

Added 2026-09-17. One named cause per gateway request that did not end in a full
answer; requests without a cause are not exported. Status codes win over
response flags. The bot's `/errors` and the `serving` alerts read it.

| cause | rule | usually means |
|---|---|---|
| `timeout` | 408 or 504 | nothing sent for 900 s (stream idle timeout) |
| `server_error` | other 5xx | router or engine failure |
| `left_before_reply` | billable, charged nothing, status 0, flag `DC` | client gave up while queued or in prefill |
| `left_mid_answer` | billable, charged nothing, flag `DC`, a status was sent | client cancelled a running answer |
| `cut_mid_answer` | billable, charged nothing, flag `SI`/`UC`/`UPE`/`UT` | stream cut upstream |
| `no_reply` | status 0 without `DC` | connection ended with no answer |
| `bad_request` | 400 | engine validation: prompt + max_tokens over 169K, bad parameter (no reason is logged) |
| `max_tokens` / `too_large` | 422 / 413 | gateway caps |
| `rate_limited` | 429 | token limit, or router queue full |
| `no_balance` / `no_key` / `not_found` | 403 / 401 / 404 | gateway refusals |
| `client_error` | any other 4xx | |

The same expression is `RequestSql.ErrorCause` in `admin-mcp/mcp.cs`, which
lists one key's failed requests for the bot. `clickhouse/error-cause_test.sql`
pins the cases; change the three together.

The direct hostname has no access-log row, so `/errors` shows Caddy's per-host
status counters for it (`caddy_http_request_duration_seconds_count{host}`),
approximate and without disconnects.

## Serving alerts — `serving` group in `prometheus/rules.yml`

Added 2026-09-17 after the two failures of 2026-09-16. Unit tests:
`promtool test rules prometheus/serving_rules_test.yml` (15 s evaluation, so
the `for:` clauses are exercised).

| rule | fires when | 2026-09-16 case |
|---|---|---|
| `node:engine_queue_stranded:bool` (recording) | 1 while a replica has a queue and another has no queue and fewer than 4 running | 14:20-14:28, r1 queued behind a 32K-token answer, r0 idle |
| `EngineQueueStrandedOnOneReplica` | the above for 2 min | same |
| `EngineQueueNotDraining` | a replica's queue has not been empty in any 1 min slice for 10 min | 11:11-11:58 orphan stall |
| `GatewayClientsLeavingBeforeReply` | 8 or more `left_before_reply` for one key in the last hour | 69 at 11:00, 8 at 12:00 |
| `GatewayTimeouts` | any `timeout` in the last hour | 12 at 11:00 |
| `GatewayServerErrors` | 3 or more `server_error` in the last hour | — |

"4 running" is `--max-running-requests`, which the engine does not export; keep
the rule in step with docker-compose.yml. The gateway alerts read 1-hour gauges,
so they resolve up to an hour after the last bad request.

Backtest over 2026-09-10..17: 76 stranded minutes (54 with the other replica
idle); `EngineQueueNotDraining` would have fired once per replica, both on
2026-09-16 (r0 from 11:23 for 18 min, r1 from 11:25 for 10 min), and never
otherwise. The bot's `/errors` prints stranded minutes from the recording rule
(`sum_over_time(...) * 15 / 60`), which only counts from the rule's deployment.

### Quota is a single total-token balance

Worth stating plainly, because every surface now shows the split and it would be
easy to assume the budget does too. It does not.

`chat_quota:<consumer>` is one number. ai-quota deducts input+output from it at
the same rate, and that is not configurable — at v2.2.4 the plugin is literally:

```go
totalToken := int(inputToken + outputToken)
config.redisClient.DecrBy(config.RedisKeyPrefix+consumer, totalToken, nil)
```

`QuotaConfig` exposes no weighting or selection field. `/newkey`, `/topup` and
`/setquota` all set or move that one total, via `POST /v1/chat/completions/quota`
`{,/refresh,/delta}`. **There is no separate input or output budget and no way
to express one without forking the plugin.**

That matters because the two directions cost nothing alike here. An output token
is a forward pass, and decode on this node is bandwidth-bound at ~50ms. An input
token is prefilled in one pass and, for a multi-turn agent re-sending its
context, usually a radix-cache hit — node-wide, 84.8% of prompt tokens have
been cache hits (190.6M of 224.7M).

Measured on real traffic, input tokens charged per output token produced:

| consumer | i:o |
|---|---|
| testafter (agent) | 47.8 |
| testone | 17.0 |
| quota-admin (synthetic probes) | 1.3 |

So an agent consumer pays roughly forty times over for context replay that the
cache largely serves for free, at the same rate as a consumer doing genuinely
generative work. The recording rules
`consumer:quota_spend:{input,output}24h` and
`consumer:token_ratio:in_per_out24h` exist so the bot, the dashboard and any
pricing conversation read one set of numbers. Nothing here changes what is
charged; it makes the basis visible.

Per-request cached-token counts are NOT obtainable: SGLang returns
`prompt_tokens_details: null`, so ai-statistics' `input_token_details` is always
`{}`. The 84.8% is node-wide only.

### The histogram is an approximation; the fact table is exact

Two things to know before trusting a p95 from Prometheus here.

**Buckets must fit LLM latency.** The exporter's defaults stop at `le="10"`,
which is right for an HTTP API and wrong for this workload. Caught on the first
real agent traffic: a consumer whose exact p95 was 16.7s and whose slowest
request took 35.7s reported a Prometheus p95 of exactly **10.00s**.
`histogram_quantile` cannot interpolate past the highest finite bucket, so a
saturated histogram pins to its top edge and reads like a healthy number.
Buckets now run to 300s.

**Changing buckets invalidates comparison across the change.** Old and new
`le` series coexist until the old ones go stale, and `sum by (le)` merges them
— so a query spanning a bucket change silently mixes two different histograms.

**And it stays coarse at low n.** Measured right after the change, with three
observations: Prometheus said 29.00s where the exact p95 was 19.45s. That is
bucket interpolation, not a bug.

So: Prometheus for trends and alerting, `gateway.requests` in ClickHouse for
any number that has to be right:

```sql
SELECT consumer, count() n,
       round(quantile(0.95)(duration_ms)/1000, 2) AS p95_s,
       round(max(duration_ms)/1000, 2)            AS max_s
FROM gateway.requests FINAL
WHERE ts > now() - INTERVAL 1 DAY
GROUP BY consumer;
```

## Grafana dashboards — `grafana/provisioning/dashboards/`

Dashboards, one folder each, auto-provisioned (read-only):

| Folder | File | Content |
|---|---|---|
| `overview` | `qwen36-27b-overview.json` | home dashboard: requests, tokens/s, latency SLOs, cache hit rate |
| `sglang` | `qwen36-27b-sglang-engine.json` | engine-level: prefill/decode, KV pool, mamba pool, spec accept rate |
| `router` | `qwen36-27b-router.json` | router queue/dispatch, per-worker split |
| `gpu` | `qwen36-27b-gpu-dcgm.json` | per-GPU util, power, clocks, occupancy, memory |
| `host` | `qwen36-27b-host.json` | node-exporter: CPU, RAM, disk, network, load |
| `edge` | `qwen36-27b-edge-caddy.json` | Caddy: traffic, 401/5xx, TTFB, body sizes, upstream health |
| `usage` | `qwen36-27b-usage-quota.json` | balances, burn rate, days-left, share of node, exact usage and status mix per consumer (Window picker), cache and latency — the operator's board |

**The checked-in JSON is the single source of truth** — edit it directly.
`allowUiUpdates: false`, so browser edits are overwritten on the next 10s scan
and are never written back to these files. (An earlier note here pointed at a
generator script under `/tmp`; it was lost with the tmpfs and following it
would have silently discarded edits.)

### Edge Gateway dashboard

Two panels carry most of the value and are easy to misread:

- **Time to first byte** (`caddy_http_response_duration_seconds`) is the
  meaningful edge latency number. For a streaming completion it is TTFT as the
  client experiences it — queueing and prefill included, plus the edge hop no
  engine metric can see.
- **Full request duration** (`caddy_http_request_duration_seconds`) under SSE
  covers the *entire generation*, so it tracks output length, not edge health.
  The gap between the two is decode time.

One label quirk worth knowing: `caddy_http_requests_total` carries **no `code`
label**. Status-code breakdowns come from the duration histogram's `_count`
series instead — same numerator, different series.

## Grafana SLO alerts — `grafana/provisioning/alerting/alertrules.yml`

Seven rules in the `qwen36-27b SLO` folder, **evaluation-only** — they
evaluate against the Prometheus datasource (UID `prometheus`) and surface in
the Grafana Alerting UI, but deliver no external notifications. The
Prometheus-native alerts remain the operational signal by design.

| Rule | Threshold | Severity |
|---|---|---|
| TTFT p95 | > 5s, for 2m | critical |
| ITL p95 | > 50ms, for 2m | critical |
| E2E p95 | > 30s, for 5m | warning |
| KV pool utilization | > 95%, for 5m | warning |
| Full token usage | > 95%, for 5m | warning |
| Radix cache hit rate | < 20%, for 10m | info |
| Speculative accept rate | < 40%, for 10m | info |

**Notable gotcha (fixed 2026-08):** in Grafana threshold rule expressions the
math node must reference the prior query as `$A` and the threshold node must
reference the math result as `B` (no `$`). `A` → `parseError ... non existent
function A`; `$B` → `missingDependentNode ... could not find dependent node
[$B]`.

To add delivery later: create a contact point, set it on the default
notification policy, and remove the per-rule `notification_settings`.

## Grafana datasource

`grafana/provisioning/datasources/datasources.yml` pins `uid: prometheus`
(not an auto-generated UID) so alert rules and dashboards reference the
datasource deterministically across volume resets. `editable: false`,
`httpMethod: POST`, `timeInterval: 5s` matching the worker/DCGM scrape.

`prometheusVersion` is pinned alongside `prometheusType` — Grafana gates PromQL
features on it and assumes an old server when it is unset, hiding newer
functions from the query builder. Keep it in step with the image pinned in
`docker-compose.metrics.yml`.

`incrementalQuerying: true` (10m overlap window) makes a dashboard refresh
re-query only the newly elapsed slice instead of the whole range. These boards
refresh at 10s over 1–6h windows, so it is the difference between re-reading
six hours of samples every ten seconds and reading ten seconds of them.

**Stale-datasource recovery:** if the Grafana DB volume holds a stale
Prometheus row with an auto-UID, full provisioning fails with `data source
not found`. Stop grafana, `DELETE FROM data_source WHERE id=1` in the sqlite
DB (`/data/grafana.db` in the `grafana_data` volume), start again —
provisioning recreates it with the pinned UID.

## Changing config without restarting

Prometheus runs with `--web.enable-lifecycle`, so scrape and rule changes apply
in place:

```bash
curl -X POST http://localhost:9090/-/reload
```

Validate first — a bad rule file is rejected wholesale, not partially:

```bash
docker run --rm --entrypoint promtool \
  -v "$PWD/prometheus:/etc/prometheus:ro" prom/prometheus:v3.13.1 \
  check config /etc/prometheus/prometheus.yml
```

### Trap: single-file bind mounts go stale on edit

`./prometheus` is mounted as a **directory**, deliberately. Mounting the two
files individually breaks in a way that is silent and actively misleading.

A single-file bind mount resolves to an inode when the container starts. Most
editors — and `sed -i`, and anything that writes a temp file and renames it —
replace the file rather than truncating it, leaving the container bound to the
old, now-unlinked inode. The host file changes; the container never sees it.

What makes it nasty is that every health signal says success. `/-/reload`
returns `200` and `prometheus_config_last_reload_successful` stays `1`, because
Prometheus genuinely did reload the stale file it can see. Hit while adding the
`edge`/`tracing`/`storage` groups: `promtool` validated 17 rules on the host
while the container kept evaluating the original 6.

Confirm a suspected case by comparing inodes:

```bash
stat -c '%i %s' prometheus/alerts.yml
docker exec qwen36-27b-prometheus stat -c '%i %s' /etc/prometheus/alerts.yml
```

Different inode means the mount is stale and only a container recreate fixes
it. The same hazard applies to the remaining single-file mounts in this repo —
`Caddyfile`, `clickhouse/config.d/*.xml`. Those are left
as file mounts on purpose (mounting a directory over ClickHouse's `config.d`
would mask the image's own `docker_related_config.xml` and break its listeners),
and it is tolerable there because none of them is edited-and-hot-reloaded —
each is applied by recreating its container, which rebinds the mount anyway.

## Verification

```bash
# datasource healthy + readOnly
curl -s http://localhost:9090/-/ready
# every scrape target's health in one line each
curl -s http://localhost:9090/api/v1/targets \
  | python3 -c "import json,sys; [print(f\"{t['labels']['job']:16}{t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]"
# alert rule evaluation state (via Grafana API)
curl -s -u "$GRAFANA_ADMIN_USER:$GRAFANA_ADMIN_PASSWORD" \
  http://localhost:3000/api/prometheus/grafana/api/v1/rules
```

## Correlation — resolving one request

Rewritten 2026-09-15, when tracing was removed. Verified end to end.

### One identity everywhere

`node=a100` and `stack=qwen36-27b` are stamped on Prometheus series
(`global.external_labels`) and on `gateway.requests.node` / `.stack`.

### Gateway row to engine row — an exact key

The engine never sees the gateway's request id (SGLang generates its own 32-hex
rid and ignores caller-supplied ones —
`entrypoints/openai/serving_base.py::_generate_request_id_base` returns `None`
unconditionally). But it does return that rid to the client as the response
`id` (`meta_info["id"]`, streamed or not), the router passes it through, and the
gateway logs it as `chat_id`. So:

```sql
SELECT e.*
FROM gateway.requests AS g FINAL
JOIN engine.requests AS e FINAL ON e.rid = g.chat_id
WHERE g.request_id = '<REQUEST_ID>';
```

Verified 2026-09-15 on every charged request since the exporter went live: one
engine row each, same consumer, same token counts — including two identical
requests 400 ms apart. The bot prints the query (`/trace <request-id>`);
admin-mcp runs it (`request_detail`).

A request cut off before its response (client disconnect) has an empty
`chat_id`, 0 tokens at the gateway, and **no engine row at all**: SGLang writes
no record for a request whose client went away.

(Do not join on timestamps. A replica's tokenizer and scheduler clocks, and the
gateway's, drift apart — r0 by ~1 s on 2026-09-15.)

### What this does NOT give you

**No metric-to-request exemplars.** SGLang emits no exemplars on its histograms
and this Prometheus runs without `--enable-feature=exemplar-storage`. Jumping
from a latency spike to the slow request means: find the window, query
`engine.requests` (or `gateway.requests`) for the slow rows in it, then join as
above. No phase waterfall inside the forward pass either — the record gives
queue, prefill and decode seconds per request, not per iteration.
