# Reboot checklist

Audit completed 2026-09-04. **The stack is ready for a full host restart.**

The approach chosen for boot was *resilience, not sequencing* — no systemd units,
everything on `restart: unless-stopped`, and the reasons order matters removed
rather than orchestrated. This records what was verified, and what to check
after the reboot.

## Why order does not need to be enforced

The Docker daemon restarts containers by policy on boot and **ignores
`depends_on` entirely** — that only applies to `docker compose up`. So boot
order is effectively arbitrary, and every dependency has to be survivable rather
than sequenced. Each one below was tested, not assumed.

| Dependency | If it is not ready yet | Verified |
|---|---|---|
| SGLang replicas → Vector / ClickHouse | No effect. Replicas append request records to a local file; Vector reads it with checkpoints and deletes a file only 3 days after reading it | by design |
| Vector → ClickHouse | Buffers to disk and replays. Killed CH for 123s with 6 requests in flight: 15 rows, 15 unique ids, no gap, no duplicates | fault-injected |
| redis-exporter → higress-redis | Stays running, reports `redis_up 0` | fault-injected |
| Prometheus → any target | Target shows down; scraping is pull and retries forever | by design |
| Alertmanager → Prometheus | Prometheus pushes with retry; a late Alertmanager loses only that window | by design |
| Prometheus `engine-usage` scrape → ClickHouse | Target down until ClickHouse is healthy; usage screens in the bot say "usage data unavailable" rather than showing zero | fault-injected 2026-09-15 (ClickHouse recreate) |
| controller, plugin-server → prepare | **Was a landmine.** `prepare` 403'd against the apiserver from the day `--auth-enabled` landed, and both gate on `service_completed_successfully`. Fixed; `prepare` exits 0 | fixed and verified |
| Replicas → GPU | `nvidia-persistenced` enabled, persistence mode Enabled on both GPUs | checked |

Nothing unsequenced is left that does not converge on its own. (Until
2026-09-14 langfuse-web restarted a few times while its four data stores came
up; Langfuse is gone.)

## Fault behaviour that is deliberate, not a bug

Verified live by stopping `higress-redis`:

| | result |
|---|---|
| redis-exporter | stayed up, `redis_up 0` |
| paid gateway path | **403** — ai-quota has no fail-open |
| direct router hostname | **200** — the escape hatch works |
| after restart | balances and tiers intact |

That 403 is the documented design: losing the ledger takes the billable routes
down rather than serving unmetered. The direct hostname exists precisely so
there is still a way in.

## What must exist on disk

These are gitignored. They **survive a reboot** — this list matters for a
rebuild on a fresh clone, not for the restart itself.

| File | If missing | Regenerate |
|---|---|---|
| `.env` | nothing starts | restore from backup — no other source |
| `alertmanager/webhook_secret` | Alertmanager crash-loops at config load | `grep '^ALERT_WEBHOOK_SECRET=' .env \| cut -d= -f2- > alertmanager/webhook_secret && chmod 644 $_` |
| `vector/secrets.json` | Vector 403s against ClickHouse | `./deploy/render-vector-secrets.sh` |
| `prometheus/secrets/engine_metrics_clickhouse.pass` | `engine-usage` scrape 401s; bot usage screens go empty | `./deploy/render-prometheus-secrets.sh` |
| `higress-standalone/consumers.conf` | no consumer can authenticate | restore from backup |
| `quota-bot/.env` | bot will not start | restore from backup |

## Networks

Four bridges must exist. Docker persists them across a reboot; they are only at
risk from `docker network prune` while everything is down.

| Network | Owner | Notes |
|---|---|---|
| `edge` | nobody | created once by hand |
| `higressint` | nobody | the compose project that made it is gone |
| `higress_higress-net` | higress project | |
| `qwen36-27b-backend` | this project | |

**Coupling:** the metrics overlay declares `higressint` as external, because
Prometheus and the redis-exporter reach the quota ledger on it. If `higressint`
disappears, the metrics stack no longer starts. (The ClickHouse overlay dropped
its own reference with the OTel collector on 2026-09-15.)

## Before rebooting

```bash
# 1. no drift: compose should want to recreate nothing
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
               -f docker-compose.clickhouse.yml up -d --dry-run 2>&1 | grep -c Recreate
# expect: 0

# 2. nothing in flight
curl -s --data-urlencode 'query=sum(sglang:num_running_reqs)+sum(sglang:num_queue_reqs)' \
  localhost:9090/api/v1/query
```

Never pass `--remove-orphans` — it would delete `qwen3-emb`, which belongs to
another project.

## After rebooting

```bash
# every container up, none unhealthy or restarting
docker ps --format '{{.Names}}\t{{.Status}}' | grep -iE 'unhealthy|Restarting'   # expect empty

# 14/14 scrape targets (engine-usage included)
curl -s localhost:9090/api/v1/targets | python3 -c "import json,sys; \
ts=json.load(sys.stdin)['data']['activeTargets']; \
print(len([t for t in ts if t['health']=='up']),'/',len(ts))"

# alerts should settle to 0 once targets are back
curl -s localhost:9090/api/v1/alerts | python3 -c "import json,sys; \
print(len(json.load(sys.stdin)['data']['alerts']))"

# the four things that must survive, not just restart:
docker exec higress-redis redis-cli --scan --pattern 'chat_quota:*' | wc -l   # 5 balances
docker exec higress-redis redis-cli --scan --pattern 'chat_tier:*'  | wc -l   # 3 tiers
# fact table still growing, and no gap across the reboot
docker exec qwen36-27b-langfuse-clickhouse clickhouse-client --password "$PW" \
  -q "SELECT count(), max(ts) FROM gateway.requests FINAL"
# engine request records still arriving (after the first real request below)
docker exec qwen36-27b-langfuse-clickhouse clickhouse-client --password "$PW" \
  -q "SELECT count(), max(finished_at) FROM engine.requests FINAL WHERE source='engine'"
```

Then one real request end to end, which exercises auth, quota, routing, the
engine, the access log and the engine's request records at once:

```bash
curl -H "Authorization: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}' \
  https://gateway.example.org/v1/chat/completions
```

It should return 200, add one row to `gateway.requests` and one to
`engine.requests` with the right consumer and identical token counts, and
decrement that consumer's balance by exactly that count.

## Expected noise, so it is not mistaken for damage

- **A burst of alerts then silence.** `PrometheusTargetDown` will fire for
  whatever is slowest to return and resolve itself. Both directions deliver to
  Telegram.
- **Envoy counters reset to zero.** `/usage`, `/top` and the Grafana rate panels
  restart from nothing; `increase()` over a window spanning the reboot
  under-reports. The Redis balances and the ClickHouse fact table do not reset —
  that asymmetry is the whole reason billing reads the ledger.
- **Vector re-reads from its checkpoint,** not from the top of the file. A
  handful of duplicate rows would be collapsed by ReplacingMergeTree anyway.

## Result — reboot performed 2026-09-04 20:40 local (15:40:29Z). PASSED.

**Full recovery in 5m14s**, bounded entirely by model load. Nothing needed a
hand.

| Check | Result |
|---|---|
| Containers running | 27/27 |
| Crash loops | none — `restarts=0` everywhere except langfuse-web at 1 |
| Scrape targets | 14/14 up |
| Alerts | 2 × `PrometheusTargetDown` while replicas loaded, then **0** |
| Quota balances | 5 keys intact |
| Consumer tiers | 3 keys intact |
| Fact table | 63 rows, 63 unique ids, **0 duplicates** |
| Trace ingest | writing, collector queue 0 |
| Wasm plugins | 8 counters, **none at zero** — no fail-open |
| Replicas ready | 15:45:43Z, `restarts=0` on both |

End to end through every layer — auth, quota, routing, engine, access log,
Vector, ClickHouse, ledger:

```
HTTP 200 in 0.254s     usage: 54 prompt + 6 completion = 60
ledger delta:  60      fact table row: quota-admin / 60 / 200 / 250ms
```

The ledger moved by exactly the token count and the fact table agrees. Billing
reconciles across a cold boot.

### Every prediction held

- **langfuse-web restarted once** (47 error lines) waiting for its data stores,
  then settled — exactly as written above.
- **`PrometheusTargetDown` fired and resolved itself** as the replicas came up.
- **Envoy counters reset**: 1,075,183 → 60. The Redis balances and the
  ClickHouse rows did not. That asymmetry is precisely why billing reads the
  ledger and never the counters.
- **Vector resumed from its checkpoint** with no gap and no duplicates across
  the restart.
- **`prepare` did not re-run**, correctly: `restart: on-failure` means a
  container that exited 0 stays exited, and pilot's config files persist on
  disk. Nothing needed re-materialising.

### The one thing worth remembering

Recovery is bounded by weight loading and CUDA graph capture — about five
minutes — during which the replicas are down, two targets are red and an alert
fires. That is normal and self-clearing. Do not intervene before ~6 minutes.
