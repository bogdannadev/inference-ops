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
| SGLang replicas → otel-collector | No effect. `process_tracing_init` returns in 0.01s against a dead endpoint — the gRPC exporter is lazy | measured against the pinned image |
| Vector → ClickHouse | Buffers to disk and replays. Killed CH for 123s with 6 requests in flight: 15 rows, 15 unique ids, no gap, no duplicates | fault-injected |
| redis-exporter → higress-redis | Stays running, reports `redis_up 0` | fault-injected |
| Prometheus → any target | Target shows down; scraping is pull and retries forever | by design |
| Alertmanager → Prometheus | Prometheus pushes with retry; a late Alertmanager loses only that window | by design |
| langfuse-web → postgres/CH/redis/minio | Restarts until they are healthy; `unless-stopped` converges | observed during this session's recreates |
| controller, plugin-server → prepare | **Was a landmine.** `prepare` 403'd against the apiserver from the day `--auth-enabled` landed, and both gate on `service_completed_successfully`. Fixed; `prepare` exits 0 | fixed and verified |
| Replicas → GPU | `nvidia-persistenced` enabled, persistence mode Enabled on both GPUs | checked |

The one genuinely unsequenced risk left is cosmetic: langfuse-web may restart a
few times before its four data stores are healthy. It converges on its own.

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

**A coupling added on 2026-09-04:** the langfuse overlay now declares
`higressint` as external, because the otel-collector joined it so the gateway
could push spans. If `higressint` disappears, the metrics/langfuse stack no
longer starts — previously only the gateway cared.

## Before rebooting

```bash
# 1. no drift: compose should want to recreate nothing
docker compose -f docker-compose.yml -f docker-compose.metrics.yml \
               -f docker-compose.langfuse.yml up -d --dry-run 2>&1 | grep -c Recreate
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

# 14/14 scrape targets
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
# trace ingest alive
docker exec qwen36-27b-langfuse-clickhouse clickhouse-client --password "$PW" \
  -q "SELECT max(modification_time) FROM system.parts WHERE active AND database='default' AND table='events_core'"
```

Then one real request end to end, which exercises auth, quota, routing, the
engine, the access log and both trace paths at once:

```bash
curl -H "Authorization: $ADMIN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}' \
  https://qw38-27b-gw.duckdns.org/v1/chat/completions
```

It should return 200, add one row to `gateway.requests` with the right consumer,
and decrement that consumer's balance by exactly the token count.

## Expected noise, so it is not mistaken for damage

- **langfuse-web restarting a few times** while its data stores come up.
- **A burst of alerts then silence.** `PrometheusTargetDown` will fire for
  whatever is slowest to return and resolve itself. Both directions deliver to
  Telegram.
- **Envoy counters reset to zero.** `/usage`, `/top` and the Grafana rate panels
  restart from nothing; `increase()` over a window spanning the reboot
  under-reports. The Redis balances and the ClickHouse fact table do not reset —
  that asymmetry is the whole reason billing reads the ledger.
- **Vector re-reads from its checkpoint,** not from the top of the file. A
  handful of duplicate rows would be collapsed by ReplacingMergeTree anyway.

## Known-unverified

Nothing in this stack has survived a full host restart since the metrics work
began — five new containers, a collector on two networks, a patched `prepare`,
and cross-project dependencies have all been added and never cold-started
together. That is what this reboot tests.
