# Operations

Day-to-day operations for the qwen36-27b A100 node. Compose commands below
use the combined files; the overlays share the project so single-file compose
runs treat them as orphans.

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.metrics.yml"
COMPOSE_ALL="$COMPOSE -f docker-compose.langfuse.yml"
```

## Status

```bash
$COMPOSE ps
# r0, r1 should be (healthy); router (healthy); caddy up
$COMPOSE logs --tail=100 <service>
```

## Health checks

Worker `/health` is a **generation probe**, not a liveness check — parameters
are deliberately loose so a worker mid-prefill does not flap. Wait for the
"fired up and ready to roll" line in the logs, then:

```bash
docker run --rm --network qwen36-27b-backend curlimages/curl:latest \
  curl -sS http://qwen36-27b-r0:8001/health
docker run --rm --network qwen36-27b-backend curlimages/curl:latest \
  curl -sS http://qwen36-27b-r1:8002/health
docker run --rm --network qwen36-27b-backend curlimages/curl:latest \
  curl -sS http://qwen36-27b-router:8000/health
```

Over the public edge (from outside): `curl -H "Authorization: Bearer
$EDGE_API_KEY" https://qw36-27b.bnna.dev/v1/models`.

## Boot checks after every replica start

Read the boot log — the following **must** hold; if not, investigate before
serving traffic:

| Check | Expected |
|---|---|
| `max_total_num_tokens` | **171008** (EAGLE 6/5 profile) |
| decode CUDA-graph `bs` | `[1, 2, 3, 4]` |
| `max_mamba_cache_size` | 43 slots (EAGLE 6/5 profile) |
| boot to healthy | ~181 s |

Confirm live server state:

```bash
docker exec qwen36-27b-r0 sh -c \
  'curl -s -H "Authorization: Bearer $SGLANG_API_KEY" http://localhost:8001/get_server_info'
```

## Rolling a replica

**Always roll one replica at a time** — the other keeps serving. Use the
script, not a bare compose command; it deregisters the worker from the router
before stopping it and re-registers on return, removing the ~159 s
dead-worker routing window and the ~71 s wasted-capacity window.

```bash
./deploy/roll-replica.sh r1    # apply current compose config to r1
./deploy/roll-replica.sh r0    # ...then r0
```

Do **not** `docker compose restart qwen36-27b-router` after a roll — obsolete:
the router tracks workers by URL and re-adds a returning worker on its own,
and the restart drops in-flight requests on both replicas.

Never use `--remove-orphans` — it deletes `qwen3-emb`, `grafana`, `prometheus`,
`dcgm` which belong to other projects. Compose will warn that `qwen3-emb` is
an orphan; ignore it.

## Full test suite

```bash
./test.sh
```

`test.sh` (POSIX sh) exercises: network existence, no published ports on
workers/router, `/health` and `/model_info` on both workers + router,
direct worker chat, router chat, auth-negative (wrong key → 401/403/404),
streaming, long-prompt prefill, 8 concurrent requests, tool-call smoke,
`/metrics` availability, known-warning scan, severe-error log scan, and GPU
placement. Overrides via env (`NETWORK`, `MODEL`, `R0_HOST`, `R1_HOST`,
`ROUTER_HOST`, ports). Needs `.env` sourced.

## Logs

```bash
$COMPOSE logs --tail=300 qwen36-27b-r0 | grep -Ei 'exception|traceback|runtimeerror|oom|cuda error'
```

Log rotation: `json-file` driver, `max-size: 50m`, `max-file: 5` on every
service.

## Metric troubleshooting

```bash
# is every target being scraped?
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {scrapeUrl, health}'
# are the worker /metrics endpoints alive?
docker run --rm --network qwen36-27b-backend curlimages/curl:latest \
  curl -sS http://qwen36-27b-r0:8001/metrics | head -20
```

If a job is absent, check the compose overlay is running (`$COMPOSE ps`) and
that the scrape config was picked up (`/api/v1/status/config`).

## Rollback / baseline

The rollback baseline is: remove the speculative flags, roll one replica at a
time:

```bash
# remove from BOTH replica command blocks:
--speculative-algorithm EAGLE
--speculative-num-draft-tokens 6
--speculative-num-steps 5
# then
./deploy/roll-replica.sh r1 && ./deploy/roll-replica.sh r0
```

## Known warnings (no action)

- `Disable prefill CUDA graph because cuda_graph_config resolved
  prefill.backend='disabled'` — fine as long as decode `[1,2,3,4]` captures.
- `Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.` — keep
  `cap_add: [SYS_NICE]`.
- Transformers deprecation warnings (`use_fast`, `torch_dtype`) — upstream
  noise, revisit on image upgrade.
- Mixed-chunked-prefill disabled message — expected with EAGLE.

## Gotchas

- **Two compose files, always.** A bare `docker compose up -d` after the
  metrics overlay exists can drop/ignore the observability tier.
- **Digest-pinned engine.** Do not let a floating tag move one replica to a
  different build — roll one at a time and verify boot gates on each.
- **Grafana home path** is baked into `docker-compose.metrics.yml`
  (`GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=.../overview/qwen36-27b-overview.json`);
  a compose `config` validates it.
- **SGLANG_API_KEY gates /metrics** on some builds — the scrape config has the
  `authorization` block commented out and a note about a credentials_file if
  it starts returning 401s.
