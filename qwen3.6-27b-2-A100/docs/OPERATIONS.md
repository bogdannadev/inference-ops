# Operations

Day-to-day operations for the qwen36-27b A100 node. Compose commands below
use the combined files; the overlays share the project so single-file compose
runs treat them as orphans.

```bash
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.metrics.yml"
COMPOSE_ALL="$COMPOSE -f docker-compose.clickhouse.yml"
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
$EDGE_API_KEY" https://$EDGE_HOST_MODEL/v1/models` — the hostname comes from
`.env`, not from this repo.

## Boot checks after every replica start

Read the boot log — the following **must** hold; if not, investigate before
serving traffic:

| Check | Expected |
|---|---|
| `max_total_num_tokens` | **182528** (DFlash2 + fp8 draft KV at mem 0.94, 2026-09-13); must stay >= 169000 |
| decode CUDA-graph `bs` | `[1, 2, 3, 4]` |
| `max_mamba_cache_size` | 43 slots |
| `available_gpu_mem` | ~2.91 GB (was 8.49 on EAGLE). Watch worker logs for `OutOfMemoryError` |
| Tree cache line | `hicache_attached=True` |
| boot to healthy | ~270 s (HiCache pins ~52 GB host RAM at boot) |

`max_total_num_tokens` was **171008** on sglang 0.5.17. The v0.5.18 upgrade
(2026-08-24) moved it to **169408** — 1,600 tokens / ~100 MiB less KV, showing
up as extra free memory (`startup_available` 8.271 -> 8.343 GB), not as a leak.
Weights (51.047 GB) and the whole Mamba pool are byte-identical across the two
builds, so this is a slightly larger pre-KV runtime workspace under torch 2.13
/ flashinfer 0.6.17 / cuDNN 9.14.

**Watch the margin.** *(2026-09-13: the DFlash2 config gives 182,528, a
13,528-token margin; the note below is the EAGLE-era history.)* 169,408 still clears `--context-length 169000`, but by
only **408 tokens** (it was 792). If a future change costs another ~500 tokens
the pool drops under the context length and long requests start failing — the
failure mode `tuning/docs/RESULTS.md` records at 137,600. There is room to buy
it back (`available_gpu_mem` is 8.34 GB); raising `--mem-fraction-static` off
0.92 is the lever, and it needs its own gated roll.

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

Removed 2026-08-08. The old `test.sh` had drifted (it still looked for a
network renamed long ago) and is being replaced by proper CI pipelines.

Until those land, verify a deployment with the benchmark harnesses, which are
maintained and were used to gate the v0.5.17 upgrade:

```bash
./benchmarks/run_worker.sh r0 <label>     # per-replica smoke + decode/TTFT
./benchmarks/byte_identity.py --help      # greedy byte-identity gate
./benchmarks/worker_ladder.py --help      # per-worker concurrency ladder
```

See `tuning/docs/UPGRADE_v0.5.17.md` for how those were combined into an
A/B gate across two builds.

**`byte_identity.py` requires a FLUSHED cache on BOTH replicas.** Learned the
hard way during the v0.5.18 roll (2026-08-24): with a warm session-radix +
Mamba `extra_buffer` cache the greedy probe is **not reproducible even against
the same build** — an untouched v0.5.17 r0 scored 1/8 against *itself*
back-to-back. `POST /flush_cache` returns "Flush cache failed." whenever the
replica has running or waiting requests; if you proceed anyway you measure
cache state, not the engine. Retry the flush until it returns "Cache flushed.",
on both replicas, then run the gate. Flushed, the same v0.5.18-vs-v0.5.17
comparison scored 8/8.

**`byte_identity.py` only gates same-shape ENGINE changes.** It compares greedy
output across two builds of the *same* weights. On a weights change it fails by
construction and the failure carries no information — gate on the boot numbers
above, `spec_accept_length` and the ladder instead. See
`tuning/docs/UPGRADE_QWEN3.8.md`, which used exactly that substitution.

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

## Usage records pipeline

Every per-key number in the bot comes from here (see `docs/METRICS-ECOSYSTEM.md`):
SGLang request files → Vector → `engine.requests` / `gateway.requests` →
`GET /engine_usage_metrics` → Prometheus job `engine-usage`.

```bash
# freshness: newest engine record, scrape health (alerts: EngineUsageRecordsStale, EngineUsageScrapeDown)
curl -s --data-urlencode 'query=time() - max(engine_usage_last_record_timestamp_seconds{source="engine"})' localhost:9090/api/v1/query
curl -s --data-urlencode 'query=up{job="engine-usage"}' localhost:9090/api/v1/query

# after editing clickhouse/engine-usage-metrics.sql
./deploy/render-engine-metrics-handler.sh        # rewrites config.d/engine-metrics-handler.xml
$COMPOSE_ALL up -d --no-deps --force-recreate langfuse-clickhouse   # handlers load at start only
./deploy/test-usage-sql.sh

# after editing vector/vector.yaml
docker run --rm --cpuset-cpus 48-55 -v "$PWD/vector:/etc/vector:ro" --entrypoint vector \
  timberio/vector:0.58.0-alpine test /etc/vector/vector.yaml /etc/vector/vector_test.yaml
$COMPOSE_ALL up -d --no-deps --force-recreate vector
```

A ClickHouse recreate makes the scrape fail for about a minute; the bot's usage
screens say "usage data unavailable" meanwhile, which is expected.

**Re-ingesting engine records** (after fixing the `engine_rows` transform).
Vector keeps each request file for 3 days after reading it
(`remove_after_secs`). A one-shot Vector with its own checkpoint directory
re-reads what is still on disk; `engine.requests` is a `ReplacingMergeTree` on
`(finished_at, rid)` by `ingest_ts`, so the new rows replace the old ones.
Done this way on 2026-09-15 to correct timings on 112 rows:

```bash
# reingest.yaml = the engine_request_files source (without remove_after_secs),
# the engine_rows transform and the clickhouse_engine sink from vector.yaml,
# data_dir /tmp/reingest, memory buffer
docker run -d --name vector-reingest --network qwen36-27b-backend --cpuset-cpus 48-55 \
  --tmpfs /tmp/reingest -v "$PWD/vector:/etc/vector:ro" -v "$PWD/reingest.yaml:/reingest.yaml:ro" \
  -v "$PWD/logs/r0/request-metrics:/var/log/sglang/r0:ro" -v "$PWD/logs/r1/request-metrics:/var/log/sglang/r1:ro" \
  timberio/vector:0.58.0-alpine --config /reingest.yaml
# wait until count() of rows with ingest_ts after the start equals the line count of the files, then
docker rm -f vector-reingest
```

## Rollback / baseline

Rollback to the pre-2026-09-13 config (EAGLE/MTP, no HiCache) — the header of
`docker-compose.yml` lists it; in both replica command blocks:

```bash
# DFLASH -> --speculative-algorithm EAGLE --speculative-num-draft-tokens 6
#          --speculative-num-steps 5 --speculative-eagle-topk 1
# drop --speculative-draft-model-path/-revision, --speculative-draft-kv-cache-dtype
# --mem-fraction-static 0.92, --chunked-prefill-size/--max-prefill-tokens 16384
# drop --enable-hierarchical-cache --hicache-ratio 3
# then
./deploy/roll-replica.sh r1 && ./deploy/roll-replica.sh r0
```

## Known warnings (no action)

- `Disable prefill CUDA graph because some layers do not apply Standard GQA`
  — expected, and it does **not** mean prefill runs eager. The target model
  captures prefill fine (boot log: `Capture target prefill CUDA graph begin.
  backend=breakable, num_tokens=[4..2048]` → `elapsed=59.73 s, mem usage=1.23
  GB`). The message comes from the EAGLE draft runner, where the hybrid's 16
  attention layers are fewer than its 64 hidden layers, so
  `cuda_graph_setup.py:337` bails. Corrected 2026-08-19 — this entry previously
  quoted `prefill.backend='disabled'`, which is not the line this build emits.
- `Multiple NUMA nodes found for GPU 0: [0, 1]. Using the first one.` — keep
  `cap_add: [SYS_NICE]`.
- Transformers deprecation warnings (`use_fast`, `torch_dtype`) — upstream
  noise, revisit on image upgrade.
- Mixed-chunked-prefill disabled message — expected with speculative decoding.

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
