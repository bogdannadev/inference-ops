-- =============================================================================
-- engine.requests — one row per request FINISHED by SGLang. The exact usage
-- record behind every customer-facing token, cache and latency number.
--
-- Written 2026-09-14. Why a table and not Prometheus counters: the engine's
-- per-consumer counters are created by a consumer's first request after a
-- replica start, so increase() never sees that request, and increase()
-- extrapolates to the window edges. Measured 2026-09-13 on three controlled
-- requests (29,079 prompt tokens): engine increase() said 11,194, gateway
-- increase() said 29,825, the sum of per-request records said 29,079.
--
-- Source: SGLang --export-metrics-to-file (hourly JSON lines per replica),
-- shipped by Vector (vector/vector.yaml, source `engine_request_files`).
-- Rows with source='gateway-backfill' were copied once from gateway.requests
-- for the period before the exporter was enabled: their tokens are the same
-- per-request usage SGLang returned (verified request by request), but they
-- carry no cache split and no engine timings, so those columns are NULL.
--
-- Apply: ./deploy/apply-engine-schema.sh (idempotent).
-- =============================================================================

CREATE DATABASE IF NOT EXISTS engine;

CREATE TABLE IF NOT EXISTS engine.requests
(
    -- When the engine finished the request (request_finished_ts). Every window
    -- attributes a request to its finish time, the same moment the ledger is
    -- charged and the engine counters increment.
    finished_at       DateTime64(3, 'UTC'),
    received_at       DateTime64(3, 'UTC'),

    -- SGLang's request id. Unique per request; the dedup key with finished_at.
    rid               String,

    -- x-request-id-labels {"consumer": ...} set by the gateway from the key.
    -- Empty for traffic that did not come through the gateway.
    consumer          LowCardinality(String),
    replica           LowCardinality(String),
    is_streaming      Bool,

    prompt_tokens     UInt32,
    completion_tokens UInt32,
    -- Unreliable at v0.5.19 (seen above completion_tokens); kept for reference,
    -- never used for accounting. completion_tokens already includes thinking.
    reasoning_tokens  Nullable(UInt32),
    cached_tokens     Nullable(UInt32),
    cached_device     Nullable(UInt32),   -- prefix still in GPU memory
    cached_host       Nullable(UInt32),   -- HiCache: reloaded from host RAM
    cached_storage    Nullable(UInt32),   -- L3, not enabled here

    queue_s           Nullable(Float32),  -- waiting before the first forward
    ttft_s            Nullable(Float32),  -- received -> prefill finished (first token)
    prefill_s         Nullable(Float32),  -- first forward -> prefill finished
    decode_s          Nullable(Float32),  -- prefill finished -> request finished
    e2e_s             Nullable(Float32),

    finish_type       LowCardinality(String),  -- stop | length | abort | ...
    num_retractions   UInt16,

    source            LowCardinality(String),  -- engine | gateway-backfill
    ingest_ts         DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(ingest_ts)
PARTITION BY toYYYYMM(finished_at)
ORDER BY (finished_at, rid)
-- Usage evidence for a cost-sensitive customer: keep well past any billing
-- dispute window. ~1 KB per request.
TTL toDateTime(finished_at) + INTERVAL 400 DAY DELETE;
