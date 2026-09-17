-- ============================================================================
-- The request fact table — one row per gateway request.
-- Applied by ./deploy/apply-gateway-schema.sh
--
-- This is the join that Prometheus cannot make. The gateway counters answer
-- "how many tokens did this consumer use" and reset on restart; the Redis ledger
-- answers "what is the balance" and holds no history. This table answers
-- "which request, by whom, cost what, and how long did it take" — durably, per
-- request, with the request_id that also appears on the SGLang spans in
-- Langfuse.
--
-- A SEPARATE DATABASE, not Langfuse's `default`. Same server, because standing
-- up a second ClickHouse for a few thousand rows a day would be absurd, but the
-- schemas stay apart: Langfuse owns `default` and rewrites it across version
-- upgrades, and this table must not be caught in that. The instance also has a
-- documented history of unbounded growth (1.85 GB of self-observation backing
-- 396 KiB of data), which is why the TTL below is explicit rather than assumed.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS gateway;

CREATE TABLE IF NOT EXISTS gateway.requests
(
    -- Envoy's start_time, the moment the request arrived.
    ts              DateTime64(3, 'UTC'),

    -- The join key to Langfuse. Envoy mints x-request-id; the SGLang router is
    -- started with --request-id-headers x-request-id and stamps it onto spans
    -- as attributes.request_id. This column is what lets a slow trace be
    -- attributed to a person.
    request_id      String,

    -- Set by key-auth from the presented credential. Empty on the 401 path,
    -- because there was no valid consumer to name.
    consumer        LowCardinality(String),

    route           LowCardinality(String),
    model           LowCardinality(String),
    status          UInt16,

    -- Whole-request wall time as Envoy saw it, including the edge hop.
    duration_ms     UInt32,
    -- ai-statistics' own view of the upstream LLM call. The gap between the two
    -- is gateway + network, which is the number that says whether the extra hop
    -- costs anything.
    llm_ms          UInt32,

    input_tokens    UInt32,
    output_tokens   UInt32,
    total_tokens    UInt32,

    -- ai-statistics derives these; chat_id is a conversation identity and
    -- chat_round its turn number.
    chat_id         String,
    chat_round      UInt16,

    upstream        LowCardinality(String),
    authority       LowCardinality(String),
    response_flags  LowCardinality(String),
    user_agent      String,
    -- First x-forwarded-for entry: the client as Caddy saw it. Added
    -- 2026-09-17 by ALTER (see deploy/apply-gateway-schema.sh); older rows
    -- are empty.
    client_ip       String DEFAULT '',

    -- When Vector shipped it. A large ingest_ts - ts gap means the tailer fell
    -- behind or replayed a backlog, which is worth being able to see.
    ingest_ts       DateTime64(3, 'UTC') DEFAULT now64(3),

    -- Cheap skip index. `consumer` is not in the sorting key (time is, because
    -- every query filters on a window first), so per-consumer filters would
    -- otherwise read every granule in the partition.
    INDEX idx_consumer consumer TYPE set(0) GRANULARITY 4,
    INDEX idx_status   status   TYPE minmax  GRANULARITY 4
)
-- ReplacingMergeTree, not MergeTree, and this is a deliberate safety net rather
-- than a preference.
--
-- Vector's ClickHouse sink is at-least-once: if a batch is acknowledged late or
-- a retry follows a partial write, the same line can land twice. A replayed row
-- is byte-identical, so it collapses onto the same (ts, request_id) sorting key
-- and the engine drops the duplicate on merge.
--
-- Dedup is EVENTUAL. Aggregations that must be exact before a merge has run
-- should use FINAL — cheap here, where a day is a few thousand rows.
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (ts, request_id)

-- 180 days. Long enough to answer a billing dispute or reconstruct a quarter,
-- short enough that this cannot become the next unbounded table on an instance
-- that has already had one. DELETE, not TO VOLUME: there is one disk.
TTL toDateTime(ts) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;
