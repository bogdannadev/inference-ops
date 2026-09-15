-- One-time backfill of engine.requests from gateway.requests for the period
-- before SGLang's per-request exporter was enabled (first engine row:
-- 2026-09-13 19:46:45 UTC on r1; r0 followed at 21:16:54 and no billable
-- gateway request fell between the two). Tokens per request are identical in
-- both tables (verified request by request 2026-09-13), so totals over any
-- window stay exact; cache split and engine timings are unknown -> NULL.
-- Idempotent: rows already present (same finished_at + rid) are skipped.
INSERT INTO engine.requests
    (finished_at, received_at, rid, consumer, replica, is_streaming,
     prompt_tokens, completion_tokens, reasoning_tokens,
     cached_tokens, cached_device, cached_host, cached_storage,
     queue_s, ttft_s, prefill_s, decode_s, e2e_s,
     finish_type, num_retractions, source)
SELECT
    g.ts + toIntervalMillisecond(g.duration_ms) AS finished_at,
    g.ts, g.request_id, g.consumer, '', false,
    g.input_tokens, g.output_tokens, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, g.duration_ms / 1000.0,
    'unknown', 0, 'gateway-backfill'
FROM gateway.requests AS g FINAL
WHERE g.ts < toDateTime64('2026-09-13 19:46:00', 3, 'UTC')
  AND g.route IN ('ai-chat', 'ai-completions')
  AND g.total_tokens > 0
  AND g.request_id NOT IN (SELECT rid FROM engine.requests WHERE source = 'gateway-backfill');
