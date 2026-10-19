-- Patch 6 (cache_diag.py): columns for the per-request "diag" object.
-- Apply BEFORE the Vector change: the sink runs skip_unknown_fields: false, so
-- Vector emitting these before they exist fails every insert. Nullable: rows
-- from before the roll, and rows of requests that never matched, stay NULL.
-- Then add the same six lines to clickhouse/engine-requests.sql for a fresh install.
ALTER TABLE engine.requests
    ADD COLUMN IF NOT EXISTS diag_kv     Nullable(UInt32) AFTER cached_storage,  -- F: KV match end
    ADD COLUMN IF NOT EXISTS diag_hit    Nullable(UInt32) AFTER diag_kv,         -- B: GDN-backed hit end
    ADD COLUMN IF NOT EXISTS diag_branch Nullable(UInt32) AFTER diag_hit,        -- branch state recorded at
    ADD COLUMN IF NOT EXISTS diag_msgs   Nullable(UInt32) AFTER diag_branch,     -- im_start in prompt
    ADD COLUMN IF NOT EXISTS diag_bnd    Nullable(UInt32) AFTER diag_msgs,       -- last message start <= F
    ADD COLUMN IF NOT EXISTS diag_after  Nullable(UInt32) AFTER diag_bnd;        -- message starts after F
