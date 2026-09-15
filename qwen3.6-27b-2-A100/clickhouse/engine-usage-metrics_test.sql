-- Regression cases for the cut-off predicate in engine-usage-metrics.sql
-- (gateway_usage_cut_requests / _cut_seconds). Moved here from the Vector unit
-- tests on 2026-09-15, when the Vector counter exporter was removed and the
-- predicate moved into SQL. KEEP IN STEP with the expression there.
--
-- A request is a cut when it is billable (ai-chat, ai-completions), was charged
-- nothing (total_tokens = 0) and ended on a flag that means it never reached
-- its final usage frame: DC client disconnect, SI stream idle timeout, UC
-- upstream reset, UPE upstream protocol error, UT upstream timeout.
--
-- Run: ./deploy/test-usage-sql.sh   (throws, and exits non-zero, on a mismatch)
-- On failure, drop the throwIf line to see which cases differ.
SELECT groupArrayIf(name, is_cut != expect) AS mismatched,
       throwIf(length(mismatched) > 0, 'cut predicate mismatch: run without throwIf to see the cases') AS ok
FROM (
    SELECT name, expect,
           route IN ('ai-chat', 'ai-completions') AND total_tokens = 0
             AND match(response_flags, '(^|,)(DC|SI|UC|UPE|UT)(,|$)') AS is_cut
    FROM values('name String, route String, total_tokens UInt32, response_flags String, expect UInt8',
        ('cut stream is a cut',                    'ai-chat',        0,  'DC',     1),
        ('normal request is not',                  'ai-chat',        81, '-',      0),
        ('combined upstream flags are a cut',      'ai-completions', 0,  'UC,UPE', 1),
        ('model listing is not billable',          'ai-models',      0,  'DC',     0),
        ('403 without a cut flag is not a cut',    'ai-chat',        0,  '-',      0),
        ('charged request that then dropped is not', 'ai-chat',      50, 'DC',     0),
        ('flag must be whole, not a substring',    'ai-chat',        0,  'UCX',    0)
    )
);
