-- Regression cases for the error-cause classifier in engine-usage-metrics.sql
-- (gateway_usage_error_requests) and RequestSql.ErrorCause in admin-mcp/mcp.cs.
-- KEEP ALL THREE IN STEP.
--
-- Rows are real shapes from gateway.requests: 2026-09-16 (504 SI after 180 s,
-- status 0 DC while queued, 200 DC mid-stream) and 2026-09-17 (400 from the
-- engine's request validation).
--
-- Run: ./deploy/test-usage-sql.sh   (throws, and exits non-zero, on a mismatch)
-- On failure, drop the throwIf line to see which cases differ.
SELECT groupArrayIf(name, cause != expect) AS mismatched,
       throwIf(length(mismatched) > 0, 'error cause mismatch: run without throwIf to see the cases') AS ok
FROM (
    SELECT name, expect,
        multiIf(
            status = 401, 'no_key',
            status = 403, 'no_balance',
            status = 429, 'rate_limited',
            status = 422, 'max_tokens',
            status = 413, 'too_large',
            status = 400, 'bad_request',
            status = 404, 'not_found',
            status IN (408, 504), 'timeout',
            status >= 500, 'server_error',
            status >= 400, 'client_error',
            route NOT IN ('ai-chat', 'ai-completions') OR total_tokens > 0, '',
            status < 200 AND match(response_flags, '(^|,)DC(,|$)'), 'left_before_reply',
            status < 200, 'no_reply',
            match(response_flags, '(^|,)DC(,|$)'), 'left_mid_answer',
            match(response_flags, '(^|,)(SI|UC|UPE|UT)(,|$)'), 'cut_mid_answer',
            '') AS cause
    FROM values('name String, status UInt16, route String, total_tokens UInt32, response_flags String, expect String',
        ('full answer is no error',              200, 'ai-chat',        5389, '-',     ''),
        ('stream idle timeout is a timeout',     504, 'ai-chat',        0,    'SI',    'timeout'),
        ('engine validation 400',                400, 'ai-chat',        0,    '-',     'bad_request'),
        ('left while queued',                    0,   'ai-chat',        0,    'DC',    'left_before_reply'),
        ('left mid-stream, charged nothing',     200, 'ai-chat',        0,    'DC',    'left_mid_answer'),
        ('left after the usage frame is fine',   200, 'ai-chat',        812,  'DC',    ''),
        ('idle cut after first byte',            200, 'ai-chat',        0,    'SI',    'cut_mid_answer'),
        ('upstream reset mid-stream',            200, 'ai-completions', 0,    'UC,UPE','cut_mid_answer'),
        ('model listing disconnect is noise',    0,   'ai-models',      0,    'DC',    ''),
        ('no route is not found',                404, '-',              0,    'NR',    'not_found'),
        ('401 wins over flags',                  401, 'ai-chat',        0,    'DC',    'no_key'),
        ('403 is no balance',                    403, 'ai-chat',        0,    '-',     'no_balance'),
        ('429 is a limit',                       429, 'ai-chat',        0,    '-',     'rate_limited'),
        ('422 is max_tokens',                    422, 'ai-chat',        0,    '-',     'max_tokens'),
        ('502 is a server error',                502, 'ai-chat',        0,    'UF',    'server_error'),
        ('other 4xx',                            405, 'ai-chat',        0,    '-',     'client_error'),
        ('status 0 without DC',                  0,   'ai-chat',        0,    'UF',    'no_reply'),
        ('flag must be whole, not a substring',  200, 'ai-chat',        0,    'DCX',   '')
    )
);
