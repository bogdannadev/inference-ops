// =============================================================================
// admin-mcp — a Model Context Protocol server for operating this inference node
//
// WHAT THIS IS FOR
//
// quota-bot already answers "what is happening" from Telegram, but it is
// deliberately blind to the per-request tables: it runs on `edge`, and
// ClickHouse (gateway.requests, the access log; engine.requests, SGLang's own
// record of every finished request) is backend-only. This process is the one
// allowed to cross that line, so an admin's client can chain "who spiked" ->
// "which requests" -> "what did the engine do" without pasting queries between
// steps — and the bot's own per-request screens are served from here too (THE
// BOT'S READ PORT, below).
//
// It is NOT a second bot. Everything here is either a read, or one of the two
// balance operations that Redis makes atomic. Key lifecycle stays in the bot —
// see WHY NO KEY LIFECYCLE below, that omission is load-bearing.
//
// PROTOCOL
//
// MCP revision 2026-07-28, Streamable HTTP, stateless. That revision removed
// protocol-level sessions, the GET stream and resumability, which is why this
// is a single POST endpoint and holds no per-client state at all. The SDK
// (v2.x, Stateless = true by default) serves it natively and still speaks
// 2025-11-25 and earlier for older clients.
//
// AUTH — TWO INDEPENDENT GATES, NEITHER SUFFICIENT ALONE
//
//   1. Caddy restricts the route to Anthropic's published egress range
//      160.79.104.0/21 (see the Caddyfile). A stolen token from anywhere else
//      never reaches this process.
//   2. This process requires a bearer token that an org admin enters once in
//      Claude (`static_headers`). Compared in constant time.
//
// Plus Origin validation, which the spec makes a MUST: a browser on a malicious
// page must not be able to drive this through DNS rebinding. Absent Origin is
// allowed (that is a non-browser client); present-and-unknown is 403.
//
// WHY NO KEY LIFECYCLE (create / revoke / set-tier)
//
// Those live in quota-bot and must stay there. Consumers are stored in one
// key-auth wasmplugin object, the Higress apiserver is file-backed and returns
// NO resourceVersion — verified 2026-09-05, a PUT is last-write-wins over the
// whole object — and quota-bot serialises its edits behind an in-process lock.
// A second writer with no shared lock and no optimistic concurrency silently
// drops one of two concurrent key creations. Balance operations do not have
// this problem: they go through the gateway's quota API, which is a Redis
// INCRBY/SET, atomic by construction. So this server does balances and the bot
// does identity.
//
// THE BOT'S READ PORT (8081)
// quota-bot shows a key's latest requests and one request end to end. It cannot
// query ClickHouse itself (edge-only, above), so it asks here, on a second
// listener that Caddy never proxies (Caddy targets :8080). That port serves
// exactly two fixed, parameterised reads under /bot/ and nothing else, behind
// BOT_READ_SECRET, which is a different secret from MCP_BEARER_TOKEN: a leaked
// bot secret reads request metadata (ids, statuses, token counts, timings — no
// prompts are stored anywhere) and cannot reach a single MCP tool. Unset, the
// port answers 404 to everything.
// =============================================================================

#:sdk Microsoft.NET.Sdk.Web
#:property PublishAot=true
#:property PackAsTool=false
#:property InvariantGlobalization=true
#:property OptimizationPreference=Size
#:property EventSourceSupport=false
#:property MetadataUpdaterSupport=false
#:property Http3Support=false
#:property TrimmerSingleWarn=false
#:property TreatWarningsAsErrors=true
#:property ILLinkTreatWarningsAsErrors=true
#:package ModelContextProtocol.AspNetCore@2.2.*

using System.ComponentModel;
using System.Globalization;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelContextProtocol.Server;

// The chiselled runtime image has no curl and no shell, so the container's
// liveness probe is the binary asking itself. Handled before any configuration
// is read: a probe must not depend on the environment being complete, or a
// misconfigured container looks dead instead of misconfigured.
if (args is ["--healthcheck"])
{
    try
    {
        using var probe = new HttpClient { Timeout = TimeSpan.FromSeconds(4) };
        var res = await probe.GetAsync("http://127.0.0.1:8080/healthz");
        return res.IsSuccessStatusCode ? 0 : 1;
    }
    catch { return 1; }
}

// ---------------------------------------------------------------- config ----
static string Req(string k) => Environment.GetEnvironmentVariable(k)
    ?? throw new InvalidOperationException($"{k} is required");
static string Opt(string k, string d) =>
    Environment.GetEnvironmentVariable(k) is { Length: > 0 } v ? v : d;

var cfg = new McpConfig(
    BearerToken:    Req("MCP_BEARER_TOKEN"),
    AllowedOrigins: Opt("MCP_ALLOWED_ORIGINS", "https://claude.ai,https://claude.com")
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .ToHashSet(StringComparer.OrdinalIgnoreCase),
    PrometheusUrl:  Opt("PROMETHEUS_URL", "http://qwen36-27b-prometheus:9090").TrimEnd('/'),
    ClickHouseUrl:  Opt("CLICKHOUSE_URL", "http://langfuse-clickhouse:8123").TrimEnd('/'),
    ClickHouseUser: Opt("CLICKHOUSE_USER", "mcp"),
    ClickHousePass: Req("CLICKHOUSE_PASSWORD"),
    GatewayUrl:     Opt("GATEWAY_URL", "http://higress:80").TrimEnd('/'),
    AdminCredential: Req("QUOTA_ADMIN_CREDENTIAL"),
    AuditPath:      Opt("AUDIT_PATH", "/data/audit.log"),
    BotUrl:         Opt("BOT_ADMIN_URL", "http://quota-bot:8080").TrimEnd('/'),
    BotSecret:      Opt("ADMIN_API_SECRET", ""),
    BotReadSecret:  Opt("BOT_READ_SECRET", ""),
    WritesEnabled:  Opt("MCP_WRITES_ENABLED", "true") == "true");

// The token is compared in constant time, so it is hashed once here rather than
// on every request. Length is checked too: a short token in the environment is
// a configuration mistake worth failing at boot rather than at 3am.
if (cfg.BearerToken.Length < 32)
    throw new InvalidOperationException("MCP_BEARER_TOKEN must be at least 32 characters");
var expectedToken = SHA256.HashData(Encoding.UTF8.GetBytes(cfg.BearerToken));

// The bot's read port. A short secret disables it rather than failing boot: the
// MCP endpoint is the reason this process exists and must not go down over it.
const int BotReadPort = 8081;
byte[]? expectedBotToken = cfg.BotReadSecret.Length >= 32
    ? SHA256.HashData(Encoding.UTF8.GetBytes(cfg.BotReadSecret)) : null;

static bool BearerMatches(HttpContext ctx, byte[] expected)
{
    var auth = ctx.Request.Headers.Authorization.ToString();
    const string scheme = "Bearer ";
    return auth.StartsWith(scheme, StringComparison.OrdinalIgnoreCase)
           && CryptographicOperations.FixedTimeEquals(
                  SHA256.HashData(Encoding.UTF8.GetBytes(auth[scheme.Length..])), expected);
}

var builder = WebApplication.CreateSlimBuilder(args);
builder.Services.AddSingleton(cfg);
builder.Services.AddSingleton(new Backends(cfg));

builder.Services.AddHttpClient("prometheus", c =>
{
    c.BaseAddress = new Uri(cfg.PrometheusUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(20);
});
builder.Services.AddHttpClient("clickhouse", c =>
{
    c.BaseAddress = new Uri(cfg.ClickHouseUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(30);
    var basic = Convert.ToBase64String(
        Encoding.UTF8.GetBytes($"{cfg.ClickHouseUser}:{cfg.ClickHousePass}"));
    c.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", basic);
});
// quota-bot's admin API. Key lifecycle goes through here rather than being
// reimplemented, because KeyStore's lock lives in that process — see the header.
builder.Services.AddHttpClient("bot", c =>
{
    c.BaseAddress = new Uri(cfg.BotUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(30);
    if (cfg.BotSecret.Length > 0)
        c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", "Bearer " + cfg.BotSecret);
});
builder.Services.AddHttpClient("gateway", c =>
{
    c.BaseAddress = new Uri(cfg.GatewayUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
    c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", cfg.AdminCredential);
});

builder.Services
    .AddMcpServer(o =>
    {
        o.ServerInfo = new() { Name = "qwen36-27b-admin", Version = "1.0.0" };
    })
    .WithHttpTransport()
    .WithTools<ReadTools>()
    .WithTools<WriteTools>()
    .WithTools<KeyTools>();

var app = builder.Build();
var log = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("mcp");

// Liveness only. Deliberately before the auth middleware and deliberately
// answering nothing about the node: it exists so compose can tell whether the
// process is up, not so the internet can.
app.MapGet("/healthz", () => Results.Text("ok"));

// ---------------------------------------------------------------- gate 2 ----
app.Use(async (ctx, next) =>
{
    // The two listeners never share a route. The bot port serves /bot/* behind
    // its own secret and nothing else; the Caddy-facing port never serves /bot/*.
    var botPort = ctx.Connection.LocalPort == BotReadPort;
    var botPath = ctx.Request.Path.StartsWithSegments("/bot");
    if (botPort || botPath)
    {
        if (!botPort || !botPath || expectedBotToken is null)
        {
            ctx.Response.StatusCode = StatusCodes.Status404NotFound;
            return;
        }
        if (!BearerMatches(ctx, expectedBotToken))
        {
            log.LogWarning("bot read port: unauthenticated request from {Ip}",
                ctx.Connection.RemoteIpAddress?.ToString() ?? "?");
            ctx.Response.StatusCode = StatusCodes.Status401Unauthorized;
            return;
        }
        await next();
        return;
    }

    if (ctx.Request.Path.StartsWithSegments("/healthz")) { await next(); return; }

    // Origin validation is a MUST in the transport spec. A non-browser client
    // sends no Origin at all, which is fine; a browser that sends one we do not
    // know is the DNS-rebinding case and gets 403, not 401 — the distinction
    // matters because 401 would tell an attacker the credential is the only
    // thing missing.
    var origin = ctx.Request.Headers.Origin.ToString();
    if (origin.Length > 0 && !cfg.AllowedOrigins.Contains(origin))
    {
        log.LogWarning("rejected origin {Origin}", origin);
        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
        return;
    }

    if (!BearerMatches(ctx, expectedToken))
    {
        log.LogWarning("unauthenticated request from {Ip}",
            ctx.Connection.RemoteIpAddress?.ToString() ?? "?");
        ctx.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return;
    }

    await next();
});

app.MapMcp("/mcp");

// ------------------------------------------------------- bot read port ----
// Rows as JSON objects of raw ClickHouse value text (null for SQL NULL); the bot
// parses and formats. Built as JsonObject and written as text, as in quota-bot's
// /admin/*: Results.Json with an anonymous type fails the AOT build (IL2026).
static JsonArray RowsJson(List<Dictionary<string, string?>> rows)
{
    var arr = new JsonArray();
    foreach (var row in rows)
    {
        var o = new JsonObject();
        foreach (var (k, v) in row) o[k] = v is null ? null : JsonValue.Create(v);
        arr.Add((JsonNode)o);
    }
    return arr;
}

static IResult JsonText(JsonObject o, int status = 200) =>
    Results.Text(o.ToJsonString(), "application/json", null, status);

static IResult ReadFailed(string error) => JsonText(new JsonObject { ["error"] = error }, 502);

app.MapGet("/bot/requests/{consumer}", async (string consumer, int? limit, Backends b,
    IHttpClientFactory http, CancellationToken ct) =>
{
    if (!Backends.SafeName(consumer)) return Results.NotFound();
    KeyValuePair<string, string>[] ps = [
        new("consumer", consumer),
        new("hours", "168"),
        new("lim", Math.Clamp(limit ?? 10, 1, 50).ToString(CultureInfo.InvariantCulture)),
    ];
    var (rows, error) = await b.ChRowsAsync(http, RequestSql.KeyLatest, ps, ct);
    return error is not null ? ReadFailed(error)
        : JsonText(new JsonObject { ["rows"] = RowsJson(rows) });
});

app.MapGet("/bot/request/{requestId}", async (string requestId, Backends b,
    IHttpClientFactory http, CancellationToken ct) =>
{
    if (!RequestSql.PlausibleId(requestId)) return Results.NotFound();
    KeyValuePair<string, string>[] ps = [new("rid", requestId)];
    var gwT = b.ChRowsAsync(http, RequestSql.GatewayRow, ps, ct);
    var enT = b.ChRowsAsync(http, RequestSql.EngineRow, ps, ct);
    await Task.WhenAll(gwT, enT);
    if ((gwT.Result.Error ?? enT.Result.Error) is { } error) return ReadFailed(error);
    return JsonText(new JsonObject
    {
        ["gateway"] = RowsJson(gwT.Result.Rows),
        ["engine"] = RowsJson(enT.Result.Rows),
    });
});

log.LogInformation("admin-mcp listening; writes={Writes} origins={Origins}",
    cfg.WritesEnabled, string.Join(",", cfg.AllowedOrigins));
app.Run();
return 0;

// ---------------------------------------------------------------- shared ----

sealed record McpConfig(
    string BearerToken, HashSet<string> AllowedOrigins,
    string PrometheusUrl, string ClickHouseUrl, string ClickHouseUser, string ClickHousePass,
    string GatewayUrl, string AdminCredential,
    string AuditPath, string BotUrl, string BotSecret, string BotReadSecret, bool WritesEnabled);

// Per-request SQL used by both an MCP tool and the bot's read port, so the two
// cannot drift. Literals with bound parameters only, like every query here.
static class RequestSql
{
    public static bool PlausibleId(string id) =>
        id.Length is >= 8 and <= 64 && id.All(c => char.IsAsciiLetterOrDigit(c) || c == '-');

    public const string GatewayRow = """
        SELECT ts, consumer, route, model, status, duration_ms, llm_ms,
               input_tokens, output_tokens, total_tokens, response_flags, chat_id, chat_round
        FROM gateway.requests FINAL
        WHERE request_id = {rid:String}
        """;

    // The engine side is bounded to the hour around the request (the gateway
    // allows 900 s), so the join never reads the whole table.
    public const string EngineRow = """
        SELECT e.finished_at, e.received_at, e.replica, e.is_streaming,
               e.prompt_tokens, e.completion_tokens, e.cached_device, e.cached_host,
               e.queue_s, e.ttft_s, e.prefill_s, e.decode_s, e.e2e_s,
               e.finish_type, e.num_retractions, e.rid
        FROM gateway.requests AS g FINAL
        INNER JOIN (
            SELECT * FROM engine.requests FINAL
            WHERE finished_at >= (SELECT min(ts) FROM gateway.requests WHERE request_id = {rid:String}) - INTERVAL 1 MINUTE
              AND finished_at <= (SELECT max(ts) FROM gateway.requests WHERE request_id = {rid:String}) + INTERVAL 1 HOUR
        ) AS e ON e.rid = g.chat_id
        WHERE g.request_id = {rid:String} AND g.chat_id != ''
        """;

    // A key's latest completion requests, each with its engine record when there
    // is one. /v1/models is left out: clients poll it and it would bury the
    // requests that cost anything. A LEFT JOIN miss leaves e.rid empty (String
    // is not Nullable) and the Nullable engine columns NULL.
    public const string KeyLatest = """
        SELECT g.ts, g.request_id, g.route, g.status, g.duration_ms,
               g.input_tokens, g.output_tokens, g.response_flags,
               e.rid, e.replica, e.cached_device, e.cached_host, e.ttft_s, e.e2e_s, e.finish_type
        FROM (
            SELECT ts, request_id, route, status, duration_ms, input_tokens, output_tokens,
                   response_flags, chat_id
            FROM gateway.requests FINAL
            WHERE consumer = {consumer:String}
              AND route != 'ai-models'
              AND ts > now() - INTERVAL {hours:UInt32} HOUR
            ORDER BY ts DESC
            LIMIT {lim:UInt32}
        ) AS g
        LEFT JOIN (
            SELECT rid, replica, cached_device, cached_host, ttft_s, e2e_s, finish_type
            FROM engine.requests FINAL
            WHERE consumer = {consumer:String}
              AND finished_at > now() - INTERVAL {hours:UInt32} HOUR - INTERVAL 1 HOUR
        ) AS e ON e.rid = g.chat_id
        ORDER BY g.ts DESC
        """;
}

// Query helpers shared by the tool classes. Everything a tool needs to reach
// lives here so the tools themselves stay readable.
sealed class Backends(McpConfig cfg)
{
    public McpConfig Cfg => cfg;

    // Consumer names reach PromQL as string literals and ClickHouse as bound
    // parameters. The PromQL side has no parameter binding, so names are
    // constrained at the door instead — this is the same rule /newkey applies.
    public static bool SafeName(string n) =>
        n.Length is > 0 and <= 40
        && n.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');

    public static bool ValidWindow(string w) => WindowSeconds(w) is not null;

    public static int? WindowSeconds(string w) => w switch
    {
        "5m" => 300, "1h" => 3600, "6h" => 21600, "24h" => 86400, "7d" => 604800, "30d" => 2592000,
        _ => null,
    };

    // Rows of a JSONCompactEachRowWithNames result as name -> raw value text
    // (null for SQL NULL). Returns the error text instead of throwing, for the
    // same reason as ChSafeAsync.
    public async Task<(List<Dictionary<string, string?>> Rows, string? Error)> ChRowsAsync(
        IHttpClientFactory http, string sql,
        IEnumerable<KeyValuePair<string, string>> parameters, CancellationToken ct)
    {
        string body;
        try { body = await ChAsync(http, sql, parameters, ct); }
        catch (Exception ex) { return ([], $"ClickHouse query failed: {Trim(ex.Message, 400)}"); }

        var rows = new List<Dictionary<string, string?>>();
        string[]? names = null;
        foreach (var line in body.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            if (JsonNode.Parse(line) is not JsonArray arr) continue;
            if (names is null) { names = arr.Select(x => x?.GetValue<string>() ?? "").ToArray(); continue; }
            var row = new Dictionary<string, string?>(StringComparer.Ordinal);
            for (var i = 0; i < names.Length && i < arr.Count; i++)
                row[names[i]] = arr[i] switch
                {
                    null => null,
                    JsonValue v when v.TryGetValue<string>(out var str) => str,
                    var n => n.ToJsonString(),
                };
            rows.Add(row);
        }
        return (rows, null);
    }

    public static double? D(Dictionary<string, string?>? row, string name) =>
        row is not null && row.TryGetValue(name, out var v)
        && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) && double.IsFinite(d)
            ? d : null;

    public async Task<JsonNode?> PromAsync(
        IHttpClientFactory http, string query, CancellationToken ct)
    {
        using var c = http.CreateClient("prometheus");
        using var r = await c.PostAsync("api/v1/query",
            new FormUrlEncodedContent([new("query", query)]), ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        return JsonNode.Parse(body);
    }

    public async Task<double?> ScalarAsync(
        IHttpClientFactory http, string query, CancellationToken ct)
    {
        var n = await PromAsync(http, query, ct);
        var res = n?["data"]?["result"]?.AsArray();
        if (res is null || res.Count == 0) return null;
        var v = res[0]?["value"]?[1]?.GetValue<string>();
        return double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
            && !double.IsNaN(d) ? d : null;
    }

    // ClickHouse over HTTP with BOUND PARAMETERS ({name:Type} plus param_name),
    // never string interpolation. Every tool in this server builds its own SQL
    // from a literal; nothing a client sends is ever concatenated into a query.
    // Returns the error as TEXT rather than throwing. The SDK turns an exception
    // into "An error occurred invoking '<tool>'", which tells the caller nothing
    // it can act on — and the caller here is a model deciding what to do next.
    // A message naming the status and the likely cause is the difference between
    // it retrying sensibly and it giving up.
    public async Task<string> ChSafeAsync(
        IHttpClientFactory http, string sql,
        IEnumerable<KeyValuePair<string, string>> parameters, CancellationToken ct)
    {
        try { return await ChAsync(http, sql, parameters, ct); }
        catch (InvalidOperationException ex) when (ex.Message.Contains("HTTP 516")
                                               || ex.Message.Contains("AUTHENTICATION_FAILED"))
        {
            return "ClickHouse refused this server's credentials. The read-only `mcp` "
                 + "user may not exist yet — it is created by "
                 + "clickhouse/users.d/mcp-readonly.xml, which needs the ClickHouse "
                 + "container recreated once. Prometheus-backed tools are unaffected.";
        }
        catch (Exception ex)
        {
            return $"ClickHouse query failed: {Trim(ex.Message, 400)}";
        }
    }

    public async Task<string> ChAsync(
        IHttpClientFactory http, string sql,
        IEnumerable<KeyValuePair<string, string>> parameters, CancellationToken ct)
    {
        // SQL goes in the BODY, settings and bound parameters go in the URL.
        // Sending the query as a form field instead corrupts it: form encoding
        // writes `>` as %3E and ClickHouse's body parser does not decode it, so
        // `ts > now()` arrives as `ts 3E+now()` and fails with a syntax error
        // pointing at a character the tool never wrote. Measured 2026-09-05.
        var url = new StringBuilder("?default_format=JSONCompactEachRowWithNames");
        // Defence in depth. The mcp ClickHouse user is readonly=2 with its own
        // capped profile; these bound a badly-shaped tool even if that user is
        // ever misconfigured, and its <constraints> stop them being raised.
        url.Append("&max_execution_time=25")
           .Append("&max_result_rows=2000")
           .Append("&result_overflow_mode=break");
        foreach (var p in parameters)
            url.Append("&param_").Append(Uri.EscapeDataString(p.Key))
               .Append('=').Append(Uri.EscapeDataString(p.Value));

        using var c = http.CreateClient("clickhouse");
        using var sqlBody = new StringContent(sql, Encoding.UTF8, "text/plain");
        using var r = await c.PostAsync(url.ToString(), sqlBody, ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        if (!r.IsSuccessStatusCode)
            throw new InvalidOperationException(
                $"ClickHouse HTTP {(int)r.StatusCode}: {Trim(body, 300)}");
        return body;
    }

    public async Task AuditAsync(string line, CancellationToken ct)
    {
        var stamped = $"{DateTimeOffset.UtcNow:O} mcp {line}\n";
        try { await File.AppendAllTextAsync(cfg.AuditPath, stamped, ct); }
        catch (Exception) { /* logged by the caller; never fails the tool */ }
    }

    public static string Trim(string s, int n) => s.Length <= n ? s : s[..n] + "…";

    public static string Num(double? v, int dp = 0) =>
        v is null || double.IsNaN(v.Value) || double.IsInfinity(v.Value)
            ? "—"
            : v.Value.ToString("N" + dp.ToString(CultureInfo.InvariantCulture),
                               CultureInfo.InvariantCulture);
}

// ============================================================================
// READ TOOLS
//
// Descriptions are written for a model, not a person: they say what the number
// MEANS and where it comes from, because a tool that returns "p95: 15.5" with
// no provenance invites the caller to compare it against a number measured
// somewhere else entirely. Two latencies exist on this node and they differ by
// the gateway filter chain, the router and two network hops.
// ============================================================================

[McpServerToolType]
sealed class ReadTools
{
    [McpServerTool(Name = "list_consumers", ReadOnly = true)]
    [Description("Every API consumer on this node with its token balance and runway. "
        + "Balances come from the Redis quota ledger, which is the billing record. "
        + "runway_days is balance divided by recent burn rate with the divisor clamped, "
        + "so a consumer that has not spent anything lately reports a runway equal to "
        + "its balance. Read a very large runway as 'idle', not as 'centuries of "
        + "budget'. Credentials are never returned by any tool here.")]
    public static async Task<string> ListConsumers(
        Backends b, IHttpClientFactory http, CancellationToken ct)
    {
        var n = await b.PromAsync(http, "consumer:quota_balance:tokens", ct);
        var rows = n?["data"]?["result"]?.AsArray();
        if (rows is null || rows.Count == 0)
            return "No consumers found. The ledger exporter may be down — call node_health.";

        var runway = await b.PromAsync(http, "consumer:quota_days_left", ct);
        var days = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var r in runway?["data"]?["result"]?.AsArray() ?? [])
        {
            var name = r?["metric"]?["ai_consumer"]?.GetValue<string>();
            if (name is not null) days[name] = r?["value"]?[1]?.GetValue<string>() ?? "";
        }

        var sb = new StringBuilder("consumer\tbalance_tokens\trunway_days\n");
        foreach (var r in rows)
        {
            var name = r?["metric"]?["ai_consumer"]?.GetValue<string>() ?? "?";
            var bal = r?["value"]?[1]?.GetValue<string>() ?? "?";
            var d = days.GetValueOrDefault(name, "—");
            sb.Append(name).Append('\t').Append(bal).Append('\t').Append(d).Append('\n');
        }
        return sb.ToString();
    }

    [McpServerTool(Name = "consumer_stats", ReadOnly = true)]
    [Description("Everything measured about one consumer over a window, as exact sums and "
        + "percentiles over per-request records — no rates, no histogram buckets. requests, "
        + "tokens_*, prefix_cache_* and engine_* come from engine.requests, SGLang's own record of "
        + "every request it finished; tokens_* are what the ledger charged. not_2xx, "
        + "rate_limited_429, quota_denied_403, cut_* and gateway_p95 come from gateway.requests, "
        + "the access log. The two latencies are NOT interchangeable: gateway_p95 is the whole "
        + "successful request as the gateway saw it, engine_* is measured inside SGLang with no "
        + "gateway, router or network in it. prefix_cache_hit covers the input whose cache split "
        + "was recorded; cache_split_coverage below 1 means part of the window predates those "
        + "records (2026-09-14). A client disconnect leaves no engine record and shows only as "
        + "cut_*. A dash means no data, which is not the same as zero. cost_usd_* is what the "
        + "same tokens would cost buying this model from OpenRouter or Alibaba Cloud — reference "
        + "prices, never what the consumer was charged; say so whenever you quote them. "
        + "Windows: 5m, 1h, 6h, 24h, 7d, 30d.")]
    public static async Task<string> ConsumerStats(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name, exactly as list_consumers reports it.")] string consumer,
        [Description("Time window: 5m, 1h, 6h, 24h, 7d or 30d. Default 24h.")] string window,
        CancellationToken ct)
    {
        if (!Backends.SafeName(consumer)) return $"Not a valid consumer name: {consumer}";
        if (window.Length == 0) window = "24h";
        if (Backends.WindowSeconds(window) is not { } secs) return $"Not a valid window: {window}";

        var led = $"{{ai_consumer=\"{consumer}\"}}";
        KeyValuePair<string, string>[] ps =
            [new("consumer", consumer), new("secs", secs.ToString(CultureInfo.InvariantCulture))];

        var engT = b.ChRowsAsync(http, """
            SELECT
                count()                                            AS requests,
                sumIf(prompt_tokens, finish_type != 'abort')       AS tokens_in,
                sumIf(completion_tokens, finish_type != 'abort')   AS tokens_out,
                sumIf(prompt_tokens, cached_tokens IS NOT NULL)    AS cache_known_in,
                sum(ifNull(cached_device, 0))                      AS cached_gpu,
                sum(ifNull(cached_host, 0))                        AS cached_hicache,
                countIf(finish_type = 'abort')                     AS aborted,
                sumIf(completion_tokens, finish_type = 'abort')    AS aborted_out,
                countIf(source = 'engine' AND e2e_s IS NOT NULL)   AS timed,
                quantileExactIf(0.95)(e2e_s, source = 'engine' AND e2e_s IS NOT NULL)     AS e2e_p95,
                quantileExactIf(0.95)(ttft_s, source = 'engine' AND ttft_s IS NOT NULL)   AS ttft_p95,
                quantileExactIf(0.95)(queue_s, source = 'engine' AND queue_s IS NOT NULL) AS queue_p95,
                countIf(source = 'engine' AND decode_s > 0 AND completion_tokens > 1)     AS decode_rows,
                quantileExactIf(0.5)(completion_tokens / decode_s,
                                     source = 'engine' AND decode_s > 0 AND completion_tokens > 1) AS decode_tok_s_p50
            FROM engine.requests FINAL
            WHERE consumer = {consumer:String}
              AND finished_at >= now64(3) - toIntervalSecond({secs:UInt32})
            """, ps, ct);
        var gwT = b.ChRowsAsync(http, """
            SELECT
                countIf(status < 200 OR status >= 300) AS not_2xx,
                countIf(status = 429)                  AS rate_limited,
                countIf(status = 403)                  AS quota_denied,
                countIf(route IN ('ai-chat', 'ai-completions') AND total_tokens = 0
                        AND match(response_flags, '(^|,)(DC|SI|UC|UPE|UT)(,|$)'))            AS cut,
                sumIf(duration_ms, route IN ('ai-chat', 'ai-completions') AND total_tokens = 0
                        AND match(response_flags, '(^|,)(DC|SI|UC|UPE|UT)(,|$)')) / 1000     AS cut_s,
                countIf(route IN ('ai-chat', 'ai-completions') AND status >= 200 AND status < 300) AS ok_chat,
                quantileExactIf(0.95)(duration_ms,
                        route IN ('ai-chat', 'ai-completions') AND status >= 200 AND status < 300) / 1000 AS gateway_p95
            FROM gateway.requests FINAL
            WHERE consumer = {consumer:String}
              AND ts >= now64(3) - toIntervalSecond({secs:UInt32})
            """, ps, ct);
        var repT = b.ChRowsAsync(http, """
            SELECT replica, sum(completion_tokens) AS output_tokens
            FROM engine.requests FINAL
            WHERE consumer = {consumer:String} AND source = 'engine' AND finish_type != 'abort'
              AND finished_at >= now64(3) - toIntervalSecond({secs:UInt32})
            GROUP BY replica ORDER BY replica
            """, ps, ct);
        var balT = b.ScalarAsync(http, $"consumer:quota_balance:tokens{led}", ct);
        var runT = b.ScalarAsync(http, $"consumer:quota_days_left{led}", ct);
        await Task.WhenAll(engT, gwT, repT, balT, runT);

        if (engT.Result.Error is { } ee) return ee;
        if (gwT.Result.Error is { } ge) return ge;
        var e = engT.Result.Rows.FirstOrDefault();
        var g = gwT.Result.Rows.FirstOrDefault();
        double? E(string n) => Backends.D(e, n);
        double? G(string n) => Backends.D(g, n);
        var timed = E("timed") is > 0;

        var tin = E("tokens_in") ?? 0;
        var tout = E("tokens_out") ?? 0;
        var known = E("cache_known_in") ?? 0;
        var gpu = E("cached_gpu") ?? 0;
        var host = E("cached_hicache") ?? 0;
        double? hit = known >= 1 ? (gpu + host) / known : null;
        double? hostHit = known >= 1 ? host / known : null;
        // For pricing: share of ALL charged input that was cached; input with
        // an unknown split is priced as uncached rather than guessed.
        var costHit = tin >= 1 ? Math.Clamp((gpu + host) / tin, 0, 1) : 0;

        var sb = new StringBuilder();
        sb.Append("consumer=").Append(consumer).Append("  window=").Append(window).Append('\n');
        sb.Append("balance_tokens=").Append(Backends.Num(balT.Result)).Append('\n');
        sb.Append("runway_days=").Append(Backends.Num(runT.Result, 1)).Append('\n');
        sb.Append("requests=").Append(Backends.Num(E("requests"))).Append("  # finished by the engine\n");
        sb.Append("tokens_in=").Append(Backends.Num(tin)).Append('\n');
        sb.Append("tokens_out=").Append(Backends.Num(tout)).Append('\n');
        sb.Append("not_2xx=").Append(Backends.Num(G("not_2xx"))).Append('\n');
        sb.Append("rate_limited_429=").Append(Backends.Num(G("rate_limited"))).Append('\n');
        sb.Append("quota_denied_403=").Append(Backends.Num(G("quota_denied"))).Append('\n');
        sb.Append("cut_unbilled_requests=").Append(Backends.Num(G("cut")))
          .Append("  # client disconnect / stream timeout / upstream error: charged 0 tokens\n");
        sb.Append("cut_unbilled_seconds=").Append(Backends.Num(G("cut_s")))
          .Append("  # their wall time; the engine may have generated for up to this long\n");
        sb.Append("engine_aborted_requests=").Append(Backends.Num(E("aborted")))
          .Append("  # scheduler-side aborts (timeout, error): charged 0\n");
        sb.Append("gateway_p95_s=").Append(Backends.Num(G("ok_chat") is > 0 ? G("gateway_p95") : null, 3))
          .Append("   # whole successful request at the gateway\n");
        sb.Append("engine_e2e_p95_s=").Append(Backends.Num(timed ? E("e2e_p95") : null, 3)).Append('\n');
        sb.Append("engine_ttft_p95_s=").Append(Backends.Num(timed ? E("ttft_p95") : null, 3)).Append("  # queue wait + prefill\n");
        sb.Append("engine_queue_p95_s=").Append(Backends.Num(timed ? E("queue_p95") : null, 3)).Append('\n');
        sb.Append("engine_decode_tok_s_p50=").Append(Backends.Num(E("decode_rows") is > 0 ? E("decode_tok_s_p50") : null, 1))
          .Append("  # per-request output tokens / decode seconds, speculative decoding included\n");
        sb.Append("prefix_cache_hit=").Append(Backends.Num(hit, 3)).Append('\n');
        sb.Append("prefix_cache_hit_hicache=").Append(Backends.Num(hostHit, 3)).Append("  # reloaded from host RAM\n");
        sb.Append("cache_split_coverage=").Append(Backends.Num(tin >= 1 ? known / tin : null, 3)).Append('\n');

        // THE LEDGER DOES NOT MEASURE COST. ai-quota deducts a flat
        // input+output total and cannot be configured to weight them, but on
        // this node an output token costs 18.14 ms, an uncached input token
        // 0.266 ms, a GPU prefix hit 0.0038 ms and a HiCache reload 0.0276 ms
        // (1.24 s for 44,992 tokens, 2026-09-13). Input with an unknown cache
        // split counts as uncached. An estimate, not an accounting record.
        var ms = tout * 18.14 + Math.Max(0, tin - gpu - host) * 0.266 + gpu * 0.0038 + host * 0.0276;
        sb.Append("gpu_seconds_est=").Append(Backends.Num(ms / 1000.0, 1))
          .Append("  # the real resource; the ledger charges flat tokens instead\n");

        // Reference cost: what these tokens would cost buying the same model
        // from public providers. Prices come from quota-bot, the one place they
        // are fetched and dated, so the bot and this tool agree.
        if (b.Cfg.BotSecret.Length > 0 && tin + tout >= 1)
        {
            try
            {
                using var c = http.CreateClient("bot");
                using var pr = await c.GetAsync("admin/prices", ct);
                if (pr.IsSuccessStatusCode && JsonNode.Parse(await pr.Content.ReadAsStringAsync(ct)) is JsonObject p)
                {
                    string Cost(string key, bool cacheAware)
                    {
                        var r = p[key];
                        var inM = r?["input_per_m"]?.GetValue<decimal>() ?? 0m;
                        var outM = r?["output_per_m"]?.GetValue<decimal>() ?? 0m;
                        var cachedM = r?["cache_read_per_m"]?.GetValue<decimal>();
                        var input = (decimal)tin;
                        var cached = cacheAware && cachedM is not null ? input * (decimal)costHit : 0m;
                        var usd = ((input - cached) * inM + cached * (cachedM ?? 0m) + (decimal)tout * outM) / 1_000_000m;
                        return usd.ToString("0.00", CultureInfo.InvariantCulture);
                    }
                    sb.Append("cost_usd_openrouter=").Append(Cost("openrouter", false)).Append('\n');
                    sb.Append("cost_usd_openrouter_cache_aware=").Append(Cost("openrouter", true)).Append('\n');
                    sb.Append("cost_usd_alibaba_singapore=").Append(Cost("alibaba_singapore", false)).Append('\n');
                    sb.Append("cost_usd_alibaba_beijing=").Append(Cost("alibaba_beijing", false))
                      .Append("  # REFERENCE prices for the same model elsewhere, not a bill; basis: ")
                      .Append(p["openrouter"]?["basis"]?.GetValue<string>() ?? "?")
                      .Append("; Alibaba Cloud as of 2026-09-12\n");
                }
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException or InvalidOperationException or FormatException)
            {
                sb.Append("cost_usd=unavailable  # quota-bot price lookup failed\n");
            }
        }
        foreach (var r in repT.Result.Rows)
            sb.Append("output_tokens[").Append(r.GetValueOrDefault("replica") ?? "?").Append("]=")
              .Append(r.GetValueOrDefault("output_tokens") ?? "?").Append('\n');
        return sb.ToString();
    }

    [McpServerTool(Name = "consumer_requests", ReadOnly = true)]
    [Description("Recent individual requests for one consumer, newest first, from both "
        + "per-request tables: gateway.requests (the access log: status, tokens charged, "
        + "request_id) and engine.requests (SGLang's record: replica, cache split, first "
        + "token, end-to-end time, finish reason). A gateway row's chat_id is its engine "
        + "row's rid; request_detail pairs them for one request. Engine rows with source "
        + "'gateway-backfill' predate 2026-09-14 and carry tokens only.")]
    public static async Task<string> ConsumerRequests(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string consumer,
        [Description("How many hours back to look. Default 24, max 720.")] int hours,
        [Description("Maximum rows to return. Default 20, max 200.")] int limit,
        CancellationToken ct)
    {
        if (!Backends.SafeName(consumer)) return $"Not a valid consumer name: {consumer}";
        hours = hours <= 0 ? 24 : Math.Min(hours, 720);
        limit = limit <= 0 ? 20 : Math.Min(limit, 200);

        KeyValuePair<string, string>[] ps = [
            new("consumer", consumer),
            new("hours", hours.ToString(CultureInfo.InvariantCulture)),
            new("lim", limit.ToString(CultureInfo.InvariantCulture)),
        ];
        var gw = await b.ChSafeAsync(http, """
            SELECT ts, request_id, route, status, duration_ms, llm_ms,
                   input_tokens, output_tokens, total_tokens, response_flags, chat_round
            FROM gateway.requests FINAL
            WHERE consumer = {consumer:String}
              AND ts > now() - INTERVAL {hours:UInt32} HOUR
            ORDER BY ts DESC
            LIMIT {lim:UInt32}
            """, ps, ct);
        var engine = await b.ChSafeAsync(http, """
            SELECT finished_at, rid, replica, prompt_tokens, completion_tokens,
                   cached_device, cached_host, queue_s, ttft_s, e2e_s, finish_type, source
            FROM engine.requests FINAL
            WHERE consumer = {consumer:String}
              AND finished_at > now() - INTERVAL {hours:UInt32} HOUR
            ORDER BY finished_at DESC
            LIMIT {lim:UInt32}
            """, ps, ct);
        return "== gateway.requests (access log) ==\n" + gw
             + "\n== engine.requests (SGLang, cached_device = GPU prefix hit, cached_host = HiCache) ==\n" + engine;
    }

    [McpServerTool(Name = "request_detail", ReadOnly = true)]
    [Description("One request, end to end, by its gateway request id: the gateway.requests "
        + "row, then the engine.requests row for the same request, joined exactly on "
        + "gateway chat_id = engine rid (the gateway logs the response id, and SGLang uses "
        + "its request id as the response id). A request cut off before its response has no "
        + "chat_id and no engine record, so it has no engine side; so has anything from "
        + "before 2026-09-13 19:46 UTC.")]
    public static async Task<string> RequestDetail(
        Backends b, IHttpClientFactory http,
        [Description("The x-request-id the gateway minted for the request.")] string requestId,
        CancellationToken ct)
    {
        if (!RequestSql.PlausibleId(requestId))
            return $"Not a plausible request id: {Backends.Trim(requestId, 60)}";

        var fact = await b.ChSafeAsync(http, RequestSql.GatewayRow, [new("rid", requestId)], ct);
        var engine = await b.ChSafeAsync(http, RequestSql.EngineRow, [new("rid", requestId)], ct);

        var sb = new StringBuilder("== gateway.requests (billing-grade) ==\n");
        // The result always carries its header line; a match adds a second.
        sb.Append(fact.Count(ch => ch == '\n') > 1 ? fact : "(no row — did this request go through the gateway?)\n");
        sb.Append("\n== engine.requests (cached_device = GPU prefix hit, cached_host = HiCache reload) ==\n");
        sb.Append(engine.Count(ch => ch == '\n') > 1 ? engine
                  : "(no match — a request cut off before its response has no engine record; or one from before 2026-09-13 19:46 UTC)\n");
        return sb.ToString();
    }

    [McpServerTool(Name = "top_consumers", ReadOnly = true)]
    [Description("Which consumers are using the node, ranked by tokens charged over a window. "
        + "Exact sums over engine.requests (requests that passed the gateway; aborted "
        + "requests excluded, as the ledger charged them nothing). Windows: 5m, 1h, 6h, 24h, 7d, 30d.")]
    public static async Task<string> TopConsumers(
        Backends b, IHttpClientFactory http,
        [Description("Time window: 5m, 1h, 6h, 24h, 7d or 30d. Default 24h.")] string window,
        CancellationToken ct)
    {
        if (window.Length == 0) window = "24h";
        if (Backends.WindowSeconds(window) is not { } secs) return $"Not a valid window: {window}";

        var rows = await b.ChSafeAsync(http, """
            SELECT consumer,
                   sumIf(prompt_tokens, finish_type != 'abort')
                     + sumIf(completion_tokens, finish_type != 'abort') AS tokens,
                   sumIf(prompt_tokens, finish_type != 'abort')         AS tokens_in,
                   sumIf(completion_tokens, finish_type != 'abort')     AS tokens_out,
                   count()                                              AS requests
            FROM engine.requests FINAL
            WHERE consumer != ''
              AND finished_at >= now64(3) - toIntervalSecond({secs:UInt32})
            GROUP BY consumer
            ORDER BY tokens DESC
            """, [new("secs", secs.ToString(CultureInfo.InvariantCulture))], ct);
        return rows.Count(ch => ch == '\n') <= 1 ? $"No gateway traffic in the last {window}." : rows;
    }

    [McpServerTool(Name = "node_health", ReadOnly = true)]
    [Description("Is the node healthy right now: scrape targets up, alerts firing, "
        + "engine throughput, KV-cache pressure, the prefix cache over the last hour (exact, "
        + "from engine.requests: GPU hits and HiCache host-RAM reloads), HiCache host tier "
        + "fill, and whether the usage records pipeline is current.")]
    public static async Task<string> NodeHealth(
        Backends b, IHttpClientFactory http, CancellationToken ct)
    {
        async Task<double?> S(string q) => await b.ScalarAsync(http, q, ct);
        var sb = new StringBuilder();
        sb.Append("targets_up=").Append(Backends.Num(await S("count(up == 1)")))
          .Append('/').Append(Backends.Num(await S("count(up)"))).Append('\n');
        sb.Append("alerts_firing=").Append(Backends.Num(await S("count(ALERTS{alertstate=\"firing\"}) or vector(0)"))).Append('\n');
        sb.Append("gen_throughput_tok_s=").Append(Backends.Num(await S("sum(sglang:gen_throughput)"), 1)).Append('\n');
        sb.Append("running_requests=").Append(Backends.Num(await S("sum(sglang:num_running_reqs)"))).Append('\n');
        sb.Append("queued_requests=").Append(Backends.Num(await S("sum(sglang:num_queue_reqs)"))).Append('\n');
        // token_usage is used / pool size; the old used/available ratio divided
        // by FREE tokens and read 187% on 2026-09-13.
        sb.Append("kv_used_pct=").Append(Backends.Num(await S("max(sglang:token_usage) * 100"), 1)).Append('\n');
        var (cache, err) = await b.ChRowsAsync(http, """
            SELECT sumIf(prompt_tokens, cached_tokens IS NOT NULL) AS known,
                   sum(ifNull(cached_device, 0)) AS gpu, sum(ifNull(cached_host, 0)) AS host
            FROM engine.requests FINAL
            WHERE finished_at >= now64(3) - INTERVAL 1 HOUR
            """, [], ct);
        var row = cache.FirstOrDefault();
        var known = Backends.D(row, "known") ?? 0;
        sb.Append("prefix_cache_hit_pct_1h=").Append(err is not null ? "error" : Backends.Num(known >= 1 ? ((Backends.D(row, "gpu") ?? 0) + (Backends.D(row, "host") ?? 0)) / known * 100 : null, 1)).Append('\n');
        sb.Append("prefix_cache_hit_hicache_pct_1h=").Append(err is not null ? "error" : Backends.Num(known >= 1 ? (Backends.D(row, "host") ?? 0) / known * 100 : null, 1)).Append('\n');
        sb.Append("hicache_host_used_pct=").Append(Backends.Num(await S("sum(sglang:hicache_host_used_tokens) / sum(sglang:hicache_host_total_tokens) * 100"), 1)).Append('\n');
        sb.Append("usage_records_scrape_up=").Append(Backends.Num(await S("max(up{job=\"engine-usage\"})"))).Append('\n');
        sb.Append("newest_engine_record_age_s=").Append(Backends.Num(await S("time() - max(engine_usage_last_record_timestamp_seconds{source=\"engine\"})"))).Append("  # grows while idle\n");
        return sb.ToString();
    }

    [McpServerTool(Name = "prometheus_query", ReadOnly = true)]
    [Description("Run an arbitrary instant PromQL query against this node's Prometheus. "
        + "Read-only by nature. Useful metric families: engine_usage_* and gateway_usage_* "
        + "(exact per-consumer sums and percentiles for window=1h|24h|7d|30d, gauges scraped "
        + "from ClickHouse — read the latest value, never rate() them), sglang:* (engine "
        + "metrics; node-level HiCache and cache counters), consumer:* (ledger recording "
        + "rules, labelled ai_consumer — note the different label name), DCGM_* per GPU, "
        + "smg_* router. Returns the raw Prometheus JSON.")]
    public static async Task<string> PrometheusQuery(
        Backends b, IHttpClientFactory http,
        [Description("A PromQL instant-vector expression.")] string query,
        CancellationToken ct)
    {
        if (query.Length is 0 or > 2000) return "Query must be 1-2000 characters.";
        var n = await b.PromAsync(http, query, ct);
        return Backends.Trim(n?.ToJsonString() ?? "null", 20000);
    }
}

// ============================================================================
// WRITE TOOLS
//
// Only balance operations, and only through the gateway's quota API, which is a
// Redis INCRBY/SET and therefore atomic. Key lifecycle is absent on purpose —
// see the header of this file.
//
// Every write requires `confirm` to equal the consumer name exactly. That is
// not ceremony: it makes a mis-parsed or hallucinated tool call fail closed,
// because the model has to state the target twice and they have to agree.
// Every write appends to the same audit log quota-bot writes to.
// ============================================================================

[McpServerToolType]
sealed class WriteTools
{
    [McpServerTool(Name = "topup_balance", Destructive = false, Idempotent = false)]
    [Description("ADD tokens to a consumer's balance. Additive — it does not replace "
        + "the balance. Requires confirm to equal the consumer name exactly.")]
    public static async Task<string> TopUp(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string consumer,
        [Description("Tokens to ADD. Positive whole number.")] long tokens,
        [Description("Must equal the consumer name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
        => await Balance(b, http, consumer, tokens, confirm, delta: true, ct);

    [McpServerTool(Name = "set_balance", Destructive = true, Idempotent = true)]
    [Description("REPLACE a consumer's balance with an exact number of tokens. This "
        + "overwrites; it does not add. Use topup_balance to add. Requires confirm to "
        + "equal the consumer name exactly.")]
    public static async Task<string> SetBalance(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string consumer,
        [Description("The balance to set, in tokens.")] long tokens,
        [Description("Must equal the consumer name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
        => await Balance(b, http, consumer, tokens, confirm, delta: false, ct);

    private static async Task<string> Balance(
        Backends b, IHttpClientFactory http,
        string consumer, long tokens, string confirm, bool delta, CancellationToken ct)
    {
        if (!b.Cfg.WritesEnabled)
            return "Writes are disabled on this server (MCP_WRITES_ENABLED is not true).";
        if (!Backends.SafeName(consumer)) return $"Not a valid consumer name: {consumer}";
        if (!string.Equals(confirm, consumer, StringComparison.Ordinal))
            return $"Refused: confirm must be exactly \"{consumer}\". "
                 + "Re-read the consumer name and call again with it in both fields.";
        if (tokens < 0 || tokens > 10_000_000_000)
            return "Tokens must be between 0 and 10,000,000,000.";
        if (delta && tokens == 0) return "Refused: a top-up of zero does nothing.";

        // FORM-ENCODED, not JSON — ai-quota answers 403 to a JSON body, which
        // reads like an auth failure and is not one. Note the field names do
        // NOT match each other: refresh takes `quota`, delta takes `value`.
        using var content = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", consumer),
            new KeyValuePair<string, string>(delta ? "value" : "quota",
                tokens.ToString(CultureInfo.InvariantCulture)),
        ]);
        var path = delta
            ? "v1/chat/completions/quota/delta"
            : "v1/chat/completions/quota/refresh";

        using var c = http.CreateClient("gateway");
        using var r = await c.PostAsync(path, content, ct);
        if (!r.IsSuccessStatusCode)
            return $"The gateway refused the change: HTTP {(int)r.StatusCode}. "
                 + "The balance is unchanged. Call node_health to check the ledger.";

        await b.AuditAsync(
            $"{(delta ? "topup" : "setquota")} name={consumer} value={tokens}", ct);

        // Read the resulting balance back rather than echoing the intent: after
        // an overdraft the two differ, and the caller needs the real one.
        using var q = await c.GetAsync(
            $"v1/chat/completions/quota?consumer={Uri.EscapeDataString(consumer)}", ct);
        var now = q.IsSuccessStatusCode
            ? JsonNode.Parse(await q.Content.ReadAsStringAsync(ct))?["quota"]?.ToString()
            : null;

        return $"{consumer}: {(delta ? "added" : "set to")} {tokens:N0} tokens. "
             + $"Balance now {now ?? "unknown"}.";
    }
}

// ============================================================================
// KEY LIFECYCLE
//
// These do not touch the key-auth object. They call quota-bot's /admin/* API,
// which owns KeyStore and its lock, so a create from here and a /newkey from
// Telegram serialise against each other instead of racing to overwrite one
// wasmplugin object that carries no resourceVersion. Same lock, same audit log,
// same credential generator.
// ============================================================================

[McpServerToolType]
sealed class KeyTools
{
    [McpServerTool(Name = "list_tiers", ReadOnly = true)]
    [Description("The policy tiers a new consumer can be given: for each, what it is for "
        + "and its DEFAULT quota, refill, daily limit, tokens per minute and max_tokens. "
        + "CALL THIS BEFORE create_key and recommend a tier that matches the stated use "
        + "case rather than inventing numbers. Every value is a default that can be "
        + "changed per consumer with set_policy. Enforced: quota (starting balance, and what "
        + "each refill resets to), refill (00:00 UTC daily/Monday/1st), daily_limit and "
        + "tokens_per_minute (gateway 429). daily_limit is a 24h window from the key's first "
        + "request, not a calendar day. max_tokens_NOT_ENFORCED cannot be applied per key — "
        + "never quote it to a consumer as a limit.")]
    public static async Task<string> ListTiers(
        Backends b, IHttpClientFactory http, CancellationToken ct)
    {
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured on this server: ADMIN_API_SECRET is unset, "
                 + "so it has no route to quota-bot's admin API.";
        using var c = http.CreateClient("bot");
        using var r = await c.GetAsync("admin/tiers", ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        return r.IsSuccessStatusCode
            ? body
            : $"quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 300)}";
    }

    [McpServerTool(Name = "create_key", Destructive = false, Idempotent = false)]
    [Description("Create a new API consumer and return its credential. THE CREDENTIAL IS "
        + "RETURNED ONCE and is not stored anywhere it can be read back — relay it to "
        + "the person who needs it and tell them it cannot be re-issued. "
        + "Give either a tier (which sets the starting quota from the tier table — call "
        + "list_tiers first) or an explicit quota in tokens, or both, in which case the "
        + "explicit quota wins. Neither: 1,000,000 tokens and no tier. "
        + "Requires confirm to equal the name exactly.")]
    public static async Task<string> CreateKey(
        Backends b, IHttpClientFactory http,
        [Description("New consumer name: 1-32 characters of a-z, 0-9, - or _.")] string name,
        [Description("Optional tier: trial, team, service, batch. Sets the starting quota.")] string tier,
        [Description("Optional explicit starting balance in tokens. Overrides the tier's quota.")] long quota,
        [Description("Must equal the name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
    {
        if (!b.Cfg.WritesEnabled)
            return "Writes are disabled on this server (MCP_WRITES_ENABLED is not true).";
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured: ADMIN_API_SECRET is unset.";
        if (!Backends.SafeName(name)) return $"Not a valid consumer name: {name}";
        if (!string.Equals(confirm, name, StringComparison.Ordinal))
            return $"Refused: confirm must be exactly \"{name}\".";

        var payload = new JsonObject { ["name"] = name };
        if (tier is { Length: > 0 }) payload["tier"] = tier;
        if (quota > 0) payload["quota"] = quota;

        using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
        using var c = http.CreateClient("bot");
        using var r = await c.PostAsync("admin/keys", content, ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        if (!r.IsSuccessStatusCode)
            return $"Not created. quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 400)}";

        await b.AuditAsync($"create_key name={name} tier={(tier.Length > 0 ? tier : "-")}", ct);
        return body;
    }

    [McpServerTool(Name = "revoke_key", Destructive = true, Idempotent = true)]
    [Description("Delete a consumer: its key stops authenticating immediately and its "
        + "balance is deleted. This cannot be undone — a replacement is a different "
        + "credential, and anything using the old one breaks at once. Check "
        + "consumer_stats first to see whether it is actively serving traffic. "
        + "Requires confirm to equal the name exactly.")]
    public static async Task<string> RevokeKey(
        Backends b, IHttpClientFactory http,
        [Description("Consumer to delete.")] string name,
        [Description("Must equal the name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
    {
        if (!b.Cfg.WritesEnabled)
            return "Writes are disabled on this server (MCP_WRITES_ENABLED is not true).";
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured: ADMIN_API_SECRET is unset.";
        if (!Backends.SafeName(name)) return $"Not a valid consumer name: {name}";
        if (!string.Equals(confirm, name, StringComparison.Ordinal))
            return $"Refused: confirm must be exactly \"{name}\".";

        using var c = http.CreateClient("bot");
        using var r = await c.DeleteAsync($"admin/keys/{Uri.EscapeDataString(name)}", ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        if (!r.IsSuccessStatusCode)
            return $"Not revoked. quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 400)}";

        await b.AuditAsync($"revoke_key name={name}", ct);
        return $"{name} revoked. Its key no longer authenticates and its balance is gone.";
    }

    [McpServerTool(Name = "set_tier", Destructive = false, Idempotent = true)]
    [Description("Record a consumer's policy tier. This is bookkeeping, not enforcement: "
        + "it does not change their balance, their rate limit or their max_tokens. Use "
        + "set_balance to change what they can actually spend. Requires confirm to equal "
        + "the name exactly.")]
    public static async Task<string> SetTier(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string name,
        [Description("Tier: trial, team, service, batch or admin.")] string tier,
        [Description("Must equal the name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
    {
        if (!b.Cfg.WritesEnabled)
            return "Writes are disabled on this server (MCP_WRITES_ENABLED is not true).";
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured: ADMIN_API_SECRET is unset.";
        if (!Backends.SafeName(name)) return $"Not a valid consumer name: {name}";
        if (!string.Equals(confirm, name, StringComparison.Ordinal))
            return $"Refused: confirm must be exactly \"{name}\".";

        var payload = new JsonObject { ["name"] = name, ["tier"] = tier };
        using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
        using var c = http.CreateClient("bot");
        using var r = await c.PostAsync("admin/tier", content, ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        return r.IsSuccessStatusCode
            ? body
            : $"Not set. quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 400)}";
    }

    [McpServerTool(Name = "get_policy", ReadOnly = true)]
    [Description("One consumer's effective settings — quota, refill, daily, tpm, max_tokens — "
        + "each with its value and its source: 'tier' when it follows the consumer's tier, "
        + "'set' when it was set on this consumer by hand, 'none' when the consumer has no "
        + "tier and nothing set. Each also says whether it is enforced: everything except "
        + "max_tokens is.")]
    public static async Task<string> GetPolicy(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string name,
        CancellationToken ct)
    {
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured: ADMIN_API_SECRET is unset.";
        if (!Backends.SafeName(name)) return $"Not a valid consumer name: {name}";

        using var c = http.CreateClient("bot");
        using var r = await c.GetAsync($"admin/policy/{Uri.EscapeDataString(name)}", ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        return r.IsSuccessStatusCode
            ? body
            : $"quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 400)}";
    }

    [McpServerTool(Name = "set_policy", Destructive = false, Idempotent = true)]
    [Description("Change ONE setting for one consumer, or put it back to its tier's value "
        + "with value 'default'. Settings: quota (tokens a refill grants), refill (manual, "
        + "daily, weekly, monthly), daily (tokens per UTC day, 0 = no limit), tpm (tokens "
        + "per minute, 0 = no limit), max_tokens (0 = gateway ceiling). Amounts accept "
        + "2000000, 2M or 500k. daily and tpm take effect at the gateway within seconds; "
        + "refill resets the balance to quota at the next UTC boundary (switching it on "
        + "never resets immediately). This never changes a balance directly — use "
        + "set_balance or topup_balance for that. Returns the consumer's whole effective "
        + "policy. Requires confirm to equal the name exactly.")]
    public static async Task<string> SetPolicy(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name.")] string name,
        [Description("quota, refill, daily, tpm or max_tokens.")] string field,
        [Description("The new value, or 'default' to follow the tier again.")] string value,
        [Description("Must equal the name exactly, or the call is refused.")] string confirm,
        CancellationToken ct)
    {
        if (!b.Cfg.WritesEnabled)
            return "Writes are disabled on this server (MCP_WRITES_ENABLED is not true).";
        if (b.Cfg.BotSecret.Length == 0)
            return "Key lifecycle is not configured: ADMIN_API_SECRET is unset.";
        if (!Backends.SafeName(name)) return $"Not a valid consumer name: {name}";
        if (!string.Equals(confirm, name, StringComparison.Ordinal))
            return $"Refused: confirm must be exactly \"{name}\".";

        var payload = new JsonObject { ["name"] = name, ["field"] = field, ["value"] = value };
        using var content = new StringContent(payload.ToJsonString(), Encoding.UTF8, "application/json");
        using var c = http.CreateClient("bot");
        using var r = await c.PostAsync("admin/policy", content, ct);
        var body = await r.Content.ReadAsStringAsync(ct);
        if (!r.IsSuccessStatusCode)
            return $"Not set. quota-bot returned HTTP {(int)r.StatusCode}: {Backends.Trim(body, 400)}";

        await b.AuditAsync($"set_policy name={name} field={field} value={value}", ct);
        return body;
    }
}
