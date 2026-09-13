// =============================================================================
// admin-mcp — a Model Context Protocol server for operating this inference node
//
// WHAT THIS IS FOR
//
// quota-bot already answers "what is happening" from Telegram, but it is
// deliberately blind to the two stores that matter most for tracing: it runs on
// `edge`, and the fact table and the span store are backend-only, so /trace
// prints SQL for a human to run instead of running it. This process is the one
// allowed to cross that line, so an admin's client can chain
// "who spiked" -> "which requests" -> "what did the engine do" without pasting
// queries between steps.
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
    LangfuseUrl:    Opt("LANGFUSE_PUBLIC_URL", "").TrimEnd('/'),
    AuditPath:      Opt("AUDIT_PATH", "/data/audit.log"),
    BotUrl:         Opt("BOT_ADMIN_URL", "http://quota-bot:8080").TrimEnd('/'),
    BotSecret:      Opt("ADMIN_API_SECRET", ""),
    WritesEnabled:  Opt("MCP_WRITES_ENABLED", "true") == "true");

// The token is compared in constant time, so it is hashed once here rather than
// on every request. Length is checked too: a short token in the environment is
// a configuration mistake worth failing at boot rather than at 3am.
if (cfg.BearerToken.Length < 32)
    throw new InvalidOperationException("MCP_BEARER_TOKEN must be at least 32 characters");
var expectedToken = SHA256.HashData(Encoding.UTF8.GetBytes(cfg.BearerToken));

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

    var auth = ctx.Request.Headers.Authorization.ToString();
    const string scheme = "Bearer ";
    var ok = auth.StartsWith(scheme, StringComparison.OrdinalIgnoreCase)
             && CryptographicOperations.FixedTimeEquals(
                    SHA256.HashData(Encoding.UTF8.GetBytes(auth[scheme.Length..])),
                    expectedToken);
    if (!ok)
    {
        log.LogWarning("unauthenticated request from {Ip}",
            ctx.Connection.RemoteIpAddress?.ToString() ?? "?");
        ctx.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return;
    }

    await next();
});

app.MapMcp("/mcp");

log.LogInformation("admin-mcp listening; writes={Writes} origins={Origins}",
    cfg.WritesEnabled, string.Join(",", cfg.AllowedOrigins));
app.Run();
return 0;

// ---------------------------------------------------------------- shared ----

sealed record McpConfig(
    string BearerToken, HashSet<string> AllowedOrigins,
    string PrometheusUrl, string ClickHouseUrl, string ClickHouseUser, string ClickHousePass,
    string GatewayUrl, string AdminCredential, string LangfuseUrl,
    string AuditPath, string BotUrl, string BotSecret, bool WritesEnabled);

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

    public static bool ValidWindow(string w) =>
        w is "5m" or "1h" or "6h" or "24h" or "7d" or "30d";

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
        + "Balances come from the Redis quota ledger, which is the billing record — "
        + "never quote token totals from Langfuse or from Prometheus counters instead. "
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
    [Description("Everything measured about one consumer over a window. Returns two "
        + "latency views that are NOT interchangeable: gateway_p95 is whole-request "
        + "duration as Envoy saw it, engine_* is measured inside SGLang with no "
        + "gateway, router or network in it. inter_token_latency and prefix_cache_hit "
        + "exist only on the engine side — the gateway cannot see them. A dash means "
        + "no data in that window, which is not the same as zero. "
        + "Windows: 5m, 1h, 6h, 24h, 7d, 30d.")]
    public static async Task<string> ConsumerStats(
        Backends b, IHttpClientFactory http,
        [Description("Consumer name, exactly as list_consumers reports it.")] string consumer,
        [Description("Time window: 5m, 1h, 6h, 24h, 7d or 30d. Default 24h.")] string window,
        CancellationToken ct)
    {
        if (!Backends.SafeName(consumer)) return $"Not a valid consumer name: {consumer}";
        if (window.Length == 0) window = "24h";
        if (!Backends.ValidWindow(window)) return $"Not a valid window: {window}";

        var sel = $"{{consumer=\"{consumer}\"}}";
        var led = $"{{ai_consumer=\"{consumer}\"}}";
        var w = window;

        async Task<double?> S(string q) => await b.ScalarAsync(http, q, ct);

        var balance = await S($"consumer:quota_balance:tokens{led}");
        var runway = await S($"consumer:quota_days_left{led}");
        var reqs = await S($"sum(increase(gateway_requests_total{sel}[{w}])) or vector(0)");
        var bad = await S($"sum(increase(gateway_requests_total{{consumer=\"{consumer}\",status_class!=\"2xx\"}}[{w}])) or vector(0)");
        var tin = await S($"sum(increase(gateway_tokens_total{{consumer=\"{consumer}\",direction=\"input\"}}[{w}])) or vector(0)");
        var tout = await S($"sum(increase(gateway_tokens_total{{consumer=\"{consumer}\",direction=\"output\"}}[{w}])) or vector(0)");
        var gw = await S($"histogram_quantile(0.95, sum by (le) (rate(gateway_request_duration_seconds_bucket{sel}[{w}])))");
        var e2e = await S($"histogram_quantile(0.95, sum by (le) (rate(sglang:e2e_request_latency_seconds_bucket{sel}[{w}])))");
        var ttft = await S($"histogram_quantile(0.95, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{sel}[{w}])))");
        var itl = await S($"histogram_quantile(0.95, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket{sel}[{w}])))");
        var cache = await S($"1 - sum(rate(sglang:uncached_prompt_tokens_histogram_sum{sel}[{w}])) / clamp_min(sum(rate(sglang:prompt_tokens_histogram_sum{sel}[{w}])), 1)");

        var perReplica = await b.PromAsync(http,
            $"sum by (instance) (increase(sglang:generation_tokens_total{sel}[{w}]))", ct);

        var sb = new StringBuilder();
        sb.Append("consumer=").Append(consumer).Append("  window=").Append(w).Append('\n');
        sb.Append("balance_tokens=").Append(Backends.Num(balance)).Append('\n');
        sb.Append("runway_days=").Append(Backends.Num(runway, 1)).Append('\n');
        sb.Append("requests=").Append(Backends.Num(reqs)).Append('\n');
        sb.Append("not_2xx=").Append(Backends.Num(bad)).Append('\n');
        sb.Append("tokens_in=").Append(Backends.Num(tin)).Append('\n');
        sb.Append("tokens_out=").Append(Backends.Num(tout)).Append('\n');
        sb.Append("gateway_p95_s=").Append(Backends.Num(gw, 3)).Append("   # Envoy, whole request\n");
        sb.Append("engine_e2e_p95_s=").Append(Backends.Num(e2e, 3)).Append('\n');
        sb.Append("engine_ttft_p95_s=").Append(Backends.Num(ttft, 3)).Append("  # queue wait + prefill\n");
        sb.Append("engine_itl_p95_s=").Append(Backends.Num(itl, 4)).Append("   # gap between output tokens\n");
        sb.Append("prefix_cache_hit=").Append(Backends.Num(cache, 3)).Append('\n');

        // Requests cut off before their final usage frame are charged ZERO by
        // ai-quota, and the engine's token counters skip them too (measured
        // 2026-09-13). No `or vector(0)`: absent means no data, not none.
        var cut = await S($"sum(increase(gateway_unbilled_requests_total{sel}[{w}]))");
        var cutS = await S($"sum(increase(gateway_unbilled_seconds_total{sel}[{w}]))");
        sb.Append("cut_unbilled_requests=").Append(Backends.Num(cut))
          .Append("  # client disconnect / stream timeout / upstream error: charged 0 tokens\n");
        sb.Append("cut_unbilled_seconds=").Append(Backends.Num(cutS))
          .Append("  # their wall time; the engine may have generated for up to this long\n");

        // THE LEDGER DOES NOT MEASURE COST. ai-quota deducts a flat
        // input+output total and cannot be configured to weight them, but an
        // output token costs ~68x an uncached input token and ~4800x a cached
        // one on this node (docs/KEY-TIERS.md, measured). So two consumers can
        // hold identical balances and consume wildly different GPU time.
        //
        // This is that missing number, reconstructed from the same measurements:
        // output 18.14 ms/token, uncached input 0.266, cached input 0.0038.
        // An estimate, not an accounting record — it uses the window's average
        // cache hit rate rather than per-request hit data.
        if (tin is not null && tout is not null)
        {
            var hit = cache is >= 0 and <= 1 ? cache.Value : 0;
            var ms = tout.Value * 18.14
                   + tin.Value * (1 - hit) * 0.266
                   + tin.Value * hit * 0.0038;
            sb.Append("gpu_seconds_est=").Append(Backends.Num(ms / 1000.0, 1))
              .Append("  # the real resource; the ledger charges flat tokens instead\n");
        }
        foreach (var r in perReplica?["data"]?["result"]?.AsArray() ?? [])
        {
            var inst = r?["metric"]?["instance"]?.GetValue<string>() ?? "?";
            sb.Append("output_tokens[").Append(inst).Append("]=")
              .Append(r?["value"]?[1]?.GetValue<string>() ?? "?").Append('\n');
        }
        return sb.ToString();
    }

    [McpServerTool(Name = "consumer_requests", ReadOnly = true)]
    [Description("Recent individual requests for one consumer, newest first, from the "
        + "gateway.requests fact table in ClickHouse. This is the billing-grade "
        + "per-request record: one row per request that passed through the gateway. "
        + "Requests that bypassed the gateway do not appear. Use the request_id from "
        + "here with request_detail to see what the engine did.")]
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

        const string sql = """
            SELECT ts, request_id, route, status, duration_ms, llm_ms,
                   input_tokens, output_tokens, total_tokens, chat_round
            FROM gateway.requests FINAL
            WHERE consumer = {consumer:String}
              AND ts > now() - INTERVAL {hours:UInt32} HOUR
            ORDER BY ts DESC
            LIMIT {lim:UInt32}
            """;
        return await b.ChSafeAsync(http, sql, [
            new("consumer", consumer),
            new("hours", hours.ToString(CultureInfo.InvariantCulture)),
            new("lim", limit.ToString(CultureInfo.InvariantCulture)),
        ], ct);
    }

    [McpServerTool(Name = "request_detail", ReadOnly = true)]
    [Description("One request, end to end, by its gateway request id. Runs the two-hop "
        + "join: the fact-table row, then the router span carrying that request id, "
        + "then the engine spans sharing the ROUTER's trace id. The engine does not "
        + "adopt the caller's request id — it mints its own — so searching the span "
        + "store for this id directly finds only the gateway span. That is why the "
        + "second hop exists.")]
    public static async Task<string> RequestDetail(
        Backends b, IHttpClientFactory http,
        [Description("The x-request-id Envoy minted for the request.")] string requestId,
        CancellationToken ct)
    {
        if (requestId.Length is < 8 or > 64
            || !requestId.All(c => char.IsAsciiLetterOrDigit(c) || c == '-'))
            return $"Not a plausible request id: {Backends.Trim(requestId, 60)}";

        var fact = await b.ChSafeAsync(http, """
            SELECT ts, consumer, route, model, status, duration_ms, llm_ms,
                   input_tokens, output_tokens, total_tokens, chat_id, chat_round
            FROM gateway.requests FINAL
            WHERE request_id = {rid:String}
            """, [new("rid", requestId)], ct);

        // Hop 1 -> the router span; hop 2 -> everything in that span's trace.
        var spans = await b.ChSafeAsync(http, """
            WITH (SELECT trace_id FROM events_core
                   WHERE service_name = 'smg'
                     AND metadata_values[indexOf(metadata_names,'attributes.request_id')] = {rid:String}
                   LIMIT 1) AS tid
            SELECT service_name, name,
                   toUnixTimestamp64Milli(start_time) AS start_ms,
                   toUnixTimestamp64Milli(end_time) - toUnixTimestamp64Milli(start_time) AS dur_ms,
                   trace_id
            FROM events_core
            WHERE tid != '' AND trace_id = tid
            ORDER BY start_time
            LIMIT 200
            """, [new("rid", requestId)], ct);

        var sb = new StringBuilder("== gateway.requests (billing-grade) ==\n");
        sb.Append(fact.Length > 0 ? fact : "(no row — did this request go through the gateway?)\n");
        sb.Append("\n== spans in the router's trace (engine waterfall) ==\n");
        sb.Append(spans.Length > 0 ? spans : "(no spans — the trace may have aged out)\n");
        if (b.Cfg.LangfuseUrl.Length > 0)
            sb.Append("\nLangfuse: ").Append(b.Cfg.LangfuseUrl).Append('\n');
        return sb.ToString();
    }

    [McpServerTool(Name = "top_consumers", ReadOnly = true)]
    [Description("Which consumers are using the node, ranked over a window. Reads the "
        + "access-log derived counters, so it counts gateway traffic only.")]
    public static async Task<string> TopConsumers(
        Backends b, IHttpClientFactory http,
        [Description("Time window: 1h, 6h, 24h, 7d or 30d. Default 24h.")] string window,
        CancellationToken ct)
    {
        if (window.Length == 0) window = "24h";
        if (!Backends.ValidWindow(window)) return $"Not a valid window: {window}";

        var n = await b.PromAsync(http,
            $"sort_desc(sum by (consumer) (increase(gateway_tokens_total[{window}])))", ct);
        var rows = n?["data"]?["result"]?.AsArray();
        if (rows is null || rows.Count == 0) return $"No gateway traffic in the last {window}.";

        var sb = new StringBuilder("consumer\ttokens\n");
        foreach (var r in rows)
            sb.Append(r?["metric"]?["consumer"]?.GetValue<string>() ?? "?").Append('\t')
              .Append(r?["value"]?[1]?.GetValue<string>() ?? "?").Append('\n');
        return sb.ToString();
    }

    [McpServerTool(Name = "node_health", ReadOnly = true)]
    [Description("Is the node healthy right now: scrape targets up, alerts firing, "
        + "engine throughput, KV-cache pressure and prefix cache hit rate. "
        + "cache_hit_rate is cumulative since engine start, so it reads near zero "
        + "for a while after a replica roll — that is not an incident.")]
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
        sb.Append("kv_used_pct=").Append(Backends.Num(await S("max((sglang:kv_used_tokens / sglang:kv_available_tokens)) * 100"), 1)).Append('\n');
        sb.Append("prefix_cache_hit_pct=").Append(Backends.Num(await S("avg(sglang:cache_hit_rate) * 100"), 1)).Append('\n');
        sb.Append("collector_queue=").Append(Backends.Num(await S("max(otelcol_exporter_queue_size)"))).Append('\n');
        return sb.ToString();
    }

    [McpServerTool(Name = "prometheus_query", ReadOnly = true)]
    [Description("Run an arbitrary instant PromQL query against this node's Prometheus. "
        + "Read-only by nature. Useful metric families: sglang:* (98 engine metrics, "
        + "labelled by consumer on the tokenizer-side ones), gateway_* (access-log "
        + "derived, labelled by consumer), consumer:* (ledger recording rules, "
        + "labelled ai_consumer — note the different label name), DCGM_* per GPU, "
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
        + "changed per consumer with set_policy. Only quota is applied, as a new key's "
        + "starting balance; fields suffixed _NOT_ENFORCED / _NOT_RUNNING are recorded "
        + "and enforced by nothing yet, so never quote them to a consumer as limits.")]
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
        + "tier and nothing set. Each also says whether it is enforced; today only the "
        + "balance is.")]
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
        + "2000000, 2M or 500k. This NEVER changes a balance — use set_balance or "
        + "topup_balance for that — and apart from the balance nothing here is enforced "
        + "yet. Returns the consumer's whole effective policy. Requires confirm to equal "
        + "the name exactly.")]
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
