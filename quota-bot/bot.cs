// =============================================================================
// quota-bot — Telegram control surface for gateway access management.
//
// One file, on purpose: .NET 10 file-based apps take `#:` directives at the top
// and need no .csproj. The whole service is here.
//
// ARCHITECTURE — the asynchrony is the point.
//   Telegram redelivers any update the webhook does not answer 2xx, so doing
//   work inside the request handler causes retry storms AND duplicate commands.
//   A duplicated /topup is money. So the handler only validates and enqueues,
//   answers 200 immediately, and a BackgroundService does the real work.
//
// NATIVE AOT — two rules that bite at runtime rather than at build time:
//   1. Source-generated JSON is mandatory. Reflection-based serialisation
//      throws once trimmed. Everything crossing the wire has a [JsonSerializable]
//      entry on BotJson below.
//   2. CreateSlimBuilder, not CreateBuilder. The full builder wires up
//      machinery that is not AOT-compatible.
//
// TRIM WARNINGS ARE ERRORS HERE (ILLinkTreatWarningsAsErrors). They collapse to
// one line per assembly by default, which makes them easy to wave through, and
// a waved-through trim warning is a runtime crash under AOT.
//
// NOT set: UseSystemResourceKeys and StackTraceSupport=false. Both trade
// diagnostics for a few hundred KB. This process moves customer balances; a
// stripped exception message costs more than the size saving is worth.
// =============================================================================

#:sdk Microsoft.NET.Sdk.Web
#:property PublishAot=true
#:property PackAsTool=false
#:property InvariantGlobalization=true
#:property OptimizationPreference=Size
#:property EventSourceSupport=false
#:property MetadataUpdaterSupport=false
#:property HttpActivityPropagationSupport=false
#:property Http3Support=false
#:property EnableUnsafeUTF7Encoding=false
#:property TrimmerSingleWarn=false
#:property ILLinkTreatWarningsAsErrors=true
// TreatWarningsAsErrors is the one that actually bites. The trim/AOT analysers
// surface IL2026 and IL3050 as ordinary compiler warnings first, and
// ILLinkTreatWarningsAsErrors alone let a real JsonValue.Create<T> hazard
// through a clean-looking build here on 2026-09-03.
#:property TreatWarningsAsErrors=true
#:property Nullable=enable
#:property ImplicitUsings=enable

using System.Buffers;
using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading.Channels;

// ---------------------------------------------------------------------------
// Health probe mode.
//
// The chiselled runtime image has no shell and no curl, so a compose healthcheck
// cannot be a command line — it has to be the binary probing itself. Runs before
// any configuration is read so a misconfigured container still reports unhealthy
// rather than failing to start the probe.
// ---------------------------------------------------------------------------
if (args is ["--healthcheck"])
{
    using var probe = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
    try
    {
        using var res = await probe.GetAsync("http://127.0.0.1:8080/healthz");
        return res.IsSuccessStatusCode ? 0 : 1;
    }
    catch { return 1; }
}

// ---------------------------------------------------------------------------
// Configuration. Every value is required except the optional ones noted; the
// bot refuses to start rather than run half-configured, because a missing
// allowlist would mean an open control surface.
// ---------------------------------------------------------------------------
static string Req(string k) =>
    Environment.GetEnvironmentVariable(k) is { Length: > 0 } v
        ? v
        : throw new InvalidOperationException($"required environment variable {k} is unset");

static string Opt(string k, string fallback) =>
    Environment.GetEnvironmentVariable(k) is { Length: > 0 } v ? v : fallback;

var cfg = new BotConfig(
    BotToken:        Req("TELEGRAM_BOT_TOKEN"),
    WebhookSecret:   Req("TELEGRAM_WEBHOOK_SECRET"),
    WebhookPath:     new Uri(Req("TELEGRAM_WEBHOOK_URL")).AbsolutePath,
    AllowedIds:      Req("TELEGRAM_ALLOWED_IDS")
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                        .Select(s => long.Parse(s, CultureInfo.InvariantCulture))
                        .ToHashSet(),
    AdminCredential: Req("QUOTA_ADMIN_CREDENTIAL"),
    GatewayUrl:      Opt("GATEWAY_URL", "http://higress:80").TrimEnd('/'),
    ApiServerUrl:    Opt("APISERVER_URL", "https://apiserver.svc:8443").TrimEnd('/'),
    RedisHost:       Opt("REDIS_HOST", "higress-redis"),
    RedisPort:       int.Parse(Opt("REDIS_PORT", "6379"), CultureInfo.InvariantCulture),
    PrometheusUrl:   Opt("PROMETHEUS_URL", "http://qwen36-27b-prometheus:9090").TrimEnd('/'),
    PublicBaseUrl:   Opt("PUBLIC_BASE_URL", "https://qw38-27b-gw.duckdns.org").TrimEnd('/'),
    ConsumersPath:   Opt("CONSUMERS_PATH", "/data/consumers.conf"),
    AuditPath:       Opt("AUDIT_PATH", "/data/audit.log"),
    ModelId:         Opt("MODEL_ID", "qwen3.8-27b"),
    ContextLimit:    int.Parse(Opt("MODEL_CONTEXT", "169000"), CultureInfo.InvariantCulture),
    OutputLimit:     int.Parse(Opt("MODEL_OUTPUT", "70000"), CultureInfo.InvariantCulture));

var builder = WebApplication.CreateSlimBuilder(args);
builder.Logging.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "yyyy-MM-ddTHH:mm:ssZ "; o.UseUtcTimestamp = true; });
builder.Services.ConfigureHttpJsonOptions(o =>
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, BotJson.Default));

var queue = Channel.CreateBounded<Update>(new BoundedChannelOptions(256)
{
    // Drop rather than block: the webhook handler must never wait on the worker.
    // A dropped update is a command the operator can simply retype; a blocked
    // handler is a Telegram retry storm.
    FullMode = BoundedChannelFullMode.DropWrite,
    SingleReader = true
});

builder.Services.AddSingleton(cfg);
builder.Services.AddSingleton(queue);
builder.Services.AddSingleton<Ledger>();
builder.Services.AddSingleton<KeyStore>();
builder.Services.AddHostedService<Worker>();

builder.Services.AddHttpClient("telegram", c =>
{
    c.BaseAddress = new Uri($"https://api.telegram.org/bot{cfg.BotToken}/");
    c.Timeout = TimeSpan.FromSeconds(20);
});
builder.Services.AddHttpClient("gateway", c =>
{
    c.BaseAddress = new Uri(cfg.GatewayUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
    c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", cfg.AdminCredential);
});
builder.Services.AddHttpClient("prometheus", c =>
{
    c.BaseAddress = new Uri(cfg.PrometheusUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
});
builder.Services.AddHttpClient("apiserver", c =>
{
    c.BaseAddress = new Uri(cfg.ApiServerUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
})
// The apiserver presents a self-signed certificate whose SAN does not cover the
// compose alias, and it accepts unauthenticated requests anyway — verified: an
// anonymous GET of the wasmplugins collection returns 200. Pinning the CA here
// would be security theatre over an API that has no authentication to protect.
// The real control is that higress-net is not routable from outside the host.
.ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
{
    ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
});

var app = builder.Build();
var log = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("webhook");

// ---------------------------------------------------------------------------
// The webhook. Validate, enqueue, 200. Nothing else.
//
// Always 200 on a well-formed request, including one we intend to ignore:
// a 4xx/5xx makes Telegram redeliver the same update indefinitely.
// ---------------------------------------------------------------------------
app.MapPost(cfg.WebhookPath, async (HttpRequest req) =>
{
    var presented = req.Headers["X-Telegram-Bot-Api-Secret-Token"].ToString();
    if (!FixedTimeEquals(presented, cfg.WebhookSecret))
    {
        log.LogWarning("rejected webhook call with bad or missing secret token from {Ip}",
            req.HttpContext.Connection.RemoteIpAddress);
        return Results.Unauthorized();
    }

    Update? update;
    try
    {
        update = await JsonSerializer.DeserializeAsync(req.Body, BotJson.Default.Update);
    }
    catch (JsonException ex)
    {
        // Malformed body: swallow it. Retrying will not make it parse.
        log.LogWarning(ex, "dropped unparseable update");
        return Results.Ok();
    }

    if (update is not null && !queue.Writer.TryWrite(update))
        log.LogError("queue full, dropped update {UpdateId}", update.UpdateId);

    return Results.Ok();
});

app.MapGet("/healthz", () => Results.Text("ok"));

app.Run();
return 0;

static bool FixedTimeEquals(string a, string b)
{
    var x = Encoding.UTF8.GetBytes(a);
    var y = Encoding.UTF8.GetBytes(b);
    return x.Length == y.Length && CryptographicOperations.FixedTimeEquals(x, y);
}

// ===========================================================================
// Worker — everything that can be slow happens here, off the request path.
// ===========================================================================
sealed class Worker(
    BotConfig cfg,
    Channel<Update> queue,
    Ledger ledger,
    KeyStore keys,
    IHttpClientFactory http,
    ILogger<Worker> log) : BackgroundService
{
    // Telegram redelivers on failure, and it can redeliver an update we already
    // handled. Bounded so a long-running process cannot grow it without limit.
    private readonly HashSet<long> _seen = [];
    private readonly Queue<long> _seenOrder = new();

    // Pending destructive operations awaiting a literal CONFIRM. In memory on
    // purpose: a restart drops them, which fails in the safe direction.
    private readonly ConcurrentDictionary<long, Pending> _pending = new();

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        await foreach (var update in queue.Reader.ReadAllAsync(ct))
        {
            try { await HandleAsync(update, ct); }
            catch (Exception ex) { log.LogError(ex, "handler failed for update {Id}", update.UpdateId); }
        }
    }

    private async Task HandleAsync(Update u, CancellationToken ct)
    {
        if (!MarkSeen(u.UpdateId)) { log.LogInformation("ignored duplicate update {Id}", u.UpdateId); return; }

        var msg = u.Message;
        if (msg?.Text is not { Length: > 0 } text || msg.From is null || msg.Chat is null) return;

        // DMs only. A group is a membership list we would have to maintain, and
        // balances would be visible to everyone in it.
        if (!string.Equals(msg.Chat.Type, "private", StringComparison.Ordinal))
        {
            log.LogWarning("ignored non-private chat {ChatId} type {Type}", msg.Chat.Id, msg.Chat.Type);
            return;
        }

        // Numeric id, never username — usernames can be released and re-registered.
        if (!cfg.AllowedIds.Contains(msg.From.Id))
        {
            log.LogWarning("DENIED user {UserId} (@{Username}) command {Text}",
                msg.From.Id, msg.From.Username ?? "-", Head(text));
            return;
        }

        var reply = await DispatchAsync(msg.From.Id, text.Trim(), ct);
        if (reply is { Length: > 0 }) await SendAsync(msg.Chat.Id, reply, ct);
    }

    private async Task<string> DispatchAsync(long userId, string text, CancellationToken ct)
    {
        // A pending CONFIRM consumes the next message from that user outright.
        if (_pending.TryGetValue(userId, out var p))
        {
            _pending.TryRemove(userId, out _);
            if (DateTimeOffset.UtcNow > p.Expires) return "Confirmation expired. Nothing was changed.";
            if (!string.Equals(text, "CONFIRM", StringComparison.Ordinal))
                return "Cancelled. Nothing was changed.";
            return await p.Run(ct);
        }

        var parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cmd = parts[0].Split('@')[0].ToLowerInvariant();
        var a1 = parts.Length > 1 ? parts[1] : null;
        var a2 = parts.Length > 2 ? parts[2] : null;

        return cmd switch
        {
            "/start" or "/help" => Help(),
            "/status"     => await StatusAsync(ct),
            "/keys"       => await KeysAsync(ct),
            "/balance"    => await BalanceAsync(a1, ct),
            "/usage"      => await UsageAsync(a1 ?? "24h", ct),
            "/opencode"   => a1 is null ? "Usage: /opencode <name>" : await OpenCodeAsync(a1, ct),
            "/newkey"     => a1 is null ? "Usage: /newkey <name> [quota]" : await NewKeyAsync(a1, a2, ct),
            "/topup"      => (a1 is null || a2 is null) ? "Usage: /topup <name> <tokens>" : await TopUpAsync(a1, a2, ct),
            "/setquota"   => (a1 is null || a2 is null) ? "Usage: /setquota <name> <tokens>" : Arm(userId, a1, a2, PendingKind.SetQuota, ct),
            "/clearquota" => a1 is null ? "Usage: /clearquota <name>" : Arm(userId, a1, "0", PendingKind.SetQuota, ct),
            "/revoke"     => a1 is null ? "Usage: /revoke <name>" : Arm(userId, a1, null, PendingKind.Revoke, ct),
            _             => $"Unknown command {Head(cmd)}. Try /help."
        };
    }

    private static string Help() => """
        Access management

        /status               infrastructure health
        /keys                 consumers and balances
        /balance [name]       one or all
        /usage [24h|7d]       tokens and requests

        /newkey <name> [n]    create a key, seed quota, return OpenCode config
        /opencode <name>      re-emit the OpenCode config for a consumer
        /revoke <name>        delete a key and its ledger entry (confirm)

        /topup <name> <n>     add tokens
        /setquota <name> <n>  overwrite the balance (confirm)
        /clearquota <name>    set the balance to zero (confirm)
        """;

    // ---- reads ------------------------------------------------------------

    private async Task<string> StatusAsync(CancellationToken ct)
    {
        var sb = new StringBuilder("Infrastructure\n\n");

        var gw = await TryAsync(async () =>
        {
            using var r = await http.CreateClient("gateway").GetAsync("v1/models", ct);
            return r.IsSuccessStatusCode ? "reachable" : $"HTTP {(int)r.StatusCode}";
        });
        sb.Append("gateway   ").Append(gw).Append('\n');

        var led = await TryAsync(async () => $"{(await ledger.ListAsync(ct)).Count} consumers");
        sb.Append("ledger    ").Append(led).Append('\n');

        var prom = await TryAsync(async () =>
        {
            using var r = await http.CreateClient("prometheus").GetAsync("-/healthy", ct);
            return r.IsSuccessStatusCode ? "healthy" : $"HTTP {(int)r.StatusCode}";
        });
        sb.Append("prometheus ").Append(prom).Append('\n');

        var api = await TryAsync(async () => $"{(await keys.ReadConsumersAsync(ct)).Count} entries in key-auth");
        sb.Append("key-auth  ").Append(api);
        return sb.ToString();
    }

    private async Task<string> KeysAsync(CancellationToken ct)
    {
        var balances = await ledger.ListAsync(ct);
        var consumers = await keys.ReadConsumersAsync(ct);
        if (consumers.Count == 0) return "No consumers configured.";

        var sb = new StringBuilder("Consumers\n\n");
        foreach (var name in consumers.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            var bal = balances.TryGetValue(name, out var b) ? b.ToString("N0", CultureInfo.InvariantCulture) : "not seeded";
            sb.Append(name.PadRight(16)).Append(bal).Append('\n');
        }
        sb.Append("\nCredentials are not shown here. Use /opencode <name>.");
        return sb.ToString();
    }

    private async Task<string> BalanceAsync(string? name, CancellationToken ct)
    {
        if (name is not null)
        {
            var q = await QuotaGetAsync(name, ct);
            return q is null ? $"{name}: no balance recorded (never seeded, or Redis is unreachable)"
                             : $"{name}: {q.Value:N0} tokens";
        }
        var all = await ledger.ListAsync(ct);
        if (all.Count == 0) return "No balances recorded.";
        var sb = new StringBuilder("Balances\n\n");
        foreach (var (k, v) in all.OrderBy(x => x.Key, StringComparer.Ordinal))
            sb.Append(k.PadRight(16)).Append(v.ToString("N0", CultureInfo.InvariantCulture)).Append('\n');
        return sb.ToString();
    }

    private async Task<string> UsageAsync(string window, CancellationToken ct)
    {
        if (window is not ("24h" or "7d" or "1h" or "30d")) return "Window must be one of 1h, 24h, 7d, 30d.";
        var tokens = await PromAsync($"sum by (ai_consumer) (increase(route_upstream_model_consumer_metric_total_token[{window}]))", ct);
        var reqs   = await PromAsync($"sum by (ai_consumer) (increase(route_upstream_model_consumer_metric_llm_duration_count[{window}]))", ct);
        if (tokens.Count == 0) return $"No usage recorded in the last {window}.";

        var sb = new StringBuilder($"Usage, last {window}\n\n");
        foreach (var (k, v) in tokens.OrderByDescending(x => x.Value))
        {
            var r = reqs.TryGetValue(k, out var rv) ? rv : 0;
            sb.Append(k.PadRight(16))
              .Append(v.ToString("N0", CultureInfo.InvariantCulture)).Append(" tok  ")
              .Append(r.ToString("N0", CultureInfo.InvariantCulture)).Append(" req\n");
        }
        sb.Append("\nCounters reset when the gateway restarts; balances are the billing record.");
        return sb.ToString();
    }

    // ---- key lifecycle ----------------------------------------------------

    private async Task<string> NewKeyAsync(string name, string? quotaArg, CancellationToken ct)
    {
        if (!IsValidName(name)) return "Name must be 1-32 chars of a-z, 0-9, - or _.";
        var existing = await keys.ReadConsumersAsync(ct);
        if (existing.ContainsKey(name)) return $"{name} already exists. Use /revoke first, or /opencode to re-read its config.";

        var quota = 1_000_000L;
        if (quotaArg is not null && !TryParseTokens(quotaArg, out quota)) return "Quota must be a positive whole number.";

        var credential = "Bearer sk-" + Base62(32);
        await keys.AddAsync(name, credential, ct);

        // Seed BEFORE announcing success. ai-quota returns the same 403 for
        // "never seeded" as for "exhausted", so an unseeded key looks broken in
        // a way that wastes an afternoon.
        await QuotaSetAsync(name, quota, ct);
        await AuditAsync($"newkey name={name} quota={quota}", ct);

        return $"Created {name} with {quota:N0} tokens.\n\nOpenCode config — this credential is shown once:\n\n{OpenCodeJson(credential)}";
    }

    private async Task<string> OpenCodeAsync(string name, CancellationToken ct)
    {
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.TryGetValue(name, out var credential)) return $"No consumer named {name}.";
        await AuditAsync($"opencode name={name}", ct);
        return $"OpenCode config for {name}:\n\n{OpenCodeJson(credential)}";
    }

    private string OpenCodeJson(string credential)
    {
        // The credential in consumers.conf carries the literal "Bearer " prefix,
        // because key-auth matches the raw header value. An OpenAI-compatible
        // client adds its own "Bearer ", so strip it here or the gateway sees
        // "Bearer Bearer sk-..." and refuses.
        var apiKey = credential.StartsWith("Bearer ", StringComparison.Ordinal)
            ? credential["Bearer ".Length..]
            : credential;

        var doc = new JsonObject
        {
            ["$schema"] = "https://opencode.ai/config.json",
            ["provider"] = new JsonObject
            {
                ["qwen-gw"] = new JsonObject
                {
                    ["npm"] = "@ai-sdk/openai-compatible",
                    ["name"] = "Qwen3.8-27B (A100 gateway)",
                    ["options"] = new JsonObject
                    {
                        ["baseURL"] = $"{cfg.PublicBaseUrl}/v1",
                        ["apiKey"] = apiKey
                    },
                    ["models"] = new JsonObject
                    {
                        [cfg.ModelId] = new JsonObject
                        {
                            ["name"] = cfg.ModelId,
                            ["limit"] = new JsonObject
                            {
                                ["context"] = cfg.ContextLimit,
                                ["output"] = cfg.OutputLimit
                            }
                        }
                    }
                }
            },
            ["model"] = $"qwen-gw/{cfg.ModelId}"
        };
        return doc.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
    }

    private string Arm(long userId, string name, string? amount, PendingKind kind, CancellationToken _)
    {
        var expires = DateTimeOffset.UtcNow.AddSeconds(60);
        switch (kind)
        {
            case PendingKind.SetQuota:
                if (amount is null || !TryParseTokens(amount, out var target)) return "Amount must be a whole number of tokens.";
                _pending[userId] = new Pending(expires, async ct =>
                {
                    var before = await QuotaGetAsync(name, ct);
                    await QuotaSetAsync(name, target, ct);
                    await AuditAsync($"setquota name={name} from={before?.ToString(CultureInfo.InvariantCulture) ?? "none"} to={target}", ct);
                    return $"{name}: balance set to {target:N0} (was {before?.ToString("N0", CultureInfo.InvariantCulture) ?? "unset"}).";
                });
                return $"This OVERWRITES {name}'s balance with {target:N0} tokens — it does not add to it.\nReply CONFIRM within 60s to apply.";

            case PendingKind.Revoke:
                _pending[userId] = new Pending(expires, async ct =>
                {
                    if (!await keys.RemoveAsync(name, ct)) return $"No consumer named {name}.";
                    await ledger.DeleteAsync(name, ct);
                    await AuditAsync($"revoke name={name}", ct);
                    return $"Revoked {name}. Its key no longer authenticates and its ledger entry is gone.";
                });
                return $"This permanently revokes {name}'s key and deletes its balance.\nReply CONFIRM within 60s to apply.";

            default:
                return "Unsupported operation.";
        }
    }

    private async Task<string> TopUpAsync(string name, string amount, CancellationToken ct)
    {
        if (!TryParseTokens(amount, out var delta)) return "Amount must be a positive whole number of tokens.";
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.ContainsKey(name)) return $"No consumer named {name}.";

        var body = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", name),
            new KeyValuePair<string, string>("value", delta.ToString(CultureInfo.InvariantCulture))
        ]);
        using var r = await http.CreateClient("gateway").PostAsync("v1/chat/completions/quota/delta", body, ct);
        if (!r.IsSuccessStatusCode) return $"Top-up failed: HTTP {(int)r.StatusCode}.";

        var now = await QuotaGetAsync(name, ct);
        await AuditAsync($"topup name={name} delta={delta}", ct);
        return $"{name}: +{delta:N0}, balance now {now?.ToString("N0", CultureInfo.InvariantCulture) ?? "unknown"}.";
    }

    // ---- upstream calls ---------------------------------------------------

    private async Task<long?> QuotaGetAsync(string name, CancellationToken ct)
    {
        using var r = await http.CreateClient("gateway")
            .GetAsync($"v1/chat/completions/quota?consumer={Uri.EscapeDataString(name)}", ct);
        if (!r.IsSuccessStatusCode) return null;
        var q = await r.Content.ReadFromJsonAsync(BotJson.Default.QuotaResponse, ct);
        return q?.Quota;
    }

    private async Task QuotaSetAsync(string name, long value, CancellationToken ct)
    {
        var body = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", name),
            new KeyValuePair<string, string>("quota", value.ToString(CultureInfo.InvariantCulture))
        ]);
        using var r = await http.CreateClient("gateway").PostAsync("v1/chat/completions/quota/refresh", body, ct);
        r.EnsureSuccessStatusCode();
    }

    private async Task<Dictionary<string, double>> PromAsync(string query, CancellationToken ct)
    {
        var url = $"api/v1/query?query={Uri.EscapeDataString(query)}";
        using var r = await http.CreateClient("prometheus").GetAsync(url, ct);
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        if (!r.IsSuccessStatusCode) return result;

        var node = JsonNode.Parse(await r.Content.ReadAsStringAsync(ct));
        if (node?["data"]?["result"] is not JsonArray arr) return result;
        foreach (var item in arr)
        {
            var consumer = item?["metric"]?["ai_consumer"]?.GetValue<string>();
            var raw = item?["value"] is JsonArray v && v.Count > 1 ? v[1]?.GetValue<string>() : null;
            if (consumer is not null && double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
                result[consumer] = d;
        }
        return result;
    }

    private async Task SendAsync(long chatId, string text, CancellationToken ct)
    {
        // Telegram caps a message at 4096 characters. Chunk on line boundaries so
        // a long /keys listing does not lose its last consumer to a hard cut.
        foreach (var chunk in Chunk(text, 3800))
        {
            var payload = new SendMessage(chatId, chunk);
            using var content = new StringContent(
                JsonSerializer.Serialize(payload, BotJson.Default.SendMessage), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r = await http.CreateClient("telegram").PostAsync("sendMessage", content, ct);
            if (!r.IsSuccessStatusCode)
                log.LogError("sendMessage failed: HTTP {Code} {Body}",
                    (int)r.StatusCode, await r.Content.ReadAsStringAsync(ct));
        }
    }

    private async Task AuditAsync(string line, CancellationToken ct)
    {
        // The bot mutates access and money. Without this the only record of who
        // got a key would be the key itself.
        var stamped = $"{DateTimeOffset.UtcNow:O} {line}\n";
        try { await File.AppendAllTextAsync(cfg.AuditPath, stamped, ct); }
        catch (Exception ex) { log.LogError(ex, "audit write failed: {Line}", line); }
    }

    // ---- helpers ----------------------------------------------------------

    private bool MarkSeen(long id)
    {
        if (!_seen.Add(id)) return false;
        _seenOrder.Enqueue(id);
        while (_seenOrder.Count > 1000) _seen.Remove(_seenOrder.Dequeue());
        return true;
    }

    private static async Task<string> TryAsync(Func<Task<string>> f)
    {
        try { return await f(); } catch (Exception ex) { return $"FAILED ({ex.GetType().Name})"; }
    }

    private static IEnumerable<string> Chunk(string s, int max)
    {
        if (s.Length <= max) { yield return s; yield break; }
        var sb = new StringBuilder();
        foreach (var line in s.Split('\n'))
        {
            if (sb.Length + line.Length + 1 > max) { yield return sb.ToString(); sb.Clear(); }
            sb.Append(line).Append('\n');
        }
        if (sb.Length > 0) yield return sb.ToString();
    }

    private static bool IsValidName(string s) =>
        s.Length is > 0 and <= 32 && s.All(c => char.IsAsciiLetterLower(c) || char.IsAsciiDigit(c) || c is '-' or '_');

    private static bool TryParseTokens(string s, out long v) =>
        long.TryParse(s.Replace("_", "").Replace(",", ""), NumberStyles.Integer, CultureInfo.InvariantCulture, out v) && v >= 0;

    private static string Head(string s) => s.Length <= 40 ? s : s[..40] + "…";

    private static string Base62(int len)
    {
        const string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        var bytes = RandomNumberGenerator.GetBytes(len);
        return string.Create(len, bytes, (span, b) =>
        {
            for (var i = 0; i < span.Length; i++) span[i] = alphabet[b[i] % alphabet.Length];
        });
    }
}

// ===========================================================================
// KeyStore — owns consumers.conf and the live key-auth object.
//
// consumers.conf is gitignored and always has been, so the bot owning it breaks
// no git invariant. The committed artefact is config/wasmplugins/key-auth.yaml,
// and apply.sh renders the same object from the same consumer table — so the two
// writers cannot diverge. Last writer wins and both are correct.
//
// Writes go through the apiserver rather than the filesystem because conf/ is
// root-owned 0700 and reaching it would mean running as root or mounting the
// Docker socket. The apiserver accepts anonymous requests on higress-net.
// ===========================================================================
sealed class KeyStore(BotConfig cfg, IHttpClientFactory http, ILogger<KeyStore> log)
{
    private const string ObjectPath =
        "apis/extensions.higress.io/v1alpha1/namespaces/higress-system/wasmplugins/key-auth";

    private readonly SemaphoreSlim _lock = new(1, 1);

    public async Task<Dictionary<string, string>> ReadConsumersAsync(CancellationToken ct)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!File.Exists(cfg.ConsumersPath)) return map;
        foreach (var raw in await File.ReadAllLinesAsync(cfg.ConsumersPath, ct))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            var sp = line.IndexOfAny([' ', '\t']);
            if (sp <= 0) continue;
            map[line[..sp]] = line[(sp + 1)..].Trim();
        }
        return map;
    }

    public async Task AddAsync(string name, string credential, CancellationToken ct)
    {
        await _lock.WaitAsync(ct);
        try
        {
            await File.AppendAllTextAsync(cfg.ConsumersPath, $"{name}          {credential}\n", ct);
            await PushAsync(ct);
        }
        finally { _lock.Release(); }
    }

    public async Task<bool> RemoveAsync(string name, CancellationToken ct)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var lines = await File.ReadAllLinesAsync(cfg.ConsumersPath, ct);
            var kept = lines.Where(l =>
            {
                var t = l.Trim();
                if (t.Length == 0 || t[0] == '#') return true;
                var sp = t.IndexOfAny([' ', '\t']);
                return sp <= 0 || !string.Equals(t[..sp], name, StringComparison.Ordinal);
            }).ToArray();

            if (kept.Length == lines.Length) return false;
            await File.WriteAllLinesAsync(cfg.ConsumersPath, kept, ct);
            await PushAsync(ct);
            return true;
        }
        finally { _lock.Release(); }
    }

    // Read the live object, replace only the consumer table and the allow list,
    // and PUT it back. Working on the live JSON rather than re-rendering the
    // YAML template keeps resourceVersion intact — PATCH does not work on these
    // custom resources, and a PUT without the current resourceVersion is
    // rejected as a conflict.
    private async Task PushAsync(CancellationToken ct)
    {
        var consumers = await ReadConsumersAsync(ct);
        var client = http.CreateClient("apiserver");

        using var get = await client.GetAsync(ObjectPath, ct);
        get.EnsureSuccessStatusCode();
        var obj = JsonNode.Parse(await get.Content.ReadAsStringAsync(ct))!.AsObject();

        var list = new JsonArray();
        var allow = new JsonArray();
        foreach (var (name, credential) in consumers.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            // Typed as JsonNode deliberately. Passing a JsonObject or a string
            // directly binds to JsonArray.Add<T>(T), which wraps the value via
            // JsonValue.Create<T> — reflection, and an IL2026/IL3050 pair that
            // AOT turns into a runtime failure. The JsonNode locals force the
            // non-generic Add(JsonNode?) overload instead.
            JsonNode entry = new JsonObject { ["name"] = name, ["credential"] = credential };
            JsonNode allowed = JsonValue.Create(name);
            list.Add(entry);
            allow.Add(allowed);
        }

        obj["spec"]!["defaultConfig"]!["consumers"] = list;
        if (obj["spec"]!["matchRules"] is JsonArray rules)
            foreach (var rule in rules)
                if (rule?["config"] is JsonObject rc) rc["allow"] = allow.DeepClone();

        using var body = new StringContent(obj.ToJsonString(), Encoding.UTF8);
        body.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        using var put = await client.PutAsync(ObjectPath, body, ct);
        if (!put.IsSuccessStatusCode)
            throw new InvalidOperationException(
                $"key-auth PUT failed: HTTP {(int)put.StatusCode} {await put.Content.ReadAsStringAsync(ct)}");

        log.LogInformation("pushed key-auth with {Count} consumers", consumers.Count);
    }
}

// ===========================================================================
// Ledger — a minimal RESP client. SCAN, MGET and DEL only.
//
// Hand-rolled rather than StackExchange.Redis: three commands do not justify a
// dependency that has been working through AOT trim warnings as recently as
// 3.1.31, and this is AOT-safe by construction. Writes that change a balance go
// through the ai-quota admin API instead, so the plugin's own semantics apply.
// ===========================================================================
sealed class Ledger(BotConfig cfg)
{
    private const string Prefix = "chat_quota:";

    public async Task<Dictionary<string, long>> ListAsync(CancellationToken ct)
    {
        var result = new Dictionary<string, long>(StringComparer.Ordinal);
        using var c = await ConnectAsync(ct);

        var keys = new List<string>();
        var cursor = "0";
        do
        {
            var reply = await c.CommandAsync(ct, "SCAN", cursor, "MATCH", Prefix + "*", "COUNT", "200");
            if (reply is not object?[] { Length: 2 } page) break;
            cursor = page[0] as string ?? "0";
            if (page[1] is object?[] batch)
                foreach (var k in batch) if (k is string s) keys.Add(s);
        } while (cursor != "0");

        if (keys.Count == 0) return result;

        var args = new List<string> { "MGET" };
        args.AddRange(keys);
        if (await c.CommandAsync(ct, [.. args]) is object?[] values)
            for (var i = 0; i < keys.Count && i < values.Length; i++)
                if (values[i] is string v && long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n))
                    result[keys[i][Prefix.Length..]] = n;

        return result;
    }

    public async Task DeleteAsync(string name, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        await c.CommandAsync(ct, "DEL", Prefix + name);
    }

    private async Task<RespConnection> ConnectAsync(CancellationToken ct)
    {
        var socket = new TcpClient();
        await socket.ConnectAsync(cfg.RedisHost, cfg.RedisPort, ct);
        return new RespConnection(socket);
    }
}

sealed class RespConnection(TcpClient client) : IDisposable
{
    private readonly NetworkStream _s = client.GetStream();
    private readonly byte[] _buf = new byte[64 * 1024];
    private int _len, _pos;

    public async Task<object?> CommandAsync(CancellationToken ct, params string[] args)
    {
        var sb = new StringBuilder().Append('*').Append(args.Length).Append("\r\n");
        foreach (var a in args)
            sb.Append('$').Append(Encoding.UTF8.GetByteCount(a)).Append("\r\n").Append(a).Append("\r\n");
        await _s.WriteAsync(Encoding.UTF8.GetBytes(sb.ToString()), ct);
        return await ReadAsync(ct);
    }

    private async Task<object?> ReadAsync(CancellationToken ct)
    {
        var type = (char)await ReadByteAsync(ct);
        var line = await ReadLineAsync(ct);
        switch (type)
        {
            case '+': return line;
            case ':': return long.Parse(line, CultureInfo.InvariantCulture);
            case '-': throw new InvalidOperationException("redis error: " + line);
            case '$':
            {
                var n = int.Parse(line, CultureInfo.InvariantCulture);
                if (n < 0) return null;
                var bytes = new byte[n];
                for (var i = 0; i < n; i++) bytes[i] = await ReadByteAsync(ct);
                await ReadByteAsync(ct); await ReadByteAsync(ct); // trailing CRLF
                return Encoding.UTF8.GetString(bytes);
            }
            case '*':
            {
                var n = int.Parse(line, CultureInfo.InvariantCulture);
                if (n < 0) return null;
                var arr = new object?[n];
                for (var i = 0; i < n; i++) arr[i] = await ReadAsync(ct);
                return arr;
            }
            default: throw new InvalidOperationException($"unexpected RESP type '{type}'");
        }
    }

    private async Task<byte> ReadByteAsync(CancellationToken ct)
    {
        if (_pos >= _len)
        {
            _len = await _s.ReadAsync(_buf, ct);
            _pos = 0;
            if (_len <= 0) throw new EndOfStreamException("redis closed the connection");
        }
        return _buf[_pos++];
    }

    private async Task<string> ReadLineAsync(CancellationToken ct)
    {
        var sb = new StringBuilder();
        while (true)
        {
            var b = await ReadByteAsync(ct);
            if (b == (byte)'\r') { await ReadByteAsync(ct); return sb.ToString(); }
            sb.Append((char)b);
        }
    }

    public void Dispose() { _s.Dispose(); client.Dispose(); }
}

// ===========================================================================
// Types and JSON. Every type crossing the wire needs a [JsonSerializable]
// entry, or serialisation throws once trimmed.
// ===========================================================================
sealed record BotConfig(
    string BotToken, string WebhookSecret, string WebhookPath, HashSet<long> AllowedIds,
    string AdminCredential, string GatewayUrl, string ApiServerUrl, string RedisHost, int RedisPort,
    string PrometheusUrl, string PublicBaseUrl, string ConsumersPath, string AuditPath,
    string ModelId, int ContextLimit, int OutputLimit);

enum PendingKind { SetQuota, Revoke }
sealed record Pending(DateTimeOffset Expires, Func<CancellationToken, Task<string>> Run);

sealed class Update
{
    [JsonPropertyName("update_id")] public long UpdateId { get; set; }
    [JsonPropertyName("message")]   public Message? Message { get; set; }
}
sealed class Message
{
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("from")] public User? From { get; set; }
    [JsonPropertyName("chat")] public Chat? Chat { get; set; }
}
sealed class User
{
    [JsonPropertyName("id")]       public long Id { get; set; }
    [JsonPropertyName("username")] public string? Username { get; set; }
}
sealed class Chat
{
    [JsonPropertyName("id")]   public long Id { get; set; }
    [JsonPropertyName("type")] public string? Type { get; set; }
}
sealed record SendMessage(
    [property: JsonPropertyName("chat_id")] long ChatId,
    [property: JsonPropertyName("text")] string Text);

sealed class QuotaResponse
{
    [JsonPropertyName("consumer")] public string? Consumer { get; set; }
    [JsonPropertyName("quota")]    public long Quota { get; set; }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(Update))]
[JsonSerializable(typeof(Message))]
[JsonSerializable(typeof(User))]
[JsonSerializable(typeof(Chat))]
[JsonSerializable(typeof(SendMessage))]
[JsonSerializable(typeof(QuotaResponse))]
internal partial class BotJson : JsonSerializerContext;
