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
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Runtime;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
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

static HashSet<long> Ids(string raw) =>
    raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
       .Select(s => long.Parse(s, CultureInfo.InvariantCulture))
       .ToHashSet();

var allowedIds = Ids(Req("TELEGRAM_ALLOWED_IDS"));

// Where alerts land. Defaults to everyone who may operate the bot, which is
// the right default for a handful of operators: an alert nobody is guaranteed
// to see is the failure mode this whole path exists to remove. Set
// ALERT_CHAT_IDS to a group chat id to send one copy there instead.
var alertChatIds = Ids(Opt("ALERT_CHAT_IDS", "")) is { Count: > 0 } ids ? ids : allowedIds;

var cfg = new BotConfig(
    BotToken:        Req("TELEGRAM_BOT_TOKEN"),
    WebhookSecret:   Req("TELEGRAM_WEBHOOK_SECRET"),
    WebhookPath:     new Uri(Req("TELEGRAM_WEBHOOK_URL")).AbsolutePath,
    AllowedIds:      allowedIds,
    AdminCredential: Req("QUOTA_ADMIN_CREDENTIAL"),
    GatewayUrl:      Opt("GATEWAY_URL", "http://higress:80").TrimEnd('/'),
    ApiServerUrl:    Opt("APISERVER_URL", "https://apiserver.svc:8443").TrimEnd('/'),
    RedisHost:       Opt("REDIS_HOST", "higress-redis"),
    RedisPort:       int.Parse(Opt("REDIS_PORT", "6379"), CultureInfo.InvariantCulture),
    PrometheusUrl:   Opt("PROMETHEUS_URL", "http://qwen36-27b-prometheus:9090").TrimEnd('/'),
    PublicBaseUrl:   Opt("PUBLIC_BASE_URL", "https://gateway.example.org").TrimEnd('/'),
    KubeConfigPath:  Opt("KUBECONFIG_PATH", "/etc/kube/config"),
    ConsumersPath:   Opt("CONSUMERS_PATH", "/data/consumers.conf"),
    AuditPath:       Opt("AUDIT_PATH", "/data/audit.log"),
    ModelId:         Opt("MODEL_ID", "qwen3.8-27b"),
    ContextLimit:    int.Parse(Opt("MODEL_CONTEXT", "169000"), CultureInfo.InvariantCulture),
    OutputLimit:     int.Parse(Opt("MODEL_OUTPUT", "70000"), CultureInfo.InvariantCulture),
    AlertSecret:     Req("ALERT_WEBHOOK_SECRET"),
    // Shared with admin-mcp, which is the only caller of /admin/*. Optional:
    // unset means those endpoints are not mapped at all, which is the right
    // default for an install that has no MCP server in front of it.
    AdminApiSecret:  Opt("ADMIN_API_SECRET", ""),
    AlertmanagerUrl: Opt("ALERTMANAGER_URL", "http://qwen36-27b-alertmanager:9093").TrimEnd('/'),
    AlertChatIds:    alertChatIds,
    LangfuseUrl:     Opt("LANGFUSE_PUBLIC_URL", "https://langfuse.example.org").TrimEnd('/'),
    // Only used to build a deep link to a consumer's traces. Empty is a
    // supported state: /key then prints the path to click through by hand
    // instead of offering a button that would 404.
    LangfuseProjectId: Opt("LANGFUSE_PROJECT_ID", ""));

// Encoded once, not on every delivery — and validated here because the
// comparison's fast path assumes one byte per character. That holds for the
// alphabet Telegram allows; enforcing it now turns "someone typed a non-ASCII
// character into .env" into a startup error naming the variable, instead of a
// bot that silently rejects every webhook it is sent.
if (cfg.WebhookSecret.Length is < 1 or > 256 ||
    !cfg.WebhookSecret.All(c => char.IsAsciiLetterOrDigit(c) || c is '_' or '-'))
    throw new InvalidOperationException(
        "TELEGRAM_WEBHOOK_SECRET must be 1-256 characters of [A-Za-z0-9_-] — Telegram's own constraint");
var secretBytes = Encoding.UTF8.GetBytes(cfg.WebhookSecret);

// Same one-byte-per-character assumption as the Telegram secret above, for the
// same reason: SecretMatches compares bytes against chars.
if (!cfg.AlertSecret.All(char.IsAscii))
    throw new InvalidOperationException("ALERT_WEBHOOK_SECRET must be ASCII");
var alertSecretBytes = Encoding.UTF8.GetBytes(cfg.AlertSecret);

var builder = WebApplication.CreateSlimBuilder(args);
builder.Logging.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "yyyy-MM-ddTHH:mm:ssZ "; o.UseUtcTimestamp = true; });

// The framework loggers emit five lines per HTTP request, and the compose
// healthcheck fires every 30s: 75% of this container's log was /healthz, which
// buried the command traffic it exists to show. Both are demoted to Warning and
// the bot logs what actually matters itself — one line per command, with its
// end-to-end duration. LOG_HTTP=debug restores the per-request framework
// tracing when something needs taking apart again.
if (!string.Equals(Opt("LOG_HTTP", ""), "debug", StringComparison.OrdinalIgnoreCase))
{
    builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);
    builder.Logging.AddFilter("System.Net.Http.HttpClient", LogLevel.Warning);
}
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

// Alert groups awaiting delivery. Bounded and DropWrite like the update queue:
// if Telegram is unreachable long enough to fill this, the newest alerts are
// the ones worth keeping, and an unbounded queue would just turn a delivery
// outage into a memory leak.
var alertQueue = Channel.CreateBounded<AmWebhook>(new BoundedChannelOptions(64)
{
    SingleReader = true,
    SingleWriter = false,
    FullMode = BoundedChannelFullMode.DropWrite
});

builder.Services.AddSingleton(cfg);
builder.Services.AddSingleton(queue);
builder.Services.AddSingleton(alertQueue);
builder.Services.AddSingleton<Telegram>();
builder.Services.AddHostedService<AlertWorker>();
builder.Services.AddSingleton<Ledger>();
builder.Services.AddSingleton<KeyStore>();
builder.Services.AddHostedService<Worker>();
builder.Services.AddSingleton<LimiterSync>();
builder.Services.AddHostedService<EnforcementWorker>();

// api.telegram.org is ~100ms away, and opening a connection to it costs ~200ms
// more (TCP 100ms + TLS 106ms, measured from this host on 2026-09-03). Every
// command is at least one round trip, so connection reuse — not the work the
// commands do, which runs in single-digit milliseconds — decides how fast this
// bot feels. Three settings that only work together:
//
//   HTTP/2            one multiplexed connection instead of one per concurrent
//                     call. api.telegram.org negotiates h2.
//   keep-alive pings  h2 PING frames hold the connection open across idle gaps.
//                     Without them the far-side nginx closes it after ~75s, so
//                     a command typed minutes after the last one pays the full
//                     handshake — which is the common case, because real use is
//                     bursty with long gaps between bursts.
//   handler lifetime  IHttpClientFactory rotates handlers every 2 MINUTES by
//                     default and the connection pool dies with them, which
//                     would quietly undo both of the above. That rotation
//                     exists to pick up DNS changes; SocketsHttpHandler already
//                     re-resolves on PooledConnectionLifetime, so disabling it
//                     costs nothing here.
builder.Services.AddHttpClient("telegram", c =>
{
    c.BaseAddress = new Uri($"https://api.telegram.org/bot{cfg.BotToken}/");
    c.Timeout = TimeSpan.FromSeconds(20);
    c.DefaultRequestVersion = HttpVersion.Version20;
    c.DefaultVersionPolicy = HttpVersionPolicy.RequestVersionOrLower;
})
.SetHandlerLifetime(Timeout.InfiniteTimeSpan)
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(10),
    PooledConnectionLifetime    = TimeSpan.FromMinutes(30),
    KeepAlivePingDelay          = TimeSpan.FromSeconds(30),
    KeepAlivePingTimeout        = TimeSpan.FromSeconds(10),
    KeepAlivePingPolicy         = HttpKeepAlivePingPolicy.Always
});
builder.Services.AddHttpClient("gateway", c =>
{
    c.BaseAddress = new Uri(cfg.GatewayUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
    c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", cfg.AdminCredential);
});
// The gateway client above is for management calls and stays at 15s so a
// wedged gateway cannot hold a command open. Generating a report is a
// different shape of request entirely — a few thousand tokens at ~100 tok/s —
// so it gets its own client rather than loosening the timeout for everything.
// Same base address, same admin credential.
builder.Services.AddHttpClient("inference", c =>
{
    c.BaseAddress = new Uri(cfg.GatewayUrl + "/");
    c.Timeout = TimeSpan.FromMinutes(5);
    c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", cfg.AdminCredential);
});
builder.Services.AddHttpClient("alertmanager", c =>
{
    c.BaseAddress = new Uri(cfg.AlertmanagerUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(5);
});

builder.Services.AddHttpClient("prometheus", c =>
{
    c.BaseAddress = new Uri(cfg.PrometheusUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
});
// Public price API, read once a day by PriceBook. Short timeout: a command
// that happens to trigger the refresh should not hang on it.
builder.Services.AddHttpClient("openrouter", c =>
{
    c.BaseAddress = new Uri("https://openrouter.ai/api/v1/");
    c.Timeout = TimeSpan.FromSeconds(10);
});
builder.Services.AddSingleton<PriceBook>();
builder.Services.AddHttpClient("apiserver", c =>
{
    c.BaseAddress = new Uri(cfg.ApiServerUrl + "/");
    c.Timeout = TimeSpan.FromSeconds(15);
})
// Client-certificate auth, using the SAME credential the controller and console
// present — read from the kubeconfig the deployment already writes, rather than
// copying the cert into this project. Copying would mean a second place to
// rotate, and a silent 401 the day someone rotates only one of them.
//
// Server-cert validation stays off: the apiserver's certificate is self-signed
// with a SAN that does not cover the compose alias, and the connection never
// leaves the host's docker network. The authentication that matters here is
// ours TO it, which the client certificate provides.
.ConfigurePrimaryHttpMessageHandler(() =>
{
    var handler = new HttpClientHandler
    {
        ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator
    };
    if (KubeClientCertificate.TryLoad(cfg.KubeConfigPath, out var cert) && cert is not null)
    {
        handler.ClientCertificates.Add(cert);
        Console.WriteLine($"apiserver client certificate loaded: subject={cert.Subject} notAfter={cert.NotAfter:O}");
    }
    else
    {
        // Not fatal while the apiserver still allows anonymous access, but it
        // WILL be the moment --auth-enabled is set, and the failure then is a
        // 401 on every key write with nothing explaining why.
        Console.WriteLine($"WARNING: no apiserver client certificate from {cfg.KubeConfigPath} - key writes will fail once the apiserver requires auth");
    }
    return handler;
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
    if (!SecretMatches(req.Headers["X-Telegram-Bot-Api-Secret-Token"].ToString(), secretBytes))
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
        //
        // Message only, no stack trace. This endpoint is on the public internet
        // and anything that guesses the path can post junk at it, so an
        // unparseable body is expected background noise rather than a defect
        // here — and the 30-line AOT stack it produced buried everything else.
        log.LogWarning("dropped unparseable update: {Reason}", ex.Message);
        return Results.Ok();
    }

    if (update is not null && !queue.Writer.TryWrite(update))
        log.LogError("queue full, dropped update {UpdateId}", update.UpdateId);

    return Results.Ok();
});

// ---------------------------------------------------------------------------
// Alertmanager's webhook. Same discipline as the Telegram handler above:
// verify, enqueue, 200, and nothing slow in the request path.
//
// Reachable only from the `edge` docker network — Caddy proxies /tg/<random>
// to this process and nothing else, so this path is not on the public
// internet. It is authenticated anyway: the bearer is the only thing standing
// between "anything on edge" and the operators' alert channel.
//
// A parse failure answers 200 on purpose. Alertmanager retries non-2xx, and a
// payload that cannot be deserialised will not deserialise on the third
// attempt either — it would just pin one alert group in a retry loop forever.
// Delivery failures that ARE worth retrying (the bot being down) never reach
// this line.
// ---------------------------------------------------------------------------
// =============================================================================
// ADMIN API — the ONE writer for consumer identity
//
// admin-mcp deliberately does not create or revoke keys itself. Consumers live
// in a single key-auth wasmplugin object, the Higress apiserver is file-backed
// and returns no resourceVersion, so a PUT is last-write-wins over the whole
// object. Two processes editing it with no shared lock silently drop one of two
// concurrent creations, and the symptom is a key that looks created and does
// not authenticate.
//
// KeyStore holds that lock, and it is a singleton in THIS process. Routing the
// MCP server's create/revoke through here means both interfaces serialise
// behind the same lock, write the same audit log, and generate credentials with
// the same rejection-sampled generator.
//
// Not mapped at all unless ADMIN_API_SECRET is set. It is reachable only on
// `edge`, from admin-mcp, and never published by Caddy.
// =============================================================================
if (cfg.AdminApiSecret.Length >= 32)
{
    var adminSecretBytes = Encoding.UTF8.GetBytes(cfg.AdminApiSecret);

    // Results.Json/BadRequest/NotFound with an anonymous type is reflection-based
    // serialisation and fails the AOT build (IL2026/IL3050) — the same family of
    // error as JsonArray.Add<T> recorded in the README. Every admin response is
    // therefore built as a JsonObject and written as text.
    static IResult J(int status, JsonObject o) =>
        Results.Text(o.ToJsonString(), "application/json", null, status);

    static bool AdminOk(HttpRequest req, byte[] secret)
    {
        var auth = req.Headers.Authorization.ToString();
        var presented = auth.StartsWith("Bearer ", StringComparison.Ordinal) ? auth[7..] : "";
        return SecretMatches(presented, secret);
    }

    // The tier table, so a caller can choose a tier by name instead of guessing
    // a token count. This is the same table /tiers renders and /tier validates
    // against — one definition, so advice and enforcement cannot drift.
    app.MapGet("/admin/tiers", (HttpRequest req) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();
        var arr = new JsonArray();
        // Enforcement is STRUCTURAL here, not a note in prose. A field that is
        // read as a promise and enforced by nothing ends up quoted to a
        // customer, so the one unenforced field says so in its own name, not
        // only in the note.
        foreach (var t in Policy.All)
            arr.Add((JsonNode)new JsonObject
            {
                ["tier"] = t.Name,
                ["for"] = t.For,
                ["quota"] = t.Quota,
                ["refill"] = Policy.RefillName(t.Refill),
                ["daily_limit"] = t.Daily,
                ["tokens_per_minute"] = t.Tpm,
                ["max_tokens_NOT_ENFORCED"] = t.MaxTokens,
                ["enforced"] = new JsonArray { (JsonNode)"quota", (JsonNode)"refill", (JsonNode)"daily_limit", (JsonNode)"tokens_per_minute" },
                ["note"] = "0 means no limit for daily_limit and tokens_per_minute, and the "
                         + $"gateway's global {cfg.OutputLimit} ceiling for max_tokens. daily_limit is a 24h "
                         + "window starting at the key's first request, not a calendar day; one request can "
                         + "overshoot a limit by its own size. refill resets the balance to quota at 00:00 UTC "
                         + "(daily, Monday, or the 1st). max_tokens cannot be enforced per key. Every value "
                         + "can be overridden per consumer (POST /admin/policy).",
            });
        return Results.Text(arr.ToJsonString(), "application/json");
    });

    // The reference prices, so admin-mcp prices usage with the same numbers the
    // bot shows instead of keeping a second copy that can drift.
    app.MapGet("/admin/prices", async (HttpRequest req, PriceBook book, CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();
        var p = await book.GetAsync(ct);
        static JsonObject Ref(PriceRef r) => new()
        {
            ["label"] = r.Label,
            ["input_per_m"] = r.InPerM,
            ["output_per_m"] = r.OutPerM,
            ["cache_read_per_m"] = r.CacheReadPerM,
            ["basis"] = r.Basis,
        };
        return J(200, new JsonObject
        {
            ["unit"] = "USD per million tokens",
            ["openrouter"] = Ref(p.OpenRouter),
            ["openrouter_providers"] = p.Providers,
            ["openrouter_output_min_per_m"] = p.OutMin,
            ["openrouter_output_max_per_m"] = p.OutMax,
            ["alibaba_singapore"] = Ref(p.AlibabaSg),
            ["alibaba_beijing"] = Ref(p.AlibabaBj),
            ["note"] = "Reference prices for the same model at public providers. Not what any consumer is charged.",
        });
    });

    // One consumer's effective settings: each value, and whether it comes from
    // the tier or was set on this consumer by hand.
    app.MapGet("/admin/policy/{name}", async (string name, HttpRequest req, KeyStore keys,
                                              Ledger ledger, CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();
        if (!Worker.IsValidName(name)) return J(400, new JsonObject { ["error"] = "invalid name" });
        if (!(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return J(404, new JsonObject { ["error"] = $"no consumer named '{name}'" });

        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        await Task.WhenAll(tierT, ovT);
        return J(200, PolicyJson(name, tierT.Result, ovT.Result));
    });

    // Set one value on one consumer, or put it back to the tier's with
    // "default". Never moves a balance: quota is what a refill grants, and the
    // balance is changed only by an explicit top-up or set.
    app.MapPost("/admin/policy", async (HttpRequest req, KeyStore keys, Ledger ledger,
                                        CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();

        JsonNode? body;
        try { body = await JsonNode.ParseAsync(req.Body, cancellationToken: ct); }
        catch (JsonException) { return J(400, new JsonObject { ["error"] = "malformed JSON body" }); }

        var name = body?["name"]?.GetValue<string>() ?? "";
        var fieldArg = body?["field"]?.GetValue<string>() ?? "";
        var valueArg = body?["value"]?.ToString() ?? "";
        if (!Worker.IsValidName(name)) return J(400, new JsonObject { ["error"] = "invalid name" });
        if (Policy.Field(fieldArg) is not { } field)
            return J(400, new JsonObject
            {
                ["error"] = $"unknown field '{fieldArg}'",
                ["known"] = new JsonArray(Policy.Fields.Select(f => (JsonNode)f.Key).ToArray()),
            });
        if (!(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return J(404, new JsonObject { ["error"] = $"no consumer named '{name}'" });

        string? stored = null;
        if (!Policy.IsDefaultWord(valueArg)
            && !Policy.TryNormalise(field, valueArg, cfg.OutputLimit, out stored, out var why))
            return J(400, new JsonObject { ["error"] = why });

        var before = await ledger.OverridesAsync(name, ct);
        if (stored is null) await ledger.ClearOverrideAsync(name, field.Key, ct);
        else await ledger.SetOverrideAsync(name, field.Key, stored, ct);

        await File.AppendAllTextAsync(cfg.AuditPath,
            $"{DateTimeOffset.UtcNow:O} admin-api policy name={name} field={field.Key} "
          + $"from={before.GetValueOrDefault(field.Key, "tier")} to={stored ?? "tier"}\n", ct);

        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        await Task.WhenAll(tierT, ovT);
        return J(200, PolicyJson(name, tierT.Result, ovT.Result));
    });

    JsonObject PolicyJson(string name, string? tier, Dictionary<string, string> overrides)
    {
        var settings = new JsonObject();
        foreach (var r in Policy.Resolve(tier, overrides))
            settings[r.Field.Key] = new JsonObject
            {
                ["value"] = r.Value,
                ["shown"] = r.Value is null ? "unset" : Policy.Show(r.Field, r.Value, cfg.OutputLimit),
                ["source"] = r.Source,
                ["enforced"] = r.Field.Enforced,
                ["status"] = r.Field.Status,
            };
        return new JsonObject
        {
            ["name"] = name,
            ["tier"] = tier ?? "unassigned",
            ["settings"] = settings,
            ["note"] = "source is 'tier' when the value follows the tier and 'set' when it was "
                     + "set on this consumer. Everything but max_tokens is enforced; "
                     + (LimiterSync.InScope(name) ? "" : "this consumer is OUTSIDE the limiter's rollout scope, so daily and tpm are not applied to it yet."),
        };
    }

    app.MapPost("/admin/keys", async (HttpRequest req, KeyStore keys, Ledger ledger,
                                      IHttpClientFactory http, CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();

        JsonNode? body;
        try { body = await JsonNode.ParseAsync(req.Body, cancellationToken: ct); }
        catch (JsonException) { return J(400, new JsonObject { ["error"] = "malformed JSON body" }); }

        var name = body?["name"]?.GetValue<string>() ?? "";
        var tier = body?["tier"]?.GetValue<string>();
        long? quota = body?["quota"] is { } q && long.TryParse(q.ToString(), out var qq) ? qq : null;

        if (!Worker.IsValidName(name))
            return J(400, new JsonObject { ["error"] = "name must be 1-32 chars of [a-z0-9_-]" });

        // A tier picks the quota; an explicit quota overrides it. Neither is
        // required, and the default matches what /newkey has always used.
        if (tier is { Length: > 0 })
        {
            if (!Policy.Tiers.TryGetValue(tier, out var t))
            {
                    var known = new JsonArray();
                    foreach (var k in Policy.Tiers.Keys) known.Add((JsonNode)k!);
                    return J(400, new JsonObject
                    {
                        ["error"] = $"unknown tier '{tier}'",
                        ["known"] = known,
                    });
            }
            quota ??= t.Quota;
        }
        var seed = quota ?? 1_000_000L;
        if (seed is < 0 or > 10_000_000_000)
            return J(400, new JsonObject { ["error"] = "quota out of range" });

        var existing = await keys.ReadConsumersAsync(ct);
        if (existing.ContainsKey(name))
            return J(409, new JsonObject { ["error"] = $"consumer '{name}' already exists" });

        var credential = "Bearer sk-" + Worker.Base62(32);
        await keys.AddAsync(name, credential, ct);

        // Seed BEFORE reporting success: ai-quota answers the same 403 for
        // "never seeded" as for "exhausted", so an unseeded key looks broken.
        // FORM-ENCODED, not JSON. ai-quota's endpoints parse a form body and
        // answer 403 to anything else, which reads exactly like an auth failure
        // and is not one. The refresh endpoint's field is `quota`; the delta
        // endpoint's is `value`. They do not match each other.
        using (var content = new FormUrlEncodedContent([
                   new KeyValuePair<string, string>("consumer", name),
                   new KeyValuePair<string, string>("quota", seed.ToString(CultureInfo.InvariantCulture)),
               ]))
        {
            using var r = await http.CreateClient("gateway")
                .PostAsync("v1/chat/completions/quota/refresh", content, ct);
            if (!r.IsSuccessStatusCode)
            {
                // Roll back the credential rather than leave a key that
                // authenticates and then 403s on every request.
                await keys.RemoveAsync(name, ct);
                return J(502, new JsonObject { ["error"] = $"quota seeding failed (HTTP {(int)r.StatusCode}); key not created" });
            }
        }

        if (tier is { Length: > 0 }) await ledger.SetTierAsync(name, tier, ct);

        await File.AppendAllTextAsync(cfg.AuditPath,
            $"{DateTimeOffset.UtcNow:O} admin-api newkey name={name} quota={seed} tier={tier ?? "-"}\n", ct);
        log.LogInformation("admin-api created consumer {Name} quota={Quota} tier={Tier}",
            name, seed, tier ?? "-");

        // The credential is returned ONCE, exactly as /newkey shows it once.
        return J(200, new JsonObject
        {
            ["name"] = name,
            ["credential"] = credential,
            ["quota"] = seed,
            ["tier"] = tier ?? "unassigned",
            ["note"] = "This credential is not stored anywhere in readable form and cannot be retrieved again.",
        });
    });

    app.MapDelete("/admin/keys/{name}", async (string name, HttpRequest req,
                                               KeyStore keys, Ledger ledger, CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();
        if (!Worker.IsValidName(name)) return J(400, new JsonObject { ["error"] = "invalid name" });

        if (!await keys.RemoveAsync(name, ct))
            return J(404, new JsonObject { ["error"] = $"no consumer named '{name}'" });
        await ledger.DeleteAsync(name, ct);

        await File.AppendAllTextAsync(cfg.AuditPath,
            $"{DateTimeOffset.UtcNow:O} admin-api revoke name={name}\n", ct);
        log.LogWarning("admin-api revoked consumer {Name}", name);
        return J(200, new JsonObject { ["name"] = name, ["revoked"] = true });
    });

    app.MapPost("/admin/tier", async (HttpRequest req, KeyStore keys, Ledger ledger,
                                      CancellationToken ct) =>
    {
        if (!AdminOk(req, adminSecretBytes)) return Results.Unauthorized();

        JsonNode? body;
        try { body = await JsonNode.ParseAsync(req.Body, cancellationToken: ct); }
        catch (JsonException) { return J(400, new JsonObject { ["error"] = "malformed JSON body" }); }

        var name = body?["name"]?.GetValue<string>() ?? "";
        var tier = body?["tier"]?.GetValue<string>() ?? "";
        if (!Worker.IsValidName(name)) return J(400, new JsonObject { ["error"] = "invalid name" });
        if (!Policy.Tiers.ContainsKey(tier))
        {
            var known = new JsonArray();
            foreach (var k in Policy.Tiers.Keys) known.Add((JsonNode)k!);
            return J(400, new JsonObject { ["error"] = $"unknown tier '{tier}'", ["known"] = known });
        }

        var existing = await keys.ReadConsumersAsync(ct);
        if (!existing.ContainsKey(name))
            return J(404, new JsonObject { ["error"] = $"no consumer named '{name}'" });

        await ledger.SetTierAsync(name, tier, ct);
        await File.AppendAllTextAsync(cfg.AuditPath,
            $"{DateTimeOffset.UtcNow:O} admin-api tier name={name} tier={tier}\n", ct);

        // Recorded, not enforced — the same caveat /tier carries. The response
        // is the consumer's whole effective policy, so a caller sees at once
        // which values the new tier moved and which were set by hand and kept.
        return J(200, PolicyJson(name, tier, await ledger.OverridesAsync(name, ct)));
    });

    log.LogInformation("admin API mapped at /admin/* (create, revoke, tier, policy)");
}
else if (cfg.AdminApiSecret.Length > 0)
{
    log.LogWarning("ADMIN_API_SECRET is set but shorter than 32 characters; /admin/* NOT mapped");
}

app.MapPost("/alert", async (HttpRequest req) =>
{
    var auth = req.Headers.Authorization.ToString();
    var presented = auth.StartsWith("Bearer ", StringComparison.Ordinal) ? auth[7..] : "";
    if (!SecretMatches(presented, alertSecretBytes))
    {
        log.LogWarning("rejected /alert with bad or missing bearer from {Ip}",
            req.HttpContext.Connection.RemoteIpAddress);
        return Results.Unauthorized();
    }

    AmWebhook? hook;
    try
    {
        hook = await JsonSerializer.DeserializeAsync(req.Body, BotJson.Default.AmWebhook);
    }
    catch (JsonException ex)
    {
        log.LogWarning("dropped unparseable alert payload: {Reason}", ex.Message);
        return Results.Ok();
    }

    if (hook?.Alerts is { Count: > 0 } && !alertQueue.Writer.TryWrite(hook))
        log.LogError("alert queue full, dropped group {Group}",
            hook.GroupLabels?.GetValueOrDefault("alertname") ?? "?");

    return Results.Ok();
});

app.MapGet("/healthz", () => Results.Text("ok"));

app.Run();
return 0;

static bool SecretMatches(ReadOnlySpan<char> presented, ReadOnlySpan<byte> expected)
{
    // Telegram caps secret_token at 256 characters and constrains it to
    // [A-Za-z0-9_-], and the value from .env is validated against exactly that
    // at startup — so the expected byte length is also its character length.
    // That lets a wrong-length header be rejected before anything is copied,
    // which is what keeps the stack buffer below provably in range: `presented`
    // arrives from the internet and its length is not ours to trust.
    //
    // A non-ASCII header of the right character count encodes to more than 256
    // bytes, TryGetBytes fails, and it is rejected — correct, since it cannot
    // equal an ASCII secret.
    //
    // Comparing lengths up front leaks the secret's length, not the secret, and
    // CryptographicOperations.FixedTimeEquals requires equal lengths regardless.
    const int MaxSecretBytes = 256;
    if (presented.Length != expected.Length) return false;

    Span<byte> buf = stackalloc byte[MaxSecretBytes];
    return Encoding.UTF8.TryGetBytes(presented, buf, out var n)
        && n == expected.Length
        && CryptographicOperations.FixedTimeEquals(buf[..n], expected);
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
    Telegram tg,
    PriceBook priceBook,
    LimiterSync limiter,
    ILogger<Worker> log) : BackgroundService
{
    // Telegram redelivers on failure, and it can redeliver an update we already
    // handled. Bounded so a long-running process cannot grow it without limit.
    private readonly HashSet<long> _seen = [];
    private readonly Queue<long> _seenOrder = new();

    // Destructive operations awaiting confirmation, keyed by an opaque token
    // that travels in the button's callback_data. In memory on purpose: a
    // restart drops them, which fails in the safe direction.
    private readonly ConcurrentDictionary<string, Pending> _pending = new();

    // Questions awaiting a typed answer, one per operator. See PendingInput.
    private readonly ConcurrentDictionary<long, PendingInput> _inputs = new();

    // Bounds how many updates are handled at once. Eight is far more than a
    // handful of operators will ever generate; the point is that the limit
    // exists, so a burst cannot open an unbounded number of Telegram calls.
    private readonly SemaphoreSlim _gate = new(8, 8);

    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        // Publish the command menu to Telegram. This is what makes commands
        // discoverable: they appear behind the "/" button with descriptions and
        // autocomplete, instead of the operator having to remember them or
        // scroll back to a /help message.
        LogRuntimeShape();
        await PublishCommandMenuAsync(ct);

        await foreach (var update in queue.Reader.ReadAllAsync(ct))
        {
            // Dedupe stays on this thread, before anything is handed off: _seen
            // and _seenOrder are a plain HashSet and Queue, and keeping the only
            // access to them single-threaded is cheaper and clearer than locking.
            if (!MarkSeen(update.UpdateId)) { log.LogInformation("ignored duplicate update {Id}", update.UpdateId); continue; }

            // Handle concurrently. Awaiting each update in turn meant a second
            // command sat behind the first one's Telegram round trips — visible
            // in the log as two updates arriving 114ms apart and the second
            // answering 800ms later. The gate keeps that bounded, and the
            // mutating path is already serialised by KeyStore's own lock, so
            // concurrency here cannot interleave two writes to consumers.conf.
            await _gate.WaitAsync(ct);
            _ = Task.Run(async () =>
            {
                try { await HandleAsync(update, ct); }
                catch (Exception ex) { log.LogError(ex, "handler failed for update {Id}", update.UpdateId); }
                finally { _gate.Release(); }
            }, ct);
        }
    }

    // What the GC actually decided, once, at startup.
    //
    // Worth logging rather than assuming: the runtime default is Workstation GC,
    // but the Web SDK flips it to Server, and since .NET 9 DATAS is on by
    // default and starts Server GC at a single heap — so the mode alone does not
    // tell you the heap count. TotalAvailableMemoryBytes is what the GC believes
    // it may use, which is the container limit when one is set and the whole
    // machine when it is not; on a shared inference box that distinction is the
    // difference between a bounded process and an unbounded one.
    private void LogRuntimeShape()
    {
        var info = GC.GetGCMemoryInfo();
        var vars = GC.GetConfigurationVariables();
        string V(string k) => vars.TryGetValue(k, out var v) ? v?.ToString() ?? "-" : "-";

        log.LogInformation(
            "gc: server={Server} concurrent={Concurrent} datas={Datas} conserve={Conserve} regionSize={RegionSize}",
            GCSettings.IsServerGC, V("gcConcurrent"), V("GCDynamicAdaptationMode"),
            V("GCConserveMemory"), V("GCRegionSize"));

        log.LogInformation(
            "gc: committed={CommittedMiB}MiB available={AvailableMiB}MiB pinned={Pinned} latency={Latency}",
            info.TotalCommittedBytes / (1024 * 1024),
            info.TotalAvailableMemoryBytes / (1024 * 1024),
            info.PinnedObjectsCount,
            GCSettings.LatencyMode);
    }

    private async Task HandleAsync(Update u, CancellationToken ct)
    {
        if (u.CallbackQuery is { } cb) { await HandleCallbackAsync(cb, ct); return; }

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

        var started = Stopwatch.GetTimestamp();
        var command = Head(text);

        // A question the bot asked is answered by the next plain text. Any
        // command abandons it, so a forgotten prompt can never swallow a later
        // /topup's arguments.
        Reply reply;
        if (text.StartsWith('/'))
        {
            _inputs.TryRemove(msg.From.Id, out _);
            reply = await DispatchWithTypingAsync(msg.Chat.Id, msg.From.Id, text.Trim(), ct);
        }
        else if (_inputs.TryRemove(msg.From.Id, out var input) && DateTimeOffset.UtcNow <= input.Expires)
        {
            command = $"(answer to {input.Kind})";
            reply = await AnswerInputAsync(msg.From.Id, input, text.Trim(), ct);
        }
        else
            reply = await DispatchWithTypingAsync(msg.Chat.Id, msg.From.Id, text.Trim(), ct);
        var worked = Stopwatch.GetElapsedTime(started);
        if (reply.Text is { Length: > 0 }) await SendAsync(msg.Chat.Id, reply, ct);

        // Three numbers, because they fail differently and the split IS the
        // diagnosis:
        //   queued  how long Telegram took to hand us the update after the
        //           operator pressed send. Not ours, and not fixable from in
        //           here — a large value means inbound delivery to this host is
        //           failing and Telegram is retrying with backoff.
        //   work    this stack and the services behind it. Ours to fix.
        //   total   work plus the round trip back out to Telegram.
        //
        // Without `queued`, a delivery that sat in Telegram's retry queue for
        // twenty minutes is indistinguishable from a fast bot, because every
        // clock inside this process starts when the update arrives.
        var queued = msg.Date > 0
            ? (int)(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - msg.Date)
            : -1;
        log.LogInformation("{Command} from {UserId} queued={QueuedS}s work={WorkMs}ms total={TotalMs}ms",
            command, msg.From.Id, queued,
            (int)worked.TotalMilliseconds,
            (int)Stopwatch.GetElapsedTime(started).TotalMilliseconds);
    }

    // A tapped button. Buttons are better than a typed CONFIRM for a
    // destructive action: one tap, nothing to mistype, and the prompt cannot be
    // answered by accident three messages later.
    private async Task HandleCallbackAsync(CallbackQuery cb, CancellationToken ct)
    {
        var chatId = cb.Message?.Chat?.Id;
        if (cb.From is null || chatId is null) return;
        if (!cfg.AllowedIds.Contains(cb.From.Id))
        {
            log.LogWarning("DENIED callback from user {UserId}", cb.From.Id);
            return;
        }

        var data = cb.Data ?? "";

        // The /key browser. These are READ-ONLY, so unlike the confirmations
        // below they carry no token, do not expire, and are not bound to the
        // operator who opened them: re-tapping a stale card just re-reads
        // Prometheus. Handled before the confirmation path because that path
        // treats every callback as `<3-char prefix><token>`.
        if (data.StartsWith("kc:", StringComparison.Ordinal)
            || data.StartsWith("kt:", StringComparison.Ordinal)
            || data.StartsWith("kr:", StringComparison.Ordinal)
            || data.StartsWith("kp:", StringComparison.Ordinal)
            || data == "kl:")
        {
            var reply = await KeyCallbackAsync(data, chatId.Value, ct);
            await AnswerCallbackAsync(cb.Id, ct);
            await SendAsync(chatId.Value, reply, ct);
            return;
        }

        // Settings screens. They EDIT the message they were tapped on, so a
        // run of changes is one screen rather than a scroll of stale copies.
        // Also not token-bound: each tap is one idempotent write of a policy
        // value (set X to 2M twice is set X to 2M), audited, and repeatable by
        // any allowlisted operator with /set anyway. Money is not in here —
        // balance buttons go through the token path below.
        if (data.Length > 3 && data[0] == 'p' && data[2] == ':')
        {
            var (reply, edit) = await PolicyCallbackAsync(data, cb.From.Id, ct);
            await AnswerCallbackAsync(cb.Id, ct);
            if (edit) await tg.EditOrSendAsync(chatId.Value, cb.Message?.MessageId ?? 0, reply, ct);
            else await SendAsync(chatId.Value, reply, ct);
            return;
        }

        var token = data.Length > 3 ? data[3..] : "";
        _pending.TryRemove(token, out var p);

        Reply result;
        if (p is null) result = new Reply("That button is no longer valid. Run the command again.");
        // The token is bound to the user who armed it, so one operator cannot
        // confirm another's pending destructive action from a shared screen.
        else if (p.UserId != cb.From.Id) result = new Reply("That confirmation belongs to someone else.");
        else if (DateTimeOffset.UtcNow > p.Expires) result = new Reply("That button expired. Nothing was changed.");
        else if (!data.StartsWith("ok:", StringComparison.Ordinal)) result = new Reply("Cancelled. Nothing was changed.");
        else result = await p.Run(ct);

        await AnswerCallbackAsync(cb.Id, ct);
        await SendAsync(chatId.Value, result, ct);
    }

    private async Task<Reply> DispatchAsync(long userId, string text, CancellationToken ct)
    {
        var parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cmd = parts[0].Split('@')[0].ToLowerInvariant();
        var a1 = parts.Length > 1 ? parts[1] : null;
        var a2 = parts.Length > 2 ? parts[2] : null;

        return cmd switch
        {
            "/start" or "/help" => new Reply(HelpText),
            "/status"     => new Reply(await StatusAsync(ct)),
            "/keys"       => new Reply(await KeysAsync(ct)),
            "/balance"    => new Reply(await BalanceAsync(a1, ct)),
            "/key"        => await KeyPickerAsync(ct),
            "/usage"      => new Reply(await UsageAsync(a1 ?? "24h", ct)),
            "/alerts"     => new Reply(await AlertsAsync(ct)),
            "/health"     => new Reply(await HealthAsync(ct)),
            "/top"        => new Reply(await TopAsync(a1 ?? "24h", ct)),
            "/p95"        => new Reply(await LatencyAsync(a1, ct)),
            "/errors"     => new Reply(await ErrorsAsync(a1 ?? "24h", ct)),
            "/tiers"      => new Reply(TiersHelp()),
            "/langfuse"   => new Reply(LangfuseHelp()),
            "/prices"     => new Reply(await PricesAsync(ct)),
            // No argument is the common case — an operator wants "show me
            // this consumer", not a request id they would have to go and find
            // first. Falling back to the picker beats a usage hint.
            "/trace"      => a1 is null ? await KeyPickerAsync(ct) : new Reply(TraceHelp(a1)),
            "/tier"       => (a1 is null || a2 is null)
                                 ? new Reply(Usage("/tier &lt;name&gt; &lt;trial|team|service|batch|admin&gt;", "/tier acme service"))
                                 : await TierAsync(a1, a2, ct),
            "/policy"     => a1 is null ? await KeyPickerAsync(ct) : await PolicyCardAsync(a1, null, ct),
            "/set"        => (a1 is null || a2 is null || parts.Length < 4)
                                 ? new Reply(Usage("/set &lt;name&gt; &lt;quota|refill|daily|tpm|max_tokens&gt; &lt;value|default&gt;",
                                                   "/set acme daily 2M"))
                                 : await SetPolicyAsync(a1, a2, parts[3], ct),
            "/opencode"   => new Reply(a1 is null ? Usage("/opencode &lt;name&gt;", "/opencode acme") : await OpenCodeAsync(a1, ct)),
            // No name: ask for one. Name only: tier buttons, one tap creates.
            // Name and a number: the original untiered form, kept for scripts
            // and muscle memory.
            "/newkey"     => a1 is null ? AskInput(userId, InputKind.NewKeyName, "", null,
                                              "Send the new key's name: 1–32 characters of lowercase letters, digits, <code>-</code> or <code>_</code>.")
                           : a2 is null ? await NewKeyPickerAsync(userId, a1.ToLowerInvariant(), ct)
                           : await NewKeyAsync(a1, null, a2, ct),
            "/topup"      => new Reply((a1 is null || a2 is null) ? Usage("/topup &lt;name&gt; &lt;tokens&gt;", "/topup acme 500000") : await TopUpAsync(a1, a2, ct)),
            "/setquota"   => (a1 is null || a2 is null) ? new Reply(Usage("/setquota &lt;name&gt; &lt;tokens&gt;", "/setquota acme 1000000")) : Arm(userId, a1, a2, PendingKind.SetQuota),
            "/clearquota" => a1 is null ? new Reply(Usage("/clearquota &lt;name&gt;", "/clearquota acme")) : Arm(userId, a1, "0", PendingKind.SetQuota),
            "/revoke"     => a1 is null ? new Reply(Usage("/revoke &lt;name&gt;", "/revoke acme")) : Arm(userId, a1, null, PendingKind.Revoke),
            // A command that is documented but lands here is a wiring bug,
            // not user error, and saying so beats "unknown command".
            _             => new Reply(IsKnownCommand(cmd)
                                 ? $"<code>{Esc(Head(cmd))}</code> is listed but not wired up. That is a bug in the bot, not in what you typed."
                                 : $"Unknown command <code>{Esc(Head(cmd))}</code>.\n\nSend /help to see what this bot can do.")
        };
    }

    // ---- command table ----------------------------------------------------
    //
    // ONE row per command. /help AND the Telegram menu are both rendered from
    // this, because they had already drifted: /langfuse shipped in the
    // dispatcher and in /help but never reached setMyCommands, so it worked
    // and was invisible in the client's command menu.
    //
    // Order here is MENU order — roughly how often an operator reaches for it,
    // since Telegram shows the list verbatim. /help regroups by Group without
    // reordering inside a group.
    //
    // Group "" hides a row from /help. /start is deliberately absent: it is
    // Telegram's implicit entry point and a pure alias of /help.
    //
    // NOT compiler-enforced: adding a `case` to DispatchAsync without a row
    // here leaves a command undocumented. The reverse — a row with no case —
    // IS caught, by the fallback branch of that switch.
    private sealed record Cmd(
        string Name, string Group, string Args, string Blurb, string MenuText);

    private static readonly string[] HelpGroups =
    [
        "Who and how much", "What they used", "Is it healthy",
        "Grant", "Destructive — these ask first"
    ];

    private static readonly Cmd[] Commands =
    [
        new("status",  "Is it healthy", "", "can I still operate the gateway",
            "Infrastructure health"),
        new("keys", "Who and how much", "", "consumers, balances and tiers",
            "Consumers and their balances"),
        new("balance", "Who and how much", "[name]",
            "balance, burn rate and runway", "Balance for one consumer or all"),
        new("key", "Who and how much", "",
            "pick a consumer from a list — its numbers, and where its traces are",
            "Per-key stats and traces"),
        new("usage", "What they used", "[1h|24h|7d|30d]",
            "tokens in/out per consumer", "Tokens and requests over a window"),
        new("health", "Is it healthy", "",
            "is the stack healthy, and can I believe it",
            "Stack and telemetry health"),
        new("alerts", "Is it healthy", "", "what is firing right now",
            "What is firing right now"),
        new("top", "What they used", "[1h|24h|7d]",
            "busiest consumers, with errors", "Busiest consumers over a window"),
        new("p95", "What they used", "[name]",
            "latency percentiles, per consumer (gateway + engine)",
            "Latency percentiles per consumer"),
        new("prices", "What they used", "",
            "reference prices for this model: OpenRouter and Alibaba Cloud",
            "Reference prices used for cost"),
        new("errors", "What they used", "[1h|24h|7d]",
            "status mix per consumer", "Status mix per consumer"),
        new("tiers", "Who and how much", "", "what each policy tier means",
            "What each policy tier means"),
        new("trace", "What they used", "&lt;request-id&gt;",
            "where to look one request up", "Where to look one request up"),
        new("langfuse", "What they used", "",
            "what Langfuse can and cannot tell you",
            "What Langfuse can and cannot show"),
        new("tier", "Who and how much", "&lt;name&gt; &lt;tier&gt;",
            "record a consumer's tier", "Record a consumer's policy tier"),
        new("policy", "Who and how much", "&lt;name&gt;",
            "a consumer's settings, and which come from its tier",
            "A consumer's settings and their source"),
        new("set", "Who and how much", "&lt;name&gt; &lt;setting&gt; &lt;value|default&gt;",
            "change one setting for one consumer",
            "Change one setting for one consumer"),
        new("newkey", "Grant", "[name]",
            "create a key: pick a tier with one tap, adjust anything after",
            "Create a key: one tap per tier"),
        new("opencode", "Grant", "&lt;name&gt;",
            "re-send an existing consumer's config",
            "Re-send a consumer's OpenCode config"),
        new("topup", "Grant", "&lt;name&gt; &lt;tokens&gt;", "add to a balance",
            "Add tokens to a consumer"),
        new("setquota", "Destructive — these ask first",
            "&lt;name&gt; &lt;tokens&gt;", "<i>replaces</i> a balance",
            "Overwrite a balance (asks to confirm)"),
        new("clearquota", "Destructive — these ask first", "&lt;name&gt;",
            "sets a balance to zero", "Set a balance to zero (asks to confirm)"),
        new("revoke", "Destructive — these ask first", "&lt;name&gt;",
            "deletes a key and its balance",
            "Delete a key and its balance (asks to confirm)"),
        new("help", "", "", "", "Show all commands")
    ];

    private static bool IsKnownCommand(string cmd) =>
        Commands.Any(c => cmd.Length == c.Name.Length + 1
                          && cmd.AsSpan(1).SequenceEqual(c.Name));

    // Built once. The text is constant, so rendering it per /help would be
    // pure waste on a command an operator hits repeatedly while learning.
    private static readonly string HelpText = BuildHelp();

    private static string BuildHelp()
    {
        var sb = new StringBuilder("<b>Gateway access management</b>\n");
        foreach (var group in HelpGroups)
        {
            sb.Append("\n<b>").Append(group).Append("</b>\n");
            foreach (var c in Commands)
            {
                if (c.Group != group) continue;
                sb.Append('/').Append(c.Name);
                if (c.Args.Length > 0) sb.Append(' ').Append(c.Args);
                sb.Append(" — ").Append(c.Blurb).Append('\n');
            }
        }
        sb.Append("\nCredentials are shown once, by /newkey. /keys lists names only.\n")
          .Append("Quota is a single TOTAL-token balance — input and output are charged\n")
          .Append("the same. See /tiers.");
        return sb.ToString();
    }

    // ---- reads ------------------------------------------------------------

    private async Task<string> StatusAsync(CancellationToken ct)
    {
        // Four independent probes, so run them at once rather than in turn. They
        // are all on-host and fast, but a status check exists to be read when
        // something is wrong — and when a component is wrong it usually hangs to
        // its timeout instead of failing. Serially that is four timeouts end to
        // end; concurrently the slowest one sets the bound.
        var gwTask = TryAsync(async () =>
        {
            using var r = await http.CreateClient("gateway").GetAsync("v1/models", ct);
            return r.IsSuccessStatusCode ? "ok" : $"HTTP {(int)r.StatusCode}";
        });
        var ledTask = TryAsync(async () => $"{(await ledger.ListAsync(ct)).Count} consumers");
        var promTask = TryAsync(async () =>
        {
            using var r = await http.CreateClient("prometheus").GetAsync("-/healthy", ct);
            return r.IsSuccessStatusCode ? "ok" : $"HTTP {(int)r.StatusCode}";
        });
        var apiTask = TryAsync(async () => $"{(await keys.ReadConsumersAsync(ct)).Count} keys");

        await Task.WhenAll(gwTask, ledTask, promTask, apiTask);
        var (gw, led, prom, api) = (gwTask.Result, ledTask.Result, promTask.Result, apiTask.Result);

        string[] rows =
        [
            $"{"gateway",-12}{gw}",
            $"{"ledger",-12}{led}",
            $"{"prometheus",-12}{prom}",
            $"{"key-auth",-12}{api}",
            $"{"limiter",-12}{limiter.Summary()}"
        ];
        var body = Table("<b>Infrastructure</b>", rows);

        // A failed ledger is not a degraded feature, it is an outage on the
        // billable routes: ai-quota has no fail-open, so every chat request 403s.
        if (led.StartsWith("FAILED", StringComparison.Ordinal))
            body += "\n\u26a0\ufe0f <b>Ledger unreachable</b> \u2014 ai-quota has no fail-open, so billable routes are returning 403 right now.";
        return body;
    }

    // Deliberately NOT the same thing as /status.
    //
    // /status answers "can I still operate the gateway" by probing the four
    // things this bot talks to. /health answers "is the stack healthy, and can
    // I believe what it is telling me" — which includes the telemetry itself.
    // Both matter: for fourteen hours in September 2026 the second was false
    // while the first was true, and nothing said so.
    private async Task<string> HealthAsync(CancellationToken ct)
    {
        // One round trip each, all at once. A health check is read when
        // something is wrong, and a wedged component hangs to its timeout
        // rather than failing fast — serially that is nine timeouts.
        var upT     = PromScalarAsync("count(up == 1)", ct);
        var totalT  = PromScalarAsync("count(up)", ct);
        var alertsT = PromScalarAsync("count(ALERTS{alertstate=\"firing\"}) or vector(0)", ct);
        var ledgerT = PromScalarAsync("max(redis_up)", ct);
        var queueT  = PromScalarAsync("max(otelcol_exporter_queue_size)", ct);
        var capT    = PromScalarAsync("max(otelcol_exporter_queue_capacity)", ct);
        var failT   = PromScalarAsync("sum(rate(otelcol_exporter_send_failed_spans[5m])) or vector(0)", ct);
        var genT    = PromScalarAsync("sum(sglang:gen_throughput)", ct);
        var ttftT   = PromScalarAsync(
            "histogram_quantile(0.95, sum(rate(sglang:time_to_first_token_seconds_bucket[5m])) by (le))", ct);
        var kvT     = PromScalarAsync("max((sglang:kv_used_tokens / sglang:kv_available_tokens)) * 100", ct);
        // Prefix-cache hit rate. Worth a row since 2026-09-05: the router moved
        // to cache_aware and this is where that shows up or fails to. Gauge is
        // since engine start, so it reads 0 for a while after a roll — that is
        // normal and deliberately NOT warned on, or every roll would cry wolf.
        var cacheT  = PromScalarAsync("avg(sglang:cache_hit_rate) * 100", ct);

        await Task.WhenAll(upT, totalT, alertsT, ledgerT, queueT, capT, failT, genT, ttftT, kvT, cacheT);

        static string N(double? v, string fmt = "N0") =>
            v is null ? "\u2014" : ((double)v).ToString(fmt, CultureInfo.InvariantCulture);

        var up = upT.Result; var total = totalT.Result;
        var alerts = alertsT.Result; var ledger = ledgerT.Result;
        var queue = queueT.Result; var cap = capT.Result;

        string[] rows =
        [
            $"{"targets",-12}{N(up)}/{N(total)} up",
            $"{"alerts",-12}{N(alerts)} firing",
            $"{"ledger",-12}{(ledger is null ? "\u2014" : ledger > 0 ? "UP" : "DOWN")}",
            $"{"spans",-12}queue {N(queue)}/{N(cap)}, {N(failT.Result, "N2")} failed/s",
            $"{"throughput",-12}{N(genT.Result)} tok/s",
            $"{"TTFT p95",-12}{N(ttftT.Result, "N2")} s",
            $"{"KV pool",-12}{N(kvT.Result, "N1")} %",
            $"{"cache hit",-12}{N(cacheT.Result, "N1")} % (since engine start)"
        ];

        var body = Table("<b>Stack health</b>", rows);

        // Lead with the things that are silently wrong. Each of these has been
        // true on this node while every other signal looked fine.
        var warn = new List<string>();
        if (up is not null && total is not null && up < total)
            warn.Add($"{N(total - up)} scrape target(s) down — that plane is blind, not quiet.");
        if (ledger is not null && ledger == 0)
            warn.Add("Ledger unreachable — ai-quota fails closed, so billable routes are 403ing now.");
        if (queue is not null && cap is > 0 && queue >= cap)
            warn.Add("Trace export queue is FULL — spans are being dropped.");
        if (failT.Result is > 0)
            warn.Add("Spans are failing to export — Langfuse is not receiving traces.");
        if (alerts is > 0)
            warn.Add($"{N(alerts)} alert(s) firing — see /alerts.");

        if (warn.Count > 0)
            body += "\n" + string.Join("\n", warn.Select(w => "\u26a0\ufe0f " + w));
        else
            body += "\n\u2705 <i>Nothing firing, every target reporting.</i>";

        return body;
    }

    private async Task<string> KeysAsync(CancellationToken ct)
    {
        var balancesT = ledger.ListAsync(ct);
        var tiersT = ledger.TiersAsync(ct);
        var consumersT = keys.ReadConsumersAsync(ct);
        var overriddenT = ledger.OverriddenAsync(ct);
        await Task.WhenAll(balancesT, tiersT, consumersT, overriddenT);
        var balances = balancesT.Result; var tiers = tiersT.Result; var consumers = consumersT.Result;
        var overridden = overriddenT.Result;

        if (consumers.Count == 0)
            return "No consumers yet.\n\nCreate one with <code>/newkey &lt;name&gt;</code>.";

        var header = $"{"consumer",-16}{"balance",14}  tier";
        var rows = consumers.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(name =>
        {
            var bal = balances.TryGetValue(name, out var b)
                ? b.ToString("N0", CultureInfo.InvariantCulture)
                : "not seeded";
            // "-" rather than a guessed default: an unassigned consumer is a
            // real state and should look like one.
            var tier = tiers.GetValueOrDefault(name, "\u2014");
            // A star rather than a column: most consumers follow their tier,
            // and the ones that do not are the ones worth a second look.
            if (overridden.Contains(name)) tier += "*";
            return $"{name,-16}{bal,14}  {tier}";
        });

        var untiered = consumers.Keys.Count(n => !tiers.ContainsKey(n));
        var body = Table($"<b>Consumers</b> ({consumers.Count})", new[] { header }.Concat(rows))
                 + "\nCredentials are not shown. Use /opencode &lt;name&gt;.";
        if (consumers.Keys.Any(overridden.Contains))
            body += "\n<i>* has settings changed from its tier \u2014 /policy &lt;name&gt;.</i>";
        if (untiered > 0)
            body += $"\n<i>{untiered} without a tier \u2014 set with /tier &lt;name&gt; &lt;tier&gt;.</i>";
        return body;
    }

    private async Task<string> BalanceAsync(string? name, CancellationToken ct)
    {
        if (name is not null)
        {
            var q = await QuotaGetAsync(name, ct);
            // Three different causes, one 403 from ai-quota. Say so rather than
            // asserting one of them.
            if (q is null)
                return $"<b>{Esc(name)}</b> has no balance recorded.\n\nEither it was never seeded, or the ledger is unreachable. "
                     + $"Seed it with <code>/topup {Esc(name)} 1000000</code>.";

            // Burn and runway come from the recording rules, so the arithmetic
            // is identical to the dashboard and the ConsumerQuotaLow alert
            // rather than a third implementation that can disagree with them.
            var sel = $"{{ai_consumer=\"{name}\"}}";
            var burnT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:tokens24h{sel})", ct);
            var daysT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_days_left{sel})", ct);
            var binT  = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:input24h{sel})", ct);
            var boutT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:output24h{sel})", ct);
            await Task.WhenAll(burnT, daysT, binT, boutT);

            var body = $"<b>{Esc(name)}</b>\n<code>{q.Value:N0}</code> tokens remaining";

            if (burnT.Result is > 0 && daysT.Result is { } days)
            {
                var bi = binT.Result ?? 0; var bo = boutT.Result ?? 0;
                string[] rows =
                [
                    $"{"burn 24h",-12}{burnT.Result:N0} tok/day",
                    $"{"  input",-12}{bi,12:N0}",
                    $"{"  output",-12}{bo,12:N0}",
                    $"{"  i:o",-12}{(bo > 0 ? bi / bo : 0),12:N1}",
                    $"{"days left",-12}{days:N1}"
                ];
                body += "\n" + Table("", rows);
                if (days < 1)
                    body += "\n\u26a0\ufe0f <b>Under a day left</b> at this rate. <code>/topup " + Esc(name) + " ...</code>";
            }
            else
            {
                // Distinguish "idle" from "no data": a consumer who sent
                // nothing in 24h has no runway problem, and saying "0 days" or
                // showing nothing would both be misread.
                body += "\n<i>No usage in the last 24h, so there is no burn rate to project.</i>";
            }
            return body;
        }
        var all = await ledger.ListAsync(ct);
        if (all.Count == 0) return "No balances recorded yet.";

        // Balances come from Redis directly — the billing record — and the
        // runway column from Prometheus. If Prometheus is unreachable the
        // balances still render, because they are the half that matters.
        var runway = await PromAsync("sum by (ai_consumer) (consumer:quota_days_left)", ct);

        var balanceRows = all.OrderBy(x => x.Key, StringComparer.Ordinal)
                             .Select(x => $"{x.Key,-16}{x.Value,14:N0}"
                                        + (runway.TryGetValue(x.Key, out var d) && d < 3650
                                            ? $"{d,10:N1} d" : "         \u2014"));
        return Table("<b>Balances</b>", balanceRows)
             + "\n<i>Runway at the last 24h burn rate. \u2014 means idle.</i>";
    }

    private async Task<string> UsageAsync(string window, CancellationToken ct)
    {
        if (window is not ("24h" or "7d" or "1h" or "30d"))
            return $"Unknown window <code>{Esc(window)}</code>.\n\nUse one of <code>1h</code>, <code>24h</code>, <code>7d</code>, <code>30d</code>.";

        // Split by direction. The ledger charges input and output identically —
        // ai-quota deducts input+output 1:1 and has no weighting option — but a
        // single total hides that a consumer at 40:1 is paying almost entirely
        // for context it re-sent, most of which the radix cache served free.
        var inT   = PromAsync($"sum by (ai_consumer) (increase(route_upstream_model_consumer_metric_input_token[{window}]))", ct);
        var outT  = PromAsync($"sum by (ai_consumer) (increase(route_upstream_model_consumer_metric_output_token[{window}]))", ct);
        var reqsT = PromAsync($"sum by (ai_consumer) (increase(route_upstream_model_consumer_metric_llm_duration_count[{window}]))", ct);
        // Engine-side hit share per consumer, for the cache-aware price. Note
        // the label: the engine says `consumer`, ai-statistics `ai_consumer`.
        var hitT  = PromAsync($"1 - sum by (consumer) (increase(sglang:uncached_prompt_tokens_histogram_sum[{window}])) "
                            + $"/ clamp_min(sum by (consumer) (increase(sglang:prompt_tokens_histogram_sum[{window}])), 1)", ct, "consumer");
        var pricesT = priceBook.GetAsync(ct);
        await Task.WhenAll(inT, outT, reqsT, hitT, pricesT);

        var inp = inT.Result; var outp = outT.Result; var reqs = reqsT.Result;
        var hits = hitT.Result; var prices = pricesT.Result;
        if (inp.Count == 0 && outp.Count == 0) return $"No usage in the last {window}.";

        var names = inp.Keys.Union(outp.Keys).Union(reqs.Keys).ToList();
        var header = $"{"consumer",-14}{"in",9}{"out",8}{"i:o",6}{"req",6}";
        var rows = names
            .OrderByDescending(n => inp.GetValueOrDefault(n) + outp.GetValueOrDefault(n))
            .Select(n =>
            {
                var i = inp.GetValueOrDefault(n);
                var o = outp.GetValueOrDefault(n);
                return $"{n,-14}{i,9:N0}{o,8:N0}{(o > 0 ? i / o : 0),6:N1}{reqs.GetValueOrDefault(n),6:N0}";
            });

        // Reference cost per consumer. Short money so four columns fit a phone.
        static string M(decimal v) => v switch
        {
            0m => "0",
            < 10m => v.ToString("0.00", CultureInfo.InvariantCulture),
            < 1000m => v.ToString("0", CultureInfo.InvariantCulture),
            _ => (v / 1000m).ToString("0.#", CultureInfo.InvariantCulture) + "K",
        };
        decimal orT = 0, orcT = 0, sgT = 0, bjT = 0;
        var costHeader = $"{"consumer",-14}{"OR",7}{"OR-c",7}{"SG",7}{"BJ",7}";
        var costRows = names
            .OrderByDescending(n => inp.GetValueOrDefault(n) + outp.GetValueOrDefault(n))
            .Select(n =>
            {
                var i = inp.GetValueOrDefault(n); var o = outp.GetValueOrDefault(n);
                double? h = hits.TryGetValue(n, out var hv) ? hv : null;
                var list = PriceBook.Cost(prices.OpenRouter, i, o);
                var cached = PriceBook.Cost(prices.OpenRouter, i, o, h);
                var sg = PriceBook.Cost(prices.AlibabaSg, i, o);
                var bj = PriceBook.Cost(prices.AlibabaBj, i, o);
                orT += list; orcT += cached; sgT += sg; bjT += bj;
                return $"{n,-14}{M(list),7}{M(cached),7}{M(sg),7}{M(bj),7}";
            }).ToList();

        var totalIn = inp.Values.Sum(); var totalOut = outp.Values.Sum();
        return Table($"<b>Usage</b> \u2014 last {window}", new[] { header }.Concat(rows))
             + $"\n<b>{totalIn + totalOut:N0}</b> tokens charged \u2014 {totalIn:N0} in, {totalOut:N0} out."
             + "\n\n" + Table("<b>At reference prices</b>, USD", new[] { costHeader }.Concat(costRows))
             + $"\nTotal: OpenRouter <b>{PriceBook.Usd(orT)}</b> (cache-aware {PriceBook.Usd(orcT)}), "
             + $"Alibaba Singapore <b>{PriceBook.Usd(sgT)}</b>, Beijing <b>{PriceBook.Usd(bjT)}</b>."
             + "\n<i>OR = OpenRouter list, OR-c = with this key's cache hits priced as cached, SG/BJ = Alibaba Cloud.</i>"
             + "\n" + PriceFootnote(prices)
             + "\n<i>Quota is a single TOTAL-token balance: input and output are deducted at the same "
             + "rate. i:o shows how much of a bill is context re-sent rather than tokens generated.</i>"
             + "\n<i>Counters reset when the gateway restarts. Balances are the billing record.</i>";
    }

    // ---- key lifecycle ----------------------------------------------------

    private static string NameProblem(string name) =>
        $"<code>{Esc(Head(name))}</code> is not a valid name.\n\nUse 1\u201332 characters: lowercase letters, digits, <code>-</code> or <code>_</code>.";

    private static string AlreadyExists(string name) =>
        $"<b>{Esc(name)}</b> already exists.\n\nUse <code>/opencode {Esc(name)}</code> to re-send its config, or <code>/revoke {Esc(name)}</code> to replace it.";

    // Step one of the one-tap flow: a button per tier, each carrying a
    // single-use token. The name is checked now so a bad one fails before the
    // operator picks anything, and checked again at creation because ten
    // minutes is long enough for someone else to take it.
    private async Task<Reply> NewKeyPickerAsync(long userId, string name, CancellationToken ct)
    {
        if (!IsValidName(name)) return new Reply(NameProblem(name));
        if ((await keys.ReadConsumersAsync(ct)).ContainsKey(name)) return new Reply(AlreadyExists(name));

        var rows = new List<InlineKeyboardButton[]>();
        var text = new StringBuilder($"<b>Create {Esc(name)}</b> \u2014 tap a tier. The key is created at once "
                                   + "with that tier's defaults; every value can be changed afterwards.\n");
        foreach (var t in Policy.All.Where(t => t.Name != "admin"))
        {
            var tier = t.Name;
            var token = Tokenize(userId, TimeSpan.FromMinutes(10), c => NewKeyAsync(name, tier, null, c));
            rows.Add([new InlineKeyboardButton(
                $"{tier} \u00b7 {Policy.Compact(t.Quota)} \u00b7 {Policy.RefillName(t.Refill)}", "ok:" + token)]);
            text.Append($"\n<b>{tier}</b> \u2014 {Esc(t.For)}: {Policy.Compact(t.Quota)} tokens, "
                      + $"{Policy.RefillName(t.Refill)} refill, {(t.Daily == 0 ? "no" : Policy.Compact(t.Daily))} per day");
        }
        var untiered = Tokenize(userId, TimeSpan.FromMinutes(10), c => NewKeyAsync(name, null, "1000000", c));
        rows.Add([new InlineKeyboardButton("no tier \u00b7 1M", "ok:" + untiered),
                  new InlineKeyboardButton("Cancel", "no:" + untiered)]);
        text.Append("\n\n<i>Balance, daily and per-minute limits and refill are enforced; max_tokens is recorded. See /tiers.</i>");
        return new Reply(text.ToString(), new InlineKeyboardMarkup(rows.ToArray()));
    }

    // tier null = untiered with quotaArg (default 1,000,000); tier set = the
    // tier's quota and the tier recorded in the same step.
    private async Task<Reply> NewKeyAsync(string name, string? tier, string? quotaArg, CancellationToken ct)
    {
        if (!IsValidName(name)) return new Reply(NameProblem(name));
        var existing = await keys.ReadConsumersAsync(ct);
        if (existing.ContainsKey(name)) return new Reply(AlreadyExists(name));

        var quota = 1_000_000L;
        if (tier is not null && Policy.Tiers.TryGetValue(tier, out var def)) quota = def.Quota;
        else if (quotaArg is not null && !TryParseTokens(quotaArg, out quota))
            return new Reply($"<code>{Esc(quotaArg)}</code> is not a token count.\n\nGive a whole number, like <code>1000000</code>.");

        var credential = "Bearer sk-" + Base62(32);
        await keys.AddAsync(name, credential, ct);

        // Seed BEFORE announcing success. ai-quota returns the same 403 for
        // "never seeded" as for "exhausted", so an unseeded key looks broken in
        // a way that wastes an afternoon.
        await QuotaSetAsync(name, quota, ct);
        if (tier is not null) await ledger.SetTierAsync(name, tier, ct);
        await AuditAsync($"newkey name={name} quota={quota} tier={tier ?? "-"}", ct);

        // This message holds the credential, so its buttons open NEW messages
        // (kp:, kc:) and never edit this one \u2014 an edited-away credential is
        // gone for good. Warning BEFORE the block, so it is not below the fold
        // on a phone.
        var keyboard = new InlineKeyboardMarkup([
            [new InlineKeyboardButton("Settings", $"kp:{name}"),
             new InlineKeyboardButton("Key card", $"kc:24h:{name}")]
        ]);
        return new Reply(
            $"Created <b>{Esc(name)}</b>{(tier is null ? "" : $" on <b>{Esc(tier)}</b>")} with <code>{quota:N0}</code> tokens.\n\n"
          + "\u26a0\ufe0f <b>This credential is shown once.</b> Tap the block to copy it.\n\n"
          + $"<pre>{Esc(OpenCodeJson(credential))}</pre>\n"
          + $"Save as <code>~/.config/opencode/opencode.json</code>.\n\n"
          + "<i>Settings changes any value for this key.</i>",
            keyboard);
    }

    // A single-use button: the token is removed on the first tap, so a double
    // tap cannot run the action twice. Expired tokens are swept here because
    // nothing else ever would \u2014 an unused button would otherwise stay in the
    // dictionary for the life of the process.
    private string Tokenize(long userId, TimeSpan ttl, Func<CancellationToken, Task<Reply>> run)
    {
        var now = DateTimeOffset.UtcNow;
        foreach (var (k, v) in _pending)
            if (v.Expires < now) _pending.TryRemove(k, out _);
        var token = Base62(16);
        _pending[token] = new Pending(userId, now + ttl, run);
        return token;
    }

    private Reply AskInput(long userId, InputKind kind, string name, string? field, string prompt)
    {
        _inputs[userId] = new PendingInput(kind, name, field, DateTimeOffset.UtcNow.AddMinutes(5));
        return new Reply(prompt + "\n\n<i>Waiting 5 minutes. Any command cancels.</i>");
    }

    private async Task<Reply> AnswerInputAsync(long userId, PendingInput input, string text, CancellationToken ct) =>
        input.Kind switch
        {
            InputKind.NewKeyName => await NewKeyPickerAsync(userId, text.ToLowerInvariant(), ct),
            InputKind.Field      => await SetPolicyAsync(input.Name, input.Field ?? "", text, ct),
            InputKind.TopUp      => new Reply(await TopUpAsync(input.Name, text, ct)),
            _                    => new Reply("Nothing was waiting for that.")
        };

    private async Task<string> OpenCodeAsync(string name, CancellationToken ct)
    {
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.TryGetValue(name, out var credential))
            return $"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.";
        await AuditAsync($"opencode name={name}", ct);
        return $"<b>{Esc(name)}</b> \u2014 OpenCode config\n\n<pre>{Esc(OpenCodeJson(credential))}</pre>\n"
             + $"Save as <code>~/.config/opencode/opencode.json</code>.";
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

    private Reply Arm(long userId, string name, string? amount, PendingKind kind)
    {
        var expires = DateTimeOffset.UtcNow.AddSeconds(120);
        var token = Base62(16);
        string prompt;

        switch (kind)
        {
            case PendingKind.SetQuota:
                if (amount is null || !TryParseTokens(amount, out var target))
                    return new Reply("Amount must be a whole number of tokens, like <code>1000000</code>.");
                _pending[token] = new Pending(userId, expires, async ct =>
                {
                    var before = await QuotaGetAsync(name, ct);
                    await QuotaSetAsync(name, target, ct);
                    await AuditAsync($"setquota name={name} from={before?.ToString(CultureInfo.InvariantCulture) ?? "none"} to={target}", ct);
                    return new Reply($"<b>{Esc(name)}</b> balance set to <code>{target:N0}</code> (was {before?.ToString("N0", CultureInfo.InvariantCulture) ?? "unset"}).");
                });
                // Say what it REPLACES, not just what it sets. The whole reason
                // this needs confirming is that people reach for it expecting
                // /topup's additive behaviour.
                prompt = $"<b>Overwrite {Esc(name)}\u2019s balance?</b>\n\n"
                       + $"This <b>replaces</b> the balance with <code>{target:N0}</code> tokens. It does not add to it.\n"
                       + $"Use /topup to add.";
                break;

            case PendingKind.Revoke:
                _pending[token] = new Pending(userId, expires, async ct =>
                {
                    if (!await keys.RemoveAsync(name, ct)) return new Reply($"No consumer named <b>{Esc(name)}</b>.");
                    await ledger.DeleteAsync(name, ct);
                    await AuditAsync($"revoke name={name}", ct);
                    return new Reply($"Revoked <b>{Esc(name)}</b>. The key no longer authenticates and the balance is gone.");
                });
                prompt = $"<b>Revoke {Esc(name)}?</b>\n\n"
                       + "Their key stops working immediately and their balance is deleted. "
                       + "This cannot be undone \u2014 a new key would be a different credential.";
                break;

            default:
                return new Reply("Unsupported operation.");
        }

        var keyboard = new InlineKeyboardMarkup([[
            new InlineKeyboardButton(kind == PendingKind.Revoke ? "Revoke" : "Overwrite", "ok:" + token),
            new InlineKeyboardButton("Cancel", "no:" + token)
        ]]);
        return new Reply(prompt, keyboard);
    }

    private async Task<string> TopUpAsync(string name, string amount, CancellationToken ct)
    {
        if (!TryParseTokens(amount, out var delta))
            return $"<code>{Esc(amount)}</code> is not a token count.\n\nGive a whole number, like <code>500000</code>.";
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.ContainsKey(name))
            return $"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.";

        var body = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", name),
            new KeyValuePair<string, string>("value", delta.ToString(CultureInfo.InvariantCulture))
        ]);
        using var r = await http.CreateClient("gateway").PostAsync("v1/chat/completions/quota/delta", body, ct);
        if (!r.IsSuccessStatusCode)
            return $"Top-up failed \u2014 the gateway returned HTTP {(int)r.StatusCode}.\n\nRun /status to check the ledger.";

        var now = await QuotaGetAsync(name, ct);
        await AuditAsync($"topup name={name} delta={delta}", ct);
        // Echo the resulting balance, not just the delta: after an overdraft the
        // consumer can still be negative and "+500,000" alone reads as fixed.
        return $"<b>{Esc(name)}</b>  +{delta:N0}\n\nBalance now <code>{now?.ToString("N0", CultureInfo.InvariantCulture) ?? "unknown"}</code>.";
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

    private Task QuotaSetAsync(string name, long value, CancellationToken ct) =>
        QuotaApi.SetAsync(http, name, value, ct);

    // ---- Stage B reads: everything below is answered by the access log -------
    //
    // These three exist because ai-statistics cannot answer them. Its seven
    // counters carry no status code and no latency histogram, so until the
    // access log became a fact table there was no per-consumer error mix and no
    // per-consumer percentile anywhere in the stack.
    //
    // They read the Vector-derived aggregates in Prometheus rather than
    // ClickHouse directly: this bot is on `edge` and the trace store is
    // backend-only, and putting an internet-reachable bot on the backend would
    // give it a route to the worker ports. ClickHouse stays the durable record
    // behind Grafana; these are the operator's glance.

    // Deliberately a signpost, not a lookup.
    //
    // The per-request record lives in ClickHouse and the spans live in
    // Langfuse, and this bot can reach neither: it runs on `edge`, both stores
    // are backend-only, and putting an internet-reachable bot on the backend
    // would give it a route to the worker ports. Rather than half-answer from
    // the aggregates — which cannot resolve a single request at all — this
    // hands over the two exact places the answer is, and the query to run.
    private string TraceHelp(string requestId)
    {
        // Cheap sanity check. A mistyped id produces an empty result in both
        // stores, which reads like "the request did not happen" rather than
        // "you typed it wrong".
        var looksLikeId = requestId.Length is >= 8 and <= 64
            && requestId.All(c => char.IsAsciiLetterOrDigit(c) || c == '-');
        if (!looksLikeId)
            return $"<code>{Esc(Head(requestId))}</code> does not look like a request id.\n\n"
                 + "They are the x-request-id Envoy mints per request — a UUID, and the same value "
                 + "appears on the gateway span and in the fact table.";

        var id = Esc(requestId);
        // The join is TWO HOPS and was verified end-to-end on 2026-09-05
        // (Stage D). The engine does NOT carry the gateway's request id — it
        // mints its own 32-hex rid and ignores caller-supplied ones — so the
        // second hop goes through the ROUTER's span, which does record the
        // gateway id and whose trace the engine spans share.
        return $"<b>Request</b> <code>{id}</code>\n\n"
             + "<b>1. What happened</b> — the fact table, ClickHouse on the host:\n"
             + $"<pre>SELECT * FROM gateway.requests FINAL\nWHERE request_id = '{id}';</pre>\n"
             + "Consumer, tokens, status and latency. This is the billing-grade record.\n\n"
             + "<b>2. Inside the engine</b> — you need the trace id, not this id:\n"
             + $"<pre>SELECT trace_id FROM events_core\nWHERE service_name = 'smg'\n  AND metadata_values[indexOf(\n        metadata_names,'attributes.request_id')] = '{id}'\nLIMIT 1;</pre>\n"
             + $"Paste that trace id into {Esc(cfg.LangfuseUrl)} for the engine waterfall — "
             + "prefill_waiting, prefill_forward, decode_forward, and a <code>Req</code> span "
             + "with the token counts and TTFT.\n\n"
             + "<i>Why two hops: the router does not honour the gateway's inbound traceparent, so "
             + "it starts a fresh trace. It does copy this request id onto its own span, and the "
             + "engine spans share the router's trace — so id gets you to the router, and the "
             + "router's trace gets you to the engine. Searching Langfuse for this id directly "
             + "finds only the gateway span.</i>";
    }


    // ---- /key: pick a consumer, then read it --------------------------------
    //
    // The point of this command is that it takes NO arguments. Every other
    // per-consumer command needs a name typed correctly, and /trace needed a
    // request id the operator had to go and find first. Here the list is the
    // interface: tap a name, get its numbers, tap again for where its traces
    // live or for a written report.
    //
    // Everything below reads Prometheus. That is a deliberate limit, not an
    // oversight: this bot runs on `edge`, and the per-request fact table and
    // the span store are backend-only, because an internet-reachable bot with a
    // route to the worker ports is a worse trade than an operator pasting one
    // SQL query. So "statistics" is answered here in full, and "the trace of
    // one request" is answered with a link and a query.

    // Names reach PromQL as string literals and reach callback_data as a
    // suffix, so they are constrained at the door rather than escaped later.
    // The set here is what /newkey can produce.
    private static bool SafeName(string n) =>
        n.Length is > 0 and <= 40
        && n.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');

    private async Task<Reply> KeyPickerAsync(CancellationToken ct)
    {
        var consumers = await keys.ReadConsumersAsync(ct);
        var names = consumers.Keys.Where(SafeName)
                             .OrderBy(k => k, StringComparer.Ordinal).ToArray();
        if (names.Length == 0)
            return new Reply("No consumers yet.\n\nCreate one with <code>/newkey &lt;name&gt;</code>.");

        var rows = new List<InlineKeyboardButton[]>();
        for (var i = 0; i < names.Length; i += 2)
        {
            rows.Add(i + 1 < names.Length
                ? [Pick(names[i]), Pick(names[i + 1])]
                : [Pick(names[i])]);
        }
        return new Reply(
            $"<b>Which consumer?</b>  ({names.Length})\n\n"
          + "<i>Numbers come from Prometheus. Balances are the ledger, which is "
          + "what bills.</i>",
            new InlineKeyboardMarkup(rows.ToArray()));

        static InlineKeyboardButton Pick(string n) => new(n, "kc:24h:" + n);
    }

    private async Task<Reply> KeyCallbackAsync(string data, long chatId, CancellationToken ct)
    {
        if (data == "kl:") return await KeyPickerAsync(ct);

        var kind = data[..3];
        var rest = data[3..];
        var window = "24h";

        if (kind == "kc:")
        {
            var i = rest.IndexOf(':', StringComparison.Ordinal);
            if (i < 0) return new Reply("Malformed selection. Run /key again.");
            window = rest[..i];
            rest = rest[(i + 1)..];
            if (!ValidWindow(window)) return new Reply(BadWindow(window));
        }

        if (!SafeName(rest))
            return new Reply("That is not a consumer name this bot recognises.\n\nRun /key again.");

        return kind switch
        {
            "kc:" => await KeyCardAsync(rest, window, ct),
            "kt:" => KeyTraceCard(rest),
            "kr:" => KeyReportStart(rest, chatId, ct),
            "kp:" => await PolicyCardAsync(rest, null, ct),
            _     => new Reply("Unknown selection. Run /key again.")
        };
    }

    // Everything one screen can honestly say about a consumer.
    private readonly record struct KeyStats(
        double? Balance, double? Runway, double? Requests, double? NotOk,
        double? TokensIn, double? TokensOut, double? GatewayP95,
        double? Ttft, double? Itl, double? E2e, double? CacheHit,
        double? Unbilled, double? UnbilledSeconds,
        Dictionary<string, double> ByReplica);

    private async Task<KeyStats> KeyStatsAsync(string name, string w, CancellationToken ct)
    {
        // SafeName has already guaranteed there is no quote in here.
        var sel = $"{{consumer=\"{name}\"}}";
        var led = $"{{ai_consumer=\"{name}\"}}";

        var balT  = PromScalarAsync($"consumer:quota_balance:tokens{led}", ct);
        var runT  = PromScalarAsync($"consumer:quota_days_left{led}", ct);
        var reqT  = PromScalarAsync($"sum(increase(gateway_requests_total{sel}[{w}])) or vector(0)", ct);
        var badT  = PromScalarAsync($"sum(increase(gateway_requests_total{{consumer=\"{name}\",status_class!=\"2xx\"}}[{w}])) or vector(0)", ct);
        // Tokens from ai-statistics (Envoy), not Vector's gateway_tokens_total.
        // Both are per direction, but Vector expires an idle series after ten
        // minutes and Prometheus then loses the first request of every burst:
        // measured 986 vs 1,520 input tokens for a sparse consumer. Envoy's
        // series lives for the gateway's lifetime. These are also what cost is
        // computed from, and what /usage reads, so the numbers agree.
        var inT   = PromScalarAsync($"sum(increase(route_upstream_model_consumer_metric_input_token{led}[{w}])) or vector(0)", ct);
        var outT  = PromScalarAsync($"sum(increase(route_upstream_model_consumer_metric_output_token{led}[{w}])) or vector(0)", ct);
        var gwT   = PromScalarAsync($"histogram_quantile(0.95, sum by (le) (rate(gateway_request_duration_seconds_bucket{sel}[{w}])))", ct);
        // Engine-side. These exist only since the consumer label was put on the
        // tokenizer metrics; before that the gateway's view was all there was.
        var ttfT  = PromScalarAsync($"histogram_quantile(0.95, sum by (le) (rate(sglang:time_to_first_token_seconds_bucket{sel}[{w}])))", ct);
        var itlT  = PromScalarAsync($"histogram_quantile(0.95, sum by (le) (rate(sglang:inter_token_latency_seconds_bucket{sel}[{w}])))", ct);
        var e2eT  = PromScalarAsync($"histogram_quantile(0.95, sum by (le) (rate(sglang:e2e_request_latency_seconds_bucket{sel}[{w}])))", ct);
        var cchT  = PromScalarAsync($"1 - sum(rate(sglang:uncached_prompt_tokens_histogram_sum{sel}[{w}])) / clamp_min(sum(rate(sglang:prompt_tokens_histogram_sum{sel}[{w}])), 1)", ct);
        var repT  = PromAsync($"sum by (instance) (increase(sglang:generation_tokens_total{sel}[{w}]))", ct, "instance");
        // Requests cut off before their final usage frame, which ai-quota
        // therefore charged nothing (measured 2026-09-13, see vector.yaml).
        // Deliberately no `or vector(0)`: before the counter existed, and for
        // a consumer idle long enough for Vector to expire its series, the
        // honest answer is "no data", not zero.
        var ubT   = PromScalarAsync($"sum(increase(gateway_unbilled_requests_total{sel}[{w}]))", ct);
        var ubsT  = PromScalarAsync($"sum(increase(gateway_unbilled_seconds_total{sel}[{w}]))", ct);

        await Task.WhenAll(balT, runT, reqT, badT, inT, outT, gwT, ttfT, itlT, e2eT, cchT, repT, ubT, ubsT);

        return new KeyStats(balT.Result, runT.Result, reqT.Result, badT.Result,
                            inT.Result, outT.Result, gwT.Result, ttfT.Result,
                            itlT.Result, e2eT.Result, cchT.Result,
                            ubT.Result, ubsT.Result, repT.Result);
    }

    // An absent series and a zero are different facts and are printed
    // differently: histogram_quantile over an idle window is NaN, which
    // Prometheus omits entirely, and rendering that as 0.000s would read as
    // "instant" rather than "no data".
    private static string Num(double? v, int dp = 0, string unit = "") =>
        v is null || double.IsNaN(v.Value) || double.IsInfinity(v.Value)
            ? "—"
            : v.Value.ToString("N" + dp.ToString(CultureInfo.InvariantCulture),
                               CultureInfo.InvariantCulture) + unit;

    private static string ShortInstance(string instance)
    {
        var host = instance.Split(':')[0];
        var dash = host.LastIndexOf('-');
        return dash >= 0 && dash + 1 < host.Length ? host[(dash + 1)..] : host;
    }

    // The four reference costs for one token count, as table rows.
    private static IEnumerable<string> CostRows(Prices p, double? tin, double? tout, double? hit)
    {
        if (tin is null || tout is null) yield break;
        yield return $"{"cost at",-14}{"(reference)",14}";
        yield return $"{"OpenRouter",-14}{PriceBook.Usd(PriceBook.Cost(p.OpenRouter, tin.Value, tout.Value)),14}";
        if (p.OpenRouter.CacheReadPerM is not null && hit is not null && double.IsFinite(hit.Value))
            yield return $"{"  cache-aware",-14}{PriceBook.Usd(PriceBook.Cost(p.OpenRouter, tin.Value, tout.Value, hit)),14}";
        yield return $"{"Alibaba SG",-14}{PriceBook.Usd(PriceBook.Cost(p.AlibabaSg, tin.Value, tout.Value)),14}";
        yield return $"{"Alibaba BJ",-14}{PriceBook.Usd(PriceBook.Cost(p.AlibabaBj, tin.Value, tout.Value)),14}";
    }

    private static string PriceFootnote(Prices p) =>
        $"<i>Reference prices for the same model, not a bill: {Esc(p.OpenRouter.Basis)}, "
      + $"{Esc(PriceBook.PerMText(p.OpenRouter))}"
      + (p.Providers > 0 && p.OutMin is { } lo && p.OutMax is { } hi
            ? $" (output ${lo:0.##}–{hi:0.##} across {p.Providers} providers)" : "")
      + $"; Alibaba Cloud {Esc(PriceBook.PerMText(p.AlibabaSg))} Singapore, {Esc(PriceBook.PerMText(p.AlibabaBj))} Beijing, "
      + "as of 2026-09-12. Cache-aware applies this key's measured prefix-cache hit share. "
      + "Cut-off requests are not in it. /prices for the table.</i>";

    private async Task<string> PricesAsync(CancellationToken ct)
    {
        var p = await priceBook.GetAsync(ct);
        string[] rows =
        [
            $"{"USD per M",-11}{"input",7}{"cached",7}{"output",7}",
            Row(p.OpenRouter), Row(p.AlibabaSg), Row(p.AlibabaBj),
        ];
        static string Row(PriceRef r) =>
            $"{r.Label,-11}{r.InPerM,7:0.###}{(r.CacheReadPerM is { } c ? c.ToString("0.###", CultureInfo.InvariantCulture) : "—"),7}{r.OutPerM,7:0.###}";

        // A worked example makes the spread concrete: one typical agent day is
        // input-heavy, and that is where the references disagree most.
        const double exIn = 30_000_000, exOut = 800_000, exHit = 0.9;
        return Table("<b>Reference prices</b> — same model, public providers", rows)
             + $"\n<b>OpenRouter</b>: {Esc(p.OpenRouter.Basis)}"
             + (p.Providers > 0 && p.OutMin is { } lo && p.OutMax is { } hi
                   ? $". Output ranges ${lo:0.##}–{hi:0.##} per M across {p.Providers} providers; the list price above is OpenRouter's headline." : ".")
             + $"\n<b>Alibaba Cloud</b>: {Esc(p.AlibabaSg.Basis)}; {Esc(p.AlibabaBj.Basis)}. No API publishes these, "
             + "so they are updated by hand in bot.cs."
             + $"\n\nExample, 30M in / 0.8M out / 90% cached: OpenRouter {PriceBook.Usd(PriceBook.Cost(p.OpenRouter, exIn, exOut))}"
             + $" (cache-aware {PriceBook.Usd(PriceBook.Cost(p.OpenRouter, exIn, exOut, exHit))}), "
             + $"Alibaba SG {PriceBook.Usd(PriceBook.Cost(p.AlibabaSg, exIn, exOut))}, BJ {PriceBook.Usd(PriceBook.Cost(p.AlibabaBj, exIn, exOut))}."
             + "\n\n<i>What the same tokens would cost bought elsewhere — a yardstick for monitoring, not a bill. "
             + "Shown in /key, /usage and the written report.</i>";
    }

    private async Task<Reply> KeyCardAsync(string name, string window, CancellationToken ct)
    {
        var statsT = KeyStatsAsync(name, window, ct);
        var pricesT = priceBook.GetAsync(ct);
        await Task.WhenAll(statsT, pricesT);
        var st = statsT.Result; var prices = pricesT.Result;

        var rows = new List<string>
        {
            $"{"balance",-14}{Num(st.Balance),14}",
            $"{"runway",-14}{Num(st.Runway, 1),14} d",
            "",
            $"{"requests",-14}{Num(st.Requests),14}",
            $"{"not 2xx",-14}{Num(st.NotOk),14}",
            $"{"tokens in",-14}{Num(st.TokensIn),14}",
            $"{"tokens out",-14}{Num(st.TokensOut),14}",
            $"{"cut, unbilled",-14}{Num(st.Unbilled),14}",
            "",
            $"{"p95 gateway",-14}{Num(st.GatewayP95, 2),14} s",
            $"{"p95 e2e",-14}{Num(st.E2e, 2),14} s",
            $"{"p95 ttft",-14}{Num(st.Ttft, 2),14} s",
            $"{"p95 itl",-14}{Num(st.Itl, 3),14} s",
            $"{"cache hit",-14}{Num(st.CacheHit is null ? null : st.CacheHit * 100, 1),14} %"
        };

        var total = st.ByReplica.Values.Sum();
        if (total > 0)
        {
            rows.Add("");
            foreach (var (inst, v) in st.ByReplica.OrderBy(x => x.Key, StringComparer.Ordinal))
                rows.Add($"{ShortInstance(inst),-14}{v / total * 100,13:N0} %");
        }

        var costRows = CostRows(prices, st.TokensIn, st.TokensOut, st.CacheHit).ToList();
        if (costRows.Count > 0) { rows.Add(""); rows.AddRange(costRows); }

        var body = Table($"<b>{Esc(name)}</b> — last {window}", rows);

        // Say which half of the stack each block came from. The two latencies
        // differ by the gateway filter chain, the router and two network hops,
        // and an operator comparing them needs to know that is expected.
        body += "\n<i>Balance and runway: the ledger. tokens: gateway ai-statistics. "
              + "requests, p95 gateway, cut-offs: the access log. ttft, itl, e2e, cache "
              + "and the replica split: the engine itself.</i>";
        if (costRows.Count > 0) body += "\n" + PriceFootnote(prices);

        if (st.Unbilled is >= 0.5)
            body += $"\n\n⚠️ <b>{Num(st.Unbilled)} request(s) were cut off and charged nothing</b> "
                  + "— client disconnect, stream timeout or upstream error before the final usage "
                  + $"frame. They ran {Num(st.UnbilledSeconds)} s in total; the engine may have "
                  + "generated for up to that long.";

        if (st.Ttft is null && st.Requests > 0)
            body += "\n\n<i>No engine-side numbers in this window. That is normal "
                  + "shortly after a replica roll — the labels start empty — and "
                  + "expected for traffic that did not go through the gateway.</i>";

        var keyboard = new InlineKeyboardMarkup([
            [new InlineKeyboardButton(window == "1h"  ? "• 1h"  : "1h",  $"kc:1h:{name}"),
             new InlineKeyboardButton(window == "24h" ? "• 24h" : "24h", $"kc:24h:{name}"),
             new InlineKeyboardButton(window == "7d"  ? "• 7d"  : "7d",  $"kc:7d:{name}")],
            [new InlineKeyboardButton("Settings", $"kp:{name}"),
             new InlineKeyboardButton("Traces", $"kt:{name}"),
             new InlineKeyboardButton("Report ↓", $"kr:{name}")],
            [new InlineKeyboardButton("← All keys", "kl:")]
        ]);
        return new Reply(body, keyboard);
    }

    // Tracing BY KEY rather than by request. Langfuse already groups by
    // consumer — ai-statistics puts the authenticated name on every gateway
    // span as langfuse.user.id — so a consumer's traces are one URL away, with
    // no request id to find first.
    private Reply KeyTraceCard(string name)
    {
        var text =
            $"<b>{Esc(name)}</b> — where its requests are\n\n"
          + "<b>Langfuse</b> groups them under this consumer already: every gateway "
          + "span carries the authenticated name as the Langfuse user id. That view "
          + "gives you each request with its tokens and latency.\n\n"
          + "<b>To open one request end to end</b>, take a request id from there or "
          + "from the fact table and run /trace on it — the engine's own spans "
          + "sit in a different trace, and that command prints the two-hop join.\n\n"
          + "<b>The billing-grade list</b>, ClickHouse on the host:\n"
          + $"<pre>SELECT ts, request_id, status, total_tokens, duration_ms\nFROM gateway.requests FINAL\nWHERE consumer = '{Esc(name)}'\nORDER BY ts DESC LIMIT 20;</pre>\n"
          + "<i>This bot cannot run that itself: it is on the edge network and both "
          + "stores are backend-only, deliberately.</i>";

        var buttons = new List<InlineKeyboardButton[]>();
        if (cfg.LangfuseProjectId.Length > 0)
            buttons.Add([new InlineKeyboardButton(
                "Open in Langfuse",
                null,
                $"{cfg.LangfuseUrl}/project/{cfg.LangfuseProjectId}/users/{Uri.EscapeDataString(name)}")]);
        else
            text += $"\n\n<i>Set LANGFUSE_PROJECT_ID to get a button here. The path is "
                  + $"{Esc(cfg.LangfuseUrl)}/project/&lt;project&gt;/users/{Esc(name)}</i>";

        buttons.Add([new InlineKeyboardButton("← Back", $"kc:24h:{name}")]);
        return new Reply(text, new InlineKeyboardMarkup(buttons.ToArray()));
    }


    // ---- the written report -------------------------------------------------
    //
    // The node writes its own report: the numbers go to qwen36-27b through the
    // gateway, authenticated with the ADMIN key, and what comes back is an HTML
    // file sent to the chat.
    //
    // Two things about that are worth stating plainly.
    //
    // 1. docs/KEY-TIERS.md describes the admin tier as "management only, never
    //    inference". This is the exception, made deliberately: it is the only
    //    credential the bot already holds, the request is the operator's own,
    //    and it is metered like any other — roughly 4k tokens a report against
    //    quota-admin's balance. The doc records the exception.
    // 2. The model is told the numbers and told not to invent any. It can still
    //    be wrong about what they MEAN. The report is a readable second opinion
    //    on data the dashboards already show; it is not a source of truth, and
    //    the file says so.
    //
    // Thinking is disabled via chat_template_kwargs. Measured on this node: with
    // it on, a report spends its first several hundred tokens deliberating and
    // the HTML arrives truncated; with it off, the first character is
    // `<!doctype html>` and 1200 tokens take 12s.
    private Reply KeyReportStart(string name, long chatId, CancellationToken ct)
    {
        // Detached on purpose. The worker gate bounds how many commands run at
        // once, and a 40-second model call has no business holding one of those
        // slots while every other command queues behind it.
        _ = Task.Run(async () =>
        {
            try { await KeyReportAsync(name, chatId, CancellationToken.None); }
            catch (Exception ex)
            {
                log.LogError(ex, "report generation failed for {Name}", name);
                await SendAsync(chatId, new Reply(
                    $"Could not generate the report for <b>{Esc(name)}</b>. "
                  + "The stats above are unaffected — /status will say if the gateway is the problem."),
                    CancellationToken.None);
            }
        }, CancellationToken.None);

        return new Reply($"Writing <b>{Esc(name)}</b>'s report on the node itself. "
                       + "Takes under a minute; the file arrives here.");
    }

    private async Task KeyReportAsync(string name, long chatId, CancellationToken ct)
    {
        // 24h and 7d together, so the report can say whether today is typical.
        var dayT = KeyStatsAsync(name, "24h", ct);
        var weekT = KeyStatsAsync(name, "7d", ct);
        var tiersT = ledger.TiersAsync(ct);
        var pricesT = priceBook.GetAsync(ct);
        await Task.WhenAll(dayT, weekT, tiersT, pricesT);
        var day = dayT.Result; var week = weekT.Result; var prices = pricesT.Result;
        var tier = tiersT.Result.GetValueOrDefault(name, "unassigned");

        var facts = new StringBuilder();
        facts.Append("consumer=").Append(name).Append("\ntier=").Append(tier).Append('\n');
        facts.Append("reference_prices_usd_per_million=")
             .Append($"OpenRouter {PriceBook.PerMText(prices.OpenRouter)} ({prices.OpenRouter.Basis}); ")
             .Append($"Alibaba Singapore {PriceBook.PerMText(prices.AlibabaSg)}; Alibaba Beijing {PriceBook.PerMText(prices.AlibabaBj)}\n");
        Block(facts, "last_24h", day, prices);
        Block(facts, "last_7d", week, prices);

        var html = await GenerateReportAsync(name, facts.ToString(), ct);
        var stamp = DateTimeOffset.UtcNow.ToString("yyyyMMdd-HHmm", CultureInfo.InvariantCulture);

        // The costs go in the caption as well, computed here: the model is told
        // to include them, but a number the operator monitors should not depend
        // on a model remembering to copy it.
        string Line(string label, KeyStats st) =>
            st.TokensIn is { } i && st.TokensOut is { } o
                ? $"{label}: OpenRouter {PriceBook.Usd(PriceBook.Cost(prices.OpenRouter, i, o))} "
                + $"(cache-aware {PriceBook.Usd(PriceBook.Cost(prices.OpenRouter, i, o, st.CacheHit))}) · "
                + $"Alibaba SG {PriceBook.Usd(PriceBook.Cost(prices.AlibabaSg, i, o))} · "
                + $"BJ {PriceBook.Usd(PriceBook.Cost(prices.AlibabaBj, i, o))}\n"
                : "";
        await SendDocumentAsync(chatId, $"{name}-{stamp}.html", Encoding.UTF8.GetBytes(html),
            $"<b>{Esc(name)}</b> — written by the node, {stamp} UTC.\n"
          + "At reference prices, not a bill:\n" + Line("24h", day) + Line("7d", week)
          + "<i>Numbers are measured; the reading of them is the model's.</i>", ct);

        static void Block(StringBuilder b, string label, KeyStats st, Prices p)
        {
            b.Append('[').Append(label).Append("]\n");
            b.Append("balance_tokens=").Append(Num(st.Balance)).Append('\n');
            b.Append("runway_days=").Append(Num(st.Runway, 1)).Append('\n');
            b.Append("requests=").Append(Num(st.Requests)).Append('\n');
            b.Append("not_2xx=").Append(Num(st.NotOk)).Append('\n');
            b.Append("tokens_in=").Append(Num(st.TokensIn)).Append('\n');
            b.Append("tokens_out=").Append(Num(st.TokensOut)).Append('\n');
            b.Append("cut_unbilled_requests=").Append(Num(st.Unbilled)).Append('\n');
            b.Append("cut_unbilled_seconds=").Append(Num(st.UnbilledSeconds)).Append('\n');
            if (st.TokensIn is { } i && st.TokensOut is { } o)
            {
                b.Append("cost_usd_openrouter=").Append(PriceBook.Usd(PriceBook.Cost(p.OpenRouter, i, o))).Append('\n');
                b.Append("cost_usd_openrouter_cache_aware=").Append(PriceBook.Usd(PriceBook.Cost(p.OpenRouter, i, o, st.CacheHit))).Append('\n');
                b.Append("cost_usd_alibaba_singapore=").Append(PriceBook.Usd(PriceBook.Cost(p.AlibabaSg, i, o))).Append('\n');
                b.Append("cost_usd_alibaba_beijing=").Append(PriceBook.Usd(PriceBook.Cost(p.AlibabaBj, i, o))).Append('\n');
            }
            b.Append("gateway_p95_s=").Append(Num(st.GatewayP95, 3)).Append('\n');
            b.Append("engine_e2e_p95_s=").Append(Num(st.E2e, 3)).Append('\n');
            b.Append("engine_ttft_p95_s=").Append(Num(st.Ttft, 3)).Append('\n');
            b.Append("engine_itl_p95_s=").Append(Num(st.Itl, 4)).Append('\n');
            b.Append("prefix_cache_hit=").Append(Num(st.CacheHit, 3)).Append('\n');
            var total = st.ByReplica.Values.Sum();
            foreach (var (inst, v) in st.ByReplica.OrderBy(x => x.Key, StringComparer.Ordinal))
                b.Append("output_tokens_").Append(ShortInstance(inst)).Append('=')
                 .Append(Num(v)).Append(total > 0 ? $" ({v / total * 100:N0}%)" : "").Append('\n');
            b.Append('\n');
        }
    }

    private const string ReportSystemPrompt =
        "You are a site reliability engineer writing a short report about ONE API consumer "
        + "of a self-hosted LLM inference node. Output ONE complete standalone HTML document "
        + "and nothing else: no markdown, no code fences, no commentary before or after it. "
        + "Start with <!doctype html>. Inline all CSS; the file is read offline. "
        + "A dash means the metric had no data in that window — say so rather than reading it "
        + "as zero. Never state a number that is not in the data you were given, and never "
        + "guess at a cause you cannot support from it.";

    // The node's own measured characteristics. Without these the model has no
    // basis for calling a number good or bad, and would either hedge on
    // everything or invent a baseline.
    private const string ReportNodeFacts =
        "Node: 2x A100 80GB PCIe, Qwen3.8-27B in BF16, tensor parallel 1 with two "
        + "independent replicas behind a cache-aware router. EAGLE speculative decoding. "
        + "Single-stream output ceiling ~55 tok/s; whole-node ceiling ~387 tok/s; engine "
        + "concurrency 8 (2 replicas x 4). Decode is memory-bandwidth bound. Measured cost: "
        + "an output token costs ~68x an uncached input token and ~4800x a cached one, so "
        + "prefix cache hit rate and the input:output ratio drive cost more than volume does. "
        + "Quota is a single total-token balance; input and output are charged the same. "
        + "A request cut off before its final usage frame (client disconnect, stream timeout, "
        + "upstream error) is charged zero tokens although the engine may have worked on it; "
        + "cut_unbilled_* counts those and their wall time. "
        + "cost_usd_* is what the same input and output tokens would cost buying this model from "
        + "public providers (OpenRouter list price, the same with cached input priced as cached, "
        + "Alibaba Cloud Singapore and Beijing). They are reference prices for monitoring, NOT what "
        + "the consumer was charged. The report MUST include a 'Reference cost' table with every "
        + "cost_usd_* value for both windows, and must call them reference prices.";

    private async Task<string> GenerateReportAsync(string name, string facts, CancellationToken ct)
    {
        var body = new JsonObject
        {
            ["model"] = cfg.ModelId,
            ["max_tokens"] = 4000,
            ["temperature"] = 0.3,
            ["stream"] = false,
            // Qwen3 reasons by default. See the note on KeyReportStart.
            ["chat_template_kwargs"] = new JsonObject { ["enable_thinking"] = false },
            // Cast to JsonNode on purpose. The collection initialiser would bind
            // JsonArray.Add<T>(T), which is RequiresDynamicCode/UnreferencedCode
            // and fails the AOT build outright — see quota-bot/README.md, which
            // records this exact pair of IL2026/IL3050 errors from the first
            // build. The JsonNode overload is the trim-safe one.
            ["messages"] = new JsonArray
            {
                (JsonNode)new JsonObject
                {
                    ["role"] = "system",
                    ["content"] = ReportSystemPrompt
                },
                (JsonNode)new JsonObject
                {
                    ["role"] = "user",
                    ["content"] = $"{ReportNodeFacts}\n\nMeasurements:\n\n{facts}\n"
                        + "Write, in this order: a one-line verdict; a table of every metric "
                        + "with a plain-language reading of each; how the last 24h compares "
                        + "with the last 7 days; what looks healthy; what is worth attention; "
                        + "and concrete next steps. Title it with the consumer name."
                }
            }
        };

        using var content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json");
        using var r = await http.CreateClient("inference").PostAsync("v1/chat/completions", content, ct);
        var raw = await r.Content.ReadAsStringAsync(ct);
        if (!r.IsSuccessStatusCode)
            throw new InvalidOperationException($"gateway returned HTTP {(int)r.StatusCode}: {Head(raw)}");

        var node = JsonNode.Parse(raw);
        var choice = node?["choices"]?[0];
        var text = choice?["message"]?["content"]?.GetValue<string>() ?? "";
        var finish = choice?["finish_reason"]?.GetValue<string>() ?? "";
        var used = node?["usage"]?["completion_tokens"]?.GetValue<int>() ?? 0;

        text = StripFences(text).Trim();
        if (text.Length == 0)
            throw new InvalidOperationException("the model returned an empty report");

        // A model that ran out of budget stops mid-tag. Rather than ship a file
        // the browser silently half-renders, say so at the top and close it.
        if (finish == "length")
            text = InsertTruncationNotice(text, used);

        if (!text.Contains("<html", StringComparison.OrdinalIgnoreCase))
            text = "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
                 + Fmt.Esc(name) + "</title></head><body>" + text + "</body></html>";
        return text;
    }

    // Belt and braces: the prompt forbids fences, and models emit them anyway.
    private static string StripFences(string t)
    {
        t = t.Trim();
        if (!t.StartsWith("```", StringComparison.Ordinal)) return t;
        var nl = t.IndexOf('\n');
        if (nl < 0) return t;
        t = t[(nl + 1)..];
        var close = t.LastIndexOf("```", StringComparison.Ordinal);
        return close >= 0 ? t[..close] : t;
    }

    private static string InsertTruncationNotice(string html, int tokens)
    {
        const string notice =
            "<p style=\"background:#fee;border:1px solid #c00;padding:.75em;margin:0 0 1em\">"
            + "<b>This report is incomplete.</b> The model hit its output limit "
            + "after {0} tokens and stopped mid-document. Everything above the cut is "
            + "still based on the measured numbers.</p>";
        var body = html.IndexOf("<body", StringComparison.OrdinalIgnoreCase);
        var open = body >= 0 ? html.IndexOf('>', body) : -1;
        var banner = string.Format(CultureInfo.InvariantCulture, notice, tokens);
        var closed = html + "\n</body></html>";
        return open >= 0 ? closed[..(open + 1)] + banner + closed[(open + 1)..] : banner + closed;
    }

    // sendDocument is multipart, unlike every other call this bot makes. Worth
    // the exception: a report is a file an operator keeps, forwards and opens
    // in a browser, and Telegram truncates a message at 4096 characters.
    private async Task SendDocumentAsync(
        long chatId, string filename, byte[] bytes, string caption, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent(chatId.ToString(CultureInfo.InvariantCulture)), "chat_id");
        form.Add(new StringContent(caption), "caption");
        form.Add(new StringContent("HTML"), "parse_mode");
        var file = new ByteArrayContent(bytes);
        file.Headers.ContentType = new MediaTypeHeaderValue("text/html");
        form.Add(file, "document", filename);

        using var r = await http.CreateClient("telegram").PostAsync("sendDocument", form, ct);
        if (!r.IsSuccessStatusCode)
            log.LogError("sendDocument failed: HTTP {Code} {Body}",
                (int)r.StatusCode, Head(await r.Content.ReadAsStringAsync(ct)));
    }

    // What Langfuse is for, and what it is NOT for. Written 2026-09-05 after
    // finding its token aggregate was 293x reality (decode_loop spans carrying
    // attributes.decode_ct, read as usage — now filtered at the collector).
    // The point of this command is that people were reading numbers off
    // Langfuse and believing them.
    private static string LangfuseHelp() =>
        """
        <b>What Langfuse shows you</b>

        <b>Use it for one thing</b>
        Opening a single request and seeing the engine's phase breakdown —
        prefill_waiting, prefill_forward, decode_forward, tokenize, and a
        <code>Req</code> span with token counts and TTFT. That waterfall is the
        only place the inside of one request is visible.

        Get there with /trace &lt;request-id&gt;. It takes two hops and the
        command prints both.

        <b>Filtering by consumer works</b>
        Every gateway span carries the consumer as Langfuse's user id, set by
        the ai-statistics plugin from the authenticated key. A new key shows up
        on its first request — nothing to register. Filter it in the Users page
        or with <code>?userId=&lt;name&gt;</code> on the observations API.

        In Grafana the same filter is the <b>Consumer</b> picker on the
        Usage &amp; Quota and AI Gateway dashboards. That list is read from the
        ledger, so a key appears there before it has sent anything.

        <b>Do NOT use it for</b>
        • <b>Billing or usage totals.</b> No model pricing is configured, so
          cost is meaningless, and token sums are derived from spans rather
          than the ledger. Use /usage, /top and /balance.
        • <b>Sessions.</b> That page is genuinely empty — nothing sets a
          session id on the span, so multi-turn chats do not group.
        • <b>Following one user INTO the engine.</b> The consumer is on the
          gateway span; the engine's phase breakdown sits in a different trace,
          because the router starts its own. /trace crosses that gap.
        • <b>Node health.</b> That is Grafana and /health.

        <b>If a Langfuse doc page 404s the API</b>
        This runs 4.5.0 in <code>events_only</code> mode. The v3 endpoints are
        gone by design and return a message saying so. Use
        <code>/api/public/v2/observations</code> and
        <code>/api/public/v2/metrics</code>.

        <b>Why 4.5.0 and not the latest</b>
        4.30.0 exists and the upgrade is cheap — two metadata-only ClickHouse
        migrations and seven Prisma ones, none touching data we hold. We stay
        because none of it addresses what was confusing: the 25 releases are
        evaluator and experiment work, and <code>events_only</code> is v4's
        intended end state, not a bug to upgrade out of. Reassessed 2026-09-05.

        <b>One number used to lie</b>
        Until 2026-09-05 its token aggregate read ~101M per day against a real
        345k, because it counted per-decode-iteration spans as usage. Those are
        dropped at the collector now, which also removed 77% of span volume.
        Numbers before that date in Langfuse are not trustworthy.

        Full map of which store answers what: <code>docs/METRICS-ECOSYSTEM.md</code>
        """;

    // Rendered from Policy.All, the same table /tier validates against and
    // /policy resolves from, so the description and the thing being applied
    // cannot drift apart.
    private string TiersHelp()
    {
        var header = $"{"tier",-8}{"quota",6}{"refill",8}{"daily",6}{"tpm",6}{"max",7}";
        var rows = Policy.All.Where(t => t.Name != "admin").Select(t =>
            $"{t.Name,-8}{Policy.Compact(t.Quota),6}{Policy.RefillName(t.Refill),8}"
          + $"{(t.Daily == 0 ? "\u221e" : Policy.Compact(t.Daily)),6}"
          + $"{(t.Tpm == 0 ? "\u221e" : Policy.Compact(t.Tpm)),6}"
          + $"{(t.MaxTokens == 0 ? "gw" : t.MaxTokens.ToString(CultureInfo.InvariantCulture)),7}");

        var body = Table("<b>Policy tiers</b> \u2014 defaults", new[] { header }.Concat(rows));
        foreach (var t in Policy.All)
            body += $"\n<b>{t.Name}</b> \u2014 {Esc(t.For)}";

        body += "\n\n<b>What each setting means</b>";
        foreach (var f in Policy.Fields)
            body += $"\n<code>{f.Key}</code> \u2014 {Esc(f.Meaning)}";

        return body
          + "\n\n<b>Every value is a default.</b> A consumer follows its tier until one value is "
          + "set on it with <code>/set &lt;name&gt; &lt;setting&gt; &lt;value&gt;</code>; that value then "
          + "stays when the tier changes, and <code>default</code> puts it back. /policy shows which is which."
          + "\n\n<b>Enforced:</b> the balance on every request; daily and tpm at the gateway (a 429 "
          + "with the reset time); refill at 00:00 UTC. A window opens at a key's first request, and one "
          + "request can overshoot a limit by its own size. <b>Not enforceable:</b> max_tokens per key — "
          + $"the gateway holds one {cfg.OutputLimit:N0} ceiling for everyone."
          + "\n\n<i>Refill replaces the balance: unused tokens do not carry over. Turning refill on never "
          + "resets a balance at once; the first reset is at the next boundary.</i>"
          + "\n\n<b>Quota is one number.</b> Input and output are deducted at the same rate, "
          + "though on this node an output token costs roughly 68\u00d7 an uncached input token "
          + "and ~4800\u00d7 a cached one \u2014 /usage and /top show the i:o ratio."
          + "\n\n<i>Setting a tier or a quota never changes a balance by itself.</i>";
    }

    private async Task<Reply> TierAsync(string name, string tier, CancellationToken ct)
    {
        tier = tier.ToLowerInvariant();
        if (!Policy.Tiers.TryGetValue(tier, out var t))
            return new Reply($"Unknown tier <code>{Esc(tier)}</code>.\n\nOne of: "
                 + string.Join(", ", Policy.Tiers.Keys.Select(k => $"<code>{k}</code>")));

        var balances = await ledger.ListAsync(ct);
        if (!balances.ContainsKey(name))
            return new Reply($"<b>{Esc(name)}</b> has no balance recorded, so it is not a live consumer.\n\n"
                 + "Create it with <code>/newkey</code> first.");

        var before = await ledger.TierAsync(name, ct);
        await ledger.SetTierAsync(name, tier, ct);
        await AuditAsync($"tier name={name} from={before ?? "none"} to={tier}", ct);

        var notice = $"<b>{Esc(name)}</b> is now <b>{Esc(tier)}</b> \u2014 {Esc(t.For)}.";

        // Say plainly where the balance stands against the tier, and do NOT
        // move it. Changing a balance is money, and it is a separate decision
        // from recording what tier someone is on.
        var bal = balances[name];
        if (tier != "admin" && bal != t.Quota)
            notice += $"\n\u26a0\ufe0f Balance is <code>{bal:N0}</code>, tier quota is <code>{t.Quota:N0}</code>. "
                    + $"Nothing was changed \u2014 <code>/setquota {Esc(name)} {t.Quota}</code> aligns it.";
        return await PolicyCardAsync(name, notice, ct);
    }

    // ---- per-consumer policy ------------------------------------------------
    //
    // What one consumer is allowed, value by value, and where each value comes
    // from. The "from" column is the point of the screen: a consumer that
    // follows its tier moves when the tier's default moves, and one that was set
    // by hand does not, and those look identical in every other view.
    private async Task<Reply> PolicyCardAsync(string name, string? notice, CancellationToken ct)
    {
        if (!SafeName(name) || !(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.");

        // The limiter's OWN counters, read from the Redis keys it writes — so
        // "used" here is exactly what it will compare against the limit, window
        // start and all, rather than a Prometheus approximation of a calendar day.
        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        var balT = ledger.ListAsync(ct);
        var dayT = ledger.CounterAsync(LimiterSync.CounterKey(name, 86_400), ct);
        var minT = ledger.CounterAsync(LimiterSync.CounterKey(name, 60), ct);
        var markT = ledger.RefillMarkerAsync(name, ct);
        await Task.WhenAll(tierT, ovT, balT, dayT, minT, markT);

        var tier = tierT.Result;
        var resolved = Policy.Resolve(tier, ovT.Result);

        var rows = new List<string> { $"{"setting",-11}{"value",14}  from" };
        foreach (var r in resolved)
            rows.Add($"{r.Field.Label,-11}{(r.Value is null ? "\u2014" : Policy.Show(r.Field, r.Value, cfg.OutputLimit)),14}  {r.Source}");

        var body = (notice is null ? "" : notice + "\n\n")
                 + Table($"<b>{Esc(name)}</b> \u2014 tier <b>{Esc(tier ?? "unassigned")}</b>", rows);

        var bal = balT.Result.TryGetValue(name, out var b) ? b.ToString("N0", CultureInfo.InvariantCulture) : "not seeded";
        body += $"\nBalance <code>{bal}</code>";

        static long? Limit(ResolvedSetting r) =>
            long.TryParse(r.Value, CultureInfo.InvariantCulture, out var n) && n > 0 ? n : null;
        string Window(string label, (long? Used, long Ttl) c, long? limit)
        {
            if (limit is null) return $"\n{label}: no limit";
            if (c.Used is not { } used) return $"\n{label}: <code>0</code> of <code>{limit:N0}</code> \u2014 no window open";
            var line = $"\n{label}: <code>{used:N0}</code> of <code>{limit:N0}</code>, resets in {Fmt.Duration(c.Ttl)}";
            return used > limit ? line + " \u26d4 <b>refusing requests</b>" : line;
        }
        var inScope = LimiterSync.InScope(name);
        body += Window("24h window", dayT.Result, inScope ? Limit(resolved[2]) : null)
              + Window("60s window", minT.Result, inScope ? Limit(resolved[3]) : null);

        // Refill: when, and to what. The marker says whether the job has seen
        // this consumer yet \u2014 the first pass only arms it, never refills.
        var mode = resolved[1].Value ?? "manual";
        if (mode != "manual" && Limit(resolved[0]) is { } q)
            body += $"\nNext refill <b>{RefillJob.Next(mode, DateTimeOffset.UtcNow):yyyy-MM-dd HH:mm} UTC</b> sets the balance to <code>{q:N0}</code>"
                  + (markT.Result is null ? " <i>(armed on the job's next pass)</i>" : "");
        else
            body += "\nRefill: manual";

        body += "\n\n" + limiter.StatusLine(name)
              + "\n<i>max_tokens is recorded only: the gateway cannot vary it per key.</i>"
              + "\n\n<i>Tap a value to change it. \u2731 marks one set on this key rather than taken from its tier.</i>";

        // One button per setting, showing its value, so the screen is both the
        // readout and the control. Labels are short: Telegram truncates a
        // button to its width, and two share a row.
        InlineKeyboardButton Btn(ResolvedSetting r) => new(
            $"{r.Field.Label}: {ShortValue(r)}{(r.Source == "set" ? " \u2731" : "")}",
            $"pe:{r.Field.Key}:{name}");
        var keyboard = new InlineKeyboardMarkup([
            [Btn(resolved[0]), Btn(resolved[1])],
            [Btn(resolved[2]), Btn(resolved[3])],
            [Btn(resolved[4]), new InlineKeyboardButton($"tier: {tier ?? "none"}", $"pt:{name}")],
            [new InlineKeyboardButton($"balance: {(balT.Result.ContainsKey(name) ? Policy.Compact(b) : "\u2014")}", $"pb:{name}")],
            [new InlineKeyboardButton("\u2190 Key card", $"kc:24h:{name}")]
        ]);
        return new Reply(body, keyboard);
    }

    private string ShortValue(ResolvedSetting r)
    {
        if (r.Value is null) return "\u2014";
        if (r.Field.Key == "refill") return r.Value;
        if (!long.TryParse(r.Value, CultureInfo.InvariantCulture, out var n)) return r.Value;
        return (r.Field.Key, n) switch
        {
            ("max_tokens", 0) => $"gw {Policy.Compact(cfg.OutputLimit)}",
            ("max_tokens", _) => n.ToString(CultureInfo.InvariantCulture),
            ("daily" or "tpm", 0) => "\u221e",
            _ => Policy.Compact(n),
        };
    }

    // ---- settings buttons -------------------------------------------------
    //
    //   pp:<name>                   the settings screen (edit in place)
    //   pe:<field>:<name>           one setting's editor
    //   pv:<field>:<value>:<name>   set it (value "d" = back to the tier)
    //   pc:<field>:<name>           ask for a typed value (field "topup" = custom top-up)
    //   pt:<name> / ps:<tier>:<name> change tier
    //   pb:<name>                   balance: top-ups and set-to-quota, token-bound
    //
    // Names cannot contain ':' (SafeName) and neither can a stored value, so
    // splitting is unambiguous. The longest, pv:max_tokens:70000:<32 chars>, is
    // 52 bytes, inside Telegram's 64-byte callback_data limit.
    private async Task<(Reply Reply, bool Edit)> PolicyCallbackAsync(string data, long userId, CancellationToken ct)
    {
        var kind = data[..3];
        var parts = data[3..].Split(':');
        var name = parts[^1];
        if (!SafeName(name)) return (new Reply("That is not a consumer name this bot recognises."), false);

        switch (kind)
        {
            case "pp:": return (await PolicyCardAsync(name, null, ct), true);
            case "pt:": return (await TierPickerAsync(name, ct), true);
            case "pb:": return (await BalanceScreenAsync(userId, name, ct), true);

            case "pe:" when parts.Length == 2 && Policy.Field(parts[0]) is { } f:
                return (await FieldEditorAsync(name, f, ct), true);

            case "pv:" when parts.Length == 3 && Policy.Field(parts[0]) is { } f:
                return (await SetPolicyAsync(name, f.Key, parts[1] == "d" ? "default" : parts[1], ct), true);

            case "ps:" when parts.Length == 2:
                return (await TierAsync(name, parts[0], ct), true);

            case "pc:" when parts.Length == 2 && parts[0] == "topup":
                return (AskInput(userId, InputKind.TopUp, name, null,
                    $"Send how many tokens to <b>add</b> to <b>{Esc(name)}</b>: 500000, 2M or 500k."), false);

            case "pc:" when parts.Length == 2 && Policy.Field(parts[0]) is { } f:
                return (AskInput(userId, InputKind.Field, name, f.Key,
                    $"Send the new <b>{Esc(f.Label)}</b> for <b>{Esc(name)}</b> \u2014 {Esc(f.Meaning)}.\n"
                  + (f.Key == "refill" ? "One of manual, daily, weekly, monthly." : "Like 2000000, 2M or 500k")
                  + (f.Key is "daily" or "tpm" ? ", or unlimited." : f.Key == "refill" ? "" : ".")
                  + " <code>default</code> follows the tier again."), false);
        }
        return (new Reply("Unknown button. Run /policy again."), false);
    }

    private static readonly Dictionary<string, string[]> Presets = new(StringComparer.Ordinal)
    {
        ["quota"]      = ["1000000", "10000000", "50000000", "100000000", "300000000", "500000000"],
        ["refill"]     = ["manual", "daily", "weekly", "monthly"],
        ["daily"]      = ["500000", "1000000", "5000000", "10000000", "20000000", "30000000", "50000000", "100000000", "0"],
        ["tpm"]        = ["200000", "300000", "600000", "1000000", "2000000", "0"],
        ["max_tokens"] = ["4096", "8192", "16384", "32768", "0"],
    };

    private async Task<Reply> FieldEditorAsync(string name, PolicyField f, CancellationToken ct)
    {
        if (!(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.");
        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        await Task.WhenAll(tierT, ovT);

        var tier = tierT.Result;
        var now = Policy.Resolve(tier, ovT.Result).First(r => r.Field.Key == f.Key);
        var tierDefault = tier is not null && Policy.Tiers.TryGetValue(tier, out var def) ? Policy.TierValue(def, f) : null;

        var text = $"<b>{Esc(name)}</b> \u2014 <b>{Esc(f.Label)}</b>\n{Esc(f.Meaning)}\n\n"
                 + $"Now <code>{(now.Value is null ? "unset" : Esc(Policy.Show(f, now.Value, cfg.OutputLimit)))}</code> "
                 + (now.Source == "set" ? "(set on this key)" : now.Source == "tier" ? "(from the tier)" : "")
                 + (tierDefault is null ? "\nNo tier, so there is no default to go back to."
                                        : $"\nTier <b>{Esc(tier!)}</b> default: <code>{Esc(Policy.Show(f, tierDefault, cfg.OutputLimit))}</code>")
                 + $"\n\n<i>{Esc(f.Status)}.</i>";

        var buttons = Presets[f.Key].Select(v => new InlineKeyboardButton(
            (v == now.Value ? "\u2022 " : "") + ShortValue(new ResolvedSetting(f, v, "")),
            $"pv:{f.Key}:{v}:{name}")).ToList();
        var rows = new List<InlineKeyboardButton[]>();
        for (var i = 0; i < buttons.Count; i += 3) rows.Add(buttons.Skip(i).Take(3).ToArray());

        var tail = new List<InlineKeyboardButton>();
        if (now.Source == "set" && tierDefault is not null)
            tail.Add(new InlineKeyboardButton("Tier default", $"pv:{f.Key}:d:{name}"));
        tail.Add(new InlineKeyboardButton("Custom\u2026", $"pc:{f.Key}:{name}"));
        rows.Add(tail.ToArray());
        rows.Add([new InlineKeyboardButton("\u2190 Settings", $"pp:{name}")]);
        return new Reply(text, new InlineKeyboardMarkup(rows.ToArray()));
    }

    private async Task<Reply> TierPickerAsync(string name, CancellationToken ct)
    {
        var tier = await ledger.TierAsync(name, ct);
        var overrides = await ledger.OverridesAsync(name, ct);
        var rows = Policy.All.Select(t => new[] { new InlineKeyboardButton(
            $"{(t.Name == tier ? "\u2022 " : "")}{t.Name} \u00b7 {Policy.Compact(t.Quota)} \u00b7 {Policy.RefillName(t.Refill)}",
            $"ps:{t.Name}:{name}") }).ToList();
        rows.Add([new InlineKeyboardButton("\u2190 Settings", $"pp:{name}")]);
        var kept = overrides.Count == 0 ? ""
            : $"\n\nKept whatever the tier: {string.Join(", ", overrides.Keys.Select(k => $"<code>{Esc(k)}</code>"))} "
            + "(set on this key). Put one back to the tier from its own button.";
        return new Reply($"<b>{Esc(name)}</b> \u2014 tier is <b>{Esc(tier ?? "unassigned")}</b>.\n\n"
                       + "A new tier changes every value that follows the tier. It never changes the balance."
                       + kept, new InlineKeyboardMarkup(rows.ToArray()));
    }

    // Money, so every button is a single-use token bound to this operator:
    // a double tap on +10M adds 10M once. Top-ups run on tap, the same as a
    // typed /topup; replacing the balance keeps its confirmation step.
    private async Task<Reply> BalanceScreenAsync(long userId, string name, CancellationToken ct)
    {
        if (!(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.");
        var balT = QuotaGetAsync(name, ct);
        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        await Task.WhenAll(balT, tierT, ovT);

        var quotaStr = Policy.Resolve(tierT.Result, ovT.Result)[0].Value;
        long? quota = long.TryParse(quotaStr, CultureInfo.InvariantCulture, out var q) && q > 0 ? q : null;
        var ttl = TimeSpan.FromMinutes(10);

        InlineKeyboardButton Add(long amount) => new($"+{Policy.Compact(amount)}",
            "ok:" + Tokenize(userId, ttl, async c => new Reply(await TopUpAsync(name, amount.ToString(CultureInfo.InvariantCulture), c))));

        var rows = new List<InlineKeyboardButton[]> { new[] { Add(1_000_000), Add(5_000_000), Add(10_000_000) } };
        if (quota is { } qv)
            rows.Add([Add(qv) with { Text = $"+quota ({Policy.Compact(qv)})" },
                      new InlineKeyboardButton($"Set to {Policy.Compact(qv)}\u2026",
                          "ok:" + Tokenize(userId, ttl, c => Task.FromResult(Arm(userId, name, qv.ToString(CultureInfo.InvariantCulture), PendingKind.SetQuota))))]);
        rows.Add([new InlineKeyboardButton("Custom top-up\u2026", $"pc:topup:{name}")]);
        rows.Add([new InlineKeyboardButton("\u2190 Settings", $"pp:{name}")]);

        return new Reply(
            $"<b>{Esc(name)}</b> \u2014 balance <code>{balT.Result?.ToString("N0", CultureInfo.InvariantCulture) ?? "unknown"}</code>"
          + (quota is null ? "" : $", quota <code>{quota:N0}</code>")
          + "\n\nTop-ups <b>add</b> and run on tap, once per button. <b>Set</b> replaces the balance and asks first."
          + "\n<i>Buttons expire in 10 minutes.</i>",
            new InlineKeyboardMarkup(rows.ToArray()));
    }

    private async Task<Reply> SetPolicyAsync(string name, string fieldArg, string valueArg, CancellationToken ct)
    {
        if (Policy.Field(fieldArg) is not { } field)
            return new Reply($"Unknown setting <code>{Esc(Head(fieldArg))}</code>.\n\nOne of: "
                 + string.Join(", ", Policy.Fields.Select(f => $"<code>{f.Key}</code>")));
        if (!IsValidName(name) || !(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.");

        string? stored = null;
        if (!Policy.IsDefaultWord(valueArg)
            && !Policy.TryNormalise(field, valueArg, cfg.OutputLimit, out stored, out var why))
            return new Reply($"{Esc(why)}\n\n<code>default</code> puts {Esc(field.Key)} back to the tier's value.");

        var before = await ledger.OverridesAsync(name, ct);
        if (stored is null) await ledger.ClearOverrideAsync(name, field.Key, ct);
        else await ledger.SetOverrideAsync(name, field.Key, stored, ct);
        await AuditAsync($"set name={name} field={field.Key} from={before.GetValueOrDefault(field.Key, "tier")} to={stored ?? "tier"}", ct);

        var notice = stored is null
            ? $"<b>{Esc(field.Label)}</b> follows the tier again."
            : $"<b>{Esc(field.Label)}</b> set to <code>{Esc(Policy.Show(field, stored, cfg.OutputLimit))}</code> for this consumer.";
        if (field.Key == "quota")
            notice += "\n<i>The balance was not changed. /topup or /setquota move it.</i>";
        return await PolicyCardAsync(name, notice, ct);
    }

    private static bool ValidWindow(string w) => w is "1h" or "24h" or "7d" or "30d";

    private static string BadWindow(string w) =>
        $"Unknown window <code>{Esc(w)}</code>.\n\nUse one of <code>1h</code>, <code>24h</code>, <code>7d</code>, <code>30d</code>.";

    private async Task<string> TopAsync(string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return BadWindow(window);

        // Split by direction. A single total hides the thing that matters: the
        // ledger charges input and output identically, and a consumer at 40:1
        // is paying almost entirely for context it re-sent, most of which the
        // radix cache served for free.
        var tokensT = PromAsync($"sum by (consumer) (increase(gateway_tokens_total[{window}]))", ct, "consumer");
        var inT     = PromAsync($"sum by (consumer) (increase(gateway_tokens_total{{direction=\"input\"}}[{window}]))", ct, "consumer");
        var outT    = PromAsync($"sum by (consumer) (increase(gateway_tokens_total{{direction=\"output\"}}[{window}]))", ct, "consumer");
        var reqsT   = PromAsync($"sum by (consumer) (increase(gateway_requests_total[{window}]))", ct, "consumer");
        var errsT   = PromAsync(
            $"sum by (consumer) (increase(gateway_requests_total{{status_class=~\"4xx|5xx\"}}[{window}]))", ct, "consumer");
        await Task.WhenAll(tokensT, reqsT, errsT, inT, outT);

        var tokens = tokensT.Result; var reqs = reqsT.Result; var errs = errsT.Result;
        if (reqs.Count == 0) return $"No gateway traffic in the last {window}.";

        var header = $"{"consumer",-14}{"in",9}{"out",8}{"i:o",6}{"req",6}";
        var rows = reqs.OrderByDescending(x => tokens.GetValueOrDefault(x.Key)).Select(x =>
        {
            var inp = inT.Result.GetValueOrDefault(x.Key);
            var outp = outT.Result.GetValueOrDefault(x.Key);
            var ratio = outp > 0 ? inp / outp : 0;
            var err = errs.GetValueOrDefault(x.Key);
            return $"{x.Key,-14}{inp,9:N0}{outp,8:N0}{ratio,6:N1}{x.Value,6:N0}"
                 + (err > 0 ? $"  {err:N0} err" : "");
        });

        return Table($"<b>Top consumers</b> \u2014 last {window}", new[] { header }.Concat(rows))
             + "\n<i>i:o is input per output token. The ledger charges both the same, but input is "
             + "prefilled and mostly cache-served \u2014 a high ratio means the bill is context re-sent, "
             + "not work done.</i>";
    }

    private async Task<string> LatencyAsync(string? name, CancellationToken ct)
    {
        // No rate() window here, deliberately. rate() over an idle window is
        // zero, and histogram_quantile of an all-zero histogram is NaN — which
        // Prometheus omits, so the command would answer "no data" for a
        // consumer who simply has not sent anything in the last few minutes.
        // The cumulative buckets always have an answer, and "p95 since Vector
        // started" is the question an operator actually means here.
        var sel = name is null ? "" : $"{{consumer=\"{name}\"}}";
        var q = (double p) =>
            $"histogram_quantile({p.ToString(CultureInfo.InvariantCulture)}, " +
            $"sum by (consumer,le) (gateway_request_duration_seconds_bucket{sel}))";

        var p50T = PromAsync(q(0.50), ct, "consumer");
        var p95T = PromAsync(q(0.95), ct, "consumer");
        var p99T = PromAsync(q(0.99), ct, "consumer");
        await Task.WhenAll(p50T, p95T, p99T);

        var p95 = p95T.Result;
        if (p95.Count == 0)
            return name is null
                ? "No latency data yet. The access-log pipeline records it from the first request after Vector starts."
                : $"No latency data for <b>{Esc(name)}</b>.";

        var rows = p95.OrderByDescending(x => x.Value).Select(x =>
            $"{x.Key,-16}{p50T.Result.GetValueOrDefault(x.Key),8:N3}{x.Value,9:N3}{p99T.Result.GetValueOrDefault(x.Key),9:N3}");

        var header = $"{"consumer",-16}{"p50",8}{"p95",9}{"p99",9}";
        var body = Table("<b>Latency</b>", new[] { header }.Concat(rows))
             + "\n<i>Seconds, whole request as Envoy saw it. Cumulative since Vector started.</i>"
             + "\n<i>Bucketed, so approximate at low request counts \u2014 the exact figure is in the "
             + "fact table.</i>";

        return body + await EngineLatencyAsync(sel, ct);
    }

    // The engine's own view of the same consumers, which the gateway cannot
    // produce. TTFT differs from the gateway figure by the gateway filter
    // chain, the router and two network hops; inter-token latency has no
    // gateway equivalent at all, because Envoy sees a stream open and a stream
    // close and nothing in between.
    //
    // Empty until BOTH replicas run with --tokenizer-metrics-allowed-custom-labels
    // and Higress injects x-custom-labels. Silent when empty rather than
    // apologetic: /p95 is a gateway command first, and a missing engine block
    // is the normal state during a rollout.
    private async Task<string> EngineLatencyAsync(string sel, CancellationToken ct)
    {
        // consumer!="" drops the health probes and anything that reached a
        // replica without passing the gateway.
        var s = sel.Length == 0 ? "{consumer!=\"\"}" : sel;
        var q = (string metric) =>
            $"histogram_quantile(0.95, sum by (consumer,le) (sglang:{metric}_bucket{s}))";

        var ttftT = PromAsync(q("time_to_first_token_seconds"), ct, "consumer");
        var itlT = PromAsync(q("inter_token_latency_seconds"), ct, "consumer");
        var e2eT = PromAsync(q("e2e_request_latency_seconds"), ct, "consumer");
        await Task.WhenAll(ttftT, itlT, e2eT);

        if (ttftT.Result.Count == 0) return "";

        var rows = ttftT.Result.OrderByDescending(x => x.Value).Select(x =>
            $"{x.Key,-16}{x.Value,8:N3}{itlT.Result.GetValueOrDefault(x.Key),9:N3}{e2eT.Result.GetValueOrDefault(x.Key),9:N2}");

        var header = $"{"consumer",-16}{"ttft",8}{"itl",9}{"e2e",9}";
        return "\n\n" + Table("<b>Engine-side p95</b>", new[] { header }.Concat(rows))
             + "\n<i>Seconds, measured inside SGLang: queue wait plus prefill for ttft, "
             + "gap between output tokens for itl. Excludes gateway, router and network.</i>";
    }

    private async Task<string> ErrorsAsync(string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return BadWindow(window);

        var series = await PromSeriesAsync(
            $"sum by (consumer,status_class) (increase(gateway_requests_total[{window}]))", ct);
        if (series.Count == 0) return $"No gateway traffic in the last {window}.";

        var by = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
        foreach (var (labels, value) in series)
        {
            if (!labels.TryGetValue("consumer", out var c)) continue;
            var cls = labels.GetValueOrDefault("status_class", "other");
            if (!by.TryGetValue(c, out var m)) by[c] = m = new(StringComparer.Ordinal);
            m[cls] = m.GetValueOrDefault(cls) + value;
        }

        var rows = by.OrderByDescending(x => x.Value.GetValueOrDefault("4xx") + x.Value.GetValueOrDefault("5xx"))
                     .Select(x =>
                        $"{x.Key,-16}{x.Value.GetValueOrDefault("2xx"),7:N0}{x.Value.GetValueOrDefault("4xx"),7:N0}{x.Value.GetValueOrDefault("5xx"),7:N0}");

        var header = $"{"consumer",-16}{"2xx",7}{"4xx",7}{"5xx",7}";
        return Table($"<b>Status mix</b> \u2014 last {window}", new[] { header }.Concat(rows))
             + "\n<i>`unauthenticated` is the 401 path: a wrong or missing key, which has no consumer to name.</i>";
    }

    // Reads Alertmanager, not Prometheus, and the difference matters: Prometheus
    // knows what is FIRING, Alertmanager knows what was actually DELIVERED and
    // holds the silences and inhibitions. "Firing but suppressed" is the state
    // most worth being able to see, and only one of the two can show it.
    private async Task<string> AlertsAsync(CancellationToken ct)
    {
        HttpResponseMessage r;
        try
        {
            r = await http.CreateClient("alertmanager")
                .GetAsync("api/v2/alerts?active=true&silenced=true&inhibited=true", ct);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            // Worth saying plainly: if this is unreachable, alerts are firing
            // into nothing again, which is the exact condition this path exists
            // to prevent.
            return "\u26a0\ufe0f <b>Alertmanager unreachable</b>\n\n"
                 + "Alerts are evaluating but nothing is being delivered.\n"
                 + $"<code>{Esc(ex.Message)}</code>";
        }

        using (r)
        {
            if (!r.IsSuccessStatusCode)
                return $"Alertmanager returned HTTP {(int)r.StatusCode}.";

            var node = JsonNode.Parse(await r.Content.ReadAsStringAsync(ct));
            if (node is not JsonArray arr || arr.Count == 0)
                return "<b>Alerts</b>\n\nNothing firing. \u2705";

            var items = new List<(string Sev, string Name, string Who, string Age, bool Suppressed)>();
            foreach (var a in arr)
            {
                var labels = a?["labels"];
                var sev = labels?["severity"]?.GetValue<string>() ?? "unknown";
                var name = labels?["alertname"]?.GetValue<string>() ?? "-";
                var who = labels?["instance"]?.GetValue<string>()
                          ?? labels?["job"]?.GetValue<string>() ?? "-";
                var age = DateTimeOffset.TryParse(
                              a?["startsAt"]?.GetValue<string>(), CultureInfo.InvariantCulture,
                              DateTimeStyles.AdjustToUniversal, out var st)
                          ? Fmt.Age(DateTimeOffset.UtcNow - st) : "-";
                var state = a?["status"]?["state"]?.GetValue<string>();
                items.Add((sev, name, who, age,
                    !string.Equals(state, "active", StringComparison.Ordinal)));
            }

            var sb = new StringBuilder();
            sb.Append("<b>Alerts</b> \u00b7 ").Append(items.Count)
              .Append(items.Count == 1 ? " firing\n" : " firing\n");

            // Severity histogram. A count alone does not show shape; five
            // warnings and one critical is a different morning to the reverse.
            var order = new[] { "critical", "warning", "info" };
            var counts = order
                .Select(sv => (Sev: sv, N: items.Count(i => i.Sev == sv)))
                .Where(x => x.N > 0).ToList();
            var other = items.Count(i => !order.Contains(i.Sev));
            if (other > 0) counts.Add(("other", other));

            if (counts.Count > 0)
            {
                var max = counts.Max(c => c.N);
                sb.Append("\n<pre>");
                foreach (var c in counts)
                    sb.Append(Fmt.Glyph(c.Sev)).Append(' ')
                      .Append(c.Sev.PadRight(8)).Append(c.N.ToString(CultureInfo.InvariantCulture).PadLeft(3))
                      .Append("  ").Append(Fmt.Bar(c.N, max, 12)).Append('\n');
                sb.Append("</pre>");
            }

            foreach (var g in items.GroupBy(i => i.Name)
                                   .OrderBy(g => Array.IndexOf(order, g.First().Sev)))
            {
                sb.Append('\n').Append(Fmt.Glyph(g.First().Sev)).Append(" <b>")
                  .Append(Esc(g.Key)).Append("</b>\n");
                foreach (var i in g)
                    sb.Append("   <code>").Append(Esc(i.Who)).Append("</code> \u00b7 ")
                      .Append(Esc(i.Age))
                      .Append(i.Suppressed ? " \u00b7 <i>suppressed</i>" : "").Append('\n');
            }
            return sb.ToString();
        }
    }

    // PromAsync above returns a map keyed by ai_consumer, which is the right
    // shape for per-consumer tables and the wrong one for "how many targets are
    // up". This returns the first sample's value, or null when the query
    // returned nothing, failed, or produced NaN — histogram_quantile over an
    // idle window does exactly that, and rendering "NaN" to an operator is
    // worse than rendering a dash.
    private async Task<double?> PromScalarAsync(string query, CancellationToken ct)
    {
        try
        {
            var url = $"api/v1/query?query={Uri.EscapeDataString(query)}";
            using var r = await http.CreateClient("prometheus").GetAsync(url, ct);
            if (!r.IsSuccessStatusCode) return null;

            var node = JsonNode.Parse(await r.Content.ReadAsStringAsync(ct));
            if (node?["data"]?["result"] is not JsonArray arr || arr.Count == 0) return null;

            var raw = arr[0]?["value"] is JsonArray v && v.Count > 1 ? v[1]?.GetValue<string>() : null;
            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
                   && !double.IsNaN(d) && !double.IsInfinity(d)
                ? d : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            return null;
        }
    }

    // `label` defaults to ai_consumer, which is what the Higress ai-statistics
    // counters carry. The Vector-derived aggregates use plain `consumer`, so
    // anything reading those has to say so — the two metric families name the
    // same thing differently and silently returning an empty map would look
    // like "no usage" rather than "wrong label".
    private async Task<Dictionary<string, double>> PromAsync(
        string query, CancellationToken ct, string label = "ai_consumer")
    {
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (labels, value) in await PromSeriesAsync(query, ct))
            if (labels.TryGetValue(label, out var key))
                result[key] = value;
        return result;
    }

    // The general form: every label of every sample. Needed wherever a result
    // is keyed by more than one dimension, such as consumer x status_class.
    private async Task<List<(Dictionary<string, string> Labels, double Value)>> PromSeriesAsync(
        string query, CancellationToken ct)
    {
        var result = new List<(Dictionary<string, string>, double)>();
        try
        {
            var url = $"api/v1/query?query={Uri.EscapeDataString(query)}";
            using var r = await http.CreateClient("prometheus").GetAsync(url, ct);
            if (!r.IsSuccessStatusCode) return result;

            var node = JsonNode.Parse(await r.Content.ReadAsStringAsync(ct));
            if (node?["data"]?["result"] is not JsonArray arr) return result;

            foreach (var item in arr)
            {
                var raw = item?["value"] is JsonArray v && v.Count > 1 ? v[1]?.GetValue<string>() : null;
                if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
                    || double.IsNaN(d)) continue;

                var labels = new Dictionary<string, string>(StringComparer.Ordinal);
                if (item?["metric"] is JsonObject mo)
                    foreach (var kv in mo)
                        if (kv.Value is not null) labels[kv.Key] = kv.Value.GetValue<string>();
                result.Add((labels, d));
            }
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            // Caller renders an empty result as "no data", which is the honest
            // answer when Prometheus is unreachable.
        }
        return result;
    }

    // Moved to the Telegram singleton when alert delivery arrived: two
    // independent producers now send messages, and the 4096-character chunking
    // rule must not exist in two places that can drift apart.
    private Task SendAsync(long chatId, Reply reply, CancellationToken ct) =>
        tg.SendAsync(chatId, reply, ct);

    // Show "typing…" only if the answer is actually going to be late.
    //
    // This used to be an awaited call in front of every command, and it was the
    // single largest source of latency in the bot: a round trip to Telegram
    // costs ~100ms warm and ~300ms cold, while most commands finish their real
    // work in under 10ms. So the indicator announcing the wait *was* the wait,
    // and it doubled the time to a reply for every fast command.
    //
    // Now the work starts first and the indicator is sent only if the work is
    // still running after TypingAfter — and never awaited, because nothing about
    // the reply depends on it. Fast commands make no extra call at all; slow
    // ones (/newkey, which writes the apiserver, seeds the ledger and re-reads)
    // still get the feedback that stops an operator retyping the command.
    private static readonly TimeSpan TypingAfter = TimeSpan.FromMilliseconds(350);

    private async Task<Reply> DispatchWithTypingAsync(long chatId, long userId, string text, CancellationToken ct)
    {
        var work = DispatchAsync(userId, text, ct);
        using var settled = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var late = Task.Delay(TypingAfter, settled.Token);

        if (await Task.WhenAny(work, late) != work)
            _ = TypingAsync(chatId, ct);
        settled.Cancel();          // releases the timer; the delay is never awaited

        return await work;
    }

    private async Task TypingAsync(long chatId, CancellationToken ct)
    {
        try
        {
            using var content = new StringContent(
                JsonSerializer.Serialize(new ChatAction(chatId, "typing"), BotJson.Default.ChatAction), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var _ = await http.CreateClient("telegram").PostAsync("sendChatAction", content, ct);
        }
        catch (Exception ex) { log.LogDebug(ex, "sendChatAction failed"); }
    }

    // Clears the button's loading spinner. Without it the client shows a
    // progress ring on the tapped button for several seconds.
    private async Task AnswerCallbackAsync(string? id, CancellationToken ct)
    {
        if (id is null) return;
        try
        {
            using var content = new StringContent(
                JsonSerializer.Serialize(new AnswerCallbackQuery(id), BotJson.Default.AnswerCallbackQuery), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var _ = await http.CreateClient("telegram").PostAsync("answerCallbackQuery", content, ct);
        }
        catch (Exception ex) { log.LogDebug(ex, "answerCallbackQuery failed"); }
    }

    private async Task PublishCommandMenuAsync(CancellationToken ct)
    {
        // Ordered by how often they are reached for, not alphabetically:
        // Telegram shows this list verbatim.
        // Rendered from the one command table, in its order — see Commands.
        var menu = Array.ConvertAll(
            Array.FindAll(Commands, c => c.MenuText.Length > 0),
            c => new BotCommand(c.Name, c.MenuText));
        try
        {
            using var content = new StringContent(
                JsonSerializer.Serialize(new SetMyCommands(menu), BotJson.Default.SetMyCommands), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r = await http.CreateClient("telegram").PostAsync("setMyCommands", content, ct);
            log.LogInformation("published command menu: HTTP {Code}", (int)r.StatusCode);
        }
        catch (Exception ex) { log.LogWarning(ex, "could not publish command menu"); }
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

    internal static bool IsValidName(string s) =>
        s.Length is > 0 and <= 32 && s.All(c => char.IsAsciiLetterLower(c) || char.IsAsciiDigit(c) || c is '-' or '_');

    private static bool TryParseTokens(string s, out long v) =>
        long.TryParse(s.Replace("_", "").Replace(",", ""), NumberStyles.Integer, CultureInfo.InvariantCulture, out v) && v >= 0;

    private static string Head(string s) => s.Length <= 40 ? s : s[..40] + "\u2026";

    // Delegates to Fmt so alert rendering and command rendering escape
    // identically. Kept as a local name because it is used at ~40 call sites.
    private static string Esc(string s) => Fmt.Esc(s);

    // One shape for every usage error: what it takes, then a real example.
    // "Usage: /topup <name> <tokens>" alone still leaves people guessing
    // whether tokens are thousands or millions.
    private static string Usage(string form, string example) =>
        $"<b>Usage</b>\n<code>{form}</code>\n\n<b>Example</b>\n<code>{Esc(example)}</code>";

    // Telegram renders message text in a PROPORTIONAL font, so space-padded
    // columns do not line up — they look ragged on every client. A <pre> block
    // is the only way to get a real table, and it also gets tap-to-copy.
    private static string Table(string header, IEnumerable<string> rows) =>
        $"{header}\n<pre>" + string.Join("\n", rows.Select(Esc)) + "</pre>";

    // Rejection sampling, not a plain `% 62` — this generates API credentials.
    //
    // A uniform byte is 0..255 and 256 is not a multiple of 62, so the modulo on
    // its own made the first eight letters of the alphabet ~1.6x likelier than
    // the other 54. That is a small but real loss of entropy in a secret.
    // Discarding the 248..255 tail and drawing again costs nothing measurable
    // and makes the distribution exact.
    //
    // stackalloc is available here only because this method is synchronous; it
    // is not an option through most of this file, since a Span cannot live
    // across an await. The length bound is what makes it safe on the stack.
    internal static string Base62(int len)
    {
        const string alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        const int unbiased = 256 - (256 % 62);              // 248
        ArgumentOutOfRangeException.ThrowIfLessThan(len, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(len, 128);

        Span<char> chars = stackalloc char[len];
        Span<byte> draw = stackalloc byte[128];
        var produced = 0;
        while (produced < len)
        {
            RandomNumberGenerator.Fill(draw);
            foreach (var b in draw)
            {
                if (b >= unbiased) continue;                // biased tail: redraw
                chars[produced++] = alphabet[b % alphabet.Length];
                if (produced == len) break;
            }
        }
        return new string(chars);
    }
}

// ===========================================================================
// Policy — what a consumer is allowed: tier defaults, per-consumer overrides.
//
// A tier is a named set of defaults for EVERY setting, and a consumer stores
// only the values set on it by hand (Ledger, chat_policy:<name>). The effective
// value is the override when there is one and the tier's otherwise. So moving
// a tier default moves every consumer still following it, and a value set by
// hand survives both that and a change of tier.
//
// ENFORCEMENT IS PER SETTING, and each one carries its own status:
//   balance     ai-quota, on every request
//   daily, tpm  ai-token-ratelimit, rules rendered by LimiterSync
//   refill      RefillJob, at the UTC period boundary
//   quota       seeds a new key, and is what a refill sets the balance to
//   max_tokens  NOT enforceable per key: a WasmPlugin matchRule selects by
//               route, domain or service, never consumer (checked in the v2.2.4
//               proto), so request-validation holds one global ceiling. Kept as
//               a recorded value, deliberately (2026-09-13).
//
// Sizing, measured from gateway.requests on 2026-09-13 rather than guessed. An
// agent request is 74k tokens at the median and 134k at p95 (max 169k); an
// active minute is 122k median, 418k p95, 1.04M max; one consumer used 32.6M in
// a day. The limiter refuses only once a counter is already over, so a
// per-minute limit below one maximum-size request means one request a minute —
// hence nothing under 200k. Quotas are sized to a month of that traffic. The
// first draft (10M a month, 60k a minute) would have throttled every agent.
// ===========================================================================
enum RefillMode { Manual, Daily, Weekly, Monthly }

// 0 means "no limit" for Daily and Tpm, and "the gateway's global ceiling" for MaxTokens.
sealed record TierDef(string Name, string For, long Quota, RefillMode Refill,
                      long Daily, long Tpm, long MaxTokens);

sealed record PolicyField(string Key, string Label, string Meaning, bool Enforced, string Status);

readonly record struct ResolvedSetting(PolicyField Field, string? Value, string Source);

static class Policy
{
    // MENU ORDER. Rendered in this order by /tiers and /admin/tiers.
    public static readonly TierDef[] All =
    [
        new("trial",   "evaluation, unvetted third parties",   1_000_000, RefillMode.Manual,     500_000,   200_000,  2_048),
        new("team",    "internal humans via OpenCode",       100_000_000, RefillMode.Monthly, 20_000_000,   600_000, 32_768),
        new("service", "production integrations",           300_000_000, RefillMode.Monthly, 30_000_000, 1_000_000, 16_384),
        new("batch",   "offline, latency-tolerant",          500_000_000, RefillMode.Monthly, 50_000_000,   300_000, 70_000),
        new("admin",   "management only, plus the bot's own reports", 0, RefillMode.Manual,            0,         0,      0),
    ];

    public static readonly Dictionary<string, TierDef> Tiers =
        All.ToDictionary(t => t.Name, StringComparer.Ordinal);

    public static readonly PolicyField[] Fields =
    [
        new("quota", "quota", "tokens a refill sets the balance to; also a new key's starting balance",
            true, "seeds a new key and is applied by each automatic refill; changing it never moves a live balance by itself"),
        new("refill", "refill", "manual, or an automatic reset of the balance to quota at 00:00 UTC each day, each Monday, or on the 1st",
            true, "enforced by the refill job; unused tokens do not carry over"),
        new("daily", "daily", "input+output tokens per 24h window, which starts at the key's first request after the last one expired; 0 = no limit",
            true, "enforced at the gateway (ai-token-ratelimit); one request can overshoot by its own size"),
        new("tpm", "tokens/min", "input+output tokens per 60s window from the first request in it; 0 = no limit",
            true, "enforced at the gateway (ai-token-ratelimit); below ~170k it means one large request a minute"),
        new("max_tokens", "max_tokens", "largest max_tokens one request may ask for; 0 = the gateway ceiling",
            false, "recorded only; the gateway cannot vary it per key and enforces one global ceiling"),
    ];

    public static PolicyField? Field(string raw) => raw.ToLowerInvariant() switch
    {
        "quota" => Fields[0],
        "refill" => Fields[1],
        "daily" or "day" or "daily_limit" => Fields[2],
        "tpm" or "tokens_per_minute" => Fields[3],
        "max_tokens" or "maxtokens" or "max" => Fields[4],
        _ => null
    };

    public static bool IsDefaultWord(string raw) =>
        raw.ToLowerInvariant() is "default" or "tier" or "reset";

    public static string RefillName(RefillMode m) => m switch
    {
        RefillMode.Daily => "daily",
        RefillMode.Weekly => "weekly",
        RefillMode.Monthly => "monthly",
        _ => "manual"
    };

    public static string TierValue(TierDef t, PolicyField f) => f.Key switch
    {
        "quota" => t.Quota.ToString(CultureInfo.InvariantCulture),
        "refill" => RefillName(t.Refill),
        "daily" => t.Daily.ToString(CultureInfo.InvariantCulture),
        "tpm" => t.Tpm.ToString(CultureInfo.InvariantCulture),
        _ => t.MaxTokens.ToString(CultureInfo.InvariantCulture),
    };

    // Source is "set" for a hand-set value, "tier" when it follows the tier, and
    // "none" for an unassigned consumer with nothing set — a real state that
    // must not be dressed up as a default.
    public static ResolvedSetting[] Resolve(string? tier, IReadOnlyDictionary<string, string> overrides)
    {
        var def = tier is not null && Tiers.TryGetValue(tier, out var t) ? t : null;
        var result = new ResolvedSetting[Fields.Length];
        for (var i = 0; i < Fields.Length; i++)
        {
            var f = Fields[i];
            result[i] = overrides.TryGetValue(f.Key, out var v) ? new(f, v, "set")
                      : def is not null ? new(f, TierValue(def, f), "tier")
                      : new(f, null, "none");
        }
        return result;
    }

    // Stored form is canonical: plain integers and lowercase refill names, so
    // the enforcement side can read the hash without re-parsing "2M".
    public static bool TryNormalise(PolicyField f, string raw, int gatewayMax, out string? stored, out string error)
    {
        stored = null; error = "";
        var s = raw.Trim().ToLowerInvariant();

        if (f.Key == "refill")
        {
            if (s is "manual" or "daily" or "weekly" or "monthly") { stored = s; return true; }
            error = "refill is one of: manual, daily, weekly, monthly.";
            return false;
        }

        if (s is "unlimited" or "none" or "off" && f.Key is "daily" or "tpm" or "max_tokens")
            s = "0";
        if (!TryParseAmount(s, out var n))
        {
            error = $"'{raw}' is not a number. Use 2000000, 2_000_000, 2M or 500k.";
            return false;
        }

        var (min, max) = f.Key switch
        {
            "quota" => (0L, 10_000_000_000L),
            "daily" => (0L, 10_000_000_000L),
            "tpm" => (0L, 100_000_000L),
            _ => (0L, (long)gatewayMax),
        };
        if (n < min || n > max)
        {
            error = f.Key == "max_tokens"
                ? $"max_tokens must be 0..{gatewayMax:N0} — the gateway refuses anything above {gatewayMax:N0} for everyone."
                : $"{f.Key} must be between {min:N0} and {max:N0}.";
            return false;
        }
        stored = n.ToString(CultureInfo.InvariantCulture);
        return true;
    }

    // 2000000, 2_000_000, 2,000,000, 2M, 1.5m, 500k. Decimal suffixes only on
    // purpose: "2M" meaning 2,097,152 would be a surprise in a token budget.
    private static bool TryParseAmount(string s, out long n)
    {
        n = 0;
        s = s.Replace("_", "").Replace(",", "");
        long mult = 1;
        if (s.EndsWith('k')) { mult = 1_000; s = s[..^1]; }
        else if (s.EndsWith('m')) { mult = 1_000_000; s = s[..^1]; }
        else if (s.EndsWith('b')) { mult = 1_000_000_000; s = s[..^1]; }
        if (!decimal.TryParse(s, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var d)) return false;
        var v = d * mult;
        if (v != decimal.Truncate(v) || v < 0 || v > long.MaxValue) return false;
        n = (long)v;
        return true;
    }

    public static string Show(PolicyField f, string stored, int gatewayMax)
    {
        if (f.Key == "refill") return stored;
        if (!long.TryParse(stored, CultureInfo.InvariantCulture, out var n)) return stored;
        return (f.Key, n) switch
        {
            ("max_tokens", 0) => "gateway max",
            ("daily" or "tpm", 0) => "unlimited",
            _ => n.ToString("N0", CultureInfo.InvariantCulture),
        };
    }

    // 100K, 10M, 1.5M. For tables that have to fit a phone.
    public static string Compact(long n) => n switch
    {
        >= 1_000_000_000 => Trim(n / 1_000_000_000m) + "B",
        >= 1_000_000 => Trim(n / 1_000_000m) + "M",
        >= 10_000 => Trim(n / 1_000m) + "K",
        _ => n.ToString(CultureInfo.InvariantCulture),
    };

    private static string Trim(decimal d) =>
        Math.Round(d, 1).ToString("0.#", CultureInfo.InvariantCulture);
}

// ===========================================================================
// QuotaApi — ai-quota's admin endpoint, shared by commands and the refill job.
// FORM-ENCODED: ai-quota answers 403 to JSON, which reads like an auth failure.
// ===========================================================================
static class QuotaApi
{
    public static async Task SetAsync(IHttpClientFactory http, string name, long value, CancellationToken ct)
    {
        using var body = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", name),
            new KeyValuePair<string, string>("quota", value.ToString(CultureInfo.InvariantCulture))
        ]);
        using var r = await http.CreateClient("gateway").PostAsync("v1/chat/completions/quota/refresh", body, ct);
        r.EnsureSuccessStatusCode();
    }
}

// ===========================================================================
// LimiterSync — renders per-consumer daily/tpm limits into ai-token-ratelimit.
//
// The bot is the ONLY writer of the rules. ../higress-standalone ships the
// object as a disabled shell (config/wasmplugins/ai-token-ratelimit.yaml, which
// documents the plugin's semantics as read from source), and apply.sh reinstalls
// that shell on every run. So this syncs on every policy or key change and also
// once a minute, which is what puts the rules back after an apply.
//
// It compares a FINGERPRINT of the rules rather than JSON text, and PUTs only
// on a real difference: the apiserver is free to reorder keys, and a text diff
// would rewrite the object — and reconfigure the plugin in every gateway worker
// — once a minute for nothing.
//
// LIMITER_SCOPE (comma-separated names) restricts the rules to those consumers.
// It exists for rollout: prove the limiter on a test key before it touches a
// real one. Unset means every consumer.
// ===========================================================================
sealed class LimiterSync(IHttpClientFactory http, KeyStore keys, Ledger ledger, ILogger<LimiterSync> log)
{
    private const string Base = "apis/extensions.higress.io/v1alpha1/namespaces/higress-system/wasmplugins/";
    public const string RuleName = "consumer-limits";

    private static readonly HashSet<string>? Scope =
        Environment.GetEnvironmentVariable("LIMITER_SCOPE") is { Length: > 0 } s
            ? s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToHashSet(StringComparer.Ordinal)
            : null;

    public static bool InScope(string name) => Scope is null || Scope.Contains(name);

    // The limiter's own Redis key for a consumer's window (see main.go,
    // AiTokenRateLimitFormat). The {…} is a cluster hash tag, kept verbatim.
    public static string CounterKey(string name, long window) =>
        $"higress-token-ratelimit:{{{RuleName}}}:limit_by_consumer:{window}:x-mse-consumer:{name}";

    // Static so Ledger and KeyStore can signal a change without a dependency
    // cycle. A kick while one is already pending coalesces into it.
    private static readonly SemaphoreSlim KickSignal = new(0, 1);
    public static void Kick()
    {
        try { if (KickSignal.CurrentCount == 0) KickSignal.Release(); }
        catch (SemaphoreFullException) { }
    }
    public static Task<bool> WaitKickAsync(TimeSpan timeout, CancellationToken ct) => KickSignal.WaitAsync(timeout, ct);

    private readonly SemaphoreSlim _lock = new(1, 1);
    private DateTimeOffset? _lastOk;
    private string? _lastError;
    private int _limited;

    public string StatusLine(string name)
    {
        if (!InScope(name))
            return "⚠️ <b>Not in the limiter's rollout scope</b> — daily and tpm are not applied to this key yet.";
        if (_lastError is { } e)
            return $"⚠️ <b>Limiter sync failing:</b> {Fmt.Esc(e)}. Limits last applied "
                 + (_lastOk is { } ok ? $"{Fmt.Duration((long)(DateTimeOffset.UtcNow - ok).TotalSeconds)} ago." : "never.");
        return _lastOk is { } t
            ? $"<i>Gateway limits in force for {_limited} key(s), checked {Fmt.Duration((long)(DateTimeOffset.UtcNow - t).TotalSeconds)} ago.</i>"
            : "<i>Gateway limits not synced yet since the bot started.</i>";
    }

    public string Summary() =>
        _lastError is { } e ? $"FAILED ({e})"
        : _lastOk is { } t ? $"{_limited} key(s) limited, {Fmt.Duration((long)(DateTimeOffset.UtcNow - t).TotalSeconds)} ago"
        : "not synced yet";

    public async Task SyncAsync(CancellationToken ct)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var consumersT = keys.ReadConsumersAsync(ct);
            var tiersT = ledger.TiersAsync(ct);
            var overridesT = ledger.AllOverridesAsync(ct);
            await Task.WhenAll(consumersT, tiersT, overridesT);

            var daily = new SortedDictionary<string, long>(StringComparer.Ordinal);
            var minute = new SortedDictionary<string, long>(StringComparer.Ordinal);
            foreach (var name in consumersT.Result.Keys)
            {
                // A name outside the alphabet cannot be a limit key safely, and
                // /newkey cannot produce one; skip rather than render it.
                if (!Worker.IsValidName(name) || !InScope(name)) continue;
                var r = Policy.Resolve(tiersT.Result.GetValueOrDefault(name),
                                       overridesT.Result.GetValueOrDefault(name) ?? new Dictionary<string, string>());
                if (long.TryParse(r[2].Value, CultureInfo.InvariantCulture, out var d) && d > 0) daily[name] = d;
                if (long.TryParse(r[3].Value, CultureInfo.InvariantCulture, out var m) && m > 0) minute[name] = m;
            }

            var client = http.CreateClient("apiserver");

            // Redis settings copied from the live ai-quota object, so the
            // limiter counts in the same ledger Redis without a second copy of
            // its address to keep in step.
            JsonNode redis = new JsonObject { ["service_name"] = "quota-redis.dns", ["service_port"] = 6379, ["timeout"] = 1000 };
            using (var q = await client.GetAsync(Base + "ai-quota", ct))
                if (q.IsSuccessStatusCode
                    && JsonNode.Parse(await q.Content.ReadAsStringAsync(ct))?["spec"]?["matchRules"]?[0]?["config"]?["redis"] is JsonObject live)
                    redis = live.DeepClone();

            using var get = await client.GetAsync(Base + "ai-token-ratelimit", ct);
            if (get.StatusCode == HttpStatusCode.NotFound)
                throw new InvalidOperationException("ai-token-ratelimit is not installed — run higress-standalone/apply.sh");
            get.EnsureSuccessStatusCode();
            var obj = JsonNode.Parse(await get.Content.ReadAsStringAsync(ct))!.AsObject();
            if (obj["spec"]?["matchRules"] is not JsonArray rules || rules.Count == 0)
                throw new InvalidOperationException("ai-token-ratelimit has no matchRules");

            var enabled = daily.Count > 0 || minute.Count > 0;
            var config = Render(daily, minute, redis);
            var want = Fingerprint(config, !enabled);

            var changed = false;
            foreach (var rule in rules)
            {
                if (rule is not JsonObject ro) continue;
                var disabled = ro["configDisable"]?.GetValueKind() == JsonValueKind.True;
                if (Fingerprint(ro["config"], disabled) == want) continue;
                ro["config"] = config.DeepClone();
                ro["configDisable"] = !enabled;
                changed = true;
            }

            if (changed)
            {
                using var body = new StringContent(obj.ToJsonString(), Encoding.UTF8);
                body.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                using var put = await client.PutAsync(Base + "ai-token-ratelimit", body, ct);
                if (!put.IsSuccessStatusCode)
                {
                    var err = await put.Content.ReadAsStringAsync(ct);
                    throw new InvalidOperationException($"PUT HTTP {(int)put.StatusCode}: {(err.Length > 200 ? err[..200] : err)}");
                }
                log.LogInformation("limiter rules pushed: {Daily} daily, {Minute} per-minute, enabled={Enabled}, scope={Scope}",
                    daily.Count, minute.Count, enabled, Scope is null ? "all" : string.Join(",", Scope));
            }

            _limited = daily.Keys.Union(minute.Keys).Count();
            _lastOk = DateTimeOffset.UtcNow;
            _lastError = null;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (_lastError != ex.Message) log.LogError(ex, "limiter sync failed");
            _lastError = ex.Message;
        }
        finally { _lock.Release(); }
    }

    private static JsonObject Render(SortedDictionary<string, long> daily, SortedDictionary<string, long> minute, JsonNode redis)
    {
        static JsonObject Item(SortedDictionary<string, long> limits, string window)
        {
            var keysArr = new JsonArray();
            foreach (var (name, n) in limits)
                keysArr.Add((JsonNode)new JsonObject { ["key"] = name, [window] = n });
            return new JsonObject { ["limit_by_consumer"] = "", ["limit_keys"] = keysArr };
        }

        // Two rule_items, because within one item the first matching key wins
        // and a consumer would get only its daily limit. Daily first, so a
        // refusal reports the daily window when both are exceeded.
        var items = new JsonArray();
        if (daily.Count > 0) items.Add((JsonNode)Item(daily, "token_per_day"));
        if (minute.Count > 0) items.Add((JsonNode)Item(minute, "token_per_minute"));
        // The plugin rejects empty rule_items even on a disabled rule, so the
        // disabled form keeps a placeholder no consumer name can match.
        if (items.Count == 0)
            items.Add((JsonNode)Item(new SortedDictionary<string, long>(StringComparer.Ordinal) { ["."] = 1 }, "token_per_day"));

        return new JsonObject
        {
            ["rule_name"] = RuleName,
            ["rule_items"] = items,
            ["rejected_code"] = 429,
            ["rejected_msg"] = "Token rate limit reached for this API key",
            ["redis"] = redis.DeepClone(),
        };
    }

    // Order-insensitive identity of a rule config: what the plugin would do,
    // not how the JSON happens to be laid out.
    private static readonly string[] Windows = ["token_per_day", "token_per_minute", "token_per_hour", "token_per_second"];

    private static string Fingerprint(JsonNode? config, bool disabled)
    {
        var parts = new List<string> { "disabled=" + disabled };
        if (config is not JsonObject c) return string.Join('|', parts);
        parts.Add("rule=" + c["rule_name"]);
        parts.Add("code=" + c["rejected_code"]);
        parts.Add("msg=" + c["rejected_msg"]);
        parts.Add("redis=" + c["redis"]?["service_name"] + ":" + c["redis"]?["service_port"]);
        if (c["rule_items"] is JsonArray items)
            foreach (var item in items)
                if (item?["limit_keys"] is JsonArray lk)
                    foreach (var k in lk)
                        foreach (var w in Windows)
                            if (k?[w] is { } v) parts.Add($"{w}:{k["key"]}={v}");
        parts.Sort(StringComparer.Ordinal);
        return string.Join('|', parts);
    }
}

// ===========================================================================
// RefillJob — resets a balance to its quota at each period boundary (UTC).
//
// Idempotent through a marker per consumer, "<mode>:<period>", written AFTER a
// successful reset: a restart, a second pass in the same minute, or a crash
// between reset and marker at worst repeats a SET to the same value.
//
// Arming, not refilling, on first sight. A consumer with no marker — every
// consumer the day this shipped, or one whose refill was just switched on —
// gets the current period recorded and keeps its balance. The first reset is at
// the NEXT boundary. The same holds when the mode changes (monthly -> weekly):
// the marker's mode no longer matches, so it re-arms instead of resetting at
// once. Nobody's balance moves because a setting was touched.
//
// Refill REPLACES the balance: unused tokens do not carry over, and an overdraft
// is cleared. Every reset is audited and announced to the alert chats.
// ===========================================================================
static class RefillJob
{
    public static string Period(string mode, DateTimeOffset now) => mode switch
    {
        "daily" => $"daily:{now:yyyy-MM-dd}",
        "weekly" => $"weekly:{ISOWeek.GetYear(now.UtcDateTime)}-W{ISOWeek.GetWeekOfYear(now.UtcDateTime):00}",
        "monthly" => $"monthly:{now:yyyy-MM}",
        _ => ""
    };

    public static DateTimeOffset Next(string mode, DateTimeOffset now)
    {
        now = now.ToUniversalTime();
        var day = new DateTimeOffset(now.UtcDateTime.Date, TimeSpan.Zero);
        switch (mode)
        {
            case "daily": return day.AddDays(1);
            case "weekly":
                // ISO weeks start on Monday; on a Monday the next one is a week away.
                var ahead = ((int)DayOfWeek.Monday - (int)day.DayOfWeek + 7) % 7;
                return day.AddDays(ahead == 0 ? 7 : ahead);
            case "monthly": return new DateTimeOffset(now.Year, now.Month, 1, 0, 0, 0, TimeSpan.Zero).AddMonths(1);
            default: return DateTimeOffset.MaxValue;
        }
    }
}

sealed class EnforcementWorker(
    BotConfig cfg, LimiterSync limiter, KeyStore keys, Ledger ledger,
    IHttpClientFactory http, Telegram tg, ILogger<EnforcementWorker> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        // Let the process settle and the first Telegram traffic through first.
        try { await Task.Delay(TimeSpan.FromSeconds(5), ct); } catch (OperationCanceledException) { return; }
        while (!ct.IsCancellationRequested)
        {
            await limiter.SyncAsync(ct);
            try { await RefillAsync(ct); }
            catch (Exception ex) when (ex is not OperationCanceledException) { log.LogError(ex, "refill pass failed"); }
            try { await LimiterSync.WaitKickAsync(TimeSpan.FromSeconds(60), ct); }
            catch (OperationCanceledException) { return; }
        }
    }

    private async Task RefillAsync(CancellationToken ct)
    {
        var consumersT = keys.ReadConsumersAsync(ct);
        var tiersT = ledger.TiersAsync(ct);
        var overridesT = ledger.AllOverridesAsync(ct);
        await Task.WhenAll(consumersT, tiersT, overridesT);
        var now = DateTimeOffset.UtcNow;

        foreach (var name in consumersT.Result.Keys)
        {
            if (!Worker.IsValidName(name)) continue;
            var r = Policy.Resolve(tiersT.Result.GetValueOrDefault(name),
                                   overridesT.Result.GetValueOrDefault(name) ?? new Dictionary<string, string>());
            var mode = r[1].Value ?? "manual";
            if (mode == "manual" || !long.TryParse(r[0].Value, CultureInfo.InvariantCulture, out var quota) || quota <= 0)
                continue;

            var period = RefillJob.Period(mode, now);
            var marker = await ledger.RefillMarkerAsync(name, ct);
            if (marker == period) continue;

            if (marker is null || !marker.StartsWith(mode + ":", StringComparison.Ordinal))
            {
                await ledger.SetRefillMarkerAsync(name, period, ct);
                await Audit($"refill-armed name={name} mode={mode} period={period} next={RefillJob.Next(mode, now):O}", ct);
                log.LogInformation("refill armed for {Name}: {Mode}, first reset {Next:O}", name, mode, RefillJob.Next(mode, now));
                continue;
            }

            var before = (await ledger.ListAsync(ct)).TryGetValue(name, out var b) ? b : (long?)null;
            await QuotaApi.SetAsync(http, name, quota, ct);
            await ledger.SetRefillMarkerAsync(name, period, ct);
            await Audit($"refill name={name} mode={mode} period={period} from={before?.ToString(CultureInfo.InvariantCulture) ?? "none"} to={quota}", ct);
            log.LogWarning("refilled {Name} to {Quota} ({Mode} {Period}), was {Before}", name, quota, mode, period, before);

            var text = $"\U0001f504 <b>{Fmt.Esc(name)}</b> refilled to <code>{quota:N0}</code> tokens "
                     + $"(was {before?.ToString("N0", CultureInfo.InvariantCulture) ?? "unset"}) — {mode} refill.";
            foreach (var chat in cfg.AlertChatIds)
                try { await tg.SendAsync(chat, new Reply(text), ct); }
                catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException) { log.LogWarning(ex, "refill notice not delivered"); }
        }
    }

    private async Task Audit(string line, CancellationToken ct)
    {
        try { await File.AppendAllTextAsync(cfg.AuditPath, $"{DateTimeOffset.UtcNow:O} {line}\n", ct); }
        catch (IOException ex) { log.LogError(ex, "audit write failed: {Line}", line); }
    }
}

// ===========================================================================
// PriceBook — what the same tokens would cost for this model elsewhere.
//
// REFERENCE PRICES, NOT A BILL. Nobody is charged these; they exist so usage
// can be read in money, and so a key's traffic can be compared with buying the
// same model from a public provider. Every surface labels them that way.
//
// OpenRouter publishes prices over a public, unauthenticated API, so those are
// fetched and refreshed daily; a snapshot stands in until the first fetch
// succeeds, and a failed refresh keeps the last good price rather than showing
// nothing. The headline model price is used, which is what openrouter.ai shows
// for the model; the spread across its providers is reported next to it,
// because it is wide (output $2.00-3.20 per M on 2026-09-13).
//
// Alibaba Cloud Model Studio has no price API. Its numbers are copied from
// alibabacloud.com/help/en/model-studio/model-pricing, page dated 2026-09-12,
// for Qwen3.8-27B (thinking and non-thinking are the same price, no cached-input
// price, no tiering below 1M context). Update them by hand, with the date.
//
// Pricing needs a price per token DIRECTION, which is why these read the
// ai-statistics counters (input and output separately) and never the ledger.
// ===========================================================================
sealed record PriceRef(string Label, decimal InPerM, decimal OutPerM, decimal? CacheReadPerM, string Basis);

sealed record Prices(PriceRef OpenRouter, PriceRef AlibabaSg, PriceRef AlibabaBj,
                     DateTimeOffset? FetchedAt, int Providers, decimal? OutMin, decimal? OutMax);

sealed class PriceBook(IHttpClientFactory http, ILogger<PriceBook> log)
{
    public static readonly string OpenRouterModel =
        Environment.GetEnvironmentVariable("PRICE_OPENROUTER_MODEL") is { Length: > 0 } m ? m : "qwen/qwen3.8-27b";

    private static readonly PriceRef AliSg = new("Alibaba SG", 0.50m, 3.00m, null,
        "Alibaba Cloud Model Studio, International (Singapore), Qwen3.8-27B, as of 2026-09-12");
    private static readonly PriceRef AliBj = new("Alibaba BJ", 0.424m, 1.696m, null,
        "Alibaba Cloud Model Studio, China (Beijing), Qwen3.8-27B, as of 2026-09-12");

    private volatile Prices _current = new(
        new PriceRef("OpenRouter", 0.214m, 2.55m, 0.15m, $"OpenRouter {OpenRouterModel}, snapshot of 2026-09-13 (not yet fetched)"),
        AliSg, AliBj, null, 0, null, null);

    private DateTimeOffset _lastAttempt = DateTimeOffset.MinValue;
    private readonly SemaphoreSlim _refresh = new(1, 1);

    public async Task<Prices> GetAsync(CancellationToken ct)
    {
        var now = DateTimeOffset.UtcNow;
        var stale = _current.FetchedAt is not { } f || now - f > TimeSpan.FromHours(24);
        // A failing refresh is retried at most every 15 minutes, so an
        // OpenRouter outage costs one slow command per quarter hour, not all of them.
        if (!stale || now - _lastAttempt < TimeSpan.FromMinutes(15) || !await _refresh.WaitAsync(0, ct))
            return _current;
        try
        {
            _lastAttempt = now;
            _current = await FetchAsync(ct) ?? _current;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException
                                      or FormatException or KeyNotFoundException or InvalidOperationException)
        {
            log.LogWarning("OpenRouter price refresh failed, keeping {Basis}: {Reason}", _current.OpenRouter.Basis, ex.Message);
        }
        finally { _refresh.Release(); }
        return _current;
    }

    private async Task<Prices?> FetchAsync(CancellationToken ct)
    {
        var client = http.CreateClient("openrouter");

        // The model list is ~0.7 MB. Streamed through JsonDocument rather than
        // materialised as a JsonNode tree, and fetched once a day.
        PriceRef? headline = null;
        await using (var s = await client.GetStreamAsync("models", ct))
        using (var doc = await JsonDocument.ParseAsync(s, cancellationToken: ct))
        {
            foreach (var model in doc.RootElement.GetProperty("data").EnumerateArray())
            {
                if (!model.TryGetProperty("id", out var id) || id.GetString() != OpenRouterModel) continue;
                var p = model.GetProperty("pricing");
                headline = new PriceRef("OpenRouter", PerM(p, "prompt")!.Value, PerM(p, "completion")!.Value,
                    PerM(p, "input_cache_read"),
                    $"OpenRouter {OpenRouterModel}, fetched {DateTimeOffset.UtcNow:yyyy-MM-dd HH:mm} UTC");
                break;
            }
        }
        if (headline is null)
        {
            log.LogWarning("OpenRouter lists no model {Model}; keeping the previous price", OpenRouterModel);
            return null;
        }

        // The provider spread is context, not the price. Losing it is not a
        // reason to discard a good headline.
        int providers = 0; decimal? min = null, max = null;
        try
        {
            await using var s = await client.GetStreamAsync($"models/{OpenRouterModel}/endpoints", ct);
            using var doc = await JsonDocument.ParseAsync(s, cancellationToken: ct);
            foreach (var e in doc.RootElement.GetProperty("data").GetProperty("endpoints").EnumerateArray())
                if (e.TryGetProperty("pricing", out var p) && PerM(p, "completion") is { } o)
                {
                    providers++;
                    min = min is null ? o : Math.Min(min.Value, o);
                    max = max is null ? o : Math.Max(max.Value, o);
                }
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or KeyNotFoundException or InvalidOperationException)
        {
            log.LogInformation("OpenRouter provider spread unavailable: {Reason}", ex.Message);
        }

        log.LogInformation("OpenRouter prices for {Model}: in {In}/M out {Out}/M cached {Cache}/M across {N} providers",
            OpenRouterModel, headline.InPerM, headline.OutPerM, headline.CacheReadPerM, providers);
        return new Prices(headline, AliSg, AliBj, DateTimeOffset.UtcNow, providers, min, max);
    }

    // OpenRouter quotes USD per token as a decimal string ("0.00000255").
    private static decimal? PerM(JsonElement pricing, string field) =>
        pricing.TryGetProperty(field, out var v) && v.ValueKind == JsonValueKind.String
        && decimal.TryParse(v.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var perToken)
            ? decimal.Round(perToken * 1_000_000m, 6)
            : null;

    // Cost of `tin` input and `tout` output tokens at `p`. With a cache-read
    // price and a hit share, the cached part of the input is priced as cached —
    // the engine measures hits per consumer over a window, not per request, so
    // this is an estimate of what a cache-discounting provider would charge.
    public static decimal Cost(PriceRef p, double tin, double tout, double? cacheHit = null)
    {
        static decimal D(double v) => double.IsFinite(v) && v > 0 ? (decimal)v : 0m;
        var input = D(tin);
        var cached = p.CacheReadPerM is { } && cacheHit is { } h && double.IsFinite(h)
            ? input * (decimal)Math.Clamp(h, 0, 1) : 0m;
        return ((input - cached) * p.InPerM + cached * (p.CacheReadPerM ?? 0m) + D(tout) * p.OutPerM) / 1_000_000m;
    }

    public static string Usd(decimal v) => v switch
    {
        0m => "$0",
        < 0.01m => "<$0.01",
        < 100m => "$" + v.ToString("0.00", CultureInfo.InvariantCulture),
        < 10_000m => "$" + v.ToString("N0", CultureInfo.InvariantCulture),
        _ => "$" + (v / 1000m).ToString("0.#", CultureInfo.InvariantCulture) + "K",
    };

    public static string PerMText(PriceRef p) =>
        $"${p.InPerM:0.###} in / ${p.OutPerM:0.###} out"
        + (p.CacheReadPerM is { } c ? $" / ${c:0.###} cached" : "") + " per M";
}

// ===========================================================================
// Reads the client certificate out of the kubeconfig the Higress deployment
// generates. Deliberately a few lines of string handling rather than a YAML
// dependency: the two fields are single-line base64 and a YAML parser is a lot
// of trim-unfriendly reflection to drag into an AOT build for that.
// ===========================================================================
static class KubeClientCertificate
{
    public static bool TryLoad(string path, out X509Certificate2? cert)
    {
        cert = null;
        try
        {
            if (!File.Exists(path)) return false;
            var certPem = Extract(path, "client-certificate-data:");
            var keyPem  = Extract(path, "client-key-data:");
            if (certPem is null || keyPem is null) return false;

            // Round-trip through PKCS#12. On Linux an HttpClient will not use a
            // certificate created straight from PEM for TLS client auth — the
            // private key is not associated with it in the way SslStream needs.
            using var fromPem = X509Certificate2.CreateFromPem(certPem, keyPem);
            cert = X509CertificateLoader.LoadPkcs12(fromPem.Export(X509ContentType.Pkcs12), null);
            return true;
        }
        catch { return false; }
    }

    private static string? Extract(string path, string key)
    {
        foreach (var line in File.ReadLines(path))
        {
            var t = line.TrimStart();
            if (!t.StartsWith(key, StringComparison.Ordinal)) continue;
            var b64 = t[key.Length..].Trim();
            return b64.Length == 0 ? null : Encoding.UTF8.GetString(Convert.FromBase64String(b64));
        }
        return null;
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
            LimiterSync.Kick();
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
            LimiterSync.Kick();
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

    // Tiers live in the same Redis as the balances, under their own prefix.
    //
    // Deliberately NOT in consumers.conf: that file is gitignored because it
    // holds credentials, so a tier recorded there would be invisible to review,
    // and apply.sh parses its format to build key-auth. Deliberately not a new
    // mounted file either — that would be a compose change for what is one
    // string per consumer.
    //
    // A tier is DESCRIPTIVE here, not enforced. ai-quota cannot vary behaviour
    // by tier, and ai-token-ratelimit is bundled but not installed. This
    // records the intended policy so it is visible in /keys and reviewable,
    // ahead of anything enforcing it.
    private const string TierPrefix = "chat_tier:";

    public async Task<Dictionary<string, string>> TiersAsync(CancellationToken ct)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        using var c = await ConnectAsync(ct);
        var keys = new List<string>();
        var cursor = "0";
        do
        {
            var reply = await c.CommandAsync(ct, "SCAN", cursor, "MATCH", TierPrefix + "*", "COUNT", "200");
            if (reply is not object?[] { Length: 2 } page) break;
            cursor = page[0] as string ?? "0";
            if (page[1] is object?[] batch)
                foreach (var k in batch) if (k is string t) keys.Add(t);
        } while (cursor != "0");
        if (keys.Count == 0) return result;

        var argv = new string[keys.Count + 1];
        argv[0] = "MGET";
        keys.CopyTo(argv, 1);
        if (await c.CommandAsync(ct, argv) is object?[] values)
            for (var i = 0; i < keys.Count && i < values.Length; i++)
                if (values[i] is string v && v.Length > 0)
                    result[keys[i][TierPrefix.Length..]] = v;
        return result;
    }

    public async Task SetTierAsync(string name, string tier, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        await c.CommandAsync(ct, "SET", TierPrefix + name, tier);
        LimiterSync.Kick();
    }

    public async Task<string?> TierAsync(string name, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        return await c.CommandAsync(ct, "GET", TierPrefix + name) as string is { Length: > 0 } t ? t : null;
    }

    // Per-consumer overrides, one hash per consumer holding ONLY the values set
    // by hand. A value absent from the hash follows the tier; that absence is
    // the whole "follows its tier" state, so there is no flag to keep in sync.
    // Same Redis and same reasoning as the tier key above.
    private const string PolicyPrefix = "chat_policy:";

    public async Task<Dictionary<string, string>> OverridesAsync(string name, CancellationToken ct)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        using var c = await ConnectAsync(ct);
        if (await c.CommandAsync(ct, "HGETALL", PolicyPrefix + name) is object?[] flat)
            for (var i = 0; i + 1 < flat.Length; i += 2)
                if (flat[i] is string k && flat[i + 1] is string v) result[k] = v;
        return result;
    }

    // Which consumers have at least one override. Only the key names are read:
    // HSET never leaves an empty hash behind and HDEL of the last field deletes
    // the key, so existence is exactly "has an override".
    public async Task<HashSet<string>> OverriddenAsync(CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        var keys = await ScanAsync(c, PolicyPrefix + "*", ct);
        return keys.Select(k => k[PolicyPrefix.Length..]).ToHashSet(StringComparer.Ordinal);
    }

    public async Task SetOverrideAsync(string name, string field, string value, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        await c.CommandAsync(ct, "HSET", PolicyPrefix + name, field, value);
        LimiterSync.Kick();
    }

    public async Task ClearOverrideAsync(string name, string field, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        await c.CommandAsync(ct, "HDEL", PolicyPrefix + name, field);
        LimiterSync.Kick();
    }

    // Every consumer's overrides at once, for LimiterSync and the refill job.
    public async Task<Dictionary<string, Dictionary<string, string>>> AllOverridesAsync(CancellationToken ct)
    {
        var result = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        using var c = await ConnectAsync(ct);
        foreach (var key in await ScanAsync(c, PolicyPrefix + "*", ct))
        {
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            if (await c.CommandAsync(ct, "HGETALL", key) is object?[] flat)
                for (var i = 0; i + 1 < flat.Length; i += 2)
                    if (flat[i] is string k && flat[i + 1] is string v) map[k] = v;
            result[key[PolicyPrefix.Length..]] = map;
        }
        return result;
    }

    // The refill job's marker: "<mode>:<period>" of the last period it handled
    // for this consumer. Makes refills idempotent across restarts and minutes.
    private const string RefillPrefix = "chat_refill:";

    public async Task<string?> RefillMarkerAsync(string name, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        return await c.CommandAsync(ct, "GET", RefillPrefix + name) as string;
    }

    public async Task SetRefillMarkerAsync(string name, string marker, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        await c.CommandAsync(ct, "SET", RefillPrefix + name, marker);
    }

    // One ai-token-ratelimit counter: tokens counted so far and seconds until
    // its window closes. Used is null when no window is open.
    public async Task<(long? Used, long Ttl)> CounterAsync(string key, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        var v = await c.CommandAsync(ct, "GET", key) as string;
        var ttl = await c.CommandAsync(ct, "TTL", key) is long l ? l : -2;
        return (long.TryParse(v, CultureInfo.InvariantCulture, out var n) ? n : null, ttl);
    }

    private static async Task<List<string>> ScanAsync(RespConnection c, string pattern, CancellationToken ct)
    {
        var keys = new List<string>();
        var cursor = "0";
        do
        {
            var reply = await c.CommandAsync(ct, "SCAN", cursor, "MATCH", pattern, "COUNT", "200");
            if (reply is not object?[] { Length: 2 } page) break;
            cursor = page[0] as string ?? "0";
            if (page[1] is object?[] batch)
                foreach (var k in batch) if (k is string s) keys.Add(s);
        } while (cursor != "0");
        return keys;
    }

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

        // Build the argv once at its final size: a List plus AddRange plus a
        // collection-expression copy was three allocations for a shape we know.
        var argv = new string[keys.Count + 1];
        argv[0] = "MGET";
        keys.CopyTo(argv, 1);
        if (await c.CommandAsync(ct, argv) is object?[] values)
            for (var i = 0; i < keys.Count && i < values.Length; i++)
                if (values[i] is string v && long.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n))
                    result[keys[i][Prefix.Length..]] = n;

        return result;
    }

    public async Task DeleteAsync(string name, CancellationToken ct)
    {
        using var c = await ConnectAsync(ct);
        // Every key. Otherwise a consumer re-created under the same name
        // silently inherits the revoked one's tier and hand-set limits.
        await c.CommandAsync(ct, "DEL", Prefix + name, TierPrefix + name, PolicyPrefix + name, RefillPrefix + name);
        LimiterSync.Kick();
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

    // Rented, not allocated. A connection is opened per command, so this was a
    // fresh 64 KB array on every /keys, /status and /balance.
    //
    // Deliberately NOT GC.AllocateArray(pinned: true). The Pinned Object Heap
    // exists for LONG-LIVED buffers that would otherwise pin a GC region and
    // block compaction. This one lives for a single command, and the POH is
    // never compacted — so pushing short-lived allocations through it fragments
    // the one heap that cannot defragment itself. Measured PinnedObjectsCount on
    // this process is 0: there is no pinning pressure here to relieve, and the
    // socket read pins the buffer only for the duration of the I/O.
    private readonly byte[] _buf = ArrayPool<byte>.Shared.Rent(64 * 1024);
    private int _len, _pos;

    public async ValueTask<object?> CommandAsync(CancellationToken ct, params string[] args)
    {
        var (buf, n) = EncodeCommand(args);
        try { await _s.WriteAsync(buf.AsMemory(0, n), ct); }
        finally { ArrayPool<byte>.Shared.Return(buf); }
        return await ReadAsync(ct);
    }

    // Framing is synchronous ON PURPOSE. `stackalloc` and Span locals are not
    // allowed to live across an await, so every span operation in this class
    // sits in a sync helper and the async methods do nothing but await and
    // bookkeep. That constraint is the whole reason this is shaped the way it
    // is, rather than a fluent async writer.
    //
    // Was: StringBuilder -> string -> Encoding.UTF8.GetBytes, three copies of a
    // payload whose size is known up front. MGET over every consumer is the
    // largest command sent here, so it is rented rather than stack-allocated —
    // its size scales with the consumer count and has no compile-time bound.
    private static (byte[] Buffer, int Length) EncodeCommand(string[] args)
    {
        var max = 16;
        foreach (var a in args) max += 16 + Encoding.UTF8.GetMaxByteCount(a.Length);

        var buf = ArrayPool<byte>.Shared.Rent(max);
        var w = 0;
        buf[w++] = (byte)'*';
        WriteInt(buf, ref w, args.Length);
        WriteCrLf(buf, ref w);
        foreach (var a in args)
        {
            buf[w++] = (byte)'$';
            WriteInt(buf, ref w, Encoding.UTF8.GetByteCount(a));
            WriteCrLf(buf, ref w);
            w += Encoding.UTF8.GetBytes(a, buf.AsSpan(w));
            WriteCrLf(buf, ref w);
        }
        return (buf, w);
    }

    // int.TryFormat's UTF-8 overload: straight to bytes, no intermediate string.
    private static void WriteInt(byte[] b, ref int w, int v)
    {
        v.TryFormat(b.AsSpan(w), out var written);
        w += written;
    }

    private static void WriteCrLf(byte[] b, ref int w) { b[w++] = (byte)'\r'; b[w++] = (byte)'\n'; }

    private async ValueTask<object?> ReadAsync(CancellationToken ct)
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
                return await ReadBulkAsync(n, ct);
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

    // A bulk string used to be read one byte at a time — one await, and one
    // state machine, per byte of every balance in the ledger. Copy whole runs
    // out of the buffer instead, and go to the socket only when it is empty.
    //
    // The fast path is the normal one: a RESP reply is small and Redis is on the
    // same host, so the value and its trailing CRLF are almost always already
    // buffered and the whole read is a single decode with no await at all.
    private async ValueTask<string> ReadBulkAsync(int n, CancellationToken ct)
    {
        if (_len - _pos >= n + 2)
        {
            var whole = Encoding.UTF8.GetString(_buf, _pos, n);
            _pos += n + 2;                                   // value + CRLF
            return whole;
        }

        var bytes = ArrayPool<byte>.Shared.Rent(n);
        try
        {
            var got = 0;
            while (got < n)
            {
                if (_pos >= _len) await FillAsync(ct);
                got += TakeInto(bytes, got, n - got);
            }
            await ReadByteAsync(ct); await ReadByteAsync(ct); // trailing CRLF
            return Encoding.UTF8.GetString(bytes, 0, n);
        }
        finally { ArrayPool<byte>.Shared.Return(bytes); }
    }

    // ValueTask, and this is the ONE place in this program where that is the
    // right call: it is called in a loop over a reply and virtually every call
    // is answered from the buffer without touching the socket. A synchronously
    // completing hot path is precisely what ValueTask is for. The buffer hit
    // below runs no state machine and allocates nothing.
    //
    // It is deliberately NOT applied to the command handlers or the HTTP paths.
    // Those are genuinely async, run a few times a day, and are handed to
    // Task.WhenAll/WhenAny — which take Task, so a ValueTask there would need
    // .AsTask() and would allocate MORE than it saves, in exchange for a real
    // footgun: a ValueTask may be awaited only once.
    private ValueTask<byte> ReadByteAsync(CancellationToken ct) =>
        _pos < _len ? new ValueTask<byte>(_buf[_pos++]) : RefillThenReadByteAsync(ct);

    private async ValueTask<byte> RefillThenReadByteAsync(CancellationToken ct)
    {
        await FillAsync(ct);
        return _buf[_pos++];
    }

    private async ValueTask FillAsync(CancellationToken ct)
    {
        _len = await _s.ReadAsync(_buf, ct);
        _pos = 0;
        if (_len <= 0) throw new EndOfStreamException("redis closed the connection");
    }

    private async ValueTask<string> ReadLineAsync(CancellationToken ct)
    {
        if (_pos >= _len) await FillAsync(ct);

        // Normal path: the whole line is buffered. Scan for CR with a vectorised
        // IndexOf and decode once, rather than appending char by char through a
        // StringBuilder with an await between each one.
        var i = IndexOfCr();
        if (i >= 0)
        {
            var line = TakeString(i);
            _pos++;                                          // the CR
            if (_pos >= _len) await FillAsync(ct);
            _pos++;                                          // the LF
            return line;
        }
        return await ReadLineAcrossRefillsAsync(ct);
    }

    // A header line split by a refill. Against a 64 KB buffer and RESP lines
    // that are a sigil plus a number this is effectively unreachable, but it
    // accumulates BYTES rather than decoded text because that is what stays
    // correct if it ever does happen — a multi-byte sequence can straddle the
    // split, and decoding each fragment separately would corrupt it.
    private async ValueTask<string> ReadLineAcrossRefillsAsync(CancellationToken ct)
    {
        var acc = new ArrayBufferWriter<byte>(256);
        while (true)
        {
            var i = IndexOfCr();
            if (i >= 0)
            {
                AccumulateTo(acc, i);
                _pos++;                                      // the CR
                if (_pos >= _len) await FillAsync(ct);
                _pos++;                                      // the LF
                return Encoding.UTF8.GetString(acc.WrittenSpan);
            }
            AccumulateTo(acc, _len - _pos);
            await FillAsync(ct);
        }
    }

    // The span helpers. Synchronous so no Span local ever has to survive an
    // await, which the compiler forbids outright.
    private int IndexOfCr() => _buf.AsSpan(_pos, _len - _pos).IndexOf((byte)'\r');

    private string TakeString(int count)
    {
        var s = Encoding.UTF8.GetString(_buf, _pos, count);
        _pos += count;
        return s;
    }

    private int TakeInto(byte[] dest, int offset, int want)
    {
        var take = Math.Min(want, _len - _pos);
        Buffer.BlockCopy(_buf, _pos, dest, offset, take);
        _pos += take;
        return take;
    }

    private void AccumulateTo(ArrayBufferWriter<byte> acc, int count)
    {
        acc.Write(_buf.AsSpan(_pos, count));
        _pos += count;
    }

    private bool _returned;

    public void Dispose()
    {
        // Guarded because returning a rented array twice hands the same buffer
        // to two callers, and the symptom would be one command reading another
        // command's bytes — corrupted balances, with nothing in the logs. The
        // `using` in Ledger already disposes exactly once; this is here so a
        // later edit cannot quietly turn that into data corruption.
        if (!_returned)
        {
            _returned = true;
            ArrayPool<byte>.Shared.Return(_buf);
        }
        _s.Dispose();
        client.Dispose();
    }
}

// ===========================================================================
// Types and JSON. Every type crossing the wire needs a [JsonSerializable]
// entry, or serialisation throws once trimmed.
// ===========================================================================
// ---------------------------------------------------------------------------
// Outbound Telegram, shared by the command worker and the alert worker.
// ---------------------------------------------------------------------------
sealed class Telegram(IHttpClientFactory http, ILogger<Telegram> log)
{
    public async Task SendAsync(long chatId, Reply reply, CancellationToken ct)
    {
        // Telegram caps a message at 4096 characters. Chunk on line boundaries so
        // a long /keys listing does not lose its last consumer to a hard cut.
        // The keyboard rides on the final chunk, where the question is.
        var chunks = Chunk(reply.Text, 3500).ToList();
        for (var i = 0; i < chunks.Count; i++)
        {
            var last = i == chunks.Count - 1;
            var payload = new SendMessage(chatId, chunks[i], "HTML", last ? reply.Keyboard : null);
            using var content = new StringContent(
                JsonSerializer.Serialize(payload, BotJson.Default.SendMessage), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r = await http.CreateClient("telegram").PostAsync("sendMessage", content, ct);
            if (!r.IsSuccessStatusCode)
                log.LogError("sendMessage failed: HTTP {Code} {Body}",
                    (int)r.StatusCode, await r.Content.ReadAsStringAsync(ct));
        }
    }

    // Replace a message's text and keyboard. Only for screens that are
    // navigation (settings, editors), never for a message carrying something
    // the operator must keep — a credential edited away is gone from the chat.
    // Falls back to a new message when the edit is refused, e.g. the original
    // is older than 48 hours; "message is not modified" is the one refusal
    // that needs nothing.
    public async Task EditOrSendAsync(long chatId, long messageId, Reply reply, CancellationToken ct)
    {
        if (messageId > 0 && reply.Text.Length <= 3500)
        {
            var payload = new EditMessageText(chatId, messageId, reply.Text, "HTML", reply.Keyboard);
            using var content = new StringContent(
                JsonSerializer.Serialize(payload, BotJson.Default.EditMessageText), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r = await http.CreateClient("telegram").PostAsync("editMessageText", content, ct);
            if (r.IsSuccessStatusCode) return;
            var body = await r.Content.ReadAsStringAsync(ct);
            if (body.Contains("message is not modified", StringComparison.Ordinal)) return;
            log.LogWarning("editMessageText failed, sending instead: HTTP {Code} {Body}", (int)r.StatusCode, body);
        }
        await SendAsync(chatId, reply, ct);
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
}

// ---------------------------------------------------------------------------
// Formatting shared between command replies and alert notifications.
// ---------------------------------------------------------------------------
static class Fmt
{
    // 5h 12m, 42s. For "resets in" lines, where precision below a minute
    // only matters when there is less than a minute left.
    public static string Duration(long seconds) => seconds switch
    {
        < 0 => "\u2014",
        < 60 => $"{seconds}s",
        < 3600 => $"{seconds / 60}m {seconds % 60}s",
        _ => $"{seconds / 3600}h {seconds % 3600 / 60}m",
    };

    // HTML parse_mode needs exactly three characters escaped. Consumer names,
    // upstream error strings and alert annotations all reach the wire, and one
    // stray '<' makes Telegram reject the entire message with a 400.
    public static string Esc(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    // Coarse on purpose. "2h14m" answers "is this new or has it been broken all
    // morning", which is the only question an age answers on a phone screen.
    public static string Age(TimeSpan t) =>
        t.TotalDays  >= 1 ? $"{(int)t.TotalDays}d{t.Hours:00}h"
      : t.TotalHours >= 1 ? $"{(int)t.TotalHours}h{t.Minutes:00}m"
      : t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes}m"
      : $"{Math.Max(0, (int)t.TotalSeconds)}s";

    // Severity as shape as well as colour: these survive a monochrome screen
    // and a colourblind reader, which a red dot alone does not.
    public static string Glyph(string severity) => severity switch
    {
        "critical" => "\U0001f534",
        "warning"  => "\U0001f7e0",
        "info"     => "\U0001f535",
        _          => "\u26aa"
    };

    // Block bar for a monospace histogram. Telegram renders <pre> in a fixed
    // font, so column alignment holds on every client.
    public static string Bar(int value, int max, int width)
    {
        if (max <= 0 || value <= 0) return "";
        var n = (int)Math.Round((double)value / max * width);
        return new string('\u2588', Math.Clamp(n, 1, width));
    }
}

// ---------------------------------------------------------------------------
// Alert delivery. Reads groups off the queue the webhook fills and renders one
// Telegram message per group.
//
// Deliberately NOT the same worker as commands: an operator waiting on /keys
// should not queue behind an alert storm, and an alert must not be dropped
// because a command is mid-flight.
// ---------------------------------------------------------------------------
sealed class AlertWorker(
    BotConfig cfg,
    Channel<AmWebhook> alerts,
    Telegram tg,
    ILogger<AlertWorker> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken ct)
    {
        await foreach (var hook in alerts.Reader.ReadAllAsync(ct))
        {
            try
            {
                var text = Render(hook);
                foreach (var chatId in cfg.AlertChatIds)
                    await tg.SendAsync(chatId, new Reply(text), ct);

                log.LogInformation("delivered {Status} group {Alert} ({Count} alerts) to {Chats} chat(s)",
                    hook.Status, hook.GroupLabels?.GetValueOrDefault("alertname") ?? "?",
                    hook.Alerts?.Count ?? 0, cfg.AlertChatIds.Count);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // Swallow and continue. One malformed group must not take the
                // delivery path down for every subsequent alert.
                log.LogError(ex, "failed to deliver alert group");
            }
        }
    }

    // One message per Alertmanager group. The group is already the unit of
    // meaning here — group_by is [alertname, severity] — so the header names
    // the rule and the body lists the instances it fired for.
    internal static string Render(AmWebhook hook)
    {
        var firing = string.Equals(hook.Status, "firing", StringComparison.OrdinalIgnoreCase);
        var list = hook.Alerts ?? [];
        var name = Get(hook.GroupLabels, "alertname") ?? Get(hook.CommonLabels, "alertname") ?? "alert";
        var sev  = Get(hook.GroupLabels, "severity")  ?? Get(hook.CommonLabels, "severity")  ?? "unknown";

        var sb = new StringBuilder();
        sb.Append(firing ? Fmt.Glyph(sev) : "\u2705")
          .Append(firing ? " <b>FIRING</b> \u00b7 " : " <b>RESOLVED</b> \u00b7 ")
          .Append("<b>").Append(Fmt.Esc(name)).Append("</b>\n")
          .Append("<i>").Append(Fmt.Esc(sev)).Append(" \u00b7 ").Append(list.Count)
          .Append(list.Count == 1 ? " alert" : " alerts").Append("</i>\n");

        // Per-instance detail in a monospace block, so the columns line up when
        // one rule fires for several targets at once.
        var rows = list.Select(a => (
            Who: Get(a.Labels, "instance") ?? Get(a.Labels, "job") ?? "-",
            Age: a.StartsAt is { } st
                 ? Fmt.Age((firing ? DateTimeOffset.UtcNow : a.EndsAt ?? DateTimeOffset.UtcNow) - st)
                 : "-")).ToList();

        if (rows.Count > 0)
        {
            var w = rows.Max(r => r.Who.Length);
            sb.Append("\n<pre>");
            foreach (var r in rows)
                sb.Append(Fmt.Esc(r.Who.PadRight(w))).Append("  ").Append(Fmt.Esc(r.Age)).Append('\n');
            sb.Append("</pre>");
        }

        // The rule's own words, deduplicated. A per-GPU rule otherwise repeats
        // one identical sentence once per device.
        foreach (var line in list
                     .Select(a => Get(a.Annotations, "summary") ?? Get(a.Annotations, "description"))
                     .Where(x => x is { Length: > 0 })
                     .Distinct(StringComparer.Ordinal))
            sb.Append('\n').Append(Fmt.Esc(line!));

        return sb.ToString();
    }

    private static string? Get(Dictionary<string, string>? d, string k) =>
        d is not null && d.TryGetValue(k, out var v) && v.Length > 0 ? v : null;
}

// Alertmanager webhook payload, schema version 4. Only the fields rendered here
// are declared; the deserialiser ignores the rest.
sealed class AmWebhook
{
    [JsonPropertyName("status")]       public string? Status { get; set; }
    [JsonPropertyName("receiver")]     public string? Receiver { get; set; }
    [JsonPropertyName("groupLabels")]  public Dictionary<string, string>? GroupLabels { get; set; }
    [JsonPropertyName("commonLabels")] public Dictionary<string, string>? CommonLabels { get; set; }
    [JsonPropertyName("alerts")]       public List<AmAlert>? Alerts { get; set; }
}

sealed class AmAlert
{
    [JsonPropertyName("status")]      public string? Status { get; set; }
    [JsonPropertyName("labels")]      public Dictionary<string, string>? Labels { get; set; }
    [JsonPropertyName("annotations")] public Dictionary<string, string>? Annotations { get; set; }
    [JsonPropertyName("startsAt")]    public DateTimeOffset? StartsAt { get; set; }
    [JsonPropertyName("endsAt")]      public DateTimeOffset? EndsAt { get; set; }
    [JsonPropertyName("fingerprint")] public string? Fingerprint { get; set; }
}

sealed record BotConfig(
    string BotToken, string WebhookSecret, string WebhookPath, HashSet<long> AllowedIds,
    string AdminCredential, string GatewayUrl, string ApiServerUrl, string KubeConfigPath,
    string RedisHost, int RedisPort,
    string PrometheusUrl, string PublicBaseUrl, string ConsumersPath, string AuditPath,
    string ModelId, int ContextLimit, int OutputLimit,
    string AlertSecret, string AdminApiSecret, string AlertmanagerUrl, HashSet<long> AlertChatIds,
    string LangfuseUrl, string LangfuseProjectId);

enum PendingKind { SetQuota, Revoke }
sealed record Pending(long UserId, DateTimeOffset Expires, Func<CancellationToken, Task<Reply>> Run);

// The next plain-text message from an operator answers a question the bot
// asked: a new key's name, a custom value, a custom top-up. Field is the setting
// for Kind "field". In memory like Pending: a restart forgets the question,
// which fails safe — the text then lands as an unknown command.
enum InputKind { NewKeyName, Field, TopUp }
sealed record PendingInput(InputKind Kind, string Name, string? Field, DateTimeOffset Expires);

// A command's answer: text plus an optional inline keyboard.
sealed record Reply(string Text, InlineKeyboardMarkup? Keyboard = null);

sealed class Update
{
    [JsonPropertyName("update_id")]     public long UpdateId { get; set; }
    [JsonPropertyName("message")]       public Message? Message { get; set; }
    [JsonPropertyName("callback_query")] public CallbackQuery? CallbackQuery { get; set; }
}
sealed class Message
{
    // Needed to edit a settings screen in place rather than stacking a new
    // message on every tap.
    [JsonPropertyName("message_id")] public long MessageId { get; set; }
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("from")] public User? From { get; set; }
    [JsonPropertyName("chat")] public Chat? Chat { get; set; }

    // Unix seconds, set by Telegram when the operator pressed send. The gap
    // between this and the moment we receive the update is the delivery leg,
    // and it is the only part of the round trip this process cannot measure
    // from the inside — see the queued= field in the per-command log line.
    [JsonPropertyName("date")] public long Date { get; set; }
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
// Outgoing message. parse_mode is HTML rather than MarkdownV2 on purpose:
// MarkdownV2 requires escaping ~15 characters, and an unescaped one from a
// consumer name or an error string makes Telegram reject the whole message with
// a 400. HTML needs three.
sealed record SendMessage(
    [property: JsonPropertyName("chat_id")] long ChatId,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("parse_mode")] string ParseMode = "HTML",
    [property: JsonPropertyName("reply_markup")] InlineKeyboardMarkup? ReplyMarkup = null);

sealed record EditMessageText(
    [property: JsonPropertyName("chat_id")] long ChatId,
    [property: JsonPropertyName("message_id")] long MessageId,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("parse_mode")] string ParseMode = "HTML",
    [property: JsonPropertyName("reply_markup")] InlineKeyboardMarkup? ReplyMarkup = null);

sealed record InlineKeyboardMarkup(
    [property: JsonPropertyName("inline_keyboard")] InlineKeyboardButton[][] Keyboard);

// Telegram requires EXACTLY ONE of callback_data and url per button, which is
// why both are nullable — the source-gen context drops nulls
// (DefaultIgnoreCondition = WhenWritingNull), so an unset one is absent from
// the wire rather than sent empty and rejected.
sealed record InlineKeyboardButton(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("callback_data")] string? CallbackData = null,
    [property: JsonPropertyName("url")] string? Url = null);

sealed record ChatAction(
    [property: JsonPropertyName("chat_id")] long ChatId,
    [property: JsonPropertyName("action")] string Action);

sealed record BotCommand(
    [property: JsonPropertyName("command")] string Command,
    [property: JsonPropertyName("description")] string Description);

sealed record SetMyCommands(
    [property: JsonPropertyName("commands")] BotCommand[] Commands);

sealed record AnswerCallbackQuery(
    [property: JsonPropertyName("callback_query_id")] string Id,
    [property: JsonPropertyName("text")] string? Text = null);

sealed class CallbackQuery
{
    [JsonPropertyName("id")]      public string? Id { get; set; }
    [JsonPropertyName("data")]    public string? Data { get; set; }
    [JsonPropertyName("from")]    public User? From { get; set; }
    [JsonPropertyName("message")] public Message? Message { get; set; }
}

sealed class QuotaResponse
{
    [JsonPropertyName("consumer")] public string? Consumer { get; set; }
    [JsonPropertyName("quota")]    public long Quota { get; set; }
}

// WhenWritingNull is not cosmetic: Telegram rejects an explicit
// "reply_markup": null with 400 "object expected as reply markup", so every
// message without a keyboard would fail. Measured 2026-09-03.
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(Update))]
[JsonSerializable(typeof(Message))]
[JsonSerializable(typeof(User))]
[JsonSerializable(typeof(Chat))]
[JsonSerializable(typeof(SendMessage))]
[JsonSerializable(typeof(EditMessageText))]
[JsonSerializable(typeof(InlineKeyboardMarkup))]
[JsonSerializable(typeof(InlineKeyboardButton))]
[JsonSerializable(typeof(CallbackQuery))]
[JsonSerializable(typeof(ChatAction))]
[JsonSerializable(typeof(BotCommand))]
[JsonSerializable(typeof(SetMyCommands))]
[JsonSerializable(typeof(AnswerCallbackQuery))]
[JsonSerializable(typeof(QuotaResponse))]
[JsonSerializable(typeof(AmWebhook))]
[JsonSerializable(typeof(AmAlert))]
internal partial class BotJson : JsonSerializerContext;
