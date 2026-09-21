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
    // Required, not defaulted: this repo is public, so no deployment's
    // hostnames are compiled in. A wrong default here would be handed to
    // customers in a generated config.
    PublicBaseUrl:   Req("PUBLIC_BASE_URL").TrimEnd('/'),
    KubeConfigPath:  Opt("KUBECONFIG_PATH", "/etc/kube/config"),
    ConsumersPath:   Opt("CONSUMERS_PATH", "/data/consumers.conf"),
    AuditPath:       Opt("AUDIT_PATH", "/data/audit.log"),
    ModelId:         Opt("MODEL_ID", "qwen3.8-27b"),
    ContextLimit:    int.Parse(Opt("MODEL_CONTEXT", "169000"), CultureInfo.InvariantCulture),
    OutputLimit:     int.Parse(Opt("MODEL_OUTPUT", "70000"), CultureInfo.InvariantCulture),
    // Caddy's `request_body { max_size 1MB }` on the paid hostname. Caddy parses
    // that with go-humanize, where MB is 10^6 — not 2^20 — so the real ceiling
    // is 1,000,000 bytes. It is the only limit that bounds an INLINE IMAGE, so
    // the generated OpenCode config sizes its attachment budget from it.
    BodyLimit:       int.Parse(Opt("MAX_BODY_BYTES", "4000000"), CultureInfo.InvariantCulture),
    AlertSecret:     Req("ALERT_WEBHOOK_SECRET"),
    // Shared with admin-mcp, which is the only caller of /admin/*. Optional:
    // unset means those endpoints are not mapped at all, which is the right
    // default for an install that has no MCP server in front of it.
    AdminApiSecret:  Opt("ADMIN_API_SECRET", ""),
    // admin-mcp's bot read port: a key's latest requests and one request end
    // to end, from ClickHouse, which this edge-only process cannot reach.
    // Unset secret means those screens say the records are unavailable.
    RecordsUrl:      Opt("ADMIN_MCP_READ_URL", "http://admin-mcp:8081").TrimEnd('/'),
    RecordsSecret:   Opt("BOT_READ_SECRET", ""),
    AlertmanagerUrl: Opt("ALERTMANAGER_URL", "http://qwen36-27b-alertmanager:9093").TrimEnd('/'),
    // The team's direct router hostname (Caddy, shared edge key, no gateway).
    // Its traffic has no consumer and no access-log row, so /errors reads its
    // status codes from Caddy's per-host counters instead — which only works
    // if this matches Caddy's `host` label exactly (EDGE_HOST_MODEL in the
    // inference project's .env). Required for the same reason as
    // PUBLIC_BASE_URL above.
    DirectHost:      Req("DIRECT_HOST"),
    AlertChatIds:    alertChatIds);

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
    // Overridable only so a test instance can send to a local capture server
    // and every reply can be read exactly as Telegram would receive it.
    c.BaseAddress = new Uri($"{Opt("TELEGRAM_API_BASE", "https://api.telegram.org").TrimEnd('/')}/bot{cfg.BotToken}/");
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
builder.Services.AddHttpClient("records", c =>
{
    c.BaseAddress = new Uri(cfg.RecordsUrl + "/");
    // Above admin-mcp's own 25 s ClickHouse cap plus its hop, so a slow query
    // surfaces as its error text rather than as our timeout.
    c.Timeout = TimeSpan.FromSeconds(30);
    if (cfg.RecordsSecret.Length > 0)
        c.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", "Bearer " + cfg.RecordsSecret);
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

// Per-key policy for Prometheus. Before this, Prometheus saw only the ledger
// balance (redis_exporter, chat_quota:*), so a key's quota, limits and refill
// date existed nowhere on a dashboard and "12.4M" could not be read against
// anything. The balance is deliberately NOT repeated here: the exporter stays
// the one source for what bills. Not published: Caddy forwards only the
// Telegram webhook path to this port.
//
// A limit metric is absent when the key has no such limit (0 would read as
// "limit of zero"). The daily counter is the limiter's own Redis key, so it is
// exactly what ai-token-ratelimit compares against.
app.MapGet("/metrics", async (KeyStore keys, Ledger ledger, CancellationToken ct) =>
{
    var consumersT = keys.ReadConsumersAsync(ct);
    var tiersT = ledger.TiersAsync(ct);
    var ovT = ledger.AllOverridesAsync(ct);
    await Task.WhenAll(consumersT, tiersT, ovT);
    var names = consumersT.Result.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
    var days = await Task.WhenAll(names.Select(n => ledger.CounterAsync(LimiterSync.CounterKey(n, 86_400), ct)));
    var now = DateTimeOffset.UtcNow;

    var info = new StringBuilder();
    var quota = new StringBuilder();
    var daily = new StringBuilder();
    var tpm = new StringBuilder();
    var used = new StringBuilder();
    var reset = new StringBuilder();
    var refill = new StringBuilder();
    static string N(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    for (var i = 0; i < names.Length; i++)
    {
        var n = names[i];
        var tier = tiersT.Result.GetValueOrDefault(n);
        var r = Policy.Resolve(tier, ovT.Result.TryGetValue(n, out var o) ? o : new Dictionary<string, string>());
        static long? Pos(ResolvedSetting x) =>
            long.TryParse(x.Value, CultureInfo.InvariantCulture, out var v) && v > 0 ? v : null;
        var mode = r[1].Value ?? "manual";
        var l = $"ai_consumer=\"{n}\"";
        info.Append($"quota_bot_consumer_info{{{l},tier=\"{tier ?? "none"}\",refill=\"{mode}\",limited=\"{(LimiterSync.InScope(n) ? "true" : "false")}\"}} 1\n");
        if (Pos(r[0]) is { } q)
        {
            quota.Append($"quota_bot_consumer_quota_tokens{{{l}}} {N(q)}\n");
            if (mode != "manual")
                refill.Append($"quota_bot_consumer_next_refill_timestamp_seconds{{{l}}} {N(RefillJob.Next(mode, now).ToUnixTimeSeconds())}\n");
        }
        if (LimiterSync.InScope(n))
        {
            if (Pos(r[2]) is { } d)
            {
                daily.Append($"quota_bot_consumer_daily_limit_tokens{{{l}}} {N(d)}\n");
                used.Append($"quota_bot_consumer_daily_used_tokens{{{l}}} {N(days[i].Used ?? 0)}\n");
                reset.Append($"quota_bot_consumer_daily_window_reset_seconds{{{l}}} {N(days[i].Used is null ? 0 : days[i].Ttl)}\n");
            }
            if (Pos(r[3]) is { } t)
                tpm.Append($"quota_bot_consumer_tpm_limit_tokens{{{l}}} {N(t)}\n");
        }
    }

    var sb = new StringBuilder();
    void Family(string name, string help, StringBuilder body)
    {
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n')
          .Append("# TYPE ").Append(name).Append(" gauge\n").Append(body);
    }
    Family("quota_bot_consumer_info", "One series per key: tier, refill mode, and whether gateway limits apply.", info);
    Family("quota_bot_consumer_quota_tokens",
        "Refill target from the key's tier or its own setting. NOT tokens in the balance; the balance is consumer:quota_balance:tokens.", quota);
    Family("quota_bot_consumer_next_refill_timestamp_seconds", "When the refill job next sets the balance to the quota. Absent for manual refill.", refill);
    Family("quota_bot_consumer_daily_limit_tokens", "Tokens allowed per rolling 24h window. Absent when the key has no daily limit.", daily);
    Family("quota_bot_consumer_daily_used_tokens", "Tokens counted in the current 24h window, from the limiter's own counter.", used);
    Family("quota_bot_consumer_daily_window_reset_seconds", "Seconds until the current 24h window closes; 0 when no window is open.", reset);
    Family("quota_bot_consumer_tpm_limit_tokens", "Tokens allowed per 60s window. Absent when the key has no per-minute limit.", tpm);
    return Results.Text(sb.ToString(), "text/plain; version=0.0.4; charset=utf-8");
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
            || data.StartsWith("kx:", StringComparison.Ordinal)
            || data.StartsWith("ko:", StringComparison.Ordinal)
            || data == "kl:")
        {
            var reply = await KeyCallbackAsync(data, chatId.Value, ct);
            await AnswerCallbackAsync(cb.Id, ct);
            await SendAsync(chatId.Value, reply, ct);
            return;
        }

        // A key card's own window buttons: the card re-renders in place.
        if (data.StartsWith("kw:", StringComparison.Ordinal))
        {
            var parts = data.Split(':', 3);
            var reply = parts.Length == 3 && ValidWindow(parts[1]) && SafeName(parts[2])
                ? await KeyCardAsync(parts[2], parts[1], ct)
                : new Reply("Malformed selection. Run /key again.");
            await AnswerCallbackAsync(cb.Id, ct);
            await tg.EditOrSendAsync(chatId.Value, cb.Message?.MessageId ?? 0, reply, ct);
            return;
        }

        // One key's errors, from /errors or the key card: re-renders in place.
        if (data.StartsWith("ke:", StringComparison.Ordinal))
        {
            var parts = data.Split(':', 3);
            var reply = parts.Length == 3 && ValidWindow(parts[1]) && SafeName(parts[2])
                ? await KeyErrorsAsync(parts[2], parts[1], ct)
                : new Reply("Malformed selection. Run /errors again.");
            await AnswerCallbackAsync(cb.Id, ct);
            await tg.EditOrSendAsync(chatId.Value, cb.Message?.MessageId ?? 0, reply, ct);
            return;
        }

        // Window buttons under /top, /usage and /errors: re-render in place.
        if (data.StartsWith("w:", StringComparison.Ordinal))
        {
            var reply = await WindowCallbackAsync(data, ct) ?? new Reply("Unknown button. Run the command again.");
            await AnswerCallbackAsync(cb.Id, ct);
            await tg.EditOrSendAsync(chatId.Value, cb.Message?.MessageId ?? 0, reply, ct);
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

    private async Task<Reply> DispatchAsync(long userId, long chatId, string text, CancellationToken ct)
    {
        var parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cmd = parts[0].Split('@')[0].ToLowerInvariant();
        var a1 = parts.Length > 1 ? parts[1] : null;
        var a2 = parts.Length > 2 ? parts[2] : null;

        return cmd switch
        {
            "/start" or "/help" => new Reply(HelpText),
            "/status"     => new Reply(await StatusAsync(ct)),
            "/keys"       => await KeysAsync(ct),
            "/balance"    => await BalanceAsync(a1, ct),
            "/key"        => await KeyPickerAsync(ct),
            "/usage"      => await UsageAsync(a1 ?? "24h", ct),
            "/alerts"     => new Reply(await AlertsAsync(ct)),
            "/health"     => new Reply(await HealthAsync(ct)),
            "/top"        => await TopAsync(a1 ?? "24h", ct),
            "/p95"        => new Reply(await LatencyAsync(
                                 a1 is not null && !ValidWindow(a1) ? a1 : a2 is not null && !ValidWindow(a2) ? a2 : null,
                                 a1 is not null && ValidWindow(a1) ? a1 : a2 is not null && ValidWindow(a2) ? a2 : "24h", ct)),
            "/errors"     => a1 is not null && !ValidWindow(a1)
                               ? await KeyErrorsAsync(a1, a2 ?? "24h", ct)
                               : await ErrorsAsync(a1 ?? "24h", ct),
            "/tiers"      => new Reply(TiersHelp()),
            "/prices"     => new Reply(await PricesAsync(ct)),
            // No argument is the common case — an operator wants "show me
            // this consumer", not a request id they would have to go and find
            // first. Falling back to the picker beats a usage hint.
            "/trace"      => a1 is null ? await KeyPickerAsync(ct)
                           : (await keys.ReadConsumersAsync(ct)).ContainsKey(a1) ? await KeyRequestsCardAsync(a1, ct)
                           : await TraceRequestAsync(a1, ct),
            "/tier"       => (a1 is null || a2 is null)
                                 ? new Reply(Usage("/tier &lt;name&gt; &lt;trial|team|service|batch|admin&gt;", "/tier acme service"))
                                 : await TierAsync(a1, a2, ct),
            "/policy"     => a1 is null ? await KeyPickerAsync(ct) : await PolicyCardAsync(a1, null, ct),
            "/set"        => (a1 is null || a2 is null || parts.Length < 4)
                                 ? new Reply(Usage("/set &lt;name&gt; &lt;quota|refill|daily|tpm|max_tokens&gt; &lt;value|default&gt;",
                                                   "/set acme daily 2M"))
                                 : await SetPolicyAsync(a1, a2, parts[3], ct),
            "/opencode"   => a1 is null
                                 ? await KeyPickerAsync(ct, "ko:", "Tap one to get its OpenCode config.")
                                 : await OpenCodeAsync(a1, chatId, ct),
            "/connect"    => a1 is null ? await KeyPickerAsync(ct) : await ConnectAsync(a1, ct),
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
    // this, because they had already drifted: a command once shipped in the
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
        new("status",  "Is it healthy", "", "can I operate the gateway",
            "Infrastructure health"),
        new("keys", "Who and how much", "", "every key, its balance and tier",
            "Consumers and their balances"),
        new("balance", "Who and how much", "[name]",
            "left of quota, burn and refill", "Balance against quota, one consumer or all"),
        new("key", "Who and how much", "",
            "one key: numbers, settings, requests",
            "Per-key stats and requests"),
        new("usage", "What they used", "[1h|24h|7d|30d]",
            "tokens and reference cost per key", "Tokens and requests over a window"),
        new("health", "Is it healthy", "",
            "is the stack healthy",
            "Stack and telemetry health"),
        new("alerts", "Is it healthy", "", "what is firing right now",
            "What is firing right now"),
        new("top", "What they used", "[1h|24h|7d|30d]",
            "busiest keys, share and errors", "Busiest consumers over a window"),
        new("p95", "What they used", "[name] [1h|24h|7d|30d]",
            "latency per key, gateway and engine",
            "Latency percentiles per consumer"),
        new("prices", "What they used", "",
            "OpenRouter and Alibaba price table",
            "Reference prices used for cost"),
        new("errors", "What they used", "[name] [1h|24h|7d|30d]",
            "why requests failed, all keys or one", "Failed requests by cause"),
        new("tiers", "Who and how much", "", "tier defaults and what they enforce",
            "What each policy tier means"),
        new("trace", "What they used", "&lt;name|request-id&gt;",
            "one key's or one request's records", "Per-request records: one key or one request"),
        new("tier", "Who and how much", "&lt;name&gt; &lt;tier&gt;",
            "record a consumer's tier", "Record a consumer's policy tier"),
        new("policy", "Who and how much", "&lt;name&gt;",
            "one key's limits, with buttons",
            "A consumer's settings and their source"),
        new("set", "Who and how much", "&lt;name&gt; &lt;setting&gt; &lt;value|default&gt;",
            "change one setting by typing it",
            "Change one setting for one consumer"),
        new("newkey", "Grant", "[name]",
            "create a key, one tap per tier",
            "Create a key: one tap per tier"),
        new("connect", "Grant", "&lt;name&gt;",
            "endpoint, model and API key to paste into any client",
            "A consumer's endpoint, model and API key"),
        new("opencode", "Grant", "&lt;name&gt;",
            "opencode.json tuned to this node, as a file",
            "Send a consumer's OpenCode config"),
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
        var icons = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Who and how much"] = "\U0001f511", ["What they used"] = "\U0001f4ca", ["Is it healthy"] = "\U0001fa7a",
            ["Grant"] = "\u2795", ["Destructive — these ask first"] = "\u26a0\ufe0f",
        };
        var sb = new StringBuilder("\U0001f916 <b>Gateway access</b>\n<i>Most screens have buttons; the arguments below are optional shortcuts.</i>\n");
        foreach (var group in HelpGroups)
        {
            sb.Append("\n").Append(icons.GetValueOrDefault(group, "\u2022")).Append(" <b>").Append(group).Append("</b>\n");
            foreach (var c in Commands)
            {
                if (c.Group != group) continue;
                sb.Append('/').Append(c.Name);
                if (c.Args.Length > 0) sb.Append(" <i>").Append(c.Args).Append("</i>");
                sb.Append(" \u2014 ").Append(c.Blurb).Append('\n');
            }
        }
        sb.Append(Fmt.Note(
            "Credentials are shown once, by /newkey; /keys lists names only.\n"
          + "The balance is one total: input and output tokens are charged the same. See /tiers."));
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

        // One line per dependency, verdict first. Every one of these values is
        // either "ok"-shaped or starts with FAILED/HTTP, which is what picks the icon.
        static string Line(string label, string value) =>
            (value.StartsWith("FAILED", StringComparison.Ordinal) || value.StartsWith("HTTP", StringComparison.Ordinal)
                ? "\u274c" : "\u2705") + $" <b>{label}</b> \u2014 {Esc(value)}";
        var lim = limiter.Summary();
        var body = "\U0001fa7a <b>Can I operate the gateway?</b>\n\n"
                 + string.Join("\n",
                       Line("Gateway", gw), Line("Ledger", led), Line("Prometheus", prom), Line("Key-auth", api),
                       (lim.StartsWith("FAILED", StringComparison.Ordinal) ? "\u274c" : "\u2705") + $" <b>Limiter</b> \u2014 {Esc(lim)}");

        // A failed ledger is not a degraded feature, it is an outage on the
        // billable routes: ai-quota has no fail-open, so every chat request 403s.
        if (led.StartsWith("FAILED", StringComparison.Ordinal))
            body += "\n\n\u26a0\ufe0f <b>Ledger unreachable</b> \u2014 ai-quota has no fail-open, so billable routes are returning 403 right now.";
        return body + Fmt.Note("For whether the stack is <i>healthy</i> \u2014 targets, alerts, traces, throughput \u2014 use /health.");
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
        // rather than failing fast — serially that is a dozen timeouts.
        var upT     = PromScalarAsync("count(up == 1)", ct);
        var totalT  = PromScalarAsync("count(up)", ct);
        var alertsT = PromScalarAsync("count(ALERTS{alertstate=\"firing\"}) or vector(0)", ct);
        var ledgerT = PromScalarAsync("max(redis_up)", ct);
        var genT    = PromScalarAsync("sum(sglang:gen_throughput)", ct);
        var ttftT   = PromScalarAsync(
            "histogram_quantile(0.95, sum(rate(sglang:time_to_first_token_seconds_bucket[5m])) by (le))", ct);
        // token_usage is used / pool size. The old used/available ratio divided
        // by FREE tokens and read 187% on 2026-09-13.
        var kvT     = PromScalarAsync("max(sglang:token_usage) * 100", ct);
        // Waiting inside the engines. The router's own queue (--queue-size) has
        // no gauge, but it only fills past --max-concurrent-requests 16, and
        // everything it admits lands here first.
        var queueT  = PromScalarAsync("sum(sglang:num_queue_reqs)", ct);
        var runT    = PromScalarAsync("sum(sglang:num_running_reqs)", ct);
        // Replicas holding a queue while admitting nothing from it. Normal
        // traffic admits ~20 per replica per 10 min; zero with a non-empty
        // queue is a long prompt waiting for KV room, or a wedged scheduler.
        var stuckT  = PromScalarAsync(
            "count((sum by (instance) (sglang:num_queue_reqs) > 0) and on (instance) "
          + "(sum by (instance) (increase(sglang:queue_time_seconds_count[5m])) == 0)) or vector(0)", ct);
        // Prefix cache over the last hour, exact: sums over every request the
        // engine finished (engine.requests), gateway traffic or not. The
        // sglang:cache_hit_rate gauge reads 0 on v0.5.19 regardless of traffic.
        var cacheT  = NodeCacheAsync("1h", ct);
        // HiCache host tier: how full, and what reloading from it cost. Node
        // counters, not billing numbers, so increase() is good enough here.
        var hUsedT  = PromScalarAsync("sum(sglang:hicache_host_used_tokens)", ct);
        var hTotT   = PromScalarAsync("sum(sglang:hicache_host_total_tokens)", ct);
        var lbTokT  = PromScalarAsync("sum(increase(sglang:load_back_tokens_total[1h]))", ct);
        var lbSecT  = PromScalarAsync("sum(increase(sglang:load_back_duration_seconds_sum[1h]))", ct);
        // The usage records every token number in this bot comes from:
        // SGLang file -> Vector -> ClickHouse -> Prometheus scrape.
        var scrapeT = PromScalarAsync("max(up{job=\"engine-usage\"})", ct);
        var ageT    = PromScalarAsync("time() - max(engine_usage_last_record_timestamp_seconds{source=\"engine\"})", ct);
        var staleT  = PromScalarAsync("count(ALERTS{alertname=\"EngineUsageRecordsStale\",alertstate=\"firing\"}) or vector(0)", ct);

        await Task.WhenAll(upT, totalT, alertsT, ledgerT, genT, ttftT, kvT, queueT, runT, stuckT, cacheT,
                           hUsedT, hTotT, lbTokT, lbSecT, scrapeT, ageT, staleT);

        static string N(double? v, string fmt = "N0") =>
            v is null ? "—" : ((double)v).ToString(fmt, CultureInfo.InvariantCulture);

        var up = upT.Result; var total = totalT.Result;
        var alerts = alertsT.Result; var ledger = ledgerT.Result;
        var scrapeUp = scrapeT.Result is > 0;
        var stale = staleT.Result is > 0;
        var cache = cacheT.Result;
        var stuck = stuckT.Result;

        static string Ok(bool good) => good ? "✅" : "⚠️";
        var targetsOk = up is not null && total is not null && up >= total;
        var lines = new List<string>
        {
            $"{Ok(targetsOk)} Targets <b>{N(up)}/{N(total)}</b> up",
            $"{Ok(alerts is null or 0)} Alerts <b>{N(alerts)}</b> firing",
            $"{Ok(ledger is > 0)} Ledger <b>{(ledger is null ? "—" : ledger > 0 ? "up" : "DOWN")}</b>",
            $"{Ok(scrapeUp && !stale)} Usage records <b>{(scrapeUp ? "up" : "DOWN")}</b>"
                + (ageT.Result is { } age ? $" · newest {Fmt.Age(TimeSpan.FromSeconds(Math.Max(0, age)))} ago" : ""),
            "",
            $"\U0001f4c8 Throughput <b>{N(genT.Result)}</b> tok/s",
            $"⏱ First token p95 <b>{Fmt.Secs(ttftT.Result)}</b>",
            $"\U0001f9e0 KV pool <b>{N(kvT.Result, "N1")}%</b>",
            $"\U0001f6a6 Queue <b>{N(queueT.Result)}</b> waiting · {N(runT.Result)}/{Replicas * RunningPerReplica} running",
        };
        if (cache.Hit is { } hit)
            lines.Add($"♻️ Cache hit <b>{hit * 100:0.#}%</b> · GPU {(hit - (cache.HostShare ?? 0)) * 100:0.#}% · HiCache {(cache.HostShare ?? 0) * 100:0.#}%");
        else
            lines.Add("♻️ Cache hit <b>—</b> · no requests in the last hour");
        if (hTotT.Result is > 0 && hUsedT.Result is { } hu)
            lines.Add($"\U0001f5c4 HiCache RAM <b>{hu / hTotT.Result.Value * 100:0}%</b> full"
                    + (lbTokT.Result is >= 1 && lbSecT.Result is { } lbs
                        ? $" · reloaded {Fmt.Num(lbTokT.Result.Value)} tok in {lbs:0.#}s"
                        : " · no reloads"));

        var body = "\U0001fa7a <b>Is the stack healthy?</b>\n\n" + string.Join("\n", lines);

        // Lead with the things that are silently wrong. Each of these has been
        // true on this node while every other signal looked fine.
        var warn = new List<string>();
        if (up is not null && total is not null && up < total)
            warn.Add($"{N(total - up)} scrape target(s) down — that plane is blind, not quiet.");
        if (ledger is not null && ledger == 0)
            warn.Add("Ledger unreachable — ai-quota fails closed, so billable routes are 403ing now.");
        if (!scrapeUp)
            warn.Add("Usage records are not reaching Prometheus — /usage, /top, /key, /p95 and burn rates go stale, then empty.");
        if (stale)
            warn.Add("The engine is serving but no new usage records arrived for 15 min — Vector or ClickHouse stopped ingesting.");
        if (stuck is > 0)
            warn.Add($"{N(stuck)} replica(s) have requests queued but admitted none in 5 min — a long prompt waiting for KV room, or a stalled scheduler.");
        if (alerts is > 0)
            warn.Add($"{N(alerts)} alert(s) firing — see /alerts.");

        if (warn.Count > 0)
            body += "\n\n" + string.Join("\n", warn.Select(w => "⚠️ " + w));
        else
            body += "\n\n✅ <i>Nothing firing, every target reporting.</i>";

        return body + Fmt.Note(
            "Throughput, first token, KV pool and queue are node-wide right now (KV pool: the fuller replica). "
          + "Queue is requests waiting inside the engines; the router's own queue is not exported.\n"
          + "Cache hit is the share of prompt tokens the engine did not recompute over the last hour, summed from "
          + "its per-request records: GPU is a prefix still in GPU memory, HiCache one reloaded from host RAM after "
          + "the GPU evicted it. HiCache RAM is how much of that host tier is in use.\n"
          + "Usage records are SGLang's per-request records in ClickHouse; every token count in this bot comes from "
          + "them. Newest is the last request finished, so it grows while the node is idle.");
    }

    private async Task<Reply> KeysAsync(CancellationToken ct)
    {
        var balancesT = ledger.ListAsync(ct);
        var tiersT = ledger.TiersAsync(ct);
        var consumersT = keys.ReadConsumersAsync(ct);
        var overriddenT = ledger.OverriddenAsync(ct);
        var allOvT = ledger.AllOverridesAsync(ct);
        await Task.WhenAll(balancesT, tiersT, consumersT, overriddenT, allOvT);
        var balances = balancesT.Result; var tiers = tiersT.Result; var consumers = consumersT.Result;
        var overridden = overriddenT.Result;

        if (consumers.Count == 0)
            return new Reply("No consumers yet.\n\nCreate one with <code>/newkey &lt;name&gt;</code>.");

        var sb = new StringBuilder($"\U0001f511 <b>Consumers</b> \u00b7 {consumers.Count}\n");
        foreach (var name in consumers.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            var bal = balances.TryGetValue(name, out var b)
                ? (b <= 0 ? $"\U0001f534 {Fmt.Num(b)}" : Fmt.Num(b))
                : "not seeded";
            // "no tier" rather than a guessed default: an unassigned consumer
            // is a real state and should look like one. The star marks values
            // set by hand — the consumers worth a second look.
            var tier = tiers.TryGetValue(name, out var tr) ? tr : "<i>no tier</i>";
            var plan = PlanOf(tiers.GetValueOrDefault(name),
                allOvT.Result.TryGetValue(name, out var ov) ? ov : new Dictionary<string, string>());
            var quota = plan.Quota is { } qn ? $" \u00b7 quota {Fmt.Num(qn)} {(plan.Refill == "manual" ? "manual" : plan.Refill)}" : "";
            sb.Append($"\n<b>{Esc(name)}</b>\n<code>{bal}</code> spendable \u00b7 {tier}{quota}{(overridden.Contains(name) ? " \u2731" : "")}");
        }

        var untiered = consumers.Keys.Count(n => !tiers.ContainsKey(n));
        var notes = new List<string> { "Spendable balance in tokens; the quota is the refill target, not tokens in the balance (/balance for burn, limits and refill dates). Credentials are never listed \u2014 /opencode &lt;name&gt; re-sends one." };
        if (consumers.Keys.Any(overridden.Contains))
            notes.Add("\u2731 has settings changed from its tier \u2014 /policy &lt;name&gt;.");
        if (untiered > 0)
            notes.Add($"{untiered} without a tier \u2014 /newkey sets one at creation, /tier &lt;name&gt; &lt;tier&gt; afterwards.");
        return new Reply(sb.ToString() + Fmt.Note(string.Join("\n", notes)), KeyPickerKeyboard(consumers.Keys));
    }

    // Two numbers that read alike and mean different things, so every screen
    // names them the same way (2026-09-14, after "12.4M of 300M" read as 96%
    // spent on a key that had simply never been refilled):
    //   spendable — the ledger balance, what requests draw from right now;
    //   quota     — the refill target from the key's tier or its own setting.
    //               It is not in the balance until a refill sets it there.
    // Keys are created with a hand-seeded balance and tiers are assigned later,
    // and assigning a tier never moves a balance, so the two routinely differ.
    private readonly record struct KeyPlan(long? Quota, string Refill, string? Tier, long? Daily, long? Tpm);

    private static KeyPlan PlanOf(string? tier, IReadOnlyDictionary<string, string> overrides)
    {
        var r = Policy.Resolve(tier, overrides);
        static long? Pos(ResolvedSetting x) =>
            long.TryParse(x.Value, CultureInfo.InvariantCulture, out var n) && n > 0 ? n : null;
        return new KeyPlan(Pos(r[0]), r[1].Value ?? "manual", tier, Pos(r[2]), Pos(r[3]));
    }

    // "refills to 300M on Oct 01 (+287.6M)" / "quota 300M, manual refill" /
    // "no quota set". Compact for lists, full sentence for one key.
    private static string RefillText(KeyPlan p, long balance, bool full, DateTimeOffset now)
    {
        if (p.Quota is not { } q)
            return full ? "No quota set: tokens are added only by /topup." : "no quota set";
        if (p.Refill == "manual")
            return full ? $"Quota {Fmt.Num(q)}, refilled manually: tokens are added only by /topup."
                        : $"quota {Fmt.Num(q)} · manual refill";
        var next = RefillJob.Next(p.Refill, now);
        var delta = q - balance;
        var change = delta > 0 ? $"+{Fmt.Num(delta)}" : delta < 0 ? $"drops {Fmt.Num(-delta)} unused" : "no change";
        return full
            ? $"Next refill <b>{next:yyyy-MM-dd HH:mm} UTC</b> sets the balance to the {p.Refill} quota <b>{Fmt.Num(q)}</b> ({change})."
            : $"refills to {Fmt.Num(q)} on {next:MMM dd} ({change})";
    }

    private async Task<Reply> BalanceAsync(string? name, CancellationToken ct)
    {
        if (name is not null) return await BalanceOneAsync(name, ct);

        var allT = ledger.ListAsync(ct);
        var tiersT = ledger.TiersAsync(ct);
        var ovT = ledger.AllOverridesAsync(ct);
        // Balances come from Redis directly — the billing record — and burn and
        // runway from the recording rules. If Prometheus is unreachable the
        // balances still render, because they are the half that matters.
        var runwayT = PromAsync("sum by (ai_consumer) (consumer:quota_days_left)", ct);
        var burnT = PromAsync("sum by (ai_consumer) (consumer:quota_spend:tokens24h)", ct);
        await Task.WhenAll(allT, tiersT, ovT, runwayT, burnT);

        var all = allT.Result;
        if (all.Count == 0) return new Reply("No balances recorded yet.");
        // A key can have tokens left and still be refused all day by its daily
        // limit; that is the more urgent fact, so read the limiter's counters.
        var dayUse = (await Task.WhenAll(all.Keys.Select(async k =>
            (Name: k, C: await ledger.CounterAsync(LimiterSync.CounterKey(k, 86_400), ct)))))
            .ToDictionary(x => x.Name, x => x.C);
        var runway = runwayT.Result; var burn = burnT.Result;
        var now = DateTimeOffset.UtcNow;

        var plans = all.Keys.ToDictionary(k => k, k => PlanOf(tiersT.Result.GetValueOrDefault(k),
            ovT.Result.TryGetValue(k, out var o) ? o : new Dictionary<string, string>()));
        double Days(string n) => runway.TryGetValue(n, out var d) && d < 3650 ? d : double.MaxValue;
        bool Refusing(string n) => plans[n].Daily is { } lim && LimiterSync.InScope(n)
            && dayUse.TryGetValue(n, out var c) && c.Used is { } u && u > lim;

        var sb = new StringBuilder($"\U0001f4b0 <b>Balances</b> · {all.Count} keys\n");
        sb.Append($"<b>{Fmt.Num(all.Values.Where(v => v > 0).Sum())}</b> spendable now, all keys\n");

        // Most urgent first: empty, then refused by a limit, then shortest runway.
        foreach (var (who, bal) in all.OrderBy(x => x.Value <= 0 ? -2 : Refusing(x.Key) ? -1 : Days(x.Key))
                                      .ThenBy(x => x.Key, StringComparer.Ordinal))
        {
            var d = Days(who);
            var plan = plans[who];
            var icon = bal <= 0 || Refusing(who) ? "\U0001f534" : d < 1 ? "\U0001f7e0" : d < 7 ? "\U0001f7e1" : "\U0001f7e2";
            sb.Append($"\n{icon} <b>{Esc(who)}</b> · {Esc(plan.Tier ?? "no tier")}\n");
            sb.Append($"<code>{Fmt.Num(bal)}</code> spendable · {RefillText(plan, bal, false, now)}\n");
            sb.Append(bal <= 0 ? "⛔ empty: requests get 403"
                     : d == double.MaxValue ? "idle in 24h"
                     : $"\U0001f525 {Fmt.Num(burn.GetValueOrDefault(who))}/24h · ⏳ lasts {Runway(d, true)}");
            if (Refusing(who) && plan.Daily is { } dl)
                sb.Append($"\n⛔ daily limit {Fmt.Num(dl)} hit · refused for {Fmt.Duration(dayUse[who].Ttl)}");
        }

        return new Reply(sb.ToString() + Fmt.Note(
            "<b>spendable</b> — the ledger balance; every request draws from it and it is what bills.\n"
          + "<b>quota</b> — the refill target from the key's tier (or a value set on the key). It is NOT in the "
          + "balance: a refill replaces the balance with it on the date shown (+ is what the refill adds; unused "
          + "tokens do not carry over). Setting a tier never changes a balance.\n"
          + "\U0001f525 tokens charged in the last 24h · ⏳ how long the spendable balance lasts at that rate.\n"
          + "⛔ over its daily token limit: 429 until that 24h window resets, whatever the balance.\n"
          + "\U0001f534 empty or refused · \U0001f7e0 under a day · \U0001f7e1 under a week · \U0001f7e2 more. "
          + "Tap a key for its detail, or /balance &lt;name&gt;."),
            KeyPickerKeyboard(all.Keys));
    }

    private static string Runway(double days, bool compact) =>
        days >= 365 ? (compact ? "1y+" : "over a year")
      : days < 1 ? (compact ? $"{days * 24:0}h" : $"{days * 24:0} hours")
      : compact ? $"{days:0.#}d" : $"{days:0.#} days";

    private static string LeftPct(double share) =>
        share >= 10 ? ">999%" : share > 0 && share < 0.01 ? "&lt;1%" : $"{share * 100:0}%";

    private async Task<Reply> BalanceOneAsync(string name, CancellationToken ct)
    {
        if (!SafeName(name) || !(await keys.ReadConsumersAsync(ct)).ContainsKey(name))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.");

        var balT = ledger.ListAsync(ct);
        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        var dayT = ledger.CounterAsync(LimiterSync.CounterKey(name, 86_400), ct);
        var minT = ledger.CounterAsync(LimiterSync.CounterKey(name, 60), ct);
        var markT = ledger.RefillMarkerAsync(name, ct);
        // Burn and runway come from the recording rules, so the arithmetic is
        // identical to the dashboard and the ConsumerQuotaLow alert.
        var sel = $"{{ai_consumer=\"{name}\"}}";
        var burnT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:tokens24h{sel})", ct);
        var daysT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_days_left{sel})", ct);
        var binT  = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:input24h{sel})", ct);
        var boutT = PromScalarAsync($"sum by (ai_consumer) (consumer:quota_spend:output24h{sel})", ct);
        await Task.WhenAll(balT, tierT, ovT, dayT, minT, markT, burnT, daysT, binT, boutT);

        var plan = PlanOf(tierT.Result, ovT.Result);
        var now = DateTimeOffset.UtcNow;
        var keyboard = new InlineKeyboardMarkup([
            [new InlineKeyboardButton("⚙️ Settings", $"pp:{name}"),
             new InlineKeyboardButton("\U0001f511 Key card", $"kc:24h:{name}")]]);

        var sb = new StringBuilder($"\U0001f4b0 <b>{Esc(name)}</b> · {(plan.Tier is null ? "no tier" : "tier " + Esc(plan.Tier))}\n\n");
        if (!balT.Result.TryGetValue(name, out var bal))
        {
            sb.Append("<b>No balance recorded.</b> It was never seeded, or the ledger is unreachable.\n")
              .Append(plan.Quota is { } q0 ? $"Its quota is <b>{Fmt.Num(q0)}</b>. " : "")
              .Append($"Seed it with <code>/topup {Esc(name)} {(plan.Quota ?? 1_000_000).ToString(CultureInfo.InvariantCulture)}</code>.");
            return new Reply(sb.ToString(), keyboard);
        }

        sb.Append($"\U0001f4b3 Spendable now <b>{Fmt.Num(bal)}</b> <i>({bal:N0})</i>\n");
        if (plan.Quota is { } q)
            sb.Append($"\U0001f3af Quota <b>{Fmt.Num(q)}</b> <i>({q:N0})</i> — the {(plan.Refill == "manual" ? "top-up guide" : "refill target")}, not tokens in the balance\n");
        sb.Append("\U0001f504 ").Append(RefillText(plan, bal, true, now));
        if (plan.Refill != "manual" && plan.Quota is not null && markT.Result is null)
            sb.Append(" <i>(armed on the job's next pass)</i>");
        sb.Append('\n');

        if (bal <= 0)
            sb.Append($"\n⛔ <b>Empty: every request is refused (403).</b> <code>/topup {Esc(name)} 10M</code>\n");

        if (burnT.Result is > 0 && daysT.Result is { } days)
        {
            var bi = binT.Result ?? 0; var bo = boutT.Result ?? 0;
            sb.Append($"\n\U0001f525 Spent in 24h <b>{Fmt.Num(burnT.Result.Value)}</b>\n")
              .Append($"⬇ {Fmt.Num(bi)} in · ⬆ {Fmt.Num(bo)} out")
              .Append(Fmt.Ratio(bi, bo) is { Length: > 0 } ratio ? $" · {ratio}\n" : "\n")
              .Append($"⏳ The spendable balance lasts <b>{Runway(days, false)}</b> at this rate\n");
            if (plan.Refill != "manual" && plan.Quota is not null && days < (RefillJob.Next(plan.Refill, now) - now).TotalDays && bal > 0)
                sb.Append($"⚠️ <b>Runs out before the refill.</b> <code>/topup {Esc(name)} 10M</code>\n");
            else if (days < 1 && bal > 0)
                sb.Append($"⚠️ <b>Under a day left.</b> <code>/topup {Esc(name)} 10M</code>\n");
        }
        else
            sb.Append("\n<i>Nothing spent in the last 24h, so there is no burn rate to project.</i>\n");

        // The limiter's own counters, so "used" is what it compares against.
        string Window(string icon, string label, (long? Used, long Ttl) c, long? limit)
        {
            if (limit is null || !LimiterSync.InScope(name)) return $"{icon} {label}: no limit\n";
            if (c.Used is not { } used) return $"{icon} {label} {Fmt.Num(limit.Value)}: nothing used, no window open\n";
            var line = $"{icon} {label} {Fmt.Num(limit.Value)}: {Fmt.Num(used)} used ({LeftPct((double)used / limit.Value)}), resets in {Fmt.Duration(c.Ttl)}";
            return line + (used > limit ? " ⛔ <b>refusing</b>\n" : "\n");
        }
        sb.Append('\n')
          .Append(Window("\U0001f4c5", "Daily limit", dayT.Result, plan.Daily))
          .Append(Window("⏱", "Per-minute limit", minT.Result, plan.Tpm));

        return new Reply(sb.ToString() + Fmt.Note(
            "<b>Spendable</b> is the ledger balance: every request draws from it, input and output tokens alike, "
          + "and one request can overdraw it by its own size.\n"
          + "<b>Quota</b> comes from the key's tier unless a value is set on the key. It is only a target: a refill "
          + "replaces the balance with it (unused tokens do not carry over), and setting a tier or quota never "
          + "moves the balance by itself.\n"
          + "The daily and per-minute windows open at the key's first request and refuse with 429 once over."), keyboard);
    }

    private async Task<Reply> UsageAsync(string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return new Reply(BadWindow(window));
        var keyboard = WindowButtons("usage", window, "1h", "24h", "7d", "30d");

        // Exact sums over the requests SGLang finished in the window. Split by
        // direction: the ledger charges input and output identically, but a
        // single total hides that a consumer at 40:1 is paying almost entirely
        // for context it re-sent, most of which the prefix cache served.
        var inT    = UsageByConsumerAsync("engine_usage_prompt_tokens", window, ct, withDirect: true);
        var outT   = UsageByConsumerAsync("engine_usage_completion_tokens", window, ct, withDirect: true);
        var reqsT  = UsageByConsumerAsync("engine_usage_requests", window, ct, withDirect: true);
        var knownT = UsageByConsumerAsync("engine_usage_cache_known_prompt_tokens", window, ct, withDirect: true);
        var devT   = UsageByConsumerAsync("engine_usage_cached_device_tokens", window, ct, withDirect: true);
        var hostT  = UsageByConsumerAsync("engine_usage_cached_host_tokens", window, ct, withDirect: true);
        var nodeT  = NodeCacheAsync(window, ct);
        var warnT  = UsageDataWarningAsync(ct);
        var pricesT = priceBook.GetAsync(ct);
        await Task.WhenAll(inT, outT, reqsT, knownT, devT, hostT, nodeT, warnT, pricesT);

        var inp = inT.Result; var outp = outT.Result; var reqs = reqsT.Result;
        var node = nodeT.Result; var prices = pricesT.Result;
        CacheSplit Cache(string n) => new(inp.GetValueOrDefault(n), knownT.Result.GetValueOrDefault(n),
                                          devT.Result.GetValueOrDefault(n), hostT.Result.GetValueOrDefault(n), 0);

        double Tok(string n) => inp.GetValueOrDefault(n) + outp.GetValueOrDefault(n);
        var all = inp.Keys.Union(outp.Keys).ToList();
        var names = all.Where(n => Tok(n) >= 0.5).OrderByDescending(Tok).ToList();
        if (names.Count == 0)
            return new Reply($"\U0001f4ca <b>Usage</b> · {window}\n{warnT.Result}\nNo tokens charged in this window.", keyboard);

        var totalIn = names.Sum(n => inp.GetValueOrDefault(n));
        var totalOut = names.Sum(n => outp.GetValueOrDefault(n));
        decimal orT = 0, orcT = 0, sgT = 0, bjT = 0;
        var partial = false;

        var sb = new StringBuilder($"\U0001f4ca <b>Usage</b> · {window}\n{warnT.Result}");
        sb.Append($"<b>{Fmt.Num(totalIn + totalOut)}</b> tokens · ⬇ {Fmt.Num(totalIn)} in · ⬆ {Fmt.Num(totalOut)} out\n");

        foreach (var n in names)
        {
            var i = inp.GetValueOrDefault(n); var o = outp.GetValueOrDefault(n);
            var c = Cache(n);
            partial |= c.Coverage is < 0.99;
            var list = PriceBook.Cost(prices.OpenRouter, i, o);
            var cached = PriceBook.Cost(prices.OpenRouter, i, o, c.CostHit);
            var sg = PriceBook.Cost(prices.AlibabaSg, i, o);
            var bj = PriceBook.Cost(prices.AlibabaBj, i, o);
            orT += list; orcT += cached; sgT += sg; bjT += bj;

            sb.Append($"\n<b>{ConsumerTitle(n)}</b>\n");
            sb.Append($"⬇ {Fmt.Num(i)} · ⬆ {Fmt.Num(o)} · {Fmt.Num(reqs.GetValueOrDefault(n))} req")
              .Append(c.Hit is { } hh ? $" · ♻️ {hh * 100:0}%" : "")
              .Append(c.HostShare is > 0 and var hs ? $" (HiCache {hs * 100:0.#}%)" : "")
              .Append('\n');
            sb.Append($"\U0001f4b5 OR {PriceBook.Usd(list)} · SG {PriceBook.Usd(sg)} · BJ {PriceBook.Usd(bj)}\n");
        }

        var idle = all.Count - names.Count;
        sb.Append("\n<b>At reference prices</b>\n")
          .Append($"OpenRouter <b>{PriceBook.Usd(orT)}</b> · with cache {PriceBook.Usd(orcT)}\n")
          .Append($"Alibaba SG <b>{PriceBook.Usd(sgT)}</b> · BJ <b>{PriceBook.Usd(bjT)}</b>");
        if (node.Hit is { } nh)
        {
            var nhost = node.HostShare ?? 0;
            sb.Append("\n\n<b>Prefix cache, node</b>\n")
              .Append($"♻️ {nh * 100:0}% of prompt tokens cached · GPU {(nh - nhost) * 100:0}% · HiCache {nhost * 100:0.#}%");
            if (node.AvoidedSeconds is { } avoided && node.PrefillRate is { } rate)
                sb.Append($"\n⏱ ≈ {Fmt.Duration((long)avoided)} of prefill avoided, at the {Fmt.Num(rate)} tok/s prefill measured in this window");
        }
        if (idle > 0) sb.Append($"\n<i>{idle} key(s) with no tokens in this window not shown.</i>");

        sb.Append(Fmt.Note(
            "⬇ input · ⬆ output · ♻️ share of input the engine served from its prefix cache; HiCache is the part "
          + "reloaded from host RAM after eviction from the GPU.\n"
          + "<b>Where the numbers come from</b>: SGLang's own record of every request it finished, summed exactly — "
          + "the same token counts the ledger charged. Requests cut off before their usage frame (client disconnect, "
          + "timeout, error) were charged nothing and are not in these totals; /top counts them as cut. Up to ~2 minutes behind.\n"
          + (partial ? "Some input in this window predates the per-request cache records (2026-09-14): ♻️ covers the "
                     + "input whose split is known, and \"with cache\" prices the rest as uncached.\n" : "")
          + "<b>Cache and cost</b>: cached input still counts in full against the balance and limits. What a hit "
          + "saves is engine time — a GPU hit is nearly free, a HiCache hit costs a reload (~1.2 s per 45K tokens "
          + "against ~14 s to recompute) — and, at reference prices, it is priced at the cached rate.\n\n"
          + "<b>Reference prices</b> — what the same tokens cost for this model elsewhere, not a bill.\n"
          + $"OR: {Esc(prices.OpenRouter.Basis)}, {Esc(PriceBook.PerMText(prices.OpenRouter))}"
          + (prices.Providers > 0 && prices.OutMin is { } lo && prices.OutMax is { } hi
                ? $"; output ${lo:0.##}–{hi:0.##} across {prices.Providers} providers" : "")
          + ". \"with cache\" prices each key's cached input at the cached rate.\n"
          + $"SG/BJ: Alibaba Cloud Singapore {Esc(PriceBook.PerMText(prices.AlibabaSg))}, "
          + $"Beijing {Esc(PriceBook.PerMText(prices.AlibabaBj))}, as of 2026-09-12.\n"
          + "Balances are the billing record."));
        return new Reply(sb.ToString(), keyboard);
    }

    // ---- exact usage reads ------------------------------------------------
    //
    // Every token, request, cache and engine-latency number in this bot comes
    // from engine.requests: one row per request SGLang finished, written by
    // the engine itself (--export-metrics-to-file), shipped by Vector, and
    // summed in ClickHouse per window and consumer. Prometheus scrapes those
    // sums once a minute as gauges (engine_usage_*, gateway_usage_*; see
    // clickhouse/engine-usage-metrics.sql). Windows are 1h, 24h, 7d and 30d.
    //
    // Why not rate()/increase() over counters, as this bot did until
    // 2026-09-14: the engine's per-consumer counters are created by a
    // consumer's first request after a replica start, so increase() never saw
    // that request, and increase() extrapolates to the window edges. Measured
    // on three controlled requests of 29,079 prompt tokens: engine increase()
    // said 11,194, gateway increase() 29,825, the per-request records 29,079.
    //
    // The bot still reads Prometheus, not ClickHouse: it runs on `edge`, and
    // an internet-reachable bot with a route to the backend is a worse trade.

    // One gauge for one window, keyed by consumer. Engine traffic that did not
    // pass the gateway has no consumer label and drops out here, as it should.
    // withDirect keeps the rows with no consumer label, under DirectConsumer.
    // Those are the direct hostname's requests: SGLang records them with an
    // empty consumer and Prometheus drops empty labels, so a lookup by label
    // silently loses them — and they can be most of the node's load.
    private async Task<Dictionary<string, double>> UsageByConsumerAsync(
        string metric, string window, CancellationToken ct, string extra = "", bool withDirect = false)
    {
        var query = $"sum by (consumer) ({metric}{{window=\"{window}\"{extra}}})";
        if (!withDirect) return await PromAsync(query, ct, "consumer");
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (labels, value) in await PromSeriesAsync(query, ct))
            result[labels.GetValueOrDefault("consumer", DirectConsumer)] = value;
        return result;
    }

    // Key names are [a-z0-9-], so the empty string cannot collide with one.
    private const string DirectConsumer = "";

    private string ConsumerTitle(string n) =>
        n == DirectConsumer ? $"{Esc(cfg.DirectHost)} · direct, shared key" : Esc(n);

    // The whole node's prefix cache for a window: every finished request,
    // gateway traffic or not.
    private async Task<CacheSplit> NodeCacheAsync(string window, CancellationToken ct)
    {
        Task<double?> Q(string m) => PromScalarAsync($"sum({m}{{window=\"{window}\"}})", ct);
        var pT = Q("engine_usage_prompt_tokens");
        var kT = Q("engine_usage_cache_known_prompt_tokens");
        var dT = Q("engine_usage_cached_device_tokens");
        var hT = Q("engine_usage_cached_host_tokens");
        var sT = Q("engine_usage_prefill_seconds");
        await Task.WhenAll(pT, kT, dT, hT, sT);
        return new CacheSplit(pT.Result ?? 0, kT.Result ?? 0, dT.Result ?? 0, hT.Result ?? 0, sT.Result ?? 0);
    }

    // If the ClickHouse scrape stops, Prometheus keeps the last sample for five
    // minutes and then returns nothing, which would render as "no usage". An
    // empty screen and a blind one must not look alike.
    private async Task<string> UsageDataWarningAsync(CancellationToken ct) =>
        await PromScalarAsync("max(up{job=\"engine-usage\"})", ct) is > 0
            ? ""
            : "⚠️ <b>Usage data unavailable</b> — the usage records scrape is down, so these numbers may be stale or missing. /health\n";

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
        var text = new StringBuilder($"\u2795 <b>Create {Esc(name)}</b>\nTap a tier \u2014 the key is created at once, and every value can be changed after.\n");
        foreach (var t in Policy.All.Where(t => t.Name != "admin"))
        {
            var tier = t.Name;
            var token = Tokenize(userId, TimeSpan.FromMinutes(10), c => NewKeyAsync(name, tier, null, c));
            rows.Add([new InlineKeyboardButton(
                $"{tier} \u00b7 {Policy.Compact(t.Quota)} \u00b7 {Policy.RefillName(t.Refill)}", "ok:" + token)]);
            text.Append($"\n<b>{tier}</b> \u2014 {Esc(t.For)}\n{Policy.Compact(t.Quota)} {Policy.RefillName(t.Refill)} \u00b7 "
                      + $"{(t.Daily == 0 ? "\u221e" : Policy.Compact(t.Daily))}/day \u00b7 {(t.Tpm == 0 ? "\u221e" : Policy.Compact(t.Tpm))}/min\n");
        }
        var untiered = Tokenize(userId, TimeSpan.FromMinutes(10), c => NewKeyAsync(name, null, "1000000", c));
        rows.Add([new InlineKeyboardButton("no tier \u00b7 1M", "ok:" + untiered),
                  new InlineKeyboardButton("Cancel", "no:" + untiered)]);
        text.Append(Fmt.Note("Balance, daily and per-minute limits and refill are enforced; max_tokens is recorded. Details: /tiers."));
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
             new InlineKeyboardButton("Key card", $"kc:24h:{name}")],
            [new InlineKeyboardButton("\U0001f50c Connect", $"kx:{name}"),
             new InlineKeyboardButton("\U0001f9e9 OpenCode config", $"ko:{name}")]
        ]);
        // The KEY is the thing shown once, so it is the thing in the message \u2014
        // one <code> block, one tap to copy on a phone. The config that wraps it
        // is regenerated on demand by /opencode, and inlining 2.4 KB of JSON
        // here only pushed the credential below the fold.
        return new Reply(
            $"Created <b>{Esc(name)}</b>{(tier is null ? "" : $" on <b>{Esc(tier)}</b>")} with <code>{quota:N0}</code> tokens.\n\n"
          + "\u26a0\ufe0f <b>This credential is shown once.</b> Tap it to copy.\n\n"
          + $"<code>{Esc(credential["Bearer ".Length..])}</code>\n\n"
          + "<b>Connect</b> is the endpoint, model and key for any client.\n"
          + "<b>OpenCode config</b> sends <code>opencode.json</code> tuned to this node.\n\n"
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

    // Everything a client needs, for any OpenAI-compatible tool (Hermes, Cursor,
    // the openai SDK, curl), not only OpenCode. Each value is its own <code> so
    // one tap copies exactly that value on a phone. Sent as a NEW message on
    // every path (command and key-card button) and never edited, like every
    // other message carrying a credential. Audited, like /opencode.
    private async Task<Reply> ConnectAsync(string name, CancellationToken ct)
    {
        if (!SafeName(name)) return new Reply("That is not a consumer name this bot recognises.");
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.TryGetValue(name, out var credential))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.");
        var apiKey = credential.StartsWith("Bearer ", StringComparison.Ordinal) ? credential["Bearer ".Length..] : credential;

        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        var balT = ledger.ListAsync(ct);
        await Task.WhenAll(tierT, ovT, balT);
        await AuditAsync($"connect name={name}", ct);

        var plan = PlanOf(tierT.Result, ovT.Result);
        static string Lim(long? n) => n is { } v ? Fmt.Num(v) : "none";
        var hasBal = balT.Result.TryGetValue(name, out var b);
        var baseUrl = $"{cfg.PublicBaseUrl}/v1";

        var text =
            $"\U0001f50c <b>{Esc(name)}</b> \u00b7 connection\n\n"
          + $"<b>Base URL</b>\n<code>{Esc(baseUrl)}</code>\n\n"
          + $"<b>Model</b>\n<code>{Esc(cfg.ModelId)}</code>\n\n"
          + $"<b>API key</b>\n<code>{Esc(apiKey)}</code>\n\n"
          + "<b>Request limits</b>\n"
          + $"Context {cfg.ContextLimit:N0} tokens, prompt + output\n"
          + $"Output at most {cfg.OutputLimit:N0} tokens per request\n"
          + $"Body at most {Fmt.Num(cfg.BodyLimit)} bytes of JSON\n\n"
          + $"<b>This key</b> \u00b7 {Esc(plan.Tier ?? "no tier")}\n"
          + (hasBal ? $"Spendable now {Fmt.Num(b)} \u00b7 {RefillText(plan, b, false, DateTimeOffset.UtcNow)}\n" : "Balance not seeded\n")
          + $"Daily limit {Lim(plan.Daily)} \u00b7 per minute {Lim(plan.Tpm)}"
          + Fmt.Note(
              "OpenAI-compatible chat completions. Clients send the key as <code>Authorization: Bearer &lt;key&gt;</code>; "
            + "most take the base URL and key as <code>OPENAI_BASE_URL</code> / <code>OPENAI_API_KEY</code>.\n"
            + "<code>/v1/models</code> does not report the context window, so set it by hand in the client "
            + $"({cfg.ContextLimit:N0}).\n"
            + "Errors: 429 daily or per-minute token limit (see Retry-After) \u00b7 403 balance exhausted \u00b7 "
            + "400 prompt longer than the context \u00b7 422 max_tokens above the output limit \u00b7 401 wrong key.\n"
            + "Every input and output token counts against the balance and the limits, cached prompt tokens included. "
            + "Spendable is what requests draw from now; the quota is only what a refill sets the balance to.\n"
            + "\u26a0\ufe0f This message holds a live credential: forward it only to the key's owner, and delete it when done.");

        return new Reply(text, new InlineKeyboardMarkup([
            [new InlineKeyboardButton("\U0001f511 Key card", $"kc:24h:{name}"),
             new InlineKeyboardButton("\u2699\ufe0f Settings", $"kp:{name}")],
            [new InlineKeyboardButton("\U0001f9e9 OpenCode config", $"ko:{name}")]]));
    }

    // Sent as a FILE, not a message. The config is ~2.5 KB; Telegram truncates a
    // message at 4096 characters and SendAsync chunks on line boundaries, which
    // would cut a <pre> block in half and get the whole message refused. A file
    // is also what the user actually needs: it lands at the documented path with
    // one tap. Audited, like /connect \u2014 it carries a live credential.
    private async Task<Reply> OpenCodeAsync(string name, long chatId, CancellationToken ct)
    {
        if (!SafeName(name)) return new Reply("That is not a consumer name this bot recognises.");
        var consumers = await keys.ReadConsumersAsync(ct);
        if (!consumers.TryGetValue(name, out var credential))
            return new Reply($"No consumer named <b>{Esc(name)}</b>.\n\nRun /keys to see who exists.");
        await AuditAsync($"opencode name={name}", ct);

        await SendDocumentAsync(chatId, "opencode.json",
            Encoding.UTF8.GetBytes(OpenCodeJson(credential)), "application/json",
            $"\U0001f9e9 <b>{Esc(name)}</b> \u2014 OpenCode config for this node.\n"
          + "Save as <code>~/.config/opencode/opencode.json</code> (or <code>opencode.json</code> "
          + "in a project, which wins over it).\n"
          + "\u26a0\ufe0f Holds a live API key.", ct);

        // Second message, and the other half of what the operator forwards: a
        // fresh OpenCode install starts on a free model that can already run
        // commands, so the customer does not have to find a config directory or
        // merge JSON by hand \u2014 they paste this and the agent installs the file.
        await SendAsync(chatId, new Reply(
            "\U0001f4cb <b>Setup prompt</b> \u2014 paste into a freshly installed OpenCode, "
          + "running on whatever free model it starts with, once the file above is downloaded. "
          + "It installs the file and tests it.\n\n"
          + $"<pre>{Esc(OpenCodeSetupPrompt())}</pre>"), ct);

        return new Reply(
            $"\U0001f9e9 <b>{Esc(name)}</b> \u00b7 OpenCode\n\n"
          + "<b>Parallel work</b>\n"
          + "Subagents are what run in parallel \u2014 <code>@general</code> for a unit of work, "
          + "<code>@explore</code> to search. Each one is its own request to this node.\n"
          + $"The node decodes <b>{Replicas * RunningPerReplica} requests at once</b> across ALL customers "
          + $"({Replicas} replicas \u00d7 {RunningPerReplica}); beyond that the router queues up to "
          + $"{RouterQueue}, and a queued request waits up to {RouterQueueTimeout / 60} minutes before it is "
          + "refused. Three or four parallel subagents is the useful range \u2014 past that they queue behind "
          + "each other and each one decodes slower, because decode here is bound by memory bandwidth the "
          + "streams share.\n"
          + "<code>subagent_depth: 1</code> keeps a subagent from starting its own, which is what turns a "
          + "fan-out into a stampede.\n\n"
          + "<b>Images</b>\n"
          + "The model reads them \u2014 paste or drag a screenshot in. The config resizes every image to "
          + $"{ImageEdge}\u00d7{ImageEdge} and {Fmt.Num(ImageBudget(cfg.BodyLimit))} bytes of base64 first, "
          + $"because the gateway refuses a request body over {Fmt.Num(cfg.BodyLimit)} bytes and every image "
          + "in a conversation is sent again on each turn. OpenCode's own default (5 MB per image) would be "
          + "refused.\n"
          + $"An image costs about one token per 32\u00d732 pixels: at {ImageEdge}\u00d7{ImageEdge} that is "
          + $"~{ImageEdge / 32 * (ImageEdge / 32):N0} tokens, charged like any other input.\n\n"
          + "<b>Thinking</b>\n"
          + "The server thinks at <code>xhigh</code> by default and thinking tokens are billed output at "
          + "~60 tokens/s. The config asks for <code>high</code> for normal work, <code>xhigh</code> only in "
          + "plan mode, less for search, and <code>none</code> for titles and summaries."
          + Fmt.Note(
              "Two model entries, one served model: the second sets "
            + "<code>reasoningEffort: none</code> and is wired to <code>small_model</code>, so a session "
            + "title never costs a round of deep reasoning.\n"
            + $"<code>limit.input</code> is {Fmt.Num(cfg.ContextLimit - OpenCodeOutput(cfg.OutputLimit))}, not the "
            + $"{Fmt.Num(cfg.ContextLimit)} window: prompt and output share it, so the prompt ceiling is the "
            + $"window minus the {Fmt.Num(OpenCodeOutput(cfg.OutputLimit))} OpenCode asks for. Getting this "
            + "wrong is the 400 \"Input length exceeds the maximum allowed length\".\n"
            + "Compaction is left LATE and <code>prune</code> off on purpose: both rewrite history, and "
            + "rewritten history misses this node's prefix cache, turning a 1-2 s prefill back into a cold "
            + "one.\n"
            + "Forward the file and the setup prompt together: a fresh OpenCode starts on a free model "
            + "that can already run commands, so the owner pastes the prompt and it installs the file, "
            + "checks the provider loaded and runs one real request. The prompt tells it not to edit any "
            + "value \u2014 every number is matched to a limit of this node.\n"
            + "\u26a0\ufe0f The file holds a live credential: forward it only to the key's owner."),
            new InlineKeyboardMarkup([
                [new InlineKeyboardButton("\U0001f50c Connect", $"kx:{name}"),
                 new InlineKeyboardButton("\u2699\ufe0f Settings", $"kp:{name}")]]));
    }

    // Kept in step with ../qwen3.6-27b-2-A100/docs/OPENCODE_SETUP_PROMPT.md,
    // which explains each instruction. Lines stay under ~60 characters: this is
    // read inside a <pre> block, and a wider one scrolls sideways on a phone.
    //
    // It forbids editing the file for a reason. Every number in the config is
    // matched to a measured limit of this node, and a model that helpfully
    // raises limit.output to the gateway's advertised ceiling, or restores
    // OpenCode's default timeouts, produces failed requests: a 400 at long
    // context, or an abort in the middle of a cold prefill the node is serving.
    private string OpenCodeSetupPrompt() =>
        """
        Set me up to use my own model endpoint in opencode. Do the
        work yourself, then tell me what to do next.

        I have a file called opencode.json from the people who run
        the endpoint. It is already tuned to their server.

        1. Find it: look in ~/Downloads, ~/Desktop, /tmp and the
           current directory. If it is not there, stop and ask me
           to paste its contents.
        2. Do not change any value inside it. Every number is
           matched to that server's limits, and "correcting" one
           causes failed requests.
        3. It holds a live API key. Do not print it, do not copy
           it into a project directory or anything tracked by git,
           and do not pass it on a shell command line.
        4. Install it globally, not per project:
           - mkdir -p ~/.config/opencode
           - if ~/.config/opencode/opencode.json already exists,
             copy it to opencode.json.bak first, then merge: keep
             my own settings, take every key from the new file,
             and tell me which ones collided.
           - otherwise move the file there.
           - chmod 600 ~/.config/opencode/opencode.json
        5. Check it parses: jq . ~/.config/opencode/opencode.json
           (or python3 -m json.tool < that file).
        6. Check opencode loaded it: opencode models qwen-gw
        7. Test it for real:
           opencode run -m qwen-gw/MODEL "Reply with: OK"
           Allow up to 15 minutes and do not kill it early. The
           first token can take a couple of minutes when the
           server is busy; that is normal, not a hang. Report the
           exit status and the reply.
        8. Install the image-batches skill. It keeps image work
           inside this server's limits and says what to do when
           a session stops working:
           - mkdir -p ~/.config/opencode/skills/image-batches
           - save this page into that folder as SKILL.md:
             https://raw.githubusercontent.com/bogdannadev/inference-ops/master/qwen3.6-27b-2-A100/docs/opencode-skills/image-batches/SKILL.md
           - check: opencode debug skill > /tmp/skills.json
             then grep -c image-batches /tmp/skills.json
             (piping it straight into grep truncates it)
        9. Then tell me to restart opencode and select
           qwen-gw/MODEL with /models. Change none of my other
           settings.

        If a step fails, stop and show me the exact error instead
        of working around it.
        """.Replace("MODEL", cfg.ModelId, StringComparison.Ordinal);

    // Deployment shape, quoted to the user and used for nothing else. Two TP=1
    // replicas at --max-running-requests 4, behind a router at --queue-size 64
    // / --queue-timeout-secs 300 (../qwen3.6-27b-2-A100/docker-compose.yml).
    private const int Replicas = 2, RunningPerReplica = 4, RouterQueue = 64, RouterQueueTimeout = 300;

    // Longest edge an image is resized to before sending. The vision tower is
    // patch 16 with spatial_merge 2, so one token covers a 32x32 block: 1280 is
    // 1,600 tokens for a full-square image and keeps a screenshot legible.
    private const int ImageEdge = 1280;

    // An eighth of the body per image. OpenCode resends every image in the
    // history on every turn, so one request has to hold several: the
    // image-batches skill puts 6 images plus up to 3 crops in one subagent.
    // At the 4 MB body cap this is 500 KB, the same per-image budget the old
    // 1 MB cap gave with a half share, so existing configs stay valid.
    private static int ImageBudget(int bodyLimit) => bodyLimit / 8;

    // What OpenCode will actually put in max_tokens. It clamps to its own
    // OUTPUT_TOKEN_MAX (32,000, provider/transform.ts) whatever the config says,
    // so declaring the gateway's 70,000 ceiling here would only misstate the
    // arithmetic that limit.input depends on.
    private static int OpenCodeOutput(int gatewayCeiling) => Math.Min(gatewayCeiling, 32_000);

    private string OpenCodeJson(string credential)
    {
        // The credential in consumers.conf carries the literal "Bearer " prefix,
        // because key-auth matches the raw header value. An OpenAI-compatible
        // client adds its own "Bearer ", so strip it here or the gateway sees
        // "Bearer Bearer sk-..." and refuses.
        var apiKey = credential.StartsWith("Bearer ", StringComparison.Ordinal)
            ? credential["Bearer ".Length..]
            : credential;

        // prompt + output share one window, so the prompt ceiling is the window
        // minus what the client will ask to generate. OpenCode reads limit.input
        // as exactly that ceiling (session/overflow.ts::usable).
        var output = OpenCodeOutput(cfg.OutputLimit);
        var input = cfg.ContextLimit - output;

        // Both entries point at the same served model through `id`; only the
        // effort differs. `attachment` + `modalities.input` are what let OpenCode
        // offer an image at all \u2014 /v1/models advertises neither.
        JsonObject Model(string label, string effort, bool reasoning) => new()
        {
            ["id"]          = cfg.ModelId,
            ["name"]        = label,
            ["attachment"]  = true,
            ["reasoning"]   = reasoning,
            ["tool_call"]   = true,
            // Explicitly off. Turning it on makes OpenCode send each turn's
            // reasoning back as reasoning_content on the NEXT request: paid
            // input tokens for text this model's chat template discards.
            ["interleaved"] = false,
            ["temperature"] = false,
            ["modalities"]  = new JsonObject
            {
                ["input"]  = new JsonArray("text", "image"),
                ["output"] = new JsonArray("text")
            },
            ["limit"] = new JsonObject
            {
                ["context"] = cfg.ContextLimit,
                ["input"]   = input,
                ["output"]  = output
            },
            ["options"] = new JsonObject { ["reasoningEffort"] = effort }
        };

        var doc = new JsonObject
        {
            ["$schema"] = "https://opencode.ai/config.json",
            ["provider"] = new JsonObject
            {
                ["qwen-gw"] = new JsonObject
                {
                    ["npm"]  = "@ai-sdk/openai-compatible",
                    ["name"] = "Qwen3.8-27B (A100 gateway)",
                    ["options"] = new JsonObject
                    {
                        ["baseURL"] = $"{cfg.PublicBaseUrl}/v1",
                        ["apiKey"]  = apiKey,
                        // Sized against the engine's --request-timeout-secs 900.
                        // The first byte can legitimately be minutes away: a
                        // queued request waits up to 300 s at the router and a
                        // cold 160K prefill then runs ~80-130 s, so OpenCode's
                        // 5-minute defaults for headers and for the gap between
                        // SSE chunks would abort requests the node is serving.
                        ["timeout"]       = 900_000,
                        ["headerTimeout"] = 600_000,
                        ["chunkTimeout"]  = 600_000
                    },
                    ["models"] = new JsonObject
                    {
                        [cfg.ModelId]           = Model(cfg.ModelId, "high", true),
                        [$"{cfg.ModelId}-fast"] = Model($"{cfg.ModelId} (no thinking)", "none", false)
                    }
                }
            },
            ["model"]       = $"qwen-gw/{cfg.ModelId}",
            // Titles and summaries are side requests on every session. At the
            // server's default xhigh they would each spend hundreds of output
            // tokens deliberating over a six-word title.
            ["small_model"] = $"qwen-gw/{cfg.ModelId}-fast",
            // Subagents are the only thing here that opens a second stream to
            // the node. Depth 1 lets a primary agent fan out and stops a
            // subagent from fanning out again, which is how a handful of
            // parallel requests becomes more than the node can decode.
            ["subagent_depth"] = 1,
            // A private endpoint. Sharing uploads the conversation elsewhere.
            ["share"] = "disabled",
            // Independent tool calls go out together instead of one per turn.
            // Every turn re-sends the whole conversation, so fewer turns is
            // fewer prefills \u2014 the dominant cost on a long session here.
            ["experimental"] = new JsonObject { ["batch_tool"] = true },
            ["compaction"] = new JsonObject
            {
                ["auto"] = true,
                // Both of these rewrite history, and rewritten history no longer
                // matches the prefix this node has cached: the next turn pays a
                // cold prefill (~3,200 tokens/s) instead of a cached one (1-2 s).
                // So compact as late as the window safely allows, and never
                // prune. 20,000 is the slack between the trigger and the
                // limit.input ceiling, for tool output that lands after it.
                ["prune"]    = false,
                ["reserved"] = 20_000
            },
            ["attachment"] = new JsonObject
            {
                ["image"] = new JsonObject
                {
                    // Resize rather than reject: OpenCode shrinks and re-encodes
                    // until the base64 fits, so an oversized screenshot still
                    // gets through instead of erroring at the gateway.
                    ["auto_resize"]      = true,
                    ["max_width"]        = ImageEdge,
                    ["max_height"]       = ImageEdge,
                    ["max_base64_bytes"] = ImageBudget(cfg.BodyLimit)
                }
            },
            // Effort per built-in agent. Subagents run in parallel and share the
            // node's decode bandwidth, so the ones doing legwork think less.
            ["agent"] = new JsonObject
            {
                ["plan"]    = new JsonObject { ["options"] = new JsonObject { ["reasoningEffort"] = "xhigh" } },
                ["build"]   = new JsonObject { ["options"] = new JsonObject { ["reasoningEffort"] = "high" } },
                ["general"] = new JsonObject { ["options"] = new JsonObject { ["reasoningEffort"] = "medium" } },
                ["explore"] = new JsonObject { ["options"] = new JsonObject { ["reasoningEffort"] = "low" } }
            }
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

    // ---- per-request records -------------------------------------------------
    //
    // Real rows, never SQL to paste. Both per-request tables — gateway.requests
    // (the access log: status, charge, request id) and engine.requests
    // (SGLang's record: cache split, timings, replica) — live in ClickHouse,
    // which is backend-only, and this bot stays edge-only. admin-mcp is
    // dual-homed and serves exactly these two reads on its bot port (:8081,
    // never proxied by Caddy) behind BOT_READ_SECRET. The SQL lives there, in
    // RequestSql, shared with its MCP tools.
    //
    // The join is exact: the gateway logs the response's `id` as chat_id, and
    // SGLang sets that id to its own request id, which is engine.requests.rid.
    private async Task<(JsonObject? Body, string? Error)> RecordsAsync(string path, CancellationToken ct)
    {
        if (cfg.RecordsSecret.Length < 32)
            return (null, "Per-request records are not connected: BOT_READ_SECRET is unset.");
        try
        {
            using var r = await http.CreateClient("records").GetAsync(path, ct);
            var text = await r.Content.ReadAsStringAsync(ct);
            var node = r.Content.Headers.ContentType?.MediaType == "application/json"
                ? JsonNode.Parse(text) as JsonObject : null;
            if (r.IsSuccessStatusCode && node is not null) return (node, null);
            return (null, node?["error"] is JsonValue ev && ev.TryGetValue<string>(out var e)
                ? "Records query failed: " + (e.Length > 200 ? e[..200] + "…" : e)
                : $"Records service answered HTTP {(int)r.StatusCode}.");
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException)
        {
            log.LogWarning(ex, "records read failed: {Path}", path);
            return (null, "Records service (admin-mcp) is unreachable.");
        }
    }

    // Row values arrive as ClickHouse's value text, or JSON null for SQL NULL.
    private static string? Col(JsonNode? row, string k) =>
        row?[k] is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;

    private static double? ColNum(JsonNode? row, string k) =>
        double.TryParse(Col(row, k), NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
        && double.IsFinite(d) ? d : null;

    // "13 Sep 17:28:03", UTC. The year is noise on a screen of recent requests.
    private static string When(string? ts) =>
        DateTime.TryParseExact(ts, "yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture,
                               DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var t)
            ? t.ToString("d MMM HH:mm:ss", CultureInfo.InvariantCulture)
            : Esc(ts ?? "—");

    // Same glyphs as /errors, with the refusal named: a bare code sends the
    // operator off to look up which plugin answers what.
    private static string StatusText(int status) => status switch
    {
        0 => "✂️ closed early",
        >= 200 and < 300 => $"✅ {status}",
        401 => "⛔ 401 bad key",
        403 => "⛔ 403 no balance",
        422 => "⚠️ 422 max_tokens",
        429 => "⏳ 429 rate limited",
        >= 500 => $"\U0001f534 {status}",
        _ => $"⚠️ {status}",
    };

    // One label/value line of a <pre> block: 28 columns, inside Fmt.PhoneCols.
    private static string Kv(string label, string value) => $"{label,-16}{value,12}\n";

    private static string NumOrDash(double? v) => v is { } d ? Fmt.Num(d) : "—";

    private async Task<Reply> TraceRequestAsync(string requestId, CancellationToken ct)
    {
        // Cheap sanity check. A mistyped id produces an empty result, which
        // reads like "the request did not happen" rather than "you typed it
        // wrong".
        var looksLikeId = requestId.Length is >= 8 and <= 64
            && requestId.All(c => char.IsAsciiLetterOrDigit(c) || c == '-');
        if (!looksLikeId)
            return new Reply($"<code>{Esc(Head(requestId))}</code> is neither a key name nor a request id.\n\n"
                 + "A request id is the UUID the gateway gives each request. /key → a consumer → Requests lists the latest ones.");

        var sb = new StringBuilder($"\U0001f50e <b>Request</b>\n<code>{Esc(requestId)}</code>\n");
        var (body, error) = await RecordsAsync($"bot/request/{Uri.EscapeDataString(requestId)}", ct);
        if (error is not null) return new Reply(sb.Append($"\n⚠️ {Esc(error)}").ToString());

        var g = body?["gateway"] is JsonArray { Count: > 0 } ga ? ga[0] : null;
        if (g is null)
            return new Reply(sb.Append("\nNo request with this id at the gateway.\n\n"
                + "Only traffic through the gateway is recorded; the direct hostname is not.").ToString());
        var e = body?["engine"] is JsonArray { Count: > 0 } ea ? ea[0] : null;
        var status = (int)(ColNum(g, "status") ?? -1);

        sb.Append($"\n<b>{Esc(Col(g, "consumer") ?? "?")}</b> · {StatusText(status)}\n")
          .Append($"{When(Col(g, "ts"))} UTC\n<pre>")
          .Append(Kv("input", NumOrDash(ColNum(g, "input_tokens"))))
          .Append(Kv("output", NumOrDash(ColNum(g, "output_tokens"))))
          .Append(Kv("charged", NumOrDash(ColNum(g, "total_tokens"))))
          .Append(Kv("gateway time", Fmt.Secs(ColNum(g, "duration_ms") / 1000)));
        if (Col(g, "response_flags") is { Length: > 0 } flags && flags != "-")
            sb.Append(Kv("envoy flags", Esc(flags)));
        sb.Append("</pre>");

        if (e is not null)
        {
            var prompt = ColNum(e, "prompt_tokens");
            var dev = ColNum(e, "cached_device");
            var host = ColNum(e, "cached_host");
            sb.Append($"\n<b>Inside the engine</b> · {Esc(Col(e, "replica") ?? "?")} · {Esc(Col(e, "finish_type") ?? "?")}\n<pre>")
              .Append(Kv("cached on GPU", NumOrDash(dev)))
              .Append(Kv("cached HiCache", NumOrDash(host)))
              .Append(Kv("computed", prompt is { } p && dev is { } d && host is { } h ? Fmt.Num(Math.Max(0, p - d - h)) : "—"))
              .Append(Kv("queue", Fmt.Secs(ColNum(e, "queue_s"))))
              .Append(Kv("first token", Fmt.Secs(ColNum(e, "ttft_s"))))
              .Append(Kv("prefill", Fmt.Secs(ColNum(e, "prefill_s"))))
              .Append(Kv("decode", Fmt.Secs(ColNum(e, "decode_s"))))
              .Append(Kv("engine total", Fmt.Secs(ColNum(e, "e2e_s"))))
              .Append(Kv("retractions", NumOrDash(ColNum(e, "num_retractions"))))
              .Append("</pre>");
        }
        else
        {
            sb.Append('\n').Append(status switch
            {
                0 => "No engine record: the client closed the connection before the response, and SGLang writes none for that.",
                >= 200 and < 300 => "No engine record: requests before 2026-09-13 19:46 UTC have none, and neither do the gateway's own quota calls.",
                _ => "Refused before the engine, so there is no engine record.",
            });
        }

        return new Reply(sb.ToString() + Fmt.Note(
            "Times are UTC. <b>charged</b> is what the balance was debited. <b>gateway time</b> is the whole "
          + "request as the gateway saw it; <b>engine total</b> is SGLang's part of it. <b>Cached</b> input was "
          + "not recomputed: GPU prefix hits are nearly free, HiCache hits are reloaded from host RAM. "
          + "The two records are joined exactly: the gateway logs the response id, which is the engine's request id."));
    }

    // ---- /key: pick a consumer, then read it --------------------------------
    //
    // The point of this command is that it takes NO arguments. Every other
    // per-consumer command needs a name typed correctly. Here the list is the
    // interface: tap a name, get its numbers, tap again for its per-request
    // records or for a written report.

    // Names reach PromQL as string literals and reach callback_data as a
    // suffix, so they are constrained at the door rather than escaped later.
    // The set here is what /newkey can produce.
    private static bool SafeName(string n) =>
        n.Length is > 0 and <= 40
        && n.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');

    // `prefix` is the callback a name button fires. The default is the key card,
    // but a command that named no consumer should land on ITS OWN screen after
    // the tap, not on the card with its screen one more tap away: /opencode with
    // no name asks which consumer, then sends that consumer's config.
    private async Task<Reply> KeyPickerAsync(
        CancellationToken ct, string prefix = "kc:24h:", string? ask = null)
    {
        var consumers = await keys.ReadConsumersAsync(ct);
        var names = consumers.Keys.Where(SafeName)
                             .OrderBy(k => k, StringComparer.Ordinal).ToArray();
        if (names.Length == 0)
            return new Reply("No consumers yet.\n\nCreate one with <code>/newkey &lt;name&gt;</code>.");

        return new Reply(
            $"\U0001f511 <b>Which consumer?</b> \u00b7 {names.Length}\n\n"
          + (ask ?? "Tap one for its numbers, settings, traces and report."),
            KeyPickerKeyboard(names, prefix));
    }

    // A name button per consumer. Two per row only while both names are short:
    // Telegram truncates a label to half the width, and "vkondratpev-demo2-c…"
    // is not a name anyone can tap with confidence.
    private static InlineKeyboardMarkup KeyPickerKeyboard(
        IEnumerable<string> consumers, string prefix = "kc:24h:")
    {
        var names = consumers.Where(SafeName).OrderBy(k => k, StringComparer.Ordinal).ToArray();
        var rows = new List<InlineKeyboardButton[]>();
        for (var i = 0; i < names.Length;)
        {
            if (i + 1 < names.Length && names[i].Length <= 14 && names[i + 1].Length <= 14)
            { rows.Add([Pick(names[i]), Pick(names[i + 1])]); i += 2; }
            else
            { rows.Add([Pick(names[i])]); i++; }
        }
        return new InlineKeyboardMarkup(rows.ToArray());

        InlineKeyboardButton Pick(string n) => new(n, prefix + n);
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
            "kt:" => await KeyRequestsCardAsync(rest, ct),
            "kr:" => KeyReportStart(rest, chatId, ct),
            "kp:" => await PolicyCardAsync(rest, null, ct),
            "kx:" => await ConnectAsync(rest, ct),
            "ko:" => await OpenCodeAsync(rest, chatId, ct),
            _     => new Reply("Unknown selection. Run /key again.")
        };
    }

    // Everything one screen can honestly say about a consumer.
    private readonly record struct KeyStats(
        double? Balance, double? Runway, double? Requests, double? NotOk,
        double? TokensIn, double? TokensOut, double? GatewayP95,
        double? Ttft, double? E2e, double? Queue, double? DecodeTps,
        CacheSplit Cache,
        double? Unbilled, double? UnbilledSeconds, double? Aborted, double? AbortedTokens,
        double? RateLimited, double? QuotaDenied,
        Dictionary<string, double> ByReplica);

    // Prefix cache for a set of requests, from exact token sums. Rates divide
    // by Known, the input of requests whose cache split was recorded: rows
    // backfilled from the gateway log (before 2026-09-14) carry tokens only.
    private readonly record struct CacheSplit(
        double Prompt, double Known, double Device, double Host, double PrefillSeconds)
    {
        public double Cached => Device + Host;
        public double? Hit => Known >= 1 ? Math.Clamp(Cached / Known, 0, 1) : null;
        public double? HostShare => Known >= 1 ? Math.Clamp(Host / Known, 0, 1) : null;
        // Share of ALL charged input that was cached, for pricing: input whose
        // split is unknown is priced as uncached rather than guessed.
        public double? CostHit => Prompt >= 1 && Known >= 1 ? Math.Clamp(Cached / Prompt, 0, 1) : null;
        public double? Coverage => Prompt >= 1 ? Math.Clamp(Known / Prompt, 0, 1) : null;
        // The prefill speed these requests actually got: input the engine did
        // compute over the seconds they spent in prefill, so batching and
        // HiCache reloads are in it. Needs enough cold work to mean anything.
        public double? PrefillRate =>
            PrefillSeconds >= 1 && Known - Cached >= 10_000 ? (Known - Cached) / PrefillSeconds : null;
        public double? AvoidedSeconds => PrefillRate is { } r && Cached >= 1 ? Cached / r : null;
    }

    private async Task<KeyStats> KeyStatsAsync(string name, string w, CancellationToken ct)
    {
        // SafeName has already guaranteed there is no quote in here.
        var ws = $"window=\"{w}\",consumer=\"{name}\"";
        var led = $"{{ai_consumer=\"{name}\"}}";
        Task<double?> G(string metric, string extra = "") => PromScalarAsync($"sum({metric}{{{ws}{extra}}})", ct);

        var balT  = PromScalarAsync($"consumer:quota_balance:tokens{led}", ct);
        var runT  = PromScalarAsync($"consumer:quota_days_left{led}", ct);
        var upT   = PromScalarAsync("max(up{job=\"engine-usage\"})", ct);
        // Engine: exact sums and percentiles over this key's finished requests.
        var reqT  = G("engine_usage_requests");
        var inT   = G("engine_usage_prompt_tokens");
        var outT  = G("engine_usage_completion_tokens");
        var knT   = G("engine_usage_cache_known_prompt_tokens");
        var devT  = G("engine_usage_cached_device_tokens");
        var hostT = G("engine_usage_cached_host_tokens");
        var pfT   = G("engine_usage_prefill_seconds");
        var abT   = G("engine_usage_aborted_requests");
        var abkT  = G("engine_usage_aborted_completion_tokens");
        var ttfT  = G("engine_usage_ttft_seconds", ",p=\"95\"");
        var e2eT  = G("engine_usage_e2e_seconds", ",p=\"95\"");
        var qT    = G("engine_usage_queue_seconds", ",p=\"95\"");
        var tpsT  = G("engine_usage_decode_tokens_per_second", ",p=\"50\"");
        var repT  = PromAsync($"sum by (replica) (engine_usage_replica_completion_tokens{{{ws}}})", ct, "replica");
        // Gateway: what the engine cannot see — refusals before it, requests
        // cut off before their usage frame (charged nothing), client latency.
        var badT  = G("gateway_usage_requests", ",status_class!=\"2xx\"");
        var gwT   = G("gateway_usage_duration_seconds", ",p=\"95\"");
        var ubT   = G("gateway_usage_cut_requests");
        var ubsT  = G("gateway_usage_cut_seconds");
        var rlT   = G("gateway_usage_rate_limited_requests");
        var qdT   = G("gateway_usage_quota_denied_requests");

        await Task.WhenAll(balT, runT, upT, reqT, inT, outT, knT, devT, hostT, pfT, abT, abkT,
                           ttfT, e2eT, qT, tpsT, repT, badT, gwT, ubT, ubsT, rlT, qdT);

        // A sum with no rows is absent, not NaN. While the scrape is up that
        // means zero; while it is down it means unknown, and prints as a dash.
        var live = upT.Result is > 0;
        double? Z(Task<double?> t) => t.Result ?? (live ? 0 : null);

        var cache = new CacheSplit(inT.Result ?? 0, knT.Result ?? 0, devT.Result ?? 0, hostT.Result ?? 0, pfT.Result ?? 0);
        return new KeyStats(balT.Result, runT.Result, Z(reqT), Z(badT),
                            Z(inT), Z(outT), gwT.Result,
                            ttfT.Result, e2eT.Result, qT.Result, tpsT.Result,
                            cache, Z(ubT), Z(ubsT), Z(abT), Z(abkT), Z(rlT), Z(qdT),
                            repT.Result);
    }

    private static string CacheLine(CacheSplit c, bool full)
    {
        if (c.Hit is not { } t) return "";
        var h = c.HostShare ?? 0;
        var sb = new StringBuilder($"♻️ Cache hit <b>{t * 100:0}%</b>");
        sb.Append(h > 0 ? $" · GPU {(t - h) * 100:0}% · HiCache {h * 100:0.#}%" : t > 0 ? " · all from GPU" : "");
        if (full && c.Cached >= 1)
        {
            sb.Append($"\n⏱ {Fmt.Num(c.Cached)} cached prompt tokens");
            if (c.AvoidedSeconds is { } secs)
                sb.Append($" ≈ {Fmt.Duration((long)secs)} of prefill avoided");
        }
        return sb.Append('\n').ToString();
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

    private async Task<string> PricesAsync(CancellationToken ct)
    {
        var p = await priceBook.GetAsync(ct);
        static string Row(PriceRef r) =>
            $"{r.Label,-11}{r.InPerM,6:0.###}{(r.CacheReadPerM is { } c ? c.ToString("0.###", CultureInfo.InvariantCulture) : "—"),7}{r.OutPerM,7:0.###}";
        string[] rows = [$"{"$ per 1M",-11}{"in",6}{"cached",7}{"out",7}", Row(p.OpenRouter), Row(p.AlibabaSg), Row(p.AlibabaBj)];

        // A worked example makes the spread concrete: one typical agent day is
        // input-heavy, and that is where the references disagree most.
        const double exIn = 30_000_000, exOut = 800_000, exHit = 0.9;
        return Table("\U0001f4b5 <b>Reference prices</b> · Qwen3.8-27B", rows)
             + "\n<b>One agent day</b> — 30M in, 0.8M out, 90% cached:\n"
             + $"OpenRouter <b>{PriceBook.Usd(PriceBook.Cost(p.OpenRouter, exIn, exOut))}</b> · with cache {PriceBook.Usd(PriceBook.Cost(p.OpenRouter, exIn, exOut, exHit))}\n"
             + $"Alibaba SG <b>{PriceBook.Usd(PriceBook.Cost(p.AlibabaSg, exIn, exOut))}</b> · BJ <b>{PriceBook.Usd(PriceBook.Cost(p.AlibabaBj, exIn, exOut))}</b>"
             + Fmt.Note(
                 "What the same tokens would cost bought elsewhere — a yardstick, not a bill. Used by /key, /top, /usage and the report.\n\n"
               + $"<b>OpenRouter</b> — {Esc(p.OpenRouter.Basis)}; the headline price for the model"
               + (p.Providers > 0 && p.OutMin is { } lo && p.OutMax is { } hi
                     ? $". Output ranges ${lo:0.##}–{hi:0.##} per M across {p.Providers} providers." : ".")
               + "\n<b>Alibaba Cloud</b> — Model Studio price page, Singapore (SG) and Beijing (BJ), as of 2026-09-12. "
               + "No API publishes these, so they are updated by hand in bot.cs.");
    }

    private async Task<Reply> KeyCardAsync(string name, string window, CancellationToken ct)
    {
        var statsT = KeyStatsAsync(name, window, ct);
        var pricesT = priceBook.GetAsync(ct);
        var tierT = ledger.TierAsync(name, ct);
        var ovT = ledger.OverridesAsync(name, ct);
        await Task.WhenAll(statsT, pricesT, tierT, ovT);
        var st = statsT.Result; var prices = pricesT.Result;
        var plan = PlanOf(tierT.Result, ovT.Result);

        static string N(double? v) => v is { } x && double.IsFinite(x) ? Fmt.Num(x) : "—";

        var sb = new StringBuilder($"\U0001f511 <b>{Esc(name)}</b> · {window}\n\n");

        // Money and runway first: it is what the card is opened for.
        sb.Append($"\U0001f4b0 Spendable <b>{N(st.Balance)}</b>");
        if (st.Runway is { } rw && double.IsFinite(rw))
            sb.Append(rw >= 3650 ? " · idle" : rw < 1 ? $" · ⏳ <b>{rw * 24:0}h</b> left" : $" · ⏳ {rw:0.#} days");
        sb.Append('\n');
        if (st.Balance is { } bv && double.IsFinite(bv))
            sb.Append($"\U0001f504 {RefillText(plan, (long)bv, false, DateTimeOffset.UtcNow)}\n");

        sb.Append($"\U0001f4e8 <b>{N(st.Requests)}</b> requests");
        if (st.NotOk is >= 0.5)
            sb.Append($" · ⚠️ {N(st.NotOk)} not 2xx")
              .Append(st.RateLimited is >= 0.5 ? $" ({N(st.RateLimited)} limited)" : "");
        if (st.Unbilled is >= 0.5) sb.Append($" · ✂️ {N(st.Unbilled)} cut");
        sb.Append('\n');
        sb.Append($"⬇ {N(st.TokensIn)} in · ⬆ {N(st.TokensOut)} out");
        if (st.TokensIn is { } ti && st.TokensOut is { } to && Fmt.Ratio(ti, to) is { Length: > 0 } ratio) sb.Append($" · {ratio}");
        sb.Append("\n\n");

        sb.Append($"\U0001f310 Gateway p95 <b>{Fmt.Secs(st.GatewayP95)}</b>\n");
        sb.Append($"⚙️ Engine p95 <b>{Fmt.Secs(st.E2e)}</b> · first token {Fmt.Secs(st.Ttft)}")
          .Append(st.DecodeTps is { } tps ? $" · {tps:0} tok/s" : "")
          .Append(st.Queue is >= 0.5 ? $" · queue {Fmt.Secs(st.Queue)}" : "").Append('\n');
        sb.Append(CacheLine(st.Cache, true));
        var total = st.ByReplica.Values.Sum();
        if (total > 0)
            sb.Append("\U0001f5a5 ").Append(string.Join(" · ",
                st.ByReplica.OrderBy(x => x.Key, StringComparer.Ordinal)
                  .Select(x => $"{ShortInstance(x.Key)} {x.Value / total * 100:0}%"))).Append('\n');

        if (st.TokensIn is { } cin && st.TokensOut is { } cout && cin + cout >= 0.5)
        {
            sb.Append("\n\U0001f4b5 <b>At reference prices</b>\n")
              .Append($"OpenRouter <b>{PriceBook.Usd(PriceBook.Cost(prices.OpenRouter, cin, cout))}</b>")
              .Append(prices.OpenRouter.CacheReadPerM is not null && st.Cache.CostHit is { } h2
                  ? $" · with cache {PriceBook.Usd(PriceBook.Cost(prices.OpenRouter, cin, cout, h2))}" : "")
              .Append($"\nAlibaba SG <b>{PriceBook.Usd(PriceBook.Cost(prices.AlibabaSg, cin, cout))}</b>")
              .Append($" · BJ <b>{PriceBook.Usd(PriceBook.Cost(prices.AlibabaBj, cin, cout))}</b>\n");
        }

        if (st.Unbilled is >= 0.5 || st.Aborted is >= 0.5)
            sb.Append($"\n✂️ <b>{N(Math.Max(st.Unbilled ?? 0, st.Aborted ?? 0))} request(s) cut off, charged nothing</b>")
              .Append(st.UnbilledSeconds is >= 0.5 and var cutS ? $" — {Fmt.Duration((long)cutS)} of wall time" : "")
              .Append(st.AbortedTokens is >= 0.5 ? $"; the engine generated {N(st.AbortedTokens)} output tokens for them" : "")
              .Append(".\n");

        var notes = new StringBuilder();
        // Which half of the stack each line came from. The two latencies
        // differ by the gateway filter chain, the router and two network hops,
        // and an operator comparing them needs to know that is expected.
        notes.Append("<b>Where each number comes from</b>\n"
                   + "Balance, runway: the ledger. Requests, tokens, engine latency, cache and the replica split: "
                   + "SGLang's own record of every request it finished — exact sums and percentiles, not estimates, "
                   + "up to ~2 minutes behind. Gateway p95, not-2xx and cut-offs: the gateway access log.\n"
                   + "Tokens are what the ledger charged; cut-off requests are counted apart.\n"
                   + "✂️ cut = client disconnect, stream timeout or upstream error before the final "
                   + "usage frame; the engine may have worked on it.\n\n");
        notes.Append($"<b>Reference prices</b> — not a bill. OpenRouter {Esc(PriceBook.PerMText(prices.OpenRouter))} "
                   + $"({Esc(prices.OpenRouter.Basis)}); Alibaba Cloud Singapore {Esc(PriceBook.PerMText(prices.AlibabaSg))}, "
                   + $"Beijing {Esc(PriceBook.PerMText(prices.AlibabaBj))}, as of 2026-09-12. "
                   + "\"with cache\" prices this key's cached input at the cached rate.\n\n"
                   + "<b>Cache hit</b> — share of prompt tokens the engine did not recompute. GPU: prefix still in "
                   + "GPU memory, nearly free. HiCache: prefix evicted from the GPU and reloaded from host RAM, "
                   + "~1.2 s per 45K tokens against ~14 s to recompute. The balance is NOT discounted: cached input "
                   + "is deducted like any input token. What a hit saves is engine time and latency; prefill avoided "
                   + "uses the prefill speed measured on this key's own requests in the window.");
        if (st.Cache.Coverage is < 0.99 && st.Cache.Known >= 1)
            notes.Append($"\n\nCache split known for {st.Cache.Coverage * 100:0}% of this window's input: requests "
                       + "before 2026-09-14 were recorded with tokens only.");
        if (st.Ttft is null && st.Requests > 0)
            notes.Append("\n\nNo engine timings in this window: its requests predate the engine's per-request "
                       + "records (2026-09-14).");
        sb.Append(Fmt.Note(notes.ToString()));

        string W(string w) => w == window ? $"• {w}" : w;
        var keyboard = new InlineKeyboardMarkup([
            [new InlineKeyboardButton(W("1h"), $"kw:1h:{name}"),
             new InlineKeyboardButton(W("24h"), $"kw:24h:{name}"),
             new InlineKeyboardButton(W("7d"), $"kw:7d:{name}")],
            [new InlineKeyboardButton("⚙️ Settings", $"kp:{name}"),
             new InlineKeyboardButton("\U0001f50e Requests", $"kt:{name}"),
             new InlineKeyboardButton("⚠️ Errors", $"ke:{window}:{name}")],
            [new InlineKeyboardButton("\U0001f4c4 Report", $"kr:{name}"),
             new InlineKeyboardButton("\U0001f50c Connect", $"kx:{name}"),
             new InlineKeyboardButton("\U0001f9e9 OpenCode", $"ko:{name}")],
            [new InlineKeyboardButton("← All keys", "kl:")]
        ]);
        return new Reply(sb.ToString(), keyboard);
    }

    // A key's latest requests, newest first, each with its engine record when
    // there is one. Real rows from admin-mcp — see "per-request records" above.
    private async Task<Reply> KeyRequestsCardAsync(string name, CancellationToken ct)
    {
        var keyboard = new InlineKeyboardMarkup([[new InlineKeyboardButton("← Key card", $"kc:24h:{name}")]]);
        var sb = new StringBuilder($"\U0001f50e <b>{Esc(name)}</b> · latest requests\n");

        var (body, error) = await RecordsAsync($"bot/requests/{Uri.EscapeDataString(name)}?limit=10", ct);
        if (error is not null) return new Reply(sb.Append($"\n⚠️ {Esc(error)}").ToString(), keyboard);
        if (body?["rows"] is not JsonArray { Count: > 0 } rows)
            return new Reply(sb.Append("\nNo requests in the last 7 days.").ToString(), keyboard);

        foreach (var row in rows)
        {
            var status = (int)(ColNum(row, "status") ?? -1);
            var input = ColNum(row, "input_tokens") ?? 0;
            var output = ColNum(row, "output_tokens") ?? 0;
            sb.Append($"\n<b>{When(Col(row, "ts"))}</b> · {StatusText(status)}\n")
              .Append($"{Fmt.Num(input)} in · {Fmt.Num(output)} out · {Fmt.Secs(ColNum(row, "duration_ms") / 1000)}");
            if (Col(row, "response_flags") is { Length: > 0 } flags && flags != "-")
                sb.Append($" · {Esc(flags)}");
            sb.Append('\n');

            // e.rid is empty when the LEFT JOIN found no engine record.
            if (Col(row, "rid") is { Length: > 0 })
            {
                var parts = new List<string> { Esc(Col(row, "replica") ?? "?") };
                if (ColNum(row, "cached_device") is { } dev && ColNum(row, "cached_host") is { } host && input >= 1)
                    parts.Add($"cache {Fmt.Pct((dev + host) / input)}");
                if (ColNum(row, "ttft_s") is { } ttft) parts.Add($"first token {Fmt.Secs(ttft)}");
                if (Col(row, "finish_type") is { Length: > 0 } fin && fin != "unknown") parts.Add(Esc(fin));
                sb.Append(string.Join(" · ", parts)).Append('\n');
            }
            sb.Append($"<code>{Esc(Col(row, "request_id") ?? "")}</code>\n");
        }

        sb.Append("\nTap an id to copy it, then send /trace &lt;id&gt; for that request in full.");
        return new Reply(sb.ToString() + Fmt.Note(
            "Newest first, last 7 days, times UTC. /v1/models calls are left out. <b>in</b> and <b>out</b> are "
          + "the tokens charged; the time is the whole request at the gateway. The line under it is the engine's "
          + "record: replica, the share of input served from cache (GPU or HiCache), time to first token and "
          + "why generation stopped. A request without one was refused before the engine, closed early, or is "
          + "from before 2026-09-13 19:46 UTC."), keyboard);
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
                + $"(cache-aware {PriceBook.Usd(PriceBook.Cost(prices.OpenRouter, i, o, st.Cache.CostHit))}) · "
                + $"Alibaba SG {PriceBook.Usd(PriceBook.Cost(prices.AlibabaSg, i, o))} · "
                + $"BJ {PriceBook.Usd(PriceBook.Cost(prices.AlibabaBj, i, o))}\n"
                : "";
        await SendDocumentAsync(chatId, $"{name}-{stamp}.html", Encoding.UTF8.GetBytes(html), "text/html",
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
            b.Append("engine_aborted_requests=").Append(Num(st.Aborted)).Append('\n');
            b.Append("engine_aborted_output_tokens=").Append(Num(st.AbortedTokens)).Append('\n');
            b.Append("rate_limited_429=").Append(Num(st.RateLimited)).Append('\n');
            b.Append("quota_denied_403=").Append(Num(st.QuotaDenied)).Append('\n');
            if (st.TokensIn is { } i && st.TokensOut is { } o)
            {
                b.Append("cost_usd_openrouter=").Append(PriceBook.UsdText(PriceBook.Cost(p.OpenRouter, i, o))).Append('\n');
                b.Append("cost_usd_openrouter_cache_aware=").Append(PriceBook.UsdText(PriceBook.Cost(p.OpenRouter, i, o, st.Cache.CostHit))).Append('\n');
                b.Append("cost_usd_alibaba_singapore=").Append(PriceBook.UsdText(PriceBook.Cost(p.AlibabaSg, i, o))).Append('\n');
                b.Append("cost_usd_alibaba_beijing=").Append(PriceBook.UsdText(PriceBook.Cost(p.AlibabaBj, i, o))).Append('\n');
            }
            b.Append("gateway_p95_s=").Append(Num(st.GatewayP95, 3)).Append('\n');
            b.Append("engine_e2e_p95_s=").Append(Num(st.E2e, 3)).Append('\n');
            b.Append("engine_ttft_p95_s=").Append(Num(st.Ttft, 3)).Append('\n');
            b.Append("engine_queue_p95_s=").Append(Num(st.Queue, 3)).Append('\n');
            b.Append("engine_decode_tokens_per_second_p50=").Append(Num(st.DecodeTps, 1)).Append('\n');
            b.Append("prefix_cache_hit=").Append(Num(st.Cache.Hit, 3)).Append('\n');
            b.Append("prefix_cache_hit_hicache_host=").Append(Num(st.Cache.HostShare, 3)).Append('\n');
            b.Append("cache_split_coverage=").Append(Num(st.Cache.Coverage, 3)).Append('\n');
            b.Append("cached_prompt_tokens=").Append(Num(st.Cache.Cached)).Append('\n');
            b.Append("prefill_tokens_per_second_measured=").Append(Num(st.Cache.PrefillRate)).Append('\n');
            b.Append("prefill_seconds_avoided_est=").Append(Num(st.Cache.AvoidedSeconds)).Append('\n');
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
        + "independent replicas behind a cache-aware router. DFlash2 speculative decoding and a "
        + "host-RAM prefix cache tier (HiCache). Single-stream output ~70 tok/s short context, "
        + "~65 tok/s at 55K; whole-node ceiling ~480 tok/s; engine "
        + "concurrency 8 (2 replicas x 4). Decode is memory-bandwidth bound. Measured cost: "
        + "an output token costs ~68x an uncached input token and ~4800x a cached one, so "
        + "prefix cache hit rate and the input:output ratio drive cost more than volume does. "
        + "Quota is a single total-token balance; input and output are charged the same. "
        + "Requests, tokens, cache and engine_* values are exact sums and percentiles over the engine's own "
        + "per-request records; cache_split_coverage below 1 means part of the window predates those records "
        + "(2026-09-14) and prefix_cache_hit covers only the rest. A HiCache hit reloads a prefix from host RAM "
        + "(~1.2 s per 45K tokens, against ~14 s to recompute); a GPU hit is nearly free. "
        + "engine_aborted_* are requests the engine's scheduler aborted (timeout, error): charged nothing; a client "
        + "disconnect leaves no engine record and shows only in cut_unbilled_*. "
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
        long chatId, string filename, byte[] bytes, string mime, string caption, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new StringContent(chatId.ToString(CultureInfo.InvariantCulture)), "chat_id");
        form.Add(new StringContent(caption), "caption");
        form.Add(new StringContent("HTML"), "parse_mode");
        var file = new ByteArrayContent(bytes);
        file.Headers.ContentType = new MediaTypeHeaderValue(mime);
        form.Add(file, "document", filename);

        using var r = await http.CreateClient("telegram").PostAsync("sendDocument", form, ct);
        if (!r.IsSuccessStatusCode)
            log.LogError("sendDocument failed: HTTP {Code} {Body}",
                (int)r.StatusCode, Head(await r.Content.ReadAsStringAsync(ct)));
    }

    // Rendered from Policy.All, the same table /tier validates against and
    // /policy resolves from, so the description and the thing being applied
    // cannot drift apart.
    private string TiersHelp()
    {
        var header = $"{"",-8}{"quota",6}{"refill",8}{"/day",5}{"/min",5}";
        var rows = Policy.All.Where(t => t.Name != "admin").Select(t =>
            $"{t.Name,-8}{Policy.Compact(t.Quota),6}{(t.Refill == RefillMode.Manual ? "manual" : "month"),8}"
          + $"{(t.Daily == 0 ? "∞" : Policy.Compact(t.Daily)),5}"
          + $"{(t.Tpm == 0 ? "∞" : Policy.Compact(t.Tpm)),5}");

        var sb = new StringBuilder(Table("\U0001f3f7 <b>Tiers</b> · defaults, tokens", new[] { header }.Concat(rows)));
        sb.Append('\n');
        foreach (var t in Policy.All)
            sb.Append($"\n<b>{t.Name}</b> — {Esc(t.For)}");
        sb.Append("\n\nEvery value is a default: change one for one key with its <b>Settings</b> buttons (/key) or "
                + "<code>/set</code>; <code>default</code> puts it back.");

        var notes = new StringBuilder("<b>What each setting means</b>\n");
        foreach (var f in Policy.Fields)
            notes.Append($"<b>{f.Key}</b> — {Esc(f.Meaning)}\n");
        notes.Append("\n<b>max_tokens defaults</b> (recorded, not enforceable per key): ")
             .Append(string.Join(", ", Policy.All.Where(t => t.Name != "admin").Select(t => $"{t.Name} {t.MaxTokens:N0}")))
             .Append($". The gateway holds one {cfg.OutputLimit:N0} ceiling for everyone.\n\n")
             .Append("<b>Enforced</b>: balance on every request; /day and /min at the gateway (429 with the reset time); "
                   + "refill at 00:00 UTC. A window opens at a key's first request, and one request can overshoot a "
                   + "limit by its own size.\n\n")
             .Append("<b>Refill replaces the balance</b>: unused tokens do not carry over. Turning refill on never resets "
                   + "at once; the first reset is the next boundary. Setting a tier or quota never moves a balance by itself.\n\n")
             .Append("<b>Quota is one number</b>: input and output are deducted alike, though an output token costs this "
                   + "node ~68× an uncached input token — /usage shows the split.");
        return sb.ToString() + Fmt.Note(notes.ToString());
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
                 + Table($"\u2699\ufe0f <b>{Esc(name)}</b> \u00b7 tier <b>{Esc(tier ?? "unassigned")}</b>", rows);

        var bal = balT.Result.TryGetValue(name, out var b) ? b.ToString("N0", CultureInfo.InvariantCulture) : "not seeded";
        body += $"\nSpendable now <code>{bal}</code> <i>(quota is only the refill target)</i>";

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
              + "\n<i>Tap a value to change it.</i>"
              + Fmt.Note("\u2731 marks a value set on this key; the rest follow its tier and move when the tier does.\n"
                       + "The 24h and 60s windows open at the key's first request, and one request can overshoot a limit by its own size.\n"
                       + "max_tokens is recorded only: the gateway cannot vary it per key.");

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

    // One row of window buttons under a report. The current window is marked
    // and tapping another EDITS the message, so flipping 1h -> 24h -> 7d is one
    // screen, not three.
    private static InlineKeyboardMarkup WindowButtons(string cmd, string current, params string[] windows) =>
        new([windows.Select(w => new InlineKeyboardButton(w == current ? $"• {w}" : w, $"w:{cmd}:{w}")).ToArray()]);

    private async Task<Reply?> WindowCallbackAsync(string data, CancellationToken ct)
    {
        var parts = data.Split(':');
        if (parts.Length != 3 || !ValidWindow(parts[2])) return null;
        return parts[1] switch
        {
            "top" => await TopAsync(parts[2], ct),
            "usage" => await UsageAsync(parts[2], ct),
            "errors" => await ErrorsAsync(parts[2], ct),
            _ => null
        };
    }

    // Who is using the node, most first. A card per consumer — name on its own
    // line — because a table column cannot hold both "testafter" and
    // "vkondratpev-demo2-cursor" and still fit a phone.
    private async Task<Reply> TopAsync(string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return new Reply(BadWindow(window));

        // Tokens and requests: exact sums over what the engine finished.
        // Errors, 401s and cut-offs: the gateway access log, the only place a
        // status code or a request that never reached the engine is counted.
        var inT   = UsageByConsumerAsync("engine_usage_prompt_tokens", window, ct, withDirect: true);
        var outT  = UsageByConsumerAsync("engine_usage_completion_tokens", window, ct, withDirect: true);
        var reqsT = UsageByConsumerAsync("engine_usage_requests", window, ct, withDirect: true);
        var errsT = UsageByConsumerAsync("gateway_usage_requests", window, ct, ",status_class=~\"4xx|5xx\"");
        var cutT  = UsageByConsumerAsync("gateway_usage_cut_requests", window, ct);
        var uaT   = PromScalarAsync($"sum(gateway_usage_unauthorized_requests{{window=\"{window}\",consumer=\"unauthenticated\"}})", ct);
        var warnT = UsageDataWarningAsync(ct);
        var pricesT = priceBook.GetAsync(ct);
        await Task.WhenAll(inT, outT, reqsT, errsT, cutT, uaT, warnT, pricesT);

        var inp = inT.Result; var outp = outT.Result; var reqs = reqsT.Result;
        var errs = errsT.Result; var cuts = cutT.Result; var prices = pricesT.Result;
        var keyboard = WindowButtons("top", window, "1h", "24h", "7d", "30d");

        double Tok(string n) => inp.GetValueOrDefault(n) + outp.GetValueOrDefault(n);
        var names = inp.Keys.Union(outp.Keys).Union(reqs.Keys).Union(errs.Keys)
                       .Where(n => n != "unauthenticated"
                                && (Tok(n) >= 0.5 || reqs.GetValueOrDefault(n) >= 0.5 || errs.GetValueOrDefault(n) >= 0.5))
                       .OrderByDescending(Tok).ThenByDescending(n => reqs.GetValueOrDefault(n))
                       .ToList();
        if (names.Count == 0)
            return new Reply($"\U0001f3c6 <b>Top consumers</b> · {window}\n{warnT.Result}\nNo traffic in this window.", keyboard);

        var total = names.Sum(Tok);
        var sb = new StringBuilder($"\U0001f3c6 <b>Top consumers</b> · {window}\n{warnT.Result}");
        sb.Append($"<b>{Fmt.Num(total)}</b> tokens · <b>{Fmt.Num(names.Sum(n => reqs.GetValueOrDefault(n)))}</b> requests\n");

        var rank = 0;
        foreach (var n in names)
        {
            rank++;
            var i = inp.GetValueOrDefault(n); var o = outp.GetValueOrDefault(n);
            var share = total > 0 ? Tok(n) / total : 0;
            var e = errs.GetValueOrDefault(n); var c = cuts.GetValueOrDefault(n);
            var cost = PriceBook.Cost(prices.OpenRouter, i, o);

            sb.Append($"\n<b>{rank}. {ConsumerTitle(n)}</b>\n");
            sb.Append($"{Fmt.ShareBar(share)} {Fmt.Pct(share)} · {Fmt.Num(Tok(n))} tok\n");
            sb.Append($"⬇ {Fmt.Num(i)} in · ⬆ {Fmt.Num(o)} out")
              .Append(Fmt.Ratio(i, o) is { Length: > 0 } ratio ? $" · {ratio}" : "").Append('\n');
            sb.Append($"{Fmt.Num(reqs.GetValueOrDefault(n))} req · ≈ {PriceBook.Usd(cost)}");
            if (e >= 0.5) sb.Append($" · ⚠️ {Fmt.Num(e)} err");
            if (c >= 0.5) sb.Append($" · ✂️ {Fmt.Num(c)} cut");
            sb.Append('\n');
        }

        if (uaT.Result is >= 0.5 and var ua)
            sb.Append($"\n\U0001f6ab {Fmt.Num(ua)} rejected with no valid key (401)\n");

        sb.Append(Fmt.Note(
            "<b>How to read it</b>\n"
          + "▰ bar and % — share of all charged tokens in the window.\n"
          + "⬇ in / ⬆ out, then input per output token. Both are charged the same, but input is "
          + "prefilled and mostly served from cache: a high ratio means the bill is context re-sent, not "
          + "work generated.\n"
          + $"≈ $ — OpenRouter list price for the same tokens ({Esc(PriceBook.PerMText(prices.OpenRouter))}); "
          + "a reference, not a bill. /usage shows all four references.\n"
          + "err — 4xx/5xx answers. cut — requests ended before their usage frame, charged nothing.\n"
          + "Tokens and requests are exact sums over the engine's per-request records; errors and cuts come "
          + "from the gateway access log.\n"
          + $"<b>{Esc(cfg.DirectHost)}</b> is the direct hostname: one shared key, no balance or limits, and no "
          + "access-log row, so it has no err/cut here — /errors shows its refusals."));
        return new Reply(sb.ToString(), keyboard);
    }

    // ---- errors -------------------------------------------------------------
    //
    // Why requests did not end in a full answer, by NAMED cause rather than
    // status class: "4xx" put a 400 from the engine next to a 429 from the
    // limiter, and a 504 stream timeout, a client that gave up in the queue and
    // one that left mid-answer were not errors at all to a status-class count.
    // The causes are classified once, in SQL, and the same expression drives
    // the gauge (gateway_usage_error_requests), admin-mcp's per-key rows and the
    // alerts, so the three cannot disagree. Order here is display order: what
    // the node did wrong first, then what the client did.
    private static readonly (string Cause, string Glyph, string Label)[] ErrorCauses =
    [
        ("timeout",           "⏱",  "timed out at the gateway (504)"),
        ("server_error",      "\U0001f534", "server error (5xx)"),
        ("left_before_reply", "\U0001f6aa", "client left before any reply"),
        ("left_mid_answer",   "✂️", "client left mid-answer"),
        ("cut_mid_answer",    "✂️", "stream cut mid-answer"),
        ("no_reply",          "⚫", "ended with no reply"),
        ("bad_request",       "❌", "rejected by the engine (400)"),
        ("max_tokens",        "\U0001f4cf", "max_tokens over the cap (422)"),
        ("too_large",         "\U0001f4e6", "body over the size cap (413)"),
        ("rate_limited",      "⏳", "limit reached (429)"),
        ("no_balance",        "⛔", "no balance (403)"),
        ("no_key",            "\U0001f6ab", "bad or missing key (401)"),
        ("not_found",         "❓", "unknown path (404)"),
        ("client_error",      "\U0001f7e0", "other 4xx"),
    ];

    private static (string Glyph, string Label) CauseText(string? cause)
    {
        foreach (var c in ErrorCauses)
            if (c.Cause == cause) return (c.Glyph, c.Label);
        return ("⚠️", Esc(cause ?? "unknown"));
    }

    private static readonly string ErrorCausesNote =
        "<b>Causes</b>\n"
      + "⏱ the gateway waited 900 s without a byte. \U0001f534 the router or engine failed.\n"
      + "\U0001f6aa the client closed the connection before the first byte — on this node that means it "
      + "gave up while queued or during a long prefill. ✂️ the answer had started and was cut, by the "
      + "client or by an upstream reset. Both are charged nothing.\n"
      + "❌ the engine refused the request before running it: usually a prompt plus max_tokens over the "
      + "169K context, or a parameter it does not accept. The engine logs no reason.\n"
      + "⏳ 429 is a key's token limit, or the router's queue being full. ⛔ \U0001f6ab \U0001f4cf are the "
      + "gateway's own refusals.\n"
      + "Counts are exact, from the gateway access log.";

    private static int WindowHours(string window) => window switch
    {
        "1h" => 1, "24h" => 24, "7d" => 168, _ => 720,
    };

    // The general report: every key's failures by cause, then the traffic the
    // access log cannot see (the direct hostname), then whether routing left a
    // queue on one replica while the other had room.
    private async Task<Reply> ErrorsAsync(string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return new Reply(BadWindow(window));

        var causesT = PromSeriesAsync(
            $"sum by (consumer,cause) (gateway_usage_error_requests{{window=\"{window}\"}})", ct);
        var totalsT = UsageByConsumerAsync("gateway_usage_requests", window, ct);
        var abortsT = UsageByConsumerAsync("engine_usage_aborted_requests", window, ct);
        var directAbortT = PromScalarAsync(
            $"sum(engine_usage_aborted_requests{{window=\"{window}\",consumer=\"\"}})", ct);
        // Caddy's per-host counters: approximate (increase() over a counter that
        // resets when Caddy restarts), but the only record of the direct route.
        var hostSel = $"host=\"{cfg.DirectHost}\"";
        var directT = PromSeriesAsync(
            $"sum by (code) (increase(caddy_http_request_duration_seconds_count{{{hostSel},code=~\"[45]..\"}}[{window}]))", ct);
        var directTotalT = PromScalarAsync(
            $"sum(increase(caddy_http_request_duration_seconds_count{{{hostSel}}}[{window}]))", ct);
        var strandedT = PromScalarAsync($"sum_over_time(node:engine_queue_stranded:bool[{window}]) * 15 / 60", ct);
        var warnT = UsageDataWarningAsync(ct);
        await Task.WhenAll(causesT, totalsT, abortsT, directAbortT, directT, directTotalT, strandedT, warnT);

        var by = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
        foreach (var (labels, value) in causesT.Result)
        {
            if (value < 0.5 || !labels.TryGetValue("consumer", out var c) || !labels.TryGetValue("cause", out var cause))
                continue;
            if (!by.TryGetValue(c, out var m)) by[c] = m = new(StringComparer.Ordinal);
            m[cause] = m.GetValueOrDefault(cause) + value;
        }
        foreach (var (c, n) in abortsT.Result)
            if (n >= 0.5 && !by.ContainsKey(c)) by[c] = new(StringComparer.Ordinal);

        var totals = totalsT.Result;
        var failed = by.Values.Sum(m => m.Values.Sum());
        var all = totals.Values.Sum();
        var sb = new StringBuilder($"\U0001f6a6 <b>Errors</b> · {window}\n{warnT.Result}");
        sb.Append(failed >= 0.5
            ? $"⚠️ <b>{Fmt.Num(failed)}</b> of {Fmt.Num(all)} gateway requests did not finish\n"
            : $"✅ Every one of {Fmt.Num(all)} gateway requests finished\n");

        var order = by.OrderByDescending(x => x.Value.Values.Sum())
                      .ThenBy(x => x.Key, StringComparer.Ordinal).ToList();
        foreach (var (name, m) in order)
        {
            var n = m.Values.Sum();
            var total = totals.GetValueOrDefault(name);
            sb.Append($"\n<b>{Esc(name)}</b>");
            if (total >= 0.5 && name != "unauthenticated")
                sb.Append($" · {Fmt.Pct(n / total)} of {Fmt.Num(total)}");
            sb.Append('\n');
            foreach (var (cause, glyph, label) in ErrorCauses)
                if (m.GetValueOrDefault(cause) is var v and >= 0.5)
                    sb.Append($"{glyph} {Fmt.Num(v)} {label}\n");
            if (abortsT.Result.GetValueOrDefault(name) is var ab and >= 0.5)
                sb.Append($"\U0001f6d1 {Fmt.Num(ab)} ended by the engine (abort)\n");
        }

        // The direct hostname: no key, no access-log row, no per-request list.
        var direct = directT.Result
            .Where(x => x.Value >= 0.5 && x.Labels.ContainsKey("code"))
            .OrderBy(x => x.Labels["code"], StringComparer.Ordinal).ToList();
        if (direct.Count > 0 || directAbortT.Result is >= 0.5)
        {
            sb.Append($"\n<b>{Esc(cfg.DirectHost)}</b> · direct, no key");
            if (directTotalT.Result is >= 0.5 and var dt) sb.Append($" · ≈{Fmt.Num(dt)} req");
            sb.Append('\n');
            foreach (var (labels, value) in direct)
            {
                var code = labels["code"];
                sb.Append(code == "401"
                    ? $"\U0001f6ab ≈{Fmt.Num(value)} wrong edge key (401)\n"
                    : $"{(code[0] == '5' ? "\U0001f534" : "\U0001f7e0")} ≈{Fmt.Num(value)} answered {Esc(code)}\n");
            }
            if (directAbortT.Result is >= 0.5 and var da)
                sb.Append($"\U0001f6d1 {Fmt.Num(da)} ended by the engine (abort)\n");
        }

        if (strandedT.Result is >= 1 and var st)
            sb.Append($"\n⏳ <b>{Fmt.Num(st)} min</b> queued on one replica, other had room\n");

        // A drill-down per key, two to a row. ke: is 3 + window + name, well
        // inside Telegram's 64-byte callback_data for a 40-character name.
        var rows = new List<InlineKeyboardButton[]>
        {
            new[] { "1h", "24h", "7d", "30d" }
                .Select(w => new InlineKeyboardButton(w == window ? $"• {w}" : w, $"w:errors:{w}")).ToArray(),
        };
        rows.AddRange(order.Where(x => SafeName(x.Key)).Take(8)
            .Select(x => new InlineKeyboardButton($"⚠️ {x.Key}", $"ke:{window}:{x.Key}"))
            .Chunk(2));

        sb.Append(Fmt.Note(
            ErrorCausesNote + "\n\n"
          + "\U0001f6d1 abort — the engine ended a request itself (timeout, error, or its client cancelled): "
          + "from the engine's records.\n"
          + $"<b>{Esc(cfg.DirectHost)}</b> skips the gateway, so it has no keys, causes or request list: its "
          + "lines are Caddy's status counters, approximate (≈) and reset when Caddy restarts. A client that "
          + "left early does not show there.\n"
          + "⏳ minutes when one replica had a queue and the other had none and a free slot: routing kept "
          + "sessions on the replica holding their cache. Counted since 2026-09-17.\n"
          + "Tap a key for its failed requests."));
        return new Reply(sb.ToString(), new InlineKeyboardMarkup(rows.ToArray()));
    }

    // One key: its causes, then the latest failed requests with ids for /trace.
    private async Task<Reply> KeyErrorsAsync(string name, string window, CancellationToken ct)
    {
        if (!ValidWindow(window)) return new Reply(BadWindow(window));
        if (!SafeName(name))
            return new Reply($"<code>{Esc(Head(name))}</code> is not a key name.\n\n"
                           + Usage("/errors [name] [1h|24h|7d|30d]", "/errors tim 7d"));

        var sel = $"window=\"{window}\",consumer=\"{name}\"";
        var causesT = PromSeriesAsync($"sum by (cause) (gateway_usage_error_requests{{{sel}}})", ct);
        var totalT = PromScalarAsync($"sum(gateway_usage_requests{{{sel}}})", ct);
        var abortT = PromScalarAsync($"sum(engine_usage_aborted_requests{{{sel}}})", ct);
        var rowsT = RecordsAsync(
            $"bot/errors/{Uri.EscapeDataString(name)}?hours={WindowHours(window)}&limit=10", ct);
        var warnT = UsageDataWarningAsync(ct);
        await Task.WhenAll(causesT, totalT, abortT, rowsT, warnT);

        string W(string w) => w == window ? $"• {w}" : w;
        var keyboard = new InlineKeyboardMarkup([
            new[] { "1h", "24h", "7d", "30d" }
                .Select(w => new InlineKeyboardButton(W(w), $"ke:{w}:{name}")).ToArray(),
            name == "unauthenticated"
                ? [new InlineKeyboardButton("← All errors", $"w:errors:{window}")]
                : [new InlineKeyboardButton("← Key card", $"kw:{window}:{name}"),
                   new InlineKeyboardButton("← All errors", $"w:errors:{window}")],
        ]);

        var causes = causesT.Result
            .Where(x => x.Value >= 0.5 && x.Labels.ContainsKey("cause"))
            .ToDictionary(x => x.Labels["cause"], x => x.Value, StringComparer.Ordinal);
        var failed = causes.Values.Sum();
        var sb = new StringBuilder($"⚠️ <b>{Esc(name)}</b> · errors · {window}\n{warnT.Result}");
        if (failed < 0.5)
            sb.Append(totalT.Result is >= 0.5 and var t0
                ? $"✅ All {Fmt.Num(t0)} requests finished\n"
                : "No gateway requests in this window.\n");
        else
        {
            sb.Append($"<b>{Fmt.Num(failed)}</b>");
            if (totalT.Result is >= 0.5 and var t && name != "unauthenticated")
                sb.Append($" of {Fmt.Num(t)} requests did not finish ({Fmt.Pct(failed / t)})\n\n");
            else
                sb.Append(" requests did not finish\n\n");
            foreach (var (cause, glyph, label) in ErrorCauses)
                if (causes.GetValueOrDefault(cause) is var v and >= 0.5)
                    sb.Append($"{glyph} {Fmt.Num(v)} {label}\n");
        }
        if (abortT.Result is >= 0.5 and var ab)
            sb.Append($"\U0001f6d1 {Fmt.Num(ab)} ended by the engine (abort)\n");

        var (body, error) = rowsT.Result;
        if (error is not null)
            sb.Append($"\n⚠️ {Esc(error)}\n");
        else if (body?["rows"] is JsonArray { Count: > 0 } rows)
        {
            sb.Append("\n<b>Latest</b>\n");
            foreach (var row in rows)
            {
                var (glyph, label) = CauseText(Col(row, "cause"));
                var status = (int)(ColNum(row, "status") ?? 0);
                // user_agent is whatever the client sent: escaped and cut to
                // a phone line, never trusted as markup.
                var ua = Col(row, "user_agent") is { Length: > 0 } u && u != "-"
                    ? u.Length > 28 ? u[..28] + "…" : u : null;
                sb.Append($"\n<b>{When(Col(row, "ts"))}</b> · {glyph} {label}");
                if (status > 0 && !label.Contains(status.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal))
                    sb.Append($" · {status}");
                sb.Append('\n')
                  .Append(Fmt.Secs(ColNum(row, "duration_ms") / 1000));
                if (ColNum(row, "input_tokens") is >= 1 and var inTok) sb.Append($" · {Fmt.Num(inTok)} in");
                if (ua is not null) sb.Append($" · {Esc(ua)}");
                // Where it came from, as Caddy saw it; empty before 2026-09-17.
                if (Col(row, "client_ip") is { Length: > 0 } ip) sb.Append($"\n\U0001f310 <code>{Esc(ip)}</code>");
                sb.Append($"\n<code>{Esc(Col(row, "request_id") ?? "")}</code>\n");
            }
            sb.Append("\nTap an id to copy it, then /trace &lt;id&gt; for that request in full.");
        }

        return new Reply(sb.ToString() + Fmt.Note(
            ErrorCausesNote + "\n\n"
          + "\U0001f6d1 abort — the engine ended a request itself; from its own records.\n"
          + "\U0001f310 is the client address as the edge saw it (recorded since 2026-09-17).\n"
          + $"<b>Latest</b> — newest first, up to 10, over the last {window}, times UTC. The time is the whole "
          + "request at the gateway: a few milliseconds is a refusal, minutes is a wait that ended badly. "
          + "Traffic on the direct hostname has no key and never appears here."), keyboard);
    }

    // Gateway and engine latency, per consumer, in one card each, so "is this
    // consumer slow at the gateway or inside the engine" is one glance.
    private async Task<string> LatencyAsync(string? name, string window, CancellationToken ct)
    {
        // Exact percentiles over every request in the window, computed in
        // ClickHouse — not histogram_quantile() over buckets, which was
        // approximate at low request counts and broke across replica rolls.
        var sel = name is null ? "" : $",consumer=\"{name}\"";
        Task<Dictionary<string, double>> Q(string metric, string p) =>
            UsageByConsumerAsync(metric, window, ct, $",p=\"{p}\"{sel}");

        var p50T = Q("gateway_usage_duration_seconds", "50");
        var p95T = Q("gateway_usage_duration_seconds", "95");
        var p99T = Q("gateway_usage_duration_seconds", "99");
        var e50T = Q("engine_usage_e2e_seconds", "50");
        var e95T = Q("engine_usage_e2e_seconds", "95");
        var t50T = Q("engine_usage_ttft_seconds", "50");
        var t95T = Q("engine_usage_ttft_seconds", "95");
        var q95T = Q("engine_usage_queue_seconds", "95");
        var tpsT = Q("engine_usage_decode_tokens_per_second", "50");
        var rowsT = UsageByConsumerAsync("engine_usage_engine_rows", window, ct, sel);
        var warnT = UsageDataWarningAsync(ct);
        var consumersT = keys.ReadConsumersAsync(ct);
        await Task.WhenAll(p50T, p95T, p99T, e50T, e95T, t50T, t95T, q95T, tpsT, rowsT, warnT, consumersT);

        // Current keys only (plus the 401 bucket): records outlive a revoked
        // consumer and every synthetic probe that ever sent a label.
        var current = consumersT.Result;
        var names = p95T.Result.Keys.Union(e95T.Result.Keys)
                       .Where(n => current.ContainsKey(n) || n == "unauthenticated")
                       .OrderByDescending(n => e95T.Result.GetValueOrDefault(n, p95T.Result.GetValueOrDefault(n)))
                       .ToList();
        if (names.Count == 0)
            return (name is null
                ? $"⏱ <b>Latency</b> · {window}\n{warnT.Result}\nNo requests in this window."
                : $"⏱ <b>Latency</b> · {Esc(name)} · {window}\n{warnT.Result}\nNo requests from this consumer in this window.");

        var sb = new StringBuilder("⏱ <b>Latency</b>" + (name is null ? "" : $" · {Esc(name)}") + $" · {window}\n{warnT.Result}");
        foreach (var n in names)
        {
            sb.Append($"\n<b>{Esc(n)}</b>\n");
            if (p95T.Result.TryGetValue(n, out var gw95))
                sb.Append($"\U0001f310 gateway p95 <b>{Fmt.Secs(gw95)}</b> · p50 {Fmt.Secs(p50T.Result.GetValueOrDefault(n))} · p99 {Fmt.Secs(p99T.Result.GetValueOrDefault(n))}\n");
            if (e95T.Result.TryGetValue(n, out var e95))
            {
                sb.Append($"⚙️ engine p95 <b>{Fmt.Secs(e95)}</b> · p50 {Fmt.Secs(e50T.Result.GetValueOrDefault(n))}\n");
                sb.Append($"⏱ first token p95 <b>{Fmt.Secs(t95T.Result.GetValueOrDefault(n))}</b> · p50 {Fmt.Secs(t50T.Result.GetValueOrDefault(n))}")
                  .Append(q95T.Result.TryGetValue(n, out var q95) ? $" · queue p95 {Fmt.Secs(q95)}" : "").Append('\n');
                sb.Append(tpsT.Result.TryGetValue(n, out var tps) ? $"⚡ decode {tps:0} tok/s median" : "⚡ decode —")
                  .Append($" · {Fmt.Num(rowsT.Result.GetValueOrDefault(n))} req\n");
            }
        }

        sb.Append(Fmt.Note(
            $"Over the last {window}; /p95 7d or /p95 &lt;name&gt; to change. Exact percentiles over every request, not buckets.\n"
          + "\U0001f310 <b>gateway</b> — the whole request as the gateway saw it, successful chat and completions requests.\n"
          + "⚙️ <b>engine</b> — measured inside SGLang, no gateway, router or network in it. First token is queue "
          + "wait plus prefill; queue is the wait before the first forward pass. Decode is each request's output "
          + "tokens over its decode seconds, median across requests; speculative decoding is in it.\n"
          + "Engine numbers exist for requests since 2026-09-14."));
        return sb.ToString();
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
                return "\U0001f6a8 <b>Alerts</b>\n\n\u2705 Nothing firing.";

            var items = new List<(string Sev, string Name, string Who, string Age, bool Suppressed)>();
            foreach (var a in arr)
            {
                var labels = a?["labels"];
                var sev = labels?["severity"]?.GetValue<string>() ?? "unknown";
                var name = labels?["alertname"]?.GetValue<string>() ?? "-";
                // Per-consumer rules (ConsumerQuotaLow, ...) carry the consumer and
                // no instance; showing "-" for them was the most useless line on
                // the screen.
                var who = labels?["ai_consumer"]?.GetValue<string>()
                          ?? labels?["consumer"]?.GetValue<string>()
                          ?? labels?["instance"]?.GetValue<string>()
                          ?? labels?["job"]?.GetValue<string>() ?? "";
                var age = DateTimeOffset.TryParse(
                              a?["startsAt"]?.GetValue<string>(), CultureInfo.InvariantCulture,
                              DateTimeStyles.AdjustToUniversal, out var st)
                          ? Fmt.Age(DateTimeOffset.UtcNow - st) : "-";
                var state = a?["status"]?["state"]?.GetValue<string>();
                items.Add((sev, name, who, age,
                    !string.Equals(state, "active", StringComparison.Ordinal)));
            }

            var sb = new StringBuilder();
            sb.Append("\U0001f6a8 <b>Alerts</b> \u00b7 ").Append(items.Count)
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
                sb.Length--;                       // no blank line before </pre>
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
        var work = DispatchAsync(userId, chatId, text, ct);
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
        : _lastOk is { } t ? $"{(_limited == 0 ? "no keys" : _limited == 1 ? "1 key" : $"{_limited} keys")} limited \u00b7 synced {Fmt.Duration((long)(DateTimeOffset.UtcNow - t).TotalSeconds)} ago"
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

    // Every reply is HTML parse_mode, where a bare '<' opens a tag: "<$0.01"
    // made Telegram refuse the whole /usage and /top reply with "Unsupported
    // start tag" (2026-09-13). UsdText is the plain form for the report facts.
    public static string Usd(decimal v) => v switch
    {
        0m => "$0",
        < 0.01m => "&lt;$0.01",
        < 100m => "$" + v.ToString("0.00", CultureInfo.InvariantCulture),
        < 10_000m => "$" + v.ToString("N0", CultureInfo.InvariantCulture),
        _ => "$" + (v / 1000m).ToString("0.#", CultureInfo.InvariantCulture) + "K",
    };

    public static string UsdText(decimal v) => v is > 0m and < 0.01m ? "<$0.01" : Usd(v);

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
        // Every key, the limiter's window counters included. Otherwise a
        // consumer re-created under the same name silently inherits the revoked
        // one's tier, hand-set limits and up to a day of spent daily limit
        // (found by the lifecycle test, 2026-09-15).
        await c.CommandAsync(ct, "DEL", Prefix + name, TierPrefix + name, PolicyPrefix + name, RefillPrefix + name,
                             LimiterSync.CounterKey(name, 86_400), LimiterSync.CounterKey(name, 60));
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
        var chunks = Chunk(Tidy(reply.Text), 3500).ToList();
        for (var i = 0; i < chunks.Count; i++)
        {
            var last = i == chunks.Count - 1;
            var payload = new SendMessage(chatId, chunks[i], "HTML", last ? reply.Keyboard : null);
            using var content = new StringContent(
                JsonSerializer.Serialize(payload, BotJson.Default.SendMessage), Encoding.UTF8);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r = await http.CreateClient("telegram").PostAsync("sendMessage", content, ct);
            if (r.IsSuccessStatusCode) continue;
            var body = await r.Content.ReadAsStringAsync(ct);
            log.LogError("sendMessage failed: HTTP {Code} {Body}", (int)r.StatusCode, body);

            // One bad character used to cost the operator the whole reply with
            // nothing on screen. On an HTML parse refusal, resend the same chunk
            // as plain text: uglier, but the numbers arrive and the log names
            // the bug.
            if (!body.Contains("can't parse entities", StringComparison.Ordinal)) continue;
            var plain = new SendMessage(chatId, PlainText(chunks[i]), null, last ? reply.Keyboard : null);
            using var retry = new StringContent(
                JsonSerializer.Serialize(plain, BotJson.Default.SendMessage), Encoding.UTF8);
            retry.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var r2 = await http.CreateClient("telegram").PostAsync("sendMessage", retry, ct);
            if (!r2.IsSuccessStatusCode)
                log.LogError("plain-text resend failed: HTTP {Code} {Body}",
                    (int)r2.StatusCode, await r2.Content.ReadAsStringAsync(ct));
        }
    }

    // The HTML reply with its tags removed and entities decoded, for the
    // parse-failure fallback above.
    private static string PlainText(string html)
    {
        var sb = new StringBuilder(html.Length);
        for (var i = 0; i < html.Length; i++)
        {
            var tag = html[i] == '<' && i + 1 < html.Length
                && (char.IsAsciiLetter(html[i + 1]) || html[i + 1] == '/');
            var end = tag ? html.IndexOf('>', i) : -1;
            if (end > i) { i = end; continue; }
            sb.Append(html[i]);
        }
        return WebUtility.HtmlDecode(sb.ToString());
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
            var payload = new EditMessageText(chatId, messageId, Tidy(reply.Text), "HTML", reply.Keyboard);
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

    // At most one blank line anywhere, and none at the ends. Screens are built
    // from parts that each end in a newline, and the joins showed up as double
    // gaps before the collapsed notes — measured in the reply capture.
    private static string Tidy(string s)
    {
        while (s.Contains("\n\n\n", StringComparison.Ordinal)) s = s.Replace("\n\n\n", "\n\n");
        return s.Trim('\n');
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
    // ---- Telegram layout rules ---------------------------------------------
    //
    // Measured by capturing every reply through a fake Bot API (2026-09-13):
    // 14 of 29 replies had <pre> lines wider than a phone shows, and every
    // table with a name column broke on a 24-character consumer name, running
    // the numbers together ("vkondratyev-demo40,606,3021,254,824"). So:
    //
    //   - a consumer NAME never sits in a padded column. It gets its own bold
    //     line and its numbers go underneath, which holds for any name length;
    //   - numbers are compact (40.6M, 18.3K) — three significant digits is all
    //     a phone screen can compare at a glance;
    //   - a <pre> table is only for fixed-width label/value pairs, at most
    //     PhoneCols characters wide;
    //   - the "why" goes last, in an expandable blockquote, so the numbers are
    //     the first screen and the explanation is one tap away instead of a
    //     wall of italics under every answer.
    public const int PhoneCols = 32;

    public static string Num(double v)
    {
        if (!double.IsFinite(v)) return "\u2014";
        var a = Math.Abs(v);
        string S(double x, string unit) =>
            (x >= 100 ? x.ToString("0", CultureInfo.InvariantCulture)
             : x >= 10 ? x.ToString("0.#", CultureInfo.InvariantCulture)
             : x.ToString("0.##", CultureInfo.InvariantCulture)) + unit;
        // Unit boundaries at the rounding edge, so 999,938 reads 1M, not 1000K.
        return a switch
        {
            >= 999.5e6 => S(v / 1e9, "B"),
            >= 999.5e3 => S(v / 1e6, "M"),
            >= 999.5 => S(v / 1e3, "K"),
            _ => Math.Round(v).ToString("0", CultureInfo.InvariantCulture),
        };
    }

    // 47ms, 4.1s, 181s.
    public static string Secs(double? s) => s switch
    {
        null => "\u2014",
        { } v when !double.IsFinite(v) => "\u2014",
        < 1 => $"{s.Value * 1000:0}ms",
        < 10 => $"{s.Value:0.0}s",
        _ => $"{s.Value:0}s",
    };

    // Input:output as the larger side over 1 — "32:1" for context-heavy
    // traffic, "1:4" for generative — so it never reads "0:1".
    public static string Ratio(double input, double output) =>
        input < 1 || output < 1 ? "" : input >= output ? $"{input / output:0}:1" : $"1:{output / input:0}";

    public static string Pct(double share) =>
        share <= 0 ? "0%" : share < 0.01 ? "&lt;1%" : $"{share * 100:0}%";   // HTML: see PriceBook.Usd

    // A share as ten cells, for a line of proportional text. ▰/▱ are the same
    // width as each other in every Telegram client font, which is all it needs.
    public static string ShareBar(double share, int cells = 10)
    {
        var n = (int)Math.Round(Math.Clamp(share, 0, 1) * cells);
        if (share > 0 && n == 0) n = 1;
        return new string('\u25b0', n) + new string('\u25b1', cells - n);
    }

    // The explanation, collapsed. Telegram shows the first lines and a tap
    // opens the rest; the HTML inside may use b/i/code.
    public static string Note(string html) => $"\n\n<blockquote expandable>{html}</blockquote>";

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
            Who: Get(a.Labels, "ai_consumer") ?? Get(a.Labels, "consumer") ?? Get(a.Labels, "instance") ?? Get(a.Labels, "job") ?? "-",
            Age: a.StartsAt is { } st
                 ? Fmt.Age((firing ? DateTimeOffset.UtcNow : a.EndsAt ?? DateTimeOffset.UtcNow) - st)
                 : "-")).ToList();

        // One line per target. A padded <pre> column broke on long instance
        // names (host:port) exactly as the consumer tables did.
        if (rows.Count > 0)
        {
            sb.Append('\n');
            foreach (var r in rows)
                sb.Append("\u2022 <code>").Append(Fmt.Esc(r.Who)).Append("</code> \u00b7 ").Append(Fmt.Esc(r.Age)).Append('\n');
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
    string ModelId, int ContextLimit, int OutputLimit, int BodyLimit,
    string AlertSecret, string AdminApiSecret, string RecordsUrl, string RecordsSecret,
    string AlertmanagerUrl, string DirectHost, HashSet<long> AlertChatIds);

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
    [property: JsonPropertyName("parse_mode")] string? ParseMode = "HTML",
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
