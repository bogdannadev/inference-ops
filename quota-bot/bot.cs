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
    PublicBaseUrl:   Opt("PUBLIC_BASE_URL", "https://qw38-27b-gw.duckdns.org").TrimEnd('/'),
    KubeConfigPath:  Opt("KUBECONFIG_PATH", "/etc/kube/config"),
    ConsumersPath:   Opt("CONSUMERS_PATH", "/data/consumers.conf"),
    AuditPath:       Opt("AUDIT_PATH", "/data/audit.log"),
    ModelId:         Opt("MODEL_ID", "qwen3.8-27b"),
    ContextLimit:    int.Parse(Opt("MODEL_CONTEXT", "169000"), CultureInfo.InvariantCulture),
    OutputLimit:     int.Parse(Opt("MODEL_OUTPUT", "70000"), CultureInfo.InvariantCulture),
    AlertSecret:     Req("ALERT_WEBHOOK_SECRET"),
    AlertmanagerUrl: Opt("ALERTMANAGER_URL", "http://qwen36-27b-alertmanager:9093").TrimEnd('/'),
    AlertChatIds:    alertChatIds,
    LangfuseUrl:     Opt("LANGFUSE_PUBLIC_URL", "https://qw38-27b-langfuse.duckdns.org").TrimEnd('/'));

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
        var reply = await DispatchWithTypingAsync(msg.Chat.Id, msg.From.Id, text.Trim(), ct);
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
        var token = data.Length > 3 ? data[3..] : "";
        _pending.TryRemove(token, out var p);

        string text;
        if (p is null) text = "That confirmation is no longer valid. Run the command again.";
        // The token is bound to the user who armed it, so one operator cannot
        // confirm another's pending destructive action from a shared screen.
        else if (p.UserId != cb.From.Id) text = "That confirmation belongs to someone else.";
        else if (DateTimeOffset.UtcNow > p.Expires) text = "Confirmation expired. Nothing was changed.";
        else if (!data.StartsWith("ok:", StringComparison.Ordinal)) text = "Cancelled. Nothing was changed.";
        else text = await p.Run(ct);

        await AnswerCallbackAsync(cb.Id, ct);
        await SendAsync(chatId.Value, new Reply(text), ct);
    }

    private async Task<Reply> DispatchAsync(long userId, string text, CancellationToken ct)
    {
        var parts = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cmd = parts[0].Split('@')[0].ToLowerInvariant();
        var a1 = parts.Length > 1 ? parts[1] : null;
        var a2 = parts.Length > 2 ? parts[2] : null;

        return cmd switch
        {
            "/start" or "/help" => new Reply(Help()),
            "/status"     => new Reply(await StatusAsync(ct)),
            "/keys"       => new Reply(await KeysAsync(ct)),
            "/balance"    => new Reply(await BalanceAsync(a1, ct)),
            "/usage"      => new Reply(await UsageAsync(a1 ?? "24h", ct)),
            "/alerts"     => new Reply(await AlertsAsync(ct)),
            "/health"     => new Reply(await HealthAsync(ct)),
            "/top"        => new Reply(await TopAsync(a1 ?? "24h", ct)),
            "/p95"        => new Reply(await LatencyAsync(a1, ct)),
            "/errors"     => new Reply(await ErrorsAsync(a1 ?? "24h", ct)),
            "/tiers"      => new Reply(TiersHelp()),
            "/langfuse"   => new Reply(LangfuseHelp()),
            "/trace"      => new Reply(a1 is null
                                 ? Usage("/trace &lt;request-id&gt;", "/trace efd62f54-d5e3-9fe3-be99-b3945d617414")
                                 : TraceHelp(a1)),
            "/tier"       => new Reply((a1 is null || a2 is null)
                                 ? Usage("/tier &lt;name&gt; &lt;trial|team|service|batch|admin&gt;", "/tier acme service")
                                 : await TierAsync(a1, a2, ct)),
            "/opencode"   => new Reply(a1 is null ? Usage("/opencode &lt;name&gt;", "/opencode acme") : await OpenCodeAsync(a1, ct)),
            "/newkey"     => new Reply(a1 is null ? Usage("/newkey &lt;name&gt; [quota]", "/newkey acme 1000000") : await NewKeyAsync(a1, a2, ct)),
            "/topup"      => new Reply((a1 is null || a2 is null) ? Usage("/topup &lt;name&gt; &lt;tokens&gt;", "/topup acme 500000") : await TopUpAsync(a1, a2, ct)),
            "/setquota"   => (a1 is null || a2 is null) ? new Reply(Usage("/setquota &lt;name&gt; &lt;tokens&gt;", "/setquota acme 1000000")) : Arm(userId, a1, a2, PendingKind.SetQuota),
            "/clearquota" => a1 is null ? new Reply(Usage("/clearquota &lt;name&gt;", "/clearquota acme")) : Arm(userId, a1, "0", PendingKind.SetQuota),
            "/revoke"     => a1 is null ? new Reply(Usage("/revoke &lt;name&gt;", "/revoke acme")) : Arm(userId, a1, null, PendingKind.Revoke),
            _             => new Reply($"Unknown command <code>{Esc(Head(cmd))}</code>.\n\nSend /help to see what this bot can do.")
        };
    }

    private static string Help() =>
        """
        <b>Gateway access management</b>

        <b>Who and how much</b>
        /keys — consumers, balances and tiers
        /balance [name] — balance, burn rate and runway
        /tiers — what each policy tier means
        /tier &lt;name&gt; &lt;tier&gt; — record a consumer's tier

        <b>What they used</b>
        /usage [1h|24h|7d|30d] — tokens in/out per consumer
        /top [1h|24h|7d] — busiest consumers, with errors
        /p95 [name] — latency percentiles, per consumer
        /errors [1h|24h|7d] — status mix per consumer
        /trace &lt;request-id&gt; — where to look one request up
        /langfuse — what Langfuse can and cannot tell you

        <b>Is it healthy</b>
        /status — can I still operate the gateway
        /health — is the stack healthy, and can I believe it
        /alerts — what is firing right now

        <b>Grant</b>
        /newkey &lt;name&gt; [tokens] — create a key, seed it, return its OpenCode config
        /opencode &lt;name&gt; — re-send an existing consumer's config
        /topup &lt;name&gt; &lt;tokens&gt; — add to a balance

        <b>Destructive — these ask first</b>
        /setquota &lt;name&gt; &lt;tokens&gt; — <i>replaces</i> a balance
        /clearquota &lt;name&gt; — sets a balance to zero
        /revoke &lt;name&gt; — deletes a key and its balance

        Credentials are shown once, by /newkey. /keys lists names only.
        Quota is a single TOTAL-token balance — input and output are charged
        the same. See /tiers.
        """;

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
            $"{"key-auth",-12}{api}"
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
        await Task.WhenAll(balancesT, tiersT, consumersT);
        var balances = balancesT.Result; var tiers = tiersT.Result; var consumers = consumersT.Result;

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
            return $"{name,-16}{bal,14}  {tier}";
        });

        var untiered = consumers.Keys.Count(n => !tiers.ContainsKey(n));
        var body = Table($"<b>Consumers</b> ({consumers.Count})", new[] { header }.Concat(rows))
                 + "\nCredentials are not shown. Use /opencode &lt;name&gt;.";
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
        await Task.WhenAll(inT, outT, reqsT);

        var inp = inT.Result; var outp = outT.Result; var reqs = reqsT.Result;
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

        var totalIn = inp.Values.Sum(); var totalOut = outp.Values.Sum();
        return Table($"<b>Usage</b> \u2014 last {window}", new[] { header }.Concat(rows))
             + $"\n<b>{totalIn + totalOut:N0}</b> tokens charged \u2014 {totalIn:N0} in, {totalOut:N0} out."
             + "\n<i>Quota is a single TOTAL-token balance: input and output are deducted at the same "
             + "rate. i:o shows how much of a bill is context re-sent rather than tokens generated.</i>"
             + "\n<i>Counters reset when the gateway restarts. Balances are the billing record.</i>";
    }

    // ---- key lifecycle ----------------------------------------------------

    private async Task<string> NewKeyAsync(string name, string? quotaArg, CancellationToken ct)
    {
        if (!IsValidName(name))
            return $"<code>{Esc(name)}</code> is not a valid name.\n\nUse 1\u201332 characters: lowercase letters, digits, <code>-</code> or <code>_</code>.";
        var existing = await keys.ReadConsumersAsync(ct);
        if (existing.ContainsKey(name))
            return $"<b>{Esc(name)}</b> already exists.\n\nUse <code>/opencode {Esc(name)}</code> to re-send its config, or <code>/revoke {Esc(name)}</code> to replace it.";

        var quota = 1_000_000L;
        if (quotaArg is not null && !TryParseTokens(quotaArg, out quota))
            return $"<code>{Esc(quotaArg)}</code> is not a token count.\n\nGive a whole number, like <code>1000000</code>.";

        var credential = "Bearer sk-" + Base62(32);
        await keys.AddAsync(name, credential, ct);

        // Seed BEFORE announcing success. ai-quota returns the same 403 for
        // "never seeded" as for "exhausted", so an unseeded key looks broken in
        // a way that wastes an afternoon.
        await QuotaSetAsync(name, quota, ct);
        await AuditAsync($"newkey name={name} quota={quota}", ct);

        // Tell them it is shown once BEFORE the block, so the warning is not
        // below the fold on a phone.
        return $"Created <b>{Esc(name)}</b> with <code>{quota:N0}</code> tokens.\n\n"
             + "\u26a0\ufe0f <b>This credential is shown once.</b> Tap the block to copy it.\n\n"
             + $"<pre>{Esc(OpenCodeJson(credential))}</pre>\n"
             + $"Save as <code>~/.config/opencode/opencode.json</code>.";
    }

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
                    return $"<b>{Esc(name)}</b> balance set to <code>{target:N0}</code> (was {before?.ToString("N0", CultureInfo.InvariantCulture) ?? "unset"}).";
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
                    if (!await keys.RemoveAsync(name, ct)) return $"No consumer named <b>{Esc(name)}</b>.";
                    await ledger.DeleteAsync(name, ct);
                    await AuditAsync($"revoke name={name}", ct);
                    return $"Revoked <b>{Esc(name)}</b>. The key no longer authenticates and the balance is gone.";
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

    private async Task QuotaSetAsync(string name, long value, CancellationToken ct)
    {
        var body = new FormUrlEncodedContent([
            new KeyValuePair<string, string>("consumer", name),
            new KeyValuePair<string, string>("quota", value.ToString(CultureInfo.InvariantCulture))
        ]);
        using var r = await http.CreateClient("gateway").PostAsync("v1/chat/completions/quota/refresh", body, ct);
        r.EnsureSuccessStatusCode();
    }

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

    // The draft tiers from docs/KEY-TIERS.md, sized against measurements taken
    // on this node: an output token costs ~68x an uncached input token, the
    // engine sustains ~387 output tok/s, and concurrency is 8.
    //
    // Quota and TPM are RECORDED, not enforced. ai-quota deducts a flat
    // input+output total and cannot vary by tier; ai-token-ratelimit is bundled
    // but not installed. Writing the intent down is what makes it reviewable
    // and is the prerequisite for enforcing it later — it is not the enforcement.
    private static readonly Dictionary<string, (long Quota, int Tpm, int MaxTokens, string For)> Tiers =
        new(StringComparer.Ordinal)
        {
            ["trial"]   = (   100_000,   3_000,  2_048, "evaluation, unvetted third parties"),
            ["team"]    = (10_000_000,  60_000, 32_768, "internal humans via OpenCode"),
            ["service"] = (50_000_000, 120_000, 16_384, "production integrations"),
            ["batch"]   = (100_000_000, 30_000, 70_000, "offline, latency-tolerant"),
            ["admin"]   = (         0,       0,      0, "management only, never inference"),
        };

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

        <b>Do NOT use it for</b>
        • <b>Billing or usage totals.</b> No model pricing is configured, so
          cost is meaningless, and token sums are derived from spans rather
          than the ledger. Use /usage, /top and /balance.
        • <b>Per-user or per-session views.</b> They are empty. Identity sits
          on the gateway span, tokens sit on the engine span, and the router
          starts a new trace between them — Langfuse never sees them together.
        • <b>Node health.</b> That is Grafana and /health.

        <b>If a Langfuse doc page 404s the API</b>
        This runs 4.5.0 in <code>events_only</code> mode. The v3 endpoints are
        gone by design and return a message saying so. Use
        <code>/api/public/v2/observations</code> and
        <code>/api/public/v2/metrics</code>.

        <b>One number used to lie</b>
        Until 2026-09-05 its token aggregate read ~101M per day against a real
        345k, because it counted per-decode-iteration spans as usage. Those are
        dropped at the collector now, which also removed 77% of span volume.
        Numbers before that date in Langfuse are not trustworthy.

        Full map of which store answers what: <code>docs/METRICS-ECOSYSTEM.md</code>
        """;

    // Rendered from the same Tiers table the /tier command validates against,
    // so the description and the thing being applied cannot drift apart.
    private static string TiersHelp()
    {
        var header = $"{"tier",-9}{"quota",12}{"tok/min",9}{"max_tok",9}";
        var rows = Tiers.Where(t => t.Key != "admin").Select(t =>
            $"{t.Key,-9}{t.Value.Quota,12:N0}{t.Value.Tpm,9:N0}{t.Value.MaxTokens,9:N0}");

        var body = Table("<b>Policy tiers</b>", new[] { header }.Concat(rows));
        foreach (var t in Tiers)
            body += $"\n<b>{t.Key}</b> \u2014 {Esc(t.Value.For)}";

        return body
          + "\n\n<b>Recorded, not enforced.</b> Nothing reads a tier at request time yet: "
          + "ai-quota charges a flat input+output total and cannot vary by consumer tier, and "
          + "rate limiting needs ai-token-ratelimit, which ships in the gateway image but is "
          + "not installed. /tier writes the intent down so it is reviewable."
          + "\n\n<b>Quota is one number.</b> Input and output are deducted at the same rate, "
          + "though on this node an output token costs roughly 68\u00d7 an uncached input token "
          + "and ~4800\u00d7 a cached one. A consumer re-sending long context can therefore burn "
          + "quota far faster than the work it asks for \u2014 /usage and /top show the i:o ratio."
          + "\n\n<i>Setting a tier does not change a balance. /tier says what to run if you "
          + "want them aligned.</i>";
    }

    private async Task<string> TierAsync(string name, string tier, CancellationToken ct)
    {
        tier = tier.ToLowerInvariant();
        if (!Tiers.TryGetValue(tier, out var t))
            return $"Unknown tier <code>{Esc(tier)}</code>.\n\nOne of: "
                 + string.Join(", ", Tiers.Keys.Select(k => $"<code>{k}</code>"));

        var balances = await ledger.ListAsync(ct);
        if (!balances.ContainsKey(name))
            return $"<b>{Esc(name)}</b> has no balance recorded, so it is not a live consumer.\n\n"
                 + "Create it with <code>/newkey</code> first.";

        await ledger.SetTierAsync(name, tier, ct);
        await AuditAsync($"tier name={name} tier={tier}", ct);

        var body = $"<b>{Esc(name)}</b> is now recorded as <b>{Esc(tier)}</b> \u2014 {Esc(t.For)}.";
        if (tier == "admin")
            return body + "\n\n<i>Management only. Nothing enforces that; it is a note to operators.</i>";

        string[] rows =
        [
            $"{"quota",-12}{t.Quota,14:N0}",
            $"{"tokens/min",-12}{t.Tpm,14:N0}",
            $"{"max_tokens",-12}{t.MaxTokens,14:N0}"
        ];
        body += "\n" + Table("", rows);

        // Say plainly where the balance stands against the tier, and do NOT
        // move it. Changing a balance is money, and it is a separate decision
        // from recording what tier someone is on.
        var bal = balances[name];
        if (bal != t.Quota)
            body += $"\n\u26a0\ufe0f Balance is <code>{bal:N0}</code>, tier says <code>{t.Quota:N0}</code>. "
                  + $"Nothing was changed \u2014 run <code>/setquota {Esc(name)} {t.Quota}</code> to align.";

        body += "\n<i>Recorded only. ai-quota charges a flat input+output total and cannot vary by tier; "
              + "rate limits need ai-token-ratelimit, which is bundled but not installed.</i>";
        return body;
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
        return Table("<b>Latency</b>", new[] { header }.Concat(rows))
             + "\n<i>Seconds, whole request as Envoy saw it. Cumulative since Vector started.</i>"
             + "\n<i>Bucketed, so approximate at low request counts \u2014 the exact figure is in the "
             + "fact table.</i>";
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
        BotCommand[] menu =
        [
            new("status",     "Infrastructure health"),
            new("keys",       "Consumers and their balances"),
            new("balance",    "Balance for one consumer or all"),
            new("usage",      "Tokens and requests over a window"),
            new("alerts",     "What is firing right now"),
            new("health",     "Stack and telemetry health"),
            new("top",        "Busiest consumers over a window"),
            new("p95",        "Latency percentiles per consumer"),
            new("errors",     "Status mix per consumer"),
            new("tiers",      "What each policy tier means"),
            new("trace",      "Where to look one request up"),
            new("tier",       "Record a consumer's policy tier"),
            new("topup",      "Add tokens to a consumer"),
            new("newkey",     "Create a key and return its OpenCode config"),
            new("opencode",   "Re-send a consumer's OpenCode config"),
            new("setquota",   "Overwrite a balance (asks to confirm)"),
            new("clearquota", "Set a balance to zero (asks to confirm)"),
            new("revoke",     "Delete a key and its balance (asks to confirm)"),
            new("help",       "Show all commands")
        ];
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

    private static bool IsValidName(string s) =>
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
    private static string Base62(int len)
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
        // Both keys. Otherwise a consumer re-created under the same name
        // silently inherits the revoked one's tier.
        await c.CommandAsync(ct, "DEL", Prefix + name, TierPrefix + name);
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
    string AlertSecret, string AlertmanagerUrl, HashSet<long> AlertChatIds,
    string LangfuseUrl);

enum PendingKind { SetQuota, Revoke }
sealed record Pending(long UserId, DateTimeOffset Expires, Func<CancellationToken, Task<string>> Run);

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

sealed record InlineKeyboardMarkup(
    [property: JsonPropertyName("inline_keyboard")] InlineKeyboardButton[][] Keyboard);

sealed record InlineKeyboardButton(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("callback_data")] string CallbackData);

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
