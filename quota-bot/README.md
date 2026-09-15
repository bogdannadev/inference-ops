# quota-bot

Telegram control surface for gateway access management. Lets a small allowlisted
group monitor the stack and run every access operation — issue and revoke keys,
set and clear quotas, hand out OpenCode configs — without a shell on the box.

Built as a single .NET 10 file-based app (`bot.cs`), published Native AOT.

## Architecture

Webhook, not long polling, and asynchronous end to end:

```
Telegram   --POST--> Caddy :443          TLS + Telegram source-IP filter
                       |
Alertmanager --POST--> | (edge network, not public)
                       v
                     quota-bot :8080     verify secret -> enqueue -> 200
                       |                 (nothing slow in the request path)
          +------------+------------+
          v                         v
        Worker                  AlertWorker
   authorize -> dispatch        render group -> send
     +--> higress:80             quota get / delta / refresh
     +--> apiserver.svc:8443     key-auth object read + write
     +--> higress-redis:6379     enumerate + delete ledger keys
     +--> prometheus:9090        usage windows, ledger runway, stack health
     +--> alertmanager:9093      what is firing now (/alerts)
     +--> api.telegram.org       sendMessage
```

Two independent queues on purpose: an operator waiting on `/keys` should not
queue behind an alert storm, and an alert must not be dropped because a command
is mid-flight.

The handler answers 200 and does no work. Telegram redelivers anything that is
not 2xx, so work in the request path means retry storms *and* duplicate
commands — and a duplicated `/topup` is money. `update_id` is deduped for the
same reason.

## Commands

Generated from the `Commands` table in `bot.cs`, which is the single
source of truth: `/help` and the Telegram command menu are both rendered
from it. This block is a copy and can go stale — the bot cannot.
`/start` is an alias of `/help`.

```
/keys                       every key, its balance and tier
/balance [name]             left of quota, burn, limits and refill
/key                        one key: numbers, settings, requests
/tiers                      tier defaults and what they enforce
/tier <name> <tier>         record a consumer's tier
/policy <name>              one key's limits, with buttons
/set <name> <setting> <value|default>   change one setting by typing it

/usage [1h|24h|7d|30d]      tokens and reference cost per key
/top [1h|24h|7d|30d]        busiest keys, share and errors
/p95 [name] [window]        latency per key, gateway and engine (default 24h)
/prices                     OpenRouter and Alibaba price table
/errors [1h|24h|7d|30d]     error answers per key
/trace <name|request-id>    one key's or one request's records

/status                     can I operate the gateway
/health                     is the stack healthy
/alerts                     what is firing right now

/newkey [name]              create a key, one tap per tier
/connect <name>             endpoint, model, API key and limits for any client
/opencode <name>            re-send a key's config
/topup <name> <tokens>      add to a balance

/setquota <name> <tokens>   replaces a balance
/clearquota <name>          sets a balance to zero
/revoke <name>              deletes a key and its balance
```

`/newkey` seeds the balance before reporting success, deliberately. ai-quota
returns the same 403 *No quota left* for "never seeded", "exhausted" and "Redis
is down", so an unseeded key looks broken in a way that wastes an afternoon.

Credentials are printed once, by `/newkey`. `/keys` lists names only.

## Layout on a phone

Every reply was captured through a fake Bot API and checked, because what reads
well in code does not read well in a 32-column code block. The rules live on
`Fmt` in bot.cs: a consumer name is never a padded column (it broke on a
24-character name and ran the numbers together), numbers are compact, a `<pre>`
table holds only fixed-width label/value pairs, and the explanation goes last in
an expandable blockquote. `/top`, `/usage`, `/errors` and the key card switch
window by button, editing the message in place.

**Check for bare `<` and `&`, not only tags.** On 2026-09-13 `/top`, `/usage`
and low-traffic key cards stopped arriving: `<$0.01` and `<1%` are not tags to a
tag-balance check, but Telegram parses them as one and rejects the whole
message (`can't parse entities: Unsupported start tag`). Values that can start
with `<` are now written `&lt;`, and a refused HTML reply is resent as plain
text so the operator still gets the numbers. The capture check must flag any
`<` that does not open an allowed tag and any `&` that is not an entity.

To re-run the capture: start a copy of the bot with
`TELEGRAM_API_BASE=http://<fake>:8081` and a dummy token next to a tiny HTTP
server that logs each JSON body, POST synthetic updates at its webhook, and
check every `text` for tag balance, length and `<pre>` width. Do not send
`/opencode` through it — the capture would hold a real credential.

## Tiers are defaults; every value can be set per consumer

A tier carries a default for every setting — `quota`, `refill`, `daily`,
`tpm`, `max_tokens` — and a consumer stores only the values set on it by hand,
in `chat_policy:<name>` next to its balance. The effective value is the
hand-set one if present, the tier's otherwise. So changing a tier default moves
everyone still following it, a hand-set value survives a tier change, and
`/set <name> <setting> default` puts one back. `/policy` and the key card's
**Settings** button show each value with where it came from; `/keys` stars any
consumer with something set by hand. The same model is served at
`GET /admin/policy/<name>` and `POST /admin/policy` for admin-mcp.

**Everything is a button.** `/newkey <name>` answers with one button per tier;
one tap creates the key with that tier's defaults and returns its config
(`/newkey` alone asks for the name first; `/newkey <name> <tokens>` is the old
untiered form). The credential message offers **Settings**, which opens a
screen where every value is a button: tap one for presets, *Tier default* or
*Custom…* (the bot asks, the next plain message answers, any command cancels).
Settings screens edit themselves in place; the credential message is never
edited, so the key cannot be scrolled or edited away. Balance buttons are money,
so each is a single-use token bound to the operator: a double tap on +10M adds
10M once, and *Set to quota* still asks first.

**Enforced since 2026-09-13**, except per-key `max_tokens`:

| Setting | Enforced by |
|---|---|
| balance | ai-quota |
| `daily`, `tpm` | ai-token-ratelimit — rules written by `LimiterSync` |
| `refill` | `RefillJob`, at 00:00 UTC (daily / Monday / 1st) |
| `quota` | the `/newkey` seed and each refill |
| `max_tokens` | nothing: gateway plugins cannot match a consumer |

**LimiterSync** is the only writer of the ai-token-ratelimit rules. The committed
object in `../higress-standalone/config/wasmplugins/` is a disabled shell, which
`apply.sh` reinstalls; the bot syncs on every policy/key change and every
minute, comparing a fingerprint and PUTting only on a real difference.
`LIMITER_SCOPE` (names, comma-separated) restricts the rules during a rollout;
`/status` shows the last sync, `/policy` the consumer's live window counters
read from the limiter's own Redis keys.

**RefillJob** keeps a marker `chat_refill:<name>` = `<mode>:<period>`. First sight
of a consumer, or a changed mode, only ARMS the marker — nobody's balance moves
because a setting was touched; the first reset is the next boundary. A reset
SETs the balance to quota (no carry-over), is audited, and is announced to the
alert chats.

Verified on `testafter` 2026-09-13: per-minute refusal at exactly the request
that found the counter over (144 of 150 allowed, 169 refused), daily refusal
with `Retry-After: 86295`, Caddy's OpenAI-shaped 429 (matched on the limiter's
reset header, so router 429s pass untouched), and requests ALLOWED while the
limiter's Redis was pointed at a host that does not exist.

## Requests cut off are charged nothing

ai-quota charges from the final usage frame. A request that ends before it —
client disconnect, stream idle timeout, upstream error — is charged zero, and
SGLang writes no per-request record for a client disconnect either. Measured
2026-09-13 and 2026-09-15 with deliberate cuts; one consumer had 24 of 666 chat
requests end that way. They are counted from the gateway access log as
`gateway_usage_cut_requests` / `_cut_seconds` (the predicate is in
`clickhouse/engine-usage-metrics.sql`, regression cases in
`clickhouse/engine-usage-metrics_test.sql`), and `/top`, `/errors`, the key
card, the written report and admin-mcp's `consumer_stats` show them.

## Usage in money: reference prices

`/key`, `/usage` and the written report price each consumer's tokens at what
the same model costs from public providers — **reference prices for
monitoring, never a bill**, and labelled that way everywhere:

| reference | source | refreshed |
|---|---|---|
| OpenRouter, list | `GET /api/v1/models`, headline price for `PRICE_OPENROUTER_MODEL` (default `qwen/qwen3.8-27b`) | daily, snapshot until the first fetch, last good price on failure |
| OpenRouter, cache-aware | the same, with the key's cached input (GPU and HiCache hits, exact per request) priced at `input_cache_read`; input whose cache split is unknown is priced as uncached | — |
| Alibaba Cloud Singapore / Beijing | Model Studio price page, Qwen3.8-27B | **by hand** in `PriceBook`, no API exists; dated 2026-09-12 |

The provider spread is reported next to the OpenRouter price because it is wide
(output $2.00–3.20 per M across 15 providers on 2026-09-13). `/prices` shows the
table and a worked example; `GET /admin/prices` serves the same numbers so
admin-mcp's `consumer_stats` prices with them rather than a second copy. The
report's caption carries the costs computed by the bot, so the monitored number
does not depend on the model copying it.

## Where the usage numbers come from

Every token, request, cache and engine-latency number is an exact sum or exact
quantile over SGLang's own per-request records (`engine.requests`), and every
refusal, cut-off and gateway latency over the access log (`gateway.requests`),
computed in ClickHouse and scraped as `engine_usage_*` / `gateway_usage_*`
gauges for the preceding 1h, 24h, 7d and 30d — hence those four windows on
every screen. Since 2026-09-14; before that the bot used `increase()` over
Envoy, Vector and engine counters, which lost the first request after each
restart and extrapolated to the window edges (three controlled requests of
29,079 prompt tokens: engine `increase()` 11,194, gateway `increase()` 29,825,
records 29,079). Details: `docs/METRICS-ECOSYSTEM.md`.

- Tokens are what the ledger charged: cut-off and aborted requests are not in
  them, and verified equal to the ledger on 2026-09-15.
- **Cache hit** = prompt tokens the engine did not recompute ÷ prompt tokens of
  requests whose split was recorded; **HiCache** is the part reloaded from host
  RAM. Cached input is still deducted in full from the balance; what a hit saves
  is engine time (GPU hit nearly free, HiCache reload ~1.2 s per 45K tokens
  against ~14 s to recompute). "Prefill avoided" uses the prefill speed measured
  on the same requests.
- Rows before 2026-09-13 19:46 UTC were copied from the gateway log and carry
  tokens only; screens say so where it matters.
- If the scrape is down, usage screens say "usage data unavailable" instead of
  showing zeros; `/health` shows the pipeline state.

## Quota is one total-token number

`/newkey`, `/topup` and `/setquota` all set or move a single balance in
`chat_quota:<consumer>`, and ai-quota deducts input+output from it at the same
rate. That is hardcoded upstream — there is no separate input or output budget.

`/usage` and `/top` therefore show the split even though the budget does not:
an agent consumer re-sending its context can sit near 50:1 input to output,
paying mostly for prefill that the radix cache serves nearly free, while a
consumer doing generative work sits near 1:1 and pays the same rate.

## /status and /health are not the same question

`/status` probes the four things this bot talks to and answers *can I still
operate the gateway*. `/health` reads Prometheus and answers *is the stack
healthy, and can I believe what it is telling me* — scrape coverage, firing
alerts, ledger reachability, whether usage records are arriving, throughput,
TTFT, KV pool, prefix cache (GPU / HiCache) and HiCache host RAM.

Both matter. For roughly fourteen hours in September 2026 the second was false
while the first was true: Langfuse ingest was dead, the correct alert was
firing, and nothing said so.

`/balance <name>` and `/usage` read the `consumer:quota_*` recording rules
rather than recomputing runway locally, so the bot, the Usage & Quota dashboard
and the ConsumerQuotaLow alert cannot disagree about what "days left" means.

## Alert delivery

`POST /alert` receives Alertmanager webhooks and renders one Telegram message
per alert group. It is reachable only from the `edge` docker network — Caddy
proxies `/tg/<random>` to this process and nothing else — but it authenticates
anyway with a bearer, because that bearer is all that stands between "anything
on edge" and the operators' alert channel.

`ALERT_WEBHOOK_SECRET` must match the `credentials_file` Alertmanager presents.
Set `ALERT_CHAT_IDS` to send one copy to a group chat; it defaults to every id
in `TELEGRAM_ALLOWED_IDS`, on the principle that an alert nobody is guaranteed
to see is the failure this path exists to remove.

A payload that will not deserialise is answered **200**, deliberately.
Alertmanager retries non-2xx, and a body that cannot be parsed will not parse on
the third attempt — it would just pin one group in a retry loop forever. Real
delivery failures (the bot being down) never reach that line and are retried
normally.

## How keys are written

`consumers.conf` is gitignored and always has been, so the bot owning it breaks
no git invariant. The committed artefact is
`../higress-standalone/config/wasmplugins/key-auth.yaml`.

On a change the bot rewrites `consumers.conf`, then reads the live `key-auth`
object from the apiserver, replaces only `spec.defaultConfig.consumers` and each
`matchRules[].config.allow`, and PUTs it back. Working on the live object keeps
`resourceVersion` intact — `PATCH` does not work on these custom resources.
`apply.sh` renders the same object from the same consumer table, so the two
writers cannot diverge; last writer wins and both are correct.

Writes go through the apiserver rather than the filesystem because `conf/` is
root-owned `0700`, and reaching it would mean running as root or mounting the
Docker socket. The bot has neither.

The apiserver requires authentication (`--auth-enabled`, set 2026-09-03 — before
that an anonymous GET of the wasmplugins collection returned 200, which meant
every consumer credential was readable by anything on `higress-net`). The bot
presents the same client certificate the controller and console use, read from
the kubeconfig at `../higress-standalone/compose/volumes/kube/config` rather
than copied — one place to rotate it, not two.

## Setup

1. `cp .env.example .env` and fill it in. Get your numeric Telegram id by
   messaging the bot once and reading the `DENIED user <id>` line from
   `docker logs quota-bot` — the bot logs the id of anyone it refuses.
2. Add the Caddy site block (see below) and reload.
3. `docker compose up -d --build`
4. `./register-webhook.sh set`, then `./register-webhook.sh info` — check
   `pending_update_count: 0` and an empty `last_error_message`. A webhook can be
   registered and failing every delivery, and nothing else reports that.

### Telegram-side settings

Everything Telegram holds for this bot lives on *their* servers, not in this
repo, so `getWebhookInfo` is the only source of truth for what is actually
registered. Audited against the Bot API docs on 2026-09-03:

| Setting | Value | Why |
|---|---|---|
| `url` | `https://<host>/tg/<random>` | 443 + TLS 1.3. Telegram allows only 443, 80, 88, 8443 |
| `secret_token` | 40 chars | Within the documented 1–256 of `[A-Za-z0-9_-]`; validated at startup |
| `allowed_updates` | `["message","callback_query"]` | **See the trap below** |
| `max_connections` | 10 | Default is 40; this bot serves a handful of operators |
| `drop_pending_updates` | true on `set` | Stops a backlog replaying after downtime — a replayed `/topup` is money |
| `ip_address` | unset | Would pin Telegram's target IP and skip its DNS lookup. Checked and not useful here: Telegram, DuckDNS and the host all agree on the address |
| `certificate` | unset | Let's Encrypt via Caddy, so no self-signed upload is needed |

**The `allowed_updates` trap.** An update type absent from that list is not
filtered by this bot — *Telegram never sends it*. The confirmation prompts for
`/revoke`, `/setquota` and `/clearquota` are inline keyboards, and a tapped
button arrives as a `callback_query`. While the webhook was registered with
`["message"]` alone, those buttons did nothing whatsoever: the spinner turned
and no update ever left Telegram.

It survived testing because the obvious way to test the handler — POST a
synthetic `callback_query` at the webhook — bypasses the filter completely and
passes. **After changing which update types the bot reacts to, re-register and
check `getWebhookInfo`.** Nothing else reports this.

`edited_message` is deliberately still absent, so that editing an already-sent
`/topup` cannot re-execute it.

Documented Telegram source subnets are `149.154.160.0/20` and `91.108.4.0/22`;
the Caddy matcher below carries both, and observed deliveries arrive from
`91.108.5.110`, inside the second.

### Caddy

The webhook needs a hostname on 443 — Telegram accepts only 443, 80, 88 and
8443, and Caddy already owns 443. Two independent defences, because Telegram
documents that its IP ranges change:

```caddyfile
bot-old.example.org {
	tls contact@example.com

	@telegram remote_ip 149.154.160.0/20 91.108.4.0/22
	handle @telegram {
		reverse_proxy quota-bot:8080
	}
	handle {
		respond "Not found" 404
	}
}
```

Edit the Caddyfile **in place** — it is a file bind-mount, and a temp-file
rename swaps the inode so `caddy reload` silently re-reads the stale original.
Verify through the admin API, not the reload exit code.

## /key — the no-arguments path

Every other per-consumer command needs a name typed correctly. `/key` takes no
arguments: the consumer list *is* the interface. Tap a name, get its numbers;
tap again for its per-request records, or for a written report.
`/trace` with no argument lands on the same picker.

The card reads Prometheus only. That is a limit, not an oversight — this bot
runs on `edge`, and ClickHouse is backend-only, because an internet-reachable
bot with a route to the worker ports is a worse trade than an operator pasting
one SQL query. So *statistics* are answered here in full, and *one request* is
answered with the query (`/trace <request-id>` joins the gateway row to the
engine row on `chat_id = rid`); admin-mcp runs the same queries.

Balance and runway come from the ledger; requests, tokens, engine p95, first
token, decode speed, cache split and the replica split from the engine's
per-request records; gateway p95, not-2xx and cut-offs from the access log.

### The report is written by the node

`Report ↓` sends the consumer's numbers to qwen36-27b through the gateway and
returns the answer as an HTML file. Three things about it are deliberate:

- **It authenticates with the admin key.** `docs/KEY-TIERS.md` describes the
  admin tier as management-only; this is the exception, recorded there. It is
  the only credential the bot holds, and it is metered like any other — about
  2.5-4k tokens a report against `quota-admin`'s balance.
- **Thinking is off** (`chat_template_kwargs.enable_thinking = false`).
  Measured on this node: with it on the model spends its first several hundred
  tokens deliberating and the HTML arrives truncated; with it off the first
  character is `<!doctype html>` and a full report is ~2.6k tokens in ~30s.
- **It runs detached.** The worker gate bounds how many commands run at once,
  and a 30-second model call has no business holding one of those slots.

The model is given the numbers and told not to invent any; it can still be
wrong about what they *mean*. The caption on every file says so. If the model
hits its token ceiling the file gets a red banner at the top rather than
silently half-rendering.


## Build

`bot.cs` carries its own build configuration in `#:property` directives, so the
file is the single source of truth for how it is compiled. The ones that matter:

| Property | Why |
|---|---|
| `PublishAot=true` | Default for file-based apps; explicit here as documentation |
| `PackAsTool=false` | File-based apps default to `true`; this is a service |
| `InvariantGlobalization=true` | No culture data needed |
| `OptimizationPreference=Size` | |
| `TrimmerSingleWarn=false` | Trim warnings collapse to one per assembly otherwise |
| `TreatWarningsAsErrors=true` | The one that actually bites — see below |
| `EventSourceSupport`, `MetadataUpdaterSupport`, `HttpActivityPropagationSupport`, `Http3Support` = `false` | Unused subsystems |

**`ILLinkTreatWarningsAsErrors` alone is not enough.** The first build here
passed clean-looking while emitting IL2026/IL3050 on `JsonArray.Add<T>` — a real
`JsonValue.Create<T>` reflection hazard that AOT turns into a runtime failure.
`TreatWarningsAsErrors=true` is what catches those, because the analysers raise
them as ordinary compiler warnings first.

Deliberately **not** set: `UseSystemResourceKeys` and `StackTraceSupport=false`.
Both trade diagnostics for a few hundred KB. This process moves customer
balances; a stripped exception message costs more than the size saving.

Image is ~71 MB: a two-stage build on the official `sdk:10.0` (which needs
`clang` and `zlib1g-dev` for the native link step), shipping on
`runtime-deps:10.0-noble-chiseled` — no shell, no package manager, no .NET
runtime, non-root. Nothing to exec into, which is the point for a process
holding an admin credential.

The chiselled image has no `curl`, so the compose healthcheck is the binary
probing itself: `/app/bot --healthcheck`.

## Verified

Full lifecycle against the live stack on 2026-09-03, then reverted:

- webhook: bad/missing secret 401, correct secret 200, malformed JSON 200 (never
  500 — a 500 makes Telegram retry forever)
- non-allowlisted user refused and logged; a group message from an *allowlisted*
  user also refused, so the allowlist alone is not the only gate
- duplicate `update_id` handled once
- `/newkey ttest 12345` wrote `consumers.conf`, the live `key-auth` object
  (4 → 5 consumers, `resourceVersion` 1 → 2) and the ledger; the issued key then
  returned **200 through the public hostname** and metered 69 tokens
- `/revoke` ignored a non-`CONFIRM` reply, then on `CONFIRM` removed all three;
  the revoked key returned 401
- `consumers.conf` byte-identical to its pre-test backup afterwards, ledger back
  to its original four entries, and an existing consumer still authenticated
- credentials never appear in the logs

## Operating

```bash
docker compose up -d --build     # rebuild and restart
docker compose logs -f           # DENIED lines carry the numeric id to allowlist
./register-webhook.sh info       # what Telegram believes
cat data/audit.log               # every key and balance change the bot made
```

`data/audit.log` is the record of who was issued a key and whose balance moved.
Without it the only evidence a key exists would be the key itself.

## Latency

Every command logs three numbers, and the split is the diagnosis:

```
/status from <your-telegram-id> queued=1s work=4ms total=190ms
```

- **`queued`** — how long Telegram held the update after the operator pressed
  send. Not ours, and not fixable from inside this process.
- **`work`** — this stack and the services behind it. Ours to fix.
- **`total`** — `work` plus the round trip back out to Telegram.

`queued` exists because every other clock in this process starts when the update
*arrives*, so an update that sat in Telegram's retry queue for twenty minutes is
indistinguishable from a fast bot. If the bot feels slow while `work` and
`total` are in the hundreds of milliseconds, read `queued` — a large value means
inbound delivery to this host is failing and Telegram is backing off.

### How delivery actually reaches this host

**Telegram cannot connect to this host at all.** Its source ranges are dropped
upstream — measured on both TCP 443 and 8443, with every local counter clean
(`ListenDrops` 0, no TLS handshake errors), outbound to Telegram flawless
(30/30, ~98 ms), other inbound traffic fine, and no AAAA records. The identical
failure on two different ports is what proved the filter keys on Telegram's
**source addresses**, so no port and no hostname could ever have fixed it.

Deliveries therefore arrive via a Cloudflare Worker (`worker.js`) on
`workers.dev`, which forwards to the origin from Cloudflare's IPs — not
filtered, because not Telegram:

```
Telegram --POST--> <name>.workers.dev        free, permanent hostname
                        |
                        v  Worker forwards, secret header intact
                   bot.example.org     Caddy, Cloudflare IPs allowed
                        |
                        v
                   quota-bot:8080
```

Measured after the switch: a **50-minute idle gap still delivered in 1 s**,
against 132–543 s before. Idle gaps were what triggered the stalls, so that is
the case that matters.

Consequences worth knowing:

- `TELEGRAM_PUBLIC_URL` (the Worker) and `TELEGRAM_WEBHOOK_URL` (the origin) are
  now different values. The bot derives its listen path from the origin; only
  Telegram's target is the Worker. Registering the origin would silently restore
  the stalling.
- Caddy's `remote_ip` allowlist was widened from Telegram's ranges to
  Cloudflare's. The IP match was never the real gate — the secret token is — but
  the endpoint is now reachable from anything on Cloudflare's network. If
  deliveries ever fail with **404 rather than timeouts**, Cloudflare changed
  their published ranges and that list needs updating.
- A Cloudflare **Tunnel** would be strictly better — no public inbound at all —
  but a tunnel's public hostname requires a domain on Cloudflare DNS, and
  `workers.dev` is not a zone. The Worker exists only to avoid buying a domain.

**Historic — the failure mode this replaced.** Telegram reaches this host only
intermittently: deliveries land in bursts separated by 15–25 minute gaps, with
`getWebhookInfo` reporting `Connection timed out` in between. Outbound to
api.telegram.org is fine (~100 ms), so it is the inbound leg specifically. This
is what the Cloudflare Tunnel below is for — it makes delivery arrive over an
outbound-initiated connection, which is the direction that works.

Telegram is ~100 ms away from this host and a *new* connection to it costs
~200 ms more (TCP 100 ms + TLS 106 ms). Since the commands themselves finish in
single-digit milliseconds, everything that matters is round trips and whether
the connection is still open. Three things follow from that, and all three were
wrong in the first cut:

- **The typing indicator is sent only after 350 ms**, and never awaited. As an
  awaited call in front of every command it *was* the latency it existed to
  excuse — a 100–300 ms round trip announcing 6 ms of work.
- **The connection is held open** with HTTP/2 keep-alive pings.
  `IHttpClientFactory` rotates handlers every 2 minutes by default and the pool
  dies with them, so before this every command typed after a gap — which is
  most of them — paid the full handshake.
- **Updates are handled concurrently** (bounded at 8). Serially, a second
  command waited out the first one's Telegram round trips.

Framework request logging is at Warning, because the 30-second healthcheck was
75% of the log and buried the command traffic. `LOG_HTTP=debug` in `.env`
restores full per-request tracing, which is what these numbers were measured
with.

## `ValueTask` and `stackalloc` — where, and where not

Both appear in this file, in a few specific places and deliberately nowhere
else. The reasoning is worth keeping, because the natural instinct is to apply
them broadly and that would make this program slower and less safe.

**`stackalloc` is illegal in an `async` method**, and almost everything here is
async. It is not a style choice: a `Span<T>` cannot live across an `await`. So
it appears only in synchronous code — `SecretMatches`, `Base62` — and the way
to use spans near async code is to put the span work in a *synchronous helper*
and let the async method do nothing but await and bookkeep. That is exactly why
`RespConnection` is shaped the way it is, with `EncodeCommand`, `IndexOfCr`,
`TakeString`, `TakeInto` and `AccumulateTo` all synchronous.

Every `stackalloc` here is bounded by a checked constant. `SecretMatches` takes
its input from the internet, so it rejects on length before it copies anything;
`Base62` range-checks its argument. An unbounded `stackalloc` is a stack
overflow, which is not a catchable exception — it kills the process.

**`ValueTask` is used in exactly one place**: the RESP reader. `ReadByteAsync`
is called in a loop over a reply and is almost always answered from the buffer
without touching the socket, so it returns a completed `ValueTask<byte>` with no
state machine and no allocation. A hot path that usually completes
synchronously is the case `ValueTask` exists for.

It is *not* used for the command handlers or the HTTP paths, and that is not an
oversight:

- Those are genuinely async — they always suspend on real network I/O, so
  `ValueTask` saves nothing and adds a struct copy.
- They are handed to `Task.WhenAll` / `Task.WhenAny`, which take `Task`.
  Converting them would force `.AsTask()` and allocate **more** than today.
- A `ValueTask` may be awaited only once. Using it where the payoff is zero
  buys a real footgun for nothing.

The measurable effect of all of this on command latency is nil — the bot spends
2–16 ms working and ~100 ms per Telegram round trip. It was worth doing because
one of the things it replaced was a genuine defect: bulk replies were read **one
`await` per byte**.

## Memory: POH, FOH and GC settings

The bot logs its own GC shape at startup, so this is measured rather than
assumed:

```
gc: server=False concurrent=- datas=1 conserve=- regionSize=1048576
gc: committed=0MiB available=96MiB pinned=0 latency=Batch
```

**The Pinned Object Heap is not applicable here, and using it would be wrong.**
The POH exists for *long-lived* buffers that would otherwise pin a GC region and
block compaction. The only candidate was `RespConnection`'s 64 KB socket buffer
— but a connection is opened per command, so that buffer is short-lived, and the
POH is never compacted: pushing short-lived allocations through it fragments the
one heap that cannot defragment itself. `PinnedObjectsCount` on this process is
**0**, so there is no pinning pressure to relieve in the first place. The real
finding there was that the buffer was freshly allocated on every command; it is
`ArrayPool`-rented now, which is the correct fix.

**Redis connects per command on purpose** — decided 2026-09-03, don't "fix" it.
Measured: one connection per command, and `/keys` runs `work=1-2ms` *including*
connect, `SCAN` and `MGET`, so setup is sub-millisecond. Redis has `timeout 0`,
so a persistent connection would survive — the point is not that it couldn't.

Connect-per-command is stateless and self-healing: Redis restarts, or this
container is redeployed, and the next command just reconnects. A persistent
connection buys stale sockets, half-open TCP and "the first command after a
Redis restart fails". A *shared* one would be worse still — `RespConnection`
holds `_buf/_len/_pos` and is not thread-safe, so with 8 concurrent handlers it
needs a lock, which serialises them and undoes the concurrency above. The only
correct alternative is a pool with health checks, reconnect and idle eviction:
real code in the path that moves customer balances, to save one millisecond.

Revisit only if Redis work becomes a visible share of `work=`, or if something
starts polling the ledger on a timer (the phase-2 low-balance alerts would).

**The Frozen Object Heap has no public allocation API.** There is no
`GC.AllocateArray(frozen: true)` equivalent — the FOH is entirely runtime-managed
for things like string literals and certain statics. Native AOT already gets this
benefit: literals are frozen into the image at build time. Nothing to apply.

**GC settings were the part with something real in them**, though not what you'd
expect:

| | before | after |
|---|---|---|
| GC flavor | Workstation | Workstation (already — nothing to win) |
| memory the GC believes it may use | **1,032,019 MiB** | **96 MiB** |
| SOH region size | 4 MiB (default) | 1 MiB |
| background GC thread | yes (`Interactive`) | no (`Batch`) |
| RSS idle / under load | 14.29 MiB | 14.05 / 16 MiB |

The headline is the third row of the log, not the RSS: with no container limit
the GC sized itself against the **whole 1 TB machine**. RSS barely moved because
Native AOT and Workstation GC had already done the real work — what changed is
that the process is now *bounded*, which on a box running paid inference is the
part that matters. `mem_limit: 128m` is what makes the GC's own heuristics
meaningful; the knobs are secondary to it.

Region size at 1 MiB is a documented recommendation for "processes that have
very small GC heaps", where it cuts the GC's native bookkeeping. Non-concurrent
GC drops a thread and trades pause time we do not care about — a 100 ms round
trip dwarfs any pause this heap can produce.

Those three knobs live in `docker-compose.yml`, not in `bot.cs`, because
`GCRegionSize`, `GCConserveMemory` and friends have **no MSBuild property** —
environment variables are the only way to set them, and their numeric values are
**hex**. `latency=Batch` in the log is the confirmation that `gcConcurrent=0`
took effect; `GCConserveMemory` is not reported by
`GC.GetConfigurationVariables()`, so it is set but not independently verifiable
from inside the process.
