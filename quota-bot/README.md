# quota-bot

Telegram control surface for gateway access management. Lets a small allowlisted
group monitor the stack and run every access operation — issue and revoke keys,
set and clear quotas, hand out OpenCode configs — without a shell on the box.

Built as a single .NET 10 file-based app (`bot.cs`), published Native AOT.

## Architecture

Webhook, not long polling, and asynchronous end to end:

```
Telegram --POST--> Caddy :443            TLS + Telegram source-IP filter
                     |
                     v
                   quota-bot :8080       verify secret -> enqueue -> 200
                     |                   (nothing slow in the request path)
                     v
                   Worker                authorize -> dispatch -> reply
                     +--> higress:80             quota get / delta / refresh
                     +--> apiserver.svc:8443     key-auth object read + write
                     +--> higress-redis:6379     enumerate + delete ledger keys
                     +--> prometheus:9090        usage windows
                     +--> api.telegram.org       sendMessage
```

The handler answers 200 and does no work. Telegram redelivers anything that is
not 2xx, so work in the request path means retry storms *and* duplicate
commands — and a duplicated `/topup` is money. `update_id` is deduped for the
same reason.

## Commands

```
/status                  gateway, ledger, Prometheus, key-auth health
/keys                    consumers and balances (never credentials)
/balance [name]
/usage [1h|24h|7d|30d]

/newkey <name> [quota]   create, install, seed, return OpenCode config
/opencode <name>         re-emit the OpenCode config for a consumer
/revoke <name>           delete the key and its ledger entry (CONFIRM)

/topup <name> <n>        add tokens
/setquota <name> <n>     overwrite the balance (CONFIRM)
/clearquota <name>       set the balance to zero (CONFIRM)
```

`/newkey` seeds the balance before reporting success, deliberately. ai-quota
returns the same 403 *No quota left* for "never seeded", "exhausted" and "Redis
is down", so an unseeded key looks broken in a way that wastes an afternoon.

Credentials are printed once, by `/newkey`. `/keys` lists names only.

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
tg-hook.duckdns.org {
	tls contact@bogdanna.dev

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
/status from 700766285 queued=1s work=4ms total=190ms
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

**That is the known failure mode here.** Telegram reaches this host only
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
