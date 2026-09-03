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

**The apiserver accepts unauthenticated requests on `higress-net`** — verified,
an anonymous GET of the wasmplugins collection returns 200. That is what makes
this work, and it is also a standing exposure: any container on that network can
read every consumer credential. Worth fixing on its own merits; the bot does not
make it worse.

## Setup

1. `cp .env.example .env` and fill it in. Get your numeric Telegram id by
   messaging the bot once and reading the `DENIED user <id>` line from
   `docker logs quota-bot` — the bot logs the id of anyone it refuses.
2. Add the Caddy site block (see below) and reload.
3. `docker compose up -d --build`
4. `./register-webhook.sh set`, then `./register-webhook.sh info` — check
   `pending_update_count: 0` and an empty `last_error_message`. A webhook can be
   registered and failing every delivery, and nothing else reports that.

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
