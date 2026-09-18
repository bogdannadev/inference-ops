# admin-mcp

An MCP server for operating this inference node from an admin's Claude client.

## Why it exists

`quota-bot` already answers "what is happening" from Telegram, and it is
deliberately blind to the per-request tables: it runs on `edge`, ClickHouse
(`gateway.requests`, the access log; `engine.requests`, SGLang's own record of
every finished request) is backend-only. That is the right call for a bot
reachable from a public Telegram webhook. The bot's per-request screens are
served from here instead — see *The bot's read port* below.

This process is the one allowed to cross that line. What it buys is not another
copy of the bot's commands — it is investigation that chains: *who spiked* →
*which requests* → *what did the engine do inside those requests*, without a
human pasting queries between the steps.

## Protocol

MCP revision **2026-07-28**, Streamable HTTP, stateless. That revision removed
protocol-level sessions, the standalone GET stream and resumability, so this is
a single POST endpoint holding no per-client state. The C# SDK v2.x serves it
natively (`Stateless = true` by default) and still speaks 2025-11-25 and earlier
for older clients.

Note the header discipline that revision added: every POST carries
`MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` for `tools/call`, and the
server must reject a request whose headers disagree with its body. The SDK
handles that; it matters when hand-testing with curl.

## Security

Four layers, and the design assumes any one of them can fail.

| | |
|---|---|
| **No published port** | Caddy is the only route in. `docker compose` publishes nothing. |
| **IP allowlist** | Caddy refuses anything outside Anthropic's published egress range `160.79.104.0/21` with a 404. |
| **Bearer token** | Checked here in constant time, SHA-256 compared. An org admin enters it once in Claude (`static_headers`). |
| **Origin validation** | A MUST in the transport spec. Absent Origin is fine (non-browser client); present-and-unknown is 403, refused *before* the token is even considered. |

Beyond the front door:

- Its ClickHouse identity is `readonly=2` with capped settings — it cannot write
  to either per-request table even if everything above is bypassed, and it can
  read only the `gateway` and `engine` databases.
  See `qwen3.6-27b-2-A100/clickhouse/users.d/mcp-readonly.xml`, which also
  explains why `readonly=2` and not `1`.
- Every SQL statement is a literal in this file with **bound parameters**.
  Nothing a client sends is ever concatenated into a query.
- Consumer names are constrained at the door (`SafeName`) because PromQL has no
  parameter binding — a name reaches it as a string literal.
- `MCP_WRITES_ENABLED=false` serves reads only, without taking the server down.

### Key lifecycle goes through quota-bot, not around it

`create_key`, `revoke_key` and `set_tier` exist here, but this process does not
touch the key-auth object. It calls quota-bot's `/admin/*` API.

That indirection is the whole point. Consumers live in a single key-auth
wasmplugin object; the Higress apiserver is file-backed and returns **no
`resourceVersion`** — verified 2026-09-05, a PUT is last-write-wins over the
whole object. `KeyStore` holds the lock that serialises edits, and it is a
singleton in the bot's process. Routing through it means a create from Claude
and a `/newkey` from Telegram queue behind the same lock, write the same audit
log, and mint credentials with the same rejection-sampled generator. A second
writer would silently drop one of two concurrent creations, and the symptom
would be a key that looks created and does not authenticate.

`/admin/*` is not mapped at all unless `ADMIN_API_SECRET` is set, is reachable
only on `edge`, and is never published by Caddy.

**ai-quota takes form-encoded bodies, not JSON**, and answers `403` to anything
else — which reads exactly like an auth failure and is not one. The field names
also differ between endpoints: `/quota/refresh` takes `quota`, `/quota/delta`
takes `value`. Both cost a debugging round here.

### The bot's read port

A second listener, `:8081`, exists only for quota-bot's **Requests** screen,
`/trace` and `/errors <name>`. Caddy proxies `admin-mcp:8080` and nothing else, and the port is not
published, so it is reachable only from `edge`.

- Three routes, all GET, all fixed SQL with bound parameters (`RequestSql` in
  `mcp.cs`, shared with `request_detail` so the two cannot drift):
  `/bot/requests/{consumer}?limit=N` (latest completion requests over 7 days,
  each LEFT JOINed to its engine record), `/bot/request/{request_id}`, and
  `/bot/errors/{consumer}?hours=H&limit=N` (latest failed requests with a named
  cause; `unauthenticated` reads the 401 path).
- The cause classifier (`RequestSql.ErrorCause`) is the same expression as the
  `gateway_usage_error_requests` gauge in
  `qwen3.6-27b-2-A100/clickhouse/engine-usage-metrics.sql`, pinned by
  `clickhouse/error-cause_test.sql`. Change all three together.
- Gated by `BOT_READ_SECRET`, deliberately **not** `MCP_BEARER_TOKEN`: a leaked
  bot secret reads request metadata — ids, statuses, token counts, timings; no
  prompts are stored anywhere — and cannot call a single MCP tool. Under 32
  characters or unset, the port answers 404 to everything.
- The listeners never share a route: `/bot/*` on `:8080` is 404, anything but
  `/bot/*` on `:8081` is 404.

## Tools

Read:

```
list_consumers                     names, balances, runway
consumer_stats <name> <window>     exact usage, cache split, engine and gateway latency, refusals, cuts, reference cost
consumer_requests <name> ...       individual requests from both per-request tables
request_detail <request_id>        one request: gateway row joined to its engine row
top_consumers <window>             ranking by tokens charged (exact)
node_health                        targets, alerts, throughput, KV pressure, cache, usage-records freshness
prometheus_query <promql>          arbitrary instant query, read-only by nature
list_tiers                         the tier table, for choosing one
get_policy <name>                  a consumer's settings, and which follow its tier
```

Write, each requiring `confirm` to equal the consumer name exactly:

```
topup_balance <name> <tokens>            ADD to a balance
set_balance   <name> <tokens>            REPLACE a balance
create_key    <name> [tier] [quota]      new consumer; credential returned ONCE
revoke_key    <name>                     delete consumer and balance
set_tier      <name> <tier>              record a tier (bookkeeping, not enforcement)
set_policy    <name> <field> <value>     change one setting, or 'default' to follow the tier
```

`create_key` takes either a **tier** — which seeds the quota from the shared
tier table — or an explicit **quota**, or both, in which case the explicit
number wins. `list_tiers` is described so the model reads it first and
recommends a tier that fits the stated use case rather than inventing a number.
The tier table is the same one `/tiers` renders and `/tier` validates against,
served from the bot, so advice here and enforcement there cannot drift.

Tiers are **enforced** since 2026-09-13, except `max_tokens`: the balance by
ai-quota, `daily_limit` and `tokens_per_minute` by ai-token-ratelimit (rules
rendered by quota-bot), and `refill` by quota-bot's refill job. `max_tokens`
cannot be varied per key at the gateway and is named `_NOT_ENFORCED`. The tool
descriptions spell out the window semantics — a daily limit is a 24h window from
a key's first request, not a calendar day — because a model that assumes
midnight resets will give bad advice about it.

The `confirm` echo is not ceremony. It makes a mis-parsed or hallucinated call
fail closed, because the model has to name the target twice and the two have to
agree. Every write appends to the **same audit log quota-bot writes to**, so
"who changed this" has one answer regardless of which interface was used.

Tool descriptions are written for a model rather than a person: they say what a
number means and where it comes from. Two latencies exist on this node and they
differ by the gateway filter chain, the router and two network hops — a tool
that returned `p95: 15.5` with no provenance would invite comparing it against
something measured elsewhere.

## Deploy

```
docker compose up -d --build
```

Both one-time steps are **done** (2026-09-05):

1. The read-only ClickHouse user, from
   `qwen3.6-27b-2-A100/clickhouse/users.d/mcp-readonly.xml` plus
   `MCP_CLICKHOUSE_PASSWORD_SHA256` in that project's `.env`. Verified by POST,
   not GET — ClickHouse treats a GET as read-only for *every* user, so a GET
   test proves nothing about the profile. POST `INSERT` and POST `DROP` both
   return `Code 164 READONLY`; POST `SELECT` works.
2. A DNS A record for the MCP hostname (`EDGE_HOST_MCP` in the inference
   project's `.env`) pointing at this host, and `caddy reload`. Do not
   reload a site block whose hostname does not resolve yet: Caddy starts an ACME
   loop it cannot win and the failures count against Let's Encrypt's rate
   limits, delaying the certificate even after the record appears.

**Edit the Caddyfile in place** — it is a file bind-mount, and a temp-file
rename swaps the inode so the reload silently succeeds against the stale
config. Verify through the admin API, not the reload's exit code.

In Claude: add a custom connector at `https://<mcp-host>/mcp` with the bearer
token from `.env` as an `Authorization` header.

Verified from this host, which is *not* in Anthropic's range: both an
unauthenticated request and one carrying a valid bearer get `404`. The
allowlist is the outer gate and the token cannot substitute for it.

## Build

Native AOT, same as quota-bot: build configuration lives in the `#:property`
directives at the top of `mcp.cs`, the SDK stage never ships, and the final
image is `runtime-deps` chiselled — no shell, no package manager, nothing to
exec into. `TreatWarningsAsErrors=true` is what catches the trimming hazards
that AOT turns into runtime failures.

The chiselled image has no curl, so the container healthcheck is the binary
probing itself: `/app/mcp --healthcheck`, which hits the unauthenticated
`/healthz` mapped ahead of the auth middleware.
