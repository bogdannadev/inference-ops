# admin-mcp

An MCP server for operating this inference node from an admin's Claude client.

## Why it exists

`quota-bot` already answers "what is happening" from Telegram, and it is
deliberately blind to the two stores that matter most for tracing: it runs on
`edge`, the fact table and the span store are backend-only, and so `/trace`
prints SQL for a human to run rather than running it. That was the right call
for a bot reachable from a public Telegram webhook.

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
  to the trace store or the fact table even if everything above is bypassed.
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

## Tools

Read:

```
list_consumers                     names, balances, runway
consumer_stats <name> <window>     gateway AND engine view of one consumer, reference cost
consumer_requests <name> ...       individual requests from the fact table
request_detail <request_id>        one request end to end, the two-hop join
top_consumers <window>             ranking by tokens
node_health                        targets, alerts, throughput, KV pressure
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

Tiers are **recorded, not enforced**: ai-quota deducts a flat input+output total
and cannot vary by tier, and the rate-limit plugin is bundled but not installed.
A tier sets the starting balance and documents intent. The tool descriptions say
so, because a model that believes `tokens_per_minute` is enforced will give bad
advice about it.

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
2. `qw38-27b-mcp.duckdns.org` → 89.250.81.248, and `caddy reload`. Do not
   reload a site block whose hostname does not resolve yet: Caddy starts an ACME
   loop it cannot win and the failures count against Let's Encrypt's rate
   limits, delaying the certificate even after the record appears.

**Edit the Caddyfile in place** — it is a file bind-mount, and a temp-file
rename swaps the inode so the reload silently succeeds against the stale
config. Verify through the admin API, not the reload's exit code.

In Claude: add a custom connector at `https://qw38-27b-mcp.duckdns.org/mcp`
with the bearer token from `.env` as an `Authorization` header.

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
