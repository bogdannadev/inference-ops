# Key tiers

**Status: ENFORCED as of 2026-09-13**, except per-key `max_tokens`.

A tier is a set of DEFAULTS for every setting — quota, refill, daily limit,
tokens/min, max_tokens — and any one of them can be set on a single consumer
(`/set`, or the Settings buttons; stored in `chat_policy:<name>`). `/policy`
shows each value, whether it follows the tier, the consumer's live limiter
counters and its next refill. The table lives in `quota-bot/bot.cs`
(`Policy.All`) and is served to admin-mcp from there.

| Setting | Enforced by | Semantics |
|---|---|---|
| balance | ai-quota, every request | 403 at or below zero; one request can overdraw |
| `daily`, `tpm` | ai-token-ratelimit, rules rendered by quota-bot's `LimiterSync` | window opens at the key's first request (not midnight); refused only once already over, so one request overshoots by its size; 429 + `Retry-After` |
| `refill` | quota-bot's refill job | balance SET to `quota` at 00:00 UTC daily / Monday / 1st; switching it on arms the next boundary, never resets at once |
| `quota` | `/newkey` seed and each refill | changing it never moves a live balance by itself |
| `max_tokens` | **nothing, per key** | a WasmPlugin matchRule cannot select by consumer (v2.2.4 proto); one global 70000 ceiling |

Rollout: proved on `testafter` 2026-09-13 — per-minute 429 at the expected
request, daily 429 with `Retry-After: 86295`, and requests allowed while the
limiter's Redis was unreachable — then applied to **every consumer** the same
day (`LIMITER_SCOPE` removed). Four keys carry limits: `danila`, `testafter`
(team), `vkondratpev-demo2-cursor` and `vkondratyev-demo` (service; the latter
with a hand-set 10 M daily limit, kept deliberately although it used 32.6 M on
2026-09-12). Setting `LIMITER_SCOPE` again narrows the rules for a future test.

Companion: `docs/METRICS-PLAN.md` (build order), `docs/OBSERVABILITY.md`
(why quota is a single total-token number),
`higress-standalone/config/wasmplugins/ai-token-ratelimit.yaml` (the plugin's
behaviour as read from source).

## What an input token and an output token actually cost here

Measured directly against the router — no gateway hop, no quota consumed. A
12,472-token prompt with `max_tokens=1` isolates prefill; a 66-token prompt with
`max_tokens=200/400` isolates decode; a 53-token prompt with `max_tokens=1`
gives the fixed per-request overhead (~0.10s) that is subtracted from both.

| | ms per token | relative |
|---|---|---|
| output (decode) | **18.14** | 1× |
| input, uncached (prefill) | **0.266** | 1/68 |
| input, cached (prefix hit) | **0.0038** | 1/4813 |

Energy, from DCGM at 221 W idle / 407 W busy:

| | J per token @407W | J per token, marginal over idle |
|---|---|---|
| output | 7.38 | 3.37 |
| input, uncached | 0.108 | 0.049 |
| input, cached | 0.0015 | 0.0007 |

**An output token costs about 68× an uncached input token and ~4800× a cached
one.** Commercial APIs charge 3–6× for output (Claude Sonnet 5×, GPT-4o 4×,
Groq's Llama 1.3×). The spread reflects how compute-bound the serving is, and
this node sits at the far end: decode is bandwidth-bound at ~18ms per token
while prefill processes a whole prompt in one pass.

**Caveat, and it matters.** These are single-request, unbatched figures. With
`--max-running-requests 4` per replica, concurrent decode amortises across the
batch, so under load the per-request output cost falls roughly 4–8× and the
ratio compresses to something like 8–17×. Still well above what anyone charges.
Treat 68× as the no-contention upper bound, not the billing ratio.

### Observed ceilings

| | value |
|---|---|
| peak output throughput (30d) | 387 tok/s |
| peak prompt-token rate (30d) | 13,788 tok/s |
| single-stream output ceiling | 55 tok/s |
| engine concurrency | 2 replicas × 4 running = 8 |

Input capacity exceeds output capacity by ~35×. **Output is the scarce
resource; input is nearly free.** Any tier that limits "tokens" without
distinguishing direction is therefore governing the wrong quantity.

### A routing finding that changes cache economics

Six identical 12k-token prompts in sequence:

```
call 1: 3.444s  COLD
call 2: 3.380s  COLD
call 3: 0.153s  warm
call 4: 0.152s  warm
```

**The first two were both cold** under the old `round_robin` policy: it sent
them to different replicas, and each had to prefill the prefix independently,
so a repeated prefix cost full prefill *twice* before it was warm on both.

**Fixed 2026-09-05.** The router now runs `cache_aware
--balance-abs-threshold 2`. Measured against a same-day `round_robin` control,
shared-prefix cache hit went 65.1% -> 97.7% with no replica starved and no
latency regression. The trace above should now show call 2 warm. The
`cache_aware` policy had been disabled in July for starving r0; the cause was
a balance threshold that could never fire at this node's scale, not the policy.
See `tuning/docs/ROUTING.md`.

Note this applies only to traffic **through the router**. Requests on the
direct hostname bypass it entirely and get no affinity, which is one more
reason per-person keys behind the gateway matter.

## What can actually be enforced

| Control | Scope | Where | Status |
|---|---|---|---|
| Total-token balance | per consumer | `ai-quota` → `chat_quota:<name>` | live |
| Tokens per 24h window / per minute | per consumer | `ai-token-ratelimit` 2.0.1 | **live 2026-09-13** |
| Balance refill | per consumer | quota-bot refill job | **live 2026-09-13** |
| `max_tokens` ceiling | per route | `request-validation` | live, global (70000); cannot vary per key |
| Concurrency + queue | node-wide | sgl-router | live as of 2026-09-04 |

Neither per-consumer control can weight input against output: ai-quota is
hardcoded `inputToken + outputToken`, and ai-token-ratelimit counts the same
total. So a tier expresses **how much** and **how fast**, never **of what**.

## Tiers

A tier is configuration attached to a consumer, not a different kind of
credential. One consumer per person or per integration; the tier is metadata.
Multiplying credential *types* would multiply the ways to get key-auth wrong,
and this deployment has already been bitten once by a stray `key-auth.internal`.

| Tier | For | Quota | Refill | Daily (24h window) | Tokens/min | `max_tokens` (recorded) |
|---|---|---|---|---|---|---|
| **trial** | evaluation, unvetted third parties | 1 M | manual | 500 K | 200 K | 2,048 |
| **team** | internal humans via OpenCode | 100 M | monthly | 20 M | 600 K | 32,768 |
| **service** | production integrations | 300 M | monthly | 30 M | 1 M | 16,384 |
| **batch** | offline/bulk, latency-tolerant | 500 M | monthly | 50 M | 300 K | 70,000 |
| **admin** | `quota-admin` | — | manual | none | none | gateway |

### Why these numbers (2026-09-13, from gateway.requests)

The first draft (10 M a month, 60 K a minute) was sized on per-token cost and
would have throttled every agent. Measured traffic instead:

| | median | p95 | max |
|---|---|---|---|
| tokens per agent request | 74 K | 134 K | 169 K |
| tokens per active minute (busiest key) | 122 K | 418 K | 1.04 M |
| tokens per day (busiest key) | | | 32.6 M |

**Per-minute limits start at 200 K.** The limiter refuses only once a counter is
already over, so a limit below one maximum-size request means one request a
minute. 600 K lets a team key's p95 minute through and trims bursts.

**Daily limits are the runaway guard, quotas the monthly budget.** A daily
window opens at the key's first request, so it is "24 hours of use", not a
calendar day — accepted rather than patching the plugin.

**The one exception to "admin never runs inference"** (2026-09-05). The bot's
`/key → Report` button asks qwen36-27b to write a consumer's report, and
authenticates that call with `quota-admin` — the only credential the bot holds.
It is metered like any other consumer: roughly 2.5-4k tokens per report against
`quota-admin`'s balance, which is why that key is funded rather than zeroed.
Nothing else about the admin tier changed: it still serves no end users, and a
report is an operator action, not traffic.


### What this does not fix

Cached input is charged the same as uncached, at ~70× its cost. That cannot be
fixed here: SGLang returns `prompt_tokens_details: null`, so cached tokens are
not attributable per request even though they are measurable node-wide.
Anthropic's approach — exclude cached input from rate limits entirely — is the
right model and is closed to us until SGLang reports it.

## Unassigned, and why

Two consumers were left without a tier rather than guessed at, because the tier
implies a quota and getting it wrong is either a lockout or a giveaway:

- **`acme`** — balance 499,637, under half of `trial`'s 1 M. The name suggests
  an external customer, which would be `service` (300 M) — a 600× increase from
  what they hold today.
- **`legacy-shared`** — balance 100,000,000, matching `batch` exactly, but the
  name suggests it is the shared credential the team used before per-person
  keys. If so it should be retired rather than tiered, and its traffic moved
  onto individual consumers.

`/keys` shows both with `—` and counts them, so they cannot be quietly
forgotten.

## How it is applied

- `higress-standalone/config/wasmplugins/ai-token-ratelimit.yaml` is a disabled
  shell; `apply.sh` installs it. quota-bot renders the real rules on every
  policy or key change and re-checks every minute, so an `apply.sh` run that
  resets the shell is undone within a minute. It PUTs only when the rules'
  fingerprint differs.
- Two rule_items (24h, 60s): within one item the first matching key wins. #4011's
  all-match overlay is why no two items may list the same window.
- The limiter shares the ledger Redis but fails OPEN; ai-quota still fails closed.
- The refill job marks `chat_refill:<name>` with `<mode>:<period>` after each
  reset, and announces every reset to the alert chats.
