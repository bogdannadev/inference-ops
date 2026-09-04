# Key tiers — draft for review

**Status: tiers are RECORDED, not ENFORCED, as of 2026-09-04.**

`/tier <name> <tier>` writes the assignment to `chat_tier:<consumer>` in the
same Redis as the balances, and `/keys` shows it. Nothing reads it at request
time: ai-quota still charges a flat input+output total and cannot vary by tier,
and `ai-token-ratelimit` is bundled but not installed. Recording the intent is
what makes it reviewable and is the prerequisite for enforcing it — it is not
the enforcement.

Assigned so far: `quota-admin` → admin, `testafter` → team, `danila` → team.
`acme` and `legacy-shared` are deliberately unassigned — see "Unassigned" below.

**No balance was changed.** `/tier` reports when a balance differs from its
tier's quota and tells you the `/setquota` to align it, but never moves money on
its own.

The tier table below remains a draft. It proposes tiers for consumer and
production workloads, sized against measurements taken on this node on
2026-09-04 rather than copied from a commercial price list.

Companion: `docs/METRICS-PLAN.md` (build order), `docs/OBSERVABILITY.md`
(why quota is a single total-token number).

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

**The first two are both cold.** `--policy round_robin` sends them to different
replicas, and each has to prefill the prefix independently. A repeated prefix
costs full prefill *twice* on this two-replica node before it is warm on both.

That is inherent to the routing choice and not a bug — `cache_aware` was
disabled deliberately because it starved r0 with a shared hot system prompt. But
it halves the value of prefix caching for a single conversation, and it means
node-wide cache-hit figures (84.8%) understate what a cache-aware policy could
achieve. Worth revisiting if agent traffic grows.

## What can actually be enforced

Four controls exist. Only the first two are per-consumer.

| Control | Scope | Where | Status |
|---|---|---|---|
| Total-token balance | per consumer | `ai-quota` → `chat_quota:<name>` | live |
| Tokens per minute | per consumer | `ai-token-ratelimit` 2.0.1 | **bundled, unused** |
| `max_tokens` ceiling | per route | `request-validation` | live, global (70000) |
| Concurrency + queue | node-wide | sgl-router | live as of 2026-09-04 |

Neither per-consumer control can weight input against output: ai-quota is
hardcoded `inputToken + outputToken`, and ai-token-ratelimit counts the same
total. So a tier expresses **how much** and **how fast**, never **of what**.

## Draft tiers

A tier is configuration attached to a consumer, not a different kind of
credential. One consumer per person or per integration; the tier is metadata.
Multiplying credential *types* would multiply the ways to get key-auth wrong,
and this deployment has already been bitten once by a stray `key-auth.internal`.

| Tier | For | Quota | Refill | Tokens/min | `max_tokens` | Concurrency posture |
|---|---|---|---|---|---|---|
| **trial** | evaluation, unvetted third parties | 100 K | one-shot | 3,000 | 2,048 | may be starved first |
| **team** | internal humans via OpenCode | 10 M | monthly | 60,000 | 32,768 | normal |
| **service** | production integrations | 50 M | monthly | 120,000 | 16,384 | normal |
| **batch** | offline/bulk, latency-tolerant | 100 M | monthly | 30,000 | 70,000 | expected to queue |
| **admin** | `quota-admin` | n/a | n/a | n/a | n/a | management only, never inference |

### Why these numbers

**Tokens/min is the load-bearing control, not quota.** Quota stops a runaway
bill over weeks; TPM stops one consumer taking the box right now. The node
sustains ~387 output tok/s ≈ 23,000 output tokens/min at peak. Because the limit
counts input+output and real agent traffic runs 45:1 input-heavy, a 60,000 TPM
ceiling corresponds to roughly 1,300 output tokens/min from that consumer — a
few percent of the node. That asymmetry is deliberate: it is generous to
context-heavy traffic, which is cheap, and tight on generation, which is not.

**`max_tokens` is the sharpest per-request lever available.** It bounds decode,
the scarce resource, directly and per request — unlike quota and TPM, which are
blunt about direction. `trial` at 2,048 caps a single request at ~37s of decode.
`batch` keeps the current 70,000 because long generation is the point.

**`trial` is sized so it cannot hurt.** 100 K total tokens is a real evaluation
(roughly 3 long agent sessions) and 3,000 TPM is under 1% of the node.

**`service` gets more quota but a lower `max_tokens` than `team`.** Production
integrations should be steady and predictable; an integration that needs 32 K
output in one call is doing something a batch job should do.

### What this does not fix

Cached input is charged the same as uncached, at ~70× its cost. That cannot be
fixed here: SGLang returns `prompt_tokens_details: null`, so cached tokens are
not attributable per request even though they are measurable node-wide.
Anthropic's approach — exclude cached input from rate limits entirely — is the
right model and is closed to us until SGLang reports it.

## Unassigned, and why

Two consumers were left without a tier rather than guessed at, because the tier
implies a quota and getting it wrong is either a lockout or a giveaway:

- **`acme`** — balance 499,637. Sits between `trial` (100 K) and `service`
  (50 M). The name suggests an external customer, which would be `service`, but
  that is a 100× increase from what they hold today.
- **`legacy-shared`** — balance 100,000,000, matching `batch` exactly, but the
  name suggests it is the shared credential the team used before per-person
  keys. If so it should be retired rather than tiered, and its traffic moved
  onto individual consumers.

`/keys` shows both with `—` and counts them, so they cannot be quietly
forgotten.

## To apply

1. Add `ai-token-ratelimit` to `config/wasmplugins/` with per-consumer rules.
   Read #4011 first: it changed matching from first-match-wins to all-match
   OR-overlay, so **every** matching rule consumes and overlapping rules
   double-charge.
2. Per-route `request-validation` limits, so `max_tokens` can differ by tier —
   currently one global 70000, quoted to customers in the Caddyfile's 422 text.
   Change one, change both.
3. Record each consumer's tier somewhere durable. `consumers.conf` is the
   natural place; quota-bot already owns that file.
4. Re-seed quotas to the tier's value via `/setquota`.

Redis becomes a second request-path dependency for rate limiting as well as
quota. It is already the hardest failure mode on the paid path — ai-quota has no
fail-open — so this raises the stakes on the ledger rather than adding a new
kind of risk.
