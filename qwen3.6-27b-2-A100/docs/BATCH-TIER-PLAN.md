# Batch tier — plan (not built)

Written 2026-09-17. Nothing here is implemented.

## Why

Batch-style clients already use this node: sequential scripts sending one
non-streaming request every few seconds, short prompts with a shared
instruction block. They are loops, not Batch API clients: there is nothing to
submit, poll or download, and no off-peak scheduling.

What they need is not scheduling:

1. **Their own keys**, so their usage is attributed and metered per consumer
   like everyone else's.
2. **A rate ceiling**, so a batch loop that grows to several workers does not
   compete with interactive users at equal priority.

SGLang 0.5.19 has **no** `/v1/batches` and no `/v1/files`
(`srt/entrypoints/openai/` has chat, completions, responses, embedding,
classify, rerank, score, tokenize, transcription — no batch). A real Batch API
would be ours to build, so this plan covers only what the existing gateway can
already enforce.

## Scope

A `batch` key tier: normal synchronous requests, per-key metering, a rate
ceiling that keeps a batch loop from crowding out people, and (optional)
pinning to one replica.

Out of scope: file upload, job ids, polling, result files, off-peak scheduling,
engine-side priority. See "If a real Batch API is wanted" below.

## Steps

1. **Tier definition** (`quota-bot/bot.cs`, `Policy.All`; table in
   `docs/KEY-TIERS.md`). Proposed defaults, to confirm against a week of
   `engine.requests` before building:
   - `tpm` ~30,000 tokens/min. Measured over 2026-09-15..17 the current job
     runs at 16.5K tokens/min on average, p95 20.7K, peak 24.7K, so 30K leaves
     headroom for a well-behaved loop while capping a runaway one. Re-measure
     before setting it:
     `SELECT quantile(0.95)(t) FROM (SELECT toStartOfMinute(received_at) m,
     sum(prompt_tokens+completion_tokens) t FROM engine.requests
     WHERE consumer='' GROUP BY m)`.
   - `daily` ~30M tokens (the job alone is ~24M/day), `quota`/`refill` as for
     a service key.
   - `max_tokens`: not enforceable per key (one global 70,000 ceiling, see
     KEY-TIERS.md).
2. **Issue keys and move the traffic off the shared key.** One key per batch
   client. Requires telling each owner to switch hostname and key; the direct
   hostname keeps working meanwhile.
3. **Optional replica pinning.** The router accepts `x-smg-target-worker`
   (tuning/docs/ROUTING.md). A Higress `transformer` rule on the chat route
   could add it for batch consumers, keeping batch on one replica and leaving
   the other free for interactive traffic. Costs the batch job its prefix cache
   on the other replica, so measure before keeping it: it is only worth it if
   both replicas are otherwise busy.
   - Blocked by the same limitation as per-key `max_tokens`: a WasmPlugin
     matchRule cannot select by consumer, so the rule would need the plugin to
     read `X-Mse-Consumer` itself. Check `transformer`'s matcher support first.
4. **Observability.** No new metric: a batch key shows up per consumer in
   `/top`, `/usage`, `/p95` and `/errors` as soon as it exists. Worth adding to
   `docs/OBSERVABILITY.md` that a batch key's 429 count is the signal that its
   ceiling is too low.
5. **Docs.** `docs/KEY-TIERS.md` (tier row), `quota-bot/README.md` (what to tell
   a batch user), `docs/OPERATIONS.md` (issuing a batch key).

## Effort and risk

Config and documentation, roughly an hour, no new service and no restart of the
engines. The gateway already enforces every limit involved; step 3 is the only
part that needs a plugin change and a measurement.

## If a real Batch API is wanted

A queue service accepting a JSONL file and returning a job id: store the file,
submit items when the node is quiet, retry failures, write a results file, plus
auth, metering and cleanup. A few days of work and a new component to run. Only
worth it if several clients send large jobs, or to sell off-peak capacity at a
discount.

True low-priority scheduling (a batch request yielding mid-flight to an
interactive one) needs engine support; SGLang's priority scheduling was
rejected for this build in the 2026-09-16 abort-fix work.

## Related

`docs/KEY-TIERS.md`, `docs/OBSERVABILITY.md`, `tuning/docs/ROUTING.md`,
`higress-standalone/OPERATING.md`.
