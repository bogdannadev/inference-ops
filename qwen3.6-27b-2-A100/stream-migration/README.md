# Stream migration: a turn continues on the other replica

A streaming chat turn running on one replica moves to the other without the
client noticing: no cut, no repeated or missing tokens. Use: draining a
replica for a roll without waiting for its long turns, and surviving a replica
crash mid-answer.

Decided 2026-10-17: **build it into smg (Rust), contributed upstream by
Danila**, plus an engine-side resume (engine patch 9, Python). Not adopted:
NVIDIA Dynamo (its Migration operator does this, but its frontend replaces
SGLang's HTTP layer and our router, so engine patches 1, 2 and 5 would be
rebuilt; spec decoding, HiCache and hybrid models are undocumented there).
Not chosen: a separate .NET relay (an extra hop; smg already owns the stream).

## How it works

```
 client ═══ Higress ═══ smg ═══ r1                 smg keeps, per streaming request:
                         │                           the request body, the worker,
                         │                           output token ids seen so far
   r1 drains (or dies) ──┤
                         │  r1 flushes the turn's KV + GDN tail to L3 (patch 8)
                         │  and ends the stream with finish_reason "migrate" (patch 9)
                         ▼
                        smg ─► r0  POST /v1/chat/completions, resume form:
                         │         original body + output token ids so far
                         │     r0 (patch 9): render the template exactly as r1 did,
                         │         append the output ids -> L3 hit up to the tail,
                         │         replay the generated text through the reasoning and
                         │         tool-call parsers silently, stream NEW chunks only,
                         │         usage cumulative over both legs
                         ▼
 client ◄══ one SSE stream: smg drops r1's "migrate" ending, splices r0's chunks
```

The router stays model-agnostic: it never tokenizes, renders a template or
parses tool calls. Everything that depends on the model stays in SGLang, the
same code on both replicas.

## Contract between smg and the engine (to be settled before coding)

| Item | Proposal |
|---|---|
| Token ids out | request `return_token_ids: true` (exists in v0.5.21 `ChatCompletionRequest`); smg strips them before the client sees a chunk, unless the client asked |
| Prompt ids | `return_prompt_token_ids: true` (exists), so a resume can be checked against the same prompt |
| Migrate ending | new `finish_reason: "migrate"` on the last chunk, sent only by a draining engine (patch 9), never reaches the client |
| Resume request | an extension field, e.g. `sglext.resume: {output_ids: [...], prompt_len: N}`; the engine refuses it if its rendering of the prompt differs from `prompt_len` |
| Usage | the resumed leg reports cumulative prompt/completion/reasoning tokens; the migrated leg is ended without usage |
| Not migrated | `n > 1`, guided decoding / JSON schema (grammar state; Dynamo refuses these too), requests past a length cap |
| Trigger | engine-initiated `migrate` ending (drain), and a stream that breaks before `[DONE]` with token state known (crash) |

## Open questions

1. Upstream home and tag of the router matching the deployed `sglang-router`
   0.3.2 (shipped in the `lmsysorg/sglang` image): fork there, at that tag.
2. Where smg buffers per-stream state, and its memory cap (Dynamo caps by
   sequence length).
3. Billing: the gateway and the ledger charge from the final usage chunk and
   from engine.requests rows. The migrated leg writes an aborted row (charged
   0) and the resumed leg must carry the whole turn. Verify against
   `docs/` metering rules before anything goes live.
4. Parser replay in SGLang: how to restore the reasoning and tool-call parser
   state from already generated text without emitting it.
5. Upstream acceptance: whether the migrate/resume fields can be proposed to
   SGLang as well, so patch 9 becomes unnecessary.

## Plan

| | What | Where |
|---|---|---|
| M0 | Fork the router repo at the deployed tag; build it; run it against drained r1 next to production | smg fork |
| M1 | Engine patch 9 behind a flag: token ids out, migrate ending, resume | `deploy/enginepatch-next/` |
| M2 | smg: per-stream token state, splice on `migrate`, resume to the other worker | smg fork |
| M3 | Test on one replica: drain r1 mid-turn into a second r1-like worker (or r1 itself after a restart, with the L3 flush), compare output and usage with an unmigrated run | drained r1 |
| M4 | Billing check against the ledger and engine.requests | |
| M5 | Live behind a router flag; propose upstream | |

Prerequisite: the L3 handoff (`../tuning/docs/CACHE_HANDOFF_PLAN_2026-10-17.md`),
so a resume hits cache instead of re-prefilling 100K+ tokens.
