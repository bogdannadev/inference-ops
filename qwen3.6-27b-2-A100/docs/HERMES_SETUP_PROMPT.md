# Hermes Agent setup prompt for this endpoint

Paste the block below into Hermes Agent (NousResearch `hermes-agent`) so it
configures itself for this node. Every number in it is measured on this
deployment as of 2026-09-13 (DFlash2 + HiCache config,
`tuning/docs/HICACHE_DFLASH2.md`). Update it when `--context-length`, the
gateway `max_tokens` ceiling or the tier limits change.

The API key is **not** in the prompt. Issue one with quota-bot `/newkey`; `/connect <name>`
shows its endpoint, model and key. Put the key in `~/.hermes/.env` as `QWEN_GW_API_KEY`.

This is the customer gateway (`qw38-27b-gw.duckdns.org`, per-key auth, quota and
limits), not the internal `qw36-27b.bnna.dev` edge.

---

```text
Configure yourself to use my self-hosted model endpoint. Edit ~/.hermes/config.yaml
(config.yaml is the single source of truth; secrets go only in ~/.hermes/.env).
Merge the settings below into the existing config without touching unrelated
keys, show me the diff, and never write the API key into config.yaml. If a key
name below does not exist in your installed version, tell me instead of
inventing one.

ENDPOINT — add it as a named custom provider
  providers:
    qwen-gw:
      api: https://qw38-27b-gw.duckdns.org/v1
      key_env: QWEN_GW_API_KEY          # I will put the key in ~/.hermes/.env
      transport: chat_completions
      default_model: qwen3.8-27b
      discover_models: false            # /v1/models here does not report the window
      request_timeout_seconds: 900
      stale_timeout_seconds: 300
      models:
        qwen3.8-27b:
          context_length: 169000
  model:
    provider: custom:qwen-gw
    default: qwen3.8-27b
- The key is sent as a normal Bearer token. "qwen3.8-27b" is Qwen3.8-27B in
  BF16: a thinking model with vision, served by SGLang behind an API gateway.

CONTEXT WINDOW — set explicitly, never rely on auto-detection
- This endpoint's /v1/models does NOT report a context length, so auto-detection
  will guess wrong. context_length 169000 above is the real window.
- 169,000 is a HARD ceiling on prompt + output together. The server never
  truncates: an over-long request fails with HTTP 400
  "Input length (N tokens) exceeds the maximum allowed length (M tokens)".
  Treat that error as "compress now and retry", never as a transient error.
- Any request asking for more than 70,000 output tokens (max_tokens or
  max_completion_tokens) is rejected by the gateway with HTTP 422. Never
  request more than 70000; leave room so prompt + max_tokens <= 169000.
- A request body may be at most 1 MB of JSON. A full 169K-token text context
  fits (~0.75 MB); inline base64 images can exceed it (HTTP 413).

COMPRESSION — compress well before the hard ceiling
Reasons, all measured on this node:
  * Every input token counts against my key's quota and per-minute limit at
    full price, including cached prefix tokens. A 150K-token turn costs 150K.
  * Uncached prefill runs at ~3,200 tokens/s: a cold 45K prompt takes ~14 s to
    the first token, a cold 160K prompt ~80-130 s. Cached prefixes come back
    in ~1-2 s.
  * Long sessions push each other out of the server's GPU prefix cache.
Settings:
  compression:
    enabled: true
    threshold: 0.60
    threshold_tokens: 100000        # compress by 100K tokens at the latest
    tail_mode: lean
    protect_first_n: 3
    protect_last_n: 20
    in_place: true
    proactive_prune_tokens: 0       # keep off: it rewrites already-sent history
                                    # and breaks the server's prefix cache
  auxiliary:
    compression:
      provider: main                # same endpoint and model
      reasoning_effort: low         # summaries do not need deep thinking

PREFIX-CACHE FRIENDLINESS (big latency win on this server)
- Keep the system prompt, tool definitions and their order byte-stable across
  turns. Do not inject timestamps, random ids or changing status lines into the
  system prompt or the early history. Append; do not rewrite earlier messages
  except during an actual compaction.

REASONING
- The server separates thinking into reasoning_content and, by default, thinks
  at reasoning_effort "xhigh". Thinking tokens are output tokens: they cost
  quota and latency (decode is ~55-70 tokens/s).
- Set agent.reasoning_effort: high for normal agent work. Use xhigh only when I
  ask for deep reasoning.
- Do not resend previous turns' reasoning_content in history; the server's
  template drops it anyway.

TOOL CALLING
- Use standard OpenAI function calling (tools / tool_calls). The server parses
  the model's native tool-call format into structured tool_calls, so do not
  switch to a text/XML tool protocol.

TIMEOUTS AND RETRIES
- Long cold prompts legitimately produce no bytes for up to ~2 minutes before
  the first token, and the server allows a request up to 900 s. That is why
  the provider entry sets request_timeout_seconds 900 and
  stale_timeout_seconds 300 (the 90 s default would kill valid 160K prefills).
- HTTP 429 type "rate_limit_exceeded": my key hit its per-minute or rolling
  24-hour token limit. Wait for the Retry-After header (seconds), then retry.
  Do not hammer it. If Retry-After is large (hours), stop and tell me.
- HTTP 403 type "insufficient_quota": the key's token balance is exhausted.
  Do NOT retry; tell me it needs a top-up.
- HTTP 401: the key is wrong or revoked. Do not retry; tell me.
- HTTP 400 context-length error: compress, then retry once.
- Keep agent.api_max_retries at 3 for 5xx and connection errors only.

VISION
- The model accepts images as OpenAI image_url content parts, but images add
  prompt tokens. Only attach them when the task needs them.

After applying: run one short test request (e.g. "Reply with OK") and report
the HTTP status, the model's reply, and usage.prompt_tokens /
completion_tokens. Then show me the compression and provider sections of the
final config.
```
