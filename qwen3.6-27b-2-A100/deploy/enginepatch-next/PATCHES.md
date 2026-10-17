# Engine patches: what each one changes

One page per patch, each as before/after flow through the engine. `README.md`
has the install, verify and roll procedure; this file is the picture. Upstream
paths are relative to `/sgl-workspace/sglang/python/sglang/srt/` in the pinned
image (SGLang v0.5.21).

```
 client ─► Caddy ─► Higress ─► smg router ─► ENGINE (r0 / r1)
                                              │
  entrypoints/openai/serving_chat.py ─────────┤ [5] effort: request → key default → server
  entrypoints/openai/protocol.py ─────────────┤ [1] skip_special_tokens forced True
  multimodal/processors/base_processor.py ────┤ [2] placeholder/payload mismatch → 400
  managers/tokenizer_manager.py ──────────────┤ [5] per-key context window
                                              ▼
  mem_cache/unified_radix_cache.py ───────────┤ [6a] prefix match recorded on the request
  mem_cache/prefill_budget.py ────────────────┤ [3] chunk clamped to physical KV
                                              ▼
  layers/quantization/fp8.py ─────────────────┤ [4] prefill GEMM M≥2049 → dequant + cuBLAS
                                              ▼
  managers/scheduler_components/output_streamer.py ─ [6b] "diag" added to cache details
```

| # | What | Kind | Live since | Hook in `sitecustomize.py` |
|---|---|---|---|---|
| 1 | force `skip_special_tokens` | bug fix | 2026-09-22 | `_patch_proto` |
| 2 | mismatch → 400 + actionable text | bug fix | 2026-09-22 | `_patch_mm` |
| 3 | clamp continuing chunk to physical KV | crash fix | 2026-10-11 | `_patch_budget` |
| 4 | FP8 prefill on dequant + cuBLAS | speed | 2026-10-12 | `_patch_fp8` |
| 5 | per-key context window + effort | policy | 2026-10-14 | `_patch_chat`, `_patch_tm` |
| 6 | cache diagnostics | observe only | not yet | `_patch_urc`, `_patch_out` |
| R | key rotation: accept `SGLANG_API_KEY_ALT` too | temporary | not yet | `_patch_auth` |

---

## 1. Force `skip_special_tokens`

Wraps `ChatCompletionRequest.to_sampling_params` (`entrypoints/openai/protocol.py`).

**Before**
```
 client request  skip_special_tokens: false   (sent with every reasoning request)
      │
      ▼
 model emits a vision control token
      │
      ▼
 detokenizer renders it as literal markup in the reply TEXT
      │
      ▼
 client stores the reply in history ─► next turn resends it
      │
      ▼
 tokenizer turns the markup back into a real IMAGE token, no image attached
      │
      ▼
 500 on every turn, on every replica, forever (the session is poisoned)
```

**After**
```
 client request  skip_special_tokens: false
      │
      ▼
 to_sampling_params ─► [1] forced True
      │
      ▼
 detokenizer drops special-flagged tokens (control, vision, audio)
      │   tool_call, tool_response, think are not special-flagged: they survive,
      │   so the tool-call and reasoning parsers are unaffected
      ▼
 nothing poisonous reaches the history
```

Measured: gateway 500s per week 11, 44, then 0 every week since.

---

## 2. Placeholder/payload mismatch is a 400

Wraps `BaseMultimodalProcessor.legacy_load_mm_data`
(`multimodal/processors/base_processor.py`). Catches the history 1 can no
longer prevent: one poisoned before the patch.

**Before**
```
 prompt has more IMAGE placeholders than images
      │
      ▼
 StopIteration ─► RuntimeError ─► HTTP 500, body ends in a dangling colon
      │
      ▼
 router: 500 is retryable ─► retry ─► each failure is a breaker credit
      │
      ▼
 10 failures ─► circuit breaker opens ─► replica out of rotation (up to 3 h 10 min)
```

**After**
```
 prompt has more IMAGE placeholders than images
      │
      ▼
 RuntimeError whose __context__ is StopIteration ─► [2] ValueError ─► HTTP 400
      │   "...Start a new conversation" (plain words, no markup)
      ▼
 router: 400 is not retryable ─► no retry, no breaker credit, one failed turn

 any other RuntimeError (GPU OOM in preprocessing) ─► still 500, still alerts
```

---

## 3. Clamp continuing prefill chunks to physical KV

Wraps `PrefillBudget.available_chunk_tokens` (`mem_cache/prefill_budget.py`).

**Before**
```
 long prompt, chunked 4096 at a time
      │
      ▼
 chunk N clamped to the budget left (say 1632) ─► budget now ≤ 0
      │
      ▼
 chunk N+1: available_chunk_tokens
      │   budget > 0 ? budget : chunk_limit      ◄ falls back to a FULL 4096
      ▼
 allocator: 4096 asked, 960 free, 0 evictable
      │
      ▼
 "Prefill out of memory" ─► scheduler dies ─► replica restarts (r0, 2026-10-09)
```

**After**
```
 chunk N+1: available_chunk_tokens
      │
      ├── budget > 0 ─► upstream, unchanged
      │
      └── budget ≤ 0 ─► [3] fits = free + evictable − 1 page, whole pages
               │
               ├── fits ≥ 1 page ─► chunk = min(4096, fits)        (clamped)
               └── fits < 1 page ─► None: park this round           (parked)
                        │
                        ▼
                  decode runs, frees or retracts KV ─► chunk resumes next round
```

Measured: a drift-injector drill crashed stock code at the 2026-10-09 state;
patched code parked, resumed, completed and leaked nothing. Live: 0 crashes.

---

## 4. FP8 prefill on dequant + cuBLAS

Wraps `Fp8LinearMethod.apply` (`layers/quantization/fp8.py`); kernel in
`fp8_prefill_dequant.py`. A100 has no FP8 tensor cores, so the FP8 checkpoint
runs weight-only on Marlin.

**Before**
```
 every FP8 linear, any M ─► Marlin (dequant inside the GEMM tile)
      │
      ├── decode / verify, small M: bandwidth-bound ─► Marlin is right
      └── prefill chunk, M = 4096: compute-bound   ─► Marlin is slow
                                                     (cold 8K prefill +38% vs bf16)
```

**After**
```
 FP8 linear
      │
      ├── M ≤ 2048, graph capture, compile ─► Marlin (unchanged)
      │
      └── M ≥ 2049 ─► [4] Triton kernel reads Marlin's tiled FP8 layout
                         ─► bf16 weight in a scratch slab (≤ 128 MB)
                         ─► cuBLAS bf16 GEMM
                     (the weight is bit-identical to what Marlin multiplies)
```

Measured: drained A/B cold prefill −20% (8K) to −14% (128K), NLL unchanged.
Live, median cold-prefill tok/s:

| prompt | bf16 weights | FP8, no patch 4 | FP8 + patch 4 |
|---|---|---|---|
| 8–16K | 3420 | 2802 | 3160 |
| 16–48K | 3204 | 2445 | 3072 |
| 48K+ | 2494 | 2153 | 2425 |

---

## 5. Per-key context window and reasoning effort

Wraps `OpenAIServingChat._convert_to_internal_request` and `_process_messages`
(`entrypoints/openai/serving_chat.py`), and `TokenizerManager._validate_one_request`
(`managers/tokenizer_manager.py`). Policy from `keypolicy.py`, file written by
quota-bot.

**Before**
```
 request: reasoning_effort "low"
      │
      ▼
 _process_messages merges --default-chat-template-kwargs
      │   server default xhigh OVERWRITES the client's effort (only "none" got through)
      ▼
 template renders xhigh for everyone

 context: one --context-length for every key
```

**After**
```
 request (consumer label set by the gateway)
      │
      ▼
 _convert_to_internal_request ─► [5] consumer remembered for this request
      │
      ▼
 _process_messages ─► [5] effort = request ▸ key default ▸ server default
      │                  pinned into chat_template_kwargs before the merge
      │                  high/max → xhigh, minimal → low (template knows 3)
      ▼
 _validate_one_request ─► upstream checks vs 262,144 (unchanged)
      │
      └─► [5] key window from /opt/keypolicy/policy.json (default 169,000)
               no max_tokens ─► window − prompt − 1
```

Measured: prompt "hi" renders 53 tokens at xhigh, 41 at low, 11 at medium;
before the patch every request rendered 53.

---

## 6. Cache diagnostics (observe only, not live yet)

Wraps `UnifiedRadixCache.match_prefix` (`mem_cache/unified_radix_cache.py`) and
`SchedulerOutputStreamer.get_cached_tokens_details`
(`managers/scheduler_components/output_streamer.py`); logic in `cache_diag.py`.

A hit on this hybrid model needs KV **and** a GDN state at its end:

```
 tokens  0 ─────────────── B ───────── bnd ──── F ──────────────── prompt end
         │ reused (hit)     │◄ recompute         │◄ client changed: unavoidable ►│
                            │ (engine loss F−B)  │
                            │◄ checkpoint at ►│  │
                            │  message start   │  │
                            │  saves bnd − B   │  │
```

### 6a. At admission

**Before**
```
 scheduler: init_next_round_input(tree_cache)   (scheduler.py, before add_one_req)
      │
      ▼
 UnifiedRadixCache.match_prefix
      │  radix walk ─► F = full_kv_hit_length
      │  mamba finalizer ─► B = deepest GDN state ≤ F, branch point
      ▼
 MatchResult ─► req: prefix (B), host hit, branch
      │             F ✗ discarded
      ▼
 prefill from B
```

**After**
```
 UnifiedRadixCache.match_prefix
      │  upstream match, unchanged
      ├──► [6a] req._cache_diag = (F, device hit, host hit, branch)
      ▼          last write wins = the match that admitted the request
 MatchResult ─► req (unchanged) ─► prefill from B (unchanged)
```

### 6b. At finish

**Before**
```
 get_cached_tokens_details(req)
      │  hit  ─► {"device", "host", "storage", "storage_backend"}
      │  cold ─► None
      ▼
 meta_info ─► request-metrics/*.jsonl ─► Vector ─► engine.requests cached_*
```

**After**
```
 get_cached_tokens_details(req)
      │  upstream dict, unchanged
      ├── not finished ─► returned untouched
      └── finished ─► [6b] describe(req): one vectorised scan for im_start (248045)
                           ─► "diag": {kv, hit, branch, msgs, bnd, after}
                           cold ─► {"device": 0, "host": 0, "diag": ...}
      ▼
 meta_info ─► same file ─► Vector (+ diag_* mapping) ─► engine.requests diag_*
```

Cost: one tuple per match, ~0.25 ms per finished request at 262K tokens.
Unchanged: scheduling, cache contents, eviction, outputs, Prometheus counters.

**Roll status:**

1. ClickHouse `diag_*` columns: applied 2026-10-17.
2. Vector mapping: pending.
3. Engine roll (r1 drained first): pending.
