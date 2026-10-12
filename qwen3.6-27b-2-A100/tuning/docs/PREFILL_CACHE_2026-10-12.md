# Prefill on the FP8 stack: where the time goes and what fixes it (2026-10-12)

Window: 2026-10-11 13:50 UTC (FP8 live) to 2026-10-12 ~10:00 UTC, both replicas.
1,031 requests. Sources: `logs/r*/request-metrics`, ClickHouse `engine.requests`
joined to `gateway.requests` on `chat_id = rid`, Prometheus `sglang:*`.

## 1. The shape of the problem

| | |
|---|---|
| prompt tokens served from cache | 95.3% (62.7M device, 0.3M host, 0 storage) |
| prefill time, share of request time | 1,641 s = 13% |
| queue time, all requests | 39 s total |
| median TTFT | 0.7–1.2 s |
| requests recomputing > 16K tokens | 40 (4%) = 57% of prefill tokens, 54% of prefill time, TTFT 13–69 s |

The service problem is the tail, not the median.

## 2. Root cause of most of the tail: the chat template rewrites history

`--default-chat-template-kwargs '{"preserve_thinking": false}'`. With it, the
Qwen3.8 template renders an assistant turn **with** its `<think>` block only
while it lies after the last real user message (`loop.index0 >
ns.last_query_index`), and **without** it once a newer user message exists.

```
cache after the last tool turn (the GPU holds exactly these tokens):

  [sys+tools][user1][a1 <think>R1</think> call1][tool1][a2 <think>R2</think> call2][tool2] … [a4 <think>R4</think> call4]
  ████████████████████████████████████████████████████████████████████████████████████████████████████████████

tool turn: the next prompt appends, nothing earlier changes

  [sys+tools][user1][a1 <think>R1</think> call1][tool1] … [a4 <think>R4</think> call4][tool4]
  ███████████████████████████████████████████ hit ███████████████████████████████████████▒▒▒▒▒▒   ← only tool4 is new

user sends a new message: last_query_index jumps, a1..a4 lose their <think> blocks

  [sys+tools][user1][a1 call1][tool1][a2 call2][tool2] … [a4 call4][user2]
  ███████████████████│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ recomputed ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                     ↑ first changed token = first assistant turn after user1
```

One changed token invalidates everything after it, for both caches:

```
attention KV:  K/V at position p are computed from the hidden state, which depends on tokens 0..p.
GDN state:     the recurrent state at p is a running summary of tokens 0..p.

radix tree:  root ── sys+tools ── user1 ──┬── a1<think>… tool1 … a4 …        (cached, now unreachable)
                                          └── a1 … tool1 … a4 … user2       (new branch, all recomputed)
```

**Offline proof** with the real template and tokenizer (synthetic session:
4 tool turns of ~6K tokens each, then a new user message):

| preserve_thinking | client resends reasoning | tool turn: hit / recompute | new user message: hit / recompute |
|---|---|---|---|
| false (live) | no | 25,333 / 7,543 | **2,706 / 30,165** |
| false (live) | yes | 26,177 / 7,823 | **2,706 / 30,165** |
| true | no | 25,333 / 7,543 | 25,333 / 7,554 |
| true | yes | 26,177 / 7,823 | 26,177 / 7,834 |

**Production confirms it.** `chat_round` from the gateway counts real user
messages, so a +1 marks a new-message turn (prompts > 20K):

| turn type | n | recompute p50 / p90 | recomputed | prefill s p50 / p90 | hit rate |
|---|---|---|---|---|---|
| tool turn | 746 | 152 / 3,304 | 976K | 0.14 / 1.7 | 97.9% |
| **new user message** | **59** | **4,216 / 30,931** | **721K** | **2.4 / 15.3** (max 46.8) | **81.0%** |

These are the turns where a person is waiting at the keyboard, so they are the
most visible TTFT on the service.

**What `preserve_thinking: true` costs.** OpenCode does resend
`reasoning_content`: in-loop hits cover the previous turn's whole output, page
aligned. So keeping past reasoning makes history longer. The template drops a
median 1.4K tokens (p90 4.7K) at each user message today; that is what
history would keep instead. Measured reasoning share of output: 47–80% per
consumer. Trade: +1.4–4.7K context per user turn vs up to 31K recomputed.
Qwen3.8's own default is `true`; the pin to `false` was made to reproduce
Qwen3.6 rendering (docker-compose.yml, Qwen3.8 upgrade notes).

`--strip-thinking-cache` is not a fix: it stops caching model output at all,
which would break the in-loop hits instead.

## 3. The rest of the tail

From a heuristic split of the 35 biggest misses (same consumer, not same session):

```
 r0                                  r1
 ┌───────────────────────┐          ┌───────────────────────┐
 │ session A ███████████ │          │                       │
 └───────────────────────┘          └───────────────────────┘
          ▲ shared L3 on host tmpfs: 15K KV pages + 549 GDN states written
          │ storage hits served in 20 h: 0 tokens
 request for A routed to r1 ──────────────────────► full prefill (6 misses, ~400K tokens)
```

- ~6 misses: the context lived on the other replica. The shared L3 should cover
  this and served nothing. Open; likely the GDN state requirement (a hit needs a
  state at the match point, and L3 holds 549 states against 15K KV pages).
- ~5: genuinely new sessions.
- The host tier is full (1.8M tokens per replica) but loaded back 160K tokens in 20 h.

## 4. Cold prefill speed (kernel side)

Cold prefill on FP8 (TRIAL_2026-10-11): 8K 3.0 s, 32K 13.1, 64K 29.4, 128K 71.6
(bf16 was 2.2 / 10.2 / 25.0 / 68.0). Marlin is weight-only: right for decode,
but in a 4096-token chunk the in-loop dequant costs tensor-core time.

```
decode / verify (M <= 32)                prefill chunk (M = 4096)
x ─► Marlin: fp8 → bf16 in registers     x ─► Triton: Marlin bytes → bf16 slab (≤128 MB scratch, ~0.3 ms/layer)
     → MMA. Bandwidth-bound: 1 B/weight      ─► cuBLAS bf16 GEMM, no dequant in the inner loop
     is why decode got +40-61%.                  (compute-bound: this is the regime cuBLAS wins)
```

Engine patch 4 (`deploy/enginepatch-next/fp8_prefill_dequant.py`) does this for
M >= 2049. CPU-verified bit-exact against upstream's own
`prepare_fp8_layer_for_marlin`.

**Measured 2026-10-12, r1 drained** (arms `P4_dequant_preserve` vs control
`P4off_preserve`: identical, patch switched off by `SGLANG_PATCH_FP8_DEQUANT=0`):

- GPU bench (`bench_fp8_dequant.py`): layout 100% bit-exact against the real
  `gptq_marlin_repack` on all six layer shapes. The first kernel did only ~200 GB/s
  (int64 division and a per-element scale gather) and LOST at M=2048. Hoisting the
  index math into per-row/per-column vectors with constant divisors gave ~500 GB/s,
  and a tile sweep (32x128, 4 warps) ~825 GB/s.
- Per forward, all FP8 linears: M=4096 1,322 → 1,064 ms (1.24x); M=2048 651 → 604 ms (1.08x).

| gate | patch off | patch 4 |
|---|---|---|
| cold prefill 8K / 32K / 64K / 128K (s) | 3.13 / 13.05 / 29.35 / 71.48 | **2.49 / 10.60 / 24.28 / 61.18** (−20 / −19 / −17 / −14%) |
| ΔNLL, 8 long docs 8K–128K | — | −0.017 ± 0.011 nats/token, 4 worse / 4 better |
| ΔNLL, 16 short docs | — | +0.003 ± 0.003, 18 worse / 14 better |
| lookup to 128K | 18/18 | 18/18 |
| smoke | — | 7/7 |
| spec_eval kv/order/arith, determinism (vs T2) | 5/5 ×3, ✓ | 5/5 ×3, ✓; decode and accept length within spread |
| mem_stress 160K + 3×60K | (T2: clean, big TTFT 185 s) | clean, 0 OOM, big TTFT **157 s** |

Better than estimated: the GEMMs gain 1.24x and the bench shows Marlin was further
from cuBLAS at M=4096 than the end-to-end +38% implied. LIVE 2026-10-12 on r1 and
r0 (copied into `deploy/enginepatch/`, one roll each).

FA2 prefill attention (B1: 15–19% faster on the attention part) is next.

## 5. Order

1. `preserve_thinking: true` on both replicas — LIVE 2026-10-12 (r1 ~12:08 UTC,
   r0 after). Re-measure with the section 2 split after 24 h.
2. Patch 4: gates passed on drained r1 (section 4); LIVE 2026-10-12.
3. Investigate the zero L3 hits (section 3).
4. FA2 attention shim.
