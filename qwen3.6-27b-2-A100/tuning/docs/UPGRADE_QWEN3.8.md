# Qwen3.6-27B → Qwen3.8-27B — 2026-08-15

**Outcome: a weights swap. No engine change, no image change, one new flag.**

`--model-path Qwen/Qwen3.6-27B` → `Qwen/Qwen3.8-27B`, plus
`--default-chat-template-kwargs`, on both replicas. The engine digest
(`sha256:16aba892…`, `v0.5.17-cu130`) is untouched.

## Why no new engine

v0.5.17 is still the newest stable release — checked 2026-08-15; v0.5.16 was
07-25 and nothing has shipped since 08-08. "Newer" means a nightly, and we do
not run nightlies.

More to the point, 3.8 needs no new code. It is the *same architecture*:

| | Qwen3.6-27B | Qwen3.8-27B |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | **same** |
| `model_type` | `qwen3_5` | **same** |
| `config.json` | — | **byte-identical except `transformers_version`** |
| tensors / `total_size` | 1199 / 51.75 GiB | **1199 / 51.75 GiB** |
| module tree | `language_model` 850, `visual` 333, `mtp` 15, `lm_head` 1 | **identical** |
| tokenizer vocab | 248044 | **248044**, same merges / normalizer / pre-tokenizer |
| layer plan | `3× linear_attention → 1× full_attention`, ×16 | **same** |
| `head_dim` / `intermediate_size` | 256 / 17408 | **same** |

Reproduce with:

```bash
for m in Qwen3.6-27B Qwen3.8-27B; do
  curl -sL "https://huggingface.co/Qwen/$m/resolve/main/config.json" -o "$m.json"
done
diff <(jq -S . Qwen3.6-27B.json) <(jq -S . Qwen3.8-27B.json)   # only transformers_version
```

Two things that *look* like changes and are not:

- **18 shards vs 15.** Repacking. `metadata.total_size` is identical to the
  byte, and so is the tensor count.
- **Tokenizer sha256 differs.** Only `added_tokens`: 7 reserved audio/TTS
  specials (248070–248076, `<|audio_start|>`, `<tts_pad>`, …). The vocab, and
  therefore every existing token ID, is unchanged.

`Qwen3_5ForConditionalGeneration` is the class v0.5.17 already registers, so
the model simply loads.

## Why not `lmsysorg/sglang:qwen38-27b-cu129`

The obvious move — the vendor's day-one image — is the wrong one here.

- Its wheel reports `0.0.0.dev0+qwen38.27b.g561c8f3`. That commit is not in
  `sgl-project/sglang`: it is the head of **PR #34859**, from a fork
  (`yhyang201`), **still open and `mergeable: false` (conflicted)**.
  125 files, +18,955 / −418.
- Its image LABEL claims `nightly-dev-cu12-20260814-c4271c3f` — a *different*
  commit, inherited from the base image. The label does not describe the
  wheel. This is precisely the provenance hole the 2026-08-08 digest pin
  closed; taking this image reopens it.
- **Nothing in the PR targets SM80.** The new kernels are `hopper_bf16_gemv`,
  `sm120_fp8_gemv`, `fp8_blockwise_scaled_mm_sm120`,
  `…_dense_bf16_gemm_sm100_splitk`, MNNVL CuteDSL all-reduce (GB200/GB300
  NVL72), DeepEP v2, MoE runner work. The companion docs commit is *"Add GB300
  cells and benchmarks for Qwen3.8-27B"*. On 2× A100 80GB PCIe (SM 8.0, TP=1,
  dense) that is all risk and no upside.
- It also moves cu130 → cu129 and pulls a dev FlashInfer
  (`0.6.18.dev20260807`) underneath `--attention-backend flashinfer`. The
  driver (610.57.04, CUDA UMD 13.3) would run it, but changing CUDA + torch +
  FlashInfer + engine at once invalidates every A/B baseline in `tuning/`.

Re-evaluate only if #34859 merges and lands in a tagged release.

## The one real change: the chat template

3.8 ships a reworked template. Both changes are silent on a model swap and
both affect every request.

### `reasoning_effort` (new)

Defaults to **`xhigh`**, which injects a 209-char *"Reasoning effort is set to
xhigh. Please think carefully through the task, validate key assumptions…"*
instruction into the system message of every request.

The tiers are **`low` / `medium` / `xhigh` only**. There is no `medium`
branch in the template, so `medium` injects *nothing* — that, not some middle
setting, is how you reproduce 3.6's rendering.

Everything else **hard-raises**, verified against the real template:

```
none     400 -> Unexpected reasoning effort none.    Supported types are xhigh (default), medium, and low.
minimal  400 -> Unexpected reasoning effort minimal. …
low      OK  -> injects instruction
medium   OK  -> no instruction
high     400 -> Unexpected reasoning effort high.    …
xhigh    OK  -> injects instruction
max      400 -> Unexpected reasoning effort max.     …
0.5      400 -> Unexpected reasoning effort 0.5.     …
```

This is a live trap. SGLang's protocol layer accepts the whole OpenAI tier
list (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`, or a float in
[0, 0.99]) and hands it straight to Jinja. **A client sending the ordinary
OpenAI `reasoning_effort: "high"` gets a 400.** On 3.6 that value is silently
ignored, so the model swap is what arms it.

Setting the value server-side disarms it — see the precedence note below.

### `preserve_thinking` (default flipped, off → on)

Controls whether *superseded* assistant turns keep their `<think>` blocks when
history is re-rendered. The current turn always keeps its thinking either way
(`loop.index0 > ns.last_query_index`), so tool-call loops within a turn are
unaffected.

3.6 stripped it; 3.8 keeps it. The problem is that `reasoning_content` is an
SGLang/vLLM extension, not OpenAI schema — a standard client parses the answer
and replays only `content`. The template still emits the wrapper around
nothing:

```
<|im_start|>assistant
<think>

</think>

I split foo() into parse() and emit().
```

Empty think blocks on every prior turn. Minor in tokens; bad as a prior for a
thinking model, more so at `xhigh`.

**Not yet verified:** whether OpenCode replays `reasoning_content`. Settle it
in Langfuse the way `session_id` was settled — open a multi-turn trace and
look for the field on assistant messages. If it *does* replay it, revisit;
`preserve_thinking: true` would then be genuinely useful.

### The flag

```
--default-chat-template-kwargs '{"reasoning_effort":"xhigh","preserve_thinking":false}'
```

`xhigh` is the deployment's chosen effort tier (and the model's own default).
`preserve_thinking: false` reproduces 3.6's rendering **byte-identically in
6/6 message shapes** — single-turn, single-turn with tools, multi-turn with
and without `reasoning_content`, tool-call round trip, and no-system-message.
Holding prompt rendering constant keeps the weights the only changed variable,
so the benchmark deltas below stay interpretable.

### Precedence — this flag is a LOCK for `reasoning_effort`

Its `--help` says "Default … applied to every request when not overridden
per-request. Per-request `chat_template_kwargs` takes precedence." **For
`reasoning_effort` that is wrong.** Traced through `serving_chat.py` @ v0.5.17:

1. `_convert_to_internal_request` **pops** `reasoning_effort` out of
   `request.chat_template_kwargs` and promotes it to `request.reasoning_effort`.
2. `_process_messages` merges the server defaults with `setdefault` — the key
   was just popped, so **our value lands**.
3. At render: `extra_template_kwargs["reasoning_effort"] = request.reasoning_effort`,
   then `.update(request.chat_template_kwargs)` — **the server value overwrites
   the request one.**

So our `xhigh` wins over both the top-level OpenAI field and the client's
`chat_template_kwargs`. That is what makes the `"high"` → 400 trap harmless,
and it is also why clients cannot dial effort down per-request. Accepted
deliberately. `preserve_thinking` is not popped, so `setdefault` behaves
normally and it stays client-overridable.

## Rollout

Names do **not** change. Containers, networks, router worker-urls, the Caddy
upstream and `--served-model-name` all stay `qwen36-27b`, so no client,
dashboard or benchmark is touched and the roll can go one replica at a time.
`--model-path` is now the only thing identifying the weights.

```bash
HEALTH_TIMEOUT=3600 ./deploy/roll-replica.sh r1   # drains from the router first
# validate r1 (gates below), then
./deploy/roll-replica.sh r0                       # cache is warm; default timeout is fine
```

r1 first: it downloads the 51.75 GiB into the shared HF cache mount, so r0's
boot reuses it and needs no override.

**`HEALTH_TIMEOUT=3600` is required on the first roll.** The script's default
is 900 s, which fits a warm boot (~181 s) but not one that fetches weights:
r1 took 591 s in weight load alone, ~13 min drain-to-healthy. Overrunning is
safe — the replica is simply left unregistered and traffic stays on the peer —
but it aborts the roll after the container is already recreated, which reads
as a failure that isn't one. The constant was made env-overridable for this.

**Mixed-weights window.** Between the two rolls, r0 serves 3.6 and r1 serves
3.8 under one model name behind a `round_robin` router — responses alternate
between two different models. That is exactly what you want for the A/B, and
exactly what you do not want for users. Keep the window deliberate.

## Boot gates

Same shapes ⇒ the memory arithmetic must not move. From the 3.6 baseline:

| gate | expected | where |
|---|---|---|
| `max_total_num_tokens` | **171008** | boot log, `Memory pool end` |
| decode cuda-graph batch sizes | **[1, 2, 3, 4]** | boot log |
| mamba pool | **43 slots**, SSM verify cache ~2.11 GB | boot log |
| `Init Unified RadixTree with components (FULL, MAMBA)` | present | boot log |
| `mm_feature_transport` | **`cpu`** | `/get_server_info` |

A deviation means something other than the weights changed — investigate
before serving. Per the standing note on issue #29857: if the pool has
collapsed (~50 GB stranded), drop the four `--speculative-*` flags and reboot
clean before digging.

Also diff `/get_server_info` between r0 and r1 during the mixed window. Same
engine, but a new model can resolve different startup defaults — that is how
v0.5.17 flipped `mm_feature_transport` underneath us.

## What to re-measure, and what is not a gate

**Byte-identity is NOT a gate here.** Different weights. `benchmarks/byte_identity.py`
is for same-shape engine changes; it will fail by construction and that means
nothing.

Re-run against both workers directly on :8001/:8002, never through the router:

```bash
./benchmarks/run_worker.sh
```

- **`spec_accept_length` — expect it to move.** The MTP head is retrained. The
  5/6 EAGLE depth (`--speculative-num-steps 5`,
  `--speculative-num-draft-tokens 6`) was tuned against 3.6's head and won
  +10.6% single-stream there; it does not automatically carry. **Re-run the
  depth A/B** before treating the current values as optimal.
- Concurrency ladder c=1..12, and cold TTFT on text neither replica has seen.
- Remember the rig's noise floor: same-build r0-vs-r1 has measured ±1.5%.
  Anything under that is unresolvable without clock locking, which we still
  cannot do.
- `xhigh` adds thinking tokens per request. Watch time-to-last-token, not just
  tok/s — throughput can be flat while answers take materially longer.

## Measured — r1 rolled 2026-08-15 12:31–12:44

r1 is on 3.8 and re-registered; r0 is still on 3.6 at time of writing.

Boot, from a cold HF cache: weight fetch + load 591 s, total drain-to-healthy
~13 min. Every gate reproduced the 3.6 baseline exactly:

| gate | 3.6 baseline | r1 on 3.8 |
|---|---|---|
| `max_total_num_tokens` | 171008 | **171008** |
| mamba cache slots | 43 | **43** |
| intermediate SSM verify cache | 2.11 GB | **2.11 GB** |
| decode cuda-graph bs | [1, 2, 3, 4] | **[1, 2, 3, 4]** |
| `available_gpu_mem` | 8.37 GB | 8.27 GB |
| `mm_feature_transport` | cpu | **cpu** |

It loaded through the existing class on the untouched engine — the whole
premise of this upgrade, confirmed in one line:

```
Load weight end. type=Qwen3_5ForConditionalGeneration  mem usage=51.05 GB
Load weight end. type=Qwen3_5ForCausalLMMTP            mem usage=5.53 GB
```

Functional checks against r1 direct, 16/16 pass: server args as configured
(`default_chat_template_kwargs={'reasoning_effort': 'xhigh', 'preserve_thinking': False}`),
generation with `reasoning_content` correctly split by the qwen3 parser, and
tool calls parsed by `qwen3_coder`.

**The `reasoning_effort` trap is confirmed disarmed.** All of
`high`/`minimal`/`max`/`none`/`medium`/`low`/`xhigh` return **200**. Without the
server-side pin, four of those seven would be 400s.

**Prompt length is +42 tokens vs 3.6**, and that is the `xhigh` instruction,
nothing else. Verified by rendering both templates on an identical 3-message
payload: the only diff is the injected system block, there are **zero** empty
`<think>` blocks (so `preserve_thinking: false` is doing its job), and the same
config at `reasoning_effort: medium` renders byte-identically to 3.6. Live
`prompt_tokens` for that payload: r0 25, r1 67.

`spec_accept_length` read 3.0 on r0 and 4.225 on r1 — **not a benchmark**.
Both are last-value gauges over a handful of requests; 3.6's converged value
under real load is 5.025. It shows EAGLE functioning against the retrained MTP
head, and nothing about whether 5/6 is still the right depth. That A/B is
still owed.

## Rollback

Revert `--model-path` to `Qwen/Qwen3.6-27B` and drop the
`--default-chat-template-kwargs` line, then roll the affected replica. The 3.6
weights stay in the HF cache, so rollback does not re-download. The engine
never moved, so there is nothing else to undo.
