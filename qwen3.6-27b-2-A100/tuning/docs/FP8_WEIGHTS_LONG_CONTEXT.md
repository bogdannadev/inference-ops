# FP8 weights vs FP8 KV at long context: literature review, 2026-10-09

**Status:** literature only. Nothing has been measured on this node yet. Every
claim below cites a paper by arXiv ID, and each one was read in full text, not
just the abstract.

## The question

We are considering serving `Qwen/Qwen3.8-27B-FP8` as **W8A16 via Marlin** on
A100: block-FP8 weights at 128×128, with activations kept in bf16. Today we
serve bf16 weights with **fp8 e4m3 KV**. That KV setting already costs us on
long exact outputs: the `seq` task scores greedy 1/5, against 3/3 on bf16
(`TRIAL_FP8KV_MAMBA.md`, `FP8KV_L3.md`).

This review asks four things:

1. Does weight-quantization error grow with context length?
2. Which hurts long context more: FP8 weights or FP8 KV?
3. How sensitive is a GDN hybrid's recurrent half?
4. Which layers are the sensitive ones?

## Bottom line

- **8-bit weights do not degrade long context in any study found.** Two kinds
  of evidence agree:
  - Studies of pure transformers at 64K–128K (2505.20276, 2411.02355).
  - A study of **our exact model**, 2609.04098, which found that even 4-bit
    W4A4 on every layer leaves a gap that *shrinks* with position.
- **KV quantization is the long-context risk, not weight quantization.**
  - At the same bit-width, KV quantization hurts long context more than
    weight-only quantization (2402.18158).
  - On our model, FP8 KV with scale 1.0 has a cost that *rises* with position
    (2609.04098). We run exactly that configuration today.
- **W8A16 is the gentlest 8-bit scheme measured.** It was the only scheme
  with negligible answer flips (2407.09141), and the reasoning study found
  8-bit lossless (2504.04823).
- **The recurrent state must stay high precision.** That is a different thing
  from the weights: quantizing the GDN *state* itself does hurt (2608.27513,
  2609.38169). We keep it in bf16 (`--mamba-ssm-dtype bfloat16`).
- **No paper covers three things we need to know:**
  - our model beyond 64K;
  - long *exact* outputs of thousands of tokens;
  - DFlash2 accept length on a quantized target.

  We have to measure these ourselves. See the gates below.

## Papers

| ID | Paper | Venue | Model(s) | Relevance |
|---|---|---|---|---|
| 2609.04098 | Kozyrev & Maiboroda, *Why Gated DeltaNet Survives 4-Bit Quantization* | preprint, Sep 2026 | **Qwen3.8-27B** | the only study of our exact model |
| 2505.20276 | Mekala et al., *Does quantization affect models' performance on long-context tasks?* | EMNLP 2025 | Llama-3.1 8B/70B, Qwen-2.5 7–72B | FP8/INT8/INT4 at 64K–128K inputs |
| 2402.18158 | Li et al., *Evaluating Quantized Large Language Models* | ICML 2024 | 11 families | weight vs KV at long context |
| 2411.02355 | Kurtic et al., *"Give Me BF16 or Give Me Death"?* | ACL 2025 | Llama-3.1 8B–405B | FP8 W8A8, including RULER |
| 2407.09141 | Dutta et al., *Accuracy is Not All You Need* | NeurIPS 2024 | several | flips and KL as distance metrics; W8A16 |
| 2504.04823 | Liu et al., *Quantization Hurts Reasoning?* | COLM 2025 | R1-distill, QwQ, Qwen3-8B | long reasoning outputs under quantization |
| 2401.18079 | Hooper et al., *KVQuant* | NeurIPS 2024 | LLaMA, Mistral | why keys are hard to quantize |
| 2608.27513 | Zhang et al., *DAMP* | preprint, Aug 2026 | Qwen3.6-35B, Kimi-Linear | GDN/KDA **state** quantization |
| 2609.38169 | Yao et al., *STEPQuant* | tech report, Sep 2026 | **Qwen3.8-27B**, Kimi-Linear | delta-rule **state** quantization |
| 2606.20474 | Chakrabarti et al., *UltraQuant* | EMNLP 2026 Industry | MiniMax-M2.5, Qwen3-235B | FP8/4-bit KV on Claude Code agent traces |

**Reproducibility:**
- 2609.04098 releases its checkpoint. Its harness is described but its code
  is not linked.
- 2505.20276, 2402.18158, 2504.04823 and 2609.38169 release code.
- 2609.04098 is a one-month-old preprint from a company that publishes its own
  quantized checkpoint, using one model, one format and one harness. Treat it
  as strong but not yet replicated.

## Q1: does weight-quantization error grow with context?

**At 8 bits, no.**

- 2505.20276 tested inputs of 64K and 128K tokens. Its FP8 was W8A8 dynamic:
  per-channel weights and per-token activations, run on H100.
  - FP8 and GPTQ-int8 were not statistically different from BF16 on
    Ruler, OneRuler, NoCha or FActScore.
  - The paper gives the average drops as FP8 0.2% / GPTQ-int8 0.8% in §1 but
    0.8% / 0.2% in §4.1. It contradicts itself on which is which; either way,
    both are under 1%.
  - One exception: FP8 lost significantly on CS4, a constraint-satisfaction
    task.
- 2411.02355 found FP8 W8A8 "essentially lossless", and quantized models
  recovered ≥98% on RULER.

**At 4 bits, it depends on the architecture.**

- In pure transformers the loss grows with input length. 2505.20276 reports
  up to −23% on average at 128K and −59% in the worst case (BNB-nf4,
  non-English). 2402.18158 sees the same at W3.
- In the GDN hybrid it shrinks. 2609.04098 split per-token NLL by position
  across a 32K window. The weight cost was largest at the start and turned
  *negative* in the last 2K tokens. Three mechanisms, measured on captured
  activations:
  1. **Block scaling contains outliers.** A16 activation blocks confine each
     outlier to its 15 neighbours. On our path activations stay bf16, so this
     doesn't even apply.
  2. **The gates squash errors.** The softplus/exponential and sigmoid gate
     parameterizations turn an ~11% GEMM error into a 2–3.6% error in the
     state.
  3. **The delta rule overwrites the state along each new key.** Injected
     noise sits on a flat plateau over 32K tokens (12.6%). A one-off 1%
     impulse to the state is erased within 80–1,382 steps, far faster than
     the decay gates alone would allow (44K–62K tokens).
- Its 5-task average was 85.10 against BF16's 85.62, inside seed noise. RULER
  NIAH single and multikey scored 100 at 32K and 64K for every recipe.
- Evidence stops at 64K. The authors call 128K+ "extrapolation".

## Q2: FP8 KV vs FP8 weights, which hurts long context more?

**KV, in every source that tests both.**

- 2402.18158 §7: "most LLMs are more sensitive to KV Cache Quantization than
  Weight-only and Weight-Activation Quantization" at ≥4K. KV8 was lossless for
  most families, but not for LongChat. Note the age of that study: contexts
  ≤16K, integer formats, and 2023-era models.
- 2609.04098 §7 and Table 4, on our model:
  - FP8 KV with scale 1.0 moves no task score, and RULER stays at 100 to 64K.
  - But its NLL cost *rises with position* and is larger on top of quantized
    weights: "the signature of an attention-path effect".
  - **Calibrated per-tensor FP8 KV scales recover 83% of that cost**, with
    throughput unchanged within 0.4%. That is 32 scale tensors across the 16
    attention layers.
- 2401.18079 explains the mechanism. Keys have outlier *channels*, and RoPE
  makes their magnitudes inconsistent across positions. A single per-tensor
  scale handles that badly, and the damage accumulates with the number of
  keys attended over.
- 2606.20474 used hardware FP8 KV as the deployment anchor on replayed
  Claude Code traces. Task quality held there. It concerns agent workloads but
  doesn't test exact long outputs.

**This applies to us now.** We run `--kv-cache-dtype fp8_e4m3` without
`--quantization-param-path`, so every scale is 1.0. SGLang's own help text
says that "may cause accuracy issues". Our measured pattern matches the
paper's: retrieval was fine (lookup 18/18 to 128K) while long exact output
failed (`seq` 1/5).

## Q3: is the GDN recurrent half fragile?

**The GDN weights are robust. The GDN state is not.**

- 2609.04098 found the weights robust (Q1). Its §5.2 table inverts the usual
  precision map: the gate projections `in_proj_a` and `in_proj_b`, which every
  public recipe keeps in bf16, are the *least* sensitive. They change the
  layer output by 2.1% and 2.6%. The plain GEMMs carry the error: out 12.7%,
  qkv 10.4%, z 9.9%.
- The **state itself** is a different matter:
  - 2608.27513 found that uniform INT8 or FP8 on the state "degrade[s] complex
    reasoning", and INT4/NVFP4 collapses it. Its outliers sit in specific key
    channels, and slow-decay heads keep the error.
  - 2609.38169, also on Qwen3.8-27B, found that state errors persist in
    long-lived memory. It needs mixed precision to stay near FP32.
- **Why the two results don't conflict:** a weight error enters each write
  once, and later writes overwrite it. A state error is re-applied to the
  stored memory at every step.
- Qwen's FP8 checkpoint keeps `A_log`, `dt_bias`, `in_proj_a`/`in_proj_b`,
  `conv1d` and the norms in bf16, and quantizes `in_proj_qkvz` and `out_proj`.
  By 2609.04098's account that is conservative in the right direction.

## Q4: which layers are sensitive?

- In the GDN block, per 2609.04098: `out_proj` > `qkv` > `z` ≫ the a/b gates.
- Attention: no paper isolates q/k/v/o sensitivity on this model at 8 bits.
  The keys' outlier channels (2401.18079) matter for the **KV cache**, not for
  `k_proj` weights at 8 bits.
- Mixed precision, as a fallback if needed: keep the 16 attention layers'
  q/k/v in bf16, about 1.3 GB. No source shows it's necessary at 8 bits.

## Long outputs and spec decoding: what the literature doesn't cover

- **Long exact outputs** have no coverage:
  - 2505.20276 capped outputs at 650 tokens.
  - 2504.04823 and 2609.04098 found that quantized models don't "think
    longer", and 2609.04098 reports long generations matching BF16 token for
    token. But nobody tests *exact* reproduction over ~7K tokens, which is our
    `seq` gate.
- **Flips:** 2407.09141 found GPTQ **W8A16** the only scheme with negligible
  flips. Every W8A8, W4 and pruning scheme flipped up to 13.6% of answers
  while accuracy stayed the same. Flips happen where the top-1 margin is
  small, which is exactly our "already-unstable code tokens" observation.
  Gate on KL divergence and flips, not on accuracy.
- **DFlash2 accept length on a quantized target:** no study. 2609.04098 serves
  without speculative decoding.
- **Measurement trap:** 2609.04098 §6 found that vLLM's multimodal-composite
  serving path scores long context differently from the text-only path. We
  serve the composite, because images are live. Compare variants only within
  one serving path.

## What this changes in the plan

> **Update 2026-10-11 (measured, `TRIAL_2026-10-11_FP8_WEIGHTS.md`):** the `seq`
> weakness does not exist. It was the 8K max_tokens cap; at 16K, fp8 KV, bf16 KV and
> both FP8-weight arms all score 10/10. FP8 weights cost +0.007 nats/token on short
> docs with no position trend; decode is +40–61%. Recommendation: FP8 weights + bf16 KV.

> **Update 2026-10-09 (measured, `TRIAL_2026-10-09_PATCH3_KVRANGE.md`):** our K/V
> values sit well inside e4m3's range at scale 1.0 (no clipping, 0.3–3% subnormal);
> a calibrated scale changes fp8 error energy by 0.0 dB on average. Item 1 below is
> therefore moot for this deployment; the remedy for long exact outputs is bf16 KV.

1. **Biggest cheap win, independent of FP8 weights: calibrated FP8 KV
   scales.** We serve fp8 KV at scale 1.0 today. Generate per-layer k/v scales
   (llm-compressor `kv_cache_scheme`, or a calibration pass of our own) and
   pass them with `--quantization-param-path`. Then re-run the `seq` gate.
   - Still unverified: whether SGLang's FlashInfer fp8 path on SM80 applies
     per-layer scales for this model. Read the source before building anything.
2. **FP8 weights (W8A16) are low risk for long context.** The remaining
   unknowns are ours to measure, not open research questions.
3. **Never quantize the GDN state** (DAMP, STEPQuant). Keep
   `--mamba-ssm-dtype bfloat16`, including the HiCache host copies.

## KV-scale support in SGLang v0.5.21 (source-verified 2026-10-09)

Read in the pinned image (`e00930c`), whose files are byte-identical to
upstream. Nothing here has been run.

**The working path is scales embedded in the checkpoint, attached by a
quantization method:**

1. A quantization config returns a KV method for every `RadixAttention`:
   - `Fp8Config` → `Fp8KVCacheMethod` (`fp8.py:463`);
   - also ModelOpt and compressed-tensors.

   It registers `k_scale`/`v_scale` parameters, initialized to −1
   (`kv_cache.py:39`).
2. `Qwen3_5ForConditionalGeneration.load_weights` remaps checkpoint tensors
   with `QWEN3_5_KV_SCALE_MAPPER` (`qwen3_5.py:1575`):
   `…self_attn.k_scale` and `…self_attn.k_proj.k_scale` become
   `…attn.k_scale`. Then it loads them.
3. `process_weights_after_loading` sets the tensor *and* `k_scale_float`. A −1
   that was never overwritten becomes 1.0. Scales must be per-tensor (one per
   layer).
4. **Write:** `set_kv_buffer` divides K and V by the scale before the fp8 cast
   (`memory_pool.py:2621`).
5. **Read:** FlashInfer FA2 (our SM80 path) multiplies `sm_scale` by `k_scale`
   and the output by `v_scale` in paged prefill, ragged prefill and decode
   (`flashinfer/prefill.py:2916, 3136`; `decode.py:661, 742`). Both
   operations are exact.

**What applies to our deployment:**

| setup | can we load scales? |
|---|---|
| **today**: bf16 checkpoint, no quant config | **No.** `RadixAttention.k_scale` stays `None`, so embedded tensors are dropped with a "Parameter … not found" warning. **`--quantization-param-path` raises `RuntimeError` at boot**, because `Qwen3_5` has no `load_kv_cache_scales` (`load_model_utils.py:119`). Adding that flag would turn a roll into a crash loop. |
| bf16 + engine patch | Possible. A sitecustomize patch sets `attn.k_scale` *and* `k_scale_float`/`v_scale_float` on the 16 attention layers. Both must be set: the write reads one, the read reads the other. |
| `Qwen/Qwen3.8-27B-FP8` | Supported. `Fp8Config` attaches the parameters. **But the checkpoint ships no `k_scale` tensors** (0 in `model.safetensors.index.json`), so we would add one extra safetensors shard with 32 scalars plus index entries. |

**Do not copy the JSON path.** `kv_cache_scales_loader`, as used by llama and
qwen2, sets `attn.k_scale` but never `k_scale_float`. The only assignment of
`k_scale_float` is in `kv_cache.py:82`. On a model without a quant config, that
path would divide on write and not multiply on read. That is a scale mismatch
read off the source; it has not been run.

**Two physics caveats.**
- e4m3 is floating point. A per-tensor scale does not add mantissa bits; it
  only moves the range away from clipping at ±448 and the subnormals below
  2⁻⁶.
  - K passes through QK-norm (`TRIAL_FP8KV_MAMBA.md`), so scale 1.0 is
    probably already well placed for K.
  - V has never been measured.
  - The 83% recovery in 2609.04098 was measured on vLLM with W4A4 K/V
    projections.
- **First measure** the per-layer |K| and |V| distribution: amax, and the
  share of values under 2⁻⁶. That one capture pass also produces the scales.

**HiCache L3 trap.** Cached fp8 bytes are only valid under the scales they
were written with. The L3 directory is shared by r0 and r1. A scale change
therefore needs a **new L3 directory**, and a rolling change would let one
replica read the other's old-scale entries. Same rule as
`hicache-l3-shared-file`: per KV dtype, per SGLang generation, and now per
scale set.

## Gates for the drained-r1 trial (2×2: weights {bf16, fp8} × KV {fp8@1.0, fp8 calibrated}, plus a bf16-KV control)

| gate | why (source) | pass |
|---|---|---|
| per-position mean KL vs bf16, 2K bins, at 8K/32K/64K/128K | separates weight cost from KV cost (2609.04098 §5.4); KL tracks flips (2407.09141) | weight-cost KL flat or falling with position |
| lookup / RULER-style NIAH to 128K | evidence stops at 64K (2609.04098 §9) | = bf16 (18/18) |
| `seq` exact ~7K tokens, greedy | not covered by any paper | ≥ current production (1/5); target ≥ bf16-KV control |
| flips on spec_eval code prompts | flips concentrate at low margin (2407.09141) | no worse than the fp8-KV config already in production |
| DFlash2 accept length | not covered by any paper | within run noise of bf16 |
| decode / prefill tok/s, mem_stress | Marlin at M=32 and M=4096 is unmeasured | decode up; any prefill regression stated |
