# Trial 2026-10-11: FP8 weights (Marlin W8A16) × KV dtype

All four arms ran on drained r1, the same day and the same image (v0.5.21,
`e00930c`) with patch 3 live. r0 served production alone from 08:05 to ~11:30
UTC. The plan and the literature are in `FP8_WEIGHTS_LONG_CONTEXT.md`.

| arm | weights | KV | file |
|---|---|---|---|
| T0 | bf16 `Qwen/Qwen3.8-27B` | fp8_e4m3 @1.0 | production flags |
| T1 | bf16 | bf16 | reference |
| T2 | FP8 `Qwen/Qwen3.8-27B-FP8` @ `017b9c7` | bf16 | candidate |
| T3 | FP8 | fp8_e4m3 @1.0 | fourth cell |

Every arm uses production flags, with two exceptions:
- `--hicache-ratio 2` instead of 8 caps host RAM, because T2 at ratio 8 would
  pin ~310 GB.
- Each arm gets its own L3 dir (`make_arm.py`).

The draft KV stays fp8 in every arm. Arms are made by `tuning/trial/make_arm.py`
(`tuning/trial/T*.yml`) and run by `tuning/trial/run_arm.sh`. The kvq_eval corpus
is the pinned image's own `/sgl-workspace/sglang` (`docs/` + `python/sglang/srt`),
so its numbers do not compare with the September (v0.5.20 tree) runs.

SGLang logs `Weight-only FP8 compression will be used leveraging the Marlin
kernel` on SM80. A boot takes ~9.5 min, against ~4 min for bf16; most of the
extra time is CUDA-graph capture.

## Results (`tuning/results/*_r1_T{0,1,2,3}_*.json`, `boot_r1_T*.log`)

| gate | T0 prod | T1 bf16/bf16 | T2 FP8 W/bf16 KV | T3 FP8 W/fp8 KV |
|---|---|---|---|---|
| KV pool (tokens) | 282,496 | 160,320 | **470,336** | **828,736** |
| smoke | 7/7 | 7/7 | 7/7 | 7/7 |
| lookup to 128K | 18/18 | 18/18 | 18/18 | 18/18 |
| kv/order/arith, determinism | 30/30, ✓ | 30/30, ✓ | 30/30, ✓ | 30/30, ✓ |
| `seq` @ 8K max_tokens (greedy, prod) | 3/5, 4/5 | 3/5, 0/5 | 1/5, 0/5 | 2/5, 0/5 |
| **`seq` @ 16K** (greedy, prod) | **5/5, 5/5** | **5/5, 5/5** | **5/5, 5/5** | **5/5, 5/5** |
| ΔNLL vs T1, 16 short docs | −0.005 (SE 0.003) | — | **+0.007** (SE 0.003) | +0.005 (SE 0.003) |
| ΔNLL vs T1, 8 long docs 8K–128K | +0.006 (SE 0.007) | — | −0.009 (SE 0.014) | −0.025 (SE 0.018) |
| decode tok/s short c1 / c4 | 74.3 / 63.8 | | **104.0 / 95.1** | 106.6 / 93.0 |
| decode tok/s long c1 / c4 | 65.1 / 51.8 | | **105.1 / 76.3** | 108.3 / 70.2 |
| aggregate tok/s short c4 / long c4 | 238.5 / 190.2 | | 333.3 / 260.7 | 339.9 / 242.8 |
| accept length (range over phases) | 3.11–3.23 | | 2.95–3.29 | 3.12–3.55 |
| cold prefill 8K / 32K / 64K / 128K (s) | 2.2 / 10.2 / 25.0 / 68.0 | 2.1 / 9.5 / 22.2 / 56.8 | 3.0 / 13.1 / 29.4 / 71.6 | 3.0 / 13.7 / 32.1 / 82.4 |
| mem_stress 160K + 3×60K, ×2 | clean, big TTFT 177 s | (pool too small, not run) | clean, 185 s | clean, 208 s |

Decode is measured against same-day T0.

## Reading

- **The `seq` gate at 8K never measured precision.** With thinking on, the 300-term
  task takes 6.2–13.8K completion tokens in every arm, full bf16 included. At
  `--max-tokens 8000` most rounds end `truncated`, and which ones do is chance.
  At 16K every arm scores 10/10 exact. This retracts the "fp8 KV hurts long
  exact outputs" cost recorded in `FP8KV_L3.md` (greedy 1/5 vs bf16 3/3). There
  the bf16 side had 3 rounds and the fp8 side had 5, all at the 8K cap.
- **fp8 KV at scale 1.0 costs nothing measurable in accuracy.** T0 vs T1 is
  within one to two SE on both doc sets. This agrees with the 2026-10-09 range
  capture. Its real cost is prefill speed: +20% at 128K (T0 vs T1), and stacked
  on Marlin +15% (T3 vs T2).
- **FP8 weights cost ~+0.007 nats/token on short high-entropy text** (T2 vs T1,
  20/12 docs worse), under the +0.01 bar. Nothing grows with position: the long
  set is noise in both directions. Retrieval, code and arithmetic are untouched.
- **Decode is up 40–61%** at the same draft acceptance. This node is
  bandwidth-bound, and the target's weight bytes drop from ~54 GB to ~31 GB.
- **Marlin's prefill cost** is +38–42% at 8K, shrinking to +5% at 128K
  (T2 vs T0), where attention dominates. Agent turns are mostly cache hits plus
  a short cold tail, so this cost lands on the first turn of a session.

## Recommendation

Serve **T2 (FP8 weights + bf16 KV)**.
- It has T3's decode at c=1 and a better one at c=4.
- Its long-prompt prefill is better than T3's and about equal to today's.
- It holds 1.66× today's tokens in bf16.

T3's 829K pool is more than `--max-running-requests 4` can use (4 × 169K = 676K),
and it pays fp8 dequant on every prefill.

Before rolling it to production (all four done the same day; see the follow-up
below and the 2026-10-11 block in `docker-compose.yml`):
1. **Pin the host tier explicitly.** At `--hicache-ratio 8`, T2 would pin
   ~247 GB KV + 38 GB draft + 27 GB GDN per replica, so two replicas exceed the
   568 GB available. Use `--hicache-size` (GB), sized so both replicas fit,
   with headroom left for the shared tmpfs L3.
2. **Use a new L3 directory.** Both the weights and the KV dtype change what a
   cached page means (`hicache-l3-shared-file`). Roll r1 then r0 onto a fresh
   `bf16` dir; never mix arms in one dir.
3. **Retune GDN slots.** `--max-mamba-cache-size 43` was sized for a full GPU;
   with ~24 GB freed, the GDN slot count, which today limits agent cache hits
   (`gdn-state-tier-binds-cache`), can rise. This is a separate drained
   measurement.
4. HF download: the checkpoint is already in the shared cache at `017b9c7`; pin
   `--revision` in compose.

## Follow-up the same day: GDN slots at a fixed host budget (drained r1)

Three arms, all T2 (FP8 weights + bf16 KV) with `--hicache-size 180` in place
of `--hicache-ratio`, differing only in `--max-mamba-cache-size`
(`tuning/trial/G{43,86,128}_fp8w_h180.yml`). `cache_replay` used 41K agent
contexts plus 1K per turn, interleaved with 3K one-offs, with L3 cleared in each
arm's private dir.

| slots | device KV | host KV / GDN / draft (GB) | 4 agents near / far | 8 agents near / far | mem_stress |
|---|---|---|---|---|---|
| 43 | 470,336 | 148.3 / 31.8 / 23.2 | 0.66 / 0.98 | 0.66 / **0.48** | clean |
| 86 | 425,856 | 132.7 / 47.4 / 20.7 | **0.98** / 0.98 | 0.66 / 0.98 | clean |
| **128** | 382,336 | 117.7 / 62.3 / 18.4 | **0.98 / 0.98** | **0.77 / 0.98** | clean |

Production (bf16 weights, 43 slots, ratio 8) measured 0.66 / 0.98 with
4 agents on 2026-09-19. Smoke was 7/7 in every arm.

- In v0.5.21 `--hicache-size` is **one** budget, split between target KV and GDN
  state in proportion to their device bytes (`hybrid_pool_assembler.py`,
  `_split_hicache_size`). The DFlash2 draft host pool is sized separately, to
  the target host pool's token count. So more device slots also buys more host
  GDN states out of the same budget.
- The near-phase misses that production always had (0.66) were a device-slot
  limit. At 86 slots and up, 4 interleaved agents hit fully.
- At 8 agents, 43 slots loses half the far returns. 128 is the first count where
  near also improves.
- Device KV at 128 slots (382K) is still 2.9× the 7-day p99 in use.

**Applied: 128 slots, `--hicache-size 180`**, ~198 GB pinned per replica.

## B1 (same window, r1 stopped)

FA2 (torch SDPA flash) beats the production FlashInfer hd256 prefill kernel by
15–19% at 4K–108K prefixes. Page layout makes no difference, and Triton is 3×
slower. Details and next steps are in `tuning/lowlevel/P1_PROFILE.md`, Stage B1.
