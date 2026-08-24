# Next session — `--enable-int8-mamba-checkpoint`

**Status: NOT TRIED. Parked 2026-08-24 during the v0.5.18 upgrade at the
operator's request ("I'll try it another time"). Nothing applied; both
replicas are on the plain v0.5.18 config.**

## Why this one and not the fp8 flags

SM80 has no native FP8 tensor cores, and `--speculative-draft-kv-cache-dtype`
offers only `auto / fp8_e5m2 / fp8_e4m3 / bf16` — **there is no int8 option**,
so an int8 draft KV pool does not exist as a knob. It is moot anyway: the
EAGLE draft KV pool is just **0.66 GB** (boot log: `#tokens 169408, K 0.33 GB,
V 0.33 GB`), so fp8 would free ~0.33 GB while perturbing draft numerics — i.e.
accept length, which the EAGLE 5/6 win rests on.

`--enable-int8-mamba-checkpoint` is the int8 lever that actually fits SM80 and
this hybrid GDN architecture. It is **not new in v0.5.18** — it exists in
v0.5.17 too (`enable_int8_mamba_checkpoint=False, int8_mamba_ckpt_size=None`).

## What it does

`mem_cache/mamba_checkpoint_pool.py`. The radix prefix cache stores cached
linear-attn states in a **separate int8 pool** instead of in the active bf16
mamba pool, giving ~2x cached-prefix capacity per byte.

- SSM temporal state: int8, **symmetric per-(head, k-channel)**; the scale
  reduces over `d_v`, which aligns with the per-k-channel decay `diag(alpha)`.
- Layout: `qdata [L, slots, H, d_v, d_k] int8` + `scale [L, slots, H, 1, d_k]`
  (bf16) + conv.
- Upstream's stated reason for int8 over fp8: a cached checkpoint is loaded
  **once** on a cache hit, and the temporal state is roughly uniformly
  distributed, which suits int8-per-(head,k) better than fp8's exponent.
- Default slot count: `int8_mamba_ckpt_size or (2 * mamba_size)` → **86** for
  our 43-slot active pool.

## Compatibility — verified against our config, it is allowed

Upstream rejects the combination up front rather than corrupting state:

| requirement | ours | ok |
|---|---|---|
| `--enable-hierarchical-cache` must be off | `enable_hierarchical_cache=False` | yes |
| `--radix-cache-backend` must be unset | `radix_cache_backend=None` | yes |
| built-in MambaRadixCache in use | `uses_mamba_radix_cache=True`, `mamba_radix_cache_strategy=extra_buffer` | yes |

## THE TRAP TO CHECK FIRST — it ALLOCATES, it does not reallocate

This is an **additional** HBM pool on top of the existing 3.09 GB active
`ssm_state`. "~2x capacity at fixed memory" is fixed *relative to doubling the
bf16 pool*, not free.

Rough arithmetic for 86 slots over 48 linear-attn layers (`H=48`, `d_v=d_k=128`
— confirm `H` from `cache_params.shape.temporal`, it is a guess here):

```
qdata  = 48 layers x 86 slots x 48 x 128 x 128 B   ~ 3.0 GiB
scale  = 48 x 86 x 48 x 128 x 2 B                  ~ 51 MiB
```

So on the order of **~3 GiB**. Two things make that dangerous here:

1. **`available_gpu_mem` at boot is not free VRAM.** `CONTEXT_262144.md`
   already corrects this: the boot log's 8.3x GB is sampled *before* the CUDA
   graphs (~1.55 GiB) and the FlashInfer workspaces + CUDA context (~1.2 GiB).
   True steady-state free is **~4.4 GiB**. A ~3 GiB pool fits the boot-time
   number comfortably and the real one only barely.
2. **The pool's own pre-check uses the boot-time number.**
   `maybe_init_int8_mamba_checkpoint_pool` calls `torch.cuda.mem_get_info`
   *at init*, before graph capture, and raises only if
   `est >= free`. So it can pass its own check and still leave too little for
   capture. Do not treat "it booted" as "it fits".

It does log the exact footprint before allocating, which is the cheap way in:

```
int8 mamba checkpoint pool: 86 slots, X.XXGB (qdata … + scale … + conv …);
active mamba pool 43 slots; free HBM Y.YYGB
```

**Read that line on a dry run before trusting any arithmetic above.** Start
with an explicit small `--int8-mamba-ckpt-size` rather than the 2x default.

Also remember the ReplaySSM precedent (`RESULTS.md`): a mamba-side change that
looked memory-neutral ended up taking 2 GB *out of KV* to buy more slots and
dropped `max_total_num_tokens` to 137,600, under the served context length.
Our margin is now thinner than it was then — **169,408 vs `--context-length`
169,000 is only 408 tokens.**

## It is LOSSY — gate it on output, not just on boot

int8-quantized SSM states change results on a **cache hit** but not on a cold
miss. So the correct gate is a *warm* comparison, which is the opposite of the
usual one.

1. Roll r1 only; hold r0 as the bf16 control.
2. Boot gates: `max_total_num_tokens` must stay **169408**,
   `max_mamba_cache_size` 43, decode bs `[1,2,3,4]`. Read the new pool line.
3. Byte-identity, **both replicas flushed** (see `docs/OPERATIONS.md` — an
   unflushed replica scores ~1/8 against itself and tells you nothing).
   Expect cold-miss output to match; that is necessary, not sufficient.
4. Then the real gate: send the same prompt **twice** and compare the second
   (cache-hit) response across r1 and r0. That is where the quantization
   shows up.
5. Accept length must not move — check `spec_accept_length` converged.

## What would count as a win

Cached-prefix capacity, so: prefix-cache hit rate and TTFT on repeated
prefixes, not throughput. `--enable-cache-report` is already on, and the
`mamba usage` field in the prefill log lines is the direct read. Baselines to
compare against live in `benchmarks/results/worker_v0517_pre518*.json` and
`worker_v0518_r1.json`.

If hit rate is already high, this buys nothing and should be reverted — the
flag only pays when prefixes are being evicted.
