# PTX on sm_80 — what this toolchain actually accepts

Probed 2026-08-24 by compiling minimal kernels with the `ptxas` inside the
pinned production image, `lmsysorg/sglang@sha256:16aba89…46155`
(CUDA 13.0, libcublas 13.0.2.14). **These are compile results, not quotations.**
A summarised reading of the PTX ISA document disagreed with the assembler on
two points below, and the assembler wins.

Method: emit one instruction into a `.target sm_80` kernel, run
`ptxas -arch=sm_80`, record accept/reject and the exact diagnostic.

---

## Accepted

```ptx
ld.global.ca.b32   %r1, [%rd2];     // cache at all levels
ld.global.cg.b32   %r1, [%rd2];     // cache global — bypass L1
ld.global.cs.b32   %r1, [%rd2];     // cache streaming — evict-first in L2
ld.global.nc.b32   %r1, [%rd2];     // non-coherent / read-only path

createpolicy.fractional.L2::evict_last.b64 %pol, 0.75;
createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %pol, 0.75;
createpolicy.range.L2::evict_last.L2::evict_first.b64 %pol, [%rd2], 4096, 8192;

ld.global.L2::cache_hint.b32          %r1, [%rd2], %pol;
ld.global.nc.L2::cache_hint.v4.b32 {%r1,%r2,%r3,%r4}, [%rd2], %pol;

cp.async.ca.shared.global [sm], [%rd2], 16;
cp.async.cg.shared.global [sm], [%rd2], 16;
cp.async.ca.shared.global [sm], [%rd2], 4;
cp.async.commit_group;
cp.async.wait_group 2;
```

## Rejected on sm_80 — and this is the load-bearing result

```ptx
ld.global.L2::evict_first.b32          %r1, [%rd2];
ld.global.L2::evict_first.v2.b32   {%r1,%r2}, [%rd2];
ld.global.L2::evict_first.v4.b32   {%r1,%r2,%r3,%r4}, [%rd2];
ld.global.L2::evict_last.v4.b32    {%r1,%r2,%r3,%r4}, [%rd2];
ld.global.nc.L2::evict_first.v4.b32 {…}, [%rd2];
st.global.L2::evict_first.v4.b32   [%rd2], {…};
```

all fail with:

```
error : Instruction 'ld' requires '.v8.b32/.v4.b64' type with '.L2::evict_first' modifier.
```

`.v8.b32` / `.v4.b64` are 32-byte vector accesses, which are not an sm_80 form.
**So the per-instruction eviction-priority suffix is effectively unavailable on
A100 with this toolchain, at every width we can actually emit.**

And:

```ptx
cp.async.cg.shared.global.L2::evict_first [sm], [%rd2], 16;
error : Illegal modifier '.level::eviction_priority' for instruction 'cp.async'
```

Also noted: `ptxas -arch=sm_70` is rejected outright — `Value 'sm_70' is not
defined for option 'gpu-name'`. CUDA 13 has dropped Volta, so any claim that a
feature is available "sm_70 and up" cannot be tested here regardless.

---

## Consequence for the plan

Two routes to L2 eviction control existed on paper. Only one survives:

| route | status on sm_80 / CUDA 13 | reaches cuBLAS? |
|---|---|---|
| `.L2::evict_*` suffix on `ld`/`st` | **unavailable** | — |
| eviction modifier on `cp.async` | **unavailable** | — |
| `createpolicy` + `.L2::cache_hint` | available | no — needs kernel source |
| runtime address-range window (`cudaAccessPolicyWindow`) | available | **yes** |

The runtime window is the same mechanism as `createpolicy.range` and, per the
A100 whitepaper, *"the memory operations themselves require no annotation"* —
so it applies to every kernel on the stream including closed-source cuBLAS.
That is why E2 in the README is specified at the stream level rather than as a
kernel edit.

Corollary for Triton: `tl.load(..., eviction_policy=…)` cannot be lowering to
the `.L2::evict_*` suffix on this target, since that form does not assemble.
Whether it lowers to `createpolicy`/`cache_hint`, to `.cs`, or is silently
dropped is **unverified** — check the generated PTX before relying on it.

---

---

## CUDA 13 runtime enum numbering — two landmines found in E2

Both of these produce a *fake null result* rather than an error, which is the
worst failure mode for an experiment. Verify enum values empirically.

**`cudaDevAttrL2CacheSize` is 38, not 78.** The widely-quoted 78 returns `1` on
this toolchain. Found by scanning attributes 1..150 for the known 40 MiB value;
exactly one matched:

```
attr  38 = 41943040   = 40.00 MiB   <- cudaDevAttrL2CacheSize
attr  78 = 1                        <- something boolean
attr 108 = 26214400   = 25.00 MiB   <- cudaDevAttrMaxPersistingL2CacheSize
attr 109 = 134213632  = 128.00 MiB  <- cudaDevAttrMaxAccessPolicyWindowSize
```

108 and 109 are correct: 128.00 MiB is exactly A100's documented
`accessPolicyMaxWindowSize`, which is what confirms the pair.

**`cudaLimitPersistingL2CacheSize` is `0x06`, not `0x05`.** `0x05` is
`cudaLimitMaxL2FetchGranularity`. Setting `0x05` to `0` **succeeds silently** —
zero is a valid fetch granularity — so a script that only ever sets the limit to
0 and then to a real size will appear to work until the real size is rejected
with `invalid argument`. If the sizes happened to be small enough to pass, the
whole experiment would have run with the persisting cache never enabled and
reported a clean null.

Guard against both: read every limit back with `cudaDeviceGetLimit`, read every
stream attribute back with `cudaStreamGetAttribute`, and sanity-check the
attribute enum against a value you already know.

---

## Reproduce

```bash
IMG=lmsysorg/sglang@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155
docker run --rm --entrypoint bash $IMG -c '
cat > /tmp/x.ptx <<EOF
.version 7.8
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 p) {
  .reg .b64 %rd<8>; .reg .b32 %r<16>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  ld.global.L2::evict_first.b32 %r1, [%rd2];
  ret;
}
EOF
ptxas -arch=sm_80 /tmp/x.ptx -o /tmp/x.cubin'
```

No GPU is required — `ptxas` is a host-side assembler.
