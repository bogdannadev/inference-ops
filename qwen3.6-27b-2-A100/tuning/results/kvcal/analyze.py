import math, sys, torch
d = torch.load(sys.argv[1])
lo, hi, nb = d.pop("_bins")
w = (hi - lo) / nb
centers = torch.tensor([2 ** (lo + (i + 0.5) * w) for i in range(nb)], dtype=torch.float64)

def fp8_err_energy(h, scale):
    """Expected e4m3 round-to-nearest error energy / signal energy for values x/scale."""
    x = centers / scale
    e = torch.floor(torch.log2(x.clamp_min(2 ** -40)))
    ulp = torch.where(x >= 2 ** -6, 2 ** (e - 3), torch.full_like(x, 2 ** -9))
    var = ulp ** 2 / 12
    var = torch.where(x < 2 ** -10, x ** 2, var)          # flushed to zero
    clipped = x > 448
    var = torch.where(clipped, (x - 448) ** 2, var)
    return float((h * var).sum() / (h * x ** 2).sum())

def frac(h, thr_log2, below=True):
    idx = int((thr_log2 - lo) / w)
    return float(h[:idx].sum() / h.sum()) if below else float(h[idx:].sum() / h.sum())

def pct(h, q):
    c = torch.cumsum(h, 0) / h.sum()
    i = int(torch.searchsorted(c, torch.tensor(q, dtype=c.dtype)))
    return float(centers[min(i, nb - 1)])

print(f"{'layer':>5} {'t':>1} {'amax':>8} {'p50':>8} {'p99.9':>8} {'<2^-6':>7} {'<2^-10':>7} {'>448':>6}  {'err@1.0':>9} {'err@amax/448':>12} {'gain':>6}")
rows = []
for lid in sorted(d):
    st = d[lid]
    for t in ("k", "v"):
        h = st[t]; amax = float(st[t + "max"])
        s = amax / 448
        e1, ec = fp8_err_energy(h, 1.0), fp8_err_energy(h, s)
        rows.append((t, e1, ec))
        print(f"{lid:>5} {t:>1} {amax:8.3f} {pct(h,.5):8.4f} {pct(h,.999):8.3f} {frac(h,-6):7.2%} {frac(h,-10):7.2%} {frac(h,math.log2(448),False):6.2%}"
              f"  {10*math.log10(e1):8.1f}dB {10*math.log10(ec):11.1f}dB {10*math.log10(e1/ec):5.1f}dB")
for t in ("k", "v"):
    e1 = sum(r[1] for r in rows if r[0] == t) / 16; ec = sum(r[2] for r in rows if r[0] == t) / 16
    print(f"mean {t}: err energy at scale 1.0 = {10*math.log10(e1):.1f} dB, calibrated = {10*math.log10(ec):.1f} dB, gain {10*math.log10(e1/ec):.1f} dB")
