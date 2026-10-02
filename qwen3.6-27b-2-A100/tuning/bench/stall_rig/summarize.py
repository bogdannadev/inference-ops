#!/usr/bin/env python3
"""Summarise run_drill.sh output: what did clients see while fw1 was broken?

  python3 summarize.py <out_dir> [inject_at] [recover_at]
"""
import collections
import glob
import json
import os
import sys

out = sys.argv[1]
INJ = float(sys.argv[2]) if len(sys.argv) > 2 else 120
REC = float(sys.argv[3]) if len(sys.argv) > 3 else 480

print(f"{'drill':<18}{'bad':>5}{'500':>5}{'hung':>6}{'stopped->fw1 after':>20}"
      f"{'worst wait':>12}{'sessions hit':>14}{'fw1 back after':>16}")
for f in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
    rs = [json.loads(l) for l in open(f)]
    win = [r for r in rs if INJ <= r["t"] < REC]
    # bad = a client request that did not get a normal reply in reasonable time
    bad = [r for r in win if r["outcome"] != "ok" or (r["ttfb"] or 0) > 30]
    e500 = sum(1 for r in bad if str(r["outcome"]).startswith("HTTP 5"))
    hung = sum(1 for r in bad if r["outcome"] in ("still_open_at_end", "timeout", "TimeoutError")
               or (r["outcome"] == "ok" and (r["ttfb"] or 0) > 30))
    last_bad = max((r["t"] for r in bad), default=None)
    worst = max((r["total"] or 0 for r in bad), default=0)
    sess = len({r["sid"] for r in bad})
    back = min((r["t"] for r in rs if r["t"] >= REC and r["worker"] == "fw1"), default=None)
    name = os.path.basename(f)[:-6]
    print(f"{name:<18}{len(bad):>5}{e500:>5}{hung:>6}"
          f"{(f'{last_bad - INJ:.0f} s' if last_bad is not None else '-'):>20}"
          f"{worst:>10.0f} s{sess:>14}"
          f"{(f'{back - REC:.0f} s' if back is not None else 'never'):>16}")
    kinds = collections.Counter(str(r["outcome"]) for r in bad)
    print(f"{'':<18}outcomes: {dict(kinds)}")
