#!/usr/bin/env python3
"""Replay gateway requests through a per-replica LRU prefix cache, with and
without a host tier, to estimate what HiCache would return.

Sessions are reconstructed per consumer: agent contexts grow monotonically, so
a request joins the session whose last prompt is the largest one <= its own
prompt (within a growth bound and idle gap). Cached prefix after a turn is the
prompt only (conservative: reasoning is often stripped from resent history).
"""
import sys, collections

POOL = 169_408
GROW = 60_000        # max prompt growth between turns of one session
GAP_MS = 90 * 60_000 # idle gap that ends a session

def load(path, t0, t1, consumer=None):
    rows = []
    for line in open(path):
        ts, cons, inp, out, dur, st, ua = line.rstrip("\n").split("\t")
        ts, inp, out, dur = int(ts), int(inp), int(out), int(dur)
        if not (t0 <= ts < t1) or (consumer and cons != consumer):
            continue
        rows.append((ts, cons, inp, out, dur))
    return rows

def sessions(rows):
    act = collections.defaultdict(list)  # consumer -> [ [sid, last_inp, last_ts] ]
    nxt = 0; out = []
    for ts, cons, inp, o, dur in rows:
        best = None
        for s in act[cons]:
            if s[1] <= inp <= s[1] + o + GROW + 0 and ts - s[2] < GAP_MS:
                if best is None or s[1] > best[1]:
                    best = s
        if best is None:
            best = [nxt, 0, ts]; nxt += 1; act[cons].append(best)
        out.append((ts, best[0], inp, o, dur))
        best[1] = inp; best[2] = ts
    return out, nxt

def simulate(reqs, host_ratio):
    dev = [collections.OrderedDict(), collections.OrderedDict()]  # sid -> tokens
    host = [collections.OrderedDict(), collections.OrderedDict()]
    home = {}
    load_ = [0, 0]
    uncached = 0; reloaded = 0; misses = []  # (recomputed tokens) per request
    for ts, sid, inp, o, dur in reqs:
        if sid not in home:  # cache_aware: new prefix goes to the emptier replica
            home[sid] = 0 if sum(dev[0].values()) <= sum(dev[1].values()) else 1
        r = home[sid]; d, h = dev[r], host[r]
        hit = min(d.get(sid, 0), inp)
        from_host = 0
        if hit == 0 and sid in h:
            from_host = min(h.pop(sid), inp)
        new = inp - hit - from_host
        uncached += new; reloaded += from_host
        misses.append(new)
        d.pop(sid, None)
        need = inp + o
        # evict LRU sessions to make room; evictee goes to host tier if any
        while d and sum(d.values()) + need > POOL:
            esid, etok = d.popitem(last=False)
            if host_ratio:
                h[esid] = etok; h.move_to_end(esid)
                while sum(h.values()) > host_ratio * POOL:
                    h.popitem(last=False)
        d[sid] = min(inp, POOL)
    return uncached, reloaded, misses

def main():
    path, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = load(path, t0, t1)
    reqs, n = sessions(rows)
    prompt = sum(r[2] for r in reqs)
    print(f"requests {len(reqs)}  sessions {n}  prompt {prompt/1e6:.2f}M")
    for R in (0, 2, 3, 4):
        u, rl, m = simulate(reqs, R)
        big = sorted(m)[int(0.95 * len(m))]
        over16 = sum(1 for x in m if x > 16_000)
        print(f"host_ratio={R}: uncached {u/1e6:.2f}M ({100*u/prompt:.1f}% of prompt)  "
              f"reloaded-from-host {rl/1e6:.2f}M  p95 recompute {big:,}  reqs>16K recompute {over16}")

main()
