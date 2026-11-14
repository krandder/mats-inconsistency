"""wm2 audit: world-model diagnostics and the T2 loop numbers, from the raw JSONs (own code; numpy only).

  PYTHONDONTWRITEBYTECODE=1 python3 audit/wmdiag.py

[a] per task / WM tag / N (waves W1, W2, W3, RQ2): validation NLL parts, location information, repeat mass, dream
    validity, best steps, l0 flag, diag_post_n (the placement-list diagnostic must be off: 0);
[b] the T2 twin claim (PREREG 17:55Z): board-network NLL of WMs vs WMsb per (N, seed);
[c] T2 forecasters: KL to the selection-aware posterior per N (seed mean, min-max), KL to the unaware posterior,
    l0 flag; the unaware-exact reference 15.13e-3 (audit/t2check.py);
[d] flags of every T1 forecaster tag (real_world_valid, l0, flag) and which tags are board-blind.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = "/tmp/wm2/runs"
WAVES = ["W1", "W2", "W3", "RQ2"]
NS = [100, 300, 1000, 3000, 10000]


def load(task, kind):
    out = []
    for w in WAVES:
        for p in sorted(glob.glob(os.path.join(R, w, task, "*", "*.json"))):
            r = json.load(open(p))
            if (r.get("kind") == "wm") == (kind == "wm"):
                r["_wave"] = w
                out.append(r)
    return out


def main():
    print("[a] world models (seed means)")
    for task in ("t1", "t2"):
        wms = load(task, "wm")
        g = defaultdict(list)
        for w in wms:
            g[(w["wmtag"].split("-e")[0], w["n"])].append(w)
        dp = sorted({w["cfg"].get("diag_post_n") for w in wms})
        print(f"  {task}: {len(wms)} WM JSONs; diag_post_n values {dp}; l0 by tag "
              f"{sorted({(w['wmtag'].split('-e')[0], w.get('l0')) for w in wms})}")
        for (t, n), ws in sorted(g.items()):
            v = lambda k: np.mean([w["val"].get(k, np.nan) for w in ws])
            li = [w["val"].get("loc_info_nats_per_episode") for w in ws]
            print(f"   {task} {t:5s} N={n:5d} x{len(ws)}: NLL board {v('board'):.3f} loc {v('loc'):.3f} read {v('read'):.3f} | loc info "
                  f"{np.mean(li):+.3f} [{min(li):+.3f},{max(li):+.3f}] | repeat mass {v('repeat_mass_per_step'):.5f} | dream valid "
                  f"{np.mean([w['dreams']['valid_frac'] for w in ws]):.3f} | best step board {np.mean([w['best_step'] for w in ws]):.0f} "
                  f"ev {np.mean([w.get('best_step_ev', 0) for w in ws]):.0f}")
    print("\n[b] T2 twin claim: board NLL of WMs vs WMsb (same N, seed)")
    wms = load("t2", "wm")
    d = {(w["wmtag"], w["n"], w["seed"]): w["val"]["board"] for w in wms}
    diffs = []
    for (t, n, s), b in sorted(d.items()):
        if t == "WMs" and ("WMsb", n, s) in d:
            diffs.append(abs(b - d[("WMsb", n, s)]))
    print(f"  pairs {len(diffs)}; max |board NLL difference| {max(diffs):.6f} nats per episode; equal to 3 decimals: "
          f"{sum(round(x, 3) == 0 for x in diffs)}/{len(diffs)}")
    hb = {(w["wmtag"], w["n"], w["seed"]): (w["best_step"], w["steps_run"]) for w in wms}
    same = sum(hb[("WMs", n, s)][0] == hb[("WMsb", n, s)][0] for (t, n, s) in hb if t == "WMs" and ("WMsb", n, s) in hb)
    print(f"  board-network best step identical in {same}/{len(diffs)} pairs")
    print("\n[c] T2 forecasters, KL x1e3 to the selection-aware posterior (seed mean [min, max]); KL to the unaware posterior")
    fcs = load("t2", "fc")
    g = defaultdict(list)
    for r in fcs:
        g[(r["tag"], r["n"])].append(r)
    for t in sorted({k[0] for k in g}):
        cells = []
        for n in NS:
            rs = g.get((t, n), [])
            if not rs:
                continue
            k = [1e3 * r["val"]["kl"] for r in rs]
            ku = [1e3 * r["val"].get("kl_vs_unaware_post", np.nan) for r in rs]
            cells.append(f"{n}: {np.mean(k):.2f} [{min(k):.2f},{max(k):.2f}] (unaware {np.mean(ku):.2f})")
        l0 = sorted({r.get("l0", True) for r in fcs if r["tag"] == t})
        print(f"  {t:14s} l0={l0}: " + "; ".join(cells))
    print("\n[d] T1 forecaster flags")
    fcs = load("t1", "fc")
    fl = defaultdict(set)
    for r in fcs:
        fl[r["tag"]].add((r.get("real_world_valid", True), r.get("l0", True), r.get("flag", "")))
    for t, v in sorted(fl.items()):
        print(f"  {t:22s} (real_world_valid, l0, flag) = {sorted(v)}")


if __name__ == "__main__":
    main()
