"""Soup reference for M5 item 2: sh_arb (the soup's one-share AND/OR coherence arbitrageur, logical-soup/docs/shadow.md)
cumulative profit, per block per epoch (blocks = 2^F/4, soup.md: questions come in blocks of four), plus coh_and/coh_or.
Reads logical-soup/runs/phase3/<cell>-s<seed>/metrics.csv read-only. Same slope conventions as M5/w3/soup_summary.py."""
import csv, json, os, re
import numpy as np
ROOT = "/home/agent/workspaces/eurisko-study/logical-soup/runs/phase3"
HERE = os.path.dirname(os.path.abspath(__file__))

def ls(c):
    t = np.arange(len(c), dtype=float); t -= t.mean()
    return float((t * (c - c.mean())).sum() / (t * t).sum())

rows = []
for cell in ["k12-f8", "k12-f10", "k12-f12", "k14-f8", "k14-f10", "k14-f12", "k16-f12"]:
    K, F = map(int, re.findall(r"\d+", cell)); nb = 2 ** F / 4
    for s in (1, 2, 3):
        p = f"{ROOT}/{cell}-s{s}/metrics.csv"
        if not os.path.exists(p):
            continue
        r = list(csv.DictReader(open(p)))
        c = np.array([float(x["sh_arb"]) for x in r]) / 65536.0 / nb
        n = len(c); h = n // 2; tail = slice(int(n * 0.8), n)
        rows.append(dict(cell=cell, K_over_Q=2 ** K / 2 ** F, seed=s, epochs=n, slope_h1=ls(c[:h]), slope_h2=ls(c[h:]),
                         phi_last20=ls(c[tail]), E_end=float(c[-1]),
                         coh_and_last20=float(np.mean([float(x["coh_and_ppm"]) for x in r[tail]]) / 1e6),
                         coh_or_last20=float(np.mean([float(x["coh_or_ppm"]) for x in r[tail]]) / 1e6)))
os.makedirs(f"{HERE}/results", exist_ok=True)
with open(f"{HERE}/results/soup_arb.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
agg = {}
for cell in sorted({r["cell"] for r in rows}):
    sel = [r for r in rows if r["cell"] == cell]
    for m in ("slope_h1", "slope_h2", "phi_last20", "E_end", "coh_and_last20", "coh_or_last20"):
        v = np.array([r[m] for r in sel])
        agg[f"{cell}:arb:{m}"] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), int((v > 0).sum()), len(v)]
json.dump(agg, open(f"{HERE}/results/soup_arb_agg.json", "w"), indent=1)
for k, v in agg.items():
    if "phi" in k or "slope" in k or "coh" in k: print(k, "%.5f ± %.5f  pos %d/%d" % tuple(v))
