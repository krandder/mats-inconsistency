"""Aggregate M7 w4 runs: one row per run (runs.csv) and mean +- sd per (arm, E, beta) (summary.csv).

Metrics (all on prequential trade-time predictions, live tokens only):
  bce_last     : BCE on the last 25% of positions.
  bce_all      : BCE over the whole pass.
  bce_postshift: mean BCE over the 2000 positions after each regime boundary (10k, 20k, 30k).
  bce_per_flop : bce_last / active FLOPs per token (FLOPs identical across arms at fixed E).
  max_load     : max over experts of the fraction of live tokens routed to it (top-2), last 25%.
  dead_frac    : fraction of experts with < 0.1% of routing slots, last 25%.
  mi_bits      : mutual information between expert (slot) and family, last 25%.
  leak_frac    : fraction of routed slots (last 25%) going to (family, expert) pairs whose mean
                 settlement over the last 25% is negative ("cross-market leak").
  leak_nats    : total negative settlement in those pairs per live token (nats/token, <= 0).
  gini_wealth  : Gini of per-expert total settlement (clipped at 0) over the last 25%.
"""
import glob, json, os, sys
import numpy as np
import csv

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def one(d):
    info = json.load(open(os.path.join(d, "info.json")))
    r = np.load(os.path.join(d, "raw.npz"))
    Z, Y, M = r["Z"].astype(np.float64), r["Y"], r["M"]
    bce = np.logaddexp(0, Z) - Y * Z
    n = len(Z); cut = int(0.75 * n)
    B = info["B"]; W = 32
    out = {k: info[k] for k in ("arm", "E", "seed", "beta", "half_life", "flops_tok", "wall")}
    out["bce_last"] = bce[cut:][M[cut:]].mean()
    out["bce_all"] = bce[M].mean()
    ps = []
    for b in (10000, 20000, 30000):
        lo = b - W
        if lo + 2000 <= n:
            ps.append(bce[lo:lo + 2000][M[lo:lo + 2000]].mean())
    out["bce_postshift"] = float(np.mean(ps)) if ps else np.nan
    out["bce_per_mflop"] = out["bce_last"] / (info["flops_tok"] / 1e6)
    sc = cut // B
    c = r["cnt_fe"][sc:].sum(0); p = r["pay_fe"][sc:].sum(0); nt = r["ntok_f"][sc:].sum()
    load = c.sum(0) / nt
    out["max_load"] = load.max()
    slot = c.sum(0) / c.sum()
    out["dead_frac"] = float((slot < 1e-3).mean())
    pj = c / c.sum(); pf = pj.sum(1, keepdims=True); pe = pj.sum(0, keepdims=True)
    nz = pj > 0
    out["mi_bits"] = float((pj[nz] * np.log2(pj[nz] / (pf @ pe)[nz])).sum())
    neg = (p < 0) & (c > 0)
    out["leak_frac"] = float(c[neg].sum() / c.sum())
    out["leak_nats"] = float(p[neg].sum() / nt)
    out["pay_per_tok"] = float(p.sum() / nt)
    out["gini_wealth"] = gini(p.sum(0))
    out["bias_abs_last"] = float(r["bias_abs"][sc:].mean())
    return out


rows = [one(os.path.dirname(f)) for f in sorted(glob.glob(os.path.join(ROOT, "*", "info.json")))]
keys = list(rows[0].keys())
with open(os.path.join(ROOT, "..", "runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
met = ["bce_last", "bce_all", "bce_postshift", "bce_per_mflop", "max_load", "dead_frac", "mi_bits",
       "leak_frac", "leak_nats", "pay_per_tok", "gini_wealth", "bias_abs_last"]
groups = {}
for r in rows:
    groups.setdefault((r["E"], r["arm"], r["beta"]), []).append(r)
out = []
for (E, arm, beta), rs in sorted(groups.items()):
    o = {"E": E, "arm": arm, "beta": beta, "n": len(rs)}
    for m in met:
        v = np.array([r[m] for r in rs], float)
        o[m + "_mean"] = float(v.mean()); o[m + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    out.append(o)
with open(os.path.join(ROOT, "..", "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0].keys())); w.writeheader(); w.writerows(out)
for o in out:
    print(o["E"], o["arm"], o["beta"], o["n"], " ".join(f"{m}={o[m+'_mean']:.4f}+-{o[m+'_sd']:.4f}" for m in
          ("bce_last", "bce_postshift", "max_load", "dead_frac", "mi_bits", "leak_frac", "leak_nats")))
