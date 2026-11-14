#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written after the R6-E3 verdicts were seen; NOT a rule test). Why does trust_pool rank rollout
errors well (E3-D2 (1)) yet pool worse than uniform weights (E3-D2 (2))? From r6/runs/s<seed>/eval/d2_rollouts.csv and
d2_actions.csv only (no model, no new rollouts):
  - Spearman(beta, -KL) between (position, action) pairs (pair means) vs within pairs (mean of per-pair Spearman over
    the 16 rollouts of the same action, which share one Q*);
  - Kish effective sample size of the trust weights per pair, (sum w)^2 / sum w^2, out of 16;
  - squared error of the pooled estimate split into the pair-level bias of each pool.
Writes r6/report/expl_d2.json.  python3 r6/expl_d2.py"""
import csv
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr

H = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def ci(v):
    v = np.asarray(v, float)
    m = v.mean()
    h = 4.302652729749464 * v.std(ddof=1) / math.sqrt(len(v))
    return [float(m), float(m - h), float(m + h)]


out = {}
per = {k: [] for k in ("between_trust", "between_stiff", "within_trust", "within_stiff", "ess_trust", "trust_weight_top1",
                       "within_kl_sd", "between_kl_sd")}
for s in (101, 102, 103):
    r = load(os.path.join(H, "runs", f"s{s}", "eval", "d2_rollouts.csv"))
    pair = r["pair"].astype(int)
    npair = pair.max() + 1
    K = np.bincount(pair)[0]
    beta = r["beta"].reshape(npair, K)
    Hh = r["H"].reshape(npair, K)
    kl = r["kl"].reshape(npair, K)
    y = -kl
    bt = spearmanr(beta.mean(1), y.mean(1)).correlation
    bs = spearmanr(-Hh.mean(1), y.mean(1)).correlation
    wt, ws = [], []
    for i in range(npair):
        if np.std(beta[i]) > 0 and np.std(y[i]) > 0:
            wt.append(spearmanr(beta[i], y[i]).correlation)
        if np.std(Hh[i]) > 0 and np.std(y[i]) > 0:
            ws.append(spearmanr(-Hh[i], y[i]).correlation)
    w = beta / beta.sum(1, keepdims=True)
    ess = 1.0 / (w ** 2).sum(1)
    per["between_trust"].append(bt)
    per["between_stiff"].append(bs)
    per["within_trust"].append(float(np.nanmean(wt)))
    per["within_stiff"].append(float(np.nanmean(ws)))
    per["ess_trust"].append(float(ess.mean()))
    per["trust_weight_top1"].append(float(w.max(1).mean()))
    per["within_kl_sd"].append(float(kl.std(1).mean()))
    per["between_kl_sd"].append(float(kl.mean(1).std()))
    out[f"s{s}"] = {k: v[-1] for k, v in per.items()}
out["three_seed"] = {k: ci(v) for k, v in per.items()}
json.dump(out, open(os.path.join(H, "report", "expl_d2.json"), "w"), indent=1)
print(json.dumps(out["three_seed"], indent=1))

# --- shrinkage check (appended in the same post hoc pass): are trusted rollouts the hedged ones (claims near 0.5)?
ext = {k: [] for k in ("within_beta_vs_extremeness", "abs_dev_unif", "abs_dev_trust", "abs_dev_qstar", "bias_toward_half_trust")}
for s in (101, 102, 103):
    r = load(os.path.join(H, "runs", f"s{s}", "eval", "d2_rollouts.csv"))
    a = load(os.path.join(H, "runs", f"s{s}", "eval", "d2_actions.csv"))
    pair = r["pair"].astype(int)
    npair = pair.max() + 1
    K = np.bincount(pair)[0]
    beta = r["beta"].reshape(npair, K)
    e = r["e"].reshape(npair, K)
    xs = [spearmanr(beta[i], np.abs(e[i] - 0.5)).correlation for i in range(npair) if np.std(beta[i]) > 0 and np.std(e[i]) > 0]
    ext["within_beta_vs_extremeness"].append(float(np.nanmean(xs)))
    ext["abs_dev_unif"].append(float(np.abs(a["q_unif"] - 0.5).mean()))
    ext["abs_dev_trust"].append(float(np.abs(a["q_trust"] - 0.5).mean()))
    ext["abs_dev_qstar"].append(float(np.abs(a["qstar_eff"] - 0.5).mean()))
    # share of pairs where trust moves the estimate toward 0.5 relative to unif
    ext["bias_toward_half_trust"].append(float((np.abs(a["q_trust"] - 0.5) < np.abs(a["q_unif"] - 0.5)).mean()))
out["shrinkage"] = {k: ci(v) for k, v in ext.items()}
out["shrinkage_per_seed"] = ext
json.dump(out, open(os.path.join(H, "report", "expl_d2.json"), "w"), indent=1)
print(json.dumps(out["shrinkage"], indent=1))

# --- variance check (same post hoc pass): how much of trust_pool's extra MSE does its smaller effective sample explain?
# Treating the 16 rollouts of a pair as i.i.d. draws with within-pair variance v, the noise part of the squared error is
# v / 16 for unif_pool and v * sum(w^2) for trust_pool.
var = {k: [] for k in ("mse_unif", "mse_trust", "noise_unif", "noise_trust", "extra_mse", "extra_noise")}
for s in (101, 102, 103):
    r = load(os.path.join(H, "runs", f"s{s}", "eval", "d2_rollouts.csv"))
    a = load(os.path.join(H, "runs", f"s{s}", "eval", "d2_actions.csv"))
    pair = r["pair"].astype(int)
    npair = pair.max() + 1
    K = np.bincount(pair)[0]
    beta = r["beta"].reshape(npair, K)
    e = r["e"].reshape(npair, K)
    w = beta / beta.sum(1, keepdims=True)
    v = e.var(1, ddof=1)
    var["mse_unif"].append(float(((a["q_unif"] - a["qstar_eff"]) ** 2).mean()))
    var["mse_trust"].append(float(((a["q_trust"] - a["qstar_eff"]) ** 2).mean()))
    var["noise_unif"].append(float((v / K).mean()))
    var["noise_trust"].append(float((v * (w ** 2).sum(1)).mean()))
    var["extra_mse"].append(var["mse_trust"][-1] - var["mse_unif"][-1])
    var["extra_noise"].append(var["noise_trust"][-1] - var["noise_unif"][-1])
out["variance"] = {k: ci(vv) for k, vv in var.items()}
out["variance_per_seed"] = var
json.dump(out, open(os.path.join(H, "report", "expl_d2.json"), "w"), indent=1)
print(json.dumps(out["variance"], indent=1))
