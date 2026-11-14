#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written after the R7-E3 verdicts were seen; NOT a rule test). Why does the new theta_trust rank
rollout errors well on the mixed bags (E3-D2 (1) passes) while no trust rule pools better than the uniform mean
(E3-D2 (2) fails)? From r7/runs/s<seed>/eval/d2_rollouts.csv and d2_actions.csv only (no model, no new rollouts):
  - squared value error (e_r - Q*_eff)^2 and KL(Q* || q_r) by tier and by terminal-like (H < 0.15) or not;
  - Spearman of beta and of -H with the NEGATIVE squared value error (the quantity pooling MSE is built from), overall and
    within (s, a) pairs, next to the pre-registered KL-based ranking;
  - the uniform pool's MSE split into a rollout-noise part (within-pair variance / 16) and the rest (bias against Q*,
    including Q*'s own playout noise), and the extra noise that trust_pool's unequal weights add (variance x sum w^2).
Writes r7/report/expl_d2.json.  python3 r7/expl_r7d2.py"""
import csv
import json
import math
import os

import numpy as np
from scipy.stats import spearmanr

H_ = os.path.dirname(os.path.abspath(__file__))
T975 = 4.302652729749464


def load(p):
    with open(p) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def ci(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    h = T975 * v.std(ddof=1) / math.sqrt(len(v))
    return [m, m - h, m + h]


def sp(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


per = {}
for s in (101, 102, 103):
    ev = os.path.join(H_, "runs", f"s{s}", "eval")
    r = load(os.path.join(ev, "d2_rollouts.csv"))
    a = load(os.path.join(ev, "d2_actions.csv"))
    alpha = json.load(open(os.path.join(ev, "trust.json")))["alpha"]
    P = int(r["pair"].max()) + 1
    g = lambda k: r[k].reshape(P, 24)
    e, H, beta, kl = g("e"), g("H"), g("beta"), g("kl")
    q = a["qstar_eff"][:, None]
    se = (e - q) ** 2
    out = {}
    for tier, sl in (("clean", slice(0, 16)), ("explore", slice(16, 24))):
        t = H[:, sl] < 0.15
        out[f"{tier}_sq_err_terminal_like"] = float(se[:, sl][t].mean())
        out[f"{tier}_sq_err_other"] = float(se[:, sl][~t].mean())
        out[f"{tier}_kl_terminal_like"] = float(kl[:, sl][t].mean())
        out[f"{tier}_kl_other"] = float(kl[:, sl][~t].mean())
        out[f"{tier}_share_terminal_like"] = float(t.mean())
    mix = np.r_[0:8, 16:24]
    em, Hm, bm, km, sem = e[:, mix], H[:, mix], beta[:, mix], kl[:, mix], se[:, mix]
    out["spearman_beta_negKL"] = sp(bm.ravel(), -km.ravel())
    out["spearman_beta_negSqErr"] = sp(bm.ravel(), -sem.ravel())
    out["spearman_negH_negSqErr"] = sp(-Hm.ravel(), -sem.ravel())
    out["within_beta_negSqErr"] = float(np.nanmean([sp(bm[i], -sem[i]) for i in range(P)]))
    out["within_negH_negSqErr"] = float(np.nanmean([sp(-Hm[i], -sem[i]) for i in range(P)]))
    v = em.var(1, ddof=1)
    w = bm ** alpha
    w = w / w.sum(1, keepdims=True)
    out["mse_unif"] = float(((em.mean(1) - a["qstar_eff"]) ** 2).mean())
    out["noise_unif"] = float((v / 16).mean())
    out["rest_unif"] = out["mse_unif"] - out["noise_unif"]
    out["noise_trust_pool"] = float((v * (w ** 2).sum(1)).mean())
    out["extra_noise_trust_pool"] = out["noise_trust_pool"] - out["noise_unif"]
    out["extra_mse_trust_pool"] = float(((a["q_trust_pool"] - a["qstar_eff"]) ** 2).mean()) - out["mse_unif"]
    out["mean_bias_unif"] = float((em.mean(1) - a["qstar_eff"]).mean())
    per[f"s{s}"] = out
keys = list(per["s101"])
res = {"per_seed": per, "three_seed": {k: ci([per[f"s{s}"][k] for s in (101, 102, 103)]) for k in keys}}
json.dump(res, open(os.path.join(H_, "report", "expl_d2.json"), "w"), indent=1)
for k in keys:
    m, lo, hi = res["three_seed"][k]
    print(f"{k:36s} {m:.5f} [{lo:.5f}, {hi:.5f}]")
