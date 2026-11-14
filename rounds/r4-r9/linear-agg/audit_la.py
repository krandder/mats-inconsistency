"""Independent self-audit (linear-agg): re-derives the headline numbers from the saved arrays with separate code.
A: MSEs from saved predictions; Diag-6 predictions, kappa ratio and influence ratio re-computed with a NumPy re-implementation
of the paper's eq. 10 from the saved omegas (finite differences, not autograd); selection efficiency from saved arrays.
B: KLs from saved pooled probabilities and exact posteriors; t-intervals; the conflict-influence Spearman with scipy.
Writes runs/audit.json and prints a comparison.   python3 audit_la.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth as S  # noqa: E402

AE = os.path.join(HERE, "runs", "a-eval")
BE = os.path.join(HERE, "runs", "b-eval")
YS = 3.0


def tci(x):
    x = np.asarray(x, float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


def diag_forward_np(om, keys, m, v, cand, keep=False):
    """NumPy re-implementation of Diag (paper eq. 10) on depth-whitened tokens; returns yhat and x-states."""
    sm = np.sqrt(m)
    x = keys / sm[..., None]
    y = v / sm / YS
    n = len(v)
    xq = np.concatenate([np.broadcast_to(S.KQ, (n, 1, S.D)), cand], 1)
    yq = np.zeros(xq.shape[:2])
    states = [x]
    for wxx, wxy, wyx, wyy in om:
        Sig = np.einsum("nkd,nke->nde", x, x)
        al = np.einsum("nk,nkd->nd", y, x)
        lam = (y * y).sum(1)
        x, y, xq, yq = (x + wxx * np.einsum("nde,nke->nkd", Sig, x) + wxy * y[..., None] * al[:, None],
                        y + wyx * np.einsum("nd,nkd->nk", al, x) + wyy * y * lam[:, None],
                        xq + wxx * np.einsum("nde,nke->nkd", Sig, xq) + wxy * yq[..., None] * al[:, None],
                        yq + wyx * np.einsum("nd,nkd->nk", al, xq) + wyy * yq * lam[:, None])
        states.append(x)
    return -YS * yq[:, 0], states, xq


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    path = os.path.join(HERE, "runs", "audit.json")
    out = json.load(open(path)) if os.path.exists(path) else {"A": {}, "B": {}}
    if "a" in which:
        out["A"] = audit_a()
    if "b" in which:
        out["B"] = audit_b()
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


def audit_a():
    out = {"A": {}}
    sa = json.load(open(os.path.join(AE, "summary.json")))
    za = np.load(os.path.join(AE, "arrays.npz"))
    worst = 0.0
    for key, val in sa["mse"].items():
        arm, k = key.rsplit("|", 1)
        if f"{arm}|{k}" in za.files:
            mse = float(np.mean((za[f"{arm}|{k}"] - za[f"yq|{k}"]) ** 2))
        elif arm == "prior0":
            mse = float(np.mean(za[f"yq|{k}"] ** 2))
        else:
            continue
        worst = max(worst, abs(mse - val))
    out["A"]["max_abs_diff_mse_vs_summary"] = worst
    pooled = S.episodes(50000, 9_100_000)
    assert np.allclose(pooled["yq"], za["yq|pooled"])
    per = {}
    for seed in (0, 1, 2):
        tr = json.load(open(os.path.join(HERE, "runs", "a", f"diag-L6-s{seed}", "train.json")))
        om = np.array(tr["omegas_xx_xy_yx_yy"])
        n = 2000
        e = {k: v[:n] for k, v in pooled.items() if isinstance(v, np.ndarray) and v.shape[:1] == (50000,)}
        yh, states, xq = diag_forward_np(om, e["keys"], e["m"], e["v"], e["cand"])
        dpred = float(np.abs(yh - za[f"diag-L6-s{seed}|pooled"][:n]).max())
        # kappa ratio on range(Sigma^0)
        S0 = np.einsum("nkd,nke->nde", states[0], states[0])
        ev, U = np.linalg.eigh(S0)
        r = int((ev > 1e-9 * ev[:, -1:]).sum(1).min())
        Ur = U[:, :, -r:]
        kap = []
        for x in (states[0], states[-1]):
            e2 = np.linalg.eigvalsh(np.einsum("ndr,nde,nes->nrs", Ur, np.einsum("nkd,nke->nde", x, x), Ur))
            kap.append(e2[:, -1] / e2[:, 0])
        kr = float(np.median(kap[1] / kap[0]))
        # influence by central finite differences on 500 episodes
        nf, h = 500, 1e-4
        I = np.zeros((nf, S.K))
        for rr in range(S.K):
            vp, vm = e["v"][:nf].copy(), e["v"][:nf].copy()
            vp[:, rr] += h
            vm[:, rr] -= h
            I[:, rr] = np.abs((diag_forward_np(om, e["keys"][:nf], e["m"][:nf], vp, e["cand"][:nf])[0]
                               - diag_forward_np(om, e["keys"][:nf], e["m"][:nf], vm, e["cand"][:nf])[0]) / (2 * h))
        bad = e["bad"][:nf]
        ir = float(I[bad].mean() / I[~bad].mean())
        # selection: top-2 of <x_q^L, x_z^L>^2, both-bridges and TV share on 2000 episodes
        sc = np.einsum("nd,nzd->nz", xq[:, 0], xq[:, 1:]) ** 2
        pk = np.argsort(-sc, 1, kind="stable")[:, :2]
        per[seed] = {"max_abs_pred_diff_vs_saved": dpred, "kappa_ratio_2000": kr,
                     "summary_kappa_ratio_5000": sa["diag"][f"diag-L6-s{seed}"]["kappa_ratio_median"],
                     "influence_ratio_fd_500": ir,
                     "summary_influence_ratio_5000": sa["diag"][f"diag-L6-s{seed}"]["influence_ratio_bad_clean"],
                     "tv_share_2000": float((S.CAND_TYPE[pk] == 2).mean()),
                     "both_bridges_2000": float((np.sort(pk, 1) == np.array([0, 1])).all(1).mean()),
                     "summary_eff_50000": sa["select"][f"diag-L6-s{seed}"]["eff"],
                     "saved_eff_mean": float(za[f"sel_eff|diag-L6-s{seed}"].mean())}
    out["A"]["diag_L6"] = per
    a1 = {}
    for arm in ("diag-L4", "diag-L6", "full-L4", "full-L6"):
        L = arm.split("L")[1]
        x = np.array([np.mean((za[f"{arm}-s{s}|pooled"] - za["yq|pooled"]) ** 2) for s in (0, 1, 2)])
        cmp_ = {}
        for b in ("mix", "sw", "dd", f"gdpp-L{L}"):
            bb = np.array([np.mean((za[(f"{b}|s{s}|pooled" if b in ("mix", "sw", "dd") else f"{b}-s{s}|pooled")]
                                    - za["yq|pooled"]) ** 2) for s in (0, 1, 2)])
            cmp_[b] = tci(bb - x)
        a1[arm] = {"mse": float(x.mean()), "diff_ci": cmp_}
    out["A"]["A1a"] = a1
    out["A"]["A1b_ratio_vs_adarr"] = {arm: {str(sg): float(np.mean([np.mean((za[f"{arm}-s{s}|s{sg}"] - za[f"yq|s{sg}"]) ** 2)
                                                                   for s in (0, 1, 2)]) / np.mean((za[f"adarr|s{sg}"] - za[f"yq|s{sg}"]) ** 2))
                                            for sg in (0.1, 0.5, 1.0, 2.0, 3.0)} for arm in ("diag-L6", "full-L6")}
    return out["A"]


def audit_b():
    out = {"B": {}}
    sb = json.load(open(os.path.join(BE, "summary.json")))
    klb = {}
    worst = 0.0
    sp = []
    for seed in (0, 1, 2):
        zb = np.load(os.path.join(BE, f"arrays-s{seed}.npz"))
        for f in ("B1", "B2"):
            p = np.clip(zb[f"pstar|{f}"], 1e-12, 1 - 1e-12)
            for arm in ("P0", "mix", "sw", "dd", "LT4", "LT1", "LT4_noconf", "rep_mix", "rep_sw", "rep_dd"):
                q = np.clip(zb[f"{arm}|{f}"], 1e-12, 1 - 1e-12)
                kl = float(np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))
                klb.setdefault(f"{arm}|{f}", []).append(kl)
                worst = max(worst, abs(kl - sb["per_seed"][str(seed)]["kl"][f"{arm}|{f}"]))
        I, c = zb["LT4_influence"], zb["test_conflict"]
        sp.append(float(np.nanmean([st.spearmanr(c[i], I[i]).statistic for i in range(len(I))])))
    out["B"]["max_abs_diff_kl_vs_summary"] = worst
    out["B"]["kl_3seed"] = {k: tci(v) for k, v in klb.items()}
    out["B"]["B_main_diff_best_pool_minus_LT4"] = {}
    for f in ("B1", "B2"):
        means = {p_: np.mean(klb[f"{p_}|{f}"]) for p_ in ("mix", "sw", "dd")}
        bp = min(means, key=means.get)
        out["B"]["B_main_diff_best_pool_minus_LT4"][f] = {"best_pool": bp, "ci": tci(np.array(klb[f"{bp}|{f}"]) - np.array(klb[f"LT4|{f}"]))}
        out["B"].setdefault("B_b_LT1_minus_LT4", {})[f] = tci(np.array(klb[f"LT1|{f}"]) - np.array(klb[f"LT4|{f}"]))
    out["B"]["B_a_spearman"] = {"per_seed": sp, "ci": tci(sp), "summary": sb["verdicts"]["B-a"]["per_seed"]}
    return out["B"]


if __name__ == "__main__":
    main()
