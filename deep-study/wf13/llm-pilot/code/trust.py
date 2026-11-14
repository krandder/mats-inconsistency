# trust.py OUT_DIR : the smallest trust head (H7, exploratory). A logistic map from the aggregated top-answer
# probability (and the family) to "the top answer is right", fitted on the CALIBRATION problems' real answers only
# (seed 300), tested on the TEST problems (seed 200). Writes OUT_DIR/RESULT-trust.json.
import sys, json, math
import numpy as np
from analyze import load, agg, top, blocks

OUT = sys.argv[1]
K = 16

def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

def fit(X, y, l2=1e-2, it=50):
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w

def pred(w, X):
    return 1 / (1 + np.exp(-np.asarray(X, float) @ w))

def rows(recs, wname, g, K, rich=False):
    X, y, raw = [], [], []
    for r in recs:
        wr = r["w"][wname]
        for sel in blocks(len(wr["id"]), K):
            d, extra = agg(r, wr, sel, g)
            a, p = top(d)
            f = [1.0, logit(p) if d else logit(0.0), 1.0 if r["fam"] == "vars" else 0.0]
            if rich:  # search statistics as extra inputs (Kelvin's "augmented submitted state")
                dv, _ = agg(r, wr, sel, "vote"); av, pv = top(dv)
                ds, ess = agg(r, wr, sel, "snis"); _, fm = agg(r, wr, sel, "dedup")
                f += [logit(pv) if dv else logit(0.0), 1.0 if av == a else 0.0, ess or 0.0, math.log(max(fm or 0.0, 1e-6)),
                      float(np.mean([wr["id"][k] >= 0 for k in sel]))]
            X.append(f); y.append(1.0 if (d and a == r["answer"]) else 0.0); raw.append(p if d else 0.0)
    return X, y, raw

def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))

def ece(p, y, nb=10):
    p = np.asarray(p); y = np.asarray(y); e = 0.0
    for b in range(nb):
        m = (p >= b / nb) & (p < (b + 1) / nb + (1e-9 if b == nb - 1 else 0))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def main():
    test = load("test"); cal = load("calib")
    writers = list(test[0]["w"].keys())
    res = {"K": K, "n_calib": len(cal), "n_test": len(test), "simple": {}, "rich": {}, "target": {}}
    for g in ("vote", "dedup", "snis"):
        Xc, yc, _ = rows(cal, "W1", g, K)
        w = fit(Xc, yc)
        res["simple"][g] = {"weights[bias,logit_p,vars]": [float(x) for x in w], "apply": {}}
        for wn in writers:
            Xt, yt, raw = rows(test, wn, g, K)
            p = pred(w, Xt)
            res["simple"][g]["apply"][wn] = {"acc": float(np.mean(yt)), "brier_raw": brier(raw, yt), "brier_trust": brier(p, yt),
                                             "ece_raw": ece(raw, yt), "ece_trust": ece(p, yt), "mean_conf_trust": float(np.mean(p))}
    # a head that also reads search statistics, fitted on the calibration problems pooled over all writers
    Xc, yc = [], []
    for wn in writers:
        a, b, _ = rows(cal, wn, "snis", K, rich=True); Xc += a; yc += b
    w = fit(Xc, yc)
    res["rich"]["weights[bias,logit_snis,vars,logit_vote,agree,ess_share,log_found,valid_share]"] = [float(x) for x in w]
    res["rich"]["apply"] = {}
    for wn in writers:
        Xt, yt, raw = rows(test, wn, "snis", K, rich=True)
        p = pred(w, Xt)
        res["rich"]["apply"][wn] = {"acc": float(np.mean(yt)), "brier_raw_snis": brier(raw, yt), "brier_trust": brier(p, yt),
                                    "ece_trust": ece(p, yt)}
    # is the target itself (M's belief after thinking, from the 256 reference rollouts) calibrated against reality?
    conf, hit, fam = [], [], []
    for r in test:
        d, _ = agg(r, r["ref"], range(len(r["ref"]["id"])), "vote"); a, p = top(d)
        conf.append(p); hit.append(1.0 if a == r["answer"] else 0.0); fam.append(r["fam"])
    conf = np.asarray(conf); hit = np.asarray(hit)
    res["target"]["brier"] = brier(conf, hit); res["target"]["ece"] = ece(conf, hit)
    res["target"]["bins"] = [{"lo": lo, "hi": hi, "n": int(((conf >= lo) & (conf < hi)).sum()),
                              "mean_conf": float(conf[(conf >= lo) & (conf < hi)].mean()) if ((conf >= lo) & (conf < hi)).any() else None,
                              "acc": float(hit[(conf >= lo) & (conf < hi)].mean()) if ((conf >= lo) & (conf < hi)).any() else None}
                             for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01))]
    json.dump(res, open(OUT + "/RESULT-trust.json", "w"), indent=1)
    print("written", OUT + "/RESULT-trust.json")

if __name__ == "__main__":
    main()
