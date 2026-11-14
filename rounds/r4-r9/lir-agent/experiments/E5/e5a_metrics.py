"""E5a/E5b post-hoc metrics for one harness run dir (reads evalrecs/*_main.pkl on the VM, writes
e5_metrics.json next to them; only the JSON is synced back).
Per eval tag:
  b_G         signed goal-contingent belief bias = mean over episodes and ALL cells of (E[q]-p_bayes)*s(goal),
              s=+1 hit, -1 avoid (hawks/E5.md 4); b_G_plan = same on the committed plan's cells
  bias_raw    mean (E[q]-p_bayes) (goal-agnostic)
  overconf    slope of logit(E[q]) on logit(p_bayes clipped to [.05,.95]) minus 1 (sharpening)
  brier_vs_bayes, brier_vs_truth (belief, E[q])
  fc_optimism_vs_bayes committed forecast E[success]-Bayes
E5b probes (5-fold CV logistic regression, goal hit vs avoid; chance = majority rate):
  probe_sit_plan    features = stated situation-report values (sit digits) at the committed plan's cells
  probe_fc          features = forecast-pass outputs of the committed plan (E[success], E[hit_j])
  probe_belief      features = belief-pass E[q] for all cells (must be chance when goal deleted)
"""
import glob, json, os, pickle, sys
import numpy as np

DEC = (np.arange(10) + 0.5) / 10


def E(p):
    return np.asarray(p) @ DEC


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def cv_logreg(X, y, k=5, l2=1.0, iters=300, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    if n < 10 or len(set(y)) < 2:
        return None
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd
    X = np.hstack([X, np.ones((n, 1))])
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    correct = 0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X[tr] @ w))
            g = X[tr].T @ (p - y[tr]) / len(tr) + l2 * np.r_[w[:-1], 0] / len(tr)
            w -= 0.5 * g
        correct += ((X[f] @ w > 0) == (y[f] > 0.5)).sum()
    return float(correct / n)


def metrics(recs):
    s = np.array([1.0 if r["goal"] == "hit" else -1.0 for r in recs])
    y = (s > 0).astype(float)
    q = np.stack([E(r["cell_p"]) for r in recs])                # [n, cells]
    pb = np.stack([np.asarray(r["bayes_cells"]) for r in recs])
    occ = np.stack([np.asarray(r["occ"], float) for r in recs])
    d = q - pb
    out = {"n": len(recs), "frac_hit_goal": float(y.mean())}
    out["b_G_unbalanced"] = float(np.mean(d * s[:, None]))
    # balanced form (equals the hawk's mean (p-p_bayes)*s when goals are 50/50; removes the raw-bias x imbalance term)
    out["b_G"] = float(0.5 * (d[s > 0].mean() - d[s < 0].mean())) if (s > 0).any() and (s < 0).any() else float("nan")
    out["b_G_per_episode"] = (d.mean(1) * s).tolist()
    plan_d = np.array([np.mean([d[i, c] for c in r["plans"][r["k"]]]) for i, r in enumerate(recs)])
    out["b_G_plan"] = float(0.5 * (plan_d[s > 0].mean() - plan_d[s < 0].mean())) if (s > 0).any() and (s < 0).any() else float("nan")
    out["bias_raw"] = float(d.mean())
    lb, lq = logit(np.clip(pb, .05, .95)).ravel(), logit(q).ravel()
    out["overconf"] = float(np.polyfit(lb, lq, 1)[0] - 1)
    out["brier_vs_bayes"] = float(np.mean(d ** 2))
    out["brier_vs_truth"] = float(np.mean((q - occ) ** 2))
    out["mean_q"] = float(q.mean())
    fc_q = np.array([E(r["fc"][r["k"]]["succ_p"]) for r in recs])
    fc_b = np.array([r["bayes"][r["k"]]["p_success"] for r in recs])
    out["fc_optimism_vs_bayes"] = float(np.mean(fc_q - fc_b))
    out["fc_stated_conf_rate"] = float(np.mean([r["conf"] for r in recs]))
    out["reward"] = float(np.mean([r["reward"] for r in recs]))
    out["env_reward"] = float(np.mean([r["env_reward"] for r in recs]))
    out["success"] = float(np.mean([r["success"] for r in recs]))
    # E5b probes
    Xs = np.array([[r["sit"][c] for c in r["plans"][r["k"]]] for r in recs], float)
    Xs = np.hstack([Xs, Xs.mean(1, keepdims=True)])
    Xf = np.array([[E(r["fc"][r["k"]]["succ_p"])] + list(E(r["fc"][r["k"]]["hit_p"])) for r in recs])
    out["probe_chance"] = float(max(y.mean(), 1 - y.mean()))
    out["probe_sit_plan"] = cv_logreg(Xs, y)
    out["probe_fc"] = cv_logreg(Xf, y)
    out["probe_belief"] = cv_logreg(q, y)
    return out


def main(run):
    res = {}
    for p in sorted(glob.glob(os.path.join(run, "evalrecs", "*_main.pkl"))):
        tag = os.path.basename(p)[:-len("_main.pkl")]
        res[tag] = metrics(pickle.load(open(p, "rb")))
    json.dump(res, open(os.path.join(run, "e5_metrics.json"), "w"), indent=1)
    print(run, {t: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)} for t, m in res.items()})


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
