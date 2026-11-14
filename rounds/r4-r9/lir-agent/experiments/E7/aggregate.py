"""E7 aggregation (CPU). Reads analyze.py JSONs + run eval.jsonl, writes results.json, tables.csv, PNGs.
Usage: python3 aggregate.py --an DIR --runs DIR --out DIR"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

BINS = [0.0, 0.25, 0.75, 1.5, 99.0]
TAGS = ["pre_rl", "rl30", "post_rl"]


def tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(-1)


def binidx(h):
    return np.clip(np.digitize(h, BINS) - 1, 0, len(BINS) - 2)


def matched(correct, h, ref):
    """entropy-matched mean of a per-state 0/1 (or real) array: sum_b ref_b mean_b, over bins present."""
    b = binidx(h)
    tot, w = 0.0, 0.0
    for k in range(len(BINS) - 1):
        m = b == k
        if m.any() and ref[k] > 0:
            tot += ref[k] * float(np.mean(correct[m])); w += ref[k]
    return tot / w if w else float("nan")


def msd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return {"mean": None, "sd": None, "n": 0}
    a = np.array(xs, float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    from math import sqrt
    tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45}.get(len(a), 2.0)
    half = tcrit * sd / sqrt(len(a)) if len(a) > 1 else None
    return {"mean": float(a.mean()), "sd": sd, "n": len(a),
            "ci95_t": None if half is None else [float(a.mean() - half), float(a.mean() + half)], "per_seed": a.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--an", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", default=TAGS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(os.path.join(a.an, "states.json")))
    B = np.array(S["bayes_test"]); W = S["W"]
    nc = B.shape[1]
    cells = np.arange(nc); col = cells % W; row = cells // W
    solver = B.argmax(-1)
    ins, fr = {}, {}
    for f in glob.glob(os.path.join(a.an, "insitu_*.json")):
        r = json.load(open(f)); ins[(r["arm"], r["seed"], r["tag"])] = r
    for f in glob.glob(os.path.join(a.an, "fresh_*.json")):
        r = json.load(open(f)); fr[(r["arm"], r["seed"], r["tag"])] = r
    arms = sorted({k[0] for k in ins}, key=lambda x: ["P-all", "P-role", "B0"].index(x))
    seeds = sorted({k[1] for k in ins})
    tags = [t for t in a.tags if any(k[2] == t for k in ins)]
    # reference entropy histogram: pooled over every in-situ unit
    allh = np.concatenate([np.array(r["H_pi"]) for r in ins.values()])
    ref = np.bincount(binidx(allh), minlength=len(BINS) - 1) / len(allh)
    # idiosyncratic subset: y != solver and y != every other seed's y (same arm, tag)
    def idio(arm, s, tag):
        y = np.array(ins[(arm, s, tag)]["y"])
        m = y != solver
        for s2 in seeds:
            if s2 != s and (arm, s2, tag) in ins:
                m &= y != np.array(ins[(arm, s2, tag)]["y"])
        return m
    # rewards from runs
    rew = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(a.runs, f"{arm}-s{s}", "eval.jsonl")
            if os.path.exists(p):
                for line in open(p):
                    e = json.loads(line); mm = e["main"]
                    rew[(arm, s, e["tag"])] = {k: mm.get(k) for k in ("reward", "env_reward", "success", "self_acc",
                                                                       "self_mode_baseline", "commit_entropy",
                                                                       "introspect_acc_vs_policy_mode")}
    for arm in arms:   # pre_rl fallback: the warm-start run's own pre_rl eval (same checkpoint, same eval seeds)
        for s in seeds:
            p = os.path.join(a.runs, f"warm-{arm}-s{s}", "eval.jsonl")
            if (arm, s, "pre_rl") not in rew and os.path.exists(p):
                mm = json.loads(open(p).readline())["main"]
                rew[(arm, s, "pre_rl")] = {k: mm.get(k) for k in ("reward", "env_reward", "success", "self_acc",
                                                                   "self_mode_baseline", "commit_entropy",
                                                                   "introspect_acc_vs_policy_mode")}
    unit = {}
    for (arm, s, tag), r in ins.items():
        y = np.array(r["y"]); pi = np.array(r["pi"]); q = np.array(r["q"]); h = np.array(r["H_pi"])
        qa = q.argmax(-1)
        mode = np.bincount(y, minlength=nc).argmax()
        u = {"H_pi": float(h.mean()), "acc_insitu": float((qa == y).mean()),
             "acc_insitu_matched": matched((qa == y).astype(float), h, ref),
             "mode_acc_test": float((y == mode).mean()), "mode_acc_matched": matched((y == mode).astype(float), h, ref),
             "tv_q_pi": float(tv(q, pi).mean()),
             "val_signed_exp": float(((q - pi) * B).sum(-1).mean()),
             "val_signed_argmax": float((B[np.arange(len(y)), qa] - B[np.arange(len(y)), y]).mean()),
             "neu_acc_row": float((row[qa] == row[y]).mean()), "neu_acc_col": float((col[qa] == col[y]).mean()),
             "neu_signed_col": float(((q - pi) * col).sum(-1).mean()),
             "neu_abs_col": float(np.abs(((q - pi) * col).sum(-1)).mean()),
             "frac_y_eq_solver": float((y == solver).mean())}
        m = idio(arm, s, tag)
        u["idio_frac"] = float(m.mean())
        u["acc_insitu_idio"] = float((qa[m] == y[m]).mean()) if m.any() else None
        u["fulfil"] = r["fulfil"]
        if "q_frozen_live_action" in r:
            qf = np.array(r["q_frozen_live_action"])
            u["acc_frozen_live_action"] = float((qf.argmax(-1) == y).mean())
            u["acc_frozen_live_action_matched"] = matched((qf.argmax(-1) == y).astype(float), h, ref)
        if (arm, s, "pre_rl") in ins and tag != "pre_rl":
            q0 = np.array(ins[(arm, s, "pre_rl")]["q"])
            u["acc_stale_warm_selfmodel"] = float((q0.argmax(-1) == y).mean())
        u.update({f"eval_{k}": v for k, v in rew.get((arm, s, tag), {}).items()})
        unit[(arm, s, tag)] = u
    # fresh predictors (Binder M1 vs M2)
    funit = {}
    for (arm, s, tag), r in fr.items():
        y = np.array(r["y"]); h = np.array(r["H_pi"])
        acc = {n: np.mean([(np.array(p).argmax(-1) == y).astype(float) for p in ps], 0) for n, ps in r["pred"].items()}
        modec = (y == r["y_train_mode"]).astype(float)
        m = idio(arm, s, tag) if (arm, s, tag) in ins else np.zeros(len(y), bool)
        u = {"n_pred_seeds": len(r["pred"]["S"]), "H_pi": float(h.mean())}
        for n in ("S", "X1", "X0"):
            u[f"acc_{n}"] = float(acc[n].mean()); u[f"acc_{n}_matched"] = matched(acc[n], h, ref)
            u[f"acc_{n}_idio"] = float(acc[n][m].mean()) if m.any() else None
            pe = np.mean([np.array(p) for p in r["pred"][n]], 0)
            u[f"val_signed_exp_{n}"] = float(((pe - np.array(r["pi"])) * B).sum(-1).mean())
        # robustness (added 10:30Z after 2 Gate-0 seeds showed collapsed predictor runs): drop predictor runs
        # whose argmax is one constant cell on > 90% of test states (a failed fit, not a property)
        for n, ps in r["pred"].items():
            keep = [np.array(p) for p in ps if np.bincount(np.array(p).argmax(-1), minlength=nc).max() / len(y) <= 0.9]
            u[f"collapsed_{n}"] = len(ps) - len(keep)
            keep = keep or [np.array(p) for p in ps]
            u[f"acc_{n}_nc"] = float(np.mean([(p.argmax(-1) == y).mean() for p in keep]))
        u["acc_mode"] = float(modec.mean()); u["acc_mode_matched"] = matched(modec, h, ref)
        u["self_adv"] = u["acc_S"] - max(u["acc_X1"], u["acc_X0"], u["acc_mode"])
        u["self_adv_matched"] = u["acc_S_matched"] - max(u["acc_X1_matched"], u["acc_X0_matched"], u["acc_mode_matched"])
        u["self_adv_nc"] = u["acc_S_nc"] - max(u["acc_X1_nc"], u["acc_X0_nc"], u["acc_mode"])
        u["self_adv_vs_X0"] = u["acc_S"] - u["acc_X0"]; u["self_adv_vs_X1"] = u["acc_S"] - u["acc_X1"]
        if m.any():
            u["self_adv_idio"] = u["acc_S_idio"] - max(u["acc_X1_idio"], u["acc_X0_idio"])
        u["_acc_state"] = {n: acc[n] for n in acc}; u["_mode_state"] = modec
        funit[(arm, s, tag)] = u
    # hierarchical bootstrap (seeds, then states) of self_adv per (arm, tag)
    rng = np.random.default_rng(0)
    boot = {}
    for arm in sorted({k[0] for k in funit}):
        for tag in sorted({k[2] for k in funit}):
            ks = [k for k in funit if k[0] == arm and k[2] == tag]
            if len(ks) < 2:
                continue
            vals = []
            for _ in range(2000):
                tot = []
                for k in [ks[i] for i in rng.integers(len(ks), size=len(ks))]:
                    u = funit[k]; n = len(u["_mode_state"]); idx = rng.integers(n, size=n)
                    sA = u["_acc_state"]["S"][idx].mean()
                    tot.append(sA - max(u["_acc_state"]["X1"][idx].mean(), u["_acc_state"]["X0"][idx].mean(),
                                        u["_mode_state"][idx].mean()))
                vals.append(np.mean(tot))
            boot[f"{arm}|{tag}"] = {"mean_of_boot": float(np.mean(vals)), "ci95": [float(np.percentile(vals, 2.5)),
                                                                                   float(np.percentile(vals, 97.5))]}
    # direction of resolution
    direc = defaultdict(list)
    for arm in arms:
        for s in seeds:
            for t0, t1 in zip(tags[:-1], tags[1:]):
                if (arm, s, t0) in ins and (arm, s, t1) in ins:
                    A, Bn = ins[(arm, s, t0)], ins[(arm, s, t1)]
                    q0, p0, q1, p1 = (np.array(A["q"]), np.array(A["pi"]), np.array(Bn["q"]), np.array(Bn["pi"]))
                    d_pred_old_act_new = float(tv(q0, p1).mean()); d_pred_new_act_old = float(tv(q1, p0).mean())
                    direc[(arm, f"{t0}->{t1}")].append({"seed": s, "d(pred_t,act_t+1)": d_pred_old_act_new,
                                                        "d(pred_t+1,act_t)": d_pred_new_act_old,
                                                        "d(pred_t,act_t)": float(tv(q0, p0).mean()),
                                                        "d(pred_t+1,act_t+1)": float(tv(q1, p1).mean()),
                                                        "d(act_t,act_t+1)": float(tv(p0, p1).mean()),
                                                        "d(pred_t,pred_t+1)": float(tv(q0, q1).mean()),
                                                        "D=d(pred_t,act_t+1)-d(pred_t+1,act_t)": d_pred_old_act_new - d_pred_new_act_old})
    # tables: mean +- sd over seeds
    def table(units, keys):
        out = {}
        for arm in arms:
            for tag in tags:
                row_ = {}
                for k in keys:
                    xs = [units[(arm, s, tag)].get(k) for s in seeds if (arm, s, tag) in units]
                    xs = [x for x in xs if x is not None and not isinstance(x, dict)]
                    if xs:
                        row_[k] = msd(xs)
                if row_:
                    out[f"{arm}|{tag}"] = row_
        return out
    ikeys = ["eval_reward", "eval_env_reward", "eval_success", "eval_commit_entropy", "eval_self_acc", "H_pi", "acc_insitu",
             "acc_insitu_matched", "mode_acc_test", "mode_acc_matched", "tv_q_pi", "val_signed_exp", "val_signed_argmax",
             "neu_acc_row", "neu_acc_col", "neu_signed_col", "neu_abs_col", "frac_y_eq_solver", "idio_frac",
             "acc_insitu_idio", "acc_frozen_live_action", "acc_frozen_live_action_matched", "acc_stale_warm_selfmodel"]
    for (k, u) in unit.items():
        for kk, v in u["fulfil"].items():
            u[f"fulfil_{kk}"] = v
    ikeys += [f"fulfil_{k}" for k in next(iter(unit.values()))["fulfil"]]
    fkeys = ["n_pred_seeds", "H_pi", "acc_S", "acc_X1", "acc_X0", "acc_mode", "self_adv", "self_adv_vs_X0", "self_adv_vs_X1",
             "acc_S_matched", "acc_X1_matched", "acc_X0_matched", "acc_mode_matched", "self_adv_matched",
             "acc_S_idio", "acc_X1_idio", "acc_X0_idio", "self_adv_idio", "val_signed_exp_S", "val_signed_exp_X0",
             "val_signed_exp_X1", "acc_S_nc", "acc_X1_nc", "acc_X0_nc", "self_adv_nc", "collapsed_S", "collapsed_X1", "collapsed_X0"]
    T_in, T_fr = table(unit, ikeys), table(funit, fkeys)
    T_dir = {f"{arm}|{tr}": {k: msd([d[k] for d in v]) for k in v[0] if k != "seed"} for (arm, tr), v in direc.items()}
    # change pre -> post per arm (per seed differences)
    delta = {}
    for arm in arms:
        for k in ["acc_insitu", "acc_insitu_matched", "val_signed_exp", "neu_abs_col", "H_pi", "eval_reward", "tv_q_pi"]:
            xs = [unit[(arm, s, "post_rl")][k] - unit[(arm, s, "pre_rl")][k] for s in seeds
                  if (arm, s, "post_rl") in unit and (arm, s, "pre_rl") in unit and unit[(arm, s, "post_rl")].get(k) is not None]
            delta[f"{arm}|{k}"] = msd(xs)
        for k in ["self_adv", "self_adv_matched", "self_adv_nc", "acc_S", "acc_X0"]:
            xs = [funit[(arm, s, "post_rl")][k] - funit[(arm, s, "pre_rl")][k] for s in seeds
                  if (arm, s, "post_rl") in funit and (arm, s, "pre_rl") in funit]
            if xs:
                delta[f"{arm}|fresh_{k}"] = msd(xs)
    # matched-reward OLS over all in-situ units: metric ~ 1 + reward + arm dummies
    ols = {}
    ks = [k for k in unit if unit[k].get("eval_reward") is not None]
    if ks:
        for met in ["acc_insitu_matched", "val_signed_exp", "tv_q_pi"]:
            X = [[1.0, unit[k]["eval_reward"]] + [1.0 * (k[0] == arm) for arm in arms[1:]] for k in ks]
            yv = [unit[k][met] for k in ks]
            coef, *_ = np.linalg.lstsq(np.array(X), np.array(yv), rcond=None)
            ols[met] = {"intercept": float(coef[0]), "reward": float(coef[1]),
                        **{f"arm_{arm}_minus_{arms[0]}": float(c) for arm, c in zip(arms[1:], coef[2:])},
                        "n_units": len(ks), "note": "pooled over seeds and tags; descriptive, no CI"}
    # primary (PREREG): post_rl val_signed_exp, B0 minus P-all, Welch t 95% CI over seeds; plus companions
    def welch(xa, xb):
        from math import sqrt
        a_, b_ = np.array(xa, float), np.array(xb, float)
        if len(a_) < 2 or len(b_) < 2:
            return None
        va, vb = a_.var(ddof=1) / len(a_), b_.var(ddof=1) / len(b_)
        d = float(a_.mean() - b_.mean()); se = sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a_) - 1) + vb ** 2 / (len(b_) - 1)) if (va + vb) > 0 else 1.0
        TT = [(1, 12.71), (2, 4.30), (3, 3.18), (4, 2.78), (5, 2.57), (6, 2.45), (7, 2.36), (8, 2.31), (9, 2.26),
              (10, 2.23), (12, 2.18), (15, 2.13), (20, 2.09), (30, 2.04)]
        t = 1.96
        for dd, tv_ in TT:   # conservative: critical value at floor(dof)
            if dof >= dd:
                t = tv_
        t = t if dof < 30 else 2.04
        return {"diff": d, "ci95": [d - t * se, d + t * se], "dof": float(dof), "n": [len(a_), len(b_)]}
    primary = {}
    def per_seed(arm, tag, k, src=unit):
        return [src[(arm, s, tag)][k] for s in seeds if (arm, s, tag) in src and src[(arm, s, tag)].get(k) is not None]
    for arm in [x for x in arms if x != "P-all"]:
        for k in ["val_signed_exp", "val_signed_argmax", "neu_abs_col", "neu_acc_col", "neu_acc_row", "acc_insitu_matched", "eval_reward", "H_pi"]:
            try:
                primary[f"{arm}-minus-P-all|post_rl|{k}"] = welch(per_seed(arm, "post_rl", k), per_seed("P-all", "post_rl", k))
            except Exception as e:
                primary[f"{arm}-minus-P-all|post_rl|{k}"] = str(e)
        for k in ["acc_insitu_matched", "val_signed_exp"]:
            da = [unit[(arm, s, "post_rl")][k] - unit[(arm, s, "pre_rl")][k] for s in seeds if (arm, s, "post_rl") in unit and (arm, s, "pre_rl") in unit]
            dp = [unit[("P-all", s, "post_rl")][k] - unit[("P-all", s, "pre_rl")][k] for s in seeds if ("P-all", s, "post_rl") in unit and ("P-all", s, "pre_rl") in unit]
            try:
                primary[f"{arm}-minus-P-all|delta_pre_to_post|{k}"] = welch(da, dp)
            except Exception as e:
                primary[f"{arm}-minus-P-all|delta_pre_to_post|{k}"] = str(e)
    res = {"primary_welch": primary, "entropy_bins": BINS, "entropy_ref_hist": ref.tolist(), "insitu": T_in, "fresh": T_fr,
           "gate0_and_selfadv_bootstrap": boot, "direction": T_dir, "delta_pre_to_post": delta, "matched_reward_ols": ols,
           "arms": arms, "seeds": seeds, "tags": tags}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "tables.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table", "arm|tag", "metric", "mean", "sd", "n"])
        for name, T in (("insitu", T_in), ("fresh", T_fr), ("direction", T_dir)):
            for key, rr in T.items():
                for met, v in rr.items():
                    w.writerow([name, key, met, v["mean"], v["sd"], v["n"]])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for arm in arms:
            xs = [t for t in tags if f"{arm}|{t}" in T_in]
            for i, met in enumerate(["acc_insitu_matched", "H_pi", "val_signed_exp"]):
                mu = [T_in[f"{arm}|{t}"][met]["mean"] for t in xs]; sd = [T_in[f"{arm}|{t}"][met]["sd"] for t in xs]
                ax[i].errorbar(range(len(xs)), mu, yerr=sd, label=arm, capsize=3, marker="o")
                ax[i].set_xticks(range(len(xs))); ax[i].set_xticklabels(xs); ax[i].set_title(met)
        ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "insitu.png"), dpi=110)
    except Exception as e:  # matplotlib optional
        print("no plot:", e)
    print(json.dumps({"gate0": boot, "fresh_selfadv": {k: v.get("self_adv") for k, v in T_fr.items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
