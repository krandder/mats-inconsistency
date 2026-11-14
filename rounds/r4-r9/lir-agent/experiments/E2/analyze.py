"""E2 analysis: reads runs/vm*/<NAME>/{eval.jsonl,steps.jsonl,e2_posthoc.json,status.json} and writes
results.json (per-run rows + per-arm mean/sd + tests) and reward_curves.png. Written before the runs."""
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

E2 = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(E2, "results.json")


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def arm_of(name):
    return re.sub(r"-s\d+$", "", name)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb) if va + vb > 0 else float("inf")
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1)) + (vb ** 2 / (len(b) - 1))) if va + vb > 0 else float("nan")
    p = t_two_sided_p(t, df) if np.isfinite(t) and np.isfinite(df) else None
    return {"diff": float(a.mean() - b.mean()), "t": float(t), "df": float(df), "p_welch": p,
            "p_perm_exact": perm_p(a, b), "n_a": len(a), "n_b": len(b)}


def t_two_sided_p(t, df, n=20001):
    """Two-sided p of Student t (numerical integration of the pdf; no scipy on this box)."""
    x = np.linspace(0, abs(t), n)
    lg = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(lg - (df + 1) / 2 * np.log1p(x ** 2 / df))
    area = float(np.trapezoid(pdf, x)) if hasattr(np, "trapezoid") else float(np.trapz(pdf, x))
    return max(0.0, 1.0 - 2 * area)


def perm_p(a, b):
    """Exact two-sided permutation p for the difference of means (all relabelings)."""
    from itertools import combinations
    z = np.r_[a, b]; n = len(a); obs = abs(a.mean() - b.mean())
    cnt = tot = 0
    for idx in combinations(range(len(z)), n):
        m = np.zeros(len(z), bool); m[list(idx)] = True
        d = abs(z[m].mean() - z[~m].mean()); tot += 1; cnt += d >= obs - 1e-12
    return cnt / tot


EV = ["reward", "env_reward", "success", "syco_note_bias", "syco_agree_rate", "belief_brier", "belief_ece",
      "belief_mae_vs_bayes", "fc_committed_signed_vs_bayes", "fc_committed_signed_vs_realised",
      "fc_nonchosen_signed_vs_bayes", "fc_all_brier", "fc_all_ece", "fc_committed_brier", "fc_committed_ece",
      "fc_all_brier_vs_bayes", "commit_consistent_with_stated", "self_acc"]


def row(d):
    name = os.path.basename(d)
    r = {"name": name, "arm": arm_of(name), "dir": os.path.relpath(d, E2)}
    st = os.path.join(d, "status.json")
    if os.path.exists(st):
        s = json.load(open(st)); r["train_rc"] = s["train_rc"]; r["posthoc_rc"] = s["posthoc_rc"]; r["train_s"] = s["train_s"]
    ev = {e["tag"]: e for e in jl(os.path.join(d, "eval.jsonl"))}
    for tag in ("pre_rl", "rl40", "post_rl"):
        if tag in ev:
            for k in EV:
                for pre in ("main", "syco"):
                    if pre in ev[tag] and k in ev[tag][pre]:
                        r[f"{tag}.{pre}.{k}"] = ev[tag][pre][k]
    for pre in ("main", "syco"):
        for k in EV:
            a, b = r.get(f"pre_rl.{pre}.{k}"), r.get(f"post_rl.{pre}.{k}")
            if a is not None and b is not None:
                r[f"d.{pre}.{k}"] = b - a
    steps = [s for s in jl(os.path.join(d, "steps.jsonl")) if s.get("phase") == "rl"]
    if steps:
        rw = [s["reward"] for s in steps]
        r["rl_reward_first10"] = float(np.mean(rw[:10])); r["rl_reward_last20"] = float(np.mean(rw[-20:]))
        r["reward_curve"] = rw
        g = [s.get("gradloc_bf_param_frac", s.get("gradloc_bf_token_frac")) for s in steps]
        g = [x for x in g if x is not None]
        r["grad_share_mean"] = float(np.mean(g)) if g else None
        if "R_diff_action" in steps[0]:
            us = []
            for s in steps:
                bf = math.hypot(s.get("R_diff_belief", 0.0), s.get("R_diff_forecast", 0.0))
                tot = math.sqrt(bf ** 2 + s.get("R_diff_action", 0.0) ** 2)
                us.append(bf / tot if tot > 0 else 0.0)
            r["update_share_mean"] = float(np.mean(us))
        r["mask_violations_steps"] = sum(1 for s in steps if any(k.endswith("MASK_VIOLATION") for k in s))
    ph = os.path.join(d, "e2_posthoc.json")
    if os.path.exists(ph):
        p = json.load(open(ph))
        if "swap_action_only" in p:
            sw = p["swap_action_only"]
            for fam in ("cell", "self", "succ", "steps", "hits"):
                if fam in sw:
                    r[f"swap.{fam}.max"] = sw[fam]["max_abs_dlogp"]; r[f"swap.{fam}.kl"] = sw[fam]["mean_kl"]
                    r[f"swap.{fam}.identical"] = sw[fam]["bit_identical"]
            r["swap.bf_identical"] = all(sw[f]["bit_identical"] for f in ("cell", "succ", "steps", "hits"))
            if p.get("syco_fixed_swap") is not None:
                r["swap.syco_shift"] = p["syco_fixed_swap"] - p["syco_fixed_warm"]
        for fam in ("cell", "succ", "steps", "hits"):
            r[f"drift.{fam}.kl"] = p["drift_all_final"][fam]["mean_kl"]
        if p.get("syco_fixed_final") is not None:
            r["fixed.dsyco"] = p["syco_fixed_final"] - p["syco_fixed_warm"]
        for t in ("warm", "final"):
            r[f"D_{t}"] = p[f"D_{t}"]["D_signed_vs_bayes"]
            r[f"D_{t}_realised"] = p[f"D_{t}"]["D_signed_vs_realised"]
            r[f"D_{t}_sel_signed_vs_bayes"] = p[f"D_{t}"]["sel"]["signed_vs_bayes"]
            r[f"D_{t}_rand_signed_vs_bayes"] = p[f"D_{t}"]["rand"]["signed_vs_bayes"]
        r["dD"] = r["D_final"] - r["D_warm"]
        fm = p["final_minus_warm_norm"]
        if "belief" in fm:
            bf = math.hypot(fm["belief"], fm["forecast"])
            r["total_update_share_bf"] = bf / math.sqrt(bf ** 2 + fm["action"] ** 2)
    return r


def main():
    allrows = [row(d) for d in sorted(glob.glob(os.path.join(E2, "runs", "vm*", "*"))) if os.path.isdir(d)
               and os.path.exists(os.path.join(d, "eval.jsonl"))]
    rows = [r for r in allrows if r.get("train_rc") == 0 and "post_rl.main.reward" in r]
    failed = [{"dir": r["dir"], "train_rc": r.get("train_rc"), "has_post_rl": "post_rl.main.reward" in r}
              for r in allrows if r not in rows]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    agg = {}
    for arm, rs in sorted(by.items()):
        a = {"n": len(rs), "seeds": [r["name"] for r in rs]}
        keys = sorted({k for r in rs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for k in keys:
            v = [r[k] for r in rs if k in r and r[k] is not None and not isinstance(r[k], bool)]
            if v:
                a[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v),
                        "min": float(np.min(v)), "max": float(np.max(v))}
        for k in ("swap.bf_identical",):
            v = [r[k] for r in rs if k in r]
            if v:
                a[k] = {"all": all(v), "n": len(v)}
        agg[arm] = a
    tests = {}

    def vals(arm, k):
        return [r[k] for r in by.get(arm, []) if r.get(k) is not None]
    for k in ("d.main.syco_note_bias", "d.main.fc_committed_signed_vs_bayes", "d.main.fc_committed_signed_vs_realised",
              "d.main.belief_ece", "d.main.fc_all_ece", "fixed.dsyco", "drift.cell.kl", "drift.succ.kl", "dD",
              "rl_reward_last20"):
        for a, b in (("A0", "A1"), ("A2", "A1"), ("A3", "A1"), ("A0e", "A1e")):
            w = welch(vals(a, k), vals(b, k))
            if w:
                tests[f"{a}-vs-{b}:{k}"] = w
    for k in ("d.syco.syco_note_bias",):
        w = welch(vals("A0e", k), vals("A1e", k))
        if w:
            tests[f"A0e-vs-A1e:{k}"] = w
    # A4 dose slopes (alpha 0 = A1 seeds 0-2)
    dose = []
    for r in rows:
        m = re.match(r"A4a([\d.]+)-s(\d+)", r["name"])
        if m:
            dose.append((float(m.group(1)), r))
        elif r["arm"] == "A1" and r["name"] in ("A1-s0", "A1-s1", "A1-s2"):
            dose.append((0.0, r))
    slopes = {}
    if len(dose) >= 3:
        for k in ("d.main.syco_note_bias", "fixed.dsyco", "d.main.fc_committed_signed_vs_bayes", "drift.cell.kl", "drift.succ.kl"):
            pts = [(al, r[k], r.get("update_share_mean"), r.get("rl_reward_last20")) for al, r in dose if r.get(k) is not None]
            if len(pts) >= 3:
                x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
                b1 = np.polyfit(x, y, 1)[0]
                u = np.array([p[2] if p[2] is not None else 0.0 for p in pts])
                b2 = float(np.polyfit(u, y, 1)[0]) if u.std() > 0 else None
                rr = np.array([p[3] for p in pts], float)
                slopes[k] = {"slope_on_alpha": float(b1), "slope_on_update_share": b2, "n": len(pts),
                             "spearman_alpha": float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]),
                             "reward_last20_by_point": rr.tolist(), "points": [(p[0], p[1], p[2]) for p in pts]}
                # matched-reward check: regress y on alpha and reward jointly
                if len(pts) >= 4 and rr.std() > 0:
                    X = np.c_[np.ones(len(x)), x, rr]
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    slopes[k]["slope_on_alpha_controlling_reward"] = float(coef[1])
    json.dump({"failed_or_incomplete": failed, "rows": rows, "arms": agg, "tests": tests, "dose": slopes}, open(OUT, "w"), indent=1, default=float)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (arm, rs) in enumerate(sorted(by.items())):
            cs = [r["reward_curve"] for r in rs if "reward_curve" in r]
            if not cs:
                continue
            n = min(len(c) for c in cs)
            m = np.mean([np.convolve(c[:n], np.ones(5) / 5, mode="valid") for c in cs], 0)
            ax.plot(m, label=f"{arm} (n={len(cs)})")
        ax.set_xlabel("RL step (5-step moving mean)"); ax.set_ylabel("batch reward"); ax.legend(fontsize=7)
        ax.set_title("E2 reward curves, mean over seeds")
        fig.tight_layout(); fig.savefig(os.path.join(E2, "reward_curves.png"), dpi=110)
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)
    print(json.dumps({a: {k: agg[a][k] for k in ("n", "d.main.syco_note_bias", "rl_reward_last20") if k in agg[a]} for a in agg}, indent=1))


if __name__ == "__main__":
    main()
