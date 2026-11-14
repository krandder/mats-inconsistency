"""E6 analysis (written before the runs). Reads runs/<arm>-s<seed>/e6eval.jsonl for X in {c,d} and
b_X, plus bank.json. Writes results.json and results.csv next to this file.

Definitions (bank groups = boards = GRPO-style groups: same start state, different plans):
  gold_acc   : fraction of non-tied (u_i != u_j) same-board pairs ordered correctly by the proxy
  tie_frac   : fraction of same-board pairs with u_i == u_j
  tie_reorder: fraction of tied pairs whose proxy order sign differs from step 0 (free drift)
  SSI_t      : over tied pairs (i,j) where X's current policy makes i at least 2x more likely than j
               (support_logp_i - support_logp_j > log 2): mean(s_t(i) - s_t(j)) - mean(s_0(i) - s_0(j)),
               same pairs, support from X's policy at step t (shared by X and b_X; arm a uses X's support too)
  fam_t      : same contrast on the agent's log-likelihood of the outcome text (familiarity covariate)
  SSI_resid  : intercept of OLS (s_t - s_0 contrast) ~ (ll_t - ll_0 contrast) over the support pairs
  norm       : ||theta_t - theta_warm|| of the proxy-feature adapter (forecast for roles, fused for B0)
"""
import json, os, sys, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
bank = json.load(open(os.path.join(HERE, "bank.json")))["bank"]
u = np.array([it["u"] for it in bank]); board = np.array([it["board"] for it in bank])
I, J = np.triu_indices(len(bank), 1)
same = board[I] == board[J]
I, J = I[same], J[same]
tied = u[I] == u[J]
nt_I, nt_J = I[~tied], J[~tied]
t_I, t_J = I[tied], J[tied]
LOG2 = math.log(2)


def load(path):
    return {r["step"]: r for r in map(json.loads, open(path))}


def metrics(ev, sup_ev, key="score"):
    steps = sorted(ev)
    s0 = np.array(ev[0][key]); l0 = np.array(ev[0]["ll"])
    out = []
    for t in steps:
        if t not in sup_ev:
            continue
        s = np.array(ev[t][key]); ll = np.array(ev[t]["ll"])
        sp = np.array(sup_ev[t]["support_logp"])
        acc = float(np.mean(np.sign(s[nt_I] - s[nt_J]) == np.sign(u[nt_I] - u[nt_J])))
        reo = float(np.mean(np.sign(s[t_I] - s[t_J]) != np.sign(s0[t_I] - s0[t_J])))
        d = sp[t_I] - sp[t_J]
        m = np.abs(d) > LOG2
        a = np.where(d[m] > 0, t_I[m], t_J[m]); b = np.where(d[m] > 0, t_J[m], t_I[m])   # a in, b out
        ds = (s[a] - s[b]) - (s0[a] - s0[b])
        dl = (ll[a] - ll[b]) - (l0[a] - l0[b]) if key == "score" else np.zeros_like(ds)
        ssi = float(ds.mean()) if len(ds) else float("nan")
        fam = float(dl.mean()) if len(dl) else float("nan")
        resid = ssi
        if key == "score" and len(ds) > 3 and dl.std() > 0:
            X = np.stack([np.ones_like(dl), dl], 1)
            resid = float(np.linalg.lstsq(X, ds, rcond=None)[0][0])
        nm = ev[t]["norms"]
        norm = nm.get("fused", nm.get("forecast"))
        out.append(dict(step=t, gold_acc=acc, tie_reorder=reo, ssi=ssi, fam=fam, ssi_resid_fam=resid,
                        n_support_pairs=int(m.sum()), norm=norm, norm_belief=nm.get("belief"),
                        norm_action=nm.get("action")))
    return out


def interp_at(rows, norm, key):
    xs = [r["norm"] for r in rows]; ys = [r[key] for r in rows]
    if norm > max(xs):
        return float("nan")
    return float(np.interp(norm, xs, ys))


def main():
    res = {"tie_frac": float(tied.mean()), "n_pairs": int(len(I)), "n_nontied": int((~tied).sum()),
           "n_tied": int(tied.sum()), "bank_items": len(bank), "arms": {}}
    seeds = sorted({int(p.split("-s")[-1]) for p in glob.glob(os.path.join(RUNS, "c-s*")) +
                    glob.glob(os.path.join(RUNS, "d-s*"))})
    rows_csv = []
    for X in ("c", "d"):
        for s in seeds:
            px, pb = os.path.join(RUNS, f"{X}-s{s}", "e6eval.jsonl"), os.path.join(RUNS, f"b{X}-s{s}", "e6eval.jsonl")
            if not os.path.exists(px):
                continue
            ex = load(px)
            arms = {X: metrics(ex, ex), f"a_{X}": metrics(ex, ex, key="a_score")}
            if os.path.exists(pb):
                arms[f"b_{X}"] = metrics(load(pb), ex)
            for name, rows in arms.items():
                res["arms"].setdefault(name, {})[s] = rows
                for r in rows:
                    rows_csv.append(dict(arm=name, seed=s, **r))
    # summaries: final step, and X interpolated at b_X's final norm (matched update norm)
    summ = {}
    for name, per in res["arms"].items():
        fin = [rows[-1] for rows in per.values()]
        summ[name] = {k: [float(np.mean([f[k] for f in fin])), float(np.std([f[k] for f in fin], ddof=1)) if len(fin) > 1 else float("nan")]
                      for k in ("gold_acc", "tie_reorder", "ssi", "fam", "ssi_resid_fam", "norm")}
        summ[name]["init_gold_acc"] = float(np.mean([rows[0]["gold_acc"] for rows in per.values()]))
        summ[name]["n_seeds"] = len(fin)
    diffs = {}
    for X in ("c", "d"):
        if X not in res["arms"] or f"b_{X}" not in res["arms"]:
            continue
        dd = {"final": [], "matched_norm": [], "final_resid": [], "gold_final": [], "gold_matched": []}
        for s, rows in res["arms"][X].items():
            if s not in res["arms"][f"b_{X}"]:
                continue
            rb = res["arms"][f"b_{X}"][s]
            common = sorted({r["step"] for r in rows} & {r["step"] for r in rb})
            rows = [r for r in rows if r["step"] in common]; rb = [r for r in rb if r["step"] in common]
            dd["final"].append(rows[-1]["ssi"] - rb[-1]["ssi"])
            dd["final_resid"].append(rows[-1]["ssi_resid_fam"] - rb[-1]["ssi_resid_fam"])
            dd["gold_final"].append(rows[-1]["gold_acc"] - rb[-1]["gold_acc"])
            nb = rb[-1]["norm"]
            dd["matched_norm"].append(interp_at(rows, nb, "ssi") - rb[-1]["ssi"])
            dd["gold_matched"].append(interp_at(rows, nb, "gold_acc") - rb[-1]["gold_acc"])
        diffs[f"{X}-b_{X}"] = {k: dict(per_seed=v, mean=float(np.mean(v)) if v else None,
                                       sd=float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in dd.items()}
    # tie fraction inside the actual GRPO groups of the logged streams (power of rank metrics in training)
    import pickle
    tf = []
    for sd in glob.glob(os.path.join(RUNS, "[cd]-s*", "stream")):
        for f in sorted(glob.glob(os.path.join(sd, "*.pkl"))):
            recs = pickle.load(open(f, "rb"))
            by = {}
            for r in recs:
                by.setdefault(r["group"], []).append(round(r["env_reward"], 3))
            for v in by.values():
                v = np.array(v); a, b = np.triu_indices(len(v), 1)
                tf += list(v[a] == v[b])
    res["stream_group_tie_frac"] = float(np.mean(tf)) if tf else None
    res["summary_final"] = summ
    res["reward_caused"] = diffs
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    import csv
    if rows_csv:
        with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    print(json.dumps({"tie_frac": res["tie_frac"], "stream_group_tie_frac": res["stream_group_tie_frac"], "summary_final": summ, "reward_caused": diffs}, indent=1))


if __name__ == "__main__":
    main()
