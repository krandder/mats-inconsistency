"""E1 aggregation (written before the runs; PREREG.md). Reads experiments/E1/runs/<arm>-<chan>-s<seed>/
{summary.json, eval.jsonl, steps.jsonl, e1_metrics.json}; writes experiments/E1/results.json with
mean, sd, n over seeds per (arm, channel, tag, metric), step-level hook aggregates, and the PREREG verdict."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("E1_RUNS", os.path.join(HERE, "runs"))
NAME = re.compile(r"^(P|D|F|B0floor|R|Pall)-(on|off)-s(\d+)$")
TAGS = ["pre_rl", "rl30", "post_rl"]
PRIMARY = ["fc_committed_ece_tok", "fc_committed_opt_vs_real_stated", "agree_false_note_stated"]
E1KEYS = PRIMARY + ["reward", "env_reward", "success", "fc_committed_ece_stated", "fc_committed_brier_tok",
                    "fc_committed_brier_stated", "fc_committed_opt_vs_real_tok", "fc_committed_opt_vs_bayes_tok",
                    "fc_committed_conf07_stated", "fc_nonchosen_ece_tok", "fc_nonchosen_ece_stated",
                    "fc_nonchosen_brier_tok", "fc_nonchosen_opt_vs_real_tok", "fc_random_opt_vs_real_tok",
                    "fc_random_opt_vs_real_stated", "fc_random_opt_vs_bayes_tok", "fc_random_ece_tok",
                    "selection_gap_tok", "sit_logscore_vs_state", "sit_kl_vs_bayes", "sit_logscore_bayes_floor",
                    "sit_ece_tok", "sit_ece_stated", "agree_false_note_tok", "agree_bayes_false_note",
                    "note_bias_vs_bayes", "note_bias_vs_bayes_false"]
EVKEYS = ["coherence_violation_rate", "martingale_mean_revision", "martingale_final_gap", "commit_consistent_with_stated",
          "self_acc", "reliance_swap_tv", "regret_vs_best_candidate_bayes", "wishful_plan_cells", "syco_agree_rate"]
STEPKEYS = ["reward", "env_reward", "R_grad_belief", "R_grad_forecast", "cos_reward_vs_B", "cos_reward_vs_F",
            "e1_R_dbel_abs", "e1_R_dbel_note", "e1_R_dbel_note_false", "e1_R_dfc_abs", "e1_R_dfc_signed",
            "e1_lag_brier_pre", "e1_lag_brier_post", "e1_lag_signed_pre", "e1_lag_signed_post",
            "gradloc_bf_token_frac", "gradloc_bf_param_frac", "fc_committed_brier", "fc_committed_signed_vs_realised",
            "syco_note_bias", "wishful_plan_cells", "belief_ece", "fc_nonchosen_signed_vs_bayes", "commit_consistent_with_stated"]


def ms(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None, "n": len(x),
            "vals": [float(v) for v in x]}


def load():
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        m = NAME.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        arm, ch, s = m.group(1), m.group(2), int(m.group(3))
        summ = json.load(open(os.path.join(d, "summary.json")))
        if summ.get("mask_violations", 1) != 0:
            print("EXCLUDED (mask violations)", d)
            continue
        e1 = json.load(open(os.path.join(d, "e1_metrics.json"))) if os.path.exists(os.path.join(d, "e1_metrics.json")) else {}
        ev = {}
        for line in open(os.path.join(d, "eval.jsonl")):
            r = json.loads(line)
            ev[r["tag"]] = r.get("main", {})
        steps = [json.loads(l) for l in open(os.path.join(d, "steps.jsonl"))]
        steps = [x for x in steps if x.get("phase") == "rl"]
        runs[(arm, ch, s)] = dict(summ=summ, e1=e1, ev=ev, steps=steps, dir=os.path.relpath(d, HERE))
    return runs


def aggregate(runs):
    agg = defaultdict(dict)
    groups = defaultdict(list)
    for (arm, ch, s), r in runs.items():
        groups[(arm, ch)].append(r)
    for (arm, ch), rs in groups.items():
        key = f"{arm}-{ch}"
        a = {"seeds": sorted(int(r["dir"].rsplit("s", 1)[1]) for r in rs), "dirs": [r["dir"] for r in rs]}
        for tag in TAGS:
            t = {}
            for k in E1KEYS:
                t[k] = ms([r["e1"].get(tag, {}).get("main", {}).get(k) for r in rs])
            for k in EVKEYS:
                t["ev_" + k] = ms([r["ev"].get(tag, {}).get(k) for r in rs])
            a[tag] = t
        # change over RL (post - pre), per seed
        a["delta"] = {k: ms([r["e1"].get("post_rl", {}).get("main", {}).get(k, np.nan) -
                             r["e1"].get("pre_rl", {}).get("main", {}).get(k, np.nan) for r in rs]) for k in E1KEYS}
        st = {}
        for k in STEPKEYS:
            per_seed_mean = [np.mean([x[k] for x in r["steps"] if k in x]) if any(k in x for x in r["steps"]) else None for r in rs]
            st[k] = ms(per_seed_mean)
            if k.startswith("R_grad") or k.startswith("e1_R_"):
                st[k + "_maxabs"] = ms([max(abs(x[k]) for x in r["steps"] if k in x) if any(k in x for x in r["steps"]) else None for r in rs])
        st["lag"] = ms([np.mean([x["e1_lag_brier_pre"] - x["e1_lag_brier_post"] for x in r["steps"] if "e1_lag_brier_pre" in x])
                        if any("e1_lag_brier_pre" in x for x in r["steps"]) else None for r in rs])
        st["syco_note_bias_max"] = ms([max(x["syco_note_bias"] for x in r["steps"] if "syco_note_bias" in x) for r in rs if r["steps"]])
        st["reward_first10"] = ms([np.mean([x["reward"] for x in r["steps"][:10]]) for r in rs if r["steps"]])
        st["reward_last10"] = ms([np.mean([x["reward"] for x in r["steps"][-10:]]) for r in rs if r["steps"]])
        st["env_reward_last10"] = ms([np.mean([x["env_reward"] for x in r["steps"][-10:]]) for r in rs if r["steps"]])
        a["steps"] = st
        # prequential trajectories (each RL batch scored before its update), 10-step windows, mean +- sd over seeds
        tr = {}
        for k in ("syco_note_bias", "fc_committed_signed_vs_realised", "reward", "env_reward", "belief_ece", "e1_R_dbel_note"):
            w = []
            for lo in range(0, 60, 10):
                w.append(ms([np.mean([x[k] for x in r["steps"][lo:lo + 10] if k in x]) for r in rs
                             if any(k in x for x in r["steps"][lo:lo + 10])]))
            tr[k] = w
        a["traj10"] = tr
        a["mask_violations"] = [r["summ"]["mask_violations"] for r in rs]
        a["sec_per_rl_step"] = ms([r["summ"]["sec_per_rl_step"] for r in rs])
        agg[key] = a
    return agg


def d_effect(a, b):
    """Cohen d of mean(a) - mean(b) with pooled seed sd."""
    if not a or not b or a.get("n", 0) < 2 or b.get("n", 0) < 2:
        return None
    sp = np.sqrt(((a["n"] - 1) * a["sd"] ** 2 + (b["n"] - 1) * b["sd"] ** 2) / (a["n"] + b["n"] - 2))
    diff = a["mean"] - b["mean"]
    if sp > 0:
        return float(diff / sp)
    return 0.0 if diff == 0 else float(np.sign(diff) * 1e9)


def badness(metric, v):
    # ECE and false-note agreement: lower is better; optimism: |v| (positive is the failure, but distance from 0 counts)
    return abs(v) if "opt" in metric else v


def beats(agg, x, y, ch, metric, tag="post_rl"):
    A, B = agg.get(f"{x}-{ch}", {}).get(tag, {}).get(metric), agg.get(f"{y}-{ch}", {}).get(tag, {}).get(metric)
    if not A or not B or A["n"] < 2 or B["n"] < 2:
        return None
    ba = {"mean": float(np.mean([badness(metric, v) for v in A["vals"]])), "sd": float(np.std([badness(metric, v) for v in A["vals"]], ddof=1)), "n": A["n"]}
    bb = {"mean": float(np.mean([badness(metric, v) for v in B["vals"]])), "sd": float(np.std([badness(metric, v) for v in B["vals"]], ddof=1)), "n": B["n"]}
    d = d_effect(bb, ba)    # positive = x better
    return {"x_bad": ba["mean"], "y_bad": bb["mean"], "gap_y_minus_x": bb["mean"] - ba["mean"], "d": d,
            "x_beats_y": bool(d is not None and d > 1)}


def verdict(agg):
    out = {"comparisons": {}}
    for ch in ("on", "off"):
        for y in ("D", "F", "B0floor", "Pall"):
            for m in PRIMARY:
                out["comparisons"][f"P_vs_{y}_{ch}_{m}"] = beats(agg, "P", y, ch, m)
            er = d_effect(agg.get(f"P-{ch}", {}).get("post_rl", {}).get("env_reward"), agg.get(f"{y}-{ch}", {}).get("post_rl", {}).get("env_reward"))
            out["comparisons"][f"P_vs_{y}_{ch}_env_reward_d"] = er
    c = out["comparisons"]
    on_all = all((c.get(f"P_vs_{y}_on_{m}") or {}).get("x_beats_y") for y in ("D", "F") for m in PRIMARY)
    env_eq = all(c.get(f"P_vs_{y}_on_env_reward_d") is not None and abs(c[f"P_vs_{y}_on_env_reward_d"]) <= 1 for y in ("D", "F"))
    half = []
    for m in PRIMARY[:2]:
        gon, goff = c.get(f"P_vs_D_on_{m}"), c.get(f"P_vs_D_off_{m}")
        if gon and goff:
            half.append(abs(goff["gap_y_minus_x"]) <= 0.5 * abs(gon["gap_y_minus_x"]))
    null_D = all((c.get(f"P_vs_D_on_{m}") or {}).get("d") is not None and abs(c[f"P_vs_D_on_{m}"]["d"]) < 0.5 for m in PRIMARY)
    if on_all and env_eq and half and all(half):
        v = "supported"
    elif null_D:
        v = "refuted"
    else:
        v = "mixed"
    out.update(on_all_beat=on_all, env_equal=env_eq, off_half_of_on=half, P_equals_D_on=null_D, verdict=v)
    # LIR signature: fused distortion ON minus OFF (post - pre deltas), and vs R floor
    sig = {}
    for arm in ("F", "D", "P"):
        for m in ("fc_committed_opt_vs_real_stated", "agree_false_note_stated", "fc_committed_ece_tok",
                  "fc_random_opt_vs_real_stated", "note_bias_vs_bayes"):
            on = agg.get(f"{arm}-on", {}).get("delta", {}).get(m, {})
            off = agg.get(f"{arm}-off", {}).get("delta", {}).get(m, {})
            r = agg.get("R-on", {}).get("delta", {}).get(m, {})
            sig[f"{arm}_{m}"] = {"delta_on": on.get("mean"), "delta_on_sd": on.get("sd"), "delta_off": off.get("mean"),
                                 "delta_off_sd": off.get("sd"), "on_minus_off": (on["mean"] - off["mean"]) if on.get("mean") is not None and off.get("mean") is not None else None,
                                 "delta_R": r.get("mean"), "delta_R_sd": r.get("sd")}
    out["lir_signature"] = sig
    return out


def main():
    runs = load()
    agg = aggregate(runs)
    res = {"n_runs": len(runs), "arms": agg, "verdict": verdict(agg)}
    with open(os.environ.get("E1_RESULTS", os.path.join(HERE, "results.json")), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("runs", len(runs), "verdict", res["verdict"]["verdict"])
    return res


if __name__ == "__main__":
    main()
