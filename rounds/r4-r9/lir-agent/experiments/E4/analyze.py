"""E4 analysis (written before the runs; changes logged in PREREG.md Deviations).
Reads experiments/E4/runs/<preset>/<arm>-s<seed>/{e4_fixed.jsonl,summary.json,steps.jsonl}; writes
results.json, tables.csv, matched.csv and curves PNGs into experiments/E4/."""
import glob, json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("E4_RUNS", os.path.join(D, "runs"))
D = os.environ.get("E4_OUT", D)
ARMS = ["B0", "M0", "M1", "P", "twin"]
W_AGREE = W_CONF = 0.5


def load():
    runs = {}
    for f in glob.glob(f"{R}/*/*-s*/e4_fixed.jsonl"):
        preset = f.split("/")[-3]; name = f.split("/")[-2]
        arm, seed = name.rsplit("-s", 1)
        evs = [json.loads(l) for l in open(f)]
        sm = os.path.join(os.path.dirname(f), "summary.json")
        summ = json.load(open(sm)) if os.path.exists(sm) else None
        runs[(preset, arm, int(seed))] = dict(evs=evs, summary=summ, path=os.path.relpath(f, D))
    return runs


def ms(x):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"), "n": int(len(x))}


KEYS = ["op_env_reward", "op_reward", "kl_init_total", "fx_fc_brier", "fx_fc_ece", "fx_fc_signed_vs_realised", "fx_fc_signed_vs_bayes",
        "fx_fc_brier_vs_bayes", "fx_sit_brier", "fx_sit_ece", "fx_sit_signed_vs_bayes", "fx_sit_mae_vs_bayes",
        "fx_fc_note_bias_index", "fx_sit_note_bias_index", "fx_sit_agree_rate", "fx_pol_note_bias_index",
        "fx_fc_notefirst_signed_vs_bayes", "commit_note_bias_first", "commit_note_bias_inplan", "op_syco_note_bias",
        "rel_swap_flip", "rel_swap_flip_given_implied", "rel_top_down03_flip", "rel_top_down03_flip_given_implied",
        "rel_second_up03_flip", "rel_second_up03_flip_given_implied", "commit_follows_stated_argmax",
        "op_reliance_swap_argmax_change", "dfc_inconsistent_argmax", "dfc_inconsistent_sampled", "agree_rate", "conf_rate",
        "op_fc_committed_brier", "op_fc_committed_signed_vs_realised", "op_fc_all_brier", "op_fc_all_signed_vs_realised",
        "op_fc_all_signed_vs_bayes", "op_belief_brier"]


def per_run(rr):
    evs = rr["evs"]
    pre = next((e for e in evs if e["tag"] == "pre_rl"), None)
    post = next((e for e in evs if e["tag"] == "post_rl"), None) or evs[-1]
    out = {"last_tag": post["tag"], "last_step": post["step"]}
    for k in KEYS:
        if pre is not None and k in pre and k in post:
            out[k + "_pre"], out[k + "_post"], out[k + "_delta"] = pre[k], post[k], post[k] - pre[k]
    # post hoc (logged 11:15Z after seeing seed 1): mean over all RL checkpoints minus pre, robust to forecaster oscillation
    rl = [e for e in evs if e["step"] > 0]
    if pre is not None and rl:
        for k in KEYS:
            if k in pre and all(k in e for e in rl):
                out[k + "_cmean_delta"] = float(np.mean([e[k] for e in rl])) - pre[k]
    # approval split (reward gain decomposition, w_agree / w_conf / env)
    if pre is not None:
        out["gain_agree"] = W_AGREE * (post["agree_rate"] - pre["agree_rate"])
        out["gain_conf"] = W_CONF * (post["conf_rate"] - pre["conf_rate"])
        out["gain_env"] = post["op_env_reward"] - pre["op_env_reward"]
        tot = out["gain_agree"] + out["gain_conf"] + out["gain_env"]
        out["gain_total"] = tot
        # selection channel: on-policy drift minus fixed-set drift
        for a, b in (("op_fc_committed_signed_vs_realised", "fx_fc_signed_vs_realised"), ("op_fc_committed_brier", "fx_fc_brier"),
                     ("op_fc_all_signed_vs_bayes", "fx_fc_signed_vs_bayes")):
            if a in pre and b in pre:
                out[f"sel_{a}_minus_{b}"] = (post[a] - pre[a]) - (post[b] - pre[b])
    return out


def main():
    runs = load()
    per = {f"{p}|{a}|{s}": dict(per_run(rr), path=rr["path"],
                               mask_violations=(rr["summary"] or {}).get("mask_violations"))
           for (p, a, s), rr in runs.items()}
    res = {"per_run": per, "agg": {}, "data_channel": {}, "matched": {}}
    presets = sorted({k.split("|")[0] for k in per})
    for p in presets:
        for a in ARMS:
            rows = [v for k, v in per.items() if k.startswith(f"{p}|{a}|") and v.get("mask_violations") in (0, None) and v.get("last_step", 0) > 0]
            if not rows:
                continue
            keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
            res["agg"][f"{p}|{a}"] = {k: ms([r.get(k) for r in rows]) for k in keys} | {"seeds": len(rows)}
        # data channel: fixed-set drift minus the no-RL twin's drift, paired by seed, at the last eval step both reached
        twr = {s: rr for (pp, a, s), rr in runs.items() if pp == p and a == "twin"}
        for a in ARMS[:-1]:
            for key in ("fx_fc_signed_vs_bayes", "fx_fc_brier", "fx_fc_ece", "fx_fc_signed_vs_realised", "fx_sit_brier",
                        "fx_sit_signed_vs_bayes", "fx_fc_note_bias_index", "fx_sit_note_bias_index"):
                d, steps = [], []
                for (pp, aa, s), rr in runs.items():
                    if pp != p or aa != a or s not in twr:
                        continue
                    ea = {e["step"]: e for e in rr["evs"]}; et = {e["step"]: e for e in twr[s]["evs"]}
                    common = sorted(set(ea) & set(et))
                    if len(common) < 2 or key not in ea[0] or key not in et[0]:
                        continue
                    st = common[-1]
                    d.append((ea[st][key] - ea[0][key]) - (et[st][key] - et[0][key])); steps.append(st)
                if d:
                    res["data_channel"][f"{p}|{a}|{key}"] = (ms(d) or {}) | {"at_steps": steps}
                # post hoc: curve-mean version over the common RL steps
                dc = []
                for (pp, aa, s), rr in runs.items():
                    if pp != p or aa != a or s not in twr:
                        continue
                    ea = {e["step"]: e for e in rr["evs"]}; et = {e["step"]: e for e in twr[s]["evs"]}
                    common = [c for c in sorted(set(ea) & set(et)) if c > 0]
                    if common and 0 in ea and 0 in et and key in ea[0]:
                        dc.append(np.mean([ea[c][key] - ea[0][key] - (et[c][key] - et[0][key]) for c in common]))
                if dc:
                    res["data_channel"][f"{p}|{a}|{key}|curve_mean"] = ms(dc)
        # matched env-reward bins over every checkpoint (pooled across arms)
        pts = []
        for (pp, a, s), rr in runs.items():
            if pp != p:
                continue
            for e in rr["evs"]:
                pts.append((a, s, e))
        if pts:
            er = np.array([e["op_env_reward"] for _, _, e in pts])
            edges = np.quantile(er, [0, 1 / 3, 2 / 3, 1])
            for bi in range(3):
                lo, hi = edges[bi], edges[bi + 1]
                for a in ARMS:
                    sel = [e for aa, _, e in pts if aa == a and (lo <= e["op_env_reward"] <= hi)]
                    if sel:
                        res["matched"][f"{p}|bin{bi}[{lo:.3f},{hi:.3f}]|{a}"] = {
                            k: ms([e.get(k) for e in sel]) for k in ("fx_fc_signed_vs_bayes", "fx_fc_brier", "fx_sit_note_bias_index",
                                                                   "fx_fc_note_bias_index", "kl_init_total", "agree_rate", "conf_rate",
                                                                   "rel_swap_flip", "dfc_inconsistent_argmax")} | {"n_ckpts": len(sel)}
    json.dump(res, open(os.path.join(D, "results.json"), "w"), indent=1)
    # flat CSV
    with open(os.path.join(D, "tables.csv"), "w") as f:
        f.write("preset,arm,metric,mean,sd,n\n")
        for pa, d in res["agg"].items():
            p, a = pa.split("|")
            for k, v in d.items():
                if isinstance(v, dict) and v:
                    f.write(f"{p},{a},{k},{v['mean']:.5g},{v['sd']:.5g},{v['n']}\n")
    # curves
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col = {"B0": "#c0392b", "M0": "#e67e22", "M1": "#8e44ad", "P": "#2471a3", "twin": "#7f8c8d"}
        for p in presets:
            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            specs = [("kl_init_total", "op_env_reward"), ("kl_init_total", "fx_fc_signed_vs_bayes"), ("kl_init_total", "fx_fc_brier"),
                     ("op_env_reward", "fx_sit_note_bias_index"), ("op_env_reward", "agree_rate"), ("step", "rel_swap_flip")]
            for axx, (xk, yk) in zip(ax.ravel(), specs):
                for (pp, a, s), rr in sorted(runs.items()):
                    if pp != p:
                        continue
                    xs = [e.get(xk) for e in rr["evs"]]; ys = [e.get(yk) for e in rr["evs"]]
                    axx.plot(xs, ys, "-o", ms=3, color=col.get(a, "k"), alpha=0.7, label=a if s == 0 else None)
                axx.set_xlabel(xk); axx.set_ylabel(yk)
            ax[0, 0].legend()
            fig.suptitle(f"E4 {p}: every checkpoint, every seed")
            fig.tight_layout(); fig.savefig(os.path.join(D, f"curves_{p}.png"), dpi=90); plt.close(fig)
    except Exception as ex:
        print("plot failed:", ex)
    print(json.dumps({k: {m: v for m, v in d.items() if m in ("op_env_reward_delta", "fx_fc_signed_vs_bayes_delta", "seeds")}
                      for k, d in res["agg"].items()}, indent=1))


if __name__ == "__main__":
    main()
