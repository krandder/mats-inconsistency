"""Aggregate E3 over seeds: reads runs/s*/ARM/{e3_metrics.json,steps.jsonl,summary.json}; writes
results/e3_tables.json, results/e3_table.csv and two PNGs. Written before the seed 1-2 results.
Reference null for P/RAPc/RAPf = P's c000:c000 pair (shared warm start); M0g/B0s use their own c000:c000."""
import glob, json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("E3_RUNS", os.path.join(D, "runs")); RES = os.environ.get("E3_RES", os.path.join(D, "results"))
ARMS = ["P", "RAPc", "RAPf", "M0g", "B0s"]
CK = ["c025", "c050", "c100"]
NK = [1, 4, 16, 64]


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan"), "n_seeds": len(xs),
            "per_seed": [float(x) for x in xs]}


def load():
    M = {}
    for sd in sorted(glob.glob(os.path.join(RUNS, "s*"))):
        s = os.path.basename(sd)
        for arm in ARMS:
            p = os.path.join(sd, arm, "e3_metrics.json")
            if os.path.exists(p):
                M.setdefault(arm, {})[s] = json.load(open(p))
    return M


def pick(m, pair, key, n):
    pr = m["pairs"].get(pair)
    if pr is None:
        return None
    return pr[key][pr["ns"].index(n)]


def steps_summary(path):
    rl = [json.loads(l) for l in open(path) if '"phase": "rl"' in l]
    if len(rl) < 4:
        return None
    k = max(1, min(10, len(rl) // 4))
    out = {}
    for key in ["reward", "env_reward", "success", "fc_committed_signed_vs_realised", "fc_committed_brier",
                "fc_all_brier_vs_bayes", "fc_nonchosen_signed_vs_bayes", "syco_note_bias", "regret_vs_best_candidate_bayes"]:
        a = [r[key] for r in rl[:k] if key in r]; b = [r[key] for r in rl[-k:] if key in r]
        if a and b:
            out[key] = {"first": float(np.mean(a)), "last": float(np.mean(b)), "delta": float(np.mean(b) - np.mean(a))}
    out["rl_steps"] = len(rl)
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    M = load()
    T = {"arms": {}, "seeds": sorted({s for a in M.values() for s in a})}
    rows = []
    for arm in ARMS:
        if arm not in M:
            continue
        A = {}
        for s, m in M[arm].items():
            ref = M["P"].get(s) if arm in ("RAPc", "RAPf") and "P" in M else m
            for n in NK:
                for key in ["gold", "oracle", "gold_deficit", "gap", "excess_gap", "sw_gap"]:
                    r0 = pick(ref, "c000:c000", key, n) if ref else None
                    A.setdefault(f"c000:c000/{key}/n{n}", []).append(r0)
                    for c in CK:
                        for f in ([c] if arm == "RAPf" else [c, "c000"]):
                            v = pick(m, f"{c}:{f}", key, n)
                            A.setdefault(f"{c}:{f}/{key}/n{n}", []).append(v)
                            if v is not None and r0 is not None:
                                A.setdefault(f"{c}:{f}/delta_{key}/n{n}", []).append(v - r0)
                for c in CK:
                    pr_ = f"{c}:{c}" if arm == "RAPf" else f"{c}:c000"
                    v = pick(m, pr_, "excess_ref_gap", n)
                    A.setdefault(f"{pr_}/excess_ref_gap/n{n}", []).append(v)
                    r0x = pick(ref, "c000:c000", "excess_gap", n) if ref else None
                    if v is not None and r0x is not None:
                        A.setdefault(f"{pr_}/delta_excess_ref_vs_c000_excess/n{n}", []).append(v - r0x)
            for pair, pr in m["pairs"].items():
                A.setdefault(f"{pair}/se_excess_gap_n64", []).append(pr["excess_gap_se"][63])
                ph = pr.get("posthoc_all_states")
                if ph:
                    for n in NK:
                        A.setdefault(f"{pair}/posthoc_excess_xfit/n{n}", []).append(ph["excess_gap_xfit"][n - 1][0])
                        A.setdefault(f"{pair}/posthoc_excess_xfit_se/n{n}", []).append(ph["excess_gap_xfit"][n - 1][1])
                        A.setdefault(f"{pair}/posthoc_gold_all/n{n}", []).append(ph["gold_all"][n - 1][0])
                        A.setdefault(f"{pair}/posthoc_gap_all/n{n}", []).append(ph["gap_all"][n - 1][0])
                A.setdefault(f"{pair}/distinct_sets_per64", []).append(pr["distinct_sets_per64_mean"])
                A.setdefault(f"{pair}/n1_realised_minus_bayes", []).append(pr["n1_realised_mean"] - pr["n1_bayes_mean"])
                A.setdefault(f"{pair}/forecast_minus_bayes_all", []).append(pr["n1_forecast_mean"] - pr["n1_bayes_mean"])
                A.setdefault(f"{pair}/n64_realised_of_argmax_minus_gold64", []).append(pr["n64_realised_of_argmax"] - pr["gold"][63])
                for n in ("1", "4", "16", "64"):
                    A.setdefault(f"{pair}/itp_gold/n{n}", []).append(pr["itp"]["test"][n]["gold"][0])
                    A.setdefault(f"{pair}/itp_gap/n{n}", []).append(pr["itp"]["test"][n]["gap"][0])
                for st in ("low", "mid", "high"):
                    A.setdefault(f"{pair}/entropy_{st}/gap16", []).append(pr["entropy_strata"][st]["gap16"][0])
                    A.setdefault(f"{pair}/entropy_{st}/excess16", []).append(pr["entropy_strata"][st]["excess16"][0])
                    A.setdefault(f"{pair}/entropy_{st}/gold16", []).append(pr["entropy_strata"][st]["gold16"][0])
                for n in ("4", "16"):
                    A.setdefault(f"{pair}/dedup_gold/n{n}", []).append(pr["dedup"][n]["gold"][0])
                    A.setdefault(f"{pair}/dedup_gap/n{n}", []).append(pr["dedup"][n]["gap"][0])
            for f, pb in m["probes"].items():
                for key in ["brier_vs_bayes", "brier_realised", "ece_realised", "ece_vs_bayes", "corr_f_bayes", "mean_f", "mean_bayes"]:
                    A.setdefault(f"probe_{f}/{key}", []).append(pb[key])
            st = os.path.join(RUNS, s, arm, "steps.jsonl")
            ss = steps_summary(st) if os.path.exists(st) else None
            if ss:
                for k, v in ss.items():
                    if isinstance(v, dict):
                        A.setdefault(f"train/{k}/first", []).append(v["first"])
                        A.setdefault(f"train/{k}/last", []).append(v["last"])
                        A.setdefault(f"train/{k}/delta", []).append(v["delta"])
                    else:
                        A.setdefault(f"train/{k}", []).append(v)
            sm = os.path.join(RUNS, s, arm, "summary.json")
            if os.path.exists(sm):
                j = json.load(open(sm))
                A.setdefault("mask_violations", []).append(j["mask_violations"])
                A.setdefault("rl_steps_done", []).append(j["rl_steps"])
        T["arms"][arm] = {k: ms(v) for k, v in A.items()}
        for k, v in T["arms"][arm].items():
            if v:
                rows.append(f'{arm},{k},{v["mean"]:.5f},{v["sd"]:.5f},{v["n_seeds"]},' + ";".join(f"{x:.5f}" for x in v["per_seed"]))
    json.dump(T, open(os.path.join(RES, "e3_tables.json"), "w"), indent=1)
    open(os.path.join(RES, "e3_table.csv"), "w").write("arm,metric,mean,sd,n_seeds,per_seed\n" + "\n".join(rows) + "\n")
    plots(M)
    print("arms", list(T["arms"]), "seeds", T["seeds"])


def plots(M):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("no matplotlib", e)
        return
    fig, ax = plt.subplots(2, len(ARMS), figsize=(4 * len(ARMS), 7), sharex=True)
    cols = {"c000": "#555", "c025": "#9ecae1", "c050": "#4292c6", "c100": "#08306b"}
    for j, arm in enumerate(ARMS):
        if arm not in M:
            continue
        for c in ["c000"] + CK:
            pair = f"{c}:{c}" if not (arm in ("RAPc", "RAPf") and c == "c000") else None
            src = M[arm] if pair else M.get("P", {})
            pair = pair or "c000:c000"
            G = [m["pairs"][pair]["gold"] for m in src.values() if pair in m["pairs"]]
            O = [m["pairs"][pair]["oracle"] for m in src.values() if pair in m["pairs"]]
            X = [m["pairs"][pair]["excess_gap"] for m in src.values() if pair in m["pairs"]]
            R = [m["pairs"][pair]["gap"] for m in src.values() if pair in m["pairs"]]
            if not G:
                continue
            n = np.arange(1, 65)
            ax[0, j].plot(n, np.mean(G, 0), color=cols[c], label=f"gold {c}")
            ax[0, j].plot(n, np.mean(O, 0), color=cols[c], ls=":", lw=1)
            ax[1, j].plot(n, np.mean(X, 0), color=cols[c], label=f"excess {c}")
            ax[1, j].plot(n, np.mean(R, 0), color=cols[c], ls="--", lw=1)
        ax[0, j].set_title(f"{arm}: gold (solid), oracle (dotted)")
        ax[1, j].set_title(f"{arm}: excess gap (solid), raw gap (dashed)")
        ax[1, j].set_xscale("log", base=2); ax[1, j].axhline(0, color="k", lw=0.5)
        ax[1, j].set_xlabel("n (candidates, subset estimator)")
    ax[0, 0].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "e3_curves.png"), dpi=110)
    # raw gap(64) vs probe Brier vs Bayes, one point per (arm, seed, forecaster pair)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mk = {"P": "o", "RAPc": "s", "RAPf": "^", "M0g": "D", "B0s": "x"}
    for arm in ARMS:
        for s, m in M.get(arm, {}).items():
            for pair, pr in m["pairs"].items():
                f = pair.split(":")[1]
                if f in m["probes"]:
                    ax.scatter(m["probes"][f]["brier_vs_bayes"], pr["gap"][63], marker=mk[arm], s=18, label=arm)
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7)
    ax.set_xlabel("probe Brier vs exact Bayes (forecaster)"); ax.set_ylabel("raw gap at n=64")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "e3_gap_vs_brier.png"), dpi=110)


if __name__ == "__main__":
    main()
