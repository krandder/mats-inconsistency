"""Aggregate out/L6-s*.json and out/L7-s*.json into out/summary.json and out/L6_growth.csv (mean, sd over seeds)."""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L6 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "out", "L6-s*.json")))]
L7 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "out", "L7-s*.json")))]
S = {"seeds_L6": [r["seed"] for r in L6], "n_L7": len(L7)}
for k in ["C6a_max_abs_logdiff_dag_vs_flat", "C6a_max_abs_logdiff_bayes_trader_vs_flat", "C6a_cyclic_max_abs_logdiff",
          "C6b_max_abs_logdiff_crp_dag_vs_flat", "C6c_max_abs_price_diff_bh_vs_flat"]:
    S[k + "__max_over_seeds"] = max(r[k] for r in L6)
for k in ["C6c_max_abs_price_diff_rebalanced_vs_flat", "C6d_bh_minus_bestleaf_min", "C6d_bh_minus_bestleaf_max"]:
    v = [r[k] for r in L6]; S[k] = [float(np.mean(v)), float(np.std(v, ddof=1))]
S["C6d_bound_ln10_over_T"] = L6[0]["C6d_bound_ln10_over_T"]
S["C6e_toy"] = L6[0]["C6e_toy"]
cols = ["best_leaf", "bh_uniform", "crp_uniform", "crp_eg_online", "crp_best_hindsight", "frac_q_eg_beats_best_leaf",
        "frac_q_bestcrp_gt_best_leaf_by_1e-3"]
lines = ["family," + ",".join(f"{c}_mean,{c}_sd" for c in cols) + ",eg_minus_best_leaf_mean,eg_minus_best_leaf_sd"]
for i, row in enumerate(L6[0]["C6d"]):
    vals = []
    for c in cols:
        v = [r["C6d"][i][c] for r in L6]; vals += [np.mean(v), np.std(v, ddof=1)]
    d = [r["C6d"][i]["crp_eg_online"] - r["C6d"][i]["best_leaf"] for r in L6]
    vals += [np.mean(d), np.std(d, ddof=1)]
    lines.append(row["family"] + "," + ",".join(f"{x:.6g}" for x in vals))
open(os.path.join(HERE, "out", "L6_growth.csv"), "w").write("\n".join(lines) + "\n")
if L7:
    convs = ["renorm", "unnorm", "sigmoid", "top1_renorm", "top1_unnorm", "dense"]
    for c in convs:
        S[f"C7a_{c}_max_abs_err"] = max(r[c]["C7a_max_abs_err"] for r in L7)
        S[f"C7a_{c}_max_rel_err"] = max(r[c]["C7a_max_abs_err"] / max(r[c]["C7a_scale_max_abs_grad"], 1e-300) for r in L7)
        S[f"C7b_{c}_max_abs_sum"] = max(r[c]["C7b_max_abs_sum_over_experts"] for r in L7)
        lo = min(r[c]["C7c_ratio_gate_plus_expert_over_R_minmax"][0] for r in L7)
        hi = max(r[c]["C7c_ratio_gate_plus_expert_over_R_minmax"][1] for r in L7)
        S[f"C7c_{c}_ratio_range"] = [lo, hi]
        S[f"C7c_{c}_broker_residual_max"] = max(max(r[c]["C7c_broker_residual_max"], r[c]["C7c_broker_into_neurons_residual_max"]) for r in L7)
    S["C7b_unnorm_mass_transfer_err_max"] = max(r["unnorm"]["C7b_routed_gain_minus_R(1-sum_S g)"] for r in L7)
    S["C7c_router_income_minus_cov_max"] = max(r["renorm"]["C7c_router_income_minus_cov_max"] for r in L7)
    S["C7c_swiglu_ratio"] = [r["swiglu_factor_books_over_product_books"] for r in L7]
    S["C7d_max_abs_z"] = [r["C7d_max_abs_z"] for r in L7]
    v = [r["C7d_censored_rel_err_vs_target"] for r in L7]; S["C7d_censored_rel_err_vs_full_info"] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    if all("C7e" in r for r in L7):
        S["C7e"] = {}
        for arm in L7[0]["C7e"]:
            S["C7e"][arm] = {m: [float(np.mean([r["C7e"][arm][m] for r in L7])), float(np.std([r["C7e"][arm][m] for r in L7], ddof=1)) if len(L7) > 1 else 0.0]
                             for m in L7[0]["C7e"][arm]}
json.dump(S, open(os.path.join(HERE, "out", "summary.json"), "w"), indent=1)
print(json.dumps(S, indent=1)); print("\n".join(lines))
