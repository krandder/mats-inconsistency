"""Re-read the T1 output files and check every number quoted in THEORY.md's lead summary.

Usage: python3 check_claims.py [--out lead/claims_check.json]
Each claim: (id, stated value, tolerance, file relative to study/T1, extractor). A claim passes
when |extracted - stated| <= tol (tol is the rounding used in the text). Exit code 1 if any fails.
"""
import csv, json, os, sys

T1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def J(path):
    return json.load(open(os.path.join(T1, path)))


def C(path, **match):
    for r in csv.DictReader(open(os.path.join(T1, path))):
        if all(r[k] == str(v) for k, v in match.items()):
            return r
    raise KeyError((path, match))


L1b = lambda arm, kind, key, i=0: (lambda: J("w1/out/L1b.json")["arms"][arm]["mean_sd"][kind][key][i])
L1c = lambda lam, key, i=0: (lambda: J("w1/out/L1c.json")["lams"][lam]["mean_sd"][key][i])
L1d = lambda key, i=0: (lambda: J("w1/out/L1d.json")["overfit_mean_sd"][key][i])
L2d = lambda key, i=0: (lambda: J("w1/out/L2.json")["d_mean_sd"][key][i])
W2 = lambda *ks: (lambda: _dig(J("w2/results/summary.json"), ks))
W3 = lambda *ks: (lambda: _dig(J("w3/out/summary.json"), ks))
DS = lambda fam, D, col, shuf=0: (lambda: float(C("w2/results/summary_delay_stable.csv", shuffle=shuf, D=D, family=fam)[col]))
DL = lambda fam, D, col, shuf=0: (lambda: float(C("w2/results/summary_delay.csv", shuffle=shuf, D=D, family=fam)[col]))
LI = lambda arm, fam, col: (lambda: float(C("w2/results/summary_li.csv", arm=arm, family=fam)[col]))
G6 = lambda fam, col: (lambda: float(C("w3/out/L6_growth.csv", family=fam)[col]))


def _dig(d, ks):
    for k in ks:
        d = d[k]
    return d


def L1a_worst():
    return J("w1/out/L1a.json")["worst_rel_err_frozen_graph"]


def L2_I(seed, key):
    return lambda: J("w1/out/L2.json")["c"][str(seed)][key]


CLAIMS = [
    # Lemma 1
    ("L1.1 Euler identity, frozen-LN graph, worst rel err (<1e-9)", 8.8e-16, 1e-16, "w1/out/L1a.json", L1a_worst),
    ("L1.2 SGD per-step identity max rel resid", 1.8e-13, 1e-14, "w1/out/L1b.json", lambda: J("w1/out/L1b.json")["arms"]["sgd"]["mean_sd"]["per_step_identity_max_rel_resid"][0]),
    ("L1.2 R2 true income vs norm growth, SGD neurons(out)", 0.989, 0.001, "w1/out/L1b.json", L1b("sgd", "neuron_out", "R2_Wtrue_vs_dnorm")),
    ("L1.2 R2 books vs true income, SGD neurons", 0.32, 0.005, "w1/out/L1b.json", L1b("sgd", "neuron_out", "R2_Wbooks_vs_Wtrue")),
    ("L1.2 R2 books vs true income, Adam neurons", 0.14, 0.005, "w1/out/L1b.json", L1b("adam", "neuron_out", "R2_Wbooks_vs_Wtrue")),
    ("L1.2 R2 books vs norm growth, Adam neurons(out)", 0.022, 0.0005, "w1/out/L1b.json", L1b("adam", "neuron_out", "R2_Wbooks_vs_dnorm")),
    ("L1.3 Spearman(income, |theta|^2), lambda=1e-2", 0.9993, 0.00005, "w1/out/L1c.json", L1c("0.01", "spearman_income_norm2")),
    ("L1.3 rent/income median, lambda=1e-2", 1.008, 0.0005, "w1/out/L1c.json", L1c("0.01", "rent_over_income_median")),
    ("L1.4 in-sample: frac noise features with income>0", 1.0, 0, "w1/out/L1d.json", L1d("insample_frac_noise_pos")),
    ("L1.4 prequential: frac noise features with income<0", 0.975, 0.0005, "w1/out/L1d.json", L1d("preq_frac_noise_neg")),
    # Lemma 2
    ("L2.1 payoff = log-score gain = tangent+KL, max abs err", 1.1e-14, 1e-15, "w1/out/L2.json", lambda: J("w1/out/L2.json")["a"]["max_abs_payoff_minus_tangent_minus_KL"]),
    ("L2.3 seed1 E[exact] from base rate", 0.2645, 0.00005, "w1/out/L2.json", L2_I(1, "mean_exact")),
    ("L2.3 seed1 I(X;Y)", 0.2656, 0.00005, "w1/out/L2.json", L2_I(1, "I_quadrature_nats")),
    ("L2.4 tangent vs IG shares, rel L1 median", 0.72, 0.005, "w1/out/L2.json", L2d("tangent_vs_IG_rel_L1_median")),
    ("L2.4 arrival-order sd / IG share, median", 0.95, 0.005, "w1/out/L2.json", L2d("arrival_order_sd_over_IG_median")),
    # Lemma 3
    ("L3.2 marginal revenue / C, concave trunk, seed1", 0.707, 0.001, "w1/out/L3.json", lambda: J("w1/out/L3.json")["b"]["1"]["marginal_recovered_frac"]),
    ("L3 M7 E8 fine-grained charge per routed token (A)", 59664, 0, "w1/out/flops_M7.json", lambda: J("w1/out/flops_M7.json")["E8_fine_grained"]["charge_per_routed_token_shapley"]),
    ("L3 M7 E8 fine-grained active FLOPs/token", 240704, 0, "w1/out/flops_M7.json", lambda: J("w1/out/flops_M7.json")["E8_fine_grained"]["active_flops_per_token"]),
    ("L3 K1 break-even nats/FLOP, lone logistic-32", 5.7e-4, 5e-6, "w1/out/flops_K1.json", lambda: J("w1/out/flops_K1.json")["breakeven_c_nats_per_flop_at_0.1_nats_per_trade"]["logistic32_alone"]),
    # Lemma 4
    ("L4.1 telescoping, tangent books, max resid", 3.3e-14, 1e-15, "w2/results/summary.json", W2("delay", "tel_tan_max")),
    ("L4.3 hindsight gap periodic D=64, lr 0.05", 0.0051, 0.00005, "w2/results/summary_delay.csv", DL("periodic", 64, "gap_mean")),
    ("L4.3 hindsight gap periodic D=512, lr 0.05", 0.046, 0.0005, "w2/results/summary_delay.csv", DL("periodic", 512, "gap_mean")),
    ("L4.3 hindsight gap periodic D=512 shuffled, lr 0.05", -0.57, 0.005, "w2/results/summary_delay.csv", DL("periodic", 512, "gap_mean", 1)),
    ("L4.3 exact income prng D=512, lr 0.05", -1.78, 0.005, "w2/results/summary_delay.csv", DL("prng", 512, "ex_mean")),
    ("L4.3 scaled step: hindsight gap prime D=512", 0.0296, 0.00005, "w2/results/summary_delay_stable.csv", DS("prime", 512, "gap_mean")),
    ("L4.3 scaled step: hindsight gap and D=512", 0.0307, 0.00005, "w2/results/summary_delay_stable.csv", DS("and", 512, "gap_mean")),
    ("L4.3 scaled step: trade-price exact income prime D=512", -0.004, 0.0005, "w2/results/summary_delay_stable.csv", DS("prime", 512, "ex_mean")),
    ("L4.3 scaled step: trade-price exact income and D=512", -0.012, 0.0005, "w2/results/summary_delay_stable.csv", DS("and", 512, "ex_mean")),
    ("L4.3 scaled step: |gap| lfsr16 D=64 (largest on lfsr16/prng)", 4.9e-4, 5e-6, "w2/results/summary_delay_stable.csv", lambda: abs(DS("lfsr16", 64, "gap_mean")())),
    ("L4.4 stale/|W_trade| D=64, lr 0.05", -0.747, 0.0005, "w2/results/summary.json", W2("delay", "stale_ratio_D64_shuf0", 0)),
    ("L4.4 stale/|W_trade| D=512, lr 0.05", -1.197, 0.0005, "w2/results/summary.json", W2("delay", "stale_ratio_D512_shuf0", 0)),
    ("L4.5 self-resolved prng D=0: exact income", 0.056, 0.0005, "w2/results/summary_delay.csv", DL("prng", 0, "self_ex_mean")),
    ("L4.5 self-resolved prng D=0: KL", 0.055, 0.0005, "w2/results/summary_delay.csv", DL("prng", 0, "kl_mean")),
    # Lemma 5
    ("L5.1 bound ln(1/eps)", 6.907755, 1e-6, "w2/results/summary.json", W2("li", "b", "bound")),
    ("L5.1 arm b max member E (<= bound)", 6.90739, 5e-6, "w2/results/summary.json", W2("li", "b", "max_member_traded_E")),
    ("L5.2 arm a reader-at-clearing E, seed 1", 48.4, 0.05, "w2/results/summary.json", W2("li", "a", "max_reader_clearing_E_per_seed", 0)),
    ("L5.3 arm a-D64 max member E", 447.81, 0.005, "w2/results/summary.json", W2("li", "a-D64", "max_member_traded_E")),
    ("L5.3 bound (D+1) ln(1/eps)", 449.00, 0.005, "w2/results/summary.json", W2("li", "a-D64", "bound")),
    ("L5.4 E_C periodic, no births (arm N)", 1239, 0.5, "w2/results/summary_li.csv", LI("N", "periodic", "E_C_end")),
    ("L5.4 E_C periodic, births rho=1e-2", 472, 0.5, "w2/results/summary_li.csv", LI("c1e-2", "periodic", "E_C_end")),
    ("L5.4 E_C diag, no births", 4535, 0.5, "w2/results/summary_li.csv", LI("N", "diag", "E_C_end")),
    ("L5.4 E_C diag, births rho=1e-2", 1065, 0.5, "w2/results/summary_li.csv", LI("c1e-2", "diag", "E_C_end")),
    ("L5.4 plateau ratio diag rho=1e-2 (<0.2 pass)", 0.18, 0.005, "w2/results/summary_li.csv", LI("c1e-2", "diag", "slope2_over_slope1")),
    ("L5.4 plateau ratio periodic rho=1e-2 (<0.2 fails)", 0.29, 0.005, "w2/results/summary_li.csv", LI("c1e-2", "periodic", "slope2_over_slope1")),
    # Lemma 6
    ("L6.1 buy-and-hold DAG vs flat mixture, max |dlogW|", 8.9e-16, 1e-17, "w3/out/summary.json", W3("C6a_max_abs_logdiff_dag_vs_flat__max_over_seeds")),
    ("L6.2 rebalanced DAG vs flat CRP, max |dlogW|", 9.1e-13, 1e-14, "w3/out/summary.json", W3("C6b_max_abs_logdiff_crp_dag_vs_flat__max_over_seeds")),
    ("L6.3 toy 50/50 CRP growth", 0.087, 0.0005, "w3/out/summary.json", W3("C6e_toy", "crp_50_50")),
    ("L6.3 toy leaf / buy-and-hold growth", -0.424, 0.0005, "w3/out/summary.json", W3("C6e_toy", "bh_asymptotic")),
    ("L6 ca110 online EG minus best leaf", 0.0132, 0.00005, "w3/out/L6_growth.csv", G6("ca110", "eg_minus_best_leaf_mean")),
    ("L6 periodic uniform CRP growth", 0.20, 0.005, "w3/out/L6_growth.csv", G6("periodic", "crp_uniform_mean")),
    # Lemma 7
    ("L7.1 closed form vs autograd, max rel err (renorm)", 8.6e-16, 1e-17, "w3/out/summary.json", W3("C7a_renorm_max_rel_err")),
    ("L7.4 top-1 censored: final argmax = best", 0.067, 0.0005, "w3/out/summary.json", W3("C7e", "top1_unnorm_censored", "frac_final_argmax_is_best", 0)),
    ("L7.4 top-1 + Exp3: final argmax = best", 0.52, 0.005, "w3/out/summary.json", W3("C7e", "top1_unnorm_exp3", "frac_final_argmax_is_best", 0)),
    ("L7.4 top-1 censored regret last 1k", 0.378, 0.0005, "w3/out/summary.json", W3("C7e", "top1_unnorm_censored", "mean_regret_last1k", 0)),
    ("L7.4 top-1 + Exp3 regret last 1k", 0.023, 0.0005, "w3/out/summary.json", W3("C7e", "top1_unnorm_exp3", "mean_regret_last1k", 0)),
    ("L7.5 censored top-2 grad rel err vs full info", 0.43, 0.005, "w3/out/summary.json", W3("C7d_censored_rel_err_vs_full_info", 0)),
    ("L7.6 broker books conserve, max resid", 2.6e-18, 1e-19, "w3/out/summary.json", W3("C7c_top1_renorm_broker_residual_max")),
]


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rows, nfail = [], 0
    for cid, stated, tol, path, fn in CLAIMS:
        try:
            got = float(fn())
            ok = abs(got - stated) <= tol + 1e-300
        except Exception as e:  # noqa: BLE001
            got, ok = repr(e), False
        nfail += not ok
        rows.append(dict(claim=cid, stated=stated, got=got, tol=tol, file="study/T1/" + path, ok=ok))
        print("%-4s %-62s stated %-10g got %s" % ("ok" if ok else "FAIL", cid, stated, got))
    print("%d claims, %d fail" % (len(rows), nfail))
    if out:
        json.dump(dict(n=len(rows), n_fail=nfail, claims=rows), open(out, "w"), indent=1)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
