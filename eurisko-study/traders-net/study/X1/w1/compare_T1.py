"""X1 worker 1: compare T1 headline claims (committed seeds 1-3) with a fresh-seed rerun (seed 4).

Usage: python3 compare_T1.py COMMITTED_T1_DIR FRESH_T1_DIR --out OUT.json
FRESH must have: w1/out/*.json from SEEDS=[4]; w2/results/summary*.{csv,json} from summarize.py on
seed-4-only delay/li JSONs; w3/out/summary.json + L6_growth.csv from aggregate.py on L6/L7-s4 only.
Kinds: stat = |x4-mean| <= 2 sd; ident = fresh value <= tol; bound = inequality holds in fresh;
same = deterministic (no seed), fresh must equal committed to 1e-9 rel; side = fresh on the same side of a threshold.
"""
import csv, json, os, sys, statistics as st

C, F = sys.argv[1], sys.argv[2]
OUT = sys.argv[sys.argv.index("--out") + 1]


def J(root, p): return json.load(open(os.path.join(root, p)))


def row(root, p, **m):
    for r in csv.DictReader(open(os.path.join(root, p))):
        if all(r[k] == str(v) for k, v in m.items()):
            return r
    raise KeyError((p, m))


def msd(root, path, *keys):
    d = J(root, path)
    for k in keys: d = d[k]
    return d  # [mean, sd]


def csvms(root, p, col, **m):
    r = row(root, p, **m); return [float(r[col]), float(r[col + "_sd"]) if col + "_sd" in r and r[col + "_sd"] != "" else None]


def perseed(vals): return [st.mean(vals), st.stdev(vals)]


L1b = lambda arm, k, key: lambda R: msd(R, "w1/out/L1b.json", "arms", arm, "mean_sd", k, key)
L1c = lambda key: lambda R: msd(R, "w1/out/L1c.json", "lams", "0.01", "mean_sd", key)
L1d = lambda key: lambda R: msd(R, "w1/out/L1d.json", "overfit_mean_sd", key)
L2d = lambda key: lambda R: msd(R, "w1/out/L2.json", "d_mean_sd", key)
DL = lambda fam, D, col, sh=0: lambda R: csvms(R, "w2/results/summary_delay.csv", col, shuffle=sh, D=D, family=fam)
DS = lambda fam, D, col, sh=0: lambda R: csvms(R, "w2/results/summary_delay_stable.csv", col, shuffle=sh, D=D, family=fam)
LI = lambda arm, fam, col: lambda R: csvms(R, "w2/results/summary_li.csv", col, arm=arm, family=fam)
G6 = lambda fam, col: lambda R: [float(row(R, "w3/out/L6_growth.csv", family=fam)[col + "_mean"]), float(row(R, "w3/out/L6_growth.csv", family=fam)[col + "_sd"])]
W2 = lambda *ks: lambda R: msd(R, "w2/results/summary.json", *ks)
W3 = lambda *ks: lambda R: msd(R, "w3/out/summary.json", *ks)
first = lambda g: lambda R: (lambda v: v[0] if isinstance(v, list) else v)(g(R))


def L2gap(R):  # |E[exact]-I| / se per seed
    c = J(R, "w1/out/L2.json")["c"]
    return max(abs(v["mean_exact"] - v["I_quadrature_nats"]) / v["se_exact"] for v in c.values())


def L3marg(R):
    b = J(R, "w1/out/L3.json")["b"]; return perseed([b[s]["marginal_recovered_frac"] for s in b]) if len(b) > 1 else [list(b.values())[0]["marginal_recovered_frac"], None]


def reader(R):
    v = J(R, "w2/results/summary.json")["li"]["a"]["max_reader_clearing_E_per_seed"]; return perseed(v) if len(v) > 1 else [v[0], None]


# (id, claim text, kind, getter, extra)
CL = [
 ("T1-L1.1", "Euler identity rel err 8.8e-16", "ident", first(lambda R: J(R, "w1/out/L1a.json")["worst_rel_err_frozen_graph"]), 1e-9),
 ("T1-L1.2a", "SGD per-step identity 1.8e-13", "ident", first(L1b("sgd", "per_step_identity_max_rel_resid", None) if False else (lambda R: J(R, "w1/out/L1b.json")["arms"]["sgd"]["mean_sd"]["per_step_identity_max_rel_resid"])), 1e-9),
 ("T1-L1.2b", "R2 books vs true income, SGD neurons 0.32", "stat", L1b("sgd", "neuron_out", "R2_Wbooks_vs_Wtrue"), None),
 ("T1-L1.2c", "R2 books vs true income, Adam neurons 0.14", "stat", L1b("adam", "neuron_out", "R2_Wbooks_vs_Wtrue"), None),
 ("T1-L1.2d", "R2 true income vs norm growth, SGD 0.989", "stat", L1b("sgd", "neuron_out", "R2_Wtrue_vs_dnorm"), None),
 ("T1-L1.3a", "Spearman(income,|theta|^2) wd 1e-2 0.9993", "stat", L1c("spearman_income_norm2"), None),
 ("T1-L1.3b", "rent/income median 1.008", "stat", L1c("rent_over_income_median"), None),
 ("T1-L1.4a", "in-sample books pay 100% of noise features", "stat", L1d("insample_frac_noise_pos"), None),
 ("T1-L1.4b", "prequential books charge 97.5% of noise features", "stat", L1d("preq_frac_noise_neg"), None),
 ("T1-L2.1", "exact = tangent + KL, err 1.1e-14", "ident", lambda R: J(R, "w1/out/L2.json")["a"]["max_abs_payoff_minus_tangent_minus_KL"], 1e-10),
 ("T1-L2.3", "E[exact] = I(X;Y) (seed 1: 0.2645 vs 0.2656); max |E-I|/se over seeds", "bound", L2gap, 3.0),
 ("T1-L2.4", "tangent vs IG rel L1 0.72", "stat", L2d("tangent_vs_IG_rel_L1_median"), None),
 ("T1-L3.2", "concave trunk marginal revenue covers 0.707 of cost", "stat", L3marg, None),
 ("T1-L3.M7", "E8 fine-grained 240,704 active FLOPs/token", "same", lambda R: J(R, "w1/out/flops_M7.json")["E8_fine_grained"]["active_flops_per_token"], 240704),
 ("T1-L3.M7b", "59,664 charge per routed token", "same", lambda R: J(R, "w1/out/flops_M7.json")["E8_fine_grained"]["charge_per_routed_token_shapley"], 59664),
 ("T1-L3.K1", "K1 break-even 5.7e-4 nats/FLOP", "same", lambda R: J(R, "w1/out/flops_K1.json")["breakeven_c_nats_per_flop_at_0.1_nats_per_trade"]["logistic32_alone"], 5.7e-4),
 ("T1-L4.1", "telescoping 3.3e-14", "ident", lambda R: J(R, "w2/results/summary.json")["delay"]["tel_tan_max"], 1e-10),
 ("T1-L4.3a", "hindsight gap periodic D=64 lr0.05 0.0051", "stat", DL("periodic", 64, "gap_mean"), None),
 ("T1-L4.3b", "hindsight gap periodic D=512 lr0.05 0.046", "stat", DL("periodic", 512, "gap_mean"), None),
 ("T1-L4.3c", "scaled step gap prime D=512 +0.030", "stat", DS("prime", 512, "gap_mean"), None),
 ("T1-L4.3d", "scaled step gap and D=512 +0.031", "stat", DS("and", 512, "gap_mean"), None),
 ("T1-L4.3e", "scaled step trade-price income prime D=512 -0.004", "stat", DS("prime", 512, "ex_mean"), None),
 ("T1-L4.3f", "scaled step trade-price income and D=512 -0.012", "stat", DS("and", 512, "ex_mean"), None),
 ("T1-L4.4a", "stale/|W_trade| D=64 -0.747", "stat", W2("delay", "stale_ratio_D64_shuf0"), None),
 ("T1-L4.4b", "stale/|W_trade| D=512 -1.197", "stat", W2("delay", "stale_ratio_D512_shuf0"), None),
 ("T1-L4.5a", "self-resolved prng D=0 exact income 0.056", "stat", DL("prng", 0, "self_ex_mean"), None),
 ("T1-L4.5b", "self-resolved prng D=0 KL 0.055", "stat", DL("prng", 0, "kl_mean"), None),
 ("T1-L5.1", "member max E 6.90739 <= ln(1/eps) 6.907755", "bound", lambda R: J(R, "w2/results/summary.json")["li"]["b"]["max_member_traded_E"], 6.907755),
 ("T1-L5.2", "reader at one-pass clearing 48.4 (seed 1) >> ln(1/eps)", "stat", reader, None),
 ("T1-L5.3", "a-D64 447.81 <= 449.00", "bound", lambda R: J(R, "w2/results/summary.json")["li"]["a-D64"]["max_member_traded_E"], 449.00),
 ("T1-L5.4a", "E_C periodic no births 1239", "stat", LI("N", "periodic", "E_C_end"), None),
 ("T1-L5.4b", "E_C periodic births rho=1e-2 472", "stat", LI("c1e-2", "periodic", "E_C_end"), None),
 ("T1-L5.4c", "E_C diag no births 4535", "stat", LI("N", "diag", "E_C_end"), None),
 ("T1-L5.4d", "E_C diag births rho=1e-2 1065", "stat", LI("c1e-2", "diag", "E_C_end"), None),
 ("T1-L5.4e", "plateau ratio diag 0.18 (<0.2 passes)", "side", lambda R: [float(row(R, "w2/results/summary_li.csv", arm="c1e-2", family="diag")["slope2_over_slope1"]), None], ("<", 0.2)),
 ("T1-L5.4f", "plateau ratio periodic 0.29 (fails <0.2)", "side", lambda R: [float(row(R, "w2/results/summary_li.csv", arm="c1e-2", family="periodic")["slope2_over_slope1"]), None], (">", 0.2)),
 ("T1-L6.1", "buy-and-hold DAG = flat mixture 8.9e-16", "ident", lambda R: J(R, "w3/out/summary.json")["C6a_max_abs_logdiff_dag_vs_flat__max_over_seeds"], 1e-10),
 ("T1-L6.2", "CRP DAG = flat CRP 9.1e-13", "ident", lambda R: J(R, "w3/out/summary.json")["C6b_max_abs_logdiff_crp_dag_vs_flat__max_over_seeds"], 1e-9),
 ("T1-L6.3", "toy CRP +0.087 vs leaf -0.424", "same", lambda R: J(R, "w3/out/summary.json")["C6e_toy"]["crp_50_50"], 0.08717669357238891),
 ("T1-L6.4", "ca110 EG beats best leaf by 0.0132", "stat", G6("ca110", "eg_minus_best_leaf"), None),
 ("T1-L7.1", "closed form vs autograd 8.6e-16", "ident", lambda R: J(R, "w3/out/summary.json")["C7a_renorm_max_rel_err"], 1e-10),
 ("T1-L7.4a", "top-1 censored finds best 6.7%", "stat", W3("C7e", "top1_unnorm_censored", "frac_final_argmax_is_best"), None),
 ("T1-L7.4b", "top-1 censored regret 0.378", "stat", W3("C7e", "top1_unnorm_censored", "mean_regret_last1k"), None),
 ("T1-L7.4c", "Exp3 finds best 52%", "stat", W3("C7e", "top1_unnorm_exp3", "frac_final_argmax_is_best"), None),
 ("T1-L7.4d", "Exp3 regret 0.023", "stat", W3("C7e", "top1_unnorm_exp3", "mean_regret_last1k"), None),
 ("T1-L7.5", "censored top-2 grad rel err 0.43", "stat", W3("C7d_censored_rel_err_vs_full_info"), None),
 ("T1-L7.6", "broker residual 2.6e-18", "ident", lambda R: J(R, "w3/out/summary.json")["C7c_top1_renorm_broker_residual_max"], 1e-12),
]

res = []
for cid, text, kind, g, extra in CL:
    r = {"id": cid, "claim": text, "kind": kind}
    try:
        cv, fv = g(C), g(F)
        cm, cs = (cv if isinstance(cv, list) else [cv, None])[:2]
        fx = fv[0] if isinstance(fv, list) else fv
        r.update(committed_mean=cm, committed_sd=cs, fresh=fx)
        if kind == "stat":
            band = 2 * cs if cs else 0.1 * abs(cm)
            r["band_note"] = "2 sd" if cs else "sd 0/none: 10% relative"
            r["z"] = (fx - cm) / cs if cs else None
            r["pass"] = abs(fx - cm) <= band
        elif kind == "ident":
            r["pass"] = abs(fx) <= extra; r["tol"] = extra
        elif kind == "bound":
            r["pass"] = fx <= extra; r["bound"] = extra
        elif kind == "same":
            r["pass"] = abs(fx - cm) <= 1e-9 * max(1, abs(cm)); r["quoted"] = extra
        elif kind == "side":
            op, th = extra; r["pass"] = (fx < th) if op == "<" else (fx > th); r["threshold"] = f"{op}{th}"
    except Exception as e:
        r["error"] = repr(e); r["pass"] = None
    res.append(r)
    print(f"{cid:10s} {kind:5s} pass={r['pass']} committed={r.get('committed_mean')}±{r.get('committed_sd')} fresh={r.get('fresh')} z={r.get('z')}")
json.dump({"committed": C, "fresh": F, "claims": res, "n_pass": sum(1 for r in res if r["pass"]), "n": len(res)}, open(OUT, "w"), indent=1, default=float)
print(sum(1 for r in res if r["pass"]), "/", len(res), "pass")
