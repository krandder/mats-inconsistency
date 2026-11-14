"""X1 w2 C1 (tolerance 6e-4 = 3-decimal display of a 4-decimal value): re-read every M1/M2/M3 headline number from the file it came from and compare with the claimed value.
Read-only. Writes results/claims_check.json. Run: python3 check_claims.py"""
import csv, json, os, datetime
S = "/home/agent/workspaces/eurisko-study/traders-net/study"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "claims_check.json")

def row(path, **kv):
    for r in csv.DictReader(open(os.path.join(S, path))):
        if all(r.get(k) == v for k, v in kv.items()): return r
    raise KeyError((path, kv))

def J(path): return json.load(open(os.path.join(S, path)))
toy = J("M1/lead/results/toy_summary.json")
sr = J("M2/w3/results/stoprule.json"); gb = J("M2/lead/gate_b.json")
lock = J("M1/CURRENCY.lock")
dec = J("M3/w2/results/decision.json")
sp = lambda gran, pred: row("M2/w3/results/spearman_summary.csv", ckpt="final", gran=gran, predictor=pred, ablation="zero")
co = lambda pred: row("M2/w3/results/coalition_summary.csv", ckpt="final", kind="random", predictor=pred)
cs = lambda c: row("M3/results/currency_summary.csv", currency=c)
lk = lambda a: row("M3/w2/results/summary.csv", arm=a)
ta = lambda st, m, col: float(row("M1/lead/results/three_agg.csv", settle=st, metric=m)[col])
sc = lambda src, cur: row("M1/lead/results/sign_census_summary.csv", source=src, currency=cur, unit="neuron", unit_set="top5pct")

C = [  # id, claim, file, claimed, found, tol
 ("M1-toy-w0", "closed-form disagreement interval upper end w0=2.1786", "M1/lead/results/toy_summary.json", 2.1786, toy["closed_form"]["w0"], 5e-5),
 ("M1-toy-V", "value at beta = mutual information 0.1109 nats", "M1/lead/results/toy_summary.json", 0.1109, toy["closed_form"]["value_at_beta"], 5e-5),
 ("M1-toy-grid", "23/23 grid points in the interval disagree in sign", "M1/lead/results/toy_summary.json", 23, toy["closed_form"]["sign_disagree_grid_points"], 0),
 ("M1-toy-adam-credit", "Adam credit of useful redundant units -75.66 +- 1.90", "M1/lead/results/toy_summary.json", -75.66, toy["sim"]["adam"]["credit_mean"], 5e-3),
 ("M1-toy-adam-settle", "Adam settlement +467.78 +- 6.02", "M1/lead/results/toy_summary.json", 467.78, toy["sim"]["adam"]["settle_mean"], 5e-3),
 ("M1-toy-adam-abl", "Adam ablation dBCE 0.0242 +- 0.0044", "M1/lead/results/toy_summary.json", 0.0242, toy["sim"]["adam"]["ablation_mean"], 5e-5),
 ("M1-census-books", "top-5% neurons with negative books_cum credit 0.256 +- 0.059", "M1/lead/results/sign_census_summary.csv", 0.256, float(sc("wealth-net final", "books_cum")["frac_neg_mean"]), 6e-4),
 ("M1-census-loo", "top-5% neurons with negative LOO settlement 0.026 +- 0.022 (held-out LOO)", "M1/lead/results/sign_census_summary.csv", 0.026, float(sc("wealth-net final", "settle_loo_heldout")["frac_neg_mean"]), 6e-4),
 ("M1-c-resid", "three-rules conserve to quadrature <= 5.1e-4 |R| at 16 steps (max over ckpts, lmsr)", "M1/lead/results/three_agg.csv", 5.1e-4, ta("lmsr", "resid_max_rel_three16", "max_over_ckpts"), 1e-5),
 ("M1-c-rank-head", "three-rules ranks = cp ranks, Spearman >= 0.97 (heads; this is a MEAN, sd 0.03)", "M1/lead/results/three_agg.csv", 0.97, ta("lmsr", "sp_three_vs_cp_head", "mean"), 6e-4),
 ("M1-c-rank-neuron", "three-rules ranks = cp ranks (neurons, mean)", "M1/lead/results/three_agg.csv", 0.988, ta("lmsr", "sp_three_vs_cp_neuron", "mean"), 6e-4),
 ("M1-lock", "CURRENCY.lock: committed-trader settlement only, no generic-neuron currency", "M1/CURRENCY.lock", "present", "present" if "settle" in json.dumps(lock).lower() else "missing", 0),
 ("M2-neuron-book", "neurons final: v2_lmsr_inst 0.199 +- 0.075", "M2/w3/results/spearman_summary.csv", 0.199, float(sp("neuron", "v2_lmsr_inst")["rho_mean"]), 6e-4),
 ("M2-neuron-taylor", "neurons final: |Taylor| 0.103 +- 0.084", "M2/w3/results/spearman_summary.csv", 0.103, float(sp("neuron", "taylor_abs")["rho_mean"]), 6e-4),
 ("M2-neuron-settle", "neurons final: settle_loo_train 0.464 +- 0.041", "M2/w3/results/spearman_summary.csv", 0.464, float(sp("neuron", "settle_loo_train")["rho_mean"]), 6e-4),
 ("M2-head-book", "heads final: v2_lmsr_inst 0.698 +- 0.339 (tie)", "M2/w3/results/stoprule.json", 0.698, sr["head"]["book_rho"], 6e-4),
 ("M2-head-taylor", "heads final: |Taylor| 0.809 +- 0.143", "M2/w3/results/stoprule.json", 0.809, sr["head"]["ref_rho"], 6e-4),
 ("M2-readout-book", "readouts final: v2_lmsr_int 0.303 +- 0.022 (loss)", "M2/w3/results/stoprule.json", 0.303, sr["readout"]["book_rho"], 6e-4),
 ("M2-readout-move", "readouts final: movement 0.814 +- 0.027", "M2/w3/results/stoprule.json", 0.814, sr["readout"]["ref_rho"], 6e-4),
 ("M2-coal-book", "random coalitions final: best book v2_lmsr_int 0.818 +- 0.027", "M2/w3/results/coalition_summary.csv", 0.818, float(co("v2_lmsr_int")["rho_mean"]), 6e-4),
 ("M2-coal-taylor", "random coalitions final: |Taylor| 0.800 +- 0.010", "M2/w3/results/coalition_summary.csv", 0.800, float(co("taylor_abs")["rho_mean"]), 6e-4),
 ("M2-coal-move", "random coalitions final: movement 0.804 +- 0.023", "M2/w3/results/coalition_summary.csv", 0.804, float(co("movement")["rho_mean"]), 6e-4),
 ("M3-A-cum", "all-time wealth (cum) Spearman vs oracle 0.043 +- 0.058", "M3/results/currency_summary.csv", 0.043, float(cs("cum")["mean"]), 6e-4),
 ("M3-A-gradnorm", "gradient-norm control 0.168 +- 0.070", "M3/results/currency_summary.csv", 0.168, float(cs("gradnorm")["mean"]), 6e-4),
 ("M3-A-settle", "exact settlement loo_train 0.417 +- 0.034", "M3/results/currency_summary.csv", 0.417, float(cs("settle_loo_train")["mean"]), 6e-4),
 ("M3-B-kelly", "Kelly post-shift learnable 0.690 +- 0.050", "M3/w2/results/summary.csv", 0.690, float(lk("kelly")["post2k_learnable_mean"]), 6e-4),
 ("M3-B-fs", "fixed share 1e-3 post-shift 0.047 +- 0.002", "M3/w2/results/summary.csv", 0.047, float(lk("fs1e-3")["post2k_learnable_mean"]), 6e-4),
 ("M3-B-lag-kelly", "Kelly median reallocation lag ~9007 steps", "M3/w2/results/summary.csv", 9007, float(lk("kelly")["lag_median_mean"]), 0.5),
 ("M3-B-lag-fs", "fixed share lag ~120 steps", "M3/w2/results/summary.csv", 120, float(lk("fs1e-3")["lag_median_mean"]), 0.5),
 ("M3-B-stat", "fixed share stationary cost ~0.001", "M3/w2/results/decision.json", 0.001, max(dec["per_arm"]["fs1e-3"]["stationary_cost"]), 6e-4),
]
res = []
for cid, claim, f, want, got, tol in C:
    ok = (want == got) if isinstance(want, str) else abs(float(got) - want) <= tol
    res.append(dict(id=cid, claim=claim, file=f, claimed=want, found=got, match=bool(ok)))
out = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           n=len(res), n_match=sum(r["match"] for r in res), rows=res,
           gate_b_20k_diffs=gb)
json.dump(out, open(OUT, "w"), indent=1, default=str)
for r in res: print(("OK  " if r["match"] else "BAD ") + r["id"], r["claimed"], r["found"])
print(out["n_match"], "/", out["n"])
