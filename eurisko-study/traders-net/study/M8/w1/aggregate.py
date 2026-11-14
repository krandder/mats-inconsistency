"""Aggregate M8 w1 sign-test runs -> results/summary.csv, results/decision.json."""
import json, os, glob, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
ARMS = ["none", "pos0.1", "neg0.1", "pos1", "neg1", "shuf1"]; SEEDS = [1, 2, 3]
runs = {(a, s): json.load(open(os.path.join(R, f"run-{a}-s{s}.json"))) for a in ARMS for s in SEEDS}
def ms(v): v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1))
def metrics(r):
    return dict(prequential_bce=r["prequential_bce"], bce_no_lfsr16=r["prequential_bce_no_lfsr16"],
                lfsr16_learnt_q=r["lfsr16_learnt_q"], bce_prng=r["family_bce"]["prng"], bce_ca30=r["family_bce"]["ca30"],
                post2k=np.mean([v for v in r["post2k"].values()]),
                post2k_switched=np.mean([v for v in r["post2k_switched"].values()]),
                pre2k=np.mean([v for v in r["pre2k"].values()]),
                heldout_bce=r["census"]["heldout_bce"], neg_share_all=r["census"]["neg_share_all"],
                neg_share_top10=r["census"]["neg_share_top10"], spearman_income=r["census"]["spearman_income"],
                spearman_absag=r["census"]["spearman_absag"], gate_max=r["gate_max"])
keys = list(metrics(runs[("none", 1)]).keys())
rows = []
for a in ARMS:
    row = {"arm": a}
    for k in keys:
        vals = [metrics(runs[(a, s)])[k] for s in SEEDS]
        m, sd = ms(vals); row[k + "_mean"] = m; row[k + "_sd"] = sd
        row[k + "_seeds"] = "/".join(f"{v:.4f}" for v in vals)
    rows.append(row)
with open(os.path.join(R, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
P = lambda a, s: runs[(a, s)]["prequential_bce"]
d_pn = [P("neg1", s) - P("pos1", s) for s in SEEDS]      # >0: pos better
d_ps = [P("shuf1", s) - P("pos1", s) for s in SEEDS]
d_pn01 = [P("neg0.1", s) - P("pos0.1", s) for s in SEEDS]
d_n0 = [P("none", s) - P("pos1", s) for s in SEEDS]
Pn = lambda a, s: runs[(a, s)]["prequential_bce_no_lfsr16"]
e_pn = [Pn("neg1", s) - Pn("pos1", s) for s in SEEDS]
e_ps = [Pn("shuf1", s) - Pn("pos1", s) for s in SEEDS]
e_n0 = [Pn("none", s) - Pn("pos1", s) for s in SEEDS]
e_pn01 = [Pn("neg0.1", s) - Pn("pos0.1", s) for s in SEEDS]
rule1_raw = all(x >= 0.005 for x in d_pn) and all(x >= 0.005 for x in d_ps)
rule1_nolfsr = all(x >= 0.005 for x in e_pn) and all(x >= 0.005 for x in e_ps)
rule1 = rule1_raw and rule1_nolfsr
rule2_stop = not all(x > 0 for x in d_ps)
cen = [(runs[("none", s)]["census"]["neg_share_top10"], runs[("none", s)]["census"]["neg_share_all"]) for s in SEEDS]
rule3 = all(t > a for t, a in cen)
dec = dict(neg1_minus_pos1=d_pn, shuf1_minus_pos1=d_ps, neg01_minus_pos01=d_pn01, none_minus_pos1=d_n0,
           nolfsr_neg1_minus_pos1=e_pn, nolfsr_shuf1_minus_pos1=e_ps, nolfsr_none_minus_pos1=e_n0,
           nolfsr_neg01_minus_pos01=e_pn01, rule1_raw=rule1_raw, rule1_nolfsr=rule1_nolfsr, rule1_sign_informative=rule1,
           census_all_arms={a: [[runs[(a, s)]["census"]["neg_share_top10"], runs[(a, s)]["census"]["neg_share_all"]] for s in SEEDS] for a in ARMS}, rule2_stop_M8=rule2_stop,
           census_none_top10_vs_all=cen, rule3_anti_informative_on_important=rule3)
json.dump(dec, open(os.path.join(R, "decision.json"), "w"), indent=1)
for r in rows:
    print(r["arm"], *(f"{k}={r[k+'_mean']:.4f}+-{r[k+'_sd']:.4f}" for k in keys))
print(json.dumps(dec))
