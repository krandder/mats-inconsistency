"""M8 lead check: runs both part checks, re-reads every number in M8.md, cross-checks the shared baseline.
Usage: PYTHONPATH=<wealth-net>/.pylib python3 check.py"""
import csv, json, os, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
bad = []
for w in ("w1", "w2"):
    out = subprocess.run([sys.executable, "check.py"], cwd=os.path.join(H, w), capture_output=True, text=True).stdout
    if "PASS" not in out: bad.append((w, out.strip()))
ld = lambda p: json.load(open(os.path.join(H, p)))
lc = lambda p: {r["arm"]: r for r in csv.DictReader(open(os.path.join(H, p)))}
D1, D2 = ld("w1/results/decision.json"), ld("w2/results/decision.json")
S1, S2 = lc("w1/results/summary.csv"), lc("w2/results/summary.csv")
def eq(name, got, want):
    if got != want: bad.append((name, got, want))
r4 = lambda xs: [round(x, 4) for x in xs]
eq("shuf", r4(D1["nolfsr_shuf1_minus_pos1"]), [-0.0008, -0.003, 0.0008])
eq("neg", r4(D1["nolfsr_neg1_minus_pos1"]), [0.0079, 0.0026, 0.0051])
eq("none", r4(D1["nolfsr_none_minus_pos1"]), [-0.0067, -0.0066, -0.0066])
ms = lambda S, a, c, nd=4: (round(float(S[a][c + "_mean"]), nd), round(float(S[a][c + "_sd"]), nd))
for a, v in {"none": (0.3554, 0.0071), "pos1": (0.3673, 0.0145), "neg1": (0.3719, 0.0135), "shuf1": (0.3665, 0.0153)}.items():
    eq(a, ms(S1, a, "prequential_bce"), v)
eq("top10", ms(S1, "none", "neg_share_top10", 3), (0.346, 0.033)); eq("all", ms(S1, "none", "neg_share_all", 3), (0.49, 0.019))
eq("sp_inc", ms(S1, "none", "spearman_income", 3), (0.106, 0.034)); eq("sp_abs", ms(S1, "none", "spearman_absag", 3), (0.203, 0.068))
eq("stop", (D2["stop"], D2["absag_vs_random"]["d_postshift_switched"]), (True, [-0.01605, 0.00067, 0.00106]))  # quoted -0.0161/+0.0007/+0.0011
for a, v in {"absag": (0.4515, 0.0447), "random": (0.4563, 0.0371), "cbp": (0.4599, 0.0358), "dense256": (0.4533, 0.033)}.items():
    eq(a, ms(S2, a, "bce_postshift_switched"), v)
eq("absag_all", ms(S2, "absag", "bce_all"), (0.3494, 0.02)); eq("random_all", ms(S2, "random", "bce_all"), (0.3623, 0.0146))
eq("absag_nl", ms(S2, "absag", "bce_all_nolfsr", 5), (0.31059, 0.01825))  # quoted 0.3106 +- 0.0183 (half-up); eq("random_nl", ms(S2, "random", "bce_all_nolfsr"), (0.3089, 0.0195))
eq("nolfsr_none", any(v["gain_3of3"] for v in D2["absag_vs_nolfsr"].values()), False)
for a, v in {"movement": (14.7, 3.5), "absag": (67.7, 23.7), "taylor": (73.0, 23.1)}.items():
    eq(a, ms(S2, a, "births_survived", 1), v)
L = D2["lfsr16_learnt_by_arm"]
k = sum(sum(L[a]) for a in ("absag", "taylor", "magnitude", "cbp")); z = sum(sum(L[a]) for a in ("random", "static128", "movement"))
eq("lottery", (k, z), (5, 0))
try:
    from scipy.stats import fisher_exact
    eq("fisher", round(fisher_exact([[k, 12 - k], [z, 9 - z]], alternative="greater")[1], 3), 0.039)
    eq("fisher_absag", round(fisher_exact([[sum(L["absag"]), 1], [sum(L["random"]), 3]], alternative="greater")[1], 2), 0.2)
except ImportError:
    bad.append("scipy missing: Fisher p not rechecked")
# cross-part: w1 'none' and w2 'dense256' are the same ungated 2x256 net coded independently
eq("x_all", (round(float(S1["none"]["prequential_bce_mean"]), 4), round(float(S2["dense256"]["bce_all_mean"]), 4)), (0.3554, 0.3546))
eq("x_post", (round(float(S1["none"]["post2k_switched_mean"]), 4), round(float(S2["dense256"]["bce_postshift_switched_mean"]), 4)), (0.4533, 0.4533))
w1s = [float(x) for x in S1["none"]["post2k_switched_seeds"].split("/")]
w2s = [json.load(open(os.path.join(H, f"w2/results/run_dense256_s{s}.json")))["bce_postshift_switched"] for s in (1, 2, 3)]
w1a = [float(x) for x in S1["none"]["prequential_bce_seeds"].split("/")]
w2a = [json.load(open(os.path.join(H, f"w2/results/run_dense256_s{s}.json")))["bce_all"] for s in (1, 2, 3)]
if max(abs(a - b) for a, b in zip(w1s + w1a, w2s + w2a)) > 0.0015: bad.append(("cross-part seeds", w1s, w2s, w1a, w2a))
print("PASS" if not bad else f"FAIL {bad}"); sys.exit(1 if bad else 0)
