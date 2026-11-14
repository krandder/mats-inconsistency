"""Re-reads every number quoted in M8-w2.md from results/summary.csv and results/decision.json.
Usage: python3 check.py  (rerun from scratch: ./runall.sh; for s in 1 2 3: python3 family.py $s; python3 aggregate.py)"""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
S = {r["arm"]: r for r in csv.DictReader(open(os.path.join(HERE, "results", "summary.csv")))}
D = json.load(open(os.path.join(HERE, "results", "decision.json")))
Q = {  # (arm, column): quoted value, 4 decimals
 ("absag", "bce_all"): (0.3494, 0.0200), ("random", "bce_all"): (0.3623, 0.0146),
 ("static128", "bce_all"): (0.3639, 0.0138), ("dense256", "bce_all"): (0.3546, 0.0071),
 ("taylor", "bce_all"): (0.3538, 0.0223), ("magnitude", "bce_all"): (0.3539, 0.0075),
 ("cbp", "bce_all"): (0.3594, 0.0208), ("movement", "bce_all"): (0.3632, 0.0144),
 ("absag", "bce_postshift_switched"): (0.4515, 0.0447), ("random", "bce_postshift_switched"): (0.4563, 0.0371),
 ("static128", "bce_postshift_switched"): (0.4584, 0.0345), ("dense256", "bce_postshift_switched"): (0.4533, 0.0330),
 ("taylor", "bce_postshift_switched"): (0.4535, 0.0271), ("magnitude", "bce_postshift_switched"): (0.4543, 0.0364),
 ("cbp", "bce_postshift_switched"): (0.4599, 0.0358), ("movement", "bce_postshift_switched"): (0.4560, 0.0322),
 ("absag", "bce_all_nolfsr"): (0.3106, 0.0183), ("random", "bce_all_nolfsr"): (0.3089, 0.0195),
 ("static128", "bce_all_nolfsr"): (0.3107, 0.0186), ("dense256", "bce_all_nolfsr"): (0.3075, 0.0191),
 ("absag", "postshift_switched_nolfsr"): (0.4070, 0.0497), ("random", "postshift_switched_nolfsr"): (0.4033, 0.0530),
 ("dense256", "postshift_switched_nolfsr"): (0.4008, 0.0478),
 ("absag", "births_survived"): (67.67, 23.69), ("random", "births_survived"): (190.67, 4.16),
 ("cbp", "births_survived"): (43.33, 5.51), ("taylor", "births_survived"): (73.0, 23.07),
 ("magnitude", "births_survived"): (35.33, 5.51), ("movement", "births_survived"): (14.67, 3.51),
}
bad = 0
for (a, c), (m, sd) in Q.items():
    tm, ts = float(S[a][c + "_mean"]), float(S[a][c + "_sd"])
    tol = 0.006 if c.startswith("births") else 6e-5
    if abs(tm - m) > tol or abs(ts - sd) > tol:
        print("MISMATCH", a, c, (m, sd), (tm, ts)); bad += 1
assert D["stop"] is True and D["absag_vs_random"]["gain_3of3"] is False
assert D["absag_vs_random"]["d_postshift_switched"] == [-0.01605, 0.00067, 0.00106]
assert D["absag_vs"]["cbp"]["gain_3of3"] is True
assert not any(v["gain_3of3"] for v in D["absag_vs_nolfsr"].values())
L = D["lfsr16_learnt_by_arm"]
assert L["absag"] == [1, 0, 1] and sum(L["random"]) == sum(L["static128"]) == sum(L["movement"]) == 0
assert sum(sum(L[a]) for a in ("absag", "taylor", "magnitude", "cbp")) == 5 and sum(L["dense256"]) == 1
print("PASS" if bad == 0 else f"FAIL {bad}"); sys.exit(1 if bad else 0)
