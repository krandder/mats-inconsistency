"""Re-read every number quoted in M8-w1.md from results/ (summary.csv, decision.json). Prints PASS/FAIL."""
import csv, json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
S = {r["arm"]: r for r in csv.DictReader(open(os.path.join(R, "summary.csv")))}
D = json.load(open(os.path.join(R, "decision.json")))
bad = []
def q(arm, key, m, sd, nd=4):
    a, b = round(float(S[arm][key + "_mean"]), nd), round(float(S[arm][key + "_sd"]), nd)
    if (a, b) != (m, sd): bad.append((arm, key, a, b, m, sd))
def v(name, vals, nd=4):
    got = [round(x, nd) for x in D[name]]
    if got != vals: bad.append((name, got, vals))
T = {  # arm: (prequential, no_lfsr16, post2k_switched)
 "none": ((0.3554, 0.0071), (0.3079, 0.0190), (0.4533, 0.0330)),
 "pos0.1": ((0.3542, 0.0066), (0.3075, 0.0186), (0.4491, 0.0364)),
 "neg0.1": ((0.3595, 0.0115), (0.3077, 0.0188), (0.4544, 0.0345)),
 "pos1": ((0.3673, 0.0145), (0.3145, 0.0189), (0.4581, 0.0330)),
 "neg1": ((0.3719, 0.0135), (0.3197, 0.0169), (0.4706, 0.0352)),
 "shuf1": ((0.3665, 0.0153), (0.3136, 0.0192), (0.4590, 0.0349)),
}
for a, (p, n, s) in T.items():
    q(a, "prequential_bce", *p); q(a, "bce_no_lfsr16", *n); q(a, "post2k_switched", *s)
q("none", "neg_share_all", 0.4902, 0.0188); q("none", "neg_share_top10", 0.3462, 0.0333)
q("none", "spearman_income", 0.1059, 0.0339); q("none", "spearman_absag", 0.2031, 0.0684)
q("neg1", "gate_max", 255.7369, 0.0012); q("pos1", "gate_max", 16.3374, 6.6114)
v("nolfsr_neg1_minus_pos1", [0.0079, 0.0026, 0.0051]); v("nolfsr_shuf1_minus_pos1", [-0.0008, -0.003, 0.0008])
v("nolfsr_none_minus_pos1", [-0.0067, -0.0066, -0.0066]); v("nolfsr_neg01_minus_pos01", [-0.0001, 0.0006, 0.0002])
v("neg1_minus_pos1", [0.0068, 0.0024, 0.0044]); v("shuf1_minus_pos1", [-0.0007, -0.0026, 0.0007])
for k, want in (("rule1_sign_informative", False), ("rule2_stop_M8", True), ("rule3_anti_informative_on_important", False)):
    if D[k] != want: bad.append((k, D[k], want))
if S["none"]["lfsr16_learnt_q_seeds"] != "0.0000/0.0000/9.0000": bad.append("lfsr none")
print("PASS" if not bad else f"FAIL {bad}")
