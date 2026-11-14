"""Writes t1_queue2.txt: follow-up DIAGNOSTIC (not pre-registered), decided after the first grid showed that
Shapley pay with an exploring maker (x05) aligned 10/20 vs 3/20 for the additive reader.  Held-out replication
on fresh seeds 2 and 3 (20 new replicates per arm): x05 {shap, loo, banz, asig}, fresh additive and Moebius
(lambda 1e-3, the validation pick) baselines, plus a dose check at exploration 0.2 for shap and loo (seeds 0, 1).
Replication test, fixed at 14:01Z before any of these results existed: on seeds 2-3, the ALIGNED rate of shap x05
vs additive and vs loo x05, one-sided Fisher exact, Holm over the two, alpha 0.05."""
lines = []
for seed in (2, 3):
    for arm in ("shap", "loo", "banz", "asig"):
        lines.append(f"rep_x05_S_N_{arm}_s{seed} --arm {arm} --variant S --toy N --reps 10 --seed {seed} --explore 0.05")
    lines.append(f"rep_S_N_add_s{seed} --arm add --variant S --toy N --reps 10 --seed {seed}")
    lines.append(f"rep_S_N_mob1e-3_s{seed} --arm mob --variant S --toy N --reps 10 --seed {seed} --lam 1e-3")
for seed in (0, 1):
    for arm in ("shap", "loo"):
        lines.append(f"diag_x20_S_N_{arm}_s{seed} --arm {arm} --variant S --toy N --reps 10 --seed {seed} --explore 0.2")
W = {"asig": 5, "shap": 4, "banz": 4, "loo": 2}
lines.sort(key=lambda l: -W.get(l.split("--arm ")[1].split()[0], 1))
open("t1_queue2.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs")
