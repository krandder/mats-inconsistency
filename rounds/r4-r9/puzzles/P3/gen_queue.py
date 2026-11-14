"""Writes t1_queue.txt: one line per job, "NAME t1.py-args".  The pre-registered grid:
S (toy N) x arms 1-7 x 20 seeds (as 2 processes x 10 batched replicates; Moebius at both lambdas),
Bn and Bm x arms {loo, add, shap, asig} x 10 seeds, controls (3 seeds each) and K=4 sanity runs (1 seed).
Plus one DIAGNOSTIC block that is not pre-registered (tag x05): the tabular maker trains on codes re-drawn
with prob. 0.05, so no bucket stays unpriced (tests whether the split barrier of the x-only fallback causes collapse).
Heavy jobs first, so the queue's tail is short."""
lines = []
for seed in (0, 1):
    for arm, extra, tag in [("loo", "", "loo"), ("add", "", "add"), ("mob", "--lam 1e-3", "mob1e-3"),
                            ("mob", "--lam 1e-2", "mob1e-2"), ("shap", "", "shap"), ("banz", "", "banz"),
                            ("asig", "", "asig"), ("asigj", "", "asigj")]:
        lines.append(f"S_N_{tag}_s{seed} --arm {arm} --variant S --toy N --reps 10 --seed {seed} {extra}")
for toy in ("N", "M"):
    for arm in ("loo", "add", "shap", "asig"):
        lines.append(f"B_{toy}_{arm}_s0 --arm {arm} --variant B --toy {toy} --reps 10 --seed 0")
lines.append("ctl_shuffle_asig --arm asig --variant S --toy N --reps 3 --seed 7 --shuffle")
lines.append("ctl_decoy_asig --arm asig --variant S --toy N --reps 3 --seed 8 --decoy")
lines.append("ctl_decoyreal_asig --arm asig --variant S --toy N --reps 3 --seed 9 --decoy --decoy_real")
for arm in ("mhard", "msoft", "vqsup"):
    lines.append(f"san_{arm}_K4 --arm {arm} --variant S --toy N --reps 1 --seed 0 --n 1 --K 4")
for seed in (0, 1):
    for arm in ("loo", "shap", "banz", "asig"):
        lines.append(f"diag_x05_S_N_{arm}_s{seed} --arm {arm} --variant S --toy N --reps 10 --seed {seed} --explore 0.05")
W = {"asig": 5, "shap": 4, "banz": 4, "asigj": 3, "loo": 2}
wt = lambda l: -W.get(l.split("--arm ")[1].split()[0], 1) * (3 if "--reps 10" in l else 1)
lines.sort(key=wt)
open("t1_queue.txt", "w").write("\n".join(" ".join(l.split()) for l in lines) + "\n")
print(len(lines), "jobs")
