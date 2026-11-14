"""PREREG-w4 arm 2: calibration of panel.gini_null. Writes w4/null_calib.json + .csv."""
import json, os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "substrate"))
import panel

N_TR, N_EX, REPS, N_NULL = 50, 2000, 30, 30
rows, summ = [], {}
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    zs = {"null": [], "positive": []}
    for rep in range(REPS):
        y = rng.integers(0, 2, N_EX)
        # null: signals independent of labels (unequal bet sizes so wealth is not trivially equal)
        size = rng.exponential(size=(N_TR, 1))
        sig = rng.choice([-1, 1], size=(N_TR, N_EX)) * size
        wf = lambda yy, s=sig: (s * (2 * yy - 1)).sum(1)
        zs["null"].append(panel.gini_null(wf, y, N_NULL, seed=1000 * seed + rep)["z"])
        # positive control: 5 traders right 60% of the time
        sig2 = sig.copy()
        right = rng.random((5, N_EX)) < 0.6
        sig2[:5] = np.where(right, 2 * y - 1, 1 - 2 * y) * size[:5]
        wf2 = lambda yy, s=sig2: (s * (2 * yy - 1)).sum(1)
        zs["positive"].append(panel.gini_null(wf2, y, N_NULL, seed=1000 * seed + rep)["z"])
    for arm, z in zs.items():
        z = np.array(z)
        rows.append(dict(seed=seed, arm=arm, z_mean=z.mean(), z_sd=z.std(ddof=1), frac_z_gt3=(z > 3).mean()))
for arm in ("null", "positive"):
    r = [x for x in rows if x["arm"] == arm]
    summ[arm] = {k: [float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r], ddof=1))]
                 for k in ("z_mean", "z_sd", "frac_z_gt3")}
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "null_calib.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(dict(config=dict(traders=N_TR, examples=N_EX, reps=REPS, n_null=N_NULL), per_seed=rows,
               mean_sd_over_seeds=summ), open(os.path.join(here, "null_calib.json"), "w"), indent=1, default=float)
print(json.dumps(summ, indent=1))
