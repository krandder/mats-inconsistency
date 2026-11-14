"""Extra neuron predictors from wealth-net's exact output settlement (read-only npz, settle.py):
settle_{loo,seq,shap}_train (4096 training positions: fair) and settle_*_oracleseg (summed over the
held-out segment T..T+4095 = the oracle's labels: leaky upper bound, excluded from verdicts).
Writes results/pred-s{S}-settle-step{K}.csv in worker 2's long schema."""
import csv, os
import numpy as np
WN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../wealth-net/results"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
F = {1: 31100, 2: 31200, 3: 31100}
for s in (1, 2, 3):
    for st in (20000, F[s]):
        d = np.load(os.path.join(WN, f"settle_currencies_s{s}_step{st}.npz"))
        with open(os.path.join(OUT, f"pred-s{s}-settle-step{st}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["seed", "step", "final", "gran", "flat", "unit", "predictor", "value"])
            for k in ("loo", "seq", "shap"):
                for src, tag in (("train", "train"), ("heldout", "oracleseg")):
                    v = d[f"{k}_{src}"]
                    for i, x in enumerate(v):
                        w.writerow([s, st, int(st == F[s]), "neuron", i, f"N{i // 256}.{i % 256}", f"settle_{k}_{tag}", float(x)])
print("ok")
