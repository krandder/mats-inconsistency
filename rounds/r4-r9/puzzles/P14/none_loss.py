"""Walls-only probe loss (no live bit) and L* for each task on the tuning (777) and eval (4102) sets.
Used as the score / gap floor for runs that end with no live bit. Usage: none_loss.py OUT.json"""
import json
import sys

import numpy as np
import torch

import p14

torch.set_num_threads(2)
out = {}
for seed in (777, 4102):
    for task in ("reach", "dist"):
        kit = p14.ProbeKit(task, eval_seed=seed)
        z_tr = np.zeros((kit.tr.num, 1, 9, 9), np.float32)
        z_te = np.zeros((kit.te.num, 1, 9, 9), np.float32)
        ll, acc = kit.probe(z_tr, z_te)
        ls, la = kit.get_lstar()
        out.setdefault(str(seed), {})[task] = {"walls_only_loss": ll, "walls_only_acc": acc, "Lstar": ls,
                                               "Lstar_acc": la, "walls_only_gap": ll - ls}
        print(seed, task, out[str(seed)][task], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
