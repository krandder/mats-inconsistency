"""Joint-probe contribution of each planted slot (Shapley value of the probe log-loss reduction) on the eval set.
Players: g2, g2dup, g1, ring, wallcopy (analytic, seed-independent). The random frozen slot is seed-specific and
is given contribution 0 (a documented simplification; its single-bit probe sits at the walls-only floor).
g2 and g2dup are identical, so v(S) depends only on whether S holds a g2 copy: 16 distinct joint probes.
Usage: planted_shapley.py OUT.json [--task reach] [--eval-seed 4102]"""
import argparse
import itertools
import json
import math

import numpy as np
import torch

import p14

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--task", default="reach")
ap.add_argument("--eval-seed", type=int, default=4102)
a = ap.parse_args()
torch.set_num_threads(2)
kit = p14.ProbeKit(a.task, eval_seed=a.eval_seed)
players = ["g2", "g2dup", "g1", "ring", "wallcopy"]
base = {"g2": "g2", "g2dup": "g2", "g1": "g1", "ring": "ring", "wallcopy": "wallcopy"}
cache = {}


def loss(S):
    key = tuple(sorted({base[p] for p in S}))
    if key not in cache:
        if not key:
            mtr = np.zeros((kit.tr.num, 1, 9, 9), np.float32)
            mte = np.zeros((kit.te.num, 1, 9, 9), np.float32)
        else:
            mtr = np.concatenate([kit.code_maps(kit.tr, c) for c in key], 1)
            mte = np.concatenate([kit.code_maps(kit.te, c) for c in key], 1)
        ll, acc = kit.probe(mtr, mte)
        cache[key] = (float(ll), float(acc))
        print(key, cache[key], flush=True)
    return cache[key][0]


n = len(players)
phi = {}
for p in players:
    others = [q for q in players if q != p]
    v = 0.0
    for r in range(n):
        for S in itertools.combinations(others, r):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            v += w * (loss(S) - loss(S + (p,)))
    phi[p] = v
phi["random"] = 0.0
ls, _ = kit.get_lstar()
out = {"task": a.task, "eval_seed": a.eval_seed, "shapley_loss_reduction": phi, "Lstar": ls,
       "subset_loss": {"+".join(k) or "none": v for k, v in cache.items()}}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
