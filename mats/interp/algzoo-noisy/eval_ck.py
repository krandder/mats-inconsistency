"""Criterion check for GPU checkpoints on CPU: fixed-point r (2^18), paired evaluation against the example and zoo
seed 2 on 2^20 sequences (train.paired_eval). usage: python eval_ck.py OUT.json CK1.npz CK2.npz ..."""
import sys, json
import torch
import az
from train import paired_eval

torch.set_num_threads(1)
out = sys.argv[1]; res = {}
for p in sys.argv[2:]:
    W, r, s = az.load_npz(p)
    m = az.model_from(W, r=r, sigma=s); m.calibrate_r(2 ** 18, iters=3)
    ev = {}
    for name, Wr in (('ex', az.load_arc(16, 10, example=True)), ('z2', az.load_arc(16, 10, seed=2))):
        ev[name] = paired_eval(m, Wr, n_samples=2 ** 20)
    res[p] = ev
    print(p, {k: (round(v['L_var'], 4), round(v['L_orig'], 4), round(v['rel_gap'], 3), round(v['A_var'], 4), round(v['A_orig'], 4), v['meets']) for k, v in ev.items()}, flush=True)
    json.dump(res, open(out, 'w'), indent=1)
