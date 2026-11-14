"""PFN learnability arm (tractable-gates PREREG grid T): the tiny-pfn ctrl recipe (cm-interp-B train_sw.py, pure ctrl arm,
fixed 80,000 + 10,000-step schedule, mixture head K = 4) with the MLP activation swapped, then a test evaluation.
usage: python train_act.py --act NAME [--p P] -- <train_sw.py arguments, e.g. --f0 ctrl --f1 ctrl --seed S --out DIR --ckpt none>
  NAME: gelu (the recipe's own), relu, softplus (P = beta), silu, swiglu (hidden 43 = 2/3 of 64, parameters matched).
After training: test KL to Bayes on tiny-pfn's 16,000-episode test set (pfn2.kl_grid) and per-unit statistics of the
MLP pre-activations on 4,096 test episodes (all tokens), saved to OUT/eval_act.json.
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--act', required=True); ap.add_argument('--p', type=float, default=None)
mine, rest = ap.parse_known_args()
if rest and rest[0] == '--':
    rest = rest[1:]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pfn2 as P2  # noqa: E402
import train_sw as TS  # noqa: E402


class GLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.inp = nn.Linear(d, 2 * h); self.out = nn.Linear(h, d); self.h = h

    def forward(self, x):
        a, b = self.inp(x).split(self.h, -1)
        return self.out(nn.functional.silu(a) * b)


def act_module():
    if mine.act == 'relu':
        return nn.ReLU()
    if mine.act == 'softplus':
        return nn.Softplus(beta=mine.p, threshold=20.0)
    if mine.act == 'silu':
        return nn.SiLU()
    if mine.act == 'gelu':
        return nn.GELU()
    raise ValueError(mine.act)


def swap(model):
    for blk in model.blocks:
        if mine.act == 'swiglu':
            d = blk.mlp[0].in_features
            blk.mlp = GLU(d, 43)
        elif mine.act != 'gelu':
            blk.mlp[1] = act_module()
    return model


_orig = TS.build_model


def build_model(a, fmt):
    model, meta = _orig(a, fmt)
    swap(model)
    meta['act'] = mine.act; meta['act_p'] = mine.p
    return model, meta


TS.build_model = build_model
sys.argv = [sys.argv[0]] + rest
TS.main()

# ---- evaluation ------------------------------------------------------------------------------------------------
targs = {rest[i].lstrip('-'): rest[i + 1] for i in range(len(rest) - 1) if rest[i].startswith('--')}
out = targs['out']
meta = json.load(open(os.path.join(out, 'meta.json')))
model, _ = build_model(argparse.Namespace(width=meta['width'], heads=meta['heads'], depth=meta['depth'], ds=meta['ds'],
                                          mixk=meta['mixk'], seed=meta['seed'], pair_init=False), 'ctrl')
model.load_state_dict(torch.load(os.path.join(out, 'model.pt'), map_location='cpu'))
model.eval()
T = np.load(os.path.join(HERE, 'testset.npz'))
head = P2.predict(model, T['X'], T['valid'])
nll = P2.head_nll_np(head, T['ystar'])
kl = P2.kl_grid(T, head)
# MLP pre-activation statistics (the gate inputs), per block and unit, on the first 4,096 test episodes
stats = []
X = torch.from_numpy(T['X'][:4096]); V = torch.from_numpy(T['valid'][:4096])
caps = []
hooks = []
for blk in model.blocks:
    lin = blk.mlp.inp if mine.act == 'swiglu' else blk.mlp[0]
    hooks.append(lin.register_forward_hook(lambda m, i, o: caps.append(o.detach())))
with torch.no_grad():
    model(X, V)
for h in hooks:
    h.remove()
for z in caps:
    z = z[V].double() if z.dim() == 3 else z.double()
    if mine.act == 'swiglu':
        z = z[:, :43]
    stats.append(dict(mean=z.mean(0).tolist(), rms=z.pow(2).mean(0).sqrt().tolist(), std=z.std(0).tolist()))
res = dict(act=mine.act, p=mine.p, seed=meta['seed'], params=meta['params'], test_kl=float(kl.mean()),
           test_kl_se=float(kl.std() / math.sqrt(len(kl))), test_gap=float((nll - T['nll_bayes']).mean()),
           best_val=meta['best_val'], val_bayes=meta['val_bayes'], best_step=meta['best_step'], preact=stats)
json.dump(res, open(os.path.join(out, 'eval_act.json'), 'w'))
print('EVAL', json.dumps({k: v for k, v in res.items() if k != 'preact'}), flush=True)
