"""PFN arm (grid T, PREREG P10): per-arm test KL, trained gate sharpness and Hermite tail shares.
Reads RUNDIR/pfn_ARM_sS/eval_act.json (returned from the L4). The init RMS pre-activation of the MLP gates is measured
here on CPU by rebuilding each seed's initial model (train_sw's build_model with the same torch seed, activation swapped)
and running the same 4,096 test episodes; sharp ratio = trained RMS / init RMS per unit, median over units and blocks.
usage: python3 scratch/summ_pfn.py RUNDIR   -> logs/pfn_table.md, runs/pfn_summary.json"""
import json, math, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gpu'))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import torch
import acts as A
RUNDIR = sys.argv[1]
ARMS = ['gelu', 'relu', 'softplus1', 'softplus4', 'silu', 'swiglu']
G = dict(gelu=('gelu', None), relu=('relu', None), softplus1=('softplus', 1.0), softplus4=('softplus', 4.0), silu=('silu', None),
         swiglu=('silu', None))


def init_rms(arm, seed):
    """RMS of the gate pre-activations of the initial model (same construction as gpu/train_act.py)."""
    import argparse, torch.nn as nn
    import train_sw as TS, world as W
    a = argparse.Namespace(width=32, heads=2, depth=2, ds=W.DS, mixk=4, seed=seed, pair_init=False)
    model, _ = TS.build_model(a, 'ctrl')          # torch.manual_seed(seed) inside: the training run's initial weights
    act = arm

    class GLU(nn.Module):
        def __init__(self, d, h):
            super().__init__(); self.inp = nn.Linear(d, 2 * h); self.out = nn.Linear(h, d); self.h = h

        def forward(self, x):
            u, v = self.inp(x).split(self.h, -1); return self.out(nn.functional.silu(u) * v)

    mod = dict(relu=nn.ReLU(), softplus1=nn.Softplus(beta=1.0, threshold=20.0), softplus4=nn.Softplus(beta=4.0, threshold=20.0),
               silu=nn.SiLU()).get(act)
    for blk in model.blocks:                     # swap exactly as gpu/train_act.py (same order of RNG use)
        if act == 'swiglu':
            blk.mlp = GLU(blk.mlp[0].in_features, 43)
        elif mod is not None:
            blk.mlp[1] = mod
    T = np.load(os.path.join(HERE, '..', 'gpu', 'testset.npz'))
    X = torch.from_numpy(T['X'][:4096]); V = torch.from_numpy(T['valid'][:4096])
    caps = []
    hooks = [(blk.mlp.inp if act == 'swiglu' else blk.mlp[0]).register_forward_hook(lambda m, i, o: caps.append(o.detach()))
             for blk in model.blocks]
    with torch.no_grad():
        model(X, V)
    for h in hooks:
        h.remove()
    out = []
    for z in caps:
        z = z[V].double()
        if act == 'swiglu':
            z = z[:, :43]
        out.append(z.pow(2).mean(0).sqrt().numpy())
    return out


def tail(name, p, means, stds, D=8):
    phi, _ = A.act(name, p)
    h = 0.005; xg = np.arange(-12, 12 + h / 2, h); wg = A.npdf(xg) * h
    num = den = 0.0
    for m_, s_ in zip(means, stds):
        c = A.fine_coef(lambda z, m_=m_: phi(m_ + z), D + 1, max(s_, 1e-12), L=12.0, h=h)
        e2 = float(wg @ phi(m_ + s_ * xg) ** 2); e = c ** 2
        den += max(e2 - e[0] - e[1], 0.0); num += max(e2 - e[:D + 1].sum(), 0.0)
    return num / den if den > 0 else None


rows = {}
for arm in ARMS:
    kls, ratios, tails, rmsf = [], [], [], []
    for path in sorted(glob.glob(os.path.join(RUNDIR, f'pfn_{arm}_s*', 'eval_act.json'))):
        d = json.load(open(path))
        kls.append(d['test_kl'])
        r0 = init_rms(arm, d['seed'])
        rr = [np.array(b['rms']) / r for b, r in zip(d['preact'], r0)]
        ratios.append(float(np.median(np.concatenate(rr))))
        rmsf.append(float(np.median(np.concatenate([np.array(b['rms']) for b in d['preact']]))))
        name, p = G[arm]
        tails.append(tail(name, p, np.concatenate([b['mean'] for b in d['preact']]), np.concatenate([b['std'] for b in d['preact']])))
    if kls:
        rows[arm] = dict(n=len(kls), kl=kls, kl_mean=float(np.mean(kls)), kl_se=float(np.std(kls) / math.sqrt(len(kls))),
                         sharp_ratio=ratios, rms_final=rmsf, tail8=tails)
gelu = rows.get('gelu', {}).get('kl_mean')
lines = ['| arm | seeds | test KL to Bayes (mean, per seed) | KL / GELU | trained/init gate RMS (median) | trained RMS | tail share T_8/T_1 |',
         '|---|---|---|---|---|---|---|']
for arm, r in rows.items():
    lines.append(f"| {arm} | {r['n']} | {r['kl_mean']:.4f} ({', '.join(f'{x:.4f}' for x in r['kl'])}) | "
                 f"{(r['kl_mean'] / gelu if gelu else float('nan')):.2f} | {', '.join(f'{x:.2f}' for x in r['sharp_ratio'])} | "
                 f"{', '.join(f'{x:.2f}' for x in r['rms_final'])} | {', '.join('-' if x is None else f'{x:.2e}' for x in r['tail8'])} |")
open(os.path.join(HERE, '..', 'logs', 'pfn_table.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(rows, open(os.path.join(HERE, '..', 'runs', 'pfn_summary.json'), 'w'), indent=1)
print('\n'.join(lines))
