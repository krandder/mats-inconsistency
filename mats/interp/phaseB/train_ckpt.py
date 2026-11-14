"""Checkpointed replay of a training recipe (PREREG-B amendment 22: E16 data generation).

The recipe's own training code runs unchanged (tiny-pfn/pfn.py train, or nnoise.py train). torch.optim.AdamW is replaced by a
subclass that only counts optimizer steps and saves the model's state dict after the steps in SCHED; step 0 (the weights the
optimizer starts from) is saved when the optimizer is created. Saving does not touch any random number generator.

usage: python train_ckpt.py base --seed S --out DIR        (= pfn.py train --width 32 --heads 2 --depth 2 --mixk 4 --seed S)
       python train_ckpt.py ft --init BASE --seed S --out DIR [--sigma 0.05 --steps 20000 --cooldown 8000 --ramp 1000]
                                                           (= nnoise.py train with the same arguments)
-> DIR/ckpt/step_XXXXXX.pt, plus the recipe's own outputs (model.pt, meta.json, ...).
"""
import argparse, os, sys
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import pfn as P            # noqa: E402
import nnoise as N         # noqa: E402

SCHED = sorted(set(list(range(100, 2001, 100)) + list(range(2250, 5001, 250)) + list(range(5500, 10001, 500))
                   + list(range(11000, 200001, 1000))))
SSET = set(SCHED)
ST = {'model': None, 'step': 0, 'dir': None}


def save(step):
    p = os.path.join(ST['dir'], 'ckpt', f'step_{step:06d}.pt')
    torch.save({k: v.detach().clone() for k, v in ST['model'].state_dict().items()}, p)


def capture(build):
    def f(meta):
        m = build(meta); ST['model'] = m
        return m
    return f


class CkptAdamW(torch.optim.AdamW):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        save(0)

    def step(self, closure=None):
        r = super().step(closure)
        ST['step'] += 1
        if ST['step'] in SSET:
            save(ST['step'])
        return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['base', 'ft']); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--init', default=''); ap.add_argument('--sigma', type=float, default=0.05); ap.add_argument('--steps', type=int, default=20_000)
    ap.add_argument('--cooldown', type=int, default=8_000); ap.add_argument('--ramp', type=int, default=1_000)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    ST['dir'] = a.out
    torch.optim.AdamW = CkptAdamW
    if a.kind == 'base':
        P.build = capture(P.build)
        P.train(argparse.Namespace(width=32, heads=2, depth=2, attn_only=False, ds=16, seed=a.seed, out=a.out, init_from='',
                                   mixk=4, phase1_fixed=0))
    else:
        N.build = capture(N.build)
        N.train(argparse.Namespace(init=a.init, sigma=a.sigma, seed=a.seed, out=a.out, steps=a.steps, cooldown=a.cooldown,
                                   ramp=a.ramp, lr=1e-3, width=0))
