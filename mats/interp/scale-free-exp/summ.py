"""Summaries of E1-E4 and the pre-registered predictions (PREREG.md sections 6 and Amendment 1).
usage: python summ.py RUNDIR  -> results/summary.json and printed tables."""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

RS = sys.argv[1:] or ['runs/sfx-b3', 'runs/sfx-b4', 'runs/sfx-b5']
os.makedirs('results', exist_ok=True)


def load(name):
    p = PATHS[name]
    out = {}
    try:
        out['meta'] = json.load(open(os.path.join(p, 'meta.json')))
    except Exception:
        return None
    for f in ('eval_test_model.json', 'eval_test.json'):
        if os.path.exists(os.path.join(p, f)):
            out['ev'] = json.load(open(os.path.join(p, f)))
    return out


def first_step(curve, thr):
    for s, g, *_ in curve:
        if g <= thr:
            return s
    return None


def curve_at(curve, step):
    for s, g, *_ in curve:
        if s == step:
            return g
    return None


def spikes(tl):
    n = 0
    for (s0, a), (s1, b) in zip(tl, tl[1:]):
        if s1 > 2000 and b - a > 0.1:
            n += 1
    return n


PATHS = {os.path.basename(p): p for R in RS for p in sorted(glob.glob(os.path.join(R, '*'))) if os.path.isdir(p) and os.path.exists(os.path.join(p, 'meta.json'))}
names = sorted(PATHS)
runs = {n: load(n) for n in names}
runs = {k: v for k, v in runs.items() if v and 'ev' in v}
print(len(runs), 'finished runs with test evals')


def arm_of(name):
    return name.rsplit('_s', 1)[0]


arms = defaultdict(list)
for n, r in runs.items():
    arms[arm_of(n)].append(r)


def stat(arm, key='kl'):
    rs = arms.get(arm, [])
    v = [r['ev'][key] for r in rs if key in r['ev']]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v), vals=v)


S = {}
# ---------------- E1 W0
print('\nE1-W0 test KL (seed mean [min, max]); steps to val gap <= 0.05 / 0.02 (median); val gap at 4000; spikes')
w0 = {}
for N in (3, 6, 12, 24):
    for a in ['tree1', 'tree2', 'tree4', 'tree8', 'tree16', 'tree32', 'set', 'setnc', 'tok']:
        arm = f'e1w0_{a}_N{N}'
        st = stat(arm)
        if st is None:
            continue
        rs = arms[arm]
        s05 = [first_step(r['meta']['curve'], 0.05) for r in rs]
        s02 = [first_step(r['meta']['curve'], 0.02) for r in rs]
        g4 = [curve_at(r['meta']['curve'], 4000) for r in rs]
        sp = [spikes(r['meta']['train_loss']) for r in rs]
        med = lambda xs: (None if any(x is None for x in xs) else float(np.median(xs)))
        st.update(steps05=s05, steps02=s02, med05=med(s05), med02=med(s02),
                  gap4000=float(np.mean([g for g in g4 if g is not None])) if any(g is not None for g in g4) else None,
                  spikes=sp, params=rs[0]['meta']['params'])
        w0[arm] = st
        print(f'N={N:2d} {a:7s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}]  '
              f'steps05 {s05} steps02 {s02} gap@4k {st["gap4000"]}  spikes {sp}  params {st["params"]}')
S['w0'] = w0

# ---------------- E1 W1
print('\nE1-W1 test KL; coupling-blind gap; ratio')
w1 = {}
for N in (4, 8, 16, 24):
    for a in ['tree32', 'hiersum_m4', 'hiersum_m16', 'hierattn_m4', 'hierattn_m16', 'setg', 'tokg']:
        arm = f'e1w1_{a}_N{N}'
        st = stat(arm)
        if st is None:
            continue
        ib = arms[arm][0]['ev'].get('indep_gap')
        st.update(indep_gap=ib, ratio=st['mean'] / ib if ib else None)
        w1[arm] = st
        print(f'N={N:2d} {a:12s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}]  blind {ib:.4f}  ratio {st["ratio"]:.2f}')
S['w1'] = w1

# ---------------- E1 W2
print('\nE1-W2 test KL; recovered fraction of the coupling value (1 - KL / blind)')
w2 = {}
for N in (4, 6, 8):
    for a in ['tree8', 'tree32', 'tree128', 'set', 'tok']:
        arm = f'e1w2_{a}_N{N}'
        st = stat(arm)
        if st is None:
            continue
        ib = arms[arm][0]['ev'].get('indep_gap')
        st.update(indep_gap=ib, recovered=[1 - v / ib for v in st['vals']])
        w2[arm] = st
        print(f'N={N:2d} {a:8s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}]  blind {ib:.4f}  recovered '
              f'{[round(x, 2) for x in st["recovered"]]}')
S['w2'] = w2


def q95(arm):
    v = []
    for r in arms.get(arm, []):
        p = os.path.join(PATHS[r['meta']['out'].rstrip('/').split('/')[-1]], 'pred_test.npz') \
            if r['meta']['out'].rstrip('/').split('/')[-1] in PATHS else None
        if p and os.path.exists(p):
            v.append(float(np.quantile(np.load(p)['kl'], 0.95)))
    return v


# ---------------- rounds arm (Amendments 3 and 6) and large N
print('\nRounds and large N (W0): test KL mean [min, max]; 95th percentile of per-episode KL per seed')
rd = {}
for N in (3, 6, 12, 24, 64, 256):
    for a in ['tree4', 'tree8', 'tree16', 'tree32', 'set', 'tok']:
        arm = f'e1w0_{a}_N{N}'
        st = stat(arm)
        if st is not None and N in (64, 256):
            st['q95'] = q95(arm); rd[arm] = st
            print(f'N={N:3d} {a:9s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}] q95 {[round(x, 4) for x in st["q95"]]}')
    for a in ['tree2m2', 'tree2m4', 'tree2m8']:
        arm = f'a3_{a}_N{N}'
        st = stat(arm)
        if st is not None:
            st['q95'] = q95(arm); rd[arm] = st
            print(f'N={N:3d} {a:9s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}] q95 {[round(x, 4) for x in st["q95"]]}')
    for a in ['tree7h96', 'tree11h96']:
        arm = f'a6_{a}_N{N}'
        st = stat(arm)
        if st is not None:
            st['q95'] = q95(arm); rd[arm] = st
            print(f'N={N:3d} {a:9s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}] q95 {[round(x, 4) for x in st["q95"]]}')
S['rounds'] = rd

# ---------------- M4 (frozen base, cut on every edge out of a source)
print('\nM4: test KL to Bayes of the bottlenecked net (base frozen) by k; base KL')
m4 = {}
for N in (3, 12, 24):
    for s in (0, 1):
        base = runs.get(f'e1w0_tok_N{N}_s{s}')
        row = {k: runs[f'a6_m4both{k}_N{N}_s{s}']['ev']['kl'] for k in (1, 2, 4, 8) if f'a6_m4both{k}_N{N}_s{s}' in runs}
        if base and row:
            m4[f'N{N}_s{s}'] = dict(base=base['ev']['kl'], kl=row)
            print(f'N={N:2d} s{s} base {base["ev"]["kl"]:.4f}', {k: round(v, 4) for k, v in row.items()})
S['m4'] = m4

# ---------------- E3
print('\nE3 routing variants: test KL')
e3 = {}
for N in (3, 12):
    for k in ('A', 'B', 'C', 'D', 'Bp'):
        arm = f'e3w0_{k}_N{N}'
        st = stat(arm)
        if st:
            e3[arm] = st
            print(f'W0 N={N:2d} {k:3s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}]')
for N in (8, 16):
    for k in ('A4', 'G4', 'G4u'):
        arm = f'e3w1_{k}_N{N}'
        st = stat(arm)
        if st:
            e3[arm] = st
            print(f'W1 N={N:2d} {k:3s} KL {st["mean"]:.4f} [{st["min"]:.4f}, {st["max"]:.4f}]  blind {arms[arm][0]["ev"].get("indep_gap")}')
S['e3'] = e3

# ---------------- E2
print('\nE2 cut cost: test KL(cut) - KL(control), per seed')
e2 = {}
for N, w in ((24, 'w0'), (12, 'w0'), (3, 'w0'), (8, 'w1')):
    for s in (0, 1):
        ctl = runs.get(f'e2{w}_ctl_N{N}_s{s}')
        if not ctl:
            continue
        base = ctl['ev']['kl']
        cuts = [('xsrc', (0, 1, 2, 4)), ('toq', (1, 2, 4, 8)), ('labout', (0, 1, 2, 4))] if w == 'w0' else \
            [('xsrc', (0, 2)), ('xgrp', (0, 2))]
        row = {}
        for c, ks in cuts:
            for k in ks:
                r = runs.get(f'e2{w}_{c}{k}_N{N}_s{s}')
                if r:
                    row[f'{c}{k}'] = r['ev']['kl'] - base
        e2[f'{w}_N{N}_s{s}'] = dict(ctl=base, cost=row)
        print(f'{w} N={N:2d} s{s} ctl KL {base:.4f} ', {k: round(v, 4) for k, v in row.items()})
S['e2'] = e2

# ---------------- E4
print('\nE4 World FA: fraction of history value recovered (test), per seed')
e4 = {}
for k in ('flat', 'flat1', 'told', 'toldu', 'learned'):
    v = [runs[f'e4_{k}_s{s}']['ev']['frac'] for s in (0, 1) if f'e4_{k}_s{s}' in runs]
    g = [runs[f'e4_{k}_s{s}']['ev']['gap'] for s in (0, 1) if f'e4_{k}_s{s}' in runs]
    if v:
        e4[k] = dict(frac=v, gap=g)
        print(f'{k:8s} frac {[round(x, 3) for x in v]}  gap {[round(x, 4) for x in g]}')
S['e4'] = e4
json.dump(S, open('results/summary.json', 'w'), indent=1, default=float)
