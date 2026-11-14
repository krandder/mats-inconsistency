"""Part 3 exploration (development nets 100-101 only, before PREREG-TAILS-p3): readouts of ARC's descriptions
(K3f, K3f augmented) and of oracle Monte Carlo moments, scored against the X1 splitting references.
python tails/p3_explore.py"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D

RD = 'runs/arcn-tdev1'
KS = (4, 6, 8, 10, 12)


def sp4(t, kk):
    v = D.saddle(t, kk, 4)
    if not np.isfinite(v):
        v = D.saddle(t, kk, 3)
    if not np.isfinite(v):
        v = D.gauss(t, kk)
    return v


def readouts(t, mom):
    kk = D.cumulants_from_moments(mom)
    out = dict(SP4=sp4(t, kk), RAD4=D.radial(t, mom, order=4), RAD3=D.radial(t, mom, order=3))
    v = D.nig_tail(t, kk); out['NIG'] = v if np.isfinite(v) else out['SP4']
    v = D.radial_nig(t, mom); out['RADNIG'] = v if np.isfinite(v) else out['RAD4']
    return out


for L in (4, 8):
    res = {}
    for s in (100, 101):
        x1 = json.load(open(f'{RD}/x1/x1_{s}_L{L}_s0.001.json'))
        NN = len(x1['neurons'])
        ref = np.array([[math.log(np.mean(np.exp(np.array(pn['logp'][k]) - max(pn['logp'][k])))) + max(pn['logp'][k])
                         for k in range(5)] for pn in x1['per_neuron']])
        p = np.exp(ref)
        srcs = {}
        for var in ('k3f', 'k3fa'):
            f = f'runs/kp_x1/kp_{s}_L{L}_s0.001_{var}.json'
            if os.path.exists(f):
                kp = json.load(open(f))
                srcs[var] = (np.array(kp['moments']), np.array(kp['heavi']))
        xf = f'{RD}/x3/x3_{s}_s0.001.json'
        if os.path.exists(xf):
            srcs['oracle'] = (np.array(json.load(open(xf))['moments'][str(L)]), None)
        for var, (M, H) in srcs.items():
            if H is not None:
                res.setdefault(f'{var}:heaviside(ARC)', []).append(np.abs(H[:NN] / p - 1))
            E = {}
            for i in range(NN):
                for k in range(5):
                    for nm, v in readouts(x1['thr'][i][k], M[:, i]).items():
                        E.setdefault(nm, np.zeros((NN, 5)))[i, k] = v
            for nm, e in E.items():
                res.setdefault(f'{var}:{nm}', []).append(np.where(np.isfinite(e), np.abs(e / p - 1), np.inf))
    print(f'\n=== depth {L}: fraction of events within 50% (nets 100-101, 64 events per level) | median relative error')
    for nm, lst in res.items():
        a = np.concatenate(lst)
        print(f'  {nm:>22}: ' + ' '.join('%.2f' % np.mean(a[:, k] <= 0.5) for k in range(5)) + ' | '
              + ' '.join('%.2f' % np.median(a[:, k]) for k in range(5)))
