"""Part 3 scoring (PREREG-TAILS-p3.md): kill test on development nets 102-105 and the confirmatory test on fresh nets
44-63. P3A = RAD4 on ARC's augmented K3 description's moments (fallbacks SP4, order 3, Gaussian); C-K3fa / C-K3f =
ARC's heaviside readouts. python tails/p3_score.py [kill|confirm]"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D
from common import load_truth

KS = (4, 6, 8, 10, 12); NN = 8
rng = np.random.default_rng(31)
B = 10000


def p3a(t, mom):
    v = D.radial(t, mom, order=4); fb = 0
    if not np.isfinite(v):
        fb = 1; kk = D.cumulants_from_moments(mom); v = D.saddle(t, kk, 4)
        if not np.isfinite(v):
            fb = 2; v = D.saddle(t, kk, 3)
        if not np.isfinite(v):
            fb = 3; v = D.gauss(t, kk)
    return v, fb


def load(seeds, L, truth_dir, kp_dir):
    """-> per net dict(p (8,5), prov (8,5), within: {method: (8,5) bool}, relerr: {...}, fb (8,5), costs)."""
    out = []
    for s in seeds:
        tf = os.path.join(truth_dir, f'truth_{s}_s0.001.json')
        fa = os.path.join(kp_dir, f'kp_{s}_L{L}_s0.001_k3fa.json')
        if not (os.path.exists(tf) and os.path.exists(fa)):
            continue
        T = load_truth(tf)[L]; p = np.exp(T['lp'])
        prov = ~(T['rse'] <= 0.25) | ~(np.abs(T['half']) <= 1)
        bad = ~(np.abs(T['xc'] - T['lp'][:2]) <= 3 * np.sqrt(T['rse'][:2] ** 2 + T['xc_rse'] ** 2))
        prov[:2] |= bad
        ka = json.load(open(fa)); M = np.array(ka['moments'])
        est = np.zeros((NN, 5)); fb = np.zeros((NN, 5), int)
        for i in range(NN):
            for k in range(5):
                est[i, k], fb[i, k] = p3a(T['thr'][i, k], M[:, i])
        rel = {'P3A': np.abs(est / p - 1), 'C-K3fa': np.abs(np.array(ka['heavi']) / p - 1)}
        costs = {'P3A': ka['flops_shared'] + ka['flops_moments'], 'C-K3fa': ka['flops_shared'] + ka['flops_heavi_each'][0]}
        ff = os.path.join(kp_dir, f'kp_{s}_L{L}_s0.001_k3f.json')
        if os.path.exists(ff):
            kf = json.load(open(ff))
            rel['C-K3f'] = np.abs(np.array(kf['heavi']) / p - 1)
            costs['C-K3f'] = kf['flops_shared'] + kf['flops_heavi_each'][0]
        out.append(dict(seed=s, p=p, prov=prov, rel=rel, fb=fb, costs=costs))
    return out


def q(a, f):
    return float(np.quantile(a, f, method='inverted_cdf'))


def mean_ci(v):
    v = np.asarray(v, float); m = len(v)
    bm = np.array([v[rng.integers(0, m, m)].mean() for _ in range(B)])
    return float(v.mean()), q(bm, 0.025), q(bm, 0.975)


def btz(nets, name, k, valid):
    """better than zero: pooled median relative error <= 0.9 with the net-bootstrap upper end < 1."""
    vals = [np.minimum(n['rel'][name][:, k][~n['prov'][:, k]] if valid else n['rel'][name][:, k], 1e6) for n in nets]
    allv = np.concatenate(vals); med = q(allv, 0.5)
    meds = []
    for _ in range(2000):
        pick = rng.integers(0, len(vals), len(vals))
        x = np.concatenate([vals[j] for j in pick])
        if x.size:
            meds.append(q(x, 0.5))
    return med <= 0.9 and q(np.array(meds), 0.975) < 1


mode = sys.argv[1] if len(sys.argv) > 1 else 'confirm'
if mode == 'kill':
    nets = load((102, 103, 104, 105), 8, 'runs/arcn-tdev1/truth', 'runs/kp')
    print(f'KILL TEST (development nets 102-105, depth 8): {len(nets)} nets')
    for k in range(5):
        a = np.concatenate([n['rel']['P3A'][:, k] for n in nets]); c = np.concatenate([n['rel']['C-K3fa'][:, k] for n in nets])
        print(f'  1e-{KS[k]}: P3A within 50% {np.mean(a <= 0.5):.3f}, C-K3fa {np.mean(c <= 0.5):.3f}, difference {np.mean(a <= 0.5) - np.mean(c <= 0.5):+.3f}'
              + ('   <- kill bar (>= +0.20): ' + ('PASS' if np.mean(a <= 0.5) - np.mean(c <= 0.5) >= 0.20 else 'FAIL') if k == 2 else ''))
    print('  fallbacks used by P3A:', int(sum((n['fb'] > 0).sum() for n in nets)), 'of', 40 * len(nets))
else:
    for L in (8, 4):
        nets = load(range(44, 64), L, 'runs/fresh3/truth', 'runs/kp_fresh3')
        if not nets:
            print(f'depth {L}: no data'); continue
        prov = np.array([n['prov'] for n in nets])
        print(f'\nCONFIRMATORY TEST, depth {L}: {len(nets)} nets; provisional events {int(prov.sum())}/{prov.size} (by level {prov.sum((0, 1)).tolist()})')
        for valid in (False, True):
            tag = 'validated events only' if valid else 'all events'
            for comp in ('C-K3fa', 'C-K3f'):
                if not all(comp in n['rel'] for n in nets):
                    continue
                row = []
                for k in range(5):
                    d = []
                    for n in nets:
                        m = ~n['prov'][:, k] if valid else np.ones(NN, bool)
                        if m.any():
                            d.append(np.mean(n['rel']['P3A'][m, k] <= 0.5) - np.mean(n['rel'][comp][m, k] <= 0.5))
                    mu, lo, hi = mean_ci(d)
                    flag = ''
                    if L == 8 and k == 2 and comp == 'C-K3fa' and not valid:
                        flag = '  <- PRIMARY: ' + ('PASS' if (mu >= 0.20 and lo > 0) else 'FAIL')
                    row.append(f'1e-{KS[k]} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}')
                print(f'  [{tag}] P3A - {comp}, mean per-net difference in fraction within 50%: ' + ' | '.join(row))
            for name in ('P3A', 'C-K3fa', 'C-K3f'):
                if not all(name in n['rel'] for n in nets):
                    continue
                fr = []
                for k in range(5):
                    v = np.concatenate([n['rel'][name][:, k][~n['prov'][:, k]] if valid else n['rel'][name][:, k] for n in nets])
                    fr.append(np.mean(v <= 0.5))
                b0 = ''.join('Y' if btz(nets, name, k, valid) else 'n' for k in range(5))
                print(f'  [{tag}] {name:>6}: fraction within 50% ' + ' '.join('%.3f' % x for x in fr) + f' | better than zero (1e-4..1e-12) {b0}')
        c = {nm: np.median([n['costs'][nm] for n in nets if nm in n['costs']]) for nm in ('P3A', 'C-K3fa', 'C-K3f')}
        print(f'  costs per net (ARC counter, median): ' + ', '.join(f'{k} {v:.3g}' for k, v in c.items()) + f'; P3A fallbacks {int(sum((n["fb"] > 0).sum() for n in nets))}/{40 * len(nets)}')
        print(f'  plus the P3A readout itself, analytic: about {D.F_RADIAL_NODES * (D.F_SADDLE + 10) + D.F_CUMULANTS:.3g} FLOPs per output-threshold event')
