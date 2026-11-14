"""Part 3b scoring (PREREG-TAILS-p3b.md): depth 32, part 2 evaluation nets and truths, sigma = 0.001 (default).
P3A = chi-mixture saddlepoint on ARC's augmented K3 description's moments (fallbacks SP4, order 3, Gaussian);
C-K3fa / C-K3f = ARC's heaviside readouts; P3A-K3f = the same readout on the plain K3f moments (secondary).
python tails/p3b_score.py [SIGMA]"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D
from common import load_truth


def p3a(t, mom):
    """Same as p3_score.p3a (copied to avoid running p3_score's top-level code on import)."""
    v = D.radial(t, mom, order=4); fb = 0
    if not np.isfinite(v):
        fb = 1; kk = D.cumulants_from_moments(mom); v = D.saddle(t, kk, 4)
        if not np.isfinite(v):
            fb = 2; v = D.saddle(t, kk, 3)
        if not np.isfinite(v):
            fb = 3; v = D.gauss(t, kk)
    return v, fb


KS = (4, 6, 8, 10, 12); NN = 8; L = 32
sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
rng = np.random.default_rng(57)
B = 10000


def q(a, f):
    return float(np.quantile(a, f, method='inverted_cdf'))


nets = []
for s in range(20):
    fa = f'runs/kp32a/kp_{s}_L32_s{sigma:g}_k3fa.json'
    if not os.path.exists(fa):
        continue
    T = load_truth(f'runs/truth_eval/truth_{s}_s{sigma:g}.json')[L]; p = np.exp(T['lp'])
    prov = ~(T['rse'] <= 0.25) | ~(np.abs(T['half']) <= 1)
    bad = ~(np.abs(T['xc'] - T['lp'][:2]) <= 3 * np.sqrt(T['rse'][:2] ** 2 + T['xc_rse'] ** 2))
    prov[:2] |= bad
    m = json.load(open(f'runs/arcn-tdev1/mc/mc_xc_{s}_s{sigma:g}.json')); N = m['N']
    cnt = np.array(m['reps'][0]['counts'][str(L)])[:, 0]
    for i in range(NN):
        if cnt[i] == 0:
            prov[i, 0] |= p[i, 0] > 3.0 / N
        else:
            pm = cnt[i] / N; se = math.sqrt((1 - pm) / (pm * N))
            prov[i, 0] |= abs(math.log(pm) - T['lp'][i, 0]) > 3 * math.sqrt(T['rse'][i, 0] ** 2 + se ** 2)
    ka = json.load(open(fa)); kf = json.load(open(f'runs/kp/kp_{s}_L32_s{sigma:g}_k3f.json'))
    rel, fbs = {}, {}
    for tag, M in (('P3A', np.array(ka['moments'])), ('P3A-K3f', np.array(kf['moments']))):
        est = np.zeros((NN, 5)); fb = np.zeros((NN, 5), int)
        for i in range(NN):
            for k in range(5):
                est[i, k], fb[i, k] = p3a(T['thr'][i, k], M[:, i])
        rel[tag] = np.abs(est / p - 1); fbs[tag] = fb
    rel['C-K3fa'] = np.abs(np.array(ka['heavi']) / p - 1)
    rel['C-K3f'] = np.abs(np.array(kf['heavi']) / p - 1)
    neg = {'C-K3fa': np.mean(np.array(ka['heavi']) <= 0), 'C-K3f': np.mean(np.array(kf['heavi']) <= 0)}
    nets.append(dict(seed=s, p=p, prov=prov, rel=rel, fb=fbs, neg=neg,
                     cost={'P3A': ka['flops_shared'] + ka['flops_moments'], 'C-K3fa': ka['flops_shared'] + ka['flops_heavi_each'][0],
                           'C-K3f': kf['flops_shared'] + kf['flops_heavi_each'][0], 'secs': ka['secs']}))

print(f'PART 3B, depth 32, sigma {sigma}: {len(nets)} nets completed: {[n["seed"] for n in nets]}')
if len(nets) < 1:
    sys.exit()
prov = np.array([n['prov'] for n in nets])
print(f'  provisional events by level: {prov.sum((0, 1)).tolist()} of {8 * len(nets)} each')
for valid in (False, True):
    tag = 'validated events only' if valid else 'all events'
    for name in ('P3A', 'C-K3fa', 'C-K3f', 'P3A-K3f'):
        fr, b0, meds = [], [], []
        for k in range(5):
            vals = [np.minimum(n['rel'][name][:, k][~n['prov'][:, k]] if valid else n['rel'][name][:, k], 1e6) for n in nets]
            allv = np.concatenate(vals)
            fr.append(np.mean(allv <= 0.5)); med = q(allv, 0.5); meds.append(med)
            bm = []
            for _ in range(4000):
                pick = rng.integers(0, len(vals), len(vals))
                x = np.concatenate([vals[j] for j in pick])
                if x.size:
                    bm.append(q(x, 0.5))
            hi = q(np.array(bm), 0.975)
            b0.append('Y' if (med <= 0.9 and hi < 1) else 'n')
        print(f'  [{tag}] {name:>8}: fraction within 50% ' + ' '.join('%.3f' % v for v in fr)
              + ' | median rel err ' + ' '.join('%.2f' % v for v in meds) + ' | better than zero (1e-4..1e-12) ' + ''.join(b0))
    for comp in ('C-K3fa', 'C-K3f'):
        row = []
        for k in range(5):
            d = []
            for n in nets:
                msk = ~n['prov'][:, k] if valid else np.ones(NN, bool)
                if msk.any():
                    d.append(np.mean(n['rel']['P3A'][msk, k] <= 0.5) - np.mean(n['rel'][comp][msk, k] <= 0.5))
            d = np.array(d)
            bm = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
            row.append(f'1e-{KS[k]} {d.mean():+.3f} [{q(bm, 0.025):+.3f}, {q(bm, 0.975):+.3f}]')
        print(f'  [{tag}] P3A - {comp}, mean per-net difference in fraction within 50%: ' + ' | '.join(row))
print('  P3A fallbacks (events not using the chi mixture):', int(sum((n['fb']['P3A'] > 0).sum() for n in nets)), 'of', 40 * len(nets),
      '; P3A-K3f fallbacks:', int(sum((n['fb']['P3A-K3f'] > 0).sum() for n in nets)))
print('  ARC heaviside estimates <= 0 (fraction): C-K3fa %.2f, C-K3f %.2f' % (np.mean([n['neg']['C-K3fa'] for n in nets]), np.mean([n['neg']['C-K3f'] for n in nets])))
c = {k: np.median([n['cost'][k] for n in nets]) for k in ('P3A', 'C-K3fa', 'C-K3f', 'secs')}
print('  costs per net (ARC counter, median): P3A %.3g, C-K3fa %.3g, C-K3f %.3g; K3fa run time %.0f s' % (c['P3A'], c['C-K3fa'], c['C-K3f'], c['secs']))
