"""E-SUD round 2 scorer (numpy only): recall-floored null search, floored count control, R2-D1a and R2-D1b.
Usage: r2.py DUMPDIR OUTDIR    (DUMPDIR holds chains_tau{1,2}_s6_seed{0,1,2}.npz from `sud.py dump`)"""
import json, math, sys, os
import numpy as np

D, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
SEEDS = [0, 1, 2]; MS = [0, 1, 2, 4, 8, 16]
TH = np.exp(np.linspace(np.log(0.01), np.log(1000), 101))
KP = np.concatenate([[0], np.exp(np.linspace(np.log(1e-3), np.log(10), 61))])
ETA = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]
TT = 4.303
rng = np.random.default_rng(0)


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def ece(p, y, nb=10):
    b = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(nb) if (b == k).any()))


def terms(Z, lq, chk, m):
    B = len(Z)
    if m == 0:
        return np.zeros(B, bool), np.zeros(B), np.zeros(B)
    found = (chk[:, :m] >= 0.5).any(1); best = chk[:, :m].max(1); Cm = np.zeros(B)
    for b in range(B):
        seen = {}
        for r in range(m):
            if chk[b, r] < 0.5:
                k = Z[b, r].tobytes(); seen[k] = max(seen.get(k, -1e9), lq[b, r])
        Cm[b] = min(sum(math.exp(v) for v in seen.values()), 1 - 1e-9)
    return found, best, Cm


def arms(p0, found, best, Cm, m, th, eta_m, kp, eta_k, th0, kp0):
    wit = np.maximum(p0, best); l0 = logit(p0)
    sig = lambda z: 1 / (1 + np.exp(-z))
    Lm = (1 - eta_m) + eta_m * np.exp(th * np.log1p(-Cm))
    Lc = (1 - eta_k) + eta_k * np.exp(-kp * m)
    return dict(P0=p0, witness=np.where(found, wit, p0),
                null_floor=np.where(found, wit, sig(l0 + np.log(np.maximum(Lm, 1e-300)))),
                count_floor=np.where(found, wit, sig(l0 + np.log(np.maximum(Lc, 1e-300)))),
                null_r1=np.where(found, wit, sig(l0 + th0 * np.log1p(-Cm))),
                count_r1=np.where(found, wit, sig(l0 - kp0 * m)))


def fit(V, rule):
    y = V['solvable'].astype(float); p0 = V['p0s']
    vt = {m: terms(V[f'{rule}_Z'], V[f'{rule}_lq'], V[f'{rule}_chk'], m) for m in MS if m}
    l0 = logit(p0)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def loss_mass(th, eta):
        tot = 0
        for m, (f, b, C) in vt.items():
            L = (1 - eta) + eta * np.exp(th * np.log1p(-C))
            tot += ll(np.where(f, np.maximum(p0, b), sig(l0 + np.log(np.maximum(L, 1e-300)))), y).mean()
        return tot / len(vt)

    def loss_cnt(kp, eta):
        tot = 0
        for m, (f, b, C) in vt.items():
            L = (1 - eta) + eta * math.exp(-kp * m)
            tot += ll(np.where(f, np.maximum(p0, b), sig(l0 + math.log(max(L, 1e-300)))), y).mean()
        return tot / len(vt)
    th, em = min(((a, e) for a in TH for e in ETA), key=lambda q: loss_mass(*q))
    kp, ek = min(((a, e) for a in KP for e in ETA), key=lambda q: loss_cnt(*q))
    th0 = min(TH, key=lambda a: loss_mass(a, 1.0))           # round-1 (unfloored) refit, for reference
    kp0 = min(KP, key=lambda a: loss_cnt(a, 1.0))
    return dict(theta=float(th), eta_mass=em, kappa=float(kp), eta_count=ek, theta_r1=float(th0), kappa_r1=float(kp0))


def load(tau, s):
    import glob
    out = {'val': {}, 'test': {}}
    for fn in sorted(glob.glob(f'{D}/chains_tau{tau}_*s6_seed{s}.npz')):   # one file per rule (R2 rerun after preemption)
        z = np.load(fn)
        for sp in ['val', 'test']:
            out[sp].update({k[len(sp) + 1:]: z[k] for k in z.files if k.startswith(sp + '_')})
    return out


res = {'fits': {}, 'curves': {}, 'per': {}}
per = {}
for tau, rules in [(1, ['rand', 'voi', 'conf']), (2, ['rand', 'conf'])]:
    for s in SEEDS:
        X = load(tau, s)
        T = X['test']; y = T['solvable'].astype(float); pos = y == 1
        for r in rules:
            f = fit(X['val'], r); res['fits'][f'tau{tau}_{r}_seed{s}'] = f
            for m in MS:
                fo, be, C = terms(T[f'{r}_Z'], T[f'{r}_lq'], T[f'{r}_chk'], m)
                A = arms(T['p0s'], fo, be, C, m, f['theta'], f['eta_mass'], f['kappa'], f['eta_count'], f['theta_r1'], f['kappa_r1'])
                row = dict(found_pos=float(fo[pos].mean()), found_neg=float(fo[~pos].mean()),
                           Cm_pos_nowit=float(C[pos & ~fo].mean()) if (pos & ~fo).any() else None, Cm_neg=float(C[~pos].mean()))
                for k, p in A.items():
                    l = ll(p, y); per[(tau, r, s, m, k)] = (l, pos)
                    row[k] = dict(ll_all=float(l.mean()), ll_pos=float(l[pos].mean()), ll_neg=float(l[~pos].mean()), ece=ece(p, y))
                res['curves'][f'tau{tau}_{r}_seed{s}_M{m}'] = row
        print('done tau', tau, 'seed', s, flush=True)


def cut(tau, r, m, arm, base, sub, mb=None):
    vals, pairs = [], []
    for s in SEEDS:
        la, pos = per[(tau, r, s, m, arm)]; lb, _ = per[(tau, r, s, m if mb is None else mb, base)]
        idx = pos if sub == 'pos' else (~pos if sub == 'neg' else np.ones_like(pos))
        vals.append(1 - la[idx].mean() / lb[idx].mean()); pairs.append((la[idx], lb[idx]))
    v = np.array(vals); mu = v.mean(); h = TT * v.std(ddof=1) / np.sqrt(3)
    bs = []
    for _ in range(4000):
        num = den = 0.0
        for k in rng.integers(0, 3, 3):
            a, b = pairs[k]; j = rng.integers(0, len(a), len(a)); num += a[j].mean(); den += b[j].mean()
        bs.append(1 - num / den)
    return dict(per_seed=[round(float(x), 4) for x in v], mean=round(float(mu), 4), seed_t_ci=[round(float(mu - h), 4), round(float(mu + h), 4)],
                boot_ci=[round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])])


def verdict(c, thr):
    return 'PASS' if c['mean'] >= thr and c['seed_t_ci'][0] > 0 else 'FAIL'


d1a = {}
for r in ['rand', 'voi', 'conf']:
    cs = dict(pos_vs_P0=cut(1, r, 8, 'null_floor', 'P0', 'pos', 0), neg_vs_P0=cut(1, r, 8, 'null_floor', 'P0', 'neg', 0),
              neg_vs_witness=cut(1, r, 8, 'null_floor', 'witness', 'neg'))
    for k in cs:
        cs[k]['verdict'] = verdict(cs[k], 0.20)
    cs['rule'] = 'PASS' if all(cs[k]['verdict'] == 'PASS' for k in ['pos_vs_P0', 'neg_vs_P0', 'neg_vs_witness']) else 'FAIL'
    cs['r1_unfloored_pos_vs_P0'] = cut(1, r, 8, 'null_r1', 'P0', 'pos', 0)
    d1a[r] = cs
res['R2_D1a'] = d1a; res['R2_D1a_verdict'] = 'PASS' if all(d1a[r]['rule'] == 'PASS' for r in d1a) else 'FAIL'
d1b = {}
for r in ['rand', 'conf']:
    c = cut(2, r, 4, 'null_floor', 'count_floor', 'all'); c['verdict'] = verdict(c, 0.05)
    d1b[r] = dict(M4=c, by_M={m: cut(2, r, m, 'null_floor', 'count_floor', 'all') for m in [1, 2, 8, 16]},
                  tau1_M4=cut(1, r, 4, 'null_floor', 'count_floor', 'all'))
res['R2_D1b'] = d1b; res['R2_D1b_verdict'] = 'PASS' if all(d1b[r]['M4']['verdict'] == 'PASS' for r in d1b) else 'FAIL'
# seed-mean curves
mean = {}
for tau, rules in [(1, ['rand', 'voi', 'conf']), (2, ['rand', 'conf'])]:
    for r in rules:
        for m in MS:
            for k in ['P0', 'witness', 'null_floor', 'count_floor', 'null_r1', 'count_r1']:
                for q in ['ll_pos', 'll_neg', 'll_all', 'ece']:
                    mean.setdefault(f'tau{tau}_{r}', {}).setdefault(k, {}).setdefault(q, {})[m] = round(float(np.mean(
                        [res['curves'][f'tau{tau}_{r}_seed{s}_M{m}'][k][q] for s in SEEDS])), 5)
            mean[f'tau{tau}_{r}'].setdefault('found_pos', {})[m] = round(float(np.mean([res['curves'][f'tau{tau}_{r}_seed{s}_M{m}']['found_pos'] for s in SEEDS])), 4)
            mean[f'tau{tau}_{r}'].setdefault('found_neg', {})[m] = round(float(np.mean([res['curves'][f'tau{tau}_{r}_seed{s}_M{m}']['found_neg'] for s in SEEDS])), 4)
res['mean'] = mean
json.dump(res, open(f'{O}/R2-ANALYSIS.json', 'w'), indent=1)
with open(f'{O}/r2_curves.csv', 'w') as fh:
    fh.write('tau_rule,arm,metric,M,mean_over_3_seeds\n')
    for tr, dd in mean.items():
        for k, v in dd.items():
            if k.startswith('found'):
                for m, x in v.items():
                    fh.write(f'{tr},-,{k},{m},{x}\n')
            else:
                for q, vv in v.items():
                    for m, x in vv.items():
                        fh.write(f'{tr},{k},{q},{m},{x}\n')
print(json.dumps({'R2_D1a': res['R2_D1a_verdict'], 'R2_D1b': res['R2_D1b_verdict']}))
print(json.dumps({r: {k: (v['mean'], v['seed_t_ci'], v['verdict']) for k, v in d1a[r].items() if isinstance(v, dict) and 'verdict' in v} for r in d1a}))
print(json.dumps({r: (d1b[r]['M4']['per_seed'], d1b[r]['M4']['mean'], d1b[r]['M4']['seed_t_ci'], d1b[r]['M4']['verdict']) for r in d1b}))
