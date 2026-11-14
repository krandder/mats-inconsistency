"""E-SUD round 3 scorer (numpy only). Usage: r3.py CHAINDIR OUTDIR
CHAINDIR holds chains_tau{t}_{rule}_s6_seed{s}.npz dumped on s6_hard_{val,test}.npz, with keys
{val,test}_{p0s,solvable,<rule>_Z,<rule>_lq,<rule>_chk,<rule>_lb} (lq = tempered proposal log-mass, lb = untempered
believed log-mass). Rules R3-D1a/b/c as in PREREG-R3.md."""
import json, math, sys, os, glob
import numpy as np

D, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
SEEDS = [0, 1, 2]; MS = [0, 1, 2, 4, 8, 16]
TH = np.exp(np.linspace(np.log(0.01), np.log(1000), 101))
KP = np.concatenate([[0], np.exp(np.linspace(np.log(1e-3), np.log(10), 61))])
ETA = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]
TT = 4.303
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def auc(score, y):
    """P(score of a positive > score of a negative), ties 1/2."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind='mergesort'); ranks = np.empty(len(allv))
    sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def terms(X, m, mass):
    Z, chk, lm = X['Z'], X['chk'], X[mass]
    B = len(Z)
    if m == 0:
        return np.zeros(B, bool), np.zeros(B), np.zeros(B)
    found = (chk[:, :m] >= 0.5).any(1); best = chk[:, :m].max(1); Cm = np.zeros(B)
    for b in range(B):
        seen = {}
        for r in range(m):
            if chk[b, r] < 0.5:
                k = Z[b, r].tobytes(); seen[k] = max(seen.get(k, -1e9), lm[b, r])
        Cm[b] = min(sum(math.exp(v) for v in seen.values()), 1 - 1e-9)
    return found, best, Cm


def pmass(p0, f, b, C, th, eta):
    L = (1 - eta) + eta * np.exp(th * np.log1p(-C))
    return np.where(f, np.maximum(p0, b), sig(logit(p0) + np.log(np.maximum(L, 1e-300))))


def pcount(p0, f, b, m, kp, eta):
    L = (1 - eta) + eta * math.exp(-kp * m)
    return np.where(f, np.maximum(p0, b), sig(logit(p0) + math.log(max(L, 1e-300))))


def fit(V):
    y = V['solvable'].astype(float); p0 = V['p0s']
    tb = {m: terms(V, m, 'lb') for m in MS if m}
    tq = {m: terms(V, m, 'lq') for m in MS if m}
    fm = lambda tt: min(((a, e) for a in TH for e in ETA), key=lambda q: np.mean([ll(pmass(p0, *tt[m], *q), y).mean() for m in tt]))
    th_b, e_b = fm(tb); th_q, e_q = fm(tq)
    kp, ek = min(((a, e) for a in KP for e in ETA), key=lambda q: np.mean([ll(pcount(p0, tb[m][0], tb[m][1], m, *q), y).mean() for m in tb]))
    return dict(theta_bel=float(th_b), eta_bel=e_b, theta_prop=float(th_q), eta_prop=e_q, kappa=float(kp), eta_count=ek)


def load(tau, rule, s):
    out = {}
    for sp in ['val', 'test']:
        if rule == 'mixed':   # chains 1..M alternate P1 (rand) and P3 (conf) at the same tau
            a = load(tau, 'rand', s)[sp]; c = load(tau, 'conf', s)[sp]
            X = {'p0s': a['p0s'], 'solvable': a['solvable']}
            for k in ['Z', 'chk', 'lq', 'lb']:
                arr = np.empty_like(a[k]); arr[:, 0::2] = a[k][:, :8]; arr[:, 1::2] = c[k][:, :8]; X[k] = arr
            out[sp] = X; continue
        z = np.load(f'{D}/chains_tau{tau:g}_{rule}_s6_seed{s}.npz')
        out[sp] = {'p0s': z[f'{sp}_p0s'], 'solvable': z[f'{sp}_solvable'],
                   **{k: z[f'{sp}_{rule}_{k}'] for k in ['Z', 'chk', 'lq', 'lb']}}
    return out


CONDS = [(1.0, 'rand'), (1.0, 'voi'), (1.0, 'conf'), (1.5, 'rand'), (1.5, 'conf'), (1.5, 'mixed'), (2.0, 'rand'), (2.0, 'conf'), (2.0, 'mixed')]
res = dict(fits={}, curves={}, auc={}); per = {}
for tau, rule in CONDS:
    for s in SEEDS:
        X = load(tau, rule, s); T = X['test']; y = T['solvable'].astype(float); pos = y == 1
        f = fit(X['val']); res['fits'][f'tau{tau:g}_{rule}_seed{s}'] = f
        for m in MS:
            fb, bb, Cb = terms(T, m, 'lb'); fq, bq, Cq = terms(T, m, 'lq')
            A = dict(P0=T['p0s'], witness=np.where(fb, np.maximum(T['p0s'], bb), T['p0s']),
                     null_floor=pmass(T['p0s'], fb, bb, Cb, f['theta_bel'], f['eta_bel']),
                     null_floor_prop=pmass(T['p0s'], fq, bq, Cq, f['theta_prop'], f['eta_prop']),
                     count_floor=pcount(T['p0s'], fb, bb, m, f['kappa'], f['eta_count']))
            row = dict(found_pos=float(fb[pos].mean()), found_neg=float(fb[~pos].mean()),
                       recall_per_chain_pos=float((T['chk'][pos, :max(m, 1)] >= 0.5).mean()),
                       Cbel_neg=float(Cb[~pos].mean()), Cbel_pos_unres=float(Cb[pos & ~fb].mean()) if (pos & ~fb).any() else None)
            for k, p in A.items():
                l = ll(p, y); per[(tau, rule, s, m, k)] = (l, pos)
                row[k] = dict(ll_all=float(l.mean()), ll_pos=float(l[pos].mean()), ll_neg=float(l[~pos].mean()))
            res['curves'][f'tau{tau:g}_{rule}_seed{s}_M{m}'] = row
            if m in (1, 2, 4):
                un = ~fb
                res['auc'][f'tau{tau:g}_{rule}_seed{s}_M{m}'] = dict(n_unres=int(un.sum()), n_unres_pos=int((un & pos).sum()),
                                                                    auc_negCbel=auc(-Cb[un], y[un]), auc_negCprop=auc(-Cq[un], y[un]))
        print('done', tau, rule, s, flush=True)


def cut(tau, rule, m, arm, base, sub, mb=None):
    vals = []
    for s in SEEDS:
        la, pos = per[(tau, rule, s, m, arm)]; lb, _ = per[(tau, rule, s, m if mb is None else mb, base)]
        idx = pos if sub == 'pos' else (~pos if sub == 'neg' else np.ones_like(pos))
        vals.append(1 - la[idx].mean() / lb[idx].mean())
    v = np.array(vals); mu = v.mean(); h = TT * v.std(ddof=1) / np.sqrt(3)
    return dict(per_seed=[round(float(x), 4) for x in v], mean=round(float(mu), 4), seed_t_ci=[round(float(mu - h), 4), round(float(mu + h), 4)])


ok = lambda c, thr: c['mean'] >= thr and c['seed_t_ci'][0] > 0
# R3-D1a
p0neg = [per[(1.0, 'rand', s, 0, 'P0')][0][~per[(1.0, 'rand', s, 0, 'P0')][1]].mean() for s in SEEDS]
res['R3_D1a'] = dict(P0_ll_neg_per_seed=[round(float(x), 4) for x in p0neg], mean=round(float(np.mean(p0neg)), 4),
                     verdict='PASS' if np.mean(p0neg) >= 0.50 else 'FAIL')
# R3-D1b
d1b = {}
for rule in ['rand', 'voi', 'conf']:
    c = dict(neg_vs_P0=cut(1.0, rule, 8, 'null_floor', 'P0', 'neg', 0), neg_vs_witness=cut(1.0, rule, 8, 'null_floor', 'witness', 'neg'),
             pos_vs_P0=cut(1.0, rule, 8, 'null_floor', 'P0', 'pos', 0))
    c['pass'] = ok(c['neg_vs_P0'], 0.5) and ok(c['neg_vs_witness'], 0.5) and ok(c['pos_vs_P0'], 0.2)
    d1b[rule] = c
res['R3_D1b'] = d1b; res['R3_D1b_verdict'] = 'PASS' if all(d1b[r]['pass'] for r in d1b) else 'FAIL'
# R3-D1c
d1c = {}
for tau in [1.5, 2.0]:
    for rule in ['rand', 'conf', 'mixed']:
        part1 = {}
        for m in (1, 2):
            a = [res['auc'][f'tau{tau:g}_{rule}_seed{s}_M{m}']['auc_negCbel'] for s in SEEDS]
            a = [x for x in a if x is not None]
            part1[m] = dict(per_seed=[round(x, 4) for x in a], mean=round(float(np.mean(a)), 4) if a else None)
        p1 = all(part1[m]['mean'] is not None and part1[m]['mean'] >= 0.80 for m in (1, 2))
        part2 = {m: cut(tau, rule, m, 'null_floor', 'count_floor', 'all') for m in (2, 4)}
        p2 = any(ok(part2[m], 0.10) for m in (2, 4))
        d1c[f'tau{tau:g}_{rule}'] = dict(auc=part1, part1=p1, mass_vs_count=part2, part2=p2, both=p1 and p2,
                                         prop_mass_vs_count={m: cut(tau, rule, m, 'null_floor_prop', 'count_floor', 'all') for m in (2, 4)})
res['R3_D1c'] = d1c
nb = sum(v['both'] for v in d1c.values())
res['R3_D1c_verdict'] = 'PASS' if nb == 6 else 'FAIL'
res['R3_D1c_conditions_passing'] = nb
json.dump(res, open(f'{O}/R3-ANALYSIS.json', 'w'), indent=1)
print(json.dumps(dict(D1a=res['R3_D1a'], D1b=res['R3_D1b_verdict'], D1c=res['R3_D1c_verdict'], D1c_n=nb)))
print(json.dumps({r: {k: (v['mean'], v['seed_t_ci']) for k, v in c.items() if isinstance(v, dict)} for r, c in d1b.items()}))
print(json.dumps({k: dict(auc={m: v['auc'][m]['mean'] for m in v['auc']}, mvc={m: (x['mean'], x['seed_t_ci']) for m, x in v['mass_vs_count'].items()}, both=v['both']) for k, v in d1c.items()}))
