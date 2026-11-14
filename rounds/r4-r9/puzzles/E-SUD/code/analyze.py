"""E-SUD analysis: D0 gate, D4 verdict (seed-t CI + hierarchical bootstrap), KL-vs-M table. Usage: analyze.py OUTDIR"""
import json, sys, glob, os
import numpy as np

O = sys.argv[1]
seeds = sorted(int(f.split('seed')[-1].split('.')[0]) for f in glob.glob(f'{O}/eval_s6_seed*.json'))
E = {s: json.load(open(f'{O}/eval_s6_seed{s}.json')) for s in seeds}
P = {s: np.load(f'{O}/perboard_s6_seed{s}.npz') for s in seeds}
MS = [0, 1, 2, 4, 8, 16]
T = {2: 4.303, 3: 3.182, 4: 2.776}
rng = np.random.default_rng(0)
out = dict(seeds=seeds)


def cut(arm, base, nm, sub, m=8):
    """Relative log-loss cut 1 - LL_arm/LL_base on subset sub ('pos'/'neg'), per seed + hierarchical bootstrap."""
    per, boots = [], []
    for s in seeds:
        sv = P[s]['solvable'] == 1
        idx = sv if sub == 'pos' else ~sv
        a = P[s][f'sll_{nm}_{arm}_M{m}'][idx]; b = P[s][f'sll_{nm}_{base}_M{0 if base == "P0" else m}'][idx]
        per.append((a, b))
    vals = np.array([1 - float(a.mean()) / float(b.mean()) for a, b in per])
    mu = vals.mean(); n = len(vals)
    half = T.get(n - 1, 2.0) * vals.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    for _ in range(10000):
        ss = rng.integers(0, n, n); num = den = 0.0
        for k in ss:
            a, b = per[k]; j = rng.integers(0, len(a), len(a))
            num += a[j].mean(); den += b[j].mean()
        boots.append(1 - num / den)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict = 'PASS' if (mu >= 0.20 and mu - half > 0) else 'FAIL'
    if verdict == 'PASS' and mu - half >= 0.20:
        verdict = 'STRONG PASS'
    return dict(per_seed=[round(float(v), 4) for v in vals], mean=round(float(mu), 4), seed_t_ci=[round(float(mu - half), 4), round(float(mu + half), 4)],
                boot_ci=[round(float(lo), 4), round(float(hi), 4)], verdict=verdict)


out['D0'] = {s: E[s]['D0'] for s in seeds}
k0 = np.mean([E[s]['D0']['kl_P0_all'] for s in seeds])
out['D0_verdict'] = ('gap >= 0.05' if E[seeds[0]]['D0']['kl_P0_all'] >= 0.05 else
                     ('small gap (0.01-0.05)' if E[seeds[0]]['D0']['kl_P0_all'] >= 0.01 else 'no gap (< 0.01)'))
out['D4'] = {}
for nm in ['P2', 'P1', 'P3']:
    if f'sll_{nm}_null_M8' not in P[seeds[0]]:
        continue
    d = dict(a_pos=cut('null', 'P0', nm, 'pos'), a_neg=cut('null', 'P0', nm, 'neg'), b_neg=cut('null', 'witness', nm, 'neg'))
    d['overall'] = 'PASS' if all(d[k]['verdict'] != 'FAIL' for k in ['a_pos', 'a_neg', 'b_neg']) else 'FAIL'
    d['guard_pooled_ll_M8'] = {s: dict(null=E[s]['solvable'][nm]['8']['null']['ll_all'], witness=E[s]['solvable'][nm]['8']['witness']['ll_all']) for s in seeds}
    d['count_control_neg_cut_vs_witness'] = cut('count', 'witness', nm, 'neg')
    out['D4'][nm] = d
# KL vs M (mean over seeds)
arms = sorted({k for s in seeds for m in E[s]['cells'] for k in E[s]['cells'][m]})
klt = {}
for m in MS:
    for a in arms:
        v = [E[s]['cells'][str(m)][a]['kl'] for s in seeds if a in E[s]['cells'].get(str(m), {})]
        if v:
            klt.setdefault(a, {})[m] = (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)) if len(v) > 1 else 0, 4))
out['kl_vs_M'] = klt
sol = {}
for nm in ['P1', 'P2', 'P3']:
    for arm in ['P0', 'witness', 'null', 'count']:
        for m in MS:
            for key in ['ll_neg', 'll_pos', 'll_all', 'ece']:
                v = [E[s]['solvable'][nm][str(m)][arm][key] for s in seeds if nm in E[s]['solvable']]
                if v:
                    sol.setdefault(nm, {}).setdefault(arm, {}).setdefault(key, {})[m] = round(float(np.mean(v)), 4)
out['solvable_vs_M'] = sol
out['chain_diag'] = {s: E[s].get('chain_diag') for s in seeds}
out['fit'] = {s: dict(theta=E[s]['fit']['theta'], kappa=E[s]['fit']['kappa']) for s in seeds}


def kcut(arm, base, m, mb):
    vals = []
    for s in seeds:
        a = P[s][f'kl_{arm}_M{m}']; b = P[s][f'kl_{base}_M{mb}']
        vals.append(1 - float(a.mean()) / float(b.mean()))
    vals = np.array(vals); mu = vals.mean(); half = T.get(len(vals) - 1, 2.0) * vals.std(ddof=1) / np.sqrt(len(vals))
    return dict(per_seed=[round(float(v), 4) for v in vals], mean=round(float(mu), 4), seed_t_ci=[round(float(mu - half), 4), round(float(mu + half), 4)])


out['cell_cuts'] = {f'{a}_vs_P0_M{m}': kcut(a, 'P0', m, 0) for a in ['P1', 'P2', 'P3'] for m in [8, 16]}
out['cell_cuts'].update({f'P2_vs_P1_M{m}': kcut('P2', 'P1', m, m) for m in [8, 16]})
out['cell_cuts'].update({f'P3_vs_P2_M{m}': kcut('P3', 'P2', m, m) for m in [8, 16]})
json.dump(out, open(f'{O}/ANALYSIS.json', 'w'), indent=1)
print(json.dumps(out['cell_cuts']))
print(json.dumps({k: out[k] for k in ['D0_verdict']}), json.dumps(out['D4'], indent=0)[:3000])
