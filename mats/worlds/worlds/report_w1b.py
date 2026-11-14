"""W1b tables. usage: report_w1b.py RUNDIR [RUNDIR ...]  (each containing {A..E}_s{seed}/)"""
import glob, json, os, sys
import numpy as np

TRAIN = ['uniform', 'zipf', 'pattern', 'repeat']
runs = {}
for R in sys.argv[1:]:
    for d in glob.glob(os.path.join(R, '?_s*')):
        if not os.path.isdir(d): continue
        arm, s = os.path.basename(d).split('_s')
        get = lambda p: json.load(open(os.path.join(d, p))) if os.path.exists(os.path.join(d, p)) else None
        runs[(arm, int(s))] = dict(meta=get('meta.json'), sum=get('eval/summary.json'), diag=get('diag.json'))
ARMS = sorted({a for a, _ in runs}); SEEDS = sorted({s for _, s in runs})
f = lambda x, p=3: '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{p}f}'


def cell(r, N, pol):
    return next((c for c in r['sum']['cells'] if c['N'] == N and c['policy'] == pol), None) if r['sum'] else None


def stat(arm, fn):
    v = [fn(runs[(arm, s)]) for s in SEEDS if (arm, s) in runs and runs[(arm, s)]['sum']]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def pm(t):
    return '' if t[0] is None else f'{t[0]:.3f}±{t[1]:.3f}'


def _v(r, N, p, key):
    x = cell(r, N, p)['splits']['all'].get(key)
    return x[0] if isinstance(x, list) else x


def per_pol(pols, key):
    def at(N):
        def fn(r):
            v = [_v(r, N, p, key) for p in pols]
            return None if any(x is None for x in v) else float(np.mean(v))
        return fn
    return at
any_r = next(r for r in runs.values() if r['sum'])
Ns = sorted({c['N'] for c in any_r['sum']['cells']})

print('## Training (100k steps; convergence rule: last-quarter best improves earlier best by < 0.003 nats)\n')
print('| arm | seed | params | val gap (best) | last-quarter improvement | converged | ms/step data+compute |\n|---|---|---|---|---|---|---|')
for (a, s), r in sorted(runs.items()):
    m = r['meta']
    if m: print(f"| {a} | {s} | {m['n_params']} | {m['best_val'] - m['val_bayes_nll']:.4f} | {f(m['last_quarter_improvement'], 4)} | "
                f"{m.get('converged')} | {m['ms_data_per_step']:.1f}+{m['ms_compute_per_step']:.1f} |")

for pols, title in ((TRAIN, 'training policies'), (['ucb'], 'held-out UCB')):
    print(f'\n## Gap to Bayes (nats), {title}, mean±sd over seeds\n')
    print('| arm | ' + ' | '.join(f'N={N}' for N in Ns) + ' |\n|---|' + '---|' * len(Ns))
    for a in ARMS:
        print(f'| {a} | ' + ' | '.join(pm(stat(a, per_pol(pols, 'gap_model')(N))) for N in Ns) + ' |')
    for ref in ('item', 'count', 'oracle'):
        print(f'| ref {ref} | ' + ' | '.join(f(float(np.mean([cell(any_r, N, p)['splits']['all']['gap_' + ref][0] for p in pols]))) for N in Ns) + ' |')

print('\n## KL(Bayes||model) split (training policies): variance part / mean part / KL total\n')
sel = [64, 512, 1024, 4096]
print('| arm | ' + ' | '.join(f'N={N} var / mean / KL' for N in sel) + ' |\n|---|' + '---|' * len(sel))
for a in ARMS:
    row = []
    for N in sel:
        v = stat(a, per_pol(TRAIN, 'kl_var_part')(N))[0]; m = stat(a, per_pol(TRAIN, 'kl_mean_part')(N))[0]
        k = stat(a, per_pol(TRAIN, 'kl_model')(N))[0]
        row.append(f'{f(v)} / {f(m)} / {f(k)}')
    print(f'| {a} | ' + ' | '.join(row) + ' |')
print('\n| arm | ' + ' | '.join(f'log var ratio N={N}' for N in sel) + ' |\n|---|' + '---|' * len(sel))
for a in ARMS:
    print(f'| {a} | ' + ' | '.join(pm(stat(a, per_pol(TRAIN, 'logvar_ratio')(N))) for N in sel) + ' |')

print('\n## Pre-registered readings (per seed)\n')
g = lambda a, s, N, key='gap_model': per_pol(TRAIN, key)(N)(runs[(a, s)]) if (a, s) in runs and runs[(a, s)]['sum'] else None
for s in SEEDS:
    gA = {N: g('A', s, N) for N in (64, 256, 512, 4096)}; gB = {N: g('B', s, N) for N in (256, 512, 4096)}
    if None in gA.values() or None in gB.values(): continue
    print(f"- seed {s}: B/A gap at N=256 {gB[256] / gA[256]:.2f}, at N=512 {gB[512] / gA[512]:.2f}; "
          f"B gap(4096)/gap(512) {gB[4096] / gB[512]:.2f}")
varA = np.mean([g('A', s, 4096, 'kl_var_part') for s in SEEDS if g('A', s, 4096, 'kl_var_part') is not None] or [np.nan])
for a in ('A', 'C', 'E', 'B', 'D'):
    out = []
    for s in SEEDS:
        if g(a, s, 64) is None: continue
        ratio = g(a, s, 4096) / g(a, s, 64); vp = g(a, s, 4096, 'kl_var_part')
        if vp is None: vp = float('nan')
        ext = ratio <= 1.5 and vp <= 0.25 * varA
        out.append(f's{s}: gap4096/gap64 {ratio:.2f}, var part {vp:.3f} ({vp / varA:.2f} x A) -> {"extrapolates" if ext else "no"}')
    print(f'- {a}: ' + '; '.join(out))

print('\n## Copies vs agreeing fresh readings of the query (D2): log-variance change, copies / fresh (Bayes: 0 / fresh)\n')
ks = [4, 16, 64, 256]
print('| arm | ' + ' | '.join(f'k={k} copies / fresh / ratio' for k in ks) + ' |\n|---|' + '---|' * len(ks))
bay = None
for a in ARMS:
    row = []
    for k in ks:
        ok = [runs[(a, s)]['diag']['pairs'] for s in SEEDS if (a, s) in runs and runs[(a, s)]['diag'] and f'D2_k{k}' in runs[(a, s)]['diag']['pairs']]
        c = [p[f'D2_k{k}']['copies']['logvr_model'][0] for p in ok]; fr = [p[f'D2_k{k}']['fresh']['logvr_model'][0] for p in ok]
        if c:
            bay = bay or {}; bay[k] = ok[0][f'D2_k{k}']['fresh']['logvr_bayes']
            row.append(f'{np.mean(c):+.2f} / {np.mean(fr):+.2f} / {np.mean(c) / np.mean(fr):.2f}')
        else: row.append('')
    print(f'| {a} | ' + ' | '.join(row) + ' |')
if bay: print('| Bayes | ' + ' | '.join(f'0 / {bay[k]:+.2f} / 0' if k in bay else '' for k in ks) + ' |')

print('\n## D4 whole context x8 (log-variance change; Bayes copies 0, fresh below) and C1 rule\n')
print('| arm | D4 copies x8 | D4 fresh x8 | C1 control KL at M=16/64/512/4048 | C1 rule |\n|---|---|---|---|---|')
for a in ARMS:
    ds = [runs[(a, s)]['diag'] for s in SEEDS if (a, s) in runs and runs[(a, s)]['diag']]
    if not ds: continue
    c = np.mean([d['pairs']['D4_k8']['copies']['logvr_model'][0] for d in ds]); fr = np.mean([d['pairs']['D4_k8']['fresh']['logvr_model'][0] for d in ds])
    ctl = '/'.join(f"{np.mean([d['C1'][f'control_M{M}']['kl_model'][0] for d in ds]):.2f}" for M in (16, 64, 512, 4048))
    rule = ','.join('vacuous' if d['C1']['rule']['vacuous'] else ('refuted' if d['C1']['rule']['refuted'] else 'pass') for d in ds)
    print(f'| {a} | {c:+.3f} | {fr:+.3f} (Bayes {ds[0]["pairs"]["D4_k8"]["fresh"]["logvr_bayes"]:+.2f}) | {ctl} | {rule} |')
