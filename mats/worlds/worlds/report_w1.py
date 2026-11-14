"""Tables for the W1 pilot from a runs directory (summary.json, diag.json, meta.json per arm). usage: report_w1.py RUNS"""
import glob, json, os, sys
import numpy as np

R = sys.argv[1]
TRAIN = ['uniform', 'zipf', 'pattern', 'repeat']
arms = {}
for d in sorted(glob.glob(os.path.join(R, 't*_L*_k*_s*'))):
    if not os.path.isdir(d): continue
    b = os.path.basename(d); t, L, k, s = b.split('_')
    m = json.load(open(os.path.join(d, 'meta.json'))) if os.path.exists(os.path.join(d, 'meta.json')) else None
    sm = json.load(open(os.path.join(d, 'eval', 'summary.json'))) if os.path.exists(os.path.join(d, 'eval', 'summary.json')) else None
    dg = json.load(open(os.path.join(d, 'diag.json'))) if os.path.exists(os.path.join(d, 'diag.json')) else None
    arms[b] = dict(tau=t[1:], L=int(L[1:]), sink=int(k[1:]), seed=int(s[1:]), meta=m, sum=sm, diag=dg)

taus = sorted({a['tau'] for a in arms.values()})
cfgs = sorted({(a['L'], a['sink']) for a in arms.values()})
f = lambda x: '' if x is None else (f'{x:.3f}' if isinstance(x, float) else str(x))


def cell(a, N, pol):
    for c in a['sum']['cells']:
        if c['N'] == N and c['policy'] == pol: return c
    return None


def avg(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def group(tau, L, sink):
    return [a for a in arms.values() if a['tau'] == tau and a['L'] == L and a['sink'] == sink and a['sum']]


print('## Training\n')
print('| arm | params | val gap to Bayes (best) | best step | last-quarter improvement | ms/step (data+compute) |')
print('|---|---|---|---|---|---|')
for b, a in sorted(arms.items()):
    m = a['meta']
    if not m: continue
    print(f"| {b} | {m['n_params']} | {m['best_val'] - m['val_bayes_nll']:.4f} | {m['best_step']} | "
          f"{f(m['last_quarter_improvement'])} | {m['ms_data_per_step']:.1f}+{m['ms_compute_per_step']:.1f} |")

any_sum = next((a for a in arms.values() if a['sum']), None)
if any_sum:
    Ns = sorted({c['N'] for c in any_sum['sum']['cells']})
    refs = [k[4:] for k in any_sum['sum']['cells'][0]['splits']['all'] if k.startswith('gap_') and k != 'gap_model' and k != 'gap_bayes']
    for pols, title in ((TRAIN, 'training policies (mean of 4)'), (['ucb'], 'held-out UCB')):
        for tau in taus:
            print(f'\n## Gap to Bayes (nats/episode), tau2={tau}, {title}\n')
            print('| predictor | ' + ' | '.join(f'N={N}' for N in Ns) + ' |')
            print('|---|' + '---|' * len(Ns))
            for L, sink in cfgs:
                g = group(tau, L, sink)
                if not g: continue
                row = [avg([avg([cell(a, N, p)['splits']['all']['gap_model'][0] for p in pols]) for a in g]) for N in Ns]
                print(f'| model L{L} sink{sink} (n={len(g)}) | ' + ' | '.join(f(x) for x in row) + ' |')
            a0 = next(a for a in arms.values() if a['tau'] == tau and a['sum'])
            for r in refs:
                row = [avg([cell(a0, N, p)['splits']['all']['gap_' + r][0] for p in pols]) for N in Ns]
                print(f'| ref {r} | ' + ' | '.join(f(x) for x in row) + ' |')
    print('\n## Abilities: fraction of the reference gain achieved by the model (training policies, mean)\n')
    for tau in taus:
        print(f'\ntau2={tau}\n')
        labs = [l for l in cell(any_sum, Ns[0], 'uniform')['abilities']]
        print('| config | ' + ' | '.join(f'{l.split(" ")[0]} N={N}' for l in labs[:2] for N in (16, 64, 256, 1024, 4096)) + ' |')
        print('|---|' + '---|' * (2 * 5))
        for L, sink in cfgs:
            g = group(tau, L, sink)
            if not g: continue
            row = []
            for l in labs[:2]:
                for N in (16, 64, 256, 1024, 4096):
                    row.append(avg([avg([(cell(a, N, p)['abilities'].get(l) or {}).get('frac') for p in TRAIN]) for a in g]))
            print(f'| L{L} sink{sink} | ' + ' | '.join(f(x) for x in row) + ' |')
    print('\n## Bias and variance of the model predictive vs Bayes (training policies, mean over arms)\n')
    print('| tau2 | config | ' + ' | '.join(f'rms bias/sd N={N}' for N in (64, 1024, 4096)) + ' | ' +
          ' | '.join(f'log var ratio N={N}' for N in (64, 1024, 4096)) + ' |')
    print('|---|---|' + '---|' * 6)
    for tau in taus:
        for L, sink in cfgs:
            g = group(tau, L, sink)
            if not g: continue
            b = [avg([avg([cell(a, N, p)['splits']['all']['bias_rms_over_sd'] for p in TRAIN]) for a in g]) for N in (64, 1024, 4096)]
            v = [avg([avg([cell(a, N, p)['splits']['all']['logvar_ratio'][0] for p in TRAIN]) for a in g]) for N in (64, 1024, 4096)]
            print(f'| {tau} | L{L} sink{sink} | ' + ' | '.join(f(x) for x in b + v) + ' |')
    print('\n## Wall clock (eval forward, ms per episode, one arm L4 sink0)\n')
    a = next((a for a in arms.values() if a['L'] == 4 and a['sink'] == 0 and a['sum']), any_sum)
    print('| N | ' + ' | '.join(str(N) for N in Ns) + ' |\n|---|' + '---|' * len(Ns))
    print('| ms/episode | ' + ' | '.join(f(avg([cell(a, N, p)['forward_ms_per_episode'] for p in TRAIN + ['ucb']])) for N in Ns) + ' |')

dg_any = [a for a in arms.values() if a['diag']]
if dg_any:
    Ms = sorted({int(k.split('_M')[1]) for k in dg_any[0]['diag']['C1'] if '_M' in k})
    print('\n## C1 contamination (KL(Bayes||model) per M; contamination = KL(p(D_M)||p(D_0)))\n')
    for tau in taus:
        print(f'\ntau2={tau}\n')
        print('| config | ' + ' | '.join(f'KL fresh M={M}' for M in Ms) + ' | ' + ' | '.join(f'KL control M={M}' for M in Ms[1:]) +
              ' | contam model/bayes fresh M=4048 | contam model/bayes copy M=4048 | attn L1 mass/share M=512 | rule |')
        print('|---|' + '---|' * (2 * len(Ms) - 1) + '---|---|---|---|')
        for L, sink in cfgs:
            g = [a for a in dg_any if a['tau'] == tau and a['L'] == L and a['sink'] == sink]
            if not g: continue
            C = lambda key, fld, i=0: avg([(a['diag']['C1'][key][fld][i] if isinstance(a['diag']['C1'][key][fld], list) else a['diag']['C1'][key][fld]) for a in g])
            fr = [C(f'fresh_M{M}', 'kl_model') for M in Ms]; ct = [C(f'control_M{M}', 'kl_model') for M in Ms[1:]]
            Mx = Ms[-1]
            cm = f"{f(C(f'fresh_M{Mx}', 'contam_model'))}/{f(C(f'fresh_M{Mx}', 'contam_bayes'))}"
            cc = f"{f(C(f'copy_M{Mx}', 'contam_model'))}/{f(C(f'copy_M{Mx}', 'contam_bayes'))}"
            at = f"{f(C('fresh_M512', 'attn_mass_dom', 0))}/{f(g[0]['diag']['C1']['fresh_M512']['share'])}" if 512 in Ms else ''
            rule = ','.join(('vacuous' if a['diag']['C1']['rule']['vacuous'] else ('refuted' if a['diag']['C1']['rule']['refuted'] else 'pass')) for a in g)
            print(f'| L{L} sink{sink} | ' + ' | '.join(f(x) for x in fr + ct) + f' | {cm} | {cc} | {at} | {rule} |')
    print('\n## Matched pairs and duplication controls\n')
    print('| tau2 | config | D1 slope N=64 | D1 slope N=256 | D2 copies k=16 dlogvar (Bayes 0) | D2 copies k=64 | '
          'D2 fresh k=4 dlogvar model/Bayes | D2 fresh k=64 model/Bayes | D4 copies x8 | D4 fresh x8 model/Bayes |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for tau in taus:
        for L, sink in cfgs:
            g = [a for a in dg_any if a['tau'] == tau and a['L'] == L and a['sink'] == sink]
            if not g: continue
            P = lambda *ks: avg([_get(a['diag']['pairs'], ks) for a in g])
            def _get(d, ks):
                for k in ks: d = d[k]
                return d[0] if isinstance(d, list) else d
            globals()['_get'] = _get
            print(f"| {tau} | L{L} sink{sink} | {f(P('D1_N64', 'slope'))} | {f(P('D1_N256', 'slope'))} | "
                  f"{f(P('D2_k16', 'copies', 'logvr_model'))} | {f(P('D2_k64', 'copies', 'logvr_model'))} | "
                  f"{f(P('D2_k4', 'fresh', 'logvr_model'))}/{f(P('D2_k4', 'fresh', 'logvr_bayes'))} | "
                  f"{f(P('D2_k64', 'fresh', 'logvr_model'))}/{f(P('D2_k64', 'fresh', 'logvr_bayes'))} | "
                  f"{f(P('D4_k8', 'copies', 'logvr_model'))} | {f(P('D4_k8', 'fresh', 'logvr_model'))}/{f(P('D4_k8', 'fresh', 'logvr_bayes'))} |")
    print('\n## Count probes (ridge R^2 per layer, embedding first; N=128 training-mix contexts)\n')
    print('| tau2 | config | log n item fresh | log n item raw | log n cell fresh |\n|---|---|---|---|---|')
    for tau in taus:
        for L, sink in cfgs:
            g = [a for a in dg_any if a['tau'] == tau and a['L'] == L and a['sink'] == sink]
            if not g: continue
            pr = lambda k: ' '.join(f(x) for x in np.mean([a['diag']['probes'][k] for a in g], 0))
            print(f"| {tau} | L{L} sink{sink} | {pr('log_n_item_fresh')} | {pr('log_n_item_raw')} | {pr('log_n_cell_fresh')} |")
