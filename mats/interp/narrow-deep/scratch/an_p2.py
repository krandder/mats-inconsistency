"""Tables for Part 2 (PREREG P1-P5) from runs/jac_*.json and runs/collapse_*.json -> stdout (markdown)."""
import json, glob, os, math
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')


def fmt(x, p=3):
    if x is None: return '-'
    if isinstance(x, float):
        if x == 0: return '0'
        if abs(x) < 1e-3 or abs(x) >= 1e4: return f'{x:.2g}'
        return f'{x:.{p}g}'
    return str(x)


print('## P1/P3/P4: per-path counts and fixed blocks (means over nets; [min, max] over nets where useful)\n')
print('| family | n | L | sigma | L\' (cut c) | nets | Nsurv(0.01) mean | Nsurv(e^-1) mean | rank-1 paths | slope log(s2/s1)/L\' | kF path (1e-4) | k*_F(M) 1e-1/1e-2/1e-4 | k*_F(S) 1e-1/1e-2/1e-4 | PR(M) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for fam in ('plain', 'res'):
    for n in (8, 16):
        for L in (64, 256):
            for sig in ('0.001', '0'):
                f = os.path.join(R, f'jac_{fam}_n{n}_L{L}_s{sig}.json')
                if not os.path.exists(f): continue
                d = json.load(open(f))
                nets = [v for v in d['res'].values() if not v.get('dead_net')]
                ndead = sum(1 for v in d['res'].values() if v.get('dead_net'))
                for lp in d['LPS']:
                    cs = [v['cuts'][str(lp)] for v in nets if str(lp) in v['cuts']]
                    if not cs: continue
                    m = lambda key, sub=None: float(np.mean([c[key] if sub is None else c[key][sub] for c in cs if (c[key] if sub is None else c[key][sub]) is not None])) if any((c[key] if sub is None else c[key][sub]) is not None for c in cs) else None
                    ns01 = np.mean([c['nsurv']['0.01']['mean'] for c in cs]); nse = np.mean([c['nsurv']['0.3679']['mean'] for c in cs])
                    kM = '/'.join(f"{np.mean([c['kM'][e] for c in cs]):.1f}" for e in ('0.1', '0.01', '0.0001'))
                    kS = '/'.join(f"{np.mean([c['kS'][e] for c in cs]):.1f}" for e in ('0.1', '0.01', '0.0001'))
                    print(f"| {fam} | {n} | {L} | {sig} | {lp} ({L-lp}) | {len(cs)}{'+'+str(ndead)+' dead' if ndead else ''} | {ns01:.2f} | {nse:.2f} | {fmt(m('rank1_frac'))} | {fmt(m('slope_log_s2s1'))} | {fmt(m('kF_path_mean'))} | {kM} | {kS} | {fmt(m('prM'))} |")
print('\n## P5: weight-derived subspaces at a cut, k = 1 (share of tr M captured: capM; first-order error share: tail)\n')
print('| family | n | L | sigma | L\' (c) | capM: top-1 of M / lin / rand / avg / att / pcs | tail: opt / lin / rand / avg / att / pcs | PR worst-case masks | log gain: real median / rand-mask max / op-norm product |')
print('|---|---|---|---|---|---|---|---|---|')
for fam in ('plain', 'res'):
    for n in (8, 16):
        for L in (64, 256):
            for sig in ('0.001',):
                f = os.path.join(R, f'jac_{fam}_n{n}_L{L}_s{sig}.json')
                if not os.path.exists(f): continue
                d = json.load(open(f))
                nets = [v for v in d['res'].values() if not v.get('dead_net')]
                for lp in d['LPS']:
                    cs = [v['cuts'][str(lp)] for v in nets if str(lp) in v['cuts']]
                    if not cs: continue
                    c1 = [c['cands']['1'] for c in cs]
                    cap = '/'.join(f"{np.mean([x[nm]['capM'] if nm != 'topM' else x['capM_opt'] for x in c1]):.2f}" for nm in ('topM', 'lin', 'rand', 'avg', 'att', 'pcs'))
                    tail = '/'.join(f"{np.mean([x[nm]['tail'] for x in c1]):.3f}" for nm in ('opt', 'lin', 'rand', 'avg', 'att', 'pcs'))
                    prw = np.mean([c['pr_worst'] for c in cs])
                    g = f"{np.mean([c['gain_real_median'] for c in cs]):.1f} / {np.mean([c['gain_rand_max'] for c in cs]):.1f} / {np.mean([c['gain_opnorm'] for c in cs]):.1f}"
                    print(f"| {fam} | {n} | {L} | {sig} | {lp} ({L-lp}) | {cap} | {tail} | {prw:.1f} | {g} |")
# per-net spread for the key plain deep-cut capM comparison
print('\n## P5 per net (plain, sigma = 1e-3): capM at k = 1, avg and att against top-1 of M, at the deepest cut with L\' = L/2\n')
for n in (8, 16):
    for L in (64, 256):
        f = os.path.join(R, f'jac_plain_n{n}_L{L}_s0.001.json')
        if not os.path.exists(f): continue
        d = json.load(open(f)); lp = str(L // 2)
        rows = []
        for s, v in d['res'].items():
            if v.get('dead_net') or lp not in v['cuts']: continue
            x = v['cuts'][lp]['cands']['1']
            rows.append(f"seed {s}: top {x['capM_opt']:.3f} avg {x['avg']['capM']:.3f} att {x['att']['capM']:.3f} lin {x['lin']['capM']:.3f} rand {x['rand']['capM']:.3f}")
        print(f"- n={n} L={L} L'={lp}: " + '; '.join(rows))
# collapse census
print('\n## P2/P4: collapse census (per-layer means over nets; plain over 64 nets, residual over 4)\n')
print('| family | n | L | sigma | nets | ever-dead fraction at 16/64/128/256 (mean over nets) | nets with all inputs dead at L | median angle at 4/16/64/128/256 (rad, alive pairs) | gate disagreement (pairs) at 4/16/64/128/256 | noise-only gate disagr. at 64/L | PR(Cov h) at 16/64/L | top-PC share of E[hh^T] at 16/64/L |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for fam in ('plain', 'res'):
    for n in (8, 16):
        for L in (64, 256):
            for sig in ('0', '0.001', '0.01'):
                f = os.path.join(R, f'collapse_{fam}_n{n}_L{L}_s{sig}.json')
                if not os.path.exists(f): continue
                d = json.load(open(f)); res = d['res']
                def col(key, l):
                    vals = [v['rows'][l - 1][key] for v in res.values() if l <= len(v['rows']) and v['rows'][l - 1][key] is not None]
                    return (float(np.median(vals)) if key.startswith('th') else float(np.mean(vals))) if vals else None
                layers = [l for l in (4, 16, 64, 128, 256) if l <= L]
                ed = '/'.join(fmt(col('ever_dead', l), 2) for l in (16, 64, 128, 256) if l <= L)
                alld = sum(1 for v in res.values() if v['rows'][L - 1]['ever_dead'] > 0.99)
                th = '/'.join(fmt(col('th_med', l), 2) for l in layers)
                gd = '/'.join(fmt(col('gd_pair', l), 2) for l in layers)
                gn = '/'.join(fmt(col('gd_noise', l), 2) for l in (64, L))
                prc = '/'.join(fmt(col('pr_cov', l), 2) for l in (16, 64, L))
                top = '/'.join(fmt(col('top_share', l), 3) for l in (16, 64, L))
                print(f"| {fam} | {n} | {L} | {sig} | {len(res)} | {ed} | {alld} | {th} | {gd} | {gn} | {prc} | {top} |")
