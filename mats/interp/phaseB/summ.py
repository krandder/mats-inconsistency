"""Tables for the phase B report (markdown to results/tables.md)."""
import json, os, glob, collections, math
import numpy as np

B = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(B, 'results')
SIG = ['0.0', '0.05', '0.1', '0.2', '0.4', '0.8', '1.6']
out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))
f2 = lambda v: '–' if v is None else f'{v:.2f}'
br = lambda lo, hi: f'[{lo:.1f}, {hi:.1f}]'


def task1():
    P('## Task 1: random nets\n')
    V = json.load(open(os.path.join(R, 'task1_validate.json')))
    P('### Estimator validation (linear-Gaussian, exact I; bits)\n')
    P('| D | scale | log2 M | I1 exact | I1 bracket | I2 exact | I2 bracket |'); P('|---|---|---|---|---|---|---|')
    for v in V:
        P(f"| {v['D']} | {v['scale']} | {v['log2M']:.0f} | {v['I1']:.2f} | {br(v['I1_lo'], v['I1_hi'])} | {v['I2']:.2f} | {br(v['I2_lo'], v['I2_hi'])} |")
    if os.path.exists(os.path.join(R, 'task1_xg.json')):
        X = json.load(open(os.path.join(R, 'task1_xg.json')))
        P('\nx-level Gaussian estimator, linear check (M_x = 4096): ' + '; '.join(
            f"D={x['D']}: [{x['xg_I_lo']:.2f}, {x['xg_I_hi']:.2f}] vs exact {x['I_true']:.2f}" for x in X if x.get('check')))
        xg = {(x['din'], x['d'], x['sigma'], x['seed']): x for x in X if not x.get('check')}
    else:
        xg = {}
    F = json.load(open(os.path.join(R, 'task1_flow.json')))
    cfg = collections.defaultdict(dict)
    for r in F:
        cfg[(r['din'], r['d'], r['sigma'], r['seed'])][r['layer']] = r
    P('\n### Information per layer (bits): parent-mixture bracket / x-level Gaussian estimate; capacity C; plug-in ReLU pattern entropy\n')
    P('| din | d | sigma | seed | C | I(X;H1) | I(X;H2) | I(X;H3) | xg H1 / H2 / H3 | H(pattern) L1 / L2 / L3 |'); P('|---|---|---|---|---|---|---|---|---|---|')
    for k in sorted(cfg):
        v = cfg[k]; x = xg.get(k)
        xs = ' / '.join(br(x[f'xg{l}_lo'], x[f'xg{l}_hi']) for l in (1, 2, 3)) if x else '–'
        P(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v[1]['cap']:.1f} | " + ' | '.join(br(v[l]['I_lo'], v[l]['I_hi']) for l in (1, 2, 3))
          + f" | {xs} | " + ' / '.join(f"{v[l]['H_pattern_plugin']:.1f}" for l in (1, 2, 3)) + ' |')
    p2 = os.path.join(R, 'task1_coarse2.json')
    if os.path.exists(p2):
        C = json.load(open(p2))
        P('\n### Coarse-graining (x-level Gaussian): I(K;H3) for classes K with 2^b cells\n')
        P('| din | d | sigma | I(X;H3) | classes | b | H(K) | I(K;H3) | R^2 of E[H3|x] by cell mean |'); P('|---|---|---|---|---|---|---|---|---|')
        for c in C:
            if c['b'] in (1, 2, 3, 4, 5, 6, 8, 10, 11):
                P(f"| {c['din']} | {c['d']} | {c['sigma']} | {br(c['I3_lo'], c['I3_hi'])} | {c['scheme']} | {c['b']} | {c['HK']:.2f} | {br(c['IK_lo'], c['IK_hi'])} | {c['r2_m']:.3f} |")


def task2():
    P('\n## Task 2: noisy tiny PFNs\n')
    P('| model | sigma | KL marg (nats) | KL single draw | NLL gap | cost vs sigma=0 |'); P('|---|---|---|---|---|---|')
    base = {}
    for m in ('L2', 'L3'):
        for s in SIG:
            p = os.path.join(B, 'runs', f'{m}_s{s}', 'eval.json')
            if not os.path.exists(p): continue
            e = json.load(open(p))
            if s == '0.0': base[m] = e['kl_marg']
            c = e['kl_marg'] - base.get(m, float('nan'))
            P(f"| {m} | {s} | {e['kl_marg']:.4f} ± {e['kl_marg_se']:.4f} | {e['kl_single']:.4f} | {e['gap']:.4f} | {c:+.4f} |")
    P('\n### I(X; H_site) (bits): parent-mixture bracket (M = 2^15) | x-level Gaussian (M_x = 4096) | per-site capacity\n')
    for m in ('L2', 'L3'):
        for s in SIG[1:]:
            p = os.path.join(B, 'runs', f'{m}_s{s}', 'mi.json')
            if not os.path.exists(p): continue
            M = json.load(open(p))
            xgp = os.path.join(B, 'runs', f'{m}_s{s}', 'mi_xg.json')
            xg = {(x['site'], x['group']): x for x in json.load(open(xgp))} if os.path.exists(xgp) else {}
            P(f'\n**{m}, sigma = {s}**\n')
            P('| site | QT bracket | QT xg | QQ0 bracket | QQ0 xg | ALL bracket | C token / C layer |'); P('|---|---|---|---|---|---|---|')
            rows = collections.defaultdict(dict)
            for r in M: rows[r['site']][r['group']] = r
            for site, g in rows.items():
                xq = xg.get((site, 'QT')); x0 = xg.get((site, 'QQ0'))
                P(f"| {site} | {br(g['QT']['I_lo'], g['QT']['I_hi'])} | {br(xq['xg_I_lo'], xq['xg_I_hi']) if xq else '–'} | "
                  f"{br(g['QQ0']['I_lo'], g['QQ0']['I_hi'])} | {br(x0['xg_I_lo'], x0['xg_I_hi']) if x0 else '–'} | "
                  f"{br(g['ALL']['I_lo'], g['ALL']['I_hi'])} | {g['QT']['cap']:.0f} / {g['ALL']['cap']:.0f} |")


def task3():
    P('\n## Task 3: q families\n')
    for p in sorted(glob.glob(os.path.join(B, 'runs', '*', 'fam.json'))):
        name = os.path.basename(os.path.dirname(p)); J = json.load(open(p))
        xgp = os.path.join(os.path.dirname(p), 'mi_xg.json')
        xg = {(x['site'], x['group']): x for x in json.load(open(xgp))} if os.path.exists(xgp) else {}
        P(f'\n**{name}** (bits; eps = KL(p||q) bracket; wasted = KL(q||p) upper estimate; eff = I_xg / bits_desc)\n')
        P('| site | tok | fam | eps | wasted <= | bits_rate | bits_desc | lost code H(K|H) | I_xg | eff | extra |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in J['rows']:
            x = xg.get((r['site'], r['group']))
            I = (x['xg_I_lo'] + x['xg_I_hi']) / 2 if x else None
            eff = I / r['bits_desc'] if I is not None and r['bits_desc'] > 0 else None
            extra = ', '.join(f'{k}={r[k]:.3g}' if isinstance(r[k], float) else f'{k}={r[k]}' for k in ('ncell', 'npat', 'nsup', 'fve', 'r2', 'singletons') if k in r)
            P(f"| {r['site']} | {r['group']} | {r['fam']} | {br(r['eps_lo'], r['eps_hi'])} | {r['wasted_ub']:.1f} | {r['bits_rate']:.1f} | {r['bits_desc']:.1f} | "
              f"{r['lost_code']:.1f} | {f2(I)} | {f2(eff)} | {extra} |")
        if J.get('chain'):
            P('\nChain a_l -> r_l (token-local): KL(F#q_a || q_r) bracket vs eps(q_r); F4: transcoder in-degree for 90% of dense R^2\n')
            P('| l | tok | fam | chain KL | eps(q_r) | graph: feats / median dense R^2 / median in-degree |'); P('|---|---|---|---|---|---|')
            for c in J['chain']:
                gtxt = f"{c['graph_nfeat']} / {c['graph_r2d_med']:.2f} / {c['graph_deg_med']:.0f}" if 'graph_nfeat' in c else '–'
                P(f"| {c['l']} | {c['group']} | {c['fam']} | {br(c['chain_lo'], c['chain_hi'])} | {br(c['eps_r_lo'], c['eps_r_hi'])} | {gtxt} |")


if __name__ == '__main__':
    for f in (task1, task2, task3):
        try:
            f()
        except Exception as e:
            P(f'\n({f.__name__} incomplete: {e!r})\n')
    open(os.path.join(R, 'tables.md'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
