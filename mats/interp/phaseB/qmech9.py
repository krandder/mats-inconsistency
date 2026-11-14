"""E9 analysis (PREREG-B amendment 16): path shares of the sharpening, flattening by patches, shared-error measures.
usage: python qmech9.py -> results/qmech9.json, results/qmech9_tables.md
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import qbayes as QB
import qmech as QM

A_NETS, B_NETS, ALL, GROUP = QM.A_NETS, QM.B_NETS, QM.ALL, QM.GROUP
N4, NB = 4000, 500
VARS = ['PVg', 'PV', 'PM', 'PVgPM', 'PVPM']
PARTS = ['R_direct', 'R_mlp', 'M_direct', 'M_mlp', 'V_direct', 'V_mlp']
LABEL = {'PVg': 'PVg value gain removed', 'PV': 'PV values = linear in reading', 'PM': 'PM MLP2 linearised',
         'PVgPM': 'PVg + PM', 'PVPM': 'PV + PM'}
f = QM.f
cov = QM.cov


def slope(y, x): return float(np.polyfit(x.ravel(), y.ravel(), 1)[0])


def sharp(w, p):
    s = w.sum(1); ok = s > 0.2
    return slope(w[ok] / s[ok, None], p[ok]), float(ok.mean())


def main():
    t8 = QM.test8000(); d4 = {k: v[:N4] for k, v in t8['d'].items()}
    p = QB.bayes_features(d4)[:, 20:23].astype(np.float64); p /= p.sum(1, keepdims=True)
    o = d4['o'][:, :, 4]; dis = o.std(1); DIS = dis >= np.quantile(dis, 2 / 3)
    Z = {nm: np.load(os.path.join('runs', nm, 'qe9.npz')) for nm in ALL if os.path.exists(os.path.join('runs', nm, 'qe9.npz'))}
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in ALL}
    E8 = {nm: np.load(os.path.join('runs', nm, 'qpatch.npz')) for nm in Z if os.path.exists(os.path.join('runs', nm, 'qpatch.npz'))}
    out = dict(nets=list(Z), per_net={}, by_group={}, predictions={}, n_dis=int(DIS.sum()),
               base_matches_E8={nm: float(np.abs(Z[nm]['base_mean'] - E8[nm]['base_mean']).max()) for nm in E8})
    rng = np.random.default_rng(0); boots = [rng.integers(0, N4, N4) for _ in range(NB)]
    ref = {}
    for key in ('mean', 'logsd'):
        R2s = {nm: S[nm][f'r2_{key}'][:N4] for nm in ALL}
        for nm in Z:
            ref[(nm, key)] = np.mean([R2s[x] for x in (A_NETS if GROUP[nm] == 'A' else B_NETS) if x != nm], 0)
    for nm, z in Z.items():
        q = z['masses_qq'].sum(1); q = q / q.sum(1, keepdims=True); w = z['path_full']
        D = w - w.sum(1, keepdims=True) * q; dd = (D * D).sum()
        shares = {pt: float(((z[f'path_{pt}'] - z[f'path_{pt}'].sum(1, keepdims=True) * q) * D).sum() / dd) for pt in PARTS}
        Sb, okb = sharp(w, p); Sq = slope(q, p)
        r = dict(group=GROUP[nm], shares=shares, S_base=Sb, S_q=Sq, ok_base=okb,
                 sens={pt: float(z[f'path_{pt}'].sum(1).mean()) for pt in PARTS}, sens_full=float(w.sum(1).mean()), patches={})
        for vr in VARS:
            Sp, okp = sharp(z[f'w_{vr}'], p)
            pr = dict(S=Sp, ok=okp, f=(Sb - Sp) / (Sb - Sq), sens=float(z[f'w_{vr}'].sum(1).mean()))
            for key in ('mean', 'logsd'):
                base, dv, rf = z[f'base_{key}'], z[f'{vr}_{key}'], ref[(nm, key)]
                pr[key] = dict(retained=cov(dv, rf) / cov(base, rf), retained_dis=cov(dv[DIS], rf[DIS]) / cov(base[DIS], rf[DIS]),
                               msd=float((dv ** 2).mean() / (base ** 2).mean()), msd_dis=float((dv[DIS] ** 2).mean() / (base[DIS] ** 2).mean()))
            pr['kl'] = float(z[f'{vr}_kl'].mean() / z['base_kl'].mean()); pr['kl_dis'] = float(z[f'{vr}_kl'][DIS].mean() / z['base_kl'][DIS].mean())
            r['patches'][vr] = pr
        bp = os.path.join('runs', nm, 'qe9b.npz')
        if os.path.exists(bp):
            own = np.load(bp)['own']; V = z['path_V_direct'] + z['path_V_mlp']; cross = V - own
            sh = lambda x: float(((x - x.sum(1, keepdims=True) * q) * D).sum() / dd)
            r['own_cross'] = dict(own=sh(own), cross=sh(cross), sens_own=float(own.sum(1).mean()), sens_cross=float(cross.sum(1).mean()))
        gp = os.path.join('runs', nm, 'qgain.npz')
        if os.path.exists(gp):
            gz = np.load(gp); gg, AA = gz['g'], gz['A']
            gbar = (AA * gg).sum(1) / AA.sum(1); u = (AA * gg).sum(1); un = u / u.sum(1, keepdims=True)
            Su = slope(un, p); s_ = w.sum(1); ok = s_ > 0.2
            r['gain'] = dict(corr_gain_p=float(np.corrcoef(gbar.ravel(), p.ravel())[0, 1]), mean_gain=float(gbar.mean()), S_u=Su,
                             share=(Su - Sq) / (Sb - Sq), corr_u_w=float(np.corrcoef(un[ok].ravel(), (w[ok] / s_[ok, None]).ravel())[0, 1]),
                             corr_q_w=float(np.corrcoef(q[ok].ravel(), (w[ok] / s_[ok, None]).ravel())[0, 1]))
        out['per_net'][nm] = r
    for g, nets0 in (('A', A_NETS), ('B', B_NETS)):
        nets = [nm for nm in nets0 if nm in Z]
        if not nets: continue
        P = out['per_net']; G = dict(n=len(nets))
        agg = lambda vals: dict(mean=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals)))
        G['shares'] = {pt: agg([P[nm]['shares'][pt] for nm in nets]) for pt in PARTS}
        G['shares_path'] = {k: agg([P[nm]['shares'][f'{k}_direct'] + P[nm]['shares'][f'{k}_mlp'] for nm in nets]) for k in ('R', 'M', 'V')}
        G['shares_route'] = {k: agg([sum(P[nm]['shares'][f'{x}_{k}'] for x in ('R', 'M', 'V')) for nm in nets]) for k in ('direct', 'mlp')}
        G['sens'] = {pt: agg([P[nm]['sens'][pt] for nm in nets]) for pt in PARTS}
        G['S_base'] = agg([P[nm]['S_base'] for nm in nets]); G['S_q'] = agg([P[nm]['S_q'] for nm in nets])
        G['patches'] = {}
        for vr in VARS:
            e = dict(f=agg([P[nm]['patches'][vr]['f'] for nm in nets]), S=agg([P[nm]['patches'][vr]['S'] for nm in nets]),
                     kl=agg([P[nm]['patches'][vr]['kl'] for nm in nets]), kl_dis=agg([P[nm]['patches'][vr]['kl_dis'] for nm in nets]))
            for key in ('mean', 'logsd'):
                e[key] = {m: agg([P[nm]['patches'][vr][key][m] for nm in nets]) for m in ('retained', 'retained_dis', 'msd', 'msd_dis')}
                bs = [float(np.mean([cov(Z[nm][f'{vr}_{key}'][ii], ref[(nm, key)][ii]) / cov(Z[nm][f'base_{key}'][ii], ref[(nm, key)][ii]) for nm in nets])) for ii in boots]
                e[key]['retained']['ci'] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            G['patches'][vr] = e
        on = [nm for nm in nets if 'own_cross' in P[nm]]
        if on: G['own_cross'] = {k: agg([P[nm]['own_cross'][k] for nm in on]) for k in ('own', 'cross', 'sens_own', 'sens_cross')}; G['own_cross']['n'] = len(on)
        gn = [nm for nm in nets if 'gain' in P[nm]]
        if gn: G['gain'] = {k: agg([P[nm]['gain'][k] for nm in gn]) for k in ('corr_gain_p', 'mean_gain', 'S_u', 'share', 'corr_u_w', 'corr_q_w')}; G['gain']['n'] = len(gn)
        out['by_group'][g] = G
        print(g, 'shares by path', {k: round(v['mean'], 2) for k, v in G['shares_path'].items()}, 'by route', {k: round(v['mean'], 2) for k, v in G['shares_route'].items()},
              'S_base %.2f S_q %.2f' % (G['S_base']['mean'], G['S_q']['mean']), flush=True)
        for vr in VARS:
            e = G['patches'][vr]
            print('  ', vr, 'f %.2f' % e['f']['mean'], 'mean: ret %.2f dis %.2f msd %.2f' % (e['mean']['retained']['mean'], e['mean']['retained_dis']['mean'], e['mean']['msd']['mean']),
                  '| logsd: ret %.2f msd %.2f | kl %.2f' % (e['logsd']['retained']['mean'], e['logsd']['msd']['mean'], e['kl']['mean']), flush=True)
    if 'A' in out['by_group']:
        GA = out['by_group']['A']; pr = out['predictions']
        pr['P79'] = dict(value=GA['shares_path']['V']['mean'], holds=bool(GA['shares_path']['V']['mean'] >= 0.5))
        pr['P80'] = dict(value=GA['shares_route']['mlp']['mean'], holds=bool(GA['shares_route']['mlp']['mean'] < 0.25))
        pr['P81'] = dict(PVg=GA['patches']['PVg']['f']['mean'], PM=GA['patches']['PM']['f']['mean'],
                         holds=bool(GA['patches']['PVg']['f']['mean'] >= 0.5 and GA['patches']['PM']['f']['mean'] < 0.25))
        hits = [(g, vr) for g in out['by_group'] for vr in VARS if out['by_group'][g]['patches'][vr]['f']['mean'] >= 0.5]
        pr['P82'] = dict(patches=[f'{g}:{vr}' for g, vr in hits], holds=bool(hits) and all(out['by_group'][g]['patches'][vr]['mean']['retained']['mean'] >= 1.2
                                                                                           and out['by_group'][g]['patches'][vr]['mean']['msd']['mean'] >= 1.5 for g, vr in hits))
        hits2 = [(g, vr) for g in out['by_group'] for vr in VARS if out['by_group'][g]['patches'][vr]['f']['mean'] >= 0.25]
        pr['P83'] = dict(patches=[f'{g}:{vr}' for g, vr in hits2], holds=bool(hits2) and all(
            abs(out['by_group'][g]['patches'][vr]['mean']['retained_dis']['mean'] - 1) > abs(out['by_group'][g]['patches'][vr]['mean']['retained']['mean'] - 1) for g, vr in hits2))
        if 'own_cross' in GA:
            pr['P85'] = dict(own=GA['own_cross']['own']['mean'], holds=bool(GA['own_cross']['own']['mean'] >= 0.5))
        if 'gain' in GA:
            pr['P84'] = dict(share=GA['gain']['share']['mean'], corr=GA['gain']['corr_gain_p']['mean'],
                             holds=bool(GA['gain']['share']['mean'] >= 0.5 and GA['gain']['corr_gain_p']['mean'] >= 0.5))
    json.dump(out, open('results/qmech9.json', 'w'), indent=1)
    tables(out)


def tables(o):
    L = []
    for g, G in o['by_group'].items():
        L += [f"\n### E9.1 Group {g} ({G['n']} nets): share of the sharpening (w − Σw·q) carried by each part, mean [range]\n",
              '| part | direct route | through MLP2 | both routes |', '|---|---|---|---|']
        for k, lab in (('R', 'R: query token\'s own residual (from layer 1)'), ('M', 'M: attention logits (readings moving the masses)'), ('V', 'V: values')):
            a, b, c = G['shares'][f'{k}_direct'], G['shares'][f'{k}_mlp'], G['shares_path'][k]
            L.append(f"| {lab} | {f(a['mean'])} [{f(a['lo'])}, {f(a['hi'])}] | {f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}] | {f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}] |")
        a, b = G['shares_route']['direct'], G['shares_route']['mlp']
        L.append(f"| all paths | {f(a['mean'])} [{f(a['lo'])}, {f(a['hi'])}] | {f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}] | 1 |")
        L.append(f"\nSharpness (slope of normalised weights on the Bayes posterior share): effective weights {f(G['S_base']['mean'])} [{f(G['S_base']['lo'])}, {f(G['S_base']['hi'])}], "
                 f"QQ masses {f(G['S_q']['mean'])} [{f(G['S_q']['lo'])}, {f(G['S_q']['hi'])}]. Mean total sensitivity by part: "
                 + ', '.join(f"{pt} {G['sens'][pt]['mean']:.3f}" for pt in PARTS) + '.')
        L += [f"\n### E9.2 Group {g}: patches (flattening f: 1 = weights as flat as the masses; retained: 1 = shared pattern unchanged)\n",
              '| patch | f [range] | mean error retained [95% CI] / on disagreeing third | mean MSD (all / disagreeing) | spread retained / disagreeing | log-sd MSD | KL ratio (all / disagreeing) |',
              '|---|---|---|---|---|---|---|']
        for vr in VARS:
            e = G['patches'][vr]; m, s = e['mean'], e['logsd']
            L.append(f"| {LABEL[vr]} | {f(e['f']['mean'])} [{f(e['f']['lo'])}, {f(e['f']['hi'])}] | {f(m['retained']['mean'])} [{f(m['retained']['ci'][0])}, {f(m['retained']['ci'][1])}] / {f(m['retained_dis']['mean'])} | "
                     f"{f(m['msd']['mean'])} / {f(m['msd_dis']['mean'])} | {f(s['retained']['mean'])} / {f(s['retained_dis']['mean'])} | {f(s['msd']['mean'])} | {f(e['kl']['mean'])} / {f(e['kl_dis']['mean'])} |")
    L += ["\n### E9.4 (post hoc) Value path split: each source's own QQ-token value vs all other tokens' values (share of the sharpening)\n",
          "| group | own QQ value | other tokens' values | mean sensitivity own / cross |", '|---|---|---|---|']
    for g, G in o['by_group'].items():
        if 'own_cross' not in G: continue
        e = G['own_cross']
        L.append(f"| {g} ({e['n']} nets) | {f(e['own']['mean'])} [{f(e['own']['lo'])}, {f(e['own']['hi'])}] | {f(e['cross']['mean'])} [{f(e['cross']['lo'])}, {f(e['cross']['hi'])}] | {e['sens_own']['mean']:.3f} / {e['sens_cross']['mean']:.3f} |")
    L += ['\n### E9.3 (post hoc) Reading gains of the QQ values: masses x gains vs the effective weights\n',
          '| group | corr(gain, Bayes share p) | mean gain | slope of masses x gains on p | share of the sharpening reproduced | corr(masses x gains, effective weights) | corr(masses, effective weights) |',
          '|---|---|---|---|---|---|---|']
    for g, G in o['by_group'].items():
        if 'gain' not in G: continue
        e = G['gain']; c = lambda k: f"{f(e[k]['mean'])} [{f(e[k]['lo'])}, {f(e[k]['hi'])}]"
        L.append(f"| {g} ({e['n']} nets) | {c('corr_gain_p')} | {c('mean_gain')} | {c('S_u')} | {c('share')} | {c('corr_u_w')} | {c('corr_q_w')} |")
    L.append('\n### Predictions\n')
    for k, v in o['predictions'].items(): L.append(f"- {k}: {'holds' if v['holds'] else 'fails'} ({', '.join(f'{a} {b:.2f}' if isinstance(b, float) else f'{a} {b}' for a, b in v.items() if a != 'holds')})")
    open('results/qmech9_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
