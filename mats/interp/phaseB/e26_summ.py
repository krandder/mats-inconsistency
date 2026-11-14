"""E26 analysis (PREREG-B amendment 34): error budget of each net's log-odds deviation from Bayes, by scale.

D(delta; x) = L_net - L_Bayes, L(delta) = log p(y* + delta | x) - log p(y* | x). Variance components (summed over the deltas;
estimators use the E26 variant design, see e26_data.py; c = common context, B_s = (Z_s, u_s)):
  Vtot  = Var(D)                                   (mean of the base and the all-new draw)
  Ec    = E Var(D | c)        = mean (D0 - D1)^2 / 2
  EB_s  = E Var(D | c, B_s)   = [mean (D0 - D[2+s])^2 + mean (D[5+s] - D1)^2] / 4
  Vc    = Vtot - Ec                               context part
  V_s   = Ec - EB_s                               own-source part of source s (main effect of B_s, incl. its interaction with c)
  VX    = sum_s EB_s - 2 Ec                       cross-source interaction = tangledness (all terms with >= 2 source blocks)
  Vcut_s = mean (D0 - D[5+s] - D[2+s] + D1)^2 / 4 interaction across the cut s | rest (exactly 0 for Bayes); VX2 = sum_s Vcut_s / 2
  Tu    = E Var(D | c, Z)     = mean over bases of the 16-draw sample variance  pattern part (every term with a direction u)
  Tsp   = the same over the 9-draw sign/assignment set                           (terms with signs or question assignment)
  Vuo_s = mean (D0 - mean D0) (D[2+s] - D[8+s]) = E Var_{u_s}(E[D | c, Z_s, u_s] | c, Z_s)   own-source pattern part
  Vu_own = sum_s Vuo_s;  Vu_cross = Tu - Vu_own  (pattern terms that involve another source's block; part of VX)
Budget (fractions of Vtot): context Vc; own sufficient sum V_s - Vu_own; own pattern Vu_own; tangled sufficient VX - Vu_cross;
tangled pattern Vu_cross.  KL-exact pattern cost: mean pc16 x 16/15 against mean kl16 (pc9 x 9/8 for signs/assignment).
Pooled ratios = sum over nets of the numerator / sum of the denominator. Intervals: 1,000 bootstrap resamples of the bases
(same resample for every net of a set), conditional on the nets.
usage: python e26_summ.py [SETS...] -> results/e26.json, results/e26_tables.md, results/e26_dyn.png
"""
import json, os, sys
import numpy as np

import e26_data as ED

DSEL_ALL = list(range(6))
DSEL_N5 = [0, 1, 4, 5]                    # deltas -0.6, -0.3, 0.3, 0.6 (the N5/N6 set)
DSEL_SC = [6]                             # the score at y* (delta -> 0 limit; its decomposition is exact for the Fisher divergence)
NBOOT = 1000
OUT = os.environ.get('E26_OUT', 'results/e26')             # plumbing checks only: E26_DATA, E26_OUT, E26_RES override paths
RES = os.environ.get('E26_RES', 'results/e26')
A_BASES = ['w32_h2_L2_mix4_ws_s4', 'w32_h2_L2_mix4_ws_s1', 'w32_h2_L2_mix4_s0', 'w32_h2_L2_mix4_s1', 'w32_h2_L2_mix4_s2',
           'w32_h2_L2_mix4_s6_nb']
B_BASES = ['w32_h2_L2_mix4_ws_s2', 'w32_h2_L2_mix4_ws_s0', 'w32_h2_L2_mix4_ws_s3', 'w32_h2_L2_mix4_s4', 'w32_h2_L2_mix4_s5_nb',
           'w32_h2_L2_mix4_s7_nb']
BAND = {5: [1400, 2000, 2250, 2500], 6: [1700, 1800, 1900, 2000, 2250, 2500, 2750]}
T_ACQ = {5: 4250, 6: 4250}


def columns(D, tu, tsp, pc16, kl16, pc9, kl9, LBd, dsel):
    """Per-base columns whose means give every component. D (n, 12, 6) deviations; LBd (n, 12, 6) Bayes log-odds."""
    D = D[:, :, dsel]; L = D + LBd[:, :, dsel]
    D0, D1 = D[:, 0], D[:, 1]
    cols = [D0, D0 ** 2, D1, D1 ** 2]
    cols.append((0.5 * (D0 - D1) ** 2).sum(1, keepdims=True))
    for s in range(3):
        cols.append((0.25 * ((D0 - D[:, 2 + s]) ** 2 + (D[:, 5 + s] - D1) ** 2)).sum(1, keepdims=True))
    for s in range(3):
        cols.append((0.25 * (D0 - D[:, 5 + s] - D[:, 2 + s] + D1) ** 2).sum(1, keepdims=True))
    cols.append(tu[:, dsel].sum(1, keepdims=True)); cols.append(tsp[:, dsel].sum(1, keepdims=True))
    for s in range(3):
        ds_ = D[:, 2 + s] - D[:, 8 + s]
        cols.append((D0 * ds_).sum(1, keepdims=True)); cols.append(ds_)
    for s in range(3):                                         # N5-style share on the net's log-odds
        Dl = L[:, 0] - L[:, 5 + s] - L[:, 2 + s] + L[:, 1]; Al = L[:, 5 + s] - L[:, 0]; Bl = L[:, 2 + s] - L[:, 0]
        for x in (Dl, Al, Bl):
            cols.append(x.sum(1, keepdims=True)); cols.append((x ** 2).sum(1, keepdims=True))
    for x in (pc16, kl16, pc9, kl9):
        cols.append(np.asarray(x)[:, None])
    return np.concatenate(cols, 1)


def comps_from_means(Mn, m, n):
    """Components from column means Mn (..., ncol) for a delta subset of size m and n bases."""
    i = 0
    def take(k):
        nonlocal i
        out = Mn[..., i:i + k]; i += k
        return out
    D0, D0s, D1, D1s = take(m), take(m), take(m), take(m)
    corr = n / (n - 1)
    Vtot = 0.5 * corr * ((D0s - D0 ** 2).sum(-1) + (D1s - D1 ** 2).sum(-1))
    Ec = take(1)[..., 0]
    EB = [take(1)[..., 0] for _ in range(3)]
    Vcut = [take(1)[..., 0] for _ in range(3)]
    Tu, Tsp = take(1)[..., 0], take(1)[..., 0]
    Vuo = []
    for s in range(3):
        prod = take(1)[..., 0]; dsm = take(m)
        Vuo.append(prod - (D0 * dsm).sum(-1))
    shares = []
    for s in range(3):
        st = []
        for _ in range(3):
            sm, sq = take(1)[..., 0], take(1)[..., 0]
            st.append(sq / m - (sm / m) ** 2)
        shares.append(st[0] / (st[1] + st[2]))
    pc16, kl16, pc9, kl9 = [take(1)[..., 0] for _ in range(4)]
    Vs = [Ec - e for e in EB]
    VX = sum(EB) - 2 * Ec
    Vu_own = sum(Vuo); Vu_cross = Tu - Vu_own
    return dict(Vtot=Vtot, Vc=Vtot - Ec, Vs=sum(Vs), VX=VX, VX2=0.5 * sum(Vcut), Vcut=sum(Vcut), Tu=Tu, Tsp=Tsp,
                Vu_own=Vu_own, Vu_cross=Vu_cross, own_suff=sum(Vs) - Vu_own, X_suff=VX - Vu_cross,
                share_n5=sum(shares) / 3, pc16=pc16 * 16 / 15, kl16=kl16, pc9=pc9 * 9 / 8, kl9=kl9)


_WCACHE = {}


def boot_weights(n, seed=34):
    if n in _WCACHE:
        return _WCACHE[n]
    rng = np.random.default_rng(seed)
    W = np.zeros((NBOOT + 1, n)); W[0] = 1.0
    for b in range(NBOOT):
        W[b + 1] = np.bincount(rng.integers(0, n, n), minlength=n)
    _WCACHE[n] = W / n
    return _WCACHE[n]


def net_comps(f, LB, W, dsel, ref=None):
    z = np.load(f)
    n = int(z['nb'])
    W = boot_weights(n)
    L = z['L'].astype(np.float64)
    R = LB[:n, :12] if ref is None else ref[:n, :12]
    D = L - R
    C = columns(D, z['tu'], z['tsp'], z['pc16'], z['kl16'], z['pc9'], z['kl9'], R, dsel)
    assert W.shape[1] == n, (W.shape, n)
    return comps_from_means(W @ C, len(dsel), n), float(z['mass'].min())


def floor_comps(Lref, LB, W, dsel, n):
    """Components of a u-invariant reference (Bayes-Gaussian, Bayes reduced to 4 components) against exact Bayes."""
    D = Lref[:n, :12] - LB[:n, :12]
    z0 = np.zeros((n, LB.shape[2]))
    C = columns(D, z0, z0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), LB[:n, :12], dsel)
    return comps_from_means(W @ C, len(dsel), n)


FRACS = [('context', lambda c: c['Vc']), ('own, sufficient', lambda c: c['own_suff']), ('own, pattern', lambda c: c['Vu_own']),
         ('tangled, sufficient', lambda c: c['X_suff']), ('tangled, pattern', lambda c: c['Vu_cross'])]


def pooled(cl, num, den):
    a = sum(num(c) for c in cl); b = sum(den(c) for c in cl)
    return a / b


def ci(x):
    return [float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]


def summarise_set(name, rows, LB, W, dsel, ref=None):
    out = {}; cl = []
    for r in rows:
        f = os.path.join(OUT, name, r['tag'] + '.npz')
        if not os.path.exists(f):
            continue
        c, mass = net_comps(f, LB, W, dsel, ref)
        cl.append(c)
        out[r['tag']] = dict(mass_min=mass, **{k: float(v[0]) for k, v in c.items()})
    return out, cl


def budget(cl):
    res = {k: ci(pooled(cl, fn, lambda c: c['Vtot'])) for k, fn in FRACS}
    res['tangled (2x2 cuts)'] = ci(pooled(cl, lambda c: c['VX2'], lambda c: c['Vtot']))
    res['pattern total Tu'] = ci(pooled(cl, lambda c: c['Tu'], lambda c: c['Vtot']))
    res['sign/assignment Tsp'] = ci(pooled(cl, lambda c: c['Tsp'], lambda c: c['Vtot']))
    res['own share of pattern'] = ci(pooled(cl, lambda c: c['Vu_own'], lambda c: c['Tu']))
    res['pattern cost / KL'] = ci(pooled(cl, lambda c: c['pc16'], lambda c: c['kl16']))
    res['sign/assignment cost / KL'] = ci(pooled(cl, lambda c: c['pc9'], lambda c: c['kl9']))
    res['mean KL'] = ci(sum(c['kl16'] for c in cl) / len(cl))
    res['mean Vtot'] = ci(sum(c['Vtot'] for c in cl) / len(cl))
    res['N5 share (net log-odds)'] = ci(sum(c['share_n5'] for c in cl) / len(cl))
    return res


def main():
    sets = sys.argv[1:] or ['M', 'G', 'L1', 'NZ', 'HD', 'DY5', 'DY6']
    z = np.load(os.environ.get('E26_DATA', 'results/e26_data.npz')); LB, LG, LT = z['LB'], z['LG'], z['LT']
    rows = json.load(open('results/e26_nets.json'))
    Ws = {n: boot_weights(n) for n in (6000, 4000) if n <= len(LB)}
    res = dict(n_bases=dict(M=6000, G=6000, L1=6000, HD=6000, NZ=4000, DY=6000))
    md = ['# E26 tables (generated by e26_summ.py)', '']
    # reference floors
    fl = {}; fl_arr = {}
    for nm, Lr in (('Bayes moment-matched Gaussian', LG), ('Bayes reduced to 4 components', LT)):
        for dn, ds_ in (('', DSEL_ALL), (' (score)', DSEL_SC)):
            nf = min(6000, len(LB)); c = floor_comps(Lr, LB, boot_weights(nf), ds_, nf)
            fl[nm + dn] = {k: float(v[0]) for k, v in c.items()}; fl_arr[nm + dn] = c
    res['floors'] = fl
    md += ['## Reference floors (deviation from exact Bayes; no pattern terms by construction)', '',
           '| reference | Vtot | context | own | tangled VX | VX (2x2) | N5 share |', '|---|---|---|---|---|---|---|']
    for nm, c in fl.items():
        md.append('| %s | %.4f | %.3f | %.3f | %.3f | %.3f | %.4f |' % (nm, c['Vtot'], c['Vc'] / c['Vtot'], c['Vs'] / c['Vtot'],
                                                                      c['VX'] / c['Vtot'], c['VX2'] / c['Vtot'], c['share_n5']))
    md.append('')
    for name in sets:
        rr = [r for r in rows if r['set'] == name]
        if not rr:
            continue
        n = {'NZ': 4000}.get(name, 6000)
        per, cl = summarise_set(name, rr, LB, Ws[n], DSEL_ALL)
        if not cl:
            continue
        S = dict(per_net=per, n_nets=len(cl))
        if not name.startswith('DY'):
            S['budget'] = budget(cl)
            S['budget_n5_deltas'] = budget(summarise_set(name, rr, LB, Ws[n], DSEL_N5)[1])
            S['budget_score'] = budget(summarise_set(name, rr, LB, Ws[n], DSEL_SC)[1])
            if name in ('G', 'L1'):
                S['budget_vs_gauss'] = budget(summarise_set(name, rr, LB, Ws[n], DSEL_ALL, ref=LG)[1])
            groups = {}
            if name == 'M':
                groups = dict(A=[r for r in rr if r['tag'] in A_BASES], B=[r for r in rr if r['tag'] in B_BASES])
            if name == 'NZ':
                groups = dict(A=[r for r in rr if r.get('group') == 'A'], B=[r for r in rr if r.get('group') == 'B'])
            if name == 'HD':
                groups = {a: [r for r in rr if r['arm'] == a] for a in ('base_g', 'L3_g', 'base_ws', 'L3_ws')}
            S['groups'] = {g: budget(summarise_set(name, gr, LB, Ws[n], DSEL_ALL)[1]) for g, gr in groups.items() if gr}
        res[name] = S
        if name.startswith('DY'):
            continue
        md += [f'## Set {name} ({len(cl)} nets): pooled budget, fractions of the deviation variance [95% CI]', '']
        md.append('| quantity | all six deltas | N5 deltas | score at y* |'); md.append('|---|---|---|---|')
        for k in S['budget']:
            a, b, c3 = S['budget'][k], S['budget_n5_deltas'][k], S['budget_score'][k]
            md.append('| %s | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] | %.4f [%.4f, %.4f] |' % (k, *a, *b, *c3))
        md.append('')
        if S.get('groups'):
            md += ['| group | ' + ' | '.join(k for k, _ in FRACS) + ' | Tu | own share of pattern | pattern cost / KL | mean KL | mean Vtot |',
                   '|---' * (len(FRACS) + 6) + '|']
            for g, b in S['groups'].items():
                md.append('| %s | ' % g + ' | '.join('%.3f' % b[k][0] for k, _ in FRACS) +
                          ' | %.3f | %.3f | %.3f | %.4f | %.4f |' % (b['pattern total Tu'][0], b['own share of pattern'][0],
                                                                      b['pattern cost / KL'][0], b['mean KL'][0], b['mean Vtot'][0]))
            md.append('')
        md += ['| net | Vtot | context | own suff | own pat | tangled suff | tangled pat | own share of pattern | pattern cost / KL | KL | N5 share |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for t, c in per.items():
            md.append('| %s | %.4f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f | %.4f | %.4f |' % (
                t, c['Vtot'], c['Vc'] / c['Vtot'], c['own_suff'] / c['Vtot'], c['Vu_own'] / c['Vtot'], c['X_suff'] / c['Vtot'],
                c['Vu_cross'] / c['Vtot'], c['Vu_own'] / c['Tu'], c['pc16'] / c['kl16'], c['kl16'], c['share_n5']))
        md.append('')
    # dynamics
    dyn = {}
    for s in (5, 6):
        name = f'DY{s}'
        if name not in res:
            continue
        rr = sorted([r for r in rows if r['set'] == name], key=lambda r: r['step'])
        traj = []
        boots = {}
        for r in rr:
            f = os.path.join(OUT, name, r['tag'] + '.npz')
            if not os.path.exists(f):
                continue
            c, _ = net_comps(f, LB, Ws[6000], DSEL_ALL)
            boots[r['step']] = c
            traj.append(dict(step=r['step'], **{k: float(v[0]) for k, v in c.items()}))
        band = [r for r in traj if r['step'] in BAND[s]]; post = [r for r in traj if r['step'] >= T_ACQ[s] + 1000]
        def med_cross(steps):
            arr = np.stack([boots[t]['Vu_cross'] / boots[t]['Tu'] for t in steps], 0)
            return ci(np.median(arr, 0))
        dyn[s] = dict(traj=traj, band_steps=[r['step'] for r in band],
                      band_median_cross=med_cross([r['step'] for r in band]) if band else None,
                      post_median_cross=med_cross([r['step'] for r in post]) if post else None, n_post=len(post))
    res['dynamics'] = dyn
    # predictions
    pr = {}
    if 'M' in res:
        b = res['M']['budget']; per = res['M']['per_net']
        own = [c['Vu_own'] / c['Tu'] for c in per.values()]
        pr['P137'] = dict(pooled_own_share=b['own share of pattern'], per_net_ge_0p6=int(sum(x >= 0.6 for x in own)), n=len(own),
                          holds=bool(b['own share of pattern'][0] >= 0.7 and sum(x >= 0.6 for x in own) >= 12))
        pr['P139'] = dict(pattern_share=b['pattern total Tu'], cost_share=b['pattern cost / KL'],
                          holds=bool(b['pattern total Tu'][0] <= 0.2 and b['pattern cost / KL'][0] <= 0.2))
        cl = summarise_set('M', [r for r in rows if r['set'] == 'M'], LB, Ws[6000], DSEL_ALL)[1]
        vx = ci(pooled(cl, lambda c: c['VX'], lambda c: c['Vtot']))
        vf = ci(sum(c['VX'] for c in cl) / (len(cl) * fl_arr['Bayes reduced to 4 components']['VX']))
        pr['P140'] = dict(tangled_share=vx, mean_VX_over_4comp_floor_VX=vf, holds=bool(vx[0] >= 0.25 and vf[0] >= 2.0))
    if 'L1' in res:
        b = res['L1']['budget']
        cross = [1 - x for x in b['own share of pattern']]
        pr['P138'] = dict(pooled_cross_share=[cross[0], cross[2], cross[1]], holds=bool(cross[0] >= 0.4))
    if dyn:
        ok = []
        for s, o in dyn.items():
            ok.append(o['band_median_cross'] is not None and o['band_median_cross'][0] >= 0.4 and o['post_median_cross'][0] <= 0.3)
        pr['P141'] = dict(per_lineage={s: dict(band=o['band_median_cross'], post=o['post_median_cross']) for s, o in dyn.items()},
                          holds=bool(len(ok) == 2 and all(ok)))
    res['predictions'] = pr
    # descriptive: relations across the 48 acquired clean nets (M and G), Spearman
    try:
        e15 = json.load(open('results/e15.json'))['per_net']
        tab = []
        for nm in ('M', 'G'):
            for t, c in res.get(nm, {}).get('per_net', {}).items():
                pn = e15.get(t + '__best')
                if pn is None:
                    continue
                tab.append([c['Tu'] / c['Vtot'], c['VX'] / c['Vtot'], c['kl16'], c['pc16'] / c['kl16'], pn['large']['rom'],
                            c['Vu_own'] / c['Tu'], c['Vtot']])
        tab = np.array(tab)
        names = ['pattern share', 'tangled share', 'KL', 'pattern cost / KL', 'rom large (E15)', 'own share of pattern', 'Vtot']
        rk = np.argsort(np.argsort(tab, 0), 0).astype(float)
        sp = np.corrcoef(rk.T)
        res['cross_net_spearman'] = dict(n=len(tab), names=names, matrix=sp.tolist())
        md += ['## Across the acquired clean nets (sets M and G; Spearman, descriptive)', '', '| | ' + ' | '.join(names) + ' |',
               '|---' * (len(names) + 1) + '|']
        for i, a in enumerate(names):
            md.append('| %s | ' % a + ' | '.join('%.2f' % sp[i, j] for j in range(len(names))) + ' |')
        md.append('')
    except Exception as ex:                                          # descriptive only
        print('cross-net table failed', ex)
    md += ['## Predictions', '']
    for k, v in pr.items():
        md.append('- %s: %s' % (k, json.dumps(v)))
    md.append('')
    if dyn:
        md += ['## Dynamics (point estimates; fractions of Vtot)', '',
               '| lineage | step | KL | Vtot | context | own suff | own pat | tangled suff | tangled pat | cross share of pattern | pattern cost / KL |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for s, o in dyn.items():
            for r in o['traj']:
                md.append('| %d | %d | %.4f | %.4f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.3f |' % (
                    s, r['step'], r['kl16'], r['Vtot'], r['Vc'] / r['Vtot'], r['own_suff'] / r['Vtot'], r['Vu_own'] / r['Vtot'],
                    r['X_suff'] / r['Vtot'], r['Vu_cross'] / r['Vtot'], r['Vu_cross'] / r['Tu'], r['pc16'] / r['kl16']))
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            for s, o in dyn.items():
                st = np.array([r['step'] for r in o['traj']])
                ax[0].plot(st, [r['kl16'] for r in o['traj']], label=f's{s} KL'); ax[0].plot(st, [r['pc16'] for r in o['traj']], '--', label=f's{s} pattern cost')
                for k, fn in FRACS:
                    ax[1].plot(st, [fn(r) / r['Vtot'] for r in o['traj']], label=f's{s} {k}')
                ax[2].plot(st, [r['Vu_cross'] / r['Tu'] for r in o['traj']], label=f's{s}')
            for a in ax:
                a.set_xscale('symlog', linthresh=1000); a.legend(fontsize=6)
            ax[0].set_yscale('log'); ax[0].set_title('KL and pattern cost (nats)'); ax[1].set_title('budget fractions'); ax[2].set_title('cross share of pattern')
            fig.tight_layout(); fig.savefig(RES + '_dyn.png', dpi=110)
        except Exception as ex:                                      # plotting is optional
            print('plot failed', ex)
    json.dump(res, open(RES + '.json', 'w'), indent=1, default=float)
    open(RES + '_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
