"""E15 analysis (PREREG-B amendment 22): the error-spread departure in clean transformers.
usage: python e15_summ.py -> results/e15.json, results/e15_tables.md
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import nnoise  # noqa: F401
import qmech as QM

NBOOT = 1000
BASES12 = ['w32_h2_L2_mix4_s0', 'w32_h2_L2_mix4_s1', 'w32_h2_L2_mix4_s2', 'w32_h2_L2_mix4_s4'] + \
          [f'w32_h2_L2_mix4_ws_s{s}' for s in range(5)] + [f'w32_h2_L2_mix4_s{s}_nb' for s in (5, 6, 7)]


def base_tag_of(noisy):
    init = json.load(open(os.path.join('runs', noisy, 'meta.json')))['init_from'].rstrip('/')
    nm = os.path.basename(init)
    if nm.startswith('base_mix4_s'):
        return f"w32_h2_L2_mix4_s{nm[len('base_mix4_s'):]}_nb"
    return nm


def main():
    z = np.load('results/e12_pairs.npz'); src, base, rg = z['src'], z['base'], z['range']; m = len(src); rows = np.arange(m)
    Wb = z['WB0'][rows, src]
    oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    WT = np.load('results/qweights_world.npz', allow_pickle=True)['test'].item()['WB']           # (4000, 3)
    nets = json.load(open('results/e15_nets.json'))
    R = {}
    for r in nets:
        for which in ('best', 'phase1'):
            p = os.path.join('results/e15', f"{r['tag']}__{which}.npz")
            if os.path.exists(p):
                R[(r['tag'], which)] = dict(np.load(p))
    info = {r['tag']: r for r in nets}
    for r in nets:
        g = R[(r['tag'], 'best')]['gap']
        r['gap'] = float(g); r['acquired'] = bool(r['depth'] == 2 and g < 0.10)

    def per(tag, which):
        q = R[(tag, which)]
        dw = q['ws1'][rows, src] - q['ws0'][rows, src]
        dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
        dg = q['g1'] - q['g0']
        out = dict(gap=float(q['gap']))
        for rr in ('large', 'middle'):
            s = rg == rr
            out[rr] = dict(mean_dw=float(dw[s].mean()), rom=float(dw[s].sum() / Wb[s].sum()), rom_other=float(dwo[s].sum() / Wb[s].sum()),
                           spec=float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum()), g_ratio=float(dg[s].sum() / q['g0'][s].sum()))
        w = q['test_w'].ravel(); W_ = WT.ravel(); out['slope'] = float(np.cov(w, W_)[0, 1] / np.var(W_, ddof=1))
        return out, dw

    P, DW = {}, {}
    for (tag, which) in R:
        P[(tag, which)], DW[(tag, which)] = per(tag, which)

    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]

    def pooled(keys, rr):
        sel = rg == rr
        def stat(ii):
            jj = ii[sel[ii]]; return float(np.mean([DW[k][jj].mean() for k in keys]))
        bs = [stat(ii) for ii in boots]
        return dict(mean_dw=stat(rows), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    rom=float(np.mean([P[k][rr]['rom'] for k in keys])), n_pos=int(sum(P[k][rr]['rom'] > 0 for k in keys)), n=len(keys))

    out = dict(n_nets=len(nets), per_net={f'{t}__{w}': v for (t, w), v in P.items()}, tests={}, predictions={})
    t, pr = out['tests'], out['predictions']
    # P100: the 12 clean bases
    kb = [(b, 'best') for b in BASES12]
    t['bases_large'] = pooled(kb, 'large'); t['bases_middle'] = pooled(kb, 'middle')
    pr['P100'] = dict(n_pos=t['bases_large']['n_pos'], ci=t['bases_large']['ci'],
                      holds=bool(t['bases_large']['n_pos'] >= 10 and t['bases_large']['ci'][0] > 0))
    # P101: acquired nets
    acq = [r['tag'] for r in nets if r['acquired']]; non = [r['tag'] for r in nets if not r['acquired']]
    npos = sum(P[(a, 'best')]['large']['rom'] > 0 for a in acq)
    t['acquired_large'] = pooled([(a, 'best') for a in acq], 'large'); t['acquired_middle'] = pooled([(a, 'best') for a in acq], 'middle')
    pr['P101'] = dict(n_pos=int(npos), n=len(acq), frac=float(npos / len(acq)), holds=bool(npos >= 0.8 * len(acq)))
    # P102: inheritance of the A/B pattern
    inh = []
    for noisy in QM.ALL:
        bt = base_tag_of(noisy); assert bt in BASES12, (noisy, bt)
        e12 = json.load(open('results/e12.json'))['per_net'][noisy]
        inh.append(dict(noisy=noisy, group=QM.GROUP[noisy], base=bt, base_middle=P[(bt, 'best')]['middle']['rom'],
                        base_large=P[(bt, 'best')]['large']['rom'], noisy_middle=e12['middle']['rom'], noisy_large=e12['large']['rom']))
    match = sum((x['base_middle'] < 0) == (x['group'] == 'A') for x in inh)
    corr = {}
    for rr in ('middle', 'large'):
        a = [x[f'base_{rr}'] for x in inh]; b = [x[f'noisy_{rr}'] for x in inh]
        corr[rr] = dict(pearson=float(pearsonr(a, b)[0]), spearman=float(spearmanr(a, b)[0]))
    pr['P102'] = dict(matches=int(match), n=len(inh), holds=bool(match >= 10), corr=corr, rows=inh)
    # P103: non-acquired nets need binding
    ok = sum(abs(P[(a, 'best')]['large']['spec']) <= 0.05 for a in non)
    pr['P103'] = dict(n_ok=int(ok), n=len(non), frac=float(ok / len(non)), holds=bool(ok >= 0.8 * len(non)))
    # P104: cooldown
    have = [a for a in acq if (a, 'phase1') in P]
    num = sum(P[(a, 'best')]['large']['rom'] for a in have); den = sum(P[(a, 'phase1')]['large']['rom'] for a in have)
    pr['P104'] = dict(n=len(have), sum_final=float(num), sum_phase1=float(den), ratio=float(num / den),
                      holds=bool(0.75 <= num / den <= 1.33),
                      n_final_lt_phase1=int(sum(P[(a, 'best')]['large']['rom'] < P[(a, 'phase1')]['large']['rom'] for a in have)))
    # P105: width and training length (means over acquired seeds)
    def cmean(cfg):
        v = [P[(r['tag'], 'best')]['large']['rom'] for r in nets if r['cfg'] == cfg and r['acquired'] and not r['tag'].endswith('_nb')]
        return float(np.mean(v)) if v else None, len(v)
    wcmp = [('w16_h1_L2', 'w32_h1_L2'), ('w16_h2_L2', 'w32_h2_L2'), ('w16_h2_L2_ds4', 'w32_h2_L2_ds4')]
    lcmp = [('w16_h1_L2_long', 'w16_h1_L2'), ('w32_h2_L2_long', 'w32_h2_L2'), ('w16_h2_L2_ds4_long', 'w16_h2_L2_ds4')]
    wres = [dict(a=a, b=b, a_mean=cmean(a), b_mean=cmean(b), a_larger=bool(cmean(a)[0] > cmean(b)[0])) for a, b in wcmp]
    lres = [dict(a=a, b=b, a_mean=cmean(a), b_mean=cmean(b), a_smaller=bool(cmean(a)[0] < cmean(b)[0])) for a, b in lcmp]
    pr['P105'] = dict(width=wres, width_holds=bool(sum(x['a_larger'] for x in wres) >= 2), length=lres,
                      length_holds=bool(sum(x['a_smaller'] for x in lres) >= 2))
    # descriptive: per config
    cfgs = sorted(set(r['cfg'] for r in nets))
    desc = {}
    for c in cfgs:
        for cls in ('acq', 'non'):
            tags = [r['tag'] for r in nets if r['cfg'] == c and (r['acquired'] == (cls == 'acq'))]
            if not tags: continue
            g = lambda f: [f(P[(a, 'best')]) for a in tags]
            desc[f'{c}|{cls}'] = dict(n=len(tags), gap=float(np.mean(g(lambda q: q['gap']))), slope=float(np.mean(g(lambda q: q['slope']))),
                                      rom_large=g(lambda q: q['large']['rom']), rom_middle=g(lambda q: q['middle']['rom']),
                                      spec_large=g(lambda q: q['large']['spec']), g_large=g(lambda q: q['large']['g_ratio']),
                                      g_middle=g(lambda q: q['middle']['g_ratio']))
    out['by_config'] = desc
    ga = [P[(a, 'best')]['gap'] for a in acq]; ra = [P[(a, 'best')]['large']['rom'] for a in acq]
    out['acq_gap_vs_rom_large_spearman'] = float(spearmanr(ga, ra)[0])
    out['acq_slope_vs_rom_large_spearman'] = float(spearmanr([P[(a, 'best')]['slope'] for a in acq], ra)[0])
    json.dump(out, open('results/e15.json', 'w'), indent=1)
    for k, v in pr.items():
        print(k, {a: b for a, b in v.items() if a not in ('rows', 'width', 'length')})
    # tables
    L = ['### E15.1 Predictions\n', '| # | test | outcome |', '|---|---|---|']
    L.append(f"| P100 | 12 clean bases: large-range rom > 0 in >= 10, pooled interval > 0 | {pr['P100']['n_pos']}/12; mean Δw {t['bases_large']['mean_dw']:+.4f} [{t['bases_large']['ci'][0]:+.4f}, {t['bases_large']['ci'][1]:+.4f}]; mean rom {t['bases_large']['rom']:+.3f}; {'holds' if pr['P100']['holds'] else 'fails'} |")
    L.append(f"| P101 | acquired nets: rom > 0 in >= 80% | {pr['P101']['n_pos']}/{pr['P101']['n']} ({pr['P101']['frac']:.0%}); {'holds' if pr['P101']['holds'] else 'fails'} |")
    L.append(f"| P102 | base's middle-range sign matches the noisy net's group in >= 10/12 | {pr['P102']['matches']}/12; {'holds' if pr['P102']['holds'] else 'fails'}; corr middle {corr['middle']['pearson']:+.2f} (Spearman {corr['middle']['spearman']:+.2f}), large {corr['large']['pearson']:+.2f} ({corr['large']['spearman']:+.2f}) |")
    L.append(f"| P103 | non-acquired: abs(spec) <= 0.05 in >= 80% | {pr['P103']['n_ok']}/{pr['P103']['n']} ({pr['P103']['frac']:.0%}); {'holds' if pr['P103']['holds'] else 'fails'} |")
    L.append(f"| P104 | Σ rom(final) / Σ rom(phase 1) in [0.75, 1.33] | {pr['P104']['ratio']:.2f} over {pr['P104']['n']} nets ({pr['P104']['n_final_lt_phase1']} smaller after cooldown); {'holds' if pr['P104']['holds'] else 'fails'} |")
    L.append(f"| P105 | width 16 > 32 in >= 2/3; long < regular in >= 2/3 | width: {sum(x['a_larger'] for x in wres)}/3; length: {sum(x['a_smaller'] for x in lres)}/3 |")
    L += ['\n### E15.2 By configuration (best checkpoints; mean over nets; rom = Σ Δw ÷ Σ W)\n',
          '| config | class | n | gap | slope on W | rom large | rom middle | spec large | Δg/g large | Δg/g middle |', '|---|---|---|---|---|---|---|---|---|---|']
    for k, v in desc.items():
        c, cls = k.split('|')
        L.append(f"| {c} | {cls} | {v['n']} | {v['gap']:.4f} | {v['slope']:.2f} | {np.mean(v['rom_large']):+.3f} [{min(v['rom_large']):+.3f}, {max(v['rom_large']):+.3f}] | "
                 f"{np.mean(v['rom_middle']):+.3f} [{min(v['rom_middle']):+.3f}, {max(v['rom_middle']):+.3f}] | {np.mean(v['spec_large']):+.3f} | {np.mean(v['g_large']):+.2f} | {np.mean(v['g_middle']):+.2f} |")
    L += ['\n### E15.3 The 12 bases and their noisy nets\n', '| noisy net | group | base | base rom large | noisy rom large | base rom middle | noisy rom middle |', '|---|---|---|---|---|---|---|']
    for x in inh:
        L.append(f"| {x['noisy']} | {x['group']} | {x['base']} | {x['base_large']:+.3f} | {x['noisy_large']:+.3f} | {x['base_middle']:+.3f} | {x['noisy_middle']:+.3f} |")
    open('results/e15_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
