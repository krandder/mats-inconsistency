"""E17 analysis (PREREG-B amendment 24): depth against layer-2 attention capacity, and the nets closest to Bayes.
usage: python e17_summ.py -> results/e17.json, results/e17_tables.md
"""
import json, os
import numpy as np

NBOOT = 1000
ARMS = ['base', 'h3', 'h4', 'w64', 'L3']


def main():
    z = np.load('results/e12_pairs.npz'); src, base, rg = z['src'], z['base'], z['range']; m = len(src); rows = np.arange(m)
    Wb = z['WB0'][rows, src]; oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    WT = np.load('results/qweights_world.npz', allow_pickle=True)['test'].item()['WB']
    nets = json.load(open('results/e17_nets.json'))
    P, DW = {}, {}
    for r in nets:
        q = np.load(os.path.join('results/e17', f"{r['tag']}.npz"))
        dw = q['ws1'][rows, src] - q['ws0'][rows, src]; dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
        d = dict(gap=float(q['gap']), acquired=bool(q['gap'] < 0.10), arm=r['arm'].split('_')[0] if r['kind'] == 'clean' else r['tag'],
                 head=r['arm'].split('_')[1] if r['kind'] == 'clean' else 'mix', kind=r['kind'])
        for rr in ('large', 'middle'):
            s = rg == rr
            d[rr] = dict(rom=float(dw[s].sum() / Wb[s].sum()), spec=float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum()),
                         tot=float(q['ws1'][s].sum() / q['ws0'][s].sum() - 1), mean_dw=float(dw[s].mean()))
        w = q['test_w'].ravel(); d['slope'] = float(np.cov(w, WT.ravel())[0, 1] / np.var(WT.ravel(), ddof=1))
        P[r['tag']] = d; DW[r['tag']] = dw
    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]

    def pooled(tags, rr='large'):
        sel = rg == rr
        def stat(ii):
            jj = ii[sel[ii]]; return float(np.mean([DW[t][jj].mean() for t in tags]))
        bs = [stat(ii) for ii in boots]
        return dict(mean_dw=stat(rows), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    mean_rom=float(np.mean([P[t][rr]['rom'] for t in tags])), n=len(tags), n_pos=int(sum(P[t][rr]['rom'] > 0 for t in tags)))
    out = dict(per_net=P, arms={}, predictions={})
    for a in ARMS:
        tags = [t for t, d in P.items() if d['kind'] == 'clean' and d['arm'] == a and d['acquired']]
        out['arms'][a] = dict(tags=tags, n_total=sum(1 for d in P.values() if d['kind'] == 'clean' and d['arm'] == a),
                              large=pooled(tags, 'large'), middle=pooled(tags, 'middle'),
                              gap=float(np.mean([P[t]['gap'] for t in tags])), slope=float(np.mean([P[t]['slope'] for t in tags])),
                              spec_large=float(np.mean([P[t]['large']['spec'] for t in tags])),
                              n_middle_neg=int(sum(P[t]['middle']['rom'] < 0 for t in tags)))
    A = out['arms']; mb = A['base']['large']['mean_rom']
    out['predictions']['P112'] = dict(L3=A['L3']['large']['mean_rom'], base=mb, ratio=A['L3']['large']['mean_rom'] / mb,
                                      holds=bool(A['L3']['large']['mean_rom'] < 0.5 * mb))
    cap = A['h3']['tags'] + A['h4']['tags'] + A['w64']['tags']
    mc = float(np.mean([P[t]['large']['rom'] for t in cap])); npos = sum(P[t]['large']['rom'] > 0 for t in cap)
    out['predictions']['P113'] = dict(mean=mc, ratio=mc / mb, n_pos=int(npos), n=len(cap), holds=bool(mc >= 0.75 * mb and npos >= 0.8 * len(cap)))
    l3 = P['noisy_L3_s0.0']
    out['predictions']['P114'] = dict(rom_large=l3['large']['rom'], gap=l3['gap'], holds=bool(l3['large']['rom'] < 0.03))
    out['near_bayes'] = {t: dict(gap=P[t]['gap'], large=pooled([t], 'large'), middle=pooled([t], 'middle'), spec_large=P[t]['large']['spec'],
                                 slope=P[t]['slope']) for t in ('noisy_L3_s0.0', 'noisy_L3_s0.05', 'noisy_L2_s0.0')}
    json.dump(out, open('results/e17.json', 'w'), indent=1)
    for k, v in out['predictions'].items():
        print(k, v)
    L = ['| arm | acquired / total | gap | slope on W | rom large (mean over nets) | mean Δw large [95% CI] | nets > 0 | rom middle | middle < 0 | spec large |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for a in ARMS:
        v = A[a]
        L.append(f"| {a} | {v['large']['n']} / {v['n_total']} | {v['gap']:.4f} | {v['slope']:.2f} | {v['large']['mean_rom']:+.3f} | {v['large']['mean_dw']:+.4f} "
                 f"[{v['large']['ci'][0]:+.4f}, {v['large']['ci'][1]:+.4f}] | {v['large']['n_pos']}/{v['large']['n']} | {v['middle']['mean_rom']:+.3f} | "
                 f"{v['n_middle_neg']}/{v['large']['n']} | {v['spec_large']:+.3f} |")
    L += ['', '| near-Bayes net | test gap | slope | rom large | mean Δw large [95% CI] | rom middle | spec large |', '|---|---|---|---|---|---|---|']
    for t, v in out['near_bayes'].items():
        L.append(f"| {t} | {v['gap']:.4f} | {v['slope']:.2f} | {v['large']['mean_rom']:+.3f} | {v['large']['mean_dw']:+.4f} [{v['large']['ci'][0]:+.4f}, {v['large']['ci'][1]:+.4f}] | "
                 f"{v['middle']['mean_rom']:+.3f} | {v['spec_large']:+.3f} |")
    L += ['', '| net | gap | rom large | rom middle | spec large |', '|---|---|---|---|---|']
    for t, d in sorted(P.items()):
        L.append(f"| {t} | {d['gap']:.4f} | {d['large']['rom']:+.3f} | {d['middle']['rom']:+.3f} | {d['large']['spec']:+.3f} |")
    open('results/e17_tables.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
