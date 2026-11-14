"""E21 (PREREG-B amendment 28): width scaling of the departure in clean 2-layer nets of the base recipe (2 heads, mixture head).
Measure: python e21_summ.py measure -> results/e21/w{W}_s{S}.npz for every runs/e21_w{W}_s{S} (pfn2 models; E15's measures)
Analyse: python e21_summ.py -> results/e21.json, results/e21_tables.md (width 32 = E15's w32_h2_L2_mix4 nets, recomputed from
results/e15/*__best.npz)
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

NBOOT = 1000
W32 = ['w32_h2_L2_mix4_s0', 'w32_h2_L2_mix4_s1', 'w32_h2_L2_mix4_s2', 'w32_h2_L2_mix4_s3', 'w32_h2_L2_mix4_s4',
       'w32_h2_L2_mix4_s5_nb', 'w32_h2_L2_mix4_s6_nb', 'w32_h2_L2_mix4_s7_nb']


def measure():
    import torch
    import e15_run as R
    sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
    import pfn2 as P2
    torch.set_num_threads(1)
    z = np.load('results/e12_pairs.npz'); tz = np.load('/home/agent/workspaces/mats/work/tiny-pfn/runs/testset.npz')
    T = {k: tz[k] for k in ('X', 'valid', 'ystar', 'nll_bayes')}
    os.makedirs('results/e21', exist_ok=True)
    for d in sorted(glob.glob('runs/e21_w*_s*')):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)[4:]; p = f'results/e21/{tag}.npz'
        if os.path.exists(p) or not os.path.exists(os.path.join(d, 'model.pt')):
            continue
        meta = json.load(open(os.path.join(d, 'meta.json')))
        m = P2.build(meta); m.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
        out = dict(gap=R.gap(m, T))
        for k in (0, 1):
            out[f'ws{k}'], out[f'mean{k}'] = R.eff_w(m, z[f'X{k}'], z[f'valid{k}'], 32)
        out['test_w'], out['test_m'] = R.eff_w(m, tz['X'][:4000], tz['valid'][:4000], 32)
        np.savez(p, **out); print(tag, 'gap %.4f' % out['gap'], flush=True)


def stats(q, z):
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); Wb = z['WB0'][rows, src]; oth = np.stack([(src + 1) % 3, (src + 2) % 3], 1)
    dw = q['ws1'][rows, src] - q['ws0'][rows, src]; dwo = (q['ws1'][rows[:, None], oth] - q['ws0'][rows[:, None], oth]).mean(1)
    d = dict(gap=float(q['gap']))
    for r in ('large', 'middle'):
        s = rg == r
        d[r] = dict(rom=float(dw[s].sum() / Wb[s].sum()), spec=float((dw[s].sum() - dwo[s].sum()) / Wb[s].sum()))
    return d, dw


def analyse():
    z = np.load('results/e12_pairs.npz'); base, rg = z['base'], z['range']
    nets = {}
    for t in W32:
        nets[(32, t)] = stats(np.load(f'results/e15/{t}__best.npz'), z)
    for f in sorted(glob.glob('results/e21/w*_s*.npz')):
        tag = os.path.basename(f)[:-4]; W = int(tag[1:].split('_')[0])
        nets[(W, tag)] = stats(np.load(f), z)
    ub = np.arange(3000); idx = {b: np.where(base == b)[0] for b in ub}
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[b] for b in rng.choice(ub, len(ub))]) for _ in range(NBOOT)]
    sel = rg == 'large'
    out = dict(per_net={f'{w}|{t}': v[0] for (w, t), v in nets.items()}, widths={}, predictions={})
    for W in sorted(set(w for w, _ in nets)):
        keys = [k for k in nets if k[0] == W and nets[k][0]['gap'] < 0.10]
        if not keys:
            continue
        def st(ii):
            jj = ii[sel[ii]]; return float(np.mean([nets[k][1][jj].mean() for k in keys]))
        bs = [st(ii) for ii in boots]
        rl = [nets[k][0]['large']['rom'] for k in keys]; rm = [nets[k][0]['middle']['rom'] for k in keys]
        out['widths'][W] = dict(n_total=sum(1 for k in nets if k[0] == W), n_acquired=len(keys), gap=float(np.mean([nets[k][0]['gap'] for k in keys])),
                                rom_large=float(np.mean(rl)), rom_large_min=float(min(rl)), rom_large_max=float(max(rl)),
                                n_pos_large=int(sum(x > 0 for x in rl)), rom_middle=float(np.mean(rm)), n_middle_neg=int(sum(x < 0 for x in rm)),
                                spec_large=float(np.mean([nets[k][0]['large']['spec'] for k in keys])),
                                mean_dw_large=st(np.arange(len(rg))), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])
    Wd = out['widths']; pr = out['predictions']
    if 128 in Wd and 256 in Wd:
        pr['P125'] = dict(holds=bool(all(Wd[W]['n_pos_large'] >= 0.8 * Wd[W]['n_acquired'] and Wd[W]['rom_large'] >= 0.05 for W in (128, 256))),
                          **{f'w{W}': dict(mean=Wd[W]['rom_large'], pos=f"{Wd[W]['n_pos_large']}/{Wd[W]['n_acquired']}") for W in (128, 256)})
        xs, ys = [], []
        for (w, t), v in nets.items():
            if v[0]['gap'] < 0.10:
                xs.append(w); ys.append(v[0]['large']['rom'])
        rho = float(spearmanr(xs, ys)[0])
        pr['P126'] = dict(spearman=rho, w32=Wd[32]['rom_large'], w256=Wd[256]['rom_large'], holds=bool(rho < 0 and Wd[256]['rom_large'] < Wd[32]['rom_large']))
        neg = Wd[128]['n_middle_neg'] + Wd[256]['n_middle_neg']; tot = Wd[128]['n_acquired'] + Wd[256]['n_acquired']
        pr['P127'] = dict(n_middle_neg=neg, n_middle_pos=tot - neg, holds=bool(neg >= 1 and tot - neg >= 1))
    json.dump(out, open('results/e21.json', 'w'), indent=1)
    for k, v in pr.items():
        print(k, v)
    T = ['| width | acquired / total | test gap | rom large: mean [min, max] | nets > 0 | mean Δw large [95% CI] | rom middle | middle < 0 (A-type) | spec large |',
         '|---|---|---|---|---|---|---|---|---|']
    for W, v in sorted(Wd.items()):
        T.append(f"| {W} | {v['n_acquired']} / {v['n_total']} | {v['gap']:.4f} | {v['rom_large']:+.3f} [{v['rom_large_min']:+.3f}, {v['rom_large_max']:+.3f}] | "
                 f"{v['n_pos_large']}/{v['n_acquired']} | {v['mean_dw_large']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['rom_middle']:+.3f} | "
                 f"{v['n_middle_neg']}/{v['n_acquired']} | {v['spec_large']:+.3f} |")
    T += ['', '| width | net | test gap | rom large | rom middle |', '|---|---|---|---|---|']
    for (w, t), v in sorted(nets.items()):
        T.append(f"| {w} | {t} | {v[0]['gap']:.4f} | {v[0]['large']['rom']:+.3f} | {v[0]['middle']['rom']:+.3f} |")
    open('results/e21_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    measure() if len(sys.argv) > 1 and sys.argv[1] == 'measure' else analyse()
