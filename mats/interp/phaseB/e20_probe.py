"""E20 post hoc (written after E20's registered results): single-error response curves of the minimal models.
For 2,000 base episodes (default_rng(20_261_108)) with three small errors (|r| = 0.3, random signs) and labels/identities as in
training, vary the fourth error r from 0 to 4 (positive and negative) and record, per model:
  P(r)      the predicted posterior precision sum_k p_k / v_k (averaged over base episodes and both signs);
  alpha(r)  the attention weight on the varied token (ln and pre models);
and Bayes's P(r). Saturation shows as P(r) flattening at large r while Bayes keeps falling.
usage: python e20_probe.py -> results/e20_probe.json, results/e20_probe.png
"""
import glob, json, math, os
import numpy as np
import torch
import minimal as M

RS = np.linspace(0, 4, 41)


def main():
    rng = np.random.default_rng(20_261_108); n = 2000
    _, y, _, sid, qid = M.sample(n, rng); base = 0.3 * rng.choice([-1.0, 1.0], (n, 3))
    out = {'r': RS.tolist(), 'bayes': [], 'models': {}}
    for r in RS:
        R = np.concatenate([np.full((n, 1), r), base], 1)
        out['bayes'].append(float((M.bayes_post(R) / M.VLEV[None]).sum(1).mean()))
    for f in sorted(glob.glob('results/e20/*_sig0.0.json')):
        meta = json.load(open(f))
        if meta['kind'] == 'ss':
            continue
        net = M.Net(meta['kind']); net.load_state_dict(torch.load(f[:-5] + '.pt')); net.eval()
        P, A = [], []
        with torch.no_grad():
            for r in RS:
                pr, al = 0, 0
                for sgn in (1.0, -1.0):
                    R = np.concatenate([np.full((n, 1), sgn * r), base], 1)
                    X = M.tokens(y, R, sid, qid); lg = torch.as_tensor(np.log((R ** 2).sum(1) + 1e-12), dtype=torch.float32)
                    pr = pr + (torch.softmax(net(X, lg), -1).numpy() / M.VLEV[None]).sum(1).mean() / 2
                    if meta['kind'] != 'lnmean':
                        h = net.tok(X); a = torch.softmax(net.k(h) @ net.q / math.sqrt(h.shape[-1]), 1)
                        al = al + float(a[:, 0].mean()) / 2
                P.append(float(pr)); A.append(al)
        out['models'][os.path.basename(f)[:-5]] = dict(kind=meta['kind'], P=P, alpha=A if meta['kind'] != 'lnmean' else None)
    json.dump(out, open('results/e20_probe.json', 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(RS, out['bayes'], 'k-', lw=2, label='Bayes')
    col = {'ln': 'C0', 'lnmean': 'C2', 'pre': 'C1'}
    for k, v in out['models'].items():
        ax[0].plot(RS, v['P'], color=col[v['kind']], alpha=0.7, label=v['kind'] if k.endswith('s0_sig0.0') else None)
        if v['alpha']:
            ax[1].plot(RS, v['alpha'], color=col[v['kind']], alpha=0.7, label=v['kind'] if k.endswith('s0_sig0.0') else None)
    ax[0].set_xlabel('size of one error (other three at 0.3)'); ax[0].set_ylabel('predicted posterior precision'); ax[0].legend()
    ax[1].set_xlabel('size of one error'); ax[1].set_ylabel('attention weight on that token'); ax[1].axhline(0.25, color='grey', lw=0.5); ax[1].legend()
    fig.tight_layout(); fig.savefig('results/e20_probe.png', dpi=110)
    for k, v in out['models'].items():
        print(k, 'P at r=0,1,2,3,4:', [round(v['P'][i], 2) for i in (0, 10, 20, 30, 40)], 'alpha:', None if not v['alpha'] else [round(v['alpha'][i], 3) for i in (0, 10, 20, 30, 40)])
    print('bayes', [round(out['bayes'][i], 2) for i in (0, 10, 20, 30, 40)])


if __name__ == '__main__':
    main()
