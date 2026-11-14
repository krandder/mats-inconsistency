"""E64b post hoc (NOT registered; seventh agent, after E64b's registered summary): the cap scale of the split nets' content normaliser.
In split, block 1's LayerNorm normaliser of the error feature is the content part's norm alone: n_j = |P(W_c X_c,j + b)| with the
content columns (o, y, missing, query flag) and the bias; at r_j = 0 that is A_j, and U = |P w_o|, so c = median A_j / U over the
labelled tokens of E27's first 500 bases (as lngeom.py, with the code columns excluded). The constrained-Bayes departure of the capped
even feature at that c is read from amendment 63's table (results/e55_cbtable.json, linear interpolation), as E55 did for uniform pools.
usage: python e64b_post.py -> results/e64b_post.json (printed)
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64b as PB  # noqa: E402

P2.build = lambda meta: PB.build_any(meta, P2, P2.fin_of(meta['ds']))
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402

E30.F = F
LAB = [5 * s + q for s in range(3) for q in range(4)]


def main():
    z = np.load('results/e27_data.npz')
    X, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'ctrl')
    tab = json.load(open('results/e55_cbtable.json')); g = np.array(tab['grid'])
    rl = np.array([tab['rom']['%.2f' % x]['large'] for x in g]); rm = np.array([tab['rom']['%.2f' % x]['middle'] for x in g])
    out = {}
    for run in sorted(glob.glob('runs/e64b/e64_split*_s*')):
        if not os.path.isdir(run):
            continue
        tag = os.path.basename(run); m, arm, meta = E30.load(run)
        W = m.inp.weight.detach().numpy().astype(np.float64); b = m.inp.bias.detach().numpy().astype(np.float64)
        T = X[:, LAB, :].reshape(-1, X.shape[2]).astype(np.float64)
        T0 = T.copy(); T0[:, 32] = T0[:, 33]; T0[:, :32] = 0                         # content part at r = 0 (codes excluded)
        a = T0 @ W.T + b; a = a - a.mean(1, keepdims=True); A = np.linalg.norm(a, axis=1)
        u = W[:, 32] - W[:, 32].mean(); U = np.linalg.norm(u); c = float(np.median(A / U))
        ev = np.load('results/e64b/%s.npz' % tag)
        out[tag] = dict(c=c, A_cv=float(A.std() / A.mean()), rom_large=float(ev['rom_large']), rom_middle=float(ev['rom_middle']),
                        cb_large=float(np.interp(c, g, rl)), cb_middle=float(np.interp(c, g, rm)), gap=float(ev['gap']))
        print(tag, {k: round(v, 4) for k, v in out[tag].items()}, flush=True)
    for arm in ('split', 'splitnc'):
        V = [v for k, v in out.items() if k.startswith('e64_%s_s' % arm)]
        print(arm, 'median c %.2f; mean rom_large %.3f against CB %.3f; rom_middle %.3f against %.3f' % (
            np.median([v['c'] for v in V]), np.mean([v['rom_large'] for v in V]), np.mean([v['cb_large'] for v in V]),
            np.mean([v['rom_middle'] for v in V]), np.mean([v['cb_middle'] for v in V])))
    json.dump(out, open('results/e64b_post.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
