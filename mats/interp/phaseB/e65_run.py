"""E65 (PREREG-B amendment 73; CPU, existing nets): E63's block-1 summary-head measures across sizes and in the sign-free format.
Cohorts (best-validation model.pt; E27's first 2,000 bases; each net in its own format; acquired = registered gap < 0.10):
  w64ctrl, w128ctrl, d3ctrl   E38's ctrl nets at width 64, width 128 and depth 3 (seeds 0-9);
  w32mag                      E37's mag nets (width 32, depth 2, seeds 0-11);
  w64mag, w128mag, d3mag      E38's mag nets.
Per net (e63_run.py's functions, unchanged): M1 (P_h, self, osum, olab, C_h, C_max, consensus, n_pool) and H3 for the pooling heads
(L_own, dlogit at |r| = 3, the coefficient of the o column's error term, the per-source normaliser's CV). In the mag format the o
column carries |r| (an even function of the signed error), so the linear coefficient is reported as K_lin and dlogit uses A / n at
|r| = +3 only (the token cannot carry -3); in ctrl both signs are averaged, as in E63.
usage: python e65_run.py COHORT -> results/e65/<cohort>.json
"""
import glob, json, math, os, sys
import numpy as np

import e63_run as E63

S, Q, QL = E63.S, E63.Q, E63.QL
NB = 2000
COH = dict(w64ctrl=('runs/e38/e38_w64ctrl_s*', 'results/e38'), w128ctrl=('runs/e38/e38_w128ctrl_s*', 'results/e38'),
           d3ctrl=('runs/e38/e38_d3ctrl_s*', 'results/e38'), w32mag=('runs/e37/e37_mag_s*', 'results/e37'),
           w64mag=('runs/e38/e38_w64mag_s*', 'results/e38'), w128mag=('runs/e38/e38_w128mag_s*', 'results/e38'),
           d3mag=('runs/e38/e38_d3mag_s*', 'results/e38'),
           plumb=('runs/e43/e43_newinit_s[0124]', 'results/e43'),                                  # plumbing only: explored nets
           plumbmag=('/tmp/e65_plumb/runs/e37_mag_s*', '/tmp/e65_plumb/results'))                  # plumbing only: step-0 mag weights


def plus_ratio(model, X):
    """A / n at r = +3 (no sign averaging), per labelled token (B, 12)."""
    W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
    lab = [j for s in range(S) for j in E63.LABS[s]]
    X0 = X[:, lab, :].astype(np.float64).copy(); X0[..., 32] = X0[..., 33]
    a = X0 @ W.T + b; a = a - a.mean(-1, keepdims=True)
    u = W[:, 32] - W[:, 32].mean(); Aj = np.linalg.norm(a, axis=-1); U = np.linalg.norm(u); rho = (a @ u) / (Aj * U)
    r = 3.0
    return Aj / np.sqrt(Aj ** 2 + 2 * r * Aj * U * rho + r ** 2 * U ** 2)


def main():
    name = sys.argv[1]; rpat, rdir = COH[name]
    os.makedirs('results/e65', exist_ok=True)
    z = np.load('results/e27_data.npz'); sl = slice(0, NB)
    res = {}
    runs = sorted(d for d in glob.glob(rpat) if os.path.isdir(d) and '_mag60_' not in d)
    for run in runs:
        tag = os.path.basename(run); evf = os.path.join(rdir, tag + '.npz')
        if not E63.acquired(evf):
            continue
        model, arm, meta = E63.load(run, None)
        evz = np.load(evf, allow_pickle=True)
        o = dict(arm=arm, width=meta['width'], depth=meta['depth'], gap=float(evz['gap']), rom_large=float(evz['rom_large']),
                 rom_middle=float(evz['rom_middle']), type='A' if float(evz['rom_middle']) < 0 else 'B')
        X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
        A, qh, blk = E63.block1_parts(model, X)
        H = A.shape[1]
        P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1)
        slf = np.stack([A[:, :, E63.SUMM[s], E63.SUMM[s]] for s in range(S)], 1)
        osum = np.stack([A[:, :, E63.SUMM[s], [E63.SUMM[t] for t in range(S) if t != s]].sum(-1) for s in range(S)], 1)
        olab = np.stack([A[:, :, E63.SUMM[s], [j for t in range(S) if t != s for j in E63.LABS[t]]].sum(-1) for s in range(S)], 1)
        o['P_h'] = [float(P[:, :, h].mean()) for h in range(H)]; o['self_h'] = [float(slf[:, :, h].mean()) for h in range(H)]
        o['osum_h'] = [float(osum[:, :, h].mean()) for h in range(H)]; o['olab_h'] = [float(olab[:, :, h].mean()) for h in range(H)]
        o['C_h'] = [o['self_h'][h] + o['osum_h'][h] for h in range(H)]; o['C_max'] = float(max(o['C_h'])); o['consensus'] = bool(o['C_max'] >= 0.5)
        pool = [h for h in range(H) if o['P_h'][h] >= 0.5]; o['n_pool'] = len(pool); o['pool_heads'] = pool
        if pool:
            L, ratio, odd = E63.h3_logits(model, X, qh, blk)
            if arm == 'mag':
                ratio = plus_ratio(model, X)
            own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
            Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); rat = np.stack([ratio[:, own_i[s]] for s in range(S)], 1)
            K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
            o['L_own_h'] = [float(Lown[:, :, h].mean()) for h in range(H)]
            o['dlogit_r3_h'] = [float((Lown[:, :, h] * (rat - 1.0)).mean()) for h in range(H)]
            o['K_mean_h'] = [float(K[:, :, h].mean()) for h in range(H)]; o['K_rms_h'] = [float(np.sqrt((K[:, :, h] ** 2).mean())) for h in range(H)]
        res[tag] = o
        print(name, tag, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in o.items()}), flush=True)
    json.dump(res, open(('/tmp/e65_plumb/%s.json' if name.startswith('plumb') else 'results/e65/%s.json') % name, 'w'), indent=1)


if __name__ == '__main__':
    main()
