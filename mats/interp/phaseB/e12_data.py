"""E12 data (PREREG-B amendment 19): fresh matched pairs (kappa 0.85 vs 0.25) for every (episode, source) row in the frozen
large (SS >= 3.182394) or middle (0.679992 <= SS < 3.182394) range. usage: python e12_data.py -> results/e12_pairs.npz, results/e12_invariance.txt
"""
import os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qsys as QS
import qweights as QWT

LO, HI = 0.679992, 3.182394
KAPPAS = [0.25, 0.85]


def main():
    p = 'results/e12_pairs.npz'
    assert not os.path.exists(p)
    rng = np.random.default_rng(20_261_031); n = 3000; d = W.sample_np(n, rng)
    semb = rng.standard_normal((n, W.S, 16)) / 4; qemb = rng.standard_normal((n, W.Q, 16)) / 4
    signs = rng.choice([-1.0, 1.0], size=(n, 3, W.QL)); pos = rng.integers(0, W.QL, size=(n, 3))
    res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; SS = (res ** 2).sum(2)
    sel = np.argwhere(SS >= LO)                                                          # rows (episode, source) in middle or large range
    b_idx, s_idx = sel[:, 0], sel[:, 1]; rng_lab = np.where(SS[b_idx, s_idx] >= HI, 'large', 'middle')
    m = len(b_idx); out = dict(base=b_idx, src=s_idx, range=rng_lab, SS=SS[b_idx, s_idx], kappas=np.array(KAPPAS))
    blocks = []
    for ki, kap in enumerate(KAPPAS):
        o = d['o'][b_idx].copy(); y = d['y'][b_idx].copy(); ss = SS[b_idx, s_idx]
        big = np.sqrt(kap * ss); small = np.sqrt((1 - kap) / 3 * ss)
        r = np.repeat(small[:, None], W.QL, 1); r[np.arange(m), pos[b_idx, s_idx]] = big; r = r * signs[b_idx, s_idx]
        o[np.arange(m), s_idx, :W.QL] = y[:, :W.QL] + r
        dd = dict(o=o, y=y, v=d['v'][b_idx].copy(), vidx=d['vidx'][b_idx].copy())
        X, valid = W.assemble(semb[b_idx], qemb[b_idx], o, y)
        T = QS.tt(dd); WB = QWT.bayes_weights({k: a.copy() for k, a in dd.items()})
        blocks.append(dict(X=X, valid=valid, T=T, WB=WB, d=dd))
        out[f'X{ki}'] = X; out[f'valid{ki}'] = valid; out[f'WB{ki}'] = WB
        for k, a in T.items(): out[f'T{ki}_{k}'] = a
        for k, a in dd.items(): out[f'd{ki}_{k}'] = a
    dm = float(np.abs(blocks[1]['T']['mean_bayes'] - blocks[0]['T']['mean_bayes']).max()); dv = float(np.abs(blocks[1]['T']['var_bayes'] - blocks[0]['T']['var_bayes']).max())
    dw = float(np.abs(blocks[1]['WB'] - blocks[0]['WB']).max())
    mask = np.ones(blocks[0]['d']['o'].shape, bool); mask[np.arange(m), s_idx, :W.QL] = False
    doth = float(np.abs(blocks[1]['d']['o'][mask] - blocks[0]['d']['o'][mask]).max())
    ssd = float(np.abs(((blocks[1]['d']['o'][:, :, :W.QL] - blocks[1]['d']['y'][:, None, :W.QL]) ** 2).sum(2) - ((blocks[0]['d']['o'][:, :, :W.QL] - blocks[0]['d']['y'][:, None, :W.QL]) ** 2).sum(2)).max())
    msg = ('rows %d (large %d, middle %d); max |change| across kappa: Bayes mean %.2e, variance %.2e, weights %.2e; every source SS %.2e; other readings %.2e'
           % (m, int((rng_lab == 'large').sum()), int((rng_lab == 'middle').sum()), dm, dv, dw, ssd, doth))
    print(msg); open('results/e12_invariance.txt', 'w').write(msg + '\n')
    assert dm < 1e-8 and dv < 1e-8 and dw < 1e-8, 'Bayes not invariant'
    out['mB'] = blocks[0]['T']['mean_bayes']
    np.savez(p, **out)


if __name__ == '__main__':
    main()
