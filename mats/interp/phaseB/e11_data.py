"""E11 data (PREREG-B amendment 18): fresh episodes for E11.1 and matched error-spread variants for E11.2.
usage: python e11_data.py -> results/e11_fresh.npz, results/e11_pairs.npz
"""
import os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qsys as QS
import qweights as QWT

KAPPAS = [0.25, 0.45, 0.65, 0.85]


def fresh():
    p = 'results/e11_fresh.npz'
    if os.path.exists(p): return
    rng = np.random.default_rng(20_261_029); d = W.sample_np(8000, rng); X, v = W.tokens_np(d, rng, 16)
    T = QS.tt(d); mB = QWT.bayes_mean(d); WB = QWT.bayes_weights({k: a.copy() for k, a in d.items()})
    np.savez(p, X=X, v=v, mB=mB, WB=WB, **{f'd_{k}': a for k, a in d.items()}, **{f'T_{k}': a for k, a in T.items()})
    print('fresh episodes saved', X.shape)


def pairs():
    p = 'results/e11_pairs.npz'
    if os.path.exists(p): return
    rng = np.random.default_rng(20_261_030); n = 1500; d = W.sample_np(n, rng)
    semb = rng.standard_normal((n, W.S, 16)) / 4; qemb = rng.standard_normal((n, W.Q, 16)) / 4
    res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; SS = (res ** 2).sum(2)                         # (n, 3)
    signs = rng.choice([-1.0, 1.0], size=(n, 3, W.QL)); pos = rng.integers(0, W.QL, size=(n, 3))
    blocks = {}
    for ki, kap in enumerate(KAPPAS):
        o_all, y_all, base, src = [], [], [], []
        for s in range(3):
            big = np.sqrt(kap * SS[:, s]); small = np.sqrt((1 - kap) / 3 * SS[:, s])
            r = np.repeat(small[:, None], W.QL, 1); r[np.arange(n), pos[:, s]] = big; r = r * signs[:, s]
            o = d['o'].copy(); o[:, s, :W.QL] = d['y'][:, :W.QL] + r
            o_all.append(o); y_all.append(d['y']); base.append(np.arange(n)); src.append(np.full(n, s))
        o_all, y_all = np.concatenate(o_all), np.concatenate(y_all); base, src = np.concatenate(base), np.concatenate(src)
        dd = dict(o=o_all, y=y_all, v=np.tile(d['v'], (3, 1)), vidx=np.tile(d['vidx'], (3, 1)))
        X, valid = W.assemble(np.tile(semb, (3, 1, 1)), np.tile(qemb, (3, 1, 1)), o_all, y_all)
        T = QS.tt(dd); WB = QWT.bayes_weights({k: a.copy() for k, a in dd.items()})
        blocks[ki] = dict(X=X, valid=valid, base=base, src=src, mB=T['mean_bayes'], vB=T['var_bayes'], WB=WB, T=T, d=dd)
    # Bayes invariance check across kappa
    dm = max(np.abs(blocks[k]['mB'] - blocks[0]['mB']).max() for k in blocks); dv = max(np.abs(blocks[k]['vB'] - blocks[0]['vB']).max() for k in blocks)
    dw = max(np.abs(blocks[k]['WB'] - blocks[0]['WB']).max() for k in blocks)
    ss_chk = max(np.abs(((blocks[k]['d']['o'][:, :, :W.QL] - blocks[k]['d']['y'][:, None, :W.QL]) ** 2).sum(2) - np.tile(SS, (3, 1))).max() for k in blocks)
    print('Bayes invariance across kappa: max |d mean| %.2e, |d var| %.2e, |d W| %.2e; SS check %.2e' % (dm, dv, dw, ss_chk))
    assert dm < 1e-8 and dv < 1e-8 and dw < 1e-6, 'Bayes not invariant'
    out = dict(kappas=np.array(KAPPAS), SS=SS, pos=pos)
    for k, b in blocks.items():
        out[f'X{k}'] = b['X']; out[f'valid{k}'] = b['valid']; out[f'WB{k}'] = b['WB']
        for kk, a in b['T'].items(): out[f'T{k}_{kk}'] = a
        for kk, a in b['d'].items(): out[f'd{k}_{kk}'] = a
    out['base'] = blocks[0]['base']; out['src'] = blocks[0]['src']; out['mB'] = blocks[0]['mB']
    np.savez(p, **out); print('pairs saved', blocks[0]['X'].shape)


if __name__ == '__main__':
    fresh(); pairs()
