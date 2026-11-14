"""E14.3 data and frozen predictions (PREREG-B amendment 21 and addendum 1). Fresh pairs with the same SS, (m, kappa) family;
predictions from the fitted forms (results/e14_models.pkl) saved before any net sees the pairs.
usage: python e14_causal_prep.py -> results/e14_pairs.npz, results/e14_predictions.npz, results/e14_invariance.txt
"""
import os, pickle
import numpy as np
import nnoise  # noqa: F401
import world as W
import qsys as QS
import qweights as QWT
import qmech as QM
import e14_fit as F

MS = [1, 2, 3]; KAPS = [0.4, 0.6, 0.8, 0.95]


def pattern(m, kap, ss, sign, perm):
    big = np.sqrt(kap * ss / m); small = np.sqrt((1 - kap) * ss / (4 - m))
    r = np.where(np.arange(4) < m, big, small)[perm] * sign; return r


def main():
    p = 'results/e14_pairs.npz'
    assert not os.path.exists(p)
    rng = np.random.default_rng(20_261_032); n = 2000; d = W.sample_np(n, rng)
    semb = rng.standard_normal((n, W.S, 16)) / 4; qemb = rng.standard_normal((n, W.Q, 16)) / 4
    res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; SS = (res ** 2).sum(2)
    b_idx = np.repeat(np.arange(n), 3); s_idx = np.tile(np.arange(3), n); m_rows = len(b_idx)
    signs = rng.choice([-1.0, 1.0], size=(m_rows, 4)); perms = np.array([rng.permutation(4) for _ in range(m_rows)])
    ma = rng.choice(MS, m_rows); ka = rng.choice(KAPS, m_rows); mb = rng.choice(MS, m_rows); kb = rng.choice(KAPS, m_rows)
    out = dict(base=b_idx, src=s_idx, SS=SS[b_idx, s_idx], ma=ma, ka=ka, mb=mb, kb=kb)
    blocks = []
    for v, (mm, kk) in enumerate(((ma, ka), (mb, kb))):
        o = d['o'][b_idx].copy(); y = d['y'][b_idx].copy()
        R = np.stack([pattern(mm[i], kk[i], SS[b_idx[i], s_idx[i]], signs[i], perms[i]) for i in range(m_rows)])
        o[np.arange(m_rows), s_idx, :W.QL] = y[:, :W.QL] + R
        dd = dict(o=o, y=y, v=d['v'][b_idx].copy(), vidx=d['vidx'][b_idx].copy()); X, valid = W.assemble(semb[b_idx], qemb[b_idx], o, y)
        T = QS.tt(dd); WB = QWT.bayes_weights({k: a.copy() for k, a in dd.items()})
        blocks.append(dict(T=T, WB=WB, R=R)); out[f'X{v}'] = X; out[f'valid{v}'] = valid; out[f'WB{v}'] = WB; out[f'R{v}'] = R
        for k, a in T.items(): out[f'T{v}_{k}'] = a
        for k, a in dd.items(): out[f'd{v}_{k}'] = a
    dm = float(np.abs(blocks[1]['T']['mean_bayes'] - blocks[0]['T']['mean_bayes']).max()); dv = float(np.abs(blocks[1]['T']['var_bayes'] - blocks[0]['T']['var_bayes']).max())
    dw = float(np.abs(blocks[1]['WB'] - blocks[0]['WB']).max())
    msg = 'rows %d; max |change| between versions: Bayes mean %.2e, variance %.2e, weights %.2e' % (m_rows, dm, dv, dw)
    print(msg); open('results/e14_invariance.txt', 'w').write(msg + '\n')
    assert dm < 1e-8 and dv < 1e-8 and dw < 1e-8
    out['mB'] = blocks[0]['T']['mean_bayes']; np.savez(p, **out)
    models = pickle.load(open('results/e14_models.pkl', 'rb')); pred = {}
    for nm in QM.ALL:
        for form in ('SS', 'SA', 'COUNT', 'SYM'):
            pred[f'{nm}|{form}'] = F.pred_form(form, models[nm][form], blocks[1]['R']) - F.pred_form(form, models[nm][form], blocks[0]['R'])
    np.savez('results/e14_predictions.npz', **pred); print('predictions saved', len(pred))


if __name__ == '__main__':
    main()
