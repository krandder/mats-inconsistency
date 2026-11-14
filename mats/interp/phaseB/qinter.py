"""E7.3 (PREREG-B amendment 14): does adding interactions explain more of the shared mean deviation?
Five-net seed-averaged mean deviation (65,536 fit episodes, K = 64) regressed on E6's named + token-level features plus
interactions: query reading x source-ID cosines, query reading x labelled residual sums, products of query readings,
query reading x its own labelled residuals, Bayes weight x query reading, query reading - Bayes mean.
Score: R^2 against the seed-averaged test deviation (8,000 test episodes) and the finite-ensemble ratio of E6.
usage: python qinter.py -> results/qinter.json
"""
import json, os
import numpy as np
import devpred as DP
import qsys as QS
import qshare as SH
import qshare2 as S2


def interactions(d, X, T):
    o = d['o'][:, :, 4]; n = len(o)
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    C = np.einsum('nsd,ntd->nst', sn, sn)
    res = d['o'][:, :, :4] - d['y'][:, None, :4]; ss = (res ** 2).sum(2)
    import qbayes as QB
    b = QB.bayes_features(d); wsrc = b[:, 20:23]; mb = b[:, 0]
    cols = [o[:, s:s + 1] * C[:, s, :] for s in range(3)]
    cols += [o * ss, o * wsrc, o - mb[:, None]]
    cols += [o[:, a:a + 1] * o[:, c:c + 1] for a in range(3) for c in range(a, 3)]
    cols += [o[:, s:s + 1] * res[:, s, :] for s in range(3)]
    return np.concatenate(cols, 1).astype(np.float32)


def main():
    dF, XF, vF, TF = S2.fit_world(); dT, XT, vT, TT = S2.test_world()
    avgF, _ = S2.load_fit_avg(); T = S2.load_test([nm for nm, _ in SH.MAIN])
    avgT = {k: np.mean([T[nm][k][0] for nm in T], 0) for k in SH.FEATS}
    zF, _ = QS.named(dF, XF, TF); zT, _ = QS.named(dT, XT, TT)
    tF, tT = S2.token_feats(dF, XF), S2.token_feats(dT, XT)
    iF, iT = interactions(dF, XF, TF), interactions(dT, XT, TT)
    k = 'mean'; sv = S2.shared_var(T, k); out = dict(rows=[])
    for name, (A, At) in {'named + token-level (E6 rerun)': (np.c_[zF, tF], np.c_[zT, tT]),
                          'named + token-level + interactions': (np.c_[zF, tF, iF], np.c_[zT, tT, iT]),
                          'interactions only': (iF, iT)}.items():
        pred, how = QS.fit_systematic(A, avgF[k], At)
        r2 = DP.r2(avgT[k], pred); out['rows'].append(dict(features=name, n_features=int(A.shape[1]), R2=r2, explained_over_shared=r2 * float(np.var(avgT[k])) / sv, model=how))
        print(out['rows'][-1], flush=True)
        json.dump(out, open('results/qinter.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
