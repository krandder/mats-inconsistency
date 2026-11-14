"""Amendment 6: pool of 4,000,000 prior episodes (c only) with q0's six query-token features, for the matched
query-token oracle O_q0m (r2qt.py). Writes /tmp/r2scratch/qtpool.npz (scratch; regenerable from seed 75000)."""
import sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp, oracle as O  # noqa: E402

W = tp.W


def feats_c(Qc, Sc, chunk=4096):
    """q0's six QT features from c (identical to tp.cell_stats to 1e-13): mean Q, Bayes mean, Bayes log var,
    labelled-only mean, labelled-only log var, sum_s beta_s Q_s."""
    n = len(Qc); out = np.empty((n, 6))
    v = 1.0 / O.PREC
    for i in range(0, n, chunk):
        Q_, S_ = Qc[i:i + chunk], Sc[i:i + chunk]
        lw = O.bayes_logw(Q_, S_); lw -= lw.max(1, keepdims=True); wb = np.exp(lw); wb /= wb.sum(1, keepdims=True)
        ll = (-0.5 * S_[:, None, :] / O.VC[None] - 2.0 * np.log(O.VC)[None]).sum(2); ll -= ll.max(1, keepdims=True)
        wl = np.exp(ll); wl /= wl.sum(1, keepdims=True)
        means = (Q_ @ O.IVC) / O.PREC[None]
        mb = (wb * means).sum(1); vb = (wb * (v[None] + means ** 2)).sum(1) - mb ** 2
        ml = (wl * means).sum(1); vl = (wl * (v[None] + means ** 2)).sum(1) - ml ** 2
        out[i:i + chunk] = np.stack([Q_.mean(1), mb, np.log(vb), ml, np.log(vl), ((wb @ O.BCOEF) * Q_).sum(1)], 1)
    return out


if __name__ == '__main__':
    T0 = time.time()
    P = 4_000_000
    rng = np.random.default_rng(75000)
    Qs, Ss = [], []
    for i in range(0, P, 500_000):
        dp = W.sample_np(500_000, rng)
        R = dp['o'][:, :, :tp.QL] - dp['y'][:, None, :tp.QL]
        Qs.append(dp['o'][:, :, tp.QL]); Ss.append((R ** 2).sum(2))
    Q, S = np.concatenate(Qs), np.concatenate(Ss)
    f = feats_c(Q, S)
    d = tp.sample_episodes(2000, 5); cs = tp.cell_stats(d); Qx, Sx, _ = O.cvars(d)
    ref = np.stack([cs['ostar'].mean(1), cs['mb'], cs['lvb'], cs['ml'], cs['lvl'], (cs['beta'] * cs['ostar']).sum(1)], 1)
    print('feature check vs cell_stats: max |diff| %.1e' % np.abs(feats_c(Qx, Sx) - ref).max())
    np.savez('/tmp/r2scratch/qtpool.npz', Q=Q, S=S, f=f)
    print('pool saved, %.0fs' % (time.time() - T0))
