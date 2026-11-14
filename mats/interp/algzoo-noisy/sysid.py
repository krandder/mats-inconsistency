"""Amendment 7, step 1: systematic vs idiosyncratic share of the network's errors.

Per input x: eps_A, eps_B = error frequencies over two independent halves of 16 noise draws, e = one more draw.
Fit E[eps | features] with LightGBM (cross-entropy on the soft label eps_A) on the train split, early stopping on the
validation split; score on the test split:
  share = noise-corrected R^2 against eps(x) = 1 - [MSE(pred, eps_B) - mean(eps_B(1-eps_B))/15] / Cov(eps_A, eps_B)
  (and the same with A and B swapped; reported as the mean), Brier skill on e (with its ceiling), AUC on e.
Feature sets: F1 (coordinator's), F2 (top-3 values and positions), F3 (all order statistics and positions).
Excess over the algorithmic readout: the same on inputs where the readout of the CL conditional means at step n
(fitted on separate base-model samples) is right.
usage: python sysid.py MODEL.npz OUT.json LOG2N
"""
import sys, json, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import az, alg

t0 = time.time()
path, out, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = [w.astype(np.float32) for w in W]; r32 = r.astype(np.float32)
d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(777)
KH = 16


def last_h(x, rng, noise=True):
    B = x.shape[0]; h = np.zeros((B, d), np.float32)
    for t in range(n):
        h = np.maximum(h @ W_hh.T + x[:, t:t + 1] * W_hi, 0)
        if noise: h = h + sigma * r32[t] * rng.standard_normal((B, d), dtype=np.float32)
    return h


def err_counts(x, K, rng, chunk=4096):
    y = np.argsort(x, -1)[:, -2]; cnt = np.zeros(x.shape[0])
    for i in range(0, x.shape[0], chunk):
        xc = np.repeat(x[i:i + chunk].astype(np.float32), K, 0)
        lo = last_h(xc, rng) @ W_oh.T
        cnt[i:i + chunk] = (lo.argmax(-1) != np.repeat(y[i:i + chunk], K)).reshape(-1, K).sum(1)
    return cnt


N = 2 ** L
x = rng.standard_normal((N, n))
eA = err_counts(x, KH, rng) / KH; eB = err_counts(x, KH, rng) / KH; e1 = err_counts(x, 1, rng)
print('sampled', N, 'p %.5f' % (0.5 * (eA.mean() + eB.mean())), '%.0fs' % (time.time() - t0), flush=True)

# algorithmic readout (CL conditional means at step n, fitted on separate base-model samples)
F_ = 7


def feats(M, S, xl):
    return np.stack([np.ones_like(M), M, S, xl, np.maximum(M, 0), np.maximum(S, 0), np.maximum(xl, 0)], -1)


xf = rng.standard_normal((2 ** 17, n)); hf = np.concatenate([last_h(xf[i:i + 8192].astype(np.float32), rng) for i in range(0, 2 ** 17, 8192)])
Af = alg.alg_states(xf); keyf = Af['a'][:, n] * n + Af['b'][:, n]
Ff = feats(Af['M'][:, n], Af['S'][:, n], xf[:, n - 1]); beta = {}
for k in np.unique(keyf):
    s = keyf == k; beta[int(k)] = np.linalg.lstsq(Ff[s], hf[s].astype(np.float64), rcond=None)[0]
A = alg.alg_states(x); key = A['a'][:, n] * n + A['b'][:, n]; Fx = feats(A['M'][:, n], A['S'][:, n], x[:, n - 1])
halg = np.zeros((N, d))
for k in np.unique(key):
    s = key == k; halg[s] = Fx[s] @ beta[int(k)]
y = A['b'][:, n]
e_alg = (halg @ W_oh.T.astype(np.float64)).argmax(1) != y
print('algorithmic readout error %.5f' % e_alg.mean(), flush=True)

# features
idx = np.argsort(-x, 1); srt = np.take_along_axis(x, idx, 1)
g12 = srt[:, 0] - srt[:, 1]; g23 = srt[:, 1] - srt[:, 2]; sg2 = (srt[:, 1] > 0).astype(float)
pos1, pos2, pos3 = idx[:, 0].astype(float), idx[:, 1].astype(float), idx[:, 2].astype(float)
tlast = np.maximum(pos1, pos2)
FS = dict(F1=np.stack([g12, g23, sg2, pos1, pos2, tlast], 1),
          F2=np.stack([g12, g23, sg2, pos1, pos2, tlast, srt[:, 0], srt[:, 1], srt[:, 2], pos3], 1),
          F3=np.concatenate([srt, idx.astype(float)], 1))
names = dict(F1=['g12', 'g23', 'sign2', 'pos1', 'pos2', 'tlast'],
             F2=['g12', 'g23', 'sign2', 'pos1', 'pos2', 'tlast', 'x1', 'x2', 'x3', 'pos3'],
             F3=['o%d' % i for i in range(n)] + ['p%d' % i for i in range(n)])
perm = rng.permutation(N); tr, va, te = perm[:N // 2], perm[N // 2:3 * N // 4], perm[3 * N // 4:]
PARAMS = dict(objective='cross_entropy', num_leaves=63, learning_rate=0.05, min_data_in_leaf=200, verbose=-1,
              num_threads=1, seed=1)


def metrics(pred, ia):
    a, b, e = eA[ia], eB[ia], e1[ia]
    var_eps = float(np.cov(a, b)[0, 1])
    r2b = 1 - (np.mean((pred - b) ** 2) - np.mean(b * (1 - b)) / (KH - 1)) / var_eps
    r2a = 1 - (np.mean((pred - a) ** 2) - np.mean(a * (1 - a)) / (KH - 1)) / var_eps
    p = e.mean(); ref = np.mean((eA[tr].mean() - e) ** 2)
    out_ = dict(n=int(len(ia)), p=float(0.5 * (a.mean() + b.mean())), var_eps=var_eps, share=float(0.5 * (r2a + r2b)),
                share_A=float(r2a), share_B=float(r2b), brier_skill=float(1 - np.mean((pred - e) ** 2) / ref),
                brier_ceiling=float(var_eps / (p * (1 - p))), auc=float(roc_auc_score(e, pred)) if 0 < e.sum() < len(e) else None,
                auc_ceiling=float(roc_auc_score(e, 0.5 * (a + b))) if 0 < e.sum() < len(e) else None)
    return out_


def fit(Fm, itr, iva, ite, nm):
    dtr = lgb.Dataset(Fm[itr], label=eA[itr], feature_name=nm); dva = lgb.Dataset(Fm[iva], label=eA[iva], reference=dtr)
    bst = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = bst.predict(Fm[ite], num_iteration=bst.best_iteration)
    m = metrics(pred, ite); m['rounds'] = int(bst.best_iteration)
    gain = bst.feature_importance('gain'); m['importance'] = {nm[i]: float(gain[i] / gain.sum()) for i in np.argsort(-gain)}
    return m


res = dict(model=path, sigma=sigma, N=N, p=float(0.5 * (eA.mean() + eB.mean())), p_alg_readout=float(e_alg.mean()),
           total={}, excess={})
for fs in ('F1', 'F2', 'F3'):
    res['total'][fs] = fit(FS[fs], tr, va, te, names[fs])
    print('total', fs, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['total'][fs].items() if k != 'importance'}, flush=True)
    json.dump(res, open(out, 'w'), indent=1)
ok = ~e_alg
res['excess_info'] = dict(frac_inputs=float(ok.mean()), p_on_subset=float(0.5 * (eA[ok].mean() + eB[ok].mean())),
                          excess_rate=float(0.5 * (eA * ok).mean() + 0.5 * (eB * ok).mean()))
for fs in ('F1', 'F2', 'F3'):
    res['excess'][fs] = fit(FS[fs], tr[ok[tr]], va[ok[va]], te[ok[te]], names[fs])
    print('excess', fs, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['excess'][fs].items() if k != 'importance'}, flush=True)
    json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print('done %.0fs' % res['seconds'])
