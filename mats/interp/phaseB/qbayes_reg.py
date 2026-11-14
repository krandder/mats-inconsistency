"""E3a diagnostic (post hoc): is the network's deviation from Bayes a function of the exact Bayes quantities at all?
Direct regression of the network's K = 64 deviation (mean, log-sd, KL) on b(x) (the 24 E3a features) with gradient-boosted
trees, trained on 16,384 fit episodes, scored by R^2 on the 2,000 test episodes against the network's independent K = 64
deviation (as E1/E3a). Also with b + identity-crosstalk features (max |cos| between source ID embeddings, from the tokens),
to test the S6 explanation.
usage: python qbayes_reg.py RUN_DIR -> RUN_DIR/qbayes_reg.json
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
import nnoise as N
import world as W
import qw
import devpred as DP
import qbayes as QB

torch.set_num_threads(1)


def id_feats(X):
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64)
    sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    cos = np.stack([(sn[:, a] * sn[:, b]).sum(1) for a, b in ((0, 1), (0, 2), (1, 2))], 1)
    qemb = X[:, 15, 16:32].astype(np.float64); qn = qemb / np.linalg.norm(qemb, axis=1, keepdims=True)
    return np.c_[cos, np.abs(cos).max(1), np.linalg.norm(semb, axis=2), np.linalg.norm(qemb, axis=1)]


def run(run_dir, n_fit=16384, K=64):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    rng = np.random.default_rng(94_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    ref, _, mom, wb = W.references(d)
    Tf = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n_test = 2000
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt, vt = Tz['X'][:n_test], Tz['valid'][:n_test]
    dT = dict(y=Tz['y'][:n_test], o=Tz['o'][:n_test], v=Tz['v'][:n_test], vidx=Tz['vidx'][:n_test])
    cache = os.path.join(run_dir, 'qbayes_reg_dev.npz')
    if os.path.exists(cache):
        z = np.load(cache); devF = {k: z['F_' + k] for k in ('mean', 'logsd', 'kl')}; devT = {k: z['T_' + k] for k in devF}; devT2 = {k: z['T2_' + k] for k in devF}
    else:
        devs = []
        for i in range(0, n_fit, 2000):
            T_ = {k: (a[i:i + 2000] if k != 'comp_vars' else a) for k, a in Tf.items()}
            h, _, _ = DP.marginal(model, X[i:i + 2000], valid[i:i + 2000], sg, {}, K, 31 + i); devs.append(qw.deviations_t(h, T_))
        devF = {k: np.concatenate([x[k] for x in devs]) for k in devs[0]}
        h1, _, _ = DP.marginal(model, Xt, vt, sg, {}, K, 1); devT = qw.deviations_t(h1, Tt)
        h2, _, _ = DP.marginal(model, Xt, vt, sg, {}, K, 7); devT2 = qw.deviations_t(h2, Tt)
        np.savez(cache, **{'F_' + k: v for k, v in devF.items()}, **{'T_' + k: v for k, v in devT.items()}, **{'T2_' + k: v for k, v in devT2.items()})
    bF, bT = QB.bayes_features(d), QB.bayes_features(dT)
    out = dict(floor={k: DP.r2(devT[k], devT2[k]) for k in devT}, rows=[])
    readings = lambda d_: np.c_[d_['o'][:, :, W.QL], ((d_['o'][:, :, :W.QL] - d_['y'][:, None, :W.QL]) ** 2).sum(2)]   # query readings, labelled SS
    sets = {'bayes': (bF, bT), 'bayes+ids': (np.c_[bF, id_feats(X)], np.c_[bT, id_feats(Xt)]), 'ids only': (id_feats(X), id_feats(Xt)),
            'bayes+readings': (np.c_[bF, readings(d)], np.c_[bT, readings(dT)]),
            'bayes+readings+ids': (np.c_[bF, readings(d), id_feats(X)], np.c_[bT, readings(dT), id_feats(Xt)]),
            'raw tokens': (X.reshape(len(X), -1), Xt.reshape(len(Xt), -1))}
    if os.path.exists(os.path.join(run_dir, 'qbayes_reg.json')):
        prev = json.load(open(os.path.join(run_dir, 'qbayes_reg.json'))); done_ = {r['features'] for r in prev['rows']}; out['rows'] = prev['rows']
    else: done_ = set()
    for name, (A, At) in sets.items():
        if name in done_: continue
        row = dict(features=name)
        for k in ('mean', 'logsd', 'kl'):
            reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(A, devF[k])
            row[k] = DP.r2(devT[k], reg.predict(At))
        out['rows'].append(row); print(row, flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, 'qbayes_reg.json'), 'w'), indent=1)
    print('done', out['seconds'], 'floor', out['floor'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
