"""Pre-training rewiring audit for World FA (PREREG-FA.md, section Audit).

Question: on the TRAINING prior, how much of Bayes's predictive mean does a structure-blind rule miss?
  structure-blind rule = 'prec': precision weighting that ignores correlations, per-trader variances from history.
  miss_R2  = 1 - R^2 of the best affine fit m_bayes ~ a + b m_prec   (gate: >= 0.15)
  miss_mse = E (m_bayes - m_prec)^2 / Var(m_bayes)
  The same with 'diag_bayes' (exact Bayes with correlations assumed away; strongest correlation-blind rule) and the
  joint miss of the affine fit on (diag_bayes, bayes_T0) (the in-context structure share).
Also reported: the same for 'equal', 'gls_plugin', 'bayes_T0'; NLL gaps to Bayes (nats); one-hop dedup and history
gate readings (Bayes's log posterior-variance ratio across f, KL between gate arms).

usage: python audit_fa.py --cfg '{"prior": "a"}' --nmin 3 --nmax 8 --E 3000 [--sweep a|b] --out audit_X.json
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
import torch

import fa_core as C

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds'))


def _r2_miss(y, *xs):
    X = np.c_[(np.ones_like(y),) + xs]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float((res ** 2).mean() / y.var())


def audit(cfg, nmin, nmax, E, seed=0, chunk=500):
    c = C.full_cfg(cfg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ('bayes', 'bayes_T0', 'diag_bayes', 'prec', 'equal', 'gls_plugin', 'oracle', 'prior')}
    nll = {k: [] for k in rows}
    Ts, Ns = [], []
    for j in range(0, E, chunk):
        e = min(chunk, E - j)
        n = np.exp(rng.uniform(np.log(nmin), np.log(nmax + 1), e)).astype(np.int64).clip(nmin, nmax)
        b = C.gen(e, torch.as_tensor(n), C.sample_T(e, c), c)
        refs = C.references(b, c)
        for k in rows:
            mu, _ = C.mixture_moments(*refs[k])
            rows[k].append(mu.numpy()); nll[k].append(C.nll_t(refs[k], b['v']).numpy())
        Ts.append(b['T'].numpy()); Ns.append(b['N'].numpy())
    m = {k: np.concatenate(v) for k, v in rows.items()}
    L = {k: np.concatenate(v) for k, v in nll.items()}
    T = np.concatenate(Ts); N = np.concatenate(Ns)
    out = dict(cfg=json.loads(json.dumps(c)), nmin=nmin, nmax=nmax, E=E, seed=seed)

    def block(sel):
        mb = m['bayes'][sel]
        r = dict(n=int(sel.sum()), var_m_bayes=float(mb.var()))
        for k in ('diag_bayes', 'prec', 'equal', 'gls_plugin', 'bayes_T0', 'oracle'):
            r['miss_R2_' + k] = _r2_miss(mb, m[k][sel])
            r['miss_mse_' + k] = float(((mb - m[k][sel]) ** 2).mean() / mb.var())
        # in-context structure share: what neither the correlation-blind Bayes (uses history, ignores correlations)
        # nor the history-free Bayes (knows the prior's structure, ignores history) explains, even combined
        r['miss_R2_joint_diag_T0'] = _r2_miss(mb, m['diag_bayes'][sel], m['bayes_T0'][sel])
        for k in L:
            d = L[k][sel] - L['bayes'][sel]
            r['gap_' + k] = (float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))))
        return r
    out['all'] = block(np.ones(len(T), bool))
    out['by_T'] = {lab: block((T >= lo) & (T <= hi)) for lab, lo, hi in
                   (('T0', 0, 0), ('T1-4', 1, 4), ('T5-16', 5, 16), ('T17+', 17, 10 ** 9)) if ((T >= lo) & (T <= hi)).sum() > 10}
    out['by_N'] = {str(n): block(N == n) for n in sorted(set(N.tolist())) if (N == n).sum() > 10}
    return out


def onehop(cfg, N, T, k, G=200, seed=1):
    """Bayes readings on the one-hop dedup and gate sets (graded f)."""
    import harness as H
    import world_fa as WF
    c = C.full_cfg(cfg)
    torch.manual_seed(seed)
    res = {}
    for kind, fs in (('dedup', WF.DEDUP_FS), ('gate', WF.GATE_FS)):
        b = C.block_sets(G, N, T, k, fs, c, gate=(kind == 'gate'))
        p = C.bayes(b, c)
        mu, var = C.mixture_moments(*p)
        A = len(fs)
        var = var.numpy().reshape(G, A); mu = mu.numpy().reshape(G, A)
        pn = [x.numpy().reshape(G, A, -1) for x in p]
        r = dict(f=list(fs), log_var_ratio_vs_f0=[float(np.log(var[:, a] / var[:, fs.index(0.0)]).mean()) for a in range(A)])
        if kind == 'gate':
            kl = [float(H.kl_bayes_model(tuple(x[:, 0] for x in pn), tuple(x[:, a] for x in pn)).mean()) for a in range(A)]
            r['kl_arm0_vs_arm'] = kl
            r['mean_shift_sd'] = [float((np.abs(mu[:, 0] - mu[:, a]) / np.sqrt(var[:, 0])).mean()) for a in range(A)]
        res[kind] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--nmin', type=int, default=3)
    ap.add_argument('--nmax', type=int, default=8); ap.add_argument('--E', type=int, default=3000)
    ap.add_argument('--onehop', default='', help='N,T,k for the one-hop readings, e.g. 6,16,3')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    t0 = time.time()
    res = audit(json.loads(a.cfg), a.nmin, a.nmax, a.E)
    if a.onehop:
        N, T, k = (int(x) for x in a.onehop.split(','))
        res['onehop'] = dict(N=N, T=T, k=k, **onehop(json.loads(a.cfg), N, T, k))
    res['seconds'] = time.time() - t0
    s = res['all']
    print(json.dumps(dict(miss_R2_diag=s['miss_R2_diag_bayes'], miss_R2_joint=s['miss_R2_joint_diag_T0'],
                          miss_R2_prec=s['miss_R2_prec'], miss_mse_prec=s['miss_mse_prec'],
                          gap_prec=s['gap_prec'], gap_T0=s['gap_bayes_T0'], gap_gls=s['gap_gls_plugin'],
                          by_T={k: round(v['miss_R2_prec'], 3) for k, v in res['by_T'].items()},
                          onehop=res.get('onehop')), indent=None))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
