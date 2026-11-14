"""Numbers for the World FA explainer page (mats.ekelvin.com/fa/). Uses the real generator, exact Bayes and
references from work/world-fa/fa_core.py and the pilot checkpoints in work/world-fa/runs/fa-pilot-1028b/.
Writes example.json (the worked episode and its answers) and prints population checks.

usage: OMP_NUM_THREADS=2 /home/agent/.venvs/aggext/bin/python make_example.py
"""
import itertools
import json
import math
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
WF = '/home/agent/workspaces/mats/work/world-fa'
sys.path[:0] = [WF, '/home/agent/workspaces/mats/work/worlds']
import numpy as np
import torch

import fa_core as C
import harness as H

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.json')
ANTI = {"prior": "f", "f_templates": [{"blocks": [[2, 0.8, 1]], "marg": 1.0}]}
NULL = {"prior": "f", "f_templates": [{"blocks": [[2, 0.8, 0]], "marg": 1.0}]}
N, T = 4, 8
npy = lambda p: tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)


def mom(p):
    m, v = H.moments(p)
    return float(m[0]), float(math.sqrt(v[0]))


def nll1(p, y):
    return float(H.nll(p, np.array([y]))[0])


def population(c, E=4000, seed=12345):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.gen(E, N, T, c)
    pb, p0 = npy(C.bayes(b, c, True)), npy(C.bayes(b, c, False))
    y = b['v'].numpy()
    kl = H.kl_bayes_model(pb, p0)
    hv = H.nll(p0, y) - H.nll(pb, y)
    _, _, post, _ = C.bayes_f(b, c, return_post=True)
    S, _, _, _ = C.family(N, C._tkey(c['f_templates']))
    true_h = np.array([int(np.argmin(np.abs(S - s.numpy()).sum((1, 2)))) for s in b['Sigma']])
    p_true = post.numpy()[np.arange(E), true_h]
    return dict(E=E, kl=kl, hv=hv, p_true=p_true)


def main():
    c = C.full_cfg(ANTI)
    S_fam, lp_fam, _, _ = C.family(N, C._tkey(c['f_templates']))
    pairs = []
    for S in S_fam:
        ij = [(i, j) for i in range(N) for j in range(i + 1, N) if abs(S[i, j]) > 0]
        pairs.append(ij[0])
    print('family M =', len(S_fam), 'pairs', pairs, 'prior', np.exp(lp_fam).round(4).tolist())

    # ---------------- population at N = 4, T = 8: per-episode history value = KL(Bayes || history-blind Bayes)
    pop = population(c)
    q = np.quantile(pop['kl'], [0.25, 0.5, 0.75, 0.9])
    print(f"T=8 population E={pop['E']}: mean KL(bayes||T0) {pop['kl'].mean():.4f}, mean NLL diff {pop['hv'].mean():.4f}"
          f" (se {pop['hv'].std() / math.sqrt(pop['E']):.4f}); KL quartiles {q.round(4).tolist()};"
          f" P(posterior on true pair > 0.9) {np.mean(pop['p_true'] > 0.9):.3f}; median p_true {np.median(pop['p_true']):.3f}")

    # ---------------- pick the example: first seed whose KL lies between the population median and 75th percentile
    seed = None
    for s in range(1000):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s)
            b = C.gen(1, N, T, c)
        pb_ = npy(C.bayes(b, c, True))
        k = float(H.kl_bayes_model(pb_, npy(C.bayes(b, c, False)))[0])
        # PIT of the realised v under the Bayes predictive (mixture CDF)
        w_, m_, v_ = pb_
        from math import erf
        pit = float(sum(w_[0, j] * 0.5 * (1 + erf((float(b['v'][0]) - m_[0, j]) / math.sqrt(2 * v_[0, j])))
                        for j in range(w_.shape[1])))
        ok = q[1] <= k <= q[2]
        print(f'seed {s}: KL {k:.4f} in [median, q75]: {ok}; PIT {pit:.3f}')
        if ok and 0.25 <= pit <= 0.75:
            seed = s
            break
    print('example seed', seed, 'KL', k, 'percentile', float((pop['kl'] < k).mean()), 'PIT', pit)

    refs = C.references(b, c)
    refs = {kk: npy(v) for kk, v in refs.items()}
    y = float(b['v'][0])
    (wf, mf, Vf), lev, post, beta = C.bayes_f(b, c, return_post=True)
    (w0, m0, V0), _, post0, beta0 = C.bayes_f(b, c, use_hist=False, return_post=True)
    Sig = b['Sigma'][0].numpy()
    true_h = int(np.argmin(np.abs(S_fam - Sig).sum((1, 2))))
    eh, Ssc = C.hist_stats(b)
    eh = eh[0].numpy(); Ssc = Ssc[0].numpy()
    r_hist = b['r_hist'][0].numpy(); v_hist = b['v_hist'][0].numpy(); r = b['r'][0].numpy()
    corr = Ssc / np.sqrt(np.outer(np.diag(Ssc), np.diag(Ssc)))

    # unordered-history check: p(column multiset | h) is the same for every h (so forgetting labels = history-blind)
    Pi = [np.linalg.inv(S) for S in S_fam]
    ld = [np.linalg.slogdet(S)[1] for S in S_fam]
    def loglik(E_, h):
        Sc = E_.T @ E_
        return -0.5 * T * (N * math.log(2 * math.pi) + ld[h]) - 0.5 * float(np.sum(Pi[h] * Sc))
    lu = []
    for h in range(len(S_fam)):
        vals = [loglik(eh[:, list(p)], h) for p in itertools.permutations(range(N))]
        mx = max(vals)
        lu.append(mx + math.log(sum(math.exp(x - mx) for x in vals)))
    print('log p(unordered history | h) for the 6 hypotheses:', np.round(lu, 10).tolist())

    # per-pair history score and posterior without / with the query factor
    score = []
    for (a_, b2) in pairs:
        s_ = eh[:, a_] + eh[:, b2]; d_ = eh[:, a_] - eh[:, b2]
        score.append(float(-(s_ ** 2).sum() + (d_ ** 2).sum() / 9))
    score = np.array(score)
    post_hist = np.exp(score - score.max()); post_hist /= post_hist.sum()
    (_, _, _), _, post_h_only, _ = C.bayes_f(b, c, query_weights=False, return_post=True)
    print('pair scores', score.round(3).tolist())
    print('posterior from scores', post_hist.round(6).tolist())
    print('bayes_f history-only posterior', post_h_only[0].numpy().round(6).tolist())
    rq = r
    qterm = []
    for h in range(len(S_fam)):
        A = S_fam[h] + 1.0
        qterm.append(float(-0.5 * rq @ np.linalg.solve(A, rq)))
    qterm = np.array(qterm)
    print('query-term (log, up to const)', (qterm - qterm.max()).round(3).tolist())

    # what the trained pilot networks say on this episode
    nets = {}
    for arm in ('anti_report_s0', 'anti_report_s1', 'anti_column_s0', 'anti_column_s1'):
        model, world, meta = H.load_run(os.path.join(WF, 'runs/fa-pilot-1028b', arm), 'best.pt')
        X, M = C.tokens(b, world.c)
        pm = H.predict(model, X.numpy(), M.numpy())
        nets[arm] = dict(mean=mom(pm)[0], sd=mom(pm)[1], nll=nll1(pm, y),
                         kl_from_bayes=float(H.kl_bayes_model(refs['bayes'], pm)[0]),
                         kl_from_T0=float(H.kl_bayes_model(refs['bayes_T0'], pm)[0]),
                         n_params=int(meta['n_params']), depth=meta['args']['depth'], width=meta['args']['width'],
                         heads=meta['args']['heads'], mix=meta['args']['mix'], steps=meta['args']['steps'],
                         best_step=meta['best_step'])
        print(arm, nets[arm])

    X, M = C.tokens(b, c)
    X = X[0].numpy(); Mk = M[0].numpy()
    di = c['d_id']; dr = c['d_round']; o = di + dr
    def tok(ix):
        x = X[ix]
        return dict(id=x[:di].round(3).tolist(), rid=x[di:o].round(3).tolist(), r=float(x[o + 0]), v=float(x[o + 1]),
                    e=float(x[o + 2]), hist=float(x[o + 3]), qrep=float(x[o + 4]), query=float(x[o + 5]),
                    logN=float(x[o + 6]), logT=float(x[o + 7]))
    tokens = dict(L=int(X.shape[0]), F=int(X.shape[1]), valid=int(Mk.sum()),
                  hist_t1_i1=tok(0), hist_t1_i2=tok(1), hist_t2_i1=tok(N), qrep_i1=tok(T * N), qrep_i2=tok(T * N + 1),
                  query=tok(T * N + N))

    # GLS weights under each hypothesis and the oracle
    u = [P.sum(1) for P in Pi]
    out = dict(
        seed=seed, N=N, T=T, cfg=ANTI, family_pairs=pairs, true_h=true_h, true_pair=list(pairs[true_h]),
        Sigma=Sig.round(6).tolist(), v_hist=v_hist.tolist(), r_hist=r_hist.tolist(), e_hist=eh.tolist(), r=r.tolist(),
        v=y, scatter=Ssc.tolist(), sample_cov=(Ssc / T).tolist(), sample_corr=corr.tolist(),
        post_full=post[0].tolist(), post_T0=post0[0].tolist(), beta_full=beta[0].tolist(), beta_T0=beta0[0].tolist(),
        gls_weights_by_h=[x.tolist() for x in u],
        answers=dict(
            plain_average=dict(mean=float(r.mean())),
            equal=dict(zip(('mean', 'sd'), mom(refs['equal'])), nll=nll1(refs['equal'], y)),
            corr_blind=dict(zip(('mean', 'sd'), mom(refs['diag_bayes'])), nll=nll1(refs['diag_bayes'], y)),
            prec=dict(zip(('mean', 'sd'), mom(refs['prec'])), nll=nll1(refs['prec'], y)),
            history_blind=dict(zip(('mean', 'sd'), mom(refs['bayes_T0'])), nll=nll1(refs['bayes_T0'], y),
                               comps=dict(w=w0[0].tolist(), m=m0[0].tolist(), V=V0[0].tolist())),
            bayes=dict(zip(('mean', 'sd'), mom(refs['bayes'])), nll=nll1(refs['bayes'], y),
                       comps=dict(w=wf[0].tolist(), m=mf[0].tolist(), V=Vf[0].tolist())),
            oracle=dict(zip(('mean', 'sd'), mom(refs['oracle'])), nll=nll1(refs['oracle'], y)),
            prior=dict(mean=0.0, sd=1.0, nll=nll1(refs['prior'], y)),
            gls_plugin=dict(zip(('mean', 'sd'), mom(refs['gls_plugin'])), nll=nll1(refs['gls_plugin'], y)),
        ),
        kl_bayes_vs_T0=k, kl_percentile=float((pop['kl'] < k).mean()),
        population_T8=dict(E=pop['E'], mean_kl=float(pop['kl'].mean()), mean_hv=float(pop['hv'].mean()),
                           se_hv=float(pop['hv'].std() / math.sqrt(pop['E'])), kl_quartiles=q.tolist(),
                           p_true_gt_09=float(np.mean(pop['p_true'] > 0.9)), median_p_true=float(np.median(pop['p_true']))),
        unordered_loglik=lu, nets=nets, tokens=tokens, pair_score=score.tolist(), post_hist_only=post_hist.tolist(),
        query_term=(qterm - qterm.max()).tolist(),
        uid_first3=b['uid'][0, :, :3].numpy().round(3).tolist(), rid_first3=b['rid'][0, :, :3].numpy().round(3).tolist(),
    )
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({k2: out[k2] for k2 in ('seed', 'true_pair', 'v', 'r', 'post_full', 'post_T0', 'beta_full', 'beta_T0')}, indent=None))
    for k2, v2 in out['answers'].items():
        print(k2, {kk: (round(vv, 4) if isinstance(vv, float) else '') for kk, vv in v2.items() if kk != 'comps'})
    print('sample corr\n', np.round(corr, 3))
    print('v_hist', np.round(v_hist, 3))
    print('e_hist\n', np.round(eh, 3))
    print('r_hist\n', np.round(r_hist, 3))


if __name__ == '__main__':
    main()
