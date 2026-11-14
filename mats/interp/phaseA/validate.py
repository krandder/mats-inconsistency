"""Known-answer validation of cmlib (cases i-iv of the task). Writes results/validation.json and results/tables.md.
Run: OMP_NUM_THREADS=1 python validate.py [--procs 4] [--quick]"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch  # noqa: E402
from cmlib import synth, estimators as E, ratio as R, qfam  # noqa: E402
from cmlib.consistency import consistency_check  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
AL = np.linspace(0, 1, 21)
NB = 200  # bootstrap replicates inside estimators
oh = lambda z, k: np.eye(k)[z]


def _covers(lo, hi, x):
    return bool(lo <= x <= hi)


def ratio_block(Gp, lwp, Gq, lwq, gp, gq, truth):
    """All ratio-based estimators of Delta = E_q G - E_p G from one set of log-ratios, plus trust checks
    T1 (bridge endpoints vs direct means: within max(2 SE, 0.05 |Delta_direct|)) and T2 (E_q w, E_p 1/w within 0.05)."""
    c = E.cov_qw(Gq, lwq, gq, B=NB)
    tq = E.temper_q(Gq, lwq, AL, groups=gq, B=NB // 2)
    tb = E.temper_bridge(Gp, lwp, Gq, lwq, AL, groups_p=gp, groups_q=gq, B=NB // 2)
    dd = abs(Gq.mean() - Gp.mean())
    t1 = (abs(tb['L'][0] - Gq.mean()) <= max(2 * np.hypot(tb['se'][0], Gq.std() / np.sqrt(len(Gq))), 0.05 * dd)
          and abs(tb['L'][-1] - Gp.mean()) <= max(2 * np.hypot(tb['se'][-1], Gp.std() / np.sqrt(len(Gp))), 0.05 * dd))
    t2 = abs(np.mean(np.exp(lwq)) - 1) <= 0.05 and abs(np.mean(np.exp(-lwp)) - 1) <= 0.05
    return dict(
        cov_plain=-c['cov_plain'], cov_plain_se=c['plain_se'], cov_plain_cov=_covers(-c['plain_hi'], -c['plain_lo'], truth),
        cov_snis=-c['cov_snis'], cov_snis_se=c['snis_se'], cov_snis_cov=_covers(-c['snis_hi'], -c['snis_lo'], truth),
        temper_q=tq['gap'], temper_q_se=tq['gap_se'], temper_q_cov=_covers(tq['gap_lo'], tq['gap_hi'], truth),
        bridge=tb['gap'], bridge_se=tb['gap_se'], bridge_cov=_covers(tb['gap_lo'], tb['gap_hi'], truth),
        bridge_int=-tb['integral'], bridge_int_se=tb['integral_se'],
        ess_q=c['ess'] / len(Gq), ess_bridge_min=float(tb['ess'].min()) / (len(Gp) + len(Gq)),
        T1=bool(t1), T2=bool(t2), L_bridge=tb['L'].tolist(), L_q=tq['L'].tolist())


def refit_block(r, Hp_tr, Hq_tr, Zp_tr, Zq_tr, Gp, Gq, Xp, Xq, Zp, Zq, gp, gq, truth, family, B, seed):
    """Add classifier-refit spread (half-sampling) to the ratio-fixed SE of cov_plain and bridge; record coverage."""
    def est(rm):
        lwp, lwq = rm.log_w(Xp, Zp), rm.log_w(Xq, Zq)
        c = E.cov_qw(Gq, lwq, gq, B=2)
        tb = E.temper_bridge(Gp, lwp, Gq, lwq, np.array([0.0, 1.0]), groups_p=gp, groups_q=gq, B=2)
        return [-c['cov_plain'], tb['gap']]
    sp = R.refit_spread(Hp_tr, Hq_tr, Zp_tr, Zq_tr, est, B=B, family=family, seed=seed)
    for j, k in enumerate(('cov_plain', 'bridge')):
        se_tot = float(np.hypot(r[k + '_se'], sp[:, j].std(ddof=1)))
        r[k + '_refit_sd'] = float(sp[:, j].std(ddof=1)); r[k + '_se_tot'] = se_tot
        r[k + '_cov_tot'] = bool(abs(r[k] - truth) <= 1.96 * se_tot)
    return r


def direct_block(Gq, Gp, truth, paired=False):
    d = E.direct_gap(Gq, Gp, paired=paired, B=NB)
    return dict(direct=d['delta'], direct_se=d['se'], direct_cov=_covers(d['lo'], d['hi'], truth))


# ============================================================================ (i) p = q
def job_null(world, family, rep, n=20000, refit=0):
    rng = np.random.default_rng(1000 + rep)
    if world == 'mix_exact':          # non-Gaussian p, q = p exactly
        W = synth.MixWorld(d=8, K=3, a=2.0, seed=0)
        draw_p = lambda z: W.sample(z, 1.0, rng); draw_q = draw_p; K = 3
        G = W.G; sz = lambda m: W.sample_z(m, rng)
    elif world == 'gauss_fitted':     # Gaussian p (a = 0), q = CondGaussian fitted on 20k separate draws
        W = synth.MixWorld(d=8, K=3, a=0.0, seed=0); K = 3
        zf = W.sample_z(n, rng); qf = qfam.CondGaussian('full').fit(W.sample(zf, 1.0, rng), zf)
        draw_p = lambda z: W.sample(z, 1.0, rng); draw_q = lambda z: qf.sample(z, rng=rng)
        G = lambda H, z: np.logaddexp(0, 2 * W.tau(H, z) - 1); sz = lambda m: W.sample_z(m, rng)
    z = sz(n); zq = sz(n)
    Hp_tr, Hq_tr = draw_p(z), draw_q(zq)
    rm, dg = R.fit_ratio(Hp_tr, Hq_tr, oh(z, K), oh(zq, K), family=family, seed=rep, null_gate=False)
    rmg, dgg = R.fit_ratio(Hp_tr, Hq_tr, oh(z, K), oh(zq, K), family=family, seed=rep, null_gate=True)
    z2 = sz(n); Hp2, Hq2 = draw_p(z2), draw_q(z2)
    Gp, Gq = G(Hp2, z2), G(Hq2, z2)
    lwp, lwq = rm.log_w(Hp2, oh(z2, K)), rm.log_w(Hq2, oh(z2, K))
    out = dict(case='i', world=world, family=family, rep=rep, truth=0.0,
               **{k: dg[k] for k in ('auc', 'auc_se', 'kl_plugin', 'dv', 'norm_q', 'norm_p', 'ece', 'js_lb',
                                     'train_minus_test_bce', 'khat_q')},
               logw_sd=float(np.std(lwq)), G_sd=float(Gp.std()),
               auc_sig=bool(abs(dg['auc'] - 0.5) > 2 * dg['auc_se']))
    out.update(direct_block(Gq, Gp, 0.0)); out.update(ratio_block(Gp, lwp, Gq, lwq, z2, z2, 0.0))
    cg = E.cov_qw(Gq, rmg.log_w(Hq2, oh(z2, K)), z2, B=NB)
    out.update(gated=dgg['gated'], gated_cov=-cg['cov_plain'],
               gated_cov_cov=_covers(-cg['plain_hi'], -cg['plain_lo'], 0.0))
    if refit:
        refit_block(out, Hp_tr, Hq_tr, oh(z, K), oh(zq, K), Gp, Gq, Hp2, Hq2, oh(z2, K), oh(z2, K), z2, z2, 0.0,
                    family, refit, rep)
    return out


# ============================================================================ (ii) analytic non-Gaussian ratio
def job_analytic(mode, rep, n=20000, refit=0):
    """mode 'true': exact w; 'mlp'/'quad': classifier w; 'fitq': q = CondGaussian fitted to p (direct only)."""
    rng = np.random.default_rng(2000 + rep)
    Ws = {g: synth.MixWorld(d=8, K=3, a=2.0, seed=0, g=g) for g in ('quartic', 'softplus', 'quadratic')}
    W = Ws['quartic']
    out = []
    if mode == 'fitq':
        zf = W.sample_z(n, rng); qf = qfam.CondGaussian('full').fit(W.sample(zf, 1.0, rng), zf)
        z2 = W.sample_z(n, rng); Hp, Hq = W.sample(z2, 1.0, rng), qf.sample(z2, rng=rng)
        for g, Wg in Ws.items():
            r = dict(case='ii', mode=mode, g=g, rep=rep, truth=Wg.delta(0.0))
            r.update(direct_block(Wg.G(Hq, z2), Wg.G(Hp, z2), r['truth'])); out.append(r)
        return out
    if mode in ('mlp', 'quad'):
        z = W.sample_z(n, rng); zq = W.sample_z(n, rng)
        Hp_tr, Hq_tr = W.sample(z, 1.0, rng), W.sample(zq, 0.0, rng)
        rm, dg = R.fit_ratio(Hp_tr, Hq_tr, oh(z, 3), oh(zq, 3), family=mode, seed=rep)
    z2 = W.sample_z(n, rng); Hp, Hq = W.sample(z2, 1.0, rng), W.sample(z2, 0.0, rng)
    if mode == 'true':
        lwp, lwq = W.log_w(Hp, z2, 0.0), W.log_w(Hq, z2, 0.0); dg = {}
    else:
        lwp, lwq = rm.log_w(Hp, oh(z2, 3)), rm.log_w(Hq, oh(z2, 3))
    lt = W.log_w(Hq, z2, 0.0)
    for g, Wg in Ws.items():
        truth = Wg.delta(0.0)
        r = dict(case='ii', mode=mode, g=g, rep=rep, truth=truth, kl_true=W.kl_pq(0.0), chi2_true=W.chi2_pq(0.0),
                 L_exact=Wg.L_alpha(0.0, AL).tolist(),
                 corr_logw=float(np.corrcoef(lwq, lt)[0, 1]) if mode != 'true' else 1.0,
                 rmse_logw=float(np.sqrt(np.mean((lwq - lt) ** 2))),
                 **{k: dg.get(k) for k in ('auc', 'kl_plugin', 'dv', 'ece', 'norm_q', 'norm_p', 'khat_q')})
        Gp, Gq = Wg.G(Hp, z2), Wg.G(Hq, z2)
        r.update(direct_block(Gq, Gp, truth)); r.update(ratio_block(Gp, lwp, Gq, lwq, z2, z2, truth))
        if mode in ('mlp', 'quad') and refit and g == 'quartic':
            refit_block(r, Hp_tr, Hq_tr, oh(z, 3), oh(zq, 3), Gp, Gq, Hp, Hq, oh(z2, 3), oh(z2, 3), z2, z2, truth,
                        mode, refit, rep)
        out.append(r)
    return out


# ============================================================================ (iii) downstream ignores structure
def job_ignore(rep, n=30000):
    rng = np.random.default_rng(3000 + rep)
    net = synth.IgnoreNet(c_bimodal=True, seed=0)
    y = rng.integers(0, 2, n); q = qfam.CondGaussian('full').fit(net.sample(y, rng), y)
    out = []
    for name, A in (('all', list(range(6))), ('A', net.A), ('B', net.B), ('C', net.C)):
        y2 = rng.integers(0, 2, n); H2 = net.sample(y2, rng); Gp = net.G(H2, y2)
        Hq = q.subset_sample(H2, A, y2, rng=rng)
        Gq = net.G(Hq, y2)
        y3 = rng.integers(0, 2, n); Hq3 = q.subset_sample(net.sample(y3, rng), A, y3, rng=rng)
        y4 = rng.integers(0, 2, n); H4 = net.sample(y4, rng)
        r = dict(case='iii', subset=name, rep=rep)
        yr = rng.integers(0, 2, 400000); Hr = net.sample(yr, rng)
        r['ref'] = float((net.G(q.subset_sample(Hr, A, yr, rng=rng), yr) - net.G(Hr, yr)).mean())
        r.update({k: v for k, v in E.direct_gap(Gq, Gp, paired=True, B=NB).items() if k in ('delta', 'se')})
        for space, f in (('h', lambda X: X), ('o', net.logits)):
            for gate in (False, True):
                rm, dg = R.fit_ratio(f(H4), f(Hq3), oh(y4, 2), oh(y3, 2), family='mlp', seed=rep, null_gate=gate)
                lwq, lwp = rm.log_w(f(Hq), oh(y2, 2)), rm.log_w(f(H2), oh(y2, 2))
                c = E.cov_qw(Gq, lwq, y2, B=NB)
                tb = E.temper_bridge(Gp, lwp, Gq, lwq, AL, groups_p=y2, groups_q=y2, B=NB // 2)
                tag = space + ('_gated' if gate else '')
                r.update({f'{tag}_auc': dg['auc'], f'{tag}_kl': dg['kl_plugin'], f'{tag}_cov': -c['cov_plain'],
                          f'{tag}_cov_se': c['plain_se'], f'{tag}_bridge': tb['gap'], f'{tag}_bridge_se': tb['gap_se'],
                          f'{tag}_gated': dg['gated'], f'{tag}_Lflat': float(np.ptp(tb['L']))})
        out.append(r)
    return out


def job_ignore_truth(n=2_000_000):
    """High-precision paired direct Delta per subset (the reference values for case iii)."""
    rng = np.random.default_rng(39)
    net = synth.IgnoreNet(c_bimodal=True, seed=0)
    y = rng.integers(0, 2, 30000); q = qfam.CondGaussian('full').fit(net.sample(y, rng), y)
    out = []
    for name, A in (('all', list(range(6))), ('A', net.A), ('B', net.B), ('C', net.C)):
        y2 = rng.integers(0, 2, n); H2 = net.sample(y2, rng)
        d = net.G(q.subset_sample(H2, A, y2, rng=rng), y2) - net.G(H2, y2)
        out.append(dict(case='iii_truth', subset=name, delta=float(d.mean()), se=float(d.std() / np.sqrt(n))))
    return out


def job_consistency(cb, rep):
    net = synth.IgnoreNet(c_bimodal=cb, seed=0)
    rng = np.random.default_rng(4000 + rep)
    yy = rng.integers(0, 2, 100000); ql = qfam.CondGaussian('full').fit(net.sample(yy, rng), yy)
    r = consistency_check(lambda z, rg: ql.sample(z, rng=rg), net.F, lambda: qfam.CondGaussian('full'),
                          lambda z, rg: net.F(net.sample(z, rg)), net.G_next, np.array([0, 1]), seed=rep, B=150)
    r = {k: v for k, v in r.items() if k not in ('q_tilde', 'q_fit')}
    r.update(case='iii_cons', c_bimodal=cb, rep=rep)
    return r


# ============================================================================ (iv) q -> p
def job_approach(t, rep, n=20000, refit=0):
    rng = np.random.default_rng(5000 + rep + int(1000 * t))
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=0)
    truth = W.delta(t)
    z = W.sample_z(n, rng); zq = W.sample_z(n, rng)
    Hp_tr, Hq_tr = W.sample(z, 1.0, rng), W.sample(zq, t, rng)
    rm, dg = R.fit_ratio(Hp_tr, Hq_tr, oh(z, 3), oh(zq, 3), family='mlp', seed=rep)
    z2 = W.sample_z(n, rng); Hp, Hq = W.sample(z2, 1.0, rng), W.sample(z2, t, rng)
    Gp, Gq = W.G(Hp, z2), W.G(Hq, z2)
    r = dict(case='iv_mix', t=t, rep=rep, truth=truth, kl_true=W.kl_pq(t), chi2_true=W.chi2_pq(t),
             auc=dg['auc'], kl_plugin=dg['kl_plugin'], dv=dg['dv'])
    r.update(direct_block(Gq, Gp, truth))
    rt = ratio_block(Gp, W.log_w(Hp, z2, t), Gq, W.log_w(Hq, z2, t), z2, z2, truth)
    r.update({'true_' + k: v for k, v in rt.items()})
    rc = ratio_block(Gp, rm.log_w(Hp, oh(z2, 3)), Gq, rm.log_w(Hq, oh(z2, 3)), z2, z2, truth)
    if refit:
        refit_block(rc, Hp_tr, Hq_tr, oh(z, 3), oh(zq, 3), Gp, Gq, Hp, Hq, oh(z2, 3), oh(z2, 3), z2, z2, truth,
                    'mlp', refit, rep)
    r.update({'mlp_' + k: v for k, v in rc.items()})
    return r


def job_scale(k, rep, n=20000, d=31, clf=False):
    rng = np.random.default_rng(6000 + rep + int(100 * k))
    S = synth.ScaleWorld(d=d, k=k)
    Hp, Hq = S.sample_p(n, rng), S.sample_q(n, rng)
    truth = S.delta()
    r = dict(case='iv_scale', k=k, d=d, rep=rep, truth=truth, chi2_true=S.chi2(), clf=clf)
    r.update(direct_block(S.G(Hq), S.G(Hp), truth))
    if clf:
        rm, dg = R.fit_ratio(S.sample_p(n, rng), S.sample_q(n, rng), family='quad', seed=rep,
                             clf_kw=dict(Cs=(0.1, 1.0), interact=None))
        lwp, lwq = rm.log_w(Hp), rm.log_w(Hq)
        r.update(auc=dg['auc'], khat_q=dg['khat_q'], kl_plugin=dg['kl_plugin'])
    else:
        lwp, lwq = S.log_w(Hp), S.log_w(Hq)
        from cmlib.stats import psis_khat
        r['khat_q'] = psis_khat(lwq)
    r.update(ratio_block(S.G(Hp), lwp, S.G(Hq), lwq, None, None, truth))
    return r


# ============================================================================ driver
def run(job):
    f, args = job
    t0 = time.time()
    res = globals()[f](*args)
    res = res if isinstance(res, list) else [res]
    for r in res:
        r['secs'] = time.time() - t0
    return res


def jobs(quick):
    R1 = 4 if quick else 1
    J = []
    J += [('job_null', ('mix_exact', 'quad', r, 20000, 16 if r < 20 // R1 else 0)) for r in range(40 // R1)]
    J += [('job_null', ('mix_exact', 'mlp', r)) for r in range(16 // R1)]
    J += [('job_null', ('gauss_fitted', 'quad', r)) for r in range(20 // R1)]
    J += [('job_analytic', ('true', r)) for r in range(40 // R1)]
    J += [('job_analytic', ('fitq', r)) for r in range(40 // R1)]
    J += [('job_analytic', (m, r, 20000, 8)) for m in ('mlp', 'quad') for r in range(12 // R1)]
    J += [('job_ignore', (r,)) for r in range(10 // R1)]
    J += [('job_ignore_truth', ())]
    J += [('job_consistency', (cb, r)) for cb in (False, True) for r in range(20 // R1)]
    J += [('job_approach', (t, r, 20000, 6)) for t in (0.0, 0.5, 0.75, 0.9, 0.97, 1.0) for r in range(12 // R1)]
    J += [('job_scale', (k, r)) for k in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5) for r in range(30 // R1)]
    J += [('job_scale', (k, r, 20000, 31, True)) for k in (0.9, 1.05, 1.2) for r in range(4 // R1 or 1)]
    return J


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default=None, help='rerun only jobs with this function name and merge into the JSON')
    a = ap.parse_args()
    J = jobs(a.quick)
    if a.only:
        J = [j for j in J if j[0] == a.only]
    # longest first
    order = {'job_ignore': 0, 'job_approach': 1, 'job_consistency': 2, 'job_null': 3, 'job_analytic': 4}
    J.sort(key=lambda j: order.get(j[0], 9))
    t0 = time.time()
    with Pool(a.procs) as P:
        res = [r for rs in P.imap_unordered(run, J) for r in rs]
    out = os.path.join(HERE, 'results', 'validation_quick.json' if a.quick else 'validation.json')

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    if a.only:
        case = {'job_consistency': 'iii_cons'}.get(a.only)
        old = [r for r in json.load(open(out)) if r['case'] != case]
        res = old + res
    json.dump(clean(res), open(out, 'w'))
    print('wrote', out, len(res), 'rows', '%.0fs' % (time.time() - t0))
