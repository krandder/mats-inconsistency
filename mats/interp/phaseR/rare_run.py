"""Tasks 2-3: rare output events of R(d, L=4, act, g, sigma, seed). usage: python rare_run.py d act g sigma seed [kmax]
Writes results/rare_<net>.json.

For each target rarity 2^-k: threshold t_k (bisection, bulk-twisted SMC); truth = mean of R runs of the lowest-variance
unbiased estimator, cross-checked against the next best; K=2 Gaussian tail; G-chain bounds 2^eps q_L(A) and
2^(eps_{1..L-1}) (K q_{L-1})(A) for eps = SMC-search estimate (lower estimate; d <= 32), rigorous envelope, rigorous
lower bound (relative); sampling's 95% certificate at equal FLOPs; estimator relative RMSE vs FLOPs.
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import beta as Beta
import rnet as R, fam as F, rare as RA, rare_gs as GS

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
L = 4


def est(net, t, method, N, gen, cache):
    """One unbiased estimate (log p, nats) + per-run FLOPs; setup (twists) cached per (method, t)."""
    key = (method, round(t, 10))
    if key not in cache:
        if method == 'mc':
            cache[key] = (None, 0)
        elif method == 'bulk':
            cache[key] = RA.twists_bulk(net, t)
        elif method in ('tail', 'ms'):
            xi, obj, fl = RA.instanton(net, t)
            if method == 'tail':
                tw, f2 = RA.twists_tail(net, t, xi); cache[key] = (tw, fl + f2)
            else:
                cache[key] = (xi, fl)
        elif method == 'iapf':              # black-box: learned from psi = 1, 4 iterations of N particles
            cache[key] = RA.twists_iapf(net, t, N, gen, iters=4)
        elif method == 'iapf_b':            # learned, initialised at the chain's bulk twists
            tw0, f0 = RA.twists_bulk(net, t); tw, f1 = RA.twists_iapf(net, t, N, gen, iters=3, init=tw0)
            cache[key] = (tw, f0 + f1)
    obj, setup = cache[key]
    if method == 'gs':
        lp, _, fl = GS.splitting(net, t, N, gen, levels=obj)
        return lp, fl, setup
    if method == 'ms':
        lp, fl = RA.meanshift_is(net, t, obj, N, gen)
    else:
        lp, fl = RA.smc(net, t, obj, N, gen)
    return lp, fl, setup


def repeat(net, t, method, N, R_, gen, cache):
    v = []; fl = setup = 0
    for _ in range(R_):
        lp, fl, setup = est(net, t, method, N, gen, cache); v.append(lp)
    v = np.array(v); p = np.exp(v - v.max()); m = p.mean()
    return dict(logp=float(np.log(m) + v.max()), rel_sd=float(p.std(ddof=1) / m) if m > 0 else float('inf'),
                logs=v.tolist(), flops_run=fl, flops_setup=setup, N=N)


def find_t(net, k, gen, cache, lo=0.0, hi=None):
    """t with log2 p(t) ~ -k (bulk-twisted SMC, N = 4096, two runs per step)."""
    hi = hi or (math.sqrt(net.d) + 12 * net.sigma)
    target = -k * LOG2
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        r = repeat(net, mid, 'bulk', 4096, 2, gen, cache)
        if r['logp'] > target: lo = mid
        else: hi = mid
        if hi - lo < 1e-3: break
    return 0.5 * (lo + hi)


def g_chain(net, gen, nfit=1 << 15, delta=0.01):
    """Moment-matched Gaussian chain with a defensive component, fitted to samples of K q_{l-1} (as in score.py)."""
    q = F.Gauss(torch.zeros(net.d, dtype=DT), torch.eye(net.d, dtype=DT)); qs = []
    for l in range(1, net.L + 1):
        X = q.sample(nfit, gen); H = net.f(l, X) + net.sigma * torch.randn(nfit, net.d, generator=gen, dtype=DT)
        q = F.Gauss.fit(H); q.delta = delta; q.infl = 4.0; qs.append(q)
    return qs, net.L * nfit * (2 * net.d ** 2 + 12 * net.d)


def kq_event(net, qprev, t, gen, N=1 << 15):
    """(K q_{L-1})(A) = E_{h ~ q_{L-1}} Phi_bar((t - w.f_L(h))/sigma) for the Gaussian (+ defensive) q_{L-1}, by mean-shift
    importance sampling at the most likely h (maximise log q(h) + log Phi_bar(...)). Returns log value (nats)."""
    out = []
    for scale, wt in ((1.0, 1 - qprev.delta), (qprev.infl, qprev.delta)):
        mu = qprev.mu; Lc = qprev.L * math.sqrt(scale)
        z = torch.zeros(net.d, dtype=DT, requires_grad=True)
        opt = torch.optim.LBFGS([z], lr=1, max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad(); h = mu + Lc @ z
            obj = 0.5 * (z * z).sum() - RA.log_sf((t - net.f(net.L, h[None])[0] @ net.w) / net.sigma)
            obj.backward(); return obj
        opt.step(closure)
        zs = z.detach()
        Z = torch.randn(N, net.d, generator=gen, dtype=DT) + zs
        lw = -(Z @ zs) + 0.5 * float(zs @ zs) + RA.log_sf((t - net.f(net.L, mu + Z @ Lc.T) @ net.w) / net.sigma)
        out.append(math.log(wt) + float(torch.logsumexp(lw, 0)) - math.log(N))
    return float(np.logaddexp(out[0], out[1]))


def kq_event_single(net, q, t, gen, N=1 << 15):
    """E_{h ~ N(q.mu, q.S)} Phi_bar((t - w.f_L(h))/sigma) by mean-shift IS at the most likely h (log, nats)."""
    mu = q.mu; Lc = q.L
    z = torch.zeros(net.d, dtype=DT, requires_grad=True)
    opt = torch.optim.LBFGS([z], lr=1, max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(); h = mu + Lc @ z
        obj = 0.5 * (z * z).sum() - RA.log_sf((t - net.f(net.L, h[None])[0] @ net.w) / net.sigma)
        obj.backward(); return obj
    opt.step(closure)
    zs = z.detach()
    Z = torch.randn(N, net.d, generator=gen, dtype=DT) + zs
    lw = -(Z @ zs) + 0.5 * float(zs @ zs) + RA.log_sf((t - net.f(net.L, mu + Z @ Lc.T) @ net.w) / net.sigma)
    return float(torch.logsumexp(lw, 0)) - math.log(N)


def kq_event_gs(net, mu, Lc, t, gen, N=4096, reps=4, n_moves=5):
    """E_{h ~ N(mu, Lc Lc^T)} Phi_bar((t - w.f_L(h))/sigma) = P(w.f_L(mu + Lc z) + sigma e > t), (z, e) ~ N(0, I):
    fixed-level generalized splitting in the Gaussian latent (levels from an adaptive pilot). Returns (log, rel sd)."""
    d = net.d

    def score(Z):
        return net.f(net.L, mu + Z[:, :d] @ Lc.T) @ net.w + net.sigma * Z[:, d]

    def gs(levels=None, rho=0.1, max_levels=40):
        Z = torch.randn(N, d + 1, generator=gen, dtype=DT); Y = score(Z); lp = 0.0; used = []; b = 0.5; k = 0
        while True:
            if levels is None:
                g = float(torch.quantile(Y, 1 - rho))
                if g >= t or k >= max_levels: g = t
            else:
                g = levels[k] if k < len(levels) else t
            above = Y > g; frac = float(above.to(DT).mean())
            if frac == 0: return -float('inf'), used
            lp += math.log(frac); used.append(g)
            if g >= t: return lp, used
            idx = torch.nonzero(above)[:, 0]; pick = idx[torch.randint(len(idx), (N,), generator=gen)]
            Z, Y = Z[pick].clone(), Y[pick].clone()
            for _ in range(n_moves):
                ZP = math.sqrt(1 - b * b) * Z + b * torch.randn(N, d + 1, generator=gen, dtype=DT); YP = score(ZP)
                acc = YP > g; Z = torch.where(acc[:, None], ZP, Z); Y = torch.where(acc, YP, Y)
                b = min(0.95, max(0.02, b * math.exp(float(acc.to(DT).mean()) - 0.3)))
            k += 1
    _, lev = gs()
    lv = [x for x in lev if x < t - 1e-12]
    v = np.array([gs(levels=lv)[0] for _ in range(reps)]); m = v.max(); p = np.exp(v - m)
    return float(np.log(p.mean()) + m), float(p.std(ddof=1) / p.mean())


def kq_event_gs_frozen(net, mu, Lc, ts, gen, N=4096, R=8, n_moves=10):
    """Review fix 2: the same latent probability for every threshold in ts at once, by fixed-level splitting with levels and
    pCN scales frozen from an independent pilot (rare_gs.frozen_estimates). Returns {t: summary dict (logp, rel_se, ...)}."""
    d = net.d
    score = lambda Z: net.f(net.L, mu + Z[:, :d] @ Lc.T) @ net.w + net.sigma * Z[:, d]
    draw = lambda n: torch.randn(n, d + 1, generator=gen, dtype=DT)
    return GS.frozen_estimates(score, draw, list(ts), gen, N, R, n_moves=n_moves)['per_target']


def kq_event_mix_gs_frozen(net, q, ts, gen, **kw):
    """(1 - delta) N(mu, S) + delta N(mu, infl S): both parts by kq_event_gs_frozen. Returns {t: (log, rel_se, parts)}."""
    a = kq_event_gs_frozen(net, q.mu, q.L, ts, gen, **kw)
    if q.delta <= 0:
        return {t: (a[t]['logp'], a[t]['rel_se'], dict(main=a[t])) for t in ts}
    b = kq_event_gs_frozen(net, q.mu, q.L * math.sqrt(q.infl), ts, gen, **kw)
    out = {}
    for t in ts:
        la, lb = math.log(1 - q.delta) + a[t]['logp'], math.log(q.delta) + b[t]['logp']
        tot = float(np.logaddexp(la, lb))
        se = math.sqrt((math.exp(la - tot) * a[t]['rel_se']) ** 2 + (math.exp(lb - tot) * b[t]['rel_se']) ** 2)
        out[t] = (tot, se, dict(main=a[t], defensive=b[t]))
    return out


def kq_event_mix_gs(net, q, t, gen):
    """(1 - delta) N(mu, S) + delta N(mu, infl S) version (G chain with its defensive component)."""
    a, ra = kq_event_gs(net, q.mu, q.L, t, gen)
    if q.delta > 0:
        b, rb = kq_event_gs(net, q.mu, q.L * math.sqrt(q.infl), t, gen)
        return float(np.logaddexp(math.log(1 - q.delta) + a, math.log(q.delta) + b)), ra
    return a, ra


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(Beta.ppf(1 - a, k + 1, n - k))


def load_eps(net):
    """Per-layer coverage numbers for the G family from the scorecard runs of the same network (layers 1..4)."""
    base = f"d{net.d}_L{{}}_{net.act}{net.g:g}_s{net.sigma:g}_n{net.seed}"
    out = {}
    for tag, Lr in (('_smc', 4), ('', 8)):
        p = os.path.join(OUT, f"score_{base.format(Lr)}{tag}.json")
        if os.path.exists(p):
            rows = json.load(open(p))['layers'].get('G', [])[:L]
            for key in ('smc_c', 'env_c', 'rel_Dinf', 'smc_KLc', 'rel_KLc'):
                vals = [r.get(key) for r in rows]
                if len(vals) == L and all(v is not None for v in vals):
                    out.setdefault(key, vals)
    return out


def run(d, act, g, sigma, seed, ks=(10, 20, 30, 40, 50, 60), budgets=None, reps=None, use_ln=True):
    big = d >= 128
    budgets = budgets or ((1024, 4096) if big else (1024, 8192)); reps = reps or (8 if big else 12)
    ntr, rtr = (8192, 6) if big else (16384, 8)
    net = R.RNet(d, L, act, g, sigma, seed, use_ln=use_ln)
    gen = torch.Generator().manual_seed(100 + seed)
    path = os.path.join(OUT, f'rare_{net.name()}.json')
    res = dict(d=d, L=L, act=act, g=g, sigma=sigma, seed=seed, levels=[], flops_sample=net.flops_sample())
    qs, chain_flops = g_chain(net, gen); res['chain_flops'] = chain_flops
    k2c = R.k2_propagate(net, keep=True); qk2 = F.Gauss(k2c[-2][0], k2c[-2][1]); qk2.delta = 0.0; qk2.infl = 4.0
    res['k2chain_flops'] = net.flops_k2() * (net.L - 1) // net.L
    eps = load_eps(net)
    # rigorous envelope bound for this chain's own q_l (defensive part N(mu, 4 Sigma), delta = 0.01)
    env = []
    for q in qs:
        A = q.infl * q.S; ev = torch.linalg.eigvalsh(A); s2 = sigma ** 2
        env.append(float('inf') if float(ev.min()) <= s2 else
                   (math.log(1 / q.delta) + 0.5 * float(torch.log(ev / s2).sum())
                    + 0.5 * (math.sqrt(d) + float(q.mu.norm())) ** 2 / float(ev.min() - s2)) / LOG2)
    eps['env_c_own'] = env
    res['eps_layers'] = eps
    cache = {}
    methods = ('mc', 'bulk', 'tail', 'iapf', 'ms', 'gs')
    # thresholds from one adaptive splitting run: level j has P(y > gamma_j) ~ 10^-j
    tl = time.time()
    levels, fl_lev = GS.level_curve(net, 8192, gen, rho=0.1, n_levels=19)
    res['levels_curve'] = levels; res['levels_secs'] = time.time() - tl
    for k in ks:
        t0 = time.time()
        j = int(round(k * math.log10(2)))
        if j - 1 >= len(levels):
            break
        t = levels[j - 1]
        lev = dict(k=k, t=t)
        lv = [x for x in levels if x < t - 1e-12]
        cache[('gs', round(t, 10))] = (lv, fl_lev)
        # truth: fixed-level generalized splitting (unbiased), cross-checked by bulk-twisted SMC
        tr = repeat(net, t, 'gs', ntr, rtr, gen, cache)
        lev['truth'] = dict(method='gs', logp=tr['logp'], rel_se=tr['rel_sd'] / math.sqrt(rtr), rel_sd_run=tr['rel_sd'], N=ntr, R=rtr)
        cb = repeat(net, t, 'bulk', 4 * ntr, 4, gen, cache)
        lev['crosscheck'] = dict(bulk_logp=cb['logp'], bulk_rel_sd=cb['rel_sd'])
        lp = tr['logp']
        # brute force (plain MC with the last layer Rao-Blackwellised) where affordable
        if k <= 14 or (k <= 20 and d <= 16):
            nb = min(1 << 24, int(2 ** (k + 8)))
            rb = repeat(net, t, 'mc', nb, 1, gen, cache)
            lev['brute'] = dict(logp=rb['logp'], N=nb)
        # K=2 Gaussian tail (ARC-style, deductive)
        lk2, fk2 = RA.k2_tail(net, t); lev['k2'] = dict(logp=lk2, flops=fk2, bits_err=(lk2 - lp) / LOG2)
        # chain bounds
        lqL = qs[-1].prob_halfspace(net.w, t)
        lkq, _rs1 = kq_event_mix_gs(net, qs[-2], t, gen)
        t1 = time.time(); lkq2, _rs2 = kq_event_gs(net, qk2.mu, qk2.L, t, gen); t_kq2 = time.time() - t1
        lev['chain'] = dict(log_qL=lqL, log_KqLm1=lkq, flops=chain_flops, log_KqLm1_k2=lkq2, bits_err_KqLm1_k2=(lkq2 - lp) / LOG2,
                            bits_err_KqLm1=(lkq - lp) / LOG2, bits_err_qL=(lqL - lp) / LOG2, secs_kq2=t_kq2, kq_method='gs',
                            rel_sd_KqLm1=_rs1, rel_sd_KqLm1_k2=_rs2)
        for key in ('smc_c', 'env_c_own', 'rel_Dinf'):
            if key in eps:
                e = eps[key]
                b1 = min(0.0, sum(e) * LOG2 + lqL); b2 = min(0.0, sum(e[:-1]) * LOG2 + lkq)
                lev['chain'][key] = dict(eps_all=sum(e), eps_Lm1=sum(e[:-1]), log_bound=b1, log_bound_lastexact=b2,
                                         bits_lost=(b1 - lp) / LOG2, bits_lost_lastexact=(b2 - lp) / LOG2)
        n_eq = int(chain_flops / net.flops_sample())
        kexp = int(round(math.exp(lp) * n_eq))
        lev['sampling_cert'] = dict(N_eq=n_eq, log_bound=math.log(cp_upper(kexp, n_eq)))
        # estimators: relative RMSE vs FLOPs
        rows = []
        for m in methods:
            for N in budgets:
                r = repeat(net, t, m, N, reps, gen, cache)
                v = np.exp(np.array(r['logs']) - lp)
                rows.append(dict(method=m, N=N, rel_rmse=float(np.sqrt(np.mean((v - 1) ** 2))), rel_bias=float(v.mean() - 1),
                                 flops=r['flops_run'] + r['flops_setup'], flops_run=r['flops_run'], flops_setup=r['flops_setup']))
        lev['estimators'] = rows
        lev['secs'] = time.time() - t0
        res['levels'].append(lev)
        print(f"k={k} t={t:.3f} truth 2^{lp / LOG2:.2f} (gs, rel_se {lev['truth']['rel_se']:.3f}) "
              f"K2 err {lev['k2']['bits_err']:+.1f} bits; " +
              ' '.join(f"{r['method']}@{r['N']}:{r['rel_rmse']:.2f}" for r in rows) + f"  [{lev['secs']:.0f}s]", flush=True)
        json.dump(res, open(path, 'w'), indent=1)
    return res


if __name__ == '__main__':
    a = sys.argv
    skip = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skip_jobs.txt')
    if os.path.exists(skip) and 'rare ' + ' '.join(a[1:6]) in [l.strip() for l in open(skip)]:
        print('skipped by skip_jobs.txt'); sys.exit(0)
    ks = tuple(int(x) for x in a[6].split(',')) if len(a) > 6 else (10, 20, 30, 40, 50, 60)
    run(int(a[1]), a[2], float(a[3]), float(a[4]), int(a[5]), ks, use_ln=not (len(a) > 7 and a[7] == "noLN"))
