"""Task 1: forward chains q_0 -> q_L per family, scorecard per layer (bits).

usage: python score.py d L act g sigma seed [smc_layers] [fams]
  writes results/score_<net>.json

Per layer l and family F (q_l^F fitted to samples of K q_{l-1}^F, with a delta = 0.01 defensive component N(mu, 4 Sigma)):
  rel_KLc  = E_{Kq}[log q* - log q]            sample mean of a quantity whose expectation lower-bounds KL(Kq || q)
                                                 (q* = an independently fitted copula code, any normalised density works)
  rel_Dinf = max_tau tau + log Kq(T > tau)       in-sample value (tau chosen on the same samples, normal approximation):
                                                 an estimate of a lower bound on D_inf(Kq || q), T = log q*/q (Markov)
  rel_cb   = review fix 1: bootstrap 95% lower bound on E T, and a held-out-tau Clopper-Pearson 95% lower bound on D_inf
             (rel_stats); these are the numbers reported as lower confidence bounds
  rel_KLw  = max_beta beta E_q[log q/q*] - log E_Kq[(q/q*)^beta]   DV lower bound on the waste KL(q || Kq)
  smc_KLc  = E_{Kq}[log Kq - log q]  with log Kq(h) by tempered SMC (accurate on Kq-typical points)       [small d]
  smc_KLw  = E_q[log q - log Kq]      (log Kq at q-points biased low -> an over-estimate; Np-doubling check)    [small d]
  smc_c    = max over Kq points (+ refinement) of log Kq - log q: a lower estimate of the coverage cost        [small d]
  dec_*    = decoupled per-neuron predictions: sums over neurons of 1-D divergences between the exact marginal of Kq
             and the family's marginal
  env_c    = rigorous upper bound on D_inf(Kq || q) for the defensive Gaussian part alone (envelope, any q_{l-1})
"""
import json, math, os, sys, time
import numpy as np
import torch
import rnet as R, fam as F, kq as KQ

torch.set_num_threads(1)
LOG2 = math.log(2)
DT = torch.float64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'); os.makedirs(OUT, exist_ok=True)
DELTA = 0.01


def logpdf_chunked(q, H, chunk=4096):
    return torch.cat([q.logpdf(H[a:a + chunk]) for a in range(0, len(H), chunk)])


def GC_M(d):
    return 32 if d <= 64 else 16


def fit(fam, S, H, sigma, d):
    if fam == 'G':
        q = F.Gauss.fit(H)
    elif fam == 'NC':
        q = F.NC.fit_from_s(S, sigma, M=GC_M(d))
    elif fam == 'GC':
        q = F.GC.fit_from_s(S, H, sigma, M=GC_M(d))
    elif fam == 'KN':
        return F.KN(d, sigma)
    else:
        raise ValueError(fam)
    return F.Mix(q, DELTA)


def marg_1d(S, sigma, grid):
    """Exact 1-D marginal of h_i = s_i + sigma eps for every neuron from samples S (N, d), on grid (G,). Returns (d, G) log."""
    N, d = S.shape; out = []
    for i in range(d):
        x = S[:, i]
        nb = int(min(800, max(100, 30 * float(x.max() - x.min()) / sigma)))
        cnt = torch.histc(x, bins=nb, min=float(x.min()), max=float(x.max()))
        edges = torch.linspace(float(x.min()), float(x.max()), nb + 1, dtype=DT); cen = 0.5 * (edges[1:] + edges[:-1])
        k = cnt > 0; lw = torch.log(cnt[k] / N)
        lr = torch.logsumexp(lw[None] - (grid[:, None] - cen[k][None]) ** 2 / (2 * sigma ** 2), 1) - 0.5 * math.log(2 * math.pi * sigma ** 2)
        out.append(lr)
    return torch.stack(out)


def family_marg(q, i, grid):
    base = q.q if isinstance(q, F.Mix) else q
    lp = base.marg_logpdf(i, grid)
    if isinstance(q, F.Mix):
        mu, S = q.D.mu, q.D.S
        ld = -0.5 * (grid - mu[i]) ** 2 / S[i, i] - 0.5 * (math.log(2 * math.pi) + torch.log(S[i, i]))
        lp = torch.logaddexp(math.log(1 - q.delta) + lp, math.log(q.delta) + ld)
    return lp


def decoupled(q, S, sigma):
    """Sums over neurons of KL(q_i||r_i), KL(r_i||q_i), D_inf(r_i||q_i) (bits)."""
    if isinstance(q, F.KN):
        return None
    grid = torch.linspace(float(S.min()) - 8 * sigma - 3, float(S.max()) + 8 * sigma + 3, 3001, dtype=DT); dg = float(grid[1] - grid[0])
    LR = marg_1d(S, sigma, grid)
    kw = kc = di = 0.0
    for i in range(S.shape[1]):
        lq = family_marg(q, i, grid); lr = LR[i]
        pq, pr = lq.exp(), lr.exp()
        kw += float((pq * (lq - lr)).sum()) * dg; kc += float((pr * (lr - lq)).sum()) * dg
        di += float((lr - lq)[pr > 1e-12].max())
    return dict(dec_KLw=kw / LOG2, dec_KLc=kc / LOG2, dec_Dinf=di / LOG2)


def envelope_c(qmix, d, sigma):
    """Rigorous bound: D_inf(Kq || (1-delta) q~ + delta N(mu, 4 S)) <= log(1/delta) + sup_{|s|=sqrt d} sup_h log N(h;s,s2)/N(h;mu,4S).
    sup_h log N(h;s,s2 I)/N(h;mu,A) = 1/2 log det(A/s2) + 1/2 (s-mu)^T (A - s2 I)^{-1} (s-mu)  (A > s2 I); sup over the sphere
    bounded by (sqrt d + |mu|)^2 / (2 lambda_min(A - s2 I))."""
    A = qmix.D.S; mu = qmix.D.mu; s2 = sigma ** 2
    ev = torch.linalg.eigvalsh(A)
    if float(ev.min()) <= s2:
        return float('inf')
    ld = float(torch.log(ev / s2).sum())
    quad = (math.sqrt(d) + float(mu.norm())) ** 2 / float(ev.min() - s2)
    return (math.log(1 / DELTA) + 0.5 * ld + 0.5 * quad) / LOG2


def cp_lower(k, n, alpha=0.05):
    """Exact one-sided (1 - alpha) Clopper-Pearson lower bound on a binomial probability (k hits of n)."""
    from scipy.stats import beta as _B
    return 0.0 if k <= 0 else float(_B.ppf(alpha, k, n - k + 1))


def rel_stats(T, seed=0, B=2000, alpha=0.05):
    """Review fix 1 (2026-10-29). T = log q*(h) - log q(h) at iid h ~ Kq (nats), q* an independently fitted normalised density.
    KL(Kq||q) >= E T: sample mean, path-level SE, and the one-sided (1 - alpha) bootstrap lower bound (B resamples of the
    sampled paths; approximate). D_inf(Kq||q) >= tau + log Kq(T >= tau) for every tau (Markov under q, E_q[q*/q] = 1): tau
    chosen on the first half of the samples, Kq(T >= tau) bounded below on the second half by the exact Clopper-Pearson
    bound, so the result holds with probability >= 1 - alpha (store tau, k, n to re-evaluate at other alpha, e.g. Bonferroni).
    Also the old in-sample value (max over sample-chosen tau, normal approximation; not a confidence bound). Bits."""
    T = T.detach().cpu().numpy().astype(float) if hasattr(T, 'detach') else np.asarray(T, dtype=float)
    n = len(T); rng = np.random.default_rng(12345 + seed)
    mean = float(T.mean()); se = float(T.std(ddof=1) / math.sqrt(n))
    bm = []
    for _ in range(B // 100):
        idx = rng.integers(0, n, (100, n)); bm.append(T[idx].mean(1))
    bm = np.concatenate(bm)
    kl_lcb = float(np.quantile(bm, alpha))
    kl_q = {f'{a:.5f}': float(np.quantile(bm, a)) / LOG2 for a in (0.05, 0.025, 0.05 / 3, 0.0125, 0.01, 0.05 / 6, 0.005)}
    # old in-sample D_inf value (as published before the review)
    Ts = np.sort(T)[::-1]; old = -1e9
    for k in (10, 30, 100, 300, 1000, 3000, 10000, n // 4, n // 2):
        if 10 <= k <= n:
            old = max(old, Ts[k - 1] + math.log(max((k - 1.645 * math.sqrt(k)) / n, 1e-300)))
    # held-out D_inf bound
    h = n // 2; Tsel, Tev = T[:h], T[h:]; ne = len(Tev)
    Ss = np.sort(Tsel)[::-1]; best = (-1e18, None)
    for k in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, h // 4, h // 2):
        if 1 <= k <= h:
            # score each candidate with a pessimistic eval-half count (10% Poisson quantile of k), so that thresholds
            # with few exceedances, whose held-out count may be 0, are not preferred (rule fixed 2026-10-29 00:41Z,
            # before any score_cb run; see PREREG-R amendment 2, note)
            tau = float(Ss[k - 1]); kp = int(math.floor(max(0.0, k - 1.2816 * math.sqrt(k))))
            if kp < 1: continue
            val = tau + math.log(max(cp_lower(kp, ne, alpha), 1e-300))
            if val > best[0]: best = (val, tau)
    tau = best[1]; ke = int((Tev >= tau).sum())
    pl = cp_lower(ke, ne, alpha)
    dinf = max(0.0, tau + math.log(pl)) if pl > 0 else 0.0
    return dict(KL_mean=mean / LOG2, KL_se=se / LOG2, KL_lcb=kl_lcb / LOG2, KL_lcb_q=kl_q, Dinf_lcb=dinf / LOG2, Dinf_tau=tau, Dinf_k=ke,
                Dinf_n=ne, Dinf_insample_old=old / LOG2, n=n, B=B, alpha=alpha)


def dinf_lcb_bits(s, alpha):
    """Re-evaluate a stored held-out D_inf bound at another level alpha (e.g. alpha / L for a Bonferroni sum)."""
    pl = cp_lower(s['Dinf_k'], s['Dinf_n'], alpha)
    return max(0.0, (s['Dinf_tau'] + math.log(pl)) / LOG2) if pl > 0 else 0.0


def rel_bounds(Hk_lq, Hk_lqs, Hq_lq, Hq_lqs):
    """log q and log q* at Kq samples (Hk_*) and at q samples (Hq_*). Original (pre-review) point values; since review
    fix 1 the confidence-bound versions from rel_stats() are added under rel_cb."""
    T = Hk_lqs - Hk_lq                      # log q*/q at Kq samples
    klc = float(T.mean()); se = float(T.std() / math.sqrt(len(T)))
    Ts, _ = torch.sort(T, descending=True); n = len(T)
    best = -1e9
    for k in (10, 30, 100, 300, 1000, 3000, 10000, n // 4, n // 2):
        if k < 10 or k > n: continue
        tau = float(Ts[k - 1])
        # 95% lower confidence for Kq(T >= tau) with k hits out of n (Wilson-ish via normal approx on log)
        p_lo = max((k - 1.645 * math.sqrt(k)) / n, 1e-300)
        best = max(best, tau + math.log(p_lo))
    Tq = Hq_lq - Hq_lqs                     # log q/q* at q samples; DV for KL(q||Kq) with test beta*Tq
    Tk = Hk_lq - Hk_lqs                     # same at Kq samples
    klw = -1e9
    for beta in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        val = beta * float(Tq.mean()) - (float(torch.logsumexp(beta * Tk, 0)) - math.log(len(Tk)))
        klw = max(klw, val)
    return dict(rel_KLc=klc / LOG2, rel_KLc_se=se / LOG2, rel_Dinf=best / LOG2, rel_KLw=klw / LOG2, rel_cb=rel_stats(T))


def smc_scores(net, l, qprev, q, Hk, Hq, gen, Np=512, npts=96):
    t0 = time.time()
    lk, _ = KQ.log_kq_smc(net, l, qprev, Hk[:npts], gen, Np=Np, ess_frac=0.7, n_moves=6)
    lqk = q.logpdf(Hk[:npts])
    r = lk - lqk
    klc = float(r.mean()); klc_se = float(r.std() / math.sqrt(npts))
    lq_, _ = KQ.log_kq_smc(net, l, qprev, Hq[:npts], gen, Np=Np, ess_frac=0.7, n_moves=6)
    lq2, _ = KQ.log_kq_smc(net, l, qprev, Hq[:npts], gen, Np=4 * Np, ess_frac=0.7, n_moves=6)
    lqq = q.logpdf(Hq[:npts])
    klw = float((lqq - lq2).mean()); klw_np = float((lqq - lq_).mean())
    # coverage: best Kq points, re-evaluated with more particles
    top = torch.argsort(r, descending=True)[:8]
    lk2, _ = KQ.log_kq_smc(net, l, qprev, Hk[top], gen, Np=4 * Np, ess_frac=0.8, n_moves=8)
    c = float((lk2 - lqk[top]).max())
    return dict(smc_KLc=klc / LOG2, smc_KLc_se=klc_se / LOG2, smc_KLw=klw / LOG2, smc_KLw_quarterNp=klw_np / LOG2,
                smc_KLw_se=float((lqq - lq2).std() / math.sqrt(npts)) / LOG2, smc_c=c / LOG2, smc_s=time.time() - t0)


def run(d, L, act, g, sigma, seed, smc_layers=(), fams=('G', 'NC', 'GC', 'KN'), nfit=1 << 15, neval=1 << 14,
        smc_fams=('G', 'GC', 'KN'), tag=''):
    net = R.RNet(d, L, act, g, sigma, seed)
    gen = torch.Generator().manual_seed(7 + seed)
    if d >= 128:
        neval = 1 << 13
    path = os.path.join(OUT, f'score_{net.name()}{tag}.json')
    res = dict(d=d, L=L, act=act, g=g, sigma=sigma, seed=seed, nfit=nfit, neval=neval, delta=DELTA,
               C_bits=0.5 * (d - 1) * math.log2(1 + d / ((d - 1) * sigma ** 2)), layers={f: [] for f in fams})
    qs = {f: F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT)) for f in fams}   # q_0 = input law exactly
    for l in range(1, L + 1):
        new = {}; samp = {}
        for f in fams:
            t0 = time.time()
            Xp = qs[f].sample(nfit + neval, gen); S = net.f(l, Xp); Hk = S + sigma * torch.randn(S.shape, generator=gen, dtype=DT)
            q = fit(f, S[:nfit], Hk[:nfit], sigma, d)
            new[f] = q; samp[f] = (Xp[nfit:], S[nfit:], Hk[nfit:], time.time() - t0, S[:nfit])
        for f in fams:
            q = new[f]; Xe, Se, Hk, tfit, Sfit = samp[f]
            t0 = time.time()
            row = dict(layer=l, t_fit=tfit, flops_push=nfit * (2 * d * d + 12 * d))
            Hq = q.sample(neval, gen)
            lq_k = logpdf_chunked(q, Hk); lq_q = logpdf_chunked(q, Hq)
            # q* = best available density for this Kq: refit a GC (with defensive part) on fresh Kq samples of this chain
            if f != 'GC':
                Xs = qs[f].sample(nfit, gen); Ss = net.f(l, Xs); Hs = Ss + sigma * torch.randn(Ss.shape, generator=gen, dtype=DT)
                qstar = F.Mix(F.GC.fit_from_s(Ss, Hs, sigma, M=GC_M(d)), DELTA)
                ls_k = logpdf_chunked(qstar, Hk); ls_q = logpdf_chunked(qstar, Hq)
                row.update(rel_bounds(lq_k, ls_k, lq_q, ls_q))
            dec = decoupled(q, Sfit[:8192], sigma)
            if dec: row.update(dec)
            if isinstance(q, F.Mix) and isinstance(q.q, F.Gauss):
                row['env_c'] = envelope_c(q, d, sigma)
            prior_cheap = (l == 1) or (f in ('G', 'KN')) or (d <= 8 and 'gcdeep' in tag)
            if l in smc_layers and f in smc_fams and prior_cheap:
                row.update(smc_scores(net, l, qs[f], q, Hk, Hq, gen))
            row['t_score'] = time.time() - t0
            res['layers'][f].append(row)
            print(f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        qs = new
        json.dump(res, open(path, 'w'), indent=1)
    return res


if __name__ == '__main__':
    a = sys.argv
    skip = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skip_jobs.txt')
    if os.path.exists(skip) and ' '.join(a[1:5]) in [l.strip() for l in open(skip)]:
        print('skipped by skip_jobs.txt'); sys.exit(0)
    smc = tuple(int(x) for x in a[7].split(',')) if len(a) > 7 and a[7] not in ('', '-') else ()
    fams = tuple(a[8].split(',')) if len(a) > 8 and a[8] != '-' else ('G', 'NC', 'GC', 'KN')
    smcf = tuple(a[9].split(',')) if len(a) > 9 and a[9] != '-' else ('G', 'GC', 'KN')
    tag = a[10] if len(a) > 10 else ''
    run(int(a[1]), int(a[2]), a[3], float(a[4]), float(a[5]), int(a[6]), smc, fams, smc_fams=smcf, tag=tag)
