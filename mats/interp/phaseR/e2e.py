"""Task 1 theory check: end-to-end divergences between the chain's q_L and the real p_L, against the sums of per-layer
numbers. usage: python e2e.py d L act g sigma seed  -> results/e2e_<net>.json

log p_L(h) = log E_xi N(h; f_L(h_{L-1}(xi)), s2 I), xi = (x, eps_1..eps_{L-1}) ~ N(0, I): tempered SMC in path space
(Gaussian prior, random-walk MH with the cloud covariance). Per family (G, GC; defensive delta = 0.01), chain fitted as in
score.py; also per-layer KL(Kq_{l-1}||q_l), KL(q_l||Kq_{l-1}) and coverage search at every layer (tempered SMC).
"""
import json, math, os, sys, time
import torch
import rnet as R, fam as F, kq as KQ

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
L2PI = math.log(2 * math.pi)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def hL_from_path(net, XI, L):
    h = XI[:, 0]
    for l in range(1, L):
        h = net.f(l, h) + net.sigma * XI[:, l]
    return net.f(L, h)


def log_p_smc(net, L, H, gen, Np=512, ess_frac=0.7, n_moves=6, max_steps=800, chunk=32):
    s2 = net.sigma ** 2; n, d = H.shape; D = d * L
    const = -0.5 * d * (L2PI + math.log(s2))
    out = torch.empty(n, dtype=DT)
    for a in range(0, n, chunk):
        Hc = H[a:a + chunk]; B = len(Hc)
        X = torch.randn(B, Np, D, generator=gen, dtype=DT)

        def loglik(X):
            r = Hc[:, None, :] - hL_from_path(net, X.reshape(-1, L, d), L).reshape(B, Np, d)
            return -0.5 * (r * r).sum(2) / s2

        ll = loglik(X); lq = -0.5 * (X * X).sum(2)
        t = torch.zeros(B, dtype=DT); logZ = torch.zeros(B, dtype=DT)
        scale = torch.full((B,), 2.38 / math.sqrt(D), dtype=DT)
        for _ in range(max_steps):
            act = t < 1 - 1e-12
            if not act.any():
                break
            lo = torch.zeros(B, dtype=DT); hi = (1 - t).clone()

            def ess(dt):
                w = torch.softmax(dt[:, None] * ll, 1); return 1.0 / (w * w).sum(1)
            full_ok = ess(hi) >= ess_frac * Np
            for _b in range(40):
                mid = 0.5 * (lo + hi); ok = ess(mid) >= ess_frac * Np
                lo = torch.where(ok, mid, lo); hi = torch.where(ok, hi, mid)
            dt = torch.where(full_ok, 1 - t, lo.clamp_min(1e-12)); dt = torch.where(act, dt, torch.zeros_like(dt))
            lw = dt[:, None] * ll
            logZ = logZ + torch.where(act, torch.logsumexp(lw, 1) - math.log(Np), torch.zeros_like(logZ))
            t = t + dt
            idx = torch.multinomial(torch.softmax(lw, 1), Np, replacement=True, generator=gen)
            X = torch.gather(X, 1, idx[:, :, None].expand(-1, -1, D)); ll = torch.gather(ll, 1, idx); lq = torch.gather(lq, 1, idx)
            Xm = X.mean(1, keepdim=True); C = (X - Xm).transpose(1, 2) @ (X - Xm) / Np
            Lc = torch.linalg.cholesky(C + 1e-8 * C.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(D, dtype=DT))
            acc_rate = torch.zeros(B, dtype=DT)
            for _m in range(n_moves):
                Xp = X + scale[:, None, None] * (torch.randn(B, Np, D, generator=gen, dtype=DT) @ Lc.transpose(1, 2))
                lqp = -0.5 * (Xp * Xp).sum(2); llp = loglik(Xp)
                la = (lqp - lq) + t[:, None] * (llp - ll)
                accp = (torch.rand(B, Np, generator=gen, dtype=DT).log() < la) & act[:, None]
                X = torch.where(accp[:, :, None], Xp, X); ll = torch.where(accp, llp, ll); lq = torch.where(accp, lqp, lq)
                acc_rate = acc_rate + accp.to(DT).mean(1) / n_moves
            scale = torch.where(act, scale * torch.exp(acc_rate - 0.25), scale)
        out[a:a + B] = logZ + const
    return out


def run(d, L, act, g, sigma, seed, npts=96, nfit=1 << 15):
    net = R.RNet(d, L, act, g, sigma, seed)
    gen = torch.Generator().manual_seed(900 + seed)
    res = dict(d=d, L=L, act=act, g=g, sigma=sigma, seed=seed, fams={})
    # validation: at depth 1 the path-space estimator must agree with the h'-space one (q_0 = N(0, I))
    H1 = net.f(1, torch.randn(32, d, generator=gen, dtype=DT)) + sigma * torch.randn(32, d, generator=gen, dtype=DT)
    a1 = log_p_smc(net, 1, H1, gen)
    b1, _ = KQ.log_kq_smc(net, 1, F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT)), H1, gen, Np=512, ess_frac=0.7, n_moves=6)
    res['check_depth1'] = dict(mean_diff=float((a1 - b1).mean()), sd_diff=float((a1 - b1).std()))
    print('depth-1 check (nats):', res['check_depth1'], flush=True)
    # real p_L samples and their log densities
    t0 = time.time()
    XI = torch.randn(npts, L, d, generator=gen, dtype=DT)
    Hp = hL_from_path(net, XI, L) + sigma * torch.randn(npts, d, generator=gen, dtype=DT)
    lp_p = log_p_smc(net, L, Hp, gen)
    res['t_logp'] = time.time() - t0
    for fam in ('G', 'GC'):
        q = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT)); per = []
        for l in range(1, L + 1):
            X = q.sample(nfit + npts, gen); S = net.f(l, X); H = S + sigma * torch.randn(S.shape, generator=gen, dtype=DT)
            qn = F.Mix(F.Gauss.fit(H[:nfit]) if fam == 'G' else F.GC.fit_from_s(S[:nfit], H[:nfit], sigma), 0.01)
            Hk = H[nfit:]; Hq = qn.sample(npts, gen)
            if fam == 'G' or l == 1:            # per-layer SMC needs log q_{l-1}: cheap for Gaussian priors only
                lk, _ = KQ.log_kq_smc(net, l, q, Hk, gen, Np=512, ess_frac=0.7, n_moves=6)
                lkq, _ = KQ.log_kq_smc(net, l, q, Hq, gen, Np=2048, ess_frac=0.7, n_moves=6)
                r1 = lk - qn.logpdf(Hk); r2 = qn.logpdf(Hq) - lkq
                per.append(dict(layer=l, KLc=float(r1.mean()) / LOG2, c_pts=float(r1.max()) / LOG2, KLw=float(r2.mean()) / LOG2))
                print(fam, per[-1], flush=True)
            q = qn
        Hq = q.sample(npts, gen)
        lp_q = log_p_smc(net, L, Hq, gen, Np=2048)
        e1 = lp_p - q.logpdf(Hp); e2 = q.logpdf(Hq) - lp_q
        res['fams'][fam] = dict(per_layer=per, KL_p_q=float(e1.mean()) / LOG2, KL_p_q_se=float(e1.std() / math.sqrt(npts)) / LOG2,
                                c_end_pts=float(e1.max()) / LOG2, KL_q_p=float(e2.mean()) / LOG2,
                                KL_q_p_se=float(e2.std() / math.sqrt(npts)) / LOG2,
                                sum_KLc=sum(x['KLc'] for x in per), sum_c_pts=sum(x['c_pts'] for x in per), sum_KLw=sum(x['KLw'] for x in per))
        print(fam, {k: v for k, v in res['fams'][fam].items() if k != 'per_layer'}, flush=True)
        json.dump(res, open(os.path.join(OUT, f'e2e_{net.name()}.json'), 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), int(a[2]), a[3], float(a[4]), float(a[5]), int(a[6]))
