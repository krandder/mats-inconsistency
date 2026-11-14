"""Surrogate chains on random noisy LN-MLPs (PREREG-R amendment 1).
usage: python surr.py d L act g bias seed  -> results/surr_<net>.json

Real path h_l = f_l(h_{l-1}) + s e_l; surrogate path h_l = g_l(h_{l-1}) + s e_l (same noise). Chain rule:
KL(real path || surrogate path) = sum_l E_real |f_l(h_{l-1}) - g_l(h_{l-1})|^2 / (2 s^2)  (nats), by sampling real paths.
Surrogates g_l: STAT (K=2's implicit statistical linearisation: K=2 mean of s_l + E[ds/dh](h - mu_{l-1})), LR (rank d/8
truncation of STAT's matrix), QUAD (per-neuron degree-2 Hermite activation + mean-field LN), PL (hardtanh / exact ReLU +
mean-field LN), MFLN (exact activation + mean-field LN).
(1) per-layer KL per surrogate; (2) Donsker-Varadhan two-sided bounds on E[y_j] and E[y_j^2/2] from STAT (Gaussian path:
exact mgf) vs plain sampling's 95% CI at equal FLOPs; (3) E[y] estimators: plain sampling, K=2, SUR = exact surrogate mean
+ sampled correction y_real - y_sur on coupled paths (STAT and LR): variance ratio rho and normalised MSE vs FLOPs.
"""
import json, math, os, sys, time
import numpy as np
import torch
import rnet as R

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
J_OUT = 8                      # outputs used for the E[y] metrics


def layer_stats(net, l, mu, S, Kh=20, ngh=48):
    """Under h_{l-1} ~ N(mu, S): K=2 mean of s_l (with the LN mean correction), Stein matrix A, pre-activation moments,
    Hermite coefficients up to order 2, mean-field LN scale nu."""
    d = net.d; W = net.W[l - 1]; b = net.b[l - 1]
    m = W @ mu + b; C = W @ S @ W.T
    Ea, Ca = R.act_moments_k2(net.phi, m, C, Kh, ngh)
    P = torch.eye(d, dtype=DT) - 1.0 / d
    mc = P @ Ea; Cc = P @ Ca @ P; n0 = (Cc.trace() + mc @ mc) / d
    covN = 2.0 * (Cc @ mc) / d; varN = (2.0 * (Cc * Cc).sum() + 4.0 * mc @ Cc @ mc) / d ** 2
    ms = (mc - covN / (2 * n0) + 3.0 * mc * varN / (8 * n0 ** 2)) / n0.sqrt()
    A = R.stein_jacobian(net, l, mu, S, Kh, ngh)
    x, wq = R.gh(ngh); sd = C.diagonal().clamp_min(1e-300).sqrt()
    Z = m[:, None] + sd[:, None] * x[None]; Ph = net.phi(Z)
    He = R.hermite_table(x, 2)
    c = (Ph * wq[None]) @ He.T                                   # (d, 3): E[phi He_k]
    return dict(ms=ms, A=A, m=m, sd=sd, c=c, nu=n0.sqrt())


def path_total_stats(tot, seed=0, B=2000, alpha=0.05):
    """Per-path whole-path KL values (nats) -> mean, path-level SE and the one-sided (1 - alpha) bootstrap upper bound (bits)."""
    n = len(tot); rng = np.random.default_rng(4242 + seed)
    bm = np.concatenate([tot[rng.integers(0, n, (100, n))].mean(1) for _ in range(B // 100)])
    return dict(mean=float(tot.mean()) / LOG2, se_path=float(tot.std(ddof=1) / math.sqrt(n)) / LOG2,
                ucb=float(np.quantile(bm, 1 - alpha)) / LOG2, B=B, alpha=alpha)


class Surrogates:
    def __init__(self, net, chain):
        self.net = net; d = net.d; self.st = []; self.mus = [c[0] for c in chain]; self.k = max(1, d // 8)
        self.P = torch.eye(d, dtype=DT) - 1.0 / d
        for l in range(1, net.L + 1):
            st = layer_stats(net, l, chain[l - 1][0], chain[l - 1][1])
            U, s, Vh = torch.linalg.svd(st['A'])
            st['Ak'] = (U[:, :self.k] * s[:self.k]) @ Vh[:self.k]
            self.st.append(st)

    def g(self, name, l, H):
        st = self.st[l - 1]; net = self.net; mu = self.mus[l - 1]
        if name == 'STAT':
            return st['ms'] + (H - mu) @ st['A'].T
        if name == 'LR':
            return st['ms'] + (H - mu) @ st['Ak'].T
        Z = H @ net.W[l - 1].T + net.b[l - 1]
        if name == 'QUAD':
            u = (Z - st['m']) / st['sd']; c = st['c']
            a = c[:, 0] + c[:, 1] * u + c[:, 2] * (u * u - 1) / 2
        elif name == 'PL':
            a = Z.clamp(-1, 1) if net.act == 'tanh' else torch.relu(Z)
        elif name == 'MFLN':
            a = net.phi(Z)
        else:
            raise ValueError(name)
        return (a - a.mean(-1, keepdim=True)) / st['nu']


def run(d, L, act, g, bias, seed, sigma=0.3, n_kl=1 << 12, n_cv=1 << 16, n_tr=1 << 17):
    net = R.RNet(d, L, act, g, sigma, seed, bias=bias)
    gen = torch.Generator().manual_seed(2000 + seed)
    t0 = time.time()
    chain = [(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT))] + R.k2_propagate(net, keep=True, psd=True)
    sur = Surrogates(net, chain)
    res = dict(d=d, L=L, act=act, g=g, bias=bias, seed=seed, sigma=sigma, k_lowrank=sur.k, flops_sample=net.flops_sample())
    names = ('STAT', 'LR', 'QUAD', 'PL', 'MFLN')
    # (1) per-layer KL on real paths
    X = torch.randn(n_kl, d, generator=gen, dtype=DT); h = X; kl = {n: [] for n in names}
    tot = {n: torch.zeros(n_kl, dtype=DT) for n in names}          # per-path total over layers (review fix 5)
    for l in range(1, L + 1):
        f = net.f(l, h)
        for n in names:
            e = ((f - sur.g(n, l, h)) ** 2).sum(1) / (2 * sigma ** 2)
            kl[n].append(dict(mean=float(e.mean()) / LOG2, se=float(e.std() / math.sqrt(n_kl)) / LOG2))
            tot[n] += e
        h = f + sigma * torch.randn(n_kl, d, generator=gen, dtype=DT)
    res['kl_bits'] = kl
    res['kl_total_bits'] = {n: path_total_stats(tot[n].numpy(), seed=17 + i) for i, n in enumerate(names)}
    # surrogate (STAT) path moments: mean = K=2 means, covariance Sigma^Q_l = A Sigma^Q A^T + s2 I
    SQ = torch.eye(d, dtype=DT)
    for l in range(1, L + 1):
        A = sur.st[l - 1]['A']; SQ = A @ SQ @ A.T + sigma ** 2 * torch.eye(d, dtype=DT)
    Wo = net.Wout[:J_OUT]; mQ = Wo @ chain[-1][0]; vQ = torch.einsum('jd,de,je->j', Wo, SQ, Wo)
    # (3) coupled real / surrogate paths (STAT and LR) for SUR, plus truth
    def coupled(n):
        Xc = torch.randn(n, d, generator=gen, dtype=DT); hp = Xc; hs = Xc.clone(); hl = Xc.clone()
        for l in range(1, L + 1):
            E = sigma * torch.randn(n, d, generator=gen, dtype=DT)
            hp = net.f(l, hp) + E; hs = sur.g('STAT', l, hs) + E; hl = sur.g('LR', l, hl) + E
        return hp @ Wo.T, hs @ Wo.T, hl @ Wo.T
    yp, ys, yl = [], [], []
    for a in range(0, n_tr, 1 << 14):
        p_, s_, l_ = coupled(min(1 << 14, n_tr - a)); yp.append(p_); ys.append(s_); yl.append(l_)
    yp = torch.cat(yp); ys = torch.cat(ys); yl = torch.cat(yl)
    var1 = yp.var(0)
    truth_sur = mQ + (yp - ys).mean(0)                                    # unbiased, low variance
    truth_mc = yp.mean(0)
    rho = ((yp - ys).var(0) / var1); rho_lr = ((yp - yl).var(0) / var1)
    k2_nmse = float((((mQ - truth_sur) ** 2) / var1).mean())
    res['eout'] = dict(rho_stat=float(rho.mean()), rho_lr=float(rho_lr.mean()), k2_nmse=k2_nmse,
                       truth_check_nmse=float((((truth_mc - truth_sur) ** 2) / var1).mean()), n_truth=n_tr,
                       truth_floor=float((rho / n_tr).mean()))
    fl_s = net.flops_sample(); fl_k2 = net.flops_k2() + L * (4 * d ** 3)          # K=2 + Stein matrices
    fl_sur_per = fl_s + L * (2 * d * d + 2 * d) + 2 * J_OUT * d                      # real + linear surrogate per coupled sample
    rows = []
    for N in (1 << 6, 1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16):
        fl = fl_k2 + N * fl_sur_per
        rows.append(dict(N=N, flops=fl, nmse_sur=float(rho.mean()) / N, nmse_mc_equal=fl_s / fl, nmse_k2=k2_nmse))
    res['eout']['curve'] = rows
    res['eout']['flops_k2'] = fl_k2
    # (2) DV bounds from STAT: F = y_j and F = y_j^2 / 2
    # whole-path KL and its upper confidence bound: path-level bootstrap of the per-path sum over layers (review fix 5;
    # the layers share paths, so their standard errors do not add in quadrature as was assumed before)
    KLn = sum(x['mean'] for x in kl['STAT']) * LOG2
    KLse_indep_old = math.sqrt(sum(x['se'] ** 2 for x in kl['STAT'])) * LOG2
    KLu = res['kl_total_bits']['STAT']['ucb'] * LOG2; KLu_old = KLn + 2 * KLse_indep_old
    tF2 = (yp ** 2 / 2).mean(0); sdF1 = yp.std(0); sdF2 = (yp ** 2 / 2).std(0)
    fl_dv = n_kl * (fl_s + L * 2 * d * d) + L * 4 * d ** 3
    n_eq = fl_dv / fl_s
    dv = []
    for j in range(J_OUT):
        m, v = float(mQ[j]), float(vQ[j])
        hw1 = math.sqrt(2 * KLu * v)
        lams = np.linspace(1e-4, 1 / v * 0.999, 2000)
        up = np.min((KLu - 0.5 * np.log(1 - lams * v) + lams * m * m / (2 * (1 - lams * v))) / lams)
        lams2 = np.geomspace(1e-4, 1e4, 4000)
        lo = np.max(-(KLu - 0.5 * np.log(1 + lams2 * v) - lams2 * m * m / (2 * (1 + lams2 * v))) / lams2)
        ci1 = 1.96 * float(sdF1[j]) / math.sqrt(n_eq); ci2 = 1.96 * float(sdF2[j]) / math.sqrt(n_eq)
        dv.append(dict(j=j, truth_y=float(truth_sur[j]), dv_y=(m - hw1, m + hw1), width_ratio_y=hw1 / ci1,
                       truth_F2=float(tF2[j]), dv_F2=(float(lo), float(up)), width_ratio_F2=(float(up) - float(lo)) / (2 * ci2)))
    res['dv'] = dict(KL_nats=KLn, KL_upper_nats=KLu, KL_upper_nats_old_indep2se=KLu_old, KL_upper_level='bootstrap one-sided 95%',
                     flops=fl_dv, n_eq=n_eq, per_output=dv,
                     width_ratio_y_median=float(np.median([x['width_ratio_y'] for x in dv])),
                     width_ratio_F2_median=float(np.median([x['width_ratio_F2'] for x in dv])),
                     contains_truth=all(x['dv_y'][0] <= x['truth_y'] <= x['dv_y'][1] and x['dv_F2'][0] <= x['truth_F2'] <= x['dv_F2'][1] for x in dv))
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f'surr_{net.name()}.json'), 'w'), indent=1)
    print(net.name(), 'KL bits/layer (mean over layers):', {n: round(float(np.mean([x['mean'] for x in kl[n]])), 2) for n in names},
          f"rho STAT {res['eout']['rho_stat']:.3f} LR {res['eout']['rho_lr']:.3f}, K2 nMSE {k2_nmse:.2e},"
          f" DV width / CI (y, y^2/2): {res['dv']['width_ratio_y_median']:.1f}, {res['dv']['width_ratio_F2_median']:.1f}"
          f" [{res['secs']:.0f}s]", flush=True)


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), int(a[2]), a[3], float(a[4]), float(a[5]), int(a[6]))
