"""Part B: estimators of E[a] (a = relu(W_1 h_0), h_0 ~ the stationary sample measure of Model S, sigma = 1) and of a
tail probability, as compute -> infinity. Cost unit: one n x n matrix-vector product ("mv").
usage: partB.py MODE n seed [--M ..] [--out file]
MODE:
  truth : plain MC, burn-in 8n from h = 0, M samples; mean, per-coordinate variance, tail thresholds (1e-2, 1e-3 of |a|
          from the first 1/4 of samples) and tail probabilities on the rest.
  mlmc  : burn-in multilevel (levels T_l = 2^l from the fixed start h = 0, coarse and fine share the noise of the last
          T_{l-1} layers): per-level variance V_l, mean m_l, cost C_l; ideal MLMC and Glynn-Rhee variance x cost; bias of
          plain MC as a function of the burn-in T.
  cv    : Gaussian-surrogate control variate. K2-Stein closure (m_l, C_l, gate probabilities p_l) along the window;
          surrogate delta_l = diag(p_l) W_l delta_{l-1} + xi_l driven by the SAME noise as each true path; E[a^G] in closed
          form; variance reduction of a - beta a^G (beta per coordinate); also the closure's own estimate E_K2[a].
  rqmc  : scrambled Sobol on the noise of the most recent d layers (d n <= 21201), MC on older layers; R replicates
          at N = 2^k; variance against MC's tr Var(a)/N.
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.special import ndtr, ndtri
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--M', type=int, default=4096); ap.add_argument('--B', type=int, default=0)
ap.add_argument('--R', type=int, default=16); ap.add_argument('--kmax', type=int, default=12)
ap.add_argument('--truth', default=None); ap.add_argument('--out', default=None)
ap.add_argument('--ptail', default='0.01'); ap.add_argument('--scales', default='1.5,2.5')
a = ap.parse_args()
n, seed, M = a.n, a.seed, a.M
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=400e6)
res = dict(mode=a.mode, n=n, seed=seed, M=M)
CH = 4096 if n <= 64 else 1024
INV2PI = 1.0 / math.sqrt(2 * math.pi)


def relu_moments(mu, s):
    s = np.maximum(s, 1e-300)
    t = mu / s
    Phi = ndtr(t); phi = np.exp(-0.5 * t * t) * INV2PI
    m1 = mu * Phi + s * phi
    m2 = (mu * mu + s * s) * Phi + mu * s * phi
    return m1, m2 - m1 * m1, Phi


def run_paths(Pn, B, rng, h=None):
    h = np.zeros((Pn, n)) if h is None else h
    for l in range(-B + 1, 1):
        h = np.maximum(h @ Wf(l).T, 0.0) + rng.standard_normal((Pn, n))
    return h


def out_a(h):
    return np.maximum(h @ Wf(1).T, 0.0)


if a.mode == 'truth':
    B = a.B or 8 * n
    rng = np.random.default_rng([5150, seed, n])
    s1 = np.zeros(n); s2 = np.zeros(n); norms = []
    done = 0
    while done < M:
        k = min(CH, M - done)
        A = out_a(run_paths(k, B, rng))
        s1 += A.sum(0); s2 += (A * A).sum(0); norms.append(np.linalg.norm(A, axis=1)); done += k
    mean = s1 / M; var = s2 / M - mean ** 2
    nr = np.concatenate(norms); q = len(nr) // 4
    tails = {}
    for pt in (1e-2, 1e-3):
        t = float(np.quantile(nr[:q], 1 - pt)); p = float(np.mean(nr[q:] > t))
        tails[str(pt)] = dict(t=t, p=p, se=math.sqrt(max(p * (1 - p), 1e-300) / (len(nr) - q)))
    res.update(B=B, mean=mean.tolist(), var=var.tolist(), trvar=float(var.sum()), nmse_truth=1.0 / M, tails=tails,
               mean2_over_trvar=float(mean @ mean / var.sum()))

elif a.mode == 'mlmc':
    Lmax = int(math.ceil(math.log2(16 * n)))
    rng = np.random.default_rng([6160, seed, n])
    levels = []
    for lev in range(0, Lmax + 1):
        Tf = 2 ** lev; Tc = 2 ** (lev - 1) if lev > 0 else 0
        Ys = []; Af = []
        done = 0
        while done < M:
            k = min(CH, M - done)
            hf = np.zeros((k, n))
            for l in range(-Tf + 1, -Tc + 1):     # fine-only layers
                hf = np.maximum(hf @ Wf(l).T, 0.0) + rng.standard_normal((k, n))
            hc = np.zeros((k, n))
            for l in range(-Tc + 1, 1):           # shared layers
                xi = rng.standard_normal((k, n)); W = Wf(l)
                hf = np.maximum(hf @ W.T, 0.0) + xi
                hc = np.maximum(hc @ W.T, 0.0) + xi
            af = out_a(hf)
            Y = af - out_a(hc) if lev > 0 else af
            Ys.append(Y); Af.append(af); done += k
        Y = np.concatenate(Ys); Afi = np.concatenate(Af)
        levels.append(dict(level=lev, Tf=Tf, Tc=Tc, V=float(np.sum(Y.var(0))), mean=Y.mean(0).tolist(),
                           C=float(Tf + Tc + (2 if lev > 0 else 1)), trvar_fine=float(np.sum(Afi.var(0)))))
    trv = levels[-1]['trvar_fine']
    # bias of plain MC with burn-in T = 2^lev (start h = 0): sum of later level means
    means = np.array([lv['mean'] for lv in levels])
    bias = []
    for lev in range(len(levels)):
        b = means[lev + 1:].sum(0) if lev + 1 < len(levels) else np.zeros(n)
        se2 = sum(lv['V'] for lv in levels[lev + 1:]) / M
        bias.append(dict(T=2 ** lev, bias2_over_trvar=float(b @ b / trv), se2_over_trvar=float(se2 / trv)))
    VC = [math.sqrt(lv['V'] * lv['C']) for lv in levels]
    res.update(levels=levels, trvar=trv, mlmc_var_cost_over_trvar=float(sum(VC) ** 2 / trv), bias=bias,
               mc_var_cost_at_B=[dict(B=2 ** lev, cost=float(2 ** lev + 1)) for lev in range(len(levels))])

elif a.mode == 'cv':
    B = a.B or 8 * n
    # K2-Stein closure from h_{-B} = 0 (so h_{-B+1} ~ N(0, I)): m, C, p per layer; surrogate covariance Sg.
    m = np.zeros(n); C = np.zeros((n, n)); Sg = np.zeros((n, n)); Sh = np.zeros((n, n))
    P_by = {}; H_by = {}
    for l in range(-B + 1, 1):
        W = Wf(l)
        mu = W @ m; S = W @ C @ W.T
        s = np.sqrt(np.maximum(np.diag(S), 0.0))
        m1, v1, Phi = relu_moments(mu, s)
        Cn = (Phi[:, None] * S) * Phi[None, :]
        np.fill_diagonal(Cn, v1)
        m = m1; C = Cn + np.eye(n)
        A = Phi[:, None] * W
        Sg = A @ Sg @ A.T + np.eye(n)
        P_by[l] = Phi
        H_by[l] = (mu > 0).astype(float)
        Ah = H_by[l][:, None] * W
        Sh = Ah @ Sh @ Ah.T + np.eye(n)
    W1 = Wf(1)
    muo = W1 @ m
    so_k2 = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', W1, C, W1), 0.0))
    EK2, _, _ = relu_moments(muo, so_k2)
    so_g = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', W1, Sg, W1), 0.0))
    EG, _, _ = relu_moments(muo, so_g)
    so_h = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', W1, Sh, W1), 0.0))
    EH, _, _ = relu_moments(muo, so_h)
    sz1 = np.linalg.norm(W1, axis=1)                    # final-noise scale of each output pre-activation (sigma = 1)
    rng = np.random.default_rng([7170, seed, n])
    As = []; AGs = []; AHs = []; RBs = []; GRBs = []
    done = 0
    while done < M:
        k = min(CH, M - done)
        h = np.zeros((k, n)); d = np.zeros((k, n)); dh = np.zeros((k, n))
        for l in range(-B + 1, 1):
            W = Wf(l); xi = rng.standard_normal((k, n))
            apre = np.maximum(h @ W.T, 0.0)
            h = apre + xi
            dpre = (d @ W.T) * P_by[l][None, :]
            d = dpre + xi
            dh = (dh @ W.T) * H_by[l][None, :] + xi
        As.append(out_a(h)); AGs.append(np.maximum((m[None, :] + d) @ W1.T, 0.0))
        AHs.append(np.maximum((m[None, :] + dh) @ W1.T, 0.0))
        sz_b = np.broadcast_to(sz1[None, :], (k, n))
        rb, _, _ = relu_moments(apre @ W1.T, sz_b)
        grb, _, _ = relu_moments((m[None, :] + dpre) @ W1.T, sz_b)     # surrogate with its final noise integrated
        RBs.append(rb); GRBs.append(grb); done += k
    Aa = np.concatenate(As); AG = np.concatenate(AGs); AH = np.concatenate(AHs); ARB = np.concatenate(RBs)
    AGRB = np.concatenate(GRBs)
    half = M // 2

    def cvfit(Y, G, EGv):
        # beta per coordinate from the first half, evaluated on the second half (no in-sample optimism)
        cov = np.mean((Y[:half] - Y[:half].mean(0)) * (G[:half] - G[:half].mean(0)), 0)
        vG = G[:half].var(0); vY = Y[:half].var(0)
        beta = np.where(vG > 1e-12 * np.maximum(vY, 1e-300), cov / np.maximum(vG, 1e-300), 0.0)   # fix 1: guard degenerate coordinates
        return Y[half:] - beta[None, :] * (G[half:] - EGv[None, :])

    resid = cvfit(Aa, AG, EG); residh = cvfit(Aa, AH, EH); resid_rb = cvfit(ARB, AGRB, EG)
    trv = float(np.sum(Aa[half:].var(0)))
    res.update(B=B, trvar=trv, cv_var_ratio=float(np.sum(resid.var(0)) / trv),
               cv_var_ratio_beta1=float(np.sum((Aa[half:] - AG[half:]).var(0)) / trv),
               cv_hard_var_ratio=float(np.sum(residh.var(0)) / trv),
               rb_var_ratio=float(np.sum(ARB[half:].var(0)) / trv),
               rb_cv_var_ratio=float(np.sum(resid_rb.var(0)) / trv),
               surrogate_mean_check=float(np.sum((AG.mean(0) - EG) ** 2) / np.sum(AG.var(0)) * M),
               surrogate_hard_mean_check=float(np.sum((AH.mean(0) - EH) ** 2) / max(np.sum(AH.var(0)), 1e-300) * M),
               mc_mean=Aa.mean(0).tolist(), cv_est=resid.mean(0).tolist(), EK2=EK2.tolist(), EG=EG.tolist(),
               setup_mv=float(B * 5 * n), per_sample_mv=float(2 * B + 2), per_sample_mv_both=float(3 * B + 3))
    if a.truth:
        T = json.load(open(a.truth)); tm = np.array(T['mean']); tv = T['trvar']
        res['nmse_K2'] = float(np.sum((EK2 - tm) ** 2) / tv)
        res['nmse_K2_truth_floor'] = 1.0 / T['M'] if 'M' in T else None

elif a.mode == 'rqmc':
    from scipy.stats import qmc
    B = a.B or 8 * n
    d_layers = min(B, 21201 // n)
    res.update(B=B, d_layers=d_layers)
    rows = []
    for k in range(6, a.kmax + 1, 2):
        N = 2 ** k
        ests = []
        for r in range(a.R):
            eng = qmc.Sobol(d=d_layers * n, scramble=True, seed=np.random.default_rng([8180, seed, n, k, r]))
            U = eng.random_base2(k)
            G = ndtri(np.clip(U, 1e-16, 1 - 1e-16))       # (N, d_layers * n): layer 0 first, then -1, ...
            rng = np.random.default_rng([8181, seed, n, k, r])
            h = np.zeros((N, n))
            for l in range(-B + 1, 1):
                age = -l
                if age < d_layers:
                    xi = G[:, age * n:(age + 1) * n]
                else:
                    xi = rng.standard_normal((N, n))
                h = np.maximum(h @ Wf(l).T, 0.0) + xi
            ests.append(out_a(h).mean(0))
        E = np.array(ests)
        rows.append(dict(N=N, var_sum=float(np.sum(E.var(0, ddof=1)))))
    res['rows'] = rows

elif a.mode == 'anti':
    # antithetic pairs: the whole noise history xi and -xi (both from h = 0 at -B); per-pair estimator (a+ + a-)/2
    B = a.B or 8 * n
    rng = np.random.default_rng([9190, seed, n])
    Ps = []; Ys = []
    done = 0
    while done < M:
        k = min(CH, M - done)
        hp = np.zeros((k, n)); hm = np.zeros((k, n))
        for l in range(-B + 1, 1):
            W = Wf(l); xi = rng.standard_normal((k, n))
            hp = np.maximum(hp @ W.T, 0.0) + xi; hm = np.maximum(hm @ W.T, 0.0) - xi
        ap_ = out_a(hp); am_ = out_a(hm)
        Ps.append(0.5 * (ap_ + am_)); Ys.append(ap_); done += k
    Pp = np.concatenate(Ps); Yy = np.concatenate(Ys)
    trv = float(np.sum(Yy.var(0)))
    res.update(B=B, trvar=trv, pair_var_over_half_trvar=float(np.sum(Pp.var(0)) / (trv / 2)))

elif a.mode == 'tail':
    # P(|a| > t), t from the truth file (the 1e-3 quantile). MC and radial-tilt IS: over ages j < K = 2n the noise is
    # drawn from N(theta_j e_l, I), e_l = direction of relu(W_l h_{l-1}); theta_j = s c_j / sqrt(sum c^2), c_j =
    # exp(-1.25 j/n); weight prod exp(-theta_j xi.e + theta_j^2/2). Relative variance per sample for s in {0,2,3,4}.
    B = a.B or 8 * n
    T = json.load(open(a.truth)); t = T['tails'][a.ptail]['t']; p_truth = T['tails'][a.ptail]['p']
    K = 2 * n
    cj = np.exp(-1.25 * np.arange(K) / n); cn = math.sqrt(float(np.sum(cj ** 2)))
    out_rows = []
    for sc in [float(x) for x in a.scales.split(',')]:
        rng = np.random.default_rng([9290, seed, n, int(sc * 10)])
        vals = []
        done = 0
        while done < M:
            k = min(CH, M - done)
            h = np.zeros((k, n)); logw = np.zeros(k)
            for l in range(-B + 1, 1):
                W = Wf(l); age = -l
                apre = np.maximum(h @ W.T, 0.0)
                xi = rng.standard_normal((k, n))
                if sc > 0 and age < K:
                    th = sc * cj[age] / cn
                    nr = np.linalg.norm(apre, axis=1, keepdims=True)
                    e = apre / np.where(nr > 0, nr, 1.0)
                    xi = xi + th * e
                    logw += -th * np.sum(xi * e, 1) + 0.5 * th * th
                h = apre + xi
            ind = (np.linalg.norm(out_a(h), axis=1) > t).astype(float)
            vals.append(np.exp(logw) * ind); done += k
        v = np.concatenate(vals)
        pe = float(v.mean())
        out_rows.append(dict(scale=sc, p_est=pe, rel_var_per_sample=float(v.var() / max(p_truth, 1e-300) ** 2),
                             frac_hits=float(np.mean(v > 0))))
    # gradient tilt: fixed shifts theta_l = s G_l / |G|, G_l = mean over a pilot of d|a|/d xi_l (reverse mode along each
    # pilot path's own gates), for ages < K. The pilot (512 paths, own seed) is charged as setup.
    Pp = 512
    rngp = np.random.default_rng([9390, seed, n])
    h = np.zeros((Pp, n)); gates = {}
    for l in range(-B + 1, 1):
        z = h @ Wf(l).T
        if -l < K:
            gates[l] = (z > 0).astype(float)
        h = np.maximum(z, 0.0) + rngp.standard_normal((Pp, n))
    z1 = h @ Wf(1).T; a1 = np.maximum(z1, 0.0); na = np.linalg.norm(a1, axis=1, keepdims=True)
    v = ((a1 / np.where(na > 0, na, 1.0)) * (z1 > 0)) @ Wf(1)       # d|a|/dh_0, per path
    G = {}
    for l in range(0, -K, -1):
        G[l] = v.mean(0)                                              # d|a|/d xi_l = d|a|/dh_l
        v = (v * gates[l]) @ Wf(l)                                    # d|a|/dh_{l-1}
    Gn = math.sqrt(sum(float(G[l] @ G[l]) for l in G))
    for sc in [float(x) for x in a.scales.split(',') if float(x) > 0]:
        rng = np.random.default_rng([9490, seed, n, int(sc * 10)])
        vals = []
        done = 0
        while done < M:
            k = min(CH, M - done)
            h = np.zeros((k, n)); logw = np.zeros(k)
            for l in range(-B + 1, 1):
                apre = np.maximum(h @ Wf(l).T, 0.0)
                xi = rng.standard_normal((k, n))
                if l in G:
                    th = sc * G[l] / Gn
                    xi = xi + th[None, :]
                    logw += -(xi @ th) + 0.5 * float(th @ th)
                h = apre + xi
            ind = (np.linalg.norm(out_a(h), axis=1) > t).astype(float)
            vals.append(np.exp(logw) * ind); done += k
        vv = np.concatenate(vals)
        out_rows.append(dict(scale=sc, kind='grad', p_est=float(vv.mean()),
                             rel_var_per_sample=float(vv.var() / max(p_truth, 1e-300) ** 2), frac_hits=float(np.mean(vv > 0))))
    res.update(B=B, t=t, p_truth=p_truth, rows=out_rows, mc_rel_var=(1 - p_truth) / p_truth, pilot_paths=Pp)

res['sec'] = time.time() - t0
s = json.dumps(res)
if a.out:
    open(a.out, 'w').write(s)
print(s[:2500], flush=True)
