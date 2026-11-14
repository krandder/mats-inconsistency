"""Part A measurements of Model S's stationary sample measure (absolute noise, sigma = 1 unless --sigma).
usage: partA.py MODE n seed [--P ..] [--B ..] [--sigma ..] [--out file]
MODE:
  cloud  : P stationary samples at layer 0 (fixed weights); attractor direction u* from noiseless directional pullbacks;
           radial share, angular spectrum, covariance spectrum vs predictions (coordinator sigma^2/(2|lam_k|);
           frozen-gate Lyapunov recursion along one sample's gates and along u*'s gates); near-Gaussianity; output gates.
  births : flips under resampling one layer's noise, per path and layer, against X_pre = |a_l|^2/(n sigma^2).
  age    : effect of resampling the noise of the layer at age j (j layers before layer 0) on h_0 and on the output,
           radial / angular split, participation ratio of Cov(Delta_j).
  lyap   : QR exponents along one stationary noisy trajectory.
  couple : coupling-from-the-past distance of two starts under common weights and noise.
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=1024); ap.add_argument('--B', type=int, default=0)
ap.add_argument('--sigma', type=float, default=1.0); ap.add_argument('--out', default=None)
ap.add_argument('--K', type=int, default=64)
a = ap.parse_args()
n, seed, P, sig = a.n, a.seed, a.P, a.sigma
B = a.B or 12 * n
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=300e6)
res = dict(mode=a.mode, n=n, seed=seed, P=P, B=B, sigma=sig)


def noise_block(l, variant, shape):
    return np.random.default_rng([4242, seed, n, l + st.OFF, variant]).standard_normal(shape)


def stationary(P_, l0, l1, variant=0, keep=None, h=None, gates0=None):
    """Paths from h (default 0) at layer l0-1 through l0..l1 with seeded per-layer noise. keep: set of layers whose
    state h_l is stored. gates0: dict filled with path 0's gates per layer. Returns final h and dict of kept states."""
    h = np.zeros((P_, n)) if h is None else h
    kept = {}
    for l in range(l0, l1 + 1):
        z = h @ Wf(l).T
        if gates0 is not None:
            gates0[l] = (z[0] > 0).astype(float)
        h = np.maximum(z, 0.0) + sig * noise_block(l, variant, (P_, n))
        if keep is not None and l in keep:
            kept[l] = h.copy()
    return h, kept


def attractor_dir(l0, l1, M=8):
    rng = np.random.default_rng([555, seed, n])
    U = rng.standard_normal((M, n)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    dead = np.zeros(M, bool)
    for l in range(l0, l1 + 1):
        Z = np.maximum(U @ Wf(l).T, 0.0)
        nr = np.linalg.norm(Z, axis=1)
        dead |= nr == 0
        nr[nr == 0] = 1.0
        U = Z / nr[:, None]
    live = U[~dead]
    if len(live) == 0:
        return None, 0, 0.0
    G = live @ live.T
    return live[0], int((~dead).sum()), float(np.min(G))


def frozen_lyap_cov(gates_fn, l0, l1):
    """Sigma_l = J_l Sigma_{l-1} J_l^T + sig^2 I with J_l = D_l W_l, gates from gates_fn(l, W) -> (n,) 0/1."""
    S = np.zeros((n, n))
    for l in range(l0, l1 + 1):
        W = Wf(l)
        D = gates_fn(l, W)
        J = D[:, None] * W
        S = J @ S @ J.T + sig ** 2 * np.eye(n)
    return S


def spec_slope(ev, k0, k1):
    k = np.arange(1, len(ev) + 1)
    sel = (k >= k0) & (k <= k1) & (ev > 0)
    if sel.sum() < 3:
        return None
    return float(np.polyfit(np.log(k[sel]), np.log(ev[sel]), 1)[0])


if a.mode == 'cloud':
    gl = {}
    h0, _ = stationary(P, -B + 1, 0, gates0=gl)
    u, nlive, mincos = attractor_dir(-B + 1, 0)
    res.update(attractor_live=nlive, attractor_min_cos=mincos)
    R2 = np.sum(h0 * h0, 1)
    res['X_quantiles'] = [float(np.quantile(R2 / (n * sig ** 2), q)) for q in (0.1, 0.5, 0.9)]
    res['X_mean'] = float(np.mean(R2) / (n * sig ** 2))
    lr = np.log(np.sqrt(R2))
    res['logR_sd_within'] = float(np.std(lr))
    m = h0.mean(0); C = np.cov(h0.T); S2 = h0.T @ h0 / P
    evC = np.sort(np.linalg.eigvalsh(C))[::-1]
    res['evC'] = (evC[:min(n, 64)] / sig ** 2).tolist()
    res['evC_slope_2_n8'] = spec_slope(evC, 2, max(4, n // 8))
    res['evC_slope_2_n4'] = spec_slope(evC, 2, max(4, n // 4))
    if u is not None:
        if float(m @ u) < 0:
            u = -u
        proj = h0 @ u
        res['radial_share'] = float(np.mean(proj ** 2) / np.mean(R2))
        res['cos_mean_dir_u'] = float(m @ u / np.linalg.norm(m))
        Pp = np.eye(n) - np.outer(u, u)
        A = Pp @ S2 @ Pp
        evA = np.sort(np.linalg.eigvalsh(A))[::-1]
        res['ang_rel'] = (evA[:min(n, 32)] / np.mean(proj ** 2)).tolist()
        res['ang_total_rel'] = float(np.trace(A) / np.mean(proj ** 2))
        res['ang_slope'] = spec_slope(evA, 1, max(4, n // 8))
        cosang = proj / np.sqrt(R2)
        res['cos_to_u_quantiles'] = [float(np.quantile(cosang, q)) for q in (0.01, 0.1, 0.5)]
    # coordinator's prediction with lambda_k = -(k + 1/4)/n
    k = np.arange(1, n + 1)
    res['coord_pred'] = (1.0 / (1 - np.exp(-2 * (k + 0.25) / n)))[:min(n, 64)].tolist()
    # frozen-gate Lyapunov recursion along one sample path's own gates (path 0, rerun with its noise) and along u*
    Lw = min(B, 8 * n)
    Sig_path = frozen_lyap_cov(lambda l, W: gl[l], -Lw + 1, 0)
    ev_path = np.sort(np.linalg.eigvalsh(Sig_path))[::-1]
    res['ev_frozen_path'] = (ev_path[:min(n, 64)] / sig ** 2).tolist()
    res['ev_frozen_path_slope'] = spec_slope(ev_path, 2, max(4, n // 8))
    if u is not None:
        # gates of the attractor direction: rerun the noiseless directional dynamics, record gates
        rng = np.random.default_rng([555, seed, n]); U = rng.standard_normal((1, n)); U /= np.linalg.norm(U)
        ga = {}
        for l in range(-B + 1, 1):
            Z = U @ Wf(l).T
            ga[l] = (Z[0] > 0).astype(float)
            Zr = np.maximum(Z, 0.0); nr = np.linalg.norm(Zr); U = Zr / (nr if nr > 0 else 1.0)
        Sig_att = frozen_lyap_cov(lambda l, W: ga[l], -Lw + 1, 0)
        ev_att = np.sort(np.linalg.eigvalsh(Sig_att))[::-1]
        res['ev_frozen_attr'] = (ev_att[:min(n, 64)] / sig ** 2).tolist()
        res['ev_frozen_attr_slope'] = spec_slope(ev_att, 2, max(4, n // 8))
        # trace comparisons
        res['tr_C'] = float(np.trace(C) / sig ** 2); res['tr_S2'] = float(np.trace(S2) / sig ** 2)
        res['tr_frozen_path'] = float(np.trace(Sig_path) / sig ** 2); res['tr_frozen_attr'] = float(np.trace(Sig_att) / sig ** 2)
    # near-Gaussianity of projections on C's top eigenvectors
    w, V = np.linalg.eigh(C); V = V[:, ::-1]
    Zc = (h0 - m) @ V[:, :8]
    sd = Zc.std(0)
    res['skew_top8'] = (np.mean(Zc ** 3, 0) / sd ** 3).tolist()
    res['exkurt_top8'] = (np.mean(Zc ** 4, 0) / sd ** 4 - 3).tolist()
    Rr = np.sqrt(R2)
    res['R_cv'] = float(Rr.std() / Rr.mean()); res['R_skew'] = float(np.mean((Rr - Rr.mean()) ** 3) / Rr.std() ** 3)
    # Amendment A3: direct two-replica angular dispersion delta = E[1 - u.u'] over disjoint pairs of samples
    U0 = h0 / Rr[:, None]
    hp_ = P // 2
    res['delta_pairs'] = float(np.mean(1.0 - np.sum(U0[:hp_] * U0[hp_:2 * hp_], 1)))
    # output-layer gates over the noise
    z1 = h0 @ Wf(1).T
    p = np.mean(z1 > 0, 0)
    H = -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
    res['out_gate_entropy_bits'] = float(H.sum()); res['out_frac_uncertain'] = float(np.mean((p > 0.01) & (p < 0.99)))

elif a.mode == 'births':
    K = a.K
    h, _ = stationary(P, -B + 1, -K)
    rows = []
    for l in range(-K + 1, 1):
        W = Wf(l); Wn = Wf(l + 1)
        z = h @ W.T; al = np.maximum(z, 0.0)
        Xpre = np.sum(al * al, 1) / (n * sig ** 2)
        x1 = noise_block(l, 0, (P, n)); x2 = noise_block(l, 7, (P, n))
        z1 = (al + sig * x1) @ Wn.T; z2 = (al + sig * x2) @ Wn.T
        flips = np.sum((z1 > 0) != (z2 > 0), 1)
        sz = sig * np.linalg.norm(Wn, axis=1)
        near = np.sum(np.abs(z1) < sz[None, :], 1)
        rows.append(np.stack([Xpre, flips, near], 1))
        h = al + sig * x1
    Rw = np.concatenate(rows)
    Xp, fl, nr_ = Rw[:, 0], Rw[:, 1], Rw[:, 2]
    edges = np.quantile(Xp, np.linspace(0, 1, 11))
    bins = []
    for i in range(10):
        s = (Xp >= edges[i]) & (Xp <= edges[i + 1])
        xm = float(np.median(Xp[s]))
        pf = math.acos(xm / (xm + 1.0)) / math.pi
        bins.append(dict(X_med=xm, flips_mean=float(fl[s].mean()), flips_pred=n * pf,
                         near_mean=float(nr_[s].mean()), near_pred_coord=n * math.sqrt(2 / math.pi) / math.sqrt(xm)))
    res.update(bins=bins, flips_mean=float(fl.mean()), X_med=float(np.median(Xp)),
               pred_sqrt_n=0.71 * math.sqrt(n))

elif a.mode == 'age':
    ages = sorted(set([0, 1, 2, 3, 4, 6, 8] + [int(round(2 ** (e / 2))) for e in range(6, 2 * int(math.log2(4 * n)) + 1)]))
    ages = [j for j in ages if j <= 4 * n]
    keep = set([-j - 1 for j in ages])
    h0, kept = stationary(P, -B + 1, 0, keep=keep)
    u, nlive, mincos = attractor_dir(-B + 1, 0)
    if u is not None and float(h0.mean(0) @ u) < 0:
        u = -u
    C = np.cov(h0.T); w, V = np.linalg.eigh(C); V = V[:, ::-1]
    a0 = np.maximum(h0 @ Wf(1).T, 0.0)
    totV = float(np.sum(np.var(h0, 0))); totVa = float(np.sum(np.var(a0, 0)))
    prof = []
    for j in ages:
        lj = -j
        h = kept[lj - 1]
        for l in range(lj, 1):
            z = h @ Wf(l).T
            h = np.maximum(z, 0.0) + sig * noise_block(l, 1 if l == lj else 0, (P, n))
        D = h0 - h
        Da = a0 - np.maximum(h @ Wf(1).T, 0.0)
        CD = D.T @ D / (2 * P)
        trD = float(np.trace(CD))
        pr = trD ** 2 / float(np.sum(CD * CD)) if trD > 0 else 0.0
        row = dict(j=j, T_h=trD / totV, T_a=float(np.sum(Da * Da) / (2 * P)) / totVa, PR=pr)
        if u is not None:
            row['radial'] = float(np.mean((D @ u) ** 2) / 2) / max(trD, 1e-300)
        proj = (D @ V[:, :16]) ** 2
        row['eig_share'] = (proj.mean(0) / 2 / max(trD, 1e-300)).tolist()
        prof.append(row)
    res.update(ages=ages, profile=prof, totV=totV / sig ** 2, totVa=totVa / sig ** 2)

elif a.mode == 'lyap':
    k = min(n, 64)
    h, _ = stationary(1, -B + 1, -30 * n)
    rng = np.random.default_rng([9, seed, n])
    lam, logr = st.lyap_qr(Wf, h, -30 * n + 1, 0, sig, rng, k=k, discard=2 * n)
    res['lam_times_n'] = (lam * n).tolist()
    res['pred_times_n'] = [-(1.25 + (i - 1)) for i in range(1, k + 1)]
    kk = np.arange(1, k + 1)
    sel = kk <= max(4, k // 4)
    res['fit_slope_intercept'] = np.polyfit(kk[sel], lam[sel] * n, 1).tolist()

elif a.mode == 'couple':
    rng = np.random.default_rng([3, seed, n])
    hA = np.zeros((P, n)); hB = rng.standard_normal((P, n)) * sig * n
    dist = []
    for t, l in enumerate(range(-B + 1, 1)):
        W = Wf(l); xi = sig * noise_block(l, 0, (P, n))
        hA = np.maximum(hA @ W.T, 0.0) + xi; hB = np.maximum(hB @ W.T, 0.0) + xi
        if t % max(1, n // 4) == 0:
            d = np.linalg.norm(hA - hB, axis=1) / np.linalg.norm(hA, axis=1)
            dist.append((t + 1, float(np.median(np.log(np.maximum(d, 1e-300))))))
    ts = np.array([d[0] for d in dist]); ld = np.array([d[1] for d in dist])
    sel = (ld > -30) & (ts > 2 * n)
    res['rate_times_n'] = float(np.polyfit(ts[sel], ld[sel], 1)[0] * n) if sel.sum() > 3 else None
    res['dist'] = dist

elif a.mode == 'bitlife':
    # Amendment A2 (post-scorecard, pre-registered before its runs): bit birth and death. Resample the noise of ONE
    # layer at age j; count, at every later layer, the gates that differ from the baseline path ("cascade"); split the
    # final change of h_0 into the linear response along the baseline gates and the nonlinear rest (the bits' part).
    J = 2 * n
    ages = sorted(set([2, 8, max(2, n // 4), max(2, n // 2), n, 2 * n]))
    keep = set([-j - 1 for j in ages])
    h, kept = stationary(P, -B + 1, -J - 1)
    kept = {-J - 1: h.copy()}
    gates = {}
    for l in range(-J, 1):
        z = h @ Wf(l).T
        gates[l] = z > 0
        h = np.maximum(z, 0.0) + sig * noise_block(l, 0, (P, n))
        if l in keep:
            kept[l] = h.copy()
    h0 = h
    totV = float(np.sum(np.var(h0, 0)))
    rows = []
    for j in ages:
        lj = -j
        hp = kept[lj - 1].copy()
        z = hp @ Wf(lj).T
        hp = np.maximum(z, 0.0) + sig * noise_block(lj, 1, (P, n))
        d = sig * (noise_block(lj, 1, (P, n)) - noise_block(lj, 0, (P, n)))   # perturbation of h_{-j}
        flips = []
        for l in range(lj + 1, 1):
            W = Wf(l)
            z = hp @ W.T
            flips.append(float(np.mean(np.sum((z > 0) != gates[l], 1))))
            hp = np.maximum(z, 0.0) + sig * noise_block(l, 0, (P, n))
            d = (d @ W.T) * gates[l]
        D = h0 - hp
        resid = D - (-d)            # h0 - hp = -(perturbed - base); linear part of (base - perturbed) is -d
        ED = float(np.sum(D * D)) / (2 * P)
        dd = float(np.sum(d * d))
        rows.append(dict(j=j, flips=flips, total_cascade=float(np.sum(flips)), first_layer_flips=flips[0] if flips else None,
                         T_h=ED / totV, bit_share=float(np.sum(resid * resid)) / max(float(np.sum(D * D)), 1e-300),
                         lin_norm_ratio=dd / max(float(np.sum(D * D)), 1e-300),
                         lin_corr=float(np.sum(D * (-d))) / math.sqrt(max(dd * float(np.sum(D * D)), 1e-300))))
    res.update(ages=ages, rows=rows, totV=totV / sig ** 2)

elif a.mode == 'cloudhist':
    # POST HOC (after Part A scoring): angular cloud at layer 0 against the radius-history-weighted relative model
    # var_k = R_0^2 sum_j sigma^2 exp(-2 gap_k j) / R_{-j}^2, gap_k = (k-1)/n + (k-1)(k+2)/n^2 (astra's rank-conditioned
    # gaps), R_l = RMS radius over the P samples at layer l; also the constant-radius version and the coordinator's.
    Wn_ = 8 * n
    Rl = {}
    h = np.zeros((P, n))
    for l in range(-B + 1, 1):
        z = h @ Wf(l).T
        h = np.maximum(z, 0.0) + sig * noise_block(l, 0, (P, n))
        if l > -Wn_:
            Rl[l] = math.sqrt(float(np.mean(np.sum(h * h, 1))))
    h0 = h
    m = h0.mean(0); C = np.cov(h0.T)
    mu = m / np.linalg.norm(m)
    Pp = np.eye(n) - np.outer(mu, mu)
    evA = np.sort(np.linalg.eigvalsh(Pp @ C @ Pp))[::-1][:min(n - 1, 32)] / sig ** 2
    R0 = Rl[0]
    ks = np.arange(2, 2 + len(evA))
    gaps = (ks - 1) / n + (ks - 1) * (ks + 2) / n ** 2
    js = np.arange(0, Wn_)
    w = np.array([R0 ** 2 / Rl[-j] ** 2 for j in js])
    pred_hist = np.array([np.sum(w * np.exp(-2 * g * js)) for g in gaps])
    pred_const = 1.0 / (1 - np.exp(-2 * gaps))
    coord = 1.0 / (1 - np.exp(-2 * (ks + 0.25) / n))
    res.update(evA=evA.tolist(), pred_hist=pred_hist.tolist(), pred_const=pred_const.tolist(), coord_k=coord.tolist(),
               ratio_hist=float(np.median(evA[:8] / pred_hist[:8])), ratio_const=float(np.median(evA[:8] / pred_const[:8])),
               ratio_coord=float(np.median(evA[:8] / coord[:8])), R0_over_meanR=float(R0 / np.mean(list(Rl.values()))),
               X0=float(R0 ** 2 / (n * sig ** 2)), slope_meas=spec_slope(evA, 1, max(4, n // 8)),
               slope_hist=spec_slope(pred_hist, 1, max(4, n // 8)))

res['sec'] = time.time() - t0
s = json.dumps(res)
if a.out:
    open(a.out, 'w').write(s)
print(s[:3000], flush=True)
