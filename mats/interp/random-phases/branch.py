"""Forgetting by nested (branching) sampling, Lyapunov exponents, and per-layer gate / collapse statistics.

usage: python branch.py REGIME N SIGMA SEED [M K]
For M base paths (inputs x_i ~ N(0, I) with their own noise), at each branch depth m in BRANCH (m = 0: the input) we
draw K independent noise continuations from h_m and run them to depth 64. For every depth L > m:
  R(m -> L) = tr Var_i(E[a_L | h_m = h_m,i]) / tr Var(a_L)   (one-way ANOVA, debiased: Var_b = Var_i(ybar_i) - W/K),
the share of the output variance explained by the state at depth m (m = 0: by the input). Jackknife over 8 blocks of
base paths gives standard errors. For m = 0 we also keep S(L) = tr(Sigma_w^-1 Sigma_b) (between-input variance whitened
by the pooled noise covariance, Wishart-corrected) and the Gaussian-proxy total variation between the output laws of
two typical inputs, TV ~ 2 Phi(sqrt(S/2)) - 1.
Lyapunov: QR on 16 tangent vectors along 16 noisy trajectories; exponents from layers 9..64. Also the mean log
growth of a fresh random direction per layer (typical-direction rate).
Gate stats (relu kinds; from the m = 0 cloud): per layer, the fraction of (input, neuron) pairs whose gate is flipped by
the noise alone (both values among the K continuations), and the fraction whose noise-only margin |E[z|x]| / SD(z|x)
is below 1, 2, 3. Collapse stats from the same cloud: participation ratios of Sigma_b, Sigma_w and the total
covariance, the number of principal components for 90% / 99% of the variance, and |E a|^2 / E|a|^2.
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnets

BRANCH = (0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56)
NB = 8


def pr_stats(C):
    ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    tot = ev.sum()
    if tot <= 0:
        return dict(pr=float('nan'), k90=0, k99=0)
    cs = np.cumsum(ev) / tot
    return dict(pr=float(tot ** 2 / (ev ** 2).sum()), k90=int(np.searchsorted(cs, 0.90) + 1),
                k99=int(np.searchsorted(cs, 0.99) + 1))


@torch.no_grad()
def main():
    regime, n, sigma, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    M = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 32
    t0 = time.time()
    net = pnets.PNet(regime, n, sigma, seed)
    L = net.L
    relu = net.kind in ('relu', 'relu_ln')
    gen = torch.Generator().manual_seed(424242 + 1000 * seed + n)
    # ---- base paths
    x = net.inputs(M, gen)
    H = {0: x}
    h = x
    for l in range(L):
        a = net.act(net.pre(l, h))
        h = net.noise(l, a, gen)
        if (l + 1) in BRANCH:
            H[l + 1] = h.clone()
    out = dict(regime=regime, n=n, sigma=sigma, seed=seed, M=M, K=K, L=L, branch=list(BRANCH), r=net.r)
    blk = np.arange(M) % NB
    R = {}; Rse = {}; tot_var = {}
    S_tv = {}; gate = {}; coll = {}
    for m in BRANCH:
        if sigma == 0 and m > 0:
            continue
        Kc = K if sigma > 0 else 1
        hh = H[m].repeat_interleave(Kc, dim=0)                     # (M*Kc, n), rows i*Kc..i*Kc+Kc-1 share h_m
        Rm = [float('nan')] * L; Rs = [float('nan')] * L; TV = [float('nan')] * L
        for l in range(m, L):
            z = net.pre(l, hh)
            a = net.act(z)
            Y = a.view(M, Kc, n).double()
            ybar = Y.mean(1)                                       # (M, n)
            if Kc > 1:
                wv = Y.var(1, unbiased=True)                       # (M, n)
            else:
                wv = torch.zeros_like(ybar)
            # block sufficient statistics
            stats = []
            for b in range(NB):
                idx = torch.from_numpy(np.where(blk == b)[0])
                yb = ybar[idx]
                stats.append((len(idx), yb.sum(0), (yb ** 2).sum(0), wv[idx].sum(0)))

            def ratio(excl=None):
                nn_ = sum(s[0] for b, s in enumerate(stats) if b != excl)
                s1 = sum(s[1] for b, s in enumerate(stats) if b != excl)
                s2 = sum(s[2] for b, s in enumerate(stats) if b != excl)
                sw = sum(s[3] for b, s in enumerate(stats) if b != excl)
                vb_raw = (s2 - s1 ** 2 / nn_) / (nn_ - 1)          # Var_i(ybar) per neuron
                w = sw / nn_                                       # mean within variance per neuron
                vb = vb_raw - w / Kc
                return float(vb.sum() / (vb.sum() + w.sum())), float((vb + w).sum() / n)
            r_all, tv_all = ratio()
            jk = np.array([ratio(b)[0] for b in range(NB)])
            Rm[l] = r_all
            Rs[l] = float(math.sqrt((NB - 1) / NB * ((jk - jk.mean()) ** 2).sum()))
            if m == 0:
                tot_var.setdefault('tot', [None] * L)[l] = tv_all
            if m == 0 and Kc > 1:
                D = (Y - ybar[:, None, :]).reshape(M * Kc, n)
                Sw = (D.T @ D / (M * (Kc - 1))).numpy()
                yc = (ybar - ybar.mean(0)).numpy()
                Cy = yc.T @ yc / (M - 1)
                Sb = Cy - Sw / Kc
                ev, U = np.linalg.eigh(Sw)
                keep = ev > 1e-9 * max(ev.max(), 1e-300)
                r_ = int(keep.sum())
                Ui = U[:, keep] / np.sqrt(ev[keep])
                dof = M * (Kc - 1)
                corr = (dof - r_ - 1) / dof                        # E[inverse Wishart] correction
                S = float(np.trace(Ui.T @ Cy @ Ui)) * corr - r_ / Kc
                TV[l] = float(2 * norm.cdf(math.sqrt(max(S, 0) / 2)) - 1)
                S_tv.setdefault('S', [None] * L)[l] = S
                S_tv.setdefault('rank', [None] * L)[l] = r_
                tot = Sb + Sw
                mu = Y.reshape(M * Kc, n).mean(0).numpy()
                e2 = float((Y ** 2).mean(0).mean(0).sum())
                coll.setdefault('b', [None] * L)[l] = pr_stats(Sb)
                coll.setdefault('w', [None] * L)[l] = pr_stats(Sw)
                coll.setdefault('tot', [None] * L)[l] = pr_stats(tot)
                coll.setdefault('mean_share', [None] * L)[l] = float((mu ** 2).sum() / e2) if e2 > 0 else float('nan')
                ev_t, U_t = np.linalg.eigh(tot)
                coll.setdefault('cos_pc1_mean', [None] * L)[l] = float(abs(U_t[:, -1] @ mu) / (np.linalg.norm(mu) + 1e-300))
                if relu:
                    Z = z.view(M, Kc, n)
                    on = (Z > 0)
                    flip = (on.any(1) & (~on).any(1)).double().mean().item()
                    zm = Z.mean(1); zs = Z.std(1).clamp_min(1e-30)
                    marg = (zm.abs() / zs)
                    p_on = on.double().mean((0, 1))
                    zt = Z.reshape(M * Kc, n).double()
                    margt = (zt.mean(0).abs() / zt.std(0).clamp_min(1e-30))
                    gate.setdefault('noise_flip', [None] * L)[l] = flip
                    for t in (1, 2, 3):
                        gate.setdefault(f'noise_margin_lt{t}', [None] * L)[l] = float((marg < t).double().mean())
                        gate.setdefault(f'total_margin_lt{t}', [None] * L)[l] = float((margt < t).double().mean())
                    for t in (0.0, 0.01, 0.1):
                        gate.setdefault(f'p_mixed_gt{t}', [None] * L)[l] = float(
                            ((torch.minimum(p_on, 1 - p_on)) > t).double().mean())
                    gate.setdefault('p_dead', [None] * L)[l] = float((p_on == 0).double().mean())
                    gate.setdefault('p_on_mean', [None] * L)[l] = float(p_on.mean())
                else:
                    gate.setdefault('sat_frac_a_gt0.9', [None] * L)[l] = float((Y.abs() > 0.9).double().mean())
                    gate.setdefault('mean_abs_a', [None] * L)[l] = float(Y.abs().mean())
            hh = net.noise(l, a, gen)
        R[m] = Rm; Rse[m] = Rs
        if m == 0:
            S_tv['TV'] = TV
    out.update(R={str(k): v for k, v in R.items()}, R_se={str(k): v for k, v in Rse.items()}, S=S_tv, gate=gate,
               collapse=coll, tot_var=tot_var)
    # ---- Lyapunov exponents (QR) along noisy trajectories
    T, kq = 16, 16
    g2 = torch.Generator().manual_seed(77 + seed)
    h = net.inputs(T, g2)
    V = torch.linalg.qr(torch.randn(T, n, kq, generator=g2, dtype=pnets.F32))[0]
    logs = []; typ = []
    for l in range(L):
        JV = net.jvp(l, h, V)
        Q, Rr = torch.linalg.qr(JV.double())
        logs.append(torch.log(torch.diagonal(Rr, dim1=1, dim2=2).abs().clamp_min(1e-300)).mean(0).numpy())
        V = Q.to(pnets.F32)
        u = torch.randn(T, n, 1, generator=g2, dtype=pnets.F32); u = u / u.norm(dim=1, keepdim=True)
        typ.append(float(torch.log(net.jvp(l, h, u).norm(dim=1)).mean()))
        a = net.act(net.pre(l, h)); h = net.noise(l, a, g2)
    logs = np.array(logs)
    out['lyap'] = dict(per_layer=logs.tolist(), lam=logs[8:].mean(0).tolist(), lam_typ_per_layer=typ,
                       lam_typ=float(np.mean(typ[8:])))
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.join(pnets.HERE, 'results', 'branch'), exist_ok=True)
    fn = os.path.join(pnets.HERE, 'results', 'branch', net.name() + '.json')
    json.dump(out, open(fn, 'w'))
    print(fn, f"{out['seconds']:.1f}s lam1={out['lyap']['lam'][0]:+.4f} lam_typ={out['lyap']['lam_typ']:+.4f}",
          'R0@8,16,32,64:', [round(R[0][l - 1], 4) for l in (8, 16, 32, 64)])


if __name__ == '__main__':
    main()
