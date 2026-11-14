"""Gaussian-mixture K2 closure with gate splits, for trained residual MLPs and RNNs (coordinator's follow-up,
2026-10-30). A component is (weight, mean, covariance) of the state. At a scheduled split (layer l, unit i) every
component is split into its z_i > 0 and z_i < 0 pieces (z = pre-activation): each piece is replaced by the Gaussian
with that piece's exact first two moments (truncated normal on z_i, linear regression for the rest), and in that
piece unit i's gate is fixed (on: b_i = z_i; off: b_i = 0). Other units use the usual Gaussian ReLU moments (exact
bivariate, or Stein-linearised). Components are never merged, so k splits give up to 2^k components.
FLOPs follow lib/k2.py's conventions (flopscope-style weights)."""
import math
import torch
import gauss_relu as GR
from k2 import mmf, relu_mom, noise_var, F_CDF, F_PDF

DT = torch.float64
SQ2PI = math.sqrt(2 * math.pi)


def split_piece(mu, C, idx, sign):
    """Moment-matched Gaussian of the piece {v_idx > 0} (sign > 0) or {v_idx < 0} of N(mu, C). Returns P, mu', C'."""
    m = mu[idx]; s2 = C[idx, idx].clamp_min(1e-300); s = s2.sqrt(); a = m / s
    phi = torch.exp(-0.5 * a * a) / SQ2PI
    if sign > 0:
        P = GR.ncdf(a); lam = phi / P.clamp_min(1e-300)
        Ez = m + s * lam; Vz = s2 * (1 - lam * (lam + a))
    else:
        P = GR.ncdf(-a); lam = phi / P.clamp_min(1e-300)
        Ez = m - s * lam; Vz = s2 * (1 - lam * (lam - a))
    Vz = Vz.clamp_min(0)
    beta = C[:, idx] / s2
    mu2 = mu + beta * (Ez - m)
    C2 = C + torch.outer(beta, beta) * (Vz - s2)
    return float(P), mu2, 0.5 * (C2 + C2.T)


def relu_mom_forced(m, Cz, forced, exact=True):
    """ReLU moments of z ~ N(m, Cz) with some gates fixed (forced: dict unit -> 1 on / 0 off).
    Returns mean of b, Cov(b), g (dE[b]/dz per unit: Phi(alpha) free, 1 on, 0 off), flops."""
    n = m.shape[0]
    s = Cz.diagonal().clamp_min(1e-300).sqrt(); Phi = GR.ncdf(m / s)
    free = [i for i in range(n) if i not in forced]
    on = [i for i, v in forced.items() if v]
    mu = torch.zeros(n, dtype=DT); C = torch.zeros(n, n, dtype=DT); fl = n * F_CDF
    if free:
        fi = torch.tensor(free)
        mf, Cf, f = relu_mom(m[fi], Cz[fi][:, fi], exact); fl += f
        mu[fi] = mf; C[fi[:, None], fi[None, :]] = Cf
    if on:
        oi = torch.tensor(on)
        mu[oi] = m[oi]
        C[oi[:, None], oi[None, :]] = Cz[oi][:, oi]
        if free:
            blk = Cz[oi][:, fi] * Phi[fi][None, :]
            C[oi[:, None], fi[None, :]] = blk; C[fi[:, None], oi[None, :]] = blk.T
            fl += 2 * len(on) * len(free)
    g = Phi.clone()
    for i, v in forced.items():
        g[i] = 1.0 if v else 0.0
    return mu, C, g, fl


def res_mix(net, splits, exact=True, return_layers=False):
    """Mixture K2 through a residual MLP (tnets.MLPNet, fam 'res'). splits: dict layer(0-based) -> list of units.
    Returns list of (w, mu, C) for h_L, flops, [per-layer list of (w, m_z, var_z) if return_layers]."""
    n = net.n
    comps = [(1.0, torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT))]
    fl = 0; layers = []
    for l in range(net.L):
        W, V = net.W[l], net.V[l]
        new = []; lay = []
        for (w, mu, C) in comps:
            m_z = W @ mu; C_hz = C @ W.T; C_zz = W @ C_hz
            fl += mmf(n, n, 1) + 2 * mmf(n, n, n)
            units = splits.get(l, [])
            if units:
                mv = torch.cat([mu, m_z]); Cv = torch.cat([torch.cat([C, C_hz], 1), torch.cat([C_hz.T, C_zz], 1)], 0)
                pieces = [(w, mv, Cv, {})]
                for i in units:
                    nxt = []
                    for (pw, pm, pC, forced) in pieces:
                        for sign in (1, -1):
                            P, m2, C2 = split_piece(pm, pC, n + i, sign)
                            fl += 3 * (2 * n) ** 2 + 2 * F_CDF + 20
                            if pw * P < 1e-14:
                                continue
                            f2 = dict(forced); f2[i] = 1 if sign > 0 else 0
                            nxt.append((pw * P, m2, C2, f2))
                    pieces = nxt
                pieces = [(pw, pm[:n], pC[:n, :n], pC[:n, n:], pm[n:], pC[n:, n:], f) for (pw, pm, pC, f) in pieces]
            else:
                pieces = [(w, mu, C, C_hz, m_z, C_zz, {})]
            for (pw, pmu, pC, pChz, pmz, pCzz, forced) in pieces:
                mb, Cb, g, f = relu_mom_forced(pmz, pCzz, forced, exact); fl += f
                if net.sigma > 0:
                    Cb = Cb + torch.diag(noise_var(net, l, n)); fl += n
                X = (pChz * g[None, :]) @ V.T
                mu2 = pmu + V @ mb
                C2 = pC + X + X.T + V @ Cb @ V.T
                fl += n * n + mmf(n, n, n) + mmf(n, n, 1) + 2 * mmf(n, n, n) + 3 * n * n
                new.append((pw, mu2, 0.5 * (C2 + C2.T)))
                if return_layers:
                    lay.append((pw, pmz, pCzz.diagonal().clone()))
        comps = new
        if return_layers:
            layers.append(lay)
    return (comps, fl, layers) if return_layers else (comps, fl)


def rnn_mix(net, splits, exact=True, return_layers=False):
    """Mixture K2 through a trained RNN (tnets.RNNNet). splits: dict step(0-based) -> list of units.
    State components are for h_t; returns list of (w, mu, C) for h_T, flops."""
    d = net.d
    comps = [(1.0, torch.zeros(d, dtype=DT), torch.zeros(d, d, dtype=DT))]
    fl = 0; layers = []
    Whh, Whi = net.W_hh, net.W_hi
    for t in range(net.T):
        new = []; lay = []
        for (w, mu, C) in comps:
            m_z = Whh @ mu; C_zz = Whh @ C @ Whh.T + torch.outer(Whi, Whi)
            fl += mmf(d, d, 1) + 2 * mmf(d, d, d) + 2 * d * d
            pieces = [(w, m_z, C_zz, {})]
            for i in splits.get(t, []):
                nxt = []
                for (pw, pm, pC, forced) in pieces:
                    for sign in (1, -1):
                        P, m2, C2 = split_piece(pm, pC, i, sign)
                        fl += 3 * d * d + 2 * F_CDF + 20
                        if pw * P < 1e-14:
                            continue
                        f2 = dict(forced); f2[i] = 1 if sign > 0 else 0
                        nxt.append((pw * P, m2, C2, f2))
                pieces = nxt
            for (pw, pm, pC, forced) in pieces:
                mb, Cb, g, f = relu_mom_forced(pm, pC, forced, exact); fl += f
                if net.sigma > 0:
                    Cb = Cb + torch.diag(noise_var(net, t, d)); fl += d
                new.append((pw, mb, 0.5 * (Cb + Cb.T)))
                if return_layers:
                    lay.append((pw, pm, pC.diagonal().clone()))
        comps = new
        if return_layers:
            layers.append(lay)
    return (comps, fl, layers) if return_layers else (comps, fl)


def readout(net, comps):
    """Output mixture: list of (w, mean, cov) of the output."""
    if hasattr(net, 'W_oh'):
        return [(w, net.W_oh @ mu, net.W_oh @ C @ net.W_oh.T) for (w, mu, C) in comps]
    return list(comps)


def mix_mean(out):
    W = sum(w for w, _, _ in out)
    return sum(w * m for w, m, _ in out) / W
