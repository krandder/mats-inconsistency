"""Gaussian closure (K = 2: mean + covariance) for the narrow-deep nets, with flopscope-style FLOP counts.

Plain layer:    z = W h ~ N(W mu, W C W^T); a = ReLU(z): exact Gaussian ReLU moments (gauss_relu.relu_moments,
                copied from work/arc-noisy) or the Stein-linearised version (exact means and variances,
                off-diagonal Cov(a_i, a_j) ~ Phi(alpha_i) Cz_ij Phi(alpha_j)); then h = a + noise (s^2 on the diagonal).
Residual layer: u = W h; b = ReLU(u) + noise; h' = h + c V b. Joint Gaussian closure of (h, u): Cov(h, b) =
                C W^T diag(Phi(alpha)) (Stein's lemma), Cov(b) from ReLU moments (+ s^2 I).
"""
import math
import torch
import gauss_relu as GR

DT = torch.float64
F_CDF, F_PDF = 96, 54


def mmf(M, K, N):
    return M * N * (2 * K - 1)


def relu_mom(m, Cz, exact=True):
    n = m.shape[0]
    if exact:
        mu, C, npairs = GR.relu_moments(m, Cz)
        return mu, C, npairs * GR.FLOPS_PAIR + n * GR.FLOPS_NEURON
    s = Cz.diagonal().clamp_min(1e-300).sqrt()
    a = m / s
    Phi = GR.ncdf(a); phi = GR.npdf(a)
    mu = s * (a * Phi + phi)
    e2 = s * s * ((a * a + 1) * Phi + a * phi)
    C = Phi[:, None] * Cz * Phi[None, :]
    C.diagonal().copy_((e2 - mu * mu).clamp_min(0))
    return mu, C, n * (F_CDF + F_PDF + 12) + 3 * n * n


def noise_var(net, l, n):
    """Per-unit noise variance after layer l (0-based): net.s[l] may be a scalar or a length-n vector."""
    s = net.s[l]
    s = s if torch.is_tensor(s) else torch.tensor(s, dtype=DT)
    return (s.to(DT) ** 2) * torch.ones(n, dtype=DT)


def plain_prop(net, mu, C, l_from, l_to, exact=True, noise_last=False):
    """Propagate N(mu, C) of h_{l_from} (0 = input) through layers l_from+1..l_to (1-based) of a plain net.
    Noise is added after every layer < l_to (and after l_to if noise_last). Returns mu, C, flops."""
    n = net.n; fl = 0
    for l in range(l_from, l_to):
        W = net.W[l]
        m = W @ mu; Cz = W @ C @ W.T
        fl += mmf(n, n, 1) + 2 * mmf(n, n, n)
        mu, C, f = relu_mom(m, Cz, exact); fl += f
        if net.sigma > 0 and (l < l_to - 1 or noise_last):
            C = C + torch.diag(noise_var(net, l, n)); fl += n
    return mu, C, fl


def res_prop(net, mu, C, l_from, l_to, exact=True):
    n = net.n; fl = 0; c = net.c
    for l in range(l_from, l_to):
        W, V = net.W[l], net.V[l]
        m = W @ mu; CW = C @ W.T; Cz = W @ CW
        fl += mmf(n, n, 1) + mmf(n, n, n) * 2
        s = Cz.diagonal().clamp_min(1e-300).sqrt()
        Phi = GR.ncdf(m / s)
        mb, Cb, f = relu_mom(m, Cz, exact); fl += f + n * F_CDF
        if net.sigma > 0:
            Cb = Cb + torch.diag(noise_var(net, l, n)); fl += n
        Chb = CW * Phi[None, :]
        X = Chb @ V.T
        mu = mu + c * (V @ mb)
        C = C + c * (X + X.T) + c * c * (V @ Cb @ V.T)
        fl += n * n + mmf(n, n, n) + 3 * n * n + mmf(n, n, 1) + 2 * mmf(n, n, n) + 2 * n
    return mu, C, fl


def prop(net, mu, C, l_from, l_to, exact=True):
    if net.fam == 'plain':
        return plain_prop(net, mu, C, l_from, l_to, exact)
    return res_prop(net, mu, C, l_from, l_to, exact)


def rnn_prop(net, mu, C, t_from, t_to, exact=True):
    """Trained RNN: h_t = ReLU(W_hh h_{t-1} + W_hi x_t) + noise, x_t ~ N(0, 1) independent. Propagate N(mu, C) of
    h_{t_from} through steps t_from+1..t_to (1-based). Noise after every step (also the last: logits read h_T)."""
    d = net.d; fl = 0
    Whh, Whi = net.W_hh, net.W_hi
    for t in range(t_from, t_to):
        m = Whh @ mu; Cz = Whh @ C @ Whh.T + Whi[:, None] * Whi[None, :]
        fl += mmf(d, d, 1) + 2 * mmf(d, d, d) + 2 * d * d
        mu, C, f = relu_mom(m, Cz, exact); fl += f
        if net.sigma > 0:
            C = C + torch.diag(noise_var(net, t, d)); fl += d
    return mu, C, fl
