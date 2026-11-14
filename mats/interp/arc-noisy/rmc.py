"""Recursive moment control (RMC): an exact Gaussian closure whose non-Gaussian corrections come from particles.

Particles h_s are samples of the TRUE noisy network (plain forward passes). Alongside, a closure carries moments
(mu_l, Sigma_l) of h_l. At each layer, with z = W h:
    mu_{l+1}    = G_mean(W mu_l, W Sigma_l W^T) + [ mean_s a_s      - G_mean(zbar, Czhat) ]
    Sigma^a_{l+1} = G_cov(W mu_l, W Sigma_l W^T) + [ cov_s a_s       - G_cov(zbar, Czhat) ]
    Sigma_{l+1} = Sigma^a_{l+1} + s_l^2 I
where G_* are the exact Gaussian ReLU moments (gauss_relu.relu_moments) and (zbar, Czhat) are the particles' own
sample moments of z. The bracket is a control-variate estimate of the non-Gaussian correction: its expectation is the
true correction (to O(1/N)), and the particle noise in the first two moments cancels inside it, so only the
non-Gaussian part of the particle noise reaches the closure. Consistent as N -> infinity; deterministic closure
recovered as N -> 0 corrections. The estimate of E[a_L] is mu_L.

Cost per layer (flopscope conventions): forward 2n^2 N + noise; particle covariances of z and a: 2 x n^2 (2N-1);
closure W Sigma W^T and particle W C W^T are not needed for the particle side (Czhat is computed directly);
closure: 2 matmuls n^2 (2n-1); two relu_moments calls: pairs x FLOPS_PAIR + n x FLOPS_NEURON.
"""
import math
import torch
import anet
import gauss_relu as GR

DT = torch.float64


def sample_cov(X):
    m = X.mean(0)
    Xc = X - m
    return m, (Xc.T @ Xc) / (X.shape[0] - 1)


@torch.no_grad()
def rmc(net, L, N, gen, x=None, keep=False, det=8.0, cov_corr='full', mean_corr=True, dtype=DT, k_modes=8):
    """Returns (estimate of E[a_L], plain particle mean of a_L, flops, info). x: optional (N, n) input particles."""
    n = net.n
    mu = torch.zeros(n, dtype=DT); Sig = torch.eye(n, dtype=DT)
    h = x if x is not None else torch.randn(N, n, generator=gen, dtype=dtype)
    flops = N * n * anet.F_NORMAL if x is None else 0
    out = []; pairs_total = 0
    for l in range(L):
        W = net.W[l]
        mz = W @ mu; Cz = W @ Sig @ W.T
        flops += 2 * anet.mm_flops(n, n, n) + anet.mm_flops(n, n, 1)
        z = h @ W.T.to(h.dtype)
        a = torch.relu(z)
        flops += anet.mm_flops(N, n, n) + N * n
        if cov_corr == 'full':
            zbar, Czh = sample_cov(z.to(DT))
            abar, Cah = sample_cov(a.to(DT))
            flops += 2 * (anet.mm_flops(n, N, n) + 3 * N * n)
        else:                                            # only per-neuron sample moments of z are needed
            zbar = z.to(DT).mean(0); zv = z.to(DT).var(0)
            abar = a.to(DT).mean(0)
            flops += 6 * N * n
        Gm_c, GC_c, p1 = GR.relu_moments(mz, Cz, det)
        flops += p1 * GR.FLOPS_PAIR + n * GR.FLOPS_NEURON
        pairs_total += p1
        if cov_corr == 'full':
            Gm_s, GC_s, p2 = GR.relu_moments(zbar, Czh, det)
            pairs_total += p2; flops += p2 * GR.FLOPS_PAIR + n * GR.FLOPS_NEURON
            C_a = GC_c + (Cah - GC_s)
        else:
            Gm_s = GR.relu_mean(zbar, zv.clamp_min(1e-300).sqrt())
            flops += n * GR.FLOPS_NEURON
            C_a = GC_c
            if cov_corr in ('diag', 'modes'):
                av = a.to(DT).var(0)
                Gv_s = GR.relu_second(zbar, zv.clamp_min(1e-300).sqrt()) - Gm_s ** 2
                C_a = C_a + torch.diag(av - Gv_s)
                flops += 4 * N * n + n * GR.FLOPS_NEURON
            if cov_corr == 'pcov':
                C_a = torch.cov(a.to(DT).T)
                flops += anet.mm_flops(n, N, n) + 3 * N * n
            if cov_corr == 'modes2':
                # replace the closure covariance inside its top-k eigenspace by the particles' covariance there
                ev, V = torch.linalg.eigh(GC_c); V = V[:, -k_modes:]
                ap = (a.to(DT) - abar) @ V
                Cap = ap.T @ ap / (N - 1)
                C_a = GC_c + V @ (Cap - V.T @ GC_c @ V) @ V.T
                flops += anet.mm_flops(N, n, k_modes) + anet.mm_flops(k_modes, N, k_modes) + 10 * n ** 3
            if cov_corr == 'modes':
                # correction inside the top-k eigenspace of the closure covariance, from projected particles
                ev, V = torch.linalg.eigh(GC_c); V = V[:, -k_modes:]
                ap = (a.to(DT) - abar) @ V                       # (N, k)
                zp_mean = None
                Cap = ap.T @ ap / (N - 1)
                # Gaussian prediction of the same block from the particles' z moments needs the full Gaussian cov;
                # approximate it by the closure's block (no particle-side Gaussian subtraction)
                blk = Cap - V.T @ GC_c @ V
                C_a = C_a + V @ blk @ V.T
                flops += anet.mm_flops(N, n, k_modes) + anet.mm_flops(k_modes, N, k_modes) + 4 * n ** 3
        mu_a = Gm_c + ((abar - Gm_s) if mean_corr else 0)
        if keep:
            out.append(mu_a.clone())
        if l < L - 1:
            s = float(net.s[l]) if net.sigma > 0 else 0.0
            if s > 0:
                e = torch.randn(N, n, generator=gen, dtype=h.dtype)
                h = a + s * e
                flops += N * n * (anet.F_NORMAL + 2)
            else:
                h = a
            mu = mu_a
            Sig = C_a + (s * s) * torch.eye(n, dtype=DT)
            Sig = (Sig + Sig.T) / 2
    info = dict(pairs=pairs_total)
    return (out if keep else mu_a), abar, flops, info
