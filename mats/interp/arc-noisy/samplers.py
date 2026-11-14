"""Input designs for the sampling baselines (device-agnostic torch).

mc:    iid N(0, I) points.
sobol: scrambled Sobol (Owen-type scrambling as implemented by torch.quasirandom.SobolEngine), mapped by ndtri.
mub:   the Kerdock design used by the best published Phase-1 samplers: antipodal pairs +-rho*Q*b over the 129
       mutually unbiased bases of R^256 (kerdock.py), Q a Haar rotation drawn per replicate (folded into W_1, so the
       first layer costs n^2 per point: one matvec per antipodal pair), bases in random order, radius per pair
       either stratified chi_n (unbiased with noise) or fixed at E|x| (exact radial integration by homogeneity when
       sigma = 0; the radius-noise interaction is O(sigma^2/n)). Blocks of 66,048 points; a new rotation per block.
All designs give every point marginal law N(0, I), so every sample mean is unbiased.

Costs (flopscope conventions, float32): per path flops_path() from anet; mc/sobol inputs 16 flops per coordinate
(sobol: generation + ndtri charged as one normal draw); mub: first layer n^2 + n per point instead of
n(2n-1) + n and no input draws, plus per block 2 n^3 (fold) + 4 n^3 (QR) + per basis 2 n^3 (W1 Q B^T).
"""
import math
import numpy as np
import torch
from scipy.stats import chi as chi_dist

import anet


def haar(n, gen, device, dtype=torch.float64):
    A = torch.randn(n, n, generator=gen, device=gen.device, dtype=torch.float64).to(device)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.to(dtype)


def chi_radii(K, n, rng, mode):
    if mode == 'fixed':
        return np.full(K, anet.chi_mean(n))
    u = (np.arange(K) + rng.random(K)) / K           # stratified
    rng.shuffle(u)
    return chi_dist.ppf(u, n)


class MUBDesign:
    """Yields first-layer pre-activations z1 (and the input points' radii) for the Kerdock antipodal design."""

    def __init__(self, B_np, W1, n=256):
        self.B = torch.from_numpy(B_np)                # (129, n, n) rows = basis vectors
        self.W1 = W1; self.n = n

    def z1_points(self, N, gen, rng, device, radius='strat', dtype=torch.float32):
        """Returns z1 (N, n) = W1 x for N design points (N even), plus setup flops."""
        n = self.n; out = []; done = 0; flops = 0
        while done < N:
            Q = haar(n, gen, device)
            W1Q = (self.W1.to(device) @ Q)                       # fold rotation into W1
            flops += 4 * n ** 3 + 2 * n ** 3 + n * n * anet.F_NORMAL
            order = rng.permutation(self.B.shape[0])
            for k in order:
                if done >= N:
                    break
                take = min(n, (N - done) // 2)
                Bk = self.B[k].to(device=device, dtype=torch.float64)
                rows = torch.from_numpy(rng.permutation(n)[:take]).to(device) if take < n else torch.arange(n, device=device)
                Z = Bk[rows] @ W1Q.T                               # (take, n): W1 Q b for each basis vector b
                flops += 2 * n ** 3
                rad = torch.from_numpy(chi_radii(take, n, rng, radius)).to(device)
                Z = Z * rad[:, None]
                out.append(torch.stack([Z, -Z], 1).reshape(-1, n).to(dtype))   # antipodal pairs adjacent
                done += 2 * take
        return torch.cat(out)[:N], flops


def sobol_points(N, n, seed, device, dtype=torch.float32):
    eng = torch.quasirandom.SobolEngine(dimension=n, scramble=True, seed=seed)
    U = eng.draw(N, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    return torch.special.ndtri(U).to(device=device, dtype=dtype)
