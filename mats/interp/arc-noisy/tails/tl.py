"""Rare output events on the noisy ARC nets (work/arc-noisy part 2): network latent, score and FLOP conventions.

Object (the same nets as part 1, anet.py): x ~ N(0, I_n), z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps_l,
W_l iid N(0, 2/n), no bias, s_l = sigma r_l from rtable.json (depth-32 table; depth L < 32 is the truncation).
Event at depth L: the last pre-activation z_L,i = W_L[i] . h_{L-1} above a threshold t (ARC's section 6.6 shape at L = 4:
three ReLU layers, then a linear layer and a threshold). The noise after a_{L-1} enters z_L; nothing after z_L does.

Latent: Z = (x, eps_1, ..., eps_{L-1}) of shape (m, L, n), iid N(0, 1). z_L is a deterministic function of Z.

FLOPs (flopscope 0.12 conventions, float32, as part 1): matvec n x n: n(2n - 1); elementwise op 1; normal draw 16;
exp/log 16; norm.cdf 96; norm.pdf 54.
"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from anet import Net, mm_flops, F_NORMAL, F_CDF, F_PDF, F_TRANS   # noqa: E402

N_WIDTH = 256


class TNet:
    """Depth-L truncation of a noisy net, with the latent / score interface used by every tail estimator."""

    def __init__(self, seed, L, sigma, device='cpu', dtype=torch.float32, n=N_WIDTH):
        base = Net(seed, n=n, L=32, sigma=sigma)
        self.seed, self.L, self.sigma, self.n = seed, L, sigma, n
        self.device, self.dtype = device, dtype
        self.W = [w.to(device=device, dtype=dtype) for w in base.W[:L]]
        self.s = [float(v) for v in base.s[:L]]             # s[l]: noise sd after the ReLU of layer l + 1
        self.W64 = base.W[:L]

    def to(self, device=None, dtype=None):
        device = device or self.device; dtype = dtype or self.dtype
        self.W = [w.to(device=device, dtype=dtype) for w in self.W64]
        self.device, self.dtype = device, dtype
        return self

    def hidden(self, Z):
        """h_{L-1} (m, n) from the latent Z (m, L, n)."""
        h = Z[:, 0]
        for l in range(self.L - 1):
            a = torch.relu_(h @ self.W[l].T)
            h = a.add_(Z[:, l + 1], alpha=self.s[l]) if self.sigma > 0 else a
        return h

    def z(self, Z, idx=None):
        """z_L for all neurons (idx None), one neuron (int) or a list of neurons."""
        h = self.hidden(Z)
        if idx is None:
            return h @ self.W[-1].T
        if isinstance(idx, int):
            return h @ self.W[-1][idx]
        return h @ self.W[-1][idx].T

    def draw(self, m, gen):
        return torch.randn(m, self.L, self.n, generator=gen, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ FLOPs per path / per GS move
    def flops_hidden(self):
        n = self.n
        f = (self.L - 1) * (mm_flops(1, n, n) + n)
        if self.sigma > 0:
            f += (self.L - 1) * 2 * n
        return f

    def flops_eval(self, k_out=1):
        """One path's z_L for k_out output neurons, latent already drawn."""
        return self.flops_hidden() + k_out * (2 * self.n - 1)

    def flops_draw(self):
        return self.L * self.n * F_NORMAL

    def flops_pcn_move(self, k_out=1):
        """pCN proposal (draw + 3 flops per latent entry), evaluation, accept (compare + select per latent entry)."""
        d = self.L * self.n
        return self.flops_draw() + 3 * d + self.flops_eval(k_out) + 2 + d


def gauss_tail_logp(t, mu, sd):
    """log P(N(mu, sd^2) > t), accurate far in the tail."""
    x = (np.asarray(t, float) - mu) / sd
    from scipy.special import log_ndtr
    return log_ndtr(-x)
