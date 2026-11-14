"""Trained narrow nets (Part 4): AlgZoo-style 2nd-argmax RNNs and sorting MLPs, loaded from train/ npz files.
Interfaces mirror nd.Net where the Part 2/3 scripts need them (forward, flops_path, s, fam)."""
import math
import numpy as np
import torch
import nd

DT = torch.float64


class RNNNet:
    """h_0 = 0; z_t = W_hh h_{t-1} + W_hi x_t; h_t = ReLU(z_t) + sigma r_t eps_t; logits = W_oh h_T."""
    fam = 'rnn'

    def __init__(self, path):
        z = np.load(path)
        self.W_hi = torch.tensor(z['W_hi'], dtype=DT); self.W_hh = torch.tensor(z['W_hh'], dtype=DT)
        self.W_oh = torch.tensor(z['W_oh'], dtype=DT); self.r = torch.tensor(z['r'], dtype=DT)
        self.sigma = float(z['sigma']); self.d = self.W_hh.shape[0]; self.T = self.W_oh.shape[0]
        self.s = self.sigma * self.r                       # (T, d)
        self.n = self.T                                    # input dimension (the sequence)
        self.path = path

    @torch.no_grad()
    def forward(self, x, gen=None, keep=None, noise=True, return_masks=False):
        P = x.shape[0]
        h = torch.zeros(P, self.d, dtype=DT)
        kept = {}; masks = []
        for t in range(self.T):
            z = h @ self.W_hh.T + x[:, t:t + 1] * self.W_hi
            a = torch.relu(z)
            if return_masks: masks.append((z > 0).to(DT))
            h = a + torch.randn(a.shape, generator=gen, dtype=DT) * self.s[t] if (noise and self.sigma > 0) else a
            if keep is not None and (t + 1) in keep: kept[t + 1] = h.clone()
        out = h @ self.W_oh.T
        if return_masks: return out, kept, masks
        return (out, kept) if keep is not None else out

    def flops_path(self):
        d, T = self.d, self.T
        f = T * nd.F_NORMAL + T * (nd.mm_flops(1, d, d) + 2 * d + d)
        if self.sigma > 0: f += T * d * (nd.F_NORMAL + 2)
        return f + nd.mm_flops(1, d, T)


class MLPNet:
    """Sorting MLP. res: u_l = W_l h; b_l = ReLU(u_l) + s_l eps; h <- h + V_l b_l; out = h_L.
    plain: h <- ReLU(W_l h) + s_l eps; out = W_out h_L."""

    def __init__(self, path):
        z = np.load(path)
        self.arch = str(z['arch']); self.fam = 'res' if self.arch == 'res' else 'plain'
        W = torch.tensor(z['W'], dtype=DT); self.L, self.n = W.shape[0], W.shape[1]
        self.W = [W[l] for l in range(self.L)]
        self.V = [torch.tensor(z['V'], dtype=DT)[l] for l in range(self.L)] if self.fam == 'res' else None
        self.W_out = torch.tensor(z['W_out'], dtype=DT) if self.fam == 'plain' else None
        self.r = torch.tensor(z['r'], dtype=DT); self.sigma = float(z['sigma']); self.s = self.sigma * self.r
        self.c = 1.0; self.path = path

    @torch.no_grad()
    def forward(self, x, gen=None, keep=None, noise=True, return_masks=False):
        h = x; kept = {}; masks = []
        for l in range(self.L):
            z = h @ self.W[l].T
            a = torch.relu(z)
            if return_masks: masks.append((z > 0).to(DT))
            e = torch.randn(a.shape, generator=gen, dtype=DT) * self.s[l] if (noise and self.sigma > 0) else 0.0
            h = h + (a + e) @ self.V[l].T if self.fam == 'res' else a + e
            if keep is not None and (l + 1) in keep: kept[l + 1] = h.clone()
        out = h if self.fam == 'res' else h @ self.W_out.T
        if return_masks: return out, kept, masks
        return (out, kept) if keep is not None else out

    def flops_path(self):
        n, L = self.n, self.L
        per = nd.mm_flops(1, n, n) + n + (nd.mm_flops(1, n, n) + n if self.fam == 'res' else 0)
        f = n * nd.F_NORMAL + L * per
        if self.sigma > 0: f += L * n * (nd.F_NORMAL + 2)
        if self.fam == 'plain': f += nd.mm_flops(1, n, n)
        return f
