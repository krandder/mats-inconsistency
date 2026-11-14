"""Narrow-deep random nets (PREREG Part 1). Plain He-ReLU (ARC's form) and a 1/sqrt(L) residual variant, with small
relative activation noise. float64 torch throughout.

Plain:    z_l = W_l h_{l-1};  a_l = ReLU(z_l);  h_l = a_l + s_l eps_l;  output a_L.     W_l iid N(0, 2/n).
Residual: u_l = W_l h_{l-1};  b_l = ReLU(u_l) + s_l eps_l;  h_l = h_{l-1} + V_l b_l / sqrt(L);  output h_L.
          W_l iid N(0, 2/n), V_l iid N(0, 1/n), h_0 = x.
Noise s_l = sigma * r_l, r_l = RMS of the clean activation of layer l (a_l, resp. ReLU(u_l)) in the noisy net,
from a seeded pilot (2^14 paths). The first L matrices are shared across depths (plain: truncation; residual: the
same matrices with that depth's 1/sqrt(L)).
FLOP conventions (flopscope, as work/arc-noisy/anet.py): matmul (M,K)@(K,N): M N (2K-1); elementwise 1; normal draw
16; exp/log/erf 16; normal cdf 96; normal pdf 54.
"""
import math
import numpy as np
import torch

DT = torch.float64
LMAX = 256
F_NORMAL, F_TRANS, F_CDF, F_PDF = 16, 16, 96, 54


def mm_flops(M, K, N):
    return M * N * (2 * K - 1)


def _rs(seed, n, fam):
    return np.random.RandomState((2_000_003 + 7919 * seed + 104729 * n + (0 if fam == 'plain' else 15_485_863)) % (2 ** 32))


def make_weights(fam, seed, n, L):
    """Depths <= LMAX share their first L matrices (stream of LMAX W's, then LMAX V's). Depths > LMAX (depth-scaling
    probe only) use a stream of L W's then L V's, so they do not share matrices with the LMAX family."""
    rs = _rs(seed, n, fam)
    lm = max(LMAX, L)
    W = [torch.from_numpy(rs.standard_normal((n, n)) * math.sqrt(2.0 / n)) for _ in range(lm)][:L]
    if fam == 'plain':
        return W, None
    V = [torch.from_numpy(rs.standard_normal((n, n)) * math.sqrt(1.0 / n)) for _ in range(lm)][:L]
    return W, V


class Net:
    def __init__(self, fam, seed, n, L, sigma=0.0, r=None, pilot_m=1 << 14):
        self.fam, self.seed, self.n, self.L, self.sigma = fam, seed, n, L, float(sigma)
        self.W, self.V = make_weights(fam, seed, n, L)
        self.c = 1.0 / math.sqrt(L) if fam == 'res' else None
        self.r = torch.ones(L, dtype=DT)
        if sigma > 0:
            self.r = torch.tensor(r, dtype=DT) if r is not None else self.pilot_rms(pilot_m)
        self.s = self.sigma * self.r

    def name(self):
        return f"{self.fam}_n{self.n}_L{self.L}_s{self.sigma:g}_seed{self.seed}"

    @torch.no_grad()
    def pilot_rms(self, m):
        g = torch.Generator().manual_seed(99_000_017 + 31 * self.seed + 7 * self.n)
        h = torch.randn(m, self.n, generator=g, dtype=DT)
        r = []
        for l in range(self.L):
            u = h @ self.W[l].T
            a = torch.relu(u)
            r.append(float(a.pow(2).mean().sqrt()))
            e = torch.randn(m, self.n, generator=g, dtype=DT) * (self.sigma * r[-1])
            if self.fam == 'plain':
                h = a + e
            else:
                h = h + (a + e) @ self.V[l].T * self.c
        return torch.tensor(r, dtype=DT)

    @torch.no_grad()
    def forward(self, x, gen=None, L=None, keep=None, noise=True):
        """x (P, n). Returns output (P, n) and, if keep is a set of layer indices (1..L), a dict l -> (h_l, z_l):
        plain: h_l = state after noise, z_l = pre-activation; residual: h_l = residual stream, z_l = u_l."""
        L = L or self.L
        h = x
        kept = {}
        for l in range(L):
            z = h @ self.W[l].T
            a = torch.relu(z)
            if noise and self.sigma > 0:
                e = torch.randn(a.shape, generator=gen, dtype=DT) * self.s[l]
            else:
                e = None
            if self.fam == 'plain':
                out = a
                h = a + e if e is not None else a
            else:
                b = a + e if e is not None else a
                h = h + (b @ self.V[l].T) * self.c
                out = h
            if keep is not None and (l + 1) in keep:
                kept[l + 1] = (h.clone(), z.clone())
        return (out, kept) if keep is not None else out

    def flops_path(self, L=None, input_draw=True):
        n = self.n; L = L or self.L
        f = n * F_NORMAL if input_draw else 0
        per = mm_flops(1, n, n) + n
        if self.fam == 'res':
            per += mm_flops(1, n, n) + 2 * n
        f += L * per
        if self.sigma > 0:
            f += (L - (1 if self.fam == 'plain' else 0)) * n * (F_NORMAL + 2)
        return f


def chi_mean(n):
    return math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))


def relu_gauss_mean(mu, sd):
    t = mu / sd
    return mu * torch.special.ndtr(t) + sd * torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)


def n_surv(s, delta):
    """s: (..., k) singular values sorted descending; counts s_k >= delta s_1."""
    return (s >= delta * s[..., :1]).sum(-1)


def kstar_F(evals, eta):
    """evals descending (k,), smallest k with tail sum <= eta * total."""
    ev = np.clip(np.asarray(evals, dtype=float), 0, None)
    tot = ev.sum()
    if tot <= 0:
        return 0
    tail = tot - np.cumsum(ev)
    ks = np.where(tail <= eta * tot)[0]
    return int(ks[0] + 1) if len(ks) else len(ev)
