"""ARC random-MLP family with small relative noise (work/arc-noisy).

Object (fixed by the brief, 2026-10-29):
    x ~ N(0, I_n);  z_l = W_l h_{l-1};  a_l = ReLU(z_l);  h_l = a_l + s_l eps_l,  s_l = sigma * r_l
    W_l iid N(0, 2/n) (contest form, every layer), no bias, no LayerNorm.
    r_l = RMS of a_l in the noisy net, fixed once per net from the weights by a seeded pilot (2^12 paths, float32,
    layer by layer; rtable.json). It is part of the object definition, so no estimator is charged for it.
Estimand: E[a_L] per neuron (final-layer post-ReLU means; the noise after a_L has mean zero).
Depth 8 is the truncation of the depth-32 net (same W_1..W_8, same r_1..r_8).
Weights come from numpy's legacy RandomState (stream stable across versions), so the GPU truth runs rebuild the same
float64 weights (and then run them in float32; largest difference 1.5e-8); weights_checksum() of the float64 weights is
stored with each truth and checked by analyze.py.

FLOP conventions (flopscope 0.12, float32 rates; everything is counted as if run in float32):
    matmul (M,K)@(K,N): M*N*(2K-1);  elementwise +,-,*,/,max,compare,sqrt: 1 per element;
    exp/log/pow/erf: 16 per element; standard normal draw: 16 per element; norm.cdf ~96, norm.pdf ~54 per element.
"""
import json, math, os
import numpy as np
import torch

DT = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
RTABLE = os.path.join(HERE, 'rtable.json')
F_NORMAL = 16       # flopscope: rng.standard_normal float32, per element
F_TRANS = 16        # exp / log / power / erf
F_CDF = 96          # flops.stats.norm.cdf per element
F_PDF = 54          # flops.stats.norm.pdf per element
SIGMAS = (0.001, 0.003, 0.01, 0.03, 0.1)


def mm_flops(M, K, N):
    return M * N * (2 * K - 1)


def make_weights(seed, n=256, L=32):
    rs = np.random.RandomState(1_000_003 + 7919 * seed)
    return [torch.from_numpy(rs.standard_normal((n, n)) * math.sqrt(2.0 / n)) for _ in range(L)]


def weights_checksum(W):
    return float(sum((w.double() * (k + 1)).sum() for k, w in enumerate(W)))


@torch.no_grad()
def pilot_rms(W, sigma, m=1 << 12, seed=0):
    """r_l = sqrt(mean_i E[a_l,i^2]) in the noisy net, from m seeded float32 paths, layer by layer."""
    g = torch.Generator().manual_seed(99_000_017 + 31 * seed)
    n = W[0].shape[0]
    h = torch.randn(m, n, generator=g, dtype=torch.float32)
    r = []
    for w in W:
        a = torch.relu(h @ w.float().T)
        r.append(float(a.double().pow(2).mean().sqrt()))
        h = a + float(sigma * r[-1]) * torch.randn(m, n, generator=g, dtype=torch.float32)
    return r


def load_rtable():
    return json.load(open(RTABLE)) if os.path.exists(RTABLE) else {}


class Net:
    """One net; estimators read W (list of (n, n), z = W h), s (noise sd added after each ReLU) and L."""

    def __init__(self, seed, n=256, L=32, sigma=0.0, r=None, W=None):
        self.seed, self.n, self.L, self.sigma = seed, n, L, sigma
        self.W = W if W is not None else make_weights(seed, n, L)
        if sigma > 0:
            if r is None:
                tab = load_rtable()
                key = f"{n}_{L}_{seed}_{sigma:g}"
                r = tab[key] if key in tab else pilot_rms(self.W, sigma, seed=seed)
            self.r = torch.tensor(r, dtype=DT)
        else:
            self.r = torch.ones(L, dtype=DT)
        self.s = sigma * self.r                     # s[l]: sd of the noise added after ReLU of layer l+1 (0-based l)

    def truncate(self, L):
        return Net(self.seed, self.n, L, self.sigma, r=self.r[:L].tolist(), W=self.W[:L])

    def name(self):
        return f"n{self.n}_L{self.L}_s{self.sigma:g}_seed{self.seed}"

    # ------------------------------------------------------------------ forward passes
    @torch.no_grad()
    def forward(self, x, gen, L=None, keep=False, rb_last=False, dtype=None, noise=None):
        """Noisy forward pass of a batch x (m, n) through L layers. Returns a_L (post-ReLU, before the last noise)
        or, if keep, the list of a_l for l = 1..L. rb_last=True returns E[a_L | a_{L-1}] with the noise after
        a_{L-1} integrated analytically per neuron (needs sigma > 0). noise: optional list of (m, n) standard
        normal tensors to use instead of fresh draws (for coupled paths)."""
        L = L or self.L
        dtype = dtype or x.dtype
        rb = rb_last and self.sigma > 0 and L >= 2
        h = x; out = []
        for l in range(L):
            W = self.W[l].to(dtype)
            if rb and l == L - 1:
                mu = h @ W.T                                           # h = a_{L-1} without its noise
                sd = (self.s[l - 1] * W.norm(dim=1)).to(dtype)          # that noise pushed through the row norms
                a = relu_gauss_mean(mu, sd[None, :])
            else:
                a = torch.relu(h @ W.T)
            if keep:
                out.append(a)
            if l < L - 1:
                if self.sigma > 0 and not (rb and l == L - 2):
                    e = noise[l] if noise is not None else torch.randn(a.shape, generator=gen, dtype=dtype)
                    h = a + self.s[l].to(dtype) * e
                else:
                    h = a
        return out if keep else a

    # ------------------------------------------------------------------ costs (flopscope conventions, float32)
    def flops_path(self, L=None, rb_last=False, input_draw=True):
        """FLOPs of one sampled path through L layers (input draw, L matvecs, ReLUs, L-1 noise draws + axpy)."""
        n = self.n; L = L or self.L
        f = n * F_NORMAL if input_draw else 0
        f += L * (mm_flops(1, n, n) + n)                    # W h, ReLU
        if self.sigma > 0:
            f += (L - 1) * n * (F_NORMAL + 2)               # noise draw, scale, add
            if rb_last:
                f += n * (F_CDF + F_PDF + 6)                # E[ReLU(N(mu, sd^2))]
        return f


def relu_gauss_mean(mu, sd):
    """E[ReLU(Z)], Z ~ N(mu, sd^2), elementwise (sd > 0)."""
    t = mu / sd
    return mu * torch.special.ndtr(t) + sd * torch.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)


def chi_mean(n):
    """E|x| for x ~ N(0, I_n)."""
    return math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))
