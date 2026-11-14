"""Nets for the sparse-codes lane (work/sparse-codes). Thin wrappers over unedited copies of the lanes' net code:
  ext/anet_copy.py  = work/arc-noisy/anet.py   (ARC He-ReLU: z = W h, a = ReLU(z), h = a + sigma r_l eps; no bias, no LN)
  ext/rnet_copy.py  = work/cm-interp-R/rnet.py (tanh + LN, bias N(0, 0.25) as in the gate-entropy census:
                      z = W h + b, a = LN(tanh(z)), h = a + sigma eps; g = 0.25 'ordered', g = 3 'chaotic')
Uniform interface (all float32 on the chosen device unless stated):
  net.n, net.L, net.kind ('relu' | 'tanh'), net.s[l] (noise sd added after layer l, 1-based), net.W[l] (n, n), net.b[l]
  net.pre(l, h)   -> z_l = W_l h + b_l
  net.act(l, z)   -> a_l (ReLU, or LN(tanh(z)))
  net.sample(N, gen, layers) -> dict l -> a_l (noiseless post-activation of layer l, the thing the next layer's noise is
                                 added to), for l in layers; streamed in chunks.
Seeds: new nets start at 64 (ARC's 0-63 and 100-111 are used elsewhere). Dev seed 64 (exploration only).
"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), 'ext')
if EXT not in sys.path:
    sys.path.insert(0, EXT)
import anet_copy as A      # noqa: E402
import rnet_copy as R      # noqa: E402

FAMILIES = ('arc', 'tord', 'tcha')
TANH_G = {'tord': 0.25, 'tcha': 3.0}


class SNet:
    def __init__(self, family, n, seed, sigma, L=32, device='cpu', dtype=torch.float32):
        self.family, self.n, self.seed, self.sigma, self.L = family, n, seed, sigma, L
        self.device, self.dtype = device, dtype
        if family == 'arc':
            r = None
            tabp = os.path.join(EXT, 'rtable_sc.json')     # this lane's seeded pilots (scratch/make_rtable.py)
            if sigma > 0 and os.path.exists(tabp):
                import json
                r = json.load(open(tabp)).get(f"{n}_{L}_{seed}_{sigma:g}")
            net = A.Net(seed, n=n, L=L, sigma=sigma, r=r)  # r None -> ARC's seeded pilot (or rtable for known nets)
            self.kind = 'relu'
            self.W = [w.to(device, dtype) for w in net.W]
            self.b = [None] * L
            self.s = [float(v) for v in net.s]                  # s[l-1] = sd of noise after layer l
            self.W64 = net.W
            self.anet = net
        else:
            net = R.RNet(n, L, 'tanh', TANH_G[family], sigma, seed, bias=0.5)
            self.kind = 'tanh'
            self.W = [w.to(device, dtype) for w in net.W]
            self.b = [b.to(device, dtype) for b in net.b]
            self.s = [float(sigma)] * L
            self.W64 = net.W
            self.rnet = net
        self.sqn = math.sqrt(n)

    def name(self):
        return f"{self.family}_n{self.n}_s{self.sigma:g}_seed{self.seed}"

    def pre(self, l, h):
        z = h @ self.W[l - 1].T
        if self.b[l - 1] is not None:
            z = z + self.b[l - 1]
        return z

    def act(self, l, z):
        if self.kind == 'relu':
            return torch.relu(z)
        a = torch.tanh(z)
        c = a - a.mean(-1, keepdim=True)
        return c * (self.sqn / c.norm(dim=-1, keepdim=True).clamp_min(1e-30))

    def noise_sd(self, l):
        """sd of the noise added after layer l (1-based)."""
        return self.s[l - 1]

    @torch.no_grad()
    def sample(self, N, gen, layers, chunk=1 << 15, upto=None, keep_pre=False):
        layers = sorted(set(layers)); upto = upto or max(layers)
        out = {l: [] for l in layers}; pre = {l: [] for l in layers} if keep_pre else None
        done = 0
        while done < N:
            m = min(chunk, N - done)
            h = torch.randn(m, self.n, generator=gen, device=self.device, dtype=self.dtype)
            for l in range(1, upto + 1):
                z = self.pre(l, h)
                a = self.act(l, z)
                if l in out:
                    out[l].append(a)
                    if keep_pre:
                        pre[l].append(z)
                h = a + self.s[l - 1] * torch.randn(a.shape, generator=gen, device=self.device, dtype=self.dtype)
            done += m
        res = {l: torch.cat(v) for l, v in out.items()}
        if keep_pre:
            return res, {l: torch.cat(v) for l, v in pre.items()}
        return res

    @torch.no_grad()
    def push(self, a, l, gen, steps=1):
        """Push states a_l (post-activation of layer l) through layers l+1..l+steps with noise; returns a_{l+steps}."""
        x = a
        for j in range(l, l + steps):
            h = x + self.s[j - 1] * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
            x = self.act(j + 1, self.pre(j + 1, h))
        return x

    @torch.no_grad()
    def next_pre(self, a, l, gen):
        """z_{l+1} = W_{l+1} (a_l + s_l eps) + b_{l+1} for states a_l."""
        h = a + self.s[l - 1] * torch.randn(a.shape, generator=gen, device=a.device, dtype=a.dtype)
        return self.pre(l + 1, h)


def input_law(name, n, seed):
    """Input laws for the E4 generalisation test (float64 numpy): returns (mu0 (n,), L0 (n, n)) with x = mu0 + L0 xi.
    std: N(0, I) (the law the transcoders were trained on); shift: N(0.5 g, I), g ~ N(0, I);
    aniso: N(0, Q diag(lam) Q^T), lam_i proportional to 1/i, trace n, Q Haar-random."""
    rs = np.random.RandomState(7_000_001 + 131 * seed + n)
    g = rs.standard_normal(n)
    Q, R = np.linalg.qr(rs.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(R))[None, :]
    if name == 'std':
        return np.zeros(n), np.eye(n)
    if name == 'shift':
        return 0.5 * g, np.eye(n)
    if name == 'aniso':
        lam = 1.0 / np.arange(1, n + 1); lam *= n / lam.sum()
        return np.zeros(n), Q * np.sqrt(lam)[None, :]
    raise ValueError(name)
