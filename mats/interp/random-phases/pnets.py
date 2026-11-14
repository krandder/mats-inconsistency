"""Random nets for the phases study (work/random-phases, 2026-10-29). Nothing here edits other lanes' files.

Families (x ~ N(0, I_n); every layer n x n; noise after every activation, a fixed fraction sigma of that layer's
activation RMS r_l, measured in the noisy net by a seeded 2^12-path pilot, part of the object):
  tanh   z_l = W_l h_{l-1} + b_l, W_l = sqrt(sw2/n) G_l, b_l = sqrt(sb2) g_l;  a_l = tanh(z_l);  h_l = a_l + sigma r_l eps
         The standard normal G_l, g_l depend only on the seed, so the tanh regimes differ only by the gain sw2.
  relu   ARC's family exactly as work/arc-noisy (ext/anet_copy.py): W_l iid N(0, 2/n) from anet.make_weights (the first
         32 matrices equal arc-noisy's depth-32 nets for the same seed), no bias, a_l = relu(z_l), r_l from anet's pilot.
  relu_ln z_l = W_l h_{l-1} (W iid N(0, 2/n), no bias); a_l = LN(relu(z_l)) with LN(u) = sqrt(n)(u - mean u)/|u - mean u|
         (no affine; RMS 1, so r_l = 1); h_l = a_l + sigma eps.
Output of the depth-L net: a_L (post-activation, before its noise), as in work/arc-noisy.
"""
import math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'ext'))
import anet_copy as anet                                          # noqa: E402

torch.set_num_threads(1)
F32, F64 = torch.float32, torch.float64

# regimes: name -> (kind, sw2, sb2)
REGIMES = {
    'tanh-ord': ('tanh', 1.0, 0.05),        # chi1 = 0.759, lambda_mf = -0.138
    'tanh-ord2': ('tanh', 0.8, 0.05),       # chi1 = 0.647, lambda_mf = -0.218
    'tanh-crit': ('tanh', 1.761, 0.05),     # chi1 = 1.000 (edge of chaos at sb2 = 0.05)
    'tanh-chaos': ('tanh', 4.0, 0.05),      # chi1 = 1.342, lambda_mf = +0.147
    'tanh-chaos2': ('tanh', 9.0, 0.05),     # chi1 = 1.843, lambda_mf = +0.306
    'relu-he': ('relu', 2.0, 0.0),          # ARC's family; chi1 = 1 (critical)
    'relu-ln': ('relu_ln', 2.0, 0.0),       # chi1 = pi/(pi-1) = 1.467 at sigma -> 0
}
SIGMAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
LMAX = 64


class PNet:
    def __init__(self, regime, n, sigma, seed, L=LMAX):
        self.regime, self.n, self.sigma, self.seed, self.L = regime, n, sigma, seed, L
        self.kind, self.sw2, self.sb2 = REGIMES[regime]
        if self.kind == 'relu':
            W = anet.make_weights(seed, n, L)                      # float64, z = W h
            self.W = [w.to(F32) for w in W]
            self.W64 = W
            self.b = None
        else:
            rs = np.random.RandomState(5_000_011 + 7919 * seed + 13 * n)
            G = [rs.standard_normal((n, n)) for _ in range(L)]
            gb = [rs.standard_normal(n) for _ in range(L)]
            self.W64 = [torch.from_numpy(g * math.sqrt(self.sw2 / n)) for g in G]
            self.W = [w.to(F32) for w in self.W64]
            if self.kind == 'tanh':
                self.b64 = [torch.from_numpy(g * math.sqrt(self.sb2)) for g in gb]
                self.b = [b.to(F32) for b in self.b64]
            else:
                self.b = None
        self.r = self.pilot_rms() if sigma > 0 else [1.0] * L
        self.s = [sigma * rr for rr in self.r]                     # noise sd after layer l (0-based)

    def name(self):
        return f"{self.regime}_n{self.n}_s{self.sigma:g}_seed{self.seed}"

    # ---------------------------------------------------------------- maps
    def pre(self, l, h):
        z = h @ self.W[l].T
        if self.b is not None:
            z = z + self.b[l]
        return z

    def act(self, z):
        if self.kind == 'tanh':
            return torch.tanh(z)
        a = torch.relu(z)
        if self.kind == 'relu_ln':
            c = a - a.mean(-1, keepdim=True)
            a = c * (math.sqrt(self.n) / c.norm(dim=-1, keepdim=True).clamp_min(1e-30))
        return a

    def noise(self, l, a, gen):
        if self.sigma <= 0 or l >= self.L:
            return a
        return a + self.s[l] * torch.randn(a.shape, generator=gen, dtype=a.dtype)

    def pilot_rms(self, m=1 << 12):
        if self.kind == 'relu':
            tab = anet.load_rtable() if os.path.exists(anet.RTABLE) else {}
            key = f"{self.n}_{self.L}_{self.seed}_{self.sigma:g}"
            if key in tab:
                return list(tab[key])
            return anet.pilot_rms(self.W64, self.sigma, m=m, seed=self.seed)
        if self.kind == 'relu_ln':
            return [1.0] * self.L
        g = torch.Generator().manual_seed(99_000_017 + 31 * self.seed + 7 * self.n)
        h = torch.randn(m, self.n, generator=g, dtype=F32)
        r = []
        for l in range(self.L):
            a = self.act(self.pre(l, h))
            r.append(float(a.double().pow(2).mean().sqrt()))
            h = a + float(self.sigma * r[-1]) * torch.randn(m, self.n, generator=g, dtype=F32)
        return r

    def inputs(self, m, gen, law='gauss', aux=None):
        """Input laws. gauss: N(0, I). aniso: N(0, C), C = U diag(lam) U^T, lam_i ~ 1/i normalised to tr C = n.
        point: x0 + 0.1 N(0, I) for one fixed x0 ~ N(0, I) (aux: dict with U/lam/x0, built once per net)."""
        x = torch.randn(m, self.n, generator=gen, dtype=F32)
        if law == 'gauss':
            return x
        if law == 'aniso':
            return (x * aux['sqlam'][None, :]) @ aux['U'].T
        if law == 'point':
            return aux['x0'][None, :] + 0.1 * x
        raise ValueError(law)

    def alt_aux(self):
        rs = np.random.RandomState(777 + self.seed)
        Q, _ = np.linalg.qr(rs.standard_normal((self.n, self.n)))
        lam = 1.0 / np.arange(1, self.n + 1)
        lam = lam * self.n / lam.sum()
        x0 = rs.standard_normal(self.n)
        return dict(U=torch.tensor(Q, dtype=F32), sqlam=torch.tensor(np.sqrt(lam), dtype=F32),
                    x0=torch.tensor(x0, dtype=F32))

    # ---------------------------------------------------------------- Jacobian-vector products
    def jvp(self, l, h, V):
        """Tangent map of the noiseless layer l at states h (T, n) applied to frames V (T, n, k) -> (T, n, k)."""
        z = self.pre(l, h)
        WV = torch.einsum('ij,tjk->tik', self.W[l], V)
        if self.kind == 'tanh':
            return (1 - torch.tanh(z) ** 2)[:, :, None] * WV
        g = (z > 0).to(z.dtype)[:, :, None]
        if self.kind == 'relu':
            return g * WV
        a = torch.relu(z); c = a - a.mean(-1, keepdim=True); nc = c.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        ch = c / nc
        U = g * WV                                                   # d relu
        U = U - U.mean(1, keepdim=True)                              # centring
        U = U - ch[:, :, None] * torch.einsum('ti,tik->tk', ch, U)[:, None, :]
        return (math.sqrt(self.n) / nc)[:, :, None] * U

    # ---------------------------------------------------------------- FLOPs per sampled path (flopscope float32 rates)
    def flops_path(self, L):
        n = self.n
        per = (2 * n - 1) * n + (n if self.b is not None else 0)
        per += 16 * n if self.kind == 'tanh' else n
        if self.kind == 'relu_ln':
            per += 6 * n + 16
        f = 16 * n + L * per
        if self.sigma > 0:
            f += (L - 1) * n * 18
        return f


def mf_standin(net, depth):
    """Mean-field per-neuron mean and variance of h_depth (depth >= 1; depth 0 is the input law), input-independent."""
    import mf
    m = mf.MF(net.kind, net.sw2, net.sb2, net.sigma)
    q = m.q1()
    for _ in range(depth - 1):
        q = m.next_q(q)
    s2 = net.sigma ** 2
    if net.kind == 'tanh':
        Qa = m.Qa(q)
        return 0.0, Qa * (1 + s2)
    if net.kind == 'relu':
        mean = math.sqrt(q / (2 * math.pi))
        return mean, q / 2 - mean ** 2 + s2 * q / 2
    return 0.0, 1.0 + s2
