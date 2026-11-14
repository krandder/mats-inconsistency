"""Phase R core: noisy random LN-MLPs, exact sampling, ARC-style K=2 moment propagation, Jacobians, FLOP counts.

Network R(d, L, act, g, sigma, seed):
    h_0 = x ~ N(0, I_d);  z_l = W_l h_{l-1} (W_l iid N(0, g^2/d), no bias);  s_l = LN(act(z_l));  h_l = s_l + sigma eps_l
    LN(a) = sqrt(d) (a - mean a) / |a - mean a|   (no affine; s_l is on the sphere of radius sqrt(d) in the centred subspace)
    outputs y = W_out h_L (W_out iid N(0, 1/d)); rare-event readout y1 = w . h_L, w = W_out[0] / |W_out[0]|.
"""
import math
import numpy as np
import torch

torch.set_num_threads(1)
LOG2 = math.log(2.0)
DT = torch.float64


def act_fns(act):
    if act == 'tanh':
        return torch.tanh, (lambda z: 1 - torch.tanh(z) ** 2)
    if act == 'relu':
        return torch.relu, (lambda z: (z > 0).to(z.dtype))
    raise ValueError(act)


class RNet:
    def __init__(self, d, L, act='tanh', g=0.75, sigma=0.3, seed=0, dtype=DT, bias=0.0, use_ln=True):
        self.d, self.L, self.act, self.g, self.sigma, self.seed, self.bias = d, L, act, g, sigma, seed, bias
        self.use_ln = use_ln                      # False: no-LN control (ARC's architecture + noise), s = act(z)
        gen = torch.Generator().manual_seed(1000 + seed)
        self.W = [torch.randn(d, d, generator=gen, dtype=DT) * (g / math.sqrt(d)) for _ in range(L)]
        self.Wout = torch.randn(d, d, generator=gen, dtype=DT) / math.sqrt(d)
        self.w = self.Wout[0] / self.Wout[0].norm()
        self.b = [torch.randn(d, generator=gen, dtype=DT) * bias for _ in range(L)]   # drawn last: W's unchanged
        self.phi, self.dphi = act_fns(act)
        self.sqd = math.sqrt(d)
        self.set_dtype(dtype)

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.Wd = [W.to(dtype) for W in self.W]; self.Woutd = self.Wout.to(dtype); self.wd = self.w.to(dtype); self.bd = [b.to(dtype) for b in self.b]

    def name(self):
        return (f"d{self.d}_L{self.L}_{self.act}{self.g:g}_s{self.sigma:g}_n{self.seed}" + (f"_b{self.bias:g}" if self.bias else "")
                + ("" if self.use_ln else "_noLN"))

    # ---------------------------------------------------------------- maps
    def ln(self, a):
        if not self.use_ln:
            return a
        c = a - a.mean(-1, keepdim=True)
        return c * (self.sqd / c.norm(dim=-1, keepdim=True).clamp_min(1e-30))

    def f(self, l, h):
        """Noiseless map of layer l (1-based) applied to h (n, d)."""
        return self.ln(self.phi(h @ self.Wd[l - 1].T + self.bd[l - 1]))

    def jac(self, l, h):
        """Jacobian ds_l/dh_{l-1} at h (n, d) -> (n, d, d)."""
        z = h @ self.Wd[l - 1].T + self.bd[l - 1]; a = self.phi(z); c = a - a.mean(-1, keepdim=True)
        if not self.use_ln:
            return self.dphi(z)[:, :, None] * self.Wd[l - 1][None]
        n = c.norm(dim=-1, keepdim=True).clamp_min(1e-30); ch = c / n
        d = self.d
        P = torch.eye(d, dtype=h.dtype)[None] - ch[:, :, None] * ch[:, None, :] - 1.0 / d
        return (self.sqd / n)[:, :, None] * (P * self.dphi(z)[:, None, :]) @ self.Wd[l - 1]

    def forward(self, x, gen, layers=False, noise_last=True):
        h = x; out = []
        for l in range(1, self.L + 1):
            s = self.f(l, h)
            h = s + self.sigma * torch.randn(s.shape, generator=gen, dtype=s.dtype) if (noise_last or l < self.L) else s
            if layers: out.append(h)
        return out if layers else h

    def sample_inputs(self, n, gen):
        return torch.randn(n, self.d, generator=gen, dtype=self.dtype)

    def mc_outputs(self, n, gen, chunk=1 << 15, rb=True):
        """Plain MC of the d outputs. rb=True: Rao-Blackwellise the last noise (E[y | s_L] = W_out s_L).
        Returns sum and sum of squares (float64) over n samples, plus the y1 = w.h_L values (rb: w.s_L)."""
        S1 = torch.zeros(self.d, dtype=DT); S2 = torch.zeros(self.d, dtype=DT); ys = []
        done = 0
        while done < n:
            m = min(chunk, n - done)
            hL = self.forward(self.sample_inputs(m, gen), gen, noise_last=not rb)
            y = (hL @ self.Woutd.T).to(DT)
            S1 += y.sum(0); S2 += (y ** 2).sum(0); ys.append((hL @ self.wd).to(DT))
            done += m
        return S1, S2, torch.cat(ys)

    # ---------------------------------------------------------------- FLOPs (analytic, multiply-add = 2)
    def flops_sample(self):
        d = self.d
        return self.L * (2 * d * d + 12 * d) + 2 * d * d   # L matvecs + elementwise/LN (~12 d) + readout of d outputs

    def flops_k2(self, Kh=20, ngh=48):
        d = self.d
        per = 4 * d ** 3 + ngh * d * (Kh + 6) + 5 * Kh * d * d + 6 * d * d
        return self.L * per + 2 * d * d


# -------------------------------------------------------------------- ARC-style K = 2 moment propagation (deductive)
_GH = {}


def gh(n):
    if n not in _GH:
        x, w = np.polynomial.hermite_e.hermegauss(n); _GH[n] = (torch.tensor(x, dtype=DT), torch.tensor(w / w.sum(), dtype=DT))
    return _GH[n]


def hermite_table(x, K):
    """He_k(x) for k = 0..K (probabilists'), x (n,) -> (K+1, n)."""
    H = [torch.ones_like(x), x.clone()]
    for k in range(1, K):
        H.append(x * H[k] - k * H[k - 1])
    return torch.stack(H[:K + 1])


def act_moments_k2(phi, m, C, Kh=20, ngh=48):
    """Gaussian z ~ N(m, C): E[phi(z)] and Cov(phi(z)) via per-neuron Hermite expansion + Mehler (exact diagonal)."""
    x, wq = gh(ngh); sd = C.diagonal().clamp_min(1e-300).sqrt()
    Z = m[:, None] + sd[:, None] * x[None]                 # (d, ngh)
    A = phi(Z)
    He = hermite_table(x, Kh)                              # (Kh+1, ngh)
    c = (A * wq[None]) @ He.T                              # (d, Kh+1): c_ik = E[phi He_k]
    Ea = c[:, 0]
    var = (A ** 2 * wq[None]).sum(1) - Ea ** 2
    rho = C / (sd[:, None] * sd[None, :])
    rho = rho.clamp(-1, 1)
    cov = torch.zeros_like(C); rk = torch.ones_like(C); fact = 1.0
    for k in range(1, Kh + 1):
        rk = rk * rho; fact *= k
        cov += (c[:, k:k + 1] * c[None, :, k]) * rk / fact
    cov.diagonal().copy_(var)
    dphi_mean = None
    return Ea, cov


def k2_propagate(net, Kh=20, ngh=48, ln_corr=True, keep=False, psd=False):
    """Deductive K=2 chain: returns list of (mu_l, Sigma_l) of h_l for l = 1..L (or only the last).
    psd=True: covariance of s taken as the centred activation covariance / n0, rescaled so tr + |mu|^2 = d (always PSD;
    the default psd=False forms M2 - mu mu^T, which with the LN mean correction can lose positive definiteness)."""
    d = net.d; P = torch.eye(d, dtype=DT) - 1.0 / d
    mu = torch.zeros(d, dtype=DT); S = torch.eye(d, dtype=DT); out = []
    for l in range(1, net.L + 1):
        W = net.W[l - 1]
        m = W @ mu + net.b[l - 1]; C = W @ S @ W.T
        Ea, Ca = act_moments_k2(net.phi, m, C, Kh, ngh)
        if not getattr(net, 'use_ln', True):                # no-LN control: s = act(z)
            mu = Ea; S = Ca + net.sigma ** 2 * torch.eye(d, dtype=DT)
            if keep: out.append((mu.clone(), S.clone()))
            continue
        mc = P @ Ea; Cc = P @ Ca @ P                        # centred activations c = P a
        n0 = (Cc.trace() + mc @ mc) / d                     # E|c|^2 / d
        mu_s = mc / n0.sqrt()
        if ln_corr:                                         # 2nd-order delta method for E[c / sqrt(|c|^2/d)] under Gaussian c
            covN = 2.0 * (Cc @ mc) / d                       # Cov(c, N), N = |c|^2/d
            varN = (2.0 * (Cc * Cc).sum() + 4.0 * mc @ Cc @ mc) / d ** 2
            mu_s = (mc - covN / (2 * n0) + 3.0 * mc * varN / (8 * n0 ** 2)) / n0.sqrt()
        if psd:
            Cs = Cc / n0; tgt = d - float(mu_s @ mu_s)
            if tgt > 0:
                Cs = Cs * (tgt / Cs.trace())
            S = Cs + net.sigma ** 2 * torch.eye(d, dtype=DT)
        else:
            M2 = (Cc + torch.outer(mc, mc)) / n0
            M2 = M2 * (d / M2.trace())                      # |s|^2 = d exactly
            S = M2 - torch.outer(mu_s, mu_s) + net.sigma ** 2 * torch.eye(d, dtype=DT)
        mu = mu_s
        if keep: out.append((mu.clone(), S.clone()))
    return out if keep else (mu, S)


def stein_jacobian(net, l, mu, S, Kh=20, ngh=48):
    """Statistical linearisation of layer l under h_{l-1} ~ N(mu, S): A = Cov(s_l, h_{l-1}) S^{-1} = E[ds/dh] (Stein),
    mean-field LN. Returns A (d, d)."""
    d = net.d; W = net.W[l - 1]
    m = W @ mu + net.b[l - 1]; C = W @ S @ W.T
    x, wq = gh(ngh); sd = C.diagonal().clamp_min(1e-300).sqrt()
    Z = m[:, None] + sd[:, None] * x[None]
    Edphi = (net.dphi(Z) * wq[None]).sum(1)
    if not getattr(net, 'use_ln', True):
        return Edphi[:, None] * W
    Ea, Ca = act_moments_k2(net.phi, m, C, Kh, ngh)
    P = torch.eye(d, dtype=DT) - 1.0 / d
    mc = P @ Ea; Cc = P @ Ca @ P; n0 = (Cc.trace() + mc @ mc) / d
    M2 = (Cc + torch.outer(mc, mc)) / n0; M2 = M2 * (d / M2.trace())
    R = torch.eye(d, dtype=DT) - M2 / d - 1.0 / d           # E[I - s s^T/d - 11^T/d]
    return (R / n0.sqrt()) @ (Edphi[:, None] * W)
