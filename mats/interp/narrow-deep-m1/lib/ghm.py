"""Gate-history mixture (GHM): a deterministic Gaussian-mixture description of the law of the state of a plain ReLU
net, indexed by realised gate patterns and pruned by mass. Vectorised over components (torch, float64).

State law at layer l: sum_j w_j N(mu_j, C_j) (state h_l after noise). One layer:
  z ~ N(W mu, W C W^T) per component. Units whose pre-activation straddles 0 (|m_i| < KAPPA sd_i) are split one at a time,
  in order of smallest |m_i|/sd_i, into the exact on/off halves of the current Gaussian (single-halfspace truncation,
  closed form), each half re-Gaussianised (moment-matched) and its gate for unit i forced. A piece whose weight is below
  TAU is not split further: its remaining straddling units get exact Gaussian ReLU moments (means, variances, and the
  pair moments by Stein: Cov(ReLU z_i, ReLU z_j) ~ Phi_i C_ij Phi_j for i != j, Cov(z_i, ReLU z_j) = C_ij Phi_j). Forced or
  decided units are linear (on) or zero (off). Then noise: C += s_l^2 I.
FLOPs are counted with flopscope weights (matmul M N (2K-1), cdf 96, pdf 54, sqrt/div 1).
"""
import math
import torch

DT = torch.float64
F_CDF, F_PDF = 96, 54
KAPPA = 4.0
SQ2PI = math.sqrt(2 * math.pi)


def ncdf(x):
    return torch.special.ndtr(x)


def npdf(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


class Flops:
    def __init__(self):
        self.n = 0.0

    def add(self, k):
        self.n += float(k)


def _trunc_half(m, Cz, i, on, fl):
    """Exact first two moments of z ~ N(m, Cz) (batched: m (B,n), Cz (B,n,n)) restricted to z_i > 0 (on) or z_i <= 0.
    i: (B,) unit indices. Returns mass (B,), mean (B,n), cov (B,n,n)."""
    B, n = m.shape
    ar = torch.arange(B)
    mi = m[ar, i]
    vi = Cz[ar, i, i].clamp_min(1e-300)
    si = vi.sqrt()
    alpha = mi / si if on else -mi / si                 # standardized distance into the kept half
    Phi = ncdf(alpha).clamp_min(1e-300)
    phi = npdf(alpha)
    lam = phi / Phi                                     # inverse Mills ratio
    # E[z_i | kept] = m_i + s * lam (on) or m_i - s * lam (off)
    shift_i = si * lam if on else -si * lam
    var_i = vi * (1 - alpha * lam - lam * lam).clamp_min(0)
    c = Cz[ar, :, i]                                    # (B, n) column i
    mean = m + c * (shift_i / vi)[:, None]
    cov = Cz + c[:, :, None] * c[:, None, :] * ((var_i - vi) / (vi * vi))[:, None, None]
    fl.add(B * (F_CDF + F_PDF + 20 + 2 * n + 3 * n * n))
    return Phi, mean, cov


def relu_layer(w, mu, C, W, s2, tau, fl, kappa=KAPPA, max_comp=None):
    """One plain layer for a batch of components. w (B,), mu (B,n), C (B,n,n); W (n,n); s2 noise variance (scalar,
    or None for no noise). Returns new (w, mu, C) of the state after noise."""
    B, n = mu.shape
    m = mu @ W.T                                        # (B, n)
    Cz = W @ C @ W.T                                    # (B, n, n)
    fl.add(B * (n * (2 * n - 1) + 2 * n * n * (2 * n - 1)))
    forced = torch.zeros(B, n, dtype=torch.int8)        # 0 = free, 1 = forced on, -1 = forced off
    done_w, done_m, done_C, done_f = [], [], [], []
    cur_w, cur_m, cur_C, cur_f = w, m, Cz, forced
    for _ in range(n + 1):
        if cur_w.numel() == 0:
            break
        sd = torch.diagonal(cur_C, dim1=1, dim2=2).clamp_min(1e-300).sqrt()
        t = cur_m.abs() / sd
        t = torch.where(cur_f != 0, torch.full_like(t, float('inf')), t)
        fl.add(cur_w.numel() * 4 * n)
        tmin, imin = t.min(1)
        split = (tmin < kappa) & (cur_w >= tau)
        if max_comp is not None and (split.sum() * 2 + sum(x.numel() for x in done_w)) > max_comp:
            split = split & False
        keep = ~split
        done_w.append(cur_w[keep]); done_m.append(cur_m[keep]); done_C.append(cur_C[keep]); done_f.append(cur_f[keep])
        if split.sum() == 0:
            cur_w = cur_w[:0]
            break
        sw, sm, sC, sf, si = cur_w[split], cur_m[split], cur_C[split], cur_f[split], imin[split]
        pon, mon, Con = _trunc_half(sm, sC, si, True, fl)
        poff, moff, Coff = _trunc_half(sm, sC, si, False, fl)
        fon = sf.clone(); fon[torch.arange(len(si)), si] = 1
        foff = sf.clone(); foff[torch.arange(len(si)), si] = -1
        cur_w = torch.cat([sw * pon, sw * poff]); cur_m = torch.cat([mon, moff]); cur_C = torch.cat([Con, Coff])
        cur_f = torch.cat([fon, foff])
        fl.add(len(si) * 2 * 2)
    w2 = torch.cat(done_w); m2 = torch.cat(done_m); C2 = torch.cat(done_C); f2 = torch.cat(done_f)
    # drop zero-weight pieces
    nz = w2 > 0
    w2, m2, C2, f2 = w2[nz], m2[nz], C2[nz], f2[nz]
    # ReLU per piece: forced/decided units linear or zero; straddling free units by exact Gaussian ReLU moments
    sd = torch.diagonal(C2, dim1=1, dim2=2).clamp_min(1e-300).sqrt()
    a = m2 / sd
    Phi = ncdf(a); phi = npdf(a)
    # effective "on-probability" per unit: forced on 1, forced off 0, free: Phi (exact mean via relu moments)
    pon = torch.where(f2 == 1, torch.ones_like(Phi), torch.where(f2 == -1, torch.zeros_like(Phi), Phi))
    relu_mean = m2 * Phi + sd * phi
    relu_e2 = (m2 * m2 + sd * sd) * Phi + m2 * sd * phi
    mean_out = torch.where(f2 == 1, m2, torch.where(f2 == -1, torch.zeros_like(m2), relu_mean))
    var_out = torch.where(f2 == 1, sd * sd, torch.where(f2 == -1, torch.zeros_like(m2),
                                                        (relu_e2 - relu_mean ** 2).clamp_min(0)))
    Cout = pon[:, :, None] * C2 * pon[:, None, :]
    idx = torch.arange(n)
    Cout[:, idx, idx] = var_out
    fl.add(len(w2) * (n * (F_CDF + F_PDF + 15) + 3 * n * n))
    if s2 is not None:
        Cout = Cout + torch.eye(n, dtype=DT) * s2
        fl.add(len(w2) * n)
    return w2, mean_out, Cout


def ghm_forward(net, c, tau, kappa=KAPPA, max_comp=2_000_000, keep_layers=None, verbose=False):
    """GHM from the input N(0, I) through layers 1..c (state after noise of layer c). Returns dict layer -> (w, mu, C),
    and the FLOP count."""
    n = net.n
    fl = Flops()
    w = torch.ones(1, dtype=DT); mu = torch.zeros(1, n, dtype=DT); C = torch.eye(n, dtype=DT)[None].clone()
    out = {}
    for l in range(c):
        s2 = float(net.s[l]) ** 2 if net.sigma > 0 else None
        w, mu, C = relu_layer(w, mu, C, net.W[l], s2, tau, fl, kappa, max_comp)
        if keep_layers is None or (l + 1) in keep_layers:
            out[l + 1] = (w.clone(), mu.clone(), C.clone(), fl.n)
        if verbose:
            print(f"  GHM tau={tau:g} layer {l+1}: {len(w)} components, weight {float(w.sum()):.6f}, flops {fl.n:.3g}",
                  flush=True)
    return out, fl.n


def mixture_mean(w, mu):
    return (w[:, None] * mu).sum(0) / w.sum()


def sample_mixture(w, mu, C, P, gen):
    """P draws from sum_j w_j N(mu_j, C_j) (for oracle evaluation only)."""
    p = (w / w.sum()).clamp_min(0)
    idx = torch.multinomial(p, P, replacement=True, generator=gen)
    Cs = 0.5 * (C + C.transpose(1, 2))
    ev, U = torch.linalg.eigh(Cs)
    Lh = U * ev.clamp_min(0).sqrt()[:, None, :]
    z = torch.randn(P, mu.shape[1], generator=gen, dtype=DT)
    return mu[idx] + (Lh[idx] @ z[:, :, None])[:, :, 0]
