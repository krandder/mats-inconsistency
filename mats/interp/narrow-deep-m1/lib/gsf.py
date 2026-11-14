"""Deterministic Gaussian-sum filter (GSF) for a plain noisy ReLU net, started from a narrow Gaussian (deep part).
Components (w, mu, C) of the state. One layer: z = W h per component; every unit whose pre-activation straddles 0
(|m_i| < KAPPA sd_i) is resolved by narrowing splits: z_i ~ N(m_i, v_i) is replaced by sum_j beta_j N(m_i + c_j sd_i,
tau^2 v_i) (Gauss-Hermite nodes c_j for N(0, 1 - tau^2): the mixture matches the moments of N(0,1) up to order 2M-1;
the rest of the component conditioned exactly), repeated up to DEPTH times on the sub-components that still straddle;
then exact Gaussian ReLU moments per component (means and variances exact, Stein cross-covariances), exact Gaussian
noise (C += s^2 I), and merging (moment-preserving) down to KMAX components by pairing the closest components.
FLOPs counted with flopscope weights."""
import math
import numpy as np
import torch

DT = torch.float64
F_CDF, F_PDF = 96, 54
SQ2PI = math.sqrt(2 * math.pi)


class Flops:
    def __init__(self):
        self.n = 0.0

    def add(self, k):
        self.n += float(k)


def gh_rule(M, tau):
    x, w = np.polynomial.hermite_e.hermegauss(M)          # probabilists' Hermite: N(0,1)
    w = w / w.sum()
    return torch.tensor(x * math.sqrt(1 - tau * tau), dtype=DT), torch.tensor(w, dtype=DT)


def split_unit(w, m, Cz, i, rule_x, rule_w, tau, fl):
    """narrowing split of components (batch) along pre-activation unit i (per component index i: (B,))."""
    B, n = m.shape
    ar = torch.arange(B)
    vi = Cz[ar, i, i].clamp_min(1e-300); si = vi.sqrt()
    c = Cz[ar, :, i]                                          # (B, n)
    Cn = Cz - (1 - tau * tau) * c[:, :, None] * c[:, None, :] / vi[:, None, None]
    M = len(rule_x)
    ws, ms, Cs = [], [], []
    for j in range(M):
        ws.append(w * rule_w[j])
        ms.append(m + c * (rule_x[j] / si)[:, None])
        Cs.append(Cn)
    fl.add(B * (3 * n * n + M * 3 * n))
    return torch.cat(ws), torch.cat(ms), torch.cat(Cs)


def relu_moments(w, m, Cz, fl):
    B, n = m.shape
    sd = torch.diagonal(Cz, dim1=1, dim2=2).clamp_min(1e-300).sqrt()
    a = m / sd
    Phi = torch.special.ndtr(a); phi = torch.exp(-0.5 * a * a) / SQ2PI
    mu = m * Phi + sd * phi
    e2 = (m * m + sd * sd) * Phi + m * sd * phi
    C = Phi[:, :, None] * Cz * Phi[:, None, :]
    idx = torch.arange(n)
    C[:, idx, idx] = (e2 - mu * mu).clamp_min(0)
    fl.add(B * (n * (F_CDF + F_PDF + 15) + 3 * n * n))
    return mu, C


def merge(w, mu, C, kmax, fl):
    """greedy moment-preserving merging down to kmax components (closest pairs by weighted squared distance of means
    in the pooled metric)."""
    while len(w) > kmax:
        B, n = mu.shape
        # pooled whitening by the weighted mean covariance (cheap metric)
        Cp = (w[:, None, None] * C).sum(0) / w.sum() + 1e-300 * torch.eye(n, dtype=DT)
        ev, U = torch.linalg.eigh(0.5 * (Cp + Cp.T))
        A = U * ev.clamp_min(ev.max() * 1e-12).rsqrt()
        y = mu @ A
        # merge the lightest components first: each with its nearest neighbour
        nmerge = len(w) - kmax
        order = torch.argsort(w)
        used = torch.zeros(B, dtype=torch.bool)
        newW, newM, newC = [], [], []
        D = torch.cdist(y, y)
        D.fill_diagonal_(float('inf'))
        cnt = 0
        for a_ in order.tolist():
            if cnt >= nmerge:
                break
            if used[a_]:
                continue
            d = D[a_].clone(); d[used] = float('inf')
            b_ = int(torch.argmin(d))
            if not torch.isfinite(d[b_]):
                continue
            used[a_] = True; used[b_] = True
            wa, wb = w[a_], w[b_]; ws_ = wa + wb
            m_ = (wa * mu[a_] + wb * mu[b_]) / ws_
            da, db = mu[a_] - m_, mu[b_] - m_
            C_ = (wa * (C[a_] + torch.outer(da, da)) + wb * (C[b_] + torch.outer(db, db))) / ws_
            newW.append(ws_); newM.append(m_); newC.append(C_)
            cnt += 1
        keep = ~used
        w = torch.cat([w[keep], torch.stack(newW)]); mu = torch.cat([mu[keep], torch.stack(newM)])
        C = torch.cat([C[keep], torch.stack(newC)])
        fl.add(B * B * (n + 2) + B * n * n + cnt * 6 * n * n)
    return w, mu, C


def gsf_run(net, mu0, C0, l_from, l_to, fl, kappa=3.0, M=5, tau=0.5, depth=2, kmax=48, wmin=1e-6):
    """propagate N(mu0, C0) of the state after layer l_from through layers l_from+1..l_to; noise after every layer
    except the last layer of the net. Returns the mixture mean of the final state."""
    n = net.n
    rx, rw = gh_rule(M, tau)
    w = torch.ones(1, dtype=DT); mu = mu0[None].clone(); C = C0[None].clone()
    for l in range(l_from, l_to):
        W = net.W[l]
        m = mu @ W.T
        Cz = W @ C @ W.T
        fl.add(len(w) * (n * (2 * n - 1) + 2 * n * n * (2 * n - 1)))
        for d in range(depth * n):
            sd = torch.diagonal(Cz, dim1=1, dim2=2).clamp_min(1e-300).sqrt()
            t = m.abs() / sd
            tmin, imin = t.min(1)
            sp = (tmin < kappa) & (w > wmin)
            if d >= depth * 1 and False:
                break
            if sp.sum() == 0:
                break
            w2, m2, C2 = split_unit(w[sp], m[sp], Cz[sp], imin[sp], rx, rw, tau, fl)
            w = torch.cat([w[~sp], w2]); m = torch.cat([m[~sp], m2]); Cz = torch.cat([Cz[~sp], C2])
            if len(w) > 8 * kmax:
                break
        mu, C = relu_moments(w, m, Cz, fl)
        if l < net.L - 1 and net.sigma > 0:
            C = C + torch.eye(n, dtype=DT) * float(net.s[l]) ** 2
            fl.add(len(w) * n)
        if len(w) > kmax:
            w, mu, C = merge(w, mu, C, kmax, fl)
    return (w[:, None] * mu).sum(0) / w.sum(), len(w)
