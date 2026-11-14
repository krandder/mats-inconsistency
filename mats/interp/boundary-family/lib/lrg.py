"""Log-radial Gaussian (LRG) propagation: the scale of the state is carried exactly through ReLU's positive
homogeneity, the direction by a Gaussian (boundary-family lane, prototype).

State at layer l (pre-activation): z = exp(rho) (u + d), u a unit reference vector (the normalized mean direction),
d orthogonal to u, and (rho, d) jointly Gaussian: E rho, V rho, E d, Cov d, c = Cov(d, rho).
ReLU(z) = exp(rho) ReLU(u + d) exactly (homogeneity). One step:
  y = W ReLU(u + d)   (moments exact under the Gaussian d: rectified means, bivariate-normal covariances)
  Cov(rho, ReLU(u_i + d_i)) = c_i Phi(t_i)   (Stein's lemma)
  z' = exp(rho) y + W s eps  (noise folded into y with the factor E exp(-2 rho))
  u' = E y / |E y|; rho' = rho + log(u'.y); d' = P' y / (u'.y), by the delta method to second order in the mean.
Readout: E[a_i] = E[exp(rho) ReLU(u_i + d_i)] = exp(E rho + V rho / 2) E[ReLU(u_i + d_i + c_i)] (Gaussian tilting),
exact under the family.
The first layer z_1 = W_1 x is exactly Gaussian; the family starts at layer l_switch (default 2) from the exact
Gaussian closure's moments (u = mean direction, rho = log(u.z) moment-matched as a lognormal).
"""
import math
import torch
from gauss_relu import relu_moments, ncdf, npdf, FLOPS_PAIR, FLOPS_NEURON

DT = torch.float64


def relu_mean(m, s):
    t = m / s
    return s * (t * ncdf(t) + npdf(t))


class LRG:
    def __init__(self, u, Er, Vr, Ed, Cd, c):
        self.u, self.Er, self.Vr, self.Ed, self.Cd, self.c = u, Er, Vr, Ed, Cd, c

    @staticmethod
    def from_gauss(m, S):
        """Moment-matched LRG of N(m, S) (first order in the fluctuation, lognormal radial part)."""
        # first-order chart (fixed after the GPT-6 critique): (rho, d) = linear image of z - m, so the joint covariance
        # [u, P]^T S [u, P] / |m|^2 is PSD by construction; second-order means as before
        nm = m.norm(); u = m / nm
        n = m.shape[0]
        P = torch.eye(n, dtype=DT) - torch.outer(u, u)
        Su = S @ u
        Vr = float(u @ Su) / float(nm) ** 2
        Er = math.log(float(nm)) - Vr / 2
        Cd = P @ S @ P / float(nm) ** 2
        c = P @ Su / float(nm) ** 2
        Ed = -P @ Su / float(nm) ** 2                       # second-order mean of P z / (u.z)
        J = torch.cat([u[None, :], P], 0) / float(nm)
        ev = torch.linalg.eigvalsh(J @ S @ J.T)
        assert float(ev.min()) > -1e-9 * float(ev.max()), 'LRG chart not PSD'
        return LRG(u, Er, Vr, Ed, Cd, c)

    def readout(self):
        sd = self.Cd.diagonal().clamp_min(1e-300).sqrt()
        return math.exp(self.Er + self.Vr / 2) * relu_mean(self.u + self.Ed + self.c, sd)

    def push(self, W, s2):
        n = W.shape[0]
        md = self.u + self.Ed
        mu, Ca, npairs = relu_moments(md, self.Cd)
        sd = self.Cd.diagonal().clamp_min(1e-300).sqrt()
        Phi = ncdf(md / sd)
        ybar = W @ mu
        noise = s2 * math.exp(-2 * self.Er + 2 * self.Vr)
        Cy = W @ (Ca + noise * torch.eye(n, dtype=DT)) @ W.T
        Cy = 0.5 * (Cy + Cy.T)
        cyr = W @ (self.c * Phi)                              # Cov(y, rho)
        g = ybar.norm(); u2 = ybar / g; gf = float(g)
        P2 = torch.eye(n, dtype=DT) - torch.outer(u2, u2)
        Cyu = Cy @ u2
        vu = float(u2 @ Cyu)
        Vr2 = self.Vr + vu / gf ** 2 + 2 * float(u2 @ cyr) / gf
        Er2 = self.Er + math.log(gf) - vu / (2 * gf ** 2)
        Cd2 = P2 @ Cy @ P2 / gf ** 2
        c2 = P2 @ (cyr + Cyu / gf) / gf
        Ed2 = -P2 @ Cyu / gf ** 2
        fl = npairs * FLOPS_PAIR + n * FLOPS_NEURON + 4 * n ** 3 + 6 * n * n
        return LRG(u2, Er2, max(Vr2, 1e-12), Ed2, 0.5 * (Cd2 + Cd2.T), c2), fl


def run(net, L, l_switch=2, keep=True):
    """Returns (estimate of E[a_L], flops, per-layer estimates)."""
    n = net.n
    W = [w.to(DT) for w in net.W[:L]]
    s = [float(v) for v in net.s[:L]]
    m = torch.zeros(n, dtype=DT); S = W[0] @ W[0].T
    fl = n * n * (2 * n - 1)
    per = []
    st = None
    for l in range(L):                       # state describes z_{l+1}
        if st is None and l + 1 >= l_switch:
            st = LRG.from_gauss(m, S)
        if st is None:
            sd = S.diagonal().sqrt()
            per.append(relu_mean(m, sd))
            if l == L - 1:
                break
            mu, Ca, npairs = relu_moments(m, S)
            m = W[l + 1] @ mu
            S = W[l + 1] @ (Ca + s[l] ** 2 * torch.eye(n, dtype=DT)) @ W[l + 1].T
            S = 0.5 * (S + S.T)
            fl += npairs * FLOPS_PAIR + n * FLOPS_NEURON + 4 * n ** 3
        else:
            per.append(st.readout())
            if l == L - 1:
                break
            st, f = st.push(W[l + 1], s[l] ** 2)
            fl += f
    return per[-1], fl, per
