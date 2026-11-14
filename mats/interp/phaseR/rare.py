"""Rare output events A_t = {w . h_L > t} of a noisy LN-MLP (phase R, tasks 2-3).

Twisted SMC (fully adapted auxiliary particle filter) with probit-linear twists psi_l(h) = Phi_bar(tau_l - alpha_l . h),
l = 0..L-1 (psi_0 acts on the input x); the last layer is exact: P(A | h_{L-1}) = Phi_bar((t - w.f_L(h_{L-1}))/sigma).
Unbiased for p(A) for any twists. psi = None -> plain MC with the last layer Rao-Blackwellised.

Twist sources: 'bulk'  statistical linearisation of the K=2 forward chain (regression of y on h_l under the chain);
               'tail'  linearisation around the most likely path to A (instanton), deductive from the weights;
               'iapf'  probit-linear regression on simulated particles (iterated auxiliary particle filter; black-box).
Also: mean-shift importance sampling at the instanton (no resampling), and the K=2 Gaussian tail.
"""
import math
import numpy as np
import torch
import rnet as R
from fam import ndtri_log

DT = torch.float64
LOG2 = math.log(2)
SQ2 = math.sqrt(2)


def log_sf(a):
    return torch.special.log_ndtr(-a)


def trunc_normal_above(mean, sd, tau, gen):
    """v ~ N(mean, sd^2) conditioned on v > tau (robust in both tails)."""
    a = (tau - mean) / sd
    ls = log_sf(a)                                         # log P(z > a)
    lu = torch.rand(mean.shape, generator=gen, dtype=DT).clamp_min(1e-300).log()
    lp = lu + ls                                           # log p, p uniform on (0, Phi_bar(a))
    lo = lp < math.log(0.5)
    z = torch.empty_like(mean)
    z[lo] = -ndtri_log(lp[lo])                              # Phi_bar^{-1}(p) = -Phi^{-1}(p)
    if (~lo).any():
        l1m = torch.log((-torch.expm1(lp[~lo])).clamp_min(1e-300))
        z[~lo] = ndtri_log(l1m)
    z = torch.maximum(z, a)                                # guard round-off
    return mean + sd * z


class Twists:
    def __init__(self, tau, alpha):
        self.tau = [torch.as_tensor(t, dtype=DT) for t in tau]; self.alpha = [a.to(DT) for a in alpha]   # l = 0..L-1


def smc(net, t, tw, N, gen, resample=True):
    """Returns log p_hat (nats) and FLOPs. tw: Twists or None."""
    d, L, s = net.d, net.L, net.sigma
    flops = 0
    if tw is None:
        x = torch.randn(N, d, generator=gen, dtype=DT); lZ = 0.0
        lpsi_prev = torch.zeros(N, dtype=DT)
    else:
        al, ta = tw.alpha[0], tw.tau[0]; na2 = float(al @ al)
        lZ = float(log_sf(ta / math.sqrt(1 + na2)))
        v = trunc_normal_above(torch.zeros(N, dtype=DT), torch.full((N,), math.sqrt(1 + na2), dtype=DT), ta.expand(N), gen)
        xi = torch.randn(N, d, generator=gen, dtype=DT); ah = al / math.sqrt(na2)
        x = (al[None] * (v / (1 + na2))[:, None] + xi - (1 - 1 / math.sqrt(1 + na2)) * ah[None] * (xi @ ah)[:, None])
        lpsi_prev = log_sf(ta - x @ al)
        flops += N * 8 * d
    h = x
    for l in range(1, L + 1):
        m = net.f(l, h); flops += N * (2 * d * d + 12 * d)
        if l == L:
            lG = log_sf((t - m @ net.w) / s) - lpsi_prev
            lZ += float(torch.logsumexp(lG, 0)) - math.log(N)
            return lZ, flops
        if tw is None:
            h = m + s * torch.randn(N, d, generator=gen, dtype=DT)
            continue
        al, ta = tw.alpha[l], tw.tau[l]; na2 = float(al @ al); vs = math.sqrt(s * s * na2 + 1)
        am = m @ al
        lK = log_sf((ta - am) / vs)                         # log (K_l psi_l)(h_{l-1})
        lG = lK - lpsi_prev
        lZ += float(torch.logsumexp(lG, 0)) - math.log(N)
        if resample:
            idx = torch.multinomial(torch.softmax(lG, 0), N, replacement=True, generator=gen)
            m, am = m[idx], am[idx]
        else:
            raise NotImplementedError
        v = trunc_normal_above(am, torch.full((N,), vs, dtype=DT), ta.expand(N), gen)
        xi = torch.randn(N, d, generator=gen, dtype=DT); ah = al / math.sqrt(na2)
        mean = m + (s * s) * al[None] * ((v - am) / (vs * vs))[:, None]
        h = mean + s * (xi - (1 - 1 / vs) * ah[None] * (xi @ ah)[:, None])
        lpsi_prev = log_sf(ta - h @ al)
        flops += N * 14 * d
    raise RuntimeError


# ------------------------------------------------------------------ twist constructions
def twists_bulk(net, t):
    """Regression of y = w.h_L on h_l under the K=2 chain with statistically linearised layers (Stein Jacobians)."""
    chain = R.k2_propagate(net, keep=True)
    mus = [torch.zeros(net.d, dtype=DT)] + [c[0] for c in chain]
    Ss = [torch.eye(net.d, dtype=DT)] + [c[1] for c in chain]
    muy = float(net.w @ mus[-1]); vy = float(net.w @ Ss[-1] @ net.w)
    beta = net.w.clone(); taus, alphas = [None] * net.L, [None] * net.L
    for l in range(net.L - 1, -1, -1):               # psi_l on h_l, l = L-1..0
        A = R.stein_jacobian(net, l + 1, mus[l], Ss[l])
        beta = A.T @ beta
        v = max(vy - float(beta @ Ss[l] @ beta), net.sigma ** 2)
        sd = math.sqrt(v)
        alphas[l] = beta / sd; taus[l] = (t - muy + float(beta @ mus[l])) / sd
    flops = net.flops_k2() + net.L * (4 * net.d ** 3 + 2 * net.d ** 2)
    return Twists(taus, alphas), flops


def path_from_noise(net, xi):
    """xi = (x, eps_1..eps_{L-1}) stacked (L, d) -> h_0..h_{L-1} (list)."""
    hs = [xi[0]]
    for l in range(1, net.L):
        hs.append(net.f(l, hs[-1][None])[0] + net.sigma * xi[l])
    return hs


def instanton(net, t, iters=300, xi0=None, nstart=4):
    """Most likely noise path to A, last layer marginalised exactly. LN without bias makes layer 1 see only the direction
    of x, so x = sqrt(d) u/|u| (typical norm, direction free, no cost) and the cost is 1/2 sum_l |eps_l|^2:
    maximise -1/2 sum |eps|^2 + log Phi_bar((t - w.f_L(h_{L-1}))/sigma). L-BFGS from several starts.
    Returns xi* (L, d) with xi*[0] = x*, objective, FLOPs."""
    d, L = net.d, net.L
    best = None; count = [0]
    starts = [torch.randn(d, generator=torch.Generator().manual_seed(17 + k), dtype=DT) for k in range(nstart)]
    if xi0 is not None:
        starts = [xi0[0].clone()] + starts[:max(1, nstart // 2)]
    for st in starts:
        u = st.clone().requires_grad_(True)
        eps = (torch.zeros(L - 1, d, dtype=DT) if xi0 is None else xi0[1:].clone()).requires_grad_(True)
        opt = torch.optim.LBFGS([u, eps], lr=1, max_iter=iters, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            x = math.sqrt(d) * u / u.norm()
            hs = path_from_noise(net, torch.cat([x[None], eps], 0))
            y = net.f(L, hs[-1][None])[0] @ net.w
            obj = 0.5 * (eps * eps).sum() - log_sf((t - y) / net.sigma)
            obj.backward(); count[0] += 1
            return obj

        opt.step(closure)
        with torch.no_grad():
            x = math.sqrt(d) * u / u.norm(); xi = torch.cat([x[None], eps], 0)
            hs = path_from_noise(net, xi); y = float(net.f(L, hs[-1][None])[0] @ net.w)
            obj = float(-0.5 * (eps * eps).sum() + log_sf(torch.tensor((t - y) / net.sigma)))
        if best is None or obj > best[1]:
            best = (xi.detach().clone(), obj)
    flops = count[0] * 3 * L * (2 * d * d + 12 * d)
    return best[0], best[1], flops


def twists_tail(net, t, xi):
    """Probit-linear twists from the linearisation around the instanton path xi."""
    d, L, s = net.d, net.L, net.sigma
    hs = path_from_noise(net, xi)                           # h_0..h_{L-1}
    gam = [None] * L                                        # gamma_l = d y / d h_l along the path
    J = net.jac(L, hs[L - 1][None])[0]; gam[L - 1] = J.T @ net.w
    for l in range(L - 2, -1, -1):
        gam[l] = net.jac(l + 1, hs[l][None])[0].T @ gam[l + 1]
    yL = float(net.f(L, hs[L - 1][None])[0] @ net.w)
    taus, alphas = [None] * L, [None] * L
    for l in range(L):
        shift = sum(s * float(gam[k] @ xi[k]) for k in range(l + 1, L))   # linearised effect of eps*_{l+1..L-1}
        var = s * s + sum(s * s * float(gam[k] @ gam[k]) for k in range(l + 1, L))
        sd = math.sqrt(var)
        ybar = yL - shift
        alphas[l] = gam[l] / sd; taus[l] = (t - ybar + float(gam[l] @ hs[l])) / sd
    flops = L * (2 * d ** 3 + 2 * d * d)
    return Twists(taus, alphas), flops


def twists_iapf(net, t, N, gen, iters=4, init=None, ridge=1e-6):
    """Iterated APF: fit psi_l by least squares in probit space to (K_{l+1} psi_{l+1})(h_l) at the particles of the
    current twisted run (black-box: only layer evaluations and the noise model are used). Returns twists, FLOPs."""
    d, L, s = net.d, net.L, net.sigma
    tw = init; flops = 0
    for it in range(iters):
        parts = sample_particles(net, tw, N, gen); flops += N * L * (2 * d * d + 30 * d)
        taus, alphas = [None] * L, [None] * L
        nxt = None
        for l in range(L - 1, -1, -1):
            H = parts[l]                                    # particles of h_l (resampled, equal weight)
            m = net.f(l + 1, H)
            if l == L - 1:
                lv = log_sf((t - m @ net.w) / s)
            else:
                a, ta = nxt; vs = math.sqrt(1 + s * s * float(a @ a))
                lv = log_sf((ta - m @ a) / vs)
            u = torch.where(lv < math.log(0.5), -ndtri_log(lv), ndtri_log(torch.log((-torch.expm1(lv)).clamp_min(1e-300))))
            X = torch.cat([torch.ones(len(H), 1, dtype=DT), -H], 1)
            A = X.T @ X / len(H) + ridge * torch.eye(d + 1, dtype=DT); b = X.T @ u / len(H)
            th = torch.linalg.solve(A, b)
            taus[l], alphas[l] = th[0], th[1:]
            nxt = (alphas[l], taus[l])
            flops += len(H) * ((d + 1) ** 2 * 2 + 2 * d * d)
        tw = Twists(taus, alphas)
    return tw, flops


def sample_particles(net, tw, N, gen):
    """Particles h_0..h_{L-1} of a twisted run (after resampling at each step); tw None -> plain forward process."""
    d, L, s = net.d, net.L, net.sigma
    out = []
    if tw is None:
        h = torch.randn(N, d, generator=gen, dtype=DT); out.append(h)
        for l in range(1, L):
            h = net.f(l, h) + s * torch.randn(N, d, generator=gen, dtype=DT); out.append(h)
        return out
    al, ta = tw.alpha[0], tw.tau[0]; na2 = float(al @ al)
    v = trunc_normal_above(torch.zeros(N, dtype=DT), torch.full((N,), math.sqrt(1 + na2), dtype=DT), ta.expand(N), gen)
    xi = torch.randn(N, d, generator=gen, dtype=DT); ah = al / math.sqrt(na2)
    h = al[None] * (v / (1 + na2))[:, None] + xi - (1 - 1 / math.sqrt(1 + na2)) * ah[None] * (xi @ ah)[:, None]
    out.append(h); lpsi_prev = log_sf(ta - h @ al)
    for l in range(1, L):
        m = net.f(l, h); al, ta = tw.alpha[l], tw.tau[l]; na2 = float(al @ al); vs = math.sqrt(s * s * na2 + 1)
        am = m @ al; lG = log_sf((ta - am) / vs) - lpsi_prev
        idx = torch.multinomial(torch.softmax(lG, 0), N, replacement=True, generator=gen)
        m, am = m[idx], am[idx]
        v = trunc_normal_above(am, torch.full((N,), vs, dtype=DT), ta.expand(N), gen)
        xi = torch.randn(N, d, generator=gen, dtype=DT); ah = al / math.sqrt(na2)
        h = m + (s * s) * al[None] * ((v - am) / (vs * vs))[:, None] + s * (xi - (1 - 1 / vs) * ah[None] * (xi @ ah)[:, None])
        out.append(h); lpsi_prev = log_sf(ta - h @ al)
    return out


def meanshift_is(net, t, xi_star, N, gen):
    """eps_l ~ N(eps*_l, I) (l = 1..L-1) and x ~ N(x*/sqrt(d), I) (a mild push of the input direction);
    weight = prior/proposal * P(A | h_{L-1}). Unbiased. Returns log p_hat (nats), FLOPs."""
    d, L, s = net.d, net.L, net.sigma
    sh = xi_star.clone(); sh[0] = sh[0] / math.sqrt(d)
    Z = torch.randn(N, L, d, generator=gen, dtype=DT); XI = Z + sh[None]
    lw = -(XI.reshape(N, -1) @ sh.reshape(-1)) + 0.5 * float((sh * sh).sum())
    h = XI[:, 0]
    for l in range(1, L):
        h = net.f(l, h) + s * XI[:, l]
    lw = lw + log_sf((t - net.f(L, h) @ net.w) / s)
    return float(torch.logsumexp(lw, 0)) - math.log(N), N * L * (2 * d * d + 12 * d)


def k2_tail(net, t):
    mu, S = R.k2_propagate(net)
    m = float(net.w @ mu); sd = math.sqrt(float(net.w @ S @ net.w))
    return float(log_sf(torch.tensor((t - m) / sd))), net.flops_k2()
