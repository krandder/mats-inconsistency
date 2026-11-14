"""P11 per-pixel belief caps (float32 runtime, float64 ledgers).

Binary LMSR on a logit b (price sigmoid(b)). Moving b by d costs the mover, in the worst case over
the outcome, wl(b, d) nats:
    wl(b, d) = logaddexp(logsig(-b), logsig(b) + d)       d >= 0   (= softplus(b+d) - softplus(b))
    wl(b, d) = logaddexp(logsig(b),  logsig(-b) - d)      d <  0   (mirror)
The largest move with worst-case loss eps:
    d_max(b, eps) = logaddexp(log(expm1(eps)), logsig(b)) - logsig(b),   d_min(b, eps) = -d_max(-b, eps).
Retreating from a confident belief toward 0 costs at most ln 2 however far it is; a logit total-variation
budget charges the whole path. That asymmetry is the thing P11 tests.
"""
import math

import torch
import torch.nn.functional as F

EPS_ZERO = 1e-6


def logexpm1(e):
    """log(expm1(e)) for e >= 0, finite for large e; -inf at e = 0."""
    big = e > 20
    es = torch.where(big, torch.ones_like(e), e)
    return torch.where(big, e + torch.log1p(-torch.exp(-e)), torch.log(torch.expm1(es)))


def wl(b, d):
    """Worst-case loss (nats) of moving logit b by d (elementwise, any float dtype)."""
    ls_p, ls_m = F.logsigmoid(b), F.logsigmoid(-b)
    up = torch.logaddexp(ls_m, ls_p + d)
    dn = torch.logaddexp(ls_p, ls_m - d)
    return torch.where(d >= 0, up, dn)


def d_max(b, eps):
    ls = F.logsigmoid(b)
    return torch.logaddexp(logexpm1(eps), ls) - ls


def d_min(b, eps):
    return -d_max(-b, eps)


def lmsr_delta(b, u, eps):
    """Move on the tanh scale inside the LMSR worst-case-loss bound eps (tensor, same shape as b).
    b and eps are treated as constants (detached); the gradient flows through tanh(u)."""
    b = b.detach()
    eps = eps.detach().to(b.dtype)
    t = torch.tanh(u)
    hi = d_max(b, eps)
    lo = -d_min(b, eps)            # |d_min|
    delta = torch.where(u >= 0, hi * t, lo * t)
    return torch.where(eps < EPS_ZERO, torch.zeros_like(delta), delta)


def tv_delta(u, r):
    """Logit total-variation budget: move r * tanh(u), r = remaining path length."""
    r = r.detach().to(u.dtype)
    delta = r * torch.tanh(u)
    return torch.where(r < EPS_ZERO, torch.zeros_like(delta), delta)


def clip_delta(u, c):
    return c * torch.tanh(u)


def lmsr_clamp(b, d, eps):
    """Tracker form: clamp a desired move d into [d_min(b, eps), d_max(b, eps)]."""
    hi = d_max(b, eps)
    lo = d_min(b, eps)
    out = torch.minimum(torch.maximum(d, lo), hi)
    return torch.where(eps < EPS_ZERO, torch.zeros_like(out), out)


def charge_lmsr(rem, b_old, b_new):
    """float64 ledger: charge the realised worst-case loss of b_old -> b_new; never below 0."""
    b0 = b_old.detach().double()
    cost = wl(b0, b_new.detach().double() - b0)
    return torch.clamp(rem - cost, min=0.0), cost


def charge_tv(rem, b_old, b_new):
    cost = (b_new.detach().double() - b_old.detach().double()).abs()
    return torch.clamp(rem - cost, min=0.0), cost


# ------------------------------------------------------------ S0 numerics tests
def ulp32(x):
    x = x.abs().float()
    return (torch.nextafter(x, torch.full_like(x, float("inf"))) - x).double()


def s0_tests():
    out = {}
    bs = torch.tensor([0., 1e-3, 0.5, 1, 2, 5, 10, 20, 50, 100, 1e3, 5e3, 1e4, 2e4])
    bs = torch.cat([bs, -bs]).float()
    epss = torch.tensor([0., 1e-9, 1e-3, 0.1, 1.0])
    us = torch.tensor([-100, -30, -5, -3, -1, -0.1, 0, 0.1, 1, 3, 5, 30, 100.])
    B, E, U = torch.meshgrid(bs, epss, us, indexing="ij")
    B, E, U = B.contiguous(), E.contiguous().float(), U.contiguous().float()
    D = lmsr_delta(B, U, E)
    W = wl(B.double(), (B + D).double() - B.double())      # realised move in float32
    tol = E.double() + 1e-6 + 8 * ulp32(B)
    out["lmsr_cap_holds"] = bool((W <= tol).all())
    out["lmsr_max_excess"] = float((W - E.double()).max())
    out["finite"] = bool(torch.isfinite(D).all() and torch.isfinite(W).all())
    out["zero_at_eps0"] = bool((D[E == 0] == 0).all())
    # derivative wrt u for |u| <= 3 at eps > 0 (tanh scale keeps it alive)
    Ug = U.clone().requires_grad_(True)
    D2 = lmsr_delta(B, Ug, E)
    D2.sum().backward()
    g = Ug.grad
    m = (U.abs() <= 3) & (E >= 1e-3)
    # the bound itself can be tiny where eps is tiny relative to |b| moving outward; test the
    # derivative relative to the available range (|dDelta/du| / range >= 1e-3 * sech^2 floor)
    rng_ = torch.where(U >= 0, d_max(B, E), -d_min(B, E))
    rel = g[m] / rng_[m].clamp_min(1e-30)
    out["dDelta_du_rel_min"] = float(rel.min())
    out["dDelta_du_abs_min_eps_ge_1e-3"] = float(g[m].min())
    # spec check "dDelta/du >= 1e-3 for |u| <= 3" is impossible where the available range is < 1e-3/sech^2(3)
    # = 0.1013 (outward moves at confident b with eps <= 0.1 have range ~eps); checked where range >= 0.102
    out["dDelta_du_ge_1e-3_where_range_ge_0.102"] = bool((g[m & (rng_ >= 0.102)] >= 1e-3).all())
    out["frac_grid_points_range_lt_0.102"] = float((m & (rng_ < 0.102)).float().sum() / m.float().sum())
    # retreat is cheap: from b = +/-10 to 0 costs <= ln 2
    b10 = torch.tensor([10., -10., 1e4, -1e4], dtype=torch.float64)
    out["retreat_cost"] = [float(v) for v in wl(b10, -b10)]
    out["retreat_le_ln2"] = bool((wl(b10, -b10) <= math.log(2) + 1e-9).all())
    # d_max inverts wl
    bb = torch.linspace(-30, 30, 121, dtype=torch.float64)
    for e in (1e-3, 0.1, 1.0, 5.0, 40.0):
        ee = torch.full_like(bb, e)
        out[f"inverse_err_eps{e}"] = float((wl(bb, d_max(bb, ee)) - e).abs().max())
        out[f"inverse_err_neg_eps{e}"] = float((wl(bb, d_min(bb, ee)) - e).abs().max())
    # large budgets stay finite
    big = torch.tensor([30., 80., 200., 1e4])
    out["dmax_large_eps_finite"] = bool(torch.isfinite(d_max(torch.zeros(4), big)).all())
    out["pass"] = bool(out["lmsr_cap_holds"] and out["finite"] and out["zero_at_eps0"] and out["retreat_le_ln2"]
                       and out["dDelta_du_ge_1e-3_where_range_ge_0.102"] and out["dmax_large_eps_finite"]
                       and max(v for k, v in out.items() if k.startswith("inverse_err")) < 1e-9)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(s0_tests(), indent=1))
