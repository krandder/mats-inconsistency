"""T1 lemma 7 check: the top-k router gradient is gate-weighted relative revenue under censored feedback;
the gate doubles tangent books (Euler degree 2), so a gate convention is needed (router = broker).

C7a closed form vs autograd for five gate conventions; C7b zero-sum / mass transfer; C7c books at the gate
(2R doubling, broker conservation down into expert neurons, router income = Cov_g(z, r), SwiGLU doubling);
C7d Exp3 shadow-credit estimator is unbiased for the full-information (dense softmax) gradient;
C7e numpy bandit lock-in toy. Torch float64 (PYTHONPATH=wealth-net/.pylib).
Usage: python3 check_L7.py --seed 1
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
from streams import make_questions  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
D, H, N, W, B = 32, 16, 8, 16, 256


def batch(seed):
    bits, fam, _ = make_questions(64, 2000, seed)
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 64, B); t = rng.integers(W, 2000, B)
    win = np.stack([bits[qq, tt - W:tt] for qq, tt in zip(q, t)]).astype(float)
    y = bits[q, t].astype(float)
    return torch.tensor(2 * win - 1), torch.tensor(y)


class MoE:
    def __init__(self, g):
        r = lambda *s: torch.randn(*s, generator=g)
        self.P = r(W, D) / W ** 0.5
        self.W1 = r(N, D, H) / D ** 0.5; self.b1 = 0.1 * r(N, H)
        self.W2 = r(N, H, D) / H ** 0.5; self.b2 = 0.1 * r(N, D)
        self.Wr = r(D, N) / D ** 0.5 * 2; self.br = 0.5 * r(N)
        self.w = r(D) / D ** 0.5


def gates(z, conv, k):
    if conv in ("renorm", "top1_renorm"):
        S = torch.topk(z, k, -1).indices
        mask = torch.zeros_like(z).scatter(-1, S, 1.0).bool()
        g = torch.softmax(z.masked_fill(~mask, -torch.inf), -1)
    elif conv in ("unnorm", "top1_unnorm"):
        S = torch.topk(z, k, -1).indices
        mask = torch.zeros_like(z).scatter(-1, S, 1.0).bool()
        g = torch.softmax(z, -1) * mask
    elif conv == "sigmoid":
        S = torch.topk(z, k, -1).indices
        mask = torch.zeros_like(z).scatter(-1, S, 1.0).bool()
        s = torch.sigmoid(z) * mask
        g = s / s.sum(-1, keepdim=True)
    elif conv == "dense":
        mask = torch.ones_like(z).bool(); g = torch.softmax(z, -1)
    return g, mask


def forward(m, xw, y, conv, k):
    x = xw @ m.P
    z = x @ m.Wr + m.br
    z.retain_grad()
    a = torch.relu(torch.einsum("bd,ndh->bnh", x, m.W1) + m.b1)          # [B,N,H]
    a.retain_grad()
    E = torch.einsum("bnh,nhd->bnd", a, m.W2) + m.b2                     # all experts computed (for r_i)
    E.retain_grad()
    g, mask = gates(z, conv, k)
    g.retain_grad()
    h = (g[..., None] * E).sum(1)
    h.retain_grad()
    logit = (x + h) @ m.w
    L = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    L.backward()
    return dict(z=z, a=a, E=E, g=g, h=h, mask=mask, x=x)


def closed_form(conv, z, g, mask, r):
    R = (g * r).sum(-1, keepdim=True)                                     # routed revenue per token
    if conv in ("renorm",):
        return g * (r - R) * mask
    if conv == "top1_renorm":
        return torch.zeros_like(z)
    if conv in ("unnorm", "top1_unnorm"):
        gN = torch.softmax(z, -1)
        return gN * (r * mask - R)
    if conv == "sigmoid":
        s = torch.sigmoid(z)
        return (1 - s) * g * (r - R) * mask
    if conv == "dense":
        return g * (r - R)


def check_grad_and_books(seed):
    xw, y = batch(seed)
    g_ = torch.Generator().manual_seed(seed)
    m = MoE(g_)
    for p in vars(m).values():
        p.requires_grad_(True)
    out = {}
    for conv, k in (("renorm", 2), ("unnorm", 2), ("sigmoid", 2), ("top1_renorm", 1), ("top1_unnorm", 1), ("dense", N)):
        f = forward(m, xw, y, conv, k)
        mgh = -f["h"].grad                                               # -dL/dh
        r = torch.einsum("bnd,bd->bn", f["E"].detach(), mgh)             # revenue at unit gate, every expert
        u_auto = -f["z"].grad
        u_cf = closed_form(conv, f["z"].detach(), f["g"].detach(), f["mask"], r)
        R = (f["g"].detach() * r).sum(-1)
        o = {"C7a_max_abs_err": float((u_auto - u_cf).abs().max()),
             "C7a_scale_max_abs_grad": float(u_auto.abs().max()),
             "C7b_max_abs_sum_over_experts": float(u_auto.sum(-1).abs().max())}
        if conv == "unnorm":
            gin = (f["g"].detach()).sum(-1)
            o["C7b_routed_gain_minus_R(1-sum_S g)"] = float((( u_auto * f["mask"]).sum(-1) - R * (1 - gin)).abs().max())
            o["C7b_frac_tokens_R_pos"] = float((R > 0).double().mean())
        # C7c books
        gb = f["g"].detach() * (-f["g"].grad)                            # gate tangent books g * dR/dg
        eb = torch.einsum("bnd,bnd->bn", f["E"].detach(), -f["E"].grad)  # expert-output books <E, dR/dE>
        ok = R.abs() > 1e-12
        o["C7c_ratio_gate_plus_expert_over_R_minmax"] = [float(((gb + eb).sum(-1)[ok] / R[ok]).min()), float(((gb + eb).sum(-1)[ok] / R[ok]).max())]
        o["C7c_broker_residual_max"] = float((eb.sum(-1) - R).abs().max())
        o["C7c_euler_split_residual_max"] = float(((gb + eb).sum(-1) / 2 - R).abs().max())
        nb = torch.einsum("bnh,bnh->bn", f["a"].detach(), -f["a"].grad)   # expert neuron books a * dR/da
        biasb = f["g"].detach() * torch.einsum("nd,bd->bn", m.b2.detach(), mgh)  # b2 book (per token)
        o["C7c_broker_into_neurons_residual_max"] = float((nb + biasb - eb).abs().max())
        zb = (f["z"].detach() * u_auto).sum(-1)                          # router logit tangent income
        gd = f["g"].detach(); zz = f["z"].detach() * f["mask"]
        zbar = (gd * zz).sum(-1, keepdim=True)
        cov = (gd * (zz - zbar) * (r - R[:, None]) * f["mask"]).sum(-1)
        if conv == "renorm":
            o["C7c_router_income_minus_cov_max"] = float((zb - cov).abs().max())
            o["C7c_router_income_mean"] = float(zb.mean()); o["C7c_R_mean"] = float(R.mean())
        out[conv] = o
        for p in vars(m).values():
            p.grad = None
    # SwiGLU doubling: m = silu(W1 x) * (W3 x); tangent books on the two factors vs the product
    x = (xw @ m.P).detach()
    W1 = torch.randn(D, 4 * D, generator=g_) / D ** 0.5; W3 = torch.randn(D, 4 * D, generator=g_) / D ** 0.5
    W2 = torch.randn(4 * D, 1, generator=g_) / (4 * D) ** 0.5
    uu = torch.nn.functional.silu(x @ W1).requires_grad_(True); vv = (x @ W3).requires_grad_(True)
    mm = uu * vv; mm.retain_grad()
    Ls = torch.nn.functional.binary_cross_entropy_with_logits((mm @ W2).squeeze(-1), y); Ls.backward()
    two = (uu * uu.grad + vv * vv.grad).sum() / (mm * mm.grad).sum()
    out["swiglu_factor_books_over_product_books"] = float(two)
    return out, m, xw, y


def check_exp3(m, xw, y, seed, draws=1000):
    """C7d: shadow credit. Route top-2 (unnorm convention for gates of routed experts); sample one unrouted expert J
    uniformly, run it at zero gate, credit r_J / q_J. Estimator of the dense-softmax gradient g^N (r - sum g^N r)."""
    with torch.no_grad():
        x = xw @ m.P
        z = x @ m.Wr + m.br
        a = torch.relu(torch.einsum("bd,ndh->bnh", x, m.W1) + m.b1)
        E = torch.einsum("bnh,nhd->bnd", a, m.W2) + m.b2
    gN = torch.softmax(z, -1).detach()
    # use the dense model's -dL/dh as the revenue direction (full-information target)
    Ed = E.detach().requires_grad_(False)
    h = (gN[..., None] * Ed).sum(1).requires_grad_(True)
    L = torch.nn.functional.binary_cross_entropy_with_logits((x.detach() + h) @ m.w.detach(), y)
    L.backward()
    r = torch.einsum("bnd,bd->bn", Ed, -h.grad)
    target = (gN * (r - (gN * r).sum(-1, keepdim=True))).sum(0)           # batch-summed (router bias gradient)
    S = torch.topk(z, 2, -1).indices
    mask = torch.zeros_like(z).scatter(-1, S, 1.0).bool()
    gen = torch.Generator().manual_seed(10 * seed + 7)
    outside = (~mask).double()
    est = torch.zeros(draws, N)
    for d in range(draws):
        J = torch.multinomial(outside, 1, generator=gen).squeeze(-1)
        rhat = r * mask
        rhat[torch.arange(B), J] = r[torch.arange(B), J] * (N - 2)          # q_J = 1/(N-k)
        est[d] = (gN * (rhat - (gN * rhat).sum(-1, keepdim=True))).sum(0)
    mean, se = est.mean(0), est.std(0) / draws ** 0.5
    censored = (gN * (r * mask - (gN * r * mask).sum(-1, keepdim=True))).sum(0)
    return {"C7d_max_abs_z": float(((mean - target) / se).abs().max()),
            "C7d_target": target.tolist(), "C7d_mean_est": mean.tolist(), "C7d_se": se.tolist(),
            "C7d_censored_top2_unnorm_grad": censored.tolist(),
            "C7d_censored_rel_err_vs_target": float((censored - target).norm() / target.norm())}


def lockin(seed, n_inst=100, NE=16, steps=5000, eta=0.1, noise=0.3):
    rng = np.random.default_rng(seed)
    MU = rng.uniform(0.1, 1.0, (n_inst, NE)); Z0 = rng.normal(0, 0.01, (n_inst, NE))  # same instances for every arm
    res = {}
    for arm in ("top1_unnorm_censored", "top2_renorm_censored", "top1_unnorm_exp3"):
        rng_a = np.random.default_rng(seed + 100)
        best_hit, share_best, regret = [], [], []
        for i in range(n_inst):
            mu = MU[i]; z = Z0[i].copy()
            hits = 0; reg = 0.0
            for t in range(steps):
                r = mu + noise * rng_a.standard_normal(NE)
                gN = np.exp(z - z.max()); gN /= gN.sum()
                if arm == "top2_renorm_censored":
                    S = np.argsort(-z)[:2]
                    g = np.zeros(NE); g[S] = gN[S] / gN[S].sum()
                    R = (g * r).sum(); u = np.zeros(NE); u[S] = g[S] * (r[S] - R)
                    sel = S
                else:
                    s = int(np.argmax(z)); sel = [s]
                    if arm == "top1_unnorm_censored":
                        rh = np.zeros(NE); rh[s] = r[s]
                        R = gN[s] * r[s]
                        u = gN * (rh - R)
                    else:
                        J = int(rng_a.integers(NE - 1)); J += int(J >= s)
                        rh = np.zeros(NE); rh[s] = r[s]; rh[J] = r[J] * (NE - 1)
                        u = gN * (rh - (gN * rh).sum())
                z = z + eta * u
                if t >= steps - 1000:
                    hits += int(np.argmax(mu) in sel); reg += mu.max() - mu[sel].mean()
            best_hit.append(int(np.argmax(z) == np.argmax(mu))); share_best.append(hits / 1000); regret.append(reg / 1000)
        res[arm] = {"frac_final_argmax_is_best": float(np.mean(best_hit)),
                    "mean_share_best_routed_last1k": float(np.mean(share_best)),
                    "mean_regret_last1k": float(np.mean(regret))}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-lockin", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    res, m, xw, y = check_grad_and_books(a.seed)
    res.update(check_exp3(m, xw, y, a.seed))
    if not a.skip_lockin:
        res["C7e"] = lockin(a.seed)
    path = os.path.join(HERE, "out", f"L7-s{a.seed}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
