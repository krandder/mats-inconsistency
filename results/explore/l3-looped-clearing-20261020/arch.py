"""EXPLORATORY lane L3: aggregators that turn keys into per-agent weights alpha_i.

Every arm outputs per-agent weights alpha (n, N) from the keys only, and the same readout combines reports:
  E2/E3:  logit = l0 + sum_i alpha_i (l_i - l0)       (the oracle clearing has exactly this form,
          alpha = G^+ diag(G) with G = C M C^T the Gram matrix of the strength-weighted coverage keys)
  E1:     prec = I + sum_i w_i (P_i - I), info = sum_i w_i P_i m_i, w_i = softplus(alpha_i) (tempered product
          with per-agent temperatures; the readout_softmax arm's weights are already >= 0 and used as is)

Arms (the "core" that maps keys -> alpha):
  indep            per-agent MLP, no cross-agent interaction at all (reference)
  readout_softmax  one standard softmax attention layer: a learned query attends over agents,
                   alpha_i = s * softmax_i(q . k_i) -- sees q . k_r only, never k_r . k_s
  stacked T        embed + T pre-LN self-attention blocks (independent weights) + per-agent linear head
  looped T         embed + ONE weight-tied self-attention block applied T times + head
  jacobi T         learned clearing system (K^T B K + Lambda) w = u, K rows k_i = phi(key_i), B = diag(b_i),
                   Lambda = diag(softplus(lam)), u = linear(maxpool_i trunk(key_i)); T damped-Jacobi steps
                   w <- w + omega_t D^-1 (u - P w) from w = 0, then alpha_i = b_i k_i . w.
                   This equals solving (K^T B K + Lambda) w = K^T B dl and reading out u . w.
  exact            the same learned system solved exactly (torch.linalg.solve)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = 2
P_DIM = 16  # feature dimension p of the learned clearing system


class SelfAttention(nn.Module):
    """Standard multi-head softmax self-attention over agents (same math as nn.MultiheadAttention)."""

    def __init__(self, d):
        super().__init__()
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        n, N, d = x.shape
        q, k, v = self.qkv(x).view(n, N, 3, HEADS, d // HEADS).unbind(2)
        att = torch.softmax(torch.einsum("nihc,njhc->nhij", q, k) / math.sqrt(d // HEADS), dim=-1)
        return self.o(torch.einsum("nhij,njhc->nihc", att, v).reshape(n, N, d))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class AttnCore(nn.Module):
    def __init__(self, key_dim, d, T, tied):
        super().__init__()
        self.T, self.tied = T, tied
        self.embed = nn.Linear(key_dim, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(1 if tied else T)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, keys):
        x = self.embed(keys)
        for t in range(self.T):
            x = self.blocks[0 if self.tied else t](x)
        return self.head(self.ln(x))[..., 0]


class IndepCore(nn.Module):
    def __init__(self, key_dim, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(key_dim, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, keys):
        return self.net(keys)[..., 0]


class ReadoutSoftmaxCore(nn.Module):
    """alpha_i = s * softmax_i(q . k_i), k_i = MLP(key_i): one softmax attention readout over agents."""
    nonneg = True

    def __init__(self, key_dim, d):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(key_dim, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.log_s = nn.Parameter(torch.zeros(()))

    def forward(self, keys):
        k = self.key(keys)
        a = torch.softmax(k @ self.q / math.sqrt(k.shape[-1]), dim=-1)
        return torch.exp(self.log_s) * a


class ClearingCore(nn.Module):
    def __init__(self, key_dim, d, T, exact):
        super().__init__()
        self.T, self.exact = T, exact
        self.trunk = nn.Sequential(nn.Linear(key_dim, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.k = nn.Linear(d, P_DIM)
        self.b = nn.Linear(d, 1)
        self.u = nn.Linear(d, P_DIM)
        self.lam = nn.Parameter(torch.full((P_DIM,), -2.0))
        self.omega = nn.Parameter(torch.zeros(max(T, 1)))  # omega_t = 2 sigmoid(.) in (0, 2), init 1

    def system(self, keys):
        h = self.trunk(keys)                                   # (n, N, d)
        K = self.k(h)                                          # (n, N, p)
        b = F.softplus(self.b(h))[..., 0] + 1e-3               # (n, N)
        u = self.u(h.max(1).values)                            # (n, p)
        P = torch.einsum("nip,ni,niq->npq", K, b, K) + torch.diag(F.softplus(self.lam) + 1e-4)
        return K, b, u, P

    def forward(self, keys):
        K, b, u, P = self.system(keys)
        if self.exact:
            w = torch.linalg.solve(P, u)
        else:
            dinv = 1.0 / torch.diagonal(P, dim1=-2, dim2=-1)
            w = torch.zeros_like(u)
            for t in range(self.T):
                w = w + 2 * torch.sigmoid(self.omega[t]) * dinv * (u - torch.einsum("npq,nq->np", P, w))
        return b * torch.einsum("nip,np->ni", K, w)


def make_core(arch, key_dim, d, T):
    if arch == "indep":
        return IndepCore(key_dim, d)
    if arch == "readout_softmax":
        return ReadoutSoftmaxCore(key_dim, d)
    if arch == "stacked":
        return AttnCore(key_dim, d, T, tied=False)
    if arch == "looped":
        return AttnCore(key_dim, d, T, tied=True)
    if arch == "jacobi":
        return ClearingCore(key_dim, d, T, exact=False)
    if arch == "exact":
        return ClearingCore(key_dim, d, T, exact=True)
    raise ValueError(arch)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_width(arch, key_dim, T, budget):
    """Width d (even, for 2 heads) whose parameter count is closest to budget."""
    best = None
    for d in range(4, 400, 2):
        n = n_params(make_core(arch, key_dim, d, T))
        if best is None or abs(n - budget) < abs(best[1] - budget):
            best = (d, n)
        if n > 1.5 * budget:
            break
    return best


class Module(nn.Module):
    def __init__(self, core, kind):
        super().__init__()
        self.core, self.kind = core, kind
        self.nonneg = getattr(core, "nonneg", False)

    def forward(self, keys, rep):
        alpha = self.core(keys)
        if self.kind == "bernoulli":
            return {"logit": rep["l0"] + (alpha * rep["dl"]).sum(1)}
        w = alpha if self.nonneg else F.softplus(alpha)
        d = rep["S"].shape[-1]
        prec = torch.eye(d) + torch.einsum("ni,nijk->njk", w, rep["S"])
        h = torch.einsum("ni,nij->nj", w, rep["h"])
        return {"prec": prec, "mean": torch.linalg.solve(prec, h[..., None])[..., 0]}


def loss_fn(out, y, kind):
    if kind == "bernoulli":
        return F.binary_cross_entropy_with_logits(out["logit"], y)
    L = torch.linalg.cholesky(out["prec"])
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    diff = y - out["mean"]
    quad = torch.einsum("ni,nij,nj->n", diff, out["prec"], diff)
    return (0.5 * (quad - logdet)).mean()


class Aggregator:
    """aggbench learned-aggregator interface: .module, .fit(view, truth), .predict(view)."""

    def __init__(self, seed, kind, arch, key_dim, d, T, steps=1000, batch=128, lr=3e-3):
        torch.manual_seed(seed)
        self.kind, self.steps, self.batch, self.lr, self.seed = kind, steps, batch, lr, seed
        self.module = Module(make_core(arch, key_dim, d, T), kind)

    def _inputs(self, view):
        keys = torch.tensor(view.keys, dtype=torch.float32)
        if view.kind == "bernoulli":
            l0 = float(view.public["prior_logit"])
            rep = {"l0": torch.tensor(l0), "dl": torch.tensor(view.reports["logit"] - l0, dtype=torch.float32)}
        else:
            P, m = view.reports["prec"], view.reports["mean"]
            S = P - np.eye(P.shape[-1])
            h = np.einsum("naij,naj->nai", P, m)
            rep = {"S": torch.tensor(S, dtype=torch.float32), "h": torch.tensor(h, dtype=torch.float32)}
        return keys, rep

    def fit(self, view, truth):
        keys, rep = self._inputs(view)
        y = torch.tensor(truth, dtype=torch.float32)
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.1)
        gen = torch.Generator().manual_seed(self.seed + 1000)
        n = len(y)
        self.trace = []
        for step in range(self.steps):
            idx = torch.randint(0, n, (self.batch,), generator=gen)
            sub = {k: (v[idx] if v.dim() > 0 else v) for k, v in rep.items()}
            loss = loss_fn(self.module(keys[idx], sub), y[idx], self.kind)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 250 == 0 or step == self.steps - 1:
                self.trace.append([step, float(loss.detach())])

    def predict(self, view):
        keys, rep = self._inputs(view)
        with torch.no_grad():
            out = self.module(keys, rep)
        if self.kind == "bernoulli":  # guard (all arms alike): a diverging random net must not give inf/NaN
            return {"logit": np.clip(np.nan_to_num(out["logit"].double().numpy(), nan=0.0), -50, 50)}
        prec = out["prec"].double().numpy()
        prec = 0.5 * (prec + np.swapaxes(prec, -1, -2))
        return {"mean": out["mean"].double().numpy(), "prec": prec}
