"""EXPLORATORY lane L1: aggregators that learn keys from provenance metadata (aggbench keys).

All arms follow the aggbench learned-aggregator interface (.module, .fit(view, truth), .predict(view)) and
see only batch.view(visibility): reports, keys, public constants. Training label is the realized truth
(no distillation to bayes).

Bernoulli envs (E2, E3-*), reports r_i are log-odds, l0 the public prior log-odds:
  lk_clear      phi_i = MLP(key_i) in R^D; G = Phi Phi^T; out = l0 + diag(G)^T (G + eps I)^{-1} (r - l0).
                With phi_ik = sqrt(2) mu_k c_ik this is exactly oracle_clearing for honest agents
                (oracle: l0 + m^T C^T (C M C^T)^+ r, m_k = 2 mu_k^2), so the solve is built in and the
                keys K = Phi are learned.
  lk_clear_cal  lk_clear plus a per-agent value calibrator: r'_i = (r_i - l0) exp(a_i) + b_i with
                (a_i, b_i) = MLP(key_i, report features of agent i only); zero-initialized output.
  attn_pool     generic set attention over tokens [key_i, (r_i - l0)/4], no solve:
                out = l0 + sum_i w_i (r_i - l0) + mean_i c_i.
Gaussian env (E1), reports (m_i, P_i); J_i = P_i - I, h_i = P_i m_i:
  lk_clear      h_i = J_i theta + noise with Cov(h_i) = J_i and Cov(h_i, h_j) = G_i G_j^T (G_i =
                A_i^T R_i^{-1} B_i). Learned: G_i = chol(J_i + eps I) Q_i, Q_i = M_i L'^{-T} with
                L'L'^T = I + M_i^T M_i (so G_i G_i^T <= J_i + eps I and Sigma is PSD), M_i = MLP(key_i, J_i).
                Sigma = blockdiag(J_i + eps I) + offdiag(G G^T); prec = I + J^T Sigma^{-1} J,
                info = J^T Sigma^{-1} h. With the true G_i it reproduces bayes (checked, 1e-13 at eps=0).
  attn_pool     set attention over [key_i, features of J_i, h_i]; prec = I + sum_i alpha_i J_i,
                info = sum_i beta_i h_i, alpha, beta > 0.
"""
import numpy as np
import torch
import torch.nn as nn

DT = torch.float32   # networks
D64 = torch.float64  # all linear solves / Cholesky factorizations


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h, dtype=DT), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o, dtype=DT))
    return nn.Sequential(*layers)


class SetBlock(nn.Module):
    def __init__(self, w, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(w, heads, batch_first=True, dtype=DT)
        self.ln1, self.ln2 = nn.LayerNorm(w, dtype=DT), nn.LayerNorm(w, dtype=DT)
        self.ff = mlp(w, 2 * w, w, depth=1)

    def forward(self, z):
        a = self.ln1(z)
        z = z + self.attn(a, a, a, need_weights=False)[0]
        return z + self.ff(self.ln2(z))


# ----------------------------------------------------------------------------- bernoulli modules

class LKClear(nn.Module):
    def __init__(self, key_dim, D=32, hidden=64, calibrate=False):
        super().__init__()
        self.phi = mlp(key_dim, hidden, D)
        self.log_eps = nn.Parameter(torch.tensor(-4.0, dtype=DT))
        self.calibrate = calibrate
        if calibrate:
            self.cal = mlp(key_dim + 4, hidden, 2)
            nn.init.zeros_(self.cal[-1].weight)
            nn.init.zeros_(self.cal[-1].bias)

    def forward(self, r, keys):          # r: (n, N) report minus l0; keys (n, N, k)
        keys, r = keys.float(), r.float()
        phi = self.phi(keys)
        G = phi @ phi.transpose(1, 2)
        dg = torch.diagonal(G, dim1=1, dim2=2)
        if self.calibrate:
            u = r / 4
            z = r / torch.sqrt(dg + 0.05)
            f = torch.cat([keys, torch.stack([u, u.abs(), torch.tanh(z / 4), torch.log(dg + 0.05)], -1)], -1)
            ab = self.cal(f)
            r = r * torch.exp(ab[..., 0].clamp(-5, 5)) + ab[..., 1]
        eps = torch.exp(self.log_eps.double()) + 1e-8
        G64, r64 = G.double(), r.double()
        eye = torch.eye(G.shape[-1], dtype=D64)
        v = torch.linalg.solve(G64 + eps * eye, r64.unsqueeze(-1))[..., 0]
        return (torch.diagonal(G64, dim1=1, dim2=2) * v).sum(-1)


class AttnPool(nn.Module):
    def __init__(self, in_dim, w=32, layers=2, out=2):
        super().__init__()
        self.emb = nn.Linear(in_dim, w, dtype=DT)
        self.blocks = nn.ModuleList([SetBlock(w) for _ in range(layers)])
        self.head = nn.Linear(w, out, dtype=DT)

    def forward(self, tokens):
        z = self.emb(tokens)
        for b in self.blocks:
            z = b(z)
        return self.head(z)


class BernAttn(nn.Module):
    def __init__(self, key_dim):
        super().__init__()
        self.net = AttnPool(key_dim + 1)

    def forward(self, r, keys):
        o = self.net(torch.cat([keys, (r / 4).unsqueeze(-1)], -1).float()).double()
        return (o[..., 0] * r).sum(-1) + o[..., 1].mean(-1)


# ----------------------------------------------------------------------------- gaussian modules

def j_features(J, h):
    n, N, d, _ = J.shape
    tr = torch.diagonal(J, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    s = 1 + tr
    return torch.cat([J.reshape(n, N, d * d) / s, torch.log(s), h / torch.sqrt(s)], -1)


class GaussLKClear(nn.Module):
    def __init__(self, key_dim, d=4, rank=3, hidden=96):
        super().__init__()
        self.d, self.rank = d, rank
        self.M = mlp(key_dim + d * d + 1, hidden, d * rank)
        self.log_eps = nn.Parameter(torch.tensor(-7.0, dtype=DT))

    def forward(self, J, h, keys):
        n, N, d, _ = J.shape
        feats = torch.cat([keys, j_features(J, h)[..., :d * d + 1]], -1)
        M = self.M(feats.float()).double().reshape(n, N, d, self.rank)
        Ir = torch.eye(self.rank, dtype=D64)
        Lp = torch.linalg.cholesky(Ir + M.transpose(-1, -2) @ M)
        Q = torch.linalg.solve_triangular(Lp, M.transpose(-1, -2), upper=False).transpose(-1, -2)  # M Lp^{-T}
        eps = torch.exp(self.log_eps.double()) + 1e-9
        Jr = J + eps * torch.eye(d, dtype=D64)
        G = torch.linalg.cholesky(Jr) @ Q                       # (n, N, d, rank)
        Gs = G.reshape(n, N * d, self.rank)
        S = Gs @ Gs.transpose(1, 2)
        S = S - _blockdiag(G @ G.transpose(-1, -2)) + _blockdiag(Jr)   # off-diagonal G G^T, diagonal J_i + eps
        Js = J.reshape(n, N * d, d)
        X = torch.linalg.solve(S, torch.cat([Js, h.reshape(n, N * d, 1)], -1))
        prec = torch.eye(d, dtype=D64) + Js.transpose(1, 2) @ X[..., :d]
        prec = 0.5 * (prec + prec.transpose(1, 2))
        info = (Js.transpose(1, 2) @ X[..., d:])[..., 0]
        return prec, info


def _blockdiag(B):                       # (n, N, d, d) -> (n, N d, N d)
    n, N, d, _ = B.shape
    out = B.new_zeros(n, N * d, N * d)
    for i in range(N):
        out[:, i * d:(i + 1) * d, i * d:(i + 1) * d] = B[:, i]
    return out


class GaussAttn(nn.Module):
    def __init__(self, key_dim, d=4):
        super().__init__()
        self.net = AttnPool(key_dim + d * d + 1 + d)

    def forward(self, J, h, keys):
        o = self.net(torch.cat([keys, j_features(J, h)], -1).float()).double()
        a = nn.functional.softplus(o[..., 0])
        b = nn.functional.softplus(o[..., 1])
        d = J.shape[-1]
        prec = torch.eye(d, dtype=D64) + (a[..., None, None] * J).sum(1)
        info = (b[..., None] * h).sum(1)
        return prec, info


# ----------------------------------------------------------------------------- aggregator wrapper

def gauss_nll(prec, info, theta):
    L = torch.linalg.cholesky(prec)
    mean = torch.cholesky_solve(info.unsqueeze(-1), L)[..., 0]
    diff = theta - mean
    quad = (diff.unsqueeze(-2) @ prec @ diff.unsqueeze(-1))[..., 0, 0]
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad - logdet)


class Aggregator:
    def __init__(self, arm, kind, key_dim, seed, steps=2000, batch=256, lr=3e-3, d=4):
        torch.manual_seed(seed)
        self.kind, self.steps, self.batch, self.lr, self.seed = kind, steps, batch, lr, seed
        if kind == "bernoulli":
            self.module = {"lk_clear": lambda: LKClear(key_dim),
                           "lk_clear_cal": lambda: LKClear(key_dim, calibrate=True),
                           "attn_pool": lambda: BernAttn(key_dim)}[arm]()
        else:
            self.module = {"lk_clear": lambda: GaussLKClear(key_dim, d),
                           "attn_pool": lambda: GaussAttn(key_dim, d)}[arm]()
        self.train_log = []

    def _inputs(self, view):
        keys = torch.tensor(view.keys, dtype=D64)
        if self.kind == "bernoulli":
            return (torch.tensor(view.reports["logit"] - view.public["prior_logit"], dtype=D64), keys)
        P = torch.tensor(view.reports["prec"], dtype=D64)
        m = torch.tensor(view.reports["mean"], dtype=D64)
        d = P.shape[-1]
        return (P - torch.eye(d, dtype=D64), (P @ m.unsqueeze(-1))[..., 0], keys)

    def _loss(self, inp, y):
        if self.kind == "bernoulli":
            return nn.functional.binary_cross_entropy_with_logits(self.module(*inp), y)
        prec, info = self.module(*inp)
        return gauss_nll(prec, info, y).mean()

    def fit(self, view, truth):
        inp = self._inputs(view)
        y = torch.tensor(truth, dtype=D64)
        n = len(y)
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.1)
        gen = torch.Generator().manual_seed(self.seed + 12345)
        for step in range(self.steps):
            idx = torch.randint(0, n, (self.batch,), generator=gen)
            loss = self._loss([t[idx] for t in inp], y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5.0)
            opt.step()
            sched.step()
            if step % 500 == 0 or step == self.steps - 1:
                self.train_log.append([step, float(loss.detach())])
        return self

    def predict(self, view, chunk=2000):
        inp = self._inputs(view)
        outs = []
        with torch.no_grad():
            for s in range(0, len(inp[0]), chunk):
                outs.append(self.module(*[t[s:s + chunk] for t in inp]))
        if self.kind == "bernoulli":
            logit = view.public["prior_logit"] + torch.cat(outs).numpy()
            return {"logit": np.nan_to_num(logit, nan=0.0, posinf=50.0, neginf=-50.0)}
        prec = torch.cat([o[0] for o in outs])
        info = torch.cat([o[1] for o in outs])
        return {"mean": torch.linalg.solve(prec, info.unsqueeze(-1))[..., 0].numpy(), "prec": prec.numpy()}
