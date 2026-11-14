"""P7 maze puzzle: hindsight-posterior settlement of per-cell latent markets.

Task. Pixel maze (random obstacles, density 0.3), a start cell S. Output per cell:
y[c] = 1 if c is reachable from S in at most H=4 moves (4-neighbour BFS through open cells).
Ground-truth sufficient variables: g_k[c] = (dist_S(c) <= k*rho) at chain level k, distance
dist_S capped at H+1, and (probe only) on-shortest-path to a random goal G.

Traders. D levels (D in 1,2,4), window radius rho = H/D, so every depth has the same total
receptive field. At level k every cell hosts one trader per unit (U=2 units, K=4 values each);
unit u at cell c reads a (2rho+1)^2 window of the input (wall, start channels) and of the
level k-1 latents (one-hot), and emits p_k(z_k[c,u] | window). The output market is a
conditional table P(y[c] | z_D[c]) (log-linear over units).

Arms.
  A  end-to-end backprop, straight-through Gumbel-softmax latents
  B  hindsight-posterior settlement, local only:
       realise z by sampling the priors; then, from the top down,
       q_D[c,u](v) ∝ p_D[c,u](v) P(y[c] | z_D[c,u]=v, other unit as realised)
       q_k[c,u](v) ∝ p_k[c,u](v) exp( sum over level-(k+1) traders t reading (c,u):
                         sum_v' q_{k+1}[t](v') log P_{k+1}(t=v' | z_k[c,u]=v, rest as realised) )
     i.e. each downstream conditional market P(z_{k+1} | z_k = v) is scored against the settled
     q_{k+1}, and the hindsight posterior on z_k weights v by that score. Each trader's prior is
     trained on CE(q_k, p_k) with q detached; conditional markets (next traders, head) are trained
     on the realised parents. No gradient crosses a trader boundary.
  Bl as B, but each level gets the likelihood of the evidence instead of the settled q above it:
       lam_D = log P(y | z_D[c,u]=v, rest);  lam_k(v) = sum_t log sum_v' P_t(v' | z_k[c,u]=v, rest) e^{lam_t(v')}
       q_k ∝ p_k e^{lam_k}  (the exact hindsight posterior P(z|x,y) ∝ p(z|x) P(y|z) on a chain; a
       sampled loopy belief-propagation approximation on the grid). Still local: no gradient crosses traders.
  C  traders frozen at random init; only the output head is trained
  D  B + information-bottleneck rate cost lam*KL(p_k || batch marginal) on every latent market
     + different input views (unit 0 sees walls only, unit 1 sees the start marker only; both
     see the upstream latents)
  Dl Bl + the same rate cost and views
"""
import argparse
import json
import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import alignment, spearman, auroc

H_REACH = 4
CX = 2  # input channels: wall, start


# ---------------------------------------------------------------- data
def bfs(walls, s):
    n = walls.shape[0]
    d = np.full((n, n), 10 ** 6, dtype=np.int64)
    d[s] = 0
    q = deque([s])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not walls[a, b] and d[a, b] > d[i, j] + 1:
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    return d


def gen_mazes(num, n, density, rng):
    walls = np.zeros((num, n, n), np.float32)
    start = np.zeros((num, n, n), np.float32)
    dist = np.zeros((num, n, n), np.int64)
    onpath = np.zeros((num, n, n), np.int64)
    for t in range(num):
        w = (rng.random((n, n)) < density)
        s = (int(rng.integers(n)), int(rng.integers(n)))
        w[s] = False
        d = bfs(w, s)
        reach = np.argwhere(d < 10 ** 6)
        g = tuple(reach[rng.integers(len(reach))])
        dg = bfs(w, g)
        walls[t] = w
        start[t][s] = 1
        dist[t] = d
        onpath[t] = ((d + dg) == d[g]) & (d < 10 ** 6)
    return walls, start, dist, onpath


def make_set(num, n, seed, density=0.3):
    rng = np.random.default_rng(seed)
    w, s, d, op = gen_mazes(num, n, density, rng)
    return {
        "x": torch.tensor(np.stack([w, s], 1)),
        "dist": d,
        "y": torch.tensor((d <= H_REACH).astype(np.float32)),
        "onpath": op,
        "open": (w == 0),
    }


# ---------------------------------------------------------------- model
def unfold(x, z, rho):
    """x (B,CX,n,n), z (B,U*K,n,n) or None -> (B*n*n, win, C) window features.
    Outside the maze: wall=1, start=0, latents=0."""
    B, _, n, _ = x.shape
    xp = F.pad(x, (rho,) * 4, value=0.0)
    xp[:, 0, :rho] = 1
    xp[:, 0, -rho:] = 1
    xp[:, 0, :, :rho] = 1
    xp[:, 0, :, -rho:] = 1
    if z is not None:
        xp = torch.cat([xp, F.pad(z, (rho,) * 4, value=0.0)], 1)
    C = xp.shape[1]
    k = 2 * rho + 1
    u = F.unfold(xp, k)  # (B, C*win, n*n)
    return u.view(B, C, k * k, n * n).permute(0, 3, 2, 1).reshape(B * n * n, k * k, C)


class Trader(nn.Module):
    """U unit-traders sharing a window; each is a one-hidden-layer MLP with its own input mask."""

    def __init__(self, rho, cz, U, K, hid, masks):
        super().__init__()
        self.rho, self.U, self.K, self.cz = rho, U, K, cz
        win = (2 * rho + 1) ** 2
        C = CX + cz
        self.W1 = nn.Parameter(torch.randn(U, hid, win, C) / math.sqrt(win * C))
        self.b1 = nn.Parameter(torch.zeros(U, hid))
        self.W2 = nn.Parameter(torch.randn(U, K, hid) / math.sqrt(hid))
        self.b2 = nn.Parameter(torch.zeros(U, K))
        self.register_buffer("mask", masks.float())  # (U, C)

    def W1m(self):
        return self.W1 * self.mask[:, None, None, :]

    def forward(self, feats):
        pre = torch.einsum("nwc,uhwc->nuh", feats, self.W1m()) + self.b1
        return torch.einsum("nuh,ukh->nuk", F.relu(pre), self.W2) + self.b2  # (N,U,K)

    @torch.no_grad()
    def message(self, feats, q_next, Us, K, B, n, form="q"):
        """Score of this trader's conditional markets for every counterfactual value v of every
        upstream unit s at every window position. feats (N,win,C) realised inputs.
        form "q": q_next (N,U,K) settled posteriors; m[c,s,v] = sum over traders t whose window
                  contains c of sum_v' q_next[t](v') log P_t(v' | z[c,s]=v)   (q-form, mean-field)
        form "l": q_next holds log-likelihood messages lam_t(v'); m[c,s,v] = sum_t
                  log sum_v' P_t(v' | z[c,s]=v) exp(lam_t(v'))               (likelihood / BP form)
        Returns m (B, Us, K, n, n)."""
        W1m = self.W1m()
        U, hid, win, C = W1m.shape
        N = feats.shape[0]
        Az = W1m[..., CX:].reshape(U, hid, win, Us, K).permute(2, 0, 3, 4, 1).contiguous()  # w a s k h
        zoh = feats[..., CX:].reshape(N, win, Us, K)
        zidx = zoh.argmax(-1)  # realised value (N,w,s)
        valid = zoh.sum(-1)  # (N,w,s): 1 inside the maze, 0 outside
        preT = (torch.einsum("nwc,uhwc->unh", feats, W1m) + self.b1[:, None]).contiguous()  # a N h
        W2T = self.W2.transpose(1, 2).contiguous()  # a h j
        qT = q_next.transpose(0, 1)[:, :, None, None, :]  # a N 1 1 j
        sidx = torch.arange(Us)[None, :]
        out = torch.zeros(N, win, Us, K)
        for w in range(win):
            Aw = Az[w]  # a s k h
            cur = Aw[:, sidx, zidx[:, w], :]  # a N s h
            p2 = (preT[:, :, None, None, :] - cur[:, :, :, None, :] + Aw[:, None]).relu_()  # a N s k h
            lg = torch.bmm(p2.view(U, -1, hid), W2T).view(U, N, Us, K, -1) + self.b2[:, None, None, None, :]
            if form == "q":
                out[:, w] = (F.log_softmax(lg, -1) * qT).sum((0, -1)) * valid[:, w, :, None]
            else:
                out[:, w] = torch.logsumexp(F.log_softmax(lg, -1) + qT, -1).sum(0) * valid[:, w, :, None]
        # fold: contributions of target cell t at window position w go to source cell t+offset(w)
        L = n * n
        o = out.view(B, L, win, Us * K).permute(0, 3, 2, 1).reshape(B, Us * K * win, L)
        k = 2 * self.rho + 1
        f = F.fold(o, (n + 2 * self.rho, n + 2 * self.rho), k)
        r = self.rho
        return f[:, :, r:r + n, r:r + n].reshape(B, Us, K, n, n)


class Head(nn.Module):
    """Conditional output market P(y[c]=1 | z_D[c]) = sigmoid(b + sum_u w_u[z_u])."""

    def __init__(self, U, K):
        super().__init__()
        self.w = nn.Parameter(torch.randn(U, K) * 0.1)
        self.b = nn.Parameter(torch.zeros(()))

    def logit(self, zoh):  # zoh (N,U,K) one-hot (or soft)
        return self.b + (zoh * self.w).sum((1, 2))

    @torch.no_grad()
    def message(self, zoh, y):
        """log P(y | z_u = v, other units as realised): (N,U,K)."""
        lg = self.logit(zoh)[:, None, None] - (zoh * self.w).sum(-1, keepdim=True) + self.w[None]
        s = 2 * y - 1
        return F.logsigmoid(s[:, None, None] * lg)


class Net(nn.Module):
    def __init__(self, D, U, K, hid, views, form="q"):
        super().__init__()
        self.D, self.U, self.K = D, U, K
        self.form = form
        self.rho = H_REACH // D
        tr = []
        for k in range(D):
            cz = 0 if k == 0 else U * K
            m = torch.ones(U, CX + cz)
            if views:  # unit 0: walls (+latents), unit 1: start marker (+latents)
                m[0, 1] = 0
                m[1, 0] = 0
            tr.append(Trader(self.rho, cz, U, K, hid, m))
        self.traders = nn.ModuleList(tr)
        self.head = Head(U, K)

    def rollout(self, x, mode, tau=1.0):
        """mode: sample | greedy | st. Returns lists of (feats, logits, one-hot z) per level."""
        B, _, n, _ = x.shape
        z = None
        levels = []
        for k in range(self.D):
            feats = unfold(x, z, self.rho)
            lg = self.traders[k](feats)
            if mode == "st":
                zoh = F.gumbel_softmax(lg, tau=tau, hard=True)
            elif mode == "sample":
                zoh = F.one_hot(torch.distributions.Categorical(logits=lg.detach()).sample(), self.K).float()
            else:
                zoh = F.one_hot(lg.argmax(-1), self.K).float()
            levels.append((feats, lg, zoh))
            z = zoh.view(B, n, n, self.U * self.K).permute(0, 3, 1, 2)
            if mode != "st":
                z = z.detach()
        return levels

    @torch.no_grad()
    def settle(self, x, y, levels):
        """Hindsight posteriors q_k (N,U,K) for every level, from the top down.
        form q: level k is settled against the settled q_{k+1} (target-propagation / mean-field).
        form l: level k gets the likelihood of the evidence, lam_k(v) = log P(y | z_k[c,u]=v, rest),
                propagated through the downstream conditional markets (sampled loopy BP)."""
        B, _, n, _ = x.shape
        qs = [None] * self.D
        feats, lg, zoh = levels[-1]
        lam = self.head.message(zoh, y.reshape(-1))
        qs[-1] = F.softmax(F.log_softmax(lg, -1) + lam, -1)
        for k in range(self.D - 2, -1, -1):
            if self.form == "q":
                m = self.traders[k + 1].message(levels[k + 1][0], qs[k + 1], self.U, self.K, B, n, "q")
            else:
                lam = lam - lam.max(-1, keepdim=True).values
                m = self.traders[k + 1].message(levels[k + 1][0], lam, self.U, self.K, B, n, "l")
            m = m.permute(0, 3, 4, 1, 2).reshape(-1, self.U, self.K)
            lam = m
            qs[k] = F.softmax(F.log_softmax(levels[k][1], -1) + m, -1)
        return qs


# ---------------------------------------------------------------- training
def train(args):
    torch.manual_seed(args.seed)
    views = args.arm in ("D", "Dl") if args.views == "auto" else args.views == "on"
    net = Net(args.D, args.U, args.K, args.hid, views=views, form="q" if args.arm in ("B", "D") else "l")
    data = make_set(args.pool, args.n, seed=1000 + args.seed)
    params = list(net.head.parameters()) if args.arm == "C" else list(net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, args.pool, (args.batch,), generator=g)
        x, y = data["x"][idx], data["y"][idx]
        yf = y.reshape(-1)
        if args.arm == "A":
            levels = net.rollout(x, "st", tau=args.tau)
            loss = F.binary_cross_entropy_with_logits(net.head.logit(levels[-1][2]), yf)
        elif args.arm == "C":
            with torch.no_grad():
                levels = net.rollout(x, "sample")
            loss = F.binary_cross_entropy_with_logits(net.head.logit(levels[-1][2]), yf)
        else:  # B, Bl, D, Dl: local settlement
            levels = net.rollout(x, "sample")
            qs = net.settle(x, y, levels)
            loss = F.binary_cross_entropy_with_logits(net.head.logit(levels[-1][2]), yf)
            for k in range(args.D):
                lp = F.log_softmax(levels[k][1], -1)
                loss = loss + (-(qs[k] * lp).sum(-1)).mean()
                if args.arm in ("D", "Dl"):
                    p = lp.exp()
                    pbar = p.mean(0, keepdim=True).detach()
                    loss = loss + args.lam * (p * (lp - torch.log(pbar + 1e-8))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss), "t": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    return net, log


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def _eval_chunk(net, x, y, S):
    B, _, n, _ = x.shape
    D = net.D
    pr = torch.zeros(B * n * n)
    kl = torch.zeros(D, B * n * n)
    for s in range(S):
        lv = net.rollout(x, "sample")
        pr += torch.sigmoid(net.head.logit(lv[-1][2]))
        qs = net.settle(x, y, lv)
        for k in range(D):
            lp = F.log_softmax(lv[k][1], -1)
            kl[k] += (qs[k] * (torch.log(qs[k] + 1e-12) - lp)).sum((1, 2))
    lv = net.rollout(x, "greedy")
    gp = torch.sigmoid(net.head.logit(lv[-1][2]))
    zs = [lv[k][2].argmax(-1) for k in range(D)]
    return pr / S, kl / S, gp, zs


@torch.no_grad()
def evaluate(net, ts, S, seed, keep_cells=False, chunk=6000):
    torch.manual_seed(seed)
    x, y = ts["x"], ts["y"]
    B, _, n, _ = x.shape
    D, U, K = net.D, net.U, net.K
    bs = max(1, chunk // (n * n))
    parts = [_eval_chunk(net, x[i:i + bs], y[i:i + bs], S) for i in range(0, B, bs)]
    pr = torch.cat([p[0] for p in parts]).clamp(1e-4, 1 - 1e-4)
    kl = torch.cat([p[1] for p in parts], 1)
    gp = torch.cat([p[2] for p in parts])
    zs = [torch.cat([p[3][k] for p in parts]) for k in range(D)]
    yf = y.reshape(-1)
    openm = ts["open"].reshape(-1)
    ll = -(yf * torch.log(pr) + (1 - yf) * torch.log(1 - pr))
    acc = ((pr > 0.5).float() == yf).float()
    gacc = ((gp > 0.5).float() == yf).float()
    om = torch.tensor(openm)
    dist = ts["dist"].reshape(-1)
    dcap = np.minimum(dist, H_REACH + 1)
    bnd = torch.tensor((np.abs(dist - H_REACH) <= 1) & openm)
    res = {
        "logloss_open": float(ll[om].mean()),
        "acc_open": float(acc[om].mean()),
        "greedy_acc_open": float(gacc[om].mean()),
        "acc_boundary": float(acc[bnd].mean()),
        "kl_level": [float(kl[k][om].mean()) for k in range(D)],
    }
    yo = yf.numpy()[openm].astype(int)
    align = []
    for k in range(D):
        zi = zs[k].numpy()
        zj = (zi * (K ** np.arange(U))).sum(1)[openm]
        gk = (dist <= (k + 1) * net.rho)[openm].astype(int)
        a = {"level": k + 1, "g_radius": (k + 1) * net.rho}
        a.update(alignment(zj, gk, yo, seed))
        ad = alignment(zj, dcap[openm], yo, seed)
        a["nmi_dist"], a["probe_dist"], a["probe_dist_base"] = ad["nmi_g"], ad["probe_g"], ad["probe_g_base"]
        a["nmi_onpath"] = alignment(zj, ts["onpath"].reshape(-1)[openm], yo, seed)["nmi_g"]
        err = (lut_fit(zj, gk)[zj] != gk)
        a["kl_auroc_err"] = auroc(kl[k].numpy()[openm], err)
        a["probe_err_rate"] = float(err.mean())
        align.append(a)
    res["align"] = align
    cells = None
    if keep_cells:
        ent = -(pr * torch.log(pr) + (1 - pr) * torch.log(1 - pr))
        cells = {"ll": ll.numpy().astype(np.float32), "kl_top": kl[-1].numpy().astype(np.float32),
                 "ent": ent.numpy().astype(np.float32)}
    return res, cells


def lut_fit(z, g):
    Z, G = int(z.max()) + 1, int(g.max()) + 1
    tab = np.zeros((Z, G))
    np.add.at(tab, (z, g), 1)
    return tab.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B", choices=["A", "B", "Bl", "C", "D", "Dl"])
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--views", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/maze")
    args = ap.parse_args()
    torch.set_num_threads(1)
    name = f"maze_{args.arm}{args.tag}_D{args.D}_s{args.seed}"
    net, log = train(args)
    out = {"args": vars(args), "log": log, "eval": {}}
    for size, num in ((9, 300), (15, 120), (21, 60)):
        ts = make_set(num, size, seed=777 + size)
        r, cells = evaluate(net, ts, args.S, seed=args.seed, keep_cells=(size == 9))
        out["eval"][str(size)] = r
        if cells is not None:
            np.savez_compressed(os.path.join(args.out, name + "_cells.npz"), **cells)
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(out, f)
    print("DONE", name, json.dumps({k: {kk: v[kk] for kk in ("logloss_open", "acc_open")} for k, v in out["eval"].items()}))


if __name__ == "__main__":
    main()
