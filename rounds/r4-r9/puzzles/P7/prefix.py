"""P7 prefix-sum puzzle: hindsight-posterior settlement along a chain of latent traders.

Task. Bits x_1..x_L. Running state s_i = (x_1+...+x_i) mod 3 (the ground-truth sufficient
variable). Output y_i = 1[s_i == 0]: binary, so a latent that only copies the output
prediction (collapse) cannot carry the 3-valued state.

Chain depth d in {1,2,4}: y is revealed only at every d-th position, so d-1 latent traders
sit between consecutive resolved outputs and must be settled through the chain.
Train at L=16, test at L=16, 64, 256 (length generalisation). Primary metrics score the
positions the training pattern reveals ((i+1) % d == 0); all-position metrics are secondary
(a chain may legitimately use a phase-dependent code between revealed outputs).

Traders. Position i hosts a trader (weights shared across positions, U=2 units x K=4 values)
that reads (x_i, z_{i-1}) and emits p(z_i | x_i, z_{i-1}). The output market is a
conditional table P(y_i | z_i) (log-linear over units). Because the trader's inputs are
discrete, each trader is a table of conditional markets over (x, parent latent state).

Arms.
  A   end-to-end backprop through the chain (straight-through Gumbel-softmax)
  B   hindsight-posterior settlement, local only (sampled chain, top-down q-form):
        q_i[u](v) ∝ p_i[u](v) * P(y_i | z_i[u]=v, rest)^{observed}
                     * exp( sum_{u',v'} q_{i+1}[u'](v') log P_{i+1}(z_{i+1}[u']=v' | z_i[u]=v, rest) )
  Bx  exact settlement: the hindsight posterior of the whole chain by forward-backward
      (input-output HMM EM, Bengio & Frasconi 1995); each trader is trained on its pairwise
      posterior, no gradient crosses positions
  Bl  as B, but position i gets the likelihood of the later evidence through the next trader's
      conditional market (log sum_v' P(z_{i+1}=v' | z_i=v, rest) e^{lam_{i+1}(v')}) instead of the
      settled q_{i+1}: the hindsight posterior q ∝ p(z|x) P(y|z) applied recursively, sampled
  C   traders frozen at random init; only the output head is trained
  D   B + rate cost lam*KL(p || batch marginal) + different views (unit 1 does not see x)
  Dl  Bl + the same rate cost and views
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import alignment

MOD = 3


def make_set(num, L, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (num, L), generator=g)
    s = torch.cumsum(x, 1) % MOD
    return {"x": x, "s": s, "y": (s == 0).float()}


class Trader(nn.Module):
    def __init__(self, U, K, hid, views):
        super().__init__()
        self.U, self.K = U, K
        C = 2 + U * K
        self.W1 = nn.Parameter(torch.randn(U, hid, C) / math.sqrt(C))
        self.b1 = nn.Parameter(torch.zeros(U, hid))
        self.W2 = nn.Parameter(torch.randn(U, K, hid) / math.sqrt(hid))
        self.b2 = nn.Parameter(torch.zeros(U, K))
        m = torch.ones(U, C)
        if views:
            m[1, :2] = 0  # unit 1 is blind to x
        self.register_buffer("mask", m)
        J = K ** U
        # all parent states: J joint latent states + START (all-zero input)
        par = torch.zeros(J + 1, U * K)
        for a in range(J):
            for u in range(U):
                par[a, u * K + (a // K ** u) % K] = 1
        self.register_buffer("par", par)
        self.register_buffer("digits", torch.tensor([[(a // K ** u) % K for u in range(U)] for a in range(J)]))

    def forward(self, xoh, zin):  # xoh (N,2), zin (N,U*K) -> logits (N,U,K)
        inp = torch.cat([xoh, zin], -1)
        pre = torch.einsum("nc,uhc->nuh", inp, self.W1 * self.mask[:, None, :]) + self.b1
        return torch.einsum("nuh,ukh->nuk", F.relu(pre), self.W2) + self.b2

    def table(self):
        """logT[x, parent, u, k] for x in {0,1}, parent in 0..J (J = START)."""
        P = self.par.shape[0]
        xoh = torch.eye(2).repeat_interleave(P, 0)
        zin = self.par.repeat(2, 1)
        return F.log_softmax(self.forward(xoh, zin), -1).view(2, P, self.U, self.K)


class Head(nn.Module):
    def __init__(self, U, K):
        super().__init__()
        self.w = nn.Parameter(torch.randn(U, K) * 0.1)
        self.b = nn.Parameter(torch.zeros(()))

    def logit(self, zoh):  # (...,U,K)
        return self.b + (zoh * self.w).sum((-1, -2))


class Net(nn.Module):
    def __init__(self, U, K, hid, views):
        super().__init__()
        self.U, self.K, self.J = U, K, K ** U
        self.trader = Trader(U, K, hid, views)
        self.head = Head(U, K)

    def joint_digits_oh(self):
        return F.one_hot(self.trader.digits, self.K).float()  # (J,U,K)

    def rollout(self, x, mode, tau=1.0, logT=None):
        """Returns z one-hot (B,L,U,K), parent index (B,L), prior log-probs (B,L,U,K)."""
        B, L = x.shape
        U, K, J = self.U, self.K, self.J
        if mode == "st":
            zin = torch.zeros(B, U * K)
            zs, lps = [], []
            for i in range(L):
                lg = self.trader(F.one_hot(x[:, i], 2).float(), zin)
                zoh = F.gumbel_softmax(lg, tau=tau, hard=True)
                zs.append(zoh)
                lps.append(F.log_softmax(lg, -1))
                zin = zoh.reshape(B, U * K)
            return torch.stack(zs, 1), None, torch.stack(lps, 1)
        if logT is None:
            logT = self.trader.table()
        par = torch.full((B,), J, dtype=torch.long)
        zs, pars, lps = [], [], []
        powers = K ** torch.arange(U)
        for i in range(L):
            lp = logT[x[:, i], par]  # (B,U,K)
            if mode == "sample":
                zi = torch.distributions.Categorical(logits=lp.detach()).sample()
            else:
                zi = lp.argmax(-1)
            zs.append(zi)
            pars.append(par)
            lps.append(lp)
            par = (zi * powers).sum(-1)
        return F.one_hot(torch.stack(zs, 1), K).float(), torch.stack(pars, 1), torch.stack(lps, 1)

    @torch.no_grad()
    def settle_local(self, x, y, obs, zoh, par, lps, logT, form="q"):
        """Hindsight posteriors (B,L,U,K), top-down along the chain.
        form q: position i is settled against the settled q_{i+1} (target propagation / mean-field).
        form l: position i gets the likelihood of all later evidence, lam_i(v) = log P(y_i | z_i[u]=v, rest)
                + sum_u' log sum_v' P(z_{i+1}[u']=v' | z_i[u]=v, rest) e^{lam_{i+1}[u'](v')}."""
        B, L = x.shape
        U, K = self.U, self.K
        powers = K ** torch.arange(U)
        zi = zoh.argmax(-1)  # (B,L,U)
        # head message: log P(y_i | z_i[u]=v, rest)
        lg = self.head.logit(zoh)[..., None, None] - (zoh * self.head.w).sum(-1, keepdim=True) + self.head.w
        mh = F.logsigmoid((2 * y - 1)[..., None, None] * lg) * obs[None, :, None, None]
        qs = torch.zeros(B, L, U, K)
        lam = torch.zeros(B, U, K)
        vals = torch.arange(K)
        for i in range(L - 1, -1, -1):
            m = mh[:, i].clone()
            if i + 1 < L:
                q1 = qs[:, i + 1]  # (B,U',K')
                # parent of i+1 with unit u set to v: (B,U,K)
                alt = (zi[:, i] * powers).sum(-1)[:, None, None] - (zi[:, i] * powers)[:, :, None] + vals[None, None, :] * powers[None, :, None]
                lpn = logT[x[:, i + 1][:, None, None], alt]  # (B,U,K,U',K')
                if form == "q":
                    m = m + (lpn * q1[:, None, None]).sum((-1, -2))
                else:
                    ln = lam - lam.max(-1, keepdim=True).values
                    m = m + torch.logsumexp(lpn + ln[:, None, None], -1).sum(-1)
            lam = m
            qs[:, i] = F.softmax(lps[:, i].detach() + m, -1)
        return qs

    def settle_exact_loss(self, x, y, obs, logT):
        """Forward-backward over joint states; returns the M-step loss (targets detached) and
        the posterior marginals over joint states (B,L,J)."""
        B, L = x.shape
        J = self.J
        D = self.joint_digits_oh()  # (J,U,K)
        # joint log transition: lJ[x, a, b] = sum_u logT[x, a, u, b_u]
        lJ = (logT[:, :, None] * D[None, None]).sum((-1, -2))  # (2, J+1, J)
        hl = self.head.logit(D)  # (J,)
        le = F.logsigmoid((2 * y - 1)[..., None] * hl) * obs[None, :, None]  # (B,L,J)
        with torch.no_grad():
            T = lJ.exp()
            e = le.exp()
            alpha = torch.zeros(B, L, J)
            a = T[x[:, 0], J] * e[:, 0]
            alpha[:, 0] = a / a.sum(-1, keepdim=True)
            for i in range(1, L):
                a = torch.bmm(alpha[:, i - 1, None], T[x[:, i], :J])[:, 0] * e[:, i]
                alpha[:, i] = a / a.sum(-1, keepdim=True)
            beta = torch.ones(B, L, J)
            for i in range(L - 2, -1, -1):
                bb = torch.bmm(T[x[:, i + 1], :J], (e[:, i + 1] * beta[:, i + 1])[..., None])[..., 0]
                beta[:, i] = bb / bb.sum(-1, keepdim=True)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(-1, keepdim=True)
            # pairwise posteriors for i >= 1
            xi = alpha[:, :-1, :, None] * T[x[:, 1:], :J] * (e[:, 1:] * beta[:, 1:])[:, :, None, :]
            xi = xi / xi.sum((-1, -2), keepdim=True)  # (B,L-1,J,J)
        lt = -(xi * lJ[x[:, 1:], :J]).sum((-1, -2)).sum(1) - (gamma[:, 0] * lJ[x[:, 0], J]).sum(-1)
        lh = -(gamma * le).sum((-1, -2))
        return (lt + lh).mean() / L, gamma

    @torch.no_grad()
    def predictive(self, x):
        """Exact forward-filtered prior predictive P(y_i=1 | x_1..i) (no hindsight)."""
        B, L = x.shape
        J = self.J
        logT = self.trader.table()
        D = self.joint_digits_oh()
        T = (logT[:, :, None] * D[None, None]).sum((-1, -2)).exp()  # (2,J+1,J)
        py = torch.sigmoid(self.head.logit(D))  # (J,)
        pi = T[x[:, 0], J]
        out = [pi @ py]
        for i in range(1, L):
            pi = torch.bmm(pi[:, None], T[x[:, i], :J])[:, 0]
            out.append(pi @ py)
        return torch.stack(out, 1)


def obs_mask(L, d):
    return torch.tensor([1.0 if (i + 1) % d == 0 else 0.0 for i in range(L)])


def train(args):
    torch.manual_seed(args.seed)
    views = args.arm in ("D", "Dl") if args.views == "auto" else args.views == "on"
    net = Net(args.U, args.K, args.hid, views=views)
    form = "q" if args.arm in ("B", "D") else "l"
    params = list(net.head.parameters()) if args.arm == "C" else list(net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    obs = obs_mask(args.L, args.d)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        ds = make_set(args.batch, args.L, seed=args.seed * 10 ** 6 + step)
        x, y = ds["x"], ds["y"]
        if args.arm == "A":
            zoh, _, _ = net.rollout(x, "st", tau=args.tau)
            loss = (F.binary_cross_entropy_with_logits(net.head.logit(zoh), y, reduction="none") * obs).sum(1).mean() / obs.sum()
        elif args.arm == "C":
            with torch.no_grad():
                zoh, _, _ = net.rollout(x, "sample")
            loss = (F.binary_cross_entropy_with_logits(net.head.logit(zoh), y, reduction="none") * obs).sum(1).mean() / obs.sum()
        elif args.arm == "Bx":
            logT = net.trader.table()
            loss, _ = net.settle_exact_loss(x, y, obs, logT)
        else:  # B, Bl, D, Dl
            logT = net.trader.table()
            zoh, par, lps = net.rollout(x, "sample", logT=logT)
            qs = net.settle_local(x, y, obs, zoh, par, lps, logT, form)
            loss = (F.binary_cross_entropy_with_logits(net.head.logit(zoh), y, reduction="none") * obs).sum(1).mean() / obs.sum()
            loss = loss + (-(qs * lps).sum(-1)).mean()
            if args.arm in ("D", "Dl"):
                p = lps.exp()
                pbar = p.mean((0, 1), keepdim=True).detach()
                loss = loss + args.lam * (p * (lps - torch.log(pbar + 1e-8))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss), "t": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    return net, log


@torch.no_grad()
def evaluate(net, L, num, d, seed, form="l"):
    torch.manual_seed(seed)
    ds = make_set(num, L, seed=424242 + L)
    x, y, s = ds["x"], ds["y"], ds["s"]
    p = net.predictive(x).clamp(1e-4, 1 - 1e-4)
    ll = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    acc = ((p > 0.5).float() == y).float()
    zoh, _, _ = net.rollout(x, "greedy")
    gacc = ((torch.sigmoid(net.head.logit(zoh)) > 0.5).float() == y).float()
    zi = zoh.argmax(-1)
    zj = (zi * (net.K ** torch.arange(net.U))).sum(-1)
    al = alignment(zj.reshape(-1).numpy(), s.reshape(-1).numpy(), y.reshape(-1).numpy().astype(int), seed)
    # outputs at the positions the training pattern reveals ((i+1) % d == 0): the task's targets
    obs = obs_mask(L, d)
    om = obs.bool()
    al_obs = alignment(zj[:, om].reshape(-1).numpy(), s[:, om].reshape(-1).numpy(), y[:, om].reshape(-1).numpy().astype(int), seed)
    obs_last = om.clone()
    obs_last[: L - max(d, L // 4)] = False
    # hindsight gap KL(q||p) with the training observation pattern
    logT = net.trader.table()
    zs, par, lps = net.rollout(x, "sample", logT=logT)
    qs = net.settle_local(x, y, obs, zs, par, lps, logT, form)
    kl = (qs * (torch.log(qs + 1e-12) - lps)).sum((-1, -2))  # (B,L)
    L4 = max(1, L // 4)
    return {
        "logloss_obs": float(ll[:, om].mean()),
        "acc_obs": float(acc[:, om].mean()),
        "greedy_acc_obs": float(gacc[:, om].mean()),
        "acc_obs_last_quarter": float(acc[:, obs_last].mean()),
        "logloss": float(ll.mean()),
        "acc": float(acc.mean()),
        "greedy_acc": float(gacc.mean()),
        "acc_last_quarter": float(acc[:, -L4:].mean()),
        "logloss_by_pos": [float(v) for v in ll.mean(0)[:: max(1, L // 16)]],
        "kl_mean": float(kl.mean()),
        "kl_obs": float((kl * obs).sum() / (obs.sum() * num)),
        "kl_unobs": float((kl * (1 - obs)).sum() / max(1.0, float((1 - obs).sum()) * num)),
        "align": al,
        "align_obs": al_obs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B", choices=["A", "B", "Bl", "Bx", "C", "D", "Dl"])
    ap.add_argument("--views", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--U", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--hid", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--out", default="results/prefix")
    args = ap.parse_args()
    torch.set_num_threads(1)
    name = f"prefix_{args.arm}{args.tag}_d{args.d}_s{args.seed}"
    net, log = train(args)
    out = {"args": vars(args), "log": log, "eval": {}}
    for L, num in ((16, 2000), (64, 1000), (256, 500)):
        out["eval"][str(L)] = evaluate(net, L, num, args.d, args.seed, "q" if args.arm in ("B", "D") else "l")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(out, f)
    print("DONE", name, json.dumps({k: {kk: round(v[kk], 4) for kk in ("logloss_obs", "acc_obs", "acc")} for k, v in out["eval"].items()}),
          json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out["eval"]["16"]["align"].items()}))


if __name__ == "__main__":
    main()
