"""M6 worker 2: output market layer, convex pools (c) k learned and (d) k from prequential wealth.

Market price l0 = per-question logistic baseline (S0 Logistic, Adam 1e-2), DETACHED; trained by
run_prequential as `baseline`. Trunk = S0 Net (d=64, L=2) up to the final LN; 16 trader heads per
question read the residual y [B,Q,d] and l0: d_i = <y, rw_i[q]> + c_i[q]*l0 + rb_i[q].

Arms
  a     plain logit sum          z = sum_i d_i                          (reference; w1 owns it)
  cg    convex pool, k learned   p = sum_i k_i sigma(l0 + d_i), k = softmax(theta), theta [16]
  cq    convex pool, k learned   same, theta [Q,16] (per question)
  d0    convex pool, k = prequential wealth, per question, pure Bayes/Kelly:
          log W_iq += committed_settlement(p_pool, p_i, y) after the label is revealed (trade-time
          p_i, p_pool from the TEST pass), k = softmax(log W). Detached (no gradient into k).
  dfs   same plus fixed share alpha=0.01 (Herbster & Warmuth 1998) after each update.
Pool loss = BCE(p, y); heads get gradient through the pool.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = "/home/agent/workspaces/eurisko-study/traders-net/study"
sys.path.insert(0, f"{ROOT}/S0/substrate"); sys.path.insert(0, f"{ROOT}/M1")
import streams_ext, prequential  # noqa
from model import Net, Logistic  # noqa
from currency import committed_settlement, LOCK  # noqa

K = 16
EPS = 1e-6


class Market(nn.Module):
    def __init__(self, Q, W, arm, base, alpha=0.0):
        super().__init__()
        self.arm, self.base, self.alpha, self.Q = arm, base, alpha, Q
        self.net = Net(Q, W, d=64, H=4, L=2)
        d = self.net.d
        self.rw = nn.Parameter(torch.randn(K, Q, d) / math.sqrt(d))
        self.rc = nn.Parameter(torch.zeros(K, Q))
        self.rb = nn.Parameter(torch.zeros(K, Q))
        if arm == "cg":
            self.theta = nn.Parameter(torch.zeros(K))
        elif arm == "cq":
            self.theta = nn.Parameter(torch.zeros(Q, K))
        self.register_buffer("logW", torch.zeros(Q, K))
        self.last = None

    def trunk(self, win):
        n = self.net
        B = win.shape[0]
        x = n.win(2.0 * win - 1.0) + n.qid.unsqueeze(0).expand(B, -1, -1)
        for blk in n.blocks:
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).view(B, n.Q, 3, n.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            x = x + blk.o((A @ v).transpose(1, 2).reshape(B, n.Q, n.d))
            x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
        return n.lnf(x)

    def forward(self, win):
        l0 = self.base(win).detach()                                  # [B,Q] market price
        y = self.trunk(win)                                           # [B,Q,d]
        d = torch.einsum("bqd,kqd->bkq", y, self.rw) + self.rc[None] * l0[:, None] + self.rb[None]
        if self.arm == "a":
            z = d.sum(1)
            self.last = None
            return z
        pi = torch.sigmoid(l0[:, None] + d)                           # [B,K,Q] trader beliefs
        if self.arm == "cg":
            k = torch.softmax(self.theta, 0)[None, :, None].expand(1, K, self.Q)
        elif self.arm == "cq":
            k = torch.softmax(self.theta, 1).T[None]                  # [1,K,Q]
        else:
            k = torch.softmax(self.logW, 1).T[None].detach()
        p = (k * pi).sum(1).clamp(EPS, 1 - EPS)
        if not self.training:
            self.last = (pi.detach(), p.detach(), k.detach())
        return torch.log(p) - torch.log1p(-p)                         # pool logit

    @torch.no_grad()
    def settle(self, Y, M):
        """Prequential wealth update from the cached TEST pass (trade-time beliefs)."""
        if self.arm not in ("d0", "dfs") or self.last is None:
            return
        pi, p, _ = self.last
        y = torch.as_tensor(Y, dtype=torch.float32)                   # [B,Q]
        m = torch.as_tensor(M, dtype=torch.float32)
        for b in range(y.shape[0]):                                   # sequential within block
            pay = committed_settlement(p[b][None].expand(K, -1), pi[b].clamp(EPS, 1 - EPS), y[b][None])  # [K,Q]
            self.logW += (pay * m[b][None]).T
            if self.alpha > 0:
                w = torch.softmax(self.logW, 1)
                w = (1 - self.alpha) * w + self.alpha / K
                self.logW.copy_(torch.log(w))
            self.logW -= self.logW.max(1, keepdim=True).values
        self.last = None


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=32000); ap.add_argument("--Q", type=int, default=64)
    ap.add_argument("--B", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=f"{ROOT}/M6/w2/results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    s = streams_ext.make_streams(Q=a.Q, T=a.T, seed=a.seed, heldout=8, release=a.T // 2, regime_every=0)
    st = streams_ext.to_array_stream(s)
    torch.manual_seed(a.seed)
    base = Logistic(st.Q, st.W)
    torch.manual_seed(1000 + a.seed)
    arm = a.arm
    m = Market(st.Q, st.W, arm, base, alpha=0.01 if arm == "dfs" else 0.0)
    own = [(n_, p) for n_, p in m.named_parameters() if not n_.startswith("base.")]
    opt = torch.optim.Adam([{"params": [p for n_, p in own if n_ != "theta"], "lr": a.lr},
                            {"params": [p for n_, p in own if n_ == "theta"], "lr": 10 * a.lr}])  # k lr 1e-2
    opt_b = torch.optim.Adam(base.parameters(), lr=1e-2)
    kent = []

    def on_step(s_, P, Z, Y, M):
        if m.last is not None and s_ % 50 == 0:
            k = m.last[2][0]                                          # [K,Q]
            kent.append((s_, float(torch.exp(-(k * torch.log(k + 1e-12)).sum(0)).mean())))
        m.settle(Y, M)

    t0 = time.time()
    r = prequential.run_prequential(m, st, opt, B=a.B, baseline=base, opt_b=opt_b, on_step=on_step)
    wall = time.time() - t0
    z, zb, y, mk = r["z"], r["zb"], r["y"], r["mask"]
    n = st.n_pos
    halves = {"all": slice(0, n), "late": slice(n // 2, n)}
    res = {"arm": arm, "seed": a.seed, "T": a.T, "Q": a.Q, "B": a.B, "lr": a.lr, "wall_s": wall,
           "families": streams_ext.FAMILIES, "currency_headline": LOCK["headline"]}
    for h, sl in halves.items():
        bn, _ = prequential.per_family(z[sl], y[sl], mk[sl], st.fam, st.n_fam)
        bb, _ = prequential.per_family(zb[sl], y[sl], mk[sl], st.fam, st.n_fam)
        res[f"bce_{h}"] = bn.tolist(); res[f"bce_l0_{h}"] = bb.tolist()
    if arm in ("d0", "dfs"):
        W = torch.softmax(m.logW, 1).numpy()
        res["wealth_gini_per_q_mean"] = float(np.mean([gini(W[q]) for q in range(st.Q)]))
        res["wealth_top_share_mean"] = float(W.max(1).mean())
    elif arm == "cg":
        k = torch.softmax(m.theta.detach(), 0).numpy(); res["k_final"] = k.tolist(); res["k_gini"] = gini(k)
    elif arm == "cq":
        k = torch.softmax(m.theta.detach(), 1).numpy()
        res["k_gini_per_q_mean"] = float(np.mean([gini(k[q]) for q in range(st.Q)])); res["k_top_share_mean"] = float(k.max(1).mean())
    res["k_eff_n_trace"] = kent
    fn = f"{a.out}/{arm}-s{a.seed}.json"
    json.dump(res, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print(fn, f"wall {wall:.0f}s")


if __name__ == "__main__":
    main()
