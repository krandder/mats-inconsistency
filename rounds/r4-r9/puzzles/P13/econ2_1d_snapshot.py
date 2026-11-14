"""Phase 2 (revised): a purely economic market ResNet. No backprop across depth.

Streams (per example, prefix-sum parity; y_i = x_1 xor ... xor x_i):
  x    raw input, re-read by every trader every step
  b    belief markets: one binary LMSR per output position (logit)
  c    stream 3, one of:
       cov   COVARIANCE markets: at level j, "is y_i == y_{i+2^j}?" per position,
             settled by the realized joint outcome (Kelvin 2026-09-29 rev. (b))
       slots PRIVATE slots: each trader owns a bank only it reads; any trader
             may write to it; the owner pays writers their Aumann-Shapley
             (path-integrated-gradient) share of its own earnings; rent per
             slot per step (rev. (c))
       none  no stream 3 (b-only economy)
  bounty (phase 2b, stream cov only): the cov markets are no longer paid by the
       house; their LMSR payout (the market maker's loss) is charged to traders.
         vol      each trader decides per episode (Bernoulli, REINFORCE on its own
                  profit) whether to fund; unfunded episodes settle no cov market;
                  funders split the payout
         lindahl  always open; the payout is charged in proportion to each
                  trader's Aumann-Shapley credit of c in its own b-earnings
  v    VOC markets (rev. (a)): "what fraction of position i's log loss is
       removed between step t and t+k", k in {5, 50, 500}; settled at t+k
       (or at the last step); used as value-of-computation for halting.

Traders: N modules, own weights + own Adam. A trader reads DETACHED state;
its only training signal is its own profit (MSR earnings on every market it
moves, + slot income - slot payments - rent - compute cost). Gradients stay
inside a trader. Mode "bptt": identical traders and streams, c/v are free
latents, trained end to end through depth on the final log loss (--ds: every
step's log loss; --covsup: also the covariance targets on c at every step;
--tbptt K: detach the state every K steps, so no gradient crosses more than K
steps; --train_pick random: a random trader per step in training, as the
economy does).
Phase-3b controls in mode bptt (audit of 09-29): --vocsup: v trained on the
economy's VOC targets at every step (as the economy's VOC markets pay);
--ptopt: one Adam optimizer and one gradient clip per trader, each trader's
loss the mean over its own steps (the economy's normalization; use with
--tbptt 1); --bidsup: the bid head regressed on each step's detached pay
(smooth L1, as the economy trains bids); --alpha: Bansal et al. 2022 mix
(1 - alpha) * loss at depth D + alpha * progressive loss (with --prog; 1 is
the pure progressive loss); --gradckpt: recompute each step in the backward
pass (gradient checkpointing, Chen et al. 2016; memory measurements only).

Belief-market regularizers (phase 3, economy only; all off by default):
  --floor eps   fixed-share floor: the belief forecast is (1-eps) sigmoid(b) + eps/2,
                in pay and in every reported log loss (loss <= log(2/eps) per bit)
  --l2b lam     the trader who moves a belief pays lam * (b_after^2 - b_before^2) per bit
  --bcap B      bounded liquidity: the belief logit is clamped to [-B, B] after every trade
  --dbcap d     trade-size cap per step: db -> d tanh(db / d)
  --liq t|sqrt  liquidity growing with depth: the belief LMSR at step t has liquidity
                beta_t = t (or sqrt t), beta_1 = 1; positions q carry over, so the opening
                logit at step t is q_{t-1} / beta_t (it relaxes toward 0 without trades), a
                trade dq moves it by dq / beta_t, and each trade settles as the log-score
                change at its own step's liquidity. With beta_t = t the logit is the running
                mean of the trades.

Architecture (shared across levels j so it applies to any length): at level
j, a per-position MLP reads x-features and b, c_j, c_{j-1} at offsets
0, +-2^j and writes dc_j(i) and a share of db(i).
"""
import argparse, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mresnet import prefix_batch

LS = F.logsigmoid
KS = [5, 50, 500]


def soft_score(z, p):
    return p * LS(z) + (1 - p) * LS(-z)


def belief_score(z, p, floor=0.0):
    """log score of the belief market; with floor eps the forecast is the fixed-share mix
    (1 - eps) sigmoid(z) + eps / 2, so a wrong bit costs at most log(2 / eps)"""
    if not floor:
        return soft_score(z, p)
    l1, l0 = math.log(1 - floor), torch.tensor(math.log(floor / 2), device=z.device)
    return p * torch.logaddexp(LS(z) + l1, l0) + (1 - p) * torch.logaddexp(LS(-z) + l1, l0)


def liq_beta(a, t):
    """liquidity of the belief market at step t (1-based)"""
    k = getattr(a, "liq", "")
    return 1.0 if not k else (float(t) if k == "t" else math.sqrt(t))


def shift(t, d):
    """out[..., i] = t[..., i + d] (zero outside)."""
    if d == 0:
        return t
    L = t.shape[-1]
    if abs(d) >= L:
        return torch.zeros_like(t)
    if d > 0:
        return F.pad(t[..., d:], (0, d))
    return F.pad(t[..., :d], (-d, 0))


def nlevels(L):
    return max(1, math.ceil(math.log2(max(L, 2))))


class Trader(nn.Module):
    def __init__(self, w=96, we=32):
        super().__init__()
        self.emb = nn.Conv1d(3, we, 3, padding=1)
        cf = 2 * we + 3 + 5 + 3 + 1
        self.mlp = nn.Sequential(nn.Conv1d(cf, w, 1), nn.ReLU(), nn.Conv1d(w, w, 1), nn.ReLU())
        self.out = nn.Conv1d(w, 2 + 3, 1)          # dc_j, db_j, dv(3)
        self.gain = nn.Sequential(nn.Conv1d(w, 32, 1), nn.ReLU(), nn.Conv1d(32, 1, 1))

    def forward(self, xoh, b, c, v, mask):
        """b (B,L), c (B,J,L), v (B,3,L). Returns db, dc, dv, per-bit gain bid."""
        B, J, L = c.shape
        e = self.emb(xoh)                                   # (B,we,L)
        tb, tc, tv = torch.tanh(b / 4), torch.tanh(c / 4), torch.tanh(v / 4)
        feats = []
        for j in range(J):
            d = 2 ** j
            cj = tc[:, j]
            cprev = tc[:, j - 1] if j > 0 else torch.zeros_like(cj)
            f = torch.cat([e, shift(e, d),
                           torch.stack([tb, shift(tb, d), shift(tb, -d)], 1),
                           torch.stack([cj, shift(cj, d), shift(cj, -d), cprev, shift(cprev, d // 2 if j else 0)], 1),
                           tv, mask.unsqueeze(1) * 0 + 1.0], 1)
            feats.append(f)
        f = torch.stack(feats, 1).flatten(0, 1)             # (B*J, cf, L)
        z = self.mlp(f)
        o = self.out(z).view(B, J, 5, L) * mask.view(B, 1, 1, L)
        dc = o[:, :, 0]
        db = o[:, :, 1].sum(1)
        dv = o[:, :, 2:].mean(1)
        g = self.gain(z.detach()).view(B, J, L).sum(1)
        bid = (g * mask).sum(1) / mask.sum(1)
        return db, dc, dv, bid


def pair_truth(y, mask, J):
    eq, m = [], []
    for j in range(J):
        d = 2 ** j
        eq.append((y == shift(y, d)).float())
        m.append(mask * shift(mask, d))
    return torch.stack(eq, 1), torch.stack(m, 1)


def worst_loss_per_bit(b, db, mask):
    lo1 = LS(b) - LS(b + db)
    lo0 = LS(-b) - LS(-(b + db))
    return (torch.maximum(lo1, lo0).clamp(min=0) * mask).sum(1) / mask.sum(1)


class Economy:
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.traders = [self.new_trader() for _ in range(a.N)]
        self.wealth = [a.W0] * a.N
        self.stats = {"respawns": 0, "entries": 0, "slot_transfers": 0.0, "rent": 0.0}
        self.fund = [[0.0, 0.0, 1.0] for _ in range(a.N)]   # bounty vol: logit, reward mean, reward var
        if getattr(a, "bounty", ""):
            self.stats.update(bounty_paid=0.0, funded_eps=0)

    def new_trader(self):
        t = Trader(w=self.a.width).to(self.dev)
        return [t, torch.optim.Adam(t.parameters(), lr=self.a.lr)]

    def add_trader(self):
        self.traders.append(self.new_trader()); self.wealth.append(self.a.W0)
        self.fund.append([0.0, 0.0, 1.0])
        self.stats["entries"] += 1

    # ---------------------------------------------------------------- rollout
    def rollout(self, x, mask, D, pick="random", gen=None, keep=True, halt=None, J=None,
                explore=None, signals=False, record_at=None):
        """No-grad forward for D steps. pick: random | roundrobin | auction.
        halt: None | ("bid", cost) | ("voc", cost). explore: (B,) 0/1, examples that
        continue after the halt rule fires (epsilon-continuation). signals: record
        the per-step halting signals (bid, voc). record_at: keep b only at these
        depths (memory O(1) in D). Returns records and trajectories."""
        a = self.a
        B, L = x.shape
        J = J or nlevels(L)
        N = len(self.traders)
        xoh = F.one_hot(x, 3).float().transpose(1, 2)
        b = torch.zeros(B, L, device=self.dev)
        v = torch.zeros(B, 3, L, device=self.dev)
        if a.stream == "slots":
            banks = torch.zeros(B, N, J, L, device=self.dev)
        c = torch.zeros(B, J, L, device=self.dev)
        recs, bs = [], [b]
        if record_at is not None:
            bs = {0: b} if 0 in record_at else {}
        active = torch.ones(B, device=self.dev)
        wanted = torch.ones(B, device=self.dev)
        self.fired = torch.full((B,), float(D + 10 ** 6), device=self.dev)  # first step the halt rule fired
        self.sig = []
        steps = torch.zeros(B, device=self.dev)
        liq = getattr(a, "liq", "")
        dbcap, bcap = getattr(a, "dbcap", 0), getattr(a, "bcap", 0)
        with torch.no_grad():
            for t in range(D):
                beta = liq_beta(a, t + 1)
                if liq and t > 0:   # the liquidity grows from beta_t to beta_{t+1}: the opening logit relaxes
                    b = b * (1 - active.view(B, 1) * (1 - liq_beta(a, t) / beta))
                if pick == "auction":
                    outs = []
                    for i in range(N):
                        ci = banks[:, i] if a.stream == "slots" else c
                        outs.append(self.traders[i][0](xoh, b, ci, v, mask))
                    bids = torch.stack([o[3] for o in outs], 1)
                    who = bids.argmax(1)
                    ar = torch.arange(B, device=self.dev)
                    db = torch.stack([o[0] for o in outs], 1)[ar, who]
                    dc = torch.stack([o[1] for o in outs], 1)[ar, who]
                    dv = torch.stack([o[2] for o in outs], 1)[ar, who]
                    bestbid = bids.max(1).values
                else:
                    i = (int(torch.randint(N, (1,), generator=gen)) if pick == "random" else t % N)
                    ci = banks[:, i] if a.stream == "slots" else c
                    db, dc, dv, bestbid = self.traders[i][0](xoh, b, ci, v, mask)
                    who = torch.full((B,), i, device=self.dev, dtype=torch.long)
                if dbcap:
                    db = dbcap * torch.tanh(db / dbcap)
                if liq:
                    db = db / beta
                if halt is not None or signals:
                    # VOC: expected nats/bit removed per step over the next 5 steps (entropy as LL proxy)
                    p = torch.sigmoid(b)
                    H = -(p * LS(b) + (1 - p) * LS(-b))
                    voc = (torch.sigmoid(v[:, 0]) * H * mask).sum(1) / mask.sum(1) / KS[0]
                    if signals:
                        self.sig.append(torch.stack([bestbid, voc], 1))
                if halt is not None and t > 0:
                    kind, cost = halt
                    go = (bestbid > cost) if kind == "bid" else (voc > cost)
                    newly = (wanted > 0) & ~go
                    self.fired = torch.where(newly, torch.full_like(self.fired, float(t)), self.fired)
                    wanted = wanted * go.float()
                    active = wanted if explore is None else torch.maximum(wanted, explore)
                s = active.clone()
                if a.wealth_cap:
                    wl = worst_loss_per_bit(b, db, mask)
                    wc = torch.tensor(self.wealth, device=self.dev).clamp(min=0)[who]
                    s = s * torch.clamp(a.frac * wc / (wl + 1e-8), max=1.0)
                if keep:
                    recs.append({"b": b, "c": (banks.clone() if a.stream == "slots" else c), "v": v,
                                 "who": who, "s": s, "act": active.clone(), "beta": beta})
                b = b + s.view(B, 1) * db
                if bcap:
                    b = b.clamp(-bcap, bcap)
                v = v + s.view(B, 1, 1) * dv
                if a.stream == "cov":
                    c = c + s.view(B, 1, 1) * dc
                elif a.stream == "slots":
                    # the acting trader writes into every OTHER trader's bank (writes to own bank allowed too)
                    banks = banks + s.view(B, 1, 1, 1) * dc.unsqueeze(1)
                    if keep:
                        recs[-1]["write"] = s.view(B, 1, 1) * dc
                if getattr(a, "clamp", 0):   # test only: cap every logit (checks that the runaway does not feed back)
                    b, c, v = b.clamp(-a.clamp, a.clamp), c.clamp(-a.clamp, a.clamp), v.clamp(-a.clamp, a.clamp)
                steps += active
                if record_at is None:
                    bs.append(b)
                elif t + 1 in record_at:
                    bs[t + 1] = b
        return recs, bs, steps

    # ---------------------------------------------------------------- training
    def train_step(self, x, y, mask, D, gen):
        """Random allocation (one trader per step for the whole batch); each
        trader then updates on ITS OWN trades only, all of them in one batch.
        Bids are trained on these randomly allocated trades (unbiased); the
        auction is used at inference."""
        a = self.a
        B, L = x.shape
        J = nlevels(L)
        N = len(self.traders)
        # VOC halting during training (decision market): real = settle against the
        # realized trajectory (self-fulfilling after a halt); eps = continue with
        # prob eps after the rule fires and settle windows that cross the halt
        # only on continued episodes, weighted 1/eps (Chen et al. 2014)
        th = a.train_halt >= 0 and getattr(self, "it", 0) > a.iters // 5
        explore = None
        if th and a.settle == "eps":
            explore = (torch.rand(B, generator=gen) < a.eps).float().to(self.dev)
        recs, bs, steps = self.rollout(x, mask, D, pick=getattr(a, "train_pick", "") or "random", gen=gen, J=J,
                                       halt=("voc", a.train_halt) if th else None, explore=explore)
        fired = self.fired
        bty = getattr(a, "bounty", "") if a.stream == "cov" else ""
        funded, fdec = 1.0, [1.0] * N
        if bty == "vol":
            fprob = [1 / (1 + math.exp(-f[0])) for f in self.fund]
            fdec = [float(torch.rand(1, generator=gen).item() < p) for p in fprob]
            funded = float(any(fdec))
        covpay_tot, ig_credit = 0.0, torch.zeros(N)
        who = [int(r["who"][0]) for r in recs]
        xoh = F.one_hot(x, 3).float().transpose(1, 2)
        nb = mask.sum(1)
        eq, meq = pair_truth(y, mask, J)
        fl = getattr(a, "floor", 0)
        LL = torch.stack([-belief_score(bb, y, fl) for bb in bs], 0)       # (D+1,B,L)
        profit = torch.zeros(N, device=self.dev)
        slot_income = torch.zeros(N, device=self.dev)
        ntr = [who.count(i) for i in range(N)]
        prices = {}
        rep = lambda z, n: z.repeat(n, *([1] * (z.dim() - 1)))
        for i, (tr, opt) in enumerate(self.traders):
            ts = [t for t in range(D) if who[t] == i]
            if not ts:
                continue
            n = len(ts)
            bi = torch.cat([recs[t]["b"] for t in ts]); vi = torch.cat([recs[t]["v"] for t in ts])
            ci = torch.cat([recs[t]["c"][:, i] if a.stream == "slots" else recs[t]["c"] for t in ts])
            si = torch.cat([recs[t]["s"] for t in ts]).view(-1, 1)
            ai = torch.cat([recs[t]["act"] for t in ts])                     # step executed (not halted)
            past = torch.cat([(fired <= t).float() for t in ts])            # halt rule had fired by step t
            X, Y, Mk, NB = rep(xoh, n), rep(y, n), rep(mask, n), rep(nb, n)
            db, dc, dv, bid = tr(X, bi, ci, vi, Mk)
            if getattr(a, "dbcap", 0):
                db = a.dbcap * torch.tanh(db / a.dbcap)
            if getattr(a, "liq", ""):
                db = db / torch.cat([torch.full((B, 1), recs[t]["beta"], device=self.dev) for t in ts])
            bnew = bi + si * db
            if getattr(a, "bcap", 0):
                bnew = bnew.clamp(-a.bcap, a.bcap)
            pay = ((belief_score(bnew, Y, fl) - belief_score(bi, Y, fl)) * Mk).sum(1) / NB
            if getattr(a, "l2b", 0):
                pay = pay - a.l2b * ((bnew ** 2 - bi ** 2) * Mk).sum(1) / NB
            if a.stream == "cov":
                EQ, MEQ = rep(eq, n), rep(meq, n)
                cp = a.lam_c * ((soft_score(ci + si.unsqueeze(1) * dc, EQ) - soft_score(ci, EQ)) * MEQ).sum((1, 2)) / NB
                pay = pay + funded * cp
                covpay_tot += funded * cp.detach().sum().item() / B
                if bty == "lindahl":   # Aumann-Shapley credit of c in this trader's b-earnings (4-point path)
                    cr = torch.zeros((), device=self.dev)
                    for al in [0.125, 0.375, 0.625, 0.875]:
                        ca = (ci.detach() * al).requires_grad_(True)
                        dba, _, _, _ = tr(X, bi, ca, vi, Mk)
                        pa = (((soft_score(bi + si * dba, Y) - soft_score(bi, Y)) * Mk).sum(1) / NB).sum()
                        cr = cr + (torch.autograd.grad(pa, ca)[0] * ci.detach()).sum() / 4
                    ig_credit[i] = cr.item() / B
            if a.voc:
                vt = torch.cat([torch.stack([((LL[t] - LL[min(t + k, D)]) / (LL[t] + 1e-6)).clamp(0, 1)
                                             for k in KS], 1) for t in ts])
                wv = torch.ones_like(vt[:, :, :1])
                if th and a.settle == "eps":
                    one = torch.ones_like(fired)
                    wv = torch.cat([torch.stack([torch.where(fired >= min(t + k, D), one, explore / a.eps)
                                                 for k in KS], 1) for t in ts]).unsqueeze(2)
                pay = pay + a.lam_v * ((soft_score(vi + si.unsqueeze(1) * dv, vt) - soft_score(vi, vt))
                                       * Mk.unsqueeze(1) * wv).sum((1, 2)) / NB
            prof = pay - a.cost * ai          # compute cost only for steps that ran
            if a.stream == "slots":
                # Aumann-Shapley price of the owner's bank: path-averaged gradient of its
                # b-earnings along alpha*bank (midpoint rule, 4 points)
                g = torch.zeros_like(ci)
                for al in [0.125, 0.375, 0.625, 0.875]:
                    ca = (ci.detach() * al).requires_grad_(True)
                    dba, _, _, _ = tr(X, bi, ca, vi, Mk)
                    pa = (((soft_score(bi + si * dba, Y) - soft_score(bi, Y)) * Mk).sum(1) / NB).sum()
                    g += torch.autograd.grad(pa, ca)[0] / 4
                for k_, t in enumerate(ts):
                    prices[t] = (i, g[k_ * B:(k_ + 1) * B].detach())
                rent = a.rent * (ci.abs() * Mk.unsqueeze(1)).sum((1, 2)) / NB
                prof = prof - rent
                self.stats["rent"] += rent.sum().item() / B
            # the bid predicts earnings BEFORE the compute cost: halting compares it with
            # the cost once (bid > cost). Training it on pay - cost and then comparing with
            # cost counted the cost twice (reviewer, 09-29; no effect at --cost 0, used by all runs)
            # Bid settlement under train-time halting (a decision market, like the VOC
            # markets): real = settle every example, so a halted step's bid is paid 0
            # and a low bid confirms itself; eps = settle only steps that ran, the
            # epsilon-continued ones after the halt with weight 1/eps.
            bw = torch.ones_like(ai)
            if th and a.settle == "eps":
                bw = ai * torch.where(past > 0, torch.full_like(past, 1.0 / a.eps), torch.ones_like(past))
            gl = (F.smooth_l1_loss(bid, (prof + a.cost * ai).detach(), reduction="none") * bw).mean()
            obj = -pay.mean() + gl
            opt.zero_grad(); obj.backward()
            torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step()
            profit[i] = prof.detach().sum() / B
        if a.stream == "slots":
            # a write at step t is paid <write, price_t2> by every later owner trade t2 > t
            # (owner != writer unless self_pay); owners pay the same (conserved transfer)
            for w_i, (tr, opt) in enumerate(self.traders):
                ts = [t for t in range(D) if who[t] == w_i]
                if not ts:
                    continue
                P = []
                for t in ts:
                    acc = torch.zeros(B, J, L, device=self.dev)
                    for t2 in range(t + 1, D):
                        o, pr = prices[t2]
                        if o == w_i and not a.self_pay:
                            continue
                        acc = acc + pr
                    P.append(acc)
                n = len(ts)
                bi = torch.cat([recs[t]["b"] for t in ts]); vi = torch.cat([recs[t]["v"] for t in ts])
                ci = torch.cat([recs[t]["c"][:, w_i] for t in ts])
                si = torch.cat([recs[t]["s"] for t in ts]).view(-1, 1, 1)
                _, dc, _, _ = tr(rep(xoh, n), bi, ci, vi, rep(mask, n))
                inc = (si * dc * torch.cat(P) * rep(mask, n).unsqueeze(1)).sum((1, 2))
                opt.zero_grad(); (-inc.mean()).backward()
                torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step()
                slot_income[w_i] += inc.detach().sum() / B
                # owners pay: split by owner
                for k_, t in enumerate(ts):
                    wr = (si[k_ * B:(k_ + 1) * B] * dc[k_ * B:(k_ + 1) * B]).detach()
                    for t2 in range(t + 1, D):
                        o, pr = prices[t2]
                        if o == w_i and not a.self_pay:
                            continue
                        slot_income[o] -= (wr * pr * mask.unsqueeze(1)).sum() / B
            self.stats["slot_transfers"] += slot_income.clamp(min=0).sum().item()
        if bty:
            charge = torch.zeros(N, device=self.dev)
            if bty == "vol" and funded:
                fs = [i for i in range(N) if fdec[i]]
                for i in fs:
                    charge[i] = covpay_tot / len(fs)
            elif bty == "lindahl":
                w = ig_credit.clamp(min=0)
                w = w / w.sum() if w.sum() > 0 else torch.full((N,), 1.0 / N)
                charge = covpay_tot * w.to(self.dev)
            profit = profit - charge
            if bty == "vol":   # REINFORCE on own episode profit, normalized advantage
                for i in range(N):
                    f = self.fund[i]; R = profit[i].item()
                    adv = (R - f[1]) / math.sqrt(f[2] + 1e-12)
                    p_ = 1 / (1 + math.exp(-f[0]))
                    f[0] = max(-8.0, min(8.0, f[0] + a.fund_lr * adv * (fdec[i] - p_)))
                    f[1] = 0.99 * f[1] + 0.01 * R; f[2] = 0.99 * f[2] + 0.01 * (R - f[1]) ** 2
            self.stats["bounty_paid"] += covpay_tot; self.stats["funded_eps"] += int(funded)
        for i in range(N):
            self.wealth[i] += (profit[i] + slot_income[i]).item()
        for i in range(N):
            if a.wealth_cap and self.wealth[i] <= 0:
                self.traders[i] = self.new_trader(); self.wealth[i] = a.W0; self.fund[i] = [0.0, 0.0, 1.0]
                self.stats["respawns"] += 1
        Sf = ((belief_score(bs[-1], y, fl) * mask).sum(1) / nb).mean().item()
        return {"final_nll_bit": -Sf, "profit": [round(p, 4) for p in profit.tolist()],
                "slot_income": [round(p, 4) for p in slot_income.tolist()], "ntrades": ntr,
                "wealth": [round(w, 2) for w in self.wealth], "train_steps": steps.mean().item(),
                "fund_p": [round(1 / (1 + math.exp(-f[0])), 3) for f in self.fund], "funded": funded,
                "cov_payout": covpay_tot}

    def train_step_online(self, x, y, mask, D, gen):
        """Streaming version of train_step (streams cov/none, no VOC): each trade is
        settled at once (b and cov markets; y is known in training) and its gradient
        accumulated in the acting trader; nothing is kept across steps, so memory
        is O(1) in depth. Parameters change once per episode, so the gradients equal
        train_step's."""
        a = self.a
        assert a.stream in ("cov", "none") and not a.voc and a.train_halt < 0
        B, L = x.shape
        J = nlevels(L)
        N = len(self.traders)
        xoh = F.one_hot(x, 3).float().transpose(1, 2)
        nb = mask.sum(1)
        eq, meq = pair_truth(y, mask, J)
        b = torch.zeros(B, L, device=self.dev)
        c = torch.zeros(B, J, L, device=self.dev)
        v = torch.zeros(B, 3, L, device=self.dev)
        for tr, opt in self.traders:
            opt.zero_grad()
        cnt = [0] * N
        profit = torch.zeros(N, device=self.dev)
        W = torch.tensor(self.wealth, device=self.dev).clamp(min=0)
        for t in range(D):
            i = int(torch.randint(N, (1,), generator=gen))
            tr = self.traders[i][0]
            db, dc, dv, bid = tr(xoh, b, c, v, mask)
            with torch.no_grad():
                s = torch.ones(B, device=self.dev)
                if a.wealth_cap:
                    s = torch.clamp(a.frac * W[i] / (worst_loss_per_bit(b, db, mask) + 1e-8), max=1.0)
            pay = ((soft_score(b + s.view(B, 1) * db, y) - soft_score(b, y)) * mask).sum(1) / nb
            if a.stream == "cov":
                pay = pay + a.lam_c * ((soft_score(c + s.view(B, 1, 1) * dc, eq) - soft_score(c, eq)) * meq).sum((1, 2)) / nb
            prof = pay - a.cost
            (-pay.mean() + F.smooth_l1_loss(bid, (prof + a.cost).detach())).backward()  # bid: before compute cost
            cnt[i] += 1
            profit[i] += prof.detach().sum() / B
            with torch.no_grad():
                b = b + s.view(B, 1) * db.detach()
                v = v + s.view(B, 1, 1) * dv.detach()
                if a.stream == "cov":
                    c = c + s.view(B, 1, 1) * dc.detach()
        for i, (tr, opt) in enumerate(self.traders):
            if cnt[i]:
                for p_ in tr.parameters():
                    if p_.grad is not None:
                        p_.grad /= cnt[i]
                torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step()
            self.wealth[i] += profit[i].item()
            if a.wealth_cap and self.wealth[i] <= 0:
                self.traders[i] = self.new_trader(); self.wealth[i] = a.W0; self.fund[i] = [0.0, 0.0, 1.0]
                self.stats["respawns"] += 1
        Sf = ((soft_score(b, y) * mask).sum(1) / nb).mean().item()
        return {"final_nll_bit": -Sf, "profit": [round(p, 4) for p in profit.tolist()], "ntrades": cnt,
                "wealth": [round(w, 2) for w in self.wealth]}


class MultiOpt:
    """One Adam per trader (--ptopt); state_dict is the list of their states."""
    def __init__(self, mods, lr):
        self.opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in mods]

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, d in zip(self.opts, sd):
            o.load_state_dict(d)


class BPTT:
    """Same traders and streams, c and v as free latents; end to end through depth."""
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.traders = nn.ModuleList([Trader(w=a.width) for _ in range(a.N)]).to(device)
        if getattr(a, "ptopt", 0):
            self.opt = MultiOpt(self.traders, a.lr)
        else:
            self.opt = torch.optim.Adam(self.traders.parameters(), lr=a.lr)

    def rollout(self, x, mask, D, grad=False, J=None, init=None, t0=0, record_at=None, gen=None, **kw):
        B, L = x.shape
        tb = getattr(self.a, "tbptt", 0) if grad else 0
        rand = gen is not None and getattr(self.a, "train_pick", "") == "random"
        C = getattr(self.a, "clamp", 0)
        J = J or nlevels(L)
        xoh = F.one_hot(x, 3).float().transpose(1, 2)
        if init is None:
            b = torch.zeros(B, L, device=self.dev)
            c = torch.zeros(B, J, L, device=self.dev)
            v = torch.zeros(B, 3, L, device=self.dev)
        else:
            b, c, v = init
        bs = [b] if record_at is None else ({0: b} if 0 in record_at else {})
        self.cs = []  # every step's c, kept only when kw["rec_c"] (for --covsup)
        self.vs, self.bids, self.who = [], [], []   # every step's v (--vocsup), bid (--bidsup), trader index
        gck = grad and getattr(self.a, "gradckpt", 0)
        with torch.set_grad_enabled(grad):
            for t in range(D):
                if tb and t and t % tb == 0:   # truncated BPTT: no gradient crosses this step
                    b, c, v = b.detach(), c.detach(), v.detach()
                i = int(torch.randint(len(self.traders), (1,), generator=gen)) if rand else (t0 + t) % len(self.traders)
                if gck:
                    db, dc, dv, bid = torch.utils.checkpoint.checkpoint(self.traders[i], xoh, b, c, v, mask, use_reentrant=False)
                else:
                    db, dc, dv, bid = self.traders[i](xoh, b, c, v, mask)
                self.who.append(i)
                if kw.get("rec_bid"):
                    self.bids.append((bid, b, c, v))
                b = b + db
                v = v + dv
                if self.a.stream != "none":
                    c = c + dc
                if C:
                    b, c, v = b.clamp(-C, C), c.clamp(-C, C), v.clamp(-C, C)
                if kw.get("rec_c"):
                    self.cs.append(c)
                if kw.get("rec_v"):
                    self.vs.append(v)
                if record_at is None:
                    bs.append(b)
                elif t + 1 in record_at:
                    bs[t + 1] = b
        self.last = (b, c, v)
        return None, bs, torch.full((B,), float(D))

    def train_step(self, x, y, mask, D, gen):
        a = self.a
        nb = mask.sum(1)
        if getattr(a, "ptopt", 0) or getattr(a, "vocsup", 0) or getattr(a, "bidsup", 0) or getattr(a, "alpha", 1.0) != 1.0:
            return self.train_step_p3b(x, y, mask, D, gen)
        if getattr(self.a, "prog", 0):
            # Bansal et al. 2022 progressive loss (alpha = 1): n no-grad steps, then
            # k steps with grad from the detached state, loss at step n + k
            n = int(torch.randint(0, D, (1,), generator=gen))
            k = int(torch.randint(1, D - n + 1, (1,), generator=gen))
            self.rollout(x, mask, n, grad=False, gen=gen)
            init = tuple(z.detach() for z in self.last)
            # --covsup: the covariance targets on c at the k steps with grad (before 09-29 evening
            # prog ignored --covsup; no reported run combined them)
            _, bs, _ = self.rollout(x, mask, k, grad=True, init=init, t0=n, gen=gen,
                                    rec_c=bool(getattr(self.a, "covsup", 0)))
        else:
            _, bs, _ = self.rollout(x, mask, D, grad=True, gen=gen, rec_c=bool(getattr(self.a, "covsup", 0)))
        final = -((soft_score(bs[-1], y) * mask).sum(1) / nb).mean()
        loss = final
        if getattr(self.a, "ds", 0):
            # deep supervision: the backprop analogue of the economy's per-step payments,
            # the mean log score over every step's belief (still one graph through depth)
            loss = -torch.stack([((soft_score(bt, y) * mask).sum(1) / nb).mean() for bt in bs[1:]]).mean()
        if getattr(self.a, "covsup", 0) and self.cs:
            # supervise c as the economy's covariance markets do ("is y_i = y_(i+2^j)?", log score,
            # weight lam_c), at every step: the same targets the economy's traders are paid on
            eq, meq = pair_truth(y, mask, self.cs[0].shape[1])
            loss = loss - self.a.lam_c * torch.stack(
                [((soft_score(ct, eq) * meq).sum((1, 2)) / nb).mean() for ct in self.cs]).mean()
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.traders.parameters(), 1.0)
        self.opt.step()
        return {"final_nll_bit": final.item()}



    def train_step_p3b(self, x, y, mask, D, gen):
        """The phase-3b controls (--vocsup, --ptopt, --bidsup, --alpha); the plain arms keep train_step."""
        a = self.a
        nb = mask.sum(1)
        cov, vsup, bsup, pt = bool(getattr(a, "covsup", 0)), bool(getattr(a, "vocsup", 0)), bool(getattr(a, "bidsup", 0)), bool(getattr(a, "ptopt", 0))
        mb = lambda z: ((z * mask).sum(1) / nb)                           # (B,) mean over real bits

        def part(bs, full_steps):
            """loss of one rollout: final, or every step's (ds), + cov, VOC and bid targets per step"""
            T = len(bs) - 1
            w = torch.ones(T, device=self.dev)
            if pt:   # each trader's loss is the mean over its own steps (the economy's pay.mean())
                cnt = torch.bincount(torch.tensor(self.who), minlength=len(self.traders)).float()
                w = (T / cnt[torch.tensor(self.who)]).to(self.dev)
            if getattr(a, "ds", 0):
                L_ = -(torch.stack([mb(soft_score(bt, y)).mean() for bt in bs[1:]]) * w).mean()
            else:
                L_ = -mb(soft_score(bs[-1], y)).mean()
            if cov and self.cs:
                eq, meq = pair_truth(y, mask, self.cs[0].shape[1])
                L_ = L_ - a.lam_c * (torch.stack([((soft_score(ct, eq) * meq).sum((1, 2)) / nb).mean()
                                                  for ct in self.cs]) * w).mean()
            if vsup and self.vs and full_steps:
                with torch.no_grad():
                    LL = torch.stack([-soft_score(bt.detach(), y) for bt in bs], 0)       # (T+1,B,L)
                    vts = [torch.stack([((LL[t] - LL[min(t + k, T)]) / (LL[t] + 1e-6)).clamp(0, 1) for k in KS], 1)
                           for t in range(T)]
                L_ = L_ - a.lam_v * (torch.stack([((soft_score(vt_, tg) * mask.unsqueeze(1)).sum((1, 2)) / nb).mean()
                                                  for vt_, tg in zip(self.vs, vts)]) * w).mean()
            if bsup and self.bids:
                gl = []
                for t, (bid, b0, c0, v0) in enumerate(self.bids):
                    with torch.no_grad():   # the step's pay, as the economy settles it (wealth scale 1)
                        pay = mb(soft_score(bs[t + 1], y) - soft_score(b0, y))
                        if cov:
                            eq, meq = pair_truth(y, mask, c0.shape[1])
                            pay = pay + a.lam_c * ((soft_score(self.cs[t], eq) - soft_score(c0, eq)) * meq).sum((1, 2)) / nb
                        if vsup and full_steps:
                            pay = pay + a.lam_v * ((soft_score(self.vs[t], vts[t]) - soft_score(v0, vts[t]))
                                                   * mask.unsqueeze(1)).sum((1, 2)) / nb
                    gl.append(F.smooth_l1_loss(bid, pay.detach()))
                L_ = L_ + (torch.stack(gl) * w).mean()
            return L_

        al = getattr(a, "alpha", 1.0) if getattr(a, "prog", 0) else 0.0
        loss = torch.zeros((), device=self.dev)
        rec = dict(rec_c=cov, rec_v=vsup, rec_bid=bsup)
        if al > 0:   # progressive part (Bansal et al. 2022): n no-grad steps, k with grad from the detached state
            n = int(torch.randint(0, D, (1,), generator=gen))
            k = int(torch.randint(1, D - n + 1, (1,), generator=gen))
            self.rollout(x, mask, n, grad=False, gen=gen)
            init = tuple(z.detach() for z in self.last)
            _, bs, _ = self.rollout(x, mask, k, grad=True, init=init, t0=n, gen=gen, **rec)
            loss = loss + al * part(bs, False)
        if al < 1:   # the loss at depth D, one graph from step 0
            _, bs, _ = self.rollout(x, mask, D, grad=True, gen=gen, **rec)
            loss = loss + (1 - al) * part(bs, True)
        final = -mb(soft_score(bs[-1], y)).mean()
        self.opt.zero_grad() if not pt else [o.zero_grad() for o in self.opt.opts]
        loss.backward()
        if pt:
            for tr, o in zip(self.traders, self.opt.opts):
                if any(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in tr.parameters()):
                    torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); o.step()
        else:
            torch.nn.utils.clip_grad_norm_(self.traders.parameters(), 1.0)
            self.opt.step()
        return {"final_nll_bit": final.item(), "loss": loss.item()}


@torch.no_grad()
def eval_depths(M, L, depths, n, device, pick="roundrobin", chunk=100, seed=555):
    gen = torch.Generator().manual_seed(seed + L)
    x, y, m = prefix_batch(n, L, L, gen, device)
    Dmax = max(depths)
    acc = {D: [] for D in depths}; nll = {D: [] for D in depths}; bit = {D: [] for D in depths}
    for c0 in range(0, n, chunk):
        xs, ys, ms = x[c0:c0 + chunk], y[c0:c0 + chunk], m[c0:c0 + chunk]
        _, bs, _ = M.rollout(xs, ms, Dmax, pick=pick, keep=False, record_at=set(depths))
        for D in depths:
            b = bs[D]
            acc[D].append((((b > 0).float() == ys) | (ms == 0)).all(1).float().cpu())
            nll[D].append((-(belief_score(b, ys, getattr(getattr(M, "a", None), "floor", 0) or 0) * ms).sum(1) / ms.sum(1)).cpu())
            bit[D].append(((((b > 0).float() == ys).float() * ms).sum(1) / ms.sum(1)).cpu())
        del bs
    return [{"L": L, "D": D, "pick": pick, "n": n, "acc": torch.cat(acc[D]).mean().item(),
             "bit_acc": torch.cat(bit[D]).mean().item(),
             "nll_bit": torch.cat(nll[D]).mean().item()} for D in depths]


@torch.no_grad()
def eval_halting(M, L, n, device, D, costs, kind, chunk=100):
    """Priced halting on mixed lengths 4..L, applied post hoc to full D-step
    trajectories (halting only freezes an example's state). kind: bid (auction,
    halt when the best bid <= cost) or voc (round robin, halt when the VOC
    market's expected nats/bit per step <= cost). Also: E = random halting with
    the same step histogram, post-halt gain, and signal calibration."""
    gen = torch.Generator().manual_seed(909 + L)
    x, y, m = prefix_batch(n, 4, L, gen, device)
    EX, NL, SG = [], [], []
    for c0 in range(0, n, chunk):
        xs, ys, ms = x[c0:c0 + chunk], y[c0:c0 + chunk], m[c0:c0 + chunk]
        _, bs, _ = M.rollout(xs, ms, D, pick="auction" if kind == "bid" else "roundrobin",
                             keep=False, signals=True)
        EX.append(torch.stack([(((b > 0).float() == ys) | (ms == 0)).all(1) for b in bs], 1).float().cpu())
        NL.append(torch.stack([-(soft_score(b, ys) * ms).sum(1) / ms.sum(1) for b in bs], 1).cpu())
        SG.append(torch.stack(M.sig, 1)[:, :, 0 if kind == "bid" else 1].cpu())
        del bs
    EX, NL, SG = torch.cat(EX), torch.cat(NL), torch.cat(SG)       # (n,D+1) (n,D+1) (n,D)
    ar = torch.arange(n)
    rng = torch.Generator().manual_seed(1)
    out = []
    for cst in costs:
        stop = SG <= cst
        stop[:, 0] = False
        stop = torch.cat([stop, torch.ones(n, 1, dtype=torch.bool)], 1)
        tau = stop.float().argmax(1)
        perm = tau[torch.randperm(n, generator=rng)]
        fx = int(round(tau.float().mean().item()))
        out.append({"kind": kind, "cost": cst, "L": L, "steps": tau.float().mean().item(),
                    "acc": EX[ar, tau].mean().item(), "nll_bit": NL[ar, tau].mean().item(),
                    "post_halt_gain": (NL[ar, tau] - NL[:, -1]).mean().item(),
                    "E_perm_acc": EX[ar, perm].mean().item(), "fixed_acc_same_mean": EX[:, fx].mean().item()})
    k = 1 if kind == "bid" else KS[0]
    real = (NL[:, :D - k + 1] - NL[:, k:]) / k                      # realized nats/bit per step
    pred = SG[:, :D - k + 1]
    cal = {"kind": kind, "calib_corr": torch.corrcoef(torch.stack([pred.flatten(), real.flatten()]))[0, 1].item(),
           "calib_mae": (pred - real).abs().mean().item(), "pred_mean": pred.mean().item(), "real_mean": real.mean().item()}
    return out, cal


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="econ")        # econ | bptt
    p.add_argument("--stream", default="cov")       # cov | slots | none
    p.add_argument("--voc", type=int, default=0)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--D", type=int, default=24)
    p.add_argument("--lmin", type=int, default=8)
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cost", type=float, default=0.0)
    p.add_argument("--W0", type=float, default=20.0)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--wealth_cap", type=int, default=1)
    p.add_argument("--lam_c", type=float, default=1.0)
    p.add_argument("--lam_v", type=float, default=0.3)
    p.add_argument("--rent", type=float, default=1e-3)
    p.add_argument("--self_pay", type=int, default=0)
    p.add_argument("--grow_every", type=int, default=0)
    p.add_argument("--train_halt", type=float, default=-1)   # VOC halting cost during training (<0: off)
    p.add_argument("--settle", default="eps")                # real (self-fulfilling bug) | eps (fixed)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--bounty", default="")                     # phase 2b: "" | vol | lindahl (stream cov)
    p.add_argument("--fund_lr", type=float, default=0.05)
    p.add_argument("--online", type=int, default=0)          # streaming local updates, O(1) memory in depth
    p.add_argument("--prog", type=int, default=0)            # bptt: Bansal et al. 2022 progressive loss
    p.add_argument("--ds", type=int, default=0)              # bptt: loss on every step's belief (deep supervision)
    p.add_argument("--covsup", type=int, default=0)          # bptt: also supervise c on the covariance targets, every step
    p.add_argument("--tbptt", type=int, default=0)           # bptt: detach the state every K steps (0: never)
    p.add_argument("--train_pick", default="")               # "" = mode default (econ random, bptt round robin) | random | roundrobin
    p.add_argument("--vocsup", type=int, default=0)          # bptt: v trained on the economy's VOC targets, every step
    p.add_argument("--ptopt", type=int, default=0)           # bptt: per-trader Adam + clip + per-trader loss mean
    p.add_argument("--bidsup", type=int, default=0)          # bptt: bid head regressed on each step's detached pay
    p.add_argument("--alpha", type=float, default=1.0)       # bptt --prog: (1-alpha) loss at depth D + alpha progressive
    p.add_argument("--gradckpt", type=int, default=0)        # bptt: gradient checkpointing per step (memory runs)
    p.add_argument("--clamp", type=float, default=0)          # test only: clamp b, c, v to +-clamp after every step
    p.add_argument("--floor", type=float, default=0)          # econ: fixed-share floor eps on the belief forecast
    p.add_argument("--l2b", type=float, default=0)            # econ: fee lam * (b_after^2 - b_before^2) per bit on the mover
    p.add_argument("--bcap", type=float, default=0)           # econ: clamp the belief logit to +-bcap after every trade
    p.add_argument("--dbcap", type=float, default=0)          # econ: per-step trade cap db -> d tanh(db/d)
    p.add_argument("--liq", default="")                       # econ: belief liquidity grows with depth: "" | t | sqrt
    p.add_argument("--ckpt_every", type=float, default=600)   # seconds between training checkpoints (<out>.ckpt.pt)
    p.add_argument("--resume", type=int, default=1)           # resume training from <out>.ckpt.pt if it exists
    p.add_argument("--load", default="")                      # eval only: <out>.pt (+ <out>.json for log/wealth)
    p.add_argument("--auctionL", default="32,320")            # lengths where the (4x costlier) auction is evaluated
    p.add_argument("--evalL", default="32,320,3200")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    assert not (a.online and (a.floor or a.l2b or a.bcap or a.dbcap or a.liq)), "regularizers: train_step only"
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(3000 + a.seed)
    M = Economy(a, a.device) if a.mode == "econ" else BPTT(a, a.device)
    log, t0 = [], time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    iters = a.iters
    import os

    def save_atomic(obj, path):
        torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

    def weights():
        return [t[0].state_dict() for t in M.traders] if a.mode == "econ" else M.traders.state_dict()

    def load_weights(ck):
        if a.mode == "econ":
            while len(M.traders) < len(ck["traders"]):
                M.add_trader()
            for t, sd in zip(M.traders, ck["traders"]):
                t[0].load_state_dict(sd)
        else:
            M.traders.load_state_dict(ck["traders"])

    old, done_ev = {}, set()
    if a.load:  # eval only, e.g. after a spot preemption lost the evals
        ck = torch.load(a.load + ".pt", map_location=a.device, weights_only=False)
        old = json.load(open(a.load + ".json")) if os.path.exists(a.load + ".json") else \
            {"log": ck.get("log") or [{}], "train_sec": ck.get("train_sec"), "train_peak_mem_gb": ck.get("train_peak_mem_gb"),
             "resumed": ck.get("resumed")}
        log = old["log"]
        load_weights(ck)
        if a.mode == "econ":
            M.wealth = list(ck.get("wealth") or log[-1]["wealth"])
            M.stats = ck.get("stats", M.stats)
        iters = 0
        if os.path.abspath(a.load) == os.path.abspath(a.out):   # resumed evals: keep the (L, pick) blocks already done
            done_ev = {(e["L"], e.get("pick")) for e in old.get("eval", [])}
    # training checkpoints every --ckpt_every seconds (spot preemption); resume from the last one
    ckp, start, sec0 = a.out + ".ckpt.pt", 0, 0.0
    if iters and a.resume and os.path.exists(ckp):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)   # RNG states must stay on the CPU
        load_weights(ck)
        if a.mode == "econ":
            for t, sd in zip(M.traders, ck["opts"]):
                t[1].load_state_dict(sd)
            M.wealth, M.stats, M.fund = ck["wealth"], ck["stats"], ck["fund"]
        else:
            M.opt.load_state_dict(ck["opts"])
        log, start, sec0 = ck["log"], ck["it"] + 1, ck["sec"]
        gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        if a.device.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(json.dumps({"resumed_at_it": start, "sec": round(sec0, 1)}), flush=True)
    t0, last_ck = time.time() - sec0, time.time()
    for it in range(start, iters):
        x, y, m = prefix_batch(a.bs, a.lmin, a.lmax, gen, a.device)
        M.it = it
        lg = (M.train_step_online if a.online else M.train_step)(x, y, m, a.D, gen)
        if a.mode == "econ" and a.grow_every and it and it % a.grow_every == 0:
            M.add_trader()
        if it % 100 == 0 or it == a.iters - 1:
            lg.update(it=it, sec=round(time.time() - t0, 1))
            if a.mode == "econ":
                lg.update(M.stats)
            log.append(lg); print(json.dumps(lg), flush=True)
        if a.ckpt_every > 0 and time.time() - last_ck > a.ckpt_every and it < iters - 1:
            save_atomic({"traders": weights(), "it": it, "sec": time.time() - t0, "log": log,
                         "opts": [t[1].state_dict() for t in M.traders] if a.mode == "econ" else M.opt.state_dict(),
                         "wealth": getattr(M, "wealth", None), "stats": getattr(M, "stats", None),
                         "fund": getattr(M, "fund", None), "gen": gen.get_state(), "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None}, ckp)
            last_ck = time.time()
    res = {"args": vars(a), "log": log, "train_sec": time.time() - t0}
    if a.load:
        res["train_sec"] = old.get("train_sec"); res["train_peak_mem_gb"] = old.get("train_peak_mem_gb")
        if old.get("resumed"):
            res["resumed"] = old["resumed"]
    elif a.device.startswith("cuda"):
        res["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9   # since the last (re)start
    if start:
        res["resumed"] = start
    if not a.load:
        save_atomic({"traders": weights(), "args": vars(a), "wealth": getattr(M, "wealth", None),
                     "stats": getattr(M, "stats", None), "log": log, "train_sec": res["train_sec"],
                     "train_peak_mem_gb": res.get("train_peak_mem_gb"), "resumed": res.get("resumed")}, a.out + ".pt")
    ev = [e for e in old.get("eval", []) if (e["L"], e.get("pick")) in done_ev]
    grid = {32: ([2, 4, 6, 8, 12, 16, 24, 32, 48, 64], 1000), 320: ([6, 8, 12, 16, 24, 32, 64, 160, 320, 640], 200),
            3200: ([8, 12, 16, 24, 32, 64, 320, 1600, 3200], 40), 32000: ([12, 16, 20, 24, 32, 64, 320, 3200], 8)}
    for L in [int(z) for z in a.evalL.split(",")]:
        depths, n = grid[L]
        picks = ["roundrobin"] + (["auction"] if str(L) in a.auctionL.split(",") else [])
        for pick in (picks if a.mode == "econ" else ["-"]):
            if (L, pick) in done_ev:
                continue
            ev += eval_depths(M, L, depths, n, a.device, pick=pick, chunk=min(n, 100))
            print(json.dumps(ev[-len(depths):]), flush=True)
            res["eval"] = ev
            json.dump(res, open(a.out + ".json.tmp", "w")); os.replace(a.out + ".json.tmp", a.out + ".json")
    res["eval"] = ev
    if a.mode == "econ":
        if old.get("halting") and done_ev:
            res["halting"], res["halting_calib"] = old["halting"], old["halting_calib"]
        else:
            costs = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]
            res["halting"], cal = eval_halting(M, 64, 500, a.device, 96, costs, "bid")
            res["halting_calib"] = [cal]
            if a.voc:
                h2, cal = eval_halting(M, 64, 500, a.device, 96, costs, "voc")
                res["halting"] += h2; res["halting_calib"].append(cal)
            print(json.dumps(res["halting_calib"]), flush=True)
    json.dump(res, open(a.out + ".json.tmp", "w")); os.replace(a.out + ".json.tmp", a.out + ".json")
