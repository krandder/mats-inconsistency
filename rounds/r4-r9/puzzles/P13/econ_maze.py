"""P13: the phase-2 market ResNet economy (market-resnet econ2.py, snapshot econ2_1d_snapshot.py) ported to
Bansal/Schwarzschild mazes, with belief-logit runaway fixes.

State per maze (pixels, H = W = 2n + 6):
  b  (B,H,W)        belief markets: one binary LMSR per pixel ("is this pixel on the path?"), a logit
  c  (B,J,2,H,W)    covariance markets: level j, axis a (0 vertical, 1 horizontal): "y(p) == y(p + 2^j e_a)?"
  v  (B,3,H,W)      VOC markets: fraction of pixel log loss removed between t and t + k, k in KS = 5/15/30
Belief markets, pay and every loss are masked to open pixels.

Trader2D (N of them, own weights): an RGB embedding recalled every step; a dilation-1 3x3 read of
[emb, tanh(b/4), tanh(v/4)] (flood fill needs b at dilation 1); per level j a 3x3 read at dilation 2^j of
[emb, tanh(b/4), c_j] plus a 3x3 read at dilation 2^(j-1) of c_(j-1) (the Hillis-Steele composition
eq_j(p) = XNOR(eq_(j-1)(p), eq_(j-1)(p + 2^(j-1))) needs c_(j-1) at offsets 0 and 2^(j-1)); an MLP shared across
levels writes db_j (summed over j), dc_j (2), dv (3, mean over j) and a per-pixel gain bid (on detached features).

Modes (one rollout function, shared):
  econ  traders read DETACHED state and are paid the LMSR log-score change of every market they move
        (belief + lam_c * cov + lam_v * VOC), wealth-scaled trades (wealth_cap), W0, respawns, ledger;
        --wealth_cap 0 is detached per-step local-loss training (unit test (b)).
  bptt  same traders/streams, end to end through depth: --ds (every step's belief), --covsup, --vocsup,
        --bidsup, --ptopt, --prog/--alpha (Bansal progressive loss), --tbptt, --gradckpt.

Fixes (identical semantics in both paths): --bcap c, --dbcap d, --liq sqrt|t, --l2b lam, --floor eps, --insq.
Test-time J options: --jopt none | scale | cap (see eval_maze.py).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LS = F.logsigmoid
KS = [5, 15, 30]
J_TRAIN = 5


# ------------------------------------------------------------------ helpers
def soft_score(z, p):
    return p * LS(z) + (1 - p) * LS(-z)


def belief_score(z, p, floor=0.0):
    """log score of the belief market; with floor eps the forecast is (1-eps) sigmoid(z) + eps/2"""
    if not floor:
        return soft_score(z, p)
    l1, l0 = math.log(1 - floor), torch.tensor(math.log(floor / 2), device=z.device, dtype=z.dtype)
    return p * torch.logaddexp(LS(z) + l1, l0) + (1 - p) * torch.logaddexp(LS(-z) + l1, l0)


def liq_beta(kind, t):
    """liquidity of the belief market at step t (1-based)"""
    return 1.0 if not kind else (float(t) if kind == "t" else math.sqrt(t))


def shift2d(t, dy, dx):
    """out[..., r, c] = t[..., r + dy, c + dx] (zero outside)."""
    H, W = t.shape[-2:]
    if abs(dy) >= H or abs(dx) >= W:
        return torch.zeros_like(t)
    out = torch.zeros_like(t)
    rs, rd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    cs, cd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[..., rd, cd] = t[..., rs, cs]
    return out


def nlevels2d(side):
    return max(1, math.ceil(math.log2(max(side, 2))))


def pair_truth2d(y, mask, J):
    """eq (B,J,2,H,W): y(p) == y(p + 2^j e_a) for a in (vertical, horizontal); m: both pixels open."""
    eq, m = [], []
    for j in range(J):
        d = 2 ** j
        eq.append(torch.stack([(y == shift2d(y, d, 0)).float(), (y == shift2d(y, 0, d)).float()], 1))
        m.append(torch.stack([mask * shift2d(mask, d, 0), mask * shift2d(mask, 0, d)], 1))
    return torch.stack(eq, 1), torch.stack(m, 1)


def mb(z, mask):
    """(B,) mean over open pixels"""
    return (z * mask).sum((1, 2)) / mask.sum((1, 2))


def worst_loss_per_pix(b, db, mask):
    lo1 = LS(b) - LS(b + db)
    lo0 = LS(-b) - LS(-(b + db))
    return mb(torch.maximum(lo1, lo0).clamp(min=0), mask)


# ------------------------------------------------------------------ trader
class Trader2D(nn.Module):
    def __init__(self, w=224, we=32, insq=0):
        super().__init__()
        self.insq, self.w, self.we = insq, w, we
        nb = 2 if insq else 1
        self.emb = nn.Conv2d(3, we, 3, padding=1)
        self.base = nn.Conv2d(we + nb + 3, w, 3, padding=1)          # [emb, b feats, tanh(v/4)] at dilation 1
        self.lev = nn.Conv2d(we + nb + 2, w, 3, bias=False)          # [emb, b feats, c_j] at dilation 2^j
        self.prev = nn.Conv2d(2, w, 3, bias=False)                   # c_(j-1) at dilation 2^(j-1)
        self.mlp = nn.Conv2d(w, w, 1)
        self.out = nn.Conv2d(w, 6, 1)                                # db_j, dc_j (2), dv (3)
        self.gain = nn.Sequential(nn.Conv2d(w, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def bfeat(self, b):
        tb = torch.tanh(b / 4).unsqueeze(1)
        if self.insq:
            return torch.cat([tb, (torch.sign(b) * torch.log1p(b.abs())).unsqueeze(1)], 1)
        return tb

    def forward(self, x, b, c, v, mask, jscale=1.0):
        B, J = c.shape[:2]
        e = self.emb(x)
        fb = self.bfeat(b)
        tc = torch.tanh(c / 4)
        base = self.base(torch.cat([e, fb, torch.tanh(v / 4)], 1))
        eb = torch.cat([e, fb], 1)
        m = mask.unsqueeze(1)
        db, dv, g, dcs = 0.0, 0.0, 0.0, []
        for j in range(J):          # levels one at a time: the same math as a level-batched MLP, less memory
            d = 2 ** j
            h = F.conv2d(torch.cat([eb, tc[:, j]], 1), self.lev.weight, None, padding=d, dilation=d) + base
            if j > 0:
                dp = 2 ** (j - 1)
                h = h + F.conv2d(tc[:, j - 1], self.prev.weight, None, padding=dp, dilation=dp)
            z = F.relu(self.mlp(F.relu(h)))
            o = self.out(z)
            db = db + o[:, 0]
            dcs.append(o[:, 1:3])
            dv = dv + o[:, 3:]
            g = g + self.gain(z.detach())[:, 0]
        db = db * mask * jscale
        dc = torch.stack(dcs, 1) * m.unsqueeze(1)
        dv = dv / J * m
        bid = (g * mask).sum((1, 2)) / mask.sum((1, 2))
        return db, dc, dv, bid

    def macs_per_pix(self, J):
        """multiply-accumulates per pixel per step (convs and 1x1s; elementwise ops ignored)"""
        w, we = self.w, self.we
        nb = 2 if self.insq else 1
        m = 3 * we * 9 + (we + nb + 3) * w * 9
        for j in range(J):
            m += (we + nb + 2) * w * 9 + (2 * w * 9 if j else 0) + w * w + 6 * w + w * 32 + 32
        return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class MultiOpt:
    def __init__(self, mods, lr):
        self.opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in mods]

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, d in zip(self.opts, sd):
            o.load_state_dict(d)


# ------------------------------------------------------------------ model (econ and bptt share the rollout)
class Model:
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.traders = nn.ModuleList([Trader2D(a.width, insq=a.insq) for _ in range(a.N)]).to(device)
        if a.mode == "econ" or a.ptopt:
            self.opt = MultiOpt(self.traders, a.lr)
        else:
            self.opt = torch.optim.Adam(self.traders.parameters(), lr=a.lr)
        self.wealth = [a.W0] * a.N
        self.stats = {"respawns": 0, "entries": 0}
        self.diag = self.new_diag()

    @staticmethod
    def new_diag():
        return {"trade_pix": 0.0, "capped_pix": 0.0, "min_wealth": float("inf"), "s_sum": 0.0, "n": 0}

    def respawn(self, i):
        a = self.a
        t = Trader2D(a.width, insq=a.insq).to(self.dev)
        self.traders[i] = t
        self.opt.opts[i] = torch.optim.Adam(t.parameters(), lr=a.lr)
        self.wealth[i] = a.W0
        self.stats["respawns"] += 1

    # -- fix semantics (identical in both paths)
    def relax(self, b, T1):
        """liquidity grows from beta_T1 to beta_(T1+1): the opening logit relaxes (T1 = global 0-based step)"""
        if self.a.liq and T1 > 0:
            return b * (liq_beta(self.a.liq, T1) / liq_beta(self.a.liq, T1 + 1))
        return b

    def tdb(self, db, beta):
        """trade transform: dbcap, then liquidity scaling. beta: float or (B,1,1) tensor"""
        if self.a.dbcap:
            db = self.a.dbcap * torch.tanh(db / self.a.dbcap)
        if self.a.liq:
            db = db / beta
        return db

    def bclamp(self, b):
        return b.clamp(-self.a.bcap, self.a.bcap) if self.a.bcap else b

    # -- the rollout
    def rollout(self, x, mask, D, pick="roundrobin", gen=None, grad=False, J=None, jscale=1.0, init=None, t0=0,
                keep=False, hook=None, rec=(), wealth_scale=True):
        """D steps from init (zeros) at global step t0. pick: random | roundrobin. keep: store each step's
        pre-trade state (econ settlement). hook(t, b, c, v): called after every step (then bs is not kept).
        rec: subset of {'c', 'v', 'bid'} kept per step (bptt supervision). Returns (recs, bs)."""
        a = self.a
        B = x.shape[0]
        H, W = mask.shape[-2:]
        J = J or nlevels2d(max(H, W))
        N = len(self.traders)
        if init is None:
            b = torch.zeros(B, H, W, device=self.dev)
            c = torch.zeros(B, J, 2, H, W, device=self.dev)
            v = torch.zeros(B, 3, H, W, device=self.dev)
        else:
            b, c, v = init
        recs, bs = [], ([b] if hook is None else None)
        self.who, self.cs, self.vs, self.bids, self.bprev = [], [], [], [], []
        tb = a.tbptt if grad else 0
        gck = grad and a.gradckpt
        econ_cap = a.mode == "econ" and a.wealth_cap and wealth_scale
        with torch.set_grad_enabled(grad):
            for t in range(D):
                T1 = t0 + t
                if tb and t and t % tb == 0:
                    b, c, v = b.detach(), c.detach(), v.detach()
                b = self.relax(b, T1)
                i = int(torch.randint(N, (1,), generator=gen)) if pick == "random" else T1 % N
                tr = self.traders[i]
                if gck:
                    db, dc, dv, bid = torch.utils.checkpoint.checkpoint(tr, x, b, c, v, mask, jscale, use_reentrant=False)
                else:
                    db, dc, dv, bid = tr(x, b, c, v, mask, jscale)
                db = self.tdb(db, liq_beta(a.liq, T1 + 1))
                s = None
                if econ_cap:
                    with torch.no_grad():
                        wl = worst_loss_per_pix(b, db, mask)
                        s = torch.clamp(a.frac * max(self.wealth[i], 0.0) / (wl + 1e-8), max=1.0)
                if keep:
                    recs.append({"b": b, "c": c, "v": v, "who": i, "s": s, "T1": T1})
                self.who.append(i)
                if "bid" in rec:
                    self.bids.append(bid)
                self.bprev.append(b)
                if s is None:
                    b = b + db
                    v = v + dv
                    c = c + dc
                else:
                    b = b + s.view(B, 1, 1) * db
                    v = v + s.view(B, 1, 1, 1) * dv
                    c = c + s.view(B, 1, 1, 1, 1) * dc
                b = self.bclamp(b)
                if "c" in rec:
                    self.cs.append(c)
                if "v" in rec:
                    self.vs.append(v)
                if hook is None:
                    bs.append(b)
                else:
                    hook(t + 1, b, c, v)
        self.last = (b, c, v)
        return recs, bs

    # -- economy settlement
    def voc_targets(self, LL, t, D):
        return torch.stack([((LL[t] - LL[min(t + k, D)]) / (LL[t] + 1e-6)).clamp(0, 1) for k in KS], 1)

    def train_step_econ(self, x, y, mask, D, gen):
        a = self.a
        B = x.shape[0]
        J = nlevels2d(mask.shape[-1])
        N = len(self.traders)
        recs, bs = self.rollout(x, mask, D, pick=a.train_pick, gen=gen, keep=True, J=J)
        who = [r["who"] for r in recs]
        nb = mask.sum((1, 2))
        eq, meq = pair_truth2d(y, mask, J)
        fl = a.floor
        with torch.no_grad():
            LL = torch.stack([-belief_score(bb, y, fl) for bb in bs], 0)       # (D+1,B,H,W)
        profit = torch.zeros(N, device=self.dev)
        covpay_tot, ig_credit = 0.0, torch.zeros(N)
        rep = lambda z, n: z.repeat(n, *([1] * (z.dim() - 1)))
        with torch.no_grad():   # economy diagnostics: trade-pixels with wealth scale s < 1
            if a.wealth_cap:
                S = torch.stack([r["s"] for r in recs])                       # (D,B)
                self.diag["trade_pix"] += (nb.view(1, B) * torch.ones_like(S)).sum().item()
                self.diag["capped_pix"] += (nb.view(1, B) * (S < 1).float()).sum().item()
                self.diag["s_sum"] += S.mean().item(); self.diag["n"] += 1
        ones = torch.ones(B, device=self.dev)
        for i in range(N):
            tr, opt = self.traders[i], self.opt.opts[i]
            ts_all = [t for t in range(D) if who[t] == i]
            if not ts_all:
                continue
            tot = len(ts_all) * B
            opt.zero_grad()
            prof_sum, cr = 0.0, 0.0
            ch = a.settle_chunk or len(ts_all)
            for c0 in range(0, len(ts_all), ch):   # chunks of steps, gradient accumulated: the same objective as one batch
                ts = ts_all[c0:c0 + ch]
                n = len(ts)
                bi = torch.cat([recs[t]["b"] for t in ts]); vi = torch.cat([recs[t]["v"] for t in ts])
                ci = torch.cat([recs[t]["c"] for t in ts])
                si = torch.cat([recs[t]["s"] if recs[t]["s"] is not None else ones for t in ts])
                X, Y, Mk, NB = rep(x, n), rep(y, n), rep(mask, n), rep(nb, n)
                beta = torch.cat([torch.full((B, 1, 1), liq_beta(a.liq, recs[t]["T1"] + 1), device=self.dev) for t in ts])
                db, dc, dv, bid = tr(X, bi, ci, vi, Mk)
                db = self.tdb(db, beta)
                bnew = self.bclamp(bi + si.view(-1, 1, 1) * db)
                if getattr(self, "debug", False):   # unit test (d): the settled state equals the rollout's
                    for k_, t in enumerate(ts):
                        sl = slice(k_ * B, (k_ + 1) * B)
                        self.dbg[t] = (bnew[sl].detach(), (ci + si.view(-1, 1, 1, 1, 1) * dc)[sl].detach(),
                                       (vi + si.view(-1, 1, 1, 1) * dv)[sl].detach())
                pay = ((belief_score(bnew, Y, fl) - belief_score(bi, Y, fl)) * Mk).sum((1, 2)) / NB
                if a.l2b:
                    pay = pay - a.l2b * ((bnew ** 2 - bi ** 2) * Mk).sum((1, 2)) / NB
                EQ, MEQ = rep(eq, n), rep(meq, n)
                cp = a.lam_c * ((soft_score(ci + si.view(-1, 1, 1, 1, 1) * dc, EQ) - soft_score(ci, EQ)) * MEQ).sum((1, 2, 3, 4)) / NB
                pay = pay + cp
                covpay_tot += cp.detach().sum().item() / B
                if a.bounty == "lindahl":   # Aumann-Shapley credit of c in this trader's b-earnings (4-point path)
                    for al in [0.125, 0.375, 0.625, 0.875]:
                        ca = (ci.detach() * al).requires_grad_(True)
                        dba = self.tdb(tr(X, bi, ca, vi, Mk)[0], beta)
                        ba = self.bclamp(bi + si.view(-1, 1, 1) * dba)
                        pa = (((belief_score(ba, Y, fl) - belief_score(bi, Y, fl)) * Mk).sum((1, 2)) / NB).sum()
                        cr = cr + (torch.autograd.grad(pa, ca)[0] * ci.detach()).sum().item() / 4
                vt = torch.cat([self.voc_targets(LL, t, D) for t in ts])
                pay = pay + a.lam_v * ((soft_score(vi + si.view(-1, 1, 1, 1) * dv, vt) - soft_score(vi, vt))
                                       * Mk.unsqueeze(1)).sum((1, 2, 3)) / NB
                prof = pay - a.cost
                gl = F.smooth_l1_loss(bid, (prof + a.cost).detach(), reduction="sum")
                obj = (-pay.sum() + gl) / tot
                obj.backward()
                prof_sum += prof.detach().sum().item()
            torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); opt.step()
            profit[i] = prof_sum / B
            ig_credit[i] = cr / B
        charge = torch.zeros(N, device=self.dev)
        if a.bounty == "lindahl":   # trader-funded cov markets: the cov payout charged by credit share
            w = ig_credit.clamp(min=0)
            w = w / w.sum() if w.sum() > 0 else torch.full((N,), 1.0 / N)
            charge = covpay_tot * w.to(self.dev)
        profit = profit - charge
        for i in range(N):
            self.wealth[i] += profit[i].item()
        self.diag["min_wealth"] = min(self.diag["min_wealth"], min(self.wealth))
        for i in range(N):
            if a.wealth_cap and self.wealth[i] <= 0:
                self.respawn(i)
        Sf = mb(belief_score(bs[-1], y, fl), mask).mean().item()
        return {"final_nll": -Sf, "profit": [round(p, 4) for p in profit.tolist()],
                "wealth": [round(w, 2) for w in self.wealth], "cov_payout": round(covpay_tot, 4),
                "maxb": bs[-1].abs().max().item()}

    # -- backprop through depth
    def part(self, bs, y, mask, full_steps):
        a = self.a
        fl = a.floor
        T = len(bs) - 1
        w = torch.ones(T, device=self.dev)
        if a.ptopt:
            cnt = torch.bincount(torch.tensor(self.who), minlength=len(self.traders)).float()
            w = (T / cnt[torch.tensor(self.who)]).to(self.dev)
        if a.ds:
            L_ = -(torch.stack([mb(belief_score(bt, y, fl), mask).mean() for bt in bs[1:]]) * w).mean()
        else:
            L_ = -mb(belief_score(bs[-1], y, fl), mask).mean()
        if a.l2b:
            L_ = L_ + a.l2b * (torch.stack([mb(bs[t + 1] ** 2 - self.bprev[t].detach() ** 2, mask).mean()
                                            for t in range(T)]) * w).mean()
        nb = mask.sum((1, 2))
        if a.covsup and self.cs:
            eq, meq = pair_truth2d(y, mask, self.cs[0].shape[1])
            L_ = L_ - a.lam_c * (torch.stack([((soft_score(ct, eq) * meq).sum((1, 2, 3, 4)) / nb).mean()
                                              for ct in self.cs]) * w).mean()
        vts = None
        if a.vocsup and self.vs and full_steps:
            with torch.no_grad():
                LL = torch.stack([-belief_score(bt.detach(), y, fl) for bt in bs], 0)
                vts = [self.voc_targets(LL, t, T) for t in range(T)]
            L_ = L_ - a.lam_v * (torch.stack([((soft_score(vt_, tg) * mask.unsqueeze(1)).sum((1, 2, 3)) / nb).mean()
                                              for vt_, tg in zip(self.vs, vts)]) * w).mean()
        if a.bidsup and self.bids:
            gl = []
            for t, bid in enumerate(self.bids):
                with torch.no_grad():   # the step's pay as the economy settles it (wealth scale 1)
                    b0 = self.bprev[t]
                    pay = mb(belief_score(bs[t + 1], y, fl) - belief_score(b0, y, fl), mask)
                    if a.l2b:
                        pay = pay - a.l2b * mb(bs[t + 1] ** 2 - b0 ** 2, mask)
                    if a.covsup:
                        c0 = self.cs[t - 1] if t > 0 else torch.zeros_like(self.cs[0])
                        pay = pay + a.lam_c * ((soft_score(self.cs[t], eq) - soft_score(c0, eq)) * meq).sum((1, 2, 3, 4)) / nb
                    if vts is not None:
                        v0 = self.vs[t - 1] if t > 0 else torch.zeros_like(self.vs[0])
                        pay = pay + a.lam_v * ((soft_score(self.vs[t], vts[t]) - soft_score(v0, vts[t]))
                                               * mask.unsqueeze(1)).sum((1, 2, 3)) / nb
                gl.append(F.smooth_l1_loss(bid, pay.detach()))
            L_ = L_ + (torch.stack(gl) * w).mean()
        return L_

    def train_step_bptt(self, x, y, mask, D, gen):
        a = self.a
        pt = bool(a.ptopt)
        rec = tuple(k for k, f in (("c", a.covsup), ("v", a.vocsup), ("bid", a.bidsup)) if f)
        al = a.alpha if a.prog else 0.0
        loss = torch.zeros((), device=self.dev)
        pick = a.train_pick
        if al > 0:   # progressive part (Bansal et al. 2022): n no-grad steps, k with grad from the detached state
            n = int(torch.randint(0, D, (1,), generator=gen))
            k = int(torch.randint(1, D - n + 1, (1,), generator=gen))
            self.rollout(x, mask, n, pick=pick, gen=gen, grad=False)
            init = tuple(z.detach() for z in self.last)
            _, bs = self.rollout(x, mask, k, pick=pick, gen=gen, grad=True, init=init, t0=n, rec=rec)
            loss = loss + al * self.part(bs, y, mask, False)
        if al < 1:
            _, bs = self.rollout(x, mask, D, pick=pick, gen=gen, grad=True, rec=rec)
            loss = loss + (1 - al) * self.part(bs, y, mask, True)
        final = -mb(belief_score(bs[-1], y, a.floor), mask).mean()
        opts = self.opt.opts if pt else [self.opt]
        for o in opts:
            o.zero_grad()
        loss.backward()
        if pt:
            for tr, o in zip(self.traders, self.opt.opts):
                if any(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in tr.parameters()):
                    torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0); o.step()
        else:
            torch.nn.utils.clip_grad_norm_(self.traders.parameters(), 1.0)
            self.opt.step()
        return {"final_nll": final.item(), "loss": loss.item(), "maxb": bs[-1].abs().max().item()}

    def train_step(self, x, y, mask, D, gen):
        return (self.train_step_econ if self.a.mode == "econ" else self.train_step_bptt)(x, y, mask, D, gen)


# ------------------------------------------------------------------ data
def load_train(frozen, device):
    from check import expand
    z = np.load(os.path.join(frozen, "maze_train9.npz"))
    code = torch.as_tensor(z["train9/code"])
    sol = torch.as_tensor(np.unpackbits(z["train9/sol"], axis=-1, count=9))
    X, Y = expand(code, sol)
    return X.to(device), Y.to(device), torch.as_tensor(z["train9/train_idx"]).long(), torch.as_tensor(z["train9/val_idx"]).long()


class Sampler:
    """P0 train.py's maze sampler: a new permutation of the 40k train indices per epoch (dgen), batch 50, drop_last"""
    def __init__(self, tr_idx, bs, seed):
        self.tr_idx, self.bs = tr_idx, bs
        self.dgen = torch.Generator().manual_seed(seed)
        self.perm, self.pos = None, 0

    def next(self):
        if self.perm is None or self.pos + self.bs > len(self.perm):
            self.perm = self.tr_idx[torch.randperm(len(self.tr_idx), generator=self.dgen)]
            self.pos = 0
        bi = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return bi

    def state(self):
        return {"dgen": self.dgen.get_state(), "perm": self.perm, "pos": self.pos}

    def load(self, s):
        self.dgen.set_state(s["dgen"]); self.perm, self.pos = s["perm"], s["pos"]


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="econ")               # econ | bptt
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--D", type=int, default=30)
    p.add_argument("--iters", type=int, default=1000)       # step budget
    p.add_argument("--bs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cost", type=float, default=0.0)
    p.add_argument("--W0", type=float, default=20.0)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--wealth_cap", type=int, default=1)
    p.add_argument("--lam_c", type=float, default=1.0)
    p.add_argument("--lam_v", type=float, default=0.3)
    p.add_argument("--rent", type=float, default=1e-3)      # slot rent in econ2; no slots here (logged only)
    p.add_argument("--bounty", default="")                  # "" (house-funded cov) | lindahl
    p.add_argument("--train_pick", default="random")        # random | roundrobin
    p.add_argument("--prog", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ds", type=int, default=0)
    p.add_argument("--covsup", type=int, default=0)
    p.add_argument("--vocsup", type=int, default=0)
    p.add_argument("--bidsup", type=int, default=0)
    p.add_argument("--ptopt", type=int, default=0)
    p.add_argument("--tbptt", type=int, default=0)
    p.add_argument("--gradckpt", type=int, default=0)
    p.add_argument("--settle_chunk", type=int, default=8)   # econ settlement: steps per forward/backward chunk (memory)
    p.add_argument("--floor", type=float, default=0)
    p.add_argument("--l2b", type=float, default=0)
    p.add_argument("--bcap", type=float, default=0)
    p.add_argument("--dbcap", type=float, default=0)
    p.add_argument("--liq", default="")
    p.add_argument("--insq", type=int, default=0)
    p.add_argument("--ckpt_every", type=float, default=600)
    p.add_argument("--final_ckpt", type=int, default=0)     # keep a resumable checkpoint at the end (pilot extension to 1.0 h)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--diag_every", type=int, default=1000)
    p.add_argument("--time_steps", type=int, default=0)     # Phase 1: time this many steps after a warm-up and exit
    p.add_argument("--kill_at", type=int, default=0)        # unit test (e): checkpoint after this iteration and exit 3
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frozen", default=os.path.join(HERE, "frozen_p0"))
    p.add_argument("--out", required=True)
    p.add_argument("--tf32", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def checkpoint_state(M, it, sec, log, gen, sampler, a):
    return {"traders": [t.state_dict() for t in M.traders], "opts": M.opt.state_dict(), "it": it, "sec": sec,
            "log": log, "wealth": M.wealth, "stats": M.stats, "diag": M.diag, "gen": gen.get_state(),
            "rng": torch.get_rng_state(), "sampler": sampler.state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None}


def load_checkpoint(M, ck, gen, sampler, a):
    for t, sd in zip(M.traders, ck["traders"]):
        t.load_state_dict(sd)
    M.opt.load_state_dict(ck["opts"])
    M.wealth, M.stats, M.diag = ck["wealth"], ck["stats"], ck["diag"]
    gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"]); sampler.load(ck["sampler"])
    if a.device.startswith("cuda") and ck.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ck["cuda_rng"])


def main(argv=None):
    a = get_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.benchmark = a.device.startswith("cuda")
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(3000 + a.seed)
    M = Model(a, a.device)
    X, Y, tr_idx, va_idx = load_train(a.frozen, a.device)
    sampler = Sampler(tr_idx, a.bs, a.seed)
    info = {"params_per_trader": n_params(M.traders[0]), "macs_per_pix_J5": M.traders[0].macs_per_pix(J_TRAIN)}
    print(json.dumps({"arch": info, "args": vars(a)}), flush=True)
    ckp = a.out + ".ckpt.pt"
    log, start, sec0 = [], 0, 0.0
    if os.path.exists(ckp) and not a.time_steps:
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        load_checkpoint(M, ck, gen, sampler, a)
        log, start, sec0 = ck["log"], ck["it"] + 1, ck["sec"]
        print(json.dumps({"resumed_at_it": start, "sec": round(sec0, 1)}), flush=True)
    t0, last_ck = time.time() - sec0, time.time()
    tl = None
    for it in range(start, a.iters):
        if a.time_steps and it == start + 20:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            tl = time.time()
        bi = sampler.next().to(a.device)
        x, y = X[bi], Y[bi]
        mask = (x.max(1)[0] > 0).float()
        lg = M.train_step(x, y, mask, a.D, gen)
        if a.time_steps and tl is not None and it + 1 - (start + 20) >= a.time_steps:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            el = time.time() - tl
            print(json.dumps({"timing": os.path.basename(a.out), "steps": a.time_steps, "sec": round(el, 2),
                              "sec_per_1k": round(1000 * el / a.time_steps, 1),
                              "peak_mem_mb": torch.cuda.max_memory_allocated() // 2 ** 20 if a.device.startswith("cuda") else 0}),
                  flush=True)
            return
        if it % a.log_every == 0 or it == a.iters - 1:
            lg.update(it=it, sec=round(time.time() - t0, 1))
            if a.mode == "econ":
                lg.update(M.stats)
            if (it % a.diag_every == 0 or it == a.iters - 1) and a.mode == "econ":
                d = M.diag
                W = np.array(M.wealth)
                lg["diag"] = {"frac_capped_pix": d["capped_pix"] / max(d["trade_pix"], 1), "mean_s": d["s_sum"] / max(d["n"], 1),
                              "min_wealth": d["min_wealth"], "shares": (W / W.sum()).round(4).tolist() if W.sum() > 0 else None,
                              "respawns": M.stats["respawns"], "entries": M.stats["entries"], "iters": d["n"]}
                M.diag = M.new_diag()
            log.append(lg); print(json.dumps(lg), flush=True)
        if a.kill_at and it + 1 == a.kill_at:
            save_atomic(checkpoint_state(M, it, time.time() - t0, log, gen, sampler, a), ckp)
            print(json.dumps({"killed_after_it": it}), flush=True)
            sys.exit(3)
        if a.ckpt_every > 0 and time.time() - last_ck > a.ckpt_every and it < a.iters - 1:
            save_atomic(checkpoint_state(M, it, time.time() - t0, log, gen, sampler, a), ckp)
            last_ck = time.time()
    res = {"args": vars(a), "arch": info, "log": log, "train_sec": time.time() - t0,
           "wealth": M.wealth, "stats": M.stats}
    if a.device.startswith("cuda"):
        res["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    save_atomic({"traders": [t.state_dict() for t in M.traders], **res}, a.out + ".pt")
    json.dump(res, open(a.out + ".json.tmp", "w")); os.replace(a.out + ".json.tmp", a.out + ".json")
    if a.final_ckpt:
        save_atomic(checkpoint_state(M, a.iters - 1, res["train_sec"], log, gen, sampler, a), ckp)
    elif os.path.exists(ckp):
        os.remove(ckp)
    print(json.dumps({"done": a.out, "sec": round(res["train_sec"], 1)}), flush=True)


if __name__ == "__main__":
    main()
