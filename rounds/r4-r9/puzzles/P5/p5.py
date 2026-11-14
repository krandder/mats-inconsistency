"""P5: private memory slots priced online (maze path-parity, answered only at T).

Arms (see RESULTS.md for the full spec):
  A1/A1w DT-Recall (width 28 / 128), BPTT + Bansal progressive loss
  A2   4 traders + private banks, full BPTT on the final belief (b path kept) + progressive loss
  A2t  same modules, BPTT through the banks on the per-step telescoped log score (b detached), global, pointwise
  A3   belief-only traders (no banks), each trained on its own Aumann-Shapley pay, ledger
  A4   one-hop procurement: price = IG gradient of the owner's own pay at t+1 only, online, ledger
  A5   offline chained AS royalties (phi=1): no-grad rollout, reverse IG sweep, per-step own-profit updates, ledger
  A6   DNI twin: bootstrapped synthetic-gradient head per bank, TD to the pointwise global one-step target, no ledger
  A7   posted prices: owner head posts a TD-learned IG price per bank cell, writers paid at write time, own profit, ledger
  A8 = A5 pointwise, A10 = A5 no ledger, A7p = A7 pointwise, A7f = A7 no ledger (stage 2)
Money: pays are per example; wealth W0 = 1 nat per trader; caps s_i and royalty scaling as in the spec.
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

import mazes

CIN, K, HID = 7, 4, 16
BANK_ARMS = {"A2", "A2t", "A4", "A5", "A6", "A7", "A8", "A10", "A7p", "A7f"}
HEAD_ARMS = {"A6", "A7", "A7p", "A7f"}
LEDGER_ARMS = {"A3", "A4", "A5", "A7", "A8", "A7p"}


# ---------------------------------------------------------------- modules
class Traders(nn.Module):
    """n traders as one grouped conv. Trader i reads x, tanh(b/4), tanh(c_i) (its own bank only) and emits
    db_i, writes w_{i->j} (K channels into every bank j) and a gate g_i on its own bank."""

    def __init__(self, n=4, head=False, banks=True):
        super().__init__()
        self.n, self.banks = n, banks
        cin = CIN + 1 + K
        self.nout = 1 + n * K + K
        self.c1 = nn.Conv2d(n * cin, n * HID, 3, padding=1, groups=n)
        self.c2 = nn.Conv2d(n * HID, n * HID, 3, padding=1, groups=n)
        self.c3 = nn.Conv2d(n * HID, n * self.nout, 1, groups=n)
        self.head = nn.Conv2d(n * (HID + K), n * K, 1, groups=n) if head else None
        if head:  # prices start at zero, as the output layer of a DNI synthetic-gradient module (Jaderberg et al. 2017)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, b, c, want_h=False):
        B, _, H, W = x.shape
        n = self.n
        tc = torch.tanh(c) if self.banks else torch.zeros_like(c)
        inp = torch.cat([x.unsqueeze(1).expand(B, n, CIN, H, W),
                         torch.tanh(b / 4).unsqueeze(1).expand(B, n, 1, H, W), tc], 2).reshape(B, n * (CIN + 1 + K), H, W)
        h2 = F.relu(self.c2(F.relu(self.c1(inp))))
        o = self.c3(h2).view(B, n, self.nout, H, W)
        db = o[:, :, 0]
        w = torch.tanh(o[:, :, 1:1 + n * K]).reshape(B, n, n, K, H, W)  # [B, writer, target, K, H, W]
        g = torch.sigmoid(o[:, :, 1 + n * K:])  # [B, owner, K, H, W]
        P = None
        if self.head is not None:
            # the head reads the trader's features but its TD loss does not train the trader body (as in DNI)
            P = self.head(torch.cat([h2.detach().view(B, n, HID, H, W), tc.detach()], 2).reshape(B, n * (HID + K), H, W)).view(B, n, K, H, W)
        return (db, w, g, P, h2) if want_h else (db, w, g, P)

    def body_params(self):
        return [p for nm, p in self.named_parameters() if not nm.startswith("head")]


class DTRecall(nn.Module):
    """Recurrent conv net with input recall: h <- relu(a + conv(a)), a = relu(conv([h, x]))."""

    def __init__(self, w=28):
        super().__init__()
        self.proj = nn.Conv2d(CIN, w, 3, padding=1)
        self.r1 = nn.Conv2d(w + CIN, w, 3, padding=1)
        self.r2 = nn.Conv2d(w, w, 3, padding=1)
        self.out = nn.Conv2d(w, 1, 1)

    def init(self, x):
        return F.relu(self.proj(x))

    def step(self, h, x):
        a = F.relu(self.r1(torch.cat([h, x], 1)))
        return F.relu(a + self.r2(a))

    def logit(self, h):
        return self.out(h)[:, 0]


# ---------------------------------------------------------------- market maths
def pbar_fn(b, S):
    """Average LMSR price along 0 -> S: (softplus(b+S) - softplus(b)) / S, Taylor branch near 0 (no NaN grads)."""
    small = S.abs() < 1e-4
    Ss = torch.where(small, torch.ones_like(S), S)
    sig = torch.sigmoid(b)
    return torch.where(small, sig + S * sig * (1 - sig) / 2, (F.softplus(b + Ss) - F.softplus(b)) / Ss)


def dscore(b, S, y):
    """Change of the log score of outcome y when the logit moves from b to b+S."""
    return y * S - (F.softplus(b + S) - F.softplus(b))


class Ctx:
    """Everything an objective needs besides the trader outputs at one step."""

    def __init__(self, y, mask, b, s, Sig_other=None, P=None, r=None, phi=1.0):
        self.y, self.mask, self.b, self.s, self.Sig_other, self.P, self.r, self.phi = y, mask, b, s, Sig_other, P, r, phi

    def rep(self, m):
        f = lambda t: None if t is None else t.repeat((m,) + (1,) * (t.dim() - 1))
        return Ctx(f(self.y), f(self.mask), f(self.b), f(self.s), f(self.Sig_other), f(self.P), f(self.r), self.phi)


def own_obj(db, w, g, c_in, cx):
    """J_i = own AS pay + phi * royalties earned on writes at posted/swept prices + carried value of own bank.
    Other traders' contributions to Sigma enter only through the detached Sig_other, so d(sum_i J_i)/d theta_i
    is trader i's own-profit gradient."""
    own = cx.s[:, :, None, None] * db * cx.mask[:, None]
    Sig = cx.Sig_other + own
    pay = (own * (cx.y[:, None] - pbar_fn(cx.b, Sig))).sum((2, 3))  # [B, n]
    J = pay
    if cx.P is not None:
        v = (w * cx.P.unsqueeze(1)).sum((3, 4, 5))  # v[b, writer j, target i] = <w_{j->i}, P_i>
        J = J + cx.phi * (v * cx.r.transpose(1, 2)).sum(2)
        J = J + ((1 - g) * c_in * cx.P).sum((2, 3, 4))
    return J


def glob_obj(db, w, g, c_in, cx):
    """Global one-step objective (s = 1): telescoped score change + <c(t+1), sg P(t+1)>."""
    Sig = (db * cx.mask[:, None]).sum(1)
    J = (dscore(cx.b[:, 0], Sig, cx.y) * cx.mask).sum((1, 2))
    if cx.P is not None:
        J = J + (((1 - g) * c_in + w.sum(1)) * cx.P).sum((1, 2, 3, 4))
    return J


def ig_grad(model, x, b, c, cx, obj, m=4):
    """Aumann-Shapley / IG price: average d obj / d c along alpha*c, alpha at m midpoints (0 -> c)."""
    B = x.shape[0]
    al = (torch.arange(m, device=x.device, dtype=x.dtype) + 0.5) / m
    cp = (al.view(m, 1, 1, 1, 1, 1) * c.unsqueeze(0)).reshape((m * B,) + c.shape[1:]).detach().requires_grad_(True)
    with torch.enable_grad():
        db, w, g, _ = model(x.repeat(m, 1, 1, 1), b.repeat(m, 1, 1, 1), cp)
        J = obj(db, w, g, cp, cx.rep(m))
        gr, = torch.autograd.grad(J.sum(), cp)
    return gr.view((m, B) + c.shape[1:]).mean(0)


def point_grad(model, x, b, c, cx, obj):
    cp = c.detach().requires_grad_(True)
    with torch.enable_grad():
        db, w, g, _ = model(x, b, cp)
        gr, = torch.autograd.grad(obj(db, w, g, cp, cx).sum(), cp)
    return gr


# ---------------------------------------------------------------- ledger
class Ledger:
    """Per-example wealth of each trader (float64). Belief pay first, then royalties scaled to the payer's wealth.
    Statistics are accumulated on the device (no host sync per step); check() asserts conservation.
    W is the realised wealth (belief pay settled against the true y). The caps must not see y, or the ledger itself
    steers b towards the answer during training (a wrong trader goes broke and is capped, the right one keeps pushing)
    while s = 1 at test. So the caps use the guaranteed (solvency) wealth G = W0 + min over y of the cumulative belief
    pay + royalty cash that is y-free (posted prices, roy_caps=True). Royalties at posted prices are also scaled to G;
    royalties at realised-y prices (A4, A5) are scaled to W and never feed back into the caps."""

    def __init__(self, B, n, dev, on=True, W0=1.0, roy_caps=False):
        self.on, self.n, self.dev, self.W0, self.roy_caps = on, n, dev, W0, roy_caps
        self.W = torch.full((B, n), W0, dtype=torch.float64, device=dev)
        self.Q = torch.zeros((B, n, 2), dtype=torch.float64, device=dev)  # cumulative belief pay if y = 0 / y = 1
        self.Rc = torch.zeros((B, n), dtype=torch.float64, device=dev)  # y-free royalty cash
        self.G = self.W.clone()
        z = lambda: torch.zeros((), dtype=torch.float64, device=dev)
        self.st = {k: z() for k in ("trades", "capped", "roy_n", "roy_scaled", "subsidy", "roy_to_writers",
                                    "cons_err", "roy_on", "roy_off", "roy_abs")}

    def caps(self, db, count=True):
        """s_i = min(1, 0.5 W_i / sum|db_i|): worst-case belief loss at most half the current wealth."""
        if not self.on:
            return torch.ones(db.shape[:2], device=db.device, dtype=db.dtype)
        mag = db.detach().abs().sum((2, 3)).double()
        sc = torch.clamp(0.5 * self.G.clamp(min=0) / (mag + 1e-12), max=1.0)
        if count:
            act = mag > 1e-9
            self.st["trades"] += act.sum()
            self.st["capped"] += (act & (sc < 1)).sum()
        return sc.to(db.dtype)

    def belief(self, pay, dsc=None, pay01=None):
        """Add belief pay; if dsc (market maker loss = telescoped score change) is given, check sum pay = dsc.
        pay01[b, i, y]: the same pay had the outcome been y (y-free guaranteed wealth for the caps)."""
        p = pay.detach().double()
        self.W += p
        if pay01 is not None:
            self.Q += pay01.detach().double()
            self.G = self.W0 + self.Q.min(-1).values + self.Rc
        self.st["subsidy"] += p.sum()
        if dsc is not None:
            self.st["cons_err"] = torch.maximum(self.st["cons_err"], (p.sum(1) - dsc.detach().double()).abs().max())

    def royalties(self, w, P, phi, onpath):
        """Owner i pays writer j phi*<w_{j->i}, P_i> (negative: writer pays owner); self-writes carry no money.
        Payer outflows are scaled down to the payer's wealth. Returns r[b, i, j] = fraction actually paid."""
        n = self.n
        v = (w * P.unsqueeze(1)).sum((3, 4, 5))  # v[b, writer j, target i]
        offd = 1 - torch.eye(n, device=w.device, dtype=torch.float64)
        R = phi * v.transpose(1, 2).double() * offd  # R[b, owner i, writer j]
        von = phi * (w * (P * onpath[:, None, None]).unsqueeze(1)).sum((3, 4, 5)).transpose(1, 2).double() * offd
        self.st["roy_on"] += von.abs().sum()
        self.st["roy_off"] += (R - von).abs().sum()
        self.st["roy_abs"] += R.abs().sum()
        Pm = R.clamp(min=0) + (-R).clamp(min=0).transpose(1, 2)  # Pm[b, payer, payee]
        out = Pm.sum(2)
        Wp = (self.G if self.roy_caps else self.W).clamp(min=0)
        scale = torch.where(out > Wp, Wp / out.clamp(min=1e-300), torch.ones_like(out)) if self.on else torch.ones_like(out)
        Pm = Pm * scale[:, :, None]
        before = self.W.sum(1)
        self.W = self.W - Pm.sum(2) + Pm.sum(1)
        if self.roy_caps:
            self.Rc = self.Rc - Pm.sum(2) + Pm.sum(1)
            self.G = self.W0 + self.Q.min(-1).values + self.Rc
        self.st["cons_err"] = torch.maximum(self.st["cons_err"], (self.W.sum(1) - before).abs().max())
        B = R.shape[0]
        payer_scale = torch.where(R > 0, scale[:, :, None].expand(B, n, n), scale[:, None, :].expand(B, n, n))
        nz = R.abs() > 1e-12
        self.st["roy_n"] += nz.sum()
        self.st["roy_scaled"] += (nz & (payer_scale < 1)).sum()
        self.st["roy_to_writers"] += (R * payer_scale).clamp(min=0).sum()
        r = payer_scale.clone()
        r[:, torch.arange(n), torch.arange(n)] = 1.0
        return r.to(w.dtype)

    def summary(self):
        s = {k: float(v) for k, v in self.st.items()}
        assert s["cons_err"] < 1e-5, f"ledger conservation error {s['cons_err']}"
        out = dict(cons_err=s["cons_err"])
        if self.on:
            W = self.W.clamp(min=0)
            gini = (W[:, :, None] - W[:, None, :]).abs().sum((1, 2)) / (2 * self.n * W.sum(1).clamp(min=1e-12))
            out.update(capped=s["capped"] / max(s["trades"], 1), roy_scaled=s["roy_scaled"] / max(s["roy_n"], 1),
                       subsidy=s["subsidy"] / len(W), roy_to_writers=s["roy_to_writers"] / len(W),
                       gini=float(gini.mean()), neg_wealth=float((self.W < -1e-9).double().mean()),
                       roy_onpath_share=s["roy_on"] / max(s["roy_abs"], 1e-12), roy_abs=s["roy_abs"] / len(W))
        return out


# ---------------------------------------------------------------- data helpers
def to_dev(batch, dev):
    x, y, m, d, e = batch
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    return t(x), t(y), t(m), d, t(e)


def init_state(model, x):
    B, _, H, W = x.shape
    return x.new_zeros(B, 1, H, W), x.new_zeros(B, model.n, K, H, W)


def bce_at(logit_map, y, mask):
    """Mean BCE over scored cells; returns (loss, n_correct, n_scored)."""
    l = F.binary_cross_entropy_with_logits(logit_map, y, reduction="none")
    loss = (l * mask).sum() / mask.sum()
    corr = (((logit_map > 0).float() == y).float() * mask).sum()
    return loss, float(corr), float(mask.sum())


# ---------------------------------------------------------------- arm steps (each does its own backward)
def rollout_banks(model, x, mask, b, c, steps):
    for _ in range(steps):
        db, w, g, _ = model(x, b, c)
        b = b + (db * mask[:, None]).sum(1, keepdim=True)
        c = (1 - g) * c + w.sum(1)
    return b, c


def step_A1(model, x, y, mask, D, alpha, rng):
    """DT-Recall: BCE at depth D + Bansal progressive loss (n ~ U{0..D-1} steps no-grad, then k ~ U{1..D-n} with grad)."""
    h = model.init(x)
    hs = h
    for _ in range(D):
        hs = model.step(hs, x)
    lmax, corr, nsc = bce_at(model.logit(hs), y, mask)
    loss = lmax
    if alpha > 0:
        n = int(rng.integers(0, D))
        k = int(rng.integers(1, D - n + 1))
        with torch.no_grad():
            hp = h
            for _ in range(n):
                hp = model.step(hp, x)
        hp = hp.detach()
        for _ in range(k):
            hp = model.step(hp, x)
        lp, _, _ = bce_at(model.logit(hp), y, mask)
        loss = (1 - alpha) * lmax + alpha * lp
    loss.backward()
    return dict(loss=float(lmax), acc=corr / nsc)


def step_A2(model, x, y, mask, D, alpha, rng):
    """Traders + banks, full BPTT on the final belief b(D) (b path kept) + progressive loss."""
    b, c = init_state(model, x)
    bD, _ = rollout_banks(model, x, mask, b, c, D)
    lmax, corr, nsc = bce_at(bD[:, 0], y, mask)
    loss = lmax
    if alpha > 0:
        n = int(rng.integers(0, D))
        k = int(rng.integers(1, D - n + 1))
        with torch.no_grad():
            bn, cn = rollout_banks(model, x, mask, b, c, n)
        bk, _ = rollout_banks(model, x, mask, bn, cn, k)
        lp, _, _ = bce_at(bk[:, 0], y, mask)
        loss = (1 - alpha) * lmax + alpha * lp
    loss.backward()
    return dict(loss=float(lmax), acc=corr / nsc)


def step_A2t(model, x, y, mask, D, do_backward=True):
    """BPTT through the banks of the per-step telescoped log score at T (b detached, s = 1, global)."""
    b, c = init_state(model, x)
    tot = 0
    for _ in range(D):
        db, w, g, _ = model(x, b, c)
        Sig = (db * mask[:, None]).sum(1)
        tot = tot + (dscore(b[:, 0], Sig, y) * mask).sum((1, 2))
        b = (b + Sig.unsqueeze(1)).detach()
        c = (1 - g) * c + w.sum(1)
    loss = -tot.mean()
    if do_backward:
        loss.backward()
    lf, corr, nsc = bce_at(b[:, 0], y, mask)
    return dict(loss=float(lf), acc=corr / nsc, obj=float(tot.mean()))


def pay01(own, pb):
    """Belief pay [B, n, 2] had the outcome been y = 0 / y = 1 (own is already masked to the scored cells)."""
    return torch.stack([(own * (0 - pb)).sum((2, 3)), (own * (1 - pb)).sum((2, 3))], -1)


def write_stats(st, w):
    n = w.shape[1]
    a = w.detach().abs()
    st["wabs_self"] += a[:, torch.arange(n), torch.arange(n)].sum()
    st["wabs_all"] += a.sum()


def step_online(model, arm, x, y, mask, D, ex, st, true_adj=None, ig_m=4):
    """Online arms A3, A4, A6, A7 (+A7p, A7f): only the current and next step graphs are alive; each step's
    own-profit (or, for A6, global) loss is backpropagated at once and gradients are accumulated over the D steps.
    true_adj (test only) replaces the posted prices by given adjoints."""
    B, n = x.shape[0], model.n
    glob = arm == "A6"
    banks = arm != "A3"
    pointwise = arm in ("A6", "A7p")
    led = Ledger(B, n, x.device, on=(arm in LEDGER_ARMS) and not ex["dense"], roy_caps=arm in HEAD_ARMS)
    ones_s = torch.ones(B, n, device=x.device)
    ones_r = torch.ones(B, n, n, device=x.device)
    b, c = init_state(model, x)
    cur = model(x, b, c)
    s = ones_s if glob else led.caps(cur[0] * mask[:, None])
    td = []
    for t in range(D):
        db, w, g, P = cur
        own = s[:, :, None, None] * db * mask[:, None]
        Sig = own.sum(1, keepdim=True)
        Sig_other = (Sig - own).detach()
        b2 = (b + Sig).detach()
        c2 = ((1 - g) * c + w.sum(1)).detach() if banks else c
        if not glob:
            pb = pbar_fn(b, Sig)
            pay = (own * (y[:, None] - pb)).sum((2, 3))
            led.belief(pay, (dscore(b[:, 0], Sig[:, 0], y) * mask).sum((1, 2)), pay01(own, pb))
        nxt = model(x, b2, c2)
        s2 = ones_s if glob else led.caps(nxt[0] * mask[:, None], count=False)  # provisional, pre-royalty
        Pn = r = None
        if banks:
            if arm in HEAD_ARMS:
                Pn = nxt[3].detach() if true_adj is None else true_adj[t + 1]
            elif arm == "A4":  # one-hop: the owner's own belief pay at t+1 only
                nown = (s2[:, :, None, None] * nxt[0] * mask[:, None]).detach()
                Pn = ig_grad(model, x, b2, c2, Ctx(y, mask, b2, s2, nown.sum(1, keepdim=True) - nown), own_obj, ig_m)
            if not glob:
                r = led.royalties(w.detach(), Pn, ex["phi"], ex["onpath"])
                s2 = led.caps(nxt[0] * mask[:, None])  # final caps on post-royalty wealth
        cx = Ctx(y, mask, b, s, Sig_other, Pn, r, ex["phi"])
        loss = -glob_obj(db, w, g, c, cx).mean() if glob else -own_obj(db, w, g, c, cx).sum(1).mean()
        if arm in HEAD_ARMS and true_adj is None:  # TD(0) toward the one-step price expression at sg P(t+1)
            obj = glob_obj if glob else own_obj
            tcx = Ctx(y, mask, b, s, Sig_other, Pn, ones_r, ex["phi"])
            tgt = point_grad(model, x, b, c, tcx, obj) if pointwise else ig_grad(model, x, b, c, tcx, obj, ig_m)
            m2 = tgt.pow(2).mean((0, 3, 4)).detach()  # per (trader, channel) second moment
            ex["ema"] = m2 + 1e-12 if ex["ema"] is None else 0.99 * ex["ema"] + 0.01 * m2
            e2 = ((P - tgt).pow(2) / (ex["ema"][None, :, :, None, None] + 1e-12)).mean()
            loss = loss + e2
            td.append(e2.detach())
        loss.backward()
        write_stats(st, w)
        b, c, cur, s = b2, c2, nxt, s2
    lf, corr, nsc = bce_at(b[:, 0], y, mask)
    out = dict(loss=float(lf), acc=corr / nsc)
    if td:
        out["td"] = float(torch.stack(td).mean())
    out.update(led.summary())
    return out


def step_A5(model, arm, x, y, mask, D, ex, st, pointwise=False, ig_m=4, obj=None, return_lams=False):
    """Offline chained AS royalties (phi = 1): (1) no-grad rollout storing b, c and the caps s (from the belief-only
    ledger); (2) reverse sweep Lambda_i(t) = IG_c_i[pay_i(t) + phi sum_j <w_ij, sg L_j(t+1)> + <(1-g_i)c_i, sg L_i(t+1)>],
    Lambda(D) = 0; (3) forward again with graph, settle belief pay and royalties at the swept prices, update each
    trader on its own profit. The settled trajectory equals the rollout because s is fixed from pass 1."""
    B, n = x.shape[0], model.n
    obj = obj or own_obj
    on = arm not in ("A10",) and not ex["dense"]
    b, c = init_state(model, x)
    led1 = Ledger(B, n, x.device, on=on)
    bs, cs, ss = [], [], []
    with torch.no_grad():
        for t in range(D):
            db, w, g, _ = model(x, b, c)
            dbm = db * mask[:, None]
            s = led1.caps(dbm)
            own = s[:, :, None, None] * dbm
            Sig = own.sum(1, keepdim=True)
            pb = pbar_fn(b, Sig)
            led1.belief((own * (y[:, None] - pb)).sum((2, 3)), None, pay01(own, pb))
            bs.append(b); cs.append(c); ss.append(s)
            b = b + Sig
            c = (1 - g) * c + w.sum(1)
    lam = [None] * (D + 1)
    lam[D] = torch.zeros_like(c)
    ones = torch.ones(B, n, n, device=x.device, dtype=x.dtype)
    for t in range(D - 1, -1, -1):
        with torch.no_grad():
            db = model(x, bs[t], cs[t])[0]
            own = ss[t][:, :, None, None] * db * mask[:, None]
            Sig_other = own.sum(1, keepdim=True) - own
        cx = Ctx(y, mask, bs[t], ss[t], Sig_other, lam[t + 1], ones, ex["phi"])
        lam[t] = (point_grad(model, x, bs[t], cs[t], cx, obj) if pointwise
                  else ig_grad(model, x, bs[t], cs[t], cx, obj, ig_m)).detach()
    if return_lams:
        return lam
    led = Ledger(B, n, x.device, on=on)
    for t in range(D):
        db, w, g, _ = model(x, bs[t], cs[t])
        own = ss[t][:, :, None, None] * db * mask[:, None]
        Sig = own.sum(1, keepdim=True)
        pb = pbar_fn(bs[t], Sig)
        led.belief((own * (y[:, None] - pb)).sum((2, 3)), (dscore(bs[t][:, 0], Sig[:, 0], y) * mask).sum((1, 2)), pay01(own, pb))
        r = led.royalties(w.detach(), lam[t + 1], ex["phi"], ex["onpath"])
        cx = Ctx(y, mask, bs[t], ss[t], (Sig - own).detach(), lam[t + 1], r, ex["phi"])
        (-own_obj(db, w, g, cs[t], cx).sum(1).mean()).backward()
        write_stats(st, w)
    lf, corr, nsc = bce_at(b[:, 0], y, mask)
    out = dict(loss=float(lf), acc=corr / nsc)
    led.st["trades"], led.st["capped"] = led1.st["trades"], led1.st["capped"]
    out.update(led.summary())
    return out



# ---------------------------------------------------------------- build / train
def build(arm, dense=False):
    if arm in ("A1", "A1w"):
        return DTRecall(128 if arm == "A1w" else 28)
    return Traders(4, head=arm in HEAD_ARMS, banks=arm != "A3")


def grads_of(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.body_params()])


def train(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    if a.size == 9:
        ops = mazes.load_train()
    else:
        r0 = np.random.default_rng(5)
        ops = np.stack([mazes.dfs_maze(a.size, a.size, r0) for _ in range(2000)])
    L_train = json.load(open(os.path.join(mazes.DATA, "meta.json")))["L_train"] if a.size == 9 else 2 * a.size * a.size
    model = build(a.arm, a.dense).to(dev)
    head_p = [p for nm, p in model.named_parameters() if nm.startswith("head")]
    body_p = [p for nm, p in model.named_parameters() if not nm.startswith("head")]
    groups = [dict(params=body_p, lr=a.lr)] + ([dict(params=head_p, lr=a.lr * a.hlr)] if head_p else [])
    opt = torch.optim.Adam(groups)
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, a.iters // 20)
    ex = dict(phi=1.0, dense=a.dense, ema=None)
    log = []
    nparams = sum(p.numel() for p in model.parameters())
    print(json.dumps(dict(arm=a.arm, seed=a.seed, params=nparams, device=str(dev), args=vars(a))), flush=True)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    t1 = None
    acc_run = []
    for it in range(a.iters):
        f = min(1.0, (it + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
        for g_, lr0 in zip(opt.param_groups, base):
            g_["lr"] = lr0 * f
        cm = L_train if (a.curr and it < 0.75 * a.iters) else None
        if a.curr_inc and it < 0.75 * a.iters:  # diagnostic (not in the spec): max S-T distance grows 2 -> L_train
            cm = max(2, int(round(L_train * (it + 1) / (0.75 * a.iters))))
        x, y, mask, dist, e = to_dev(mazes.make_batch(ops, rng, a.bs, dense=a.dense, curriculum_max=cm), dev)
        ex["onpath"] = e[:, 0]
        st = dict(wabs_self=0.0, wabs_all=0.0)
        opt.zero_grad(set_to_none=True)
        if a.arm in ("A1", "A1w"):
            out = step_A1(model, x, y, mask, a.D, a.alpha, rng)
        elif a.arm == "A2":
            out = step_A2(model, x, y, mask, a.D, a.alpha, rng)
        elif a.arm == "A2t":
            out = step_A2t(model, x, y, mask, a.D)
        elif a.arm in ("A5", "A8", "A10"):
            out = step_A5(model, a.arm, x, y, mask, a.D, ex, st, pointwise=a.arm == "A8")
        else:
            out = step_online(model, a.arm, x, y, mask, a.D, ex, st)
        do_cos = a.cos_every and a.arm in ("A5", "A6", "A7", "A4", "A8", "A10", "A7p", "A7f") and it % a.cos_every == 0
        if do_cos:  # cosine between this arm's trader-body gradient and A2t's BPTT gradient on the same batch
            saved = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
            ga = grads_of(model)
            model.zero_grad(set_to_none=True)
            step_A2t(model, x, y, mask, a.D)
            out["cos_A2t"] = float(F.cosine_similarity(ga, grads_of(model), dim=0))
            for p, gp in zip(model.parameters(), saved):
                p.grad = gp
        gn = float(torch.nn.utils.clip_grad_norm_(body_p, a.clip))  # body and price head clipped separately
        if head_p:
            out["gnorm_head"] = float(torch.nn.utils.clip_grad_norm_(head_p, a.clip))
        opt.step()
        acc_run.append(out["acc"])
        if it == 0:
            t1 = time.time()
        if torch.is_tensor(st["wabs_all"]):
            out["cross_write_share"] = float(1 - st["wabs_self"] / st["wabs_all"].clamp(min=1e-12))
        if it % a.log_every == 0 or it == a.iters - 1 or do_cos:
            out.update(it=it, gnorm=gn, acc_avg=float(np.mean(acc_run[-a.log_every:])), s_per_it=(time.time() - t0) / (it + 1))
            log.append(out)
            print(json.dumps(out), flush=True)
    res = dict(arm=a.arm, seed=a.seed, args=vars(a), params=nparams, log=log, s_per_it=(time.time() - t0) / a.iters,
               s_per_it_after_first=(time.time() - t1) / max(1, a.iters - 1), train_acc_last=float(np.mean(acc_run[-200:])),
               peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else None)
    if a.out:
        sd = {k: (v.half() if a.arm == "A1w" else v).cpu() for k, v in model.state_dict().items()}
        torch.save(dict(state=sd, arm=a.arm, seed=a.seed, args=vars(a)), a.out)
        json.dump(res, open(a.out.replace(".pt", ".train.json"), "w"))
    return model, res


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--hlr", type=float, default=1.0)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--curr", action="store_true")
    ap.add_argument("--curr_inc", action="store_true", help="diagnostic: incremental curriculum")
    ap.add_argument("--cos_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="")
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
