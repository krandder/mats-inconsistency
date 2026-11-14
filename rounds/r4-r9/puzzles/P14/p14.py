"""P14: value-priced event publishing (LEG training + a LOCO listing market) against tuned sparse
backprop, on 9x9 mazes whose output is the whole map Y[c] = [dist(start, c) <= 4].

k = 4 publisher slots. Each slot is a trader MLP on the 5x5 (wall, start) window (50 -> 32 ReLU -> logit),
weights shared over cells; bits b ~ Bern(sigmoid(l)) in training, greedy [l > 0] at test. The reader R sees
the 5x5 windows of the k bit maps plus the 5x5 wall window (no start channel): 125 -> 32 ReLU -> P(Y[c]).
Loss and every metric cover open cells only; outside the maze counts as wall (bits outside = 0).

Estimators: ST (straight-through Bernoulli), STG (straight-through Gumbel-sigmoid), DisARM (Dong, Mnih &
Tucker 2020), LEG (the local-expectation gradient of Titsias & Lazaro-Gredilla 2015: exact whole-output flip
values V1 - V0 through R, computed as broadcast adds on R's first layer), LEG-local (own-cell term only).
Selection: listing market (prequential LOCO VoI from shadow readers R_-j, fee c, one delisting per check,
W-step cool-down, grace G, respawn with reader-row and Adam reset) or wealth market (per-bit no-refit
difference-reward pay, fee, seed subsidy, bankruptcy). Penalties: L1 on expected firing, L0 hard-concrete
slot gates (Louizos et al. 2018).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "P7")):
    if os.path.exists(os.path.join(p, "maze.py")):
        sys.path.insert(0, p)
        break
from maze import make_set  # noqa: E402  (P7's fixed-set generator, reused as specified)

N9 = 9
DENS = 0.3
K = 4
H = 32
LR = 3e-3
BATCH = 32
WIN = 250          # EMA window and delisting cool-down (steps)
EVERY = 100        # delisting / bankruptcy check period (steps)
STEPS_V = 6000     # V's step count; B = V's counted FLOPs over these steps
NCKPT = 12
# Probe optimiser. The specified 400 full-batch Adam steps at 3e-3 failed gate g6 (the exact g2 code only
# reached probe acc 0.976, loss 0.058; see RESULTS.md), so the probe uses minibatch Adam instead:
# 2000 steps of 4096 open-cell rows, lr 1e-2 with cosine decay (g2: loss 0.0018, acc 0.9998).
PROBE_STEPS = 2000
PROBE_LR = 1e-2
PROBE_BS = 4096
DCAP = 8           # training-stream BFS cap (T-reach needs <= 4, T-dist <= 5, codes <= 2)
UNREACH = 99

# L0 hard-concrete constants (Louizos et al. 2018)
L0_BETA, L0_GAMMA, L0_ZETA = 2.0 / 3.0, -0.1, 1.1

PLANTED = ["g2", "g2dup", "g1", "ring", "wallcopy", "random"]
G7CODES = ["zero", "random", "random", "random"]   # g7: slot 0 is switched to g2 mid-run

ARMS = {
    # est: leg | leg-local | st | stg | disarm | frozen | planted | oracle
    # sel: none | loco | wealth ; pen: none | l1 | l0
    "V": dict(est="leg", sel="loco"),
    "V-noD": dict(est="leg"),
    "BP-D": dict(est="st", sel="loco"),
    "BP": dict(est="st"),
    "V-L1": dict(est="leg", pen="l1"),
    "BP-L1-cm": dict(est="st", pen="l1"),
    "BP-L0-cm": dict(est="st", pen="l0"),
    "BP-L1-D-cm": dict(est="st", pen="l1", sel="loco"),
    "DisARM-L1-cm": dict(est="disarm", pen="l1"),
    "DisARM": dict(est="disarm"),
    "STG-L1-cm": dict(est="stg", pen="l1"),
    "STG": dict(est="stg"),
    "V-W": dict(est="leg", sel="wealth"),
    "V-local": dict(est="leg-local", sel="loco"),
    "N-sel": dict(est="frozen", sel="loco"),
    "N0": dict(est="frozen"),
    "P-LOCO": dict(est="planted", sel="loco", respawn=False, steps=STEPS_V),
    "P-W": dict(est="planted", sel="wealth", respawn=False, steps=STEPS_V),
    "ORACLE": dict(est="oracle", steps=STEPS_V),
    "G7": dict(est="planted", sel="loco", codes=G7CODES, steps=4500),
}


# ------------------------------------------------------------------------------------------ data
def batch_dist(walls, si, sj, cap=DCAP):
    """Vectorised BFS distance by frontier dilation; values > cap and unreachable -> UNREACH."""
    B, n, _ = walls.shape
    openm = ~walls
    dist = np.full((B, n, n), UNREACH, np.int64)
    fr = np.zeros((B, n, n), bool)
    fr[np.arange(B), si, sj] = True
    dist[fr] = 0
    seen = fr.copy()
    for d in range(1, cap + 1):
        nb = np.zeros_like(fr)
        nb[:, 1:] |= fr[:, :-1]
        nb[:, :-1] |= fr[:, 1:]
        nb[:, :, 1:] |= fr[:, :, :-1]
        nb[:, :, :-1] |= fr[:, :, 1:]
        nb &= openm & ~seen
        if not nb.any():
            break
        dist[nb] = d
        seen |= nb
        fr = nb
    return dist


class Stream:
    """Training mazes generated on the fly from rng seed 100000 + seed (paired across arms)."""

    def __init__(self, seed, n=N9):
        self.rng = np.random.default_rng(100000 + seed)
        self.n = n

    def next(self, B):
        n = self.n
        walls = self.rng.random((B, n, n)) < DENS
        si = self.rng.integers(n, size=B)
        sj = self.rng.integers(n, size=B)
        walls[np.arange(B), si, sj] = False
        start = np.zeros((B, n, n), bool)
        start[np.arange(B), si, sj] = True
        return walls, start, batch_dist(walls, si, sj)


def target(dist, task):
    d = np.minimum(dist, UNREACH)
    if task == "reach":
        return torch.tensor((d <= 4).astype(np.float32))
    return torch.tensor(np.minimum(d, 5).astype(np.int64))


CODES = {
    "g2": lambda d, w: (d <= 2),
    "d02": lambda d, w: (d == 0) | (d == 2),
    "d12": lambda d, w: (d >= 1) & (d <= 2),
    "ring": lambda d, w: (d == 2),
    "g1": lambda d, w: (d <= 1),
    "g0": lambda d, w: (d == 0),
}


def code_map(name, dist, walls):
    """Analytic code maps on all in-maze cells (complements: '~name', 1 at walls too)."""
    if name == "wallcopy":
        return walls.astype(np.float32)
    if name.startswith("~"):
        return 1.0 - code_map(name[1:], dist, walls)
    base = name.replace("dup", "")
    return CODES[base](dist, walls).astype(np.float32)


class EvalSet:
    """A fixed maze set (P7's make_set) with padded tensors and open-cell masks."""

    def __init__(self, num, n, seed):
        s = make_set(num, n, seed)
        self.n, self.num, self.seed = n, num, seed
        self.walls = s["x"][:, 0].numpy() > 0.5
        self.start = s["x"][:, 1].numpy() > 0.5
        self.dist = np.minimum(s["dist"], UNREACH)
        self.open = s["open"]
        self.onpath = s["onpath"]
        self.xpad = pad_x(torch.tensor(self.walls), torch.tensor(self.start))
        self.wpad = pad_w(torch.tensor(self.walls))
        self.openf = torch.tensor(self.open.astype(np.float32))

    def y(self, task):
        return target(self.dist, task)


# ------------------------------------------------------------------------------------------ models
def pad_x(walls, start):
    w = F.pad(walls.float()[:, None], (2, 2, 2, 2), value=1.0)
    s = F.pad(start.float()[:, None], (2, 2, 2, 2), value=0.0)
    return torch.cat([w, s], 1)


def pad_w(walls):
    return F.pad(walls.float()[:, None], (2, 2, 2, 2), value=1.0)


def uinit(shape, fan_in, gen):
    b = 1.0 / math.sqrt(fan_in)
    return (torch.rand(shape, generator=gen, dtype=torch.float64) * 2 - 1).mul(b).float()


class Traders:
    """k trader MLPs (50 -> H ReLU -> 1) as one grouped conv."""

    def __init__(self, k, gen):
        self.k = k
        self.W1 = uinit((k * H, 2, 5, 5), 50, gen)
        self.b1 = uinit((k * H,), 50, gen)
        self.W2 = uinit((k, H), H, gen)
        self.b2 = uinit((k,), H, gen)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def slot_slices(self, j):
        s = slice(j * H, (j + 1) * H)
        return [(0, s), (1, s), (2, slice(j, j + 1)), (3, slice(j, j + 1))]

    def fresh_slot(self, gen):
        return [uinit((H, 2, 5, 5), 50, gen), uinit((H,), 50, gen), uinit((1, H), H, gen), uinit((1,), H, gen)]

    def __call__(self, xpad):
        h = F.relu(F.conv2d(xpad, self.W1, self.b1))
        return F.conv2d(h, self.W2.view(self.k, H, 1, 1), self.b2, groups=self.k)


class ReaderBank:
    """G readers in one grouped conv. Group 0 is R; group 1+j (if shadows) is R_-j (slot j masked)."""

    def __init__(self, k, G, out, gen):
        self.k, self.G, self.out = k, G, out
        fin = (k + 1) * 25
        self.W1 = uinit((G * H, k + 1, 5, 5), fin, gen)
        self.b1 = uinit((G * H,), fin, gen)
        self.W2 = uinit((G * out, H), H, gen)
        self.b2 = uinit((G * out,), H, gen)
        self.mask = torch.ones(G, k)
        for g in range(1, G):
            self.mask[g, g - 1] = 0.0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, bits0, bits_rest, wpad):
        """bits0: (B,k,n,n) input to group 0 (may carry grad); bits_rest: detached input to groups >= 1."""
        B, k, n, _ = bits0.shape
        x0 = torch.cat([F.pad(bits0 * self.mask[0][None, :, None, None], (2, 2, 2, 2)), wpad], 1)
        h = F.conv2d(x0, self.W1[:H], self.b1[:H])
        if self.G > 1:
            xs = []
            for g in range(1, self.G):
                xs.append(F.pad(bits_rest * self.mask[g][None, :, None, None], (2, 2, 2, 2)))
                xs.append(wpad)
            hr = F.conv2d(torch.cat(xs, 1), self.W1[H:], self.b1[H:], groups=self.G - 1)
            h = torch.cat([h, hr], 1)
        lg = F.conv2d(F.relu(h), self.W2.view(self.G * self.out, H, 1, 1), self.b2, groups=self.G)
        return h, lg.view(B, self.G, self.out, n, n)

    def group0(self):
        return self.W1[:H], self.b1[:H], self.W2[:self.out], self.b2[:self.out]


def cell_ll(logit, y, task):
    """logit (B,out,n,n) -> per-cell log-likelihood (B,n,n)."""
    if task == "reach":
        return F.logsigmoid((2 * y - 1) * logit[:, 0])
    return F.log_softmax(logit, 1).gather(1, y[:, None])[:, 0]


def head_ll(h, W2, b2, y, task):
    lg = torch.einsum("bhij,oh->boij", F.relu(h), W2) + b2[None, :, None, None]
    return cell_ll(lg, y, task)


OFFS = [(di, dj) for di in range(-2, 3) for dj in range(-2, 3)]


def flip_values(h, W1g, W2g, b2g, bits, y, openf, task, local=False):
    """Exact whole-output flip values D[b,j,c] = V1 - V0, V_v = log P_R(Y | b_j(c) = v, rest).
    R's first layer is linear in the bits, so flipping b_j(c) adds +-W1[:, j, 2-o] to the pre-activation
    of each target cell t = c + o (|o| <= 2): 25 broadcast adds per slot. local=True keeps only o = 0."""
    B, Hh, n, _ = h.shape
    k = bits.shape[1]
    hL = h.permute(0, 2, 3, 1)                                           # (B,n,n,H)
    W2t = W2g.t()                                                        # (H,out)

    def ll_last(hh, yy):
        lg = F.relu(hh) @ W2t + b2g                                      # (...,n,n,out)
        if task == "reach":
            return F.logsigmoid((2 * yy - 1) * lg[..., 0])
        return F.log_softmax(lg, -1).gather(-1, yy.expand(lg.shape[:-1]).unsqueeze(-1))[..., 0]

    base = ll_last(hL, y) * openf
    offs = [(0, 0)] if local else OFFS
    delta = 1 - 2 * bits
    dpad = F.pad(delta, (2, 2, 2, 2))
    # dsh[j,o,b,t] = delta[b,j,t-o];  w[j,o,:] = W1[:, j, 2-di, 2-dj]
    dsh = torch.stack([dpad[:, :, 2 - di:2 - di + n, 2 - dj:2 - dj + n] for di, dj in offs], 0)   # (O,B,k,n,n)
    dsh = dsh.permute(2, 0, 1, 3, 4)                                                             # (k,O,B,n,n)
    w = torch.stack([W1g[:, :k, 2 - di, 2 - dj] for di, dj in offs], 0).permute(2, 0, 1)        # (k,O,H)
    hn = hL[None, None] + dsh.unsqueeze(-1) * w[:, :, None, None, None, :]                       # (k,O,B,n,n,H)
    d = (ll_last(hn, y) - base) * openf                                                          # (k,O,B,n,n)
    acc = torch.zeros(k, B, n + 4, n + 4, dtype=h.dtype)
    for o, (di, dj) in enumerate(offs):
        acc[:, :, 2 - di:2 - di + n, 2 - dj:2 - dj + n] += d[:, o]
    return acc[:, :, 2:2 + n, 2:2 + n].permute(1, 0, 2, 3) * delta


class EAdam:
    """Adam with element-wise step counts, so a respawn can zero the moments AND step counts of a slice."""

    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [torch.zeros_like(x) for x in params]
        self.v = [torch.zeros_like(x) for x in params]
        self.t = [torch.zeros_like(x) for x in params]

    def step(self):
        with torch.no_grad():
            for p, m, v, t in zip(self.p, self.m, self.v, self.t):
                if p.grad is None:
                    continue
                g = p.grad
                t.add_(1)
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                mh = m / (1 - self.b1 ** t)
                vh = v / (1 - self.b2 ** t)
                p.sub_(self.lr * mh / (vh.sqrt() + self.eps))
                p.grad = None

    def reset(self, i, idx):
        self.m[i][idx] = 0
        self.v[i][idx] = 0
        self.t[i][idx] = 0

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load(self, s):
        for a, b in zip(self.m, s["m"]):
            a.copy_(b)
        for a, b in zip(self.v, s["v"]):
            a.copy_(b)
        for a, b in zip(self.t, s["t"]):
            a.copy_(b)


# ------------------------------------------------------------------------------------------ FLOPs
class Flops:
    """Analytic FLOP counter, incremented at every module call from the call's shapes.
    mm = conv/matmul FLOPs (2 x MACs; what torch's FlopCounterMode counts, checked by gate g5);
    ew = the element-wise work of the flip evaluation (adds, ReLU, log-likelihood), counted for V's side."""

    def __init__(self):
        self.mm = 0
        self.ew = 0

    @property
    def total(self):
        return self.mm + self.ew

    def trader(self, N, k, backward):
        f1 = 2 * N * k * H * 50
        f2 = 2 * N * k * H
        self.mm += f1 + f2
        if backward:  # conv1: grad-weight only (input needs no grad); conv2: grad-input + grad-weight
            self.mm += f1 + 2 * f2

    def reader(self, N, k, G, out, backward, input_grad):
        f1 = 2 * N * G * H * (k + 1) * 25
        f2 = 2 * N * G * H * out
        self.mm += f1 + f2
        if backward:
            self.mm += f1 + 2 * f2 + (f1 // G if input_grad else 0)  # input grad only reaches group 0's bits

    def flips(self, N, k, O, out):
        self.mm += 2 * N * k * O * H * out
        self.ew += N * k * O * (3 * H + 6 * out)


# ------------------------------------------------------------------------------------------ market
class Listing:
    """Prequential LOCO VoI (EMA of d_j over W steps) against fee c; one delisting per check, W-step
    cool-down, grace G. Money is inert: VoI is a readout, c a threshold."""

    def __init__(self, k, fee, grace, respawn=True):
        self.k, self.fee, self.grace, self.respawn = k, fee, grace, respawn
        self.num = np.zeros(k)
        self.den = np.zeros(k)
        self.age = np.zeros(k, np.int64)
        self.listed = np.ones(k, bool)
        self.cool_until = 0
        self.events = []

    def update(self, d):
        a = 1.0 / WIN
        self.num = (1 - a) * self.num + a * d
        self.den = (1 - a) * self.den + a
        self.age += 1

    def voi(self):
        return self.num / np.maximum(self.den, 1e-12)

    def check(self, step):
        if step % EVERY or step < self.cool_until:
            return None
        v = self.voi()
        cand = [j for j in range(self.k) if self.listed[j] and v[j] < self.fee and self.age[j] > self.grace]
        if not cand:
            return None
        j = min(cand, key=lambda i: v[i])
        self.cool_until = step + WIN
        self.events.append({"step": step, "slot": j, "voi": float(v[j]), "voi_all": [float(x) for x in v],
                            "age": int(self.age[j])})
        if self.respawn:
            self.num[j] = self.den[j] = 0.0
            self.age[j] = 0
        else:
            self.listed[j] = False
        return j

    def state(self):
        return dict(num=self.num.tolist(), den=self.den.tolist(), age=self.age.tolist(),
                    listed=self.listed.tolist(), cool_until=self.cool_until, events=self.events)

    def load(self, s):
        self.num, self.den = np.array(s["num"]), np.array(s["den"])
        self.age, self.listed = np.array(s["age"], np.int64), np.array(s["listed"], bool)
        self.cool_until, self.events = s["cool_until"], s["events"]


class Wealth:
    """Money decides survival: pay_j = sum_c [log P_R(Y|B) - log E_{b~pi_j} P_R(Y | b_j(c)=b, rest)]
    (no-refit difference reward), fee c per maze, seed subsidy S = c*G, bankruptcy = lowest wealth < 0
    after grace (one per check, W-step cool-down). Conservation is asserted every step."""

    def __init__(self, k, fee, grace, respawn=True):
        self.k, self.fee, self.grace, self.respawn = k, fee, grace, respawn
        self.S = fee * grace
        self.wealth = np.full(k, self.S)
        self.subsidies = self.S * k
        self.minted = 0.0
        self.fees = 0.0
        self.written_off = 0.0
        self.pi_num = np.zeros(k)
        self.pi_den = np.zeros(k)
        self.age = np.zeros(k, np.int64)
        self.listed = np.ones(k, bool)
        self.cool_until = 0
        self.events = []
        self.max_err = 0.0

    def pi(self):
        p = np.where(self.pi_den > 0, self.pi_num / np.maximum(self.pi_den, 1e-12), 0.5)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def update(self, pay, rate):
        a = 1.0 / WIN
        self.pi_num = (1 - a) * self.pi_num + a * rate
        self.pi_den = (1 - a) * self.pi_den + a
        pay = np.where(self.listed, pay, 0.0)
        fee = np.where(self.listed, self.fee, 0.0)
        self.wealth += pay - fee
        self.minted += float(pay.sum())
        self.fees += float(fee.sum())
        self.age += 1
        err = abs(self.wealth[self.listed].sum() + self.written_off - (self.subsidies + self.minted - self.fees))
        scale = 1.0 + abs(self.subsidies) + abs(self.minted) + abs(self.fees)
        self.max_err = max(self.max_err, err / scale)
        assert err <= 1e-6 * scale, f"wealth ledger not conserved: err={err}"

    def check(self, step):
        if step % EVERY or step < self.cool_until:
            return None
        cand = [j for j in range(self.k) if self.listed[j] and self.wealth[j] < 0 and self.age[j] > self.grace]
        if not cand:
            return None
        j = min(cand, key=lambda i: self.wealth[i])
        self.cool_until = step + WIN
        self.events.append({"step": step, "slot": j, "wealth": float(self.wealth[j]),
                            "wealth_all": [float(x) for x in self.wealth], "age": int(self.age[j])})
        self.written_off += float(self.wealth[j])
        if self.respawn:
            self.wealth[j] = self.S
            self.subsidies += self.S
            self.age[j] = 0
            self.pi_num[j] = self.pi_den[j] = 0.0
        else:
            self.wealth[j] = 0.0
            self.listed[j] = False
        return j

    def books(self):
        return dict(S=self.S, subsidies=self.subsidies, minted=self.minted, fees=self.fees,
                    written_off=self.written_off, wealth=self.wealth.tolist(), bankruptcies=len(self.events),
                    max_rel_conservation_err=self.max_err)

    def state(self):
        d = self.books()
        d.update(pi_num=self.pi_num.tolist(), pi_den=self.pi_den.tolist(), age=self.age.tolist(),
                 listed=self.listed.tolist(), cool_until=self.cool_until, events=self.events)
        return d

    def load(self, s):
        self.S, self.subsidies, self.minted, self.fees = s["S"], s["subsidies"], s["minted"], s["fees"]
        self.written_off, self.wealth = s["written_off"], np.array(s["wealth"])
        self.pi_num, self.pi_den = np.array(s["pi_num"]), np.array(s["pi_den"])
        self.age, self.listed = np.array(s["age"], np.int64), np.array(s["listed"], bool)
        self.cool_until, self.events, self.max_err = s["cool_until"], s["events"], s["max_rel_conservation_err"]


# ------------------------------------------------------------------------------------------ system
class System:
    """One arm: traders (or planted / frozen / oracle bit sources), reader bank, selection, optimiser."""

    def __init__(self, arm, seed, task="reach", fee=None, lam=0.0, tau=1.0, grace=1000, codes=None):
        cfg = dict(sel="none", pen="none", respawn=True)
        cfg.update(ARMS[arm])
        if codes is not None:
            cfg["codes"] = codes
        self.arm, self.cfg, self.seed, self.task = arm, cfg, seed, task
        self.est, self.sel, self.pen = cfg["est"], cfg["sel"], cfg["pen"]
        self.fee, self.lam, self.tau, self.grace = fee, lam, tau, grace
        self.out = 1 if task == "reach" else 6
        self.codes = list(cfg.get("codes", PLANTED)) if self.est == "planted" else None
        self.k = len(self.codes) if self.est == "planted" else K
        g = torch.Generator().manual_seed(seed)          # paired init: same trader + reader init per seed
        self.traders = Traders(self.k, g)
        G = self.k + 1 if self.sel == "loco" else 1
        self.bank = ReaderBank(self.k, G, self.out, g)
        self.log_alpha = torch.zeros(self.k)
        self.gen = torch.Generator().manual_seed(10_000 + seed)       # bit sampling / noise
        self.respawn_gen = torch.Generator().manual_seed(20_000 + seed)
        self.respawns = 0
        trainable_traders = self.est in ("leg", "leg-local", "st", "stg", "disarm")
        self.params = self.bank.params() + (self.traders.params() if trainable_traders else [])
        if self.pen == "l0":
            self.params = self.params + [self.log_alpha]
        self.n_bank = len(self.bank.params())
        for p in self.params:
            p.requires_grad_(True)
        self.opt = EAdam(self.params)
        self.market = None
        if self.sel == "loco":
            self.market = Listing(self.k, fee, grace, respawn=cfg["respawn"])
        elif self.sel == "wealth":
            self.market = Wealth(self.k, fee, grace, respawn=cfg["respawn"])
        self.flops = Flops()
        self.step_n = 0
        self.snapshots = []   # greedy-bit snapshots at each delisting / bankruptcy

    # ---- bit sources
    def listed(self):
        if self.market is None:
            return np.ones(self.k, bool)
        return self.market.listed.copy()

    def logits(self, xpad, dist=None, walls=None):
        if self.est == "planted":
            rnd = self.traders(xpad)
            maps = []
            for j, nm in enumerate(self.codes):
                if nm == "random":
                    maps.append(rnd[:, j])
                elif nm == "zero":
                    maps.append(torch.full(rnd[:, j].shape, -20.0))
                else:
                    maps.append(torch.tensor(code_map(nm, dist, walls)) * 40 - 20)
            return torch.stack(maps, 1)
        if self.est == "oracle":
            B, n = xpad.shape[0], xpad.shape[-1] - 4
            l = torch.full((B, self.k, n, n), -20.0)
            l[:, 0] = torch.tensor(code_map("g2", dist, walls)) * 40 - 20
            return l
        return self.traders(xpad)

    def gates_test(self):
        if self.pen != "l0":
            return torch.ones(self.k)
        with torch.no_grad():
            return (torch.sigmoid(self.log_alpha) * (L0_ZETA - L0_GAMMA) + L0_GAMMA).clamp(0, 1)

    # ---- one training step
    def train_step(self, stream):
        walls, start, dist = stream.next(BATCH)
        y = target(dist, self.task)
        openf = torch.tensor((~walls).astype(np.float32))
        wt, st = torch.tensor(walls), torch.tensor(start)
        xpad, wpad = pad_x(wt, st), pad_w(wt)
        B, n = BATCH, N9
        N = B * n * n
        k, G, out = self.k, self.bank.G, self.out
        est = self.est
        trains_traders = est in ("leg", "leg-local", "st", "stg", "disarm")
        if trains_traders:
            l = self.logits(xpad)
        else:
            with torch.no_grad():
                l = self.logits(xpad, dist, walls)
        if est != "oracle":   # planted systems run all k trader MLPs for their frozen random slots
            self.flops.trader(N, k, backward=trains_traders)
        s = torch.sigmoid(l)
        u = torch.rand(l.shape, generator=self.gen)
        hard = (u < s.detach()).float()
        listed_mask = torch.tensor(self.listed().astype(np.float32))[None, :, None, None]
        loss = torch.zeros(())
        info = {}
        if est == "disarm":
            hard2 = ((1 - u) < s.detach()).float()
            h1, lg1 = self.bank.forward(hard, hard, wpad)
            h2, lg2 = self.bank.forward(hard2, hard2, wpad)
            ll1 = (cell_ll(lg1[:, 0], y, self.task) * openf).sum((1, 2))
            ll2 = (cell_ll(lg2[:, 0], y, self.task) * openf).sum((1, 2))
            self.flops.reader(N, k, 1, out, backward=True, input_grad=False)
            self.flops.reader(N, k, 1, out, backward=True, input_grad=False)
            loss = loss - 0.5 * (ll1.mean() + ll2.mean())
            f1, f2 = -ll1.detach(), -ll2.detach()
            gi = 0.5 * (f1 - f2)[:, None, None, None] * (1 - 2 * hard2) * (hard != hard2).float() \
                * torch.sigmoid(l.detach().abs())
            loss = loss + (l * gi).sum() / B
            bits_used = hard
        else:
            if est in ("st",):
                b0 = hard + s - s.detach()
            elif est == "stg":
                ul = torch.rand(l.shape, generator=self.gen).clamp(1e-6, 1 - 1e-6)
                ys = torch.sigmoid((l + torch.log(ul) - torch.log1p(-ul)) / self.tau)
                hard = (ys.detach() > 0.5).float()
                b0 = hard + ys - ys.detach()
            else:
                b0 = hard
            if self.pen == "l0":
                ug = torch.rand(k, generator=self.gen).clamp(1e-6, 1 - 1e-6)
                sg = torch.sigmoid((torch.log(ug) - torch.log1p(-ug) + self.log_alpha) / L0_BETA)
                z = (sg * (L0_ZETA - L0_GAMMA) + L0_GAMMA).clamp(0, 1)
                b0 = b0 * z[None, :, None, None]
                hard_in = hard * z.detach()[None, :, None, None]
            else:
                hard_in = hard
            b0 = b0 * listed_mask
            h, lg = self.bank.forward(b0, (hard_in * listed_mask).detach(), wpad)
            self.flops.reader(N, k, G, out, backward=True, input_grad=est in ("st", "stg") or self.pen == "l0")
            yG = y.unsqueeze(1).expand((B, G) + tuple(y.shape[1:])).reshape((B * G,) + tuple(y.shape[1:]))
            ll = cell_ll(lg.reshape(B * G, out, n, n), yG, self.task).view(B, G, n, n)
            llsum = (ll * openf[:, None]).sum((2, 3))          # (B,G) nats per maze
            loss = loss - llsum.mean(0).sum()
            if self.sel == "loco":
                d = (llsum[:, :1] - llsum[:, 1:]).mean(0).detach().numpy()   # prequential: before update
                self.market.update(d)
            if est in ("leg", "leg-local") or self.sel == "wealth":
                W1g, b1g, W2g, b2g = self.bank.group0()
                with torch.no_grad():
                    D = flip_values(h[:, :H].detach(), W1g.detach(), W2g.detach(), b2g.detach(),
                                    (hard_in * listed_mask).detach(), y, openf, self.task,
                                    local=(est == "leg-local"))
                self.flops.flips(N, k, 1 if est == "leg-local" else 25, out)
                if est in ("leg", "leg-local"):
                    loss = loss - (s * D * listed_mask).sum() / B
                if self.sel == "wealth":
                    pi = torch.tensor(self.market.pi(), dtype=torch.float32)[None, :, None, None]
                    hb = hard * listed_mask
                    # pay = V_b - log(pi e^{V1} + (1-pi) e^{V0}),  V1 - V_b = D (b=0) or 0;  V0 - V_b = -D (b=1) or 0
                    v1 = D * (1 - hb)
                    v0 = -D * hb
                    pay = -torch.logaddexp(torch.log(pi) + v1, torch.log1p(-pi) + v0)
                    pay_j = (pay * listed_mask).sum((2, 3)).mean(0).numpy()
                    rate = hard.mean((0, 2, 3)).numpy()
                    self.market.update(pay_j.astype(np.float64), rate)
            bits_used = hard
        if self.pen == "l1":
            loss = loss + self.lam * (s * listed_mask).sum() / B
        elif self.pen == "l0":
            p_open = torch.sigmoid(self.log_alpha - L0_BETA * math.log(-L0_GAMMA / L0_ZETA))
            loss = loss + self.lam * p_open.sum()
        loss.backward()
        self.opt.step()
        self.step_n += 1
        if self.market is not None:
            j = self.market.check(self.step_n)
            if j is not None:
                self.on_delist(j)
        info["loss"] = float(loss.detach())
        info["fire"] = float(bits_used.mean())
        return info

    # ---- respawn
    def fresh_rows(self):
        return uinit((H, 5, 5), (self.k + 1) * 25, self.respawn_gen)

    def on_delist(self, j):
        snap = None
        if getattr(self, "snap_fn", None) is not None:
            snap = self.snap_fn(j)
        self.snapshots.append({"step": self.step_n, "slot": int(j), "event": len(self.market.events) - 1,
                               "bits": snap})
        if not self.cfg["respawn"]:
            self.bank.mask[:, j] = 0.0
            return
        self.respawn(j)

    def respawn(self, j):
        self.respawns += 1
        with torch.no_grad():
            if self.est in ("leg", "leg-local", "st", "stg", "disarm", "frozen"):
                fresh = self.traders.fresh_slot(self.respawn_gen)
                tp = self.traders.params()
                for (i, sl), val in zip(self.traders.slot_slices(j), fresh):
                    tp[i][sl] = val
                    if self.est != "frozen":
                        self.opt.reset(self.n_bank + i, sl)
            for g in range(self.bank.G):
                if g == 1 + j and self.sel == "loco":
                    continue
                rows = slice(g * H, (g + 1) * H)
                self.bank.W1[rows, j] = self.fresh_rows()
                self.opt.reset(0, (rows, j))


# ------------------------------------------------------------------------------------------ probes
def windows(bits, wpad, openm):
    """bits (B,C,n,n) float, wpad (B,1,n+4,n+4) -> open-cell 5x5 windows (M, (C+1)*25)."""
    x = torch.cat([F.pad(bits, (2, 2, 2, 2)), wpad], 1)
    B, C, _, _ = x.shape
    n = bits.shape[-1]
    u = F.unfold(x, 5).view(B, C * 25, n * n).permute(0, 2, 1).reshape(B * n * n, C * 25)
    return u[torch.tensor(openm.reshape(-1))]


def train_probe(Xtr, ytr, Xte, yte, out=1, steps=PROBE_STEPS, lr=PROBE_LR, seed=0, n_draws=None):
    """Fresh probe reader (R's architecture on the given windows), full-batch Adam; returns test
    (log loss nats/cell, accuracy). Xte may be a list of draws (sampled bits): the scores are averaged."""
    g = torch.Generator().manual_seed(seed)
    D = Xtr.shape[1]
    W1 = uinit((D, H), D, g).requires_grad_(True)
    b1 = uinit((H,), D, g).requires_grad_(True)
    W2 = uinit((H, out), H, g).requires_grad_(True)
    b2 = uinit((out,), H, g).requires_grad_(True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
    gb = torch.Generator().manual_seed(seed + 1)
    M = Xtr.shape[0]

    def fwd(X):
        return F.relu(X @ W1 + b1) @ W2 + b2

    def nll(lg, yy):
        if out == 1:
            return F.binary_cross_entropy_with_logits(lg[:, 0], yy)
        return F.cross_entropy(lg, yy)

    for t in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * t / steps))
        idx = torch.randint(M, (PROBE_BS,), generator=gb)
        opt.zero_grad()
        loss = nll(fwd(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        tests = Xte if isinstance(Xte, list) else [Xte]
        ls, accs = [], []
        for X in tests:
            lg = fwd(X)
            ls.append(float(nll(lg, yte)))
            pred = (lg[:, 0] > 0).float() if out == 1 else lg.argmax(1)
            accs.append(float((pred == yte).float().mean()))
    return float(np.mean(ls)), float(np.mean(accs))


def entropy_bits(b):
    p = float(np.mean(b))
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def _H(c):
    c = c[c > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def mi(a, b):
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    nb = int(b.max()) + 1 if len(b) else 1
    return _H(np.bincount(a)) + _H(np.bincount(b)) - _H(np.bincount(a * nb + b))


def nmi(a, b):
    """I(a;b)/sqrt(H(a)H(b)); 0 if either is constant."""
    ha, hb = _H(np.bincount(np.asarray(a, np.int64))), _H(np.bincount(np.asarray(b, np.int64)))
    if ha < 1e-12 or hb < 1e-12:
        return 0.0
    return float(mi(a, b) / math.sqrt(ha * hb))


def nmi_min(a, b):
    ha, hb = _H(np.bincount(np.asarray(a, np.int64))), _H(np.bincount(np.asarray(b, np.int64)))
    if min(ha, hb) < 1e-12:
        return 0.0
    return float(mi(a, b) / min(ha, hb))


class ProbeKit:
    """Probe-train / probe-eval sets, cached wall windows and the g2-oracle probe loss L*."""

    def __init__(self, task, eval_seed, eval_num=300, train_seed=4101, train_num=1000):
        self.task = task
        self.out = 1 if task == "reach" else 6
        self.tr = EvalSet(train_num, N9, train_seed)
        self.te = EvalSet(eval_num, N9, eval_seed)
        self.ytr = self._y(self.tr)
        self.yte = self._y(self.te)
        self.lstar = None

    def _y(self, s):
        y = s.y(self.task).reshape(-1)[torch.tensor(s.open.reshape(-1))]
        return y.float() if self.task == "reach" else y

    def X(self, s, maps):
        """maps (B,C,n,n) numpy/tensor bits -> windows."""
        return windows(torch.as_tensor(maps, dtype=torch.float32), s.wpad, s.open)

    def probe(self, maps_tr, maps_te):
        # each bit channel is mean-centred on in-grid cells (probe-train mean) before zero padding, so a
        # code and its complement give the probe the same inputs up to a sign (deviation, see RESULTS.md)
        maps_tr = np.asarray(maps_tr, np.float32)
        mu = maps_tr.mean(axis=(0, 2, 3), keepdims=True)
        maps_tr = maps_tr - mu
        maps_te = [np.asarray(m, np.float32) - mu for m in maps_te] if isinstance(maps_te, list) \
            else np.asarray(maps_te, np.float32) - mu
        Xte = [self.X(self.te, m) for m in maps_te] if isinstance(maps_te, list) else self.X(self.te, maps_te)
        return train_probe(self.X(self.tr, maps_tr), self.ytr, Xte, self.yte, out=self.out)

    def code_maps(self, s, name):
        return code_map(name, s.dist, s.walls)[:, None]

    def get_lstar(self):
        if self.lstar is None:
            if self.task == "reach":
                self.lstar = self.probe(self.code_maps(self.tr, "g2"), self.code_maps(self.te, "g2"))
            else:  # the sufficient distance-class code {0,1,2,>2} as thermometer bits g0, g1, g2
                mtr = np.concatenate([self.code_maps(self.tr, c) for c in ("g0", "g1", "g2")], 1)
                mte = np.concatenate([self.code_maps(self.te, c) for c in ("g0", "g1", "g2")], 1)
                self.lstar = self.probe(mtr, mte)
        return self.lstar


# ------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def slot_logits(sysm, s):
    out = []
    for i in range(0, s.num, 100):
        out.append(sysm.logits(s.xpad[i:i + 100], s.dist[i:i + 100], s.walls[i:i + 100]))
    return torch.cat(out)


def live_slots(sysm, greedy_te, openm, extra=None):
    listed = sysm.listed()
    if extra is not None:
        listed[extra] = True
    gates = sysm.gates_test().numpy()
    live = []
    for j in range(sysm.k):
        if not listed[j] or gates[j] <= 0:
            continue
        if entropy_bits(greedy_te[:, j][openm]) >= 0.01:
            live.append(j)
    return live


@torch.no_grad()
def reader_eval(sysm, s, task, S=8, seed=0):
    """R's output log loss / accuracy on a fixed set: greedy bits and the 8-sample average predictive."""
    g = torch.Generator().manual_seed(seed)
    y = s.y(task)
    om = torch.tensor(s.open)
    gates = sysm.gates_test()
    listed = torch.tensor(sysm.listed().astype(np.float32))
    res = {}
    lgs = slot_logits(sysm, s)
    mult = (gates * listed)[None, :, None, None]

    def predict(bits):
        outs = []
        for i in range(0, s.num, 50):
            _, lg = sysm.bank.forward(bits[i:i + 50] * mult, bits[i:i + 50] * mult, s.wpad[i:i + 50])
            outs.append(lg[:, 0])
        lg = torch.cat(outs)
        if task == "reach":
            return torch.sigmoid(lg[:, 0])
        return F.softmax(lg, 1)

    pg = predict((lgs > 0).float())
    ps = sum(predict((torch.rand(lgs.shape, generator=g) < torch.sigmoid(lgs)).float()) for _ in range(S)) / S
    for nm, p in (("greedy", pg), ("sample8", ps)):
        if task == "reach":
            p = p.clamp(1e-6, 1 - 1e-6)
            ll = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
            acc = ((p > 0.5).float() == y).float()
        else:
            p = p.clamp_min(1e-6)
            ll = -torch.log(p.gather(1, y[:, None])[:, 0])
            acc = (p.argmax(1) == y).float()
        res[nm] = {"logloss": float(ll[om].mean()), "acc": float(acc[om].mean())}
    return res


def checkpoint_eval(sysm, kit, full, sets=None, seed=0):
    """Single-bit probes of every live slot (gap_1, SUFF1); full=True adds the joint probe, sampled-bit
    probes, code identity, saturation and output metrics on the fixed test sets."""
    t0 = time.time()
    task = kit.task
    Ls, Lacc = kit.get_lstar()
    lg_tr = slot_logits(sysm, kit.tr)
    lg_te = slot_logits(sysm, kit.te)
    gtr = (lg_tr > 0).float().numpy()
    gte = (lg_te > 0).float().numpy()
    om = kit.te.open
    live = live_slots(sysm, gte, om)
    res = {"step": sysm.step_n, "flops": sysm.flops.total, "live": live, "listed": sysm.listed().tolist(),
           "Lstar": Ls, "Lstar_acc": Lacc}
    probes = {}
    seen = {}
    for j in live:
        key = gtr[:, j].tobytes() + gte[:, j].tobytes()
        if key in seen:
            probes[j] = dict(probes[seen[key]], dup_of=seen[key])
            continue
        seen[key] = j
        ll, acc = kit.probe(gtr[:, j:j + 1], gte[:, j:j + 1])
        probes[j] = {"loss": ll, "acc": acc, "gap": ll - Ls}
    res["probes"] = {str(j): v for j, v in probes.items()}
    if live:
        jb = min(live, key=lambda j: probes[j]["gap"])
        res["best_slot"] = jb
        res["gap_1"] = probes[jb]["gap"]
        res["suff1"] = bool(probes[jb]["gap"] <= 0.02 and probes[jb]["acc"] >= 0.995)
    else:
        res["best_slot"], res["gap_1"], res["suff1"] = None, None, False
    if full:
        g = torch.Generator().manual_seed(1234 + seed)
        if live:
            ll, acc = kit.probe(gtr[:, live], gte[:, live])
            res["gap_all"], res["acc_all"] = ll - Ls, acc
            res["suffall"] = bool(ll - Ls <= 0.02 and acc >= 0.995) if task == "reach" else None
            # sampled bits: probe trained on one sampled draw, scored on 8 sampled test draws
            str_ = (torch.rand(lg_tr.shape, generator=g) < torch.sigmoid(lg_tr)).float().numpy()
            ste = [(torch.rand(lg_te.shape, generator=g) < torch.sigmoid(lg_te)).float().numpy() for _ in range(8)]
            sp = {}
            for j in live:
                ll, acc = kit.probe(str_[:, j:j + 1], [d[:, j:j + 1] for d in ste])
                sp[str(j)] = {"loss": ll, "acc": acc, "gap": ll - Ls}
            res["probes_sample8"] = sp
            jbs = min(live, key=lambda j: sp[str(j)]["gap"])
            res["gap_1_sample8"] = sp[str(jbs)]["gap"]
            res["suff1_sample8"] = bool(sp[str(jbs)]["gap"] <= 0.02 and sp[str(jbs)]["acc"] >= 0.995)
        else:
            res.update(gap_all=None, acc_all=None, suffall=False, gap_1_sample8=None, suff1_sample8=False)
        # code identity on probe-eval open cells
        te = kit.te
        fam = ["g2", "d02", "d12", "ring", "g1", "g0"]
        cm = {c: code_map(c, te.dist, te.walls)[om].astype(np.int64) for c in fam}
        dcls = np.minimum(te.dist, 3)[om]
        onp = te.onpath[om].astype(np.int64)
        ident = {}
        for j in live:
            b = gte[:, j][om].astype(np.int64)
            d = {c: nmi(b, cm[c]) for c in fam}
            d["dist_class"] = nmi(b, dcls)
            d["onpath"] = nmi(b, onp)
            # walls: a bit that encodes the local wall pattern (count of open neighbours) as a negative control
            wn = walls_nb(te.walls)[om]
            d["wall_nbrs"] = nmi(b, wn)
            d["fires_per_maze"] = float(gte[:, j].sum() / te.num)
            ident[str(j)] = d
        res["identity"] = ident
        if live:
            joint = np.zeros(int(om.sum()), np.int64)
            for j in live:
                joint = joint * 2 + gte[:, j][om].astype(np.int64)
            res["joint_nmi"] = {"g2": nmi(joint, cm["g2"]), "dist_class": nmi(joint, dcls),
                                "n_states": int(len(np.unique(joint)))}
            res["fires_per_maze_live"] = float(gte[:, live].sum() / te.num)
        # saturation: mean binary entropy of sigmoid(l) over live slots, open cells
        sg = torch.sigmoid(lg_te).clamp(1e-7, 1 - 1e-7).numpy()
        ent = -(sg * np.log(sg) + (1 - sg) * np.log(1 - sg))
        res["mean_bit_entropy"] = float(np.mean([ent[:, j][om].mean() for j in live])) if live else 0.0
        res["mean_bit_entropy_all"] = float(np.mean([ent[:, j][om].mean() for j in range(sysm.k)]))
        res["collapse"] = len(live) == 0
        res["n_live"] = len(live)
        if sets:
            res["output"] = {str(sz): reader_eval(sysm, st, task, seed=seed) for sz, st in sets.items()}
        res["live_bits_te"] = {str(j): np.packbits(gte[:, j].astype(np.uint8)).tolist() for j in live} \
            if task == "reach" else None
    res["eval_s"] = round(time.time() - t0, 1)
    return res


def walls_nb(walls):
    w = np.pad(walls, ((0, 0), (1, 1), (1, 1)), constant_values=True)
    return (w[:, :-2, 1:-1].astype(int) + w[:, 2:, 1:-1] + w[:, 1:-1, :-2] + w[:, 1:-1, 2:])


def snapshot_bits(sysm, kit, j):
    """Greedy bits of every slot on probe-train/probe-eval at a delisting (before the respawn); live set
    as it was just before slot j was delisted."""
    lg_tr = slot_logits(sysm, kit.tr)
    lg_te = slot_logits(sysm, kit.te)
    gtr = (lg_tr > 0).numpy().astype(np.uint8)
    gte = (lg_te > 0).numpy().astype(np.uint8)
    live = live_slots(sysm, gte.astype(np.float32), kit.te.open, extra=j)
    return {"tr": np.packbits(gtr), "te": np.packbits(gte), "shape_tr": gtr.shape, "shape_te": gte.shape,
            "live_before": live}


def delisting_consequences(snaps, kit):
    """Consequence-defined false delisting: the live set's joint-probe gap goes from <= 0.02 (before, incl. the
    delisted slot) to > 0.02 (after, without it). Computed offline from the greedy-bit snapshots."""
    Ls, _ = kit.get_lstar()
    om = kit.te.open
    out = []
    fam = ["g2", "d02", "d12", "ring", "g1", "g0"]
    cm = {c: code_map(c, kit.te.dist, kit.te.walls)[om].astype(np.int64) for c in fam}
    for sn in snaps:
        b = sn["bits"]
        if b is None:
            continue
        gtr = np.unpackbits(b["tr"])[:int(np.prod(b["shape_tr"]))].reshape(b["shape_tr"]).astype(np.float32)
        gte = np.unpackbits(b["te"])[:int(np.prod(b["shape_te"]))].reshape(b["shape_te"]).astype(np.float32)
        j = sn["slot"]
        before = list(b["live_before"])
        if j not in before:
            before_set = before
        else:
            before_set = before
        rec = {"step": sn["step"], "slot": j, "live_before": before}
        bj = gte[:, j][om].astype(np.int64)
        rec["nmi_min_family"] = {c: nmi_min(bj, cm[c]) for c in fam}
        rec["delisted_was_live"] = j in before
        if before_set:
            ll, acc = kit.probe(gtr[:, before_set], gte[:, before_set])
            rec["gap_before"] = ll - Ls
            after = [i for i in before_set if i != j]
            if rec["gap_before"] <= 0.02 and j in before_set:
                if after:
                    ll2, _ = kit.probe(gtr[:, after], gte[:, after])
                    rec["gap_after"] = ll2 - Ls
                else:
                    rec["gap_after"] = kit.probe(np.zeros_like(gtr[:, :1]), np.zeros_like(gte[:, :1]))[0] - Ls
                rec["false"] = bool(rec["gap_after"] > 0.02)
            else:
                rec["false"] = False
        else:
            rec["false"] = False
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------ budget
def per_step_flops(arm, task="reach"):
    """Measure one step's counted FLOPs (independent of hyper-parameters) on a throwaway system."""
    s = System(arm, seed=0, task=task, fee=1.0, lam=0.0, tau=1.0, grace=10 ** 9)
    st = Stream(12345)
    s.train_step(st)
    return s.flops.total


def budget(task="reach"):
    return STEPS_V * per_step_flops("V", task)


# ------------------------------------------------------------------------------------------ run
def run(args):
    torch.set_num_threads(1)
    task = args.task
    cfg = dict(sel="none", pen="none")
    cfg.update(ARMS[args.arm])
    name = f"{args.arm}_{task}_{args.mode}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out, exist_ok=True)
    fjson = os.path.join(args.out, name + ".json")
    if os.path.exists(fjson):
        print("SKIP exists", fjson)
        return
    fpart = os.path.join(args.out, name + ".partial")
    B = args.budget if args.budget else budget(task)
    sysm = System(args.arm, args.seed, task, fee=args.fee, lam=args.lam, tau=args.tau, grace=args.grace)
    fixed_steps = cfg.get("steps")
    if fixed_steps and args.max_steps:  # smoke tests only
        fixed_steps = min(fixed_steps, args.max_steps)
    stream = Stream(args.seed)
    tune = args.mode == "tune"
    kit = ProbeKit(task, eval_seed=777 if tune else 4102)
    if not tune and sysm.market is not None:
        sysm.snap_fn = lambda j: snapshot_bits(sysm, kit, j)
    sets = None
    if not tune:
        sets = {9: EvalSet(300, 9, 786), 15: EvalSet(120, 15, 792), 21: EvalSet(60, 21, 798)}
    else:
        sets = {9: EvalSet(300, 9, 777)}
    per_step = None
    ckpts = [] if tune else [B * i / NCKPT for i in range(1, NCKPT + 1)]
    state = {"ckpt_i": 0, "results": [], "step6000": None, "log": [], "t_train": 0.0}
    t_start = time.time()
    if os.path.exists(fpart):
        st = torch.load(fpart, weights_only=False)
        load_system(sysm, st["sys"])
        stream.rng.bit_generator.state = st["stream"]
        state = st["state"]
        print("RESUMED", name, "step", sysm.step_n, flush=True)
    last_save = time.time()

    def done():
        if fixed_steps:
            return sysm.step_n >= fixed_steps
        return sysm.flops.total >= B

    t0 = time.time()
    while not done():
        info = sysm.train_step(stream)
        if sysm.step_n % 500 == 0:
            rec = {"step": sysm.step_n, "loss": round(info["loss"], 4), "fire": round(info["fire"], 4),
                   "flops": sysm.flops.total, "t": round(time.time() - t0, 1)}
            if sysm.sel == "loco":
                rec["voi"] = [round(float(x), 3) for x in sysm.market.voi()]
            if sysm.sel == "wealth":
                rec["wealth"] = [round(float(x), 2) for x in sysm.market.wealth]
            state["log"].append(rec)
            print(json.dumps(rec), flush=True)
        if per_step is None:
            per_step = sysm.flops.total / sysm.step_n
        # equal-steps view
        if not tune and sysm.step_n == STEPS_V and not (fixed_steps or abs(B - sysm.flops.total) < per_step):
            r = checkpoint_eval(sysm, kit, full=True, sets=sets, seed=args.seed)
            state["step6000"] = r
        # FLOP checkpoints (planted / oracle arms: equally spaced steps)
        prog = (sysm.step_n / fixed_steps * B) if fixed_steps else sysm.flops.total
        while state["ckpt_i"] < len(ckpts) and prog >= ckpts[state["ckpt_i"]] - 1e-6:
            last = state["ckpt_i"] == len(ckpts) - 1
            if not last:
                r = checkpoint_eval(sysm, kit, full=False, seed=args.seed)
                r["i"] = state["ckpt_i"] + 1
                state["results"].append(r)
                print("CKPT", r["i"], "gap_1", r["gap_1"], "suff1", r["suff1"], "live", r["live"], flush=True)
            state["ckpt_i"] += 1
            last_save = save_partial(fpart, sysm, stream, state)
        if time.time() - last_save > 600:
            last_save = save_partial(fpart, sysm, stream, state)
    state["t_train"] = time.time() - t_start
    final = checkpoint_eval(sysm, kit, full=True, sets=sets, seed=args.seed)
    final["i"] = NCKPT
    if not tune:
        state["results"].append(final)
    out = {"arm": args.arm, "task": task, "seed": args.seed, "mode": args.mode,
           "hp": {"fee": args.fee, "lam": args.lam, "tau": args.tau, "grace": args.grace},
           "B": B, "steps": sysm.step_n, "flops": sysm.flops.total, "flops_mm": sysm.flops.mm,
           "flops_ew": sysm.flops.ew, "wall_s": round(time.time() - t_start, 1), "final": final,
           "checkpoints": state["results"], "step6000": state["step6000"], "log": state["log"],
           "respawns": sysm.respawns}
    if sysm.est == "oracle":
        out["final"] = {"output": final.get("output")}
    if sysm.market is not None:
        out["market"] = sysm.market.state() if sysm.sel == "loco" else sysm.market.state()
        if sysm.sel == "loco":
            out["market"]["final_voi"] = [float(x) for x in sysm.market.voi()]
        out["delistings"] = len(sysm.market.events)
        if not tune:
            out["delisting_quality"] = delisting_consequences(sysm.snapshots, kit)
            out["false_delistings"] = int(sum(1 for r in out["delisting_quality"] if r.get("false")))
    if sysm.pen == "l0":
        out["gates_test"] = sysm.gates_test().tolist()
    with open(fjson + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(fjson + ".tmp", fjson)
    if os.path.exists(fpart):
        os.remove(fpart)
    print("DONE", name, json.dumps({"gap_1": final.get("gap_1"), "suff1": final.get("suff1"),
                                    "steps": sysm.step_n, "wall_s": out["wall_s"]}), flush=True)


def save_partial(fpart, sysm, stream, state):
    st = {"sys": system_state(sysm), "stream": stream.rng.bit_generator.state, "state": state}
    torch.save(st, fpart + ".tmp")
    os.replace(fpart + ".tmp", fpart)
    return time.time()


def system_state(s):
    d = {"traders": [p.detach().clone() for p in s.traders.params()],
         "bank": [p.detach().clone() for p in s.bank.params()], "mask": s.bank.mask.clone(),
         "log_alpha": s.log_alpha.detach().clone(), "opt": s.opt.state(), "gen": s.gen.get_state(),
         "respawn_gen": s.respawn_gen.get_state(), "respawns": s.respawns, "step": s.step_n,
         "flops": (s.flops.mm, s.flops.ew), "snapshots": s.snapshots}
    if s.market is not None:
        d["market"] = s.market.state()
    return d


def load_system(s, d):
    with torch.no_grad():
        for p, v in zip(s.traders.params(), d["traders"]):
            p.copy_(v)
        for p, v in zip(s.bank.params(), d["bank"]):
            p.copy_(v)
        s.bank.mask.copy_(d["mask"])
        s.log_alpha.copy_(d["log_alpha"])
    s.opt.load(d["opt"])
    s.gen.set_state(d["gen"])
    s.respawn_gen.set_state(d["respawn_gen"])
    s.respawns, s.step_n = d["respawns"], d["step"]
    s.flops.mm, s.flops.ew = d["flops"]
    s.snapshots = d["snapshots"]
    if s.market is not None:
        s.market.load(d["market"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="reach", choices=["reach", "dist"])
    ap.add_argument("--mode", default="eval", choices=["eval", "tune"])
    ap.add_argument("--fee", type=float, default=None)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--grace", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests: cap fixed-step arms")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runs"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
