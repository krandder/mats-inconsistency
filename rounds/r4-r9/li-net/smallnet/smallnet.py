"""ShipHunt-hard small-network data-efficiency experiment: one training run per call (eurisko, 2026-10-10).

  python3 smallnet.py --arm k0 --n 300 --seed 0 --out RUN.json [--set key=value ...] [--test]

Every arm trains the same Net (model.py) on N training boards. Each board has ONE observed probe trajectory
and its realised true board; stages = prefixes after 0, 2, 4, 6 probes. Symmetry augmentation (8 D4
symmetries, random per board per step) in every arm. The exact posterior is used only for evaluation and by
the evaluation-only reference arm 'refpost'. Early stopping uses the validation boards' OUTCOMES (per-cell
log-loss against the true cells), never the posterior. Metrics: single-cell KL(P* || q) per stage (primary),
outcome log-loss, ECE, joint log-likelihood of the true board, compute (FLOPs) per training board.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bninfer as BI  # noqa: E402
import data as D  # noqa: E402
from netdef import NC, PAIRS, Net  # noqa: E402

# ----------------------------------------------------------------------------------------------- config
BASE = dict(
    # network (fixed across arms)
    d=128, layers=4, heads=4, ff=256, m=64, drop=0.1,
    # optimisation
    lr=1e-3, wd=0.05, B=64, warmup=100, clip=1.0, ema=0.0,   # lr and ema are set per N by steps_for()
    max_steps=0, eval_every=0, patience=25, es_boards=1000,
    # what the arm trains
    k0w=1.0,            # per-cell log score of the single-cell head vs the true board (MC target)
    target="board",     # 'board' | 'post' (refpost: exact posterior marginals; evaluation-only reference)
    stages="even",      # 'even' prefixes 0,2,4,6 | 'all' prefixes 0..6 | 'perm' prefixes 0..6 of a random reorder of the observed probes
    td=-1.0,            # TD(lambda) across stages on the single-cell head (-1 = off; 0 = pure TD; 1 = MC)
    td_warm=0,          # steps of plain MC before TD switches on
    mart=0.0,           # martingale drift penalty weight (label-free; batch-mean drift per stage and forecast bin)
    bn=0, bnw=1.0, bn_fixed=0, bn_beta=0.1, bn_tau=1.0,   # Bayes-net loss with k = bn parents
    masked=0.0, mask_R=1,   # masked-cell auxiliary: reveal each cell w.p. r ~ U(0,1), predict the hidden ones
    chain=0.0, chain_R=1,   # full true-value chain (orderless autoregressive objective)
    rbtd=0, rbtd_lam=0.0, rbtd_w=1.0, rbtd_R=1,  # reveal-chain TD: levels of revealed true cells, each trained toward the next; R chains
    cell_q=0,           # cell-by-cell arm: query passes per board-stage (25 = all cells)
    readout="marg",     # 'marg' | 'bn' | 'cell'
    eval_M=4,           # random orders averaged for Bayes-net readouts at the final evaluation
    amp=0,              # 1 = bf16 autocast for the training forward pass (CUDA only); evaluation stays fp32
    teacher=0,          # 1 = TD / rbtd bootstrap targets come from the EMA network in eval mode (mean teacher), not the online net
)

ARMS = {
    "k0": dict(),                                                   # full map: 25 cell probabilities at once
    "k0cell": dict(k0w=0.0, cell_q=25, readout="cell"),             # cell by cell: one query per pass
    "k1": dict(k0w=0.0, bn=1, readout="bn"),
    "k2": dict(k0w=0.0, bn=2, readout="bn"),
    "k2fixed": dict(k0w=0.0, bn=2, bn_fixed=1, readout="bn"),
    "masked": dict(masked=1.0),
    "chain": dict(k0w=0.0, chain=1.0),
    "td": dict(td=0.0),
    "mart": dict(mart=100.0),
    "refpost": dict(target="post"),
    "k1marg": dict(bn=1),
    "k2marg": dict(bn=2),
    "rbtd": dict(k0w=0.0, rbtd=4, rbtd_lam=0.0),
    "allprefix": dict(stages="all"),
    "perm": dict(stages="perm"),
}

# real-world validity of each ingredient (see PREREG.md); refpost uses the posterior in training.
NOT_REAL_WORLD = {"target": ("post",)}


def steps_for(n: int):
    """Per-N defaults (max_steps, eval_every, lr, ema), calibrated on k0 (PREREG deviations 16:35Z):
    smoke + cal2 runs showed lr 1e-3 overfits within ~100 steps at N=100 (lower lr better there) but is clearly
    best at N=300; EMA of weights damps the large eval-to-eval swings seen at N >= 1000."""
    table = {100: (2000, 10, 3e-4, 0.95), 300: (4000, 20, 1e-3, 0.95), 1000: (10000, 50, 1e-3, 0.99),
             3000: (20000, 100, 1e-3, 0.99), 10000: (40000, 200, 1e-3, 0.99)}
    if n in table:
        return table[n]
    keys = sorted(table)
    k = min(keys, key=lambda a: abs(math.log(a) - math.log(n)))
    return table[k]


# ----------------------------------------------------------------------------------------------- data on device
class TrainSet:
    def __init__(self, n, seed, dev, need_post=False):
        d = D.load_split("train", max(10_000, n), seed=seed)   # nested prefixes; N > 10k only for the final check
        self.occ = torch.tensor(d["occ"][:n], dtype=torch.long, device=dev)
        self.pc = torch.tensor(d["pc"][:n], dtype=torch.long, device=dev)
        self.pr = torch.tensor(d["pr"][:n], dtype=torch.long, device=dev)
        self.n = n
        self.post = None
        if need_post:   # evaluation-only reference arm 'refpost' (exact posterior targets)
            assert n <= 10_000, "refpost posteriors are cached for the first 10k training boards only"
            path = os.path.join(D.DATA_DIR, f"train-post-n10000-s{seed}.npy")
            if os.path.exists(path):
                post = np.load(path)
            else:
                post, _ = D.exact_posteriors(d["seeds"])
                np.save(path, post.astype(np.float32))
            self.post = torch.tensor(post[:n], dtype=torch.float32, device=dev)  # [n,4,25] stages 0,2,4,6
        self.sym = torch.tensor(D.SYM, dtype=torch.long, device=dev)
        self.dev = dev

    def batch(self, B, stages, gen):
        idx = torch.randint(0, self.n, (B,), device=self.dev, generator=gen)
        g = torch.randint(0, 8, (B,), device=self.dev, generator=gen)
        perm = self.sym[g]                                    # [B,25]
        x = self.occ[idx].gather(1, perm)                     # true board in the transformed frame
        pc, pr = self.pc[idx], self.pr[idx]
        if stages == "perm":
            o = torch.argsort(torch.rand(B, 6, device=self.dev, generator=gen), 1)
            pc, pr = pc.gather(1, o), pr.gather(1, o)
        sizes = [0, 2, 4, 6] if stages == "even" else list(range(7))
        S = len(sizes)
        ev = torch.zeros(B, S, NC, dtype=torch.long, device=self.dev)
        for si, s in enumerate(sizes):
            for k in range(s):
                ev[torch.arange(B, device=self.dev), si, pc[:, k]] = 1 + pr[:, k]
        ev = ev.gather(2, perm[:, None, :].expand(B, S, NC))
        post = None
        if self.post is not None:
            post = self.post[idx].gather(2, perm[:, None, :].expand(B, 4, NC))
        return x, ev, post


class EvalSet:
    def __init__(self, split, n, dev):
        d = D.load_split(split, {"val": 3000, "test": 5000}[split], with_post=True)
        self.occ = d["occ"][:n].astype(np.int64)
        self.post = d["post"][:n]
        self.ltrue = d["ltrue"][:n]
        ev = np.stack([D.evidence_states(d["pc"][:n], d["pr"][:n], s) for s in D.STAGES], 1)  # [n,4,25]
        self.ev = torch.tensor(ev, dtype=torch.long, device=dev)
        self.x = torch.tensor(self.occ, dtype=torch.long, device=dev)
        self.n = n
        rng = np.random.default_rng(12345)
        self.orders = torch.tensor(np.stack([np.stack([rng.permutation(NC) for _ in range(n * 4)]) for _ in range(8)]),
                                   dtype=torch.long, device=dev)  # [8, n*4, 25] fixed for all arms
        self.tie = torch.tensor(np.random.default_rng(777).random((n * 4, NC)) * 1e-3, dtype=torch.float32, device=dev)


# ----------------------------------------------------------------------------------------------- helpers
COORD = torch.tensor([(i // 5, i % 5) for i in range(NC)], dtype=torch.float32)
DIST = ((COORD[:, None] - COORD[None]) ** 2).sum(-1)  # [25,25]
PAIR_ID = torch.full((NC, NC), -1, dtype=torch.long)
for _i, (_a, _b) in enumerate(PAIRS.tolist()):
    PAIR_ID[_a, _b] = _i
    PAIR_ID[_b, _a] = _i


def bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y, reduction="none")


def positions(order):
    pos = torch.empty_like(order)
    pos.scatter_(1, order, torch.arange(NC, device=order.device).expand_as(order))
    return pos


def nearest_earlier(pos, noise):
    """Two nearest earlier cells (grid distance, tie-break noise) for every cell. Returns p1,p2 [N,25] (-1 if none)."""
    dist = DIST.to(pos.device)[None] + noise[:, None, :]          # [N,j,p]
    earlier = pos[:, None, :] < pos[:, :, None]                   # p earlier than j
    dd = torch.where(earlier, dist, torch.full_like(dist, 1e9))
    v, ix = dd.topk(2, dim=2, largest=False)
    p1 = torch.where(v[..., 0] < 1e8, ix[..., 0], torch.full_like(ix[..., 0], -1))
    p2 = torch.where(v[..., 1] < 1e8, ix[..., 1], torch.full_like(ix[..., 1], -1))
    return p1, p2


def bn_terms(net, h, x, order, cfg, ml, noise=None):
    """Per-cell training loss of the Bayes-net arm [N,25] (expected loss over soft parent choice)."""
    k = cfg["bn"]
    N = h.shape[0]
    dev = h.device
    pos = positions(order)
    xf = x.float()
    loss = torch.zeros(N, NC, device=dev)
    first = pos == 0
    loss = torch.where(first, bce(ml, xf), loss)
    T1 = net.k1_table(h)                                          # [N,j,p,3]
    l1 = torch.where(x[:, None, :].bool(), T1[..., 2], T1[..., 1])  # logit given parent's true value
    ce1 = bce(l1, xf[:, :, None].expand(N, NC, NC))
    cand1 = pos[:, None, :] < pos[:, :, None]                     # [N,j,p]
    if k == 1:
        rest = pos >= 1
        sc = T1[..., 0] / cfg["bn_tau"]
        sc = sc.masked_fill(~cand1, -1e9)
        w = torch.softmax(sc, dim=2)
        cnt = cand1.sum(2).clamp(min=1)
        lj = (w * ce1).sum(2) + cfg["bn_beta"] * (ce1 * cand1).sum(2) / cnt
        return torch.where(rest, lj, loss)
    # k = 2: cells at position 1 use the single earlier cell with the k1 table
    second = pos == 1
    o0 = order[:, 0]
    ce_second = ce1.gather(2, o0[:, None, None].expand(N, NC, 1)).squeeze(2)
    loss = torch.where(second, ce_second, loss)
    rest = pos >= 2
    T2 = net.k2_table(h)                                          # [N,j,300,5]
    P = net.pairs
    v = x[:, P[:, 0]] + 2 * x[:, P[:, 1]]                         # [N,300]
    l2 = T2[..., 1:].gather(3, v[:, None, :, None].expand(N, NC, P.shape[0], 1)).squeeze(3)
    ce2 = bce(l2, xf[:, :, None].expand(N, NC, P.shape[0]))
    if cfg["bn_fixed"]:
        p1, p2 = nearest_earlier(pos, noise)
        pid = PAIR_ID.to(dev)[p1.clamp(min=0), p2.clamp(min=0)]    # valid where pos>=2
        lj = ce2.gather(2, pid.clamp(min=0)[..., None]).squeeze(2)
        return torch.where(rest, lj, loss)
    pp = pos[:, P]                                                # [N,300,2]
    cand2 = (pp[:, None, :, 0] < pos[:, :, None]) & (pp[:, None, :, 1] < pos[:, :, None])
    sc = (T2[..., 0] / cfg["bn_tau"]).masked_fill(~cand2, -1e9)
    w = torch.softmax(sc, dim=2)
    cnt = cand2.sum(2).clamp(min=1)
    lj = (w * ce2).sum(2) + cfg["bn_beta"] * (ce2 * cand2).sum(2) / cnt
    return torch.where(rest, lj, loss)


def bn_harden(net, h, order, cfg, ml, noise):
    """Hardened net for evaluation: par [N,25,2], cpt [N,25,4] (probabilities)."""
    k = cfg["bn"]
    N = h.shape[0]
    dev = h.device
    pos = positions(order)
    ar = torch.arange(N, device=dev)
    pm = torch.sigmoid(ml)                                        # [N,25]
    par = torch.full((N, NC, 2), -1, dtype=torch.long, device=dev)
    cpt = pm[..., None].expand(N, NC, 4).clone()
    T1 = net.k1_table(h)
    cand1 = pos[:, None, :] < pos[:, :, None]
    if k == 1:
        sc = T1[..., 0].masked_fill(~cand1, -1e9)
        best = sc.argmax(2)                                       # [N,25]
        t = torch.sigmoid(T1.gather(2, best[..., None, None].expand(N, NC, 1, 3)).squeeze(2)[..., 1:])  # [N,25,2]
        has = pos >= 1
        par[..., 0] = torch.where(has, best, par[..., 0])
        cpt = torch.where(has[..., None], torch.cat([t, t], -1), cpt)
        return par, cpt
    o0 = order[:, 0]
    t1 = torch.sigmoid(T1[ar[:, None], torch.arange(NC, device=dev)[None], o0[:, None]][..., 1:])  # [N,25,2]
    second = pos == 1
    par[..., 0] = torch.where(second, o0[:, None].expand(N, NC), par[..., 0])
    cpt = torch.where(second[..., None], torch.cat([t1, t1], -1), cpt)
    T2 = net.k2_table(h)
    P = net.pairs
    rest = pos >= 2
    if cfg["bn_fixed"]:
        p1, p2 = nearest_earlier(pos, noise)
        pid = PAIR_ID.to(dev)[p1.clamp(min=0), p2.clamp(min=0)]
    else:
        pp = pos[:, P]
        cand2 = (pp[:, None, :, 0] < pos[:, :, None]) & (pp[:, None, :, 1] < pos[:, :, None])
        pid = T2[..., 0].masked_fill(~cand2, -1e9).argmax(2)
    t2 = torch.sigmoid(T2.gather(2, pid.clamp(min=0)[..., None, None].expand(N, NC, 1, 5)).squeeze(2)[..., 1:])
    par[..., 0] = torch.where(rest, P[pid.clamp(min=0), 0], par[..., 0])
    par[..., 1] = torch.where(rest, P[pid.clamp(min=0), 1], par[..., 1])
    cpt = torch.where(rest[..., None], t2, cpt)
    return par, cpt


def drift_penalty(q, nbins=10):
    """q [B,S,25] probabilities. Sum over stages of the bin-weighted squared mean drift E[q_{t+1}-q_t | bin(q_t)]."""
    pen = q.new_zeros(())
    B = q.shape[0]
    for t in range(q.shape[1] - 1):
        dq = (q[:, t + 1] - q[:, t]).reshape(-1)
        b = (q[:, t].detach().reshape(-1) * nbins).long().clamp(max=nbins - 1)
        s = torch.zeros(nbins, device=q.device).index_add(0, b, dq)
        c = torch.zeros(nbins, device=q.device).index_add(0, b, torch.ones_like(dq))
        pen = pen + ((s ** 2) / c.clamp(min=1)).sum() / B
    return pen


# ----------------------------------------------------------------------------------------------- training loss
def train_loss(net, cfg, batch, gen, teacher=None):
    x, ev, post = batch
    B, S, _ = ev.shape
    dev = x.device
    N = B * S
    evf = ev.reshape(N, NC)
    xr = x[:, None, :].expand(B, S, NC).reshape(N, NC)
    xf = xr.float()
    fb = net.flops_backbone()
    flops = 0.0
    terms = {}
    total = torch.zeros((), device=dev)
    need_plain = cfg["k0w"] > 0 or cfg["bn"] > 0 or cfg["td"] >= 0 or cfg["mart"] > 0 or cfg["rbtd"] > 0
    if need_plain:
        h = net.backbone(evf)
        ml = net.marg_logits(h)                                   # [N,25]
        flops += N * fb
        q = torch.sigmoid(ml).view(B, S, NC)
        if cfg["k0w"] > 0:
            if cfg["target"] == "post":
                y = post.reshape(N, NC)
                l0 = bce(ml, y).sum(1).mean()
            elif cfg["td"] >= 0 and cfg.get("_step", 0) >= cfg["td_warm"]:
                lam = cfg["td"]
                if teacher is not None:
                    with torch.no_grad():
                        qt = torch.sigmoid(teacher.marg_logits(teacher.backbone(evf))).view(B, S, NC)
                        flops += N * fb / 3
                else:
                    qt = q.detach()
                ys = [None] * S
                ys[S - 1] = x.float()
                for t in range(S - 2, -1, -1):
                    ys[t] = (1 - lam) * qt[:, t + 1] + lam * ys[t + 1]
                y = torch.stack(ys, 1).reshape(N, NC)
                l0 = bce(ml, y).sum(1).mean()
            else:
                l0 = bce(ml, xf).sum(1).mean()
            total = total + cfg["k0w"] * l0
            terms["k0"] = float(l0.detach())
        if cfg["mart"] > 0:
            pen = drift_penalty(q)
            total = total + cfg["mart"] * pen
            terms["mart"] = float(pen.detach())
        if cfg["bn"] > 0:
            order = torch.argsort(torch.rand(N, NC, device=dev, generator=gen), 1)
            noise = torch.rand(N, NC, device=dev, generator=gen) * 1e-3
            lbn = bn_terms(net, h, xr, order, cfg, ml, noise).sum(1).mean()
            total = total + cfg["bnw"] * lbn
            terms["bn"] = float(lbn.detach())
            flops += N * (net.flops_k1() + (net.flops_k2() if cfg["bn"] == 2 else 0))
        if cfg["rbtd"] > 0:
            m = cfg["rbtd"]
            lam = cfg["rbtd_lam"]
            lr_ = torch.zeros((), device=dev)
            for _r in range(cfg["rbtd_R"]):          # R independent reveal chains per board-stage (compute per sample)
                order = torch.argsort(torch.rand(N, NC, device=dev, generator=gen), 1)
                pos = positions(order)
                cuts = torch.sort(torch.argsort(torch.rand(N, NC - 1, device=dev, generator=gen), 1)[:, : m - 1] + 1, 1).values
                levels = [torch.zeros(N, dtype=torch.long, device=dev)] + [cuts[:, i] for i in range(m - 1)]
                logits = [ml]
                for i in range(1, m):
                    revd = pos < levels[i][:, None]
                    rev = torch.where(revd, 1 + xr, torch.zeros_like(xr))
                    logits.append(net.marg_logits(net.backbone(evf, rev)))
                    flops += N * fb
                y = xf
                for i in range(m - 1, -1, -1):
                    hidden = pos >= levels[i][:, None]
                    lr_ = lr_ + (bce(logits[i], y) * hidden).sum(1).mean()
                    if i > 0:
                        if teacher is not None:
                            with torch.no_grad():
                                revd_i = pos < levels[i][:, None]
                                rev_i = torch.where(revd_i, 1 + xr, torch.zeros_like(xr))
                                qi = torch.sigmoid(teacher.marg_logits(teacher.backbone(evf, rev_i)))
                                flops += N * fb / 3
                        else:
                            qi = torch.sigmoid(logits[i]).detach()
                        # target for level i-1: cells hidden at level i get (1-lam) q_i + lam y_i; cells revealed at level i get x
                        y = torch.where(hidden, (1 - lam) * qi + lam * y, xf)
            lr_ = lr_ / cfg["rbtd_R"]
            total = total + cfg["rbtd_w"] * lr_
            terms["rbtd"] = float(lr_.detach())
    if cfg["masked"] > 0:
        lm = torch.zeros((), device=dev)
        for _ in range(cfg["mask_R"]):
            r = torch.rand(N, 1, device=dev, generator=gen)
            revd = torch.rand(N, NC, device=dev, generator=gen) < r
            rev = torch.where(revd, 1 + xr, torch.zeros_like(xr))
            lg = net.marg_logits(net.backbone(evf, rev))
            lm = lm + (bce(lg, xf) * (~revd)).sum(1).mean()
            flops += N * fb
        lm = lm / cfg["mask_R"]
        total = total + cfg["masked"] * lm
        terms["masked"] = float(lm.detach())
    if cfg["chain"] > 0:
        lc = torch.zeros((), device=dev)
        for _ in range(cfg["chain_R"]):
            order = torch.argsort(torch.rand(N, NC, device=dev, generator=gen), 1)
            pos = positions(order)
            dcut = torch.randint(0, NC, (N, 1), device=dev, generator=gen)
            revd = pos < dcut
            rev = torch.where(revd, 1 + xr, torch.zeros_like(xr))
            lg = net.marg_logits(net.backbone(evf, rev))
            wgt = NC / (NC - dcut.float())
            lc = lc + ((bce(lg, xf) * (~revd)).sum(1, keepdim=True) * wgt).mean()
            flops += N * fb
        lc = lc / cfg["chain_R"]
        total = total + cfg["chain"] * lc
        terms["chain"] = float(lc.detach())
    if cfg["cell_q"] > 0:
        Q = cfg["cell_q"]
        if Q >= NC:
            qcells = torch.arange(NC, device=dev)[None].expand(N, NC)
        else:
            qcells = torch.argsort(torch.rand(N, NC, device=dev, generator=gen), 1)[:, :Q]
        Qn = qcells.shape[1]
        evq = evf[:, None, :].expand(N, Qn, NC).reshape(N * Qn, NC)
        qf = F.one_hot(qcells.reshape(-1), NC)
        lg = net.marg_logits(net.backbone(evq, None, qf))
        lq = lg.gather(1, qcells.reshape(-1, 1)).squeeze(1)
        yq = xf.gather(1, qcells).reshape(-1)
        lcq = bce(lq, yq).view(N, Qn).sum(1).mean() * (NC / Qn)
        total = total + lcq
        terms["cell"] = float(lcq.detach())
        flops += N * Qn * fb
    return total, terms, 3.0 * flops


# ----------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def readout(net, cfg, es: EvalSet, nb: int, M: int, joint=False, chunk=0):
    """Single-cell forecasts [nb,4,25] (+ joint log-lik of the true board [nb,4] when joint=True)."""
    was_training = net.training
    net.eval()
    dev = es.x.device
    ev = es.ev[:nb].reshape(nb * 4, NC)
    x = es.x[:nb][:, None, :].expand(nb, 4, NC).reshape(nb * 4, NC)
    out = np.zeros((nb * 4, NC))
    jl = np.zeros(nb * 4) if joint else None
    chunk = chunk or {"marg": 4096, "cell": 128, "bn": 256}[cfg["readout"]]
    for s in range(0, nb * 4, chunk):
        e = ev[s: s + chunk]
        n = e.shape[0]
        if cfg["readout"] == "cell":
            qcells = torch.arange(NC, device=dev)[None].expand(n, NC)
            evq = e[:, None, :].expand(n, NC, NC).reshape(n * NC, NC)
            qf = F.one_hot(qcells.reshape(-1), NC)
            lg = net.marg_logits(net.backbone(evq, None, qf)).view(n, NC, NC)
            p = torch.sigmoid(torch.diagonal(lg, dim1=1, dim2=2))
            out[s: s + n] = p.double().cpu().numpy()
            if joint:
                pc = p.clamp(1e-7, 1 - 1e-7)
                jl[s: s + n] = torch.where(x[s: s + n] > 0, pc.log(), (1 - pc).log()).sum(1).double().cpu().numpy()
            continue
        h = net.backbone(e)
        ml = net.marg_logits(h)
        if cfg["readout"] == "marg":
            p = torch.sigmoid(ml)
            out[s: s + n] = p.double().cpu().numpy()
            if joint:
                pc = p.clamp(1e-7, 1 - 1e-7)
                jl[s: s + n] = torch.where(x[s: s + n] > 0, pc.log(), (1 - pc).log()).sum(1).double().cpu().numpy()
            continue
        acc = np.zeros((n, NC))
        lls = []
        for mi in range(M):
            order = es.orders[mi, s: s + n]
            par, cpt = bn_harden(net, h, order, cfg, ml, es.tie[s: s + n])
            if cfg["bn"] == 1:
                mg = BI.k1_marginals(order, par[..., 0], cpt[..., :2].double()).cpu().numpy()
            else:
                mg = BI.k2_marginals(order, par, cpt)
            acc += mg
            if joint:
                if cfg["bn"] == 1:
                    lls.append(BI.bn_loglik(x[s: s + n], par[..., 0], cpt[..., :2], 1).double().cpu().numpy())
                else:
                    lls.append(BI.bn_loglik(x[s: s + n], par, cpt, 2).double().cpu().numpy())
        out[s: s + n] = acc / M
        if joint:
            L = np.stack(lls)                                     # [M,n]: mixture of the M nets
            jl[s: s + n] = np.logaddexp.reduce(L, 0) - math.log(M)
    net.train(was_training)
    out = out.reshape(nb, 4, NC)
    return out, (jl.reshape(nb, 4) if joint else None)


@torch.no_grad()
def ar_joint(net, es: EvalSet, nb: int, chunk=1024):
    """Autoregressive log-lik of the true board for arms trained with revealed true cells (one random order)."""
    was_training = net.training
    net.eval()
    dev = es.x.device
    ev = es.ev[:nb].reshape(nb * 4, NC)
    x = es.x[:nb][:, None, :].expand(nb, 4, NC).reshape(nb * 4, NC)
    out = np.zeros(nb * 4)
    for s in range(0, nb * 4, chunk):
        e, xx = ev[s: s + chunk], x[s: s + chunk]
        n = e.shape[0]
        order = es.orders[0, s: s + n]
        pos = positions(order)
        ll = torch.zeros(n, device=dev)
        for t in range(NC):
            revd = pos < t
            rev = torch.where(revd, 1 + xx, torch.zeros_like(xx))
            p = torch.sigmoid(net.marg_logits(net.backbone(e, rev))).clamp(1e-7, 1 - 1e-7)
            c = order[:, t]
            pc = p.gather(1, c[:, None]).squeeze(1)
            xc = xx.gather(1, c[:, None]).squeeze(1)
            ll += torch.where(xc > 0, pc.log(), (1 - pc).log())
        out[s: s + n] = ll.double().cpu().numpy()
    net.train(was_training)
    return out.reshape(nb, 4)


def kl_bern(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def ece(p, y, bins=15):
    p, y = np.asarray(p, float).ravel(), np.asarray(y, float).ravel()
    o = np.argsort(p, kind="stable")
    tot = 0.0
    for ch in np.array_split(o, bins):
        tot += len(ch) * abs(p[ch].mean() - y[ch].mean())
    return tot / len(p)


def metrics(q, es: EvalSet, nb, jl=None, ar=None):
    post = es.post[:nb]
    y = es.occ[:nb][:, None, :].astype(float)
    kl = kl_bern(post, q)                                         # [nb,4,25]
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(qc) + (1 - y) * np.log(1 - qc))             # outcome log-loss per cell
    m = {
        "kl": float(kl.mean()), "kl_stage": kl.mean((0, 2)).tolist(),
        "logloss": float(ll.mean()), "logloss_stage": ll.mean((0, 2)).tolist(),
        "ece": float(ece(q, np.broadcast_to(y, q.shape))),
        "exact_logloss": float((-(y * np.log(np.clip(post, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - post, 1e-12, 1)))).mean()),
        "nb": nb,
    }
    if jl is not None:
        m["joint_ll"] = float(jl.mean())
        m["joint_ll_stage"] = jl.mean(0).tolist()
    if ar is not None:
        m["joint_ll_ar"] = float(ar.mean())
        m["joint_ll_ar_stage"] = ar.mean(0).tolist()
    m["joint_ll_exact"] = float(es.ltrue[:nb].mean())
    pc = np.clip(post, 1e-12, 1 - 1e-12)
    m["joint_ll_exact_indep"] = float((y * np.log(pc) + (1 - y) * np.log(1 - pc)).sum(2).mean())
    return m


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true", help="also evaluate on TEST (final champion runs only)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-n", type=int, default=3000)
    ap.add_argument("--ar-n", type=int, default=500, help="boards for the autoregressive joint log-lik (reveal arms)")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--tag", default="", help="experiment label stored in the result")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(BASE)
    cfg.update(ARMS[a.arm])
    ms, ee, lr_n, ema_n = steps_for(a.n)
    cfg["lr"], cfg["ema"] = lr_n, ema_n                    # per-N defaults; --set overrides
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in BASE, f"unknown key {k}"
        cfg[k] = type(BASE[k])(v) if not isinstance(BASE[k], str) else v
    cfg["max_steps"] = cfg["max_steps"] or ms
    cfg["eval_every"] = cfg["eval_every"] or ee
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else (a.device if a.device != "auto" else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(2000 + a.seed)
    net = Net(cfg["d"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["drop"], cfg["m"]).to(dev)
    ema = copy.deepcopy(net) if cfg["ema"] > 0 else None
    if ema is not None:
        ema.eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    tr = TrainSet(a.n, a.seed, dev, need_post=(cfg["target"] == "post"))
    va = EvalSet("val", a.val_n, dev)
    es_n = min(cfg["es_boards"], a.val_n)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    hist, best, best_state, best_step, bad = [], float("inf"), None, 0, 0
    flops, flops_best = 0.0, 0.0
    t0 = time.time()
    t_eval = 0.0
    step = 0
    stop_reason = "max_steps"
    # SMALLNET_DEADLINE (unix time, set by gpu/remote.sh): stop training early enough to evaluate before the VM ends
    deadline = float(os.environ.get("SMALLNET_DEADLINE", "0") or 0) - (600 if cfg["readout"] == "bn" else 300)
    if deadline < 0:
        deadline = 0
    for step in range(1, cfg["max_steps"] + 1):
        cfg["_step"] = step
        batch = tr.batch(cfg["B"], cfg["stages"], gen)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(cfg["amp"]) and dev.type == "cuda"):
            loss, terms, fl = train_loss(net, cfg, batch, gen, teacher=(ema if (cfg["teacher"] and ema is not None) else None))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        flops += fl
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(cfg["ema"]).add_(pn.detach(), alpha=1 - cfg["ema"])
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            te = time.time()
            model = ema if ema is not None else net
            q, _ = readout(model, cfg, va, es_n, M=1)
            mm = metrics(q, va, es_n)
            t_eval += time.time() - te
            rec = {"step": step, "val_logloss": mm["logloss"], "val_kl": mm["kl"], "loss": float(loss.detach()), **terms,
                   "flops": flops, "sec": time.time() - t0}
            hist.append(rec)
            if mm["logloss"] < best - 1e-6:
                best, best_step, bad, flops_best = mm["logloss"], step, 0, flops
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            if bad >= cfg["patience"]:
                stop_reason = "patience"
                break
            if a.max_minutes and (time.time() - t0) > 60 * a.max_minutes:
                stop_reason = "max_minutes"
                print("max-minutes reached", flush=True)
                break
            if deadline and time.time() > deadline:
                stop_reason = "deadline"
                print("rental deadline reached: stopping training, evaluating the best checkpoint", flush=True)
                break
    train_sec = time.time() - t0
    model = ema if ema is not None else net
    model.load_state_dict(best_state)
    res = {"tag": a.tag or a.arm, "arm": a.arm, "n": a.n, "seed": a.seed, "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "set": a.set, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "steps_run": step, "best_step": best_step, "stop_reason": stop_reason, "train_sec": train_sec, "eval_sec_during_training": t_eval,
           "flops_total": flops, "flops_at_best": flops_best, "flops_per_board_at_best": flops_best / a.n,
           "params": sum(p.numel() for p in model.parameters()), "history": hist,
           "real_world_valid": cfg["target"] != "post"}
    t1 = time.time()
    M = cfg["eval_M"] if cfg["readout"] == "bn" else 1
    q, jl = readout(model, cfg, va, a.val_n, M=M, joint=True)
    ar = ar_joint(model, va, min(a.ar_n, a.val_n)) if (cfg["masked"] > 0 or cfg["chain"] > 0 or cfg["rbtd"] > 0) else None
    res["val"] = metrics(q, va, a.val_n, jl, ar)
    if cfg["readout"] == "bn":   # the single-cell head of the same net, as a secondary readout
        c2 = dict(cfg, readout="marg")
        q2, _ = readout(model, c2, va, a.val_n, M=1)
        res["val_marghead"] = metrics(q2, va, a.val_n)
        if M > 1:
            q1, _ = readout(model, cfg, va, a.val_n, M=1)
            res["val_bn_M1"] = metrics(q1, va, a.val_n)
    if a.test:
        ts = EvalSet("test", 5000, dev)
        q, jl = readout(model, cfg, ts, 5000, M=M, joint=True)
        ar = ar_joint(model, ts, min(a.ar_n, 5000)) if (cfg["masked"] > 0 or cfg["chain"] > 0 or cfg["rbtd"] > 0) else None
        res["test"] = metrics(q, ts, 5000, jl, ar)
        res["test_kl_per_board"] = kl_bern(ts.post, q).mean((1, 2)).tolist()
    res["final_eval_sec"] = time.time() - t1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, a.out)
    print("DONE", a.arm, a.n, a.seed, "val_kl", round(res["val"]["kl"], 5), "best_step", best_step, "train_sec", round(train_sec, 1), flush=True)


if __name__ == "__main__":
    main()
