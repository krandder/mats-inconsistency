"""wm2: learn an L0 world model (WM) from N ShipHunt episodes and train the forecaster on its dreams (eurisko, 2026-10-11).

  python3 job.py --task t1 wm --n 300 --seed 0 --wmtag WMc [--set k=v ...] --ckpt CKPT.pt --out WM.json
      Generic joint WM: round-1 cell-token transformer + probe-order embedding + per-cell location and reading heads.
      p(episode) = p(board, any-order autoregressive over cells) * prod_t p(cell_t, reading_t | board, earlier probes).
      Trained by maximum likelihood on the N real episodes (8 symmetries as augmentation); early stopping on the
      validation episodes' NLL (outcomes only). Saves the best (EMA) weights and diagnostics (evaluation-only parts
      are marked).
  python3 job.py --task t1 fc --n 300 --seed 0 --arm D-only --tag TAG --wm-ckpt CKPT.pt [--set ...] --out RUN.json
      Forecaster trained on dreams from the frozen WM (+ real episodes); early stopping on validation outcomes;
      same protocol, readout and JSON format as round-1 smallnet.py (whose train_loss / readout / metrics it calls).
Nothing here reads the exact posterior except the evaluation code (metrics, diagnostics flagged EVAL-ONLY).
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
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SMALLNET = os.environ.get("WM2_SMALLNET", os.path.join(os.path.dirname(HERE), "smallnet"))
if SMALLNET not in sys.path:
    sys.path.insert(0, SMALLNET)
import data as D  # noqa: E402  (round 1, unchanged; DATA_DIR from SMALLNET_DATA)
import smallnet as SN  # noqa: E402  (round 1, unchanged)
from netdef import NC, Net  # noqa: E402

# ----------------------------------------------------------------------------------------------- config
WM_BASE = dict(
    d=128, layers=4, heads=4, ff=256, m=64, drop=0.1,
    lr=1e-3, wd=0.05, B=64, warmup=100, clip=1.0, ema=0.0,
    max_steps=40000, eval_every=0, patience=25, es_boards=3000,
    wm_board="chain",   # 'chain' (any-order MLE) | 'rbtd' (reveal-chain TD on the board conditionals)
    wm_R=1,             # board passes per step
    wm_rbtd=8,          # levels when wm_board=rbtd
    loc_blind=0,        # 1 = location head does not see the board (NOT L0 in T1: hand-codes independence)
    wm_ev_w=1.0,        # weight of the evidence (location + reading) terms
    wm_board_w=1.0,     # weight of the board terms (0 = evidence-only network, used by wm_split)
    wm_split=0,         # 1 = two networks: board model and probe model, each early-stopped on its own validation NLL
    diag_post_n=0,      # validation episodes for the valid-board WM posterior diagnostic (EVAL-ONLY; 13 min per WM in the smoke); 0 = off
    init_seed=0,        # extra init / batch-order seed (ensembles of WMs on the same training boards)
    wm_twin=0,          # 1 (with wm_split): one board network, two probe networks (L0 and board-blind); writes CKPT and CKPT.blind.pt
)

FC_BASE = dict(SN.BASE)
FC_BASE.update(max_steps=40000, es_boards=3000, patience=25,
               rho=1.0,          # fraction of each batch that is dreamed (rest real)
               dream_buf=4096,   # dreams generated per refill
               dream_reuse=4,    # each dream is drawn about this many times (with a fresh symmetry) before the refill
               shared=0,         # 1 = the forecaster IS the WM network (init from it, keeps the WM loss on real data)
               shared_w=1.0,     # weight of the WM loss in the shared arm
               post_target=0,    # 1 = arm 4: target = WM's own posterior (importance sampling) -- FLAGGED
               is_M=8192, is_K=1024,   # arm 4: dream pool size, prior samples per estimate
               init_wm=0,        # 1 = separate forecaster initialised from the WM weights (no WM loss)
               fc_arm="k0")      # forecaster objective: 'k0' or 'rbtd' (with rbtd levels from SN.BASE['rbtd'])

ARMS = {   # forecaster arms on a WM (name -> FC_BASE overrides)
    "D-only": dict(rho=1.0),
    "D-mix50": dict(rho=0.5),
    "D-mix88": dict(rho=0.875),
    "D-rbtd8": dict(rho=0.5, fc_arm="rbtd", rbtd=8, k0w=0.0),
    "D-rbtd8-only": dict(rho=1.0, fc_arm="rbtd", rbtd=8, k0w=0.0),
    "S-only": dict(rho=1.0, shared=1),
    "S-mix50": dict(rho=0.5, shared=1),
    "D-init": dict(rho=1.0, init_wm=1),
    "D-post": dict(rho=1.0, post_target=1, target="post"),   # arm 4, flagged
    "R-k0": dict(rho=0.0),   # harness check: real data only through this loop = k0
}


def steps_for(n):
    return SN.steps_for(n)   # round-1 per-N (max_steps, eval_every, lr, ema)


# ----------------------------------------------------------------------------------------------- the world model
class WMNet(Net):
    """Round-1 Net + probe-order embedding + per-cell next-probe-location and reading heads."""

    def __init__(self, d=128, layers=4, heads=4, ff=256, drop=0.1, m=64):
        super().__init__(d, layers, heads, ff, drop, m)   # base modules first: same init as Net for the same seed
        self.ordemb = nn.Embedding(8, d)
        nn.init.normal_(self.ordemb.weight, std=0.02)
        self.loc = nn.Linear(d, 1)
        self.read = nn.Linear(d, 1)

    def backbone(self, ev, rev=None, q=None, ordi=None):
        x = self.pos[None] + self.ev(ev)
        x = x + (self.rev(rev) if rev is not None else self.rev.weight[0])
        x = x + (self.qf(q) if q is not None else self.qf.weight[0])
        x = x + (self.ordemb(ordi) if ordi is not None else self.ordemb.weight[0])
        for li, blk in enumerate(self.blocks):
            x = blk(x, self.relb[li][:, self.relidx])
        return self.lnf(x)


def evidence_steps(pc, pr):
    """pc, pr [m,6] -> ev, ordi [m,7,25]: per-cell evidence state / probe index after 0..6 probes."""
    m = pc.shape[0]
    dev = pc.device
    ev = torch.zeros(m, 7, NC, dtype=torch.long, device=dev)
    od = torch.zeros(m, 7, NC, dtype=torch.long, device=dev)
    ar = torch.arange(m, device=dev)
    for t in range(1, 7):
        ev[:, t] = ev[:, t - 1]
        od[:, t] = od[:, t - 1]
        ev[ar, t, pc[:, t - 1]] = 1 + pr[:, t - 1].long()
        od[ar, t, pc[:, t - 1]] = t
    return ev, od


def evidence_logp(wm, x, pc, pr, loc_blind=False, mask_probed=False, want_repeat=False):
    """log p(cell_t) and log p(reading_t) under the WM for each probe step: [m,6] each (+ mass on probed cells)."""
    m = x.shape[0]
    ev, od = evidence_steps(pc, pr)
    e6 = ev[:, :6].reshape(m * 6, NC)
    o6 = od[:, :6].reshape(m * 6, NC)
    rev = (1 + x.long())[:, None, :].expand(m, 6, NC).reshape(m * 6, NC)
    h = wm.backbone(e6, rev, None, o6)
    hl = wm.backbone(e6, None, None, o6) if loc_blind else h
    ll = wm.loc(hl).squeeze(-1)
    rep = None
    if want_repeat:
        rep = (torch.softmax(ll, -1) * (e6 > 0)).sum(1).view(m, 6)
    if mask_probed:
        ll = ll.masked_fill(e6 > 0, float("-inf"))
    lp = torch.log_softmax(ll, -1).gather(1, pc.reshape(-1, 1)).view(m, 6)
    rl = wm.read(h).squeeze(-1).gather(1, pc.reshape(-1, 1)).squeeze(1)
    lr = -SN.bce(rl, pr.reshape(-1).float()).view(m, 6)
    return lp, lr, rep


def board_nll(wm, x, order):
    """Autoregressive NLL of boards x [m,25] in the given cell order (no evidence): [m]."""
    pos = SN.positions(order)
    ev0 = torch.zeros_like(x)
    nll = torch.zeros(x.shape[0], device=x.device)
    for t in range(NC):
        rev = torch.where(pos < t, 1 + x, torch.zeros_like(x))
        p = torch.sigmoid(wm.marg_logits(wm.backbone(ev0, rev))).clamp(1e-7, 1 - 1e-7)
        c = order[:, t:t + 1]
        pcl = p.gather(1, c).squeeze(1)
        xc = x.gather(1, c).squeeze(1)
        nll -= torch.where(xc > 0, pcl.log(), (1 - pcl).log())
    return nll


class EpisodeSet:
    """Real training episodes (board + probe trajectory) for the WM, with the round-1 symmetry augmentation."""

    def __init__(self, n, seed, dev):
        d = D.load_split("train", max(10_000, n), seed=seed)
        self.occ = torch.tensor(d["occ"][:n], dtype=torch.long, device=dev)
        self.pc = torch.tensor(d["pc"][:n], dtype=torch.long, device=dev)
        self.pr = torch.tensor(d["pr"][:n], dtype=torch.long, device=dev)
        self.n = n
        self.sym = torch.tensor(D.SYM, dtype=torch.long, device=dev)
        self.isym = torch.argsort(self.sym, 1)
        self.dev = dev

    def batch(self, B, gen):
        idx = torch.randint(0, self.n, (B,), device=self.dev, generator=gen)
        g = torch.randint(0, 8, (B,), device=self.dev, generator=gen)
        x = self.occ[idx].gather(1, self.sym[g])                 # new[i] = old[perm[i]]
        pc = self.isym[g].gather(1, self.pc[idx])                 # old cell c -> new index inv[c]
        return x, pc, self.pr[idx]


def wm_loss(wm, cfg, x, pc, pr, gen):
    B = x.shape[0]
    dev = x.device
    fb = wm.flops_backbone()
    flops = 0.0
    terms = {}
    xf = x.float()
    ev0 = torch.zeros_like(x)
    lb = torch.zeros((), device=dev)
    if cfg.get("wm_board_w", 1.0) == 0:
        pass
    elif cfg["wm_board"] == "chain":
        lb = torch.zeros((), device=dev)
        for _ in range(cfg["wm_R"]):
            order = torch.argsort(torch.rand(B, NC, device=dev, generator=gen), 1)
            pos = SN.positions(order)
            dcut = torch.randint(0, NC, (B, 1), device=dev, generator=gen)
            revd = pos < dcut
            rev = torch.where(revd, 1 + x, torch.zeros_like(x))
            lg = wm.marg_logits(wm.backbone(ev0, rev))
            wgt = NC / (NC - dcut.float())
            lb = lb + ((SN.bce(lg, xf) * (~revd)).sum(1, keepdim=True) * wgt).mean()
            flops += B * fb
        lb = lb / cfg["wm_R"]
    else:   # reveal-chain TD on the board conditionals (round-1 rbtd at the empty-evidence stage)
        c2 = dict(SN.BASE, k0w=0.0, rbtd=cfg["wm_rbtd"], rbtd_R=cfg["wm_R"])
        lb, _t, fl = SN.train_loss(wm, c2, (x, ev0[:, None, :], None), gen)
        flops += fl / 3.0
    terms["board"] = float(lb.detach())
    total = cfg.get("wm_board_w", 1.0) * lb
    if cfg["wm_ev_w"] > 0:
        lp, lr, _ = evidence_logp(wm, x, pc, pr, loc_blind=bool(cfg["loc_blind"]))
        flops += B * 6 * fb * (2 if cfg["loc_blind"] else 1)
        lloc, lread = -lp.sum(1).mean(), -lr.sum(1).mean()
        terms["loc"], terms["read"] = float(lloc.detach()), float(lread.detach())
        total = total + cfg["wm_ev_w"] * (lloc + lread)
    return total, terms, 3.0 * flops


class WMVal:
    """Validation (or test) episodes for the WM's early stopping: NLL of the realised episodes (outcomes only)."""

    def __init__(self, split, n, dev):
        d = D.load_split(split, {"val": 3000, "test": 5000}[split], with_post=True)
        self.x = torch.tensor(d["occ"][:n], dtype=torch.long, device=dev)
        self.pc = torch.tensor(d["pc"][:n], dtype=torch.long, device=dev)
        self.pr = torch.tensor(d["pr"][:n], dtype=torch.long, device=dev)
        rng = np.random.default_rng(4242)
        self.order = torch.tensor(np.stack([rng.permutation(NC) for _ in range(n)]), dtype=torch.long, device=dev)
        self.other = torch.tensor((np.arange(n) + 1) % n, dtype=torch.long, device=dev)
        self.post = d["post"][:n]          # EVAL-ONLY (diagnostics)
        self.n = n


@torch.no_grad()
def wm_eval(wm, cfg, V: WMVal, nb, chunk=2048, diag=False, we=None, parts=("board", "ev")):
    """Validation NLL per episode (board / probe cells / readings). we = separate probe-model net (wm_split)."""
    we = wm if we is None else we
    was, wase = wm.training, we.training
    wm.eval()
    we.eval()
    tot = {"board": 0.0, "loc": 0.0, "read": 0.0}
    extra = {"loc_info": 0.0, "repeat_mass": 0.0}
    for s in range(0, nb, chunk):
        sl = slice(s, min(nb, s + chunk))
        x, pc, pr = V.x[sl], V.pc[sl], V.pr[sl]
        if "board" in parts:
            tot["board"] += float(board_nll(wm, x, V.order[sl]).sum())
        if "ev" in parts:
            lp, lr, rep = evidence_logp(we, x, pc, pr, loc_blind=bool(cfg["loc_blind"]), want_repeat=diag)
            tot["loc"] -= float(lp.sum())
            tot["read"] -= float(lr.sum())
            if diag:
                xo = V.x[V.other[sl]]
                lpo, _, _ = evidence_logp(we, xo, pc, pr, loc_blind=bool(cfg["loc_blind"]))
                extra["loc_info"] += float((lp - lpo).sum())
                extra["repeat_mass"] += float(rep.sum())
    out = {k: v / nb for k, v in tot.items()}
    out["total"] = out["board"] + out["loc"] + out["read"]
    if diag:
        out["loc_info_nats_per_episode"] = extra["loc_info"] / nb
        out["repeat_mass_per_step"] = extra["repeat_mass"] / (nb * 6)
    wm.train(was)
    we.train(wase)
    return out


# ----------------------------------------------------------------------------------------------- dreams
@torch.no_grad()
def dream(wm, M, gen, loc_blind=False, chunk=4096, we=None):
    """Sample M episodes from the WM: board cell by cell in a random order, then 6 probes (cell among unprobed cells,
    reading). Returns occ, pc, pr (long) and stats (repeat mass of the location head before masking).
    we = separate probe-model net (wm_split); default: the same net."""
    we = wm if we is None else we
    was, wase = wm.training, we.training
    wm.eval()
    we.eval()
    dev = next(wm.parameters()).device
    occs, pcs, prs = [], [], []
    rep_tot = 0.0
    for s in range(0, M, chunk):
        m = min(chunk, M - s)
        order = torch.argsort(torch.rand(m, NC, device=dev, generator=gen), 1)
        x = torch.zeros(m, NC, dtype=torch.long, device=dev)
        rev = torch.zeros_like(x)
        ev0 = torch.zeros_like(x)
        for t in range(NC):
            p = torch.sigmoid(wm.marg_logits(wm.backbone(ev0, rev)).float())
            c = order[:, t:t + 1]
            v = (torch.rand(m, 1, device=dev, generator=gen) < p.gather(1, c)).long()
            x.scatter_(1, c, v)
            rev.scatter_(1, c, 1 + v)
        ev = torch.zeros_like(x)
        od = torch.zeros_like(x)
        pc = torch.zeros(m, 6, dtype=torch.long, device=dev)
        pr = torch.zeros(m, 6, dtype=torch.long, device=dev)
        for t in range(6):
            h = we.backbone(ev, 1 + x, None, od)
            hl = we.backbone(ev, None, None, od) if loc_blind else h
            ll = we.loc(hl).squeeze(-1).float()
            rep_tot += float((torch.softmax(ll, -1) * (ev > 0)).sum())
            ll = ll.masked_fill(ev > 0, float("-inf"))
            c = torch.multinomial(torch.softmax(ll, -1), 1, generator=gen)
            r = (torch.rand(m, 1, device=dev, generator=gen) < torch.sigmoid(we.read(h).squeeze(-1).float().gather(1, c))).long()
            ev.scatter_(1, c, 1 + r)
            od.scatter_(1, c, t + 1)
            pc[:, t:t + 1] = c
            pr[:, t:t + 1] = r
        occs.append(x)
        pcs.append(pc)
        prs.append(pr)
    wm.train(was)
    we.train(wase)
    passes = NC + 6 * (2 if loc_blind else 1)
    return torch.cat(occs), torch.cat(pcs), torch.cat(prs), {"repeat_mass": rep_tot / (M * 6), "fwd_passes_per_dream": passes}


def dream_fast(wm, M, gen, loc_blind=False, chunk=4096, we=None):
    """dream() under bf16 autocast on CUDA (sampling only; probabilities are computed in fp32 after the autocast region)."""
    dev = next(wm.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        return dream(wm, M, gen, loc_blind=loc_blind, chunk=chunk, we=we)


@torch.no_grad()
def evidence_cum(wm, boards, pc, pr, loc_blind=False, max_seq=8192):
    """WM log-likelihood of each episode's evidence prefix under each board: [E, K, 7] (stage 0..6), probed-cell masked.
    boards [K,25]; pc, pr [E,6]."""
    was = wm.training
    wm.eval()
    K = boards.shape[0]
    E = pc.shape[0]
    dev = boards.device
    ev, od = evidence_steps(pc, pr)                      # [E,7,25]
    out = torch.zeros(E, K, 7, device=dev)
    ce = max(1, max_seq // K)
    revb = 1 + boards.long()
    for s in range(0, E, ce):
        e1 = min(E, s + ce)
        n = e1 - s
        acc = torch.zeros(n, K, device=dev)
        for t in range(6):
            evt = ev[s:e1, t][:, None, :].expand(n, K, NC).reshape(n * K, NC)
            odt = od[s:e1, t][:, None, :].expand(n, K, NC).reshape(n * K, NC)
            rv = revb[None].expand(n, K, NC).reshape(n * K, NC)
            h = wm.backbone(evt, rv, None, odt)
            hl = wm.backbone(evt, None, None, odt) if loc_blind else h
            ll = wm.loc(hl).squeeze(-1).float().masked_fill(evt > 0, float("-inf"))
            c = pc[s:e1, t][:, None].expand(n, K).reshape(-1, 1)
            lp = torch.log_softmax(ll, -1).gather(1, c).squeeze(1)
            rl = wm.read(h).squeeze(-1).float().gather(1, c).squeeze(1)
            r = pr[s:e1, t][:, None].expand(n, K).reshape(-1).float()
            acc = acc + (lp - SN.bce(rl, r)).view(n, K)
            out[s:e1, :, t + 1] = acc
    wm.train(was)
    return out


def is_marginals(cum, boards, stages=(0, 2, 4, 6)):
    """Self-normalised importance-sampling marginals [E,len(stages),25] and ESS [E,len(stages)]."""
    bf = boards.float()
    margs, ess = [], []
    for st in stages:
        w = torch.softmax(cum[:, :, st], 1)              # [E,K]
        margs.append(w @ bf)
        ess.append(1.0 / (w ** 2).sum(1))
    return torch.stack(margs, 1), torch.stack(ess, 1)


# ----------------------------------------------------------------------------------------------- diagnostics (EVAL-ONLY parts flagged)
_VALID = None


def valid_codes():
    global _VALID
    if _VALID is None:
        P = D.placements(D.CFG).astype(np.int64)
        _VALID = set((P * (1 << np.arange(NC))).sum(1).tolist())
    return _VALID


def dream_stats(occ):
    """EVAL-ONLY: fraction of dreamed boards that are valid ShipHunt placements; occupied-count distribution."""
    o = occ.cpu().numpy().astype(np.int64)
    codes = (o * (1 << np.arange(NC))).sum(1)
    vs = valid_codes()
    k = o.sum(1)
    return {"valid_frac": float(np.mean([c in vs for c in codes.tolist()])), "occ_mean": float(k.mean()),
            "occ7_frac": float((k == 7).mean()), "distinct_frac": float(len(set(codes.tolist())) / len(codes))}


@torch.no_grad()
def wm_post_diag(wm, cfg, V: WMVal, nE, gen, we=None):
    """EVAL-ONLY diagnostic: the WM's posterior restricted to the 11088 valid placements (prior from the WM's board
    model in one fixed random order per placement, likelihood from its evidence model) on nE validation episodes;
    single-cell KL to the exact posterior per stage. Uses the placement list, so it is never used for any decision."""
    dev = V.x.device
    P = torch.tensor(D.placements(D.CFG).astype(np.int64), device=dev)
    rng = np.random.default_rng(99)
    order = torch.tensor(np.stack([rng.permutation(NC) for _ in range(len(P))]), dtype=torch.long, device=dev)
    lprior = torch.cat([-board_nll(wm, P[s:s + 4096], order[s:s + 4096]) for s in range(0, len(P), 4096)])
    cum = evidence_cum(wm if we is None else we, P, V.pc[:nE], V.pr[:nE], loc_blind=bool(cfg["loc_blind"]))   # [nE,11088,7]
    cum = cum + lprior[None, :, None]
    marg, _ess = is_marginals(cum, P)
    q = marg.double().cpu().numpy()
    kl = SN.kl_bern(V.post[:nE], q)
    return {"wm_post_valid_kl": float(kl.mean()), "wm_post_valid_kl_stage": kl.mean((0, 2)).tolist(), "n": nE,
            "valid_mass_logsumexp": float(torch.logsumexp(lprior, 0))}


# ----------------------------------------------------------------------------------------------- generic training loop
def deadline_from_env(margin):
    d = float(os.environ.get("SMALLNET_DEADLINE", "0") or 0)
    return max(0.0, d - margin) if d else 0.0


def train_loop(net, cfg, step_fn, eval_fn, dev, max_minutes=0.0, margin=300, log_prefix=""):
    """Mirror of smallnet.main's loop: AdamW, warmup, grad clip, weight EMA (evaluated), early stopping on eval_fn
    (lower = better) every eval_every steps with patience, best checkpoint kept, rental deadline respected."""
    ema = copy.deepcopy(net) if cfg["ema"] > 0 else None
    if ema is not None:
        ema.eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]))
    hist, best, best_state, best_step, bad = [], float("inf"), None, 0, 0
    flops, flops_best = 0.0, 0.0
    t0 = time.time()
    t_eval = 0.0
    step = 0
    stop_reason = "max_steps"
    deadline = deadline_from_env(margin)
    for step in range(1, cfg["max_steps"] + 1):
        cfg["_step"] = step
        loss, terms, fl = step_fn(net, step, ema)
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
            score, rec = eval_fn(model)
            t_eval += time.time() - te
            rec = {"step": step, **rec, "loss": float(loss.detach()), **terms, "flops": flops, "sec": time.time() - t0}
            hist.append(rec)
            if score < best - 1e-6:
                best, best_step, bad, flops_best = score, step, 0, flops
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            print(log_prefix + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            if bad >= cfg["patience"]:
                stop_reason = "patience"
                break
            if max_minutes and (time.time() - t0) > 60 * max_minutes:
                stop_reason = "max_minutes"
                break
            if deadline and time.time() > deadline:
                stop_reason = "deadline"
                print("rental deadline reached: stopping training, evaluating the best checkpoint", flush=True)
                break
    model = ema if ema is not None else net
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"steps_run": step, "best_step": best_step, "stop_reason": stop_reason, "train_sec": time.time() - t0,
                   "eval_sec_during_training": t_eval, "flops_total": flops, "flops_at_best": flops_best, "history": hist,
                   "best_score": best}


def apply_sets(cfg, sets, base):
    for kv in sets:
        k, v = kv.split("=", 1)
        assert k in base, f"unknown key {k}"
        cfg[k] = type(base[k])(v) if not isinstance(base[k], str) else v
    return cfg


def device_from(a):
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else (a.device if a.device != "auto" else "cpu"))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


def save_json(res, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out)


# ----------------------------------------------------------------------------------------------- mode: wm
def main_wm(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wmtag", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-n", type=int, default=3000)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--task", default=os.environ.get("WM2_TASK", "?"))
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(WM_BASE)
    ms, ee, lr_n, ema_n = steps_for(a.n)
    cfg["lr"], cfg["ema"], cfg["eval_every"] = lr_n, ema_n, ee
    cfg = apply_sets(cfg, a.set, WM_BASE)
    dev = device_from(a)
    torch.manual_seed(3000 + a.seed + 100 * cfg["init_seed"])
    gen = torch.Generator(device=dev)
    gen.manual_seed(4000 + a.seed + 100 * cfg["init_seed"])
    ep = EpisodeSet(a.n, a.seed, dev)
    V = WMVal("val", a.val_n, dev)
    es_n = min(cfg["es_boards"], a.val_n)

    def fit(c, parts, tag):
        net = WMNet(c["d"], c["layers"], c["heads"], c["ff"], c["drop"], c["m"]).to(dev)

        def step_fn(net_, step, ema):
            x, pc, pr = ep.batch(c["B"], gen)
            return wm_loss(net_, c, x, pc, pr, gen)

        def eval_fn(model):
            r = wm_eval(model, c, V, es_n, parts=parts)
            sc = (r["board"] if "board" in parts else 0.0) + ((r["loc"] + r["read"]) if "ev" in parts else 0.0)
            return sc, {"val_nll": sc, "val_board": r["board"], "val_loc": r["loc"], "val_read": r["read"]}

        return train_loop(net, c, step_fn, eval_fn, dev, a.max_minutes, margin=600, log_prefix=tag)

    twin = None
    if cfg["wm_split"]:   # two networks, each early-stopped on its own part of the validation NLL
        cb = dict(cfg, wm_ev_w=0.0)
        ce = dict(cfg, wm_board_w=0.0)
        model, info_b = fit(cb, ("board",), "WMB ")
        torch.manual_seed(3500 + a.seed)
        model_e, info_e = fit(ce, ("ev",), "WME ")
        if cfg["wm_twin"]:   # the board-blind twin's probe network (T2 ablation), trained the same way on the same data
            torch.manual_seed(3500 + a.seed)
            ceb = dict(ce, loc_blind=1)
            model_eb, info_eb = fit(ceb, ("ev",), "WMEB ")
            twin = (model_eb, info_eb, ceb)
        info = {"steps_run": info_b["steps_run"] + info_e["steps_run"], "best_step": info_b["best_step"],
                "best_step_ev": info_e["best_step"], "steps_run_ev": info_e["steps_run"],
                "stop_reason": info_b["stop_reason"] + "/" + info_e["stop_reason"],
                "train_sec": info_b["train_sec"] + info_e["train_sec"],
                "eval_sec_during_training": info_b["eval_sec_during_training"] + info_e["eval_sec_during_training"],
                "flops_total": info_b["flops_total"] + info_e["flops_total"],
                "flops_at_best": info_b["flops_at_best"] + info_e["flops_at_best"],
                "history": info_b["history"], "history_ev": info_e["history"]}
        state = {"state": model.state_dict(), "state_e": model_e.state_dict()}
    else:
        model, info = fit(cfg, ("board", "ev"), "WM ")
        model_e = model
        state = {"state": model.state_dict()}
    os.makedirs(os.path.dirname(os.path.abspath(a.ckpt)), exist_ok=True)
    if twin is not None:   # blind twin first, so its checkpoint exists whenever the L0 one does
        model_eb, info_eb, ceb = twin
        torch.save({"state": model.state_dict(), "state_e": model_eb.state_dict(), "cfg": dict(cfg, loc_blind=1, wm_twin=0), "n": a.n, "seed": a.seed,
                    "wmtag": a.wmtag + "b", "flops_at_best": info_b["flops_at_best"] + info_eb["flops_at_best"],
                    "flops_total": info_b["flops_total"] + info_eb["flops_total"]}, a.ckpt + ".blind.pt.tmp")
        os.replace(a.ckpt + ".blind.pt.tmp", a.ckpt + ".blind.pt")
    torch.save({**state, "cfg": cfg, "n": a.n, "seed": a.seed, "wmtag": a.wmtag,
                "flops_at_best": info["flops_at_best"], "flops_total": info["flops_total"]}, a.ckpt + ".tmp")
    os.replace(a.ckpt + ".tmp", a.ckpt)
    t1 = time.time()
    res = {"kind": "wm", "wmtag": a.wmtag, "task": a.task, "n": a.n, "seed": a.seed, "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "set": a.set, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "params": sum(p.numel() for p in model.parameters()) * (2 if cfg["wm_split"] else 1), **{k: v for k, v in info.items()},
           "flops_per_board_at_best": info["flops_at_best"] / a.n, "l0": not cfg["loc_blind"]}
    res["val"] = wm_eval(model, cfg, V, a.val_n, diag=True, we=model_e)
    dgen = torch.Generator(device=dev)
    dgen.manual_seed(6000 + a.seed)
    occ, pc, pr, st = dream(model, 8192, dgen, loc_blind=bool(cfg["loc_blind"]), we=model_e)
    res["dreams"] = {**dream_stats(occ), **st}                       # valid_frac etc.: EVAL-ONLY
    if cfg["diag_post_n"] > 0:
        res["diag_post"] = wm_post_diag(model, cfg, V, min(cfg["diag_post_n"], a.val_n), dgen, we=model_e)   # EVAL-ONLY
    if twin is not None:   # the twin's own WM record (its board network is the L0 WM's)
        model_eb, info_eb, ceb = twin
        rb = dict(res, wmtag=a.wmtag + "b", cfg={k: v for k, v in ceb.items() if not k.startswith("_")}, l0=False,
                  best_step_ev=info_eb["best_step"], steps_run_ev=info_eb["steps_run"], history_ev=info_eb["history"],
                  flops_at_best=info_b["flops_at_best"] + info_eb["flops_at_best"], flops_total=info_b["flops_total"] + info_eb["flops_total"],
                  flops_per_board_at_best=(info_b["flops_at_best"] + info_eb["flops_at_best"]) / a.n,
                  note="board network shared with the L0 twin (identical in distribution: loc_blind changes only the probe network)")
        rb["val"] = wm_eval(model, ceb, V, a.val_n, diag=True, we=model_eb)
        occb, _, _, stb = dream(model, 8192, dgen, loc_blind=True, we=model_eb)
        rb["dreams"] = {**dream_stats(occb), **stb}
        ob = a.out.replace(f"/{a.wmtag}/{a.wmtag}-", f"/{a.wmtag}b/{a.wmtag}b-")
        assert ob != a.out
        save_json(rb, ob)
    res["final_eval_sec"] = time.time() - t1
    save_json(res, a.out)
    print("DONE wm", a.wmtag, a.n, a.seed, "val_nll", round(res["val"]["total"], 4), "best_step", info["best_step"],
          "dream_valid", round(res["dreams"]["valid_frac"], 3), "diag_kl", res.get("diag_post", {}).get("wm_post_valid_kl"), flush=True)


def load_wm(path, dev, wait_min=0.0):
    t0 = time.time()
    while not os.path.exists(path):
        if os.path.exists(path + ".failed"):
            raise SystemExit(f"WM job failed: {path}.failed")
        if time.time() - t0 > 60 * wait_min:
            raise SystemExit(f"no WM checkpoint {path} after {wait_min} min")
        dl = deadline_from_env(900)
        if dl and time.time() > dl:
            raise SystemExit("rental deadline while waiting for the WM checkpoint")
        time.sleep(15)
    ck = torch.load(path, map_location=dev)
    c = ck["cfg"]
    wm = WMNet(c["d"], c["layers"], c["heads"], c["ff"], c["drop"], c["m"]).to(dev)
    wm.load_state_dict(ck["state"])
    wm.eval()
    we = wm
    if "state_e" in ck:   # wm_split: separate probe model
        we = WMNet(c["d"], c["layers"], c["heads"], c["ff"], c["drop"], c["m"]).to(dev)
        we.load_state_dict(ck["state_e"])
        we.eval()
    object.__setattr__(wm, "we", we)   # the probe-model net (itself unless split); plain attribute, NOT a submodule (2026-10-11 fix)
    return wm, ck


# ----------------------------------------------------------------------------------------------- mode: fc
class DreamSet:
    """Dream buffer with the attribute layout of smallnet.TrainSet, so TrainSet.batch (symmetries, stages) is reused."""

    def __init__(self, wm, cfg, dev, dgen, loc_blind, fixed=None):
        self.wms = wm if isinstance(wm, list) else [wm]     # an ensemble: refills cycle through its members
        self.k = 0
        self.wm, self.cfg, self.dev, self.dgen, self.loc_blind = self.wms[0], cfg, dev, dgen, loc_blind
        self.sym = torch.tensor(D.SYM, dtype=torch.long, device=dev)
        self.post = None
        self.flops = 0.0
        self.n_dreamed = 0
        self.stats = None
        self.fixed = fixed is not None
        if fixed is not None:
            self.occ, self.pc, self.pr, self.post = fixed
            self.n = self.occ.shape[0]
        else:
            self.n = cfg["dream_buf"]
            self.refill()
        self.used = 0

    def refill(self):
        w = self.wms[self.k % len(self.wms)]
        self.k += 1
        occ, pc, pr, st = dream_fast(w, self.n, self.dgen, loc_blind=self.loc_blind, we=w.we)
        self.occ, self.pc, self.pr = occ, pc, pr
        self.flops += self.n * st["fwd_passes_per_dream"] * self.wm.flops_backbone()
        self.n_dreamed += self.n
        if self.stats is None:
            self.stats = {**dream_stats(occ), **st}
        self.used = 0

    def batch(self, B, stages, gen):
        if not self.fixed and self.used + B > self.n * self.cfg.get("dream_reuse", 1):
            self.refill()
        self.used += B
        return SN.TrainSet.batch(self, B, stages, gen)


def main_fc(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--wm-ckpt", default="")
    ap.add_argument("--wm-wait", type=float, default=200.0, help="minutes to wait for the WM checkpoint")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-n", type=int, default=3000)
    ap.add_argument("--ar-n", type=int, default=500)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--task", default=os.environ.get("WM2_TASK", "?"))
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(FC_BASE)
    cfg.update(ARMS[a.arm])
    if cfg["fc_arm"] == "rbtd":
        cfg.update(SN.ARMS["rbtd"])
        cfg.update(ARMS[a.arm])
    ms, ee, lr_n, ema_n = steps_for(a.n)
    cfg["lr"], cfg["ema"], cfg["eval_every"] = lr_n, ema_n, ee
    cfg = apply_sets(cfg, a.set, FC_BASE)
    dev = device_from(a)
    # --- identical to smallnet.main up to the data: same init for the same seed
    torch.manual_seed(1000 + a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(2000 + a.seed)
    shared = bool(cfg["shared"])
    wm, ck, ens = None, None, []
    if cfg["rho"] > 0 or shared or cfg["init_wm"]:
        for path in a.wm_ckpt.split(","):      # a comma list = an ensemble of WMs (dreams cycle through members)
            w_, c_ = load_wm(path, dev, a.wm_wait)
            for p_ in list(w_.parameters()) + list(w_.we.parameters()):
                p_.requires_grad_(False)
            ens.append((w_, c_))
        wm, ck = ens[0]
        if len(ens) > 1:
            assert not (shared or cfg["init_wm"] or cfg["post_target"]), "ensembles only for plain dream arms"
            ck = dict(ck, flops_at_best=sum(c_["flops_at_best"] for _, c_ in ens), flops_total=sum(c_["flops_total"] for _, c_ in ens),
                      wmtag="+".join(c_["wmtag"] for _, c_ in ens))
    wcfg = ck["cfg"] if ck else None
    if (shared or cfg["init_wm"]) and wcfg and wcfg.get("wm_split"):
        raise SystemExit("shared / init_wm need a single-network WM (wm_split=0)")
    loc_blind = bool(wcfg["loc_blind"]) if wcfg else False
    if shared or cfg["init_wm"]:
        net = WMNet(wcfg["d"], wcfg["layers"], wcfg["heads"], wcfg["ff"], wcfg["drop"], wcfg["m"]).to(dev)
        net.load_state_dict(ck["state"])
        net.train()
    else:
        net = Net(cfg["d"], cfg["layers"], cfg["heads"], cfg["ff"], cfg["drop"], cfg["m"]).to(dev)
    tr = SN.TrainSet(a.n, a.seed, dev)
    va = SN.EvalSet("val", a.val_n, dev)
    es_n = min(cfg["es_boards"], a.val_n)
    ep = EpisodeSet(a.n, a.seed, dev) if shared else None
    dgen = torch.Generator(device=dev)
    dgen.manual_seed(5000 + a.seed)
    Bd = int(round(cfg["B"] * cfg["rho"]))
    Br = cfg["B"] - Bd
    ds = None
    is_info = {}
    t_prep = time.time()
    if Bd > 0:
        if cfg["post_target"]:   # arm 4 (flagged): fixed dream pool, targets = WM posterior by importance sampling
            occ, pc, pr, st = dream_fast(wm, cfg["is_M"], dgen, loc_blind=loc_blind, we=wm.we)
            pool, _, _, _ = dream_fast(wm, cfg["is_K"], dgen, loc_blind=loc_blind, we=wm.we)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                cum = evidence_cum(wm.we, pool, pc, pr, loc_blind=loc_blind).float()
            post, ess = is_marginals(cum, pool)
            fb = wm.flops_backbone()
            is_flops = (cfg["is_M"] + cfg["is_K"]) * st["fwd_passes_per_dream"] * fb + cfg["is_M"] * cfg["is_K"] * 6 * fb * (2 if loc_blind else 1)
            ds = DreamSet(wm, cfg, dev, dgen, loc_blind, fixed=(occ, pc, pr, post.float()))
            ds.flops = is_flops
            ds.n_dreamed = cfg["is_M"] + cfg["is_K"]
            ds.stats = {**dream_stats(occ), **st}
            is_info = {"is_ess_stage": ess.mean(0).tolist(), "is_ess_min_stage": ess.min(0).values.tolist(), "is_M": cfg["is_M"], "is_K": cfg["is_K"]}
        else:
            ds = DreamSet([w_ for w_, _ in ens], cfg, dev, dgen, loc_blind)
    prep_sec = time.time() - t_prep
    teacher_on = cfg["teacher"]
    fcfg = dict(cfg)
    if cfg["post_target"]:
        fcfg["target"] = "post"

    def step_fn(net_, step, ema):
        parts = []
        if Br > 0:
            parts.append(tr.batch(Br, cfg["stages"], gen))
        if Bd > 0:
            xb, evb, pb = ds.batch(Bd, cfg["stages"], gen)
            parts.append((xb, evb, pb))
        if len(parts) == 1:
            batch = parts[0]
        else:
            (x1, e1, p1), (x2, e2, p2) = parts
            if p2 is not None:   # real episodes get their 0/1 board as the "posterior" target
                p1 = x1[:, None, :].float().expand(-1, e1.shape[1], -1)
                pb_ = torch.cat([p1, p2])
            else:
                pb_ = None
            batch = (torch.cat([x1, x2]), torch.cat([e1, e2]), pb_)
        loss, terms, fl = SN.train_loss(net_, fcfg, batch, gen, teacher=(ema if (teacher_on and ema is not None) else None))
        if shared:
            x, pc, pr = ep.batch(cfg["B"], gen)
            lw, tw, flw = wm_loss(net_, wcfg, x, pc, pr, gen)
            loss = loss + cfg["shared_w"] * lw
            fl += flw
            terms = {**terms, **{"wm_" + k: v for k, v in tw.items()}}
        return loss, terms, fl

    def eval_fn(model):
        q, _ = SN.readout(model, cfg, va, es_n, M=1)
        mm = SN.metrics(q, va, es_n)
        return mm["logloss"], {"val_logloss": mm["logloss"], "val_kl": mm["kl"]}

    model, info = train_loop(net, cfg, step_fn, eval_fn, dev, a.max_minutes, margin=300)
    res = {"tag": a.tag, "arm": a.arm, "task": a.task, "n": a.n, "seed": a.seed, "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "set": a.set, "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           **{k: v for k, v in info.items() if k != "best_score"},
           "params": sum(p.numel() for p in model.parameters()),
           "real_world_valid": True, "l0": not loc_blind, "flag": "arm4-own-posterior-target (Kelvin has not ruled)" if cfg["post_target"] else "",
           "wm_ckpt": a.wm_ckpt, "wm_tag": ck["wmtag"] if ck else None, "prep_sec": prep_sec, **is_info}
    # compute accounting (FLOPs, forward+backward = 3x forward; dream/IS forward passes at 1x)
    dflops_total = ds.flops if ds is not None else 0.0
    frac_best = info["best_step"] / max(1, info["steps_run"])
    dflops_best = dflops_total if cfg["post_target"] else dflops_total * frac_best
    wm_best = ck["flops_at_best"] if ck else 0.0
    wm_tot = ck["flops_total"] if ck else 0.0
    res["flops_wm_at_best"], res["flops_wm_total"] = wm_best, wm_tot
    res["flops_dreams_total"], res["flops_dreams_at_best"] = dflops_total, dflops_best
    res["flops_fc_at_best"], res["flops_fc_total"] = info["flops_at_best"], info["flops_total"]
    res["flops_all_at_best"] = wm_best + dflops_best + info["flops_at_best"]
    res["flops_all_total"] = wm_tot + dflops_total + info["flops_total"]
    res["flops_per_board_at_best"] = res["flops_all_at_best"] / a.n
    res["flops_per_board_total"] = res["flops_all_total"] / a.n
    res["n_dreams"] = ds.n_dreamed if ds is not None else 0
    res["dream_stats"] = ds.stats if ds is not None else None
    t1 = time.time()
    q, jl = SN.readout(model, cfg, va, a.val_n, M=1, joint=True)
    ar = SN.ar_joint(model, va, min(a.ar_n, a.val_n)) if (cfg["rbtd"] > 0 and not shared) else None
    res["val"] = SN.metrics(q, va, a.val_n, jl, ar)
    if "post_unaware" in D.load_split("val", 3000, with_post=True):   # T2: also score against the selection-unaware posterior
        pu = D.load_split("val", 3000, with_post=True)["post_unaware"][:a.val_n]
        res["val"]["kl_vs_unaware_post"] = float(SN.kl_bern(pu, q).mean())
    if a.test:
        ts = SN.EvalSet("test", 5000, dev)
        q, jl = SN.readout(model, cfg, ts, 5000, M=1, joint=True)
        ar = SN.ar_joint(model, ts, min(a.ar_n, 5000)) if (cfg["rbtd"] > 0 and not shared) else None
        res["test"] = SN.metrics(q, ts, 5000, jl, ar)
        res["test_kl_per_board"] = SN.kl_bern(ts.post, q).mean((1, 2)).tolist()
    res["final_eval_sec"] = time.time() - t1
    save_json(res, a.out)
    print("DONE fc", a.tag, a.n, a.seed, "val_kl", round(res["val"]["kl"], 6), "val_ll", round(res["val"]["logloss"], 6),
          "best_step", info["best_step"], "stop", info["stop_reason"], "train_sec", round(info["train_sec"], 1), flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv.pop(0)
    if mode == "wm":
        try:
            main_wm(argv)
        except BaseException:
            for i, x in enumerate(argv):
                if x == "--ckpt":
                    open(argv[i + 1] + ".failed", "w").write("failed\n")
                    open(argv[i + 1] + ".blind.pt.failed", "w").write("failed\n")
            raise
    elif mode == "fc":
        main_fc(argv)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
