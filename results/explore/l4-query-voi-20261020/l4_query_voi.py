#!/usr/bin/env python3
"""EXPLORATORY (PROTOCOL 10.1), stage-1 lane L4: which sources to consult under a budget, by value of information.

An agent's KEY (what it saw) is free; its REPORT costs one consultation. With a budget of b < n consultations a
policy picks agents one at a time and then aggregates the b reports it bought. Arms (selection + aggregation):

  random+exact, fixed+exact      random order / index order (agent order is shuffled by the env, so index order is
                                 also random in distribution), exact aggregation of the consulted reports
  static_voi+exact               rank agents by their stand-alone information, ignore overlap (zero-parameter readout
                                 of the keys)
  greedy_voi+exact               exact-Bayes greedy VoI: next = argmax_j info(S + j)
  best_subset+exact              argmax over all b-subsets of info(S): the Bayes-optimal b-policy in expectation (here
                                 adaptivity cannot help, see README); "the best b-subset ceiling"
  hindsight_best+exact           per episode the b-subset whose posterior is closest to Bayes (uses the realized Bayes
                                 posterior: an unachievable lower bound, reported for scale only)
  learned_query+learned          QueryNet: a query from the consulted set ("what the consensus still needs") attends
                                 over the keys of unconsulted agents, picks the next one, and an attention aggregator
                                 combines the consulted reports. Trained end to end on truth log loss (no distillation),
                                 straight-through Gumbel selection, one network for all budgets.
  learned_query+exact            the trained network's selections, exact aggregation (selection quality alone)
  random+learned                 random selections, the trained network's aggregator (aggregation alone)

info(S): E2/E3 q_S = m^T C_S^T (C_S M C_S^T)^+ C_S m (the posterior log-odds given S's reports is N(+-q_S, 2 q_S)
given Y); E1 log det of the posterior precision given S. "Exact aggregation" = the exact posterior given the consulted
reports and the structure read from the full-visibility keys (coverage and strengths in E2; A, B, sigma in E1). Both
are functions of the view at visibility "full"; no oracle field reaches a policy or the network.

Run from the repo root:  python3 results/explore/l4-query-voi-20261020/l4_query_voi.py --out DIR [--quick]
"""
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from aggbench import make_env  # noqa: E402
from aggbench.baselines import all_baselines  # noqa: E402
from aggbench.controls import random_weights, untrained_init  # noqa: E402
from aggbench.envs import View, clearing, report_posterior_logit  # noqa: E402
from aggbench.metrics import paired_difference, per_episode, summarize  # noqa: E402
from aggbench.results import make_result, split_record, write_result  # noqa: E402
from aggbench.verify import oracle_check  # noqa: E402

RUN_ID = "l4-query-voi-20261020"
VIS = "full"
SEL_POLICIES = ("random", "fixed", "static_voi", "greedy_voi", "best_subset", "hindsight_best")


# ----------------------------------------------------------------------------- exact subset posteriors from the view

def subset_tables(view, cfg):
    """For every episode and every non-empty subset (bitmask) of agents: info value and exact posterior.

    Uses only the view (reports + full-visibility keys) and public config constants."""
    n, N = view.keys.shape[:2]
    masks = np.arange(1, 2 ** N)
    members = [np.flatnonzero((mk >> np.arange(N)) & 1) for mk in masks]
    if view.kind == "bernoulli":
        K = view.public["n_pieces"]
        l0 = view.public["prior_logit"]
        cov = view.keys[..., :K] > 0.5
        mu = view.keys[..., K:].max(1)                        # strength of each piece any agent covers
        rep = view.reports["logit"]
        info = np.zeros((n, 2 ** N))
        post = np.full((n, 2 ** N), l0)
        ones, zeros = np.ones(N), np.zeros(N)
        for e in range(n):
            m = 2 * mu[e] ** 2
            for mk, S in zip(masks, members):
                C = cov[e, S].astype(float)
                Sm = (C * m) @ C.T
                c = (C * m).sum(1)
                info[e, mk] = c @ np.linalg.pinv(Sm, rcond=1e-10, hermitian=True) @ c
                post[e, mk] = report_posterior_logit(rep[e, S], cov[e, S], mu[e], ones[S], zeros[S], l0)
        return {"info": info, "logit": post}
    d, m_, r = cfg["d"], cfg["obs_dim"], cfg["n_factors"]
    A = view.keys[..., :m_ * d].reshape(n, N, m_, d)
    Bf = view.keys[..., m_ * d:m_ * d + m_ * r].reshape(n, N, m_, r)
    sig = np.exp(view.keys[..., -1])
    info = np.zeros((n, 2 ** N))
    means = np.zeros((n, 2 ** N, d))
    precs = np.tile(np.eye(d), (n, 2 ** N, 1, 1))
    for e in range(n):
        s = np.zeros((N, m_))
        for i in range(N):  # recover s_i from the report, as baselines.oracle_clearing does
            R = Bf[e, i] @ Bf[e, i].T + sig[e, i] ** 2 * np.eye(m_)
            h = view.reports["prec"][e, i] @ view.reports["mean"][e, i]
            s[i] = np.linalg.lstsq(A[e, i].T @ np.linalg.inv(R), h, rcond=None)[0]
        for mk, S in zip(masks, members):
            k = len(S)
            Kmat = A[e, S].reshape(k * m_, d)
            Bk = Bf[e, S].reshape(k * m_, r)
            Rk = Bk @ Bk.T + np.diag(np.repeat(sig[e, S] ** 2, m_))
            mean, P = clearing(Kmat, np.linalg.inv(Rk), np.eye(d), s[S].ravel())
            means[e, mk], precs[e, mk] = mean, P
            info[e, mk] = np.linalg.slogdet(P)[1]
    return {"info": info, "mean": means, "prec": precs}


def masks_to_bits(mask):
    return (mask.astype(np.int64) << np.arange(mask.shape[1])).sum(1)


def forecast_from_masks(table, mask, kind):
    bits = masks_to_bits(mask)
    idx = np.arange(len(bits))
    if kind == "bernoulli":
        return {"logit": table["logit"][idx, bits]}
    return {"mean": table["mean"][idx, bits], "prec": table["prec"][idx, bits]}


def order_masks(order, b):
    n, N = order.shape
    mask = np.zeros((n, N), dtype=bool)
    np.put_along_axis(mask, order[:, :b], True, axis=1)
    return mask


def policy_masks(table, b, N, batch_for_hindsight, kind, rng_order):
    """Dict policy -> (n, N) boolean selection mask for budget b."""
    n = table["info"].shape[0]
    info = table["info"]
    single = info[:, 1 << np.arange(N)]
    out = {"random": order_masks(rng_order, b),
           "fixed": order_masks(np.tile(np.arange(N), (n, 1)), b),
           "static_voi": order_masks(np.argsort(-single, axis=1, kind="stable"), b)}
    cur = np.zeros(n, dtype=np.int64)
    for _ in range(b):
        cand = np.full((n, N), -np.inf)
        for j in range(N):
            free = ((cur >> j) & 1) == 0
            cand[free, j] = info[free, cur[free] | (1 << j)]
        cur = cur | (1 << np.argmax(cand, axis=1))
    out["greedy_voi"] = ((cur[:, None] >> np.arange(N)) & 1).astype(bool)
    sized = np.array([mk for mk in range(1, 2 ** N) if bin(mk).count("1") == b])
    best = sized[np.argmax(info[:, sized], axis=1)]
    out["best_subset"] = ((best[:, None] >> np.arange(N)) & 1).astype(bool)
    # hindsight: smallest KL to the realized Bayes posterior over all b-subsets (evaluation-only ceiling)
    kls = np.stack([per_episode(forecast_from_masks(table, np.tile(((mk >> np.arange(N)) & 1).astype(bool), (n, 1)),
                                                    kind), batch_for_hindsight)["kl"] for mk in sized], axis=1)
    hb = sized[np.argmin(kls, axis=1)]
    out["hindsight_best"] = ((hb[:, None] >> np.arange(N)) & 1).astype(bool)
    return out


def greedy_order_fast(view, cfg):
    """Greedy-VoI consultation order (n, N) from the view, batched over episodes (teacher for mode "imitate").

    Same info(S) as subset_tables; unconsulted agents are masked out instead of sliced out."""
    n, N = view.keys.shape[:2]
    if view.kind == "bernoulli":
        K = view.public["n_pieces"]
        cov = (view.keys[..., :K] > 0.5).astype(float)
        m = 2 * view.keys[..., K:].max(1) ** 2                                   # (n, K)

        def info(sel):
            C = cov * sel[..., None]
            Sm = np.einsum("nik,nk,njk->nij", C, m, C)
            c = np.einsum("nik,nk->ni", C, m)
            return np.einsum("ni,nij,nj->n", c, np.linalg.pinv(Sm, rcond=1e-10, hermitian=True), c)
    else:
        d, m_, r = cfg["d"], cfg["obs_dim"], cfg["n_factors"]
        A = view.keys[..., :m_ * d].reshape(n, N * m_, d)
        Bk = view.keys[..., m_ * d:m_ * d + m_ * r].reshape(n, N * m_, r)
        R = np.einsum("nir,njr->nij", Bk, Bk) + np.einsum("ni,ij->nij", np.repeat(np.exp(view.keys[..., -1]) ** 2,
                                                                                  m_, axis=1), np.eye(N * m_))

        def info(sel):
            rows = np.repeat(sel, m_, axis=1)                                     # (n, N*m)
            Rm = R * rows[:, :, None] * rows[:, None, :] + np.einsum("ni,ij->nij", 1 - rows, np.eye(N * m_))
            Km = A * rows[..., None]
            P = np.eye(d) + np.einsum("nid,nij,nje->nde", Km, np.linalg.inv(Rm), Km)
            return np.linalg.slogdet(P)[1]
    sel = np.zeros((n, N))
    order = np.zeros((n, N), dtype=np.int64)
    for t in range(N):
        cand = np.full((n, N), -np.inf)
        for j in range(N):
            free = sel[:, j] == 0
            s2 = sel.copy()
            s2[:, j] = 1
            cand[free, j] = info(s2)[free]
        order[:, t] = np.argmax(cand, axis=1)
        sel[np.arange(n), order[:, t]] = 1
    return order


# ----------------------------------------------------------------------------- learned query network

def mlp(i, h, o):
    return torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU(),
                               torch.nn.Linear(h, o))


class QueryNet(torch.nn.Module):
    def __init__(self, kind, key_dim, width=64, heads=4):
        super().__init__()
        self.kind, self.width, self.heads = kind, width, heads
        self.enc = mlp(key_dim, width, width)          # agent embedding from its key (free information)
        self.phi = torch.nn.Linear(width, width)       # contribution of a consulted agent to the consensus state
        self.state = mlp(width + 2, width, width)      # consensus state (+ step, remaining budget)
        self.wq = torch.nn.Linear(width, width)        # QUERY: what the consensus still needs
        self.wk = torch.nn.Linear(width, width)        # KEY of each candidate
        self.aq = torch.nn.Linear(width, width)        # aggregator: attention among consulted agents
        self.ak = torch.nn.Linear(width, width)
        self.av = torch.nn.Linear(width, width)
        self.gate = mlp(2 * width, width, 1)           # per-agent weight on its report

    def select(self, e, b, tau=1.0, train=False, forced=None):
        n, N, _ = e.shape
        if forced is not None:
            return forced
        w = torch.zeros(n, N, device=e.device, dtype=e.dtype)
        k, pe = self.cand_keys(e), self.phi(e)
        for t in range(b):
            s = self.scores(k, pe, w, t, b)
            if train:
                u = torch.rand_like(s).clamp(1e-10, 1 - 1e-7)
                g = -torch.log(-torch.log(u))
                soft = torch.softmax(s / tau + g, dim=1)
                hard = torch.nn.functional.one_hot(soft.argmax(1), N).to(s.dtype)
                y = hard - soft.detach() + soft
            else:
                y = torch.nn.functional.one_hot(s.argmax(1), N).to(s.dtype)
            w = w + y
        return w

    def cand_keys(self, e):
        n, N, _ = e.shape
        return self.wk(e).view(n, N, self.heads, self.width // self.heads)

    def scores(self, k, pe, w, t, b):
        """Attention logits of the consensus QUERY over unconsulted candidates' KEYS (consulted ones masked)."""
        n, N, H, dh = k.shape
        pooled = (w[..., None] * pe).sum(1)
        feats = torch.cat([pooled, torch.full((n, 1), t / N, device=k.device),
                           torch.full((n, 1), (b - t) / N, device=k.device)], 1)
        q = self.wq(self.state(feats)).view(n, H, dh)
        s = torch.einsum("nhd,njhd->nj", q, k) / math.sqrt(dh)
        return s.masked_fill(w.detach() > 0.5, -1e9)

    def aggregate(self, e, w, rep):
        lg = torch.einsum("nid,njd->nij", self.aq(e), self.ak(e)) / math.sqrt(self.width)
        a = w[:, None, :] * torch.exp(lg - lg.max(-1, keepdim=True).values)
        a = a / (a.sum(-1, keepdim=True) + 1e-9)
        ctx = a @ self.av(e)
        g = torch.nn.functional.softplus(self.gate(torch.cat([e, ctx], -1)))[..., 0] * w   # (n, N)
        if self.kind == "bernoulli":
            l0, r = rep
            return {"logit": l0 + (g * (r - l0)).sum(1)}
        P, h = rep                                       # (n, N, d, d), (n, N, d) information form
        eye = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
        prec = eye + (g[..., None, None] * (P - eye)).sum(1)
        info = (g[..., None] * h).sum(1)
        return {"prec": prec, "info": info}


class QueryAggregator:
    """aggbench learned-aggregator interface: .module, .fit(view, truth), .predict(view).

    predict reads the budget from view.public["budget"] and, optionally, forced selections from
    view.public["forced_mask"] (random+learned arm). It returns the forecast plus the selection mask it used."""

    def __init__(self, seed, kind, key_dim, steps=4000, batch=512, lr=2e-3, tau=(1.0, 0.3), device="cpu",
                 mode="truth", aux=None):
        """mode "truth": loss = log loss of the truth (default). "distill": loss = KL(train-split Bayes || forecast)
        (declared distillation, aux["bayes"]). "imitate": truth loss + cross-entropy of the query's choice against
        the greedy-VoI order computed from the train view (aux["teacher"], teacher-forced)."""
        torch.manual_seed(seed)
        self.mode, self.aux = mode, aux or {}
        self.seed, self.kind, self.steps, self.batch, self.lr, self.tau = seed, kind, steps, batch, lr, tau
        self.device = device
        self.module = QueryNet(kind, key_dim).to(device)
        self.log = []

    def _tensors(self, view):
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        keys = t(view.keys)
        if self.kind == "bernoulli":
            return keys, (float(view.public["prior_logit"]), t(view.reports["logit"]))
        P = t(view.reports["prec"])
        return keys, (P, torch.einsum("naij,naj->nai", P, t(view.reports["mean"])))

    def _loss(self, out, y, idx=None):
        if self.mode == "distill":
            return self._kl_to_bayes(out, idx)
        if self.kind == "bernoulli":
            return torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], y)
        L = torch.linalg.cholesky(out["prec"])
        mean = torch.cholesky_solve(out["info"][..., None], L)[..., 0]
        diff = y - mean
        quad = torch.einsum("ni,nij,nj->n", diff, out["prec"], diff)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return 0.5 * (quad - logdet).mean()

    def _kl_to_bayes(self, out, idx):
        bt = self._bayes
        if self.kind == "bernoulli":
            return torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], torch.sigmoid(bt[0][idx]))
        mb, covb = bt[0][idx], bt[1][idx]                 # KL(N(mb, covb) || N(mq, Pq^-1)) up to a constant
        L = torch.linalg.cholesky(out["prec"])
        mq = torch.cholesky_solve(out["info"][..., None], L)[..., 0]
        diff = mb - mq
        tr = torch.einsum("nij,nji->n", out["prec"], covb)
        quad = torch.einsum("ni,nij,nj->n", diff, out["prec"], diff)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return 0.5 * (tr + quad - logdet).mean()

    def fit(self, view, truth):
        keys, rep = self._tensors(view)
        t = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=self.device)
        if self.mode == "distill":
            b = self.aux["bayes"]
            self._bayes = ((t(b["logit"]),) if self.kind == "bernoulli" else
                           (t(b["mean"]), t(np.linalg.inv(b["prec"]))))
        if self.mode == "imitate":
            teacher = t(self.aux["teacher"], torch.long)
        y = torch.tensor(np.asarray(truth), dtype=torch.float32, device=self.device)
        n, N = keys.shape[:2]
        rng = np.random.default_rng(1000 + self.seed)
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        self.module.train()
        for step in range(self.steps):
            idx = torch.tensor(rng.choice(n, self.batch, replace=False), device=self.device)
            b = int(rng.integers(1, N + 1))
            tau = self.tau[0] * (self.tau[1] / self.tau[0]) ** (step / max(self.steps - 1, 1))
            e = self.module.enc(keys[idx])
            w = self.module.select(e, b, tau=tau, train=True)
            r = (rep[0], rep[1][idx]) if self.kind == "bernoulli" else (rep[0][idx], rep[1][idx])
            loss = self._loss(self.module.aggregate(e, w, r), y[idx], idx)
            if self.mode == "imitate":
                k, pe = self.module.cand_keys(e), self.module.phi(e)
                wt = torch.zeros_like(w)
                for st in range(b):
                    nxt = teacher[idx, st]
                    loss = loss + torch.nn.functional.cross_entropy(self.module.scores(k, pe, wt, st, b), nxt) / b
                    wt = wt + torch.nn.functional.one_hot(nxt, N).to(wt.dtype)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5.0)
            opt.step()
            sched.step()
            if step % 500 == 0 or step == self.steps - 1:
                self.log.append((step, b, float(loss)))
        self.module.eval()
        return self

    def predict(self, view):
        keys, rep = self._tensors(view)
        b = int(view.public["budget"])
        forced = view.public.get("forced_mask")
        with torch.no_grad():
            e = self.module.enc(keys)
            fm = None if forced is None else torch.tensor(forced, dtype=torch.float32, device=self.device)
            w = self.module.select(e, b, forced=fm)
            out = self.module.aggregate(e, w, rep)
        mask = w.cpu().numpy() > 0.5
        if self.kind == "bernoulli":
            return {"logit": out["logit"].double().cpu().numpy(), "mask": mask}
        P = out["prec"].double().cpu().numpy()
        return {"mean": np.linalg.solve(P, out["info"].double().cpu().numpy()[..., None])[..., 0], "prec": P,
                "mask": mask}


def budget_view(view, b, forced=None):
    return View(view.kind, view.reports, view.keys, {**view.public, "budget": b, "forced_mask": forced})


# ----------------------------------------------------------------------------- experiment

def dup_rate(mask, keys):
    """Fraction of episodes in which the selection holds two agents with identical keys (a Sybil duplicate)."""
    out = np.zeros(len(mask), dtype=bool)
    for e in range(len(mask)):
        S = np.flatnonzero(mask[e])
        kk = keys[e, S]
        out[e] = len(np.unique(np.round(kk, 9), axis=0)) < len(S)
    return float(out.mean())


def run_env(name, args, device, log):
    env = make_env(name)
    t0 = time.time()
    tr, dv = env.sample("train", args.n_train), env.sample("dev", args.n_dev)
    tv, dvv = tr.view(VIS), dv.view(VIS)
    cfg = env.config.__dict__
    N = dv.keys[VIS].shape[1]
    kind = dv.kind
    log(f"[{name}] sampled train {len(tr)} dev {len(dv)} N={N} in {time.time() - t0:.0f}s")
    t0 = time.time()
    table = subset_tables(dvv, cfg)
    # sanity: exact aggregation of all agents from the view equals the harness oracle_clearing (= Bayes in E1)
    full = forecast_from_masks(table, np.ones((len(dv), N), dtype=bool), kind)
    base, alpha = all_baselines(tr, dv, VIS)
    key = "logit" if kind == "bernoulli" else "mean"
    check_full = float(np.abs(full[key] - base["oracle_clearing"][key]).max())
    log(f"[{name}] subset tables in {time.time() - t0:.0f}s; |view-exact(all) - oracle_clearing| = {check_full:.2e}")
    assert check_full < 1e-6, "exact-from-view aggregation disagrees with harness oracle_clearing"

    forecasts = dict(base)
    rng = np.random.default_rng(7)
    rng_order = np.argsort(rng.random((len(dv), N)), axis=1)
    budgets = list(range(1, N))
    sel_masks = {}
    for b in budgets:
        pm = policy_masks(table, b, N, dv, kind, rng_order)
        sel_masks[b] = pm
        for pol, m in pm.items():
            forecasts[f"b{b}/{pol}+exact"] = forecast_from_masks(table, m, kind)

    key_dim = dv.keys[VIS].shape[-1]
    aux, teacher_check = None, None
    if args.train_mode == "distill":
        aux = {"bayes": tr.bayes}                       # declared distillation: train-split Bayes posterior as label
    elif args.train_mode == "imitate":
        t0 = time.time()
        aux = {"teacher": greedy_order_fast(tv, cfg)}   # computed from the train VIEW (keys), no oracle field
        dev_order = greedy_order_fast(dvv, cfg)
        ar = np.arange(len(dv))   # same info(S) as the exact greedy sets (sets may differ only on exact ties)
        teacher_check = {f"b{b}": float(np.abs(table["info"][ar, masks_to_bits(order_masks(dev_order, b))]
                                               - table["info"][ar, masks_to_bits(sel_masks[b]["greedy_voi"])]).max())
                         for b in budgets}
        log(f"[{name}] greedy teacher in {time.time() - t0:.0f}s; dev max |info(teacher set) - info(greedy set)| {teacher_check}")
    make = lambda s: QueryAggregator(s, kind, key_dim, steps=args.steps, batch=args.batch, device=device,
                                     mode=args.train_mode, aux=aux)
    learned_arms, controls, extra_learn = [], {}, {}
    for seed in range(args.seeds):
        t0 = time.time()
        trained = make(seed).fit(tv, tr.truth)
        # aggbench.controls.random_weights draws on the CPU, so build the control from a CPU copy, then move back
        trained.module.cpu()
        rnd = random_weights(trained, seed + 1)
        trained.module.to(device)
        rnd.module.to(device)
        nets = {"trained": trained,
                "control:untrained_init": untrained_init(make, seed),
                "control:random_network": rnd}
        log(f"[{name}] seed {seed} trained in {time.time() - t0:.0f}s; loss log {trained.log[-3:]}")
        extra_learn[f"s{seed}"] = {"train_log": trained.log, "train_seconds": time.time() - t0}
        for b in budgets + [N]:
            for tag, net in nets.items():
                sfx = f":s{seed}" + ("" if tag == "trained" else f":{tag}")
                lq = net.predict(budget_view(dvv, b))
                if b == N:
                    forecasts[f"b{N}/all+learned{sfx}"] = lq
                    continue
                forecasts[f"b{b}/learned_query+learned{sfx}"] = lq
                forecasts[f"b{b}/learned_query+exact{sfx}"] = forecast_from_masks(table, lq["mask"], kind)
                forecasts[f"b{b}/random+learned{sfx}"] = net.predict(budget_view(dvv, b, sel_masks[b]["random"]))
        for b in budgets + [N]:
            arms = ([f"b{N}/all+learned"] if b == N else
                    [f"b{b}/learned_query+learned", f"b{b}/learned_query+exact", f"b{b}/random+learned"])
            for a in arms:
                arm = f"{a}:s{seed}"
                learned_arms.append(arm)
                controls[arm] = {"random_network": f"{arm}:control:random_network",
                                 "untrained_init": f"{arm}:control:untrained_init"}

    t0 = time.time()
    clean = {k: {kk: vv for kk, vv in f.items() if kk != "mask"} for k, f in forecasts.items()}
    methods = {k: summarize(f, dv, n_boot=args.n_boot) for k, f in clean.items()}
    paired, selection = {}, {}
    for b in budgets:
        ceil = clean[f"b{b}/best_subset+exact"]
        for k in clean:
            if k.startswith(f"b{b}/") and k != f"b{b}/best_subset+exact":
                paired[f"{k} - b{b}/best_subset+exact"] = paired_difference(clean[k], ceil, dv, n_boot=args.n_boot)
        for seed in range(args.seeds):
            for tag in ("", ":control:random_network", ":control:untrained_init"):
                lq = clean[f"b{b}/learned_query+learned:s{seed}{tag}"]
                paired[f"b{b}/learned_query+learned:s{seed}{tag} - b{b}/random+learned:s{seed}{tag}"] = \
                    paired_difference(lq, clean[f"b{b}/random+learned:s{seed}{tag}"], dv, n_boot=args.n_boot)
                paired[f"b{b}/learned_query+learned:s{seed}{tag} - b{b}/random+exact"] = \
                    paired_difference(lq, clean[f"b{b}/random+exact"], dv, n_boot=args.n_boot)
            paired[f"b{b}/learned_query+learned:s{seed} - b{b}/learned_query+learned:s{seed}:control:random_network"] = \
                paired_difference(clean[f"b{b}/learned_query+learned:s{seed}"],
                                  clean[f"b{b}/learned_query+learned:s{seed}:control:random_network"], dv,
                                  n_boot=args.n_boot)
        # selection diagnostics: agreement with greedy VoI and with the best subset, Sybil duplicate rate
        g, bs = sel_masks[b]["greedy_voi"], sel_masks[b]["best_subset"]
        sels = {pol: m for pol, m in sel_masks[b].items()}
        for seed in range(args.seeds):
            for tag in ("", ":control:random_network", ":control:untrained_init"):
                sels[f"learned_query:s{seed}{tag}"] = forecasts[f"b{b}/learned_query+learned:s{seed}{tag}"]["mask"]
        selection[f"b{b}"] = {pol: {"same_set_as_greedy_voi": float((m == g).all(1).mean()),
                                    "same_set_as_best_subset": float((m == bs).all(1).mean()),
                                    "holds_sybil_duplicate": dup_rate(m, dvv.keys)} for pol, m in sels.items()}
    log(f"[{name}] metrics in {time.time() - t0:.0f}s")
    for b in budgets:
        line = " ".join(f"{a}={methods[f'b{b}/{a}']['regret']['mean']:.3f}" for a in
                        ("random+exact", "static_voi+exact", "greedy_voi+exact", "best_subset+exact",
                         "learned_query+exact:s0", "learned_query+learned:s0", "random+learned:s0",
                         "learned_query+learned:s0:control:random_network"))
        log(f"[{name}] b={b} {line}")

    extra = {"train_mode": args.train_mode, "teacher_dev_max_info_diff": teacher_check, "budgets": budgets, "n_consultable": N, "extremize_alpha": alpha, "paired": paired,
             "selection": selection, "learn": extra_learn, "view_exact_vs_oracle_clearing_max_abs": check_full,
             "hyperparameters": {"width": 64, "heads": 4, "steps": args.steps, "batch": args.batch, "lr": 2e-3,
                                 "tau": [1.0, 0.3], "budget_sampling": "uniform 1..N per minibatch",
                                 "random_order_seed": 7, "seeds": args.seeds},
             "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "repo_commit": args.repo_commit, "device": device, "quick": args.quick}}
    sfx = "" if args.train_mode == "truth" else f"-{args.train_mode}"
    result = make_result(run_id=f"{RUN_ID}-{name}{sfx}", track="SMOKE" if args.quick else "EXPLORATORY",
                         smoke=args.quick, env=env, visibility=VIS,
                         splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
                         oracle_check=oracle_check(env), methods=methods, learned_arms=learned_arms,
                         controls=controls, extra=extra)
    h = result["harness"]
    if h["git_commit"] == "unknown":  # VM copy has no .git: prove the harness sources by hash instead
        if not args.harness_sha256 or h["sha256"] != args.harness_sha256:
            raise SystemExit(f"harness sha256 {h['sha256']} does not match expected {args.harness_sha256}")
        h["git_commit"] = args.repo_commit
        h["git_dirty"] = False
        extra["harness_provenance"] = ("VM checkout without .git; aggbench sha256 matched the sha256 computed at "
                                       "the repo commit before launch")
    return write_result(args.out, result, name=f"result-{name}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="E2,E3-sybil,E1")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repo-commit", default="unknown")
    ap.add_argument("--harness-sha256", default="")
    ap.add_argument("--train-mode", default="truth", choices=["truth", "distill", "imitate"])
    ap.add_argument("--quick", action="store_true", help="SMOKE: tiny sizes, no verdicts")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_dev, args.steps, args.seeds, args.n_boot = 3000, 200, 150, 1, 200
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    logf = open(Path(args.out) / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} device={device} args={vars(args)}")
    for name in args.envs.split(","):
        log(f"wrote {run_env(name, args, device, log)}")
    log("DONE")


if __name__ == "__main__":
    main()
