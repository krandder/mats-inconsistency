"""Global joint fit of Qwen3-32B's stated beliefs at one date (analysis only, no API calls, no training).

Input: scripts/worldmodel_global.py output (events.jsonl, queries.jsonl, log.jsonl, graph.json).

Model: pairwise binary Markov random field (Ising in {0,1} coordinates) over the N events,
    log p(x) = sum_i h_i x_i + sum_{(i,j) in E} J_ij x_i x_j - log Z,
with E = the queried pairs (pair queries plus pairs inside 3-way queries).  J_ij is the
conditional log odds ratio of i and j given every other event.
Fit: minimise sum_q KL_Bernoulli(stated_q || implied_q) + lambda * R(J), R = sum J^2 (L2) or
sum |J| (L1, smoothed), h unpenalised; L-BFGS (torch autograd).
Inference: the queried graph has treewidth far above what a junction tree can hold (bound
printed), so implied probabilities come from damped loopy belief propagation (Bethe),
run to convergence (max message change < 1e-10) without gradients, then 30 differentiable
iterations from the fixed point (truncated backprop; the values are the fixed point):
    marginal P(i)                -> BP node belief
    pair on an edge              -> BP pair belief
    3-way AND with all 3 edges   -> Kirkwood closure P_ij P_jk P_ik / (P_i P_j P_k), normalised
    anything else (held-out pair without an edge, etc.) -> sequential clamping:
                                    P(l1) P(l2 | l1) P(l3 | l1, l2), each by BP with the
                                    earlier literals clamped
A chromatic Gibbs sampler (1,000 parallel chains, 600 sweeps) checks the BP beliefs of the final fit,
and every held-out and accuracy number is also computed from the fitted MRF itself by Gibbs
sampling (1,000 chains x 500 kept sweeps), which needs no Bethe approximation.
Penalty: the largest lambda whose validation KL is within 5% of the best.
Stated values are clipped to [0.01, 0.99]; missing answers are left out.

Held-out: 20% of pair units (a pair with its "A and B" and "A and not B" queries; stratified
sim/rand) and 20% of the 3-way queries are removed from the fit.  lambda (L1 or L2 grid) is
chosen on a further 15% validation split of the remaining units, then the fit is redone on
all non-held-out queries.  Held-out answers are predicted by the fit, by independence of the
stated marginals, and compared with the raw answers against outcomes.

Controls: independence of the stated marginals, and a marginals-only fit (J = 0, h fitted to
all the same queries), which absorbs any uniform tendency (e.g. stated ANDs above the product
of stated marginals) without any pairwise structure.  Structure is also reported for other L2
penalties (lambda sensitivity).

Local ("pairwise") projection: for a pair, the 2x2 fitted to that pair's queries only
(singles A, B, A and B, A and not B); for a 3-way query, the 2^3 table fitted to the queries
among its three events; for a marginal, the mean of the pair projections that contain it.

Usage: worldmodel_global_fit.py --data /home/agent/data/worldmodel-global-20260924 \
          --out-json docs/worldmodel-global-20260924.json --charts docs/worldmodel-global-20260924-charts.json
Needs torch + networkx (/home/agent/workspaces/venv-cft/bin/python).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.asof import AsOf, to_ms, DAY  # noqa: E402

torch.set_default_dtype(torch.float64)
THREADS = 2
torch.set_num_threads(THREADS)
LO = 0.01
SEED = 20260924
GRAD_ITERS = 30
DAMP = 0.5
CLAMP = 30.0
STRONG = 0.5
T_ISO = "2025-09-15"
LAMBDAS = {"l2": [0.01, 0.03, 0.1, 0.3], "l1": [0.01, 0.03]}
F = torch.nn.functional


def clip(p):
    return min(max(p, LO), 1 - LO)


def kl(s, p):
    s, p = clip(s), clip(p)
    return s * math.log(s / p) + (1 - s) * math.log((1 - s) / (1 - p))


def ll(p, y):
    p = clip(p)
    return -math.log(p if y else 1 - p)


def brier(p, y):
    return (clip(p) - y) ** 2


def lits(f):
    """Conjunction of literals -> [(event, value)]."""
    if f["op"] == "var":
        return [(f["i"], 1)]
    if f["op"] == "not":
        return [(f["a"]["i"], 0)]
    assert f["op"] == "and"
    out = []
    for a in f["args"]:
        out += lits(a)
    return out


# --- loopy BP -------------------------------------------------------------------

class Graph:
    def __init__(self, n: int, edges: list[tuple[int, int]]):
        self.n = n
        self.edges = sorted({(min(i, j), max(i, j)) for i, j in edges})
        self.eid = {e: k for k, e in enumerate(self.edges)}
        E = len(self.edges)
        u = torch.tensor([e[0] for e in self.edges], dtype=torch.long)
        v = torch.tensor([e[1] for e in self.edges], dtype=torch.long)
        self.u, self.v, self.E = u, v, E
        self.src = torch.cat([u, v])  # directed d: src -> dst
        self.dst = torch.cat([v, u])
        self.rev = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

    def bp(self, h, J, iters=None, mu0=None, tol=1e-10, max_iters=3000, grad_iters=GRAD_ITERS):
        """h: (B, N), J: (E,).  Returns node logits H (B, N), messages mu (B, 2E).

        iters=None: run damped BP without gradients until the largest message change is below
        tol (or max_iters), then run grad_iters differentiable iterations from that fixed point
        (the values are the fixed point; the gradient is truncated backprop through grad_iters
        steps).  self.last_residual records the final residual.  iters=k: exactly k iterations."""
        B = h.shape[0]
        J2 = torch.cat([J, J])[None, :]

        def step(mu, h_, J2_):
            H = h_.index_add(1, self.dst, mu)
            a = H[:, self.src] - mu[:, self.rev]
            return DAMP * mu + (1 - DAMP) * (F.softplus(a + J2_) - F.softplus(a))
        mu = torch.zeros(B, 2 * self.E) if mu0 is None else mu0
        if iters is not None:
            for _ in range(iters):
                mu = step(mu, h, J2)
        else:
            with torch.no_grad():
                hd, Jd = h.detach(), J2.detach()
                res = float("inf")
                for k in range(max_iters):
                    new = step(mu, hd, Jd)
                    if k % 10 == 0:
                        res = float((new - mu).abs().max())
                        if res < tol:
                            mu = new
                            break
                    mu = new
                self.last_residual = res
                self.last_iters = k + 1
            mu = mu.detach()
            for _ in range(grad_iters):
                mu = step(mu, h, J2)
        H = h.index_add(1, self.dst, mu)
        return H, mu

    def pair_logp(self, H, mu, J, eids):
        """log P(x_u, x_v) for edges eids: (B, len, 4) in order 00, 01, 10, 11 (x_u, x_v)."""
        eids = torch.as_tensor(eids, dtype=torch.long)
        u, v = self.u[eids], self.v[eids]
        au = H[:, u] - mu[:, eids + self.E]  # message v->u is the reverse direction
        av = H[:, v] - mu[:, eids]
        Je = J[eids][None, :]
        z = torch.stack([torch.zeros_like(au), av, au, au + av + Je], -1)
        return z - torch.logsumexp(z, -1, keepdim=True)


def prob_on_graph(g: Graph, H, mu, J, lit_list):
    """Implied P(conjunction) for one conjunction under BP beliefs (differentiable); None if not direct."""
    if len(lit_list) == 1:
        (i, x), = lit_list
        p = torch.sigmoid(H[0, i])
        return p if x else 1 - p
    if len(lit_list) == 2:
        (i, xi), (j, xj) = lit_list
        e = (min(i, j), max(i, j))
        if e not in g.eid:
            return None
        lp = g.pair_logp(H, mu, J, [g.eid[e]])[0, 0]
        if i > j:
            xi, xj = xj, xi
        return lp[2 * xi + xj].exp()
    if len(lit_list) == 3:
        ids = [i for i, _ in lit_list]
        es = [(min(a, b), max(a, b)) for a, b in ((ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2]))]
        if not all(e in g.eid for e in es):
            return None
        lp = g.pair_logp(H, mu, J, [g.eid[e] for e in es])[0]  # (3, 4)
        lpi = F.logsigmoid(torch.stack([H[0, i] for i in ids]))
        lpn = F.logsigmoid(-torch.stack([H[0, i] for i in ids]))
        tabs = []
        for x in range(8):
            xs = [(x >> 2) & 1, (x >> 1) & 1, x & 1]
            t = 0
            for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                ea = es[k]
                xa, xb = xs[a], xs[b]
                if ids[a] > ids[b]:
                    xa, xb = xb, xa
                t = t + lp[k, 2 * xa + xb]
            for a in range(3):
                t = t - (lpi[a] if xs[a] else lpn[a])
            tabs.append(t)
        tabs = torch.stack(tabs)
        want = sum(((x_ << (2 - k)) for k, (_, x_) in enumerate(lit_list)))
        return (tabs[want] - torch.logsumexp(tabs, 0)).exp()
    return None


def clamped_probs(g: Graph, h, J, lit_lists, batch=256):
    """Sequential clamping: P(l1) P(l2|l1) ... for each conjunction (no grad)."""
    out = np.ones(len(lit_lists))
    with torch.no_grad():
        H0, _ = g.bp(h[None, :], J)
        p0 = torch.sigmoid(H0[0]).numpy()
        for k, L in enumerate(lit_lists):
            i, x = L[0]
            out[k] = p0[i] if x else 1 - p0[i]
        maxlen = max(len(L) for L in lit_lists)
        for stage in range(1, maxlen):
            idx = [k for k, L in enumerate(lit_lists) if len(L) > stage]
            for s0 in range(0, len(idx), batch):
                chunk = idx[s0:s0 + batch]
                hb = h[None, :].repeat(len(chunk), 1).clone()
                for r, k in enumerate(chunk):
                    for i, x in lit_lists[k][:stage]:
                        hb[r, i] = CLAMP if x else -CLAMP
                Hb, _ = g.bp(hb, J)
                pb = torch.sigmoid(Hb).numpy()
                for r, k in enumerate(chunk):
                    i, x = lit_lists[k][stage]
                    out[k] *= pb[r, i] if x else 1 - pb[r, i]
    return out


# --- fitting --------------------------------------------------------------------

def fit_mrf(n, train_q, marg0, reg, lam, max_iter=300, couplings=True):
    """train_q: list of (lits, stated). Returns (graph, h, J, info)."""
    edges = set()
    for L, _ in train_q:
        ids = [i for i, _ in L]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                edges.add((min(ids[a], ids[b]), max(ids[a], ids[b])))
    g = Graph(n, sorted(edges))
    h = torch.tensor([math.log(clip(p) / (1 - clip(p))) for p in marg0], requires_grad=True)
    J = torch.zeros(g.E, requires_grad=couplings)
    s = torch.tensor([clip(p) for _, p in train_q])
    # group queries by arity for vectorised implied probabilities
    singles = [(k, L[0]) for k, (L, _) in enumerate(train_q) if len(L) == 1]
    pairs = [(k, L) for k, (L, _) in enumerate(train_q) if len(L) == 2]
    triples = [(k, L) for k, (L, _) in enumerate(train_q) if len(L) == 3]
    s_idx = torch.tensor([k for k, _ in singles]); s_i = torch.tensor([l[0] for _, l in singles])
    s_x = torch.tensor([l[1] for _, l in singles], dtype=torch.float64)
    p_idx = torch.tensor([k for k, _ in pairs])
    p_e, p_cell = [], []
    for _, L in pairs:
        (i, xi), (j, xj) = L
        if i > j:
            (i, xi), (j, xj) = (j, xj), (i, xi)
        p_e.append(g.eid[(i, j)]); p_cell.append(2 * xi + xj)
    p_e = torch.tensor(p_e, dtype=torch.long); p_cell = torch.tensor(p_cell, dtype=torch.long)
    t_idx = [k for k, _ in triples]

    def implied(H, mu):
        p = torch.empty(len(train_q))
        ps = torch.sigmoid(H[0, s_i])
        p = p.index_put((s_idx,), s_x * ps + (1 - s_x) * (1 - ps)) if len(singles) else p
        if len(pairs):
            lp = g.pair_logp(H, mu, J, p_e)[0]
            p = p.index_put((p_idx,), lp.gather(1, p_cell[:, None])[:, 0].exp())
        if triples:
            pt = torch.stack([prob_on_graph(g, H, mu, J, L) for _, L in triples])
            p = p.index_put((torch.tensor(t_idx),), pt)
        return p.clamp(1e-9, 1 - 1e-9)

    def loss_fn():
        H, mu = g.bp(h[None, :], J)
        p = implied(H, mu)
        data = (s * torch.log(s / p) + (1 - s) * torch.log((1 - s) / (1 - p))).sum()
        pen = (J ** 2).sum() if reg == "l2" else torch.sqrt(J ** 2 + 1e-4).sum()
        return data + lam * pen, data

    opt = torch.optim.LBFGS([h, J] if couplings else [h], lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        tot, _ = loss_fn()
        tot.backward()
        return tot
    t0 = time.time()
    opt.step(closure)
    with torch.no_grad():
        tot, data = loss_fn()
        H, mu = g.bp(h[None, :], J)
        resid = g.last_residual
    info = {"reg": reg, "lam": lam, "edges": g.E, "loss": float(tot), "data_kl": float(data),
            "mean_kl": float(data) / len(train_q), "bp_residual": resid, "bp_iters": g.last_iters, "seconds": round(time.time() - t0, 1)}
    return g, h.detach(), J.detach(), info


def predict(g, h, J, lit_lists):
    """Implied probabilities for arbitrary conjunctions under a fitted model (no grad)."""
    out = np.full(len(lit_lists), np.nan)
    with torch.no_grad():
        H, mu = g.bp(h[None, :], J)
        rest = []
        for k, L in enumerate(lit_lists):
            p = prob_on_graph(g, H, mu, J, L)
            if p is None:
                rest.append(k)
            else:
                out[k] = float(p)
    if rest:
        out[rest] = clamped_probs(g, h, J, [lit_lists[k] for k in rest])
    return out


def gibbs_check(g, h, J, chains=1000, sweeps=600, burn=200, seed=SEED):
    """Node and pair P11 marginals by chromatic Gibbs sampling (greedy colouring) vs converged BP."""
    torch.manual_seed(seed)
    n = g.n
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(g.edges)
    col = nx.greedy_color(G, strategy="largest_first")
    groups = [torch.tensor([v for v in range(n) if col[v] == c]) for c in range(max(col.values()) + 1)]
    W = torch.zeros(n, n)
    W[g.u, g.v] = J; W[g.v, g.u] = J
    with torch.no_grad():
        H, mu = g.bp(h[None, :], J)
        bp_node = torch.sigmoid(H[0])
        bp_pair = g.pair_logp(H, mu, J, torch.arange(g.E))[0, :, 3].exp()
        X = (torch.rand(chains, n) < bp_node).double()
        acc_node = torch.zeros(n); acc_pair = torch.zeros(g.E); cnt = 0
        for sweep in range(sweeps):
            for gr in groups:
                X[:, gr] = (torch.rand(chains, len(gr)) < torch.sigmoid(h[gr] + X @ W[:, gr])).double()
            if sweep >= burn:
                acc_node += X.mean(0); acc_pair += (X[:, g.u] * X[:, g.v]).mean(0); cnt += 1
    gn, gp = acc_node / cnt, acc_pair / cnt
    lor = lambda p11, pa, pb: torch.log(p11.clamp_min(1e-4) * (1 - pa - pb + p11).clamp_min(1e-4)
                                        / ((pa - p11).clamp_min(1e-4) * (pb - p11).clamp_min(1e-4)))
    lg = lor(gp, gn[g.u], gn[g.v]); lb = lor(bp_pair, bp_node[g.u], bp_node[g.v])
    return {"chains": chains, "sweeps": sweeps, "burn_in": burn, "colours": len(groups),
            "bp_residual": g.last_residual, "bp_iters": g.last_iters,
            "node_mean_abs_diff": float((gn - bp_node).abs().mean()), "node_max_abs_diff": float((gn - bp_node).abs().max()),
            "pair11_mean_abs_diff": float((gp - bp_pair).abs().mean()), "pair11_max_abs_diff": float((gp - bp_pair).abs().max()),
            "log_or_corr_gibbs_vs_bp": float(np.corrcoef(lg.numpy(), lb.numpy())[0, 1]),
            "strong_edge_agreement": float(((lg.abs() > STRONG) == (lb.abs() > STRONG)).double().mean())}


def gibbs_probs(g, h, J, lit_lists, chains=1000, sweeps=700, burn=200, seed=SEED + 1):
    """P(conjunction) under the fitted MRF itself, by chromatic Gibbs sampling (no BP)."""
    torch.manual_seed(seed)
    n = g.n
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(g.edges)
    col = nx.greedy_color(G, strategy="largest_first")
    groups = [torch.tensor([v for v in range(n) if col[v] == c]) for c in range(max(col.values()) + 1)]
    W = torch.zeros(n, n)
    W[g.u, g.v] = J; W[g.v, g.u] = J
    by_len = defaultdict(list)
    for k, L in enumerate(lit_lists):
        by_len[len(L)].append(k)
    idx = {m: torch.tensor([[i for i, _ in lit_lists[k]] for k in ks]) for m, ks in by_len.items()}
    val = {m: torch.tensor([[x for _, x in lit_lists[k]] for k in ks], dtype=torch.bool) for m, ks in by_len.items()}
    acc = {m: torch.zeros(len(ks)) for m, ks in by_len.items()}
    cnt = 0
    with torch.no_grad():
        X = (torch.rand(chains, n) < torch.sigmoid(h)).float()
        W32, h32 = W.float(), h.float()
        for sweep in range(sweeps):
            for gr in groups:
                X[:, gr] = (torch.rand(chains, len(gr)) < torch.sigmoid(h32[gr] + X @ W32[:, gr])).float()
            if sweep >= burn:
                for m in by_len:
                    lit = X[:, idx[m]]  # (chains, q, m)
                    lit = torch.where(val[m][None], lit, 1 - lit)
                    acc[m] += lit.prod(-1).mean(0).double()
                cnt += 1
    out = np.zeros(len(lit_lists))
    for m, ks in by_len.items():
        out[ks] = (acc[m] / cnt).numpy()
    return out


# --- local projection ------------------------------------------------------------

def local_projection(units: list[tuple[int, list]]):
    """units: (k events, [(lits over local indices, stated)]) -> fitted atom tables (numpy)."""
    out = []
    for k in sorted({u[0] for u in units}):
        idx = [r for r, u in enumerate(units) if u[0] == k]
        K = 2 ** k
        rows, qs, ss = [], [], []
        for r_local, r in enumerate(idx):
            for L, s in units[r][1]:
                mask = np.ones(K, dtype=bool)
                for i, x in L:
                    mask &= (((np.arange(K) >> (k - 1 - i)) & 1) == x)
                rows.append(r_local); qs.append(mask); ss.append(clip(s))
        A = torch.tensor(np.array(qs), dtype=torch.float64)
        R = torch.tensor(rows); s = torch.tensor(ss)
        z = torch.zeros(len(idx), K, requires_grad=True)
        opt = torch.optim.LBFGS([z], lr=1, max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            q = torch.softmax(z, -1)
            p = (A * q[R]).sum(-1).clamp(1e-9, 1 - 1e-9)
            loss = (s * torch.log(s / p) + (1 - s) * torch.log((1 - s) / (1 - p))).sum() \
                + 1e-4 * (q * torch.log(q.clamp_min(1e-300))).sum()
            loss.backward()
            return loss
        opt.step(closure)
        q = torch.softmax(z, -1).detach().numpy()
        for r_local, r in enumerate(idx):
            out.append((r, q[r_local]))
    res = [None] * len(units)
    for r, q in out:
        res[r] = q
    return res


def table_prob(q, k, L):
    K = 2 ** k
    mask = np.ones(K, dtype=bool)
    for i, x in L:
        mask &= (((np.arange(K) >> (k - 1 - i)) & 1) == x)
    return float(q[mask].sum())


# --- helpers ---------------------------------------------------------------------

def log_or_from(pa, pb, pab, eps=0.005):
    c11, c10, c01 = pab, pa - pab, pb - pab
    c00 = 1 - pa - pb + pab
    coherent = min(c11, c10, c01, c00) > 0
    c = np.clip([c00, c01, c10, c11], eps, None)
    c = c / c.sum()
    return float(math.log(c[3] * c[0] / (c[1] * c[2]))), coherent


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def boot_mean_diff(per_unit_a, per_unit_b, rng, n=2000):
    a, b = np.asarray(per_unit_a), np.asarray(per_unit_b)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return {"a": float(a.mean()), "b": float(b.mean()), "diff": float(d.mean()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "units": int(len(d))}


def boot_ci(x, rng, n=2000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return None
    idx = rng.integers(0, len(x), (n, len(x)))
    return [float(np.percentile(x[idx].mean(1), 2.5)), float(np.percentile(x[idx].mean(1), 97.5))]


def boot_corr(a, b, rng, fn=spearman, n=1000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 10:
        return None
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        v = fn(a[idx], b[idx])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def price_series(store, events, t_ms, days=60):
    out = {}
    for e in events:
        hist = store.price_history(e["id"], t_ms, points=days + 1, start=t_ms - days * DAY)
        ps = np.array([np.nan if r["prob"] is None else r["prob"] for r in hist], float)
        out[e["index"]] = ps
    return out


def grid_point(job):
    torch.set_num_threads(THREADS)
    n, train_q, marg, reg, lam, val_lits, val_stated = job
    g, h, J, info = fit_mrf(n, train_q, marg, reg, lam)
    pv = predict(g, h, J, val_lits)
    info["val_mean_kl"] = float(np.mean([kl(s_, p) for s_, p in zip(val_stated, pv)]))
    return info


def fit_params(job):
    torch.set_num_threads(THREADS)
    n, train_q, marg, reg, lam = job[:5]
    g, h, J, info = fit_mrf(n, train_q, marg, reg, lam, couplings=job[5] if len(job) > 5 else True)
    return Graph(n, g.edges), h, J, info


# --- main -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--charts", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="smaller lambda grid and Gibbs run")
    ap.add_argument("--procs", type=int, default=6, help="parallel fits for the lambda grid")
    a = ap.parse_args()
    t_start = time.time()
    rng_py = random.Random(SEED)
    rng = np.random.default_rng(SEED)
    events = [json.loads(l) for l in (a.data / "events.jsonl").read_text().splitlines()]
    queries = [json.loads(l) for l in (a.data / "queries.jsonl").read_text().splitlines()]
    graph = json.loads((a.data / "graph.json").read_text())
    log = {}
    for l in (a.data / "log.jsonl").read_text().splitlines():
        r = json.loads(l); log[r["query_id"]] = r
    n = len(events)
    y = [e["label"] for e in events]
    cos_of = {}
    for (i, j), c in zip(graph["sim_pairs"], graph["sim_cosine"]):
        cos_of[(i, j)] = c
    for (i, j), c in zip(graph["rand_pairs"], graph["rand_cosine"]):
        cos_of[(i, j)] = c
    Q = []
    for q in queries:
        r = log.get(q["query_id"])
        Q.append({**{k: q[k] for k in ("query_id", "kind", "pair_type", "events", "formula_str", "label")},
                  "lits": lits(q["formula"]), "stated": None if r is None else r["p"]})
    missing = sum(q["stated"] is None for q in Q)
    marg = [None] * n
    for q in Q:
        if q["kind"] == "single":
            marg[q["events"][0]] = q["stated"]
    assert all(m is not None for m in marg), "missing single answers"

    # units: pairs (and2 + andnot) and triples
    unit_of = {}
    for k, q in enumerate(Q):
        if q["kind"] in ("and2", "andnot"):
            i, j = sorted(q["events"])
            unit_of[k] = ("pair", q["pair_type"], i, j)
        elif q["kind"] == "and3":
            unit_of[k] = ("tri", "sim", *sorted(q["events"]))
    units = sorted(set(unit_of.values()))
    strata = defaultdict(list)
    for u in units:
        strata[(u[0], u[1])].append(u)
    held, val = set(), set()
    for key, us in sorted(strata.items()):
        us = sorted(us); rng_py.shuffle(us)
        nh = round(0.2 * len(us)); nv = round(0.15 * (len(us) - nh))
        held |= set(us[:nh]); val |= set(us[nh:nh + nv])

    def qset(filter_fn):
        return [k for k, q in enumerate(Q) if q["stated"] is not None and filter_fn(k, q)]
    base = lambda k, q: q["kind"] == "single"
    train_idx = qset(lambda k, q: base(k, q) or (k in unit_of and unit_of[k] not in held and unit_of[k] not in val))
    val_idx = qset(lambda k, q: k in unit_of and unit_of[k] in val)
    trainval_idx = qset(lambda k, q: base(k, q) or (k in unit_of and unit_of[k] not in held))
    held_idx = qset(lambda k, q: k in unit_of and unit_of[k] in held)
    all_idx = qset(lambda k, q: True)
    tq = lambda idx: [(Q[k]["lits"], Q[k]["stated"]) for k in idx]
    print(f"{n} events, {len(Q)} queries ({missing} missing); train {len(train_idx)} val {len(val_idx)} "
          f"held {len(held_idx)}", flush=True)

    # 1. lambda selection on validation
    grid = [(r, l) for r, ls in LAMBDAS.items() for l in ls]
    if a.quick:
        grid = [("l2", 0.03), ("l2", 0.3), ("l1", 0.03)]
    val_lits = [Q[k]["lits"] for k in val_idx]
    val_stated = [Q[k]["stated"] for k in val_idx]
    jobs = [(n, tq(train_idx), marg, reg_, lam_, val_lits, val_stated) for reg_, lam_ in grid]
    with ProcessPoolExecutor(min(len(jobs), a.procs), mp_context=mp.get_context("spawn")) as ex:
        sel = list(ex.map(grid_point, jobs))
    for info in sel:
        print(f"  {info['reg']} {info['lam']}: train kl {info['mean_kl']:.4f} val kl {info['val_mean_kl']:.4f} "
              f"bp resid {info['bp_residual']:.1e} ({info['seconds']}s)", flush=True)
    sel = [r for r in sel if r["bp_residual"] < 1e-6] or sel  # never choose a fit whose BP did not converge
    # largest penalty within 5% of the best validation KL (a one-standard-error style rule: the
    # first run chose L2 0.03, whose BP stopped converging on the full query set, while 0.1 scored
    # 4% worse on validation and converges in ~50 iterations)
    best_kl = min(r["val_mean_kl"] for r in sel)
    near = [r for r in sel if r["val_mean_kl"] <= 1.05 * best_kl]
    best = max(near, key=lambda r: (r["lam"] if r["reg"] == "l2" else r["lam"] * 0.5, -r["val_mean_kl"]))
    reg, lam = best["reg"], best["lam"]
    print(f"chosen {reg} lambda={lam}", flush=True)

    # 2. held-out evaluation (fit on train + val)
    sens_lams = [l_ for l_ in LAMBDAS["l2"] if l_ != lam and l_ >= 0.03] if reg == "l2" else []
    with ProcessPoolExecutor(min(a.procs, 4 + len(sens_lams)), mp_context=mp.get_context("spawn")) as ex:
        f_h = ex.submit(fit_params, (n, tq(trainval_idx), marg, reg, lam))
        f_all = ex.submit(fit_params, (n, tq(all_idx), marg, reg, lam))
        f_h0 = ex.submit(fit_params, (n, tq(trainval_idx), marg, reg, lam, False))
        f_all0 = ex.submit(fit_params, (n, tq(all_idx), marg, reg, lam, False))
        f_sens = {l_: ex.submit(fit_params, (n, tq(all_idx), marg, "l2", l_)) for l_ in sens_lams}
        g_h, h_h, J_h, info_h = f_h.result()
        g_all_fit = f_all.result()
        g_h0, h_h0, J_h0, info_h0 = f_h0.result()
        g_a0, h_a0, J_a0, info_a0 = f_all0.result()
        sens_fits = {l_: f.result() for l_, f in f_sens.items()}
    print(f"marginals-only fits: held {info_h0['mean_kl']:.4f}, all {info_a0['mean_kl']:.4f}", flush=True)
    held_lits = [Q[k]["lits"] for k in held_idx]
    p_fit_held = predict(g_h, h_h, J_h, held_lits)
    p_fit_held_gibbs = gibbs_probs(g_h, h_h, J_h, held_lits)
    p_h0_held = predict(g_h0, h_h0, J_h0, held_lits)
    # also a zero-coupling (independence) prediction for each held-out query
    def indep(L, m):
        p = 1.0
        for i, x in L:
            p *= clip(m[i]) if x else 1 - clip(m[i])
        return p
    by_unit = defaultdict(lambda: defaultdict(list))
    for k, pf, pg, p0 in zip(held_idx, p_fit_held, p_fit_held_gibbs, p_h0_held):
        q = Q[k]
        pi = indep(q["lits"], marg)
        u = unit_of[k]
        key = "triple" if u[0] == "tri" else f"pair/{u[1]}"
        d = by_unit[key][u]
        d.append({"kl_fit": kl(q["stated"], pf), "kl_ind": kl(q["stated"], pi),
                  "ll_fit": ll(pf, q["label"]), "ll_ind": ll(pi, q["label"]), "ll_raw": ll(q["stated"], q["label"]),
                  "kl_gibbs": kl(q["stated"], pg), "ll_gibbs": ll(pg, q["label"]),
                  "kl_h0": kl(q["stated"], p0), "ll_h0": ll(p0, q["label"]), "br_h0": brier(p0, q["label"]),
                  "br_fit": brier(pf, q["label"]), "br_ind": brier(pi, q["label"]), "br_raw": brier(q["stated"], q["label"]),
                  "on_edge": len(q["lits"]) < 3 and (min(q["events"]), max(q["events"])) in g_h.eid,
                  "stated": q["stated"], "fit": float(pf), "ind": pi})
    heldout = {}
    for key in sorted(by_unit) + ["all"]:
        us = [v for kk in by_unit for v in by_unit[kk].values()] if key == "all" else list(by_unit[key].values())
        mean = lambda f: [float(np.mean([r[f] for r in u])) for u in us]
        heldout[key] = {
            "units": len(us), "queries": sum(len(u) for u in us),
            "share_on_fitted_edge": float(np.mean([r["on_edge"] for u in us for r in u])),
            "kl_to_stated_fit_vs_indep": boot_mean_diff(mean("kl_fit"), mean("kl_ind"), rng),
            "logloss_fit_vs_indep": boot_mean_diff(mean("ll_fit"), mean("ll_ind"), rng),
            "logloss_fit_vs_raw": boot_mean_diff(mean("ll_fit"), mean("ll_raw"), rng),
            "gibbs_kl_to_stated_fit_vs_indep": boot_mean_diff(mean("kl_gibbs"), mean("kl_ind"), rng),
            "kl_to_stated_fit_vs_marginals_only_fit": boot_mean_diff(mean("kl_fit"), mean("kl_h0"), rng),
            "logloss_fit_vs_marginals_only_fit": boot_mean_diff(mean("ll_fit"), mean("ll_h0"), rng),
            "gibbs_logloss_fit_vs_indep": boot_mean_diff(mean("ll_gibbs"), mean("ll_ind"), rng),
            "brier": {"fit": float(np.mean(mean("br_fit"))), "indep": float(np.mean(mean("br_ind"))),
                      "marginals_only_fit": float(np.mean(mean("br_h0"))),
                      "raw": float(np.mean(mean("br_raw")))},
            "corr_stated_fit": pearson([r["stated"] for u in us for r in u], [r["fit"] for u in us for r in u]),
            "corr_stated_indep": pearson([r["stated"] for u in us for r in u], [r["ind"] for u in us for r in u]),
        }
        print(f"held {key}: KL fit {heldout[key]['kl_to_stated_fit_vs_indep']['a']:.4f} vs marginals-only fit "
              f"{heldout[key]['kl_to_stated_fit_vs_marginals_only_fit']['b']:.4f} vs indep "
              f"{heldout[key]['kl_to_stated_fit_vs_indep']['b']:.4f}; ll fit/ind/raw "
              f"{heldout[key]['logloss_fit_vs_indep']['a']:.4f}/{heldout[key]['logloss_fit_vs_indep']['b']:.4f}/"
              f"{heldout[key]['logloss_fit_vs_raw']['b']:.4f}", flush=True)
    # held-out pairs split by whether the pair's events share a neighbour (a path of length 2 in the fit)
    nbr = defaultdict(set)
    for (i, j) in g_h.edges:
        nbr[i].add(j); nbr[j].add(i)
    path2 = {"with_common_neighbour": [], "without": []}
    for key in ("pair/sim", "pair/rand"):
        for u, rows in by_unit.get(key, {}).items():
            i, j = u[2], u[3]
            tag = "with_common_neighbour" if nbr[i] & nbr[j] else "without"
            path2[tag].append((float(np.mean([r["kl_fit"] for r in rows])), float(np.mean([r["kl_ind"] for r in rows]))))
    heldout["pairs_by_common_neighbour"] = {k: boot_mean_diff([x[0] for x in v], [x[1] for x in v], rng) if len(v) > 2 else None
                                            for k, v in path2.items()}

    # 3. final fit on all queries
    g, h, J, info_all = g_all_fit
    print(f"final fit: {info_all}", flush=True)
    gib = gibbs_check(g, h, J, chains=500 if a.quick else 1000, sweeps=200 if a.quick else 600,
                      burn=50 if a.quick else 200)
    print(f"gibbs check: {gib}", flush=True)
    all_lits = [Q[k]["lits"] for k in all_idx]
    p_global = dict(zip(all_idx, predict(g, h, J, all_lits)))
    p_gibbs = dict(zip(all_idx, gibbs_probs(g, h, J, all_lits)))
    p_h0 = dict(zip(all_idx, predict(g_a0, h_a0, J_a0, all_lits)))
    # lambda sensitivity of the structure (same data, other L2 penalties)
    sens = []
    for l_, (gs_, hs_, Js_, inf_) in sorted(sens_fits.items()) + [(lam, (g, h, J, info_all))]:
        gc = gibbs_check(gs_, hs_, Js_, chains=500, sweeps=400, burn=150)
        with torch.no_grad():
            Hs, mus = gs_.bp(hs_[None, :], Js_)
            lps = gs_.pair_logp(Hs, mus, Js_, torch.arange(gs_.E))[0].exp().numpy()
        lor_s = np.log(lps[:, 3] * lps[:, 0] / (lps[:, 1] * lps[:, 2]))
        simmask = np.array([e in {tuple(p_) for p_ in graph["sim_pairs"]} for e in gs_.edges])
        sens.append({"lam": l_, "bp_residual": inf_["bp_residual"], "bp_iters": inf_["bp_iters"], "mean_kl_all": inf_["mean_kl"],
                     "gibbs_node_mean_abs_diff": gc["node_mean_abs_diff"], "gibbs_strong_edge_agreement": gc["strong_edge_agreement"],
                     "frac_abs_lor_gt_0.5_sim": float(np.mean(np.abs(lor_s[simmask]) > STRONG)),
                     "frac_abs_lor_gt_0.5_rand": float(np.mean(np.abs(lor_s[~simmask]) > STRONG)),
                     "strong_edges": int((np.abs(lor_s) > STRONG).sum()),
                     "mean_abs_J": float(Js_.abs().mean()), "max_abs_J": float(Js_.abs().max())})
        print(f"sensitivity lam {l_}: {sens[-1]}", flush=True)
    with torch.no_grad():
        H, mu = g.bp(h[None, :], J)
        fit_marg = torch.sigmoid(H[0]).numpy()
        lp = g.pair_logp(H, mu, J, torch.arange(g.E))[0].exp().numpy()  # 00 01 10 11
    fit_lor = {e: float(np.log(lp[k, 3] * lp[k, 0] / (lp[k, 1] * lp[k, 2]))) for k, e in enumerate(g.edges)}
    fit_p11 = {e: float(lp[k, 3]) for k, e in enumerate(g.edges)}

    # 4. local projection
    pair_q = defaultdict(list)
    tri_q = defaultdict(list)
    for k in all_idx:
        q = Q[k]
        if k in unit_of and unit_of[k][0] == "pair":
            pair_q[(unit_of[k][2], unit_of[k][3])].append(k)
    pkeys = sorted(pair_q)
    punits = []
    for (i, j) in pkeys:
        loc = {i: 0, j: 1}
        qs = [([(0, 1)], marg[i]), ([(1, 1)], marg[j])] + [([(loc[e], x) for e, x in Q[k]["lits"]], Q[k]["stated"]) for k in pair_q[(i, j)]]
        punits.append((2, qs))
    tkeys = sorted({unit_of[k][2:] for k in all_idx if k in unit_of and unit_of[k][0] == "tri"})
    tri_query = {unit_of[k][2:]: k for k in all_idx if k in unit_of and unit_of[k][0] == "tri"}
    for x in tkeys:
        loc = {e: r for r, e in enumerate(x)}
        qs = [([(r, 1)], marg[e]) for r, e in enumerate(x)]
        for a_, b_ in ((x[0], x[1]), (x[0], x[2]), (x[1], x[2])):
            for k in pair_q.get((a_, b_), []):
                qs.append(([(loc[e], v) for e, v in Q[k]["lits"]], Q[k]["stated"]))
        kq = tri_query[x]
        qs.append(([(loc[e], v) for e, v in Q[kq]["lits"]], Q[kq]["stated"]))
        punits.append((3, qs))
    tabs = local_projection(punits)
    p_local = {}
    local_marg = defaultdict(list)
    local_lor = {}
    for r, (i, j) in enumerate(pkeys):
        q = tabs[r]
        for k in pair_q[(i, j)]:
            loc = {i: 0, j: 1}
            p_local[k] = table_prob(q, 2, [(loc[e], x) for e, x in Q[k]["lits"]])
        local_marg[i].append(q[2] + q[3]); local_marg[j].append(q[1] + q[3])
        local_lor[(i, j)] = float(np.log(max(q[3], 1e-6) * max(q[0], 1e-6) / (max(q[1], 1e-6) * max(q[2], 1e-6))))
    for r2, x in enumerate(tkeys):
        q = tabs[len(pkeys) + r2]
        loc = {e: rr for rr, e in enumerate(x)}
        k = tri_query[x]
        p_local[k] = table_prob(q, 3, [(loc[e], v) for e, v in Q[k]["lits"]])
    for k in all_idx:
        if Q[k]["kind"] == "single":
            i = Q[k]["events"][0]
            p_local[k] = float(np.mean(local_marg[i])) if local_marg[i] else Q[k]["stated"]

    # 5. accuracy vs outcomes
    acc_rows = []
    kinds = [("single", lambda q: q["kind"] == "single"), ("and2/sim", lambda q: q["kind"] == "and2" and q["pair_type"] == "sim"),
             ("and2/rand", lambda q: q["kind"] == "and2" and q["pair_type"] == "rand"),
             ("andnot", lambda q: q["kind"] == "andnot"), ("and3", lambda q: q["kind"] == "and3"),
             ("all joints", lambda q: q["kind"] != "single"), ("all", lambda q: True)]
    for name, fn in kinds:
        ks = [k for k in all_idx if fn(Q[k])]
        yl = [Q[k]["label"] for k in ks]
        row = {"kind": name, "queries": len(ks), "yes_rate": float(np.mean(yl))}
        for m, src in (("raw", {k: Q[k]["stated"] for k in ks}), ("local", p_local), ("global", p_global),
                       ("global_gibbs", p_gibbs), ("marginals_only_fit", p_h0)):
            row[f"ll_{m}"] = float(np.mean([ll(src[k], Q[k]["label"]) for k in ks]))
            row[f"brier_{m}"] = float(np.mean([brier(src[k], Q[k]["label"]) for k in ks]))
            row[f"kl_to_stated_{m}"] = float(np.mean([kl(Q[k]["stated"], src[k]) for k in ks]))
        dd = np.array([ll(p_global[k], Q[k]["label"]) - ll(Q[k]["stated"], Q[k]["label"]) for k in ks])
        row["ll_gain_global_vs_raw_ci"] = boot_ci(dd, rng)
        dl = np.array([ll(p_global[k], Q[k]["label"]) - ll(p_local[k], Q[k]["label"]) for k in ks])
        row["ll_gain_global_vs_local_ci"] = boot_ci(dl, rng)
        acc_rows.append(row)
        print(f"acc {name:10} n={len(ks)} ll raw {row['ll_raw']:.4f} local {row['ll_local']:.4f} global "
              f"{row['ll_global']:.4f} | brier {row['brier_raw']:.4f} {row['brier_local']:.4f} {row['brier_global']:.4f}", flush=True)

    # 6. structure
    sim_pairs = {tuple(p) for p in graph["sim_pairs"]}
    stated_and = {}
    for k in all_idx:
        q = Q[k]
        if q["kind"] == "and2":
            stated_and[tuple(sorted(q["events"]))] = q["stated"]
    struct = {}
    pair_rows = []
    for (i, j), pab in sorted(stated_and.items()):
        slor, coh = log_or_from(clip(marg[i]), clip(marg[j]), clip(pab))
        pair_rows.append({"i": i, "j": j, "type": "sim" if (i, j) in sim_pairs else "rand", "cos": cos_of.get((i, j)),
                          "stated_lor": slor, "coherent": coh, "fit_lor": fit_lor[(i, j)], "local_lor": local_lor.get((i, j)),
                          "J": float(J[g.eid[(i, j)]]), "same_parent": events[i]["parent_id"] == events[j]["parent_id"]})
    for typ in ("sim", "rand", "sim_same_parent", "sim_other_parent"):
        rows = [r for r in pair_rows if (r["type"] == typ) or (typ == "sim_same_parent" and r["type"] == "sim" and r["same_parent"])
                or (typ == "sim_other_parent" and r["type"] == "sim" and not r["same_parent"])]
        if not rows:
            continue
        struct[typ] = {"pairs": len(rows),
                       "stated_frac_abs_lor_gt_0.5": float(np.mean([abs(r["stated_lor"]) > STRONG for r in rows])),
                       "stated_frac_incoherent_2x2": float(np.mean([not r["coherent"] for r in rows])),
                       "stated_frac_abs_lor_gt_0.5_coherent_only": float(np.mean([abs(r["stated_lor"]) > STRONG for r in rows if r["coherent"]] or [np.nan])),
                       "stated_median_abs_lor": float(np.median([abs(r["stated_lor"]) for r in rows])),
                       "fit_frac_abs_lor_gt_0.5": float(np.mean([abs(r["fit_lor"]) > STRONG for r in rows])),
                       "fit_frac_positive_strong": float(np.mean([r["fit_lor"] > STRONG for r in rows])),
                       "fit_frac_negative_strong": float(np.mean([r["fit_lor"] < -STRONG for r in rows])),
                       "fit_median_abs_lor": float(np.median([abs(r["fit_lor"]) for r in rows])),
                       "local_frac_abs_lor_gt_0.5": float(np.mean([abs(r["local_lor"]) > STRONG for r in rows if r["local_lor"] is not None])),
                       "frac_abs_J_gt_0.5": float(np.mean([abs(r["J"]) > STRONG for r in rows])),
                       "frac_J_zeroish": float(np.mean([abs(r["J"]) < 0.05 for r in rows])),
                       "corr_stated_fit_lor": pearson([r["stated_lor"] for r in rows], [r["fit_lor"] for r in rows])}
    # by cosine bin (all pairs)
    bins = [(-1, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 1.01)]
    struct["by_cosine"] = []
    for lo_, hi_ in bins:
        rows = [r for r in pair_rows if r["cos"] is not None and lo_ <= r["cos"] < hi_]
        if rows:
            struct["by_cosine"].append({"cos": [lo_, hi_], "pairs": len(rows), "sim_share": float(np.mean([r["type"] == "sim" for r in rows])),
                                        "stated_frac_strong": float(np.mean([abs(r["stated_lor"]) > STRONG for r in rows])),
                                        "fit_frac_strong": float(np.mean([abs(r["fit_lor"]) > STRONG for r in rows])),
                                        "fit_mean_lor": float(np.mean([r["fit_lor"] for r in rows]))})
    # graphs
    def graph_stats(edge_list, name):
        G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edge_list)
        comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
        nontriv = [c for c in comps if c > 1]
        tw_md, _ = nx.algorithms.approximation.treewidth_min_degree(G)
        try:
            tw_mf, _ = nx.algorithms.approximation.treewidth_min_fill_in(G) if G.number_of_edges() < 3000 else (None, None)
        except Exception:
            tw_mf = None
        largest = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        out = {"graph": name, "edges": G.number_of_edges(), "nodes_with_edges": sum(1 for v in G if G.degree(v) > 0),
               "components_nontrivial": len(nontriv), "isolated_nodes": comps.count(1),
               "largest_component": comps[0], "component_sizes_top10": nontriv[:10],
               "component_size_hist": dict(sorted(Counter(nontriv).items())),
               "treewidth_upper_min_degree": tw_md, "treewidth_upper_min_fill": tw_mf,
               "max_clique": max((len(c) for c in nx.find_cliques(G)), default=1) if G.number_of_edges() < 20000 else None,
               "mean_degree": 2 * G.number_of_edges() / n}
        if G.number_of_edges():
            comm = nx.community.louvain_communities(G, seed=SEED)
            sizes = sorted((len(c) for c in comm if len(c) > 1), reverse=True)
            out.update(louvain_communities=len(sizes), louvain_sizes_top10=sizes[:10],
                       modularity=float(nx.community.modularity(G, comm)))
        return out, G
    struct["graphs"] = []
    gs_all, _ = graph_stats(g.edges, "queried (all fitted edges)")
    struct["graphs"].append(gs_all)
    gs_sim, _ = graph_stats([e for e in g.edges if e in sim_pairs], "queried similar pairs")
    struct["graphs"].append(gs_sim)
    strong_fit = [e for e in g.edges if abs(fit_lor[e]) > STRONG]
    gs_strong, G_strong = graph_stats(strong_fit, "strong dependence (fit |log OR| > 0.5)")
    struct["graphs"].append(gs_strong)
    strong_st = [(r["i"], r["j"]) for r in pair_rows if abs(r["stated_lor"]) > STRONG]
    gs_st, _ = graph_stats(strong_st, "strong dependence (stated |log OR| > 0.5)")
    struct["graphs"].append(gs_st)
    strong_J = [e for e in g.edges if abs(float(J[g.eid[e]])) > STRONG]
    gs_J, _ = graph_stats(strong_J, "strong couplings (|J| > 0.5)")
    struct["graphs"].append(gs_J)
    for r in struct["graphs"]:
        print(f"graph {r['graph']}: {r['edges']} edges, {r['components_nontrivial']} comps, largest {r['largest_component']}, "
              f"tw<= {r['treewidth_upper_min_degree']}/{r['treewidth_upper_min_fill']}", flush=True)

    # 7. dependencies vs prices and outcomes
    store = AsOf()
    t_ms = to_ms(T_ISO)
    series = price_series(store, events, t_ms)
    def lo(p):
        p = np.clip(p, 0.01, 0.99)
        return np.log(p / (1 - p))
    rets = {}
    for i, ps in series.items():
        if np.isnan(ps).any():
            continue
        d = np.diff(lo(ps))
        if (np.abs(d) > 1e-9).sum() >= 5:
            rets[i] = d
    price_rows = []
    for r in pair_rows:
        if r["i"] in rets and r["j"] in rets:
            c = pearson(rets[r["i"]], rets[r["j"]])
            if c is not None:
                price_rows.append({**r, "price_corr": c})
    dep = {"events_with_price_moves": len(rets), "window_days": 60, "returns": "daily change in log-odds, prob clipped [0.01, 0.99]",
           "min_nonzero_days": 5}
    for typ in ("all", "sim", "rand"):
        rows = [r for r in price_rows if typ == "all" or r["type"] == typ]
        if len(rows) < 10:
            continue
        pc = [r["price_corr"] for r in rows]
        d = {"pairs": len(rows),
             "spearman_price_vs_stated_lor": spearman(pc, [r["stated_lor"] for r in rows]),
             "spearman_price_vs_stated_lor_ci": boot_corr(pc, [r["stated_lor"] for r in rows], rng),
             "spearman_price_vs_fit_lor": spearman(pc, [r["fit_lor"] for r in rows]),
             "spearman_price_vs_fit_lor_ci": boot_corr(pc, [r["fit_lor"] for r in rows], rng),
             "spearman_price_vs_local_lor": spearman(pc, [r["local_lor"] for r in rows]),
             "spearman_price_vs_cosine": spearman(pc, [r["cos"] for r in rows]),
             "frac_abs_price_corr_gt_0.3": float(np.mean([abs(x) > 0.3 for x in pc]))}
        strong_px = [r for r in rows if abs(r["price_corr"]) > 0.3]
        if strong_px:
            d["sign_agreement_fit_when_abs_price_corr_gt_0.3"] = float(np.mean([np.sign(r["fit_lor"]) == np.sign(r["price_corr"]) for r in strong_px]))
            d["sign_agreement_stated_when_abs_price_corr_gt_0.3"] = float(np.mean([np.sign(r["stated_lor"]) == np.sign(r["price_corr"]) for r in strong_px]))
            d["pairs_abs_price_corr_gt_0.3"] = len(strong_px)
        dep[typ] = d
        print(f"price {typ}: n={len(rows)} rho(price, stated lor)={d['spearman_price_vs_stated_lor']:.3f} "
              f"rho(price, fit lor)={d['spearman_price_vs_fit_lor']:.3f}", flush=True)
    # realised joint outcomes: 4-cell log loss of the fit's pair table vs independence of fitted marginals
    joint = {}
    for typ in ("sim", "rand", "sim_strong_fit", "rand_strong_fit"):
        base_t = typ.split("_")[0]
        rows = [r for r in pair_rows if r["type"] == base_t and (not typ.endswith("strong_fit") or abs(r["fit_lor"]) > STRONG)]
        a_, b_ = [], []
        for r in rows:
            i, j = r["i"], r["j"]
            k = g.eid[(i, j)]
            cell = 2 * y[i] + y[j]
            a_.append(-math.log(max(lp[k, cell], 1e-6)))
            pi_, pj_ = fit_marg[i], fit_marg[j]
            ind = [(1 - pi_) * (1 - pj_), (1 - pi_) * pj_, pi_ * (1 - pj_), pi_ * pj_][cell]
            b_.append(-math.log(max(ind, 1e-6)))
        if len(rows) > 2:
            joint[typ] = {**boot_mean_diff(a_, b_, rng), "meaning": "a = fit pair table, b = product of fitted marginals; "
                          "4-cell log loss of the realised (A, B) outcome"}
    # realised co-outcome rate by fitted dependence sign
    co = {}
    for name_, fn in (("fit_lor>0.5", lambda r: r["fit_lor"] > STRONG), ("fit_lor<-0.5", lambda r: r["fit_lor"] < -STRONG),
                      ("|fit_lor|<=0.5", lambda r: abs(r["fit_lor"]) <= STRONG)):
        rows = [r for r in pair_rows if fn(r)]
        if rows:
            obs = [y[r["i"]] * y[r["j"]] for r in rows]
            exp_ind = [fit_marg[r["i"]] * fit_marg[r["j"]] for r in rows]
            exp_fit = [fit_p11[(r["i"], r["j"])] for r in rows]
            co[name_] = {"pairs": len(rows), "both_yes_observed": float(np.mean(obs)), "both_yes_fit": float(np.mean(exp_fit)),
                         "both_yes_indep": float(np.mean(exp_ind)),
                         "realised_phi": pearson([y[r["i"]] for r in rows], [y[r["j"]] for r in rows])}
    dep["realised_joint_logloss"] = joint
    dep["co_outcome_by_fit_dependence"] = co

    # 8. outputs
    out = {"t": T_ISO, "events": n, "yes_rate": float(np.mean(y)), "queries": len(Q), "missing": missing,
           "queries_by_kind": dict(Counter(f"{q['kind']}/{q['pair_type']}" for q in Q)),
           "calls": {"n": len(log), "retried": sum(r["retried"] for r in log.values()),
                     "cost_usd_cap_price": sum(r["cost"] for r in log.values()),
                     "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) + (r.get("usage_retry") or {}).get("prompt_tokens", 0) for r in log.values()),
                     "completion_tokens": sum(r["usage"].get("completion_tokens", 0) + (r.get("usage_retry") or {}).get("completion_tokens", 0) for r in log.values()),
                     "served_models": dict(Counter(r["served_model"] for r in log.values()))},
           "split": {"train": len(train_idx), "validation": len(val_idx), "held_out": len(held_idx),
                     "held_units": dict(Counter(f"{u[0]}/{u[1]}" for u in held))},
           "lambda_selection": sel, "chosen": {"reg": reg, "lam": lam},
           "fit_heldout": info_h, "fit_all": info_all, "gibbs_check": gib,
           "marginals_only_fit": {"heldout": info_h0, "all": info_a0}, "structure_lambda_sensitivity": sens,
           "heldout": heldout, "accuracy": acc_rows, "structure": struct, "dependence_vs_prices_and_outcomes": dep,
           "seconds": round(time.time() - t_start, 1)}
    a.out_json.write_text(json.dumps(out, indent=1))
    with (a.data / "fit-per-query.jsonl").open("w") as f:
        for k in all_idx:
            q = Q[k]
            f.write(json.dumps({"query_id": q["query_id"], "kind": q["kind"], "pair_type": q["pair_type"],
                                "formula": q["formula_str"], "label": q["label"], "stated": q["stated"],
                                "local": p_local.get(k), "global": float(p_global[k]), "global_gibbs": float(p_gibbs[k]),
                                "held_out": k in held_idx}) + "\n")
    with (a.data / "fit-pairs.jsonl").open("w") as f:
        for r in pair_rows:
            f.write(json.dumps(r) + "\n")
    torch.save({"edges": g.edges, "h": h, "J": J}, a.data / "fit-params.pt")

    # chart json: dependence network (strong fitted edges), positions from a spring layout
    Gs = nx.Graph(); Gs.add_nodes_from(range(n))
    for e in strong_fit:
        Gs.add_edge(*e, weight=abs(fit_lor[e]))
    pos = nx.spring_layout(Gs, seed=SEED, k=0.08, iterations=200)
    comm = nx.community.louvain_communities(Gs, seed=SEED) if Gs.number_of_edges() else []
    comm_of = {v: c for c, vs in enumerate(sorted(comm, key=len, reverse=True)) for v in vs}
    nodes = [{"id": e["index"], "label": (e["parent_question"][:70] + (f" — {e['answer_text'][:40]}" if e["answer_text"] else "")),
              "parent": e["parent_id"], "stated": marg[e["index"]], "fit": float(fit_marg[e["index"]]), "outcome": e["label"],
              "degree_strong": Gs.degree(e["index"]), "community": comm_of.get(e["index"]),
              "x": float(pos[e["index"]][0]), "y": float(pos[e["index"]][1])} for e in events]
    edges_c = [{"source": i, "target": j, "fit_lor": round(fit_lor[(i, j)], 3), "J": round(float(J[g.eid[(i, j)]]), 3),
                "type": "sim" if (i, j) in sim_pairs else "rand", "cos": cos_of.get((i, j)),
                "same_parent": events[i]["parent_id"] == events[j]["parent_id"]} for i, j in strong_fit]
    charts = {"dependence_network": {"description": f"nodes = {n} events at {T_ISO}; edges = queried pairs with |fitted log odds "
                                                    f"ratio| > {STRONG}; x/y = spring layout (seed {SEED}); community = Louvain",
                                     "nodes": nodes, "edges": edges_c},
              "structure_by_pair_type": {k: v for k, v in struct.items() if k in ("sim", "rand", "sim_same_parent", "sim_other_parent")},
              "structure_by_cosine": struct["by_cosine"], "graphs": struct["graphs"],
              "heldout": heldout, "accuracy": acc_rows,
              "lor_scatter_sample": [{k: r[k] for k in ("type", "stated_lor", "fit_lor", "cos")}
                                     for r in random.Random(SEED).sample(pair_rows, min(1500, len(pair_rows)))],
              "price_vs_lor_scatter": [{k: r[k] for k in ("type", "price_corr", "stated_lor", "fit_lor")} for r in price_rows],
              "lambda_selection": sel}
    a.charts.write_text(json.dumps(charts))
    print(f"done in {out['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
