"""Per-week global joint fits of Qwen3-32B's answers (scripts/weekly_worldmodel.py output).

Model per week (binary log-linear over the week's open events, x in {0,1}):
    log p(x) = sum_i h_i x_i + sum_links J_ij x_i x_j + sum_triangles K_ijk x_i x_j x_k - log Z
  M1  h only (independence; h fitted to every training answer, singles, pairs and triples)
  M2  h + J on every Luna link open that week
  M3  h + J + K on every triangle open that week
Fit: minimise sum_q KL_Bernoulli(stated_q || implied_q) + lam * (sum J^2 + sum K^2) over the training
answers (everything except the 20% held-out and2/and3 queries); stated values clipped to [0.01, 0.99];
missing answers left out, never imputed.  The objective separates over connected components of the
week's link graph, so each component is fitted on its own:
  exact junction tree (scripts/jt_exact.py) when the component's min-fill treewidth <= TW_MAX,
  else damped loopy BP on a factor graph (one factor per triangle carrying its K and a 1/n share of each
  of its links' J, n = triangles on that link; one pair factor per link in no triangle).  BP runs to a
  max message change < 1e-9 without gradients, then GRAD_ITERS differentiable iterations from the fixed
  point.  After the fit, BP is rerun from zero messages to check convergence (residual recorded; a fit
  whose BP does not converge is refitted with heavier damping and flagged), and a chromatic Gibbs sampler
  (1,000 chains) checks the BP singles, link P11 and triangle P111 at the fitted parameters.
M2b/M3b ("singles-anchored"): as M2/M3 but every stated single gets weight ANCHOR_W in the loss, so the
fitted marginals stay at the model's own stated singles and the joint answers are explained by J/K; this
stops an AND-bias in the joint answers from dragging the marginals.
Isolated events (no link): every model's fitted marginal is the stated one (KL optimum).
lam: chosen per model (M2, M3) on an inner 20% split of the training and2/and3 answers of the tuning
weeks (largest lam within 5% of the best inner KL), then fixed for every week.

Usage: weekly_worldmodel_fit.py tune|fit|score  [--data DIR] [--weeks ...] [--procs N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.approximation import treewidth_min_fill_in

sys.path.insert(0, str(Path(__file__).resolve().parent))
from example_week_fit import Model as JTModel  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
DATA = Path("/home/agent/data/weekly-worldmodel-20260927")
LO = 0.01
SEED = 20260927
TW_MAX = 12
DAMP = 0.5
GRAD_ITERS = 30
LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0]
MODELS = ("M1", "M2", "M3", "M2b", "M3b")
ANCHOR_W = 50.0


def clip(p):
    return min(max(p, LO), 1 - LO)


def logit(p):
    p = clip(p); return math.log(p / (1 - p))


def kl_terms(s, p):
    s = torch.as_tensor(s).clamp(LO, 1 - LO)
    return s * torch.log(s / p) + (1 - s) * torch.log((1 - s) / (1 - p))


# --- loopy BP on the triangle/pair factor graph -------------------------------------------------

X3 = torch.tensor(list(itertools.product([0, 1], repeat=3)), dtype=torch.float64)  # (8, 3)
X2 = torch.tensor(list(itertools.product([0, 1], repeat=2)), dtype=torch.float64)  # (4, 2)


class BPModel:
    """Same interface as example_week_fit.Model: .linked, .links, .tris, implied(h, J, K, qs), fit(qs, lam, h0)."""

    anchor = 1.0

    def __init__(self, links, triangles, use_k=True):
        self.linked = sorted({v for e in links for v in e})
        self.loc = {v: k for k, v in enumerate(self.linked)}
        self.links = [tuple(sorted(e)) for e in links]
        self.lid = {e: k for k, e in enumerate(self.links)}
        self.tris = [tuple(sorted(t)) for t in triangles] if use_k else []
        tri_all = [tuple(sorted(t)) for t in triangles]  # regions exist for every triangle (queries), K only in M3
        self.tri_all = tri_all
        self.tid = {t: k for k, t in enumerate(tri_all)}
        ntri = defaultdict(int)
        for t in tri_all:
            for e in itertools.combinations(t, 2):
                ntri[e] += 1
        self.n = len(self.linked)
        L = self.loc
        self.t_vars = torch.tensor([[L[v] for v in t] for t in tri_all], dtype=torch.long).reshape(-1, 3)
        # per triangle, its three link ids (slot pairs (0,1), (0,2), (1,2)) and share 1/n
        self.t_lid = torch.tensor([[self.lid[(t[a], t[b])] for a, b in ((0, 1), (0, 2), (1, 2))] for t in tri_all],
                                  dtype=torch.long).reshape(-1, 3)
        self.t_share = torch.tensor([[1.0 / ntri[(t[a], t[b])] for a, b in ((0, 1), (0, 2), (1, 2))] for t in tri_all]).reshape(-1, 3)
        self.p_links = [e for e in self.links if ntri[e] == 0]
        self.p_vars = torch.tensor([[L[e[0]], L[e[1]]] for e in self.p_links], dtype=torch.long).reshape(-1, 2)
        self.p_lid = torch.tensor([self.lid[e] for e in self.p_links], dtype=torch.long)
        # home region for every link query: first triangle containing it, else its pair factor
        self.link_home = {}
        for k, t in enumerate(tri_all):
            for a, b in ((0, 1), (0, 2), (1, 2)):
                self.link_home.setdefault((t[a], t[b]), ("t", k, (a, b)))
        for k, e in enumerate(self.p_links):
            self.link_home[e] = ("p", k, None)
        # indicator columns: triangle table = K*[111] + sum share*J*[x_a x_b]
        self.ind3 = torch.stack([X3[:, 0] * X3[:, 1], X3[:, 0] * X3[:, 2], X3[:, 1] * X3[:, 2]], 1)  # (8, 3)
        self.ind111 = X3.prod(1)
        self.mu3 = None  # warm start
        self.mu2 = None
        self.last_residual = None
        self.last_iters = 0
        self.damp = DAMP

    def tables(self, J, K):
        th3 = (J[self.t_lid] * self.t_share) @ self.ind3.T if len(self.tri_all) else torch.zeros(0, 8)
        if len(self.tris):
            th3 = th3 + K[:, None] * self.ind111[None, :]
        th2 = J[self.p_lid][:, None] * (X2[:, 0] * X2[:, 1])[None, :] if len(self.p_links) else torch.zeros(0, 4)
        return th3, th2

    def _step(self, h, th3, th2, m3, m2):
        """m3: (F3, 3) factor->var log-ratio messages, m2: (F2, 2)."""
        H = h.clone()
        H = H.index_add(0, self.t_vars.reshape(-1), m3.reshape(-1)) if len(th3) else H
        H = H.index_add(0, self.p_vars.reshape(-1), m2.reshape(-1)) if len(th2) else H
        new3, new2 = m3, m2
        if len(th3):
            nu = H[self.t_vars] - m3  # var->factor
            E = th3 + nu @ X3.T  # (F3, 8)
            outs = []
            for s in range(3):
                on = X3[:, s] == 1
                outs.append(torch.logsumexp(E[:, on] - nu[:, s:s + 1], 1) - torch.logsumexp(E[:, ~on], 1))
            new3 = torch.stack(outs, 1)
        if len(th2):
            nu = H[self.p_vars] - m2
            E = th2 + nu @ X2.T
            outs = []
            for s in range(2):
                on = X2[:, s] == 1
                outs.append(torch.logsumexp(E[:, on] - nu[:, s:s + 1], 1) - torch.logsumexp(E[:, ~on], 1))
            new2 = torch.stack(outs, 1)
        d = self.damp
        return d * m3 + (1 - d) * new3, d * m2 + (1 - d) * new2

    def run_bp(self, h, J, K, tol=1e-9, max_iters=4000, grad_iters=GRAD_ITERS, cold=False):
        th3, th2 = self.tables(J, K)
        m3 = torch.zeros(len(th3), 3) if (cold or self.mu3 is None) else self.mu3
        m2 = torch.zeros(len(th2), 2) if (cold or self.mu2 is None) else self.mu2
        with torch.no_grad():
            hd, t3d, t2d = h.detach(), th3.detach(), th2.detach()
            res, k = float("inf"), 0
            for k in range(max_iters):
                n3, n2 = self._step(hd, t3d, t2d, m3, m2)
                res = max(float((n3 - m3).abs().max()) if len(n3) else 0.0, float((n2 - m2).abs().max()) if len(n2) else 0.0)
                m3, m2 = n3, n2
                if res < tol:
                    break
            self.last_residual, self.last_iters = res, k + 1
            self.mu3, self.mu2 = m3.clone(), m2.clone()
        for _ in range(grad_iters):
            m3, m2 = self._step(h, th3, th2, m3, m2)
        H = h.index_add(0, self.t_vars.reshape(-1), m3.reshape(-1)) if len(th3) else h
        H = H.index_add(0, self.p_vars.reshape(-1), m2.reshape(-1)) if len(th2) else H
        b3 = b2 = None
        if len(th3):
            nu = H[self.t_vars] - m3
            b3 = torch.log_softmax(th3 + nu @ X3.T, 1)
        if len(th2):
            nu = H[self.p_vars] - m2
            b2 = torch.log_softmax(th2 + nu @ X2.T, 1)
        return H, b3, b2

    def _prepare(self, qs):
        if getattr(self, "_prep_key", None) == id(qs):
            return
        s_q, s_v, t2_q, t2_f, t2_c, p2_q, p2_f, t3_q, t3_f = [], [], [], [], [], [], [], [], []
        for k, q in enumerate(qs):
            ev = q["events"]
            if q["kind"] == "single":
                s_q.append(k); s_v.append(self.loc[ev[0]])
            elif q["kind"] == "and2":
                e = tuple(sorted(ev))
                kind, f, ab = self.link_home[e]
                if kind == "t":
                    # cells of the triangle table where both slots a, b are 1
                    t2_q.append(k); t2_f.append(f); t2_c.append(ab)
                else:
                    p2_q.append(k); p2_f.append(f)
            else:
                t3_q.append(k); t3_f.append(self.tid[tuple(sorted(ev))])
        T = lambda x: torch.tensor(x, dtype=torch.long)
        masks = {ab: (X3[:, ab[0]] * X3[:, ab[1]]).bool() for ab in ((0, 1), (0, 2), (1, 2))}
        self._prep = (T(s_q), T(s_v), T(t2_q), T(t2_f), [masks[ab] for ab in t2_c], T(p2_q), T(p2_f), T(t3_q), T(t3_f))
        # group the triangle-pair queries by slot pair for vectorisation
        g = defaultdict(list)
        for i, ab in enumerate(t2_c):
            g[ab].append(i)
        self._t2_groups = [(masks[ab], T([t2_q[i] for i in ix]), T([t2_f[i] for i in ix])) for ab, ix in g.items()]
        self._nq = len(qs)
        self._prep_key = id(qs)

    def implied(self, h, J, K, qs):
        self._prepare(qs)
        H, b3, b2 = self.run_bp(h, J, K)
        s_q, s_v, _, _, _, p2_q, p2_f, t3_q, t3_f = self._prep
        p = torch.zeros(self._nq)
        if len(s_q):
            p = p.index_put((s_q,), torch.sigmoid(H[s_v]))
        for mask, qq, ff in self._t2_groups:
            p = p.index_put((qq,), torch.logsumexp(b3[ff][:, mask], 1).exp())
        if len(p2_q):
            p = p.index_put((p2_q,), b2[p2_f, 3].exp())
        if len(t3_q):
            p = p.index_put((t3_q,), b3[t3_f, 7].exp())
        return p.clamp(1e-9, 1 - 1e-9)

    def fit(self, qs, lam, h0, use_j=True):
        h = torch.tensor(h0, requires_grad=True)
        J = torch.zeros(len(self.links), requires_grad=use_j)
        K = torch.zeros(len(self.tris), requires_grad=True)
        s = torch.tensor([clip(q["p"]) for q in qs])
        w = torch.tensor([self.anchor if q["kind"] == "single" else 1.0 for q in qs])
        params = [h] + ([J] if use_j else []) + ([K] if len(self.tris) else [])

        def loss():
            p = self.implied(h, J, K, qs)
            data = (w * kl_terms(s, p)).sum()
            return data + lam * ((J ** 2).sum() + (K ** 2).sum()), data
        opt = torch.optim.LBFGS(params, lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-11)

        def closure():
            opt.zero_grad(); tot, _ = loss(); tot.backward(); return tot
        opt.step(closure)
        with torch.no_grad():
            tot, data = loss()
        return h.detach(), J.detach(), K.detach(), float(data)

    def check(self, h, J, K):
        """Cold-start BP to convergence at the fitted parameters; returns residual, iterations."""
        with torch.no_grad():
            self.run_bp(h, J, K, cold=True, max_iters=20000, grad_iters=0)
        return self.last_residual, self.last_iters

    def gibbs(self, h, J, K, chains=600, sweeps=500, burn=150, seed=SEED):
        """P(node), P11(link), P111(triangle) under the fitted log-linear model itself, by chromatic Gibbs
        sampling (sparse local fields; no Bethe approximation), and their gaps to the BP beliefs."""
        torch.manual_seed(seed)
        n = self.n
        G = nx.Graph(); G.add_nodes_from(range(n))
        G.add_edges_from((self.loc[a], self.loc[b]) for a, b in self.links)
        col = nx.greedy_color(G, strategy="largest_first")
        groups = [torch.tensor([v for v in range(n) if col[v] == c]) for c in range(max(col.values()) + 1)]
        lu = torch.tensor([self.loc[a] for a, _ in self.links]); lv = torch.tensor([self.loc[b] for _, b in self.links])
        tv = torch.tensor([[self.loc[v] for v in t] for t in self.tris], dtype=torch.long).reshape(-1, 3)
        tall = self.t_vars
        with torch.no_grad():
            H, b3, b2 = self.run_bp(h, J, K, cold=True, grad_iters=0)
            X = (torch.rand(chains, n) < torch.sigmoid(H)).double()
            acc1 = torch.zeros(n); acc3 = torch.zeros(len(self.tri_all)); acc2 = torch.zeros(len(self.links)); cnt = 0
            for sweep in range(sweeps):
                for gr in groups:
                    field = h.expand(chains, n).clone()
                    field = field.index_add(1, lv, X[:, lu] * J).index_add(1, lu, X[:, lv] * J)
                    if len(tv):
                        for s_, (a, b) in enumerate(((1, 2), (0, 2), (0, 1))):
                            field = field.index_add(1, tv[:, s_], K[None, :] * X[:, tv[:, a]] * X[:, tv[:, b]])
                    X[:, gr] = (torch.rand(chains, len(gr)) < torch.sigmoid(field[:, gr])).double()
                if sweep >= burn:
                    acc1 += X.mean(0); acc2 += (X[:, lu] * X[:, lv]).mean(0)
                    if len(tall):
                        acc3 += X[:, tall].prod(-1).mean(0)
                    cnt += 1
            g1, g2, g3 = acc1 / cnt, acc2 / cnt, acc3 / cnt
            self._prep_key = None
            bp2 = self.implied(h, J, K, [{"kind": "and2", "events": list(e)} for e in self.links])
            self._prep_key = None
            bp1 = torch.sigmoid(H)
            bp3 = b3[:, 7].exp() if b3 is not None else torch.zeros(0)
        f = lambda x, y: (float((x - y).abs().mean()), float((x - y).abs().max())) if len(x) else (0.0, 0.0)
        est = {("single", (v,)): float(g1[self.loc[v]]) for v in self.linked}
        est.update({("and2", e): float(g2[k]) for k, e in enumerate(self.links)})
        est.update({("and3", t): float(g3[k]) for k, t in enumerate(self.tri_all)})
        return est, {"chains": chains, "sweeps": sweeps, "burn_in": burn, "node_mean_max_abs": f(g1, bp1),
                     "link11_mean_max_abs": f(g2, bp2), "tri111_mean_max_abs": f(g3, bp3)}


class AnchoredJT(JTModel):
    """example_week_fit.Model with per-query weights (singles weighted by self.anchor)."""
    anchor = 1.0

    def fit(self, qs, lam, h0):
        h = torch.tensor(h0, requires_grad=True)
        J = torch.zeros(len(self.links), requires_grad=True)
        K = torch.zeros(len(self.tris), requires_grad=True)
        s = torch.tensor([clip(q["p"]) for q in qs])
        w = torch.tensor([self.anchor if q["kind"] == "single" else 1.0 for q in qs])

        def loss():
            data = (w * kl_terms(s, self.implied(h, J, K, qs))).sum()
            return data + lam * ((J ** 2).sum() + (K ** 2).sum()), data
        opt = torch.optim.LBFGS([h, J, K], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-11)

        def closure():
            opt.zero_grad(); tot, _ = loss(); tot.backward(); return tot
        opt.step(closure)
        with torch.no_grad():
            tot, data = loss()
        return h.detach(), J.detach(), K.detach(), float(data)


class IndepModel:
    """M1: h only.  Implied probabilities are products of sigmoid(h)."""

    def __init__(self, links, triangles):
        self.linked = sorted({v for e in links for v in e})
        self.loc = {v: k for k, v in enumerate(self.linked)}
        self.links, self.tris = [], []

    def implied(self, h, J, K, qs):
        if getattr(self, "_key", None) != id(qs):
            m = max(len(q["events"]) for q in qs)
            idx = torch.tensor([[self.loc[v] for v in q["events"]] + [-1] * (m - len(q["events"])) for q in qs])
            self._idx, self._mask, self._key = idx.clamp(min=0), (idx >= 0).double(), id(qs)
        lp = torch.nn.functional.logsigmoid(h)[self._idx] * self._mask
        return lp.sum(1).exp().clamp(1e-9, 1 - 1e-9)

    def fit(self, qs, lam, h0):
        h = torch.tensor(h0, requires_grad=True)
        s = torch.tensor([clip(q["p"]) for q in qs])
        opt = torch.optim.LBFGS([h], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad(); t = kl_terms(s, self.implied(h, None, None, qs)).sum(); t.backward(); return t
        opt.step(closure)
        return h.detach(), torch.zeros(0), torch.zeros(0), float(kl_terms(s, self.implied(h.detach(), None, None, qs)).sum())


# --- one week ------------------------------------------------------------------------------------

def load_week(d: Path, week: str):
    wd = d / "weeks" / week
    events = [json.loads(l) for l in (wd / "events.jsonl").read_text().splitlines()]
    graph = json.loads((wd / "graph.json").read_text())
    log = {}
    for l in (d / "log.jsonl").read_text().splitlines():
        r = json.loads(l); log[r["query_id"]] = r
    qs, missing = [], defaultdict(int)
    for l in (d / "queries.jsonl").read_text().splitlines():
        q = json.loads(l)
        if q["week"] != week:
            continue
        r = log.get(q["query_id"])
        if r is None or r["p"] is None:
            missing[q["kind"]] += 1
            continue
        qs.append({k: q[k] for k in ("query_id", "kind", "events", "label", "heldout")} | {"p": r["p"]})
    return events, graph, qs, dict(missing)


def components(n, links, tris):
    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(links)
    out = []
    for c in nx.connected_components(g):
        if len(c) < 2:
            continue
        sub = g.subgraph(c)
        tw = treewidth_min_fill_in(sub)[0]
        cl = [tuple(e) for e in links if e[0] in c]
        ct = [tuple(t) for t in tris if t[0] in c]
        out.append({"nodes": sorted(c), "links": cl, "tris": ct, "tw": tw})
    return out


def make(model, comp):
    exact = comp["tw"] <= TW_MAX
    if model == "M1":
        return IndepModel(comp["links"], comp["tris"]), "indep"
    use_k = model.startswith("M3")
    tris = comp["tris"] if use_k else []
    if exact:
        m = AnchoredJT(0, comp["links"], tris)
    else:
        m = BPModel(comp["links"], comp["tris"], use_k=use_k)
    m.anchor = ANCHOR_W if model.endswith("b") else 1.0
    return m, ("exact" if exact else "bp")


def fit_component(model, comp, train, lam, stated_single, check=False):
    m, how = make(model, comp)
    h0 = [logit(stated_single.get(i, 0.5)) for i in m.linked]
    t0 = time.time()
    h, J, K, data = m.fit(train, lam, h0)
    info = {"how": how, "tw": comp["tw"], "nodes": len(comp["nodes"]), "links": len(comp["links"]),
            "tris": len(comp["tris"]), "seconds": round(time.time() - t0, 1), "data_kl": data}
    if how == "bp":
        res, it = m.check(h, J, K)
        info.update(bp_residual=res, bp_iters=it)
        if res > 1e-6:  # not converged: refit with heavier damping
            m.damp = 0.85; m.mu3 = m.mu2 = None
            h, J, K, data = m.fit(train, lam, h0)
            res, it = m.check(h, J, K)
            info.update(refit_damp=0.85, bp_residual=res, bp_iters=it, data_kl=data)
        info["bp_converged"] = res <= 1e-6
        if check:
            m.gibbs_est, info["gibbs"] = m.gibbs(h, J, K)
    return m, h, J, K, info


def predict(m, h, J, K, qs):
    if not qs:
        return []
    with torch.no_grad():
        return m.implied(h, J, K, qs).tolist()


def fit_week(args):
    d, week, lams, check, inner = args
    events, graph, qs, missing = load_week(d, week)
    n = len(events)
    links = [tuple(x) for x in graph["links"]]
    tris = [tuple(x) for x in graph["triangles"]]
    comps = components(n, links, tris)
    comp_of = {v: k for k, c in enumerate(comps) for v in c["nodes"]}
    stated_single = {q["events"][0]: q["p"] for q in qs if q["kind"] == "single"}
    train_all = [q for q in qs if not q["heldout"]]
    if inner:  # lambda tuning: split the training joint answers again, 20% inner validation
        rng = random.Random(f"{SEED}-inner-{week}")
        joint = [q for q in train_all if q["kind"] != "single"]
        rng.shuffle(joint)
        val_ids = {q["query_id"] for q in joint[: round(0.2 * len(joint))]}
        target = [q for q in train_all if q["query_id"] in val_ids]
        train_all = [q for q in train_all if q["query_id"] not in val_ids]
    else:
        target = [q for q in qs if q["heldout"]]
    out = {"week": week, "events": n, "links": len(links), "triangles": len(tris), "missing": missing,
           "components": [{k: c[k] for k in ("tw",)} | {"n": len(c["nodes"])} for c in comps], "fits": {}}
    preds = defaultdict(dict)  # model -> query_id -> value
    for model in MODELS:
        for lam in (lams[model] if isinstance(lams[model], list) else [lams[model]]):
            key = f"{model}@{lam}"
            infos = []
            pr, prg = {}, {}
            for ci, comp in enumerate(comps):
                cn = set(comp["nodes"])
                tr = [q for q in train_all if q["events"][0] in cn]
                tg = [q for q in target if q["events"][0] in cn]
                allc = [q for q in qs if q["events"][0] in cn]
                m, h, J, K, info = fit_component(model, comp, tr, lam, stated_single,
                                                 check=check and model != "M1")
                vals = predict(m, h, J, K, allc)
                pr.update({q["query_id"]: v for q, v in zip(allc, vals)})
                # the same model's values by Gibbs sampling (BP components) or exact (junction tree / M1)
                ge = getattr(m, "gibbs_est", None)
                prg.update({q["query_id"]: (ge[(q["kind"], tuple(sorted(q["events"])))] if ge else v)
                            for q, v in zip(allc, vals)})
                if model.startswith("M3") and not inner:
                    info["K_abs_mean"] = float(K.abs().mean()) if len(K) else 0.0
                if model != "M1" and not inner:
                    info["J_abs_mean"] = float(J.abs().mean()) if len(J) else 0.0
                infos.append(info)
            for q in qs:  # isolated events: fitted = stated
                if q["events"][0] not in comp_of:
                    pr[q["query_id"]] = prg[q["query_id"]] = clip(q["p"])
            tgt_kl = float(np.mean([float(kl_terms([q["p"]], torch.tensor([clip(pr[q["query_id"]])]))[0]) for q in target])) if target else None
            out["fits"][key] = {"target_kl": tgt_kl, "n_target": len(target), "components": infos}
            preds[key] = pr
            if not inner:
                preds[key + ":gibbs"] = prg
            print(f"{week} {key}: target KL {tgt_kl:.4f} "
                  f"({sum(i['seconds'] for i in infos):.0f}s, bp {[round(i.get('bp_residual', 0), 12) for i in infos if i['how'] == 'bp']})",
                  flush=True)
    out["preds"] = preds
    return out


def cmd_tune(d, weeks, procs):
    lams = {"M1": [0.0]} | {m: LAMBDAS for m in MODELS[1:]}
    with Pool(procs) as pool:
        res = pool.map(fit_week, [(d, w, lams, False, True) for w in weeks])
    grid = defaultdict(dict)
    for model in MODELS[1:]:
        for lam in LAMBDAS:
            key = f"{model}@{lam}"
            tot = sum(r["fits"][key]["target_kl"] * r["fits"][key]["n_target"] for r in res)
            n = sum(r["fits"][key]["n_target"] for r in res)
            grid[model][lam] = tot / n
    chosen = {}
    for model, g in grid.items():
        best = min(g.values())
        chosen[model] = max(l for l, v in g.items() if v <= best * 1.05)
    (d / "lambda.json").write_text(json.dumps({"tuning_weeks": weeks, "grid_inner_kl": grid, "chosen": chosen,
                                               "rule": "largest lam within 5% of the best inner-validation KL"}, indent=1))
    print(json.dumps({"grid": grid, "chosen": chosen}, indent=1))


def cmd_fit(d, weeks, procs, lam_file=None, fits_dir="fits"):
    ch = json.loads((Path(lam_file) if lam_file else d / "lambda.json").read_text())["chosen"]
    lams = {"M1": 0.0} | {m: ch[m] for m in MODELS[1:]}
    (d / fits_dir).mkdir(exist_ok=True)
    todo = [w for w in weeks if not (d / fits_dir / f"{w}.json").exists()]
    with Pool(procs) as pool:
        for r in pool.imap_unordered(fit_week, [(d, w, lams, True, False) for w in todo]):
            (d / fits_dir / f"{r['week']}.json").write_text(json.dumps(r))
            print(f"saved {r['week']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "fit"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weeks")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--lambda-file", help="fit: lambda JSON other than <data>/lambda.json (sensitivity runs)")
    ap.add_argument("--fits-dir", default="fits")
    a = ap.parse_args()
    weeks = a.weeks.split(",") if a.weeks else json.loads((a.data / "build-summary.json").read_text())["weeks"]
    if a.cmd == "tune":
        cmd_tune(a.data, weeks, a.procs)
    else:
        cmd_fit(a.data, weeks, a.procs, a.lambda_file, a.fits_dir)


if __name__ == "__main__":
    main()
