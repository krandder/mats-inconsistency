"""Global joint fit for one example week (scripts/example_week.py output); exact junction tree.

Model: binary log-linear (Ising in {0,1}) over the week's events,
    log p(x) = sum_i h_i x_i + sum_links J_ij x_i x_j + sum_triangles K_ijk x_i x_j x_k - log Z.
Fit: minimise sum_q KL_Bernoulli(stated_q || implied_q) + lam * (sum J^2 + sum K^2), L-BFGS,
implied probabilities exact by junction tree (scripts/jt_exact.py):
    single P(A); "A and B" P(A,B); "A and B and C" P(A,B,C); conditional P(B | A) = P(A,B) / P(A).
Stated values clipped to [0.01, 0.99]; missing answers left out.
lam: grid, chosen on 20% held-out link units (the link's AND and both conditionals) and 20% of
triangles; the largest lam within 5% of the best held-out KL; then refit on everything.
Same objective and selection rule as scripts/worldmodel_global_fit.py (c6b7e5c, fa18070), with
triple terms added and BP replaced by exact inference.

Usage: example_week_fit.py --data /home/agent/data/example-week-20260927
Writes <data>/fit.json (numbers) and <data>/week.json (everything the view needs).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jt_exact import JunctionTree  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # small tensors on a shared, loaded box: one thread is ~20x faster than eight
LO = 0.01
SEED = 20260927
LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0]


def clip(p):
    return min(max(p, LO), 1 - LO)


def logit(p):
    p = clip(p); return math.log(p / (1 - p))


def ll(p, y):
    p = clip(p); return -math.log(p if y else 1 - p)


class Model:
    """Joint over the linked events only; an event with no link is independent of everything, so its
    fitted marginal is its stated marginal (the KL optimum) and it needs no junction tree."""

    def __init__(self, n, links, triangles):
        self.linked = sorted({v for e in links for v in e})
        self.loc = {v: k for k, v in enumerate(self.linked)}
        self.jt = JunctionTree(len(self.linked), [tuple(self.loc[v] for v in e) for e in links],
                               [tuple(self.loc[v] for v in t) for t in triangles])
        back = {tuple(sorted(self.loc[v] for v in e)): tuple(sorted(e)) for e in list(links) + list(triangles)}
        self.links = [back[e] for e in self.jt.pairs]
        self.lid = {p: k for k, p in enumerate(self.links)}
        self.tris = [back[t] for t in self.jt.triples]

    def lp(self, lits):
        return self.jt.logp([(self.loc[i], x) for i, x in lits])

    def _prepare(self, qs):
        """Index every query by the all-YES subsets it needs; one 0/1 matrix per bag (vectorised)."""
        key = id(qs)
        if getattr(self, "_prep_key", None) == key:
            return
        subsets, sid = [], {}

        def need(vs):
            vs = tuple(sorted(self.loc[v] for v in vs))
            if vs not in sid:
                sid[vs] = len(subsets); subsets.append(vs)
            return sid[vs]
        num, den = [], []
        for q in qs:
            if q["kind"] == "cond":
                a, b = q["events"]
                num.append(need((a, b))); den.append(need((a,)))
            else:
                num.append(need(q["events"])); den.append(-1)
        by_bag = defaultdict(list)
        for k, vs in enumerate(subsets):
            by_bag[self.jt.home[vs]].append(k)
        self._groups = []
        for b, ks in by_bag.items():
            bag = self.jt.bags[b]
            d = len(bag)
            grid = torch.tensor(list(itertools.product([0, 1], repeat=d)), dtype=torch.float64) if d else torch.ones(1, 0)
            M = torch.stack([grid[:, [bag.index(v) for v in subsets[k]]].prod(1) for k in ks])
            self._groups.append((b, torch.tensor(ks), M))
        self._n_sub = len(subsets)
        self._num = torch.tensor(num)
        self._den = torch.tensor(den)
        self._prep_key = key
        self._qs_ref = qs

    def implied(self, h, J, K, qs):
        self._prepare(qs)
        self.jt.calibrate(h, J, K)
        probs = torch.zeros(self._n_sub)
        for b, ks, M in self._groups:
            probs = probs.index_put((ks,), M @ self.jt.beliefs[b].reshape(-1).exp())
        p = probs[self._num]
        has_den = self._den >= 0
        p = torch.where(has_den, p / probs[self._den.clamp(min=0)], p)
        return p.clamp(1e-9, 1 - 1e-9)

    def fit(self, qs, lam, h0):
        """qs: queries on linked events only; h0: initial logits for self.linked."""
        h = torch.tensor(h0, requires_grad=True)
        J = torch.zeros(len(self.links), requires_grad=True)
        K = torch.zeros(len(self.tris), requires_grad=True)
        s = torch.tensor([clip(q["p"]) for q in qs])

        def loss():
            p = self.implied(h, J, K, qs)
            data = (s * torch.log(s / p) + (1 - s) * torch.log((1 - s) / (1 - p))).sum()
            return data + lam * ((J ** 2).sum() + (K ** 2).sum()), data
        opt = torch.optim.LBFGS([h, J, K], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-11)

        def closure():
            opt.zero_grad(); tot, _ = loss(); tot.backward(); return tot
        opt.step(closure)
        with torch.no_grad():
            tot, data = loss()
        return h.detach(), J.detach(), K.detach(), float(data)


def kl_mean(stated, implied):
    t = 0.0
    for s, p in zip(stated, implied):
        s, p = clip(s), clip(p)
        t += s * math.log(s / p) + (1 - s) * math.log((1 - s) / (1 - p))
    return t / max(len(stated), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    d = a.data
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    queries = {json.loads(l)["query_id"]: json.loads(l) for l in (d / "queries.jsonl").read_text().splitlines()}
    log = {}
    for l in (d / "log.jsonl").read_text().splitlines():
        r = json.loads(l); log[r["query_id"]] = r
    graph = json.loads((d / "graph.json").read_text())
    n = len(events)
    links = [tuple(x) for x in graph["links"]]
    tris = [tuple(x) for x in graph["triangles"]]
    qs = []
    for qid, q in queries.items():
        r = log.get(qid)
        if r is None or r["p"] is None:
            continue
        kind = q["kind"]
        qs.append({"query_id": qid, "kind": kind, "events": q["events"], "p": r["p"], "label": q["label"]})
    missing = len(queries) - len(qs)
    m = Model(n, links, tris)
    stated_single = {q["events"][0]: q["p"] for q in qs if q["kind"] == "single"}
    h0 = [logit(stated_single.get(i, 0.5)) for i in m.linked]
    linked = set(m.linked)
    qs_l = [q for q in qs if all(i in linked for i in q["events"])]
    print(f"events {n} links {len(links)} triangles {len(tris)} treewidth {m.jt.treewidth} bags {len(m.jt.bags)} "
          f"queries {len(qs)} (missing {missing}), on linked events {len(qs_l)}", flush=True)

    # held-out split for lam
    rng = random.Random(SEED)
    unit = {}
    for q in qs_l:
        if q["kind"] in ("and2", "cond"):
            unit[q["query_id"]] = ("l", tuple(sorted(q["events"])))
        elif q["kind"] == "and3":
            unit[q["query_id"]] = ("t", tuple(sorted(q["events"])))
    units = sorted(set(unit.values()))
    rng.shuffle(units)
    held = set(units[: round(0.2 * len(units))])
    train = [q for q in qs_l if unit.get(q["query_id"]) not in held]
    val = [q for q in qs_l if unit.get(q["query_id"]) in held]
    grid = {}
    for lam in LAMBDAS:
        t0 = time.time()
        h, J, K, _ = m.fit(train, lam, h0)
        with torch.no_grad():
            pv = m.implied(h, J, K, val).tolist()
        grid[lam] = kl_mean([q["p"] for q in val], pv)
        print(f"lam {lam}: held-out KL {grid[lam]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    best = min(grid.values())
    lam = max(l for l, v in grid.items() if v <= best * 1.05)
    h, J, K, data_kl = m.fit(qs_l, lam, h0)
    with torch.no_grad():
        fl = dict(zip([q["query_id"] for q in qs_l], m.implied(h, J, K, qs_l).tolist()))
        fitted = [fl.get(q["query_id"], clip(q["p"])) for q in qs]  # unlinked singles: fitted = stated
        m.jt.calibrate(h, J, K)
        marg = [float(m.lp([(i, 1)]).exp()) if i in linked else clip(stated_single.get(i, 0.5)) for i in range(n)]
        pair = {(i, j): float(m.lp([(i, 1), (j, 1)]).exp()) for (i, j) in m.links}
        tri_p = {t: float(m.lp([(v, 1) for v in t]).exp()) for t in m.tris}
    # independence-of-stated-marginals control, for the same queries
    def indep(q):
        ps = [clip(stated_single.get(i, 0.5)) for i in q["events"]]
        return ps[1] if q["kind"] == "cond" else math.prod(ps)

    # outcome log loss by kind (conditionals scored where the condition resolved YES)
    by = defaultdict(lambda: {"n": 0, "raw": 0.0, "fit": 0.0, "indep": 0.0})
    for q, f in zip(qs, fitted):
        if q["label"] is None:
            continue
        kinds = (q["kind"], "all") + (("single_linked",) if q["kind"] == "single" and q["events"][0] in linked else ())
        for k in kinds:
            b = by[k]; b["n"] += 1
            b["raw"] += ll(q["p"], q["label"]); b["fit"] += ll(f, q["label"]); b["indep"] += ll(indep(q), q["label"])
    loss = {k: {"n": v["n"], "raw": v["raw"] / v["n"], "fit": v["fit"] / v["n"], "indep": v["indep"] / v["n"]} for k, v in by.items()}
    stated_kl = {k: kl_mean([q["p"] for q in qs if q["kind"] == k], [f for q, f in zip(qs, fitted) if q["kind"] == k])
                 for k in ("single", "and2", "and3", "cond")}

    g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(links)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    comp_of = {v: k for k, c in enumerate(comps) for v in c}
    stated = defaultdict(dict)
    for q in qs:
        stated[q["kind"]][tuple(q["events"])] = q["p"]
    out_links = []
    for (i, j), k in m.lid.items():
        p11 = pair[(i, j)]
        pi, pj = marg[i], marg[j]
        lor = math.log(max(p11, 1e-12)) + math.log(max(1 - pi - pj + p11, 1e-12)) - math.log(max(pi - p11, 1e-12)) - math.log(max(pj - p11, 1e-12))
        s_and = stated["and2"].get((i, j), stated["and2"].get((j, i)))
        out_links.append({"a": i, "b": j, "rel": graph["rel"][f"{i}-{j}"], "J": float(J[k]),
                          "fit_and": p11, "fit_b_given_a": p11 / pi, "fit_a_given_b": p11 / pj, "fit_log_or": lor,
                          "stated_and": s_and, "stated_b_given_a": stated["cond"].get((i, j)),
                          "stated_a_given_b": stated["cond"].get((j, i))})
    out_tris = [{"v": list(t), "K": float(K[k]), "fit": tri_p[t], "stated": stated["and3"].get(tuple(sorted(t))) or
                 next((p for e, p in stated["and3"].items() if tuple(sorted(e)) == t), None)} for k, t in enumerate(m.tris)]
    out_events = [{**{k: e[k] for k in ("id", "short", "question", "rel_types") if k in e}, "i": i, "y": e["label"],
                   "stated": stated_single.get(i), "fit": marg[i], "comp": comp_of[i]} for i, e in enumerate(events)]
    fit = {"events": n, "links": len(links), "triangles": len(tris), "treewidth": m.jt.treewidth,
           "components": len(comps), "largest_component": len(comps[0]),
           "components_ge2": sum(len(c) >= 2 for c in comps), "queries_fitted": len(qs), "missing": missing,
           "lambda_grid": grid, "lambda": lam, "data_kl_mean": data_kl / len(qs_l), "linked_events": len(m.linked), "stated_vs_fit_kl": stated_kl,
           "logloss": loss, "yes_rate": sum(e["label"] for e in events) / n}
    (d / "fit.json").write_text(json.dumps(fit, indent=1))
    (d / "week.json").write_text(json.dumps({"meta": json.loads((d / "week-choice.json").read_text()), "fit": fit,
                                             "events": out_events, "links": out_links, "triangles": out_tris}))
    print(json.dumps(fit, indent=1))


if __name__ == "__main__":
    main()
