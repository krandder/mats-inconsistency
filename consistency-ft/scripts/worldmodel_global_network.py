"""Post-fit additions for the global world-model fit: dependence-network chart data, and the
accuracy of plain independence of the stated marginals (added to both JSONs).

At the chosen penalty (L2 0.3) the fit keeps a single edge with |log OR| > 0.5, so a
network drawn at that threshold is empty.  This script adds two drawable networks:
  lambda_0.1   refit at L2 0.1 on all queries (validation KL 4% above the best, BP converges);
               edges with |fitted log OR| > 0.5
  chosen_top   the chosen fit (fit-params.pt), edges with |fitted log OR| > 0.1
Each: nodes (event, short label, stated/fitted marginal, outcome, Louvain community, spring
layout x/y) and edges (fitted log OR, pair type sim/rand, cosine, same parent).

Usage: worldmodel_global_network.py --data /home/agent/data/worldmodel-global-20260924 \
          --charts docs/worldmodel-global-20260924-charts.json --summary docs/worldmodel-global-20260924.json
          [--skip-refit]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

import worldmodel_global_fit as W


def network(events, marg, g, h, J, sim_pairs, cos_of, thr, note):
    with torch.no_grad():
        H, mu = g.bp(h[None, :], J)
        fm = torch.sigmoid(H[0]).numpy()
        lp = g.pair_logp(H, mu, J, torch.arange(g.E))[0].exp().numpy()
    lor = np.log(lp[:, 3] * lp[:, 0] / (lp[:, 1] * lp[:, 2]))
    keep = [k for k in range(g.E) if abs(lor[k]) > thr]
    G = nx.Graph()
    for k in keep:
        G.add_edge(*g.edges[k], weight=float(abs(lor[k])))
    pos = nx.spring_layout(G, seed=W.SEED, k=0.25, iterations=300) if G.number_of_edges() else {}
    comm = nx.community.louvain_communities(G, seed=W.SEED) if G.number_of_edges() else []
    comm_of = {v: c for c, vs in enumerate(sorted(comm, key=len, reverse=True)) for v in vs}
    nodes = []
    for v in sorted(G.nodes):
        e = events[v]
        nodes.append({"id": v, "label": e["parent_question"][:70] + (f" — {e['answer_text'][:40]}" if e["answer_text"] else ""),
                      "parent": e["parent_id"], "stated": marg[v], "fit": float(fm[v]), "outcome": e["label"],
                      "degree": G.degree(v), "community": comm_of.get(v),
                      "x": float(pos[v][0]), "y": float(pos[v][1])})
    edges = [{"source": g.edges[k][0], "target": g.edges[k][1], "fit_lor": round(float(lor[k]), 3),
              "J": round(float(J[k]), 3), "type": "sim" if g.edges[k] in sim_pairs else "rand",
              "cos": cos_of.get(g.edges[k]),
              "same_parent": events[g.edges[k][0]]["parent_id"] == events[g.edges[k][1]]["parent_id"]} for k in keep]
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    tw = nx.algorithms.approximation.treewidth_min_fill_in(G)[0] if G.number_of_edges() else 0
    summary = {"edges": len(edges), "nodes": len(nodes), "components": len(comps), "component_sizes": comps[:15],
               "treewidth_upper_min_fill": tw, "louvain_sizes": sorted((len(c) for c in comm), reverse=True)[:15],
               "sim_share": float(np.mean([e["type"] == "sim" for e in edges])) if edges else None,
               "same_parent_share": float(np.mean([e["same_parent"] for e in edges])) if edges else None,
               "positive_share": float(np.mean([e["fit_lor"] > 0 for e in edges])) if edges else None}
    return {"description": note, "threshold_abs_log_or": thr, "summary": summary, "nodes": nodes, "edges": edges}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--charts", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--skip-refit", action="store_true", help="only add the independence accuracy rows")
    a = ap.parse_args()
    events = [json.loads(l) for l in (a.data / "events.jsonl").read_text().splitlines()]
    queries = [json.loads(l) for l in (a.data / "queries.jsonl").read_text().splitlines()]
    log = {json.loads(l)["query_id"]: json.loads(l) for l in (a.data / "log.jsonl").read_text().splitlines()}
    graph = json.loads((a.data / "graph.json").read_text())
    sim_pairs = {tuple(p) for p in graph["sim_pairs"]}
    cos_of = {tuple(p): c for p, c in zip(graph["sim_pairs"] + graph["rand_pairs"], graph["sim_cosine"] + graph["rand_cosine"])}
    train = [(W.lits(q["formula"]), log[q["query_id"]]["p"]) for q in queries if log[q["query_id"]]["p"] is not None]
    marg = [None] * len(events)
    for q in queries:
        if q["kind"] == "single":
            marg[q["events"][0]] = log[q["query_id"]]["p"]
    charts = json.loads(a.charts.read_text())
    summ = json.loads(a.summary.read_text())
    rows = {}
    for q in queries:
        L = W.lits(q["formula"])
        p = 1.0
        for i, x in L:
            p *= W.clip(marg[i]) if x else 1 - W.clip(marg[i])
        kind = q["kind"] + (f"/{q['pair_type']}" if q["kind"] == "and2" else "")
        for key in {kind, "all"} | ({"all joints"} if q["kind"] != "single" else set()):
            rows.setdefault(key, []).append((W.ll(p, q["label"]), W.brier(p, q["label"])))
    indep = [{"kind": k, "queries": len(v), "ll_indep_stated": float(np.mean([x[0] for x in v])),
              "brier_indep_stated": float(np.mean([x[1] for x in v]))} for k, v in sorted(rows.items())]
    summ["accuracy_independence_of_stated_marginals"] = indep
    charts["accuracy_independence_of_stated_marginals"] = indep
    a.summary.write_text(json.dumps(summ, indent=1))
    print(json.dumps(indep))
    if a.skip_refit:
        a.charts.write_text(json.dumps(charts))
        return
    d = torch.load(a.data / "fit-params.pt", weights_only=False)
    g0 = W.Graph(len(events), d["edges"])
    charts["dependence_network_chosen_top"] = network(
        events, marg, g0, d["h"], d["J"], sim_pairs, cos_of, 0.1,
        "chosen fit (L2 0.3): queried pairs with |fitted log odds ratio| > 0.1 (only 1 exceeds 0.5); "
        "x/y spring layout of this subgraph, community = Louvain")
    g, h, J, info = W.fit_mrf(len(events), train, marg, "l2", 0.1)
    torch.save({"edges": g.edges, "h": h, "J": J, "info": info}, a.data / "fit-params-l2-0.1.pt")
    charts["dependence_network_lambda_0.1"] = network(
        events, marg, g, h, J, sim_pairs, cos_of, 0.5,
        f"refit at L2 0.1 on all queries (BP residual {info['bp_residual']:.1e}): queried pairs with |fitted log odds "
        "ratio| > 0.5; x/y spring layout of this subgraph, community = Louvain")
    for k in ("dependence_network_chosen_top", "dependence_network_lambda_0.1"):
        print(k, json.dumps(charts[k]["summary"]))
    charts["dependence_network"]["note"] = ("threshold 0.5 at the chosen penalty leaves 1 edge; see "
                                            "dependence_network_lambda_0.1 and dependence_network_chosen_top")
    a.charts.write_text(json.dumps(charts))


if __name__ == "__main__":
    main()
