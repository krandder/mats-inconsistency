#!/usr/bin/env python3
"""Export the final Bayes-net structure of market-game runs for the network page.

For each run: replay the final pass from the log, then emit nodes (every linked question
plus summary stats for unlinked ones), edges (conditional links) with dependency strength
|P(a|b=yes) - P(a|b=no)| and mutual information, components, and the marginal histogram
split by resolution.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, "/home/agent/workspaces/bayes-engine")
import harness.market_game as mg  # noqa: E402
import score_market_game as smg  # noqa: E402


def mutual_info(m: mg.Market, a: str, b: str) -> float:
    pb = m.engine.marginal(b)["yes"]
    pa_b1 = m.engine.marginal(a, {b: "yes"})["yes"]
    pa_b0 = m.engine.marginal(a, {b: "no"})["yes"]
    pa = pb * pa_b1 + (1 - pb) * pa_b0
    mi = 0.0
    for vb, pvb in ((1, pb), (0, 1 - pb)):
        pa_given = pa_b1 if vb else pa_b0
        for va in (1, 0):
            pj = pvb * (pa_given if va else 1 - pa_given)
            pm = (pa if va else 1 - pa) * pvb
            if pj > 1e-12 and pm > 1e-12:
                mi += pj * math.log(pj / pm)
    return mi


def export(run: Path, label: str) -> dict:
    cfg = json.loads((run / "config.json").read_text()).get("config", {})
    qs = {json.loads(l)["question_id"]: json.loads(l) for l in open(run / "questions.jsonl") if l.strip()}
    ids = list(qs)
    passes = int(cfg.get("passes", 3))
    snaps = smg.replay_markets(run, passes, ids)
    last = max(p for p in snaps if isinstance(p, int))
    m = mg.Market.from_snapshot(snaps[last])
    prices = m.prices()
    linked = sorted(m.links)
    edges, seen = [], set()
    for a in linked:
        for b in sorted(m.links[a]):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            d = abs(m.engine.marginal(a, {b: "yes"})["yes"] - m.engine.marginal(a, {b: "no"})["yes"])
            edges.append({"s": a, "t": b, "d": round(d, 4), "mi": round(mutual_info(m, a, b), 5)})
    idx = {q: i for i, q in enumerate(linked)}
    nodes = [{"id": q, "i": idx[q], "p": round(prices[q], 4), "y": int(qs[q]["resolution"]), "text": qs[q]["text"][:160],
              "deg": len(m.links[q])} for q in linked]
    # components
    parent = {q: q for q in linked}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e in edges:
        parent[find(e["s"])] = find(e["t"])
    comp = {}
    for q in linked:
        comp.setdefault(find(q), []).append(q)
    sizes = sorted((len(v) for v in comp.values()), reverse=True)
    for n in nodes:
        n["c"] = find(n["id"])
    # joint vs product for each component: log-likelihood gain on the resolved outcomes
    hist = {"bins": [i / 20 for i in range(21)], "yes": [0] * 20, "no": [0] * 20, "yes_linked": [0] * 20, "no_linked": [0] * 20}
    for q in ids:
        b = min(19, int(prices[q] * 20))
        k = "yes" if qs[q]["resolution"] else "no"
        hist[k][b] += 1
        if q in m.links:
            hist[k + "_linked"][b] += 1
    stats = m.stats()
    return {"label": label, "pass": last, "variables": len(ids), "linked": len(linked), "edges": len(edges),
            "components": len(comp), "component_sizes": sizes[:40], "treewidth": stats.get("treewidth"),
            "max_clique_states": stats.get("maxCliqueStates"), "nodes": nodes, "links": edges, "hist": hist}


def main() -> None:
    runs = [("astra-n8", "GPT-6 Astra, 8 per session"), ("luna-n8", "GPT-5.6 Luna, 8 per session"),
            ("astra-n16", "GPT-6 Astra, 16 per session")]
    out = [export(ROOT / "runs/market-game" / r, lab) for r, lab in runs]
    dest = ROOT / "site" / "market-network-data.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    for o in out:
        print(o["label"], "linked", o["linked"], "edges", o["edges"], "components", o["components"], "largest", o["component_sizes"][:5], "tw", o["treewidth"])
    print("bytes", dest.stat().st_size)


if __name__ == "__main__":
    main()
