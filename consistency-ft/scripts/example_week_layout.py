"""Layout for the example-week view: node positions and visible labels (reads week.json, writes week-view.json).

The largest component gets its own panel (spring layout); the other linked components are packed in
shelves below it; events without a link are a strip of dots.  Visible labels are placed greedily
(highest degree first) where they do not overlap another label or node; every node keeps its full
question on hover.  The page itself is drawn by ekelvin-research/build.py (page5, week.html).
Usage: example_week_layout.py --data /home/agent/data/example-week-20260927
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx

W = 790
MAIN_H = 600
CH = 6.3  # px per character at 11.5px
SEED = 20260927


def spring(g, w, h, pad=14):
    if g.number_of_nodes() == 1:
        return {next(iter(g)): (w / 2, h / 2)}
    pos = nx.spring_layout(g, seed=SEED, k=1.6 / math.sqrt(g.number_of_nodes()), iterations=300)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = (w - 2 * pad) / max(x1 - x0, 1e-9); sy = (h - 2 * pad) / max(y1 - y0, 1e-9)
    return {v: (pad + (p[0] - x0) * sx, pad + (p[1] - y0) * sy) for v, p in pos.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    wk = json.loads((a.data / "week.json").read_text())
    g = nx.Graph()
    g.add_nodes_from(e["i"] for e in wk["events"])
    g.add_edges_from((l["a"], l["b"]) for l in wk["links"])
    comps = sorted((c for c in nx.connected_components(g) if len(c) > 1), key=lambda c: (-len(c), min(c)))
    isolated = sorted(v for v in g if g.degree(v) == 0)
    pos = {}
    main = g.subgraph(comps[0])
    for v, p in spring(main, W, MAIN_H).items():
        pos[v] = p
    y = MAIN_H + 30
    x, shelf_h = 0, 0
    for c in comps[1:]:
        n = len(c)
        side = 46 + 30 * math.sqrt(n)
        if x + side > W:
            x, y = 0, y + shelf_h + 10
            shelf_h = 0
        for v, p in spring(g.subgraph(c), side, side, pad=8).items():
            pos[v] = (x + p[0], y + p[1])
        x += side + 10
        shelf_h = max(shelf_h, side)
    y += shelf_h + 34
    iso_y0 = y
    per_row = int((W - 20) // 14)
    for k, v in enumerate(isolated):
        pos[v] = (10 + (k % per_row) * 14, y + (k // per_row) * 14)
    H = y + (math.ceil(len(isolated) / per_row)) * 14 + 10
    # greedy labels: largest-component nodes by degree, then the top node of each small component
    ev = {e["i"]: e for e in wk["events"]}
    lab_file = a.data / "labels.jsonl"
    if lab_file.exists():  # model-written display labels (example_week_labels.py)
        lab = {r["id"]: r["label"] for r in map(json.loads, lab_file.read_text().splitlines())}
        for e in ev.values():
            e["short"] = lab.get(e["id"], e["short"])
    boxes = [(px - 8, py - 8, px + 8, py + 8) for v, (px, py) in pos.items()]
    labels = {}

    def free(b):
        return all(b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3] for o in boxes) and 0 <= b[0] and b[2] <= W

    order = sorted(comps[0], key=lambda v: -g.degree(v)) + [max(c, key=lambda v: g.degree(v)) for c in comps[1:]]
    for v in order:
        px, py = pos[v]
        t = ev[v]["short"]
        wlab = len(t) * CH
        for anchor, b in (("start", (px + 9, py - 7, px + 9 + wlab, py + 6)),
                          ("end", (px - 9 - wlab, py - 7, px - 9, py + 6)),
                          ("middle", (px - wlab / 2, py - 22, px + wlab / 2, py - 9)),
                          ("middle-below", (px - wlab / 2, py + 9, px + wlab / 2, py + 22))):
            if free(b):
                labels[v] = {"anchor": anchor, "x": {"start": px + 10, "end": px - 10}.get(anchor, px),
                             "y": {"middle": py - 12, "middle-below": py + 19}.get(anchor, py + 4)}
                boxes.append(b)
                break
    # visible edge labels "P(B|A) / P(A|B)" (fitted) on the strongest main-panel links, where they fit
    big = set(comps[0])
    edge_labels = []
    for l in sorted((l for l in wk["links"] if l["a"] in big), key=lambda l: -abs(l["fit_log_or"])):
        if len(edge_labels) >= 12:
            break
        (ax, ay), (bx, by) = pos[l["a"]], pos[l["b"]]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        b = (mx - 30, my - 7, mx + 30, my + 5)
        if math.hypot(bx - ax, by - ay) > 40 and free(b):
            boxes.append(b)
            edge_labels.append({"a": l["a"], "b": l["b"], "x": round(mx, 1), "y": round(my + 3, 1),
                                "text": f'{l["fit_b_given_a"]:.2f} / {l["fit_a_given_b"]:.2f}'})
    out = {"W": W, "H": round(H), "main_h": MAIN_H, "shelf_top": MAIN_H + 30, "iso_top": iso_y0,
           "components": [sorted(c) for c in comps], "isolated": isolated,
           "pos": {str(v): [round(p[0], 1), round(p[1], 1)] for v, p in pos.items()},
           "labels": {str(v): l for v, l in labels.items()},
           "edge_labels": edge_labels,
           "short": {str(v): e["short"] for v, e in ev.items()}}
    (a.data / "week-view.json").write_text(json.dumps(out))
    print(f"components {len(comps)} (largest {len(comps[0])}), isolated {len(isolated)}, labels {len(labels)}, H {H:.0f}")


if __name__ == "__main__":
    main()
