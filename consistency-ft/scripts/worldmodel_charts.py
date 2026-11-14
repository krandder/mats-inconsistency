"""Chart-ready JSON for the world-model cluster experiment (reads the projection outputs; no fitting).

Adds one mix-adjusted view the projection summary lacks: the gain on every query except
negations (the negation count is fixed at 2 per cluster, so its share falls from 25% at
n=2 to 6% at n=8).

Usage: worldmodel_charts.py --data /home/agent/data/worldmodel-clusters-20260924 \
          --summary docs/worldmodel-clusters-20260924.json --out docs/worldmodel-clusters-20260924-charts.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import worldmodel_projection as wp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    clusters = {c["cluster_id"]: c for c in map(json.loads, (a.data / "clusters.jsonl").read_text().splitlines())}
    per = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for r in map(json.loads, (a.data / "projections.jsonl").read_text().splitlines()):
        if r["label"] is None or r["raw"] is None or r["kind"] == "neg":
            continue
        c = clusters[r["cluster_id"]]
        for t in ("all", c["type"]):
            acc = per[(t, c["n"])][r["cluster_id"]]
            acc[0] += wp.ll(r["raw"], r["label"]); acc[1] += wp.ll(r["kl|all"], r["label"]); acc[2] += 1
    rng = np.random.default_rng(wp.SEED)
    excl_neg = [{"type": t, "n": n, "method": "kl", **wp.bootstrap(list(v.values()), rng)}
                for (t, n), v in sorted(per.items())]
    pick = lambda rows, **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    out = {
        "units": "log loss per query (clip 0.01); gain = projected - raw, negative = better; ci = 95% paired "
                 "bootstrap over clusters",
        "raw_vs_projected_by_n": pick(s["main"], metric="log_loss"),
        "brier_by_n": pick(s["main"], metric="brier"),
        "gain_excluding_negations_by_n": excl_neg,
        "query_count_curve": [{k: r[k] for k in ("type", "n", "method", "stage", "queries_in_fit_per_cluster",
                                                 "target", "gain", "ci", "queries")} for r in s["curve"] if "gain" in r],
        "gain_by_query_kind": s["by_kind"],
        "shuffled_structure_control": pick(s["control"], metric="log_loss"),
        "calls": s["calls"],
    }
    a.out.write_text(json.dumps(out, indent=1))
    for r in excl_neg:
        print(f"excl neg {r['type']:5} n={r['n']} raw {r['raw']:.4f} proj {r['proj']:.4f} gain {r['gain']:+.4f} "
              f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
