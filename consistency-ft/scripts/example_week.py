"""One example week of the full conditional structure over the Luna link graph (Qwen3-32B, DashScope).

Kelvin, 27 Sep: "I'd like to see the full conditional structure in a given example week."

weeks   For every Monday t of the TEST split of /home/agent/data/luna-linkgraph-20260927: the events
        open and unresolved at t (open_time < t < end_time and resolution_time > t), the Luna links
        among them, the triangles of that graph, and the query counts and cost estimate for one
        week: marginals n, "A and B" per link, "A and B and C" per triangle, P(B | A) and P(A | B)
        per link.  Writes weeks.json (the full-run extrapolation uses it).
build   One week t (--week): events re-checked in the as-of store (training/asof.py): market and
        option exist before t, status open at t, no resolution logged at or before t; the question,
        option and criteria texts are the as-of-t versions.  Queries, one statement per call, v6_direct
        question-only prompt as in scripts/worldmodel_clusters.py (BASE_SYSTEM, "Assume today is t",
        question blocks, statement, ANSWER_DIRECT, " /no_think"; no market data):
          single  every event                       "Question A resolves YES"
          and2    every link (random order)         "Both of the following are true: ..."
          and3    every triangle (random order)     "All of the following are true: ..."
          cond    every link, both directions       "Suppose Question A resolves YES. What is the
                                                     probability that Question B resolves YES?"
run     worldmodel_clusters.run: qwen3-32b, thinking off, temperature 0, max_tokens 20, one retry,
        no model substitution, spend cap at the conservative 0.7/2.8 $/M price, <= 8 workers.

Usage:
  example_week.py weeks --out /home/agent/data/example-week-20260927
  example_week.py build --out /home/agent/data/example-week-20260927 --week 2025-MM-DD
  example_week.py run   --out /home/agent/data/example-week-20260927 --workers 8 --max-usd 2
Then scripts/example_week_fit.py --data <out>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import worldmodel_clusters as wc  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402
from training.asof import AsOf, to_ms  # noqa: E402

LG = Path("/home/agent/data/luna-linkgraph-20260927/test")
SEED = 20260927
CHARS_PER_TOKEN = 3.5
OUT_TOKENS = 8
FIXED_CHARS = 520  # system + as-of line + statement + answer instruction, measured on built prompts


def load_graph():
    nodes = [json.loads(l) for l in (LG / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(l) for l in (LG / "edges.jsonl").read_text().splitlines()]
    return nodes, edges


def open_at(nodes, t):
    return [n for n in nodes if n["open_time"] < t < n["end_time"] and n["resolution_time"] > t]


def week_graph(open_nodes, edges):
    ids = {n["id"] for n in open_nodes}
    g = nx.Graph()
    g.add_nodes_from(ids)
    for e in edges:
        if e["u"] in ids and e["v"] in ids:
            g.add_edge(e["u"], e["v"], type=e["type"], types=e["types"], mutual=e.get("mutual"))
    return g


def triangles(g) -> list[tuple]:
    out = set()
    for u, v in g.edges:
        for w in set(g[u]) & set(g[v]):
            out.add(tuple(sorted((u, v, w))))
    return sorted(out)


def block_chars(n) -> int:
    crit = min(len(vb.strip_creator_edits(n["criteria_text"] or "")[0]), vb.MAX_CRITERIA_CHARS)
    return 120 + len(n["question"] or "") + len(n["answer_text"] or "") + crit


def cost_est(calls_chars: list[int]) -> float:
    return sum((c + FIXED_CHARS) / CHARS_PER_TOKEN * wc.PRICE_IN / 1e6 + OUT_TOKENS * wc.PRICE_OUT / 1e6 for c in calls_chars)


def week_counts(nodes, edges, t_iso):
    t = to_ms(t_iso)
    op = open_at(nodes, t)
    g = week_graph(op, edges)
    tri = triangles(g)
    bc = {n["id"]: block_chars(n) for n in op}
    chars = [bc[i] for i in g.nodes] + [bc[u] + bc[v] for u, v in g.edges] * 3 + [sum(bc[x] for x in T) for T in tri]
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    linked = g.subgraph([v for v in g if g.degree(v)])
    try:
        from networkx.algorithms.approximation import treewidth_min_fill_in
        tw = max((treewidth_min_fill_in(g.subgraph(c))[0] for c in nx.connected_components(g) if len(c) > 1), default=0)
    except Exception:  # noqa: BLE001
        tw = None
    return {"week": t_iso, "events": g.number_of_nodes(), "linked_events": linked.number_of_nodes(),
            "parents": len({n["parent_id"] for n in op}), "links": g.number_of_edges(), "triangles": len(tri),
            "marginals": g.number_of_nodes(), "and2": g.number_of_edges(), "and3": len(tri),
            "conditionals": 2 * g.number_of_edges(),
            "calls": g.number_of_nodes() + 3 * g.number_of_edges() + len(tri),
            "est_usd_cap_price": round(cost_est(chars), 3),
            "largest_component": comps[0] if comps else 0, "components_ge3": sum(c >= 3 for c in comps),
            "treewidth": tw, "link_types": dict(Counter(d["type"] for _, _, d in g.edges(data=True))),
            "yes_rate": round(sum(n["resolution"] == "YES" for n in op) / max(1, len(op)), 3)}


def cmd_weeks(out: Path) -> None:
    nodes, edges = load_graph()
    lo = min(n["open_time"] for n in nodes)
    d = dt.date(2025, 5, 5)
    rows = []
    while to_ms(d.isoformat()) < max(n["end_time"] for n in nodes):
        if to_ms(d.isoformat()) > lo:
            rows.append(week_counts(nodes, edges, d.isoformat()))
            r = rows[-1]
            print(f"{r['week']} ev {r['events']:4d} links {r['links']:5d} tri {r['triangles']:5d} calls {r['calls']:6d} "
                  f"${r['est_usd_cap_price']:.2f} largest {r['largest_component']} tw {r['treewidth']} {r['link_types']}", flush=True)
        d += dt.timedelta(days=7)
    tot = {k: sum(r[k] for r in rows) for k in ("marginals", "and2", "and3", "conditionals", "calls", "est_usd_cap_price")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "weeks.json").write_text(json.dumps({"source": str(LG), "price_in_out_per_M": [wc.PRICE_IN, wc.PRICE_OUT],
                                                "weeks": rows, "total_all_weeks": tot}, indent=1))
    print(json.dumps(tot))


def short_label(q: str, opt: str | None, n: int = 40) -> str:
    q = re.sub(r"\s+", " ", q or "").strip()
    q = re.sub(r"^(Will|Which|Who|What|When|How many|How much|Does|Did|Is|Are|Can)\b\s*", "", q, flags=re.I)
    q = q.rstrip("?").strip()
    s = (opt.strip() + " · " + q) if opt else q
    s = s[0].upper() + s[1:] if s else s
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def cmd_build(out: Path, week: str) -> None:
    nodes, edges = load_graph()
    t = to_ms(week)
    op = open_at(nodes, t)
    store = AsOf()
    rng = random.Random(SEED)
    stats = Counter()
    events = []
    for n in sorted(op, key=lambda x: x["id"]):
        m = store.get_market(n["parent_id"], t)
        if m is None or m["status"] != "open":
            stats["drop_not_open_in_store"] += 1; continue
        logged = store.db.execute("SELECT count(*) FROM resolutions r JOIN markets mk ON mk.mid = r.mid "
                                  "WHERE mk.id = ? AND r.resolved <= ?", (n["parent_id"], t)).fetchone()[0]
        if logged:
            stats["drop_resolution_logged"] += 1; continue
        ans = None
        if n["answer_text"] is not None:
            aid = n["id"].split(":", 1)[1]
            ans = next((a for a in m["answers"] if a["id"] == aid), None)
            if ans is None or ans.get("resolution") is not None:
                stats["drop_option_not_open"] += 1; continue
        events.append({"id": n["id"], "parent_id": n["parent_id"], "parent_question": m["question"],
                       "answer_text": ans["text"] if ans else None, "criteria_text": m["description"] or "",
                       "closing_week": wc.monday(m["close"]) if m["close"] and m["close"] < to_ms("2100-01-01")
                       else "an unset date (no close date)",
                       "label": int(n["resolution"] == "YES"), "resolution_time": n["resolution_time"],
                       "outcome_type": n["outcome_type"], "url": n.get("url"),
                       "short": short_label(m["question"], ans["text"] if ans else None)})
        events[-1]["question"] = m["question"] + (f" — {events[-1]['answer_text']}" if ans else "")
    idx = {e["id"]: k for k, e in enumerate(events)}
    g = week_graph([{"id": e["id"]} for e in events], edges)
    links = sorted((min(idx[u], idx[v]), max(idx[u], idx[v])) for u, v in g.edges)
    rel = {}
    for u, v, d in g.edges(data=True):
        i, j = sorted((idx[u], idx[v]))
        rel[f"{i}-{j}"] = {"type": d["type"], "types": d["types"], "mutual": d["mutual"]}
    tris = sorted(tuple(sorted(idx[x] for x in T)) for T in triangles(g))
    cluster = {"as_of": week, "events": events}
    qs = [{"kind": "single", "f": wc.var(i)} for i in range(len(events))]
    for i, j in links:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        qs.append({"kind": "and2", "f": {"op": "and", "args": [wc.var(a), wc.var(b)]}})
        qs.append({"kind": "cond", "f": {"op": "cond", "given": wc.var(i), "then": wc.var(j)}})
        qs.append({"kind": "cond", "f": {"op": "cond", "given": wc.var(j), "then": wc.var(i)}})
    for T in tris:
        xs = list(T); rng.shuffle(xs)
        qs.append({"kind": "and3", "f": {"op": "and", "args": [wc.var(v) for v in xs]}})
    labels = [e["label"] for e in events]
    out.mkdir(parents=True, exist_ok=True)
    queries = []
    for k, q in enumerate(qs):
        msgs = wc.messages(cluster, q["f"])
        queries.append({"query_id": f"w-q{k:05d}", "kind": q["kind"], "events": wc.events_of(q["f"]),
                        "formula": q["f"], "formula_str": wc.formula_str(q["f"]), "label": wc.truth(q["f"], labels),
                        "messages": msgs, "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
    with (out / "events.jsonl").open("w") as f:
        for k, e in enumerate(events):
            f.write(json.dumps({**e, "index": k}) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    (out / "graph.json").write_text(json.dumps({"links": links, "triangles": tris, "rel": rel}))
    chars = sum(len(m["content"]) for q in queries for m in q["messages"])
    summary = {"week": week, "events_open_in_linkgraph": len(op), "events": len(events), "stats": stats,
               "links": len(links), "triangles": len(tris), "queries": len(queries),
               "queries_by_kind": Counter(q["kind"] for q in queries), "yes_rate": sum(labels) / max(1, len(labels)),
               "link_types": Counter(r["type"] for r in rel.values()), "prompt_chars": chars,
               "est_usd_cap_price": chars / CHARS_PER_TOKEN * wc.PRICE_IN / 1e6 + len(queries) * OUT_TOKENS * wc.PRICE_OUT / 1e6}
    (out / "build-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weeks", "build", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--week")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "weeks":
        cmd_weeks(a.out)
    elif a.cmd == "build":
        cmd_build(a.out, a.week)
    else:
        wc.run(a.out, min(a.workers, 8), a.max_usd, a.limit, a.key_secret)


if __name__ == "__main__":
    main()
