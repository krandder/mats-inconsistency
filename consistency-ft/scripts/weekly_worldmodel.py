"""Weekly world model of Qwen3-32B over the TEST split's Luna link graph (Kelvin, 27 Sep 2026).

For each as-of Monday t of the test period, the events are the test nodes open and unresolved at t
(open_time < t < end_time, resolution_time > t, re-checked in the as-of store: market and option exist
and are open at t, no resolution logged at or before t; texts are the as-of-t versions).  Links are
Luna's pass-1 and pass-2 edges (edges.jsonl + edges-pass2.jsonl, every relation type, no embedding
bridges) with both ends open at t; triangles are the triangles of that week graph.

Queries, one statement per call, v6_direct question-only prompt (scripts/worldmodel_clusters.messages:
BASE_SYSTEM, "Assume today is t", question blocks, statement, ANSWER_DIRECT, " /no_think"):
  single  every open event           "Question A resolves YES"
  and2    every link                 "Both of the following are true: (1) ...; (2) ..."   (random order)
  and3    every triangle             "All of the following are true: ..."                 (random order)
20% of the and2 and 20% of the and3 queries of each week are marked held out (seeded per week).

Commands:
  plan    exact prompt characters per week (all Mondays) and the cost of every stride (1..6 weeks);
          chars -> tokens at the measured 3.85 chars/token of the 2025-06-23 example week.
  build   --weeks w1,w2,...   writes <out>/weeks/<t>/{events.jsonl,graph.json} and <out>/queries.jsonl
  run     qwen3-32b on DashScope, thinking off, temperature 0, max_tokens 20, parser + one retry,
          no substitution (a served model other than qwen3-32b stops the run), <= 12 workers, spend
          cap (conservative 0.7/2.8 $/M).  Answers with the identical prompt (sha256) from the
          example-week run are reused (same model, same prompt, temperature 0) and marked reused.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import worldmodel_clusters as wc  # noqa: E402
from example_week import short_label, triangles  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402
from training.asof import AsOf, to_ms  # noqa: E402

LG = Path("/home/agent/data/luna-linkgraph-20260927/test")
OUT = Path("/home/agent/data/weekly-worldmodel-20260927")
EXAMPLE = Path("/home/agent/data/example-week-20260927")
SEED = 20260927
HOLDOUT = 0.2
CHARS_PER_TOKEN = 3.85  # measured on the example week's usage (3.79 single .. 3.90 and3)
OUT_TOKENS = 7
MODEL = "qwen3-32b"


def load_graph():
    nodes = [json.loads(l) for l in (LG / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(l) for l in (LG / "edges.jsonl").read_text().splitlines()]
    p2 = LG / "edges-pass2.jsonl"
    seen = {(e["u"], e["v"]) for e in edges}
    n2 = 0
    if p2.exists():
        for e in map(json.loads, p2.read_text().splitlines()):
            if (e["u"], e["v"]) not in seen:
                edges.append(e); seen.add((e["u"], e["v"])); n2 += 1
    return nodes, edges, n2


def mondays(nodes):
    lo, hi = min(n["open_time"] for n in nodes), max(n["end_time"] for n in nodes)
    d, out = dt.date(2025, 5, 5), []
    while to_ms(d.isoformat()) < hi:
        if to_ms(d.isoformat()) > lo:
            out.append(d.isoformat())
        d += dt.timedelta(days=7)
    return out


class Weeks:
    def __init__(self):
        self.nodes, self.edges, self.n_pass2 = load_graph()
        self.store = AsOf()

    def events_at(self, week: str):
        t = to_ms(week)
        cand = [n for n in self.nodes if n["open_time"] < t < n["end_time"] and n["resolution_time"] > t]
        stats, events, mcache = Counter(), [], {}
        for n in sorted(cand, key=lambda x: x["id"]):
            if n["parent_id"] not in mcache:
                mcache[n["parent_id"]] = self.store.get_market(n["parent_id"], t)
            m = mcache[n["parent_id"]]
            if m is None or m["status"] != "open":
                stats["drop_not_open_in_store"] += 1; continue
            logged = self.store.db.execute("SELECT count(*) FROM resolutions r JOIN markets mk ON mk.mid = r.mid "
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
                           "end_time": n["end_time"], "open_time": n["open_time"],
                           "outcome_type": n["outcome_type"], "short": short_label(m["question"], ans["text"] if ans else None)})
        idx = {e["id"]: k for k, e in enumerate(events)}
        g = nx.Graph(); g.add_nodes_from(range(len(events)))
        rel = {}
        for e in self.edges:
            if e["u"] in idx and e["v"] in idx:
                i, j = sorted((idx[e["u"]], idx[e["v"]]))
                g.add_edge(i, j)
                rel[f"{i}-{j}"] = {"type": e["type"], "types": e["types"], "mutual": e.get("mutual"),
                                   "pass": 2 if e.get("proposer") == "luna-pass2" else 1}
        links = sorted(tuple(sorted(x)) for x in g.edges)
        tris = triangles(g)
        return events, links, tris, rel, stats, len(cand)


def _sha(msgs):
    return hashlib.sha256(json.dumps(msgs).encode()).hexdigest()


def week_queries(week, events, links, tris, cache: set | None = None):
    """cache: prompt hashes answered before; a link or triangle whose prompt in some order is there
    takes that order (the order was random there too), so the answer can be reused."""
    rng = random.Random(f"{SEED}-{week}")
    cluster = {"as_of": week, "events": events}
    fs = [("single", wc.var(i)) for i in range(len(events))]

    def pick(order, kind):
        f = {"op": "and", "args": [wc.var(v) for v in order]}
        if cache:
            for perm in itertools.permutations(order):
                g = {"op": "and", "args": [wc.var(v) for v in perm]}
                if _sha(wc.messages(cluster, g)) in cache:
                    return (kind, g)
        return (kind, f)
    for i, j in links:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        fs.append(pick((a, b), "and2"))
    for T in tris:
        xs = list(T); rng.shuffle(xs)
        fs.append(pick(xs, "and3"))
    # held-out: 20% of and2 and 20% of and3, stratified per week and kind
    held = set()
    for kind in ("and2", "and3"):
        ks = [k for k, (kd, _) in enumerate(fs) if kd == kind]
        rng.shuffle(ks)
        held |= set(ks[: round(HOLDOUT * len(ks))])
    labels = [e["label"] for e in events]
    out = []
    for k, (kind, f) in enumerate(fs):
        msgs = wc.messages(cluster, f)
        out.append({"query_id": f"{week}-q{k:05d}", "week": week, "kind": kind, "events": wc.events_of(f),
                    "formula_str": wc.formula_str(f), "label": wc.truth(f, labels), "heldout": k in held,
                    "messages": msgs, "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
    return out


def cmd_plan(out: Path):
    W = Weeks()
    rows = []
    for week in mondays(W.nodes):
        events, links, tris, rel, stats, n_cand = W.events_at(week)
        qs = week_queries(week, events, links, tris)
        chars = Counter()
        for q in qs:
            chars[q["kind"]] += sum(len(m["content"]) for m in q["messages"])
        tok = sum(chars.values()) / CHARS_PER_TOKEN
        usd = tok * wc.PRICE_IN / 1e6 + len(qs) * OUT_TOKENS * wc.PRICE_OUT / 1e6
        tw = max((nx.algorithms.approximation.treewidth_min_fill_in(c)[0] for c in
                  (nx.Graph(links).subgraph(cc) for cc in nx.connected_components(nx.Graph(links)))), default=0) if links else 0
        r = {"week": week, "candidates": n_cand, "events": len(events), "links": len(links), "triangles": len(tris),
             "calls": len(qs), "prompt_tokens_est": round(tok), "usd_est": round(usd, 4), "treewidth_minfill": tw,
             "drops": dict(stats), "link_types": dict(Counter(v["type"] for v in rel.values())),
             "pass2_links": sum(v["pass"] == 2 for v in rel.values())}
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("week", "events", "links", "triangles", "calls", "usd_est", "treewidth_minfill")}), flush=True)
    strides = {}
    for s in range(1, 7):
        strides[s] = []
        for off in range(s):
            sel = rows[off::s]
            strides[s].append({"stride": s, "offset": off, "weeks": [r["week"] for r in sel],
                               "usd_est": round(sum(r["usd_est"] for r in sel), 2), "calls": sum(r["calls"] for r in sel)})
            print(f"stride {s} offset {off}: {len(sel)} weeks, {strides[s][-1]['calls']} calls, ${strides[s][-1]['usd_est']}"
                  f"{' (has 2025-06-23)' if '2025-06-23' in strides[s][-1]['weeks'] else ''}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"graph": str(LG), "pass2_edges_used": W.n_pass2,
                                               "price_in_out_per_M": [wc.PRICE_IN, wc.PRICE_OUT],
                                               "chars_per_token": CHARS_PER_TOKEN, "weeks": rows, "strides": strides}, indent=1))


def cmd_build(out: Path, weeks: list[str]):
    W = Weeks()
    ex_week = json.loads((EXAMPLE / "build-summary.json").read_text())["week"] if (EXAMPLE / "log.jsonl").exists() else None
    cache = {json.loads(l)["prompt_sha256"] for l in (EXAMPLE / "log.jsonl").read_text().splitlines()} if ex_week else set()
    allq = []
    summ = []
    for week in weeks:
        events, links, tris, rel, stats, n_cand = W.events_at(week)
        qs = week_queries(week, events, links, tris, cache if week == ex_week else None)
        d = out / "weeks" / week
        d.mkdir(parents=True, exist_ok=True)
        with (d / "events.jsonl").open("w") as f:
            for k, e in enumerate(events):
                f.write(json.dumps({**e, "index": k}) + "\n")
        (d / "graph.json").write_text(json.dumps({"links": links, "triangles": tris, "rel": rel}))
        allq += qs
        summ.append({"week": week, "candidates": n_cand, "events": len(events), "links": len(links),
                     "triangles": len(tris), "calls": len(qs), "drops": dict(stats),
                     "heldout": Counter(q["kind"] for q in qs if q["heldout"])})
        print(json.dumps(summ[-1]), flush=True)
    with (out / "queries.jsonl").open("w") as f:
        for q in allq:
            f.write(json.dumps(q) + "\n")
    chars = sum(len(m["content"]) for q in allq for m in q["messages"])
    tot = {"weeks": weeks, "calls": len(allq), "by_kind": Counter(q["kind"] for q in allq), "prompt_chars": chars,
           "usd_est": chars / CHARS_PER_TOKEN * wc.PRICE_IN / 1e6 + len(allq) * OUT_TOKENS * wc.PRICE_OUT / 1e6,
           "pass2_edges_used": W.n_pass2, "per_week": summ}
    (out / "build-summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps({k: v for k, v in tot.items() if k != "per_week"}))


def cmd_run(out: Path, workers: int, max_usd: float, key_secret: str):
    import precutoff_client as pc
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()]
    log = out / "log.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[r["query_id"]] = r
    # reuse identical prompts answered in the example-week run (same model, settings, prompt)
    ex = {}
    if (EXAMPLE / "log.jsonl").exists():
        for l in (EXAMPLE / "log.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r.get("p") is not None and r.get("served_model") == MODEL:
                ex[r["prompt_sha256"]] = r
    reused = 0
    with log.open("a") as f:
        for q in rows:
            if q["query_id"] not in done and q["prompt_sha256"] in ex:
                r = dict(ex[q["prompt_sha256"]])
                r.update(query_id=q["query_id"], reused_from=str(EXAMPLE / "log.jsonl"), original_cost=r["cost"], cost=0.0)
                f.write(json.dumps(r) + "\n"); done[q["query_id"]] = r; reused += 1
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False, "temperature": 0}, wc.PRICE_IN, wc.PRICE_OUT)
    spent = [sum(r["cost"] for r in done.values())]
    todo = [r for r in rows if r["query_id"] not in done]
    random.Random(SEED).shuffle(todo)  # representative early projection
    print(f"{len(rows)} calls, {len(done)} done ({reused} reused now), {len(todo)} to run; spent ${spent[0]:.3f}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    n_run = [0]
    reason = [None]

    def one(row):
        if stop.is_set():
            return None
        msgs = row["messages"]
        first = pc.chat(MODEL, msgs, max_tokens=20)
        if first["served_model"] and first["served_model"] != MODEL:
            reason[0] = f"served model {first['served_model']} != {MODEL}"; stop.set(); return None
        p, why = vb.parse_probability(first["text"])
        rec = {"query_id": row["query_id"], "prompt_sha256": row["prompt_sha256"], "raw": first["text"],
               "served_model": first["served_model"], "usage": first["usage"], "errors": first["errors"],
               "finish_reason": first["finish_reason"], "parsed_first": p, "reason_first": why, "retried": False,
               "raw_retry": None, "p": p, "reason": why}
        cost = wc.cost_of(first["usage"])
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=20)
            p2, why2 = vb.parse_probability(second["text"])
            cost += wc.cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(min(workers, 12)) as pool, log.open("a") as f:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == 300:
                    proj = spent[0] + spent[0] / n_run[0] * (len(todo) - n_run[0]) if not done else \
                        spent[0] + (spent[0] - sum(r["cost"] for r in done.values())) / n_run[0] * (len(todo) - n_run[0])
                    print(f"projection after 300: ${proj:.2f} (cap ${max_usd})", flush=True)
                    if proj > max_usd:
                        reason[0] = "projection exceeds cap"; stop.set()
                if spent[0] >= max_usd:
                    reason[0] = "spend reached cap"; stop.set()
                if n_run[0] % 1000 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {len(done) + n_run[0]}/{len(rows)} spent ${spent[0]:.3f}", flush=True)
    recs = {json.loads(l)["query_id"]: json.loads(l) for l in log.read_text().splitlines()}
    print(f"{'STOPPED (' + str(reason[0]) + ') ' if stop.is_set() else ''}logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs.values())}, missing {sum(r['p'] is None for r in recs.values())}, "
          f"spend ${sum(r['cost'] for r in recs.values()):.3f}", flush=True)


def main():
    global LG
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build", "run"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--weeks")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    ap.add_argument("--graph", type=Path, default=LG, help="link-graph split dir (default: the test split)")
    a = ap.parse_args()
    LG = a.graph
    if a.cmd == "plan":
        cmd_plan(a.out)
    elif a.cmd == "build":
        cmd_build(a.out, a.weeks.split(","))
    else:
        cmd_run(a.out, a.workers, a.max_usd, a.key_secret)


if __name__ == "__main__":
    main()
