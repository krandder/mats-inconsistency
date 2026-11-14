"""Global world model at one timestamp: many events, marginals + pair + triple queries (Qwen3-32B, DashScope).

Question (Kelvin, 24 Sep): how much structure is in Qwen3-32B's beliefs over MANY events at a
single date t?  Analysis only; the joint is fitted post hoc (scripts/worldmodel_global_fit.py).

build   t = 2025-09-15 00:00 UTC.  Events are Manifold binary markets and multiple-choice options:
        * parents from the TEST split of jev-disjoint-v1-filtered-20260919 first, then "unused"
          parents (in no split of jev-disjoint-v1, jev-disjoint-v1-filtered or jev-pairings-v1);
          never a train/validation parent, never one of the 178 excluded parents;
        * public; market (and option) created before t; open at t (not closed, not resolved, no
          resolution logged at or before t); final resolution YES/NO strictly after t (after the
          Qwen3 cutoff window: every test-set event closes in 2025-05 or later, t is 4.5 months
          after the Qwen3 release);
        * unused parents need >= 10 traders before t (outcome-blind quality filter);
        * at most 4 options per multiple-choice parent (most bets before t; outcome-blind).
        Text is the as-of-t version from the store (training/asof.py).
        Similarity: BGE-small (bge-small-en-v1.5) cosine on "question [option]" as of t.
        Queries (one per call, formula recorded):
          single   every event
          sim      union of each event's top-10 neighbours: "A and B" (random order)
          rand     as many random non-neighbour pairs: "A and B"
          andnot   10% of sim and of rand pairs also get "A and not B" (random direction)
          and3     ~200 3-way ANDs among triangles of the neighbour graph (highest mean cosine,
                   each event in at most 2)
        Prompt: v6_direct, as in scripts/worldmodel_clusters.py (BASE_SYSTEM, "Assume today is t",
        question blocks, statement, ANSWER_DIRECT, " /no_think").  No market data.
run     worldmodel_clusters.run (qwen3-32b, thinking off, temperature 0, max_tokens 20, one retry,
        missing recorded, no model substitution, spend cap).

Usage:
  worldmodel_global.py build --out /home/agent/data/worldmodel-global-20260924
  worldmodel_global.py run   --out /home/agent/data/worldmodel-global-20260924 --workers 12 --max-usd 5.5
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import worldmodel_clusters as wc  # noqa: E402
from training.asof import AsOf, to_ms  # noqa: E402

T_ISO = "2025-09-15"
SEED = 20260924
TARGET = 400
K_NEIGH = 10
ANDNOT_FRAC = 0.10
N_TRIPLES = 200
MAX_OPTS = 4
MIN_TRADERS_UNUSED = 10
DATA = Path("/home/agent/data")
EMB_MODEL, EMB_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def parents_of(root: Path, splits) -> set[str]:
    out = set()
    for s in splits:
        p = root / s / "propositions.jsonl"
        if p.exists():
            out |= {json.loads(l)["parent_id"] for l in p.read_text().splitlines() if l.strip()}
    return out


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL, revision=EMB_REV)
    model = AutoModel.from_pretrained(EMB_MODEL, revision=EMB_REV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            v = model(**b).last_hidden_state[:, 0]  # CLS pooling (sentence-transformers config of bge)
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(out).astype(np.float32)


def select_events(store: AsOf, t: int, rng: random.Random) -> tuple[list[dict], dict]:
    db = store.db
    filt = DATA / "jev-disjoint-v1-filtered-20260919"
    test = parents_of(filt, ["test"])
    excluded = set(json.loads((filt / "excluded-parent-ids.json").read_text()))
    trainval = set()
    for root in (filt, DATA / "jev-disjoint-v1", DATA / "jev-pairings-v1"):
        trainval |= parents_of(root, ["train", "validation"])
    every = test | trainval
    for root in (DATA / "jev-disjoint-v1", DATA / "jev-pairings-v1"):
        every |= parents_of(root, ["test"])
    test -= excluded
    assert not test & trainval, "test parent also in a train/validation split"
    stats = Counter()
    rows = db.execute("SELECT id, mid, outcome_type, resolution, resolution_time, created, visibility FROM markets "
                      "WHERE created < ? AND resolution_time > ? AND outcome_type IN ('BINARY','MULTIPLE_CHOICE') "
                      "AND coalesce(deleted, 0) = 0", (t, t)).fetchall()
    events_by_source = {"test": [], "unused": []}
    for r in rows:
        cid = r["id"]
        if cid in excluded or cid in trainval:
            stats["skip_train_or_excluded"] += 1; continue
        src = "test" if cid in test else ("unused" if cid not in every else None)
        if src is None:
            stats["skip_other_split"] += 1; continue
        if r["visibility"] != "public":
            stats["skip_private"] += 1; continue
        n_logged = db.execute("SELECT count(*) FROM resolutions WHERE mid = ? AND resolved <= ?", (r["mid"], t)).fetchone()[0]
        m = store.get_market(cid, t)
        if m is None or m["status"] != "open" or n_logged:
            stats["skip_not_open"] += 1; continue
        if src == "unused" and m["n_traders"] < MIN_TRADERS_UNUSED:
            stats["skip_unused_thin"] += 1; continue
        if r["outcome_type"] == "BINARY":
            if r["resolution"] not in ("YES", "NO"):
                stats["skip_binary_not_yes_no"] += 1; continue
            events_by_source[src].append(event_row(m, None, int(r["resolution"] == "YES"), r["resolution_time"], src))
        else:
            final = {a[0]: (a[1], a[2]) for a in db.execute("SELECT id, resolution, resolution_time FROM answers WHERE mid = ?", (r["mid"],))}
            opts = []
            for a in m["answers"]:
                res, rt = final.get(a["id"], (None, None))
                if a["is_other"] or a["resolution"] is not None or res not in ("YES", "NO") or rt is None or rt <= t:
                    continue
                opts.append((-(a["n_bets"] or 0), a["id"], a, res, rt))
            opts.sort(key=lambda x: (x[0], x[1]))
            if not opts:
                stats["skip_mc_no_option"] += 1; continue
            for _, _, a, res, rt in opts[:MAX_OPTS]:
                events_by_source[src].append(event_row(m, a, int(res == "YES"), rt, src))
    events = sorted(events_by_source["test"], key=lambda e: e["id"])
    if len(events) > TARGET:
        events = sorted(rng.sample(events, TARGET), key=lambda e: e["id"])
    unused = sorted(events_by_source["unused"], key=lambda e: e["id"])
    rng.shuffle(unused)
    # fill by parent so a multiple-choice parent enters with all its chosen options
    by_parent = defaultdict(list)
    order = []
    for e in unused:
        if e["parent_id"] not in by_parent:
            order.append(e["parent_id"])
        by_parent[e["parent_id"]].append(e)
    for pid in order:
        if len(events) >= TARGET:
            break
        events += by_parent[pid]
    stats.update({f"available_{k}": len(v) for k, v in events_by_source.items()})
    return events, stats


def closing_week(close: int | None) -> str:
    if close is None or close >= to_ms("2100-01-01"):
        return "an unset date (no close date)"
    return wc.monday(close)


def event_row(m: dict, a: dict | None, label: int, res_time: int, src: str) -> dict:
    return {"id": m["id"] + (f":{a['id']}" if a else ""), "parent_id": m["id"], "source": src,
            "parent_question": m["question"], "answer_text": a["text"] if a else None,
            "criteria_text": m["description"] or "", "closing_week": closing_week(m["close"]), "label": label, "resolution_time": res_time,
            "outcome_type": m["outcome_type"], "groups": m["groups"],
            "prob_at_t": (a["prob"] if a else m["prob"]), "n_traders": m["n_traders"]}


def build(out: Path) -> None:
    t = to_ms(T_ISO)
    rng = random.Random(SEED)
    store = AsOf()
    events, stats = select_events(store, t, rng)
    n = len(events)
    texts = [e["parent_question"] + (f" — {e['answer_text']}" if e["answer_text"] else "") for e in events]
    emb = embed(texts)
    cos = emb @ emb.T
    np.fill_diagonal(cos, -1)
    sim = set()
    for i in range(n):
        for j in np.argsort(-cos[i])[:K_NEIGH]:
            sim.add((min(i, int(j)), max(i, int(j))))
    sim = sorted(sim)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in set(sim)]
    rand = sorted(rng.sample(all_pairs, len(sim)))
    # triangles of the neighbour graph, by mean cosine, each event in at most 2
    adj = defaultdict(set)
    for i, j in sim:
        adj[i].add(j); adj[j].add(i)
    tri = set()
    for i, j in sim:
        for k in adj[i] & adj[j]:
            tri.add(tuple(sorted((i, j, k))))
    tri = sorted(tri, key=lambda x: (-(cos[x[0], x[1]] + cos[x[0], x[2]] + cos[x[1], x[2]]), x))
    used, triples = Counter(), []
    for x in tri:
        if len(triples) >= N_TRIPLES:
            break
        if all(used[v] < 2 for v in x):
            triples.append(x); used.update(x)

    cluster = {"as_of": T_ISO, "events": events}
    qs = [{"kind": "single", "pair_type": None, "f": wc.var(i)} for i in range(n)]
    for kind, pairs in (("sim", sim), ("rand", rand)):
        for i, j in pairs:
            a, b = (i, j) if rng.random() < 0.5 else (j, i)
            qs.append({"kind": "and2", "pair_type": kind, "f": {"op": "and", "args": [wc.var(a), wc.var(b)]}})
        for i, j in rng.sample(pairs, round(ANDNOT_FRAC * len(pairs))):
            a, b = (i, j) if rng.random() < 0.5 else (j, i)
            qs.append({"kind": "andnot", "pair_type": kind,
                       "f": {"op": "and", "args": [wc.var(a), {"op": "not", "a": wc.var(b)}]}})
    for x in triples:
        xs = list(x); rng.shuffle(xs)
        qs.append({"kind": "and3", "pair_type": "sim", "f": {"op": "and", "args": [wc.var(v) for v in xs]}})

    out.mkdir(parents=True, exist_ok=True)
    labels = [e["label"] for e in events]
    queries = []
    for k, q in enumerate(qs):
        msgs = wc.messages(cluster, q["f"])
        queries.append({"query_id": f"g-q{k:05d}", "kind": q["kind"], "pair_type": q["pair_type"],
                        "events": wc.events_of(q["f"]), "formula": q["f"], "formula_str": wc.formula_str(q["f"]),
                        "label": wc.truth(q["f"], labels), "messages": msgs,
                        "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
    for k, e in enumerate(events):
        e["index"] = k
    with (out / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    np.save(out / "embeddings.npy", emb)
    (out / "graph.json").write_text(json.dumps({"sim_pairs": sim, "rand_pairs": rand, "triples": triples,
                                                "sim_cosine": [float(cos[i, j]) for i, j in sim],
                                                "rand_cosine": [float(cos[i, j]) for i, j in rand]}))
    chars = sum(len(m["content"]) for q in queries for m in q["messages"])
    summary = {"t": T_ISO, "events": n, "by_source": Counter(e["source"] for e in events),
               "by_type": Counter(e["outcome_type"] for e in events), "parents": len({e["parent_id"] for e in events}),
               "yes_rate": sum(labels) / n, "queries": len(queries),
               "queries_by": Counter(f"{q['kind']}/{q['pair_type']}" for q in queries),
               "sim_pairs": len(sim), "rand_pairs": len(rand), "triangles_available": len(tri), "triples": len(triples),
               "stats": stats, "prompt_chars": chars,
               "est_cost_cap_price": chars / 3.5 * wc.PRICE_IN / 1e6 + len(queries) * 8 * wc.PRICE_OUT / 1e6,
               "embedding": {"model": EMB_MODEL, "revision": EMB_REV, "pooling": "cls", "normalized": True}}
    (out / "build-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=5.5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out)
    else:
        wc.run(a.out, min(a.workers, 12), a.max_usd, a.limit, a.key_secret)


if __name__ == "__main__":
    main()
