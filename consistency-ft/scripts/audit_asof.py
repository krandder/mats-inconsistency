#!/usr/bin/env python3
"""Leak audit, determinism, coverage and duplicate-resolution report for the as-of evidence packs.

For random JEV pairs: build each pack twice (must be byte-identical) and check every
item and every date in it against the raw store with independent SQL:
  * markets and comments used were created before t;
  * every rendered date of a comment, price point or resolution is before t;
  * a pair market resolved at/after t is never shown as resolved;
  * no comment written at/after t on a pair market appears verbatim.
Also counts pairs with >=1 related market that traded before t, pack token sizes,
and propositions whose answer is effectively revealed by a near-duplicate market that
resolved before t (legitimate information at t, reported only).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import asof  # noqa: E402

DATASET = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_split(split: str):
    root = DATASET / split
    pairs = [json.loads(x) for x in (root / "pairs.jsonl").read_text().splitlines() if x.strip()]
    props = {p["id"]: p for p in (json.loads(x) for x in (root / "propositions.jsonl").read_text().splitlines() if x.strip())}
    labels = {}
    lab = root / "single-targets.jsonl"
    if lab.exists():
        labels = {r["id"]: r["label"] for r in (json.loads(x) for x in lab.read_text().splitlines() if x.strip())}
    return pairs, props, labels


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in asof.STOPWORDS and len(w) > 1}


def jaccard(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def description_at(db: sqlite3.Connection, mid: int, t: int) -> tuple[str, str]:
    """Question/description at t straight from the raw edits pre-images (independent of market_versions)."""
    q, d = db.execute("SELECT question, description FROM markets WHERE mid = ?", (mid,)).fetchone()
    prev = None
    for when, eq, ed in db.execute("SELECT time, question, description FROM edits WHERE mid = ? ORDER BY time, eid",
                                   (mid,)).fetchall():
        ed = prev if ed == "\x00same" else ed
        prev = ed if ed is not None else prev
        if when >= t:
            return (eq if eq is not None else q), (ed if ed is not None else d)
    return q, d


def check_pack(db: sqlite3.Connection, text: str, items: dict, pair: dict, t: int) -> list[str]:
    leaks = []
    t_date = asof.iso(t)
    for cid in items["markets"] + items["related"]:
        row = db.execute("SELECT created FROM markets WHERE id = ?", (cid,)).fetchone()
        if row is None or row[0] >= t:
            leaks.append(f"market {cid} created at/after t")
    for c in items["comments"]:
        row = db.execute("SELECT time FROM comments WHERE id = ?", (c,)).fetchone()
        if row is None or row[0] >= t:
            leaks.append(f"comment {c} at/after t")
    for line in text.splitlines():
        dates = []
        if line.startswith(("- [", "  [")):
            dates += DATE.findall(line[:16])
        if line.startswith("Price history:"):
            dates += DATE.findall(line)
        for m in re.finditer(r"resolved (?:\S+|\"[^\"]*\")(?: at [0-9.]+%)? on (\d{4}-\d{2}-\d{2})", line):
            dates.append(m.group(1))
        leaks += [f"date {d} >= t in: {line[:120]}" for d in dates if d >= t_date]
    blocks = text.split("## Related markets")[0].split("## Question ")
    for pid, block in zip((pair["p_id"], pair["q_id"]), blocks[1:]):
        cid, aid = asof.split_prop(pid)
        mid, res_t = db.execute("SELECT mid, resolution_time FROM markets WHERE id = ?", (cid,)).fetchone()
        if res_t is not None and res_t >= t and re.search(r"; resolved (?:\S+|\"[^\"]*\") on", block.split("\n")[1]):
            leaks.append(f"pair market {cid} shown resolved before its resolution")
        if aid:
            a = db.execute("SELECT resolution_time FROM answers WHERE id = ?", (aid,)).fetchone()
            opt = [x for x in block.split("\n") if x.startswith("Option ")]
            if a and a[0] is not None and a[0] >= t and opt and "; resolved " in opt[0]:
                leaks.append(f"pair option {aid} shown resolved before its resolution")
        pre_t = None
        for (body,) in db.execute("SELECT text FROM comments WHERE mid = ? AND time >= ?", (mid, t)):
            body = re.sub(r"\s+", " ", body or "").strip()
            if len(body) >= 40 and body[:60] in text:
                if pre_t is None:  # text that also existed before t (description, earlier comments) is not a leak
                    q, d = description_at(db, mid, t)
                    olds = [x for (x,) in db.execute("SELECT text FROM comments WHERE mid = ? AND time < ?", (mid, t))]
                    pre_t = re.sub(r"\s+", " ", " ".join([q or "", d or ""] + [o or "" for o in olds]))
                if body[:60] not in pre_t:
                    leaks.append(f"post-t comment text of {cid} appears")
    return leaks


def duplicates(store: asof.AsOf, prop_id: str, t: int, label, threshold: float = 0.6) -> list[dict]:
    """Markets resolved before t whose question nearly matches the proposition's question."""
    cid, aid = asof.split_prop(prop_id)
    m = store.get_market(cid, t)
    if m is None:
        return []
    question = m["question"]
    if aid:
        ans = [a for a in m["answers"] if a["id"] == aid]
        question += " " + (ans[0]["text"] if ans else "")
    out = []
    for r in store.search_markets(question, t, k=20, exclude=[cid]):
        other = store.get_market(r["id"], t, answers=False)
        if other is None or other["status"] != "resolved":
            continue
        sim = jaccard(question, other["question"])
        if sim >= threshold:
            agrees = None
            if label is not None and other["resolution"] in ("YES", "NO"):
                agrees = (other["resolution"] == "YES") == bool(label)
            out.append({"id": r["id"], "question": other["question"], "similarity": round(sim, 3),
                        "resolution": other["resolution"], "resolved": asof.iso(other["resolution_time"]),
                        "agrees_with_label": agrees})
    return out


def audit(store: asof.AsOf, n_pairs: int = 200, seed: int = 20260923, sizes=("small", "large"),
          split: str = "test") -> dict:
    db = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True)
    pairs, props, labels = load_split(split)
    sample = random.Random(seed).sample(pairs, min(n_pairs, len(pairs)))
    leaks, nondet, examples = 0, 0, []
    tokens = {s: [] for s in sizes}
    with_priced_related, dup_props, dup_agree, dup_examples, n_props = 0, 0, 0, [], 0
    for pair in sample:
        t = asof.to_ms(asof.pair_as_of(pair, props))
        priced = False
        for size in sizes:
            text, items = asof.build_pack(pair, t, size, store=store, return_items=True)
            if asof.build_pack(pair, t, size, store=store) != text:
                nondet += 1
            found = check_pack(db, text, items, pair, t)
            leaks += len(found)
            examples += [{"pair": pair["pair_id"], "size": size, "leak": x} for x in found[:3]]
            tokens[size].append(asof.count_tokens(text))
            for rid in items["related"]:
                m = store.get_market(rid, t, answers=False)
                if m and m["n_trades"] > 0:
                    priced = True
        with_priced_related += priced
        for pid in (pair["p_id"], pair["q_id"]):
            n_props += 1
            d = duplicates(store, pid, t, labels.get(pid))
            if d:
                dup_props += 1
                dup_agree += any(x["agrees_with_label"] for x in d)
                if len(dup_examples) < 8:
                    dup_examples.append({"prop": pid, "t": asof.iso(t), "duplicates": d[:2]})
    summary = {s: {"median": statistics.median(v), "p90": sorted(v)[int(0.9 * (len(v) - 1))], "max": max(v)}
               for s, v in tokens.items() if v}
    return {"split": split, "pairs": len(sample), "sizes": list(sizes), "leaks": leaks, "leak_examples": examples[:20],
            "nondeterministic": nondet, "pairs_with_priced_related_market": with_priced_related,
            "pack_tokens": summary, "tokenizer": "qwen3" if asof._tokenizer() is not None else "chars/3.5",
            "propositions": n_props, "props_with_near_duplicate_resolved_before_t": dup_props,
            "props_where_a_duplicate_agrees_with_label": dup_agree, "duplicate_examples": dup_examples}


def main():
    global DATASET
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="/home/agent/data/jev-disjoint-v1-filtered-20260919")
    ap.add_argument("--out", default="/home/agent/data/manifold-asof-20260923/audit-test200.json")
    a = ap.parse_args()
    DATASET = Path(a.dataset)
    store = asof.AsOf()
    report = audit(store, a.pairs, a.seed, split=a.split)
    db = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True)
    report["store"] = json.loads(db.execute("SELECT value FROM meta WHERE key = 'build'").fetchone()[0])
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("duplicate_examples", "store")}, indent=1))


if __name__ == "__main__":
    main()
