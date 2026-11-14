#!/usr/bin/env python3
"""Does the context agent's selection lean toward the eventual outcome?

Outcome-blind direction of an item on proposition X (known at t, from the record itself):
  * a comment on X's market with a bet attached, or with the commenter's position recorded
    at comment time, on X's option (or the binary market): direction = YES/NO of that bet/position;
  * a related market that had resolved YES/NO before t and whose question nearly matches X
    (token Jaccard >= 0.5): direction = its resolution.
Metric B (denser): among chosen related binary markets still open at t, the share whose price at t
was on the side they later resolved.  A hindsight-driven selector would favour markets that "called it".
For each arm (agent selection, deterministic pack, and the pool of all eligible comments on the
pair markets before t) we report the share of directional items agreeing with X's eventual label,
and the correlation across propositions of the arm's lean (agree - disagree)/n with the market
residual (label - price at t).  If the agent leaned on hindsight, its share and correlation would
exceed the pack's and the pool's.  Labels are read only here, after the contexts are frozen.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from training import asof  # noqa: E402
from audit_asof import jaccard  # noqa: E402

PAGES = Path("/home/agent/data/manifold-history/pages/contract_comments")


def comment_directions(contract_ids: set[str]) -> dict[str, list[tuple[str | None, str]]]:
    """comment id -> [(answer_id or None, 'YES'/'NO'), ...] from the bet and the position at comment time."""
    out = {}
    for path in sorted(glob.glob(str(PAGES / "*.json.gz"))):
        with gzip.open(path) as fh:
            rows = json.load(fh)
        for r in rows:
            if r.get("contract_id") not in contract_ids:
                continue
            d = r.get("data") or {}
            dirs = []
            if d.get("betOutcome") in ("YES", "NO"):
                dirs.append((d.get("betAnswerId"), d["betOutcome"]))
            for pre in ("commenterPosition", "commentorPosition"):
                if d.get(pre + "Outcome") in ("YES", "NO"):
                    dirs.append((d.get(pre + "AnswerId"), d[pre + "Outcome"]))
            if dirs:
                out[r["comment_id"]] = dirs
    return out


def direction_on(dirs, answer_id):
    for aid, outcome in dirs or []:
        if (answer_id is None and aid is None) or (answer_id is not None and aid == answer_id):
            return outcome
    return None


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="/home/agent/data/manifold-asof-20260923/context-agent/validation.jsonl")
    ap.add_argument("--dataset", default="/home/agent/data/jev-market-context-open-20260923")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = [json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r.get("context")]
    root = Path(a.dataset) / a.split
    labels = {r["id"]: r["label"] for r in map(json.loads, (root / "single-targets.jsonl").read_text().splitlines())}
    store = asof.AsOf()
    cids = {asof.split_prop(r[k])[0] for r in recs for k in ("p_id", "q_id")}
    dirs = comment_directions(cids)
    arms = {"agent": [0, 0], "pack_large": [0, 0], "pool": [0, 0]}
    lean = {k: ([], []) for k in arms}
    per_pair = []
    called = {"agent": [0, 0], "pack_large": [0, 0]}
    for r in recs:
        t = r["t"]
        pair = {"p_id": r["p_id"], "q_id": r["q_id"]}
        _, pack_items = asof.build_pack(pair, t, "large", store=store, return_items=True)
        agent_comments = {it["comment_id"] for it in r["used"] if it["kind"] == "comment"}
        agent_markets = {asof.split_prop(it["market_id"])[0] for it in r["used"] if it["kind"] == "market"}
        row = {"pair_id": r["pair_id"]}
        for pid in (r["p_id"], r["q_id"]):
            y = labels.get(pid)
            if y is None:
                continue
            cid, aid = asof.split_prop(pid)
            m = store.get_market(cid, t)
            if aid:
                p_t = next((x["prob"] for x in m["answers"] if x["id"] == aid), None)
            else:
                p_t = m["prob"]
            question = m["question"] + (" " + next((x["text"] for x in m["answers"] if x["id"] == aid), "") if aid else "")
            pool = [c["id"] for c in store.comments(cid, t, limit=10_000)]
            sets = {"agent": set(agent_comments) & set(pool), "pack_large": set(pack_items["comments"]) & set(pool),
                    "pool": set(pool)}
            mk = {"agent": agent_markets - {cid}, "pack_large": set(pack_items["related"]), "pool": set()}
            for arm in arms:
                agree = disagree = 0
                for c in sets[arm]:
                    d = direction_on(dirs.get(c), aid)
                    if d is None:
                        continue
                    agree += (d == "YES") == bool(y)
                    disagree += (d == "YES") != bool(y)
                for other in mk[arm]:
                    om = store.get_market(other, t, answers=False)
                    if om and om["status"] == "resolved" and om["resolution"] in ("YES", "NO") \
                            and jaccard(question, om["question"]) >= 0.5:
                        agree += (om["resolution"] == "YES") == bool(y)
                        disagree += (om["resolution"] == "YES") != bool(y)
                arms[arm][0] += agree
                arms[arm][1] += disagree
                n = agree + disagree
                if n and p_t is not None:
                    lean[arm][0].append((agree - disagree) / n)
                    lean[arm][1].append(y - p_t)
                row[f"{pid}:{arm}"] = [agree, disagree]
        # metric B: chosen related binary markets open at t -- was their price at t on the side they later resolved?
        for arm, ids in (("agent", agent_markets - {asof.split_prop(r["p_id"])[0], asof.split_prop(r["q_id"])[0]}),
                         ("pack_large", set(pack_items["related"]))):
            for other in ids:
                om = store.get_market(other, t, answers=False)
                row_now = store.db.execute("SELECT resolution, resolution_time, outcome_type FROM markets WHERE id = ?",
                                           (other,)).fetchone()
                if not om or om["status"] == "resolved" or om["prob"] is None or row_now["outcome_type"] != "BINARY" \
                        or row_now["resolution"] not in ("YES", "NO") or abs(om["prob"] - 0.5) < 1e-9:
                    continue
                called[arm][0] += (om["prob"] > 0.5) == (row_now["resolution"] == "YES")
                called[arm][1] += 1
        per_pair.append(row)
    report = {"pairs": len(recs), "arms": {k: {"agree": v[0], "disagree": v[1],
                                               "share_agree": round(v[0] / (v[0] + v[1]), 3) if sum(v) else None,
                                               "props_with_directional_items": len(lean[k][0]),
                                               "corr_lean_vs_label_minus_price": (
                                                   round(corr(*lean[k]), 3) if corr(*lean[k]) is not None else None)}
                                           for k, v in arms.items()},
              "related_markets_that_called_their_own_outcome": {
                  k: {"called": v[0], "n": v[1], "share": round(v[0] / v[1], 3) if v[1] else None}
                  for k, v in called.items()},
              "per_pair": per_pair}
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != "per_pair"}, indent=1))


if __name__ == "__main__":
    main()
