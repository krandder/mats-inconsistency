#!/usr/bin/env python3
"""Add a dependence stratum and both prices at t to every pair of a market-context dataset.

Stratum (outcome-blind, from market metadata only):
* ``same_market_siblings``: options of one multiple-choice market (mutually
  exclusive only if the market sums to one; none here do).  The pairing rule
  excludes same-parent pairs, so this is expected to be empty.
* ``same_topic``: the two parent markets share at least one Manifold group
  (``groupSlugs`` as of t, from the contract state at t).
* ``other``.

``pairs.jsonl`` gets ``stratum``, ``shared_groups``, ``p_price_t`` and
``q_price_t`` (no outcomes).  The outcome-dependence signal needs the labels,
so it goes to a separate analysis file, ``pair-outcomes.jsonl`` (never a model
input): both outcomes, the joint outcome, and whether both markets resolved
YES/NO.  The manifest gets per-stratum counts and the empirical dependence
(joint YES rate vs the product of marginals, phi).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import sqlite3
from pathlib import Path

SPLITS = ("train", "validation", "test")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def groups_at(db: sqlite3.Connection, cid: str, t_ms: int) -> set[str]:
    edits = db.execute("SELECT created_time, data FROM contract_edits WHERE contract_id=? ORDER BY created_time, id",
                       (cid,)).fetchall()
    k = bisect.bisect_left([e[0] for e in edits], t_ms)
    data = json.loads(edits[k][1]) if k < len(edits) else json.loads(
        db.execute("SELECT data FROM contracts WHERE id=?", (cid,)).fetchone()[0])
    return set(data.get("groupSlugs") or [])


def dependence(rows: list[tuple[int, int]]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    pp = sum(p for p, _ in rows) / n
    pq = sum(q for _, q in rows) / n
    both = sum(p and q for p, q in rows) / n
    den = math.sqrt(pp * (1 - pp) * pq * (1 - pq))
    return {"n": n, "p_yes": round(pp, 4), "q_yes": round(pq, 4), "both_yes": round(both, 4),
            "independence_both_yes": round(pp * pq, 4), "phi": round((both - pp * pq) / den, 4) if den else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    a = ap.parse_args()
    db = sqlite3.connect(a.index)
    manifest_path = a.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["strata"] = {}
    for split in SPLITS:
        d = a.data / split
        props = {r["id"]: r for r in read(d / "propositions.jsonl")}
        labels = {r["id"]: r["label"] for r in read(d / "single-targets.jsonl")}
        pairs = read(d / "pairs.jsonl")
        outcomes, by = [], collections.defaultdict(list)
        for pair in pairs:
            p, q = props[pair["p_id"]], props[pair["q_id"]]
            t = p["context"]["t_ms"]
            if p["parent_id"] == q["parent_id"]:
                stratum, shared = "same_market_siblings", []
            else:
                shared = sorted(groups_at(db, p["parent_id"], t) & groups_at(db, q["parent_id"], t))
                stratum = "same_topic" if shared else "other"
            pair.update(stratum=stratum, shared_groups=shared, p_price_t=p["context"]["price"]["p_t"],
                        q_price_t=q["context"]["price"]["p_t"])
            yp, yq = int(labels[pair["p_id"]]), int(labels[pair["q_id"]])
            outcomes.append({"pair_id": pair["pair_id"], "stratum": stratum, "p_label": yp, "q_label": yq,
                             "joint": f"{'Y' if yp else 'N'}{'Y' if yq else 'N'}", "both_resolved_binary": True})
            by[stratum].append((yp, yq))
        write(d / "pairs.jsonl", pairs)
        write(d / "pair-outcomes.jsonl", outcomes)
        manifest["strata"][split] = {s: dependence(v) for s, v in sorted(by.items())}
        manifest["strata"][split]["all"] = dependence([x for v in by.values() for x in v])
        print(split, json.dumps(manifest["strata"][split]))
    for path in sorted(a.data.rglob("*.jsonl")):
        manifest["files_sha256"][str(path.relative_to(a.data))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["strata_note"] = ("stratum is outcome-blind (same parent / shared Manifold group at t / other); "
                               "pair-outcomes.jsonl holds labels for dependence analysis only, never a model input")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
