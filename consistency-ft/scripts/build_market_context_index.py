#!/usr/bin/env python3
"""Build a per-contract SQLite index of Manifold history for the JEV parents.

One streaming pass over the complete Manifold page dump
(/home/agent/data/manifold-history/pages, finished 2026-09-18) keeps only rows
whose contract is a parent of a proposition in the JEV pair dataset:

* contracts           current contract row (resolution time, initial prob, answers)
* contract_edits      pre-image snapshots (``data`` is the contract *before* the edit)
* bets                every bet (price, user, amounts, fills) on those contracts
* comments            every comment, plus comment_edits (also pre-images)
* answers             multiple-choice answer rows

Bets pages are global newest-first; pages whose newest bet predates the
earliest parent contract are skipped.  Rows are deduplicated by primary key.
Workers parse pages in parallel; only the main process writes SQLite.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path

PAGES = Path("/home/agent/data/manifold-history/pages")
PARENTS: frozenset[str] = frozenset()
MIN_CREATED = 0

SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (id TEXT PRIMARY KEY, created_time INTEGER, resolution_time INTEGER,
  close_time INTEGER, resolution TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS contract_edits (id INTEGER PRIMARY KEY, contract_id TEXT, created_time INTEGER,
  updated_keys TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS bets (id TEXT PRIMARY KEY, contract_id TEXT, answer_id TEXT, created_time INTEGER,
  prob_before REAL, prob_after REAL, amount REAL, shares REAL, user_id TEXT, outcome TEXT, is_redemption INTEGER,
  is_cancelled INTEGER, limit_prob REAL, first_fill_time INTEGER, n_fills INTEGER);
CREATE TABLE IF NOT EXISTS comments (id TEXT PRIMARY KEY, contract_id TEXT, created_time INTEGER, user_id TEXT,
  data TEXT);
CREATE TABLE IF NOT EXISTS comment_edits (id INTEGER PRIMARY KEY, contract_id TEXT, comment_id TEXT,
  created_time INTEGER, data TEXT);
CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, contract_id TEXT, created_time INTEGER,
  resolution_time INTEGER, text TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS bets_c ON bets(contract_id, answer_id, created_time);
CREATE INDEX IF NOT EXISTS comments_c ON comments(contract_id, created_time);
CREATE INDEX IF NOT EXISTS cedits_c ON comment_edits(comment_id, created_time);
CREATE INDEX IF NOT EXISTS edits_c ON contract_edits(contract_id, created_time);
"""


def iso_ms(text: str | None) -> int | None:
    if not text:
        return None
    import datetime as dt
    return int(dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


def _init(parents, min_created):
    global PARENTS, MIN_CREATED
    PARENTS, MIN_CREATED = frozenset(parents), min_created


def _load(path: str) -> list[dict]:
    with gzip.open(path) as handle:
        return json.load(handle)


def bets_page(path: str):
    rows = _load(path)
    if not rows:
        return path, [], None
    newest = max(r["createdTime"] for r in rows)
    if newest < MIN_CREATED:
        return path, [], newest
    out = []
    for b in rows:
        if b.get("contractId") not in PARENTS:
            continue
        fills = b.get("fills") or []
        fill_times = [f.get("timestamp") for f in fills if f.get("timestamp") is not None]
        out.append((b["id"], b["contractId"], b.get("answerId"), b["createdTime"], b.get("probBefore"),
                    b.get("probAfter"), b.get("amount"), b.get("shares"), b.get("userId"), b.get("outcome"),
                    int(bool(b.get("isRedemption"))), int(bool(b.get("isCancelled"))), b.get("limitProb"),
                    min(fill_times) if fill_times else None, len(fills)))
    return path, out, newest


def comments_page(path: str):
    out = []
    for r in _load(path):
        if r.get("contract_id") in PARENTS:
            d = r["data"]
            out.append((r["comment_id"], r["contract_id"], d.get("createdTime") or iso_ms(r.get("created_time")),
                        d.get("userId"), json.dumps(d)))
    return path, out, None


def comment_edits_page(path: str):
    out = []
    for r in _load(path):
        if r.get("contract_id") in PARENTS:
            out.append((r["id"], r["contract_id"], r["comment_id"], iso_ms(r["created_time"]), json.dumps(r["data"])))
    return path, out, None


def answers_page(path: str):
    out = []
    for r in _load(path):
        if r.get("contract_id") in PARENTS:
            out.append((r["id"], r["contract_id"], iso_ms(r.get("created_time")), iso_ms(r.get("resolution_time")),
                        r.get("text"), json.dumps(r)))
    return path, out, None


def contracts_page(path: str):
    out = []
    for r in _load(path):
        if r["id"] in PARENTS:
            d = r["data"]
            out.append((r["id"], d.get("createdTime"), d.get("resolutionTime") or iso_ms(r.get("resolution_time")),
                        d.get("closeTime"), d.get("resolution") or r.get("resolution"), json.dumps(d)))
    return path, out, None


def parents_from_dataset(root: Path) -> set[str]:
    parents = set()
    for split in ("train", "validation", "test"):
        for line in (root / split / "propositions.jsonl").read_text().splitlines():
            if line.strip():
                parents.add(json.loads(line)["parent_id"])
    return parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/agent/data/jev-disjoint-v1-filtered-20260919")
    ap.add_argument("--edits", default="/home/agent/data/manifold-history/selected-edits.jsonl")
    ap.add_argument("--out", default="/home/agent/data/manifold-context-20260923/index.sqlite")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    parents = parents_from_dataset(Path(a.dataset))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    started = time.time()

    _init(parents, 0)
    for path in sorted(glob.glob(str(PAGES / "contracts" / "*.json.gz"))):
        db.executemany("INSERT OR IGNORE INTO contracts VALUES (?,?,?,?,?,?)", contracts_page(path)[1])
    n_contracts = db.execute("SELECT count(*) FROM contracts").fetchone()[0]
    min_created = db.execute("SELECT min(created_time) FROM contracts").fetchone()[0]
    print(f"contracts {n_contracts}/{len(parents)} min_created {min_created}", flush=True)

    n_edits = 0
    with open(a.edits) as handle:
        for line in handle:
            e = json.loads(line)
            if e["contract_id"] in parents:
                n_edits += 1
                db.execute("INSERT OR IGNORE INTO contract_edits VALUES (?,?,?,?,?)",
                           (e["id"], e["contract_id"], iso_ms(e["created_time"]), json.dumps(e.get("updated_keys")),
                            json.dumps(e["data"])))
    db.commit()
    print(f"contract_edits {n_edits}", flush=True)

    jobs = [("comments", comments_page, "INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?)"),
            ("comment_edits", comment_edits_page, "INSERT OR IGNORE INTO comment_edits VALUES (?,?,?,?,?)"),
            ("answers", answers_page, "INSERT OR IGNORE INTO answers VALUES (?,?,?,?,?,?)"),
            ("bets", bets_page, "INSERT OR IGNORE INTO bets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")]
    dirs = {"comments": "contract_comments", "comment_edits": "contract_comment_edits", "answers": "answers",
            "bets": "api_bets"}
    stats = {}
    with Pool(a.workers, initializer=_init, initargs=(parents, min_created)) as pool:
        for name, fn, sql in jobs:
            paths = sorted(glob.glob(str(PAGES / dirs[name] / "*.json.gz")))
            rows = pages = skipped = 0
            for path, out_rows, newest in pool.imap_unordered(fn, paths, chunksize=8):
                pages += 1
                if name == "bets" and newest is not None and newest < min_created:
                    skipped += 1
                if out_rows:
                    db.executemany(sql, out_rows)
                    rows += len(out_rows)
                if pages % 2000 == 0:
                    db.commit()
                    print(f"{name} {pages}/{len(paths)} pages, {rows} rows, {time.time() - started:.0f}s", flush=True)
            db.commit()
            stored = db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            stats[name] = {"pages": pages, "rows_seen": rows, "rows_stored": stored, "pages_before_min_created": skipped}
            print(f"{name} done {stats[name]}", flush=True)
    stats["contracts"] = {"rows_stored": n_contracts, "parents": len(parents)}
    stats["contract_edits"] = {"rows_stored": n_edits}
    db.execute("INSERT OR REPLACE INTO meta VALUES ('stats', ?)", (json.dumps(stats),))
    db.execute("INSERT OR REPLACE INTO meta VALUES ('source', ?)", (json.dumps(
        {"pages": str(PAGES), "edits": a.edits, "dataset": a.dataset, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),))
    db.commit()
    db.execute("ANALYZE")
    db.close()
    os.replace(tmp, out)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
