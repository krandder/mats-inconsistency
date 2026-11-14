#!/usr/bin/env python3
"""Build the "Manifold as of t" SQLite store (one streaming pass per table).

Source: the complete Manifold table pages in /home/agent/data/manifold-history/pages
(through 2026-09-18).  Output: /home/agent/data/manifold-asof-20260923/asof.sqlite,
read by ``training/asof.py``, which applies the time cutoff in SQL.

Time semantics (all times are epoch milliseconds):

* ``contract_edits`` and ``contract_comment_edits`` rows hold the record *before*
  the edit (pre-image; see sources/record-contract-edit.ts).  A market's text at t
  is therefore the pre-image of its earliest edit at or after t, else the current
  text.  Edits were recorded from 2023-05-04 on; versions that start before that
  are flagged ``exact = 0`` (unrecorded earlier edits are possible).
* Bets: a bet record at time x gives the price at x (``prob_after``).  Limit-order
  amounts in the dump are cumulative to today, so the store keeps the amount filled
  at creation plus the later fills with their own timestamps.
* Resolutions: the current resolution, plus earlier resolutions undone later
  (pre-image of an unresolve edit), each with its own time window.
* Close time: resolving early rewrites closeTime to the resolution time without an
  edit record, so a close time equal to a later resolution time is withheld.

Workers parse gzip pages in parallel; only the main process writes.  Memory is
bounded by a few id maps (markets, answers, users).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PAGES = Path("/home/agent/data/manifold-history/pages")
OUT = Path("/home/agent/data/manifold-asof-20260923/asof.sqlite")
FAR = 1 << 62  # "still valid" upper bound
EDIT_RECORDING_START = 1683237968635  # first contract_edits row, 2023-05-04T22:06:08Z

SCHEMA = """
PRAGMA page_size = 8192;
CREATE TABLE markets (mid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, slug TEXT, url TEXT,
  question TEXT, description TEXT, creator_id TEXT, creator_username TEXT, created INTEGER NOT NULL,
  close INTEGER, resolution TEXT, resolution_time INTEGER, resolution_probability REAL,
  outcome_type TEXT, mechanism TEXT, initial_prob REAL, sum_to_one INTEGER, visibility TEXT,
  deleted INTEGER, group_slugs TEXT);
CREATE TABLE answers (aid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, mid INTEGER NOT NULL,
  text TEXT, created INTEGER NOT NULL, idx INTEGER, is_other INTEGER, resolution TEXT,
  resolution_time INTEGER, resolution_probability REAL);
CREATE TABLE users (uid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, username TEXT, name TEXT);
CREATE TABLE groups (slug TEXT PRIMARY KEY, gid TEXT, name TEXT, created INTEGER, privacy TEXT);
CREATE TABLE market_groups (mid INTEGER NOT NULL, slug TEXT NOT NULL, PRIMARY KEY (mid, slug)) WITHOUT ROWID;
-- one row per recorded edit; text columns are pre-images (state just before ``time``)
CREATE TABLE edits (eid INTEGER PRIMARY KEY, mid INTEGER NOT NULL, time INTEGER NOT NULL, keys TEXT,
  question TEXT, description TEXT, close INTEGER, group_slugs TEXT, answers TEXT,
  is_resolved INTEGER, resolution TEXT, resolution_time INTEGER, resolution_probability REAL);
-- bet rows: (mid, aid, time, id) clustered; aid = 0 for binary markets
CREATE TABLE bets (mid INTEGER NOT NULL, aid INTEGER NOT NULL, time INTEGER NOT NULL, id TEXT NOT NULL,
  prob_before REAL, prob_after REAL, amount0 REAL, shares0 REAL, outcome TEXT, uid INTEGER,
  is_redemption INTEGER, limit_prob REAL, order_amount REAL, expires INTEGER, cancel_time INTEGER,
  later_fills TEXT, PRIMARY KEY (mid, aid, time, id)) WITHOUT ROWID;
CREATE TABLE comments (cid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, mid INTEGER NOT NULL,
  time INTEGER NOT NULL, uid INTEGER, username TEXT, reply_to TEXT, text TEXT, edited INTEGER,
  hidden_time INTEGER, deleted_time INTEGER, bet_amount REAL, bet_outcome TEXT);
CREATE TABLE comment_edits (id INTEGER PRIMARY KEY, comment_id TEXT NOT NULL, time INTEGER NOT NULL, text TEXT);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

DERIVED = """
CREATE TABLE market_versions (mid INTEGER NOT NULL, valid_from INTEGER NOT NULL, valid_to INTEGER NOT NULL,
  question TEXT, description TEXT, close INTEGER, group_slugs TEXT, exact INTEGER,
  PRIMARY KEY (mid, valid_from)) WITHOUT ROWID;
CREATE TABLE answer_versions (aid INTEGER NOT NULL, valid_from INTEGER NOT NULL, valid_to INTEGER NOT NULL,
  text TEXT, exact INTEGER, PRIMARY KEY (aid, valid_from)) WITHOUT ROWID;
CREATE TABLE resolutions (mid INTEGER NOT NULL, resolved INTEGER NOT NULL, undone INTEGER NOT NULL,
  resolution TEXT, resolution_probability REAL, PRIMARY KEY (mid, resolved)) WITHOUT ROWID;
-- search documents: kind 'm' = question + description version, 'a' = answer text version
CREATE TABLE docs (did INTEGER PRIMARY KEY, mid INTEGER NOT NULL, aid INTEGER, created INTEGER NOT NULL,
  valid_from INTEGER NOT NULL, valid_to INTEGER NOT NULL);
CREATE VIRTUAL TABLE docs_fts USING fts5(text, content='', tokenize='porter unicode61 remove_diacritics 2');
"""

INDEXES = """
CREATE INDEX answers_mid ON answers(mid, created);
CREATE INDEX edits_mid ON edits(mid, time);
CREATE INDEX bets_mid_time ON bets(mid, time);
CREATE INDEX comments_mid ON comments(mid, time);
CREATE INDEX comment_edits_c ON comment_edits(comment_id, time);
CREATE INDEX market_groups_slug ON market_groups(slug, mid);
CREATE INDEX markets_created ON markets(created);
"""


# --- parsing helpers (worker side) ------------------------------------------

def iso_ms(text):
    if not text:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    return int(dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


_TIPTAP = None


def text_of(node) -> str:
    global _TIPTAP
    if _TIPTAP is None:
        from training.market_context import tiptap_text
        _TIPTAP = tiptap_text
    try:
        return _TIPTAP(node)
    except Exception:  # malformed documents: keep the raw text leaves
        out = []

        def walk(n):
            if isinstance(n, dict):
                if n.get("type") == "text":
                    out.append(n.get("text", ""))
                for c in n.get("content") or []:
                    walk(c)
        walk(node)
        return " ".join(out)


def load(path):
    with gzip.open(path) as handle:
        return json.load(handle)


def num(x):
    return float(x) if isinstance(x, (int, float)) else None


def p_contracts(path):
    out = []
    for r in load(path):
        d = r.get("data") or {}
        created = d.get("createdTime") or iso_ms(r.get("created_time"))
        if created is None:
            continue
        slug = d.get("slug") or r.get("slug")
        user = d.get("creatorUsername")
        out.append(dict(
            id=r["id"], slug=slug, url=f"https://manifold.markets/{user}/{slug}" if user and slug else None,
            question=d.get("question") or r.get("question"), description=text_of(d.get("description")),
            creator_id=d.get("creatorId") or r.get("creator_id"), creator_username=user,
            creator_name=d.get("creatorName"), created=int(created),
            close=d.get("closeTime") or iso_ms(r.get("close_time")),
            resolution=d.get("resolution") or r.get("resolution"),
            resolution_time=d.get("resolutionTime") or iso_ms(r.get("resolution_time")),
            resolution_probability=num(d.get("resolutionProbability")),
            outcome_type=d.get("outcomeType") or r.get("outcome_type"),
            mechanism=d.get("mechanism") or r.get("mechanism"), initial_prob=num(d.get("initialProbability")),
            sum_to_one=int(bool(d.get("shouldAnswersSumToOne"))), visibility=d.get("visibility") or r.get("visibility"),
            deleted=int(bool(d.get("deleted") or r.get("deleted"))),
            group_slugs=d.get("groupSlugs") or r.get("group_slugs") or [],
            answers=[(a.get("id"), a.get("text"), a.get("createdTime"), a.get("index"), a.get("isOther"),
                      a.get("resolution"), a.get("resolutionTime"), a.get("resolutionProbability"))
                     for a in d.get("answers") or [] if isinstance(a, dict) and a.get("id")]))
    return out


def p_answers(path):
    return [(r["id"], r["contract_id"], r.get("text"), iso_ms(r.get("created_time")), r.get("index"),
             int(bool(r.get("is_other"))), r.get("resolution"), iso_ms(r.get("resolution_time")),
             num(r.get("resolution_probability"))) for r in load(path)]


def p_edits(path):
    out = []
    for r in load(path):
        d = r.get("data") or {}
        answers = d.get("answers")
        ans = None
        if isinstance(answers, list):
            ans = {a["id"]: a.get("text") for a in answers if isinstance(a, dict) and a.get("id")}
        out.append((r["id"], r["contract_id"], iso_ms(r["created_time"]), r.get("updated_keys"),
                    d.get("question"), text_of(d.get("description")) if "description" in d else None,
                    d.get("closeTime"), d.get("groupSlugs"), ans, d.get("isResolved"), d.get("resolution"),
                    d.get("resolutionTime"), num(d.get("resolutionProbability"))))
    return out


def p_comments(path):
    out = []
    for r in load(path):
        d = r.get("data") or {}
        created = d.get("createdTime") or iso_ms(r.get("created_time"))
        out.append((r["comment_id"], r["contract_id"], created, d.get("userId") or r.get("user_id"),
                    d.get("userUsername"), d.get("userName"), d.get("replyToCommentId"), text_of(d.get("content")),
                    d.get("editedTime"),
                    d.get("hiddenTime") or (created if d.get("hidden") and not d.get("hiddenTime") else None),
                    d.get("deletedTime") or (0 if d.get("deleted") and not d.get("deletedTime") else None),
                    num(d.get("betAmount")), d.get("betOutcome")))
    return out


def p_comment_edits(path):
    return [(r["id"], r["comment_id"], iso_ms(r["created_time"]), text_of((r.get("data") or {}).get("content")))
            for r in load(path)]


def p_bets(path):
    out = []
    for b in load(path):
        created = b.get("createdTime")
        if created is None or not b.get("contractId"):
            continue
        fills = b.get("fills") or []
        amount0 = shares0 = None
        later = []
        if b.get("limitProb") is not None and fills:
            amount0 = shares0 = 0.0
            for f in fills:
                ts = f.get("timestamp")
                if ts is None or ts <= created:
                    amount0 += f.get("amount") or 0
                    shares0 += f.get("shares") or 0
                else:
                    later.append([ts, round(f.get("amount") or 0, 6), round(f.get("shares") or 0, 6)])
        elif b.get("limitProb") is not None:
            # no fill list: only trust the amount when the order never changed after creation
            upd = b.get("updatedTime")
            if not b.get("amount") or upd is None or upd <= created:
                amount0, shares0 = b.get("amount") or 0.0, b.get("shares") or 0.0
        else:
            amount0, shares0 = b.get("amount"), b.get("shares")
        cancel = None
        if b.get("isCancelled"):
            cancel = b.get("updatedTime") or FAR  # unknown cancel time: never shown as cancelled
        out.append((b["contractId"], b.get("answerId"), created, b["id"], num(b.get("probBefore")),
                    num(b.get("probAfter")), amount0, shares0, b.get("outcome"), b.get("userId"),
                    int(bool(b.get("isRedemption"))), num(b.get("limitProb")), num(b.get("orderAmount")),
                    b.get("expiresAt"), cancel, json.dumps(later) if later else None))
    return out


def p_groups(path):
    return [(r["slug"], r["id"], r.get("name"), iso_ms(r.get("created_time")), r.get("privacy_status"))
            for r in load(path)]


def p_group_contracts(path):
    return [(r["group_id"], r["contract_id"]) for r in load(path)]


# --- writer ------------------------------------------------------------------

class Builder:
    def __init__(self, db: sqlite3.Connection, workers: int):
        self.db, self.workers = db, workers
        self.mids: dict[str, int] = {}
        self.aids: dict[str, int] = {}
        self.uids: dict[str, int] = {}
        self.started = time.time()
        self.stats: dict[str, dict] = {}

    def log(self, msg):
        print(f"[{time.time() - self.started:7.0f}s] {msg}", flush=True)

    def pages(self, name):
        return sorted(glob.glob(str(PAGES / name / "*.json.gz")))

    def stream(self, name, fn, ordered=False):
        paths = self.pages(name)
        # bounded in-flight work: at most ``window`` parsed pages wait for the single writer
        window = 8 * self.workers
        k = 0
        with Pool(self.workers) as pool:
            for lo in range(0, len(paths), window):
                batch = paths[lo:lo + window]
                it = pool.imap(fn, batch) if ordered else pool.imap_unordered(fn, batch)
                for rows in it:
                    k += 1
                    yield rows
                    if k % 2000 == 0:
                        self.db.commit()
                        self.log(f"{name} {k}/{len(paths)} pages")
        self.db.commit()

    def uid(self, user_id, username=None, name=None):
        if not user_id:
            return None
        u = self.uids.get(user_id)
        if u is None:
            u = self.uids[user_id] = len(self.uids) + 1
            self.db.execute("INSERT INTO users VALUES (?,?,?,?)", (u, user_id, username, name))
        elif username:
            self.db.execute("UPDATE users SET username = coalesce(username, ?), name = coalesce(name, ?) "
                            "WHERE uid = ?", (username, name, u))
        return u

    def aid(self, answer_id, mid, created=None):
        a = self.aids.get(answer_id)
        if a is None:
            a = self.aids[answer_id] = len(self.aids) + 1
        return a

    def contracts(self):
        n = dup = 0
        embedded = []
        for rows in self.stream("contracts", p_contracts, ordered=True):
            for r in rows:
                if r["id"] in self.mids:
                    dup += 1
                    continue
                mid = self.mids[r["id"]] = len(self.mids) + 1
                self.uid(r["creator_id"], r["creator_username"], r["creator_name"])
                self.db.execute("INSERT INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    mid, r["id"], r["slug"], r["url"], r["question"], r["description"], r["creator_id"],
                    r["creator_username"], r["created"], r["close"], r["resolution"], r["resolution_time"],
                    r["resolution_probability"], r["outcome_type"], r["mechanism"], r["initial_prob"],
                    r["sum_to_one"], r["visibility"], r["deleted"], json.dumps(r["group_slugs"])))
                self.db.executemany("INSERT OR IGNORE INTO market_groups VALUES (?,?)",
                                    [(mid, s) for s in r["group_slugs"] if isinstance(s, str)])
                for a in r["answers"]:
                    embedded.append((mid,) + a)
                n += 1
        self.stats["markets"] = {"rows": n, "duplicate_ids": dup}
        self.log(f"markets {n} (dup {dup}); embedded answers {len(embedded)}")
        return embedded

    def answers(self, embedded):
        n = orphan = 0
        for rows in self.stream("answers", p_answers):
            for (aid_s, cid, text, created, idx, is_other, res, res_t, res_p) in rows:
                mid = self.mids.get(cid)
                if mid is None or created is None or aid_s in self.aids:
                    orphan += 1
                    continue
                a = self.aid(aid_s, mid)
                self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)",
                                (a, aid_s, mid, text, created, idx, is_other, res, res_t, res_p))
                n += 1
        extra = 0
        for (mid, aid_s, text, created, idx, is_other, res, res_t, res_p) in embedded:
            if aid_s in self.aids or created is None:
                continue  # legacy DPM answers ('1', '2', ...) are not globally unique
            if not (isinstance(aid_s, str) and len(aid_s) >= 8):
                continue
            a = self.aid(aid_s, mid)
            self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (a, aid_s, mid, text, created, idx, int(bool(is_other)), res, res_t, num(res_p)))
            extra += 1
        self.stats["answers"] = {"rows": n + extra, "from_contract_rows": extra, "skipped": orphan}
        self.log(f"answers {n}+{extra}")

    def edits(self):
        n = orphan = 0
        last_desc: dict[int, bytes] = {}
        for rows in self.stream("contract_edits", p_edits, ordered=True):
            for (eid, cid, t, keys, q, desc, close, groups, ans, is_res, res, res_t, res_p) in rows:
                mid = self.mids.get(cid)
                if mid is None or t is None:
                    orphan += 1
                    continue
                # descriptions repeat across a market's edits: store NULL when equal to the previous pre-image
                if desc is not None:
                    h = hashlib.blake2b(desc.encode(), digest_size=8).digest()
                    stored = None if last_desc.get(mid) == h else desc
                    last_desc[mid] = h
                else:
                    stored = None
                self.db.execute("INSERT OR IGNORE INTO edits VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    eid, mid, t, json.dumps(keys) if keys is not None else None, q,
                    stored if stored is not None else (None if desc is None else "\x00same"),
                    close, json.dumps(groups) if groups is not None else None,
                    json.dumps(ans) if ans is not None else None,
                    None if is_res is None else int(bool(is_res)), res, res_t, res_p))
                n += 1
        self.stats["edits"] = {"rows": n, "orphan": orphan}
        self.log(f"edits {n} (orphan {orphan})")

    def comments(self):
        n = orphan = 0
        seen = set()
        for rows in self.stream("contract_comments", p_comments):
            for (cid_s, contract, created, user, username, name, reply, text, edited, hidden, deleted, bamt,
                 bout) in rows:
                mid = self.mids.get(contract)
                if mid is None or created is None or cid_s in seen:
                    orphan += 1
                    continue
                seen.add(cid_s)
                self.db.execute("INSERT INTO comments (id, mid, time, uid, username, reply_to, text, edited, "
                                "hidden_time, deleted_time, bet_amount, bet_outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                (cid_s, mid, created, self.uid(user, username, name), username, reply, text,
                                 edited, hidden, deleted, bamt, bout))
                n += 1
        m = 0
        for rows in self.stream("contract_comment_edits", p_comment_edits):
            self.db.executemany("INSERT OR IGNORE INTO comment_edits VALUES (?,?,?,?)", rows)
            m += len(rows)
        self.stats["comments"] = {"rows": n, "skipped": orphan, "comment_edits": m}
        self.log(f"comments {n}, comment edits {m}")

    def groups(self):
        for rows in self.stream("groups", p_groups):
            self.db.executemany("INSERT OR IGNORE INTO groups VALUES (?,?,?,?,?)", rows)
        slug_of = dict(self.db.execute("SELECT gid, slug FROM groups"))
        n = 0
        for rows in self.stream("group_contracts", p_group_contracts):
            batch = [(self.mids[c], slug_of[g]) for g, c in rows if c in self.mids and g in slug_of]
            self.db.executemany("INSERT OR IGNORE INTO market_groups VALUES (?,?)", batch)
            n += len(batch)
        self.log(f"groups; group_contracts {n}")

    def bets(self):
        n = orphan = 0
        sql = "INSERT OR IGNORE INTO bets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        for rows in self.stream("api_bets", p_bets):
            batch = []
            for r in rows:
                mid = self.mids.get(r[0])
                if mid is None:
                    orphan += 1
                    continue
                if not isinstance(r[2], int) or not 0 < r[2] < FAR:
                    orphan += 1
                    continue
                aid = self.aid(r[1], mid) if r[1] else 0
                row = (mid, aid) + r[2:9] + (self.uid(r[9]),) + r[10:]
                # a few legacy records carry integers beyond SQLite's 64-bit range
                batch.append(tuple(float(x) if isinstance(x, int) and not -FAR < x < FAR else x for x in row))
            self.db.executemany(sql, batch)
            n += len(batch)
        self.stats["bets"] = {"rows_seen": n, "orphan_contract": orphan}
        self.log(f"bets {n} (orphan {orphan})")

    # --- derived tables -----------------------------------------------------

    def derive(self):
        db = self.db
        db.executescript(DERIVED)
        cur = db.execute("SELECT mid, created, question, description, close, group_slugs, resolution, "
                         "resolution_time, resolution_probability FROM markets ORDER BY mid")
        ins_v = "INSERT OR REPLACE INTO market_versions VALUES (?,?,?,?,?,?,?,?)"
        nv = 0
        rd = db.cursor()
        for (mid, created, q, desc, close, groups, res, res_t, res_p) in cur.fetchall():
            edits = rd.execute("SELECT time, question, description, close, group_slugs, keys, is_resolved, "
                               "resolution, resolution_time, resolution_probability FROM edits "
                               "WHERE mid = ? ORDER BY time, eid", (mid,)).fetchall()
            versions = []
            start = created
            prev_desc = None
            for (t, eq, edesc, eclose, egroups, keys, is_res, eres, eres_t, eres_p) in edits:
                if edesc == "\x00same":
                    edesc = prev_desc
                prev_desc = edesc if edesc is not None else prev_desc
                if t > start:
                    versions.append([start, t, eq if eq is not None else q, edesc if edesc is not None else desc,
                                     eclose, egroups, int(start >= EDIT_RECORDING_START)])
                    start = t
                if is_res and eres and eres_t is not None and keys and "isResolved" in keys:
                    # the market was resolved before this edit and the edit undid it
                    db.execute("INSERT OR IGNORE INTO resolutions VALUES (?,?,?,?,?)", (mid, eres_t, t, eres, eres_p))
            versions.append([start, FAR, q, desc, close, groups, int(start >= EDIT_RECORDING_START)])
            # merge adjacent identical versions
            merged = []
            for v in versions:
                if merged and merged[-1][2:6] == v[2:6]:
                    merged[-1][1] = v[1]
                    merged[-1][6] = min(merged[-1][6], v[6])
                else:
                    merged.append(v)
            for v in merged:
                db.execute(ins_v, (mid, v[0], v[1], v[2], v[3], v[4], v[5], v[6]))
            nv += len(merged)
            if res and res_t is not None:
                db.execute("INSERT OR IGNORE INTO resolutions VALUES (?,?,?,?,?)", (mid, res_t, FAR, res, res_p))
        db.commit()
        self.log(f"market_versions {nv}")

        # answer text versions from 'answers' pre-images
        na = 0
        cur = db.execute("SELECT aid, id, mid, text, created FROM answers ORDER BY mid")
        last_mid, pre = None, []
        for (aid, aid_s, mid, text, created) in cur.fetchall():
            if mid != last_mid:
                pre = rd.execute("SELECT time, answers FROM edits WHERE mid = ? AND answers IS NOT NULL "
                                 "ORDER BY time, eid", (mid,)).fetchall()
                pre = [(t, json.loads(a)) for t, a in pre]
                last_mid = mid
            versions, start = [], created
            for t, amap in pre:
                if t <= start or aid_s not in amap:
                    continue
                versions.append([start, t, amap[aid_s] if amap[aid_s] is not None else text])
                start = t
            versions.append([start, FAR, text])
            merged = []
            for v in versions:
                if merged and merged[-1][2] == v[2]:
                    merged[-1][1] = v[1]
                else:
                    merged.append(v)
            for v in merged:
                db.execute("INSERT OR REPLACE INTO answer_versions VALUES (?,?,?,?,?)",
                           (aid, v[0], v[1], v[2], int(v[0] >= EDIT_RECORDING_START)))
            na += len(merged)
        db.commit()
        self.log(f"answer_versions {na}")

        # search documents
        db.execute("INSERT INTO docs (mid, aid, created, valid_from, valid_to) "
                   "SELECT v.mid, NULL, m.created, v.valid_from, v.valid_to FROM market_versions v "
                   "JOIN markets m USING (mid) WHERE m.visibility IS NOT 'private' ORDER BY v.mid, v.valid_from")
        db.execute("INSERT INTO docs (mid, aid, created, valid_from, valid_to) "
                   "SELECT a.mid, v.aid, m.created, v.valid_from, v.valid_to FROM answer_versions v "
                   "JOIN answers a USING (aid) JOIN markets m ON m.mid = a.mid "
                   "WHERE m.visibility IS NOT 'private' ORDER BY a.mid, v.valid_from")
        db.commit()
        cur = db.execute(
            "SELECT d.did, v.question, v.description FROM docs d JOIN market_versions v "
            "ON v.mid = d.mid AND v.valid_from = d.valid_from WHERE d.aid IS NULL "
            "UNION ALL SELECT d.did, v.text, NULL FROM docs d JOIN answer_versions v "
            "ON v.aid = d.aid AND v.valid_from = d.valid_from WHERE d.aid IS NOT NULL")
        nd = 0
        w = db.cursor()
        while True:
            chunk = cur.fetchmany(20000)
            if not chunk:
                break
            w.executemany("INSERT INTO docs_fts (rowid, text) VALUES (?, ?)",
                          [(did, (q or "") + "\n" + (d or "")[:20000]) for did, q, d in chunk])
            nd += len(chunk)
        db.execute("INSERT INTO docs_fts (docs_fts) VALUES ('optimize')")
        db.commit()
        self.log(f"docs {nd}")


def build(out: Path, workers: int = 12, only_derive: bool = False, pages: Path | None = None) -> dict:
    global PAGES
    if pages is not None:
        PAGES = Path(pages)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".building")
    os.environ.setdefault("SQLITE_TMPDIR", "/tmp")
    if only_derive:
        tmp = out
    elif tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("PRAGMA cache_size = -4000000")
    db.execute("PRAGMA temp_store = FILE")
    b = Builder(db, workers)
    if only_derive:
        for t in ("market_versions", "answer_versions", "resolutions", "docs", "docs_fts"):
            db.execute(f"DROP TABLE IF EXISTS {t}")
        b.derive()
    else:
        db.executescript(SCHEMA)
        embedded = b.contracts()
        b.answers(embedded)
        del embedded
        b.groups()
        b.edits()
        b.comments()
        b.bets()
        b.log("indexes")
        db.executescript(INDEXES)
        db.commit()
        b.derive()
    counts = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in (
        "markets", "answers", "bets", "comments", "comment_edits", "edits", "market_versions", "answer_versions",
        "resolutions", "docs", "users", "market_groups")}
    rng = db.execute("SELECT min(time), max(time) FROM bets").fetchone()
    info = {"counts": counts, "stats": b.stats, "bet_time_range": rng,
            "markets_created_range": db.execute("SELECT min(created), max(created) FROM markets").fetchone(),
            "source": str(PAGES), "edit_recording_start": EDIT_RECORDING_START,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    db.execute("INSERT OR REPLACE INTO meta VALUES ('build', ?)", (json.dumps(info),))
    db.commit()
    db.execute("ANALYZE")
    db.close()
    if tmp != out:
        os.replace(tmp, out)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pages", default=str(PAGES))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only-derive", action="store_true", help="rebuild derived tables in an existing store")
    a = ap.parse_args()
    info = build(Path(a.out), a.workers, a.only_derive, Path(a.pages))
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    main()
