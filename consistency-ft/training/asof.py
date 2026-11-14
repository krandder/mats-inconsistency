"""Manifold as of t: read-only queries over the as-of store, every one cut off at t in SQL.

Store: /home/agent/data/manifold-asof-20260923/asof.sqlite (scripts/build_asof_store.py).

Every public function takes ``t`` (epoch ms, ISO date/datetime string, or datetime)
and returns only what existed strictly before t: markets created before t, text
versions in force just before t, bets/comments created before t, resolutions made
before t (and not undone before t), limit-order fills before t.  The cutoff is a
bound SQL parameter in each query, never a caller-side filter.

Known limits (reported by the leak audit, not hidden):
* BM25 term statistics in search come from the whole corpus (ranking weights only;
  no post-t document can be returned).
* Text versions before 2023-05-04 (edit recording start) may miss earlier edits
  (``exact = False``).
* Group membership has no timestamps; only groups created before t are used.

Main entry points:
  AsOf.search_markets / get_market / price_history / trades / comments / related_markets
  build_pack(pair, t, size)  deterministic per-pair evidence pack ("small" ~2k, "large" ~8k tokens)
  TOOL_SCHEMAS + AsOf.call_tool   OpenAI-compatible function calling (and the CLI in __main__)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_STORE = Path(os.environ.get("ASOF_STORE", "/home/agent/data/manifold-asof-20260923/asof.sqlite"))
TOKENIZER_JSON = Path("/home/agent/data/models/Qwen/Qwen3-0.6B/6130ef31402718485ca4d80a6234f70d9a4cf362/tokenizer.json")
TOKENIZER_PYLIB = "/home/agent/data/manifold-context-20260923/pylib"
FAR = 1 << 62
DAY = 86_400_000

STOPWORDS = frozenset("""a an and are as at be been before by can could did do does for from has have how i if in
into is it its of on or s than that the their them then there these they this to was were what when where which who
whom why will with would you your yes no market markets manifold resolve resolves resolved resolution question
""".split())


# --- time ---------------------------------------------------------------------

def to_ms(t) -> int:
    """Epoch ms from int/float ms, ISO string (date or datetime; naive = UTC) or datetime."""
    if isinstance(t, bool):
        raise TypeError("t must be a time")
    if isinstance(t, (int, float)):
        return int(t)
    if isinstance(t, str) and t.strip().isdigit():
        return int(t.strip())
    if isinstance(t, str):
        t = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
    if isinstance(t, dt.date) and not isinstance(t, dt.datetime):
        t = dt.datetime(t.year, t.month, t.day)
    if isinstance(t, dt.datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return int(t.timestamp() * 1000)
    raise TypeError(f"unsupported time: {t!r}")


def iso(ms: int | None, minutes: bool = False) -> str | None:
    if ms is None or not -62135596800000 < ms < 253402300800000:  # outside years 1..9999
        return None
    d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)
    return d.strftime("%Y-%m-%d %H:%M UTC" if minutes else "%Y-%m-%d")


def split_prop(prop_id: str) -> tuple[str, str | None]:
    """'contract' or 'contract:answer' (JEV proposition ids)."""
    cid, _, aid = prop_id.partition(":")
    return cid, (aid or None)


# --- tokens -------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tokenizer():
    try:
        if TOKENIZER_PYLIB not in sys.path:
            sys.path.append(TOKENIZER_PYLIB)
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(TOKENIZER_JSON))
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Qwen3 tokenizer count when available, else a chars/3.5 estimate (deterministic either way)."""
    tok = _tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False).ids)
    return math.ceil(len(text) / 3.5)


def fts_query(text: str, max_terms: int = 24) -> str | None:
    words, seen = [], set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'.$%-]*", text.lower()):
        w = w.strip("'.-%$")
        if len(w) < 2 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        words.append('"' + w.replace('"', "") + '"')
        if len(words) >= max_terms:
            break
    return " OR ".join(words) if words else None


def pct(p: float | None) -> str:
    if p is None:
        return "n/a"
    v = 100 * p
    return f"{v:.1f}%" if v < 1 or v > 99 else f"{v:.0f}%"


def clip(text: str | None, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


# --- store --------------------------------------------------------------------

# Parameter convention in the SQL below: :t is the cutoff; "before t" is `< :t`;
# a version [valid_from, valid_to) is in force just before t when valid_from < :t <= valid_to.

_MARKET_AT = """
SELECT m.mid, m.id, m.slug, m.creator_username, m.created, m.outcome_type, m.mechanism, m.initial_prob,
       m.sum_to_one, m.visibility, v.question, v.description, v.group_slugs, v.exact,
       CASE WHEN v.close IS NOT NULL AND v.close >= :t AND EXISTS (
              SELECT 1 FROM resolutions r WHERE r.mid = m.mid AND r.resolved >= :t
                AND abs(r.resolved - v.close) < 2000) THEN NULL
            ELSE v.close END AS close,
       (SELECT r.resolution FROM resolutions r WHERE r.mid = m.mid AND r.resolved < :t AND r.undone >= :t
          ORDER BY r.resolved DESC LIMIT 1) AS resolution,
       (SELECT r.resolved FROM resolutions r WHERE r.mid = m.mid AND r.resolved < :t AND r.undone >= :t
          ORDER BY r.resolved DESC LIMIT 1) AS resolution_time,
       (SELECT r.resolution_probability FROM resolutions r WHERE r.mid = m.mid AND r.resolved < :t
          AND r.undone >= :t ORDER BY r.resolved DESC LIMIT 1) AS resolution_probability
FROM markets m JOIN market_versions v ON v.mid = m.mid AND v.valid_from < :t AND v.valid_to >= :t
WHERE m.id = :id AND m.created < :t
"""

_ANSWERS_AT = """
SELECT a.aid, a.id, v.text, a.created, a.idx, a.is_other,
       CASE WHEN a.resolution_time < :t THEN a.resolution END AS resolution,
       CASE WHEN a.resolution_time < :t THEN a.resolution_time END AS resolution_time,
       CASE WHEN a.resolution_time < :t THEN a.resolution_probability END AS resolution_probability,
       (SELECT b.prob_after FROM bets b WHERE b.mid = a.mid AND b.aid = a.aid AND b.time < :t
          ORDER BY b.time DESC LIMIT 1) AS prob,
       (SELECT count(*) FROM bets b WHERE b.mid = a.mid AND b.aid = a.aid AND b.time < :t) AS n_bets
FROM answers a JOIN answer_versions v ON v.aid = a.aid AND v.valid_from < :t AND v.valid_to >= :t
WHERE a.mid = :mid AND a.created < :t
ORDER BY coalesce(a.idx, 1e9), a.created, a.id
"""

_BINARY_PRICE = """
SELECT b.prob_after, b.time FROM bets b WHERE b.mid = :mid AND b.aid = 0 AND b.time < :t
ORDER BY b.time DESC LIMIT 1
"""

_FILLED_AT = """(b.amount0 + coalesce((SELECT sum(json_extract(f.value, '$[1]')) FROM json_each(b.later_fills) f
                 WHERE json_extract(f.value, '$[0]') < :t), 0))"""
_SHARES_AT = """(b.shares0 + coalesce((SELECT sum(json_extract(f.value, '$[2]')) FROM json_each(b.later_fills) f
                 WHERE json_extract(f.value, '$[0]') < :t), 0))"""

_TRADES = f"""
SELECT b.time, b.id, b.aid, b.outcome, b.prob_before, b.prob_after, {_FILLED_AT} AS amount, {_SHARES_AT} AS shares,
       b.limit_prob, b.order_amount, CASE WHEN b.cancel_time < :t THEN 1 ELSE 0 END AS cancelled,
       b.is_redemption, u.username, u.id AS user_id
FROM bets b LEFT JOIN users u ON u.uid = b.uid
WHERE b.mid = :mid AND b.time < :t AND (:aid IS NULL OR b.aid = :aid)
ORDER BY b.time DESC, b.id DESC LIMIT :limit
"""

_TRADE_STATS = f"""
SELECT count(*) AS n, count(DISTINCT uid) AS traders, coalesce(sum(abs(amt)), 0) AS volume,
       coalesce(sum(time >= :since), 0) AS n_recent,
       coalesce(sum(CASE WHEN time >= :since THEN abs(amt) ELSE 0 END), 0) AS volume_recent
FROM (SELECT b.uid, b.time, {_FILLED_AT} AS amt FROM bets b
      WHERE b.mid = :mid AND b.time < :t AND (:aid IS NULL OR b.aid = :aid) AND b.is_redemption = 0)
WHERE amt IS NOT NULL AND amt != 0
"""

_COMMENTS = """
SELECT c.id, c.time, c.username, c.reply_to, c.bet_amount, c.bet_outcome,
       coalesce((SELECT e.text FROM comment_edits e WHERE e.comment_id = c.id AND e.time >= :t
                 ORDER BY e.time, e.id LIMIT 1), c.text) AS text
FROM comments c
WHERE c.mid = :mid AND c.time < :t
  AND (c.hidden_time IS NULL OR c.hidden_time >= :t)
  AND c.deleted_time IS NULL
  AND (c.edited IS NULL OR c.edited < :t
       OR EXISTS (SELECT 1 FROM comment_edits e WHERE e.comment_id = c.id AND e.time >= :t))
ORDER BY c.time DESC, c.id DESC LIMIT :limit OFFSET :offset
"""

_COMMENT_COUNT = """
SELECT count(*) FROM comments c WHERE c.mid = :mid AND c.time < :t
  AND (c.hidden_time IS NULL OR c.hidden_time >= :t) AND c.deleted_time IS NULL
  AND (c.edited IS NULL OR c.edited < :t
       OR EXISTS (SELECT 1 FROM comment_edits e WHERE e.comment_id = c.id AND e.time >= :t))
"""

_SEARCH = """
SELECT m.id, min(s.score) AS score
FROM (SELECT rowid AS did, bm25(docs_fts) AS score FROM docs_fts WHERE docs_fts MATCH :q
      ORDER BY rank, rowid LIMIT :inner) s
JOIN docs d ON d.did = s.did
JOIN markets m ON m.mid = d.mid
WHERE d.created < :t AND m.created < :t AND d.valid_from < :t AND d.valid_to >= :t
  AND m.visibility = 'public' AND m.id NOT IN (SELECT value FROM json_each(:exclude))
GROUP BY m.id ORDER BY score, m.id LIMIT :k
"""


class AsOf:
    def __init__(self, path: str | Path = DEFAULT_STORE):
        self.path = Path(path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only = 1")
        self.db.execute("PRAGMA cache_size = -200000")

    # -- markets ---------------------------------------------------------------

    def _market_row(self, cid: str, t: int):
        return self.db.execute(_MARKET_AT, {"id": cid, "t": t}).fetchone()

    def get_market(self, market_id: str, t, answers: bool = True) -> dict | None:
        """Market state just before t, or None if it did not exist yet."""
        t = to_ms(t)
        cid, _ = split_prop(market_id)
        m = self._market_row(cid, t)
        if m is None:
            return None
        status = "resolved" if m["resolution"] is not None else (
            "closed" if m["close"] is not None and m["close"] < t else "open")
        out = {
            "id": m["id"], "question": m["question"], "description": m["description"] or "",
            "url": f"https://manifold.markets/{m['creator_username']}/{m['slug']}" if m["slug"] else None,
            "creator": m["creator_username"], "created": m["created"], "created_date": iso(m["created"]),
            "close": m["close"], "close_date": iso(m["close"]), "outcome_type": m["outcome_type"],
            "status": status, "resolution": m["resolution"], "resolution_time": m["resolution_time"],
            "resolution_probability": m["resolution_probability"],
            "groups": self._groups_at(m["group_slugs"], t), "text_exact": bool(m["exact"]),
            "as_of": t,
        }
        if m["outcome_type"] in ("BINARY", "PSEUDO_NUMERIC", "STONK"):
            row = self.db.execute(_BINARY_PRICE, {"mid": m["mid"], "t": t}).fetchone()
            out["prob"] = row["prob_after"] if row else m["initial_prob"]
            out["last_trade_time"] = row["time"] if row else None
        else:
            out["prob"] = None
        if answers:
            out["answers"] = [self._answer(a) for a in self.db.execute(_ANSWERS_AT, {"mid": m["mid"], "t": t})]
        if m["resolution"] is not None and m["outcome_type"] not in ("BINARY", "PSEUDO_NUMERIC", "STONK"):
            row = self.db.execute("SELECT text FROM answers WHERE id = ? AND mid = ?", (m["resolution"], m["mid"])).fetchone()
            if row is not None:  # multiple choice resolved to one option (resolved before t, so its text is known)
                out["resolution_text"] = row[0]
        stats = self.trade_stats(m["mid"], t)
        out.update(n_trades=stats["n"], n_traders=stats["traders"], volume=round(stats["volume"], 2))
        out["n_comments"] = self.db.execute(_COMMENT_COUNT, {"mid": m["mid"], "t": t}).fetchone()[0]
        return out

    @staticmethod
    def _answer(a) -> dict:
        return {"id": a["id"], "text": a["text"], "created": a["created"], "prob": a["prob"], "n_bets": a["n_bets"],
                "is_other": bool(a["is_other"]), "resolution": a["resolution"],
                "resolution_time": a["resolution_time"], "resolution_probability": a["resolution_probability"]}

    def _groups_at(self, slugs_json: str | None, t: int) -> list[str]:
        slugs = json.loads(slugs_json) if slugs_json else []
        if not slugs:
            return []
        rows = self.db.execute("SELECT slug FROM groups WHERE slug IN (SELECT value FROM json_each(?)) "
                               "AND (created IS NULL OR created < ?) ORDER BY slug", (json.dumps(slugs), t))
        return [r[0] for r in rows]

    def _mid(self, market_id: str, t: int) -> int | None:
        row = self.db.execute("SELECT mid FROM markets WHERE id = ? AND created < ?", (split_prop(market_id)[0], t)).fetchone()
        return row[0] if row else None

    def _aid(self, mid: int, answer_id: str | None, t: int) -> int | None:
        if not answer_id:
            return None
        row = self.db.execute("SELECT aid FROM answers WHERE id = ? AND mid = ? AND created < ?", (answer_id, mid, t)).fetchone()
        return row[0] if row else -1  # -1: answer did not exist yet -> matches nothing

    def trade_stats(self, mid: int, t: int, aid: int | None = None, since: int | None = None) -> dict:
        row = self.db.execute(_TRADE_STATS, {"mid": mid, "t": t, "aid": aid, "since": since if since is not None else t})
        r = row.fetchone()
        return {k: (r[k] or 0) for k in ("n", "traders", "volume", "n_recent", "volume_recent")}

    # -- time series -------------------------------------------------------------

    def price_history(self, market_id: str, t, points: int = 12, answer_id: str | None = None,
                      start=None) -> list[dict]:
        """``points`` evenly spaced prices from creation (or ``start``) to just before t.

        For multi-choice markets pass ``answer_id`` (or a 'contract:answer' id)."""
        t = to_ms(t)
        cid, aid_s = split_prop(market_id)
        answer_id = answer_id or aid_s
        m = self.db.execute("SELECT mid, created, initial_prob, outcome_type FROM markets WHERE id = ? AND created < ?",
                            (cid, t)).fetchone()
        if m is None:
            return []
        aid = self._aid(m["mid"], answer_id, t) if answer_id else 0
        if aid == -1:
            return []  # the option did not exist yet
        begin = m["created"]
        if answer_id:
            begin = max(begin, self.db.execute("SELECT created FROM answers WHERE aid = ?", (aid,)).fetchone()[0])
        if start is not None:
            begin = max(begin, to_ms(start))
        points = max(2, int(points))
        end = t - 1
        if end <= begin:
            stamps = [end]
        else:
            stamps = [begin + (end - begin) * k // (points - 1) for k in range(points)]
        initial = m["initial_prob"] if not answer_id else None
        out = []
        for s in stamps:
            # s < t always (s <= t - 1), and the SQL bound is min(s + 1, t)
            row = self.db.execute("SELECT prob_after FROM bets WHERE mid = ? AND aid = ? AND time < min(?, ?) "
                                  "ORDER BY time DESC LIMIT 1", (m["mid"], aid, s + 1, t)).fetchone()
            out.append({"time": s, "date": iso(s), "prob": row[0] if row else initial})
        return out

    def trades(self, market_id: str, t, limit: int = 50, answer_id: str | None = None) -> list[dict]:
        """Newest-first bets created before t; amounts are what had filled before t."""
        t = to_ms(t)
        cid, aid_s = split_prop(market_id)
        mid = self._mid(cid, t)
        if mid is None:
            return []
        aid = self._aid(mid, answer_id or aid_s, t)
        answer_ids = dict(self.db.execute("SELECT aid, id FROM answers WHERE mid = ?", (mid,)))
        rows = self.db.execute(_TRADES, {"mid": mid, "t": t, "aid": aid, "limit": int(limit)})
        return [{"time": r["time"], "id": r["id"], "answer_id": answer_ids.get(r["aid"]), "outcome": r["outcome"],
                 "prob_before": r["prob_before"], "prob_after": r["prob_after"],
                 "amount": None if r["amount"] is None else round(r["amount"], 4),
                 "shares": None if r["shares"] is None else round(r["shares"], 4),
                 "limit_prob": r["limit_prob"], "order_amount": r["order_amount"], "cancelled": bool(r["cancelled"]),
                 "redemption": bool(r["is_redemption"]), "user": r["username"] or r["user_id"]} for r in rows]

    def comments(self, market_id: str, t, limit: int = 20, offset: int = 0) -> list[dict]:
        """Newest-first comments made before t, text as it stood just before t."""
        t = to_ms(t)
        mid = self._mid(market_id, t)
        if mid is None:
            return []
        rows = self.db.execute(_COMMENTS, {"mid": mid, "t": t, "limit": int(limit), "offset": int(offset)})
        return [{"id": r["id"], "time": r["time"], "date": iso(r["time"]), "user": r["username"],
                 "reply_to": r["reply_to"], "text": r["text"] or "",
                 "bet": None if r["bet_amount"] is None else {"amount": r["bet_amount"], "outcome": r["bet_outcome"]}}
                for r in rows]

    # -- search ------------------------------------------------------------------

    def search_markets(self, query: str, t, k: int = 10, exclude: list[str] | tuple = ()) -> list[dict]:
        """Public markets created before t whose text (as of t) matches ``query``; BM25 order."""
        t = to_ms(t)
        q = fts_query(query)
        if not q:
            return []
        rows, inner = [], 400
        while True:
            rows = self.db.execute(_SEARCH, {"q": q, "t": t, "inner": inner, "k": int(k),
                                             "exclude": json.dumps([split_prop(e)[0] for e in exclude])}).fetchall()
            if len(rows) >= k or inner >= 200_000:
                break
            total = self.db.execute("SELECT count(*) FROM (SELECT 1 FROM docs_fts WHERE docs_fts MATCH ? LIMIT ?)",
                                    (q, inner + 1)).fetchone()[0]
            if total <= inner:
                break
            inner *= 8
        out = []
        for r in rows:
            m = self._market_row(r["id"], t)
            if m is None:
                continue
            out.append({"id": r["id"], "question": m["question"], "score": round(-r["score"], 4),
                        "created_date": iso(m["created"])})
        return out

    def related_markets(self, market_id: str, t, k: int = 10, exclude: list[str] | tuple = (),
                        query: str | None = None) -> list[dict]:
        """Markets related to ``market_id`` at t: BM25 over the question (as of t), boosted by shared groups."""
        t = to_ms(t)
        cid, _ = split_prop(market_id)
        m = self._market_row(cid, t)
        if m is None and query is None:
            return []
        text = query if query is not None else m["question"]
        own_groups = set(self._groups_at(m["group_slugs"], t)) if m is not None else set()
        own_groups = {g for g in own_groups if not g.endswith("-default")}
        cands = self.search_markets(text, t, k=max(40, 4 * k), exclude=[cid, *exclude])
        scored = []
        for c in cands:
            row = self._market_row(c["id"], t)
            groups = {g for g in self._groups_at(row["group_slugs"], t) if not g.endswith("-default")}
            shared = sorted(own_groups & groups)
            scored.append((-(c["score"] * (1 + 0.25 * min(len(shared), 2))), c["id"], c, shared))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [{**c, "score": round(-s, 4), "shared_groups": shared} for s, _, c, shared in scored[:k]]

    # -- tools -------------------------------------------------------------------

    def call_tool(self, name: str, args: dict, t) -> Any:
        """Dispatch an OpenAI-style tool call; ``t`` is fixed by the caller, never by the model."""
        t = to_ms(t)
        args = dict(args or {})
        args.pop("t", None)
        if name == "search_markets":
            return self.search_markets(str(args.get("query", "")), t, k=min(int(args.get("k", 10)), 25))
        if name == "get_market":
            return self.get_market(str(args["id"]), t)
        if name == "price_history":
            return self.price_history(str(args["id"]), t, points=min(int(args.get("points", 12)), 60))
        if name == "trades":
            return self.trades(str(args["id"]), t, limit=min(int(args.get("limit", 20)), 200))
        if name == "comments":
            return self.comments(str(args["id"]), t, limit=min(int(args.get("limit", 20)), 100),
                                 offset=int(args.get("offset", 0)))
        if name == "related_markets":
            return self.related_markets(str(args["id"]), t, k=min(int(args.get("k", 10)), 25))
        raise ValueError(f"unknown tool {name}")


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


_ID = {"type": "string", "description": "Manifold market id, or 'market:answer' for one option"}
TOOL_SCHEMAS = [
    _fn("search_markets", "Full-text search over Manifold markets that existed at the forecast time.",
        {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 25}}, ["query"]),
    _fn("get_market", "Market as it stood at the forecast time: question, description, status, price, options.",
        {"id": _ID}, ["id"]),
    _fn("price_history", "Evenly spaced prices from market creation up to the forecast time.",
        {"id": _ID, "points": {"type": "integer", "minimum": 2, "maximum": 60}}, ["id"]),
    _fn("trades", "Most recent trades before the forecast time.",
        {"id": _ID, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, ["id"]),
    _fn("comments", "Most recent comments before the forecast time (newest first).",
        {"id": _ID, "limit": {"type": "integer", "minimum": 1, "maximum": 100},
         "offset": {"type": "integer", "minimum": 0}}, ["id"]),
    _fn("related_markets", "Markets related to a market (text similarity and shared topics), as of the forecast time.",
        {"id": _ID, "k": {"type": "integer", "minimum": 1, "maximum": 25}}, ["id"]),
]


# --- evidence pack ----------------------------------------------------------------

SIZES = {
    # tokens, description chars, history points, own comments, comment chars, related markets, related comments
    "small": dict(budget=2000, desc=1000, points=6, comments=5, comment_chars=240, related=8, rel_comments=0,
                  rel_desc=0, options=6),
    "large": dict(budget=8000, desc=4000, points=12, comments=15, comment_chars=500, related=25, rel_comments=3,
                  rel_desc=400, options=10),
}


def pair_as_of(pair: dict, props: dict[str, dict]) -> str:
    """JEV rule: t = 00:00 UTC of (min(closing_week of P, Q) - 7 days)."""
    week = min(props[pair["p_id"]]["closing_week"], props[pair["q_id"]]["closing_week"])
    return (dt.date.fromisoformat(week) - dt.timedelta(days=7)).isoformat()


def _prob_line(store: AsOf, market: dict, answer_id: str | None, t: int, n_options: int) -> list[str]:
    lines = []
    if answer_id:
        ans = [a for a in market["answers"] if a["id"] == answer_id]
        if not ans:
            return ["This option did not exist yet at t."]
        a = ans[0]
        state = f"resolved {a['resolution']} on {iso(a['resolution_time'])}" if a["resolution"] else "open"
        lines.append(f"Option \"{clip(a['text'], 160)}\": price {pct(a['prob'])} ({a['n_bets']} trades on it; {state}).")
        others = sorted((x for x in market["answers"] if x["id"] != answer_id),
                        key=lambda x: (-(x["prob"] or 0), x["id"]))[:n_options]
        if others:
            lines.append("Other options: " + "; ".join(f"\"{clip(x['text'], 60)}\" {pct(x['prob'])}" +
                                                      (f" (resolved {x['resolution']})" if x["resolution"] else "")
                                                      for x in others) + ".")
    elif market["answers"]:
        top = sorted(market["answers"], key=lambda x: (-(x["prob"] or 0), x["id"]))[:n_options]
        lines.append("Options: " + "; ".join(f"\"{clip(x['text'], 60)}\" {pct(x['prob'])}" for x in top) + ".")
    else:
        lines.append(f"Price (probability of YES): {pct(market['prob'])}.")
    return lines


def history_text(hist: list[dict]) -> str:
    """'date p' points, dropping a point identical to the previous one."""
    out = []
    for h in hist:
        item = f"{h['date']} {pct(h['prob'])}"
        if not out or out[-1] != item:
            out.append(item)
    return ", ".join(out)


def _status(market: dict) -> str:
    if market["status"] == "resolved":
        rp = (f" at {pct(market['resolution_probability'])}" if market["resolution"] == "MKT"
              and market.get("outcome_type") in ("BINARY", "PSEUDO_NUMERIC", "STONK") else "")
        what = f"\"{clip(market['resolution_text'], 60)}\"" if market.get("resolution_text") else market["resolution"]
        return f"resolved {what}{rp} on {iso(market['resolution_time'])}"
    if market["status"] == "closed":
        return f"closed to trading on {market['close_date']}, not yet resolved"
    return f"open; closes {market['close_date']}" if market["close_date"] else "open"


def own_market_block(store: AsOf, label: str, prop_id: str, t: int, cfg: dict) -> tuple[list[str], dict]:
    cid, answer_id = split_prop(prop_id)
    m = store.get_market(cid, t)
    if m is None:
        return [f"## {label}: market {cid} did not exist yet at t."], {"markets": [], "comments": []}
    lines = [f"## {label}: {m['question']}", f"Market id {prop_id}; created {m['created_date']}; {_status(m)}."]
    lines += _prob_line(store, m, answer_id, t, cfg["options"])
    hist = store.price_history(prop_id, t, points=cfg["points"])
    if hist:
        lines.append("Price history: " + history_text(hist) + ".")
    mid = store._mid(cid, t)
    aid = store._aid(mid, answer_id, t) if answer_id else None
    s = store.trade_stats(mid, t, aid=aid, since=t - 7 * DAY)
    lines.append(f"Trading before t: {s['n']} trades by {s['traders']} traders, M{s['volume']:.0f} volume (play money); "
                 f"last 7 days {s['n_recent']} trades, M{s['volume_recent']:.0f}.")
    if m["description"]:
        lines.append("Description (as of t): " + clip(m["description"], cfg["desc"]))
    cs = store.comments(cid, t, limit=cfg["comments"])
    if cs:
        lines.append(f"Recent comments (newest first, {len(cs)} of {m['n_comments']}):")
        lines += [f"- [{c['date']}] {c['user']}: {clip(c['text'], cfg['comment_chars'])}" for c in cs]
    return lines, {"markets": [cid], "comments": [c["id"] for c in cs]}


def related_block(store: AsOf, m: dict, t: int, cfg: dict) -> tuple[list[str], list[str]]:
    lines = [f"- {m['question']} [id {m['id']}; {_status(m)}]"]
    if m["answers"]:
        top = sorted(m["answers"], key=lambda x: (-(x["prob"] or 0), x["id"]))[:4]
        lines[0] += " Options: " + "; ".join(f"\"{clip(x['text'], 50)}\" {pct(x['prob'])}" for x in top) + "."
    elif m["status"] != "resolved":
        lines[0] += f" Price {pct(m['prob'])}; {m['n_trades']} trades."
    if cfg["rel_desc"] and m["description"]:
        lines.append("  Description: " + clip(m["description"], cfg["rel_desc"]))
    ids = []
    if cfg["rel_comments"]:
        for c in store.comments(m["id"], t, limit=cfg["rel_comments"]):
            lines.append(f"  [{c['date']}] {c['user']}: {clip(c['text'], 180)}")
            ids.append(c["id"])
    return lines, ids


def build_pack(pair: dict, t, size: str = "small", store: AsOf | None = None, budget_tokens: int | None = None,
               return_items: bool = False):
    """Deterministic per-pair evidence pack, retrieval only.

    ``pair`` needs ``p_id`` and ``q_id`` (proposition ids, 'contract' or 'contract:answer').
    Related markets exclude both pair markets.  Output is byte-identical for the same
    (pair, t, size) and store.  With ``return_items`` also returns the ids used.
    """
    store = store or default_store()
    t = to_ms(t)
    cfg = dict(SIZES[size])
    budget = budget_tokens or cfg["budget"]
    header = [f"# Manifold evidence as of {iso(t, minutes=True)}",
              "Manifold is a play-money prediction market; prices are crowd probabilities. Everything below existed "
              "before this time; nothing after it is included."]
    own_ids = [split_prop(pair["p_id"])[0], split_prop(pair["q_id"])[0]]
    items = {"markets": [], "comments": [], "related": []}

    def render(cfg_now, n_related):
        parts = list(header)
        its = {"markets": [], "comments": [], "related": []}
        for label, pid in (("Question A market", pair["p_id"]), ("Question B market", pair["q_id"])):
            lines, got = own_market_block(store, label, pid, t, cfg_now)
            parts += [""] + lines
            its["markets"] += got["markets"]
            its["comments"] += got["comments"]
        rel = related_for_pair(store, pair, t, n_related, own_ids)
        if rel:
            parts += ["", "## Related markets (as of t)"]
            for m in rel:
                lines, cids = related_block(store, m, t, cfg_now)
                parts += lines
                its["related"].append(m["id"])
                its["comments"] += cids
        return "\n".join(parts) + "\n", its

    n_rel = cfg["related"]
    text, items = render(cfg, n_rel)
    # shrink deterministically until within budget: related markets first, then comments, then description
    while count_tokens(text) > budget:
        if n_rel > 3:
            n_rel -= 1
        elif cfg["rel_comments"] > 0:
            cfg["rel_comments"] -= 1
        elif cfg["comments"] > 1:
            cfg["comments"] -= 1
        elif cfg["desc"] > 300:
            cfg["desc"] = int(cfg["desc"] * 0.7)
        elif n_rel > 0:
            n_rel -= 1
        elif cfg["comment_chars"] > 80:
            cfg["comment_chars"] = int(cfg["comment_chars"] * 0.7)
        else:
            break
        text, items = render(cfg, n_rel)
    return (text, items) if return_items else text


def related_for_pair(store: AsOf, pair: dict, t: int, k: int, own_ids: list[str]) -> list[dict]:
    """Interleave the related lists of both questions (A1, B1, A2, B2, ...), deduplicated."""
    if k <= 0:
        return []
    lists = []
    for pid in (pair["p_id"], pair["q_id"]):
        cid, aid = split_prop(pid)
        query = None
        if aid:
            m = store.get_market(cid, t)
            ans = [a for a in (m or {}).get("answers", []) if a["id"] == aid]
            if m is not None:
                query = m["question"] + " " + (ans[0]["text"] if ans else "")
        lists.append(store.related_markets(cid, t, k=k, exclude=own_ids, query=query))
    seen, out = set(own_ids), []
    for i in range(max(len(x) for x in lists) if lists else 0):
        for lst in lists:
            if i < len(lst) and lst[i]["id"] not in seen:
                seen.add(lst[i]["id"])
                m = store.get_market(lst[i]["id"], t)
                if m is not None:
                    out.append(m)
            if len(out) >= k:
                return out
    return out


def topk_context(pair: dict, t, k: int = 15, budget_tokens: int = 8000, store: AsOf | None = None) -> str:
    """Pack with up to ``k`` related markets within ``budget_tokens`` (small layout under 4k, else large)."""
    size = "small" if budget_tokens < 4000 else "large"
    store = store or default_store()
    saved = SIZES[size]["related"]
    try:
        SIZES[size]["related"] = k
        return build_pack(pair, t, size=size, store=store, budget_tokens=budget_tokens)
    finally:
        SIZES[size]["related"] = saved


@lru_cache(maxsize=1)
def default_store() -> AsOf:
    return AsOf(DEFAULT_STORE)


# --- CLI (used by the context agent as its only data access) ------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Manifold as of a fixed time t. t comes from --t or $ASOF_T; "
                                 "every result is limited to records before t.")
    ap.add_argument("--t", default=os.environ.get("ASOF_T"), help="cutoff (ISO date/time or epoch ms)")
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search_markets"); s.add_argument("query"); s.add_argument("--k", type=int, default=10)
    for name in ("get_market", "price_history", "trades", "comments", "related_markets"):
        p = sub.add_parser(name)
        p.add_argument("id")
        p.add_argument("--points", type=int, default=12)
        p.add_argument("--limit", type=int, default=20)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    if not a.t:
        ap.error("--t or ASOF_T is required")
    store = AsOf(a.store)
    args = {k: v for k, v in vars(a).items() if k not in ("t", "store", "cmd")}
    result = store.call_tool(a.cmd, args, a.t)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
