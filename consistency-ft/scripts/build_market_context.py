#!/usr/bin/env python3
"""Build the market-context version of the JEV disjoint pair dataset.

For every pair the forecast time t is 00:00 UTC of the existing as-of date
(earlier closing week minus seven days), shared by both propositions.  Each
proposition gets a ``context`` object built only from records created strictly
before t (index from scripts/build_market_context_index.py):

* price at t, an 8-point price history, trades and traders before t;
* the question, option and criteria text as they stood at t, rebuilt from
  contract_edits pre-images (fallback: current text with creator edits
  stripped, flagged);
* the newest comments before t (at most 15, about 2,000 characters), each as
  it read at t (comment_edits pre-images).

Pairs are removed when either proposition's market or answer resolved at or
before t (or a resolution event is logged before t).  All removals are
counted.  Markets or options created at or after t are kept with
``open_at_t: false``: no price, no comments, and the text at creation.  A leakage
audit and price distribution go into the manifest; per-proposition audit
facts that involve post-t data (resolution time) go to context-audit.jsonl,
never into the context itself.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import hashlib
import json
import re
import shutil
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import market_context as mc  # noqa: E402
from training.supervised_pairs import strip_creator_edits  # noqa: E402

SPLITS = ("train", "validation", "test")
DAY = 86_400_000
RESOLUTION_WORDS = {
    "resolved": re.compile(r"\bresolved\b", re.I),
    "resolves_yes_no": re.compile(r"\bresolv(?:e|es|ed|ing)\s+(?:to\s+|as\s+)?(?:yes|no)\b", re.I),
    "n_a": re.compile(r"\bN/A\b"),
    "resolve_any": re.compile(r"\bresolv\w*", re.I),
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def t_ms_of(as_of: str) -> int:
    return int(dt.datetime.fromisoformat(as_of).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


class Index:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.contracts = {r["id"]: {**dict(r), "data": json.loads(r["data"])}
                          for r in self.db.execute("SELECT * FROM contracts")}
        self.edits: dict[str, list[dict]] = collections.defaultdict(list)
        for r in self.db.execute("SELECT * FROM contract_edits ORDER BY contract_id, created_time, id"):
            self.edits[r["contract_id"]].append({"id": r["id"], "created_time": r["created_time"],
                                                 "keys": json.loads(r["updated_keys"] or "[]"),
                                                 "data": json.loads(r["data"])})
        self.answers = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM answers")}
        self.chain_breaks = self._chain_breaks()

    def _chain_breaks(self) -> dict[str, int]:
        """Contracts whose description changed without an edit row (pre-images disagree)."""
        breaks = collections.Counter()
        for cid, edits in self.edits.items():
            snapshots = [e for e in edits]
            for a, b in zip(snapshots, snapshots[1:]):
                if "description" not in a["keys"] and a["data"].get("description") != b["data"].get("description"):
                    breaks[cid] += 1
            last = snapshots[-1]
            if "description" not in last["keys"] and last["data"].get("description") != \
                    self.contracts[cid]["data"].get("description"):
                breaks[cid] += 1
        return dict(breaks)

    def bets(self, cid: str, answer_id: str | None) -> list[dict]:
        if answer_id is None:
            rows = self.db.execute("SELECT * FROM bets WHERE contract_id=? AND answer_id IS NULL "
                                   "ORDER BY created_time, id", (cid,))
        else:
            rows = self.db.execute("SELECT * FROM bets WHERE contract_id=? AND answer_id=? ORDER BY created_time, id",
                                   (cid, answer_id))
        return [dict(r) for r in rows]

    def comments_before(self, cid: str, t_ms: int) -> list[dict]:
        out = []
        for r in self.db.execute("SELECT * FROM comments WHERE contract_id=? AND created_time<? ORDER BY created_time",
                                 (cid, t_ms)):
            data = json.loads(r["data"])
            if data.get("hidden") or data.get("deleted"):
                continue
            edit = self.db.execute("SELECT data, created_time FROM comment_edits WHERE comment_id=? AND created_time>=? "
                                   "ORDER BY created_time, id LIMIT 1", (r["id"], t_ms)).fetchone()
            content = json.loads(edit["data"]).get("content") if edit else data.get("content")
            out.append({"id": r["id"], "created_time": r["created_time"], "date": mc.iso_date(r["created_time"]),
                        "user": data.get("userUsername") or data.get("userName") or "user",
                        "text": mc.tiptap_text(content), "edited_after_t": bool(edit)})
        return out


def snapshot_at(index: Index, cid: str, t_ms: int) -> tuple[dict, dict]:
    """Contract data as it stood at t, and facts about the edit history.

    ``contract_edits.data`` is the pre-image, so the state at t is the
    pre-image of the first edit at or after t, or the current row when no
    edit happened after t.
    """
    edits = index.edits.get(cid, [])
    times = [e["created_time"] for e in edits]
    k = bisect.bisect_left(times, t_ms)
    data = edits[k]["data"] if k < len(edits) else index.contracts[cid]["data"]
    before, after = edits[:k], edits[k:]
    facts = {
        "description_edits_before_t": sum("description" in e["keys"] for e in before),
        "description_edits_after_t": sum("description" in e["keys"] for e in after),
        "question_edits_after_t": sum("question" in e["keys"] for e in after),
        "resolution_event_before_t": any({"isResolved", "resolution", "resolutions"} & set(e["keys"]) for e in before),
    }
    return data, facts


def sibling_prices(index: Index, cid: str, answer_id: str, t_ms: int, data_t: dict) -> list[dict]:
    """Prices at t of the other options of a multiple-choice market that existed before t.

    Same cutoff as the proposition: only bets created before t.  An option
    that resolved before t shows its resolution (public before t).  Text is
    the option text at t.  Sorted by price, highest first.
    """
    out = []
    texts_t = {a.get("id"): a.get("text") for a in data_t.get("answers") or []}
    current = {a.get("id"): a for a in index.contracts[cid]["data"].get("answers") or []}
    ids = (set(current) | {aid for aid, a in index.answers.items() if a["contract_id"] == cid}) - {answer_id}
    for aid in ids:
        row, cur = index.answers.get(aid) or {}, current.get(aid) or {}
        created = row.get("created_time") or cur.get("createdTime")
        if created is None or created >= t_ms:
            continue
        resolved = row.get("resolution_time") or cur.get("resolutionTime")
        text = texts_t.get(aid) or row.get("text") or cur.get("text") or ""
        item = {"answer_id": aid, "text": text,
                "p_t": round(mc.price_at(index.bets(cid, aid), t_ms, 0.5), 4), "resolved_before_t": None}
        if resolved is not None and resolved < t_ms:
            item["resolved_before_t"] = (row.get("resolution") or cur.get("resolution"))
        out.append(item)
    return sorted(out, key=lambda x: (-x["p_t"], x["text"]))


def not_open_context(index: Index, prop: dict, as_of: str, reason: str) -> dict:
    """Context for a market (or option) that did not exist yet at t: no prices, no comments.

    Nothing is known at t, so the text shown is the text at creation: the
    pre-image of the first contract edit, or the current text when the market
    was never edited.
    """
    cid = prop["parent_id"]
    answer_id = prop["id"].split(":", 1)[1] if ":" in prop["id"] else None
    edits = index.edits.get(cid, [])
    data0 = edits[0]["data"] if edits else index.contracts[cid]["data"]
    if cid in index.chain_breaks:
        criteria, source = strip_creator_edits(str(prop.get("criteria_text") or "")), "current_text_creator_edits_stripped"
    else:
        criteria, source = mc.tiptap_text(data0.get("description")), "text_at_creation"
    answer0 = next((a for a in data0.get("answers") or [] if a.get("id") == answer_id), None)
    return {"as_of": as_of, "t_ms": t_ms_of(as_of), "open_at_t": False, "not_open_reason": reason,
            "contract_id": cid, "answer_id": answer_id, "created_date": None,
            "question_at_t": data0.get("question") or prop.get("parent_question"),
            "answer_text_at_t": ((answer0 or {}).get("text") or prop.get("answer_text")) if answer_id else None,
            "criteria": {"text": criteria, "source": source, "reconstructed": source != "current_text_creator_edits_stripped",
                         "description_edits_before_t": 0,
                         "description_edits_after_t": sum("description" in e["keys"] for e in edits)},
            "price": None, "price_history": [], "sibling_prices": None, "n_comments_before_t": 0, "comments": []}


def build_context(index: Index, prop: dict, as_of: str) -> tuple[dict | None, str | None, dict]:
    """Return (context, removal reason, audit facts) for one proposition."""
    t_ms = t_ms_of(as_of)
    cid = prop["parent_id"]
    answer_id = prop["id"].split(":", 1)[1] if ":" in prop["id"] else None
    contract = index.contracts.get(cid)
    if contract is None:
        return None, "contract_missing", {}
    cdata = contract["data"]
    audit = {"id": prop["id"], "as_of": as_of, "resolution_time": contract["resolution_time"],
             "close_time": contract["close_time"]}
    if contract["resolution_time"] is not None and contract["resolution_time"] <= t_ms:
        return None, "market_resolved_at_or_before_t", audit
    if cdata["createdTime"] >= t_ms:
        return not_open_context(index, prop, as_of, "market_created_at_or_after_t"), None, audit
    data_t, facts = snapshot_at(index, cid, t_ms)
    if facts["resolution_event_before_t"]:
        return None, "resolution_event_before_t", audit
    start_ms, initial, answer_text_t = cdata["createdTime"], cdata.get("initialProbability"), None
    if answer_id:
        answer = index.answers.get(answer_id)
        current = next((a for a in cdata.get("answers") or [] if a.get("id") == answer_id), None)
        if answer is None and current is None:
            return None, "answer_missing", audit
        a_created = (answer or {}).get("created_time") or current.get("createdTime")
        a_resolved = (answer or {}).get("resolution_time") or (current or {}).get("resolutionTime")
        audit["resolution_time"] = a_resolved or contract["resolution_time"]
        if a_resolved is not None and a_resolved <= t_ms:
            return None, "answer_resolved_at_or_before_t", audit
        if a_created >= t_ms:
            return not_open_context(index, prop, as_of, "answer_created_at_or_after_t"), None, audit
        start_ms, initial = a_created, 0.5  # independent answers all open at 0.5 (checked on every first bet)
        answer_t = next((a for a in data_t.get("answers") or [] if a.get("id") == answer_id), None)
        answer_text_t = (answer_t or {}).get("text") or (answer or {}).get("text") or prop.get("answer_text")
    if initial is None:
        initial = 0.5
    bets = [b for b in index.bets(cid, answer_id) if b["created_time"] < t_ms]
    n_trades, n_traders = mc.trade_counts(bets, t_ms)
    p_t = mc.price_at(bets, t_ms, initial)
    price_source = "last_bet" if any(b.get("prob_after") is not None for b in bets) else "initial_probability"
    if cid in index.chain_breaks:
        criteria = strip_creator_edits(str(prop.get("criteria_text") or ""))
        criteria_source = "current_text_creator_edits_stripped"
    else:
        criteria = mc.tiptap_text(data_t.get("description"))
        criteria_source = "edit_history"
    siblings = sibling_prices(index, cid, answer_id, t_ms, data_t) if answer_id else None
    comments = index.comments_before(cid, t_ms)
    chosen = mc.select_comments(comments)
    context = {
        "as_of": as_of,
        "t_ms": t_ms,
        "open_at_t": True,
        "not_open_reason": None,
        "contract_id": cid,
        "answer_id": answer_id,
        "created_date": mc.iso_date(start_ms),
        "question_at_t": data_t.get("question") or prop.get("parent_question"),
        "answer_text_at_t": answer_text_t,
        "criteria": {"text": criteria, "source": criteria_source,
                     "reconstructed": criteria_source == "edit_history",
                     "description_edits_before_t": facts["description_edits_before_t"],
                     "description_edits_after_t": facts["description_edits_after_t"]},
        "price": {"p_t": round(p_t, 6), "source": price_source, "n_bets": n_trades, "n_traders": n_traders,
                  "n_bet_records": len(bets),
                  "last_bet_date": mc.iso_date(bets[-1]["created_time"]) if bets else None},
        "price_history": mc.price_history(bets, start_ms, t_ms, initial),
        "sibling_prices": siblings,
        "n_comments_before_t": len(comments),
        "comments": [{"date": c["date"], "user": c["user"], "text": c["text"], "created_time": c["created_time"]}
                     for c in chosen],
    }
    # Hard guard: nothing timestamped at or after t.
    stamps = [b["created_time"] for b in bets] + [c["created_time"] for c in chosen]
    if any(s >= t_ms for s in stamps):
        raise AssertionError(f"{prop['id']}: record at or after t")
    audit.update({"open_at_t": True, "close_before_t": contract["close_time"] is not None and contract["close_time"] <= t_ms,
                  "question_edits_after_t": facts["question_edits_after_t"],
                  "criteria_changed_vs_export": squash_eq(criteria, prop.get("criteria_text")),
                  "comments_edited_after_t": sum(c["edited_after_t"] for c in chosen),
                  "comment_days_to_resolution": [
                      (audit["resolution_time"] - c["created_time"]) / DAY if audit["resolution_time"] else None
                      for c in chosen]})
    return context, None, audit


def squash_eq(a: str | None, b: str | None) -> bool:
    """True when the as-of text differs from the exported (current) text."""
    return re.sub(r"\s+", "", a or "") != re.sub(r"\s+", "", b or "")


def as_of_for(pair: dict, props: dict[str, dict]) -> str:
    week = min(props[pair["p_id"]]["closing_week"], props[pair["q_id"]]["closing_week"])
    return (dt.date.fromisoformat(week) - dt.timedelta(days=7)).isoformat()


def leakage_audit(contexts: list[tuple[dict, dict]]) -> dict:
    hits = collections.Counter()
    near = collections.Counter()
    days = []
    examples = []
    n = 0
    for context, audit in contexts:
        for c, d in zip(context["comments"], audit.get("comment_days_to_resolution", [])):
            n += 1
            if d is not None:
                days.append(d)
                for cut in (1, 7, 30):
                    if d <= cut:
                        near[f"within_{cut}d_of_resolution"] += 1
            for name, rx in RESOLUTION_WORDS.items():
                if rx.search(c["text"]):
                    hits[name] += 1
                    if name == "resolves_yes_no" and len(examples) < 8:
                        examples.append({"id": audit["id"], "date": c["date"], "text": c["text"][:200]})
    q = statistics.quantiles(days, n=10) if len(days) > 10 else []
    return {"comments_included": n, "resolution_word_hits": dict(hits), "comment_date_vs_resolution": dict(near),
            "days_to_resolution_deciles": [round(x, 1) for x in q], "min_days_to_resolution": min(days) if days else None,
            "resolves_yes_no_examples": examples}


def price_distribution(ps: list[float]) -> dict:
    edges = [0, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0]
    counts = collections.Counter()
    for p in ps:
        if p <= 0.01:
            counts["<=0.01"] += 1
        elif p >= 0.99:
            counts[">=0.99"] += 1
        else:
            k = bisect.bisect_left(edges, p)
            counts[f"({edges[k - 1]},{edges[k]}]"] += 1
    n = len(ps)
    return {"n": n, "bins": dict(sorted(counts.items())), "frac_le_0.01": counts["<=0.01"] / n,
            "frac_ge_0.99": counts[">=0.99"] / n, "frac_extreme": (counts["<=0.01"] + counts[">=0.99"]) / n,
            "median": statistics.median(ps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/home/agent/data/jev-disjoint-v1-filtered-20260919")
    ap.add_argument("--index", default="/home/agent/data/manifold-context-20260923/index.sqlite")
    ap.add_argument("--out", default="/home/agent/data/jev-market-context-20260923")
    ap.add_argument("--require-open", action="store_true",
                    help="also remove pairs where either proposition is not open at t (Kelvin, 2026-09-23)")
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.out)
    index = Index(Path(a.index))
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": str(source), "require_open": a.require_open, "source_manifest_sha256": sha256(source / "manifest.json"),
                      "index": a.index, "t_rule": "t = 00:00 UTC of (min(closing_week of P, Q) - 7 days); "
                      "only records with createdTime < t", "counts": {}, "removals": {}, "audit": {},
                      "price_distribution": {}, "criteria_sources": {}, "files_sha256": {},
                      "contract_edit_chain_breaks": len(index.chain_breaks)}
    for split in SPLITS:
        src, dst = source / split, out / split
        dst.mkdir(parents=True, exist_ok=True)
        props = {r["id"]: r for r in jsonl(src / "propositions.jsonl")}
        pairs = jsonl(src / "pairs.jsonl")
        prop_reasons: collections.Counter = collections.Counter()
        pair_reasons: collections.Counter = collections.Counter()
        kept_pairs, contexts, audits = [], {}, []
        for pair in pairs:
            as_of = as_of_for(pair, props)
            reasons = []
            built = {}
            for pid in (pair["p_id"], pair["q_id"]):
                context, reason, audit = build_context(index, props[pid], as_of)
                if not reason and a.require_open and not context["open_at_t"]:
                    reason = "not_open_at_t:" + context["not_open_reason"]
                if reason:
                    reasons.append(reason)
                    prop_reasons[reason] += 1
                else:
                    built[pid] = (context, audit)
            if reasons:
                pair_reasons[sorted(reasons)[0]] += 1
                continue
            kept_pairs.append(pair)
            contexts.update(built)
        kept_pair_ids = {p["pair_id"] for p in kept_pairs}
        kept_props = {pid for p in kept_pairs for pid in (p["p_id"], p["q_id"])}
        write_jsonl(dst / "pairs.jsonl", kept_pairs)
        write_jsonl(dst / "propositions.jsonl",
                    ({**props[pid], "context": contexts[pid][0]} for pid in sorted(kept_props)))
        audits = [contexts[pid][1] for pid in sorted(kept_props)]
        write_jsonl(dst / "context-audit.jsonl", audits)
        counts = {"pairs_source": len(pairs), "pairs": len(kept_pairs), "propositions": len(kept_props)}
        for name in ("queries.jsonl", "full-label-targets.jsonl", "one-label-targets.jsonl", "hidden-label-audit.jsonl"):
            rows = jsonl(src / name)
            if name == "queries.jsonl" or name == "hidden-label-audit.jsonl":
                keep = [r for r in rows if r["pair_id"] in kept_pair_ids]
            else:
                keep = [r for r in rows if r["query_id"].rsplit("-q", 1)[0] in kept_pair_ids]
            write_jsonl(dst / name, keep)
            counts[name] = len(keep)
        singles = [r for r in jsonl(src / "single-targets.jsonl") if r["id"] in kept_props]
        write_jsonl(dst / "single-targets.jsonl", singles)
        counts["single-targets.jsonl"] = len(singles)
        manifest["counts"][split] = counts
        manifest["removals"][split] = {"pairs_removed": len(pairs) - len(kept_pairs),
                                       "pairs_by_first_reason": dict(pair_reasons),
                                       "propositions_by_reason": dict(prop_reasons)}
        ctx = [contexts[pid] for pid in sorted(kept_props)]
        manifest["audit"][split] = leakage_audit(ctx)
        manifest["audit"][split].update({
            "markets_closed_before_t": sum(x.get("close_before_t", False) for _, x in ctx),
            "question_edited_after_t": sum(x.get("question_edits_after_t", 0) > 0 for _, x in ctx),
            "criteria_differs_from_export_text": sum(x.get("criteria_changed_vs_export", False) for _, x in ctx),
            "comments_edited_after_t_shown_as_of_t": sum(x.get("comments_edited_after_t", 0) for _, x in ctx),
            "records_at_or_after_t": 0,
        })
        opened = [(c, x) for c, x in ctx if c["open_at_t"]]
        both_open = sum(contexts[p["p_id"]][0]["open_at_t"] and contexts[p["q_id"]][0]["open_at_t"] for p in kept_pairs)
        counts.update({"propositions_open_at_t": len(opened), "pairs_both_open_at_t": both_open,
                       "not_open_by_reason": dict(collections.Counter(c["not_open_reason"] for c, _ in ctx
                                                                      if not c["open_at_t"]))})
        ps = [c["price"]["p_t"] for c, _ in opened]
        manifest["price_distribution"][split] = {
            "all": price_distribution(ps),
            "binary": price_distribution([c["price"]["p_t"] for c, _ in opened if not c["answer_id"]]),
            "option": price_distribution([c["price"]["p_t"] for c, _ in opened if c["answer_id"]]),
            "no_bets_before_t": sum(c["price"]["source"] != "last_bet" for c, _ in opened),
            "median_trades": statistics.median(c["price"]["n_bets"] for c, _ in opened),
            "median_traders": statistics.median(c["price"]["n_traders"] for c, _ in opened),
            "median_comments_before_t": statistics.median(c["n_comments_before_t"] for c, _ in opened),
            "with_no_comments": sum(not c["comments"] for c, _ in opened),
        }
        manifest["criteria_sources"][split] = dict(collections.Counter(c["criteria"]["source"] for c, _ in ctx))
        print(split, json.dumps(counts), json.dumps(manifest["removals"][split]), flush=True)
    for name in ("excluded-parent-ids.json",):
        shutil.copy(source / name, out / name)
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json",):
            manifest["files_sha256"][str(path.relative_to(out))] = sha256(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("removals", "audit", "price_distribution", "criteria_sources")},
                     indent=1))


if __name__ == "__main__":
    main()
