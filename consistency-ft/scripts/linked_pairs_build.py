#!/usr/bin/env python3
"""Linked-pairs search experiment (docs/linked-pairs-search-20261001.md): question table, candidate pairs,
outcome-blind link judge, pair selection. Nothing here reads an outcome.

  linked_pairs_build.py questions    -> OUT/questions.jsonl        (Metaculus, MiniBench, Manifold; no outcomes)
  linked_pairs_build.py generate     -> OUT/generated.jsonl        (Opus lists linked pairs from MiniBench/Metaculus titles)
  linked_pairs_build.py candidates   -> OUT/candidates.jsonl       (structural + generated + lexical pairs with a feasible t)
  linked_pairs_build.py judge        -> OUT/judge.jsonl            (Opus 5.5 link judge; titles and criteria only)
  linked_pairs_build.py select       -> OUT/pairs.jsonl, OUT/selection-counts.json

Outcome blindness: the Metaculus list API returns resolution = null for every post (checked: 700 of 700
binary posts), so the raw fetch has no outcomes. Manifold resolutions are read only as "is it YES or NO
(not CANCEL/MKT)" in SQL, never the side. MiniBench truth/ files are not opened here.

Resolution criteria: the Metaculus API returns empty resolution_criteria, fine_print and description for every resolved
post (checked 2026-10-01 on the list and detail endpoints and on the 09-20 API dump), so Metaculus questions carry their
title only. MiniBench questions take their criteria from the bt-scale snapshots (taken while the question was open).

Dates: "end" = the earlier of the actual and scheduled resolution (Manifold: close or resolution) and sets t only.
Prompts, the generator and the judge never see it, because an early end reveals an early resolution. They see
"shown_date": the scheduled resolution date for Metaculus/MiniBench (scheduled_resolve_time; unchanged by
resolution: 179 of 179 MiniBench snapshots agree with the API) and, for Manifold, the close date as of t
(market_versions), recomputed with the title, description and answer text at each selected pair's t.
"""
import argparse
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path("/home/agent/data/linked-pairs-20261001")
MX_RAW = OUT / "raw" / "metaculus_posts.jsonl"
BT = Path("/home/agent/workspaces/calibrated-ai/minibench/backtest/bt-scale/snapshots")
MANIFOLD = "file:/home/agent/data/manifold-asof-20260923/asof.sqlite?mode=ro"
SEED = 20261001
T_MIN = datetime(2026, 7, 1, tzinfo=timezone.utc)      # forecast date t on or after the model's knowledge cutoff
LEAD = timedelta(days=7)                                # t = 00:00 UTC of (earlier end - 7 days)
MANIFOLD_MIN_BETTORS = 15
MANIFOLD_BAD_GROUPS = re.compile(r"personal|will-i|nonpredictive|coin-?flip|whale|^fun$|-fun$|self-resol|meta-market|"
                                 r"unranked|joke|nonsense|random|dice|daily-?coin|test-market|sweepstakes", re.I)
LEX_MIN = {"manifold": 0.20, "metaculus": 0.10, "minibench": 0.10}   # title cosine for a lexical candidate
LEX_TOP = 5             # nearest lexical neighbours kept per question
SIB_PER_CLUSTER = 3     # structural sibling candidates per Metaculus group / Manifold multiple-choice market
JUDGE_CAP_PER_SOURCE = 450
JUDGE_BATCH = 20
TARGET = {"manifold": 34, "metaculus": 33, "minibench": 33}
SPILL_ORDER = ("metaculus", "manifold", "minibench")
MAX_USES = 2            # each question in at most 2 pairs
CLUSTER_CAP = 3         # selected pairs per structural cluster

STOP = set("""a an the of in on at to for by with from and or not no yes will be is are was were been being as it its this
that these those before after between during than then which who whom whose what when where how why do does did
any all each other into over under above below up down out off again further once more most such only own same
so very can could should would may might must shall about against per via vs versus least than 2026 end
january february march april may june july august september october november december jan feb mar apr jun jul aug
sep sept oct nov dec least more than inclusive between""".split()) | {str(d) for d in range(1, 32)}


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(s / 1000, tz=timezone.utc)
    s = s.replace("Z", "+00:00")
    if re.search(r"\.\d+\+", s):
        s = re.sub(r"\.(\d+)\+", lambda m: "." + m.group(1)[:6].ljust(6, "0") + "+", s)
    return datetime.fromisoformat(s)


def criteria_of(q):
    parts = [str(q.get("resolution_criteria") or "").strip(), str(q.get("fine_print") or "").strip()]
    return "\n\n".join(p for p in parts if p)


# --- questions ---------------------------------------------------------------------------------------------

def metaculus_questions():
    out = {}
    series_of = {}
    for line in open(MX_RAW):
        p = json.loads(line)
        proj = p.get("projects") or {}
        slugs = [(s.get("slug") or s.get("name") or "") for k in ("question_series", "tournament") for s in proj.get(k, [])]
        mb = any("minibench" in s.lower() for s in slugs)
        if p["_ft"] == "binary":
            qs = [(p["question"], None)]
        elif p["_ft"] == "group_of_questions":
            g = p["group_of_questions"]
            qs = [({**q, "resolution_criteria": q.get("resolution_criteria") or g.get("resolution_criteria"),
                    "fine_print": q.get("fine_print") or g.get("fine_print")}, f"mxg:{p['id']}")
                  for q in g["questions"] if q["type"] == "binary"]
        else:
            continue
        for q, cluster in qs:
            if q.get("type") != "binary" or q.get("status") != "resolved":
                continue
            art, srt = parse(q.get("actual_resolve_time")), parse(q.get("scheduled_resolve_time"))
            if not art or not srt:
                continue
            end = min(art, srt)
            src = "minibench" if mb else "metaculus"
            qid = f"{'mb' if mb else 'mx'}:{q['id']}"
            out[qid] = {"qid": qid, "source": src, "cluster": cluster, "post_id": p["id"], "question_id": q["id"],
                        "title": q.get("title") or p.get("title"), "option": None, "criteria": criteria_of(q),
                        "created": iso(parse(q.get("open_time") or p.get("open_time"))), "end": iso(end),
                        "shown_date": iso(srt), "resolved_at": iso(art), "series": [s for s in slugs if s][:3],
                        "url": f"https://www.metaculus.com/questions/{p['id']}/"}
            series_of[qid] = slugs
    # MiniBench snapshots from the bt-scale backtest (binary only); the same question may already be in the list
    for f in sorted(BT.glob("*.json")):
        s = json.loads(f.read_text())
        if not isinstance(s, dict) or s.get("type") != "binary":
            continue
        qid = f"mb:{s['question_id']}"
        if qid in out:
            if not out[qid]["criteria"]:
                out[qid]["criteria"] = criteria_of(s)
            continue
        srt = parse(s["scheduled_resolve_time"])
        out[qid] = {"qid": qid, "source": "minibench", "cluster": None, "post_id": int(s["post_id"]),
                    "question_id": int(s["question_id"]), "title": s["title"], "option": None,
                    "criteria": criteria_of(s), "created": iso(parse(s["open_time"])), "end": iso(srt),
                    "shown_date": iso(srt), "resolved_at": None, "series": [f"round-{s.get('round')}"], "url": s.get("url")}
    return out


def manifold_questions():
    con = sqlite3.connect(MANIFOLD, uri=True)
    t_floor = int((T_MIN + LEAD).timestamp() * 1000)
    rows = con.execute("""select mid, id, question, created, close, resolution_time, group_slugs, outcome_type, url, slug
        from markets where resolution_time >= ? and (deleted is null or deleted = 0)
        and (visibility is null or visibility = 'public')
        and ((outcome_type = 'BINARY' and resolution in ('YES','NO')) or outcome_type = 'MULTIPLE_CHOICE')""",
                       (t_floor,)).fetchall()
    out = {}

    def version(mid, at):
        r = con.execute("""select question, description from market_versions where mid = ? and valid_from <= ?
                           and valid_to > ? order by valid_from desc limit 1""", (mid, at, at)).fetchone()
        return r

    for mid, mkid, q, cr, cl, rt, gs, ot, url, slug in rows:
        groups = json.loads(gs or "[]")
        if any(MANIFOLD_BAD_GROUPS.search(g) for g in groups):
            continue
        if ot == "BINARY":
            end = min(x for x in (cl, rt) if x)
            tt = end - LEAD.total_seconds() * 1000
            if tt <= cr:
                continue
            nb = con.execute("select count(distinct uid) from bets where mid = ? and time < ?", (mid, tt)).fetchone()[0]
            if nb < MANIFOLD_MIN_BETTORS:
                continue
            v = version(mid, int(tt))
            qid = f"mf:{mid}"
            out[qid] = {"qid": qid, "source": "manifold", "cluster": None, "mid": mid, "aid": None,
                        "title": (v[0] if v else q), "option": None, "criteria": (v[1] if v else "") or "",
                        "created": iso(parse(cr)), "end": iso(parse(end)), "resolved_at": iso(parse(rt)),
                        "series": groups[:3], "bettors": nb, "url": url or f"https://manifold.markets/{slug}"}
        else:
            ans = con.execute("""select aid, text, created, resolution, resolution_time, is_other from answers
                                 where mid = ? and resolution in ('YES','NO')""", (mid,)).fetchall()
            for aid, text, acr, ares, art, other in ans:
                if other:
                    continue
                art = art or rt
                end = min(x for x in (cl, art) if x)
                tt = end - LEAD.total_seconds() * 1000
                if tt <= max(cr, acr) or end < t_floor:
                    continue
                nb = con.execute("select count(distinct uid) from bets where mid = ? and aid = ? and time < ?",
                                 (mid, aid, tt)).fetchone()[0]
                if nb < MANIFOLD_MIN_BETTORS:
                    continue
                v = version(mid, int(tt))
                av = con.execute("""select text from answer_versions where aid = ? and valid_from <= ? and valid_to > ?
                                    order by valid_from desc limit 1""", (aid, int(tt), int(tt))).fetchone()
                qid = f"mf:{mid}:{aid}"
                out[qid] = {"qid": qid, "source": "manifold", "cluster": f"mfm:{mid}", "mid": mid, "aid": aid,
                            "title": (v[0] if v else q), "option": (av[0] if av else text),
                            "criteria": (v[1] if v else "") or "", "created": iso(parse(max(cr, acr))),
                            "end": iso(parse(end)), "resolved_at": iso(parse(art)), "series": groups[:3],
                            "bettors": nb, "url": url or f"https://manifold.markets/{slug}"}
    return out


def cmd_questions(_a):
    qs = {**metaculus_questions(), **manifold_questions()}
    with open(OUT / "questions.jsonl", "w") as f:
        for q in qs.values():
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    c = Counter(q["source"] for q in qs.values())
    print(json.dumps({"questions": len(qs), "by_source": c, "with_cluster": Counter(q["source"] for q in qs.values() if q["cluster"])}))


# --- candidates ----------------------------------------------------------------------------------------------

def tokens(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9$%.'-]*[a-z0-9%]|[a-z0-9]", text.lower()) if w not in STOP and len(w) > 1]


def tfidf(texts):
    docs = [Counter(tokens(t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    vecs = []
    for d in docs:
        v = {w: (1 + math.log(c)) * math.log((1 + n) / (1 + df[w])) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def pair_t(a, b):
    """t = 00:00 UTC of (earlier end - 7 days), or the first 00:00 UTC after both questions opened if that is later.
    Feasible if t >= 2026-07-01, t is at least 1 day before the earlier end and both are unresolved at t."""
    end = min(parse(a["end"]), parse(b["end"]))
    t = (end - LEAD).replace(hour=0, minute=0, second=0, microsecond=0)
    opened = max(parse(a["created"]), parse(b["created"]))
    first = opened.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    t = max(t, first)
    if t < T_MIN or end - t < timedelta(days=1):
        return None
    for q in (a, b):
        if q.get("resolved_at") and parse(q["resolved_at"]) <= t:
            return None
    return t


def text_of(q):
    return q["title"] + (" " + q["option"] if q.get("option") else "")


def cmd_candidates(_a):
    qs = [json.loads(l) for l in open(OUT / "questions.jsonl")]
    rng = random.Random(SEED)
    cands, seen = [], set()

    def add(a, b, kind, sim):
        key = tuple(sorted((a["qid"], b["qid"])))
        if key in seen or a["qid"] == b["qid"]:
            return
        if text_of(a).strip().lower() == text_of(b).strip().lower() and a["criteria"] == b["criteria"]:
            return  # identical question (near-duplicate by definition)
        t = pair_t(a, b)
        if t is None:
            return
        seen.add(key)
        p, q = (a, b) if a["qid"] < b["qid"] else (b, a)
        cands.append({"pair_id": f"{p['qid']}|{q['qid']}", "p": p["qid"], "q": q["qid"], "source": p["source"],
                      "kind": kind, "sim": round(sim, 3), "t": iso(t), "cluster": p["cluster"] if kind == "structural" else None})

    by_src = defaultdict(list)
    for q in qs:
        by_src[q["source"]].append(q)
    Qd = {q["qid"]: q for q in qs}
    gen = OUT / "generated.jsonl"
    generated = [json.loads(l) for l in open(gen)] if gen.exists() else []
    for src, items in by_src.items():
        vecs = tfidf([text_of(q) for q in items])
        # structural siblings: same Metaculus group or same Manifold multiple-choice market
        clusters = defaultdict(list)
        for i, q in enumerate(items):
            if q["cluster"]:
                clusters[q["cluster"]].append(i)
        for cl, idx in sorted(clusters.items()):
            pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
            rng.shuffle(pairs)
            n = 0
            for i, j in pairs:
                if n >= SIB_PER_CLUSTER:
                    break
                before = len(cands)
                add(items[i], items[j], "structural", cos(vecs[i], vecs[j]))
                n += len(cands) > before
        # generated: pairs Opus listed from the titles of one MiniBench round or one Metaculus end-date window
        for g in generated:
            a, b = Qd.get(g["a"]), Qd.get(g["b"])
            if a and b and a["source"] == src and b["source"] == src:
                add(a, b, "generated", cos(vecs[items.index(a)], vecs[items.index(b)]))
        # lexical: title cosine >= LEX_MIN, top LEX_TOP per question, outside the question's own cluster
        for i, q in enumerate(items):
            sims = []
            for j, r in enumerate(items):
                if j == i or (q["cluster"] and q["cluster"] == r["cluster"]):
                    continue
                s = cos(vecs[i], vecs[j])
                if s >= LEX_MIN[src]:
                    sims.append((s, j))
            for s, j in sorted(sims, reverse=True)[:LEX_TOP]:
                add(q, items[j], "lexical", s)
    with open(OUT / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    print(json.dumps({"candidates": len(cands), "by_source_kind": Counter(f"{c['source']}/{c['kind']}" for c in cands)}))


# --- generated candidates (MiniBench and Metaculus: titles are topically diverse, so title overlap finds few) ----

GEN_PROMPT = """Below is a list of forecasting questions (id, question, resolution date). Find pairs of questions whose outcomes are genuinely linked: one logically constrains the other (implication, mutual exclusion, nested thresholds or deadlines on the same quantity), or learning one outcome would move a careful forecaster's probability of the other by about 10 percentage points or more. Use only the question texts; you do not know and must not guess how any question resolved. Skip near-duplicates (two questions about essentially the same event). List every such pair you find, at most {k}, strongest links first.

Reply with only a JSON array of id pairs: [["id1", "id2"], ...]

QUESTIONS:
"""
GEN_CHUNK = 60          # Metaculus questions per window (sorted by end date)
GEN_K = 40              # pairs asked for per group


def gen_groups(qs):
    floor = iso(T_MIN + LEAD)
    live = [q for q in qs if q["end"] >= floor and q["source"] in ("minibench", "metaculus")]
    groups = []
    rounds = defaultdict(list)
    for q in live:
        if q["source"] == "minibench":
            rounds[q["series"][0] if q["series"] else "-"].append(q)
    groups += [(f"minibench/{k}", v) for k, v in sorted(rounds.items())]
    mx = sorted((q for q in live if q["source"] == "metaculus"), key=lambda q: (q["shown_date"], q["qid"]))
    groups += [(f"metaculus/window{i // GEN_CHUNK}", mx[i:i + GEN_CHUNK]) for i in range(0, len(mx), GEN_CHUNK)]
    return groups


def cmd_generate(a):
    from linked_pairs_search import Budget  # noqa: E402
    qs = [json.loads(l) for l in open(OUT / "questions.jsonl")]
    budget = Budget()

    def run(g):
        name, items = g
        lines = "\n".join(f"{q['qid']} | {q['title']} | scheduled to resolve {q['shown_date'][:10]}" for q in items)
        got, cost, raw = None, 0.0, []
        for attempt in range(2):  # one retry if the reply is not a JSON array
            budget.check(0.5)
            try:
                r = opus_json(JUDGE_SYSTEM, GEN_PROMPT.format(k=GEN_K) + lines, effort="medium")
            finally:
                budget.release(0.5)
            budget.add("llm", r["cost"], f"generate {name}")
            cost += r["cost"]
            raw.append(r["text"])
            m = re.search(r"\[\s*(?:\[.*\]\s*)?\]", r["text"], re.S)
            try:
                got = json.loads(m.group(0)) if m else None
            except ValueError:
                got = None
            if isinstance(got, list):
                break
        (OUT / "generate-raw").mkdir(exist_ok=True)
        (OUT / "generate-raw" / f"{name.replace('/', '__')}.json").write_text(json.dumps(raw))
        got = got or []
        ids = {q["qid"] for q in items}
        rows = [{"group": name, "a": x[0], "b": x[1]} for x in got
                if isinstance(x, list) and len(x) == 2 and x[0] in ids and x[1] in ids and x[0] != x[1]]
        return name, len(items), rows, cost

    out, stats = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for name, n, rows, cost in ex.map(run, gen_groups(qs)):
            out += rows
            stats.append({"group": name, "questions": n, "pairs": len(rows), "cost": round(cost, 4)})
    with open(OUT / "generated.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"generated": len(out), "groups": stats}))


# --- judge ---------------------------------------------------------------------------------------------------

JUDGE_SYSTEM = "You are a careful analyst of forecasting questions."
JUDGE_PROMPT = """Below are pairs of forecasting questions (A and B). For each pair, judge how strongly the two outcomes are linked, using only the question texts. You do not know and must not guess how either question resolved.

link (0-3):
3 = logical constraint: one outcome makes the other impossible or certain in at least one direction (implication, mutual exclusion, nested thresholds or deadlines on the same quantity).
2 = strong causal or statistical dependence: learning one outcome would move a careful forecaster's probability of the other by about 10 percentage points or more.
1 = same topic, weak dependence.
0 = unrelated.
near_duplicate: true if A and B ask about essentially the same event, so they would resolve the same way in almost every world (only wording, source or tiny threshold differences).
forecastable: true if both are about public real-world events that news or public data could inform before the stated date (false for personal, random, self-resolving or joke questions).

Reply with only a JSON array, one object per pair, in order: [{"id": "...", "link": 0-3, "near_duplicate": true/false, "forecastable": true/false}]

PAIRS:
"""


def judge_block(c, Q):
    def one(name, q):
        crit = re.sub(r"\s+", " ", q["criteria"] or "").strip()[:500]
        opt = f' | option: "{q["option"]}"' if q.get("option") else ""
        return f"{name}: {q['title']}{opt}\n   criteria: {crit or '(none given)'}"
    return f"id: {c['pair_id']}\nas of: {c['t'][:10]}\n{one('A', Q[c['p']])}\n{one('B', Q[c['q']])}"


def opus_json(system, user, effort="medium"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from linked_pairs_search import claude_call  # noqa: E402
    return claude_call(system, user, effort=effort, purpose="judge")


def cmd_judge(a):
    from linked_pairs_search import Budget  # noqa: E402
    Q = {q["qid"]: q for q in map(json.loads, open(OUT / "questions.jsonl"))}
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    rng = random.Random(SEED + 1)
    by = defaultdict(list)
    for c in cands:
        by[c["source"]].append(c)
    todo = []
    for src in sorted(by):
        cs = by[src]
        rng.shuffle(cs)
        todo += cs[:JUDGE_CAP_PER_SOURCE]
    done = set()
    jpath = OUT / "judge.jsonl"
    if jpath.exists():
        done = {json.loads(l)["pair_id"] for l in open(jpath)}
    todo = [c for c in todo if c["pair_id"] not in done]
    batches = [todo[i:i + JUDGE_BATCH] for i in range(0, len(todo), JUDGE_BATCH)]
    budget = Budget()

    def run(batch):
        user = JUDGE_PROMPT + "\n\n".join(judge_block(c, Q) for c in batch)
        budget.check(0.3)
        try:
            r = opus_json(JUDGE_SYSTEM, user, effort="medium")
        finally:
            budget.release(0.3)
        budget.add("llm", r["cost"], "judge batch")
        m = re.search(r"\[.*\]", r["text"], re.S)
        try:
            got = {str(x.get("id")): x for x in json.loads(m.group(0)) if isinstance(x, dict)} if m else {}
        except ValueError:
            got = {}
        rows = []
        for c in batch:
            x = got.get(c["pair_id"])
            rows.append({"pair_id": c["pair_id"], "judge": x, "cost": r["cost"] / len(batch)})
        return rows

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(jpath, "a") as f:
        for rows in ex.map(run, batches):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
    js = [json.loads(l) for l in open(jpath)]
    print(json.dumps({"judged": len(js), "missing": sum(1 for j in js if not j["judge"]),
                      "links": Counter(str((j["judge"] or {}).get("link")) for j in js)}))


# --- selection -----------------------------------------------------------------------------------------------

def cmd_select(_a):
    Q = {q["qid"]: q for q in map(json.loads, open(OUT / "questions.jsonl"))}
    cands = {c["pair_id"]: c for c in map(json.loads, open(OUT / "candidates.jsonl"))}
    judged = {j["pair_id"]: j["judge"] for j in map(json.loads, open(OUT / "judge.jsonl"))}
    judged = {p: j for p, j in judged.items() if p in cands}   # judged earlier, no longer a candidate: ignored
    counts = {"questions": Counter(q["source"] for q in Q.values()),
              "candidates": Counter(c["source"] for c in cands.values()),
              "judged": Counter(cands[p]["source"] for p in judged),
              "judge_missing": Counter(cands[p]["source"] for p, j in judged.items() if not j)}
    ok = []
    for p, j in judged.items():
        if not j:
            continue
        c = cands[p]
        if j.get("forecastable") is False:
            counts.setdefault("drop_unforecastable", Counter())[c["source"]] += 1
            continue
        if j.get("near_duplicate"):
            counts.setdefault("drop_near_duplicate", Counter())[c["source"]] += 1
            continue
        if not isinstance(j.get("link"), int) or j["link"] < 2:
            counts.setdefault("drop_link_below_2", Counter())[c["source"]] += 1
            continue
        ok.append({**c, "link": j["link"]})
    con = sqlite3.connect(MANIFOLD, uri=True)
    asof_n = Counter()

    def shown(qid, t):
        """The question as shown at t: Manifold title, description, answer text and close date as of t."""
        q = dict(Q[qid])
        if q["source"] != "manifold":
            return q
        at = int(parse(t).timestamp() * 1000)
        v = con.execute("""select question, description, close from market_versions where mid = ? and valid_from <= ?
                           and valid_to > ? order by valid_from desc limit 1""", (q["mid"], at, at)).fetchone()
        if v:
            q["title"], q["criteria"] = v[0] or q["title"], v[1] or ""
            asof_n["text_asof_t"] += 1
        if v and v[2]:
            q["shown_date"] = iso(parse(v[2]))
            asof_n["close_asof_t"] += 1
        else:
            q["shown_date"] = None
            asof_n["close_missing"] += 1
        if q.get("aid"):
            av = con.execute("""select text from answer_versions where aid = ? and valid_from <= ? and valid_to > ?
                                order by valid_from desc limit 1""", (q["aid"], at, at)).fetchone()
            if av:
                q["option"] = av[0]
        return q

    # both questions open at t: a Manifold market whose close date as of t is not after t was closed for trading
    keep = []
    for c in ok:
        sp, sq = shown(c["p"], c["t"]), shown(c["q"], c["t"])
        if any(x["shown_date"] is None or parse(x["shown_date"]) <= parse(c["t"]) for x in (sp, sq)):
            counts.setdefault("drop_closed_at_t", Counter())[c["source"]] += 1
            continue
        keep.append(c)
    ok = keep
    asof_n.clear()
    counts["eligible"] = Counter(c["source"] for c in ok)
    rng = random.Random(SEED + 2)
    ok.sort(key=lambda c: c["pair_id"])
    rng.shuffle(ok)
    uses, cl_n, chosen = Counter(), Counter(), []
    quota = dict(TARGET)
    by = defaultdict(list)
    for c in ok:
        by[c["source"]].append(c)

    def take(src, n):
        got = []
        for c in by[src]:
            if len(got) >= n:
                break
            if c in chosen or uses[c["p"]] >= MAX_USES or uses[c["q"]] >= MAX_USES:
                continue
            if c["cluster"] and cl_n[c["cluster"]] >= CLUSTER_CAP:
                continue
            uses[c["p"]] += 1
            uses[c["q"]] += 1
            if c["cluster"]:
                cl_n[c["cluster"]] += 1
            chosen.append(c)
            got.append(c)
        return len(got)

    short = 0
    for src in TARGET:
        short += quota[src] - take(src, quota[src])
    for src in SPILL_ORDER:
        if short <= 0:
            break
        short -= take(src, short)
    leftover = [c for src in SPILL_ORDER for c in by[src] if c not in chosen
                and uses[c["p"]] < MAX_USES and uses[c["q"]] < MAX_USES]
    order = list(range(len(chosen)))
    rng.shuffle(order)
    with open(OUT / "pairs.jsonl", "w") as f:
        for rank, i in enumerate(order):
            c = chosen[i]
            f.write(json.dumps({**c, "run_order": rank, "P": shown(c["p"], c["t"]), "Q": shown(c["q"], c["t"])},
                               ensure_ascii=False) + "\n")
    counts["manifold_asof"] = dict(asof_n)
    smoke = leftover[0] if leftover else None
    if smoke:
        (OUT / "smoke-pair.json").write_text(json.dumps({**smoke, "P": shown(smoke["p"], smoke["t"]),
                                                         "Q": shown(smoke["q"], smoke["t"])}, ensure_ascii=False))
    counts["selected"] = Counter(c["source"] for c in chosen)
    counts["selected_kind"] = Counter(f"{c['source']}/{c['kind']}" for c in chosen)
    counts["selected_link"] = Counter(str(c["link"]) for c in chosen)
    counts["smoke_pair"] = smoke["pair_id"] if smoke else None
    (OUT / "selection-counts.json").write_text(json.dumps(counts, indent=1, default=dict))
    print(json.dumps(counts, default=dict))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("questions")
    g = sub.add_parser("generate")
    g.add_argument("--workers", type=int, default=6)
    sub.add_parser("candidates")
    j = sub.add_parser("judge")
    j.add_argument("--workers", type=int, default=6)
    sub.add_parser("select")
    a = ap.parse_args()
    {"questions": cmd_questions, "generate": cmd_generate, "candidates": cmd_candidates, "judge": cmd_judge, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
