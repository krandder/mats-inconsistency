"""Structured-pair set from Manifold: pairs whose logical dependence is built in.

Question (Kelvin, 24 Sep): on pairs with KNOWN structure, do Qwen3 models (+ consistency
projection) capture it, and beat market prices + independence on the joint forms?

Types (relation recorded per pair; P/Q order is a seeded coin flip):
  threshold  same quantity, different thresholds (scripts/structured_pairs_rules.py);
             the stricter rung implies the looser one.
  date       same event "by/before" an earlier vs a later date; earlier implies later.
  mc_excl    two options of one single-answer multiple-choice market (sum_to_one = 1,
             resolved to one option): mutually exclusive.  The two options with the highest
             price at t (outcome-blind).
  control    random pairs of different parents with no shared group at t: no known relation.
  (explicit win/final-style relations were not built: no cheap rule with a checked direction.)

Universe: public Manifold BINARY markets and MULTIPLE_CHOICE options created on or after
2025-04-28 (the earliest creation week of the JEV pool; all after the Qwen3 release),
resolved YES/NO (single-answer MC: the chosen option YES, the rest NO).  Parents: never a
train/validation parent of jev-pairings-v1, jev-disjoint-v1(-filtered) or the open
market-context set, never one of the 178 excluded parents; test parents and unused parents only.
Candidate generation uses the final text; each pair is then re-checked on the as-of-t text
(same rule, same relation) and dropped otherwise.

t (JEV rule): e = min(close, resolution time) per proposition; closing week = Monday of e;
t = 00:00 UTC of (min closing week - 7 days).  Both open at t: market and option created
before t, resolution strictly after t, no resolution logged at or before t, status open at t;
a price at t exists; the market has >= 5 traders before t (outcome-blind quality filter).
Per ladder group at most 3 pairs, per MC market 1 pair; seeded choices, outcomes never read
for selection.  Labels are written to labels.jsonl for scoring only.

Prompts: exactly the market-context zero-shot layout (scripts/market_context_zeroshot.py
user_prompt, v6_direct, both question blocks with the as-of-t text), arms Q and P
(build_pack "small", the main test's size).  Runs use market_context_zeroshot.run with
split name "structured".

Usage:
  structured_pairs.py build [--out DIR]
  structured_pairs.py project|run --model M --arm Q|P [--cap 15]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import structured_pairs_rules as R  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402
from training.asof import AsOf, to_ms, iso  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

OUT = Path("/home/agent/data/structured-pairs-20260924")
DATA = Path("/home/agent/data")
SEED = 20260924
LO_CREATED = "2025-04-28"
MIN_TRADERS = 5
PER_GROUP = 3
MAX_USES = 2
TARGET = {"threshold": 300, "date": 300, "mc_excl": 300, "control": 300}
DAY = 86_400_000


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parents_of(root: Path, splits) -> set[str]:
    out = set()
    for s in splits:
        p = root / s / "propositions.jsonl"
        if p.exists():
            out |= {json.loads(l)["parent_id"] for l in p.read_text().splitlines() if l.strip()}
    return out


def monday(ms: int) -> dt.date:
    d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()
    return d - dt.timedelta(days=d.weekday())


# --- universe -------------------------------------------------------------------

def split_parents() -> dict[str, set[str]]:
    """Parents per JEV split, pooled over every JEV-derived data set."""
    filt = DATA / "jev-disjoint-v1-filtered-20260919"
    roots = (filt, DATA / "jev-disjoint-v1", DATA / "jev-pairings-v1", DATA / "jev-market-context-open-20260923")
    return {s: set().union(*(parents_of(r, [s]) for r in roots)) for s in ("train", "validation", "test")}


SPLIT_TARGET = {"train": 600, "validation": 120}  # per structured type; controls = structured total


def universe(store: AsOf, split: str | None = None) -> tuple[list[dict], dict]:
    """split=None: the structured TEST universe (test + unused parents, created >= LO_CREATED).

    split="train"/"validation" (training data for the 0.6B, 27 Sep): parents of that JEV split that are
    in no other JEV split, plus pre-JEV markets (created before LO_CREATED, in no JEV split) whose
    parent hash bucket (sha256 % 10) is 0 for validation and 1-9 for train; for the pre-JEV markets
    every proposition must resolve before the earliest structured-test t, so no training label
    postdates any test forecast time.  Never an excluded parent, a JEV test parent or a parent of the
    structured test set (/home/agent/data/structured-pairs-20260924)."""
    db = store.db
    filt = DATA / "jev-disjoint-v1-filtered-20260919"
    excluded = set(json.loads((filt / "excluded-parent-ids.json").read_text()))
    sp = split_parents()
    trainval, test = sp["train"] | sp["validation"], sp["test"]
    jev_ok = pre_ok = None
    lo = lo_created = to_ms(LO_CREATED)
    if split is not None:
        test_rows = jl(OUT / "pairs.jsonl")
        test_set = {r["p_parent"] for r in test_rows} | {r["q_parent"] for r in test_rows}
        cutoff = min(r["t_ms"] for r in test_rows)
        others = set().union(*(v for k, v in sp.items() if k != split))
        jev_ok = sp[split] - others - excluded - test_set
        anyjev = set().union(*sp.values())

        def pre_ok(mid_: str, created: int) -> bool:
            bucket = int(hashlib.sha256(mid_.encode()).hexdigest(), 16) % 10
            return (created < lo_created and mid_ not in anyjev and mid_ not in excluded and mid_ not in test_set
                    and (bucket == 0) == (split == "validation"))
        lo = 0
    stats = collections.Counter()
    props = []
    rows = db.execute("SELECT mid, id, question, created, close, resolution, resolution_time, outcome_type, sum_to_one, "
                      "visibility, coalesce(deleted, 0) AS del FROM markets WHERE created >= ? AND "
                      "outcome_type IN ('BINARY', 'MULTIPLE_CHOICE') AND resolution IS NOT NULL",
                      (lo,)).fetchall()
    for m in rows:
        if m["del"] or m["visibility"] != "public":
            stats["skip_private_or_deleted"] += 1; continue
        if split is not None:
            if m["id"] not in jev_ok and not pre_ok(m["id"], m["created"]):
                stats["skip_not_in_split"] += 1; continue
        elif m["id"] in excluded or m["id"] in trainval:
            stats["skip_trainval_or_excluded"] += 1; continue
        base = dict(mid=m["mid"], parent_id=m["id"], question=m["question"], created=m["created"], close=m["close"],
                    outcome_type=m["outcome_type"], sum_to_one=m["sum_to_one"],
                    src=split if split else ("test" if m["id"] in test else "unused"))
        if m["outcome_type"] == "BINARY":
            if m["resolution"] not in ("YES", "NO"):
                continue
            props.append(dict(base, id=m["id"], answer_id=None, answer_text=None, a_created=None,
                              label=int(m["resolution"] == "YES"), res_time=m["resolution_time"]))
            continue
        ans = db.execute("SELECT id, text, created, is_other, resolution, resolution_time FROM answers WHERE mid = ?",
                         (m["mid"],)).fetchall()
        if m["sum_to_one"] == 1:
            if m["resolution"] not in {a["id"] for a in ans}:
                continue  # MKT / CANCEL / CHOOSE_MULTIPLE: no single YES option
            for a in ans:
                if not a["is_other"]:
                    props.append(dict(base, id=f"{m['id']}:{a['id']}", answer_id=a["id"], answer_text=a["text"],
                                      a_created=a["created"], label=int(a["id"] == m["resolution"]),
                                      res_time=m["resolution_time"]))
        else:
            for a in ans:
                if a["is_other"] or a["resolution"] not in ("YES", "NO") or a["resolution_time"] is None:
                    continue
                props.append(dict(base, id=f"{m['id']}:{a['id']}", answer_id=a["id"],
                                  answer_text=a["text"], a_created=a["created"], label=int(a["resolution"] == "YES"),
                                  res_time=a["resolution_time"]))
    if split is not None:
        n0 = len(props)
        props = [p for p in props if p["parent_id"] in jev_ok or (p["res_time"] or 1 << 62) < cutoff]
        stats["pre_jev_dropped_resolved_after_test_t0"] = n0 - len(props)
        stats["by_origin"] = dict(collections.Counter("jev" if p["parent_id"] in jev_ok else "pre_jev" for p in props))
    for p in props:
        p["e"] = min(p["close"] or 1 << 62, p["res_time"])
    stats["propositions"] = len(props)
    stats["by_kind"] = dict(collections.Counter(f"{p['outcome_type']}/{p['sum_to_one']}/{p['src']}" for p in props))
    return props, stats


def created_date(ms: int) -> dt.date:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()


def ladder_key(kind: str, question: str, answer: str | None, ref: dt.date):
    """Rule key for one proposition's text: (template, value[, direction, unit])."""
    if kind == "threshold":
        if answer is None:
            k = R.threshold_key(question, question, ref)
        else:
            dtext = answer if R._direction(answer) else question + " " + answer
            k = R.threshold_key(answer, dtext, ref)
        return None if k is None else (k[0], k[2], k[3]), (k[1] if k else None)
    if answer is None:
        k = R.date_key(question, ref)
        return (None, None) if k is None else ((k[0],), k[1])
    k = R.date_key(answer, ref)
    if k is not None:
        return (k[0],), k[1]
    # option is a bare date and the question asks "... by ...?"
    if (not R.BAD.search(R._strip_comparators(question)) and not R.SELF_REF.search(question)
            and re.search(r"\b(by|before)\b\s*(\.\.\.|…|_+|\?|:|\(|\[|$)", question.strip(), re.I)):
        ds = R.find_dates(answer, ref)
        if len(ds) == 1 and ds[0][1] - ds[0][0] >= len(answer.strip()) - 3:
            return ("<optdate>",), ds[0][2]
    return None, None


def ladder_groups(props: list[dict], kind: str) -> dict:
    groups = collections.defaultdict(list)
    for p in props:
        if p["outcome_type"] == "MULTIPLE_CHOICE" and p["sum_to_one"] == 1:
            continue  # single-answer options are exclusive buckets, not ladders
        key, v = ladder_key(kind, p["question"], p["answer_text"], created_date(p["created"]))
        if key is None:
            continue
        scope = ("bin",) if p["answer_text"] is None else ("mc", p["parent_id"])
        groups[scope + key].append((v, p))
    return {k: v for k, v in groups.items() if len({x[0] for x in v}) >= 2}


def implies(kind: str, key: tuple, va, vb_):
    """True if the proposition with value va implies the one with vb_ (va != vb_)."""
    if kind == "date":
        return va < vb_
    direction = key[-2]
    return va > vb_ if direction == "up" else va < vb_


# --- as-of checks ----------------------------------------------------------------

class Checker:
    def __init__(self, store: AsOf):
        self.store = store
        self.cache = {}

    def state(self, p: dict, t: int) -> dict | None:
        """As-of-t state of a proposition, or None if it is not open / has no price at t."""
        key = (p["id"], t)
        if key in self.cache:
            return self.cache[key]
        out = None
        db = self.store.db
        if p["created"] < t and p["res_time"] > t and (p["a_created"] is None or p["a_created"] < t):
            n_logged = db.execute("SELECT count(*) FROM resolutions WHERE mid = ? AND resolved <= ?",
                                  (p["mid"], t)).fetchone()[0]
            m = self.store.get_market(p["parent_id"], t)
            if m is not None and m["status"] == "open" and (n_logged == 0 or p["answer_id"]) and m["n_traders"] >= MIN_TRADERS:
                if p["answer_id"]:
                    a = [x for x in m["answers"] if x["id"] == p["answer_id"]]
                    ok = a and a[0]["resolution"] is None and a[0]["prob"] is not None
                    if ok:
                        out = {"price": a[0]["prob"], "question": m["question"], "answer_text": a[0]["text"],
                               "criteria": m["description"] or "", "close": m["close"], "groups": m["groups"],
                               "n_traders": m["n_traders"]}
                elif m["prob"] is not None:
                    out = {"price": m["prob"], "question": m["question"], "answer_text": None,
                           "criteria": m["description"] or "", "close": m["close"], "groups": m["groups"],
                           "n_traders": m["n_traders"]}
        self.cache[key] = out
        return out


def pair_t(a: dict, b: dict) -> int:
    w = min(monday(a["e"]), monday(b["e"]))
    return to_ms((w - dt.timedelta(days=7)).isoformat())


def make_pair(kind: str, a: dict, b: dict, sa: dict, sb: dict, t: int, relation: dict, group: str, rng) -> dict:
    """a, b as found; P/Q order is a coin flip.  relation uses 'A'/'B' -> mapped to P/Q."""
    swap = rng.random() < 0.5
    p, q, sp, sq = (b, a, sb, sa) if swap else (a, b, sa, sb)
    rel = dict(relation)
    if rel.get("implies"):
        src = rel["implies"]  # "A->B" or "B->A"
        first = a if src == "A->B" else b
        rel["implies"] = "P->Q" if first is p else "Q->P"
    return {"type": kind, "group": group, "t": iso(t), "t_ms": t, "p_id": p["id"], "q_id": q["id"],
            "p_price_t": sp["price"], "q_price_t": sq["price"], "relation": rel,
            "p_parent": p["parent_id"], "q_parent": q["parent_id"], "p_src": p["src"], "q_src": q["src"],
            "p_text": prop_row(p, sp, t), "q_text": prop_row(q, sq, t),
            "_labels": (p["label"], q["label"])}


def prop_row(p: dict, s: dict, t: int) -> dict:
    close = s["close"] if s["close"] and t < s["close"] < to_ms("2100-01-01") else p["e"]
    return {"id": p["id"], "parent_id": p["parent_id"], "parent_question": s["question"], "answer_text": s["answer_text"],
            "criteria": s["criteria"], "closing_week": monday(close).isoformat()}


def asof_same_relation(kind, key, a, b, sa, sb, va, vb_) -> bool:
    ka, xa = ladder_key(kind, sa["question"], sa["answer_text"], created_date(a["created"]))
    kb, xb = ladder_key(kind, sb["question"], sb["answer_text"], created_date(b["created"]))
    return ka is not None and ka == kb and xa is not None and xb is not None and xa != xb and \
        implies(kind, ka, xa, xb) == implies(kind, key, va, vb_)


def build_ladders(kind: str, props, chk: Checker, rng, stats) -> list[dict]:
    groups = ladder_groups(props, kind)
    stats[f"{kind}_groups"] = len(groups)
    stats[f"{kind}_groups_bin"] = sum(k[0] == "bin" for k in groups)
    out = []
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    for key in keys:
        rungs = groups[key]
        # one proposition per value (first by id), then all value pairs, adjacent ones first
        by_v = {}
        for v, p in sorted(rungs, key=lambda x: (x[0], x[1]["id"])):
            by_v.setdefault(v, p)
        vals = sorted(by_v)
        cands = [(i, j) for i in range(len(vals)) for j in range(i + 1, len(vals))]
        rng.shuffle(cands)
        cands.sort(key=lambda ij: ij[1] - ij[0])
        got, used = 0, set()
        for i, j in cands:
            if got >= PER_GROUP:
                break
            a, b = by_v[vals[i]], by_v[vals[j]]
            if a["parent_id"] == b["parent_id"] and key[0] == "bin":
                continue
            if (a["id"], b["id"]) in used:
                continue
            t = pair_t(a, b)
            sa, sb = chk.state(a, t), chk.state(b, t)
            if sa is None or sb is None:
                stats[f"{kind}_not_open"] += 1; continue
            if not asof_same_relation(kind, key, a, b, sa, sb, vals[i], vals[j]):
                stats[f"{kind}_asof_text_changed"] += 1; continue
            direction = "A->B" if implies(kind, key, vals[i], vals[j]) else "B->A"
            rel = {"implies": direction, "values": [str(vals[i]), str(vals[j])],
                   "scope": key[0], "template": key[-3] if kind == "threshold" else key[-1]}
            out.append(make_pair(kind, a, b, sa, sb, t, rel, "|".join(map(str, key)), rng))
            used.add((a["id"], b["id"]))
            got += 1
    stats[f"{kind}_available_pairs"] = len(out)
    return out


def build_mc(props, chk: Checker, rng, stats, target: int = TARGET["mc_excl"]) -> list[dict]:
    by_market = collections.defaultdict(list)
    for p in props:
        if p["outcome_type"] == "MULTIPLE_CHOICE" and p["sum_to_one"] == 1:
            by_market[p["parent_id"]].append(p)
    mkts = sorted(by_market)
    rng.shuffle(mkts)
    out = []
    for mk in mkts:
        if len(out) >= target * 2:
            break
        opts = by_market[mk]
        t = to_ms((monday(opts[0]["e"]) - dt.timedelta(days=7)).isoformat())
        st = [(chk.state(p, t), p) for p in opts]
        st = [(s, p) for s, p in st if s is not None]
        if len(st) < 2:
            stats["mc_excl_not_open"] += 1; continue
        st.sort(key=lambda x: (-x[0]["price"], x[1]["id"]))
        (sa, a), (sb, b) = st[0], st[1]
        # The store has no arbitrage fills for sum-to-one markets, so per-option last-bet prices are stale
        # and need not sum to 1.  Record the sum over all open options (incl. "Other") to normalize.
        m = chk.store.get_market(mk, t)
        tot = sum(x["prob"] for x in m["answers"] if x["resolution"] is None and x["prob"] is not None)
        pr = make_pair("mc_excl", a, b, sa, sb, t, {"exclusive": True, "n_options_open": len(st),
                                                    "sum_open_prices": tot}, mk, rng)
        pr["p_price_norm"] = pr["p_price_t"] / tot
        pr["q_price_norm"] = pr["q_price_t"] / tot
        out.append(pr)
    stats["mc_excl_markets"] = len(by_market)
    stats["mc_excl_scanned_pairs"] = len(out)
    return out


def build_control(props, chk: Checker, rng, stats, n: int) -> list[dict]:
    # one proposition per parent (seeded), random partners
    by_parent = collections.defaultdict(list)
    for p in props:
        by_parent[p["parent_id"]].append(p)
    pool = [rng.choice(sorted(v, key=lambda x: x["id"])) for _, v in sorted(by_parent.items())]
    out, tries = [], 0
    while len(out) < n and tries < 60 * n:
        tries += 1
        a, b = rng.sample(pool, 2)
        t = pair_t(a, b)
        sa, sb = chk.state(a, t), chk.state(b, t)
        if sa is None or sb is None:
            continue
        if set(sa["groups"]) & set(sb["groups"]):
            stats["control_shared_group"] += 1; continue
        out.append(make_pair("control", a, b, sa, sb, t, {"none": True}, f"{a['parent_id']}|{b['parent_id']}", rng))
    stats["control_tries"] = tries
    return out


# --- prompts ---------------------------------------------------------------------

def question_block(row: dict) -> str:
    return vb._question_lines(row, row["criteria"].strip(), vb.MAX_CRITERIA_CHARS)


def user_prompt(pair: dict, form: str, context: str | None) -> str:
    parts = [f"Assume today is {pair['t'][:10]}. Use only information available up to that date; do not use knowledge of later events."]
    if context:
        parts.append(context.strip())
    parts.append("Question A\n" + question_block(pair["p_text"]))
    parts.append("Question B\n" + question_block(pair["q_text"]))
    parts.append(f"Statement to assess: {vb.plain_statement(form)}\nWhat is the probability that this statement is true?")
    parts.append(vb.ANSWER_DIRECT)
    return "\n\n".join(parts) + " /no_think"


_STORE = None


def _pack(pair):
    global _STORE
    from training.asof import build_pack, count_tokens, default_store
    if _STORE is None:
        _STORE = default_store()
    text, items = build_pack(pair, pair["t_ms"], "small", _STORE, return_items=True)
    return pair["pair_id"], text, items, count_tokens(text)


def write_prompts(pairs: list[dict], out: Path, procs: int) -> None:
    from training.asof import count_tokens
    import market_context_zeroshot as mcz
    pack_path = out / "packs-structured.jsonl"
    packs = {r["pair_id"]: r for r in jl(pack_path)} if pack_path.exists() else {}
    todo = [{"pair_id": p["pair_id"], "p_id": p["p_id"], "q_id": p["q_id"], "t_ms": p["t_ms"]}
            for p in pairs if p["pair_id"] not in packs]
    if todo:
        with ProcessPoolExecutor(procs) as ex, pack_path.open("a") as f:
            for n, (pid, text, items, ntok) in enumerate(ex.map(_pack, todo, chunksize=4), 1):
                rec = {"pair_id": pid, "size": "small", "tokens": ntok, "items": items, "text": text}
                packs[pid] = rec
                f.write(json.dumps(rec) + "\n")
                if n % 100 == 0:
                    print(f"packs: {n}/{len(todo)}", flush=True)
    for arm in ("Q", "P"):
        rows, by_pair = [], {}
        for p in pairs:
            ctx = packs[p["pair_id"]]["text"] if arm == "P" else None
            for k, form in enumerate(FORM_ORDER):
                msgs = [{"role": "system", "content": vb.BASE_SYSTEM},
                        {"role": "user", "content": user_prompt(p, form, ctx)}]
                if p["pair_id"] not in by_pair:
                    by_pair[p["pair_id"]] = count_tokens(msgs[0]["content"] + msgs[1]["content"])
                rows.append({"query_id": f"{p['pair_id']}-q{k:02d}", "pair_id": p["pair_id"], "form": form, "arm": arm,
                             "stratum": p["type"], "messages": msgs,
                             "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                             "est_prompt_tokens": by_pair[p["pair_id"]] + mcz.TEMPLATE_OVERHEAD})
        with (out / f"prompts-structured-{arm}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        toks = sorted(by_pair.values())
        print(f"{arm}: {len(rows)} prompts, prompt tokens median {toks[len(toks) // 2]} p90 {toks[int(.9 * len(toks))]} "
              f"total {sum(toks) * 12 / 1e6:.2f}M", flush=True)


def _cap(kind: str, ps: list[dict], target: int, stats) -> list[dict]:
    # a proposition may appear in at most 2 pairs of a type (ladder rungs are shared); then cap (seeded)
    seen, keep = collections.Counter(), []
    for p in ps:
        if seen[p["p_id"]] >= MAX_USES or seen[p["q_id"]] >= MAX_USES:
            continue
        seen.update([p["p_id"], p["q_id"]])
        keep.append(p)
    r = random.Random(f"{SEED}-{kind}")
    if len(keep) > target:
        keep = r.sample(keep, target)
    stats[f"{kind}_kept"] = len(keep)
    return sorted(keep, key=lambda x: (x["group"], x["p_id"], x["q_id"]))


def build(out: Path, procs: int, split: str | None = None) -> None:
    """split=None: the structured test set (caps TARGET).  split="train"/"validation": caps SPLIT_TARGET
    per structured type, binary-ladder groups that also occur in the test set dropped, and as many
    control pairs as structured pairs."""
    out.mkdir(parents=True, exist_ok=True)
    store = AsOf()
    props, stats = universe(store, split)
    print("universe", dict(stats), flush=True)
    chk = Checker(store)
    target = TARGET if split is None else {k: SPLIT_TARGET[split] for k in TARGET}
    found = {"threshold": build_ladders("threshold", props, chk, random.Random(SEED + 1), stats),
             "date": build_ladders("date", props, chk, random.Random(SEED + 2), stats),
             "mc_excl": build_mc(props, chk, random.Random(SEED + 3), stats, target["mc_excl"])}
    if split is not None:
        test_groups = {r["group"] for r in jl(OUT / "pairs.jsonl") if r["type"] in ("threshold", "date")}
        for kind in ("threshold", "date"):
            n0 = len(found[kind])
            found[kind] = [p for p in found[kind] if p["group"] not in test_groups]
            stats[f"{kind}_dropped_test_group"] = n0 - len(found[kind])
    pairs = []
    for kind in ("threshold", "date", "mc_excl"):
        pairs += _cap(kind, found[kind], target[kind], stats)
    n_control = TARGET["control"] if split is None else len(pairs)
    ctrl = build_control(props, chk, random.Random(SEED + 4), stats, n_control)
    pairs += _cap("control", ctrl, n_control, stats)
    prefix = "sp" if split is None else {"train": "sptr", "validation": "spva"}[split]
    labels = []
    for k, p in enumerate(pairs):
        p["pair_id"] = f"{prefix}-{p['type']}-{k:04d}"
        yp, yq = p.pop("_labels")
        labels.append({"pair_id": p["pair_id"], "p": yp, "q": yq})
    with (out / "pairs.jsonl").open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    with (out / "labels.jsonl").open("w") as f:
        for l in labels:
            f.write(json.dumps(l) + "\n")
    stats["counts"] = dict(collections.Counter(p["type"] for p in pairs))
    (out / "build-stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str), flush=True)
    write_prompts(pairs, out, procs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "prompts", "project", "run"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--split", choices=["train", "validation"], help="build: training/validation pairs from JEV parents")
    ap.add_argument("--model")
    ap.add_argument("--arm", choices=["Q", "P"])
    ap.add_argument("--pairs", type=int)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=float, default=15.0)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.procs, a.split)
    elif a.cmd == "prompts":
        write_prompts(jl(a.out / "pairs.jsonl"), a.out, a.procs)
    else:
        import market_context_zeroshot as mcz
        mcz.run(a.model, a.arm, "structured", a.out, a.pairs, min(a.workers, 12), a.cap, a.key_secret, a.cmd == "project")


if __name__ == "__main__":
    main()
