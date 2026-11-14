"""Leak tests for the Manifold as-of store (training/asof.py).

Part 1 builds a tiny synthetic store through the real builder (scripts/build_asof_store.py)
with every edge case the cutoff must handle.  Part 2 runs property checks against the real
store when it exists (skipped otherwise): random (market, t) samples and JEV test pairs.
"""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_asof_store as builder  # noqa: E402
from training import asof  # noqa: E402

D = 86_400_000
T0 = 1_700_000_000_000  # 2023-11-14, after edit recording started
T = {k: T0 + k * D for k in range(12)}
REAL_STORE = asof.DEFAULT_STORE
PAIRS = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919/test")


def doc(text):
    return {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]}


def iso(ms):
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat()


def contract(cid, question, desc, created, **extra):
    data = {"id": cid, "slug": cid.lower(), "question": question, "description": doc(desc), "createdTime": created,
            "creatorId": "u1", "creatorUsername": "alice", "creatorName": "Alice", "visibility": "public",
            "mechanism": "cpmm-1", "outcomeType": "BINARY", "initialProbability": 0.5, **extra}
    return {"id": cid, "data": data}


def edit(eid, cid, when, keys, **pre):
    return {"id": eid, "contract_id": cid, "created_time": iso(when), "updated_keys": keys, "data": pre}


def bet(bid, cid, when, p0, p1, amount=10, answer=None, **extra):
    b = {"id": bid, "contractId": cid, "createdTime": when, "probBefore": p0, "probAfter": p1, "amount": amount,
         "shares": amount * 2, "outcome": "YES", "userId": "u2", **extra}
    if answer:
        b["answerId"] = answer
    return b


def comment(cid_, contract_id, when, text, **extra):
    return {"contract_id": contract_id, "comment_id": cid_, "created_time": iso(when),
            "data": {"id": cid_, "userId": "u3", "userUsername": "bob", "userName": "Bob", "createdTime": when,
                     "content": doc(text), **extra}}


def write_pages(root: Path, tables: dict[str, list]):
    for name in ("contracts", "answers", "api_bets", "contract_comments", "contract_edits", "contract_comment_edits",
                 "groups", "group_contracts"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "0000000.json.gz", "wt") as f:
            json.dump(tables.get(name, []), f)


def build_synthetic(root: Path) -> Path:
    tables = {
        "contracts": [
            # M1: question/description edited at T2; resolved YES at T8 with the close rewritten to T8
            contract("M1", "Kangaroo question", "new description", T[0], closeTime=T[8], resolution="YES",
                     resolutionTime=T[8], isResolved=True, groupSlugs=["zoo"]),
            # M2: resolved NO at T3, unresolved at T4, resolved YES at T6
            contract("M2", "Wombat question", "d2", T[0], closeTime=T[10], resolution="YES", resolutionTime=T[6],
                     isResolved=True, groupSlugs=["zoo"]),
            # M3: multiple choice, answer B added at T3, answer A text edited at T2, A resolved at T5
            contract("M3", "Which marsupial?", "d3", T[0], outcomeType="MULTIPLE_CHOICE", mechanism="cpmm-multi-1",
                     closeTime=T[10], shouldAnswersSumToOne=True),
            # M4: created at T3 (after early cutoffs), unique word
            contract("M4", "Zebra question", "stripes", T[3], closeTime=T[10]),
        ],
        "answers": [
            {"id": "ANSWER_AAAA", "contract_id": "M3", "text": "beta", "created_time": iso(T[0]), "index": 0,
             "resolution": "YES", "resolution_time": iso(T[5]), "resolution_probability": None},
            {"id": "ANSWER_BBBB", "contract_id": "M3", "text": "gamma", "created_time": iso(T[3]), "index": 1},
        ],
        "contract_edits": [
            edit(1, "M1", T[2], ["question", "description"], question="Platypus question",
                 description=doc("old description"), closeTime=T[10], groupSlugs=["zoo"]),
            edit(2, "M3", T[2], ["answers"], question="Which marsupial?", description=doc("d3"),
                 answers=[{"id": "ANSWER_AAAA", "text": "alpha"}]),
            edit(3, "M2", T[4], ["isResolved", "resolution", "resolutionTime"], question="Wombat question",
                 description=doc("d2"), isResolved=True, resolution="NO", resolutionTime=T[3], closeTime=T[10]),
        ],
        "api_bets": [
            bet("b1", "M1", T[1], 0.5, 0.6),
            bet("b2", "M1", T[3], 0.6, 0.7),
            # limit order: 5 filled at creation, 10 more at T4, cancelled at T5
            bet("L1", "M1", T[1] + 1000, 0.6, 0.6, amount=15, limitProb=0.55, orderAmount=50, isCancelled=True,
                updatedTime=T[5], fills=[{"timestamp": T[1] + 1000, "amount": 5, "shares": 9},
                                         {"timestamp": T[4], "amount": 10, "shares": 18}]),
            bet("b3", "M3", T[1], 0.5, 0.3, answer="ANSWER_AAAA"),
            bet("b4", "M3", T[4], 0.5, 0.8, answer="ANSWER_BBBB"),
        ],
        "contract_comments": [
            comment("C1", "M1", T[1], "edited text", editedTime=T[3]),
            comment("C2", "M1", T[1] + 1, "edited later, no record", editedTime=T[3]),
            comment("C3", "M1", T[3], "made after early t"),
            comment("C4", "M1", T[1] + 2, "hidden later", hidden=True, hiddenTime=T[2] + D // 2),
            comment("C5", "M1", T[1] + 3, "deleted", deleted=True),
        ],
        "contract_comment_edits": [
            {"id": 1, "contract_id": "M1", "comment_id": "C1", "created_time": iso(T[3]),
             "data": {"content": doc("original text")}},
        ],
        "groups": [{"id": "g1", "slug": "zoo", "name": "Zoo", "created_time": iso(T[0] - D)}],
        "group_contracts": [{"group_id": "g1", "contract_id": "M1"}, {"group_id": "g1", "contract_id": "M2"}],
    }
    write_pages(root / "pages", tables)
    out = root / "asof.sqlite"
    builder.build(out, workers=2, pages=root / "pages")
    return out


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return asof.AsOf(build_synthetic(tmp_path_factory.mktemp("asof")))


def test_question_and_description_as_of_t(store):
    early = store.get_market("M1", T[1] + D // 2)
    assert early["question"] == "Platypus question" and early["description"] == "old description"
    late = store.get_market("M1", T[3])
    assert late["question"] == "Kangaroo question" and late["description"] == "new description"
    # an edit at exactly t is not visible (strictly before t)
    assert store.get_market("M1", T[2])["question"] == "Platypus question"


def test_market_created_at_or_after_t_is_invisible(store):
    assert store.get_market("M4", T[3]) is None
    assert store.get_market("M4", T[3] + 1) is not None
    assert store.search_markets("zebra", T[3]) == []
    assert [m["id"] for m in store.search_markets("zebra", T[4])] == ["M4"]


def test_search_uses_text_as_of_t(store):
    assert [m["id"] for m in store.search_markets("platypus", T[1])] == ["M1"]
    assert store.search_markets("platypus", T[5]) == []
    assert store.search_markets("kangaroo", T[1]) == []
    assert [m["id"] for m in store.search_markets("kangaroo", T[5])] == ["M1"]
    # answer texts: 'alpha' before the answer edit, 'gamma' only after answer B exists
    assert [m["id"] for m in store.search_markets("alpha", T[1])] == ["M3"]
    assert store.search_markets("alpha", T[5]) == []
    assert store.search_markets("gamma", T[2]) == []


def test_resolution_only_if_resolved_before_t(store):
    m = store.get_market("M1", T[7])
    assert m["status"] == "open" and m["resolution"] is None and m["resolution_time"] is None
    # close was rewritten to the (later) resolution time: withheld
    assert m["close"] is None
    assert store.get_market("M1", T[8])["resolution"] is None  # resolved exactly at t: not yet
    r = store.get_market("M1", T[9])
    assert r["status"] == "resolved" and r["resolution"] == "YES"


def test_unresolved_market_history(store):
    assert store.get_market("M2", T[2])["status"] == "open"
    m = store.get_market("M2", T[3] + 1)
    assert (m["status"], m["resolution"]) == ("resolved", "NO")
    assert store.get_market("M2", T[5])["status"] == "open"
    assert store.get_market("M2", T[7])["resolution"] == "YES"


def test_prices_and_trades(store):
    t = T[2]
    assert store.get_market("M1", t)["prob"] == 0.6
    tr = store.trades("M1", t)
    assert {x["id"] for x in tr} == {"b1", "L1"}
    lim = [x for x in tr if x["id"] == "L1"][0]
    assert lim["amount"] == 5 and lim["cancelled"] is False
    lim = [x for x in store.trades("M1", T[4] + 1) if x["id"] == "L1"][0]
    assert lim["amount"] == 15 and lim["cancelled"] is False
    assert [x for x in store.trades("M1", T[6]) if x["id"] == "L1"][0]["cancelled"] is True
    hist = store.price_history("M1", t, points=5)
    assert all(h["time"] < t for h in hist) and hist[-1]["prob"] == 0.6 and hist[0]["prob"] == 0.5


def test_answers_as_of_t(store):
    m = store.get_market("M3", T[2] - 1)
    assert [(a["id"], a["text"], a["prob"], a["resolution"]) for a in m["answers"]] == [
        ("ANSWER_AAAA", "alpha", 0.3, None)]
    m = store.get_market("M3", T[6])
    assert [(a["text"], a["resolution"]) for a in m["answers"]] == [("beta", "YES"), ("gamma", None)]
    assert store.price_history("M3:ANSWER_BBBB", T[2]) == [] or all(
        h["prob"] is None for h in store.price_history("M3:ANSWER_BBBB", T[2]))


def test_comments_as_of_t(store):
    early = {c["id"]: c["text"] for c in store.comments("M1", T[2])}
    assert early == {"C1": "original text", "C4": "hidden later"}
    late = {c["id"]: c["text"] for c in store.comments("M1", T[4])}
    assert late == {"C1": "edited text", "C2": "edited later, no record", "C3": "made after early t"}


def test_related_and_pack_are_deterministic_and_pre_t(store):
    rel = store.related_markets("M1", T[5], k=5)
    assert "M1" not in [r["id"] for r in rel] and all(store.get_market(r["id"], T[5]) for r in rel)
    pair = {"p_id": "M1", "q_id": "M3:ANSWER_AAAA"}
    a = asof.build_pack(pair, T[2], "small", store=store)
    b = asof.build_pack(pair, T[2], "small", store=store)
    assert a == b
    assert "Platypus question" in a and "Kangaroo" not in a and "made after early t" not in a
    assert "resolved" not in a.split("## Related")[0].replace("not yet resolved", "")


# --- real store ------------------------------------------------------------------

real = pytest.mark.skipif(not REAL_STORE.exists(), reason="as-of store not built")


@pytest.fixture(scope="module")
def big():
    return asof.AsOf(REAL_STORE)


def version_text_independent(db, mid, t):
    """Question/description at t straight from the raw edits table (independent of market_versions)."""
    row = db.execute("SELECT question, description FROM markets WHERE mid = ?", (mid,)).fetchone()
    q, d = row
    edits = db.execute("SELECT time, question, description FROM edits WHERE mid = ? ORDER BY time, eid",
                       (mid,)).fetchall()
    prev = None
    resolved = []
    for when, eq, ed in edits:
        ed = prev if ed == "\x00same" else ed
        prev = ed if ed is not None else prev
        resolved.append((when, eq, ed))
    for when, eq, ed in resolved:
        if when >= t:
            return (eq if eq is not None else q), (ed if ed is not None else d)
    return q, d


@real
def test_random_market_samples_never_leak(big):
    db = sqlite3.connect(f"file:{REAL_STORE}?mode=ro", uri=True)
    rng = random.Random(20260923)
    n_mid = db.execute("SELECT max(mid) FROM markets").fetchone()[0]
    checked = 0
    while checked < 150:
        mid = rng.randint(1, n_mid)
        row = db.execute("SELECT id, created, resolution_time FROM markets WHERE mid = ?", (mid,)).fetchone()
        if row is None:
            continue
        cid, created, res_t = row
        end = res_t + 30 * D if res_t else created + 400 * D
        t = rng.randint(created - 5 * D, max(created + 1, end))
        m = big.get_market(cid, t)
        if created >= t:
            assert m is None
            continue
        checked += 1
        assert m["created"] < t
        if m["resolution"] is not None:
            assert m["resolution_time"] < t
        if res_t is not None and res_t >= t:
            assert m["resolution"] is None or m["resolution_time"] < t  # only an earlier, later-undone resolution
        for a in m.get("answers") or []:
            assert a["created"] < t
            assert a["resolution"] is None or a["resolution_time"] < t
        q, d = version_text_independent(db, mid, t)
        assert (m["question"], m["description"]) == (q, d or "")
        for x in big.trades(cid, t, limit=200):
            assert x["time"] < t
        for c in big.comments(cid, t, limit=100):
            assert c["time"] < t
            edited = db.execute("SELECT edited FROM comments WHERE id = ?", (c["id"],)).fetchone()[0]
            if edited is not None and edited >= t:
                pre = db.execute("SELECT text FROM comment_edits WHERE comment_id = ? AND time >= ? "
                                 "ORDER BY time, id LIMIT 1", (c["id"], t)).fetchone()
                assert pre is not None and pre[0] == c["text"]
        for h in big.price_history(cid, t, points=8):
            assert h["time"] < t
        for r in big.search_markets(m["question"], t, k=10):
            assert db.execute("SELECT created FROM markets WHERE id = ?", (r["id"],)).fetchone()[0] < t


@real
@pytest.mark.skipif(not PAIRS.exists(), reason="JEV test pairs missing")
def test_pack_for_test_pairs_is_pre_t_and_deterministic(big):
    sys.path.insert(0, str(ROOT / "scripts"))
    import audit_asof
    report = audit_asof.audit(big, n_pairs=25, seed=7, sizes=("small",))
    assert report["leaks"] == 0, report["leak_examples"]
    assert report["nondeterministic"] == 0
