"""World-model clusters: n related events, many logical queries, one call per query (Qwen3-32B, DashScope).

Question: does "adjust to be consistent" (coherence projection) gain more when the
model's answers cover a larger world model (clusters of n = 2, 3, 5, 8 events)
instead of pairs?

build   Pick clusters from TEST-split parents only (jev-disjoint-v1-filtered-20260919;
        the 178 excluded parents are removed).  Every event resolved YES or NO (N/A etc.
        dropped), closes in the week of 2025-05-05 or later (the test-set eligibility,
        after the Qwen3 training cutoff), and is open at a common t = (earliest closing
        week - 7 days) 00:00 UTC: market and option created before t, not resolved at or
        before t, no resolution logged at or before t (training/asof.py, cut off in SQL).
        Two cluster types:
          mc     n options of one multiple-choice market.  None of the test multiple-choice
                 parents is sum-to-one or says "only/exactly/at most one" in its text, so
                 no exclusivity structure is imposed (recorded per cluster).
          group  n binary markets sharing a Manifold group (non "-default" groups first),
                 the earliest-closing one sets t, the rest are the nearest closing weeks.
        Queries per cluster (each asked separately, formula recorded): all singles; all
        pairwise ANDs (random 12 for n=8); ~n pairwise conditionals "Suppose A resolves
        YES. What is the probability that B resolves YES?"; 3-way ANDs/ORs (n=2: the
        pairwise OR); 2 negations.
        Prompt: v6_direct (training/verbalized_baseline.py): system BASE_SYSTEM; "Assume
        today is t"; for each question in the query: "Question X" + question_block (text
        and criteria AS OF t from the store, creator edits stripped, 1,500-char cap);
        statement; ANSWER_DIRECT; " /no_think".  No market data.
run     DashScope qwen3-32b, enable_thinking false, temperature 0, max_tokens 20, one
        retry with RETRY_REMINDER, missing recorded as None.  Resumable log; stops before
        --max-usd (list price used for the cap is the conservative 0.7/2.8 $/M).  Never
        substitutes another model: a served-model mismatch stops the run.

Usage:
  worldmodel_clusters.py build --out /home/agent/data/worldmodel-clusters-20260924
  worldmodel_clusters.py run   --out /home/agent/data/worldmodel-clusters-20260924 --workers 8 --max-usd 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import verbalized_baseline as vb  # noqa: E402
from training.asof import AsOf, to_ms  # noqa: E402

DATA = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919")
SEED = 20260924
SIZES = (2, 3, 5, 8)
TARGET = 50
MIN_WEEK = "2025-05-05"  # earliest closing week in the existing test split
MODEL = "qwen3-32b"
PRICE_IN, PRICE_OUT = 0.7, 2.8  # conservative $/M used for the cap (third-party list: 0.16/0.64)
LETTERS = "ABCDEFGH"


def monday(ms: int) -> str:
    d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()
    return (d - dt.timedelta(days=d.weekday())).isoformat()


# --- formulas ----------------------------------------------------------------
# {"op": "var", "i": k} | {"op": "not", "a": f} | {"op": "and"/"or", "args": [f, ...]} |
# {"op": "cond", "given": f, "then": g}

def var(i): return {"op": "var", "i": i}


def formula_str(f) -> str:
    op = f["op"]
    if op == "var":
        return f"E{f['i']}"
    if op == "not":
        return f"NOT {formula_str(f['a'])}"
    if op in ("and", "or"):
        return "(" + f" {op.upper()} ".join(formula_str(a) for a in f["args"]) + ")"
    return f"{formula_str(f['then'])} | {formula_str(f['given'])}"


def events_of(f) -> list[int]:
    out = []

    def walk(g):
        if g["op"] == "var":
            if g["i"] not in out:
                out.append(g["i"])
        elif g["op"] == "not":
            walk(g["a"])
        elif g["op"] == "cond":
            walk(g["given"]); walk(g["then"])
        else:
            for a in g["args"]:
                walk(a)
    walk(f)
    return out


def truth(f, labels: list[int]) -> int | None:
    """Truth value under the resolved labels; a conditional is None when its condition is false."""
    op = f["op"]
    if op == "var":
        return labels[f["i"]]
    if op == "not":
        return 1 - truth(f["a"], labels)
    if op == "and":
        return int(all(truth(a, labels) for a in f["args"]))
    if op == "or":
        return int(any(truth(a, labels) for a in f["args"]))
    return truth(f["then"], labels) if truth(f["given"], labels) else None


def cluster_queries(n: int, rng: random.Random) -> list[dict]:
    q = [{"kind": "single", "f": var(i)} for i in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    if n == 8:
        pairs = sorted(rng.sample(pairs, 12))
    q += [{"kind": "and2", "f": {"op": "and", "args": [var(i), var(j)]}} for i, j in pairs]
    ordered = list(itertools.permutations(range(n), 2))
    for i, j in rng.sample(ordered, min(n, len(ordered))):
        q.append({"kind": "cond", "f": {"op": "cond", "given": var(i), "then": var(j)}})
    if n == 2:
        q.append({"kind": "or", "f": {"op": "or", "args": [var(0), var(1)]}})
    else:
        triples = list(itertools.combinations(range(n), 3))
        k = 1 if n == 3 else 2
        for op in ("and", "or"):
            for t in rng.sample(triples, k):
                q.append({"kind": op + "3", "f": {"op": op, "args": [var(i) for i in t]}})
    for i in rng.sample(range(n), 2):
        q.append({"kind": "neg", "f": {"op": "not", "a": var(i)}})
    return q


# --- prompts -------------------------------------------------------------------

def literal(f, name: dict[int, str]) -> str:
    if f["op"] == "var":
        return f"Question {name[f['i']]} resolves YES"
    if f["op"] == "not" and f["a"]["op"] == "var":
        return f"Question {name[f['a']['i']]} resolves NO"
    raise ValueError(f)


def statement(f, name) -> str:
    op = f["op"]
    if op in ("var", "not"):
        return f"Statement to assess: {literal(f, name)}.\nWhat is the probability that this statement is true?"
    if op in ("and", "or"):
        items = "; ".join(f"({k + 1}) {literal(a, name)}" for k, a in enumerate(f["args"]))
        head = ("Both of the following are true" if len(f["args"]) == 2 else "All of the following are true") \
            if op == "and" else "At least one of the following is true"
        return f"Statement to assess: {head}: {items}.\nWhat is the probability that this statement is true?"
    return f"Suppose {literal(f['given'], name)}. What is the probability that {literal(f['then'], name)}?"


def messages(cluster: dict, f) -> list[dict]:
    evs = events_of(f)
    name = {i: LETTERS[k] for k, i in enumerate(evs)}
    parts = [f"Assume today is {cluster['as_of']}. Use only information available up to that date; "
             "do not use knowledge of later events."]
    for i in evs:
        parts.append(f"Question {name[i]}\n" + vb.question_block(cluster["events"][i]))
    parts.append(statement(f, name))
    parts.append(vb.ANSWER_DIRECT)
    return [{"role": "system", "content": vb.BASE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts) + " /no_think"}]


# --- build ---------------------------------------------------------------------

class Pool:
    def __init__(self, store: AsOf):
        self.s = store
        self.cache: dict = {}

    def market_at(self, cid: str, t_ms: int):
        key = (cid, t_ms)
        if key not in self.cache:
            self.cache[key] = self.s.get_market(cid, t_ms)
        return self.cache[key]

    def logged_before(self, cid: str, t_ms: int) -> bool:
        row = self.s.db.execute("SELECT count(*) FROM resolutions r JOIN markets m ON m.mid = r.mid "
                                "WHERE m.id = ? AND r.resolved <= ?", (cid, t_ms)).fetchone()
        return row[0] > 0


def event_row(m_at: dict, answer: dict | None, week: str, label: int, cid: str) -> dict:
    return {"id": cid + (f":{answer['id']}" if answer else ""), "parent_id": cid,
            "parent_question": m_at["question"], "answer_text": answer["text"] if answer else None,
            "criteria_text": m_at["description"] or "", "closing_week": week, "label": label,
            "groups": m_at["groups"]}


def build(out: Path) -> None:
    store = AsOf()
    pool = Pool(store)
    props = [json.loads(l) for l in (DATA / "test" / "propositions.jsonl").read_text().splitlines()]
    excluded = set(json.loads((DATA / "excluded-parent-ids.json").read_text()))
    other = set()
    for s in ("train", "validation"):
        other |= {json.loads(l)["parent_id"] for l in (DATA / s / "propositions.jsonl").read_text().splitlines()}
    parents = sorted({p["parent_id"] for p in props} - excluded)
    assert not set(parents) & other, "test parent also in train/validation"
    jev_week = {p["id"]: p["closing_week"] for p in props}
    rng = random.Random(SEED)
    db = store.db
    final = {}
    for cid in parents:
        r = db.execute("SELECT mid, outcome_type, resolution, close, sum_to_one FROM markets WHERE id = ?", (cid,)).fetchone()
        final[cid] = dict(r)
    clusters = []
    stats = Counter()

    # (1) multiple-choice options
    mc = [c for c in parents if final[c]["outcome_type"] == "MULTIPLE_CHOICE"]
    mc_ready = {}
    for cid in mc:
        f = final[cid]
        week = monday(f["close"])
        if week < MIN_WEEK:
            stats["mc_early_week"] += 1; continue
        t = to_ms(vb.as_of_date(week))
        m_at = pool.market_at(cid, t)
        if m_at is None or m_at["status"] != "open" or pool.logged_before(cid, t):
            stats["mc_not_open"] += 1; continue
        labels = {a[0]: a[1] for a in db.execute("SELECT id, resolution FROM answers WHERE mid = ?", (f["mid"],))}
        opts = []
        for a in m_at["answers"]:
            if a["is_other"] or a["resolution"] is not None or labels.get(a["id"]) not in ("YES", "NO"):
                continue
            w = jev_week.get(f"{cid}:{a['id']}")
            if w is not None and w != week:
                stats["mc_week_mismatch"] += 1
            opts.append(event_row(m_at, a, week, int(labels[a["id"]] == "YES"), cid))
        if f["sum_to_one"]:
            stats["mc_sum_to_one"] += 1
        mc_ready[cid] = (vb.as_of_date(week), opts)
    for n in SIZES:
        cands = [c for c, (_, o) in sorted(mc_ready.items()) if len(o) >= n]
        rng.shuffle(cands)
        for cid in cands[:TARGET]:
            as_of, opts = mc_ready[cid]
            evs = sorted(rng.sample(opts, n), key=lambda e: e["id"])
            clusters.append({"type": "mc", "n": n, "parent": cid, "as_of": as_of, "structure": "none", "events": evs})

    # (2) binary markets sharing a group
    binaries = []
    for cid in parents:
        f = final[cid]
        if f["outcome_type"] != "BINARY" or f["resolution"] not in ("YES", "NO"):
            stats["bin_not_yes_no" if f["outcome_type"] == "BINARY" else "skip"] += 1; continue
        week = jev_week.get(cid) or monday(f["close"])
        if week < MIN_WEEK:
            stats["bin_early_week"] += 1; continue
        binaries.append((cid, week, int(f["resolution"] == "YES")))
    week_of = {c: w for c, w, _ in binaries}
    label_of = {c: y for c, _, y in binaries}
    groups_final = {}
    for cid, _, _ in binaries:
        gs = json.loads(db.execute("SELECT group_slugs FROM markets WHERE id = ?", (cid,)).fetchone()[0] or "[]")
        groups_final[cid] = set(gs)
    by_group = defaultdict(set)
    for c, gs in groups_final.items():
        for g in gs:
            by_group[g].add(c)

    for n in SIZES:
        used = Counter(); made = set(); count = 0
        seeds = sorted(week_of); rng.shuffle(seeds)
        for seed in seeds:
            if count >= TARGET:
                break
            if used[seed] >= 2:
                continue
            t_iso = vb.as_of_date(week_of[seed]); t = to_ms(t_iso)
            s_at = pool.market_at(seed, t)
            if s_at is None or s_at["status"] != "open" or pool.logged_before(seed, t):
                continue
            # specific groups first, then "-default" topics; smallest group first
            gs = sorted(groups_final[seed], key=lambda g: (g.endswith("-default"), len(by_group[g]), g))
            chosen = None
            for g in gs:
                cand = []
                for c in sorted(by_group[g] - {seed}):
                    if week_of[c] < week_of[seed] or used[c] >= 2:
                        continue
                    m_at = pool.market_at(c, t)
                    if m_at is None or m_at["status"] != "open" or pool.logged_before(c, t):
                        continue
                    cand.append((week_of[c], rng.random(), c))
                if len(cand) >= n - 1:
                    cand.sort()
                    members = [seed] + [c for _, _, c in cand[:n - 1]]
                    if frozenset(members) not in made:
                        chosen = (g, members); break
            if chosen is None:
                continue
            g, members = chosen
            made.add(frozenset(members)); count += 1
            for c in members:
                used[c] += 1
            evs = [event_row(pool.market_at(c, t), None, week_of[c], label_of[c], c) for c in sorted(members)]
            clusters.append({"type": "group", "n": n, "group": g, "as_of": t_iso, "structure": "none", "events": evs})

    out.mkdir(parents=True, exist_ok=True)
    queries = []
    for k, c in enumerate(clusters):
        c["cluster_id"] = f"{c['type']}-n{c['n']}-{k:04d}"
        crng = random.Random(f"{SEED}-{c['cluster_id']}")
        labels = [e["label"] for e in c["events"]]
        for j, q in enumerate(cluster_queries(c["n"], crng)):
            msgs = messages(c, q["f"])
            queries.append({"query_id": f"{c['cluster_id']}-q{j:02d}", "cluster_id": c["cluster_id"], "type": c["type"],
                            "n": c["n"], "kind": q["kind"], "formula": q["f"], "formula_str": formula_str(q["f"]),
                            "label": truth(q["f"], labels), "messages": msgs,
                            "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
    with (out / "clusters.jsonl").open("w") as f:
        for c in clusters:
            f.write(json.dumps(c) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    chars = sum(len(m["content"]) for q in queries for m in q["messages"])
    summary = {"clusters": Counter(f"{c['type']} n={c['n']}" for c in clusters), "queries": len(queries),
               "queries_by": Counter(f"{q['type']} n={q['n']}" for q in queries), "stats": stats,
               "prompt_chars": chars, "est_prompt_tokens": chars / 3.5,
               "est_cost_cap_price": chars / 3.5 * PRICE_IN / 1e6 + len(queries) * 8 * PRICE_OUT / 1e6,
               "mc_parents_used": len({c["parent"] for c in clusters if c["type"] == "mc"}),
               "group_markets_used": len({e["id"] for c in clusters if c["type"] == "group" for e in c["events"]})}
    (out / "build-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# --- run -----------------------------------------------------------------------

def cost_of(usage: dict) -> float:
    return usage.get("prompt_tokens", 0) * PRICE_IN / 1e6 + usage.get("completion_tokens", 0) * PRICE_OUT / 1e6


def run(out: Path, workers: int, max_usd: float, limit: int | None, key_secret: str) -> None:
    import precutoff_client as pc
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False, "temperature": 0}, PRICE_IN, PRICE_OUT)
    log = out / "log.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[r["query_id"]] = r
    spent = [sum(r["cost"] for r in done.values())]
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"{len(rows)} calls, {len(done)} done, {len(todo)} to run; spent ${spent[0]:.3f}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    n_run = [0]

    def one(row):
        if stop.is_set() or spent[0] >= max_usd:
            stop.set(); return None
        msgs = row["messages"]
        first = pc.chat(MODEL, msgs, max_tokens=20)
        if first["served_model"] and first["served_model"] != MODEL:
            stop.set(); raise RuntimeError(f"served model {first['served_model']} != {MODEL}")
        p, why = vb.parse_probability(first["text"])
        rec = {"query_id": row["query_id"], "prompt_sha256": row["prompt_sha256"], "raw": first["text"],
               "served_model": first["served_model"], "usage": first["usage"], "errors": first["errors"],
               "finish_reason": first["finish_reason"], "parsed_first": p, "reason_first": why, "retried": False,
               "raw_retry": None, "p": p, "reason": why}
        cost = cost_of(first["usage"])
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=20)
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(workers) as pool, log.open("a") as f:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == 200:
                    proj = spent[0] / (len(done) + n_run[0]) * len(rows)
                    print(f"projection after 200: ${proj:.2f} (cap ${max_usd})", flush=True)
                    if proj > max_usd:
                        print("STOP: projection exceeds cap", flush=True); stop.set()
                if n_run[0] % 500 == 0:
                    print(f"{len(done) + n_run[0]}/{len(rows)} spent ${spent[0]:.3f}", flush=True)
    recs = {json.loads(l)["query_id"]: json.loads(l) for l in log.read_text().splitlines()}
    print(f"{'STOPPED ' if stop.is_set() else ''}logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs.values())}, missing {sum(r['p'] is None for r in recs.values())}, "
          f"spend ${sum(r['cost'] for r in recs.values()):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out)
    else:
        run(a.out, min(a.workers, 8), a.max_usd, a.limit, a.key_secret)


if __name__ == "__main__":
    main()
