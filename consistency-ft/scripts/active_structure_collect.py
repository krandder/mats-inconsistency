"""Active structure elicitation, data collection: a dense query pool over sets of n=12 events (Qwen3-32B, DashScope).

The dense pool is the "oracle" that the active-query methods draw from offline
(scripts/active_structure.py): asking a query = reading its stored t=0 answer.

build   Sets of N=12 binary events from TEST-split parents only (same eligibility and
        as-of rules as scripts/worldmodel_clusters.py: resolved YES/NO, closing week >=
        2025-05-05, all open at t = earliest closing week - 7 days).  A seed market sets
        t; members are the seed's Manifold-group mates (specific groups first, nearest
        closing weeks), filled up to 12 with other open binaries at t (nearest closing
        weeks), so each set mixes related and unrelated events.  A market is used in at
        most 2 sets.
        Queries per set: 12 singles, all 66 pairwise ANDs, all 132 ordered conditionals
        "Suppose A resolves YES. What is the probability that B resolves YES?", 12 screening
        lists ("which of the other questions would change your forecast of A?"), and 12
        random 3-way ANDs (held out; never offered to the query-selection methods).
run     Same runner protocol as worldmodel_clusters.py (t=0, thinking off, one retry,
        served-model check, --max-usd cap at 0.7/2.8 $/M).  Screening queries use
        max_tokens 60 and their own parser.

Usage:
  active_structure_collect.py build --out /home/agent/data/active-structure-20260924 --sets 24
  active_structure_collect.py run   --out /home/agent/data/active-structure-20260924 --workers 8 --max-usd 5
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import worldmodel_clusters as wc  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402
from training.asof import AsOf, to_ms  # noqa: E402

N = 12
SEED = 20260924
LETTERS = "ABCDEFGHIJKL"
SCREEN_ANSWER = ("Reply with exactly one line of this form and nothing else:\n"
                 "Answer: <letters separated by commas, or NONE>")
SCREEN_RETRY = ("Your reply did not end with a valid line of the form 'Answer: ...'. Reply with exactly one line, "
                "'Answer: ' followed by the letters separated by commas, or NONE, and nothing else.")


def screen_messages(s: dict, target: int) -> list[dict]:
    others = [i for i in range(N) if i != target]
    name = {target: "A"} | {i: LETTERS[k + 1] for k, i in enumerate(others)}
    parts = [f"Assume today is {s['as_of']}. Use only information available up to that date; "
             "do not use knowledge of later events.",
             "Question A\n" + vb.question_block(s["events"][target])]
    for i in others:
        parts.append(f"Question {name[i]}\n" + vb.question_block(s["events"][i]))
    parts.append("Consider your forecast for Question A. Which of the other questions (B to L), if you learned "
                 "how it resolved (YES or NO), would change your probability for Question A by 5 percentage "
                 "points or more?")
    parts.append(SCREEN_ANSWER)
    return [{"role": "system", "content": vb.BASE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts) + " /no_think"}], name


def parse_screen(text: str | None, name: dict[int, str]):
    if not text:
        return None, "empty"
    m = re.findall(r"Answer:\s*(.*)", text)
    if not m:
        return None, "no_answer_line"
    body = m[-1].strip().upper()
    if body.startswith("NONE"):
        return [], "ok"
    letters = re.findall(r"\b([B-L])\b", body)
    inv = {v: k for k, v in name.items()}
    return sorted({inv[c] for c in letters}), "ok"


def build(out: Path, n_sets: int) -> None:
    store = AsOf()
    pool = wc.Pool(store)
    D = wc.DATA
    props = [json.loads(l) for l in (D / "test" / "propositions.jsonl").read_text().splitlines()]
    excluded = set(json.loads((D / "excluded-parent-ids.json").read_text()))
    other = set()
    for s in ("train", "validation"):
        other |= {json.loads(l)["parent_id"] for l in (D / s / "propositions.jsonl").read_text().splitlines()}
    parents = sorted({p["parent_id"] for p in props} - excluded)
    assert not set(parents) & other
    jev_week = {p["id"]: p["closing_week"] for p in props}
    db = store.db
    binaries = []
    for cid in parents:
        r = db.execute("SELECT outcome_type, resolution, close FROM markets WHERE id = ?", (cid,)).fetchone()
        if r["outcome_type"] != "BINARY" or r["resolution"] not in ("YES", "NO"):
            continue
        week = jev_week.get(cid) or wc.monday(r["close"])
        if week < wc.MIN_WEEK:
            continue
        binaries.append((cid, week, int(r["resolution"] == "YES")))
    week_of = {c: w for c, w, _ in binaries}
    label_of = {c: y for c, _, y in binaries}
    groups = {c: set(json.loads(db.execute("SELECT group_slugs FROM markets WHERE id = ?", (c,)).fetchone()[0] or "[]"))
              for c in week_of}
    by_group = defaultdict(set)
    for c, gs in groups.items():
        for g in gs:
            by_group[g].add(c)
    rng = random.Random(SEED)
    seeds = sorted(week_of); rng.shuffle(seeds)
    used = Counter(); sets = []

    def open_at(c, t):
        m = pool.market_at(c, t)
        return m is not None and m["status"] == "open" and not pool.logged_before(c, t)

    for seed in seeds:
        if len(sets) >= n_sets:
            break
        if used[seed] >= 1:
            continue
        t_iso = vb.as_of_date(week_of[seed]); t = to_ms(t_iso)
        if not open_at(seed, t):
            continue
        members = [seed]
        gs = sorted(groups[seed], key=lambda g: (g.endswith("-default"), len(by_group[g]), g))
        related = []
        for g in gs:
            cand = sorted((week_of[c], rng.random(), c) for c in by_group[g] - {seed} - set(members)
                          if week_of[c] >= week_of[seed] and used[c] < 2)
            for _, _, c in cand:
                if len(members) >= 8:
                    break
                if c not in members and open_at(c, t):
                    members.append(c); related.append(g)
        if len(members) < 4:  # want real within-set structure
            continue
        n_rel = len(members)
        fill = sorted((week_of[c], rng.random(), c) for c in week_of
                      if c not in members and week_of[c] >= week_of[seed] and used[c] < 2)
        for _, _, c in fill:
            if len(members) >= N:
                break
            if open_at(c, t):
                members.append(c)
        if len(members) < N:
            continue
        for c in members:
            used[c] += 1
        order = sorted(members)
        evs = [wc.event_row(pool.market_at(c, t), None, week_of[c], label_of[c], c) for c in order]
        sets.append({"set_id": f"s{len(sets):02d}", "as_of": t_iso, "seed": seed, "n_related": n_rel,
                     "related_ids": sorted(members[:n_rel]), "groups_used": sorted(set(related)), "events": evs})
        print(f"set {len(sets)}: t={t_iso} related={n_rel} groups={sorted(set(related))[:3]}", flush=True)

    write_queries(sets, out)


def write_queries(sets, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    queries = []
    for s in sets:
        srng = random.Random(f"{SEED}-{s['set_id']}")
        labels = [e["label"] for e in s["events"]]
        qs = [{"kind": "single", "f": wc.var(i)} for i in range(N)]
        qs += [{"kind": "and2", "f": {"op": "and", "args": [wc.var(i), wc.var(j)]}}
               for i, j in itertools.combinations(range(N), 2)]
        qs += [{"kind": "cond", "f": {"op": "cond", "given": wc.var(i), "then": wc.var(j)}}
               for i, j in itertools.permutations(range(N), 2)]
        for tr in srng.sample(list(itertools.combinations(range(N), 3)), 12):
            qs.append({"kind": "and3", "f": {"op": "and", "args": [wc.var(i) for i in tr]}})
        for k, q in enumerate(qs):
            msgs = wc.messages(s, q["f"])
            queries.append({"query_id": f"{s['set_id']}-q{k:03d}", "set_id": s["set_id"], "kind": q["kind"],
                            "formula": q["f"], "formula_str": wc.formula_str(q["f"]),
                            "label": wc.truth(q["f"], labels), "messages": msgs,
                            "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
        for i in range(N):
            msgs, name = screen_messages(s, i)
            queries.append({"query_id": f"{s['set_id']}-scr{i:02d}", "set_id": s["set_id"], "kind": "screen",
                            "target": i, "letters": {str(k): v for k, v in name.items()}, "messages": msgs,
                            "label": None, "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest()})
    with (out / "sets.jsonl").open("w") as f:
        for s in sets:
            f.write(json.dumps(s) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    chars = sum(len(m["content"]) for q in queries for m in q["messages"])
    summ = {"sets": len(sets), "queries": len(queries), "by_kind": Counter(q["kind"] for q in queries),
            "prompt_chars": chars, "est_cost_cap_price": chars / 3.5 * wc.PRICE_IN / 1e6 + len(queries) * 10 * wc.PRICE_OUT / 1e6,
            "n_related": [s["n_related"] for s in sets], "yes_rate": sum(e["label"] for s in sets for e in s["events"]) / (N * len(sets))}
    (out / "build-summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


GLOBAL = Path("/home/agent/data/worldmodel-global-20260924")


def build_aligned(out: Path, n_sets: int) -> None:
    """Sets on the brute-force global run's event set and date (t = 2025-09-15, 400 events):
    seed + its 7 nearest BGE neighbours + 4 random others, each event in at most 2 sets."""
    import numpy as np
    evs = [json.loads(l) for l in (GLOBAL / "events.jsonl").read_text().splitlines()]
    E = np.load(GLOBAL / "embeddings.npy")
    cos = E @ E.T
    t_iso = json.loads((GLOBAL / "build-summary.json").read_text())["t"]
    rng = random.Random(f"{SEED}-aligned")
    used = Counter(); sets = []
    order = list(range(len(evs))); rng.shuffle(order)
    for seed in order:
        if len(sets) >= n_sets:
            break
        if used[seed]:
            continue
        nb = [int(j) for j in np.argsort(-cos[seed]) if j != seed and used[j] < 2][:7]
        rest = [j for j in range(len(evs)) if j != seed and j not in nb and used[j] < 2]
        members = [seed] + nb + rng.sample(rest, N - 8)
        for j in members:
            used[j] += 1
        members = sorted(members)
        rows = [{k: evs[j][k] for k in ("id", "parent_id", "parent_question", "answer_text", "criteria_text",
                                           "closing_week", "label", "groups")} for j in members]
        sets.append({"set_id": f"g{len(sets):02d}", "as_of": t_iso, "seed": evs[seed]["id"], "n_related": 8,
                     "related_ids": sorted(evs[j]["id"] for j in [seed] + nb), "global_index": members,
                     "cosine": [[float(cos[a, b]) for b in members] for a in members], "events": rows})
    write_queries(sets, out)


def run(out: Path, workers: int, max_usd: float, limit: int | None, key_secret: str) -> None:
    import precutoff_client as pc
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    M = wc.MODEL
    pc.MODELS[M] = ("dashscope", M, {"enable_thinking": False, "temperature": 0}, wc.PRICE_IN, wc.PRICE_OUT)
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

    def chat(msgs, max_tokens):
        # DashScope sometimes answers 429 insufficient_quota for a minute or two when the account's
        # token-rate limit is hit (shared key); back off instead of dying.  Never switches model.
        for attempt in range(12):
            try:
                return pc.chat(M, msgs, max_tokens=max_tokens)
            except RuntimeError as e:
                if "insufficient_quota" not in str(e) or attempt == 11:
                    raise
                time.sleep(30 + 15 * attempt)

    def one(row):
        if stop.is_set() or spent[0] >= max_usd:
            stop.set(); return None
        msgs = row["messages"]
        screen = row["kind"] == "screen"
        name = {int(a): b for a, b in row.get("letters", {}).items()}
        parse = (lambda t: parse_screen(t, name)) if screen else vb.parse_probability
        mt = 60 if screen else 20
        first = chat(msgs, mt)
        if first["served_model"] and first["served_model"] != M:
            stop.set(); raise RuntimeError(f"served model {first['served_model']} != {M}")
        p, why = parse(first["text"])
        rec = {"query_id": row["query_id"], "prompt_sha256": row["prompt_sha256"], "raw": first["text"],
               "served_model": first["served_model"], "usage": first["usage"], "errors": first["errors"],
               "finish_reason": first["finish_reason"], "retried": False, "raw_retry": None, "p": p, "reason": why}
        cost = wc.cost_of(first["usage"])
        if p is None:
            second = chat(msgs + [{"role": "assistant", "content": first["text"] or ""},
                                 {"role": "user", "content": SCREEN_RETRY if screen else vb.RETRY_REMINDER}], mt)
            p2, why2 = parse(second["text"])
            cost += wc.cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(workers) as ex, log.open("a") as f:
        futs = [ex.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == 300:
                    proj = spent[0] / (len(done) + n_run[0]) * len(rows)
                    print(f"projection after 300: ${proj:.2f} (cap ${max_usd})", flush=True)
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
    ap.add_argument("cmd", choices=["build", "build-aligned", "run"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.sets)
    elif a.cmd == "build-aligned":
        build_aligned(a.out, a.sets)
    else:
        run(a.out, min(a.workers, 8), a.max_usd, a.limit, a.key_secret)


if __name__ == "__main__":
    main()
