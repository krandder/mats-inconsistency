"""Bias-balanced triangle questions for Qwen3-32B (Kelvin approved 27 Sep 2026).

Why: the weekly world-model run (scripts/weekly_worldmodel.py) asks only "A and B" and "A and B and C";
the 32B states those ~+0.07 above the product of its singles, so fits that match the joints drag the
singles up and never beat independence on joints.  Here every triangle is asked in a sign-balanced way.

Per triangle (three linked events of the test-split Luna graph, same weeks and events as the weekly run),
a uniformly random sign pattern s over (A, B, C); literal L_k = "Question k resolves YES" if s_k = 1,
else "... resolves NO".  16 statements, one per call:
  chain (10)   T = L_A and L_B and L_C; sib_k = T with literal k flipped (3); pair_k = T without
               literal k (3); single_k = L_k (3)
  count (4)    same literals, counting wording, no and/or: "none", "exactly one", "exactly two",
               "all three" of these statements are true   ("all three" is T in counting words)
  mirror (2)   the OR form of not-T ("At least one of: flipped literals") and of not pair_k for one
               random k
Prompts: scripts/worldmodel_clusters.messages (v6_direct, question-only, as-of line); the counting forms
use the same parts with a counting statement.  Singles use the weekly run's wording ("Question A ...");
pairs take the weekly run's and2 order of that link and every triangle-level form takes the weekly
and3 order of that triangle, so every all-positive single / pair / triple is the identical prompt the
weekly run already answered (reused by sha256, cost 0).  Duplicates inside this run (the same signed
single or pair in two triangles) are asked once.

Weeks: 2025-05-12, 06-23, 08-04, 09-15, 10-27, 12-08; up to 150 triangles per week, seeded, stratified
by the sorted tuple of Luna relation types of the triangle's three links (largest-remainder allocation).

Usage: balanced_triangles.py build|run  [--out DIR] [--per-week 150] [--workers 8] [--max-usd 9.5]
Analysis: scripts/balanced_triangles_analyze.py
"""

from __future__ import annotations

import argparse
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
import worldmodel_clusters as wc  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

WEEKLY = Path("/home/agent/data/weekly-worldmodel-20260927")
OUT = Path("/home/agent/data/balanced-triangles-20260927")
WEEKS = ["2025-05-12", "2025-06-23", "2025-08-04", "2025-09-15", "2025-10-27", "2025-12-08"]
SEED = 20260927
MODEL = "qwen3-32b"
CHARS_PER_TOKEN = 3.85
OUT_TOKENS = 7
COUNT_WORDS = {0: "none of them is true", 1: "exactly one of them is true", 2: "exactly two of them are true",
               3: "all three of them are true"}
FORMS = (["T"] + [f"sib{k}" for k in range(3)] + [f"pair{k}" for k in range(3)] + [f"single{k}" for k in range(3)]
         + [f"count{c}" for c in range(4)] + ["mirT", "mirP"])


def ftype(form: str) -> str:
    return form.rstrip("0123456789") if not form.startswith("count") else form


def lit(i, positive):
    return wc.var(i) if positive else {"op": "not", "a": wc.var(i)}


def atoms(form: str, s, mir_k: int) -> list[int]:
    """Outcome indices x = (xA, xB, xC) as 4*xA + 2*xB + xC (positions in the triangle's and3 order)
    where the form is true.  Literal k is true iff x_k == s_k."""
    out = []
    for x in itertools.product((0, 1), repeat=3):
        t = [int(x[k] == s[k]) for k in range(3)]
        if form == "T":
            ok = all(t)
        elif form.startswith("sib"):
            k = int(form[3:]); ok = all(t[j] if j != k else not t[j] for j in range(3))
        elif form.startswith("pair"):
            k = int(form[4:]); ok = all(t[j] for j in range(3) if j != k)
        elif form.startswith("single"):
            ok = bool(t[int(form[6:])])
        elif form.startswith("count"):
            ok = sum(t) == int(form[5:])
        elif form == "mirT":
            ok = not all(t)
        elif form == "mirP":
            ok = not all(t[j] for j in range(3) if j != mir_k)
        else:
            raise ValueError(form)
        if ok:
            out.append(4 * x[0] + 2 * x[1] + x[2])
    return out


def messages_for(week, events, order, pair_order, s, form, mir_k):
    cluster = {"as_of": week, "events": events}
    L = [lit(order[k], s[k]) for k in range(3)]
    flip = [lit(order[k], not s[k]) for k in range(3)]

    def pair_args(k):  # the two literals other than k, in the weekly and2 order of that link
        ij = [order[j] for j in range(3) if j != k]
        po = pair_order[tuple(sorted(ij))]
        return [L[order.index(v)] for v in po]
    if form == "T":
        f = {"op": "and", "args": L}
    elif form.startswith("sib"):
        k = int(form[3:]); f = {"op": "and", "args": [flip[j] if j == k else L[j] for j in range(3)]}
    elif form.startswith("pair"):
        f = {"op": "and", "args": pair_args(int(form[4:]))}
    elif form.startswith("single"):
        f = L[int(form[6:])]
    elif form == "mirT":
        f = {"op": "or", "args": flip}
    elif form == "mirP":
        po = pair_args(mir_k)
        f = {"op": "or", "args": [{"op": "not", "a": a} if a["op"] == "var" else a["a"] for a in po]}
    elif form.startswith("count"):
        f = {"op": "and", "args": L}  # carrier for the question blocks; statement replaced below
        msgs = wc.messages(cluster, f)
        name = {i: wc.LETTERS[k] for k, i in enumerate(wc.events_of(f))}
        items = "; ".join(f"({k + 1}) {wc.literal(a, name)}" for k, a in enumerate(L))
        old = wc.statement(f, name)
        new = (f"Statement to assess: Of the following three statements, {COUNT_WORDS[int(form[5:])]}: {items}.\n"
               "What is the probability that this statement is true?")
        assert old in msgs[1]["content"]
        msgs[1]["content"] = msgs[1]["content"].replace(old, new)
        return msgs, f"count{form[5:]}[{'; '.join(wc.formula_str(a) for a in L)}]"
    else:
        raise ValueError(form)
    return wc.messages(cluster, f), wc.formula_str(f)


def sha(msgs):
    return hashlib.sha256(json.dumps(msgs).encode()).hexdigest()


def weekly_orders(week):
    """and2 order per link and and3 order per triangle, as asked in the weekly run."""
    pair_order, tri_order = {}, {}
    with (WEEKLY / "sixweek" / "queries.jsonl").open() as f:
        for line in f:
            if f'"week": "{week}"' not in line[:80]:
                continue
            q = json.loads(line)
            if q["kind"] == "and2":
                pair_order[tuple(sorted(q["events"]))] = q["events"]
            elif q["kind"] == "and3":
                tri_order[tuple(sorted(q["events"]))] = q["events"]
    return pair_order, tri_order


def stratified(tris, rel, n, rng):
    def stratum(t):
        return "|".join(sorted(rel[f"{a}-{b}"]["type"] for a, b in itertools.combinations(sorted(t), 2)))
    groups = defaultdict(list)
    for t in tris:
        groups[stratum(t)].append(t)
    if len(tris) <= n:
        return [(t, stratum(t)) for t in tris]
    quota = {g: n * len(v) / len(tris) for g, v in groups.items()}
    take = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: (-(quota[g] - take[g]), g))[: n - sum(take.values())]:
        take[g] += 1
    out = []
    for g in sorted(groups):
        out += [(t, g) for t in rng.sample(sorted(groups[g]), take[g])]
    return out


def cmd_build(out: Path, per_week: int):
    rows, items, summ = [], {}, []
    weekly_sha = {}
    with (WEEKLY / "sixweek" / "log.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("p") is not None and r.get("served_model") == MODEL:
                weekly_sha[r["prompt_sha256"]] = r["query_id"]
    for week in WEEKS:
        rng = random.Random(f"{SEED}-balanced-{week}")
        events = [json.loads(l) for l in (WEEKLY / "weeks" / week / "events.jsonl").read_text().splitlines()]
        for e in events:
            e.pop("index", None)
        g = json.loads((WEEKLY / "weeks" / week / "graph.json").read_text())
        pair_order, tri_order = weekly_orders(week)
        sample = stratified([tuple(t) for t in g["triangles"]], g["rel"], per_week, rng)
        labels = [e["label"] for e in events]
        for n_t, (t, stratum) in enumerate(sample):
            order = tri_order[tuple(sorted(t))]
            s = [rng.random() < 0.5 for _ in range(3)]
            mir_k = rng.randrange(3)
            y = 4 * labels[order[0]] + 2 * labels[order[1]] + labels[order[2]]
            tri = {"tri_id": f"{week}-t{n_t:03d}", "week": week, "events": order, "signs": [int(v) for v in s],
                   "n_neg": 3 - sum(s), "mir_k": mir_k, "stratum": stratum, "outcome": y,
                   "labels": [labels[v] for v in order], "forms": {}}
            for form in FORMS:
                msgs, fstr = messages_for(week, events, order, pair_order, s, form, mir_k)
                h = sha(msgs)
                at = atoms(form, s, mir_k)
                tri["forms"][form] = {"sha": h, "atoms": at, "label": int(y in at), "formula": fstr}
                if h not in items:
                    items[h] = {"prompt_sha256": h, "week": week, "messages": msgs, "formula": fstr,
                                "weekly_query_id": weekly_sha.get(h)}
            rows.append(tri)
        summ.append({"week": week, "triangles_in_week": len(g["triangles"]), "sampled": len(sample),
                     "strata": dict(Counter(st for _, st in sample))})
        print(json.dumps({k: summ[-1][k] for k in ("week", "triangles_in_week", "sampled")}), flush=True)
    # sanity: an all-positive T must be the weekly and3 prompt, an all-positive pair the weekly and2 prompt
    for tri in rows:
        if all(tri["signs"]):
            assert tri["forms"]["T"]["sha"] in weekly_sha, tri["tri_id"]
        for k in range(3):
            if tri["signs"][k]:
                assert tri["forms"][f"single{k}"]["sha"] in weekly_sha, (tri["tri_id"], k)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "triangles.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for q in items.values():
            f.write(json.dumps(q) + "\n")
    new = [q for q in items.values() if q["weekly_query_id"] is None]
    chars = sum(len(m["content"]) for q in new for m in q["messages"])
    usd = chars / CHARS_PER_TOKEN * wc.PRICE_IN / 1e6 + len(new) * OUT_TOKENS * wc.PRICE_OUT / 1e6
    by_form_reuse = Counter()
    for tri in rows:
        for form, v in tri["forms"].items():
            by_form_reuse[(ftype(form), v["sha"] in weekly_sha)] += 1
    tot = {"weeks": WEEKS, "triangles": len(rows), "statements": 16 * len(rows), "unique_prompts": len(items),
           "reused_from_weekly": len(items) - len(new), "new_calls": len(new), "new_prompt_chars": chars,
           "usd_est": round(usd, 3),
           "statements_reused_by_type": {k: by_form_reuse[(k, True)] for k in sorted({a for a, _ in by_form_reuse})},
           "statements_by_type": {k: by_form_reuse[(k, True)] + by_form_reuse[(k, False)]
                                  for k in sorted({a for a, _ in by_form_reuse})},
           "per_week": summ}
    (out / "build-summary.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps({k: v for k, v in tot.items() if k != "per_week"}, indent=1))


def cmd_run(out: Path, workers: int, max_usd: float, key_secret: str):
    import precutoff_client as pc
    workers = min(workers, 8)
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text().splitlines()]
    log = out / "log.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[r["prompt_sha256"]] = r
    wl = {}
    need = {q["prompt_sha256"] for q in rows if q["weekly_query_id"]}
    with (WEEKLY / "sixweek" / "log.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["prompt_sha256"] in need:
                wl[r["prompt_sha256"]] = r
    reused = 0
    with log.open("a") as f:
        for q in rows:
            h = q["prompt_sha256"]
            if h not in done and h in wl:
                r = dict(wl[h]); r.update(reused_from=str(WEEKLY / "sixweek" / "log.jsonl"),
                                           weekly_query_id=r.pop("query_id"), original_cost=r["cost"], cost=0.0)
                f.write(json.dumps(r) + "\n"); done[h] = r; reused += 1
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False, "temperature": 0}, wc.PRICE_IN, wc.PRICE_OUT)
    spent = [sum(r["cost"] for r in done.values())]
    base = spent[0]
    todo = [q for q in rows if q["prompt_sha256"] not in done]
    random.Random(SEED).shuffle(todo)
    print(f"{len(rows)} prompts, {len(done)} done ({reused} reused now), {len(todo)} to run; spent ${spent[0]:.3f}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    n_run, reason = [0], [None]

    def one(q):
        if stop.is_set():
            return None
        msgs = q["messages"]
        first = pc.chat(MODEL, msgs, max_tokens=20)
        if first["served_model"] and first["served_model"] != MODEL:
            reason[0] = f"served model {first['served_model']} != {MODEL}"; stop.set(); return None
        p, why = vb.parse_probability(first["text"])
        rec = {"prompt_sha256": q["prompt_sha256"], "raw": first["text"], "served_model": first["served_model"],
               "usage": first["usage"], "errors": first["errors"], "finish_reason": first["finish_reason"],
               "parsed_first": p, "reason_first": why, "retried": False, "raw_retry": None, "p": p, "reason": why}
        cost = wc.cost_of(first["usage"])
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=20)
            if second["served_model"] and second["served_model"] != MODEL:
                reason[0] = f"served model {second['served_model']} != {MODEL}"; stop.set(); return None
            p2, why2 = vb.parse_probability(second["text"])
            cost += wc.cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(workers) as pool, log.open("a") as f:
        futs = [pool.submit(one, q) for q in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == 300:
                    proj = spent[0] + (spent[0] - base) / n_run[0] * (len(todo) - n_run[0])
                    print(f"projection after 300: ${proj:.2f} (cap ${max_usd})", flush=True)
                    if proj > max_usd:
                        reason[0] = "projection exceeds cap"; stop.set()
                if spent[0] >= max_usd:
                    reason[0] = "spend reached cap"; stop.set()
                if n_run[0] % 1000 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {len(done) + n_run[0]}/{len(rows)} spent ${spent[0]:.3f}", flush=True)
    recs = {json.loads(l)["prompt_sha256"]: json.loads(l) for l in log.read_text().splitlines()}
    print(f"{'STOPPED (' + str(reason[0]) + ') ' if stop.is_set() else ''}logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs.values())}, missing {sum(r['p'] is None for r in recs.values())}, "
          f"spend ${sum(r['cost'] for r in recs.values()):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-week", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=9.5)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a.out, a.per_week)
    else:
        cmd_run(a.out, a.workers, a.max_usd, a.key_secret)


if __name__ == "__main__":
    main()
