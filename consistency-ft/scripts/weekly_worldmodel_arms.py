"""Follow-up arms on the six sampled weeks of the weekly world-model run (pre-registered in
docs/weekly-worldmodel-20260927.md, section "Follow-up arms ...").

C  AND-correction: the base joint prompt plus one correction paragraph before the answer line; qwen3-32b,
   thinking off, temperature 0, max_tokens 20, one retry (same as the base run).
T  thinking: qwen3-32b with enable_thinking (streamed), base prompts without " /no_think", temperature 0.6,
   top_p 0.95, thinking_budget 4000, max_tokens 6000, one retry; joints plus the singles of their events.

Both arms take a prefix of one seeded permutation (seed "arms-20260930") of the 17,531 six-week joints.
Any served model other than qwen3-32b stops the run.  --max-usd caps the spend of both arms together.

Usage:
  weekly_worldmodel_arms.py run --arm C --n 6000
  weekly_worldmodel_arms.py run --arm T --n 20          (pilot)
  weekly_worldmodel_arms.py cost                        (spend and per-call cost so far)
  weekly_worldmodel_arms.py score                       -> <data>/arms/summary.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import verbalized_baseline as vb  # noqa: E402

DATA = Path("/home/agent/data/weekly-worldmodel-20260927")
BASE = DATA / "sixweek"
ARMS = DATA / "arms"
MODEL = "qwen3-32b"
PERM_SEED = "arms-20260930"
PRICE_IN, PRICE_OUT, PRICE_OUT_THINK = 0.7, 2.8, 8.4  # list $/M
CORRECTION = ("Note: forecasters tend to rate statements like this one too high. The statement needs every question "
              "above to resolve YES, so its probability cannot be higher than that of the least likely question, and "
              "unless the questions are closely linked it is close to the product of their separate probabilities, "
              "which is usually much lower than any one of them.")
SETTINGS = {"C": ({"enable_thinking": False, "temperature": 0}, 20, PRICE_OUT),
            "R": ({"enable_thinking": False, "temperature": 0}, 20, PRICE_OUT),  # base prompt, joints (confirm)
            "S": ({"enable_thinking": False, "temperature": 0}, 20, PRICE_OUT),  # base prompt, singles (confirm)
            "T": ({"enable_thinking": True, "temperature": 0.6, "top_p": 0.95, "thinking_budget": 4000}, 6000,
                  PRICE_OUT_THINK)}


def configure(base: Path | None, arms: Path | None, seed: str | None):
    """Confirmatory runs point the same code at another built query set (e.g. the train split)."""
    global BASE, ARMS, PERM_SEED
    BASE, ARMS, PERM_SEED = base or BASE, arms or ARMS, seed or PERM_SEED


def base_queries():
    return [json.loads(l) for l in (BASE / "queries.jsonl").read_text().splitlines()]


def permutation(qs):
    joints = sorted(q["query_id"] for q in qs if q["kind"] != "single")
    random.Random(PERM_SEED).shuffle(joints)
    return joints


def _sha(msgs):
    return hashlib.sha256(json.dumps(msgs).encode()).hexdigest()


def build(arm: str, n: int):
    qs = base_queries()
    by = {q["query_id"]: q for q in qs}
    single_of = {(q["week"], q["events"][0]): q["query_id"] for q in qs if q["kind"] == "single"}
    perm = permutation(qs)[:n]
    out, seen = [], set()
    for qid in perm:
        q = by[qid]
        singles = [single_of[(q["week"], i)] for i in q["events"]]
        ids = {"C": [qid], "R": [qid], "S": singles, "T": singles + [qid]}[arm]
        for k in ids:
            if k in seen:
                continue
            seen.add(k)
            b = by[k]
            msgs = copy.deepcopy(b["messages"])
            c = msgs[1]["content"]
            if arm == "C":
                tail = "\n\n" + vb.ANSWER_DIRECT
                assert c.count(tail) == 1, k
                c = c.replace(tail, "\n\n" + CORRECTION + tail)
            elif arm == "T":
                assert c.endswith(" /no_think"), k
                c = c[: -len(" /no_think")]
            msgs[1]["content"] = c
            out.append({"query_id": k, "week": b["week"], "kind": b["kind"], "events": b["events"],
                        "messages": msgs, "prompt_sha256": _sha(msgs)})
    return out


def spent(arm):
    log = ARMS / arm / "log.jsonl"
    return sum(json.loads(l)["cost"] for l in log.read_text().splitlines()) if log.exists() else 0.0


def cmd_run(arm, n, workers, max_usd, key_secret, stop_at):
    import precutoff_client as pc
    d = ARMS / arm
    d.mkdir(parents=True, exist_ok=True)
    rows = build(arm, n)
    (d / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = d / "log.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[r["query_id"]] = r
    other = sum(spent(a) for a in SETTINGS if a != arm)
    extra, max_tokens, pout = SETTINGS[arm]
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, extra, PRICE_IN, pout)

    def cost_of(u):
        return u.get("prompt_tokens", 0) * PRICE_IN / 1e6 + u.get("completion_tokens", 0) * pout / 1e6

    before = sum(r["cost"] for r in done.values())
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"arm {arm}: {len(rows)} calls, {len(done)} done, {len(todo)} to run; arm spend ${before:.3f}, "
          f"other arms ${other:.3f}, cap ${max_usd}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    tot, n_run, reason = [other + before], [0], [None]
    check_at = 50 if arm == "T" else 300

    def one(row):
        if stop.is_set() or (stop_at and time.time() > stop_at):
            return None
        msgs = row["messages"]
        first = pc.chat(MODEL, msgs, max_tokens=max_tokens)
        if first["served_model"] and first["served_model"] != MODEL:
            reason[0] = f"served model {first['served_model']} != {MODEL}"; stop.set(); return None
        p, why = vb.parse_probability(first["text"])
        u = first["usage"] or {}
        rec = {"query_id": row["query_id"], "arm": arm, "kind": row["kind"], "prompt_sha256": row["prompt_sha256"],
               "raw": first["text"], "served_model": first["served_model"], "usage": u, "errors": first["errors"],
               "finish_reason": first["finish_reason"], "parsed_first": p, "reason_first": why, "retried": False,
               "raw_retry": None, "p": p, "reason": why}
        if arm == "T":
            rec["reasoning"] = first["reasoning"]
        cost = cost_of(u)
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=max_tokens)
            if second["served_model"] and second["served_model"] != MODEL:
                reason[0] = f"served model {second['served_model']} != {MODEL}"; stop.set()
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"] or {})
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2,
                       finish_reason_retry=second["finish_reason"])
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(min(workers, 12)) as pool, log.open("a") as f:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                tot[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == check_at:
                    new = tot[0] - other - before
                    proj = tot[0] + new / n_run[0] * (len(todo) - n_run[0])
                    print(f"projection after {check_at}: total ${proj:.2f} (cap ${max_usd})", flush=True)
                    if proj > max_usd:
                        reason[0] = "projection exceeds cap"; stop.set()
                if tot[0] >= max_usd:
                    reason[0] = "spend reached cap"; stop.set()
                if n_run[0] % 200 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {len(done) + n_run[0]}/{len(rows)} total ${tot[0]:.3f}", flush=True)
    recs = {json.loads(l)["query_id"]: json.loads(l) for l in log.read_text().splitlines()}
    print(f"{'STOPPED (' + str(reason[0]) + ') ' if stop.is_set() else ''}arm {arm} logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs.values())}, missing {sum(r['p'] is None for r in recs.values())}, "
          f"arm spend ${sum(r['cost'] for r in recs.values()):.3f}, all arms ${sum(spent(a) for a in SETTINGS):.3f}",
          flush=True)


def cmd_cost():
    """Spend and mean cost per call by kind (reads usage only, never answers)."""
    out = {}
    for a in SETTINGS:
        log = ARMS / a / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(l) for l in log.read_text().splitlines()]
        by = defaultdict(list)
        for r in recs:
            by[r["kind"]].append(r)
        out[a] = {"calls": len(recs), "spend": round(sum(r["cost"] for r in recs), 4),
                  "served": sorted({str(r["served_model"]) for r in recs}),
                  "by_kind": {k: {"n": len(v), "mean_cost": round(float(np.mean([r["cost"] for r in v])), 6),
                                  "mean_completion_tokens": round(float(np.mean(
                                      [(r["usage"] or {}).get("completion_tokens", 0) for r in v])), 1),
                                  "max_completion_tokens": max((r["usage"] or {}).get("completion_tokens", 0) for r in v),
                                  "finish_length": sum(r["finish_reason"] == "length" for r in v)}
                              for k, v in by.items()}}
        if a == "T" and recs:
            t = [(r["finished_at"]) for r in recs]
            out[a]["calls_per_min"] = round(len(recs) / max(1e-9, (max(t) - min(t)) / 60), 1) if len(recs) > 1 else None
    print(json.dumps(out, indent=1))


def cmd_score():
    import weekly_worldmodel_score as S
    _, rows, _, _, _ = S.load(BASE)
    boot = S.Boot(rows)
    byq = {r["qid"]: r for r in rows}
    single = {(r["week"], r["idx"][0]): r for r in rows if r["kind"] == "single"}
    meta = {}
    for a in SETTINGS:
        log = ARMS / a / "log.jsonl"
        recs = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        meta[a] = {"calls": len(recs), "missing": sum(r["p"] is None for r in recs), "spend": sum(r["cost"] for r in recs),
                   "retried": sum(r["retried"] for r in recs), "served": sorted({str(r["served_model"]) for r in recs})}
        for r in recs:
            if r["query_id"] in byq:
                byq[r["query_id"]][a] = r["p"]
        if a == "T":
            ct = [(r["usage"] or {}).get("completion_tokens", 0) for r in recs]
            meta[a].update(completion_tokens_mean=float(np.mean(ct)) if ct else None,
                           completion_tokens_median=float(np.median(ct)) if ct else None,
                           finish_length=sum(r["finish_reason"] == "length" for r in recs))
    for r in rows:
        if r["kind"] != "single":
            ps = [single[(r["week"], i)].get("T") if (r["week"], i) in single else None for i in r["idx"]]
            r["M0_T"] = float(np.prod(S.clip(ps))) if all(p is not None for p in ps) else None

    def ll(vals, rs):
        return S.logloss(vals, np.array([r["y"] for r in rs], dtype=float))

    def sel(pred, cols):
        return [r for r in rows if pred(r) and all(r.get(c) is not None for c in cols)]

    joint = lambda r: r["kind"] != "single"  # noqa: E731
    out = {"meta": meta, "prereg": "docs/weekly-worldmodel-20260927.md (Follow-up arms, PREREG)"}
    colsC = ["C", "M0", "raw"]
    out["P1_C_joints"] = S.compare(sel(joint, colsC), boot, ll, colsC,
                                   [("C", "M0"), ("C", "raw"), ("raw", "M0")], week_block=True)
    colsT = ["T", "M0", "raw", "M0_T"]
    rT = sel(joint, colsT)
    out["P2_T_joints"] = S.compare(rT, boot, ll, colsT,
                                   [("T", "M0"), ("T", "raw"), ("T", "M0_T"), ("M0_T", "M0"), ("raw", "M0")],
                                   week_block=True)
    rs = sel(lambda r: r["kind"] == "single", ["T", "raw"])
    out["P3_T_singles"] = S.compare(rs, boot, ll, ["T", "raw"], [("T", "raw")], week_block=True)
    out["S_C_vs_T_on_T_sample"] = S.compare(sel(joint, ["C", "T", "M0"]), boot, ll, ["C", "T", "M0"],
                                            [("C", "T"), ("C", "M0"), ("T", "M0")])
    for kind in ("and2", "and3"):
        k = lambda r, kind=kind: r["kind"] == kind  # noqa: E731
        out[f"S_C_{kind}"] = S.compare(sel(k, colsC), boot, ll, colsC, [("C", "M0"), ("C", "raw")])
        out[f"S_T_{kind}"] = S.compare(sel(k, colsT), boot, ll, colsT, [("T", "M0"), ("T", "raw"), ("T", "M0_T")])
    # AND bias: stated joint vs product of the singles (the arm's own singles; base singles for C)
    bias = {}
    for a, prod in (("raw", "M0"), ("C", "M0"), ("T", "M0_T"), ("T_vs_base_singles", "M0")):
        col = "T" if a.startswith("T") else a
        for kind in ("and2", "and3"):
            rr = [r for r in rows if r["kind"] == kind and r.get(col) is not None and r.get(prod) is not None]
            if a == "raw":  # base raw on C's sample, for a like-for-like reference
                rr = [r for r in rr if r.get("C") is not None]
            if not rr:
                continue
            singles_of = "T" if prod == "M0_T" else "raw"
            mins = [min(single[(r["week"], i)][singles_of] for i in r["idx"]) for r in rr]
            bias[f"{a}/{kind}"] = {"n": len(rr), "stated": float(np.mean([r[col] for r in rr])),
                                   "product": float(np.mean([r[prod] for r in rr])),
                                   "above_min_single": float(np.mean([r[col] > m + 1e-9 for r, m in zip(rr, mins)])),
                                   "yes_rate": float(np.mean([r["y"] for r in rr]))}
    rs1 = [r for r in rows if r["kind"] == "single" and r.get("T") is not None]
    if rs1:
        bias["singles_T_sample"] = {"n": len(rs1), "T": float(np.mean([r["T"] for r in rs1])),
                                    "raw": float(np.mean([r["raw"] for r in rs1])),
                                    "yes_rate": float(np.mean([r["y"] for r in rs1]))}
    out["and_bias"] = bias
    (ARMS / "summary.json").write_text(json.dumps(out, indent=1))
    for key in ("P1_C_joints", "P2_T_joints", "P3_T_singles", "S_C_vs_T_on_T_sample"):
        o = out[key]
        print(key, "n", o.get("n"), "events", o.get("events"))
        for c, v in o.items():
            if isinstance(v, dict) and "mean" in v:
                wb = v.get("week_block")
                print(f"   {c:10s} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                      + (f"  wk [{wb['lo']:+.4f},{wb['hi']:+.4f}]" if wb else ""))
    print(json.dumps({"meta": meta, "and_bias": bias}, indent=1))


def cmd_score_confirm():
    """Confirmatory run: C (frozen correction) vs M0 (product of the base singles S), R = base joints."""
    import weekly_worldmodel_score as S
    summ = json.loads((BASE / "build-summary.json").read_text())
    ev = {w: [json.loads(l) for l in (BASE / "weeks" / w / "events.jsonl").read_text().splitlines()]
          for w in summ["weeks"]}
    rows = [{"qid": q["query_id"], "week": q["week"], "kind": q["kind"], "y": q["label"], "idx": q["events"],
             "eids": [ev[q["week"]][i]["id"] for i in q["events"]]} for q in base_queries()]
    byq = {r["qid"]: r for r in rows}
    meta = {}
    for a in ("C", "R", "S"):
        log = ARMS / a / "log.jsonl"
        recs = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        meta[a] = {"calls": len(recs), "missing": sum(r["p"] is None for r in recs), "spend": sum(r["cost"] for r in recs),
                   "retried": sum(r["retried"] for r in recs), "served": sorted({str(r["served_model"]) for r in recs})}
        for r in recs:
            byq[r["query_id"]][a] = r["p"]
    single = {(r["week"], r["idx"][0]): r.get("S") for r in rows if r["kind"] == "single"}
    for r in rows:
        if r["kind"] != "single":
            ps = [single.get((r["week"], i)) for i in r["idx"]]
            r["M0"] = float(np.prod(S.clip(ps))) if all(p is not None for p in ps) else None
    boot = S.Boot(rows)

    def ll(vals, rs):
        return S.logloss(vals, np.array([r["y"] for r in rs], dtype=float))
    cols = ["C", "M0", "R"]
    sel = lambda pred: [r for r in rows if pred(r) and all(r.get(c) is not None for c in cols)]  # noqa: E731
    out = {"meta": meta, "weeks": summ["weeks"], "correction_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
           "prereg": "docs/weekly-worldmodel-20260927.md (Confirmatory run, PREREG)"}
    out["P_C_joints"] = S.compare(sel(lambda r: r["kind"] != "single"), boot, ll, cols,
                                  [("C", "M0"), ("C", "R"), ("R", "M0")], week_block=True)
    for kind in ("and2", "and3"):
        out[f"S_{kind}"] = S.compare(sel(lambda r, kind=kind: r["kind"] == kind), boot, ll, cols,
                                     [("C", "M0"), ("C", "R"), ("R", "M0")])
    bias = {}
    for a in ("R", "C"):
        for kind in ("and2", "and3"):
            rr = sel(lambda r, kind=kind: r["kind"] == kind)
            if not rr:
                continue
            mins = [min(single[(r["week"], i)] for i in r["idx"]) for r in rr]
            bias[f"{a}/{kind}"] = {"n": len(rr), "stated": float(np.mean([r[a] for r in rr])),
                                   "product": float(np.mean([r["M0"] for r in rr])),
                                   "above_min_single": float(np.mean([r[a] > m + 1e-9 for r, m in zip(rr, mins)])),
                                   "yes_rate": float(np.mean([r["y"] for r in rr]))}
    out["and_bias"] = bias
    (ARMS / "summary.json").write_text(json.dumps(out, indent=1))
    for key in ("P_C_joints", "S_and2", "S_and3"):
        o = out[key]
        print(key, "n", o.get("n"), "events", o.get("events"))
        for c, v in o.items():
            if isinstance(v, dict) and "mean" in v:
                wb = v.get("week_block")
                print(f"   {c:6s} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                      + (f"  wk [{wb['lo']:+.4f},{wb['hi']:+.4f}]" if wb else ""))
    print(json.dumps({"meta": meta, "and_bias": bias}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "cost", "score", "score-confirm"])
    ap.add_argument("--base", type=Path, help="built query set (default: <data>/sixweek)")
    ap.add_argument("--arms-dir", type=Path, help="output dir of the arms (default: <data>/arms)")
    ap.add_argument("--perm-seed", help="permutation seed (default: arms-20260930)")
    ap.add_argument("--arm", choices=list(SETTINGS))
    ap.add_argument("--n", type=int)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=15.0)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    ap.add_argument("--stop-at", help="UTC ISO time after which no new call starts")
    a = ap.parse_args()
    configure(a.base, a.arms_dir, a.perm_seed)
    if a.cmd == "run":
        import datetime as dt
        stop_at = dt.datetime.fromisoformat(a.stop_at).replace(tzinfo=dt.timezone.utc).timestamp() if a.stop_at else None
        cmd_run(a.arm, a.n, a.workers, a.max_usd, a.key_secret, stop_at)
    elif a.cmd == "cost":
        cmd_cost()
    elif a.cmd == "score-confirm":
        cmd_score_confirm()
    else:
        cmd_score()


if __name__ == "__main__":
    main()
