"""Transfer test of the frozen AND-correction prompt to the Manifold pair test set (pre-registered in
docs/and-correction-transfer-20260930.md).

Prompts: the v7b_explicit test prompts of scripts/prompt_negation_fix.py (1,066 pairs, one form per call).
Arm C: the four AND forms, each with the frozen CORRECTION paragraph of scripts/weekly_worldmodel_arms.py
inserted before the answer line (no edits); qwen3-32b, thinking off, temperature 0, max_tokens 20, one retry
with RETRY_REMINDER.  A served model other than qwen3-32b stops the run.
Comparison: the existing uncorrected qwen3-32b answers (U) in
/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl, for all 12 forms.

  run    [--workers 12] [--max-usd 15] [--stop-at EPOCH]   -> <out>/log.jsonl (resumable)
  check                                                    prompt checks and the pre-run sizing (no calls)
  score                                                    -> <out>/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from training.supervised_pairs import FORM_ORDER  # noqa: E402
from weekly_worldmodel_arms import CORRECTION  # noqa: E402

CORRECTION_SHA = "31e151649af70f4372baa6a8b15085c4b7e781c2b6595f27dc8260ba9ce4875c"
OUT = Path("/home/agent/data/and-correction-transfer-20260930")
U_LOG = Path("/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl")
MODEL = "qwen3-32b"
PRICE_IN, PRICE_OUT = 0.7, 2.8  # list $/M, thinking off (the ledger's rate for this model)
LO = 0.01
AND_FORMS = {"(P) AND (Q)": ("P", "Q"), "(P) AND (NOT Q)": ("P", "NOT Q"),
             "(NOT P) AND (Q)": ("NOT P", "Q"), "(NOT P) AND (NOT Q)": ("NOT P", "NOT Q")}  # form -> its two literals
LITERALS = ["P", "NOT P", "Q", "NOT Q"]
OR_FORMS = [f for f in FORM_ORDER if " OR " in f]
B, SEED = 2000, 20260930


def _sha(msgs):
    return hashlib.sha256(json.dumps(msgs).encode()).hexdigest()


def corrected(msgs):
    msgs = [dict(m) for m in msgs]
    c = msgs[-1]["content"]
    tail = "\n\n" + vb.ANSWER_DIRECT
    assert c.count(tail) == 1, "answer line not found exactly once"
    msgs[-1]["content"] = c.replace(tail, "\n\n" + CORRECTION + tail)
    return msgs


def build():
    from prompt_negation_fix import rows_for
    assert hashlib.sha256(CORRECTION.encode()).hexdigest() == CORRECTION_SHA, "correction text changed"
    rows = [r for r in rows_for("test", "v7b_explicit") if r["form"] in AND_FORMS]
    for r in rows:
        r["base_sha256"] = _sha(r["_msgs"])
        r["messages"] = corrected(r.pop("_msgs"))
        r["prompt_sha256"] = _sha(r["messages"])
    return rows


def load_u():
    return {r["query_id"]: r for r in map(json.loads, U_LOG.read_text().splitlines())}


def cmd_check():
    rows = build()
    u = load_u()
    same = sum(u[r["query_id"]]["prompt_sha256"] == r["base_sha256"] for r in rows)
    print(f"{len(rows)} corrected AND-form prompts over {len({r['pair_id'] for r in rows})} pairs; "
          f"uncorrected base prompt identical to the logged U prompt for {same}/{len(rows)}")
    ex = rows[0]["messages"][-1]["content"]
    print("example tail:\n" + ex[-900:])
    s = scores(None)
    print(json.dumps({k: s[k] for k in ("n_pairs", "yes_rate_and", "mean_p_and", "ll", "sizing")}, indent=1))


def cmd_run(workers, max_usd, stop_at):
    import precutoff_client as pc
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build()
    log = OUT / "log.jsonl"
    done = {json.loads(l)["query_id"]: json.loads(l) for l in log.read_text().splitlines()} if log.exists() else {}
    k = subprocess.run(["secret", "get", "dashscope-api-key-01"], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit("secret dashscope-api-key-01 unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False, "temperature": 0}, PRICE_IN, PRICE_OUT)

    def cost_of(u):
        return u.get("prompt_tokens", 0) * PRICE_IN / 1e6 + u.get("completion_tokens", 0) * PRICE_OUT / 1e6

    before = sum(r["cost"] for r in done.values())
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"{len(rows)} calls, {len(done)} done, {len(todo)} to run; spend ${before:.3f}, cap ${max_usd}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    tot, n_run, reason = [before], [0], [None]

    def one(row):
        if stop.is_set() or (stop_at and time.time() > stop_at):
            return None
        msgs = row["messages"]
        first = pc.chat(MODEL, msgs, max_tokens=20)
        if first["served_model"] and first["served_model"] != MODEL:
            reason[0] = f"served model {first['served_model']} != {MODEL}"; stop.set(); return None
        p, why = vb.parse_probability(first["text"])
        u = first["usage"] or {}
        rec = {k2: row[k2] for k2 in ("pair_id", "query_id", "form", "observed", "label", "p_id", "q_id",
                                      "prompt_sha256", "base_sha256")}
        rec.update(arm="C", raw=first["text"], served_model=first["served_model"], usage=u, errors=first["errors"],
                   finish_reason=first["finish_reason"], retried=False, raw_retry=None, p=p, reason=why)
        cost = cost_of(u)
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=20)
            if second["served_model"] and second["served_model"] != MODEL:
                reason[0] = f"served model {second['served_model']} != {MODEL}"; stop.set()
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"] or {})
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(min(workers, 12)) as pool, log.open("a") as f:
        for fut in as_completed([pool.submit(one, r) for r in todo]):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                tot[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == 300:
                    proj = tot[0] + (tot[0] - before) / n_run[0] * (len(todo) - n_run[0])
                    print(f"projection after 300: total ${proj:.2f} (cap ${max_usd})", flush=True)
                    if proj > max_usd:
                        reason[0] = "projection exceeds cap"; stop.set()
                if tot[0] >= max_usd:
                    reason[0] = "spend reached cap"; stop.set()
                if n_run[0] % 500 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {len(done) + n_run[0]}/{len(rows)} ${tot[0]:.3f}", flush=True)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    print(f"{'STOPPED (' + str(reason[0]) + ') ' if stop.is_set() else ''}logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs)}, missing {sum(r['p'] is None for r in recs)}, "
          f"served {sorted({r['served_model'] for r in recs})}, spend ${sum(r['cost'] for r in recs):.3f}", flush=True)


# --- scoring -------------------------------------------------------------------

def ll(p, y):
    p = 0.5 if p is None else min(1 - LO, max(LO, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def m0(f, lit):
    a, b = AND_FORMS[f]
    return lit[a] * lit[b]


def m0c(f, lit):
    """Product with 1 - p(A) for a negated literal instead of the answer to the NOT form."""
    pos = {"P": lit["P"], "Q": lit["Q"]}
    a, b = AND_FORMS[f]
    g = lambda x: 1 - pos[x[4:]] if x.startswith("NOT ") else pos[x]
    return g(a) * g(b)


def boot(per_pair: np.ndarray):
    """per_pair: (n_pairs,) differences; mean and paired bootstrap 95% CI over pairs."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_pair), size=(B, len(per_pair)))
    d = per_pair[idx].mean(1)
    return {"diff": float(per_pair.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def scores(c_log: Path | None):
    from coherence_projection import project
    u = list(load_u().values())
    by_u = defaultdict(dict)
    for r in u:
        by_u[r["pair_id"]][r["form"]] = r
    pairs = sorted(by_u)
    by_c = defaultdict(dict)
    if c_log is not None:
        for r in map(json.loads, c_log.read_text().splitlines()):
            by_c[r["pair_id"]][r["form"]] = r
        missing_pairs = [p for p in pairs if len(by_c[p]) != 4]
        assert not missing_pairs, f"{len(missing_pairs)} pairs without all four corrected AND forms"
    lit_p = {p: {f: (0.5 if by_u[p][f]["p"] is None else min(1 - LO, max(LO, by_u[p][f]["p"]))) for f in LITERALS}
             for p in pairs}
    y = {p: {f: by_u[p][f]["label"] for f in FORM_ORDER} for p in pairs}
    arr = lambda fn: np.array([fn(p) for p in pairs], dtype=float)
    ands = list(AND_FORMS)
    out = {"n_pairs": len(pairs), "yes_rate_and": float(np.mean([y[p][f] for p in pairs for f in ands])),
           "mean_p_and": {"U": float(np.mean([by_u[p][f]["p"] for p in pairs for f in ands])),
                          "M0": float(np.mean([m0(f, lit_p[p]) for p in pairs for f in ands]))}}
    L = {"U": arr(lambda p: np.mean([ll(by_u[p][f]["p"], y[p][f]) for f in ands])),
         "M0": arr(lambda p: np.mean([ll(m0(f, lit_p[p]), y[p][f]) for f in ands])),
         "M0c": arr(lambda p: np.mean([ll(m0c(f, lit_p[p]), y[p][f]) for f in ands])),
         "singles": arr(lambda p: np.mean([ll(by_u[p][f]["p"], y[p][f]) for f in LITERALS]))}
    per_form = {f: {"U": arr(lambda p, f=f: ll(by_u[p][f]["p"], y[p][f])),
                    "M0": arr(lambda p, f=f: ll(m0(f, lit_p[p]), y[p][f]))} for f in ands}
    # sizing from the existing answers only: SD of the per-pair AND-form difference U - M0
    sd = float(np.std(L["U"] - L["M0"], ddof=1))
    se = sd / math.sqrt(len(pairs))
    out["sizing"] = {"sd_per_pair_U_minus_M0": sd, "se": se, "mde_80pct_power_alpha05": 2.8 * se,
                     "U_minus_M0": boot(L["U"] - L["M0"])}
    if c_log is None:
        out["ll"] = {k: float(v.mean()) for k, v in L.items()}
        return out
    L["C"] = arr(lambda p: np.mean([ll(by_c[p][f]["p"], y[p][f]) for f in ands]))
    for f in ands:
        per_form[f]["C"] = arr(lambda p, f=f: ll(by_c[p][f]["p"], y[p][f]))
    out["mean_p_and"]["C"] = float(np.mean([0.5 if by_c[p][f]["p"] is None else by_c[p][f]["p"]
                                            for p in pairs for f in ands]))
    out["ll"] = {k: float(v.mean()) for k, v in L.items()}
    out["primary_C_minus_M0"] = boot(L["C"] - L["M0"])
    out["C_minus_U"] = boot(L["C"] - L["U"])
    out["U_minus_M0"] = boot(L["U"] - L["M0"])
    out["C_minus_M0c"] = boot(L["C"] - L["M0c"])
    out["per_form"] = {f: {"ll": {k: float(v.mean()) for k, v in d.items()},
                           "C_minus_M0": boot(d["C"] - d["M0"]), "C_minus_U": boot(d["C"] - d["U"])}
                       for f, d in per_form.items()}
    out["and_ge_min_literal"] = {
        k: float(np.mean([(src[p][f]["p"] or 0.5) > min(lit_p[p][AND_FORMS[f][0]], lit_p[p][AND_FORMS[f][1]]) + 1e-9
                          for p in pairs for f in ands])) for k, src in (("U", by_u), ("C", by_c))}
    out["missing_C"] = sum(by_c[p][f]["p"] is None for p in pairs for f in ands)
    # coherence projection (kl): 12 forms with corrected AND forms vs all 12 uncorrected
    base = [{"pair_id": r["pair_id"], "form": r["form"], "label": r["label"], "observed": r["observed"],
             "p": r["p"], "query_id": None} for r in u]
    mixed = [dict(r, p=by_c[r["pair_id"]][r["form"]]["p"]) if r["form"] in AND_FORMS else r for r in base]
    proj = {}
    for name, rows in (("U", base), ("C", mixed)):
        bp = defaultdict(dict)
        for r in project(rows, "kl"):
            bp[r["pair_id"]][r["form"]] = r["p"]
        proj[name] = bp
    raw = {"U": {p: {f: by_u[p][f]["p"] for f in FORM_ORDER} for p in pairs}}
    raw["C"] = {p: {f: (by_c[p][f]["p"] if f in AND_FORMS else by_u[p][f]["p"]) for f in FORM_ORDER} for p in pairs}
    groups = {"all12": list(FORM_ORDER), "literals": LITERALS, "and": ands, "or": OR_FORMS}

    def g_ll(src, g):
        return arr(lambda p: np.mean([ll(src[p][f], y[p][f]) for f in groups[g]]))

    out["projection_kl"] = {}
    for g in groups:
        pu, pc_, ru, rc = g_ll(proj["U"], g), g_ll(proj["C"], g), g_ll(raw["U"], g), g_ll(raw["C"], g)
        out["projection_kl"][g] = {"raw_U": float(ru.mean()), "raw_C": float(rc.mean()), "proj_U": float(pu.mean()),
                                   "proj_C": float(pc_.mean()), "projC_minus_projU": boot(pc_ - pu),
                                   "projC_minus_rawC": boot(pc_ - rc)}
    return out


def cmd_score():
    s = scores(OUT / "log.jsonl")
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    f = lambda d: f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
    print(f"pairs {s['n_pairs']}, YES rate AND {s['yes_rate_and']:.3f}, mean p AND {s['mean_p_and']}, missing C {s['missing_C']}")
    print("LL AND forms:", {k: round(v, 4) for k, v in s["ll"].items()})
    print("PRIMARY C - M0", f(s["primary_C_minus_M0"]), "| C - U", f(s["C_minus_U"]), "| U - M0", f(s["U_minus_M0"]),
          "| C - M0c", f(s["C_minus_M0c"]))
    for form, d in s["per_form"].items():
        print(f"  {form:22s} C {d['ll']['C']:.4f} U {d['ll']['U']:.4f} M0 {d['ll']['M0']:.4f}  C-M0 {f(d['C_minus_M0'])}  C-U {f(d['C_minus_U'])}")
    print("AND > min literal:", s["and_ge_min_literal"])
    for g, d in s["projection_kl"].items():
        print(f"  proj {g:8s} rawU {d['raw_U']:.4f} rawC {d['raw_C']:.4f} projU {d['proj_U']:.4f} projC {d['proj_C']:.4f} "
              f"projC-projU {f(d['projC_minus_projU'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check", "score"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=15.0)
    ap.add_argument("--stop-at", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.workers, a.max_usd, a.stop_at)
    elif a.cmd == "check":
        cmd_check()
    else:
        cmd_score()


if __name__ == "__main__":
    main()
