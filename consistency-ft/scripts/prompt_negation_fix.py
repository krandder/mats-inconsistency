"""Negation-explicit prompt (v7_explicit) vs v6_direct, Qwen3-32B on DashScope, thinking off.

Protocol = the averaging-vs-structure t0 arm (scripts/run_decomposition_sampling.py):
one form per call, temperature 0, max_tokens 20, the v6 parser, one retry at
temperature 0 with RETRY_REMINDER.  A served model other than --model stops the run.
--model qwen3-32b (default) or qwen3-235b-a22b (thinking off, list price below).

  run      --split validation|test --variant NAME   (log: <out>/log-<split>-<variant>.jsonl, resumable)
  analyze  --split S --variant NAME --against LABEL=PATH[:key] ...

Coherence (per pair, events A = P and A = Q with their negations):
  corr(p(A), p(not A)) (ideal -1), share p(A) == p(not A), share |p(A) + p(not A) - 1| > 0.2.
Log loss per form, raw and after the kl projection of scripts/coherence_projection.py.
Paired bootstrap over pairs (2,000 resamples) for every difference.
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
import precutoff_client as pc  # noqa: E402
from coherence_projection import project  # noqa: E402
from training.supervised_pairs import FORM_ORDER, load_split  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

MODEL = "qwen3-32b"
# Model Studio intl list price, $/M tokens (input, output with thinking off)
PRICES = {"qwen3-32b": (0.16, 0.64), "qwen3-235b-a22b": (0.7, 2.8)}
PRICE_IN, PRICE_OUT = PRICES[MODEL]
DATA = "/home/agent/data/jev-disjoint-v1-filtered-20260919"
LO = 0.01


def rows_for(split: str, variant: str) -> list[dict]:
    rates = vb.train_base_rates(DATA)
    rows = []
    for ex in load_split(DATA, split, labels_file="full-label-targets.jsonl"):
        for form in FORM_ORDER:
            qid = ex.queries[form]["query_id"]
            rows.append({"pair_id": ex.pair["pair_id"], "query_id": qid, "form": form,
                         "observed": qid == ex.pair["observed_query_id"], "label": ex.labels[qid],
                         "p_id": ex.pair["p_id"], "q_id": ex.pair["q_id"],
                         "_msgs": vb.messages(ex, form, variant, rates=rates)})
    return rows


def cost_of(usage: dict) -> float:
    return usage.get("prompt_tokens", 0) * PRICE_IN / 1e6 + usage.get("completion_tokens", 0) * PRICE_OUT / 1e6


def run(args) -> None:
    global MODEL, PRICE_IN, PRICE_OUT
    MODEL = args.model
    PRICE_IN, PRICE_OUT = PRICES[MODEL]
    if args.workers > 12:
        raise SystemExit("workers capped at 12")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = out / f"log-{args.split}-{args.variant}.jsonl"
    spent_all = sum(json.loads(l)["cost"] for p in out.glob("log-*.jsonl") for l in p.read_text().splitlines())
    done = {json.loads(l)["query_id"] for l in log.read_text().splitlines()} if log.exists() else set()
    todo = [r for r in rows_for(args.split, args.variant) if r["query_id"] not in done]
    # A prompt identical to one already answered (same split, another variant's log) is reused, not re-sent:
    # the call is temperature 0, and iterations change only some forms.
    seen = {}
    for p in sorted(out.glob(f"log-{args.split}-*.jsonl")):
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if not r.get("reused_from"):
                seen.setdefault(r["prompt_sha256"], (r, p.name))
    reused = []
    for r in todo:
        h = hashlib.sha256(json.dumps(r["_msgs"]).encode()).hexdigest()
        if h in seen:
            old, name = seen[h]
            rec = {k2: v for k2, v in r.items() if k2 != "_msgs"}
            rec.update({k2: old[k2] for k2 in ("raw", "served_model", "usage", "errors", "retried", "raw_retry", "p", "reason")},
                       variant=args.variant, prompt_sha256=h, reused_from=name, cost=0.0)
            reused.append(rec)
    if reused:
        with log.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in reused)
        got = {r["query_id"] for r in reused}
        todo = [r for r in todo if r["query_id"] not in got]
    print(f"{args.split}/{args.variant}: reused {len(reused)} identical prompts, {len(todo)} calls to run; "
          f"all logs so far ${spent_all:.3f}", flush=True)
    k = subprocess.run(["secret", "get", "dashscope-api-key-01"], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit("secret dashscope-api-key-01 unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False}, PRICE_IN, PRICE_OUT)
    spent, stop, lock = [spent_all], threading.Event(), threading.Lock()

    def one(row):
        if stop.is_set() or spent[0] >= args.max_usd:
            stop.set(); return None
        msgs = row["_msgs"]
        first = pc.chat(MODEL, msgs, max_tokens=20, extra={"temperature": 0})
        if first["served_model"] and first["served_model"] != MODEL:
            stop.set(); raise RuntimeError(f"served model {first['served_model']} != {MODEL}")
        p, why = vb.parse_probability(first["text"])
        rec = {k2: v for k2, v in row.items() if k2 != "_msgs"}
        rec.update(variant=args.variant, prompt_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                   raw=first["text"], served_model=first["served_model"], usage=first["usage"], errors=first["errors"],
                   retried=False, raw_retry=None, p=p, reason=why)
        cost = cost_of(first["usage"])
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}],
                             max_tokens=20, extra={"temperature": 0})
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], p=p2, reason=why2)
        rec["cost"] = cost
        return rec

    with ThreadPoolExecutor(args.workers) as pool, log.open("a") as f:
        for fut in as_completed([pool.submit(one, r) for r in todo]):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush(); spent[0] += rec["cost"]
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    print(f"log {len(recs)} calls, retried {sum(r['retried'] for r in recs)}, missing {sum(r['p'] is None for r in recs)}, "
          f"this log ${sum(r['cost'] for r in recs):.3f}, all logs ${spent[0]:.3f}" + (" STOPPED" if stop.is_set() else ""),
          flush=True)


# --- analysis ----------------------------------------------------------------

def load(path: str, key: str = "p", sample: str | None = None) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if sample is not None and r.get("sample") != sample:
            continue
        rows.append({"pair_id": r["pair_id"], "form": r["form"], "label": r["label"], "observed": r["observed"],
                     "p": r[key]})
    return rows


def by_pair(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["pair_id"]][r["form"]] = r
    return d


def events(bp, pairs):
    """Per pair: list of (p(A), p(not A)) for A = P and A = Q (missing -> 0.5)."""
    ev = {}
    for pid in pairs:
        f = bp[pid]
        g = lambda k: 0.5 if f[k]["p"] is None else f[k]["p"]
        ev[pid] = [(g("P"), g("NOT P")), (g("Q"), g("NOT Q"))]
    return ev


def coherence(ev, pairs):
    x = np.array([a for p in pairs for a, _ in ev[p]]); y = np.array([b for p in pairs for _, b in ev[p]])
    return {"n_events": len(x), "corr": float(np.corrcoef(x, y)[0, 1]), "identical": float(np.mean(np.isclose(x, y))),
            "sum_off_0.2": float(np.mean(np.abs(x + y - 1) > 0.2)), "mean_sum": float(np.mean(x + y))}


def ll_terms(bp, pairs):
    """pair -> {form: log loss}."""
    out = {}
    for pid in pairs:
        out[pid] = {}
        for form, r in bp[pid].items():
            p = 0.5 if r["p"] is None else min(1 - LO, max(LO, r["p"]))
            out[pid][form] = -(r["label"] * math.log(p) + (1 - r["label"]) * math.log(1 - p))
    return out


def ll_summary(t, pairs):
    s = {f: float(np.mean([t[p][f] for p in pairs])) for f in FORM_ORDER}
    s["all12"] = float(np.mean([t[p][f] for p in pairs for f in FORM_ORDER]))
    lit = ["P", "NOT P", "Q", "NOT Q"]
    s["literals"] = float(np.mean([t[p][f] for p in pairs for f in lit]))
    s["compounds"] = float(np.mean([t[p][f] for p in pairs for f in FORM_ORDER if f not in lit]))
    return s


def boot_diff(fn, pairs, n=2000, seed=20260929):
    """Paired bootstrap over pairs of fn(resampled pairs) (a scalar difference)."""
    rng = np.random.default_rng(seed)
    point = fn(pairs)
    arr = np.array(pairs)
    d = [fn(list(arr[rng.integers(0, len(pairs), len(pairs))])) for _ in range(n)]
    return {"diff": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def analyze_one(rows):
    bp = by_pair(rows)
    pairs = sorted(bp)
    proj = project([dict(r, query_id=None) for r in rows], "kl")
    bpp = by_pair(proj)
    return {"bp": bp, "pairs": pairs, "ev": events(bp, pairs), "raw": ll_terms(bp, pairs), "proj": ll_terms(bpp, pairs),
            "missing": sum(r["p"] is None for r in rows)}


def analyze(args) -> None:
    arms = {}
    for spec in [f"{args.variant}={args.out}/log-{args.split}-{args.variant}.jsonl"] + args.against:
        label, rest = spec.split("=", 1)
        parts = rest.split(":")
        path, key, sample = parts[0], (parts[1] if len(parts) > 1 else "p"), (parts[2] if len(parts) > 2 else None)
        rows = load(path, key, sample)
        arms[label] = analyze_one(rows)
    main = arms[args.variant]
    pairs = main["pairs"]
    report = {"split": args.split, "pairs": len(pairs), "arms": {}, "vs": {}}
    for label, a in arms.items():
        assert set(a["pairs"]) >= set(pairs), label
        report["arms"][label] = {"missing": a["missing"], "coherence": coherence(a["ev"], pairs),
                                 "log_loss_raw": ll_summary(a["raw"], pairs), "log_loss_proj_kl": ll_summary(a["proj"], pairs)}
    for label, a in arms.items():
        if label == args.variant:
            continue
        cmp = {}
        for m in ("corr", "identical", "sum_off_0.2"):
            cmp[m] = boot_diff(lambda ps, m=m: coherence(main["ev"], ps)[m] - coherence(a["ev"], ps)[m], pairs)
        for kind in ("raw", "proj"):
            for grp in ("all12", "literals", "compounds", "P", "NOT P", "Q", "NOT Q"):
                cmp[f"ll_{kind}_{grp}"] = boot_diff(
                    lambda ps, k=kind, g=grp: ll_summary(main[k], ps)[g] - ll_summary(a[k], ps)[g], pairs, n=1000)
        report["vs"][label] = cmp
    Path(args.report).write_text(json.dumps(report, indent=1))
    for label, r in report["arms"].items():
        c, lr, lp = r["coherence"], r["log_loss_raw"], r["log_loss_proj_kl"]
        print(f"{label:28s} corr {c['corr']:+.3f} ident {c['identical']:.3f} |sum-1|>.2 {c['sum_off_0.2']:.3f} "
              f"mean sum {c['mean_sum']:.3f} | LL raw {lr['all12']:.4f} (lit {lr['literals']:.4f} cmp {lr['compounds']:.4f}) "
              f"proj {lp['all12']:.4f} | missing {r['missing']}")
    for label, cmp in report["vs"].items():
        print(f"{args.variant} - {label}: " + "; ".join(
            f"{k} {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in cmp.items()
            if k in ("corr", "identical", "sum_off_0.2", "ll_raw_all12", "ll_proj_all12", "ll_raw_literals")))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); a = sub.add_parser("analyze")
    for p in (r, a):
        p.add_argument("--split", required=True, choices=["train", "validation", "test"])
        p.add_argument("--variant", required=True)
        p.add_argument("--out", default="/home/agent/data/prompt-negation-fix-20260929")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--model", default="qwen3-32b", choices=sorted(PRICES))
    r.add_argument("--max-usd", type=float, default=5.0, help="cap over all logs in --out")
    a.add_argument("--against", nargs="*", default=[], help="LABEL=PATH[:key[:sample]]")
    a.add_argument("--report", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else analyze(args)


if __name__ == "__main__":
    main()
