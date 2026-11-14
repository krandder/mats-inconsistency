"""Averaging vs structure: repeated one-form-per-call samples from Qwen3-32B over the DashScope API.

Same prompt, parser, max_tokens and single retry as scripts/run_precutoff_perform.py
(commit 4541343, v6_direct from training/verbalized_baseline.py @37578ec); only the
model (qwen3-32b, thinking off) and the sampling differ.  Per (pair, form):
  t0      temperature 0, one call (retry at temperature 0), the baseline protocol
  w1..w3  one sample each (same sampling as s*) of three fixed paraphrases of the
          statement (see paraphrase()); the rest of the prompt is unchanged
  s0..s4  temperature 0.7, top_p 0.8, top_k 20 (Qwen3 non-thinking defaults), each call
          with its own seed (sha256 of query_id and sample name), so samples of different
          forms do not share a random stream; an unparseable reply is retried once at the
          same temperature with a different seed.
Missing stays None.

Pairs: --pairs subset = a fixed random subset (--subset-size, --subset-seed) of the test
pairs; --pairs all = every test pair (the subset's finished calls are reused).  Every call
is appended to <out>/log.jsonl keyed by (query_id, sample), so the run resumes.  The run
stops before new calls once --max-usd (all calls in the log) is spent, and after
--check-after new calls stops if the projected total would pass --max-usd.  A served model
other than qwen3-32b stops the run (no substitution).

Usage: run_decomposition_sampling.py --data-root D --out DIR --pairs subset [--workers 12] [--max-usd 15]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import precutoff_client as pc  # noqa: E402
from run_precutoff_perform import build_rows  # noqa: E402
from training.supervised_pairs import load_split  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

MODEL = "qwen3-32b"
# Model Studio international (Singapore) list price, qwen3-32b model page, checked 2026-09-24.
PRICE_IN, PRICE_OUT = 0.16, 0.64
SAMPLES = ["t0", "s0", "s1", "s2", "s3", "s4"]
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
WORDINGS = ["w1", "w2", "w3"]
NEG = {"YES": "NO", "NO": "YES"}


def paraphrase(form: str, wording: str) -> str:
    """Logically equivalent rewrites of plain_statement(form) (v6_direct's statement text).

    w1 reorders / restates (compounds: the two conditions in the order B, A; literals:
       "The outcome of Question A is YES."),
    w2 plain sentence ("Question A resolves YES and Question B resolves NO." /
       "... or ... (or both)."; literals: "Question A will resolve YES."),
    w3 negated complement (De Morgan: AND -> "It is false that at least one of ...
       (negated parts)"; OR -> "It is false that both of ... (negated parts)";
       literals: "It is false that Question A resolves NO.").
    """
    op, a, b = vb._parts(form)
    w = vb._SIGN_WORD
    if op is None:
        name, sign = ("A", a) if a else ("B", b)
        return {"w1": f"The outcome of Question {name} is {w[sign]}.",
                "w2": f"Question {name} will resolve {w[sign]}.",
                "w3": f"It is false that Question {name} resolves {NEG[w[sign]]}."}[wording]
    one, two = f"Question A resolves {w[a]}", f"Question B resolves {w[b]}"
    if wording == "w1":
        lead = "Both of the following are true" if op == "AND" else "At least one of the following is true"
        return f"{lead}: (1) {two}; (2) {one}."
    if wording == "w2":
        return f"{one} and {two}." if op == "AND" else f"{one} or {two} (or both)."
    n1, n2 = f"Question A resolves {NEG[w[a]]}", f"Question B resolves {NEG[w[b]]}"
    if op == "AND":
        return f"It is false that at least one of the following is true: (1) {n1}; (2) {n2}."
    return f"It is false that both of the following are true: (1) {n1}; (2) {n2}."


def reworded(msgs: list[dict], form: str, wording: str) -> list[dict]:
    old = "Statement to assess: " + vb.plain_statement(form) + "\n"
    user = msgs[1]["content"]
    assert user.count(old) == 1, form
    return [msgs[0], {"role": "user", "content": user.replace(old, "Statement to assess: " + paraphrase(form, wording) + "\n")}]


def subset_ids(pair_ids: list[str], size: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(pair_ids), size))


def seed_of(query_id: str, sample: str, attempt: int) -> int:
    return int(hashlib.sha256(f"{query_id}|{sample}|{attempt}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


def cost_of(usage: dict) -> float:
    return usage.get("prompt_tokens", 0) * PRICE_IN / 1e6 + usage.get("completion_tokens", 0) * PRICE_OUT / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", choices=["subset", "all"], required=True)
    ap.add_argument("--subset-size", type=int, default=300)
    ap.add_argument("--subset-seed", type=int, default=20260924)
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--wordings", action="store_true", help="subset only: add w1..w3 paraphrase samples")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=15.0)
    ap.add_argument("--check-after", type=int, default=600)
    ap.add_argument("--limit-calls", type=int, help="smoke: run at most this many new calls")
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    args = ap.parse_args()
    if args.workers > 12:
        raise SystemExit("workers capped at 12 (shared DashScope key)")

    examples = load_split(args.data_root, "test", labels_file="full-label-targets.jsonl")
    sub = subset_ids([ex.pair["pair_id"] for ex in examples], args.subset_size, args.subset_seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "subset.json").write_text(json.dumps({"seed": args.subset_seed, "size": args.subset_size,
                                                 "method": "random.Random(seed).sample(sorted(test pair_ids), size)",
                                                 "pair_ids": sub}, indent=1))
    if args.pairs == "subset":
        keep = set(sub); examples = [ex for ex in examples if ex.pair["pair_id"] in keep]
    else:  # subset first, so a stop leaves the subset complete
        first = set(sub); examples = sorted(examples, key=lambda ex: ex.pair["pair_id"] not in first)
    rows = build_rows(examples, vb.train_base_rates(args.data_root))
    samples = args.samples.split(",")
    jobs = [(r, s) for s in samples for r in rows]
    if args.pairs == "subset" and args.wordings:
        jobs += [(r, w) for w in WORDINGS for r in rows]

    k = subprocess.run(["secret", "get", args.key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {args.key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False}, PRICE_IN, PRICE_OUT)

    log = out / "log.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[(r["query_id"], r["sample"])] = r
    spent = [sum(r["cost"] for r in done.values())]
    todo = [(r, s) for r, s in jobs if (r["query_id"], s) not in done]
    if args.limit_calls:
        todo = todo[:args.limit_calls]
    n_in_plan_done = len(jobs) - len([1 for r, s in jobs if (r["query_id"], s) not in done])
    print(f"{args.pairs}: {len(examples)} pairs, {len(jobs)} calls planned, {n_in_plan_done} done, {len(todo)} to run; "
          f"spent ${spent[0]:.3f}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    n_run = [0]; run_cost = [0.0]

    def call(msgs, sample, qid, attempt):
        extra = {"temperature": 0} if sample == "t0" else {**SAMPLING, "seed": seed_of(qid, sample, attempt)}
        return pc.chat(MODEL, msgs, max_tokens=20, extra=extra), extra

    def one(job):
        row, sample = job
        if stop.is_set() or spent[0] >= args.max_usd:
            stop.set(); return None
        msgs = row["_msgs"] if sample in SAMPLES else reworded(row["_msgs"], row["form"], sample)
        first, extra = call(msgs, sample, row["query_id"], 0)
        if first["served_model"] and first["served_model"] != MODEL:
            stop.set(); raise RuntimeError(f"served model {first['served_model']} != {MODEL}")
        p, why = vb.parse_probability(first["text"])
        rec = {k2: v for k2, v in row.items() if k2 != "_msgs"}
        rec.update(sample=sample, request=extra, prompt_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                   raw=first["text"], finish_reason=first["finish_reason"], served_model=first["served_model"],
                   usage=first["usage"], errors=first["errors"], parsed_first=p, reason_first=why,
                   retried=False, raw_retry=None, p=p, reason=why)
        cost = cost_of(first["usage"])
        if p is None:
            second, extra2 = call(msgs + [{"role": "assistant", "content": first["text"] or ""},
                                          {"role": "user", "content": vb.RETRY_REMINDER}], sample, row["query_id"], 1)
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], request_retry=extra2,
                       errors_retry=second["errors"], p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool, log.open("a") as f:
        futs = [pool.submit(one, j) for j in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; run_cost[0] += rec["cost"]; n_run[0] += 1
                if n_run[0] == args.check_after:
                    proj = spent[0] + run_cost[0] / n_run[0] * (len(todo) - n_run[0])
                    print(f"projection after {n_run[0]} new calls: total ${proj:.2f} (cap ${args.max_usd:.2f}); "
                          f"{n_run[0] / (time.time() - t0):.1f} calls/s", flush=True)
                    if proj > args.max_usd:
                        print("STOP: projection exceeds cap", flush=True); stop.set()
                if n_run[0] % 2000 == 0:
                    print(f"{n_run[0]}/{len(todo)} new calls, spent ${spent[0]:.3f}, "
                          f"{n_run[0] / (time.time() - t0):.1f} calls/s", flush=True)
    if stop.is_set():
        print(f"STOPPED (spent ${spent[0]:.3f})", flush=True)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    print(f"log has {len(recs)} calls, retried {sum(r['retried'] for r in recs)}, missing {sum(r['p'] is None for r in recs)}, "
          f"total spend ${sum(r['cost'] for r in recs):.3f}", flush=True)


if __name__ == "__main__":
    main()
