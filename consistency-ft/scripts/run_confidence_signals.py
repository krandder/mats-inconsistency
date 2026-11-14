"""Per-answer confidence signals from Qwen3-32B over DashScope (one form per call, v6_direct family).

Variants per (pair, form), temperature 0, token logprobs on (top 5, the DashScope maximum):
  lp     the v6_direct prompt unchanged: the stated probability plus the logprobs of its digits
  range  v6_direct with a second answer line, a 90% interval for the probability
  know   v6_direct with a second answer line, self-rated knowledge 1-5
  yes    the same question as a YES/NO answer ("Will this statement be true?", "Answer: YES or NO");
         P(YES) from the YES/NO token logprobs (DashScope streams no logprobs for the first token,
         hence the "Answer:" prefix)
The question text, statement and the "What is the probability ..." line are unchanged in
lp/range/know; only the answer instruction differs (range, know).  Parser: vb.parse_probability
(the second line has no "probability" word, so the probability parse is unaffected).  One retry
on an unparseable reply (the vb retry reminder); missing stays None.

Pairs: the 300-pair subset of docs/averaging-vs-structure-20260924.md (--data old), or the
structured pairs' question-only prompts (--data structured, prompts-structured-Q.jsonl).
Every call goes to <out>/log.jsonl keyed by (query_id, variant); the run resumes, stops at
--max-usd (all calls in the log), and stops if the served model is not qwen3-32b.

Usage: run_confidence_signals.py --out DIR --variants lp,range,know,yes [--data old|structured] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
PRICE_IN, PRICE_OUT = 0.16, 0.64  # Model Studio intl list price (as in run_decomposition_sampling.py)
DATA_OLD = "/home/agent/data/jev-disjoint-v1-filtered-20260919"
SUBSET = "/home/agent/data/decomposition-32b-20260924/subset.json"
SP_PROMPTS = "/home/agent/data/structured-pairs-20260924/prompts-structured-Q.jsonl"

ANSWER_RANGE = ("Reply with exactly two lines of this form and nothing else:\nProbability: 0.xx\nRange: 0.xx-0.xx\n"
                "(the first line: a single number between 0.01 and 0.99; the second line: a 90% interval for where your "
                "probability would land if you knew everything that can be known today).")
ANSWER_KNOW = ("Reply with exactly two lines of this form and nothing else:\nProbability: 0.xx\nKnowledge: k\n"
               "(the first line: a single number between 0.01 and 0.99; the second line: how much you know about the "
               "events in this statement, from 1 = guessing to 5 = know them well).")
YES_Q = "Will this statement be true?"
ANSWER_YES = "Reply with exactly one line of this form and nothing else:\nAnswer: YES or NO"
MAXTOK = {"lp": 20, "range": 40, "know": 30, "yes": 5}


def variant_msgs(msgs: list[dict], v: str) -> list[dict]:
    user = msgs[1]["content"]
    assert user.count(vb.ANSWER_DIRECT) == 1
    if v == "lp":
        return msgs
    if v == "range":
        user = user.replace(vb.ANSWER_DIRECT, ANSWER_RANGE)
    elif v == "know":
        user = user.replace(vb.ANSWER_DIRECT, ANSWER_KNOW)
    elif v == "yes":
        q = "What is the probability that this statement is true?"
        assert user.count(q) == 1
        user = user.replace(q, YES_Q).replace(vb.ANSWER_DIRECT, ANSWER_YES)
    else:
        raise ValueError(v)
    return [msgs[0], {"role": "user", "content": user}]


def old_rows():
    examples = load_split(DATA_OLD, "test", labels_file="full-label-targets.jsonl")
    keep = set(json.loads(Path(SUBSET).read_text())["pair_ids"])
    examples = [ex for ex in examples if ex.pair["pair_id"] in keep]
    return build_rows(examples, vb.train_base_rates(DATA_OLD))


def structured_rows():
    rows = []
    for l in Path(SP_PROMPTS).read_text().splitlines():
        r = json.loads(l)
        rows.append({"pair_id": r["pair_id"], "query_id": r["query_id"], "form": r["form"], "_msgs": r["messages"]})
    return rows


def cost_of(usage: dict) -> float:
    return usage.get("prompt_tokens", 0) * PRICE_IN / 1e6 + usage.get("completion_tokens", 0) * PRICE_OUT / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="lp,range,know,yes")
    ap.add_argument("--data", choices=["old", "structured"], default="old")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--limit-calls", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    args = ap.parse_args()
    if args.workers > 8:
        raise SystemExit("workers capped at 8")
    rows = old_rows() if args.data == "old" else structured_rows()
    jobs = [(r, v) for v in args.variants.split(",") for r in rows]
    k = subprocess.run(["secret", "get", args.key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {args.key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False}, PRICE_IN, PRICE_OUT)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"
    done = set()
    spent = [0.0]
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done.add((r["query_id"], r["variant"])); spent[0] += r["cost"]
    todo = [(r, v) for r, v in jobs if (r["query_id"], v) not in done]
    if args.limit_calls:
        todo = todo[:args.limit_calls]
    print(f"{len(jobs)} planned, {len(todo)} to run, spent ${spent[0]:.3f}", flush=True)
    stop = threading.Event()
    lock = threading.Lock()

    def one(job):
        row, v = job
        if stop.is_set() or spent[0] >= args.max_usd:
            stop.set(); return None
        msgs = variant_msgs(row["_msgs"], v)
        extra = {"temperature": 0, "logprobs": True, "top_logprobs": 5}
        r1 = pc.chat(MODEL, msgs, max_tokens=MAXTOK[v], extra=extra)
        if r1["served_model"] and r1["served_model"] != MODEL:
            stop.set(); raise RuntimeError(f"served model {r1['served_model']} != {MODEL}")
        rec = {kk: vv for kk, vv in row.items() if kk != "_msgs"}
        rec.update(variant=v, prompt_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest(), raw=r1["text"],
                   logprobs=r1["logprobs"], served_model=r1["served_model"], usage=r1["usage"], errors=r1["errors"],
                   finish_reason=r1["finish_reason"], retried=False)
        cost = cost_of(r1["usage"])
        if v != "yes":
            p, why = vb.parse_probability(r1["text"])
            if p is None:
                r2 = pc.chat(MODEL, msgs + [{"role": "assistant", "content": r1["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=MAXTOK[v], extra=extra)
                cost += cost_of(r2["usage"])
                p2, why2 = vb.parse_probability(r2["text"])
                rec.update(retried=True, raw_retry=r2["text"], logprobs_retry=r2["logprobs"])
                p, why = p2, why2
            rec.update(p=p, reason=why)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    t0 = time.time(); n = [0]
    with ThreadPoolExecutor(args.workers) as pool, log.open("a") as f:
        for fut in as_completed([pool.submit(one, j) for j in todo]):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; n[0] += 1
                if n[0] % 1000 == 0:
                    print(f"{n[0]}/{len(todo)} spent ${spent[0]:.3f} {n[0] / (time.time() - t0):.1f}/s", flush=True)
    print(f"done {n[0]} new calls, total spent ${spent[0]:.3f}{' STOPPED' if stop.is_set() else ''}", flush=True)


if __name__ == "__main__":
    main()
