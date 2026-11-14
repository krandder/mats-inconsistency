"""Paid calls for eliciting projection weights (docs/eliciting-weights-20260927.md).

Variants, Qwen3 over DashScope, v6_direct family, question only:
  per (pair, form), one form per call:
    lp      the v6_direct prompt unchanged, T=0, digit logprobs (the quick answer; for 235B, which has no
            same-day quick answer yet)
    rev     predicted revision: v6_direct plus a second line "Change: 0.xx", how far the model expects its
            probability to move if it thought much longer (T=0, logprobs)
    think   measured revision: the same prompt with thinking on (thinking_budget, default 1024), T=0.6/top_p 0.95
  per pair, one call showing all 12 statements (shuffled order, seeded by pair id), T=0, logprobs:
    critique  the 12 statements with the model's quick answers (lp); for each: trust 1-5 and a revised probability
    evidence  the 12 statements without answers; for each: how directly the model can judge it, 1-5

The question block (system prompt, Question A/B texts) is copied verbatim from the pair's AND-form prompt.
Every call goes to <out>/log.jsonl keyed by (query_id, variant); resumable; stops at --max-usd; stops if the served
model differs from --model.

Usage: run_eliciting_weights.py --model qwen3-32b --out DIR --variants rev,think,critique,evidence
                                [--data old|structured] [--pairs N | --sample N] [--quick-log FILE] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import precutoff_client as pc  # noqa: E402
from run_confidence_signals import old_rows, structured_rows  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

PRICES = {"qwen3-32b": (0.16, 0.64, 0.64), "qwen3-235b-a22b": (0.7, 2.8, 8.4)}  # in, out, thinking out (intl list)

ANSWER_REV = ("Reply with exactly two lines of this form and nothing else:\nProbability: 0.xx\nChange: 0.xx\n"
              "(the first line: your probability now, a single number between 0.01 and 0.99; the second line: how far, "
              "in absolute terms, you expect your probability would move if you thought about this statement much "
              "longer and more carefully, from 0.00 = not at all to 0.50 = completely).")
CRITIQUE = """Below are {n} statements about Questions A and B, each with the probability you gave earlier when you were asked about it on its own, quickly and without seeing the others.

{items}

Some of these earlier answers may be mistakes: some may be inconsistent with the others, or based on a misreading or a guess. Review them together. For each statement give:
- trust: how much you trust your earlier answer, from 1 = probably a mistake to 5 = confident it is right;
- revised: your probability for the statement now (a number between 0.01 and 0.99).

Reply with exactly {n} lines of this form and nothing else:
S1: trust=k revised=0.xx
...
S{n}: trust=k revised=0.xx /no_think"""
EVIDENCE = """Below are {n} statements about Questions A and B.

{items}

Do not estimate probabilities. For each statement, rate how directly you can judge it from what you know, from 1 = you would have to guess or derive it from other judgments to 5 = you know facts that bear directly on it. Compare the statements with each other and use the full range.

Reply with exactly {n} lines of this form and nothing else:
S1: k
...
S{n}: k /no_think"""
STMT = re.compile(r"Statement to assess: (.*)\n")


def order_of(pair_id: str) -> list[int]:
    o = list(range(12))
    random.Random(pair_id).shuffle(o)
    return o


def pair_prompt(rows12: dict, v: str, quick: dict | None) -> tuple[list[dict], list[int]]:
    """rows12: form -> row.  Returns messages and the form-index order shown as S1..S12."""
    base = rows12["(P) AND (Q)"]["_msgs"]
    user = base[1]["content"]
    head = user[:user.index("Statement to assess:")].rstrip()
    order = order_of(rows12["P"]["pair_id"])
    items = []
    for k, j in enumerate(order, 1):
        f = FORM_ORDER[j]
        s = STMT.search(rows12[f]["_msgs"][1]["content"]).group(1).strip()
        if v == "critique":
            p = quick.get(f)
            items.append(f"S{k}: {s}\n    your earlier probability: {'none' if p is None else f'{p:.2f}'}")
        else:
            items.append(f"S{k}: {s}")
    body = (CRITIQUE if v == "critique" else EVIDENCE).format(n=12, items="\n".join(items))
    return [base[0], {"role": "user", "content": head + "\n\n" + body}], order


def form_msgs(msgs: list[dict], v: str) -> list[dict]:
    user = msgs[1]["content"]
    assert user.count(vb.ANSWER_DIRECT) == 1
    if v == "lp":
        return msgs
    if v == "rev":
        return [msgs[0], {"role": "user", "content": user.replace(vb.ANSWER_DIRECT, ANSWER_REV)}]
    if v == "think":
        assert user.rstrip().endswith("/no_think")
        return [msgs[0], {"role": "user", "content": user.rstrip()[:-len("/no_think")].rstrip()}]
    raise ValueError(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRICES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="rev,critique,evidence")
    ap.add_argument("--data", choices=["old", "structured"], default="old")
    ap.add_argument("--pairs", type=int, help="first N pairs (sorted ids) only")
    ap.add_argument("--sample", type=int, help="a seeded random sample of N pairs (seed 20260927)")
    ap.add_argument("--quick-log", help="log.jsonl holding the lp quick answers for critique (default: <out>/log.jsonl, "
                                         "variant lp)")
    ap.add_argument("--thinking-budget", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--limit-calls", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    args = ap.parse_args()
    if args.workers > 8:
        raise SystemExit("workers capped at 8")
    model = args.model
    pin, pout, pthink = PRICES[model]
    rows = old_rows() if args.data == "old" else structured_rows()
    pids = sorted({r["pair_id"] for r in rows})
    if args.pairs:
        keep = set(pids[:args.pairs]); rows = [r for r in rows if r["pair_id"] in keep]; pids = pids[:args.pairs]
    if args.sample:
        keep = set(random.Random(20260927).sample(pids, args.sample)); rows = [r for r in rows if r["pair_id"] in keep]
        pids = sorted(keep)
    byp = defaultdict(dict)
    for r in rows:
        byp[r["pair_id"]][r["form"]] = r
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"
    quick = defaultdict(dict)
    qlog = Path(args.quick_log) if args.quick_log else log
    if qlog.exists():
        for l in qlog.read_text().splitlines():
            r = json.loads(l)
            if r["variant"] == "lp" and r.get("p") is not None:
                quick[r["pair_id"]][r["form"]] = r["p"]
    jobs = []
    for v in args.variants.split(","):
        if v in ("critique", "evidence"):
            for p in pids:
                if v == "critique" and len(quick[p]) < 12:
                    continue
                jobs.append(({"pair_id": p, "query_id": f"{p}-pair"}, v))
        else:
            jobs += [(r, v) for r in rows]
    done, spent = set(), [0.0]
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done.add((r["query_id"], r["variant"])); spent[0] += r["cost"]
    todo = [(r, v) for r, v in jobs if (r["query_id"], v) not in done]
    if args.limit_calls:
        todo = todo[:args.limit_calls]
    print(f"{len(jobs)} planned, {len(todo)} to run, spent ${spent[0]:.3f}", flush=True)
    k = subprocess.run(["secret", "get", args.key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {args.key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[model] = ("dashscope", model, {"enable_thinking": False}, pin, pout)
    stop = threading.Event()
    lock = threading.Lock()

    def cost_of(u, thinking):
        return u.get("prompt_tokens", 0) * pin / 1e6 + u.get("completion_tokens", 0) * (pthink if thinking else pout) / 1e6

    def one(job):
        row, v = job
        if stop.is_set() or spent[0] >= args.max_usd:
            stop.set(); return None
        rec = {kk: vv for kk, vv in row.items() if kk != "_msgs"}
        if v in ("critique", "evidence"):
            msgs, order = pair_prompt(byp[row["pair_id"]], v, quick.get(row["pair_id"]))
            rec["order"] = [FORM_ORDER[j] for j in order]
            extra, mt = {"temperature": 0, "logprobs": True, "top_logprobs": 5}, 400 if v == "critique" else 150
        elif v == "think":
            msgs = form_msgs(row["_msgs"], v)
            extra = {"enable_thinking": True, "thinking_budget": args.thinking_budget, "temperature": 0.6, "top_p": 0.95}
            mt = args.thinking_budget + 200
        else:
            msgs = form_msgs(row["_msgs"], v)
            extra, mt = {"temperature": 0, "logprobs": True, "top_logprobs": 5}, 20 if v == "lp" else 40
        r1 = pc.chat(model, msgs, max_tokens=mt, extra=extra)
        if r1["served_model"] and r1["served_model"] != model:
            stop.set(); raise RuntimeError(f"served model {r1['served_model']} != {model}")
        rec.update(variant=v, model=model, prompt_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                   raw=r1["text"], reasoning=r1.get("reasoning"), logprobs=r1["logprobs"], served_model=r1["served_model"],
                   usage=r1["usage"], errors=r1["errors"], finish_reason=r1["finish_reason"], retried=False)
        cost = cost_of(r1["usage"], v == "think")
        if v in ("lp", "rev", "think"):
            p, why = vb.parse_probability(r1["text"] or "")
            if p is None and v != "think":
                r2 = pc.chat(model, msgs + [{"role": "assistant", "content": r1["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=mt, extra=extra)
                cost += cost_of(r2["usage"], False)
                p, why = vb.parse_probability(r2["text"] or "")
                rec.update(retried=True, raw_retry=r2["text"], logprobs_retry=r2["logprobs"])
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
                if n[0] % 500 == 0:
                    print(f"{n[0]}/{len(todo)} spent ${spent[0]:.3f} {n[0] / (time.time() - t0):.1f}/s", flush=True)
    print(f"done {n[0]} new calls, total spent ${spent[0]:.3f}{' STOPPED' if stop.is_set() else ''}", flush=True)


if __name__ == "__main__":
    main()
