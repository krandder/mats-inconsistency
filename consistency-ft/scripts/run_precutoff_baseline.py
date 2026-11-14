"""Pre-cutoff zero-shot baseline: the Luna ceiling setup on a model trained before the test questions.

Same prompt, system rule, strict-JSON parse, single retry and missing handling
as scripts/run_luna_ceiling.py (commit de67871); only the transport differs:
a direct chat-completions API call (scripts/precutoff_client.py) instead of a
Codex session.  Missing forms stay None (never imputed).  Every finished pair
is appended to <out>/log-<split>.jsonl, so the run resumes after a restart,
and the run stops before starting new pairs once --max-usd is spent.

Usage: run_precutoff_baseline.py --data-root D --split validation --model qwen3-235b-a22b --out DIR
       [--thinking on|off] [--thinking-budget N] [--workers 8] [--limit N] [--max-usd 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_luna_ceiling import RETRY_REMINDER, RULE, pair_prompt, parse_forms, to_rows  # noqa: E402
import precutoff_client as pc  # noqa: E402
from training.supervised_pairs import load_split  # noqa: E402

SPENT = [0.0]


def call(prompt: str, model: str) -> dict:
    r = pc.chat(model, [{"role": "system", "content": RULE}, {"role": "user", "content": prompt}], max_tokens=16000)
    SPENT[0] += r["cost"] or 0.0
    return r


def run_pair(ex, model) -> dict:
    prompt = pair_prompt(ex)
    first = call(prompt, model)
    got1, why1 = parse_forms(first["text"])
    rec = {"pair_id": ex.pair["pair_id"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
           "raw": first["text"], "reasoning_chars": len(first["reasoning"] or ""), "usage": first["usage"],
           "cost": first["cost"], "served_model": first["served_model"], "finish_reason": first["finish_reason"],
           "errors": first["errors"], "reason_first": why1, "parsed_first": got1, "retried": False,
           "raw_retry": None, "usage_retry": None, "parsed_retry": None, "reason_retry": None}
    final = dict(got1)
    if why1 != "ok":
        second = call(prompt + "\n\n" + RETRY_REMINDER, model)
        got2, why2 = parse_forms(second["text"])
        rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], parsed_retry=got2,
                   reason_retry=why2, errors_retry=second["errors"], cost_retry=second["cost"])
        final = {**got1, **got2} if why2 != "ok" else got2
    rec["final"] = final
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, choices=sorted(pc.MODELS))
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--thinking-budget", type=int, help="DashScope thinking_budget (reasoning-token cap)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-usd", type=float, default=10.0)
    args = ap.parse_args()
    if args.model.startswith("qwen3"):
        provider, mid, extra, pin, pout = pc.MODELS[args.model]
        # Non-thinking output is billed at the lower rate on DashScope ($2.8/M vs $8.4/M).
        extra = {**extra, "enable_thinking": args.thinking == "on"}
        if args.thinking == "on" and args.thinking_budget:
            extra["thinking_budget"] = args.thinking_budget
        pc.MODELS[args.model] = (provider, mid, extra, pin,
                                 pout if args.thinking == "on" else 2.8)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    examples = load_split(args.data_root, args.split, labels_file="full-label-targets.jsonl")
    if args.limit:
        examples = examples[:args.limit]
    log = out / f"log-{args.split}.jsonl"
    done = {json.loads(l)["pair_id"] for l in log.read_text().splitlines()} if log.exists() else set()
    todo = [ex for ex in examples if ex.pair["pair_id"] not in done]
    print(f"{args.split}: {len(examples)} pairs, {len(done)} done, {len(todo)} to run", flush=True)
    lock, stop = threading.Lock(), threading.Event()

    def guarded(ex):
        if stop.is_set() or SPENT[0] >= args.max_usd:
            stop.set()
            return None
        return run_pair(ex, args.model)

    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(guarded, ex) for ex in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is None:
                continue
            rec.update(model=args.model, provider_model=pc.MODELS[args.model][1], thinking=args.thinking,
                       thinking_budget=args.thinking_budget, request_extra=pc.MODELS[args.model][2],
                       split=args.split, finished_at=time.time())
            with lock, log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 25 == 0 or i == len(todo):
                print(f"{args.split}: {i}/{len(todo)} (last {rec['pair_id']}: {rec['reason_first']}) spent ${SPENT[0]:.2f}", flush=True)
    if stop.is_set():
        print(f"STOPPED at budget cap ${args.max_usd:.2f} (spent ${SPENT[0]:.2f})", flush=True)
    recs = {}
    for l in log.read_text().splitlines():
        r = json.loads(l)
        recs[r["pair_id"]] = r
    rows = [row for ex in examples if ex.pair["pair_id"] in recs for row in to_rows(ex, recs[ex.pair["pair_id"]])]
    with (out / f"predictions-{args.split}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows, missing {sum(r['p'] is None for r in rows)}, run spend ${SPENT[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
