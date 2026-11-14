"""Pre-cutoff zero-shot baseline, one form per call: the Qwen3-32B v6_direct setup over the DashScope API.

Same prompt, rendering, parser, single retry and missing handling as
scripts/run_verbalized_baseline.py (commit 37578ec, variant v6_direct):
system + user message from training.verbalized_baseline.messages (user text
ends with " /no_think"), enable_thinking false, temperature 0, max_tokens 20;
one retry at temperature 0 with the assistant reply and RETRY_REMINDER
appended; missing stays None.  Only the transport differs (DashScope
chat-completions instead of local vLLM).

Every finished call is appended to <out>/log-<split>.jsonl, so the run
resumes after a restart.  The run stops before new calls once --max-usd
(cumulative over all logs in --out) is spent, and after --check-after calls it
stops if the projected split cost would push the total past --max-usd.

--verify-template TOKENIZER_DIR renders the same messages with the Qwen3 chat
template (jinja2, enable_thinking=False) and compares the prompt sha256 with
--verify-against (the 32B predictions file), without calling the API.

Usage: run_precutoff_perform.py --data-root D --split validation --out DIR [--workers 16] [--max-usd 12]
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
from training.supervised_pairs import FORM_ORDER, load_split  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

VARIANT = "v6_direct"
MODEL = "qwen3-235b-a22b"
PRICE_IN, PRICE_OUT = 0.7, 2.8  # $/M, DashScope intl list price, non-thinking output


def build_rows(examples, rates):
    rows = []
    for ex in examples:
        for form in FORM_ORDER:
            qid = ex.queries[form]["query_id"]
            rows.append({"pair_id": ex.pair["pair_id"], "query_id": qid, "form": form,
                         "observed": qid == ex.pair["observed_query_id"], "label": ex.labels[qid],
                         "p_id": ex.pair["p_id"], "q_id": ex.pair["q_id"],
                         "_msgs": vb.messages(ex, form, VARIANT, rates=rates)})
    return rows


def cost_of(usage: dict) -> float:
    return usage.get("prompt_tokens", 0) * PRICE_IN / 1e6 + usage.get("completion_tokens", 0) * PRICE_OUT / 1e6


def verify_template(rows, tok_dir: Path, against: Path) -> None:
    import jinja2
    cfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    tmpl = env.from_string(cfg["chat_template"])
    ref = {r["query_id"]: r["prompt_sha256"] for r in map(json.loads, against.read_text().splitlines())}
    same = diff = absent = 0
    for r in rows:
        text = tmpl.render(messages=r["_msgs"], add_generation_prompt=True, enable_thinking=False)
        h = hashlib.sha256(text.encode()).hexdigest()
        if r["query_id"] not in ref:
            absent += 1
        elif ref[r["query_id"]] == h:
            same += 1
        else:
            diff += 1
    print(json.dumps({"verify_template": str(against), "same": same, "different": diff, "absent": absent}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=12.0)
    ap.add_argument("--check-after", type=int, default=400)
    ap.add_argument("--limit-pairs", type=int)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    ap.add_argument("--verify-template")
    ap.add_argument("--verify-against")
    args = ap.parse_args()

    examples = load_split(args.data_root, args.split, labels_file="full-label-targets.jsonl")
    if args.limit_pairs:
        examples = examples[:args.limit_pairs]
    rows = build_rows(examples, vb.train_base_rates(args.data_root))
    if args.verify_template:
        verify_template(rows, Path(args.verify_template), Path(args.verify_against))
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    k = subprocess.run(["secret", "get", args.key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {args.key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pc.MODELS[MODEL] = ("dashscope", MODEL, {"enable_thinking": False, "temperature": 0}, PRICE_IN, PRICE_OUT)

    spent_before = 0.0
    for lg in out.glob("log-*.jsonl"):
        if lg.name != f"log-{args.split}.jsonl":
            spent_before += sum(json.loads(l)["cost"] for l in lg.read_text().splitlines())
    log = out / f"log-{args.split}.jsonl"
    done = {}
    if log.exists():
        for l in log.read_text().splitlines():
            r = json.loads(l); done[r["query_id"]] = r
    spent = [spent_before + sum(r["cost"] for r in done.values())]
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"{args.split}: {len(rows)} calls, {len(done)} done, {len(todo)} to run; spent so far ${spent[0]:.3f}", flush=True)
    lock, stop = threading.Lock(), threading.Event()
    n_run = [0]; split_cost = [sum(r["cost"] for r in done.values())]; n_done = [len(done)]

    def one(row):
        if stop.is_set() or spent[0] >= args.max_usd:
            stop.set(); return None
        msgs = row["_msgs"]
        first = pc.chat(MODEL, msgs, max_tokens=20)
        if first["served_model"] and first["served_model"] != MODEL:
            stop.set(); raise RuntimeError(f"served model {first['served_model']} != {MODEL}")
        c1 = cost_of(first["usage"])
        p, why = vb.parse_probability(first["text"])
        rec = {k2: v for k2, v in row.items() if k2 != "_msgs"}
        rec.update(prompt_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest(), raw=first["text"],
                   finish_reason=first["finish_reason"], served_model=first["served_model"], usage=first["usage"],
                   errors=first["errors"], prompt_tokens=first["usage"].get("prompt_tokens"),
                   output_tokens=first["usage"].get("completion_tokens"), parsed_first=p, reason_first=why,
                   retried=False, raw_retry=None, usage_retry=None, p=p, reason=why)
        cost = c1
        if p is None:
            second = pc.chat(MODEL, msgs + [{"role": "assistant", "content": first["text"] or ""},
                                            {"role": "user", "content": vb.RETRY_REMINDER}], max_tokens=20)
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(second["usage"])
            rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], errors_retry=second["errors"],
                       p=p2, reason=why2)
        rec.update(cost=cost, finished_at=time.time())
        return rec

    with ThreadPoolExecutor(args.workers) as pool, log.open("a") as f:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                spent[0] += rec["cost"]; split_cost[0] += rec["cost"]; n_run[0] += 1; n_done[0] += 1
                if n_run[0] == args.check_after:
                    proj = spent_before + split_cost[0] / n_done[0] * len(rows)
                    print(f"projection after {n_done[0]} calls: split ${split_cost[0] / n_done[0] * len(rows):.2f}, "
                          f"total ${proj:.2f} (cap ${args.max_usd:.2f})", flush=True)
                    if proj > args.max_usd:
                        print("STOP: projection exceeds cap", flush=True); stop.set()
                if n_run[0] % 500 == 0:
                    print(f"{args.split}: {n_done[0]}/{len(rows)} spent ${spent[0]:.3f}", flush=True)
    if stop.is_set():
        print(f"STOPPED (spent ${spent[0]:.3f})", flush=True)
    recs = {}
    for l in log.read_text().splitlines():
        r = json.loads(l); recs[r["query_id"]] = r
    final = [recs[r["query_id"]] for r in rows if r["query_id"] in recs]
    with (out / f"predictions-{args.split}.jsonl").open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(final)}/{len(rows)} rows, retried {sum(r['retried'] for r in final)}, "
          f"missing {sum(r['p'] is None for r in final)}, total spend ${spent[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
