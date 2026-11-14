"""Market-context zero-shot test: one form per call, three input arms, DashScope Qwen3.

Design: docs/conditional-joint-design-20260923.md.  The model states its own
probability for each of the 12 forms of a pair, one form per call.  Market
prices are evidence in the prompt, not constraints.

Arms (the input is identical for all 12 forms of a pair and for every model):
  Q  question only: both question blocks, no market data.
  P  deterministic evidence pack, training/asof.py build_pack(pair, t, "small"),
     i.e. a 2,000-token budget (Qwen3 tokenizer), then the same question blocks.
  L  the Luna context-agent output (context-agent/{split}.jsonl, field `context`),
     then the same question blocks.

Prompt (v6_direct wording, training/verbalized_baseline.py): system BASE_SYSTEM;
user = "Assume today is t ..." + [context] + "Question A" block + "Question B"
block + "Statement to assess: ..." + ANSWER_DIRECT + " /no_think".  Question blocks
use the as-of-t question and criteria text (question_block(context="asof_text")).
Both questions are shown for every form (v6_direct showed only the relevant one
for literal forms) so that the input is the same across the 12 forms.
t = 00:00 UTC of (earlier closing week - 7 days), as in the dataset.

build   packs + prompts for a split:  --split validation|test
project token/cost projection for (model, arm, split[, pairs]) from the built prompts
run     DashScope, thinking off, temperature 0, max_tokens 20, one retry with
        RETRY_REMINDER, missing recorded as None.  Resumable log per job.  The global
        ledger (all logs in --out) is checked before and during the run: the job is
        refused if spent + projection > --cap, and stops if spend reaches the cap.
        A served-model mismatch stops the run (no substitution).  Pairs run in a
        seeded random order so a partial job (--pairs N) is a random subset.

Labels are never read here.
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import verbalized_baseline as vb  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

DATA = Path("/home/agent/data/jev-market-context-open-20260923")
LUNA = Path("/home/agent/data/manifold-asof-20260923/context-agent")
OUT = Path("/home/agent/data/market-context-zeroshot-20260924")
PACK_SIZE = "small"  # 2,000-token budget
SEED = 20260924
# Model Studio Singapore list prices, $/M tokens (input, output non-thinking), checked 2026-09-24.
PRICES = {"qwen3-32b": (0.16, 0.64), "qwen3-235b-a22b": (0.7, 2.8)}
TEMPLATE_OVERHEAD = 20  # chat-template tokens per call (system/user markers), added to the projection
OUT_TOKENS = 7          # "Probability: 0.xx" in Qwen3 tokens
STATEMENT_MARGIN = 30   # est_prompt_tokens uses the shortest statement (form P); compound forms add ~20


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# --- build ----------------------------------------------------------------------

_STORE = None


def _pack(args):
    global _STORE
    from training.asof import build_pack, count_tokens, default_store
    pair, t = args
    if _STORE is None:
        _STORE = default_store()
    text, items = build_pack(pair, t, PACK_SIZE, _STORE, return_items=True)
    return pair["pair_id"], text, items, count_tokens(text)


def user_prompt(p_row: dict, q_row: dict, t: str, form: str, context: str | None) -> str:
    parts = [f"Assume today is {t}. Use only information available up to that date; do not use knowledge of later events."]
    if context:
        parts.append(context.strip())
    parts.append("Question A\n" + vb.question_block(p_row, context="asof_text"))
    parts.append("Question B\n" + vb.question_block(q_row, context="asof_text"))
    parts.append(f"Statement to assess: {vb.plain_statement(form)}\nWhat is the probability that this statement is true?")
    parts.append(vb.ANSWER_DIRECT)
    return "\n\n".join(parts) + " /no_think"


def build(split: str, out: Path, procs: int) -> None:
    from training.asof import count_tokens, pair_as_of
    d = DATA / split
    props = {r["id"]: r for r in jl(d / "propositions.jsonl")}
    pairs = jl(d / "pairs.jsonl")
    queries = jl(d / "queries.jsonl")
    qid = {(q["pair_id"], q["form"]): q["query_id"] for q in queries}
    luna = {r["pair_id"]: r for r in jl(LUNA / f"{split}.jsonl")}
    t_of = {p["pair_id"]: pair_as_of(p, props) for p in pairs}
    out.mkdir(parents=True, exist_ok=True)
    pack_path = out / f"packs-{split}.jsonl"
    packs = {r["pair_id"]: r for r in jl(pack_path)} if pack_path.exists() else {}
    todo = [(p, t_of[p["pair_id"]]) for p in pairs if p["pair_id"] not in packs]
    if todo:
        with ProcessPoolExecutor(procs) as ex, pack_path.open("a") as f:
            for n, (pid, text, items, ntok) in enumerate(ex.map(_pack, todo, chunksize=4), 1):
                rec = {"pair_id": pid, "t": t_of[pid], "size": PACK_SIZE, "tokens": ntok, "items": items, "text": text}
                packs[pid] = rec
                f.write(json.dumps(rec) + "\n")
                if n % 100 == 0:
                    print(f"packs {split}: {n}/{len(todo)}", flush=True)
    for arm in ("Q", "P", "L"):
        rows = []
        for p in pairs:
            pid = p["pair_id"]
            lr = luna[pid]
            from training.asof import to_ms
            assert to_ms(t_of[pid]) == lr["t"], pid  # Luna and the pack use the same t
            ctx = {"Q": None, "P": packs[pid]["text"], "L": lr["context"]}[arm]
            ctx_tokens = 0 if ctx is None else count_tokens(ctx)
            for form in FORM_ORDER:
                msgs = [{"role": "system", "content": vb.BASE_SYSTEM},
                        {"role": "user", "content": user_prompt(props[p["p_id"]], props[p["q_id"]], t_of[pid], form, ctx)}]
                rows.append({"query_id": qid[(pid, form)], "pair_id": pid, "form": form, "arm": arm,
                             "stratum": p["stratum"], "messages": msgs,
                             "prompt_sha256": hashlib.sha256(json.dumps(msgs).encode()).hexdigest(),
                             "context_tokens": ctx_tokens})
        # prompt tokens: count once per pair (forms differ only in the statement line)
        by_pair = {}
        for r in rows:
            if r["pair_id"] not in by_pair:
                by_pair[r["pair_id"]] = count_tokens(r["messages"][0]["content"] + r["messages"][1]["content"])
            r["est_prompt_tokens"] = by_pair[r["pair_id"]] + TEMPLATE_OVERHEAD
        with (out / f"prompts-{split}-{arm}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        toks = sorted(by_pair.values())
        print(f"{split} {arm}: {len(rows)} prompts, prompt tokens median {toks[len(toks) // 2]} "
              f"p90 {toks[int(.9 * len(toks))]} max {toks[-1]} total {sum(toks) * 12 / 1e6:.2f}M", flush=True)


# --- run ------------------------------------------------------------------------

def pair_order(pair_ids: list[str]) -> list[str]:
    ids = sorted(pair_ids)
    random.Random(f"{SEED}-order").shuffle(ids)
    return ids


def select(split: str, arm: str, out: Path, n_pairs: int | None) -> list[dict]:
    rows = jl(out / f"prompts-{split}-{arm}.jsonl")
    order = pair_order(list({r["pair_id"] for r in rows}))
    keep = set(order[:n_pairs] if n_pairs else order)
    return [r for r in rows if r["pair_id"] in keep]


def cost_of(model: str, usage: dict) -> float:
    pin, pout = PRICES[model]
    return usage.get("prompt_tokens", 0) * pin / 1e6 + usage.get("completion_tokens", 0) * pout / 1e6


def job_log(out: Path, model: str, arm: str, split: str) -> Path:
    return out / "logs" / f"{model}-{arm}-{split}.jsonl"


def ledger(out: Path) -> float:
    total = 0.0
    for p in sorted((out / "logs").glob("*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                total += json.loads(l)["cost"]
    return total


def projection(model: str, rows: list[dict], done: set[str]) -> float:
    pin, pout = PRICES[model]
    todo = [r for r in rows if r["query_id"] not in done]
    return sum(r["est_prompt_tokens"] + STATEMENT_MARGIN for r in todo) * pin / 1e6 + len(todo) * OUT_TOKENS * pout / 1e6


def run(model: str, arm: str, split: str, out: Path, n_pairs: int | None, workers: int, cap: float,
        key_secret: str, dry: bool) -> None:
    import precutoff_client as pc
    rows = select(split, arm, out, n_pairs)
    log = job_log(out, model, arm, split)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["query_id"] for l in log.read_text().splitlines()} if log.exists() else set()
    spent = ledger(out)
    proj = projection(model, rows, done)
    print(f"{model} {arm} {split}: {len(rows) // 12} pairs, {len(rows)} calls, {len(done)} done; "
          f"projected ${proj:.2f}; ledger ${spent:.2f}; after job ${spent + proj:.2f} (cap ${cap})", flush=True)
    if dry:
        return
    if spent + proj > cap:
        raise SystemExit("REFUSED: projected total exceeds cap")
    k = subprocess.run(["secret", "get", key_secret], capture_output=True, text=True, timeout=60).stdout.strip()
    if not k:
        raise SystemExit(f"secret {key_secret} unavailable")
    pc._KEYS["dashscope"] = k
    pin, pout = PRICES[model]
    pc.MODELS[model] = ("dashscope", model, {"enable_thinking": False, "temperature": 0}, pin, pout)
    todo = [r for r in rows if r["query_id"] not in done]
    lock, stop = threading.Lock(), threading.Event()
    total = [spent]
    n_run = [0]
    t0 = time.time()

    def call(msgs):
        """pc.chat, but DashScope's transient 429 'insufficient_quota' (seen once mid-run while other jobs
        shared the key) is retried for up to ~10 minutes before the job stops cleanly."""
        for attempt in range(10):
            try:
                return pc.chat(model, msgs, max_tokens=20)
            except RuntimeError as e:
                if "credit/quota" not in str(e) or stop.is_set():
                    raise
                print(f"quota refusal (attempt {attempt + 1}/10), waiting 60 s: {str(e)[:120]}", flush=True)
                time.sleep(60)
        stop.set()
        print("STOP: quota refusals persisted", flush=True)
        return None

    def one(row):
        if stop.is_set():
            return None
        msgs = row["messages"]
        first = call(msgs)
        if first is None:
            return None
        if first["served_model"] and first["served_model"] != model:
            stop.set()
            raise RuntimeError(f"served model {first['served_model']} != {model}")
        p, why = vb.parse_probability(first["text"])
        rec = {"query_id": row["query_id"], "pair_id": row["pair_id"], "form": row["form"], "arm": arm,
               "model": model, "split": split, "prompt_sha256": row["prompt_sha256"], "raw": first["text"],
               "served_model": first["served_model"], "usage": first["usage"], "errors": first["errors"][-3:],
               "n_errors": len(first["errors"]), "finish_reason": first["finish_reason"], "parsed_first": p,
               "reason_first": why, "retried": False, "raw_retry": None, "p": p, "reason": why}
        cost = cost_of(model, first["usage"])
        if p is None:
            second = call(msgs + [{"role": "assistant", "content": first["text"] or ""},
                                  {"role": "user", "content": vb.RETRY_REMINDER}])
            if second is None:
                return None
            p2, why2 = vb.parse_probability(second["text"])
            cost += cost_of(model, second["usage"])
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
                total[0] += rec["cost"]; n_run[0] += 1
                if total[0] >= cap:
                    print("STOP: ledger reached cap", flush=True); stop.set()
                if n_run[0] % 1000 == 0:
                    rate = n_run[0] / (time.time() - t0)
                    print(f"{len(done) + n_run[0]}/{len(rows)} ledger ${total[0]:.3f} {rate:.1f} calls/s", flush=True)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    keep = {r["query_id"] for r in rows}
    recs = [r for r in recs if r["query_id"] in keep]
    print(f"{'STOPPED ' if stop.is_set() else ''}{model} {arm} {split}: logged {len(recs)}/{len(rows)}, "
          f"retried {sum(r['retried'] for r in recs)}, missing {sum(r['p'] is None for r in recs)}, "
          f"job ${sum(r['cost'] for r in recs):.3f}, ledger ${ledger(out):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "project", "run"])
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--model", choices=sorted(PRICES))
    ap.add_argument("--arm", choices=["Q", "P", "L"])
    ap.add_argument("--pairs", type=int, help="first N pairs of the seeded order (default all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=float, default=25.0)
    ap.add_argument("--key-secret", default="dashscope-api-key-01")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, a.out, a.procs)
    else:
        run(a.model, a.arm, a.split, a.out, a.pairs, min(a.workers, 12), a.cap, a.key_secret, a.cmd == "project")


if __name__ == "__main__":
    main()
