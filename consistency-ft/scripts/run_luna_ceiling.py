"""Frontier-model ceiling: all 12 Boolean forms of each pair from one Codex session.

Uses the same question blocks, as-of line and plain-language statements as the
Qwen3-32B verbalized baseline (training/verbalized_baseline.py, variant
v6_direct), but asks for the twelve statements at once and reads strict JSON
{"forms": {"<form>": p, ...}}.  One retry (a fresh session with a reminder)
on invalid JSON or missing/invalid keys; forms still missing stay None.

Calls go through codex-any (General's subscriptions via the proxy), reasoning
effort medium, web search and shell disabled.  Every finished pair is appended
to <out>/log-<split>.jsonl, so the run resumes after a restart.  Labels are read
only to be stored for offline scoring; they never enter a prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.supervised_pairs import FORM_ORDER, load_split
from training import verbalized_baseline as vb

CODEX = "/home/agent/.local/bin/codex-any"
RULE = (vb.BASE_SYSTEM + " Answer from your existing knowledge only. Do not use tools, browse, read files, "
        "consult memory, or delegate. Reply with the requested JSON object and nothing else.")
ANSWER_JSON = ("Reply with strict JSON only, no other text and no code fence, of exactly this shape:\n"
               '{"forms": {"P": 0.xx, "NOT P": 0.xx, ...}}\n'
               "with all 12 keys exactly as written above and each value a single number between 0.01 and 0.99.")
RETRY_REMINDER = ("Your previous reply was not valid: it must be strict JSON of the form "
                  '{"forms": {...}} with all 12 keys exactly as listed and each value a number between 0.01 and 0.99. '
                  "Reply with that JSON only.")
TRANSIENT = ("account_unavailable", "503", "502", "429", "rate limit", "overloaded", "stream disconnected",
             "Service Unavailable", "timed out")


def pair_prompt(example) -> str:
    p_row = example.propositions[example.pair["p_id"]]
    q_row = example.propositions[example.pair["q_id"]]
    today = vb.as_of_date(min(p_row["closing_week"], q_row["closing_week"]))
    statements = "\n".join(f'"{form}": {vb.plain_statement(form)}' for form in FORM_ORDER)
    parts = [f"Assume today is {today}. Use only information available up to that date; do not use knowledge of later events.",
             "Question A\n" + vb.question_block(p_row),
             "Question B\n" + vb.question_block(q_row),
             "For each statement below, estimate the probability that it is true. Each line gives the key, then the statement.\n"
             + statements,
             ANSWER_JSON]
    return "\n\n".join(parts)


def parse_forms(text: str | None) -> tuple[dict[str, float], str]:
    """Strict parse. Returns ({form: p} for valid keys only, reason)."""
    if not text:
        return {}, "empty"
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return {}, "invalid_json"
    if not isinstance(obj, dict) or not isinstance(obj.get("forms"), dict):
        return {}, "no_forms_object"
    got = {}
    for form in FORM_ORDER:
        v = obj["forms"].get(form)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
            got[form] = float(v)
    extra = set(obj["forms"]) - set(FORM_ORDER)
    if len(got) == 12 and not extra:
        return got, "ok"
    return got, f"missing_or_invalid_{12 - len(got)}" + ("_extra_keys" if extra else "")


def call(prompt: str, model: str, effort: str, empty: Path, *, max_wait: float = 3600) -> dict:
    """One Codex session. Backs off on transient proxy errors; returns text, usage, attempts."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "XAI_", "ZAI_", "ZHIPU_", "CODEX_API_"))}
    env.pop("TMUX", None)
    env["PATH"] = "/home/agent/.local/bin:" + env.get("PATH", "/usr/bin:/bin")
    env.update(FLEET_ACTOR="consistency-luna-ceiling", FUTARCHY_AGENT_NAME="consistency-luna-ceiling",
               AI_AGENT_NAME="consistency-luna-ceiling")
    start, delay, attempts, errors = time.time(), 20.0, 0, []
    while True:
        attempts += 1
        with tempfile.TemporaryDirectory(dir=empty.parent) as td:
            out = Path(td) / "last.txt"
            cmd = [CODEX, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
                   "-C", str(empty), "--model", model, "-c", "model_reasoning_effort=" + json.dumps(effort),
                   "-c", 'service_tier="default"', "-c", "project_doc_max_bytes=0", "-c", 'web_search="disabled"',
                   "-c", "features.multi_agent_v2=false", "-c", "features.shell_tool=false",
                   "-c", "developer_instructions=" + json.dumps(RULE), "-o", str(out), "-"]
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, timeout=900)
                stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired:
                stdout, stderr, rc = "", "timed out", -1
            text = out.read_text() if out.exists() else None
        usage, fatal, tool_items = None, None, 0
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage")
            elif ev.get("type") in ("error", "turn.failed"):
                fatal = json.dumps(ev)[:500]
            elif ev.get("type") == "item.completed" and ev["item"].get("type") not in ("agent_message", "reasoning", "error"):
                tool_items += 1
        if usage is not None and text is not None:
            return {"text": text, "usage": usage, "attempts": attempts, "tool_items": tool_items, "errors": errors}
        err = (fatal or "") + " " + stderr[-500:]
        errors.append({"rc": rc, "err": err.strip()[:500]})
        if "not supported" in err or "model_not_found" in err:
            raise RuntimeError("model rejected: " + err[:300])  # never substitute a model
        transient = any(t.lower() in err.lower() for t in TRANSIENT) or rc != 0
        if not transient or time.time() - start > max_wait:
            return {"text": text, "usage": usage, "attempts": attempts, "tool_items": tool_items, "errors": errors}
        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 2, 300)


def run_pair(ex, model, effort, empty) -> dict:
    prompt = pair_prompt(ex)
    first = call(prompt, model, effort, empty)
    got1, why1 = parse_forms(first["text"])
    rec = {"pair_id": ex.pair["pair_id"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
           "raw": first["text"], "usage": first["usage"], "attempts": first["attempts"], "tool_items": first["tool_items"],
           "errors": first["errors"], "reason_first": why1, "parsed_first": got1, "retried": False,
           "raw_retry": None, "usage_retry": None, "parsed_retry": None, "reason_retry": None}
    final = dict(got1)
    if why1 != "ok":
        second = call(prompt + "\n\n" + RETRY_REMINDER, model, effort, empty)
        got2, why2 = parse_forms(second["text"])
        rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], parsed_retry=got2,
                   reason_retry=why2, errors_retry=second["errors"])
        final = {**got1, **got2} if why2 != "ok" else got2
    rec["final"] = final
    return rec


def to_rows(ex, rec) -> list[dict]:
    rows = []
    for form in FORM_ORDER:
        qid = ex.queries[form]["query_id"]
        p = rec["final"].get(form)
        first = rec["parsed_first"].get(form)
        rows.append({"pair_id": ex.pair["pair_id"], "query_id": qid, "form": form,
                     "observed": qid == ex.pair["observed_query_id"], "label": ex.labels[qid],
                     "p_id": ex.pair["p_id"], "q_id": ex.pair["q_id"], "parsed_first": first,
                     "reason_first": rec["reason_first"], "retried": rec["retried"], "p": p,
                     "reason": "ok" if p is not None else (rec["reason_retry"] or rec["reason_first"])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    empty = out / "empty"; empty.mkdir(exist_ok=True)
    examples = load_split(args.data_root, args.split, labels_file="full-label-targets.jsonl")
    if args.limit:
        examples = examples[:args.limit]
    log = out / f"log-{args.split}.jsonl"
    done = {json.loads(l)["pair_id"] for l in log.read_text().splitlines()} if log.exists() else set()
    todo = [ex for ex in examples if ex.pair["pair_id"] not in done]
    print(f"{args.split}: {len(examples)} pairs, {len(done)} done, {len(todo)} to run", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_pair, ex, args.model, args.effort, empty): ex for ex in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rec.update(model=args.model, effort=args.effort, split=args.split, finished_at=time.time())
            with lock, log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if i % 25 == 0 or i == len(todo):
                print(f"{args.split}: {i}/{len(todo)} (last {rec['pair_id']}: {rec['reason_first']})", flush=True)
    recs = {}
    for l in log.read_text().splitlines():
        r = json.loads(l)
        recs[r["pair_id"]] = r
    rows = [row for ex in examples if ex.pair["pair_id"] in recs for row in to_rows(ex, recs[ex.pair["pair_id"]])]
    with (out / f"predictions-{args.split}.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows, missing {sum(r['p'] is None for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
