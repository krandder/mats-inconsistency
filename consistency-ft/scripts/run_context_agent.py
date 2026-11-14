#!/usr/bin/env python3
"""Context agent: one GPT-6 Luna session per pair picks verbatim Manifold evidence as of t.

The agent (codex-any exec, model gpt-6-luna, web search and shell off) can only reach
Manifold through the as-of MCP server (training/asof_mcp.py), which fixes t from the
environment and enforces it in SQL.  It must answer with a JSON list of items that point
at store records (markets, comments, description excerpts, price histories); it writes no
summaries or probabilities.  Every item is validated against the store (record exists
before t, quote is an exact excerpt of the text as of t); failures are dropped.  The
context text is then rendered deterministically from the surviving items, capped at
``--budget`` tokens, and shared by every form and model for the pair.

No silent model fallback: a model error is recorded as a failed pair, never retried on
another model.  Output: <out>/<split>.jsonl (resumable), one record per pair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import asof  # noqa: E402

CODEX = "/home/agent/.local/bin/codex-any"
MODEL = "gpt-6-luna"
DATASET = Path("/home/agent/data/jev-market-context-open-20260923")
OUT = Path("/home/agent/data/manifold-asof-20260923/context-agent")
MCP = ROOT / "training" / "asof_mcp.py"
TRANSIENT = ("account_unavailable", "503", "502", "429", "rate limit", "overloaded", "stream disconnected",
             "Service Unavailable", "timed out")

RULE = ("You prepare evidence files for a forecasting test. Use only the manifold-asof tools. Do not browse, "
        "do not use other tools, and do not use your own memory of events. Reply with the requested JSON only.")

PROMPT = """You are preparing the evidence file for a forecasting TEST that other models will take.
The test is dated {date}. The forecasters will see only the evidence you select, and they will be asked
about two questions (A and B), including how they relate to each other (for example "A and B", "A or not B").

Be fair. Act as if you do not know anything that happened after {date}: do not use any memory of how these
events or markets turned out, and do not let such memory steer what you pick. Include the relevant evidence on
both sides, regardless of which way it points. Prefer completeness and balance over persuasion.

Question A
{a}

Question B
{b}

You have tools that return Manifold data exactly as it stood just before {date} (nothing later exists in them):
search_markets, get_market, price_history, trades, comments, related_markets. Look at both markets, their
descriptions and comments, and search for other markets that bear on A, on B, or on how A and B relate
(the same event, conditions, correlated events, earlier or later versions of the same question).

Reply with one JSON object and nothing else:
{{"items": [ ... ]}}
Each item is exactly one of:
  {{"kind": "market", "market_id": "<id>"}}                       include a market (its title and price at the date are added automatically)
  {{"kind": "price_history", "market_id": "<id or id:answer>"}}  include that market's price history
  {{"kind": "comment", "market_id": "<id>", "comment_id": "<id>", "quote": "<exact excerpt of the comment>"}}
  {{"kind": "description", "market_id": "<id>", "quote": "<exact excerpt of the market description>"}}
Rules: quotes must be copied exactly, character for character, from the tool output (one contiguous excerpt each;
no ellipses, no paraphrase). Write nothing of your own: no summaries, notes, probabilities or opinions.
Put the most relevant items first. At most 40 items and about 15,000 characters of quotes in total.
The two question markets ({a_id} and {b_id}) are always included; add their comments and description excerpts
only if relevant."""


def question_text(store: asof.AsOf, prop: dict, t: int) -> str:
    cid, aid = asof.split_prop(prop["id"])
    m = store.get_market(cid, t)
    lines = [f"Market id: {prop['id']}", f"Question: {m['question'] if m else prop['parent_question']}"]
    if aid:
        ans = [a for a in (m or {}).get("answers", []) if a["id"] == aid]
        lines.append(f"Option: {ans[0]['text'] if ans else prop.get('answer_text')}")
        lines.append("This proposition is true if the market resolves YES for this option.")
    return "\n".join(lines)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def get_comment(store: asof.AsOf, market_id: str, comment_id: str, t: int) -> dict | None:
    """The comment as shown by AsOf.comments at t (same SQL cutoff), or None."""
    mid = store._mid(market_id, t)
    if mid is None:
        return None
    row = store.db.execute(asof._COMMENTS.replace("WHERE c.mid = :mid", "WHERE c.mid = :mid AND c.id = :cid"),
                           {"mid": mid, "t": t, "limit": 1, "offset": 0, "cid": comment_id}).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "time": row["time"], "date": asof.iso(row["time"]), "user": row["username"],
            "text": row["text"] or "", "bet_outcome": row["bet_outcome"]}


def validate(store: asof.AsOf, items: list, t: int) -> tuple[list[dict], list[dict]]:
    kept, dropped, seen = [], [], set()
    for it in items if isinstance(items, list) else []:
        reason = None
        if not isinstance(it, dict) or it.get("kind") not in ("market", "price_history", "comment", "description"):
            reason = "bad item"
        else:
            kind, mid = it["kind"], str(it.get("market_id") or "")
            key = (kind, mid, it.get("comment_id"), norm(it.get("quote") or ""))
            m = store.get_market(asof.split_prop(mid)[0], t, answers=bool(asof.split_prop(mid)[1]))
            if key in seen:
                reason = "duplicate"
            elif m is None:
                reason = "market not in store before t"
            elif kind == "price_history" and asof.split_prop(mid)[1] and not any(
                    a["id"] == asof.split_prop(mid)[1] for a in m.get("answers") or []):
                reason = "option not in store before t"
            elif kind == "comment":
                c = get_comment(store, asof.split_prop(mid)[0], str(it.get("comment_id") or ""), t)
                q = norm(it.get("quote") or "")
                if c is None:
                    reason = "comment not in store before t"
                elif not q or q not in norm(c["text"]):
                    reason = "quote not an exact excerpt of the comment"
            elif kind == "description":
                q = norm(it.get("quote") or "")
                if not q or q not in norm(m["description"]):
                    reason = "quote not an exact excerpt of the description at t"
            seen.add(key)
        (dropped if reason else kept).append({**(it if isinstance(it, dict) else {"raw": it}), **(
            {"reason": reason} if reason else {})})
    return kept, dropped


def market_line(store: asof.AsOf, market_id: str, t: int) -> str:
    cid, aid = asof.split_prop(market_id)
    m = store.get_market(cid, t)
    head = f"[market {cid}] {m['question']} ({asof._status(m)}"
    if m["answers"]:
        top = sorted(m["answers"], key=lambda x: (-(x["prob"] or 0), x["id"]))[:6]
        if aid:
            top = [a for a in m["answers"] if a["id"] == aid] + [a for a in top if a["id"] != aid][:5]
        opts = "; ".join(f"\"{asof.clip(a['text'], 70)}\" {asof.pct(a['prob'])}" for a in top)
        return f"{head}; {m['n_trades']} trades) Options at the date: {opts}."
    return f"{head}; price {asof.pct(m['prob'])}; {m['n_trades']} trades)"


def render(store: asof.AsOf, pair: dict, kept: list[dict], t: int, budget: int) -> tuple[str, list[dict]]:
    """Deterministic text from validated items, in the agent's order; pair markets first."""
    own = [{"kind": "market", "market_id": pair["p_id"]}, {"kind": "market", "market_id": pair["q_id"]}]
    items = own + [it for it in kept if not (it["kind"] == "market" and asof.split_prop(it["market_id"])[0] in
                                            {asof.split_prop(pair["p_id"])[0], asof.split_prop(pair["q_id"])[0]})]
    head = [f"# Evidence selected from Manifold as of {asof.iso(t, minutes=True)}",
            "Verbatim items (market titles with prices at that time, comments and description excerpts). "
            "Nothing after that time is included."]
    lines, used = [], []
    questions: dict[str, str] = {}

    def title(mid):
        cid = asof.split_prop(mid)[0]
        if cid not in questions:
            questions[cid] = store.get_market(cid, t, answers=False)["question"]
        return questions[cid]

    for it in items:
        kind, mid = it["kind"], it["market_id"]
        if kind == "market":
            text = "- " + market_line(store, mid, t)
        elif kind == "price_history":
            hist = store.price_history(mid, t, points=8)
            text = f"- [price history {mid}] {title(mid)}: " + asof.history_text(hist)
        elif kind == "comment":
            c = get_comment(store, asof.split_prop(mid)[0], it["comment_id"], t)
            text = f"- [comment {c['id']}, {c['date']}, {c['user']}, on \"{asof.clip(title(mid), 90)}\"] \"{norm(it['quote'])}\""
        else:
            text = f"- [description of \"{asof.clip(title(mid), 90)}\"] \"{norm(it['quote'])}\""
        candidate = "\n".join(head + lines + [text]) + "\n"
        if asof.count_tokens(candidate) > budget and len(used) >= 2:
            continue
        lines.append(text)
        used.append(it)
    return "\n".join(head + lines) + "\n", used


def parse_items(text: str | None):
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        obj = json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    return obj.get("items") if isinstance(obj, dict) else None


def run_codex(prompt: str, t: int, work: Path, effort: str, timeout: float) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work / "empty").mkdir(exist_ok=True)
    log = work / "tool-calls.jsonl"
    final = work / "final.txt"
    for p in (log, final):
        if p.exists():
            p.unlink()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENAI_", "XAI_", "ZAI_", "ZHIPU_", "CODEX_API_"))}
    env.pop("TMUX", None)
    env.update(FLEET_ACTOR="consistency-context-agent", FUTARCHY_AGENT_NAME="consistency-context-agent",
               AI_AGENT_NAME="consistency-context-agent", TMPDIR=str(work))
    mcp_env = {"ASOF_T": str(t), "ASOF_STORE": str(asof.DEFAULT_STORE), "ASOF_LOG": str(log)}
    cmd = [CODEX, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "--sandbox", "read-only",
           "-C", str(work / "empty"), "--model", MODEL, "-c", "model_reasoning_effort=" + json.dumps(effort),
           "-c", 'web_search="disabled"', "-c", "features.shell_tool=false", "-c", "project_doc_max_bytes=0",
           "-c", "features.multi_agent_v2=false", "-c", 'approval_policy="never"',
           "-c", "developer_instructions=" + json.dumps(RULE),
           "-c", 'mcp_servers.asof.command="python3"', "-c", "mcp_servers.asof.args=" + json.dumps([str(MCP)]),
           "-c", 'mcp_servers.asof.default_tools_approval_mode="approve"',
           "-c", "mcp_servers.asof.env={" + ", ".join(f"{k}={json.dumps(v)}" for k, v in mcp_env.items()) + "}",
           "-o", str(final), "-"]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env,
                              cwd=work / "empty")
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = exc.stdout or "", exc.stderr or "", "timeout"
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
    (work / "stdout.jsonl").write_text(stdout or "")
    (work / "stderr.txt").write_text(stderr or "")
    events = [json.loads(x) for x in (stdout or "").splitlines() if x.startswith("{")]
    usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
    other_tools = [e["item"].get("tool") or e["item"].get("type") for e in events
                   if e.get("type") == "item.completed" and e.get("item", {}).get("type") in
                   ("command_execution", "web_search", "file_change")
                   or (e.get("item", {}).get("type") == "mcp_tool_call" and e["item"].get("server") not in ("asof", "codex"))]
    errors = [e["item"].get("message") for e in events if e.get("type") == "item.completed"
              and e.get("item", {}).get("type") == "error"
              and "unrecognized configuration" not in (e["item"].get("message") or "")
              and "service tier" not in (e["item"].get("message") or "")
              and "Model metadata" not in (e["item"].get("message") or "")]
    errors += [e.get("error", {}).get("message") if isinstance(e.get("error"), dict) else str(e.get("message"))
               for e in events if e.get("type") in ("turn.failed", "error")]
    text = final.read_text() if final.exists() else None
    calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return {"text": text, "exit": code, "usage": usage, "errors": errors, "other_tools": other_tools,
            "tool_calls": [{"tool": c["tool"], "args": c["args"]} for c in calls],
            "elapsed": round(time.monotonic() - started, 1), "stderr_tail": (stderr or "")[-1500:]}


def load(split: str, dataset: Path):
    root = dataset / split
    pairs = [json.loads(x) for x in (root / "pairs.jsonl").read_text().splitlines() if x.strip()]
    props = {p["id"]: p for p in (json.loads(x) for x in (root / "propositions.jsonl").read_text().splitlines()
                                  if x.strip())}
    return pairs, props


def one_pair(pair, props, out_dir: Path, effort: str, timeout: float, budget: int) -> dict:
    store = asof.AsOf()  # one connection per pair/thread
    t = asof.to_ms(asof.pair_as_of(pair, props))
    date = asof.iso(t, minutes=True)
    prompt = PROMPT.format(date=date, a=question_text(store, props[pair["p_id"]], t),
                           b=question_text(store, props[pair["q_id"]], t), a_id=pair["p_id"], b_id=pair["q_id"])
    work = out_dir / "sessions" / pair["pair_id"]
    res = None
    for attempt in range(3):
        res = run_codex(prompt, t, work / f"attempt-{attempt}", effort, timeout)
        transient = any(any(x in (e or "") for x in TRANSIENT) for e in res["errors"]) or res["exit"] == "timeout"
        if not transient:
            break
        time.sleep(30 * (attempt + 1))
    items = parse_items(res["text"])
    kept, dropped = validate(store, items or [], t)
    context, used = render(store, pair, kept, t, budget)
    return {"pair_id": pair["pair_id"], "p_id": pair["p_id"], "q_id": pair["q_id"], "t": t, "as_of": date,
            "model": MODEL, "effort": effort, "ok": items is not None and not res["other_tools"],
            "n_items": len(items or []), "kept": kept, "dropped": dropped, "used": used,
            "context": context, "context_tokens": asof.count_tokens(context),
            "tool_calls": res["tool_calls"], "other_tools": res["other_tools"], "errors": res["errors"],
            "usage": res["usage"], "elapsed": res["elapsed"], "exit": res["exit"], "raw": res["text"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--timeout", type=float, default=1500)
    ap.add_argument("--budget", type=int, default=8000)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{a.split}.jsonl"
    done = {json.loads(x)["pair_id"] for x in out_file.read_text().splitlines()} if out_file.exists() else set()
    pairs, props = load(a.split, Path(a.dataset))
    pairs = sorted(pairs, key=lambda p: p["pair_id"])[: a.limit or None]
    todo = [p for p in pairs if p["pair_id"] not in done]
    lock = threading.Lock()
    print(f"{len(todo)} pairs to run ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(one_pair, p, props, out_dir, a.effort, a.timeout, a.budget): p for p in todo}
        for f in as_completed(futs):
            p = futs[f]
            try:
                rec = f.result()
            except Exception as exc:  # recorded, never silently skipped
                rec = {"pair_id": p["pair_id"], "ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            with lock, out_file.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{rec['pair_id']} ok={rec.get('ok')} items={rec.get('n_items')} kept={len(rec.get('kept', []))} "
                  f"dropped={len(rec.get('dropped', []))} tokens={rec.get('context_tokens')} "
                  f"elapsed={rec.get('elapsed')}", flush=True)


if __name__ == "__main__":
    main()
