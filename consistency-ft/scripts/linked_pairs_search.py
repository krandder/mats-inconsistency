#!/usr/bin/env python3
"""Linked-pairs search experiment (docs/linked-pairs-search-20261001.md): forecast calls with a runtime Exa tool.

Every call: claude -p, pinned to claude-opus-5-5 at effort medium (as MiniBench), through Prime, no built-in
tools; one MCP tool, exa_search (scripts/exa_mcp_server.py), whose cutoff = t is enforced in the server
(MiniBench strict leak guard). The model writes its own queries in every call; at most 4 per call.

  linked_pairs_search.py smoke            one non-sample pair (OUT/smoke-pair.json), all 16 calls
  linked_pairs_search.py run [--workers]  the 100 selected pairs in run_order; stops at the cap
  linked_pairs_search.py spend            print the spend so far (LLM list price, Exa cash)

Per pair, 16 calls: the 12 Boolean forms once each (sample 0) plus samples 1 and 2 of P and of Q.
Outputs under OUT: calls.jsonl (one line per call: prompt sha, parsed answer, cost, usage, tool use),
raw/<call_id>.json (the compact transcript), queries.jsonl (written by the MCP server), exa_cost.jsonl.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import verbalized_baseline as vb  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

OUT = Path("/home/agent/data/linked-pairs-20261001")
LEDGER = Path("/home/agent/workspaces/calibrated-ai/BUDGET-20261001.md")
SERVER = ROOT / "scripts" / "exa_mcp_server.py"
CLAUDE_BIN = "/home/agent/.local/bin/claude"
PRIME = "http://127.0.0.1:7800"
AGENT = "calibrated-ai"
MODEL = "claude-opus-5-5"
EFFORT = "medium"
CAP_USD = 250.0
MAX_QUERIES = 4
NUM_RESULTS = 6
HIGHLIGHT_CHARS = 1000
EXTRA_SAMPLES = 2          # samples 1..2 of P and of Q on top of the lattice's sample 0
EST_CALL = 0.12            # reserve per call (LLM + Exa) when checking the cap before starting a call
LOCK = threading.Lock()


class Stop(Exception):
    """Model unavailable or substituted, or the cap is reached: stop the run, never switch models."""


# --- spend ---------------------------------------------------------------------------------------------------

def _sum(path, key="cost_usd"):
    tot = 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                tot += float(json.loads(line).get(key) or 0.0)
            except ValueError:
                pass
    return tot


def spend():
    """(llm list-price USD, exa cash USD) from OUT/costs.jsonl (LLM calls) and OUT/exa_cost.jsonl (MCP server)."""
    return _sum(OUT / "costs.jsonl"), _sum(OUT / "exa_cost.jsonl")


class Budget:
    def __init__(self, cap=CAP_USD):
        self.cap = cap
        self.reserved = 0.0

    def check(self, est=EST_CALL):
        with LOCK:
            llm, exa = spend()
            if llm + exa + self.reserved + est > self.cap:
                raise Stop(f"cap ${self.cap:.0f} reached: llm ${llm:.2f} + exa ${exa:.2f} + in flight ${self.reserved:.2f}")
            self.reserved += est

    def release(self, est=EST_CALL):
        with LOCK:
            self.reserved = max(0.0, self.reserved - est)

    def add(self, kind, usd, purpose, **kw):
        with LOCK, open(OUT / "costs.jsonl", "a") as f:
            f.write(json.dumps({"ts": now(), "kind": kind, "cost_usd": round(float(usd or 0), 6),
                                "purpose": purpose, **kw}) + "\n")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ledger_row(desc):
    """Append a row to the experiment's budget ledger: LLM list price and Exa cash, with the running total."""
    llm, exa = spend()
    with LOCK, open(LEDGER, "a") as f:
        f.write(f"| {datetime.now(timezone.utc).strftime('%m-%d %H:%MZ')} | {desc} | ${llm:.2f} | ${exa:.2f} | "
                f"${llm + exa:.2f} |\n")


# --- one Claude call -----------------------------------------------------------------------------------------

_EXA_KEY = {}


def exa_key():
    if "k" not in _EXA_KEY:
        out = subprocess.run(["secret", "get", "exa-api-key"], capture_output=True, text=True, timeout=60)
        if out.returncode != 0 or not out.stdout.strip():
            raise Stop("secret get [removed] failed")
        _EXA_KEY["k"] = out.stdout.strip()
    return _EXA_KEY["k"]


def base_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE", "FLEET_", "AI_AGENT", "FUTARCHY_", "TMUX", "EXA_"))}
    env.update(HOME="/home/agent", PATH="/home/agent/.local/bin:/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
               ANTHROPIC_BASE_URL=PRIME, ANTHROPIC_AUTH_TOKEN=f"token-proxy-managed:{AGENT}",
               CLAUDE_CODE_NO_MODEL_FALLBACK="1")
    return env


def compact_stream(stdout):
    """(result record, compact transcript): assistant text, thinking text and tool calls; tool results are kept
    only as their length (the full results are in queries.jsonl)."""
    result, turns = None, []
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = d.get("type")
        if t == "result":
            result = d
        elif t == "assistant":
            for c in d.get("message", {}).get("content", []):
                if c.get("type") == "text":
                    turns.append({"role": "assistant", "text": c.get("text")})
                elif c.get("type") == "thinking":
                    if c.get("thinking"):
                        turns.append({"role": "thinking", "text": c.get("thinking")})
                elif c.get("type") == "tool_use":
                    turns.append({"role": "tool_use", "name": c.get("name"), "input": c.get("input")})
        elif t == "user":
            for c in d.get("message", {}).get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    txt = c.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    turns.append({"role": "tool_result", "chars": len(txt or ""), "is_error": c.get("is_error")})
        elif t == "system" and d.get("subtype") == "init":
            turns.append({"role": "init", "model": d.get("model"), "tools": d.get("tools"),
                          "mcp": d.get("mcp_servers")})
    return result, turns


def claude_once(system, user, effort, mcp=None, timeout=600):
    env = base_env()
    if mcp:
        env["EXA_API_KEY"] = exa_key()
        cfg = {"mcpServers": {"exa": {"type": "stdio", "command": "/usr/bin/python3",
                                      "args": [str(SERVER), "--cutoff", mcp["cutoff"], "--call-id", mcp["call_id"],
                                               "--log", str(OUT / "queries.jsonl"),
                                               "--cost-path", str(OUT / "exa_cost.jsonl"),
                                               "--max-queries", str(MAX_QUERIES), "--num", str(NUM_RESULTS),
                                               "--highlight-chars", str(HIGHLIGHT_CHARS)]}}}
        tools = ["--strict-mcp-config", "--mcp-config", json.dumps(cfg), "--allowedTools", "mcp__exa__exa_search"]
    else:
        tools = ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    cmd = [CLAUDE_BIN, "-p", "--tools", "", *tools, "--settings", '{"disableAllHooks":true}', "--setting-sources", "",
           "--disable-slash-commands", "--model", MODEL,
           "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--effort", effort,
           "--system-prompt", system]
    (OUT / "tmp").mkdir(exist_ok=True)
    cwd = tempfile.mkdtemp(prefix="lp-", dir=str(OUT / "tmp"))
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    d, turns = compact_stream(p.stdout)
    if d is None:
        raise RuntimeError(f"no result record rc={p.returncode}: {p.stdout[-200:]!r} {p.stderr[-300:]!r}")
    used = sorted((d.get("modelUsage") or {}).keys())
    if d.get("is_error"):
        msg = f"{str(d.get('result'))[:300]} status={d.get('api_error_status')}"
        if any(k in msg.lower() for k in ("model_not_found", "not_found_error", "invalid model", "does not exist")):
            raise Stop(f"{MODEL} rejected: {msg}")
        raise RuntimeError(msg)
    if used != [MODEL]:
        raise Stop(f"asked for {MODEL}, answer came from {used}")
    u = d.get("usage") or {}
    usage = {k: u.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                   "output_tokens")}
    usage["thinking_tokens"] = (u.get("output_tokens_details") or {}).get("thinking_tokens")
    return {"text": d.get("result") or "", "model": used[0], "usage": usage, "cost": float(d.get("total_cost_usd") or 0.0),
            "num_turns": d.get("num_turns"), "duration_ms": d.get("duration_ms"), "turns": turns}


def claude_call(system, user, effort=EFFORT, mcp=None, purpose="", tries=5, **_):
    delay, errors = 10.0, []
    for attempt in range(tries):
        try:
            r = claude_once(system, user, effort, mcp=mcp)
            r["transient_errors"] = errors
            return r
        except Stop:
            raise
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:300])
            if attempt + 1 == tries:
                raise
            time.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 240)


# --- prompts -------------------------------------------------------------------------------------------------

RESEARCH = ("You have one tool, exa_search, which searches news and web pages published before today. Before you "
            f"answer, research the question(s) with it: write your own search queries (at most {MAX_QUERIES} searches).")
ANSWER = ("When you are done, reply with exactly one line of this form and nothing else:\nProbability: 0.xx\n"
          "(a single number between 0.01 and 0.99).")


def row_of(q):
    """The v7 question-block row: question, option, criteria and the scheduled date (never the actual end, which
    would reveal an early resolution)."""
    return {"parent_question": q["title"], "answer_text": q.get("option") or "", "criteria_text": q.get("criteria") or "",
            "closing_week": q["shown_date"][:10]}


def block(row, name):
    d = row["closing_week"]
    text = vb.question_block_v7(row, name)
    assert f"Closes in the week of {d}." in text
    return text.replace(f"Closes in the week of {d}.", f"Scheduled close date: {d}.")


def user_prompt(pair, form):
    """v7b_explicit (prompt_negation_fix's winner) plus the research instruction; singles show only their own
    question; "today" is the pair's t for all 16 calls."""
    p_row, q_row = row_of(pair["P"]), row_of(pair["Q"])
    op, a, b = vb._parts(form)
    show_a, show_b = (True, True) if op else (bool(a), bool(b))
    parts = [f"Assume today is {pair['t'][:10]}. Use only information available up to that date; do not use "
             "knowledge of later events."]
    if show_a:
        parts.append(block(p_row, "A"))
    if show_b:
        parts.append(block(q_row, "B"))
    statement = vb.explicit_statement(form, p_row, q_row, negation_note=True)
    parts.append(f"Statement: {statement}\nWhat is the probability that this statement is true?")
    parts.append(RESEARCH)
    parts.append(ANSWER)
    return "\n\n".join(parts)


def calls_for(pair):
    out = [(form, 0) for form in FORM_ORDER]
    out += [(form, s) for s in range(1, EXTRA_SAMPLES + 1) for form in ("P", "Q")]
    return out


def call_id(pair, form, sample):
    tag = form.replace("(", "").replace(")", "").replace(" ", "_")
    return hashlib.sha256(f"{pair['pair_id']}|{form}|{sample}".encode()).hexdigest()[:12] + "-" + tag + f"-s{sample}"


# --- running -------------------------------------------------------------------------------------------------

def done_calls(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("prob") is not None or r.get("final"):
                out[r["call_id"]] = r
    return out


def one_call(pair, form, sample, budget, log_path, stop_state):
    cid = call_id(pair, form, sample)
    if stop_state.get("stop"):
        return None
    try:
        budget.check()
    except Stop as e:
        stop_state["stop"] = str(e)
        return None
    user = user_prompt(pair, form)
    rec = {"ts": now(), "call_id": cid, "pair_id": pair["pair_id"], "form": form, "sample": sample, "t": pair["t"],
           "prompt_sha256": hashlib.sha256((vb.BASE_SYSTEM + "\n" + user).encode()).hexdigest()}
    try:
        r = claude_call(vb.BASE_SYSTEM, user, mcp={"cutoff": pair["t"], "call_id": cid}, purpose="forecast")
        p, why = vb.parse_probability(r["text"])
        cost, retry = r["cost"], None
        if p is None:  # the protocol's one retry with the reminder
            r2 = claude_call(vb.BASE_SYSTEM, user + "\n\n" + vb.RETRY_REMINDER,
                             mcp={"cutoff": pair["t"], "call_id": cid + "-retry"}, purpose="forecast-retry")
            p, why = vb.parse_probability(r2["text"])
            cost += r2["cost"]
            retry = {"text": r2["text"][-500:], "cost": r2["cost"], "turns": r2["turns"]}
        budget.add("llm", cost, "forecast", call_id=cid)
        n_search = sum(1 for x in r["turns"] if x["role"] == "tool_use")
        rec.update(prob=p, parse=why, text_tail=r["text"][-300:], cost=cost, usage=r["usage"],
                   num_turns=r["num_turns"], n_tool_calls=n_search, duration_ms=r["duration_ms"],
                   transient_errors=r["transient_errors"], retry=bool(retry), final=True, model=r["model"])
        (OUT / "raw").mkdir(exist_ok=True)
        (OUT / "raw" / f"{cid}.json").write_text(json.dumps({**rec, "user": user, "text": r["text"],
                                                             "turns": r["turns"], "retry": retry}))
    except Stop as e:
        stop_state["stop"] = str(e)
        rec.update(error=f"Stop: {e}", final=False)
    except Exception as e:  # noqa: BLE001
        rec.update(error=str(e)[:300], final=True, prob=None)
    finally:
        budget.release()
    with LOCK, open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_pairs(pairs, log_path, workers, label):
    budget = Budget()
    done = done_calls(log_path)
    todo = [(pr, f, s) for pr in pairs for (f, s) in calls_for(pr) if call_id(pr, f, s) not in done]
    stop_state = {}
    print(f"{label}: {len(todo)} calls to make ({len(done)} done)", flush=True)
    n, t0 = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_call, pr, f, s, budget, log_path, stop_state) for pr, f, s in todo]
        for fu in futs:
            r = fu.result()
            n += 1
            if n % 50 == 0:
                llm, exa = spend()
                print(f"{n}/{len(todo)} calls, {time.time() - t0:.0f}s, llm ${llm:.2f} exa ${exa:.2f}", flush=True)
            if n % 200 == 0:
                ledger_row(f"{label}: {n}/{len(todo)} calls")
    if stop_state.get("stop"):
        print("STOPPED:", stop_state["stop"], flush=True)
    ledger_row(f"{label}: finished ({n} calls this pass){'; STOPPED: ' + stop_state['stop'] if stop_state.get('stop') else ''}")
    return stop_state.get("stop")


def cmd_smoke(a):
    pair = json.loads((OUT / "smoke-pair.json").read_text())
    stop = run_pairs([pair], OUT / "smoke-calls.jsonl", a.workers, "smoke (1 non-sample pair)")
    rows = [json.loads(l) for l in open(OUT / "smoke-calls.jsonl")]
    print(json.dumps({"calls": len(rows), "parsed": sum(r.get("prob") is not None for r in rows),
                      "cost": round(sum(r.get("cost") or 0 for r in rows), 4),
                      "tool_calls": [r.get("n_tool_calls") for r in rows], "stop": stop}))


def cmd_run(a):
    pairs = sorted(map(json.loads, open(OUT / "pairs.jsonl")), key=lambda p: p["run_order"])
    if a.limit:
        pairs = pairs[:a.limit]
    stop = run_pairs(pairs, OUT / "calls.jsonl", a.workers, f"main run ({len(pairs)} pairs)")
    if stop:
        sys.exit(f"stopped: {stop}")


# --- truth (after the run only) -------------------------------------------------------------------------------

BT_TRUTH = Path("/home/agent/workspaces/calibrated-ai/minibench/backtest/bt-scale/truth")
MANIFOLD_DB = "file:/home/agent/data/manifold-asof-20260923/asof.sqlite?mode=ro"


def page_resolution(text):
    """'yes'/'no'/'annulled'/'ambiguous' from a Metaculus question page's text (document.body.innerText), else None.
    A binary question's page shows its status and outcome as consecutive lines: 'RESOLVED' then 'Yes' or 'No'."""
    m = re.search(r"\bRESOLVED\s*\n\s*(Yes|No)\s*\n", text or "")
    if m:
        return m.group(1).lower()
    m = re.search(r"\b(ANNULLED|AMBIGUOUS)\b", text or "")
    return m.group(1).lower() if m else None


def cmd_truth(_a):
    """Outcomes of the selected questions, read only after the main run: MiniBench from the bt-scale truth/ files,
    else the Metaculus post detail (GET only); Manifold from the as-of database. yes/no -> 1/0; anything else
    (annulled, ambiguous, N/A) -> None (the pair is left out and counted)."""
    import sqlite3
    sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/minibench/bot")
    pairs = [json.loads(l) for l in open(OUT / "pairs.jsonl")]
    qs = {x["qid"]: x for p in pairs for x in (p["P"], p["Q"])}
    out, client, con = {}, None, sqlite3.connect(MANIFOLD_DB, uri=True)
    # the API hides a closed question's resolution from accounts that did not forecast it (Metaculus API change of
    # 2026-03-09); the public question page carries it. metaculus-resolutions.jsonl = each sampled question's fields
    # read from its page's embedded data in the fleet browser (read-only); metaculus-pages.jsonl = the page text
    # header ("RESOLVED" then "Yes"/"No"), used as a cross-check for standalone questions.
    page_res = {}
    if (OUT / "metaculus-resolutions.jsonl").exists():
        page_res = {str(r["question_id"]): r for r in map(json.loads, open(OUT / "metaculus-resolutions.jsonl"))}
    pages = {}
    if (OUT / "metaculus-pages.jsonl").exists():
        pages = {str(r["post_id"]): r["text"] for r in map(json.loads, open(OUT / "metaculus-pages.jsonl"))}
    for qid, q in sorted(qs.items()):
        res, how = None, None
        if q["source"] == "manifold":
            if q.get("aid"):
                r = con.execute("select resolution from answers where aid = ?", (q["aid"],)).fetchone()
            else:
                r = con.execute("select resolution from markets where mid = ?", (q["mid"],)).fetchone()
            res, how = (r[0] or "").lower() if r else None, "manifold-db"
        else:
            f = BT_TRUTH / f"{q['question_id']}.json"
            if q["source"] == "minibench" and f.exists():
                res, how = str(json.loads(f.read_text()).get("resolution")).lower(), "bt-scale-truth"
            else:
                from mb.metaculus_read import GetOnlyClient
                client = client or GetOnlyClient()
                d = client.post_detail(q["post_id"])
                cands = [d.get("question")] + list((d.get("group_of_questions") or {}).get("questions") or [])
                for x in cands:
                    if x and int(x.get("id") or -1) == int(q["question_id"]):
                        res = str(x.get("resolution")).lower()
                how = "metaculus-api"
                if res in (None, "none") and str(q["question_id"]) in page_res:
                    pr = page_res[str(q["question_id"])]
                    res, how = (pr.get("resolution") or None), "metaculus-page-data"
                    txt = page_resolution(pages.get(str(q["post_id"]), ""))
                    if txt is not None and txt != res:
                        raise SystemExit(f"page text and page data disagree for {qid}: {txt} vs {res}")
        out[qid] = {"qid": qid, "label": {"yes": 1, "no": 0}.get(res), "resolution": res, "how": how}
    with open(OUT / "truth.jsonl", "w") as f:
        for r in out.values():
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(json.dumps({"questions": len(out), "labels": Counter(str(r["label"]) for r in out.values()),
                      "how": Counter(r["how"] for r in out.values())}))


# --- analysis -------------------------------------------------------------------------------------------------

LO = 0.01
BOOT, BOOT_SEED = 2000, 20261001
NO_GAIN_BAR = -0.01


def clip(p):
    return min(1 - LO, max(LO, p))


def ll(p, y):
    p = clip(p)
    return -math.log(p) if y else -math.log(1 - p)


def brier(p, y):
    return (clip(p) - y) ** 2


def load_answers(path, drop=()):
    """(pair_id, form, sample) -> probability or None; dropped call ids count as missing."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if not r.get("final"):
            continue
        out[(r["pair_id"], r["form"], r["sample"])] = None if r["call_id"] in drop else r.get("prob")
    return out


def arms_for_pair(pr, ans):
    """Per pair: singles under each arm for P and Q, the fitted atoms, and the 3 plain samples per side."""
    import numpy as np
    from coherence_projection import FORM_ATOMS, fit
    s0 = {f: ans.get((pr["pair_id"], f, 0)) for f in FORM_ORDER}
    have = [f for f in FORM_ORDER if s0[f] is not None]
    atoms = fit(have, np.array([s0[f] for f in have], dtype=float), "kl") if have else np.full(4, 0.25)
    out = {"atoms": [float(x) for x in atoms], "s0": s0}
    for side in ("P", "Q"):
        samples = [s0[side]] + [ans.get((pr["pair_id"], side, k)) for k in range(1, EXTRA_SAMPLES + 1)]
        got = [clip(x) for x in samples if x is not None]
        b = float(np.median(got)) if got else 0.5
        x, nx = s0[side], s0["NOT " + side]
        if x is not None and nx is not None:
            cself = (clip(x) + 1 - clip(nx)) / 2
        elif x is not None:
            cself = clip(x)
        elif nx is not None:
            cself = 1 - clip(nx)
        else:
            cself = 0.5
        cpair = float(np.dot(FORM_ATOMS[side], atoms)) if have else 0.5
        out[side] = {"B": b, "C_self": cself, "C_pair": cpair, "raw": 0.5 if x is None else clip(x),
                     "samples": samples, "spread": (max(got) - min(got)) if len(got) >= 2 else None}
    return out


def cluster_boot(values, clusters, seed=BOOT_SEED, n=BOOT):
    """Mean of `values` (one per unit) and its 95% percentile CI, resampling clusters with replacement."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    keys = sorted(set(clusters))
    idx = {k: i for i, k in enumerate(keys)}
    cl = np.array([idx[c] for c in clusters])
    sums = np.bincount(cl, weights=values, minlength=len(keys))
    cnts = np.bincount(cl, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=n)
    est = (w @ sums) / (w @ cnts)
    return {"value": float(values.mean()), "ci95": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))],
            "n_units": int(len(values)), "n_clusters": len(keys)}


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def bar(ci, problems):
    if problems:
        return "Problem"
    if ci[0] >= NO_GAIN_BAR:
        return "No gain"
    if ci[1] < 0:
        return "Gain"
    return "Inconclusive"


def cmd_analyze(a):
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    drop = set()
    if a.drop_calls and Path(a.drop_calls).exists():
        drop = {l.strip() for l in open(a.drop_calls) if l.strip() and not l.startswith("#")}
    pairs = sorted(map(json.loads, open(OUT / "pairs.jsonl")), key=lambda p: p["run_order"])
    calls_path = Path(a.calls)
    ans = load_answers(calls_path, drop)
    truth = {r["qid"]: r["label"] for r in map(json.loads, open(OUT / "truth.jsonl"))}
    stops = [json.loads(l) for l in open(calls_path) if '"Stop: ' in l]
    complete = [p for p in pairs if all((p["pair_id"], f, s) in ans for f, s in calls_for(p))]
    scored = [p for p in complete if truth.get(p["p"]) is not None and truth.get(p["q"]) is not None]
    planned = len(complete) * len(calls_for(pairs[0]))
    missing = sum(1 for p in complete for f, s in calls_for(p) if ans[(p["pair_id"], f, s)] is None)
    units, joints = [], []
    for p in scored:
        r = arms_for_pair(p, ans)
        for side, qid in (("P", p["p"]), ("Q", p["q"])):
            y = truth[qid]
            u = {"pair_id": p["pair_id"], "side": side, "qid": qid, "source": p["source"], "y": y,
                 "spread": r[side]["spread"]}
            for arm in ("B", "C_self", "C_pair", "raw"):
                u["ll_" + arm], u["br_" + arm] = ll(r[side][arm], y), brier(r[side][arm], y)
                u["p_" + arm] = r[side][arm]
            units.append(u)
        yp, yq = truth[p["p"]], truth[p["q"]]
        bp, bq = r["P"]["B"], r["Q"]["B"]
        prod = {"(P) AND (Q)": bp * bq, "(P) AND (NOT Q)": bp * (1 - bq), "(NOT P) AND (Q)": (1 - bp) * bq,
                "(NOT P) AND (NOT Q)": (1 - bp) * (1 - bq)}
        lab = {"(P) AND (Q)": yp * yq, "(P) AND (NOT Q)": yp * (1 - yq), "(NOT P) AND (Q)": (1 - yp) * yq,
               "(NOT P) AND (NOT Q)": (1 - yp) * (1 - yq)}
        cp = dict(zip(("(P) AND (Q)", "(P) AND (NOT Q)", "(NOT P) AND (Q)", "(NOT P) AND (NOT Q)"), r["atoms"]))
        raw = {f: (0.5 if r["s0"][f] is None else r["s0"][f]) for f in prod}
        joints.append({"pair_id": p["pair_id"], "cluster": p["pair_id"],
                       "ll_C_pair": np.mean([ll(cp[f], lab[f]) for f in prod]),
                       "ll_B_product": np.mean([ll(prod[f], lab[f]) for f in prod]),
                       "ll_raw": np.mean([ll(raw[f], lab[f]) for f in prod])})
    cl = [u["qid"] for u in units]

    def diff(key_a, key_b, rows=units, clusters=cl):
        return cluster_boot([u[key_a] - u[key_b] for u in rows], clusters)

    primary = diff("ll_C_pair", "ll_C_self")
    problems = []
    if stops:
        problems.append(f"P1 model stop: {stops[0].get('error')}")
    if planned and missing / planned > 0.05:
        problems.append(f"P2 missing {missing}/{planned}")
    if len(scored) < a.min_pairs:
        problems.append(f"P3 only {len(scored)} pairs complete and scored")
    if a.leaks is not None and a.leaks >= 2:
        problems.append(f"P5 leak audit: {a.leaks} of 20 calls with post-t information")
    res = {
        "pairs": {"selected": len(pairs), "complete": len(complete), "scored": len(scored),
                  "by_source": dict(__import__("collections").Counter(p["source"] for p in scored))},
        "calls": {"planned_for_complete": planned, "missing_after_retry": missing, "dropped_by_leak_audit": len(drop)},
        "singles_log_loss": {arm: float(np.mean([u["ll_" + arm] for u in units])) for arm in ("raw", "B", "C_self", "C_pair")},
        "singles_brier": {arm: float(np.mean([u["br_" + arm] for u in units])) for arm in ("raw", "B", "C_self", "C_pair")},
        "primary_C_pair_minus_C_self": primary,
        "bar": bar(primary["ci95"], problems), "problems": problems,
        "secondary": {
            "C_pair_minus_B": diff("ll_C_pair", "ll_B"),
            "C_self_minus_B": diff("ll_C_self", "ll_B"),
            "B_minus_raw": diff("ll_B", "ll_raw"),
            "brier_C_pair_minus_C_self": diff("br_C_pair", "br_C_self"),
            "brier_C_pair_minus_B": diff("br_C_pair", "br_B"),
            "brier_C_self_minus_B": diff("br_C_self", "br_B"),
            "joints_C_pair_minus_B_product": cluster_boot([j["ll_C_pair"] - j["ll_B_product"] for j in joints],
                                                          [j["cluster"] for j in joints]),
            "joints_raw_minus_B_product": cluster_boot([j["ll_raw"] - j["ll_B_product"] for j in joints],
                                                       [j["cluster"] for j in joints]),
            "joints_log_loss": {k: float(np.mean([j[k] for j in joints])) for k in ("ll_C_pair", "ll_B_product", "ll_raw")},
        },
    }
    sp = [u for u in units if u["spread"] is not None]
    gains = [u["ll_C_pair"] - u["ll_C_self"] for u in sp]
    res["secondary"]["spread_vs_gain"] = {
        "spearman_spread_vs_C_pair_minus_C_self": spearman([u["spread"] for u in sp], gains),
        "spearman_spread_vs_C_pair_minus_B": spearman([u["spread"] for u in sp], [u["ll_C_pair"] - u["ll_B"] for u in sp]),
        "mean_spread": float(np.mean([u["spread"] for u in sp])),
        "share_spread_zero": float(np.mean([u["spread"] == 0 for u in sp])),
        "gain_when_spread_zero": cluster_boot([g for u, g in zip(sp, gains) if u["spread"] == 0],
                                              [u["qid"] for u in sp if u["spread"] == 0]) if any(u["spread"] == 0 for u in sp) else None,
        "gain_when_spread_positive": cluster_boot([g for u, g in zip(sp, gains) if u["spread"] > 0],
                                                  [u["qid"] for u in sp if u["spread"] > 0]) if any(u["spread"] > 0 for u in sp) else None,
    }
    res["secondary"]["primary_by_source"] = {
        src: diff("ll_C_pair", "ll_C_self", [u for u in units if u["source"] == src],
                  [u["qid"] for u in units if u["source"] == src])
        for src in sorted({u["source"] for u in units})}
    # search behaviour (from the MCP server's log)
    qrows = [json.loads(l) for l in open(OUT / "queries.jsonl")] if (OUT / "queries.jsonl").exists() else []
    ids = {call_id(p, f, s) for p in complete for f, s in calls_for(p)}
    qrows = [q for q in qrows if q["call_id"].removesuffix("-retry") in ids]
    per_call = __import__("collections").Counter(q["call_id"] for q in qrows if not q.get("refused") and not q.get("error"))
    res["search"] = {"queries": sum(per_call.values()), "calls_with_search": len(per_call),
                     "mean_queries_per_call": sum(per_call.values()) / max(1, len(ids)),
                     "refused_over_limit": sum(1 for q in qrows if q.get("refused")),
                     "errors": sum(1 for q in qrows if q.get("error")),
                     "mean_kept_per_query": float(np.mean([q.get("n_kept", 0) for q in qrows if "n_kept" in q] or [0]))}
    llm, exa = spend()
    res["spend"] = {"llm_list_usd": round(llm, 2), "exa_cash_usd": round(exa, 2), "total_usd": round(llm + exa, 2)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.units:
        with open(a.units, "w") as f:
            for u in units:
                f.write(json.dumps(u) + "\n")
    print(json.dumps({k: res[k] for k in ("pairs", "calls", "singles_log_loss", "primary_C_pair_minus_C_self", "bar",
                                          "problems")}, indent=1))


def cmd_spend(_a):
    llm, exa = spend()
    print(json.dumps({"llm_list_usd": round(llm, 4), "exa_cash_usd": round(exa, 4), "total": round(llm + exa, 4)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("spend")
    sub.add_parser("truth")
    z = sub.add_parser("analyze")
    z.add_argument("--calls", default=str(OUT / "calls.jsonl"))
    z.add_argument("--out", default=str(OUT / "analysis.json"))
    z.add_argument("--units", default=str(OUT / "units.jsonl"))
    z.add_argument("--drop-calls", default=str(OUT / "leak-drop.txt"))
    z.add_argument("--leaks", type=int, default=None, help="calls with confirmed post-t information among the 20 audited")
    z.add_argument("--min-pairs", type=int, default=90)
    a = ap.parse_args()
    {"smoke": cmd_smoke, "run": cmd_run, "spend": cmd_spend, "truth": cmd_truth, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
