"""Does the projection gain shrink as the model gets stronger? (2026-09-30)

Pre-registration: docs/strength-scaling-20260930.md.  Same 300 test pairs for four models:
Qwen3-32B and Qwen3-235B-A22B (existing v7b_explicit answers, DashScope, thinking off, t0) and two
new arms, gpt-6-luna (fleet Codex route: the codex-any proxy, Responses API) and claude-opus-5-5
(Prime, headless `claude -p`).  One Boolean form per call, no tools, the v7b_explicit prompt of
scripts/prompt_negation_fix.py with the Qwen-only " /no_think" suffix removed.

  smoke    --model M               one validation pair (12 calls): route, served model, parse, cost
  run      --model M [--workers N] the 300 subset pairs in pre-registered order (resumable)
  analyze  --out J --chart C       raw vs kl-projected log loss, paired bootstrap, bars

No silent model fallback: a served model other than the requested one, or a rejected model, stops
the arm (STOP-<model> file) and is reported.  Spend is costed at list price and capped per arm and
over all logs in the output directory.
"""

from __future__ import annotations

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
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prompt_negation_fix as pn  # noqa: E402
from coherence_projection import project  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

VARIANT = "v7b_explicit"
OUT = Path("/home/agent/data/strength-scaling-20260930")
SUBSET = "/home/agent/data/decomposition-32b-20260924/subset.json"
SUBSET_SEED, SUBSET_N = 20260924, 300
EXISTING = {
    "qwen3-32b": "/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl",
    "qwen3-235b-a22b": "/home/agent/data/consistency-distill-v2-20260929/teacher235/log-test-v7b_explicit.jsonl",
}
V6_REFERENCE = {  # Kelvin's quoted numbers (v6_direct wording), for continuity only
    "qwen3-32b v6_direct (vLLM)": "/home/agent/data/qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl",
    "qwen3-235b v6_direct (per-form)": "/home/agent/data/precutoff-baseline-20260923/per-form/predictions-test.jsonl",
}
NEW = ("gpt-6-luna", "claude-opus-5-5")
ORDER = ["qwen3-32b", "qwen3-235b-a22b", "gpt-6-luna", "claude-opus-5-5"]
CAP_ARM = {"gpt-6-luna": 8.0, "claude-opus-5-5": 48.0}
CAP_TOTAL = 58.0  # all logs in OUT, smoke tests included; Kelvin's cap is $60
LUNA_URL = "http://127.0.0.1:7810/responses"
LUNA_PRICE = (0.10, 0.01, 0.50)  # $/M input, cached input, output (standard tier, short context)
LUNA_EFFORTS = ("none", "minimal", "low")  # pre-registered order: the first the route accepts
OPUS_EFFORT = "low"
PRIME = "http://127.0.0.1:7800"
CLAUDE_BIN = "/home/agent/.local/bin/claude"
AGENT = "calibrated-ai"
TRANSIENT = ("429", "500", "502", "503", "504", "529", "overloaded", "timeout", "timed out", "temporarily",
             "ECONNRESET", "Connection reset", "proxy_retryable", "rate", "Remote end closed", "stream")
SEED = 20260930
N_BOOT = 2000
LITERALS = ["P", "NOT P", "Q", "NOT Q"]


class Stop(RuntimeError):
    """The arm must stop (model rejected or substituted). Never retried, never replaced."""


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]", "-", model.lower())


# --- subset and prompts ----------------------------------------------------------------------

def subset_order(all_ids: list[str]) -> list[str]:
    stored = json.loads(Path(SUBSET).read_text())
    order = random.Random(SUBSET_SEED).sample(sorted(all_ids), SUBSET_N)
    assert set(order) == set(stored["pair_ids"]), "subset does not recompute"
    return order


def strip_qwen_switch(msgs: list[dict]) -> tuple[str, str]:
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and user.endswith(" /no_think")
    return system, user[: -len(" /no_think")]


def rows(split: str, pair_order: list[str] | None) -> list[dict]:
    rs = pn.rows_for(split, VARIANT)
    if pair_order is None:
        return rs
    rank = {p: i for i, p in enumerate(pair_order)}
    rs = [r for r in rs if r["pair_id"] in rank]
    rs.sort(key=lambda r: (rank[r["pair_id"]], FORM_ORDER.index(r["form"])))
    return rs


# --- transports --------------------------------------------------------------------------------

def luna_call(system: str, turns: list[tuple[str, str]], effort: str) -> dict:
    body = {"model": "gpt-6-luna", "instructions": system, "store": False, "stream": True,
            "reasoning": {"effort": effort},
            "input": [{"type": "message", "role": role,
                       "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}]}
                      for role, text in turns]}
    req = urllib.request.Request(LUNA_URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Bearer proxy-managed-client", "Content-Type": "application/json",
        "Accept": "text/event-stream", "X-Futarchy-Agent": AGENT})
    text, done, err = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                t = ev.get("type", "")
                if t == "response.output_text.delta":
                    text.append(ev.get("delta", ""))
                elif t == "response.completed":
                    done = ev["response"]
                elif t in ("response.failed", "error", "response.incomplete"):
                    err = json.dumps(ev)[:600]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}"
    except Exception as e:  # noqa: BLE001 - network errors are classified below
        err = f"{type(e).__name__}: {e}"[:600]
    if done is None:
        low = (err or "no response.completed").lower()
        if any(k in low for k in ("model_not_found", "not supported", "unsupported model", "does not exist",
                                   "invalid model")):
            raise Stop(f"gpt-6-luna rejected: {err}")
        raise RuntimeError(err or "no response.completed")
    served = done.get("model") or ""
    if not re.fullmatch(r"gpt-6-luna(-\d{4}-\d{2}-\d{2})?", served):
        raise Stop(f"served model {served!r} != gpt-6-luna")
    if not text:
        for item in done.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    text.append(c.get("text", ""))
    u = done.get("usage") or {}
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    usage = {"input_tokens": u.get("input_tokens", 0), "cached_input_tokens": cached,
             "output_tokens": u.get("output_tokens", 0),
             "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
    pin, pcache, pout = LUNA_PRICE
    cost = ((usage["input_tokens"] - cached) * pin + cached * pcache + usage["output_tokens"] * pout) / 1e6
    return {"text": "".join(text), "served_model": served, "usage": usage, "cost": cost}


def opus_call(system: str, user: str, effort: str = OPUS_EFFORT) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "CLAUDE", "FLEET_", "AI_AGENT", "FUTARCHY_", "TMUX"))}
    env.update(HOME="/home/agent", PATH="/home/agent/.local/bin:/home/agent/bin:/usr/local/bin:/usr/bin:/bin",
               ANTHROPIC_BASE_URL=PRIME, ANTHROPIC_AUTH_TOKEN=f"token-proxy-managed:{AGENT}",
               CLAUDE_CODE_NO_MODEL_FALLBACK="1")
    cmd = [CLAUDE_BIN, "-p", "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--settings", json.dumps({"disableAllHooks": True, "alwaysThinkingEnabled": False}),
           "--setting-sources", "", "--exclude-dynamic-system-prompt-sections", "--disable-slash-commands",
           "--model", "claude-opus-5-5", "--output-format", "json", "--no-session-persistence",
           "--effort", effort, "--system-prompt", system]
    cwd = tempfile.mkdtemp(prefix="ss-claude-", dir=str(OUT / "tmp"))
    try:
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=420, env=env, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    try:
        d = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError(f"unparseable CLI output rc={p.returncode}: {p.stdout[:200]!r} {p.stderr[-300:]!r}")
    if isinstance(d, list):
        d = next((x for x in d if isinstance(x, dict) and x.get("type") == "result"), {})
    used = sorted((d.get("modelUsage") or {}).keys())
    if d.get("is_error"):
        msg = f"{str(d.get('result'))[:300]} status={d.get('api_error_status')}"
        if any(k in msg.lower() for k in ("model_not_found", "not_found_error", "invalid model", "does not exist")):
            raise Stop(f"claude-opus-5-5 rejected: {msg}")
        raise RuntimeError(msg)
    if used != ["claude-opus-5-5"]:
        raise Stop(f"asked for claude-opus-5-5, answer came from {used}")
    u = d.get("usage") or {}
    usage = {k: u.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                   "output_tokens")}
    return {"text": d.get("result") or "", "served_model": used[0], "usage": usage,
            "cost": float(d.get("total_cost_usd") or 0.0)}


def with_backoff(fn, *a, tries: int = 6, **k):
    delay, errors = 10.0, []
    for attempt in range(tries):
        try:
            r = fn(*a, **k)
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


def ask(model: str, system: str, user: str, effort: str, prev: str | None = None) -> dict:
    """First call (prev None) or the protocol's one retry after an unparseable reply."""
    if model == "gpt-6-luna":
        turns = [("user", user)] if prev is None else [("user", user), ("assistant", prev or ""),
                                                        ("user", vb.RETRY_REMINDER)]
        return with_backoff(luna_call, system, turns, effort)
    text = user if prev is None else user + "\n\n" + vb.RETRY_REMINDER
    return with_backoff(opus_call, system, text, effort)


# --- runner ------------------------------------------------------------------------------------

def spent(path_glob: str = "log-*.jsonl") -> float:
    tot = 0.0
    for p in OUT.glob(path_glob):
        for line in p.read_text().splitlines():
            tot += json.loads(line).get("cost", 0.0)
    return tot


def run_rows(model: str, todo: list[dict], log: Path, effort: str, workers: int, cap_arm: float) -> dict:
    stop_file = OUT / f"STOP-{slug(model)}"
    if stop_file.exists():
        raise SystemExit(f"{stop_file} exists: {stop_file.read_text()}")
    arm_glob = f"log*-{slug(model)}.jsonl"
    state = {"arm": spent(arm_glob), "all": spent(), "stopped": None}
    lock = threading.Lock()
    per_call = 0.02 if model == "claude-opus-5-5" else 0.002

    def one(row):
        with lock:
            if state["stopped"]:
                return None
            if state["arm"] + per_call > cap_arm or state["all"] + per_call > CAP_TOTAL:
                state["stopped"] = state["stopped"] or f"cap reached (arm ${state['arm']:.2f}, all ${state['all']:.2f})"
                return None
        system, user = strip_qwen_switch(row["_msgs"])
        try:
            first = ask(model, system, user, effort)
        except Stop as e:
            with lock:
                state["stopped"] = str(e)
            stop_file.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {e}\n")
            return None
        except Exception as e:  # noqa: BLE001 - persistent transport failure: record as missing
            first = {"text": None, "served_model": None, "usage": None, "cost": 0.0, "transient_errors": [str(e)[:300]]}
        p, why = vb.parse_probability(first["text"]) if first["text"] is not None else (None, "call_failed")
        rec = {k: v for k, v in row.items() if k != "_msgs"}
        rec.update(model=model, variant=VARIANT, effort=effort,
                   prompt_sha256=hashlib.sha256(json.dumps([system, user]).encode()).hexdigest(),
                   raw=first["text"], served_model=first["served_model"], usage=first["usage"],
                   transient_errors=first["transient_errors"], retried=False, raw_retry=None, usage_retry=None,
                   p=p, reason=why, cost=first["cost"], ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if p is None and first["text"] is not None:
            try:
                second = ask(model, system, user, effort, prev=first["text"])
                p2, why2 = vb.parse_probability(second["text"])
                rec.update(retried=True, raw_retry=second["text"], usage_retry=second["usage"], p=p2, reason=why2,
                           cost=rec["cost"] + second["cost"])
            except Stop as e:
                with lock:
                    state["stopped"] = str(e)
                stop_file.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {e}\n")
            except Exception as e:  # noqa: BLE001
                rec.update(retried=True, raw_retry=None, reason=f"retry_failed: {str(e)[:200]}")
        return rec

    n = 0
    with ThreadPoolExecutor(workers) as pool, log.open("a") as f:
        futs = [pool.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is None:
                continue
            with lock:
                f.write(json.dumps(rec) + "\n"); f.flush()
                state["arm"] += rec["cost"]; state["all"] += rec["cost"]
                n += 1
                if n % 300 == 0:
                    print(f"{model}: {n}/{len(todo)} calls, arm ${state['arm']:.3f}, all ${state['all']:.3f}", flush=True)
    return state


def cmd_smoke(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tmp").mkdir(exist_ok=True)
    val = rows("validation", None)
    pid = sorted({r["pair_id"] for r in val})[0]
    todo = [r for r in val if r["pair_id"] == pid]
    log = OUT / f"log-smoke-{slug(args.model)}.jsonl"
    efforts = LUNA_EFFORTS if args.model == "gpt-6-luna" else (OPUS_EFFORT,)
    chosen = None
    for eff in efforts:  # pre-registered order; the first effort the route accepts is kept
        system, user = strip_qwen_switch(todo[0]["_msgs"])
        try:
            r = ask(args.model, system, user, eff)
        except Stop as e:
            (OUT / f"STOP-{slug(args.model)}").write_text(f"{e}\n")
            raise SystemExit(f"STOP: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"effort {eff}: not accepted ({str(e)[:200]})", flush=True)
            continue
        print(f"effort {eff}: served {r['served_model']}, text {r['text']!r}, usage {r['usage']}, ${r['cost']:.5f}")
        chosen = eff
        break
    if chosen is None:
        raise SystemExit("no effort accepted")
    st = run_rows(args.model, todo, log, chosen, 6, CAP_ARM[args.model])
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    cost = sum(r["cost"] for r in recs)
    print(json.dumps({"model": args.model, "effort": chosen, "pair": pid, "calls": len(recs),
                      "missing": sum(r["p"] is None for r in recs), "retried": sum(r["retried"] for r in recs),
                      "served": sorted({r["served_model"] or "none" for r in recs}), "cost": round(cost, 5),
                      "cost_per_pair": round(cost / max(1, len({r['pair_id'] for r in recs})), 5),
                      "projected_300": round(cost * 300, 2), "stopped": st["stopped"],
                      "answers": {r["form"]: r["p"] for r in recs}}, indent=1))
    (OUT / f"effort-{slug(args.model)}.txt").write_text(chosen + "\n")


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "tmp").mkdir(exist_ok=True)
    effort = (OUT / f"effort-{slug(args.model)}.txt").read_text().strip()
    test = rows("test", None)
    order = subset_order(sorted({r["pair_id"] for r in test}))
    if args.pairs:
        order = order[: args.pairs]
    todo_all = rows("test", order)
    log = OUT / f"log-test-{slug(args.model)}.jsonl"
    done = {json.loads(l)["query_id"] for l in log.read_text().splitlines()} if log.exists() else set()
    todo = [r for r in todo_all if r["query_id"] not in done]
    print(f"{args.model} effort {effort}: {len(order)} pairs, {len(done)} done, {len(todo)} calls to run; "
          f"all logs ${spent():.3f}", flush=True)
    st = run_rows(args.model, todo, log, effort, args.workers, CAP_ARM[args.model])
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    print(f"{args.model}: log {len(recs)} calls, pairs complete "
          f"{sum(1 for c in _count_by_pair(recs).values() if c == 12)}, retried {sum(r['retried'] for r in recs)}, "
          f"missing {sum(r['p'] is None for r in recs)}, served {sorted({r['served_model'] or 'none' for r in recs})}, "
          f"arm ${sum(r['cost'] for r in recs):.3f}, all logs ${spent():.3f}"
          + (f" STOPPED: {st['stopped']}" if st["stopped"] else ""), flush=True)


def _count_by_pair(recs):
    c = defaultdict(int)
    for r in recs:
        c[r["pair_id"]] += 1
    return c


# --- analysis ----------------------------------------------------------------------------------

def load_rows(path: str, pairs: set[str]) -> list[dict]:
    return [r for r in pn.load(path) if r["pair_id"] in pairs]


def project_safe(rs: list[dict]) -> list[dict]:
    """kl projection per pair; a pair with no stated form at all is left as is (scored at 0.5)."""
    have = {r["pair_id"] for r in rs if r["p"] is not None}
    empty = [r for r in rs if r["pair_id"] not in have]
    return project([dict(r, query_id=None) for r in rs if r["pair_id"] in have], "kl") + empty


def per_pair(rs: list[dict], pairs: list[str]) -> dict:
    """pair -> {form: (p used for scoring, label)}; missing -> 0.5, clipped to [0.01, 0.99]."""
    d = defaultdict(dict)
    for r in rs:
        p = 0.5 if r["p"] is None else min(1 - pn.LO, max(pn.LO, r["p"]))
        d[r["pair_id"]][r["form"]] = (p, r["label"], r["p"])
    assert all(len(d[p]) == 12 for p in pairs), "a pair lacks forms"
    return d


def ll_matrix(d: dict, pairs: list[str]) -> np.ndarray:
    """(pairs, 12) log loss in FORM_ORDER."""
    m = np.zeros((len(pairs), 12))
    for i, pid in enumerate(pairs):
        for j, f in enumerate(FORM_ORDER):
            p, y, _ = d[pid][f]
            m[i, j] = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return m


def brier_matrix(d: dict, pairs: list[str]) -> np.ndarray:
    return np.array([[(d[pid][f][0] - d[pid][f][1]) ** 2 for f in FORM_ORDER] for pid in pairs])


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def boot_mean(x: np.ndarray, idx: np.ndarray) -> dict:
    return {"value": float(x.mean()), "ci95": ci(x[idx].mean(1))}


def coherence_stats(d: dict, pairs: list[str], idx: np.ndarray) -> dict:
    """corr(p(A), p(not A)) over the 2 events per pair (A = P, Q); missing -> 0.5 as in pn.events."""
    x = np.array([[d[p]["P"][2] if d[p]["P"][2] is not None else 0.5, d[p]["Q"][2] if d[p]["Q"][2] is not None else 0.5]
                  for p in pairs])
    y = np.array([[d[p]["NOT P"][2] if d[p]["NOT P"][2] is not None else 0.5,
                   d[p]["NOT Q"][2] if d[p]["NOT Q"][2] is not None else 0.5] for p in pairs])

    def stats(xx, yy):
        xx, yy = xx.ravel(), yy.ravel()
        return (np.corrcoef(xx, yy)[0, 1], np.mean(np.isclose(xx, yy)), np.mean(np.abs(xx + yy - 1) > 0.2),
                np.mean(xx + yy))
    point = stats(x, y)
    bs = np.array([stats(x[i], y[i]) for i in idx])
    names = ("corr", "identical", "sum_off_0.2", "mean_sum")
    return {n: {"value": float(point[k]), "ci95": ci(bs[:, k])} for k, n in enumerate(names)}


def recall_stats(rs: list[dict]) -> dict:
    lit = [r for r in rs if r["form"] in LITERALS and r["p"] is not None]
    conf = [r for r in lit if r["p"] <= 0.05 or r["p"] >= 0.95]
    acc = float(np.mean([(r["p"] >= 0.5) == (r["label"] == 1) for r in conf])) if conf else None
    share = len(conf) / len(lit) if lit else 0.0
    return {"confident_literal_share": share, "confident_literal_accuracy": acc,
            "recall_flag": bool(share >= 0.25 and acc is not None and acc >= 0.95)}


def analyze_model(rs: list[dict], pairs: list[str], idx: np.ndarray) -> dict:
    raw = per_pair(rs, pairs)
    proj = per_pair(project_safe(rs), pairs)
    L_raw, L_proj = ll_matrix(raw, pairs), ll_matrix(proj, pairs)
    lit = [FORM_ORDER.index(f) for f in LITERALS]
    cmp_ = [j for j in range(12) if j not in lit]
    out = {"n_pairs": len(pairs), "missing_forms": sum(r["p"] is None for r in rs),
           "missing_share": sum(r["p"] is None for r in rs) / (12 * len(pairs))}
    for name, cols in (("all12", list(range(12))), ("literals", lit), ("compounds", cmp_)):
        r_ = L_raw[:, cols].mean(1); p_ = L_proj[:, cols].mean(1)
        out[f"ll_raw_{name}"] = boot_mean(r_, idx)
        out[f"ll_proj_{name}"] = boot_mean(p_, idx)
        out[f"gain_{name}"] = boot_mean(p_ - r_, idx)
    out["brier_raw"] = boot_mean(brier_matrix(raw, pairs).mean(1), idx)
    out["brier_proj"] = boot_mean(brier_matrix(proj, pairs).mean(1), idx)
    out["coherence"] = coherence_stats(raw, pairs, idx)
    out["recall"] = recall_stats(rs)
    out["_pp"] = {"raw": L_raw.mean(1), "proj": L_proj.mean(1)}
    return out


def bar(m: dict) -> str:
    g = m["gain_all12"]
    lo, hi = g["ci95"]
    if lo >= -0.01:
        return "No gain" + (" (small but significant)" if hi < 0 else "")
    if hi < 0:
        return "Gain"
    return "Inconclusive"


def cmd_analyze(args) -> None:
    test = rows("test", None)
    order = subset_order(sorted({r["pair_id"] for r in test}))
    sources = dict(EXISTING)
    for m in NEW:
        sources[m] = str(OUT / f"log-test-{slug(m)}.jsonl")
    loaded = {m: load_rows(p, set(order)) for m, p in sources.items() if Path(p).exists()}
    complete = {m: {p for p, c in _count_by_pair(rs).items() if c == 12} for m, rs in loaded.items()}
    pairs = sorted(set(order).intersection(*complete.values()))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(pairs), size=(N_BOOT, len(pairs)))
    problems = []
    res = {"question": "Does the projection gain shrink as the model gets stronger?",
           "prereg": "docs/strength-scaling-20260930.md", "subset": SUBSET, "n_pairs": len(pairs),
           "bootstrap": {"resamples": N_BOOT, "seed": SEED, "unit": "pair"}, "models": {}, "contrasts": {},
           "v6_reference": {}, "secondary_uncertain_pairs": {}}
    if len(pairs) < 250:
        problems.append(f"P3: only {len(pairs)} pairs complete for every model")
    for m in NEW:
        stop = OUT / f"STOP-{slug(m)}"
        if stop.exists():
            problems.append(f"P1: {m} arm stopped: {stop.read_text().strip()}")
        if m not in loaded:
            problems.append(f"P1: no answers for {m}")
    for m, rs in loaded.items():
        rs = [r for r in rs if r["pair_id"] in set(pairs)]
        res["models"][m] = analyze_model(rs, pairs, idx)
        served = sorted({json.loads(l).get("served_model") or "none" for l in Path(sources[m]).read_text().splitlines()
                         if json.loads(l)["pair_id"] in set(pairs)})
        res["models"][m]["served_models"] = served
        if m in NEW and res["models"][m]["missing_share"] > 0.05:
            problems.append(f"P2: {m} missing {res['models'][m]['missing_share']:.1%} of forms")
        if m in NEW:
            res["models"][m]["bar"] = bar(res["models"][m])
    pp = {m: res["models"][m].pop("_pp") for m in res["models"]}
    for a in res["models"]:
        for b in res["models"]:
            if a == b or ORDER.index(a) < ORDER.index(b):
                continue
            ga = pp[a]["proj"] - pp[a]["raw"]; gb = pp[b]["proj"] - pp[b]["raw"]
            res["contrasts"][f"{a} - {b}"] = {"raw_ll": boot_mean(pp[a]["raw"] - pp[b]["raw"], idx),
                                             "proj_ll": boot_mean(pp[a]["proj"] - pp[b]["proj"], idx),
                                             "gain": boot_mean(ga - gb, idx)}
    # scaling reading per frontier model, against 235B (pre-registered)
    scaling = {}
    for m in NEW:
        if m not in res["models"]:
            continue
        c = res["contrasts"].get(f"{m} - qwen3-235b-a22b")
        stronger = c["raw_ll"]["ci95"][1] < 0
        smaller = c["gain"]["ci95"][0] > 0
        larger = c["gain"]["ci95"][1] < 0
        scaling[m] = {"stronger_than_235b_raw": stronger, "gain_smaller_than_235b": smaller,
                      "gain_larger_than_235b": larger,
                      "reading": ("shrinks" if stronger and smaller else
                                  "does not shrink" if stronger and not smaller else
                                  "not stronger than 235B raw: no scaling reading")}
    res["scaling"] = scaling
    # continuity with the v6 numbers Kelvin quoted (no new calls)
    for name, path in V6_REFERENCE.items():
        rs = load_rows(path, set(pairs))
        mm = analyze_model(rs, pairs, idx)
        mm.pop("_pp")
        res["v6_reference"][name] = {k: mm[k] for k in ("ll_raw_all12", "ll_proj_all12", "gain_all12", "coherence")}
    # secondary: pairs on which a frontier model was uncertain on both literals (label-free selection)
    for m in NEW:
        if m not in loaded:
            continue
        d = per_pair([r for r in loaded[m] if r["pair_id"] in set(pairs)], pairs)
        U = [p for p in pairs if all(d[p][f][2] is not None and 0.10 <= d[p][f][2] <= 0.90 for f in ("P", "Q"))]
        sub = {"n_pairs": len(U), "gains": {}}
        if len(U) >= 30:
            ui = np.array([pairs.index(p) for p in U])
            uidx = np.random.default_rng(SEED).integers(0, len(U), size=(N_BOOT, len(U)))
            for mm in res["models"]:
                r_, p_ = pp[mm]["raw"][ui], pp[mm]["proj"][ui]
                sub["gains"][mm] = {"ll_raw": boot_mean(r_, uidx), "gain": boot_mean(p_ - r_, uidx)}
        res["secondary_uncertain_pairs"][m] = sub
    res["problems"] = problems
    res["verdict"] = {m: ("Problem" if any(m in p for p in problems) or any(p.startswith("P3") for p in problems)
                          else res["models"][m]["bar"]) for m in NEW if m in res["models"]}
    Path(args.out).write_text(json.dumps(res, indent=1))
    chart = {"title": "Projection gain vs model strength (same 300 test pairs, v7b_explicit, one form per call)",
             "x": "raw all-12 log loss (lower = stronger)", "y": "kl projection gain (projected - raw; negative helps)",
             "points": [{"model": m, "raw_ll": v["ll_raw_all12"]["value"], "raw_ci95": v["ll_raw_all12"]["ci95"],
                         "proj_ll": v["ll_proj_all12"]["value"], "gain": v["gain_all12"]["value"],
                         "gain_ci95": v["gain_all12"]["ci95"], "negation_corr": v["coherence"]["corr"]["value"],
                         "negation_corr_ci95": v["coherence"]["corr"]["ci95"], "bar": v.get("bar", "existing")}
                        for m, v in res["models"].items()],
             "n_pairs": len(pairs)}
    Path(args.chart).write_text(json.dumps(chart, indent=1))
    for m, v in res["models"].items():
        print(f"{m:18s} raw {v['ll_raw_all12']['value']:.4f} proj {v['ll_proj_all12']['value']:.4f} "
              f"gain {v['gain_all12']['value']:+.4f} {[round(x, 4) for x in v['gain_all12']['ci95']]} "
              f"corr {v['coherence']['corr']['value']:+.3f} missing {v['missing_forms']} "
              f"recall {v['recall']} bar {v.get('bar', '-')}")
    print("scaling", json.dumps(scaling))
    print("problems", problems)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke"); r = sub.add_parser("run"); a = sub.add_parser("analyze")
    for p in (s, r):
        p.add_argument("--model", required=True, choices=NEW)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--pairs", type=int, help="first N pairs of the pre-registered order (budget rule)")
    a.add_argument("--out", required=True)
    a.add_argument("--chart", required=True)
    args = ap.parse_args()
    {"smoke": cmd_smoke, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
