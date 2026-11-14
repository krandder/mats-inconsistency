#!/usr/bin/env python3
"""Rebuild site/market-game.html from market-game run directories.

Usage: build_market_game_page.py RUN_DIR [RUN_DIR ...]
Runs the scorer for each run (offline, fast), reads sessions.jsonl and the
pass snapshots, and injects a JSON snapshot into the page template.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site" / "market-game.template.html"
OUT = ROOT / "site" / "market-game.html"
ENGINE = "/home/agent/workspaces/bayes-engine"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def resolution_map(run: Path) -> dict[str, int]:
    return {r["question_id"]: int(r["resolution"]) for r in load_jsonl(run / "questions.jsonl")}


def prices_from_snapshot(path: Path) -> dict[str, float] | None:
    """Marginal P(yes) per question from a snapshot, via the engine."""
    if not path.exists():
        return None
    sys.path.insert(0, ENGINE)
    from inference.factored_market import FactoredMarket  # type: ignore
    data = json.loads(path.read_text())
    market = FactoredMarket.from_snapshot(data["engine"], max_width=data.get("max_width"))
    return {q: market.marginal(q)["yes"] for q in data["question_ids"]}


def summarize_run(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    cfg = cfg.get("config", cfg)
    sessions = load_jsonl(run / "sessions.jsonl")
    final = [s for s in sessions if s.get("type", "session") == "session" and s.get("status") in ("ok", "failed", "done", "complete") or s.get("status") is None]
    # keep the last record per session id
    last: dict[str, dict] = {}
    for s in sessions:
        if s.get("session_id"):
            last[s["session_id"]] = s
    sessions = list(last.values())
    groups = json.loads((run / "groups.json").read_text()) if (run / "groups.json").exists() else {}
    passes_cfg = int(cfg.get("passes", 3))
    per_pass_total = {p: len(groups.get(str(p), groups.get(p, []))) for p in range(1, passes_cfg + 1)} if groups else {}
    scores = None
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "score_market_game.py"), str(run)], capture_output=True, timeout=600,
                       env={"PYTHONPATH": f"{ROOT}:{ENGINE}", "PATH": "/usr/bin:/bin"})
        if (run / "scores.json").exists():
            scores = json.loads((run / "scores.json").read_text())
    except Exception:
        scores = None
    per_pass_scored = {int(p["pass"]): p for p in (scores or {}).get("per_pass", [])}
    passes = []
    tokens = 0
    for p in range(1, passes_cfg + 1):
        ss = [s for s in sessions if int(s.get("pass", 0)) == p]
        done = len([s for s in ss if s.get("status") in ("completed", "failed")])
        proposed = sum(int(s.get("proposed") or 0) if isinstance(s.get("proposed"), int) else len(s.get("proposed") or []) for s in ss)
        executed = sum(len(s.get("executed") or []) for s in ss)
        rej: dict[str, int] = {}
        for s in ss:
            for r in s.get("rejected") or []:
                reason = r.get("reason") if isinstance(r, dict) else str(r)
                rej[reason] = rej.get(reason, 0) + 1
        kls = [float(s.get("kl_used") or 0) for s in ss if s.get("kl_used") is not None]
        sc = per_pass_scored.get(p)
        passes.append({"pass": p, "sessions_done": done, "sessions_total": per_pass_total.get(p, len(ss) or 0),
                       "brier": sc["brier"] if sc else None, "logloss": sc["log_loss"] if sc else None,
                       "joint": sc.get("joint_log_loss") if sc else None,
                       "extreme_rate": sc["extremization_rate"] if sc else None,
                       "valid_rate": (executed / proposed) if proposed else None, "executed": executed, "proposed": proposed,
                       "rejections": rej, "kl_mean": (sum(kls) / len(kls)) if kls else None,
                       "kl_p90": sorted(kls)[int(0.9 * (len(kls) - 1))] if kls else None})
        for s in ss:
            u = s.get("usage") or {}
            if isinstance(u, dict):
                tokens += int(u.get("input_tokens") or 0) + int(u.get("output_tokens") or 0)
    # KL histogram
    bins = [0, 0.5, 1, 2, 3, 5, 8, 12, 20]
    counts = {}
    for p in range(1, passes_cfg + 1):
        ks = [float(s.get("kl_used") or 0) for s in sessions if int(s.get("pass", 0)) == p]
        if not ks:
            continue
        c = [0] * len(bins)
        for k in ks:
            i = max(j for j, b in enumerate(bins) if k >= b)
            c[i] += 1
        counts[str(p)] = c
    # price paths for 20 random questions
    res = resolution_map(run)
    snaps = [prices_from_snapshot(run / "snapshots" / f"pass-{p}.json") for p in range(1, passes_cfg + 1)]
    snaps = [s for s in snaps if s]
    paths = []
    if snaps:
        rng = random.Random(7)
        qs = rng.sample(sorted(snaps[0].keys()), min(20, len(snaps[0])))
        paths = [{"q": q[:12], "res": res.get(q), "p": [0.5] + [s[q] for s in snaps]} for q in qs]
    herd = None
    if scores and scores.get("herding", {}).get("consecutive_pairs"):
        pairs = scores["herding"]["consecutive_pairs"]
        herd = pairs[-1]["autocorrelation"]
    # sample trace from the latest pass with a response
    sample = None
    latest = [s for s in sessions if s.get("raw_response")]
    if latest:
        s = max(latest, key=lambda x: (int(x.get("pass", 0)), x.get("finished_at") or ""))
        sample = {"pass": s.get("pass"), "session_id": s.get("session_id"), "thread_id": s.get("thread_id"),
                  "n_questions": s.get("n") or len(s.get("question_ids") or []), "executed": len(s.get("executed") or []),
                  "rejected": len(s.get("rejected") or []), "text": (s.get("raw_response") or "")[:6000]}
    started = min((s.get("started_at") or "" for s in sessions), default="") or None
    elapsed_h = None
    if sessions:
        ts = [s.get("finished_at") for s in sessions if s.get("finished_at")]
        try:
            t0 = time.mktime(time.strptime(started[:19], "%Y-%m-%dT%H:%M:%S")); t1 = time.mktime(time.strptime(max(ts)[:19], "%Y-%m-%dT%H:%M:%S"))
            elapsed_h = (t1 - t0) / 3600
        except Exception:
            elapsed_h = None
    all_done = all(p["sessions_total"] and p["sessions_done"] >= p["sessions_total"] for p in passes) or (run.parent / "CAMPAIGN_DONE").exists()
    status = "done" if all_done and passes else ("running" if sessions else "queued")
    model = {"astra": "GPT-6 Astra", "luna": "GPT-5.6 Luna", "mock": "mock"}.get(cfg.get("provider", "?"), cfg.get("provider", "?"))
    return {"id": run.name, "model": model, "effort": cfg.get("effort"), "n": cfg.get("n_per_session") or cfg.get("n"),
            "passes": passes, "status": status, "started": started, "elapsed_h": elapsed_h, "tokens": tokens,
            "herding": herd, "kl_hist": {"bins": bins, "counts": counts} if counts else None, "paths": paths, "sample": sample}


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]]
    data = {"runs": [summarize_run(r) for r in runs if r.exists()],
            "baselines": {"constant": 0.25, "base_model": 0.215, "o1": 0.2017, "remax": 0.1907, "polymarket": 0.1509},
            "synced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    html = TEMPLATE.read_text()
    a = html.index('<script id="data" type="application/json">') + len('<script id="data" type="application/json">')
    b = html.index("</script>", a)
    OUT.write_text(html[:a] + json.dumps(data, separators=(",", ":")) + html[b:])
    print("built", OUT, [(r["id"], r["status"], [p["brier"] for p in r["passes"]]) for r in data["runs"]])


if __name__ == "__main__":
    main()
