#!/usr/bin/env python3
"""Rebuild per-pass market snapshots of a market-game run by replaying logged fills in version order.

Needed when a resume-and-retry cycle re-ran sessions of an earlier pass after later passes
and overwrote pass-N.json. Fills carry a global execution version, so the market at the end
of the original pass N is the replay of all fills with version <= max version of that pass's
original sessions. Late (retried) fills are reported separately. Verifies every replayed
post price against the log before writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/home/agent/workspaces/bayes-engine")
from harness.market_game import Market, Trade  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--late-after", required=True, help="ISO timestamp; sessions finished at/after it are retries")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    rows = [json.loads(l) for l in (run / "sessions.jsonl").read_text().splitlines() if l.strip()]
    cfg = json.loads((run / "config.json").read_text()).get("config", {})
    qids = [json.loads(l)["question_id"] for l in (run / "questions.jsonl").read_text().splitlines() if l.strip()]
    fills = []
    boundary: dict[int, int] = {}
    late = 0
    for r in rows:
        if r.get("status") != "completed":
            continue
        is_late = (r.get("finished_at") or "") >= a.late_after
        for e in r.get("executed") or []:
            fills.append((e["version"], int(r["pass"]), is_late, e))
            if not is_late:
                boundary[int(r["pass"])] = max(boundary.get(int(r["pass"]), 0), e["version"])
            else:
                late += 1
    fills.sort(key=lambda x: x[0])
    versions = [f[0] for f in fills]
    assert versions == list(range(1, len(versions) + 1)), "fill versions are not contiguous"
    print("fills", len(fills), "late", late, "pass boundaries", boundary)
    market = Market.fresh(qids, liquidity=float(cfg.get("liquidity", 100.0)), max_width=int(cfg.get("max_width", 8)))
    max_err = 0.0
    out = {}
    for version, p, is_late, e in fills:
        t = e["trade"]
        trade = Trade(t["variable"], float(t["target"]), tuple((c["variable"], c["outcome"]) for c in t.get("context") or []))
        got, rej = market.execute([trade], None)
        if rej:
            raise SystemExit(f"replay rejected fill {version}: {rej[0]['reason']} {rej[0].get('detail')}")
        max_err = max(max_err, abs(got[0]["post"] - e["post"]))
        for pn, b in boundary.items():
            if version == b:
                out[pn] = market.snapshot(meta={"rebuilt_from_log": True, "at_version": version, "pass": pn})
                print(f"pass {pn}: snapshot at version {version}, max |post error| so far {max_err:.2e}")
    print("max replay error", max_err)
    if max_err > 1e-6:
        raise SystemExit("replay diverges from the log; not writing")
    if a.write:
        for pn, snap in out.items():
            (run / "snapshots" / f"pass-{pn}.json").write_text(json.dumps(snap))
        (run / "snapshots" / "latest.json").write_text(json.dumps(market.snapshot(meta={"rebuilt_from_log": True, "includes_late_retries": True})))
        (run / "snapshots" / "REBUILT.txt").write_text(f"rebuilt from log, late_after={a.late_after}, late fills={late}, boundaries={boundary}\n")
        print("written")


if __name__ == "__main__":
    main()
