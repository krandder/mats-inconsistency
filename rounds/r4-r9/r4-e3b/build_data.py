"""R4-E3b data: 2,500 resolved binary Manifold markets, each frozen at one as-of time t (evidence as of t only).

Source: /home/agent/data/manifold-asof-20260923/asof.sqlite (read-only).
Per market: t = midpoint of [created, min(close, resolution_time)]; question and description from the version
valid at t (market_versions); market probability at t = prob_after of the last non-redemption bet at or before t.
Temporal splits (strict): train markets RESOLVED before C_VAL; val markets with t in [C_VAL, C_TEST) and resolved
before C_TEST; test markets with t >= C_TEST. So no training outcome is known later than any val/test as-of time.
At most 4 markets per creator per split, and no question prefix (first 40 chars, lowercased) repeats anywhere.

  python3 r4-e3b/build_data.py --out r4-e3b/data
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DB = "/home/agent/data/manifold-asof-20260923/asof.sqlite"
MS = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)  # noqa: E731
START = MS("2023-06-01")          # market_versions are exact after edit recording started (2023-05-04)
C_VAL = MS("2025-09-01")
C_TEST = MS("2026-01-01")
N = {"train": 1700, "val": 300, "test": 500}
DAY = 86_400_000


def asof_item(c, row):
    mid, question, desc, creator, created, close, res_time, res = row
    end = min(x for x in (close, res_time) if x)
    if end - created < 3 * DAY:
        return None
    t = created + (end - created) // 2
    nb = c.execute("select count(*) from bets where mid=? and time<=? and coalesce(is_redemption,0)=0",
                   (mid, t)).fetchone()[0]
    if nb < 10:
        return None
    p = c.execute("select prob_after from bets where mid=? and time<=? and coalesce(is_redemption,0)=0 "
                  "and prob_after is not null order by time desc limit 1", (mid, t)).fetchone()
    if not p or not (0.05 <= p[0] <= 0.95):
        return None
    v = c.execute("select question, description from market_versions where mid=? and valid_from<=? and valid_to>? "
                  "order by valid_from desc limit 1", (mid, t, t)).fetchone()
    q, d = (v if v and v[0] else (question, desc))
    return {"mid": mid, "question": (q or "").strip(), "description": " ".join((d or "").split())[:300],
            "creator": creator, "created": created, "t": t, "close": close, "resolution_time": res_time,
            "asof_date": datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "close_date": datetime.fromtimestamp((close or end) / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "market_prob": round(float(p[0]), 3), "n_bets_asof": nb, "y": 1 if res == "YES" else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = c.execute("select mid, question, description, creator_id, created, close, resolution_time, resolution "
                     "from markets where outcome_type='BINARY' and resolution in ('YES','NO') "
                     "and coalesce(deleted,0)=0 and created>=? and resolution_time is not null",
                     (START,)).fetchall()
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    out = {k: [] for k in N}
    per_creator = {k: Counter() for k in N}
    seen_prefix = set()
    for row in rows:
        if all(len(out[k]) >= N[k] for k in N):
            break
        created, res_time = row[4], row[6]
        close = row[5] or res_time
        end = min(close, res_time)
        t_est = created + (end - created) // 2
        if res_time < C_VAL:
            split = "train"
        elif t_est >= C_VAL and t_est < C_TEST and res_time < C_TEST:
            split = "val"
        elif t_est >= C_TEST:
            split = "test"
        else:
            continue
        if len(out[split]) >= N[split] or per_creator[split][row[3]] >= 4:
            continue
        pre = (row[1] or "").lower()[:40]
        if pre in seen_prefix:
            continue
        it = asof_item(c, row)
        if it is None:
            continue
        seen_prefix.add(pre)
        per_creator[split][row[3]] += 1
        out[split].append(it)
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for k, items in out.items():
        items.sort(key=lambda x: x["t"])
        for i, it in enumerate(items):
            it["id"] = f"{k}-{i:04d}"
        (a.out / f"{k}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
        ys = [x["y"] for x in items]
        ps = [x["market_prob"] for x in items]
        br = sum(ys) / len(ys)
        stats[k] = {"n": len(items), "base_rate": round(br, 3),
                    "brier_base": round(sum((br - y) ** 2 for y in ys) / len(ys), 4),
                    "brier_market": round(sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys), 4),
                    "t_min": items[0]["asof_date"], "t_max": items[-1]["asof_date"],
                    "max_resolution": datetime.fromtimestamp(max(x["resolution_time"] for x in items) / 1000,
                                                             timezone.utc).strftime("%Y-%m-%d")}
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
