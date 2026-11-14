#!/usr/bin/env python3
"""Prequential study, step 1: one time-ordered stream of all pairs (CPU, no API calls).

Reads the consistency-distillation bundle (train, validation and test rows of
jev-disjoint-v1-filtered-20260919: 12 forms per pair, teacher raw and kl-projected answers,
student prompts) and the closing week of every proposition, and writes one stream:

  stream order  pairs sorted by their first close date (the earlier closing week of P and Q), which
                fixes the forecast date: as_of = first close - 7 days, the "Assume today is" date of
                every prompt (checked). Ties: the later close date, then sha256(pair_id).
  blocks        the 3,652 pairs cut into NBLOCKS consecutive blocks of ~100 pairs; block time
                T_k = the earliest as_of in block k.
  availability  teacher targets (t_raw, t_proj) of a row are known at its pair's as_of;
                its outcome is known at the end of the closing week of every event it depends on:
                P, NOT P -> close(P) + 7 days; Q, NOT Q -> close(Q) + 7 days; the 8 joints -> the later
                of the two + 7 days (conservative: a joint settled early by one event still waits).

A training item may be used before scoring block k only if it was scored in an earlier block and is
available at T_k (the student script enforces this; the stream only carries the dates).

Outputs in --out (default /home/agent/data/prequential-20260929/bundle):
  stream.jsonl.gz  one row per (pair, form) in stream order: idx block pair_id query_id form split
                   label t_raw t_proj as_of avail_outcome prompt
  blocks.json      block sizes, times and date ranges
  manifest.json    sha256 of inputs and outputs, counts, availability summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DISTILL = Path("/home/agent/data/consistency-distill-20260929/bundle")
DATA = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919")
SPLITS = ("train", "validation", "test")
SINGLES = {"P": "P", "NOT P": "P", "Q": "Q", "NOT Q": "Q"}


def day(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/agent/data/prequential-20260929/bundle")
    ap.add_argument("--blocks", type=int, default=36)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    close = {}
    for s in SPLITS:
        for line in (DATA / s / "propositions.jsonl").read_text().splitlines():
            r = json.loads(line); close[r["id"]] = r["closing_week"]
    rows_by_pair = defaultdict(list)
    for s in SPLITS:
        for line in (DISTILL / f"{s}.jsonl").read_text().splitlines():
            r = json.loads(line); r["split"] = s
            rows_by_pair[r["pair_id"]].append(r)
    assert all(len(v) == 12 for v in rows_by_pair.values())

    meta = {}
    for pid, rs in rows_by_pair.items():
        cp, cq = day(close[rs[0]["p_id"]]), day(close[rs[0]["q_id"]])
        as_of = min(cp, cq) - dt.timedelta(days=7)
        for r in rs:
            m = re.search(r"Assume today is (\d{4}-\d{2}-\d{2})\.", r["prompt"])
            assert m and m.group(1) == as_of.isoformat(), (pid, m and m.group(1), as_of)
        meta[pid] = {"close_p": cp, "close_q": cq, "as_of": as_of, "first": min(cp, cq), "last": max(cp, cq)}

    order = sorted(meta, key=lambda p: (meta[p]["first"], meta[p]["last"], hashlib.sha256(p.encode()).hexdigest()))
    n = len(order); k = args.blocks
    bounds = [round(i * n / k) for i in range(k + 1)]
    block_of = {p: b for b in range(k) for p in order[bounds[b]:bounds[b + 1]]}

    week = dt.timedelta(days=7)
    idx = 0
    blocks = []
    avail_count = Counter()
    with gzip.open(out / "stream.jsonl.gz", "wt") as f:
        for b in range(k):
            pairs = order[bounds[b]:bounds[b + 1]]
            T = min(meta[p]["as_of"] for p in pairs)
            blocks.append({"block": b, "pairs": len(pairs), "rows": 12 * len(pairs), "T": T.isoformat(),
                           "as_of_min": T.isoformat(), "as_of_max": max(meta[p]["as_of"] for p in pairs).isoformat(),
                           "last_close_max": max(meta[p]["last"] for p in pairs).isoformat(),
                           "splits": dict(Counter(rows_by_pair[p][0]["split"] for p in pairs))})
            for p in pairs:
                m = meta[p]
                for r in sorted(rows_by_pair[p], key=lambda r: r["query_id"]):
                    ev = SINGLES.get(r["form"])
                    avail = (m["close_p"] if ev == "P" else m["close_q"] if ev == "Q" else m["last"]) + week
                    row = {"idx": idx, "block": b, "pair_id": p, "query_id": r["query_id"], "form": r["form"],
                           "split": r["split"], "label": r["label"], "t_raw": r["t_raw"], "t_proj": r["t_proj"],
                           "as_of": m["as_of"].isoformat(), "avail_outcome": avail.isoformat(), "prompt": r["prompt"]}
                    f.write(json.dumps(row) + "\n"); idx += 1
    # how many outcome rows become usable before the last block is scored (the O arm's whole training set)
    T = [day(b["T"]) for b in blocks]
    usable = 0
    first_use = []
    for b, pairs in enumerate(order[bounds[i]:bounds[i + 1]] for i in range(k)):
        for p in pairs:
            m = meta[p]
            for form in ("P", "Q", "J"):
                a = (m["close_p"] if form == "P" else m["close_q"] if form == "Q" else m["last"]) + week
                nrows = 2 if form in "PQ" else 8
                later = [j for j in range(b + 1, k) if T[j] >= a]
                if later:
                    usable += nrows; first_use.append(later[0] - b)
    (out / "blocks.json").write_text(json.dumps(blocks, indent=1) + "\n")
    first_use.sort()
    manifest = {
        "pairs": n, "rows": idx, "blocks": k, "block_bounds": bounds,
        "order": "first close date (min closing week of P and Q) = as_of + 7 days; ties: last close, sha256(pair_id)",
        "rows_by_split": {s: 12 * sum(rows_by_pair[p][0]["split"] == s for p in order) for s in SPLITS},
        "as_of_range": [meta[order[0]]["as_of"].isoformat(), meta[order[-1]]["as_of"].isoformat()],
        "teacher_rows_trainable": idx - blocks[-1]["rows"],
        "outcome_rows_trainable": usable,
        "outcome_delay_blocks": {"median": first_use[len(first_use) // 2] if first_use else None,
                                 "p90": first_use[int(0.9 * len(first_use))] if first_use else None},
        "inputs": {**{f"distill/{s}.jsonl": sha256(DISTILL / f"{s}.jsonl") for s in SPLITS},
                   **{f"jev/{s}/propositions.jsonl": sha256(DATA / s / "propositions.jsonl") for s in SPLITS}},
        "outputs": {"stream.jsonl.gz": sha256(out / "stream.jsonl.gz"), "blocks.json": sha256(out / "blocks.json")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k2: v for k2, v in manifest.items() if k2 not in ("block_bounds", "inputs", "outputs")}, indent=1))
    for b in blocks[:3] + blocks[-3:]:
        print(b)


if __name__ == "__main__":
    main()
