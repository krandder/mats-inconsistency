"""Training data root for the Qwen3-0.6B structured-pairs run (27 Sep 2026).

Turns three structured-pair sets (scripts/structured_pairs.py) into the split layout that
training/supervised_pairs.py loads:

  train       structured_pairs.py build --split train       (JEV train + pre-JEV parents)
  validation  structured_pairs.py build --split validation  (JEV validation + pre-JEV parents)
  test        /home/agent/data/structured-pairs-20260924    (780 structured + 297 controls)

Per split: propositions.jsonl, pairs.jsonl, queries.jsonl (12 forms), full-label-targets.jsonl
(all 12 forms from the two outcomes), one-label-targets.jsonl (one seeded observed form per pair)
and prompts.jsonl (per pair a shared prefix and 12 suffixes).

Prompt (arm P): the zero-shot test's content -- system line, "Assume today is t", the
deterministic evidence pack (training/asof.build_pack(pair, t, "small"), ~2k tokens, the same pack
file the zero-shot runs used for the test), both as-of-t question blocks -- ending in the P(YES)
readout of the earlier 0.6B runs:  "Statement to assess: ...\\nIs this statement true? Answer [YES/NO]:".
The prefix is identical for the 12 forms of a pair; only the statement suffix differs.

Usage: build_qwen06b_structured_data.py --out /home/agent/data/qwen06b-structured-20260927/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import verbalized_baseline as vb  # noqa: E402
from training.supervised_pairs import FORM_ORDER, _expected_signs, _statement  # noqa: E402

BASE = Path("/home/agent/data/qwen06b-structured-20260927")
SOURCES = {"train": BASE / "pairs-train", "validation": BASE / "pairs-validation",
           "test": Path("/home/agent/data/structured-pairs-20260924")}
SEED = 20260927


def truth(form: str, p: int, q: int) -> int:
    """Label of a form from the two outcomes (FORM_ORDER names)."""
    ps, qs = _expected_signs(form)
    a = p if ps == 1 else 1 - p
    b = q if qs == 1 else 1 - q
    if qs == 0:
        return a
    if ps == 0:
        return b
    return int(a and b) if " AND " in form else int(a or b)


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def block(row: dict) -> str:
    return vb._question_lines(row, row["criteria"].strip(), vb.MAX_CRITERIA_CHARS)


def prefix(pair: dict, pack: str) -> str:
    return (f"{vb.BASE_SYSTEM}\n\nAssume today is {pair['t'][:10]}. Use only information available up to that date; "
            f"do not use knowledge of later events.\n\n{pack.strip()}\n\n"
            f"Question A\n{block(pair['p_text'])}\n\nQuestion B\n{block(pair['q_text'])}\n\n")


def suffix(form: str) -> str:
    return f"Statement to assess: {_statement(form)}\nIs this statement true? Answer [YES/NO]:"


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_split(split: str, src: Path, out: Path) -> dict:
    pairs = jl(src / "pairs.jsonl")
    labels = {r["pair_id"]: (r["p"], r["q"]) for r in jl(src / "labels.jsonl")}
    packs = {r["pair_id"]: r["text"] for r in jl(src / "packs-structured.jsonl")}
    rng = random.Random(f"{SEED}-{split}")
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    props, prow, qrows, full, one, prompts = {}, [], [], [], [], []
    for p in pairs:
        pid = p["pair_id"]
        for side in ("p", "q"):
            t = p[f"{side}_text"]
            props.setdefault(t["id"], {"id": t["id"], "parent_id": t["parent_id"], "parent_question": t["parent_question"],
                                       "answer_text": t["answer_text"], "criteria_text": t["criteria"],
                                       "closing_week": t["closing_week"]})
        yp, yq = labels[pid]
        qids = {}
        for k, form in enumerate(FORM_ORDER):
            qid = f"{pid}-q{k:02d}"
            qids[form] = qid
            ps, qs = _expected_signs(form)
            qrows.append({"query_id": qid, "pair_id": pid, "form": form, "p_id": p["p_id"], "q_id": p["q_id"],
                          "p_sign": ps, "q_sign": qs})
            full.append({"query_id": qid, "label": truth(form, yp, yq)})
        observed = rng.choice(FORM_ORDER)
        one.append({"query_id": qids[observed], "label": truth(observed, yp, yq)})
        prow.append({"pair_id": pid, "p_id": p["p_id"], "q_id": p["q_id"], "observed_query_id": qids[observed],
                     "type": p["type"], "group": p["group"], "t": p["t"], "relation": p["relation"],
                     "p_family": p["p_parent"], "q_family": p["q_parent"], "pair_weight": 1.0})
        prompts.append({"pair_id": pid, "prefix": prefix(p, packs[pid]), "suffix": {f: suffix(f) for f in FORM_ORDER}})
    write(d / "propositions.jsonl", list(props.values()))
    write(d / "pairs.jsonl", prow)
    write(d / "queries.jsonl", qrows)
    write(d / "full-label-targets.jsonl", full)
    write(d / "one-label-targets.jsonl", one)
    write(d / "prompts.jsonl", prompts)
    from collections import Counter
    return {"pairs": len(prow), "by_type": dict(Counter(r["type"] for r in prow)),
            "parents": len({r["p_family"] for r in prow} | {r["q_family"] for r in prow}),
            "source": str(src), "pairs_sha256": hashlib.sha256((src / "pairs.jsonl").read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BASE / "data")
    a = ap.parse_args()
    stats = {s: build_split(s, src, a.out) for s, src in SOURCES.items()}
    parents = {s: {json.loads(l)["p_family"] for l in (a.out / s / "pairs.jsonl").open()} |
               {json.loads(l)["q_family"] for l in (a.out / s / "pairs.jsonl").open()} for s in SOURCES}
    stats["parent_overlap"] = {f"{x}&{y}": len(parents[x] & parents[y])
                               for x, y in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(stats["parent_overlap"].values()):
        raise SystemExit(f"parent overlap between splits: {stats['parent_overlap']}")
    (a.out / "build-stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
