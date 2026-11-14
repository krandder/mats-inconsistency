"""Data for the GRPO consistency pilot (docs/grpo-consistency-pilot-20261008.md). CPU only, no API calls.

Pairs: every pair of jev-market-context-open-20260923 (the September pairing graph with the open rule: both
questions open at t = 00:00 UTC of the earlier closing week minus 7 days), pooled over its three family splits
and re-split BY TIME:

  1. order pairs by t (ties: sha256 of the pair id);
  2. cut at the 60% and 75% quantiles of that order: train (earliest), validation, test (latest);
  3. embargo: a train pair is kept only if both of its questions resolved (asof.sqlite resolution_time) before
     the first validation t, so no training label postdates any validation or test forecast time (validation
     labels are used only for gates and model selection, never as training targets, so they are not embargoed);
  4. hold out questions: a validation or test pair is dropped if its P or Q parent market or family appears in
     an earlier split (and a test pair if it shares one with validation).

Prompt: the v7b_explicit wording of training/verbalized_baseline.py (one form per call, the literal forms show
only their own question), with three changes for this pilot, applied to every split and arm:
  * the question, option name and criteria are the text AS OF t (the market-context ``question_at_t``,
    ``answer_text_at_t`` and ``criteria.text``), then creator-edit paragraphs are stripped as before;
  * the option name of a multiple-choice proposition is always shown (the fixed builder);
  * the answer instruction asks for brief reasoning, then "Probability: 0.xx" (ANSWER_WITH_REASONING).

Single-forecast test sets (never trained on): the P and Q literal prompts of the test pairs (Manifold, 2025),
plus external 2026 binary questions with known truth from minibench/autoresearch (Metaculus MiniBench rounds
and Summer FutureEval, and 2026 Manifold markets), rendered as a literal "P" prompt of a one-question pair.

Outputs (--out): train.jsonl, validation.jsonl, test.jsonl (12 rows per pair, pair-contiguous, ordered by t),
singles-test.jsonl, singles-validation.jsonl, manifest.json (counts, cut dates, sha256 of every output).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.supervised_pairs import FORM_ORDER, PairExample, load_split  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

PAIRS = Path("/home/agent/data/jev-market-context-open-20260923")
ASOF_DB = Path("/home/agent/data/manifold-asof-20260923/asof.sqlite")
MB = Path("/home/agent/workspaces/calibrated-ai/minibench")
VARIANT = "v7b_explicit"
CUTS = (0.60, 0.75)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asof_proposition(row: dict) -> dict:
    """The proposition with its question, option and criteria text as they stood at t."""
    ctx = row.get("context") or {}
    out = copy.deepcopy({k: v for k, v in row.items() if k != "context"})
    if ctx.get("question_at_t"):
        out["parent_question"] = ctx["question_at_t"]
    if row.get("answer_text") is not None or ctx.get("answer_text_at_t"):
        if ctx.get("answer_text_at_t"):
            out["answer_text"] = ctx["answer_text_at_t"]
    crit = ctx.get("criteria")
    if isinstance(crit, dict) and crit.get("text") is not None:
        out["criteria_text"] = crit["text"]
    out["_asof_text"] = bool(ctx)
    return out


def user_prompt(ex: PairExample, form: str) -> str:
    text = vb.user_prompt(ex, form, VARIANT)
    assert text.endswith(vb.ANSWER_DIRECT + " /no_think"), "v7b answer line changed"
    return text[: -len(vb.ANSWER_DIRECT + " /no_think")] + vb.ANSWER_WITH_REASONING + " /no_think"


def resolution_times(db: Path, ids: set[str]) -> dict[str, int | None]:
    """Proposition id -> resolution time in ms (market, or the answer for 'market:answer' ids)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, int | None] = {}
    for pid in ids:
        if ":" in pid:
            _, aid = pid.split(":", 1)
            r = con.execute("select resolution_time from answers where id=?", (aid,)).fetchone()
            if r is None or r[0] is None:  # sum-to-one markets resolve all answers with the market
                r = con.execute("select resolution_time from markets where id=?", (pid.split(":")[0],)).fetchone()
        else:
            r = con.execute("select resolution_time from markets where id=?", (pid,)).fetchone()
        out[pid] = r[0] if r else None
    return out


def ms(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def build_pairs(out: Path) -> dict:
    examples: list[PairExample] = []
    for split in ("train", "validation", "test"):
        examples += load_split(PAIRS, split, labels_file="full-label-targets.jsonl")
    for ex in examples:
        for k in ("p_id", "q_id"):
            pid = ex.pair[k]
            ex.propositions[pid] = asof_proposition(ex.propositions[pid])
    ids = {ex.pair[k] for ex in examples for k in ("p_id", "q_id")}
    res = resolution_times(ASOF_DB, ids)
    info = []
    for ex in examples:
        rp, rq = ex.propositions[ex.pair["p_id"]], ex.propositions[ex.pair["q_id"]]
        t = vb.as_of_date(min(rp["closing_week"], rq["closing_week"]))
        rt = [res[ex.pair["p_id"]], res[ex.pair["q_id"]]]
        info.append({"ex": ex, "t": t, "resolved_ms": None if None in rt else max(rt),
                     "parents": {rp["parent_id"], rq["parent_id"]}, "families": {rp["family_id"], rq["family_id"]},
                     "key": hashlib.sha256(ex.pair["pair_id"].encode()).hexdigest()})
    info.sort(key=lambda r: (r["t"], r["key"]))
    n = len(info)
    a, b = int(n * CUTS[0]), int(n * CUTS[1])
    raw = {"train": info[:a], "validation": info[a:b], "test": info[b:]}
    t_val, t_test = raw["validation"][0]["t"], raw["test"][0]["t"]
    counts = {"pooled_pairs": n, "raw": {k: len(v) for k, v in raw.items()}}
    kept = {"train": [r for r in raw["train"] if r["resolved_ms"] is not None and r["resolved_ms"] < ms(t_val)],
            "validation": [r for r in raw["validation"] if r["resolved_ms"] is not None],
            "test": [r for r in raw["test"] if r["resolved_ms"] is not None]}
    counts["after_embargo"] = {k: len(v) for k, v in kept.items()}
    seen_par, seen_fam = set(), set()
    for split in ("train", "validation", "test"):
        if split != "train":
            kept[split] = [r for r in kept[split] if not (r["parents"] & seen_par or r["families"] & seen_fam)]
        for r in kept[split]:
            seen_par |= r["parents"]
            seen_fam |= r["families"]
    counts["after_question_holdout"] = {k: len(v) for k, v in kept.items()}
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in kept.items():
        with (out / f"{split}.jsonl").open("w") as fh:
            for r in rows:
                ex = r["ex"]
                rp, rq = ex.propositions[ex.pair["p_id"]], ex.propositions[ex.pair["q_id"]]
                for form in FORM_ORDER:
                    qid = ex.queries[form]["query_id"]
                    fh.write(json.dumps({
                        "pair_id": ex.pair["pair_id"], "form": form, "query_id": qid, "label": int(ex.labels[qid]),
                        "t": r["t"], "source": "manifold-2025", "p_id": ex.pair["p_id"], "q_id": ex.pair["q_id"],
                        "p_parent": rp["parent_id"], "q_parent": rq["parent_id"],
                        "system": vb.BASE_SYSTEM, "user": user_prompt(ex, form)}) + "\n")
    # literal P/Q singles of the test and validation pairs
    for split in ("validation", "test"):
        with (out / f"singles-{split}.jsonl").open("w") as fh:
            for r in kept[split]:
                ex = r["ex"]
                for form, key in (("P", "p_id"), ("Q", "q_id")):
                    qid = ex.queries[form]["query_id"]
                    prop = ex.propositions[ex.pair[key]]
                    fh.write(json.dumps({"qid": ex.pair[key], "cluster": prop["parent_id"], "source": "manifold-2025",
                                         "pair_id": ex.pair["pair_id"], "form": form, "label": int(ex.labels[qid]),
                                         "t": r["t"], "system": vb.BASE_SYSTEM, "user": user_prompt(ex, form)}) + "\n")
    labels = Counter()
    for split, rows in kept.items():
        for r in rows:
            labels[split] += 1
    meta = {"t_validation_start": t_val, "t_test_start": t_test,
            "t_range": {k: [v[0]["t"], v[-1]["t"]] if v else None for k, v in kept.items()},
            "asof_text_share": sum(r["ex"].propositions[r["ex"].pair["p_id"]]["_asof_text"] for v in kept.values()
                                   for r in v) / max(1, sum(len(v) for v in kept.values())),
            "counts": counts}
    return meta


def week_monday(iso: str) -> str:
    d = dt.date.fromisoformat(iso[:10])
    return (d - dt.timedelta(days=d.weekday())).isoformat()


DECIDED_AT_ASOF = {"mf-66016", "mf-98908"}  # "before 2026" questions asked in 2026 (verifier, 2026-10-08)


def build_external_singles(out: Path) -> dict:
    """2026 binary questions with known truth (MiniBench, FutureEval, Manifold) from minibench/autoresearch.

    Manifold 2026 questions use the title and description as of their as-of time (asof.sqlite market_versions,
    mid = the registry post_id); the snapshots were taken after resolution.
    """
    reg = json.loads((MB / "autoresearch/data/registry.json").read_text())
    con = sqlite3.connect(f"file:{ASOF_DB}?mode=ro", uri=True)
    rows, skipped = [], Counter()
    for key, v in sorted(reg.items()):
        if v["type"] != "binary" or key == "mb-45000":  # mb-45000: confirmed leak (autoresearch manifest)
            continue
        if key in DECIDED_AT_ASOF:
            skipped["decided_at_asof"] += 1
            continue
        truth = json.loads((MB / v["truth"]).read_text())
        if truth.get("resolution") not in ("yes", "no"):
            skipped["not_yes_no"] += 1
            continue
        snap = json.loads((MB / v["snapshot"]).read_text())
        crit = "\n\n".join(x for x in (snap.get("resolution_criteria") or snap.get("description") or "",
                                        snap.get("fine_print") or "") if x).strip()
        asof = v["asof"][:10]
        title = snap["title"]
        if v["source"] == "manifold":
            at = ms(asof)
            ver = con.execute("""select question, description from market_versions where mid = ? and valid_from <= ?
                                 and valid_to > ? order by valid_from desc limit 1""", (int(v["post_id"]), at, at)).fetchone()
            if ver is None:
                skipped["manifold_no_asof_version"] += 1
                continue
            title, crit = ver[0] or title, (ver[1] or "").strip()
            skipped["manifold_asof_text_used"] += 1
        row = {"parent_question": title, "answer_text": None, "criteria_text": crit,
               "closing_week": week_monday(snap.get("scheduled_resolve_time") or snap["scheduled_close_time"]),
               "parent_id": key, "family_id": key}
        pair = {"pair_id": key, "p_id": key, "q_id": key + "#b"}
        ex = PairExample(pair=pair, queries={}, propositions={key: row, key + "#b": row}, labels={})
        text = vb.user_prompt(ex, "P", VARIANT)
        # one-question frame: today = the as-of of the question (its open/snapshot time), not close - 7 d
        head = text.split("\n\n", 1)
        text = (f"Assume today is {asof}. Use only information available up to that date; do not use knowledge of "
                f"later events.\n\n{head[1]}")
        text = text[: -len(vb.ANSWER_DIRECT + " /no_think")] + vb.ANSWER_WITH_REASONING + " /no_think"
        src = {"minibench": "metaculus-minibench-2026", "futureeval": "metaculus-futureeval-2026",
               "manifold": "manifold-2026"}[v["source"]]
        rows.append({"qid": key, "cluster": key, "source": src, "pair_id": None, "form": "P",
                     "label": int(truth["resolution"] == "yes"), "t": asof, "system": vb.BASE_SYSTEM, "user": text})
    with (out / "singles-external.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return {"external_singles": dict(Counter(r["source"] for r in rows)), "external_skipped": dict(skipped),
            "external_yes_rate": sum(r["label"] for r in rows) / max(1, len(rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = build_pairs(args.out)
    meta.update(build_external_singles(args.out))
    meta["inputs"] = {"pairs": str(PAIRS), "pairs_manifest_sha256": sha(PAIRS / "manifest.json"),
                      "asof_db": str(ASOF_DB), "minibench_registry_sha256": sha(MB / "autoresearch/data/registry.json"),
                      "variant": VARIANT, "cuts": CUTS}
    meta["outputs_sha256"] = {p.name: sha(p) for p in sorted(args.out.glob("*.jsonl"))}
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "outputs_sha256"}, indent=1))


if __name__ == "__main__":
    main()
