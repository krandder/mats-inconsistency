#!/usr/bin/env python3
"""Build Turtel-format training/validation prompts from the Exa evidence pool and the question set.

Inputs (defaults point at General's copies):
  --questions  polymarket-train-20260922/questions.jsonl  (question_full, description, close_date, resolution, split)
  --pool       polymarket-evidence-exa-20260922/pool.jsonl (prediction_date, dated results with highlights)
  --exclude-flagged leak-flags.jsonl                       (optional; produced later by the leak-screen job)

Output (``--out-dir``): ``train.jsonl``, ``validation.jsonl`` with one record per question
  {id, prompt, resolution, prediction_date, close_date, split, n_articles, dataset_split, question,
   n_flagged_dropped, prompt_chars}
ordered by prediction_date (then id), plus ``manifest.json`` (counts, args, sha256 of inputs and outputs).

Prompt format: ``training/turtel_prompt.py`` (verified byte-exact against the 1,265 released test prompts with
``--verify``). Mapping: question_full -> Question; "" -> Question Background (empty in every test row);
description -> Resolution Criteria; prediction_date -> Today's date; close_date -> Question close date at
00:00:00; Exa results -> numbered articles (title, publishedDate as returned, host without www., highlights
joined by a space as the Summary, cut at a sentence end before --max-summary-chars), newest first, at most
--max-articles, deduplicated by url.

Leak flags contract (``--exclude-flagged``): one JSON object per line with ``id`` and ``url``; optional
``snippet_index`` (drop only that highlight; otherwise the whole result). A row is a flag when any of
``flagged``/``leak`` is true, ``verdict`` is in {leak, yes, flag}, or ``answers[k]`` is truthy/"yes" for a key in
--flag-keys (default: outcome_leak, post_date_event, date_inconsistent), so the leak-questions.jsonl scaffold
with answers filled in works unchanged. Questions whose evidence is entirely flagged away are dropped;
questions that never had evidence keep the test set's "No information found" block (33 of 1,265 test rows).

Split policy: ``tail`` (default) holds out the last --validation-size questions by prediction_date as the
chronological validation slice; ``dataset`` uses the split field from questions.jsonl (close < 2024-11-01 train).
The original dataset split is always kept in ``dataset_split``.

Stdlib only. ``--verify`` needs pyarrow (venv-cft) unless the test set is given as JSONL.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.turtel_prompt import (Article, UNKNOWN_DATE, close_date_to_prompt_datetime,  # noqa: E402
                                    iso_to_prompt_datetime, order_articles, render_prompt, source_from_url,
                                    structural_signature, verify_against_test_prompts)

QUESTIONS = Path("/home/agent/data/polymarket-train-20260922/questions.jsonl")
POOL = Path("/home/agent/data/polymarket-evidence-exa-20260922/pool.jsonl")
TEST_PARQUET = Path("/home/agent/data/forecasting-benchmarks/LightningRodLabs--outcome-rl-test-dataset/data/"
                    "train-00000-of-00001.parquet")
DEFAULT_FLAG_KEYS = ("outcome_leak", "post_date_event", "date_inconsistent")
TRUTHY = {"yes", "true", "1", "y", "leak", "flag", "flagged"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY
    return bool(value)


def load_flags(path: Path, flag_keys: tuple[str, ...]) -> tuple[dict, collections.Counter]:
    """-> {(id, url): set of snippet indices or {None} meaning the whole result}, counts."""
    flags: dict[tuple[str, str], set] = collections.defaultdict(set)
    counts = collections.Counter()
    for row in iter_jsonl(path):
        counts["rows"] += 1
        flagged = is_truthy(row.get("flagged")) or is_truthy(row.get("leak"))
        verdict = row.get("verdict")
        if isinstance(verdict, str) and verdict.strip().lower() in TRUTHY:
            flagged = True
        answers = row.get("answers")
        if isinstance(answers, dict):
            for key in flag_keys:
                if is_truthy(answers.get(key)):
                    flagged = True
                    counts[f"answers.{key}"] += 1
        if not flagged:
            continue
        if "id" not in row or "url" not in row:
            counts["flag_rows_without_id_url"] += 1
            continue
        counts["flagged_rows"] += 1
        flags[(row["id"], row["url"])].add(row.get("snippet_index"))
    return flags, counts


def truncate_text(text: str, limit: int, stats: collections.Counter, key: str) -> str:
    """Cut at the last sentence end (or Exa's ' ... ' excerpt separator) before ``limit`` characters."""
    if limit <= 0 or len(text) <= limit:
        return text
    stats[key] += 1
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind("? "), head.rfind("! "), head.rfind(" ... "))
    if cut >= limit // 3:
        head = head[:cut + 1]
    return head.rstrip(" .") + "."


def build_articles(pool_row: dict, flags: dict, args, stats: collections.Counter) -> tuple[list[Article], int]:
    seen = set()
    candidates = []
    dropped = 0
    for result in pool_row.get("results") or []:
        url = result.get("url") or ""
        if not url or url in seen:
            stats["duplicate_or_empty_url"] += 1
            continue
        seen.add(url)
        highlights = [h.strip() for h in (result.get("highlights") or []) if h and h.strip()]
        flag = flags.get((pool_row["id"], url))
        if flag is not None:
            if None in flag:
                highlights = []
            else:
                highlights = [h for i, h in enumerate(highlights) if i not in flag]
            if not highlights:
                dropped += 1
                stats["results_dropped_by_flags"] += 1
                continue
            stats["results_with_partial_flags"] += 1
        if not highlights:
            stats["results_without_highlights"] += 1
            continue
        summary = " ".join(" ".join(h.split()) for h in highlights)
        summary = truncate_text(summary, args.max_summary_chars, stats, "summaries_truncated")
        title = " ".join((result.get("title") or "").split()) or url
        if args.max_title_chars > 0 and len(title) > args.max_title_chars:
            stats["titles_truncated"] += 1
            title = title[:args.max_title_chars].rstrip() + "..."
        published = result.get("publishedDate") or UNKNOWN_DATE
        candidates.append(((result.get("rank_in_query", 0), result.get("query_idx", 0)),
                           Article(title=title, published=published, source=source_from_url(url),
                                   summary=summary)))
    candidates.sort(key=lambda item: item[0])
    chosen = [article for _, article in candidates[:args.max_articles]]
    stats["results_beyond_max_articles"] += max(0, len(candidates) - args.max_articles)
    return order_articles(chosen), dropped


def load_test_prompts(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # venv-cft
        return [str(p) for p in pq.read_table(str(path), columns=["prompt"]).column("prompt").to_pylist()]
    prompts = []
    for row in iter_jsonl(path):
        if "prompt" in row:
            prompts.append(row["prompt"])
    return prompts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out-dir", type=Path, default=Path("data/turtel-prompts-20260922"))
    parser.add_argument("--exclude-flagged", type=Path, default=None, help="leak-flags.jsonl (see module doc)")
    parser.add_argument("--flag-keys", default=",".join(DEFAULT_FLAG_KEYS))
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--max-summary-chars", type=int, default=600,
                        help="cap per-article summary (Exa highlights: median 1,320 chars; Turtel's LLM summaries: "
                             "median 274, max 882); 0 = no cap")
    parser.add_argument("--max-title-chars", type=int, default=300, help="0 = no cap (test max 236)")
    parser.add_argument("--max-description-chars", type=int, default=4000)
    parser.add_argument("--split-policy", choices=["tail", "dataset"], default="tail")
    parser.add_argument("--validation-size", type=int, default=500, help="tail policy: last N by prediction_date")
    parser.add_argument("--order", choices=["prediction_date", "close_date"], default="prediction_date")
    parser.add_argument("--verify", type=Path, nargs="?", const=TEST_PARQUET, default=None,
                        help="re-render the released test prompts (parquet or jsonl) and diff structures")
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N pool rows")
    args = parser.parse_args(argv)

    started = time.time()
    stats: collections.Counter = collections.Counter()
    flag_keys = tuple(k.strip() for k in args.flag_keys.split(",") if k.strip())
    flags, flag_counts = ({}, collections.Counter())
    if args.exclude_flagged is not None:
        flags, flag_counts = load_flags(args.exclude_flagged, flag_keys)
        print(f"leak flags: {dict(flag_counts)} -> {len(flags)} (id,url) pairs", flush=True)

    questions = {}
    for row in iter_jsonl(args.questions):
        questions[row["id"]] = {"question": row.get("question_full") or row["question"],
                                "description": row.get("description") or "",
                                "close_date": row["close_date"], "resolution": int(row["resolution"]),
                                "split": row.get("split")}
    print(f"questions: {len(questions)}", flush=True)

    records = []
    for n, pool_row in enumerate(iter_jsonl(args.pool)):
        if args.limit is not None and n >= args.limit:
            break
        stats["pool_rows"] += 1
        question = questions.get(pool_row["id"])
        if question is None:
            stats["pool_rows_without_question"] += 1
            continue
        had_results = bool(pool_row.get("results"))
        articles, dropped = build_articles(pool_row, flags, args, stats)
        if had_results and not articles and dropped:
            stats["questions_dropped_all_evidence_flagged"] += 1
            continue
        if not articles:
            stats["questions_without_evidence"] += 1
        criteria = question["description"]
        if len(criteria) > args.max_description_chars:
            criteria = criteria[:args.max_description_chars].rstrip() + " [...]"
            stats["descriptions_truncated"] += 1
        prompt = render_prompt(question=question["question"], background="", criteria=criteria,
                               today=iso_to_prompt_datetime(pool_row["prediction_date"]),
                               close=close_date_to_prompt_datetime(question["close_date"]), articles=articles)
        records.append({"id": pool_row["id"], "prompt": prompt, "resolution": question["resolution"],
                        "prediction_date": pool_row["prediction_date"], "close_date": question["close_date"],
                        "split": None, "n_articles": len(articles), "dataset_split": question["split"],
                        "question": question["question"], "n_flagged_dropped": dropped,
                        "prompt_chars": len(prompt)})
        stats[f"articles_{len(articles)}"] += 1

    key = (lambda r: (r["prediction_date"], r["id"])) if args.order == "prediction_date" else \
        (lambda r: (r["close_date"], r["prediction_date"], r["id"]))
    records.sort(key=key)
    if args.split_policy == "tail":
        cut = max(0, len(records) - args.validation_size)
        for i, record in enumerate(records):
            record["split"] = "train" if i < cut else "validation"
    else:
        for record in records:
            record["split"] = record["dataset_split"] or "train"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ("train", "validation"):
        path = args.out_dir / f"{split}.jsonl"
        rows = [r for r in records if r["split"] == split]
        with path.open("w") as handle:
            for r in rows:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        outputs[split] = {"path": str(path), "n": len(rows), "sha256": sha256_file(path),
                          "yes_rate": round(sum(r["resolution"] for r in rows) / len(rows), 4) if rows else None,
                          "prediction_date_range": [rows[0]["prediction_date"], rows[-1]["prediction_date"]] if rows else None,
                          "mean_articles": round(sum(r["n_articles"] for r in rows) / len(rows), 3) if rows else None,
                          "prompt_chars": {"min": min(r["prompt_chars"] for r in rows), "max": max(r["prompt_chars"] for r in rows),
                                           "mean": round(sum(r["prompt_chars"] for r in rows) / len(rows))} if rows else None}
        print(f"{split}: {len(rows)} rows -> {path}", flush=True)

    verification = None
    if args.verify is not None:
        test_prompts = load_test_prompts(args.verify)
        verification = verify_against_test_prompts(test_prompts)
        # Structure diff: every training prompt's signature must equal the signature of a test prompt with
        # articles (or of a "No information found" test prompt when it has none).
        test_signatures = {}
        for prompt in test_prompts:
            test_signatures.setdefault("articles" if "ARTICLES" in prompt else "none", structural_signature(prompt))
        mismatches = []
        for record in records:
            kind = "articles" if record["n_articles"] else "none"
            signature = structural_signature(record["prompt"])
            if signature != test_signatures.get(kind):
                mismatches.append({"id": record["id"], "diff": [f"{a!r} != {b!r}" for a, b in
                                                                 zip(signature, test_signatures.get(kind, []))
                                                                 if a != b][:5]})
        verification["training_prompts_checked"] = len(records)
        verification["training_prompts_structure_mismatch"] = len(mismatches)
        verification["structure_identical"] = not mismatches
        verification["first_structure_mismatch"] = mismatches[0] if mismatches else None
        verification["test_signature_with_articles"] = test_signatures.get("articles")
        print("verify: " + json.dumps({k: v for k, v in verification.items() if k != "test_signature_with_articles"}),
              flush=True)

    manifest = {"name": args.out_dir.name, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - started, 1),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "inputs": {"questions": {"path": str(args.questions), "sha256": sha256_file(args.questions)},
                           "pool": {"path": str(args.pool), "sha256": sha256_file(args.pool)},
                           "leak_flags": ({"path": str(args.exclude_flagged), "sha256": sha256_file(args.exclude_flagged),
                                           "counts": dict(flag_counts)} if args.exclude_flagged else None)},
                "counts": {"records": len(records), **{k: stats[k] for k in sorted(stats)}},
                "articles_histogram": {k[len("articles_"):]: stats[k] for k in sorted(stats, key=lambda s: (len(s), s)) if k.startswith("articles_")},
                "outputs": outputs, "verification": verification,
                "format": "training/turtel_prompt.py render_prompt; background empty; description as criteria"}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"counts": manifest["counts"], "outputs": {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                                                                  for k, v in outputs.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
