"""Consistency distillation, step 1: teacher targets and student prompts (CPU, no API calls).

Reads the Qwen3-32B v7b_explicit answers (one form per call, DashScope, temperature 0) for the
train, validation and test pairs of jev-disjoint-v1-filtered-20260919, projects each pair onto
coherent answers (scripts/coherence_projection.project, kl), and writes one row per (pair, form):

  bundle/<split>.jsonl  pair_id query_id form observed label p_id q_id
                        t_raw   teacher answer clipped to [0.01, 0.99] (None if the teacher gave none)
                        t_proj  the kl-projected teacher answer (clipped to [0.01, 0.99])
                        prompt  the student prompt: the v7b_explicit messages with the answer line
                                swapped for a YES/NO line, rendered with the Qwen3 chat template
                                (enable_thinking=False)
  bundle/manifest.json  counts, missing answers, prompt token lengths, sha256 of every file

The teacher prompt of every row is re-rendered and checked against the logged prompt_sha256.

Usage: distill_data.py --tokenizer /path/to/Qwen3-0.6B   (run with torch-free transformers is fine)
       distill_data.py --tokenizer ... --teacher 235b   the Qwen3-235B-A22B answers (v2 run) instead:
           bundle-235b/ and teacher235/proj-<split>.jsonl in /home/agent/data/consistency-distill-v2-20260929;
           every student prompt is checked to be identical to the 32B bundle's, so only the targets differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coherence_projection import project  # noqa: E402
import prompt_negation_fix as pnf  # noqa: E402
from training import verbalized_baseline as vb  # noqa: E402

VARIANT = "v7b_explicit"
OUT = Path("/home/agent/data/consistency-distill-20260929")
TEACHER_LOGS = {
    "train": OUT / "teacher/log-train-v7b_explicit.jsonl",
    "validation": Path("/home/agent/data/prompt-negation-fix-20260929/log-validation-v7b_explicit.jsonl"),
    "test": Path("/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl"),
}
STUDENT_ANSWER = "Reply with exactly one word: YES if the statement is true, NO if it is false."
OUT2 = Path("/home/agent/data/consistency-distill-v2-20260929")
TEACHERS = {  # teacher -> (served model, default out dir, bundle dir name, teacher dir name, logs)
    "32b": ("qwen3-32b", OUT, "bundle", "teacher", TEACHER_LOGS),
    "235b": ("qwen3-235b-a22b", OUT2, "bundle-235b", "teacher235",
             {s: OUT2 / f"teacher235/log-{s}-v7b_explicit.jsonl" for s in ("train", "validation", "test")}),
}
LO = 0.01


def clip(p):
    return None if p is None else min(1 - LO, max(LO, float(p)))


def student_messages(msgs: list[dict]) -> list[dict]:
    user = msgs[1]["content"]
    if user.count(vb.ANSWER_DIRECT) != 1:
        raise ValueError("answer line not found exactly once")
    return [msgs[0], {"role": "user", "content": user.replace(vb.ANSWER_DIRECT, STUDENT_ANSWER)}]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--teacher", default="32b", choices=sorted(TEACHERS))
    ap.add_argument("--out", default=None, help="default: the teacher's output directory")
    ap.add_argument("--splits", nargs="*", default=list(TEACHER_LOGS))
    args = ap.parse_args()
    served, out_default, bname, tname, logs = TEACHERS[args.teacher]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out or out_default); (out / bname).mkdir(parents=True, exist_ok=True); (out / tname).mkdir(exist_ok=True)
    manifest = {"variant": VARIANT, "teacher": served, "student_answer_line": STUDENT_ANSWER, "tokenizer": args.tokenizer,
                "splits": {}}
    for split in args.splits:
        log = logs[split]
        teacher = {}
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("served_model", served) != served:
                raise SystemExit(f"{log}: {r['query_id']} was served by {r.get('served_model')}, not {served}")
            teacher[r["query_id"]] = r
        rows = pnf.rows_for(split, VARIANT)
        bad_hash = 0
        for r in rows:
            t = teacher[r["query_id"]]
            if hashlib.sha256(json.dumps(r["_msgs"]).encode()).hexdigest() != t["prompt_sha256"]:
                bad_hash += 1
            r["p"] = t["p"]
        if bad_hash:
            raise SystemExit(f"{split}: {bad_hash} teacher prompts differ from the logged prompt_sha256")
        proj = {r["query_id"]: r["p"] for r in project([dict(r) for r in rows], "kl")}
        lengths = []
        with (out / bname / f"{split}.jsonl").open("w") as f, (out / tname / f"proj-{split}.jsonl").open("w") as g:
            for r in rows:
                prompt = tok.apply_chat_template(student_messages(r["_msgs"]), tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
                lengths.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
                row = {k: r[k] for k in ("pair_id", "query_id", "form", "observed", "label", "p_id", "q_id")}
                row.update(t_raw=clip(r["p"]), t_proj=clip(proj[r["query_id"]]), prompt=prompt)
                f.write(json.dumps(row) + "\n")
                g.write(json.dumps({k: r[k] for k in ("pair_id", "query_id", "form", "observed", "label")}
                                   | {"p_raw": r["p"], "p": proj[r["query_id"]]}) + "\n")
        lengths.sort()
        q = lambda x: lengths[min(len(lengths) - 1, int(x * len(lengths)))]
        manifest["splits"][split] = {
            "teacher_log": str(log), "teacher_log_sha256": sha256(log),
            "pairs": len({r["pair_id"] for r in rows}), "rows": len(rows),
            "teacher_missing": sum(r["p"] is None for r in rows),
            "teacher_cost_usd": sum(t.get("cost", 0) for t in teacher.values()),
            "prompt_tokens": {"median": q(.5), "p95": q(.95), "max": lengths[-1], "mean": sum(lengths) / len(lengths),
                              "total": sum(lengths)},
            "bundle_sha256": sha256(out / bname / f"{split}.jsonl"),
        }
        if args.teacher != "32b":  # the students must see exactly the prompts of the 32B bundle
            ref = {json.loads(l)["query_id"]: json.loads(l)["prompt"] for l in (OUT / "bundle" / f"{split}.jsonl").open()}
            mine = {json.loads(l)["query_id"]: json.loads(l)["prompt"] for l in (out / bname / f"{split}.jsonl").open()}
            if ref != mine:
                raise SystemExit(f"{split}: student prompts differ from the 32B bundle")
            manifest["splits"][split]["prompts_identical_to_32b_bundle"] = True
        print(split, json.dumps(manifest["splits"][split]), flush=True)
    mpath = out / bname / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {"splits": {}}
    manifest["splits"] = old["splits"] | manifest["splits"]
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
