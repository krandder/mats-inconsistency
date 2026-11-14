"""Consistency distillation v2: score the 1.7B students and the 235B-teacher students on the 1,066 test pairs.

Students (test predictions, one test evaluation each):
  Qwen3-0.6B  Z, R, C from the Qwen3-32B teacher (v1 run, cdistill-a); R, C from the Qwen3-235B-A22B teacher (v2)
  Qwen3-1.7B  Z (v1 run); R, C from the 32B teacher and R, C from the 235B teacher (v2, checkpointed spot VMs)
References: each teacher's own v7b_explicit test answers, raw and kl-projected; the train base-rate table.

Metrics as in distill_analyze.py (log loss all 12 / singles / joints, raw and after kl projection of the
student's own 12 answers, Brier, ECE, accuracy, coherence). Paired bootstrap 95% CIs over pairs (2,000
resamples, the same resampled pairs for every term) for:
  C-R per size and teacher (key: does C beat R at 1.7B?), C-Z per size and teacher,
  (C-R at 1.7B) - (C-R at 0.6B) per teacher (does the edge grow with student size?),
  235B-32B for C and for R per size (does a better teacher help?), 1.7B-0.6B per arm and teacher,
  and each student against its own teacher's raw answers.

Usage: distill2_analyze.py [--report docs/consistency-distill-v2-20260929.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coherence_projection import project  # noqa: E402
from distill_analyze import arm_stats, load, mats, summarize  # noqa: E402

V1 = Path("/home/agent/data/consistency-distill-20260929")
V2 = Path("/home/agent/data/consistency-distill-v2-20260929")
TEACHER_TEST = {"32b": Path("/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl"),
                "235b": V2 / "teacher235/log-test-v7b_explicit.jsonl"}
TEACHER_NAME = {"32b": "Qwen3-32B", "235b": "Qwen3-235B-A22B"}
SIZES = ["Qwen3-0.6B", "Qwen3-1.7B"]


def student_files() -> dict:
    """(model, arm, teacher) -> predictions-test.jsonl; teacher None for Z."""
    f = {("Qwen3-0.6B", "Z", None): V1 / "gpu/cdistill-a/Qwen3-0.6B/Z/predictions-test.jsonl",
         ("Qwen3-0.6B", "R", "32b"): V1 / "gpu/cdistill-a/Qwen3-0.6B/R/predictions-test.jsonl",
         ("Qwen3-0.6B", "C", "32b"): V1 / "gpu/cdistill-a/Qwen3-0.6B/C/predictions-test.jsonl",
         ("Qwen3-1.7B", "Z", None): V1 / "gpu/cdistill-a/Qwen3-1.7B/Z/predictions-test.jsonl"}
    for g in sorted((V2 / "gpu").glob("*/current")):
        if g.parent.name == "t":  # the resume test, not a result
            continue
        for p in sorted(g.glob("Qwen3-*/*-t*/predictions-test.jsonl")):
            arm, teacher = p.parent.name.split("-t", 1)
            if "-" in teacher or not (p.parent / "result.json").exists():
                continue  # tagged benchmark runs, or unfinished
            key = (p.parent.parent.name, arm, teacher)
            if key in f:
                raise SystemExit(f"duplicate {key}: {f[key]} and {p}")
            f[key] = p
    return f


def lincomb(terms: list[tuple[float, dict]], pairs: list[str], n: int, seed: int = 20260929) -> dict:
    """Paired bootstrap over pairs of sum_i w_i * LL_i (per pair), for all/singles/joints, raw and projected."""
    from distill_analyze import CMP, LIT
    rng = np.random.default_rng(seed)
    idx = np.vstack([np.arange(len(pairs))] + [rng.integers(0, len(pairs), len(pairs)) for _ in range(n)])
    out = {}
    for kind, k in (("raw", 0), ("proj", 1)):
        d = sum(w * mats(s, pairs)[k] for w, s in terms)
        for grp, cols in (("all12", list(range(12))), ("literals", LIT), ("compounds", CMP)):
            v = d[:, cols].mean(1)[idx].mean(1)
            out[f"ll_{kind}_{grp}"] = {"diff": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)),
                                                                      float(np.percentile(v[1:], 97.5))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(Path(__file__).resolve().parent.parent / "docs/consistency-distill-v2-20260929.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    refs = {}
    for t, path in TEACHER_TEST.items():
        if path.exists():
            rows = load(path)
            refs[f"teacher_{t}_raw"] = arm_stats(rows)
            refs[f"teacher_{t}_projected"] = arm_stats(project([dict(r, query_id=None) for r in rows], "kl"))
    pairs = refs["teacher_32b_raw"]["pairs"]
    rate = defaultdict(list)
    for line in (V1 / "bundle/train.jsonl").read_text().splitlines():
        r = json.loads(line); rate[r["form"]].append(r["label"])
    rate = {f: float(np.mean(v)) for f, v in rate.items()}
    refs["base_rate"] = arm_stats([dict(r, p=rate[r["form"]]) for r in load(TEACHER_TEST["32b"])])

    st, info = {}, {}
    for key, path in student_files().items():
        st[key] = arm_stats(load(path))
        assert st[key]["pairs"] == pairs, key
        res = path.parent / "result.json"
        info[key] = json.loads(res.read_text()) if res.exists() else {}
    name = lambda k: f"{k[0]} {k[1]}" + (f"@{k[2]}" if k[2] else "")

    report = {"test_pairs": len(pairs), "teachers": TEACHER_NAME,
              "references": {k: summarize(v, pairs) for k, v in refs.items()}, "base_rate_table": rate,
              "students": {}, "contrasts": {}}
    for k in sorted(st, key=lambda k: (SIZES.index(k[0]), k[2] or "", k[1])):
        s = summarize(st[k], pairs)
        r = info[k]
        s.update(best=r.get("best"), steps_run=r.get("steps_run"), total_steps=r.get("total_steps"),
                 stopped_early=r.get("stopped_early"), seconds=r.get("seconds"), source=str(student_files()[k]))
        report["students"][name(k)] = s

    C = report["contrasts"]
    def add(label, terms):
        if all(t in st for _, t in terms if isinstance(t, tuple)):
            C[label] = lincomb([(w, st[t] if isinstance(t, tuple) else refs[t]) for w, t in terms], pairs, args.boot)
    for m in SIZES:
        for t in ("32b", "235b"):
            add(f"{m} C-R @{t}", [(1, (m, "C", t)), (-1, (m, "R", t))])
            add(f"{m} C-Z @{t}", [(1, (m, "C", t)), (-1, (m, "Z", None))])
            add(f"{m} R-Z @{t}", [(1, (m, "R", t)), (-1, (m, "Z", None))])
            for a in ("C", "R"):
                if f"teacher_{t}_raw" in refs:
                    add(f"{m} {a}@{t} - teacher {t} raw", [(1, (m, a, t)), (-1, f"teacher_{t}_raw")])
        for a in ("C", "R"):
            add(f"{m} {a}: 235b-32b", [(1, (m, a, "235b")), (-1, (m, a, "32b"))])
    for t in ("32b", "235b"):
        add(f"(C-R) 1.7B minus (C-R) 0.6B @{t}", [(1, ("Qwen3-1.7B", "C", t)), (-1, ("Qwen3-1.7B", "R", t)),
                                                  (-1, ("Qwen3-0.6B", "C", t)), (1, ("Qwen3-0.6B", "R", t))])
        for a in ("C", "R"):
            add(f"{a}@{t}: 1.7B-0.6B", [(1, ("Qwen3-1.7B", a, t)), (-1, ("Qwen3-0.6B", a, t))])
    if "teacher_235b_raw" in refs:
        C["teacher 235b-32b raw"] = lincomb([(1, refs["teacher_235b_raw"]), (-1, refs["teacher_32b_raw"])], pairs, args.boot)
        C["teacher 235b-32b projected"] = lincomb([(1, refs["teacher_235b_projected"]), (-1, refs["teacher_32b_projected"])],
                                                  pairs, args.boot)

    chart = {"title": "Consistency distillation v2: test log loss, all 12 forms (1,066 pairs; lower is better)",
             "references": {k: v["log_loss"]["all12"] for k, v in report["references"].items()},
             "students": {k: {"raw": v["log_loss"]["all12"], "projected": v["log_loss_projected"]["all12"]}
                          for k, v in report["students"].items()},
             "key_contrasts_all12_raw": {k: v["ll_raw_all12"] for k, v in C.items()}}
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.report).with_name(Path(args.report).stem + "-chart.json").write_text(json.dumps(chart, indent=1) + "\n")

    for k, s in list(report["references"].items()) + list(report["students"].items()):
        l, p = s["log_loss"], s["log_loss_projected"]
        print(f"{k:28s} LL {l['all12']:.4f} (sing {l['literals']:.4f} joint {l['compounds']:.4f}) proj {p['all12']:.4f} "
              f"(sing {p['literals']:.4f} joint {p['compounds']:.4f}) acc {s['accuracy']:.3f} ECE {s['ece']:.4f} "
              f"corr {s['coherence']['corr']:+.3f}")
    for k, c in C.items():
        print(f"  {k}: " + "; ".join(f"{n} {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                                     for n, v in c.items() if n in ("ll_raw_all12", "ll_raw_literals", "ll_raw_compounds",
                                                                    "ll_proj_all12")))


if __name__ == "__main__":
    main()
