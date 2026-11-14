"""Consistency distillation, step 4: score the students on the 1,066 test pairs.

For every student (model x arm: Z zero-shot, R teacher raw, C teacher projected, O outcomes,
CO projected+outcomes), the teacher (Qwen3-32B v7b_explicit, raw and kl-projected) and the base-rate
table (per-form YES rate of the train labels), report on the test split:

  log loss (clip 0.01) over all 12 forms, the 4 singles (literals) and the 8 joints (compounds),
  Brier, ECE (10 bins), accuracy, coherence (corr of p(A) with p(not A); mean relation_penalties total),
  and the same log loss after kl projection of the student's own 12 answers.

Paired bootstrap 95% CIs over pairs (2,000 resamples): C-R (key), C-O, CO-O, C-Z per student size,
plus C, CO and O against the teacher's raw answers and the base-rate table.

Usage: distill_analyze.py [--gpu-dir DIR] [--report docs/consistency-distill-20260929.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coherence_projection import project  # noqa: E402
from prompt_negation_fix import by_pair, coherence, events, ll_summary, ll_terms  # noqa: E402
from training.supervised_pairs import FORM_ORDER, probability_metrics, relation_penalties  # noqa: E402

OUT = Path("/home/agent/data/consistency-distill-20260929")
TEACHER_TEST = Path("/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl")
ARMS = ["Z", "R", "C", "O", "CO"]
CONTRASTS = [("C", "R"), ("C", "O"), ("CO", "O"), ("C", "Z")]
LO = 0.01


def load(path: Path, key: str = "p") -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.append({"pair_id": r["pair_id"], "form": r["form"], "label": r["label"], "observed": r["observed"], "p": r[key]})
    return rows


def arm_stats(rows: list[dict]) -> dict:
    bp = by_pair(rows)
    pairs = sorted(bp)
    assert all(len(bp[p]) == 12 for p in pairs)
    proj = by_pair(project([dict(r, query_id=None) for r in rows], "kl"))
    return {"bp": bp, "pairs": pairs, "raw": ll_terms(bp, pairs), "proj": ll_terms(proj, pairs), "ev": events(bp, pairs)}


def penalties(bp, pairs) -> float:
    g = lambda r: 0.5 if r["p"] is None else r["p"]
    return float(np.mean([relation_penalties({f: g(bp[p][f]) for f in FORM_ORDER})["total"] for p in pairs]))


def summarize(a: dict, pairs: list[str]) -> dict:
    ps = [0.5 if a["bp"][p][f]["p"] is None else min(1 - LO, max(LO, a["bp"][p][f]["p"])) for p in pairs for f in FORM_ORDER]
    ys = [a["bp"][p][f]["label"] for p in pairs for f in FORM_ORDER]
    m = probability_metrics(ps, ys)
    return {"log_loss": ll_summary(a["raw"], pairs), "log_loss_projected": ll_summary(a["proj"], pairs),
            "brier": m["brier"], "ece": m["ece"], "accuracy": m["accuracy"],
            "coherence": coherence(a["ev"], pairs), "relation_penalty": penalties(a["bp"], pairs)}


LIT = [FORM_ORDER.index(f) for f in ("P", "NOT P", "Q", "NOT Q")]
CMP = [i for i in range(12) if i not in LIT]


def mats(a: dict, pairs: list[str]):
    raw = np.array([[a["raw"][p][f] for f in FORM_ORDER] for p in pairs])
    proj = np.array([[a["proj"][p][f] for f in FORM_ORDER] for p in pairs])
    ev = np.array([a["ev"][p] for p in pairs])  # pairs x 2 events x (p(A), p(not A))
    return raw, proj, ev


def contrast(A: dict, B: dict, pairs: list[str], n: int, seed: int = 20260929) -> dict:
    """Paired bootstrap over pairs (same resampled pairs for A and B): A minus B, point and 95% CI."""
    (ra, pa, ea), (rb, pb, eb) = mats(A, pairs), mats(B, pairs)
    rng = np.random.default_rng(seed)
    idx = np.vstack([np.arange(len(pairs))] + [rng.integers(0, len(pairs), len(pairs)) for _ in range(n)])
    def corr(e, ix):
        x = e[ix][:, :, :, 0].reshape(len(ix), -1); y = e[ix][:, :, :, 1].reshape(len(ix), -1)
        x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
        return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))
    out = {}
    for kind, ma, mb in (("raw", ra, rb), ("proj", pa, pb)):
        d = ma - mb
        for grp, cols in (("all12", list(range(12))), ("literals", LIT), ("compounds", CMP)):
            per_pair = d[:, cols].mean(1)
            v = per_pair[idx].mean(1)
            out[f"ll_{kind}_{grp}"] = {"diff": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
    v = corr(ea, idx) - corr(eb, idx)
    out["corr"] = {"diff": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-dir", default=str(OUT / "gpu"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parent.parent / "docs/consistency-distill-20260929.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    refs = {"teacher_raw": arm_stats(load(TEACHER_TEST)), "teacher_projected": None}
    t = refs["teacher_raw"]
    refs["teacher_projected"] = arm_stats(project([dict(r, query_id=None) for r in load(TEACHER_TEST)], "kl"))
    train = [json.loads(l) for l in (OUT / "bundle/train.jsonl").read_text().splitlines()]
    rate = defaultdict(list)
    for r in train:
        rate[r["form"]].append(r["label"])
    rate = {f: float(np.mean(v)) for f, v in rate.items()}
    refs["base_rate"] = arm_stats([dict(r, p=rate[r["form"]]) for r in load(TEACHER_TEST)])
    pairs = t["pairs"]

    students = defaultdict(dict)  # model -> arm -> stats
    sources = {}
    for f in sorted(Path(args.gpu_dir).glob("*/**/predictions-test.jsonl")):
        arm, model = f.parent.name, f.parent.parent.name
        if arm not in ARMS or not model.startswith("Qwen3-"):
            continue
        if arm in students[model]:
            raise SystemExit(f"duplicate {model}/{arm}: {sources[model, arm]} and {f}")
        students[model][arm] = arm_stats(load(f))
        res = f.parent / "result.json"
        sources[model, arm] = {"predictions": str(f), "result": json.loads(res.read_text()) if res.exists() else None}
        assert students[model][arm]["pairs"] == pairs, (model, arm)

    report = {"test_pairs": len(pairs), "references": {k: summarize(v, pairs) for k, v in refs.items()},
              "base_rate_table": rate, "students": {}, "contrasts": {}}
    for model in sorted(students, key=lambda m: float(m.split("-")[1].rstrip("B"))):
        report["students"][model] = {}
        for arm in ARMS:
            if arm in students[model]:
                s = summarize(students[model][arm], pairs)
                res = sources[model, arm]["result"] or {}
                s["best"] = res.get("best"); s["train_rows"] = res.get("train_rows"); s["steps_run"] = res.get("steps_run")
                s["seconds"] = res.get("seconds")
                report["students"][model][arm] = s
        cmp = {}
        arms = students[model]
        pairs_list = [(a, b) for a, b in CONTRASTS if a in arms and b in arms]
        pairs_list += [(a, ref) for a in ("C", "CO", "O") if a in arms for ref in ("teacher_raw", "base_rate")]
        for a, b in pairs_list:
            A = arms[a]; B = arms[b] if b in arms else refs[b]
            cmp[f"{a}-{b}"] = contrast(A, B, pairs, args.boot)
        report["contrasts"][model] = cmp

    chart = {"title": "Consistency distillation: test log loss (all 12 forms, 1,066 pairs)",
             "references": {k: {"raw": v["log_loss"]["all12"], "projected": v["log_loss_projected"]["all12"]}
                            for k, v in report["references"].items()},
             "students": {m: {a: {"raw": s["log_loss"]["all12"], "projected": s["log_loss_projected"]["all12"]}
                              for a, s in d.items()} for m, d in report["students"].items()},
             "ci_all12_raw": {m: {k: v["ll_raw_all12"] for k, v in c.items()} for m, c in report["contrasts"].items()}}
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.report).with_name(Path(args.report).stem + "-chart.json").write_text(json.dumps(chart, indent=1) + "\n")

    def line(name, s):
        l, p = s["log_loss"], s["log_loss_projected"]
        return (f"{name:22s} LL {l['all12']:.4f} (sing {l['literals']:.4f} joint {l['compounds']:.4f}) proj {p['all12']:.4f} "
                f"Brier {s['brier']:.4f} ECE {s['ece']:.4f} acc {s['accuracy']:.3f} corr {s['coherence']['corr']:+.3f} "
                f"pen {s['relation_penalty']:.4f}")
    for k, s in report["references"].items():
        print(line(k, s))
    for m, d in report["students"].items():
        for a, s in d.items():
            print(line(f"{m} {a}", s))
        for k, c in report["contrasts"][m].items():
            print(f"  {m} {k}: " + "; ".join(f"{n} {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
                                            for n, v in c.items() if n in ("ll_raw_all12", "ll_proj_all12", "ll_raw_literals", "corr")))


if __name__ == "__main__":
    main()
