#!/usr/bin/env python3
"""Prequential study, step 3: code lengths, savings, learning speed and the split-protocol comparison (CPU).

Reads the stream (scripts/preq_data.py) and the per-row predictions of every finished run
(<gpu-dir>/*/<model>/<arm>-s<seed>/preds.csv + result.json, from scripts/preq_student.py), and reports
for each arm (mean over seeds; Z is deterministic and has one seed):

  bits12      prequential code length of the 12 answers per pair: sum over rows of -log2 p(label),
              p clipped to [0.01, 0.99] (the 12 forms restate two outcomes, so this counts each outcome
              several times; it is the loss the arms were trained on)
  bits12_proj the same after kl projection of each pair's 12 answers onto one joint distribution
  bits_pair   code length of the outcomes themselves: per pair -log2 q(true atom), q = the projected
              joint over (P,Q), (Q,not Q) ... mixed with 4% uniform: 2 bits per pair for a know-nothing coder

and the same for reference forecasters: uniform, a prequential base rate (per form, Laplace, from the
outcomes available at the block time), and the 32B teacher (raw and projected; not a learner, the
ceiling the R and C arms imitate).

Intervals: paired bootstrap over the 36 blocks (10,000 resamples) of block-level differences (seed means).
Learning speed: bits saved vs Z per row in blocks 1-12, 13-24 and 25-35 (block 0 is scored before any
training). Split protocol: the multi-epoch train/test numbers of docs/consistency-distill-20260929.json
next to the prequential numbers on all rows, on test rows only and on the last third of the stream.

Usage: preq_analyze.py [--gpu-dir DIR] [--report docs/prequential-20260929.json] [--chart docs/prequential-20260929-chart.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coherence_projection import FORM_ATOMS, fit  # noqa: E402

DATA = Path("/home/agent/data/prequential-20260929")
ARMS = ["Z", "R", "C", "O", "CO"]
CONTRASTS = [("C", "R"), ("C", "O"), ("R", "O"), ("CO", "O"), ("CO", "C"), ("C", "Z"), ("R", "Z"), ("O", "Z"), ("CO", "Z")]
LO, MIX = 0.01, 0.04
THIRDS = {"early (blocks 1-12)": range(1, 13), "middle (13-24)": range(13, 25), "late (25-35)": range(25, 36)}
ATOM_FORM = {(1, 1): "(P) AND (Q)", (1, 0): "(P) AND (NOT Q)", (0, 1): "(NOT P) AND (Q)", (0, 0): "(NOT P) AND (NOT Q)"}


def load_stream(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line); r.pop("prompt"); rows.append(r)
    assert [r["idx"] for r in rows] == list(range(len(rows)))
    return rows


def _proj(args):
    forms, p = args
    q = fit(forms, np.asarray(p, dtype=float), "kl")
    return [float(np.dot(FORM_ATOMS[f], q)) for f in forms], q


class Scorer:
    def __init__(self, rows: list[dict], pool: Pool):
        self.rows, self.pool = rows, pool
        self.y = np.array([r["label"] for r in rows], dtype=float)
        self.block = np.array([r["block"] for r in rows])
        self.nb = int(self.block.max()) + 1
        self.pairs = defaultdict(list)
        for r in rows:
            self.pairs[r["pair_id"]].append(r["idx"])
        self.pair_ids = list(self.pairs)
        self.pair_block = np.array([rows[self.pairs[p][0]]["block"] for p in self.pair_ids])
        self.pair_split = [rows[self.pairs[p][0]]["split"] for p in self.pair_ids]
        self.true_atom = []
        for p in self.pair_ids:
            lab = {rows[i]["form"]: rows[i]["label"] for i in self.pairs[p]}
            self.true_atom.append(("P", "Q", (lab["P"], lab["Q"])))
            assert lab["P"] + lab["NOT P"] == 1 and lab["Q"] + lab["NOT Q"] == 1

    def bits_rows(self, p: np.ndarray) -> np.ndarray:
        q = np.clip(p, LO, 1 - LO)
        return -(self.y * np.log2(q) + (1 - self.y) * np.log2(1 - q))

    def score(self, p: np.ndarray) -> dict:
        jobs = [([self.rows[i]["form"] for i in self.pairs[pid]], [p[i] for i in self.pairs[pid]]) for pid in self.pair_ids]
        res = self.pool.map(_proj, jobs, chunksize=64)
        pp = np.empty_like(p)
        pair_bits = np.empty(len(self.pair_ids))
        for j, (pid, (proj, q)) in enumerate(zip(self.pair_ids, res)):
            for i, v in zip(self.pairs[pid], proj):
                pp[i] = v
            a = self.true_atom[j][2]
            qa = float(np.dot(FORM_ATOMS[ATOM_FORM[a]], q))
            pair_bits[j] = -math.log2((1 - MIX) * qa + MIX / 4)
        r, rp = self.bits_rows(p), self.bits_rows(pp)
        per_block = lambda v, b: np.bincount(b, weights=v, minlength=self.nb)
        test_rows = np.array([x["split"] == "test" for x in self.rows])
        test_pairs = np.array([s == "test" for s in self.pair_split])
        return {"bits12": per_block(r, self.block), "bits12_proj": per_block(rp, self.block),
                "bits_pair": per_block(pair_bits, self.pair_block),
                "test": {"bits12": float(r[test_rows].sum()), "bits12_proj": float(rp[test_rows].sum()),
                         "bits_pair": float(pair_bits[test_pairs].sum()), "rows": int(test_rows.sum()), "pairs": int(test_pairs.sum())}}


def base_rate(rows: list[dict], nb: int) -> np.ndarray:
    """Prequential per-form base rate: Laplace estimate from outcomes available at T_k of rows of earlier blocks."""
    T = [min(r["as_of"] for r in rows if r["block"] == b) for b in range(nb)]
    p = np.empty(len(rows))
    by_form = defaultdict(list)
    for r in rows:
        by_form[r["form"]].append(r)
    for form, rs in by_form.items():
        for k in range(nb):
            seen = [x["label"] for x in rs if x["block"] < k and x["avail_outcome"] <= T[k]]
            rate = (sum(seen) + 1) / (len(seen) + 2)
            for r in rs:
                if r["block"] == k:
                    p[r["idx"]] = rate
    return p


def boot_sum(d: np.ndarray, n: int = 10000, seed: int = 20260929) -> list[float]:
    """95% interval of sum(d) under resampling blocks with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s = d[idx].sum(axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=str(DATA / "bundle/stream.jsonl.gz"))
    ap.add_argument("--gpu-dir", default=str(DATA / "gpu"))
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--distill", default=str(Path(__file__).resolve().parent.parent / "docs/consistency-distill-20260929.json"))
    ap.add_argument("--report", default=str(Path(__file__).resolve().parent.parent / "docs/prequential-20260929.json"))
    ap.add_argument("--chart", default=str(Path(__file__).resolve().parent.parent / "docs/prequential-20260929-chart.json"))
    ap.add_argument("--distill-z", default="/home/agent/data/consistency-distill-20260929/gpu/cdistill-a/Qwen3-0.6B/Z/predictions-test.jsonl")
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()

    rows = load_stream(Path(args.stream))
    N = len(rows)
    runs = {}
    for res in sorted(Path(args.gpu_dir).glob(f"*/{args.model}/*-s*/result.json")):
        arm, seed = res.parent.name.rsplit("-s", 1)
        r = json.loads(res.read_text())
        p = np.full(N, np.nan)
        for line in (res.parent / "preds.csv").read_text().splitlines()[1:]:
            i, v = line.split(","); p[int(i)] = float(v)
        assert not np.isnan(p).any(), res
        runs[(arm, int(seed))] = {"p": p, "result": r, "path": str(res.parent)}
    print("runs:", sorted(runs), file=sys.stderr)

    with Pool(args.procs) as pool:
        S = Scorer(rows, pool)
        nb = S.nb
        scored = {k: S.score(v["p"]) for k, v in runs.items()}
        refs = {"uniform": np.full(N, 0.5), "base rate (prequential)": base_rate(rows, nb),
                "teacher raw (32B)": np.array([r["t_raw"] for r in rows]),
                "teacher projected (32B)": np.array([r["t_proj"] for r in rows])}
        ref_scored = {k: S.score(v) for k, v in refs.items()}

    metrics = ("bits12", "bits12_proj", "bits_pair")
    rows_per_block = np.bincount(S.block, minlength=nb)
    pairs_per_block = np.bincount(S.pair_block, minlength=nb)
    unit = {"bits12": rows_per_block, "bits12_proj": rows_per_block, "bits_pair": pairs_per_block}
    seeds = defaultdict(list)
    for (arm, seed) in scored:
        seeds[arm].append(seed)
    arms = [a for a in ARMS if a in seeds]
    # arm block curves: mean over seeds
    curve = {a: {m: np.mean([scored[(a, s)][m] for s in seeds[a]], axis=0) for m in metrics} for a in arms}
    report = {"stream": {"rows": N, "pairs": len(S.pair_ids), "blocks": nb, "runs": {f"{a}:s{s}": runs[(a, s)]["path"] for a, s in sorted(runs)}},
              "notes": {"bits12": "sum over rows of -log2 p(label), p clipped to [0.01, 0.99]",
                        "bits12_proj": "same after kl projection of the pair's 12 answers",
                        "bits_pair": f"per pair -log2((1-{MIX}) q(true atom) + {MIX}/4), q from the kl projection; uniform = 2 bits/pair",
                        "ci95": "paired bootstrap over blocks (10,000 resamples) of block-level differences of seed-mean code lengths"},
              "arms": {}, "references": {}, "contrasts": {}, "speed": {}, "split_protocol": {}}
    for a in arms:
        e = {"seeds": sorted(seeds[a])}
        for m in metrics:
            tot = [float(scored[(a, s)][m].sum()) for s in seeds[a]]
            e[m] = {"total": float(np.mean(tot)), "per_unit": float(np.mean(tot) / unit[m].sum()), "by_seed": tot,
                    "test_rows_total": float(np.mean([scored[(a, s)]["test"][m] for s in seeds[a]]))}
            if a != "Z" and "Z" in curve:
                d = curve["Z"][m] - curve[a][m]
                e[m]["saved_vs_Z"] = float(d.sum()); e[m]["saved_vs_Z_ci95"] = boot_sum(d)
                e[m]["saved_vs_Z_by_seed"] = [float((curve["Z"][m] - scored[(a, s)][m]).sum()) for s in seeds[a]]
        report["arms"][a] = e
    for name, sc in ref_scored.items():
        report["references"][name] = {m: {"total": float(sc[m].sum()), "per_unit": float(sc[m].sum() / unit[m].sum()),
                                          "test_rows_total": sc["test"][m]} for m in metrics}
    for a, b in CONTRASTS:
        if a in curve and b in curve:
            report["contrasts"][f"{a}-{b}"] = {m: {"diff_bits": float((curve[a][m] - curve[b][m]).sum()),
                                                   "ci95": boot_sum(curve[a][m] - curve[b][m]),
                                                   "diff_per_unit": float((curve[a][m] - curve[b][m]).sum() / unit[m].sum())}
                                               for m in metrics}
            common = sorted(set(seeds[a]) & set(seeds[b]))
            if b != "Z" and a != "Z" and common:  # same-seed differences of totals (seed = LoRA init and dropout)
                for m in metrics:
                    report["contrasts"][f"{a}-{b}"][m]["diff_bits_by_seed"] = {
                        str(s): float((scored[(a, s)][m] - scored[(b, s)][m]).sum()) for s in common}
    # arms against the reference forecasters (block-bootstrap intervals as above)
    for a in arms:
        for rn in ("base rate (prequential)", "teacher projected (32B)"):
            report["contrasts"][f"{a}-{rn}"] = {m: {"diff_bits": float((curve[a][m] - ref_scored[rn][m]).sum()),
                                                    "ci95": boot_sum(curve[a][m] - ref_scored[rn][m]),
                                                    "diff_per_unit": float((curve[a][m] - ref_scored[rn][m]).sum() / unit[m].sum()),
                                                    # block 0 is the zero-shot model for every arm; without it:
                                                    "diff_bits_blocks_1_on": float((curve[a][m] - ref_scored[rn][m])[1:].sum()),
                                                    "ci95_blocks_1_on": boot_sum((curve[a][m] - ref_scored[rn][m])[1:])}
                                                for m in metrics}
    # learning speed: saved vs Z per row in thirds, and C-R / C-O / R-O per row in thirds
    for a in arms:
        if a == "Z" or "Z" not in curve:
            continue
        report["speed"][a] = {}
        for tn, rg in THIRDS.items():
            rg = [k for k in rg if k < nb]
            d = (curve["Z"]["bits12"] - curve[a]["bits12"])[rg]
            u = rows_per_block[rg].sum()
            ci = boot_sum(d)
            report["speed"][a][tn] = {"saved_per_row": float(d.sum() / u), "ci95_per_row": [c / u for c in ci],
                                      "saved_per_pair_bits_pair": float((curve["Z"]["bits_pair"] - curve[a]["bits_pair"])[rg].sum() / pairs_per_block[rg].sum()),
                                      "bits_per_row": float(curve[a]["bits12"][rg].sum() / u)}
        cum = np.cumsum(curve["Z"]["bits12"] - curve[a]["bits12"])
        half = next((k for k in range(nb) if cum[k] >= 0.5 * cum[-1]), None) if cum[-1] > 0 else None
        report["speed"][a]["block_reaching_half_of_final_savings"] = half
    # the thirds differ in difficulty: bits per row of Z and the reference forecasters in each third
    for name, bl in [("Z", curve["Z"]["bits12"] if "Z" in curve else None)] + [(n, sc["bits12"]) for n, sc in ref_scored.items()]:
        if bl is not None:
            report["speed"][f"bits_per_row {name}"] = {tn: float(bl[[k for k in rg if k < nb]].sum() / rows_per_block[[k for k in rg if k < nb]].sum())
                                                       for tn, rg in THIRDS.items()}
    for a, b in [("C", "R"), ("C", "O"), ("R", "O"), ("CO", "O")]:
        if a in curve and b in curve:
            report["speed"][f"{a}-{b}"] = {}
            for tn, rg in THIRDS.items():
                rg = [k for k in rg if k < nb]
                d = (curve[a]["bits12"] - curve[b]["bits12"])[rg]; u = rows_per_block[rg].sum()
                report["speed"][f"{a}-{b}"][tn] = {"diff_per_row": float(d.sum() / u), "ci95_per_row": [c / u for c in boot_sum(d)]}
    # split protocol comparison
    dist = json.loads(Path(args.distill).read_text())
    ds = dist["students"].get(args.model, {})
    late = [k for k in THIRDS["late (25-35)"] if k < nb]
    comp = {}
    n_test_rows = next(iter(ref_scored.values()))["test"]["rows"]
    for a in arms:
        c = {"prequential_all_rows_bits_per_row": report["arms"][a]["bits12"]["per_unit"],
             "prequential_test_rows_bits_per_row": report["arms"][a]["bits12"]["test_rows_total"] / n_test_rows,
             "prequential_late_third_bits_per_row": float(curve[a]["bits12"][late].sum() / rows_per_block[late].sum()),
             "prequential_all_rows_proj_bits_per_row": report["arms"][a]["bits12_proj"]["per_unit"]}
        if a in ds:
            c["split_test_bits_per_row"] = ds[a]["log_loss"]["all12"] / math.log(2)
            c["split_test_proj_bits_per_row"] = ds[a]["log_loss_projected"]["all12"] / math.log(2)
        comp[a] = c
    keys = ["split_test_bits_per_row", "prequential_all_rows_bits_per_row", "prequential_test_rows_bits_per_row",
            "prequential_late_third_bits_per_row", "split_test_proj_bits_per_row", "prequential_all_rows_proj_bits_per_row"]
    ranking = {k: sorted([a for a in comp if k in comp[a]], key=lambda a: comp[a][k]) for k in keys}
    savings = {}
    if "Z" in comp:
        for a in comp:
            if a == "Z":
                continue
            savings[a] = {k: comp["Z"][k] - comp[a][k] for k in keys if k in comp[a] and k in comp["Z"]}
    report["split_protocol"] = {"per_arm": comp, "ranking_best_first": ranking, "saved_vs_Z_per_row": savings,
                                "split_protocol_source": args.distill,
                                "note": "split = LoRA on the train split (0.6B: 2 epochs), early stopping on validation, scored on the 1,066 test pairs; "
                                        "prequential test rows = the same test pairs scored inside the stream (model trained on everything earlier, "
                                        "including earlier test pairs' teacher answers for R/C and resolved outcomes for O/CO)"}
    # chart
    chart = {"x_block": list(range(nb)), "block_T": [min(r["as_of"] for r in rows if r["block"] == b) for b in range(nb)],
             "rows_per_block": rows_per_block.tolist(), "arms": {}, "references": {}}
    for a in arms:
        chart["arms"][a] = {"seeds": sorted(seeds[a]),
                            "bits_per_row_by_block": (curve[a]["bits12"] / rows_per_block).tolist(),
                            "cum_bits": np.cumsum(curve[a]["bits12"]).tolist(),
                            "cum_bits_pair": np.cumsum(curve[a]["bits_pair"]).tolist()}
        if "Z" in curve and a != "Z":
            chart["arms"][a]["cum_saved_vs_Z"] = np.cumsum(curve["Z"]["bits12"] - curve[a]["bits12"]).tolist()
            per_seed = [np.cumsum(curve["Z"]["bits12"] - scored[(a, s)]["bits12"]) for s in seeds[a]]
            chart["arms"][a]["cum_saved_vs_Z_seed_min"] = np.min(per_seed, axis=0).tolist()
            chart["arms"][a]["cum_saved_vs_Z_seed_max"] = np.max(per_seed, axis=0).tolist()
    for name, sc in ref_scored.items():
        chart["references"][name] = {"bits_per_row_by_block": (sc["bits12"] / rows_per_block).tolist(),
                                     "cum_bits": np.cumsum(sc["bits12"]).tolist(), "cum_bits_pair": np.cumsum(sc["bits_pair"]).tolist()}
        if "Z" in curve:
            chart["references"][name]["cum_saved_vs_Z"] = np.cumsum(curve["Z"]["bits12"] - sc["bits12"]).tolist()
    # sanity: block 0 is scored before any training, so every arm equals Z there (up to bf16 noise)
    if "Z" in curve:
        report["sanity_block0_bits_vs_Z"] = {a: float(curve[a]["bits12"][0] - curve["Z"]["bits12"][0]) for a in arms}
    # sanity: packed Z on the test rows vs the distillation run's ordinary Z predictions (same prompts, same weights)
    dz = Path(args.distill_z)
    if ("Z", 1) in runs and dz.exists():
        key = {r["query_id"]: r["idx"] for r in rows}
        zp = runs[("Z", 1)]["p"]
        a, b = [], []
        for line in dz.read_text().splitlines():
            r = json.loads(line)
            if r["query_id"] in key and r["p"] is not None:
                a.append(r["p"]); b.append(zp[key[r["query_id"]]])
        a, b = np.array(a), np.array(b)
        lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
        report["sanity_Z_vs_distill_Z_test"] = {"rows": len(a), "max_abs_prob_diff": float(np.abs(a - b).max()),
                                                "mean_abs_prob_diff": float(np.abs(a - b).mean()),
                                                "mean_abs_logit_diff": float(np.abs(lg(a) - lg(b)).mean()),
                                                "corr": float(np.corrcoef(a, b)[0, 1])}
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.chart).write_text(json.dumps(chart) + "\n")
    brief = {a: {m: round(report["arms"][a][m]["total"], 1) for m in metrics} for a in arms}
    print(json.dumps({"totals": brief, "contrasts": {k: {m: [round(v[m]["diff_bits"], 1)] + [round(x, 1) for x in v[m]["ci95"]] for m in metrics}
                                                     for k, v in report["contrasts"].items()},
                      "ranking": ranking}, indent=1))


if __name__ == "__main__":
    main()
