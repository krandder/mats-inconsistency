#!/usr/bin/env python3
"""R6-E1 audit, check 7: every measured number in RESULTS-R6-E1.md against the files.

Automatic rule (to be validated by a hand-labelled sample):
  - extract decimal numbers, percentages, "Nx" ratios and comma-grouped integers, skipping identifiers (R5-E1, E1-D2.1, K4,
    s101, r6e1, 2x3, 4/3/2, 85/15 ...), times, dates, hashes, tau / N / K labels and the spec-section number;
  - a token MATCHES if some value from the files rounds to it at the token's own number of decimals (percent tokens: 100 x the
    value); DOUBLE-ROUNDING if a value matches only after first rounding to one more decimal (e.g. 0.7547 -> 0.755 -> 0.76);
    otherwise NO MATCH.
  Value pool: the auditor's re-derivation (rederive-output.json: per seed, 3-seed means, CI bounds), analysis/results.json,
  summary.json (fit + test, per seed and 3-seed mean), the ledger, parameter counts and the R5-E1 baselines quoted by the spec.
Writes claims-tokens.csv (one row per token) and prints totals. The hand-labelled sample is drawn with seed 2026 (40 tokens).
Run: PYTHONPATH=/tmp/r4e3a/pylib python3 claims_check.py
"""
import csv
import json
import math
import os
import random
import re

import numpy as np

R6 = "/home/agent/workspaces/eurisko-study/li-net/r6e1"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)


def flatten(o, path="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flatten(v, f"{path}[{i}]", out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(o):
        out[path] = float(o)
    return out


def pool():
    P = {}
    red = json.load(open(os.path.join(OUT, "rederive-output.json")))
    for k, v in flatten({"mine": red["per_seed_mine"], "rules": red["rules"]["t_exact"], "checks": red["checks"]}).items():
        P[k] = v
    for k, v in flatten(json.load(open(os.path.join(R6, "results", "main", "analysis", "results.json")))).items():
        P["results." + k] = v
    S = {s: json.load(open(os.path.join(R6, "results", "main", f"s{s}", "summary.json"))) for s in SEEDS}
    for s in SEEDS:
        for k, v in flatten({"fit": S[s]["fit"], "test": S[s]["test"], "train_sec": S[s]["train_sec"],
                             "total_sec": S[s]["total_sec"], "theta_real_params": S[s]["theta_real_params"]}).items():
            P[f"s{s}.{k}"] = v
    f0 = flatten({"fit": S[101]["fit"], "test": S[101]["test"]})
    for k in f0:
        try:
            vals = [flatten({"fit": S[s]["fit"], "test": S[s]["test"]})[k] for s in SEEDS]
        except KeyError:
            continue
        P[f"mean3.{k}"] = float(np.mean(vals))
    # 3-seed mean of the per-seed ratios that the write-up quotes without a stats entry
    T = {s: S[s]["test"] for s in SEEDS}
    P["mean3.ap_union_jsd_ratio_open_over_puct"] = float(np.mean([T[s]["ap_open_jsd_union"] / T[s]["ap_puct_jsd_union"] for s in SEEDS]))
    P["ratio_of_means.ap_union_jsd"] = float(np.mean([T[s]["ap_open_jsd_union"] for s in SEEDS]) /
                                             np.mean([T[s]["ap_puct_jsd_union"] for s in SEEDS]))
    P["mean3.coverage_gain_points"] = 100 * float(np.mean([T[s]["ap_coverage_puct"] - T[s]["ap_coverage_open"] for s in SEEDS]))
    P["coverage_shortfall_points"] = 100 * (0.95 - float(np.mean([T[s]["ap_coverage_puct"] for s in SEEDS])))
    P["params.looped_over_attnK4_inverse"] = 108228 / 2789
    for k, v in {"ledger.vm1": 0.28, "ledger.vm2": 0.82, "ledger.total": 1.10, "ledger.ceiling": 2.50, "rate": 0.87,
                 "group_cap": 12, "r5e1.key_pearson": 0.3749, "r5e1.key_gap": 0.2177, "n_patterns": 49504,
                 "remote_min_planned": 100, "remote_min_used": 90, "prereg.tau_eff_1.5": 0.239, "prereg.tau_eff_2.5": 0.346,
                 "prereg.reflip_1.5": 0.127, "prereg.reflip_2.5": 0.28, "sonar_flip": 0.15, "t975": 4.303,
                 "spec.bar_cov_open": 0.87, "run_minutes": 52, "preempt_minutes_into_training": 15}.items():
        P[k] = float(v)
    return P


SKIP_PREFIX = re.compile(r"(R\d-E\d|E1-D[0-9.]*|K\d|s\d{3}|r\de\d|attn_L\d|lin_T\d|#\d+|tau[ =]*|N ?[=in{]*|K ?= ?|"
                         r"t ?= ?|W\(|sha256 |\bU\{|section |spec |\$?\d+x\d|\d+/\d+/|\d+/)$")
TOKEN = re.compile(r"(?<![\w./#-])(\$?)(-?\d{1,3}(?:,\d{3})+|-?\d+\.\d+|-?\d+)(%|x)?(?![\w/])")


def tokens(text):
    out = []
    for ln, line in enumerate(text.split("\n"), 1):
        if "@@AUDIT@@" in line or line.startswith("| VM |") or line.startswith("|---"):
            continue
        for m in TOKEN.finditer(line):
            raw = m.group(2)
            pre = line[max(0, m.start() - 12):m.start()]
            post = line[m.end():m.end() + 3]
            if re.search(r"\d{1,2}:\d{2}", line[max(0, m.start() - 3):m.end() + 3]):        # times
                continue
            if re.search(r"20\d\d-\d\d", line[max(0, m.start() - 8):m.end() + 6]):          # dates / VM names
                continue
            if SKIP_PREFIX.search(pre) or post.startswith(("x3", "x2", "x6", "/")) or pre.endswith(("x", "/", "W(", "(1", "-")):
                continue
            if re.fullmatch(r"-?\d+", raw) and "," not in raw and m.group(3) is None:
                continue                                                                      # plain integers: checked by hand
            out.append({"line": ln, "token": m.group(0), "value": float(raw.replace(",", "")), "pct": m.group(3) == "%",
                        "dec": len(raw.split(".")[1]) if "." in raw else 0, "context": line.strip()[:160]})
    return out


def classify(t, P):
    v, d = t["value"], t["dec"]
    exact, double = [], []
    for k, c in P.items():
        for cc, tag in ((c, ""), (100 * c, "x100"), (1 - c, "1-")):
            if t["pct"] and tag != "x100" and not (tag == "" and abs(cc) > 1.5):
                continue
            if abs(round(cc, d) - v) < 1e-9 or (d == 0 and abs(round(cc) - v) < 1e-9):
                exact.append(tag + k)
            elif abs(round(round(cc, d + 1), d) - v) < 1e-9 and abs(round(cc, d) - v) > 1e-9:
                double.append(tag + k)
    return ("MATCH" if exact else "DOUBLE-ROUNDING" if double else "NO MATCH"), exact[:4], double[:4]


def main():
    P = pool()
    text = open(os.path.join(R6, "RESULTS-R6-E1.md")).read()
    toks = tokens(text)
    rows = []
    for i, t in enumerate(toks):
        st, ex, db = classify(t, P)
        rows.append({"id": i, **t, "auto": st, "matches": "; ".join(ex), "double_only": "; ".join(db)})
    with open(os.path.join(OUT, "claims-tokens.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    c = {s: sum(r["auto"] == s for r in rows) for s in ("MATCH", "DOUBLE-ROUNDING", "NO MATCH")}
    print(f"{n} measured-number tokens extracted from RESULTS-R6-E1.md (sha256 of the file below); auto: {c}")
    random.seed(2026)
    sample = sorted(random.sample(range(n), 40))
    print("hand-label sample ids (seed 2026):", sample)
    for r in rows:
        if r["auto"] != "MATCH":
            print(f"  line {r['line']}: {r['token']!r} -> {r['auto']}; double-rounding candidates {r['double_only'][:120]}")


if __name__ == "__main__":
    main()
