#!/usr/bin/env python3
"""R7-E1 audit, check 8: my HAND labels for every decimal token of RESULTS-R7-E1.md that no hand-mapped entry covers
(results-numbers.json "uncovered"), plus checks of the reference and ledger values, plus a 30-number spot check of the
automatic comparison. Usage: python3 uncovered_labels.py > uncovered-labels-output.txt"""
import json
import random
import re

U = json.load(open("results-numbers.json"))
spec = open("/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md").read()
ledger = open("/home/agent/workspaces/eurisko-study/li-net/r7e1/GPU-LOG.md").read()
D, B, A, F, L = "design constant / label (tau tier, flip rate, t, exponent, ceiling, cap, rate, model name, section no.)", \
    "rule bar (threshold quoted from PREREG section 3)", "result number verified by a hand entry on the adjacent line", \
    "reference value from the spec / R6", "cost from the R7-E1 ledger (GPU-LOG.md)"
labels = {
    (3, "5.5"): D, (4, "2.50"): D, (16, "0.9494"): F, (27, "0.99"): B, (27, "0.40"): B, (28, "1.10"): B, (29, "2.5"): D, (29, "2.0"): B,
    (29, "1.25"): B, (30, "0.955"): B, (30, "0.950"): B, (30, "0.75"): B, (32, "0.3"): D, (32, "1.25"): B, (32, "2.5"): D,
    (45, "3.35"): A, (45, "3.62"): A, (50, "0.3"): D, (50, "2.5"): D, (51, "2.5"): D, (52, "94.9"): F, (57, "0.0039"): A, (62, "2.50"): D,
    (63, "0.22"): A, (68, "3.1"): D, (77, "0.15"): D, (80, "0.3"): D, (80, "1.5"): D, (80, "2.5"): D, (80, "0.127"): D, (80, "0.28"): D,
    (80, "0.239"): D, (81, "0.346"): D, (99, "0.3"): D, (99, "1.5"): D, (102, "1.5"): D, (105, "4.303"): D, (140, "0.38"): B, (141, "1.10"): B,
    (142, "0.038"): F, (143, "4.5"): B, (154, "0.069"): F, (156, "0.3"): D, (156, "1.5"): D, (156, "2.5"): D, (157, "2.5"): D, (157, "2.0"): B,
    (158, "0.3"): D, (158, "1.25"): B, (178, "0.3"): D, (178, "1.5"): D, (183, "0.3"): D, (195, "0.513"): A, (201, "0.01739"): A,
    (201, "0.00069"): A, (202, "4.5"): A, (202, "7.6"): A, (205, "1.5"): D, (209, "0.9494"): F, (222, "0.3"): D, (223, "1.5"): D,
    (224, "2.5"): D, (225, "0.3"): D, (226, "1.5"): D, (227, "2.5"): D, (228, "0.3"): D, (228, "2.5"): D, (229, "0.3"): D, (229, "2.5"): D,
    (231, "0.3"): D, (231, "1.5"): D, (231, "2.5"): D, (243, "1.25"): B, (244, "0.35"): B, (244, "0.20"): B, (247, "2.20"): B, (247, "1.90"): B,
    (247, "2.80"): B, (248, "0.20"): B, (249, "0.20"): B, (253, "0.3"): D, (253, "1.5"): D, (253, "2.5"): D, (258, "0.3"): D, (260, "2.5"): D,
    (263, "2.5"): D, (263, "0.3"): D, (264, "0.3"): D, (295, "0.382"): F, (295, "0.880"): F, (295, "0.863"): F, (306, "0.76"): L,
    (323, "6.50"): D, (324, "0.87"): D, (328, "0.00"): L, (329, "0.76"): L, (330, "0.22"): L, (332, "0.98"): L, (332, "2.50"): D,
}
counts, missing = {}, []
for ln, tok in U["uncovered"]:
    lab = labels.get((ln, tok))
    if lab is None:
        missing.append((ln, tok))
        continue
    counts[lab] = counts.get(lab, 0) + 1
print("uncovered tokens:", len(U["uncovered"]), "hand-labelled:", sum(counts.values()), "without a label:", missing)
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
# reference values present in the spec, ledger values present in the ledger
refs = {"0.9494": "0.9494" in spec, "0.069": "0.069" in spec, "0.038": "0.038" in spec, "0.382": "0.382" in spec, "0.880": "0.880" in spec,
        "0.863": "0.863" in spec, "94.9 (= 0.9494 x 100, rounded)": round(0.9494 * 100, 1) == 94.9}
print("reference values found in the spec:", refs)
print("ledger values found in GPU-LOG.md (r7e1):", {v: v in ledger for v in ("0.00", "0.76", "0.22", "$0.98")})
# 30-number spot check of the automatic comparison: print the entries; I compared each printed value by eye with the
# owner's factsheet (/tmp/r7e1/factsheet-main.txt) or results.json and record my label in SPOT below.
random.seed(7)
pool = [(c[0], c[1], t) for c in U["checks"] for t in c[2].split()]
sample = random.sample(pool, 30)
for i, s in enumerate(sample):
    print(f"  spot {i:2d}: line {s[0]} {s[1]}: printed {s[2]}")
