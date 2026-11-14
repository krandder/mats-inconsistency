"""Track A2 tables: costly-copy world vs original-world control (no keys, depth 1/2/4, 3 seeds each).

usage: python a2_tables.py CTL_DIR A2_DIR TIED_CTL_JSON > a2_tables.md
"""
import json, os, sys
from collections import defaultdict
import numpy as np

ctl, a2, tied_ctl = sys.argv[1:4]
rng = np.random.default_rng(0)


def load(root):
    g = defaultdict(list)
    for m in json.load(open(os.path.join(root, 'split_indist.json')))['models']:
        g[m['depth']].append(m)
    return g


def stat(m, v, idx=None):
    w0, w1 = (np.array(x) for x in m['w'][v])
    if idx is not None:
        w0, w1 = w0[idx], w1[idx]
    r = w0.mean() / w1.mean()
    return (4 - r) / 3 if v == 'copy3' else 2 - r


def cell(ms, v, B=500):
    n = len(ms[0]['w'][v][0])
    vals = [stat(m, v) for m in ms]
    bs = [np.mean([stat(m, v, i) for m in ms]) for i in (rng.integers(0, n, n) for _ in range(B))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{np.mean(vals):.2f} [{lo:.2f}, {hi:.2f}]"


C, A = load(ctl), load(a2)
print("### A2-1. In-context deduplication, in distribution (original-world episodes; 95% bootstrap CI over episodes)\n")
print("One copy: 2 − rho (0 = no within-episode dedup, 1 = full). Four rows (s0 plus 3 copies; in distribution only for "
      "the costly world): (4 − rho)/3. Cue removal as in section 0 table B (out of distribution for both worlds).\n")
print("| depth | world | conv | one copy | four rows | no query tie | no labelled equality | new identity |")
print("|---|---|---|---|---|---|---|---|")
for L in (1, 2, 4):
    for name, g in (('original (control)', C), ('costly', A)):
        ms = g[L]
        print(f"| {L} | {name} | {sum(m['conv'] for m in ms)}/{len(ms)} | {cell(ms, 'copy')} | {cell(ms, 'copy3')} | "
              f"{np.mean([stat(m, 'qtie_off') for m in ms]):.2f} | {np.mean([stat(m, 'lab_off') for m in ms]):.2f} | "
              f"{np.mean([stat(m, 'id_off') for m in ms]):.2f} |")

print("\n### A2-2. Tied copy-vs-independent test (withdrawn as a headline after review 2; kept for the preregistered P1 "
      "continuity): 1 − pull(copy)/pull(independent row with the same query reading)\n")
print("| depth | original (control) | costly |")
print("|---|---|---|")
tc = defaultdict(list); ta = defaultdict(list)
for m in json.load(open(tied_ctl))['models']:
    if m['keys'] == 0:
        tc[m['depth']].append(1 - m['pull']['copy'] / m['pull']['indep'])
for m in json.load(open(os.path.join(a2, 'split_copy.json')))['models']:
    ta[m['depth']].append(1 - m['pull']['copy'] / m['pull']['indep'])
for L in (1, 2, 4):
    print(f"| {L} | {np.mean(tc[L]):.2f} | {np.mean(ta[L]):.2f} |")

print("\n### A2-3. Copy sweep: pull toward the copied reading vs number of rows m (mech2; Bayes 0 for every m)\n")
print("| depth | world | m=2 | m=3 | m=4 | m=8 |")
print("|---|---|---|---|---|---|")
for name, path in (('original (control)', os.path.join(ctl, 'mech2.json')), ('costly', os.path.join(a2, 'mech2.json'))):
    if not os.path.exists(path):
        continue
    g = defaultdict(list)
    for m in json.load(open(path))['models']:
        if m['trained'] and m['keys'] == 0:
            g[m['depth']].append(m['pull'])
    for L in (1, 2, 4):
        print(f"| {L} | {name} | " + " | ".join(f"{np.mean([p[k] for p in g[L]]):.3f}" for k in ('2', '3', '4', '8')) + " |")
