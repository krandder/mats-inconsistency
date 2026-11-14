"""Tables for the rebuilt split section (after review 2): in-distribution rho with CIs and conv, cue removal,
three-tier shared-error split with both normalisations.

usage: python split_tables2.py RUNS_DIR > split_tables2.md
"""
import json, os, sys
from collections import defaultdict
import numpy as np

root = sys.argv[1]
I = json.load(open(os.path.join(root, 'split_indist.json')))
S = json.load(open(os.path.join(root, 'split.json')))
rng = np.random.default_rng(0)
VAR = ('copy', 'qtie_off', 'lab_off', 'id_off')

g = defaultdict(list)
for m in I['models']:
    g[(m['depth'], m['keys'])].append(m)


def dedup(m, v, idx=None):
    w0, w1 = (np.array(x) for x in m['w'][v])
    if idx is not None:
        w0, w1 = w0[idx], w1[idx]
    return 2 - w0.mean() / w1.mean()


def ci(ms, v, B=1000):
    n = len(ms[0]['w'][v][0])
    bs = [np.mean([dedup(m, v, i) for m in ms]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(bs, [2.5, 97.5])


print("### A. In-distribution test: how much of an exact copy does each model remove using within-episode evidence?\n")
print(f"{I['n']} episodes with 6 singleton sources plus one exact copy of source s0 (seed 78000). "
      "rho = (d mu / d o* for s0, moving both its rows) / (d mu / d o* for another source s1 in the same episode). "
      "A reader that cannot tell which rows are copies has rho = 2; full deduplication has rho = 1. "
      "In-context dedup = 2 − rho. Mean over 3 seeds; 95% CI from a bootstrap over episodes (seeds fixed); "
      "seed range in brackets. conv = seeds meeting the preregistered convergence rule.\n")
print(f"References: R1 (additive, no comparison) 2 − rho = {2 - I['refs']['R1']['copy']:.2f}; "
      f"R2 (Bayes on query readings, no identities or labels) {2 - I['refs']['R2']['copy']:.2f}; Bayes 1.00.\n")
print("| depth | keys | conv | in-context dedup 2 − rho | 95% CI | seed range | copy pull | untied independent pull (Bayes 0.121) |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(g):
    ms = g[k]
    vals = [dedup(m, 'copy') for m in ms]; lo, hi = ci(ms, 'copy')
    print(f"| {k[0]} | {k[1]} | {sum(m['conv'] for m in ms)}/{len(ms)} | {np.mean(vals):.2f} | [{lo:.2f}, {hi:.2f}] | "
          f"{min(vals):.2f}..{max(vals):.2f} | {np.mean([m['pull_copy'] for m in ms]):.3f} | "
          f"{np.mean([m['pull_indep'] for m in ms]):.3f} |")

print("\n### B. Which cue does the deduplication use? Remove one cue from the copy at a time (out of distribution)\n")
print("In-context dedup 2 − rho for an exact copy and for copies with one cue removed: the exact query tie (copy's "
      "query reading offset by delta ~ N(0, 0.5)); equality of the 8 labelled readings (fresh values); identity "
      "vector and group key (new ones). A large drop means that cue is used. These copies never occur in training.\n")
print("| depth | keys | exact copy | no query tie | no labelled equality | new identity and key |")
print("|---|---|---|---|---|---|")
print(f"| R1 | | " + " | ".join(f"{2 - I['refs']['R1'][v]:.2f}" for v in VAR) + " |")
print(f"| R2 | | " + " | ".join(f"{2 - I['refs']['R2'][v]:.2f}" for v in VAR) + " |")
for k in sorted(g):
    ms = g[k]
    print(f"| {k[0]} | {k[1]} | " + " | ".join(f"{np.mean([dedup(m, v) for m in ms]):.2f}" for v in VAR) + " |")

print("\n### C. Shared errors in three tiers (preregistered split, corrected reading)\n")
print("MSE of the predictive mean to the arm's Bayes mean on shared-error episodes without duplicates (n = "
      f"{S['n']['shared_nodup']}), mean over 3 seeds. Tiers: prior (R1, no comparison), query-value comparison "
      "(R1 → R2), labelled/identity/key evidence (beyond R2). Two normalisations: the preregistered share of the "
      "model's gain over count-all, and the share of R's remaining error that the model removes. R1 and R2 are "
      "fitted per row count n (here n = 6 always).\n")
gs = defaultdict(list)
for m in S['models']:
    gs[(m['depth'], m['keys'])].append(m['mse']['shared_nodup'])
conv = {k: sum(m['conv'] for m in v) for k, v in g.items()}
print("| depth | keys | conv | MSE model | count-all | R1 | R2 | gain over count-all: prior / query-value / beyond R2 | "
      "R1's error removed | R2's error removed |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(gs):
    r = S['refs'][str(k[1])]['shared_nodup']
    e = np.mean(gs[k]); a, r1, r2 = r['a'], r['R1'], r['R2']
    gain = a - e
    print(f"| {k[0]} | {k[1]} | {conv[k]}/3 | {e:.4f} | {a:.4f} | {r1:.4f} | {r2:.4f} | "
          f"{(a - r1) / gain:.2f} / {(r1 - r2) / gain:.2f} / {(r2 - e) / gain:.2f} | {(r1 - e) / r1:.2f} | {(r2 - e) / r2:.2f} |")
