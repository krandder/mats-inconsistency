"""Markdown tables for the prior-level vs in-context split, from split.json and split_copy.json.

usage: python split_tables.py RUNS_DIR > split_tables.md
"""
import json, os, sys
from collections import defaultdict
import numpy as np

root = sys.argv[1]
S = json.load(open(os.path.join(root, 'split.json')))
C = json.load(open(os.path.join(root, 'split_copy.json')))
SPL = ('dup_singletons', 'shared_nodup', 'singletons_nodup')

g = defaultdict(list)
for m in S['models']:
    g[(m['depth'], m['keys'])].append(m['mse'])
c = defaultdict(list)
for m in C['models']:
    c[(m['depth'], m['keys'])].append(m['pull'])

print("### A. The decisive test: one extra row that is a copy vs an independent source with the same query reading\n")
print("Pull toward the duplicated reading o_s0 when one row is added to a 6-source episode (600 episodes; 3 seeds). "
      "(i) true copy; (ii) same values on all 9 questions, new identity; (iv) new independent source (fresh labelled "
      "readings) with the same query reading. A reference with no within-episode comparison gives the same pull for "
      f"all three: R1 {C['ref']['R1']:.3f}, R2 {C['ref']['R2']:.3f}. Bayes: copy 0, independent 0.176. "
      "In-context dedup = 1 − pull(i)/pull(iv): 0 = the model cannot tell a copy from an independent source.\n")
print("| depth | keys | (i) copy | (ii) new identity, same values | (iv) independent source | in-context dedup 1 − (i)/(iv) |")
print("|---|---|---|---|---|---|")
for k in sorted(c):
    P = c[k]
    f = lambda key: np.mean([p[key] for p in P])
    fr = [1 - p['copy'] / p['indep'] for p in P]
    print(f"| {k[0]} | {k[1]} | {f('copy'):.3f} | {f('newid'):.3f} | {f('indep'):.3f} | {np.mean(fr):.2f} "
          f"[{min(fr):.2f}..{max(fr):.2f}] |")

print("\n### B. Preregistered split: share of each model's MSE improvement over count-all that is beyond R1 / R2\n")
print("MSE of the predictive mean to the arm's Bayes mean, mean over 3 seeds. In-context share beyond R = "
      "(E_R − E_model)/(E_count-all − E_model); prior-level share = 1 − that. Shares only where the model improves on "
      "count-all by > 0.005 (on dup_singletons and singletons_nodup count-all's iid assumption is nearly exact, so "
      "'improvement over count-all' is ill-posed there; see C).\n")
for keys in (0, 1):
    r = S['refs'][str(keys)]
    print(f"\nkeys = {keys}. References: " + "; ".join(
        f"{s}: count-all {r[s]['a']:.4f}, R1 {r[s]['R1']:.4f}, R2 {r[s]['R2']:.4f}" for s in SPL) + "\n")
    print("| depth | MSE shared_nodup | in-context beyond R1 | beyond R2 | MSE singletons_nodup (R1, R2 above) | MSE dup_singletons |")
    print("|---|---|---|---|---|---|")
    for k in sorted(x for x in g if x[1] == keys):
        e = {s: np.mean([m[s] for m in g[k]]) for s in SPL}
        a, r1, r2 = r['shared_nodup']['a'], r['shared_nodup']['R1'], r['shared_nodup']['R2']
        imp = a - e['shared_nodup']
        sh = (f"{(r1 - e['shared_nodup']) / imp:.2f} | {(r2 - e['shared_nodup']) / imp:.2f}" if imp > 0.005 else "n/a | n/a")
        print(f"| {k[0]} | {e['shared_nodup']:.4f} | {sh} | {e['singletons_nodup']:.4f} | {e['dup_singletons']:.4f} |")

print("\n### C. Duplicate excess (post-hoc, PREREG addendum): MSE(dup_singletons) − MSE(singletons_nodup)\n")
r = S['refs']['0']
ex = lambda x: r['dup_singletons'][x] - r['singletons_nodup'][x]
print(f"References (no keys): count-all {ex('a'):.4f}, R1 {ex('R1'):.4f}, R2 {ex('R2'):.4f}, "
      f"R2eq (drops exact equal values: leaks copy identity) {ex('R2eq'):.4f}, Bayes 0. "
      "In-context share = (excess_R1 − excess_model)/(excess_count-all − excess_model).\n")
print("| depth | keys | model excess | in-context share beyond R1 |")
print("|---|---|---|---|")
for k in sorted(g):
    rr = S['refs'][str(k[1])]
    exr = lambda x: rr['dup_singletons'][x] - rr['singletons_nodup'][x]
    em = np.mean([m['dup_singletons'] - m['singletons_nodup'] for m in g[k]])
    print(f"| {k[0]} | {k[1]} | {em:.4f} | {(exr('R1') - em) / (exr('a') - em):.2f} |")
