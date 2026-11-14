"""Markdown tables for the report from analysis2.json (mean-based dedup, probe baselines) and mech2.json (copy sweep).

usage: python make_tables.py RUNS_DIR > tables.md
"""
import json, os, sys
from collections import defaultdict
import numpy as np

root = sys.argv[1]
A = json.load(open(os.path.join(root, 'analysis2.json')))
M = json.load(open(os.path.join(root, 'mech2.json')))
MS = [str(m) for m in range(1, 9)]


def mse(v):
    v = np.array(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


g = defaultdict(list)
for r in A:
    g[(r['depth'], r['keys'])].append(r)

print("### Predictive mean vs the Bayes mean (MSE; separates deduplication from variance calibration)\n")
print("Duplicate excess = MSE on duplicate episodes with all-singleton partitions minus MSE on all-singleton episodes "
      "without duplicates. Count-all (a) has excess ≈ 0.018; Bayes has 0. Fraction = model excess / (a) excess.\n")
print("| depth | keys | n seeds | conv | MSE singletons, no dup | MSE dup | duplicate excess | fraction of (a) | MSE shared_any (a: 0.126, b: 0.116) | untrained MSE shared_any |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(g):
    rs = g[k]
    a_ex = rs[0]['trained']['splits']['dup_singletons']['mse_a'] - rs[0]['trained']['splits']['singletons_nodup']['mse_a']
    nod = [r['trained']['splits']['singletons_nodup']['mse_model'] for r in rs]
    dup = [r['trained']['splits']['dup_singletons']['mse_model'] for r in rs]
    ex = np.array(dup) - np.array(nod)
    sh = [r['trained']['splits']['shared_any']['mse_model'] for r in rs]
    un = [r['untrained']['splits']['shared_any']['mse_model'] for r in rs]
    conv = sum(r['converged'] for r in rs)
    print(f"| {k[0]} | {k[1]} | {len(rs)} | {conv}/{len(rs)} | {mse(nod)} | {mse(dup)} | {mse(ex)} | {ex.mean() / a_ex:.2f} | {mse(sh)} | {np.mean(un):.2f} |")

print("\n### Probes (retrained ridge per layer) and attention, trained vs untrained (same init seed); mean over seeds\n")
print("| depth | keys | n | group-size R² by layer, trained | untrained | query → Bayes-mean R², trained | untrained | corr(query attn per source, GLS weight) by layer, trained | untrained |")
print("|---|---|---|---|---|---|---|---|---|")
f = lambda x: ", ".join(f"{v:.2f}" for v in np.mean(x, 0))
for k in sorted(g):
    rs = [r for r in g[k] if 'mech' in r['trained']]
    if not rs:
        continue
    T = lambda t, key: [r[t]['mech'][key] for r in rs]
    print(f"| {k[0]} | {k[1]} | {len(rs)} | {f(T('trained', 'probe_groupsize'))} | {f(T('untrained', 'probe_groupsize'))} | "
          f"{f(T('trained', 'probe_query_bayesmean'))} | {f(T('untrained', 'probe_query_bayesmean'))} | "
          f"{f(T('trained', 'attn_corr_gls'))} | {f(T('untrained', 'attn_corr_gls'))} |")

ref = M['reference']
h = defaultdict(list)
for r in M['models']:
    h[(r['depth'], r['keys'], r['trained'])].append(r)
print("\n### Copy sweep: one source's row copied to m rows (m = 1..8; training only ever had m ≤ 2)\n")
print("Pull = OLS slope of (μ_m − μ_1) on (o_dup − μ_1). True duplicates (Bayes): 0 for every m.\n")
print("| model | n | m=2 | m=3 | m=4 | m=6 | m=8 |")
print("|---|---|---|---|---|---|---|")
print("| Bayes, true duplicates | | 0 | 0 | 0 | 0 | 0 |")
print("| GLS + independent copies | | " + " | ".join(f"{ref['indep_pull'][m]:.3f}" for m in ['2', '3', '4', '6', '8']) + " |")
print("| count-all (a) | | 0.121 | 0.216 | 0.293 | 0.408 | 0.491 |")
for k in sorted(h):
    rs = h[k]
    if not k[2]:
        continue
    vals = [np.mean([r['pull'][m] for r in rs]) for m in ['2', '3', '4', '6', '8']]
    un = np.mean([r['pull']['8'] for r in h[(k[0], k[1], False)]])
    print(f"| L{k[0]} keys {k[1]} | {len(rs)} | " + " | ".join(f"{v:.3f}" for v in vals) + f" |  (untrained m=8: {un:.3f})")

print("\nLog-precision shift (log σ_1 − log σ_m; positive = more confident). Bayes duplicates 0; independent copies "
      + ", ".join(f"m={m}: {ref['indep_logprec_shift'][m]:.3f}" for m in ['2', '4', '8']) + ".\n")
print("| model | m=2 | m=4 | m=8 |")
print("|---|---|---|---|")
for k in sorted(h):
    if k[2]:
        rs = h[k]
        print(f"| L{k[0]} keys {k[1]} | " + " | ".join(f"{np.mean([r['logprec_shift'][m] for r in rs]):.3f}" for m in ['2', '4', '8']) + " |")

print("\n### Query attention mass on the copies' query-question tokens (sum over copies), last layer\n")
print("Perfect deduplication by readout attention: total constant in m (per-copy mass ∝ 1/m). "
      "Ratio = total(m=2)/total(m=1); per-head ratios in brackets.\n")
print("| model | total m=1 | m=2 | m=4 | m=8 | ratio m2/m1 (heads) | untrained ratio | min head ratio over all layers |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(h):
    if not k[2]:
        continue
    rs = h[k]; un = h[(k[0], k[1], False)]; L = k[0] - 1
    tot = lambda rr, m, li=L: np.mean([np.mean(r['attn_total'][m][li]) for r in rr])
    hd = lambda rr, m, li=L: np.mean([r['attn_total'][m][li] for r in rr], 0)
    hr = hd(rs, '2') / hd(rs, '1')
    allh = min(float(np.min(hd(rs, '2', li) / hd(rs, '1', li))) for li in range(k[0]))
    print(f"| L{k[0]} keys {k[1]} | {tot(rs, '1'):.3f} | {tot(rs, '2'):.3f} | {tot(rs, '4'):.3f} | {tot(rs, '8'):.3f} | "
          f"{tot(rs, '2') / tot(rs, '1'):.2f} ({', '.join(f'{v:.2f}' for v in hr)}) | {tot(un, '2') / tot(un, '1'):.2f} | {allh:.2f} |")

print("\n### Is the copy count linearly decodable from the copies? (retrained ridge, R² for log m, by layer; 0 = input)\n")
print("| model | trained, copies' states | untrained | shuffled target | trained, within-episode h_m − h_1 | untrained within |")
print("|---|---|---|---|---|---|")
for k in sorted(h):
    if not k[2]:
        continue
    rs = h[k]; un = h[(k[0], k[1], False)]
    print(f"| L{k[0]} keys {k[1]} | {f([r['count_r2'] for r in rs])} | {f([r['count_r2'] for r in un])} | "
          f"{f([r['count_r2_shuffled'] for r in rs])} | {f([r['count_r2_within'] for r in rs])} | {f([r['count_r2_within'] for r in un])} |")

print("\n### Intervention: m = 4 copies, copies' residual after layer l replaced by the m = 1 state of the original token (superseded by the controls table below)\n")
print("If the copies' rewritten state carries the discount, the pull rises toward the independent-copies value "
      f"({ref['indep_pull']['4']:.3f}) when patched early. Patching after the last layer is a no-op control.\n")
print("| model | unpatched pull m=4 | patched after layer 0, 1, ... |")
print("|---|---|---|")
for k in sorted(h):
    if k[2]:
        rs = h[k]
        print(f"| L{k[0]} keys {k[1]} | {np.mean([r['patching']['pull_m'] for r in rs]):.3f} | "
              f"{f([[p['pull'] for p in r['patching']['patched']] for r in rs])} |")

p3 = os.path.join(root, 'mech3.json')
if os.path.exists(p3):
    Z = json.load(open(p3))
    zz = defaultdict(list)
    for r in Z:
        zz[(r['depth'], r['keys'], r['trained'])].append(r)
    print("\n### Do the query's attention scores deduplicate copies beyond softmax arithmetic?\n")
    print("Pooled over heads per layer: (arith − observed) / (arith − q1) for the total mass on the copies, where arith = "
          "m·q1/(1+(m−1)·P1) is what identical scores give and q1 is what perfect score-based deduplication gives. "
          "0 = no deduplication in the scores, 1 = full. Mean over seeds (range in brackets). Depth 1 is 0 by construction.\n")
    print("| model | m | layer 0, 1, ... trained | untrained (max abs) |")
    print("|---|---|---|---|")
    for k in sorted(zz):
        if not k[2]:
            continue
        for m in ['2', '4']:
            def pooled(rr):
                out = []
                for r in rr:
                    out.append([(sum(l['arith']) - sum(l['obs'])) / (sum(l['arith']) - sum(l['q1'])) for l in r['frac'][m]])
                return np.array(out)
            t = pooled(zz[k]); u = pooled(zz[(k[0], k[1], False)])
            cells = ", ".join(f"{a:.2f} [{lo:.2f}..{hi:.2f}]" for a, lo, hi in zip(t.mean(0), t.min(0), t.max(0)))
            print(f"| L{k[0]} keys {k[1]} | {m} | {cells} | {np.abs(u).max():.2f} |")

p4 = os.path.join(root, 'mech4.json')
if os.path.exists(p4):
    Z = json.load(open(p4))
    zz = defaultdict(list)
    for r in Z:
        zz[(r['depth'], r['keys'])].append(r)
    print("\n### Patching controls after layer 0 (reviewer-requested): which tokens' m=1 state removes the discount?\n")
    print("Pull toward the duplicated value; mean over seeds. 'random' = copies perturbed by a random vector of the same "
          "norm as the copies patch. Independent-copies reference: m=2 0.176, m=4 0.384; true duplicates 0.\n")
    print("| model | m | n | unpatched | copies ← m=1 | rest ← m=1 | all ← m=1 | random |")
    print("|---|---|---|---|---|---|---|---|")
    for k in sorted(zz):
        rs = zz[k]
        for m in ['2', '4']:
            c = lambda key: np.mean([r['ctl'][m]['0'][key] for r in rs])
            print(f"| L{k[0]} keys {k[1]} | {m} | {len(rs)} | {np.mean([r['ctl'][m]['unpatched'] for r in rs]):.3f} | "
                  f"{c('copies'):.3f} | {c('rest'):.3f} | {c('all'):.3f} | {c('random'):.3f} |")
