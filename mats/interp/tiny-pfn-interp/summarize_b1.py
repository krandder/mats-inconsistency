"""B1 (PREREG Amendment 2): fresh seeds 5-9, H1 (layer-1 tilt rule) and H2 (combined replacement)."""
import json
import numpy as np
rows = []
print('| seed | test gap | L1 own/other ratio H0, H1 (QQ) | tilts (c1, c2) H0; H1 | H1: score, NLL_repl - NLL_model | H1 PASS | H2: score (NLL cost, model gap) | H2 PASS | 3b score | step-4 avg16 KL removed |')
print('|---|---|---|---|---|---|---|---|---|---|')
for s in range(5, 10):
    r1 = json.load(open(f'b1/out/main_s{s}.json')); r = json.load(open(f'b1/out/steps_main_s{s}.json'))
    s3 = r['step3']; gap = r1['nll_gap']
    a = r1['attn_trained']
    t = s3['ph_attn_tilt_vs_flat']; dn = t['nll_replaced'] - s3['nll_model']
    c = s3['ph_tilt_subspace_3b']
    plateau = gap > 0.05
    h1 = (t['score'] >= 0.8) and (dn <= 0.002)
    h2 = c['score'] >= 0.8
    tl = s3['tilt']
    print(f"| {s} | {gap:.4f}{' (plateau, excluded)' if plateau else ''} | {a['L1H0']['QQ']['ratio_lab']:.1f}, {a['L1H1']['QQ']['ratio_lab']:.1f} | "
          f"{tl['0'][0]:+.2f}, {tl['0'][1]:+.2f}; {tl['1'][0]:+.2f}, {tl['1'][1]:+.2f} | {t['score']:.2f}, {dn:+.4f} | {'-' if plateau else h1} | "
          f"{c['score']:.2f} ({c['nll_replaced'] - s3['nll_model']:+.3f}, {s3['nll_model'] - s3['nll_bayes']:.4f}) | {'-' if plateau else h2} | {s3['3b']['score']:.2f} | "
          f"{r['step4']['avg16']['frac_kl_removed']:.2f} |")
    if not plateau: rows.append((h1, h2))
n = len(rows); need = n - 1 if n < 5 else 4
print(f'\nNon-plateau seeds: {n}; threshold {need}. H1 PASS {sum(x[0] for x in rows)}/{n} -> {"CONFIRMED" if sum(x[0] for x in rows) >= need else "NOT confirmed"}; '
      f'H2 PASS {sum(x[1] for x in rows)}/{n} -> {"CONFIRMED" if sum(x[1] for x in rows) >= need else "NOT confirmed"}')
