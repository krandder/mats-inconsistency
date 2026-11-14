"""E5 table: deduction vs Monte Carlo for the tree nets (results/e5.jsonl). The reference is the 8M-episode MC run.
Departure from Bayes: E[D^2] = E[mean^2] - 2 E[mean y*] + E[mean_bayes^2], with E[mean_bayes^2] = 1 - E[post var]
(world constant, low variance). Gap = E[NLL_net] - E[posterior entropy] (world constant).
Equal-time MC: a run with M episodes in t seconds has SE scaling as 1/sqrt(M); for tails, expected events = p M."""
import json, math, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'results/e5.jsonl')]
world = [r for r in rows if r['how'] == 'world' and 'H_ent' in r]
W = world[-1] if world else None
by = defaultdict(list)
for r in rows:
    if r['how'] != 'world':
        by[r['run'].rstrip('/').split('/')[-1]].append(r)
for net, rs in by.items():
    ref = max((r for r in rs if r['how'] == 'mc'), key=lambda r: r['n'])
    rate = ref['n'] / ref['seconds']                         # MC episodes per second
    print(f'\n== {net}  reference MC: {ref["n"]:,} episodes, {ref["seconds"]:.0f} s ({rate:,.0f} episodes/s)')
    if W:
        d2 = ref['mean2'] - 2 * ref['mean_y'] + W['Emb2_var']
        print(f'   world: H = E[post entropy] {W["H_ent"]:.4f} (se {W["H_ent_se"]:.4f}); E[mean_B^2] {W["Emb2_var"]:.5f} '
              f'(se {W["Emb2_var_se"]:.5f}); ref gap {ref["nll"] - W["H_ent"]:.4f}, ref E[D^2] {d2:.5f}')
    hdr = f'   {"method":24s} {"sec":>6s} {"E[NLL]":>9s} {"dNLL":>8s} {"E[D^2]":>9s} ' + ' '.join(f'P>{int(L):<2d}' + ' ' * 6 for L in (4, 6, 8, 10, 12, 14, 16, 20))
    print(hdr)
    for r in rs:
        a = r['args']
        lab = r['how'] + ('' if r['how'] == 'mc' else f' s{a["nsrc"]} y{a["ny"]}' + (f' G{a["G"]}' if r['how'] == 'exact' else ''))
        if r['how'] == 'mc':
            lab += f' M{r["n"] // 1000}k'
        d2 = r['mean2'] - 2 * r['mean_y'] + (W['Emb2_var'] if W else float('nan'))
        tails = ' '.join(f'{r["tails"][str(float(L))]:.2e}' for L in (4, 6, 8, 10, 12, 14, 16, 20))
        print(f'   {lab:24s} {r["seconds"]:6.0f} {r["nll"]:9.4f} {r["nll"] - ref["nll"]:+8.4f} {d2:9.5f} {tails}')
    # equal-time MC for each deduction run
    for r in rs:
        if r['how'] == 'mc':
            continue
        M_eq = rate * r['seconds']
        se = ref['nll_se'] * math.sqrt(ref['n'] / M_eq)
        ev = ' '.join(f'{ref["tails"][str(float(L))] * M_eq:8.1f}' for L in (4, 6, 8, 10, 12, 14, 16, 20))
        print(f'   equal-time MC for {r["how"]} ({r["seconds"]:.0f} s): M = {M_eq:,.0f}, SE(E[NLL]) {se:.4f}, '
              f'expected events at the ref tail rates: {ev}')
