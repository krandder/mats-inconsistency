"""Score PREREG Part C (PC1-PC7) from runs/partC_*.json; print a table."""
import glob, json, math, os
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary'
rows = [json.load(open(f)) for f in glob.glob(f'{D}/runs/partC_n*_L*_sig*_s*.json')]
rows.sort(key=lambda r: (r['n'], r['sigma'], r['L'], r['seed']))
out = {}
pc = {k: [] for k in ['PC1a', 'PC1b', 'PC2', 'PC3', 'PC5a1', 'PC5a2', 'PC5b1', 'PC5b2', 'PC5c', 'PC7a', 'PC7b', 'PC7c']}
for r in rows:
    Tm = r['T_mem_heuristic']; L = r['L']; n = r['n']
    sx = r['input_share'] / (r['input_share'] + r['noise_share'])   # review finding 7: between / (between + within)
    r['input_share_corr'] = sx
    if Tm >= n / 8:
        if L <= Tm / 4:
            pc['PC1a'].append(sx >= 0.9)
        if L >= 2 * Tm:
            pc['PC1b'].append(sx <= 0.2)
    elif L >= 2 * Tm:
        pc['PC1b'].append(sx <= 0.2)
    if L <= Tm / 2:
        pred = r['sigma'] * math.exp(5 * L / (4 * n))
        pc['PC2'].append(1 / 2.5 <= r['eps_med'] / pred <= 2.5)
        pb = math.sqrt(2) / math.pi * n * r['eps_med']
        pc['PC3'].append(1 / 1.5 <= r['births'] / pb <= 1.5 if pb > 0 else None)
    if sx >= 0.9:
        pc['PC5a1'].append(r['anti_noise_pair'] >= 1.5)
        if L <= n / 2:
            pc['PC5b1'].append(r['rqmc_x_ratio_N256'] >= 2)
        if r.get('anova_interaction') is not None:
            pc['PC7a'].append(r['anova_interaction'] <= 0.05)
    if sx <= 0.2:
        pc['PC5a2'].append(r['anti_noise_pair'] <= 0.7)
        pc['PC5b2'].append(r['rqmc_x_ratio_N256'] < 1.5)
    if L <= Tm / 4:
        pc['PC5c'].append(r['coupled_diff_var'] <= 0.01)
    if r.get('anova_interaction') is not None:
        pc['PC7b'].append(r['anova_interaction'] <= 0.35)
        if L >= 2 * Tm:
            pc['PC7c'].append(r['anova_input'] <= 0.1)
for k, v in pc.items():
    v2 = [x for x in v if x is not None]
    out[k] = dict(hit=all(v2) if v2 else None, n_true=sum(v2), n=len(v2))
# PC4: ARC regime
arc = [r for r in rows if r['n'] == 256 and r['L'] == 32 and abs(r['sigma'] - 1e-3) < 1e-12]
if arc:
    r = arc[0]
    out['PC4'] = dict(hit=(r['input_share'] / (r['input_share'] + r['noise_share']) >= 0.999 and r['noise_effect_nmse'] <= 1e-5 and r['births'] <= 0.3 and
                           r['input_gate_entropy_bits'] >= 128 and r['coupled_diff_var'] <= 1e-3),
                      values={k: r[k] for k in ['input_share', 'noise_effect_nmse', 'births', 'input_gate_entropy_bits', 'coupled_diff_var',
                                                'anti_both_pair', 'rqmc_x_ratio_N256', 'anova_input', 'anova_noise', 'anova_interaction']})
# PC6: crossover of births*n/2 vs input gate entropy along L for each (n, sigma, seed)
cross = []
for key in sorted(set((r['n'], r['sigma'], r['seed']) for r in rows)):
    rs = sorted([r for r in rows if (r['n'], r['sigma'], r['seed']) == key], key=lambda r: r['L'])
    Lc = next((r['L'] for r in rs if r['births'] * r['n'] / 2 > r['input_gate_entropy_bits']), None)
    Tm = rs[0]['T_mem_heuristic']
    ok = None if Lc is None else (Tm / 8 <= Lc <= 4 * max(Tm, 1e-9) or Tm == 0)
    cross.append(dict(n=key[0], sigma=key[1], seed=key[2], L_cross=Lc, T_mem=Tm, hit=ok))
out['PC6'] = dict(hit=all(c['hit'] for c in cross if c['hit'] is not None), rows=cross)
json.dump(dict(out=out, rows=rows), open(f'{D}/results/partC_scores.json', 'w'), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:600])
print('| n | sigma | L | T_mem | input share | ANOVA input/noise/inter | eps_med | births | input gate H | noise effect | coupled diff var | anti noise | anti both | RB | RQMC-x |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    an = f"{r.get('anova_input', float('nan')):.3f}/{r.get('anova_noise', float('nan')):.3f}/{r.get('anova_interaction', float('nan')):.3f}"
    print(f"| {r['n']} | {r['sigma']:g} | {r['L']} | {r['T_mem_heuristic']:.0f} | {r['input_share']:.4f} | {an} | {r['eps_med']:.2e} | {r['births']:.3f} | "
          f"{r['input_gate_entropy_bits']:.1f} | {r['noise_effect_nmse']:.2e} | {r['coupled_diff_var']:.2e} | {r['anti_noise_pair']:.2f} | "
          f"{r['anti_both_pair']:.2f} | {r['rb_ratio']:.3f} | {r['rqmc_x_ratio_N256']:.2f} |")
