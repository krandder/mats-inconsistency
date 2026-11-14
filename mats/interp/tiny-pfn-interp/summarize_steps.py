"""Tables for steps 2-4 (out/steps_*.json)."""
import json, os
import numpy as np

TAGS = [f'main_s{s}' for s in range(5)] + ['small_s2', 'small_s4', 'plateau_s3']
R = {t: json.load(open(f'out/steps_{t}.json')) for t in TAGS if os.path.exists(f'out/steps_{t}.json')}

print('### Step 2: shift of the model\'s predictive mean / log variance vs the Bayes shift (slope, R^2)\n')
V = ['full_swap', 'a_QQ_mlp1', 'b_QQ_attn1_pattern', 'c_LAB_mlp1', 'R_random_QQ_mlp1']
print('| network | ' + ' | '.join(V) + ' | untrained full / (a) |')
print('|---|' + '---|' * (len(V) + 1))
for t, r in R.items():
    a, u = r['step2_trained'], r['step2_untrained']
    f = lambda d: f"{d['mean']['slope']:.2f}, {d['mean']['r2']:.2f} / {d['logvar']['slope']:.2f}, {d['logvar']['r2']:.2f}"
    print(f'| {t} | ' + ' | '.join(f(a[v]) for v in V) + f" | {u['full_swap']['mean']['r2']:.2f} / {u['a_QQ_mlp1']['mean']['r2']:.2f} |")
print('\nR^2 of the patched mean shift on the model\'s own full-swap shift:')
for t, r in R.items():
    print(f'- {t}: ' + ', '.join(f"{v} {r['step2_trained'][v]['mean_vs_full']['r2']:.2f}" for v in V[1:]))

print('\n### Step 3: fraction of performance retained (score); NLL_ablated - NLL_model in parentheses\n')
K = ['3a_attn', '3a_attn_poolheads', '3a_mlp', '3a_mlp_z2', '3a_both', '3b', '3b_lab_too', '3_all',
     'ph_attn_tilt_vs_flat', 'ph_mlp_subspace', 'ph_tilt_subspace_3b']
print('| network | ' + ' | '.join(K) + ' |')
print('|---|' + '---|' * len(K))
for t, r in R.items():
    s3 = r['step3']
    print(f'| {t} | ' + ' | '.join(f"{s3[k]['score']:.2f} ({s3[k]['denom']:.3f})" if k in s3 else '-' for k in K) + ' |')
print('\nMLP1-output reconstruction R^2 (held-out) from [logSS, E] / +p:', {t: [round(r['step3']['mlp_fit_r2']['z1'], 2), round(r['step3']['mlp_fit_r2']['z2'], 2)] for t, r in R.items()})
print('Fitted layer-1 tilts (c1 on r, c2 on r^2) per head:', {t: {h: [round(x, 2) for x in v] for h, v in r['step3']['tilt'].items()} for t, r in R.items()})
print('Model NLL - Bayes NLL on held-out:', {t: round(r['step3']['nll_model'] - r['step3']['nll_bayes'], 4) for t, r in R.items()})

print('\n### Step 4: identity vectors (all 16000 episodes)\n')
print('| network | original gap / KL / sd(mean err) | const frame 0 / 1 / 2: KL (frac KL removed) | avg over 16 ID draws: gap / KL (frac KL removed, frac mean-err var removed) |')
print('|---|---|---|---|')
for t, r in R.items():
    s4 = r['step4']; o = s4['original']
    cf = ' / '.join(f"{s4[f'const_frame{f}']['kl']:.4f} ({s4[f'const_frame{f}']['frac_kl_removed']:+.2f})" for f in range(3))
    a = s4['avg16']
    print(f"| {t} | {o['gap']:.4f} / {o['kl']:.4f} / {o['sd_mean_err']:.4f} | {cf} | {a['gap']:.4f} / {a['kl']:.4f} ({a['frac_kl_removed']:+.2f}, {a['frac_meanerr_var_removed']:+.2f}) |")
