"""Amendment 6 tables."""
import json
import numpy as np

L3 = [json.load(open(f'l3out/L3_s{s}.json')) for s in range(10, 15)]
BA = [json.load(open(f'l3out/base_s{s}.json')) for s in (10, 12, 13, 14)]


def table(R, sites, need, title):
    print(f'\n### Probes, {title}: held-out R^2 trained (untrained); * = POSITIVE in >= {need}/{len(R)} seeds\n')
    print('| target | token | ' + ' | '.join(sites) + ' |'); print('|---|---|' + '---|' * len(sites))
    rows = [('E_lab', 'QQ'), ('E_bayes', 'QQ'), ('p_bayes', 'QQ'), ('dE', 'QQ'), ('dp', 'QQ'), ('beta', 'QQ'), ('dE', 'LAB'), ('dp', 'LAB'),
            ('dmean', 'QT'), ('dmean_coup', 'QT'), ('logvar', 'QT')]
    for t, k in rows:
        cells = []
        for s in sites:
            key = f'{t}|{k}|{s}'
            tr = np.array([r['probes_trained'][key][0] for r in R]); un = np.array([r['probes_untrained'][key][0] for r in R])
            sh = np.array([r['probes_trained'][key][1] for r in R])
            pos = ((tr >= 0.5) & (tr - np.maximum(un, sh) >= 0.1)).sum() >= need
            cells.append(f'{tr.mean():.2f}{"*" if pos else ""} ({un.mean():.2f})')
        print(f'| {t} | {k} | ' + ' | '.join(cells) + ' |')


table(L3, ['emb', 'attn1', 'mlp1', 'attn2', 'mlp2', 'attn3', 'mlp3'], 4, '3-layer (seeds 10-14)')
table(BA, ['emb', 'attn1', 'mlp1', 'attn2', 'mlp2'], 3, '2-layer base (seeds 10, 12, 13, 14)')

print('\n### Readout weights: R^2 of QT log-weights over QQ tokens on log beta / log beta_lab (mean over seeds), QQ mass\n')
print('| net | head | a (beta) | R^2 beta | R^2 beta_lab | QQ mass |'); print('|---|---|---|---|---|---|')
for nm, R in (('L3', L3), ('base', BA)):
    for k in R[0]['readout']:
        f = lambda g: np.mean([g(r['readout'][k]) for r in R])
        print(f"| {nm} | {k} | {f(lambda d: d['beta']['a']):.2f} | {f(lambda d: d['beta']['r2']):.2f} | {f(lambda d: d['beta_lab']['r2']):.2f} | {f(lambda d: d['qq_mass']):.2f} |")

print('\n### Replacements (held-out KL; top coupling quintile in parentheses)\n')
kb = np.mean([r['replace']['model']['kl'] for r in BA])
print(f'base mean KL {kb:.4f}; per L3 seed gap = KL_base(mean) - KL_L3\n')
keys = ['model', 'A_dE_1', 'A_rand_1', 'A_dE_2', 'A_rand_2', 'A_skipQQ2', 'A_attn3', 'A_mlp3', 'A_L3']
print('| seed | ' + ' | '.join(keys) + ' | dE fit R^2 after block 1 / 2 | A_dE_2 rise / gap | A_rand_2 rise / gap |')
print('|---|' + '---|' * (len(keys) + 3))
npass = 0
for r in L3:
    rp = r['replace']; gap = kb - rp['model']['kl']
    up = (rp['A_dE_2']['kl'] - rp['model']['kl']) / gap; ur = (rp['A_rand_2']['kl'] - rp['model']['kl']) / gap
    npass += (up >= 0.5) and (ur < 0.25)
    print(f"| {r['tag']} | " + ' | '.join(f"{rp[k]['kl']:.4f} ({rp[k]['kl_topcoup']:.4f})" for k in keys)
          + f" | {rp['dE_fit_r2_after_block1']:.2f} / {rp['dE_fit_r2_after_block2']:.2f} | {up:+.2f} | {ur:+.2f} |")
print(f'\nRule (A_dE_2 >= 50% of gap, A_rand_2 < 25%): {npass}/5 seeds')
print('\nbase:')
for r in BA:
    rp = r['replace']
    print(f"- {r['tag']}: model {rp['model']['kl']:.4f} ({rp['model']['kl_topcoup']:.4f}); A_dE_1 {rp['A_dE_1']['kl']:.4f} ({rp['A_dE_1']['kl_topcoup']:.4f}); "
          f"A_rand_1 {rp['A_rand_1']['kl']:.4f}; dE fit R^2 after block 1 {rp['dE_fit_r2_after_block1']:.2f}")
