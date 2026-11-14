"""Tables for the post-hoc additions (extra.py)."""
import json
import numpy as np
SITES = ('emb', 'attn1', 'mlp1', 'attn2', 'mlp2')
G = {'main': [f'out/extra_main_s{s}.json' for s in range(5)], 'small': ['out/extra_small_s2.json', 'out/extra_small_s4.json'],
     'plateau': ['out/extra_plateau_s3.json']}
NEED = {'main': 4, 'small': 2, 'plateau': 1}
for g, fs in G.items():
    rs = [json.load(open(f)) for f in fs]
    print(f'\n### Post-hoc probes, {g}: R^2 trained (untrained); * = positive by the PREREG rule in >= {NEED[g]} seeds (shuffled baseline not rerun; it was <= 0.014 everywhere)\n')
    print('| target | token | ' + ' | '.join(SITES) + ' |'); print('|---|---|' + '---|' * 5)
    for name, kind in (('mean_a', 'QT'), ('dmean', 'QT'), ('dmean_lab', 'QT'), ('dmean', 'QQ'), ('dE', 'QQ'), ('dE', 'LAB')):
        cells = []
        for s in SITES:
            k = f'{name}|{kind}|{s}'
            tr = np.array([np.mean(r['probes_trained'][k]) for r in rs]); un = np.array([np.mean(r['probes_untrained'][k]) for r in rs])
            ok = ((tr >= 0.5) & (tr - np.maximum(un, 0.0) >= 0.1)).sum() >= NEED[g]
            cells.append(f'{tr.mean():.2f}{"*" if ok else ""} ({un.mean():.2f})')
        print(f'| {name} | {kind} | ' + ' | '.join(cells) + ' |')
    print(f'\nAttention weight to own-source labelled keys vs key residual, {g} (per seed, trained; untrained mean in parentheses):\n')
    print('| head | from | r(attn, signed r) | r(attn, r^2) |'); print('|---|---|---|---|')
    for k in rs[0]['attn_trained']:
        a = [r['attn_trained'][k] for r in rs]; u = [r['attn_untrained'][k] for r in rs]
        print(f"| {k.split('|')[0]} | {k.split('|')[1]} | {', '.join(f'{x['r_signed']:+.2f}' for x in a)} ({np.mean([x['r_signed'] for x in u]):+.2f}) | "
              f"{', '.join(f'{x['r_sq']:+.2f}' for x in a)} ({np.mean([x['r_sq'] for x in u]):+.2f}) |")
