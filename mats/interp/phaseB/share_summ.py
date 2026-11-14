"""Tables for E6 (results/share.json, share_step2.json, share_step3.json).
usage: python share_summ.py -> results/share_tables.md
"""
import json, os

F = ('mean', 'logsd', 'kl'); LAB = {'mean': 'mean', 'logsd': 'log-sd', 'kl': 'KL'}
f = lambda x: f'{x:.2f}'.replace('-', '−')
s = json.load(open('results/share.json')); out = []
out.append('### Shared fraction of the noise-free deviation across independently trained nets (8,000 test episodes)\n')
out.append('| feature | same-recipe nets: mean over pairs [95% CI] (pair range) | cross-recipe pairs | E5 named-feature fraction, same scale (seeds 4 / 1 / 2) | reliability of one K = 64 estimate (range over nets) |')
out.append('|---|---|---|---|---|')
for k in F:
    r = s['features'][k]; m = r['main']; c = r.get('cross')
    e5 = ' / '.join(f"{s['e5_named'][nm][k]['latent_fraction']:.2f}" for nm in ('L2_s0.05', 'L2seed1_s0.05', 'L2seed2_s0.05'))
    rel = list(r['reliability'].values())
    out.append(f"| {LAB[k]} | {f(m['mean'])} [{f(m['ci'][0])}, {f(m['ci'][1])}] ({f(m['pair_min'])} to {f(m['pair_max'])}) | "
               + (f"{f(c['mean'])} [{f(c['ci'][0])}, {f(c['ci'][1])}] ({f(c['pair_min'])} to {f(c['pair_max'])})" if c else '–')
               + f" | {e5} | {min(rel):.3f}–{max(rel):.3f} |")
out.append(f"\nNets: {', '.join(s['nets'])}; {len(s['pairs_main'])} same-recipe pairs, {len(s['pairs_cross'])} cross-recipe pairs.\n")
if os.path.exists('results/share_step2.json'):
    t = json.load(open('results/share_step2.json'))
    out.append('### Step 2: what carries the shared mean deviation\n')
    out.append('| features | ' + ' | '.join(f'{LAB[k]}: R² vs seed average (explained / shared)' for k in F) + ' |'); out.append('|---|---|---|---|')
    for r in t['rows']:
        out.append(f"| {r['features']} | " + ' | '.join(f"{f(r[k]['R2'])} ({f(r[k]['explained_over_shared'])})" for k in F) + ' |')
    out.append('\n| description | ' + ' | '.join(f'{LAB[k]}: R² vs seed average / vs own deviation' for k in F) + ' |'); out.append('|---|---|---|---|')
    for d, r in t['descriptions'].items():
        out.append(f'| {d} | ' + ' | '.join(f"{f(r[k]['R2_vs_seed_average'])} / {f(r[k]['R2_vs_own'])}" for k in F) + ' |')
if os.path.exists('results/share_step3.json'):
    t = json.load(open('results/share_step3.json'))
    out.append('\n### Step 3: the systematic spread (log-sd) error\n')
    out.append('Permutation importance (drop in held-out R² of the seed-averaged log-sd deviation): ' + '; '.join(f"{r['feature']} {r['drop_in_R2']:.3f}" for r in t['importance'][:6]) + '.\n')
    c = t['compact']
    out.append(f"Compact model on {', '.join(t['top3'])} ({c['n_params']} parameters): R² vs the seed average {f(c['R2_vs_seed_average'])} "
               f"(explained / shared {f(c['explained_over_shared'])}); all ~50 named features: {f(c['full_named_R2_vs_seed_average'])} "
               f"({f(c['full_named_explained_over_shared'])}); per net: " + ', '.join(f"{nm} {f(v)}" for nm, v in c['R2_per_net'].items()) + '.\n')
    for q, sh in c['shapes'].items():
        out.append(f"- {q}: at its 5/25/50/75/95% quantiles {', '.join(f'{x:.2f}' for x in sh['quantiles'])} the predicted log-sd deviation is {', '.join(f'{x:+.3f}' for x in sh['prediction'])}")
    if 'probes' in t:
        out.append('\nRidge probes in seed 4 (held-out R²): own log-sd deviation / seed-averaged log-sd deviation / ' + ' / '.join(t['top3']) + '\n')
        out.append('| site / view | own | seed average | ' + ' | '.join(t['top3']) + ' |'); out.append('|---|---|---|' + '---|' * len(t['top3']))
        for sv, r in t['probes'].items():
            vals = list(r.values())
            out.append(f'| {sv} | ' + ' | '.join(f(v) for v in vals) + ' |')
txt = '\n'.join(out); open('results/share_tables.md', 'w').write(txt); print(txt)
