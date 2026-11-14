"""Figure for the frontier (logs/pareto.png): x = sorting nMSE relative to its family's ReLU (learnability cost), y = K2's
error relative to MC at equal FLOPs on the trained sorting nets (log scale; below 1 = the deductive estimator beats MC).
Two families: free gates (circles, labelled) and sharpness-pinned softplus (squares, connected in beta order).
Palette: the dataviz skill's validated categorical slots 1-2 (#2a78d6, #eb6834; validate_palette.js: all checks pass).
usage: python3 scratch/pareto_fig.py   (reads runs/pareto.json)"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, '..', 'runs', 'pareto.json')))
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.axhline(1.0, color='#8a8a85', lw=1, ls='--', zorder=1)
ax.axvline(1.2, color='#8a8a85', lw=1, ls=':', zorder=1)
ax.text(0.86, 1.06, 'K2 = MC at equal FLOPs', fontsize=7.5, color='#55554f')
ax.text(1.195, 0.83, '<- within 20% of ReLU', fontsize=7.5, color='#55554f', ha='right')
free = [(a, r) for a, r in P['unpinned'].items() if r.get('sort_rel') and r.get('trained_k2_mc')]
ax.scatter([r['sort_rel'] for _, r in free], [r['trained_k2_mc'] for _, r in free], s=46, color='#2a78d6',
           edgecolor='#fcfcfb', linewidth=1.5, zorder=3, label='free gates (labelled)')
OFF = {'relu': (6, -10), 'silu': (-24, 4)}
for a, r in free:
    ax.annotate(a, (r['sort_rel'], r['trained_k2_mc']), xytext=OFF.get(a, (5, 3)), textcoords='offset points', fontsize=7.5,
                color='#2b2b28')
order = ['softplus1', 'softplus2', 'softplus4', 'softplus8', 'softplus16']
pin = [(a, P['pinned'][a]) for a in order if a in P['pinned'] and P['pinned'][a].get('trained_k2_mc')]
ax.plot([r['sort_rel'] for _, r in pin], [r['trained_k2_mc'] for _, r in pin], color='#eb6834', lw=2, zorder=2)
ax.scatter([r['sort_rel'] for _, r in pin], [r['trained_k2_mc'] for _, r in pin], s=46, marker='s', color='#eb6834',
           edgecolor='#fcfcfb', linewidth=1.5, zorder=3, label='pinned softplus, beta = 1 .. 16')
POFF = {'softplus16': (6, 1), 'softplus8': (-30, -4)}
for a, r in pin:
    ax.annotate('b=' + a[len('softplus'):], (r['sort_rel'], r['trained_k2_mc']), xytext=POFF.get(a, (5, -9)),
                textcoords='offset points', fontsize=7, color='#2b2b28')
rp = P['pinned'].get('relu', {})
if rp.get('sort_rel') and rp.get('trained_k2_mc'):
    ax.scatter([rp['sort_rel']], [rp['trained_k2_mc']], s=46, marker='s', facecolor='#fcfcfb', edgecolor='#eb6834', lw=1.5,
               zorder=3, label='pinned ReLU')
ax.set_yscale('log')
ax.set_xlabel('sorting nMSE relative to ReLU in the same family (lower = learns better)', fontsize=8.5, color='#2b2b28')
ax.set_ylabel('trained net: K2 error x N_eq (log; < 1 beats MC)', fontsize=8.5, color='#2b2b28')
ax.tick_params(labelsize=7.5, colors='#55554f')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color('#b9b9b2')
ax.grid(True, which='major', color='#e8e8e3', lw=0.6, zorder=0)
ax.legend(fontsize=7.5, frameon=False, loc='upper right')
ax.set_title('No gate both learns like ReLU and makes K2 beat MC; RQMC matches or beats K2 on every trained net', fontsize=8.5,
             color='#2b2b28', loc='left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, '..', 'logs', 'pareto.png'), facecolor=fig.get_facecolor())
print('wrote logs/pareto.png')
