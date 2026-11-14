"""Figure: effective theory against the exact Gram chain and the networks (figs/theory_vs_exact.png)."""
import sys, json, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
G = json.load(open(f'{D}/results/partG_scores.json'))
G7 = json.load(open(f'{D}/results/partG7_scores.json'))
ns = sorted(int(k) for k in G['G2']['rows'])
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
ax[0].plot(ns, [G['G2']['rows'][str(n)]['ratio'] for n in ns], 'o-', color=CAT[0], ms=4,
           label='cloud angle delta: exact / kernel fixed point')
ax[0].plot(ns, [G['G3']['rows'][str(n)]['cv2_ratio'] for n in ns], 's-', color=CAT[1], ms=4,
           label='radial CV^2: exact / closure law')
ax[0].plot(ns, [G['G4']['Du_bin_vs_L0'][str(n)][0] / G['G4']['Du_bin_vs_L0'][str(n)][1] for n in ns], '^-', color=CAT[2],
           ms=4, label='direction interaction order: exact / L0')
ax[0].axhline(1, color=INK2, lw=0.8)
ax[0].set_xscale('log', base=2); ax[0].set_xlabel('width n'); ax[0].set_ylabel('exact Gram chain / effective theory')
ax[0].set_title('Fixed point of the noisy kernel recursion'); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1.6)
# the 5-8x
for lab, rows, mk, col in (('fresh nets (registered)', G7['fresh'], 'o', CAT[0]), ('Part A nets (post hoc)', G7['partA_posthoc'], 's', CAT[1])):
    r = [x for x in rows if x['n'] >= 32]
    ax[1].scatter([x['ratio_pred'] for x in r], [x['ratio_meas'] for x in r], marker=mk, s=22, color=col, label=lab)
xx = np.array([1.5, 14])
ax[1].plot(xx, xx, color=INK2, lw=0.8, label='measured = predicted')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel('predicted 2.1 Q delta*(Q) / (H_n - 1)'); ax[1].set_ylabel('measured eigenvalue / Lyapunov formula, k = 2-8')
ax[1].set_title('The cloud excess is the finite-angle fixed point (n >= 32)'); ax[1].legend(fontsize=7)
# participation ratio of the angular cloud vs harmonic
pr = G['G6']['rows']
npr = sorted(int(k) for k in pr)
ax[2].plot(npr, [pr[str(n)]['PR_u'] for n in npr], 'o-', color=CAT[0], ms=4, label='exact (4 replicas, median-radius bin)')
ax[2].plot(npr, [pr[str(n)]['harmonic'] for n in npr], '--', color=INK2, lw=1, label='1/k spectrum, k = 2..n')
ax[2].set_xscale('log', base=2); ax[2].set_xlabel('width n'); ax[2].set_ylabel('participation ratio of the angular cloud')
ax[2].set_title('Angular spectrum close to 1/k'); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{D}/figs/theory_vs_exact.png'); plt.close(fig)
print('ok')
