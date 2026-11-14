"""mkfigs.py OUTROOT WORLD [WORLD ...] : plain figures from each world's summary.json. EVALUATION CODE."""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NS = [64, 256, 1024, 4096, 20480]
SHOW = [('data', None, 'real outcomes only'), ('i', 10.0, '(i) pairs, weight 10'), ('iio', 10.0, '(ii) other order, weight 10'),
        ('iis', 1.0, '(ii) score, weight 1'), ('iii', 1.0, '(iii) over compute, weight 1'), ('synth', 1.0, "M's sampled outcomes, weight 1")]


def mean(S, arm, lam, n, f):
    rs = [f(m) for m in S['runs'].values() if m.get('arm') == arm and m.get('n') == n and (lam is None or m.get('lam') == lam) and (m.get('r') == 0 or n == 0)]
    return float(np.mean(rs)) if rs else np.nan


if __name__ == '__main__':
    root, worlds = sys.argv[1], sys.argv[2:]
    SS = [json.load(open('%s/w%s/summary.json' % (root, w))) for w in worlds]
    Ks = sorted(int(k) for k in next(m for m in SS[0]['runs'].values() if m.get('arm') == 'data')['t1'])
    # Figure 1: excess against the exact posterior by number of real check vectors
    fig, axs = plt.subplots(1, len(SS), figsize=(6 * len(SS), 4.8), squeeze=False)
    for ax, S, w in zip(axs[0], SS, worlds):
        for arm, lam, lab in SHOW:
            ax.plot(NS, [mean(S, arm, lam, n, lambda m: m['t1_avg']) for n in NS], marker='o', label=lab)
        ax.axhline(S['ref']['p_one'], color='gray', linestyle=':', label='one pass of M')
        ax.axhline(S['ref']['score_T'], color='black', linestyle='--', label="the score's exact answer")
        z = mean(S, 'iis', 1.0, 0, lambda m: m['t1_avg'])
        if not np.isnan(z):
            ax.axhline(z, color='C3', linestyle='-.', label='(ii) score alone, no real outcome')
        ax.set_xscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([format(n, ',') for n in NS])
        ax.set_xlabel('real check vectors with outcomes')
        ax.set_ylabel('excess log loss against the exact posterior (nats)')
        ax.set_title('world %s' % w)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(root + '/../fig/fig1_excess_by_n.png', dpi=130)
    # Figure 2: pair mismatch on held-out real triples, by n, at the largest K
    K = str(Ks[-1])
    fig, axs = plt.subplots(1, len(SS), figsize=(6 * len(SS), 4.8), squeeze=False)
    for ax, S, w in zip(axs[0], SS, worlds):
        for arm, lam, lab in SHOW:
            ax.plot(NS, [mean(S, arm, lam, n, lambda m: m['t2'][K]['D']) for n in NS], marker='o', label=lab)
        ax.plot(NS, [mean(S, 'data', None, n, lambda m: m['t2'][K].get('D_noise', np.nan)) for n in NS], color='gray', linestyle=':', label='noise floor of real-outcomes-only (two draws, one order)')
        ax.axhline(S['runs']['fixed-one_n0_d0_r0']['t2'][K]['D'], color='black', linestyle='--', label='one pass of M')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(NS)
        ax.set_xticklabels([format(n, ',') for n in NS])
        ax.set_xlabel('real check vectors with outcomes')
        ax.set_ylabel('order-against-order mismatch on pairs, K = %s' % K)
        ax.set_title('world %s' % w)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(root + '/../fig/fig2_pair_mismatch_by_n.png', dpi=130)
    print('figures written')
