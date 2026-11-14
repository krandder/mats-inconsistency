"""E51 analysis (PREREG-B amendment 59). Inputs: results/e51/<setting>.npz from e51_run.py (w1, d1, d2).
For the signed (ctrl) nets of a setting, with m_s[k, i] the predictive mean of net s on family member k (k = 0 is the base) and
episode i:
  h_s[k, i] = m_s[k, i] - mean_k m_s[k, i]           (the net's sign-induced shift; mean over k is zero)
  rho_h = the mean over pairs of nets (s < t) of corr(h_s, h_t) over all (k, i)    (shared vs net-specific sign shifts)
  sys_h = Var_{k,i}(mean_s h_s) / mean_s Var_{k,i}(h_s)                        (the shared share of the shift's variance)
  rho_g = the same pairwise correlation for the nets' sign-averaged deviations from the Bayes mean, g_s = mean_k m_s - m_Bayes
          (and for invariant formats, their base deviations)                    (E6's shared error, for reference)
  c_sign = sum_s pc_s / sum_s kl_s over the ctrl nets (the KL reduction of averaging each net over its 9 family members)
  c_seed(F) = mean over the 5 subsets of pc / kl for the geometric mean of 9 different nets of format F on the base input.
Intervals: 2,000 bootstrap resamples of nets (rho_h over pairs of resampled distinct nets; c_sign over nets).
Predictions (registered in amendment 59):
  P204  world 2, D = 1: rho_h <= 0.2 (the sign shifts are mostly net-specific);
  P205  world 2, D = 1: c_seed(canon) >= 0.75 x c_sign(ctrl) (averaging one ctrl net over sign patterns gains about what
        averaging 9 independently trained invariant nets gains).
usage: python e51_summ.py -> results/e51.json, results/e51_tables.md
"""
import json, os
import numpy as np

rng = np.random.default_rng(510)
NB = 2000


def pair_corr(A, idx=None):
    """Mean pairwise correlation over the rows of A (nets x features), using the nets in idx (with repeats ignored)."""
    idx = sorted(set(range(len(A)) if idx is None else idx))
    Z = A[idx] - A[idx].mean(1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    C = Z @ Z.T
    iu = np.triu_indices(len(idx), 1)
    return float(C[iu].mean())


def main():
    res = {}
    for st in ('w1', 'd1', 'd2'):
        p = 'results/e51/%s.npz' % st
        if not os.path.exists(p):
            continue
        z = np.load(p); o = {}
        M = z['ctrl_M']                                                      # (nets, 9, n)
        H = (M - M.mean(1, keepdims=True)).reshape(len(M), -1)
        G = M.mean(1) - z['bayes_mean'][None]
        o['n_nets'] = int(len(M))
        o['rho_h'] = pair_corr(H)
        o['sys_h'] = float(H.mean(0).var() / H.var(1).mean())
        o['rho_g_ctrl'] = pair_corr(G)
        o['c_sign'] = float(z['ctrl_fam_pc'].sum() / z['ctrl_fam_kl'].sum())
        o['c_sign_per_net'] = (z['ctrl_fam_pc'] / z['ctrl_fam_kl']).tolist()
        o['ctrl_family_kl'] = float(z['ctrl_fam_kl'].mean()); o['ctrl_sym_kl'] = float((z['ctrl_fam_kl'] - z['ctrl_fam_pc']).mean())
        bs_r, bs_c = [], []
        for _ in range(NB):
            b = rng.choice(len(M), len(M))
            if len(set(b.tolist())) >= 2:
                bs_r.append(pair_corr(H, b.tolist()))
            bs_c.append(z['ctrl_fam_pc'][b].sum() / z['ctrl_fam_kl'][b].sum())
        o['rho_h_ci'] = [float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))]   # registered; subsampling (see below)
        # post-review correction (2026-10-30): pair_corr drops repeated nets, so the resampling above is a random subsampling of
        # the distinct nets, not a bootstrap. The interval for rho_h is now a leave-one-net-out jackknife (normal approximation).
        jk = np.array([pair_corr(H, [j for j in range(len(M)) if j != i]) for i in range(len(M))])
        se = float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum()))
        o['rho_h_jack_se'] = se; o['rho_h_jack_ci'] = [o['rho_h'] - 1.96 * se, o['rho_h'] + 1.96 * se]
        o['sys_h_floor_1_over_n'] = 1.0 / len(M)
        o['c_sign_ci'] = [float(np.percentile(bs_c, 2.5)), float(np.percentile(bs_c, 97.5))]
        for arm in ('ctrl', 'mag', 'canon', 'prod'):
            if '%s_seed_kl' % arm in z:
                c = z['%s_seed_pc' % arm] / z['%s_seed_kl' % arm]
                o['c_seed_%s' % arm] = float(c.mean()); o['c_seed_%s_range' % arm] = [float(c.min()), float(c.max())]
                o['seed_ens_kl_%s' % arm] = float((z['%s_seed_kl' % arm] - z['%s_seed_pc' % arm]).mean())
                o['seed_member_kl_%s' % arm] = float(z['%s_seed_kl' % arm].mean())
            if arm != 'ctrl' and '%s_M' % arm in z:
                o['rho_g_%s' % arm] = pair_corr(z['%s_M' % arm][:, 0] - z['bayes_mean'][None])
        res[st] = o
    pr = {}
    d1 = res.get('d1')
    if d1:
        pr['P204'] = dict(rho_h=d1['rho_h'], jackknife_ci=d1['rho_h_jack_ci'], subsampling_range=d1['rho_h_ci'], holds=bool(d1['rho_h'] <= 0.2))
        pr['P205'] = dict(c_seed_canon=d1.get('c_seed_canon'), c_sign_ctrl=d1['c_sign'],
                          holds=bool(d1.get('c_seed_canon') is not None and d1['c_seed_canon'] >= 0.75 * d1['c_sign']))
    res['predictions'] = pr
    md = ['# E51 tables (generated by e51_summ.py)', '',
          '| setting | ctrl nets | rho_h [95% jackknife CI] | ensemble-mean variance ratio (1/N floor 0.08-0.09) | rho_g ctrl | c_sign ctrl [95% CI] | c_seed ctrl | c_seed invariant formats |',
          '|---' * 8 + '|']
    for st, o in res.items():
        if st == 'predictions':
            continue
        inv = '; '.join('%s %.3f' % (a, o['c_seed_%s' % a]) for a in ('mag', 'canon', 'prod') if 'c_seed_%s' % a in o)
        md.append('| %s | %d | %.3f [%.3f, %.3f] | %.3f | %.3f | %.3f [%.3f, %.3f] | %.3f | %s |' % (
            st, o['n_nets'], o['rho_h'], o['rho_h_jack_ci'][0], o['rho_h_jack_ci'][1], o['sys_h'], o['rho_g_ctrl'], o['c_sign'], o['c_sign_ci'][0],
            o['c_sign_ci'][1], o.get('c_seed_ctrl', float('nan')), inv))
    md += ['', '| setting | ctrl family KL | ctrl sign-averaged KL | seed ensembles: member KL -> ensemble KL |', '|---|---|---|---|']
    for st, o in res.items():
        if st == 'predictions':
            continue
        ens = '; '.join('%s %.4f -> %.4f' % (a, o['seed_member_kl_%s' % a], o['seed_ens_kl_%s' % a]) for a in ('ctrl', 'mag', 'canon', 'prod')
                        if 'seed_member_kl_%s' % a in o)
        md.append('| %s | %.4f | %.4f | %s |' % (st, o['ctrl_family_kl'], o['ctrl_sym_kl'], ens))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open('results/e51.json', 'w'), indent=1, default=float)
    open('results/e51_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
