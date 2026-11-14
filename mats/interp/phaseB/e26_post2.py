"""E26 post-review checks (after the Codex review, results/review_e26.md; not registered).
  1. Pattern costs without the finite-draw correction (the exact KL reduction for the 16 or 9 draws used), per set and group.
  2. Grid truncation: mean and maximum of (1 - smallest grid mass) per set; an upper bound on the truncation bias of -log Zbar.
  3. Mean bias: sum over deltas of (mean D)^2 for the base and all-new draws, against Vtot (the budget covers Var(D) only).
  4. Three-source planted term: VX counts it once, VX2 (half the sum of the 2x2 cuts) 1.5 times.
usage: python e26_post2.py -> results/e26_post2.json
"""
import json, os
import numpy as np
import e26_summ as SM
import e26_data as ED
import e26_check as CK

A_NZ = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05']
B_NZ = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']


def main():
    rows = json.load(open('results/e26_nets.json'))
    LB = np.load('results/e26_data.npz')['LB']
    res = {}
    groups = dict(M=[r['tag'] for r in rows if r['set'] == 'M'], G=[r['tag'] for r in rows if r['set'] == 'G'],
                  L1=[r['tag'] for r in rows if r['set'] == 'L1'], NZ=[r['tag'] for r in rows if r['set'] == 'NZ'],
                  M_A=SM.A_BASES, M_B=SM.B_BASES, NZ_A=A_NZ, NZ_B=B_NZ)
    for g, tags in groups.items():
        s = g.split('_')[0]
        kl = pc16 = pc9 = 0.0; miss = []; bias2 = vtot = 0.0
        for t in tags:
            z = np.load(os.path.join('results/e26', s, t + '.npz')); n = int(z['nb'])
            kl += z['kl16'].mean(); pc16 += z['pc16'].mean(); pc9 += z['pc9'].mean()
            miss.append(1 - z['mass'])
            D = z['L'].astype(np.float64)[:, :2, :6] - LB[:n, :2, :6]
            bias2 += 0.5 * ((D[:, 0].mean(0) ** 2).sum() + (D[:, 1].mean(0) ** 2).sum())
            vtot += 0.5 * (D[:, 0].var(0, ddof=1).sum() + D[:, 1].var(0, ddof=1).sum())
        miss = np.concatenate(miss)
        res[g] = dict(pattern_cost_share_uncorrected=pc16 / kl, sign_assign_cost_share_uncorrected=pc9 / kl,
                      pattern_cost_share_corrected=pc16 * 16 / 15 / kl, sign_assign_cost_share_corrected=pc9 * 9 / 8 / kl,
                      mean_pattern_cost_uncorrected=pc16 / len(tags), mean_sign_assign_cost_uncorrected=pc9 / len(tags),
                      mean_KL=kl / len(tags), missing_mass_mean=float(miss.mean()), missing_mass_max=float(miss.max()),
                      mean_bias_sq_over_Vtot=bias2 / vtot)
    for s in ('M', 'NZ'):
        a, b = res[s + '_A'], res[s + '_B']
        dkl = b['mean_KL'] - a['mean_KL']
        res[s + '_B_minus_A_uncorrected'] = dict(
            pattern_share_of_KL_diff=(b['mean_pattern_cost_uncorrected'] - a['mean_pattern_cost_uncorrected']) / dkl,
            sign_assign_share_of_KL_diff=(b['mean_sign_assign_cost_uncorrected'] - a['mean_sign_assign_cost_uncorrected']) / dkl)
    # 4. three-source planted check on the check data (seed 777), if present
    p = '/tmp/e26dev/check_data.npz'
    if os.path.exists(p):
        z = np.load(p); n = len(z['y'])
        def f(k):
            ys, e, u = CK.factors(z, k)
            return 0.4 * e[:, 0] * e[:, 1] * e[:, 2] + 0.3 * e.sum(1)
        L = np.stack([f(k) for k in range(ED.NVAR)], 1)[:, :, None] * np.ones(6)[None, None]
        zeros = np.zeros(n)
        C = SM.columns(L[:, :12], L[:, ED.PAT].var(1, ddof=1), L[:, ED.SP].var(1, ddof=1), zeros, zeros + 1, zeros, zeros + 1,
                       np.zeros_like(L[:, :12]), SM.DSEL_ALL)
        c = {k: float(v[0]) for k, v in SM.comps_from_means(np.ones((1, n)) / n @ C, 6, n).items()}
        ys, e, u = CK.factors(z, 0)
        planted = 6 * float((0.4 * e[:, 0] * e[:, 1] * e[:, 2]).var())
        res['three_source_check'] = dict(planted_triple=planted, VX=c['VX'], VX2=c['VX2'], VX2_over_planted=c['VX2'] / planted,
                                         VX_over_planted=c['VX'] / planted)
    json.dump(res, open('results/e26_post2.json', 'w'), indent=1)
    for k, v in res.items():
        print(k, {kk: round(vv, 5) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
