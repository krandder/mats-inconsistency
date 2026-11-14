"""E26 post-hoc descriptive analyses (written after seeing E26's registered results; not registered).
  1. Absolute KL costs (nats) by group: mean KL, pattern cost (x16/15), sign/assignment cost (x9/8), and the remainder
     (pattern minus sign/assignment, a rough "magnitude shape" share; the two costs do not add exactly).
  2. The A-B difference in mean KL, and how much of it the pattern and sign/assignment costs account for (bootstrap over bases).
  3. Depth with acquired nets only (heads cohort: E17's acquisition, test gap < 0.10).
usage: python e26_post.py -> results/e26_post.json (and prints a table)
"""
import json, os
import numpy as np
import e26_summ as SM

A_NZ = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05']
B_NZ = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']


def costs(set_name, tags):
    """Per-net arrays over bootstrap rows: KL, pattern cost, sign/assignment cost."""
    out = []
    for t in tags:
        f = os.path.join('results/e26', set_name, t + '.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f); n = int(z['nb']); W = SM.boot_weights(n)
        M = W @ np.stack([z['kl16'], z['pc16'] * 16 / 15, z['pc9'] * 9 / 8], 1)
        out.append(M)
    return np.stack(out, 0) if out else None                              # (nets, B+1, 3)


def row(M):
    m = M.mean(0)                                                         # mean over nets, per bootstrap row
    return dict(KL=SM.ci(m[:, 0]), pattern=SM.ci(m[:, 1]), sign_assign=SM.ci(m[:, 2]), shape_rest=SM.ci(m[:, 1] - m[:, 2]),
                pattern_share=SM.ci(m[:, 1] / m[:, 0]), sign_assign_share=SM.ci(m[:, 2] / m[:, 0]), n=int(M.shape[0]))


def gap_split(MA, MB):
    a, b = MA.mean(0), MB.mean(0)
    d = b - a
    return dict(KL_diff=SM.ci(d[:, 0]), pattern_diff=SM.ci(d[:, 1]), sign_assign_diff=SM.ci(d[:, 2]),
                pattern_share_of_diff=SM.ci(d[:, 1] / d[:, 0]), sign_assign_share_of_diff=SM.ci(d[:, 2] / d[:, 0]))


def tangled_abs(set_name, tags, LB):
    """Per-net absolute components over bootstrap rows: VX, its pattern part Vu_cross, its sufficient part, Vtot, N5 share."""
    out = []
    for t in tags:
        f = os.path.join('results/e26', set_name, t + '.npz')
        if not os.path.exists(f):
            continue
        c, _ = SM.net_comps(f, LB, None, SM.DSEL_ALL)
        out.append(np.stack([c['VX'], c['Vu_cross'], c['X_suff'], c['Vtot'], c['share_n5']], 1))
    return np.stack(out, 0) if out else None                              # (nets, B+1, 5)


def tangled_split(TA, TB):
    a, b = TA.mean(0), TB.mean(0); d = b - a
    return dict(VX_A=SM.ci(a[:, 0]), VX_B=SM.ci(b[:, 0]), pattern_part_A=SM.ci(a[:, 1]), pattern_part_B=SM.ci(b[:, 1]),
                sufficient_part_A=SM.ci(a[:, 2]), sufficient_part_B=SM.ci(b[:, 2]), VX_diff=SM.ci(d[:, 0]),
                pattern_part_diff=SM.ci(d[:, 1]), sufficient_part_diff=SM.ci(d[:, 2]),
                n5_share_A=SM.ci(a[:, 4]), n5_share_B=SM.ci(b[:, 4]))


def main():
    res = {}
    LB = np.load('results/e26_data.npz')['LB']
    for s, (ta, tb) in (('M', (SM.A_BASES, SM.B_BASES)), ('NZ', (A_NZ, B_NZ))):
        TA, TB = tangled_abs(s, ta, LB), tangled_abs(s, tb, LB)
        if TA is not None and TB is not None and len(TA) == 6 and len(TB) == 6:
            res['%s_tangled_B_vs_A' % s] = tangled_split(TA, TB)
    groups = {('M', 'A'): SM.A_BASES, ('M', 'B'): SM.B_BASES, ('NZ', 'A'): A_NZ, ('NZ', 'B'): B_NZ}
    Ms = {k: costs(k[0], v) for k, v in groups.items()}
    for k, M in Ms.items():
        if M is not None:
            res['%s_%s' % k] = row(M)
    for s in ('M', 'NZ'):
        if Ms.get((s, 'A')) is not None and Ms.get((s, 'B')) is not None:
            res['%s_B_minus_A' % s] = gap_split(Ms[(s, 'A')], Ms[(s, 'B')])
    e17 = json.load(open('results/e17.json'))['per_net']
    rows = json.load(open('results/e26_nets.json'))
    for arm in ('base_ws', 'L3_ws', 'base_g', 'L3_g'):
        tags = [r['tag'] for r in rows if r['set'] == 'HD' and r['arm'] == arm and e17.get(r['tag'], {}).get('acquired')]
        M = costs('HD', tags)
        if M is not None:
            res['HD_acquired_' + arm] = row(M)
    # 4. training: absolute costs per checkpoint (point estimates)
    for s in (5, 6):
        traj = []
        for r in sorted([r for r in rows if r['set'] == f'DY{s}'], key=lambda r: r['step']):
            f = os.path.join('results/e26', f'DY{s}', r['tag'] + '.npz')
            if os.path.exists(f):
                z = np.load(f)
                traj.append(dict(step=r['step'], KL=float(z['kl16'].mean()), pattern=float(z['pc16'].mean() * 16 / 15),
                                 sign_assign=float(z['pc9'].mean() * 9 / 8), mass_min=float(z['mass'].min())))
        if traj:
            res[f'DY{s}_costs'] = traj
    json.dump(res, open('results/e26_post.json', 'w'), indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {kk: (round(vv[0], 5) if isinstance(vv, list) else vv) for kk, vv in v.items()})
        else:
            print(k, 'trajectory with', len(v), 'checkpoints')


if __name__ == '__main__':
    main()
