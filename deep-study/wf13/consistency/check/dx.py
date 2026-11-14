"""dx.py: the cross-order pair mismatch with the trajectory-sampling noise removed, as a MEASUREMENT (not a loss), on the
work's stored forecasts. Dx = mean over triples of sum_xy w (log J1 - log J2)_drawA x (log J1 - log J2)_drawB; the two
draws use disjoint trajectories, so sampling noise cancels in expectation and Dx estimates the squared EXPECTED mismatch
between the two orders. D = the raw mismatch (one draw), Dn = the note's noise floor."""
import numpy as np, sys
R = '/home/agent/workspaces/mats/work/deep-study/wf13/consistency'
lsig = lambda z: -np.logaddexp(0.0, -z.astype(np.float64))
def joints(z):
    l = lambda a: np.stack([lsig(-a), lsig(a)], -1)
    return l(z[:, 0])[:, :, None] + np.stack([l(z[:, 2]), l(z[:, 3])], 1), l(z[:, 1])[:, None, :] + np.stack([l(z[:, 4]), l(z[:, 5])], 2)
def boot(v, nb=2000):
    rng = np.random.default_rng(3); m = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return '%.4f [%.4f, %.4f]' % (v.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975))
for w in (101, 102):
    A = np.load('%s/out/w%d/fixed-one_n0_d0_r0.npz' % (R, w)); l1, l2 = joints(A['t2_K4_r0'])
    print('world', w, 'one pass of M: D = %.4f' % (0.5 * (np.exp(l1) + np.exp(l2)) * (l1 - l2) ** 2).sum((1, 2)).mean())
    for tag in ['data_n%d_d%d_r0' % (n, d) for n in (64, 1024, 20480) for d in (0, 1)] + ['synth@1_n20480_d0_r0', 'iis@1_n20480_d0_r0', 'i@1_n20480_d0_r0', 'i@10_n20480_d0_r0']:
        A = np.load('%s/out/w%d/%s.npz' % (R, w, tag)); out = []
        for K in (1, 4):
            l1, l2 = joints(A['t2_K%d_r0' % K]); m1, m2 = joints(A['t2_K%d_r1' % K])
            D = (0.5 * (np.exp(l1) + np.exp(l2)) * (l1 - l2) ** 2).sum((1, 2))
            Dx = (0.25 * (np.exp(l1) + np.exp(l2) + np.exp(m1) + np.exp(m2)) * (l1 - l2) * (m1 - m2)).sum((1, 2))
            out.append('K=%d D %.3f Dx %s' % (K, D.mean(), boot(Dx)))
        print('  %-22s' % tag, ' | '.join(out))
