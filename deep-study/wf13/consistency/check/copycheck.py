"""copycheck.py WORLD: in the restricted-label test (2e), is the 'repaired' forecast a copy of one pass of M (an input of the head)?
Per size bin (0-9, 10-19, 20-29 revealed), K = 4: mean |forecast - one pass|, mean |forecast - exact|, mean |one pass - exact|,
and the share of the one-pass-to-exact gap that the forecast closes (1 = exact, 0 = one pass; negative = moved away)."""
import sys, glob, os, numpy as np, torch
w = sys.argv[1]; R = '/home/agent/workspaces/mats/work/deep-study/wf13/consistency'
T1 = torch.load('%s/exactwork/w%s/t1.pt' % (R, w)); ps = np.load('%s/check/out/myexact_w%s.npz' % (R, w))['t1_ps']
mask = T1['mask'].numpy(); one = T1['p_one'].numpy().astype(np.float64); mk = ~mask; size = mask.sum(1)
sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))
def row(p):
    out = []
    for lo, hi in ((0, 9), (10, 19), (20, 29)):
        s = mk & ((size >= lo) & (size <= hi))[:, None]
        g = (one - ps)[s]; f = (p - ps)[s]
        out.append('to_one %.3f to_exact %.3f closes %+.2f' % (np.abs(p - one)[s].mean(), np.abs(f).mean(), 1 - (f * g).sum() / (g * g).sum()))
    return ' | '.join(out)
print('one pass vs exact, mean abs:', [round(float(np.abs(one - ps)[mk & ((size >= lo) & (size <= hi))[:, None]].mean()), 3) for lo, hi in ((0, 9), (10, 19), (20, 29))])
for d, tags in (('out/w%s_ph25' % w, ['data_n20480_d0_r0', 'i@10_n20480_d0_r0', 'iio@10_n20480_d0_r0', 'iis@1_n20480_d0_r0', 'synth@1_n20480_d0_r0']),
                ('out/w%s' % w, ['data_n20480_d0_r0', 'i@10_n20480_d0_r0', 'synth@1_n0_d0_r0'])):
    for t in tags:
        f = '%s/%s/%s.npz' % (R, d, t)
        if os.path.exists(f):
            print('%-12s %-22s' % (d[4:], t), row(sig(np.load(f)['t1_K4'])))
