"""E11.2 Bayes-invariance record (recomputed from results/e11_pairs.npz). usage: python e11_invariance.py > results/e11_invariance.txt"""
import numpy as np

z = np.load('results/e11_pairs.npz'); K = len(z['kappas'])
dm = max(float(np.abs(z[f'T{k}_mean_bayes'] - z['T0_mean_bayes']).max()) for k in range(K))
dv = max(float(np.abs(z[f'T{k}_var_bayes'] - z['T0_var_bayes']).max()) for k in range(K))
dw = max(float(np.abs(z[f'WB{k}'] - z['WB0']).max()) for k in range(K))
ss = [((z[f'd{k}_o'][:, :, :4] - z[f'd{k}_y'][:, None, :4]) ** 2).sum(2) for k in range(K)]
dss = max(float(np.abs(s - ss[0]).max()) for s in ss)
other = max(float(np.abs(np.delete(z[f'd{k}_o'], 0, 0) - np.delete(z['d0_o'], 0, 0)).max()) * 0 for k in range(K))
same_y = all(np.array_equal(z[f'd{k}_y'], z['d0_y']) for k in range(K))
src = z['src']; rows = np.arange(len(src))
diff_other = 0.0
for k in range(K):
    o, o0 = z[f'd{k}_o'], z['d0_o']
    mask = np.ones(o.shape, bool); mask[rows, src, :4] = False                       # everything except the modified source's labelled readings
    diff_other = max(diff_other, float(np.abs(o[mask] - o0[mask]).max()))
print('kappas', z['kappas'].tolist())
print('max |change in Bayes mean| across kappa: %.2e' % dm)
print('max |change in Bayes variance| across kappa: %.2e' % dv)
print('max |change in Bayes weights W| across kappa: %.2e' % dw)
print('max |change in every source SS| across kappa: %.2e' % dss)
print('labels identical across kappa:', same_y, '| max |change| in all other readings: %.2e' % diff_other)
