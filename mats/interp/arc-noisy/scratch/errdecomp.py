import sys, json, numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import truthlib
T = truthlib.load(100, 0.001)
for L in (8, 32):
    m = T['mean'][L - 1]; v = T['var'][L - 1].mean(); mh = m / np.linalg.norm(m)
    for kf in ('k2', 'k2a', 'k3f'):
        p = f'results/kprop_100_s0.001_L{32 if L == 32 else 8}_{kf}.json'
        try:
            e = np.array(json.load(open(p))['means'][L - 1]) - m
        except Exception:
            continue
        par = (e @ mh) ** 2; tot = e @ e
        # best global rescale of the estimate
        est = e + m; a = (est @ m) / (est @ est); e2 = a * est - m
        print(f"L={L} {kf:4s}: nMSE {tot / 256 / v:.2e}  share along m {par / tot:.2f}  scale ratio est/true along m {1 + (e @ mh) / np.linalg.norm(m):.5f}  nMSE after best rescale {(e2 @ e2) / 256 / v:.2e}")
