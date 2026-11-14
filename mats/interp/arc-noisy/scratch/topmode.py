import sys, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import truthlib
T = truthlib.load(100, 0.001); d = T['d']; N = T['N']
for l in (8, 16, 24, 31, 32):
    m = d['S1'][l - 1] / N; M2 = d[f'M2_{l}'] / N; C = M2 - np.outer(m, m)
    ev, V = np.linalg.eigh(C); ev = ev[::-1]; V = V[:, ::-1]
    mh = m / np.linalg.norm(m)
    print(f"layer {l}: |m|^2 {m @ m:.1f} tr C {np.trace(C):.2f}  top eig {ev[0]:.2f} ({ev[0]/np.trace(C):.2f} of var)  2nd {ev[1]:.2f}  "
          f"cos(v1, m) {abs(V[:, 0] @ mh):.3f}  cos(v2, m) {abs(V[:, 1] @ mh):.3f}  var along m {mh @ C @ mh:.2f} -> rel. norm sd {np.sqrt(mh @ C @ mh) / np.linalg.norm(m):.3f}")
