"""POST HOC check (not pre-registered): ARC's kprop K=2 fed the cut mean with ZERO covariance reduces to mean
propagation g_b(mu) (FORMAL-ext C2 applied to ARC's own closure). 8 fronts x 8 backs of (n, c, d) = (8, 64, 8)."""
import sys, math
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/cut-rank/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
import crnet as cn
from mlp_kprop import kprop_harmonic as KH
from mlp_kprop.wick import WICK_COEF_D
import logging; logging.disable(logging.WARNING)
torch.set_num_threads(1)
n, c, d = 8, 64, 8
tag = cn.cfg_tag(n, c, d); Er = cn.mean_radius(n)
worst = 0.0
for f in range(8):
    Wf = [W.astype(np.float64) for W in cn.front_weights(tag, n, c, f)]
    H = cn.directions(n, 4096, tag, 77, f).astype(np.float64)
    for W in Wf:
        H = np.maximum(H @ W.T, 0)
    mu = Er * H.mean(0)
    mu = mu / (np.linalg.norm(mu) + 1e-300)   # unit scale: ARC's code floors variances at 1e-10 in absolute terms
    if np.all(mu == 0):
        continue
    for b in range(8):
        Wb = [torch.tensor(W.astype(np.float64)) for W in cn.back_weights(tag, n, d, b)]
        K = KH.coerce_input({1: torch.tensor(mu), 2: torch.zeros(n, n, dtype=torch.float64)}, k_max=2, kind=KH.SIMPLE)
        m = torch.tensor(mu)
        for W in Wb:
            K = KH.linear_kprop(K, W, k_max=2, set_metric=2.0, bias=None)
            K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=2, kind=KH.SIMPLE, use_pK=True, factor=False)
            m = torch.clamp(W @ m, min=0)
        arc = K[1].to_tensor()
        rel = float(torch.linalg.norm(arc - m) / (torch.linalg.norm(torch.tensor(mu)) + 1e-300))
        if False: print(f"f{f} b{b} |mu| {np.linalg.norm(mu):.3g} |g(mu)| {float(torch.linalg.norm(m)):.3g} |arc| {float(torch.linalg.norm(arc)):.3g} rel {rel:.2e}")
        worst = max(worst, rel)
print(f'max |ARC2(mu, 0) - g_b(mu)| / |mu| (mu scaled to unit norm) over 64 (front, back) pairs: {worst:.2e}')
