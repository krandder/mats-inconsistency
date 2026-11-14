"""POST HOC illustration (not pre-registered) of astra R2.5: ARC's basic closure (kprop_harmonic, kind SIMPLE), fed the
EXACT cumulants of the ray front h = alpha_t v, alpha_t = X_+ + t (-X)_+, X ~ N(0,1), one back layer, is biased for
every fixed K. Exact answer: E relu(W h) = relu(W v) E alpha_t. Prints the relative bias |est - exact| / |exact| for
K = 1..4, several t and three back layers W. n = 4, v and W fixed by seed."""
import sys, math
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
from mlp_kprop import kprop_harmonic as KH
from mlp_kprop.wick import WICK_COEF_D
import logging
logging.disable(logging.WARNING)
torch.set_num_threads(1)
DT = torch.float64
n = 4
g = np.random.default_rng(20261031)
v = np.abs(g.standard_normal(n))
v /= np.linalg.norm(v)
Ws = [g.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(3)]


def c_abs(r):
    """E X_+^r = E|X|^r / 2 for X ~ N(0, 1)."""
    return 2 ** (r / 2 - 1) * math.gamma((r + 1) / 2) / math.sqrt(math.pi)


def cumulants(t, R):
    m = [1.0] + [c_abs(r) * (1 + t ** r) for r in range(1, R + 1)]
    k = [0.0] * (R + 1)
    for r in range(1, R + 1):
        k[r] = m[r] - sum(math.comb(r - 1, j - 1) * k[j] * m[r - j] for j in range(1, r))
    return k


def tensor(vec, r):
    T = np.array(1.0)
    for _ in range(r):
        T = np.multiply.outer(T, vec)
    return T


print('relative bias of ARC basic kprop fed exact cumulants of alpha_t v, one back layer (three back matrices W)')
print('t      ' + '  '.join(f'K={K}:W{w}' for K in (1, 2, 3, 4) for w in range(len(Ws))))
for t in (0.25, 0.5, 1.0, 2.0, 4.0):
    row = []
    for K in (1, 2, 3, 4):
        R = K + 1 if K % 2 == 1 else K
        kap = cumulants(t, R)
        K_in = {r: torch.tensor(kap[r] * tensor(v, r), dtype=DT) for r in range(1, R + 1)}
        for W in Ws:
            exact = np.maximum(W @ v, 0) * (c_abs(1) * (1 + t))
            try:
                Kt = KH.coerce_input(dict(K_in), k_max=K, kind=KH.SIMPLE)
                Kt = KH.linear_kprop(Kt, torch.tensor(W, dtype=DT), k_max=K, set_metric=2.0, bias=None)
                Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=K, kind=KH.SIMPLE, use_pK=True,
                                     factor=False)
                est = Kt[1].to_tensor().numpy()
                row.append(f'{np.linalg.norm(est - exact) / np.linalg.norm(exact):.2e}')
            except Exception as e:
                row.append('err:' + type(e).__name__)
    print(f'{t:<6} ' + '  '.join(row), flush=True)
