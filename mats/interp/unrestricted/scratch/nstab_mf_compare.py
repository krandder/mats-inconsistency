"""Post hoc: compare registered M1 noise-stability measurements with astra's mean-field formula (round 2, 3(a)):
S_L(rho) ~ E_T K_{rho,L}(T) - 1/pi, K_{rho,0}(t) = t, K_{rho,l+1} = kappa(rho K_{rho,l}), T = U.V on S^{n-1}."""
import math, json, numpy as np
def kap(t): return (np.sqrt(np.clip(1 - t * t, 0, None)) + (math.pi - np.arccos(np.clip(t, -1, 1))) * t) / math.pi
rng = np.random.default_rng(0)
for n in (16, 32):
    U = rng.standard_normal((200000, n)); U /= np.linalg.norm(U, axis=1, keepdims=True); T = U[:, 0]
    for L in (4, 16, 32):
        def S(rho):
            K = T.copy()
            for _ in range(L): K = kap(rho * K)
            return K.mean() - 1 / math.pi
        S1 = S(1.0)
        r = json.load(open(f'runs/nstab_n{n}_L{L}_K48.json'))
        meas = {x['delta']: x['D'] for x in r['rows']}; D1 = meas[1.0]
        out = [f"{d:g}: mf {1 - S(1 - d) / S1:.2g} meas {meas[d]:.2g} norm {meas[d] / D1:.2g}" for d in (1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1)]
        print(f"n={n} L={L} D(1)={D1:.2f}: " + '; '.join(out))
