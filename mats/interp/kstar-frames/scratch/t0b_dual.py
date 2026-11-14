"""T0b (theory, Gaussian surrogate only): multi-step dual transport orth(H_c^{-T} ... H_{c-t+1}^{-T} F_{c-t}) with QR at
each step, against the true frame F_c; plus the same with the frame of a nearby REFERENCE path's past (not needed here).
Same conventions as t0_surrogate.py."""
import math, sys
import numpy as np
sys.path.insert(0, 'scratch')
from t0_surrogate import orth, topk_right, score

def run(m, Lp, k, paths=48, seed=1):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(paths):
        g = lambda: rng.standard_normal((m, m)) / math.sqrt(m)
        fut = [g() for _ in range(Lp)]
        past = [g() for _ in range(16)]
        Jc = np.eye(m)
        for H in fut:
            Jc = H @ Jc
        Fc, s = topk_right(Jc, k)
        for t in (1, 2, 4, 8, 16):
            Jt = Jc.copy()
            for H in past[:t]:
                Jt = Jt @ H
            G, _ = topk_right(Jt, k)
            for H in reversed(past[:t]):
                G = orth(np.linalg.solve(H.T, G))
            acc.setdefault(f'dual_{t}', []).append(score(G, Fc, Jc, s, k))
    return {key: np.array(v).mean(0) for key, v in acc.items()}

for m, Lp, k in ((32, 16, 12), (64, 16, 22), (64, 24, 16)):
    r = run(m, Lp, k)
    print(f"m={m} L'={Lp} k={k}: " + "  ".join(f"{key} O={v[0]:.3f} CS={v[1]:.4f}" for key, v in r.items()))
    sys.stdout.flush()
