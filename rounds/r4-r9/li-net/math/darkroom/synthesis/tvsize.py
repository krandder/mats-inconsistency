"""Noisier-TV test (Burda et al.'s worry): does attention to the noisy TV grow with its size?
Same world and agent as relevance.py; only the TV alphabet K varies.  E=1000 episodes, 16 seeds.
Reports cumulative TV looks at E=1000 (out of 3000) and the last-quarter share of S."""
from multiprocessing import Pool
import numpy as np
import relevance as R

E, SEEDS = 1000, 16
print(f"E={E} x {R.L} looks, {SEEDS} seeds; cumulative looks at the TV by its alphabet size K (mean over seeds)")
with Pool(16) as pool:
    for prior in ["slab", "spike"]:
        for pol in ["goal-thompson", "curiosity"]:
            row = []
            for K in [4, 16, 64, 256]:
                res = pool.map(R.run, [(pol, E, 1000 + s, 1.0, prior, 0.5, 0, K) for s in range(SEEDS)])
                tv = np.mean([lk[:, 2].sum() for lk, _ in res])
                sS = np.mean([lk[3 * E // 4:, 3].sum() / lk[3 * E // 4:].sum() for lk, _ in res])
                row.append(f"K={K}: TV {tv:6.0f} (S last-qtr {sS:.3f})")
            print(f"{prior:5s} {pol:13s} " + " | ".join(row), flush=True)
