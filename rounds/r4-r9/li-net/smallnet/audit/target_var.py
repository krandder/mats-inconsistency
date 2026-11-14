"""Audit diagnostic (uses the exact posterior; evaluation-only, NOT part of any training):
how noisy is rbtd8's level-0 target compared with the 0/1 board target, if the level-1 forecaster were exact?
target_j = x_j if cell j is among the c1 revealed cells, else P*(x_j | evidence, revealed true cells).
Ratio = E[(target - P*(x_j|e))^2] / E[(x_j - P*(x_j|e))^2]."""
import sys, numpy as np
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/li-net/smallnet")
import data as D
from env.shiphunt import ShipHunt, placements
P = placements(D.CFG).astype(np.float64)
rng = np.random.default_rng(0)
num = np.zeros(4); den = np.zeros(4); numc = np.zeros(4); c1s = []
nb = 300
for i, s in enumerate(D.seeds_for("val", nb)):
    e = ShipHunt(D.CFG, int(s)); full = list(e.probes); occ = e.occ.astype(bool)
    for k, st in enumerate(D.STAGES):
        e.probes = full[:st]; w = e.posterior(); p = w @ P
        den[k] += (p * (1 - p)).sum()                       # E[(x - p)^2] summed over cells
        for _ in range(20):
            order = rng.permutation(25)
            c1 = int(np.sort(rng.choice(np.arange(1, 25), 7, replace=False))[0]); c1s.append(c1)
            S = order[:c1]
            m = (P[:, S] == occ[S]).all(1)
            w2 = w * m; w2 /= w2.sum(); p2 = w2 @ P
            tgt = p2.copy(); tgt[S] = occ[S]
            # expected squared deviation of the target around p, averaging over boards consistent with e is approximated
            # by using the realised board (which is a draw from P*(.|e)); summed over cells
            num[k] += ((tgt - p) ** 2).sum() / 20
            numc[k] += ((occ.astype(float) - p) ** 2).sum() / 20
print("mean c1 (cells revealed at level 1):", round(float(np.mean(c1s)), 2))
for k, st in enumerate(D.STAGES):
    print(f"stage {st} probes: target MSE / 0-1 MSE = {num[k]/numc[k]:.3f}   (0-1 MSE realised {numc[k]/nb/25:.4f}, expected {den[k]/nb/25:.4f})")
print(f"all stages: {num.sum()/numc.sum():.3f}")
