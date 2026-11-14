"""E1b (post hoc, measurements only): (a) the soft tree that reads the score at terminal leaves, under the
selective rollout policy, at larger compute; (b) distillation of self-normalised sets with a finite batch,
replacing the policy each round against moving it a quarter of the way.  Output: out/e1b_extra.out."""
import numpy as np
from world import A, D, DX, children, make_world, noisy_policy, traj_law, tv
from amps import weights, snis_sets, SoftTree

W = make_world(seed=1, sM=1.0, goal=0.5)
R, T = W["R"], W["T"]
lines = []
QB, rB = noisy_policy(R, 1.5, 2.0, 22)
QS = {k: v.copy() for k, v in QB.items()}
for (a, pre), q in QS.items():
    for ci, c in enumerate(children(pre)):
        x = dict(pre + (c,))
        if x.get(0) == 1 and x.get(1) == 1:
            q[ci] = -np.inf
KS = [1024, 4096, 16384]
acc = {K: [] for K in KS}
for rep in range(5):
    tr = SoftTree(QS, rB, R, option=1, rng=np.random.default_rng(300 + rep))
    done = 0
    for K in KS:
        for _ in range(K - done):
            tr.simulate()
        done = K
        acc[K].append(tv(tr.law().sum(0), T))
lines.append("(a) soft tree, score read at terminal leaves, selective policy S (36 of 48 demonstrations shown):")
lines.append("    TV to T at K = " + ", ".join(f"{K}: {np.mean(acc[K]):.4f}" for K in KS)
             + f"   (score reads in the last tree: {tr.evals}; nodes expanded: {len(tr.Qt)})")

P0 = traj_law(QB, rB)
for K in (4, 16):
    for step, rounds in ((1.0, 24), (0.25, 24)):
        rows = []
        for s in range(20):
            g = np.random.default_rng(500 + s)
            P, row = P0.copy(), []
            for r in range(rounds):
                meas, _, _ = snis_sets(P, weights(P, R, "marg"), K, 64, g)
                P = (1 - step) * P + step * (meas.mean(0) + 1e-6)
                P = P / P.sum()
                row.append(tv(P.sum(0), T))
            rows.append(row)
        m = np.mean(rows, 0)
        lines.append(f"(b) policy B (TV 0.4800), K = {K}, 64 sets per round, step {step}: TV after rounds 1, 3, 6, 12, 24 = "
                     + ", ".join(f"{m[i]:.4f}" for i in (0, 2, 5, 11, 23)))
open("../out/e1b_extra.out", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
