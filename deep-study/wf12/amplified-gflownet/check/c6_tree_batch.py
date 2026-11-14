"""Check of claim 3 (simulated soft tree under a rollout policy without full support) and of the finite-batch
floor of claim 5.  The tree is the worker's class (scripts/amps.py) run on a different world and fresh seeds;
the batch part is the checker's own code on the checker's world.  Output: out/c6_tree_batch.out"""
import sys
import numpy as np

L = []
sys.path.insert(0, "/home/agent/workspaces/mats/work/deep-study/wf12/amplified-gflownet/scripts")
import world as ww
from amps import SoftTree

W = ww.make_world(seed=7, sM=1.0, goal=0.5)
R, T = W["R"], W["T"]
QB, rB = ww.noisy_policy(R, 1.5, 2.0, 123)
QS = {k: v.copy() for k, v in QB.items()}
for (a, pre), q in QS.items():
    for ci, c in enumerate(ww.children(pre)):
        x = dict(pre + (c,))
        if x.get(0) == 1 and x.get(1) == 1:
            q[ci] = -np.inf
shown = (ww.traj_law(QS, rB).sum(0) > 0).sum()
for opt in (1, 2):
    KS = [1024, 4096, 16384]
    acc = {K: [] for K in KS}
    for rep in range(4):
        tr = SoftTree(QS, rB, R, option=opt, rng=np.random.default_rng(9100 + rep))
        done = 0
        for K in KS:
            for _ in range(K - done):
                tr.simulate()
            done = K
            acc[K].append(ww.tv(tr.law().sum(0), T))
    L.append(f"worker's tree, world seed 7, selective policy ({shown}/48 shown), score {'read' if opt == 1 else 'never read'} at terminal leaves: "
             + ", ".join(f"K = {K}: TV {np.mean(acc[K]):.4f}" for K in KS) + f"; score reads in last tree {tr.evals}")

sys.path.pop(0)
for m in ("world",):
    sys.modules.pop(m, None)
import cw

Wc = cw.make_world(601)
R, T = Wc["R"], Wc["T"]
P0 = cw.law(*cw.noisy_policy(R, 1.5, 77, lean=2.0))
n = 2 * cw.D
for K in (4, 16):
    for step in (1.0, 0.25):
        ends = {r: [] for r in (1, 6, 24, 48)}
        for s in range(20):
            g = np.random.default_rng(600 + s)
            P = P0.copy()
            for r in range(1, 49):
                w = np.repeat((R / P.sum(0))[None, :], 2, 0).ravel()
                idx = g.choice(n, size=(64, K), p=P.ravel())
                wt = w[idx]
                wn = wt / wt.sum(1, keepdims=True)
                meas = np.zeros(n)
                np.add.at(meas, idx.ravel(), wn.ravel())
                P = (1 - step) * P + step * (meas.reshape(2, cw.D) / 64 + 1e-6)
                P /= P.sum()
                if r in ends:
                    ends[r].append(cw.tv(P.sum(0), T))
        L.append(f"own world 601, policy B (TV {cw.tv(P0.sum(0), T):.3f}), K = {K}, 64 sets per round, step {step}: TV after rounds 1, 6, 24, 48 = "
                 + ", ".join(f"{np.mean(v):.3f}" for v in ends.values()))
open("out/c6_tree_batch.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
