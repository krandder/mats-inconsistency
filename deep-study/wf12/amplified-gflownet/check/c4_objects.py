"""Check of claims 7 and 8 (note section 3, E3a): own estimators, fresh world (401) and seeds.
(i) order against order from one shared set (weighted and unweighted) and from fresh sets;
(ii) variance over trajectories of [sum of log amplified step predictions] - [log score of the terminal demonstration];
the order-by-order reading of the target.  Output: out/c4_objects.out"""
import itertools
import numpy as np
from cw import *

L = []
rng = np.random.default_rng(4242)
DXF = np.tile(DXV, (2, 1))
for ws in (401, 402):
    W = make_world(ws)
    R, T = W["R"], W["T"]
    P = law(*noisy_policy(R, 1.5, 50 + ws, lean=2.0))
    Pf, Rf = P.ravel(), np.tile(R, 2)
    Tx = vmarg(T)
    L.append(f"\n== world {ws}; rollout policy B: TV to T {tv(P.sum(0), T):.3f}")
    # (i) from ONE shared set: P(Xi=vi) P(Xj=vj | Xi=vi) against the other order; error of the answers to T
    for K in (64, 1024, 16384):
        for weighted in (True, False):
            mism, err, cnt = [], [], 0
            for rep in range(200):
                idx = rng.choice(2 * D, size=K, p=Pf)
                wt = Rf[idx] / Pf[idx] if weighted else np.ones(K)
                X = DXF[idx]
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    for vi, vj in itertools.product((0, 1), repeat=2):
                        mi_, mj_ = X[:, i] == vi, X[:, j] == vj
                        if (wt * mi_ * mj_).sum() == 0:
                            continue
                        a = np.log((wt * mi_).sum() / wt.sum()) + np.log((wt * mi_ * mj_).sum() / (wt * mi_).sum())
                        b = np.log((wt * mj_).sum() / wt.sum()) + np.log((wt * mi_ * mj_).sum() / (wt * mj_).sum())
                        mism.append((a - b) ** 2)
                        tj = Tx[(XA[:, i] == vi) & (XA[:, j] == vj)].sum()
                        err.append((np.exp(a) - tj) ** 2)
            L.append(f"(i) one shared set, K = {K:5d}, {'weighted  ' if weighted else 'unweighted'}: RMS order mismatch "
                     f"{np.sqrt(np.mean(mism)):.2e}; RMS error of the joint answer to T {np.sqrt(np.mean(err)):.4f}")
    # (ii) residual variance; reference trajectories from the rollout policy; K fresh rollouts per step
    node_groups = {}
    for kind in ("weighted", "unweighted"):
        row = []
        for K in (4, 16, 64, 256, 1024):
            M = 500
            rho = np.zeros(M)
            ref = rng.choice(2 * D, size=M, p=Pf)
            for m, t in enumerate(ref):
                a, d = t // D, t % D
                for step in range(4):                      # alien step, then three demonstration steps
                    if step == 0:
                        grp = np.arange(2 * D)
                        hit_of = lambda u: (u // D) == a
                    else:
                        pre = PATH[d][:step - 1]
                        grp = a * D + UNDER[pre] if pre in UNDER else None
                        nxt = PATH[d][step - 1]
                        hit_of = lambda u: np.array([PATH[x % D][step - 1] == nxt for x in u])
                    pc = Pf[grp] / Pf[grp].sum()
                    u = grp[rng.choice(len(grp), size=K, p=pc)]
                    wt = Rf[u] / Pf[u] if kind == "weighted" else np.ones(K)
                    nch = 2 if step == 0 else 2 * (N - (step - 1))
                    share = (wt[hit_of(u)].sum() + 1e-3 * wt.sum() / nch) / (wt.sum() * (1 + 1e-3))
                    rho[m] += np.log(share)
                rho[m] -= np.log(0.5 * R[d])
            row.append(np.var(rho))
        L.append(f"(ii) residual variance, {kind:10s} steps, K = 4, 16, 64, 256, 1024: " + ", ".join(f"{v:.3f}" for v in row))
    r0 = np.log(Pf) - np.log(0.5 * Rf)
    L.append(f"     the rollout policy's own residual variance: {(Pf * r0 ** 2).sum() - (Pf * r0).sum() ** 2:.3f}")
    # the two readings of the target: T(values | order of the demonstration) against T(values)
    Tm = np.zeros((6, 8))
    np.add.at(Tm, (ORD, VAL), T)
    L.append(f"target read order by order: max |T(values | order) - T(values)| = "
             f"{np.abs(Tm / Tm.sum(1, keepdims=True) - Tx[None, :]).max():.3f}; T(order) ranges "
             f"{Tm.sum(1).min():.3f} to {Tm.sum(1).max():.3f}")
open("out/c4_objects.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
