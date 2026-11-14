"""E2: amplify, then distil the amplified result into the rollout policy, for several rounds.
The policy is tabular on a tree, so it can represent any law over the 96 augmented trajectories exactly:
'distil' = set the policy to the (expected or sampled) amplified law.  Output: out/e2_distill.out/.json."""
import json, time
import numpy as np
from world import A, D, PREFIXES, children, make_world, noisy_policy, traj_law, tv, kl, lse
from amps import weights, snis_sets, SoftTree, mh_kernel

t0 = time.time()
W = make_world(seed=1, sM=1.0, goal=0.5)
R, T = W["R"], W["T"]
rng = np.random.default_rng(5)
ROUNDS = 6
lines, out = [], {}


def mi(P):
    P = P / P.sum()
    pa, pd = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    m = P > 0
    return float((P[m] * np.log(P[m] / (pa @ pd)[m])).sum())


def mu_exact2(P, w):
    """Expected self-normalised measure of K = 2 rollouts, exactly (all 96 x 96 pairs)."""
    p, ww = P.ravel(), w.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ww[:, None] + ww[None, :] > 0, ww[:, None] / (ww[:, None] + ww[None, :]), 0.0)
    return (2 * p * (frac @ p)).reshape(A, D)


def mu_mc(P, w, K, reps):
    meas, _, _ = snis_sets(P, w, K, reps, rng)
    return meas.mean(0)


def fmt(v):
    return "".join(f"{x:9.4f}" for x in v)


for pname, (eps, bias, seed) in {"A mild": (0.5, 0.0, 21), "B strong+alien": (1.5, 2.0, 22)}.items():
    Q0, r0 = noisy_policy(R, eps, bias, seed)
    P0 = traj_law(Q0, r0)
    lines.append(f"\n== start policy {pname}: TV = {tv(P0.sum(0), T):.4f}, I(alien; demonstration) = {mi(P0):.4f} nats")
    lines.append("TV(policy, T) after round   " + "".join(f"{r:9d}" for r in range(ROUNDS + 1)))
    res = {}
    # (1) self-normalised weighting, exact distillation of the expected amplified law
    for mode in ("uniform", "marg"):
        for K in (1, 2, 4, 16):
            P, row = P0.copy(), [tv(P0.sum(0), T)]
            for r in range(ROUNDS):
                w = weights(P, R, mode)
                P = mu_exact2(P, w) if K == 2 else mu_mc(P, w, K, 20000)
                P = P / P.sum()
                row.append(tv(P.sum(0), T))
            lines.append(f"snis-{mode[:4]} K={K:<3d} expected  {fmt(row)}   I(a;d) end {mi(P):.4f}")
            res[f"snis_{mode}_K{K}"] = row
    w = weights(P0, R, "marg")
    chk = tv(mu_exact2(P0, w).sum(0), mu_mc(P0, w, 2, 200000).sum(0))
    lines.append(f"check: exact K=2 map against 200,000 Monte Carlo sets, TV = {chk:.4f}")
    # (1b) the same with a finite batch: 64 sets of K rollouts per round, 20 seeds
    for K in (1, 4, 16):
        rows = []
        for s in range(20):
            g = np.random.default_rng(100 + s)
            P, row = P0.copy(), [tv(P0.sum(0), T)]
            for r in range(ROUNDS):
                meas, _, _ = snis_sets(P, weights(P, R, "marg"), K, 64, g)
                P = meas.mean(0) + 1e-6
                P = P / P.sum()
                row.append(tv(P.sum(0), T))
            rows.append(row)
        lines.append(f"snis-marg K={K:<3d} batch 64  {fmt(np.mean(rows, 0))}")
        res[f"snis_marg_K{K}_batch64"] = list(np.mean(rows, 0))
    # (2) soft tree: regress the policy's Q on the tree's Q at the nodes the tree expanded (8 trees per round)
    for opt, K in ((1, 16), (1, 64), (1, 256), (2, 64)):
        rows, evs = [], []
        for s in range(5):
            Q = {k: v.copy() for k, v in Q0.items()}; Qr = r0.copy()
            row, ev = [tv(P0.sum(0), T)], 0
            for r in range(ROUNDS):
                acc, accr = {}, []
                for b in range(8):
                    tr = SoftTree(Q, Qr, R, option=opt, rng=np.random.default_rng(7000 + 97 * s + 13 * r + b))
                    for _ in range(K):
                        tr.simulate()
                    ev += tr.evals
                    for k, v in tr.Qt.items():
                        (accr.append(v) if k == "root" else acc.setdefault(k, []).append(v))
                    if opt == 2:   # the paper's terminal loss: one sampled trajectory per tree reads the score
                        L = tr.law().ravel()
                        t = tr.rng.choice(A * D, p=L / L.sum())
                        a, d = t // D, t % D
                        from world import STEPS
                        pre = STEPS[d][:2]
                        ci = children(pre).index(STEPS[d][2])
                        acc.setdefault((a, pre), [Q[(a, pre)].copy()])
                        for arr in acc[(a, pre)]:
                            arr[ci] = np.log(R[d]) + np.log(0.5)
                        ev += 1
                for k, v in acc.items():
                    Q[k] = np.mean(v, 0)
                Qr = np.mean(accr, 0)
                row.append(tv(traj_law(Q, Qr).sum(0), T))
            rows.append(row); evs.append(ev / ROUNDS)
        lines.append(f"tree opt{opt} K={K:<4d} 8 trees  {fmt(np.mean(rows, 0))}   score reads/round {np.mean(evs):.0f}")
        res[f"tree_opt{opt}_K{K}"] = list(np.mean(rows, 0))
    # (3) back-and-forth refinement with the Metropolis-Hastings filter, exact
    for K in (1, 4, 16):
        P, row = P0.copy(), [tv(P0.sum(0), T)]
        for r in range(ROUNDS):
            P = (P.ravel() @ np.linalg.matrix_power(mh_kernel(P, R, "mh"), K)).reshape(A, D)
            row.append(tv(P.sum(0), T))
        lines.append(f"refine-mh K={K:<3d} exact     {fmt(row)}   I(a;d) end {mi(P):.4f}")
        res[f"refine_mh_K{K}"] = row
    out[pname] = res
lines.append(f"\nruntime {time.time() - t0:.0f}s")
open("../out/e2_distill.out", "w").write("\n".join(lines) + "\n")
json.dump(out, open("../out/e2_distill.json", "w"), indent=1)
print("\n".join(lines))
