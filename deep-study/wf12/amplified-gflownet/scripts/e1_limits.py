"""E1: what each amplification returns as compute K grows, for several rollout policies.
Output: out/e1_limits.out (table) and out/e1_limits.json.  All distances are total variation to T over the
48 demonstrations (exact arrays; Monte Carlo only over the amplifier's own randomness)."""
import json, sys, time
import numpy as np
from world import A, D, STEPS, PREFIXES, DX, children, make_world, noisy_policy, traj_law, tv, vmarg
from amps import weights, snis_sets, SoftTree, mh_kernel

t0 = time.time()
W = make_world(seed=1, sM=1.0, goal=0.5)          # soft, order-inconsistent M and a goal factor
R, T = W["R"], W["T"]
KS = [1, 4, 16, 64, 256, 1024]
rng = np.random.default_rng(11)


def selective(Q):
    """Never shows a demonstration with x0 = 1 and x1 = 1 (zero support there)."""
    Q = {k: v.copy() for k, v in Q.items()}
    for (a, pre), q in Q.items():
        for ci, c in enumerate(children(pre)):
            x = dict(pre + (c,))
            if x.get(0) == 1 and x.get(1) == 1:
                q[ci] = -np.inf
    return Q


pols = {}
QA, rA = noisy_policy(R, 0.5, 0.0, 21); pols["A mild"] = (QA, rA)
QB, rB = noisy_policy(R, 1.5, 2.0, 22); pols["B strong+alien"] = (QB, rB)
pols["S selective"] = (selective(QB), rB)
out, lines = {}, []
hid = (DX[:, 0] == 1) & (DX[:, 1] == 1)
lines.append(f"world: Z={W['Z']:.4f}  T-mass hidden by S: {T[hid].sum():.4f}")
for name, (Q, Qr) in pols.items():
    P = traj_law(Q, Qr)
    q = P.sum(0)
    res = {"policy_tv": tv(q, T)}
    lines.append(f"\n== rollout policy {name}: TV(policy, T) = {tv(q, T):.4f}; support = {(q > 0).sum()}/48")
    lines.append("K            " + "".join(f"{k:>9d}" for k in KS))
    # counting and SNIS (bias of the expected weighted measure; mean TV of one set)
    for mode in ("uniform", "marg"):
        w = weights(P, R, mode)
        bias, noise, cnt_tv = [], [], []
        for K in KS:
            reps = 4000 if K <= 256 else 1000
            meas, cnt, ess = snis_sets(P, w, K, reps, rng)
            md = meas.sum(1)
            bias.append(tv(md.mean(0), T)); noise.append(np.mean(0.5 * np.abs(md - T).sum(1)))
            cnt_tv.append(tv(cnt.sum(1).mean(0), T))
        if mode == "uniform":
            lines.append("count: E-law " + "".join(f"{v:9.4f}" for v in cnt_tv)); res["count"] = cnt_tv
        lines.append(f"snis-{mode[:4]} E-law" + "".join(f"{v:9.4f}" for v in bias))
        lines.append(f"snis-{mode[:4]} 1set " + "".join(f"{v:9.4f}" for v in noise))
        res[f"snis_{mode}_bias"] = bias; res[f"snis_{mode}_oneset"] = noise
    # soft tree, options 1 and 2
    for opt in (1, 2):
        acc = {K: [] for K in KS}
        for rep in range(40):
            tr = SoftTree(Q, Qr, R, option=opt, rng=np.random.default_rng(1000 + rep))
            done = 0
            for K in KS:
                for _ in range(K - done):
                    tr.simulate()
                done = K
                acc[K].append(tv(tr.law().sum(0), T))
        row = [float(np.mean(acc[K])) for K in KS]
        lines.append(f"tree opt{opt}   " + "".join(f"{v:9.4f}" for v in row)); res[f"tree_opt{opt}"] = row
    if opt == 2:
        # exact limit of option 2: full expansion, terminal edges keep the policy's Q
        Qe = {k: v.copy() for k, v in Q.items()}
        from world import lse
        for a in range(A):
            for pre in reversed(PREFIXES):
                if len(pre) < 2:
                    for ci, c in enumerate(children(pre)):
                        Qe[(a, pre)][ci] = lse(Qe[(a, pre + (c,))])
        lim2 = traj_law(Qe, np.array([lse(Qe[(a, ())]) for a in range(A)])).sum(0)
        lines.append(f"tree opt2 exact limit (terminal edges never read the score): TV = {tv(lim2, T):.4f}")
        res["tree_opt2_limit"] = tv(lim2, T)
    # back-and-forth refinement: exact law after K moves started from one rollout
    for rule in ("mh", "printed", "greedy"):
        Km = mh_kernel(P, R, rule)
        row = [tv((P.ravel() @ np.linalg.matrix_power(Km, K)).reshape(A, D).sum(0), T) for K in KS]
        lim = (P.ravel() @ np.linalg.matrix_power(Km, 2 ** 20)).reshape(A, D).sum(0)
        lines.append(f"refine {rule:7s}" + "".join(f"{v:9.4f}" for v in row) + f"   limit {tv(lim, T):.4f}")
        res[f"refine_{rule}"] = row; res[f"refine_{rule}_limit"] = tv(lim, T)
    out[name] = res
lines.append(f"\nruntime {time.time() - t0:.0f}s")
open("../out/e1_limits.out", "w").write("\n".join(lines) + "\n")
json.dump(out, open("../out/e1_limits.json", "w"), indent=1)
print("\n".join(lines))
