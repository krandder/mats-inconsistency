"""E3a: Kelvin's consistency requirement as three candidate objects, measured on amplifiers whose true error is
known.  (i) order against order; (ii) variance over trajectories of the residual between the amplified step
predictions and the score of the terminal demonstration; (iii) agreement with larger compute.
Amplifiers: count (unweighted fractions of fresh rollouts: converges to the policy's own law, a WRONG answer),
snis (self-normalised weights, fresh rollouts per query), shared (one weighted set answers every query).
Output: out/e3a_objects.out/.json."""
import json, time
import numpy as np
from world import A, D, N, STEPS, DESC, DX, children, make_world, noisy_policy, traj_law, vmarg
from queries import Amp, KEYS, QUERIES, cond, match, DXF

t0 = time.time()
W = make_world(seed=1, sM=1.0, goal=0.5)
R, T = W["R"], W["T"]
Q, Qr = noisy_policy(R, 1.5, 2.0, 22)
P = traj_law(Q, Qr)
amp = Amp(Q, Qr, R)
rng = np.random.default_rng(3)
Tx, qx = vmarg(T), vmarg(P.sum(0))
KS = [4, 16, 64, 256, 1024]
REP = 1000
REP3 = 300
clip = lambda p: np.clip(p, 1e-3, 1 - 1e-3)
lines, out = [], {}


def key_with(key, j, v):
    k = list(key); k[j] = v
    return tuple(k)


def fresh(kind, qi, qk, K):
    pw, pu, _ = amp.stats(qi, qk, K, rng)
    return pw if kind == "snis" else pu


def shared_sets(K, reps):
    idx = rng.choice(A * D, size=(reps, K), p=P.ravel())
    return idx, (0.5 * np.tile(R, A) / P.ravel())[idx]


def shared_answer(idx, Wt, i, key, unweighted=False):
    if unweighted:
        Wt = np.ones_like(Wt)
    m = match(key, DXF)[idx]
    num, den = (Wt * m * (DXF[idx, i] == 1)).sum(1), (Wt * m).sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.5)


empty = (-1, -1, -1)
pairs = [(i, j, vi, vj) for i in range(N) for j in range(i + 1, N) for vi in (0, 1) for vj in (0, 1)]
lines.append("columns: K = " + ", ".join(map(str, KS)))
for kind in ("count", "snis", "shared-count", "shared"):
    err, order, slope, gap = [], [], [], []
    for K in KS:
        e, o = [], []
        if kind.startswith("shared"):
            idx, Wt = shared_sets(K, REP)
            ans = lambda i, key: shared_answer(idx, Wt, i, key, kind == "shared-count")
        else:
            ans = lambda i, key: fresh(kind, np.full(REP, i), [key] * REP, K)
        for (i, key) in QUERIES:
            e.append((ans(i, key) - cond(Tx, i, key)) ** 2)
        for (i, j, vi, vj) in pairs:     # P(X_i = vi) P(X_j = vj | X_i = vi) against the other order
            def pr(var, val, key):
                p = clip(ans(var, key))
                return np.log(p if val == 1 else 1 - p)
            a = pr(i, vi, empty) + pr(j, vj, key_with(empty, i, vi))
            b = pr(j, vj, empty) + pr(i, vi, key_with(empty, j, vj))
            o.append((a - b) ** 2)
        err.append(float(np.sqrt(np.mean(e)))); order.append(float(np.sqrt(np.mean(o))))
        # (iii) against 16 K: regression slope of (big - small) on small, and mean signed gap, over all queries
        num = den = g = 0.0
        S_, B_ = [], []
        if not kind.startswith("shared") and K <= 256:
            for (i, key) in QUERIES:
                s = fresh(kind, np.full(REP3, i), [key] * REP3, K)
                b = fresh(kind, np.full(REP3, i), [key] * REP3, 16 * K)
                S_.append(s); B_.append(b)
                g += abs(np.mean(b - s))
            S_, B_ = np.concatenate(S_), np.concatenate(B_)
            slope.append(float(np.sum((B_ - S_) * (S_ - S_.mean())) / np.sum((S_ - S_.mean()) ** 2)))
            gap.append(float(g / len(QUERIES)))
        elif not kind.startswith("shared"):
            slope.append(float("nan")); gap.append(float("nan"))
    lines.append(f"\n{kind}: RMS error to T      " + "".join(f"{v:9.4f}" for v in err))
    lines.append(f"{kind}: (i) order mismatch  " + "".join(f"{v:9.4f}" for v in order))
    if slope:
        lines.append(f"{kind}: (iii) pooled slope  " + "".join(f"{v:9.4f}" for v in slope))
        lines.append(f"{kind}: (iii) mean |gap|    " + "".join(f"{v:9.4f}" for v in gap))
    out[kind] = dict(err=err, order=order, slope=slope, gap=gap)

# (ii) step-level amplified policy at each augmented state, and its residual against the terminal score
Pf = P.ravel()
for kind in ("count", "snis"):
    var, tvs = [], []
    for K in KS:
        M = 600
        tr = rng.choice(A * D, size=M, p=Pf)             # reference trajectories from the rollout policy
        rho = np.zeros(M)
        for m, t in enumerate(tr):
            a, d = t // D, t % D
            # alien step, then the three demonstration steps, each amplified by K fresh rollouts from that state
            idx = rng.choice(A * D, size=K, p=Pf)
            w = 0.5 * R[idx % D] / Pf[idx] if kind == "snis" else np.ones(K)
            rho[m] += np.log((w[(idx // D) == a].sum() + 1e-3 * w.sum() / A) / (w.sum() * (1 + 1e-3)))
            for L in range(N):
                pre = STEPS[d][:L]
                ds = DESC[pre]
                pc = P[a, ds] / P[a, ds].sum()
                j = rng.choice(len(ds), size=K, p=pc)
                w = 0.5 * R[ds][j] / pc[j] if kind == "snis" else np.ones(K)
                hit = np.array([STEPS[ds[x]][L] == STEPS[d][L] for x in j])
                nch = len(children(pre))
                rho[m] += np.log((w[hit].sum() + 1e-3 * w.sum() / nch) / (w.sum() * (1 + 1e-3)))
            rho[m] -= np.log(0.5 * R[d])
        var.append(float(np.var(rho)))
    lines.append(f"\n{kind}: (ii) variance of the trajectory residual  " + "".join(f"{v:9.4f}" for v in var))
    out[kind]["tb_var"] = var
# the base policy's own residual variance (K -> infinity limit of 'count'), and log Z from the mean residual
rho0 = np.log(Pf) - np.log(0.5 * np.tile(R, A))
v0 = float(np.sum(Pf * rho0 ** 2) - np.sum(Pf * rho0) ** 2)
lines.append(f"rollout policy itself: residual variance {v0:.4f}; -mean residual {-np.sum(Pf * rho0):.4f}; log Z {np.log(W['Z']):.4f}")
# readings in the limit of compute: T conditional on the demonstration's order against T's marginal over orders
Tm = T.reshape(6, 8)
spread = np.abs(Tm / Tm.sum(1, keepdims=True) - Tx[None, :]).max()
lines.append(f"exact T: max over orders and values of |T(x | order) - T(x)| = {spread:.4f} (0 iff the score factorises)")
out["T_order_spread"] = float(spread)
lines.append(f"runtime {time.time() - t0:.0f}s")
open("../out/e3a_objects.out", "w").write("\n".join(lines) + "\n")
json.dump(out, open("../out/e3a_objects.json", "w"), indent=1)
print("\n".join(lines))
