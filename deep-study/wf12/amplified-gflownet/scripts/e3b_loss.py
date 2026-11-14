"""E3b (registered in PREREG.md): a consistency loss on the amplified predictions as a data-efficiency device.
Head f = sigmoid(alpha * u + beta * v) on the weighted (u) and unweighted (v) logit of K fresh masked rollouts.
Output: out/e3b_loss.out / .json."""
import json, time
import numpy as np
from scipy.optimize import minimize
from world import N, XARR, PERMS, make_world, noisy_policy, vmarg
from queries import Amp, KEYS, cond, sig

t0 = time.time()
U, NTEST, NREAL, LAMS = 2000, 4000, [8, 32, 128, 512], [1.0, 10.0]
REG = {"R1 score right": 0.0, "R2 score wrong": 1.0}
EMPTY = (-1,) * N


def draw_queries(X, rng):
    """For value vectors X (n, 3): a random variable i and the values of a random subset of the others."""
    n = len(X)
    qi = rng.integers(0, N, n)
    inc = rng.random((n, N)) < 0.5
    keys = [tuple(int(X[m, j]) if (inc[m, j] and j != qi[m]) else -1 for j in range(N)) for m in range(n)]
    return qi, keys, X[np.arange(n), qi]


def feats(amp, qi, keys, K, rng):
    pw, pu, _ = amp.stats(qi, keys, K, rng)
    return amp.lg(pw), amp.lg(pu), pw


def logf(th, u, v, val):
    z = th[0] * u + th[1] * v
    return -np.logaddexp(0, np.where(val == 1, -z, z))


def build(amp, K, rng):
    """Unlabeled material for the three losses (no real outcome is used)."""
    Xu = XARR[rng.integers(0, 8, U)]
    # (i) two variables, both orders
    i = rng.integers(0, N, U); j = (i + 1 + rng.integers(0, N - 1, U)) % N
    vi, vj = Xu[np.arange(U), i], Xu[np.arange(U), j]
    ki = [tuple(int(vi[m]) if c == i[m] else -1 for c in range(N)) for m in range(U)]
    kj = [tuple(int(vj[m]) if c == j[m] else -1 for c in range(N)) for m in range(U)]
    o = [feats(amp, i, [EMPTY] * U, K, rng)[:2] + (vi,), feats(amp, j, ki, K, rng)[:2] + (vj,),
         feats(amp, j, [EMPTY] * U, K, rng)[:2] + (vj,), feats(amp, i, kj, K, rng)[:2] + (vi,)]
    # (ii) a random order of all three, against the estimated score of the values
    perm = np.array(PERMS)[rng.integers(0, 6, U)]
    chain = []
    for t in range(N):
        keys = [tuple(int(Xu[m, c]) if c in perm[m, :t] else -1 for c in range(N)) for m in range(U)]
        chain.append(feats(amp, perm[:, t], keys, K, rng)[:2] + (Xu[np.arange(U), perm[:, t]],))
    logS = np.log(amp.score_of_values([tuple(int(c) for c in x) for x in Xu], 64, rng))
    # (iii) the same query at 64 K
    qi, keys, _ = draw_queries(Xu, rng)
    small = feats(amp, qi, keys, K, rng)[:2]
    big = np.clip(feats(amp, qi, keys, 64 * K, rng)[2], 0.02, 0.98)
    return o, (chain, logS), (small, big)


def losses(th, mat, c):
    o, (chain, logS), (small, big) = mat
    if c == "i":
        a = logf(th, *o[0]) + logf(th, *o[1]); b = logf(th, *o[2]) + logf(th, *o[3])
        return np.mean((a - b) ** 2)
    if c == "ii":
        return np.var(sum(logf(th, *s) for s in chain) - logS)
    return -np.mean(big * logf(th, *small, 1) + (1 - big) * logf(th, *small, 0))


def fit(obj):
    r = minimize(obj, np.array([1.0, 0.0]), method="Nelder-Mead", options=dict(maxiter=300, xatol=1e-3, fatol=1e-6))
    return r.x


res = {}
for reg, sM in REG.items():
    for wseed in (1, 2, 3):
        Wd = make_world(seed=wseed, sM=sM, goal=0.0)
        Q, Qr = noisy_policy(Wd["R"], 1.5, 2.0, 22)
        amp = Amp(Q, Qr, Wd["R"])
        Tx, px = vmarg(Wd["T"]), Wd["pstar"]
        for K in (2, 8):
            rt = np.random.default_rng(999 + wseed)
            Xt = XARR[rt.choice(8, NTEST, p=px)]
            qi, keys, _ = draw_queries(Xt, rt)
            ut, vt, _ = feats(amp, qi, keys, K, rt)
            pt = np.array([cond(px, i, k) for i, k in zip(qi, keys)])
            Tt = np.array([cond(Tx, i, k) for i, k in zip(qi, keys)])

            def evaluate(th):
                f = np.clip(sig(th[0] * ut + th[1] * vt), 1e-9, 1 - 1e-9)
                ex = pt * np.log(pt / f) + (1 - pt) * np.log((1 - pt) / (1 - f))
                return float(np.mean(ex)), float(np.mean(np.abs(f - Tt)))

            cell = res.setdefault(f"{reg}|K={K}", {})
            cell.setdefault("raw", []).append(evaluate([1.0, 0.0]))
            cell.setdefault("T itself", []).append(float(np.mean(pt * np.log(pt / Tt) + (1 - pt) * np.log((1 - pt) / (1 - Tt)))))
            for seed in range(20):
                rng = np.random.default_rng(10000 * wseed + 100 * K + seed)
                mat = build(amp, K, rng)
                Xr = XARR[rng.choice(8, max(NREAL), p=px)]
                ri, rk, ry = draw_queries(Xr, rng)
                ru, rv, _ = feats(amp, ri, rk, K, rng)
                for c in ("i", "ii", "iii"):
                    th = fit(lambda th: losses(th, mat, c))
                    cell.setdefault(f"alone {c}", []).append(evaluate(th) + tuple(th))
                for n in NREAL:
                    data = lambda th: -np.mean(logf(th, ru[:n], rv[:n], ry[:n]))
                    cell.setdefault(f"n={n} data", []).append(evaluate(fit(data)))
                    for lam in LAMS:
                        for c in ("i", "ii", "iii"):
                            th = fit(lambda th: data(th) + lam * losses(th, mat, c))
                            cell.setdefault(f"n={n} data+{c} lam={lam:g}", []).append(evaluate(th))
        print(reg, wseed, f"{time.time() - t0:.0f}s", flush=True)

lines = []
for cellname, cell in res.items():
    lines.append(f"\n== {cellname}  (excess log loss vs reality, nats | mean |f - T| ; mean over 3 worlds x 20 seeds)")
    lines.append(f"T's own conditional: excess {np.mean(cell['T itself']):.4f}")
    for arm, v in cell.items():
        if arm == "T itself":
            continue
        v = np.array(v)
        extra = f"   alpha {v[:, 2].mean():+.2f} beta {v[:, 3].mean():+.2f}" if arm.startswith("alone") else ""
        ref = None
        if "data+" in arm:
            ref = np.array(cell[arm.split(" data+")[0] + " data"])[:, 0]
            extra += f"   vs data only {100 * (v[:, 0].mean() / ref.mean() - 1):+.0f}%  better in {100 * np.mean(v[:, 0] < ref):.0f}% of runs"
        lines.append(f"{arm:26s} {v[:, 0].mean():.4f} | {v[:, 1].mean():.4f}{extra}")
lines.append(f"\nruntime {time.time() - t0:.0f}s")
open("../out/e3b_loss.out", "w").write("\n".join(lines) + "\n")
json.dump({k: {a: np.array(v).tolist() for a, v in c.items()} for k, c in res.items()}, open("../out/e3b_loss.json", "w"))
print("\n".join(lines))
