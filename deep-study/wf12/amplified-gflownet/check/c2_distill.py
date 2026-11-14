"""Check of claims 5, 6, 8: distillation rate (exact K = 2 and K = 3 maps), K = 1 drift, alien coordinate,
refinement monotonicity when the kernel is built from the policy itself, and the residual-variance loss when
the alien bit is informative.  Own world, fresh seeds.  Output: out/c2_distill.out"""
import numpy as np
from cw import *

L = []
rng = np.random.default_rng(777)
n = 2 * D
FULL = [(t // D,) + PATH[t % D] for t in range(n)]
G = [np.array([[FULL[u][:4 - j] == FULL[t][:4 - j] for u in range(n)] for t in range(n)]) for j in (1, 2, 3, 4)]


def wts(P, R, mode):
    if mode == "unif":
        return 0.5 * R[None, :] / P
    return np.repeat((R / P.sum(0))[None, :], 2, 0)


def map2(P, w):
    p, ww = P.ravel(), w.ravel()
    return (2 * p * ((ww[:, None] / (ww[:, None] + ww[None, :])) @ p)).reshape(2, D)


def map3(P, w):
    p, ww = P.ravel(), w.ravel()
    S = ww[:, None] + ww[None, :]                         # (j, k)
    inner = np.array([(p[:, None] * p[None, :] * (wi / (wi + S))).sum() for wi in ww])
    return (3 * p * inner).reshape(2, D)


def mh(P, R):
    Pf, Rf = P.ravel(), np.tile(R, 2)
    w = 0.5 * Rf / Pf
    K = np.zeros((n, n))
    for Gj in G:
        q = Gj * Pf[None, :]
        q = q / q.sum(1, keepdims=True)
        K += 0.25 * q * np.minimum(1.0, w[None, :] / w[:, None])
    K[np.arange(n), np.arange(n)] = 0
    K[np.arange(n), np.arange(n)] = 1 - K.sum(1)
    return K


for ws in (201, 202, 203):
    W = make_world(ws)
    R, T = W["R"], W["T"]
    L.append(f"\n== world {ws}")
    for pname, (eps, lean) in {"A": (0.5, 0.0), "B": (1.5, 2.0)}.items():
        P0 = law(*noisy_policy(R, eps, 31 + ws, lean=lean))
        for mode in ("marg", "unif"):
            for K, mp in ((2, map2), (3, map3)):
                P, row = P0.copy(), [tv(P0.sum(0), T)]
                for r in range(10):
                    P = mp(P, wts(P, R, mode))
                    P /= P.sum()
                    row.append(tv(P.sum(0), T))
                rat = [row[i + 1] / row[i] for i in range(len(row) - 1)]
                L.append(f"policy {pname} {mode} K={K} exact map: TV by round " + " ".join(f"{v:.4f}" for v in row[:8])
                         + f" | ratio first {rat[0]:.3f}, last {rat[-1]:.3f} (1/K = {1 / K:.3f}); I(alien; d) {mi(P0):.3f} -> {mi(P):.4f}")
        # K = 1: expected map is the identity; finite batch of 64 sets drifts
        ends = []
        for s in range(20):
            P = P0.copy()
            for r in range(6):
                idx = rng.choice(n, size=64, p=P.ravel())
                P = (np.bincount(idx, minlength=n) / 64 + 1e-6).reshape(2, D)
                P /= P.sum()
            ends.append(tv(P.sum(0), T))
        L.append(f"policy {pname} K=1, 64 rollouts per round, 6 rounds, 20 seeds: TV {tv(P0.sum(0), T):.3f} -> {np.mean(ends):.3f}")
        # refinement with the kernel built from the policy itself: joint and demonstration-level TV after one move
        P = P0.copy()
        tj, tm = [tv(P.ravel(), np.tile(T, 2) / 2)], [tv(P.sum(0), T)]
        for r in range(6):
            P = (P.ravel() @ mh(P, R)).reshape(2, D)
            tj.append(tv(P.ravel(), np.tile(T, 2) / 2)); tm.append(tv(P.sum(0), T))
        L.append(f"policy {pname} refinement, 1 move per round: TV over demonstrations " + " ".join(f"{v:.4f}" for v in tm))
    # counterexample hunt for 'a refinement round never increases the error' with error = TV over demonstrations
    worst = (0, None)
    for trial in range(3000):
        c = rng.dirichlet([0.2, 0.2], size=D).T
        base = T if trial % 3 else T * np.exp(0.1 * rng.normal(size=D))
        P = base[None, :] * c
        P /= P.sum()
        P = np.maximum(P, 1e-9); P /= P.sum()
        P1 = (P.ravel() @ mh(P, R)).reshape(2, D)
        d = tv(P1.sum(0), T) - tv(P.sum(0), T)
        dj = tv(P1.ravel(), np.tile(T, 2) / 2) - tv(P.ravel(), np.tile(T, 2) / 2)
        if d > worst[0]:
            worst = (d, (tv(P.sum(0), T), tv(P1.sum(0), T), dj))
    L.append(f"hunt (policy = T x skewed alien bit, kernel built from that policy, one move): largest rise of TV over "
             f"demonstrations {worst[0]:.4f} (from {worst[1][0]:.4f} to {worst[1][1]:.4f}; joint TV change {worst[1][2]:+.4f})")
    # residual variance of note section 3 (ii), as coded in e3a (target 0.5 R(d) on (alien, d)), for a sampler whose
    # demonstration law is exactly T but whose alien bit is informative (what 'count each demonstration once' keeps)
    PB = law(*noisy_policy(R, 1.5, 31 + ws, lean=2.0))
    c = PB / PB.sum(0, keepdims=True)
    Pstar = T[None, :] * c
    rho = np.log(Pstar) - np.log(0.5 * R[None, :])
    var_joint = float((Pstar * rho ** 2).sum() - (Pstar * rho).sum() ** 2)
    rho_m = np.log(Pstar.sum(0)) - np.log(R)
    var_marg = float((T * rho_m ** 2).sum() - (T * rho_m).sum() ** 2)
    rho_b = np.log(Pstar) - np.log(c * R[None, :])
    var_pb = float((Pstar * rho_b ** 2).sum() - (Pstar * rho_b).sum() ** 2)
    L.append(f"sampler with demonstration law exactly T and informative alien bit (I = {mi(Pstar):.3f} nats): residual variance "
             f"with fixed 1/2 backward law {var_joint:.4f}; alien bit summed out {var_marg:.2e}; with its own backward law {var_pb:.2e}")
open("out/c2_distill.out", "w").write("\n".join(L) + "\n")
print("\n".join(L))
