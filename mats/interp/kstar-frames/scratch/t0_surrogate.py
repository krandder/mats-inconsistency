"""T0 (theory, before any ReLU-net data): one-path frame transport in the Gaussian surrogate of Theorem C.

By Theorem C (lyapunov lane, PROVEN for sigma = 0 on the no-dead-layer event), the angular part of one ReLU path's
downstream Jacobian is a product of independent Gaussian matrices of half width (entries of variance 1/m, m ~ n/2),
and each layer's angular factor H_c is independent of the future product. So in the surrogate:
  J_c   = H_L ... H_{c+1}            (L' factors, m x m, iid N(0, 1/m))
  F_c   = top-k right singular vectors of J_c                       (the "true frame" at layer c)
  F_c-t = top-k right singular vectors of J_c H_c ... H_{c-t+1}      (true frame t layers earlier)
Predictors of F_c:
  fwd_t  = orth(H_c ... H_{c-t+1} F_{c-t})       push-forward by the layers' own Jacobians, QR (t = 1, 2, 4, 8)
  dual_1 = orth(H_c^{-T} F_{c-1})                push-forward by the inverse transpose (covariant for right frames)
  bwd_1  = orth(H_{c+1}^T F_{c+1})               one step backward from the next layer's frame (consistency check)
  blv    = top-k left singular vectors of the past product H_c ... H_1 (what forward QR converges to)
  rand   = a uniformly random k-frame (chance)
Scores: overlap O = tr(P_G P_F)/k (mean cos^2 of principal angles); captured sensitivity CS = |J_c G|_F^2 / sum_{i<=k} s_i^2.
Prints means over paths. Pure Gaussian surrogate: no ReLU net is touched.
nice -n 19 ionice -c3 python t0_surrogate.py"""
import math, sys
import numpy as np


def orth(A):
    q, _ = np.linalg.qr(A)
    return q


def topk_right(J, k):
    _, s, vt = np.linalg.svd(J)
    return vt[:k].T, s


def score(G, F, J, s, k):
    O = np.linalg.norm(F.T @ G) ** 2 / k
    CS = np.linalg.norm(J @ G) ** 2 / np.sum(s[:k] ** 2)
    return O, CS


def run(m, Lp, k, cpast=16, paths=64, seed=0):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(paths):
        g = lambda: rng.standard_normal((m, m)) / math.sqrt(m)
        fut = [g() for _ in range(Lp + 1)]            # fut[0] = H_{c+1}, ..., fut[Lp-1] = H_L ; fut[Lp] unused
        past = [g() for _ in range(cpast)]            # past[0] = H_c, past[1] = H_{c-1}, ...
        Jc = np.eye(m)
        for H in fut[:Lp]:
            Jc = H @ Jc
        Fc, s = topk_right(Jc, k)
        res = {}
        # forward t-step from the true frame t layers earlier
        for t in (1, 2, 4, 8):
            Jt = Jc.copy()
            for H in past[:t]:
                Jt = Jt @ H
            Ft, _ = topk_right(Jt, k)
            G = Ft
            for H in reversed(past[:t]):              # H_{c-t+1} first, H_c last
                G = orth(H @ G)
            res[f'fwd_{t}'] = score(G, Fc, Jc, s, k)
            if t == 1:
                G = orth(np.linalg.solve(past[0].T, Ft))   # H_c^{-T} F_{c-1}
                res['dual_1'] = score(G, Fc, Jc, s, k)
        # backward one step from the next layer's frame: J_{c+1} = H_L ... H_{c+2}, F_{c+1} its frame
        J1 = np.eye(m)
        for H in fut[1:Lp]:
            J1 = H @ J1
        F1, _ = topk_right(J1, k)
        res['bwd_1'] = score(orth(fut[0].T @ F1), Fc, Jc, s, k)
        # BLV: top-k left singular vectors of the past product H_c ... H_1
        P = np.eye(m)
        for H in reversed(past):
            P = H @ P
        u, _, _ = np.linalg.svd(P)
        res['blv'] = score(u[:, :k], Fc, Jc, s, k)
        res['rand'] = score(orth(rng.standard_normal((m, k))), Fc, Jc, s, k)
        for key, v in res.items():
            acc.setdefault(key, []).append(v)
    return {key: np.array(v).mean(0) for key, v in acc.items()}


if __name__ == '__main__':
    print("Gaussian surrogate (half width m, entries var 1/m). Columns: overlap O (mean cos^2), captured sensitivity CS")
    for m, Lp, k in ((16, 16, 6), (32, 16, 12), (64, 16, 22), (32, 8, 18), (64, 8, 34), (64, 24, 16)):
        r = run(m, Lp, k)
        print(f"m={m} L'={Lp} k={k} chance k/m={k/m:.3f}: " +
              "  ".join(f"{key} O={v[0]:.3f} CS={v[1]:.3f}" for key, v in r.items()))
        sys.stdout.flush()
