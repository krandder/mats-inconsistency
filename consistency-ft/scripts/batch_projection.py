"""Batched weighted coherence projection over the 4 atoms (numpy, CPU only).

Vectorized over pairs: atoms q = softmax([z1, z2, z3, 0]); damped Newton (numerical Hessian of
the analytic gradient, Levenberg damping, per-pair backtracking), two starts (uniform and the
independence product of the stated marginals), keep the better one per pair.

Losses (per pair, summed over forms f with weight w_f; w_f = 0 drops a form, e.g. a missing answer):
  kl   sum_f w_f KL(Bernoulli(s_f) || Bernoulli(m_f))
  lsl  sum_f w_f (logit m_f - t_f)^2
where m = A q (clipped to [1e-6, 1 - 1e-6]), t_f = the target log-odds and s_f = sigmoid(t_f).
With w = 1 and t = logit(clip(stated, 0.01)), kl matches scripts/coherence_projection.fit(.., "kl").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coherence_projection import FORM_ATOMS  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

A = np.array([FORM_ATOMS[f] for f in FORM_ORDER], dtype=float)  # 12 x 4
MEPS = 1e-6
ZMAX = 25.0


def _q(z):
    zz = np.concatenate([z, np.zeros((len(z), 1))], 1)
    zz = zz - zz.max(1, keepdims=True)
    e = np.exp(zz)
    return e / e.sum(1, keepdims=True)


def _loss_grad(z, t, w, kind, grad=True):
    q = _q(z)
    m_raw = q @ A.T
    m = np.clip(m_raw, MEPS, 1 - MEPS)
    inside = (m_raw > MEPS) & (m_raw < 1 - MEPS)
    if kind == "kl":
        s = 1 / (1 + np.exp(-t))
        L = (w * (s * np.log(s / m) + (1 - s) * np.log((1 - s) / (1 - m)))).sum(1)
        dm = w * (m - s) / (m * (1 - m))
    elif kind == "lsl":
        r = np.log(m / (1 - m)) - t
        L = (w * r ** 2).sum(1)
        dm = 2 * w * r / (m * (1 - m))
    else:
        raise ValueError(kind)
    if not grad:
        return L
    dm = dm * inside
    gq = dm @ A  # N x 4
    gz = q * (gq - (q * gq).sum(1, keepdims=True))
    return L, gz[:, :3]


def _newton(z, t, w, kind, iters=60):
    N = len(z)
    mu = np.full(N, 1e-3)
    L, g = _loss_grad(z, t, w, kind)
    h = 1e-4
    for _ in range(iters):
        H = np.zeros((N, 3, 3))
        for k in range(3):
            dz = np.zeros(3); dz[k] = h
            _, gp = _loss_grad(z + dz, t, w, kind)
            _, gm = _loss_grad(z - dz, t, w, kind)
            H[:, :, k] = (gp - gm) / (2 * h)
        H = 0.5 * (H + H.transpose(0, 2, 1))
        # make positive definite: eigen floor
        ev, V = np.linalg.eigh(H)
        ev = np.maximum(np.abs(ev), 1e-8)
        improved = np.zeros(N, bool)
        for _ls in range(8):
            Hi = np.einsum("nij,nj,nkj->nik", V, 1 / (ev + mu[:, None]), V)
            step = -np.einsum("nij,nj->ni", Hi, g)
            zn = np.clip(z + step, -ZMAX, ZMAX)
            Ln = _loss_grad(zn, t, w, kind, grad=False)
            ok = (Ln <= L) & ~improved
            z = np.where(ok[:, None], zn, z)
            L = np.where(ok, Ln, L)
            improved |= ok
            mu = np.where(ok, mu * 0.3, np.where(improved, mu, mu * 10))
            if improved.all():
                break
        mu = np.clip(mu, 1e-9, 1e9)
        L, g = _loss_grad(z, t, w, kind)
        if np.abs(g).max() < 1e-9:
            break
    return z, L


def project(t: np.ndarray, w: np.ndarray, kind: str = "kl") -> np.ndarray:
    """t, w: N x 12 (FORM_ORDER). Returns atoms N x 4 (PQ, P~Q, ~PQ, ~P~Q)."""
    t = np.asarray(t, float); w = np.asarray(w, float)
    t = np.where(w > 0, t, 0.0)
    N = len(t)
    best_z, best_L = None, None
    iP, iQ = FORM_ORDER.index("P"), FORM_ORDER.index("Q")
    pP = np.where(w[:, iP] > 0, 1 / (1 + np.exp(-t[:, iP])), 0.5)
    pQ = np.where(w[:, iQ] > 0, 1 / (1 + np.exp(-t[:, iQ])), 0.5)
    ind = np.stack([pP * pQ, pP * (1 - pQ), (1 - pP) * pQ, (1 - pP) * (1 - pQ)], 1)
    for q0 in (np.full((N, 4), 0.25), ind):
        q0 = np.clip(q0, 1e-4, None)
        z0 = np.log(q0[:, :3] / q0[:, 3:4])
        z, L = _newton(z0, t, w, kind)
        if best_z is None:
            best_z, best_L = z, L
        else:
            better = L < best_L
            best_z = np.where(better[:, None], z, best_z)
            best_L = np.where(better, L, best_L)
    return _q(best_z)


def log_or(atoms: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    a = np.clip(atoms, floor, None)
    return np.log(a[:, 0]) + np.log(a[:, 3]) - np.log(a[:, 1]) - np.log(a[:, 2])


def joint_with_or_batch(a: np.ndarray, b: np.ndarray, lt: np.ndarray, iters: int = 100) -> np.ndarray:
    """Atoms with marginals a, b and log odds ratio lt (vectorized bisection on p11)."""
    lo = np.maximum(0.0, a + b - 1); hi = np.minimum(a, b)
    for _ in range(iters):
        m = (lo + hi) / 2
        c = np.clip(np.stack([m, a - m, b - m, 1 - a - b + m], 1), 1e-300, None)
        cur = np.log(c[:, 0]) + np.log(c[:, 3]) - np.log(c[:, 1]) - np.log(c[:, 2])
        lo = np.where(cur < lt, m, lo); hi = np.where(cur < lt, hi, m)
    m = (lo + hi) / 2
    return np.stack([m, a - m, b - m, 1 - a - b + m], 1)
