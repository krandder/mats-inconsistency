"""pexact.py -- EVALUATION CODE ONLY: exact answers by enumeration over all 2^n assignments (n <= 24).
No arm (pw.py, ptree.py, ppol.py, ploop training / search code paths) may use a number produced here except to score
results after the fact; check_firewall.py tests the imports."""
import numpy as np


class Enum:
    def __init__(self, checks, n):
        z = np.arange(2 ** n, dtype=np.int64)
        self.n = n
        self.bits = ((z[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)          # [2^n, n]
        b = self.bits[:, checks].astype(np.int64)
        self.cfg = (b[..., 0] + 2 * b[..., 1] + 4 * b[..., 2]).astype(np.uint8)           # [2^n, m]
        self.bf = self.bits.astype(np.float64)

    def log_R(self, loga, logf, x):
        lr = self.bf @ (loga[:, 1] - loga[:, 0]) + loga[:, 0].sum()
        for c in range(self.cfg.shape[1]):
            lr += logf[c, :, int(x[c])][self.cfg[:, c]]
        return lr

    def marg(self, lr, mask=None):
        """Exact P(h_j = 1) under exp(lr) (optionally restricted to a boolean mask), and log of the total."""
        if mask is not None:
            lr = np.where(mask, lr, -np.inf)
        mx = lr.max()
        w = np.exp(lr - mx)
        s = w.sum()
        return (w @ self.bf) / s, mx + np.log(s)

    def cond(self, lr, A):
        """Exact conditional marginals at partial states A [B, n] in {0, 1, 2 = open} (pilot stand-in only)."""
        if getattr(self, '_lr_id', None) is not lr:
            self._lr_id, self._w = lr, np.exp(lr - lr.max())
            self._z = np.arange(2 ** self.n, dtype=np.int64)
        out = np.zeros(A.shape)
        pw2 = (1 << np.arange(self.n)).astype(np.int64)
        for i, a in enumerate(A):
            mb, vb = int(pw2[a != 2].sum()), int(pw2[a == 1].sum())
            w = self._w * ((self._z & mb) == vb)
            out[i] = (w @ self.bf) / w.sum()
        return out


def true_table(n, m, prior, eps):
    p = np.clip(prior, 1e-12, 1 - 1e-12)
    loga = np.stack([np.log(1 - p), np.log(p)], -1)
    par = np.array([bin(v).count('1') & 1 for v in range(8)])
    f1 = np.where(par == 1, 1 - eps, eps)
    logf = np.broadcast_to(np.stack([np.log(1 - f1), np.log(f1)], -1)[None], (m, 8, 2)).copy()
    return loga, logf


def kl_bern(p, b):
    b = np.clip(b, 1e-6, 1 - 1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(b)), 0.0)
        t0 = np.where(p < 1, (1 - p) * (np.log(np.maximum(1 - p, 1e-300)) - np.log(1 - b)), 0.0)
    return t1 + t0


class EnumSub:
    """Exact answers when some assumptions are given: enumeration over the open ones only (at most 24 of them)."""

    def __init__(self, checks, root):
        self.open = np.flatnonzero(root == 2)
        k, n = len(self.open), len(root)
        z = np.arange(2 ** k, dtype=np.int64)
        full = np.repeat(np.where(root == 2, 0, root).astype(np.uint8)[None], 2 ** k, 0)
        full[:, self.open] = ((z[:, None] >> np.arange(k)[None, :]) & 1).astype(np.uint8)
        b = full[:, checks].astype(np.int64)
        self.cfg = (b[..., 0] + 2 * b[..., 1] + 4 * b[..., 2]).astype(np.uint8)
        self.full = full

    def answer(self, loga, logf, x):
        f = self.full.astype(np.float64)
        lr = f @ (loga[:, 1] - loga[:, 0]) + loga[:, 0].sum()
        for c in range(self.cfg.shape[1]):
            lr += logf[c, :, int(x[c])][self.cfg[:, c]]
        mx = lr.max()
        w = np.exp(lr - mx)
        return (w @ f) / w.sum(), mx + np.log(w.sum())
