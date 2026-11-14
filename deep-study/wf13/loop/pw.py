"""pw.py -- parity world: the specification's books. NO exact inference and no network in this file.

The one score (wf7): R(h) = prod_i a_i(h_i) * prod_c f_c(x_c | h_pa(c)), x = the observed checks (evidence as a factor).
A demonstration = the assumptions in some order, then the deductions (each check read off its three parents). The score
is the same for every order of the assumptions, and every filling has the same number of orders, so counting each
filling once counts each demonstration once up to one constant, which cancels in every answer (NOTE section 1).

Books for a partial demonstration s (some assumptions assigned). Factors already fixed:
  - a_i for every assigned assumption; f_c for every check whose three parents are assigned;
  - a check with exactly ONE open parent j is summed out together with a_j (exact: one variable, two terms);
  - a check with two or three open parents is only bounded: its largest and smallest value over the open parents.
So  L(s) <= total(s) <= U(s)  with U/L = prod over checks with >= 2 open parents of (max f / min f); a state with no such
check is CLOSED: its total and all its marginals are known exactly (the open assumptions are independent there).
"""
import numpy as np

UN = 2                                   # code of an unassigned assumption


class Inst:
    """One instance: the score table of its world and one observed check vector. Precomputes, per check and per
    pattern of its three parents (27 patterns), the fixed factor, the single-open-parent terms and the bounds."""

    def __init__(self, tab, x):
        self.loga = np.asarray(tab['loga'], dtype=np.float64)              # [n, 2]
        logf = np.asarray(tab['logf'], dtype=np.float64)                   # [m, 8, 2]
        self.checks = np.asarray(tab['checks'], dtype=np.int64)            # [m, 3]
        self.n, self.m = self.loga.shape[0], self.checks.shape[0]
        self.x = np.asarray(x, dtype=np.int64)
        n, m = self.n, self.m
        lf = logf[np.arange(m), :, self.x]                                 # [m, 8]: log f_c(x_c | parent configuration)
        D = np.array([[(code // 3 ** t) % 3 for t in range(3)] for code in range(27)])        # digits of a pattern
        cons = np.ones((27, 8), dtype=bool)                                # configuration consistent with the pattern
        for t in range(3):
            bit = (np.arange(8) >> t) & 1
            cons &= (D[:, t, None] == UN) | (D[:, t, None] == bit[None, :])
        nun = (D == UN).sum(1)
        big = np.where(cons[None], lf[:, None, :], -np.inf).max(2)         # [m, 27]
        small = np.where(cons[None], lf[:, None, :], np.inf).min(2)
        self.NUN = nun
        self.COMP = np.where(nun[None] == 0, big, 0.0)
        self.LMAX = np.where(nun[None] >= 2, big, 0.0)
        self.LMIN = np.where(nun[None] >= 2, small, 0.0)
        pos = np.argmax(D == UN, 1)                                        # position of the single open parent
        base = sum(np.where(D[:, t] == 1, 1 << t, 0) for t in range(3))
        self.S0 = np.where(nun[None] == 1, lf[:, base], 0.0)               # log f with the open parent = 0
        self.S1 = np.where(nun[None] == 1, lf[:, np.minimum(base + (1 << pos), 7)], 0.0)
        self.JV = np.where(nun[None] == 1, self.checks[:, pos], n)         # [m, 27] index of that parent (n = none)
        self.cidx = np.arange(m)[None, :]
        self.prior_logit = self.loga[:, 1] - self.loga[:, 0]
        # the order-free total of the root's bound, used as the unit of all masses
        self.off = 0.0
        self.calls = 0

    def ev(self, A):
        """Books of a batch of partial states A [B, n] in {0, 1, 2 = open}. Returns a dict of arrays."""
        B, n, m = A.shape[0], self.n, self.m
        Ai = A.astype(np.int64)
        P = self.checks
        codes = Ai[:, P[:, 0]] + 3 * Ai[:, P[:, 1]] + 9 * Ai[:, P[:, 2]]   # [B, m]
        nun = self.NUN[codes]
        comp = self.COMP[self.cidx, codes].sum(1)
        lmax = self.LMAX[self.cidx, codes].sum(1)
        lmin = self.LMIN[self.cidx, codes].sum(1)
        jv = self.JV[self.cidx, codes]
        flat = (np.arange(B)[:, None] * (n + 1) + jv).ravel()
        G0 = np.bincount(flat, weights=self.S0[self.cidx, codes].ravel(), minlength=B * (n + 1)).reshape(B, n + 1)[:, :n]
        G1 = np.bincount(flat, weights=self.S1[self.cidx, codes].ravel(), minlength=B * (n + 1)).reshape(B, n + 1)[:, :n]
        op = Ai == UN
        l0, l1 = self.loga[None, :, 0] + G0, self.loga[None, :, 1] + G1
        E = np.logaddexp(l0, l1)
        fixed = np.where(Ai == 1, self.loga[None, :, 1], self.loga[None, :, 0])
        mid = np.where(op, E, fixed).sum(1) + comp
        multi = nun >= 2
        # an open assumption is FREE when no check with >= 2 open parents contains it: its split is then exact
        inm = np.zeros((B, n), dtype=bool)
        rows = np.arange(B)[:, None]
        for t in range(3):
            inm[np.broadcast_to(rows, (B, m))[multi], np.broadcast_to(P[None, :, t], (B, m))[multi]] = True
        inm &= op
        tl = np.where(op, l1 - l0, 0.0)                                    # the books' local log-odds of each open one
        p1 = np.where(op, 1.0 / (1.0 + np.exp(-tl)), Ai.astype(np.float64))
        return dict(A=A, logU=mid + lmax, logL=mid + lmin, closed=~multi.any(1), p1=p1, tl=tl, op=op, free=op & ~inm,
                    nun=nun, codes=codes, nmulti=multi.sum(1))

    def log_R(self, H):
        """Score of complete assignments H [B, n] (all checks read off their parents)."""
        e = self.ev(H.astype(np.int8))
        return e['logU']


def log_mix(q, eps):
    """Floor on the value probabilities: (1 - eps) q + eps / 2."""
    return (1.0 - eps) * q + 0.5 * eps
