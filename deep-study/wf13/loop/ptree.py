"""ptree.py -- parity world: the two amplifications. The rollout policy proposes; the specification keeps the books.
NO exact inference here (pexact.py is evaluation only) and no network (policies are passed in).

Tree form. A rollout walks down the opened tree by the policy's value probabilities and opens the first unopened state
it reaches (drawn without replacement: Gumbel top-k on the path log-probabilities). When a state is opened the POLICY
picks the assumption to branch on; the SPECIFICATION creates both children and computes their books (pw.Inst.ev).
A share rho of the openings is owned by the specification (largest upper bound first; no network number is read).
books(): found score for and against every assumption, bounded score of what was not opened -> an interval that
contains the target's answer for every policy (GUARANTEE; NOTE section 2).
audit(): the specification's own draws below the open frontier -> a probabilistic interval (empirical Bernstein).
Importance form (run_is): K rollouts to closed states, weight = exact total of the closed state / policy probability.
"""
import numpy as np

from pw import UN, log_mix


class Tree:
    def __init__(self, inst, cap):
        n = inst.n
        self.inst, self.N = inst, 0
        self.A = np.full((cap, n), UN, dtype=np.int8)
        self.logU, self.logL, self.lq = np.zeros(cap), np.zeros(cap), np.zeros(cap)
        self.p1 = np.zeros((cap, n))
        self.free = np.zeros((cap, n), dtype=bool)
        self.status = np.zeros(cap, dtype=np.int8)            # 0 open frontier, 1 opened, 2 closed (exact total)
        self.parent = np.full(cap, -1, dtype=np.int64)
        self.bvar = np.full(cap, -1, dtype=np.int64)
        self.opened, self.calls = 0, 0

    def add(self, e, parent, lq):
        k = len(e['logU'])
        s = slice(self.N, self.N + k)
        self.A[s], self.logU[s], self.logL[s], self.p1[s], self.free[s] = e['A'], e['logU'], e['logL'], e['p1'], e['free']
        self.status[s] = np.where(e['closed'], 2, 0)
        self.parent[s], self.lq[s] = parent, lq
        self.N += k


def wave_size(opened):
    return 8 if opened < 64 else (32 if opened < 1024 else 128)


def grow(T, policy, target, rng, rho=0.5, eps=0.02):
    """Open states until `target` openings in total (or nothing is left to open)."""
    inst = T.inst
    while T.opened < target:
        fr = np.flatnonzero(T.status[:T.N] == 0)
        if len(fr) == 0:
            break
        k = min(wave_size(T.opened), target - T.opened, len(fr))
        ks = int(np.floor(rho * k + rng.random()))                      # specification-owned share of this wave
        ks = min(max(ks, 0), k)
        sel = np.zeros(0, dtype=np.int64)
        if ks > 0:
            o = np.argpartition(-T.logU[fr], ks - 1)[:ks]
            sel = fr[o]
            fr = np.delete(fr, o)
        if k - ks > 0:
            fr = fr[np.isfinite(T.lq[fr])]               # a state the policy gives probability 0 is never reached
            kp = min(k - ks, len(fr))
            if kp > 0:
                g = T.lq[fr] + rng.gumbel(size=len(fr))
                o = np.argpartition(-g, kp - 1)[:kp]
                sel = np.concatenate([sel, fr[o]])
        if len(sel) == 0:
            break                                        # nothing left that this search would ever open
        e = inst.ev(T.A[sel])
        q, j = policy(inst, e, rng)
        T.calls += len(sel) * policy.calls_per_state
        ar = np.arange(len(sel))
        qj = log_mix(np.clip(q[ar, j], 0.0, 1.0), eps)
        A0, A1 = T.A[sel].copy(), T.A[sel].copy()
        A0[ar, j], A1[ar, j] = 0, 1
        ech = inst.ev(np.concatenate([A0, A1]))
        with np.errstate(divide='ignore'):
            lq = np.concatenate([T.lq[sel] + np.log(1 - qj), T.lq[sel] + np.log(qj)])
        T.add(ech, np.concatenate([sel, sel]), lq)
        T.status[sel], T.bvar[sel] = 1, j
        T.opened += len(sel)


def books(T):
    """Certain interval for P(h_j = 1) for every j, from found and bounded score. Unit of mass: the root's bound."""
    N, off = T.N, T.logU[0]
    st = T.status[:N]
    cl, fr = st == 2, st == 0
    tot = np.exp(T.logU[:N][cl] - off)
    Z = tot.sum()
    ZA = tot @ T.p1[:N][cl]
    ZN = Z - ZA
    U, L = np.exp(T.logU[:N][fr] - off), np.exp(T.logL[:N][fr] - off)
    A, p1, free = T.A[:N][fr], T.p1[:N][fr], T.free[:N][fr]
    sa = (A == 1) + free * p1                       # share of the state known to be on the "for" side
    sn = (A == 0) + free * (1 - p1)
    und = (A == UN) & ~free
    UA, LA, UNn, LN, Uu = U @ sa, L @ sa, U @ sn, L @ sn, U @ und
    with np.errstate(invalid='ignore', divide='ignore'):
        lo = (ZA + LA) / (ZA + LA + ZN + UNn + Uu)
        hi = (ZA + UA + Uu) / (ZA + UA + Uu + ZN + LN)
        pt = ZA / Z if Z > 0 else 0.5 * (lo + hi)
    lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
    pt = np.clip(np.nan_to_num(pt, nan=0.5), lo, hi)
    UF = U.sum()
    return dict(lo=lo, hi=hi, pt=pt, Z=Z, ZA=ZA, UF=UF, LF=L.sum(), n_open=int(fr.sum()), n_closed=int(cl.sum()),
                unfound_ub=UF / (Z + UF) if Z + UF > 0 else 1.0, logZ_lo=np.log(max(Z + L.sum(), 1e-300)) + off,
                logZ_hi=np.log(max(Z + UF, 1e-300)) + off)


def audit(T, bk, n_draw, rng, delta=0.05):
    """Probabilistic interval from the specification's OWN draws: pick an open state with probability proportional
    to its upper bound, complete it with the books' local numbers (no policy, no network), and read the score.
    y = (score of the completion) / (upper bound of the state) lies in [0, 1], and unfound score = UF * E[y].
    Four one-sided empirical-Bernstein bounds (Maurer and Pontil 2009) at delta / 4 each."""
    inst, N = T.inst, T.N
    fr = np.flatnonzero(T.status[:N] == 0)
    if len(fr) == 0 or n_draw < 2:
        return dict(lo=bk['lo'].copy(), hi=bk['hi'].copy(), ybar=0.0)
    U = np.exp(T.logU[fr] - T.logU[0])
    pick = fr[rng.choice(len(fr), size=n_draw, p=U / U.sum())]
    A, p1, free = T.A[pick], T.p1[pick], T.free[pick]
    op = A == UN
    H = np.where(op, rng.random(A.shape) < p1, A).astype(np.int64)
    P = inst.checks
    Ai = A.astype(np.int64)
    multi = inst.NUN[Ai[:, P[:, 0]] + 3 * Ai[:, P[:, 1]] + 9 * Ai[:, P[:, 2]]] >= 2
    codeH = H[:, P[:, 0]] + 3 * H[:, P[:, 1]] + 9 * H[:, P[:, 2]]
    lf = inst.COMP[inst.cidx, codeH]                                 # log f_c(x_c | parents of the completion)
    lmax = inst.LMAX[inst.cidx, Ai[:, P[:, 0]] + 3 * Ai[:, P[:, 1]] + 9 * Ai[:, P[:, 2]]]
    y = np.exp(((lf - lmax) * multi).sum(1))                         # in (0, 1]
    Hp = np.where(op & free, p1, H)                                  # exact split for the free open assumptions
    ya, yn = y[:, None] * Hp, y[:, None] * (1 - Hp)
    ln = np.log(8.0 / delta)

    def bound(v, sign):
        mu, var = v.mean(0), v.var(0, ddof=1)
        return mu + sign * (np.sqrt(2 * var * ln / n_draw) + 7 * ln / (3 * (n_draw - 1)))
    a_lo, a_hi = np.clip(bound(ya, -1), 0, 1), np.clip(bound(ya, +1), 0, 1)
    n_lo, n_hi = np.clip(bound(yn, -1), 0, 1), np.clip(bound(yn, +1), 0, 1)
    UF, ZA = bk['UF'], bk['ZA']
    ZN = bk['Z'] - ZA
    with np.errstate(invalid='ignore', divide='ignore'):
        lo = (ZA + UF * a_lo) / (ZA + UF * a_lo + ZN + UF * n_hi)
        hi = (ZA + UF * a_hi) / (ZA + UF * a_hi + ZN + UF * n_lo)
    lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
    pt = (ZA + UF * ya.mean(0)) / max(bk['Z'] + UF * y.mean(), 1e-300)
    return dict(lo=np.maximum(lo, bk['lo']), hi=np.minimum(hi, bk['hi']), ybar=float(y.mean()), pt=pt,
                draws=(pick, UF * y / n_draw, Hp))


def run_is(inst, policy, K, rng, eps=0.02):
    """Importance form: K rollouts, each ends at a closed state (exact total, exact marginals of what is still open).
    weight = total / (product of the policy's value probabilities). The policy's order choice is a function of the
    state, so every filling has exactly one route and the order cancels (NOTE section 1).
    Returns log weights [K], Rao-Blackwellised belief of each rollout [K, n], network calls."""
    n = inst.n
    A = np.full((K, n), UN, dtype=np.int8)
    lq, logw, bel = np.zeros(K), np.zeros(K), np.zeros((K, n))
    active = np.ones(K, dtype=bool)
    calls = 0
    while active.any():
        ia = np.flatnonzero(active)
        e = inst.ev(A[ia])
        done = e['closed']
        if done.any():
            d = ia[done]
            logw[d] = e['logU'][done] - lq[d]
            bel[d] = e['p1'][done]
            active[d] = False
        if (~done).any():
            keep = ~done
            sub = {k: v[keep] for k, v in e.items()}
            q, j = policy(inst, sub, rng)
            calls += int(keep.sum()) * policy.calls_per_state
            r = ia[keep]
            ar = np.arange(len(r))
            qj = log_mix(np.clip(q[ar, j], 0.0, 1.0), eps)
            v = rng.random(len(r)) < qj
            with np.errstate(divide='ignore'):
                lq[r] += np.where(v, np.log(qj), np.log(1 - qj))
            A[r, j] = v
    return logw, bel, calls


def snis(logw, bel, marks):
    out = {}
    for K in marks:
        lw = logw[:K]
        w = np.exp(lw - lw.max())
        out[K] = dict(b=(w[:, None] * bel[:K]).sum(0) / w.sum(), ess=float(w.sum() ** 2 / (w * w).sum()),
                      logZ=float(lw.max() + np.log(w.mean())))
    return out
