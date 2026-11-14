"""sk.py -- 6x6 Sudoku: the specification's books, the tree and the importance form. NO exact inference, no solver.

The score (wf7): a demonstration is a filling together with its order; each step is scored by the rule table read from
M's one-cell contexts (allowed[i, u, j, v]); the goal factor is "all cells filled". The score is 0 or 1 and the same for
every order of a valid filling, so counting fillings once counts demonstrations once up to the constant 36!/clues.
Books of a partial demonstration s: its candidates cand[j, v] = no collision of (j, v) with any filled cell (the step
factors already fixed). total(s) = number of valid completions:  0 <= total(s) <= prod over empty cells of |cand_j|;
it is KNOWN to be 0 when some empty cell has no candidate, and 1 when s is complete.
"""
import numpy as np

K, N = 6, 36


class Rules:
    def __init__(self, allowed):
        self.allowed = allowed                      # [36, 6, 36, 6] bool, read from M

    def cand_of(self, board):
        c = np.ones((N, K), dtype=bool)
        for i in np.flatnonzero(board >= 0):
            c &= self.allowed[i, board[i]]
        return c


def log_bound(cand, empty):
    cnt = np.where(empty, cand.sum(-1), 1)
    with np.errstate(divide='ignore'):
        return np.log(cnt.astype(np.float64)).sum(-1)            # -inf when an empty cell has no candidate


class Tree:
    def __init__(self, rules, puz, cap):
        self.rules, self.cap, self.N = rules, cap, 1
        self.board = np.full((cap, N), -1, dtype=np.int8)
        self.cand = np.zeros((cap, N, K), dtype=bool)
        self.logU, self.lq = np.zeros(cap), np.zeros(cap)
        self.status = np.zeros(cap, dtype=np.int8)       # 0 open, 1 opened, 2 complete (found), 3 dead (total = 0)
        self.parent = np.full(cap, -1, dtype=np.int64)
        self.board[0] = puz
        self.cand[0] = rules.cand_of(puz)
        emp = puz < 0
        ok = all(self.cand[0][i, puz[i]] for i in np.flatnonzero(~emp))       # the clues do not collide
        self.logU[0] = log_bound(self.cand[0], emp) if ok else -np.inf
        self.status[0] = 3 if not np.isfinite(self.logU[0]) else (2 if not emp.any() else 0)
        self.opened, self.calls, self.first_found = 0, 0, -1
        self.full = False

    def n_open(self):
        return int((self.status[:self.N] == 0).sum())


def grow(T, policy, target, rng, rho=0.5, eps=0.02, wave=16):
    """Open states until `target` openings (or the frontier is empty, or memory is full). The policy picks the cell
    and gives value probabilities; the specification creates one child per candidate value and books it."""
    al = T.rules.allowed
    while T.opened < target and not T.full:
        fr = np.flatnonzero(T.status[:T.N] == 0)
        if len(fr) == 0:
            break
        k = min(wave, target - T.opened, len(fr))
        ks = min(k, int(np.floor(rho * k + rng.random())))
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
        if T.N + K * len(sel) > T.cap:
            T.full = True
            break
        b, c = T.board[sel], T.cand[sel]
        q, j = policy(b, c, rng)                              # q [k, 36, 6] over values, j [k]
        T.calls += len(sel) * policy.calls_per_state
        ar = np.arange(len(sel))
        cj = c[ar, j]                                         # [k, 6] candidates of the chosen cell
        qj = np.where(cj, q[ar, j], 0.0)
        qj = qj / np.maximum(qj.sum(1, keepdims=True), 1e-300)
        qj = np.where(cj, (1 - eps) * qj + eps / np.maximum(cj.sum(1, keepdims=True), 1), 0.0)
        for v in range(K):
            rows = np.flatnonzero(cj[:, v])
            if len(rows) == 0:
                continue
            s = slice(T.N, T.N + len(rows))
            nb = b[rows].copy()
            nb[np.arange(len(rows)), j[rows]] = v
            nc = c[rows] & al[j[rows], v]
            emp = nb < 0
            lu = log_bound(nc, emp)
            T.board[s], T.cand[s], T.logU[s] = nb, nc, lu
            T.status[s] = np.where(~np.isfinite(lu), 3, np.where(emp.any(1), 0, 2))
            T.parent[s] = sel[rows]
            with np.errstate(divide='ignore'):
                T.lq[s] = T.lq[sel[rows]] + np.log(qj[rows, v])
            T.N += len(rows)
        T.status[sel] = 1
        T.opened += len(sel)
        if T.first_found < 0 and (T.status[:T.N] == 2).any():
            T.first_found = T.opened


def books(T):
    """Certain interval for T(cell j has value v), all 216 pairs: found solutions for / against, bounded number of
    completions of what was not opened. lo = ZA / (ZA + ZN + U_N + U_u), hi = (ZA + U_A + U_u) / (ZA + U_A + U_u + ZN)."""
    st = T.status[:T.N]
    sol = T.board[:T.N][st == 2]
    Z = len(sol)
    ZA = np.zeros((N, K))
    for v in range(K):
        ZA[:, v] = (sol == v).sum(0)
    fr = st == 0
    U = np.exp(T.logU[:T.N][fr])
    b, c = T.board[:T.N][fr], T.cand[:T.N][fr]
    filled = b >= 0
    isv = np.stack([(b == v) for v in range(K)], -1)             # [F, 36, 6]
    und = (~filled)[:, :, None] & c                              # empty and v still a candidate
    UA = np.tensordot(U, isv.astype(np.float64), 1)
    Uu = np.tensordot(U, und.astype(np.float64), 1)
    # states that exclude (j, v): summed directly. (Before 17:27Z this was U.sum() - UA - Uu, which loses the small
    # terms when the frontier bounds are ~1e15 and produced 2 false lower bounds; PREREG-ADDENDA A1.)
    UN = np.tensordot(U, (~isv & ~und).astype(np.float64), 1)
    ZN = Z - ZA
    with np.errstate(invalid='ignore', divide='ignore'):
        lo = ZA / (ZA + ZN + UN + Uu)
        hi = (ZA + UA + Uu) / (ZA + UA + Uu + ZN)
    lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
    pt = ZA / Z if Z > 0 else np.full((N, K), np.nan)
    return dict(lo=lo, hi=hi, pt=pt, Z=Z, sol=sol, n_open=int(fr.sum()), closed=bool(fr.sum() == 0), UF=float(U.sum()))


def run_is(rules, puz, policy, Kr, rng, eps=0.02):
    """Importance form: Kr rollouts; a rollout stops when complete (weight 1 / its probability) or dead (weight 0).
    Returns boards [Kr, 36], log weights [Kr] (-inf for dead), network calls."""
    al = rules.allowed
    b = np.repeat(puz[None].astype(np.int8), Kr, 0)
    c = np.repeat(rules.cand_of(puz)[None], Kr, 0)
    lq = np.zeros(Kr)
    alive = np.ones(Kr, dtype=bool)
    calls = 0
    while True:
        emp = b < 0
        dead = alive & ((emp & ~c.any(-1)).any(1))
        alive &= ~dead
        ia = np.flatnonzero(alive & emp.any(1))
        if len(ia) == 0:
            break
        q, j = policy(b[ia], c[ia], rng)
        calls += len(ia) * policy.calls_per_state
        ar = np.arange(len(ia))
        cj = c[ia, j]
        qj = np.where(cj, q[ar, j], 0.0)
        qj = qj / np.maximum(qj.sum(1, keepdims=True), 1e-300)
        qj = np.where(cj, (1 - eps) * qj + eps / cj.sum(1, keepdims=True), 0.0)
        v = (qj.cumsum(1) < rng.random((len(ia), 1))).sum(1).clip(0, K - 1)
        with np.errstate(divide='ignore'):
            lq[ia] += np.log(qj[ar, v])
        b[ia, j] = v
        c[ia] &= al[j, v]
    done = alive & ~(b < 0).any(1)
    return b, np.where(done, -lq, -np.inf), calls
