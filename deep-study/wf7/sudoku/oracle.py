"""oracle.py -- world generator and exact truth for K x K Sudoku (K = 4, 6, 9).

FIREWALL: this module holds everything that knows the rules of Sudoku in code: grid enumeration, validity checks,
exact counting of completions, uniqueness checks, puzzle generation and difficulty rating. It may be imported only by
data-generation scripts (gen_*.py) and scoring scripts (the `score_*` functions / eval sections). No arm (core.py,
arms.py: models, trust policies, searchers, baselines) imports it. `check_firewall.py` enforces this.
"""
import itertools, math
import numpy as np

BOXSHAPE = {4: (2, 2), 6: (2, 3), 9: (3, 3)}


def geom(K):
    br, bc = BOXSHAPE[K]
    N = K * K
    row = np.arange(N) // K
    col = np.arange(N) % K
    box = (row // br) * (K // bc) + (col // bc)
    return N, row, col, box


def peers(K):
    N, row, col, box = geom(K)
    P = (row[:, None] == row[None, :]) | (col[:, None] == col[None, :]) | (box[:, None] == box[None, :])
    P &= ~np.eye(N, dtype=bool)
    return P


def is_valid_batch(X, K):
    """X: [m, N] ints in 0..K-1. True where every row, column and box is a permutation."""
    N, row, col, box = geom(K)
    X = np.asarray(X)
    ok = np.ones(len(X), bool)
    for grp in (row, col, box):
        for g in range(K):
            ok &= np.all(np.sort(X[:, grp == g], axis=1) == np.arange(K), axis=1)
    return ok


def enumerate_base(K):
    """All valid grids whose first row is 0..K-1 (one representative per digit-relabelling class)."""
    N, row, col, box = geom(K)
    rowm = [0] * K; colm = [0] * K; boxm = [0] * K
    x = [-1] * N
    out = []

    def place(i, v):
        b = 1 << v
        rowm[row[i]] |= b; colm[col[i]] |= b; boxm[box[i]] |= b; x[i] = v

    def unplace(i, v):
        b = ~(1 << v)
        rowm[row[i]] &= b; colm[col[i]] &= b; boxm[box[i]] &= b; x[i] = -1

    for j in range(K):
        place(j, j)

    def rec(i):
        if i == N:
            out.append(x.copy()); return
        used = rowm[row[i]] | colm[col[i]] | boxm[box[i]]
        for v in range(K):
            if not (used >> v) & 1:
                place(i, v); rec(i + 1); unplace(i, v)

    rec(K)
    return np.array(out, dtype=np.int8)


def all_relabelings(B, K):
    """All grids = base grids x all K! digit relabellings. Only for K = 4."""
    perms = np.array(list(itertools.permutations(range(K))), dtype=np.int8)      # [K!, K]
    return perms[:, B].reshape(-1, B.shape[1])                                   # perms[p][B[g]] -> [K!, nb, N]


def cell_symmetries(K):
    """Cell permutations that map valid grids to valid grids: rows within bands, bands, columns within stacks,
    stacks, and transposition when boxes are square. Returns [n_sym, N] index arrays t with (g o t)[i] = g[t[i]]."""
    br, bc = BOXSHAPE[K]
    N = K * K

    def line_perms(block, nblocks):          # permutations of K lines: permute blocks, and lines within each block
        out = []
        for bp in itertools.permutations(range(nblocks)):
            for inner in itertools.product(list(itertools.permutations(range(block))), repeat=nblocks):
                out.append([bp[b] * block + inner[b][k] for b in range(nblocks) for k in range(block)])
        return out

    rps = line_perms(br, K // br)            # rows: bands of br rows
    cps = line_perms(bc, K // bc)            # cols: stacks of bc columns
    syms = []
    for rp in rps:
        for cp in cps:
            t = np.array([rp[r] * K + cp[c] for r in range(K) for c in range(K)])
            syms.append(t)
            if br == bc:
                syms.append(np.array([cp[c] * K + rp[r] for r in range(K) for c in range(K)]))  # transposed
    return np.array(syms)


def _keys(G, K):
    """Relabel-normalise (first row -> identity) and encode each grid as a pair of uint64 keys."""
    N = K * K
    first = G[:, :K].astype(np.int64)
    inv = np.argsort(first, axis=1)                                   # inv[first[j]] = j
    Gn = np.take_along_axis(inv, G.astype(np.int64), axis=1)
    half = N // 2
    w1 = (K ** np.arange(half, dtype=np.float64))[::-1]
    hi = (Gn[:, :half].astype(np.uint64) * w1.astype(np.uint64)).sum(1, dtype=np.uint64)
    w2 = (K ** np.arange(N - half, dtype=np.float64))[::-1]
    lo = (Gn[:, half:].astype(np.uint64) * w2.astype(np.uint64)).sum(1, dtype=np.uint64)
    return hi, lo


def orbit_ids(B, K):
    """Orbit (essentially-different class) id of every base grid under cell symmetries x digit relabelling."""
    assert K in (4, 6), "keys need 2 x uint64; fine for K <= 6"
    syms = cell_symmetries(K)
    best_hi = None
    for t in syms:
        hi, lo = _keys(B[:, t], K)
        if best_hi is None:
            best_hi, best_lo = hi, lo
        else:
            better = (hi < best_hi) | ((hi == best_hi) & (lo < best_lo))
            best_hi = np.where(better, hi, best_hi); best_lo = np.where(better, lo, best_lo)
    canon = np.stack([best_hi, best_lo], axis=1)
    uniq, ids = np.unique(canon, axis=0, return_inverse=True)
    return ids.reshape(-1), len(uniq)


class Counter:
    """Exact counting of completions of a partial filling, over ALL valid grids, using the base table and the
    digit-relabelling symmetry. state: array [N] with -1 = empty."""

    def __init__(self, B, K):
        self.B = np.ascontiguousarray(B.astype(np.int8)); self.K = K; self.N = K * K
        self.fact = [math.factorial(k) for k in range(K + 1)]

    def _consistent(self, s):
        K = self.K
        cells = np.where(s >= 0)[0]
        ok = np.ones(len(self.B), bool)
        E = []                                        # base digit that maps to each present target digit
        present = []
        for d in range(K):
            cd = cells[s[cells] == d]
            if len(cd) == 0:
                continue
            e = self.B[:, cd[0]]
            for c in cd[1:]:
                ok &= self.B[:, c] == e
            for e2 in E:
                ok &= e2 != e
            E.append(e); present.append(d)
        return ok, E, present

    def count(self, s):
        ok, E, present = self._consistent(s)
        return int(ok.sum()) * self.fact[self.K - len(present)]

    def conditionals(self, s):
        """Returns (n_completions, P) with P[i, v] = p(X_i = v | s) for every empty cell i (rows of filled cells
        are zero). Exact, over all valid grids."""
        K, N = self.K, self.N
        ok, E, present = self._consistent(s)
        nok = int(ok.sum()); m = len(present)
        C = np.zeros((N, K), dtype=np.float64)
        if nok == 0:
            return 0, C
        Bok = self.B[ok]
        seen = np.zeros(N, dtype=np.int64)
        for e, d in zip(E, present):
            cnt = (Bok == e[ok][:, None]).sum(0)
            C[:, d] = cnt * self.fact[K - m]; seen += cnt
        unused = [v for v in range(K) if v not in present]
        if unused:
            new = (nok - seen) * self.fact[K - m - 1]
            for v in unused:
                C[:, v] = new
        total = nok * self.fact[K - m]
        P = C / total
        P[s >= 0] = 0.0
        return total, P

    def solutions(self, s, cap=100000):
        """All completions of s (as an array [n, N]); only when the count is <= cap."""
        K = self.K
        ok, E, present = self._consistent(s)
        m = len(present); unused = [v for v in range(K) if v not in present]
        if int(ok.sum()) * self.fact[K - m] > cap:
            return None
        out = []
        idx = np.where(ok)[0]
        for g in idx:
            base = self.B[g]
            mp = -np.ones(K, dtype=np.int64)
            for e, d in zip(E, present):
                mp[e[g]] = d
            free = [b for b in range(K) if mp[b] < 0]
            for perm in itertools.permutations(unused):
                mp2 = mp.copy()
                for b, v in zip(free, perm):
                    mp2[b] = v
                out.append(mp2[base])
        return np.array(out, dtype=np.int64).reshape(-1, self.N)


def make_puzzle(sol, counter, rng, target=0):
    """Remove clues in random order while the solution stays unique. Stop at `target` clues (0 = until minimal)."""
    s = sol.astype(np.int64).copy()
    for i in rng.permutation(len(s)):
        if target and (s >= 0).sum() <= target:
            break
        v = s[i]; s[i] = -1
        if counter.count(s) != 1:
            s[i] = v
    return s


def rate(puz, K):
    """Difficulty rating, for stratifying results only. 0: naked singles suffice; 1: naked + hidden singles
    suffice; 2: singles do not finish the puzzle ('hard')."""
    N, row, col, box = geom(K)
    P = peers(K)
    units = [np.where(g == u)[0] for g in (row, col, box) for u in range(K)]

    def run(use_hidden):
        s = puz.copy()
        while True:
            cand = np.ones((N, K), bool)
            for i in np.where(s >= 0)[0]:
                cand[P[i], s[i]] = False
            progress = False
            for i in np.where(s < 0)[0]:
                if cand[i].sum() == 1:
                    s[i] = int(np.argmax(cand[i])); progress = True; break
            if progress:
                continue
            if use_hidden:
                for u in units:
                    for v in range(K):
                        if v in s[u]:
                            continue
                        pos = [i for i in u if s[i] < 0 and cand[i, v]]
                        if len(pos) == 1:
                            s[pos[0]] = v; progress = True; break
                    if progress:
                        break
            if not progress:
                return (s >= 0).all()

    if run(False):
        return 0
    if run(True):
        return 1
    return 2


def local_candidates(S, K):
    """S: [m, N] partial states (-1 empty). Returns [m, N, K] bool: value not used by any filled peer."""
    P = peers(K)
    S = np.asarray(S); m, N = S.shape
    used = np.zeros((m, N, K), bool)
    for v in range(K):
        has = (S == v)                                   # [m, N]
        used[:, :, v] = (has[:, None, :] & P[None, :, :]).any(2)
    return ~used


def random_grid_dfs(K, rng):
    """A random valid grid by randomised backtracking (NOT uniform). Used only for K = 9."""
    N, row, col, box = geom(K)
    P = [np.where(p)[0].tolist() for p in peers(K)]
    x = [-1] * N

    def rec(i):
        if i == N:
            return True
        used = {x[j] for j in P[i]}
        for v in rng.permutation(K).tolist():
            if v not in used:
                x[i] = v
                if rec(i + 1):
                    return True
                x[i] = -1
        return False

    rec(0)
    return np.array(x, dtype=np.int64)


def count_solutions_dfs(puz, K, cap=2):
    """Number of solutions (up to cap) by backtracking with most-constrained-cell choice. For K = 9 generation."""
    N, row, col, box = geom(K)
    P = [np.where(p)[0].tolist() for p in peers(K)]
    x = puz.tolist()

    def rec():
        best, bestc = None, None
        for i in range(N):
            if x[i] < 0:
                used = {x[j] for j in P[i]}
                c = [v for v in range(K) if v not in used]
                if best is None or len(c) < len(bestc):
                    best, bestc = i, c
                    if len(c) <= 1:
                        break
        if best is None:
            return 1
        tot = 0
        for v in bestc:
            x[best] = v; tot += rec(); x[best] = -1
            if tot >= cap:
                break
        return tot

    return rec()
