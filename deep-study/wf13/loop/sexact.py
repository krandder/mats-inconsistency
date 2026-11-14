"""sexact.py -- EVALUATION CODE ONLY: all solutions of a 6x6 puzzle under the TRUE rules, by an independent
depth-first search (fewest options first). No arm uses a number from here except to be scored after the fact."""
import numpy as np

K, N = 6, 36
_r, _c = np.arange(N) // K, np.arange(N) % K
_b = (_r // 2) * 2 + _c // 3
PEERS = [np.flatnonzero(((_r == _r[i]) | (_c == _c[i]) | (_b == _b[i])) & (np.arange(N) != i)) for i in range(N)]


def solutions(puz, cap=100000):
    b = [int(v) for v in puz]
    out = []

    def opts(i):
        used = {b[p] for p in PEERS[i]}
        return [v for v in range(K) if v not in used]

    def rec():
        if len(out) >= cap:
            return
        best, bo = -1, None
        for i in range(N):
            if b[i] < 0:
                o = opts(i)
                if bo is None or len(o) < len(bo):
                    best, bo = i, o
                    if len(o) <= 1:
                        break
        if best < 0:
            out.append(list(b))
            return
        for v in bo:
            b[best] = v
            rec()
        b[best] = -1
    for i in range(N):                                   # clues that collide: no solution
        if b[i] >= 0 and any(b[p] == b[i] for p in PEERS[i]):
            return np.zeros((0, N), dtype=np.int64)
    rec()
    return np.array(out, dtype=np.int64).reshape(-1, N)


def marginals(sols):
    return np.stack([(sols == v).mean(0) for v in range(K)], -1)          # [36, 6]
