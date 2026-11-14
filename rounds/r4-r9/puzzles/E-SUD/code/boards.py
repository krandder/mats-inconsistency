"""E-SUD boards: 4x4 and 6x6 Sudoku boards with several solutions (exact cell marginals) and unsolvable boards.

The generator (the environment) uses Sudoku rules; the MODEL never sees them. Cells r*N+c, digits 1..N, 0 = empty.
Stored per board: the solvable bit and `outcome`: for solvable boards one completion drawn uniformly from all
completions; for unsolvable boards the original grid before the clues were corrupted (used only to reveal extra cells
in training states, never as a cell target). These outcomes are the only training targets. The exact marginals (mean one-hot over all completions) are
kept for scoring only.
"""
import numpy as np, random, sys

GEOM = {4: (2, 2), 6: (2, 3)}


def geom(N):
    br, bc = GEOM[N]
    box = [(i // N // br) * (N // bc) + (i % N) // bc for i in range(N * N)]
    return br, bc, box


def solve_all(g, N, cap):
    """All completions of board g (list of N*N), stops after cap+1. Bitmask DFS with MRV."""
    br, bc, box = geom(N)
    C = N * N
    full = (1 << N) - 1
    R = [0] * N; Cm = [0] * N; B = [0] * N
    g = list(g)
    for i, d in enumerate(g):
        if d:
            b = 1 << (d - 1); r, c = divmod(i, N)
            if (R[r] | Cm[c] | B[box[i]]) & b:
                return []
            R[r] |= b; Cm[c] |= b; B[box[i]] |= b
    sols = []

    def rec(empt):
        if len(sols) > cap:
            return
        if not empt:
            sols.append(g.copy()); return
        best = None; bm = 0; bn = 99
        for i in empt:
            r, c = divmod(i, N)
            m = full & ~(R[r] | Cm[c] | B[box[i]])
            n = bin(m).count('1')
            if n < bn:
                best, bm, bn = i, m, n
                if n <= 1:
                    break
        if bn == 0:
            return
        rest = [j for j in empt if j != best]
        r, c = divmod(best, N); bx = box[best]
        m = bm
        while m:
            b = m & -m; m ^= b
            R[r] |= b; Cm[c] |= b; B[bx] |= b
            g[best] = b.bit_length()
            rec(rest)
            g[best] = 0
            R[r] ^= b; Cm[c] ^= b; B[bx] ^= b
    rec([i for i in range(C) if not g[i]])
    return sols


def valid_full(g, N):
    """Environment verifier: a full grid satisfies every row/col/box (used only as an outcome check)."""
    return 0 not in g and len(solve_all(g, N, 0)) == 1


def peer_conflict(g, N):
    br, bc, box = geom(N)
    for i in range(N * N):
        if not g[i]:
            continue
        for j in range(i + 1, N * N):
            if g[j] == g[i] and (i // N == j // N or i % N == j % N or box[i] == box[j]):
                return True
    return False


def random_grid(N, rng):
    """A random full grid: random seed digits completed by DFS, then random relabel and row/col permutations."""
    br, bc, box = geom(N)
    while True:
        g = [0] * (N * N)
        idx = list(range(N * N)); rng.shuffle(idx)
        for i in idx[:N]:
            g[i] = rng.randint(1, N)
            if peer_conflict(g, N):
                g[i] = 0
        s = solve_all(g, N, 0)
        if s:
            g = s[0]; break
    g = np.array(g).reshape(N, N)
    perm = np.array([0] + rng.sample(range(1, N + 1), N))
    g = perm[g]
    nb, ns = N // br, N // bc
    rows = [b * br + r for b in rng.sample(range(nb), nb) for r in rng.sample(range(br), br)]
    cols = [s * bc + c for s in rng.sample(range(ns), ns) for c in rng.sample(range(bc), bc)]
    return g[rows][:, cols].reshape(-1).tolist()


def make_board(N, rng, kgiv, cap, want_solvable):
    """Solvable: givens from a random grid with 2..cap completions (ambiguous on purpose).
    Unsolvable: the same kind of board (same clue count) with 1 or 2 givens changed so that there is no direct peer
    clash but zero completions (it looks locally fine; the contradiction needs search). The original grid is kept:
    training states reveal its cells on top of the corrupted board (still unsolvable; the clash becomes local)."""
    grid = random_grid(N, rng)
    idx = rng.sample(range(N * N), kgiv)
    b = [0] * (N * N)
    for i in idx:
        b[i] = grid[i]
    if want_solvable:
        sols = solve_all(b, N, cap)
        if len(sols) < 2 or len(sols) > cap:
            return None
        return {"board": b, "solvable": 1, "sols": sols, "grid": grid}
    ncor = rng.choice([1, 2])
    for _ in range(30):
        bb = list(b)
        for i in rng.sample(idx, ncor):
            bb[i] = rng.choice([d for d in range(1, N + 1) if d != b[i]])
        if peer_conflict(bb, N):
            continue
        if not solve_all(bb, N, 0):
            return {"board": bb, "solvable": 0, "sols": [], "grid": grid}
    return None


def gen_split(N, n, seed, kgiv_range, cap, frac_pos=0.5):
    rng = random.Random(seed)
    boards, solv, outc, marg, nsol = [], [], [], [], []
    while len(boards) < n:
        want = rng.random() < frac_pos
        r = make_board(N, rng, rng.randint(*kgiv_range), cap, want)
        if r is None:
            continue
        boards.append(r["board"]); solv.append(r["solvable"])
        if r["solvable"]:
            S = np.array(r["sols"])
            outc.append(S[rng.randrange(len(S))])
            marg.append(np.stack([(S == d).mean(0) for d in range(1, N + 1)], -1)); nsol.append(len(S))
        else:
            outc.append(np.array(r["grid"])); marg.append(np.zeros((N * N, N))); nsol.append(0)
    return dict(board=np.array(boards, np.int8), solvable=np.array(solv, np.int8), outcome=np.array(outc, np.int8),
                marg=np.array(marg, np.float32), nsol=np.array(nsol, np.int32))


CFG = {4: dict(kgiv=(3, 8), cap=10 ** 6), 6: dict(kgiv=(10, 20), cap=400)}

if __name__ == "__main__":
    N, n, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    from multiprocessing import Pool
    nj = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    with Pool(nj) as p:
        parts = p.starmap(gen_split, [(N, n // nj + (j < n % nj), seed * 1000 + j, CFG[N]["kgiv"], CFG[N]["cap"]) for j in range(nj)])
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    if 'train' in out:
        d.pop('marg')  # training files carry no exact posterior
    np.savez_compressed(out, **d)
    print(N, n, seed, "pos", int(d["solvable"].sum()), "mean nsol(pos)", float(d["nsol"][d["solvable"] == 1].mean()))
