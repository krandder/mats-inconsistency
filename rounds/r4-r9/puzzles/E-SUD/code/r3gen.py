"""E-SUD round 3 board pool: 6x6 boards over a wide clue-count range (solvable with >= 1 completion, and unsolvable
boards made the same way as before: 1-2 givens changed, no direct peer clash, zero completions). Only the solvable
bit is needed (round 3 scores Solvable only); nsol is counted up to a cap for description.
Usage: r3gen.py OUT.npz N_PER_K SEED KMIN KMAX [NPROC]"""
import sys, random
import numpy as np
from boards import solve_all, peer_conflict, random_grid

N = 6


def one(rng, k, want):
    grid = random_grid(N, rng)
    idx = rng.sample(range(N * N), k)
    b = [0] * (N * N)
    for i in idx:
        b[i] = grid[i]
    if want:
        s = solve_all(b, N, 200)
        return dict(board=b, solvable=1, nsol=len(s), k=k)
    ncor = rng.choice([1, 2])
    for _ in range(30):
        bb = list(b)
        for i in rng.sample(idx, ncor):
            bb[i] = rng.choice([d for d in range(1, N + 1) if d != b[i]])
        if peer_conflict(bb, N):
            continue
        if not solve_all(bb, N, 0):
            return dict(board=bb, solvable=0, nsol=0, k=k)
    return None


def gen(n_per_k, seed, kmin, kmax):
    rng = random.Random(seed); out = []
    for k in range(kmin, kmax + 1):
        for want in (1, 0):
            got = 0
            while got < n_per_k:
                r = one(rng, k, want)
                if r is not None:
                    out.append(r); got += 1
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    out, n, seed, kmin, kmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    nj = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    with Pool(nj) as p:
        parts = p.starmap(gen, [(n // nj + (j < n % nj), seed * 1000 + j, kmin, kmax) for j in range(nj)])
    rows = [r for q in parts for r in q]
    np.savez_compressed(out, board=np.array([r['board'] for r in rows], np.int8), solvable=np.array([r['solvable'] for r in rows], np.int8),
                        nsol=np.array([r['nsol'] for r in rows], np.int32), k=np.array([r['k'] for r in rows], np.int16))
    print(out, len(rows))
