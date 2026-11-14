"""E-SUD round 3 hard split (gemini spec, overnight-followups-round-3.md Rank 1). 6x6, givens k in [10, 15].
Unsolvable: 1-2 givens changed, no direct peer clash, zero completions, AND prop_depth >= 3 (naked + hidden singles
need >= 3 rounds to expose the contradiction, or stall: 999).
Solvable: 2..400 completions AND dead_end_rate >= 0.25 (10 random-MRV rollouts; a rollout picks an MRV cell, ties at
random, fills a uniformly random legal digit, and dead-ends when some empty cell has no legal digit).
50/50 by construction. Usage: r3hard.py OUT.npz N SEED NPROC"""
import sys, random
import numpy as np
from boards import solve_all, peer_conflict, random_grid, geom

box = geom(6)[2]
PEERS = [set(j for j in range(36) if j != i and (j // 6 == i // 6 or j % 6 == i % 6 or box[j] == box[i])) for i in range(36)]
UNITS = ([[r * 6 + c for c in range(6)] for r in range(6)] + [[r * 6 + c for r in range(6)] for c in range(6)]
         + [[i for i in range(36) if box[i] == b] for b in range(6)])


def prop_depth(board):  # verbatim from the spec
    g = list(board)
    for step in range(1, 25):
        moves = []
        for i in range(36):
            if g[i] != 0: continue
            used = {g[j] for j in PEERS[i] if g[j] != 0}
            if len(used) == 6: return step
            if len(used) == 5: moves.append((i, (set(range(1, 7)) - used).pop()))
        for unit in UNITS:
            for d in range(1, 7):
                if any(g[i] == d for i in unit): continue
                places = [i for i in unit if g[i] == 0 and all(g[j] != d for j in PEERS[i])]
                if len(places) == 0: return step
                if len(places) == 1: moves.append((places[0], d))
        if not moves: return 999
        for i, d in moves:
            if g[i] != 0 and g[i] != d: return step
            g[i] = d
        for i in range(36):
            if g[i]:
                for j in PEERS[i]:
                    if j > i and g[j] == g[i]: return step
    return 999


def dead_end_rate(board, rng, trials=10):
    dead = 0
    for _ in range(trials):
        g = list(board)
        while True:
            empt = [i for i in range(36) if g[i] == 0]
            if not empt:
                break
            cands = {i: [d for d in range(1, 7) if all(g[j] != d for j in PEERS[i])] for i in empt}
            mn = min(len(c) for c in cands.values())
            if mn == 0:
                dead += 1; break
            i = rng.choice([i for i in empt if len(cands[i]) == mn])
            g[i] = rng.choice(cands[i])
    return dead / trials


def gen(n_each, seed):
    rng = random.Random(seed); pos, neg = [], []
    tries = dict(pos=0, neg=0)
    while len(pos) < n_each or len(neg) < n_each:
        k = rng.randint(10, 15)
        grid = random_grid(6, rng); idx = rng.sample(range(36), k)
        b = [0] * 36
        for i in idx:
            b[i] = grid[i]
        if len(pos) < n_each:
            tries['pos'] += 1
            sols = solve_all(b, 6, 400)
            if 2 <= len(sols) <= 400:
                der = dead_end_rate(b, rng)
                if der >= 0.25:
                    S = np.array(sols)
                    pos.append(dict(board=b, solvable=1, k=k, nsol=len(sols), pdepth=prop_depth(b), der=der,
                                    outcome=S[rng.randrange(len(S))].tolist()))
        if len(neg) < n_each:
            tries['neg'] += 1
            bb = list(b)
            for i in rng.sample(idx, rng.choice([1, 2])):
                bb[i] = rng.choice([d for d in range(1, 7) if d != b[i]])
            if not peer_conflict(bb, 6) and not solve_all(bb, 6, 0):
                pdp = prop_depth(bb)
                if pdp >= 3:
                    neg.append(dict(board=bb, solvable=0, k=k, nsol=0, pdepth=pdp, der=-1.0, outcome=grid))
    return pos + neg, tries


if __name__ == '__main__':
    from multiprocessing import Pool
    out, n, seed, nj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    each = n // 2
    with Pool(nj) as p:
        parts = p.starmap(gen, [(each // nj + (j < each % nj), seed * 1000 + j) for j in range(nj)])
    rows = [r for q, _ in parts for r in q]
    tries = {k: sum(t[k] for _, t in parts) for k in ['pos', 'neg']}
    rng = np.random.default_rng(seed); rng.shuffle(rows)
    np.savez_compressed(out, board=np.array([r['board'] for r in rows], np.int8), solvable=np.array([r['solvable'] for r in rows], np.int8),
                        k=np.array([r['k'] for r in rows], np.int16), nsol=np.array([r['nsol'] for r in rows], np.int32),
                        pdepth=np.array([r['pdepth'] for r in rows], np.int16), der=np.array([r['der'] for r in rows], np.float32),
                        outcome=np.array([r['outcome'] for r in rows], np.int8))
    print(out, len(rows), 'candidates tried', tries)
