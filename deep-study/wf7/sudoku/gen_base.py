"""gen_base.py -- enumerate grids, compute essentially-different classes (orbits), make orbit-disjoint splits,
self-test the exact counter, and print generator statistics from a puzzle pilot (no model involved).
Run: nice -n 19 ionice -c3 python -u gen_base.py > out/gen_base.out
"""
import time, sys
import numpy as np
import oracle as O

T0 = time.time()
rng = np.random.default_rng(20261110)

# ---------------------------------------------------------------- 4x4
B4 = O.enumerate_base(4)
ids4, n4 = O.orbit_ids(B4, 4)
G4 = O.all_relabelings(B4, 4)                       # [24, 12, 16] flattened -> [288, 16]
orb4 = np.tile(ids4, 24)
assert len(G4) == 288 and O.is_valid_batch(G4, 4).all() and len(np.unique(G4, axis=0)) == 288
print(f"4x4: base grids {len(B4)}, all grids {len(G4)}, orbits {n4}, orbit sizes {np.bincount(orb4).tolist()}")
np.savez("data/k4.npz", B=B4, G=G4, orbit=orb4, base_orbit=ids4)

# ---------------------------------------------------------------- 6x6
t = time.time()
B6 = O.enumerate_base(6)
print(f"6x6: base grids {len(B6)} (x 720 = {len(B6)*720}) ({time.time()-t:.0f}s)")
assert O.is_valid_batch(B6[:2000], 6).all()
t = time.time()
ids6, n6 = O.orbit_ids(B6, 6)
sizes = np.bincount(ids6) * 720
print(f"6x6: orbits {n6}; orbit sizes min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()} ({time.time()-t:.0f}s)")

# orbit-disjoint split: shuffle orbits with a fixed seed; test ~20% of grids, val ~10%, rest train
order = np.random.default_rng(606).permutation(n6)
tot = sizes.sum(); split = np.zeros(n6, dtype=np.int8); acc = 0
for o in order:
    if acc < 0.20 * tot:
        split[o] = 2
    elif acc < 0.30 * tot:
        split[o] = 1
    acc += sizes[o]
base_split = split[ids6]
for name, k in (("train", 0), ("val", 1), ("test", 2)):
    print(f"   {name}: orbits {(split==k).sum()}, base grids {(base_split==k).sum()}, grids {(base_split==k).sum()*720} "
          f"({(base_split==k).mean():.3f})")
np.savez("data/k6_base.npz", B=B6, base_orbit=ids6, base_split=base_split, orbit_split=split)

# ---------------------------------------------------------------- counter self-test
C6 = O.Counter(B6, 6)
empty = -np.ones(36, dtype=np.int64)
n0, P0 = C6.conditionals(empty)
print(f"counter: completions of the empty grid {n0}; conditional at empty: min {P0.min():.4f} max {P0.max():.4f}")
assert n0 == 28200960
bad = 0
for trial in range(200):
    g = B6[rng.integers(len(B6))].astype(np.int64); g = rng.permutation(6)[g]
    k = int(rng.integers(8, 30)); s = -np.ones(36, dtype=np.int64); idx = rng.choice(36, k, replace=False); s[idx] = g[idx]
    n, P = C6.conditionals(s)
    emp = np.where(s < 0)[0]
    assert np.allclose(P[emp].sum(1), 1.0) and P[emp, g[emp]].min() > 0
    if k >= 14:                                        # cross-check against backtracking
        nd = O.count_solutions_dfs(s, 6, cap=10 ** 9)
        bad += int(nd != n)
        i = emp[0]
        for v in range(6):
            s2 = s.copy(); s2[i] = v
            assert abs(C6.count(s2) / n - P[i, v]) < 1e-12
print(f"counter: 200 random states, conditionals normalised, mismatches against backtracking count: {bad}")
assert bad == 0
C4 = O.Counter(B4, 4)
assert C4.count(-np.ones(16, dtype=np.int64)) == 288

# ---------------------------------------------------------------- puzzle pilot (generator statistics only)
t = time.time()
tb = np.where(base_split == 0)[0]
clues, rates = [], []
for trial in range(400):
    g = rng.permutation(6)[B6[tb[rng.integers(len(tb))]].astype(np.int64)]
    p = O.make_puzzle(g, C6, rng, target=0)
    clues.append(int((p >= 0).sum())); rates.append(O.rate(p, 6))
clues = np.array(clues); rates = np.array(rates)
print(f"pilot: 400 minimal 6x6 puzzles ({(time.time()-t)/400*1000:.0f} ms each)")
print("   clue count histogram:", {int(c): int((clues == c).sum()) for c in np.unique(clues)})
print("   rating histogram (0 naked singles, 1 +hidden singles, 2 hard):", np.bincount(rates, minlength=3).tolist())
for c in np.unique(clues):
    print(f"   clues {c}: n {int((clues==c).sum())}, hard fraction {np.mean(rates[clues==c]==2):.2f}")
for target in (12, 14, 16, 18):
    rr = []
    for trial in range(150):
        g = rng.permutation(6)[B6[tb[rng.integers(len(tb))]].astype(np.int64)]
        p = O.make_puzzle(g, C6, rng, target=target)
        rr.append((int((p >= 0).sum()), O.rate(p, 6)))
    rr = np.array(rr)
    print(f"   target {target}: clue counts {dict(zip(*[a.tolist() for a in np.unique(rr[:,0], return_counts=True)]))}, "
          f"rating histogram {np.bincount(rr[:,1], minlength=3).tolist()}")
# 4x4 minimal puzzles
G4i = G4.astype(np.int64); cl4 = []; r4 = []
for trial in range(300):
    p = O.make_puzzle(G4i[rng.integers(288)], C4, rng, 0); cl4.append(int((p >= 0).sum())); r4.append(O.rate(p, 4))
print("4x4 minimal puzzles: clue histogram", dict(zip(*[a.tolist() for a in np.unique(cl4, return_counts=True)])),
      "rating histogram", np.bincount(r4, minlength=3).tolist())
print(f"total {time.time()-T0:.0f}s")
