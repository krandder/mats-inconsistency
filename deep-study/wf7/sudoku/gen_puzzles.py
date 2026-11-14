"""gen_puzzles.py -- puzzle sets for 6x6 (and 4x4), from orbit-disjoint grid splits and disjoint seeds.
Usage: python -u gen_puzzles.py train|evalsets
  train    : train_min (12000 minimal puzzles, train orbits, seed 1001), train_12p (4000 puzzles with 12-16 clues,
             train orbits, seed 1002)
  evalsets : val_min (1000 minimal, val orbits, seed 2001); test_min (minimal, test orbits, seed 3001; generated
             until 300 'hard' ones exist or 8000 puzzles); test_12p (500 with 12-16 clues, test orbits, seed 3002);
             test_multi (200 puzzles with 2..12 solutions, test orbits, seed 3003); 4x4 minimal puzzles (seed 4001)
"""
import sys, time
import numpy as np
import oracle as O

mode = sys.argv[1]
d = np.load("data/k6_base.npz"); B6 = d["B"]; bs = d["base_split"]
C6 = O.Counter(B6, 6)
T0 = time.time()


def draw_grid(rng, split):
    idx = np.where(bs == split)[0]
    return rng.permutation(6)[B6[idx[rng.integers(len(idx))]].astype(np.int64)]


def gen(n, split, seed, target_fn, rate_it=True, stop_hard=None, cap=None, tag=""):
    rng = np.random.default_rng(seed); P, S, R = [], [], []
    hard = 0
    while len(P) < (cap or n):
        g = draw_grid(rng, split); p = O.make_puzzle(g, C6, rng, target=target_fn(rng))
        P.append(p); S.append(g); r = O.rate(p, 6) if rate_it else -1; R.append(r); hard += int(r == 2)
        if len(P) % 500 == 0:
            print(f"  {tag}: {len(P)} puzzles, hard {hard} ({time.time()-T0:.0f}s)", flush=True)
        if stop_hard is not None and hard >= stop_hard and len(P) >= n:
            break
    return np.array(P), np.array(S), np.array(R)


if mode == "train":
    P, S, R = gen(12000, 0, 1001, lambda r: 0, rate_it=False, tag="train_min")
    np.savez("data/k6_train_min.npz", puz=P, sol=S)
    P, S, R = gen(4000, 0, 1002, lambda r: int(r.integers(12, 17)), rate_it=False, tag="train_12p")
    np.savez("data/k6_train_12p.npz", puz=P, sol=S)
    print("clue counts train_12p:", np.bincount((P >= 0).sum(1)).tolist())
else:
    P, S, R = gen(1000, 1, 2001, lambda r: 0, tag="val_min")
    np.savez("data/k6_val_min.npz", puz=P, sol=S, rating=R)
    P, S, R = gen(1000, 2, 3001, lambda r: 0, stop_hard=300, cap=8000, tag="test_min")
    np.savez("data/k6_test_min.npz", puz=P, sol=S, rating=R)
    print("test_min:", len(P), "ratings", np.bincount(R, minlength=3).tolist(), "clues", np.bincount((P >= 0).sum(1)).tolist())
    P, S, R = gen(500, 2, 3002, lambda r: int(r.integers(12, 17)), tag="test_12p")
    np.savez("data/k6_test_12p.npz", puz=P, sol=S, rating=R)
    # puzzles with several solutions: remove one or two more clues from a minimal puzzle
    rng = np.random.default_rng(3003); MP, MS, MN = [], [], []
    while len(MP) < 200:
        g = draw_grid(rng, 2); p = O.make_puzzle(g, C6, rng, 0)
        for _ in range(int(rng.integers(1, 3))):
            cl = np.where(p >= 0)[0]; p[cl[rng.integers(len(cl))]] = -1
        n = C6.count(p)
        if 2 <= n <= 12:
            MP.append(p); MS.append(g); MN.append(n)
    np.savez("data/k6_test_multi.npz", puz=np.array(MP), sol=np.array(MS), nsol=np.array(MN))
    print("test_multi: solution-count histogram", np.bincount(MN).tolist())
    # 4x4
    d4 = np.load("data/k4.npz"); G4 = d4["G"].astype(np.int64); orb = d4["orbit"]; C4 = O.Counter(d4["B"], 4)
    rng = np.random.default_rng(4001); P4, S4, O4 = [], [], []
    for o in (0, 1):
        idx = np.where(orb == o)[0]
        for g in G4[idx]:
            for rep in range(2):
                P4.append(O.make_puzzle(g, C4, rng, 0)); S4.append(g); O4.append(o)
    np.savez("data/k4_puzzles.npz", puz=np.array(P4), sol=np.array(S4), orbit=np.array(O4))
    print("4x4 puzzles:", len(P4), "clues", np.bincount((np.array(P4) >= 0).sum(1)).tolist())
print(f"done {mode} ({time.time()-T0:.0f}s)")
