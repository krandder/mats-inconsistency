"""P10 grids: Sudoku geometry, 4x4 enumeration + exact posteriors, 6x6 unique-puzzle
generator, singles/DFS labeller, atoms, symmetry canonicaliser, streaming Sudoku-Extreme loader.

Cells are indexed r*N+c. Digits 1..N, 0 = empty/masked.
"""
import itertools
import random
import numpy as np


class Geom:
    def __init__(self, N, br, bc):
        self.N, self.br, self.bc = N, br, bc
        self.C = N * N
        C = self.C
        self.box = np.array([(i // N // br) * (N // bc) + (i % N) // bc for i in range(C)])
        self.row = np.array([i // N for i in range(C)])
        self.col = np.array([i % N for i in range(C)])
        self.peers = [sorted(j for j in range(C) if j != i and (self.row[j] == self.row[i] or self.col[j] == self.col[i]
                                                                  or self.box[j] == self.box[i])) for i in range(C)]
        self.peer_mat = np.zeros((C, C), bool)
        for i in range(C):
            self.peer_mat[i, self.peers[i]] = True
        self.units = ([[r * N + c for c in range(N)] for r in range(N)] + [[r * N + c for r in range(N)] for c in range(N)]
                      + [[i for i in range(C) if self.box[i] == b] for b in range(N)])


G4 = Geom(4, 2, 2)
G6 = Geom(6, 2, 3)
G9 = Geom(9, 3, 3)


# ---------------------------------------------------------------- DFS / singles
def solve(g, geom, limit=2, rng=None):
    """DFS with true-candidate MRV. Returns up to `limit` solutions (lists)."""
    g = list(g)
    N, C, peers = geom.N, geom.C, geom.peers
    sols = []

    def rec():
        if len(sols) >= limit:
            return
        best, bc = None, None
        for i in range(C):
            if g[i] == 0:
                used = {g[j] for j in peers[i]}
                cand = [d for d in range(1, N + 1) if d not in used]
                if not cand:
                    return
                if best is None or len(cand) < len(bc):
                    best, bc = i, cand
                    if len(cand) == 1:
                        break
        if best is None:
            sols.append(g.copy())
            return
        if rng:
            rng.shuffle(bc)
        for d in bc:
            g[best] = d
            rec()
            g[best] = 0
    rec()
    return sols


def singles(g, geom, max_rounds=99):
    """Parallel naked+hidden singles rounds.
    Returns dict: depth (rounds used), solved flag, force_round per cell (0 = given, -1 = never),
    naked[r-1][cell], hidden[r-1][cell] booleans for the round in which the cell was fixed."""
    N, C, peers, units = geom.N, geom.C, geom.peers, geom.units
    known = {i: g[i] for i in range(C) if g[i]}
    force = np.full(C, -1, np.int16)
    for i in known:
        force[i] = 0
    naked, hidden = [], []
    rounds = 0
    while len(known) < C and rounds < max_rounds:
        new, nk, hd = {}, np.zeros(C, bool), np.zeros(C, bool)
        cands = {}
        for i in range(C):
            if i in known:
                continue
            cand = set(range(1, N + 1)) - {known[j] for j in peers[i] if j in known}
            cands[i] = cand
            if len(cand) == 1:
                new[i] = next(iter(cand))
                nk[i] = True
        for u in units:
            present = {known[i] for i in u if i in known}
            for d in range(1, N + 1):
                if d in present:
                    continue
                pos = [i for i in u if i not in known and d in cands[i]]
                if len(pos) == 1:
                    new[pos[0]] = d
                    hd[pos[0]] = True
        if not new:
            break
        rounds += 1
        for i in new:
            force[i] = rounds
        known.update(new)
        naked.append(nk)
        hidden.append(hd)
    return {"depth": rounds, "solved": len(known) == C, "force": force, "naked": naked, "hidden": hidden}


# ---------------------------------------------------------------- 4x4
def enum4():
    """All 288 4x4 Sudoku grids as a (288,16) uint8 table (digits 1..4)."""
    sols = solve([0] * 16, G4, limit=10 ** 6)
    t = np.array(sols, np.uint8)
    assert t.shape == (288, 16)
    return t


def posterior4(table, vis_vals, vis_mask):
    """Exact posterior given visible cells. vis_vals (B,16) digits, vis_mask (B,16) bool.
    Returns cons (B,288) bool, post (B,16,4) probabilities, n (B,) consistent count."""
    eq = (table[None, :, :] == vis_vals[:, None, :]) | (~vis_mask[:, None, :])
    cons = eq.all(-1)
    n = cons.sum(1)
    onehot = np.eye(4, dtype=np.float32)[table.astype(np.int64) - 1]  # (288,16,4)
    post = np.einsum('bg,gcd->bcd', cons.astype(np.float32), onehot) / n[:, None, None]
    return cons, post, n


def pairwise_mi4(table, cons_row):
    """Exact I(z_c; z_q | X) for all cell pairs, from the uniform posterior over consistent grids."""
    S = table[cons_row].astype(np.int64) - 1  # (n,16)
    n = len(S)
    oh = np.eye(4)[S]  # (n,16,4)
    pc = oh.mean(0)  # (16,4)
    joint = np.einsum('ncd,nqe->cqde', oh, oh) / n  # (16,16,4,4)
    with np.errstate(divide='ignore', invalid='ignore'):
        Hc = -(np.where(pc > 0, pc * np.log(pc), 0)).sum(-1)
        Hj = -(np.where(joint > 0, joint * np.log(joint), 0)).sum((-1, -2))
    mi = Hc[:, None] + Hc[None, :] - Hj
    np.fill_diagonal(mi, 0)
    return np.maximum(mi, 0), Hc


# ---------------------------------------------------------------- symmetries
def sym_perms(geom, transpose=False):
    """All cell permutations from row swaps within bands, band swaps, column swaps within stacks,
    stack swaps (and transpose if allowed and square boxes). Returns (P, C) int array: new[i] = old[perm[i]]."""
    N, br, bc = geom.N, geom.br, geom.bc
    nb, ns = N // br, N // bc  # number of bands (groups of br rows), stacks (groups of bc cols)
    row_orders = []
    for bperm in itertools.permutations(range(nb)):
        for inner in itertools.product(*[list(itertools.permutations(range(br)))] * nb):
            row_orders.append([bperm[b] * br + inner[b][k] for b in range(nb) for k in range(br)])
    col_orders = []
    for sperm in itertools.permutations(range(ns)):
        for inner in itertools.product(*[list(itertools.permutations(range(bc)))] * ns):
            col_orders.append([sperm[s] * bc + inner[s][k] for s in range(ns) for k in range(bc)])
    perms = []
    for ro in row_orders:
        for co in col_orders:
            perms.append([ro[i // N] * N + co[i % N] for i in range(N * N)])
    P = np.array(perms, np.int16)
    if transpose and br == bc:
        tr = np.array([(i % N) * N + i // N for i in range(N * N)])
        P = np.concatenate([P, P[:, tr]], 0)
    return P


def relabel_first_occurrence(G, N):
    """Relabel digits so they appear in first-occurrence order 1..N (row-major). G (P,C) with digits 1..N."""
    P, C = G.shape
    out = np.zeros_like(G)
    # position of first occurrence of each digit
    first = np.full((P, N + 1), C, np.int64)
    for d in range(1, N + 1):
        hit = G == d
        first[:, d] = np.where(hit.any(1), hit.argmax(1), C)
    order = np.argsort(first[:, 1:], 1)  # order[p, k] = digit-1 that is k-th to appear
    mapping = np.zeros((P, N + 1), G.dtype)
    rows = np.arange(P)[:, None]
    mapping[rows, order + 1] = np.arange(1, N + 1)[None, :]
    out = mapping[np.arange(P)[:, None], G]
    return out


def canonical(sol, geom, P):
    """Canonical form of a full solution grid under the group P (cell perms) x digit relabelling."""
    G = np.asarray(sol).astype(np.uint8)[P]  # (P,C)
    G = relabel_first_occurrence(G, geom.N)
    idx = np.lexsort(G.T[::-1])[0]
    return G[idx].tobytes()


# ---------------------------------------------------------------- 6x6 generator
def gen6_one(seed):
    rng = random.Random(seed)
    full = solve([0] * 36, G6, 1, rng)[0]
    g = full.copy()
    order = list(range(36))
    rng.shuffle(order)
    target = rng.randint(8, 16)
    for i in order:
        if sum(1 for x in g if x) <= target:
            break
        v = g[i]
        g[i] = 0
        if len(solve(g, G6, 2)) != 1:
            g[i] = v
    s = singles(g, G6)
    nr = 3
    nk = np.zeros((nr, 36), bool)
    hd = np.zeros((nr, 36), bool)
    for r in range(min(nr, len(s["naked"]))):
        nk[r] = s["naked"][r]
        hd[r] = s["hidden"][r]
    return (np.array(g, np.uint8), np.array(full, np.uint8), s["depth"], s["solved"], s["force"], nk, hd)


# ---------------------------------------------------------------- atoms (S6)
def atom_specs(geom):
    """Atom library. Each atom: (kind, cellish, params). kinds:
    'bl_row' (box b, digit d, row r): digit d of box b lies in row r (non-cell)
    'bl_col' (box b, digit d, col c)                                   (non-cell)
    'naked' (cell i, round r), 'hidden' (cell i, round r)              (non-cell)
    'forced' (cell i, round r): cell i forced by round r (1<=force<=r) (non-cell)
    'eq' (cell i, digit d): y_i = d (cell)      'le3' (cell i): y_i <= 3 (cell)"""
    N = geom.N
    specs = []
    for b in range(N):
        rows = sorted({int(geom.row[i]) for i in range(geom.C) if geom.box[i] == b})
        cols = sorted({int(geom.col[i]) for i in range(geom.C) if geom.box[i] == b})
        for d in range(1, N + 1):
            for r in rows[:-1] if len(rows) == 2 else rows:
                specs.append(('bl_row', b, d, r))
            for c in cols:
                specs.append(('bl_col', b, d, c))
    for r in range(1, 4):
        for i in range(geom.C):
            specs.append(('naked', i, r))
            specs.append(('hidden', i, r))
            specs.append(('forced', i, r))
    ncell_start = len(specs)
    for i in range(geom.C):
        for d in range(1, N + 1):
            specs.append(('eq', i, d))
        specs.append(('le3', i))
    return specs, ncell_start


def atom_values(specs, sol, force, naked, hidden, geom):
    """Vectorised atoms. sol (B,C) digits, force (B,C), naked/hidden (B,3,C). Returns (B, A) uint8."""
    B = sol.shape[0]
    out = np.zeros((B, len(specs)), np.uint8)
    for a, s in enumerate(specs):
        k = s[0]
        if k == 'bl_row':
            _, b, d, r = s
            cells = [i for i in range(geom.C) if geom.box[i] == b and geom.row[i] == r]
            out[:, a] = (sol[:, cells] == d).any(1)
        elif k == 'bl_col':
            _, b, d, c = s
            cells = [i for i in range(geom.C) if geom.box[i] == b and geom.col[i] == c]
            out[:, a] = (sol[:, cells] == d).any(1)
        elif k == 'naked':
            out[:, a] = naked[:, s[2] - 1, s[1]]
        elif k == 'hidden':
            out[:, a] = hidden[:, s[2] - 1, s[1]]
        elif k == 'forced':
            f = force[:, s[1]]
            out[:, a] = (f >= 1) & (f <= s[2])
        elif k == 'eq':
            out[:, a] = sol[:, s[1]] == s[2]
        elif k == 'le3':
            out[:, a] = sol[:, s[1]] <= 3
    return out


# ---------------------------------------------------------------- Sudoku-Extreme streaming loader (S9)
def stream_sudoku_extreme(split, n_keep, seed=0, url_base="https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/"):
    """Stream the HF CSV and keep a stratified reservoir sample (strata = rating buckets).
    Never writes the CSV to disk. Returns (puzzles uint8 (n,81), solutions uint8 (n,81), rating int32)."""
    import requests
    rng = random.Random(seed)
    buckets = {}  # rating bucket -> reservoir list
    seen = {}
    per = None
    r = requests.get(url_base + f"{split}.csv", stream=True, timeout=60)
    r.raise_for_status()
    header = None
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if header is None:
            header = line.split(',')
            continue
        parts = line.split(',')
        rec = dict(zip(header, parts))
        q, a, rt = rec['question'], rec['answer'], int(rec['rating'])
        bk = 0 if rt == 0 else (1 if rt <= 2 else (2 if rt <= 10 else (3 if rt <= 50 else 4)))
        seen[bk] = seen.get(bk, 0) + 1
        res = buckets.setdefault(bk, [])
        cap = n_keep  # reservoir per bucket; trimmed at the end
        if len(res) < cap:
            res.append((q, a, rt))
        else:
            j = rng.randrange(seen[bk])
            if j < cap:
                res[j] = (q, a, rt)
    per = n_keep // len(buckets)
    keep = []
    for bk in sorted(buckets):
        keep += buckets[bk][:per]
    rng.shuffle(keep)
    P = np.array([[0 if ch == '.' else int(ch) for ch in q] for q, a, rt in keep], np.uint8)
    S = np.array([[int(ch) for ch in a] for q, a, rt in keep], np.uint8)
    R = np.array([rt for q, a, rt in keep], np.int32)
    return P, S, R, seen


# ---------------------------------------------------------------- build S6 (python grids.py s6 NPROC)
def build_s6(n_total=24000, nproc=3, out="/tmp/p10/s6.npz", seed0=10_000):
    import multiprocessing as mp
    import time
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        res = pool.map(gen6_one, range(seed0, seed0 + n_total), chunksize=200)
    puz = np.stack([r[0] for r in res]); sol = np.stack([r[1] for r in res])
    depth = np.array([r[2] for r in res], np.int16); solved = np.array([r[3] for r in res])
    force = np.stack([r[4] for r in res]); nk = np.stack([r[5] for r in res]); hd = np.stack([r[6] for r in res])
    P = sym_perms(G6)
    canon = [canonical(s, G6, P) for s in sol]
    uniq = {c: k for k, c in enumerate(sorted(set(canon)))}
    cid = np.array([uniq[c] for c in canon])
    # split by canonical solution class: ~10% val, ~10% test of classes (then trim to 20k/2k/2k)
    rng = np.random.default_rng(0)
    cls = rng.permutation(len(uniq))
    part = np.zeros(len(uniq), np.int8)
    part[cls[: len(uniq) // 12]] = 1
    part[cls[len(uniq) // 12: 2 * len(uniq) // 12]] = 2
    split = part[cid]
    np.savez_compressed(out, puz=puz, sol=sol, depth=depth, solved=solved, force=force, naked=nk, hidden=hd,
                        cid=cid, split=split)
    print("built", len(puz), "puzzles in", round(time.time() - t0), "s; classes", len(uniq),
          "split counts", np.bincount(split), "stall", round(1 - solved.mean(), 4))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "s6":
        build_s6(int(sys.argv[2]), int(sys.argv[3]))
