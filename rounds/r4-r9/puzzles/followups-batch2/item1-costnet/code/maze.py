"""P12 maze families on a cell grid (generator adapted from pilot/pilot.py and pilot/wflow.py).

A maze of width n is an n x n grid of cells; every cell has 4 edge slots E,S,W,N (d = 0..3).
Arrays per maze:
  open  (n,n,4) bool   edge slot is open (symmetric: open[c,d] == open[nb(c,d), opp(d)])
  code  (n,n)   uint16 9-bit cell code (W only; 0 for U1/U2)
  cost  (n,n)   float32 cell cost c = 1 + 8 sigma(g(code)) (W) or 1 (U)
  S, G  (2,)    start and goal cells
  path  (n,n,4) bool   label: edge slots on the min-cost S-G path
Families:
  W   DFS spanning tree (maze-dataset gen_dfs process) + each remaining wall opened with p;
      edge cost = mean of its two cell costs; label = Dijkstra min-cost path; exact ties rejected.
  U1  p = 0 (a tree), U2 p = 0.1; unit cost; label = Dijkstra with a seeded 1e-3 edge-cost jitter.
S uniform; G uniform among cells at BFS distance >= n from S (no uniqueness filter).
"""
import heapq
from collections import deque
import numpy as np

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E S W N
OPP = [2, 3, 0, 1]
NBITS = 9


# ---------------------------------------------------------------- hidden cost function
def _g_table():
    """g: fixed random 9 -> 16 -> 1 tanh MLP (seed 0) on the +-1 code bits, standardised over
    the 512 codes to mean 0, sd 1.5. Returns cost per code, c = 1 + 8 sigma(g)."""
    r = np.random.default_rng(0)
    W1 = r.normal(size=(16, NBITS)); b1 = r.normal(size=16) * 0.5
    W2 = r.normal(size=16); b2 = 0.0
    bits = ((np.arange(512)[:, None] >> np.arange(NBITS)[None, :]) & 1).astype(np.float64)
    g = np.tanh((2 * bits - 1) @ W1.T + b1) @ W2 + b2
    g = (g - g.mean()) / g.std() * 1.5
    return 1.0 + 8.0 / (1.0 + np.exp(-g))


COST_OF_CODE = _g_table()  # (512,) float64


def code_bits(code):
    """(..., ) uint16 -> (..., 9) float32 bits."""
    return ((code[..., None].astype(np.int64) >> np.arange(NBITS)) & 1).astype(np.float32)


# ---------------------------------------------------------------- generator
def dfs_tree(n, rng):
    oE = np.zeros((n, n - 1), bool); oS = np.zeros((n - 1, n), bool)
    seen = np.zeros((n, n), bool)
    st = [(int(rng.integers(n)), int(rng.integers(n)))]; seen[st[0]] = True
    while st:
        i, j = st[-1]
        nb = [(i + a, j + b) for a, b in DIRS if 0 <= i + a < n and 0 <= j + b < n and not seen[i + a, j + b]]
        if not nb:
            st.pop(); continue
        a, b = nb[rng.integers(len(nb))]; seen[a, b] = True; st.append((a, b))
        if a == i: oE[i, min(j, b)] = True
        else: oS[min(i, a), j] = True
    return oE, oS


def slots(oE, oS):
    n = oE.shape[0]; o = np.zeros((n, n, 4), bool)
    o[:, :-1, 0] = oE; o[:-1, :, 1] = oS; o[:, 1:, 2] = oE; o[1:, :, 3] = oS
    return o


def bfs(op, s):
    n = op.shape[0]; d = np.full((n, n), 10 ** 6, np.int64); d[s] = 0; q = deque([s])
    while q:
        u = q.popleft()
        for k, (a, b) in enumerate(DIRS):
            if op[u[0], u[1], k]:
                v = (u[0] + a, u[1] + b)
                if d[v] > d[u] + 1: d[v] = d[u] + 1; q.append(v)
    return d


def edge_costs(op, cost):
    """(n,n,4) edge cost per slot = mean of the two cell costs; inf where closed."""
    n = op.shape[0]; ec = np.full((n, n, 4), np.inf)
    ec[:, :-1, 0] = (cost[:, :-1] + cost[:, 1:]) / 2; ec[:-1, :, 1] = (cost[:-1, :] + cost[1:, :]) / 2
    ec[:, 1:, 2] = ec[:, :-1, 0]; ec[1:, :, 3] = ec[:-1, :, 1]
    return np.where(op, ec, np.inf)


def dijkstra(op, ec, s):
    n = op.shape[0]; dist = np.full((n, n), np.inf); dist[s] = 0.0; prev = {}; h = [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]: continue
        for k, (a, b) in enumerate(DIRS):
            if op[u[0], u[1], k]:
                v = (u[0] + a, u[1] + b); nd = d + ec[u[0], u[1], k]
                if nd < dist[v]: dist[v] = nd; prev[v] = (u, k); heapq.heappush(h, (nd, v))
    return dist, prev


def label_path(op, ec, S, G):
    """Min-cost path slots, its length L (edges), optimal cost, and whether the optimum is tied."""
    dS, prev = dijkstra(op, ec, S); dG, _ = dijkstra(op, ec, G); D = dS[G]
    path = np.zeros(op.shape, bool); v = G; L = 0
    while v != S:
        u, k = prev[v]; path[u[0], u[1], k] = True; path[v[0], v[1], OPP[k]] = True; v = u; L += 1
    # tight edges: dS[u] + w + dG[v] == D (tolerance), counted once per edge
    n = op.shape[0]; tight = 0; tol = 1e-9 * max(D, 1.0)
    for k in (0, 1):
        a, b = DIRS[k]
        du = dS[: n - a, : n - b]; dv = dS[a:, b:]; gu = dG[: n - a, : n - b]; gv = dG[a:, b:]
        w = ec[: n - a, : n - b, k]
        t1 = np.abs(du + w + gv - D) <= tol; t2 = np.abs(dv + w + gu - D) <= tol
        tight += int(((t1 | t2) & np.isfinite(w)).sum())
    return path, L, float(D), tight != L


def make_maze(n, p, rng, weighted=True, jitter_rng=None):
    """One accepted maze. Returns dict and counters (tries for S/G placement, ties rejected)."""
    tries = 0; ties = 0
    while True:
        oE, oS = dfs_tree(n, rng)
        if p > 0:
            oE = oE | (rng.random(oE.shape) < p); oS = oS | (rng.random(oS.shape) < p)
        op = slots(oE, oS)
        for _ in range(4):
            tries += 1
            S = (int(rng.integers(n)), int(rng.integers(n))); dS = bfs(op, S)
            far = np.argwhere(dS >= n)
            if len(far): break
        else:
            continue
        G = tuple(int(x) for x in far[rng.integers(len(far))])
        if weighted:
            code = rng.integers(0, 512, size=(n, n)).astype(np.uint16)
            cost = COST_OF_CODE[code].astype(np.float64)
            ec = edge_costs(op, cost)
        else:
            code = np.zeros((n, n), np.uint16); cost = np.ones((n, n))
            jr = jitter_rng if jitter_rng is not None else rng
            ec = edge_costs(op, cost)
            jit = np.zeros((n, n, 4)); jE = jr.random((n, n - 1)) * 1e-3; jS = jr.random((n - 1, n)) * 1e-3
            jit[:, :-1, 0] = jE; jit[:, 1:, 2] = jE; jit[:-1, :, 1] = jS; jit[1:, :, 3] = jS
            ec = ec + jit
        path, L, D, tied = label_path(op, ec, S, G)
        if tied:
            ties += 1; continue
        return dict(open=op, code=code, cost=cost.astype(np.float32), S=np.array(S, np.int16),
                    G=np.array(G, np.int16), path=path, L=L, opt=D), tries, ties


def make_set(n, p, num, seed, weighted=True):
    rng = np.random.default_rng(seed); jr = np.random.default_rng(seed + 10 ** 6)
    ms = []; tries = 0; ties = 0
    for _ in range(num):
        m, t, z = make_maze(n, p, rng, weighted, jr); ms.append(m); tries += t; ties += z
    return pack(ms), {"n": n, "p": p, "num": num, "seed": seed, "weighted": weighted,
                      "SG_placement_tries_per_accept": tries / num, "tie_rejects": ties,
                      "tie_rate": ties / (num + ties)}


def pack(ms):
    return dict(open=np.stack([m["open"] for m in ms]), code=np.stack([m["code"] for m in ms]),
                cost=np.stack([m["cost"] for m in ms]), S=np.stack([m["S"] for m in ms]),
                G=np.stack([m["G"] for m in ms]), path=np.stack([m["path"] for m in ms]),
                L=np.array([m["L"] for m in ms], np.int32), opt=np.array([m["opt"] for m in ms], np.float64))


def save(path, d, meta=None):
    np.savez_compressed(path, **d, **({"meta": np.array(repr(meta))} if meta else {}))


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}


# ---------------------------------------------------------------- dihedral maps (numpy, per maze)
def dihedral(d, k):
    """Apply dihedral map k (0..7) = (transpose if k&4) then (flip rows if k&2) then (flip cols if k&1)
    to a packed dict of mazes (batched arrays with leading maze axis). Codes move with their cells."""
    n = d["open"].shape[1]
    op, pa, code, cost = d["open"], d["path"], d["code"], d["cost"]
    S, G = d["S"].astype(np.int64), d["G"].astype(np.int64)
    if k & 4:
        op = op.transpose(0, 2, 1, 3)[..., [1, 0, 3, 2]]; pa = pa.transpose(0, 2, 1, 3)[..., [1, 0, 3, 2]]
        code = code.transpose(0, 2, 1); cost = cost.transpose(0, 2, 1); S = S[:, ::-1]; G = G[:, ::-1]
    if k & 2:
        op = op[:, ::-1][..., [0, 3, 2, 1]]; pa = pa[:, ::-1][..., [0, 3, 2, 1]]
        code = code[:, ::-1]; cost = cost[:, ::-1]; S = np.stack([n - 1 - S[:, 0], S[:, 1]], 1); G = np.stack([n - 1 - G[:, 0], G[:, 1]], 1)
    if k & 1:
        op = op[:, :, ::-1][..., [2, 1, 0, 3]]; pa = pa[:, :, ::-1][..., [2, 1, 0, 3]]
        code = code[:, :, ::-1]; cost = cost[:, :, ::-1]; S = np.stack([S[:, 0], n - 1 - S[:, 1]], 1); G = np.stack([G[:, 0], n - 1 - G[:, 1]], 1)
    out = dict(d); out.update(open=np.ascontiguousarray(op), path=np.ascontiguousarray(pa), code=np.ascontiguousarray(code),
                              cost=np.ascontiguousarray(cost), S=S.astype(np.int16), G=G.astype(np.int16))
    return out
