"""P12 pilot: (1) DFS / percolated maze generator on a cell grid, unique-shortest-path labels;
(2) flow-conservation arbitrageur = sum-product BP on degree-constraint factors;
(3) timing of a width-32 DT-recall step on CPU."""
import sys, time, math, json
from collections import deque
import numpy as np

DIRS = [(0,1),(1,0),(0,-1),(-1,0)]  # E S W N

def gen_maze(n, p, rng, deadend_start=False):
    """Recursive-backtracker spanning tree (maze-dataset gen_dfs equivalent), then open each
    remaining wall with prob p (percolation). Returns openE (n,n-1), openS (n-1,n), S, G, path edges.
    Rejects mazes whose S-G shortest path is not unique."""
    while True:
        oE = np.zeros((n, n-1), bool); oS = np.zeros((n-1, n), bool)
        seen = np.zeros((n, n), bool); st = [(int(rng.integers(n)), int(rng.integers(n)))]; seen[st[0]] = True
        while st:
            i, j = st[-1]; nb = [(i+a, j+b) for a, b in DIRS if 0 <= i+a < n and 0 <= j+b < n and not seen[i+a, j+b]]
            if not nb: st.pop(); continue
            a, b = nb[rng.integers(len(nb))]; seen[a, b] = True; st.append((a, b))
            if a == i: oE[i, min(j, b)] = True
            else: oS[min(i, a), j] = True
        if p > 0:
            oE |= rng.random(oE.shape) < p; oS |= rng.random(oS.shape) < p
        cells = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(2)]
        S, G = cells
        if S == G: continue
        dS, cS = bfs(oE, oS, S); dG, _ = bfs(oE, oS, G)
        L = dS[G]
        if p > 0 and cS[G] != 1: continue  # unique shortest path only
        onp = (dS + dG == L)
        pE = onp[:, :-1] & onp[:, 1:] & oE & (np.abs(dS[:, :-1] - dS[:, 1:]) == 1)
        pS = onp[:-1, :] & onp[1:, :] & oS & (np.abs(dS[:-1, :] - dS[1:, :]) == 1)
        return oE, oS, S, G, pE, pS, L

def nbrs(oE, oS, i, j):
    n = oE.shape[0]
    if j < n-1 and oE[i, j]: yield i, j+1
    if j > 0 and oE[i, j-1]: yield i, j-1
    if i < n-1 and oS[i, j]: yield i+1, j
    if i > 0 and oS[i-1, j]: yield i-1, j

def bfs(oE, oS, s):
    n = oE.shape[0]; d = np.full((n, n), 10**6); c = np.zeros((n, n), np.int64)
    d[s] = 0; c[s] = 1; q = deque([s])
    while q:
        u = q.popleft()
        for v in nbrs(oE, oS, *u):
            if d[v] > d[u] + 1: d[v] = d[u] + 1; c[v] = c[u]; q.append(v)
            elif d[v] == d[u] + 1: c[v] = min(c[v] + c[u], 2)
    return d, c

# ---------------- arbitrageur: sum-product BP, variables = edges, factors = cells
def edge_grids(oE, oS):
    """open mask per (cell, dir) as (n,n,4) bool; plus endpoint mask."""
    n = oE.shape[0]; o = np.zeros((n, n, 4), bool)
    o[:, :-1, 0] = oE; o[:-1, :, 1] = oS; o[:, 1:, 2] = oE; o[1:, :, 3] = oS
    return o

