"""Weighted check: directed-flow min-sum BP with TRUE continuous cell costs vs Dijkstra labels.
Maze = DFS tree + percolation p; S,G at BFS distance >= n; edge cost = mean of endpoint cell costs."""
import sys, time, json, heapq, numpy as np
sys.path.insert(0, '/tmp/p12rev3')
from pilotnp import gen_maze, edge_grids, bfs
from flownp3 import fac, shift, V0, VS, VG, BIG
def make(n, p, rng):
    while True:
        oE, oS, *_ = gen_maze(n, 0.0, rng)
        oE = oE | (rng.random(oE.shape) < p); oS = oS | (rng.random(oS.shape) < p)
        S = (int(rng.integers(n)), int(rng.integers(n))); dS, _ = bfs(oE, oS, S)
        far = np.argwhere(dS >= n)
        if len(far): G = tuple(far[rng.integers(len(far))]); break
    c = 1 + 8 / (1 + np.exp(-rng.normal(size=(n, n)) * 2))
    op = edge_grids(oE, oS); ec = np.full((n, n, 4), BIG)
    D = [(0,1),(1,0),(0,-1),(-1,0)]
    for i in range(n):
        for j in range(n):
            for k,(a,b) in enumerate(D):
                if op[i,j,k]: ec[i,j,k] = (c[i,j] + c[i+a,j+b]) / 2
    dist = np.full((n,n), np.inf); prev = {}; dist[S] = 0; h = [(0.0, S)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]: continue
        for k,(a,b) in enumerate(D):
            if op[u][k]:
                v = (u[0]+a, u[1]+b); nd = d + ec[u][k]
                if nd < dist[v]: dist[v] = nd; prev[v] = (u, k); heapq.heappush(h, (nd, v))
    tgt = np.zeros((n,n,4), bool); v = G; L = 0
    while v != S:
        u, k = prev[v]; tgt[u][k] = True; tgt[v][(k+2)%4] = True; v = u; L += 1
    return op, ec, S, G, tgt, L
def run(n, p, num, seed, T, checks, damp=0.5):
    rng = np.random.default_rng(seed); mz = [make(n, p, rng) for _ in range(num)]
    op = np.stack([m[0] for m in mz]); un = np.stack([m[1] for m in mz]); tgt = np.stack([m[4] for m in mz])
    S = np.zeros(op.shape[:3], bool); G = np.zeros_like(S)
    for k, m in enumerate(mz): S[k][m[2]] = True; G[k][m[3]] = True
    mo = np.zeros(op.shape); mi = np.zeros(op.shape); res = {}
    for t in range(1, T+1):
        cin = np.clip(np.concatenate([un + shift(mi), un + shift(mo)], -1), -BIG, BIG); new = np.zeros_like(cin)
        for mask, V in ((~S & ~G, V0), (S, VS), (G, VG)): new[mask] = fac(cin[mask], V)
        new = np.clip(new, -BIG, BIG)
        mo = (1-damp)*new[..., :4] + damp*mo; mi = (1-damp)*new[..., 4:] + damp*mi
        if t in checks:
            belief = un + mo + shift(mi); arc = (belief < 0) & op
            edge = (arc | shift(arc.astype(float)).astype(bool)) & op
            res[t] = float((edge == tgt).all(axis=(1,2,3)).mean())
    return res, float(np.median([m[5] for m in mz]))
for n, num, T in [(9, 20, 300)]:
    t0 = time.time(); r, L = run(n, 0.3, num, 1, T, (30, 100, 300))
    print(json.dumps({"n": n, "p": 0.3, "num": num, "median_L": L, "weighted_flow_minsum_exact_by_T": r, "sec": round(time.time()-t0, 1)}), flush=True)
