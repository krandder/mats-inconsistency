"""Vendored verbatim from puzzles/P7/maze.py (sha256 28266830bc22c316e0554fb2686924b97c50968de932a332fd24471589dcc0dc):
functions bfs and gen_mazes, unchanged. Do not edit; P8 builds its own channels on top."""
from collections import deque

import numpy as np


def bfs(walls, s):
    n = walls.shape[0]
    d = np.full((n, n), 10 ** 6, dtype=np.int64)
    d[s] = 0
    q = deque([s])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n and not walls[a, b] and d[a, b] > d[i, j] + 1:
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    return d


def gen_mazes(num, n, density, rng):
    walls = np.zeros((num, n, n), np.float32)
    start = np.zeros((num, n, n), np.float32)
    dist = np.zeros((num, n, n), np.int64)
    onpath = np.zeros((num, n, n), np.int64)
    for t in range(num):
        w = (rng.random((n, n)) < density)
        s = (int(rng.integers(n)), int(rng.integers(n)))
        w[s] = False
        d = bfs(w, s)
        reach = np.argwhere(d < 10 ** 6)
        g = tuple(reach[rng.integers(len(reach))])
        dg = bfs(w, g)
        walls[t] = w
        start[t][s] = 1
        dist[t] = d
        onpath[t] = ((d + dg) == d[g]) & (d < 10 ** 6)
    return walls, start, dist, onpath
