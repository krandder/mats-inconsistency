"""Verbatim snapshot of bfs() and gen_mazes() from P7/maze.py (read-only source).

Why a snapshot: the P9 spec asks to import P7/maze.py and assert sha256 prefix 35a7c67608176edc.
At 2026-09-29 12:20Z the file had been edited again (mtime 12:14:37Z) and hashes 28266830bc22c316,
so the assert would abort every run, and P7 is still being edited by another agent. These two
functions were copied verbatim (via importlib as 'p7maze_ro', sys.dont_write_bytecode=True) so
P9's data cannot drift; p9util.py asserts FUNC_SHA256 over the source text below.
P7/maze.py sha256 at copy time: 28266830bc22c316e0554fb2686924b97c50968de932a332fd24471589dcc0dc
"""
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


FUNC_SHA256 = "29480392f569e82fe7a3223904c58a60111623b687f5e362d81a53052a34441d"
P7_FILE_SHA256_AT_COPY = "28266830bc22c316e0554fb2686924b97c50968de932a332fd24471589dcc0dc"
