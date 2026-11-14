"""P11 vendored copies (P7 and P6 are being edited today, so they are copied, not imported).

Sources and their sha256 at vendoring time (2026-09-29 ~12:25Z). The spec pinned
maze.py 35a7c676... and p6_entry.py c6c72b7f...; both files had been edited again
by the time P11 vendored them, so the hashes below are the ones actually copied.
  puzzles/P7/maze.py:bfs                 28266830bc22c316e0554fb2686924b97c50968de932a332fd24471589dcc0dc
  puzzles/P7/common.py:ci (+T975)        53c728c3d9677a84164cb96288f9965083b522b424f563ba5368c1704ac03118
  puzzles/P1/make_mazes_upstream.py      d73f921673c74c146a212cd903558d002a008c214f4302aaae0daf4750062ea4
      (rendering block of gen_dataset below; the whole file is vendored verbatim as
       vendored_upstream.py for the S0 round-trip test)
  puzzles/P6/p6_entry.py:DTL             b755fd3f0665323221f4cc00255aa9971d8c27c670e0e16195d51a36d0e5c4c3
      (ported from Conv1d to Conv2d; nothing else changed)
"""
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_SHA256 = {
    "P7/maze.py": "28266830bc22c316e0554fb2686924b97c50968de932a332fd24471589dcc0dc",
    "P7/common.py": "53c728c3d9677a84164cb96288f9965083b522b424f563ba5368c1704ac03118",
    "P1/make_mazes_upstream.py": "d73f921673c74c146a212cd903558d002a008c214f4302aaae0daf4750062ea4",
    "P6/p6_entry.py": "b755fd3f0665323221f4cc00255aa9971d8c27c670e0e16195d51a36d0e5c4c3",
}
SPEC_PINNED_PREFIX = {"P7/maze.py": "35a7c676", "P7/common.py": "53c728c3",
                      "P1/make_mazes_upstream.py": "d73f9216", "P6/p6_entry.py": "c6c72b7f"}


# ------------------------------------------------------------ P7/maze.py:bfs (verbatim)
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


# ------------------------------------------------------------ P7/common.py:ci (verbatim)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return None
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "ci95": None, "n": 1}
    sd = float(np.std(xs, ddof=1))
    return {"mean": m, "ci95": T975.get(n - 1, 1.96) * sd / math.sqrt(n), "n": n}


# ------------------------------------------------------------ P1/make_mazes_upstream.py rendering block
def upstream_render(data_array, solution_array, maze_size):
    """The upscaling block of upstream gen_dataset, verbatim apart from the function wrapper.
    data_array (N, 2m+1, 2m+1, 3), solution_array (N, 2m+1, 2m+1, 1)."""
    num_images = data_array.shape[0]
    img_size = 2 * (2 * (maze_size) + 1) + 2
    border = (img_size - 4 * maze_size) // 2

    big_data_array = np.zeros((num_images, img_size, img_size, 3))
    big_data_array[:, border-1:-border:2, border-1:-border:2, :] = data_array
    big_data_array[:, border:-border+1:2, border:-border+1:2, :] = data_array
    big_data_array[:, border:-border+1:2, border-1:-border:2, :] = data_array
    big_data_array[:, border-1:-border:2, border:-border+1:2, :] = data_array

    big_solution_array = np.zeros((num_images, img_size, img_size, 3))
    big_solution_array[:, border-1:-border:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border:-border+1:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border-1:-border:2, border:-border+1:2, :] = solution_array
    return big_data_array, big_solution_array


# ------------------------------------------------------------ P6/p6_entry.py:DTL, Conv1d -> Conv2d
class DTL(nn.Module):
    """DT-L (Bear, Prugel-Bennett & Hare 2024): recall = SN conv on h + plain biased conv on x,
    ELU, two learned per-channel interpolated skips (1-g) id + g f, spectral norm on every
    recurrent conv acting on h (so each step is <= 1-Lipschitz in h)."""
    def __init__(self, w=32):
        super().__init__()
        SN = torch.nn.utils.parametrizations.spectral_norm
        self.proj = nn.Conv2d(3, w, 3, padding=1)
        self.rh = SN(nn.Conv2d(w, w, 3, padding=1, bias=False))
        self.rx = nn.Conv2d(3, w, 3, padding=1)
        self.c1 = SN(nn.Conv2d(w, w, 3, padding=1))
        self.c2 = SN(nn.Conv2d(w, w, 3, padding=1))
        self.g1 = nn.Parameter(torch.zeros(w))
        self.g2 = nn.Parameter(torch.zeros(w))
        self.head = nn.Sequential(nn.Conv2d(w, w, 3, padding=1), nn.ELU(),
                                  nn.Conv2d(w, w // 2, 3, padding=1), nn.ELU(),
                                  nn.Conv2d(w // 2, 1, 3, padding=1))

    def init_state(self, xoh):
        return F.elu(self.proj(xoh))

    def step(self, h, xoh):
        a1 = torch.sigmoid(self.g1).view(1, -1, 1, 1)
        a2 = torch.sigmoid(self.g2).view(1, -1, 1, 1)
        h1 = (1 - a1) * h + a1 * F.elu(self.rh(h) + self.rx(xoh))
        return (1 - a2) * h1 + a2 * F.elu(self.c2(F.elu(self.c1(h1))))

    def out(self, h):
        return self.head(h)[:, 0]
