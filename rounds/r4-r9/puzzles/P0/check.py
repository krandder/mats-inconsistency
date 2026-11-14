"""Maze path checkers, vectorised (torch, any device).

Pixel layout (easy-to-hard and maze-dataset with extend_pixels=True, asserted in
build_frozen.py): an n x n lattice (n = maze size, walls included), each lattice
cell a 2x2 pixel block, 3 pixels of padding on every side, so H = W = 2n + 6.
Lattice codes: 0 wall, 1 open, 2 start (red), 3 goal (green).

check(): our lattice-level checker. A prediction passes when
  - every 2x2 block is fully on or fully off,
  - start and goal are on, every on cell is open,
  - start and goal have exactly 1 on 4-neighbour, every other on cell exactly 2,
  - the number of on cells equals the BFS shortest-path cell count.
With exactly two degree-1 cells and all others degree 2 the on set is one
start-goal path plus disjoint cycles; the count then forces the path to be a
shortest path and the cycles to be empty, so no connectivity pass is needed.

knutson(): Knutson et al. 2024 shortest_path_checker (pixel count equals the
solution's, prediction contains the red start and green goal pixels); a count
not divisible by 4 (their assert) is scored as a failure.
"""
import torch
import torch.nn.functional as F


def lattice_pred(pred_px, n):
    """pred_px (B, H, W) 0/1 -> (uniform (B,), on (B, n, n) bool)."""
    B = pred_px.shape[0]
    inner = pred_px[:, 3:3 + 2 * n, 3:3 + 2 * n].reshape(B, n, 2, n, 2)
    s = inner.sum((2, 4))
    outside = pred_px.sum((1, 2)) - inner.sum((1, 2, 3, 4))
    uniform = ((s == 0) | (s == 4)).all(2).all(1) & (outside == 0)
    return uniform, s == 4


def nbr_count(on):
    o = F.pad(on.float(), (1, 1, 1, 1))
    return (o[:, :-2, 1:-1] + o[:, 2:, 1:-1] + o[:, 1:-1, :-2] + o[:, 1:-1, 2:]).round().long()


def check(pred_px, code, sp_len):
    """pred_px (B,H,W) 0/1 masked prediction; code (B,n,n) uint8 lattice codes; sp_len (B,) cells on a shortest path."""
    n = code.shape[-1]
    uniform, on = lattice_pred(pred_px.long(), n)
    code = code.long()
    ends = (code == 2) | (code == 3)
    deg = nbr_count(on)
    want = torch.where(ends, torch.ones_like(deg), torch.full_like(deg, 2))
    deg_ok = (~on | (deg == want)).all(2).all(1)
    ends_on = (~ends | on).all(2).all(1)
    open_ok = (~on | (code > 0)).all(2).all(1)
    count_ok = on.sum((1, 2)) == sp_len.long()
    return uniform & deg_ok & ends_on & open_ok & count_ok


def knutson(inputs, pred_px, sol_px):
    """inputs (B,3,H,W) float; pred_px, sol_px (B,H,W) 0/1."""
    num = pred_px.long().sum((1, 2))
    num_sol = sol_px.long().sum((1, 2))
    red = (inputs[:, 0] == 1) & (inputs[:, 1] == 0) & (inputs[:, 2] == 0)
    green = (inputs[:, 0] == 0) & (inputs[:, 1] == 1) & (inputs[:, 2] == 0)
    on = pred_px > 0
    has_s = (on & red).flatten(1).any(1)
    has_g = (on & green).flatten(1).any(1)
    return (num % 4 == 0) & (num == num_sol) & has_s & has_g


# ------------------------------------------------------------------ lattice utilities (numpy, data build)
def expand(code, sol=None):
    """Lattice codes (N,n,n) uint8 -> pixel inputs (N,3,H,W) float32 and solutions (N,H,W) float32."""
    code = torch.as_tensor(code).long()
    N, n, _ = code.shape
    rgb = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
    px = rgb[code].permute(0, 3, 1, 2)                      # (N,3,n,n)
    px = px.repeat_interleave(2, 2).repeat_interleave(2, 3)
    px = F.pad(px, (3, 3, 3, 3))
    if sol is None:
        return px
    s = torch.as_tensor(sol).float().repeat_interleave(2, 1).repeat_interleave(2, 2)
    return px, F.pad(s, (3, 3, 3, 3))


def to_lattice(inputs, sols):
    """Pixel tensors -> lattice codes and solution bits; asserts the round trip is exact."""
    inputs = torch.as_tensor(inputs).float()
    sols = torch.as_tensor(sols).float()
    H = inputs.shape[-1]
    n = (H - 6) // 2
    assert H == 2 * n + 6, H
    c = inputs[:, :, 3:3 + 2 * n:2, 3:3 + 2 * n:2]            # (N,3,n,n)
    code = torch.zeros(c.shape[0], n, n, dtype=torch.uint8)
    code[(c[:, 0] == 1) & (c[:, 1] == 1) & (c[:, 2] == 1)] = 1
    code[(c[:, 0] == 1) & (c[:, 1] == 0) & (c[:, 2] == 0)] = 2
    code[(c[:, 0] == 0) & (c[:, 1] == 1) & (c[:, 2] == 0)] = 3
    sol = (sols[:, 3:3 + 2 * n:2, 3:3 + 2 * n:2] > 0.5).to(torch.uint8)
    px, spx = expand(code, sol)
    assert torch.equal(px, inputs), "maze pixel layout differs from the lattice expansion"
    assert torch.equal(spx, sols), "solution pixel layout differs from the lattice expansion"
    return code, sol


def bfs_len(code):
    """Cells on a shortest start-goal path (numpy BFS on open lattice cells), per maze."""
    import numpy as np
    from collections import deque
    out = []
    for c in np.asarray(code):
        n = c.shape[0]
        s = tuple(np.argwhere(c == 2)[0]); g = tuple(np.argwhere(c == 3)[0])
        dist = -np.ones((n, n), np.int64); dist[s] = 1
        q = deque([s])
        while q:
            i, j = q.popleft()
            if (i, j) == g:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n and c[a, b] > 0 and dist[a, b] < 0:
                    dist[a, b] = dist[i, j] + 1; q.append((a, b))
        out.append(int(dist[g]))
    return np.array(out, np.int32)


def deadend_rounds(code):
    """Parallel dead-end filling on the lattice: rounds until no open non-end cell has <= 1 open
    neighbour. Knutson et al. found DT-Recall matches dead-end filling; depth ~ 2*rounds/5 iterations."""
    c = torch.as_tensor(code).long()
    open_ = c > 0
    ends = (c == 2) | (c == 3)
    r = torch.zeros(c.shape[0], dtype=torch.long)
    while True:
        deg = nbr_count(open_)
        kill = open_ & ~ends & (deg <= 1)
        any_ = kill.flatten(1).any(1)
        if not any_.any():
            return r.numpy().astype("int32")
        r += any_.long()
        open_ = open_ & ~kill
