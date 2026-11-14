"""P9b data: on-shortest-path mask on P7 mazes (density 0.35), G = unique onpath argmax of dist, d(S,G) >= 2.

Inputs (4 channels): wall, start, goal, inside (constant 1 on the grid, 0 in the conv zero padding, so the
net can tell the border from an open cell; P7/Bansal mazes have explicit border walls instead).
Seed ranges: train pool 10_000 + seed; dev 25x25 = 50_025; tests 60_000 + n; importance/IIA sets 70_000 + n.
"""
import numpy as np
import torch

from p9util import gen_mazes

DENSITY = 0.35


def make_onpath(num, n, seed, min_d=2):
    rng = np.random.default_rng(seed)
    W, S, G, Y, D = [], [], [], [], []
    have = 0
    while have < num:
        k = max(64, int((num - have) * 1.4))
        w, s, d, op = gen_mazes(k, n, DENSITY, rng)
        for i in range(k):
            dd = np.where(op[i] > 0, d[i], -1)
            gi = np.unravel_index(np.argmax(dd), dd.shape)
            dsg = int(d[i][gi])
            if dsg < min_d:
                continue
            assert (dd == dsg).sum() == 1  # unique onpath argmax of dist
            g = np.zeros((n, n), np.float32)
            g[gi] = 1
            W.append(w[i]); S.append(s[i]); G.append(g); Y.append(op[i].astype(np.float32)); D.append(dsg)
            have += 1
            if have == num:
                break
    W, S, G, Y = (np.stack(a) for a in (W, S, G, Y))
    x = np.stack([W, S, G, np.ones_like(W)], 1)
    return {"x": torch.tensor(x), "y": torch.tensor(Y), "dsg": np.array(D), "n": n}


def subset(ds, idx):
    idx = np.asarray(idx)
    return {"x": ds["x"][idx], "y": ds["y"][idx], "dsg": ds["dsg"][idx], "n": ds["n"]}
