import sys, time, json, itertools, numpy as np
from pilotnp import gen_maze, edge_grids
BIG = 1e4
def cfgs(net): return np.array([v for v in itertools.product([0,1], repeat=8) if sum(v[:4]) - sum(v[4:]) == net], np.float64)
V0, VS, VG = cfgs(0), cfgs(1), cfgs(-1)
def shift(m):
    out = np.zeros_like(m)
    out[:, :, :-1, 0] = m[:, :, 1:, 2]; out[:, :-1, :, 1] = m[:, 1:, :, 3]
    out[:, :, 1:, 2] = m[:, :, :-1, 0]; out[:, 1:, :, 3] = m[:, :-1, :, 1]
    return out
def fac(c, V):
    E = c @ V.T; out = np.empty_like(c)
    for k in range(8):
        Ek = E - V[:, k] * c[:, k:k+1]
        out[:, k] = np.where(V[:, k] > 0, Ek, 3*BIG).min(-1) - np.where(V[:, k] == 0, Ek, 3*BIG).min(-1)
    return out
def run(n, p, num, seed, T, checks, cost=1.0, damp=0.5):
    rng = np.random.default_rng(seed); mz = [gen_maze(n, p, rng) for _ in range(num)]
    op = np.stack([edge_grids(z[0], z[1]) for z in mz]); tgt = np.stack([edge_grids(z[4], z[5]) for z in mz])
    S = np.zeros(op.shape[:3], bool); G = np.zeros_like(S)
    for k, z in enumerate(mz): S[k][z[2]] = True; G[k][z[3]] = True
    un = np.where(op, cost, BIG); mo = np.zeros(op.shape); mi = np.zeros(op.shape); res = {}
    for t in range(1, T+1):
        cin = np.clip(np.concatenate([un + shift(mi), un + shift(mo)], -1), -BIG, BIG); new = np.zeros_like(cin)
        for mask, V in ((~S & ~G, V0), (S, VS), (G, VG)): new[mask] = fac(cin[mask], V)
        new = np.clip(new, -BIG, BIG)
        mo = (1-damp)*new[..., :4] + damp*mo; mi = (1-damp)*new[..., 4:] + damp*mi
        if t in checks:
            belief = un + mo + shift(mi); arc = (belief < 0) & op
            edge = (arc | shift(arc.astype(float)).astype(bool)) & op
            res[t] = float((edge == tgt).all(axis=(1,2,3)).mean())
    return res
