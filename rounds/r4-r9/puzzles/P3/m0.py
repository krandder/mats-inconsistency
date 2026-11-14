"""P3 / Stage 3, maze oracle M0 (CPU, no training of a code net).

Mazes from maze-dataset 1.4.2 (gen_dfs = perfect, gen_dfs_percolation = percolated), rasterised to a
(2n+1)^2 pixel grid (cell (r,c) -> pixel (2r+1, 2c+1)).  Three local cellular-automaton rules, each defined
on ARBITRARY states and updated synchronously with a 4-neighbour stencil:
  DEF   dead-end filling, protected endpoints.  State: alive / filled (K=2).  An alive, open, non-endpoint
        pixel with <= 1 alive 4-neighbour is filled.  Output: the alive set.
  BFS_S flood from the start + backtrack from the goal; BFS_G flood from the goal + backtrack from the start.
        State code (K=8): 0 unreached, 1+d reached with d = distance mod 3, 4+d marked (on the path), 7 unused.
        Unreached open pixel: the source becomes reached with d=0; otherwise, if some 4-neighbour is reached
        or marked, it becomes reached with d = (min over those neighbours of d_nb + 1) mod 3.
        Backtrack: the destination, once reached, is marked; a reached pixel with a marked neighbour whose
        d = (own d + 1) mod 3 is marked.  Output: the marked set.
Build gate: at the path-final round the CA equals the global simulators (P7 bfs distances mod 3 and reach;
a global queue-based dead-end pruning) and maze-dataset's solution, on 1000 mazes per family, 100%.
Reports (i) rounds p50/p99 and T, (ii) the DEF failure share vs p on 9x9 non-dead-end test mazes,
(iii) a bounded 3x3 reader R fitted on the exact oracle codes, S_R per round and chunk gains for c in {1,6,T}.
"""
import argparse
import json
import math
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P7")  # read-only import
from maze import bfs  # noqa: E402  (P7's global BFS on a pixel grid)
from maze_dataset import MazeDataset, MazeDatasetConfig  # noqa: E402
from maze_dataset.generation import LatticeMazeGenerators  # noqa: E402

BIG = 10 ** 6


# ---------------------------------------------------------------- mazes
def gen(n, family, p, num, seed, deadend):
    """Returns dict of arrays: walls [N,P,P] bool, s, g [N,2] pixel coords, sol [N,P,P] bool, start_deg [N]."""
    ctor = LatticeMazeGenerators.gen_dfs if family == "perfect" else LatticeMazeGenerators.gen_dfs_percolation
    kw = {} if family == "perfect" else dict(p=p)
    ek = dict(deadend_start=deadend, deadend_end=deadend, endpoints_not_equal=True, except_on_no_valid_endpoint=False)
    cfg = MazeDatasetConfig(name=f"p3m0-{family}-{n}-{p}-{seed}-{deadend}", grid_n=n, n_mazes=num, seed=seed,
                            maze_ctor=ctor, maze_ctor_kwargs=kw, endpoint_kwargs=ek)
    ds = MazeDataset.from_config(cfg, do_download=False, load_local=False, do_generate=True, save_local=False,
                                 verbose=False, gen_parallel=False)
    P = 2 * n + 1
    W, S, G, SOL, DEG = [], [], [], [], []
    for m in ds:
        if m is None or m.solution is None or len(m.solution) < 2:
            continue
        px = m.as_pixels(show_endpoints=False, show_solution=False)
        walls = px.sum(-1) == 0
        sol = np.zeros((P, P), bool)
        cs = np.asarray(m.solution)
        for a, b in zip(cs[:-1], cs[1:]):
            sol[2 * a[0] + 1, 2 * a[1] + 1] = sol[2 * b[0] + 1, 2 * b[1] + 1] = True
            sol[a[0] + b[0] + 1, a[1] + b[1] + 1] = True
        W.append(walls); SOL.append(sol)
        S.append(2 * np.asarray(m.start_pos) + 1); G.append(2 * np.asarray(m.end_pos) + 1)
        DEG.append(int(m.coord_degrees()[tuple(m.start_pos)]))
    return dict(walls=np.array(W), s=np.array(S), g=np.array(G), sol=np.array(SOL), start_deg=np.array(DEG))


def n_shortest(walls, s, g):
    """Number of shortest s-g paths (capped), from BFS layers."""
    d = bfs(walls, tuple(s))
    P = walls.shape[0]
    cnt = np.zeros((P, P), np.int64)
    cnt[tuple(s)] = 1
    order = np.argsort(d, axis=None)
    for idx in order:
        i, j = divmod(int(idx), P)
        if d[i, j] >= BIG or (i, j) == tuple(s):
            continue
        tot = 0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < P and 0 <= b < P and d[a, b] == d[i, j] - 1:
                tot += cnt[a, b]
        cnt[i, j] = min(tot, 1000)
    return int(cnt[tuple(g)]), d


def drop_ties(M):
    keep = []
    for k in range(len(M["walls"])):
        c, _ = n_shortest(M["walls"][k], M["s"][k], M["g"][k])
        keep.append(c == 1)
    keep = np.array(keep)
    return {k: v[keep] for k, v in M.items()}, int((~keep).sum())


# ---------------------------------------------------------------- CA rules (vectorised over mazes)
def shifts(a, fill):
    """4-neighbour views of a [N,P,P] array: up, down, left, right (value of the neighbour at each pixel)."""
    out = []
    for ax, sh in ((1, 1), (1, -1), (2, 1), (2, -1)):
        b = np.roll(a, sh, axis=ax)
        if ax == 1:
            (b[:, 0] if sh == 1 else b[:, -1])[...] = fill
        else:
            (b[:, :, 0] if sh == 1 else b[:, :, -1])[...] = fill
        out.append(b)
    return out


def endpoint_mask(M):
    N, P = M["walls"].shape[:2]
    e = np.zeros((N, P, P), bool)
    e[np.arange(N), M["s"][:, 0], M["s"][:, 1]] = True
    e[np.arange(N), M["g"][:, 0], M["g"][:, 1]] = True
    return e


def def_step(alive, walls, ends):
    alive = alive & ~walls
    nb = sum(v.astype(np.int64) for v in shifts(alive, False))
    return alive & ~((nb <= 1) & ~ends)


def bfs_step(code, walls, src, dst):
    """code [N,P,P] in 0..7 (see module doc); src/dst [N,P,P] bool masks."""
    code = np.where(walls, 0, code)
    reached = (code >= 1) & (code <= 6)
    d = np.where(code >= 4, code - 4, code - 1)  # valid where reached
    marked = code >= 4
    nd = [np.where((v >= 1) & (v <= 6), (np.where(v >= 4, v - 4, v - 1) + 1) % 3, 99) for v in shifts(code, 0)]
    cand = np.minimum.reduce(nd)  # (d_nb + 1) mod 3 of reached neighbours, 99 if none
    new = code.copy()
    unr = (~reached) & (~walls)
    new = np.where(unr & src, 1, new)
    new = np.where(unr & ~src & (cand < 99), 1 + np.where(cand < 99, cand, 0), new)
    # backtrack
    mnb = [(v >= 4) & ((v - 4) == (d + 1) % 3) for v in shifts(code, 0)]
    mark = reached & ~marked & (dst | np.logical_or.reduce(mnb))
    new = np.where(mark, 4 + d, new)
    return new


def rollout(M, algo, T, keep_states=False):
    """Returns per-round outputs: path-final round, fixed-point round, final output mask, (states [T+1,N,P,P])."""
    walls = M["walls"]
    N, P = walls.shape[:2]
    ends = endpoint_mask(M)
    if algo == "DEF":
        st = ~walls
    else:
        st = np.zeros((N, P, P), np.int64)
        s_m = np.zeros((N, P, P), bool); s_m[np.arange(N), M["s"][:, 0], M["s"][:, 1]] = True
        g_m = np.zeros((N, P, P), bool); g_m[np.arange(N), M["g"][:, 0], M["g"][:, 1]] = True
        src, dst = (s_m, g_m) if algo == "BFS_S" else (g_m, s_m)
    out = lambda z: z if algo == "DEF" else (z >= 4)
    states = [st.astype(np.int8)] if keep_states else None
    outs = [out(st)]
    fixed = np.full(N, -1)
    t = 0
    while t < T:
        nst = def_step(st, walls, ends) if algo == "DEF" else bfs_step(st, walls, src, dst)
        t += 1
        same = (nst == st).reshape(N, -1).all(1)
        fixed = np.where((fixed < 0) & same, t - 1, fixed)
        st = nst
        outs.append(out(st))
        if keep_states:
            states.append(st.astype(np.int8))
        if (fixed >= 0).all() and not keep_states:
            break
    outs = np.array(outs)  # R,N,P,P
    final = outs[-1]
    # path-final round: first round after which the output never changes again
    chg = (outs[1:] != outs[:-1]).reshape(len(outs) - 1, N, -1).any(-1)  # R-1,N
    last = np.where(chg.any(0), (len(chg) - 1) - np.argmax(chg[::-1], axis=0) + 1, 0)
    return dict(path_round=last, fixed_round=fixed, final=final, states=np.array(states) if keep_states else None, last_state=st)


def global_def(walls, s, g):
    """Queue-based dead-end pruning (global simulator for DEF)."""
    P = walls.shape[0]
    alive = ~walls.copy()
    ends = {tuple(s), tuple(g)}
    deg = lambda i, j: sum(1 for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= i + di < P and 0 <= j + dj < P and alive[i + di, j + dj])
    q = deque((i, j) for i in range(P) for j in range(P) if alive[i, j] and (i, j) not in ends and deg(i, j) <= 1)
    while q:
        i, j = q.popleft()
        if not alive[i, j] or deg(i, j) > 1:
            continue
        alive[i, j] = False
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < P and 0 <= b < P and alive[a, b] and (a, b) not in ends and deg(a, b) <= 1:
                q.append((a, b))
    return alive


# ---------------------------------------------------------------- build gate
def gate(M, family, Tmax):
    N = len(M["walls"])
    res = {}
    for algo in ("DEF", "BFS_S", "BFS_G"):
        r = rollout(M, algo, Tmax)
        assert (r["fixed_round"] >= 0).all(), f"{algo} did not reach a fixed point in {Tmax} rounds"
        if algo == "DEF":
            glob = np.array([global_def(M["walls"][k], M["s"][k], M["g"][k]) for k in range(N)])
            ok_sim = (r["final"] == glob).reshape(N, -1).all(1)
        else:
            src = M["s"] if algo == "BFS_S" else M["g"]
            ok = []
            for k in range(N):
                d = bfs(M["walls"][k], tuple(src[k]))
                code = r["last_state"][k]
                reach = d < BIG
                cd = np.where(code >= 4, code - 4, code - 1)
                ok.append(bool(((code > 0) == reach).all() and (cd[reach] == d[reach] % 3).all()))
            ok_sim = np.array(ok)
        ok_sol = (r["final"] == M["sol"]).reshape(N, -1).all(1)
        res[algo] = dict(match_global_sim=float(ok_sim.mean()), match_solution=float(ok_sol.mean()), n=N)
    return res


# ---------------------------------------------------------------- reader R
class Reader(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.c1 = nn.Conv2d(K + 3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 1, 1)

    def forward(self, code, x):  # code [B,P,P] int, x [B,3,P,P]
        oh = F.one_hot(code, self.K).permute(0, 3, 1, 2).float()
        return self.c2(F.relu(self.c1(torch.cat([oh, x], 1))))[:, 0]


def xin(M):
    N, P = M["walls"].shape[:2]
    x = np.zeros((N, 3, P, P), np.float32)
    x[:, 0] = M["walls"]
    x[np.arange(N), 1, M["s"][:, 0], M["s"][:, 1]] = 1
    x[np.arange(N), 2, M["g"][:, 0], M["g"][:, 1]] = 1
    return torch.from_numpy(x)


def fit_reader(states, M, K, steps, seed, bs=256, lr=3e-3):
    """states [R,N,P,P] codes; target = final path (the solution), per pixel."""
    torch.manual_seed(seed)
    R_, N = states.shape[:2]
    x = xin(M)
    y = torch.from_numpy(M["sol"].astype(np.float32))
    st = torch.from_numpy(states)  # int8
    net = Reader(K)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for it in range(steps):
        ri = torch.randint(0, R_, (bs,), generator=g)
        ni = torch.randint(0, N, (bs,), generator=g)
        lo = net(st[ri, ni].long(), x[ni])
        loss = F.binary_cross_entropy_with_logits(lo, y[ni], reduction="none").sum((1, 2)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def score(net, states, M, chunk=64):
    """S_R per round per maze: minus the summed per-pixel log loss on the path target. [R,N]"""
    x = xin(M)
    y = torch.from_numpy(M["sol"].astype(np.float32))
    R_, N = states.shape[:2]
    out = np.zeros((R_, N))
    for r in range(R_):
        st = torch.from_numpy(states[r]).long()
        for a in range(0, N, chunk):
            lo = net(st[a:a + chunk], x[a:a + chunk])
            out[r, a:a + chunk] = -F.binary_cross_entropy_with_logits(lo, y[a:a + chunk], reduction="none").sum((1, 2)).numpy()
    return out


def chunk_gains(S, c, T):
    """S [T+1,N] -> gains [n_chunks,N] for chunks [a, min(a+c,T)]."""
    starts = list(range(0, T, c))
    return np.array([S[min(a + c, T)] - S[a] for a in starts]), starts


def pct(a, q):
    return float(np.percentile(a, q))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_gate", type=int, default=1000)
    ap.add_argument("--n_rounds", type=int, default=1000)
    ap.add_argument("--n_def", type=int, default=1000)
    ap.add_argument("--n_fit", type=int, default=600)
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--fit_steps", type=int, default=3000)
    ap.add_argument("--p_grid", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P3/results/m0.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    out = dict(config=vars(a))
    log = lambda *s: print("[%5.0fs]" % (time.time() - t0), *s, flush=True)

    # (ii) DEF failure share vs p on 9x9 non-dead-end percolated test mazes (start degree >= 2, ties dropped)
    share = {}
    p_fix = None
    for p in [float(v) for v in a.p_grid.split(",")]:
        M = gen(9, "perc", p, a.n_def, seed=300 + int(p * 1000), deadend=False)
        M = {k: v[M["start_deg"] >= 2] for k, v in M.items()}
        M, nt = drop_ties(M)
        r = rollout(M, "DEF", 2000)
        wrong = ~(r["final"] == M["sol"]).reshape(len(M["walls"]), -1).all(1)
        share[p] = dict(share_wrong=float(wrong.mean()), n=int(len(wrong)), ties_dropped=nt)
        log("DEF failure share p=%.2f: %.3f (n=%d, ties dropped %d)" % (p, wrong.mean(), len(wrong), nt))
        if wrong.mean() >= 0.30:
            p_fix = p
            break
    out["def_failure_vs_p"] = share
    out["p_fixed"] = p_fix
    assert p_fix is not None, "no p in the grid reaches a 30% DEF failure share"

    # build gate: 1000 mazes per family (9x9 training-style endpoints), ties dropped
    gates = {}
    for fam, p in (("perfect", 0.0), ("perc", p_fix)):
        M = gen(9, fam, p, a.n_gate, seed=11, deadend=True)
        M, nt = drop_ties(M)
        gates[fam] = gate(M, fam, 5000)
        gates[fam]["ties_dropped"] = nt
        log("gate", fam, gates[fam])
    out["gate"] = gates
    gate_pass = all(gates[f][al]["match_global_sim"] == 1.0 for f in gates for al in gates[f] if al != "ties_dropped") and \
        all(gates[f][al]["match_solution"] == 1.0 for f in gates for al in gates[f] if al not in ("ties_dropped",)
            and not (al == "DEF" and f == "perc"))
    out["gate_pass"] = bool(gate_pass)
    log("GATE", "PASS" if gate_pass else "FAIL")

    # (i) rounds on 9x9 and 13x13 (training-style endpoints; percolated at p_fix and perfect)
    rounds = {}
    for n in (9, 13):
        for fam, p in (("perfect", 0.0), ("perc", p_fix)):
            M = gen(n, fam, p, a.n_rounds, seed=21 + n, deadend=True)
            M, nt = drop_ties(M)
            row = {}
            for algo in ("DEF", "BFS_S", "BFS_G"):
                r = rollout(M, algo, 5000)
                row[algo] = dict(path_p50=pct(r["path_round"], 50), path_p99=pct(r["path_round"], 99),
                                 fixed_p50=pct(r["fixed_round"], 50), fixed_p99=pct(r["fixed_round"], 99),
                                 path_max=int(r["path_round"].max()), n=int(len(r["path_round"])))
            rounds[f"{fam}_{n}x{n}"] = row
            log("rounds", fam, n, row)
    out["rounds"] = rounds
    p99 = max(rounds[f"perc_9x9"][al]["path_p99"] for al in ("BFS_S", "BFS_G"))
    T = int(max(30, math.ceil(p99 / 6.0) * 6))
    out["T"] = T
    out["T_rule"] = "max(30, p99 of BFS+backtrack path-final rounds on 9x9 percolated (both BFS variants), rounded up to a multiple of 6 so c=6 chunks tile T)"
    log("T =", T)

    # (iii) reader R on exact oracle codes; S_R per round and chunk gains (9x9 percolated, training endpoints)
    Mf = drop_ties(gen(9, "perc", p_fix, a.n_fit, seed=41, deadend=True))[0]
    Me = drop_ties(gen(9, "perc", p_fix, a.n_eval, seed=42, deadend=True))[0]
    rd = {}
    for algo, K in (("DEF", 2), ("BFS_S", 8), ("BFS_G", 8)):
        sf = rollout(Mf, algo, T, keep_states=True)["states"]
        re = rollout(Me, algo, T, keep_states=True)
        se = re["states"]
        net = fit_reader(sf, Mf, K, a.fit_steps, seed=1)
        S = score(net, se, Me)  # [T+1, N]
        Sm = S.mean(1)
        voi = S[-1] - S[0]
        row = dict(K=K, S_R_per_round=Sm.round(3).tolist(), S_R_final_mean=float(Sm[-1]),
                   total_voi_mean=float(voi.mean()), path_round_p50=pct(re["path_round"], 50))
        if algo != "DEF":
            # backtrack start = round the destination gets marked = flood distance L; chunks ending at or before it
            L = (Me["sol"].reshape(len(Me["sol"]), -1).sum(1) - 1)  # path length in pixel steps
            row["backtrack_start_p50"] = pct(L, 50)
        for c in sorted({1, 6, T}):
            gns, starts = chunk_gains(S, c, T)
            ends = np.array([min(s0 + c, T) for s0 in starts])
            d = dict(mean_gain_per_chunk=gns.mean(1).round(4).tolist(), chunk_starts=starts,
                     share_chunks_positive=float((gns > 1e-9).mean()))
            if algo != "DEF":
                pre = ends[:, None] <= (L[None, :] + 1)  # chunk ends before the backtrack mark leaves the destination
                d["pre_backtrack_share_of_voi"] = float((gns * pre).sum() / voi.sum())
            else:
                active = np.array([[s0 < re["path_round"][k] for k in range(len(voi))] for s0 in starts])  # chunk, maze
                d["share_active_maze_chunks_positive"] = float((gns[active] > 1e-9).mean()) if active.any() else None
                live = active.mean(1) >= 0.10  # chunks in which >= 10% of mazes are still being filled
                d["n_live_chunks"] = int(live.sum())
                d["min_mean_gain_live_chunk"] = float(min(gns[i].mean() for i in range(len(starts)) if live[i]))
                d["def_positive_every_live_chunk"] = bool(all(gns[i].mean() > 0 for i in range(len(starts)) if live[i]))
            row[f"c{c}"] = d
        rd[algo] = row
        log("reader", algo, {k: v for k, v in row.items() if k != "S_R_per_round"})
    out["reader"] = rd
    b_share = max(rd[al]["c6"]["pre_backtrack_share_of_voi"] for al in ("BFS_S", "BFS_G"))
    def_ok = rd["DEF"]["c6"]["def_positive_every_live_chunk"]
    out["prereg_reading"] = dict(bfs_pre_backtrack_share_c6_max=b_share, def_earns_every_active_chunk=bool(def_ok),
                                 prediction_b_settled=bool(b_share < 0.01 and def_ok),
                                 stage4_gate_ii_open=bool(not (b_share < 0.01 and def_ok)))
    log("prereg", out["prereg_reading"])
    out["sec"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
