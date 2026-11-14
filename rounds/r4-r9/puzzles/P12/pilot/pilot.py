"""P12 pilot: (1) DFS / percolated maze generator on a cell grid, unique-shortest-path labels;
(2) flow-conservation arbitrageur = sum-product BP on degree-constraint factors;
(3) timing of a width-32 DT-recall step on CPU."""
import sys, time, math, json
from collections import deque
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
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

def shift_from_nb(m):
    """m (B,n,n,4) factor->edge messages indexed by (cell, dir). Return for each (c,d) the message
    the neighbour across d sent on the same edge: m[nb(c,d), opp(d)]."""
    out = torch.zeros_like(m)
    out[:, :, :-1, 0] = m[:, :, 1:, 2]   # E neighbour's W message
    out[:, :-1, :, 1] = m[:, 1:, :, 3]   # S neighbour's N message
    out[:, :, 1:, 2] = m[:, :, :-1, 0]
    out[:, 1:, :, 3] = m[:, :-1, :, 1]
    return out

NEG = -30.0
def bp_step(m, unary, openm, endm, damp=0.0, maxprod=False):
    """unary (B,n,n,4) edge log-odds seen from each cell (symmetric), openm (B,n,n,4) bool,
    endm (B,n,n) bool. Returns new factor->edge messages."""
    nin = unary + shift_from_nb(m)            # variable->factor message into cell c on edge d
    nin = torch.where(openm, nin, torch.full_like(nin, NEG))
    red = (lambda x, dim: x.max(dim).values) if maxprod else torch.logsumexp
    new = []
    for d in range(4):
        o = [k for k in range(4) if k != d]; a, b, c = (nin[..., k] for k in o)
        l1 = red(torch.stack([a, b, c], -1), -1)                       # log sum r (exactly one other on)
        l2 = red(torch.stack([torch.zeros_like(a), a+b, a+c, b+c], -1), -1)  # log(1 + e2)
        md = torch.where(endm, -l1, l1 - l2)
        new.append(md)
    new = torch.stack(new, -1).clamp(-30, 30)
    new = torch.where(openm, new, torch.zeros_like(new))
    return (1 - damp) * new + damp * m

def beliefs(m, unary, openm):
    b = unary + m + shift_from_nb(m)
    return torch.where(openm, b, torch.full_like(b, NEG))

def local_incoherence(p, openm, endm):
    """Semantic loss of each cell's degree constraint under independent edge prices p (B,n,n,4):
    -log P(valid local configuration). Sum over cells = arbitrage 'profit' proxy."""
    p = torch.where(openm, p, torch.zeros_like(p)).clamp(1e-6, 1-1e-6)
    q = 1 - p; P0 = q.prod(-1)
    r = p / q
    e1 = r.sum(-1); e2 = sum(r[..., i]*r[..., j] for i in range(4) for j in range(i+1, 4))
    valid = torch.where(endm, P0*e1, P0*(1+e2))
    return -torch.log(valid.clamp_min(1e-30))

def batch(mazes):
    B = len(mazes); n = mazes[0][0].shape[0]
    op = torch.zeros(B, n, n, 4, dtype=torch.bool); en = torch.zeros(B, n, n, dtype=torch.bool)
    tgt = torch.zeros(B, n, n, 4, dtype=torch.bool)
    for k, (oE, oS, S, G, pE, pS, L) in enumerate(mazes):
        op[k] = torch.tensor(edge_grids(oE, oS)); tgt[k] = torch.tensor(edge_grids(pE, pS))
        en[k][S] = True; en[k][G] = True
    return op, en, tgt

def run_arb0(n, p, num, seed, Tmax, maxprod=False, cost=0.0, damp=0.0):
    rng = np.random.default_rng(seed)
    mz = [gen_maze(n, p, rng) for _ in range(num)]
    op, en, tgt = batch(mz)
    un = torch.full(op.shape, -cost)
    m = torch.zeros(op.shape)
    solved_at = torch.full((num,), -1)
    inc0 = None
    for t in range(1, Tmax+1):
        m = bp_step(m, un, op, en, damp=damp, maxprod=maxprod)
        b = beliefs(m, un, op)
        pred = (b > 0) & op
        ok = (pred == tgt).all(dim=(1, 2, 3))
        solved_at = torch.where((solved_at < 0) & ok, torch.full_like(solved_at, t), solved_at)
        if t == 1 or t == Tmax:
            inc = local_incoherence(torch.sigmoid(b), op, en).sum(dim=(1, 2)).mean().item()
            if t == 1: inc0 = inc
            else: incT = inc
    final_ok = ok.float().mean().item()
    L = np.array([z[6] for z in mz])
    sa = solved_at.numpy()
    return {"n": n, "p": p, "maxprod": maxprod, "cost": cost, "Tmax": Tmax, "exact_final": final_ok,
            "solved_frac": float((sa > 0).mean()), "iters_median": float(np.median(sa[sa > 0])) if (sa > 0).any() else None,
            "iters_max": int(sa.max()), "pathlen_median": float(np.median(L)), "incoh_t1": inc0, "incoh_T": incT}

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "arb":
        for (n, p, mp, cost, T) in [(9,0,False,0,200),(15,0,False,0,400),(33,0,False,0,1500),(65,0,False,0,5000),
                                   (9,0.1,False,0,300),(15,0.1,False,0,400),(33,0.1,False,0,1500),
                                   (9,0.1,True,0.1,300),(15,0.1,True,0.1,400),(33,0.1,True,0.1,1500)]:
            t0 = time.time(); r = run_arb0(n, p, 40 if n < 60 else 16, 0, T, maxprod=mp, cost=cost)
            r["sec"] = round(time.time()-t0, 1); print(json.dumps(r), flush=True)
    else:
        class DT(nn.Module):
            def __init__(s, w=32):
                super().__init__(); s.emb = nn.Conv2d(6, w, 3, padding=1)
                s.proj = nn.Conv2d(2*w, w, 3, padding=1)
                s.blocks = nn.ModuleList([nn.Conv2d(w, w, 3, padding=1) for _ in range(4)])
                s.head = nn.Sequential(nn.Conv2d(w, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 5, 3, padding=1))
            def forward(s, x, T):
                e = s.emb(x); h = torch.zeros_like(e)
                for _ in range(T):
                    h = s.proj(torch.cat([h, e], 1))
                    for k in range(0, 4, 2):
                        h = F.relu(h + s.blocks[k+1](F.relu(s.blocks[k](h))))
                return s.head(h)
        net = DT(32); print("params", sum(p.numel() for p in net.parameters()))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        for n, B, T in [(9, 32, 30), (33, 4, 300)]:
            x = torch.randn(B, 6, n, n)
            if n == 9:
                t0 = time.time()
                for _ in range(3):
                    out = net(x, T); loss = out.pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
                print(json.dumps({"train_step_sec": round((time.time()-t0)/3, 3), "n": n, "B": B, "T": T}))
            with torch.no_grad():
                t0 = time.time(); net(x, T); dt = time.time()-t0
                print(json.dumps({"infer_sec_per_maze_iter": dt/(B*T), "n": n, "B": B, "T": T}))
