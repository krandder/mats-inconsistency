"""ck.py -- helpers of the independent checker. Written by the checker, not by the worker.
Uses the worker's files only for: the network class and attention masks (core.RelFormer, core.relmask: the
checkpoints cannot be loaded without them), the checkpoints, and the saved data arrays. Everything that knows
a Sudoku rule here (peers, validity, exact counting) is my own code and is cross-checked below; oracle.py and
eval_lib.py of the worker are never imported."""
import sys, math, itertools
import numpy as np, torch

W = "/home/agent/workspaces/mats/work/deep-study/wf7/sudoku"
sys.path.insert(0, W)
import core                      # network definition only
torch.set_num_threads(1)

BOX = {4: (2, 2), 6: (2, 3)}


def geom(K):
    br, bc = BOX[K]; N = K * K
    r = np.arange(N) // K; c = np.arange(N) % K; b = (r // br) * (K // bc) + c // bc
    return N, r, c, b


def peers(K):
    N, r, c, b = geom(K)
    P = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])
    P[np.arange(N), np.arange(N)] = False
    return P


def n_violations(X, K):
    """number of unordered peer pairs with equal digits, per complete board [m, N]."""
    X = np.asarray(X); P = peers(K)
    return ((X[:, :, None] == X[:, None, :]) & P[None]).sum((1, 2)) // 2


def is_valid(X, K):
    return n_violations(X, K) == 0


class M:
    """frozen network; my own forward wrapper (log-softmax of the head). calls are counted."""

    def __init__(self, name, K):
        ck = torch.load(f"{W}/ckpt/{name}.pt", map_location="cpu")
        self.m = core.RelFormer(**ck["cfg"]); self.m.load_state_dict(ck["state"]); self.m.eval()
        self.K = K; self.N = K * K; self.mask = core.relmask(K); self.calls = 0; self.name = name
        self.params = sum(p.numel() for p in self.m.parameters()); self.kmax = self.m.kmax
        self.flops = 2 * self.params * self.N

    @torch.no_grad()
    def logp(self, s, flag=None, chunk=2048):
        s = torch.as_tensor(s, dtype=torch.long); self.calls += len(s); out = []
        for a in range(0, len(s), chunk):
            x = s[a:a + chunk]; x = torch.where(x < 0, torch.full_like(x, self.kmax), x)
            f = torch.zeros_like(x) if flag is None else torch.as_tensor(flag[a:a + chunk]).long()
            out.append(torch.log_softmax(self.m(x, f, self.mask), -1))
        return torch.cat(out)


class Trust:
    """the worker's trust net as an order policy: lowest score first. My own wrapper around the checkpoint."""

    def __init__(self, path, K):
        ck = torch.load(path, map_location="cpu"); self.m = core.RelFormer(**ck["cfg"]); self.m.load_state_dict(ck["state"])
        self.m.eval(); self.mask = core.relmask(K); self.calls = 0
        self.params = sum(p.numel() for p in self.m.parameters()); self.flops = 2 * self.params * K * K

    @torch.no_grad()
    def __call__(self, s, lp):
        self.calls += len(s); emp = s < 0
        p = lp.exp() * emp[..., None]
        ent = -(p * lp.clamp(min=-50)).sum(-1, keepdim=True)
        x = torch.where(s < 0, torch.full_like(s, self.m.kmax), s)
        return self.m(x, torch.zeros_like(x), self.mask, extra=torch.cat([p, ent], -1)).squeeze(-1)


# order rules: score per cell, lowest first (my own statements of the rules named in PREREG section 4)
def o_entropy(s, lp):
    p = lp.exp(); return -(p * lp.clamp(min=-50)).sum(-1)


def o_maxprob(s, lp):
    return -lp.max(-1).values


def o_margin(s, lp):
    t = lp.exp().topk(2, -1).values; return -(t[..., 0] - t[..., 1])


def o_anti(s, lp):
    return lp.max(-1).values


def o_raster(s, lp):
    return torch.arange(s.shape[1], dtype=torch.float32).expand(s.shape[0], -1).clone()


def walk(net, order, start, X=None, flag=None, greedy=False, floor=0.0, gen=None, stop_below=None, keep=False):
    """my own walk of the state graph. start [B, N] (-1 empty). One cell per step, chosen by `order`.
    X given: route to that board (returns its log density under the net's atoms). Else greedy or sampled values.
    Returns final boards, logp (sum of log atoms of the realised values), logq (log density of the draws incl. floor),
    steps used, and if keep: lists of (state, lp, cell, value) per step."""
    s = start.clone(); B, N = s.shape; logp = torch.zeros(B, dtype=torch.float64); logq = torch.zeros(B, dtype=torch.float64)
    live = torch.ones(B, dtype=torch.bool); used = torch.zeros(B, dtype=torch.long); trace = []
    while True:
        act = (s < 0).any(1) & live
        if not act.any():
            break
        ia = torch.where(act)[0]; sa = s[ia]
        lp = net.logp(sa, None if flag is None else flag[ia]); used[ia] += 1
        sc = order(sa, lp).masked_fill(sa >= 0, float("inf")); cell = sc.argmin(1); ar = torch.arange(len(ia))
        lpc = lp[ar, cell]
        if X is not None:
            v = X[ia, cell]
        elif greedy:
            v = lpc.argmax(1)
        else:
            pr = (1 - floor) * lpc.exp() + floor / lpc.shape[1]
            v = torch.multinomial(pr, 1, generator=gen).squeeze(1)
        lv = lpc[ar, v].double(); logp[ia] += lv
        logq[ia] += torch.log((1 - floor) * lv.exp() + floor / lpc.shape[1]) if floor > 0 else lv
        if keep:
            trace.append((ia, sa.clone(), lp, cell, v))
        s[ia, cell] = v
        if stop_below is not None:
            live = live & (logp >= stop_below)
    return dict(s=s, logp=logp, logq=logq, used=used, live=live, trace=trace)


def pair_table(net):
    """Q[i, u, j, v] = log q(x_j = v | only cell i revealed, with value u). N*K forward passes: every one-cell context."""
    K, N = net.K, net.N
    S = torch.full((N * K, N), -1, dtype=torch.long)
    for i in range(N):
        for u in range(K):
            S[i * K + u, i] = u
    return net.logp(S)[:, :, :K].reshape(N, K, N, -1).double()


def pair_score(Q, X, stop_below=None):
    """min over ordered pairs i != j of Q[i, x_i, j, x_j]; from the table (no further network call).
    With stop_below also returns the number of one-cell contexts read (i = 0, 1, ...) before the minimum drops below it."""
    X = torch.as_tensor(X, dtype=torch.long); B, N = X.shape; ar = torch.arange(N)
    # rows[b, i, j] = Q[i, X[b, i], j, X[b, j]]
    rows = Q[ar[None, :, None], X[:, :, None], ar[None, None, :], X[:, None, :]]
    rows[:, ar, ar] = float("inf")
    per_i = rows.min(2).values                                   # [B, N]
    sc = per_i.min(1).values
    if stop_below is None:
        return sc
    run = torch.cummin(per_i, 1).values; below = run < stop_below
    used = torch.where(below.any(1), below.float().argmax(1) + 1, torch.full((B,), N))
    return sc, used


# ---------------------------------------------------------------- exact world (my own counter)
class Exact:
    """exact conditionals of the uniform law over ALL valid K x K grids, from a table of base grids whose first
    row is the identity (every grid is one base grid with one digit relabelling). Table checked in __init__."""

    def __init__(self, K, B=None):
        self.K = K; self.N = K * K
        if B is None:
            B = np.load(f"{W}/data/k{K}_base.npz")["B"] if K == 6 else np.load(f"{W}/data/k4.npz")["B"]
        B = np.asarray(B, dtype=np.int64)
        assert (B[:, :K] == np.arange(K)).all(), "first row must be the identity"
        assert len(np.unique(B, axis=0)) == len(B), "duplicate base grids"
        for a in range(0, len(B), 4000):
            assert is_valid(B[a:a + 4000], K).all(), "invalid base grid"
        self.B = B; self.fact = [math.factorial(k) for k in range(K + 1)]

    def cond(self, s):
        """s [N] with -1 empty. Returns (number of completions, P [N, K]) with P[i, v] = p(x_i = v | s); rows of
        filled cells are zero."""
        K, N, B = self.K, self.N, self.B
        s = np.asarray(s); C = np.where(s >= 0)[0]
        if len(C) == 0:
            return len(B) * self.fact[K], np.full((N, K), 1.0 / K)
        BC = B[:, C]; sC = s[C]
        mn = np.full((len(B), K), 99); mx = np.full((len(B), K), -1)
        for e in range(K):
            hit = BC == e
            mn[:, e] = np.where(hit, sC[None, :], 99).min(1); mx[:, e] = np.where(hit, sC[None, :], -1).max(1)
        ok = ((mn == mx) | (mx == -1)).all(1)                                  # base digit -> one target digit
        a = np.where(mx >= 0, mx, 100 + np.arange(K)[None, :])                  # assigned target digit or a dummy
        srt = np.sort(a, 1); ok &= (srt[:, 1:] != srt[:, :-1]).all(1)          # injective
        m = len(np.unique(sC)); nok = int(ok.sum())
        P = np.zeros((N, K))
        if nok == 0:
            return 0, P
        amap = np.where(mx[ok] >= 0, mx[ok], -1)                                # [nok, K]
        A = np.take_along_axis(amap, B[ok], 1)                                  # target digit at each cell, -1 = free digit
        unused = [v for v in range(K) if v not in set(sC.tolist())]
        for v in range(K):
            P[:, v] = (A == v).sum(0) * self.fact[K - m]
        if unused:
            free = (A == -1).sum(0) * self.fact[K - m - 1]
            for v in unused:
                P[:, v] += free
        tot = nok * self.fact[K - m]
        P = P / tot; P[C] = 0.0
        return tot, P


def count_fast(ex, s):
    """number of completions only (used when generating puzzles). Same answer as ex.cond(s)[0]; checked in c3 'fresh'."""
    K, B = ex.K, ex.B; s = np.asarray(s); ok = np.ones(len(B), bool); reps = []
    for d in range(K):
        cd = np.where(s == d)[0]
        if len(cd) == 0:
            continue
        e = B[:, cd[0]]
        if len(cd) > 1:
            ok &= (B[:, cd[1:]] == e[:, None]).all(1)
        for e2 in reps:
            ok &= e2 != e
        reps.append(e)
    return int(ok.sum()) * ex.fact[K - len(reps)]


def brute_count(s, K, cap=10 ** 7):
    """independent backtracking count of the completions of s (for cross-checking Exact on dense states)."""
    P = [np.where(p)[0].tolist() for p in peers(K)]; x = list(map(int, s)); N = K * K
    for i in range(N):
        if x[i] >= 0 and any(x[j] == x[i] for j in P[i]):
            return 0

    def rec():
        best = None; bc = None
        for i in range(N):
            if x[i] < 0:
                used = {x[j] for j in P[i]}; c = [v for v in range(K) if v not in used]
                if best is None or len(c) < len(bc):
                    best, bc = i, c
                    if len(c) <= 1:
                        break
        if best is None:
            return 1
        t = 0
        for v in bc:
            x[best] = v; t += rec(); x[best] = -1
            if t >= cap:
                break
        return t
    return rec()


def boot(x, n=2000, seed=0):
    x = np.asarray(x, dtype=np.float64); rng = np.random.default_rng(seed)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def sample_grids(K, split, n, seed):
    """uniform grids of the orbits of one split (0 train, 1 validation, 2 test) of the worker's 6x6 split; my own draw."""
    d = np.load(f"{W}/data/k6_base.npz"); B = d["B"][d["base_split"] == split].astype(np.int64)
    rng = np.random.default_rng(seed); g = B[rng.integers(len(B), size=n)]
    perm = np.stack([rng.permutation(K) for _ in range(n)])
    return torch.from_numpy(np.take_along_axis(perm, g, 1))
