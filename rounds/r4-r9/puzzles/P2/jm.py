"""P2 core: recurrent trunk, U / M heads, exact scores, shortest-path DAG dynamic programs,
quadtree latent-parent market, decoders and checkers. Works in 1D (toy) and 2D (mazes).

Streams (all additive along depth, every arm starts uniform, i.e. all-zero logits):
  U family : m        per-pixel logit
  M arms   : theta    (B,K,*S) emission logits of each code
             trans[l] (B,K,K,*G_l) transition logits log p(z_child | z_parent) (tree only)
             root     (B,K,1..) root prior (tree) / prior (B,K,*G0) per tile (tile, global)
Scores (natural log, per example):
  U    : sum_open log sigma(+-m_i)
  U-any: sum_open log sigma(-m_i) + logsumexp_{shortest paths} sum_{i in path} m_i
  M    : exact log sum_z P(z) prod_i P(y_i | z_leaf(i)) by one upward sum-product pass
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e6
CLAMP = 30.0
MOVES = ((1, 0), (0, 1), (-1, 0), (0, -1))  # D, R, U, L
U_ARMS = ("U", "U-canon", "U-min", "U-any")
M_ARMS = ("M-tree", "M-tile", "M-global")


def softclamp(v):
    return CLAMP * torch.tanh(v / CLAMP)


def conv(dim, cin, cout, k):
    C = nn.Conv1d if dim == 1 else nn.Conv2d
    return C(cin, cout, k, padding=k // 2)


# ---------------------------------------------------------------- trunk
class Trunk(nn.Module):
    """DT-Recall-style recurrent block: tied weights, raw input re-injected every iteration,
    current prices fed back through a 1x1 conv."""

    def __init__(self, dim=2, cin=3, w=48, nfb=1, nres=2):
        super().__init__()
        self.proj = conv(dim, cin, w, 3)
        self.fb = conv(dim, nfb, w, 1)
        self.recall = conv(dim, w + cin, w, 3)
        self.res = nn.ModuleList([conv(dim, w, w, 3) for _ in range(2 * nres)])

    def init_h(self, x):
        return F.relu(self.proj(x))

    def forward(self, h, x, fb):
        z = F.relu(self.recall(torch.cat([h + self.fb(fb), x], 1)))
        for i in range(0, len(self.res), 2):
            z = F.relu(z + self.res[i + 1](F.relu(self.res[i](z))))
        return z


# ---------------------------------------------------------------- tree geometry
class Geom:
    """Padded dyadic tree over leaf tiles. S: lattice shape (tuple, 1 or 2 dims).
    kind 'tree': leaves of `leaf` positions per dim, P = leaf * 2^Lv >= max(S), levels 0..Lv.
    kind 'tile': the same leaves, no parents. kind 'global': one leaf = the whole lattice."""

    def __init__(self, S, leaf, kind):
        self.S, self.dim, self.kind = tuple(S), len(S), kind
        if kind == "global":
            self.leaf, self.Lv, self.P = max(S), 0, max(S)
        else:
            Lv = 0
            while leaf * 2 ** Lv < max(S):
                Lv += 1
            self.leaf, self.P = leaf, leaf * 2 ** Lv
            self.Lv = Lv if kind == "tree" else 0
        self.G0 = self.P // self.leaf

    def pad(self, x, value=0.0):
        """Pad trailing spatial dims of x from S to P."""
        pads = []
        for s in reversed(self.S):
            pads += [0, self.P - s]
        return F.pad(x, pads, value=value)

    def crop(self, x):
        if self.dim == 1:
            return x[..., : self.S[0]]
        return x[..., : self.S[0], : self.S[1]]

    def tiles(self, x, op="sum"):
        """(..., P[,P]) -> (..., G0[,G0]) reducing each leaf tile."""
        lead = x.shape[: -self.dim]
        if self.dim == 1:
            y = x.reshape(*lead, self.G0, self.leaf)
            return y.sum(-1) if op == "sum" else y.amax(-1)
        y = x.reshape(*lead, self.G0, self.leaf, self.G0, self.leaf)
        return y.sum((-3, -1)) if op == "sum" else y.amax((-3, -1))

    def up(self, x, op="sum"):
        """Children -> parent (factor 2 per dim)."""
        lead = x.shape[: -self.dim]
        if self.dim == 1:
            G = x.shape[-1] // 2
            y = x.reshape(*lead, G, 2)
            return y.sum(-1) if op == "sum" else y.amax(-1)
        G = x.shape[-1] // 2
        y = x.reshape(*lead, G, 2, G, 2)
        return y.sum((-3, -1)) if op == "sum" else y.amax((-3, -1))

    def down(self, x, f=2):
        y = x
        for d in range(self.dim):
            y = y.repeat_interleave(f, dim=-1 - d)
        return y

    def valid(self, device):
        return self.pad(torch.ones(self.S, device=device))

    def node_of(self, idx_tuple, level):
        """Node coordinates at `level` for a lattice position."""
        size = self.leaf * 2 ** level
        return tuple(int(i) // size for i in idx_tuple)


def pool_levels(f, geom):
    """Masked [mean, max] pooling of features f (B,C,*S) for every tree level -> list of (B,2C,*G_l)."""
    v = geom.valid(f.device)
    fp = geom.pad(f)
    s = geom.tiles(fp * v)
    c = geom.tiles(v)
    mx = geom.tiles(fp.masked_fill(v == 0, -1e4), "max")
    out = []
    for lvl in range(geom.Lv + 1):
        if lvl > 0:
            s, c, mx = geom.up(s), geom.up(c), geom.up(mx, "max")
        mean = s / c.clamp(min=1)
        m2 = torch.where(c > 0, mx, torch.zeros_like(mx))
        out.append(torch.cat([mean, m2], 1))
    return out


# ---------------------------------------------------------------- model
class Model(nn.Module):
    def __init__(self, arm, dim=2, cin=3, w=48, K=4, leaf=8, nres=2, fdim=12):
        super().__init__()
        self.arm, self.dim, self.K, self.leaf = arm, dim, K, leaf
        self.is_m = arm in M_ARMS
        self.kind = {"M-tree": "tree", "M-tile": "tile", "M-global": "global"}.get(arm)
        self.trunk = Trunk(dim, cin, w, 1 + (K if self.is_m else 0), nres)
        if not self.is_m:
            self.mhead = conv(dim, w, 1, 1)
        else:
            self.thead = conv(dim, w, K, 1)
            self.fproj = conv(dim, w, fdim, 1)
            if self.kind == "tree":
                self.trans = nn.Linear(4 * fdim, K * K)  # ONE level-shared head (child, parent)
            self.root = nn.Linear(2 * fdim, K)

    def geom(self, S):
        return Geom(S, self.leaf, self.kind) if self.is_m else None

    # ---- state
    def init_state(self, x):
        B, S = x.shape[0], tuple(x.shape[2:])
        st = {"h": self.trunk.init_h(x)}
        if not self.is_m:
            st["m"] = torch.zeros(B, 1, *S, device=x.device)
            return st
        g = self.geom(S)
        K = self.K
        st["theta"] = torch.zeros(B, K, *S, device=x.device)
        if g.kind == "tree":
            st["trans"] = [torch.zeros(B, K, K, *([g.G0 // 2 ** l] * g.dim), device=x.device) for l in range(g.Lv)]
            st["root"] = torch.zeros(B, K, *([1] * g.dim), device=x.device)
        else:
            st["root"] = torch.zeros(B, K, *([g.G0] * g.dim), device=x.device)
        return st

    def leaf_prior(self, st, g):
        """Prior marginals of leaf codes (B,K,*G0) by a downward pass."""
        pi = torch.softmax(st["root"], 1)
        if g.kind != "tree":
            return pi
        for l in range(g.Lv - 1, -1, -1):
            T = torch.softmax(st["trans"][l], 2)  # (B,Kp,Kc,*G)
            pi = (g.down(pi).unsqueeze(2) * T).sum(1)
        return pi

    def pixel_marg(self, st):
        """(B,1,*S) logit of the marginal P(y_i = 1) and leaf-code prior broadcast to pixels (B,K,*S)."""
        if not self.is_m:
            return st["m"], None
        g = self.geom(tuple(st["theta"].shape[2:]))
        pi = self.leaf_prior(st, g)
        pix = g.crop(g.down(pi, g.leaf) if g.kind != "global" else pi.expand(*pi.shape[:2], *([g.P] * g.dim)))
        q = (pix * torch.sigmoid(st["theta"])).sum(1, keepdim=True).clamp(1e-7, 1 - 1e-7)
        return torch.log(q) - torch.log1p(-q), pix

    def step(self, st, x):
        m, pix = self.pixel_marg(st)
        fb = torch.sigmoid(m) if pix is None else torch.cat([torch.sigmoid(m), pix], 1)
        h = self.trunk(st["h"], x, fb)
        new = {"h": h}
        if not self.is_m:
            new["m"] = softclamp(st["m"] + self.mhead(h))
            return new
        g = self.geom(tuple(x.shape[2:]))
        new["theta"] = softclamp(st["theta"] + self.thead(h))
        feats = pool_levels(self.fproj(h), g)
        if g.kind == "tree":
            K = self.K
            tr = []
            for l in range(g.Lv):
                inp = torch.cat([feats[l], g.down(feats[l + 1])], 1)
                d = self.trans(inp.movedim(1, -1)).movedim(-1, 1)
                tr.append(softclamp(st["trans"][l] + d.reshape(d.shape[0], K, K, *d.shape[2:])))
            new["trans"] = tr
            new["root"] = softclamp(st["root"] + self.root(feats[g.Lv].movedim(1, -1)).movedim(-1, 1))
        else:
            new["root"] = softclamp(st["root"] + self.root(feats[0].movedim(1, -1)).movedim(-1, 1))
        return new

    def run(self, x, T, st=None, keep=None, cb=None):
        """T iterations from st (or from scratch). cb(t, st) is called after iterations in keep."""
        if st is None:
            st = self.init_state(x)
        for t in range(1, T + 1):
            st = self.step(st, x)
            if cb is not None and (keep is None or t in keep):
                cb(t, st)
        return st


@torch.no_grad()
def pair_trunk(model, ref):
    """Paired seeds: copy a reference trunk (1 feedback channel) into model.trunk, so every arm of a
    seed starts from the same trunk; extra feedback channels (code priors) keep their own init."""
    t = model.trunk
    for name in ("proj", "recall"):
        getattr(t, name).load_state_dict(getattr(ref, name).state_dict())
    t.res.load_state_dict(ref.res.state_dict())
    t.fb.weight[:, :1] = ref.fb.weight
    t.fb.bias.copy_(ref.fb.bias)


def detach_state(st):
    return {k: ([t.detach() for t in v] if isinstance(v, list) else v.detach()) for k, v in st.items()}


# ---------------------------------------------------------------- scores
def unary_terms(m, y):
    return y * F.logsigmoid(m) + (1 - y) * F.logsigmoid(-m)


def score_unary(m, y, mask):
    """m (B,1,*S) logits, y (B,*S) {0,1}, mask (B,*S): sum_open log sigma(+-m)."""
    return (unary_terms(m[:, 0], y) * mask).flatten(1).sum(1)


def leaf_loglik(theta, y, mask, g, maxmode=False):
    """(B,K,*G0): per leaf and code, sum over its open pixels of log P(y_i | code)
    (maxmode: log max_y P(y_i | code), for the MAP decode)."""
    if maxmode:
        t = F.logsigmoid(theta.abs())
    else:
        t = y.unsqueeze(1) * F.logsigmoid(theta) + (1 - y.unsqueeze(1)) * F.logsigmoid(-theta)
    t = t * mask.unsqueeze(1)
    if g.kind == "global":
        return t.flatten(2).sum(2).reshape(t.shape[0], t.shape[1], *([1] * g.dim))
    return g.tiles(g.pad(t))


def score_m(st, y, mask, g):
    """Exact joint log score log P(y) (B,) by one upward sum-product pass."""
    beta = leaf_loglik(st["theta"], y, mask, g)
    if g.kind != "tree":
        lp = torch.log_softmax(st["root"], 1)
        return torch.logsumexp(lp + beta, 1).flatten(1).sum(1)
    for l in range(g.Lv):
        logT = torch.log_softmax(st["trans"][l], 2)  # (B,Kp,Kc,*G)
        msg = torch.logsumexp(logT + beta.unsqueeze(1), 2)  # (B,Kp,*G)
        beta = g.up(msg)
    lp = torch.log_softmax(st["root"], 1)
    return torch.logsumexp(lp + beta, 1).flatten(1).sum(1)


def tree_map(st, mask, g):
    """Exact joint MAP over (codes, pixels): upward max-product + backtrack.
    Returns pixel decode (B,*S) bool and leaf codes (B,*G0)."""
    beta = leaf_loglik(st["theta"], None, mask, g, maxmode=True)
    lp = torch.log_softmax(st["root"], 1)
    if g.kind != "tree":
        z = (lp + beta).argmax(1)  # (B,*G0)
    else:
        args = []
        for l in range(g.Lv):
            logT = torch.log_softmax(st["trans"][l], 2)
            val, arg = (logT + beta.unsqueeze(1)).max(2)  # (B,Kp,*G)
            args.append(arg)
            beta = g.up(val)
        z = (lp + beta).argmax(1)  # (B,1..)
        for l in range(g.Lv - 1, -1, -1):
            zp = g.down(z)  # parent code at each child
            z = torch.gather(args[l], 1, zp.unsqueeze(1)).squeeze(1)
    return pixels_from_codes(st["theta"], z, g) > 0, z


def pixels_from_codes(theta, z, g):
    """theta (B,K,*S), leaf codes (B,*G0) -> emission logit of the chosen code per pixel (B,*S)."""
    if g.kind == "global":
        zp = z.reshape(z.shape[0], *([1] * g.dim)).expand(z.shape[0], *theta.shape[2:])
    else:
        zp = g.crop(g.down(z, g.leaf))
    return torch.gather(theta, 1, zp.unsqueeze(1)).squeeze(1)


def tree_sample(st, g, nsamp, gen=None):
    """Ancestral samples of (codes at every level, pixels). Returns list of codes per level
    (level 0 = leaves), each (N,B,*G_l), and pixels (N,B,*S) bool."""
    B = st["theta"].shape[0]

    def cat(logits):  # logits (B,K,...) -> (N,B,...)
        p = torch.softmax(logits, 1).movedim(1, -1)
        flat = p.reshape(-1, p.shape[-1])
        s = torch.multinomial(flat, nsamp, replacement=True, generator=gen)  # (M,N)
        return s.t().reshape(nsamp, *p.shape[:-1])

    if g.kind != "tree":
        z = cat(st["root"])
        codes = [z]
    else:
        z = cat(st["root"])  # (N,B,1..)
        codes = [z]
        for l in range(g.Lv - 1, -1, -1):
            T = st["trans"][l]  # (B,Kp,Kc,*G)
            zp = g.down(z)  # (N,B,*G)
            Te = T.unsqueeze(0).expand(nsamp, *T.shape)
            idx = zp.unsqueeze(2).unsqueeze(2).expand(nsamp, B, 1, T.shape[2], *zp.shape[2:])
            logits = torch.gather(Te, 2, idx).squeeze(2)  # (N,B,Kc,*G)
            p = torch.softmax(logits, 2).movedim(2, -1)
            flat = p.reshape(-1, p.shape[-1])
            z = torch.multinomial(flat, 1, generator=gen).reshape(p.shape[:-1])
            codes.insert(0, z)
    zl = codes[0]
    th = st["theta"].unsqueeze(0).expand(nsamp, *st["theta"].shape).reshape(nsamp * B, *st["theta"].shape[1:])
    em = pixels_from_codes(th, zl.reshape(nsamp * B, *zl.shape[2:]), g).reshape(nsamp, B, *st["theta"].shape[2:])
    u = torch.rand(em.shape, generator=gen, device=em.device)
    return codes, u < torch.sigmoid(em)


# ---------------------------------------------------------------- lattice helpers (2D)
def nb(x, d, fill):
    """Value of x at (i+di, j+dj) placed at (i,j); `fill` outside the lattice."""
    di, dj = d
    L1, L2 = x.shape[-2], x.shape[-1]
    y = torch.full_like(x, fill)
    si = slice(max(di, 0), L1 + min(di, 0))
    ti = slice(max(-di, 0), L1 + min(-di, 0))
    sj = slice(max(dj, 0), L2 + min(dj, 0))
    tj = slice(max(-dj, 0), L2 + min(-dj, 0))
    y[..., ti, tj] = x[..., si, sj]
    return y


def any_nb(x):
    """OR of the four lattice neighbours of a bool map (B,L,L)."""
    P = F.pad(x, (1, 1, 1, 1))
    L1, L2 = x.shape[-2], x.shape[-1]
    v = [P[..., 1 + di:1 + di + L1, 1 + dj:1 + dj + L2] for di, dj in MOVES]
    return (v[0] | v[1]) | (v[2] | v[3])


def bfs_dist(op, src):
    """Batched BFS distance by dilation. op (B,L,L) bool, src (B,) flat index. -1 = unreachable.
    Syncs with the host only every 16 layers (extra layers after the front empties are no-ops)."""
    B, L, _ = op.shape
    d = torch.full((B, L, L), -1, dtype=torch.long, device=op.device)
    front = torch.zeros(B, L * L, dtype=torch.bool, device=op.device)
    front[torch.arange(B, device=op.device), src] = True
    front = front.view(B, L, L)
    seen = front.clone()
    t = 0
    while True:
        for _ in range(16):
            d = torch.where(front, t, d)
            t += 1
            front = any_nb(front) & op & ~seen
            seen |= front
        if not bool(front.any()):
            return d


def dag_forward(w, ds, dag, mode="lse"):
    """Layered DP over the shortest-path DAG: F_i = w_i + reduce_{pred j} F_j, from the start.
    mode 'lse' (log-sum over paths) or 'max' (max-sum). Off-DAG = NEG.
    Fast form (few kernels per layer): F lives in a NEG-padded buffer and the four neighbours are
    shifted views of it; neighbours of a layer-t pixel lie in layers t-1 (set) and t+1 (still NEG),
    so reducing over all four equals reducing over the predecessors. Checked against
    dag_forward_ref and path enumeration in tests.py. The buffer is updated in place, so when a
    gradient is needed (U-any's training loss) the out-of-place dag_forward_ref is used instead."""
    if torch.is_grad_enabled() and w.requires_grad:
        return dag_forward_ref(w, ds, dag, mode)
    B, L1, L2 = w.shape
    lay = torch.where(dag, ds, torch.full_like(ds, -1))
    Dmax = int(lay.max())
    P = torch.full((B, L1 + 2, L2 + 2), NEG, dtype=w.dtype, device=w.device)
    inner = P[:, 1:-1, 1:-1]
    inner.copy_(torch.where(lay == 0, w, torch.full_like(w, NEG)))
    v = [P[:, 1 + di:1 + di + L1, 1 + dj:1 + dj + L2] for di, dj in MOVES]
    for t in range(1, Dmax + 1):
        if mode == "lse":
            red = torch.logaddexp(torch.logaddexp(v[0], v[1]), torch.logaddexp(v[2], v[3]))
        else:
            red = torch.maximum(torch.maximum(v[0], v[1]), torch.maximum(v[2], v[3]))
        inner.copy_(torch.where(lay == t, w + red, inner))
    return inner.clone()


def dag_backtrack(Fv, ds, dag, g_idx, gumbel=False, gen=None):
    """Backtrack one path from the goal choosing, at each layer, the predecessor with the largest
    F (plus Gumbel noise if gumbel: then P(pred j) = exp(F_j) / sum exp(F_j')).
    Fast form: a per-pixel table of the 4 neighbours plus a 'stay' slot (valid only at the start),
    all Gumbel noise drawn in one call, path pixels scattered once at the end."""
    B, L, _ = Fv.shape
    dev = Fv.device
    ar = torch.arange(B, device=dev)
    Ff = Fv.reshape(B, -1)
    lay = torch.where(dag, ds, torch.full_like(ds, -9)).reshape(B, -1).int()
    ii, jj = torch.meshgrid(torch.arange(L, device=dev), torch.arange(L, device=dev), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    mv = torch.tensor(MOVES + ((0, 0),), device=dev)
    ni, nj = ii[:, None] + mv[:, 0], jj[:, None] + mv[:, 1]  # (LL,5)
    inb = (ni >= 0) & (ni < L) & (nj >= 0) & (nj < L)
    tab = ni.clamp(0, L - 1) * L + nj.clamp(0, L - 1)  # (LL,5) neighbour flat index
    ok = torch.zeros(B, L * L, 5, dtype=torch.bool, device=dev)
    for k in range(4):  # neighbour k is a DAG predecessor: one layer lower (off-DAG layer is -9)
        ok[:, :, k] = inb[:, k] & (lay[:, tab[:, k]] == lay - 1) & (lay >= 1)
    ok[:, :, 4] = lay == 0  # stay put once at the start
    cur = g_idx.clone()
    steps = int(lay[ar, cur].max())
    if steps <= 0:
        path = torch.zeros(B, L * L, dtype=torch.bool, device=dev)
        path[ar, cur] = True
        return path.view(B, L, L)
    noise = None
    if gumbel:
        u = torch.rand((steps, B, 5), generator=gen, device=dev).clamp(1e-12, 1 - 1e-12)
        noise = -torch.log(-torch.log(u))
    visited = [cur]
    for s in range(steps):
        nidx = tab[cur]  # (B,5)
        sc = Ff.gather(1, nidx)
        sc = torch.where(ok[ar, cur], sc if noise is None else sc + noise[s], torch.full_like(sc, 10 * NEG))
        cur = nidx.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)
        visited.append(cur)
    path = torch.zeros(B, L * L, dtype=torch.bool, device=dev)
    path.scatter_(1, torch.stack(visited, 1), True)
    return path.view(B, L, L)


def dag_forward_ref(w, ds, dag, mode="lse"):
    """Layered DP over the shortest-path DAG: F_i = w_i + reduce_{pred j} F_j, from the start.
    mode 'lse' (log-sum over paths) or 'max' (max-sum). Off-DAG = NEG."""
    Fv = torch.where(dag & (ds == 0), w, torch.full_like(w, NEG))
    Dmax = int(torch.where(dag, ds, torch.zeros_like(ds)).max())
    for t in range(1, Dmax + 1):
        cand = torch.stack([nb(Fv, mv, NEG) for mv in MOVES])
        red = torch.logsumexp(cand, 0) if mode == "lse" else cand.max(0).values
        Fv = torch.where(dag & (ds == t), w + red, Fv)
    return Fv


def dag_backtrack_ref(Fv, ds, dag, g_idx, gumbel=False, gen=None):
    """Backtrack one path from the goal choosing, at each layer, the predecessor with the largest
    F (plus Gumbel noise if gumbel: then P(pred j) = exp(F_j) / sum exp(F_j'))."""
    B, L, _ = Fv.shape
    ar = torch.arange(B, device=Fv.device)
    Ff, dsf, dagf = Fv.reshape(B, -1), ds.reshape(B, -1), dag.reshape(B, -1)
    path = torch.zeros(B, L * L, dtype=torch.bool, device=Fv.device)
    cur = g_idx.clone()
    path[ar, cur] = True
    steps = int(dsf[ar, cur].max())
    mv = torch.tensor(MOVES, device=Fv.device)
    for _ in range(steps):
        ci, cj = cur // L, cur % L
        ni, nj = ci[:, None] + mv[:, 0], cj[:, None] + mv[:, 1]  # (B,4)
        inb = (ni >= 0) & (ni < L) & (nj >= 0) & (nj < L)
        nidx = (ni.clamp(0, L - 1) * L + nj.clamp(0, L - 1))
        dcur = dsf[ar, cur]
        valid = inb & dagf.gather(1, nidx) & (dsf.gather(1, nidx) == dcur[:, None] - 1)
        sc = Ff.gather(1, nidx)
        if gumbel:
            u = torch.rand(sc.shape, generator=gen, device=sc.device).clamp(1e-12, 1 - 1e-12)
            sc = sc - torch.log(-torch.log(u))
        sc = torch.where(valid, sc, torch.full_like(sc, 10 * NEG))
        nxt = nidx.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)
        cur = torch.where(dcur > 0, nxt, cur)
        path[ar, cur] = True
    return path.view(B, L, L)


def sample_uniform_path(ds, dag, g_idx, gen=None):
    logN = dag_forward(torch.zeros(ds.shape, device=ds.device), ds, dag, "lse")
    return dag_backtrack(logN, ds, dag, g_idx, gumbel=True, gen=gen)


def viterbi_path(w, ds, dag, g_idx):
    return dag_backtrack(dag_forward(w, ds, dag, "max"), ds, dag, g_idx)


def score_any(m, op, ds, dag, g_idx):
    """U-any: exact log P(a factorised sample is some shortest path) (B,)."""
    mm = m[:, 0]
    base = (F.logsigmoid(-mm) * op).flatten(1).sum(1)
    Fv = dag_forward(mm, ds, dag, "lse")
    return base + Fv.flatten(1).gather(1, g_idx[:, None]).squeeze(1)


# ---------------------------------------------------------------- decoders / checkers
def decode_thr(m):
    return m[:, 0] > 0


def decode_cl(S, op, s_idx, g_idx, sweeps=5):
    """Local cleanup, <= 5 sweeps: drop path pixels with <= 1 path neighbour (not start/goal),
    fill 1-pixel gaps between two path pixels on opposite sides."""
    B, L, _ = S.shape
    sg = torch.zeros(B, L * L, dtype=torch.bool, device=S.device)
    ar = torch.arange(B, device=S.device)
    sg[ar, s_idx] = True
    sg[ar, g_idx] = True
    sg = sg.view(B, L, L)
    S = S & op
    for _ in range(sweeps):
        n = [nb(S, mv, False) for mv in MOVES]
        cnt = sum(x.long() for x in n)
        rem = S & (cnt <= 1) & ~sg
        gap = op & ~S & ((n[0] & n[2]) | (n[1] & n[3]))
        S = (S & ~rem) | gap
    return S


def vsp(S, ds, dg, D, g_idx):
    """Decoded set is exactly one valid shortest start-goal path (fast form, see tests for the
    reference: start+goal in set, |set| = D+1, connected, open pixels only)."""
    B = S.shape[0]
    ondag = (ds >= 0) & (dg >= 0) & (ds + dg == D[:, None, None])
    c1 = ~(S & ~ondag).flatten(1).any(1)
    c2 = S.flatten(1)[torch.arange(B, device=S.device), g_idx]
    c3 = S.flatten(1).sum(1) == D + 1
    haspred = torch.zeros_like(S)
    for mv in MOVES:
        haspred |= nb(S, mv, False) & (nb(ds, mv, -5) == ds - 1)
    c4 = ~(S & (ds > 0) & ~haspred).flatten(1).any(1)
    return c1 & c2 & c3 & c4


def connected_from(S, src):
    B, L, _ = S.shape
    R = torch.zeros(B, L * L, dtype=torch.bool, device=S.device)
    R[torch.arange(B, device=S.device), src] = True
    R = R.view(B, L, L) & S
    while True:
        prev = R
        for _ in range(16):
            g = R.clone()
            for mv in MOVES:
                g |= nb(R, mv, False)
            R = g & S
        if torch.equal(R, prev):
            return R


def coherent(S, op, s_idx, g_idx):
    """Decoded set is a simple start-goal path of any length."""
    B, L, _ = S.shape
    ar = torch.arange(B, device=S.device)
    cnt = sum(nb(S, mv, False).long() for mv in MOVES)
    sg = torch.zeros(B, L * L, dtype=torch.bool, device=S.device)
    sg[ar, s_idx] = True
    sg[ar, g_idx] = True
    sg = sg.view(B, L, L)
    ok = ~(S & ~op).flatten(1).any(1)
    ok &= S.flatten(1)[ar, s_idx] & S.flatten(1)[ar, g_idx]
    ok &= ~(S & sg & (cnt != 1)).flatten(1).any(1)
    ok &= ~(S & ~sg & (cnt != 2)).flatten(1).any(1)
    R = connected_from(S, s_idx)
    ok &= ~(S & ~R).flatten(1).any(1)
    return ok


def near_miss(S, ds, dag, D, g_idx):
    """Exact distance |S xor P| to the nearest shortest path P (max-overlap DP over the DAG)."""
    Fv = dag_forward(S.float(), ds, dag, "max")
    ov = Fv.flatten(1).gather(1, g_idx[:, None]).squeeze(1)
    return S.flatten(1).sum(1).float() + (D + 1).float() - 2 * ov


def contested_mask(ds, dag, D):
    """Pixels on some but not all shortest paths (layer holds > 1 DAG pixel)."""
    B, L, _ = ds.shape
    Dm = int(D.max())
    lay = torch.where(dag, ds, torch.full_like(ds, Dm + 1)).flatten(1)
    cnt = torch.zeros(B, Dm + 2, device=ds.device, dtype=torch.long)
    cnt.scatter_add_(1, lay, torch.ones_like(lay))
    return dag & (cnt.gather(1, lay).view(B, L, L) > 1)
