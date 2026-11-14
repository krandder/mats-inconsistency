"""P16 episode generator.

An episode: hidden x ~ U{0,1}^nx, outputs y in {0,1}^w, m listed bit columns B, one candidate bit b born at row H+1.
Every column is a Col: prob(c=1 | x, parent listed columns) as a table (X, 8) over parent configs, or y-dependent
(leak slice), or Markov noise over rows. Every candidate type enumerates its full parameter grid (with weights), so
the generator and the L-all / L-train likelihood libraries use exactly the same instances.
Seeds: (namespace, index, sub) -> SeedSequence; namespaces are disjoint by construction (asserted in tests).
"""
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

P16_SEED = 0x16292026
NAMESPACES = {'train': 1, 'pilot': 2, 'tune': 3, 'test': 4, 'shift': 5, 'menu': 6, 'adv': 7, 'unit': 8, 'fresh': 9}
assert len(set(NAMESPACES.values())) == len(NAMESPACES)
H = 64

ID_TYPES = ['T0', 'T1', 'T2', 'T3a', 'T3b', 'T4', 'T5', 'T6']
LEAK_TYPES = ['Ly', 'Lpar', 'LinS', 'Lw4']
HP_TYPES = ['HP1', 'HP2', 'HP3a', 'HP3b']
HN_TYPES = ['HN1', 'HN2', 'HN3', 'HN4']
ALL_TYPES = ID_TYPES + HP_TYPES + HN_TYPES + LEAK_TYPES
POS_RETRY = {'T2', 'T4', 'T5', 'T6', 'HP1', 'HP2', 'HP3a', 'HP3b'}   # test: redraw until CMI >= 0.05
NEG_RETRY = {'HN2'}                                                  # test: redraw until CMI < 1e-12
TRAIN_FAMS = ['pal', 'even', 'const', 'w4']
HELD_FAMS = ['per2', 'sorted']
FLAT_FAMS = {'pal', 'even', 'const', 'per2'}
LISTED_W = {'T0': .1, 'T1': .25, 'T2': .2, 'T3a': .05, 'T3b': .1, 'T4': .1, 'T5': .1, 'T6': .1}
CB = ((np.arange(8)[:, None] >> np.arange(3)) & 1)                   # (8,3) parent bits of a cfg


def rng_for(ns, idx, sub=0):
    return np.random.default_rng(np.random.SeedSequence([P16_SEED, NAMESPACES[ns], int(idx), int(sub)]))


_BT = {}


def bits_table(n):
    if n not in _BT:
        t = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
        t.flags.writeable = False
        _BT[n] = t
    return _BT[n]


def op_apply(op, cols):  # cols (..., k) 0/1
    s = cols.sum(-1)
    k = cols.shape[-1]
    if op == 'copy': return cols[..., 0]
    if op == 'not': return 1 - cols[..., 0]
    if op == 'and': return (s == k).astype(np.int64)
    if op == 'or': return (s > 0).astype(np.int64)
    if op == 'xor': return s % 2
    if op == 'maj': return (2 * s > k).astype(np.int64)
    raise ValueError(op)


def fam_mask(fam, w):
    YB = bits_table(w)
    wt = YB.sum(1)
    if fam == 'pal': return np.all(YB == YB[:, ::-1], 1)
    if fam == 'even': return wt % 2 == 0
    if fam == 'const': return (wt == 0) | (wt == w)
    if fam == 'w4': return wt >= 4
    if fam == 'per2': return np.all(YB[:, 2:] == YB[:, :-2], 1)
    if fam == 'sorted': return np.all(YB[:, 1:] >= YB[:, :-1], 1)
    raise ValueError(fam)


@dataclass
class Col:
    typ: str
    desc: str
    par: tuple = ()
    P1: np.ndarray = None      # (X, 8) prob of 1 given x and parent cfg
    ydep: np.ndarray = None    # (2^w,) prob of 1 given y
    markov: float = 0.0        # > 0: Markov noise over rows (stickiness), marginal Bern(.5), independent of all else
    kt: bool = False


@dataclass
class Episode:
    ns: str
    idx: int
    nx: int
    w: int
    m: int
    R: np.ndarray
    gtab: np.ndarray           # (w, X)
    ztab: np.ndarray           # (X,)
    branch: str
    eta: float
    fam: str
    py: np.ndarray             # (X, 2^w)
    cols: list                 # m listed + candidate (last)
    ctype: str
    cmi: float = float('nan')
    tries: int = 1
    info: dict = field(default_factory=dict)

    @property
    def cand(self):
        return self.cols[-1]


def xfun_col(typ, desc, f, eps=0.0):
    f = f.astype(np.float64)
    P = f * (1 - eps) + (1 - f) * eps
    return Col(typ, desc, (), np.repeat(P[:, None], 8, 1))


def make_y_model(rng, nx, w, fam, force_mix):
    X = 2 ** nx
    XB = bits_table(nx)
    r = int(rng.integers(2, 6))
    R = np.sort(rng.choice(nx, r, replace=False))
    gtab = np.zeros((w, X), np.int64)
    for j in range(w):
        nb = int(rng.integers(1, min(3, r) + 1))
        bits = rng.choice(R, nb, replace=False)
        op = 'copy' if nb == 1 else ['and', 'or', 'xor'][int(rng.integers(3))]
        gtab[j] = op_apply(op, XB[:, bits])
    nb = int(rng.integers(1, min(3, r) + 1))
    zb = rng.choice(R, nb, replace=False)
    while True:
        tt = rng.integers(0, 2, 2 ** nb)
        if 0 < tt.sum() < 2 ** nb: break
    ztab = tt[(XB[:, zb] << np.arange(nb)).sum(1)]
    branch = 'mix' if force_mix else ('bit' if rng.random() < .5 else 'mix')
    eta = float([0, .05, .15][int(rng.integers(3))])
    YB = bits_table(w)
    pbit = np.ones((X, 2 ** w))
    for j in range(w):
        pbit *= np.where(YB[None, :, j] == gtab[j][:, None], 1 - eta, eta)
    if branch == 'mix':
        S = fam_mask(fam, w).astype(np.float64)
        py = ztab[:, None] * (S / S.sum())[None] + (1 - ztab[:, None]) * pbit
    else:
        py = pbit
    return R, gtab, ztab, branch, eta, py


def col_mi_y(col, py):
    """I(c; Y) for a parentless x-column."""
    p1 = col.P1[:, 0]
    X = len(p1)
    pc_y = np.stack([((1 - p1)[:, None] * py).sum(0), (p1[:, None] * py).sum(0)]) / X
    pc = pc_y.sum(1, keepdims=True); pyy = pc_y.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(pc_y > 0, pc_y * np.log2(pc_y / (pc * pyy)), 0)
    return float(t.sum())


# ---------------------------------------------------------------- type parameter grids
def type_params(typ, ep):
    """[(weight, params)] for every instance of the type in this episode (weights sum to 1)."""
    nx, w, m, X = ep.nx, ep.w, ep.m, 2 ** ep.nx
    out = []

    def add(items, wt):
        items = list(items)
        for it in items: out.append((wt / len(items), it))
    if typ == 'T0':
        add([(p,) for p in np.linspace(.1, .9, 17)], 1)
    elif typ == 'T1':
        add([('s', i, ng, e) for i in range(nx) for ng in (0, 1) for e in (0, .1, .25)], .5)
        add([('p', i, k, op, e) for i, k in combinations(range(nx), 2) for op in ('and', 'or', 'xor') for e in (0, .1, .25)], .5)
    elif typ == 'T2':
        add([(j, e) for j in range(w) for e in (0, .1, .25, .4)], 1)
    elif typ == 'T3a':
        add([(k, ng) for k in range(m) for ng in (0, 1)], 1)
    elif typ == 'T3b':
        if m >= 2: add([(k, l, op) for k, l in combinations(range(m), 2) for op in ('xor', 'and')], 1)
    elif typ in ('T4', 'T6'):
        add([(j, k) for j in range(w) for k in range(m)], 1)
    elif typ == 'T5':
        add([('z', 0)], 1 / 3); add([('z', .1), ('z', .25)], 1 / 3); add([('zg', j) for j in range(w)], 1 / 3)
    elif typ == 'HP1':
        if m >= 2: add([(j, k, l) for j in range(w) for k, l in combinations(range(m), 2)], 1)
    elif typ == 'HP2':
        if m >= 3: add([(j,) + t for j in range(w) for t in combinations(range(m), 3)], 1)
        elif m == 2: add([(j, 0, 1) for j in range(w)], 1)
    elif typ == 'HP3a':
        add([('z', 0)], .5); add([('z', .1), ('z', .25)], .5)
    elif typ == 'HP3b':
        add([('z', 0)], 1)
    elif typ == 'HN1':
        k = ep.info.get('hn1_k')
        if k is not None: add([(k, .1), (k, .25)], 1)
    elif typ == 'HN2':
        outR = [i for i in range(nx) if i not in set(ep.R.tolist())]
        add([t + (op,) for t in combinations(outR, 3) for op in ('maj', 'xor')], 1)
    elif typ == 'HN3':
        add([(0.9,)], 1)
    elif typ == 'HN4':
        if m >= 3: add([t + (op,) for t in combinations(range(m), 3) for op in ('maj', 'xor')], 1)
        elif m == 2: add([(0, 1, 'majcoin')], 1)
    elif typ == 'Ly':
        add([(j, e) for j in range(w) for e in (.1, .25)], 1)
    elif typ in ('Lpar', 'LinS', 'Lw4'):
        add([()], 1)
    else:
        raise ValueError(typ)
    return out


def build_col(typ, prm, ep):
    nx, w, X = ep.nx, ep.w, 2 ** ep.nx
    XB = bits_table(nx)
    g, z = ep.gtab, ep.ztab
    d = f'{typ}{prm}'
    pa = CB  # (8,3)
    if typ == 'T0':
        return Col(typ, d, (), np.full((X, 8), prm[0]), kt=True)
    if typ == 'T1':
        if prm[0] == 's':
            _, i, ng, e = prm
            return xfun_col(typ, d, XB[:, i] ^ ng, e)
        _, i, k, op, e = prm
        return xfun_col(typ, d, op_apply(op, XB[:, [i, k]]), e)
    if typ == 'T2':
        j, e = prm
        return xfun_col(typ, d, g[j], e)
    if typ in ('T5', 'HP3a', 'HP3b'):
        kind, a = prm
        if kind == 'z': return xfun_col(typ, d, z, a)
        return xfun_col(typ, d, z & g[a], 0)
    if typ == 'T3a':
        k, ng = prm
        f = (pa[:, 0] ^ ng).astype(float)
        return Col(typ, d, (k,), np.repeat(f[None], X, 0))
    if typ == 'T3b':
        k, l, op = prm
        f = op_apply(op, pa[:, :2]).astype(float)
        return Col(typ, d, (k, l), np.repeat(f[None], X, 0))
    if typ == 'T4':
        j, k = prm
        return Col(typ, d, (k,), (g[j][:, None] ^ pa[None, :, 0]).astype(float))
    if typ == 'T6':
        j, k = prm
        P = np.where(pa[None, :, 0] == 1, g[j][:, None].astype(float), .5)
        return Col(typ, d, (k,), P)
    if typ == 'HP1':
        j, k, l = prm
        return Col(typ, d, (k, l), (g[j][:, None] ^ pa[None, :, 0] ^ pa[None, :, 1]).astype(float))
    if typ == 'HP2':
        if ep.m >= 3:
            j, k, l, n = prm
            gate = op_apply('maj', pa)
            par = (k, l, n)
        else:
            j, k, l = prm
            gate = pa[:, 0] ^ pa[:, 1]
            par = (k, l)
        return Col(typ, d, par, np.where(gate[None] == 1, g[j][:, None].astype(float), .5))
    if typ == 'HN1':
        k, e = prm
        f = pa[:, 0].astype(float)
        return Col(typ, d, (k,), np.repeat((f * (1 - e) + (1 - f) * e)[None], X, 0))
    if typ == 'HN2':
        i, k, l, op = prm
        return xfun_col(typ, d, op_apply(op, XB[:, [i, k, l]]), 0)
    if typ == 'HN3':
        return Col(typ, d, (), np.full((X, 8), .5), markov=prm[0])
    if typ == 'HN4':
        if prm[2] == 'majcoin':
            f = np.where(pa[:, 0] == pa[:, 1], pa[:, 0], .5)
            return Col(typ, d, (0, 1), np.repeat(f[None].astype(float), X, 0))
        k, l, n, op = prm
        return Col(typ, d, (k, l, n), np.repeat(op_apply(op, pa)[None].astype(float), X, 0))
    YB = bits_table(w)
    if typ == 'Ly':
        j, e = prm
        f = YB[:, j].astype(float)
        return Col(typ, d, ydep=f * (1 - e) + (1 - f) * e)
    if typ == 'Lpar':
        return Col(typ, d, ydep=(YB.sum(1) % 2).astype(float))
    if typ == 'LinS':
        return Col(typ, d, ydep=fam_mask(ep.fam, w).astype(float))
    if typ == 'Lw4':
        return Col(typ, d, ydep=(YB.sum(1) >= 4).astype(float))
    raise ValueError(typ)


def sample_type_col(typ, ep, rng):
    ps = type_params(typ, ep)
    if not ps:
        return None
    wts = np.array([p[0] for p in ps]); wts = wts / wts.sum()
    return build_col(typ, ps[int(rng.choice(len(ps), p=wts))][1], ep)


def make_structure(rng, ns, idx, ctype, m, w, nx=10):
    fam_list = TRAIN_FAMS
    fam = {'HP3a': 'per2', 'HP3b': 'sorted'}.get(ctype, fam_list[int(rng.integers(len(fam_list)))])
    force_mix = ctype in ('T5', 'HP3a', 'HP3b')
    R, gtab, ztab, branch, eta, py = make_y_model(rng, nx, w, fam, force_mix)
    ep = Episode(ns, idx, nx, w, 0, R, gtab, ztab, branch, eta, fam, py, [], ctype)
    ltypes = list(LISTED_W); lw = np.array([LISTED_W[t] for t in ltypes])
    for i in range(m):
        ep.m = i
        while True:
            t = ltypes[int(rng.choice(len(ltypes), p=lw / lw.sum()))]
            c = sample_type_col(t, ep, rng)
            if c is not None: break
        ep.cols.append(c)
    ep.m = m
    if ctype == 'HN1':
        cands = [(col_mi_y(c, py), k) for k, c in enumerate(ep.cols) if not c.par and c.ydep is None and not c.markov]
        ep.info['hn1_k'] = max(cands)[1] if cands else 0
    c = sample_type_col(ctype, ep, rng)
    assert c is not None, (ctype, m)
    ep.cols.append(c)
    return ep


def make_episode(ns, idx, ctype=None, m=None, w=None, nx=10, label=True):
    """Test-style episode (retry rules applied when label=True). Train: ctype/m/w drawn when None."""
    from oracle import cmi_episode
    rng0 = rng_for(ns, idx, 0)
    if ctype is None:
        ctype = (LEAK_TYPES[int(rng0.integers(4))] if rng0.random() < .1 else ID_TYPES[int(rng0.integers(8))])
    if m is None: m = int(rng0.integers(2, 13))
    if w is None: w = int(rng0.integers(4, 7))
    for tr in range(1, 31):
        ep = make_structure(rng_for(ns, idx, tr), ns, idx, ctype, m, w, nx)
        ep.tries = tr
        if not label:
            return ep
        ep.cmi = cmi_episode(ep)
        if ctype in POS_RETRY and ep.cmi < .05: continue
        if ctype in NEG_RETRY and ep.cmi >= 1e-12: continue
        return ep
    return ep


def sample_rows(ep, T, rng, birth=H):
    """Rows: xi (T,), y (T,) int, V (T, m+1) int8 with -1 = absent (candidate absent before index `birth`)."""
    X = 2 ** ep.nx
    xi = rng.integers(0, X, T)
    cp = np.cumsum(ep.py[xi], 1)
    y = (cp < rng.random(T)[:, None] * cp[:, -1:]).sum(1)
    y = np.minimum(y, 2 ** ep.w - 1)
    V = np.zeros((T, len(ep.cols)), np.int8)
    for c, col in enumerate(ep.cols):
        if col.ydep is not None:
            p = col.ydep[y]
        else:
            cfg = np.zeros(T, np.int64)
            for i, pi in enumerate(col.par): cfg |= V[:, pi].astype(np.int64) << i
            p = col.P1[xi, cfg]
        v = (rng.random(T) < p).astype(np.int8)
        if col.markov:
            ref = rng.random(T) >= col.markov
            ref[0] = True
            if birth < T: ref[birth] = True
            last = np.maximum.accumulate(np.where(ref, np.arange(T), 0))
            v = v[last]
        V[:, c] = v
    V[:birth, -1] = -1
    return xi, y, V


def ybits(y, w):
    return ((y[:, None] >> np.arange(w)) & 1).astype(np.int8)


def test_design(n_per, strata=(2, 6, 12), types=None, ns='test', w=6):
    """List of (idx, ctype, m): each type x stratum gets n_per episodes with disjoint indices."""
    types = types or ALL_TYPES
    out = []
    for t in types:
        ti = ALL_TYPES.index(t)
        for si, m in enumerate(strata):
            for r in range(n_per):
                out.append((ti * 100000 + si * 10000 + r, t, m))
    return out
