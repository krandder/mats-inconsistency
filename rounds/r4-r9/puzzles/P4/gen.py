"""P4 programs: random layered Boolean circuits with planted nodes, hand circuits
H1/H2/H3, and exact program-level tables (stage 0).

A program is a list of levels; level 0 is the input x (NIN bits), levels 1..3
are hidden, the last level holds the output gates. A node is a dict
  {type, srcs, role, name}
with srcs a list of (kind, idx, neg):
  kind 'x' : input bit idx (any level may read x, as the nets do)
  kind 'n' : node idx of the previous level
  kind 'm' : MAJ3 of three previous-level nodes idx=(i, j, k)  (triplicate copies)
role: gate | copy | epi | inert | out.  Candidates at a level = all its nodes.
"""
import itertools
import math
import numpy as np

NIN = 12
TYPES = ['XOR2', 'XOR3', 'AND2', 'OR2', 'MAJ3']
ARITY = {'XOR2': 2, 'XOR3': 3, 'AND2': 2, 'OR2': 2, 'MAJ3': 3, 'ID': 1, 'CONST0': 0}
EPS_NOISE = np.array([.25] * 4 + [.02] * 4)


def allx(nin=NIN):
    return ((np.arange(2 ** nin)[:, None] >> np.arange(nin)) & 1).astype(np.int8)


def gate(t, a):
    if t == 'XOR2':
        return a[0] ^ a[1]
    if t == 'XOR3':
        return a[0] ^ a[1] ^ a[2]
    if t == 'AND2':
        return a[0] & a[1]
    if t == 'OR2':
        return a[0] | a[1]
    if t == 'MAJ3':
        return ((a[0].astype(np.int16) + a[1] + a[2]) >= 2).astype(np.int8)
    if t == 'ID':
        return a[0].copy()
    raise ValueError(t)


def node_val(nd, X, prev):
    """X: (N, nin) int8; prev: (n_prev, N) int8 values of the previous level."""
    if nd['type'] == 'CONST0':
        return np.zeros(X.shape[0], np.int8)
    if nd['type'] == 'PAR':  # parity of input bits (H2 only)
        v = np.zeros(X.shape[0], np.int8)
        for _, j, s in nd['srcs']:
            v = v ^ X[:, j] ^ np.int8(s)
        return v
    a = []
    for kind, idx, neg in nd['srcs']:
        if kind == 'x':
            v = X[:, idx]
        elif kind == 'n':
            v = prev[idx]
        else:
            v = gate('MAJ3', [prev[i] for i in idx])
        a.append(v ^ np.int8(neg))
    return gate(nd['type'], a)


def level_vals(prog, l, X, prev):
    return np.array([node_val(nd, X, prev) for nd in prog['levels'][l]], dtype=np.int8)


def values(prog, X):
    """All level values: list V[0..D] with V[0] = X.T, V[l] = (n_l, N)."""
    V = [X.T.astype(np.int8)]
    for l in range(1, len(prog['levels'])):
        V.append(level_vals(prog, l, X, V[-1]))
    return V


def run_from(prog, l, hl, X):
    """Outputs when level l holds hl (n_l, N) and x = X (base input) for later x reads."""
    h = hl
    for k in range(l + 1, len(prog['levels'])):
        h = level_vals(prog, k, X, h)
    return h


def _consumers(prog, l):
    """Set of node ids of level l that are read by level l+1."""
    s = set()
    for nd in prog['levels'][l + 1]:
        for kind, idx, _ in nd['srcs']:
            if kind == 'n':
                s.add(idx)
            elif kind == 'm':
                s.update(idx)
    return s


# ---------------------------------------------------------------- random circuits
def gen_base(rng, nlev=3, G=6, NOUT=8, split_cones=False):
    """Prototype generator (/tmp/p4review/prog.py): each gate reads the previous
    level; constant gates and duplicates up to negation are rejected."""
    X = allx()
    for _ in range(1000):
        levels = [None]
        prev_vals = X.T
        nprev = NIN
        ok = True
        for l in range(1, nlev + 2):
            n = G if l <= nlev else NOUT
            gates, vs = [], []
            for g in range(n):
                pool = np.arange(nprev)
                if split_cones and l == nlev + 1:
                    pool = np.arange(0, nprev // 2) if g < NOUT // 2 else np.arange(nprev // 2, nprev)
                for _ in range(400):
                    t = rng.choice(TYPES)
                    k = ARITY[t]
                    if k > len(pool):
                        continue
                    ins = rng.choice(pool, k, replace=False)
                    neg = rng.integers(0, 2, k)
                    kind = 'x' if l == 1 else 'n'
                    nd = dict(type=str(t), srcs=[(kind, int(i), int(s)) for i, s in zip(ins, neg)],
                              role='out' if l == nlev + 1 else 'gate', name=f'L{l}g{g}')
                    v = node_val(nd, X, prev_vals)
                    if v.min() == v.max():
                        continue
                    if any((v == w).all() or (v != w).all() for w in vs):
                        continue
                    break
                else:
                    ok = False
                    break
                gates.append(nd)
                vs.append(v)
            if not ok:
                break
            levels.append(gates)
            prev_vals = np.array(vs)
            nprev = n
        if ok:
            return dict(levels=levels, nin=NIN, name='rand')
    raise RuntimeError('generator failed')


def flip_loo_program(prog, l, X, xi, xp):
    """Reach r_k: fraction of output bits flipped when node k alone is resampled."""
    V = values(prog, X)
    Y = V[-1][:, xi]
    hx, hxp = V[l][:, xi], V[l][:, xp]
    r = []
    for k in range(hx.shape[0]):
        h = hx.copy()
        h[k] = hxp[k]
        r.append((run_from(prog, l, h, X[xi]) != Y).mean())
    return np.array(r)


def entropy_codes(codes):
    c = np.bincount(codes)
    p = c[c > 0] / codes.size
    return float(-(p * np.log2(p)).sum())


def pack(bits):
    """bits (k, N) -> int codes (N,)"""
    if bits.shape[0] == 0:
        return np.zeros(bits.shape[1], np.int64)
    return (bits.astype(np.int64) << np.arange(bits.shape[0])[:, None]).sum(0)


def mi_bits(A, Y):
    """I(A; Y) in bits by counting; A (k, N), Y (m, N) binary."""
    a, y = pack(A), pack(Y)
    j = a * (1 << Y.shape[0]) + y
    return entropy_codes(a) + entropy_codes(y) - entropy_codes(j)


def add_plants(base, rng, X=None):
    """Triplicate one gate per hidden level (copies read downstream through MAJ3),
    add one epiphenomenal copy (max-reach at levels 1 and 3, max-MI at level 2,
    second-highest MI if that is the max-reach gate) and one inert node."""
    X = allx() if X is None else X
    nlev = len(base['levels']) - 2
    xi = np.random.default_rng(99).integers(0, len(X), 4096)
    xp = np.random.default_rng(98).integers(0, len(X), 4096)
    Vb = values(base, X)
    Y = Vb[-1]
    epi_src = {}
    for l in range(1, nlev + 1):
        r = flip_loo_program(base, l, X, xi, xp)
        mi = np.array([mi_bits(Vb[l][k:k + 1], Y) for k in range(len(Vb[l]))])
        if l == 2:
            order = np.argsort(-mi, kind='stable')
            src = int(order[0]) if order[0] != int(np.argmax(r)) else int(order[1])
        else:
            src = int(np.argmax(r))
        epi_src[l] = src
    prog = dict(levels=[None] + [[dict(nd, srcs=list(nd['srcs'])) for nd in lv] for lv in base['levels'][1:]],
                nin=base['nin'], name=base['name'])
    plants = {}
    for l in range(1, nlev + 1):
        lv = prog['levels'][l]
        cons = sorted(_consumers(prog, l))
        g = int(rng.choice(cons))
        src_nd = lv[g]
        epi_nd_src = lv[epi_src[l]]
        # new node list: gates except g, then 3 copies of g, epi, inert
        keep = [i for i in range(len(lv)) if i != g]
        newidx = {old: new for new, old in enumerate(keep)}
        copies = [len(keep) + j for j in range(3)]
        newlv = [lv[i] for i in keep]
        newlv += [dict(src_nd, role='copy', name=f'{src_nd["name"]}c{j}') for j in range(3)]
        newlv.append(dict(epi_nd_src, role='epi', name=f'L{l}epi({epi_nd_src["name"]})'))
        newlv.append(dict(type='CONST0', srcs=[], role='inert', name=f'L{l}inert'))
        prog['levels'][l] = newlv
        # rewire readers at level l+1
        for nd in prog['levels'][l + 1]:
            srcs = []
            for kind, idx, neg in nd['srcs']:
                if kind == 'n':
                    srcs.append(('m', tuple(copies), neg) if idx == g else ('n', newidx[idx], neg))
                elif kind == 'm':
                    srcs.append(('m', tuple(newidx[i] for i in idx), neg))
                else:
                    srcs.append((kind, idx, neg))
            nd['srcs'] = srcs
        plants[l] = dict(trip=src_nd['name'], copies=copies,
                         epi=len(newlv) - 2, inert=len(newlv) - 1, epi_src_name=epi_nd_src['name'],
                         epi_src=(newidx[epi_src[l]] if epi_src[l] != g else copies[0]))
    prog['plants'] = plants
    # the planted program computes the same outputs as the base
    assert (values(prog, X)[-1] == Y).all()
    return prog


def random_circuit(seed, split_cones=False):
    rng = np.random.default_rng(seed)
    base = gen_base(rng, split_cones=split_cones)
    prog = add_plants(base, rng)
    prog['name'] = ('sep' if split_cones else 'rand') + str(seed)
    return prog


# ---------------------------------------------------------------- hand circuits
def N(t, srcs, role='gate', name=''):
    return dict(type=t, srcs=[(k, i, s) for k, i, s in srcs], role=role, name=name)


def h1():
    """p = parity(x1..x8) decides 7 of 11 outputs, yet I(p; Y) = 0."""
    L1 = [N('XOR2', [('x', 0, 0), ('x', 1, 0)], name='a12'), N('XOR2', [('x', 2, 0), ('x', 3, 0)], name='a34'),
          N('XOR2', [('x', 4, 0), ('x', 5, 0)], name='a56'), N('XOR2', [('x', 6, 0), ('x', 7, 0)], name='a78'),
          N('AND2', [('x', 8, 0), ('x', 9, 0)], name='u1'), N('OR2', [('x', 10, 0), ('x', 11, 0)], name='u2'),
          N('CONST0', [], 'inert', 'L1inert')]
    L2 = [N('XOR2', [('n', 0, 0), ('n', 1, 0)], name='t1'), N('XOR2', [('n', 2, 0), ('n', 3, 0)], name='t2'),
          N('XOR2', [('n', 4, 0), ('n', 5, 0)], name='v1'), N('AND2', [('n', 4, 0), ('n', 5, 1)], name='v2'),
          N('CONST0', [], 'inert', 'L2inert')]
    L3 = [N('OR2', [('n', 2, 0), ('n', 3, 0)], name='w1'), N('AND2', [('n', 2, 1), ('x', 10, 0)], name='w2')]
    L3 += [N('XOR2', [('n', 0, 0), ('n', 1, 0)], 'copy', f'pc{j}') for j in range(3)]
    L3 += [N('XOR2', [('n', 0, 0), ('n', 1, 0)], 'epi', 'L3epi(p)'), N('CONST0', [], 'inert', 'L3inert')]
    P = ('m', (2, 3, 4), 0)
    L4 = [N('XOR2', [('x', i, 0), P], 'out', f'y{i + 1}') for i in range(7)]
    L4 += [N('XOR2', [('n', 0, 0), ('x', 8, 0)], 'out', 'y8'), N('AND2', [('n', 1, 0), ('x', 11, 0)], 'out', 'y9'),
           N('OR2', [('n', 0, 1), ('x', 9, 0)], 'out', 'y10'), N('XOR3', [('n', 0, 0), ('n', 1, 0), ('x', 11, 0)], 'out', 'y11')]
    prog = dict(levels=[None, L1, L2, L3, L4], nin=NIN, name='H1')
    prog['plants'] = {3: dict(copies=[2, 3, 4], epi=5, inert=6, epi_src=2, trip='p', epi_src_name='p'),
                      1: dict(copies=[], epi=None, inert=6), 2: dict(copies=[], epi=None, inert=4)}
    prog['tree'] = {1: [0, 1, 2, 3], 2: [0, 1], 3: [2, 3, 4]}
    return prog


def h3():
    """r = MAJ3(x1,x2,x3) carried to level 3; y_i = MAJ3(r, z_a, z_b). No plants."""
    L1 = [N('MAJ3', [('x', 0, 0), ('x', 1, 0), ('x', 2, 0)], name='r'),
          N('XOR2', [('x', 3, 0), ('x', 4, 0)], name='a'), N('AND2', [('x', 5, 0), ('x', 6, 1)], name='b'),
          N('OR2', [('x', 7, 0), ('x', 8, 0)], name='c'), N('XOR2', [('x', 9, 0), ('x', 10, 0)], name='d'),
          N('MAJ3', [('x', 9, 0), ('x', 11, 0), ('x', 3, 1)], name='e'), N('CONST0', [], 'inert', 'L1inert')]
    L2 = [N('ID', [('n', 0, 0)], name='r'), N('XOR2', [('n', 1, 0), ('n', 2, 0)], name='f'),
          N('OR2', [('n', 3, 0), ('n', 4, 1)], name='g'), N('AND2', [('n', 5, 0), ('n', 1, 0)], name='h'),
          N('XOR2', [('n', 3, 0), ('n', 5, 0)], name='i'), N('CONST0', [], 'inert', 'L2inert')]
    L3 = [N('ID', [('n', 0, 0)], name='r'), N('XOR2', [('n', 1, 0), ('n', 2, 0)], name='z1'),
          N('AND2', [('n', 2, 0), ('n', 3, 1)], name='z2'), N('OR2', [('n', 3, 0), ('n', 4, 0)], name='z3'),
          N('XOR2', [('n', 4, 0), ('n', 1, 1)], name='z4'), N('MAJ3', [('n', 1, 0), ('n', 2, 0), ('n', 4, 0)], name='z5'),
          N('AND2', [('n', 1, 1), ('n', 4, 0)], name='z6'), N('CONST0', [], 'inert', 'L3inert')]
    pairs = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (1, 6), (2, 5)]
    L4 = [N('MAJ3', [('n', 0, 0), ('n', a, 0), ('n', b, 0)], 'out', f'y{i + 1}') for i, (a, b) in enumerate(pairs)]
    prog = dict(levels=[None, L1, L2, L3, L4], nin=NIN, name='H3')
    prog['plants'] = {l: dict(copies=[], epi=None, inert=len(prog['levels'][l]) - 1) for l in (1, 2, 3)}
    return prog


def h2(n=16):
    """Prefix parity, n = 16 (program only): level 1 pair parities a_i, level 2
    pair-prefix parities S_i, outputs y_2i = S_i, y_2i-1 = S_i-1 xor x_2i-1."""
    L1 = [N('XOR2', [('x', 2 * i, 0), ('x', 2 * i + 1, 0)], name=f'a{i}') for i in range(n // 2)]
    # S_i = a_0 ^ ... ^ a_i = parity(x_1..x_2i+2), read from x directly (x is read at any level)
    L2 = []
    for i in range(n // 2):
        L2.append(dict(type='PAR', srcs=[('x', j, 0) for j in range(2 * i + 2)], role='gate', name=f'S{i}'))
    L3 = []
    for i in range(n // 2):
        if i == 0:
            L3.append(N('ID', [('x', 0, 0)], 'out', 'y1'))
        else:
            L3.append(N('XOR2', [('n', i - 1, 0), ('x', 2 * i, 0)], 'out', f'y{2 * i + 1}'))
        L3.append(N('ID', [('n', i, 0)], 'out', f'y{2 * i + 2}'))
    return dict(levels=[None, L1, L2, L3], nin=n, name='H2')


# ---------------------------------------------------------------- exact tables
def subsets(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=bool)[:, ::-1]  # row s: bit k = (s>>k)&1


def shapley_from_table(vals, n):
    """vals[s] for coalition bitmask s (int). Exact Shapley."""
    w = np.array([math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)])
    sizes = np.array([bin(s).count('1') for s in range(1 << n)])
    phi = np.zeros(n)
    for k in range(n):
        s = np.arange(1 << n)
        without = s[(s >> k) & 1 == 0]
        phi[k] = (w[sizes[without]] * (vals[without | (1 << k)] - vals[without])).sum()
    return phi


def loo_marg(vals, n):
    full = (1 << n) - 1
    loo = np.array([vals[full] - vals[full ^ (1 << k)] for k in range(n)])
    marg = np.array([vals[1 << k] - vals[0] for k in range(n)])
    return loo, marg


def program_tables(prog, l, X, xi, xp, eps=EPS_NOISE):
    """Exact per-level tables on the program: FLIP, noisy-PAY and MI games,
    LOO/MARG/Shapley, m_k, u_k."""
    V = values(prog, X)
    Y = V[-1]
    Yb = Y[:, xi]
    hx, hxp = V[l][:, xi], V[l][:, xp]
    n = hx.shape[0]
    nout = Y.shape[0]
    eps = eps if len(eps) == nout else np.full(nout, eps.mean())
    wgt = (1 - 2 * eps) * np.log((1 - eps) / eps)
    flip = np.zeros(1 << n)
    pay = np.zeros(1 << n)
    mi = np.zeros(1 << n)
    Xb = X[xi]
    for s in range(1 << n):
        m = ((s >> np.arange(n)) & 1).astype(bool)
        h = np.where(m[:, None], hx, hxp)
        out = run_from(prog, l, h, Xb)
        fl = (out != Yb).mean(1)
        flip[s] = 1 - fl.mean()
        pay[s] = -(fl * wgt).sum()
        mi[s] = mi_bits(V[l][m], Y) if m.any() else 0.0
    res = {}
    for g, vals in (('flip', flip), ('pay', pay), ('mi', mi)):
        loo, marg = loo_marg(vals, n)
        res[g] = dict(loo=loo, marg=marg, shap=shapley_from_table(vals, n), vals=vals)
        full = (1 << n) - 1
        res[g]['eff_gap'] = float(res[g]['shap'].sum() - (vals[full] - vals[0]))
    res['m'] = np.array([mi_bits(V[l][k:k + 1], Y) for k in range(n)])
    res['r'] = res['flip']['loo'] / 1.0  # reach = fraction of output bits flipped = flip LOO
    res['u'] = res['pay']['loo']
    res['roles'] = [nd['role'] for nd in prog['levels'][l]]
    res['names'] = [nd['name'] for nd in prog['levels'][l]]
    res['types'] = [nd['type'] for nd in prog['levels'][l]]
    return res
