"""W3 discrete chain world (DESIGN-v1.md, section W3 / crux C2).

Episode: M disjoint chains of n nodes each; latent bits x ~ Bern(1/2) iid. Node ids are a random draw (without
replacement) from a vocabulary of V ids, so a node's id carries no information about its chain or position.
Messages (all flip rates are known and shown on the token as reliability c = 1 - 2p):
  UNARY (a, r, c):        r = x_a xor Bern(p)
  XOR   (a, b, r, c):     r = x_a xor x_b xor Bern(p), for adjacent a, b; token order (a, b) or (b, a) at random
  MAJ   (a, b, d, r, c):  r = maj(x_a, x_b, x_d) xor Bern(p) over three consecutive nodes (optional, off by default)
One query node q. Every chain (query chain and distractors alike) is drawn by the same procedure around a
"virtual query" at path distance l' from the nearest unary report, so the query chain is not marked by structure.

Information matching across l (Codex r3): a chain is "exact" (XOR flip rate 0) with probability p_exact; otherwise
its XOR reliability is c_e = c_path ** (1 / max(l', 1)) with c_path drawn from a fixed set, so the reliability of the
path from the nearest report to the query, c_u * c_e**l, has the same distribution at every l. We additionally
report the Bayes information KL(Bayes || prior) per l, and excess loss relative to it.

The data-generating policy depends only on visible structure (positions, l), never on latent bits, so it is
ignorable: the exact posterior is a function of the messages alone (tested in test_world3_chains.py).

Exact Bayes: forward-backward on the query chain (other chains are independent of x_q). Pair-state
(x_j, x_{j+1}) forward-backward, so majority factors over three consecutive nodes are exact too.
References: prior (1/2); unary-only (only unary reports on q itself); nearest-report-only (the nearest unary
report(s) plus all relations on the chain); exact Bayes.
"""
import numpy as np

UNARY, XOR, MAJ, QUERY = 0, 1, 2, 3
N_TYPES = 4

DEFAULT = dict(M=8, n=12, V=128, rho=0.25, p_exact=1 / 3, c_unary=(0.6, 0.8, 0.9), c_path=(0.7, 0.9),
               maj=False, rho_maj=0.3, c_maj=(0.8,), flip=0.5, signed=False)


def cfg_of(**kw):
    c = dict(DEFAULT)
    for k, v in kw.items():
        if k not in c:
            raise KeyError(k)
        c[k] = v
    return c


# ----------------------------------------------------------------------------------------------- generation
def sample_chains(B, rng, cfg, l_query, l_other, flip=0.5):
    """Latent chains for B episodes. l_query: (B,) distances for chain 0 (the query chain before shuffling ids);
    l_other: callable size -> distances for the distractor chains. Returns a dict of arrays shaped (B, M, ...)."""
    M, n = cfg['M'], cfg['n']
    L = np.empty((B, M), dtype=np.int64)
    L[:, 0] = l_query
    L[:, 1:] = l_other((B, M - 1))
    if L.max() > n - 1:
        raise ValueError('distance exceeds chain length')
    x = rng.integers(0, 2, size=(B, M, n))
    if flip != 0.5:                                  # training-only curriculum: Markov latent, P(x_{i+1} != x_i) = flip
        steps = (rng.random((B, M, n - 1)) < flip).astype(np.int64)
        x = (x[..., :1] + np.concatenate([np.zeros((B, M, 1), np.int64), np.cumsum(steps, -1)], -1)) % 2
    # virtual query position and side of the anchor
    side = np.where(rng.random((B, M)) < 0.5, -1, 1)
    # valid qpos for side=+1: [0, n-1-L]; for side=-1: [L, n-1]
    span = n - L                                     # number of valid positions
    off = (rng.random((B, M)) * span).astype(np.int64)
    qpos = np.where(side > 0, off, L + off)
    anchor = qpos + side * L
    pos = np.arange(n)[None, None, :]
    dist = np.abs(pos - qpos[..., None])
    U = (dist > L[..., None]) & (rng.random((B, M, n)) < cfg['rho'])
    U |= (dist == L[..., None]) & (rng.random((B, M, n)) < cfg['rho'])   # far-side node at the same distance
    U[np.arange(B)[:, None], np.arange(M)[None, :], anchor] = True
    cu = np.asarray(cfg['c_unary'], dtype=np.float64)[rng.integers(0, len(cfg['c_unary']), size=(B, M, n))]
    cu = np.where(U, cu, 0.0)
    flip_u = rng.random((B, M, n)) < (1 - cu) / 2
    ru = (x ^ flip_u) * U
    exact = rng.random((B, M)) < cfg['p_exact']
    cp = np.asarray(cfg['c_path'], dtype=np.float64)[rng.integers(0, len(cfg['c_path']), size=(B, M))]
    ce = np.where(exact, 1.0, cp ** (1.0 / np.maximum(L, 1)))
    ce = np.broadcast_to(ce[..., None], (B, M, n - 1)).copy()
    flip_e = rng.random((B, M, n - 1)) < (1 - ce) / 2
    re = x[..., :-1] ^ x[..., 1:] ^ flip_e
    out = dict(x=x, L=L, qpos=qpos, U=U, ru=ru, cu=cu, re=re, ce=ce, flip=np.full((B, M), float(flip)))
    if cfg['maj']:
        Mj = rng.random((B, M, n - 2)) < cfg['rho_maj']
        cm = np.asarray(cfg['c_maj'], dtype=np.float64)[rng.integers(0, len(cfg['c_maj']), size=(B, M, n - 2))]
        cm = np.where(Mj, cm, 0.0)
        maj = ((x[..., :-2] + x[..., 1:-1] + x[..., 2:]) >= 2).astype(np.int64)
        rm = (maj ^ (rng.random((B, M, n - 2)) < (1 - cm) / 2)) * Mj
        out.update(Mj=Mj, rm=rm, cm=cm)
    return out


def sample(B, rng, cfg, l_query, l_other=(0, 4), flip=None):
    """Draw B episodes. l_query: int or (B,) array of query distances. l_other: (lo, hi) inclusive range for the
    distractor chains' virtual distances (the training range by default, so distractors look the same in train and
    test). Returns dict with chain-level arrays ('chains'), token arrays and the query bit."""
    if flip is None:
        flip = cfg.get('flip', 0.5)
    if np.isscalar(l_query):
        l_query = np.full(B, int(l_query))
    lo, hi = l_other
    ch = sample_chains(B, rng, cfg, np.asarray(l_query), lambda s: rng.integers(lo, hi + 1, size=s), flip)
    tok = tokenize(ch, rng, cfg)
    y = ch['x'][np.arange(B), 0, ch['qpos'][:, 0]]
    return dict(chains=ch, y=y, l=np.asarray(l_query), **tok)


def tokenize(ch, rng, cfg):
    """Messages as tokens: typ (B,T), ids (B,T,3) (pad id = V), bit (B,T), rel (B,T), valid (B,T).
    Message order is shuffled; the query token is the last position. Chain 0 holds the query; its identity is
    hidden by the random node ids and the shuffle."""
    B, M, n = ch['x'].shape
    V = cfg['V']
    if M * n > V:
        raise ValueError('vocabulary too small')
    ids = np.argsort(rng.random((B, V)), axis=1)[:, :M * n].reshape(B, M, n)
    rows = []                                          # each: typ, a, b, d, bit, rel, valid  -> (B, M, k)
    k = n
    rows.append((np.full((B, M, k), UNARY), ids, np.full((B, M, k), V), np.full((B, M, k), V),
                 ch['ru'], ch['cu'], ch['U']))
    k = n - 1
    a, b = ids[..., :-1], ids[..., 1:]
    sw = rng.random((B, M, k)) < 0.5
    rows.append((np.full((B, M, k), XOR), np.where(sw, b, a), np.where(sw, a, b), np.full((B, M, k), V),
                 ch['re'].astype(np.int64), ch['ce'], np.ones((B, M, k), bool)))
    if 'Mj' in ch:
        k = n - 2
        tri = np.stack([ids[..., :-2], ids[..., 1:-1], ids[..., 2:]], -1)
        perm = np.argsort(rng.random((B, M, k, 3)), -1)
        tri = np.take_along_axis(tri, perm, -1)
        rows.append((np.full((B, M, k), MAJ), tri[..., 0], tri[..., 1], tri[..., 2], ch['rm'], ch['cm'], ch['Mj']))
    cat = lambda i: np.concatenate([r[i].reshape(B, -1) for r in rows], 1)
    typ, A, Bb, D, bit, rel, valid = (cat(i) for i in range(7))
    key = np.where(valid, rng.random(valid.shape), 2.0)
    order = np.argsort(key, 1)
    nmax = int(valid.sum(1).max())
    order = order[:, :nmax]
    g = lambda arr: np.take_along_axis(arr, order, 1)
    typ, A, Bb, D, bit, rel, valid = map(g, (typ, A, Bb, D, bit, rel, valid))
    A = np.where(valid, A, V); Bb = np.where(valid, Bb, V); D = np.where(valid, D, V)
    q = ids[np.arange(B), 0, ch['qpos'][:, 0]]
    T = nmax + 1
    out_typ = np.concatenate([typ, np.full((B, 1), QUERY)], 1)
    out_ids = np.concatenate([np.stack([A, Bb, D], -1), np.stack([q, np.full(B, V), np.full(B, V)], -1)[:, None]], 1)
    out_bit = np.concatenate([np.where(valid, bit, 0), np.full((B, 1), 2)], 1)
    out_rel = np.concatenate([np.where(valid, rel, 0.0), np.zeros((B, 1))], 1)
    out_valid = np.concatenate([valid, np.ones((B, 1), bool)], 1)
    return dict(typ=out_typ.astype(np.int64), ids=out_ids.astype(np.int64), bit=out_bit.astype(np.int64),
                rel=out_rel.astype(np.float32), valid=out_valid, node_ids=ids)


def features(ep, cfg):
    """Float token features (B, T, F) and valid mask:
    multi-hot over node ids (a message's nodes; orientation-free), message type one-hot (4), bit one-hot (0/1/none),
    reliability c = 1 - 2p. F = V + 8. A linear embedding of the multi-hot is the sum of learned id embeddings."""
    V = cfg['V']
    B, T = ep['typ'].shape
    F = N_FEAT(cfg)
    X = np.zeros((B * T, F), np.float32)
    r = np.arange(B * T)
    ids = ep['ids'].reshape(B * T, 3)
    for s in range(3):
        m = ids[:, s] < V
        X[r[m], ids[m, s]] = 1.0
    X[r, V + ep['typ'].ravel()] = 1.0
    X[r, V + N_TYPES + ep['bit'].ravel()] = 1.0        # bit 0 / 1 / 2 (= none, query)
    X[:, V + N_TYPES + 3] = ep['rel'].ravel()
    if cfg.get('signed'):                              # signed evidence (2r - 1) c; 0 on the query
        X[:, V + N_TYPES + 4] = np.where(ep['bit'].ravel() < 2, 2.0 * ep['bit'].ravel() - 1, 0.0) * ep['rel'].ravel()
    X[~ep['valid'].ravel()] = 0.0
    X = X.reshape(B, T, F)
    return X, ep['valid']


N_FEAT = lambda cfg: cfg['V'] + N_TYPES + 4 + int(bool(cfg.get('signed')))


# ----------------------------------------------------------------------------------------------- exact Bayes
def _pair_tables(n):
    # pair state s = 2*x_j + x_{j+1}
    s = np.arange(4)
    return s >> 1, s & 1


def chain_marginals(U, ru, cu, re, ce, Mj=None, rm=None, cm=None, flip=None):
    """Exact posterior P(x_j = 1 | messages) for every node of a batch of chains (any leading shape, chain axis
    last). Pair-state forward-backward; float64; exact also with reliability 1 (deterministic factors)."""
    lead = U.shape[:-1]
    n = U.shape[-1]
    U = U.reshape(-1, n); ru = ru.reshape(-1, n); cu = cu.reshape(-1, n)
    re = re.reshape(-1, n - 1); ce = ce.reshape(-1, n - 1)
    N = U.shape[0]
    fl = None if flip is None else np.broadcast_to(np.asarray(flip, float).reshape(-1), (N,))[:, None]
    # unary likelihood per node: lik[:, j, v]
    cuu = np.where(U, cu, 0.0)
    lik = np.empty((N, n, 2))
    for v in (0, 1):
        lik[:, :, v] = np.where(ru == v, (1 + cuu) / 2, (1 - cuu) / 2)
    lik = lik * 2                                     # no report -> factor 1
    hi, lo = _pair_tables(n)
    # local factor of pair state j: psi_j(x_j, x_{j+1}) * unary(x_j) (and unary(x_{n-1}) for the last pair)
    par = (hi ^ lo)[None, :]                          # (1, 4)
    F = np.empty((N, n - 1, 4))
    for j in range(n - 1):
        e = np.where(par == re[:, j:j + 1], (1 + ce[:, j:j + 1]) / 2, (1 - ce[:, j:j + 1]) / 2)
        if fl is not None:                            # Markov latent prior P(x_{j+1} != x_j) = flip
            e = e * np.where(par == 1, fl, 1 - fl) * 2
        f = e * lik[:, j, :][:, hi]
        if j == n - 2:
            f = f * lik[:, n - 1, :][:, lo]
        F[:, j] = f
    # transition between pair state j=(a,b) and j+1=(b',c): requires b == b'; majority factor over (a,b,c)
    compat = (lo[:, None] == hi[None, :]).astype(np.float64)       # (4, 4)
    Tm = np.broadcast_to(compat, (N, n - 2, 4, 4)).copy()
    if Mj is not None:
        Mj = Mj.reshape(-1, n - 2); rm = rm.reshape(-1, n - 2); cm = cm.reshape(-1, n - 2)
        a = hi[:, None]; b = lo[:, None]; c = lo[None, :]
        mj = ((a + b + c) >= 2).astype(np.int64)                    # (4, 4) majority of (a, b, c)
        cmm = np.where(Mj, cm, 0.0)[..., None, None]
        fac = np.where(mj[None, None] == rm[..., None, None], (1 + cmm) / 2, (1 - cmm) / 2) * 2
        Tm = Tm * fac
    alpha = np.empty((N, n - 1, 4)); beta = np.empty((N, n - 1, 4))
    a0 = 0.25 * F[:, 0]
    alpha[:, 0] = a0 / a0.sum(1, keepdims=True)
    for j in range(1, n - 1):
        a_ = np.einsum('ns,nst->nt', alpha[:, j - 1], Tm[:, j - 1]) * F[:, j]
        alpha[:, j] = a_ / a_.sum(1, keepdims=True)
    beta[:, n - 2] = 1.0
    for j in range(n - 3, -1, -1):
        b_ = np.einsum('nst,nt->ns', Tm[:, j], F[:, j + 1] * beta[:, j + 1])
        beta[:, j] = b_ / b_.sum(1, keepdims=True)
    post = alpha * beta
    post = post / post.sum(-1, keepdims=True)         # (N, n-1, 4)
    p1 = np.empty((N, n))
    p1[:, :n - 1] = post[:, :, 2] + post[:, :, 3]     # x_j = 1  (hi bit)
    p1[:, n - 1] = post[:, n - 2, 1] + post[:, n - 2, 3]
    return p1.reshape(*lead, n)


def _query_chain(ch, i0=0):
    keys = ['U', 'ru', 'cu', 're', 'ce'] + (['Mj', 'rm', 'cm'] if 'Mj' in ch else [])
    d = {k: ch[k][:, i0] for k in keys}
    if 'flip' in ch and not np.all(ch['flip'][:, i0] == 0.5):
        d['flip'] = ch['flip'][:, i0]
    return d


def references(ep):
    """P(x_q = 1) under each reference, (B,) arrays: prior, unary_only, nearest_only, bayes."""
    ch = ep['chains']
    B = ch['x'].shape[0]
    qc = _query_chain(ch)
    qpos = ch['qpos'][:, 0]
    r = np.arange(B)
    bayes = chain_marginals(**qc)[r, qpos]
    # unary-only: only reports on q itself
    n = ch['x'].shape[-1]
    onq = np.zeros_like(qc['U']); onq[r, qpos] = True
    Uq = qc['U'] & onq
    cu, ru = qc['cu'][r, qpos], qc['ru'][r, qpos]
    unary = np.where(Uq[r, qpos], np.where(ru == 1, (1 + cu) / 2, (1 - cu) / 2), 0.5)
    # nearest-report-only: drop every unary report farther than the nearest
    dist = np.abs(np.arange(n)[None] - qpos[:, None])
    dmin = np.where(qc['U'], dist, 10 ** 6).min(1)
    qn = dict(qc); qn['U'] = qc['U'] & (dist == dmin[:, None])
    nearest = chain_marginals(**qn)[r, qpos]
    return dict(prior=np.full(B, 0.5), unary_only=unary, nearest_only=nearest, bayes=bayes)


def bern_kl(p, q, eps=1e-12):
    """KL(Bern(p) || Bern(q)) in nats, elementwise."""
    q = np.clip(q, eps, 1 - eps)
    t1 = np.where(p > 0, p * (np.log(np.clip(p, eps, 1)) - np.log(q)), 0.0)
    t0 = np.where(p < 1, (1 - p) * (np.log(np.clip(1 - p, eps, 1)) - np.log(1 - q)), 0.0)
    return t1 + t0


def brute_marginals(U, ru, cu, re, ce, Mj=None, rm=None, cm=None, flip=None):
    """Enumeration over all 2^n assignments of one chain (tests only)."""
    n = U.shape[-1]
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None]) & 1)
    w = np.ones(2 ** n)
    for j in range(n):
        if U[j]:
            w *= np.where(X[:, j] == ru[j], (1 + cu[j]) / 2, (1 - cu[j]) / 2)
    for j in range(n - 1):
        w *= np.where((X[:, j] ^ X[:, j + 1]) == re[j], (1 + ce[j]) / 2, (1 - ce[j]) / 2)
        if flip is not None:
            w *= np.where(X[:, j] != X[:, j + 1], flip, 1 - flip)
    if Mj is not None:
        for j in range(n - 2):
            if Mj[j]:
                mj = (X[:, j] + X[:, j + 1] + X[:, j + 2]) >= 2
                w *= np.where(mj == rm[j], (1 + cm[j]) / 2, (1 - cm[j]) / 2)
    return (w[:, None] * X).sum(0) / w.sum()


def posterior_from_tokens(ep, b, cfg):
    """Exact P(x_q = 1) recomputed from episode b's TOKENS only (message list), by enumeration over the
    connected component of q. Used to test that the oracle is a function of the visible messages."""
    V = cfg['V']
    typ, ids, bit, rel, valid = ep['typ'][b], ep['ids'][b], ep['bit'][b], ep['rel'][b], ep['valid'][b]
    q = ids[-1, 0]
    adj = {}
    msgs = [(typ[t], [i for i in ids[t] if i != V], bit[t], rel[t]) for t in range(len(typ) - 1) if valid[t]]
    for ty, nodes, _, _ in msgs:
        for u in nodes:
            adj.setdefault(u, set()).update(nodes)
    comp, stack = {q}, [q]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in comp:
                comp.add(v); stack.append(v)
    comp = sorted(comp)
    idx = {u: i for i, u in enumerate(comp)}
    k = len(comp)
    X = ((np.arange(2 ** k)[:, None] >> np.arange(k)[None]) & 1)
    w = np.ones(2 ** k)
    for ty, nodes, r, c in msgs:
        if nodes[0] not in idx:
            continue
        cols = [X[:, idx[u]] for u in nodes]
        if ty == UNARY:
            v = cols[0]
        elif ty == XOR:
            v = cols[0] ^ cols[1]
        else:
            v = ((cols[0] + cols[1] + cols[2]) >= 2).astype(np.int64)
        w *= np.where(v == r, (1 + c) / 2, (1 - c) / 2)
    return float((w * X[:, idx[q]]).sum() / w.sum())


# ----------------------------------------------------------------------------------------------- harness adapter
class ChainWorld:
    """Harness world (see harness.py WORLD INTERFACE). The harness's size axis N is used as the query path
    distance l (N = l); train with --nmin 0 --nmax 4, eval with --Ns 0,1,...,8. Sequence length does not grow
    with l: every episode has M chains of n nodes, so test episodes differ from training ones only in l.
    Policies: 'std' (distractor chains' virtual distances in the training range [0, 4]); 'far' (distractor
    virtual distances also drawn up to l_max, so the query chain cannot be spotted by its sparse reports)."""
    target_kind = 'cat'
    n_classes = 2
    train_policies = ['std']
    heldout = ['far']
    test_policies = ['std', 'far']

    def __init__(self, cfg=None):
        c = dict(DEFAULT, l_train=(0, 4), l_max=8, curriculum=0, anneal=0, flip0=0.05)
        c.update(cfg or {})
        c['l_train'] = list(c['l_train'])
        for k in ('c_unary', 'c_path', 'c_maj'):
            c[k] = list(c[k])
        self.cfg = c
        self.name = 'chains'
        self.wcfg = {k: c[k] for k in DEFAULT}
        self._calls = 0
        self.feat_dim = N_FEAT(self.wcfg)
        self.splits = {'all': lambda m: np.ones(len(m['l']), bool),
                       'exact_chain': lambda m: m['exact'].astype(bool),
                       'noisy_chain': lambda m: ~m['exact'].astype(bool)}
        self.abilities = [('path_tracing', 'prior', 'nearest_only', 'all'),
                          ('beyond_nearest', 'nearest_only', 'bayes', 'all')]

    def _finish(self, ep):
        X, mask = features(ep, self.wcfg)
        ep['X'] = X; ep['mask'] = mask; ep['target'] = ep['y'].astype(np.int64)
        return ep

    def sample(self, rng, E, N, policy='std', n=None):
        lo, hi = self.cfg['l_train']
        other = (lo, hi) if policy == 'std' else (0, self.cfg['l_max'])
        return self._finish(sample(E, rng, self.wcfg, int(N), other, self.cfg['flip']))

    def sample_train(self, rng, E, nmin, nmax):
        """Training batch, l ~ U{nmin..nmax}. With cfg curriculum = S > 0 the range grows during training: the
        k-th call after the first uses l ~ U{nmin..min(nmax, nmin + k // S)} (the harness's first call draws the
        validation set, which keeps the full range). Distractor chains always use the full training range.
        With cfg anneal = A > 0, training latents are a Markov chain with P(x_{i+1} != x_i) rising linearly from
        flip0 to 1/2 over the first A steps (then iid, the target world). XOR messages then carry a first-order
        signal early on (breaks the parity plateau). Validation and evaluation always use the target world."""
        hi = nmax
        S = self.cfg['curriculum']
        k = self._calls - 1                              # training step index (call 0 = validation set)
        if S and k >= 0:
            hi = min(nmax, nmin + k // S)
        flip = self.cfg['flip']
        A = self.cfg['anneal']
        if A and k >= 0:                                 # correlated-latent curriculum, flip0 -> target flip over A steps
            flip = self.cfg['flip0'] + (self.cfg['flip'] - self.cfg['flip0']) * min(1.0, k / A)
        self._calls += 1
        return self._finish(sample(E, rng, self.wcfg, rng.integers(nmin, hi + 1, size=E), tuple(self.cfg['l_train']),
                                   flip))

    @staticmethod
    def _cat(p):
        return np.stack([1 - p, p], 1)

    def references(self, ep):
        return {k: self._cat(v) for k, v in references(ep).items()}

    def exact_bayes(self, ep):
        return self.references(ep)['bayes']

    def meta(self, ep):
        ch = ep['chains']
        p = references(ep)['bayes']
        return dict(l=ep['l'], exact=(ch['ce'][:, 0, 0] == 1.0), n_reports=ch['U'][:, 0].sum(1),
                    info=bern_kl(p, np.full_like(p, 0.5)))


def make_world(cfg=None):
    return ChainWorld(cfg)


# ----------------------------------------------------------------------------------------------- dense queries
def dense_queries(ep, rng, cfg, K, dmax=4):
    """K extra query tokens per episode (training-only dense supervision, w3_dense.py): nodes on any chain whose
    nearest unary report is at distance <= dmax (the training range), sampled with equal weight per distance.
    Returns features (B, K, F) shaped like the query token and the nodes' true bits (B, K)."""
    ch = ep['chains']
    B, M, n = ch['x'].shape
    V = cfg['V']
    D = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    dmin = np.where(ch['U'][..., None, :], D[None, None], 10 ** 6).min(-1).reshape(B, M * n)
    ok = dmin <= dmax
    w = np.zeros((B, M * n))
    for d in range(dmax + 1):
        m = ok & (dmin == d)
        cnt = m.sum(1, keepdims=True)
        w += np.where(m, 1.0 / np.maximum(cnt, 1), 0.0)
    cum = np.cumsum(w, 1)
    u = rng.random((B, K)) * cum[:, -1:]
    idx = np.minimum((cum[:, None, :] <= u[..., None]).sum(-1), M * n - 1)       # (B, K)
    nid = np.take_along_axis(ep['node_ids'].reshape(B, M * n), idx, 1)
    y = np.take_along_axis(ch['x'].reshape(B, M * n), idx, 1)
    Xq = np.zeros((B, K, N_FEAT(cfg)), np.float32)
    bi, ki = np.meshgrid(np.arange(B), np.arange(K), indexing='ij')
    Xq[bi, ki, nid] = 1.0
    Xq[..., V + QUERY] = 1.0
    Xq[..., V + N_TYPES + 2] = 1.0
    return Xq, y.astype(np.int64), np.take_along_axis(dmin, idx, 1)
