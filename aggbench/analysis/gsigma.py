"""Listed-securities gap G(Sigma) and the log-likelihood span view, computed exactly (2026-10-22).

Setting. A finite world state w with prior pi(w); target Y = h(w); agent i sees a symbol o_i drawn from lik_i[w, o]
(agents conditionally independent given w; a deterministic reading is a 0/1 column). A security Z is a function of
w with finitely many outcomes; a listed set Sigma is a list of securities (a tuple of names inside Sigma is one
combinatorial security on their joint outcome).

1. Gap. F_Sigma = every agent's own posterior P(Z = z | o_i) for each Z in Sigma (a calibrated forecast from its
   own evidence only). G(Sigma) = I(Y; O_1..O_N | F_Sigma) = H(Y | F_Sigma) - H(Y | O): the Y-information that a
   decoder knowing the whole joint cannot recover from the listed forecasts. Adding a security never raises G.
   `g_sequential` is the one-pass market variant: agent t forecasts Sigma from its evidence AND the forecasts
   already posted.
2. Span. phi(Sigma) = outcome indicators of every security in Sigma (one reference outcome dropped). Agent i's
   trade theta_i(o) moves the market from pi to its own forecasts on Sigma: q_theta(w) ~ pi(w) exp(theta . phi(w))
   with E_q[phi] = E_{p(w|o)}[phi] (moment matching = the LMSR trade = the KL projection of its posterior onto the
   family). Residual D_i(o) = KL(p(w | o) || q_theta_i): zero iff log lik_i(., o) lies in span(1, phi), i.e. the
   log-likelihood-span condition. Aggregating by summing trades gives q(w | o) ~ pi exp(sum_i theta_i . phi).
   Its regret is E KL(p(Y | O) || q(Y | O)) = G(Sigma) + pool(Sigma) (chain rule; the trade is a function of the
   forecast), and every residual 0 implies regret 0 (then q is the exact posterior).
3. Cost. Bits per agent to send its trade (a) when the decoder knows which securities the agent trades on and
   its accuracy (H of the trade given the structure), and (b) when it does not (support label + fixed-point
   value; computed by the driver from the per-agent supports and trade values returned here).

Everything is exact enumeration over forecast-class tuples (or over w when every reading is deterministic); above
EXACT_LIMIT tuples it falls back to Monte Carlo over observation draws with common random numbers (reported with a
standard error). Units: bits, except trades and L2 residuals, which are in nats.
"""
from dataclasses import dataclass, field
import itertools
import math

import numpy as np

LN2 = math.log(2.0)
EXACT_LIMIT = 2 ** 17
CHUNK_CELLS = 2 ** 21
CAP = 30.0          # |theta| cap in nats: a deterministic reading needs an infinite trade
ROUND = 9


# ============================================================================= cells

@dataclass
class Cell:
    name: str
    prior: np.ndarray                 # (W,)
    target: np.ndarray                # (W,) int
    lik: list                         # per agent (W, A_i), each row sums to 1
    sec: dict                         # security name -> (W,) int outcome
    latent: tuple = ()                # base variables (graph nodes for tree width)
    defs: dict = field(default_factory=dict)   # name -> [scope option, ...] (tuples of names)
    prior_cliques: tuple = ()         # scopes of the prior's factors
    reads: list = None                # per agent: names of the variables it reads (labels only)
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.lik)

    @property
    def W(self):
        return len(self.prior)

    def values(self, z):
        if isinstance(z, tuple):
            codes = np.zeros(self.W, np.int64)
            for name in z:
                v = self.sec[name]
                codes = codes * (int(v.max()) + 1) + v
            return np.unique(codes, return_inverse=True)[1]
        return np.unique(self.sec[z], return_inverse=True)[1]

    def deterministic(self):
        return all(np.all((L == 0) | (L == 1)) for L in self.lik)


def _onehot(v, n=None):
    n = int(v.max()) + 1 if n is None else n
    out = np.zeros((len(v), n))
    out[np.arange(len(v)), v] = 1.0
    return out


def forecast_matrix(cell, sigma):
    """(W, d) full outcome indicators of every security in sigma (d = total outcomes)."""
    if not sigma:
        return np.zeros((cell.W, 0))
    return np.concatenate([_onehot(cell.values(z)) for z in sigma], 1)


def basis(cell, sigma):
    """phi (W, d'): outcome indicators with the first outcome of each security dropped (span(1, phi) is the span
    of all functions additive over the listed securities)."""
    if not sigma:
        return np.zeros((cell.W, 0))
    return np.concatenate([_onehot(cell.values(z))[:, 1:] for z in sigma], 1)


def agent_posteriors(cell, i):
    """(A_i, W) p(w | o) and (A_i,) p(o)."""
    joint = cell.prior[:, None] * cell.lik[i]
    po = joint.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        post = np.where(po > 0, joint / po, 0.0).T
    return post, po


def forecast_classes(cell, sigma):
    """Per agent: (class id per symbol, forecast vector per class (C, d), p(class)). Symbols of zero probability
    go to class 0 (they never occur)."""
    F = forecast_matrix(cell, sigma)
    out = []
    for i in range(cell.N):
        post, po = agent_posteriors(cell, i)
        f = np.round(post @ F, ROUND) if F.shape[1] else np.zeros((len(po), 0))
        live = po > 0
        uniq, inv = np.unique(f[live], axis=0, return_inverse=True)
        cls = np.zeros(len(po), np.int64)
        cls[live] = inv.ravel()
        pc = np.bincount(cls[live], po[live], minlength=len(uniq))
        out.append((cls, uniq, pc))
    return out


def identity_classes(cell):
    out = []
    for i in range(cell.N):
        po = (cell.prior[:, None] * cell.lik[i]).sum(0)
        out.append((np.arange(len(po)), None, po))
    return out


def merged_channels(cell, classes):
    return [cell.lik[i] @ _onehot(c[0], len(c[2])) for i, c in enumerate(classes)]


# ============================================================================= tuple iteration

def _lse_rows(a):
    m = np.max(a, 1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(a - m).sum(1, keepdims=True)))[:, 0]


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(x)


def sample_obs(cell, n, rng):
    """n draws of (w, o_1..o_N) from the model; returns (w (n,), X (n, N))."""
    w = rng.choice(cell.W, n, p=cell.prior)
    X = np.empty((n, cell.N), np.int64)
    for i, L in enumerate(cell.lik):
        cdf = np.cumsum(L[w], 1)
        X[:, i] = np.minimum((rng.random((n, 1)) > cdf).sum(1), L.shape[1] - 1)
    return w, X


def tuples(cell, classes, X_mc=None):
    """Yield (codes (t, N), log joint (t, W) = log pi + sum_i log channel_i, weight (t,)): expectations are
    sum(weight * f). Exact: weight = p(codes). Monte Carlo (X_mc given): codes are the sampled observations mapped
    to classes, weight 1/n."""
    chans = merged_channels(cell, classes)
    logch = [_log(C) for C in chans]
    logpi = _log(cell.prior)
    sizes = [C.shape[1] for C in chans]
    total = int(np.prod(sizes, dtype=float))
    if X_mc is not None:
        codes_all = np.stack([classes[i][0][X_mc[:, i]] for i in range(cell.N)], 1)
        step = max(1, CHUNK_CELLS // cell.W)
        for s in range(0, len(codes_all), step):
            codes = codes_all[s:s + step]
            lj = logpi[None] + sum(logch[i][:, codes[:, i]].T for i in range(cell.N))
            yield codes, lj, np.full(len(codes), 1.0 / len(codes_all))
        return
    if cell.deterministic():
        sym = np.stack([np.argmax(L, 1) for L in cell.lik], 1)                     # (W, N)
        cw = np.stack([classes[i][0][sym[:, i]] for i in range(cell.N)], 1)
        uniq, inv = np.unique(cw, axis=0, return_inverse=True)
        inv = inv.ravel()
        pc = np.bincount(inv, cell.prior, minlength=len(uniq))
        step = max(1, CHUNK_CELLS // cell.W)
        for s in range(0, len(uniq), step):
            codes = uniq[s:s + step]
            lj = np.full((len(codes), cell.W), -np.inf)
            rows = inv - s
            ok = (rows >= 0) & (rows < len(codes))
            lj[rows[ok], np.flatnonzero(ok)] = logpi[ok]
            yield codes, lj, pc[s:s + step]
        return
    if total > EXACT_LIMIT:
        raise OverflowError(f"{total} tuples > EXACT_LIMIT; pass X_mc")
    step = max(1, CHUNK_CELLS // cell.W)
    for s in range(0, total, step):
        idx = np.arange(s, min(total, s + step))
        codes = np.stack(np.unravel_index(idx, sizes), 1)
        lj = logpi[None] + sum(logch[i][:, codes[:, i]].T for i in range(cell.N))
        lp = _lse_rows(lj)
        keep = np.isfinite(lp)
        yield codes[keep], lj[keep], np.exp(lp[keep])


def _py_given(lj, ymat):
    """p(Y | codes) (t, nY) from log joint (t, W)."""
    m = np.max(lj, 1, keepdims=True)
    p = np.exp(lj - m) @ ymat
    return p / p.sum(1, keepdims=True)


def _H(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), 1)


def _kl(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log2(p) - np.log2(np.maximum(q, 1e-300))), 0.0)
    return t.sum(1)


def _accumulate(vals, w, mc):
    mean = float((vals * w).sum())
    if not mc:
        return mean, 0.0
    return mean, float(vals.std() / math.sqrt(len(vals))) if len(vals) > 1 else 0.0


def cond_entropy_Y(cell, classes, X_mc=None):
    """H(Y | class tuple) in bits, and its MC standard error (0 when exact)."""
    ymat = _onehot(cell.target)
    vals, ws = [], []
    for codes, lj, w in tuples(cell, classes, X_mc):
        vals.append(_H(_py_given(lj, ymat)))
        ws.append(w)
    return _accumulate(np.concatenate(vals), np.concatenate(ws), X_mc is not None)


def _needs_mc(cell, classes):
    if cell.deterministic():
        return False
    return int(np.prod([len(c[2]) for c in classes], dtype=float)) > EXACT_LIMIT


# ============================================================================= trades (moment matching)

def mm_trade(prior, phi, m, iters=200, tol=1e-12):
    """theta minimizing KL(p || q_theta), q_theta ~ prior exp(phi theta), given m = E_p[phi]: damped Newton on
    log Z(theta) - theta . m, with |theta| <= CAP."""
    d = phi.shape[1]
    th = np.zeros(d)
    if d == 0:
        return th
    logpi = _log(prior)

    def f(t):
        s = logpi + phi @ t
        return float(_lse_rows(s[None])[0] - t @ m)

    f0 = f(th)
    for _ in range(iters):
        s = logpi + phi @ th
        q = np.exp(s - _lse_rows(s[None])[0])
        mu = q @ phi
        g = mu - m
        if np.max(np.abs(g)) < tol:
            break
        Hm = (phi * q[:, None]).T @ phi - np.outer(mu, mu) + 1e-10 * np.eye(d)
        try:
            step = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            step = g
        t = 1.0
        while t > 1e-12:
            cand = np.clip(th - t * step, -CAP, CAP)
            f1 = f(cand)
            if f1 < f0:
                break
            t *= 0.5
        if t <= 1e-12:
            break
        th, f0 = cand, f1
    return th


def kl_nats(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(p > 0, p * (np.log(p) - np.log(np.maximum(q, 1e-300))), 0.0).sum())


def agent_trades(cell, sigma, classes=None):
    """Per agent: trades (C_i, d') per forecast class, KL residual per symbol (nats), p(symbol), L2 residual per
    symbol (nats; NaN where the log-likelihood is infinite)."""
    phi = basis(cell, sigma)
    classes = classes or forecast_classes(cell, sigma)
    logpi = _log(cell.prior)
    out = []
    design = np.concatenate([np.ones((cell.W, 1)), phi], 1)
    sw = np.sqrt(cell.prior)
    for i in range(cell.N):
        cls, fvec, pc = classes[i]
        post, po = agent_posteriors(cell, i)
        th = np.zeros((len(pc), phi.shape[1]))
        for c in range(len(pc)):
            sym = np.flatnonzero((cls == c) & (po > 0))
            if not len(sym):
                continue
            th[c] = mm_trade(cell.prior, phi, post[sym[0]] @ phi)
        res = np.zeros(len(po))
        l2 = np.full(len(po), np.nan)
        for o in np.flatnonzero(po > 0):
            s = logpi + phi @ th[cls[o]]
            q = np.exp(s - _lse_rows(s[None])[0])
            res[o] = kl_nats(post[o], q)
            ll = _log(cell.lik[i][:, o])
            if np.all(np.isfinite(ll)):
                coef = np.linalg.lstsq(design * sw[:, None], ll * sw, rcond=None)[0]
                l2[o] = float(np.sqrt((cell.prior * (ll - design @ coef) ** 2).sum()))
        out.append({"theta": th, "cls": cls, "pc": pc, "res": res, "po": po, "l2": l2})
    return out


def pool_loss(cell, sigma, classes, trades, X_mc=None, quantize=None):
    """E_c KL(p(Y | c) || q(Y | c)) in bits, q from the summed trades (optionally rounded to a grid)."""
    phi = basis(cell, sigma)
    ymat = _onehot(cell.target)
    logpi = _log(cell.prior)
    V = []
    for t in trades:
        th = t["theta"] if quantize is None else np.round(t["theta"] / quantize) * quantize
        V.append(th @ phi.T)                                            # (C_i, W)
    vals, ws = [], []
    for codes, lj, w in tuples(cell, classes, X_mc):
        p = _py_given(lj, ymat)
        s = logpi[None] + sum(V[i][codes[:, i]] for i in range(cell.N))
        q = _py_given(s, ymat)
        vals.append(_kl(p, q))
        ws.append(w)
    return _accumulate(np.concatenate(vals), np.concatenate(ws), X_mc is not None)


# ============================================================================= main evaluation

class Evaluator:
    """Caches H(Y | O) and the Monte Carlo draw of one cell."""

    def __init__(self, cell, n_mc=20000, seed=0):
        self.cell = cell
        self.rng = np.random.default_rng(seed)
        ident = identity_classes(cell)
        self.mc = _needs_mc(cell, ident)
        self.X = sample_obs(cell, n_mc, self.rng)[1] if self.mc else None
        self.hyx, self.hyx_se = cond_entropy_Y(cell, ident, self.X)
        ymat = _onehot(cell.target)
        py = cell.prior @ ymat
        self.hy = float(_H(py[None])[0])

    def G(self, sigma):
        cls = forecast_classes(self.cell, sigma)
        h, se = cond_entropy_Y(self.cell, cls, self.X if self.mc else None)
        return h - self.hyx, se

    def full(self, sigma, quantize=()):
        cell = self.cell
        cls = forecast_classes(cell, sigma)
        X = self.X if self.mc else None
        h, se = cond_entropy_Y(cell, cls, X)
        tr = agent_trades(cell, sigma, cls)
        pool, pse = pool_loss(cell, sigma, cls, tr, X)
        res = sum(float((t["res"] * t["po"]).sum()) for t in tr) / LN2       # bits, summed over agents
        resmax = max(float(t["res"].max()) for t in tr) / LN2
        l2 = [t["l2"] for t in tr]
        l2sum = (float(sum(np.sqrt((t["po"] * np.nan_to_num(t["l2"]) ** 2).sum()) for t in tr))
                 if all(np.all(np.isfinite(x[t["po"] > 0])) for x, t in zip(l2, tr)) else None)
        H_cls = [float(_H((c[2] / c[2].sum())[None])[0]) for c in cls]
        names = [z if isinstance(z, str) else "+".join(z) for z in sigma]
        supports = []
        for t in tr:
            nz = np.abs(t["theta"]).max(0) > 1e-6 if t["theta"].size else np.zeros(0, bool)
            supports.append(tuple(sorted(set(_coord_owner(cell, sigma)[j] for j in np.flatnonzero(nz)))))
        out = {"sigma": names, "G": h - self.hyx, "G_se": se, "pool": pool, "pool_se": pse,
               "regret_sum": h - self.hyx + pool, "residual_kl_bits": res, "residual_kl_max_bits": resmax,
               "residual_l2_nats": l2sum, "H_trade_bits": H_cls, "n_trades": [int((c[2] > 0).sum()) for c in cls],
               "supports": supports,
               "theta_nonzero": [np.abs(t["theta"][np.abs(t["theta"]) > 1e-6]).tolist() for t in tr],
               "mc": self.mc}
        for q in quantize:
            out[f"pool_q{q}"] = pool_loss(cell, sigma, cls, tr, X, quantize=q)[0]
        return out


def _coord_owner(cell, sigma):
    names = []
    for z in sigma:
        n = len(np.unique(cell.values(z))) - 1
        names += [z if isinstance(z, str) else "+".join(z)] * n
    return names


# ============================================================================= sequential (one pass)

def g_sequential(cell, sigma, order, max_cells=2 ** 23):
    """G after one pass: agent order[t] posts its forecast on sigma given its evidence and all posted forecasts.
    Exact enumeration of observation tuples (or of w for deterministic readings)."""
    F = forecast_matrix(cell, sigma)
    ymat = _onehot(cell.target)
    if cell.deterministic():
        X = np.stack([np.argmax(L, 1) for L in cell.lik], 1)
        rows_w = np.arange(cell.W)
        P = None
        wts = cell.prior
    else:
        sizes = [L.shape[1] for L in cell.lik]
        T = int(np.prod(sizes))
        if T * cell.W > max_cells:
            raise OverflowError("too many tuples for the sequential enumeration")
        X = np.stack(np.unravel_index(np.arange(T), sizes), 1)
        P = cell.prior[None] * np.prod([cell.lik[i][:, X[:, i]].T for i in range(cell.N)], 0)   # (T, W)

    def group(key):
        u, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        M = np.zeros((len(u), cell.W))
        if P is None:
            np.add.at(M, (inv, rows_w), wts)
        else:
            o = np.argsort(inv, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(inv[o])) + 1]
            M = np.add.reduceat(P[o], starts, axis=0)
        return inv, M

    state = np.zeros(len(X), np.int64)
    for a in order:
        A = cell.lik[a].shape[1]
        inv, M = group(state * A + X[:, a])
        tot = M.sum(1, keepdims=True)
        f = np.round(np.where(tot > 0, M / np.where(tot > 0, tot, 1), 0) @ F, ROUND)
        fid = np.unique(f, axis=0, return_inverse=True)[1].ravel()
        pair = state * (fid.max() + 1) + fid[inv]
        state = np.unique(pair, return_inverse=True)[1].ravel()
    inv, M = group(state)
    p = M @ ymat
    ps = p.sum(1)
    h_c = float((ps * _H(p / ps[:, None])).sum())
    inv, M = group(np.unique(X, axis=0, return_inverse=True)[1].ravel())
    p = M @ ymat
    ps = p.sum(1)
    h_x = float((ps * _H(p / ps[:, None])).sum())
    return h_c - h_x


# ============================================================================= tree width

def treewidth(nodes, cliques):
    """Exact tree width of the graph on `nodes` whose edges join every pair inside each clique (DP over vertex
    subsets, Bodlaender et al. 2006); fine up to ~16 nodes."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return 0
    ix = {v: k for k, v in enumerate(nodes)}
    adj = [0] * n
    for c in cliques:
        c = [ix[v] for v in c if v in ix]
        for a in c:
            for b in c:
                if a != b:
                    adj[a] |= 1 << b

    def q(S, v):
        seen, stack, out = 1 << v, [v], 0
        while stack:
            u = stack.pop()
            nb = adj[u] & ~seen
            while nb:
                b = nb & -nb
                k = b.bit_length() - 1
                nb ^= b
                seen |= b
                if S >> k & 1:
                    stack.append(k)
                else:
                    out += 1
        return out

    tw = {0: -1}
    for size in range(1, n + 1):
        for comb in itertools.combinations(range(n), size):
            S = sum(1 << k for k in comb)
            best = n
            for v in comb:
                Sv = S & ~(1 << v)
                best = min(best, max(tw[Sv], q(Sv, v)))
            tw[S] = best
    return tw[(1 << n) - 1]


def market_graph(cell, sigma):
    """Nodes and cliques of the market's model for pricing Y from sigma: base variables, Y and every listed
    non-base security, each defined through its cheapest available scope (base variables or listed securities)."""
    listed = [z for z in sigma if isinstance(z, str)]
    for z in sigma:
        if isinstance(z, tuple):
            listed += list(z)
    avail = set(cell.latent) | set(listed)
    nodes = list(cell.latent)
    cliques = [tuple(c) for c in cell.prior_cliques]
    todo = [z for z in dict.fromkeys(listed + ["Y"]) if z not in cell.latent]
    for z in todo:
        nodes.append(z)
        opts = [o for o in cell.defs.get(z, []) if set(o) <= (avail | set(cell.latent)) - {z}]
        scope = min(opts, key=len) if opts else tuple(cell.latent)
        cliques.append((z,) + tuple(scope))
    for z in sigma:
        if isinstance(z, tuple):
            cliques.append(z)
    return nodes, cliques


def market_treewidth(cell, sigma):
    nodes, cliques = market_graph(cell, sigma)
    used = set(v for c in cliques for v in c)
    return treewidth([v for v in nodes if v in used], cliques)


# ============================================================================= cell builders

def _bits(W, n):
    w = np.arange(W)
    return (w[:, None] >> np.arange(n)) & 1


def _bsc(read, acc):
    """(W, 2) channel of a binary reading of `read` (W,) with accuracy acc."""
    return np.stack([np.where(read == 0, acc, 1 - acc), np.where(read == 1, acc, 1 - acc)], 1)


def xor_cell(acc=1.0):
    b = _bits(4, 2)
    y = b[:, 0] ^ b[:, 1]
    return Cell("XOR", np.full(4, .25), y, [_bsc(b[:, 0], acc), _bsc(b[:, 1], acc)],
                {"Y": y, "b1": b[:, 0], "b2": b[:, 1], "and": b[:, 0] & b[:, 1]},
                latent=("b1", "b2"), defs={"Y": [("b1", "b2")], "and": [("b1", "b2")]},
                reads=[("b1",), ("b2",)], meta={"acc": acc})


def and_cell(acc=1.0):
    b = _bits(4, 2)
    y = b[:, 0] & b[:, 1]
    return Cell("AND", np.full(4, .25), y, [_bsc(b[:, 0], acc), _bsc(b[:, 1], acc)],
                {"Y": y, "b1": b[:, 0], "b2": b[:, 1]}, latent=("b1", "b2"), defs={"Y": [("b1", "b2")]},
                reads=[("b1",), ("b2",)], meta={"acc": acc})


def copy_cell(src=0.85, read=0.95, n_copies=2):
    """Duplicate source: S = Y with prob src; n_copies agents read S with accuracy `read` (read = 1: exact copies).
    w = (Y, S)."""
    b = _bits(4, 2)
    y, s = b[:, 0], b[:, 1]
    prior = 0.5 * np.where(s == y, src, 1 - src)
    return Cell("COPY", prior, y, [_bsc(s, read) for _ in range(n_copies)], {"Y": y, "S": s},
                latent=("Y", "S"), prior_cliques=(("Y", "S"),), reads=[("S",)] * n_copies,
                meta={"src": src, "read": read})


def mix_cell(kind, t, acc=0.9):
    """Scaling probe. Each agent reads Y with prob 1 - t and its own variable with prob t (accuracy acc); at t = 0
    every log-likelihood lies in span(1, Y). kind: 'and' (w = b1, b2; Y = b1 & b2), 'copy' (w = Y, S; both read S)."""
    b = _bits(4, 2)
    if kind == "and":
        y = b[:, 0] & b[:, 1]
        own = [b[:, 0], b[:, 1]]
        prior = np.full(4, .25)
        sec = {"Y": y, "b1": b[:, 0], "b2": b[:, 1]}
    else:
        y, s = b[:, 0], b[:, 1]
        own = [s, s]
        prior = 0.5 * np.where(s == y, 0.8, 0.2)
        sec = {"Y": y, "S": s}
    lik = [(1 - t) * _bsc(y, acc) + t * _bsc(o, acc) for o in own]
    return Cell(f"mix-{kind}", prior, y, lik, sec, meta={"t": t})


def _parity_secs(bits, k, nd, nv):
    sec = {"Y": bits[:, :k].sum(1) % 2}
    for j in range(k):
        sec[f"b{j + 1}"] = bits[:, j]
    for j in range(nd):
        sec[f"d{j + 1}"] = bits[:, k + j]
    for j in range(2, k):
        sec[f"s{j}"] = bits[:, :j].sum(1) % 2
    sec["maj"] = (bits[:, :k].sum(1) * 2 >= k).astype(np.int64)
    latent = tuple(f"b{j + 1}" for j in range(k)) + tuple(f"d{j + 1}" for j in range(nd))
    bs = tuple(f"b{j + 1}" for j in range(k))
    defs = {"Y": [bs] + ([(f"s{k - 1}", f"b{k}")] if k >= 3 else []), "maj": [bs]}
    for j in range(2, k):
        defs[f"s{j}"] = [bs[:j]] + ([(f"s{j - 1}", f"b{j}")] if j >= 3 else [])
    return sec, latent, defs


def _var_name(v, k):
    return f"b{v + 1}" if v < k else f"d{v - k + 1}"


def z9_cell(seed, **overrides):
    from aggbench.zoo import KParityEnv
    env = KParityEnv("Z9", **overrides)
    st = env.structure(np.random.default_rng(seed))
    prior, target, lik, _ = env.tables(st)
    k, nd = env.config.n_bits, env.config.n_distractor_bits
    nv = k + nd
    bits = _bits(len(prior), nv)
    sec, latent, defs = _parity_secs(bits, k, nd, nv)
    return Cell(f"Z9-k{k}", prior, target, [lik[i] for i in range(len(lik))], sec, latent, defs,
                reads=[(_var_name(v, k),) for v in st["var"]], meta={"seed": seed, **overrides})


def z9p2_cell(seed, **overrides):
    from aggbench.compression import PiecesPerAgent
    env = PiecesPerAgent("Z9", 2, **overrides)
    st = env.base.structure(np.random.default_rng(seed))
    prior, target, lik = env.grouped_tables(st)
    k, nd = env.base.config.n_bits, env.base.config.n_distractor_bits
    bits = _bits(len(prior), k + nd)
    sec, latent, defs = _parity_secs(bits, k, nd, k + nd)
    reads = [tuple(_var_name(v, k) for v in st["var"][2 * i:2 * i + 2]) for i in range(len(lik))]
    return Cell("Z9-p2", prior, target, [lik[i] for i in range(len(lik))], sec, latent, defs, reads=reads,
                meta={"seed": seed, **overrides})


def z6_cell(seed, **overrides):
    from aggbench.zoo import ParityEnv
    env = ParityEnv("Z6", **overrides)
    st = env.structure(np.random.default_rng(seed))
    prior, target, lik, _ = env.tables(st)
    k = env.config.n_bits
    bits = _bits(len(prior), k)
    sec, latent, defs = _parity_secs(bits, k, 0, k)
    if k == 2:
        sec["and"] = bits[:, 0] & bits[:, 1]
        defs["and"] = [("b1", "b2")]
    return Cell(f"Z6-k{k}", prior, target, [lik[i] for i in range(len(lik))], sec, latent, defs,
                reads=[(f"b{v + 1}",) for v in st["var"]], meta={"seed": seed, **overrides})


def z5_cell(seed, **overrides):
    from aggbench.zoo import MultiQuestionEnv
    env = MultiQuestionEnv("Z5", **overrides)
    st = env.structure(np.random.default_rng(seed))
    prior, target, lik, _ = env.tables(st)
    a, b, c = env.bits()
    names = ("A", "B", "C")
    return Cell("Z5", prior, target, [lik[i] for i in range(len(lik))], {"Y": target, "A": a, "B": b, "C": c},
                latent=names, defs={"Y": [names]}, prior_cliques=(("A", "B"), ("A", "C"), ("B", "C")),
                reads=[(names[v],) for v in st["var"]], meta={"seed": seed, **overrides})


def e3_sybil_cell(seed, n_pieces=12, n_agents=5, n_sybil=3, pieces_per_agent=(2, 6), strength_levels=(0.2, 0.5)):
    """E3-sybil discretised. Pieces x_k = sign of the continuous reading (P(x_k = Y) = Phi(mu_k)), strengths mu_k on
    two levels (continuous strengths would make every agent's Y-forecast reveal its readings); agents read their
    covered pieces exactly; the last honest agent is copied n_sybil times. w = (Y, x_1..x_K)."""
    rng = np.random.default_rng(seed)
    K = n_pieces
    mu = rng.choice(np.asarray(strength_levels), K)
    acc = np.array([0.5 * math.erfc(-m / math.sqrt(2)) for m in mu])
    cov = np.zeros((n_agents, K), bool)
    for i in range(n_agents):
        cov[i, rng.choice(K, rng.integers(pieces_per_agent[0], pieces_per_agent[1] + 1), replace=False)] = True
    cov = np.vstack([cov, np.repeat(cov[-1:], n_sybil, 0)])
    cov = cov[rng.permutation(len(cov))]
    W = 2 ** (K + 1)
    bits = _bits(W, K + 1)
    y, x = bits[:, 0], bits[:, 1:]
    prior = 0.5 * np.prod(np.where(x == y[:, None], acc, 1 - acc), 1)
    lik = []
    for i in range(len(cov)):
        idx = np.flatnonzero(cov[i])
        sym = (x[:, idx] << np.arange(len(idx))).sum(1)
        lik.append(_onehot(sym, 2 ** len(idx)))
    sec = {"Y": y}
    for k in range(K):
        sec[f"x{k + 1}"] = x[:, k]
    latent = ("Y",) + tuple(f"x{k + 1}" for k in range(K))
    return Cell("E3-sybil-d", prior, y, lik, sec, latent, {}, prior_cliques=tuple(("Y", f"x{k + 1}") for k in range(K)),
                reads=[tuple(f"x{k + 1}" for k in np.flatnonzero(c)) for c in cov],
                meta={"seed": seed, "covered": int(cov.any(0).sum())})


# ============================================================================= best span (SVD), principal angles, bound

def loglik_matrix(cell):
    """Row-centred log-likelihood matrix in L2(pi): one row per (agent, symbol) with p(o) > 0, row weight sqrt p(o),
    column weight sqrt pi(w); each row centred by its pi-mean (constants are free). None if a log-likelihood is
    infinite (deterministic readings)."""
    rows, sw = [], np.sqrt(cell.prior)
    for L in cell.lik:
        po = cell.prior @ L
        for o in np.flatnonzero(po > 0):
            ll = _log(L[:, o])
            if not np.all(np.isfinite(ll[cell.prior > 0])):
                return None
            rows.append(math.sqrt(po[o]) * (ll - cell.prior @ ll) * sw)
    return np.array(rows)


def span_svd(cell, rtol=1e-8):
    """Singular values of the row-centred log-likelihood matrix, the minimal span dimension d_omega (numerical
    rank) and the right singular vectors (sqrt(pi)-weighted coordinates). The best rank-d L2 span is the top d."""
    M = loglik_matrix(cell)
    if M is None:
        return None
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    d = int((s > rtol * max(s.max(), 1e-300)).sum())
    return {"sv": s, "d_omega": d, "V": Vt.T, "M": M}


def _weighted_basis(cell, sigma):
    phi = basis(cell, sigma)
    if phi.shape[1] == 0:
        return np.zeros((cell.W, 0))
    B = (phi - cell.prior @ phi) * np.sqrt(cell.prior)[:, None]
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, s > 1e-10 * max(s.max(), 1e-300)]


def principal_angles(cell, sigma, d=None, svd=None):
    """Principal angles (degrees, largest first) between the top-d log-likelihood span (d = d_omega by default) and
    span(phi), and the share of log-likelihood energy outside span(phi). All angles 0 iff the top-d span lies
    inside span(phi)."""
    svd = svd or span_svd(cell)
    if svd is None:
        return None
    d = svd["d_omega"] if d is None else d
    Q = _weighted_basis(cell, sigma)
    V = svd["V"][:, :d]
    if Q.shape[1] == 0:
        ang = np.full(d, 90.0)
    else:
        cos = np.linalg.svd(V.T @ Q, compute_uv=False)
        cos = np.r_[cos, np.zeros(max(0, d - len(cos)))][:d]
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    M = svd["M"]
    outside = M - (M @ Q) @ Q.T if Q.shape[1] else M
    return {"angles_deg": np.sort(ang)[::-1], "dim_phi": int(Q.shape[1]), "d": d,
            "energy_outside": float((outside ** 2).sum() / max((M ** 2).sum(), 1e-300))}


def osc_bound_check(cell, sigma, max_tuples=2 ** 16):
    """Exact over observation tuples. With R_x(w) = sum_i [log lik_i(w, o_i) - theta_i . phi(w)], the exact posterior
    is p(w | x) ~ q(w | x) e^{R_x} (q = summed trades), so KL_w(p || q) <= osc(R_x)^2 / 8 (nats), and the Y-regret is
    at most KL_w. Returns the worst ratio KL_w / (osc^2 / 8), the count of violations and the means (nats)."""
    sizes = [L.shape[1] for L in cell.lik]
    T = int(np.prod(sizes, dtype=float))
    live = cell.prior > 0
    if T > max_tuples or any(not np.all(L[live] > 0) for L in cell.lik):
        return None
    cls = forecast_classes(cell, sigma)
    tr = agent_trades(cell, sigma, cls)
    phi = basis(cell, sigma)
    ymat = _onehot(cell.target)
    logpi = _log(cell.prior)
    X = np.stack(np.unravel_index(np.arange(T), sizes), 1)
    ll = sum(_log(cell.lik[i][:, X[:, i]]).T for i in range(cell.N))                      # (T, W)
    tq = sum(tr[i]["theta"][cls[i][0][X[:, i]]] @ phi.T for i in range(cell.N))           # (T, W)
    lp, lq = logpi + ll, logpi + tq
    px = np.exp(_lse_rows(lp))
    p = np.exp(lp - _lse_rows(lp)[:, None])
    q = np.exp(lq - _lse_rows(lq)[:, None])

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * (np.log(a) - np.log(np.maximum(b, 1e-300))), 0.0).sum(1)

    klw, kly = kl(p, q), kl(p @ ymat, q @ ymat)
    R = (ll - tq)[:, live]
    bound = (R.max(1) - R.min(1)) ** 2 / 8
    ratio = np.where(bound > 1e-12, klw / np.maximum(bound, 1e-300), 0.0)
    return {"max_ratio": float(ratio.max()), "violations": int((klw > bound + 1e-9).sum()),
            "y_above_w": int((kly > klw + 1e-9).sum()),
            "E_KL_Y": float(px @ kly), "E_KL_w": float(px @ klw), "E_bound": float(px @ bound)}
