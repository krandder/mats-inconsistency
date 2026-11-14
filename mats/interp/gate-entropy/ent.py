"""Entropy estimators for high-dimensional binary patterns (gate-entropy, 2026-10-29). All entropies in bits.

Point estimates and bounds:
  plugin(c)        maximum likelihood; E[plugin] <= H (negatively biased for every distribution) -> a lower bound in
                   expectation; saturates at log2 N.
  miller_madow(c)  plugin + (K_obs - 1) / (2 N ln 2).
  nsb(c, logK)     Nemenman-Shafee-Bialek posterior mean over the symmetric-Dirichlet mixture with the NSB prior;
                   reliable only when there are coincidences (N - K_obs >> 1); returns the beta-posterior spread too.
  collision(c)     Renyi-2 entropy from the unbiased pair-collision estimate; H >= H_2, so a lower bound; usable up to
                   about 2 log2 N bits. With zero collisions, returns the bound log2(#pairs / 3) (rule-of-three on pairs,
                   approximate because pairs are not independent).
  indep_upper(G)   sum of per-unit entropies (exact upper bound; per-unit plug-in bias is negligible at our N).
  made_upper(...)  cross-entropy of an autoregressive model (MADE, one hidden layer, or FVSBN) fitted on a train split and
                   evaluated on a disjoint test split: an unbiased estimate of a cross-entropy, hence of an upper bound.
"""
import math
import numpy as np
import torch
from scipy.special import gammaln, digamma, polygamma

LN2 = math.log(2.0)


def pattern_ids(G):
    """G (N, m) bool/uint8 -> (counts per distinct row, inverse index)."""
    G = np.ascontiguousarray(G, dtype=bool)
    if G.shape[1] == 0:
        return np.array([len(G)]), np.zeros(len(G), dtype=np.int64)
    P = np.packbits(G, axis=1)
    v = np.ascontiguousarray(P).view(np.dtype((np.void, P.shape[1])))[:, 0]
    _, inv, c = np.unique(v, return_inverse=True, return_counts=True)
    return c, inv


def plugin(c):
    c = np.asarray(c, dtype=np.float64); c = c[c > 0]; p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def miller_madow(c):
    c = np.asarray(c); N = c.sum()
    return plugin(c) + (np.count_nonzero(c) - 1) / (2 * N * LN2)


def collision(c):
    """Renyi-2 entropy (bits) from unbiased collision probability; (H2, n_collision_pairs, n_pairs)."""
    c = np.asarray(c, dtype=np.float64); N = c.sum()
    npairs = N * (N - 1) / 2
    coll = (c * (c - 1) / 2).sum()
    if coll == 0:
        return float(math.log2(npairs / 3.0)), 0.0, npairs, True
    return float(-math.log2(coll / npairs)), float(coll), npairs, False


def nsb(c, logK_bits, ngrid=1500):
    """NSB estimate (bits) for counts c over an alphabet of size K = 2^logK_bits.
    Returns dict(H, sd_beta (spread of E[H|n,beta] under the beta posterior), delta = N - K_obs)."""
    c = np.asarray(c, dtype=np.float64); c = c[c > 0]
    N = c.sum(); K1 = len(c)
    K = 2.0 ** logK_bits if logK_bits < 1000 else np.inf
    if not np.isfinite(K):
        raise ValueError('alphabet too large for float64')
    # grid over log(K beta) (concentration a = K beta), from 1e-3 to 1e3 * N
    la = np.linspace(math.log(1e-4), math.log(1e4 * max(N, 10)), ngrid)
    a = np.exp(la); beta = a / K
    # log evidence p(n | beta) up to a constant
    uniq, mult = np.unique(c, return_counts=True)
    le = gammaln(a) - gammaln(N + a)
    le = le + (mult[None, :] * (gammaln(uniq[None, :] + beta[:, None]) - gammaln(beta[:, None]))).sum(1)
    # NSB prior d xi / d beta, with d beta = beta d(log a)
    dxi = K * polygamma(1, a + 1) - polygamma(1, beta + 1)
    lw = le + np.log(np.maximum(dxi, 1e-300)) + np.log(beta)
    lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    # posterior mean entropy (nats) given beta
    EH = digamma(N + a + 1) - (mult[None, :] * (uniq[None, :] + beta[:, None]) / (N + a)[:, None]
                                * digamma(uniq[None, :] + beta[:, None] + 1)).sum(1) \
        - (K - K1) * beta / (N + a) * digamma(beta + 1)
    Hm = float((w * EH).sum()) / LN2
    sd = float(math.sqrt(max((w * (EH / LN2 - Hm) ** 2).sum(), 0.0)))
    return dict(H=Hm, sd_beta=sd, delta=float(N - K1), K_obs=int(K1), N=int(N))


def h2(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def indep_upper(G, groups=None):
    """Sum of per-unit entropies. groups: optional list of column-index arrays treated as one categorical unit."""
    G = np.asarray(G, dtype=bool)
    if groups is None:
        return float(h2(G.mean(0)).sum())
    return float(sum(plugin(pattern_ids(G[:, g])[0]) for g in groups))


class MADE(torch.nn.Module):
    """Autoregressive Bernoulli model over m bits in natural order. hidden=0 -> FVSBN (masked logistic)."""

    def __init__(self, m, hidden=0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m, self.hidden = m, hidden
        if hidden == 0:
            self.W = torch.nn.Parameter(torch.zeros(m, m))
            self.register_buffer('M', torch.tril(torch.ones(m, m), -1))          # out i reads in j < i
        else:
            deg_in = torch.arange(m)
            deg_h = torch.randint(0, max(m - 1, 1), (hidden,), generator=g)
            self.register_buffer('M1', (deg_h[:, None] >= deg_in[None, :]).float())   # (h, m)
            self.register_buffer('M2', (deg_in[:, None] > deg_h[None, :]).float())    # (m, h)
            self.W1 = torch.nn.Parameter(torch.randn(hidden, m, generator=g) * 0.01)
            self.b1 = torch.nn.Parameter(torch.zeros(hidden))
            self.W2 = torch.nn.Parameter(torch.randn(m, hidden, generator=g) * 0.01)
            self.Wd = torch.nn.Parameter(torch.zeros(m, m))                       # direct connections
            self.register_buffer('M', torch.tril(torch.ones(m, m), -1))
        self.b = torch.nn.Parameter(torch.zeros(m))

    def logits(self, x):
        if self.hidden == 0:
            return x @ (self.W * self.M).T + self.b
        h = torch.relu(x @ (self.W1 * self.M1).T + self.b1)
        return h @ (self.W2 * self.M2).T + x @ (self.Wd * self.M).T + self.b

    def nll_bits(self, x):
        """per-sample negative log-likelihood in bits (N,)"""
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), x, reduction='none').sum(1) / LN2


def made_upper(G, hidden=0, n_test=None, epochs=60, bs=1024, lr=3e-3, wd=1e-5, seed=0, max_train=1 << 16, verbose=False):
    """Fit on a train split, early-stop on a validation split, report mean test NLL (bits) and its SE.
    Constant columns are removed first (they contribute 0 bits). Returns dict(H_up, se, n_train, n_test, m_eff)."""
    G = np.asarray(G, dtype=bool)
    keep = (G.mean(0) > 0) & (G.mean(0) < 1)
    X = torch.from_numpy(G[:, keep].astype(np.float32))
    N, m = X.shape
    if m == 0:
        return dict(H_up=0.0, se=0.0, n_train=0, n_test=N, m_eff=0)
    rng = np.random.default_rng(seed); perm = torch.from_numpy(rng.permutation(N))
    n_test = n_test or N // 4
    te, va, tr = perm[:n_test], perm[n_test:n_test + max(n_test // 2, 1024)], perm[n_test + max(n_test // 2, 1024):]
    tr = tr[:max_train]
    torch.manual_seed(seed)
    model = MADE(m, hidden, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), bs):
            xb = X[p[i:i + bs]]
            loss = model.nll_bits(xb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = model.nll_bits(X[va]).mean().item()
        if v < best - 1e-4:
            best, bad = v, 0; best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4: break
        if verbose: print(ep, v, flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        nl = model.nll_bits(X[te]).numpy().astype(np.float64)
    return dict(H_up=float(nl.mean()), se=float(nl.std() / math.sqrt(len(nl))), n_train=int(len(tr)), n_test=int(len(te)), m_eff=int(m))


def all_estimates(G, logK_bits=None, made=True, hidden=None, groups=None, max_made_N=1 << 17, seed=0):
    """Full estimator panel for patterns G (N, m)."""
    G = np.asarray(G, dtype=bool)
    N, m = G.shape
    logK = logK_bits if logK_bits is not None else m
    c, _ = pattern_ids(G)
    out = dict(N=int(N), m=int(m), K_obs=int(len(c)), plugin=plugin(c), mm=miller_madow(c),
               indep_upper=indep_upper(G, groups))
    H2, ncoll, npairs, zero = collision(c)
    out.update(H2=H2, H2_collisions=ncoll, H2_zero=bool(zero))
    try:
        out['nsb'] = nsb(c, min(logK, 900))
    except Exception as e:  # noqa
        out['nsb'] = dict(error=str(e))
    if made and m > 0:
        sub = G[:max_made_N]
        out['fvsbn'] = made_upper(sub, hidden=0, seed=seed)
        h = hidden if hidden is not None else min(max(2 * m, 64), 512)
        out['made'] = made_upper(sub, hidden=h, seed=seed)
        out['upper'] = min(out['indep_upper'], out['fvsbn']['H_up'] + 2 * out['fvsbn']['se'], out['made']['H_up'] + 2 * out['made']['se'])
    else:
        out['upper'] = out['indep_upper']
    out['lower'] = max(out['plugin'], out['H2'] if not out['H2_zero'] else out['plugin'])
    out['lower_H2_zero_bound'] = out['H2'] if out['H2_zero'] else None
    return out
