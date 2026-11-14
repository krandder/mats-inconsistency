"""Task 2 (a second world for M2's "across tasks"): in-context linear regression with an unknown noise level.
Episode: w ~ N(0, I_D); noise variance v uniform over VLEV = (0.1, 0.5, 2.0); N context examples x_i ~ N(0, I_D),
y_i = w.x_i + e_i, e_i ~ N(0, v); a query x*, target y* = w.x* + e*, e* ~ N(0, v).
Exact Bayes predictive: a mixture over the three levels. Component v is Gaussian (ridge with noise variance v):
  A = I + X^T X / v, S = A^-1, m = S X^T y / v, mean x*.m, variance v + x*^T S x*;
its weight is proportional to the marginal likelihood N(y; 0, X X^T + v I) (uniform prior on the level), computed with the
determinant lemma and Woodbury: log det(X X^T + v I) = N log v + log det A, y^T (X X^T + v I)^-1 y = (y.y - b^T S b / v) / v with
b = X^T y.
Symmetry: the posterior is invariant under flipping any context example, (x_i, y_i) -> (-x_i, -y_i) (X^T X, X^T y and y.y are
unchanged). Token formats:
  ctrl   context tokens [x_i (D), y_i, 0]; query token [x*, 0, 1];
  canon  each context example replaced by its orbit representative with y_i >= 0 (x_i co-flipped); the query is unchanged.
The sampler consumes the same random draws in both formats.
"""
import math
import numpy as np

VLEV = np.array([0.1, 0.5, 2.0])
FMTS = ('ctrl', 'canon')


def sample(n, rng, D, N):
    w = rng.standard_normal((n, D))
    li = rng.integers(0, len(VLEV), n); v = VLEV[li]
    x = rng.standard_normal((n, N + 1, D))
    e = rng.standard_normal((n, N + 1)) * np.sqrt(v)[:, None]
    y = np.einsum('nkd,nd->nk', x, w) + e
    return dict(x=x[:, :N], y=y[:, :N], xq=x[:, N], yq=y[:, N], li=li, w=w)


def tokens(d, fmt):
    x, y, xq = d['x'], d['y'], d['xq']
    n, N, D = x.shape
    if fmt == 'canon':
        s = np.where(y < 0, -1.0, 1.0)
        x = x * s[:, :, None]; y = y * s
    X = np.zeros((n, N + 1, D + 2), dtype=np.float32)
    X[:, :N, :D] = x; X[:, :N, D] = y
    X[:, N, :D] = xq; X[:, N, D + 1] = 1.0
    return X, np.ones((n, N + 1), dtype=bool)


def bayes(d):
    """Exact predictive per episode: component means, variances and normalised weights, each (n, 3)."""
    x, y, xq = d['x'].astype(np.float64), d['y'].astype(np.float64), d['xq'].astype(np.float64)
    n, N, D = x.shape
    G = np.einsum('nkd,nke->nde', x, x); b = np.einsum('nkd,nk->nd', x, y); yy = (y * y).sum(1)
    mus, vas, lls = [], [], []
    for v in VLEV:
        A = np.eye(D)[None] + G / v
        S = np.linalg.inv(A); m = np.einsum('nde,ne->nd', S, b) / v
        mus.append(np.einsum('nd,nd->n', xq, m)); vas.append(v + np.einsum('nd,nde,ne->n', xq, S, xq))
        quad = (yy - np.einsum('nd,nde,ne->n', b, S, b) / v) / v
        lls.append(-0.5 * (N * math.log(2 * math.pi) + N * math.log(v) + np.linalg.slogdet(A)[1] + quad))
    L = np.stack(lls, 1); L -= L.max(1, keepdims=True); wts = np.exp(L); wts /= wts.sum(1, keepdims=True)
    return np.stack(mus, 1), np.stack(vas, 1), wts


def mix_nll(means, vars_, wts, y):
    lp = np.log(wts + 1e-300) - 0.5 * np.log(2 * math.pi * vars_) - 0.5 * (y[:, None] - means) ** 2 / vars_
    mx = lp.max(1, keepdims=True)
    return -(mx[:, 0] + np.log(np.exp(lp - mx).sum(1)))


def references(d):
    """Bayes NLL of y*, and two references: level-blind (the mixture with the prior's level weights) and the oracle level."""
    mu, va, wt = bayes(d)
    yq = d['yq'].astype(np.float64)
    out = dict(bayes=mix_nll(mu, va, wt, yq), blind=mix_nll(mu, va, np.full_like(wt, 1.0 / len(VLEV)), yq))
    oi = d['li']
    out['oracle'] = 0.5 * np.log(2 * math.pi * va[np.arange(len(oi)), oi]) + 0.5 * (yq - mu[np.arange(len(oi)), oi]) ** 2 / va[np.arange(len(oi)), oi]
    return out, (mu, va, wt)
