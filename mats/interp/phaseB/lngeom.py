"""LayerNorm geometry of block 1's labelled tokens (sixth agent; a tool, used by a later registered episode).
For a world-1 net and real labelled tokens X_i (o_i = y_i + r_i), the input embedding is x_i = W X_i + b. With P the centring
projection, write P x_i = a_i + r_i u~, where a_i = P x_i at r_i = 0 (the token with o replaced by y) and u~ = P W[:, o] (the
o column). Block 1's LayerNorm divides by n_i = |a_i + r_i u~| = sqrt(A_i^2 + 2 r_i A_i U rho_i + r_i^2 U^2), so a uniform mean of
the tokens' values carries the even, saturating per-token feature 1 / n_i, approximately 1 - (1 + r^2 / c^2)^(-1/2) up to an
affine map, with cap scale c_i = A_i / U.
Returns per-token A_i, U, rho_i = a_i.u~ / (A_i U) and c_i on the given tokens, and summaries (median c, the spread of A_i).
usage (module): geom(model, X) or geom_wb(W, b, X), X tokens in the net's own format (n, T, F); only labelled tokens are
used; "r" is the o column minus the y column (r, |r| or r^2 by format).
"""
import numpy as np
import torch

LAB = [5 * s + q for s in range(3) for q in range(4)]
OCOL, YCOL = 32, 33


def geom(model, X):
    return geom_wb(model.inp.weight.detach().numpy(), model.inp.bias.detach().numpy(), X)


def geom_wb(W, b, X):
    """The same from the input embedding's weight (d, F) and bias (d,) arrays."""
    W = np.asarray(W, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    T = X[:, LAB, :].reshape(-1, X.shape[2]).astype(np.float64)
    r = T[:, OCOL] - T[:, YCOL]
    T0 = T.copy(); T0[:, OCOL] = T0[:, YCOL]                       # the same token with r = 0
    a = T0 @ W.T + b; a = a - a.mean(1, keepdims=True)
    u = W[:, OCOL] - W[:, OCOL].mean()
    A = np.linalg.norm(a, axis=1); U = float(np.linalg.norm(u))
    rho = (a @ u) / (A * U)
    c = A / U
    return dict(A=A, U=U, rho=rho, c=c, r=r, c_median=float(np.median(c)), A_cv=float(A.std() / A.mean()),
                rho_abs_median=float(np.median(np.abs(rho))))


def check(model, X, tol=1e-4):
    """The decomposition reproduces the LayerNorm's norm exactly: |P x_i| = |a_i + r_i u~|."""
    W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
    T = X[:, LAB, :].reshape(-1, X.shape[2]).astype(np.float64)
    x = T @ W.T + b; x = x - x.mean(1, keepdims=True)
    g = geom(model, X)
    n = np.sqrt(g['A'] ** 2 + 2 * g['r'] * g['A'] * g['U'] * g['rho'] + g['r'] ** 2 * g['U'] ** 2)
    return float(np.abs(n - np.linalg.norm(x, axis=1)).max()) < tol * float(np.linalg.norm(x, axis=1).max())
