"""Deterministic transfer operator of Model S (sigma = 1) on an n-dimensional grid, for one fixed net (n = 2, 3, 4).
Layer map: p_l(y) = int phi(y - relu(W_l x)) p_{l-1}(x) dx, phi = N(0, I_n) density.
Step: (1) push the cell masses of p_{l-1} through x -> relu(W_l x); (2) deposit them on the new grid with linear
(cloud-in-cell) weights; (3) convolve with the sampled unit Gaussian, separably along each axis. No sampling anywhere.
The new box is the bounding box of the pushed mass (cells above a 1e-18 relative mass) plus a margin of `marg`
noise SDs. Error: O((dx/sigma)^2) from the midpoint and cloud-in-cell steps, smoothed by the Gaussian; checked by
refinement in tests/test_grid_transfer.py.
Weights: W_l from the stationary lane's st.W_layer(seed, n, l) (the same nets)."""
import math, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st


class Grid:
    def __init__(self, lo, dx, shape):
        self.lo = np.asarray(lo, float)
        self.dx = float(dx)
        self.shape = tuple(int(s) for s in shape)

    def axes(self):
        return [self.lo[k] + self.dx * np.arange(self.shape[k]) for k in range(len(self.shape))]

    def points(self):
        g = np.meshgrid(*self.axes(), indexing='ij')
        return np.stack([x.ravel() for x in g], 1)


def gaussian_init(n, dx, center=None, scale=1.0):
    center = np.zeros(n) if center is None else np.asarray(center, float)
    half = 7.0 * scale
    m = int(math.ceil(2 * half / dx)) + 1
    G = Grid(center - half, dx, (m,) * n)
    X = G.points() - center[None]
    p = np.exp(-0.5 * np.sum(X * X, 1) / scale ** 2)
    p = p.reshape(G.shape)
    return G, p / p.sum()


def deposit(Gnew, Y, w):
    """Cloud-in-cell deposit of masses w at points Y (N, n) onto grid Gnew."""
    n = Y.shape[1]
    u = (Y - Gnew.lo[None]) / Gnew.dx
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    shape = Gnew.shape
    strides = np.array([int(np.prod(shape[k + 1:])) for k in range(n)], np.int64)
    size = int(np.prod(shape))
    out = np.zeros(size)
    for corner in range(2 ** n):
        bits = np.array([(corner >> k) & 1 for k in range(n)])
        idx = i0 + bits[None]
        wt = w.copy()
        for k in range(n):
            wt = wt * (f[:, k] if bits[k] else (1 - f[:, k]))
        ok = np.all((idx >= 0) & (idx < np.array(shape)[None]), 1)
        out += np.bincount(idx[ok] @ strides, weights=wt[ok], minlength=size)
    return out.reshape(shape)


def smooth(nu, dx):
    out = nu
    for k in range(nu.ndim):
        out = gaussian_filter1d(out, sigma=1.0 / dx, axis=k, mode='constant', truncate=8.0)
    return out


def layer(G, mass, W, dx, marg=8.0, rel_cut=1e-18, return_points=False):
    """One layer. Returns (new grid, new cell masses[, pushed points A, their masses m, preactivations Z])."""
    X = G.points()
    m = mass.ravel()
    keep = m > rel_cut * m.max()
    X, m = X[keep], m[keep]
    Z = X @ W.T
    A = np.maximum(Z, 0.0)
    lo = A.min(0) - marg
    hi = A.max(0) + marg
    shape = np.ceil((hi - lo) / dx).astype(int) + 1
    Gn = Grid(lo, dx, shape)
    out = smooth(deposit(Gn, A, m), dx)
    if return_points:
        return Gn, out, A, m, Z
    return Gn, out


def pullback(seed, n, T, dx, init=None, marg=8.0, return_points=True):
    """Iterate the layers -T+1 .. 0 of net `seed` from an initial law (default N(0, I)); returns the grid and cell
    masses at layer 0, plus the pushed points of the last layer (pre-noise atoms of relu(W_0 h_-1))."""
    G, mass = gaussian_init(n, dx) if init is None else init
    for l in range(-T + 1, 1):
        W = st.W_layer(seed, n, l)
        if l == 0:
            return layer(G, mass, W, dx, marg, return_points=return_points)
        G, mass = layer(G, mass, W, dx, marg)


def resample_to(Gsrc, msrc, Gdst):
    """Linear interpolation of cell masses (as densities) from one grid to another (for TV comparisons)."""
    from scipy.interpolate import RegularGridInterpolator
    nd = len(Gsrc.shape)
    f = RegularGridInterpolator(Gsrc.axes(), msrc / Gsrc.dx ** nd, bounds_error=False, fill_value=0.0)
    return f(Gdst.points()).reshape(Gdst.shape) * Gdst.dx ** nd
