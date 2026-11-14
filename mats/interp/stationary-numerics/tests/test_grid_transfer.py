"""Tests for lib/grid_transfer.py (dev net seed 3).
1. Refinement: TV(dx, dx/2) and relative moment differences at layer 0 are small (n = 2: dx 0.1 vs 0.05; n = 3: 0.2
   vs 0.14).
2. Pullback convergence: two different initial laws (N(0, I) and N(5 e1, 4 I)) give TV -> 0 as T grows.
3. Monte Carlo: 4e5 noise paths through the same weights (T layers from h = 0) match the grid's mean and covariance
   within 4 SE and its orthant masses within 4 SE.
Run: python tests/test_grid_transfer.py (exit 0 = pass)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
import grid_transfer as gt
import st

fails = []


def check(name, ok, info):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + info, flush=True)
    if not ok:
        fails.append(name)


def moments(G, mass):
    X = G.points()
    m = mass.ravel()
    mu = (X * m[:, None]).sum(0) / m.sum()
    C = ((X - mu) * m[:, None]).T @ (X - mu) / m.sum()
    return mu, C


seed = 3
for n, dxs, T in ((2, (0.1, 0.05), 40), (3, (0.2, 0.14), 30)):
    t0 = time.time()
    G1, m1, *_ = gt.pullback(seed, n, T, dxs[0])
    G2, m2, *_ = gt.pullback(seed, n, T, dxs[1])
    tv = 0.5 * np.abs(gt.resample_to(G1, m1, G2) - m2).sum()
    mu1, C1 = moments(G1, m1)
    mu2, C2 = moments(G2, m2)
    rel = max(np.max(np.abs(mu1 - mu2)) / np.sqrt(np.trace(C2) / n), np.max(np.abs(C1 - C2)) / np.max(np.abs(C2)))
    check(f'n={n} refine', tv < 5e-3 and rel < 5e-3, f'TV {tv:.2e}, max rel moment diff {rel:.2e} ({time.time() - t0:.0f}s)')
    # convergence in T from two starts
    for Tt in (5, 10, 20, T):
        Ga, ma, *_ = gt.pullback(seed, n, Tt, dxs[0])
        Gb, mb, *_ = gt.pullback(seed, n, Tt, dxs[0], init=gt.gaussian_init(n, dxs[0], center=[5.0] + [0.0] * (n - 1), scale=2.0))
        tvT = 0.5 * np.abs(gt.resample_to(Ga, ma, Gb) - mb).sum()
        print(f'   n={n} T={Tt}: TV between starts {tvT:.2e}', flush=True)
    check(f'n={n} pullback', tvT < 1e-3, f'TV at T={T}: {tvT:.2e}')
    # Monte Carlo
    rng = np.random.default_rng([99, n])
    P = 400000
    h = np.zeros((P, n))
    for l in range(-T + 1, 1):
        h = np.maximum(h @ st.W_layer(seed, n, l).T, 0.0) + rng.standard_normal((P, n))
    muM = h.mean(0)
    CM = np.cov(h.T)
    se_mu = np.sqrt(np.diag(CM) / P)
    z = np.max(np.abs(muM - mu2) / se_mu)
    check(f'n={n} MC mean', z < 4, f'max |z| = {z:.2f}')
    # covariance SE via batch means
    Cb = np.array([np.cov(hb.T) for hb in h.reshape(40, -1, n)])
    zc = np.max(np.abs(CM - C2) / (Cb.std(0) / math.sqrt(40)))
    check(f'n={n} MC cov', zc < 4, f'max |z| = {zc:.2f}')
    X = G2.points()
    for orth in range(2 ** n):
        sgn = np.array([(orth >> k) & 1 for k in range(n)])
        # cell-fraction rule for nodes within dx/2 of a coordinate plane
        wnode = np.ones(X.shape[0])
        for k in range(n):
            fpos = np.clip(X[:, k] / G2.dx + 0.5, 0.0, 1.0)  # fraction of the node's cell on the positive side
            s_k = fpos if sgn[k] else 1.0 - fpos
            wnode = wnode * s_k
        pg = float((m2.ravel() * wnode).sum())
        pm = np.mean(np.all((h > 0) == sgn[None].astype(bool), 1))
        zz = (pm - pg) / math.sqrt(max(pg * (1 - pg), 1e-12) / P)
        if abs(zz) >= 4:
            check(f'n={n} orthant {orth}', False, f'grid {pg:.5f} MC {pm:.5f} z {zz:.2f}')
    check(f'n={n} orthants', True, 'all orthant masses within 4 SE (no FAIL lines above)')

print('FAILS:', fails)
sys.exit(1 if fails else 0)
