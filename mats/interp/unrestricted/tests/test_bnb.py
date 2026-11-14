"""Tests for lib/bnb.py and lib/nets.py. Run: python3 tests/test_bnb.py"""
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, make_weights, mean_radius, pass_flops, sphere_from_gauss  # noqa: E402
from bnb import CertifiedBnB, cell_moments, gl_nodes, default_q  # noqa: E402


def test_moments_partition():
    for n in (2, 3, 5, 8, 16):
        t, w = gl_nodes(n, 48)
        m = n - 1
        lo = -np.ones((1, m)); hi = np.ones((1, m))
        mass, Mf, Mt = cell_moments(lo, hi, t, w)
        # 2n faces, each of mass 1/(2n) by symmetry
        assert abs(mass[0] * 2 * n - 1) < 1e-10, (n, mass[0] * 2 * n)
        # E[|x_k| 1(face k, s)] summed over faces = E max_k |x_k| ... check the face's E[s x_k 1] by MC below
    print('moments partition ok')


REF = {}


def test_moments_quadrature():
    rng = np.random.default_rng(1)
    for n in (4, 8, 16, 32):
        REF[n] = gl_nodes(n, 400)
    for n in (4, 8, 16, 32):
        m = n - 1
        for trial in range(20):
            c = rng.uniform(-1, 1, m)
            wd = 10 ** rng.uniform(-4, 0, m)
            lo = np.clip(c - wd, -1, 1); hi = np.clip(c + wd, -1, 1)
            lo, hi = np.minimum(lo, hi - 1e-9), hi
            ref = cell_moments(lo[None], hi[None], *REF[n])
            q = default_q(n)
            got = cell_moments(lo[None], hi[None], *gl_nodes(n, q))
            rm = abs(got[0][0] / ref[0][0] - 1)
            rf = np.max(np.abs(got[1][0] - ref[1][0])) / (np.max(np.abs(ref[1][0])) + 1e-300)
            rt = abs(got[2][0] / ref[2][0] - 1)
            assert max(rm, rf, rt) < 1e-9, (n, q, rm, rf, rt)
    print('moments quadrature ok (default q within 1e-9 of q=400)')


def test_moments_mc():
    rng = np.random.default_rng(2)
    n, m = 5, 4
    k, s = 2, -1
    lo = np.array([-0.3, 0.1, -1.0, 0.2]); hi = np.array([0.4, 0.7, -0.2, 0.9])
    t, w = gl_nodes(n, 48)
    mass, Mf, Mt = cell_moments(lo[None], hi[None], t, w)
    X = rng.standard_normal((4_000_000, n))
    fr = [i for i in range(n) if i != k]
    tt = s * X[:, k]
    Y = X[:, fr] / np.where(tt > 0, tt, np.nan)[:, None]
    inside = (tt > 0) & np.all((Y >= lo) & (Y <= hi), axis=1)
    pm = inside.mean()
    se = math.sqrt(pm * (1 - pm) / len(X))
    assert abs(pm - mass[0]) < 5 * se, (pm, mass[0], se)
    mf = (X[:, fr] * inside[:, None]).mean(0)
    mt = (tt * inside).mean()
    assert np.max(np.abs(mf - Mf[0])) < 5e-3 and abs(mt - Mt[0]) < 5e-3, (mf, Mf, mt, Mt)
    print('moments vs MC ok', pm, mass[0])


def test_one_layer_exact():
    # L = 1: E relu(w.x) = |w| / sqrt(2 pi) per output. Cells containing a kink never become exactly stable
    # (box cells cannot align with the kink hyperplanes), so the gap shrinks with refinement but never closes.
    for n in (2, 3, 4):
        W = make_weights(7, n, 1)
        bb = CertifiedBnB(W)
        bb.run(max_passes=1e9, max_leaves=4000)
        st = bb.state()
        exact = np.linalg.norm(W[0], axis=1) / math.sqrt(2 * math.pi)
        hw = np.max(st['ub'] - st['lb']) / 2
        perr = np.max(np.abs(st['point'] - exact))
        assert np.all(st['lb'] <= exact + 1e-10) and np.all(st['ub'] >= exact - 1e-10)
        assert perr <= hw + 1e-12, (perr, hw)
        print(f'one layer n={n}: {st["leaves"]} leaves, {st["passes"]:.0f} passes, exact mass '
              f'{st["mass_exact"]:.6f}, half-width {hw:.2e}, point error {perr:.2e}')


def test_bounds_contain_truth():
    rng = np.random.default_rng(3)
    for (n, L) in ((3, 4), (4, 8), (6, 4)):
        net = Net(11, n, L)
        bb = CertifiedBnB(net.W)
        out = bb.run(max_passes=2e5)
        st = out[-1]
        U = sphere_from_gauss(rng.standard_normal((1 << 20, n)))
        g = net.forward(U, noise=False)
        mc = mean_radius(n) * g.mean(0)
        se = mean_radius(n) * g.std(0) / math.sqrt(len(U))
        assert np.all(st['lb'] <= mc + 5 * se) and np.all(st['ub'] >= mc - 5 * se), (st['lb'], mc, st['ub'])
        err = np.max(np.abs(st['point'] - mc) / (se + 1e-12))
        print(f'n={n} L={L}: leaves {st["leaves"]}, exact mass {st["mass_exact"]:.4f}, '
              f'half-width max {np.max(st["ub"] - st["lb"]) / 2:.2e}, point vs MC {err:.1f} SE')


def test_cell_bounds_pointwise():
    """Sample points in random leaves and check l(x) <= f(x) <= u(x) via the stored integrated bounds is not
    possible pointwise; instead check the symbolic bounds directly on one batch."""
    rng = np.random.default_rng(4)
    n, L = 4, 6
    net = Net(5, n, L)
    bb = CertifiedBnB(net.W)
    m = n - 1
    for trial in range(50):
        k = rng.integers(n); s = rng.choice([-1, 1])
        c = rng.uniform(-1, 1, m); wd = 10 ** rng.uniform(-3, -0.5, m)
        lo = np.clip(c - wd, -1, 1); hi = np.clip(c + wd, -1, 1)
        Hhi, Hlo, exact, smear, fl = bb._bounds(np.array([k]), np.array([s]), lo[None], hi[None])
        Y = rng.uniform(lo, hi, size=(2000, m))
        X = np.zeros((2000, n)); fr = [i for i in range(n) if i != k]
        X[:, fr] = Y; X[:, k] = s
        f = net.forward(X, noise=False)
        Ya = np.concatenate([Y, np.ones((2000, 1))], axis=1)
        up = Ya @ Hhi[0].T; dn = Ya @ Hlo[0].T
        assert np.all(f <= up + 1e-9) and np.all(f >= dn - 1e-9), (np.max(f - up), np.max(dn - f))
        if exact[0]:
            assert np.max(np.abs(up - f)) < 1e-9
    print('pointwise symbolic bounds ok')


if __name__ == '__main__':
    test_moments_partition()
    test_moments_quadrature()
    test_moments_mc()
    test_cell_bounds_pointwise()
    test_one_layer_exact()
    test_bounds_contain_truth()
    print('ALL OK')
