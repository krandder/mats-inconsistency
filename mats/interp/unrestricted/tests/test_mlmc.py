"""Tests for lib/mlmc.py: unbiasedness against a radius-exact truth, and the level identity. Run: python3 tests/test_mlmc.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, mean_radius, sobol_directions  # noqa: E402
from mlmc import ReferencePath, level_samples, pilot_level_stats, allocate, mlmc_estimate  # noqa: E402


def test_telescoping_identity():
    # On one sample, G_0 + sum of level samples equals Y exactly (same U in every level).
    rng = np.random.default_rng(0)
    net = Net(9101, 16, 32)
    ref = ReferencePath(net.W, rng)
    levels = [0, 2, 8]
    U = rng.standard_normal((5, 16)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    tot = ref.G0()[None, :] + sum(level_samples(net.W, ref, levels, j, U) for j in range(1, len(levels) + 1))
    h = U
    for Wl in net.W:
        h = np.maximum(h @ Wl.T, 0.0)
    Y = mean_radius(16) * h
    assert np.max(np.abs(tot - Y)) < 1e-10 * max(1.0, np.max(np.abs(Y)))
    print('telescoping identity ok')


def test_unbiased():
    # Average of independent MLMC estimates matches a high-precision RQMC truth within its standard error.
    rng = np.random.default_rng(1)
    n, L = 16, 64
    net = Net(9102, n, L)
    U = sobol_directions(n, 18, scramble=True, seed=3)
    h = U
    for Wl in net.W:
        h = np.maximum(h @ Wl.T, 0.0)
    truth = mean_radius(n) * h.mean(0)
    reps = []
    for r in range(40):
        ref = ReferencePath(net.W, rng)
        levels = [0, 16]
        est, cost = mlmc_estimate(net.W, ref, levels, [2048, 256], rng)
        reps.append(est)
    reps = np.array(reps)
    m = reps.mean(0); se = reps.std(0, ddof=1) / np.sqrt(len(reps))
    z = np.max(np.abs(m - truth) / (se + 1e-15))
    assert z < 5, z
    print(f'unbiased ok (max |z| = {z:.2f} over {n} outputs, cost per estimate {cost:.0f} passes)')


def test_allocate_budget():
    from mlmc import allocate
    for V, C, B in (([1.0, 0.0], [0.5, 1.0], 10), ([2.0, 0.3, 0.01], [0.1, 0.4, 1.0], 37.5), ([1.0], [1.0], 3)):
        N = allocate(V, C, B)
        assert sum(n * c for n, c in zip(N, C)) <= B + 1e-9, (V, C, B, N)
        assert all(n >= 1 for n in N)
    try:
        allocate([1.0, 1.0], [0.5, 1.0], 1.0)
        raise AssertionError('infeasible budget accepted')
    except ValueError:
        pass
    print('allocate respects the budget')


if __name__ == '__main__':
    test_allocate_budget()
    test_telescoping_identity()
    test_unbiased()
    print('ALL OK')
