"""draw_queries, shared by e3b_loss.py (which defines the same function inline) and e4_trust.py."""
import numpy as np
from world import N


def draw_queries(X, rng):
    """For value vectors X (n, 3): a random variable i and the values of a random subset of the others."""
    n = len(X)
    qi = rng.integers(0, N, n)
    inc = rng.random((n, N)) < 0.5
    keys = [tuple(int(X[m, j]) if (inc[m, j] and j != qi[m]) else -1 for j in range(N)) for m in range(n)]
    return qi, keys, X[np.arange(n), qi]
