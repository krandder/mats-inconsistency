"""Fuss-Catalan FC_N: law of squared singular values of a product of N iid Ginibre (entries var 1/n), n -> inf.
Parametrisation (Haagerup-Moller): x(p) = sin((N+1)p)^(N+1) / (sin p sin(Np)^N), p in (0, pi/(N+1)), x decreasing,
density rho(x(p)) = sin(p)^2 sin(Np)^(N-1) / (pi sin((N+1)p)^N). Checked (scratch/fc_check.py): mass 1, mean 1,
second moment N+1."""
import math
from scipy.integrate import quad


def x_of(p, N):
    return math.exp((N + 1) * math.log(math.sin((N + 1) * p)) - math.log(math.sin(p)) - N * math.log(math.sin(N * p)))


def dlogx_dp(p, N):
    return (N + 1) ** 2 / math.tan((N + 1) * p) - 1 / math.tan(p) - N * N / math.tan(N * p)


def rho(p, N):
    return math.exp(2 * math.log(math.sin(p)) + (N - 1) * math.log(math.sin(N * p))
                    - N * math.log(math.sin((N + 1) * p))) / math.pi


def edge(N):
    return math.exp((N + 1) * math.log(N + 1) - N * math.log(N))


def p_at(y, N):
    lo, hi = 1e-14, math.pi / (N + 1) - 1e-14
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if x_of(mid, N) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tail_mass(y, N):
    """FC_N mass above y."""
    if y >= edge(N):
        return 0.0
    py = p_at(y, N)
    f = lambda p: rho(p, N) * x_of(p, N) * abs(dlogx_dp(p, N))
    return quad(f, 1e-12, py, limit=500)[0]


def quantile_from_top(frac, N):
    """y such that the FC_N mass above y equals frac."""
    llo, lhi = math.log(1e-30), math.log(edge(N))
    for _ in range(100):
        mid = 0.5 * (llo + lhi)
        if tail_mass(math.exp(mid), N) > frac:
            llo = mid
        else:
            lhi = mid
    return math.exp(0.5 * (llo + lhi))
