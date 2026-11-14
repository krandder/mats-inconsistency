"""Corrected synchronisation exponent (GPT-6 review item 4): radial and first angular exponents with the same
conditioning, m >= 2 (a layer with one active unit annihilates the angular perturbation; m = 0 kills the path):
gap_1 = (1/2) E[psi(m/2) - psi((m-1)/2) | m >= 2], m ~ Bin(n, 1/2)."""
import numpy as np
from scipy.special import digamma, comb
for n in (4, 8, 16, 32, 256):
    p = np.array([comb(n, m) / 2 ** n for m in range(n + 1)]); ms = np.arange(n + 1)
    sel = ms >= 2; pr = p[sel] / p[sel].sum()
    g = 0.5 * (pr * (digamma(ms[sel] / 2) - digamma((ms[sel] - 1) / 2))).sum()
    print(f"n={n}: gap_1 = {g:.5f} per layer (1/n = {1/n:.5f}); P(m<=1 per layer) = {p[:2].sum():.3e}")
