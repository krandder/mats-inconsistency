"""Seed-count power calculation for item 3 (PREREG.md). Paired difference d = tau(R) - tau(R1) over n seeds,
two-sided t-CI at 97.5% (Bonferroni over R4 and R8), "Gain" needs lower bound > 0 and mean >= 0.05.
Power = P(lower bound > 0) under true mean DELTA, sd SIGMA (noncentral t). Also the expected 97.5% half-width."""
import math

from scipy.stats import nct, t as tdist

for sigma, delta in [(0.011, 0.05), (0.02, 0.05), (0.041, 0.05), (0.041, 0.10)]:
    row = []
    for n in range(3, 11):
        tc = tdist.ppf(1 - 0.025 / 2, n - 1)
        ncp = delta / (sigma / math.sqrt(n))
        row.append(f"n={n}: {1 - nct.cdf(tc, n - 1, ncp):.2f}")
    print(f"sigma={sigma} delta={delta}: " + ", ".join(row))
for sigma in (0.011, 0.041):
    print(f"sigma={sigma} expected 97.5% half-width: " +
          ", ".join(f"n={n}: {tdist.ppf(1 - 0.025 / 2, n - 1) * sigma / math.sqrt(n):.3f}" for n in range(3, 11)))
