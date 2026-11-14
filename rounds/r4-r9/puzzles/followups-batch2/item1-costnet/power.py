"""Item 1 seed counts, written before any item-1 run. One-sided alpha 0.025 (a 95% two-sided t-CI lower bound
above the bar), power from the noncentral t distribution.

1a (P12 weighted mazes, CostNet+BB, P12 recipe): primary = per-seed pooled exact on the wide-maze cell (P12's
   PRIMARY sets test_W33+W49+W65 with L > 21). The only prior is P12 seed 0 at 27.0% (step 9000). Planning mean
   17% (a 10-point shrink for a lucky single seed), bar 8.06% (= best P12 DT arm 3.06% + 5 points), per-seed SD
   10 to 15 points (unknown; one seed). Chosen: n = 32 (power >= 0.90 at SD 15).
1b (Bansal mazes, CostNet+BB vs released DT checkpoint): primary = e2h_maze_59 exact, a +-5-point equivalence bar
   around DT. Planning: detect a 15-point gap with power >= 0.80 at SD 15. Chosen: n = 20.
"""
from scipy import stats
import numpy as np


def power(n, delta, sd, alpha=0.025):
    df = n - 1; nc = delta / (sd / np.sqrt(n)); tc = stats.t.ppf(1 - alpha, df)
    return 1 - stats.nct.cdf(tc, df, nc)


print("1a: mean 17 vs bar 8.06 (delta 8.94)")
for n in (8, 16, 24, 32, 40):
    print(f"  n={n:3d}  " + "  ".join(f"SD{sd}: {power(n, 17 - 8.06, sd):.3f}" for sd in (10, 12, 15, 18)))
print("  CI half-width at n=32: " + "  ".join(f"SD{sd}: {stats.t.ppf(0.975, 31) * sd / np.sqrt(32):.1f}" for sd in (10, 15)))
print("1b: gap 15 vs bar 5 (delta 10)")
for n in (8, 12, 16, 20, 24):
    print(f"  n={n:3d}  " + "  ".join(f"SD{sd}: {power(n, 10, sd):.3f}" for sd in (10, 12, 15)))
