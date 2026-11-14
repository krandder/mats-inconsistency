# M9(a) worker 1: pre-registration (written 2026-09-28 ~09:05Z, before any fit)

**Question.** Does a lottery model, Brier_f(x) = B0 − Δ_f·(1 − (1 − π_f)^x) with x = K/Q, describe the
soup's per-family Brier (logical-soup `runs/phase3`), and what K/Q would LFSR and CA need?

**Data (read-only).** `../../../../logical-soup/runs/phase3/k{12,14,16}-f{8,10,12}-s{1,2,3}/metrics.csv`,
7 base cells × 3 seeds = 21 runs (K/Q ∈ {1,4,16,64}; K/Q=16 at three K). Per run: mean of the last
50 epochs of `brier_<family>` / 1e6 (the same statistic as REPORT-SOUP.md). Variants (boot, r1,
heldpi) excluded from the fit; reported only as out-of-sample K/Q=4 checks.
Families: periodic, prime, and, or, lfsr, ca30, ca110, pi.

**Arms (models, per family).**
- L2: lottery, B0 = 0.25 fixed (the LMSR prior price 0.5), free Δ_f (sign free), π_f ∈ (0,1).
- L3: lottery, B0 free (sensitivity).
- LOG: Brier = a + b·ln x (2 params, the no-saturation alternative).
- LK / LQ: lottery L2 with x = K/4096 or x = 4096/Q (tests whether K/Q, not K or Q, is the variable).
- NULL: constant mean.

**Metric.** R² over the 21 runs (pooled), and per-seed fits (7 points each) → π_f mean ± sd over
the 3 seeds. Model comparison by AIC (Gaussian residuals). 1000 bootstrap resamples of runs for
a 95% CI on π_f.

**Controls.** (1) Shuffled soup table: permute the K/Q labels over the 21 runs (1000 perms),
refit L2, report the null R² distribution and p. (2) Families that were never priced (lfsr, ca110,
pi) act as negative controls: L2 R² should be ≈0.

**Prediction for LFSR / CA.** (i) Empirical bound: assume the achievable gain Δ*_f from a stream
simulation (best lookup predictor on k ≤ 4 history ages, `streams_check.py`), take the one-sided
95% bound on the Brier gain at K/Q=64 and invert to an upper bound on π_f, hence a lower bound on
the K/Q needed for half the achievable gain. (ii) Description-length extrapolation (a model
assumption, labelled): π_f ≈ π_periodic · 32^−(L_f − L_periodic), L = number of constrained
bytes in the shortest hand-written VM program.

**Stop rule.** From STUDY-PLAN M9 row: if L2 does not beat NULL and LOG by AIC on the priced
families (periodic, prime, and) and the shuffled-table p ≥ 0.05, report the lottery law as absent
for the soup. Otherwise report π_f and the predictions. CPU only, one process, < 30 min.

**Seeds.** The soup's own 3 seeds (1,2,3) per cell; no new soup runs (out of scope per plan §5).
