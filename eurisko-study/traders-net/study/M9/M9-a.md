# M9(a): the soup's K/Q law as a lottery (worker 1)

Checked at 2026-09-28T09:15Z. Scripts and outputs: `study/M9/w1/` (PREREG.md, fit_lottery.py,
streams_check.py, predict.py). Rerun: `cd study/M9/w1 && PYTHONPATH=/tmp/traders-net/M9/w1/pylib
python3 fit_lottery.py && python3 streams_check.py && python3 predict.py` (matplotlib only for the
PNG; ~20 s + ~10 min on one core). Data: 21 base runs of `logical-soup/runs/phase3` (7 cells × seeds
1-3; mean of the last 50 of 5000 epochs), listed in `w1/soup_runs.csv`.

## TL;DR

1. **The lottery model fits the priced families.** Brier = 0.25 − Δ·(1 − (1 − π)^{K/Q}) gives
   R² 0.94 (periodic), 0.92 (prime), 0.94 (A∧B), 0.79 (A∨B), and beats a log-linear law by
   ΔAIC 9-14 and a generic saturating (Michaelis-Menten) curve by ΔAIC 3-7. Shuffled soup table:
   p = 0.001 on all four (1000 permutations). K alone (R² 0.38-0.44) and Q alone (0.14-0.21) fit far
   worse, which confirms K/Q is the variable. A small residual K effect remains: adding c·ln K
   gives c = −0.007 to −0.009 per e-fold of K (ΔAIC −5 to −11), i.e. ~0.02 Brier over 16x K at fixed K/Q.
   [`w1/lottery_fit.csv`, `w1/lottery_fit.json`, `w1/lottery_fit.png`]
2. **Per-ticket success π** (pooled fit; per-seed log10 π mean ± sd; bootstrap 95% CI):
   periodic 0.066 (−1.18 ± 0.04; 0.044-0.078), prime 0.107 (−0.96 ± 0.08; 0.081-0.151),
   A∧B 0.117 (−0.92 ± 0.06; 0.095-0.161), A∨B 0.042 (−1.39 ± 0.2; 0.001-0.057). Half the fitted
   gain needs K/Q ≈ 10 (periodic), 6 (prime), 6 (A∧B), 16 (A∨B). Out-of-sample check on the three
   K/Q=4 ablation cells (boot, r1, heldpi, 9 runs): mean abs error 0.005-0.010 (A∨B 0.005; lead fix, was 0.007-0.010). [`w1/lottery_fit.csv`]
3. **Why LFSR, CA and pi are never priced: two different reasons.** A stream simulation of the soup's
   own families (`w1/streams_check.csv`; best lookup table on k history ages, fitted on 45 seeds,
   scored on 45 others):
   - **CA30 is unlearnable from the VM's view**, not a search failure: the next cell needs the left
     neighbour at age 32, one past the 32-bit history; 4 reads give 0.2497.
   - **LFSR is a 4-sparse parity** (exact recurrence at ages 10, 12, 13, 15): Brier 0.25 for every
     k ≤ 3, exactly 0 at k = 4. Selection gets no partial credit.
   - **CA110 needs 2 reads** (ages 30, 31: Brier 0.093), with only a 0.011 step at 1 read (0.239).
     This is the one genuine search failure among the unpriced families.
   - **pi** is a 64-digit table that repeats, so it is learnable from history (4 reads: 0.119), but
     only with a lookup table in the program.
4. **Predicted K/Q** (`w1/predictions.csv`). Empirical bound (one-sided 95% on the gain at K/Q=64,
   inverted through the lottery): CA110 needs K/Q ≥ 1.4k for half its achievable gain (π ≤ 5.1e-4,
   i.e. the second read costs ≥ 130x in π relative to periodic). A read-ladder model (assumption:
   π_f = π_periodic · r^{extra reads}, r = 2^-7 (bound), 2^-10, 2^-20 (4 random bytes at 1/32)) gives
   CA110 1.3k / 11k / 1.1e7 and LFSR 2.2e7 / 1.1e10 / 1.2e19. The LFSR bound is not informative:
   LFSR Brier *rises* with K/Q (Δ = −0.013, 0.2505 → 0.2632; spill from lineages that trade it
   anyway). **Falsifiable prediction for the soup lane's next run (REPORT-SOUP §6.1, K/Q = 256
   and 1024):** LFSR stays at ≥ 0.25; CA110 ends at 0.185-0.25 at K/Q = 1024 (0.24 at r = 2^-10).
5. **Contrast with a gradient net** (sibling M9(b), `M9/M9-b.md`): wealth-net prices LFSR and CA110
   by ~31k steps; the prequential net (S0 P4/P16) needs 10^9.0 (CA110) and 10^10.0-10.5 (LFSR) FLOPs per question [lead fix: wealth-net's own B=128 arm crosses at 10^9.95 / 10^10.83, `b/compute_to_brier_summary.csv`], versus the soup's maximum of 10^7.54 VI per
   question. The soup needs K/Q exponential in the number of reads that bring no partial credit;
   SGD on a k-sparse parity over n bits needs about n^{O(k)} iterations, and it makes hidden progress
   through a Fourier gap in the population gradient (Barak et al. 2022,
   https://arxiv.org/abs/2207.08799, abstract verified). With n = 32 and k = 4 that is compute,
   not population.

## Controls and stop rule

- Shuffled soup table (K/Q labels permuted over the 21 runs, 1000 draws): null R² 95th percentile
  0.16-0.17 against the observed 0.79-0.94; p = 0.001 for periodic, prime, A∧B, A∨B.
- Negative controls: CA30, CA110 and pi have R² 0.02-0.10 with shuffled p 0.09-0.41, so no law, as
  expected. LFSR fits a *harm* curve (R² 0.55, p = 0.001, Δ < 0).
- B0 fixed at 0.25 (the LMSR prior price 0.5) is harmless: with B0 free the fit gives 0.255-0.260
  and R² changes by ≤ 0.004 on the priced families.
- Prereg stop rule (L2 must beat NULL and LOG by AIC on periodic, prime and A∧B, with shuffled
  p < 0.05): passed, so the law is reported as present in the soup.

## Side finding (flag for the soup report)

REPORT-SOUP §3.1 says primality at 0.091 is "below what a base-rate trader can do (about 0.10)".
0.10 is the prime *density* for 16-bit n (6542/65536 = 0.0998). The base-rate *Brier* is
p(1 − p) = 0.0899. The soup's 0.091 ± 0.003 at K/Q = 64 is at the base rate, not beyond it.
The simulation's best 4-read table reaches 0.076 (`w1/streams_check.csv`), so structure exists
and was not found. (Computed by sieve in this session; `w1/streams_check.csv` k = 0 gives 0.0918
on finite windows.)

## Lottery-ticket link

- **Soup.** Each of the K/Q traders per question is an independent ticket. A question is priced
  once one ticket wins, P = 1 − (1 − π)^{K/Q}, and π falls geometrically with the number of program
  parts that must be right before any reward arrives. That is why density (K/Q) is the lever and
  fuel is not (REPORT-SOUP §3.6).
- **Nets, strong lottery tickets.** A random wide net contains a subnetwork that performs well with
  no training (Ramanujan et al., https://arxiv.org/abs/1911.13299; proof: Malach et al.,
  https://arxiv.org/abs/2002.00585). Width plays the role of K/Q.
  Pensia et al. (https://arxiv.org/abs/2006.07990) need only O(log(d·l)) extra width, because a
  ticket is assembled from many shared random pieces (subset-sum), not drawn whole. All three
  abstracts verified 2026-09-28.
- **The difference is shared computation.** Soup traders are monolithic, so the required density
  grows like 1/π, exponential in program length. Composable tickets make it logarithmic, and a
  gradient adds hidden partial credit even for parities.
- **Prediction for M9(c)** (worker 3's sweep): in a gradient market layer the K/Q collapse should be
  weak or absent at fixed total parameters. Any K/Q effect should saturate by K/Q of a few, not keep
  improving to 64. The original iterative-magnitude-pruning result is Frankle & Carbin,
  https://arxiv.org/abs/1803.03635.

## Unverified

- The read-ladder factor r is an assumption; only the CA110 bound (r ≤ 2^-7) is measured.
- π is an effective per-(trader, question) success over 5000 epochs of selection and mutation, not
  the probability that a random program wins. Each trader can emit on up to 8 questions, so the
  absolute scale of π is convention-dependent; ratios between families are what carry over.
- Periodic questions have periods 2-16 inside one family, so the family π averages over difficulties.
- The n^{O(k)} constant for LFSR in a net is not estimated here; M9(b) measured the net's actual
  cost.
