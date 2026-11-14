## 5. What this means for M1 and M3

- **Narrow width does not remove the M1 difficulty. It moves it.**
  - In plain nets it moves into the first ~n ln(1/tolerance) layers (the synchronisation length). After that, depth
    is free: one weight-derived trajectory per radius bin is enough. Before it, the state is a net of depth
    proportional to width on Gaussian input, which is exactly M1's regime, now at width 8-16. No sample-free method
    we tried works there: K2 is off by nMSE 0.016-27, and exact activation cells number at least 1e5-1e6.
  - In residual nets nothing synchronises, but a sample-free Gaussian closure is depth-stable: its error is a floor
    that does not grow from depth 16 to 1024 [RD]. It beats the best sampler below ~100-1,000 paths and loses above.
    This is the closest this program has come to M1's "matches sampling at depths that grow" on any family, though
    only below about 1,000 paths and only at random initialisation.
- **What would move it.**
  1. A closure whose bias vanishes on the residual family: cumulants beyond K = 2, or Gaussian mixtures split along
     the few directions where the map is not quadratic. The floor is a bias, not variance, so this is the
     identifiable next step.
  2. For plain nets, a deductive method for the shallow block only. Its depth is ~n ln(1/tol), so at n = 8 it is
     ~16-30 layers, the regime where earlier lanes found no method either.
- **Trained nets (4.3): the closure fails, by 2-3 orders of magnitude.**
- **M3 (certified tails): not reached.** The closure gets the tails wrong by 12-76x. The useful structure for tails
  is positive homogeneity: the radius integral is exact, so tails cost 3-40x fewer samples. That gives no
  certificate.

## 6. Pre-registered predictions: scorecard

| prediction | outcome |
|---|---|
| P1 per path: N_surv(0.01) <= 2 at L' >= 32 (n=8) / 64 (n=16) | hit (n=8: 1.0; n=16: 1.32 at L'=64, 1.0 from 192) |
| P1 s2/s1 slope within 25% of -gap_1 at L' >= 64 | hit for n=16 (-0.089 to -0.098 vs -0.086); n=8 borderline (-0.266 vs -0.214 at L'=64; at L' >= 128 every path is rank 1) |
| P1 rank-1 fraction within 0.05 of binomial at L'=64 (n=8) | hit as an average (0.92 vs 0.90), but rank-1 events are shared by all inputs of a net after synchronisation, so 4 nets test this weakly |
| P2 angle < 0.05 rad by depth 64 (n=8) / 128 (n=16), sigma=0 | hit (mean 1.8e-5 and 2.1e-3) |
| P2 log-slope within 30% of -gap_1 | hit (n=16: -0.0867 vs -0.0864; n=8: -0.26 vs -0.214, 22%) |
| P2 noise floor 1e-3 to 1e-2 rad at sigma=1e-3 | hit (0.008-0.013) |
| P2 gate disagreement < 1% by 64 (n=8) / 128 (n=16) | hit (0 and 0.08%) |
| P2 dead fraction within 0.1 of 1-(1-2^-n)^L; bimodal per net | hit (0.19/0.41/0.63 vs 0.22/0.39/0.63; 20 / 5 / 39 nets below 5% / between / above 95%) |
| P3 k*_F(M;1e-2) <= 3 after sync, >= n/2 at cut 4; k*_F(S;1e-2) <= 2 | hit |
| P4 residual: no collapse, gates differ > 10% | hit (angles 1.3-1.6 rad, gates 41-50%) |
| P4 residual: N_surv(e^-1) >= n/2 at every L' | miss (falls to ~0.35 n at L' >= 192; N_surv(0.01) = n throughout) |
| P5 gate-free candidates capture <= 2k/n + 0.1 of tr S | miss as worded: after synchronisation every candidate captures ~all of tr S because the state is one-dimensional (Amendment 1). On the sensitivity share (tr M): lin/rand 0.04-0.36, about chance, hit except 0.36 at n=8 |
| P5 worst-case masks: PR >= n/2, sup-gain ratio >= 10 | n=16 hit (12-13 of 16; e^30-e^240); n=8 PR misses at deep cuts (2.2-3.8 < 4) |
| P5 avg / att capture >= 0.9 of tr S | hit (trivially, as above); on tr M avg is within 0.1 of the optimum on 5 of 6 configurations, att less reliable at n=8 |
| P6 deep part from the exact law at l0 reaches 1e-4 for l0 >= 48 (n=8) / 96 (n=16) | hit on the 5 nets whose input explains >= 96% of the output variance; miss on the 3 near-dead nets where noise carries 23-89% |
| P7 hybrid >= 0.5 L/l0 better than MC at L=256 | hit at 256-4,096 paths (5.6-11.6x), miss at 16k (1.6x); vs the best simple sampler only 2-4x |
| P8 K2 from the input misses by nMSE >= 1e-2 (plain); no sample-free method matches sampling at L=64, n=16 (plain) | hit (0.016-27); hit |
| P9 trained nets do not collapse; gates differ >= 10%; K2 loses to MC | residual sorter: hit / hit / hit (0.024 vs MC 1e-3 at 1,024 paths). RNNs: no collapse (hit), but gates settle to 3-7% (miss). [P9RNN] |
| P10 (Amendment 5) residual K2 nMSE ratio L=1024 / L=64 within 3x | [RD] |
| Trainability: T=32 >= 50% | miss (38-40%) |
| T=64 above 5x chance but below T=32 | first half hit, second half miss (best 44% vs 40%) |
| residual sorter nMSE <= 0.05 at L=64; plain fails at 64, better at 16 | hit; hit; hit |

Deviations: all in the six amendments.
- Plain n = 8 analyses are conditioned on survival.
- The P5 score was supplemented.
- Samplers were trimmed to R = 16, and only one plain-net sampler seed ran (Amendment 6 says two; one is correct).
- The MLP at L = 128 was stopped.
- K2's depth scaling, the linearity shares and the Imhof tail were added after the first residual results.

## 7. What is proven, measured, heuristic

- PROVEN (from Theorem C of the lyapunov lane, noiseless):
  - along one path, the active counts m_l are iid Bin(n, 1/2);
  - rank(J_x) <= min_l m_l;
  - the per-path death probability by depth L is 1 - (1 - 2^-n)^L;
  - after exact synchronisation all inputs share every later gate, so a net dies whole;
  - for two infinitesimally separated inputs, the log angle drifts at mu_1 - lambda_r = -gap_1 per layer. Corollary C1
    gives the exponents conditional on the chain keeping at least 2 active units. A layer with one active unit
    synchronises the two exactly.
- MEASURED: everything else in sections 2-4, on 4-64 nets per configuration as stated.
  - "Matches the theory" for the synchronisation rate compares the median angle of finite pair separations (layers
    32-128, 64 nets) with the infinitesimal exponent. Agreement within 0.5% at n = 16 is measured, not proven, for
    finite angles.
- HEURISTIC:
  - why K2's error does not grow with depth in 1/sqrt(L) residual nets (closure errors enter along independent random
    directions V_l and add like a random walk of L steps of size 1/sqrt(L));
  - that the residual nets' near-quadratic structure is what lets K2 work.

## 8. Review

[REVIEW]

## 9. Files

- `PREREG.md`, `prereg_hashes.txt`, `GPU-LOG.md`; `notes/refs.md` (verified literature, 77 entries).
- Code:
  - `lib/nd.py` (random nets, FLOP counts);
  - `lib/k2.py` (Gaussian closures: plain, residual with Stein cross-covariance, RNN);
  - `lib/gauss_relu.py` (copied unchanged from work/arc-noisy);
  - `lib/tnets.py` (trained nets);
  - `scratch/p2_*.py` (collapse, Jacobians, subspaces), `scratch/p3_*.py` (truth, oracle deep part, samplers, tails,
    cells, variance split, linearity, depth scaling), `scratch/p4_*.py` (trained nets);
  - `train/` and `gpu/` (trainers, launch).
- Results: `runs/*.json`, `runs/p3/*.json`, `runs/p4/*.json`; tables in `results/p2_tables.md`; logs in `logs/`.
- Trained weights: `train/gpu_out/*.npz` (finals and checkpoints).
