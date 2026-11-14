### 5.3 Rounds: broadcasting a consensus turns global functions into local statistics (T4)

One round (each source writes a message without seeing anything else; merge = sum): the message must fix the
log-likelihood on the whole region where the final posterior could lie, i.e. a global function of the open latents.
For location-type families its size is roughly the number of resolution cells of that region, (prior width / feature
width)^D times a polylog in N/delta, so it grows exponentially in the number D of open latents (**HEURISTIC**: it is
the eigenvalue count of a smooth translation-invariant kernel on a box, as in Landau-Slepian time-frequency limiting;
a reviewer expects standard n-width lower bounds to make it a theorem, not checked here; **MEASURED** below). In
words: one round must ship the likelihood everywhere it might matter; two rounds ship it only where it does.

Two rounds: round 1 sends a few additive numbers per source for a pilot; the root broadcasts it; round 2 sends the
C(D + p, p) - 1 non-constant coefficients of a degree-p expansion of each source's log-likelihood on a window of
radius about sqrt(log N / N) around the pilot; the root sums, truncates to the window and normalises. Inside the
window the per-source remainder is O(h^{p+1}), so the log-posterior error is about N (log N / N)^{(p+1)/2}, and the
KL, about half the variance of that error under the posterior, is about N^{1-p} up to logs. **PROVEN UNDER**: p + 1
bounded derivatives near the truth; posterior concentration, with mass outside the window o(1) (a global
identifiability / Bernstein-von Mises condition); a pilot inside the window with high probability (moderate-deviation
concentration at rate sqrt(log N / N)); and a unimodal posterior (label symmetry, as with exchangeable hidden groups,
must be broken first). The bound holds with high probability over the data. The pilot condition holds for the models
simulated here (the tiny-PFN GLS pilot; the N2 contamination pilot sum x / ((1 - eta) N)); for general mixtures,
moment pilots can converge more slowly than root-N (Wu and Yang, arXiv:1807.07237).
This is the classical one-step estimator (van der Vaart 1998, sec. 5.7); related distributed methods add one round
of communication (Huang and Huo, arXiv:1511.01443; Jordan, Lee and Yang, arXiv:1605.07689), with lower bounds for
protocols with and without broadcast (Duchi, Jordan, Wainwright, Zhang, arXiv:1405.0782). It is also Kelvin's design
for the aggregation track: agents write their messages after reading the consensus.

**MEASURED**:
- Tiny-PFN world, one open latent (N1e, globally normalised; two rounds = a 3-number GLS pilot from the labelled
  residuals, then p + 1 Chebyshev coefficients of each source's log-likelihood on a window of 10 pilot standard
  deviations; KL at the 95th percentile; "fails" = the window missed the exact posterior in more than 5% of episodes):

  | N | one round, 16 numbers | one round, 64 numbers | two rounds, 3 + 5 = 8 numbers | two rounds, 3 + 7 = 10 numbers |
  |---|---|---|---|---|
  | 64 | 0.36 | 7.0e-5 | fails (window missed the posterior in 10% of episodes; N1c's looser test: 1.8e-3) | fails (same; N1c: 7.6e-5) |
  | 1024 | 3.3 | 9.8e-5 | 8.6e-8 | 5.4e-11 |
  | 4096 | 14 | 1.0e-4 | 2.5e-10 | 1.7e-12 |

  At N >= 1024 the two-round error falls steeply with N at fixed size (about N^{-4} from 1024 to 4096 at p = 4,
  at least as fast as N^{1-p} predicts), while the one-round error stays put or creeps up; at a similar budget, one
  round is useless (16 numbers: KL 3.3 at N = 1024, against 8.6e-8 for 8 numbers over two rounds). Below N of about 64 the window is not local: two rounds fail
  at N = 3 in 89% of episodes and at N = 64 in 10% (N1c, with a looser window test, found 2.5% at N = 64). Rounds pay
  off once the posterior is narrow compared with the features of each source's likelihood.
- Small codes of the kind a network might learn (N1d; mean KL over 300 episodes, 100 at N = 1024):

  | N | one round, 2 numbers (GLS with E[1/v given residuals]) | two rounds, 2 + 2 numbers (pilot, then sums of l' and l'' at it; Laplace) | three rounds, 2 + 2 + 2 numbers |
  |---|---|---|---|
  | 8 | 0.13 | 1.2e-2 | 7.0e-3 |
  | 64 | 0.23 | 1.9e-4 | 8.6e-5 |
  | 256 | 0.20 | 1.1e-5 | 4.5e-6 |
  | 1024 | 0.16 | 4.9e-7 | 2.7e-7 |

  The 2-number one-round code has a floor that does not fall with N (it cannot use how far each reading sits from the
  consensus, which also signals reliability); 4 numbers over two rounds fall roughly like N^-2 here. Budgets differ
  (2 vs 4 numbers), but the equal-budget comparison in the table above points the same way. At N = 3 neither works
  (Laplace is poor for a wide posterior). A re-implementation of exact Bayes in the same script gives KL below 1e-15.
- D open latents (N2; sources of unknown reliability reporting on D shared latents: reliable with probability 0.7,
  N(theta, 0.25 I), otherwise N(0, 4 I)). Column 2 is the empirical linear epsilon-rank of 3,000 sampled
  log-likelihood functions on the box (RMS error), a property of the sample, not a lower bound for all encoders;
  columns 3-4 are tensor Chebyshev codes scored on a window around the exact posterior (approximation mass elsewhere
  ignored):

  | D | empirical one-round linear rank, RMS 1e-2 / 1e-3 | one round, tensor code, KL <= 1e-3 | two rounds, tensor code, KL <= 1e-3 |
  |---|---|---|---|
  | 1 | 12 / 19 | 24 (N = 512) | 7 (N = 512) |
  | 2 | 128 / 334 | 1,024 (N = 512) | 28 (N = 512) |
  | 3 | 746 / 1,646 (likely low: 3,000 samples) | 32,768 (N = 256) | 347 (N = 256) |

  The empirical one-round rank grows about tenfold per extra open latent. The two-round tensor codes are much smaller,
  though a tensor basis itself grows exponentially in D ((p + 1)^D); a total-degree basis would need
  C(D + 4, 4) + D + 1 = 7, 18, 39 numbers at degree 4 (count only, not run). Two rounds fail at N = 64 for D = 3 with
  degree <= 6 (window not yet local).

Reading: rounds, not hierarchy shape, are the main lever on message size once routing is solved. A transformer gets
rounds for free (each layer reads pooled context, then writes), which is one way depth can substitute for width.
