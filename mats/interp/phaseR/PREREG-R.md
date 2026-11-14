# Phase R pre-registration: the noisy-channel program on random, untrained networks (written 2026-10-28 ~19:10Z)

Written before any network experiment of this phase. Already done when this was written: reading the agreed program
(`outbox/cm-interp-program-2026-10-28.md`), the public page, phase B's code and logs, ARC's blog post and arXiv 2605.05179,
and one 1-D calculation (`explore/theory1d.py`, decoupled per-neuron model, no network chain) whose numbers are quoted
below as the basis of the predictions.

## Setup (exact)
Network R(d, L, act, g, sigma, seed): h_0 = x ~ N(0, I_d) (input width = d, as ARC). For l = 1..L:
z_l = W_l h_{l-1}, W_l iid N(0, g^2/d), no biases (as ARC); s_l = LN(act(z_l)), LN(a) = sqrt(d)(a - mean a)/|a - mean a|
(no affine); h_l = s_l + sigma eps_l, eps_l ~ N(0, I_d), noise kept on. Outputs y = W_out h_L, W_out iid N(0, 1/d).
Rare-event readout y_1 = w.h_L with w = W_out[0]/|W_out[0]|; events A_t = {y_1 > t}.
- act = tanh with gain g in {0.25, 0.75, 3} (ordered / critical / chaotic; chaos factor chi = g^2 E[act'(z)^2]/Var(act(z))
  = 0.92 / 1.01 / 1.99 at sigma = 0.3); act = ReLU with g = sqrt 2 (ARC's He init; with LN and no bias the gain drops out,
  chi = 1.35 at sigma = 0.3).
- sigma in {0.1, 0.3, 1.0} (1/2 log2(1 + 1/sigma^2) = 3.3 / 1.8 / 0.5 bits per dimension); width d in {8, ..., 256};
  depth L up to 16.

## Forward chain (task 1)
q_0 = N(0, I_d) = the input distribution exactly. Layer l: K q_{l-1} is sampled exactly (draw from q_{l-1}, apply the
noiseless layer, add noise); q_l is fitted to K q_{l-1} from 2^15-2^16 such draws (draws of the description, never of the
real activations p_l for l >= 2). Families:
- G: moment-matched Gaussian (the K = 2 description; ARC's cumulant propagation at K = 2 is its deductive version);
- NC (neuron code, singletons): product over neurons of the exact per-neuron marginals (1-D Gaussian mixtures);
- GC (neuron code with one Gaussian dependence group): Gaussian copula over the NC marginals;
- KN: know-nothing reference, uniform on the LN sphere convolved with N(0, sigma^2 I);
- defensive version of each: (1 - delta) q + delta N(mu, 4 Sigma), delta = 0.01;
- (optional, d <= 32) MIX16: 16-cell Gaussian mixture.
Scorecard per layer, in bits: coverage cost c_l = D_inf(K q_{l-1} || q_l); KL(K q_{l-1} || q_l) (a lower bound on c_l);
waste w_l = KL(q_l || K q_{l-1}) and D_inf(q_l || K q_{l-1}); compute. Estimators: (i) importance sampling for
log K q(h) (Laplace proposal) where its effective sample size allows (expected d <= 32); (ii) Donsker-Varadhan lower bounds
with per-neuron and Gaussian test functions, valid at any width up to Monte Carlo error; (iii) for c_l, gradient search
(a lower estimate) and, separately, a rigorous upper bound (Gaussian envelope, valid for any q_{l-1}). Which estimator
produced each number is stated next to it.

## Rare events (tasks 2 and 3)
True p(A_t): exact-weight sequential Monte Carlo over the noisy layers (unbiased), cross-checked between two
independent estimators and against brute force where p >= 2^-20. Certified statement: p_L(A) <= 2^eps q_L(A),
eps = sum_l c_l; also the variant with the last layer exact, p_L(A) <= 2^(c_1 + ... + c_{L-1}) (K q_{L-1})(A).
Bits of rarity lost = log2(bound / true p). Sampling at equal compute certifies about -log2(3/N) bits (zero hits,
95% upper bound). Estimators compared (task 3): plain MC; K = 2 propagation (ARC-style, deductive); our chain;
twisted SMC with probit-linear backward functions psi_l from the chain (anchored at the forward means, "bulk", or at the
most likely path to A, "tail"); the same SMC with psi_l learned by regression on simulated particles (iterated
auxiliary particle filter, black-box control); mean-shift importance sampling at the most likely path. Compute in FLOPs
(counted analytically per method) and wall time. E[output] metric as ARC: per-output MSE normalised by the
single-sample variance, averaged over outputs and seeds.

## Basis: decoupled per-neuron numbers (1-D, moment-matched Gaussian vs exact marginal; bits per neuron)
| act, g | sigma | KL(q||Kq) | KL(Kq||q) | D_inf(Kq||q) |
|---|---|---|---|---|
| tanh 0.25 | 0.3 | 0.007 | 0.005 | 0.15 |
| tanh 0.75 | 0.3 | 0.20 | 0.09 | 0.72 |
| tanh 3 | 0.3 | 0.85 | 0.44 | 1.32 |
| tanh 0.75 | 0.1 | 1.52 | 0.17 | 1.07 |
| tanh 0.75 | 1.0 | 0.006 | 0.006 | 0.16 |
| ReLU | 0.3 | 1.0 | 0.40 | unbounded (heavier tail than the matched Gaussian; capped only by LN) |
In the joint, the LN norm constraint stops all neurons from sitting at their marginal worst point at once, so the joint
D_inf per neuron should lie between the KL(Kq||q) and D_inf columns.

## Predictions
Task 1 (scorecard)
- R1. G: KL(Kq||q) and the waste KL(q||Kq) per layer grow linearly with width: value/d varies by less than 2x across
  d = 16..256 wherever the 1-D number above is >= 0.01 bits/neuron; per-neuron KL(Kq||q) within 2x of the table.
- R2. G: coverage c_l (search estimate where available, lower bound elsewhere) is at least 0.09 d bits per layer at
  sigma = 0.3, g = 0.75 (so > 20 bits per layer at d = 256); for ReLU it exceeds the tanh value at the same d by >= 2x.
- R3. NC: KL(Kq||q) per layer also grows linearly with d (it pays the total correlation of the layer, >= 0.05 bits per
  neuron at sigma = 0.3, g = 0.75): no better than G at d >= 64.
- R4. GC (neuron code with Gaussian dependence) is the best family: its KL waste per layer grows sublinearly
  (value at d = 32 <= 2x value at d = 8, against ~4x for G) and is below G's by >= 3x at d = 32, sigma = 0.3, g = 0.75.
- R5. KN: waste per layer ~ (capacity C - information carried) ~ order d bits; worst family for coverage of structure
  but its waste per neuron is bounded.
- R6. Depth: per-layer waste and coverage of G and GC are depth-independent within 1.5x over layers 2..L in the ordered
  and critical regimes; in the chaotic regime they are 3-5x the critical values and also stationary in depth (no
  blow-up, because LN caps the state); no family's per-layer cost grows with depth.
- R7. D_inf waste D_inf(q || K q) is infinite for every Gaussian-tailed family (coverage forces tails wider than the
  noise around a bounded image), so the program's bound D_inf(q_L||p_L) <= sum of D_inf wastes is vacuous; tightness
  must be measured event by event.
- R8. End-to-end 1-D output divergence KL(q_L^y || p_L^y) is stationary in depth beyond L = 4 (noisy layers forget), far
  below the sum of per-layer wastes.
Task 2 (certified rare-event bounds; L = 4, sigma = 0.3)
- R9. The rigorous (envelope) eps costs ~2.5 d bits per layer at sigma = 0.3: the rigorously certified bound is vacuous
  for every event with p >= 2^-60 at d >= 8.
- R10. With the estimated eps, the G-family bound loses >= 20 bits at every rarity for d >= 16 (vacuous for p >= 2^-20);
  the GC bound loses <= 15 bits at p = 2^-30 for d <= 32 and beats sampling at equal compute for p <= 2^-25.
- R11. Bits lost grow with rarity beyond ~2^-40: the matched Gaussian's output tail is heavier than the true tail
  (LN bounds y), so log2(q_L(A)/p(A)) >= 10 bits at p = 2^-60 for G at d <= 64.
Task 3 (estimation)
- R12. E[output]: K = 2 beats plain MC at equal FLOPs for d >= 64 at L <= 4 (tanh critical and ReLU, sigma = 0.3);
  at d = 16 it does not. Our chain adds nothing for averages (G = K = 2; sampled families cost as much as MC).
- R13. Rare events: K = 2's Gaussian tail is off by >= 1 bit for p <= 2^-20 at every width; twisted SMC with the
  tail-anchored chain psi reaches <= 30% relative RMSE at p = 2^-40 with <= 1e10 FLOPs at d = 64; bulk-anchored psi has
  >= 10x its variance at p <= 2^-30; the learned psi comes within 3x of tail-chain variance but costs more FLOPs
  including its pilot runs; plain mean-shift IS degrades faster with depth than twisted SMC.
- R14 (verdict-level). On random networks the paradigm's certified bounds are useful only with a neuron-code family
  and at small width; its strongest contribution is psi for guided sampling, which beats ARC-style cumulant estimates
  for p < 2^-20.

## Amendment 1 (2026-10-28 ~22:55Z): surrogate chains (coordinator addition)

Written after the phase-R results above; before any surrogate-chain run. Object: surrogate per-layer maps g_l run on the
same path with the same Gaussian noise; KL(real path || surrogate path) = sum_l E_real |f_l(h) - g_l(h)|^2 / (2 sigma^2)
(chain rule, exact), estimated by sampling real paths. Networks: as in E[output] (tanh 0.25 / 0.75 / 3 with bias N(0, 0.25),
ReLU He without bias), sigma = 0.3, d = 16 / 64 / 256, L = 2 / 4 / 8.
Surrogates: STAT, the statistical linearisation that K = 2 propagates implicitly (g = K=2 mean of s_l + E[ds/dh] (h - mu));
LR, its rank-d/8 truncation; QUAD, per-neuron degree-2 Hermite activation with mean-field LN; PL, per-neuron piecewise-linear
activation (hardtanh for tanh, exact ReLU) with mean-field LN; MFLN, exact activation with mean-field LN (isolates LN).
Estimators of E[y]: plain sampling; K = 2; SUR = exact surrogate mean (= K = 2's mean for STAT) + sampled correction
y_real - y_surrogate on coupled paths (same input and noise), unbiased. DV two-sided bounds on E[y_1] and on E[y_1^2 / 2] from
the STAT surrogate (exact Gaussian moment generating functions) and the sampled KL.
Predictions.
- S1. STAT per-layer KL grows linearly with width and is large outside the ordered regime: >= 0.5 nats per neuron per layer
  for critical tanh and ReLU at sigma = 0.3 (residual non-linearity at noise resolution); < 0.1 nats per neuron ordered.
- S2. QUAD cuts the critical-tanh KL by >= 3x relative to STAT; PL (hardtanh) is worse than QUAD; for ReLU, PL = MFLN and its
  KL (the LN mean-field error alone) stays O(1) per layer, not growing with width.
- S3. DV two-sided width on E[y_1] from STAT is wider than plain sampling's 95% CI at equal compute in every critical, chaotic
  and ReLU configuration; it is narrower only in the ordered regime, at small compute.
- S4. SUR's variance ratio rho = Var(y_real - y_sur)/Var(y_real) is < 0.1 ordered, 0.2-0.8 critical and ReLU, >= 0.8
  chaotic; SUR beats plain sampling at equal FLOPs wherever rho < 0.5, and beats K = 2 beyond N ~ rho / MSE_K2 samples (so at
  large compute it always beats K = 2, which has a fixed bias).
- S5. Skip-merge with noise sigma^2 (I + J J^T): the Gaussian check (moment-matched sampled two-step kernel vs the merged
  Gaussian, a lower bound on the true KL) is < 1 bit per pair ordered and > 5 bits critical and chaotic at d = 64; dropping
  the J J^T term costs more than the curvature error in every regime.

## Amendment 2 (2026-10-29 ~00:15Z): fixes after the GPT-6 review (disclosure; decision rules fixed before the reruns)

Written after the whole phase-R report and the review (/tmp/prrev/review.md, 2026-10-28 23:27Z), before any rerun below.
Nothing here is a new hypothesis about the networks; it fixes how the existing quantities are estimated and reported.
- Coverage lower bounds. KL(Kq||q) >= E_Kq[log q*/q] is reported as a one-sided 95% lower confidence bound from a
  bootstrap over sampled paths (2,000 resamples), not as a sample mean. D_inf(Kq||q) >= tau + log Kq(T > tau): tau is
  chosen on one half of the Kq samples and Kq(T > tau) is bounded below on the other half by the exact one-sided
  Clopper-Pearson bound; sums over layers use a Bonferroni split of the 5%. These are called lower confidence bounds
  (95%), not rigorous bounds. Eps for the rare-event bounds is measured on the same fitted chain whose (K q_3)(A) is used
  (re-created from its seed), never transferred from another fit. The envelope (an upper bound on c_l for the exact chain)
  is unchanged.
- Splitting truth. The pCN scale is frozen per level and move from an independent pilot (levels fixed as before); then
  one fixed-level run per repeat to the deepest threshold gives every level. Repeats: 32 at d <= 32 (N = 16,384), 24 at
  d = 64 (N = 16,384), 16 at d = 256 (N = 8,192), 10 moves per level. A level is marked provisional if any of: relative SE
  > 0.25; the two halves of the repeats differ by > 1 bit; the saved bulk-twisted-SMC cross-check differs by > 1 bit; the
  earlier published truth differs by > 1 bit. The chain's own estimates (K q_3)(A) are recomputed with the same frozen
  scheme.
- Copula tails. The copula sampler gets an exact tail inverse (bisection on the mixture CDF in log space beyond |u| = 6)
  instead of the bounded table; the defensive part of the copula chain's (K q_3)(A) uses latent splitting instead of crude
  Monte Carlo; the copula bound uses a coverage search on the same chain.
- E[output]: report the default K = 2 (with the LN mean correction) and the variant without it, side by side; subtract the
  truth's noise floor only in a separate column.
- Surrogate KL upper bound for DV: path-level bootstrap of the summed per-path KL (one-sided 95%), not independent layers.
- Weight seeds: layer-1 width scorecard and depth profiles get seeds 1 and 2 where affordable.
Expectations (not tests of the theory): the held-out D_inf bounds will be 1-5 bits per layer below the in-sample values;
the frozen-tuning truths will agree with the published ones within 1 bit at 2^-10 to 2^-30 and may move by several bits
at 2^-50 to 2^-60 for ReLU and chaotic tanh, so some of those levels will be provisional.
Note added 2026-10-29 00:41Z, before any scorecard or eps result was used: in the first d = 256 eps run the threshold chosen
on the selection half (its 5th-largest value) had no exceedance in the held-out half, so that layer's D_inf bound was 0.
The candidate thresholds are now scored with a pessimistic held-out count (the 10% Poisson quantile of the selection count),
which stops the rule from preferring thresholds with a handful of exceedances. That run is discarded
(results/prereview/epslb_d256_firstrule_discarded.json) and repeated with fresh samples. Since D_inf >= KL, the bootstrap
KL bound is also a lower bound on D_inf; per-layer eps bounds use the larger of the two, each at 5%/6 for a three-layer sum.
