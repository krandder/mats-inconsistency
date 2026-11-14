# Does a summary of half a deep ReLU net need exp(k*) numbers? Cut rank of critical random nets (30-31 October 2026)

Lane: cut-rank (theory). Bead: pool-ttdzrk.

Work directory `work/cut-rank/`:
- `FORMAL.md`: the statement, with an erratum appended.
- `PREREG.md`: the pre-registration; hashes in `prereg_hashes.txt`.
- `coprover/PROVENANCE.md`: what came from the co-prover.
- `notes/literature.md` and `notes/arc_moment.md`: notes.
- `lib/` and `scratch/`: code; `runs/`: results; `logs/`: logs; `review/`: reviews.

Co-prover: gpt-6-astra (three rounds, xhigh effort). Final adversarial review: GPT-6, in a fresh session
(`review/review_gpt6_final.md`). All 19 of its findings are addressed below (section 12). Labels: PROVEN,
PROVEN UNDER X, HEURISTIC, CONJECTURE, MEASURED.

## Summary for Kelvin

**Short answer.**
- **False as a law.** Your premise, made precise as "a front-half summary that is read out linearly needs about
  exp(c k*) numbers", is wrong. In wide nets at fixed depth and precision the count is provably polynomial in the width.
- **What is proven instead: an exponential in precision.** For any fixed architecture whose front has at least three
  layers, the count grows exactly like (1/eps)^{2(n-1)/3} as eps -> 0. That outgrows sampling's 1/eps^2 once the width
  n >= 5.
  - The proof rests on networks whose inputs have merged onto one direction.
  - In the genuine ensemble it rests on rare networks of that kind, so the constants are astronomically small.
- **Not a running-time bound.** It says nothing about ARC's nonlinear closures. If the fronts' laws are smooth in a
  stated sense, the count is o(1/eps^2) (proven).
- **The sampled counts are modest.** At widths 8-64 and eps = 0.1 they are 14-139 once each network's overall scale is
  divided out; the raw criterion gives 1-117.
  - The count falls as the back deepens.
  - It is roughly flat across widths at fixed depth/width.
  - It does not collapse when the inputs synchronise.
  - The apparent raw collapse came from rare high-energy networks that our samples missed.

1. **What is formalised.**
   - Cut the net into a random front and a random back.
   - A one-way linear summary sends s numbers computed from the front alone, and the back's answer is linear in them.
   - The smallest s that matches N = 1/eps^2 samples in ARC's averaged criterion is the eps-rank r(eps) of the cut
     operator.
   - Covered: moments read out linearly, harmonic or kernel truncations, one-cut meet-in-the-middle with a linear join.
   - Not covered: ARC's closures, sampling, two-way or joint methods. For those the question is running time.
2. **What role k* plays (PROVEN, with care).**
   - The back's weight law is rotation invariant. So, averaged over backs, the back enters only through a 1-D kernel
     on the sphere, and the front only through its spherical-harmonic moments.
   - The per-path orientations of the surviving directions disappear. k* itself (a count) need not, but k* alone does
     not determine the rank: fronts that differ only by scale give rank one, whatever the back's Jacobian
     (gpt-6-astra).
3. **The premise as stated is false (PROVEN, gpt-6-astra; I checked it).**
   - E_b||Dg v||^2 = ||v||^2 at any finite width, so r(eps) <= C(n+m, m) with m ~ 1/(eps^2 Vbar).
   - At fixed depth and precision that is polynomial in the width, while exp(c k*) with k* ~ n/L' is exponential in it.
4. **Where an exponential is real (PROVEN).** In every case below the exponent counts the dimensions in which the
   fronts vary, times log(1/eps). It is not k*.
   - Rays: one direction per front, uniform on the sphere, with an independent radial factor. With one back layer,
     r(eps) is about eps^{-2(n-1)/3} (my Theorem E, exact). Gpt-6-astra extended it to any fixed back depth, with
     matching upper bounds.
   - Genuine Gaussian fronts at a fixed architecture: the same exponent holds through a rare event. A layer with one
     all-positive row and all others all-negative merges every input onto one ray (found by the final GPT-6 review;
     I checked it). The constants are tiny, around 2^{-n^2}.
5. **Measured on real nets (n = 8-64, 17 configurations plus controls, pre-registered).** My eight predictions: P5-P8
   held; P1, P2 and P4 failed on the registered (raw) estimator; P3 cannot be scored validly.
   - The raw criterion is dominated by rare high-energy networks. Past synchronisation our samples had energy
     E a^2 = 1e-11 to 2e-3 against an ensemble value of 1, so the raw "collapse" (P3) does not describe the population.
   - With each network's scale divided out (post hoc):
     - synchronised fronts at width 8 need 54-57 numbers at eps = 0.1, more than the 22-46 inside the window;
     - very long backs need 14-20, so depth lowers the count without collapsing it;
     - at depth/width = 1/2 the count stays at 36-87 across widths.
   - From eps = 0.3 to 0.03 counts grow like eps^{-0.4..-1.5}, slower than 1/eps^2. The asymptotic exponent of item 4 is
     not visible at these precisions.
6. **Your crux lemma (measured, descriptive).**
   - On a k-dimensional slice through a cut state, the back's value function needs a number of components that fits
     C(k+D, k), the number of polynomials of degree <= D in k variables. The fit gives D = 3.2-5.4 at eps = 0.1 and
     5.4-14 at eps = 0.03.
   - That count is exponential in k only while k < D.
   - At k <= 4, the top relevant directions and random directions give similar relative counts. That does not show that
     relevance adds nothing.
7. **"Almost always for a fixed net".**
   - For each summary fixed in advance, "most nets defeat it" could be true, but it needs an unproven small-ball
     condition.
   - The reverse order, "on most nets every summary chosen after seeing the net fails", is impossible.
   - Measured: an in-sample rank-s approximation at half the eps = 0.1 rank misses 0.01 of the per-net variance on
     60-99% of sampled (front, back) pairs. That is descriptive, not the optimal protocol.
8. **For ARC's agenda.**
   - At fixed width, the dimension of any one-way linear summary eventually exceeds the Monte Carlo sample count as
     eps -> 0. This is a statement about summary size, not running time. It is not a failure of ARC's L^2-sketching
     guarantee, which uses a different normalisation.
   - Nothing here bounds ARC's nonlinear closures.
   - ARC's error law is an upper bound on error, so "the hierarchy loses to sampling" does not follow from it. The
     cost arithmetic in section 10 is illustrative only.
9. **Bottleneck conjecture.** What is still unproven is a useful precision window for the actual fronts: the
   eps-exponent must kick in at practical eps, uniformly as width and depth grow. That needs front averaging to keep
   enough harmonic energy, dispersed over many harmonics, at the needed degree (section 9, C1).

## 1. The question, formalised

**Nets.**
- Width n, depth L, cut after layer c, so L' = L - c layers remain.
- Input x ~ N(0, I_n). Layers: z_l = W_l h_{l-1}, h_l = relu(z_l), with W_l iid N(0, 2/n), no biases, noiseless.
- The front is W_1..W_c and the back is W_{c+1}..W_L, independent of each other.
- P_f is the law of the cut state h_c(x).
- Estimand: M_j(f,b) = E_x a_{L,j}. Per-net variance: V_j(f,b) = Var_x a_{L,j}. Vbar is their mean.

**Why the decoder must be restricted.**
- With arbitrary decoders the premise is false. Two short descriptions already work:
  - the front weights, n^2 c numbers;
  - N samples of the cut state, N n numbers, with error Var/N.
- Lipschitz decoders do no better than entropy numbers (Cohen-DeVore-Petrova-Wojtaszczyk 2022, Thm 3.3/4.1).
- So any lower bound on summary size above the parameter count needs a restricted decoder. Otherwise the question
  is one of running time.

**Protocol class and eps-rank.**
- The front computes phi(f) in R^s without seeing the back. The back computes psi_j(b) without seeing the front.
- The estimate is <phi(f), psi_j(b)>, with error Err = E(M_j - <phi, psi_j>)^2.
- r(eps) is the smallest s with Err <= eps^2 Vbar. This is ARC's averaged criterion at N = 1/eps^2.
- Fact 1 (Schmidt/Eckart-Young). r(eps) is the smallest s for which the singular-value tail of the cut operator T,
  sum_{i>s} sigma_i^2, is at most eps^2 Vbar.
- Lifting. A decoder that is a polynomial of degree p in s message numbers is a rank-C(s+p, p) linear protocol.

**Covered.**
- Moments read out linearly.
- Hermite, harmonic and kernel truncations.
- One-cut meet-in-the-middle with a linear join.

**Not covered.**
- ARC's closures and sampling.
- Adaptive or two-way protocols. Two-way does not reduce to one-way in general: the back could send its weights
  (gpt-6-astra). That extension is deferred, per Kelvin's ordering.
- Unrestricted computation. A running-time lower bound would need a complexity assumption.

**Known before this lane** (`notes/literature.md`; sources opened unless marked there).
- *Rank and communication.*
  - A c-bit protocol gives rank <= 2^c (Lee-Shraibman 2009, Thm 4.3).
  - Approximate rank is only a lower bound on communication; the log-approximate-rank conjecture was refuted
    (Chattopadhyay-Mande-Sherif 2019).
- *Widths.*
  - Linear widths of Sobolev balls decay like N^{-s/d}, and stable nonlinear widths match entropy numbers (CDPW 2022;
    Siegel 2024).
  - Linear methods cannot beat N^{-1/d} on ridge classes (Barron 1993, Thm 6).
  - The number of stably bounded features is at least the count of eigenvalues above threshold (Bach 2017b, Prop. 3).
- *Kernels on spheres.*
  - Arc-cosine kernels (Cho-Saul 2009).
  - Exact ReLU Gegenbauer coefficients (Bach 2017; Bietti-Mairal 2019).
  - An endpoint singularity (1-t)^nu gives eigenvalues ~ k^{-(d+2nu-1)}, i.e. k^{-(n+2)} for ReLU. For deep kernels the
    constant grows linearly with depth (Bietti-Bach 2021, Thm 1 and Cor. 2). Infinite width only.
- *High dimension.* Fitting degree-l polynomials needs about d^l features (Ghorbani-Mei-Misiakiewicz-Montanari 2021).
- *Integration.* Deterministic rules reach N^{-s/d}, randomised ones N^{-s/d-1/2} (Bakhvalov; Novak's survey). Smooth
  integration has a curse of dimensionality (Hinrichs-Novak-Ullrich-Wozniakowski 2014).
- *ARC.*
  - Wu et al. 2026 treat this ensemble. Polynomial activations give O(n/eps^2), and the regime eps -> 0 at fixed n is
    listed as open.
  - Hilton's halfspace note projects onto top eigenfunctions, with error relative to the head's L^2 norm.
- *Average-case tools.*
  - The low-degree method needs a planted-versus-null problem.
  - At infinite width a random ReLU net on the sphere is a mixed spherical p-spin field (b_p ~ p^{-5/2}). Free energies
    describe the disorder, not instance estimators.
- *Not found.* No paper bounds the rank across a depth cut of a network.

## 2. Exact structure (PROVEN)

**Theorem 1 (harmonic factorisation).** Found by me (`FORMAL.md` Lemma B) and independently by gpt-6-astra. The
normalisation was corrected by the final review.
- Rotation invariance (W_{c+1} Q has the same law as W_{c+1}) plus positive homogeneity give
  E_{b,j} g_j(h) g_j(h') = |h||h'| kappa(u.u') / n, where kappa(1) = 1 and kappa is positive definite at every finite
  width and depth.
- Write kappa = sum_l a_l P_l, with a_l >= 0 and sum_l a_l = 1.
- Define the front moments m_{l,a}(f) = E_x[(|h|/sqrt(n)) Y_{l,a}(h/|h|)], taken as 0 when h = 0.
- Then the sigma_i^2 of T are the eigenvalues of Lambda^{1/2} A Lambda^{1/2}, with Lambda = diag(a_l/N(n,l)) and
  A = E_f m(f) m(f)^T.

**Consequences.**
- The orientations of the per-path surviving directions disappear from the operator.
- Singular-value information can still enter through the shape of kappa. k* alone does not determine r(eps), but it is
  not proven irrelevant either.

## 3. The premise as stated is false (PROVEN, gpt-6-astra; checked here)

**Lemma 2.** E_b||Dg_b(h)v||^2 = ||v||^2 for the vector output at every nonzero h. My Monte Carlo check gives 0.97-1.03
+- 0.04 at three sizes.

**Theorem 3.**
- Truncating the back at harmonic degree m gives a rank-S(n,m) <= C(n+m, m) protocol with error at most
  Q (n-1)/((m+1)(m+n-1)), where Q = E||h_c||^2/n (Q = 1 for He fronts).
- At fixed n the precision exponent is therefore at most n-1.
- At fixed depth and eps, Vbar tends to a positive constant, so r(eps) = n^{O(1)}.
- Since k* ~ n G/(2L'), the universally quantified (T1), r >= exp(c0 k*), fails. It also fails along slowly diverging
  depths.
- This refutes (T1) as a law. It does not refute every narrower version of the premise, and it does not cap the rank at
  256 x 32 (the bound there is astronomically loose).

## 4. Where the exponential is real (PROVEN)

**Tools.**
- *Block lemma* (gpt-6-astra). Err_s >= tr H_B - s||H_B||_op for any block B of harmonics.
- *Depth ordering* (gpt-6-astra; re-derived by me). kappa_{n,d} = kappa_{n,d-1}/2 + R_{n,d}, with R PSD (relu = (z +
  |z|)/2; the cross terms vanish under w -> -w). So K_d - 2^{-(d-1)} K_1 is PSD, and Err_{d,opt}(s) >=
  2^{-(d-1)} Err_{1,opt}(s) for every rank s and every front ensemble.
- *Smooth fronts* (gpt-6-astra; I checked the step). If the radially weighted front laws have densities q_f with respect
  to one common angular measure and E_f||q_f||^2 < infinity, then T is a product of two Hilbert-Schmidt operators and
  r(eps) = o(eps^{-2}). Any eps-exponent above 2 therefore requires this condition to fail. Gaussian ReLU fronts do fail
  it: one-active-unit cones produce ray atoms, and the laws sit on coordinate faces.

**(a) Rays, one back layer (Theorem E, mine; the harmonic count was corrected by gpt-6-astra).**
- Assumptions: fronts h = R u, u uniform on S^{n-1}, R >= 0 independent of u, E R = 1, 0 < Var R = vR < infinity. (For
  mean mu, use vR/mu^2.)
- Result: r(eps) ~ [1/(n-1)!][C_n/(3 eps^2 vR)]^{(n-1)/3} as eps -> 0 at fixed n, with
  C_n = n 2^{n-1} Gamma(n/2)^2/(pi^2 Gamma(n-1)).
- Exact local exponents: 1.33, 1.99, 2.66, 3.33, 4.66, 5.99, 7.31, 9.96 at n = 3, 4, 5, 6, 8, 10, 12, 16, against
  2(n-1)/3 (`logs/ray_eps_scaling.log`).
- So it outgrows 1/eps^2 iff n >= 5.

**(b) Rays, any fixed finite depth d (Theorem E'; gpt-6-astra, round 3).**
- Result: r_d(eps) = Theta_{n,d}(eps^{-2(n-1)/3}).
- Lower bound: depth ordering plus (a), giving r_d >= c_n [2^{-(d-1)}/(eps^2 vR)]^{(n-1)/3}.
- Upper bound: each realised net is linear on finitely many cones, so the tail is O_{n,d}(D^{-3}); I checked this at
  the level of the argument.
- This is an eps -> 0 statement at fixed (n, d). It is not a uniform exp(Theta(n log(1/eps))) law as n and d grow.
- My own first version dropped the factor 2^{-(d-1)} and got only the exponent 2(n-3)/3.

**(c) Genuine Gaussian fronts at a fixed architecture (final GPT-6 review, finding 5; I checked it).**
- Take c >= 3. With positive probability, W_{c-1} has one entrywise-positive row and all other rows entrywise negative.
  Since h_{c-2} >= 0, this forces h_{c-1}(x) = a(x) e_1 for every input.
- If, in addition, the first column of W_c is positive (probability 2^{-n}), then h_c(x) = a(x) W_c e_1. That is a ray
  whose direction is uniform on the positive orthant and independent of its norm.
- Restricting the error to this event (probability p, about 2^{-n^2}) gives Err >= p x Err(conditional ray problem).
  Hence r(eps) = Omega_{n,c,d}(eps^{-2(n-1)/3}).
- The upper bound O(eps^{-2(n-1)/3}) follows from the cone argument and Jensen.
- Result: for the actual ensemble at every fixed (n, c >= 3, d), r(eps) = Theta(eps^{-2(n-1)/3}).
- The constants are astronomically small, so this says nothing about practical precision.

**(d) Simplified exponentials in k.**
- *Rays on the positive orthant of S^{k-1}, one back layer, eps^2 <= C k^{-3/2}* (gpt-6-astra; I checked the
  closed-form eigenvalues to 5e-8): r_+ >= exp((ln 2 - o(1)) k).
- *Cap fronts, my Lemma C* (corrected by gpt-6-astra; the exact spectrum is mu_l b_l^2). At eps = 0.1:
  - per-direction spread 0.03: r = 22, 217, 1,622, 46,479, 713,593, 46,983,832 at k = 2, 3, 4, 6, 8, 12;
  - spread 0.1: r = 11, 39, 112, 462, 1,527, 7,114.
  - Fixed spread and fixed eps give no exponential in k.
- *Exactly merged ReLU fronts with a fresh last layer, face by face* (gpt-6-astra B1/B4): log r >= (ln 2/2) n -
  (d-1) ln 2 - O(sqrt n) at budget ~ 2^{-(d-1)} n^{-3/2}. This needs face probabilities and radial weighting.
  Approximate merging needs an energy-weighted L^2 certificate.

## 5. Measurements on real nets (pre-registered; MEASURED)

**Design.**
- `PREREG.md` was hashed at 23:22:50Z, before any data. Amendment 1 (23:29Z) also came before data. Amendment 2 is post
  hoc.
- Fronts: 256-512 per configuration.
- Columns: C = 512 (back, neuron) pairs, or 256 for the deepest runs. They come from 8-64 backs with n neurons each;
  that is fewer independent backs than the brief asked for, because cost scales per back.
- Inputs: 8 independent splits of 1024 directions per front, with the radius integrated exactly for the means.
- Tail estimators:
  - r_in: in-sample, from a pair-averaged Gram that is unbiased given the sampled nets. Optimising the subspace makes it
    biased low in expectation.
  - r_ho: post hoc; held-out fronts with a learned column subspace. Its residual is biased high in expectation.
  - Neither is a confidence bound.
- Controls: two identical-front controls (true rank 1) give tails of -1.1e-4 to +2e-4 of Vbar. Reading these as a
  magnitude floor is post hoc, and two controls do not quantify uncertainty across configurations. eps = 0.01 is
  treated as unresolved.
- Synthetic-spectrum tests pass (`tests/test_spec.py`).

**Energy sampling (post hoc; review finding 1).**
- He nets satisfy E a^2 = 1 over the ensemble. Their energy is heavy tailed: E Q_L^2 = (1+2/n)(1+5/n)^L.
- With our sample sizes:
  - Window configurations (n = 8-64): sample E a^2 = 0.3-1.8; the energy-bearing backs number 4.6-26 of 8-64.
  - (8,8,8): the energy-bearing fronts number only 28 of 512.
  - The five synchronised configurations: sample E a^2 = 8e-12 to 2e-3. Only 2-16 fronts and 1-8 backs carry energy;
    in (8,64,64) one back carries 99.7%.
- So the raw ranks there describe a handful of sampled networks, not the population operator.
- **Post hoc remedy.** I also divide each front row by its RMS output and each back's columns by its RMS gain. This
  separable rescaling keeps the operator bilinear (`scratch/scale_norm.py`, `logs/scale_norm.log`).
  - It answers a per-network-scale version of the criterion, closer to ARC's per-net nMSE.
  - It is not the registered estimand.

| (n, c, L') | F | k* | raw r_in at eps 0.3/0.1/0.03 | raw r_ho(0.1) | scale-normalised r_in at 0.3/0.1/0.03 | scale-norm r_ho(0.1) | sample E a^2 |
|---|---|---|---|---|---|---|---|
| (8,2,8) | 512 | 1.8 | 3/9/27 | 10 | 4/22/55 | 24 | 1.7 |
| (8,4,4) | 512 | 2.6 | 6/27/84 | 36 | 8/37/114 | 49 | 1.2 |
| (8,8,8) | 512 | 1.7 | 4/15/43 | 22 | 11/44/94 | 55 | 0.82 |
| (8,8,8), F = 2048 (post hoc) | 2048 | 1.7 | 5/18/52 | 25 | 11/46/100 | 49 | |
| (8,64,8) merged front | 512 | 1.6 | 2/3/6 | 1 | 15/54/111 | 71 | 2e-3 |
| (8,64,8), F = 2048 (post hoc) | 2048 | 1.6 | 3/7/16 | 21 | 15/57/121 | 61 | |
| (8,8,64) merged back | 256 | 0.7 | 1/3/5 | 3 | 8/20/28 | 25 | 3e-7 |
| (8,64,64) both | 256 | 0.5 | 1/1/2 | 1 | 7/15/22 | 19 | 8e-12 |
| (16,8,2) | 512 | 5.9 | 28/117/245 | 257+ | 29/139/294 | 257+ | 1.1 |
| (16,8,4) | 512 | 5.5 | 18/89/200 | 242 | 25/119/248 | 232 | 1.3 |
| (16,8,8) | 512 | 4.1 | 15/60/137 | 123 | 24/87/189 | 140 | 0.58 |
| (16,8,16) | 512 | 2.5 | 7/16/38 | 21 | 16/37/75 | 42 | 0.94 |
| (16,8,32) | 512 | 1.9 | 4/9/17 | 10 | 15/31/40 | 33 | 0.28 |
| (16,128,16) merged front | 512 | 2.5 | 3/8/17 | 41 | 23/58/123 | 72 | 2e-4 |
| (16,16,128) merged back | 256 | 1.0 | 1/2/2 | 2 | 8/14/16 | 15 | 2e-5 |
| (32,8,8) | 256 | 8.3 | 20/70/145 | 129+ | 21/78/162 | 129+ | 1.2 |
| (32,16,16) | 256 | 5.3 | 5/23/62 | 39 | 16/51/115 | 91 | 1.8 |
| (64,8,8) | 256 | 16.9 | 36/108/187 | 129+ | 37/117/199 | 129+ | 0.55 |
| (64,32,32) | 256 | 6.6 | 8/30/74 | 60 | 9/36/93 | 54 | 0.29 |

Table notes:
- "+" means the held-out range was exhausted. That is not a measured value.
- k* is the mean N_surv(J; 0.01) over 64 paths.

**Pre-registered tests** (registered raw estimators; `scratch/score_prereg.py`).
- P1, back depth. r(L'=4)/r(L'=32) = 9.9 >= 4. This supports the premise's direction and refutes my prediction (<= 2).
  Scale-normalised (post hoc) the ratio is 3.8, which is inconclusive.
- P2, width at fixed depth/width. r = 27, 60, 23, 30 at n = 8, 16, 32, 64 (n = 32 over n = 8: 0.85). This supports the
  premise ("flat") and refutes mine (>= 4). Scale-normalised: 37, 87, 51, 36, also flat.
- P3, synchronisation. 4 of 5 raw ranks are <= 5, so "collapse" by the rule. But the raw measurement fails the
  energy-sampling check, so P3 cannot be scored validly. Scale-normalised (post hoc), nothing collapses: 14-58.
- P4, front depth (n = 8, L' = 8). Raw r = 9, 15, 3 at c = 2, 8, 64: refuted. Scale-normalised it is 22, 44, 54,
  monotone as I had predicted, but post hoc.
- P5, fixed back, 9.8x: supported. The readouts used unit-norm inputs (section 8).
- P6, first moments leave 8-9% of Vbar at n = 8: supported.
- P7, crux r(8)/r(4) = 2.23: supported (but see section 6).
- P8: 89% of the sampled pairs exceed the threshold: supported, as a descriptive statistic (section 8).

**Deviations from the pre-registration** (review finding 10).
- R2 thresholds use zero plus quantiles of the positive outputs, not unconditional quantiles.
- Moment regression fits and scores on split halves.
- The n = 16 crux has k <= 8 only; the registered list went to 16.
- (8,64,8) ran with F = 512 and C = 512, against Amendment 1's 256/256.
- The scale normalisation, the held-out tail, the F = 2048 runs, the ridge readout, the effective-degree fit and the
  kink measurement are all post hoc.

**Reading.**
- Ranks at eps = 0.1:
  - raw: 1-117;
  - scale-normalised: 14-139.
  These are fewer than the 100 samples they replace except at n = 16 with short backs (117-139) and n = 64 with
  c = L' = 8 (108-117).
- A power law in depth ratios fits log r(0.1) (raw, F' = 256) better than exp(c k*): R^2 0.95 against 0.56 over 17
  configurations, 0.92 against 0.58 inside the window.
- Secant exponents from eps = 0.3 to 0.03 are 0.4-1.5 raw and about 0.6-0.7 scale-normalised at F = 2048. These are
  slower than 1/eps^2. Theorem 4(c)'s exponent belongs to eps far below what we can resolve.

## 6. The crux lemma, measured (registered design; reading post hoc and descriptive)

**Setup.**
- For each of 256 backs, take a reference cut state and J = d a_L / d z_c.
- 1024 points z_0 + rho |z_0| V_k R w, with V_k the top-k right singular vectors of J.
- Control: random active directions.
- Results are in `runs/crux_*.json`.

r_loc(0.1) at rho = 0.2 (relevant / random):

| config (k*) | k = 1 | k = 2 | k = 3 | k = 4 | k = 6 | k = 8 |
|---|---|---|---|---|---|---|
| (16,8,8) (4.0) | 5/6 | 15/19 | 33/35 | 60/67 | 105/106 | 134/133 |
| (32,8,8) (8.4) | 5/6 | 15/20 | 42/54 | 88/98 | 200/197 | 288/270 |

- **Polynomial-count fit.** Inverting C(k+D, k) = r_loc gives:
  - at eps = 0.1: D = 3.2-5.4 for k <= 4;
  - at eps = 0.03: D = 5.4-14;
  - the ratio between the two falls from 2.2-2.8 at k = 1-2 to 1.6-1.7 at k = 4. The kink law predicts 2.23.

  This D is a descriptive parameter. Polynomial approximation was not validated directly, and no eps^{-2/3} law is
  established.
- **Relevant versus random directions** (review finding 13). At k <= 4 both kinds of slice give similar relative
  counts. At larger k the comparison is not dimension-matched: many reference Jacobians have rank below k, and the
  random control uses fewer than k directions when fewer coordinates are active. The two kinds of slice also have
  different variance normalisations, and the rotation was not exactly Haar (QR without the sign fix). So the data show
  no sign that relevance adds structure at k <= 4. They do not show that it adds none.
- **Radius sets the structure** (gpt-6-astra). A slice inside one activation cell is affine, so its rank is at most
  k+1 however strongly its directions survive.

## 7. The depth window and synchronisation (revised after the review)

**Long backs.**
- PROVEN in the iterated infinite-width limit (gpt-6-astra; checked): the rank-one protocol has
  Err_1/Vbar -> 2(c-1)/d. A long back makes different fronts look alike.
- At finite width (post hoc, scale-normalised), the rank falls from 139 to 31 as L' goes from 2 to 32 at n = 16, and is
  14-20 at L' = 64-128. It falls but does not collapse.

**Merged fronts.**
- Scale-normalised (post hoc), merged fronts have the largest ranks at n = 8: 54-57 at eps = 0.1, against 22-46 in the
  window.
- That matches the theory: rays are the extreme case of concentrated fronts, the case Theorem 4(c) rests on.
- The raw "collapse" is withdrawn as a population claim (finding 1).

**The window, for this formalisation.**
- No collapse past synchronisation is established.
- Back depth lowers the rank; front merging raises it.
- At fixed depth/width the rank is roughly flat in width, at the eps we can resolve.

**Kink measurement (post hoc; review finding 19).** This is a finite-angle statistic, not the asymptotic kink
coefficient. The relative variogram deficit at angle theta = 0.1, divided by theta:

| depth d | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| n = 16 | 0.23 | 0.38 | 0.75 | 1.17 | 2.3 | 4.8 |
| n = 32 | 0.24 | 0.46 | 0.82 | 1.45 | 2.3 | 2.6 |

- The infinite-width prediction at small angles is 0.21 d.
- For d >= 16, theta = 0.1 is outside the theta << 1/d regime, and the smaller-angle estimates are noisy.
- The deficit grows with depth up to d = 8. The data cannot separate causes of the rank trends.

## 8. Fixed networks

**Quantifiers** (gpt-6-astra; corrected by the review, finding 16).
- An ensemble lower bound says that, for each protocol fixed in advance, the per-net error Z has a large mean.
- Turning that into "most nets defeat it" needs more:
  - a second-moment bound on Z, which is not proven uniformly over protocols, for a positive fraction (Paley-Zygmund);
  - a small-ball bound on Z/V, normalised by the optimal rank-s error, for probability tending to one.
- "For every predetermined summary, most nets defeat it" may be true (conjecture C4).
- "On most nets, every summary chosen after seeing the net fails" is false.

**Measured (descriptive; finding 11).**
- The statistic is the in-sample rank-s SVD of split 0, with s set from r_in, scored by split pairs on the same nets.
- On that statistic, 60-99% of sampled (front, back) pairs have estimated error above 0.01 V(f,b).
- It is not the optimal protocol. It is not an independently fixed protocol. Thresholding noisy error estimates biases
  the fraction.

**kstar-frames lane.** Per-path frames stay aligned only while under about 2.5% of gates differ, i.e. angle about
0.08 rad, at any width. Our inputs are 0.1-0.9 rad apart within one front, so the frames rotate within a single front's
law.

**Fixed back with rich readouts (relabelled, finding 9).**
- The 512 half-space events were evaluated on unit-norm input directions, not Gaussian inputs. They are spherical-input
  experiments.
- One fixed back, rank at eps = 0.1:
  - 4-13 at n = 8 in the window;
  - 59-91 at n = 16 with L' <= 4, and 6-8 with L' >= 16;
  - 37-58 at n = 32-64;
  - 23 and 7 for the n = 8 merged-front configurations, and 36-103 (median 41.5) at n = 16.
- Nothing is proven linking the ensemble and fixed-back versions.

## 9. Conjecture ledger

Each conjecture names a fixed-(n,c,d) limit or a joint sequence. Finite experiments refute only quantitative versions.

**C1, ensemble one-way. THE BOTTLENECK: a useful precision window for the actual fronts.**
- Statement. Along a stated sequence (n, c, d, eps) inside the window, with eps polynomial in 1/n, a harmonic block B
  satisfies tr H_B >= 2 eps^2 Vbar and tr H_B/||H_B|| >= 2 g(eps), with eps^2 g(eps) -> infinity. This needs uniform
  constants, not the 2^{-n^2} of Theorem 4(c).
- Status: new. No equivalent named open problem was found.
- For:
  - the ray, face and rare-event theorems;
  - persistent gate kinks;
  - scale-normalised merged fronts having the highest measured ranks.
- Against:
  - smooth fronts give o(eps^{-2});
  - front averaging smooths;
  - measured exponents are 0.4-1.5 at resolvable eps;
  - ensemble energy is heavy tailed.
- Cheapest test. Use Theorem 1's factorisation at n = 8 up to degree 6 (2508 harmonics). Estimate the fronts' harmonic
  second moments A from at least 5000 fronts (front-only, cheap) and kappa's a_l from a variogram, then compute
  tr H_B and its spread per degree beyond the cap on fronts.

**C2, approximate merging transfers the ray theorem with usable constants.**
- Statement. ||M - M~||_2 = o(eps sqrt Vbar) for a ray surrogate M~, with the energy weighting built in.
- Status: new.
- For: angle 0 at n = 8, c = 64.
- Against: heavy-tailed energy; rare fronts.
- Test: the paired surrogate residual.

**C3, depth-uniform constants.**
- Statement. T_{n,d}(D_eps) >= c T_{n,1}(D_eps) at the needed cutoff, uniformly.
- Status: new at finite width. At infinite width it is related to Bietti-Bach Cor. 2.
- For: the gate-jump identity B3; the finite-angle deficit growing to d = 8.
- Against: at fixed resolution the kernel tends to a constant.
- Test: T_d(D)/T_1(D) with a declared onset D >> d.

**C4, quantifier (small ball).**
- Statement. P[Z/V <= t E_s/Vbar] <= h(t) + o(1) for each predetermined protocol family.
- Status: new.
- For: the descriptive 60-99%.
- Against: dead nets and norm intermittency.
- Test: held-out protocols on fresh nets.

**C5, fixed back with rich readouts.**
- Statement. With probability 1 - o(1) over backs, there is a block certificate for T_b under a stated readout law.
- Status: new.
- For: spherical-input event ranks of 36-103 where the raw ensemble collapses.
- Against: finitely many readouts cap the rank.
- Test: freeze backs, use Gaussian inputs, grow fronts and readouts.

**C6, local slices.**
- Statement. For a stated slice law and radius, there is a block certificate at k ~ k*.
- Status: new.
- For: counts that fit C(k+D, k).
- Against: a slice inside one cell is affine.

C1 is the step every route needs. Theorem 4(c) settled the exponent at fixed architecture, so what remains is
constants and uniformity.

## 10. ARC's moment class (task 4; `notes/arc_moment.md`)

**Illustrative arithmetic** (finding 2). Assume ARC's conjectured law nMSE <= c_K (L/n)^K were sharp, with controlled
K-dependent constants and 1 < L < n.
- Reaching RMSE eps then needs K ~ 2 ln(1/eps)/ln(n/L), at cost ~ L n (1/eps)^{2 ln n/ln(n/L)}, against sampling's
  2 L n^2 (1/eps)^2.
- The exponent exceeds 2 for every L > 1.
- Reading (L/n)^K as an RMS error instead gives the coordinator's "(1/eps)^{ln n/ln(n/L)}, losing when L > sqrt(n)".
- The law is an upper bound on error, not a necessary order, so none of this proves the hierarchy loses.

**Measured** (earlier lanes, at fixed L/n).
- K = 2's gain over Monte Carlo falls like n^{-0.56}, against the law's n^{-1}.
- Factorised K = 3 falls like n^{-1.58}, against n^{-2}.

**Linear readouts of cut moments** (finding 14).
- Least squares on Monte Carlo first moments leaves 8-9% of Vbar at n = 8 and 21-74% at n = 16-64, on held-out fronts.
- ARC's K = 2 closure has median nMSE 1.9e-2 to 4.7e-3 at n = 32-64 and depth 16, on other nets and with a different
  aggregation.
- The "15-70x" comparison is withdrawn. It is not a matched comparison of the same information, nets or metric.

## 11. What this proves, and does not, about ARC's agenda

**Proves.**
- r(eps) lower-bounds every one-way linear summary of a cut.
- For the actual He-ReLU ensemble at any fixed architecture with c >= 3, r(eps) = Theta(eps^{-2(n-1)/3}) as eps -> 0.
  At fixed width, linear-summary dimension therefore eventually exceeds the Monte Carlo sample count once n >= 5.
- exp(c k*) is not the law: the rank is polynomial in width at fixed depth and precision.
- An exponent above 2 needs fronts that fail the common-L^2-density condition.

**Does not prove.**
- That anything happens at practical precision: the constants are astronomically small.
- A running time bound, or anything about ARC's nonlinear closures or about sampling-free algorithms in general.
- Anything about ARC's L^2-sketching guarantee, which uses a different normalisation.
- The window behaviour of the population operator: our samples miss the rare high-energy nets.

## 12. Process and review

**Co-prover.** Three rounds with gpt-6-astra (`coprover/PROVENANCE.md`).
- From gpt-6-astra: the gradient identity, the truncation refutation, the ray and face bounds, the depth ordering, the
  any-depth Theorem E', the smooth-front theorem and the ledger. I checked each, numerically where possible.
- From me: the harmonic factorisation (found independently by both), the cap toy (its rank formula corrected by
  gpt-6-astra), Theorem E (its harmonic count corrected by gpt-6-astra), all measurements and the crux design.

**Final review** (GPT-6, fresh session, 19 findings; verdict "substantial revision"). It confirmed the factorisation
(after a normalisation fix), the gradient identity, the T1 refutation, the block lemma, the depth ordering and the ray
exponents. It also reproduced five ranks from the raw data, the scores and the R^2 values. Every finding was applied:
- 1, energy undersampling: raw synchronisation conclusions withdrawn; energy diagnostics added; post-hoc scale
  normalisation.
- 2, ARC law: labelled illustrative.
- 3, normalisation: fixed (also in the `FORMAL.md` erratum).
- 4, hypotheses: stated.
- 5, rare-event theorem: added as 4(c).
- 6, k* wording: corrected.
- 7, smooth-front wording: corrected.
- 8, bias versus coverage: corrected; the noise floor is post hoc.
- 9, spherical readouts: relabelled.
- 10, deviations: listed.
- 11, per-net statistic: re-described.
- 12, the D ranges: corrected and labelled descriptive.
- 13, the dimension-matching caveat: added.
- 14, "15-70x": withdrawn.
- 15, size versus time: stated.
- 16, quantifiers: fixed.
- 17, numbers: fixed.
- 18, `FORMAL.md` erratum and `toy_cap.py`: closed form, log regenerated.
- 19, kink: re-described as a finite-angle statistic.

**Compute.** CPU only, nice 19, at most 4 heavy processes. No GPU.

## Handoff

**Code.**
- `lib/crnet.py`: nets and Jacobians.
- `lib/spec.py`: pair Gram, cross-fitted and held-out tails, probe protocol, per-net errors.
- `scratch/run_cut.py`: simulation.
- `scratch/analyze.py`, `tabulate.py`, `score_prereg.py`, `scale_norm.py`: analysis and scoring.
- `scratch/crux_local.py`, `toy_cap.py`, `ray_eps_scaling.py`, `kink_depth.py`, `check_astra_r1.py`: crux, toys and
  checks.

**Test.** `python3 tests/test_spec.py`.

**Queues.** Run by `scratch/run_queue.sh`. The post-hoc F = 2048 identical-front control is still running; its absence
does not change any number above.

**Next if wanted.** The C1 harmonic-moment test at n = 8, and independent ensemble replications with rare-event
weighting for deep configurations.
