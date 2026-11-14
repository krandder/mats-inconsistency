# Can a gate be smooth enough for summaries to beat sampling and still learn? (31 October 2026)

Lane: tractable-gates (theory and measurements). Bead: pool-2wj2bh. Author: opus-mats-gates (Claude Opus 5.5).

Work directory `work/tractable-gates/`:
- `PREREG.md`: the pre-registration, with three amendments; hashes in `prereg_hashes.txt`.
- `notes/theory.md`: theory v2.1.
  - Co-prover check: `coprover/astra_r1_answer.md`.
  - Final review: `review/review_gpt6_final.md`.
- `lib/`, `scratch/`, `train/`, `gpu/`: code. `tests/`: tests (all pass).
- `runs/`: results. `logs/`: logs and tables.
- `logs/pareto.png`: the frontier figure. `GPU-LOG.md`: one L4, $1.10.

Labels: PROVEN, PROVEN UNDER (stated), HEURISTIC, CONJECTURE, COMPUTED (numerics of a stated model, convergence-checked
but not certified), MEASURED (finite-sample estimates). Post-hoc analyses are marked.

## Summary for Kelvin

**Short answer.**
- **Asymptotically, yes.** In the cut-rank sense smooth gates are more tractable as eps -> 0 (PROVEN, one back layer,
  under conditions on the input-scale distribution).
- **Where we can measure, gate smoothness is not the main factor we found.**
  - Summary sizes are associated with the network's phase (ordered, critical or chaotic).
  - The tested deductive estimator's standing depends on its implementation and cost, and on the phase.
- **Learning.** In the 45 trained sorting nets where we ran the tested closure (Gaussian closure K2), it never beat both
  MC and RQMC. That includes the gates that learn as well as ReLU: GELU, SiLU and sharp softplus. Their closures stay
  1.4-3.8x behind MC and 13-31x behind RQMC.
- **Open.** Other estimators, and summary sizes of trained nets, were not tested.
- **On random nets.** The closure wins at depth only for ordered gates whose outputs are nearly constant.

1. **Theory.**
   - PROVEN UNDER bounded-scale or tail conditions: for one back layer and any front law, a gate whose Hermite spectrum
     decays stretched-exponentially needs only polylog(1/eps) summary numbers at fixed width.
     - That covers softplus, GELU, SiLU and tanh.
     - SwiGLU and GEGLU have their gate's class.
     - ReLU needs eps^{-2(n-1)/3} on the lane's ray fronts.
   - Finite smoothness (ReLU^2) still loses at large enough width.
   - The conditions matter. With a heavy-tailed input scale, even Gaussian-smoothed ReLU can lose (the reviewer's
     counterexample).
2. **The constants are large** (COMPUTED). Model: each front a ray with an exponential radius; one back unit.
   - At width 8:
     - the softest gates beat sampling at every tested eps from 0.1 down to 1e-6: softplus at sharpness beta*s = 1.4,
       and ReLU with Gaussian noise of 0.7 of its input scale;
     - SiLU and SwiGLU win at some precisions and lose at others, and win at the finest tested (1e-4, 1e-6);
     - GELU wins only at 1e-6.
   - At width 16:
     - softplus, GELU, SiLU, SwiGLU and noisy ReLU lose at every resolved tested eps down to 1e-6. Unresolved:
       softplus16/8 at fine eps, softplus4 at 1e-6;
     - ReLU^2 wins at eps = 0.03 only;
     - the bilinear GLU's rank is exactly finite (136).
3. **Random nets at practical precision (MEASURED; finite-ensemble estimates).** At eps = 0.1 the in-sample ranks are
   9-93 against sampling's 100, as the lane found for ReLU. The held-out estimates, biased high, are 10-86, except biased
   tanh at 103-146. So most gates, ReLU included, need fewer summary numbers than sampling at this precision. Between
   gates, at width 16 (an association; phase was not isolated from splicing or scale):
   - chaotic GELU and SiLU need 2.5-4x fewer than ReLU;
   - the ordered softplus1 needs 2x more;
   - saturating tanh needs 2.5x more;
   - the sharp softplus16 is close to ReLU (33 vs 37 raw).

   At width 8 the raw ReLU rank is anomalously low (9; two backs carry 80% of the energy).
4. **Deductive estimation (Gaussian closure) against Monte Carlo (MC) and randomised QMC (RQMC)**, within the same
   network-evaluation budget.
   - As registered, ReLU was costed with its exact closed form, about 6x dearer than GELU's or SiLU's closure. Under
     that costing, smooth gates looked better at depth.
   - Post hoc, with ReLU on the same quadrature machinery and costing as the others, its closure beats both baselines on
     all 4 nets up to depth 8 at width 64 (2 of 4 at depth 16). That is as well as or better than GELU, SiLU and
     softplus4.
   - Equal machinery is not equal accuracy, so this is an implementation-specific observation. It does not show that
     smoothness buys much here.
   - By the registered rule (at least 3 of 4 nets), only soft ordered gates win at depth 16 at widths 16-32. The softest
     do so with outputs that are nearly input-independent: at depth 16, output variance is 1e-5 of its mean energy.
5. **Learnability (sorting MLP, 3 seeds; PFN Bayes task, 3 seeds on one L4).** Sorting nMSE relative to ReLU:
   - as good as ReLU: GELU 0.91, SiLU 0.97, softplus16 0.99, smooth step 1.00;
   - slightly worse: softplus4 1.16, noisy ReLU 1.17 (post hoc);
   - much worse: softplus1 1.89, tanh 1.87;
   - SwiGLU and bilinear diverge at initialisation without a pre-norm; with one, they reach 1.20x and 1.28x a pre-norm
     ReLU;
   - with per-seed gradient clipping (post hoc; the registered recipe shares one clipping norm across seeds), the ratios
     are GELU 0.88, softplus4 1.08, softplus1 1.66.

   On the PFN only softplus1 is clearly worse (2x GELU's KL); seed variance is 2-3x.
6. **Training sharpens some gates.**
   - On the PFN, softplus1's gate inputs grow 4-6x and softplus4's 2.2x against their measured initial scale (post hoc).
     On the registered statistic, softplus1's trained layer RMS is 2.05-2.72 x sqrt 2. This is the effect you worried
     about.
   - On sorting, measured growth from initialisation is modest: 1.2-1.6x for most gates, including ReLU (1.28). Tanh's
     inputs shrink (0.40).
7. **Trained sorting nets.**
   - The tested closure beat both MC and RQMC on 0 of 45 nets (free single-gate, pinned and noisy arms).
   - It beat MC alone on 6. One softplus1 net is a statistical tie with RQMC.
   - RQMC is more accurate than the closure at the same budget: 6-31x for the free gates, 12-92x for the pinned ones
     (medians).
8. **The frontier.**
   - With each gate's input scale pinned by normalisation (so training cannot sharpen it), a softplus ladder trades the
     two: the closure's error relative to MC falls from about 12x to 1.3x as beta*s goes from 23 to 1.4, while the
     sorting loss rises from 1.0x to 1.5x.
   - The trade is an overall trend with exceptions: softplus8 beats softplus16 on both axes.
   - No tested point both learns like ReLU and has a deductive win on a trained net.
   - The cut rank of trained nets was not measured, so whether a trained smooth net admits small summaries is open.
9. **Your 10M+ bar.**
   - These experiments (up to 65,536 weights in random nets; 8k-19k parameters trained) do not establish that changing
     the gate suffices at 10M+ scale.
   - The asymptotic theory says the smooth classes (including SwiGLU) have the better exponent. What sets the constants
     at scale is not known.

![frontier](../work/tractable-gates/logs/pareto.png)

## 1. The question and what "tractable" means here

"This sense" is the cut-rank lane's.
- Cut a net into a front and a back. A one-way linear summary of the front, read linearly by the back, needs r(eps)
  numbers to match N = 1/eps^2 samples. This is ARC's averaged criterion: error at most eps^2 Vbar, Vbar the mean per-net
  variance.
- The lane proved r(eps) ~ eps^{-2(n-1)/3} for ReLU at fixed width n, because ReLU's kinks put power-law tails into the
  back's value function. It also proved that smooth front laws give o(eps^-2).

Two readings are measured:
- **Summary size.** r(eps) on random nets, with the lane's estimators.
- **Running time.** The Gaussian closure K2 (mean and full covariance propagated layer by layer) against MC and RQMC.
  - Scoring follows ARC: nMSE = |est - truth|^2 / tr Var. MC's error is the expected 1/N_eq, where N_eq is the number of
    samples K2's FLOPs would buy.
  - RQMC runs at the largest power of 2 <= N_eq. That uses 51-99% of the budget, and input generation is not counted.
  - RQMC is scipy's scrambled Sobol (LMS plus digital shift), 16 replicates.
  - This is a network-evaluation FLOP comparison, not end-to-end runtime.

**Gates.**
- ReLU.
- softplus_beta(z) = log(1 + e^{beta z})/beta (beta -> infinity is ReLU).
- GELU and SiLU/Swish.
- tanh with biases. Bias-free tanh nets are odd, so E[output] = 0 exactly and the problem is degenerate (Amendment 1).
- gsrelu0.5: ReLU with integrated Gaussian pre-activation noise of std 0.5, i.e. 0.35 of the pre-activation scale.
- smoothstep1: a C-infinity gate with exact zeros, z S(z), S a smooth step from 0 (z <= -1) to 1 (z >= 1).
- SwiGLU (silu(W1 h) * (W2 h)) and the bilinear GLU.
- ReLU^2: theory only.

**Sharpness.** What matters is a gate's sharpness relative to the scale of its inputs: beta*s for softplus, s the RMS
pre-activation.
- Every random net keeps s = sqrt 2 in every layer, by a per-net, per-layer normalisation on a pilot sample. So
  softplus16, 4 and 1 have beta*s = 23, 5.7 and 1.4.
- In grid C, each back is normalised on its own front and then attached to other fronts. In the 64 sampled spliced
  pairs per gate at width 16, the gate inputs have RMS 0.45-3.6 at the first back layer and 0.06-8.5 at the last
  (logs/posthoc_review.log). So sharpness is fixed only in intact nets.

## 2. Theory (notes/theory.md v2.1)

Seven claims of v1 were wrong (gpt-6-astra). The final review corrected four more statements. All fixes are applied.

**A summary bound that needs nothing about the fronts (PROVEN UNDER measurability and finite second moments).**
- For any s functions in L^2 of the cut state's marginal law:
  - the front sends its integrals of them;
  - the back sends its projection coefficients;
  - the error is at most the back's mean squared distance to their span.
- Hence r(eps) <= C(n + D, n) whenever degree-D polynomials reach eps^2 Vbar.
- For one back unit phi(w.h) with Gaussian w, this is bounded by approximating the 1-D gate on its own pre-activation law,
  a Gaussian scale mixture. That holds for any front law, ReLU's faces and rays included.

**Hermite spectra of the gates (PROVEN; tests/test_acts.py).** c_k = E[phi(sZ) He_k(Z)]/sqrt(k!).
- *Power laws.* ReLU k^{-5/2}; ReLU^2 k^{-7/2}; a step (saturated tanh) k^{-3/2}.
- *Geometric.* ReLU plus Gaussian noise tau, and GELU: c_k^2 ~ rho^k, rho = s^2/(s^2 + tau^2) (tau = 1 for GELU).
- *Root-exponential.* softplus, SiLU and tanh.
  - The logistic law is a Gaussian scale mixture with scale 2K, K Kolmogorov-distributed. This gives exact coefficients.
  - For softplus, log c_k^2 = -2 pi sqrt(k)/(beta s) + O(log k).
  - Below degree about (beta s / 2 pi)^2, a softplus is indistinguishable from ReLU.
- *A C-infinity gate with exact zeros.* smoothstep1 is no smoother than ReLU up to degree ~16 and stays within 2-3x of
  it up to ~128.
- *GLUs.* An independent linear branch shifts the kernel's Taylor coefficients by one, so each GLU has its gate's class.
- *Bilinear.* A depth-d bilinear back is a degree-2^d polynomial, so its rank is exactly finite. Its variance map
  q -> A q^2 is unstable. That proof covers bilinear gates only; SwiGLU's divergence here is observed, not proven.

**Fixed width, eps -> 0.**
- *Stretched-exponential decay (PROVEN UNDER the scale conditions).* r(eps) = O(polylog(1/eps)) = o(eps^-2).
  - The conditions are bounded scale with an inverse moment, or tail and uniform-approximation bounds.
  - The final review's counterexample shows they are needed. With radius tail P(R > r) = r^{-5/2}, radial averaging turns
    the noisy ReLU's geometric decay into k^{-4} and the width-8 ray rank into eps^{-7/3}.
- *Power-class decay (PROVEN UNDER regular variation, on ray fronts).* c_k^2 ~ k^{-alpha} gives
  r ~ eps^{-(n-1)/(alpha-1)}, which loses once n > 2 alpha - 1: n >= 5 for ReLU, n >= 7 for ReLU^2.
- *Constants (COMPUTED).* Model: the lane's ray fronts with radius Gamma(1, 1), one back unit. Source:
  logs/ray_cert_vR1.log. Truncation K vs 2K and radial quadrature 24 vs 48 nodes agree within 2%, but nothing bounds the
  omitted energy; "-" = not converged.

| width 8, eps | 0.1 | 0.01 | 1e-4 | 1e-6 |
|---|---|---|---|---|
| sampling 1/eps^2 | 100 | 1e4 | 1e8 | 1e12 |
| ReLU (exact) | 184 | 1.6e6 | 3.3e15 | 7.1e24 |
| softplus16 | 162 | 5.2e5 | - | - |
| softplus4 | 45 | 5.0e4 | 2.4e9 | 2.3e12 |
| softplus1 | 42 | 1.7e3 | 5.7e6 | 1.75e9 |
| GELU | 110 | 4.5e4 | 2.8e8 | 6.2e10 |
| SiLU | 44 | 1.2e4 | 4.2e7 | 1.1e10 |
| SwiGLU | 122 | 8.1e3 | 5.8e7 | 1.9e10 |
| ReLU + noise 0.5 | 44 | 3.9e4 | 7.8e8 | 3.9e11 |
| ReLU + noise 1.0 | 43 | 6.1e3 | 3.4e7 | 9.7e9 |
| ReLU^2 | 53 | 796 | 1.3e8 | - |

- At width 16, softplus1 needs 144 / 3.0e5 / 1.6e12 / 2.2e17 and SiLU 311 / 8.6e6 / 1.2e14 / 1.1e19 at the same four
  eps. Both lose at every tested eps; so do GELU, SwiGLU and the noisy ReLUs.
- ReLU^2 wins at eps = 0.03 (935 against 1,111).
- The one-layer bilinear GLU's rank is n(n+1)/2: 36, 136 and 2,080 at widths 8, 16 and 64.
- *Fixed eps, growing width.* The polynomial bound grows like n^{D(eps)}. Smoothness lowers D(eps): at eps = 0.01 it is
  116 for ReLU and 14 for GELU (logs/hermite_classes.log, recomputed with direct tail sums after review finding 11). An
  upper bound above sampling does not show a loss.

**Depth (PROVEN for infinite-width kernels with 1 < R_1 < infinity, i.e. the geometric class: GELU, noisy ReLU).**
- The d-fold composition's radius is R_d = min(R_1, f^{-1}(R_{d-1})).
- It contracts geometrically in the chaotic phase: GELU at s = sqrt 2 has chi_1 = 1.06, and R_d - 1 = 0.50, 0.25, 0.063,
  0.0025 at depth 1, 4, 16, 64.
- It contracts like 1/d at criticality and stays above 1 in the ordered phase.
- The root-exponential gates already have radius 1, so these statements do not apply to them.
- No fixed-eps degree growth is claimed.
- Ordering depends on the gate and scale: softplus1 at s = sqrt 2 is ordered (chi_1 = 0.46); z^2 is smooth and
  non-negative but chaotic (4/3).

**Learnability side.**
- A sorting network whose comparators use softplus_beta has uniform error at most (comparator depth) x log 2/beta
  (PROVEN). This is a sufficient construction, not a proof that accurate smooth nets must be sharp.
- ||T_softplus - T_ReLU|| <= C/beta for coupled networks (PROVEN). This is a stability statement, not a learnability bound.

**Smooth fronts.**
- With phi' > 0 and invertible layers, fronts have densities: no faces, no ray atoms (PROVEN).
- Whether the densities are square-integrable on average is open.

## 3. Random nets (pre-registered grids C and K)

**Grid C, cut ranks** (logs/C_table.md).
- 512 fronts; 256 columns of (back, unit) pairs.
- Each cell: raw in-sample r_in(0.1) / held-out r_ho(0.1) / scale-normalised r_in(0.1). The registration made raw r_in
  primary and the scale-normalised rank a co-statistic.
- r_in is biased low and r_ho biased high; neither is a confidence bound.
- Identical-front controls give rank 1. Their tails, up to 2.2e-4 Vbar, leave eps = 0.01 unresolved.

| config | ReLU | softplus16 | softplus4 | softplus1 | GELU | SiLU | gsrelu0.5 | smoothstep1 | tanh (biased) |
|---|---|---|---|---|---|---|---|---|---|
| (8,4,4) | 9 / 10 / 30 | 21 / 24 / 33 | 25 / 29 / 32 | 37 / 41 / 41 | 18 / 21 / 25 | 13 / 15 / 23 | 24 / 27 / 32 | 12 / 14 / 29 | 68 / 103 / 68 |
| (16,8,8) | 37 / 48 / 47 | 33 / 39 / 39 | 32 / 38 / 49 | 71 / 86 / 75 | 15 / 17 / 25 | 9 / 10 / 16 | 36 / 42 / 43 | 31 / 38 / 35 | 93 / 146 / 93 |
| (16,8,8) Vbar / E M^2 | 0.57 | 0.48 | 0.36 | 0.0003 | 2.3 | 4.2 | 0.32 | 1.08 | 0.67 |

- **Smoothness alone changes little.** softplus16 (beta s = 23) and smoothstep1 are within 25% of ReLU on the
  scale-normalised ranks at both widths and on the raw ranks at width 16. At width 8, ReLU's anomalous raw 9 makes the raw
  ratios 1.3-2.3.
- **Differences are associated with the phase.**
  - Chaotic GELU and SiLU (chi_1 = 1.06-1.07) have 2.5-4x smaller ranks than ReLU at width 16. Their outputs vary more
    within a net (Vbar / E M^2 = 2.3-4.2), so the relative budget eps^2 Vbar is larger.
  - The ordered softplus1 needs about 2x more (Vbar / E M^2 = 3e-4).
  - Saturating tanh (biased; chi_1 about 1.17 with the bias accounted for) needs 2.5x more.
  - Because the spliced gate inputs vary (section 1), "phase causes the difference" is an interpretation, not a
    controlled result.
- **ReLU at (8,4,4).** Its raw rank is held down by 2 of 32 backs carrying 80% of the energy.
- **Asymptotics are not visible.** Secant exponents from eps = 0.1 to 0.03 are 0.4-1.1 for every gate.

**Grid K, Gaussian closure against MC and RQMC** (logs/k2_table.md).
- 4 nets per cell; truth from 2^21 samples (its own nMSE floor is 4.8e-7).
- Each cell shows "ratio (wins)":
  - ratio = nMSE_K2 x N_eq, so below 1 means K2 beats MC;
  - wins = nets out of 4 where K2 beats both MC and RQMC.
- K2's FLOPs include its quadrature: 64 nodes, 128-512 for the near-kinked softplus. The registered ReLU uses its exact
  bivariate-normal closed form, about 2,500 FLOPs per unit pair with heavier CDF weights. The last column is the post-hoc
  ReLU with the others' quadrature and costing (scratch/k2_relu_mehler.py).

| width, depth | ReLU (registered) | softplus16 | softplus4 | softplus2 | softplus1 | GELU | SiLU | gsrelu0.5 | smoothstep1 | tanh (biased) | ReLU, same quadrature (post hoc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16, 2 | 1.8 (0) | 1.2 (0) | 0.31 (1) | 0.08 (3) | 0.010 (4) | 0.21 (2) | 0.12 (3) | 0.07 (3) | 0.54 (0) | 0.045 (4) | 0.34 (1) |
| 16, 16 | 120 (0) | 72 (0) | 7.1 (0) | 0.28 (1) | 0.053 (3) | 6.2 (0) | 17 (0) | 3.4 (0) | 58 (0) | 7.9 (0) | 23 (0) |
| 32, 2 | 0.43 (0) | 0.20 (2) | 0.08 (3) | 0.015 (4) | 0.002 (4) | 0.064 (4) | 0.030 (4) | 0.034 (4) | 0.15 (4) | 0.012 (4) | 0.077 (4) |
| 32, 4 | 2.7 (0) | 2.5 (0) | 0.61 (0) | 0.084 (4) | 0.006 (4) | 0.69 (0) | 0.45 (2) | 0.19 (4) | 1.9 (0) | 0.15 (4) | 0.39 (3) |
| 32, 16 | 27 (0) | 15 (0) | 5.2 (0) | 0.083 (4) | 0.010 (4) | 7.1 (0) | 8.4 (0) | 0.85 (0) | 26 (0) | 3.8 (0) | 4.0 (0) |
| 64, 2 | 0.10 (4) | 0.053 (4) | 0.013 (4) | 0.006 (4) | 0.0007 (4) | 0.026 (4) | 0.011 (4) | 0.009 (4) | 0.031 (4) | 0.004 (4) | 0.028 (4) |
| 64, 8 | 3.1 (0) | 1.3 (0) | 0.62 (1) | 0.041 (4) | 0.003 (4) | 2.0 (0) | 0.38 (4) | 0.18 (4) | 1.6 (0) | 0.32 (4) | 0.53 (4) |
| 64, 16 | 4.9 (0) | 2.3 (0) | 0.87 (0) | 0.045 (4) | 0.003 (4) | 4.8 (0) | 14 (0) | 0.16 (4) | 4.2 (0) | 1.6 (0) | 0.74 (2) |

- **Width helps every gate.** Depth hurts most of them; softplus1 and gsrelu0.5 at width 64 are exceptions.
- **Implementation matters as much as the gate (implementation-specific observation).** As registered, ReLU's closure was costed higher: N_eq about 1,300,
  against about 200 for GELU and SiLU (the near-kinked softplus16 also reached 1,200-1,500 through its quadrature). With the same quadrature and costing (post hoc), ReLU's closure wins on all 4 nets up to
  depth 8 at width 64 and on 3 of 4 at depth 4 at width 32. That matches or beats GELU, SiLU, softplus4 and the smooth
  step. Smoothness does not by itself extend the depth at which the closure wins.
- **At depth 16 only soft ordered gates win by the registered rule** (softplus with beta s <= 2.8; gsrelu0.5 at
  width 64). Same-quadrature ReLU wins on 2 individual nets at width 64. Their outputs
  approach constants:
  - at depth 16, width 64, output variance relative to mean energy is 1e-5 for softplus1, 0.005-0.011 for softplus2 and
    0.05-0.09 for gsrelu0.5;
  - for comparison: 0.14-0.31 for ReLU, 1.3-5.4 for GELU and SiLU;
  - at width 8 their outputs are also more linear (66-83% of the variance in the input's linear part, against 16-66%;
    post hoc).
- **RQMC.** At fixed N it gains 2-190x over MC, more for smoother gates.
  - The largest gains at 2^14 points sit near the truth's noise floor (4.8e-7), so reference noise biases them; they
    carry no uncertainty analysis.
  - At N_eq, RQMC's registered pooled gain relative to ReLU's is 2.2x for softplus1 but only 0.51x for GELU and 0.34x for
    SiLU. Those gates were run at smaller N_eq.

## 4. Learnability (grid T)

**Sorting MLP.**
- Recipe: narrow-deep's bias-free residual net, width 16, 16 layers, 2^14 steps, 3 seeds.
- Metric: nMSE on 2^18 fresh inputs.
- The registered recipe stacks the seeds and clips their joint gradient norm, so seeds are coupled (inherited from
  narrow-deep). The last column reruns four arms with per-seed clipping (post hoc).

| gate | nMSE, mean (per seed) | relative to ReLU | per-seed clipping: mean (relative) |
|---|---|---|---|
| ReLU | 0.103 (0.106, 0.107, 0.097) | 1 | 0.113 (1) |
| GELU | 0.094 (0.095, 0.102, 0.085) | 0.91 | 0.099 (0.88) |
| SiLU | 0.101 (0.095, 0.109, 0.099) | 0.97 | |
| softplus16 | 0.102 (0.099, 0.103, 0.104) | 0.99 | |
| smoothstep1 | 0.103 (0.108, 0.103, 0.098) | 1.00 | |
| softplus4 | 0.120 (0.135, 0.111, 0.115) | 1.16 | 0.122 (1.08) |
| gsrelu0.5 (post hoc) | 0.121 (0.131, 0.115, 0.116) | 1.17 | |
| softplus1 | 0.196 (0.200, 0.195, 0.193) | 1.89 | 0.187 (1.66) |
| tanh, trainable biases | 0.194 (0.182, 0.198, 0.202) | 1.87 | |
| SwiGLU, bilinear (no norm) | NaN: the residual stream explodes at initialisation | - | |
| pre-norm ReLU / SwiGLU / bilinear (Amendment 3) | 0.134 / 0.161 / 0.172 | 1 / 1.20 / 1.28 (vs pre-norm ReLU) | |

The pre-norm hurts sorting (0.134 against 0.103 for ReLU). It probably does so by removing the input's scale; that is
untested.

**Sharpness-pinned softplus** (Amendment 2). Each gate input is normalised per unit to RMS sqrt 2, so beta s cannot grow.

| beta (beta s) | 16 (23) | 8 (11) | 4 (5.7) | 2 (2.8) | 1 (1.4) | pinned ReLU |
|---|---|---|---|---|---|---|
| nMSE (relative to pinned ReLU) | 0.094 (1.01) | 0.093 (1.00) | 0.096 (1.04) | 0.112 (1.21) | 0.139 (1.50) | 0.093 |

**PFN Bayes task.**
- Recipe: the tiny-pfn ctrl recipe from cm-interp-B (2 blocks, width 32, 90,000 steps), with the MLP activation swapped.
- Metric: test KL to exact Bayes on 16,000 episodes.

| gate | KL, mean (per seed) | median / GELU's | gate input RMS, trained / initial (post hoc) |
|---|---|---|---|
| GELU (the recipe's) | 0.0075 (0.0069, 0.0084, 0.0073) | 1 | 1.2-1.5 |
| softplus4 | 0.0094 (0.0067, 0.0080, 0.0137) | 1.10 | 2.2 |
| SwiGLU (hidden 43, parameters matched) | 0.0104 (0.0126, 0.0068, 0.0118) | 1.62 | 1.2-1.8 |
| SiLU | 0.0116 (0.0073, 0.0088, 0.0186) | 1.21 | 1.1-1.4 |
| ReLU | 0.0122 (0.0071, 0.0082, 0.0213) | 1.12 | 1.2-1.4 |
| softplus1 | 0.0152 (0.0150, 0.0167, 0.0138) | 2.06 | 4.2-6.4 |

- Seed variance is large: one seed of ReLU and one of SiLU end 2-3x worse, like phase B's stalled seeds.
- Training sharpens the soft gates even on this smooth target: softplus1's beta s goes from about 0.5 to about 2.8.

## 5. Trained nets (grid R)

On the trained sorting nets (median of 3 seeds; E[output] for x ~ N(0, I_16)):

| gate | K2 / MC | K2 / RQMC | K2 wins (of 3) | RQMC gain over MC | tail beyond degree 8 |
|---|---|---|---|---|---|
| ReLU | 2.6 | 29 | 0 | 11 | 2.3e-2 |
| softplus16 | 3.8 | 31 | 0 | 8 | 2.1e-2 |
| softplus4 | 1.05 | 5.8 | 0 | 5 | 9.5e-3 |
| softplus1 | 0.84 | 9.3 | 0 | 12 | 6.9e-4 |
| GELU | 1.4 | 13 | 0 | 7 | 1.5e-2 |
| SiLU | 2.6 | 21 | 0 | 7 | 7.4e-3 |
| tanh (biased) | 1.7 | 22 | 0 | 11 | 5.8e-3 |
| smoothstep1 | 2.0 | 18 | 0 | 9 | 3.1e-2 |
| gsrelu0.5 (post hoc) | 2.1 | 16 | 0 | 8 | 9.4e-3 |
| pinned softplus, beta 16 / 8 / 4 / 2 / 1 | 11.7 / 10.6 / 6.4 / 4.3 / 1.3 | 92 / 74 / 39 / 26 / 12 | 0 each | 5-9 | 2.0e-2 / 1.5e-2 / 6.6e-3 / 1.0e-3 / 2.6e-5 |
| pinned ReLU | 7.5 | 71 | 0 | 9 | 2.2e-2 |

- The closure beat both baselines on 0 of 45 nets and beat MC on 6.
- One softplus1 net is a statistical tie with RQMC: ratio 1.03, with RQMC's standard error larger than the difference.
- Numerical convergence of the closure, rechecked with 4x the quadrature nodes and Mehler order 128 (post hoc,
  logs/posthoc_review.log): the change in the estimate is
  - under 0.5% of the reported error for most gates;
  - 1-3% for softplus16;
  - 5-14% for smoothstep1.

  No conclusion changes.
- The pre-norm and GLU arms are not scored: no closure through RMSNorm or GLU products was implemented.

## 6. The frontier

Full table: logs/pareto.md. Figure: logs/pareto.png.
- **Free gates.**
  - The ones that learn like ReLU leave the closure 1.4-3.8x behind MC on their trained nets.
  - The only free gate whose closure beats MC on the median net (softplus1, 0.84) learns sorting 1.9x worse. RQMC still
    beats it on every net, one of them a tie.
- **Pinned ladder.** Lowering beta s trades learnability for closure accuracy overall:
  - closure error relative to MC goes 11.7x -> 1.3x;
  - sorting loss goes 1.0x -> 1.5x pinned ReLU's.
  - There are exceptions: softplus8 beats softplus16 on both axes, and pinned ReLU beats softplus16 on both.
- **Answer to the brief.**
  - Within this family, and with the tested closure, no point had a deductive win on a trained net while learning as
    well as ReLU.
  - On random nets, the points where the closure wins at depth are ordered and nearly constant. On trained sorting nets
    no such pattern is established: trained softplus1's output is about as linear as the others' (55% vs about 50%).
  - Registered P12 holds, but it is a composite of ranks against ReLU and closure wins, not "summary size below 1/eps^2".
  - Whether trained smooth nets admit small summaries (the cut rank of trained nets) was not measured.

## 7. Pre-registered predictions (registered rules; runs/C_scores.json, runs/K_scores.json, logs/posthoc_review.log)

| # | prediction (short) | result |
|---|---|---|
| P1 | no arm with sharpness >= 5 below 2/3 of ReLU's raw r_in(0.1) | **refuted** at width 16: SiLU 9, GELU 15 against 24.7 |
| P2 | softplus1 >= ReLU at width 8 | holds (37 vs 9) |
| P3 | tanh (biased) >= ReLU at both widths | holds (68 vs 9; 93 vs 37) |
| P4 | secant exponents < 2; analytic arms not >= 0.4 below ReLU at both widths | holds (0.37-1.10; complete after smoothstep1 at width 16) |
| P5 | (a) no K2 win at depth 16, width <= 16; (b) K2/MC spread < 3x among the listed arms | **refuted**: (a) softplus1 wins 3/4 at width 16 (collapsed nets); (b) spread >= 3x in 16/16 cells |
| P6 | RQMC gain at N_eq >= 1.5x ReLU's for softplus1, GELU, SiLU at width 8; < 2 at width >= 32 | **refuted**: pooled 2.20, 0.51, 0.34; gains >= 2 at width 32-64 |
| P7 | K2's error falls along the softplus ladder in >= 12/16 cells | holds (16/16) |
| P8 | sorting: GELU, SiLU, softplus4, softplus16, SwiGLU within 20% of ReLU; softplus1 >= 20% worse | single-gate arms hold. SwiGLU diverged as registered; pre-norm SwiGLU is 1.2002x pre-norm ReLU, **refuted by 0.0002** (within seed noise) |
| P9 | training sharpens softplus1 and softplus4 on sorting by >= 1.5x (trained RMS / sqrt 2) | **refuted**: softplus1 1.51 (holds), softplus4 1.41 (fails). On this nominal statistic ReLU gives 1.47 and SiLU 1.88; measured growth from initialisation is 1.2-1.6x, and tanh shrinks |
| P10 | PFN: ReLU, softplus4, SiLU, SwiGLU within 20% of GELU; softplus1 within 30%; sharpening < 1.5 | **refuted**: KL means 1.62, 1.26, 1.54, 1.38 and 2.02 x GELU. On the literal layer statistic softplus1 is 2.05-2.72 (median 2.32) and the others 0.46-1.02; 4-6x against measured initialisation |
| P11 | trained nets: K2 loses for every arm; smooth arms' tail share at degree 8 >= 0.5x ReLU's | first part holds (0/24 single-gate free nets); second **refuted** for softplus4, SiLU, tanh, softplus1 |
| P12 | no arm within 20% of ReLU's loss and (a K2 winner, or half ReLU's rank at both widths) | holds |
| P13 | pinned: loss falls with beta (rho <= -0.8); pinned softplus1 >= 1.5x pinned ReLU | first holds (-0.9); second **refuted by 0.003** (1.497x) |
| P14 | pinned: K2/MC falls as beta falls; tail share within 2x of its initial value | holds (rho = -1; 0.92-1.15x) |
| P15 | no pinned arm within 20% of pinned ReLU and a K2 winner | holds |

## 8. Deviations, amendments and post-hoc analyses

**Amendments.**
- *Amendment 1* (hashed 01:57Z). Bias-free tanh is degenerate. Its runs are reported, not scored; tanh with biases
  replaces it. The sorting tanh arm has trainable biases (+3% parameters).
- *Amendment 2* (02:05Z). The pinned arms.
- *Amendment 3* (03:25Z). Pre-norm GLU arms, after the registered SwiGLU and bilinear arms diverged.

The final review confirmed all four hashes and that every amendment precedes its data.

**Registered statistics that changed.**
- *Quadrature.* The closure's per-gate quadrature (128-512 nodes for softplus with beta >= 4, instead of 64) was set after
  the K2 test and before grid K, and is costed.
- *Cost conventions.* They are not uniform. gsrelu is charged 154 per evaluation (CDF/PDF weights), GELU 20, the
  registered ReLU 2,500 per pair. The trained-net convergence check changed quadrature and Mehler order together. Closure
  comparisons across gates are therefore implementation-specific.
- *P9 and P10.* Their sharpness statistics are scored literally above; the measured-initialisation ratios are post hoc.
- *P6.* Re-aggregated as registered (pooled over nets and depths); it remains refuted.
- *P1's wording.* P1's list calls GELU and SiLU "sharpness >= 5", although GELU's s/width is 1.4. Scored as written.

**Corrections from the reviews.**
- *Theory.* The v1 ray table was called exact. It was replaced by a convergence-checked table, still not certified.
- *Hermite degree table.* Its tails came from a clipped Parseval residual; they are recomputed by direct sums with a 2K
  check, and unresolved entries are marked.
- *Metadata.* tanh-with-bias's phase was mislabelled 1.34; it is about 1.17 with the bias variance (lib/gnet.py fixed).

**Design limits.**
- Grid C used 256 columns (the lane used 512), so ranks are capped at 256.
- Grid C's backs see spliced fronts: gate-input RMS 0.45-3.6 at the first back layer and 0.06-8.5 at the last, not a
  fixed sqrt 2.

**Post-hoc analyses.**
- ReLU with the same quadrature as the others.
- Per-seed clipping reruns.
- Trained-net closure convergence.
- Linearity and collapse diagnostics.
- The gsrelu0.5 sorting arm.
- The splice-scale check.
- Fixed-N RQMC gains.

**Process slip.** One early theory script used multithreaded BLAS (about 3.6 cores) for about 25 minutes. Everything else
ran at nice 19 / ionice -c3 with at most 3 heavy processes.

## 9. Process and review

- **Pre-registration.** PREREG.md was hashed at 01:37Z, after theory-only numbers and smoke tests and before any network
  measurement.
- **Co-prover.** gpt-6-astra, xhigh effort, one round on theory v1.
  - Seven claims were wrong, three were upgraded to PROVEN, and it added two new results.
  - All are applied (`coprover/astra_r1_answer.md`).
- **Final review.** A fresh GPT-6 session, high effort (`review/review_gpt6_final.md`): 19 findings (15 major, 4 minor);
  verdict "major revision".
  - It confirmed the projection lemmas, gate identities, logistic mixture, softplus asymptotic, density results and
    sorting construction.
  - It reproduced selected ranks, closure entries, PFN initialisations and one test KL; the tests pass; it modified no
    files.
  - The broad frontier and scale conclusions did not survive as written. Every finding is applied:
    - 1: the scale conditions and the counterexample;
    - 2: P12 described as registered; trained-summary tractability left open;
    - 3: the 10M+ sentence replaced;
    - 4: the sorting construction labelled sufficient;
    - 5: the ReLU implementation confound, with the post-hoc same-quadrature rerun;
    - 6: closure convergence on trained nets;
    - 7: the splice scales;
    - 8: held-out ranks shown, "finite-ensemble estimates";
    - 9: coupled clipping disclosed, with a per-seed rerun;
    - 10: literal P9/P10 statistics;
    - 11: the Hermite tails recomputed;
    - 12: the ray model specified, the ReLU^2 exception corrected;
    - 13: reference floor, the tie and the 6 MC wins;
    - 14: the budget wording;
    - 15: depth conditions;
    - 16: bilinear-only instability;
    - 17: tanh phase;
    - 18: P6 pooling, P4 completeness, scale-normalised ranks as registered;
    - 19: non-monotone trends.
- **Second review round.** A fresh GPT-6 session (`review/review_gpt6_r2.md`) checked the fixes.
  - It judged 11 of the 19 findings fully addressed and 8 partly.
  - It listed 10 remaining issues (6 major), all now applied:
    - the summary scoped to the tested closure and to associations;
    - held-out ranks in the summary;
    - the sorting-tanh exception to "inputs grow";
    - P10 scored on the literal layer statistic;
    - the comparison labelled implementation-specific;
    - separate bounded and unbounded-scale exponents in the theory note;
    - "resolved tested eps" in the ray claims;
    - both splice-scale ranges;
    - "win" meaning the registered 3-of-4 rule;
    - four rounding corrections.
  - Its verdict before these fixes was "major revision". No third round was run.
- **Compute.**
  - CPU at nice 19 / ionice -c3.
  - One on-demand L4 for the PFN arm: 1.27 h, $1.10; deleted, instance list checked at 03:27Z (GPU-LOG.md).

## Handoff

**Code.**
- `lib/acts.py`: gates and their exact Hermite spectra.
- `lib/gnet.py`: random nets with pilot normalisation.
- `lib/k2gen.py`: Gaussian closure with FLOPs.
- `scratch/run_cut_act.py` + `analyze_cut.py` + `score_C.py`: grid C.
- `scratch/k2_grid.py` + `summ_k2.py` + `score_K.py` + `k2_relu_mehler.py`: grid K.
- `scratch/ray_family.py`, `ray_cert.py`, `hermite_classes.py`, `depth_radius.py`, `meanfield.py`: theory numbers.
- `train/sort_act.py`: sorting, with modes pin / pn and CLIP_PER_SEED.
- `gpu/train_act.py`: the PFN arm.
- `scratch/k2_trained.py`: grid R.
- `scratch/pareto.py`, `pareto_fig.py`: the frontier.
- `scratch/posthoc_review.py`: the review checks.

**Tests.** `python3 tests/test_acts.py`; `python3 tests/test_k2.py`.

**Next if wanted.** Each is bounded and CPU-sized.
- Measure the cut rank on trained nets, with a family of input laws as fronts. This is the missing "trained summary"
  test.
- A closure through RMSNorm and GLU products, to score SwiGLU on trained nets.
- An error-matched closure comparison: every gate at equal quadrature error, with RQMC rescored at each budget.
