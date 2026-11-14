# Tractable gates: assumptions and open questions (leaf q, 2026-10-31 ~01:10Z)

Lane: tractable-gates (bead pool-2wj2bh). Author: opus-mats-gates (Claude Opus 5.5).
Each item: the assumption or question, then how it gets checked. "This sense" = the cut-rank lane's sense
(outbox/cut-rank-lower-bound-2026-10-30.md): the eps-rank r(eps) of the front-to-back cut operator, i.e. the size of a
one-way linear summary of the front that matches N = 1/eps^2 samples in ARC's averaged criterion.

## Definitions to fix before data
1. Tractability has two readings, and both are reported.
   (a) Summary size: r(eps) against 1/eps^2 (the cut-rank lane's criterion). Not a running time.
   (b) Running time: a deductive estimator (Gaussian closure, K2) against Monte Carlo and RQMC at equal FLOPs, scored
       as ARC does (nMSE = |est - truth|^2 / tr Var; MC with N samples scores 1/N).
   Check: both written into PREREG.md with estimators, budgets and win rules.
2. Effective sharpness. For softplus_beta(z) = log(1 + e^{beta z})/beta acting on pre-activations of standard deviation
   s, the dimensionless sharpness is beta*s; GELU's built-in Phi has unit width, so its sharpness is s; Gaussian noise
   of std sigma added before a ReLU gives E relu(z + sigma xi) = sigma g(z/sigma), sharpness s/sigma.
   Check: theory note gives the Hermite coefficients as functions of beta*s (or s/sigma); measured s per layer in random
   and trained nets.

## Theory questions
3. Does every C-infinity activation give r(eps) = o(eps^-2) at fixed width, for any depth and any front law, while
   finite smoothness loses at large width? Expected: yes, via a polynomial-approximation bound
   r(eps) <= C(n + D(eps), n) that needs no assumption on the fronts.
   Check: proof in notes/theory.md, co-prover check (gpt-6-astra), exact ray-front numbers.
4. Which decay class does each activation fall in? Expected: ReLU and leaky ReLU, power law (c_k^2 ~ k^{-5/2});
   step, k^{-3/2}; tanh, sigmoid, softplus, SiLU/Swish (poles at finite imaginary distance), exp(-a sqrt k) with
   a proportional to (strip half-width)/s; erf, GELU and Gaussian-smoothed ReLU (entire, Gaussian type), geometric
   rho^k with rho = s^2/(s^2 + tau^2). Random input radius (a scale mixture) weakens every class by a power of k in the
   exponent but keeps it super-polynomial.
   Check: numerical Hermite coefficients to degree >= 200 at several s; fits of log c_k^2.
5. How does depth change this? Expected: composing dual kernels shrinks the distance from t = 1 to the nearest
   singularity like c/L at criticality, so the degree needed grows about linearly in depth.
   Check: compose dual kernels numerically (infinite width) and read the radius from the Taylor coefficients.
6. Where is the crossover eps*(n, activation, depth) below which r(eps) < 1/eps^2? Expected: at width 8 and moderate
   sharpness, eps* is far below 0.01 in the worst case (ray fronts); for wide nets it is astronomically small, so
   smoothness alone does not make summaries beat sampling at practical eps unless fronts also spread.
   Check: exact ray-front ranks (scratch/ray_family.py) for n = 3..64, eps = 0.3 .. 1e-6.
7. Multiplicative gating. Expected: phi(W1 h) * (W2 h) with independent Gaussian W1, W2 has kernel t*k_phi(t), so the
   same decay class as phi; the bilinear GLU (no phi) is a polynomial net with exactly finite cut rank.
   Check: theory note; include SwiGLU and bilinear GLU in the random-net and training grids.
8. Smooth fronts. Expected: for strictly monotone smooth phi with phi' > 0 and invertible weights, front laws have
   densities (no ReLU faces or ray atoms); non-monotone GELU/SiLU have folds whose density singularities can leave
   L^2. The ReLU rare-event ray theorem transfers to softplus only down to eps of order the residual spread.
   Check: theory note, labelled PROVEN / HEURISTIC / CONJECTURE.

## Measurement assumptions
9. Init. A common recipe across activations: bias-free weights N(0, g^2/n) with g chosen so the pre-activation second
   moment is a fixed point (q = 2, the He-ReLU value) where one exists; tanh has no nonzero bias-free critical point,
   so it uses ARC's critical init with biases (sigma_w^2 = 2.025, sigma_b^2 = 0.111, q* = 0.85).
   Check: report q_l and chi_1 per activation; flag ordered/chaotic.
10. Inputs. Non-homogeneous activations cannot use the cut-rank lane's exact radius integration, so inputs are Gaussian
   x ~ N(0, I_n). Check: ReLU run both ways reproduces the lane's ranks within noise.
11. Energy heavy tails (cut-rank finding 1) may differ by activation. Check: energy diagnostics and the post-hoc
   scale-normalised rank next to the raw one, both declared in the PREREG.
12. Resolution. The cut-rank lane resolved eps >= 0.03 only. Check: identical-front controls per activation; eps = 0.01
   reported as unresolved unless the control tail is below it.

## Learnability questions
13. Do smooth gates and GLU variants learn the program's tasks as well as ReLU at matched parameters and steps?
   Tasks: the narrow-deep residual sorting MLP (width 16; piecewise-linear target with kinks) and the tiny-pfn Bayes
   task (smooth target). Check: task loss over >= 3 seeds per arm.
14. Does training sharpen the gates? Expected: on sorting (a kinked target) beta*s grows; on the PFN (a smooth target)
   it need not. Check: per-layer pre-activation std before/after training, and the Hermite tail of each trained unit
   on its own trained pre-activation law.
15. Does tractability survive training? Check: K2 vs MC/RQMC and the per-unit Hermite tail on the trained nets.
16. Kelvin's bar ("could scale to 10M+ parameter models"). GELU and SwiGLU are what large models use; ReLU's exact
   sparsity is a real cost of smoothing. Check: literature note; the report separates "learns as well at our scale" from
   "known at scale".

## Compute and process constraints
17. CPU only, nice 19, ionice -c3, at most 3 heavy processes (General is loaded, about 19 on 16 cores). GPU only after
   asking main (cap about $3). Check: process list before each launch; GPU-LOG.md if any GPU is used.
