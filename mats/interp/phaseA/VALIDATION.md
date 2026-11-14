# cmlib validation on known-answer cases (2026-10-28)

**TL;DR.** On known answers the toolkit is correct when it has the true ratio w. The direct (weight-free) gap is unbiased in every case. The classifier-estimated ratio ŵ is the weak link:
- **It biases every ratio-based gap toward 0.** The bias is −6% to −15% when the classifier AUC is about 0.65. It reaches −80% to −100% once AUC ≤ 0.52, yet the true gap can still be large there, because the structure G cares about sits in the tails.
- **The quadratic family is blind** to anything beyond second moments.
- **Ratio-fixed bootstrap CIs are far too narrow.** The variance from refitting the classifier must be added.

The direct estimate should therefore be the primary number, with ŵ used only for localisation and naming (as METHOD.md §1 #3 and §3.5 already lean). No real model has been touched. No decision is needed.

Everything here is reproducible:
- `OMP_NUM_THREADS=1 python validate.py --procs 4` (about 22 min on the shared box) writes `results/validation.json`.
- `python summarize.py` turns it into `results/tables.md`, which has every number below and more.
- `python -m pytest tests` runs 18 tests in about 70 s, and all of them pass.

Checked at 2026-10-28 ~15:00Z against `results/validation.json` (796 rows).

Throughout, Δ = L(q) − L(p) = E_q G − E_p G = −Cov_q(G, w). Kelvin's L(0) − L(1) is the same quantity.

## Toolkit (`cmlib/`)

| Module | What it does | METHOD.md |
|---|---|---|
| `ratio.py` | Calibrated classifier ratio log ŵ = a·score + b. Families: quadratic logistic in whitened h with z-interactions, and an MLP. Classes balanced by weights. Split into train / cal (C selection, Platt) / test (diagnostics). Diagnostics: AUC (+ Hanley–McNeil SE), BCE, JS bound, DV bound, plug-in KL, E_q ŵ, E_p 1/ŵ, PSIS k̂ on both sides, χ² (only when k̂ < 0.5), ECE, train − test BCE. Also `null_gate` (log ŵ ≡ 0 when the AUC is within 2 SE of 0.5), `crossfit_log_w`, and `refit_spread` (classifier-refit variance by half-sampling). | §3.3, T2–T6 |
| `estimators.py` | `direct_gap` (paired: studentised bootstrap and Hill tail index; or unpaired), `cell_rms` (unbiased E_c Δ(c)² from two halves), `group_rms` (the same over discrete z), `cov_qw` (q-side identity, plain and SNIS, per group, weights p(z)), `temper_q` (Kelvin's w^α SNIS from q), `temper_bridge` (pooled bridge normalised per group, with dL/dα = Cov_{p_α}(G, log w) and its integral, ESS), `region_split` (exact mass + within parts). | §3.1–3.5 |
| `qfam.py` | `CondGaussian` (per latent group; covariance full / diag / iso / lowrank:r PPCA) and `LinearGaussian` (ridge mean, λ by grouped CV, optional per-group residual covariance). Both provide sample, logpdf and `subset_sample`, which does knock-out h_A ~ q(h_A \| h_−A, z) and knock-in by Gaussian conditioning. `params()` exposes Cholesky factors for quantisation. | §7.1, subset localisation |
| `consistency.py` | Tier i `identity_check`. Tier ii `consistency_check`: refits q̃ = Proj(F#q_l) and q^fit, and reports Γ_proj, Γ_fit, Δ(q̃) − Δ(q^fit) and AUC(q̃ vs q^fit). SEs include the variance of refitting both families. Reports the METHOD pass rule, the Γ-only rule and a two-sided rule. | §6 |
| `dl.py` | Grammar bits, uniform quantisation with an fp16 scale and smallest b* within a tolerance, BIC cross-check, `Ledger` with the lower convex-hull frontier and `orders_agree`, and break-even N*. | §8–9 |
| `synth.py` | The known-answer worlds below, each with exact 1-D quadrature for Δ, L(α), KL and χ². | – |

**Not implemented yet:**
- the equivariant DeepSets classifier;
- uLSIF;
- PSIS smoothing (only k̂ is computed);
- the σ-smoothing rule;
- mixture / factor q families (q1e/q1f);
- rank pruning in DL_param;
- the tiny-PFN sites, the exact conditional sampler and the G quadrature (METHOD §11's `lib/`).

These are next, alongside R1.0.

## (i) p = q: gap 0, w = 1

World: MixWorld (non-Gaussian, d = 8, three latent groups), with q = p exactly. In the second row, p is Gaussian and q is a conditional Gaussian fitted on 20k separate draws. Training uses 20k draws per class, and estimation uses 20k fresh draws. The first two numeric columns give the share of replicates whose 95% CI excludes 0 (nominal 0.05).

| Setup | R | AUC | Direct CI misses 0 | Cov CI misses 0, ratio held fixed | With refit SE added | Gated (log ŵ ≡ 0 if AUC n.s.) | Mean \|Cov\| / direct SE |
|---|---|---|---|---|---|---|---|
| q = p, quadratic classifier | 40 | 0.5007 ± 0.0064 | 0.05 | 0.88 | 0.05 (R = 20) | 0.05 | 0.15 |
| q = p, MLP | 16 | 0.5000 ± 0.0034 | 0.12 | 0.44 | – | 0.00 | 0.13 |
| p Gaussian, q fitted on 20k | 20 | 0.508 ± 0.012 | 0.15 | 0.70 | – | 0.30 | 0.31 |

Other diagnostics for q = p:
- plug-in KL 4e-5;
- E_q ŵ = 1.0003;
- ECE 0.0003;
- sd of log ŵ 0.02.

**Findings:**
- The spurious Cov from classifier noise is small, about 0.15 direct-SE.
- The bootstrap that holds ŵ fixed reports it as significant in 44–88% of nulls. Adding the half-sampling refit spread restores the nominal 5%.
- In the third row, the fitted q really differs from p by its fit error, and the classifier sees that (AUC significant in 40% of replicates).

## (ii) Non-Gaussian p with an analytic ratio

World:
- p(h\|z) = ½N(μ_z ± 2 m_z, I) in d = 8, with z ∈ {0, 1, 2}.
- q(h\|z) is the moment-matched Gaussian N(μ_z, I + 4 m_z m_zᵀ).
- KL(p‖q) = 0.172 nats and χ² = 0.315 (exact, by quadrature).
- Three choices of G(τ), where τ = m_z·(h − μ_z):
  - τ⁴/10, with Δ = 2a⁴/10 = 3.2 in closed form;
  - softplus(2τ − 1), with Δ = −0.1451 by quadrature;
  - τ², with Δ = 0 exactly even though w ≠ 1 (q matches the first two moments).

Bias, with 95% CI coverage in brackets:

| w used | R | G = τ⁴/10 (Δ = 3.2) | G = softplus (Δ = −0.1451) | G = τ² (Δ = 0) |
|---|---|---|---|---|
| true w: Cov plain / SNIS / L(0) − L(1) SNIS / bridge | 40 | +0.034 / +0.039 / +0.039 / +0.032 (0.82–0.88) | +0.001 for all four (0.93–0.95) | +0.002 to +0.004 (0.82–0.85) |
| true w: −∫ dL/dα (21-point trapezoid) | 40 | +0.057 | +0.001 | +0.007 |
| q = CondGaussian fitted to 20k p-draws, direct | 40 | +0.021 (0.97) | −0.002 (0.95) | +0.004 (0.95) |
| MLP ŵ: Cov plain / bridge | 12 | −0.286 (−9%) / −0.435 (−14%); coverage 0.58 / 0.42 ratio-fixed, 0.92 / 0.92 with refit SE | +0.018 (−12%) / +0.017 | +0.003 / −0.017, sd 0.11–0.13 (ratio-fixed SE 0.03) |
| quadratic ŵ | 12 | −3.20 (−100%) | +0.145 (−100%) | 0 (trivially) |

- **Exact tempered path:** for G = τ⁴/10, L(α) at α = 0, 0.2, …, 1 is 7.500, 5.655, 4.942, 4.596, 4.409, 4.300.
  - With true w, the bridge gives 7.532, 5.657, 4.941, 4.595, 4.408, 4.300.
  - With MLP ŵ, it gives 7.261, 6.061, 5.360, 4.934, 4.668, 4.497. The whole path is compressed toward its middle.
- **MLP ratio quality:**
  - AUC 0.652;
  - corr(log ŵ, log w) = 0.905;
  - plug-in KL 0.149 against a true 0.172 (DV bound 0.149);
  - ECE 0.016;
  - E_q ŵ = 1.000.

  So calibration looks perfect while the gap is still attenuated. **ECE and normalisation checks (T2, T5) cannot detect the attenuation.** T1 (bridge endpoints against the direct means) does: it fails in 58% of the τ⁴ replicates.
- **Quadratic family:** AUC 0.4997. For a moment-matched Gaussian q, every quadratic feature has the same mean under p and q, so the quadratic classifier can only see second-moment misfit. METHOD.md calls this family mandatory family 1; on its own it would report "q = p" here. T1 fails in 100% of these replicates.
- **Under-coverage with the true w:** even with true w or the direct estimate, the CIs for τ⁴ and τ² cover only 0.82–0.88 (R = 40; the coverage estimate has an SE of about 0.035). A 200-replicate check of the direct estimate gives sd/SE = 1.16 and coverage 0.90, against 1.04 and 0.935 for softplus. The Hill check in `direct_gap` is there for this.

## (iii) A hand-built net that ignores some structure

Setup:
- h = [A | B | C] ∈ R⁶, with label y.
- A is Gaussian given y.
- In B, b1 is strongly bimodal and has exactly zero weight into the net; b2 is Gaussian and is read.
- In C, c1 is bimodal and is read through 8 ReLUs.
- G = cross-entropy of the 2-layer net.
- q = CondGaussian(y). A subset is swapped by knock-out: h_S ~ q(h_S \| h_−S, y), with the rest real.
- The reference is the paired direct Δ at n = 400k with the same q, per replicate (10 replicates).
- The table shows estimate − reference, mean ± sd. MLP classifiers. "Output space" means the ratio is fitted on the logits.

| Swapped | Δ reference | Direct err | h-space AUC | h-space −Cov err | Output-space AUC | Output-space −Cov err | Gated output-space err | ptp L(α), output space |
|---|---|---|---|---|---|---|---|---|
| B (ignored structure) | −0.0013 | +0.001 ± 0.004 | **0.812** | −0.015 ± 0.022 | **0.499** | −0.001 ± 0.013 | +0.001 ± 0.004 | 0.006 |
| C (read) | −0.160 | −0.006 ± 0.022 | 0.710 | +0.014 ± 0.035 | 0.639 | +0.022 ± 0.045 | +0.022 ± 0.045 | 0.143 |
| A (q exact) | −0.002 | +0.001 ± 0.005 | 0.500 | −0.002 ± 0.010 | 0.501 | +0.003 ± 0.013 | +0.002 ± 0.005 | 0.008 |
| all | −0.164 | +0.003 ± 0.018 | 0.866 | −0.011 ± 0.038 | 0.639 | +0.002 ± 0.043 | +0.002 ± 0.043 | 0.163 |

**Swapping B:**
- Cov ≈ 0 and L(α) is flat, as required, even though the h-space classifier separates B easily (AUC 0.81).
- The output-space ratio is the clean tool here: AUC 0.50, Cov ≈ 0, exactly 0 when gated.
- The h-space ratio leaks a spurious −0.015 (≈ 2 SE of the mean over replicates). This supports METHOD #3: run the tempering path and identities in output space, and use h-space ŵ only to say *where*.
- All ratio-based gaps are 1.5–2× noisier than the direct gap.

**Cross-layer consistency:**
- Layer l is h, F is the ReLU layer, and both levels use a conditional Gaussian family. There are 20 replicates per case. SEs include the variance of refitting both families.
- The tier-i identity holds to exactly 0.

| Structure at layer l that F reads | AUC(q̃ vs q^fit) | Γ_fit ± SE | Δ(q̃) − Δ(q^fit) ± SE | Pass: METHOD rule | Γ rule alone | Two-sided |
|---|---|---|---|---|---|---|
| none (only the ignored b1 is non-Gaussian): should pass | 0.507 (max 0.518) | −0.001 ± 0.023 | −0.002 ± 0.021 | 20/20 | 20/20 | 17/20 |
| c1 bimodal: should fail | 0.660 | −0.051 ± 0.025 | +0.070 ± 0.022 | 0/20 | **20/20** | 0/20 |

- **The one-sided Γ_fit rule alone misses the inconsistency** in every replicate: the derived q̃ has a *smaller* rms gap than q^fit. Only the AUC term (or the two-sided gap difference) catches it.
- Before the refit variance was added to the SE, the correct case falsely failed 2/6 times.

## (iv) Estimator behaviour as q → p

**Mixture path.**
- q_t = ½N(±t·2m, I + (1 − t²)4mmᵀ): t = 0 is the moment-matched Gaussian and t = 1 is p.
- G = τ⁴/10, with an MLP classifier.
- 12 replicates per t.

| t | true Δ | KL(p‖q) | AUC | Bias of direct | Bias, true w (Cov / bridge) | Bias, MLP ŵ (Cov / bridge) | Coverage of MLP bridge, incl. refit | T1 passes |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.200 | 0.172 | 0.650 | +1% | +1% / +1% | −6% / −11% | 0.92 | 0.42 |
| 0.5 | 3.000 | 0.166 | 0.644 | −0% | −1% / −1% | −9% / −15% | 0.92 | 0.25 |
| 0.75 | 2.188 | 0.127 | 0.615 | +2% | +1% / +1% | −19% / −25% (one replicate's fit failed: sd 0.54) | 0.92 | 0.17 |
| 0.9 | 1.100 | 0.054 | 0.519 | +1% | 0% / 0% | −82% / −84% | 0.08 | 0.08 |
| 0.97 | 0.367 | 0.009 | 0.499 | +1% | +1% / 0% | −104% / −103% | 0.00 | 0.17 |
| 1 | 0 | 0 | 0.501 | −0.014 ± 0.093 | 0 | +0.005 / +0.006 | 0.92 | 1.00 |

- At t = 0.9, q is still off by 1.1 (a third of the starting gap). The classifier, though, is at AUC 0.52 and recovers only 16% of the gap.
- The structure G weighs is in the tails. **KL can be small while Δ is not.**
- The direct estimate stays unbiased. T1 flags the classifier failure in 83–92% of replicates.

**Scale path in d = 31** (the width of S2a).
- p = N(0, I), q = N(0, κ²I), G = \|h\|²/d. True w, 30 replicates, n = 20k.

| κ | χ²(p‖q) | k̂ | ESS_q/n | sd: direct | sd: Cov plain IS | sd: Cov SNIS | sd: bridge | Min bridge ESS / pool |
|---|---|---|---|---|---|---|---|---|
| 0.8 | 363 | 0.67 | 0.021 | 0.002 | 0.090 | 0.052 | 0.002 | 0.68 |
| 0.9 | 1.4 | 0.32 | 0.41 | 0.002 | 0.005 | 0.004 | 0.001 | 0.87 |
| 1.1 | 0.61 | −0.01 | 0.62 | 0.003 | 0.003 | 0.002 | 0.002 | 0.89 |
| 1.2 | 3.6 | 0.16 | 0.22 | 0.003 | 0.010 | 0.005 | 0.003 | 0.74 |
| 1.5 | 304 | 0.81 | 0.006 | 0.004 | 0.162 | 0.035 | 0.004 | 0.48 |

**Plain IS from q, i.e. Kelvin's identity E_q[Gw]:**
- Its sd grows with χ² to 40× the direct sd.
- SNIS reaches 73% coverage at κ = 1.5.
- k̂ flags this (> 0.7 at κ = 1.5; 0.67 at κ = 0.8).

**The bridge stays at the direct estimator's precision throughout.**

**Quadratic classifier**, which is the exactly specified family here (527 features, 12k per class), 4 replicates each:
- the bias is still toward 0: −36% at κ = 1.05 (AUC 0.58), −11% at κ = 0.9, and −4% to −6% at κ = 1.2;
- regularisation plus Platt scaling of a noisy score shrinks log ŵ.

## Estimator limits found

1. **Attenuation.** Every ŵ-based gap (Cov, the tempering endpoints, ∫dL/dα) is biased toward 0, so q looks better than it is. The bias is about 10% when AUC ≈ 0.65 and close to 100% when AUC ≤ 0.52. That second regime is exactly where q is nearly right but still loses a large Δ through tails G cares about.
   - Use the direct Δ / Δ_rms to decide whether a rung closes.
   - Use ŵ only for localisation and naming, and always run T1 against the direct estimate.
2. **The quadratic logistic family cannot see residual structure of a moment-matched Gaussian q.** It should not be a mandatory family beyond checking the first two moments.
3. **Ratio-fixed bootstrap CIs are too narrow.** They false-positive in 44–88% of nulls and cover only 42–58% in (ii). Use `refit_spread`, or METHOD's refit bootstrap; this restores 0.05 false positives and 0.92 coverage. `null_gate` removes noise when the AUC is not significant.
4. **The h-space ratio leaks spurious covariance from structure G ignores**, and it is noisier than the output-space ratio. The output-space ratio gives 0 for ignored structure.
5. **Plain IS from q has variance that scales with χ²** (≈ 300 at d = 31 for κ = 0.8 or 1.5). The bridge is bounded and safe.
6. **Consistency rule.**
   - The one-sided Γ_fit rule alone passes a known-inconsistent case 20/20, so keep the AUC term and/or the two-sided gap test. The two-sided test false-fails 15% of the time at 2 SE with two tests.
   - Fit variance of q̃ and q^fit must enter the SE.
7. **Heavy-tailed G** (τ⁴ here) under-covers even for the direct estimate (0.90 at a nominal 0.95). The tiny PFN's KL is also heavy-tailed (the top 10% of episodes carry 43%), so keep the Hill check.
