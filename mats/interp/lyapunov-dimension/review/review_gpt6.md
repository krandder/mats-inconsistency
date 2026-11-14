codex
Major revision required. I reviewed the changing draft through **19:53 UTC, 29 October 2026**. No files were edited.

1. **Critical — §§3–4, C2, summary: proportional-depth counts are not proved.** Fixed-\(N\) Fuss–Catalan convergence, followed by \(N\to\infty\), does not establish the simultaneous limit \(N/n\to\tau>0\). Here the predicted surviving count is \(O(1)\), so even convergence of the empirical spectral distribution is insufficient. The revised §11 correctly calls this heuristic, but the summary and “our regime is the free one” still overclaim. **Fix:** distinguish the iterated-limit theorem from the measured proportional-depth approximation; obtain a joint-limit edge/count result before claiming a width-independent theorem.

2. **Critical — C2’s newly added radial-row justification is false.** \(r=J^Te_N\) includes accumulated angular-to-radial leakage; its norm is not merely the forward radial gain. Conditional on balanced active counts \(m=n/2\), the triangular decomposition gives
   \[
   \mathbb E\|r_N\|^2=\sum_{j=0}^{N}\left(\frac{m-1}{m}\right)^j,
   \]
   which is approximately \(N+1\), not order one uniformly in depth; it is **29.19 at \(m=128,N=32\)**. Interlacing controls counts at an absolute threshold, not thresholds normalized by two different top singular values. **Fix:** prove control of \(s_1(J)/s_1(\widehat J)\); remove the current no-outlier argument.

3. **Critical — §6, `a1_tails.py`, `a4_fit.py`: the predictors describe different families from those measured.** The scripts use \(P\operatorname{Cov}(z)P\), with the mean direction projected out. The actual PB families use **radius-weighted PCs of \(w=z/\|z\|\)**; PBW uses weighted direction covariance, while PBS/PBA also use different subspaces from those reconstructed here. Carrying an independent radius is not carrying the linear functional \(u^Tz\) exactly. Even exact conditioning on \(u^Tz\) would generally require a Schur complement, not orthogonal projection. **Fix:** calculate tails in the actual chart, with the actual fitted blocks and induced perturbations. Until then, the “optimal block needs 20–30 directions” figures are extrapolations from a mismatched proxy.

4. **Critical — §10.2–10.3 and summary: no sampling lower bound follows.** Large joint dimension and failed closures do not establish that every deterministic method costs exponentially, or that beating sampling requires an exactly preserved closed-form family. The displayed grid complexity is a cost for a particular approximation scheme, not a lower bound. Uncovered routes include approximate nonlinear latent closures, backward-observable approximation, conditional integration, control variates, adaptive cubature, and bias correction. Gate entropy and pairwise independence do not exclude these. **Fix:** replace “nothing beats sampling unless…” with a statement about the tested families and explicitly leave algorithmic complexity open.

5. **Major — C1: the conditioned-chain repair remains mathematically inconsistent.** At fixed width, the physical noiseless network eventually dies almost surely; conditioning merely on non-extinction still permits narrow layers that permanently destroy angular ranks. Conditioning on every \(m_\ell>k\) changes **all** terms in the \(k\)-volume formula to expectations under \(m\mid m>k\). Using \(m\mid m>j\) separately for each exponent describes different chains, not one spectrum. **Fix:** state a finite-horizon conditional volume identity with one common truncation, or define a modified chain with a fixed minimum width. Do not label these the physical network’s infinite-depth exponents.

6. **Major — Theorem A: the revised “supremum” claim is still false at fixed radius.** The identity and Cauchy–Schwarz inequality are correct under integrability assumptions. But for \(P=N(0,1)\), no block and \(F(z)=z\), equality requires \(dQ/dP=1+cz\), which is negative somewhere for every nonzero \(c\). The bound cannot even be approached at a fixed positive chi-square radius: near-equality would force convergence in \(L^2(P)\) to that invalid density. **Fix:** claim local asymptotic sharpness as the radius tends to zero, using truncated **conditionally recentered** scores. State exact attainment only under suitable one-sided boundedness.

7. **Major — §§2,8: Theorem A is not an established bound for the fitted empirical families.** Their block laws are empirical approximations, and their discrete fitted distributions need not be absolutely continuous relative to the continuous noisy-state law; chi-square distance can be infinite. No finite distance is measured. **Fix:** separate the abstract theorem from its applicability, account for block-fit error, and establish a finite discrepancy before invoking its quantitative bound.

8. **Major — Theorem B’s “MEASURED within 0.6–1.0” label tests the wrong approximation.** `n6_flips.py` measures
   \[
   \frac{\mathbb E\|F(z+\Delta)-F(z)\|^2}{\mathbb E\|J(z)\Delta\|^2},
   \]
   not the replacement of the denominator by \(\operatorname{tr}(M\operatorname{Cov}\Delta)\). Its perturbations are random residual directions scaled by \(\|z-\mu\|\), not actual PB scrambles. **Fix:** measure both approximation steps separately on real family couplings. For arbitrary couplings, independence also requires using \(\mathbb E[\Delta\Delta^T]\), unless \(\mathbb E\Delta=0\).

9. **Major — §§1–2: noise-integrated and pathwise Jacobians are conflated.** For the defined \(F(z)=\mathbb E_\xi f(z,\xi)\), \(J_F=\mathbb E_\xi J_\xi\), whereas the numerics estimate \(\mathbb E_{z,\xi}J_\xi^TJ_\xi\). These differ by a positive-semidefinite conditional variance term. **Fix:** distinguish the two operators; justify the pathwise one as a surrogate or coupling bound, rather than the exact sensitivity of \(F\).

10. **Major — §6, fit scripts: \(R^2=0.96\) supports a descriptive association, not the predictions made from it.** The ten observations are correlated family means from four shared nets, including four hard-coded, rounded R1 means. Selection uses the response threshold \(nMSE>2\times10^{-6}\); PB64 and PB128 are excluded. I reproduce **\(R^2=0.768\)** on the 25 individual net/family observations above that threshold. The added bootstrap resamples family means, not independent nets, so its intervals do not measure between-network uncertainty. **Fix:** retain per-net data, model measurement floors, bootstrap by net, and validate on held-out nets and actual whitened-block families.

11. **Major — §6: the mechanism interpretation does not follow from the fitted slope.** Varying \(k\) changes the perturbation’s geometry, gate correlations and effective curvature, not just its amplitude. Thus \(\beta\approx1\) does not reject every second-order mechanism. Also, the mean/pathwise-RMS ratio is not simply \(\sqrt C\): it includes the normalization \(2\operatorname{tr}(M\Sigma_P)/\operatorname{tr}\operatorname{Var}(a_L)\). **Fix:** conduct an amplitude sweep for a fixed perturbation family and calculate that ratio directly.

12. **Major — §7, `n4_classes.py`: cross-fitting is sound, but it cannot establish a class-count lower bound.** The pilot and train/test halves are genuinely separated. However, test error of an estimated class block is an **upper bound on achievable error**, not a lower bound on the population optimum. At \(K=64\), training averages only about 64 Jacobians per class. Sparse classes are omitted: GATE-64 retains 63/64 classes for net 113 and 62/64 for net 114. **Fix:** report coverage and uncertainty, include fallback blocks for omitted classes, and measure convergence with training size.

13. **Major — §7: the single-path comparison uses a different normalization.** The class statistic is a ratio of expected residual energy to expected total energy; `single_path_tail` is the mean of individual residual-energy fractions. Theorem D compares unnormalized expectations, so those two displayed ratios are not its direct endpoints. **Fix:** use trace-weighted single-path tails for the theorem comparison.

14. **Major — §7: the \(10^7\)–\(10^{10}\) extrapolation is unsupported, and “near-full” is numerically wrong.** A participation ratio is not a quantization dimension. Clustering only eight PCs can leave an irreducible angular floor; measurements through \(K=64\) do not justify a power law over another six to nine orders of magnitude. Moreover, at disagreement \(1/(2L')\), the report’s own coherence sum is only **0.640 \(L'\)** for \(L'=32\), not near-full. **Fix:** label the counts illustrative scenarios, define the desired coherence fraction, and test additional dimensions, classings and sample sizes. Remove “not at any practical number of classes.”

15. **Major — §10.1: composition counts are forecasts, not necessary dimensions.** The hardness report’s exact telescoping identity sums **signed error vectors**; squared errors have cross terms and can cancel or reinforce. Two PB64/PB128 composition measurements do not establish additive nMSE, a universal per-layer budget, or the revised 90–130-PC/65–95-optimal requirement. **Fix:** measure transported local error vectors and their cross terms, then test the proposed blocks in composition. Also remove “as an estimator must”: layerwise reprojection is one estimator class.

16. **Major — §9: finite-width stability is not \(O(1)\) scaling.** Two widths for \(C\), three widths for a sparse PB ladder, and floors near the target cannot distinguish bounded growth from slow growth. The \(C\propto1/n\) mechanism is correctly called conjectural, but the asymptotic dimension claim remains stronger than the observations. **Fix:** say “roughly stable over widths 128–512,” with uncertainty and fixed error targets above the floors.

17. **Major — §10.2: both QMC arguments are insufficient.** Participation ratio 5–10 does not establish the dimension-uniform weight conditions required by weighted QMC theory. For example, eigenvalues \(1,1/n,\ldots,1/n\) have bounded participation ratio while retaining substantial mass across \(n\) directions. ReLU kinks also do not exclude every useful QMC or smoothing result; a constant-factor Kerdock measurement establishes no rate ceiling. **Fix:** identify the actual function space, derivative bounds and weights, or leave QMC open.

18. **Major — Theorem C versus its summary: preserve the qualifications.** The Gaussian conditional-independence argument and decomposition are valid with suitable measurable bases on surviving paths. But **\(\widehat J\)** is the Gaussian product; \(J\) includes the radial row. Conditional on survival, the active counts are truncated binomials, not unconditioned binomials. Half-width square asymptotics additionally require the initial active fraction to approach \(1/2\). Noise agreement is empirical; an \(O(\sigma)\) leakage claim needs norm and probability qualifications. **Fix:** state these conditions consistently, especially in the summary.

19. **Minor — Theorem B and D: strict equality characterizations need spectral-gap hypotheses.** PBW can be optimal without the selected \(V_k\) being \(\Sigma\)-invariant when \(S\) has ties; \(S=I\) is a counterexample. Likewise, D requires a common **maximizing** \(k\)-subspace, not uniquely identical top-\(k\) subspaces. **Fix:** add a gap assumption or use the common-maximizer formulation.

20. **Minor — §§5–6,9: several numerical descriptions overstate precision.** “Dead” directions are zero in the sampled Jacobians, not proved inactive under the noisy population law. Tails do not agree to two digits across horizons: net 114’s optimal \(k=16\) tail changes **0.0518→0.0422**. VJP/full-Jacobian PB32 tails for net 113 are **0.0800655 versus 0.0803615**, not three-significant-digit agreement. **Fix:** use “empirically inactive,” quantify horizon variation, and report estimator discrepancies.

21. **Minor — §6 and artifact references: tighten reproducibility.** Using the stated \(C=2.7\times10^{-4}\), dense saved spectra give optimal dimensions **28/18** and **57/38**, and PC dimensions **51/33** and **98/71**, for the two targets; the report gives rounded estimates including approximately 90 instead of 98. `logs/a4_fit.log` and `logs/a5_width.log` were absent; the available width log is `a5_width_partial.log`. The older Gaussian log also contains invalid FC counts, although the report correctly uses the repaired `fc_counts.log`. **Fix:** link exact artifacts and distinguish recomputed thresholds from interpolations.

The following checks passed. Theorem B’s Schur/Ky Fan optimization and Theorem D’s inequality are correct with the qualifications above. Newman's Gaussian-product exponents and the displayed iterated-limit edge calculation are consistent. **All three one-layer Wick formulas are correct for a fixed square \(Y\) and Gaussian \(W\) independent of the masks**; conditioning on actual ReLU masks does not satisfy that assumption. An independent nonsymmetric-\(Y\) check gave approximately \(420.63,142.52,198.84\), against exact values \(420,142,198\).

I spot-checked substantially more than ten reported numbers:

| Location / source | Recomputed or artifact value | Assessment |
|---|---:|---|
| §3, `a2_theory.log`, independent quadrature | \(G(e^{-1})=0.794019;\ G(0.01)=6.334849\) | Matches |
| §3, same | \(33\nu_{32}(0.01)=5.875003\) | Matches 5.88 |
| §4, FC calculation | Half-width count \(22.787891\) | Matches 22.8 |
| §4, noisy/noiseless `n2` JSON | Counts \(24.484375/24.359375\) | Matches 24.5/24.4 |
| §5, `a2_theory.log` | PR(single) 4.38; PR(M) 20.61; prediction 18.72 | Matches |
| §5, saved matrix, thresholds recomputed | \(k_F^*=63/142/186\) | Matches |
| §6, `a1_tails_a.log`, \(k=32\) | Optimal 0.0277; PC 0.0804; PBW proxy 0.0581 | Matches |
| §6, reconstructed ten-point fit | \(\beta=0.997667,\ C=2.68649\times10^{-4}, R^2=0.955920\) | Matches |
| §6, same fit | RMS residual factor 1.23956 | Matches 1.24 |
| §7, net-113 class JSON, \(k=32\) | Global 0.200178; KM64 0.174765 | Matches |
| §7, same | Class state-weighted tail 0.013694 | Matches 0.014 |
| §7, `n5` JSON | Global angle proxy 0.284875; KM64 0.207601 | Matches rounding |
| §8, `n6` log, \(\epsilon=0.01\) | Nonlinearity 0.169; energy ratio 0.979 | Matches |
| §9, four-net JSON means | State PR 9.67440/11.31215 at widths 128/256 | Matches |

**The summary for Kelvin is not accurate as a statement of established limits.** Its defensible core is that measured individual Jacobians concentrate strongly, input averaging broadens sensitivity, and larger empirical joint blocks improve these tested families. It presents heuristic proportional-depth counts, mismatched-tail predictions, speculative class counts and an unsupported sampling barrier as conclusions. Retain the empirical observations, distinguish the valid conditional algebra from the forecasts, and state that the minimum joint dimension and best achievable estimator complexity remain open.
tokens used
146,794
Major revision required. I reviewed the changing draft through **19:53 UTC, 29 October 2026**. No files were edited.

1. **Critical — §§3–4, C2, summary: proportional-depth counts are not proved.** Fixed-\(N\) Fuss–Catalan convergence, followed by \(N\to\infty\), does not establish the simultaneous limit \(N/n\to\tau>0\). Here the predicted surviving count is \(O(1)\), so even convergence of the empirical spectral distribution is insufficient. The revised §11 correctly calls this heuristic, but the summary and “our regime is the free one” still overclaim. **Fix:** distinguish the iterated-limit theorem from the measured proportional-depth approximation; obtain a joint-limit edge/count result before claiming a width-independent theorem.

2. **Critical — C2’s newly added radial-row justification is false.** \(r=J^Te_N\) includes accumulated angular-to-radial leakage; its norm is not merely the forward radial gain. Conditional on balanced active counts \(m=n/2\), the triangular decomposition gives
   \[
   \mathbb E\|r_N\|^2=\sum_{j=0}^{N}\left(\frac{m-1}{m}\right)^j,
   \]
   which is approximately \(N+1\), not order one uniformly in depth; it is **29.19 at \(m=128,N=32\)**. Interlacing controls counts at an absolute threshold, not thresholds normalized by two different top singular values. **Fix:** prove control of \(s_1(J)/s_1(\widehat J)\); remove the current no-outlier argument.

3. **Critical — §6, `a1_tails.py`, `a4_fit.py`: the predictors describe different families from those measured.** The scripts use \(P\operatorname{Cov}(z)P\), with the mean direction projected out. The actual PB families use **radius-weighted PCs of \(w=z/\|z\|\)**; PBW uses weighted direction covariance, while PBS/PBA also use different subspaces from those reconstructed here. Carrying an independent radius is not carrying the linear functional \(u^Tz\) exactly. Even exact conditioning on \(u^Tz\) would generally require a Schur complement, not orthogonal projection. **Fix:** calculate tails in the actual chart, with the actual fitted blocks and induced perturbations. Until then, the “optimal block needs 20–30 directions” figures are extrapolations from a mismatched proxy.

4. **Critical — §10.2–10.3 and summary: no sampling lower bound follows.** Large joint dimension and failed closures do not establish that every deterministic method costs exponentially, or that beating sampling requires an exactly preserved closed-form family. The displayed grid complexity is a cost for a particular approximation scheme, not a lower bound. Uncovered routes include approximate nonlinear latent closures, backward-observable approximation, conditional integration, control variates, adaptive cubature, and bias correction. Gate entropy and pairwise independence do not exclude these. **Fix:** replace “nothing beats sampling unless…” with a statement about the tested families and explicitly leave algorithmic complexity open.

5. **Major — C1: the conditioned-chain repair remains mathematically inconsistent.** At fixed width, the physical noiseless network eventually dies almost surely; conditioning merely on non-extinction still permits narrow layers that permanently destroy angular ranks. Conditioning on every \(m_\ell>k\) changes **all** terms in the \(k\)-volume formula to expectations under \(m\mid m>k\). Using \(m\mid m>j\) separately for each exponent describes different chains, not one spectrum. **Fix:** state a finite-horizon conditional volume identity with one common truncation, or define a modified chain with a fixed minimum width. Do not label these the physical network’s infinite-depth exponents.

6. **Major — Theorem A: the revised “supremum” claim is still false at fixed radius.** The identity and Cauchy–Schwarz inequality are correct under integrability assumptions. But for \(P=N(0,1)\), no block and \(F(z)=z\), equality requires \(dQ/dP=1+cz\), which is negative somewhere for every nonzero \(c\). The bound cannot even be approached at a fixed positive chi-square radius: near-equality would force convergence in \(L^2(P)\) to that invalid density. **Fix:** claim local asymptotic sharpness as the radius tends to zero, using truncated **conditionally recentered** scores. State exact attainment only under suitable one-sided boundedness.

7. **Major — §§2,8: Theorem A is not an established bound for the fitted empirical families.** Their block laws are empirical approximations, and their discrete fitted distributions need not be absolutely continuous relative to the continuous noisy-state law; chi-square distance can be infinite. No finite distance is measured. **Fix:** separate the abstract theorem from its applicability, account for block-fit error, and establish a finite discrepancy before invoking its quantitative bound.

8. **Major — Theorem B’s “MEASURED within 0.6–1.0” label tests the wrong approximation.** `n6_flips.py` measures
   \[
   \frac{\mathbb E\|F(z+\Delta)-F(z)\|^2}{\mathbb E\|J(z)\Delta\|^2},
   \]
   not the replacement of the denominator by \(\operatorname{tr}(M\operatorname{Cov}\Delta)\). Its perturbations are random residual directions scaled by \(\|z-\mu\|\), not actual PB scrambles. **Fix:** measure both approximation steps separately on real family couplings. For arbitrary couplings, independence also requires using \(\mathbb E[\Delta\Delta^T]\), unless \(\mathbb E\Delta=0\).

9. **Major — §§1–2: noise-integrated and pathwise Jacobians are conflated.** For the defined \(F(z)=\mathbb E_\xi f(z,\xi)\), \(J_F=\mathbb E_\xi J_\xi\), whereas the numerics estimate \(\mathbb E_{z,\xi}J_\xi^TJ_\xi\). These differ by a positive-semidefinite conditional variance term. **Fix:** distinguish the two operators; justify the pathwise one as a surrogate or coupling bound, rather than the exact sensitivity of \(F\).

10. **Major — §6, fit scripts: \(R^2=0.96\) supports a descriptive association, not the predictions made from it.** The ten observations are correlated family means from four shared nets, including four hard-coded, rounded R1 means. Selection uses the response threshold \(nMSE>2\times10^{-6}\); PB64 and PB128 are excluded. I reproduce **\(R^2=0.768\)** on the 25 individual net/family observations above that threshold. The added bootstrap resamples family means, not independent nets, so its intervals do not measure between-network uncertainty. **Fix:** retain per-net data, model measurement floors, bootstrap by net, and validate on held-out nets and actual whitened-block families.

11. **Major — §6: the mechanism interpretation does not follow from the fitted slope.** Varying \(k\) changes the perturbation’s geometry, gate correlations and effective curvature, not just its amplitude. Thus \(\beta\approx1\) does not reject every second-order mechanism. Also, the mean/pathwise-RMS ratio is not simply \(\sqrt C\): it includes the normalization \(2\operatorname{tr}(M\Sigma_P)/\operatorname{tr}\operatorname{Var}(a_L)\). **Fix:** conduct an amplitude sweep for a fixed perturbation family and calculate that ratio directly.

12. **Major — §7, `n4_classes.py`: cross-fitting is sound, but it cannot establish a class-count lower bound.** The pilot and train/test halves are genuinely separated. However, test error of an estimated class block is an **upper bound on achievable error**, not a lower bound on the population optimum. At \(K=64\), training averages only about 64 Jacobians per class. Sparse classes are omitted: GATE-64 retains 63/64 classes for net 113 and 62/64 for net 114. **Fix:** report coverage and uncertainty, include fallback blocks for omitted classes, and measure convergence with training size.

13. **Major — §7: the single-path comparison uses a different normalization.** The class statistic is a ratio of expected residual energy to expected total energy; `single_path_tail` is the mean of individual residual-energy fractions. Theorem D compares unnormalized expectations, so those two displayed ratios are not its direct endpoints. **Fix:** use trace-weighted single-path tails for the theorem comparison.

14. **Major — §7: the \(10^7\)–\(10^{10}\) extrapolation is unsupported, and “near-full” is numerically wrong.** A participation ratio is not a quantization dimension. Clustering only eight PCs can leave an irreducible angular floor; measurements through \(K=64\) do not justify a power law over another six to nine orders of magnitude. Moreover, at disagreement \(1/(2L')\), the report’s own coherence sum is only **0.640 \(L'\)** for \(L'=32\), not near-full. **Fix:** label the counts illustrative scenarios, define the desired coherence fraction, and test additional dimensions, classings and sample sizes. Remove “not at any practical number of classes.”

15. **Major — §10.1: composition counts are forecasts, not necessary dimensions.** The hardness report’s exact telescoping identity sums **signed error vectors**; squared errors have cross terms and can cancel or reinforce. Two PB64/PB128 composition measurements do not establish additive nMSE, a universal per-layer budget, or the revised 90–130-PC/65–95-optimal requirement. **Fix:** measure transported local error vectors and their cross terms, then test the proposed blocks in composition. Also remove “as an estimator must”: layerwise reprojection is one estimator class.

16. **Major — §9: finite-width stability is not \(O(1)\) scaling.** Two widths for \(C\), three widths for a sparse PB ladder, and floors near the target cannot distinguish bounded growth from slow growth. The \(C\propto1/n\) mechanism is correctly called conjectural, but the asymptotic dimension claim remains stronger than the observations. **Fix:** say “roughly stable over widths 128–512,” with uncertainty and fixed error targets above the floors.

17. **Major — §10.2: both QMC arguments are insufficient.** Participation ratio 5–10 does not establish the dimension-uniform weight conditions required by weighted QMC theory. For example, eigenvalues \(1,1/n,\ldots,1/n\) have bounded participation ratio while retaining substantial mass across \(n\) directions. ReLU kinks also do not exclude every useful QMC or smoothing result; a constant-factor Kerdock measurement establishes no rate ceiling. **Fix:** identify the actual function space, derivative bounds and weights, or leave QMC open.

18. **Major — Theorem C versus its summary: preserve the qualifications.** The Gaussian conditional-independence argument and decomposition are valid with suitable measurable bases on surviving paths. But **\(\widehat J\)** is the Gaussian product; \(J\) includes the radial row. Conditional on survival, the active counts are truncated binomials, not unconditioned binomials. Half-width square asymptotics additionally require the initial active fraction to approach \(1/2\). Noise agreement is empirical; an \(O(\sigma)\) leakage claim needs norm and probability qualifications. **Fix:** state these conditions consistently, especially in the summary.

19. **Minor — Theorem B and D: strict equality characterizations need spectral-gap hypotheses.** PBW can be optimal without the selected \(V_k\) being \(\Sigma\)-invariant when \(S\) has ties; \(S=I\) is a counterexample. Likewise, D requires a common **maximizing** \(k\)-subspace, not uniquely identical top-\(k\) subspaces. **Fix:** add a gap assumption or use the common-maximizer formulation.

20. **Minor — §§5–6,9: several numerical descriptions overstate precision.** “Dead” directions are zero in the sampled Jacobians, not proved inactive under the noisy population law. Tails do not agree to two digits across horizons: net 114’s optimal \(k=16\) tail changes **0.0518→0.0422**. VJP/full-Jacobian PB32 tails for net 113 are **0.0800655 versus 0.0803615**, not three-significant-digit agreement. **Fix:** use “empirically inactive,” quantify horizon variation, and report estimator discrepancies.

21. **Minor — §6 and artifact references: tighten reproducibility.** Using the stated \(C=2.7\times10^{-4}\), dense saved spectra give optimal dimensions **28/18** and **57/38**, and PC dimensions **51/33** and **98/71**, for the two targets; the report gives rounded estimates including approximately 90 instead of 98. `logs/a4_fit.log` and `logs/a5_width.log` were absent; the available width log is `a5_width_partial.log`. The older Gaussian log also contains invalid FC counts, although the report correctly uses the repaired `fc_counts.log`. **Fix:** link exact artifacts and distinguish recomputed thresholds from interpolations.

The following checks passed. Theorem B’s Schur/Ky Fan optimization and Theorem D’s inequality are correct with the qualifications above. Newman's Gaussian-product exponents and the displayed iterated-limit edge calculation are consistent. **All three one-layer Wick formulas are correct for a fixed square \(Y\) and Gaussian \(W\) independent of the masks**; conditioning on actual ReLU masks does not satisfy that assumption. An independent nonsymmetric-\(Y\) check gave approximately \(420.63,142.52,198.84\), against exact values \(420,142,198\).

I spot-checked substantially more than ten reported numbers:

| Location / source | Recomputed or artifact value | Assessment |
|---|---:|---|
| §3, `a2_theory.log`, independent quadrature | \(G(e^{-1})=0.794019;\ G(0.01)=6.334849\) | Matches |
| §3, same | \(33\nu_{32}(0.01)=5.875003\) | Matches 5.88 |
| §4, FC calculation | Half-width count \(22.787891\) | Matches 22.8 |
| §4, noisy/noiseless `n2` JSON | Counts \(24.484375/24.359375\) | Matches 24.5/24.4 |
| §5, `a2_theory.log` | PR(single) 4.38; PR(M) 20.61; prediction 18.72 | Matches |
| §5, saved matrix, thresholds recomputed | \(k_F^*=63/142/186\) | Matches |
| §6, `a1_tails_a.log`, \(k=32\) | Optimal 0.0277; PC 0.0804; PBW proxy 0.0581 | Matches |
| §6, reconstructed ten-point fit | \(\beta=0.997667,\ C=2.68649\times10^{-4}, R^2=0.955920\) | Matches |
| §6, same fit | RMS residual factor 1.23956 | Matches 1.24 |
| §7, net-113 class JSON, \(k=32\) | Global 0.200178; KM64 0.174765 | Matches |
| §7, same | Class state-weighted tail 0.013694 | Matches 0.014 |
| §7, `n5` JSON | Global angle proxy 0.284875; KM64 0.207601 | Matches rounding |
| §8, `n6` log, \(\epsilon=0.01\) | Nonlinearity 0.169; energy ratio 0.979 | Matches |
| §9, four-net JSON means | State PR 9.67440/11.31215 at widths 128/256 | Matches |

**The summary for Kelvin is not accurate as a statement of established limits.** Its defensible core is that measured individual Jacobians concentrate strongly, input averaging broadens sensitivity, and larger empirical joint blocks improve these tested families. It presents heuristic proportional-depth counts, mismatched-tail predictions, speculative class counts and an unsupported sampling barrier as conclusions. Retain the empirical observations, distinguish the valid conditional algebra from the forecasts, and state that the minimum joint dimension and best achievable estimator complexity remain open.
