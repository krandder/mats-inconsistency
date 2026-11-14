codex
1. critical — The measured synchronization collapse does not establish collapse of the population operator.
Location: report lines 22–24, 56–57, 262–264, 324–347; runs/analysis_cut_n8_c64_L64.json:7–8; runs/analysis_cut_n16_c128_L16.json:7–8.
Problem: These ensembles are severely undersampled in energy. Recomputing from the raw NPZ files, (8,64,64) has mean output second moment approximately 8.12e-12, whereas its exact ensemble expectation is 1. One sampled back contributes 99.7% of that energy. At (16,128,16), the second moment is approximately 1.92e-4 and the effective number of energy-bearing fronts is approximately 4, despite F=512. Moreover, the true Gaussian-input Vbar is at least 0.06044 for n=8 and 0.03075 for n=16, far above the reported deep-run estimates. These are direct signs that rare, high-energy networks are missing.
Fix: Describe collapse as a property of the sampled matrices. Withdraw the population conclusion until independent ensemble replications or a justified rare-event estimator resolve the energy and normalized tails. Holding out fronts while retaining the same backs does not fix this.

2. critical — ARC’s conjectured error upper bound is used as a lower bound on required computation.
Location: report lines 78–79 and 447–453; notes/arc_moment.md:5–17.
Problem: MSE <= c_K(L/n)^K supplies a sufficient error guarantee, not a necessary order K. It cannot prove that the hierarchy “loses to sampling.” The calculation also drops K-dependent error and cost constants while taking K→infinity, and requires 1<L<n. For L>=n, the displayed logarithmic inversion does not give a convergent guarantee.
Fix: Label this an illustrative cost calculation assuming a sharp geometric error law and controlled K-dependent constants. A lower-bound conclusion requires additional assumptions or a separate theorem.

3. major — Theorem 1 loses a factor of n.
Location: report lines 129–135 and 143–152; compare coprover/astra_r1_answer.md:179–201 and 236–249.
Problem: Under the stated He scaling and averaging over output neurons,
E_{b,j} g_j(h)^2 = ||h||^2/n.
Consequently, the report cannot simultaneously use K(h,h')=||h||||h'||κ(u·u'), sum a_l=1, and moments E[||h||Y_l]. Its displayed squared-singular-value operator is too large by n.
Fix: Use K=||h||||h'||κ/n, κ(1)=1, and moments E[(||h||/sqrt(n))Y_l], with the integrand zero at h=0. Define Q=E||h||²/n separately from the ray quantity β=E[(E_x||h||)^2]/n. Lemma 2 concerns the vector output at nonzero h; averaging a scalar output derivative introduces 1/n. With these conventions, the gradient identity, harmonic tail bound, and truncation argument survive.

4. major — The ray theorem omits essential hypotheses and mixes asymptotic regimes.
Location: report lines 16–19, 193–205, 207–212, and 473–475.
Problem: The exact constant in Theorem E assumes R>=0, E R=1, and 0<Var R<infinity. With mean μ instead, the relevant radial parameter is Var(R)/μ². The precision exponent is an asymptotic at fixed dimension and fixed finite architecture; it does not establish a uniform exp(Θ(n log(1/eps))) law as n and depth grow. The face extension also requires face probabilities, squared-radial-mean weighting, and absolute error budgets. Approximate angular synchronization alone does not supply these.
Fix: State the radial normalization and fixed-(n,d) limit directly in the theorem and summary. Give the face-conditioned statement separately. Retain the energy-weighted approximation requirement wherever applying it to approximately synchronized fronts. The stronger E′ exponent 2(n−1)/3 is defensible; the earlier 2(n−3)/3 argument is weaker and its dropped q factor was correctly identified.

5. major — The claim that a fixed-architecture lower bound for genuine Gaussian fronts remains open misses a rare-event proof.
Location: report lines 80–81, 374–405, 442–443, and 483; coprover/astra_r3_answer.md:2.
Problem: For c>=3, a positive-probability sign event can force exact synchronization before the final front layer. Require all but one row of W_{c−1} to be strictly negative and the remaining row strictly positive. Since h_{c−2} is nonnegative, h_{c−1}(x)=a(x)e_1. Earlier layers can be restricted to a positive-probability event ensuring E_x a>0. The fresh W_c then gives h_c(x)=a(x)ReLU(W_c e_1). Conditional on its first column being positive, the direction is uniform on the full-dimensional positive orthant and independent of its norm.

Restricting front integration to this event gives a population error lower bound equal to its probability times the conditional lower bound. The existing orthant-ray and depth-ordering arguments therefore yield Ω_{n,c,d}(eps^{-2(n−1)/3}) for genuine Gaussian fronts at fixed architecture. The polyhedral harmonic upper bound plus Jensen supplies the matching upper exponent.
Fix: Add this rare-event consequence, with its potentially disastrous constants. What remains open is a useful precision window, uniform joint-limit bounds, or typical-network hardness. This argument does not refute the fixed-eps polynomial-in-width upper bound.

6. major — “k* averages out” confuses orientation with singular-value information.
Location: report lines 34–38, 125–138, and 476–477.
Problem: Rotation invariance removes preferred directions from the covariance kernel. It does not erase Jacobian singular values: k* itself is invariant under rotations. Its dependence on width and depth can remain correlated with the kernel spectrum. FORMAL.md:75 makes the more careful statement that such dependence can enter through κ.
Fix: Say that pathwise orientations disappear and k* alone does not determine cut rank. The truncation argument refutes the universally quantified T1; it does not establish statistical independence from k* or refute every narrower depth/precision version of the original premise.

7. major — The smooth-front theorem does not make singularity necessary.
Location: report lines 20–21, 169–175, and 478.
Problem: The proved sufficient condition is E_f||q_f||²_{L²(μ)}<infinity for a common angular measure. Its failure does not imply singular front laws. Absolutely continuous laws can fail L² integrability, and individually smooth laws can have infinite ensemble-average squared density norm.
Fix: State the sufficient condition exactly. Replace “any hardness of this kind must come from singular cut laws” with “this hardness requires failure of the common-L²-density condition.”

8. major — Bias directions are presented as population bounds and a validated noise floor.
Location: report lines 221–228, 281–284, and 337; PREREG.md:25–29; scratch/score_prereg.py:3–4.
Problem: The pair Gram is correctly unbiased conditional on sampled fronts and backs. Optimized in-sample tails are downward biased, and independently evaluated subspaces have upward-biased residuals, in expectation. These facts do not make realized ranks confidence bounds. Thus “population ranks are bounded only from below” and “at most 41 held out” are unsupported.

The controls give tail_in(1)/Vbar approximately −8.63e-5 and −1.10e-4, with cross-fitted values +1.52e-4 and +1.95e-4. Two single-front controls do not establish uncertainty across configurations, ranks, or heavy-tailed ensembles. The registered signed floor is ill-defined for these negative values; the scorer documents a magnitude interpretation but implements no control-based exclusion.
Fix: Distinguish expected bias from statistical coverage. Label the magnitude reinterpretation post hoc, repeat matched controls, and report uncertainty. Treat unresolved held-out ranks as failures of the tested learned subspace, not lower bounds on optimal population rank.

9. major — Fixed-back readouts use spherical inputs instead of the declared Gaussian inputs.
Location: scratch/run_cut.py:104–125; report lines 254, 349–351, and 365–370.
Problem: R2 and R3 threshold outputs evaluated on unit input directions. The Gaussian radius is correctly integrated for means, but is omitted for indicators. Homogeneity does not permit dropping it from 1[R a(u)>t]. These are probabilities for a different input distribution, affecting P5 and the fixed-back conclusions.
Fix: Integrate the chi-distributed radius through each threshold event, or sample it independently. Recompute pilot thresholds, event probabilities, and variances under the same Gaussian input law. Alternatively, relabel these experiments explicitly as spherical-input experiments.

10. major — Several deviations from the pre-registration are unreported.
Location: PREREG.md:30–43 and 82–88; scratch/run_cut.py:107–110; scratch/analyze.py:35–36; scratch/crux_local.py:22; scratch/queue1.txt and scratch/queue2.txt.
Problem: R2 uses zero plus positive-output quantiles instead of the registered unconditional 1/9,…,8/9 quantiles. Moment regression fits and scores using half-splits rather than the registered response/scoring allocation. The n=16 crux omits registered k=12,16. The (8,64,8) run uses F=512,C=512 although Amendment 1 specifies F=256,C=256 for that configuration.
Fix: Add these to the deviation ledger and distinguish literal registered tests from tests on modified measurements. The scoring script reproduces the reported P1–P8 outcomes, but that does not establish adherence of the upstream measurements.

11. major — The per-network experiment does not evaluate “the best summary” or an independently fixed protocol.
Location: report lines 68–69, 255–256, and 363; scratch/analyze.py:105–110; lib/spec.py:87–96.
Problem: The evaluated predictor is a rank-s SVD approximation of noisy split 0, fitted on the same fronts and backs being scored. It is not the optimal population protocol. Additionally, s is chosen from r_in using all splits, so the claimed independence of the scoring splits fails for this data-dependent choice. Thresholding noisy unbiased squared-error estimates does not give an unbiased failure fraction. For r_in=1, max(1,r_in//2) evaluates rank 1, although the registered floor(r_in/2) is 0.
Fix: Describe the existing statistic accurately. Choose rank independently, fit a deployable protocol, and evaluate fresh front/back pairs with enough integration accuracy to classify failures. Correct the rank-zero case.

12. major — The effective-degree numbers do not support the crux headline.
Location: report lines 62–65 and 306–311; runs/crux_n16_c8_L8.json:127–138.
Problem: Inverting binom(k+D,k)=r_loc gives, for (16,8,8), D approximately 4.00,4.00,3.89,3.76 at eps=0.1 and 9.00,9.28,7.77,6.48 at eps=0.03 for k=1,2,3,4. For (16,16,16), the k=4 values are approximately 3.18 and 5.39. These contradict the stated ranges 4.0–5.4 and 9–14 for k<=4. At k=4, the first example’s degree ratio is 1.72, not 2.2–2.5.

Furthermore, assigning an effective D by inverting a count does not show that polynomial features achieve that rank or establish an eps^{-2/3} law.
Fix: Correct the numerical ranges. Label D as a descriptive parameter, and validate polynomial approximation directly on held-out points before claiming the crux lemma or gate-kink scaling has been measured.

13. major — The relevant-versus-random comparison is not consistently a comparison of k-dimensional slices.
Location: scratch/crux_local.py:41–60; report lines 66 and 312–315.
Problem: The random control uses only min(k,number of active coordinates) dimensions, padding the rest with zeros, while “relevant” directions retain k SVD vectors, including null directions. Reconstructing the seeded designs: at n=16,k=8, 91/256 controls have fewer than eight active coordinates, and 255/256 reference Jacobians have rank below eight. At n=32,k=16, the corresponding counts are 110/256 and 256/256. The cases also normalize by their own different Vloc. Finally, QR without diagonal-sign correction does not produce the claimed Haar rotation on O(k).
Fix: Match actual slice dimensions, specify how null directions are handled, use a common error-budget comparison alongside relative ranks, and correct the Haar construction. Similar normalized counts do not establish that backward selection “adds no complexity.”

14. major — The ARC K=2 comparison does not isolate nonlinear readout.
Location: report lines 76–77 and 462–466; notes/arc_moment.md:42–50.
Problem: At n=32 and n=64, this lane computes only first-moment features. These finite-sample least-squares predictors are compared with ARC’s second-order closure. The moments themselves are Monte Carlo estimates, and fitted held-out error is not the error of the best linear predictor from exact moments. The cited ARC figures are also reported as median normalized errors from other experiments, unlike this lane’s aggregated error/Vbar.
Fix: Withdraw “15–70x better than the best linear readout of the exact cut moments.” Compare matched second-order information, networks, normalization, and error aggregation, with learning curves that separate fitting error from approximation error.

15. major — Summary dimension is repeatedly converted into a computational comparison with sampling.
Location: report lines 18–19, 74–75, 272–273, and 470–475.
Problem: The theorem compares linear-summary coordinates with a sample count. It supplies no feature-construction or decoding costs. A stored cut sample has n coordinates, and sampling uses a decoder outside the linear class. ARC’s cited L² sketching guarantee also uses a head-density L² normalization, not eps²Vbar; ideal ray laws do not satisfy that density assumption.
Fix: Say that, at fixed n in the stated ray model, linear-summary dimension eventually grows faster than Monte Carlo sample count. Do not call this a runtime lower bound or a failure of ARC’s sketching guarantee without aligning the computational model and normalization.

16. major — The summary mishandles the fixed-network quantifiers.
Location: report lines 67–71; compare lines 356–360.
Problem: No protocol-uniform second-moment bound for Z is proved, so Paley–Zygmund does not supply the asserted quantitative failure fraction. It also concerns Z relative to E Z, not automatically Z relative to the random V(f,b). The sentence “Every summary fails on most nets is impossible” reverses the relevant quantifiers: “for every predetermined summary, most nets defeat it” can be true. The impossible version is “on most nets, every retrospectively chosen summary fails.”
Fix: Restore the rank restriction, moment/small-ball assumptions, and quantifier order in the summary. Keep the measured fractions separate from this conditional theorem.

17. minor — Several summary and table entries are incorrect.
Location: report lines 22, 243, 254, and 272–273.
Problem: The range at eps>=0.03 is not 1–117: the table reaches 245 at eps=0.03. The (16,128,16) R3 ranks are 36,103,46,37, whose median is 41.5, not the displayed 46. “Fewer numbers than the 100–1100 samples matched” is false for the reported rank 117 at eps=0.1.
Fix: Generate these statements from the JSON and preserve half-integer medians or declare a rounding rule.

18. minor — Corrected mathematics is not reflected in the designated formal statement or executable cap calculation.
Location: FORMAL.md:77–84; scratch/toy_cap.py:29–56; logs/toy_cap.log.
Problem: FORMAL.md still contains the rejected rank inequality counting null odd degrees and the entire boundary shell. The executable cap calculation still uses unstable quadrature and a truncated variance calculation. Its logged ranks 713598 and 47056446 differ from the report’s corrected 713593 and 46983832. Independent recomputation with the closed-form spectrum confirms the report’s corrected values.
Fix: Preserve the hashed original as a historical document, but mark the lemma superseded and provide a corrected formal version. Update the executable calculation and regenerate its log.

19. minor — The kink experiment measures a finite-angle statistic, not the asymptotic kink coefficient.
Location: report lines 331–334 and 414; scratch/kink_depth.py:3–7; logs/kink_depth_n16.log.
Problem: The interpretation requires theta much smaller than 1/d, but the quoted series uses theta=0.1 through d=32. The smaller-angle estimates have substantial uncertainty. The data therefore do not identify the asymptotic coefficient or establish the stated causal explanation of rank collapse.
Fix: Report the measured finite-angle deficit, retain uncertainty, and separate it from the valid expected gate-jump identity.

Verdict: Substantial revision required. The normalized harmonic factorization, gradient identity, truncation refutation of T1, block lemma, depth ordering, and fixed-architecture ray exponents survive review. The population-collapse, crux, and ARC conclusions substantially exceed their evidence.
Verification: supplied estimator tests passed; five raw-data rank recomputations matched the JSON; scoring outcomes and regression R² values reproduced; current and reconstructed preregistration hashes matched. No files were modified.
