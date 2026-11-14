Locations below use “report” for tractable-gates-2026-10-31.md, version modified 2026-10-31 04:25:17 UTC. Other paths are relative to work/tractable-gates.

1. [major] The “any front” polylogarithmic claim drops essential assumptions.

Location: report Summary item 1 and §2; notes/theory.md:23–33, 61–64.

Lemma P is valid under its stated conditions. Lemma R is also valid, but is an upper bound, not the “exact reduction” claimed in the report. Neither converts fixed-scale Gaussian Hermite decay into polylogarithmic rank for arbitrary front laws. That requires the bounded-scale/inverse-moment assumptions, or quantitative scale-tail and uniform approximation bounds.

There is a counterexample to the unrestricted claim. For Gaussian-smoothed ReLU on ray fronts with Pareto radius P(R>r)=r^(-5/2), the second moment is finite, but radial averaging gives b_k proportional to k^(-4). At width 8, the ray rank consequently grows like eps^(-7/3), exceeding sampling asymptotically despite geometric Hermite decay at every fixed scale.

Fix: carry the scale-distribution conditions into the summary and theorem statement. Restrict the finite-smoothness lower-bound claim to the stated ray/regular-variation setting as well.

2. [major] P12 does not test the headline claim about summaries beating sampling.

Location: report Summary opening, §6, §7/P12; PREREG.md:104–105.

P12 compares random-net rank with half ReLU’s rank. The stated summary-versus-sampling criterion is instead r(eps)<eps^(-2). These are different tests. For example, the reported GELU ranks at eps=0.1 are 18 and 15, both below 100, while its sorting loss is 0.91 times ReLU’s.

Those numbers do not establish a trained-summary win either: ranks were measured on random nets, learning on separate trained nets, and summary construction cost was not measured.

Fix: say that no arm met the registered P12 composite criterion, and that the tested trained-net K2 implementation had no observed joint MC/RQMC wins. Leave trained-summary tractability unresolved.

3. [major] The conclusion at 10M+ parameters does not follow.

Location: report Summary item 9, especially “So swapping the gate alone will not make large models deductively estimable.”

The largest random architecture tested has approximately 65,536 weights; sorting nets have about 8,192 parameters and PFNs about 18,700. More fundamentally, a polynomial-summary upper bound exceeding sampling is not a lower bound on the best summary, and summary dimension is not running time. The single-back-layer theorem does not establish tractability or intractability of deep trained models.

ARC’s qualitatively similar depth results support a limited comparison with these experiments, not the universal conclusion.

Fix: replace the sentence with “These experiments do not establish that changing the gate suffices at 10M+ scale.” Keep linear-summary theory distinct from ARC’s nonlinear closures.

4. [major] A sufficient sorting construction is presented as a necessity theorem.

Location: report §2, “Smooth gates can learn kinked targets only by becoming sharp”; notes/theory.md:123–132.

The comparator construction correctly gives error at most d_sort log(2)/beta. It establishes one way to approximate sorting. It does not prove that every accurate smooth network must increase effective sharpness.

The perturbation inequality is also a valid stability statement for suitably coupled networks with the required weight moments. It is not a learnability lower bound, and its application to separately pilot-normalized ensembles would require another argument.

Fix: replace “only by becoming sharp” with “can approximate sorting through increasingly sharp comparators.” Label necessity, optimization behavior, and consequences for trained summaries as unresolved.

5. [major] The activation ranking confounds smoothness with numerical implementation and inconsistent cost conventions.

Location: report Summary item 4 and §3/Grid K; lib/k2gen.py:16, 54–84; ../narrow-deep/lib/gauss_relu.py:11–23; scratch/k2_grid.py:69–80.

ReLU is charged for the expensive bivariate-CDF implementation, while other gates use truncated Mehler expansions. The saved alternative ReLU results materially change the comparison:

- Width 16, depth 2: K2/MC changes from 1.809 to 0.340.
- Width 64, depth 8: 3.071 to 0.527.
- Width 64, depth 16: 4.894 to 0.738.

These do not establish RQMC wins, because RQMC must be rescored at the alternative budget. They do show that the headline comparison is implementation-dependent.

The accounting also mixes legacy CDF/PDF costs of 96/54 in the ReLU implementation with the stated 16-operation transcendental convention elsewhere. gsrelu’s evaluation cost of 154 similarly differs from the convention used for GELU.

Fix: retain the registered results, but add an explicitly post-hoc, error-matched implementation comparison with consistent operation costs and corresponding RQMC budgets. Do not attribute the resulting depth ranking solely to smoothness.

6. [major] Numerical solver error is not uniformly negligible on trained networks.

Location: lib/k2gen.py:29–36, 54–84; tests/test_k2.py; scratch/k2_trained.py:77–80.

The tests validate selected one-layer Gaussian cases, not the shifted distributions and correlations encountered throughout trained nets.

I recomputed trained closures with four times the quadrature nodes and Mehler order 128 instead of 32. For smoothstep seeds 1 and 2, the mean-estimate change was approximately 14% of the originally reported error norm. Softplus16 changes reached approximately 3%. Thus some reported “closure error” includes appreciable numerical approximation error.

Fix: report convergence checks on trained networks, separating quadrature order from Mehler order. Select tolerances before comparing computational costs. The existing tests passing does not establish this convergence.

7. [major] Grid C does not preserve the claimed gate-input scale after fronts and backs are spliced.

Location: report §1/Sharpness and Summary item 3; scratch/run_cut_act.py:37, 50, 64; lib/gnet.py:47–65.

Each back is normalized using its own original front, then attached to independent fronts. Its normalization does not transfer to those compositions. The diagnostic at line 64 checks separate intact networks, not the measured front/back combinations.

Reconstructing four fronts crossed with four backs at width 16 gave final-back-layer RMS ranges:

- softplus16: 0.112–3.150;
- GELU: 0.127–3.410;
- SiLU: 0.124–3.145.

These are far from a fixed sqrt(2). Effective sharpness therefore varies substantially in the actual cut experiment.

Fix: describe the splice ensemble accurately and measure its preactivation distributions. Treat “phase causes the rank difference” as an interpretation, not an isolated causal result.

8. [major] Empirical in-sample ranks are described too strongly as required summary sizes.

Location: report Summary item 3 and §3/Grid C; scratch/analyze_cut.py:15–25; ../cut-rank/lib/spec.py:3–8, 25–29.

The pair-Gram is unbiased; its ordered eigenvalue tail is not. Selecting leading eigenvectors from the same noisy matrix biases the tail downward. The identical-front controls correctly recover rank one but do not validate higher-rank estimates across activation-dependent distributions.

Recomputation confirms the reported raw ranks, but also substantial held-out differences: biased tanh at width 16 gives r_in=93 and r_ho=146. ReLU gives 37 versus 48.

Fix: call these finite-ensemble estimates, show held-out results beside primary ranks, and add uncertainty before saying a gate “needs” a particular number of summaries. Retain eps=0.01 as unresolved; the largest control-tail magnitude is approximately 2.19e-4 Vbar, slightly above the reported range.

9. [major] The three sorting seeds are coupled by gradient clipping.

Location: train/sort_act.py:60–70, 117–120.

Parameters for all seeds are stacked into shared tensors. After summing their losses, clip_grad_norm_(params, 1.0) computes one norm across all seeds. A large gradient in one seed changes the update scale of the others.

This is not three independent executions of the registered per-network clipping recipe. It also weakens interpretations of seed variation and close threshold crossings.

Fix: clip separately along the seed dimension or train independent models. Rerun the sorting comparison and resulting trained-net analyses, or explicitly label the existing results as a coupled-training deviation.

10. [major] P9, and the sharpening part of P10, use unregistered statistics.

Location: PREREG.md:98–101; scratch/k2_trained.py:93–95; scratch/summ_pfn.py:76–78; report §7/P9–P10.

P9 explicitly divides trained layer RMS by sqrt(2). The implementation divides by measured, layer-dependent initialization RMS.

For softplus1, the registered per-seed statistics are approximately 1.506, 1.338, and 2.375. Their median is 1.506, whereas the report gives 1.21. Under the report’s median-over-seeds convention, that clause crosses the registered threshold. Softplus4 remains below it, so the joint P9 prediction still fails.

P10 additionally changes from P9’s layer statistic to a median of per-unit ratios against reconstructed initialization.

Fix: score the literal registered statistics and label measured-initialization ratios as a justified post-hoc correction. The reconstructed PFN initialization itself checked out.

11. [major] The Hermite degree table still contains demonstrably underestimated tails.

Location: scratch/hermite_classes.py:35–56; lib/acts.py:250–262; runs/hermite_classes.json; logs/hermite_classes.log.

The code computes missing energy as max(quadrature_second_moment − coefficient_sum, 0). Underestimated quadrature moments silently erase the omitted tail.

Independent recomputation gives:

- softplus32: energy in degrees 4001–6000 alone is approximately 1.015e-9, versus an eps=1e-6 budget of 6.81e-13. The reported D=4000 is invalid.
- softplus16: energy beyond the reported D=3590 through degree 6000 is approximately 1.052e-12, already exceeding its 6.80e-13 budget.

Subtraction also produces negative tails and meaningless fitted slopes in some smooth arms.

Fix: sum tails directly, bound the remaining coefficients, and mark degrees unresolved when necessary. This is the same failure mode identified in the earlier ray audit; it was not repaired throughout the analysis.

12. [major] The ray discussion overstates its model, numerical certainty, and coverage.

Location: report Summary item 2 and §2/ray table; scratch/ray_cert.py:5–6, 27–43, 70–72; scratch/ray_family.py:4–11, 44–50.

The table uses a particular radial law, Gamma(1,1). It is not a proved worst case over front laws. Changing radial variance changes both the spectrum and the error budget.

The script explicitly calls its procedure a convergence check, not a certificate. Doubling truncation and radial nodes does not bound omitted energy or quadrature error, especially while using the clipped Parseval residual. The retained GELU entry did remain stable in my larger K=24,000, 96-node check.

There is also a direct contradiction: at width 16, ReLU² has rank about 935 at eps=0.03, below sampling’s 1,111. ReLU² is piecewise polynomial, not a polynomial gate exempt from the “no smooth or kinked gate wins” statement.

Fix: specify the radial law, say “numerically convergence-checked,” restrict comparisons to tested eps values, and correct the ReLU² exception.

13. [major] Reference noise and replicate uncertainty limit the categorical sampling claims.

Location: scratch/k2_grid.py:57–98; scratch/k2_trained.py:67–99; report Summary items 4 and 7, §5.

The 2^21-sample reference has expected normalized squared error 4.77e-7. Comparing estimates to that reference adds its error to measured MSE. In 181 nondegenerate grid-K cases, the reported 16,384-point RQMC error is below five times this floor. Large fixed-N gain estimates therefore need reference-error analysis.

For trained softplus1 seed 1, I used a more accurate independent reference and reproduced the original RQMC replicates. K2/RQMC was approximately 1.023, but the RQMC MSE standard error was about 1.00e-4 against a difference of only 1.39e-5. Its winner is statistically unresolved.

Also, six of the 45 trained nets beat the analytic MC benchmark. “Never beats sampling” means zero observed wins against both baselines, not zero MC wins.

Fix: preserve the registered observed count, explicitly distinguish MC from RQMC, report uncertainty, and improve the reference for floor-limited comparisons.

14. [minor] “Equal FLOPs” is actually a rounded, partial budget comparison.

Location: PREREG.md:55–58; scratch/k2_grid.py:83–98; scratch/k2_trained.py:84–92; report §3 and §5.

RQMC receives the largest power-of-two sample count below N_eq. Across grid K, it uses approximately 50.9–98.6% of the nominal network-evaluation budget. Input generation and inverse-normal transformation costs are omitted.

The 1/N_eq MC value is a valid expected-error convention, not an observed MC error or an exact finite integer-sample comparison. SciPy’s scrambling is LMS plus digital shift; calling it simply Owen scrambling is imprecise.

Fix: say “within the same network-evaluation FLOP budget, rounded down for Sobol,” document the scrambling method, and distinguish this benchmark from end-to-end runtime.

15. [major] The depth claims again omit conditions retained in the theory note.

Location: report §2/Depth; notes/theory.md:87–100.

The radius recurrence requires a nonconstant positive-coefficient series with 1<R_1<infinity. Root-exponential kernels already have radius one, so the chaotic/critical/ordered radius statements cannot be applied indiscriminately to softplus, SiLU, or tanh.

The note also asserts geometric growth of required degree at fixed eps, then acknowledges that radius alone does not establish finite-eps degrees. Those are inconsistent levels of certainty.

Finally, nonnegativity and smoothness do not imply an ordered gate. For example, phi(z)=z² has normalized chi_1=4/3.

Fix: retain the explicit radius assumptions, remove the unsupported fixed-eps degree conclusion, and limit ordering claims to the particular checked gates and scales.

16. [major] Bilinear variance dynamics are incorrectly generalized to all GLUs.

Location: report §2/Bilinear backs; notes/theory.md:106–108.

The q→Aq² map applies to bilinear or appropriately homogeneous gates. For a general GLU with independent branches, the map instead contains q E[phi(g sqrt(q) Z)²], up to the branch gain. It is not generally quadratic, and its fixed-point derivative is not universally two.

The independent-branch kernel identity and finite polynomial rank of bilinear backs are sound. Neither proves instability of every GLU architecture.

Fix: restrict the quadratic-map proof to its actual gate class. Present SwiGLU divergence as an observed failure of this initialization and architecture, or analyze its own variance map.

17. [minor] Biased tanh is assigned the bias-free phase parameter.

Location: lib/gnet.py:28–37; logs/C_table.md; runs/k2_tanhb_n*.json.

Spec computes g2=q0/Ephi² and chi=g2 Ephi'² even when bias_frac=0.13. The fixed-scale biased calculation instead uses (q0−bias_variance)/Ephi².

At q0=2, the resulting chi is approximately 1.170, not 1.344. Both are above one, so this does not reverse the phase label, but the reported metadata is wrong.

Fix: include the bias variance in the mean-field diagnostic and label pilot-normalized finite-width phase assignments as approximations.

18. [minor] Several registration descriptions and scores need correction.

Location: PREREG.md:44–46, 85–95; scratch/score_K.py:28–29; scratch/score_C.py:73–83; report §7/P4/P6 and §8.

P6 calls for a median over nets and depths. The implementation takes a median over depth-wise medians. Pooling the registered observations gives relative gains of approximately 2.20, 0.514, and 0.343 for softplus1, GELU, and SiLU, versus the stored 1.66, 0.345, and 0.266. P6 remains refuted.

P4’s universal statement is marked “holds” while smoothstep at width 16 is missing. It should be provisional.

Scale-normalized ranks were explicitly included in the original hashed registration, yet the report labels them post hoc.

Fix: correct aggregation, completeness labels, and provenance descriptions. I found no hash mismatch or amendment/data chronology reversal in the preserved evidence.

19. [minor] The empirical frontier and depth effects are not monotone as claimed.

Location: report Summary item 8, §6/Pinned ladder, and §3 “depth hurts every gate”; runs/pareto.json and runs/k2_*.json.

Pinned softplus8 has lower mean sorting loss than softplus16, 0.092751 versus 0.093508, and lower K2/MC, approximately 10.6 versus 11.7. It improves both measured axes. Pinned ReLU also dominates softplus16 on these point estimates.

At width 64, softplus1’s K2/MC improves from 0.003084 at depth 8 to 0.002595 at depth 16; gsrelu0.5 improves from 0.1821 to 0.1569.

Fix: describe an overall empirical trend with exceptions and uncertainty, not a monotone trade or an established Pareto frontier.

Verdict: major revision required. The core projection lemmas, activation identities, logistic mixture, softplus asymptotic, conditional density results, and sorting approximation survive review; the broad frontier and scale conclusions do not.

Both test suites passed. Selected cut/K2 entries, PFN initialization ratios, and GELU seed-0 test KL reproduced. All four registration hashes matched, and preserved run-start times followed the relevant registrations. Bias-free tanh and collapsed softplus nets are disclosed appropriately. No files were modified.
tokens used
200,690
Locations below use “report” for tractable-gates-2026-10-31.md, version modified 2026-10-31 04:25:17 UTC. Other paths are relative to work/tractable-gates.

1. [major] The “any front” polylogarithmic claim drops essential assumptions.

Location: report Summary item 1 and §2; notes/theory.md:23–33, 61–64.

Lemma P is valid under its stated conditions. Lemma R is also valid, but is an upper bound, not the “exact reduction” claimed in the report. Neither converts fixed-scale Gaussian Hermite decay into polylogarithmic rank for arbitrary front laws. That requires the bounded-scale/inverse-moment assumptions, or quantitative scale-tail and uniform approximation bounds.

There is a counterexample to the unrestricted claim. For Gaussian-smoothed ReLU on ray fronts with Pareto radius P(R>r)=r^(-5/2), the second moment is finite, but radial averaging gives b_k proportional to k^(-4). At width 8, the ray rank consequently grows like eps^(-7/3), exceeding sampling asymptotically despite geometric Hermite decay at every fixed scale.

Fix: carry the scale-distribution conditions into the summary and theorem statement. Restrict the finite-smoothness lower-bound claim to the stated ray/regular-variation setting as well.

2. [major] P12 does not test the headline claim about summaries beating sampling.

Location: report Summary opening, §6, §7/P12; PREREG.md:104–105.

P12 compares random-net rank with half ReLU’s rank. The stated summary-versus-sampling criterion is instead r(eps)<eps^(-2). These are different tests. For example, the reported GELU ranks at eps=0.1 are 18 and 15, both below 100, while its sorting loss is 0.91 times ReLU’s.

Those numbers do not establish a trained-summary win either: ranks were measured on random nets, learning on separate trained nets, and summary construction cost was not measured.

Fix: say that no arm met the registered P12 composite criterion, and that the tested trained-net K2 implementation had no observed joint MC/RQMC wins. Leave trained-summary tractability unresolved.

3. [major] The conclusion at 10M+ parameters does not follow.

Location: report Summary item 9, especially “So swapping the gate alone will not make large models deductively estimable.”

The largest random architecture tested has approximately 65,536 weights; sorting nets have about 8,192 parameters and PFNs about 18,700. More fundamentally, a polynomial-summary upper bound exceeding sampling is not a lower bound on the best summary, and summary dimension is not running time. The single-back-layer theorem does not establish tractability or intractability of deep trained models.

ARC’s qualitatively similar depth results support a limited comparison with these experiments, not the universal conclusion.

Fix: replace the sentence with “These experiments do not establish that changing the gate suffices at 10M+ scale.” Keep linear-summary theory distinct from ARC’s nonlinear closures.

4. [major] A sufficient sorting construction is presented as a necessity theorem.

Location: report §2, “Smooth gates can learn kinked targets only by becoming sharp”; notes/theory.md:123–132.

The comparator construction correctly gives error at most d_sort log(2)/beta. It establishes one way to approximate sorting. It does not prove that every accurate smooth network must increase effective sharpness.

The perturbation inequality is also a valid stability statement for suitably coupled networks with the required weight moments. It is not a learnability lower bound, and its application to separately pilot-normalized ensembles would require another argument.

Fix: replace “only by becoming sharp” with “can approximate sorting through increasingly sharp comparators.” Label necessity, optimization behavior, and consequences for trained summaries as unresolved.

5. [major] The activation ranking confounds smoothness with numerical implementation and inconsistent cost conventions.

Location: report Summary item 4 and §3/Grid K; lib/k2gen.py:16, 54–84; ../narrow-deep/lib/gauss_relu.py:11–23; scratch/k2_grid.py:69–80.

ReLU is charged for the expensive bivariate-CDF implementation, while other gates use truncated Mehler expansions. The saved alternative ReLU results materially change the comparison:

- Width 16, depth 2: K2/MC changes from 1.809 to 0.340.
- Width 64, depth 8: 3.071 to 0.527.
- Width 64, depth 16: 4.894 to 0.738.

These do not establish RQMC wins, because RQMC must be rescored at the alternative budget. They do show that the headline comparison is implementation-dependent.

The accounting also mixes legacy CDF/PDF costs of 96/54 in the ReLU implementation with the stated 16-operation transcendental convention elsewhere. gsrelu’s evaluation cost of 154 similarly differs from the convention used for GELU.

Fix: retain the registered results, but add an explicitly post-hoc, error-matched implementation comparison with consistent operation costs and corresponding RQMC budgets. Do not attribute the resulting depth ranking solely to smoothness.

6. [major] Numerical solver error is not uniformly negligible on trained networks.

Location: lib/k2gen.py:29–36, 54–84; tests/test_k2.py; scratch/k2_trained.py:77–80.

The tests validate selected one-layer Gaussian cases, not the shifted distributions and correlations encountered throughout trained nets.

I recomputed trained closures with four times the quadrature nodes and Mehler order 128 instead of 32. For smoothstep seeds 1 and 2, the mean-estimate change was approximately 14% of the originally reported error norm. Softplus16 changes reached approximately 3%. Thus some reported “closure error” includes appreciable numerical approximation error.

Fix: report convergence checks on trained networks, separating quadrature order from Mehler order. Select tolerances before comparing computational costs. The existing tests passing does not establish this convergence.

7. [major] Grid C does not preserve the claimed gate-input scale after fronts and backs are spliced.

Location: report §1/Sharpness and Summary item 3; scratch/run_cut_act.py:37, 50, 64; lib/gnet.py:47–65.

Each back is normalized using its own original front, then attached to independent fronts. Its normalization does not transfer to those compositions. The diagnostic at line 64 checks separate intact networks, not the measured front/back combinations.

Reconstructing four fronts crossed with four backs at width 16 gave final-back-layer RMS ranges:

- softplus16: 0.112–3.150;
- GELU: 0.127–3.410;
- SiLU: 0.124–3.145.

These are far from a fixed sqrt(2). Effective sharpness therefore varies substantially in the actual cut experiment.

Fix: describe the splice ensemble accurately and measure its preactivation distributions. Treat “phase causes the rank difference” as an interpretation, not an isolated causal result.

8. [major] Empirical in-sample ranks are described too strongly as required summary sizes.

Location: report Summary item 3 and §3/Grid C; scratch/analyze_cut.py:15–25; ../cut-rank/lib/spec.py:3–8, 25–29.

The pair-Gram is unbiased; its ordered eigenvalue tail is not. Selecting leading eigenvectors from the same noisy matrix biases the tail downward. The identical-front controls correctly recover rank one but do not validate higher-rank estimates across activation-dependent distributions.

Recomputation confirms the reported raw ranks, but also substantial held-out differences: biased tanh at width 16 gives r_in=93 and r_ho=146. ReLU gives 37 versus 48.

Fix: call these finite-ensemble estimates, show held-out results beside primary ranks, and add uncertainty before saying a gate “needs” a particular number of summaries. Retain eps=0.01 as unresolved; the largest control-tail magnitude is approximately 2.19e-4 Vbar, slightly above the reported range.

9. [major] The three sorting seeds are coupled by gradient clipping.

Location: train/sort_act.py:60–70, 117–120.

Parameters for all seeds are stacked into shared tensors. After summing their losses, clip_grad_norm_(params, 1.0) computes one norm across all seeds. A large gradient in one seed changes the update scale of the others.

This is not three independent executions of the registered per-network clipping recipe. It also weakens interpretations of seed variation and close threshold crossings.

Fix: clip separately along the seed dimension or train independent models. Rerun the sorting comparison and resulting trained-net analyses, or explicitly label the existing results as a coupled-training deviation.

10. [major] P9, and the sharpening part of P10, use unregistered statistics.

Location: PREREG.md:98–101; scratch/k2_trained.py:93–95; scratch/summ_pfn.py:76–78; report §7/P9–P10.

P9 explicitly divides trained layer RMS by sqrt(2). The implementation divides by measured, layer-dependent initialization RMS.

For softplus1, the registered per-seed statistics are approximately 1.506, 1.338, and 2.375. Their median is 1.506, whereas the report gives 1.21. Under the report’s median-over-seeds convention, that clause crosses the registered threshold. Softplus4 remains below it, so the joint P9 prediction still fails.

P10 additionally changes from P9’s layer statistic to a median of per-unit ratios against reconstructed initialization.

Fix: score the literal registered statistics and label measured-initialization ratios as a justified post-hoc correction. The reconstructed PFN initialization itself checked out.

11. [major] The Hermite degree table still contains demonstrably underestimated tails.

Location: scratch/hermite_classes.py:35–56; lib/acts.py:250–262; runs/hermite_classes.json; logs/hermite_classes.log.

The code computes missing energy as max(quadrature_second_moment − coefficient_sum, 0). Underestimated quadrature moments silently erase the omitted tail.

Independent recomputation gives:

- softplus32: energy in degrees 4001–6000 alone is approximately 1.015e-9, versus an eps=1e-6 budget of 6.81e-13. The reported D=4000 is invalid.
- softplus16: energy beyond the reported D=3590 through degree 6000 is approximately 1.052e-12, already exceeding its 6.80e-13 budget.

Subtraction also produces negative tails and meaningless fitted slopes in some smooth arms.

Fix: sum tails directly, bound the remaining coefficients, and mark degrees unresolved when necessary. This is the same failure mode identified in the earlier ray audit; it was not repaired throughout the analysis.

12. [major] The ray discussion overstates its model, numerical certainty, and coverage.

Location: report Summary item 2 and §2/ray table; scratch/ray_cert.py:5–6, 27–43, 70–72; scratch/ray_family.py:4–11, 44–50.

The table uses a particular radial law, Gamma(1,1). It is not a proved worst case over front laws. Changing radial variance changes both the spectrum and the error budget.

The script explicitly calls its procedure a convergence check, not a certificate. Doubling truncation and radial nodes does not bound omitted energy or quadrature error, especially while using the clipped Parseval residual. The retained GELU entry did remain stable in my larger K=24,000, 96-node check.

There is also a direct contradiction: at width 16, ReLU² has rank about 935 at eps=0.03, below sampling’s 1,111. ReLU² is piecewise polynomial, not a polynomial gate exempt from the “no smooth or kinked gate wins” statement.

Fix: specify the radial law, say “numerically convergence-checked,” restrict comparisons to tested eps values, and correct the ReLU² exception.

13. [major] Reference noise and replicate uncertainty limit the categorical sampling claims.

Location: scratch/k2_grid.py:57–98; scratch/k2_trained.py:67–99; report Summary items 4 and 7, §5.

The 2^21-sample reference has expected normalized squared error 4.77e-7. Comparing estimates to that reference adds its error to measured MSE. In 181 nondegenerate grid-K cases, the reported 16,384-point RQMC error is below five times this floor. Large fixed-N gain estimates therefore need reference-error analysis.

For trained softplus1 seed 1, I used a more accurate independent reference and reproduced the original RQMC replicates. K2/RQMC was approximately 1.023, but the RQMC MSE standard error was about 1.00e-4 against a difference of only 1.39e-5. Its winner is statistically unresolved.

Also, six of the 45 trained nets beat the analytic MC benchmark. “Never beats sampling” means zero observed wins against both baselines, not zero MC wins.

Fix: preserve the registered observed count, explicitly distinguish MC from RQMC, report uncertainty, and improve the reference for floor-limited comparisons.

14. [minor] “Equal FLOPs” is actually a rounded, partial budget comparison.

Location: PREREG.md:55–58; scratch/k2_grid.py:83–98; scratch/k2_trained.py:84–92; report §3 and §5.

RQMC receives the largest power-of-two sample count below N_eq. Across grid K, it uses approximately 50.9–98.6% of the nominal network-evaluation budget. Input generation and inverse-normal transformation costs are omitted.

The 1/N_eq MC value is a valid expected-error convention, not an observed MC error or an exact finite integer-sample comparison. SciPy’s scrambling is LMS plus digital shift; calling it simply Owen scrambling is imprecise.

Fix: say “within the same network-evaluation FLOP budget, rounded down for Sobol,” document the scrambling method, and distinguish this benchmark from end-to-end runtime.

15. [major] The depth claims again omit conditions retained in the theory note.

Location: report §2/Depth; notes/theory.md:87–100.

The radius recurrence requires a nonconstant positive-coefficient series with 1<R_1<infinity. Root-exponential kernels already have radius one, so the chaotic/critical/ordered radius statements cannot be applied indiscriminately to softplus, SiLU, or tanh.

The note also asserts geometric growth of required degree at fixed eps, then acknowledges that radius alone does not establish finite-eps degrees. Those are inconsistent levels of certainty.

Finally, nonnegativity and smoothness do not imply an ordered gate. For example, phi(z)=z² has normalized chi_1=4/3.

Fix: retain the explicit radius assumptions, remove the unsupported fixed-eps degree conclusion, and limit ordering claims to the particular checked gates and scales.

16. [major] Bilinear variance dynamics are incorrectly generalized to all GLUs.

Location: report §2/Bilinear backs; notes/theory.md:106–108.

The q→Aq² map applies to bilinear or appropriately homogeneous gates. For a general GLU with independent branches, the map instead contains q E[phi(g sqrt(q) Z)²], up to the branch gain. It is not generally quadratic, and its fixed-point derivative is not universally two.

The independent-branch kernel identity and finite polynomial rank of bilinear backs are sound. Neither proves instability of every GLU architecture.

Fix: restrict the quadratic-map proof to its actual gate class. Present SwiGLU divergence as an observed failure of this initialization and architecture, or analyze its own variance map.

17. [minor] Biased tanh is assigned the bias-free phase parameter.

Location: lib/gnet.py:28–37; logs/C_table.md; runs/k2_tanhb_n*.json.

Spec computes g2=q0/Ephi² and chi=g2 Ephi'² even when bias_frac=0.13. The fixed-scale biased calculation instead uses (q0−bias_variance)/Ephi².

At q0=2, the resulting chi is approximately 1.170, not 1.344. Both are above one, so this does not reverse the phase label, but the reported metadata is wrong.

Fix: include the bias variance in the mean-field diagnostic and label pilot-normalized finite-width phase assignments as approximations.

18. [minor] Several registration descriptions and scores need correction.

Location: PREREG.md:44–46, 85–95; scratch/score_K.py:28–29; scratch/score_C.py:73–83; report §7/P4/P6 and §8.

P6 calls for a median over nets and depths. The implementation takes a median over depth-wise medians. Pooling the registered observations gives relative gains of approximately 2.20, 0.514, and 0.343 for softplus1, GELU, and SiLU, versus the stored 1.66, 0.345, and 0.266. P6 remains refuted.

P4’s universal statement is marked “holds” while smoothstep at width 16 is missing. It should be provisional.

Scale-normalized ranks were explicitly included in the original hashed registration, yet the report labels them post hoc.

Fix: correct aggregation, completeness labels, and provenance descriptions. I found no hash mismatch or amendment/data chronology reversal in the preserved evidence.

19. [minor] The empirical frontier and depth effects are not monotone as claimed.

Location: report Summary item 8, §6/Pinned ladder, and §3 “depth hurts every gate”; runs/pareto.json and runs/k2_*.json.

Pinned softplus8 has lower mean sorting loss than softplus16, 0.092751 versus 0.093508, and lower K2/MC, approximately 10.6 versus 11.7. It improves both measured axes. Pinned ReLU also dominates softplus16 on these point estimates.

At width 64, softplus1’s K2/MC improves from 0.003084 at depth 8 to 0.002595 at depth 16; gsrelu0.5 improves from 0.1821 to 0.1569.

Fix: describe an overall empirical trend with exceptions and uncertainty, not a monotone trade or an established Pareto frontier.

Verdict: major revision required. The core projection lemmas, activation identities, logistic mixture, softplus asymptotic, conditional density results, and sorting approximation survive review; the broad frontier and scale conclusions do not.

Both test suites passed. Selected cut/K2 entries, PFN initialization ratios, and GELU seed-0 test KL reproduced. All four registration hashes matched, and preserved run-start times followed the relevant registrations. Bias-free tanh and collapsed softplus nets are disclosed appropriately. No files were modified.
