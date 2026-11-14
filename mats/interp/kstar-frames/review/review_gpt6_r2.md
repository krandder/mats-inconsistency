codex
Checked **2026-10-31 03:21 UTC**, read-only. Items 1–19 follow the original review; references to “report” mean the revised report.

1. **Partly fixed — covering numerics.** Both tests and the original regression pass: **0.267171**, matching SVD; another 16,384 sampled pairs agree within **3×10⁻⁸**, but the numerical-rank convention needs qualification (§20 below).
2. **Partly fixed — universal negative conclusion.** Report §7 now limits failure to tested maps, but “an exp(k*) lower bound would have to come from the joint law” still does not follow from an approximate learned predictor.
3. **Partly fixed — heuristic atlas.** §5 correctly calls it uncalibrated; the opening summary still asserts that an atlas “grows steeply with width” without limiting this to finite-store diagnostics and the heuristic.
4. **Partly fixed — P13 sensitivity.** Rerun reproduces **5.75–6.05** across binnings and **5.70** with per-net radii; `theta_coh()` still represents right-censoring as infinity, and per-net sensitivity silently excludes such nets.
5. **Partly fixed — Proposition 1.** §2 now supplies the rank/exactness conditions and effective-rank caveat; summary line 16 still incorrectly calls the measured k* transport an **exact rule**.
6. **Fixed — Proposition 2 theorem.** The stated inequality follows under invariant spectral-gap conditioning and fixed/past-measurable k; the radial exception is correctly allowed.
7. **Fixed — conditional experiment.** `t2_prop2_cond.py` genuinely fixes pasts and resamples independent futures; its invariant filtering is valid, and rerunning reproduces the logged conditional means.
8. **Fixed — aggregation.** Recomputed P2–P7 and P11c use net means: P2 **0/90**, P5 **68/90 and 48/48**, P11c **2/144**, matching the report.
9. **Fixed — ball-mass labels.** §5 now distinguishes median-over-nets path means from path medians; both columns reproduce.
10. **Partly fixed — learning curves/comparisons.** §3 discloses all-reference initialization and reports paired gains, but P9 still compares two learned nets with three fixed nets; its **0.08–0.09** gain is unmatched.
11. **Fixed — fixed-frame optimality.** §1 explicitly restricts optimality to unmasked frames and withdraws it after projection.
12. **Fixed — noise ceiling.** §3 correctly describes fresh-noise frames as a repeatability benchmark.
13. **Fixed — transport mechanism.** §2 distinguishes generic kernel combinations, unmeasured amplification, and dual overlap averaging **0.05 above chance**.
14. **Partly fixed — coverage interpretation.** Union coverage, unresolved empirical zeros, and fixed-R probabilities are now distinguished; the COHERENT implementation still has the logical discrepancy in §23 below.
15. **Partly fixed — coherence precision.** Sensitivity and overlap radii are now separated, but the overlap column should explicitly say **raw/unprojected**, matching `pairs()[3]`.
16. **Partly fixed — uncertainty.** Three-net limitations and alternative growth laws are acknowledged, but “the radius does not depend on width” (§4) remains stronger than passing a factor-two test on three widths.
17. **Partly fixed — preregistration/execution.** Original, Amendment 1, Amendment 2, and all other listed hashes verify; Amendment 2 correctly declares post-review exposure and omissions, which remain uncompleted.
18. **Not fixed — edge cases.** `orth()` still lacks rank detection and `n_surv()` still counts a zero Jacobian as full rank; §10 accurately acknowledges both.
19. **Fixed/confirmed — numerical checks.** The scorecard rerun reproduces the saved numerical output; the only log difference is warning placement. Both supplied test suites pass.

20. **Projector qualification:** [cover.py:18](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:18) discards Gram eigenvalues below `1e-10` relative to the largest. With active coordinates 1–2 and reference columns \(e_1,\;10^{-6}e_2+\sqrt{1-10^{-12}}e_3\), a target supported equally on \(e_1,e_2\) scores **CS=O=0.5**, versus exact range-projector scores **1**. V2 fixes the observed instability, but implements numerical-rank truncation; disclose that convention or use a suitable direct-SVD fallback.

21. **V2 table spot-checks:** For \(n=32,L=32\), median-over-net mean mass is **0.08976060**; for \(n=128,L=16\), it is **4.88758087×10⁻⁶**. Both match §5 rounding; the latter’s median-input mass is **0**.

22. **New proposition-note overstatement:** “The radial weight alpha is small” is unsupported: direct measurement gives **0.300–0.428**, while the original past frame already contains **≥0.999825** of the radial direction. Also, G excludes rank **below k**, not all rank-deficient futures.

23. **COHERENT logic:** [r9_score.py:483](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:483) requires both depths to pass directly or both heuristically, excluding mixed evidence; it also requires every net to pass directly rather than applying the general net-mean rule. Neither discrepancy changes the current **Q3=false** result; P13(a) correctly uses Bcover2.

24. **Small numerical correction:** §10’s maximum ball-mass change is **0.062686%**, slightly above the stated **0.06%** (`n64_s402_L16`, δ=0.05).

**Verdict: minor revisions.**
tokens used
81,197
Checked **2026-10-31 03:21 UTC**, read-only. Items 1–19 follow the original review; references to “report” mean the revised report.

1. **Partly fixed — covering numerics.** Both tests and the original regression pass: **0.267171**, matching SVD; another 16,384 sampled pairs agree within **3×10⁻⁸**, but the numerical-rank convention needs qualification (§20 below).
2. **Partly fixed — universal negative conclusion.** Report §7 now limits failure to tested maps, but “an exp(k*) lower bound would have to come from the joint law” still does not follow from an approximate learned predictor.
3. **Partly fixed — heuristic atlas.** §5 correctly calls it uncalibrated; the opening summary still asserts that an atlas “grows steeply with width” without limiting this to finite-store diagnostics and the heuristic.
4. **Partly fixed — P13 sensitivity.** Rerun reproduces **5.75–6.05** across binnings and **5.70** with per-net radii; `theta_coh()` still represents right-censoring as infinity, and per-net sensitivity silently excludes such nets.
5. **Partly fixed — Proposition 1.** §2 now supplies the rank/exactness conditions and effective-rank caveat; summary line 16 still incorrectly calls the measured k* transport an **exact rule**.
6. **Fixed — Proposition 2 theorem.** The stated inequality follows under invariant spectral-gap conditioning and fixed/past-measurable k; the radial exception is correctly allowed.
7. **Fixed — conditional experiment.** `t2_prop2_cond.py` genuinely fixes pasts and resamples independent futures; its invariant filtering is valid, and rerunning reproduces the logged conditional means.
8. **Fixed — aggregation.** Recomputed P2–P7 and P11c use net means: P2 **0/90**, P5 **68/90 and 48/48**, P11c **2/144**, matching the report.
9. **Fixed — ball-mass labels.** §5 now distinguishes median-over-nets path means from path medians; both columns reproduce.
10. **Partly fixed — learning curves/comparisons.** §3 discloses all-reference initialization and reports paired gains, but P9 still compares two learned nets with three fixed nets; its **0.08–0.09** gain is unmatched.
11. **Fixed — fixed-frame optimality.** §1 explicitly restricts optimality to unmasked frames and withdraws it after projection.
12. **Fixed — noise ceiling.** §3 correctly describes fresh-noise frames as a repeatability benchmark.
13. **Fixed — transport mechanism.** §2 distinguishes generic kernel combinations, unmeasured amplification, and dual overlap averaging **0.05 above chance**.
14. **Partly fixed — coverage interpretation.** Union coverage, unresolved empirical zeros, and fixed-R probabilities are now distinguished; the COHERENT implementation still has the logical discrepancy in §23 below.
15. **Partly fixed — coherence precision.** Sensitivity and overlap radii are now separated, but the overlap column should explicitly say **raw/unprojected**, matching `pairs()[3]`.
16. **Partly fixed — uncertainty.** Three-net limitations and alternative growth laws are acknowledged, but “the radius does not depend on width” (§4) remains stronger than passing a factor-two test on three widths.
17. **Partly fixed — preregistration/execution.** Original, Amendment 1, Amendment 2, and all other listed hashes verify; Amendment 2 correctly declares post-review exposure and omissions, which remain uncompleted.
18. **Not fixed — edge cases.** `orth()` still lacks rank detection and `n_surv()` still counts a zero Jacobian as full rank; §10 accurately acknowledges both.
19. **Fixed/confirmed — numerical checks.** The scorecard rerun reproduces the saved numerical output; the only log difference is warning placement. Both supplied test suites pass.

20. **Projector qualification:** [cover.py:18](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:18) discards Gram eigenvalues below `1e-10` relative to the largest. With active coordinates 1–2 and reference columns \(e_1,\;10^{-6}e_2+\sqrt{1-10^{-12}}e_3\), a target supported equally on \(e_1,e_2\) scores **CS=O=0.5**, versus exact range-projector scores **1**. V2 fixes the observed instability, but implements numerical-rank truncation; disclose that convention or use a suitable direct-SVD fallback.

21. **V2 table spot-checks:** For \(n=32,L=32\), median-over-net mean mass is **0.08976060**; for \(n=128,L=16\), it is **4.88758087×10⁻⁶**. Both match §5 rounding; the latter’s median-input mass is **0**.

22. **New proposition-note overstatement:** “The radial weight alpha is small” is unsupported: direct measurement gives **0.300–0.428**, while the original past frame already contains **≥0.999825** of the radial direction. Also, G excludes rank **below k**, not all rank-deficient futures.

23. **COHERENT logic:** [r9_score.py:483](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:483) requires both depths to pass directly or both heuristically, excluding mixed evidence; it also requires every net to pass directly rather than applying the general net-mean rule. Neither discrepancy changes the current **Q3=false** result; P13(a) correctly uses Bcover2.

24. **Small numerical correction:** §10’s maximum ball-mass change is **0.062686%**, slightly above the stated **0.06%** (`n64_s402_L16`, δ=0.05).

**Verdict: minor revisions.**
