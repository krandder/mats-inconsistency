codex
**Verdict: major revisions needed.** The core experiments are useful, but I found a reproducible covering-score bug, incorrect scorecard aggregation, and claims stronger than the evidence. No files were edited.

References below use **report** for [kstar-frames-2026-10-30.md](/home/agent/workspaces/mats/outbox/kstar-frames-2026-10-30.md); other paths are relative to the supplied work directory.

1. **The covering calculation produces impossible scores and false covers.**  
   In [lib/cover.py:34](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:34), the Gram matrix is formed in float32, then inverted in float64 with an eigenvalue cutoff of \(10^{-10}\). Casting afterward cannot recover the lost precision. Small genuine eigenvalues and rounding errors are amplified.

   Concrete reproduction: `B_n32_s401_L16_c8.npz`, held-out index **1106**, reference index **11345**, zero-based. With the production-sized batch, `pair_scores` returns **CS = 2.1550903**. A float64 SVD projector gives **0.2671709**. This falsely declares coverage at CS ≥ 0.9. For a projector of rank at most \(k\), Ky Fan’s inequality requires CS ≤ 1.

   The saved covering matrices contain **34 entries above 1.001**, across five width-32/64 configurations. Stable recomputation puts **33 of those below 0.9**. These are only detected outliers; numerically wrong scores inside [0,1] remain possible. Recompute affected covering results using stable projectors before treating their masses, greedy choices, or R99 as verified. The existing covering test passes because its cases do not expose this conditioning problem.

2. **“No predictable, shared map” is not established.**  
   Report:218–230 moves from failure of particular predictors and a finite reference bank to “There is no shared exact one” and an apparent exhaustive list of estimator options.

   The supported conclusion is: **none of the tested methods met the preregistered joint CS/overlap gate in all configurations**. Proposition 2 excludes only predictors measurable from the past alone. It excludes neither predictors using downstream weights nor better learned algorithms. Indeed, the learned map is positive evidence for substantial predictability.

   The distinction from Kelvin’s joint-law premise at report:57–61 is correct. However, these experiments neither bound that law’s description length nor establish an estimator lower bound. Computing derivatives per evaluation does not itself force Monte Carlo; deterministic quadrature and other integration schemes can also evaluate derivatives.

3. **The heuristic atlas count is not a covering-number estimate with a demonstrated calibration.**  
   Report:195–207; [scratch/r9_score.py:323](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:323).

   \(C(\theta)\) is a **mean state-space pair mass**. Its reciprocal is generally neither an optimal frame-covering count nor a 99%-coverage count. The conversion lacks three ingredients:

   - A guarantee that state proximity implies the required frame score; the radius uses a conditional **median**.
   - Control of low-mass states, which dominate high-coverage requirements.
   - Control of distant states sharing frames, already observed here.

   The reference-frame loss is also asymmetric and includes target-dependent masking; it is not an ordinary subspace metric. Labeling the calculation HEURISTIC is appropriate, but “order of magnitude at best,” “conservative,” and the exponential-atlas interpretation still suggest more calibration than has been demonstrated. The acknowledged approximately 100-fold state/frame mass discrepancy is already larger than one order of magnitude.

4. **P13’s numerical pass depends on unspecified analysis choices.**  
   [scratch/r9_score.py:228](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:228), :368–378.

   The implementation uses 24 logarithmic angle bins, accepts bins with eight observations, takes the first median crossing, and assigns the geometric bin midpoint rather than the observed median angle. These choices were not fixed in the preregistration.

   On the same width-128, depth-32 data:

   | Angle bins | Heuristic log10 count |
   |---:|---:|
   | 16 | 5.939 |
   | 20 | 5.975 |
   | 24, reported | 6.046 |
   | 40 | 5.839 |
   | 48 | 5.748 |

   Thus P13(b)’s threshold of 6 changes verdict under modest binning changes. Also, the code applies the **across-net median radius to every net’s state distribution**. Using each net’s own measured radius changes the depth-32 average from **6.046 to 5.696**. These are sensitivity checks, not proposed replacement scoring rules.

   An uncrossed curve is right-censored at the observed angles, not evidence that the radius is infinite. The declared median amendment mitigates one aggregation problem but does not resolve this censoring issue.

5. **Proposition 1’s linear algebra is essentially right, but its headline interpretation drops necessary conditions.**  
   [notes/props.md:7](/home/agent/workspaces/mats/work/kstar-frames/notes/props.md:7); report:17–29, :86–93, :125–129.

   The row-space identity and pseudoinverse-projector identity are correct. Exact recovery requires \(J_c\) to have rank exactly \(k\), and its row space to lie in \(\operatorname{range}(B_c)\). Non-growing active count makes the latter automatic for a generic full-row-rank active block.

   However:

   - If \(B_c^T\) reduces rank, the previous “top-\(k\) frame” includes arbitrary null-space completion; it is not simply the lower-dimensional row space.
   - Proposition 1(c)’s “iff invariant” additionally needs rank preservation. An invariant subspace contained in the kernel is an immediate counterexample to sufficiency.
   - The actual \(k^*\) is a thresholded effective rank. Finite-horizon top-\(k\) singular subspaces are generally **not exactly inverse-transpose covariant**.
   - The claimed exact all-layer backward construction costing \(O(L'n^2k^*)\) is not demonstrated. Run A forms full Jacobians and decomposes them. Propagating truncated frames backward is approximate, as the measured overlap below 1 confirms.

6. **Proposition 2’s invariance proof works with explicit spectral-event conditioning; the report omits that qualification.**  
   [notes/props.md:23](/home/agent/workspaces/mats/work/kstar-frames/notes/props.md:23); report:96–104.

   Line by line:

   - Given the past, \(e_c,B\) are fixed; Gaussian orthogonal projections make \(g=W_{c+1}e_c\) and \(M=W_{c+1}B\) independent.
   - At \(\sigma=0\), the future **trajectory and gates** depend on \(W_{c+1}\) only through \(g\). The Jacobian additionally depends on \(M\). The displayed \(QD_{c+1}M\) is consistent because the extra mask is redundant.
   - \(MO\) has the same conditional law for every orthogonal \(O\), independently of \(g\) and later weights.
   - Therefore the Jacobian law is invariant under right multiplication by \(\operatorname{diag}(1,O)\).
   - Where the top-\(k\) spectral projector is uniquely defined, its expectation has form \(\operatorname{diag}(\alpha,\beta I)\). The trace identity and stated inequality follow.

   The repair is to write the expectation conditional on a suitable invariant event, such as a positive spectral gap at the cutoff. Dead or rank-deficient future products have positive probability, and an arbitrary SVD tie-breaking rule need not be equivariant. Merely assuming \(a_c\ne0,m_c\ge3\), as the report does, does not address this.

   The theorem also needs fixed or appropriately past-measurable \(k\); the experiment’s globally calibrated, downstream-dependent \(k^*\) is not automatically covered. Finally, the bound allows a radial advantage: “nothing … better than chance” at report:18–19 is too strong.

7. **The numerical Proposition 2 check does not test the conditional expectation claimed.**  
   [scratch/t1_prop2.py:14](/home/agent/workspaces/mats/work/kstar-frames/scratch/t1_prop2.py:14).

   It averages inputs through two fixed complete networks per configuration. It does not fix the past and resample independent futures. It also filters only output survival, rather than explicitly enforcing the theorem’s spectral and active-dimension conditions. The observed inequalities are useful sanity checks, but are weak evidence for the conditional invariance statement. The proof, with the qualification above, carries the result.

8. **The scorecard violates the written net-averaging rule, changing P2’s verdict.**  
   [PREREG.md:60](/home/agent/workspaces/mats/work/kstar-frames/PREREG.md:60) says “path means, then net means.” [scratch/r9_score.py:79](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:79) instead tests each seed separately.

   Recalculation using the written aggregation gives:

   - **P2: 0/90 failures; holds**, rather than 2/270 failures.
   - P5 one-step: **68/90 failures**, rather than 164/270.
   - P5 eight-step: **48/48 failures**, rather than 144/144.
   - P11(c), applying the general net-mean rule: **2/144 failures**, rather than 12/431.

   Per-net failures are valuable supplementary results, but must be distinguished from preregistered scoring. P11(c) still fails either way.

9. **The reported “median ball masses” are medians across nets of path means.**  
   Report:48–53, :183–190; [scratch/r2b_analyze.py:95](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:95).

   For example, width 128, depth 16 has saved per-net path medians **0, 0, 0**. The reported approximately \(5\times10^{-6}\) is the middle **path mean**, \(4.88758\times10^{-6}\). See [runs/Bres_n128_s401_L16.json:303](/home/agent/workspaces/mats/work/kstar-frames/runs/Bres_n128_s401_L16.json:303).

   At depth 32, the medians across the saved per-net path medians are:

   | Width | Reported “median” | Median of per-net path medians |
   |---:|---:|---:|
   | 32 | 0.09 | 0.042664 |
   | 64 | 0.013 | 0.004333 |
   | 128 | 0.0012 | 0.000244 |

   Rename the statistic explicitly. It describes average pair mass on a median net, not the typical new input’s mass.

10. **The learning curves reuse supervision beyond the stated training size, including validation labels.**  
    [scratch/r2_big.py:69](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2_big.py:69), :80; [scratch/r4_mlp.py:30](/home/agent/workspaces/mats/work/kstar-frames/scratch/r4_mlp.py:30), :44, :82.

    `GM` is estimated from **all** reference paths. Every learning-curve run starts from it, including the nominal \(N=1024\) run. Consequently, the initialization contains information from the validation subset and from paths beyond \(N\).

    I found no corresponding leakage from the final held-out B paths: they use a distinct random stream and are not used for early stopping. Final B scores remain meaningful, but the curves are not end-to-end sample-complexity curves with \(N\) labeled paths, and validation is not independent of initialization.

    Comparisons also mix two-net learned means with three-net fixed means. For example, width 64/depth 16 has a **paired** mean improvement of **0.06766 CS**, whereas subtracting the displayed unmatched means gives **0.08591**.

11. **The “optimal fixed frame” is optimal only before target-dependent projection.**  
    [scratch/r2_big.py:70](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2_big.py:70); [scratch/r2b_analyze.py:46](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:46).

    Taking the leading eigenvectors of the averaged normalized sensitivity correctly maximizes empirical mean CS among **unmasked fixed orthonormal frames**.

    It does not maximize
    \[
    E\,CS_x\!\left(\operatorname{orth}(D_xG)\right).
    \]
    Masking and reorthogonalization make that a different nonlinear optimization problem. The report’s main “best fixed” baseline uses this latter operation without optimizing its objective. Rename it “projected unmasked-optimal frame,” or optimize the projected baseline before claiming optimality.

12. **A fresh-noise frame is a repeatability benchmark, not a prediction ceiling.**  
    [notes/questions.md:37](/home/agent/workspaces/mats/work/kstar-frames/notes/questions.md:37); [scratch/r1_layers.py:97](/home/agent/workspaces/mats/work/kstar-frames/scratch/r1_layers.py:97).

    The resampling implementation correctly holds \(z_c\) fixed and redraws noise from layer \(c\) onward. The interpretation is wrong: a frame optimized against the conditional expected normalized sensitivity can outperform a frame from one independent noise draw. The code’s `noise_avg4` already demonstrates higher CS.

    For overlap, the optimal conditional predictor uses the leading eigenspace of \(E[P_F\mid z_c]\); independent-frame overlap is a different quantity. Also, `noise_avg4` normalizes by total energy, rather than the top-\(k\) energy defining CS, so it is not exactly the CS-optimal conditional estimator either.

13. **The active-count split supports kernel loss, but does not establish the entire multi-step explanation.**  
    Report:116–129; [scratch/r10_tables.py:51](/home/agent/workspaces/mats/work/kstar-frames/scratch/r10_tables.py:51).

    I reproduced the split. At depth 32, no-growth/growth overlaps are:

    - Width 32: **0.994775 / 0.807024**.
    - Width 64: **0.991576 / 0.854655**.
    - Width 128: **0.990950 / 0.879875**.

    This strongly supports Proposition 1(b)’s mechanism. But the invisible directions are generic kernel combinations within the active space, not necessarily the coordinate axes of newly activated neurons.

    Amplification is mathematically plausible, but the analysis does not separately measure discarded components, conditioning, and truncation errors. The square-block surrogate omitted the varying rectangular dimensions already present in Theorem C.

    “At chance after eight steps” is also overstated for dual transport: its net-averaged excess over chance is **0.05196 on average**, reaching **0.10336**. “Much closer to chance” is supported.

14. **Greedy coverage and unresolved R99 are overinterpreted.**  
    [lib/cover.py:50](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:50); [scratch/r2b_analyze.py:82](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:82).

    The B1/B2 separation is correctly implemented. However, greedy selection stops once B1 reaches 99%, even when additional centers could improve B2 coverage. Failure of that stopped sequence to reach 99% on B2 is not failure “with all of A.” The separately computed union statistic is the relevant finite-bank check.

    The R99 implementation correctly follows the preregistered Poisson formula. Nevertheless:

    - Empirical zero masses are unresolved probabilities, not established population zeros.
    - Failure of a random reference bank is not a lower bound on an optimized atlas.
    - \(e^{-Rp}\) is a Poisson approximation; fixed-count independent sampling gives \((1-p)^R\).
    - Report:192’s “a few dozen frames cover 99% at every width” is not supported by the recorded stopped greedy results; several R99 values also exceed 100.

    The COHERENT gate code at `r9_score.py:425` checks only the extrapolation, omitting the preregistered alternative of a resolved direct count ≤ \(10^4\). This does not rescue the current Q3 decision, but is a rule-implementation discrepancy.

15. **The coherence result uses a much looser precision than the path’s own frame.**  
    Report:20–22, :155–174; [scratch/r3_pairs.py:62](/home/agent/workspaces/mats/work/kstar-frames/scratch/r3_pairs.py:62).

    The headline radius is defined by **CS = 0.9**, not high subspace overlap or near-own-frame accuracy. At width 128, depth 32, cut 16, the scorecard gives radius **0.0897** for projected CS ≥ 0.9, but only **0.0190** for raw overlap ≥ 0.9. The own-frame tail is around \(10^{-4}\). These are substantially different tolerances.

    The surrogate predicts **raw mean CS** under independently flipped masks; the headline uses **projected median CS** under actual coupled paths. Its numerical agreement is useful heuristic evidence, not a matched validation.

    I found the two-replica recurrence correctly transcribed. Its normalized moment ratio remains heuristic for mean normalized coherence, and conditioning real masks on weights violates the independent-mask derivation, as the report acknowledges.

16. **Uncertainty and alternative explanations need more weight.**  
    Report:145–150, :195–207, :218–230.

    Three networks per width, two for learned maps, and shared prefixes across depths do not establish width independence or asymptotic scaling. Thousands of paths do not replace independent networks. The small-ball counts are dependent pair statistics, so “100 pairs” is a resolution rule, not an uncertainty guarantee.

    Plausible alternatives to intrinsic unpredictability include CS-only training despite an overlap gate, weakly weighted directions near the singular cutoff, architectural/optimization limitations, and discarded state magnitude under fixed absolute noise scales. No experiment separates these explanations.

    The longer training check is useful but covers one configuration; its saved patience is 30, versus the script’s default 10. Report that change alongside the epoch cap. Finally, “1–7% of each input’s sensitivity” should say **mean residual sensitivity**; means do not establish a per-input guarantee.

17. **Preregistration integrity checks pass, but execution was incomplete in declared places.**  
    [PREREG.md:43](/home/agent/workspaces/mats/work/kstar-frames/PREREG.md:43), :47, :102.

    I reconstructed the pre-amendment text and verified its original hash, verified the amended document’s hash, and verified the listed theory/code/log hashes. Amendment 1 explicitly declares the observed width-32 A–C and width-64/128 E data, including the outcome-informed median choice. That disclosure is good.

    Remaining discrepancies:

    - The promised kernel-smoothed-reference diagnostic is absent.
    - The promised net-401 learned runs at cuts 8 and 24 are absent at all three widths; the report uses net 400 alone there.
    - P12 is scored using a forward secant across two grid intervals, rather than the stored local gradient. The resulting reported slopes are **13.77/6.15**; the stored gradients give **15.01/6.74**. Both pass, but the estimator choice should be stated.
    - The net-aggregation discrepancy in finding 8 is not covered by the amendment.

18. **The remaining core code checks passed; some edge cases remain unsafe.**  
    Both supplied test suites passed under the requested CPU/thread limits. Jacobian chains, past products, ordinary frame scores, marginal-gate product indexing, and loop/batched/eigh transport tests check out. The maximum saved truncation bound is **\(1.99\times10^{-7}\)**, too small to explain the headline differences.

    Generic `orth()` uses QR without rank detection, so rank-deficient projected frames receive arbitrary extra directions. The covering amendment’s range-projector convention is therefore not consistently applied to other projected predictors. In three checked midpoint nearest-reference configurations, all selected projections were full rank, so this did not affect those checks. Also, `n_surv()` returns \(n\) for an exactly zero Jacobian because it counts \(0\ge0\); dead paths need an explicit convention.

19. **Numerical spot-checks: these report values reproduce from the saved artifacts.**  
    The following table checks **19 scalar quantities**, in addition to the discrepancies and growth splits above. Values are shown before report rounding.

    | Configuration and statistic | Recomputed value |
    |---|---:|
    | \(n=32,L=32,c=16\): mean \(k\), chance | 5.333333; 0.373938 |
    | Same: push-forward one/eight-step overlap | 0.605400; 0.369822 |
    | Same: dual one/eight-step overlap | 0.905225; 0.420421 |
    | \(n=64,L=32,c=16\): dual one-step; new-noise overlap | 0.943966; 0.971005 |
    | \(n=128,L=32,c=16\): dual one-step; backward overlap | 0.955420; 0.988190 |
    | Same: projected fixed CS/overlap | 0.871011; 0.678632 |
    | Same: learned CS/overlap | 0.950872; 0.766545 |
    | \(n=64,L=16,c=8\): learned CS/overlap | 0.982093; 0.833159 |
    | \(n=128,L=32,c=16\): reported-procedure coherence radius | 0.089666 |
    | Longer-training check: epochs; held-out CS | 437; 0.959184 |

    Sources: `runs/A_*.json`, `runs/Bres_*.json`, `runs/D_*.json`, the run-C arrays, and `runs/posthoc/D_n64_s400_L32_c16_N16384.json`; corresponding summaries appear in `logs/scorecard.log`, `logs/tables.md`, and the GPU/post-hoc logs.

**Short verdict:** retain the conditional symmetry theorem, measured transport failure, and positive learned-map result. Repair the covering numerics and scorecard, qualify the atlas extrapolation, and replace the universal negative conclusion with “the tested methods did not clear the preregistered gate.” Not running Q3 remains justified by that gate; no impossibility result about the joint law follows.
tokens used
158,617
**Verdict: major revisions needed.** The core experiments are useful, but I found a reproducible covering-score bug, incorrect scorecard aggregation, and claims stronger than the evidence. No files were edited.

References below use **report** for [kstar-frames-2026-10-30.md](/home/agent/workspaces/mats/outbox/kstar-frames-2026-10-30.md); other paths are relative to the supplied work directory.

1. **The covering calculation produces impossible scores and false covers.**  
   In [lib/cover.py:34](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:34), the Gram matrix is formed in float32, then inverted in float64 with an eigenvalue cutoff of \(10^{-10}\). Casting afterward cannot recover the lost precision. Small genuine eigenvalues and rounding errors are amplified.

   Concrete reproduction: `B_n32_s401_L16_c8.npz`, held-out index **1106**, reference index **11345**, zero-based. With the production-sized batch, `pair_scores` returns **CS = 2.1550903**. A float64 SVD projector gives **0.2671709**. This falsely declares coverage at CS ≥ 0.9. For a projector of rank at most \(k\), Ky Fan’s inequality requires CS ≤ 1.

   The saved covering matrices contain **34 entries above 1.001**, across five width-32/64 configurations. Stable recomputation puts **33 of those below 0.9**. These are only detected outliers; numerically wrong scores inside [0,1] remain possible. Recompute affected covering results using stable projectors before treating their masses, greedy choices, or R99 as verified. The existing covering test passes because its cases do not expose this conditioning problem.

2. **“No predictable, shared map” is not established.**  
   Report:218–230 moves from failure of particular predictors and a finite reference bank to “There is no shared exact one” and an apparent exhaustive list of estimator options.

   The supported conclusion is: **none of the tested methods met the preregistered joint CS/overlap gate in all configurations**. Proposition 2 excludes only predictors measurable from the past alone. It excludes neither predictors using downstream weights nor better learned algorithms. Indeed, the learned map is positive evidence for substantial predictability.

   The distinction from Kelvin’s joint-law premise at report:57–61 is correct. However, these experiments neither bound that law’s description length nor establish an estimator lower bound. Computing derivatives per evaluation does not itself force Monte Carlo; deterministic quadrature and other integration schemes can also evaluate derivatives.

3. **The heuristic atlas count is not a covering-number estimate with a demonstrated calibration.**  
   Report:195–207; [scratch/r9_score.py:323](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:323).

   \(C(\theta)\) is a **mean state-space pair mass**. Its reciprocal is generally neither an optimal frame-covering count nor a 99%-coverage count. The conversion lacks three ingredients:

   - A guarantee that state proximity implies the required frame score; the radius uses a conditional **median**.
   - Control of low-mass states, which dominate high-coverage requirements.
   - Control of distant states sharing frames, already observed here.

   The reference-frame loss is also asymmetric and includes target-dependent masking; it is not an ordinary subspace metric. Labeling the calculation HEURISTIC is appropriate, but “order of magnitude at best,” “conservative,” and the exponential-atlas interpretation still suggest more calibration than has been demonstrated. The acknowledged approximately 100-fold state/frame mass discrepancy is already larger than one order of magnitude.

4. **P13’s numerical pass depends on unspecified analysis choices.**  
   [scratch/r9_score.py:228](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:228), :368–378.

   The implementation uses 24 logarithmic angle bins, accepts bins with eight observations, takes the first median crossing, and assigns the geometric bin midpoint rather than the observed median angle. These choices were not fixed in the preregistration.

   On the same width-128, depth-32 data:

   | Angle bins | Heuristic log10 count |
   |---:|---:|
   | 16 | 5.939 |
   | 20 | 5.975 |
   | 24, reported | 6.046 |
   | 40 | 5.839 |
   | 48 | 5.748 |

   Thus P13(b)’s threshold of 6 changes verdict under modest binning changes. Also, the code applies the **across-net median radius to every net’s state distribution**. Using each net’s own measured radius changes the depth-32 average from **6.046 to 5.696**. These are sensitivity checks, not proposed replacement scoring rules.

   An uncrossed curve is right-censored at the observed angles, not evidence that the radius is infinite. The declared median amendment mitigates one aggregation problem but does not resolve this censoring issue.

5. **Proposition 1’s linear algebra is essentially right, but its headline interpretation drops necessary conditions.**  
   [notes/props.md:7](/home/agent/workspaces/mats/work/kstar-frames/notes/props.md:7); report:17–29, :86–93, :125–129.

   The row-space identity and pseudoinverse-projector identity are correct. Exact recovery requires \(J_c\) to have rank exactly \(k\), and its row space to lie in \(\operatorname{range}(B_c)\). Non-growing active count makes the latter automatic for a generic full-row-rank active block.

   However:

   - If \(B_c^T\) reduces rank, the previous “top-\(k\) frame” includes arbitrary null-space completion; it is not simply the lower-dimensional row space.
   - Proposition 1(c)’s “iff invariant” additionally needs rank preservation. An invariant subspace contained in the kernel is an immediate counterexample to sufficiency.
   - The actual \(k^*\) is a thresholded effective rank. Finite-horizon top-\(k\) singular subspaces are generally **not exactly inverse-transpose covariant**.
   - The claimed exact all-layer backward construction costing \(O(L'n^2k^*)\) is not demonstrated. Run A forms full Jacobians and decomposes them. Propagating truncated frames backward is approximate, as the measured overlap below 1 confirms.

6. **Proposition 2’s invariance proof works with explicit spectral-event conditioning; the report omits that qualification.**  
   [notes/props.md:23](/home/agent/workspaces/mats/work/kstar-frames/notes/props.md:23); report:96–104.

   Line by line:

   - Given the past, \(e_c,B\) are fixed; Gaussian orthogonal projections make \(g=W_{c+1}e_c\) and \(M=W_{c+1}B\) independent.
   - At \(\sigma=0\), the future **trajectory and gates** depend on \(W_{c+1}\) only through \(g\). The Jacobian additionally depends on \(M\). The displayed \(QD_{c+1}M\) is consistent because the extra mask is redundant.
   - \(MO\) has the same conditional law for every orthogonal \(O\), independently of \(g\) and later weights.
   - Therefore the Jacobian law is invariant under right multiplication by \(\operatorname{diag}(1,O)\).
   - Where the top-\(k\) spectral projector is uniquely defined, its expectation has form \(\operatorname{diag}(\alpha,\beta I)\). The trace identity and stated inequality follow.

   The repair is to write the expectation conditional on a suitable invariant event, such as a positive spectral gap at the cutoff. Dead or rank-deficient future products have positive probability, and an arbitrary SVD tie-breaking rule need not be equivariant. Merely assuming \(a_c\ne0,m_c\ge3\), as the report does, does not address this.

   The theorem also needs fixed or appropriately past-measurable \(k\); the experiment’s globally calibrated, downstream-dependent \(k^*\) is not automatically covered. Finally, the bound allows a radial advantage: “nothing … better than chance” at report:18–19 is too strong.

7. **The numerical Proposition 2 check does not test the conditional expectation claimed.**  
   [scratch/t1_prop2.py:14](/home/agent/workspaces/mats/work/kstar-frames/scratch/t1_prop2.py:14).

   It averages inputs through two fixed complete networks per configuration. It does not fix the past and resample independent futures. It also filters only output survival, rather than explicitly enforcing the theorem’s spectral and active-dimension conditions. The observed inequalities are useful sanity checks, but are weak evidence for the conditional invariance statement. The proof, with the qualification above, carries the result.

8. **The scorecard violates the written net-averaging rule, changing P2’s verdict.**  
   [PREREG.md:60](/home/agent/workspaces/mats/work/kstar-frames/PREREG.md:60) says “path means, then net means.” [scratch/r9_score.py:79](/home/agent/workspaces/mats/work/kstar-frames/scratch/r9_score.py:79) instead tests each seed separately.

   Recalculation using the written aggregation gives:

   - **P2: 0/90 failures; holds**, rather than 2/270 failures.
   - P5 one-step: **68/90 failures**, rather than 164/270.
   - P5 eight-step: **48/48 failures**, rather than 144/144.
   - P11(c), applying the general net-mean rule: **2/144 failures**, rather than 12/431.

   Per-net failures are valuable supplementary results, but must be distinguished from preregistered scoring. P11(c) still fails either way.

9. **The reported “median ball masses” are medians across nets of path means.**  
   Report:48–53, :183–190; [scratch/r2b_analyze.py:95](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:95).

   For example, width 128, depth 16 has saved per-net path medians **0, 0, 0**. The reported approximately \(5\times10^{-6}\) is the middle **path mean**, \(4.88758\times10^{-6}\). See [runs/Bres_n128_s401_L16.json:303](/home/agent/workspaces/mats/work/kstar-frames/runs/Bres_n128_s401_L16.json:303).

   At depth 32, the medians across the saved per-net path medians are:

   | Width | Reported “median” | Median of per-net path medians |
   |---:|---:|---:|
   | 32 | 0.09 | 0.042664 |
   | 64 | 0.013 | 0.004333 |
   | 128 | 0.0012 | 0.000244 |

   Rename the statistic explicitly. It describes average pair mass on a median net, not the typical new input’s mass.

10. **The learning curves reuse supervision beyond the stated training size, including validation labels.**  
    [scratch/r2_big.py:69](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2_big.py:69), :80; [scratch/r4_mlp.py:30](/home/agent/workspaces/mats/work/kstar-frames/scratch/r4_mlp.py:30), :44, :82.

    `GM` is estimated from **all** reference paths. Every learning-curve run starts from it, including the nominal \(N=1024\) run. Consequently, the initialization contains information from the validation subset and from paths beyond \(N\).

    I found no corresponding leakage from the final held-out B paths: they use a distinct random stream and are not used for early stopping. Final B scores remain meaningful, but the curves are not end-to-end sample-complexity curves with \(N\) labeled paths, and validation is not independent of initialization.

    Comparisons also mix two-net learned means with three-net fixed means. For example, width 64/depth 16 has a **paired** mean improvement of **0.06766 CS**, whereas subtracting the displayed unmatched means gives **0.08591**.

11. **The “optimal fixed frame” is optimal only before target-dependent projection.**  
    [scratch/r2_big.py:70](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2_big.py:70); [scratch/r2b_analyze.py:46](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:46).

    Taking the leading eigenvectors of the averaged normalized sensitivity correctly maximizes empirical mean CS among **unmasked fixed orthonormal frames**.

    It does not maximize
    \[
    E\,CS_x\!\left(\operatorname{orth}(D_xG)\right).
    \]
    Masking and reorthogonalization make that a different nonlinear optimization problem. The report’s main “best fixed” baseline uses this latter operation without optimizing its objective. Rename it “projected unmasked-optimal frame,” or optimize the projected baseline before claiming optimality.

12. **A fresh-noise frame is a repeatability benchmark, not a prediction ceiling.**  
    [notes/questions.md:37](/home/agent/workspaces/mats/work/kstar-frames/notes/questions.md:37); [scratch/r1_layers.py:97](/home/agent/workspaces/mats/work/kstar-frames/scratch/r1_layers.py:97).

    The resampling implementation correctly holds \(z_c\) fixed and redraws noise from layer \(c\) onward. The interpretation is wrong: a frame optimized against the conditional expected normalized sensitivity can outperform a frame from one independent noise draw. The code’s `noise_avg4` already demonstrates higher CS.

    For overlap, the optimal conditional predictor uses the leading eigenspace of \(E[P_F\mid z_c]\); independent-frame overlap is a different quantity. Also, `noise_avg4` normalizes by total energy, rather than the top-\(k\) energy defining CS, so it is not exactly the CS-optimal conditional estimator either.

13. **The active-count split supports kernel loss, but does not establish the entire multi-step explanation.**  
    Report:116–129; [scratch/r10_tables.py:51](/home/agent/workspaces/mats/work/kstar-frames/scratch/r10_tables.py:51).

    I reproduced the split. At depth 32, no-growth/growth overlaps are:

    - Width 32: **0.994775 / 0.807024**.
    - Width 64: **0.991576 / 0.854655**.
    - Width 128: **0.990950 / 0.879875**.

    This strongly supports Proposition 1(b)’s mechanism. But the invisible directions are generic kernel combinations within the active space, not necessarily the coordinate axes of newly activated neurons.

    Amplification is mathematically plausible, but the analysis does not separately measure discarded components, conditioning, and truncation errors. The square-block surrogate omitted the varying rectangular dimensions already present in Theorem C.

    “At chance after eight steps” is also overstated for dual transport: its net-averaged excess over chance is **0.05196 on average**, reaching **0.10336**. “Much closer to chance” is supported.

14. **Greedy coverage and unresolved R99 are overinterpreted.**  
    [lib/cover.py:50](/home/agent/workspaces/mats/work/kstar-frames/lib/cover.py:50); [scratch/r2b_analyze.py:82](/home/agent/workspaces/mats/work/kstar-frames/scratch/r2b_analyze.py:82).

    The B1/B2 separation is correctly implemented. However, greedy selection stops once B1 reaches 99%, even when additional centers could improve B2 coverage. Failure of that stopped sequence to reach 99% on B2 is not failure “with all of A.” The separately computed union statistic is the relevant finite-bank check.

    The R99 implementation correctly follows the preregistered Poisson formula. Nevertheless:

    - Empirical zero masses are unresolved probabilities, not established population zeros.
    - Failure of a random reference bank is not a lower bound on an optimized atlas.
    - \(e^{-Rp}\) is a Poisson approximation; fixed-count independent sampling gives \((1-p)^R\).
    - Report:192’s “a few dozen frames cover 99% at every width” is not supported by the recorded stopped greedy results; several R99 values also exceed 100.

    The COHERENT gate code at `r9_score.py:425` checks only the extrapolation, omitting the preregistered alternative of a resolved direct count ≤ \(10^4\). This does not rescue the current Q3 decision, but is a rule-implementation discrepancy.

15. **The coherence result uses a much looser precision than the path’s own frame.**  
    Report:20–22, :155–174; [scratch/r3_pairs.py:62](/home/agent/workspaces/mats/work/kstar-frames/scratch/r3_pairs.py:62).

    The headline radius is defined by **CS = 0.9**, not high subspace overlap or near-own-frame accuracy. At width 128, depth 32, cut 16, the scorecard gives radius **0.0897** for projected CS ≥ 0.9, but only **0.0190** for raw overlap ≥ 0.9. The own-frame tail is around \(10^{-4}\). These are substantially different tolerances.

    The surrogate predicts **raw mean CS** under independently flipped masks; the headline uses **projected median CS** under actual coupled paths. Its numerical agreement is useful heuristic evidence, not a matched validation.

    I found the two-replica recurrence correctly transcribed. Its normalized moment ratio remains heuristic for mean normalized coherence, and conditioning real masks on weights violates the independent-mask derivation, as the report acknowledges.

16. **Uncertainty and alternative explanations need more weight.**  
    Report:145–150, :195–207, :218–230.

    Three networks per width, two for learned maps, and shared prefixes across depths do not establish width independence or asymptotic scaling. Thousands of paths do not replace independent networks. The small-ball counts are dependent pair statistics, so “100 pairs” is a resolution rule, not an uncertainty guarantee.

    Plausible alternatives to intrinsic unpredictability include CS-only training despite an overlap gate, weakly weighted directions near the singular cutoff, architectural/optimization limitations, and discarded state magnitude under fixed absolute noise scales. No experiment separates these explanations.

    The longer training check is useful but covers one configuration; its saved patience is 30, versus the script’s default 10. Report that change alongside the epoch cap. Finally, “1–7% of each input’s sensitivity” should say **mean residual sensitivity**; means do not establish a per-input guarantee.

17. **Preregistration integrity checks pass, but execution was incomplete in declared places.**  
    [PREREG.md:43](/home/agent/workspaces/mats/work/kstar-frames/PREREG.md:43), :47, :102.

    I reconstructed the pre-amendment text and verified its original hash, verified the amended document’s hash, and verified the listed theory/code/log hashes. Amendment 1 explicitly declares the observed width-32 A–C and width-64/128 E data, including the outcome-informed median choice. That disclosure is good.

    Remaining discrepancies:

    - The promised kernel-smoothed-reference diagnostic is absent.
    - The promised net-401 learned runs at cuts 8 and 24 are absent at all three widths; the report uses net 400 alone there.
    - P12 is scored using a forward secant across two grid intervals, rather than the stored local gradient. The resulting reported slopes are **13.77/6.15**; the stored gradients give **15.01/6.74**. Both pass, but the estimator choice should be stated.
    - The net-aggregation discrepancy in finding 8 is not covered by the amendment.

18. **The remaining core code checks passed; some edge cases remain unsafe.**  
    Both supplied test suites passed under the requested CPU/thread limits. Jacobian chains, past products, ordinary frame scores, marginal-gate product indexing, and loop/batched/eigh transport tests check out. The maximum saved truncation bound is **\(1.99\times10^{-7}\)**, too small to explain the headline differences.

    Generic `orth()` uses QR without rank detection, so rank-deficient projected frames receive arbitrary extra directions. The covering amendment’s range-projector convention is therefore not consistently applied to other projected predictors. In three checked midpoint nearest-reference configurations, all selected projections were full rank, so this did not affect those checks. Also, `n_surv()` returns \(n\) for an exactly zero Jacobian because it counts \(0\ge0\); dead paths need an explicit convention.

19. **Numerical spot-checks: these report values reproduce from the saved artifacts.**  
    The following table checks **19 scalar quantities**, in addition to the discrepancies and growth splits above. Values are shown before report rounding.

    | Configuration and statistic | Recomputed value |
    |---|---:|
    | \(n=32,L=32,c=16\): mean \(k\), chance | 5.333333; 0.373938 |
    | Same: push-forward one/eight-step overlap | 0.605400; 0.369822 |
    | Same: dual one/eight-step overlap | 0.905225; 0.420421 |
    | \(n=64,L=32,c=16\): dual one-step; new-noise overlap | 0.943966; 0.971005 |
    | \(n=128,L=32,c=16\): dual one-step; backward overlap | 0.955420; 0.988190 |
    | Same: projected fixed CS/overlap | 0.871011; 0.678632 |
    | Same: learned CS/overlap | 0.950872; 0.766545 |
    | \(n=64,L=16,c=8\): learned CS/overlap | 0.982093; 0.833159 |
    | \(n=128,L=32,c=16\): reported-procedure coherence radius | 0.089666 |
    | Longer-training check: epochs; held-out CS | 437; 0.959184 |

    Sources: `runs/A_*.json`, `runs/Bres_*.json`, `runs/D_*.json`, the run-C arrays, and `runs/posthoc/D_n64_s400_L32_c16_N16384.json`; corresponding summaries appear in `logs/scorecard.log`, `logs/tables.md`, and the GPU/post-hoc logs.

**Short verdict:** retain the conditional symmetry theorem, measured transport failure, and positive learned-map result. Repair the covering numerics and scorecard, qualify the atlas extrapolation, and replace the universal negative conclusion with “the tested methods did not clear the preregistered gate.” Not running Q3 remains justified by that gate; no impossibility result about the joint law follows.
