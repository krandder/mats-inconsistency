# Part 2 research-code review

Checked 2026-10-29 07:04 UTC. I read `code/PREREG-TAILS.md`, including both amendments, before reviewing the implementation. The files in `code/` still match every entry in `reviewed_sha256.txt`. This review makes no source changes.

**The splitting estimator and ARC adaptation survived the checks. The scoring does not yet implement the registered comparison, and one discrepancy reverses U1's kill verdict.** Findings are ordered by consequence. “Critical” below means a demonstrated change to candidate selection, not evidence that the underlying probability estimator is biased.

Reproduction scripts and their output are in `checks/`. They use `/home/agent/.venvs/arc-noisy/bin/python`, `OMP_NUM_THREADS=1`, CPU only, and at most two concurrent processes. The small ARC test uses a local copy of its source so import-time caches stay inside this review directory. Saved U1/U3/U4/U5 development results were read from the live `tails/runs/arcn-tdev1/` directory; no new large network simulations were run.

## Findings

1. **Critical: U1 is killed by an unregistered cutoff; it passes the stated amended rule.**

   Location: `code/killtests.py:94`, `code/PREREG-TAILS.md:165–173`.

   The code implements `rmse < 0.999`, then prints “strict rMSE < 1.” The two predicates differ on actual development events. For seed 105, outputs 2 and 6, L=32, k=8, the relative MSEs are respectively **0.9999917592177527** and **0.9999999454155093**. Both have nonzero estimates; these are not rounding artifacts from all-zero replicates. Both beat C-GS on the registered amortised-B50 comparison: approximately 7.79e9 versus 8.54e11 and 8.61e11 FLOPs.

   `checks/check_mode_kill.py` gives **17/32 = 0.53125**, not 15/32 = 0.46875, under the written `< 1` rule. U1 therefore passes the “at least half” bar. Adding the omitted hidden-layer recomputation cost in finding 7 still gives 17/32. `checks/killtests.txt` reproduces the published 0.47 only with the code's stricter cutoff.

   Fix: implement the stated predicate and correct the kill record and downstream eligibility. Compute relative MSE as `mean((E / p - 1)**2)` so all-zero replicates give exactly 1. If estimates this close to zero should also fail, that needs an explicit further protocol change; it is not equivalent to amendment 2. Passing this empirical rule is not evidence that eight replicates reliably characterize U1's variance.

2. **Major: amendment 2 is not applied to Kerdock, simulated MC, or C-GS.**

   Location: `code/analyze_tails.py:53–54,120`, `code/common.py:60–63`.

   Kerdock and simulated MC still use `rmse <= 1`; C-GS assigns a finite B50 without any tested-budget eligibility check. The strict rule was explicitly made applicable to every randomized method. In the supplied development truth, **36/160 L=32 events have single-run relative variance at least 1**, yet all receive finite C-GS B50.

   Concrete check: `checks/check_review.py` supplies 16 all-zero Kerdock replicates. The scorer assigns finite B50 to **12 of the 16 output/threshold pairs at k=4,6**. The other four are excluded only because the current expression rounds their mathematically unit rMSE slightly above 1. Changing only `<=` to `<` is insufficient to make all-zero handling numerically reliable.

   Fix: centralize the amended eligibility predicate, use normalized residuals, explicitly reject all-zero replicates, and apply the registered tested-budget condition to C-GS as well. If C-GS is to be exempt because its theoretical unbiasedness permits extrapolation, record that as a protocol change rather than silently exempting this competitor.

3. **Major: provisional truth never filters any result, and missing MC validation is silently accepted.**

   Location: `code/analyze_tails.py:31–44,161–205`.

   The code computes provisional flags but uses them only in a printed count. B50 summaries, reached fractions, better-than-zero decisions and sample-free comparisons all include every event. No “without provisional events” result exists, although section 2 requires both versions of every claim. Already **25/160 development events at L=32** fail the RSE/split-half checks, before adding cross-check failures.

   Cross-check (b) is simply skipped if its file is missing. In the missing-file fixture, all eight MC-failure bits remain clear. A zero MC count also produces an enormous relative SE that lets the log-ratio check pass; a zero count should not certify agreement through this undefined relative-error approximation.

   Fix: produce paired all-event and validated-event tables and comparisons, using the same mask for competing methods. Record missing/inapplicable/failed checks separately; require the registered MC sample size and treat missing or unusable validation as unverified/provisional. Use a suitable binomial bound for zero counts.

4. **Major: Kerdock's first-layer multiplication is counted twice.**

   Location: `code/samplers.py:60–61`, `code/kerdock_tail.py:34,48`, `code/analyze_tails.py:118–120`.

   `z1_points` charges `2*n**3` for each basis product `Bk @ W1Q.T` and returns this inside `setup_flops`. The scorer then adds `n**2` per point for the same first-layer work. With N=65,536 and n=256, 128 basis products contribute **4,294,967,296 FLOPs** to setup, and the per-point calculation charges another **4,294,967,296**. The intended reduced first-layer cost itself is correct; the overlap with “setup” is the problem.

   There is also a cost-model distinction to settle: fresh rotations and basis products are per-replicate/per-block work, not a reusable fit. Extrapolating variance by averaging independently randomized blocks requires repeating their costs. The current formula pays all `setup_flops` once while scaling only `per`; this follows amendment 1's algebra but misclassifies these operations as reusable setup.

   Fix: return separate rotation/folding, first-layer, radius-generation and reusable-setup counters. Charge each operation once. State which blocks must be regenerated when reducing variance and scale those costs with the run count; reconcile that distinction with amendment 1.

5. **Major: reported B50 and better-than-zero statistics use the wrong median.**

   Location: `code/analyze_tails.py:177–178,191–193`.

   Section 4's cell summary is the median over the 160 events, with uncertainty obtained by resampling nets. The implementation first takes each net's median and then takes the median of those medians. Net-level medians belong in the separately specified win-ratio statistic, not in the overall event summary.

   Concrete check: ten nets each containing `[1,1,1,1,1,100,100,100]` and ten containing eight 10s have pooled median **10**, while the code reports **5.5**. Better-than-zero uses the same incorrect reduction, although the printed relative-error summaries use the pooled median.

   Fix: resample whole nets and compute the pooled event median within each bootstrap replicate. Preserve the specified median-of-net-ratios calculation separately for win inference. Bootstrap the reached/error summaries too if supplying the registered summary CIs.

6. **Major: the scorer does not produce the registered method selection and win analysis.**

   Location: `code/analyze_tails.py:84–110,183–205`.

   Biased estimators remain separate `@m20` and `@m22` rows; no method-level B50 selects the smallest successful tested budget per event. There is no sampling/hybrid best-competitor comparison or bootstrap CI for the per-net B50 ratio, and randomized better-than-zero at the two specified budgets is absent. The comment saying those results are “in the report” is not an implementation or evidence of them in the supplied code.

   The only comparison block is labeled “sample-free win” in source and evaluates killed U2 variants against selected K3 competitors. Its output lacks the required post-hoc/excluded-from-wins label. It neither selects the best competitor, including C-K2, nor checks the equal-or-lower-cost premise for using success fractions. For the saved seed-102 L=4 result, U2 costs **2.29 times C-K3f per event**, though it is cheaper under the five-threshold amortisation. The success-fraction shortcut cannot establish a per-event cost win here.

   Fix: aggregate biased-method budgets event by event, implement the category comparator and registered net bootstrap, and compute randomized better-than-zero only at justified measured/analytic budgets. Mark U2 variants post hoc in both human and machine outputs and exclude them from win claims. Keep their accuracy differences as explicitly descriptive comparisons; apply the cost premise when using any fraction-based substitute.

7. **Major: U1's cost omits a full hidden-network evaluation at every readout band.**

   Location: `code/mode.py:38–43`, `code/killtests.py:78–85`.

   `readout` recomputes `net.hidden(...)` for all particles. Its cost is additional to the score evaluations counted in `run_evals`; the readout receives latents and scores, not cached hidden states. The kill-test counter adds only the final output matmul and threshold comparisons.

   With L=32, N=2,000 and 23 finite levels, this omits **195,796,992,000 FLOPs per direction per frozen run**. `checks/check_mode_kill.txt` shows the corrected comparisons for the two events relevant to finding 1. This undercharges U1 but does not rescue its recorded kill verdict: all 17 qualifying events still beat GS after the correction.

   Fix: count hidden recomputation in every band, including the final band, or actually cache the required states and count the resulting implementation. Account explicitly for `H.T @ H`, eigendecomposition and mode projection in the setup. Charging `M0*c_path` is not an itemized count of that pilot, even though the unused final-layer path charge approximately offsets the Gram-matrix cost.

8. **Major: infinite B50 values turn bootstrap bounds into NaNs, concealed as “inf.”**

   Location: `code/analyze_tails.py:142–152`.

   `np.quantile`'s default interpolation performs invalid arithmetic between infinite order statistics. `boot_median([np.inf]*20)` returns **`(inf, nan, nan)`**, reproduced in `checks/check_review.txt`. `fmt` then displays NaN as `inf`, hiding the numerical failure, while JSON retains NaNs. Failed events are deliberately infinite in this study, so this is an ordinary case, not an exotic input.

   Fix: use an explicitly specified extended-real/order-statistic quantile rule that preserves infinite bounds. Distinguish invalid numbers from genuine infinite cost. When adding the missing ratio analysis, also specify treatment of finite/inf, inf/finite and inf/inf rather than silently dropping failed events.

9. **Major: the saddlepoint scan can reject a root satisfying the registered conditions.**

   Location: `code/deduce.py:47–60`.

   Each grid step tests `Kpp(nx) <= 0` before checking whether a root lies between the preceding point and `nx`. A step can cross both the root and the curvature boundary, returning NaN despite a valid smaller positive root. The scan also stops at the first nonconvex interval rather than examining later positive-curvature intervals, although the stated selection rule only requires positive curvature at the selected root.

   Concrete check in `checks/check_numerics.py`: cumulants `[0,1,-0.1,-0.01]`, threshold **3.9871647423554393**, root **7.309761264970011**, curvature **0.0018608247487189034**. The implementation returns NaN; the LR formula evaluated at that root is **5.037540235567555e-5**. Thus this can trigger a different-order fallback at a relevant tail scale. I found no such missed admissible root in the 480 supplied development K3f event/threshold pairs; this finding does not explain their recorded kill outcomes.

   Fix: solve the degree-two/three equation directly, or partition the positive axis at the exact roots of K'' and bracket on each interval. Select the smallest admissible real root, check the deviance and finite output, and record the actual fallback used. If only the convex branch connected to zero is intended, state that extra restriction explicitly.

10. **Minor: “all setup and readouts counted” is not true outside the ARC counter.**

    Location: `code/analyze_tails.py:70–82,90–110`, `code/mcx.py:33–38`, `code/samplers.py:31–36,62–64`, `code/common.py:60–63`.

    Moment methods are charged only `M*c_path`; the four accumulated powers, cumulant conversion, Gaussian/Edgeworth/LR evaluations and chi-mixture quadrature are omitted. U2 and U2r receive identical costs despite the latter making 161 conditional saddlepoint evaluations per threshold. Root iterations and fallback evaluations have no counter. Kerdock radius generation/scaling/permutations and GS resampling/adaptation work are likewise absent. Conversely, GS charges the initial draw/evaluation as a full pCN move, slightly overcharging it.

    Fix: add analytic counters for these operations, splitting threshold-independent and per-threshold costs before amortisation. Record actual root/fallback work. These omissions are smaller than findings 4 and 7, but systematically favor the more elaborate readouts and contradict the “everything” convention. The base `c_path` and `c_move` formulas themselves agree with the amended conventions.

11. **Minor: all-dead truth runs crash the loader instead of producing provisional events.**

    Location: `code/common.py:33–35,48–54`; related target-completion assumption at `code/truth.py:47,57`.

    `rel_var([-inf]*16)` raises a zero-size reduction `ValueError`. `logmeanexp` already handles the same case, and the splitting implementation correctly preserves zero probability estimates after death, but the loader cannot consume them. A pilot that dies before visiting every stop, or reaches its cap and jumps past a stop, also leaves the threshold-column lookup indexing an empty array.

    Fix: represent an all-dead reference as unusable/provisional with explicit infinite uncertainty, and prevent division by its zero reference probability during scoring. Verify every requested threshold was visited before frozen estimation; fail with a diagnostic or use a declared retuning policy instead of an indexing exception. No all-dead reference was present in the supplied development files.

12. **Minor: K3's moment documentation and the “no extra cost” amendment are false.**

    Location: `code/kprop_tail.py:61–63`, `code/PREREG-TAILS.md:174`; ARC `kprop_harmonic.py:get_d_max/get_r_x`.

    `k_max=3` is a harmonic approximation budget, not a cutoff setting the fourth cumulant to zero. SIMPLE retains a radial fourth cumulant; AUGMENT retains additional fourth-order structure. In the small-network check the fourth cumulants range from about **0.155 to 7.568**, and the four polynomial readouts match the moments reconstructed from the retained cumulants to **2.84e-14**. Calling them the description's moments is legitimate in that formal sense; the zero-fourth-cumulant explanation is not.

    They also require additional readout work: the saved L=4 K3f result charges **5.912e9** FLOPs for the four moment steps. Sharing the hidden propagation does not make them free. The main scorer does correctly include `flops_moments` in U2's cost.

    Fix: correct the documentation and the post-hoc rationale while preserving the numerical readout. Distinguish moments of the approximate cumulant description from guaranteed moments of a positive probability distribution.

13. **Minor: seed namespaces overlap between development truth and evaluation cross-checks.**

    Location: `code/truth.py:40,53`.

    Additive offsets do not give globally distinct seeds. For example, main truth for `(seed=102,L=32,sigma=.001)` and cross-check (c) for `(seed=0,L=4,sigma=.01)` both use **70128**. The same collision occurs for development seeds 103–107 versus evaluation seeds 1–5. `checks/check_saved.txt` lists these pairs.

    These are different networks/depths; I found no same-event threshold/main/cross-check collision. Thus this does not demonstrate biased truth or an invalid same-event check. It does undermine a blanket claim of disjoint development/evaluation random streams and is avoidable.

    Fix: derive seeds from a structured tuple of role, network, depth, sigma and replicate, with an explicit collision check. Continuing one generator from pilot into frozen runs is not itself adaptive leakage: those phases consume fresh random draws, and the frozen kernels do not inspect acceptance to retune.

## Checked and found correct

- **Fixed-level splitting:** multinomial resampling is uniform over survivors; pCN is reversible for the Gaussian prior and accepting only proposals above the current level preserves the restricted target. Frozen move scales do not adapt to frozen particles. Product-of-fraction estimates are unbiased conditional on the independent pilot, subject to the usual ideal-randomness/exact-arithmetic qualification. Repeated levels have fraction one after selection; dead systems retain zero weight. In-place `Z.copy_(where(...))` uses a separate proposal and a fully evaluated right-hand side; the actual network score does not alias or overwrite the latent.
- **Feynman–Kac readout:** bands are accumulated before selecting at the next level, weighted by the preceding survival product. The final band is sampled after selecting at the last level and uses the last survival weight. Reusing a frozen scale for that final transition preserves invariance. The omitted tail band concern does not apply here. In 3,000 small Gaussian systems, survival estimates and independent/event readouts agreed with analytic values within three Monte Carlo SEs; the constant-one readout was exactly one, repeated-level estimates exactly equal, and dead-system readouts finite. A separate 500-run scalar-GS check also agreed with the Gaussian tail.
- **Latent and noise:** there are exactly L vectors, comprising x and L−1 noises. A direct forward-formula test agreed to floating-point precision; changing the last hidden noise changed z_L by the expected `sigma*r*W_L` contribution, and evaluation left Z unchanged. The coupled x2 construction shares x and supplies independent noise vectors to its two paths.
- **Truth mapping and ordinary summaries:** the first exact threshold occurrence selects the correct probability column, not a padded later column. Means are means of probabilities, not means of logs. Relative SE is sample relative SD divided by sqrt(R), and half comparisons use probability means. The rho=.3 cross-check uses a separate generator. The MC pass records z before ReLU/noise at the requested depth, with distinct tagged streams for MC roles. Same-generator pilot/frozen sequencing does not reuse pilot particles.
- **ARC adaptation:** L−1 ReLUs, Gaussian variance additions after every hidden ReLU including the last, then the final linear layer with bias −t and heaviside. K3f uses factorization and the exact metric. A small noiseless network matched upstream `mlp_kprop` configured like `run_lpe` exactly. Per-output threshold broadcasting is correct; shared propagation and separate terminal-step counters are separated correctly. Polynomial moments passed the independent retained-cumulant reconstruction check described above.
- **Readout formulas:** moment-to-cumulant conversion, Edgeworth signs, and the LR survival correction `sf(w)+pdf(w)*(1/u-1/w)` are correct. Gaussian cumulants reproduce Gaussian survival probabilities in the test. Chi moments and conditional scaling by r^j are correct under noiseless homogeneity. At four saved k=12 events, increasing quadrature from 161 to 641 nodes and expanding its radius bounds changed results by at most 1.4e-8 relatively. Positive noise breaks exact homogeneity, so the preregistered radial variant remains a heuristic; these checks do not establish its accuracy for all events. Fallback counts are computed for U2 but never reported; U3 does not retain them, so the registered fallback reporting remains missing.
- **Basic cost/amortisation formulas:** amended `c_path` includes all hidden-noise draws; `c_move` matches the stated one-output latent-move convention. Analytic MC uses `4(1-p)/p` and divides amortised cost by `256*(number of levels served)`. GS uses the event's level count and the specified computed 500-particle pilot. Kprop shared work is divided by 5×256, terminal threshold work by 256; moment work is shared over all five thresholds. These checks do not remove the omitted/double-counted operations identified above.
- **Other amendment-2 numbers reproduce:** `checks/killtests.txt` gives U2/C-K3f fractions 0.65625/0.1875 at L=4 and 0.3125/0.1875 at L=8; U2r gives 0.65625 and 0.34375. U3 gives 0.8125 at L=4. U4's median relative variances are approximately 3,650, 873 and 656 for ranks 0, 3 and 10. U5's printed per-net/cell medians span approximately 0.455–1.110. Ratios above one can occur from independent noisy estimates of the numerator and denominator; they do not by themselves indicate an error in the coupled-event construction. U1's printed 0.84/0.47 also reproduces, but the latter is for `< .999`, not the registered `< 1`.
