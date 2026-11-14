codex
The main problems are substantive: Part 3 scores unaffordable budgets, H2.5 requires a cell the runner never generates, the Hermite estimators are not exactly unbiased, and the registered Gamma calculation is defective. The heavy-tail evidence supports an annealed amplitude plateau, not a reliable measured decline.

Checked against files through **2026-10-31 19:04 UTC**. All existing library tests passed; I independently recomputed scores and reran the \(n=64\) replica chain. No files changed.

**T1. Scorer audit**

**(OK) Registration integrity.** The three frozen preregistrations match their SHA256 hashes and are prefixes of the current PREREG. The registered runner hashes match, except that the original `part1c.py` is now preserved as `part1c_as_registered.py`; the current version contains the disclosed normalization fix. Part 3’s scorer itself matches its registered hash. Sources: [hashes](/home/agent/workspaces/mats/work/input-bits/prereg_hashes.txt:1), [corrected normalization](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:35).

Part 1, rule by rule:

| Rule | Finding and fix |
|---|---|
| P1.1 | **(OK)** Correct cell, medians, and three bands. [Scorer:60](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:60). |
| P1.2 | **(OK)** Computes the median of each net’s \(V_{\rm dir}/V_{\rm amp}\), with the right factor-two and strict one-sided comparisons. This assumes “\(n\ge64\)” scopes both clauses, a reasonable reading. [Scorer:65](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:65). |
| P1.3 | **(CONCERN)** Correct sequences, but interprets “monotone” as strictly monotone. Specify whether ties pass; current data have no ties. [Scorer:80](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:80). |
| P1.4 | **(OK)** Correct cells and band. Measurement is conditional on surviving inputs if any die; see T2. No sampled dead inputs occurred in the saved Part 1a runs. [Scorer:85](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:85). |
| P1.5 | **(OK)** Uses pooled net direction share and arithmetic mean net variance, exactly as registered. The statistical comparison is poorly powered, discussed in T3. [Scorer:95](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:95). |
| P1.6 | **(OK)** Correct per-net comparisons and memory-time cutoff. The noisy “shares” are an as-if-homogeneous diagnostic, not an exact noisy variance decomposition. [Scorer:111](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:111). |
| P1.7 | **(OK)** Correct \(\theta_{\ell-1}\), array index \(\ell-1\), and separate chain comparison wherever median Hamming is at least two. **Fix reporting:** emit every chain failure with numbers; currently only its count is saved. [Scorer:143](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:143). |
| P1.8 | **(OK)** Fits log median Hamming against log layer over every registered layer. [Scorer:160](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:160). |
| P1.9 | **(OK)** Median of per-net entropy/Hamming ratios; eligibility uses median Hamming. Correct band and layers. [Scorer:166](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:166). |
| P1.10 | **(OK)** Index 30 is layer 31, and every net is tested separately. [Scorer:181](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:181). |
| P1.11 | **(OK)** Correct median and band. [Scorer:195](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:195). |
| P1.12 | **(OK)** Correct ratio of the two cell medians. [Scorer:196](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:196). |
| P1.13 | **(OK)** Correct median band and per-net kernel comparison. The apparent missing \(n\) cancels the \(n\) in the registered denominator. [Scorer:199](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:199). |
| P1.14 | **(OK)** Correct per-net inequality, matched to Part 1a at the same depth. Restrict the glob-loaded nets to registered seeds. [Scorer:217](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:217). |
| P1.15 | **(OK)** Uses the default cell median. The registration does not say “every net” here. [Scorer:228](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:228). |
| P1.16 | **(CONCERN)** Uses median paired ratios and median paired excesses. Those differ from comparing the separately computed medians of the named shares. The prose does not uniquely settle this. **Fix:** report both interpretations and explicitly record the convention; do not call it unambiguously “exactly registered.” Neither interpretation rescues the present off-diagonal failures. [Scorer:242](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:242). |
| P1.17 | **(DEVIATION)** Tests only powers of two, whereas the prose specifies every layer \(2\le\ell\le n/4\). Ten points are tested instead of 78. The frozen runner also records only powers of two, so this is an internal protocol inconsistency. **Fix:** label the existing result “sampled layers”; measure the missing layers for a separately identified full check. [Scorer:263](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:263), [registration:48](/home/agent/workspaces/mats/work/input-bits/PREREG.md:48). |
| P1.18 | **(DEVIATION)** Same sparse-layer problem: 10 tangent-PR checks and 14 finite-versus-tangent checks, rather than all specified layers. The inequalities themselves are correct. Same fix. [Scorer:268](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:268). |
| P1.19 | **(DEVIATION)** Fits the slope of layerwise median ratios; the natural reading of “slope … (median over nets)” is the median of per-net slopes. P1.8 explicitly specifies the former construction, P1.19 does not. Also uses sparse layers and the defective original Gamma measurements. **Fix:** report both aggregations, preserve the original score, and score corrected measurements separately. [Scorer:276](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:276). |
| P1.20 | **(OK)** Correct depths and per-net radial-share tests. [Scorer:274](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:274). |

**(BUG) Missing data can silently become a pass.** Numerous loops skip absent runs, and `all([])` is true. This affects both Part 1 and Part 2; Part 3 also lacks a required-run manifest. Some Part 1 rules ingest extra glob-matched seeds while others use the allowlist. **Fix:** validate the complete registered Cartesian product first; distinguish `PENDING`, `INVALID`, `HIT`, and `MISS`. Require finite numbers and report actual denominators. Sources: [Part 1 loading](/home/agent/workspaces/mats/work/input-bits/scratch/score_part1.py:13), [Part 2 skipping](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:16).

**(OK) Independent recomputations from the run JSONs:**

| Score | My recomputation | Saved result |
|---|---:|---|
| P1.1, \(256\times32\) | Median \((s_{\rm amp},s_{\rm dir},s_{\rm rad})=(0.494003,\ 0.469804,\ 0.034234)\) | HIT, matches |
| P1.8 | Slopes \(-0.780292\) at \(n=128\), \(-0.776266\) at \(n=256\) | HIT, matches |
| P1.12 | \(14.586554/4.846646=3.009618\) | HIT, matches |

For P1.19, median per-net slopes are **1.407943** and **1.621486**, versus the scorer’s **1.435084** and **1.637405**. Both slope conventions pass; the Gamma band causes the saved miss.

The saved scorecard mechanically reports **12 hits, eight misses**. Because of the sparse-layer and Gamma issues, that is not twelve fully validated claims.

Part 2:

| Rule | Finding and fix |
|---|---|
| H2.1 | **(OK)** Correct explicit comparison of median gain against \(0.75/(1-\operatorname{median}S_1)\), and correct \(N\) values. Subject to completeness checks. [Scorer:60](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:60). |
| H2.2 | **(DEVIATION)** Builds \(G_{\rm pred}\) from separately medianed \(S_1,W_{2,\rm off}\), then divides median \(G\) by it. This generally differs from the median per-net \(G/G_{\rm pred}\) specified by the default aggregation rule. It also changes the lower bound from **0.67** to **\(1/1.5=0.666667\)**. **Fix:** join measurements by seed, calculate each ratio first, then take its median; use 0.67 literally. [Scorer:48](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:48), [band:67](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:67). |
| H2.3 | **(OK)** Median \(S_1\), median parity, and median per-net excess factor \(E=G(1-S_1)\) are implemented correctly. Missing widths can nevertheless be skipped or trigger a lookup failure. [Scorer:71](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:71). |
| H2.4 | **(OK)** Correct per-net antithetic consistency check on seeds 1000/1001 and median odd residual share by width. Its interpretation concerns Hermite parity, not ANOVA order. [Scorer:75](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:75). |
| H2.5 | **(BUG)** The required \(n=256,L=32\) cell is never generated or read: both runner and scorer require \(L\ge n/4\). Thus `best_L32` necessarily contains `None`, making the combined rule fail even with every scheduled run complete. Also, “does not increase over the widths” is reduced to an endpoint comparison, allowing an intermediate increase. **Fix:** include \(L=32\) unconditionally and test both adjacent differences. The ratio-of-medians convention in clause (a) should also be declared explicitly. [Runner:23](/home/agent/workspaces/mats/work/input-bits/scratch/part2_mlmc.py:23), [scorer:86](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:86), [scorer:102](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:102). |
| H2.6 | **(DEVIATION)** The noise support check includes H2.1 at \(n=8\), although this verdict concerns \(16\to128\). Input residual trends use \(N=1024\) only; the registration does not specify that restriction for \(G/G_{\rm pred}\). Strict sign monotonicity has no uncertainty tolerance. **Fix:** scope the floor to the named widths and declare the \(N\)-aggregation. The verdict’s deeper logical problem is in T4. [Scorer:107](/home/agent/workspaces/mats/work/input-bits/scratch/score_part2.py:107). |

Part 3:

**(BUG) None of P3.1–P3.6 is actually scored.** The script emits per-seed statistics and a table, then exits. It never evaluates the registered inequalities or verdict.

| Rule | Missing implementation |
|---|---|
| P3.1 | Per-cell median exponent bands, including the narrower \(n=256,L\ge128\) band |
| P3.2 | Width comparison for RQMC-RB at \(L=32\) |
| P3.3 | Reading and testing `cvq.ratio` |
| P3.4 | Counting cells where the rotation gain ratio reaches 1.2 |
| P3.5 | ML2 width trend and ML2-RQMC dominance count |
| P3.6 | The PROGRESS/constants-only verdict |

Source: [entire scoring loop and output](/home/agent/workspaces/mats/work/input-bits/scratch/score_part3.py:41). **Fix:** implement these six rules with the registered seed and cell manifest.

Three additional Part 3 scorer faults matter more than the missing booleans:

- **(BUG) Unaffordable charged gains.** The target is \(4096L\) mv, but the pilot alone costs \(8192L\) or \(16384L\). Nevertheless, `interp_gain` extrapolates below the first charged point. There is no estimator at that budget. **Fix:** return `UNAFFORDABLE` when setup consumes the budget; for affordable MC-type methods use \(K/(C-C_{\rm setup})\), not log-linear extrapolation through an additive overhead. Sources: [registered pilot sizes](/home/agent/workspaces/mats/work/input-bits/PREREG.md:92), [extrapolation](/home/agent/workspaces/mats/work/input-bits/scratch/score_part3.py:26).
- **(BUG) ML2-RQMC has no charged-pilot curve.** Its `cost` includes the reference pass but omits the pilot that selected `rbest`; no `cost_pilot` or `cost_charged` is saved. **Fix:** save both accounting variants consistently. [Runner:159](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:159).
- **(BUG) The “envelope” is not an envelope at fixed compute.** It minimizes `nmse * cost` separately at each \(N_2\). That criterion is justified for common exponent one, not general RQMC curves. Allocation changes can also reorder costs, so the last two cost-sorted points need not represent the registered \(N_2=1024\to4096\) step. **Fix:** interpolate each allocation’s curve separately, then minimize error at common costs; identify endpoint measurements explicitly. [Envelope](/home/agent/workspaces/mats/work/input-bits/scratch/score_part3.py:17), [last-step slope](/home/agent/workspaces/mats/work/input-bits/scratch/score_part3.py:54).

**T2. Estimator audit**

**(OK) `var_split` estimates its three variance numerators without bias.** For iid sphere directions, the U-statistics for \((EA)^2\) and \(\|Eg\|^2\) are correct, and the three estimated terms sum algebraically to the estimated total. The returned *ratios* are consistent, not exactly unbiased. Source: [ib.py:81](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:81). **Fix:** distinguish unbiased variance components from ratio estimates in documentation.

**(BUG) The Hermite “two independent halves” claim is false with a shared fitted centre.** `univariate_shares`, `deg2_share`, and `stein_shares` subtract the mean of the entire sample before splitting. Their halves are therefore dependent.

For the scalar counterexample \(F=X\), \(P=2h\), the expected raw degree-one numerator is
\[
E[\widehat g_1\widehat g_2]
=1-\frac1h+\frac{1}{2h^2},
\]
rather than one. At \(P=16\), I obtained \(0.88362\pm0.00138\), matching the exact biased value \(0.8828125\). The returned variance normalization introduces additional ratio bias.

**Fix:** estimate the coefficient in each half using its own unbiased sample covariance,
\[
\widehat c_k=
\frac{1}{h-1}\sum_{i\in{\rm half}}
(F_i-\bar F_{\rm half})H_k(X_i),
\]
then cross-multiply the independent coefficient estimates. The variance numerator is then unbiased; its normalized share remains a ratio estimate. Sources: [univariate](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:203), [quadratic](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:220), [Stein](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:138).

The bias is small at \(P=16384\); it does not automatically explain the registered misses. The exact-unbiasedness claim still needs correction.

**(CONCERN) Output subsampling changes the target.** At \(n=256\), `deg2_share` divides the quadratic energy of 64 outputs by those same outputs’ variance. As input sample size tends to infinity, this converges to the subset’s share, not the full vector-output share. Randomly choosing outputs does not make a ratio of subset totals unbiased.

**Fix:** estimate the full numerator by \(n/J\) times the subset numerator and divide by the full-output variance, or evaluate all outputs. Retain output-sampling uncertainty. Sources: [subset choice](/home/agent/workspaces/mats/work/input-bits/scratch/part1b.py:39), [subset normalization](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:226).

**(OK) Jansen’s numerator and coordinate scaling are correct.** Replacing one coordinate and taking half the squared output difference estimates that coordinate’s total effect. Uniform coordinate subsampling with multiplier \(n/m\) is valid sampling estimation of the total.

**(CONCERN)** With a fixed sampled coordinate set, increasing `PJ` alone does not converge to the full sum. `D_se` measures coordinate dispersion but omits shared-input Monte Carlo uncertainty and the finite-population correction. **Fix:** use independent input batches and coordinate-subset replicates, or all coordinates; report both uncertainty sources. [part1b.py:83](/home/agent/workspaces/mats/work/input-bits/scratch/part1b.py:83).

**(OK) The fitted rotation is independent of its scoring sample.** `Xp` and `X` have separate seeds, and \(Y=X\,\mathrm{Vec}\) is the correct row-vector transformation. Reusing the evaluation outputs \(F(X)\) is legitimate. Conditional on the pilot, \(Y\) is standard Gaussian. Across saved runs I checked:
\[
\max|\widehat W_1-\widehat W_{1,\rm rot}|<5.0\times10^{-16},\qquad
\max|\widehat W_2-\widehat W_{2,\rm rot}|<6.2\times10^{-16}.
\]
These are useful invariance checks. Source: [part1b.py:66](/home/agent/workspaces/mats/work/input-bits/scratch/part1b.py:66).

**(CONCERN)** Diagonalizing the Hessian of \(\|a\|^2\) need not diagonalize every output’s quadratic component. Pilot error also grows with dimension. Thus P1.16 can fail without disproving the existence of useful input coordinates. **Fix:** measure held-out captured quadratic energy versus pilot size, and separate proxy mismatch from estimation error.

**(BUG) Registered Gamma normalization.** The original call to `ib.sphere` on an \(R\times M_{\rm ref}\times n\) array normalizes axis 1, not the coordinate axis. Before tangent projection, the mean squared perturbation norm is \(n/M_{\rm ref}\), which is four at \(n=256\).

The current axis-2 fix is correct. Dividing old results by four corrects the ensemble scale, but is not an exact replacement for rerunning the same realized perturbations. Source: [original:36](/home/agent/workspaces/mats/work/input-bits/scratch/part1c_as_registered.py:36), [fix:37](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:37).

Further Gamma details:

- **(OK)** Repeated tangent rescaling is restored by `exp(2*logS)` and `exp(2*logSr)`; no accumulated scale is lost.
- **(CONCERN)** Projecting a random unit vector onto the input tangent plane leaves expected squared norm \((n-1)/n\). “Per unit input angle” therefore needs a factor \(n/(n-1)\), or tangent vectors sampled with unit norm.
- **(CONCERN)** `Gamma_mean` averages multiple references, while its finite-fraction denominator uses one different reference. It is not an average of matched sensitivity ratios.
- **(PROVEN)** More seriously, the unregularized population mean angular sensitivity need not be finite. See T6.

Sources: [tangent projection](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:32), [Gamma computation](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:86).

**(CONCERN) Part 1a’s noisy split and log-amplitude statistic have narrower meanings than their names suggest.** With additive noise, \(a(X)/|X|\) depends on the input radius, so passing it through the noiseless identity is not a consistent estimator of an exact noisy radius/amplitude/direction decomposition. The runner acknowledges this. The coupled difference variance is valid. `logA_var` excludes dead inputs and consequently estimates a survival-conditioned quantity.

**Fix:** retain these as explicitly named diagnostics; use conditional variance decomposition if an exact noisy split is needed. Define the conditioning in the registration. Sources: [noisy split](/home/agent/workspaces/mats/work/input-bits/scratch/part1a.py:30), [log amplitude](/home/agent/workspaces/mats/work/input-bits/scratch/part1a.py:55).

**(OK) The replica transition is exact for the annealed model.** The conditional bivariate Gaussian construction and the split-term formulas are correct. `chainB` correctly estimates a one-coordinate total effect. Ratios such as `D_ann` are consistent ratios, not unbiased quantities. Sources: [transition](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:168), [terms](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:261), [chain runner](/home/agent/workspaces/mats/work/input-bits/scratch/part1chain.py:44).

**(CONCERN)** The separately sampled total-variance integrand does not equal the sum of the three split integrands realization by realization. Consequently the reported shares need not sum to one; at \(n=64,L=256\), `s_amp` is **1.048**.

**Fix:** use the sum of the three estimated components as the denominator for displayed decomposition shares, and retain the independent total as a consistency diagnostic with a paired difference SE. “Exact chain” should describe the transition law, not the numerical precision.

**(OK) Noise regeneration and window indexing match the stationary lane.** Both use layers \(-B+1,\ldots,0\), output layer 1, \(B=8n\), age \(=-\ell\), and the most recent
\[
d_{\rm win}=\min(B,\lfloor21201/n\rfloor)
\]
layers. The regenerated noise has exactly the same seeds and shapes as during the forward pass. Negating the entire history matches the antithetic estimand. Sources: [noise runner](/home/agent/workspaces/mats/work/input-bits/scratch/part2_noise.py:24), [stationary coordinates](/home/agent/workspaces/mats/work/stationary/scratch/partB.py:185).

**(BUG) Noise-share centring has a smaller, different bias.** Here each half has its own mean, so the halves are independent, but division by \(h\) shrinks each coefficient by \((h-1)/h\). Their product shrinks by its square. **Fix:** divide the centered coefficient sums by \(h-1\). Also enforce an even number of 1024-row batches: arbitrary `--P` values are silently truncated, and odd batch counts break the half-size bookkeeping. [part2_noise.py:28](/home/agent/workspaces/mats/work/input-bits/scratch/part2_noise.py:28), [centres and accumulation](/home/agent/workspaces/mats/work/input-bits/scratch/part2_noise.py:51).

**(BUG) `G_res_true` is not the true-projection residual gain.** Let the true degree-\(\le4\) additive projection be \(p\), its fitted version \(p+e\), and \(r=f-Ef-p\). Conditional on the pilot,
\[
\operatorname{Var}_{MC}(f-\widehat f_1)
=\operatorname{Var}(r)+\|e\|_{L^2}^2.
\]
Replacing the numerator by \(V(1-S_1)\) removes coefficient-error energy from the numerator. The denominator still measures the RQMC variance of \(r-e\):
\[
\operatorname{Var}(Q(r-e))
=\operatorname{Var}(Qr)+\operatorname{Var}(Qe)-2\operatorname{Cov}(Qr,Qe).
\]
RQMC does not integrate these fitted Gaussian Hermite terms exactly.

**Fix:** call this a noise-adjusted-numerator diagnostic; retain the valid gain for the fitted residual. To estimate the true-projection gain, increase pilot size and demonstrate convergence, or use independent coefficient fits to estimate the remaining contamination. Sources: [pilot](/home/agent/workspaces/mats/work/input-bits/scratch/part2_input.py:43), [correction](/home/agent/workspaces/mats/work/input-bits/scratch/part2_input.py:69).

**(OK) Multilevel definitions match the unrestricted implementation.** The prefix control, suffix scale, constant \(G_0\), and final correction agree. Sharing pilot directions across candidate depths is valid; production levels must be independent, as the RQMC implementation makes them. Sources: [Part 2 multilevel](/home/agent/workspaces/mats/work/input-bits/scratch/part2_mlmc.py:38), [Part 3 pilot](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:109), [reference definitions](/home/agent/workspaces/mats/work/unrestricted/lib/mlmc.py:94).

**(CONCERN)** ML2 and MLD curves are plug-in ideal-risk predictions, not measured estimator MSEs. ML2 selects the minimum estimated risk and reports that same minimum, introducing selection optimism. Integer allocation and its actual risk are absent.

**Fix:** select the allocation on the pilot, evaluate its level variances on held-out directions, and use actual integer sample counts. Keep the current curves labeled “pilot-estimated ideal allocation.”

**(PROVEN) CVQ has exactly zero mean under the continuous Gaussian target, conditional on its pilot.**
\[
E[XG^\top]=0,\qquad
E\!\left[\tfrac12\sum_i\lambda_i((v_i^\top X)^2-1)\right]=0.
\]
The pilot is independent of both MC validation and production scrambles. Estimating \(G,H,q\) imperfectly changes efficiency, not unbiasedness. Source: [CVQ construction](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:65).

**(CONCERN) Its quadratic fitting noise is large enough to erase the gain.** I applied the same centered quadratic-moment fit and top-32 truncation to the *purely linear*, unit-variance function \(f(X)=X_1\). Its true quadratic component is zero. Across four independent pilots, the median variance of the spurious fitted quadratic was:

| \(n\) | Pilot size | Added quadratic variance |
|---:|---:|---:|
| 64 | 8192 | 0.240 |
| 128 | 8192 | 0.669 |
| 256 | 16384 | 0.802 |

The variance was calculated exactly from the fitted eigenvalues as \(\frac12\sum_{i\le32}\lambda_i^2\), not estimated on another finite sample. This is a calibration counterexample, not a network result.

**Fix:** estimate the quadratic after removing a cross-fitted linear component, shrink eigenvalues, and select rank on held-out residual variance. Do not interpret failure of this pilot as failure of quadratic controls in principle.

**T3. Theory follow-up**

**(PROVEN) A falling total variance does not itself contradict monotone amplitude variance.** Since
\[
\bar V_L=v_n+E_WV_{{\rm amp},L}+E_WV_{{\rm dir},L},
\]
the total can decrease as direction variance disappears. However, monotonicity implies, for \(L_2>L_1\),
\[
\bar V_{L_1}-\bar V_{L_2}\le E_WV_{{\rm dir},L_1}.
\]
Thus a large deep decline exceeding the entire remaining direction term cannot be explained by direction loss. Source: [FORMAL:38](/home/agent/workspaces/mats/work/input-bits/FORMAL.md:38). **Fix:** compare paired depth differences with this inequality, not with total-variance monotonicity.

**(HEURISTIC) The recorded chain is severely tail-limited at \(L\sim n\).** Repeating the original \(n=64,P=20000\) chain reproduced its saved means and exposed the following amplitude-term distribution:

| \(L\) | Mean | Median | 99th percentile | Top 1% contribution | Top 0.1% contribution |
|---:|---:|---:|---:|---:|---:|
| 16 | 6.016 | 1.594 | 63.52 | 17.8% | 4.1% |
| 32 | 6.667 | 0.935 | 94.72 | 30.1% | 8.7% |
| 64 | 7.450 | 0.261 | 114.19 | 55.4% | 26.9% |
| 128 | 5.390 | 0.0194 | 67.37 | 80.0% | 44.2% |
| 256 | 1.202 | 0.000113 | 7.80 | 93.5% | 65.6% |

At \(L=64\), the empirical second and third raw moments of the amplitude term were approximately **8,949** and **\(5.32\times10^7\)**. At \(L=256\), one replica contributed **23.1%** of the entire sample sum. Source and reproduction path: [part1chain.py:39](/home/agent/workspaces/mats/work/input-bits/scratch/part1chain.py:39), invoked without `--out`.

The other saved chain summaries imply:

| \(n,L\) | Mean amplitude term | Empirical CV | Contribution ESS \(=(\sum T)^2/\sum T^2\), out of 20,000 |
|---|---:|---:|---:|
| 64, 64 | 7.450 | 12.66 | 124 |
| 128, 128 | 7.377 | 7.95 | 312 |
| 256, 256 | 8.518 | 12.28 | 132 |
| 256, 512 | 7.442 | 31.28 | 20 |

These ESS values describe observed concentration; they do not account for unseen extremes.

The four-net amplitude samples at \(L=n\) are also insufficient to estimate an annealed mean:

| \(n\) | Per-net \(V_{\rm amp}\), seeds 4000–4003 | Median | Mean |
|---:|---|---:|---:|
| 64 | 0.321, 2.391, 0.589, 0.451 | 0.520 | 0.938 |
| 128 | 2.843, 0.507, 118.992, 0.0696 | 1.675 | 30.603 |
| 256 | 0.579, 0.772, 0.0968, 1.780 | 0.676 | 0.807 |

At \(n=128\), one net supplies **97.2%** of the amplitude sum.

**(PROVEN) Multiplicative norm moments explain the instability.** For a single unit-input trajectory, its squared norm is a product of iid gains
\[
Q_L=\prod_{\ell=1}^L M_\ell,\qquad
M=\frac2n\sum_{i=1}^n Z_{i,+}^2.
\]
Exactly,
\[
EM=1,\quad EM^2=1+\frac5n,\quad
EM^3=1+\frac{15}{n}+\frac{44}{n^2}.
\]
Therefore
\[
EQ_L^2=(1+5/n)^L,\qquad
EQ_L^3=(1+15/n+44/n^2)^L.
\]
At \(L=\alpha n\), these approach \(e^{5\alpha}\) and \(e^{15\alpha}\). At \(n=256,L=256\), they are approximately **141.4** and **\(2.52\times10^6\)**.

Once trajectories largely align, their amplitude difference inherits much of this common multiplicative variability. That last connection is **(HEURISTIC)**; the norm-moment identities are exact.

**Fix for reporting:** preserve registered medians for typical-net statements. Use arithmetic means of variance terms and pooled ratios \(\sum V_j/\sum V\) for annealed statements, with adequate independent weight sampling. Report both, clearly distinguished. Do not substitute a median for an unstable annealed mean and retain the annealed interpretation. Save replica contributions, paired depth-difference SEs, and tail-contribution diagnostics; consider conditional-increment/Rao–Blackwell estimators for the amplitude plateau.

**(CONCERN) Substantive Part 1 misses have different causes.**

- **P1.2:** the sole deep failure is \(n=256,L=512\): median direction/amplitude ratio **0.008808**, prediction **0.003924**, factor **2.245**. This is a typical-net ratio; an annealed directional upper bound does not guarantee it. **Fix:** retain the miss and distinguish the ensemble statements.
- **P1.5:** six of twelve cells fail. Four nets cannot reliably approximate the annealed mean in this distribution; for \(256\times256\), chain variance divided by mean net variance is **10.69**. This challenges the sampling design, not the exact replica transition. **Fix:** more independent nets and tail diagnostics.
- **P1.6:** 15 of 72 net/depth checks fail. At \(n=32,L=64,\) seed 4002, coupled noise variance is **1.861 times** noiseless variance, despite lying below the nominal cutoff. **Fix:** state that a typical-radius memory time does not give a uniform per-net guarantee; report realized signal/noise diagnostics.
- **P1.7:** kernel failures are **3/116**, chain-versus-median failures **131/472**. An annealed mean is not generally within 25% of a four-net median, and layerwise quenched fluctuations remain substantial. **Fix:** separate approximation error from mean-versus-median mismatch.
- **P1.13:** the two failures are seed 4002 at \(n=32,L=8,16\). \(W_1\) is **0.2919 versus \(2\,GP=0.4512\)**, and **0.2048 versus 0.2260**. The broad finite-width linear component survives; the every-net multiplier was too strong.
- **P1.16:** off-diagonal ratios are about **0.62** at \(n=128\), **0.87–0.89** at \(n=256\), against 0.6. Pilot estimation and the scalar proxy are plausible causes. **Fix:** pilot-size and proxy diagnostics before drawing a negative conclusion about rotations.
- **P1.19:** the original \(n=256\) Gamma scale is wrong. Dividing by four would still put its final tested median near **3.040**, barely outside the band, but a corrected rerun is required. Do not turn an invalid measurement into a theoretical refutation.
- **P1.20:** two \(n=64,L=64\) nets have radial shares **0.8847** and **0.8950**. The “every net \(\ge0.9\)” assertion misses; approximate radial collapse remains supported.

Sources: [scorecard](/home/agent/workspaces/mats/work/input-bits/results/part1_scores.json:1), with the scorer locations listed in T1.

**(PROVEN, fixed-depth expansion) There is a computable first finite-width correction to the direction variance.** Write the unit-input Gram state as \((q_1,q_2,r)\), with deterministic map
\[
F(q_1,q_2,r)=
\left(q_1,q_2,\sqrt{q_1q_2}\,
\kappa\!\left(\frac r{\sqrt{q_1q_2}}\right)\right).
\]
A second-order delta-method recursion propagates its \(1/n\) mean correction \(b_\ell\) and covariance coefficient \(\Sigma_\ell\):
\[
\Sigma_{\ell+1}=J_\ell\Sigma_\ell J_\ell^\top+Q_\ell,\qquad
b_{\ell+1}=J_\ell b_\ell+\tfrac12D^2F_\ell:\Sigma_\ell.
\]
Here \(Q_\ell\) is the covariance of
\((2Z_+^2,2Z'_+{}^2,2Z_+Z'_+)\) at correlation \(c_\ell\);
\(\Sigma_0=\operatorname{diag}(0,0,1)\), \(b_0=0\).
If \(\beta_L\) is the third coordinate of \(b_L\), then
\[
E V_{\rm dir}
=n\delta_L-C_L-\beta_L-\tfrac12\delta_L+O(n^{-1}).
\]
This follows from the same exact transition used in [ib.py:168](/home/agent/workspaces/mats/work/input-bits/lib/ib.py:168).

My computed additive corrections at \(L=32,64,128\) are **−1.84865, −1.12737, −0.62532**. Their leading large-\(L\) relative behavior is approximately \(-2L/n\).

**(CONCERN)** This expansion is not uniform at \(L\sim n/2\); its linear truncation can become negative. It is not a valid closed-form answer throughout the requested range.

**(HEURISTIC) The exact-chain measurements give a useful finite-width correction there:**

| \(n\) | \(E V_{\rm dir}/(n\delta_L)\), \(L=n/4\) | Same, \(L=n/2\) |
|---:|---:|---:|
| 64 | 0.558 | 0.318 |
| 128 | 0.559 | 0.324 |
| 256 | 0.571 | 0.333 |

For the *direction share*, relative to \(n\delta_L/(v_n+C_L+n\delta_L)\), the corresponding factors are approximately **0.755/0.445**, **0.674/0.397**, and **0.679/0.386**. **Fix:** use the finite-width chain with uncertainty in this regime; do not extrapolate the fixed-depth correction as a theorem.

**T4. Synthesis design red team**

**(OK) The calibrated formula is a fair test of a specific surrogate model.** Calibration uses no network data and the same Gaussian-coordinate Sobol construction. Passing would show that this particular low-order approximation predicts gain reasonably well.

**(CONCERN) It is not a clean general test of “gain is set by low-order share.”** Failure can arise from:

- Resolution and coordinate-pair dependence of Sobol projection quality.
- Nonexchangeable quadratic energy in a particular fixed net.
- Higher-degree two-coordinate terms: \(W_{2,\rm off}\) measures total Hermite degree two, not all ANOVA order-two energy.
- Univariate degrees above four omitted from the registered \(S_1\).
- Reduction of higher-order components, despite the model setting their efficiency to one. The calibration itself reports substantial reduction of some cubic functions.
- Pilot and share-estimation noise, including the output-subset target mismatch.
- The noise regime’s unstructured old-history contribution.

Sources: [model](/home/agent/workspaces/mats/work/input-bits/PREREG.md:66), [calibration](/home/agent/workspaces/mats/work/input-bits/scratch/calib_rho2.py:14).

**Fix:** describe H2.2 as a test of the stated surrogate, report residual discrepancies by these mechanisms, and use the exact quadratic MC variance \(2\|M\|_F^2\) in calibration instead of estimating it on 16,384 samples.

**(CONCERN) The noise experiment changes both dimension and the structured window.** Its Sobol dimensions are:

| \(n\) | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|
| Structured coordinates | 512 | 2048 | 8192 | 21184 | 21120 |

After the cap binds, increasing width shortens the window in layer units. For \(f(U,Z)\), with recent noise \(U\) structured and old noise \(Z\) sampled independently,
\[
N\operatorname{Var}(\widehat\mu)
=N\operatorname{Var}_{RQMC}(Qg)
+E_U\operatorname{Var}_Z(f\mid U),\qquad g=E[f\mid U].
\]
**Fix:** measure the second term’s variance share \(\eta\); otherwise shrinking gain can be attributed to interaction order when it is caused by the hybrid MC floor. Source: [FORMAL:30](/home/agent/workspaces/mats/work/input-bits/FORMAL.md:30).

**(BUG) H2.6 is not logically sound as a verdict on growing interaction order.**

1. Fixed \(S_1\) does not imply fixed higher-order structure. Moving variance from order two to order ten while leaving \(S_1=0.3\) unchanged can greatly reduce gain; the rule would say REFUTED.
2. A fall in degree-\(\le4\) additive share can be a shift to higher *univariate degree*, with no increase in ANOVA interaction order.
3. SUPPORTED does not explicitly require gain to fall.
4. “No monotone trend” in three or four noisy estimates is not evidence of no trend.

The present input data already illustrate the first distinction: along \(L=n/2\), median \(D\) rises from **4.85 to 14.59**, while the truncated \(S_1\) changes from **0.303 to 0.320**. The completed noise-share files give medians **0.658, 0.592, 0.648, 0.659, 0.622** across widths 8–128; they show no simple decline.

**Fix:** retain H2.6 as a registered operational classification, but call its scientific target “declining measured additive share,” not “growing interaction order.” Sources: [verdict](/home/agent/workspaces/mats/work/input-bits/PREREG.md:86), [dimension and shares](/home/agent/workspaces/mats/work/input-bits/results/part1_tables.md:36).

**(PROVEN design; HEURISTIC runtime) One additional cheap measurement: Bernoulli coordinate-sharing noise stability.**

For each noise-regime net, generate two histories. In the recent window, independently retain each scalar noise coordinate from history A with probability \(t\), otherwise take it from history B. Always use independent old histories. Measure
\[
C(t)=\frac{\operatorname{trCov}(f(A),f(A^{[t]}))}{V}
=\sum_{\varnothing\ne S\subseteq{\rm window}}t^{|S|}\frac{V_S}{V}.
\]
Thus \(C(1)=1-\eta\), and \(C(t)/C(1)\) is the probability-generating function of ANOVA order in the conditional window integrand. Adding more coordinates at a fixed order does not change this normalized curve.

Use \(t=0.25,0.5,0.75,1\), widths 16, 64, 128, four seeds, and 4096 paired paths. Reuse the A trajectory across \(t\). Estimate covariance through
\[
\operatorname{trCov}(A,B)=V-\tfrac12E\|A-B\|^2
\]
and give batch-based uncertainty.

A downward shift at fixed \(t<1\) directly indicates a shift toward higher interaction order. Stable curves with shrinking RQMC gain implicate sampling efficiency or the measured hybrid floor. Based on the saved \(n=128\) forward timing, this should take roughly **5–10 CPU minutes**, comfortably below 30. Implementation starting point: [noise forward path](/home/agent/workspaces/mats/work/input-bits/scratch/part2_noise.py:37).

**T5. Part 3 predictions**

**(PROVEN) With the registered pilot charged, five methods are unaffordable at \(C_*=4096L\): RQMC-ROT, CVQ, RQMC-CVQ, ML2, and MLD. ML2-RQMC is also unaffordable when its selection pilot is correctly charged.** Their charged gain is **undefined**, not an extrapolated number. Sources: [pilot sizes](/home/agent/workspaces/mats/work/input-bits/PREREG.md:92), [costs](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:64). **Fix:** reduce pilot size under a disclosed amended experiment, or evaluate larger total budgets.

**(HEURISTIC) The following are forecasts with the pilot/setup amortized away.** They incorporate the actual finite-pilot fitting risk, rather than assuming an oracle quadratic control. Entries are gain triples for **\(n=(64,128,256)\)**. Treat most values as uncertain by roughly a factor 1.5; reference-dependent multilevel gains may vary more. These are predictions, not Part 3 observations.

| Method | \(L=32\) | \(L=n/2\) | \(L=n\) | \(L=2n\) |
|---|---|---|---|---|
| MC | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 | 1, 1, 1 |
| RBMC | 1.08, 1.05, 1.04 | 1.08, 1.06, 1.05 | 1.08, 1.06, 1.05 | 1.08, 1.06, 1.05 |
| RQMC | 2.6, 2.3, 2.0 | 2.6, 2.4, 2.2 | 2.5, 2.4, 2.2 | 2.5, 2.4, 2.2 |
| RQMC-RB | 2.6, 2.3, 2.0 | 2.6, 2.4, 2.2 | 2.5, 2.4, 2.2 | 2.5, 2.4, 2.2 |
| RQMC-ROT | 2.7, 2.3, 2.0 | 2.7, 2.4, 2.2 | 2.6, 2.4, 2.2 | 2.6, 2.4, 2.2 |
| CVQ | 1.3, 1.0, 0.9 | 1.3, 0.9, 0.8 | 1.3, 0.9, 0.7 | 1.3, 0.9, 0.7 |
| RQMC-CVQ | 2.5, 2.0, 1.7 | 2.5, 2.0, 1.7 | 2.4, 2.0, 1.6 | 2.4, 2.0, 1.6 |
| ML2 | 1.1, 0.9, 0.7 | 1.1, 1.2, 1.4 | 1.3, 1.5, 1.8 | 1.6, 1.9, 2.3 |
| MLD | 0.8, 0.7, 0.6 | 0.8, 0.9, 1.1 | 1.0, 1.1, 1.4 | 1.3, 1.5, 1.8 |
| ML2-RQMC | 2.2, 1.8, 1.4 | 2.2, 2.3, 2.5 | 2.8, 3.0, 3.4 | 3.3, 3.8, 4.5 |

The duplicate \(n=64,L=32=n/2\) entries agree intentionally.

**(PROVEN) MC, RBMC, CVQ, ML2, and MLD have exponent one for fixed fitted objects and allocation proportions, excluding additive setup.** The latter two curves in this runner have exponent one by construction; measuring that exponent does not validate their risk constants. Source: [analytic curves](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:134).

**(HEURISTIC) My fitted-exponent forecasts over 64–4096 points are below.** They apply approximately at all four depths for each width, with uncertainty around ±0.15 before finite-replicate measurement noise:

| Method | \(n=64\) | \(n=128\) | \(n=256\) |
|---|---:|---:|---:|
| RQMC | 1.15 | 1.12 | 1.10 |
| RQMC-RB | 1.15 | 1.12 | 1.10 |
| RQMC-ROT | 1.10 | 1.10 | 1.10 |
| RQMC-CVQ | 1.18 | 1.23 | 1.28 |
| ML2-RQMC | 1.20 | 1.18 | 1.15 |

The relatively high CVQ-RQMC exponent can reflect rapid removal of *pilot-induced quadratic noise* as \(N\) grows, even while its absolute gain is worse. I expect the plain RQMC methods’ last-step exponents to be closer to one than their whole-range fits, with substantial sampling noise.

**(HEURISTIC) Rotation should offer little extra gain at \(N=4096\) through the quadratic mechanism alone.** At this \(N\), the calibration already removes most off-diagonal quadratic variance. Even a perfect rotation only additionally removes approximately \(\rho_2W_{2,\rm off}\), not the whole \(W_{2,\rm off}\). A 20% improvement would require useful higher-order rearrangement or unusually favorable projections. Sources: [calibration](/home/agent/workspaces/mats/work/input-bits/scratch/calib_rho2.py:25), [rotation](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:98). **Fix/test:** measure the quadratic and higher-order contributions to any rotation improvement separately.

**(PROVEN under the stated ensemble residual and cost bounds) Multilevel methods are the plausible source of width-growing gain.** At \(L=\alpha n\), the bound
\[
E\|Y-G_r\|^2=O(r^{-1}+n/L^2)
\]
gives an ideal two-level gain of order \(\sqrt n\) using \(r\asymp\sqrt n\), and a dyadic gain of order \(n/\log^2n\). ML2-RQMC can inherit a growing gain if its level sampling efficiencies remain controlled. These are ensemble guarantees under the relevant assumptions, not a guarantee for two selected nets.

Fixed-degree controls and rotations retain a nonvanishing higher-degree residual, so I do not expect their gains to grow without bound. At fixed \(L=32\), there is no growing depth saving for multilevel methods. Source: [multilevel derivation](/home/agent/workspaces/mats/work/unrestricted/coprover/astra_r3_answer.md:16). **Fix:** distinguish asymptotic ensemble scaling from the finite-width, pilot-charged experiment.

**T6. Other findings that change a conclusion**

**(PROVEN) The population angular-sensitivity mean can be infinite.** This affects the interpretation of Gamma more fundamentally than the normalization bug.

At the first layer, condition on \(K\) active neurons. For a unit reference and an isotropic unit tangent perturbation, averaging over the tangent Gaussian weights gives angular amplification proportional to
\[
\frac{K-1}{S},\qquad S\sim\chi^2_K.
\]
For \(K=2\), \(E(1/S)=\infty\), and
\[
P(K=2)=\binom n2\,2^{-n}>0.
\]
Thus the annealed mean is infinite at every finite width \(n\ge2\).

For a fixed invertible first-layer matrix and \(n\ge3\), the same issue occurs near inputs with two small positive preactivations and all remaining preactivations negative: angular sensitivity behaves like
\[
\frac{\text{positive constant}}{z_1^2+z_2^2}.
\]
Integrating over the two transverse coordinates diverges logarithmically. The event is extremely rare at these widths, but rarity does not restore integrability.

**Fix:** specify a truncated sensitivity with an output-norm threshold, a median/log statistic, or an amplitude-weighted ratio of expectations. A 16-reference mean is a finite pilot statistic; it cannot substantiate a bounded population mean. Source of the measured quantity: [Gamma denominator](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:88).

**(CONCERN) P3.6 can declare progress from variance-estimation noise.** With \(R=10\) and nearly scalar output fluctuations, an estimated variance has relative SE about
\[
\sqrt{2/(R-1)}\approx47\%.
\]
Under a scalar Gaussian approximation, the single-net last-step exponent has SE around **0.5**, and inverse-variance gain estimates are upward biased. Selecting the best method, allocation, and cell compounds this.

**Fix:** report uncertainty for gain and slopes, validate selected methods on fresh scrambles, and require improvement at an affordable budget. Preserve the registered verdict as descriptive rather than treating it as a statistical significance test. Sources: [registered \(R\)](/home/agent/workspaces/mats/work/input-bits/PREREG.md:92), [variance estimates](/home/agent/workspaces/mats/work/input-bits/scratch/part3.py:101), [verdict](/home/agent/workspaces/mats/work/input-bits/PREREG.md:105).

**(CONCERN) The QR basis can contain arbitrary completion directions.** At \(n=64,k=64\), a ReLU layer typically has only about 32 active rows. QR therefore returns columns beyond the propagated tangent rank. Those columns cannot all be interpreted as surviving Lyapunov directions. This does not invalidate covariance-based participation ratios.

**Fix:** record numerical rank and restrict Lyapunov-direction interpretations to supported columns. Source: [QR update](/home/agent/workspaces/mats/work/input-bits/scratch/part1c.py:52).

**(CONCERN) A null offset experiment does not establish that structure sets the gain.** Amendment 2A’s final sentence overinterprets a ratio near one: all tested coordinate blocks could have similarly poor projections, or the same hybrid floor could dominate every offset.

**Fix:** conclude only that the tested placement changes had little measured effect. Pair the experiment with the order-stability and hybrid-floor measurement in T4. Source: [Amendment conclusion](/home/agent/workspaces/mats/work/input-bits/PREREG.md:115).

**(OK) The central amplitude-versus-direction conclusion survives this audit.** The exact split, independent-pilot rotation, replica transition, CVQ zero-mean identity, and multilevel telescoping definitions are sound. What does not survive unchanged is the strength of the scorer verdicts, the interpretation of Gamma as a finite population mean, and any Part 3 claim based on a charged gain at \(4096L\).
