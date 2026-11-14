The negative result is credible **for the implemented closure**, but section 10 overstates its causal explanation and some tail-estimator gains. I found no layer-index or normalization bug. Checked read-only on 2026-10-30, through 02:07 UTC.

1. **Repeated splits do not preserve the gate-conditioned law, which limits what the experiment refutes.**  
   [k2mix.py:80](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:80), [k2mix.py:124](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:124), [report:876](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:876).

   `split_piece` correctly computes one Gaussian half-space’s moments. However, the next split conditions its **Gaussian replacement**, which has already lost the previous half-space constraint. Likewise, `relu_mom_forced` and the Stein cross-covariance are correct for the replacement Gaussian, not for the actual truncated distribution.

   This is material in the tested configuration: the noisy RNN’s five S1 splits at zero-based step 2 produce 32 components; **26 components, carrying 15.17% of the weight, have a mean contradicting at least one forced gate sign**. An “on” gate can consequently have a negative activation mean. In a simple check with identical Gaussian coordinates, sequential splitting assigns 4.64% probability to each impossible opposite-sign combination.

   This is an approximation built into the method, not an algebra error. Nevertheless, “exact truncated moments” needs qualification, and the result refutes this sequentially re-Gaussianized implementation rather than an accurately conditioned gate mixture. It does not establish that splitting itself cannot help.

2. **The suffix experiment does not justify the causal claim that early RNN errors “do not matter.”**  
   [p5_locate.py:100](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_locate.py:100), [report:859](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:859), [report:911](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:911).

   The suffix calculation is implemented consistently: state moments and output truth come from the same paths, and the restart index is correct. But restarting changes both the starting moments and the subsequent approximation trajectory. Differences between squared output-mean errors are not additive contributions from intervening layers; cancellation and amplification remain possible.

   For the noisy RNN, restarting after step 8 reduces nMSE from **0.22379 to 0.13648**, a **39% reduction**. That contradicts an unqualified dismissal of early errors. For the noiseless RNN, the last eight closure steps can regenerate substantial error, but this does not prove that early errors have no effect.

   The sorter’s gate-error distribution supports “spread” under the registered criterion. It does not establish that corrections are necessary at nearly every layer. Also, the reported “state errors” measure **state-mean error only**, not covariance or distributional error ([p5_locate.py:83](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_locate.py:83)).

3. **The noisy RB column reports variance, not MSE; its advertised gains omit bias.**  
   [p5_rbvar.py:55](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_rbvar.py:55), [report:930](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:930).

   For noisy nets,
   \[
   \mathrm{relative\ MSE}=\frac{\operatorname{Var}(\widehat p)}{p^2}
   +\left(\frac{E\widehat p-p}{p}\right)^2.
   \]
   `gain_exact` includes only the first term. Thus the table’s “relative MSE” introduction and “RB’s gain over MC” label are incorrect for those rows.

   For the noisy sorter’s sorting error at \(q=10^{-3}\), the saved variance is **0.0210**, and the estimated relative bias is **0.0732**. Including that point estimate gives approximately **0.0264**, reducing the gain from **2.85× to 2.27×**. This is illustrative, not a precise bias-corrected estimate, because the bias itself is uncertain.

   The noisy sorter’s **310,000× norm-tail gain** is especially unsupported as an MSE gain: its estimated discrepancy is 2.95% ± 5.01%, while the reported relative variance is only \(2.09\times10^{-6}\). The bias check cannot resolve bias at the scale needed to substantiate that gain. For noiseless nets, mathematical unbiasedness makes the variance comparison appropriate.

4. **The noisy-RNN Imhof values are numerical failures; the sampling replacement is credible but has limited coverage.**  
   [p5_split.py:50](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_split.py:50), [p5_tailcheck.py:18](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_tailcheck.py:18), [logs/p5_split.log:19](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_split.log:19).

   The Imhof formula is correct, but the code discards the quadrature error estimate and retains results despite convergence warnings. The noisy RNN’s saved \(q=10^{-4}\) probability of approximately **0.5** is invalid.

   An independent check using the same integral with eigenvalues and threshold rescaled gives **\(5.88045\times10^{-5}\)** for k = 0. This agrees within sampling error with `tailcheck`’s **\(6.19888\times10^{-5}\)**, against truth \(9.01222\times10^{-5}\). Trust the sampled RNN tails, with sampling uncertainty, rather than the failed integral.

   However, `tailcheck` validates only **exact k = 0 and exact S1 k = 8**. It does not repair every intermediate k, S2, S0, or Stein tail result.

5. **The sorter tail checks do not establish the quoted numerical precision.**  
   [report:848](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:848), [report:907](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:907), [logs/p5_tailcheck.log:1](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_tailcheck.log:1).

   At \(q=10^{-3}\), the k = 0 closure samples contain only **26 exceedances** for the noiseless sorter and **100** for the noisy sorter: approximately 20% and 10% relative standard errors. Agreement with Imhof “within 3%” describes the observed point estimates, not 3%-accurate validation.

   At \(q=10^{-4}\), the corresponding counts are **0 and 2**. These support severe underestimation, but cannot independently validate precise 120–560× factors. Similarly, these samples cannot establish a small percentage change between splitter variants. Report counts or confidence intervals alongside these comparisons.

6. **The largest RQMC ratios are not established lower bounds.**  
   [p5_split.py:87](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_split.py:87), [report:902](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:902).

   Subtracting \(1/N_{\text{truth}}\) is a reasonable expected truth-noise correction, but it does not remove the realized truth error or its cross-term. The corrected RQMC estimate can fluctuate in either direction. It actually becomes **negative, \(-1.28\times10^{-7}\)**, for noisy-RNN exact S1 k = 8 ([logs/p5_split.log:48](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_split.log:48)).

   Therefore, “the largest ratios are lower bounds” does not follow. Those ratios are unresolved near the reference’s precision limit. The large qualitative loss remains persuasive; precise extreme ratios need a better reference or uncertainty bounds.

7. **“Equal FLOPs” is an approximate, conditional comparison, not complete estimator cost.**  
   [k2mix.py:71](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:71), [k2mix.py:144](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:144), [p5_split.py:92](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_split.py:92), [p5_rb.py:75](/home/agent/workspaces/mats/work/narrow-deep/scratch/p5_rb.py:75).

   Component-dependent propagation costs are counted, including the exponential growth after early splits. RQMC correctly receives the largest power-of-two sample count within the nominal budget, which favors the closure.

   Important exclusions remain: the \(2^{20}\)-path oracle used to select gates; readout and mixture aggregation; eigendecomposition and Imhof integration for tails; and RB direction normalization. `F_CHI = 200` and Gaussian-pair costs are fixed approximations, while Sobol/inverse-CDF generation is charged as ordinary Gaussian sampling.

   Also, k = 0 RNN mixture propagation computes an extra normal CDF vector, making its Stein propagation count about **7.6% larger** than `k2.rnn_prop`. These details will not erase the large exact-closure losses, but costs should be labeled as approximate propagation budgets, conditional on oracle-selected gates.

8. **Two headline statements need narrower wording.**  
   [report:809](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:809), [report:925](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:925).

   The **1.13× maximum** compares each split method with its own matching exact/Stein k = 0 baseline. Relative to exact K2, a Stein split reaches about **1.86×** on the noiseless RNN. The 3× rejection survives, but the headline should say “incremental gain from splitting.”

   The radial formula must use **\(\chi_{16}\) for sorters and \(\chi_{64}\) for RNNs**. The code correctly uses `nin`; the report incorrectly writes \(\chi_{16}\) universally. All three statistics are positively homogeneous without noise, and the margin’s label is scale-invariant. The positive-\(G\) handling is correct for these positive thresholds. With fixed additive noise, the implemented typical-radius substitution is approximate. Likewise, the “first state lies on two rays” explanation is exact only without noise ([report:857](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:857)).

9. **Pre-registration scoring is mostly faithful, with qualifications for P13–P16.**  
   [PREREG.md:204](/home/agent/workspaces/mats/work/narrow-deep/PREREG.md:204), [report:963](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:963).

   **P11:** correctly scored hit; the maximum log-suffix drops are 42.77% and 25.89%. **P12:** correctly mixed on accumulated error: top-four shares are 50.33% and 46.83%; one-step results should remain a separate interpretation.

   **P13:** hit for the tested implementation and selection rules, subject to findings 1 and 8. **P14:** correctly partial; very large ratios require finding 6’s qualification. **P15:** correctly falsified as a universal prediction by the noisy-RNN S1 k = 8 result, but unvalidated configurations remain unvalidated. The registration also conflates “factor-of-two error” with relative squared error \(>0.25\); these are not equivalent for overestimates.

   **P16:** correctly partial overall because both RNN margins miss 3×. The noisy-sorter claim needs the variance/MSE correction and an explicit interpretation of the noisy gain threshold: at \(q=10^{-3}\), even its variance-only point gain is **2.85×**. Its later bias estimate, **7.32% ± 2.37%**, does not establish a 10% bound at conventional 95% confidence.

   The current `PREREG.md` SHA-256 exactly matches the recorded **01:14:09Z** hash. File timestamps support the stated ordering before the first result at 01:14:47Z; the header’s “~01:20Z” is inconsistent with that timestamp.

10. **The core formulas, indexing, and weight handling passed the requested checks.**  
    [k2mix.py:17](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:17), [k2mix.py:34](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:34), [k2mix.py:151](/home/agent/workspaces/mats/work/narrow-deep/lib/k2mix.py:151).

    k = 0 reproduces K2 to floating-point precision on all four nets, for both exact and Stein moments. A \(2^{20}\)-sample Gaussian check supports the forced-gate moments and Stein cross-covariances. A layer-0 split of the trained noiseless residual net agrees with next-layer MC moments within sampling error.

    Layer indices agree between diagnostics and splitting. Means and tails normalize component weights. Components below weight \(10^{-14}\) are deliberately dropped; with eight splits, the total possible discarded mass is at most approximately \(5.1\times10^{-12}\). I found no evidence that accidental dropping or normalization explains the negative result.

11. **Numerical spot-checks matched the artifacts.**

    | Section-10 quantity | Recomputed/saved value | Evidence |
    |---|---:|---|
    | Sorter σ=0 diagnostic nMSE | 0.0236652 | [locate log:1](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_locate.log:1) |
    | Sorter σ=.01 diagnostic nMSE | 0.0450615 | [locate log:19](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_locate.log:19) |
    | RNN σ=0 diagnostic nMSE | 0.0392399 | [locate log:13](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_locate.log:13) |
    | RNN σ=.01 diagnostic nMSE | 0.223790 | [locate log:7](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_locate.log:7) |
    | Sorters’ worst-eight accumulated shares | 4.9099%, 4.8365% | [noiseless JSON:1](/home/agent/workspaces/mats/work/narrow-deep/runs/p5/locate_mlp_res_L64_s0_sig0.0.json:1), [noisy JSON:1](/home/agent/workspaces/mats/work/narrow-deep/runs/p5/locate_mlp_res_L64_s0_sig0.01.json:1) |
    | RNN σ=0 suffix after 56 steps | 0.0567670 | [locate log:14](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_locate.log:14) |
    | Sorters’ exact k=0 path-equivalent costs | 669.603, 523.076 | [split log:1](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_split.log:1), [split log:62](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_split.log:62) |
    | Best incremental splitting gain | 1.13025× | [split log:79](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_split.log:79) |
    | Noiseless sorting-error RB variance gain, \(10^{-4}\) | 3.74941× | [rbvar log:4](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_rbvar.log:4) |
    | RNN margin gains, \(10^{-4}\), σ=0/.01 | 1.76986×, 1.82743× | [rbvar log:10](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_rbvar.log:10), [rbvar log:20](/home/agent/workspaces/mats/work/narrow-deep/logs/p5_rbvar.log:20) |

Verdict: **Revise before finalizing.** Keep the measured gate-concentration results and the failure of this particular split closure. Narrow the causal conclusions, disclose the repeated Gaussian projection, correct noisy RB variance-versus-MSE claims, and remove unsupported precision and lower-bound language.
