The recorded K1′ and K-F failures are correct. The main problems are interpretation, cost accounting, and a surviving LRG bug.

1. **Major — Report §1, “No push rule can do better,” and Summary, “threefold.”** Representation error is not a lower bound: closure error can cancel it. Subtracting two nMSEs does not isolate propagation error. SPLIT3 reduces total push nMSE **1.57×**; **3.13×** is merely the ratio of nMSE differences. Report that distinction and measure push-versus-exact output differences directly.

2. **Major — Summary, “merges nothing at threshold 0.8”; §2.2 downstream groups.** `runs/diag7/*.json` contradict this: downstream-effect groups fall **136→111** on net 102/layer 16, **101→95** on 101/16, and **92→89** on 600/8. Correct the counts; distinguish no co-switching merges from limited downstream-effect merging.

3. **Major — Summary, “switch nearly independently,” “Gates do not compress,” and M1’s required gate bits.** Small pairwise correlations establish neither joint independence nor an information requirement for estimating means. Summed marginal entropy is not joint entropy. State only what these thresholds and single-linkage diagnostics found. The **26–68%** range is correct, but the predictor lacks a majority-sign baseline, so it does not measure latent information alone.

4. **Major — Summary, relevance-axes bullet.** A leading eigenvalue carrying **8.6–11.7%** of trace is far from isotropic in 256 dimensions: isotropy gives **0.39%**. Also, net 100’s AS1 representation error **1.85e-4** beats PCA1’s **2.091e-4**. Replace “nearly isotropic” with “less concentrated than the state covariance,” and restrict the consistent disadvantage to **AS3 versus PCA3**.

5. **Major — §3, “All were taken”; [lib/lrg.py:33](/home/agent/workspaces/mats/work/boundary-family/lib/lrg.py:33).** The criticized covariance construction remains unchanged. With `m=(1,0)` and `S=[[1,.9],[.9,1]]`, its joint covariance has minimum eigenvalue **−0.06641**. Correct the completion claim and validate/fix the chart. The width-64 scale numbers reproduce, but do not establish the proposed mechanism; width, depth, and seed also differ between settings.

6. **Major — Summary’s factored-model verdict and M1 conclusions.** K-F rejects **F3/F8/F3C with the specified PCA latents and Gaussian diagonal residuals**, not every conditionally independent model with ≤8 latent dimensions. K1′ rejects the registered fitted constructions under its operational rule. Say these prototypes failed the development gates; three development nets do not rule out M1 or broader cell families.

7. **Major — Summary, “perfect cell description” and “reaches the sampler’s error”; §§2.1–2.3.** Fits from \(2^{18}\) paths are not perfect, and K1′ retains representation-sampling and fitting uncertainty. Its **7.728e-6** push result is also above Kerdock’s **6.7e-6**, on different nets and budgets. K-F uses \(2^{17}\)-path fits, references, and model draws; `A_vs_B` is one noisy comparison, not a universal floor. Present these as oracle diagnostics, not matched estimator comparisons.

8. **Major — Summary’s FLOP estimates; §2.4; §3 “cost-aware.”** [lib/lcg.py:106](/home/agent/workspaces/mats/work/boundary-family/lib/lcg.py:106) returns no cost; re-binning, eigendecompositions, slicing, merging, and readouts lack complete accounting. Neither registered kill rule contains a FLOP condition. Label **2e9/4e10** as unverified estimates. At an assumed **2e9**, ordinary Monte Carlo’s expected nMSE is **7.4688e-5**, supporting “near 8e-5” only conditionally.

9. **Major — Summary tails bullet; §2.6 “representation ceiling.”** One \(2^{18}\)-path fit on one net, with eight outputs per level, establishes failure of these fitted mixtures—not a Gaussian-cell ceiling. “16 radial cells” actually means **16 PCA1 bins**. Qualify the conclusion and include fitting/reference uncertainty; splitting-reference relative SEs reach approximately **42%**.

10. **Major — §2.6, “chi-mixture saddlepoint … 56% … 18%.”** Those numbers belong to the **plain saddlepoint U3**. ARC’s raw `tails/results/summary_s0.001.json` gives chi-mixture **U3r: 55% and 28.75%** on all events. Correct the method attribution and identify the different evaluation-net set.

11. **Major — §2.4, “residual given the latent is not Gaussian.”** [scratch/diag1w.py:61](/home/agent/workspaces/mats/work/boundary-family/scratch/diag1w.py:61) measures standardized **z within finite bins**, not the residual conditional on continuous latent coordinates. Bin truncation itself creates non-Gaussianity. Rename the claim “within-bin distributions are non-Gaussian”; testing continuous conditional residuals requires another measurement.

12. **Major — §2.4 width-64 LCG numbers and §2.5.** The supplied logs/JSON do not archive the reported width-64 LCG, corrected split/grid results, or withdrawn-run variance shrinkage. `DESIGN.md` repeats some claims but is not raw evidence. Supply the original outputs and configurations or mark these numbers unverified.

13. **Major — §4, “Third cumulants give conditional means quadratic … Fourth cumulants give heteroscedasticity.”** Finite cumulants do not determine conditional means or covariances. Describe this as a proposed polynomial closure, then test its reconstructed moments and representation error.

14. **Minor — Summary ratios; §2.4 layer 8; Handoff truths.** Raw recomputation gives **0.84276–8.19007×**, hence **0.84–8.2×**; tail factors from median log ratios are **3.6899 / 7.5328 / 241.976**, hence **3.7 / 7.5 / 240×**. Layer-8 excess kurtosis is **0.044775**. Net 100’s current truth contains **\(2^{26}\)** paths, not \(2^{25}\). The **2.2–4.3×**, **26–68%**, and **2–7×** ranges check out.

15. **Minor — §2.3 introduction and §1 metric.** `diag8.py` measures mean nMSE of **\(a_{m+1}=\mathrm{ReLU}(z_{m+1})\)**, not \(z_{m+1}\), and does **not** subtract reference \(1/N\). Variance error is the median absolute relative error across neurons. Correct these labels.

16. **Minor — Registration/artifact provenance.** Original/current PREREG hashes, `diag8.py`, and append-only DESIGN prefixes match; ARC weights and noise scales match. The added `diag5.py` SPLIT branch does not affect registered families and is not a substantive deviation. However, `diag5.py`/dependencies were not hashed, and SPLIT3 overwrote net 101’s K1′ JSON. Preserve separate run artifacts and identify the surviving K1′ log as its evidence.
