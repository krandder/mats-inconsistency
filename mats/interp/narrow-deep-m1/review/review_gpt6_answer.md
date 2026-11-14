The evidence supports **“these implementations did not reach M1.”** It does not support several stronger claims about unavoidable bottlenecks, universal error floors, or why every isolated win should be discounted.

Checked against the supplied code, JSON results, logs, and registration hashes on **2026-10-31, through 01:18 UTC**. No files changed. Numerical checks used the specified Python, CPU only, at most two threads, under `nice -n 19`.

1. **The headline 330× TS advantage violates the registered truth-resolution rule, and its supposedly losing neighbours actually win.**

   [PREREG.md:46](/home/agent/workspaces/mats/work/narrow-deep-m1/PREREG.md:46) permits ratios only when the method’s nMSE is at least \(10/P\). Here that threshold is **5.9605e-7**; the celebrated net-217 result is **8.9891e-8**, only 1.51 times the truth’s expected error. Reporting its **334×** ratio as resolved is not justified.

   This does **not** remove net 217 from the winning-net count. Its other, truth-resolved configurations include:

   | TS configuration, net 217 | Cost, paths | nMSE | RQMC/TS |
   |---|---:|---:|---:|
   | GHM, c=2, τ=1e-3 | 1,143.7 | 2.1773e-5 | 7.42 |
   | GHM, c=4, τ=1e-3 | 2,461.4 | 1.3996e-5 | 5.15 |
   | GHM, c=4, τ=1e-4 | 15,390.9 | 1.8927e-6 | 3.14 |
   | K2X, c=2 | 28.45 | 2.5069e-4 | 54.01 |

   Thus “its neighbours lose” is false. The strongest admissible ratio on this net is approximately **54×**, using the existing interpolation.

2. **The cancellation diagnosis is partly supported, but “the wins do not count as a method advantage” goes beyond the evidence.**

   [bias_decomp.py:33](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/bias_decomp.py:33) defines its “forward bias” as `TS estimate − exact-law downstream estimate`. That combines **forward-law error, meet quadrature/projection error, and potentially different reference-class choices**. It does not isolate the forward law.

   For net 217, substantial opposing errors are convincing: approximately +4.75% and −4.78% of output SD. Their individual estimates have approximately **0.27 percentage points** of Monte Carlo SE, however; their very precise cancellation is an algebraic consequence of subtracting the same diagnostic estimate.

   BO cancellation has stronger direct support. I reconstructed its registered directions and refined the k=4 rule:

   | Net | Registered winning m | Winning noisy nMSE | m=32 noisy nMSE | m=64 noisy nMSE |
   |---|---:|---:|---:|---:|
   | 220 | 8 | 7.7980e-5 | 3.1867e-4 | 3.1748e-4 |
   | 221 | 4 | 9.1433e-5 | 6.4045e-3 | 6.3778e-3 |

   On 221, coarse quadrature clearly cancels a large projection error. On 220, cancellation also helps, but the refined noiseless projected estimate itself has nMSE **1.45e-5**, so the projection is reasonably effective.

   These are unstable finite-budget advantages, not fictitious advantages. Moreover, **TS’s K2X win on net 221 is not explained by the reported cancellation**: at c=4 its decomposition has both terms positive, approximately +3.11% and +0.47%. A genuine advantage for that configuration and budget remains compatible with the evidence. Generalisation requires a configuration-selection rule and fresh nets.

3. **The claim that every error is effectively one signed scalar is directly false.**

   The report says the orthogonal component is below 1% of the along-direction component. Existing results contradict this:

   - Net 220, downstream diagnostic at c=2: along **−0.013888 SD**, perpendicular **0.006525 SD**, a ratio of **47%**.
   - Net 220, winning BO configuration: along **−0.005860 SD**, perpendicular **0.006606 SD**, a ratio of **113%**. The perpendicular component accounts for approximately **56% of its nMSE**.
   - Net 203, winning BO configuration: perpendicular/along is approximately **15%**.

   Noiseless synchronisation does not imply that the noisy mean, noiseless mean, and all estimator errors share one direction. The vector residual must remain in the decomposition.

4. **The S-based direction construction does not implement the usual whitened sensitivity projection.**

   In [bvf.py:174](/home/agent/workspaces/mats/work/narrow-deep-m1/lib/bvf.py:174) and [oracle_k_by_cut.py:117](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/oracle_k_by_cut.py:117), eigenvectors \(v\) of
   \[
   S=C^{1/2}MC^{1/2}
   \]
   become Euclidean observation directions through `qr(C½ @ V)`. The retained coordinates are then \(U^\top h\).

   For the corresponding whitened coordinates, the observation directions should instead span **\(C^{-1/2}V\)**; \(C^{1/2}V\) supplies reconstruction directions. These are different when C is anisotropic. I checked a rank-one linear example where one correctly chosen coordinate captures the function exactly; the implemented construction captures only **99.03% of its variance**.

   Also, outside the Gaussian input, linear regression supplies the **best affine predictor**, not generally the conditional mean asserted in the report.

   Consequently the per-cut counts describe this particular projection procedure, not the minimum number of directions the value function “needs.” This matters particularly at cuts 1–8.

5. **The direction-count scoring was changed after registration.**

   Amendment 1 specifies the **smallest k** meeting the error threshold. [analyze.py:159](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/analyze.py:159) instead requires **every subsequent k** also to pass: “stably below.”

   This materially changes reported numbers at the 1e-5 threshold:

   | Net, cut | Registered smallest k | Implemented stable k |
   |---|---:|---:|
   | 203, c=7 | 2 | 4 |
   | 216, c=8 | 1 | 3 |
   | 217, c=1 | 6 | 8 |
   | 217, c=3 | 5 | 7 |
   | 217, c=8 | 3 | 7 |
   | 221, c=3 | 4 | 8 |

   Stability is a defensible additional diagnostic, but it must be labelled post hoc and shown alongside the registered statistic. Threshold classifications also need the stored `se_nmse` uncertainty; currently it is ignored.

6. **GHM has additional approximations and support violations that prevent attributing its floor solely to piece shape at the next layer.**

   The single-halfspace formulas in [ghm.py:38](/home/agent/workspaces/mats/work/narrow-deep-m1/lib/ghm.py:38) are correct. My mixture recombination check recovered the original mass, mean, and covariance to floating-point precision.

   The subsequent processing has important limitations:

   - Free/free cross-covariances use \(\Phi_i C_{ij}\Phi_j\), **not exact Gaussian ReLU covariances**, contrary to the registration’s “exact K2 ReLU moments” wording. For standard normals with correlation 0.9, this gives **0.225**, versus the exact **0.2956143**.
   - Re-Gaussianising a halfspace-conditioned component discards its support. Later splits can move the mean of a previously “forced-on” coordinate below zero; the final forced-gate operation leaves that negative mean unchanged.
   - On dev net 103’s first noiseless layer, I found **15 components with negative output means**, carrying **0.002659 total weight**; the most negative entry was **−0.41839**. A true post-ReLU conditional mean cannot be negative.
   - Even a two-dimensional correlation-0.9 example, fully split, gives one marginal mean **0.413601** instead of **0.398942**.
   - “Pruning” stops further splitting of light components; it does not remove their mass. That is reasonable, but it is not an exact retained mixture of realised histories. The `max_comp` guard also omits current unsplit components from its prospective count, so it is not a strict cap.

   [ghm_shape_test.py](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/ghm_shape_test.py) demonstrates that better layer-1 moments do not cure this implementation. It does not separate Gaussian shape loss from approximate cross-covariances, sequential forced-gate inconsistencies, or stopping rules. Its “exact” initial moments are empirical estimates, especially uncertain for rare pieces.

7. **“Every operation is charged” is false, although the major propagation costs are included.**

   The counters generally charge forward matrix operations, probe Jacobian propagation, reference-class passes, and downstream evaluations. Important omissions or inconsistencies remain:

   - BO: angular-rule construction, trigonometric evaluations, weight normalisation, final weighted reductions, and the direction eigensolve.
   - TS: some probe construction/decomposition, sorting/indexing, sigma-point construction, spread calculations, and final reductions.
   - `meet_direct` charges \(U^\top C U\) as `2*mmf(k,n,k)`. For n=8, k=4, that is **480 operations/component**, whereas the two dense products require **720** under the stated convention.
   - GSF merging omits eigendecomposition and sorting costs; its pairwise-distance charge is also approximate.
   - TS tail costs are accumulated into a discarded `Flops()` object. BO’s chi-survival calculations and reductions are uncharged, whereas RB explicitly adds 200 FLOPs per survival evaluation.
   - RQMC is assigned MC’s input-normal cost. Sobol scrambling/generation and inverse-CDF conversion have no separate accounting.

   There are overcharges and inefficiencies too: direct meets evaluate zero-weight centre nodes, and SH constructs unused grids and directions after the switch to the direct variant. Thus these are approximate counters, not merely systematic undercounts.

   SH and HYB also call `flops_path(L=shallow_cut)`, which excludes noise after that cut, although their shallow simulations include it: **144 FLOPs/sample missing at width 8**.

   I reproduced the BO-tail medians using **`cost_full`**, so the reported tail table does appear to charge full noiseless propagation, rather than incorrectly using the cheaper D0 cost. Adding its missing 200 FLOPs/node changes those medians only slightly.

   A factor-two sensitivity check leaves M1 unmet: halving deterministic costs gives three winning nets for each arm; doubling them leaves two. Individual thresholds and P9’s “at most two wins” are therefore cost-sensitive, while the four-net M1 criterion survives this check.

8. **RQMC generation is sound, but its statistical presentation and interpolation need correction.**

   [samplers.py:65](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/samplers.py:65) correctly uses independently scrambled Sobol input points, inverse-normal transformation, and pseudorandom independent layer noise. This is the registered input-only RQMC baseline; it is not QMC over all noise coordinates.

   The subtraction of \(1/P_{\rm truth}\) is correct **in expectation** for an independent MC truth, up to the estimated variance denominator. I reproduced the stored sampler nMSE calculations from their raw errors.

   Problems remain:

   - All replicates share one noisy truth. Subtracting its expected error does not eliminate its realised error; the reported replicate SE omits uncertainty from that common truth.
   - `np.std` uses `ddof=0`, underestimating the usual estimated SE, particularly at R=2.
   - Registration specifies R=16 at \(2^{17}\) and R=4 at \(2^{20}\); code uses **8 and 2**. These reductions are not documented in the amendments.
   - `analyze.py` silently discards nonpositive corrected nMSEs but retains positive ones, including truth-limited \(2^{20}\) results. This is selective censoring, not a resolution policy. It contradicts “\(2^{20}\) is not used.”
   - `compare.py` does not incorporate the small-budget files; below 128 paths it extrapolates the old curve. It therefore need not reproduce `analyze.py`’s comparisons.
   - Log-log interpolation is a reasonable descriptive convention between measured budgets, but it is not a measured arbitrary-N Sobol result. Ratios lack propagated uncertainty, particularly near a 3× decision threshold.

   The small-budget additions themselves are useful, appropriately labelled post hoc, and do not invalidate the experiment.

9. **P1/P4 are scored using a different kprop meet from the registered one.**

   [PREREG.md:75](/home/agent/workspaces/mats/work/narrow-deep-m1/PREREG.md:75) specifies a degree-K polynomial meet using kprop’s moments. The report principally uses a **Gaussian reconstruction from only its mean and covariance**, selected automatically ahead of polynomial results in `analyze.py`.

   These metrics differ enormously. For net 203, KP3f at c=2 has:

   - Gaussian meet: approximately **3.80e-4**;
   - registered degree-3 polynomial meet: approximately **291**.

   The amendments disclose Gaussian diagnostic results on development nets but do not clearly redefine the scored forward-law metric. Both metrics should be reported and their roles distinguished.

   Further deviations include reducing mixture meet draws from \(2^{20}\) to \(2^{18}\), polynomial-fit draws to \(2^{19}\), and omitting the registered kprop ×1.3 cost sensitivity.

   P4’s “same total FLOPs” comparison also lacks an executable downstream/meet cost. Comparing a stochastic oracle meet’s error plus only 1–13 paths of forward cost against RQMC at an assumed 30 paths is useful diagnostic evidence, but not an equal-compute estimator comparison.

   Finally, “P4 holds at cuts 4–8” contradicts the report’s own net-217 KP2 c=4 Gaussian-meet counterexample.

10. **The deep-noise diagnostic is useful, but several implementation details weaken the interpretation and reproducibility.**

   In the current [bvf.Deep constructor](/home/agent/workspaces/mats/work/narrow-deep-m1/lib/bvf.py:74), `noise_table=False` is the default. Neither `deep_diag3.py` nor `deep_noise_effect.py` explicitly enables it. Therefore their currently labelled “D1 K2-Stein table” / “K2-table residual” values are actually **D0**. The older development logs reflect a different default and cannot be reproduced by running the supplied scripts unchanged.

   Additional issues:

   - `deep_noise_effect.py` normalises by the variance of outputs averaged over four future-noise draws. That equals
     \[
     \operatorname{Var}(E[Y\mid H])+\tfrac14 E[\operatorname{Var}(Y\mid H)],
     \]
     rather than the target total variance. Its nMSEs are inflated; this matters especially on noise-driven secondary nets.
   - The ray tables initialise covariance with layer-32 noise, then apply the correction to states **already after layer-32 noise**. That is a proxy model, not the exact conditional value function of those states.
   - The quoted K2 radius comparisons sometimes use the nearest table radius rather than the MC radius.
   - GSF’s integrated residual also includes reference-ray approximation, radius interpolation/clipping, closure error, merging, and finite-sample error. It does not isolate a single Gaussian-closure mechanism.

   Nevertheless, the small-radius closure failure is real. I evaluated K2 at the same stated radii. Stein effects were **204.06** and **9.1678**, versus logged MC effects approximately **64.1** and **2.04**. Switching to exact bivariate Gaussian ReLU moments worsened them to **255.00** and **11.5401**. This strengthens the conclusion that improving pair moments alone does not solve this deep closure problem; it still does not establish an unavoidable deterministic barrier.

11. **The exact-piece cost argument is not an accuracy-certified lower bound.**

   [ghm_shape_test.py](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/ghm_shape_test.py) tests one orthant with one lattice construction. The reported MC probability is **0.00386405**, whose relative SE is **0.784%**. The \(2^{18}\)-point lattice estimate **0.00382511** differs from it by approximately **1.01%**.

   Agreement between two lattice resolutions does not establish **0.1% accuracy**. There is neither a sufficiently accurate reference nor an error bound.

   Multiplying one orthant’s cost by estimated piece and moment counts is a plausible naive implementation estimate, not evidence that all exact-piece algorithms cost “at least \(10^5\) paths.” Shared calculations, adaptive integration, alternative representations, and required aggregate accuracy are unexamined.

   Also, beyond layer 1, the noisy network’s exact history regions involve the injected noises as well as the eight input coordinates. Calling all exact pieces eight-dimensional Gaussian polytope integrals is only appropriate to a noiseless formulation or an additional approximation.

12. **Exact point evaluation is repeatedly confused with exact integration or a global linear map.**

   The noiseless passes and Jacobians are correctly indexed; I reproduced \(J(h)h=f(h)\) to approximately **9.1e-16 relative error**.

   But:

   - `J_ref` is exact **only inside its gate class**. The implementation applies it to other states without checking class membership or supplying an error bound.
   - A class carrying 99% of sampled output mass is not the entire support.
   - `meet_direct` evaluates its nodes exactly under D0, but its 2k-point integration rule drops other covariance directions. Its spread threshold does not bound error near a gate boundary.
   - A fixed noiseless-bias magnitude is an **asymptotic bias of that approximation**, not a lower bound on every finite-budget estimate: other errors can cancel it, as these results demonstrate.

   The TS-tail approximation has a related problem. Even for a locally linear scalar output \(g^\top h\), the correct variance is \(g^\top Cg\); `tail_from_meet` instead uses input-radial variance scaled by a norm gain. Therefore TS-tail failure cannot be attributed exclusively to Gaussian pieces being unable to reproduce radial tails.

13. **The registration hashes check out, but the scorecard needs substantial repair.**

   I reproduced all four recorded SHA-256 hashes, including the earlier documents as prefixes of the final registration. The available result timestamps and BO elapsed-time logs are consistent with amendments preceding affected deterministic results; I found no evidence of a late amendment. Amendment 3 openly acknowledges earlier evaluation baseline results.

   The unlisted changes above mean “all deviations registered” is nevertheless false. The maximum over an evaluation ladder should also remain explicitly an oracle envelope, rather than a demonstrated deployable configuration-selection policy.

   My reading of P1–P10 is:

   | Prediction | Defensible scoring |
   |---|---|
   | **P1** | Mean-error clause supported on the three completed forward-study nets; minimum **4.3199e-3**. Meet clause needs its registered metric and denominator restored. One failing net alone does not refute an original “≥5 of 6” condition. No observed e₈≤1e-4 refutation. |
   | **P2** | Supported on the measured nets; mark the remaining net pending rather than claiming the complete six-net condition. |
   | **P3** | Tested D0 downstream misses. “Cannot reach at any cost” and the formal >10,000-path refutation are stronger than the finite oracle experiment establishes. |
   | **P4** | Strong bottleneck assertion is not established; the reported Gaussian diagnostic supports a cut-2 miss, but its metric and total-cost qualification must be explicit. “Holds at cuts 4–8” is incorrect. |
   | **P5(a)** | Not met: **2/6** nets, even after excluding the unresolved 334× ratio. |
   | **P5(b)** | Not met: clear counterexamples with losses exceeding 2×. |
   | **P6** | Miss: maximum measured SH advantage in the registered range is **1.974×**, on net 221. |
   | **P7** | Broadly supported for the tested configurations. Preserve tail-truth uncertainty and RB’s noise-rescaling bias. |
   | **P8** | Partial/pending, as requested. |
   | **P9** | Broad empirical support, subject to corrected direction definitions/statistics and cost sensitivity. BO has two measured winning nets under the stated counter. |
   | **P10** | The first clause is supported: four healthy nets exceed 1e-4 deep-noise effect. The scorecard omits the second clause. Net 217 has a **3.14× TS win at 15,391 paths**, above \(2^{13}\). If “stays within” means uniform performance across all larger budgets rather than a pointwise claim, state that interpretation and test it explicitly. |

14. **Several remaining summary statements need narrower wording or numerical corrections.**

   - Input counts are **7–8**, not always 7. The report’s own stable-count table has **2–3 directions at cut 8 on three of five nets**, not four; two nets have 7.
   - “Converged BO projection bias” is not established by m=8–16; those figures still contain quadrature error.
   - “HYB loses on all six nets” is false if it means every measured budget: net 221’s HYB beats RQMC at both \(2^9\) and \(2^{13}\).
   - TS tails are not uniformly too small: net 203 includes an approximately **4.4× overestimate**. Zero estimated tails also have no finite “1,000×” error factor.
   - BO’s median tail advantage of **0.8×** on net 203 is a loss, not a lead. I reproduced the other quoted medians, approximately **2.52, 2.97, 4.72, and 2.34**.
   - One width-16 net versus heterogeneous width-8 nets cannot establish approximately linear synchronisation scaling, general Gaussian improvement, or exponential fixed-error cost. The comparison also mixes upstream-history and downstream-class counts.
   - “As hard as the original integral” does not follow from needing seven directions under one projection rule.
   - The noise scale is based on the activation before the current noise injection **in an already noisy pilot trajectory**. “Clean activation” should be clarified to avoid suggesting a noiseless pilot.

The BO sphere weights and \(\mathbb E[\chi_k]\) factor are mathematically correct for the noiseless projected Gaussian problem. The midpoint rule is simply low order and orientation-dependent: for k=4, its maximum second-moment error was **0.015165 at m=4**, falling to **0.000811 at m=16**. GSF’s narrowing split and merge also preserved total first and second moments in my check, to approximately **4.4e-16**.

**Verdict: major revision.** Retain the narrow negative M1 result; revise the causal claims, scoring, and accounting before treating this as evidence of fundamental blockers.

Numbers independently reproduced: all four registration hashes; the sampler \(1/P\) corrections; **54.01×** admissible TS advantage on net 217; **10.29×/14.46×** BO advantages on 220/221; the BO refinement results above; GHM’s **0.225 versus 0.295614** covariance discrepancy and negative component means; and the deep K2 effects **204.06/9.1678**, worsening to **255.00/11.5401** with exact pair moments.
