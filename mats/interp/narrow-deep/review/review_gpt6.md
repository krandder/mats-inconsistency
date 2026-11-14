**Verdict: major revision.** The residual K2 mean-estimation advantage largely reproduces, but two analysis bugs change the subspace results, and several theoretical and statistical claims exceed the evidence. No files were edited.

Checked through **2026-10-30 00:42 UTC**. Neither complete L=1024 JSON existed; one n=16, seed-0 result had appeared in `logs/resdepth.log`. References below use the current [report](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md) and files under `work/narrow-deep/`.

1. **The noisy Jacobian analysis computes the wrong population sensitivity matrix.**  
   [scratch/p2_jac.py:76](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_jac.py:76) marks a path permanently ineligible when it encounters any all-off layer; lines 102–114 subsequently exclude it from \(M\). With additive noise, that path can revive. Moreover, death **before the cut** does not imply zero derivative with respect to the state **at the cut**.

   Reproducing the calculation at n=8, L=256, c=128, excluded paths contribute **83.39% of total sensitivity for seed 2 and 99.12% for seed 4**. Including all paths changes mean \(k_F^*(M;0.01)\) from **2.5 to 3.5**, `avg` capture from approximately **0.87 to 0.72**, and `att` from **0.65 to 0.50**. These are material changes to sections 2.3–2.4. If survival-conditioned sensitivity is intended, it must be named explicitly, and its covariance and theoretical comparison must use compatible conditioning.

2. **Euler’s identity does not identify the top right singular vector.**  
   [Report:243](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:243) infers “the state’s own direction is the top one” from \(Jz=F(z)\). That implication is false. A rank-one Jacobian \(J=u v^\top\) satisfies \(Jz=u(v^\top z)\), while its sensitive **input** direction is \(v\), which need not align with \(z\). Radial output alignment and dominant input sensitivity are different objects.

   Also, Euler’s identity in this form applies to the noiseless homogeneous map. The displayed measurements use additive noise; holding those noise draws fixed generally makes the downstream map affine within a gate region.

3. **The experiments do not establish that no gate-free subspace bound exists.**  
   [Report:269](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:269) tests two particular constructions, randomly sampled masks, and a loose product-of-norms bound. Their failure establishes failure of those constructions on those nets. It does not establish impossibility for other weight-derived constructions.

   Distinguish a bound uniform over arbitrary masks from a bound for realizable trajectories under a specified input law. Random masks include unrealizable sequences. Furthermore, weights and the input law already determine the sensitivity matrix; an impossibility claim needs a precise restriction on the permitted computation.

4. **The quoted Lyapunov gaps use inconsistent conditioning.**  
   [scratch/theory_numbers.py:13](/home/agent/workspaces/mats/work/narrow-deep/scratch/theory_numbers.py:13) computes the radial exponent conditional on \(m\ge1\), but the first angular exponent conditional on \(m\ge2\), then subtracts them. Corollary C1 requires **the same conditioning for both**.

   For the infinitesimal angular perturbation, the correct expression is
   \[
   g_1=\frac12\mathbb E\!\left[
   \psi(m/2)-\psi((m-1)/2)\mid m\ge2
   \right].
   \]
   Direct calculation gives **0.25034560 at n=8** and **0.08682311 at n=16**, rather than 0.214 and 0.0864. Without conditioning, the displayed expectation is not finite: \(m=1\) annihilates the angular perturbation, and \(m=0\) kills the reference path. Section 7 states the needed conditioning, but the numbers were not calculated that way.

5. **The random-mask participation ratio includes arbitrary singular vectors of zero matrices.**  
   [scratch/p2_jac.py:138](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_jac.py:138) includes every returned top singular vector, including when \(J_D=0\). Such a matrix has no distinguished top direction.

   Reproducing n=8, seed 2, c=128: **39.94%** of random-mask products are zero. The reported PR is **3.777**; excluding zero products gives **7.010**. Thus the apparent narrow spread and the n=8 P5 “miss” are substantially numerical artifacts.

6. **The “worst-case gain” evidence confuses an upper bound with a lower bound.**  
   [PREREG.md:73](/home/agent/workspaces/mats/work/narrow-deep/PREREG.md:73) specifies the sampled maximum gain divided by actual median gain. The scorecard instead cites the product-of-operator-norms ratio, \(e^{30}\)–\(e^{260}\). A very loose upper bound cannot prove that the true supremum is large. The saved sampled-max results do support the n=16 threshold in the checked configurations, but they are different numbers.

   Separately, the residual quantity at [scratch/p2_jac.py:146](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_jac.py:146), \(\prod_l\|I+cV_lW_l\|\), is **not** a bound over masks. Removing a mask can introduce cancellation. A valid elementary bound is \(\prod_l(1+c\|V_l\|\|W_l\|)\).

7. **The “weights plus one trajectory’s gates” positive result is overstated.**  
   [scratch/p2_jac.py:88](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_jac.py:88) estimates `avg` using **1,024 trajectories’ marginal gates**, not one forward pass. `att` uses a selected surviving reference and constructs a Jacobian and its singular vectors; obtaining gates alone is the one-forward-pass operation.

   The reported reference-path failure—capture 0.07 on one net—and finding 1 above preclude a general sufficiency statement. A valid statement is that one reference trajectory often gives a useful sensitivity direction on the tested, sufficiently synchronized nets.

8. **The death theorem needs stopping and quantifier qualifications.**  
   [Report:617](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:617) says active counts are iid binomial along a noiseless path. They have the binomial conditional law **while the state is nonzero**; after death, every subsequent count is zero. The survival probability \((1-2^{-n})^L\) and the plain-chain rank bound are correct.

   Exact positive-ray synchronization of a collection of nonzero noiseless states does imply shared future gates and collective death. Approximate synchronization does not imply identical gates. A finite census of 4,096 inputs does not establish that the function vanishes for **every** input. The headline’s universal language should become a statement about sampled inputs and the specified ensemble.

9. **The noisy collapse table labels “ever encountered an all-off layer” as current death.**  
   [scratch/p2_collapse.py:43](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_collapse.py:43) accumulates `ever_dead`, while noise can immediately revive the state. The n=8 noisy counts **10/21/37** are counts based on this historical event.

   In the saved data, **zero** nets have `dead_now > 0.95` at depths 64, 128 or 256. Consequently, the table heading “every input dead” and its “>95% dead” qualification are incorrect for the noisy row. The angle measurements also exclude previously dead, revived paths, so they describe a selected population.

10. **The angle fit is an interesting consistency check, not a precision validation of the infinitesimal theorem.**  
    I reproduced the n=16 mean fitted slope **−0.08671388** and median **−0.08621724** using positive recorded median angles over layers 32–128. But these are finite separations, with survival selection, random gate crossings, correlated observations across layers, and network-to-network variation. The theorem concerns infinitesimal perturbations under truncated-chain conditioning.

    [scratch/p2_collapse.py:54](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_collapse.py:54) computes angles with `acos`, which loses resolution near zero. Excluding angles below \(10^{-7}\) changes the mean slope to **−0.08618724**, with a between-network standard error about **0.00535**. “Within 0.5%” is a numerical coincidence much smaller than the sampling uncertainty. The n=8 window changes from the preregistered 32–128 to 8–40, and its claimed excess contraction largely disappears after correcting the gap.

11. **The rank-one test does not test the stated binomial event.**  
    [scratch/p2_jac.py:107](/home/agent/workspaces/mats/work/narrow-deep/scratch/p2_jac.py:107) defines rank one by \(s_2/s_1<10^{-12}\), rather than checking a one-active-unit bottleneck. Strong numerical contraction can satisfy that threshold without exact rank loss. Its “slope” also discards values below that threshold.

    The quoted probability 0.90 includes both \(m=0\) and \(m=1\), whereas the measured sample excludes dead paths. For a noiseless path conditioned to survive 64 layers, the corresponding one-unit-event probability is
    \[
    1-\left(1-\frac{8}{255}\right)^{64}=0.869973,
    \]
    before accounting for the cut gate and network-selection procedure. The comparison **0.92 versus 0.90** is therefore not the claimed exact-law check.

12. **The oracle experiment reuses the truth samples and should not be described as an exact population law.**  
    [scratch/p3_truth.py:38](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_truth.py:38) obtains both final-output truth and intermediate oracle moments from the same paths. [scratch/p3_att.py:39](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_att.py:39) then compares an estimator built from those moments with that same truth.

    This is useful as a paired test of downstream approximation, but shared sampling error can cancel. Errors around \(10^{-7}\) cannot be interpreted as independently validated population nMSE with a generic \(1/N\) truth floor. Also, the table mixes binned K2 and an unbinned noiseless mean push. Binned K2 propagates covariance and ReLU moments; its computational cost is not literally one ordinary trajectory per bin.

13. **The FLOP comparison is approximately defined, not a complete equal-cost accounting.**  
    Charging K2 \(O(n^3)\) and each sampled path \(O(n^2)\) per layer is legitimate: covariance propagation requires matrix–matrix products. [lib/k2.py:61](/home/agent/workspaces/mats/work/narrow-deep/lib/k2.py:61) counts the principal products; I found no missing cubic operation that would erase the residual advantage.

    There are nevertheless concrete omissions: [scratch/p3_samplers.py:44](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_samplers.py:44) prices residual `HYB-lin` as a plain branch; its shallow plain-path cost omits noise at the cut; binning and covariance estimation are uncharged. Regression fitting and Sobol/inverse-CDF generation lack matching cost accounting. [scratch/p4_est.py:52](/home/agent/workspaces/mats/work/narrow-deep/scratch/p4_est.py:52) omits the deductive readout cost. These affect particular methods in different directions. Describe the results as using the stated approximate operation model, and avoid wall-clock implications.

14. **“Sample-free for 18–35 paths’ cost” excludes a substantial sampled setup.**  
    [lib/nd.py:48](/home/agent/workspaces/mats/work/narrow-deep/lib/nd.py:48) obtains the noise scales through a **16,384-path pilot**, uncharged to K2. This is defensible if those scales are supplied as part of the network definition and setup is excluded for every estimator. It is not an end-to-end sample-free, 35-path computation starting from weights and \(\sigma\). State that distinction explicitly.

15. **The own-budget experiment does not evaluate the full declared sampler menu at equal expenditure.**  
    [scratch/p3_resdepth.py:33](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_resdepth.py:33) uses 32 Sobol paths against roughly 35 K2 path-equivalents, and 16 against roughly 18.5. It omits ANT entirely, despite ANT being in the report’s definition of “best simple sampler.” It also saves only mean errors, without replicate uncertainty.

    The same-sample regression estimator is biased in general; the preregistration acknowledges this, and measuring its MSE is valid. It should not be presented as an unbiased control variate. The large K2 advantage looks credible against the tested methods, but the exact **16–250×** range is conditional on this menu, rounding and cost model.

16. **Truth-error subtraction is valid in expectation, but does not remove uncertainty in the realized truth.**  
    The sampler scripts correctly subtract approximately \(1/N_{\rm truth}\) for an independent MC truth estimate. However, all replicates share that estimate, and the reported replicate SE does not include its uncertainty. K2 squared errors are not similarly corrected.

    This matters particularly for the n=8, L=64, seed-0 RQMC-reg result **\(5.91\times10^{-8}\)** at 16,384 paths, which is below the MC truth’s own expected squared error. Independent truth batches or a suitable uncertainty calculation are needed before interpreting such small differences or fitted sampler convergence rates precisely. See [scratch/p3_samplers.py:112](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_samplers.py:112).

17. **The residual depth heuristic has a sound martingale version, but does not establish the reported small normalized bias.**  
    In [report:373](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:373), future closure discrepancies depend on earlier \(V_l\), so the increments are not independent. However, writing the mean error as
    \[
    e_{l+1}=e_l+L^{-1/2}V_l b_l
    \]
    does give martingale differences when \(b_l\) is determined before \(V_l\). Since \(\mathbb E V_l^\top V_l=I\),
    \[
    \mathbb E\|e_L\|^2=L^{-1}\sum_l\mathbb E\|b_l\|^2.
    \]
    A useful depth bound still needs control of the discrepancy magnitudes and, for nMSE, the output-variance denominator.

    The experiments increase discretization depth while keeping total squared residual scale fixed. That is a fixed effective-time regime. Near-linearity, small noise, normalization and the small widths remain alternative explanations for the favorable errors. Two to four nets per row do not establish a general depth-independent floor. Also, L=1024 shares the earlier **W prefix**, but changes the V stream; it is not a completely separate weight stream as the documentation suggests.

18. **The explanatory diagnostics are insufficient to establish the claimed mechanism.**  
    The updated Gaussianity ranges now agree reasonably with the logs. But small average marginal skewness does not establish joint Gaussianity or accurate ReLU expectations near thresholds. At the output, measured average absolute skewness reaches **0.459**. A quadratic regression explaining 96–97% of variance also does not establish correct means or rare tails.

    Likewise, [scratch/p3_cells.py:35](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_cells.py:35) counts joint gate histories. That is not a lower bound on the size of every exact representation or integration algorithm: distinct histories may yield the same map or admit aggregation. The cell census supports difficulty for explicit enumeration, not a general shallow-block obstruction.

19. **The radial-tail validation statement is contradicted by its log.**  
    [Report:459](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:459) claims agreement within 2% at both quantiles, with and without noise. [logs/check_rb.log:2](/home/agent/workspaces/mats/work/narrow-deep/logs/check_rb.log:2) instead gives noiseless \(q=10^{-4}\):
    **MC \(1.035\times10^{-4}\)** versus **RB \(9.046\times10^{-5}\)**, a **12.6% difference**. The quoted relative standard errors are 6.8% and 3.1%, so this is not evidence that the exact noiseless identity fails; it is evidence that the claimed 2% validation was not achieved.

    With noise, replacing radius-dependent noise scaling by scaling at \(E[R]\) genuinely changes the estimand. A check on one residual net does not validate that approximation on near-dead plain nets.

20. **Tail truth and Imhof validation need stronger error accounting.**  
    [scratch/p3_truth.py:54](/home/agent/workspaces/mats/work/narrow-deep/scratch/p3_truth.py:54) chooses the threshold from part of the truth sample and then reuses those observations to estimate its exceedance probability. The quoted fixed-threshold binomial SE does not account for that adaptive reuse. Tail estimator MSEs also retain uncertainty from the shared truth.

    The Imhof formula itself is correct. Its implementation discards quadrature error, and [logs/p3_tails.log:81](/home/agent/workspaces/mats/work/narrow-deep/logs/p3_tails.log:81) contains a convergence warning. Sampling \(2^{22}\) Gaussian draws and observing zero exceedances cannot validate probabilities near \(10^{-9}\). An independent, exponentially tilted inversion gave **\(1.89808\times10^{-9}\)** versus the saved **\(1.91094\times10^{-9}\)** for n=8, L=64, and close agreement at L=256. Thus the roughly **60,000×** tail underestimate survives, but the stated validation was inadequate.

21. **Several preregistration “hits” and scope statements need correction.**  
    All seven hashes match their corresponding document prefixes, including the final file. The problems are scoring and deviations, not hash inconsistency.

    [Report:591](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:591) marks P3 a hit despite mean \(k_F^*(M)=4.0\) at n=16, c=248, and individual values up to 8 in the claimed post-synchronization region. The noise-floor “hit” includes 0.013 above the registered 0.01 ceiling. The P5 claim that every candidate has zero tail after c≥64 is contradicted by `lin`/`rand` tails **0.083/0.118 at c=248** and other late cuts.

    The main truth budget was reduced from \(2^{24}\) to \(2^{21}\)–\(2^{22}\) without an explicit corresponding amendment. P6’s noiseless arm is not represented in the supplied main truth runs. “Everything else … on 4–64 nets” is false: several diagnostics use two nets, the plain sampler comparison uses one, and trained estimation uses a selected best seed per configuration.

22. **The literature summary conflates approximation settings and omits essential hypotheses.**  
    [Report:97](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:97) states that width ≤ input dimension cannot approximate general functions, while the cited compact-domain \(L^p\) result explicitly permits width \(d_x\) in relevant cases. Uniform approximation, compact-domain \(L^p\), and whole-space \(L^p\) must remain separate. Kidger–Lyons gives a sufficient width under activation assumptions, not the universal necessity suggested by “need.”

    “Disconnected decision regions need a wider layer” also requires hypotheses absent here. For example, \(f(x_1,x_2)=\operatorname{ReLU}(x_1)+\operatorname{ReLU}(-x_1)-1\) has width 2 and a disconnected positive decision region. Recheck the Nguyen theorem’s rank/activation assumptions rather than relying on the compressed statement in `notes/refs.md`.

    Finally, these experiments are **bias-free**. Plain and residual architectures remain positively homogeneous and cannot approximate arbitrary continuous targets merely by crossing a width threshold. “A residual stream removes the obstruction” needs that qualification.

23. **The initialization literature is overstated in ways especially relevant at width 8–16.**  
    [Report:118](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:118) says the variance of Jacobian entries grows exponentially. The relevant instability concerns higher moments/squared derivatives. For the square He-ReLU chain, sign symmetry gives \(\mathbb E J_{ij}^2=1/n\); the ordinary entry variance cannot grow exponentially without bound.

    The Hanin–Nica lognormal statement is an asymptotic approximation with conditions and error bounds, not an exact finite-width law. The noiseless norm distribution has an atom at zero. It also does not establish the report’s later claim that the radius across inputs of a **fixed** network is lognormal. The Du–Hu width condition is a sufficient convergence condition, not a theorem that training requires width growing with depth. Li–Nica–Roy’s log-Gaussian statement likewise needs its joint-limit regime.

24. **Some numerical transcriptions and causal training claims remain too strong.**  
    In the block-size table, n=16, c=64 has mean \(k_F^*(S)=\mathbf{1.0}\), not 1.2; n=8, c=224 has mean \(k_F^*(M)=\mathbf{2.75}\), not 3.0. The n=16 “0.4% of paths die” is the ensemble prediction; the noiseless census measures **0.01488%**.

    [Report:495](/home/agent/workspaces/mats/outbox/narrow-deep-2026-10-29.md:495) calls the RNN capacity-limited without separating optimization and training-budget effects; the best run was still improving. “Sorting needs depth and a residual stream” exceeds a comparison of these recipes. And nMSE **0.81243** is better than the mean predictor’s 1, although it fails the chosen success criterion. The approximate sorting success itself is supported.

The following numerical spot-checks reproduce the report’s rounded values, independently of the interpretive problems above:

| Reported quantity | Value recovered | Evidence |
|---|---:|---|
| Noiseless n=8 sampled-all-dead nets, L=64/128/256 | 10 / 25 / 39 | `runs/collapse_plain_n8_L256_s0.json` |
| Mean dead-input fractions, same depths | 0.191013 / 0.410622 / 0.626503 | Same file |
| Noiseless n=16 median angles, L=64/128 | 0.00309676 / 0.0000140347 | `runs/collapse_plain_n16_L256_s0.json` |
| n=16 mean angle at L=128 | 0.00214672 | Same file |
| n=16 \(N_{\rm surv}(0.01)\), L′=4/64 | 5.01858 / 1.31681 | `runs/jac_plain_n16_L256_s0.001.json` |
| n=16, L=64, seed 0: Stein K2 error / cost | \(2.3095\times10^{-5}\) / 35.015 paths | `runs/resdepth_n16_L64.json` |
| Same net: best tested own-budget sampler error | 0.00322367 | Same file |
| n=8, L=256, seed 0: Stein K2 error / cost | 0.000398556 / 18.546 paths | `runs/resdepth_n8_L256.json` |
| All 20 completed depth-row sampler/K2 ratios | min 15.769 / median 42.800 / max 245.918 | `runs/resdepth_*.json` |
| Plain hybrid, budget 256: best hybrid / best sampler | 0.000306012 / 0.00130824 | `runs/p3/*plain_n16_L256*seed0_samplers.json` |
| Plain hybrid, budget 4096: improvement over best sampler | 1.60681× | Same file |
| RNN T=64 best accuracy | 44.148% | [logs/p4_eval.log:73](/home/agent/workspaces/mats/work/narrow-deep/logs/p4_eval.log:73) |
| Residual sorter L=64, seed 0, noiseless nMSE | 0.03586 | [logs/p4_eval.log:41](/home/agent/workspaces/mats/work/narrow-deep/logs/p4_eval.log:41) |
| Noise-trained RNN: exact K2 nMSE | 0.223700 | `runs/p4/est_rnn_T64_s1_sig0.01.json` |
| Partial L=1024, n=16, seed-0 Stein K2 result | \(4.51\times10^{-5}\), about 35 paths | [logs/resdepth.log:21](/home/agent/workspaces/mats/work/narrow-deep/logs/resdepth.log:21) |

Checks that passed: plain and residual Jacobian chains matched autodiff to \(4.4\times10^{-16}\); the residual Stein cross-covariance passed a Gaussian one-step MC check. I found no algebraic error in the plain, residual or independent-input RNN moment updates. The noiseless radial RB identity and Imhof formula are correct.

**Short verdict:** retain the residual K2 result as a promising exploratory mean-estimation result. Recompute the noisy sensitivity analysis and random-mask PR, correct the conditioned exponents, and narrow the synchronization, gate-free, cost and validation claims before circulation. The report’s conclusion that M1/M3 have not yet been reached is justified.
