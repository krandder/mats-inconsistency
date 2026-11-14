# Noisy AlgZoo RNNs: can the noisy-channel program interpret ARC's 2nd-argmax models?

29 October 2026. Work directory `work/algzoo-noisy/` (code, logs, `results/`, `PREREG.md` with sha256 and times in
`prereg_hashes.txt`, `GPU-LOG.md`). Reviewed in four adversarial rounds (GPT-6, `work/algzoo-noisy/review/`). The
progress log follows the summary.

## Summary

*Final, 05:01Z. The noise criterion uses the final models. The structure analyses of the wide models use their
2^19-step checkpoints, whose losses are within 4% of the final ones; the DAG sizes were rechecked on the final models
of two seeds.*

We added Gaussian noise to every hidden activation of ARC's 2nd-argmax RNNs, at each step, at a fixed fraction sigma of
each neuron's activation scale. We then applied the noisy-channel program to the variants that are as good as the
originals. Every number below gives sigma, the model and the compute.

1. **Noise.**
   - The criterion is Kelvin's: noise on, loss within 2% of the original, accuracy not lower.
   - At ARC's widths only one variant passes: a fine-tune of ARC's 432-parameter example M_{16,10} at sigma = 0.001,
     about 10 bits per neuron per step. Its loss is 13% lower and its accuracy 0.6 points higher.
   - At sigma = 0.003 the fine-tune misses even when run 4x longer (loss +8.3%). Training from scratch at ARC's full
     length also misses (loss +27% for the best seed, whose noiseless twin beats the example by 18%). At sigma = 0.01
     the gap is +175%. M_{2,2} and M_{4,3} pass at no sigma >= 0.001.
   - With spare width the criterion is met at practical noise.
     - Width 32 and width 64 at sigma = 0.01 (6.6 bits) pass in all three seeds each, with loss 0.08-0.11 and
       0.05-0.06 against 0.147.
     - Width 64 at sigma = 0.03 (5.1 bits) has 8-16% lower loss but 0.23-0.70 points lower accuracy than the example.
       Two of its three seeds pass against zoo seed 2.
2. **Interpretation.**
   - The sigma = 0.001 model (T16 below) is ARC's network plus negligible noise: same fragility, same structure.
   - Its question ("is the output the 2nd argmax?") has no sparse structure in any form we tried:
     - DAGs need 78-92% of the 2,480 edges, whether the target is per-input agreement, the error rate or the loss;
     - simplifications flip answers both ways but raise the error rate on net, so targeting the average adds little
       slack;
     - relative to the task's own variables (running max and 2nd max, their positions), it needs 58%;
     - the question needs up to 13 of 16 directions per step (9-13 from step 4 on).
   - We reproduce ARC's partial interpretation and add to it:
     - the max-tracking neurons {1, 2, 4, 6, 7}, plus 8, are the algorithmic part (R^2 >= 0.99);
     - neuron 7 alone carries 39-52% of the noise that reaches the decision.
   - The errors:
     - they are the network's deviations from its own algorithmic content: a readout of the algorithm-conditional
       state errs 1.0% against the network's 4.0%;
     - they concentrate on negative second maxima (76% error) and on near-ties of the 2nd and 3rd values (half of all
       errors);
     - they are decided late, at new-max or new-2nd-max steps at twice chance.
   - Seven of nine preregistered scenario predictions hit, on T16 and on width 32.
   - The wide noisy models are sturdier and sparser with width: 75-80% of edges at width 32 and 43-58% at width 64
     (sigma = 0.01, two seeds each); 22 of 32 and 25-32 of 64 directions. Relative to the algorithm, width 64 at
     sigma = 0.01 needs only about 30% of edges for 1% flips (one seed).
   - Fitted algorithm-indexed descriptions predict the wide models' error rates to within 3-6% (input-exact on width
     32, algorithmic variables only on width 64 at sigma = 0.03).
3. **ARC's bar** (three columns, MSE against FLOPs over weight-perturbed copies; the advice is never fitted to the
   copies).
   - Column (a), deductive estimates: every description we built for M_{16,10}, M_{4,3} and the wide models loses to
     Monte Carlo, by 10^2 to 10^5 at equal FLOPs. The cause is that Gaussian closure misplaces probability across the
     decision thresholds: a single description step from exact states is already 14 points off.
   - Column (b), the same descriptions plus a little sampling, is no better than Monte Carlo at equal FLOPs. It is
     worse at small budgets, where the deductive setup dominates.
   - Column (c): plain sampling. The best samplers are RQMC (1.3-2x) and stratification (1.2-1.4x).
   - The exception is M_{2,2}. ARC's region analysis plus one linear-response noise term gives its error rate in closed
     form at 1,432 FLOPs, about 10^6 times lower MSE than Monte Carlo, and a certified upper bound 65x tighter than
     sampling's.
4. **Blocker.** To match sampling at about 10^8 FLOPs, a deductive estimate of these error rates needs a bias below
   about 0.2 points. That is 3-5% relative accuracy.
   - Only descriptions that carry the input variables exactly reach this (M_{2,2}, n = 2). That is ARC's
     region-counting route, which does not scale with sequence length.
   - Fitted Gaussian-residual descriptions of the wide noisy models come within 3-6%, but no deductive way of building
     them that we tried does. The closest deductive estimate is for width 64 at sigma = 0.03, with advice fitted on the
     same model: 8-9% low.
   - On weight-perturbed copies the description must recompute the last step with each copy's weights. On the wide
     models that version is 0.8-6.6 points high on the copies on average, and 1.8-8.6 points high on the base models
     themselves, so the problem is the description, not the copies being unseen. It loses to Monte Carlo by 2 x 10^3
     to 3 x 10^4 at equal FLOPs.
5. **Follow-up (section 9, Amendment 7).** The errors are mostly systematic in the algorithm's own variables: the
   top-3 values and positions explain 54% (T16) and 79% (W64 at sigma = 0.01) of the input-to-input variance of the
   error probability (lower bounds).
   - A deductive estimator that integrates exactly over those variables and uses only the weights at each point
     (TOP3) is off by +10% and +27% when integrated exactly. The earlier descriptions were off by 3x to 44x.
   - It still loses ARC's bar by 64-4,400x, because its quadrature converges slowly. With a little sampling it gains
     up to 2-3x per sequence and breaks even at equal FLOPs.

## Progress log

- 00:20Z. Started. Read the briefing files, ARC's AlgZoo post and the three linked notes (Robinson, Hilton, Chen),
  cloned `alg-zoo` and downloaded the 2nd-argmax weights (2x2, 4x3, 16x10 and others, all 5 seeds; the example 16x10).
- 00:27Z. PREREG Part 1 and Amendment 1 (Kelvin's stricter criterion: loss within 2%, accuracy not lower, noise on).
- 00:34Z. Amendment 2: from-scratch arm, best-of-5 comparison, sigma = 0.1 stopped for 16x10 (training diverged).
- 00:48Z. PREREG Part 2 (interpretation predictions, kill tests, endpoints), written before any per-scenario error
  breakdown.
- 01:07Z. Step 1 (fine-tunes): only the example M_{16,10} at sigma = 0.001 meets the criterion. First interpretation
  results on it: the question-driven DAG keeps 82% of edges at 1.3% flipped answers (kill test K1 fires); 13 of 16
  directions per step are needed; the deductive Gaussian-mixture description predicts 73% error vs 3.8% true; errors
  are decided late (half of the error information first appears at step 9). Message sent to main.
- 01:13Z. Amendment 3 (coordinator's request): wider noisy RNNs (d = 32, 64 at sigma 0.01, 0.03) and full-length
  from-scratch d = 16 runs, on one on-demand L4. L4 stockouts in 17 zones; the VM came up in asia-southeast1-b at
  01:22Z (`GPU-LOG.md`).
- 01:30Z. Scenario tests, propagated-noise analysis, meeting diagnostic, family scorecard and the M_{2,2} closed form
  done; ARC's bar for M_{16,10} and M_{4,3} running.
- 01:37Z. ARC's bar done for T16, M_{4,3} and M_{2,2}. The deductive description loses by about 10^5 (T16) and 200x
  (M_{4,3}) to Monte Carlo at equal FLOPs. On M_{2,2} the closed form wins by about 10^6, and a certified one-sided
  bound is 65x tighter than sampling's. Adversarial review 1 (GPT-6, high effort) started on the T16 sections. The GPU
  run is at 8% of 2^20 steps: at step 82k, d = 64 at sigma = 0.01 has training loss 0.09 and d = 32 has 0.14-0.16,
  against the example's 0.147. If the evaluation confirms it, width rescues the criterion at sigma = 0.01. ETA about
  04:00Z.
- 02:37Z. Review 1 (GPT-6) fixes applied: cross-fitted information; step-matched event rates (errors are decided at
  algorithmic events about 2x chance, reversing the first claim); all-competitor linear response (gate flips do not
  explain the noise gap); held-out DAG target (85%); per-step ranks; corrected CIs, FLOP caveats and the prediction
  scorecard.
- Kelvin's two follow-ups were added as Amendments 4/4b and 5:
  - the accuracy- and loss-targeted DAGs need 82-92% of edges and 13-15 directions, the same as the flip-DAG;
  - in the three-column ARC's bar for T16, (a) the deductive estimate is 10^5x worse than Monte Carlo and (b) adding a
    little sampling is no better than (c) plain Monte Carlo;
  - on M_{2,2}, columns (a) and (b) are at the truth's noise floor, 10^4-10^6x better than (c).
- 2^19-step GPU checkpoints fetched. Criterion evaluation and the interpretation battery are running on width-32 and
  width-64 models.
- 02:55Z. Wide-arm preview at 2^19 steps:
  - d = 32 and d = 64 pass at sigma = 0.01 (loss 0.050-0.095 against 0.147);
  - d = 64 at 0.03 passes the loss clause but not the accuracy clause against the example (94.8-95.1% against
    95.35%); it passes against zoo seed 2;
  - d = 16 from scratch still fails at 0.003.
  Message sent to main. Amendment 6 (descriptions relative to the algorithm, the coordinator's steer):
  - algorithm-relative ablation keeps 58% of T16's edges (unconditional ablation: 85%);
  - the algorithm-mean readout errs 1.0%, and the Gaussian-residual description predicts 12.9%, against the true 4.05%.
- 03:40Z. Review 2 (GPT-6, high effort) fixes applied:
  - DAG sizes now meet their held-out targets (W64 60%, W32 algorithm-relative 60%);
  - equal-FLOP comparisons for column (b) corrected (worse than Monte Carlo at small budgets; M_{2,2} (b) 1,200-2,100x,
    not 10^4-10^6x);
  - "no cancellations" withdrawn (0.30% of answers corrected and 0.41% broken);
  - L3 scored as not run as registered;
  - the r-recalibration boundary and the advice compute stated;
  - the quadrature accuracy of the algorithm-indexed description is being checked.
- 03:50Z. GPU run finished (2^20 steps; $2.12; VM deleted, instance list checked). sigma* = 0.01 at widths 32 and 64 in
  every seed; width 64 at 0.03 misses the example's accuracy. Message sent to main.
- 04:00Z. Review 3 (GPT-6) fixes applied: C1 scored on the registered edge set (a near miss at 30.4%), B1 split into
  its two clauses, overstatements about advice transfer and deductive misses corrected, final-run numbers in place of
  checkpoint numbers, S4 and rounding fixed. Quadrature checks done (the default grid is 0.1-0.5 points low). Wording
  pass done. Still running: the width-64 algorithm-relative bar (partial results reported).
- 04:17Z. Seed check of the DAG sizes on the final GPU models: W32 at sigma = 0.01 needs 76-80% of edges in seeds 0
  and 1 (seed 0's checkpoint: 75%), and seed 1 needs at most 22 of 32 directions (seed 0: 22); W64 at sigma = 0.01
  seed 0 needs 58% (seed 1's checkpoint: 43%). Section 5.2 and the summary give these ranges (corrected in review 4).
  Width-64 bars at 8 and 7 of 16 copies.
- 04:56Z. Width-64 bars finished. The algorithm-indexed description loses to Monte Carlo by 3 x 10^4 (sigma = 0.03,
  bias +6.6 points) and 1.7 x 10^3 (sigma = 0.01, bias +0.8 points) at equal FLOPs, and is worse than the oracle
  constant. Its bias on the copies is no larger than on the base model, where the last-step version is already 13.6%
  against 5.0% (sigma = 0.03): the description fails, not the transfer. Column (b) has 0.8-1.1x Monte Carlo's MSE
  per sequence and is 2.5-32x worse at equal FLOPs. Review 4 (GPT-6) started on these changes.
- 05:01Z. Review 4 fixes applied: DAG sizes counted only where the held-out target is met (W32 75-80%, W64 at sigma =
  0.01 43% and 58%), the untested rank 21 no longer counted, column-(b) ratios taken from the saved Monte Carlo
  rows, and the W64 prediction rows qualified as one seed. The biases, MSEs, correlations and oracle comparisons of
  the width-64 bars were confirmed. Report final.
- 05:09Z. The coordinator's bounded next step pre-registered as Amendment 7: how much of the error is systematic in
  the algorithmic variables, and a deductive estimator with no closure in them (TOP3).
- 05:13Z. Step 1 done: the top-3 values and positions explain 54% (T16) and 79% (W64) of the error-probability
  variance on held-out inputs. The stop rule (30%) did not fire.
- 05:57Z. TOP3 done on both families. Integrated exactly it is +10% (T16) and +27% (W64) off; column (a) loses by
  64-4,400x at equal FLOPs (slow quadrature); column (b) gains up to 2-3x per sequence and breaks even at equal FLOPs.
  Section 9 written; review 5 (GPT-6) started.



## 1. Setup

**Models.** These are AlgZoo's 2nd-argmax RNNs:
- h_0 = 0, h_{t+1} = ReLU(W_hh h_t + W_hi x_t), logits = W_oh h_n, inputs x ~ N(0, I_n);
- the target is the position of the second-largest input.

We used:
- M_{2,2} (10 parameters; zoo seed 2);
- M_{4,3} (32 parameters; zoo seed 0);
- M_{16,10} (432 parameters): ARC's `example_2nd_argmax()` ("the example", accuracy 95.3%) and
  `zoo_2nd_argmax(16, 10)` seeds 0-4 (the best is seed 2, at 94.9%).

The weights come from ARC's public bucket. Our forward pass reproduces the example's accuracy (95.35%).

**Noisy variant.**
- h_{t+1} = ReLU(W_hh h_t + W_hi x_t) + sigma r_{t+1} eps_{t+1}, with eps ~ N(0, I_d), at every step including the last.
- r_{t,i} is the RMS of neuron i's post-ReLU activation at step t under the noisy model, frozen after training.
- Per-neuron scales keep the model invariant under ARC's neuron-rescaling symmetry.
- sigma = 0.001 / 0.003 / 0.01 / 0.03 / 0.1 corresponds to about 10 / 8.4 / 6.6 / 5.1 / 3.3 bits per neuron per step
  (Gaussian channel at SNR 1/sigma^2).

**Criterion (Kelvin's, fixed in PREREG Amendment 1 before any noisy run finished).**
- Evaluation: noise on at test time, one draw per input, 2^22 fresh sequences, paired with the original.
- Pass: loss L_var <= 1.02 L_orig, and accuracy not significantly lower (A_var >= A_orig - 2 SE).
- We did not use the coordinator's alternative ("within the originals' seed-to-seed spread"). For M_{16,10} it would
  be far looser, since ARC's five seeds span loss 0.165-0.328.

## 2. Step 1: noise

**Test-time noise on ARC's weights.** No retraining; single-draw accuracy on 2^22 sequences. The table for all seeds
and sizes is in `work/algzoo-noisy/results/step1.md`.

| model | noiseless | sigma 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|
| M_{2,2} seed 2 | 0.9999 | 0.9967 | 0.9892 | 0.9561 |
| M_{4,3} seed 0 | 0.9846 | 0.9552 | 0.8853 | 0.6796 |
| M_{16,10} example | 0.9532 | 0.8263 | 0.5791 | 0.2680 |
| M_{16,10} seed 2 | 0.9486 | 0.7302 | 0.4279 | 0.2009 |
| M_{32,10} seed 0 | 0.9944 | 0.8004 | 0.4966 | 0.2593 |

**Fine-tuned with the noise on.**
- Fine-tunes start from ARC's weights and run 2^16 steps; rows marked L ran 2^18.
- Evaluation is paired, on 2^22 sequences. "Gap" is L_var / L_orig - 1.
- The 95% CI of L_var - L_orig is in each run's JSON (`eval.dL_ci95`).
  - Its width is under 1% of L_orig for every M_{16,10} and M_{4,3} run at sigma <= 0.003, and 1.8% for M_{2,2} at
    0.001.
  - It is wider only where the gap is itself many times larger: up to 7.5% of L_orig for M_{16,10} at 0.03, and 39-158%
    for M_{2,2} at sigma >= 0.01, where the gaps are 9x to 88x.

| model | sigma | loss var / orig (gap) | accuracy var / orig | meets |
|---|---|---|---|---|
| M_{16,10} example | 0.001 | 0.1274 / 0.1470 (-13.3%) | 0.9593 / 0.9532 | **yes** |
| | 0.003 (L) | 0.1591 / 0.1470 (+8.3%) | 0.9457 / 0.9532 | no |
| | 0.01 (L) | 0.4038 / 0.1470 (+175%) | 0.8859 / 0.9532 | no |
| | 0.03 | 2.4253 / 0.1470 (+1550%) | 0.7344 / 0.9532 | no |
| | 0.1 | diverged (loss > 1,000 at 37k steps; stopped) | | no |
| M_{16,10} seed 2 | 0.001 | 0.1781 / 0.1657 (+7.5%) | 0.9443 / 0.9484 | no |
| | 0.003 (L) | 0.2214 / 0.1657 (+34%) | 0.9273 / 0.9484 | no |
| M_{4,3} seed 0 | 0.001 | 0.0589 / 0.0575 (+2.5%); L: 0.0593 (+3.2%) | 0.9848 / 0.9845; L: 0.9841 | no |
| | 0.003 | 0.0689 / 0.0575 (+20%) | 0.9790 / 0.9845 | no |
| M_{2,2} seed 2 | 0.001 | 0.0013 / 0.0011 (+11%); L: +14% | 0.9997 / 0.9999 | no |
| | 0.003 | 0.0021 / 0.0011 (+89%) | 0.9991 / 0.9999 | no |

- The example passes at sigma = 0.001 because the fine-tune keeps training a model that was below its optimum: its
  noiseless error falls from 4.6% to 3.8%, and the noise then adds 0.3 points back.
- M_{2,2} cannot pass at a practical sigma. Its errors become noise-dominated: the noiseless error is 7.7e-5, and
  the closed form of section 4 gives about 0.32 sigma for the fine-tuned variants (3.2e-4 at 0.001, 9.4e-4 at 0.003,
  3.2e-3 at 0.01). Relative noise cannot be outscaled by larger weights.
- M_{4,3} misses at sigma = 0.001 by 0.5-1.2% of loss beyond the 2% allowance.
- The noise-marginalised predictive (16 draws) beats the single-draw one: loss 0.1324 vs 0.1591 at sigma = 0.003.
  Under a marginal-loss criterion, the example would pass at sigma = 0.003. We kept the single-draw criterion we had
  fixed in advance.

**Why the networks need about 10 bits per neuron (the propagated-noise metric).** These are the program's node weights
for the DAG.
- Along each input's noiseless path, with the gates held fixed, we compute how much noise injected at (step t,
  neuron i) moves the target-logit margin m. Its noise s.d. is s(x).
- Code: `noisesens.py`, on 2^17 inputs. Measured errors come from 2^20 noisy samples.

| model, test sigma | error noiseless / linear, top competitor / linear, all competitors / measured | median and 10% quantile of abs(m)/s | share of decision noise: neuron 7 / {1, 2, 6, 7, 8} / steps 8-10 |
|---|---|---|---|
| ARC example, 0.01 | 0.046 / 0.144 / 0.178 / 0.174 | 1.43 / 0.35 | 52% / 89% / 53% |
| ARC example, 0.001 | 0.046 / 0.048 / 0.047 / 0.049 | 14.3 / 3.5 | 46% / 88% / 56% |
| T16 (fine-tuned at 0.001), 0.001 | 0.038 / 0.040 / 0.039 / 0.041 | 14.5 / 3.5 | 39% / 87% / 63% |
| T16, 0.01 | 0.038 / 0.138 / 0.173 / 0.174 | 1.45 / 0.35 | 44% / 88% / 60% |

- At 1% noise, half of all inputs have a margin of less than 1.45 noise s.d.
- Linear response with every competitor logit and the gates held fixed reproduces the extra error at sigma = 0.01:
  17.30% predicted against 17.35% measured for T16, and 17.8% against 17.4% for ARC's weights. With only the top
  competitor it gives 13.8%. So gate flips add almost nothing. (A first version used the top competitor only and
  attributed the gap to gate flips; review 1 found the error.)
- At sigma = 0.001 the noise adds only 0.2-0.3 points. The linear prediction recovers about half of that, which is at
  the level of the sampling error of the 2^15-input prediction.
- The decision noise enters almost entirely through ARC's max-tracking neurons. (The node shares are for the margin
  against the noiseless top competitor, weighted toward inputs near the boundary; they are not an exhaustive
  decomposition of all errors.) Neuron 7 alone carries 39-52% of it.
  ARC describes it as roughly max(0, x_0..x_{t-1}) - x_{t-1}, a difference of large terms that is read out with a
  large gain. Together, neurons 1, 2, 6, 7 and 8 carry 87-89%.
- Fine-tuning at sigma = 0.001 did not make the network sturdier. T16 has the same margin-to-noise distribution as
  ARC's weights; the fine-tune only lowered the noiseless error. In effect, T16 is ARC's network plus noise at
  10 bits per neuron.

## 3. Interpretation of T16 (the example fine-tuned at sigma = 0.001)

The question is "does the model output the 2nd argmax?". Its input distribution is x ~ N(0, I_10). Its answer, for a
given input, is the error probability over the noise. T16's error rate is 4.06% (2^24 samples; 3.78% without noise).

### 3.1 Forward chain: describing h_t step by step

**Least-squares members of each family, fitted to the real activations.** These are the best members for
reconstructing the state, not for predicting the error rate. Another member could predict the scalar error rate better
(review 1). Code: `fam.py`, 2^18 sequences; error predictions on 2^18 fresh sequences.
- R^2 is the fraction of h_t's variance the family explains.
- "Unexplained bits" is 1/2 log2 det(I + Sigma_res / (sigma^2 diag r_t^2)), averaged over cells. It is a
  Gaussian-channel indicator of how much the residual exceeds the noise. It is not a proven minimum description cost
  for these non-Gaussian residuals.

| family (cells; within-cell model) | parameters | R^2 at t = 3 / 6 / 10 | unexplained bits at t = 3 / 6 / 10 | predicted error rate (true 0.041) |
|---|---|---|---|---|
| G: one Gaussian | 152 | 0 / 0 / 0 | 103 / 131 / 133 | 0.900 |
| AC: algorithmic cells (a_t, b_t), one Gaussian each | 13,680 | 0.37 / 0.45 / 0.46 | 71 / 89 / 92 | 0.186 |
| CL: cells, linear in (M, S, x_last) and their ReLUs | 22,320 | 0.985 / 0.945 / 0.925 | 45 / 68 / 73 | 0.132 |
| CL3: CL plus the 3rd max and its position | 39,600 | 0.996 / 0.972 / 0.968 | 41 / 65 / 70 | 0.133 |
| CLX: cells, linear in every input so far and its ReLU | 42,480 | 0.996 / 0.981 / 0.976 | 41 / 61 / 66 | 0.106 |

Here a_t and b_t are the positions of the running max and 2nd max, and M and S their values.
- The algorithmic state describes ARC's neurons almost exactly. In the noiseless example, a per-cell linear fit in
  (M, S, x_last) and their ReLUs gives R^2 >= 0.993 at every step for neurons 1, 2, 4, 6, 7 and 8: ARC's subcircuit
  plus neuron 8, which Chen's notes describe as about ReLU(2nd max - x_t) early on. The other ten neurons fall to
  0.72-0.93 by step 10.
- Even so, the fitted members get the error rate wrong in every family, by 2.6x for CLX up to 22x for the single
  Gaussian. The residual is 2-7% of the variance, but it is not Gaussian and it is not independent of the decision.
  Spread as a Gaussian, it puts probability on the wrong side of decision boundaries that the network respects by
  construction.
- At this sigma, even the input-exact linear family leaves 66 bits per step unspecified, about 4 bits per neuron.
- We did not compute the program's per-step coverage cost D_inf(K q || q) or waste KL(q || K q) for these families.
  Both need log K q, which at sigma = 0.001 is a 17-dimensional integral concentrated in a region of width sigma.
  For any full-dimensional family the waste grows as 1/sigma^2 (phase R), so here it would be in the tens of
  thousands of bits. The unexplained
  bits and the predicted error rate are the per-step and end-of-chain scores we can report.

**Deductive chain** (`adf.py`; no samples of the network).
- The description is a Gaussian mixture over (h_t, M_t, S_t), indexed by the algorithmic cell and optionally by
  selected gate patterns.
- Each step applies exact one-constraint truncations for the algorithmic events and for up to k gates, statistically
  linearises the other gates, adds the noise exactly, and merges by moment matching.

Results:
- The algorithmic-cell weights are exact to about 4% (each 0.049-0.052, against 0.05, at t = 5).
- The h-moments are not: at t = 5 a neuron that is near zero in a cell has mean 0.08 but mean 1.04 in the description,
  and its s.d. is 0.54 against 2.53.
- Predicted error rate: 0.745 (k = 0, 4e7 FLOPs), 0.737 (k = 2, 1.4e8), 0.731 (k = 4, 5e8), 0.714 (k = 4 with the gate
  patterns of neurons 4 and 9 kept as mixture components, 209 components, 1.2e9).
- On noiseless M_{2,2} it predicts 10.6% (true 0.008%); on M_{4,3}, 30% (true 1.5%).

### 3.2 Backward functions and meeting in the middle

**v_t(z) = Pr[error | state at step t]**, with z = (h_t, a_t, b_t, M_t, S_t), estimated by 64 continuations from each
of 8,192 real states (`backward.py`). We bin states by v_t and measure the plug-in mutual information with the
realised error of the original run. This gives a lower bound on I(E; Z_t); the permutation control is 0.0005 bits.

| step t | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| lower bound on I(E; Z_t) / H(E), network state + algorithmic state | 0.01 | 0.02 | 0.05 | 0.08 | 0.17 | 0.29 | 0.51 | 1.00 |
| cross-fitted lower bound on I(E; A_t) / H(E), algorithmic state alone (a, b, M and S quartiles) | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.01 |

- Errors are mostly decided late, with the caveat that these are lower bounds: they cannot show that information is
  absent earlier. The bound is at most 8% of H(E) through step 6 and 51% at step 9.
- Of the 311 errors, 228 are decided only at steps 9-10, meaning v_t stays at or above 0.5 from then on.
- Errors are decided at algorithmic events about twice as often as chance. Of the decided errors, 25% fall on a step
  with a new max and 19% on a new 2nd max. Matched to the same decision steps, chance gives 11.5% each (at step t a
  new max and a new 2nd max each have probability 1/t).
- This replaces a first version that used all-step base rates and concluded the opposite; review 1 caught it.
- A coarse algorithmic state (cell plus quartiles of M and S) carries at most a few percent of the error information
  by a cross-fitted, binned estimate. That estimator is weak with 1,440 sparse bins, so this is only a lower bound.
- A first version that was not cross-fitted reported 37% at step 10. It was an in-sample artefact (review 1).
- Consistency check: E_{p_t}[v_t] = 0.040-0.041 at every t, against 0.038-0.041 by direct sampling.

**Meeting with the deductive description on either side** (`meet.py`). We use the identity
Pr[E] - <q_m, psi_m> = <p_m - q_m, v_m> + <q_m, v_m - psi_m>, with q_m the ADF mixture after m steps and psi_m the ADF
run from a point state to the end.

| m | 1 | 2 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|
| <q_m, v_m> (forward description, exact continuation) | 0.23 | 0.35 | 0.49 | 0.59 | 0.67 | 0.71 |
| <p_m, psi_m> (real states, backward description) | | | 0.65 | 0.55 | 0.35 | 0.18 |
| <q_m, psi_m> (both) | | | 0.71 | 0.69 | 0.73 | 0.71 |

The truth is 0.041. One step of the description is already wrong, in either direction:
- Forward: the merged Gaussian at step 1 smears the two rays h_1 = ReLU(+-W_hi x_0) into states the network never
  visits.
- Backward: a single final step from exact real states, with the last input's truncation moment-matched, gives 0.18.
  The question's events are thresholds on the same variables that the closure smears. So the Gaussian closure fails
  within a single step; depth is not needed to produce the error.

### 3.3 Question-driven DAG

**Construction.**
- Nodes are (neuron, step). The edges are 2,304 recurrent, 160 input and 16 readout, 2,480 in all.
- An ablated edge is replaced by its mean contribution. Surrogates run on the same inputs and noise as the network.
- The question divergence is flip = P(the surrogate's correctness differs from the network's).
- The description cost is the number of kept edges. The forward-only cost is the chain-rule path KL in nats.
- Code: `dag_run.py`. Edges are ranked by first-order, near-boundary saliency; the program finds the smallest prefix
  meeting the flip target on 2^13 fit inputs, then removes edges greedily, and evaluates on 2^15 held-out inputs.

**Results.** These searches are heuristic (a binary search over a non-monotone curve plus one greedy pass), so they show
which dense surrogates work, not that no sparse one exists.
- No sparse surrogate was found. Fit-set targets 2% / 1% / 0.5% keep 1,933 (78%) / 2,038 (82%) / 2,106 (85%) edges;
  their held-out flips are 2.4% / 1.34% / 0.87%.
- The smallest DAG that meets the preregistered held-out 1% target keeps 2,106 of 2,480 edges (85%).
- For scale, two independent noise draws on the same inputs already flip 1.5% of answers.
- Keeping the top 62% of edges by saliency flips 16.5% of answers; keeping 41% flips 52%.
- Structure from the question helps only a little. Ranking edges by their forward (path-KL) saliency flips 6.1% of
  answers at 2,048 kept edges, against 1.2% for the question ranking.
- The surrogates above ablate 370-550 edges and have path KL of 4.5e6-5.7e6 nats at sigma = 0.001. At this
  precision a forward-only description, scored by path KL, would have to keep nearly everything.
- Per step, the flip <= 1% DAG keeps these recurrent edges by source step t = 1..9: 77, 204, 219, 234, 232, 233, 241,
  239, 236 of 256. It keeps 6-14 of 16 input edges per step and all 16 readout edges.
- The only context-specific sparsity is at step 1: ten neurons have no input weight to speak of, so their outgoing
  edges can go.
- ARC's subcircuit is fully used: 216 of its 225 internal edges are kept, and every one of neurons 1, 2, 4, 6, 7 has
  out-degree >= 13 at every step from 3 on. The DAG keeps just as much of everything else.

**Question subspaces** (`lowrank.py`; `lowrank_step.py` for the per-step version). At each step the description keeps
the top k directions of C^{1/2} F C^{1/2}, where C = Cov(h_t) and F is the near-boundary Fisher of the margin. The rest
is replaced by the best linear reconstruction. Same inputs and noise; 2^16 held-out inputs.

| kept directions per step (of 16) | 8 | 10 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|
| flip, question directions | 0.48 | 0.23 | 0.061 | 0.0070 | 0.0036 | 0.0001 |
| flip, top principal components | 0.75 | 0.71 | 0.58 | 0.51 | 0.27 | 0.039 |

- Per step (review 1 pointed out that a uniform rank cannot show what each step needs): reducing one step at a time,
  with the others full, the question needs these numbers of directions for flip <= 1% at steps 1-10: 2, 4, 7, 9, 10,
  10, 11, 12, 13, 9. The top principal components need 2, 5, 8, 13, 15, 16, 16, 16, 16, 15.
  - Early steps need about as many directions as inputs seen plus one or two.
  - The last step needs 9, the logit differences.
  - The middle and late steps need 10-13 of 16.
- Second-order "question variance" is concentrated: 4 directions hold 98.7% of it at step 5 and 6 directions hold
  97.5% at step 10. Yet the answers need 13.
- The structure is the same without noise. On ARC's noiseless example the question needs 13 directions too (flip
  0.8%), and the saliency-prefix DAG needs 2,099 edges (85%) for a held-out flip of 0.89%, against 2,081 for T16 before
  greedy pruning (`results/lowrank_orig_ex.json`, `results/dag_orig_ex.json`).
- The decisions depend on low-variance directions through gate thresholds and the argmax, which a second-order metric
  does not see. The same blind spot makes the edge-saliency ranking non-monotone.

### 3.4 Structure for the average: accuracy- and loss-targeted DAGs (Amendments 4, 4b)

Kelvin asked whether we search for the structure that minimises the error of the loss prediction. The flip-DAG asks
for each input's correctness to match, which is stricter than matching the average.

**The average-targeted search** (`ldag_run.py`, `lowrank_avg.py`):
- The question divergence is the bias of the average: dp = |p_sur - p_net| (primary; ARC's estimand is accuracy) and
  dL = |E CE_sur - E CE_net| (secondary).
- The tolerance is sampling's standard error at N_ref network samples, N_ref = 2^10, 2^14, 2^18.
- Against cancellation, the bias must also stay within each stratum's own sampling SE in 13 strata fixed in advance:
  4 quartiles of the top-two gap and 9 groups of (max position, 2nd-max position).
- Mean ablation as before, with edges ranked by loss saliency. The prefix is searched on 2^15 inputs; held-out checks
  use 2^16 inputs. The table is prefix-only (`results/ldag_T16_prefix.json`). A block-greedy pruning pass for dp at 2^14
  ran 45 minutes on the loaded host without finishing and was stopped. On the flip-DAG a greedy pass removed only 2%
  beyond the prefix.

| question | tolerance (N_ref) | edges kept (of 2,480) | held-out bias, worst stratum / tolerance | directions per step (same rank at every step) |
|---|---|---|---|---|
| per-input flip <= 1% (section 3.3) | | 2,106 (85%) | | 13 |
| dp (error rate) | 6.1e-3 (2^10) | 2,043 (82%) | 0.47 | 13 |
| dp | 1.5e-3 (2^14) | 2,139 (86%) | 0.73 | 13 |
| dp | 3.8e-4 (2^18) | 2,272 (92%) | 0.78 | 15 |
| dL (cross-entropy) | 1.6e-2 nats (2^10) | 2,040 (82%) | 0.47 | 13 |
| dL | 4.0e-3 (2^14) | 2,106 (85%) | 0.46 | 13 |
| dL | 1.0e-3 (2^18) | 2,261 (91%) | 0.51 | 15 |

The average question needs about as much structure as the per-input one:
- Simplifications flip answers both ways. For the dp-targeted surrogate at 2^14, 0.30% of inputs are corrected and
  0.41% broken, so some cancellation occurs and the search uses it. But the net change is upward in every sweep and
  subspace row we ran: dp >= 0 in all 28 sweep rows and all 64 subspace rows, with one held-out exception of -3e-5,
  within noise.
- So the average-targeted search finds little extra sparsity under its tolerances. (A first version said there were no
  cancellations and appealed to the loss optimum; review 2 showed both were unsupported.)

### 3.5 Deviation prediction and scenarios

These predictions were written in PREREG Part 2 before any per-scenario breakdown. They came from the mechanism:
- ARC's features are max(0, ·) features, so they cannot rank negative values;
- the noise smears near-ties.

Each scenario was tested on 2^20 fresh inputs with one noise draw each (`devpred.py`). The overall error rate there is
p = 0.0404.

| scenario | prediction | network: error rate [95% CI] | DAG surrogate (82% of edges) | result |
|---|---|---|---|---|
| S1: exactly one positive input (the 2nd max is negative) | >= 5p | 0.756 [0.748, 0.765] (18.7p) | 0.784 | hit |
| S2: all inputs negative | >= 0.5 | 0.878 [0.859, 0.898] | 0.902 | hit |
| S3: max - 2nd max in its lowest decile | >= 2p | 0.116 [0.114, 0.118] (2.9p) | 0.124 | hit |
| S4: 2nd max - 3rd max in its lowest decile | >= 2p | 0.211 [0.209, 0.214] (5.2p) | 0.224 | hit |
| S5: the max comes after the 2nd max | >= 1.3x the reverse order | 0.0451 vs 0.0357, ratio 1.26 | 0.0481 vs 0.0384 | miss |
| S6: the max or the 2nd max is the last input | <= 0.5p | 0.0503 (1.25p) | 0.0512 | miss (wrong direction) |
| S7: noise-sensitive inputs (0.05 < eps(x) < 0.95) | share >= 3x higher near ties | 19.0% near ties vs 4.4% overall | | hit |
| S8: DAG rates within 25% of the network's on S1-S6 | all six | all six | | hit |
| V1: R^2 of the DAG's per-input error probability | >= 0.75 | 0.81 (noise floor 0.985) | | hit |

Where the errors are:
- Inputs whose 2nd max is negative (S1 and S2, about 1.1% of inputs) are wrong 76-88% of the time, which is about 20%
  of all errors. This is the max(0, ·) blind spot that Chen's notes describe for the handcrafted model.
- Near-ties between the 2nd and 3rd max (lowest decile) hold about half of all errors.
- By position, the error rate is roughly flat: every (max, 2nd max) position cell lies between 1.6% and 10%. The
  highest are max at 9 with 2nd max at 0 (9.8%), and max at 0 with 2nd max at 7 (9.9%).
- The prediction that the last input is easy failed. Errors are decided at the last steps whatever the positions.

The deviation predictor is weak as a description: it is the network with 18% of its edges mean-ablated. The scenario
hits come from the mechanism-level reading (max(0, ·) features, near-ties); no compact description produced them.

### 3.6 Descriptions relative to the algorithm (Amendment 6, coordinator's steer)

Mean ablation replaces a part by its unconditional mean, and on net that pushes toward error. Here an ablated edge
instead carries the conditional mean given the task algorithm's state: the running max and 2nd max, their positions,
and the last input. The conditional mean is the CL regression of section 3.1, fitted once on the base model; it is
advice. Input edges are always kept, so 2,320 edges can be ablated. Code: `algrel.py`.

| T16 | value |
|---|---|
| every recurrent and readout edge carries only its conditional mean | flip 3.6%; error rate 1.0% (the network's is 4.0%) |
| saliency prefix with fit-set flip <= 2% / <= 1% (held-out flip) | 1,128, 49% (2.05%) / 1,347, 58% (0.97%) of 2,320 edges |
| same with unconditional means (section 3.3, out of 2,480) | 78% / 85% |
| description size at step n (90 cells, B and Sigma) | 22,320 parameters, against the network's 432 |

- Relative to the algorithm the network is sparser: 58% of edges against 85%. Prediction C1 (<= 40%) misses on T16.
- The surrogate that keeps nothing but the algorithm-conditional means is a better 2nd-argmax solver than the network
  (1.0% error against 4.0%). The network's errors are its deviations from its own algorithmic content.
- The algorithm-indexed description with a Gaussian residual and linear-response noise does not predict them. Its
  deductive average error is 12.9% with the default quadrature, and 13.2-13.3% with a denser grid or a Monte Carlo
  integral (true 4.05%). Its per-input predictions have R^2 = -0.96 against the network's per-input error
  probabilities. The Gaussian residual, the same in each cell, spreads the logits across boundaries that the real
  residual respects. Prediction C2's T16 clause (>= 2x off): hit.
- The default quadrature is 0.5 points low. That centering error also biases the column-(b) control variate below by
  beta x 0.005 (review 2). It does not change any conclusion.
- ARC's bar with this description (T16 family; the advice is fitted on the base model only):
  - Column (a) is deductive. It takes the base model's description of h_9, recomputes the last step with each variant's
    weights (statistical linearisation of the ReLU, exact noise, every competitor), and integrates the error over the
    algorithmic cells by quadrature. MSE 2.4e-3 (bias +4.0 points) at 1.3e10 FLOPs. That is worse than the oracle
    constant (9.6e-4) and about 6 x 10^4 times worse than Monte Carlo at equal FLOPs.
  - This last-step version is already 11.0% on the base model (true 4.05%; the step-n description gives 12.9%). So
    the bias does not come from the copies being unseen. It is the description's own error on the base model: the
    per-cell Gaussian residual spreads the logits across the decision boundary (above).
  - Column (b) uses the description's per-input prediction as a control variate, with its mean computed deductively
    (2.4e9 FLOPs, once). It gives MSE 5.2e-5 / 1.8e-5 / 5.7e-6 at 1,024 / 4,096 / 16,384 sequences, which is no lower
    per sample than Monte Carlo's 6.3e-5 / 1.6e-5 / 5.1e-6. The prediction barely correlates with the errors, so the
    deductive setup cost buys nothing. Prediction C3 for T16: hit.

## 4. ARC's bar: estimating the noisy model's error rate

Each family is the base model plus weight-perturbed copies. Every copy has r recalibrated, and its truth comes from
2^24 samples (`estim.py`). Each estimator's MSE is averaged over 16 models x 8 repetitions.

**What the estimators may use (Amendment 5; review 2).**
- A noisy model is its weights plus its frozen noise scales r. Each weight-perturbed copy is constructed with its own
  r recalibrated, as PREREG Part 2 specifies, and r is handed to every estimator as part of the model.
- No estimator in column (a) samples otherwise.
- Advice is fitted once, on the base model only:
  - the dp-targeted edge set and its mean biases, about 2.4e10 FLOPs to find;
  - the algorithm-indexed CL description, about 4e9 FLOPs for T16 and 1.2e10 for W64.
- For comparison, training costs 1.3e12 FLOPs for T16's fine-tune (2e13 for ARC's original training) and 3e14 for a
  width-64 model.

FLOPs are approximate analytic counts (review 1).
- Per sampled sequence we count matmul and elementwise operations: 6,283 for M_{16,10} and 188 for M_{4,3}.
- Not charged: random-number generation (20 FLOPs per Gaussian draw would add 3,400 and 300), the stratified sampler's
  sorting and permutation, and Sobol's setup and inverse CDF. This favours the samplers.
- The ADF's counter covers truncations, linear maps, merges and the final orthant step; it omits a few small mean
  transforms. None of this changes an ordering.

**T16 family.** Relative weight perturbation 0.3%, chosen once so that the error rates spread by a few points: 4.1% to
15.8%, mean 7.5%, s.d. 2.9 points.

Three columns (Amendment 5; `bar3.py`, `estim.py`); the table gives MSE by FLOPs:
- (a) is deductive.
- (b) adds a little sampling to the deductive description: Neyman allocation over the 90 cells from the ADF's per-cell
  predictions, or the ADF's per-cell prediction as a control variate. The ADF's cost (4e7 FLOPs) is included.
- (c) is plain Monte Carlo.
- Stratified (proportional) sampling and RQMC are sampling-only references.

| FLOPs | (a) deductive ADF | (b) Neyman from ADF / control variate from ADF | (c) Monte Carlo | stratified (90 cells) / RQMC |
|---|---|---|---|---|
| 6.4e6 | | | 6.3e-5 | 7.9e-5 / 4.5e-5 |
| 2.6e7 | | | 1.6e-5 | 1.7e-5 / 1.1e-5 |
| 4.0e7 | 0.451 (k = 0; bias +67 points) | | | |
| 4.6e7 | | 8.2e-5 / 5.8e-5 | | |
| 6.6e7 | | 2.1e-5 / 1.5e-5 | | |
| 1.0e8 | | | 5.1e-6 | 3.5e-6 / 2.6e-6 |
| 1.35e8 | 0.441 (k = 2) | | | |
| 1.4e8 | | 5.0e-6 / 4.0e-6 | | |
| 4.1e8 | | | 1.2e-6 | 8.5e-7 / 6.7e-7 |
| 4.5e8 | | 1.25e-6 / 1.05e-6 | | |
| 5.1e8 | 0.433 (k = 4) | | | |

- For scale, a constant estimate (the leave-one-out family mean, an oracle that uses the other models' truths) has
  MSE 9.6e-4.
- Column (a) is about 10^5 times worse than Monte Carlo at equal FLOPs, and 450 times worse than the constant.
- Column (b) is worse than Monte Carlo at equal FLOPs. At the smallest budget the 4e7-FLOP deductive setup dominates
  and it is 6-9x worse; at the largest it is about equal (review 2 corrected a per-sequence comparison).
- Per sequence it gains at most 1.2x. The ADF's per-cell predictions are too poor to allocate samples well or to
  correlate with the errors, and the per-cell error rates are flat anyway (1.6-10%).
- With the accuracy-targeted structure (the dp surrogate at N_ref 2^14, 2,139 edges; the advice found once on the
  base model, with base-model mean biases; `bar3.py surrogate_only`):
  - in column (a), the ADF run on the surrogate has MSE 0.452 and bias +67 points, unchanged from the full network
    (prediction L4: hit);
  - in column (b), a two-level estimator (N surrogate samples + N/4 coupled network-minus-surrogate samples) has MSE
    1.25e-4 at 8.2e6 FLOPs and 6.4e-6 at 1.3e8. It is 1.3-2.5x worse than Monte Carlo at equal FLOPs, because the
    surrogate is only 18% cheaper per sequence (5,121 against 6,283 FLOPs). This is not the plug-in estimator that
    prediction L3 names (section 7).
- Stratification over the algorithmic cells gains 1.4x from 16,384 sequences on. RQMC gains 1.4x at 1,024-4,096
  sequences and 1.8-2.0x at 16,384-65,536, the best of the samplers. Both are hybrids: they sample, so they do not count
  toward ARC's bar.
- The stratified sampler draws 90 floor(N/90) sequences (for example 4,050 for N = 4,096) but is charged for N.

**Wide noisy models** (2^19-step checkpoints; one seed each).
- The weight perturbation is calibrated once per model so that the copies' error rates spread by about a point, as for
  T16: 0.1% relative for W32 and 0.05% for W64. The noise-trained wide models are far more sensitive to weight noise
  than to activation noise. Their recurrent matrices have spectral norm 22 (W32) and 44 (W64), against 5 for T16.
- FLOPs per sequence: 22,763 (W32), 86,443 (W64).
- Truths come from 2^22 samples (2^21 for W64 at sigma = 0.01), not the 2^24 of PREREG Part 2, for CPU time. Their
  sampling error (s.e. 1-2e-4) is below the MSEs being compared.

| model (copies' error rates) | (a) deductive ADF | (a) algorithm-indexed description, last step with each copy's weights | (b) its control variate, 16,384 sequences | (c) Monte Carlo, 16,384 sequences | RQMC, 16,384 |
|---|---|---|---|---|---|
| W32, sigma 0.01 (3.1-9.8%) | MSE 0.343 (bias +58.5 points), 2.5e8 FLOPs | MSE 3.0e-4 (bias +1.5 points), 2.0e10 FLOPs | 4.4e-6 at 3.2e9 FLOPs | 3.95e-6 at 3.7e8 FLOPs | 3.0e-6 |
| W64, sigma 0.03 (5.0-12.5%) | MSE 0.661 (bias +81 points), 1.7e9 FLOPs | MSE 4.5e-3 (bias +6.6 points), 5.0e10 FLOPs | 4.3e-6 at 4.4e9 FLOPs | 5.35e-6 at 1.4e9 FLOPs | 4.6e-6 |
| W64, sigma 0.01 (2.1-5.4%) | MSE 0.760 (bias +87 points), 1.7e9 FLOPs | MSE 1.0e-4 (bias +0.8 points), 5.0e10 FLOPs | 2.2e-6 at 4.4e9 FLOPs | 2.39e-6 at 1.4e9 FLOPs | 1.65e-6 |

- Column (a) loses on every wide model too: by 10^4-10^5 for the ADF. The algorithm-indexed description loses by about
  4 x 10^3 (W32), 3 x 10^4 (W64 at sigma = 0.03) and 1.7 x 10^3 (W64 at sigma = 0.01): Monte Carlo at its 2.0e10
  and 5.0e10 FLOPs would have MSE about 7e-8, 1.4e-7 and 6e-8 (mean p(1-p) x FLOPs per sequence / budget). It is
  also worse than the oracle constant on every wide model (2.4e-4, 2.9e-4 and 8.4e-5).
- Its bias on the copies (+1.5, +6.6 and +0.8 points) is no larger than its bias on the base model, where the
  last-step version gives 6.1%, 13.6% and 3.8% against true 3.1%, 5.0% and 2.07%. On W64 at sigma = 0.03 the
  linearised last step adds 9 points to the step-n description's 4.45%. Correlation with the copies' truths: 0.90,
  0.86 and 0.84.
- Column (b) has 0.8-1.1x Monte Carlo's MSE at the same number of sequences (the MSEs themselves have about 18%
  sampling error) and pays the 2.4-2.5e9-FLOP deductive setup on top. At equal FLOPs (each Monte Carlo row scaled by
  its FLOPs) it is 2.5-32x worse on W64 and 10-85x worse on W32.
- Each family's 16 models include the base model itself (copy 0), on which the advice was fitted. Without it the
  description's MSEs are 4-14% lower (W32 2.6e-4, W64 4.3e-3 and 8.9e-5) and the oracle constant's 13-34% lower
  (1.6e-4, 2.0e-4 and 7.3e-5). No comparison changes.
- Prediction W5: hit. C3 on W32 and on W64 at sigma = 0.01: hit. One seed per configuration (caveats).

Splitting and importance sampling were not needed: the error rates here are 2-15%, not rare events.

**M_{4,3} (fine-tuned at sigma = 0.001; it fails the criterion and is analysed as a check).** Relative weight
perturbation 5%, giving error rates of 1.5% to 40%. MSE:
- Monte Carlo: 1.7e-4 at 1.9e5 FLOPs, 1.1e-5 at 3.1e6.
- Stratified (6 cells): 1.0-1.4x better.
- RQMC: 2.6x better at 1.9e5 FLOPs and 10x at 1.2e7. The input is only 3-dimensional.
- Deductive ADF: MSE 0.022 (bias +14 points) at 3.6e5 FLOPs, about 200x worse than Monte Carlo and 2x worse than the
  constant (0.011).

Hilton's notes report that ARC's interpretation-based estimators for noiseless M_{4,3} reach about 3% RMS error at
about 71,000 operations, roughly 2x worse than sampling. Our generic chain is far behind that.

**M_{2,2} (fine-tuned at sigma = 0.001; fails the criterion; analysed as a check).** ARC's region analysis extends to
the noisy model in closed form (`exact22.py`):
- On each linear piece of the input plane the margin is rho (A cos theta + B sin theta). The noise adds a Gaussian
  term of fixed s.d. s: the step-2 noise enters linearly, and the step-1 noise is propagated through the step-2 gates
  of the noiseless path. The result is exact for that fixed-gate approximation, not for the noisy network.
- The radial integral of Phi(-rho m / s) against the Rayleigh density is (1 - c / sqrt(1 + c^2)) / 2, with c = m/s.
- Its angular integral over a piece is elementary (an arcsine).
- Pieces are bounded by gate rays and the line x_0 = x_1.

Accuracy and cost:
- On the fine-tuned M_{2,2}, the closed form agrees with 2^24-sample truths within 1-2 SE at sigma = 0.001-0.01. At
  sigma = 0.03-0.1 it is 1-2% low, the linearisation's gate-flip error.
- It costs 1,432 FLOPs.
- Over 16 copies with weight perturbation N(0, 0.05^2) (Hilton's protocol; error rates 0.03%-1.1%), its MSE against
  2^26-sample truths is 8.0e-11. The truths' own sampling variance is 6.7e-11, so the closed form's error cannot be
  told apart from zero.
- Monte Carlo at 1,432 FLOPs (30 sequences) has MSE 1.5e-4. Matching the closed form would take 5.7e7 sequences.

This is ARC's result for M_{2,2} with noise added. The description works because every input variable is carried
exactly and the noise needs only one linear-response term. Its fixed-gate approximation agrees with sampling to 1-2 SE
at sigma <= 0.01 (`results/exact22_check.json`).

**A certified one-sided bound (M_{2,2}).** The linear-response computation is exact unless the step-1 noise flips a
step-2 gate. So |p - p_lin| <= P(some step-2 gate flips), and that probability has the same kind of closed form
(`exact22.flip_bound`, `results/exact22_certified.json`).
- At sigma = 0.001 this certifies p <= 8.7e-4 at 2,404 FLOPs, against a truth of 3.3e-4.
- Sampling at equal FLOPs (about 50 sequences, all likely correct) certifies only p <= 0.06 at 95% (Clopper-Pearson).
- The lower bound is vacuous. The union bound counts every gate flip, including the many that do not move the output.
- This is the program's intended use (a one-sided certificate on a rare event) working on a 10-parameter model.

**M_{2,2} in the three columns of Amendment 5** (`bar22.py`; 16 copies x 64 repetitions; truths from 2^26 samples,
whose own noise contributes 6.7e-11 to every MSE):

| FLOPs | (a) closed form | (b) closed form as control variate + N coupled samples | (c) Monte Carlo |
|---|---|---|---|
| 1.4e3 | 8.0e-11 | | |
| 3.0e3 | | | 6.4e-5 |
| 2.5e4 | | 8.0e-11 (N = 256) | 1.69e-5 at 1.2e4 |
| 3.9e5 | | 2.6e-10 (N = 4,096) | 1.18e-6 at 1.9e5 |
| 1.5e6 | | 1.1e-10 (N = 16,384) | 2.95e-7 at 7.7e5 |

- Column (a) is already at the truth's noise floor.
- Column (b) is unbiased but adds variance from the rare samples where a step-2 gate flips. It cannot improve on (a)
  until the truths are about 100x more precise.
- At equal FLOPs (a) beats (c) by about 10^6, and (b) beats it by 1,200-2,100x at its two largest budgets (review 2
  corrected an overstatement here). Prediction B3 (>= 100x): hit.

## 5. From-scratch and wider noisy networks

### 5.1 Training and the criterion

**d = 16 from scratch, CPU, 2^28 sequences (a quarter of ARC's length).** The shortened schedule is too short even
without noise, so these runs say nothing about noise:
- the sigma = 0 controls reach loss 0.376 (+127% against ARC's same seed, 0.165) for seed 2 and 0.329 (+24%) for
  seed 0;
- at sigma = 0.003, seeds 0-4 reach 0.31-0.52, between +7% and +114% against ARC's same-seed models; none passes;
- seed 2 at sigma = 0.003 (0.312) does better than the same seed without noise (0.376).

The full-length runs on the GPU replace these.

**GPU run azn-wide-3: ARC's recipe at the full 2^30 sequences.**
- Widths 16 (sigma 0, 0.003, 0.01; seeds 0-1), 32 and 64 (sigma 0.01, 0.03; seeds 0-2), on one on-demand L4
  (`GPU-LOG.md`).
- Training is batched across models (`train_batched.py`).
- Criterion as in Amendment 1, against ARC's example (loss 0.147, accuracy 95.3%) and against zoo seed 2 (0.165,
  94.9%). Each model is judged on its own.

*Criterion at the 2^19-step checkpoint (half of ARC's length, after the learning-rate decay). Evaluated on CPU on
2^20 paired sequences, not 2^22 (`eval_ck.py`). The final 2^20-step evaluation is in the next table:*

| width d | sigma (bits per neuron) | seed: loss, noise on (gap vs example) | accuracy | meets vs example / vs zoo seed 2 |
|---|---|---|---|---|
| 16 | 0 | s0: 0.127 (-14%); s1: 0.211 (+43%) | 0.960; 0.930 | s0 yes / yes; s1 no / no |
| 16 | 0.003 (8.4) | s0: 0.183 (+25%); s1: 0.366 (+150%) | 0.935; 0.861 | no / no |
| 16 | 0.01 (6.6) | s0: 0.294; s1: 0.512 | 0.886; 0.802 | no / no |
| 32 | 0.01 (6.6) | s0: 0.079 (-47%); s1: 0.093 (-37%); s2: 0.095 (-36%) | 0.969; 0.964; 0.963 | yes / yes (all three) |
| 32 | 0.03 (5.1) | 0.199-0.215 (+35% to +46%) | 0.916-0.922 | no / no |
| 64 | 0.01 (6.6) | 0.050-0.060 (-59% to -66%) | 0.975-0.979 | yes / yes (all three) |
| 64 | 0.03 (5.1) | 0.124-0.130 (-11% to -16%) | 0.948-0.951 | no / yes (all three): accuracy 0.2-0.5 points below the example |

**Final evaluation, 2^20 steps** (on the GPU, 2^22 paired sequences, fixed-point r; `results/gpu_table.json`):

| width d | sigma (bits per neuron) | loss with noise on, per seed (gap vs example) | accuracy per seed | meets vs example / vs zoo seed 2 |
|---|---|---|---|---|
| 16 | 0 | 0.120 (-19%); 0.205 (+39%) | 0.962; 0.931 | yes / yes; no / no |
| 16 | 0.003 (8.4) | 0.185 (+27%); 0.368 (+151%) | 0.933; 0.860 | no / no; no / no |
| 16 | 0.01 (6.6) | 0.298 (+104%); 0.528 (+261%) | 0.884; 0.795 | no / no; no / no |
| 32 | 0.01 (6.6) | 0.082 (-44%); 0.108 (-26%); 0.109 (-25%) | 0.968; 0.959; 0.958 | yes / yes (all three) |
| 32 | 0.03 (5.1) | 0.225 (+54%); 0.224 (+53%); 0.209 (+43%) | 0.912; 0.914; 0.919 | no / no (all three) |
| 64 | 0.01 (6.6) | 0.048 (-67%); 0.053 (-64%); 0.058 (-61%) | 0.980; 0.979; 0.976 | yes / yes (all three) |
| 64 | 0.03 (5.1) | 0.135 (-8%); 0.124 (-16%); 0.129 (-12%) | 0.946; 0.951; 0.949 | no / no; no / yes; no / yes |

- Against the example, sigma* is 0.001 at d = 16 (fine-tune only), 0.01 at d = 32 and 0.01 at d = 64.
- At d = 64 and sigma = 0.03 the loss is 8-16% lower than the example's, but accuracy is 0.23-0.70 points lower. Two
  seeds pass against zoo seed 2.
- The battery in section 5.2 used the 2^19-step checkpoints of W32 seed 0 and W64 seed 1. Their final losses differ by
  less than 4% (W32: 0.079 against 0.082; W64: 0.1235 against 0.1237).

What this says about noise and width:
- At ARC's width, noise costs too much even from scratch at full length. The same seed reaches 0.120 without noise and
  0.185 at sigma = 0.003 (final).
- Our noiseless d = 16 seed 0 beats ARC's example (0.120 against 0.147), so the recipe is not the obstacle.
- With 2x and 4x width, 6.6 bits per neuron are enough in every seed against both references.
- With 4x width, 5.1 bits pass against the zoo model in two of three seeds and trail the example's accuracy by
  0.23-0.70 points.
- This is the program's expectation: "a few bits of precision per dimension ... possibly with somewhat wider layers".

### 5.2 Interpretation of the wide noisy models

The same battery was run on the 2^19-step checkpoints of W32 (d = 32, sigma = 0.01, seed 0) and W64 (d = 64,
sigma = 0.03, seed 1). Each is the best seed of its configuration by loss, and each passes against at least one
reference.

The propagated-noise numbers, rerun on the final models, agree with the checkpoints:
- error with own noise: 3.18% (W32) and 5.00% (W64), against 3.10% and 5.01% at the checkpoints;
- median margin: 8.35 and 6.52 noise s.d., against 9.0 and 6.7;
- neurons carrying 80% of the decision noise: 12 and 28, against 11 and 26.

Width 64 at sigma = 0.01 (seed 1, checkpoint) got the DAG and algorithm-relative analyses only.

Seed check on the final models (`results/dag_W32s*final.json`, `results/lowrank_W32s1final.json`,
`results/dag_W64s0final.json`):
- W32 at sigma = 0.01: 75-80% of the 9,568 edges. Seed 1's final model meets the 1% held-out target with 76.0%
  (0.99%). Seed 0's final model misses it narrowly with 73.3% (1.04%); its smallest saved size that meets it is 80.0%
  (0.55%), and its checkpoint met it with 75.2%. Seed 1 needs at most 22 of 32 directions (rank 22 gives 0.89% and
  rank 20 gives 1.54%; rank 21 was not run); seed 0 needed 22.
- W64 at sigma = 0.01: seed 1's checkpoint meets the target with 43.1% of the 37,568 edges. Seed 0's final model
  misses it narrowly with 49.8% (1.05%) and meets it with 58.5% (0.49%), its smallest saved size that does. So the
  width-64 figure is 43% in one seed and 58% in the other.

| quantity | T16 (d = 16, sigma 0.001) | W32 (d = 32, sigma 0.01) | W64 (d = 64, sigma 0.03) |
|---|---|---|---|
| error rate with its own noise (noiseless) | 4.07% (3.78%) | 3.10% (1.89%) | 5.01% (2.63%) |
| median margin in its own noise s.d. | 14.5 | 9.0 | 6.7 |
| error rate under 1% test noise | 17.4% | 3.1% | 3.0% |
| neurons carrying 80% of the decision noise (top neuron's share) | 5 (39%) | 11 (17%) | 26 (6.7%) |
| directions the question needs at flip <= 1% (same rank every step; principal components) | 13 of 16 (16) | 22 of 32 (32) | 25-32 of 64 (56) |
| DAG edges, unconditional means: smallest tested size with held-out flip <= 1% | 85% | 75% | 60% (57% gives 1.04%) |
| DAG edges, algorithm-relative means: smallest tested size with held-out flip <= 1% | 58% | 60% (54% gives 1.05%) | 50% (41% gives 1.03%) |
| error rate of the algorithm-mean readout | 1.0% | 0.2% | 0.06% |
| fitted families' predicted error rate, CL / CLX (true rate in the first row) | 13.2% / 10.6% | 7.5% / 3.0% | 4.7% / 3.2% |
| deductive ADF (k = 0), against the true rate above | 75% | 65% | 90% (step-131k checkpoint) |
| algorithm-indexed Gaussian description, deductive average (base-model advice; default grid, with the dense-grid or Monte Carlo integral in brackets where checked) | 12.9% (13.2-13.3%) | 7.1% | 4.45% (4.55-4.58%) |
| scenario hits (S1-S8, V1) | 7 of 9 | 7 of 9 | 5 of 9 |
| error rate with a negative 2nd max (S1) / all inputs negative (S2) | 76% / 88% | 29% / 73% | 16% / 32% |
| description size at step n (cells; parameters) against the network | 90; 22,320 against 432 | 90; 67,680 against 1,376 | 90; 227,520 against 4,800 |

What changes with noise and width:
- They are sturdier. The wide noisy models spread the decision-relevant noise over many neurons: W64's top neuron
  carries 6.7% of it, against 39% for T16's neuron 7. Under 1% test noise they keep their accuracy, where T16 falls to
  83%.
- They are sparser, more so at larger width and lower noise.
  - The fraction of edges the question needs at 1% held-out flips falls with width: 85% (T16), 75-80% (W32, two
    seeds), 43-58% (W64 at sigma = 0.01, two seeds), 60% (W64 at sigma = 0.03).
  - The fraction of directions falls from 81% to 39-50%, but the absolute numbers grow (13, 22, 25-32 directions).
  - Relative to the algorithm the fractions are 58%, 60%, 30% and 50%.
  - Review 2 corrected three sizes that missed their held-out targets by a few hundredths of a point.
- They are more describable, mostly by fitted descriptions.
  - On W32 the input-exact family CLX, fitted to the real activations, predicts the error rate within 3% (3.05%
    against 3.14%); on T16 it is off by 2.6x. That description has 90 cells and about 108k parameters, against the
    network's 1,376.
  - On W64 at sigma = 0.03, the family restricted to the algorithmic variables (CL) is within 6% (4.7% against
    5.0%).
  - Its deductive version (base-model advice, quadrature over the exact law of the algorithmic variables) gives 4.45%
    with the default grid against the true 4.99%.
    - A denser grid gives 4.55%, and a Monte Carlo integral of the same description gives 4.58% +- 0.02%. So the
      description itself is 8-9% low, and the default grid adds 0.1 points (review 2, `results/algrel_quad_*.json`).
    - This is the closest deductive estimate in this work, and it comes from the noisiest model.
  - That number is for the base model, on which the advice was fitted. ARC's rules allow that, but the bar is scored
    on weight-perturbed copies the advice never saw.
    - Copies need the last step recomputed with each copy's weights. That version already gives 13.6% on the base
      model (true 4.99%): the linearised last step adds 9 points.
    - On the family's 16 models it tracks the truths (correlation 0.86) but sits 6.6 points high (MSE 4.5e-3,
      against 2.9e-4 for the oracle constant).
    - For W64 at sigma = 0.01 it sits 0.8 points high on the copies (MSE 1.0e-4, oracle constant 8.4e-5) and 1.7
      points high on the base model (3.8% against 2.07%).
  - The deductive ADF misses the wide models by 21x (W32), 18x (W64 at sigma = 0.03) and 44x (W64 at sigma = 0.01).
    Its Gaussian closure fails at the decision thresholds.
- Width 64 at sigma = 0.01 is the model Amendment 6 named (2^19-step checkpoint of seed 1; true error rate 2.07%).
  - Relative to the algorithm it is sparse: 17% of the ablatable edges keep held-out flips at 1.8%, and 29% keep
    them at 0.83%. Counting the always-kept input edges, that is 30.4% of all 37,568 edges. Prediction C1's wide
    clause (<= 30% of the same edge set) is therefore a near miss (review 3).
  - With unconditional means it needs 43% of edges (16,179 of 37,568) for held-out flips of 0.99%.
  - The algorithm-mean readout errs 0.06%.
  - The algorithm-indexed Gaussian description's deductive average is 3.20% (+55%; C2 within 25%: miss).
- The error mechanism is shared with T16.
  - Errors concentrate on 2nd/3rd near-ties (S4: 20% error rate in W32, 29% in W64).
  - Errors are decided at new-max or new-2nd-max steps at about twice chance (W32: 23.5% and 27% of decided errors,
    against 11.2% each by chance).
  - The negative-2nd-max blind spot is much smaller: 29% error in W32 and 16% in W64 at sigma = 0.03, against 76%.


## 6. Comparison with ARC's interpretation

- M_{2,2}: as in ARC's analysis, the margin is a fixed linear form on each linear piece, and the errors come from the
  pieces where the two coefficients are nearly equal, next to the line x_0 = x_1. With noise this becomes a closed
  form whose cost does not grow with sigma. The error rate rises from 7.7e-5 without noise to about 0.32 sigma,
  because the noise smears the boundary.
- M_{16,10}: we find ARC's leave-one-out-max neurons in three independent ways:
  - they are the neurons the algorithmic state describes (R^2 >= 0.99, with neuron 8 added);
  - they carry 87-89% of the decision-relevant noise;
  - the question-driven DAG keeps 96% of their internal edges.

  We do not get beyond ARC's own obstacle. The other ten neurons are dense, the question needs 13 of 16 directions, and
  no tractable family turns the partial understanding into an error estimate. ARC: "we have not turned this
  understanding into an adequate mechanistic estimate".
- What noise adds at the precision ARC's 16-neuron models tolerate:
  - It adds a calibrated way to ask how much precision each part needs (the propagated-noise metric), and a measure of
    when errors are decided (backward functions).
  - It does not make the network simpler or the estimate easier: sigma = 0.001 leaves the polyhedral structure of the
    noiseless map intact.
- What changes with spare width (section 5.2):
  - Noise-trained networks with 2-4x the neurons tolerate 6.6 bits per neuron and stay as good as ARC's example in
    every seed. At 5.1 bits, width 64 matches the zoo model in two of three seeds but not the example. They are
    sturdier.
  - Their decision noise is spread over many neurons, and their structure relative to the task's own variables is
    somewhat sparser.
  - Fitted Gaussian-residual descriptions indexed by the algorithmic state predict their error rate to within 3-6%
    (CLX on W32, CL on W64).
  - Deductive versions of those descriptions are mixed:
    - the ADF misses by 18-44x;
    - the algorithm-indexed description with base-model advice gives 0.89-0.92x (W64 at sigma = 0.03, default or
      dense quadrature), 1.55x (W64 at sigma = 0.01) and 2.3x (W32) the true error rate;
    - the version that runs on weight-perturbed copies (last step recomputed with each copy's weights) is 0.8-6.6
      points high on them on average, and 1.8-8.6 points high on the base models themselves.
    None comes near the 3-5% relative accuracy that matching sampling needs.

**What the results point to (not tested).**
- The candidate for a deductive estimator is a conditional-linear description indexed by the algorithmic state and
  propagated by exact quadrature over the algorithmic variables: a regression closure in place of moment matching
  through the ReLUs. Its fitted versions have nearly the needed capacity on the wide noisy models. Whether a deductive
  version keeps that accuracy is open.
- Relative to the algorithm, width 64 at sigma = 0.01 confines the answer-changing deviations to about 30% of edges. A
  description of those deviations alone may be cheaper than a description of the whole state.
- ARC's bar asks for 3-5% relative accuracy on these error rates at about 10^8 FLOPs. The closest deductive number here
  is 8-9% low, with advice fitted on the same model; the others are 50% to 44x off.

## 7. Predictions (PREREG.md) against results

Amendment 7's predictions are scored in section 9.5.

| # | prediction (abridged) | result |
|---|---|---|
| P1 (Part 1, superseded) | under the 1-point criterion, M_{16,10} passes at sigma 0.03, fails at 0.1; the M_{2,2} error ratio is > 10 at every grid sigma | miss: it fails at 0.03 (-21.9 points) and 0.01 (-7.2), passes only at <= 0.003 (-0.82). M_{2,2} part hit (ratio 40 at 0.01) |
| P1' (Amendment 1) | example and seed 2 pass at 0.01 or 0.03; M_{4,3} only at <= 0.003; M_{2,2} at none | miss: the example passes only at 0.001, seed 2 at none, M_{4,3} at none (+2.5% at 0.001). M_{2,2} part hit |
| D1 | flip-DAG at held-out flip <= 1% keeps <= 25% of edges | miss: 85% (2,106 of 2,480) |
| D2 | ARC's neurons used at every step >= 3, and neuron 2 has the largest out-degree | half: used (out-degree >= 13 at every step >= 3); largest out-degree is neuron 8 (144) and neuron 1 (143), neuron 2 has 123 |
| D3 | the forward ranking needs >= 3x the edges at equal flip; the question DAG's path KL >= 1,000 nats | half: the forward ranking needs at most 1.2x (6.1% flip at 2,048 edges, against 1.2%); path KL 5e6 nats |
| D4 | >= 3 neurons dropped at steps <= 3 | half: 10 neurons dropped at step 1, none at steps 2-3 |
| V1 | R^2 of the DAG's per-input error probability >= 0.75 | hit: 0.81 (noise floor 0.985) |
| S1-S4 | the error rate is >= 5p with a negative 2nd max, >= 0.5 when all inputs are negative, and >= 2p in the top-two and the 2nd/3rd near-tie deciles | hit, all four (18.7p, 0.88, 2.9p, 5.2p) |
| S5 | a max after the 2nd max: >= 1.3x | miss: 1.26x |
| S6 | the last position is easier, <= 0.5p | miss: 1.25p |
| S7 | noise-sensitive inputs >= 3x more common near ties | hit: 19.0% against 4.4% (scored without the CI rule, as for S8; review 1) |
| S8 | the DAG's scenario rates are within 25% of the network's | hit: all six |
| E1 | the deductive ADF bias is >= 1 point and it loses to Monte Carlo above 10^4 samples; the M_{2,2} description wins >= 100x | hit: bias +66 points, about 10^5x worse; M_{2,2} about 10^6x better |
| E2 | stratified <= 1.5x, RQMC <= 2x over Monte Carlo (T16) | hit: 1.4x and 1.4-2.0x |
| E3 | no non-vacuous certificate for T16; certified intervals only for M_{2,2} | hit for T16 (none attempted beyond the fitted families: their Gaussian-residual costs are 66+ bits per step); for M_{2,2} only a one-sided upper bound (8.7e-4 against a truth of 3.3e-4), not an interval |
| K1 | the DAG needs >= 80% of edges, so it is reported as a blocker | fired (85%) |
| K2 | every deductive estimate is >= 10x worse than sampling on M_{4,3} and T16, so ARC's bar is failed with its cause | fired (200x and 10^5x; cause: Gaussian closure across the decision thresholds, section 3.2) |
| L1 (dp primary) | the average-targeted DAG keeps <= 50% at N_ref 2^10, <= 75% at 2^14, >= 80% at 2^18 | 2/3 miss: 82% and 86%; the 2^18 clause holds (92%) |
| L2 | the average question needs >= 11 of 16 directions at 2^14 | hit for the same rank at every step (13); the per-step version was run only for the flip question (9-13 from step 4) |
| L3 | the plug-in estimator (sampling the surrogate, with its bias) gains at most 2x over Monte Carlo | not run as registered (review 2): we ran an unbiased two-level variant (it loses 1.3-2.5x). From the measured numbers the plug-in gains at most 1.23x per FLOP (the surrogate is 18% cheaper) and adds a squared bias of 1.2e-6, so it cannot gain 2x at any budget |
| L4 | the ADF on the surrogate is no closer (bias >= 30 points) | hit (+67 points) |
| B1 | column (a) loses to (c) at every budget, bias >= 10 points (T16 and every wide model) | split: "loses" hit everywhere tested (one seed per wide configuration); "bias >= 10 points" hit for the ADF (58-87 points), miss for the algorithm-indexed description (T16 4.0, W32 1.5, W64 0.8 and 6.6 points) |
| B2 | the best column-(b) estimator gains <= 2x over (c) on T16 at >= 10^7 FLOPs | hit (it gains nothing) |
| B3 | on M_{2,2} the closed form as a control variate beats (c) by >= 100x at >= 10^4 FLOPs | hit (1,200-2,100x at the largest budgets; column (a) about 10^6x) |
| W1 | some d in {32, 64} passes against the example at sigma 0.01; none at 0.03 | hit (final: d = 32 and 64 pass at 0.01 in all seeds; at 0.03, d = 32 fails and d = 64 misses the example's accuracy) |
| W2 | a passing wide model's flip-DAG keeps <= 50% of edges | mixed: W64 at sigma 0.01 hit for seed 1 (43%), miss for seed 0 (58%; 49.8% gives 1.05%); W32 miss (75-80%, two seeds); W64 at sigma 0.03 (passes only against zoo seed 2) 60% |
| W3 | it needs <= d/2 directions per step | half: W32 22 of 32 (miss; seed 1: rank 20 fails, 21 not run); W64 25-32 of 64 (hit at the grid's resolution, but measured on the sigma = 0.03 model, which passes only against zoo seed 2) |
| W4 | the deductive ADF is still off by >= 5x | hit: 65% against 3.1% (W32), 90% against 6% (W64, step 131k) |
| W5 | no deductive estimate beats Monte Carlo on it | hit for the tested seed of each (ADF on all three wide models; the algorithm-indexed description on all three, by 1.7 x 10^3 to 3 x 10^4) |
| C1 | algorithm-relative ablation keeps <= 40% (T16) and <= 30% (wide) of edges at flip <= 1% | miss: T16 58%, W32 60%, W64 at sigma 0.01 30.4% of all edges (29.2% of the ablatable ones), a near miss; W64 at sigma 0.03 (not registered) 50% |
| C2 | the algorithm-indexed description is within 25% on the wide models, >= 2x off on T16 | T16 clause hit (12.9% against 4.05%); wide clause miss: W32 7.1% against 3.1%, W64 at sigma 0.01 3.2% against 2.07%. W64 at sigma 0.03 (not registered): 4.45-4.58% against 4.99%, with advice fitted on the same model |
| C3 | column (a) loses at every budget >= 10^6 FLOPs; (b) gains <= 3x | hit on T16, W32 and W64 at sigma 0.01, one seed each ((a) loses by 1.7 x 10^3 there; (b) has 0.88-0.93x Monte Carlo's MSE per sequence and is 2.8-28x worse at equal FLOPs) |
| C4 | description size 90 cells, 90 (7d + d(d+1)/2) parameters at step n | as computed: 22,320 (T16), 67,680 (W32) and 227,520 (W64) at step n; 82,088, 248,912 and 836,768 over all steps |

## 8. Caveats

- One model carries the main interpretation (T16). The other fine-tunes fail the criterion. The wider and from-scratch
  models are in section 5.
- sigma* = 0.001 is so small that T16 is ARC's network with negligible noise (section 2). The program's claims about
  "a few bits per dimension" are not tested on T16; the wide models test them.
- The DAG and subspace searches are heuristic: mean ablation, a first-order saliency ranking, a binary search on a
  non-monotone curve, and one greedy pass. A better search, another ablation (resampling, conditional means) or another
  basis could find sparser surrogates. The per-step ranks rule out much below 9-13 directions at the late steps only
  for this family of linear projections.
- The families in section 3.1 are least-squares fits; they are not optimised for predicting the error rate.
  "Unexplained bits" is a Gaussian-channel indicator, not a proven description cost.
- The backward-information numbers are lower bounds from binning. The algorithmic-state version uses a coarse,
  cross-fitted estimator.
- FLOPs are approximate analytic counts. The comparisons that matter are orders of magnitude (10^5x, 200x, 10^6x), and
  they survive any reasonable recount. The small sampler gains (1.4-2x) do not count toward ARC's bar anyway.
- One noise seed per run; the T16 family uses 16 weight-perturbed copies (ARC's protocol), not retrained seeds.
- Amendment 5 asked for the structure and the family to be analysed per seed. For the wide models we analysed one
  seed per configuration (the best by loss: W32 seed 0, W64 seed 1 at each sigma), for lack of CPU time. Only the
  flip-DAG size and, for W32, the direction count were rechecked on a second seed (section 5.2).
- Section 9 uses one model per width (T16 and the seed-1 checkpoint of W64 at sigma = 0.01). Its systematic shares
  are lower bounds set by one regressor with fixed settings.
- Reproducibility: the T16 flip-DAG sweep was run with an earlier version of `dag_run.py` that used an absolute K grid
  (16, 32, ..., 2,048, 2,480). The current version uses fractions of the edge count. The diagnostics quoted in section
  3.1 are saved in `results/adf_diag.json`, and the M_{2,2} checks in `results/exact22_check.json`.

## 9. Errors given the algorithmic variables, and a deductive estimator with no closure in them (Amendment 7)

*The coordinator asked for this as a bounded next step (CPU only). It was pre-registered as Amendment 7 (05:09Z)
before any run, and reviewed by GPT-6 (round 5). Models: T16 and W64 at sigma = 0.01 (seed 1, 2^19-step
checkpoint), the two models of Amendment 6.*

**Summary of this section.**
1. The errors are mostly systematic in the task's own variables. On held-out inputs, a gradient-boosted fit on the
   top-3 values and positions explains 54% (T16) and 79% (W64) of the input-to-input variance of the error
   probability. The coordinator's six features explain 44% and 76%. The stop rule (30%) did not fire. These shares
   are lower bounds: the same fit on a lossless encoding of the input reaches only 41%.
2. A deductive model with no closure in those variables comes much closer than the earlier ones. TOP3 integrates
   exactly over the law of the top-3 values and positions. At each point it uses only the weights: the network
   linearised at a representative input, plus the linear-response noise term. Integrated exactly, it gives 4.47% on
   T16 (true 4.05%, +10%) and 2.62% on W64 (true 2.07%, +27%). The ADF gives 72-75% and 91%, and the fitted
   algorithm-indexed description 12.9% and 3.20%. So most of the earlier deductive bias came from the closure in the
   algorithmic variables.
3. It still loses ARC's bar, by 64-4,400x at equal FLOPs in column (a).
   - Its quadrature converges slowly, because its error probability is close to a step function of the values. On
     T16, 1.5 million nodes leave it 0.96 points below its own exact integral.
   - Its best point on W64 (MSE 1.2e-6 at 1.5e11 FLOPs, bias -0.03 points) is a cancellation between that quadrature
     error and the model's own +27%. It is not an accurate estimator.
4. With a little sampling (column (b)) its predictions gain 1.1-2.9x per sequence over Monte Carlo, about as much as
   RQMC on T16 and more on W64. At equal FLOPs they only break even: at best 1.08x (T16) and 1.16x (W64), at the
   largest budget and within the MSEs' sampling error.
5. Where the model is wrong. On T16 it predicts 8% error where the 2nd max is negative (true 77%), and its
   predictions are too extreme. On W64 it ranks inputs well (per-input share 0.68, against the fit's 0.79) but
   predicts too many errors away from the near-ties (0.97% against 0.72%). On T16 the misses partly cancel.

### 9.1 Step 1: how much of the error is systematic (`sysid.py`)

Per input, eps(x) is estimated from two independent halves of 16 noise draws; one more draw gives a 0/1 error. A
LightGBM fit (hyperparameters fixed in advance) is trained on half the inputs and scored on a held-out quarter. The
share is the noise-corrected R^2 of the prediction against eps(x). 2^18 inputs for T16 and 2^17 for W64.

| | T16 (error rate 4.0%) | W64, sigma 0.01 (2.1%) |
|---|---|---|
| share, F1: g12, g23, sign of the 2nd max, positions of max and 2nd max, last event step | 0.44 | 0.76 |
| share, F2: top-3 values and positions | 0.54 | 0.79 |
| share, F3: all order statistics and positions (capacity check) | 0.41 | 0.41 |
| Brier skill on one draw, F2 (ceiling set by the noise) | 0.44 (0.80) | 0.31 (0.36) |
| AUC on one draw, F2 | 0.973 | 0.979 |
| largest F2 gains | g23 0.31, g12 0.20, sign 0.15 | g23 0.51, g12 0.39 |
| errors the algorithmic readout does not make: share F1 / F2 | 0.36 / 0.46 | 0.77 / 0.79 |
| inputs where that readout is right; network error rate on them | 98.9%; 3.35% | 99.96%; 2.09% |
| TOP3's per-input prediction (weights only), share and AUC | 0.49, 0.967 | 0.68, 0.975 |

- The last event step equals max(pos1, pos2), so F1 carries five independent features.
- On W64, 64% of the 0/1 variance is noise (Brier ceiling 0.36), and the fit reaches 84% of what is reachable.
- F3 does worse than F2 with the same settings, so the regressor limits all shares. They are lower bounds on what
  the full input, or the top three, determine.

### 9.2 TOP3 (`top3.py`)

- Exact law. The positions of the max, 2nd and 3rd max are uniform over the 720 ordered triples and independent of
  the values. Three uniform coordinates map exactly to the values: the 3rd largest by its inverse CDF, then the
  smaller and larger of two N(0,1) truncated above it. Boxes are equal in the first coordinate and graded toward the
  near-ties in the other two; their probabilities are exact (chi-square test on 4 x 10^5 sampled inputs, p = 0.81).
- Per box, from the weights alone. The representative input has the top three at their positions and the 7 others at
  E[Z | Z < x(3)]. One noiseless pass fixes the ReLU pattern. The three closest logit differences are linearised in
  the 7 lower inputs (variance Var(Z | Z < x(3)) each) and in the noise at every step (variance (sigma r)^2). The
  error probability is one minus the bivariate normal orthant of the two most dangerous competitors times the third's
  normal CDF. Nothing is fitted; each family member uses its own weights and r.
- Cost per node: 26,659 FLOPs (T16) and 349,219 (W64), about 4x one sampled sequence. A budget (KS, KG) has
  720 KS KG^2 nodes. Budgets (6, 10) and (8, 16) were added after the smoke test, to see convergence.

### 9.3 Accuracy on the base models

| | T16 | W64, sigma 0.01 |
|---|---|---|
| true error rate (2^24 and 2^21 samples) | 4.05% | 2.07% |
| TOP3, exact integral of its own model (2^18 / 2^16 sampled inputs; a diagnostic) | 4.47% +- 0.04 (+10%) | 2.62% +- 0.04 (+27%) |
| TOP3 quadrature (4, 6), 103,680 nodes | 3.00% | 2.03% |
| (8, 16), 1.47 million nodes | 3.51% | 2.40% |
| (12, 24), 5.0 million nodes (diagnostic) | 3.89% | 2.50% |
| (16, 32), 11.8 million nodes (diagnostic) | 4.18% | not run |
| ADF; algorithm-indexed description (step n, fitted advice) | 72-75%; 12.9% | 91%; 3.20% |

- By scenario on T16 (TOP3 boxes at (8, 16) against 2^20 samples): near-ties of the 2nd and 3rd value (lowest
  decile of the gap, 9.7% of inputs) 20.4% predicted against 19.7%; the rest 1.6% against 1.5%; negative 2nd max
  (1.1% of inputs) 8.2% against 77%.
- Calibration on T16: boxes predicted below 0.1% (89% of inputs) err 1.0%; boxes predicted above 60% (2.8%) err 58%.
- On W64: negative 2nd max 4.8% against 8.6%; near-ties 15.2% against 13.7%; the rest 0.97% against 0.72%.
  Calibration is close in the middle and too high at the top (70% predicted against 54%).
- Per (pos1, pos2) cell the predictions correlate 0.82 (T16) and 0.77 (W64) with the sampled rates.

### 9.4 ARC's bar

16 models per family (base plus weight-perturbed copies, as in section 4). "Monte Carlo at equal FLOPs" is
mean p(1-p) x FLOPs per sequence / budget.

| TOP3 budget | nodes | T16: FLOPs, (a) MSE, bias | T16: Monte Carlo at equal FLOPs | W64: FLOPs, (a) MSE, bias | W64: Monte Carlo at equal FLOPs |
|---|---|---|---|---|---|
| (1, 2) | 2,880 | 7.7e7, 1.9e-3, -4.3 points | 5.6e-6 (330x) | 1.0e9, 9.0e-4, -2.9 | 2.9e-6 (310x) |
| (2, 2) | 5,760 | 1.5e8, 1.5e-3, -3.8 | 2.8e-6 (530x) | 2.0e9, 7.5e-4, -2.7 | 1.5e-6 (510x) |
| (2, 3) | 12,960 | 3.5e8, 6.6e-4, -2.6 | 1.2e-6 (530x) | 4.5e9, 1.8e-4, -1.3 | 6.5e-7 (270x) |
| (3, 4) | 34,560 | 9.2e8, 3.6e-4, -1.9 | 4.6e-7 (770x) | 1.2e10, 6.3e-5, -0.8 | 2.4e-7 (260x) |
| (4, 6) | 103,680 | 2.8e9, 1.8e-4, -1.3 | 1.5e-7 (1,200x) | 3.6e10, 1.2e-5, -0.3 | 8.1e-8 (150x) |
| (6, 10) | 432,000 | 1.15e10, 8.7e-5, -0.9 | 3.7e-8 (2,300x) | 1.5e11, 1.2e-6, -0.03 | 1.9e-8 (64x) |
| (8, 16) | 1,474,560 | 3.9e10, 4.8e-5, -0.7 | 1.1e-8 (4,400x) | 5.2e11, 2.6e-6, +0.12 | 5.7e-9 (450x) |

- For comparison, column (a) of the earlier descriptions: the ADF has MSE 0.45 (T16) and 0.76 (W64); the
  algorithm-indexed description 2.4e-3 at 1.3e10 FLOPs and 1.0e-4 at 5.0e10; the oracle constant 9.6e-4 and 8.4e-5.
  On T16, TOP3 beats the algorithm-indexed description at every budget and the oracle constant from (2, 3) on; on
  W64 it beats both from (3, 4) on.
- TOP3 tracks the copies (correlation 0.99-1.00 with their truths from budget (2, 3) on). Its bias is common to the
  family, and it is mostly quadrature error.
- On W64 the MSE falls to 1.2e-6 at (6, 10) and rises again at (8, 16), as the quadrature error (negative) shrinks
  below the model's own bias (positive). Converged, its bias on the base model would be +0.55 points.
- Column (b), each estimator's MSE against Monte Carlo's at the same number of sequences and at equal FLOPs (setup
  included; 64 squared errors per point, s.e. 15-20%):
  - (b1) control variate with the box prediction: 1.1-2.3x per sequence on T16 and 1.1-2.9x on W64, higher with
    finer setups. At equal FLOPs at most 1.08x (T16) and 1.06x (W64), with the cheapest setup at N = 2^16. Below
    that the setup cost makes it worse.
  - (b2) stratified over 8 groups by predicted error, setup (2, 3): 1.1-1.7x per sequence on T16 and 2.0-2.1x on
    W64. At equal FLOPs 0.89x (T16) and 1.16x (W64) at N = 2^16.
  - For reference, column (c): RQMC gains 1.2-2.0x (T16) and 0.8-1.45x (W64); stratification over the 90 cells
    gains 0.7-1.45x and 0.5-1.14x.

### 9.5 Predictions (Amendment 7)

| # | prediction (abridged) | result |
|---|---|---|
| Y1 | share (F2) 0.25-0.50 on T16, 0.35-0.65 on W64 | miss, higher on both: 0.54 and 0.79 |
| Y2 | F1 is 0-0.10 below F2 | hit: 0.09 (T16) and 0.03 (W64) below |
| Y3 | excess share 0.10-0.35 on T16; on W64 within 0.05 of the total | T16 miss (0.46); W64 hit (0.79 and 0.79) |
| Y4 | AUC (F2) 0.85-0.93 on T16, 0.90-0.97 on W64 | miss, higher: 0.973 and 0.979 |
| Y5 | F3 >= 0.60 on both | miss (0.41 and 0.41): the shares are lower bounds limited by the regressor |
| E7.1 | at the finest budget within 30% (T16) and 40% (W64) of the base error rate | hit at the finest registered budget, (4, 6): -26% and -2%. Also hit at the added (8, 16) (-13%, +16%) and for the exact integral (+10%, +27%). The (4, 6) and (8, 16) values are partly quadrature error cancelling model bias |
| E7.2 | column (a) loses to Monte Carlo at equal FLOPs at every budget, by >= 10x | hit: 330-4,400x (T16), 64-510x (W64) |
| E7.3 | on T16 (a) beats the algorithm-indexed description and the oracle constant; on W64 it does not beat the oracle constant | T16 hit (the description at every budget, the constant from (2, 3) on). W64 miss: it beats the constant from (3, 4) on |
| E7.4 | the best (b) gains 1.5-4x per sequence on both; at equal FLOPs it beats Monte Carlo only at N >= 2^14 | per-sequence clause hit (2.3x and 2.9x). Equal-FLOP clause hit only marginally: it ties or barely wins at N = 2^16 (1.08x, 1.16x, within the s.e.) and loses below |
| K7 | TOP3 misses by > 50% on both models | did not fire |

**What it points to (not tested).**
- The quadrature is the cost bottleneck. TOP3's error probability is close to a step function of the top-3 values
  (on T16, 89% of the mass is predicted below 0.1% and 3% above 60%). A product rule then converges slowly, by
  roughly 1/K per axis. A deductive estimator would need to locate the decision boundaries in the value space and
  integrate across them in closed form; within one ReLU pattern the linearised margin is affine in the values.
- The per-point model is the accuracy bottleneck. It misses where the lower inputs act non-linearly (T16's
  negative 2nd max) and is too confident. Candidates are carrying the 4th order statistic exactly, or a small mixture
  of representative arrangements of the lower inputs.
- Even exact, a 10-27% bias is far from the 3-5% relative accuracy that matching sampling at about 10^8 FLOPs needs.
  What remains is in the per-point model given the algorithmic variables, not in the closure over them.

## Files

`work/algzoo-noisy/`:
- `PREREG.md`, `prereg_hashes.txt`, `GPU-LOG.md`.
- Model and training: `az.py` (model, noise, evaluation), `check_arc.py` (test-time noise on ARC's weights), `train.py`
  (fine-tune and from scratch, CPU), `train_batched.py` (several models at once, GPU), `gpu/` (launch scripts),
  `runq*.py` (CPU queues), `eval_ck.py` and `gpu_table.py` (criterion on checkpoints and final models).
- Analysis:
  - `alg.py` (algorithmic state, sampling);
  - `dag.py`, `dag_run.py` (flip-DAG), `ldag_run.py` (accuracy- and loss-targeted DAG);
  - `lowrank.py`, `lowrank_step.py`, `lowrank_avg.py` (question subspaces);
  - `backward.py` (v_t and information), `meet.py` (meeting diagnostic);
  - `fam.py` (family scorecard), `adf.py`, `adf_diag.py` (deductive chain; `adf_v1_backup.py` is the version without
    mask support used for the first T16 runs, which gives identical results);
  - `algrel.py`, `algrel_quad.py` (descriptions relative to the algorithm, quadrature check);
  - `devpred.py` (scenarios), `noisesens.py` (propagated-noise metric);
  - `estim.py`, `bar3.py` (ARC's bar, three columns), `exact22.py`, `estim22.py`, `exact22_check.py`, `bar22.py`
    (M_{2,2} closed form, certificate, family, checks). `ldag_estim.py` was superseded by `bar3.py surrogate_only` and
    not run;
  - `run_all.sh` (the battery), `step1_table.py`, `compare.py`;
  - `sysid.py` (section 9.1), `top3.py`, `bar_top3.py`, `top3_conv.py` (TOP3, its bar and its convergence check;
    results in `results/sysid_*.json`, `results/top3_*.json`, `results/top3conv_*.json`).
- Results: `runs/*.json` (training evaluations; GPU models under `runs/gpu/`, checkpoints under `runs/gpu_ck/`),
  `results/*.json`, `results/step1.md`. Reviews: `review/`.
