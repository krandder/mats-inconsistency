# Noisy AlgZoo RNNs: can the noisy-channel program interpret ARC's 2nd-argmax models?

29 October 2026. Work directory `work/algzoo-noisy/` (code, logs, `results/`, `PREREG.md` with sha256 and times in
`prereg_hashes.txt`, `GPU-LOG.md`). Draft; sections marked *pending* are still running. The progress log is at the end
of the summary.

## Summary

*Draft, 01:30Z; rewritten when the pending runs finish.*

1. **Noise.** ARC's 2nd-argmax RNNs are fragile under activation noise. Under Kelvin's strict criterion (noise on, loss
   within 2% of the original, accuracy not lower), the only passing variant found so far is a fine-tune of ARC's
   432-parameter example M_{16,10} at sigma = 0.001, i.e. Gaussian noise of 0.1% of each neuron's activation scale,
   about 10 bits per neuron per step. At sigma = 0.003 a fine-tune 4x longer still misses: loss +8.3%, accuracy -0.74
   points. At sigma = 0.01: loss +220%. No sigma >= 0.001 passes for M_{2,2} or M_{4,3}. From-scratch training with
   noise, and the wider networks that Kelvin's program allows, are *pending*.
2. **Interpretation of the passing model (T16: FT example, sigma = 0.001).** At this precision the noise changes
   neither the network nor its fragility: the margin, measured in units of its own noise, has the same distribution as
   in ARC's noiseless weights. The question ("does the model output the 2nd argmax?") has no sparse structure:
   - the question-driven DAG needs 82% of the 2,480 edges to keep held-out flipped answers at 1.3%;
   - the question needs 13 of the 16 directions at every step;
   - every tractable description we tried mispredicts the error rate, including best-fit members fitted to the real
     activations (true 4.1%; best family 10.6%; deductive chain 73%).
   The causes we could identify are below. The analysis does reproduce and extend part of ARC's interpretation:
   - ARC's max-tracking neurons {1, 2, 4, 6, 7}, plus neuron 8, are described almost exactly by the algorithmic state
     (R^2 >= 0.99);
   - neuron 7 alone carries 39-52% of the noise that reaches the decision;
   - errors are decided late (half of the information about the eventual error first appears at step 9);
   - 7 of 9 preregistered scenario predictions hit. The two misses are informative: the last position is not easier,
     and a max that comes after the 2nd max is only 1.26x harder.
3. **ARC's bar.** *Pending for M_{16,10} and M_{4,3}.* On M_{2,2} with noise, ARC's region analysis plus a linear noise
   term gives the error rate in closed form at 1,432 FLOPs, about 10^6 times lower MSE than Monte Carlo at equal FLOPs
   over 16 weight-perturbed models. Nothing comparable exists for M_{16,10}.

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
- In every row, the 95% CI of L_var - L_orig is narrower than 1% of L_orig.

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
- M_{2,2} cannot pass at a practical sigma. Its errors become noise-dominated: the closed form of section 5 gives
  error ≈ 7.7e-5 + 0.32 sigma. Relative noise cannot be outscaled by larger weights.
- M_{4,3} misses at sigma = 0.001 by 0.5-1.2% of loss beyond the 2% allowance.
- The noise-marginalised predictive (16 draws) beats the single-draw one: loss 0.1324 vs 0.1591 at sigma = 0.003.
  Under a marginal-loss criterion, the example would pass at sigma = 0.003. We kept the single-draw criterion we had
  fixed in advance.

**Why the networks need about 10 bits per neuron (the propagated-noise metric).** These are the program's node weights
for the DAG.
- Along each input's noiseless path, with the gates held fixed, we compute how much noise injected at (step t,
  neuron i) moves the target-logit margin m. Its noise s.d. is s(x).
- Code: `noisesens.py`, on 2^17 inputs. Measured errors come from 2^20 noisy samples.

| model, test sigma | error noiseless / predicted (linear) / measured | median and 10% quantile of abs(m)/s | share of decision noise: neuron 7 / {1, 2, 6, 7, 8} / steps 8-10 |
|---|---|---|---|
| ARC example, 0.01 | 0.046 / 0.144 / 0.174 | 1.43 / 0.35 | 52% / 89% / 53% |
| ARC example, 0.001 | 0.046 / 0.048 / 0.049 | 14.3 / 3.5 | 46% / 88% / 56% |
| T16 (fine-tuned at 0.001), 0.001 | 0.038 / 0.040 / 0.041 | 14.5 / 3.5 | 39% / 87% / 63% |
| T16, 0.01 | 0.038 / 0.138 / 0.174 | 1.45 / 0.35 | 44% / 88% / 60% |

- At 1% noise, half of all inputs have a margin of less than 1.45 noise s.d.
- Linear response predicts 70-80% of the extra error; gate flips account for the rest.
- The decision noise enters almost entirely through ARC's max-tracking neurons. Neuron 7 alone carries 39-52% of it.
  ARC describes it as roughly max(0, x_0..x_{t-1}) - x_{t-1}, a difference of large terms that is read out with a
  large gain. Together, neurons 1, 2, 6, 7 and 8 carry 87-89%.
- Fine-tuning at sigma = 0.001 did not make the network sturdier. T16 has the same margin-to-noise distribution as
  ARC's weights; the fine-tune only lowered the noiseless error. In effect, T16 is ARC's network plus noise at
  10 bits per neuron.

## 3. Interpretation of T16 (the example fine-tuned at sigma = 0.001)

The question is "does the model output the 2nd argmax?". Its input distribution is x ~ N(0, I_10). Its answer, for a
given input, is the error probability over the noise. T16's error rate is 4.06% (2^24 samples; 3.78% without noise).

### 3.1 Forward chain: describing h_t step by step

**Best member of each family, fitted to the real activations.** This is a capacity test: a deductive chain built from
the same family cannot do better. Code: `fam.py`, 2^18 sequences; error predictions on 2^18 fresh sequences.
- R^2 is the fraction of h_t's variance the family explains.
- "Unexplained bits" is 1/2 log2 det(I + Sigma_res / (sigma^2 diag r_t^2)), averaged over cells. It is the information
  a description in that family would still need, per step, to reach the noise precision.

| family (cells; within-cell model) | parameters | R^2 at t = 3 / 6 / 10 | unexplained bits at t = 3 / 6 / 10 | predicted error rate (true 0.041) |
|---|---|---|---|---|
| G: one Gaussian | 152 | 0 / 0 / 0 | 103 / 131 / 133 | 0.900 |
| AC: algorithmic cells (a_t, b_t), one Gaussian each | 13,680 | 0.37 / 0.45 / 0.46 | 71 / 89 / 92 | 0.186 |
| CL: cells, linear in (M, S, x_last) and their ReLUs | 22,320 | 0.985 / 0.945 / 0.925 | 45 / 68 / 73 | 0.132 |
| CL3: CL plus the 3rd max and its position | 39,600 | 0.996 / 0.972 / 0.968 | 41 / 65 / 70 | 0.133 |
| CLX: cells, linear in every input so far and its ReLU | 42,480 | 0.996 / 0.981 / 0.976 | 41 / 61 / 66 | 0.106 |

Here a_t and b_t are the positions of the running max and 2nd max, and M and S their values.
- The algorithmic state describes ARC's neurons almost exactly. In the noiseless example, cell x linear in (M, S,
  x_last, ReLUs) gives R^2 >= 0.993 at every step for neurons 1, 2, 4, 6, 7 and 8: ARC's subcircuit plus neuron 8, which
  Chen's notes describe as about ReLU(2nd max - x_t) early on. The other ten neurons fall to 0.72-0.93 by step 10.
- Even so, the error rate is wrong in every family, by 2.6x for CLX up to 22x for the single Gaussian. The residual is
  2-7% of the variance, but it is not Gaussian and it is not independent of the decision. Spread as a Gaussian, it
  puts probability on the wrong side of decision boundaries that the network respects by construction.
- At this sigma, even the input-exact linear family leaves 66 bits per step unspecified, about 4 bits per neuron.

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
| I(E; Z_t) / H(E), network state + algorithmic state | 0.01 | 0.02 | 0.05 | 0.08 | 0.17 | 0.29 | 0.51 | 1.00 |
| I(E; A_t) / H(E), algorithmic state alone (a, b, M and S quartiles) | 0.01 | 0.02 | 0.02 | 0.03 | 0.06 | 0.11 | 0.19 | 0.37 |

- Errors are decided late. Half of the information about the eventual error first appears at step 9.
- Of the 311 errors, 228 are decided only at steps 9-10, meaning v_t stays at or above 0.5 from then on.
- The step at which an error is decided does not favour algorithmic events. Of the decided errors, 56% fall on steps
  with no new max or 2nd max, 25% on a new max and 19% on a new 2nd max. The base rates are 52%, 29% and 19%.
- The algorithmic state carries about a third of the error information. The rest is held in the network's own state.
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
  The question's events are thresholds on the same variables that the closure smears. So the failure is not error
  accumulating with depth; it is the Gaussian closure itself.

### 3.3 Question-driven DAG

**Construction.**
- Nodes are (neuron, step). The edges are 2,304 recurrent, 160 input and 16 readout, 2,480 in all.
- An ablated edge is replaced by its mean contribution. Surrogates run on the same inputs and noise as the network.
- The question divergence is flip = P(the surrogate's correctness differs from the network's).
- The description cost is the number of kept edges. The forward-only cost is the chain-rule path KL in nats.
- Code: `dag_run.py`. Edges are ranked by first-order, near-boundary saliency; the program finds the smallest prefix
  meeting the flip target on 2^13 fit inputs, then removes edges greedily, and evaluates on 2^15 held-out inputs.

**Results.**
- **No sparse structure.** At flip <= 2% (held-out 2.4%), the DAG keeps 1,933 edges (78%); at 1% (held-out 1.3%),
  2,038 (82%); at 0.5% (held-out 0.87%), 2,106 (85%).
- For scale, two independent noise draws on the same inputs already flip 1.5% of answers.
- Keeping the top 62% of edges by saliency flips 16.5% of answers; keeping 41% flips 52%.
- **Structure from the question helps only a little.** Ranking edges by their forward (path-KL) saliency flips 6.1% of
  answers at 2,048 kept edges, against 1.2% for the question ranking.
- Any surrogate that ablates more than a few edges has a path KL of 4.5e6-5.7e6 nats at sigma = 0.001. Here the
  forward-only description must keep everything.
- **Per step (flip <= 1% DAG).** Recurrent edges kept, by source step t = 1..9: 77, 204, 219, 234, 232, 233, 241, 239,
  236 of 256. Input edges kept: 6-14 of 16. Readout edges kept: 16 of 16.
- The only context-specific sparsity is at step 1: ten neurons have no input weight to speak of, so their outgoing
  edges can go.
- **ARC's subcircuit is fully used.** 216 of its 225 internal edges are kept, and every one of neurons 1, 2, 4, 6, 7 has
  out-degree >= 13 at every step from 3 on. The DAG, though, keeps just as much of everything else.

**Question subspaces** (`lowrank.py`). At each step the description keeps the top k directions of
C^{1/2} F C^{1/2}, where C = Cov(h_t) and F is the near-boundary Fisher of the margin. The rest is replaced by the best
linear reconstruction. Same inputs and noise; 2^16 held-out inputs.

| kept directions per step (of 16) | 8 | 10 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|
| flip, question directions | 0.48 | 0.23 | 0.061 | 0.0070 | 0.0036 | 0.0001 |
| flip, top principal components | 0.75 | 0.71 | 0.58 | 0.51 | 0.27 | 0.039 |

- Second-order "question variance" is concentrated: 4 directions hold 98.7% of it at step 5 and 6 directions hold
  97.5% at step 10. Yet the answers need 13.
- The decisions depend on low-variance directions through gate thresholds and the argmax, which a second-order metric
  does not see. The same blind spot makes the edge-saliency ranking non-monotone.

### 3.4 Deviation prediction and scenarios

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
hits come from the mechanism-level reading (max(0, ·) features, near-ties), not from a compact description.

## 4. ARC's bar: estimating the noisy model's error rate

Each family is the base model plus weight-perturbed copies. Every copy has r recalibrated, and its truth comes from
2^24 samples (`estim.py`). Each estimator's MSE is averaged over 16 models x 8 repetitions.

FLOPs per sampled sequence are matmul and elementwise operations: 6,283 for M_{16,10} and 188 for M_{4,3}. Counting
20 FLOPs per Gaussian draw would add 3,400 and 300; this changes no ordering.

The deductive estimates count every FLOP, including truncations, maps, merges and the final orthant step.

**T16 family.** Relative weight perturbation 0.3%, chosen once so that the error rates spread by a few points: 4.1% to
15.8%, mean 7.5%, s.d. 2.9 points. MSE:

| FLOPs | Monte Carlo | stratified (90 cells) | RQMC (Sobol over x) | deductive ADF |
|---|---|---|---|---|
| 6.4e6 (1,024 sequences) | 6.3e-5 | 7.9e-5 | 4.5e-5 | |
| 2.6e7 | 1.6e-5 | 1.7e-5 | 1.1e-5 | |
| 4.0e7 | | | | 0.451 (k = 0; bias +67 points) |
| 1.0e8 | 5.1e-6 | 3.5e-6 | 2.6e-6 | |
| 1.3e8 | | | | 0.441 (k = 2) |
| 4.1e8 | 1.2e-6 | 8.5e-7 | 6.7e-7 | |
| 5.1e8 | | | | 0.433 (k = 4) |

- For scale, a constant estimate (the leave-one-out family mean) has MSE 9.6e-4.
- The deductive description is about 10^5 times worse than Monte Carlo at equal FLOPs, and 450 times worse than the
  constant.
- Stratification over the algorithmic cells gains 1.4x. RQMC gains 1.8-2.0x, the best of the samplers.
- Below 4,096 sequences, the stratified sampler draws fewer than N sequences (N // 90 per cell); this affects only the
  first row.

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
  term of fixed s.d. s: the step-2 noise enters linearly, and the step-1 noise is propagated through the step-2 gates.
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

This is ARC's result for M_{2,2} with noise added. The description is exact because every input variable is carried
exactly, and the noise needs only one linear-response term.
