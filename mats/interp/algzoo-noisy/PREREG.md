# PREREG: noisy AlgZoo 2nd-argmax RNNs (work/algzoo-noisy)

Part 1 was drafted about 2026-10-29 00:24Z, before any noisy training run and before reading the test-time-noise check
(`check_arc.py`, started 00:22Z). The first write was blocked by a shell hook, so Part 1 and Amendment 1 were saved
together at 00:27Z (times corrected at 00:27Z from the system clock; the draft had guessed 00:30Z and 00:36Z). Amendments are dated; nothing above an amendment is edited afterwards. Hashes: `prereg_hashes.txt`.

## Part 1. Step 1: noisy variants (drafted about 00:24Z)

**Models.** 2nd-argmax RNNs from AlgZoo: M_{2,2} (zoo seed 2, the best of 5), M_{4,3} (zoo seed 0, best), M_{16,10}
= `example_2nd_argmax()` (ARC's interpreted 432-parameter model) and `zoo_2nd_argmax(16, 10, seed)` for seeds 0-4.

**Noisy model.** h_{t+1} = ReLU(W_hh h_t + W_hi x_t) + sigma r_{t+1} eps_{t+1}, eps ~ N(0, I_d), at every step
t+1 = 1..n including the last (whose noise passes linearly into the logits). r_{t,i} is the RMS of neuron i's
post-ReLU activation at step t under the noisy model itself (inputs N(0, I_n), earlier noise on): an EMA of batch RMS
during training, then frozen by a fixed-point calibration on 2^20 sequences. Per-neuron scales keep the noisy model
equivariant under ARC's neuron-rescaling symmetry. Noise stays on at test time and in every analysis.
Grid: sigma in {0.01, 0.03, 0.1} (about 6.6, 5.1 and 3.3 bits per neuron per step at SNR 1/sigma^2).

**Training.**
- Fine-tune (FT) from ARC's weights with the noise on: AdamW (weight decay 0.001, ARC's), batch 1024, 2^16 steps
  (2^26 sequences), learning rate with ARC's schedule shape compressed: peak 0.1 x 0.04/sqrt(d), warmup 2^10 steps,
  exponential decay to 0.1 x peak at half the run, then constant (ends at ARC's final learning rate).
- From scratch (FS), M_{16,10} only: ARC's recipe (`training.py` defaults: AdamW lr 0.04/sqrt(d), wd 0.001, batch 1024,
  warmup 2^14 steps, decay to 0.01x) but with n_train = 2^28 instead of 2^30 (CPU budget), seeds 0-4, at sigma = 0
  (control for the shorter run) and at the chosen sigma.

**Metric.** Single-draw accuracy with noise on (one noise draw per input), 2^22 fresh sequences, binomial SE reported;
against A0 = the accuracy of ARC's original noiseless weights on 2^22 sequences.

**Criterion for "doesn't degrade performance much" (primary, superseded by Amendment 1):** A0 - A_sigma <= 1.0
accuracy point, for the FT model. The chosen sigma* for a model is the largest grid value meeting it. If none does,
add 0.003 and report.
**Secondary (reported, not used to choose):** error ratio (1 - A_sigma)/(1 - A0) <= 1.25; and A_sigma against the
spread (min, max, SD) of ARC's 5 noiseless seeds.
**Prediction (P1):** M_{16,10} FT meets the primary criterion at sigma = 0.03 and fails it at 0.1; M_{2,2}'s error ratio
exceeds 10 at every sigma in the grid (a noise-free model with error 1/13,000 has almost no margin at the x_0 = x_1
boundary, and per-neuron relative noise cannot be outscaled by larger weights).

## Amendment 1 (2026-10-29 00:27Z): stricter criterion from Kelvin, via the coordinator

Kelvin: "ideally I'd like our noisy variants to be just as good, or almost indistinguishable in loss." The coordinator's
instruction: with the noise on at test time, a variant's test loss must lie within the originals' seed-to-seed spread
or within a small relative gap fixed in advance, and its accuracy must not be lower; train or fine-tune longer, or lower
sigma, until a variant meets it; use the largest sigma that meets it; report the loss gap with CIs; if no sigma meets
it for some model, say so with numbers before interpreting that model.

**Criterion (replaces the primary criterion of Part 1).** For a variant fine-tuned or trained for a given original
(same d, n; FT variants are compared with the exact weights they started from):
- Loss: L_var <= 1.02 x L_orig, where L is the mean cross-entropy of the logits on 2^22 fresh N(0, I_n) sequences,
  L_var with the noise on (one draw per input) and L_orig noiseless. The fixed relative gap is 2%. The 95% CI of
  L_var - L_orig is reported (paired over the same inputs).
- Accuracy: A_var >= A_orig - 2 SE(A_var - A_orig) (not significantly lower, one-sided ~97.5%), same inputs.
- Both must hold. Where the variant sits inside ARC's 5-seed spread (min, max, SD of L and A) is reported but not used.
- Loss with the noise-marginalised predictive (-log of the mean softmax over 16 noise draws) is reported as secondary.

**Search.** sigma grid extended downward: {0.1, 0.03, 0.01, 0.003, 0.001}. FT length up to 2^18 steps (2^28
sequences), with the same schedule shape stretched. sigma* = the largest grid value whose variant meets the criterion.
If no grid value meets it for a model, that model is reported as failing, with numbers, before any interpretation of it.

**Prediction (P1').** M_{16,10} (example and zoo seed 2) meets the criterion at sigma = 0.01 or 0.03 after FT (the noiseless
originals are far from the task optimum, 95% against 99% for ARC's handcrafted 22-neuron model, so there is slack for
the fine-tune to absorb the noise); M_{4,3} meets it only at sigma <= 0.003; M_{2,2} meets it at no grid value.

## Amendment 2 (2026-10-29 00:34Z): from-scratch arm and best-of-5 comparison; sigma = 0.1 runs stopped

Seen before writing this: the first 4k-37k fine-tune steps of the M_{16,10} runs (validation accuracy, noise on, about
94% at sigma 0.003, 88% at 0.01, 73% at 0.03, and divergence at 0.1, where the loss rose above 1,000; those two runs were
stopped at about 37k steps). No final evaluation of any noisy run has been read.
- **From-scratch arm (FS), M_{16,10}:** ARC's recipe as in Part 1 at n_train = 2^28, seeds 0-4, sigma in {0, 0.003,
  0.01}. sigma = 0 at 2^28 is the control for the shorter run. If the sigma = 0 control is clearly worse than ARC's 2^30
  originals, the best noisy candidates are rerun at 2^30.
- **Comparison for FS (added):** ARC's zoo model is the best of 5 seeds, so the FS variant for (d, n, sigma) is the
  best of our 5 seeds, chosen by noisy validation loss on a separate 2^16-sequence set, and it is held to the Amendment 1
  criterion against ARC's zoo model (best of 5) and, for M_{16,10}, also against the example model. Seed-matched
  comparisons are reported too.
- **sigma = 0.1** is dropped for M_{16,10} (divergent); it stays in the grid for M_{2,2} and M_{4,3}.

## Part 2 (2026-10-29 00:48Z): Step 2, interpretation. Written before any per-scenario error breakdown of any model.

Already seen when this was written: Step-1 evaluations (FT of the example model meets the criterion at sigma = 0.001
and fails at 0.003; seed 2 fails at 0.001); test-time noise on ARC's weights; R^2 of h_t on algorithmic-state
features for the noiseless example model (neurons 1, 2, 4, 6, 7, 8 at >= 0.99; others 0.7-0.99); the ADF error
estimate on the noiseless M_{2,2} (10.6% vs 0.008%) and M_{4,3} (30% vs 1.5%) from a first test of `adf.py`. A first
DAG run on FT-example sigma = 0.001 was started (results/dag_ft_ex_s0.001.json) and its output was not read.

**Targets.** T16 = the M_{16,10} variant used for interpretation: the FT of ARC's example model at sigma*, the largest
grid sigma at which it meets the Amendment 1 criterion (0.001 at the time of writing; 0.003 if the 2^18-step FT meets
it). If an FS model meets the criterion at sigma >= 0.01, it is analysed as a second target, T16-FS. Checks: M_{2,2}
and M_{4,3} at their sigma* (or at sigma = 0.001 if none meets the criterion, reported as such).

**DAG (structure from the question).** Edges: 2,304 recurrent (step t neuron j -> step t+1 neuron i, t = 1..9), 160
input, 16 readout. Ablated edges are mean-ablated; surrogates run on common inputs and noise. Question divergence:
flip = P(surrogate's correctness != network's correctness); forward-only cost: chain-rule path KL in nats.
- D1. On T16, the question-driven DAG with flip <= 1% (held-out) keeps <= 25% of the 2,480 edges.
- D2. ARC's subcircuit neurons {1, 2, 4, 6, 7} are all used (in- and out-degree >= 1) at every step t >= 3, and
  neuron 2 (running max) has the largest summed out-degree.
- D3. Ranking edges by their forward (path-KL) saliency needs >= 3x as many edges as the question ranking to reach the
  same held-out flip of 1%; the question DAG's path KL is >= 1,000 nats (a forward-only description at this sigma must
  keep almost everything).
- D4. At least 3 neurons have no kept outgoing edges at steps t <= 3 (context-specific sparsity).

**Deviation prediction.** Per-input error probability eps(x) (K = 32 noise draws) on 2^15 held-out inputs.
- V1. R^2 between the DAG surrogate's eps_hat(x) (flip <= 1% DAG) and the network's eps(x) >= 0.75; the noise floor
  (two independent K-draw estimates of the network) is reported.
- Scenario predictions (mechanism: max(0, .) features from ARC's analysis; noise smears near-ties). Overall error p.
  - S1 (exactly one positive input, so the 2nd max is negative): error rate >= 5 p.
  - S2 (all inputs negative): error rate >= 0.5.
  - S3 (max - 2nd max in its lowest decile): error rate >= 2 p.
  - S4 (2nd max - 3rd max in its lowest decile): error rate >= 2 p.
  - S5 (the max comes after the 2nd max, a > b): error rate >= 1.3 x that for a < b.
  - S6 (the max or the 2nd max is the last input, position 9): error rate <= 0.5 p.
  - S7 (noise-sensitive inputs, 0.05 < eps(x) < 0.95): their share is >= 3x higher within S3 or S4 than overall.
  - S8 (DAG quantitative): for each of S1-S6 the DAG surrogate's error rate is within 25% (relative) of the network's.
  A scenario is a hit if its stated inequality holds on held-out data with the 95% interval on the right side.

**ARC's bar (error rate of the noisy model, MSE vs FLOPs).** Family: T16 plus weight-perturbed copies (i.i.d.
Gaussian perturbation of every weight, relative size chosen once so the error rates spread by a few points), r
recalibrated per copy, truths from 2^24 samples each. All FLOPs counted, including setup.
- E1. Deductive estimates from descriptions (the alg-cell x gate-pattern Gaussian chain `adf.py`, and variants): on
  M_{16,10}, |bias| >= 1 percentage point, so they lose to Monte Carlo at every budget above 10^4 samples. On M_{2,2},
  a description that carries the named variables (here all inputs) exactly and integrates the noise analytically beats
  Monte Carlo by >= 100x in MSE at equal FLOPs.
- E2. Stratified sampling over the 90 algorithmic cells gains <= 1.5x over Monte Carlo in MSE; RQMC over x (noise by
  plain MC) gains <= 2x.
- E3. No non-vacuous certified bound for T16 (coverage certificates at this precision); certified intervals only for
  M_{2,2}.
**Kill tests.** K1: if the question DAG needs >= 80% of edges for flip <= 1%, the DAG interpretation is reported as a
blocker (no sparse question structure). K2: if every deductive estimate is >= 10x worse in MSE than sampling at its
own FLOPs on M_{4,3} and T16, ARC's bar is reported as failed, with the identified cause.

## Amendment 3 (2026-10-29 01:13Z): wider noisy RNNs (coordinator's request) and full-length from-scratch runs on a GPU

Seen before writing this: Step-1 fine-tune results (only FT-example at sigma 0.001 meets the criterion; the 2^18-step
FT at 0.003 gives loss +8.3%, accuracy -0.74 points); the interpretation results on T16 reported to main at 01:07Z.
No wide model has been trained.
- **Arm W (wide):** from scratch with ARC's recipe at the full 2^30 sequences, widths d = 32 and 64 on the M_{16,10}
  task (n = 10), sigma in {0.01, 0.03}, seeds 0-2, plus d = 16 at sigma in {0, 0.003, 0.01}, seeds 0-1 (sigma 0 checks
  that this code reproduces ARC's noiseless level at full length). One on-demand L4 (`gpu/launch_gpu.sh`, run id
  azn-wide-1), batched training (`train_batched.py`).
- **Criterion:** Amendment 1 against ARC's example M_{16,10} (the stricter reference: loss 0.147, accuracy 95.3%) and
  against zoo seed 2 (best of 5); each wide model is judged on its own (no best-of-5 selection), and the best per
  (d, sigma) is reported with how far it misses.
- **Predictions.** W1: at least one d in {32, 64} meets the criterion against the example model at sigma = 0.01; none
  does at sigma = 0.03. W2: if one passes, its question DAG (flip <= 1%) keeps <= 50% of its edges (T16: 82%). W3: its
  question needs <= d/2 directions per step (T16: 13 of 16). W4: the deductive ADF still misestimates its error rate by
  >= 5x. W5: no deductive estimate beats Monte Carlo at equal FLOPs on it.
- **Interpretation tests on a passing wide model** (if any): DAG sweep and greedy, question subspaces, the family
  scorecard (fam.py), the ADF, devpred scenarios S1-S8, and ARC's bar, as for T16.

## Amendment 4 (2026-10-29 01:39Z): loss-targeted DAG (Kelvin's question, via the coordinator)

Seen before writing this: all T16 results reported to main (flip-DAG 2,038 of 2,480 edges at flip <= 1%; 13 of 16
directions; ADF bias +66 points; scenarios); GPU training logs at step 82k. No loss-targeted search has been run.
- **Question divergence for the average (the estimand ARC cares about):** dL = |E[CE_sur] - E[CE_net]| (cross-entropy,
  one noise draw, common inputs and noise) and, separately, dp = |p_sur - p_net| for the error rate.
- **Tolerance relative to sampling:** for a reference budget of N_ref network samples, tol = SE_net(N_ref) =
  sd(per-sample CE or error indicator) / sqrt(N_ref), with N_ref in {2^10, 2^14, 2^18}.
- **Guard against cancellation:** the bias must also stay within tolerance in each of 13 held-out strata fixed here:
  4 quartiles of the top-two gap (max - 2nd max), and 9 position groups (max position in {0-2, 3-6, 7-9} x 2nd-max
  position in the same three groups). Stratum k's tolerance is its own sampling SE under proportional allocation,
  sd_k / sqrt(N_ref pi_k). Bias estimates on the 2^15-input fit set must meet tol (and 0.5 tol for the overall
  average, to leave room for estimation noise); results are reported on 2^16 held-out inputs.
- **Search:** the same edge set, mean ablation, and saliency-ranked prefix then greedy removal as the flip-DAG; ranking
  by a loss saliency E[(dCE/dz_i W_ij (h_j - mu_j))^2]. Directions per step: the lowrank family (question and PCA
  directions) scored by the same loss criterion.
- **Use in ARC's bar:** (i) plug-in estimator that samples the loss-targeted surrogate (cheaper per sample), MSE vs FLOPs
  including its bias, over the T16 family; (ii) the deductive ADF run on the surrogate's sparser, biased map; (iii) the
  meeting diagnostic (meet.py) on the surrogate.
- **Predictions.** L1: the loss-targeted DAG keeps <= 50% of edges at N_ref = 2^10, <= 75% at 2^14, and >= 80% at
  2^18 (the dp version needs at least as many as the dL version at the same N_ref). L2: the loss question needs >= 11
  of 16 directions per step at N_ref = 2^14. L3: the plug-in estimator beats Monte Carlo on the network by at most 2x
  in MSE at any budget. L4: the ADF on the surrogate is no closer to the truth than on the full network (bias >= 30
  points).

## Amendment 4b (2026-10-29 01:40Z): correction from the coordinator, before any loss-targeted run

- **Primary question for the average-targeted DAG is the error rate** (dp, ARC's estimand for M_{16,10} is accuracy);
  the cross-entropy (dL) is secondary. The greedy pass is run for dp at N_ref = 2^14 (prefix searches for all six
  combinations). Predictions L1-L2 are read with dp as primary (L1's ordering clause is dropped).
- **ARC's bar follows ARC's protocol:** a deductive estimator (no sampling) of the noisy model's error rate (noise on),
  its MSE over an ensemble of weight-perturbed copies large enough to change accuracy significantly (the 16-copy T16
  family of `estim.py`), compared with random sampling at each compute budget; matching sampling is the bar.
  Deductive estimators here: the ADF on the full network, the ADF on the dp-targeted surrogate, the M_{2,2} closed form.
- **Hybrids are reported separately and do not count toward the bar:** stratified sampling, RQMC, and the plug-in
  estimator that samples the dp-targeted surrogate (prediction L3 refers to it).

## Amendment 5 (2026-10-29 01:59Z): how ARC's bar is reported (Kelvin's decisions, via the coordinator)

Seen before writing this: T16 results for the deductive ADF (bias about +66 points) and the hybrids (stratified 1.4x,
RQMC 1.4-2.0x); the ADF on two wide checkpoints (step 131k) misestimates by 0.90 vs 0.061 and 0.69 vs 0.041. The
accuracy-targeted DAG greedy is still running; ldag_estim.py has not been run.
- **Three columns, MSE against FLOPs:** (a) deductive only; (b) deductive plus a little sampling, starting from the
  deductive cost (control variates built from the deductive description; sampling guided by the structure, e.g. Neyman
  allocation over algorithmic cells from the deductive per-cell predictions; multilevel sampling with the cheaper
  surrogate); (c) plain sampling. Stratified (proportional) and RQMC stay as sampling-only references.
- **Advice rules (ARC's "train and explain"):** structure (the advice) is found once per trained model and never sees
  the weight-noise variants; its compute is reported next to the training compute. For each variant only the
  estimator is recomputed. Regenerating anything per variant is allowed only inside the estimator, deductively and
  counted; if it samples, the estimator moves to column (b).
  Concretely: the dp-targeted surrogate's edge set and mean biases mu_t come from the base T16 only (no per-variant
  sampling) for column (a); per-variant sampled means put that estimator in (b).
- **Per seed:** the wide models (3 seeds per width and sigma) get their own structure and family per seed.
- **Predictions.** B1: column (a) loses to (c) at every budget on T16 and on every wide model (bias >= 10 points).
  B2: the best column-(b) estimator gains at most 2x over (c) on T16 at budgets >= 10^7 FLOPs. B3: on M_{2,2}, the
  closed form used as a control variate (the network's error indicator minus the fixed-gate model's, on the same input
  and noise, plus the closed-form mean) beats (c) by >= 100x in MSE at every budget >= 10^4 FLOPs.

## Amendment 6 (2026-10-29 02:48Z): descriptions relative to the algorithm (coordinator's steer; analog of phase B's E3)

Seen before writing this: fitted family scorecards on T16 (CL predicts 13.2% vs 4.1%) and on a width-32 checkpoint at
step 131k (CL 5.4% vs 4.2%, CLX 3.5%); the 2^19-checkpoint criterion results; structure previews on the wide
checkpoints. Nothing below has been run.
- **Algorithm-relative ablation:** an ablated edge (t, j -> i) carries W_ij E[h_tj | algorithm state] instead of
  W_ij E[h_tj]. The conditional mean is the CL regression (cell (a_t, b_t) x linear in M, S, x_last and their ReLUs),
  fitted once on the base model's own samples (advice). Same edge set, saliency ranking, flip and dp targets as before.
- **Algorithm-indexed description with linear-response noise:** per cell at step n, logits ~ N(W_oh B_c f(u),
  W_oh Sigma_c W_oh^T) with the network's own noise added; P(error | cell, u) by a fixed deterministic set of Gaussian
  nodes; the deductive average sums over cells and integrates u = (M, S, x_last) over its exact law by quadrature.
  The per-input prediction eps_hat(x) is the same formula at the input's own (cell, u).
- **ARC's bar, three columns, on T16 and the passing wide models (d = 32 and 64 at sigma = 0.01):** (a) the
  description's deductive average; for weight-noise variants the advice (B, Sigma) is the base model's and only the
  last step is recomputed with the variant's weights (one exact linear-response step from the base description of
  h_{n-1}); (b) control variate: Monte Carlo of e(x) - beta (eps_hat(x) - its deductive mean); (c) Monte Carlo.
- **Predictions.** C1: algorithm-relative ablation keeps <= 40% of edges at held-out flip <= 1% on T16 and <= 30% on
  the wide models (unconditional: 85% and 73-75%). C2: the description's deductive average is within 25% (relative)
  of the true error rate on the wide models but off by >= 2x on T16. C3: column (a) loses to (c) at every budget
  >= 10^6 FLOPs on all three models; column (b) gains <= 3x over (c). C4: the description has 90 cells and
  90 x (7d + d(d+1)/2) parameters at step n (T16: 22,320; d = 32: 67,680; d = 64: 227,520).

## Amendment 7 (2026-10-29 05:09Z): systematic vs idiosyncratic errors, and a deductive estimator with no closure in the algorithmic variables (coordinator's bounded next step)

Seen before writing this: everything in the report of 05:01Z (scenario rates S1-S8 on T16 and W32/W64, the
algorithm-mean readout's 1.0% on T16 and 0.06% on W64 at sigma 0.01, the reliability of per-input error
probabilities, the wide-model bars). Nothing below has been run. Models: T16 (`runs/ft_16_10_ex_s0.001.npz`) and
W64 at sigma = 0.01 (`runs/gpu_ck/w64_s1_sig0.01_ck524288.npz`, the model of Amendment 6). CPU only.

**Step 1, systematic share (`sysid.py`).**
- Data: 2^18 inputs (T16) and 2^17 (W64), split 50/25/25 into train/validation/test. Per input: 32 noise draws in
  two halves of 16 (eps_A, eps_B, each an estimate of eps(x) = Pr_noise[error | x]) and one further single draw e.
- Features. F1 (the coordinator's list): g12 = x(1) - x(2), g23 = x(2) - x(3), sign(x(2)), pos1, pos2, and the
  step of the last new max or new 2nd max (equal to max(pos1, pos2), kept anyway). F2: F1 plus x(1), x(2), x(3) and
  pos3, i.e. the top-3 values and positions in full, which is what step 2 integrates over exactly. F3 (capacity
  check): all 10 order statistics and their positions, a lossless re-encoding of x.
- Regressor: LightGBM, cross-entropy objective on the soft label eps_A, num_leaves 63, learning rate 0.05,
  min_data_in_leaf 200, up to 3000 rounds with early stopping (100) on the validation split. Fixed in advance.
- Metrics on the test split. Primary, the systematic share: noise-corrected R^2 of the prediction against eps(x),
  R^2 = 1 - [MSE(pred, eps_B) - mean(eps_B (1 - eps_B)) / 15] / Cov(eps_A, eps_B). Also the Brier skill score on e
  against the training base rate, with its ceiling Cov(eps_A, eps_B) / (p (1 - p)), and the AUC on e.
- Excess over the algorithmic readout: the readout of the CL conditional means at step n (fitted on 2^17 separate
  base-model samples, as in `algrel.py`) defines e_alg(x). The same fits and metrics are run on the inputs with
  e_alg(x) = 0, with eps(x) as the target there.
- Stop rule: if the F2 share is below 0.30 on a model, step 2 is not run on it; if below 0.30 on both, stop and report
  (the coordinator's rule: a different idea is then needed, and that is Kelvin's call). F1 is reported alongside; if
  F1 < 0.30 <= F2 we go on and say so.
- Predictions. Y1: share (F2) 0.25-0.50 on T16 and 0.35-0.65 on W64. Y2: F1 is 0-0.10 below F2. Y3: the excess
  share on T16 is 0.10-0.35; on W64 within 0.05 of its total share. Y4: AUC (F2) 0.85-0.93 on T16 and 0.90-0.97 on
  W64. Y5: F3 reaches >= 0.60 on both; if not, the shares are limited by the regressor and are lower bounds.

**Step 2, deductive estimator TOP3 (`top3.py`).** Nothing in it is fitted to samples; it uses only the weights and
r of the model it is applied to.
- Exact law: the ordered positions (i, j, k) of the max, 2nd and 3rd max are uniform over the 720 triples and
  independent of the values. Coordinates (c_s, c_lo, t) uniform on (0,1)^3 map exactly to the values: s = x(3) by
  the inverse CDF of the 3rd order statistic of 10 N(0,1); given s, the top two are two iid N(0,1) truncated to
  (s, inf); with q_lo = 1 - sqrt(1 - c_lo) the quantile of the smaller and q_hi = q_lo + (1 - q_lo) t that of the
  larger. Boxes: KS equal intervals in c_s; KG intervals in c_lo and in t with edges (i/KG)^2, so near-ties of the
  2nd/3rd (c_lo near 0) and 1st/2nd (t near 0) are resolved. Box probabilities are exact; each box is represented by
  its centre in these coordinates.
- Per-node error model, from the weights alone: the representative input has the top three at positions (i, j, k)
  with the node's values and the other 7 inputs at E[Z | Z < s] = -phi(s)/Phi(s). One noiseless forward pass fixes
  the ReLU pattern. The logit differences D_c = l_j - l_c of the 3 competitors with the smallest D_c are linearised:
  their Jacobians with respect to the 7 lower inputs (each with variance Var(Z | Z < s), independent) and to the
  post-ReLU activation at every step (variance (sigma r_t,i)^2, the linear-response noise term) give Cov(D).
  P(error | node) = 1 - Phi2(z_a, z_b; rho_ab) Phi(z_c) for D ~ N(D(x0), Cov(D)), the two most dangerous competitors
  exactly (bivariate normal by Gauss-Legendre quadrature) and the third as independent. The estimate is the
  probability-weighted sum over cells and boxes. FLOPs counted analytically.
- Budgets (KS, KG): (1, 2), (2, 2), (2, 3), (3, 4), (4, 6), i.e. 4 to 144 nodes per cell.
- ARC's bar on the existing families (`results/estim_T16.json`, `results/estim_W64s001ck.json`: 16 models, base plus
  weight-perturbed copies with their recalibrated r, truths from 2^24 / 2^21 samples). Column (a): TOP3 on each
  model's own weights, MSE and bias against the truths at each budget. Column (b), hybrids, reported separately:
  (b1) control variate e(x) - beta (g(x) - E g), g(x) the TOP3 prediction of the box containing x, E g exact;
  (b2) stratified sampling over 8 strata of (cell, box) units grouped by predicted error, Neyman allocation with the
  predicted s.d. (floored at 0.02), exact sampling inside each box. N = 2^10, 2^12, 2^14, 2^16, 4 repetitions per
  model; FLOPs include the setup. Column (c): Monte Carlo, stratified (90 cells) and RQMC from the estim files.
  Compared with the ADF, the algorithm-indexed description (Amendment 6) and the oracle constant.
- Diagnostics (not estimators): TOP3 against sampled error rates per (pos1, pos2) cell and per box on the base
  models, to locate misses.
- Predictions. E7.1: at the finest budget TOP3 is within 30% (relative) of the base T16's error rate and within 40%
  of the base W64's. E7.2: column (a) loses to Monte Carlo at equal FLOPs at every budget on both models, by >= 10x.
  E7.3: on T16 column (a) beats the algorithm-indexed description (MSE 2.4e-3) and the oracle constant (9.6e-4); on
  W64 it does not beat the oracle constant (8.4e-5). E7.4: the best column-(b) estimator gains 1.5-4x over Monte
  Carlo per sequence on both models; including the setup it beats Monte Carlo at equal FLOPs only at N >= 2^14.
- Kill test K7: if TOP3 misses the base error rate by more than 50% (relative) on both models, the blocker is the
  linearised residual model given the algorithmic variables, not the closure in them; reported as such.

## Amendment 8 (2026-10-29 13:33Z): width sweep for "wider networks are sparser" (coordinator's request)

Seen before writing this: the DAG sizes in the report (T16 85%; W32 at sigma 0.01 75-80%; W64 at sigma 0.01 43% and
58%; W64 at sigma 0.03 60%; algorithm-relative 58%, 60%, 29-30%, 50%), the GPU models' training logs and criterion
results, and a CPU timing test. Nothing below has been run. CPU only, at most 4 processes, nice, one thread each.
- **Models.** Series S (new, CPU): `train_batched.py` with ARC's recipe compressed to 2^16 steps (NTRAIN = 2^26
  sequences, batch 1024, AdamW lr 0.04/sqrt(d), weight decay 0.001, decay to 0.01x at half the run), the warmup
  scaled with the run (2^10 steps, the same 1/64 of the run as ARC's 2^14 of 2^20). Widths 32, 64, 128 at sigma 0.01,
  seeds 0-2; noise control at width 64, sigma 0, seeds 0-2 (same initialisations and data streams as sigma 0.01);
  width 16 at sigma 0.01, seeds 0-2, if a slot is free. Width 256 is not cheap on CPU (about 4 s per step for three
  seeds) and is dropped. Series L (existing, GPU, ARC's full 2^20 steps): widths 16 (seeds 0-1), 32 and 64 (seeds
  0-2) at sigma 0.01, and width 16 at sigma 0 (seeds 0-1). Series S matches training across widths; series L checks
  the trend at full length.
- **Criterion.** Amendment 1's strict criterion against ARC's example (loss <= 1.02 x 0.1470, accuracy not lower by
  more than 2 s.e.), from the paired evaluation at the end of `train_batched.py` (2^22 sequences). Only passing models
  are analysed; failing widths are reported with their numbers.
- **Measures per passing model (`sweep_an.py`).** Edge set as in `dag.py` (inputs, recurrent edges at steps 1..n-1,
  readout units). Held-out target: flip <= 1% on 2^15 test inputs with paired noise; the size is the smallest prefix
  of the saliency ranking found by binary search on 2^13 fit inputs, stepped up until the held-out target is met.
  M1: fraction of edges with mean ablation. M2: the same with resample ablation (an ablated edge carries the source
  unit's clean value on another random input; saliency computed with those deviations). M3: algorithm-relative
  ablation (conditional means from the CL regression on 2^17 base samples, as in `algrel.py`; input edges kept),
  reported as a fraction of all edges. M4: the smallest uniform number of question directions per step (projections
  of `lowrank.py`) with held-out flip <= 1%, as a fraction of d. M5: the share of edges whose mean ablation alone
  flips >= 0.1% of answers (8,192 inputs, paired noise), from 1,000 uniformly sampled edges per model.
- **Report.** Each measure against width, per seed (median and range); series S and L separately; the noise control
  (width 64, sigma 0 against 0.01, paired by seed).
- **Predictions.** W8.1: in series S, width 128 and width 64 pass in all seeds at sigma 0.01; width 32 fails in at
  least 2 of 3 seeds (short training); width 16 fails in all. W8.2: M1 falls with width among passing models
  (median at 128 below 64 below 32, where available), with width 128 at <= 40%. W8.3: M2 is at least M1 for each
  model and falls with width in the same order. W8.4: M3 and M4 fall with width. W8.5: M5 falls with width. W8.6
  (noise control): at width 64, sigma 0 needs >= 10 points more edges (M1) than sigma 0.01 in at least 2 of 3 seeds.

## Amendment 9 (2026-10-29 14:03Z): redundancy, width against noise, search stability (coordinator, after the literature check)

Seen before writing this: the literature note (`outbox/lit-width-sparsity-2026-10-29.md`), the series-S training
logs up to 2^15 steps, and the Amendment 8 measures on one series-L model (width 32, seed 0: M1 75%, M2 77%, M3 54%,
M4 21 of 32, M5 57%). No other sweep result. All analysis runs below use `sweep_an.py` as extended here; the width-32
seed-0 analysis is rerun with it.
- **Redundancy test.** After M1, forbid its kept set: ablate exactly those edges and keep all others. If that meets
  the held-out 1% target, a disjoint DAG exists; its size is then found as for M1 (saliency order restricted to the
  complement). Reported: whether it exists, its share and count, and the flip with the first set ablated.
- **Width against noise.** Noiseless twins (sigma 0, same initialisation and data as sigma 0.01) at widths 16, 32, 64
  and 128 in series S, seeds 0-2. ARC's noiseless zoo models 16x10 and 32x10 (five seeds each, trained by ARC on 2^30
  sequences) as given. The controls are analysed whether or not they pass the criterion (their loss and pass status
  are reported), because the comparison is at matched loss: every measure is tabulated against loss, and each noisy
  model is compared with the noiseless model of closest loss.
- **Search stability.** M1 repeated on two more fit samples (other seeds); pairwise Jaccard overlap of the kept sets
  and the spread of their sizes. Random-edge baseline: M1's search with one random edge order instead of saliency.
- **Counts.** Every DAG is reported as a share and as an absolute number of kept edges.
- **Predictions.** R9.1: a disjoint DAG exists for the sigma 0.01 models at widths 64 and 128 in every seed, at width
  32 in at most 1 of 3 seeds, and for no width-16 model (twins or ARC's). R9.2: at matched loss, width matters more
  than noise: each noiseless twin at width 64 and 128 is within 10 points (M1) of its sigma 0.01 sibling, and ARC's
  32x10 models have lower median M1 than ARC's 16x10 models. R9.3: repeat searches overlap with Jaccard >= 0.8 at
  every width; the random order needs >= 95% of edges at every width. R9.4: the absolute M1 count grows with width
  (median at 128 above 64 above 32).

## Amendment 10 (2026-10-29 14:58Z): scope cut for time, before any series-S analysis result (the first started 14:57Z)

The host is loaded and CPU training ran at about half the timed speed. To stay within the time box: noiseless twins
only at width 64 (seeds 0-2) and width 128 (seed 0); no twins at widths 16 and 32, and no series-S width 16. The
noiseless controls at widths 16 and 32 are ARC's zoo models (five seeds each) and the series-L width-16 sigma 0
model. Predictions that name the dropped runs are scored on what remains, and say so.
