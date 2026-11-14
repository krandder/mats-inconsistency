# PREREG: tiny PFNs that approximate exact Bayes on a minimal reliability world (tiny-pfn, 2026-10-27)

Written before any trained result. Purpose: models small enough to understand completely (~2k-20k parameters) that
get close to exact Bayes, plus a map of where they deviate. Interpretability itself is out of scope here.
Parent: ../pfn-dedup/ (world.py, pfn.py, test_world.py copied and adapted; nothing there is modified).

## Design history (all before any code was trained)
- 2026-10-26 ~23:25Z: task as given: S = 3 sources (5 partitions, shared-error groups), Q = 5, duplicates w.p. 0.3.
- 2026-10-26 ~23:30Z, Kelvin via coordinator: duplicate probability 0 (no copied rows).
- 2026-10-26 ~23:35Z, Kelvin via coordinator: no shared-error groups either. Three INDEPENDENT sources with unknown
  reliability (below). Keys arm dropped (no groups); its runs go to seeds 3 and 4.
No run was started on either earlier world, so nothing was discarded.

## World (world.py)
- Episode: Q = 5 questions (4 labelled, 1 query), S = 3 sources. y_q ~ N(0, 1) iid.
- Reading o_{s,q} = y_q + e_{s,q}, e ~ N(0, v_s); v_s iid uniform on {0.1, 0.5, 2.0} per source per episode, the same
  for all questions of the episode.
- Labelled questions show y_q; the query question shows only the three readings.

## Tokens (parent format, no-keys arm)
One token per (source, question) = 15 context tokens: [source-identity vector (N(0,1)/4 in R^16, fresh per episode),
question-identity vector (same), value o, label y (0 if missing), missing flag, 0]. One query token: [0, question
identity of q*, 0, 0, 0, query flag 1]. Context tokens attend to all context tokens; the query token attends to the
context tokens and itself; nothing attends to the query token. Output from the query token: Gaussian mean and
log-std (clamped to [-6, 3]); loss = NLL of y_{q*}.

## Exact references for y_{q*} (numpy, test set)
- (a) equal weights: every source at the prior-mean variance 0.8667; Gaussian.
- (b) oracle precision weighting with the true v_s; Gaussian. Not attainable (v is hidden); a lower bound.
- Bayes: posterior over the 27 variance combinations, weights = prior x prod_s p(labelled residuals of s | v_s)
  x N(query readings; 0, diag(v) + 11^T), then the mixture of the 27 Gaussian predictives. Note: the labelled part
  factorises over sources, the query-readings marginal couples them; both are included.
- (lab) like Bayes, but the weights use the labelled residuals only (factorised over sources). Diagnostic reference:
  tells whether a model uses the information the query readings carry about the reliabilities.
- Gaussian floor: per episode KL(Bayes mixture || its moment-matched Gaussian). A Gaussian head cannot do better.
Unit tests (test_world.py, must pass before training): Bayes with a point-mass prior on the true combination equals
(b); Bayes equals an independent full-joint-Gaussian enumeration (no factorisation); Bayes matches Monte Carlo (prior
sampling of the variances + generic joint-Gaussian conditioning, 2e5 draws, tol 0.01); calibration on 40000 episodes
from an independent sampler (PIT uniform, E[(y*-mean)^2] = E[var], slope 1, per-source reliability posterior
calibrated, ordering NLL(b) < NLL(Bayes) < NLL(lab), NLL(a)); reductions of (a)/(b); token layout.

## Models (pfn.py)
Parent block: pre-norm, LayerNorm -> multi-head attention (no positional encoding) -> residual, LayerNorm -> MLP
(GELU) -> residual; final LayerNorm on the query token, linear head to (mean, log-std).
Grid: width {16, 32} x heads {1, 2} x depth {1, 2}, MLP 2x width; plus attention-only (no MLP, no ln2) at width 32,
depth 2, 2 heads. 9 configs x seeds 0-4 = 45 runs. Parameter counts (exact, from n_params): w16 L1 2882; w16 L2 5106;
w32 L1 9858; w32 L2 18402; w32 L2 attention-only 9890 (heads do not change the count).

## Training
CPU only, OMP_NUM_THREADS=1 per process, 8 processes in parallel (16 cores; half left free).
Fresh episodes every batch (batch 256), never repeated; AdamW lr 1e-3, weight decay 0.01, grad-clip 1.0, linear
warmup 1000 steps, then constant. Validation: 4096 fixed episodes (seed 10000) every 1000 steps.
Convergence rule (parent): at step 40000, converged if the best validation NLL improved by < 0.003 nats over the
last 25% of steps (30000-40000) relative to the best before; otherwise extend the constant phase once to 80000 and
apply the rule again (60000-80000). A run that fails at 80000 is UNCONVERGED and reported as such.
Deviation from the parent, decided now: after the constant phase, a 10000-step linear cooldown of the lr to 0
(the 0.02-nat target is below the lr-noise level of constant-lr training). The convergence flag refers to the
constant phase. The reported checkpoint is the best validation checkpoint over the whole run; the best checkpoint of
the constant phase is also saved and evaluated (shows what the cooldown buys).

## Metrics (fixed test set, n = 16000, seed 20000, disjoint from validation and training streams)
- Per run: NLL gap to Bayes (paired, per episode) and the exact expected gap KL(Bayes || model) per episode
  (= Gaussian floor + mean-mismatch term + scale-mismatch term). Per config: mean +- se over the 5 seeds.
- Splits: reliability pattern of the episode: all three variances equal / exactly two equal / all different; and
  number of reliable (v = 0.1) sources 0/1/2/3. (Redundancy splits of the original task no longer exist.)
- Closest reference: mean squared distance of the model's predictive mean to (a), (b), Bayes, (lab) means; NLL gaps of
  the references themselves.
- Smallest good config: the config with the fewest parameters whose mean (over seeds) test NLL gap to Bayes is
  <= 0.02 nats; tie on parameters -> smaller gap. Reported with the KL estimate alongside.
- Residual map of that config (pooled over its seeds, also per seed): per-episode KL(Bayes || model) and its
  decomposition, binned (quintiles unless stated) by Bayes sufficient statistics and their functions:
  true variance multiset (10 types); entropy of the 27-way Bayes posterior; per-source labelled sum of squared
  residuals (min and max over sources) i.e. how ambiguous each source's reliability is; query coupling
  KL(w_Bayes || w_lab) (how much the query readings move the reliability posterior); Gaussian floor; spread of the
  three query readings (max - min). Report mean KL per bin, share of total excess, Spearman correlation per statistic.
- Every run is reported, converged or not.

## Outputs for the interpretability step
runs/<cfg>_s<seed>/: model.pt (best), model_phase1.pt, meta.json (curve, convergence), eval.json, test_preds.npz
(per-episode mu, log-std, NLL, KL and its parts). runs/testset.npz: tokens, raw world variables, all reference
predictions, Bayes mixture components and weights, the labelled-only per-source posteriors, sufficient statistics.

## Budget
CPU only; no GPU, no paid services. Code sha256 in code_sha256.txt (written with this file).

## Amendment 1 (2026-10-26 23:50Z; written before any trained result; the grid launched at ~23:45Z was stopped while
## it was still building the test set, before any training step)
Kelvin, via coordinator: add micro transformers, target ~100-300 parameters, run first.
- Micro grid: width {4, 8}, 1 head, 1 layer; with MLP (2x width) and attention-only; seeds 0-4 (20 runs). Same
  training, convergence rule, cooldown, test protocol.
- Token change for these arms: source/question identity vectors in R^4 instead of R^16 (feature dim 12 instead of 36),
  so the input embedding does not dominate. World draws are identical (the test set's episodes are the same; only the
  random identity vectors differ in dimension).
- Exact parameter counts, ALL parameters including the input embedding (n_params):
  w4 MLP 242; w4 attention-only 158; w8 MLP 738; w8 attention-only 442. (Only width 4 is inside 100-300.)
- Main grid keeps R^16 (as pre-registered). Control for "does R^4 hurt": w16_h2_L2 and w32_h2_L2 also at R^4, seeds
  0-4 (10 runs); compared with the same configs at R^16. Main-grid counts at R^4 would be: w16 L1 2498, w16 L2 4722,
  w32 L1 9090, w32 L2 17634, w32 L2 attention-only 9122.
- For each micro config: whether it beats (a) equal weights and (b) oracle precision weighting (mean-over-seeds test
  NLL below the reference's, and the paired per-seed differences).
- Added diagnostic reference (qonly): Bayes with combination weights from the query readings only (ignores the
  labelled questions) = the best possible predictor that does not use the labelled questions. Relevant for 1-layer
  models, where the query token reads only first-layer values and per-source reliability must be bound to the
  source's query reading inside one attention step. Weak prediction (not a rule): 1-layer models sit near or above
  qonly.
- Run order: micro arms, then the main grid seed-major, then the R^4 controls; 8 processes in parallel.
Code hashes after this amendment replace the earlier ones in code_sha256.txt (earlier ones kept below them).

## Amendment 2 (2026-10-27 03:05Z; written before any follow-up result; coordinator request after the first report)
Three follow-ups, CPU only, at most 6 processes in parallel (local dedup evals share the machine). Same world, test set
(seed 20000, n = 16000), validation set, optimiser, warmup, lr and cooldown as above unless stated.
1. 2-layer micro arms: width {4, 8}, 1 head, depth 2, R^4 identity vectors, with MLP (2x width) and attention-only;
   seeds 0-4 (20 runs); standard convergence rule (40k, extend once to 80k) + 10k cooldown. Exact parameter counts:
   w4 MLP 414; w4 attention-only 246; w8 MLP 1338; w8 attention-only 746. Report: NLL gap, KL, whether each beats
   (a), (b), query-only Bayes and lab-only Bayes (per seed and mean), closest reference.
2. K = 4 Gaussian-mixture output head (logits, means, log-stds; log-stds clamped to [-6, 3] like the Gaussian head;
   training loss = mixture NLL) on w16_h1_L2 (R^16, 5276 params), w16_h2_L2 at R^4 (4892 params) and w32_h2_L2
   (R^16, 18732 params); seeds 0-4 (15 runs); standard convergence rule + cooldown. KL(Bayes || model) per episode by
   trapezoid quadrature on a per-episode grid of 2001 points spanning +-9 sd of every Bayes and model component
   (checked before any training: equals the analytic Gaussian KL to 4e-14; KL(Bayes||Bayes) = 5e-16; a mixture of
   identical components equals the Gaussian). The Gaussian floor does not apply (floor = 0).
   Selection: smallest mixture-head config (parameters) with mean-over-seeds test NLL gap <= 0.02; KL alongside.
   Residual map of that config (or, if none qualifies, of the best mixture config): per-episode KL binned by the same
   statistics as before (analyze2.py), plus the model's effective number of components and how its between-component
   variance tracks Bayes's.
3. Longer training, Gaussian head, same three configs (w16_h1_L2, w16_h2_L2_ds4, w32_h2_L2), seeds 0-4 (15 runs):
   phase 1 at constant lr FIXED at 160k steps, then the same 10k cooldown. Interpretation of "up to 160k": the
   convergence rule stopped about half of the original seeds at 40k, so applying it again would mostly rerun the
   original arm with a new data stream and could not test whether training length limits the gap; the phase is
   therefore fixed, and the rule's verdicts at 40k, 80k and 160k are recorded (the "converged" flag = rule at 160k).
   Runs of 2 and 3 use new training-data streams (the stream seed includes the config name with suffix _mix4 / _long).
Code: pfn.py gains the mixture head, kl_grid, --phase1_fixed; Gaussian-head behaviour unchanged (re-evaluating the
original w32_h2_L2 seed-0 checkpoint reproduces its eval.json to 2e-9). New: run_followup.sh, analyze2.py. The
pfn.py used for the main grid is kept as pfn_amend1.py.

## Amendment 3 (2026-10-27 06:55Z; before any warm-start result; coordinator request after the follow-up report)
Seen before writing: the follow-up results (4/15 cold-started mixture runs stuck at test gap ~0.144).
1. Warm-started K = 4 mixture head, configs w16_h2_L2 at R^4 (4892 params) and w32_h2_L2 at R^16 (18732 params),
   seeds 0-4 (10 runs). Source per seed: that seed's Gaussian-head checkpoint from the 160k-step arm
   (runs/<cfg>_long_s<seed>/model.pt; convergence rule passed at 160k for all w32 seeds and for w16_ds4 seeds
   0, 2, 3, 4; seed 1 did not pass and is used anyway, flagged). Body weights copied; mixture head initialised around
   the Gaussian output: component means = Gaussian mean weights (+ N(0, 1e-3) jitter) with bias offsets
   -0.15, -0.05, +0.05, +0.15; component log-stds = Gaussian log-std; equal mixture weights. Initial validation NLL is
   about 0.004 nats above the source's (checked on seed 0 of both configs before any warm-start training).
   Training: fresh AdamW, the same schedule as every other run (1000-step warmup to 1e-3, constant, convergence rule
   at 40k / extend once to 80k, 10k cooldown), new data stream (config suffix _mix4_ws), best-validation checkpoint.
   Evaluation as Amendment 2 (KL by quadrature). Rule (unchanged, all seeds counted): a config qualifies if its
   mean-over-5-seeds test NLL gap <= 0.02; also reported: whether every seed is below 0.02.
2. Plateau diagnostic (no new training), on every run's validation curve (every 1000 steps; gap = val NLL minus
   validation-set Bayes NLL): "on the query-only plateau" = at least 10 consecutive evaluations (>= 10k steps)
   with 0.14 <= gap <= 0.18. Per run: ever on the plateau; entry step (first evaluation of the first such stretch);
   escape step (first evaluation after the last such stretch with gap < 0.14; "never" if none). Reported per config
   and pooled as the distribution of escape steps. Caveat fixed in advance: for 1-layer models the band contains
   their final capacity-limited level, so "never escaped" there is not an optimisation failure; reported separately.
Code: pfn.py gains warm_start/--init_from (no change to other paths); new run_ws.sh, plateau.py, analyze3.py.
