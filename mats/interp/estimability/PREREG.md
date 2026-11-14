# PREREG: training a small transformer with an estimability penalty (estimability-penalty, 2026-10-27)

Written before any penalty-trained result exists. Coordinator: opus-mats. Question (Kelvin): can a small deep
transformer be trained with an added loss on the gap between layer-wise propagated cumulants (a deterministic,
mechanistic estimate in the style of ARC's cumulant propagation) and the sampled cumulants, without degrading task
performance? K = 4 main, K = 2 alongside.

The only trained runs before this file: one CPU pilot with lambda = 0 (depth 4, seed 0, 3000 steps) to choose the step
budget (validation gap to Bayes 0.013 nats at 3000 steps, flat after ~1250), and 4-step CPU smoke tests of the
penalty code. Neither is part of the results.

## World (copy of pfn-dedup/world.py, shrunk; pfn-dedup/ is not modified)
o_{s,q} = y_q + z_{g(s),q} + e_{s,q}, y ~ N(0,1), z ~ N(0,1) shared by a group, e ~ N(0,0.25); partition prior
0.7 Ewens(1) + 0.3 all-singletons. S = 3 sources, Q = 3 questions (2 labelled + query), identity embeddings of dim
8; tokens = one per (source, question) + one query token, T = 10, 20 features. Fresh episodes every batch.
Exact Bayes predictive (mixture over the 5 partitions) gives the reference NLL.
Deviations from the coordinator's design, with reasons:
1. No duplicate rows. Duplicates make the attention mask random (a variable number of valid tokens); the input law
   would be a mixture over masks. Without them the mask is fixed. Shared errors (the partition) remain the
   inferential problem; Bayes still has to infer the partition from labelled residuals.
2. Input cumulants are exact, not from a large sample: given the partition the input (tokens and y*) is exactly
   Gaussian, so the input law is a 5-component Gaussian mixture. Each partition shape is propagated separately and
   mixed exactly at read-out (law of total cumulance; the network and the propagator are permutation-equivariant,
   so each orbit's other members are token permutations). Exact input cumulants make the propagator stronger, not
   weaker; test_input_law_matches_sampler checks them against a 200k sample.

## Model
Pre-norm transformer, width d = 32, 2 heads, GELU MLP 4d, LayerNorm (affine), softmax attention, residuals, no
positional encoding; context tokens attend to the context, the query token to the context and itself. Head on the
query token: mean mu and log-std logs (no clamp); loss = Gaussian NLL of y*. Depth 8 (main) and 4 (contrast).
PyTorch default initialisation. AdamW lr 1e-3, weight decay 0.01, warm-up 300 steps then cosine to 10%, batch 512,
3000 steps, grad-norm clip 1. Same init and same data stream for all arms of a (depth, seed).

## Propagator (prop.py; deterministic, differentiable)
State per mixture component, over R = [residual stream (320 coords); y*]: mean, full joint covariance, and (K=4)
higher cumulants in source form kappa_r = sum_j c_r[j] a_j^(x r), r = 3, 4: a source is created at every scalar output
of a non-linear op (its marginal kappa_3, kappa_4 not explained by transported sources), transported exactly by linear
maps and by linear response through non-linear ops. Each layer's state is computed from the previous propagated
state only (errors compound); MC enters nowhere.
Per op:
- linear / residual add: exact (mean, covariance, sources).
- GELU: marginal moments by Gauss-Hermite (48 nodes) under a Gaussian (K=2) or Edgeworth (K=4) marginal;
  off-diagonal covariance by the Hermite (Mehler) series to 3rd order; cross-covariances by Stein (linear response).
- LayerNorm: within a token exact for Gaussian input (r^k as a Laplace integral, tilted Gaussians, 32-node
  Gauss-Laguerre); between tokens exact for Gaussian input via a 2-D Laplace integral per token pair (6x6 nodes);
  to the residual by the exact mean Jacobian (Stein). K=4: new sources from the exact marginal kappa_3/4.
- attention scores q.k/sqrt(dh): exact mean and covariance for Gaussian (q,k) (Isserlis); K=4 sources from the exact
  Gaussian quadratic-form cumulants.
- softmax: a_ts = sigmoid(-l_ts), l_ts = logsumexp_{w != s}(s_tw) - s_ts by a Clark-style pairwise reduction
  (each step a 1-D Gauss-Hermite moment match, all running maxima kept jointly Gaussian with every score, value and
  residual coordinate, full covariance between all running l's); then the sigmoid by Gauss-Hermite.
- a.v: exact for jointly Gaussian (l, v) by Stein's lemma twice + bivariate Gauss-Hermite, plus the Wick third
  cumulant kappa3(v, l, v') and kappa3(l, v, r) carried from the bilinear scores through Clark's linear coefficients.
- read-out: E[mu], E[logs] linear; E[nll] in closed form from the joint (mu, logs, y*) (CGF to 4th order if K=4).
- robustness: the closures do not guarantee a positive semi-definite covariance (on a trained lambda=0 CPU-pilot
  model the residual covariance after the first attention had a negative eigenvalue, and the next LayerNorm
  integral diverged). Every residual-stream covariance after a residual add and every LayerNorm-output covariance
  is projected onto the PSD cone (eigenvalue clipping; straight-through in the backward pass); the clipped negative
  mass relative to the trace is logged (3% at most on that pilot model). Hermite coefficients of elementwise
  functions are computed as E[f^(k)] with analytic derivatives (equal by Stein, stable for near-constant inputs).
Truncations (stated exactly): cumulants of order >= 5 are zero; joint 3rd/4th cumulants are only the source-form
ones (one-vertex creation + linear transport; K=2 has none); multi-vertex terms are dropped except those named above
(LayerNorm pairs, softmax via Clark, a.v Stein terms, kappa3 through scores); the softmax treats the running
log-sum-exp as Gaussian at each step; LayerNorm's response to incoming non-Gaussianity is linear only; GELU
off-diagonal covariances ignore incoming non-Gaussianity; 4th joint cumulants of LayerNorm outputs across
tokens/features are not represented (this is the largest known error source: see op tests).
Op-level tests (test_prop.py, MC on small dims) all pass; their printed table is the op accuracy record.

## Penalty (train.py)
loss = task NLL + lambda * mean over the 2L residual points (after every attention and MLP sub-block) of
D = e_mean + e_cov (+ e_skew + e_kurt if K = 4):
  e_mean = mean_i (mu_p - mu_e)^2 / v_i;  e_cov = ||D^-1/2 (S_p - S_e) D^-1/2||_F^2 / ||D^-1/2 S_e D^-1/2||_F^2;
  e_skew = mean_i (k3_p - k3_e)^2 / v_i^3;  e_kurt = mean_i (k4_p - k4_e)^2 / v_i^4,
p = propagated (mixture), e = batch-empirical; v, D from the batch (not detached), so every term is invariant to
rescaling the activations and cannot be met by shrinking them. Each square is estimated as (p - e_half1)(p - e_half2)
over the two batch halves, so the sampling-noise floor does not bias the gradient. Gradients flow through both
the propagator and the batch statistics. Penalty at init (depth 2, measured): ~0.07, mostly skew/kurtosis.

## Arms
depth {8, 4} x seeds {0, 1, 2} x: lambda = 0; K = 4 with lambda in {0.3, 3, 30} (penalty ~0.02, 0.2, 2 nats-equivalent
at init: weak, middle, strong); K = 2 with lambda = 3. 30 runs. Plus a random-features baseline per (depth, seed):
body frozen at init, only final LN and head trained (6 runs).
Budget rule: if the GPU pilot shows the K=4 depth-8 penalty step makes the sweep exceed the $15 cap, the penalty is
evaluated every k-th step with weight k*lambda (k the smallest of 2, 4, 8 that fits), for all penalty arms, recorded
as a timestamped amendment before any penalty result.

## Metrics (evaluate.py; every trained model, the random-features models and the untrained inits)
1. Task: NLL on 65536 fresh test episodes (seed 20000), gap to exact Bayes (paired), se.
2. Estimation (ARC-style): K=2 and K=4 propagated E[mu], E[logs], E[nll] vs Monte Carlo with 2^22 episodes;
   nse = (est - truth)^2 / Var(output) (= 1/N for N-sample MC), N_eff = 1/nse; per-point discrepancies.
3. Matched compute: N_match = propagator FLOPs / forward FLOPs per episode (torch FlopCounter, matmul-type ops);
   sampling at matched compute has nse = 1/N_match.
4. How the penalty was met: fraction of GELU pre-activations with |z| < 1.5, attention entropy / log(#keys), rms
   standardised skew and excess kurtosis of residual coordinates, LN-input norm CV, relative weight change from init,
   task gap of the random-features baseline.

## Thresholds and verdict (per depth; depth 8 is the headline)
- Delta = gap(arm) - gap(lambda=0), paired by seed. No degradation: mean Delta <= max(2 * se(Delta), 0.005 nats),
  se from the 3 paired differences.
- E_err = geometric mean over {mu, logs, nll} of the K=4 nse, geometric mean over seeds. Estimability improved:
  E_err(lambda=0) / E_err(arm) >= 5. (nse below 4/2^22 is reported as at the MC floor.)
- SUCCESS: some penalty arm meets both. TRADE-OFF: some arm improves >= 5x but every such arm degrades. NO GAIN: no
  arm improves >= 5x. The K=2 arm is judged on the same criteria with the K=2 error for comparison.
Predictions: the penalty is met mostly by making the network more predictable for the closure: higher attention
entropy, fewer curved GELU units, more Gaussian residual marginals; strong lambda drifts toward the random-features
baseline's loss. K=4 and K=2 propagators give similar errors (op tests show the source form adds little).

## Compute
One GCP L4 via run_on_l4.sh, hard cap $15 total; every VM deleted and checked with gcloud compute instances list.
A first spot VM for a timing pilot was preempted during boot (no output, <= $0.30 at the ceiling rate). The main job
(gpu_sweep.sh, on-demand) therefore starts with the timing measurement (8 penalty steps per (depth, K) from init)
and applies the budget rule above automatically: k = smallest of 1, 2, 4, 8 for which the projected wall time is
<= 4 h on the 6 h VM; k is written to pen_every.txt before training starts. Then: op tests, init validation
(seeds 0-2, qk-stress 1 and 2, depths 1/2/4/8, MC 2^20), the 36 runs (3 in parallel), evaluation, summary.
Code frozen: sha256 of every file in code_sha256.txt (recorded before the job starts).

## Amendment 1 (2026-10-27 01:20Z, before any penalty-trained result)
The first on-demand job (estsweep-1027b) measured the penalty step from init: the K=4 depth-8 step ran out of GPU
memory (22 GB) and the K=2/depth-4 steps took ~2-4 s (many small kernels). Changes, none affecting any number:
(a) per-block gradient checkpointing in the propagator (loss and gradients verified identical, max |diff| = 0);
(b) the constant selector matrices of the Clark step are cached; (c) the sweep is split into one VM per depth
(DEPTHS=8 and DEPTHS=4, run concurrently), with the timing rule applied per job and timing measured as the
difference between the step-3 and step-6 records (excludes warm-up); a failed timing defaults to k=8.
At ~2-4 s per penalty step the preregistered rule will give k=8 (penalty every 8th step, weight 8*lambda); the
value actually used is in pen_every.txt of each job. The init validation comes from estsweep-1027b (same frozen
propagator numerics). With Adam, an every-8th-step penalty of weight 8*lambda is not equivalent to lambda every step;
the lambda grid (100x span) is kept and the effective strength is read off the measured penalty values.

## Amendment 2 (2026-10-27 01:25Z, before any penalty-trained result)
Measured penalty step (from init, single process): depth 8: 7.6 s (K=4), 6.1 s (K=2); depth 4: 3.4 s / 3.0 s; the rule
gives k=8 for both depths (pen_every.txt). With the GPU saturated (97% with two processes) the depth-8 job would take
~9 GPU-hours, over the 6 h VM limit. The depth-8 penalty arms of seeds 1 and 2 are therefore run on two extra VMs
(SEEDS=1 / SEEDS=2, ONLY_PEN=1, PEN_EVERY_FIXED=8); on the first depth-8 VM those eight runs are marked as moved and
skipped. Same code, same k, same data streams; results are merged locally. Projected total spend ~$14 at the $0.87/h
ceiling rate.
