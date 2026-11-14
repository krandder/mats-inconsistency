# Estimability penalty on a small transformer (2026-10-27)

Work dir: /home/agent/workspaces/mats/work/estimability/ (PREREG.md with two pre-result amendments; code_sha256.txt;
results/summary.md; results/runs/*/eval.json; results/validate_init/; results/test_prop.txt).

## Question
Kelvin's question: can a small deep transformer be trained with an extra loss that penalises the gap between layer-wise
propagated cumulants and the sampled cumulants of the same activations, without degrading the task? The propagated
cumulants are a deterministic estimate over the input distribution, in the style of ARC's cumulant propagation; the
main arm uses K = 4, with K = 2 alongside. The outcome to test against is certified training, where tight
propagation costs clean accuracy.

## Setup
Task: the pfn-dedup source world, copied (pfn-dedup/ untouched) and shrunk to 3 sources and 3 questions (2 labelled
plus the query), with no duplicate rows. That gives 10 tokens and an exact Bayes predictive over 5 partitions. Fresh
episodes every batch.

Model: pre-norm transformer, width 32, 2 heads, GELU MLP 4x, LayerNorm, softmax attention, depth 8 and depth 4.
Training: 3000 steps, batch 512.

Input law: given the partition, the input is exactly Gaussian, so each partition shape is propagated separately and
mixed exactly. The brief asked for input cumulants from a large sample; exact ones are strictly better.

Penalty: the mean over all residual points (after every attention and MLP sub-block) of four scale-free squared
discrepancies between propagated and batch statistics: mean, full covariance, marginal skew and marginal excess
kurtosis. The K = 2 arm uses mean and covariance only. Each square is estimated from two batch halves, so sampling
noise does not bias the gradient.

Arms: lambda in {0, 0.3, 3, 30} with K = 4; lambda = 3 with K = 2; a random-features baseline (body frozen at init);
3 seeds each; same init and data stream across arms.

Penalty schedule: a penalty step costs 3-8 s on an L4, so the preregistered budget rule applied the penalty every 8th
step with weight 8*lambda.

## What the propagator keeps and drops
Kept, per partition component:
- the mean and the full joint covariance of the residual stream and y*;
- for K = 4, joint 3rd and 4th cumulants in "source" form: each non-linear unit creates a source carrying its marginal
  kappa3/kappa4; linear maps carry sources exactly and non-linear ops carry them by linear response.

Exact for Gaussian inputs:
- linear maps;
- GELU marginals;
- score moments;
- LayerNorm within a token and between token pairs (a Laplace-integral form, needed because the simple product
  closure was 10-60% off).

Approximated:
- softmax, by a Clark-style pairwise log-sum-exp reduction with a full joint covariance;
- the a·v product: exact for jointly Gaussian (l, v) by Stein's lemma, plus the Wick third cumulant that the bilinear
  scores create.

Dropped:
- order >= 5;
- every joint 3rd/4th cumulant the source form cannot represent. The largest is LayerNorm's cross-feature and
  cross-token non-Gaussianity (its outputs have fixed norm);
- the remaining multi-vertex terms.

Covariances are projected onto the PSD cone after each residual add and each LayerNorm. This was needed once the
attention closure became inconsistent on trained models.

Op tests against Monte Carlo all pass (results/test_prop.txt):
- LayerNorm, GELU and linear ops are within about 1-2% on covariance.
- Attention is within about 2-5% at small score variance and 5-15% at larger score variance.

Full ARC K = 4 is not tractable here. With N = 320 coordinates, one 4th-order tensor has 10^10 entries, and the
factorised form needs N^3 per tensor plus diagram rules for softmax and LayerNorm that do not exist.

## Validation at init (before training; 3 seeds, MC 2^20)
nse = (estimate - truth)^2 / Var(output), so N_eff = 1/nse is the number of samples that would give the same error.
Values are the geometric mean of nse over E[mu], E[log sigma] and E[NLL].

Default init, depth 1 / 2 / 4 / 8:
- N_eff, K = 2: 24k / 2.9k / 2.2k / 690
- N_eff, K = 4: 27k / 2.8k / 2.2k / 700
- final-layer covariance error (relative Frobenius, correlation units): 0.07 / 0.10 / 0.17 / 0.21

With query/key weights doubled, N_eff is 1.0k / 7.5k / 1.7k / 730 and the covariance error 0.20-0.30.

Error compounds with depth. K = 4 is no better than K = 2. Neither predicts the residual stream's marginal skew or
kurtosis: the error equals the observed rms.

A propagation costs as much as N_match forward passes (K = 2: 38k; K = 4: 66k at depth 4, 94k at depth 8). So even at
init the estimate loses to sampling at matched compute, consistent with ARC's small-width and depth results.

## Main results (mean ± se over 3 seeds)
Columns:
- gap: test NLL minus exact Bayes NLL, nats, on 65536 episodes;
- Delta: change in gap vs lambda = 0, paired by seed;
- E_err: geometric mean of nse over the three expectations (MC truth 2^22);
- x: E_err(lambda=0) / E_err(arm), K = 4 propagator;
- nse(E[NLL]): K = 4 propagator.

Depth 8:

| arm | gap | Delta | E_err K=2 | E_err K=4 | x | nse(E[NLL]) |
|---|---|---|---|---|---|---|
| untrained init | - | - | 1.5e-3 | 1.4e-3 | - | 8.7e-4 |
| lambda=0 | 0.0109±0.0005 | 0 | 2.3e-3 | 2.2e-3 | 1 | 1.2e-2 |
| lambda=0.3, K=4 | 0.0136±0.0007 | +0.0027±0.0002 | 2.5e-4 | 2.4e-4 | 9 | 5.5e-4 |
| lambda=3, K=4 | 0.0146±0.0003 | +0.0037±0.0002 | 2.2e-5 | 3.0e-5 | 74 | 3.9e-4 |
| lambda=30, K=4 | 0.0145±0.0004 | +0.0036±0.0002 | 4.8e-5 | 5.0e-5 | 45 | 3.7e-4 |
| lambda=3, K=2 | 0.0136±0.0002 | +0.0027±0.0003 | 1.6e-4 | 1.6e-4 | 14 | 5.8e-4 |
| random features | 0.306±0.007 | +0.295 | 6.5e-4 | 6.1e-4 | 4 | 3.8e-3 |

Depth 4:

| arm | gap | Delta | E_err K=2 | E_err K=4 | x | nse(E[NLL]) |
|---|---|---|---|---|---|---|
| untrained init | - | - | 3.7e-4 | 4.1e-4 | - | 8.8e-5 |
| lambda=0 | 0.0108±0.0003 | 0 | 3.9e-3 | 3.9e-3 | 1 | 1.2e-2 |
| lambda=0.3, K=4 | 0.0136±0.0003 | +0.0028±0.0001 | 1.3e-4 | 1.3e-4 | 30 | 7.2e-4 |
| lambda=3, K=4 | 0.0148±0.0004 | +0.0040±0.0002 | 2.8e-5 | 2.0e-5 | 191 | 6.3e-4 |
| lambda=30, K=4 | 0.0149±0.0003 | +0.0041±0.0001 | 4.8e-5 | 4.6e-5 | 83 | 6.0e-4 |
| lambda=3, K=2 | 0.0138±0.0005 | +0.0030±0.0003 | 8.3e-5 | 8.3e-5 | 47 | 5.7e-4 |
| random features | 0.311±0.007 | +0.300 | 4.3e-4 | 4.1e-4 | 9 | 2.9e-3 |

What drives the drop in E_err:
- Mostly E[log sigma]: at depth 8, lambda = 3 cuts its nse from 2.3e-2 to 2.4e-4 (98x).
- E[NLL] improves 20-30x.
- E[mu] reaches the MC floor (about 1e-6).

How estimable the networks are:
- Training without the penalty makes the network harder to estimate than at init. At depth 8, lambda = 0 has
  N_eff = 87 for E[NLL] against 1150 at init.
- Penalised networks are easier to estimate than the init.
- Absolute estimability is still poor: N_eff for the expected loss is at most about 2.7k, while one propagation costs
  38k-94k forward passes. Sampling at matched compute wins by 15-50x on E[NLL].

## How the penalty was satisfied
Propagated and sampled statistics come close: at depth 8, the final-layer covariance error is 0.03-0.04 against 0.49
at lambda = 0 and 0.21 at init. The penalty fell from about 0.2 at init to about 0.005.

It is not met by becoming linear or kernel-like:
- weights move slightly more than at lambda = 0 (relative change 0.28 vs 0.22);
- the random-features baseline is 20x worse on the task;
- attention entropy barely changes (depth 8: 0.76 to 0.78-0.79; depth 4: 0.71 to 0.80-0.82);
- GELU inputs sit in the curved zone (|z| < 1.5) about 98% of the time in every arm, so that metric does not
  discriminate.

What does change:
- With the K = 4 penalty, the residual-stream marginals become nearly Gaussian: rms excess kurtosis falls from about
  1.2 to about 0.1, and rms skew from about 0.7 to about 0.04.
- The coefficient of variation of the LayerNorm input norm roughly halves (0.10-0.12 to 0.06-0.09), which makes
  LayerNorm closer to a fixed rescaling.

So the K = 4 terms are met by removing non-Gaussianity, not by predicting it: the K = 4 propagator still predicts
essentially none of the remaining skew.

The K = 2 arm reaches a similar covariance match (0.026) and keeps the skew (0.34). It costs slightly less task loss
than the K = 4 arms, and its estimability gain is 14x (depth 8) and 47x (depth 4). As estimators, K = 2 and K = 4
agree within noise in every arm.

## Verdict against the preregistered thresholds
The thresholds were:
- no degradation: mean Delta <= max(2 se, 0.005 nats);
- improved: E_err ratio >= 5.

Both are met by every penalty arm at both depths, so the preregistered verdict is SUCCESS. Two qualifications matter
as much as that word:
1. The task cost is real and highly significant: +0.003 to +0.004 nats, 9-40 se, about 25-38% of the remaining gap to
   Bayes. It passes only because the preregistered tolerance (0.005 nats) is about 45% of the lambda = 0 gap; a
   relative tolerance such as 10% of the gap would turn every arm into a trade-off. The cost hardly grows with lambda:
   lambda = 0.3 already pays +0.0027 for 9-30x, and lambda = 3 pays +0.004 for 74-191x.
2. The gain is relative to an unpenalised network. By ARC's criterion (beat sampling at matched compute), every model
   still loses, before and after the penalty.

Compared with certified training: the propagation becomes tight without the large clean-accuracy loss typical of IBP
(here about 1/3 of an already small gap). But this is an average-case moment estimate, not a worst-case bound.

## Caveats
- K = 4 here is the structured form above, not ARC's full K = 4. As an estimator it adds nothing over K = 2 in this
  architecture; its effect is only through the extra skew/kurtosis terms in the penalty.
- The penalty was applied every 8th step (weight 8*lambda) under Adam, so lambda is not a linear strength.
- Small world and small model, 3000 steps, 3 seeds, an easy task (lambda = 0 is 0.011 nats from Bayes). E[mu] errors in
  the penalised arms sit at the MC floor, so the geometric-mean ratios understate the E[mu] gain and are driven by
  E[log sigma] and E[NLL].
- The propagator needed PSD projection and several closures. Its error floor is attention: the a·v product and
  LayerNorm's joint non-Gaussianity. A stronger propagator could change both the costs and the gains.
- GPU: about $12.2 at the $0.87/h ceiling rate. One spot VM was preempted during boot and one on-demand VM was stuck
  on an apt mirror, both before running anything. All VMs deleted; gcloud compute instances list shows none of mine.
