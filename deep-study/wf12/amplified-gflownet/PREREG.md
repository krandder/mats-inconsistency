# PREREG (wf12/amplified-gflownet), written before any run of `scripts/e3b_loss.py` or `scripts/e4_trust.py`

E1 (`e1_limits.py`) and E2 (`e2_distill.py`) are exact or Monte Carlo measurements of limits, not comparisons of
arms; they ran before this file. This file registers the two comparison runs.

## E3b: does a consistency loss on the amplified predictions help when real data are few?
World: `world.py`, 3 binary variables, 48 demonstrations, alien bit. Regimes: R1 score right (sM = 0, goal = 0: T's
marginal over values equals reality); R2 score wrong (sM = 1, goal = 0). Worlds: seeds 1, 2, 3. Rollout policy:
`noisy_policy(R, 1.5, 2.0, 22)`. Query: P(x_i = 1 | C), C = values of a subset of the other variables; rollouts are
masked to C. Amplified prediction at compute K in {2, 8}: f = sigmoid(alpha * u + beta * v), u, v = clipped logits
(0.02, 0.98) of the weighted and the unweighted fraction of K fresh rollouts. Raw amplifier: alpha = 1, beta = 0.
Arms (fit of alpha, beta; scipy Nelder-Mead from (1, 0)): data only (log loss on n_real real examples, n_real in
{8, 32, 128, 512}); data + lambda * each of three losses on 2,000 unlabeled queries; each loss alone.
 (i) order: squared difference of log f(X) + log f(Y|X) and log f(Y) + log f(X|Y), fresh rollouts per call.
 (ii) residual: variance over random (order, values) of sum_t log f(step t) - log(score of the values), the score of
      the values estimated from 64 rollouts masked to those values (no enumeration of orders in the arm).
 (iii) compute: cross-entropy of f at K to the raw weighted fraction at 64 K (fresh rollouts).
lambda = 1 primary, 10 secondary. Seeds 0..19 per cell. Metric: excess log loss against reality's exact conditional
(evaluation code only) on 4,000 fresh test examples; second metric: mean |f - T's conditional|.
Bins for a relative change of excess: major >= 20 %, minor 5-20 %, null < 5 %, negative: worse.
Predictions:
 H1 (i) alone has no anchor: its fit is no better than the raw amplifier at K = 2 (not better by 5 %) in >= 5 of 6
    world x regime cells.
 H2 (ii) alone and (iii) alone, with no real data, improve on raw at K = 2 in R1: major.
 H3 at n_real = 8, K = 2, R1: data + (ii) and data + (iii) beat data only (major; better in >= 70 % of seeds);
    data + (i): null.
 H4 in R2 at n_real = 512: data + (ii), data + (iii) are worse than data only; at n_real = 8 they are better.

## E4: where the reality-trained trust head sits
Same world family, R2 with one reliable route (`stiff_order = (0, 1, 2)`, sM = 1.5), worlds 1, 2, 3.
 F1 trust in kinds of demonstration: weights c(order) on the score, fitted on 2,000 real samples (a) exactly (sum over
    the six orders) and (b) through K rollouts masked to the real values, K in {1, 4, 16, 64, 256}, under rollout
    policies A (0.5, 0) and B (1.5, 2.0).
 F2 free head on search statistics: f = sigmoid(alpha u + beta v) fitted at K = 8 under policy A.
 F3 search-effort trust that fades: f = sigmoid(lam u + (1 - lam) v), lam = ESS / (ESS + kappa), kappa fitted.
Measured: the exact K -> infinity limit of each form under A, B and the selective policy S (never shows x0 = x1 = 1);
max over queries of |limit - target's conditional|.
Predictions: P1 F1(a) and F3 have the same limit under A and B (difference 0 to machine precision, by construction;
 a guarantee, listed to be checked); P2 F2's limit differs between A and B by > 0.02; P3 F1(b) at K = 1 equals the
 policy's own order frequencies (proof in the note) and so differs between A and B by > 0.1 in total variation, and the
 difference falls below 0.02 by K = 256; P4 under S every form that reads rollouts is moved by > 0.05; P5 the fitted
 F1(a) target is closer to reality than the unweighted score (excess KL lower by >= 20 %) in all three worlds.
