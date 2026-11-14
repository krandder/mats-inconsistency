# The amplified GFlowNet: what exists, one exact construction, and the consistency requirement as an object (wf12/amplified-gflownet)

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 15:50-16:25Z. CPU only, no cluster job. Words follow
`wf9/FRAME.md` section 5: demonstration (the terminal state), score, target T (the score normalised), rollout policy
(the GFlowNet), amplification (the search on top of K rollouts), distillation, alien coordinates.
Labels: PROVEN (short proof here), CHECKED (script and output named), SOURCE (quoted, with locator), GUARANTEE (true by
construction), NOT RUN. "Post hoc" = not in `PREREG.md` (hashed 16:09:43Z, before E3b and E4 ran; `PREREG.sha256`).
Everything is an exact table on a world of 3 binary variables. Nothing here is a trained network.

## 0. In plain words

The context first. On 10 Nov you said the GFlowNet is the rollout policy, like AlphaZero's policy network, and the
search you want is an amplification on top of many rollouts, which is then distilled back; you said you did not know how
to do it, and asked whether your consistency requirement needs "an amplified version of a GFlowNet".

1. The pieces exist, but nobody has written your condition. One 2024 workshop paper puts a soft tree search on top of a
   GFlowNet and distils it back; it proves nothing, and in its experiments the search never reads the score. Others
   refine rollouts by backtracking, or reweight them by the score. None states a condition on the amplified predictions.
2. I built your two-level design as a small exact world (a demonstration is an order plus values, scored step by step;
   the rollout policy carries an alien bit that is dropped at submission) and put three amplifications on it. All three
   reach "in proportion to the score" with enough compute, whatever the rollout policy's errors, if they read the score
   at terminal states. Reweighting and backtracking also need the policy to be able to produce every demonstration;
   the tree does not, but its guarantee is bought by visiting every terminal state.
3. Distilling back works and the rounds compound: with K rollouts per amplification the policy's error falls by about
   1/K per round. With K = 1 nothing is learned and the policy drifts.
4. Your consistency requirement: order-against-order agreement has no anchor. Answering every question from one shared
   set of rollouts satisfies it exactly at any compute, and so does always answering one half. The version I would use
   is the variance over trajectories of (log amplified probability of the trajectory minus log score of its terminal
   demonstration). It needs no enumeration of orders, and it is zero only when the amplified sampler is exactly
   score-proportional. That is a GFlowNet loss applied to the amplified predictor: yes, "an amplified GFlowNet".
   In the toy it was worth about 500 real examples for the part of the head that calibrates the search.
5. Trust: the coordinator's split holds in the limit, with one trap. Trust in kinds of demonstration can shape the
   target, but if it is fitted through K rollouts it copies the rollout policy's preferences (at K = 1 exactly).
I don't know: any of this on a trained network or at a size where enumeration is impossible; decisions (your Q4).

## 1. What exists (Part 1)

Sources read as TeX or PDF under `wf9/bengio/src/` and `src/` here; locators are file and line. I searched by known
titles and arXiv ids, not exhaustively; I found no paper that proves convergence of amplify-then-distil to a
score-proportional sampler, and none that puts a consistency condition on amplified predictions.

| Source | The amplification | Limit with unbounded compute | Distilled back how | Proved / measured |
|---|---|---|---|---|
| Morozov et al., "Improving GFlowNets with MCTS", 2406.13655 (`main.tex`) | Soft tree search (MENTS form) on the policy's Q = log edge flow: leaf statistics start at the network's Q, backups are log-sum-exp plus log P_B (l.246-248) | Not stated. Two options at terminal leaves (l.249): read log R, or "skip the update ..., leaving it as it was initialized by Q_theta"; "All experiments ... are carried out with this option". Under the second the limit is the network's own terminal numbers backed up exactly, not T (section 2: TV 0.11-0.48). Under the first it is T for any Q (PROVEN below) | Regress Q_theta on Q_tree, loss (Q_theta - Q_tree)^2 (l.256-258); terminal edges on log P_B + log R (l.262) | No theorem. Measured: "using MENTS to compute targets for the training of Q_theta provides a noticeable boost even when the model is evaluated without MCTS" (l.299), hypergrid and bit sequences |
| Tiapkin et al., "GFlowNets as entropy-regularized RL", 2310.12934 (`source/representation.tex`) | None; it is the bridge | Thm 1 (l.28-47): with step payoffs log P_B(s given s') and terminal payoff log R, the optimal soft policy "is equal to P_F" and V*, Q* "coincide with log F(s) and log F(s -> s')". Without the P_B term, soft RL samples n(x) R(x) (l.12) | - | Proved. Prop. (l.171-174): V^pi(s0) = log Z - KL(q^pi, P_B); so "the policy error controls the distribution approximation error" (l.185-190) |
| Xiao et al., MENTS, NeurIPS 2019 (`src/ments/ments.txt`, sections 4.1-4.2) | Tree search with soft-max backups, eq. (8); tree policy = soft-max of Q mixed with uniform, weight eps A / log(N + 1), eq. (7); a policy network enters only as the initial Q of new leaves ("Q_sft(s_T, a) = log pi~(a given s_T)") | "after its subtree has been fully explored, the estimated softmax value will converge to the optimal value at an exponential rate" (before Thm 3; Thm 4); Thm 5: wrong root action with probability <= C t exp(-t / (log t)^3). The limit does not depend on the prior | None in the paper | Proved (supplement not read). It returns the best root action; a sampler is Morozov's use of it |
| Kim et al., "Local Search GFlowNets", 2310.02710 (`sections/4_method.tex`) | Backtrack K steps with P_B, rebuild with P_F, then filter: accept only if the score rises (default: "we use deterministic filtering as a default setting"), or Metropolis-Hastings ("helps maintain the sampling objective") | Not stated. In my construction: greedy filter collapses to the top demonstration (TV 0.91); MH with the standard ratio reaches T; the ratio as printed (forward over reverse proposal) does not (TV 0.49-0.78). I did not read their code | Accepted and rejected trajectories go to the training set; trajectory balance on them, top-decile prioritised | No theorem. Measured on molecule and sequence tasks |
| Bengio et al., Scientist AI, 2502.15657 (`plan.tex`) | l.226: "amortized inference neural networks can be complemented by additional run-time computation to refine the predictions, along lines similar to Monte-Carlo Tree Search in AlphaGo"; l.269-271: a "variable-time component", "a GFlowNet objective that seeks short explanations that effectively reduce the uncertainty in the predicted probabilities" | The zero-loss statement is about the trained network (l.233: "when the GFlowNet training loss is exactly zero"), not about run-time search | Not described | A proposal: no algorithm, no experiment |
| Richter et al., VarGrad, 2010.10436 (`manuscript.tex`); Malkin et al. 2210.00580 | Not an amplification: a loss. Variance under any reference r of log(q / p) (l.543-548): "it is zero if and only if q = p" when r covers both supports | - | Gradient at r = q equals the KL gradient (Prop., l.566-572); trajectory balance gradients are KL gradients (2210.00580 Prop. 1, `camera_ready.tex` l.249-256) | Proved. This is candidate (ii) of section 3 once q is the amplified predictor |
| Le et al., reweighted wake-sleep, 1805.10469 (`main.tex`) | K rollouts of the proposal, self-normalised weights against the model (l.987) | "although its estimator is biased, this bias decreases as K increases" (l.1002); "consistent" (l.1039) | Maximum likelihood of the proposal on the weighted particles | At K = 1 the update "encourages the inference network to concentrate mass" (l.1359): my E2 shows that drift |
| Macfarlane et al., SPO, 2402.07963 (`content/method.tex`); Anthony et al., ExIt, 1705.08439; Grill et al. 2007.12509 | SMC with resampling as the improvement step; tree search as "expert"; AlphaZero's visit counts as a regularised policy | SPO Prop. (l.165-168): the bound rises monotonically with exact steps; "leave derivations regarding the impact of this estimation on overall convergence for future work" (l.170). Grill: the prior's weight lambda_N is "O~(1/sqrt N)" (l.345) | Projection of the search policy onto the network ("as performed in expert iteration style methods such as AlphaZero", SPO l.152) | Targets there are reward-maximising, not score-proportional |
| Zhao et al., twisted SMC, 2404.17546 (`1_icml/02_back_rb.tex` l.236) | SMC with learned intermediate targets | "an unbiased estimate of the partition function ..., regardless of the intermediate" targets and proposal | Twist learning (not read in full) | Proved there; not rebuilt here (NOT RUN) |

Reading. The rollout policy as a GFlowNet, a search on top, and distillation of the search into the policy all exist
(Morozov; local search; wake-sleep). What is proved is one of three things: the zero-loss policy samples T (Tiapkin); a
soft tree converges to soft-optimal values whatever its prior (MENTS); weighted rollouts are consistent (wake-sleep,
SMC). No source joins them into "the distilled loop converges to T", and the one paper closest to your design drops the
only step that makes its limit independent of the network (reading the score at terminal leaves).

## 2. Your design as one exact construction (Part 2)

Files: `scripts/world.py`, `amps.py`; runs `e1_limits.py`, `e2_distill.py`, `e1b_extra.py` with outputs in `out/`.

- **Terminal state.** A demonstration d = (the order in which the 3 binary variables were filled, their values): 48
  of them. The simple world model M (a table of any-order conditionals: reality's own, plus noise of scale 1 in the
  logit, so its numbers disagree across orders) scores it step by step, times a goal factor on the values:
  R(d) = prod over steps of M(value given variable, context) x goal. T(d) = R(d) / Z. The order is part of d.
- **Rollout policy.** A flow over an augmented state: it first picks an alien bit a, then builds d step by step, and a
  is dropped at submission (96 augmented trajectories, 2 per demonstration). It is stored as log edge flows Q
  (the form Tiapkin and Morozov use); P_F = softmax(Q). Imperfect policies: A = exact flows plus noise 0.5 (TV to T
  0.24); B = noise 1.5 and each alien value pushing a different variable first (TV 0.48); S = B that never shows a
  demonstration with x0 = x1 = 1 (hides 0.15 of T; no full support).
- **Three amplifications on K rollouts.** (1) Self-normalised weighting: weight of a rollout = score / probability.
  (2) Soft tree with sum backups, MENTS form as in Morozov; option 1 reads the score at terminal leaves, option 2
  never does. (3) Back-and-forth refinement: drop the last j of the 4 components, rebuild with the policy, filter.

**Table 1. Total variation to T over the 48 demonstrations** (`out/e1_limits.out`, `e1b_extra.out`). Cells: policy A / B / S.

| amplification | K = 4 | K = 64 | K = 1,024 | limit |
|---|---|---|---|---|
| unweighted counting of rollouts | 0.24 / 0.48 / 0.55 | 0.24 / 0.48 / 0.54 | 0.24 / 0.48 / 0.54 | the policy's own law |
| weighting, expected law (each d counted once) | 0.078 / 0.318 / 0.406 | 0.007 / 0.076 / 0.182 | 0.003 / 0.008 / 0.151 | T / T / T on what S shows (0.1505) |
| weighting, one set of K (bias and noise) | 0.87 / 0.88 / 0.86 | 0.34 / 0.53 / 0.56 | 0.09 / 0.18 / 0.25 | same |
| soft tree, score read at terminal leaves | 0.203 / 0.470 / 0.524 | 0.112 / 0.347 / 0.344 | 0.002 / 0.019 / 0.097 | T / T / T (S: 0.045 at 4,096, 0.011 at 16,384) |
| soft tree, score never read (Morozov's experiments) | 0.203 / 0.470 / 0.524 | 0.132 / 0.475 / 0.499 | 0.114 / 0.437 / 0.485 | 0.114 / 0.435 / 0.484 (exact) |
| refinement, Metropolis-Hastings filter (exact) | 0.057 / 0.315 / 0.404 | 0.000 / 0.097 / 0.193 | 0.000 / 0.004 / 0.151 | T / T / T on what S shows |
| refinement, ratio as printed in 2310.02710 | 0.429 / 0.652 / 0.712 | 0.491 / 0.782 / 0.778 | 0.491 / 0.783 / 0.778 | 0.491 / 0.783 / 0.778 |
| refinement, accept only if the score rises (their default) | 0.447 / 0.634 / 0.696 | 0.828 / 0.909 / 0.909 | 0.908 / 0.910 / 0.910 | the top demonstration |

What each returns as K grows, and whether the limit is T for every full-support rollout policy:
- **Weighting: yes** (PROVEN). For any normalised backward law P_B(a given d) over the alien bit, the weight
  w = R(d) P_B(a given d) / P_F(a, d) has E[w 1_E] = sum over (a, d) in E of R(d) P_B(a given d) = R(E), so weighted
  shares converge to T(E given C). It returns a weighted set; its error at finite K is mostly noise (third row).
  Without full support the limit is T restricted to what the policy can show.
- **Soft tree, score read at terminal leaves: yes, and without full support** (PROVEN: the tree is finite; the uniform
  share of the tree policy visits every edge infinitely often; once every node is expanded and every terminal edge has
  been read, the backups equal the exact log flows by backward induction, Tiapkin Thm 1, whatever Q the leaves started
  from). The price is in the proof: every terminal state is read. It returns a policy at every node of the tree, and
  log Z at the root. **Score never read: no.** The limit is the network's own terminal numbers backed up exactly.
- **Refinement with the Metropolis-Hastings filter: yes** (PROVEN: each move resamples a block from the policy and
  accepts with min(1, w'/w); T is invariant; full support makes the chain irreducible). The greedy filter and the ratio
  as printed do not target T.
- **Alien coordinates.** The weight must count each demonstration once: divide by q(d) = sum over a of P_F(a, d), or
  use a backward law over the alien coordinate. This is where the search's freedom lives (FRAME row 15). With a fixed
  uniform backward law, distillation drives the alien bit to carry nothing about the demonstration (information
  0.336 nats at the start, 0.001 after six rounds at K = 16); counted once, it keeps it (0.31) and the target is the
  same (`out/e2_distill.out`).

**Table 2. Amplify, then distil into the rollout policy: TV(policy, T) by round, start policy B** (`out/e2_distill.out`).
The policy is tabular on a tree, so distilling = setting it to the amplified law (expected, or of a finite batch).

| amplification, compute per call | round 0 | 1 | 2 | 3 | 6 |
|---|---|---|---|---|---|
| weighting K = 1, expected | 0.480 | 0.478 | 0.478 | 0.480 | 0.482 |
| weighting K = 1, 64 sets per round | 0.480 | 0.572 | 0.601 | 0.642 | 0.710 |
| weighting K = 2, expected (exact map) | 0.480 | 0.390 | 0.316 | 0.250 | 0.081 |
| weighting K = 4, expected | 0.480 | 0.318 | 0.189 | 0.086 | 0.011 |
| weighting K = 16, expected | 0.480 | 0.185 | 0.026 | 0.006 | 0.004 |
| soft tree (score read), 8 trees x 64 simulations | 0.480 | 0.227 | 0.168 | 0.135 | 0.077 |
| soft tree (score read), 8 trees x 256 | 0.480 | 0.101 | 0.040 | 0.020 | 0.005 |
| soft tree (score not read in search; one scored trajectory per tree) | 0.480 | 0.414 | 0.417 | 0.386 | 0.293 |
| refinement, 1 move (exact) | 0.480 | 0.410 | 0.348 | 0.299 | 0.176 |
| refinement, 4 moves | 0.480 | 0.315 | 0.199 | 0.108 | 0.002 |
| refinement, 16 moves | 0.480 | 0.203 | 0.053 | 0.001 | 0.000 |

- **One round.** Weighting with K rollouts maps a policy with relative error delta to one with error about delta / K
  (PROVEN to first order: with q = T(1 + delta) the weights are 1 - delta, and K q E[w_1 / sum w] = T(1 + delta / K)).
  CHECKED from policy A at K = 2: 0.238, 0.136, 0.075, 0.040, 0.021, 0.011, 0.005 (ratio 0.50-0.57). Far from T it is
  slower (0.81 per round from B at K = 2). K = 1 is the identity in expectation, and with finite batches a random walk
  away from T. Refinement never increases the error (PROVEN: a kernel with invariant law T contracts total variation).
- **Rounds compound**, geometrically, in expectation. With a finite batch that replaces the policy each round they
  stop at one batch's noise (0.32-0.38 at K = 4, 0.13-0.15 at K = 16, 64 sets); moving the policy a quarter of the way
  per round they continue (0.12 and 0.04 after 24 rounds; `out/e1b_extra.out`, post hoc).
- The 0.004-0.006 floor in the K = 16 row is the Monte Carlo error of 20,000 sets, not a property of the method.

**What the amplified predictions are, and order consistency at finite K** (`out/e3a_objects.out`).
- Weighting: P(X) is the weighted share of the K demonstrations with that value; P(Y given X) the share inside those
  with X. Tree: a ratio of root values with and without the evidence as a factor of the score (NOT RUN as a query
  engine). Refinement: frequencies along the chain.
- Answered from ONE shared set, "P(X) then P(Y given X)" equals "P(Y) then P(X given Y)" exactly at every K, right or
  wrong: both are read off one joint (measured 0.0000 at K >= 256; nonzero below only because a set can contain no
  rollout that matches the evidence). Answered from fresh rollouts per question, the two orders differ by noise:
  RMS log mismatch 3.9, 1.8, 1.06, 0.69, 0.46 at K = 4, 16, 64, 256, 1,024.
- A trap in the limit. If "P(X) then P(Y given X)" is read as the amplified step probabilities along the demonstration
  order X, Y, the limit is T given that order. With a soft M the score depends on the order, and T(values given order)
  differs from T(values) by up to 0.27 at infinite compute. Under your two-level answer the consistent object is the
  marginal over submitted demonstrations (orders summed), with evidence as a factor of the score. In a rule world, or
  with a consistent M, the two readings coincide.

## 3. Your consistency requirement, as an object (Part 3)

Your words (10 Nov): amplified P(X) then P(Y given X) "we expect ... to get consistent" with the other order "in the
limit of more compute"; "any variance is a signal that this amplification procedure can be improved"; it is "a
data-efficiency thing"; "actually estimating the variance over permutations is hard; in the Bengio perpective this
would be a variance over trajectories I think". Three candidate objects, measured on amplifiers whose true error is
known (`scripts/e3a_objects.py`, `out/e3a_objects.out`; rollout policy B, K = 4, 16, 64, 256, 1,024).

| | (i) order against order | (ii) residual along a trajectory | (iii) agreement with larger compute |
|---|---|---|---|
| The object | mean square of log[A(X) A(Y given X)] - log[A(Y) A(X given Y)], A = amplified prediction | variance over sampled trajectories of [sum of the log amplified step predictions along the trajectory] - [log score of its terminal demonstration] | the K-compute prediction equals the average of the same prediction at larger compute (a martingale over compute, pooled over questions) |
| From samples, without enumerating orders? | Yes. Two random orders of the same variables give an unbiased estimate of the variance over orders | Yes. Trajectories from any full-support reference; one amplification call per step, one score per trajectory; Z is not needed | Yes. Two calls per question |
| Driving it to zero implies | the answers are conditionals of ONE joint law. Nothing about which | the amplified sampler is exactly T, and minus the mean residual is log Z (PROVEN: a constant residual on a full-support reference means amplified probability = c x score for every trajectory; both sides normalise; VarGrad states the same for any q) | the small-compute answer is an unbiased forecast of the large-compute answer; it is right exactly when that one is |
| Can a wrong answer satisfy it? | Yes. Unweighted rollouts answered from one shared set: mismatch 0.0000 at K = 1,024 with error 0.16. A constant one half: exactly 0 | Not against T. (Against reality, only if the score is wrong; that is the score's side) | Yes. Unweighted counting: pooled slope -0.016 at K = 256, mean gap 0.001, error 0.18. It has no anchor of its own; it inherits the limit's |
| On the weighting amplifier (error 0.33 -> 0.09) | 3.9, 1.8, 1.06, 0.69, 0.46: falls slowly, pure noise | 22, 5.6, 1.6, 0.50, 0.13: falls to 0. On unweighted counting it stops at the policy's own 1.2 (1.37 at K = 1,024) | slope -0.82, -0.74, -0.61, -0.45: fresh weighted sets are over-dispersed at every finite K although their limit is right |

Two remarks. (i) does detect one real error: evidence handled by masking the rollouts and counting them stays at 0.32
however large K is. And (ii) at small K is dominated by the amplifier's own noise (22 at K = 4): it is mismatch plus
noise, and both are "the amplification can be improved", which is your sentence.

**As an extra loss, with few real data** (registered; `scripts/e3b_loss.py`, `out/e3b_loss.out`). The amplified
prediction of P(x_i given C) at compute K is a two-parameter head on K fresh rollouts masked to the evidence:
f = sigmoid(alpha u + beta v), u and v the logits of the weighted and unweighted shares. Real data: n questions with
real outcomes. Each consistency loss uses 2,000 questions with NO outcome. For (ii) at this level the score of a value
vector is estimated from 64 rollouts masked to it (no enumeration inside the arm). 3 worlds x 20 seeds.

**Table 3. Excess log loss against reality, nats (score right, R1). Cells: K = 2 / K = 8.**

| real examples n | data only | data + (i) | data + (ii) | data + (iii) |
|---|---|---|---|---|
| 0 (the loss alone) | raw head 0.678 / 0.253 | 0.086 / 0.086 | 0.067 / 0.049 | 0.067 / 0.049 |
| 8 | 2.04 / 1.42 | 0.082 / 0.076 | 0.069 / 0.050 | 0.099 / 0.077 |
| 32 | 0.112 / 0.101 | 0.082 / 0.072 | 0.067 / 0.049 | 0.074 / 0.058 |
| 128 | 0.076 / 0.063 | 0.081 / 0.072 | 0.067 / 0.049 | 0.069 / 0.052 |
| 512 | 0.069 / 0.051 | 0.081 / 0.072 | 0.067 / 0.049 | 0.067 / 0.049 |

- **(i) alone collapses**: in all 120 runs the fit is alpha = beta = 0 (largest absolute value 0.0009), the constant
  one half, the same 0.086 at K = 2 and K = 8. Added to data it acts as a pull toward that constant: it rescues n = 8
  (where the unregularised fit explodes) and costs 7-41 % from n = 128 on.
- **(ii) and (iii) alone, with no real outcome, equal 512 real examples** (0.067 against 0.069; 0.049 against 0.051),
  and adding them to any n never hurt (better in 88-100 % of runs). With lambda = 10 the same.
- The reason is worth more than the number: this head's whole job is on the search side (how far to believe K
  rollouts), and that can be learned against the score. Real outcomes are needed only for what the score gets wrong.
  With the score wrong (R2; T itself is 0.011 nats from reality) the picture is the same: (ii) alone 0.070 / 0.052,
  512 real examples 0.071 at K = 2. My prediction that the anchored losses would hurt there at n = 512 was wrong
  (-2 %, -1 %): a two-parameter head cannot express a 0.011 difference.

**Recommendation.** Define the requirement as (ii): "the variance, over trajectories, of the gap between the amplified
step predictions and the score of the terminal demonstration goes to zero". It is your "variance over trajectories",
it is estimable from samples, and it is the trajectory-balance / log-variance loss with the amplified predictor in the
place of the one-step policy, so "an amplified GFlowNet" is the right name. Use (iii) as the cheap way to train the
small-compute member (it is what AlphaZero's policy loss does), knowing it only passes down what the large-compute
member knows. Keep (i) as a diagnostic: a system with one shared search state satisfies it for free, so as a loss it
either does nothing or pulls toward a constant. Limits: (ii) needs one amplification call per step of each sampled
trajectory, and a head of two parameters says little about a free-form one (wf10 found no gain from a
consistency-over-compute loss on a trained aggregator; I did not test (ii) there).

## 4. Where the reality-trained trust head sits (Part 4)

The question: if the final prediction is a learned function of the amplification's statistics, which forms keep the
limit of compute equal to the score-proportional target whatever the rollout policy? wf10/aggregator showed that a
free learned head moves with a searcher that shows only some demonstrations. Registered run: `scripts/e4_trust.py`,
`out/e4_trust.out`; worlds 1, 2, 3 in which M is exact along one route (order 0, 1, 2) and noisy elsewhere.

| Form | What it reads | Limit of compute under policies A / B (max over 27 questions of the distance to the target) | Under the selective S |
|---|---|---|---|
| F1 trust in KINDS OF DEMONSTRATION: a weight c(kind) inside the score; here kind = the demonstration's order, i.e. which of M's numbers it used | the submitted demonstration only | The target becomes T_c = c x score, normalised. Weighting reaches it under A and under B: 0.0000 / 0.0000 (GUARANTEE, section 2; CHECKED) | not a search-side object |
| F2 free head on search statistics, f = sigmoid(alpha u + beta v), fitted on reality at K = 8 under A | weighted and unweighted shares | 0.24-0.31 from the target under both; its limit differs between A and B by 0.03-0.05 | 0.20-0.63 |
| F3 search-effort trust that fades: weight ESS / (ESS + kappa) on the weighted share, the rest on the unweighted one; kappa fitted on reality | the same, plus the effective sample size | 0.0000 / 0.0000 for every kappa (GUARANTEE: the weight goes to 1) | 0.50-0.82 |

- **F1 shapes the target and helps**: fitted on 2,000 real samples by an exact sum over the six orders, the excess of
  the target against reality falls from 0.076 / 0.041 / 0.070 nats (all routes equal) to 0.0003 / 0.0014 / 0.0011, with
  weight 0.95 / 0.91 / 0.69 on the reliable route. No searcher is involved, so nothing of the search enters T.
- **The trap: F1 fitted THROUGH the amplification.** If the fit sees each real outcome only through K rollouts masked
  to it (the search finds demonstrations of what happened), then at K = 1 the fitted weights are the rollout policy's
  own order frequencies (PROVEN: the likelihood of one rollout is c(order) x a constant, and the maximiser over the
  simplex is the empirical frequency of the sampled orders; CHECKED: TV 0.02-0.03). They are 0.55-0.81 in total
  variation from the exact fit and differ between A and B by 0.13-0.28. They converge slowly: at K = 256 the weights
  still differ between A and B by 0.13-0.28 (my registered "below 0.02" was wrong); the targets they imply differ by
  0.02 and are 0.001-0.009 nats from reality against 0.0003-0.0014 for the exact fit (post hoc). The mechanism is the
  bias of the logarithm of a noisy estimate: routes the policy shows often are estimated with less noise and get more
  weight. This is FRAME Q1's warning with numbers: reality-training a part of the score through a finite search carries
  the search policy into T.
- **F2 against F3 at finite compute** (excess against reality, worlds 1 / 2 / 3, policy A): at K = 8 raw 0.54 / 0.46 /
  0.31, F2 0.03 / 0.05 / 0.10, F3 0.14 / 0.12 / 0.13; at K = 512 raw 0.016 / 0.013 / 0.008, F2 0.028 / 0.046 / 0.070,
  F3 0.046 / 0.012 / 0.007. F2 wins at the compute it was fitted at by almost ignoring the rollouts (alpha 0.01-0.28)
  and then cannot use more compute. F3's kappa is not identified from one compute level (world 1: 2,981); fitted
  jointly at K = 2, 8, 32, 128 (post hoc) it gives 0.017 / 0.010 / 0.007 at K = 512, the raw amplifier's level, and
  under the swapped policy B 0.087 / 0.032 / 0.018 against raw 0.084 / 0.046 / 0.033.
- **Nothing that reads rollouts survives a policy without full support.** Under S the effective sample size looks
  healthy on what S shows, so the fading head trusts a wrong answer (0.50-0.82). What does see it is the total weight:
  E[w] = Z x (share of T that the policy can show) (PROVEN, one line), which needs Z, known here because M is
  normalised and the goal is 1 (NOT RUN as a head input). wf11/searcher-proof built the general version: the
  specification's own share of the search and an interval.

**The coordinator's proposal, tested.** "Trust in kinds of demonstration reads only the submitted demonstration and may
shape the target; trust in the search effort may read search-side information but only calibrates finite-compute
estimates and fades with compute." It passes the limit test: F1 with F3 has the same limit under A and B, and that limit
is the reality-fitted target. Three conditions have to be added. (1) The kinds-of-demonstration trust must be fitted
with the search taken out: an exact sum over routes, or enough compute that the fit no longer moves, or a debiased
estimate; fitted through few rollouts it IS the rollout policy's preference. (2) "Fades with compute" has to be a
constraint of the form (the head tends to the weighted share), and its rate must be fitted across compute levels. (3)
Both hold for full-support rollout policies only. A general learned head keeps the limit exactly when it tends to the
identity on a consistent estimator and its dependence on every other statistic vanishes; F2 does neither. In your 20 Oct
table this splits "params_real" in two: the world model with the kinds-of-demonstration trust (inside the score, no
searcher in its fit) and the head that reads "search node stats" (reality-calibrated, but constrained to fade).

## 5. Registered predictions, limits, files

| # | Registered (`PREREG.md`) | Outcome |
|---|---|---|
| H1 | (i) alone is no better than the raw amplifier at K = 2 | WRONG as registered: the constant it collapses to (0.086) beats the raw two-rollout amplifier (0.68) in 6 of 6 cells. The collapse itself: confirmed |
| H2 | (ii) and (iii) alone improve on raw at K = 2, score right: major | Right (-90 %) |
| H3 | at n = 8, K = 2: data + (ii), + (iii) major; data + (i) null | First part right (-97 %, -95 %); second WRONG (-96 %: it acts as a regulariser) |
| H4 | score wrong, n = 512: the anchored losses hurt; n = 8: they help | First part WRONG (-2 %, -1 %); second right |
| P1 | F1 (exact) and F3: same limit under A and B | Right: 0.0000 |
| P2 | F2's limit differs between A and B by > 0.02 | Right: 0.03-0.05 |
| P3 | F1 through K = 1 rollouts equals the policy's order frequencies; A and B agree within 0.02 by K = 256 | First right (0.02-0.03); second WRONG (0.13-0.28 in weights, 0.02 in implied targets) |
| P4 | under S every form that reads rollouts moves by > 0.05 | Right: 0.20-0.82 |
| P5 | the fitted F1 target is at least 20 % closer to reality | Right: 0.076 -> 0.0003, 0.041 -> 0.0014, 0.070 -> 0.0011 |

Limits.
- One family of tiny worlds (3 binary variables, 48 demonstrations, one alien bit). Every "limit" is enumeration; the
  tree's guarantee is exhaustive reading of terminal states, which is what a real search cannot do. Nothing was trained.
- The alien coordinate is one bit chosen first. Alien coordinates that change along the way, or a search state shared
  across rollouts (FRAME row 30), were not built. SMC with resampling was read (Zhao et al.) and NOT RUN.
- Distillation is exact (tabular). With a network, a round's error is the search's plus the fit's; the 1/K rate is
  first order near T and in expectation.
- "Data only" at n = 8 is an unregularised fit; (i) shows what a plain pull to a constant buys. At K = 8, (ii) beats
  that by 0.026 and 0.023 nats at n = 8 and 32 (0.050, 0.049 against 0.076, 0.072); (iii) by 0.015 at n = 32 only.
- Under your answers of 10 Nov nothing here depends on an open question except Q4 (decisions), which is not touched:
  the goal factor in E1-E3a multiplies every history, which is reading A of Q4.
- The check on 2310.02710's printed acceptance ratio is in my construction only; their code was not read.

Files (all under `/home/agent/workspaces/mats/work/deep-study/wf12/amplified-gflownet/`): `PREREG.md`, `PREREG.sha256`;
`scripts/world.py` (world, score, policies), `amps.py` (the three amplifications), `queries.py`, `e3b_helpers.py`;
`e1_limits.py`, `e1b_extra.py`, `e2_distill.py`, `e3a_objects.py`, `e3b_loss.py`, `e4_trust.py`; outputs `out/*.out`,
`out/*.json`, `out/e3b_loss.log`; sources `src/` (MENTS PDF and text, arXiv TeX of 2310.02710, 2010.10436, 2402.07963,
2007.12509, 1805.10469, 2402.05234 (not read), 1705.08439, 2404.17546). Run: `cd scripts && python3 e1_limits.py`
(46 s), `e2_distill.py` (52 s), `e3a_objects.py` (48 s), `e3b_loss.py` (189 s), `e4_trust.py` (3 s).
