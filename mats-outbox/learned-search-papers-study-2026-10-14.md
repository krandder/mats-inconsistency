# Learned search, learned aggregation, and LIR: study of 9 papers (2026-10-14)

Scope: what MCTSnet, Thinker, I2A, IBP, DVRL, PF-net, L2HMC, Stream of Search and Richardson's PDG/LIR papers say about Kelvin's setup (k discrete variables, learned forward model as the target, inversion by SMC or branch-and-sum, any-order network used only for proposals, a search policy over variables and roles, one coherent bounded-treewidth belief, outcome-only training).

Papers (key, arXiv id):
- mctsnet: MCTSnet, 1802.04697 (ICML 2018)
- thinker: Thinker, 2307.14993 (NeurIPS 2023)
- i2a: Imagination-Augmented Agents, 1707.06203 (NIPS 2017)
- ibp: Imagination-Based Planner, 1707.06170 (preprint 2017; later venue [unverified])
- dvrl: Deep Variational RL, 1806.02426 (ICML 2018)
- pfnet: Particle Filter Networks, 1805.08975 (CoRL 2018)
- l2hmc: Learning to generalize HMC, 1711.09268 (ICLR 2018)
- sos: Stream of Search, 2404.03683 (2024; COLM 2024 [unverified])
- lir: Local Inconsistency Resolution, 2604.17140 (AISTATS 2026), with Loss as Inconsistency, 2202.11862 (AISTATS 2022)

Sources: structured deep-reads of each paper by separate readers. Where a reader marked a claim unverified or as their own derivation, it is marked [unverified] or [derivation] here. Statements marked [this report] are my own analysis and appear in none of the papers.

## 0. Short answer to "what is the application of LIR to this?"

The idea is not scooped. LIR's own conclusion lists "Can we learn adaptive Refocus mechanisms?" as open (lir, Sec 6), and none of the 9 papers produces a coherent belief from a learned search or calibrates "searched and found nothing". What LIR gives is a precise vocabulary and a fixed update rule:

1. The incoherence of the any-order network is a defined number: put one arc per queried conditional q(x_i | x_S) into a PDG; the network is incoherent exactly when the inconsistency <<M>> is above 0 (lir, Eqs 1 to 3).
2. Grounding is a control mask: forward-model arcs p(z), p(x|z) get fixed attention and control chi = 0, evidence gets beta = infinity. With only these arcs, <<M>> = -log p(x) and the resolved joint is the exact posterior p(z|x) (Loss paper 2202.11862, Prop 4, as read by the lir reader).
3. The search policy is Refocus: the attention mask phi says which arcs a rollout looks at, the control mask chi says which belief parameters it may move. Bounded-treewidth belief maintenance by message passing is itself an LIR run (lir, Prop 4: LIR runs with message foci are exactly BP under different schedules).
4. Proposal learning is the EM alternation (lir, Prop 2). Varying the trust beta_q on the proposal arc moves between the reweighted-wake-sleep update (beta_q near 0) and the ELBO regime (high beta_q) [derivation, lir reader].
5. Aggregation semantics are fixed by gamma: at gamma = 0, unconditional reports pool geometrically with exponents beta_a / sum beta (Loss paper Lemma 10 and Sec 8), so agreeing reports do not sharpen the belief; at gamma = 1 with alpha = beta, factors multiply (factor-graph semantics, Loss paper Prop 14), so independent evidence accumulates, which is the "trade adds to a natural parameter" view.

The most useful consequence is a clarification [this report]. With the forward model at chi = 0 and the evidence clamped, the PDG target is already the exact posterior; the rollouts add no arcs to the semantics. They are approximate computation of the argmin. So there are only two places where a learned aggregator can legitimately act:
- (a) the trust (beta) on heuristic arcs, i.e. the any-order network's conditionals, which should fall as compute grows; and
- (b) a model of the approximation error: what to believe about posterior mass the search has not yet reached.

Item (b) is "computation as evidence", and it is also the logical-induction question (beliefs coherent with what compute t has derived, calibrated about the rest). On this reading, "learn the MCTS aggregation" and "learn the search state" become one problem: learn the beta schedule and the unexplored-mass model, conditioned on the search state, scored on outcomes.

A second consequence is about "mixture vs log pool" [this report]. Which combination rule is right depends on the type of report:
- Monte Carlo rollouts that estimate the same posterior (the default in this design): combine at the level of unnormalized weights on hypotheses. Take the union of the discovered support with exact forward scores; repeated unbiased estimates of the same branch evidence are averaged linearly, weighted by inverse variance. Normalizing gives a weighted mixture, not a geometric pool. A geometric pool of two rollouts that found different modes zero-forces both modes, which is wrong.
- Opinions about the same variable with independent errors (for example any-order conditionals reached through different orders): geometric pool, the gamma = 0 LIR semantics (Loss paper Sec 8 argues against the mixture here because "redundancy is not used to correct errors").
- Independent likelihood evidence (distinct observations): product, the gamma = 1 semantics.

LIR has gamma as one global scalar and gives no per-arc choice (lir reader). The mcts reader's note that a mixture needs an explicit selector variable in PDG terms [unverified] is the same point.

## 1. Component table

Cells say what each paper does for the component. "absent" means the paper has nothing for it.

| Component | mctsnet 1802.04697 | thinker 2307.14993 | i2a 1707.06203 | ibp 1707.06170 | dvrl 1806.02426 | pfnet 1805.08975 | l2hmc 1711.09268 | sos 2404.03683 | lir 2604.17140 (+2202.11862) |
|---|---|---|---|---|---|---|---|---|---|
| Forward model / target | Exact known deterministic simulator | Learned deterministic dual net (state-reward + value-policy), supervised, staged | Learned next-frame model, pretrained by NLL, then frozen | Interaction net trained only on real transitions as a separate subgraph; perfect model in mazes | Learned generative model inside SMC, trained by ELBO plus RL gradients (co-adapted) | Hand-set transition noise; learned discriminative "likelihood" f with no generative grounding | Given energy U(x), exact | absent: the LM simulates transitions itself | Forward cpds as arcs; chi = 0 freezes them; evidence beta = inf gives mu* = p(z\|x) |
| Proposal / inverse | Policy prior psi_p from a separate model-free net; no IS correction | Base policy distilled from the planner's real actions | Rollout policy distilled from agent policy with stop-gradient; no correction | Controller proposes imagined actions, used at face value | Learned q_phi(z\|h,a,o) with exact weights p/q; ELBO + RL | Bootstrap proposal; soft-resampling with IS correction | Learned invertible leapfrog proposal with exact MH-Green correction | The LM is the proposer | q as a controlled arc; EM by alternating foci (Prop 2) |
| Search policy (variables, roles) | Learned pi(a\|h) = w0 psi_p + w1 u(parent, child), REINFORCE | Learned RL policy over imagined action plus reset bit, K steps | Fixed: one rollout per first action, depth tau | Learned manager over 3 routes {act, root, current leaf}; full tree not trainable | absent (fixed temporal filter, fixed K) | absent (fixed schedule) | absent (fixed random masks) | Implicit in LM generation over a serialized tree | Refocus; fixed finite foci (uniform, partial, hub); learned Refocus listed as open |
| Search state | Per-node learned 128-d memory h_s, tree, counts | Hand-coded u_tree (mean/max return, counts, step k) plus LSTM-attention memory | absent (5 rollout embeddings concatenated) | LSTM plan context plus counters j, k | Particle set plus GRU summary; resampling deletes failures | Particle set only | Memoryless by design (v, d resampled) | Full token context, unbounded | Parameters theta; no record of foci tried; residual inconsistency definable but unused |
| Aggregation into a state | Learned gated residual backup h <- h + g f | Fixed hints (mean, max, count) plus learned decision head | Concatenation plus MLP | LSTM; maze version adds updates to a tensor, policy = Q + c | Fixed Bayes weights inside; learned GRU summary for the policy | Fixed Bayes reweighting plus weighted mean | absent (single chain, fixed MH) | absent (first path to goal wins) | Fixed: argmin of OInc + gamma SDef; geometric pool at gamma = 0, factor product at gamma = 1 |
| Grounding vs co-adaptation | Sham-model control (T(s,a) = s) does worse; gate can still zero a subtree | Over-reliance at K = 1, fixed by masking u_tree with p = 0.2 | Copy-model control "far worse"; nothing forces use | Model trained separately on real data only | None; summary may ignore weights (untested) | Fixed Bayes update forces weights to matter; f is ungrounded | MH correction: learned part changes speed, not the limit | absent | chi = 0 and fixed beta available; no rule; beta -> 0 lets attention ignore an arc |
| Computation as evidence | Anytime loss after every simulation, r_m = -(l_m - l_(m-1)) | Step counter k, no calibration | absent (retry loop acts on success only) | Counters plus per-step cost tau | absent (log mean weight is only a loss) | absent; accuracy grows with test-time K | absent | absent (failure = reward 0) | absent; Lemma 1: more arcs never lower inconsistency |
| Training signal | Imitation of oracle MCTS labels (about 2000 sims); pathwise + REINFORCE | Outcome RL (IMPALA) plus supervised model losses | A3C on reward; model frozen | Three independent losses; REINFORCE manager; BPTT via SVG | A2C plus ELBO, end to end | MSE on the true state at every step (privileged) | ESJD plus reciprocal stuck penalty; energy only | Imitation of symbolic traces, then STaR / APA on outcomes | Inconsistency only; no outcome loss |
| Guarantees / limit | None; MCTS convergence lost | None | None | None | Weights exact relative to a co-adapted model | Bayes-consistent relative to learned f only | Detailed balance (exact stationary distribution) | None | Lemma 1 monotonicity; convexity only in special cases (Prop 1); convergence open; NP-hard in general, polynomial at bounded treewidth |

## 2. Per component: the strongest existing design, its evidence, and what changes for Kelvin

### 2.1 Forward model / target
Strongest design: ibp's separation (model trained on real data only, never through the planner's loss), with i2a's staging (pretrain by likelihood, freeze, then train everything else on outcomes).
Evidence: indirect. i2a's copy-model ablation shows the frozen model is used. In i2a Sec 4.2, under a poor model the fixed Monte Carlo planner "degrades catastrophically" while the learned summarizer "performs similarly well" (single seeds, confounded by the extra model-free path). dvrl is the counterexample: RL gradients into the model help return (the "No joint optim" ablation is worse on ChopperCommand), but the weights are then only correct relative to a task-shaped model.
What changes: Kelvin's model must give likelihoods, not next states, because the importance weights are the grounded input. Rule: only likelihood or outcome proper-score gradients reach the forward model, never aggregator or policy losses.

### 2.2 Proposal / inverse
Strongest design: l2hmc and dvrl. A learned proposal plus a fixed exact correction (MH ratio, or weight p/q) means a bad proposal costs efficiency, never correctness.
Evidence: l2hmc, ESS per MH step against tuned HMC: 50-d ICG 0.783 vs 0.0165, strongly correlated Gaussian 0.497 vs 0.00469, mixture of Gaussians 0.0324 vs much less than 2.61e-4 (HMC never leaves one mode). Note that l2hmc's own loss can be gamed: plain ESJD "can lead to regions of state space where almost no mixing occurs", which they patch with a hand-shaped reciprocal term.
What changes: discrete bits have no Jacobian or leapfrog. The any-order network gives exact factor probabilities along an order, so the order plays the role of l2hmc's masks, chosen by the policy instead of fixed and random. Distilling the search's weighted outputs into q (thinker and i2a distil the planner's actions into the base policy; lir Prop 2 is the EM version) transfers formally.

### 2.3 Search policy over variables and roles
Strongest design: mctsnet's modulated logits psi = w0 psi_p + w1 u(parent, child), together with thinker's fully learned imaginary-action plus reset policy.
Evidence:
- mctsnet Sokoban: 84 +/- 1% solved at M = 25, against about 30% for UCT and about 65% for PUCT at 25 simulations. But a random simulation policy "already performs reasonably" because the learned backup and readout compensate (Fig 4a; exact numbers only in plots [unverified]).
- thinker: 95% of Sokoban levels at 5e7 frames (DRC 93%, raw MDP 56.7%). Random imaginary actions make performance "significantly deteriorate"; removing the base policy/value heuristic means it "fails to learn".
- ibp: a learned manager gets about 25% closer to optimal reward than 1-step or n-step at equal budget on 5x5 mazes ("preliminary"), but the full tree was not trainable and routes were cut to 3.
What changes: Kelvin's action space is (node x variable x role), combinatorial and variable-sized. That is exactly what ibp could not train. A pointer or attention head over candidates is needed, and none of these papers tests one. In LIR terms the policy is Refocus. The one published adaptive rule is the GFlowNet surprisal attention phi = log P_B / P_F (lir Sec 4.6), a fixed formula. Hub foci (all arcs at one variable) resolved 99.37% vs 72.19% for random half-graphs on tiny synthetic PDGs: weak evidence for variable-centred foci.

### 2.4 Search state
Strongest design: thinker. A fixed grounded summary (per-node mean and max return, visit count, step counter) plus a learned recurrent memory.
Evidence: thinker App E: "either the hints or an RNN agent is sufficient"; with neither, "the agent fails to learn to plan". An MLP with hints matches the RNN at about 3 vs 7 days of compute. mctsnet's learned per-node memory carries the load even under a random policy. sos's raw trace works on Countdown but is unbounded (5-input traces reach about 60k tokens against a 4096 context).
What changes: the grounded part should be importance-weight statistics per node: log-sum-exp of weights, max weight, effective sample size, branches refuted (zero weight or constraint violation), compute spent. LIR adds a defined quantity nobody uses: residual inconsistency per focus. Restricting attention gives a lower bound on global inconsistency (Lemma 1); restricting the family of mu gives an upper bound (Loss paper Sec 6). dvrl and pfnet show the trap: resampling deletes exactly the failure record that "found nothing" evidence needs.

### 2.5 Aggregation (backup) into a coherent state
Strongest learned design: mctsnet's gated residual backup. Strongest fixed design with semantics: lir's resolved joint.
Evidence:
- mctsnet: gated residual beats a plain MLP backup at M = 10, and the plain MLP is "simply numerically unstable" for M > 25. There is no direct learned-vs-mean-backup ablation.
- i2a: learned vs fixed summarizer as in 2.1 (confounded, single seeds).
- pfnet: a fixed Bayes aggregator generalizes across budget without retraining. Trained at K = 30; global localization success at K = 500 / 1000 / 2000 / 3000 is 46.1 / 57.9 / 68.5 / 76.3%.
- No paper compares mixture vs log pool empirically (lir reader).
What changes: none of the learned aggregators output a distribution, so none is coherent. The coherent version is: the fixed semantics hold the form, and the learned part sets beta on heuristic arcs, chi on belief parameters, and the unexplored-mass term (Section 0). A full-control LIR step on an exponential-family belief arc is moment matching (prices = resolved marginals), and a small-chi step changes natural parameters by -chi * beta_b * (E_b[phi] - E_mu*[phi]), an LMSR-like trade [derivation, lir reader]. Training on outcomes requires differentiating through the argmin mu* (implicit differentiation). LIR's envelope-theorem gradient covers only the inconsistency value, not mu*.

### 2.6 Grounding against co-adaptation
Strongest design: l2hmc's structure (the learned part cannot change the fixed point), plus the standing controls of i2a / mctsnet (sham or copy model) and thinker's input masking.
Evidence: thinker App D: at K = 1 the critic "overly relies on this estimation, thereby neglecting the real state"; masking the tree summary with p = 0.2 "restored the performance". That is over-reliance on the search summary, the opposite failure from ignoring it, so both directions must be monitored. The mctsnet sham model and the i2a copy model both show a drop (exact mctsnet numbers in plots only [unverified]).
What changes: in PDG form, forward-model and evidence arcs get chi = 0 and fixed beta, so learned attention can only re-weight heuristic arcs. This blocks "ignore the forward model" by construction. It does not block "ignore the search", because with the forward arcs fixed, a learned unexplored-mass term could still absorb everything. The guard must be the test, not only the structure: belief quality must improve monotonically with test-time budget (the pfnet Table 1c pattern) and must drop under a sham forward model.

### 2.7 Computation as evidence
Strongest design: mctsnet's anytime credit assignment. Score the readout after every simulation and treat the decrease in loss as the reward for that simulation, with gamma < 1 (gamma = 1, the true objective, was not best).
Evidence: only the gamma curves (Fig 4b, plots only). ibp's tau sweep shows a policy that cuts imagination smoothly as step cost rises (1e-4 to 1, plots only).
What changes: nobody calibrates the null result. Kelvin needs a cpd p(search outcome O | true state) for "branch searched with budget n, no high-weight completion", scored on outcomes at every budget. Section 0 argues this is the main legitimate place for learned aggregation.

### 2.8 Training signal
Strongest outcome-only designs: thinker (augmented MDP, imagined steps earn 0, only the real outcome pays) and sos (imitation, then STaR / APA on outcomes: 51.27% to 56.52% on seen targets).
What changes: the reward is a strictly proper score of the committed coherent belief, which is a supervised loss and lower variance than RL return. Policy gradients for the search policy stay high-variance; mctsnet's lower-variance labels came from an oracle search, which Kelvin's rules exclude. pfnet's ablation shows why the loss must be proper: MSE on the mean scored 39% vs 67% for a probabilistic loss under high uncertainty, because "our loss can be dominated by the distant particles".

### 2.9 Guarantees / limit
Strongest: l2hmc's exactness via detailed balance, and the trivial SMC fact that adaptive proposals keep the weights unbiased given the proposal used (standard theory, not from these papers [unverified as a paper claim]). lir: Lemma 1 monotonicity, convexity only for constant or unconditional log-concave parameterizations with beta >= gamma * alpha (Prop 1), argmin at chi = infinity conjectured in general, convergence open.
What changes: the logical-induction limit (derived constraints enter as beta = infinity arcs; belief at compute t coherent with all of them) is expressible in PDGs and proved by nobody. Note zero-forcing: a beta = infinity arc with P_a(x) = 0 forces mu*(x) = 0, which is right for derived constraints and wrong for an overconfident heuristic.

## 3. What must be invented (deduplicated, ranked)

1. **An outcome-trained aggregator whose output is a coherent bounded-treewidth state.** Closest: lir (the right fixed semantics, but no outcome loss and no learning of phi, chi); mctsnet (learned backup, but a free vector with no joint, trained by imitation); dvrl (learned summary that reads importance weights, but outputs a policy feature). Shortfall: nobody differentiates a proper score through the resolution argmin mu*.
2. **A calibrated model of unexplored posterior mass (computation as evidence), conditioned on the search state.** Closest: mctsnet anytime loss (scores every budget, no null-result model); ibp counters and tau (compute-aware, uncalibrated); pfnet test-time K scaling (monotone, fixed aggregator). Shortfall: in all of them, "found nothing" is not a likelihood about the search process.
3. **Per-report combination semantics and a trust rule.** Monte Carlo reports combine by weight (mixture over discovered support); heuristic opinions by geometric pool; independent evidence by product. beta should come from ESS, variance and correlation between rollouts that share proposals or prefixes. Closest: Loss paper Lemma 10 and Sec 8 (two of the three cases, one global gamma). Shortfall: beta means "effective number of independent reports", and correlated rollouts violate that; no rule exists.
4. **A grounding structure plus standing tests against co-adaptation in both directions.** Closest: l2hmc (exact correction), pfnet (fixed Bayes), mctsnet sham / i2a copy-model controls, thinker masking. Shortfall: no paper has both a structure that blocks discarding the forward model and a test that the aggregator still uses the search.
5. **A learned Refocus over (node x variable x role) with priors from the any-order network** (entropy, estimated influence on the query, pruning power, residual local inconsistency). Closest: mctsnet (fixed small action set), thinker (actions in time order), ibp (failed beyond 3 routes). LIR names it open.
6. **A bounded search-state summary that keeps failures.** Closest: thinker hints plus memory; mctsnet per-node h; sos trace. Shortfall: either unbounded, or failures deleted by resampling, or no weight statistics.
7. **Proposal learning from the search's own weighted outputs for a discrete any-order network.** Closest: thinker / i2a distillation (no IS), lir Prop 2 (EM, exact mu*), l2hmc-DLGM (q trained by ELBO, not distilled). Shortfall: none distils importance-weighted posterior samples into an any-order proposal.
8. **Variance control for outcome-only training of the policy.** Closest: mctsnet gamma < 1 telescoping. Shortfall: tested with oracle labels, not single outcomes.
9. **A logical-induction-style limit theorem.** Closest: lir (beta = infinity arcs, Lemma 1). Shortfall: no convergence or coherence-in-the-limit result anywhere.

## 4. Proposed experiments (proposals only, nothing to run)

### E1. Is there a gap to learn? Fixed aggregation semantics vs learned trust, on ShipHunt
Setup: fixed forward model; any-order proposals; rollouts from a fixed simple policy at budgets M in {4, 16, 64, 256}. Aggregators:
- (a) exact forward scoring of the discovered support, SNIS-normalized (the Monte Carlo rule from Section 0);
- (b) geometric pool of per-rollout marginals (LIR gamma = 0);
- (c) linear mixture of per-rollout SNIS estimates;
- (d) (a) plus learned beta on the any-order arcs and a learned unexplored-mass term, both conditioned on search-state statistics;
- (e) a free network readout (mctsnet / i2a style).
Train (d) and (e) on log score of revealed boards only. Evaluate log score on outcomes, and KL to the exact posterior (evaluation only).
Hypothesis: (a) beats (b) whenever rollouts find different ship configurations; (d) improves on (a) only at small M; (e) matches (d) at the training M but stops improving or degrades at larger test M (the pfnet K pattern, the mctsnet instability above M = 25).
What it means:
- If (a) is already near the exact posterior at realistic M, ShipHunt leaves nothing for a learned aggregator. Move to harder instances or larger boards before building anything.
- If (b) loses badly to (a), the default LIR opinion-arc encoding is wrong for Monte Carlo reports, and rollouts should enter as support, not as arcs.
- If (d) helps mainly through the unexplored-mass term, Section 0's reframing holds.

### E2. Is "searched and found nothing" calibrated? ShipHunt null-result cases
Setup: select instances and budgets where the search fails to find the true configuration. Compare (a) against (d) from E1, with and without the search-state inputs (branches refuted, compute spent, ESS).
Hypothesis: the search-state-conditioned unexplored-mass term reduces calibration error (reliability diagrams, log score) on null-result cases without hurting found cases.
What it means:
- If it helps only when it reads compute spent and refutations, the search state carries evidence beyond the weights. This is the first measurable instance of computation as evidence.
- If it helps equally without them, the gain is an amortized prior, not search evidence.

### E3. Does the learned system use the search, and does the policy matter? ShipHunt plus Sudoku
Controls on the E1 system:
- (i) sham forward model (constant weights) at equal compute;
- (ii) shuffled rollouts;
- (iii) random search policy vs entropy-based policy vs learned policy at fixed M;
- (iv) test-time budget sweep;
- (v) Sudoku: derived constraints entered as beta = infinity arcs.
Hypothesis: the score drops under (i) and (ii); the score is monotone in M; the learned policy beats random by more than the mctsnet margin; on Sudoku, after t derivations, the belief puts no mass on assignments that violate derived constraints.
What it means:
- No drop under (i): the aggregator ignores the search, which is the co-adaptation failure.
- Learned policy close to random: as in mctsnet, the value is in the aggregator, not in the policy, and the policy work should be deprioritized.
- Mass on violating assignments in Sudoku: the coherence-with-derived-facts property fails even with hard arcs, and the beta = infinity encoding needs checking.

## 5. Reading order for Kelvin

1. **lir, 2604.17140**: Sec 2 to 3 (definitions, attention vs control, Algorithm 1, Prop 1), Sec 4.2 (EM, Prop 2), Sec 4.4 (BP, Prop 4), Sec 4.6 (surprisal attention), Sec 6 (open questions). Then **2202.11862** Sec 8 and Lemma 10 (how two sources combine, and why not a mixture) and Lemma 1. This decides the semantics question.
2. **mctsnet, 1802.04697**: Sec 3.4 to 3.6 (gated backup, loss, anytime credit assignment) and Sec 4.2 to 4.3 (backup stability, random policy, gamma). This is the closest learned-backup design and the most reusable training trick.
3. **thinker, 2307.14993**: Sec 4.2 to 4.4 (tree representation, learned improvement operator), Sec 5.1 (hints vs memory, heuristic necessity) and App D (over-reliance and masking). This is the best evidence on learned search state and co-adaptation.

Optional: i2a Sec 4.2 (learned vs fixed summarizer under a bad model); pfnet Table 1c and Sec 5.3 (budget generalization, proper loss).

## 6. Honest assessment

The loose version, "learn the MCTS backup and a search-state vector", is incremental: mctsnet, thinker, ibp and i2a have done it for planning, without coherent beliefs. The narrow version is well posed and not done by any of the 9 papers: fix the PDG semantics (forward arcs at chi = 0, evidence clamped, discovered support scored exactly), learn only the Refocus policy, the trust on heuristic arcs, and a search-state-conditioned model of unexplored mass, all trained by the proper score of the coherent state on outcomes. LIR names the learned Refocus as open, and nobody calibrates null search results. The single most likely failure mode: on toys with an exact forward model and small k, the fixed rule (exact scoring of discovered hypotheses) is already close to the exact posterior at modest budgets. The learned parts then have almost no gap to close, and the noisy outcome-only gradient lets them collapse onto either the fixed rule or the amortized any-order prior. The result would read as "learning adds nothing" rather than as a test of the idea. E1's first question (is there a measurable gap between the best fixed rule and the exact posterior at realistic budgets?) should be answered before anything is built.
