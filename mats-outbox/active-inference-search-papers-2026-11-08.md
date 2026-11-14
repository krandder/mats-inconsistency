# Search in active inference: three papers read against Kelvin's plan (2026-11-08)

Kelvin asked (2026-11-08): "Please study the three papers carefully first."

**Sources read in full** (arXiv sources fetched 2026-11-08 into `/tmp/aif-papers/src/`):
- Friston, Da Costa, Hafner, Hesp, Parr, *Sophisticated Inference* (SI), arXiv 2006.04120. Only a PDF exists. It has numbered equations but no numbered sections, so I cite the section headings. I read the text with pypdf and checked equations (1.10), (1.11) and (1.13)–(1.15) against rendered page images.
- Fountas, Sajid, Mediano, Friston, *Deep active inference agents using Monte-Carlo methods* (DAIMC), arXiv 2006.04176, from the TeX source. I also read the authors' public code (`zfountas/deep-active-inference-mc`): the original TF commit `bec21c8` (2021-12-01) and the current PyTorch master.
- Champion, Da Costa, Bowman, Grześ, *Branching Time Active Inference: the theory and its generality* (BTAI), arXiv 2111.11107, from the TeX source. Its companions, also from TeX: *BTAI with Bayesian Filtering* (BTAI-BF), arXiv 2112.07406, and *BTAI: empirical study and complexity class analysis* (BTAI-emp), arXiv 2111.11276.
- Background: Da Costa et al., *Active inference on discrete state-spaces: a synthesis*, arXiv 2001.07203. I grepped and read the relevant sections only.

**Labels.** PROVEN = stated or derived in the paper. MEASURED = the paper's own experiments. CODE = read in the authors' public code. OUR READING = my interpretation, open to dispute.

---

## 1. Sophisticated Inference (Friston et al., 2006.04120)

### Objective
- PROVEN. The baseline is the generalised free energy over policies (1.1), with priors over policies `ln P(π) = −E(π) − G(π)`. Expected free energy is risk plus ambiguity (1.7):
  `G(π) = D_KL[Q(s_τ,A|π) || P(s_τ,A)] + E_Q[−ln P(o_τ|s_τ,A)]`.
  The intrinsic/extrinsic form is (1.9). With novelty (parameter information gain) it is (1.18).
- PROVEN. The sophisticated (recursive) form is (1.11), section "Sophisticated inference":
  `G(o_τ,u_τ) = E_{P(o_{τ+1}|s_{τ+1})Q(s_{τ+1}|u_{<τ+1})}[ln Q(s_{τ+1}|u_{<τ+1}) − ln P(s_{τ+1}) − ln P(o_{τ+1}|s_{τ+1})] + E_{Q(u_{τ+1}|o_{τ+1})Q(o_{τ+1}|u_{≤τ})}[G(o_{τ+1},u_{τ+1})]`,
  with `Q(u_τ|o_τ) = σ[G(o_τ,u_τ)]` and `Q(o_τ|u_{<τ}) = E_{Q(s_τ|u_{<τ})}[P(o_τ|s_τ)]`. The softmax is printed as σ[G]. OUR READING: given (1.1) and (1.6) it means σ[−G].
- PROVEN. In sufficient statistics, (1.13):
  `G(u_τ,o_τ) = s^u_{τ+1}·[ln s^u_{τ+1} + C + H] + u^o_{τ+1}·G(u_{τ+1},o_{τ+1}) o^u_{τ+1}`,
  where `s_τ ∝ (A·o_τ)⊙s^u_τ`, `s^u_τ = B(u_{τ−1}) s_{τ−1}`, `o^u_τ = A s^u_τ`, `u^o_τ = σ[G(u_τ,o_τ)]`, `C = −ln P(s_τ)` and `H = −diag(A·ln A)`. (1.15) is the version with preferences over outcomes, `C = ln P(o_τ)`. (1.16) averages over outcome distributions instead of enumerating outcomes.
- PROVEN. Comparison to Bellman: "Readers familiar with the Bellman optimality principle ... may recognise a formal similarity between (1.11) and the Bellman equation ... The sophisticated inference scheme deals with functionals (functions of belief distributions over states), while the Bellman equation deals directly with functions of states." The introduction says SI "recovers Bayes-adaptive reinforcement learning ... in the zero temperature limit". That is asserted, not derived in this paper.

### What is searched and how
- PROVEN. The search is over interleaved actions and outcomes, i.e. over sequences of belief states. Abstract: "this search is over sequences of belief states, as opposed to states per se". It is a full recursive (depth-first) evaluation. Branches are pruned by fixed thresholds: "we can restrict the search over future outcomes ... that have a non-trivial posterior probability (e.g., greater than 1/16). Similarly, we only need to evaluate those policies ... with an expected free energy that renders their prior probability nontrivial (e.g., greater than 1/16)." In the simulations: "searches over paths were terminated when the predictive probability fell below 1/16."
- PROVEN. Depth is 1–4 in the experiments. Branching is 4 actions (T-maze) or 5 actions on 64 states (maze). On cost: "there were five actions and 64 hidden states leading to a large number of potential paths (1.0486·10^10 for a planning horizon of four and 1.0737·10^15 for a planning horizon of six). However, only a tiny fraction of these paths is actually evaluated – usually several hundred, which takes a few hundred milliseconds on a personal computer" (Conclusion).
- OUR READING. The order of computation is fixed (exhaustive recursion), and the only "choice" of what to compute is the 1/16 threshold. There is no learned or adaptive choice of which branch to expand next. The paper does not say what a pruned branch contributes to the parent's average: zero, or renormalised away.

### Distillation or amortisation
- PROVEN. None for planning. Amortisation is discussed only for perception, (1.5): "amortized inference is 'learning to infer' ... the cerebellum might learn from inferential processes implemented by the cerebral cortex". The planner is re-run from scratch at every step.

### Preferences
- PROVEN. Preferences are a prior over future (final) states, `C = −ln P(s_τ)` (1.14), or over outcomes, `C = ln P(o_τ)` (1.15). They enter only the risk term of predicted states or outcomes. Text on (1.1): "outcomes on the horizon are random variables with a likelihood distribution, whereas outcomes in the past are realised variables."
- PROVEN, from the update equations. The current-state posterior (1.12), `Q(s_t) = P(o_t|s_t)Q(s_t|u_{t−1})`, contains no C. Preferences touch only future predictions.

### Observations
- PROVEN. An exact Bayesian filter with no mean-field, (1.10): "This scheme is exact because we have made no mean-field approximations". Parameters A and D are learned by Dirichlet count accumulation (1.17). Inside the tree, the same filter is applied counterfactually to each plausible outcome. That is the "beliefs about beliefs" part.

### Experiments (MEASURED; small, deterministic, no external baselines)
- **T-maze.** 32 trials of 2 moves, 4 actions, cue validity 95%, costs −2/+2, MAP action selection. With horizon 2 the agent switches from exploration to exploitation at trial 16; with horizon 1, at trial 10, with "less confident" early choices. With the cue made mildly aversive (cost 1), the sophisticated agent still visits the cue (switch at trial 12), while the horizon-1 agent "just stays at the starting position and learns nothing" (Figs. 3–4).
- **8×8 maze with known aversive cells.** Horizons 1 and 3 get stuck behind a barrier; horizon 4 finds the shortest path (Fig. 5).
- **Same maze, unknown aversive cells** (Dirichlet prior 1/64, no location preference). In 64 moves the agent visits nearly every cell. A "Bayesian risk" agent with no novelty term returns to the start and stays (Fig. 6). With location preferences restored (5 exposures × 8 moves, horizon 4), it plans the shortest path after 4 exposures (Fig. 7).
- Baselines are ablations only (horizon 1, and risk without epistemic terms). There are no seeds or variance (deterministic), and no comparison with POMDP solvers or RL.

### Limitations
- PROVEN. The authors' own words: "we have only illustrated this scheme on rather trivial problems". The cost is exponential in depth apart from pruning.
- OUR READING. The pruning thresholds are unprincipled with respect to compute (no value of computation). There is no learning of the planner. Actions are discrete and models tabular.

---

## 2. Deep active inference with Monte-Carlo methods (Fountas et al., 2006.04176, NeurIPS 2020)

### Objective
- PROVEN. Eq. (2) is `G(π,τ) = E_{P(o_τ|s_τ,θ)} E_{Q_φ(s_τ,θ|π)}[log Q_φ(s_τ,θ|π) − log P(o_τ,s_τ,θ|π)]`. Policies are selected by (3), `P(π) = σ(−Σ_{τ>t} G(π,τ))`, and actions by (5), `P(a) = Σ_{π:a_1=a} P(π)`.
- PROVEN. The decomposition (7a–c) is: (a) `−E[log P(o_τ|π)]` ("plays an analogous role to the notion of reward"); (b) state information gain; (c) parameter information gain. It is re-written as entropies for Monte-Carlo estimation in (8a–c), with θ sampled by MC dropout.
- PROVEN. Perception and learning minimise the per-step VFE (4): `F_t = −E[log P(o_t|s_t)] + KL[Q(s_t)||P(s_t|s_{t−1},a_{t−1})] + E[KL[Q_{φa}(a_t)||P(a_t)]]`.

### What is searched and how
- PROVEN. The search is MCTS over latent-state trajectories (Sec. 3.2). The selection score is (9), `U(s,a) = G̃(s,a) + c_explore · Q_{φa}(a|s) · 1/(1+N(a,s))`, a PUCT form with the habit network as prior. The paper prints `+G̃`; BTAI restates it as `−G̃`, and the code stores `W −= G`. "Actions are drawn from `P(a_t) = N(a_t,s_t)/Σ_j N(a_{j,t},s_t)`." Planning stops early "if `max P(a_t) − 1/|A| > T_dec`" or at the loop budget.
- PROVEN. Settings (Supp.): `c_explore = 1`, `T_dec = 0.8` by default, "the depth of MCTS simulation rollouts was set to 3", at most 300 loops (dSprites) or 100 (Animal-AI). There are 4 or 3 actions and a 10-d latent.
- PROVEN, Sec. 5 claim with no matching mechanism in Sec. 3. "MCTS offers an Occam's window for policy pruning; that is, we stop evaluating a policy path if its EFE becomes much higher than a particular upper confidence bound."
- CODE (`src/mcts.py`, same structure in the TF and PyTorch versions):
  - The leaf value is a single rollout of depth 3. Its actions are sampled from the habit network (`mcts_step_simulate` → `model_top.encode_s`).
  - The same leaf value is added to every node on the path (`W −= G; N += 1`), and `Q = W/N` is a plain mean.
  - The habit prior in the selection score is off by default (`using_prior_for_exploration=False`, never set in `test_demo.py`).
  - There is an optional "habit-first" shortcut that skips planning when the habit is confident (`use_habit`). Its CLI flag is inverted: `params.use_habit = args.no_habit`.

### Distillation or amortisation
- PROVEN, two inconsistent descriptions.
  - Sec. 3.2: "this network is trained to reproduce the last executed action `a_{t−1}` (selected by sampling `P(a_t)`) using the last state `s_{t−1}`."
  - Supp. Algorithm 1: "Run planner and compute prior policy `P(a_t)` … Apply a gradient step on `φ_a` using `D_t = KL[Q_{φa}(a_t)||P(a_t)]`."
  - So on paper the "habitual network" distils the planner's action distribution (MCTS visit counts) with a KL loss. This is AlphaZero-like.
- CODE (original TF `train.py`, commit `bec21c8`). The habit target is not MCTS. It is `softmax(−sum_G)` from `make_batch_dsprites_active_inference(..., deepness=1)`, i.e. a one-step EFE of four "repeat one action" policies, each step spanning 5 environment frames. OUR READING: the released dSprites training distils a one-step EFE, not the tree search. The paper's text does not let one tell which target produced the reported curves. Animal-AI was trained off-policy on data from "a simple rule: move in the direction of the greenest pixels".
- PROVEN, a coupling to note. The habit–planner disagreement `D_t = KL[Q_{φa}||P(a)]` sets the transition precision (10): `ω_t = α/(1+exp(−(b−D_{t−1})/c)) + d`, which is fixed and logistic. It is called "top-down attention", a scalar precision on `P(s_t|s_{t−1},a_{t−1})`. OUR READING: this is a hand-set gain on the transition prior, not attention over computations and not a learned trust.

### Preferences
- PROVEN. "we encode the actual reward from both environments as the prior distribution of future expected observations `log P(o_τ|π)`". Reward is drawn into the top pixels of the observation, so the decoder predicts it. Preferences enter only term (7a)/(8a) on predicted future observations. The current state comes from the encoder `Q_{φs}(s_t)` given `o_t`, with no C.

### Observations
- PROVEN. The amortised encoder `Q_{φs}(s_t)` is computed from the current frame alone. In the lights-off test, `P_{θs}` "is defined as a feed-forward network, without the ability to maintain memory of states before t." The transition prior enters only through the training KL. Planning uses state means (code: `use_means=True` in expansion).

### Experiments
- MEASURED, dynamic dSprites (4 actions, "object sorting").
  - Agents with three EFE variants and a reward-maximising baseline collect similar reward (Fig. 3B). The AIF agents explore more (3C).
  - The AIF agents reconstruct reward better (3D). Agents using ω show more disentanglement (lower total correlation, 3F).
  - Performance peaks at `T_dec ≈ 0.8` (4Aiii).
- MEASURED, Animal-AI "preferences" task.
  - Best at 16 MCTS loops and `T_dec ≈ 0.1` (4Bii). Lights-off performance degrades with blackout length (4Biii).
  - Against OpenAI-baselines DQN, A2C and PPO2 at default settings and 2M iterations, it is "considerably better than DQN and A2C, and is comparable to PPO2". Authors: "not ... a thorough benchmark".
  - Compute: training about 26 h per agent; MCTS evaluation curves about 4 days (Supp.).
- Not shown: any ablation of search depth or loops against the habit alone on return with confidence intervals, beyond Fig. 3A; any measure of how well the habit approximates the planner. MEASURED elsewhere: BTAI-emp ran the released DAIMC on dSprites and got 84.1% solved (GPU), 79.8% (CPU).

### Limitations
- PROVEN. The authors say it is "not clear whether deep active inference ... has significant performance benefits over simpler reward-seeking agents".
- OUR READING.
  - The value estimates come from depth-3 rollouts under the habit, so search quality is tied to the policy being distilled.
  - Visit-count actions depend on `c_explore`, `T_dec` and the rollout policy.
  - The text and the public code disagree on the habit target and on whether the PUCT prior is used.

---

## 3. Branching Time Active Inference: theory (Champion et al., 2111.11107)

### Objective
- PROVEN, Sec. 5.3.3. The node cost is any of:
  - `g^classic_I = KL[Q(O_I)||V(O_I)] + E_{Q(S_I)}[H[P(O_I|S_I)]]` (risk + ambiguity);
  - `g^feef_I = KL[Q(O_I,S_I)||V(O_I,S_I)]`;
  - `g^pcost_I = KL[Q(S_I)||V(S_I)] + KL[Q(O_I)||V(O_I)]`.
- PROVEN, propagation (Sec. 5.3.4).
  - Forward: `G^aggr_I ← g_I + G^aggr_K`.
  - Backward (eq. backprop): `G^aggr_J ← G^aggr_J + g_I ∀J ∈ A_I`.
  - Min-backward (eq. minbackprop): `G^aggr_J ← G^aggr_J + min_a g_{K::a}`.
  - The average is `ḡ_J = G^aggr_J / n_J`.
  - "The forward propagation ... will only be used for theoretical purpose ... Practical implementation of BTAI should use the backward schemes."
- PROVEN, Thm 1 and Sec. 6. Forward propagation equals classic active inference's EFE "under the assumption that the probability of observations and states are independent of future actions". Backward propagation is only *related* to SI: "the second term of the RHS of (SI) is an expectation, while the second term of the RHS of (BTAI) is a summation over the children that have already been expanded … in SI the aggregated EFE … is weighted by the probability of the next action and states … there are no such weights in BTAI".

### What is searched and how
- PROVEN. The search is incremental tree expansion, framed as "planning as structure learning": the generative model itself grows by adding `S_I` and `O_I` nodes. Node selection uses UCT, `UCT_J = −ḡ_J + C_p √(ln n / n_J)`. Expansion adds all children. Inference over the new nodes is local or global VMP. Cost evaluation can use optional uniform-random rollouts (Algorithm 1).
- PROVEN. The paper keeps the two explorations apart: "The UCT criterion can be understood as a trade-off between exploitation and exploration at the tree level, which is different to the exploitation and exploration dilemma at the model level. This dilemma is handled by the EFE."
- OUR READING. The tree branches over actions only, never over observations. Future observations are latent variables whose posterior is the prediction (Sec. 5.2: `Q*(O_I) = σ(Å ⊙ D̂_I)`; BF version: `B(S_J) = E_{B(S_I)}[P(S_J|S_I)]`). So a BTAI plan is open-loop within a branch. It does not do SI's counterfactual "what would I believe after seeing o". Nuijten & de Vries (2026, below) argue this closed-loop property is SI's real advantage.
- PROVEN, complexity (BTAI-emp Sec. 3.1). Classic AcI needs `O(|π|×T×|S|)` with `|π| = |U|^T`. BTAI needs `O([K+t]×|S|)` for K expansions.

### Distillation or amortisation
- None. Future work: "use neural networks as function approximators ... could be directly compared to the method of [Fountas]". The tree is discarded after each action (BTAI-emp conclusion: "all the branches in the tree are pruned").

### Preferences
- PROVEN, and one of the paper's points. There are three distributions: "the prior defines the agent's beliefs before sampling any observation; the posterior is an updated version of the prior ...; finally, the target distribution encodes the agent's prior preferences in terms of future observations and hidden states". Here `V(O_I,S_I) = Π V(O_I)V(S_I)` over expanded future nodes only (eq. target distribution). Preferences are therefore a separate object from the generative model. They never enter the posterior over past or present states.

### Observations
- PROVEN. VMP over the whole factor graph, past and tree (Sec. 5.2), with Dirichlet learning of A, B and D. BTAI-BF replaces this with a forward Bayes filter: "BTAI with BF ran 69 ± 26 times faster than BTAI with VMP".

### Experiments
- MEASURED in the theory paper: only App. H. SPM run time for planning N = 2…6 steps grows exponentially, and N = 7 "crashed after trying to allocate an array of 9.5GB".
- MEASURED, BTAI-emp. The model (A, B, D) is given to the agent in all its experiments.
  - **Deep reward graph.** SPM AcI solves policy sizes 3 and 5 (1177 s for 5) and crashes at 8. BTAI solves all three, at 10, 10 and 20 iterations (≤0.84 s).
  - **Mazes with Manhattan-distance observations.** At 10–15 iterations the agent is stuck in a local minimum; at 20 it escapes. State preferences (`g^pcost`) let it escape at 15. `g^classic` and `g^pcost` are equal there because every column of A has the same entropy.
  - **Frozen lake.** BTAI needs 20 or 50 iterations. POMCP needs about 1000 for 97%. Cumulative reward is similar, and which agent wins depends on episode length.
  - **Simplified dSprites** (tabular, model given). 96.9% solved, against 84.1% for released DAIMC on the original task, "an unfair comparison" in substance: the authors list the differences themselves.
- MEASURED, BTAI-BF. Deep reward graphs (n = 2–3 good paths, m = 5 bad) are solved with P(goal) = 1 at 25–100 iterations.

### Limitations
- PROVEN. Tabular only. Number of iterations hand-set (BTAI-BF future work: "automatically identifying this optimal number of planning iterations").
- OUR READING. There is no convergence statement linking the backed-up sum to any well-defined objective. Action selection is `σ(−γ g/N)` (emp) or "lowest average cost" (BF).

---

## 4. Background check: where the "LTP / dreaming" language comes from (synthesis, 2001.07203)

- PROVEN, Sec. "Pruning of policy trees". "we stop evaluating the expected free energy of a policy if it becomes much higher than the best ... policy ... Although effective and biologically plausible, the Occam window for pruning policy trees cannot deal with large policy spaces ... Further research is needed to characterise the processes in which biological agents reduce large policy spaces to tractable subspaces." The proposed answer is deep temporal hierarchies, not better search.
- PROVEN, Sec. "Learning". "These synaptic dynamics (e.g., long-term potentiation and depression) evolve at a slower timescale than action and perception". This sentence is about learning the likelihood A, not policies.
- PROVEN, Sec. "Bayesian model reduction". Model reduction is "reminiscent of the synaptic mechanisms of sleep (e.g., REM sleep ...)". It "has been used to emulate sleep and reflection in abstract rule learning". Again this concerns model structure, not policies.
- PROVEN, footnote in Sec. "Planning and decision-making". Habits E are an extra prior over policies, `σ(−log E − F − G)`. Learning E is slow (Dirichlet counts in the cited work).
- Terminology trap. The synthesis's "policy-independent state-estimation" (Sec. 6.2) means a Bayesian model average over policies, `Q(s_τ) = Σ_π Q(s_τ|π)Q(π)`. It has nothing to do with Kelvin's "policy independence of T".

---

## (i) Is Kelvin's claim accurate?

Claim: active inference POMDP models "do not go very deeply on the search to improve policies; they only discuss changing policies as a long-term potentiation thing mediated by dreaming or etc."

**Verdict: partly accurate. Wrong about these three papers; right about depth and about the canonical literature.**

- **Wrong (PROVEN).** All three papers make within-decision search central:
  - SI: a recursive tree over actions and counterfactual belief updates, (1.11)/(1.13).
  - DAIMC: MCTS with visit-count action selection, eq. (9) and `P(a)=N/ΣN`, plus a habit network trained by gradient against the planner's distribution (Alg. 1). This is fast amortisation, not LTP or dreaming.
  - BTAI: UCT-guided incremental expansion of the generative model.
- **Right about the language, wrong attachment (PROVEN).** The LTP and sleep language exists in the canonical discrete-state account (synthesis, Secs. "Learning" and "Bayesian model reduction"). There it attaches to learning A/B/D and structure, not to improving policies. The synthesis treats deep search as an open problem, solved by pruning plus hierarchy.
- **Right about depth (MEASURED).** The searches are shallow and small:
  - SI: depth ≤4, 4–5 actions, 64 states.
  - DAIMC: rollouts of depth 3, ≤300 loops, 3–4 actions.
  - BTAI: ≤50 iterations (100 in BF), with the model given.
  - None studies how quality scales with search compute. None treats the search procedure itself as something to learn or improve, except DAIMC's habit prior in PUCT, which is off by default in the released code (CODE).
- So the accurate version of his sentence is: "they do explicit but shallow, hand-tuned search. Only one distils search into a policy, and none learns *how to search*."

## (ii) Plan components against the three papers

| Kelvin's component | SI (2006.04120) | DAIMC (2006.04176) | BTAI (2111.11107 + companions) |
|---|---|---|---|
| Learned world model W_L | No. Tabular A, B, D with Dirichlet counts (1.17) | **Partly.** Learned VAE + transition + decoder, fixed (forward) order | No. A, B, D given in all experiments |
| Any-order predictor q(x_i \| x_S) | No | No | No |
| Utility calculators inside W_L | Partly. C is a log-prior inside the model (1.14/1.15) | Partly. Reward is drawn into the observation, so the decoder predicts it | Partly. A separate target V (prior, posterior and target kept apart) |
| Searcher optimising T with W_L | **Yes.** Exhaustive recursion with 1/16 pruning | **Yes.** MCTS | **Yes.** UCT expansion |
| T independent of the search policy | Mostly yes, trivially: exhaustive recursion. T does depend on the agent's own future σ(−G), Bellman-style (OUR READING) | **No.** Visit counts and mean backups depend on c_explore, T_dec and habit rollouts (PROVEN + CODE) | **No.** "summation over the children that have already been expanded" (PROVEN); no convergence result |
| "Actions are beliefs" | Yes, as planning-as-inference: policies are random variables with posterior Q(π) (1.1) | Yes, Q_φa(a) is a variational posterior | Yes, actions are latent variables |
| Attention policy π choosing which inference to compute next | No. Fixed recursion | Partly. UCT/PUCT picks which node to expand (bandit heuristic) | Partly. UCT picks which node to expand; tree-level and model-level exploration explicitly separated |
| π̂ = MCTS visit counts | No. Softmax of G | **Yes.** P(a)=N/ΣN | No. σ(−γ g/N); visit counts only mentioned as an alternative |
| π̂ distilled into a fast policy | No. Amortisation of perception only (1.5) | **Yes, on paper** (KL to P(a), Alg. 1). CODE: released dSprites training uses a one-step EFE target | No |
| Value of information guides search | VOI of *observations* (epistemic terms) shapes the tree's values | Same | Same, and EFE drives node expansion. No value of *computation* anywhere |
| Learned trust B over inference steps | No | Weak analogue: scalar precision ω_t on the transition, a fixed logistic of D_t (10). Not learned, not per step | No |
| Joint training of q, B, π on accuracy + inconsistency | No | Partly. VFE (accuracy + complexity) for the model and KL(habit‖planner) for π, with separate optimisers and no inconsistency term | No |
| How observations update beliefs | Exact Bayes filter (1.10); counterfactual updates in the tree | Amortised encoder from o_t alone | VMP over the whole graph, or a Bayes filter (BF); no observation branching in the tree |
| What T is | Recursive EFE (risk + ambiguity [+ novelty]) | Summed EFE (7)/(8) | g^classic, g^feef or g^pcost, summed over the tree |

## (iii) What would be new in his plan, relative to these papers

OUR READING, stated narrowly.

1. **The searcher's actions are inference computations, not environment actions.** Choosing which conditional `q(x_i|x_S)` to evaluate next, inside an any-order and possibly incoherent model, is not done in any of the three. Their trees search environment action sequences. UCT node choice is the nearest thing, and it is a regret heuristic on EFE estimates, not a decision about which belief to compute.
2. **A target T defined independently of the search policy.** SI gets this only by being exhaustive. DAIMC and BTAI do not have it: their outputs are functions of the search trajectory. A plan where the search policy affects cost but not the answer, so that π can be trained freely, is a real difference. It needs T to be a fixed point or optimum of the model alone (Kelvin's earlier "inconsistency minimisation with proximity and prior edges" reference target would be one candidate). Outside AIF, Grill et al. 2020 show AlphaZero's visit counts approximate a *prior-regularised* policy optimum, i.e. a search-policy-dependent target. That is the precise thing his plan would avoid.
3. **Learned, per-step trust B over inference steps, trained jointly with q and π against accuracy plus inconsistency.** Nothing in the three does this. DAIMC's ω is a hand-set scalar gain.
4. **Inconsistency between inference paths as a training signal.** Of the three, only DAIMC's `KL(habit‖planner)` signal is close (used as loss and as precision). It is one fixed pair of paths, not inconsistency across inference orders.
5. **Not new.** Treating actions as inferred beliefs (planning as inference); using information gain/VOI over observations; MCTS visit counts as an improved policy distilled into a fast network (DAIMC on paper, and AlphaZero before it). Metareasoning in general is not new either: Russell–Wefald, Hay et al. 2012, and Callaway et al. 2018 (abstracts below). What is new is its combination with an incoherent any-order belief model and a search-independent target.

## (iv) Most useful pieces to borrow

1. **SI's recursion (1.11)/(1.13) as the reference target for small problems.** It is computable exactly on toy POMDPs. It branches over observations (closed loop), and it is independent of any search order up to pruning. It gives a ground-truth T against which a learned attention policy π can be scored on compute spent versus error, the same role exact NashConv plays on Kuhn. Use it together with BTAI's finding that sum-over-expanded-children diverges from the expectation. That divergence is a ready-made measurement of search-policy dependence: compute BTAI's `ḡ` and SI's G on the same tree under different expansion orders.
2. **BTAI's three-distribution separation** (prior / posterior / target V, eq. target distribution) and its explicit split between "tree-level" and "model-level" exploration. Both map directly onto his split between beliefs q, target T and attention π. Together they stop preferences from leaking into current beliefs and stop compute-exploration from being confused with epistemic action.
3. **DAIMC's experiment design, and what to repair in it.** Report task performance against planner iterations and decision threshold (their Fig. 4Aii/iii, 4Bii) for planner, habit and planner-with-habit-prior. Also log `KL(habit‖planner)` over training. Do not copy the confounds: report the habit's actual training target, use matched compute for baselines, and add seeds with intervals.

## Later work (2022–2026). Each listed only after reading at least its abstract (arXiv API or Europe PMC, 2026-11-08)

**Tree search and scaling in AIF**
- Maisto, Gregoretti, Friston, Pezzulo, *Active Inference Tree Search in Large POMDPs*, arXiv 2103.13860 (2021, pre-window but central). MCTS with an AIF objective; reproduces state-of-the-art POMDP solutions on RockSample.
- Champion, Grześ, Bowman, *Multi-Modal and Multi-Factor BTAI*, arXiv 2206.12503 (2022). Factorised likelihood and transitions; 100% on metadata-dSprites in 2.6 s.
- Champion, Grześ, Bonheme, Bowman, *Deconstructing deep active inference*, arXiv 2303.01618 (2023). Rebuilds DAIMC-style agents. Only reward-maximising variants solve dSprites; the EFE agent collapses to always choosing "down". Directly relevant to DAIMC's claims.
- Paul, Sajid, Da Costa, Razi, *On efficient computation in active inference*, arXiv 2307.00504 (2023). Replaces SI's tree with backward dynamic programming over EFE; grid-worlds.
- Hodson et al., *Sophisticated Learning*, arXiv 2308.08029 (2023). Adds parameter learning inside SI branches (planning to learn); beats SI and Bayes-adaptive RL on a 10×10 foraging task.
- Dao & Peter, *Boosting MCTS with Free Energy Minimization*, arXiv 2501.13083 (2025). MCTS plus CEM at the root with information-gain bonuses; continuous control.
- Cooper & Velasquez, *EFE as Belief-Dependent Utility for ρ-POMDPs*, arXiv 2607.16981 (2026). Proves EFE equals a ρ-POMDP with information-gain utility at weight 1; MCTS-EFE up to 65k states.
- Nuijten & de Vries, *Sophisticated Policies from Epistemic Priors*, arXiv 2607.19518 (2026). Argues SI's advantage is closed-loop control (future actions depend on future observations), not tree search per se.
- Mylonas & Moreno-Bote, *Belief-Based Maximum Occupancy Principle and AIF*, arXiv 2609.39342 (2026). Bellman reformulation of EFE solved by offline value iteration over belief space.

**Amortised planning in AIF**
- Millidge & Buckley, *Successor Representation Active Inference*, arXiv 2207.09897 (2022). Swaps planning for a successor matrix; longer horizon at lower cost.
- Paul, Isomura, Razi, *Predictive planning and counterfactual learning in AIF*, arXiv 2403.12417 (2024). Mixes planning with learning from experience via a bias parameter; the nearest AIF paper to a planning-versus-cached-policy arbitration.
- Taheri Yeganeh, Jafari, Matta, *Deep AIF agents for delayed and long-horizon environments*, arXiv 2505.19867 (2025). A policy network trained on EFE gradients with multi-step latent transitions removes planning from the control loop.

**Planning as variational inference**
- Lázaro-Gredilla, Ku, Murphy, George, *What type of inference is planning?*, arXiv 2406.17863 (2024). Planning is a specific entropy-weighting of variational inference.
- de Vries et al., *EFE-based planning as variational inference*, arXiv 2504.14898 (2025). EFE planning falls out of a VFE with epistemic priors, plus a complexity term for bounded compute.

**Mental action and metacognition in AIF** (the closest to "attention as action"; the relation to choosing *which inference to compute* is weak)
- Sandved-Smith et al., *Computational phenomenology of mental action*, Neurosci. Conscious. 2021. Mental action is policy selection over higher-level attentional states that set likelihood precision; mind-wandering simulation.
- Parr, Holmes, Friston, Pezzulo, *Cognitive effort and active inference*, Neuropsychologia 2023. Effort is covert action that departs from habitual priors; Stroop.
- Sandved-Smith & Da Costa, *Metacognitive particles, mental action and the sense of agency*, arXiv 2405.12941 (2024). Formal, no algorithm.
- Proietti, Parr, Tessari, Friston, Pezzulo, *Active inference and cognitive control: balancing deliberation and habits through precision optimization*, Phys. Life Rev. 2025. A metacognitive level controls the precision that weighs deliberation against habit. This is the AIF version of "when to deliberate", not "which computation".
- I found no AIF paper that selects planning computations by their value (no value of computation). Outside AIF, read now: Hay, Russell, Tolpin, Shimony, *Selecting Computations*, arXiv 1207.5879; Callaway et al., *Learning to select computations*, arXiv 1711.06892; Grill et al., *MCTS as Regularized Policy Optimization*, arXiv 2007.12509.

**Not covered.** I did not read full texts of the later work. Its experimental claims above are as stated in the abstracts.
