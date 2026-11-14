# ML advances that bear on the POMDP + search + trust plan: 24 papers read in full (2026-11-08)

Bead pool-0xcjds. Kelvin asked (2026-11-08): "You can have a subagent do exploration of new papers that are actual advances (by reading them all)". The plan is in `work/plan-photos/TRANSCRIPT.md`: a learned any-order predictor q (W_L), a searcher optimising a target T that must not depend on the search policy, an attention policy π over which inference to compute (guided by VoI, improved to π̂ by search), and a learned trust B over inference steps, trained jointly with q and π on accuracy + inconsistency.

## TL;DR

1. I read 24 papers (2012 to Nov 2026). Each piece of the plan exists somewhere on its own. No paper puts them together, and nobody trains q, B or π on inconsistency across the inference orders of an any-order model.
2. Strongest results: a model-defined target plus a search procedure roughly matches RL (Karan & Du; Yue et al.; ICM); adaptive decoding order in any-order models takes Sudoku from 7% to 90% (Kim et al.); π̂ from search is a prior-regularised optimum (Grill; Gumbel MuZero).
3. Three clarifications change how the plan should be posed. The search target is order-independent only when q is consistent (Kim, SAS, AOP all say this). Visit-count π̂ is not policy-independent (Grill). Rollout-based step trust depends on which policy completes the rollout (PAV).
4. Risk: self-consistency objectives collapse through shared parameters (Ding 2026), and coherence optimisation needed anchoring to avoid mode collapse (Qiu et al. 2026).
5. Read first: Qiu et al. 2601.13566, Kim et al. 2502.06768, Setlur et al. 2410.08146.
6. Coordination: I did not read the active-inference papers. `outbox/active-inference-search-papers-2026-11-08.md` covers category 8.

## How this was read, and the labels

- Every paper below was fetched on 2026-11-08 from arXiv (HTML where available, otherwise PDF text via pypdf) and read by me, with no summarising tools. Gumbel MuZero has no arXiv version and OpenReview returned 403, so I read it in Danihelka's UCL thesis (2023). Chapter 5, "Policy Improvement by Planning with Gumbel", is the published paper (per the thesis's research-paper declarations), and Appendix C holds its proofs.
- "Read in full" means the whole main text, plus the appendices that hold the theorem statements or limits I cite. The "Depth" column in the Sources table says exactly what was and was not read for each paper. For DeepSeek-R1 I read the main paper and the methodological supplementary sections (B.4.4, B.5, F, G), not the long evaluation and prompt tables.
- Labels. PROVEN: a theorem in the paper (I note when its assumptions are strong or the proof is informal; I did not re-check proofs). MEASURED: the paper's own experiments, with scale, baselines and seeds as reported. My assessment is labelled "Assessment". Its strength label is **strong** (clear advance, adequate scale and controls), **moderate** (real but narrow, weakly controlled or small), or **weak** (suggestive only).
- Quotes are verbatim from the arXiv text. The section number comes first; math is described rather than quoted where the text extraction mangled it.

## Table: papers × plan components

Cell values: **does** = the paper implements this component. **partly** = some of it, or under stronger assumptions. **clarifies** = a result that changes how the component should be posed. **analogy** = only a structural resemblance. **neg** = evidence against the component as usually built. Blank = not relevant.

| Paper | q / W_L (any-order, world model) | T independent of search policy | π: choose computations (VoI) | π̂ improved by search | B: trust over steps | Inconsistency as training signal | Inconsistency vs RL evidence | Strength |
|---|---|---|---|---|---|---|---|---|
| Hay, Russell et al. 2012, Selecting Computations | | partly | does (theory) | | | | | moderate |
| Guez et al. 2018, MCTSnets | | partly | partly | partly | analogy | | | moderate |
| Chung et al. 2023, Thinker | partly | | does (rollouts) | partly | | | | strong |
| Grill et al. 2020, MCTS as reg. policy opt. | | clarifies | | does | | | | strong |
| Danihelka et al. 2022, Gumbel MuZero | | partly | | does | | | | strong |
| Oren et al. 2026, Towards Optimal Policy Improvement | | | | does | partly | | | moderate |
| Lightman et al. 2023, Let's Verify | | | | | partly | | | strong (2023) |
| Setlur et al. 2024, PAV | | clarifies | | | partly + clarifies | | | strong |
| Yuan et al. 2024, Implicit PRM | | | | | partly | analogy | | moderate |
| Burns et al. 2022, CCS | | | | | | partly | | moderate |
| Wen et al. 2025, ICM | | partly | | analogy | | partly | does | strong |
| Qiu et al. 2026, Coherence optimisation | clarifies | partly | analogy | | | partly (theory) | partly | moderate |
| Ding 2026, Collective self-reinforcement | | | | | | neg (warning) | does | moderate |
| Liu et al. 2023, Marginalization models | does | | | | | does (q only) | | moderate |
| Kim et al. 2025, Train for worst / plan for best | does | clarifies | partly (heuristic) | | | | | strong |
| Hussein Mohamud et al. 2026, Adaptive Order Policies | does | clarifies | partly (learned) | | partly (fused with π) | | | moderate |
| Xu et al. 2026, Scheduling Thoughts (SAS) | does | clarifies | partly (RL) | | | | | moderate |
| Karan & Du 2025, Reasoning with Sampling | | does | | | | | does | strong (with caveats) |
| DeepSeek-AI 2025, R1 | | | | neg (MCTS) | neg (PRM) | | does | strong |
| Yue et al. 2025, Does RL incentivize...? | | | | | | | does | strong |
| Ruoss et al. 2024, Chess without search | analogy | | | partly (amortised) | | | | strong |
| Hafner et al. 2023, DreamerV3 | does (latent belief) | | | partly (amortised) | | | | strong |
| Hafner et al. 2025, Dreamer 4 | does | | | partly (amortised) | | | | strong |
| Richens et al. 2025, General agents need world models | clarifies | | | | | | | moderate (theorem; toy test) |

## Category 1. Computation as an action, learned metareasoning

**Hay, Russell, Tolpin, Shimony, "Selecting Computations: Theory and Applications", arXiv 1207.5879 (UAI 2012).** Foundational; the other agent listed it from the abstract only, so I read it.
- Claim (Abstract): "we develop a theoretical basis for metalevel decisions in the statistical framework of Bayesian selection problems, arguing (as others have done) that this is more appropriate than the bandit framework." Sec 1: "UCT applies bandit algorithms to problems that are not bandit problems."
- PROVEN: Def 3 makes metareasoning an MDP over sequences of computation outcomes, with stopping reward max_i μ_i(s), μ_i(s) = E[U_i | e_1..e_n], and cost c per computation. Thm 4: V = E[−cN + max_i μ_i(S_N)]. Thm 5: the expected number of computations of the optimal policy is at most (value of perfect information)/c. Example 3: an optimal policy can compute arbitrarily long. Sec 3, Example 4: no Gittins-style index policy exists for selection problems. Thm 16: the "blinkered" policy decomposes into k one-action problems. Thms 17-18: distribution-free Hoeffding bounds on VOI. All of this assumes a coherent joint distribution over utilities and computation results (Def 1) and "independent actions" (Def 13) for the blinkered policy.
- MEASURED: 25-arm Bernoulli selection (1000 and 10000 trials): the blinkered and VOI policies beat UCB1 in regret. 9x9 Go with the Pachi UCT engine, 1000 games per setting: a VOI-aware root wins up to 64% against UCT at 10k samples per ply. Sec 6: "At present we have no better proposal for internal nodes than to use UCT there."
- Assessment: **moderate**. The theory is rigorous, the experiments small.
- Plan: this is the formal version of "actions are computations, chosen by VoI". The target (decision quality under the belief model) does not depend on the meta-policy; the meta-policy only changes cost. The catch for Kelvin is Def 1: VOI needs a coherent joint over computation outcomes. An incoherent any-order q gives no such joint. That is where a learned trust B would have to come in, and nothing here provides one.

**Guez et al., "Learning to Search with MCTSnets", arXiv 1802.04697 (ICML 2018).** Anchor.
- Claim (Abstract): "In this paper we instead learn where, what and how to search... When applied to small searches in the well-known planning problem Sokoban, the learned search algorithm significantly outperformed MCTS baselines."
- Method: learned embedding, simulation policy, gated-residual backup ("allowing it to selectively ignore information originating from a node's subtree", Sec 3.4) and readout. Training is supervised on labels from an MCTS with about 2000 simulations (App A.1). Sec 3.6 credits each simulation with the decrease in loss of the anytime output it caused; "Letting γ=0 leads to a greedy behavior... linked to the single-step assumption proposed by Russell & Wefald".
- MEASURED: Sokoban, M=25 simulations: 84 ± 1% solved. MCTS with a value net reaches about 30% at 25 simulations and "requires 20 more times simulations" to match. PUCT with policy and value nets reaches about 65% at 25 simulations.
- Assessment: **moderate**. A clean idea, but one domain, a true simulator, and an expensive teacher.
- Plan: the per-computation reward ("decrease in loss") is a learned value of computation for π: partly. The target (the expert label) is fixed and independent of the search policy, which is the supervised version of policy independence. The gated backup is the nearest thing to a trust B, but it is a vector update trained only on final accuracy: analogy.

**Chung, Anokhin, Krueger, "Thinker: Learning to Plan and Act", arXiv 2307.14993 (NeurIPS 2023).** Anchor.
- Claim (Abstract): "The Thinker algorithm wraps the environment with a world model and introduces new actions designed for interacting with the world model... Thinker is the first work showing that an RL agent can learn to plan with a learned world model in complex environments."
- Method (Sec 4): K−1 imaginary steps (an action plus a reset flag) then one real step, with a compact tree summary as input. Sec 4.3: "no reward is given for imaginary steps"; the agent is "performing one step of policy improvement in generalized policy iteration, but the policy improvement operator is learned instead of handcrafted."
- MEASURED: Sokoban, 5 seeds: 95% solved at 5e7 frames (DRC original 93%; other baselines at most 45%). Atari-57 at 200M frames: median 261% vs 102% on the raw MDP (Rainbow 223%). App H, true model and a fixed weak base policy: the agent at K=10 matches 3-step exhaustive search (155 simulations) and MCTS with 60 simulations. At K=40 it beats 5-step exhaustive search (3905 simulations). App E: without the base policy and value, the agent fails to learn to plan.
- Assessment: **strong** for "a learned policy can choose model queries". It has no VoI; the imaginary policy learns only from real return.
- Plan: does "computations as actions" for environment rollouts. It does not choose which belief to compute inside a predictive model. Target and search policy are entangled by design (both are trained on return). No trust, no inconsistency.

## Category 2. Trust or value over reasoning steps

**Lightman et al., "Let's Verify Step by Step", arXiv 2305.20050.** Anchor.
- Claim (Abstract): "process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set."
- MEASURED: GPT-4-based, best-of-1860 on 500 MATH problems: PRM 78.2%, ORM 72.4%, majority voting 69.6% (Fig 3), with the gap widening as N grows. At small scale, with the large PRM as a synthetic labeller and 3 seeds, process supervision beats both kinds of outcome supervision at every data size. Active learning is about 2.6x more data-efficient. OOD (AP and AMC, 224 questions): PRM 72.9%, ORM 63.8%. Sec 3 admits that the large-scale ORM and PRM training sets are "not directly comparable". Sec 6.3: contamination cannot be ruled out.
- Assessment: **strong** for 2023, but it uses human step labels and is measured only by best-of-N.
- Plan: B trained on human step-correctness labels. It has no inconsistency signal and no joint training.

**Setlur et al., "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning" (PAV), arXiv 2410.08146 (ICLR 2025).**
- Claim (Abstract): "the process reward for a step should measure progress: a change in the likelihood of producing a correct response in the future, before and after taking the step... Crucially, this progress should be measured under a prover policy distinct from the base policy... weak prover policies can substantially improve a stronger base policy."
- Key facts. Sec 3.1: "Q^π fails to decouple the 'evaluation' of an action (step), from the 'promise' shown by the previous state." Sec 3.2: setting the prover equal to the base policy "results in exactly the same policy gradient update as only optimizing outcome evaluation". A very strong prover gives advantages near 0 on every step. App G: RL against Q^π + αQ^μ converges to a "REPHRASE THE PROBLEM" first step.
- PROVEN (idealised): Thm 3.1, stated informally, in a tabular softmax setting with natural policy gradient and oracle Q and A. One-step improvement ≳ γ·E[Var_{a∼π} A^μ] (distinguishability) + γ·E[A^μ A^π] (alignment). Prop F.1: a complementary prover with |V^μ − V^π| = η gives improvement ≳ η².
- MEASURED: Gemma2 2B, 9B and 27B SFT on MATH. Beam search against PAV (prover Bo4) vs best-of-N with an ORM: more than 8% more accurate and 1.5-5x more compute-efficient; 8x more efficient than beam search on Q^π (9B). Bo2 and Bo32 provers do worse than Bo4. For the 27B base, the 9B prover is best. RL: more than 7% above ORM-RL and about 6x more sample-efficient. Sec 7, Limitations: "it is unclear how to automatically design a flexible class of optimal (or very good) prover policies".
- Assessment: **strong** empirically on one domain. The theory is idealised. The clarification is the main contribution.
- Plan: this is the key clarification for B and for policy independence. Any rollout-based trust in a step is defined relative to the policy that completes the rollout. Trust in an inference step trained on outcome accuracy is therefore not policy-independent unless it is defined without a completing policy, for example by logical validity or by consistency.

**Yuan et al., "Free Process Rewards without Process Labels" (implicit PRM), arXiv 2412.01981 (ICML 2025).**
- Claim (Abstract): "an implicit PRM can be obtained at no additional cost, by simply training an ORM on the cheaper response-level labels... our implicit PRM outperforms a strong MCTS-based baseline a la Math-Shepherd using less than 1/38 of the training data."
- PROVEN: Prop 3.1. If r_θ = β log π_θ/π_ref, the prefix partial sum equals β log E_{π_ref}[exp(r_θ/β) | prefix]. My check: this is a telescoping identity of the parameterisation, so prefix scores are soft values of r_θ under π_ref by construction. How accurate they are depends entirely on how well r_θ fits outcomes.
- MEASURED (Table 1): best-of-N on MATH-500 with three generators. Average: DPO-implicit 50.4, Math-Shepherd reimplementation 47.8, best off-the-shelf PRM 49.1, with no seeds or CIs. Math-Shepherd costs 38.8x the FLOPs. Sec 5.4: PRM quality is anti-correlated with quality as a policy.
- Assessment: **moderate**. Small accuracy gains, a large cost gain.
- Plan: B from outcome labels only. The interesting structural point is that step scores agree with the outcome score by construction (a built-in Bellman consistency): analogy for a trust trained on consistency.

## Category 3. Training beliefs on consistency without labels

**Burns, Ye, Klein, Steinhardt, "Discovering Latent Knowledge in Language Models Without Supervision" (CCS), arXiv 2212.03827 (ICLR 2023).** Anchor.
- Claim (Abstract): "finding a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values... it outperforms zero-shot accuracy by 4% on average."
- MEASURED (Table 1): CCS 71.2% vs calibrated zero-shot 67.2%, with a logistic-regression ceiling of 83.7% (6 models, 10 binary datasets). Table 2: the top principal component of the contrast differences gets 70.1% without any consistency loss. Sec 2.2: the sign is not identified: "we take the maximum accuracy over the two possible ways of labeling".
- Assessment: **moderate**. It is a proof of concept. The near-parity with plain PCA means salience does most of the work, not the consistency loss. (Farquhar et al. 2312.10029 criticise it; I did not read that paper.)
- Plan: a probe, not a generative any-order q; only negation consistency; no search. Analogy-level support, plus a warning that consistency + confidence has many non-truth solutions.

**Wen et al., "Unsupervised Elicitation of Language Models" (ICM), arXiv 2506.10139 v2 (Jan 2026; ICLR 2026 per listing).** Anchor.
- Claim (Abstract): "On GSM8k-verification, TruthfulQA, and Alpaca reward modeling tasks, our method matches the performance of training on golden labels and outperforms training on crowdsourced human supervision... The resulting assistant matches its counterpart trained on production-grade human labels on average, with higher scores on chat and safety yet lower scores on math and coding."
- Method (Sec 2.2-2.3): the target is U(D) = α·(mutual predictability, Σ_i log P(y_i | x_i, D∖i)) − (count of pairwise logical inconsistencies). It is maximised by simulated annealing over label sets, with a consistency-fix step, and the model is then fine-tuned on the labels.
- MEASURED: Llama 3 8B and 70B, 3 runs: matches gold labels on TruthfulQA and GSM8K, beats human labels on Alpaca, and matches gold at about 80% on author gender where humans get 60%. On a deliberately non-salient task ("sun" poems) it is at chance. A Claude 4 Sonnet reward model from 5000 ICM-labelled pairs, then 250k RL episodes graded by an Opus reward model, matches the human-label reward model on average. Ablation: removing the logical-consistency term barely matters on TruthfulQA but causes collapse on Alpaca. Sec 6: "we understand mutual predictability as the most important term."
- Assessment: **strong** in practice. Classification only, partly model-graded, and it depends on salience.
- Plan: the closest existing case of "a searcher optimising a model-defined target T". U depends only on the model and the labels, not on the annealing order, so T is policy-independent by construction, and the search result is distilled back. Its inconsistency term is hand-written pairwise logic. It has no attention policy, no trust B, and is not a POMDP.

**Qiu, Ismail, He, Feng, "Self-Improvement as Coherence Optimization: A Theoretical Account", arXiv 2601.13566 v2 (TMLR 06/2026).**
- Claim (Abstract), about debate, bootstrap and ICM: "We show that they can all be understood as coherence optimization, the search for a context-to-behavior mapping that is most compressible and jointly predictable... among all such regularization schemes, coherence regularization with a prior derived from a pretrained model optimizes a lower bound of worst-case accuracy for semi-supervised learning (Theorem 5.5)."
- Definitions that matter for Kelvin. Def 3.2 requires a chain rule: conditioning on a1 then a2 gives the same joint as a2 then a1. Def 3.7: coherence χ(π) is the sum of sequential log-probabilities, and "The chain rule ensures that χ is well-defined." Remark 3.6: "A formal bound on the approximation error of the chain rule for in-context learning and finetuning systems remains an open question."
- PROVEN: Thm 4.2 is standard Gibbs convergence ("We claim no novelty for the sampler"). Prop 4.4: debate is synchronous Gibbs ("follows by tautology"). Thm 5.2: a uniform-convergence bound with −2χ as description length. Thm 5.5: a KL-optimal prior maximises a worst-case lower bound. The authors themselves warn, Sec 5.2: "Maximizing a lower bound is weaker than being the best regularizer in practice." Conjecture 5.9 is heuristic only.
- MEASURED (preliminary): Llama-3.2-1B in-context Gibbs on GSM8K, 100 seeds: in-sample accuracy 24.9% → peak 45.9% (final 35.6%); held-out about 30% → 40% (App E). It needed an anchoring mixture with the initial context to stop mode collapse. A 0.6B judge's coherence (PMI) separates honest from prompted-deceptive GPT-5 solutions where plain log-probability is fooled.
- Assessment: **moderate**. It gives clarifying definitions and standard bounds, with small experiments. Sec 7 says "the optimality claim is not yet empirically established".
- Plan: their chain rule is exactly the any-order consistency condition for q. Coherence of a set of beliefs is defined only if sequential conditioning is order-independent; Kelvin's "inconsistency" is the violation, and they leave it unquantified (Remark 3.6). Gibbs over contexts is an attention policy with a policy-independent target (softmax over coherence). It also warns about over-optimisation.

**Ding, "Collective Dynamics of Self-Reinforcement: How Cross-Task Coupling Drives Policy Collapse in Unsupervised RLVR", arXiv 2610.37553 (27 Oct 2026).**
- Claim (Abstract): "Label-free RLVR rewards a language model for agreeing with its own majority vote. It improves reasoning at first but eventually collapses... We explain this dataset-size dependence as a collective instability... In label-free RL of a small Transformer, removing the shared direction prevents collapse entirely, restoring it causes collapse".
- PROVEN (toy model): K=3 majority-vote drift F(z) = p(1−p)(2p−1), unstable for |z| < ln(2+√3) ≈ 1.317. Prop 1: a shared shift grows if and only if ⟨F′⟩ > 0. Eq 7: N_c ≈ ln(1/δ)/(λ² ln[F*/F(a)]). Sec 4.5: a kernel version with a protective "prediction-aligned" coupling.
- MEASURED: simulations confirm the predicted transition. In a small Transformer (6 seeds × 7 sizes per condition), an antisymmetric readout with no shared direction never collapses. The kernel-based predictor agrees 71% of the time (AUC 0.87) but underestimates collapse (a model with initial G = 0 still collapsed in 17 of 42 runs). An entropy bonus stops uniform collapse, but accuracy still falls to chance. Sec 7: "we did not test pretrained language models".
- Assessment: **moderate**. Mechanistic and honest about its negative calibration results, but toy-scale, single-author and unreviewed.
- Plan: the clearest warning for "train on inconsistency instead of RL". A self-agreement objective over many problems with shared parameters is unstable towards a degenerate, consistent answer. It is analogy-level for cross-order coherence, but the mechanism (shared directions collect the signal from every problem) carries over to any self-consistency loss.

## Category 4. Search distilled into policy; policy improvement with little search

**Grill et al., "Monte-Carlo Tree Search as Regularized Policy Optimization", arXiv 2007.12509 (ICML 2020).** Foundational and not in the anchors.
- Claim (Abstract): "AlphaZero's search heuristics, along with other common ones such as UCT, are an approximation to the solution of a specific regularized policy optimization problem."
- PROVEN: Def 1 / Eq 7: π̄ = argmax_y qᵀy − λ_N KL[π_θ, y], with λ_N ∝ √N/(|A|+N). Props 1-2: PUCT selects the action that ascends this objective in the visit counts, and π̂(a*) ≤ π̄(a*). Eq 12 bounds ‖π̄ − π̂‖∞ ≤ (|A|−1)/(|A|+N), "under the assumption that π̄ is a constant" (Sec D.3). This is a tracking result, not convergence.
- The key remark (Sec 4.2): "AlphaZero entangles search and learning, which is not desirable... the Learn variant partially disentangles these components... the new learning target is π̄(·|x_root) which is computed from search Q-values, rendering it less sensitive to e.g., the action selection formula."
- MEASURED: scaled-down MuZero. Ms Pacman, 8 seeds: much better at 5 simulations, similar at 24 or more. 6 Atari games, 6 seeds: better at 50 simulations. DM Control, 3 seeds. Hyperparameters were tuned for the baseline.
- Assessment: **strong** as a clarification (Gumbel MuZero builds on it). The experiments are modest.
- Plan: π̂ as visit counts tracks a prior-regularised optimum of the search Q-values, so it depends on the prior π_θ and, through q, on the search policy. Not policy-independent. The other agent reached the same point from the abstract; the full text confirms it.

**Danihelka, Guez, Schrittwieser, Silver, "Policy Improvement by Planning with Gumbel" (ICLR 2022).** Anchor. Read in the thesis version, Ch. 5 and App C.
- Claim (Ch. 5 abstract): "AlphaZero can fail to improve its policy network, if not visiting all actions at the root of a search tree... Gumbel AlphaZero and Gumbel MuZero... match the state of the art on Go, chess, and Atari, and significantly improve prior performance when planning with few simulations."
- Key statements. Sec 5.1 on the root: "the performance of the Monte-Carlo tree search therefore only depends upon the final recommended action at the root node, and not upon the intermediate actions selected during search (Bubeck et al., 2011)", which is the reason for Sequential Halving and simple regret. Sec 5.3.2 counterexample: with q = (0,0,1), π = (.5,.3,.2) and n = 2, picking the best of the top-n actions gets 0, less than the policy's 0.2.
- PROVEN: App C.1 (Gumbel-top-k plus argmax of g + logits + σ(q) improves on π for every Gumbel draw). App C.2: π′ = softmax(logits + σ(completedQ)) improves on π. The proof is rigorous for one visited action; the multi-action case is argued informally.
- MEASURED: 9x9 Go, 2 seeds, evaluation with 800 simulations: "MuZero fails to learn from 16 or fewer simulations. Strikingly, Gumbel MuZero learns reliably even with 2 simulations." 19x19 Go and chess at 400 simulations: not worse. Ms Pacman, 10 seeds: learns at 2 simulations where MuZero fails at 4 or fewer.
- Assessment: **strong**. The guarantee holds when the visited actions are evaluated correctly; Go uses few seeds.
- Plan: the best existing answer to "π̂ improved by search with little search". It also gives the bandit version of policy independence: the root recommendation is scored by simple regret, so search choices change cost and not the target. Partly matches Kelvin's attention-over-inferences setting.

**Oren et al., "Towards Optimal Policy Improvement", arXiv 2610.01566 (1 Oct 2026).**
- Claim (Abstract): "optimal improvement restricted to a set of states is equivalent to solving an induced MDP... We formulate greedification under this constraint as probabilistic decision-making under uncertainty and derive a novel operator that is optimal with respect to the resulting objective."
- PROVEN: Thms 1-2 (informal in the main text): induced semi-MDP and sub-MDP. Eqs 18-19: I_opt ∝ π · exp(Â/η(s,a)), where η(s,a) = (τ² + σ²_A(s,a))β/τ², so each action's advantage is shrunk by its posterior precision τ²/(τ²+σ²). Thm 3: it is greedification. Thm 4: almost-sure convergence in GPI as noise vanishes. Remark: "Optimality under approximate evaluation is necessarily relative to a decision formulation".
- MEASURED: GumbelAlphaZero 9x9 Go, 5 seeds: +108 Elo, and +277 Elo with a Q critic (the baseline gets worse with the critic). SAC on 10 DMC tasks and ReBRAC on 4 D4RL tasks: aggregate gains. Noised policy iteration on 20 grid MDPs: best of all operators tested.
- Assessment: **moderate**. Principled; "optimal" only for its certainty-equivalent objective; very recent.
- Plan: a precise form for B inside π̂. A step's evaluation enters the improved policy weighted by its precision. Here the precision comes from visit counts or ensembles, not from inconsistency: partly.

**Ruoss et al., "Amortized Planning with Large-Scale Transformers: A Case Study on Chess", arXiv 2402.04494 (NeurIPS 2024).** Anchor.
- Claim (Sec 1): the largest models "enable play at grandmaster level (Lichess blitz Elo of 2895) against humans... without using explicit search at test time", and "the performance gap cannot be fully closed".
- MEASURED (Table 1): 270M action-value transformer: tournament Elo 2299, puzzles 95.4%. AlphaZero with 400 MCTS simulations: 2470 and 95.6%. Leela value net without search: 2418. The Stockfish oracle: 2711. Trained on 15.3B Stockfish action-values. Sec 4, "Indecisiveness in the face of overwhelming victory": several moves fall into the top value bin and the agent switches between plans.
- Assessment: **strong** and carefully caveated.
- Plan: amortising a search target into q̂ works to within a few hundred Elo with enough data. The amortised policy lacks commitment across steps, a plan inconsistency over time that a consistency term would target. Analogy for W_L amortising the searcher.

**DeepSeek-R1's failed MCTS** is in Category 7.

## Category 5. Learned world models as beliefs plus planning

**Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), arXiv 2301.04104 (Nature 2025).** Anchor.
- Claim (Abstract): "a general algorithm that outperforms specialized methods across over 150 diverse tasks, with a single configuration... the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula."
- Method: an RSSM belief state (h_t, z_t). Actor and critic are trained purely in imagination; "we select actions by sampling from the actor network without lookahead planning".
- MEASURED: fixed hyperparameters across Atari-57, ProcGen, DMLab, Atari100k, proprioceptive and visual control, BSuite, beating tuned experts. Minecraft diamonds on 1 GPU in 9 days. Ablations: performance "rests predominantly on the unsupervised reconstruction loss"; larger models are more data-efficient.
- Assessment: **strong**.
- Plan: the standard learned latent POMDP belief as W_L, with search amortised into the actor. No attention over computations, no trust, no inconsistency; the target is reward.

**Hafner, Yan, Lillicrap, "Training Agents Inside of Scalable World Models" (Dreamer 4), arXiv 2509.24527.**
- Claim (Abstract): "the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction."
- MEASURED: 2B parameters, 2.5K hours of VPT contractor data, 1000 episodes: diamonds 0.7%, iron pickaxe 29%; VPT reaches sticks at 53%, a Gemma-3 VLA reaches iron pickaxe at 11%. 14 of 16 human-interaction tasks vs 5 of 16 for Oasis. 100 hours of action labels give about 85% of full-label PSNR. Sec 6: "far from a full clone of the game, especially due to short memory" (9.6 s context).
- Assessment: **strong** engineering and empirical advance; the diamond rate is low.
- Plan: W_L at scale, good enough that RL inside it transfers. Memory, the belief over latents, is the stated limit.

**Richens, Abel, Bellot, Everitt, "General agents need world models", arXiv 2506.01622 (ICML 2025).**
- Claim (Abstract): "any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment... this model can be extracted from the agent's policy".
- PROVEN: Thm 1: in a finite, fully observed, communicating cMP, a policy with regret at most δ on depth-n goals determines P̂ with error O(δ/√n) + O(1/n). Thm 2: myopic agents need no world model. Sec 4, Limitations: "considers only fully observed environments".
- MEASURED: a random 20-state cMP; recovered-model error falls as n^(−1/2).
- Assessment: **moderate**. A clean theorem that changes how model-free vs model-based should be posed, but only a toy test, and the POMDP case is open.
- Plan: supports grounding in an explicit W_L. What must be learned about latents under partial observability, which is Kelvin's setting, is open.

TD-MPC2 is an anchor I did not read (see Sources).

## Category 6. Any-order predictors used for reasoning or planning

**Kim, Shah, Kontonis, Kakade, Chen, "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions", arXiv 2502.06768 (ICML 2025).**
- Claim (Abstract): "MDMs indeed train on computationally intractable subproblems... a suitable strategy for adaptively choosing the token decoding order significantly enhances the capabilities of MDMs... boost solving accuracy in pretrained MDMs from <7% to ≈90%, even outperforming ARMs with 7× as many parameters... trained via teacher forcing to learn the right order of decoding."
- PROVEN: Prop 2.1: the MDM loss equals the any-order AR loss averaged over permutations. Prop 3.3 (hardness of typical masking problems) holds "under the 1RSB cavity prediction (see Conjecture B.13)", so it is conditional on a statistical-physics conjecture.
- The key statement (Sec 4): "when we have an ideal MDM that perfectly solves all masking problems... then using any sampling path (unmasking the tokens in any order) results in the same distribution". With an imperfect MDM the order matters.
- MEASURED: Sudoku, 6M MDM: vanilla 6.88%, top-probability 18.51%, margin 89.49%, vs ARM with ordering (42M) 87.18%. Hard Sudoku: margin 49.88% vs ARM 32.57%. Zebra: about 98% vs 91%. LLaDA-8B: margin raises HumanEval-infill and Math. The tables give no seeds.
- Assessment: **strong**.
- Plan: this is "q any-order + π picks which belief to compute next" in its simplest form, with π a hand-written certainty rule: partly. More important is the clarification: order (policy) independence is a consistency property of q. Kelvin's inconsistency training is one route to restoring it.

**Hussein Mohamud, Hasan, Ravanelli, Bengio, "Adaptive Order Policies for Masked Diffusion", arXiv 2606.00295 v3 (Oct 2026).**
- Claim (Abstract): "results in a policy that prefers positions where the denoiser is more likely to be correct... (i) training solely the policy while using a frozen pre-trained denoiser, and (ii) training the policy and denoiser jointly". Sec 2.2: "a perfect denoiser is able to sample from the target regardless of the unmasking order".
- Method: Eq 3 weights each masked position's cross-entropy by q_φ(i | x_t). The optimum puts all mass on the lowest-cross-entropy position, which needs ground truth during training. Eq 4 is the joint denoiser loss weighted by (1 + q_φ).
- MEASURED (3 seeds): Sudoku: margin 91.53, policy 93.75, oracle 100. 3-SAT: 88.33 → 89.00 (90.57 jointly). DPLM pLDDT +2-3. Matched-budget continued denoiser training gives no gain. LLaDA-8B: mixed, mostly within noise.
- Assessment: **moderate**.
- Plan: the closest existing case of "π trained jointly with q". Here π learns where q is reliable (a learned trust used as attention), from supervised cross-entropy, not inconsistency, and it picks the most reliable position rather than the most informative (no VoI).

**Xu, Liu, Agrawal, Chen, Huang, "Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models" (SAS), arXiv 2606.23567 (Jun 2026).**
- Claim (Abstract): "We derive a tractable upper bound on the sequential decoding mismatch... This bound induces a dense self-aware reward... On Sudoku with 1B MDM, SAS improves puzzle accuracy from 82.0% (best heuristic schedule) to 91.8%".
- PROVEN: Thm 3.1 is Jensen. Thm 3.2: KL(data ‖ decoder) ≤ L_SAS − H = a joint KL over (x, σ), a latent-order ELBO. Remark 3.4: "In the regime of a perfect diffusion model p_θ, the upper bound... is tight... for any order policy φ. Thus, order learning is most useful when the denoiser is imperfect".
- MEASURED: Sudoku with SMDM-1B: confidence 82.0, human expert order 76.6, SAS 91.8; 97.5 after second-stage fine-tuning, the same as fine-tuning on the expert order (97.8). LLaDA-8B GSM8K: 64 → 76. Single numbers, no seeds. Limitations: the reward is "the frozen denoiser's pathwise likelihood on ground-truth trajectories", which leaves a gap between training and inference.
- Assessment: **moderate**.
- Plan: π learned by RL with a dense per-step reward equal to q's own predictive log-likelihood on data, an accuracy term. The remark repeats the policy-independence point. No inconsistency, no VoI, no trust.

**Liu, Ramadge, Adams, "Generative Marginalization Models", arXiv 2310.12920 (ICML 2024).** Foundational for "q trained on consistency".
- Claim (Abstract): "explicitly modeling all induced marginal distributions... We propose scalable methods for learning the marginals, grounded in the concept of 'marginalization self-consistency'."
- Method: Eq 7 is the sampled cross-order constraint p_θ(x_<d)·p_φ(x_d | x_<d) ≈ p_θ(x_≤d) for random orders σ, enforced by a squared log-error. Prop 1 (assuming universal approximators): MLE plus the constraint is equivalent to fitting AO-ARM conditionals and then distilling marginals by minimising the consistency error.
- MEASURED: NLL equal to AO-ARM on Binary MNIST, CIFAR-10, ImageNet32 and text8, with marginal inference about 10⁴x faster and Pearson 0.95-0.99 against AO-ARM marginals. Energy-based Ising at D=900, where fixed-order ARM runs out of memory.
- Assessment: **moderate**. A solid method; consistency is a distillation constraint, and accuracy comes from MLE.
- Plan: the most direct prior art for the q-side of "accuracy + inconsistency" over an any-order model. No trust, no attention, no search target.

MDLM and LLaDA are anchors I did not re-read; Kim et al., AOP and SAS all use them as base models.

## Category 7. Test-time compute and RL on chain of thought

**DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv 2501.12948 (Nature 2025).**
- Claim (Abstract): "the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories."
- MEASURED: R1-Zero (GRPO with rule-based accuracy and format rewards only) raises AIME 2024 pass@1 from 15.6% to 77.9% (cons@16 86.7%), with response length growing during training. Total cost about 147K H800 GPU hours. F.1: distillation beats pure RL at 32B (AIME 72.6 vs 47.0). G.1: 7B dense and 16B MoE bases "consistently failed to yield meaningful improvements". B.5: a learned helpfulness reward rises while CodeForces accuracy falls.
- G.2, unsuccessful attempts, verbatim. PRM: "once a model-based PRM is introduced, it inevitably leads to reward hacking". MCTS: "Training a fine-grained value model is inherently difficult... iteratively boosting model performance through self-search remains a significant challenge."
- Assessment: **strong** methodological landmark. The negative results are anecdotal ("this does not imply that these approaches are incapable").
- Plan: the RL side of "inconsistency vs RL". The direct warnings for B (learned step trust gets hacked) and for π̂ by self-search at token level.

**Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", arXiv 2504.13837 (NeurIPS 2025).**
- Claim (Sec 1): "base models consistently surpass RLVR models across all benchmarks and LLM families as k increases"; "Reasoning paths generated by current RLVR model already exist in its base model."
- MEASURED: several model families and RL frameworks, pass@k up to 1024. Table 2: on AIME24, 13.3% of problems are solvable by the base only and 0.0% by RL only. Manual checks find valid chains of thought on the hardest solved problems. PPO, GRPO, Reinforce++, RLOO, ReMax and DAPO all leave a sampling-efficiency gap of more than 40 points. Training pass@1 rises while pass@256 falls.
- Assessment: **strong** at the RL compute tested. It is not a proof of a limit in principle. ProRL (2505.24864) disputes it under prolonged RL; I did not read ProRL.
- Plan: current outcome RL mostly reweights what W_L can already produce.

**Karan & Du, "Reasoning with Sampling: Your Base Model is Smarter Than You Think", arXiv 2510.14901 (ICLR 2026).**
- Claim (Abstract): "our algorithm offers substantial boosts in reasoning that nearly match and even outperform those from RL on a wide variety of single-shot tasks, including MATH500, HumanEval, and GPQA... our method does not require training, curated datasets, or a verifier".
- PROVEN: Prop 1: "Low-temperature sampling does not sample from the power distribution p^α". Observation 1: p^α "upweights tokens with few but high likelihood future paths". The MH sampler targets block-wise p^α.
- MEASURED (Table 1, single numbers, no seeds): Qwen2.5-Math-7B MATH500: base 0.496, power sampling 0.748, GRPO 0.785. HumanEval 0.573 vs GRPO 0.537. Pass@k above both base and GRPO for k > 1. About 8.8x the inference tokens. Caveat: Phi-3.5's GRPO HumanEval (0.134) falls below its base (0.213), which suggests a weak RL baseline there.
- Assessment: **strong** as a clarification, with caveats: Qwen2.5-Math, no seeds, unmeasured mixing.
- Plan: the cleanest existing case of Kelvin's framing. T (= p^α) is defined by W_L alone, a searcher (MCMC) optimises it, and the search procedure changes cost, not the target. Together with Yue, it says a model-defined target plus search recovers most RLVR gains on these benchmarks. No attention, trust or inconsistency; a single AR model is coherent by construction.

## Category 8. Active-inference bridges

Not read here, by design. `outbox/active-inference-search-papers-2026-11-08.md` reads Sophisticated Inference, Fountas et al. and BTAI in full and lists 2022-2026 work from abstracts. It reaches the same conclusion from the AIF side: none of those papers chooses which belief to compute, has a target independent of the search policy, or learns a trust. One candidate I found that bears on VoI: Cooper & Velasquez, "EFE as Belief-Dependent Utility for ρ-POMDPs", arXiv 2607.16981, which claims EFE equals a ρ-POMDP with information-gain utility at weight 1. That agent lists it from the abstract; I did not read it.

## What nobody has done yet, relative to the plan (with evidence)

1. **Inconsistency across inference orders of an any-order model, used to train q, B and π together.** MaM trains q on marginal/conditional consistency but has no trust or attention. AOP and SAS train π on supervised accuracy or likelihood with ground truth. Qiu et al. define coherence only under an exact chain rule and say the chain-rule error is "an open question" (Remark 3.6). ICM uses hand-written pairwise logic, not cross-order coherence. A targeted search for an order-consistency loss in any-order models (2026-11-08) found analyses of why orders diverge (for example 2602.00286, not read) but no training method of this kind.
2. **VoI-driven selection of which inference to compute in an incoherent model.** Hay et al. give VoI for computations only under a coherent joint (Def 1). Kim, AOP and SAS choose positions by confidence, learned reliability or likelihood, never by expected change in the decision. MCTSnets uses loss decrease against an expert label. Thinker uses return.
3. **A target T that is provably independent of the search policy when q is incoherent.** Kim, AOP and SAS all say order-independence holds only for a perfect model. Grill shows the visit-count target depends on the prior. PAV shows rollout-based step values depend on the completing policy. Power sampling and ICM have policy-independent targets only because their models are treated as coherent (an AR LM) or the target is a fixed function of labels. No paper defines T for an incoherent q (for example as a fixed point of inconsistency resolution) and then shows π can be trained freely.
4. **A trust B learned from inconsistency rather than from outcomes or variance.** Existing B-like objects: human step labels (Lightman), prover-relative advantages (PAV), implicit soft values (Yuan), precision from visit counts or ensembles (Oren), learned reliability against ground truth (AOP), a gated backup (MCTSnets). None is trained on disagreement between inference paths.
5. **Protection against the known failure mode.** Ding 2026 shows self-consistency rewards collapse through shared parameters, and Qiu needed anchoring. No paper tests whether cross-order coherence, unlike majority vote, has a protective coupling. This is a gap and a risk for the plan at once.

## The three papers to read yourself first

1. **Qiu et al., 2601.13566 (coherence optimisation).** Its chain-rule condition (Def 3.2) is your any-order consistency condition, and Gibbs over contexts is an attention policy with a model-defined target. It states your open problem in its own words (Remark 3.6), and it is careful about what its optimality theorem does not show (Sec 5.2).
2. **Kim et al., 2502.06768 (train for the worst, plan for the best).** The cleanest evidence that choosing which conditional to evaluate next is worth a lot (Sudoku 7% → 90%), and the plainest statement that order-independence holds only for a consistent q. That gives the reason your inconsistency term should make the search policy matter less.
3. **Setlur et al., 2410.08146 (PAV).** It settles a question your B must answer. Any rollout-based value of a reasoning step depends on the policy that completes the rollout, and using the base policy itself adds nothing beyond outcome RL. Read it before deciding what B is trained to predict.

Next after these: Karan & Du (model-defined target + searcher ≈ RL), then Grill et al. with Gumbel MuZero (what π̂ from search actually is).

## Sources

All accessed 2026-11-08.

| Paper | Link | Depth |
|---|---|---|
| Hay, Russell, Tolpin, Shimony 2012 | https://arxiv.org/abs/1207.5879 | full PDF |
| Guez et al. 2018, MCTSnets | https://arxiv.org/abs/1802.04697 | full HTML |
| Chung, Anokhin, Krueger 2023, Thinker | https://arxiv.org/abs/2307.14993 | main text + App E, H (A-D, F, G headings only) |
| Grill et al. 2020 | https://arxiv.org/abs/2007.12509 | full HTML incl. App B-D statements |
| Danihelka et al. 2022, Gumbel MuZero | https://discovery.ucl.ac.uk/id/eprint/10167022/2/ivo_danihelka_thesis.pdf (Ch. 5 + App C; ICLR page https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/) | Ch. 5 and App C.1-C.3 |
| Oren et al. 2026 | https://arxiv.org/abs/2610.01566 | main text; proofs not read |
| Lightman et al. 2023 | https://arxiv.org/abs/2305.20050 | main text + App F, G, H |
| Setlur et al. 2024, PAV | https://arxiv.org/abs/2410.08146 | main text + App F statements, Prop F.1 proof, App G |
| Yuan et al. 2024, implicit PRM | https://arxiv.org/abs/2412.01981 | main text |
| Burns et al. 2022, CCS | https://arxiv.org/abs/2212.03827 | main text |
| Wen et al. 2025, ICM | https://arxiv.org/abs/2506.10139 | main text (v2) |
| Qiu et al. 2026 | https://arxiv.org/abs/2601.13566 | main text + limitations; proofs not read |
| Ding 2026 | https://arxiv.org/abs/2610.37553 | main text; derivations not checked |
| Liu, Ramadge, Adams 2023, MaM | https://arxiv.org/abs/2310.12920 | main text + App B.1 |
| Kim et al. 2025 | https://arxiv.org/abs/2502.06768 | main text; App B proofs not read |
| Hussein Mohamud et al. 2026, AOP | https://arxiv.org/abs/2606.00295 | main text |
| Xu et al. 2026, SAS | https://arxiv.org/abs/2606.23567 | main text |
| Karan & Du 2025 | https://arxiv.org/abs/2510.14901 | main text (v1) |
| DeepSeek-AI 2025, R1 | https://arxiv.org/abs/2501.12948 | main text + supp. B.4.4, B.5, F, G |
| Yue et al. 2025 | https://arxiv.org/abs/2504.13837 | main text |
| Ruoss et al. 2024 | https://arxiv.org/abs/2402.04494 | main text |
| Hafner et al. 2023, DreamerV3 | https://arxiv.org/abs/2301.04104 | main text pp. 1-11 |
| Hafner, Yan, Lillicrap 2025, Dreamer 4 | https://arxiv.org/abs/2509.24527 | main text Secs 1-6 |
| Richens et al. 2025 | https://arxiv.org/abs/2506.01622 | main text + App A.5 |

**Anchors not read, so not in the table:** Stochastic MuZero (OpenReview PDF blocked, 403); TD-MPC2 (2310.16828); MDLM (2406.07524); LLaDA (2502.09992).

**Candidates found but not read** (abstract only; nothing claimed about them above beyond what is stated):
- Verifier-Guided Self-Consistency, 2611.06606 (Nov 2026): verifier-tilted sampling with a trust exponent.
- Learning Unmasking Policies for Diffusion LMs, 2512.09106 (ICML 2026).
- Mask-Aware Policy Gradients, 2607.15200.
- ProRL, 2505.24864.
- Spurious Rewards, 2506.10947.
- Revealed Rationality, 2608.05015.
- Dutch Books for LMs, 2609.02797.
- Training LMs to Be Coherent Decision-Makers, 2610.09164.
- On the Role of Computation in RL, 2602.05999.
- Expanding LLM Reasoning, 2610.05584.
- Generation Order and Parallel Decoding (information-theoretic), 2602.00286.
- From Interface to Inference, 2607.26504.
- EFE as ρ-POMDP utility, 2607.16981.
- Farquhar et al., CCS critique, 2312.10029.

Discovery used web search, the arXiv API and earlier outbox reviews (`learned-search-papers-study-2026-10-14.md`, `lit-extending-transformer-search-2026-10-26.md`); every paper in the table was re-read for this report. Reading notes: `/tmp/mlread/notes*.md` (scratch; text extracts under `/tmp/mlread/txt/`).
