# Learned world models + a separate reasoner: survey and five designs

eurisko, 2026-10-02. For Kelvin's framing: (1) learn, slowly if need be, an explicit-ish world model (a theory: claims with confidences, or a PDG) that *explains the data*; (2) a separate reasoning process (repeated blocks = thinking time) that derives the model's consequences with logical-induction properties. Context: `BRIEF-MATH.md`, `IDEAS-LI-NET.md` (world-seat market, PDG projection fixed point).

Every URL in Sources was fetched on 2026-10-02 (title checked). Claims taken from the fetched abstracts are unmarked. Claims from memory are marked **(unverified)**. Where I give my own analysis, I say so.

**TL;DR.** Every family that worked had the same three parts: a *hypothesis language*, a *score* (likelihood or entailment of the examples plus a size penalty), and a *search* (constraint-driven, MCMC, gradient, or an LLM proposing candidates). They broke in three places: (a) scoring a theory needs inference, which is #P-hard, so learning stalls on large groundings; (b) differentiable relaxations (fuzzy truth, soft unification) scale the search but give up probabilistic coherence or identifiability; (c) program search without a strong proposer does not scale past small programs. LLM-proposed programs (2023-2025) fixed (c) for small domains. Kelvin's split helps with (a): learning can call slow, exact or sampled inference on small cases and train the fast block reasoner on what it finds. Richardson's 2026 *Local Inconsistency Resolution* is the closest thing to "PDG learning" and treats learning and inference as one algorithm.

## 1. What exists

### 1.1 Inductive logic programming (ILP)
- **Model:** a logic program (Horn or ASP clauses) over given background predicates.
- **Explains the data:** classic ILP requires the program to entail every positive example and no negative one (Cropper & Dumančić's ILP-at-30 survey). Popper's *learning from failures* loops generate, test, constrain: each failed hypothesis becomes a constraint that prunes the space. Its MDL variant (Hocquette et al. 2023) scores by description length to cope with noise and "scale[s] to moderate amounts of noise". Metagol searches using metarules (second-order templates). ILASP learns answer-set programs, including weak constraints, i.e. preferences.
- **Reasoning:** a Prolog or ASP solver, which is exact.
- **Where it broke:** the hypothesis space grows combinatorially with clause size and the number of predicates. Metarules and language biases are hand-chosen and decide what can be learned at all. Classic ILP did not handle noise or raw perception, which is ∂ILP's stated motivation. It gives truth, not probabilities, so it has no confidences. **(unverified):** Popper and ILASP top out at roughly a few clauses of a few literals each; ILASP's cost grows fast with the size of the hypothesis space.

### 1.2 Neural / differentiable ILP
- **∂ILP** (Evans & Grefenstette 2017): rule templates; each candidate clause gets a soft weight; forward chaining is run differentiably for T steps; trained by backprop on likelihood; robust to noise and mislabelling. **(unverified):** memory grows with the number of candidate clauses times the number of ground atoms, so tasks stay tiny, and the templates are hand-given.
- **NeuralLP** (Yang, Yang & Cohen 2017): learns chain-like rules for knowledge-base completion as compositions of TensorLog operators chosen by an attention controller. It scales to real knowledge bases but only for path-shaped rules.
- **Neural Logic Machines** (Dong et al. 2019): tensor layers over predicates of each arity. After training on small instances they recover lifted rules that generalise to larger ones, but the rules are implicit in the weights, not readable clauses.
- **Where it broke:** fixed templates, a fixed chaining depth T, and local minima (**unverified**, widely reported for ∂ILP). The learned theory is only explicit when the weights become crisp.

### 1.3 Probabilistic logic programming and learning
- **ProbLog** (De Raedt, Kimmig & Toivonen 2007): Prolog clauses annotated with probabilities; distribution semantics over possible worlds. Inference is weighted model counting, done by knowledge compilation **(unverified detail)**. Parameters are learned by EM from interpretations (LFI-ProbLog, **unverified**). This is the clean "theory with confidences" object, and its distribution is coherent by construction.
- **DeepProbLog** (Manhaeve et al. 2018): ProbLog with neural predicates (e.g. digit classifiers); trained end to end through exact inference. **Broke on:** exact inference cost grows exponentially (**unverified**, known from follow-ups) and *reasoning shortcuts*: Marconato et al. 2023 show neuro-symbolic predictors "can attain high accuracy but by leveraging concepts with unintended semantics" and that shortcuts "are difficult to deal with". This is the identifiability failure of learning concepts and theory together.
- **Markov logic networks** (Richardson & Domingos 2006): weighted first-order clauses; P(world) ∝ exp(Σ w_i · n_i(world)). Weights are learned by maximum (pseudo-)likelihood; structure is learned by clause search. Inference uses MCMC (MC-SAT) or MaxWalkSAT (**unverified** details). **Broke on:** grounding blow-up; likelihood learning needs inference inside the loop; structure learning is very slow; near-deterministic clauses make MCMC mix badly (**unverified**).
- **PSL / hinge-loss MRFs** (Bach et al. 2015): soft truth values in [0,1]; rules become hinge potentials; MAP inference is convex and fast; weights are learned by approximate maximum likelihood or max-margin. **Broke on:** truth values are Łukasiewicz-relaxed, not probabilities, so the output is not a distribution over worlds (my analysis: no LI-style coherence).
- **Semantic loss** (Xu et al. 2018): a loss equal to −log P(constraint is satisfied) under the network's outputs, computed by compilation. It is the closest existing loss to "the price table must respect the axioms".

### 1.4 PDGs (Richardson)
- **PDG** (Richardson & Halpern 2020): a set of conditional-probability claims, each with a confidence; may be inconsistent; the inconsistency is a scoring function. **Loss as inconsistency** (Richardson 2022): standard losses are inconsistencies of the right PDG, and regularisers correspond to priors. So "explains the data" can itself be written as a PDG's inconsistency: data edges plus theory edges plus a prior or MDL edge.
- **Inference** (Richardson et al. 2023): the first tractable algorithm. It is a convex exponential-cone program, polynomial for bounded treewidth, solved with interior-point methods.
- **Learning:** *Local Inconsistency Resolution* (Richardson et al., arXiv 2604.17140, 2026) is "a generic algorithm for learning and approximate inference": repeatedly pick a subset of the model and resolve inconsistency using the parameters under control. It recovers EM, belief propagation, GANs and GFlowNets as special cases, and has been tested on synthetic discrete PDGs. **This is the answer to "does PDG learning exist":** yes, as parameter learning by local inconsistency resolution. I found no structure learning (which claims to include).

### 1.5 Neural theorem provers and differentiable reasoning
- **NTP** (Rocktäschel & Riedel 2017): backward chaining with soft unification (an RBF kernel on symbol embeddings); induces rules from rule templates. **Broke on scale:** it must consider all proof paths. GNTP (2019) keeps only the top proof paths, "orders of magnitude more efficient". CTP (2020) learns which rule to select and does well on CLUTRR, which trains on small graphs and tests on larger ones.
- **DeepLogic** (Cingillioglu & Russo 2018): RNN-based iterative networks that learn whether a character-level logic program entails a query. Accuracy fell as programs grew harder and longer (**unverified detail**). This is the closest prior work to our block reasoner, and its limit was length generalisation.

### 1.6 Learning to reason over learned rules (LNN, LTN)
- **Logical Neural Networks** (Riegel et al. 2020): one neuron per subformula; weighted real-valued logic; upper and lower bounds on truth values; inference runs in every direction; the loss penalises contradiction. The bounds are the nearest analogue to LI prices with uncertainty.
- **Logic Tensor Networks** (Badreddine et al. 2020): "Real Logic", fuzzy first-order logic with t-norms; axioms enter as differentiable satisfaction terms.
- **Broke on (my analysis):** fuzzy truth values do not form a distribution over worlds. The best fuzzy satisfaction of a theory is not its model frequency, and t-norm conjunction does not respect identities such as P(A∧B)+P(A∧¬B)=P(A). This is the key reason to keep a world-based or PDG semantics instead of t-norms.

### 1.7 World models as programs / theory learning
- **DreamCoder** (Ellis et al. 2020): wake-sleep. Wake: search for programs, guided by a recognition network, that solve tasks. Sleep (abstraction): compress solutions into new library functions. Sleep (dreaming): train the recognition network on replayed and imagined tasks. It rediscovered basic functional programming and vector algebra and Newton's and Coulomb's laws. **Broke on (unverified):** heavy compute, and enumerative search limits program length.
- Ellis et al. 2022 (Nature Comms) apply Bayesian program induction to grammar theories.
- **Kemp & Tenenbaum 2008, "The discovery of structural form"** (PNAS): selects among structural forms (tree, ring, chain, grid…) and the structure itself by Bayesian posterior; the prior penalises complexity.
- **Ullman, Goodman & Tenenbaum 2012, "Theory learning as stochastic search in the language of thought"** (Cognitive Development): theories are logical rule sets, learned by MCMC over a grammar of theories. **Broke on (unverified):** slow, noisy search; it reproduced children's learning dynamics on toy domains only.
- **Theory-based RL / EMPA** (Tsividis et al. 2021): Bayesian theory induction over a game-description language, plus exploration and planning; human-level sample efficiency on Atari-like games. **TheoryCoder** (Ahmed, Tenenbaum & Bates 2025): an LLM synthesises the low-level transition program and bilevel planning uses it; it notes that earlier TBRL theory languages were "restrictive" and planning was "not scalable".

### 1.8 The "explains the data" criterion
- **MDL:** choose the theory minimising L(theory) + L(data | theory) (the two-part code). Hocquette et al. 2023 apply it to logic programs.
- **Solomonoff induction:** a prior 2^{−|p|} over programs; optimal in the limit but incomputable (Wikipedia; LI shares this "dominates every computable predictor" flavour). Levin search (unverified recollection: time-bounded enumeration weighted by 2^{−|p|}·t) is the computable variant.
- **Bayesian theory induction** (Kemp, Ullman, Tenenbaum, EMPA): posterior ∝ simplicity prior × likelihood.
- **PDG inconsistency:** covers likelihood (cross-entropy), regularisers-as-priors and divergences in one quantity (Richardson 2022). With a prior edge it can play the MDL role. This is my mapping, not a result in the paper.

### 1.9 LLM era (2023-2026): an LLM proposes symbolic models, a solver checks or runs them
- **Hypothesis Search** (Wang et al. 2023): the LLM proposes natural-language hypotheses, implements them in Python and verifies them on the examples; 30% on a 100-task ARC subset versus 17% for direct prompting. **Hypotheses-to-Theories** (Zhu et al. 2023): the LLM induces a library of rules that are often correct, then deduces with it; 10-30 point gains.
- **Doing experiments and revising rules** (Piriyakulkij et al. 2024): LLM proposals, Monte Carlo belief updates over fuzzy, probabilistic rules, and information-theoretic experiment design.
- **WorldCoder** (Tang, Key & Ellis 2024): the world model is a Python program that must explain the agent's interactions, plus an optimism constraint; more sample-efficient than deep RL. **GIF-MCTS** (Dainese et al. 2024): MCTS over generate, improve and fix steps for code world models.
- **PoE-World** (Piriyakulkij et al. 2025): an *exponentially weighted product of many small LLM-written programmatic experts*; it learns stochastic Pong and Montezuma models "from just a few observations". Closest to "a theory = many weighted claims": an MLN whose clauses are programs.
- **Logic-LM, LLM+P** (2023): the LLM translates into a solver language, an exact solver reasons, and solver errors drive refinement. Nothing is learned across problems.
- **BeliefBank / REFLEX / ConCoRD** (2021-2023): raw LLM beliefs become a weighted MaxSAT or factor-graph problem and a solver restores consistency. This is our pattern (intuition, then consistency projection), but MAP-based, not frequency-calibrated.
- **AlphaGeometry** (Nature 2024): a language model proposes constructions, a symbolic engine derives consequences, and training uses synthetic theorems: learned proposer + exact deriver + self-generated data.
- **ARC:** induction (programs) and transduction solve different tasks, and ensembling them nears human level (Li et al. 2024). ARC Prize 2024 moved the state of the art from 33% to 55.5% through program synthesis and test-time training, and the benchmark stayed unbeaten.
- **Broke on:** dependence on the LLM's prior, proposal cost, and brittle translation. "Explains the data" is mostly pass/fail or trajectory likelihood, with no calibrated posterior over theories (PoE-World and Piriyakulkij are the exceptions).

## 2. Cross-cutting lessons for our system

1. **Score = PDG inconsistency of (data edges, theory edges, prior edge).** This unifies likelihood, MDL and Bayes in the object Kelvin already chose. Structure (which claims exist) needs discrete search; confidences can come from LIR or gradients.
2. **Learning needs inference; let it be slow.** ProbLog, MLN and DeepProbLog stalled because every learning step calls #P inference (my reading, partly unverified). Under Kelvin's framing, learning can use exact enumeration (≤ 16 atoms), sampling, or the block reasoner itself run for many blocks.
3. **Keep world semantics, not fuzzy truth.** LTN, LNN and PSL break the identities LI needs. ProbLog, MLN and PDG semantics keep them.
4. **Identifiability (reasoning shortcuts).** If concepts and theory are learned jointly from the final loss alone, the theory can fit the data with unintended meanings. Hold the atoms' semantics fixed, or supervise them.
5. **Proposers and self-generated data carry the wins** (AlphaGeometry, DreamCoder, LLM proposers): a strong proposer plus an exact engine that labels small cases for a fast learner.

## 3. Five candidate designs (learner + block reasoner with the coherent-projection fixed point)

All five share the reasoner from `IDEAS-LI-NET.md`: world seats + clerk + Metropolis + running-average prices, with fixed point = the PDG distribution of {house view, axioms at confidence ∞, intuition at finite β}. The designs differ in what the house view and axioms are and how they are learned.

**D1. PDG theory learner (MDL as inconsistency) → world-seat reasoner.** *Data:* observed partial assignments of atoms, possibly noisy. *Learned:* claims (clauses or conditionals P(head|body)=p) with confidences β, plus a size-penalty edge. Structure comes from Popper-style generate/test/constrain; confidences come from LIR or gradients on inconsistency. *Reasoning:* β=∞ claims become the clerk, finite-β claims shape the house view, and the blocks find worlds. *Risk:* structure search (ILP's old bottleneck); learning calls inference, which is cheap only when theories are small. *Why:* it is literally Kelvin's object, and the score and the reasoner's fixed point are the same PDG.

**D2. MLN-style weighted-clause house view, contrastive weight learning.** *Data:* full or partial worlds. *Learned:* clause weights, so log P(w) = Σ w_i n_i(w), fit by contrastive divergence with negatives drawn from the block seats themselves (my proposal). Clauses come from a fixed pool, or are added greedily by likelihood gain. *Reasoning:* the seat market is MC-SAT-like sampling, and prices are MLN model frequencies. *Risk:* poor mixing with near-hard clauses, which hurts learning and reasoning alike; grounding blow-up on first-order theories. *Cheapest:* the IDEAS house view already has this form.

**D3. LLM-proposed programmatic experts (PoE-World) compiled to propositions.** *Data:* a few trajectories or observations. *Learned:* the LLM proposes small rules or programs; product weights are fit by likelihood and pruned by a size prior; each expert compiles to clauses over fixed atoms (features at t and t+1). *Reasoning:* consequence queries ("is Y reachable in k steps") are priced by the blocks. *Risk:* compilation cost, dependence on an external LLM, and prompt-bounded expressivity. *Fastest route past toy domains.*

**D4. End-to-end soft-clause theory (∂ILP/NeuralLP style) trained jointly with the reasoner.** *Data:* (observations, query, answer). *Learned:* soft weights over a clause pool, annealed toward crisp values so the clerk can use them, with the loss at a random block from 1 to 20. *Risk:* reasoning shortcuts and non-identifiability (Marconato), local minima, fixed templates, and a theory that may come out uninterpretable. Use it as an ablation.

**D5. DreamCoder-style wake-sleep with a reasoner trained on dreamed theories.** *Data:* a stream of small domains. *Learned:* a library of clause schemas compressed from solved domains; a recognition net proposing theories (for the learner) and worlds (for the reasoner); the block reasoner. Sleep samples theories from the library and labels them exactly (entailment and model frequencies by enumeration or #SAT, ≤16 atoms). *Reasoning:* blocks trained on those exact labels, tested on bigger theories at 100 and 500 blocks. *Risk:* dreams may not match real theories; compute; the abstraction step is complex. The AlphaGeometry recipe applied to LI properties.

**My recommendation (proposal, not decided):** start with D2 on top of the current IDEAS prototype, since the house view is the MLN and the seats are its sampler. Add D1's structure search and MDL edge once weights fit. Use D5's exact-label dreaming as the training-data source throughout.

## Sources (all fetched 2026-10-02)

ILP
- Cropper & Morel, Learning programs by learning from failures (Popper): https://arxiv.org/abs/2005.02259 ; MLJ version https://doi.org/10.1007/s10994-020-05934-z ; code https://github.com/logic-and-learning-lab/Popper
- Cropper & Dumančić, ILP at 30: https://arxiv.org/abs/2008.07912
- Hocquette, Niskanen, Järvisalo, Cropper, Learning MDL logic programs from noisy data: https://arxiv.org/abs/2308.09393
- Metagol: https://github.com/metagol/metagol
- ILASP: https://github.com/ilaspltd/ILASP-releases (ilasp.com failed TLS on fetch)
- Evans & Grefenstette, ∂ILP: https://arxiv.org/abs/1711.04574
- Yang, Yang & Cohen, NeuralLP: https://arxiv.org/abs/1702.08367
- Dong et al., Neural Logic Machines: https://arxiv.org/abs/1904.11694

Probabilistic logic
- De Raedt, Kimmig, Toivonen, ProbLog (IJCAI 2007): https://www.ijcai.org/Proceedings/07/Papers/396.pdf ; site https://dtai.cs.kuleuven.be/problog/
- Manhaeve et al., DeepProbLog: https://arxiv.org/abs/1805.10872
- Marconato et al., Reasoning shortcuts: https://arxiv.org/abs/2305.19951
- Richardson & Domingos, Markov logic networks: https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf ; https://doi.org/10.1007/s10994-006-5833-1
- Bach et al., Hinge-loss MRFs and PSL: https://arxiv.org/abs/1505.04406 ; https://psl.linqs.org/
- Xu et al., Semantic loss: https://arxiv.org/abs/1711.11157

PDGs
- Richardson & Halpern, Probabilistic Dependency Graphs: https://arxiv.org/abs/2012.10800
- Richardson, Loss as the inconsistency of a PDG: https://arxiv.org/abs/2202.11862
- Richardson et al., Inference for PDGs: https://arxiv.org/abs/2311.05580
- Richardson et al., Local Inconsistency Resolution (2026): https://arxiv.org/abs/2604.17140

Neural provers / differentiable reasoning
- Rocktäschel & Riedel, NTP: https://arxiv.org/abs/1705.11040
- Minervini et al., GNTP: https://arxiv.org/abs/1912.10824
- Minervini et al., CTP: https://arxiv.org/abs/2007.06477
- Cingillioglu & Russo, DeepLogic: https://arxiv.org/abs/1805.07433
- Riegel et al., Logical Neural Networks: https://arxiv.org/abs/2006.13155
- Badreddine et al., Logic Tensor Networks: https://arxiv.org/abs/2012.13635

Programs and theories
- Ellis et al., DreamCoder: https://arxiv.org/abs/2006.08381
- Ellis et al., Synthesizing theories of human language: https://www.nature.com/articles/s41467-022-32012-w
- Kemp & Tenenbaum, The discovery of structural form: https://www.pnas.org/doi/10.1073/pnas.0802631105 (PubMed https://pubmed.ncbi.nlm.nih.gov/18669663/)
- Ullman, Goodman, Tenenbaum, Theory learning as stochastic search: https://doi.org/10.1016/j.cogdev.2012.07.005
- Tsividis et al., Theory-based RL (EMPA): https://arxiv.org/abs/2107.12544
- Ahmed, Tenenbaum, Bates, TheoryCoder: https://arxiv.org/abs/2503.20124
- MDL: https://en.wikipedia.org/wiki/Minimum_description_length
- Solomonoff induction: https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference
- Garrabrant et al., Logical Induction: https://arxiv.org/abs/1609.03543

LLM era
- Wang et al., Hypothesis Search: https://arxiv.org/abs/2309.05660
- Zhu et al., Hypotheses-to-Theories: https://arxiv.org/abs/2310.07064
- Piriyakulkij et al., Doing experiments and revising rules: https://arxiv.org/abs/2402.06025
- Tang, Key, Ellis, WorldCoder: https://arxiv.org/abs/2402.12275
- Dainese et al., GIF-MCTS code world models: https://arxiv.org/abs/2405.15383
- Piriyakulkij et al., PoE-World: https://arxiv.org/abs/2505.10819
- Pan et al., Logic-LM: https://arxiv.org/abs/2305.12295
- Liu et al., LLM+P: https://arxiv.org/abs/2304.11477
- Kassner et al., BeliefBank: https://arxiv.org/abs/2109.14723
- Kassner et al., REFLEX (Language Models with Rationality): https://arxiv.org/abs/2305.14250
- Mitchell et al., ConCoRD: https://arxiv.org/abs/2211.11875
- Trinh et al., AlphaGeometry: https://www.nature.com/articles/s41586-023-06747-5
- Li et al., Combining induction and transduction (ARC): https://arxiv.org/abs/2411.02272
- Chollet et al., ARC Prize 2024 technical report: https://arxiv.org/abs/2412.04604 ; https://arcprize.org/
