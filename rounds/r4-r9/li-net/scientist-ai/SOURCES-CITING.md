# Scientist AI: citing and adjacent work (tracks a and b)

Compiled 2026-10-03 for eurisko. Scope: work that cites Bengio et al., "Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?" (arXiv 2502.15657) or "Can a Bayesian Oracle Prevent Harm from an Agent?" (arXiv 2408.05284), plus adjacent work on (a) a consistent world model expressed or queryable in words and (b) a reasoner whose estimates improve with more thinking. Bengio/LawZero primary papers are left to the sibling file.

Legend. [F] = full text fetched (arXiv HTML or PDF) and searched on 2026-10-03; [A] = abstract only (arXiv or Semantic Scholar API). "Cites SAI": "yes (text)" = reference to 2502.15657 found in the fetched text; "yes (S2)" = listed by the Semantic Scholar citation graph, text not checked; "no" = fetched text has no reference to 2502.15657, 2408.05284 or the LawZero paper; "not in S2" = abstract only and absent from the S2 citer list. Grades: DS = implemented and demonstrated at scale (frontier or >=30B models, real tasks); TOY = implemented in toy or narrow experiments; SPEC = specified, not implemented; POS = position only. Numbers are as reported by the authors (claimed, not reproduced).

## 0. Citation sweep

- Semantic Scholar lists 90 citers of 2502.15657 and 13 of 2408.05284 (queried 2026-10-03). Google Scholar shows "Cited by 127" for 2502.15657, but its cited-by pages returned a CAPTCHA, so about 37 Google-only citers are unchecked.
- By title, most of the 90 are governance, policy, security, survey or AI-scientist-agent papers. About a dozen touch tracks (a)/(b): entries A6, A7, A20, A21, B6 and section (a5) below.
- No citing paper outside LawZero implements the Scientist AI world model or inference machine. Primary items surfaced by the sweep and left to the sibling: Bengio, Richardson, Gavenčiak, Cohen et al., "Safety from Honesty in a Disinterested AI Predictor", arXiv 2606.29657 (2026-06-28); Fornasiere, Richardson, Gendron, Serban, Bengio, "The Scientist AI: safe by design, by not desiring" (LawZero 2026, https://lawzero.org/en/publication/scientist-ai-safe-design-not-desiring, cited by A6).
- PDGs applied to LLMs: none found among citers or by web search. Richardson et al., "Local Inconsistency Resolution" (arXiv 2604.17140, with Bengio) reads transformer attention as an LIR step (section 4.5) but has no LLM experiments and does not cite SAI (local copy lir2026.txt searched).
- The brief's "Yang et al. 2024, Verbalized PGM" is Huang et al., arXiv 2406.05516 (entry A9).

## 1. Track (a): a consistent world model expressed or queryable in words

### (a1) An LLM writes an explicit probabilistic model; an inference engine answers the query

**A1. Wong, Grand, Lew, Goodman, Mansinghka, Andreas, Tenenbaum, "From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought", 2023-06-22.** https://arxiv.org/abs/2306.12672 [F]. Cites SAI: no (predates).
- Recipe: few-shot Codex translates each utterance into Church code; Church inference answers questions. No training.
- Shown: worked examples in four domains (probabilistic, relational, physical, social reasoning); the authors call them "illustrative and pedagogical". No benchmark.
- Missing: training, scale, evaluation. Grade: SPEC (framework plus demos).

**A2. Wong, Collins, Ying, Zhang, Weller, Gerstenberg, O'Donnell, Lew et al., "Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models" (MSA), 2025-07-16.** https://arxiv.org/abs/2507.12547 [F]. Cites SAI: no.
- Recipe: Llama-3.1-70B retrieves relevant considerations and writes a bespoke WebPPL model per vignette; WebPPL inference gives the answer. No training.
- Shown: three human experiments on "Model Olympics" sports vignettes; MSA matches human judgments better than the same LM answering directly or with CoT.
- Missing: one model per query (no persistent world model, nothing ties answers across queries); one domain. Grade: TOY.

**A3. Zhang, Jagadish, Lake, Griffiths, "Using Probabilistic Programs to Train Inductive Reasoning in Large Language Models" (PPT), 2026-05-26.** https://arxiv.org/abs/2606.09856 [F]. Cites SAI: no.
- Recipe: a scaled MSA synthesizes 10,000 scenarios (sports, healthcare, general) as probabilistic programs; Pyro inference gives posterior distributions for queries; LoRA fine-tuning on those soft labels.
- Shown: Llama-3-8B-Instruct and Qwen-2-7B-Instruct; lower error against program posteriors on held-out motifs and a held-out domain; correlation with human judgments 0.509 (base) to 0.775 (tuned), vs 0.769 for MSA itself; Gemini-3.1-Pro did worse than base Llama-3-8B on these queries; transfer gains on seven external benchmarks; NLL/ECE gains on multiple-choice tasks that remain after temperature scaling.
- Missing: inherits the synthesized programs' errors; no check of coherence across queries; 7-8B only. Grade: TOY. The most recipe-like path found: synthesize models, run probabilistic inference, distil the soft posteriors into weights.

**A4. Li, Fox, Goodman, "Automated Statistical Model Discovery with Language Models" (BoxLM), 2024-02-27.** https://arxiv.org/abs/2402.17879 [F]. Cites SAI: no.
- Recipe: GPT-4 alternates proposing probabilistic programs and critiquing them (Box's loop). No training.
- Shown: models "on par with human expert designed models" (abstract) on classic datasets (Eight Schools, Dugongs, Lotka-Volterra among them).
- Missing: per-dataset models; no language query interface. Grade: TOY.

**A5. Wahl, Schenk, Farnoud, Macke, Gedon, "A Probabilistic Framework for LLM-Based Model Discovery" (ModelSMC), 2026-02-20.** https://arxiv.org/abs/2602.18266 [F]. Cites SAI: no.
- Recipe: model discovery as Sequential Monte Carlo; the LLM proposes and refines simulator code as particles, which are weighted by likelihood and resampled.
- Shown: real scientific systems (e.g. Hodgkin-Huxley); interpretable mechanisms, better posterior predictive checks.
- Missing: no training; mechanistic simulators, not world knowledge in words. Grade: TOY.

**A6. Karwowski, Kaddar, Ye, Whitammer, Staton, "Likelihood Hacking in Probabilistic Program Synthesis", 2026-03-25.** https://arxiv.org/abs/2603.24126 [F]. Cites SAI: yes (text; frames the language an AI scientist states theories in, and cites the LawZero paper).
- Recipe: GRPO-train Qwen3-4B-Instruct-2507 to write PyMC programs with a marginal-likelihood reward; define a safe language fragment that provably cannot inflate likelihood with unnormalized programs (SafeStan, plus a SafePyMC gate).
- Shown: the policy finds likelihood-hacking exploits within the first few RL steps in both runs; the checkers reject every exploit family found.
- Missing: a reward fix, not a trained world model. Grade: TOY. The most direct technical citer of SAI on the "theories as programs scored by likelihood" route.

**A7. Yu, Yuan, Xiao, Xia, Fu, Zhang, Lin, Liu, "Generating Symbolic World Models via Test-time Scaling of Large Language Models", 2025-02-07.** https://arxiv.org/abs/2502.04728 [F]. Cites SAI: yes (text). Also track (b).
- Recipe: no training; Best-of-N sampling of PDDL domains, then iVML refinement (an optimizer LLM edits the domain in words).
- Shown: Qwen2.5-Coder-7B reaches 85.2% (NL2Domain) and 71.4% (Prob2Domain) vs o1-mini 41.7%/33.7%. Compute curve: the BoN validation rate rises steeply, then saturates near 90% from N=32 with slight degradation; iVML keeps improving past the BoN plateau.
- Missing: deterministic planning domains, no uncertainty. Grade: TOY.

**A8. Feng, Zhou, Lin, Roth, "BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models", 2024-04-18.** https://arxiv.org/abs/2404.12494 [F]. Cites SAI: no.
- Recipe: the LLM abduces natural-language factors; a Bayesian network over them has its conditional tables fitted to the LLM's coarse predictions on factor combinations; deduction gives the probability.
- Shown: estimates agree with human judgments on 30% more instances than direct LLM probabilities.
- Missing: one network per question, fitted to the LLM's own estimates. Grade: TOY.

**A9. Huang, Shen, Wang, Meng, Liu, Duchene, Wang, Bhatt, "BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling" (vPGM), 2024-06-08.** https://arxiv.org/abs/2406.05516 [F]. Cites SAI: no.
- Recipe: a prompt has the LLM state latent variables and dependencies in words and infer them; BayesVPGM adds a Dirichlet posterior over the prediction, fitted by L-BFGS with a differentiable class-wise calibration loss.
- Shown: ScienceQA, ChatCoach, A-OKVQA-noisy (accuracy 61.03% vs 59.04% for Chameleon+, with lower ECE).
- Missing: the graph lives in a prompt; the LLM is not trained. Grade: TOY.

**A10. Gkountouras, Lindemann, Lippe, Gavves, Titov, "Language Agents Meet Causality: Bridging LLMs and Causal World Models", 2024-10-25.** https://arxiv.org/abs/2410.19923 [A]. Cites SAI: not in S2. Causal representation learning gives a causal world model whose variables are linked to language; the LLM queries it as a simulator for causal inference and planning. Grade: TOY.

### (a2) Hypotheses or parameters written in words, with a fitted model or a posterior

**A11. Zhong, Wang, Klein, Steinhardt, "Explaining Datasets in Words: Statistical Models with Natural Language Parameters", 2024-09-13.** https://arxiv.org/abs/2409.08466 [F]. Cites SAI: no.
- Recipe: clustering, time-series and classification models whose parameters are predicates in words; optimize continuous relaxations by gradient descent, then discretize by prompting an LM.
- Shown: chat taxonomies, model comparisons, math-problem clustering; relaxation and refinement raise F1 on five datasets.
- Missing: describes datasets; no uncertainty over hypotheses. Grade: TOY.

**A12. Xiao, Bamler, Schölkopf, Liu, "Verbalized Machine Learning: Revisiting Machine Learning with Language Models", 2024-06-06.** https://arxiv.org/abs/2406.04344 [F]. Cites SAI: no.
- Recipe: the model is a prompt theta in words, run by an LLM; an optimizer LLM rewrites theta from data batches.
- Shown: regression, classification, text and medical-image tasks with Llama-3-70B and GPT-4o.
- Missing: one hypothesis, no uncertainty; results vary run to run (the motivation for A13). Grade: TOY.

**A13. Zhang, Lian, Li, "Verbalized Particle Posterior: Bayesian Inference over Natural Language Hypotheses" (VPP), 2026-07-25.** https://arxiv.org/abs/2607.22961 [F]. Cites SAI: no. Also track (b).
- Recipe: K hypotheses in words as particles, updated by Metropolis-Hastings or SMC (tempering, resampling); predictions by Bayesian model averaging; black-box Llama-3-70B-Instruct.
- Shown: better than a single VML run on every benchmark; matches or beats an oracle best-of-5 VML ensemble on most; no catastrophic single-run failures. Particle-count sweep: linear-regression MSE improves monotonically with K; two-circles is saturated; sum-parity improves, then dips at the largest K (mild degeneracy).
- Missing: small synthetic tasks, one LLM; the acceptance ratio keeps only the likelihood ratio and drops the prior terms, so the posterior is approximate. Grade: TOY.

**A14. Feng, Kothari, Zier, Singh, Tan, "Bayesian Concept Bottleneck Models with LLM Priors" (BC-LLM), 2024-10-21.** https://arxiv.org/abs/2410.15555 [F]. Cites SAI: no.
- Recipe: Bayesian selection over an unbounded set of concepts in words; the LLM serves as prior, proposal (split-sample) and annotator; Metropolis-Hastings corrects its mistakes.
- Shown: image (CUB birds), clinical notes (MIMIC) and tabular data; beats interpretable baselines and sometimes black-box models; reaches relevant concepts faster.
- Missing: per-task predictors, not a world model. Grade: TOY.

**A15. The Ellis line.** Ellis, "Human-like Few-Shot Learning via Bayesian Reasoning over Natural Language", 2023, https://arxiv.org/abs/2306.02797 [F]; Piriyakulkij, Langenfeld, Le, Ellis, "Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning", 2024, https://arxiv.org/abs/2402.06025 [F]; Piriyakulkij, Acquaviva, Langenfeld, Tenenbaum, Ellis, "Induction and Inquiry via Probabilistic Reasoning over Language and Code", 2026-09-01, https://arxiv.org/abs/2609.01815 [F]. Cites SAI: no (all three).
- Recipe: an LLM proposes hypotheses in words (later words plus code); weighting by prior times likelihood (2023), then online SMC with LLM-guided revision and information-gain experiment choice (2024, 2026).
- Shown: number-concept judgments fitted to humans (2023, Codex proposals); more accurate rule recovery on Zendo/ActiveACRE than LLM revise-hypothesis baselines (2024); human anchoring and garden-pathing reproduced, while pure LLMs and classic Bayesian models fail the task, miss the human effects, or need "exorbitant" compute (2026).
- Missing: cognitive-task scale; the proposer is not trained. Grade: TOY.

### (a3) Training the LLM's own credences to be consistent

**A16. Akyürek, Akyürek, Choshen, Wijaya, Andreas, "Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability" (DCT), 2024-01-16.** https://arxiv.org/abs/2401.08574 [F]. Cites SAI: no.
- Recipe: the LM generates implications and contradictions of seed text, reasons about which generated statements are jointly consistent, and is fine-tuned on those; self-generated seeds make it unsupervised.
- Shown: Llama-2-7B; supervised DCT +3-26% on CREAK, MQuAKE and Reversal Curse; unsupervised DCT +12% verification accuracy on CREAK.
- Missing: binary truth values, not graded probabilities; 7B. Grade: TOY.

**A17. Calanzone, Teso, Vergari, "Logically Consistent Language Models via Neuro-Symbolic Integration" (LoCo-LMs), 2024-09-09.** https://arxiv.org/abs/2409.13724 [F]. Cites SAI: no.
- Recipe: a semantic loss (the probability that given logical constraints hold: implication, negation, combinations) is the fine-tuning objective, against a fact and rule base.
- Shown: Macaw-Large and Llama-2-7B on BeliefBank/EntailmentBank: more self-consistent than SFT, ConCoRD and a prompted Llama-2-70B; extrapolates to unseen facts.
- Missing: constraints must be supplied; narrow knowledge bases. Grade: TOY.

**A18. Qiu, Sha, Allen, Kim, Linzen, van Steenkiste, "Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models", 2025-03-21.** https://arxiv.org/abs/2503.17523 [F]. Cites SAI: no.
- Recipe: SFT on interactions of a normative Bayesian assistant (multi-round flight recommendation with latent user preferences), not on oracle answers.
- Shown: Gemma-2-9B, Llama-3-8B and Qwen-2.5-7B agree with the Bayesian assistant about 80% of the time; this beats oracle teaching and generalizes to unseen hotel-recommendation and web-shopping tasks.
- Missing: one latent-preference model family; needs an exact Bayesian teacher. Grade: TOY.

**A19. Kim, Lee, Fong, Lee, Lee, "From Drift to Coherence: Stabilizing Beliefs in LLMs", 2026-06-16.** https://arxiv.org/abs/2606.17832 [F]. Cites SAI: no. Also track (b).
- Recipe: prompted predictive resampling (the model writes a sequence of answers to one multiple-choice question) exposes martingale violations as early drift; fixes are seed answers, or a self-consistency (c.i.d.) loss that amortizes the resampling.
- Shown: Llama-3.1-8B on CommonsenseQA and other MCQA; early drift, then self-stabilization to a coherent predictive (like MCMC burn-in); fine-tuning reduces drift and improves calibration with no accuracy loss.
- Missing: per-question predictive coherence only; discrete answers. Grade: TOY.

**A20. Xu, Zhao, Yao, Xu, Wang, Deng, Deng, Pan et al., "Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency", 2026-01-09.** https://arxiv.org/abs/2601.05905 [F]. Cites SAI: yes (text, reference list).
- Recipe: measure belief by coherence across a neighborhood of related facts (NCB); Structure-Aware Training optimizes context-invariant belief structure.
- Shown: facts that Qwen3-30B-A3B answers correctly in 30/30 samples still collapse under mild contextual interference; SAT cuts long-tail knowledge brittleness by about 30%.
- Missing: robustness of factual recall, not probabilistic coherence. Grade: TOY.

**A21. Pres, Li, Ruis, Guo, Hu, Damani, Puri, Lubana et al., "Position: It's Time to Optimize LLMs for Self-Consistency", 2026-07-31.** https://arxiv.org/abs/2608.05188 [F]. Cites SAI: yes (text; cites it on the concern that a self-model enabling self-locating inference emerges as consistency constraints grow).
- Claim: fixes for robustness, sycophancy, factual coherence and calibration are special cases of one "consistency optimization" over related input-output pairs; proposes generally consistent LMs. No new experiments. Grade: POS.

**A22. Andrews, "Revealed Rationality: Label-Free Evaluation and Regularization from Representation Theorems", 2026-08-05.** https://arxiv.org/abs/2608.05015 [F]. Cites SAI: no. Continuous penalties, zero exactly when behavior is rationalizable (de Finetti coherence, Afriat, Echenique-Saito), computed from the model's own answers without labels; discusses implementation, no training run. Grade: SPEC.

**A23. Lee, Yang, Kwak, Siegel, "Advancing Event Forecasting through Massive Training of Large Language Models", 2025-07-25.** https://arxiv.org/abs/2507.19477 [A]. Cites SAI: not in S2. Proposes hypothetical event Bayesian networks, counterfactual events and auxiliary rewards (including logical consistency) for forecaster RL. Grade: POS.

Consistency measurements without training (none in the S2 citer list):
- Paleka, Pallavi Sudhir, Alvarez, Bhat, Shen, Wang, Tramèr, "Consistency Checks for Language Model Forecasters", 2024-12-24, https://arxiv.org/abs/2412.18544 [F]. Arbitrage and frequentist violation scores over 10 check types; correlation with later Brier varies by check and dataset (Negation arbitrage 0.60 on scraped questions, -0.36 on NewsAPI); the conditional (Bayesian) checks are the best proxies. An inference-time ArbitrageForecaster improves only the checks it targets and does not improve accuracy. The authors suggest training on the metrics but do not do it.
- Zhu, Griffiths, "Incoherent Probability Judgments in Large Language Models", 2024, https://arxiv.org/abs/2401.16646 [A]. LLM probability judgments violate probability identities, with human-like biases.
- Falck, Wang, Holmes, "Is In-Context Learning in LLMs Bayesian? A Martingale Perspective", 2024, https://arxiv.org/abs/2406.00793 [A]. Martingale violations in synthetic in-context learning.
- Chen, Jörke, Goliński, Fedzechkina, Sapiro, Williamson, Foti, "LLMs are not (consistently) Bayesian", 2026, https://arxiv.org/abs/2605.06915 [A]. Some ways of feeding evidence give near-Bayesian updates, others heuristics; the heuristics often beat exact Bayes downstream, i.e. the LLM's implied world model is misspecified.
- Wolf, Kleine Buening, Krause, Mendler-Dünner, "Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models", 2026, https://arxiv.org/abs/2607.15277 [A]. Tests the law of total probability over recursive population partitions.
- Kotawala, "Locally Coherent, Globally Incoherent: Bounding Compositional Incoherence in Multi-Component LLM Agents", 2026, https://arxiv.org/abs/2605.30335 [A]. Claims composed from locally coherent components violate the probability axioms; a hierarchical Boyle-Dykstra projection onto the coherent polytope repairs them at runtime.

### (a4) Text world models with an explicit latent state

**A24. Gao, Dong, Yao, Jonnalagedda, Das, "Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation", 2026-06-26.** https://arxiv.org/abs/2606.27681 [F]. Cites SAI: no.
- Recipe: the world model writes a variable-length text state; predictions may depend only on (state, action) (strict mediation); trained with factorized, tree-structured GRPO.
- Shown: Qwen2.5-0.5B, Qwen3-4B and Qwen3-32B on TextWorld and ScienceWorld; one-step accuracy preserved, up to 57% better representation quality and up to 98% better rollouts; models that see history predict well while learning uninformative states.
- Missing: deterministic text games; the state is a summary, not a calibrated distribution. Grade: TOY. The closest recipe found for an inspectable textual latent state.

**A25. Li, Wang, Qiu, Yin, Zhang, Qian, "From Word to World: Can Large Language Models be Implicit Text-based World Models?", 2025-12-21.** https://arxiv.org/abs/2512.18832 [A]. Cites SAI: not in S2. Trained next-state predictors in five text environments keep a coherent latent state, scale with data and model size, and help agents; the gains depend on behavioral coverage. Grade: TOY.

**A26. Zuo, Xiao, Sheng, Huang, Tu, Liu et al., "Qwen-AgentWorld: Language World Models for General Agents", 2026-06-23.** https://arxiv.org/abs/2606.24597 [A]. Cites SAI: not in S2. Pipeline: CPT on state transitions and professional corpora, SFT for next-state prediction, then RL with rubric-and-rule rewards, over >10M trajectories in 7 domains; 35B-A3B and 397B-A17B models. Missing: environment simulation only, with no beliefs, calibration or coherence objective. Grade: DS (deterministic simulation only).

**A27. Kumar, Kumar, Ahuja, Jha, "Towards a Belief-Based World Model for LLM Agents", 2026-08-31.** https://arxiv.org/abs/2609.00455 [A]. Cites SAI: not in S2. A world model keeps a belief that the LLM policy can query about what is known and uncertain; the paper tests whether exposing that belief helps, before learning BB-WMs. Grade: TOY.

**A28. Murphy, "Agentic Forecasting with Structured Linguistic Beliefs" (BLF), 2026-04-20.** https://arxiv.org/abs/2604.18576 [F]. Cites SAI: no. Also track (b).
- Recipe: a ReAct forecaster whose context carries a JSON belief state (probability, confidence, evidence for and against), rewritten each step; K independent trials averaged (arithmetic or logit mean); post-hoc calibration. No training; the update is "Bayesian-inspired", not exact.
- Shown: Gemini-3.1-Pro, Gemini-3-Flash and GPT-5.4 on 400 ForecastBench question-dates; highest Brier Index among the external agents compared, and the only one significantly above a crowd plus empirical-prior baseline. The belief state helps Pro and Flash but not GPT; Brier Index rises with more trials K; Markov memory (belief state only) scores 75.12 vs 72.72 for full history on 100 fresh questions.
- Missing: the belief updater is not trained or checked for coherence. Grade: TOY (frontier models, 400 instances).

### (a5) Other SAI citers that touch the world model (one line each)
- Worden, "AI and World Models", 2026-01-25, https://arxiv.org/abs/2601.17796 [A], cites SAI (S2): essay; a guardrail world model must model the human social world. POS.
- Da Costa, Namjoshi, Ansari, Schölkopf, "Natural Building Blocks for Structured World Models", 2025-11-03, https://arxiv.org/abs/2511.02091 [A], cites SAI (S2): hierarchical HMM/sLDS world models, interpretable but not in language; scaling is the open problem. POS/TOY.
- Zinage, "SVI-DAG: A Structured Variational Inference Approach to Bayesian Causal Discovery", 2026-08-05, https://arxiv.org/abs/2608.04930 [A], cites SAI (S2): posterior over causal DAGs (normalizing flows, SVGD); numeric variables, not language. TOY.
- Papamarkou, Alquier, Bauer, Buntine et al., "Position: agentic AI orchestration should be Bayes-consistent", 2026-05-01, https://arxiv.org/abs/2605.00742 [A], cites 2408.05284 (S2): Bayesian beliefs in the orchestration layer, not inside the LLM. POS.
- Heitzig, "Unbiased Canonical Set-Valued Oracles Via Lattice Theory", 2026-06-24, https://arxiv.org/abs/2606.26418 [A], cites SAI (S2): a performative oracle reports the least fixed point of a credal-set reaction map. SPEC (theory).
- Friedl, Ward, Rapoport, Everitt, Richens, "The Impossibility of Eliciting Latent Knowledge", 2026-06-10, https://arxiv.org/abs/2606.12268 [A], cites SAI (S2): formal ELK with causal influence diagrams. Theory.
- Liu, Dhole, Wang, Wen et al., "Toward Honest Language Models for Deductive Reasoning", 2025-11-12, https://arxiv.org/abs/2511.09222 [A], cites SAI (S2): ANCHOR injects ground-truth trajectories into GRPO rollouts so the model abstains when premises do not entail an answer. TOY.

## 2. Track (b): a scientist reasoner whose estimates improve with more thinking

### (b1) Trained reasoners with calibrated estimates

**B1. Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas, "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty" (RLCR), 2025-07-22, ICLR 2026 (per FutureSim's references).** https://arxiv.org/abs/2507.16806 [F]. Cites SAI: no.
- Recipe: GRPO with reward = correctness + Brier score of a verbalized confidence given after an uncertainty analysis; the paper proves this reward favors both accuracy and calibration.
- Shown: Qwen2.5-7B base; RLVR-level accuracy with ECE 0.37 to 0.03 (HotpotQA) and 0.26 to 0.10 (math); better out-of-domain calibration than RLVR or post-hoc classifiers. Test-time: confidence-weighted voting beats plain majority as samples N grow; Brier improves "modestly" with an ensemble of K re-sampled analyses; explicit uncertainty reasoning lowers Brier and ECE.
- Missing: needs checkable answers; no curve of Brier against reasoning length. Grade: TOY (7B), clear recipe.

**B2. Chandak, Goel, Prabhu, Hardt, Geiping, "Scaling Open-Ended Reasoning to Predict the Future" (OpenForecaster), 2025-12-31.** https://arxiv.org/abs/2512.25070 [F]. Cites SAI: no.
- Recipe: 50,000 open-ended questions (OpenForesight) synthesized from 250,000 offline news articles; RL on Qwen3 thinking models (4B, 8B) with retrieval and an accuracy + Brier reward.
- Shown: OpenForecaster-8B matches much larger proprietary models on held-out May-August 2025 questions; calibration gains transfer to other benchmarks; consistency improves 44% (arbitrage) and 19% (frequentist) on Paleka et al.'s checks without training on them; the accuracy-only reward hurts calibration and the Brier-only reward hurts exploration.
- Missing: no test-time compute curve. Grade: TOY (8B, real questions).

**B3. Turtel, Franklin, Skotheim, Hewitt, Schoenegger, "Outcome-based Reinforcement Learning to Predict the Future", 2025-05-23.** https://arxiv.org/abs/2505.17989 [F]. Cites SAI: no.
- Recipe: RLVR (ReMax, modified GRPO) on resolved prediction-market questions with news; stability guardrails; ensembling at inference.
- Shown: DeepSeek-R1-14B base Brier 0.215; ReMax on 10k questions 0.199 (ECE 0.067); 100k questions plus a 7-sample ensemble 0.190 (ECE 0.062); o1 0.202; 1,265 held-out questions; simulated Polymarket ROI above 10%.
- Missing: data-scale and ensemble gains are not separated. Grade: TOY (14B, real questions).

**B4. Turtel, Wilczewski, Skotheim, Satopää, Tetlock, "How Proper Scoring Rules Shape LLM Forecasting", 2026-08-28.** https://arxiv.org/abs/2608.28482 [F]. Cites SAI: no.
- Shown: RL on GPT-OSS-120b with five proper scoring rules. The Brier-trained model has the best Brier (0.1648) and AUC; the log-trained model has the best log score and ECE (0.0434). Training-time output length differs up to about 7x across rules, with smaller differences in accuracy; at evaluation the Brier reward averages 1,089 tokens vs 454 for the log reward, at Brier 0.1648 vs 0.1653.
- Missing: no thinking-budget sweep. Grade: DS (120B, real events). More tokens did not buy accuracy here.

**B5. Halawi, Zhang, Chen, Steinhardt, "Approaching Human-Level Forecasting with Language Models", 2024-02-28.** https://arxiv.org/abs/2402.18563 [F]. Cites SAI: no. Retrieval, reasoning prompts, a fine-tuned GPT-4 and trimmed-mean ensembling: Brier .179 vs .149 for the crowd, accuracy 71.5% vs 77.0%. Grade: DS; no compute curve.

**B6. Ma, Lee, Yang, Chen, Ma, Chua, Chang, "Less Data Approximates More: Earning Faithful Confidence in High-Stakes Domains" (HyTuning), 2026-04-09.** https://arxiv.org/abs/2604.08454 [F]. Cites SAI: yes (text). Progressive Reasoning Gain scores whether confidence in the final answer builds up along the reasoning, and uses it to weight reasoning distillation against RL from internal feedback; e.g. Qwen2.5-3B on domain and general benchmarks. Grade: TOY.

**B7. Xi, Zhang, Lv, Luo, Zhao, "Calibration, Not Answer Selection: Distilling Internal Confidence in Reasoning Models", 2026-09-27.** https://arxiv.org/abs/2609.33290 [A]. A linear probe between the CoT and the answer has 5-38x lower ECE than verbalized confidence but only nearly ties majority voting at choosing answers; Probe-SD distils the probe's confidence into the stated confidence. Grade: TOY.

### (b2) Estimate quality vs thinking or sampling compute

Positive or mixed-positive:
**B8. Yoon, Kim, Yang, Kim, Kim, Kim, Choi, Kim et al., "Reasoning Models Better Express Their Confidence", 2025-05-20.** https://arxiv.org/abs/2505.14489 [F]. Cites SAI: no. Six reasoning models against their non-reasoning counterparts on six datasets: strictly better calibration in 33 of 36 settings. Brier over cumulative CoT segments falls as the CoT unfolds (not for non-reasoning models), and removing slow-thinking behaviors hurts. Budget forcing ("Wait") does not necessarily improve calibration further. Grade: DS (natural CoT; forced extra thinking does not help).

**B9. Goel, Chandak, Arun, Prabhu, Staab, Hardt, Andriushchenko, Geiping, "FutureSim: Replaying World Events to Evaluate Adaptive Agents", 2026-05-14.** https://arxiv.org/abs/2605.15188 [F]. Cites SAI: no. Agents forecast January-March 2026 events against a replayed news stream; the best agent reaches 25% accuracy and many have negative Brier skill. GPT-5.5 swept over five reasoning efforts: accuracy and Brier skill rise with effort, with the largest gain leaving "none" (very negative Brier skill) and a plateau after "high". Higher effort also means more tool calls (confounded with search). Grade: DS (one model, controlled sweep).

**B10. Lyu, Shridhar, Malaviya, Zhang, Elazar, Tandon, Apidianaki, Sachan et al., "Calibrating Large Language Models with Sample Consistency", 2024-02-21.** https://arxiv.org/abs/2402.13904 [F]. Cites SAI: no. Confidence from agreement among samples beats post-hoc methods on nine reasoning datasets; Brier improves with more samples on 3 of 4 datasets (GPT-3.5-turbo, CoT). Grade: DS (sampling axis).

**B11. Alur, Stadie, Kang, Chen, McManus, Rickert et al., "AIA Forecaster: Technical Report", 2025-11-10.** https://arxiv.org/abs/2511.07678 [F]. Cites SAI: no. Agentic search, many independent forecasts, a supervisor agent that reconciles them with more search, then Platt scaling (shown equivalent to a class of extremization). Claims parity with superforecasters on ForecastBench; below market consensus on a liquid-markets benchmark but adds information in an ensemble with it. Single runs are unstable, and ensembling is "absolutely essential". Grade: DS (claimed).

**B12. Gottweis, Weng, Daryin, Tu et al., "Accelerating scientific discovery with Co-Scientist", 2025-02-26.** https://arxiv.org/abs/2502.18864 [F]. Cites SAI: no. Hypothesis Elo (the system's own auto-evaluation) rises with test-time compute over 203 research goals; the Evolution agent raises GPQA precision from 70.9% to 75.4%. No probabilities. Grade: DS (self-evaluated).

**B13. Peng, Luo, Chen, Tang, "Two Heads Are Better Than One: Aggregating Weaker LLMs for Better Forecasts", 2026-09-27.** https://arxiv.org/abs/2609.33257 [A]. Learned linear pooling finds a pair of weaker forecasters that matches or beats the strongest individual in 11 of 16 ForecastBench groups. Grade: DS (aggregation axis).

Also on this axis: VPP particle count (A13), BLF trial averaging (A28), PDDL BoN then iVML (A7), resampling until coherent (A19).

Negative (where more thinking hurts):
**B14. Lacombe, Wu, Dilworth, "Don't Think Twice! Over-Reasoning Impairs Confidence Calibration", 2025-08-20.** https://arxiv.org/abs/2508.15050 [F]. Cites SAI: no. Task: predict the confidence level experts assigned (IPCC statements, IARC carcinogen groups). Overconfidence grows with thinking budget: Gemini 2.5 Flash goes from +6% to +21.3% on IPCC (0 to 384 tokens) and from +15.6% to +35.6% on IARC (to 320 tokens). With search, accuracy reaches 89.3%, vs 48.7% for the best pure reasoner. Grade: DS (narrow task, small budgets).

**B15. Zhao, Hooi, Ng, "Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet", 2025-09-08.** https://arxiv.org/abs/2509.06861 [A]. Fourteen reasoning models on closed-book knowledge tasks: more test-time compute does not reliably raise accuracy and often raises hallucination, because the model attempts more answers; extended reasoning shows confirmation-bias patterns. Argument: compute-only scaling of a fixed model cannot add information about the ground truth. Grade: DS.

**B16. Nel, "KalshiBench", 2025-12-17.** https://arxiv.org/abs/2512.16030 [F]. 300 Kalshi questions; all five frontier models are overconfident; Claude Opus 4.5 ECE 0.120 vs GPT-5.2-XHigh 0.395 at similar accuracy; only one model has positive Brier skill. A cross-model comparison, not an effort sweep.

**B17. Gema, Hägele, Chen, Arditi et al., "Inverse Scaling in Test-Time Compute", 2025-07-19.** https://arxiv.org/abs/2507.14417 [F]. Tasks where longer reasoning lowers accuracy. On a student-grades regression (an estimation task) in the zero-shot controlled setting, RMSE worsens with reasoning length for every model (more for Claude Opus 4 and DeepSeek R1 than o3), as models move from reasonable priors to spurious features. Grade: DS.

**B18.** Zhou, Ling, Chen, Wang, Fan, Wang, "When More Thinking Hurts", 2026-04-12, https://arxiv.org/abs/2604.10739 [A]: diminishing returns, and extended reasoning abandons correct answers. Hiremath and Hiremath, "Calibration Drift Under Reasoning", 2026, https://arxiv.org/abs/2606.11211 [F]: non-monotone ECE in budget for Llama-3.1-8B on 47 trap questions; 70B inconclusive (weak evidence).

### (b3) Amortized posterior sampling over reasoning (the GFlowNet line)

**B19. Hu, Jain, Elmoznino, Kaddar, Lajoie, Bengio, Whitammer, "Amortizing intractable inference in large language models", 2023-10-06, ICLR 2024.** https://arxiv.org/abs/2310.04363 [F]. Cites SAI: no (predates; Mila, not LawZero).
- Recipe: fine-tune the LLM as a GFlowNet sampler of chains of thought Z with reward p_LM(XZY), i.e. sampling p(Z|X,Y). At test time, sample several Z and answers and take a majority vote (Bayesian model averaging; 10 rationales for subjectivity classification).
- Shown: GPT-2 and GPT-J 6B; +10.9% over SFT on subjectivity with 10 labels; +63% over SFT and PPO on integer arithmetic with 50 demonstrations; better out-of-distribution.
- Missing: no curve of estimate quality vs number of samples; small tasks. Grade: TOY.

**B20. Follow-ups (none in the S2 SAI citer list; all report accuracy or diversity, not calibrated estimates).** Bartoldson, Venkatraman, Diffenderfer, Jain et al., "Trajectory Balance with Asynchrony", 2025, https://arxiv.org/abs/2503.18929 [A]: off-policy TB post-training from Pythia-410M to Qwen2.5-7B, at least 4x speedups, gains as data generation scales. Zhu, Cheng, Zhang et al., "FlowRL", 2025, https://arxiv.org/abs/2509.15207 [F]: reward-distribution matching with a learned partition function on Qwen2.5-7B/32B, +10.0% over GRPO and +5.1% over PPO on math. Zou, Huang, Li, Zhou, "Latent Thought Flow", 2026, https://arxiv.org/abs/2606.16222 [A]: continuous-latent GFlowNet reasoning, +9.5% accuracy and 27.2% shorter reasoning than latent baselines. Grade: TOY to DS for accuracy; nothing on estimates.

### (b4) Where forecasting stands
**B21.** Karger, Bastani, Chen, Jacobs, Halawi, Zhang, Tetlock, "ForecastBench", 2024, https://arxiv.org/abs/2409.19839 [F]: superforecaster median Brier 0.096 vs best LLM 0.122 (Claude 3.5 Sonnet) in the paper. FRI update (Bastani, Kučinskas, Karger, 2025-10-08), https://forecastingresearch.substack.com/p/ai-llm-forecasting-model-forecastbench-benchmark [fetched]: best LLM GPT-4.5 at 0.101 vs superforecasters at 0.081 (difficulty-adjusted); LLMs improve about 0.016 Brier per year; projected parity November 2026 (95% CI December 2025 to January 2028). The post makes no claim about reasoning compute.

## 3. Distance to a recipe

(a) A consistent world model expressed or queryable in words.
- An integrated recipe (one trained model whose answers to any question in words are coherent probabilities under one world model) is not implemented in any citing or adjacent work found. No non-LawZero citer of SAI implements its world model; technical citers address pieces (A6: reward hacking when theories are scored by likelihood; A7: test-time search for symbolic world models; A20: consistency training). Grade: SPEC (only in the primary SAI papers; sibling scope).
- An LLM writes an explicit probabilistic model and inference answers (A1, A2, A4, A5, A8, A9): TOY. Coherence holds inside each synthesized model; nothing ties models across queries.
- Distilling those models into weights (A3; A18 for a single model family): TOY at 7-9B, with transfer and calibration gains and human correlation 0.775 (A3). The most recipe-like path: synthesize models, run probabilistic inference, train on the soft posteriors. Not shown: scale, cross-query coherence, open-domain knowledge.
- Training credences to be coherent directly (A16, A17, A19, A20; consistency as a by-product of forecasting RL in B2): TOY at 8B or below. Enforcing consistency at inference fixed only the targeted checks and did not raise accuracy (Paleka et al.); a coherence penalty from de Finetti is specified but untested (A22).
- Posterior over hypotheses written in words (A13, A14, A15): TOY; works on synthetic and narrow tasks with black-box LLMs.
- Text world models with an explicit latent state (A24-A28): TOY for inspectable belief states (A24 at 32B or below on text games; A28 with frontier models but untrained); DS only for deterministic environment simulation without calibrated beliefs (A26, 397B).
- PDGs applied to LLM credences: not found.

(b) A scientist reasoner whose estimates improve with more thinking.
- More independent samples, trials or particles, then aggregation: DS in forecasting and QA (B10, B11, B13, A28) and TOY for posteriors over hypotheses in words (A13). This is the most consistently positive compute axis found, with saturation (B10: Brier improves on 3 of 4 datasets; A13: monotone on one task, saturated or dipping on others).
- A longer single chain: mixed. DS positive in agentic forecasting with search (B9, plateau after "high") and within a natural CoT (B8). DS negative on closed-book knowledge and expert-confidence tasks (B14, B15; B16 is cross-model only) and on an estimation task with spurious features (B17). Forcing extra tokens did not help calibration (B8). B15 argues that compute without new information cannot add information about the answer.
- Training for calibrated reasoning: TOY to DS (RLCR at 7B; forecasting RL at 8B-120B: B2, B3, B4). No paper found trains for the property that estimates improve with budget (an anytime or martingale property over reasoning length). A19 trains a martingale-style loss over resampled answers, not reasoning length; B6 rewards confidence building up along a trace, not calibration across budgets.
- Amortized posterior over reasoning (GFlowNets): TOY for estimates; 2025-2026 follow-ups at 7-32B report accuracy, not calibrated probabilities (B19, B20).
- Net: two halves are demonstrated separately. RL with a proper scoring rule gives calibrated forecasters at 8B-120B (B2-B4); retrieval, an ensemble of independent runs, a reconciling supervisor and Platt scaling on frontier models are claimed to reach superforecaster Brier (B11), and FRI's trend puts LLM parity around late 2026 (B21). No source combines them in one trained system. A recipe in which longer thinking without new evidence reliably improves estimates is not demonstrated: the evidence shows a plateau with search (B9) and reversals in closed-book settings (B14, B15, B17).

## 4. Not verified or caveats
- About 37 Google Scholar citers of 2502.15657 were unchecked (CAPTCHA). [A] entries rest on abstracts; their SAI-citation status comes from the S2 list only.
- Venues (ICLR 2024, ICLR 2026) are taken from the paper or other papers' reference lists, not venue sites.
- Seen only in search snippets, not fetched, so not used: latent visual CoT GFlowNet (arXiv 2510.23925), ThoughtTracing (2502.11881), AutoToM (2502.15676), Prophet Arena (2510.17638), and a reported dispute over AIA Forecaster's parity claim.
- Cross-model comparisons (B16) are not effort sweeps; B9's effort sweep is one model with search confounded.
