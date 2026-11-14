# Scientist AI: primary sources (Bengio group / LawZero), Feb 2024 to 2026-10-03

Scope: papers and technical posts by Bengio's group / LawZero that define or advance the Scientist AI (SAI) proposal, plus the building blocks those texts themselves cite. Every source below was fetched on 2026-10-03 (arXiv PDF + arXiv API metadata, lawzero.org, yoshuabengio.org, matsprogram.org, 80000hours.org) and the relevant sections were read. Quotes are verbatim from the fetched text (PDF-extraction spacing repaired). Anything not fetched is marked **unverified**.

Grades: **SCALE** = implemented and demonstrated at scale; **TOY** = implemented in toy or small experiments; **SPEC** = specified but not implemented; **POS** = position only.

## Index

| # | Date | Source | Type | Grade |
|---|---|---|---|---|
| P1 | 2024-02-26 | Bengio, "Towards a Cautious Scientist AI with Convergent Safety Bounds" | technical blog | POS |
| P2 | 2024-08-09 (UAI 2025) | Bengio et al., "Can a Bayesian Oracle Prevent Harm from an Agent?" arXiv 2408.05284, + blog 2024-08-29 | theory + toy simulation | TOY (exact Bayes, 1,024 theories) |
| P3 | 2025-02-21 | Bengio et al., "Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?" arXiv 2502.15657 | position / research plan | POS |
| P4 | 2025-06-03 | Bengio, "Introducing LawZero" | blog | POS |
| P5 | 2026-02-03 draft; 2026-03-06 final | Fornasiere*, Richardson*, Gendron, Serban, Bengio, "The Scientist AI: Safe by Design, by Not Desiring" (LawZero) | non-technical overview | POS |
| P6 | 2026-05-07 | Bengio on the 80,000 Hours podcast | interview (status, plans) | POS |
| P7 | 2026-06-28 (v2 07-10) | Bengio, Richardson, ... Ghosn, "Safety from Honesty in a Disinterested AI Predictor" arXiv 2606.29657 | formal theory | SPEC |
| P8 | 2026-06-23 to 07-14 | LawZero blog series, incl. "The Case for Scientist AI" (07-02) | blog | POS |
| P9 | 2026-09-11 | Bengio, "Why are AI agents lying, cheating and coordinating?" | blog (status) | POS |
| P10 | undated (Winter 2027 cohort) | "LawZero at MATS" stream page | research agenda | POS |
| C1 | 2026-04-18 (AISTATS 2026) | Richardson et al., "Local Inconsistency Resolution" arXiv 2604.17140 | method + small experiments | TOY |
| C2 | 2026-04-19 | Fornasiere, Bronzi et al., activation-perturbation recognition, arXiv 2604.17465 | empirical, 8B-32B LLMs | implemented; not a world-model or reasoner component |
| C3 | 2026-08-11 (v2 09-05) | Paradise, Richardson, Bengio, Goldwasser, "How to Verify Probabilistic Consistency of Predictive Models" arXiv 2608.11181 | complexity theory | SPEC |
| C4 | 2026-09-09 | Lu & Bengio, "AI Safety: Not Optional, Not Later" arXiv 2609.10630 | position | POS |
| B1 | 2023-10-06 (ICLR 2024) | Hu et al., "Amortizing intractable inference in large language models" arXiv 2310.04363 | method + experiments | TOY (LMs up to 6B) |
| B2 | 2022-02-28 / 2023-05-30 | Deleu et al., DAG-GFlowNet arXiv 2202.13903 (UAI 2022); JSP-GFN arXiv 2305.19366 (NeurIPS 2023) | method + experiments | TOY (DAGs up to 20 variables) |
| B3 | 2023-10-03 (ICLR 2024) | Falet et al., "Delta-AI" arXiv 2310.02423 | method | TOY (cited only by P10) |

## Core Scientist AI documents

### P1. Towards a Cautious Scientist AI with Convergent Safety Bounds (blog, 2024)
- **Citation:** Y. Bengio, yoshuabengio.org, 26 Feb 2024. https://yoshuabengio.org/en/blog/towards-cautious-scientist-ai-convergent-safety-bounds. It builds on "AI Scientists: Safe and Useful AI?" (7 May 2023, https://yoshuabengio.org/en/blog/ai-scientists-safe-and-useful-ai, fetched, before the window).
- **Proposal (quote):** "The path I am suggesting is based on learned amortized inference, in which we train a neural network to estimate the required conditional probabilities." Goal: "safety guarantees will converge to an exact upper bound on risk as the amount of available compute and the efficiency of our learning methods increase."
- **World model:** a Bayesian posterior P(H|D) over hypotheses; "Hypotheses could be represented as computer programs", with a description-length prior "(e.g. proportional to 2^-L)" and causal graphs favoured. On words, he proposes a regularizer so hypotheses "can be converted to natural language and back with as little error as possible"; harm is to be stated in natural language.
- **Inference and compute:** reject action a if some plausible H has P(H, harm | a, c, D) > threshold, using Monte-Carlo confidence intervals. At query time: "at run-time, we can continue training or at the very least estimate the error made by the neural network through a sampling process ... like taking the time to think harder". No curve.
- **Demonstrated:** nothing in the post. It points to causal-graph generation "as we have shown on a small scale already" (the link is arXiv 2305.19366, B2).
- **Missing, in his words:** "overcome the known inefficiency of Bayesian posterior inference"; missing modes are "the really serious danger"; "much remains to be done to achieve the central goal of efficient and reliable probabilistic inference over potentially harmful actions".

### P2. Can a Bayesian Oracle Prevent Harm from an Agent? (2024; UAI 2025)
- **Citation:** Y. Bengio, M. K. Cohen, N. Malkin, M. MacDermott, D. Fornasiere, P. Greiner, Y. Kaddar. arXiv:2408.05284 (v1 2024-08-09, v3 2025-06-16), https://arxiv.org/abs/2408.05284; code github.com/saifh-github/conservative-bayesian-public. Companion blog post: "Bounding the probability of harm from an AI to create a guardrail", 29 Aug 2024, https://yoshuabengio.org/en/blog/bounding-probability-harm-ai-create-guardrail
- **Objective (quote):** "Our main results involve searching for cautious but plausible hypotheses, obtained by a maximization that involves Bayesian posteriors over hypotheses." The paper proves harm-probability bounds for the i.i.d. and non-i.i.d. cases.
- **World model:** a set of theories (data-generating processes) with a Bayesian prior and posterior. In the experiment these are theories τ_v indexed by v ∈ {0,1}^10. Nothing is expressed in language.
- **Inference and compute:** exact Bayesian computation. The guardrail takes the maximum harm probability over the plausible set. Curves are over the thresholds α and C, not over compute.
- **Demonstrated:** an "exploding bandit" with 10 arms and feature dimension d=10; harm is defined as reward > E; episodes last at most 25 steps; 10,000 episodes. The Prop. 3.3 and 4.5 guardrails are compared with posterior-predictive and "cheating" (true-theory) guardrails (Figs. 1-2).
- **Missing (paper §6 and blog):** "Tractability of posterior estimation ... a plausible answer would rely on amortized inference"; efficient search for a cautious theory; partial specification of theories; "Safety specification in natural language. How is a safety specification provided in natural language ... turned into a random variable"; the effect of approximation error; moderate overcautiousness. The blog adds: "this does not require asymptotic data, but 'only' asymptotic compute."

### P3. Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? (2025)
- **Citation:** Y. Bengio, M. Cohen, D. Fornasiere, J. Ghosn, P. Greiner, M. MacDermott, S. Mindermann, A. Oberman, J. Richardson, O. Richardson, M.-A. Rondeau, P.-L. St-Charles, D. Williams-King. arXiv:2502.15657 (v1 2025-02-21, v2 2025-02-24), 58 pp., https://arxiv.org/abs/2502.15657. Affiliations: Mila, UdeM, Berkeley, Imperial, McGill.
- **Type:** position paper and research plan. "Without using any equations, this paper argues ..." (§1.3). It contains no experiments.
- **Architecture (quote):** "a world model that generates causal theories to explain a set of observations obtained from the world, and an inference machine that answers questions based on the theories generated by the world model. Both components are ideally Bayesian". Also: "Both the theories generated by the world model and the queries processed by the inference machine are expressed using logical statements, expressed either in natural language or using a formal language." And: "There is a unique correct probability (according to the world model) associated with any query, which is the result of globally optimizing a Bayesian training objective".
- **Objective:**
  - The world model is an amortized sampler of the posterior over theories (description-length prior × likelihood), trained with GFlowNet / amortized-VI objectives. "the global optimum of such energy-based objectives (i.e., when the GFlowNet training loss is exactly zero) corresponds to exactly achieving the desired conditional probability".
  - The inference machine is trained on synthetic data from the world model to approximate the posterior predictive.
  - Text is handled as "someone wrote X", with X treated as a latent cause (§3.6.4).
- **World model object:** a distribution over causal models whose variables are logical statements, in natural language or a formal language. It is queried as P(Y|X) on statement pairs: "we compute the probabilities of events instead of the probabilities of event descriptions."
- **Compute:**
  - The paper claims "Scientist AI becomes safer and more accurate with additional computing power" and "we can use as large a network as we can afford and we will always get improvements in performance". This is mainly about training compute.
  - At query time (§3.5.5): "we may add a variable-time component to the probability calculation of the run-time inference machine, akin to 'System 2' ... This time can be used to generate explanations, arguments, and proofs, as part of the run-time deliberation which, in turn, will improve the inference machine's predictions". This would use "a GFlowNet objective that seeks short explanations that effectively reduce the uncertainty in the predicted probabilities".
  - Residual error is called "computational uncertainty" and is to be covered by confidence intervals (§3.5.4).
  - An experiment generator using GFlowNets plus an information-gain estimator is proposed, with humans running the experiments (§3.8.1).
  - No curve.
- **Plan:** in the short term, "build a guardrail ... obtained by fine-tuning an existing frontier model for the generation of explanatory hypotheses". In the long term, "Training from scratch with the full Bayesian posterior objective".
- **Missing, in their words:**
  - The amortized methods "have so far only been explored on domain-specific theories whose description is short enough to be generated by a neural network much smaller than those of frontier AIs, and it remains to be shown how these methods can be scaled further" (§3.3.2).
  - "these algorithms have not yet been explored at the scale of current frontier AI" (§3.4.1).
  - The choice of language for theories "remains open".
  - Still to do: "learn to transform a sequence of words into a probability distribution over sequences of logical statements each expressed in natural language, and vice-versa" (§3.8.2).
  - Exploration (missing modes) has "unavoidable challenges".
  - Emergent agency needs "further research" (§3.7).

### P4. Introducing LawZero (blog, 2025-06-03)
- **Citation:** Y. Bengio, https://yoshuabengio.org/en/blog/introducing-lawzero
- **Recipe sentence (quote):** "Mathematically, this is to be implemented with structured and honest chains-of-thoughts seen as latent variables that can explain the observed facts, which include the things that people say or write, not taken as truths but as observations of their actions. The aim is to obtain a completely non-agentic, and memoryless and state-less AI that can provide Bayesian posterior probabilities for statements, given other statements."
- No results; it points to P3 as the "first text".

### P5. The Scientist AI: Safe by Design, by Not Desiring (LawZero, 2026)
- **Citation:** D. Fornasiere*, O. Richardson*, G. Gendron, I. Serban, Y. Bengio. Listed 05-02-2026 at https://lawzero.org/en/publication/scientist-ai-safe-design-not-desiring. There are two PDFs:
  - a draft dated February 3, 2026, labelled "LawZero preprint. Work in progress": https://lawzero.org/sites/default/files/publications/81/safetyargumentblogpost.pdf
  - an authored version dated March 6, 2026: https://lawzero.org/sites/default/files/publications/81/safetyargumentblogpost-1.pdf

  A word-level diff shows the two texts are substantively identical.
- **Type:** in their words, a "non-technical overview". No results.
- **Architecture (quote):** "the SAI consists of two interacting components: an estimator of arbitrary probabilities of interest (such as 'this apple falls onto me'), and a generator that explores hypotheses about the world (such as 'the apple is still green')."
  - Contextualization: "a transformation of the training data that makes the epistemic provenance of claims explicit by attributing sources. In doing so, the truth of the original statement becomes a latent variable to be inferred from patterns of evidence".
  - Consequence invariance: "the training signal must be myopic, concerned only with making an accurate prediction for the given query, given the data at hand."
  - The estimator is initialized at random.
- **World model:** implicit in the estimator. The generator, by "extracting relevant parts of the causal world, which were previously only implicit within the estimator", makes it explicit, and variable meanings are "anchored (via natural language) by reuse". The goal: "ultimately we aim to go further and equip the SAI with a full causal world model. Moreover, we believe this can be done with the ingredients already at hand".
- **Inference and compute:** the generator "produces reasoning ... allowing for complex arguments and the kind of 'thinking' exhibited by reasoning language models". Its affordances are gated by the estimator's approval (a prover-verifier framing). Nothing is said about compute versus quality.
  - On experiments, P5 says "the SAI neither performs experiments itself, nor does it desire 'new' understanding beyond what it can deduce from past data". Its footnote: given two equally consistent theories, the SAI "rests in that ambiguity and averages across both possibilities".
- **Missing:** "Ensuring accurate evaluation of these counterfactuals remains an open problem". Also: "The work described here is currently underway, and much remains to be done. We are proceeding incrementally, with empirical evaluation of each component at increasing scales."

### P6. Bengio interview, 80,000 Hours podcast (2026-05-07)
- **Citation:** R. Wiblin and Y. Bengio, "Yoshua Bengio thinks he knows how to build safe superintelligence", published May 7, 2026. https://80000hours.org/podcast/episodes/yoshua-bengio-scientist-ai/ (transcript fetched).
- **Plans (Bengio's words):**
  - "We can train really small models — of the kind that academic organisations have been training, like less than 10 billion weights or something — from scratch, but using the Scientist AI objective and the data representation". These would be compared head to head with same-size open-weight models.
  - The second track: "take an existing pretrained model ... and then fine-tune it using the Scientist AI objective and data representation ... but we lose the mathematical guarantee."
  - Short term: "put out what we call the 'contextualisation pipeline'" and "a smaller-scale guardrail or a guardrail obtained by fine-tuning an existing open weight model". The agentic version "might take more years than months".
  - "the training objective is different, we do need to try it out and see how it works."
  - Outputs are to include "a confidence interval around the probabilities".
- **Demonstrated:** no results are reported.

### P7. Safety from Honesty in a Disinterested AI Predictor (2026)
- **Citation:** Y. Bengio, O. Richardson, T. Gavenčiak, M. Cohen, R. Svarc, D. Fornasiere, G. Gendron, D. Hyland, A. Kamanda, A. Oberman, F. R. Ward, A. Gavenčiak, J. L. Slosser, V. Mai, I. Serban, J. Ghosn. arXiv:2606.29657 (v1 2026-06-28, v2 2026-07-10; the PDF is dated July 14, 2026), 41 pp. https://arxiv.org/abs/2606.29657. LawZero page dated 2026-07-02: https://lawzero.org/en/publication/safety-honesty-disinterested-ai-predictor
- **Type:** formal theory. It gives definitions, requirements and theorems, and labels informal "Claims/Assumptions" as such. No experiments.
- **Architecture and objective (quote):** "The SAI Predictor Q takes as input a query (x,y) about variable y in context x, and approximates a Bayesian posterior, P_n(·) = P_0(·|D_n), over joint truth-assignments to these statement-variables by conditioning a prior P_0 on a contextualized dataset D_n".
  - Req. 3.34: training optimizes "a non-negative objective J ... whose unique global minimizer is the Bayesian posterior P_n".
  - Req. 3.36: no consequence-dependent training signal. A sufficient construction is J = J_past(Q_past; D_n) + J_future(Q_future; sg(Q_past)), "(although we have not yet analyzed whether this is enough to formally satisfy Req. 5.4)".
  - The pipeline is Ingest (contextualization) → Predict (System 1 / System 2) → Deploy (scaffold + guardrail).
- **World model:**
  - The variables are Boolean statement-variables, "each named by a valid statement in natural language (including mathematics and code)".
  - Causal models W are programs that specify the structural causal mechanisms; the prior is biased toward short descriptions.
  - Queries are Boolean combinations of statements, including do()-interventions on deployed predictions. So the model can be queried in words by construction.
  - Caveat: "in practice, the predictor Q will never exactly represent a consistent probability distribution over that space."
- **Inference and compute:** "The Predictor can either return its estimate directly ('system 1') or indirectly, by Monte-Carlo averaging over a small set of sampled latent statements proposed by an Explainer ('system 2')."
  - The Explainer is a "learned amortized proposal model". "At run time, the Explainer can improve predictions of Q via Monte-Carlo averaging". Req. 4.6: when a query involves a communication act asserting x, the Explainer must include the truth variable for x as a latent.
  - The accuracy claims are "asymptotic in data and compute". The error δ_t is "controlled by compute", with "the quantification of this computational uncertainty left to future work".
  - No curve.
- **Demonstrated:** nothing empirical. The main theorem (Thm 5.26) bounds the probability that training yields a dangerous Predictor: ν_t{Q dangerous} ≤ C_bad · R_shell, under the stated requirements.
- **Missing, in their words:**
  - "contextualization is specified solely at the level of requirements (i.e., we do not provide a completed algorithm)".
  - For Claim 4.9 (LLM-level expressivity is retained): "the present paper does not empirically verify" it.
  - Assumptions 4.10 and 4.11 are "empirical adequacy premises".
  - "the SAI Predictor is not initialized from a pre-trained LLM", although experiments that fine-tune from an LLM "could serve as evidence".
  - Future work: "empirical validation of the metrics associated with contextualization and epistemic-caution; the empirical stress-testing of our semantic assumptions; obtaining bounds and falsification protocols of R_shell ...; the efficient approximation of the Bayesian posterior with calibrated uncertainty; and building safety cases for agentic Scaffolds".
  - Proposed tests: minimal-pair evaluations (factual versus communication-act syntax, paraphrase and translation invariance), inspection of Explainer samples, and SGLD probes of loss bands.

### P8. LawZero blog series (June-July 2026)
- **Citation:** https://lawzero.org/en/blog. The series:
  - "Trained to please: Sycophancy and the design of language models" (23-06-2026)
  - "Your AI sounds confident but that doesn't mean it's right." (25-06)
  - "Goals Without Authors: The Problem of Implicit Agency" (30-06)
  - "The Case for Scientist AI" (02-07, https://lawzero.org/en/blog/case-scientist-ai)
  - "Are you the customer, or the product?" (14-07)

  The fetched text shows no authors.
- **What it says:** only "The Case for Scientist AI" describes the system: "the SAI Predictor (we call it Q) takes a valid statement (y) in natural language as its input, and outputs a probability that the statement is true", and "we think of the Predictor as something more like an interactive, 'queryable' network of theories".
  - It describes the guardrail γQ: it compares the harm probability of releasing q against a no-release baseline and returns "noprediction" above a threshold. Its toxin example uses made-up figures (55%, 30%) to illustrate, not results.
  - The Explainer is deferred to "coming posts". Jailbreak resistance "requires more study".

### P9. Why are AI agents lying, cheating and coordinating? (blog, 2026-09-11)
- **Citation:** Y. Bengio, https://yoshuabengio.org/en/blog/why-are-ai-agents-lying-cheating-and-coordinating
- **Status (quote):** "I have argued, and presented theoretical evidence, that there are ways to design AIs, including the Scientist AI framework, that are honest and make coherent predictions untainted by goals of their own ... consider helping LawZero demonstrate that such designs are achievable."

### P10. LawZero at MATS, Winter 2027 (stream page, fetched 2026-10-03)
- **Citation:** https://www.matsprogram.org/stream/lawzero. Mentors: Bengio, Bronzi, Fornasiere, Richardson, Falet.
- **Agenda (quote):** "(2) Uncertainty estimation for partially trained models. We are studying ensembles, epistemic neural networks, calibration, conformal prediction techniques, and other methods in synthetic environments. We are especially interested in projects that use amortized inference methods (such as GFlowNets [2111.09266, 2310.04363, 2310.02423]) to approximate posteriors over latent variables, such as (i) sources or (ii) predictors behind an autoregressive model, such that predictive uncertainty can be estimated from learned distributions, as opposed to single-point estimates."
  - Direction (1), situational awareness, cites C2.
  - Richardson's bio says his inconsistency theory may "lead to powerful generalist AI systems that are safer because they fundamentally do not meaningfully have goals or desires."
- No results.

## Group papers on components the SAI texts name

### C1. Local Inconsistency Resolution (AISTATS 2026)
- **Citation:** O. E. Richardson, M. Samiei, M. Shakerinava, J. D. Viviano, A. El Kabid, A. Parviz, Y. Bengio. arXiv:2604.17140 (2026-04-18), AISTATS 2026, PMLR 300. https://arxiv.org/abs/2604.17140
- **Link to SAI:** the affiliations include LawZero, but the paper does not mention "Scientist AI". P7 grounds its training in "the removal of epistemic inconsistencies [Richardson, 2022, 2024]". P7's reference list identifies these as "Loss as the inconsistency of a probabilistic dependency graph" (arXiv 2202.11862) and Richardson's Cornell PhD thesis (2024). C1 builds on both.
- **Method (quote):** "iteratively focus on a subset of the model and resolve inconsistencies using the parameters under control". It recovers EM, belief propagation, adversarial training, GANs and GFlowNets as special cases.
- **World model:** a parametric probabilistic dependency graph (PDG) over discrete variables, with conditional distributions weighted by confidence and possibly inconsistent. Not natural language.
- **Compute:** computing a PDG's inconsistency "is NP-hard in general". LIR resolves tractable local pieces instead. The experiments track how much inconsistency is removed, and at what compute cost, over iterations (T=20 timesteps). These are training and inference iterations, not query-time reasoning.
- **Demonstrated:**
  - random small discrete PDGs (each variable takes 2-3 values);
  - modified GFlowNet TB and LPV losses on 4 HyperGrid environments (5 seeds × top-3 configurations). "All models covered all modes", and the modified losses "generally show faster convergence at the earlier stages of training" (Fig. 3: L1 distance to the true posterior over training iterations).
- **Missing (quote):** "Under what assumptions does LIR converge? Can we learn adaptive Refocus mechanisms ...?" Whether inconsistency is quasi-convex in conditional distributions "remains a key open question".

### C2. Language models recognize dropout and Gaussian noise applied to their activations (2026)
- **Citation:** D. Fornasiere*, M. Bronzi*, S. Kitts*, A. Palmas, Y. Bengio†, O. Richardson†. arXiv:2604.17465 (v1 2026-04-19, v2 2026-04-30); all authors at LawZero. LawZero lists it at 13-05-2026. https://arxiv.org/abs/2604.17465
- **Relevance:** not about the world model or reasoner. P7 cites it as evidence that systems may detect being "evaluated (or, possibly, even trained [Fornasiere et al., 2026])".
- **Demonstrated:** Llama-3.1-8B-Instruct, Olmo-3.1-32B-Instruct, Qwen3-14B and Qwen3-32B detect and localize activation perturbations, "often with perfect accuracy". Qwen3-32B's zero-shot identification of the perturbation type improves with perturbation strength.

### C3. How to Verify Probabilistic Consistency of Predictive Models (2026)
- **Citation:** O. Paradise, O. Richardson, Y. Bengio, S. Goldwasser. arXiv:2608.11181 (v1 2026-08-11, v2 2026-09-05), 66 pp., cs.CC. https://arxiv.org/abs/2608.11181
- **Framing (quote):** "The Scientist AI [...] project aims to produce an unbiased consistent Bayesian predictor ... This approach can only work if the guardrail is based on a predictor that reflects a consistent probability distribution about events in the world".
- **Object:** a predictive model (P, Q): a prediction circuit P(y|x) over Boolean variables and a confidence circuit Q. Inconsistency is an ℓ2 measure, a simplified version of PDG inconsistency. Claims may be "described by a sequence of natural language sentences".
- **Results (theory only):**
  - a consistency witness with support of size m+1;
  - explicit consistency is in NP, with certificates of length O(mn + log B);
  - an interactive PCP in which a polynomial-time verifier checks approximate consistency of (P, Q).
- **Not implemented:** "deploying the full proof system at scale remains, for now, out of reach". Implementing the general (black-box) construction is "well beyond the scope of an empirical pilot". The next step they name: "the eventual training of predictive models to prove their own consistency".

### C4. AI Safety: Not Optional, Not Later (2026)
- **Citation:** Q. Lu (CSIRO) and Y. Bengio (LawZero, UdeM, Mila). arXiv:2609.10630 (2026-09-09), 6 pp.
- **Content:** a position paper. It places "the Scientist AI Predictor" as a model-level supervisory component inside a system-level assurance architecture, and restates consequence-invariant training and "epistemic contextualisation". No new method and no results.

## Building blocks cited by the SAI texts

### B1. Amortizing intractable inference in large language models (ICLR 2024)
- **Citation:** E. J. Hu, M. Jain, E. Elmoznino, Y. Kaddar, G. Lajoie, Y. Bengio, N. Malkin. ICLR 2024; arXiv:2310.04363. arXiv v3 (2026-08-31) lists the last author as Esmeralda S. Whitammer; P7's reference list gives "Nikolay Malkin". https://arxiv.org/abs/2310.04363
- **Cited by:**
  - P3 §3.5.5, for query-time explanations: "fine-tuning LLMs using a GFlowNet objective to approximately sample from a posterior over a chain-of-thought seen as a latent variable";
  - P7, which compares the Explainer to chains of thought [Hu et al., 2024];
  - P10.
- **Method (quote):** "we interpret chain-of-thought reasoning as a latent variable modeling problem". GFlowNet fine-tuning samples the latent Z in proportion to p_LM(XZY).
- **Demonstrated:**
  - models: GPT-J 6B, GPT-2 XL and GPT-2 Large;
  - subjectivity classification (SUBJ) with 10 labelled examples: 71.4% with GFlowNet fine-tuning versus 64.3% with supervised fine-tuning;
  - integer arithmetic with tool use, including out-of-distribution cases.

  For SUBJ, it samples 10 rationales at test time and takes a majority vote. I found no curve of accuracy against the number of samples. Limitation: "our experiments use models up to 6B parameters".
- **World model:** none. It is a posterior over text latents of a pretrained LM.

### B2. Bayesian structure learning with GFlowNets (UAI 2022; NeurIPS 2023)
- **Citation:** T. Deleu, A. Góis, C. Emezue, M. Rankawat, S. Lacoste-Julien, S. Bauer, Y. Bengio, arXiv:2202.13903, UAI 2022. T. Deleu et al., "Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network", arXiv:2305.19366, NeurIPS 2023.
- **Cited by:** P3 §3.3.2 ("can be used to generate descriptions of causal models over data ... and to approximately sample them from the Bayesian posterior"); P7 ("Deleu et al. [2022] for an early implementation"); P1.
- **Theories:** Bayesian-network DAGs, plus their parameters in JSP-GFN. Not language.
- **Demonstrated:** comparison with the exact posterior is only "feasible for graphs with no more than 5 nodes" (d=5). JSP-GFN also runs on simulated data with d=20 and on real flow-cytometry data (Sachs, d=11).

### B3. Delta-AI (ICLR 2024)
- **Citation:** J.-P. Falet et al., "Delta-AI: Local objectives for amortized inference in sparse graphical models", arXiv:2310.02423 (ICLR 2024).
- Cited only by P10. It names a GFlowNet limitation: "they scale poorly in the length of the generative trajectory". I did not read its experiments in detail.

## Other LawZero-affiliated papers found that do not mention Scientist AI (not graded)
- M. Kim, J.-P. Falet, O. E. Richardson, ..., Y. Bengio, "Latent Veracity Inference for Identifying Errors in Stepwise Reasoning", arXiv:2505.11824. It adds a latent veracity variable to each reasoning step, with Veracity Search and Amortized Veracity Inference (Qwen3-4B and 8B fine-tuned), tested on ProntoQA, GSM8K and CommonsenseQA.
- O. Boussif, ..., D. Fornasiere, ..., Y. Bengio, E. S. Whitammer, "Bayesian Symbolic Regression with Entropic Reinforcement Learning" (ERRLESS), arXiv:2608.09617. It learns a posterior over symbolic expressions, tested on the Feynman benchmark.
- I text-searched other 2025-26 arXiv papers listing a LawZero affiliation: 2509.26626, 2512.21852, 2603.16928, 2606.11063, 2605.08496, 2607.05316, 2607.07918 and 2608.14795. None mentions "Scientist AI" except 2608.14795, which only cites P3.

## Unverified, or searched for and not found
- arXiv 2404.01019 is Khalifa et al., "Source-Aware Training Enables Knowledge Attribution in Language Models" (COLM 2024), not a Bengio or LawZero paper. A text search of P2, P3, P7, C1-C4 and B1 found no citation of it.
- No LawZero technical report or roadmap with milestones was found. The 2026-09-16 funding release (https://lawzero.org/en/news/lawzero-receives-commitment-300m-joint-funding-canada-and-germany) mentions "the next phase of LawZero's technical roadmap" without any technical content.
- No public contextualization pipeline, code, dataset or trained SAI Predictor was found (web searches on 2026-10-03).
- Not fetched (unverified):
  - Bengio's talks (Simons Institute; FAR.AI Singapore Alignment Workshop, April 2025);
  - S. Lee et al. 2025 (GFlowNet red-teaming, which P3 cites for short-term explanation sampling);
  - Richardson's 2024 PhD thesis (a copy is at li-net/oli-dissertation.pdf; not read here).

## Distance to a recipe

### (a) A consistent world model expressible or queryable in words: SPEC
The target and the requirements are specified, but nothing is implemented. The components that do exist are TOY, and none of them works in natural language.
- **The object is fully specified, the recipe is not.** P7 defines the target: the posterior P_n over truth-assignments of natural-language statement-variables, causal models as programs, and queries as Boolean combinations of statements. It also states the loss requirement (J must have the unique minimizer P_n) and a sufficient stop-gradient past/future split.
  - It gives no concrete loss, no procedure for training the Explainer, and no contextualization algorithm (contextualization is "specified solely at the level of requirements").
- **Nothing has been trained.** No fetched source reports training an SAI Predictor or a world model over natural-language statements at any scale.
  - Bengio, May 2026: "we do need to try it out". Sept 2026: "consider helping LawZero demonstrate that such designs are achievable".
  - P5: "currently underway".
- **The implemented pieces are small and not in words:**
  - amortized posteriors over DAG theories with at most 20 variables (B2);
  - exact Bayes over 1,024 bandit theories (P2);
  - inconsistency resolution on small discrete PDGs (C1).

  Verifying the consistency of (P, Q) predictors is theory only (C3). The one implemented piece that works on language, B1 (LMs up to 6B), samples latent text; it is not a consistent model of statements.
- **The authors' own gaps:**
  - scaling amortized posteriors over theories to frontier size "remains to be shown" (P3);
  - mapping words to distributions over logical statements is still to do (P3 §3.8.2);
  - Q "will never exactly represent a consistent probability distribution" (P7);
  - the semantic anchoring of factual latents is an untested "empirical adequacy premise" (P7).
- **Stated next steps (P6):** release the contextualization pipeline; train models under 10B from scratch with the SAI objective and compare them with matched baselines; fine-tune an open-weight model as a guardrail.

### (b) A scientist reasoner whose estimates improve with more thinking: SPEC
The mechanism is specified. That more compute gives better estimates is argued, not demonstrated (POS).
- **The mechanism is specified:**
  - P3 §3.5.5: a variable-time System-2 step that generates explanations under a GFlowNet objective for "short explanations that effectively reduce the uncertainty".
  - P7, Def. 4.3 and Rem. 4.4: an Explainer that Monte-Carlo-averages over sampled latent statements whose truth values Q evaluates.
  - P1: keep training or sample at query time, "like taking the time to think harder".
  - P5: generator "thinking" gated by the estimator.
- **The convergence claims are about training compute.** In P1, P2's blog, P3 and P7 the claim is that more training compute (network size, training time, synthetic samples) brings Q closer to P_n: "does not require asymptotic data, but 'only' asymptotic compute". P7 treats this as asymptotic and leaves quantifying computational uncertainty to future work. No source states a convergence rate.
- **No compute curve exists for an SAI-style system.** No fetched source plots estimate quality against query-time compute for such a system. The nearest data:
  - C1 Fig. 3: posterior L1 error against training iterations for GFlowNet losses on HyperGrid;
  - B1: a fixed 10 rationales with a majority vote.
- **The documents disagree on experiments.** P5 (2026) says the SAI "neither performs experiments itself, nor does it desire 'new' understanding beyond what it can deduce from past data". P3 (2025) proposed a GFlowNet experiment generator whose experiments humans would run (§3.8.1).
