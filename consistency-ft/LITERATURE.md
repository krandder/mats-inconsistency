# Literature digest (research sweep 2026-08-19)

## The anchor paper: Paleka et al., "Consistency Checks for Language Model Forecasters"
arXiv:2412.18544 (ICML 2025). Our v0 evaluation harness.

- **10 checks**: Negation (F(P)+F(¬P)=1), Paraphrase, Consequence (F(P)≤F(Q)),
  AndOr (F(P)+F(Q)=F(P∨Q)+F(P∧Q)), And/Or Fréchet bounds, But
  (F(P∨Q)=F(P)+F(¬P∧Q)), Cond (F(P)·F(Q|P)=F(P∧Q)), CondCond, ExpEvidence
  (total probability).
- **Arbitrage metric**: prices as log-market-maker quotes with $1 subsidy;
  violation = max-min guaranteed arbitrage profit over outcomes consistent
  with the logical relation. Closed forms for Negation/Paraphrase/Cond; ODE
  numerically otherwise. Threshold V ≥ 0.01 ≈ "110% on two parties".
- **Questions**: scraped (Metaculus/Manifold), NewsAPI-derived (Perplexity
  resolution, <5% error), synthetic (2028-resolving 3,000-tuple benchmark).
  Tuples composed and verified by GPT-4o → noisy constraint labels.
- **ArbitrageForecaster** (inference-time repair): improves only the arbitraged
  checks; no held-out-check or Brier gains. The null hypothesis we must beat.
- **Their stated future work**: training on consistency metrics,
  AlphaZero-style self-distillation from recursive arbitrage, adversarial
  check generation vs Goodharting, non-binary questions.

Limitations → our expansions: eval-only; local tuples not global belief
graphs; LLM-verified constraints; forecasting/binary-only; single verbalized
elicitation.

## Direct competitor: RLE ("Rethinking LLM Confidence: From Calibration to Coherence")
Matta, Naphade, Zou — ICML 2026 CTB workshop poster (OpenReview SmwCItfHoC,
bot-gated). arXiv:2607.19367 is the earlier eval-only draft ("Rethinking
Uncertainty Evaluation..."); full RLE formulation in Matta's CMU thesis
"Coherent Confidence in LLMs" (May 2026, reports-archive.adm.cs.cmu.edu).

- Reward per episode of n related prompts with logical relation set S:
  `R(c, e, S) = −Exploit(c, S) − λ·(1/n)Σ(cᵢ−eᵢ)²`
- Exploit = LP value: adversary stakes αᵢ∈[−1,1], worst case over consistent
  outcomes of Σαᵢ(cᵢ−ωᵢ); ≥0, =0 iff coherent. All templates (normalization,
  conjunction, entailment, paraphrase) are constraint sets in the same LP.
- Brier term anchors to realized correctness — prevents trivial coherence,
  but restricts to resolvable questions. **Our lane: KL-to-previous-beliefs
  anchor instead.**
- Trainable adapter on frozen answer model, confidence turns only. Workshop
  claims: beats Brier-only finetuning on structural coherence in- and OOD.
  Eval side: calibration ⊥ coherence; models put lower confidence on strictly
  easier questions 31% of the time; RLHF collapses AUROC.
- No public code (krishmatta/consistency = empty placeholder, 2026-07-29).

## Training on consistency (what exists)

- **LoCo-LMs** (2409.13724, ICLR 2025; code ddidacus/loco-llm): semantic-loss
  finetuning of LLaMA-2-7B on implication/negation constraints (BeliefBank);
  0.99 consistency, transfers zero-shot to EntailmentBank; data-efficient.
  Factuality supervision on antecedents is what prevents collapse.
- **OpenForesight RL** (2512.25070): outcome-RL on synthesized forecasting
  questions; improved And/Consequence consistency, regressed Negation.
- **WikiLFC etc.** (2412.16100): KG-triplet queries composed with ¬,∧,∨ (no
  implication) + SFT; improves consistency. Nearest dataset to our
  by-construction plan; the full atom+¬+∧+→ cluster set is unbuilt.
- **ICM** (Anthropic, 2506.10139): consistency-only objectives collapse;
  their fix is mutual predictability, not KL. KL-anchored consistency
  training appears unpublished.
- **Semantic loss** (1711.11157; pseudo-semantic 2312.03905): the
  differentiable constraint-loss machinery.

## Degeneracy & coherentization theory

- Predd/Osherson/Kulkarni/Poor: Coherent Approximation Principle — project
  incoherent judgments onto nearest coherent set; local-coherence scaling.
- Karvetski et al. 2013: coherentization + coherence-weighting improves human
  forecast accuracy (consistency pressure launders noise).
- Revealed Rationality (2608.05015): label-free rationality penalties from
  representation theorems (de Finetti etc.).
- Collapse evidence: medical-VLM paraphrase-consistency loss → always-"Yes"
  (2603.00148); CCS's confidence term exists to kill the 0.5 solution.

## Elicitation readouts disagree

- Kadavath 2207.05221: base logprobs calibrated in-format; P(True)/P(IK) head
  works in-distribution, weak OOD.
- Tian 2305.14975: RLHF wrecks logprob calibration; verbalized better.
- Probes better-calibrated than verbalization (2607.08046: ECE 0.044 vs
  0.093); probes catch errors verbal confidence misses (2608.07528); probe
  beliefs can diverge from generative behavior (2410.02707) — the decoupling
  risk for any separate proposition head.

## Citation sweep of 2412.18544 (20 papers, mostly eval)

Notable: law-of-total-probability violations across partitions (2607.15277);
Bayesian Coherence Coefficient — larger models more Bayesian (2507.17951);
locally-coherent-globally-incoherent multi-agent compounding + projection
repair (2605.30335); OptimismBench signed bias (2607.26981); forecaster-eval
pitfalls / temporal leakage (2506.00723, Paleka team); consistency survey
(2505.00268). Adjacent harness code: lijuliana/coherencebench-v0.
