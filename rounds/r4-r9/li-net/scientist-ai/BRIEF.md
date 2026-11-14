# Scientist AI: distance to a training recipe (eurisko, 2026-10-03 17:20Z)

Kelvin's question: on the latest Scientist AI papers (Bengio, LawZero) and works citing them, how close are they to
(a) a training recipe for a consistent world model expressible or queryable in words, and (b) a recipe for a
"scientist reasoner" whose estimates improve with more thinking?

Evidence: SOURCES-PRIMARY.md (Bengio/LawZero corpus 2024-2026, 17 sources fetched) and SOURCES-CITING.md
(90 Semantic-Scholar citers of arXiv 2502.15657 screened, 40+ adjacent papers fetched). Grades: DS = shown at
scale, TOY = toy or narrow, SPEC = specified only, POS = position only. Every number below is in those files
with its URL.

## (a) Consistent world model in words: SPEC, nothing trained

- The object is now precisely specified: Bengio, Richardson et al., "Safety from Honesty in a Disinterested AI
  Predictor", arXiv 2606.29657 (June 2026, 41 pp, no experiments). Variables are Boolean statements named by
  natural-language sentences (incl. maths and code); causal models are programs with a short-description prior;
  the Predictor Q approximates the posterior P_n over truth assignments given a contextualized dataset; queries
  are Boolean combinations of statements incl. do()-interventions. Training is constrained, not given: Req. 3.34
  (a loss whose unique minimiser is P_n) and Req. 3.36 (no consequence-dependent signal). Their own gaps:
  contextualization "specified solely at the level of requirements", no concrete loss, no Explainer training
  procedure, "Q will never exactly represent a consistent probability distribution", semantic anchoring is an
  untested "empirical adequacy premise", not initialised from a pretrained LLM.
- Nothing has been trained. Bengio, May 2026: "we do need to try it out"; Sept 2026: help LawZero "demonstrate
  that such designs are achievable". Stated next steps: release the contextualization pipeline; train sub-10B
  models from scratch with the objective against matched baselines; fine-tune an open-weight guardrail.
- The implemented pieces are small and not in language: amortized posteriors over DAGs of at most 20 variables
  (Deleu et al.), exact Bayes over 1,024 bandit theories (2408.05284), LIR on small discrete PDGs (2604.17140),
  GFlowNet sampling of latent text at up to 6B (2310.04363). Consistency verification of a (P, Q) predictor pair
  is complexity theory only (2608.11181).
- Outside LawZero nobody implements the world model; no citer applies PDGs to LLMs. The most recipe-like path
  found: an LLM writes probabilistic programs, inference yields posteriors, a 7-8B model is fine-tuned on the
  soft posteriors (Zhang, Jagadish, Lake, Griffiths 2026, arXiv 2606.09856; human correlation 0.509 to 0.775;
  no cross-query coherence check). Coherence training of credences (deductive closure training, LoCo-LMs, the
  c.i.d./martingale loss of Kim et al. 2606.17832, structure-aware training of Xu et al. 2601.05905) is all at
  8B or below and per question or per neighbourhood; Paleka et al. (2412.18544) found that enforcing
  consistency at inference fixes only the targeted checks and does not raise accuracy. Textual belief-state world
  models (2026) are TOY at 32B or below.
- Net: specification complete, recipe absent; components exist in toy form; nothing ties answers across queries
  under one model. Their own plan puts the first from-scratch trial in the next year.

## (b) Reasoner whose estimates improve with thinking: SPEC for LawZero; mixed evidence elsewhere

- LawZero's mechanism (2502.15657 §3.5.5; 2606.29657 Def. 4.3): "System 2" = an Explainer proposes a small set
  of latent statements, Q evaluates them, the estimate is a Monte-Carlo average. Their accuracy claims are
  asymptotic in training compute; the residual "computational uncertainty" is "controlled by compute" with its
  quantification "left to future work". No source in the corpus plots estimate quality against query-time
  compute. Note the mechanism is the sampling axis (average over sampled explanations), not a longer single chain.
- Demonstrated elsewhere, by axis:
  - More independent samples then aggregation improves estimates reliably, with saturation: sample consistency
    (2402.13904), AIA Forecaster (2511.07678, claimed superforecaster parity, disputed), aggregating weaker
    forecasters (2609.33257). DS.
  - A longer single chain is not reliably monotone. Positive: reasoning models are better calibrated in 33 of 36
    settings and calibration improves as the chain unfolds, but forced extra thinking does not help (Yoon et al.
    2505.14489); FutureSim (2605.15188) shows GPT-5.5 Brier skill rising with reasoning effort then plateauing,
    search confounded. Negative: over-reasoning raises overconfidence on expert-confidence tasks (2508.15050),
    test-time scaling does not help closed-book knowledge and often raises hallucination (2509.06861), inverse
    scaling on an estimation task (2507.14417), non-monotone calibration drift (2606.11211).
  - Training calibrated reasoners by RL with proper scoring rules works: RLCR at 7B (2507.16806; ECE 0.37 to
    0.03 on HotpotQA), outcome-based forecasting RL at 8-14B (2505.17989, 2512.25070), Turtel et al. 2026 on
    GPT-OSS-120B (2608.28482; Brier-trained Brier 0.1648; output length varies 7x across scoring rules with
    small accuracy differences: more tokens did not buy accuracy).
  - No paper trains for the anytime property (estimates that improve with budget). Nearest: the c.i.d. loss
    over resampled answers (2606.17832, 8B, MCQA); GFlowNet chain-of-thought follow-ups (FlowRL, Latent Thought
    Flow) report accuracy, not calibrated estimates.
- Net: the ensemble axis of query compute works and is in production forecasters; the single-chain axis has no
  training recipe and mixed empirics; LawZero's System 2 is specified as the ensemble axis and has no curve.

## Caveats
About 37 Google-Scholar-only citers were not checked (CAPTCHA); entries marked [A] in SOURCES-CITING.md rest on
abstracts; Bengio's talks and the 2025 red-teaming paper were not fetched.
