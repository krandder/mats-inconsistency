# Coherence of learned world models, and buying it with compute: literature for ShipHunt round 2

*2026-10-11. Read-only literature research for Kelvin (MATS). Context: `shiphunt-worldmodel-spec-2026-10-11.md` (round 2 arms: generic joint world model, dreaming, dreaming with posterior targets) and `td-gvf-martingale-note-2026-10-10.md`.*

**How this was checked.** Every citation below was confirmed to exist (authors, venue, arXiv id) through web search on 2026-10-11, and the claims attributed to each paper come from its abstract, its listing pages, or summaries of it. **I did not read full texts.** So quoted numbers are the abstracts' numbers. Tags:
- **[preprint]**: not peer reviewed as far as I could tell.
- **[unverified]**: a claim or detail I could not confirm.
- **[ours]**: my own derivation or hypothesis, not a claim made by the cited paper.

Bibliographic details for a few classics (XLNet, AlphaZero, wake-sleep, PFNs, CCS, deep ensembles) come from memory and were not re-fetched. They are marked **[memory]**.

---

## 0. Deep insights (the 7 ideas that matter most)

**D1. With an overcomplete set of conditionals, incoherence is the default. Fitting each conditional well does not make them agree.**
- **Insight.** A masked or any-order network outputs far more conditionals than a single joint distribution has degrees of freedom. A per-conditional log loss never forces them to agree. They agree only at the infinite-data, infinite-capacity optimum, and a masked loss can be almost optimal while the joint is badly wrong.
- **Strongest sources:**
  - Heckerman et al. 2000 (dependency networks: pseudo-Gibbs on inconsistent conditionals has a stationary distribution that depends on the visiting order).
  - Young & You 2023 (inconsistency is present from BERT-base to UL2-20B).
  - Sana et al. 2026 [preprint]: incompatibility "persists across model sizes" and is "an intrinsic structural artifact of the pseudo-likelihood objective".
  - Cai & Shi 2026 [preprint], "mode blindness": masked-prediction excess risk can be exponentially small while whole modes of the joint get the wrong weight. Their fix is to put mask-schedule mass on low-visibility or full masks.
- **ShipHunt implication.** Expect the arm-2 world model to be incoherent, worst at small N. Measure it (section 1). Make sure the mask schedule includes the fully masked board, which is the board prior, so the prior is actually trained.

**D2. The amortization gap: a single forward pass pays a price that extra inference compute can recover, and the price grows off-distribution.**
- **Insight.** An amortized predictor (one network maps evidence to posterior) is worse than per-query inference with the same model. The gap grows from training to validation data. It is separate from the model's own error.
- **Strongest sources:**
  - Cremer, Li & Duvenaud 2018 (the gap splits into an approximation part and an amortization part; amortization is usually the larger, and it grows on held-out data).
  - Kim, Shah, Kontonis, Kakade & Chen 2025, ICML Outstanding Paper ("train for the worst, plan for the best"). Any-order training must learn some intractable conditionals. Choosing the decoding order adaptively at inference sidesteps them: Sudoku solve rate goes from under 7% to about 90% with no retraining.
- **ShipHunt implication [ours].** The forward direction, p(reading | board, location), is easy to learn. The inverse, p(board | evidence), is the hard one. A world model plus inference compute (inversion by importance sampling or MCMC) can give a better posterior than the same network's direct conditional query. The forecaster distilled from it inherits two things: the world model's error, which compute cannot fix, and the amortization gap, which compute can.

**D3. Compute cannot create information about the world, but it can create usable information. The ceiling is Bayes with a good prior on the same data, and real networks sit far below it.**
- **Insight.** By the data-processing inequality, no amount of compute adds Shannon information beyond the N episodes. Xu et al. 2020 (V-information) formalize the escape hatch: for computationally bounded learners, usable information can be created by computation. Fixed-data gains therefore come from closing the gap between the network and an ideal Bayesian learner on the same data, for example through regularization, ensembles (approximate Bayesian model averaging), augmentation by symmetry or ordering, and distillation.
- **How big the gains are:**
  - Kim, Kotha, Liang & Hashimoto 2025 (ICLR 2026): at fixed data and unlimited compute, the best recipe (heavy weight decay, epochs, more parameters, ensembles) reaches about 5.17 times the data efficiency of the baseline. The ensemble's benefit can be distilled into a model 8 times smaller that keeps about 83% of it.
  - Muennighoff et al. 2023: about 4 epochs of repeated data are nearly free; returns fall to zero by about 40 epochs.
  - Prabhudesai et al. 2025: masked-diffusion (any-order) models extract much more from repeated data than left-to-right models. Learning many orderings acts as augmentation.
- **The info-theoretic limit is real but loose.** No paper I found bounds how far a given architecture sits from the Bayes ceiling, so nobody can say in advance how much compute can buy. **The field has no answer here.**
- **ShipHunt implication [ours].** "Dreaming" cannot beat the world model it dreams from (see D6). Its value is amortization and regularization. Big gains need the world model itself to get better through ensembles, regularization, ordering augmentation, or inference-time inversion.

**D4. The truth is coherent, so projecting onto coherence is a free improvement in a precise sense. But coherence is necessary, not sufficient, and treating it as the target invites Goodhart.**
- **Why it is free:**
  - de Finetti's dominance theorem, as proved by Predd et al. 2009: if a set of forecasts is incoherent, some coherent set scores strictly better under any strictly proper additive score, whatever the outcome.
  - Mohri, Schneider & Wu 2025 [preprint]: a Bregman projection onto the "coherent" set gives a monotone improvement (generalized Pythagoras), with no labels needed.
  - Kong et al. 2026 [preprint]: label-free strict improvement is possible exactly when two predictors are not "mutually calibrated".
- **Why it is not sufficient:**
  - Paleka et al. 2025: ArbitrageForecaster made forecasts more consistent but did not reliably improve Brier score.
  - Dragonevskiy 2026 [preprint, single author]: training against a Dutch-book adversary without a calibration anchor collapsed to uninformative forecasts in half the runs. The author also reports a "scaling illusion": near-zero incoherence alongside unjustified confidence.
  - Farquhar et al. 2023: arbitrary features satisfy CCS's consistency structure.
  - Simplest counterexample: the prior is perfectly coherent.
- **The correlation is real but cross-model.** Across models, consistency correlates strongly with accuracy:
  - Paleka et al.: R up to 0.92 for the conditional-probability check.
  - Andrews & Sarkar 2026 [preprint]: Spearman 0.91 between Dutch-book profit and Brier score across 15 LLMs, and the profit metric separates models better than Brier score does.
  - Whether pushing coherence within one model improves its accuracy is open, apart from the projection guarantee. **The field has no general answer.**
- **ShipHunt implication.**
  - Use coherence projections at inference time (safe by the dominance and projection theorems, when computed in the right geometry).
  - Use coherence losses during training only as regularizers anchored by the outcome loss.
  - Test the within-model correlation directly. ShipHunt can do this because it has an exact oracle for scoring.

**D5. The martingale and exchangeability view: Bayesian updating means a coherent sequence of predictions, and that gives a label-free check and regularizer from the model's own imagined futures.**
- **Insight.** A Bayesian predictor's forecasts form a martingale: today's forecast equals the expected forecast after the next observation, taken under its own predictive. On exchangeable data, its predictions do not depend on the order of observations. Fong, Holmes & Walker 2023 rebuild Bayesian inference from this predictive-coherence condition ("martingale posteriors").
- **Evidence that learned predictors violate it:**
  - Falck, Wang & Holmes 2024: LLMs roughly satisfy the martingale property over short horizons and drift over long ones, so in-context learning is not Bayesian.
  - Nagler & Rügamer 2025: TabPFN fails it too.
  - Chlon et al. 2025 [preprint, contested]: positional encodings break exchangeability, and the violations of order log(n)/n may still be "Bayesian in expectation over orderings".
  - Kim, Lee, Fong, Lee & Lee 2026 (venue reported as ICML 2026 [unverified]): beliefs drift under predictive resampling and then stabilize. A self-consistency fine-tuning loss removes the early drift without losing accuracy.
- **ShipHunt implication.** Exact posterior marginals are a martingale across probe stages. Under T1 they are also invariant to probe order. Both properties can be measured and trained without labels, which links to the TD and martingale arms in the 10-10 note.

**D6. Search as policy improvement (compute in, better targets out, distill, repeat) only helps when the improvement operator is guaranteed not to make things worse, and when real data stays in the loop.**
- **Insight.** Expert iteration and AlphaZero turn inference compute (tree search) into better training targets, then distill them into the fast network. Gumbel AlphaZero (Danihelka et al. 2022) shows a search operator that provably improves the policy even with very few simulations.
- **Probabilistic analogues:**
  - Markovian score climbing (Naesseth, Lindsten & Blei 2020): MCMC samples train an amortized posterior toward the inclusive KL with vanishing bias.
  - Reweighted wake-sleep: the sleep phase trains the inference network on dreamed (latent, data) pairs from the model.
  - Ensemble distillation (Allen-Zhu & Li 2023; Kim et al. 2025).
- **The key catch.** The operator improves only relative to the model's own joint. Iterating on self-generated data alone collapses (Shumailov et al. 2024, Nature). Keeping the real data and adding synthetic data on top keeps error bounded (Gerstgrasser et al. 2024).
- **ShipHunt implication [ours, important for arms 3 and 4].**
  - The arm-3 "dream" target (the dreamed board's 0/1 cells) is the classic sleep-phase / neural-posterior-estimation objective. Its minimizer is exactly the world model's own posterior p_θ(cell | evidence).
  - The arm-4 target (the world model's posterior) has the same fixed point. It is only a variance-reduced (Rao-Blackwellized) version.
  - So arm 4 adds no information beyond arm 3, and Kelvin's pending ruling on it matters less than it seems.
  - A real "compute buys better targets" arm must use a better operator than the direct query: inversion, MCMC, or an ensemble (arm R2-C).

**D7. Coherence can elicit what a model already knows, but cannot create what it doesn't.**
- **Insight.** Wen et al. 2025 (ICM; ICLR 2026) fine-tune on labels chosen to be mutually predictable and logically consistent under the model, with no external labels. They match training on gold labels for GSM8K verification, TruthfulQA and Alpaca reward modeling.
- **Theory.** Qiu, Ismail, He & Feng 2026 [preprint] show that such methods are coherence optimization, which equals description-length regularization with the pretrained model as the prior. They argue this is optimal for semi-supervised learning.
- **Limit.** Both papers say the target concept must already be represented in the model.
- **ShipHunt implication [ours].** A world model trained on N episodes may contain structure, for example a good forward noise model, that its direct posterior query and the forecaster do not use. Coherence between the world model's heads, and between world model and forecaster, is the channel for eliciting it. That is a different claim from "coherence improves accuracy" in general.

**Where the field has no answer (as of this search):**
1. Whether reducing incoherence within a model improves its accuracy, beyond exact Bregman projections. The empirical results are mixed (D4).
2. Scaling laws for incoherence as a function of data N and capacity. Sana et al. say it persists with scale, but I found no N-sweep for learned world models.
3. Which coherent completion to pick. Projection geometry matters. Torroba Hennigen & Kim 2023 found that the joint "closest to the MLM's conditionals" works best, but only for two tokens.
4. When MCMC on inconsistent conditionals helps and when it amplifies errors. Li et al. 2024 warn it "may amplify errors"; CoCoNet (Huang et al. 2017) and Large Language Gibbs (Choi et al. 2026 [preprint]) find it helps in practice. There is no theory that separates the two cases.
5. Whether a model's self-generated ("dreamed") data can improve on the generator without an external verifier or real data. The big LLM synthetic-data gains used a stronger outside generator: EntiGraph used GPT-4. Self-play from zero data (Cowsik et al. 2026 [preprint]) is a prior over computable structure, not data about the world.

---

## 1. How coherence is defined and measured: concrete metrics for ShipHunt

**Definitions in the literature**
- **Compatibility of conditionals (Arnold & Press 1989).** A family of conditionals is compatible if some joint distribution has them as its conditionals. For discrete variables this reduces to ratio and cycle conditions.
- **Order-consistency of AO-ARMs** (Uria et al. 2014; Shih et al. 2022). Every factorization order should give the same p(x).
- **Kolmogorov coherence of probabilities.** Equivalently there is no Dutch book (de Finetti; Paleka et al. 2025; Andrews & Sarkar 2026).
- **Martingale / c.i.d. coherence of predictive sequences** (Fong et al. 2023; Falck et al. 2024).
- **Behavioral consistency of a world model** (Vafa et al. 2024). Myhill–Nerode "compression": sequences that lead to the same state should get the same continuations. "Distinction": sequences that lead to different states should get different continuations.
- **Invariance under task-preserving transformations** (Mohri et al. 2025 [preprint]).

**Label-free metrics usable on ShipHunt.** All need only the model, so all carry over to real forecasting. Validate each against the exact posterior, which is used for scoring only.

| # | Metric | How to compute on ShipHunt | Zero for the true model? | Source |
|---|---|---|---|---|
| M1 | **Order spread of joint log-likelihood** | For held-out episodes x (board + probes), compute log p_σ(x) under K random factorization orders σ. Report the standard deviation across σ, and also log-mean-exp minus mean (the order-ensemble gain). | Yes | Uria et al. 2014; Shih et al. 2022. If order ensembles beat single orders, the orders disagree. |
| M2 | **Pairwise compatibility (log-odds-ratio symmetry)** | For cells i, j and context r: Δ_ij = logOR(x_i; x_j given r), computed from p(x_i \| x_j, r), minus the same log-odds ratio computed from p(x_j \| x_i, r). For a coherent joint, Δ_ij = 0. This is Besag/Brook's-lemma symmetry, and a "rectangle test". It also works for (cell, reading) pairs. | Yes | Arnold & Press 1989; Sana et al. 2026 rectangle test [preprint] |
| M3 | **Bayes-symmetry gap** | For evidence e and boards b drawn from the model's posterior: Var_b[log p(b) + log p(e \| b) − log p(b \| e)]. Each term comes from a different factorization, for example board-first versus evidence-first. Zero variance means Bayes' rule holds inside the model. | Yes | Schmitt et al. 2024 (as a loss) |
| M4 | **Direct-query versus sampled gap** | KL between the direct single-cell query p(x_i \| e) and the frequency of x_i in S ancestral or Gibbs board samples given e. Also check whether Gibbs results depend on the visiting order. | Yes, up to Monte Carlo error | Heckerman et al. 2000; Goyal et al. 2022; Li et al. 2024 |
| M5 | **Martingale gap and exchangeability** | (a) f_t versus E[f_{t+1}], where the expectation is over the model's own predicted next probe (location and reading). (b) Under T1, spread of f over permutations of the same probe set. | (a) Yes [ours: holds for any probe policy, as long as the expectation also averages over the model's prediction of the next location]. (b) Yes when probes are exchangeable given the board, which T1 satisfies. | Fong et al. 2023; Falck et al. 2024; Kim et al. 2026 |
| M6 | **Dutch-book profit on derived events** | Elicit the model's probabilities for cells, complements, pairwise AND/OR, row counts and conditionals. Solve the arbitrage LP for the maximum guaranteed profit. ShipHunt also has invariants such as "exactly 7 ship cells", so the marginals must sum to 7. That one is domain-specific, so treat it as a diagnostic, not a training rule. | Yes | Paleka et al. 2025; Andrews & Sarkar 2026 [preprint] |
| M7 | **Symmetry equivariance gap** | Spread of the predictions across the 8 board symmetries, mapped back to the original frame. | Yes | Mohri et al. 2025 [preprint] |
| M8 | **Mutual calibration between world model and forecaster** | Check whether some joint distribution makes both the forecaster and the world-model posterior calibrated. If not, a label-free projection strictly improves the primary predictor. | n/a | Kong et al. 2026 [preprint] |

**Validation protocol.** Across checkpoints, seeds and N, report the Spearman correlation of each metric with single-cell KL to the exact posterior. Report it within a model family as well as across families: D4 says the cross-model correlation is known and the within-model one is not.

---

## 2. Known results on incoherence

**Masked models and dependency networks**
- **Heckerman et al. 2000, dependency networks** (JMLR 1:49–75). They allow inconsistent local conditionals and sample with an "ordered pseudo-Gibbs sampler", which has a unique stationary joint, but that joint (a) can disagree with the conditionals that produced it and (b) depends on the visiting order. Conditionals learned from the same data should be "almost consistent". https://www.jmlr.org/papers/volume1/heckerman00a/html/
  - Lowd 2012 makes the same points. https://arxiv.org/abs/1210.4896
  - Bengio et al.'s GSN papers repair ergodicity by choosing the variable to resample at random. https://arxiv.org/abs/1306.1091
  - In multiple imputation the same procedure is called "fully conditional specification" (MICE), where incompatibility is a known weakness. https://stefvanbuuren.name/fimd/sec-FCS.html
- **Arnold & Press 1989, "Compatible conditional distributions"** (JASA 84:152–156). They give necessary and sufficient conditions for discrete conditionals to come from one joint. https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478750
- **Young & You 2023, "On the inconsistencies of conditionals learned by masked language models"** (later "Inconsistencies in Masked Language Models", with Yunan Chen; arXiv 2301.00068).
  - Bigram conditionals in T5- and BERT-style MLMs cannot come from one joint distribution.
  - The later version finds the inconsistency "ubiquitous" from BERT-base to UL2-20B, and shows that an inference-time "Ensemble of Conditionals" improves accuracy (Lambada, MMLU, BigBench).
  - https://arxiv.org/abs/2301.00068, code https://github.com/tomyoung903/MLM_inconsistencies
- **Goyal, Dyer & Berg-Kirkpatrick, ICLR 2022** (arXiv 2106.02736, 2021): MLM conditionals do not define a valid Gibbs sampler. They read the MLM as an implicit energy (sequence score) and sample with Metropolis–Hastings instead. https://arxiv.org/abs/2106.02736
- **Torroba Hennigen & Kim, ACL 2023, "Deriving language models from masked language models".** On two-token joints, the best method finds the joint whose conditionals are closest to the MLM's. It beats MRF-based constructions, and the gain is largest when the masked tokens are close together, that is, more dependent. https://aclanthology.org/2023.acl-short.99/
- **Li et al., ICML 2024, "Promises and pitfalls of generative masked language modeling".**
  - Gibbs sampling on inconsistent conditionals "may amplify errors".
  - The asymptotic sample complexity of masked-prediction losses is tied to the mixing time of the corresponding Markov chain.
  - Larger masks improve statistical efficiency.
  - https://proceedings.mlr.press/v235/li24af.html
- **Sana et al. 2026, "Mixing times of Glauber dynamics on MLMs"** [preprint]. A rectangle test for incompatibility; incompatibility persists across scale; fast mixing at high temperature and metastable traps at low temperature (checked on BERT). https://arxiv.org/abs/2605.16378
- **Cai & Shi 2026** [preprint]: the mode-blindness identifiability theory from D1. https://arxiv.org/abs/2608.01383

**Any-order autoregressive models (AO-ARMs) and masked diffusion**
- **Uria, Murray & Larochelle, ICML 2014, "A deep and tractable density estimator".** Order-agnostic NADE trains all orderings with shared weights. Ensembles over orderings come for free and are better than any single ordering, which shows the orderings disagree. https://proceedings.mlr.press/v32/uria14.html
- **Shih, Sadigh & Ermon, NeurIPS 2022 (Oral), "Training and inference on any-order autoregressive models the right way" (MAC).**
  - AO-ARMs are redundant: they define the same distribution in many ways.
  - Training on a smaller set of univariate conditionals that still supports arbitrary conditional inference, and upweighting the conditionals used most at inference, improves likelihood.
  - https://arxiv.org/abs/2205.13554
- **XLNet** (Yang et al., NeurIPS 2019) [memory]. A permutation language-modelling objective over factorization orders with shared parameters. Nothing forces the orders to agree. https://arxiv.org/abs/1906.08237
- **Huang et al., ISMIR 2017, "Counterpoint by convolution" (CoCoNet, an orderless NADE).** Ancestral sampling in random order gives poor samples. Blocked Gibbs "significantly improves sample quality" even though both procedures nominally target the same distribution. The authors attribute this to some badly modeled conditionals. https://arxiv.org/abs/1903.07227
- **Liu, Ramadge & Adams, ICML 2024, "Generative marginalization models".** They model all marginals explicitly and train a one-step marginalization self-consistency loss across orderings: p(x_S) · p(x_j | x_S) = p(x_{S∪j}). https://arxiv.org/abs/2310.12920
- **Kim et al. 2025, ICML Outstanding Paper** (the D2 source). https://arxiv.org/abs/2502.06768
- **Prabhudesai et al., NeurIPS 2025, "Diffusion beats autoregressive in data-constrained settings".** With 25–100M unique tokens and up to 800 epochs, masked diffusion keeps improving where AR models plateau. The critical compute grows as about U^2.17. https://arxiv.org/abs/2507.15857

**LLM probabilistic coherence**
- **Fluri, Paleka & Tramèr, SaTML 2024**: inconsistencies in chess engines, GPT-4 forecasts and legal judgments. https://arxiv.org/abs/2306.09983
- **Paleka et al., ICLR 2025 Oral, "Consistency checks for language model forecasters".** An arbitrage-based consistency metric with an automated pipeline. Consistency correlates with Brier score; the COND check reaches R = 0.92. ArbitrageForecaster improves consistency but not reliably Brier score. There is a consistency benchmark that resolves in 2028. https://arxiv.org/abs/2412.18544, code https://github.com/dpaleka/consistency-forecasting
- **Andrews & Sarkar 2026, "Dutch books for language models"** [preprint]. An LP gives the maximum Dutch-book profit as a label-free incoherence metric on stock-market questions for 15 LLMs. Incoherence rises with richer logical structure, and irrelevant context can raise it tenfold. Profit varies about 100-fold across models against a narrow Brier range, with rank correlation 0.91. https://arxiv.org/abs/2609.02797
- **Dragonevskiy 2026** [preprint]: the "bounded Dutch book" definition and the Arbitr training method from D4. He claims the exact optimum of next-token prediction is incoherent across question formats, and that scale does not remove it [unverified, single author]. https://arxiv.org/abs/2609.39341
- **Zhu & Griffiths, CogSci 2024**: LLM probability judgments break probability identities in human-like ways. https://arxiv.org/abs/2401.16646
- **Kassner et al., EMNLP 2021 (BeliefBank)** and **Mitchell et al., EMNLP 2022 (ConCoRD).** Both add a MaxSAT layer over model beliefs to improve consistency and accuracy at test time. BeliefBank uses hand-written constraints; ConCoRD uses an NLI model and gains 5–8 points on QA/VQA.
  - https://aclanthology.org/2021.emnlp-main.697/
  - https://arxiv.org/abs/2211.11875

**In-context learning as Bayesian inference**
- **Falck, Wang & Holmes, ICML 2024**: the D5 result. https://proceedings.mlr.press/v235/falck24a.html
- **Jesson, Beltran-Velez & Blei, ICLR 2025.** A "generative predictive p-value" that runs posterior-predictive checks using only samples and log-probabilities from the model. https://arxiv.org/abs/2412.06033
- **Fong, Holmes & Walker, JRSSB 2023, "Martingale posterior distributions".** https://doi.org/10.1093/jrsssb/qkad005
- **Chlon et al. 2025** [preprint, contested]: the "Bayesian in expectation" argument from D5. https://arxiv.org/abs/2507.11768
- **Nagler & Rügamer 2025** (ICLR 2025 workshop): TabPFN fails the martingale check; their fix adds martingale-respecting forward resampling. https://arxiv.org/abs/2505.11325
- **Müller et al. 2025, position paper "The future of Bayesian prediction is prior-fitted".** https://arxiv.org/abs/2505.23947
- **Kim, Lee, Fong, Lee & Lee 2026**: drift, then stabilization, under predictive resampling, plus a self-consistency loss. https://arxiv.org/abs/2606.17832

**Evaluating world models**
- **Li et al., ICLR 2023 (Othello-GPT)**: an emergent board representation that can be intervened on. https://arxiv.org/abs/2210.13382
- **Vafa et al., NeurIPS 2024.** Models that pass next-token validity tests fail Myhill–Nerode compression and distinction tests; the paper calls their world models "incoherent". https://arxiv.org/abs/2406.03689
- **Vafa et al., ICML 2025, inductive-bias probes.** Models trained on orbits predict well but do not adapt as Newtonian mechanics would; they use task-specific heuristics. https://arxiv.org/abs/2507.06952
- **Relevance to ShipHunt.** Next-reading accuracy is the analogue of next-token validity, so it is not enough. The posterior-level metrics M1–M8, and KL to the exact posterior, are the analogues of their state-level tests.

---

## 3. How compute can buy coherence or accuracy from fixed data

| Lever | Where the compute goes | Evidence | Caveat |
|---|---|---|---|
| Ensembles over orderings or conditionals | Inference | Uria 2014 (order ensembles beat single orders); Young & You (Ensemble of Conditionals improves MMLU etc.) | Ensembles repair incoherence by averaging; they do not create a joint. |
| Better inference order | Inference | Kim et al. 2025: Sudoku <7% to ~90% | Needs the model's confidence to be informative about which conditional is easy. |
| MCMC refinement (Gibbs, MH, block Gibbs) | Inference | CoCoNet (better samples); Goyal 2022 (MH with the MLM as energy); Large Language Gibbs 2026 [preprint] (removes order bias, "a compromise between all local conditionals") https://arxiv.org/abs/2606.19264 | Li 2024: may amplify errors. Mixing can be slow (Sana 2026). |
| Iterative or semi-amortized inference | Inference | Cremer 2018; semi-amortized VAEs (Kim et al., ICML 2018) https://proceedings.mlr.press/v80/kim18e.html | Fixes amortization error only, not model error. |
| Coherence projection (exact) | Inference | Predd 2009 dominance https://doi.org/10.1109/TIT.2009.2027573 [DOI from memory; paper confirmed via PhilPapers https://philpapers.org/rec/PREPCA]; Mohri 2025 [preprint] https://arxiv.org/abs/2511.08440; Kong 2026 [preprint] https://arxiv.org/abs/2601.19862 | Guaranteed only in the matching Bregman geometry, and gains can be small. |
| Regularization + epochs + ensembles + distillation | Training | Kim et al. 2025: ~5.17x data efficiency, distillation keeps ~83% of the ensemble benefit https://arxiv.org/abs/2509.14786; Muennighoff 2023 https://arxiv.org/abs/2305.16264; Allen-Zhu & Li 2023 (theory of why ensembles and self-distillation help: "multi-view" data) https://arxiv.org/abs/2012.09816 | LLM-scale evidence; transfer to small nets is [unverified]. |
| Any-order / masked training as augmentation | Training | Prabhudesai 2025; Pan et al. 2025 (masked fine-tuning of AR models narrows the gap, reversal curse) https://arxiv.org/abs/2510.09885 | Gains in data efficiency come with incoherence (D1). |
| Self-consistency losses | Training | Schmitt et al., ICML 2024 (Bayes-symmetry loss; large gains in low-data regimes) https://arxiv.org/abs/2310.04395 and its unlabeled-data follow-up https://arxiv.org/abs/2501.13483; MaM marginalization loss; Kim 2026 martingale loss; Arbitr [preprint] | Can collapse to uninformative predictions without an outcome or calibration anchor. |
| Coherence as unsupervised training signal | Training | ICM (Wen et al. 2025) https://arxiv.org/abs/2506.10139; theory: Qiu et al. 2026 [preprint] https://arxiv.org/abs/2601.13566 | Elicits only what is already latent. |
| Compute → better targets → distill (expert iteration) | Both | ExIt (Anthony et al., NeurIPS 2017) https://arxiv.org/abs/1705.08439; AlphaZero (Silver et al., Science 2018) [memory]; Gumbel AlphaZero (ICLR 2022) https://openreview.net/forum?id=bERaNdoegnO [OpenReview id from memory; paper confirmed at https://iclr.cc/virtual/2022/poster/6418]; Markovian score climbing https://arxiv.org/abs/2003.10374 | The operator must be a real improvement; keep real data in the loop. |
| Training on dreams | Training | Sleep phase of wake-sleep (Hinton et al. 1995 [memory]; RWS https://arxiv.org/abs/1406.2751; Le et al. UAI 2019 https://proceedings.mlr.press/v115/le20a.html) learns the model's own posterior. Accumulating real plus synthetic data avoids collapse (Gerstgrasser 2024 https://arxiv.org/abs/2404.01413) | The fixed point is the generator's posterior (D6). Replacing real data collapses (Shumailov 2024 https://www.nature.com/articles/s41586-024-07566-y). |
| Test-time compute scaling in general | Inference | Snell et al., ICLR 2025: compute-optimal allocation per prompt beats a 14x larger model on easy and medium prompts https://arxiv.org/abs/2408.03314 | On the hardest prompts more parameters still won. |
| Synthetic data from a stronger generator | Training | EntiGraph (Yang et al., ICLR 2025): log-linear scaling with synthetic tokens https://arxiv.org/abs/2409.07431; megadocs (Kim et al. 2026 [preprint]): 1.48x to 1.80x data efficiency https://arxiv.org/abs/2603.18534 | Imports knowledge from an outside generator, so not a self-dreaming result [ours]. |

**Net reading [ours].** Under ShipHunt's rules (outcomes only, no outside model), the evidence-backed levers rank as follows:
1. Inference-time inversion and ensembling on the world model. These are cheap and fix amortization and incoherence error.
2. Regularized ensembles distilled into one network. This is the largest measured fixed-data gain.
3. Label-free self-consistency losses, which are strongest at small N (Schmitt et al.).
4. Dreaming with 0/1 targets, which helps only through amortization and regularization.

---

## 4. Proposed experiment arms for ShipHunt round 2

All arms keep the round-1 harness: single-cell KL to the exact posterior (scoring only), N ∈ {100, 300, 1k, 3k, 10k}, 3 seeds in the loop and 5 for the final TEST, compute per board reported. All use outcomes and model outputs only, with no hand-coded noise model, so each has a direct real-forecasting analogue. Ordered by cost.

**R2-A. Coherence panel (diagnostic only, near-zero cost).**
- **Procedure.** Compute M1–M8 for the arm-2 world model and its forecasters at each N, checkpoint and seed.
- **Pre-registered question:** within one model family, does each label-free metric rank checkpoints by KL to the exact posterior? Report Spearman with a CI over seeds.
- **Literature basis:** Paleka 2025 and Andrews & Sarkar 2026 found the cross-model correlation; whether it holds within a model is open (D4).
- **Real-world analogue:** label-free evaluation of LLM forecasters before questions resolve.
- **Why first:** it tells us which incoherence is worth fixing, and it costs only inference.

**R2-B. Inference-compute ladder on a fixed world model (no retraining).** For each evidence set, compute the posterior marginals with increasing compute:
1. Direct conditional query.
2. Plus averaging over the 8 symmetries and K factorization orders (Uria; Young & You).
3. Plus Bayes inversion by self-normalized importance sampling: sample boards from the model's own prior (full-mask generation), weight each by the model's own learned p(readings | board, locations), and average.
4. Plus MH or Gibbs refinement, using the any-order joint as the energy (Goyal), or adaptive-order decoding (Kim 2025).

- **Output:** a KL-versus-FLOPs curve at each N.
- **Hypothesis (D2) [ours]:** step 3 beats the direct query at small N, because the forward direction is easier to learn than the inverse. M3 (Bayes-symmetry gap) should predict how big that gain is.
- **Real-world analogue:** an LLM forecaster that samples scenarios and weights them by likelihood, versus one that states a probability directly.

**R2-C. Expert iteration: compute-improved targets, distilled into the forecaster.**
- **Procedure.**
  1. Use the best step of R2-B as the teacher, on real and dreamed evidence.
  2. Distill its marginals into the forecaster with soft targets.
  3. Optionally refit the world model and repeat, always on the real N episodes plus synthetic data (accumulate, never replace: Gerstgrasser).
- **Comparisons:**
  - Arm 3 (0/1 dream targets), which has the world model's direct posterior as its fixed point, and arm 4, which has the same fixed point with lower variance (D6).
  - Prediction [ours]: arms 3 and 4 converge to the same KL given enough dreams, and R2-C beats both by about the R2-B gain.
- **Flag:** like arm 4, this trains on the model's own posterior. Report it separately until Kelvin rules.
- **Real-world analogue:** distilling a slow ensemble/sampling LLM forecaster into a fast one (ExIt, AlphaZero, Kim et al.'s 83% distillation result).

**R2-D. Label-free consistency regularizers in world-model training.** Add these to the outcome loss with weight β and sweep β:
1. Cross-order agreement: the squared difference of log p(x) between two random orders, or MaM's one-step marginalization loss.
2. A Bayes-symmetry loss on real and dreamed evidence (Schmitt et al.).
3. A martingale loss on forecaster stages under the model's own imagined next probes (Fong; Kim et al. 2026; the 10-10 note).
4. A mask schedule with mass on full and low-visibility masks (Cai & Shi).

- **Predictions:** the largest gains at N ≤ 1k (Schmitt et al.'s low-data result).
- **Watch:** collapse toward the prior at large β (the Arbitr and ICM lesson). Report M1–M8 next to KL so that a coherent-but-wrong model is caught.
- **Real-world analogue:** consistency fine-tuning of LLM forecasters (Arbitr; Kim et al. 2026) anchored by resolved outcomes.

**R2-E. Fixed-data compute scaling.**
- **Procedure.**
  - Apply Kim et al. 2025's recipe to the world model: tuned heavy weight decay, many epochs, K ∈ {1, 2, 4, 8} independently trained world models ensembled (by logits or by sample pooling), then distilled.
  - Cross this with any-order versus fixed-order training (Prabhudesai).
- **Fits.** Fit KL against N for each recipe, and report an "effective data multiplier": the boards the baseline needs to match. Note whether the ensemble gain shrinks after R2-B, which would mean the ensemble mostly fixes incoherence.
- **Real-world analogue:** ensembles of forecasters, and when to spend compute on more models rather than more data.

**Budget note.** The round-2 cap is $100. A and B need inference only, so run them first. E is the most expensive (K training runs per N), so run it at K ≤ 4 and N ≤ 3k unless Kelvin raises the cap.

---

## Sources (all confirmed to exist via search on 2026-10-11 unless marked [memory])
- Heckerman et al. 2000, JMLR: https://www.jmlr.org/papers/volume1/heckerman00a/html/
- Arnold & Press 1989, JASA: https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478750
- Young, (Chen), You 2023: https://arxiv.org/abs/2301.00068
- Torroba Hennigen & Kim 2023: https://aclanthology.org/2023.acl-short.99/
- Goyal, Dyer, Berg-Kirkpatrick 2022: https://arxiv.org/abs/2106.02736
- Li et al. 2024: https://proceedings.mlr.press/v235/li24af.html
- Sana et al. 2026 [preprint]: https://arxiv.org/abs/2605.16378
- Cai & Shi 2026 [preprint]: https://arxiv.org/abs/2608.01383
- Choi, Gouk, Whitammer 2026 (Large Language Gibbs) [preprint; venue unclear]: https://arxiv.org/abs/2606.19264
- Uria, Murray, Larochelle 2014: https://proceedings.mlr.press/v32/uria14.html
- Shih, Sadigh, Ermon 2022: https://arxiv.org/abs/2205.13554
- XLNet [memory]: https://arxiv.org/abs/1906.08237
- Huang et al. 2017 (CoCoNet): https://arxiv.org/abs/1903.07227
- Liu, Ramadge, Adams 2024 (MaM): https://arxiv.org/abs/2310.12920
- Kim, Shah, Kontonis, Kakade, Chen 2025: https://arxiv.org/abs/2502.06768
- Prabhudesai et al. 2025: https://arxiv.org/abs/2507.15857
- Pan et al. 2025: https://arxiv.org/abs/2510.09885
- Fluri, Paleka, Tramèr 2024: https://arxiv.org/abs/2306.09983
- Paleka et al. 2025: https://arxiv.org/abs/2412.18544
- Andrews & Sarkar 2026 [preprint]: https://arxiv.org/abs/2609.02797
- Dragonevskiy 2026 [preprint]: https://arxiv.org/abs/2609.39341
- Zhu & Griffiths 2024: https://arxiv.org/abs/2401.16646
- Kassner et al. 2021 (BeliefBank): https://aclanthology.org/2021.emnlp-main.697/
- Mitchell et al. 2022 (ConCoRD): https://arxiv.org/abs/2211.11875
- Burns et al. 2023 (CCS) [memory]: https://arxiv.org/abs/2212.03827
- Farquhar et al. 2023: https://arxiv.org/abs/2312.10029
- Wen et al. 2025 (ICM): https://arxiv.org/abs/2506.10139
- Qiu et al. 2026 [preprint]: https://arxiv.org/abs/2601.13566
- Mohri, Schneider, Wu 2025 [preprint]: https://arxiv.org/abs/2511.08440
- Kong et al. 2026 [preprint]: https://arxiv.org/abs/2601.19862
- Predd et al. 2009: https://philpapers.org/rec/PREPCA
- Falck, Wang, Holmes 2024: https://proceedings.mlr.press/v235/falck24a.html
- Jesson, Beltran-Velez, Blei 2025: https://arxiv.org/abs/2412.06033
- Fong, Holmes, Walker 2023: https://doi.org/10.1093/jrsssb/qkad005
- Chlon et al. 2025 [preprint]: https://arxiv.org/abs/2507.11768
- Nagler & Rügamer 2025: https://arxiv.org/abs/2505.11325
- Müller et al. 2025 (position): https://arxiv.org/abs/2505.23947
- PFNs, Müller et al. 2022 [memory]: https://arxiv.org/abs/2112.10510
- Kim, Lee, Fong, Lee, Lee 2026: https://arxiv.org/abs/2606.17832
- Li et al. 2023 (Othello-GPT): https://arxiv.org/abs/2210.13382
- Vafa et al. 2024: https://arxiv.org/abs/2406.03689
- Vafa et al. 2025: https://arxiv.org/abs/2507.06952
- Cremer, Li, Duvenaud 2018: https://arxiv.org/abs/1801.03558
- Kim et al. 2018 (semi-amortized VAEs): https://proceedings.mlr.press/v80/kim18e.html
- Xu et al. 2020 (V-information): https://arxiv.org/abs/2002.10689
- Kim, Kotha, Liang, Hashimoto 2025 (ICLR 2026): https://arxiv.org/abs/2509.14786
- Kim et al. 2026 (megadocs) [preprint]: https://arxiv.org/abs/2603.18534
- Muennighoff et al. 2023: https://arxiv.org/abs/2305.16264
- Allen-Zhu & Li 2023: https://arxiv.org/abs/2012.09816
- Yang et al. 2025 (EntiGraph): https://arxiv.org/abs/2409.07431
- Cowsik et al. 2026 (self-play pretraining) [preprint]: https://arxiv.org/abs/2609.30063
- Schmitt et al. 2024: https://arxiv.org/abs/2310.04395
- Anthony, Tian, Barber 2017: https://arxiv.org/abs/1705.08439
- Danihelka et al. 2022: https://iclr.cc/virtual/2022/poster/6418
- AlphaZero, Silver et al. 2018, Science [memory]
- Naesseth, Lindsten, Blei 2020: https://arxiv.org/abs/2003.10374
- Bornschein & Bengio 2015 (RWS): https://arxiv.org/abs/1406.2751
- Le et al. 2019: https://proceedings.mlr.press/v115/le20a.html
- Hinton et al. 1995 (wake-sleep), Science [memory]
- Shumailov et al. 2024: https://www.nature.com/articles/s41586-024-07566-y
- Gerstgrasser et al. 2024: https://arxiv.org/abs/2404.01413
- Snell et al. 2024/2025: https://arxiv.org/abs/2408.03314
