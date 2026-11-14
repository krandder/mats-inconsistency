# World models at several levels of abstraction, and the order of sampling

*2026-10-11. Read-only literature research for Kelvin's MATS work (ShipHunt round 2 and the LLM forecaster). No one was contacted.*

**Kelvin's question (verbatim):** "For real-world applications, there are world-models *at different levels of abstraction*. One would want the dreams, and also the learning of the world-model, to happen naturally at these different levels whenever needed, without any kind of explicit architecture. But then the 'sampling' of the world is not a trivial, because the right 'order' of sampling matters enormously."

**How the sources were checked.** Each arXiv ID was checked against the arXiv API (title, first author and date match). Each DOI was checked against Crossref (title, venue and year match). Other URLs were fetched and returned HTTP 200, or a search engine confirmed them.
- *[abstract]* means the claim comes from the abstract I fetched today.
- *[memory]* means the claim comes from my prior knowledge of the paper and was not re-read today. Treat these as likely but unverified.
- *[secondary]* means a search-engine summary of the paper, not the paper itself.
- *Our hypothesis* means the claim is mine, not the paper's.

---

## A. Deep insights

1. **Order matters only because the learned conditionals are imperfect, so a good order routes around your model's worst conditionals.**
   - Training on random orders makes a masked or any-order model face many infilling subproblems, some of them provably hard. Choosing the decoding order adaptively at inference sidesteps them: on Sudoku, accuracy went from <7% to about 90% with no retraining.
   - Source: Kim et al. 2025, "Train for the Worst, Plan for the Best" (ICML 2025), [arXiv 2502.06768](https://arxiv.org/abs/2502.06768) [abstract].
   - Implication: keep random-order reveal-chain TD for training. Spend effort on choosing the order at dream and sampling time, using the model's own confidence. That is outcomes-only and needs no architecture change.

2. **Learning the order is now cheap: a small policy on top of a frozen any-order model, trained from the model's own loss or from outcome reward.**
   - Jazbec et al. 2025 train a one-layer unmasking policy with GRPO ([arXiv 2512.09106](https://arxiv.org/abs/2512.09106), ICML 2026) [abstract].
   - Mohamud et al. 2026 reweight the masked-diffusion loss by policy probabilities, so the policy learns to prefer positions where the denoiser is likely right ([arXiv 2606.00295](https://arxiv.org/abs/2606.00295)) [abstract].
   - Garg et al. 2025 prove the MDM objective decomposes exactly into weighted autoregressive losses over orders, so the order can be learned inside the training loss ([arXiv 2511.19152](https://arxiv.org/abs/2511.19152)) [abstract].
   - Implication: a learned reveal order for reveal-chain TD is a small, rule-compliant meta-policy, not a hand-designed hierarchy.

3. **Coarse-to-fine emerges from the noise process, not from the architecture.**
   - In diffusion on hierarchical data, high-level features (the class) are decided at a sharp phase transition in noise level, while low-level details change smoothly. The same holds for masking diffusion.
   - Sources: Sclocchi, Favero & Wyart, PNAS 2025, [arXiv 2402.16991](https://arxiv.org/abs/2402.16991); follow-up [arXiv 2410.13770](https://arxiv.org/abs/2410.13770) [abstract].
   - Implication (our hypothesis): in reveal-chain TD, "fraction revealed" already acts as an abstraction axis. The early-stage forecasts are the coarse world model. The levels already exist implicitly; the question is only which variables to commit first.

4. **Bootstrapping on your own latents may be what learns abstractions, and explicit hierarchy may be redundant.**
   - For data from a probabilistic context-free grammar of depth L, token-level supervised or self-supervised learning needs samples exponential in L to recover the latent tree. Predicting the model's own latents needs a number constant in L (up to logs).
   - The paper also shows that data2vec implicitly does hierarchical latent prediction, and it concludes that "explicit stacking such as H-JEPA is largely redundant."
   - Source: Korchinski, Favero & Wyart 2026, [arXiv 2605.27734](https://arxiv.org/abs/2605.27734) [abstract; v1 preprint, not peer-reviewed].
   - Implication: this is the strongest theoretical support I found for Kelvin's "no explicit architecture" position, and for reveal-chain TD's self-targets. It is directly testable on a hierarchical ShipHunt.

5. **Generic sequence training does produce linearly decodable latent world states, but good predictions can sit on an incoherent world model.**
   - Positive evidence:
     - Belief states over the hidden generator appear in the residual stream (Shai et al. 2024, [arXiv 2405.15943](https://arxiv.org/abs/2405.15943)).
     - Grammar non-terminals are encoded (Allen-Zhu & Li, [arXiv 2305.13673](https://arxiv.org/abs/2305.13673)).
     - The Othello board is linearly decodable ([arXiv 2309.00941](https://arxiv.org/abs/2309.00941)).
   - Negative evidence: Vafa et al. 2024 ([arXiv 2406.03689](https://arxiv.org/abs/2406.03689)) and 2025 ([arXiv 2507.06952](https://arxiv.org/abs/2507.06952), Kepler-like prediction without Newtonian structure).
   - Implication: score dreams on coherence (order-invariance, valid ship shapes, how the ship count is distributed), not only on per-cell KL.

6. **Any-order models are not automatically one coherent joint distribution.**
   - Different orders or masks imply different joints (Young et al. 2022, [arXiv 2301.00068](https://arxiv.org/abs/2301.00068); Shih et al. 2022, [arXiv 2205.13554](https://arxiv.org/abs/2205.13554)).
   - Enforcing marginalization self-consistency is its own training objective (Liu et al., ICML 2024, [arXiv 2310.12920](https://arxiv.org/abs/2310.12920)).
   - Implication (our hypothesis): reveal-chain TD is in effect a cross-order coherence regularizer, which is exactly Kelvin's "coherent joint beliefs." The order-dependence of log p(board) is a cheap, label-free diagnostic to track.

7. **Coherence across levels improves forecast accuracy when it is imposed. For LLMs we only have correlation so far.**
   - Projection reconciliation is guaranteed to improve accuracy for a class of distance-based losses, and MinT is optimal in that class (Panagiotelis et al. 2021, [doi:10.1016/j.ijforecast.2020.06.004](https://doi.org/10.1016/j.ijforecast.2020.06.004) [abstract]).
   - Coherence weighting improves human forecasters by up to 30% (Karvetski et al. 2013, [doi:10.1287/deca.2013.0279](https://doi.org/10.1287/deca.2013.0279) [secondary]).
   - For LLMs, consistency correlates with Brier score (Paleka et al., [arXiv 2412.18544](https://arxiv.org/abs/2412.18544) [abstract]). Subquestion decomposition gave only a small gain: Brier 0.141 to 0.132 with overlapping CIs (Bosse et al. 2026, [arXiv 2601.22444](https://arxiv.org/abs/2601.22444) [secondary]).
   - Implication: cross-level coherence is a label-free training signal with a theoretical basis. Whether coherence training improves LLM accuracy is still untested causally.

8. **Choosing which variable to sample or reveal next is a value-of-computation problem. Today's learned unmasking policies are metareasoning policies with the wrong objective for forecasting.**
   - Classic sources: Russell & Wefald 1991, [doi:10.1016/0004-3702(91)90015-C](https://doi.org/10.1016/0004-3702(91)90015-C); Callaway et al. 2018, [arXiv 1711.06892](https://arxiv.org/abs/1711.06892).
   - dLLM unmasking policies optimize the quality and speed of one output. They do not optimize the information value for a downstream belief.
   - Implication (our hypothesis): in reveal-chain TD the reveal order chooses how much of the future each target bootstraps from. A VOI-first order works like a per-step adaptive λ. That gives a principled objective for a learned order: minimize the forecaster's held-out outcome loss.

---

## B. Where the field has no answer

1. **The best training-time order for a bootstrapped or TD target.** Kim et al. separate train order (random) from inference order (adaptive). Garg, Mohamud and Sato learn orders to improve generation or learnability. No paper studies how the order shapes the *targets* in self-distillation or TD (bias vs variance of the bootstrapped target). This gap is specific to Kelvin's method.
2. **Whether a model can invent a high-level variable and sample it first when that variable is not an addressable token.** VAR, cascaded/latent diffusion and multiscale PixelCNN get coarse-to-fine from the architecture or the data transform. Masked-diffusion orders are over existing tokens. The generic ways a model gets an "unnamed latent" to commit to first are:
   - chain-of-thought or latent thoughts: Phan et al. TRICE, [arXiv 2312.02179](https://arxiv.org/abs/2312.02179); Ruan et al., [arXiv 2503.18866](https://arxiv.org/abs/2503.18866); Coconut, [arXiv 2412.06769](https://arxiv.org/abs/2412.06769);
   - the noise level (insight 3).

   No result shows a flat model discovering *and* ordering such latents end-to-end from outcomes alone.
3. **A sample-complexity separation between fixed-order and any-order training.** None exists in general (secondary search summary; I found none either). Korchinski et al. cover only PCFG data.
4. **Does enforcing coherence across levels improve LLM forecaster accuracy (not just correlate with it)?** The reconciliation theory is linear and covers point or Gaussian-like forecasts. Discrete or nonlinear probabilistic reconciliation is recent and small (e.g. [arXiv 2305.18809](https://arxiv.org/abs/2305.18809), [arXiv 2604.26668](https://arxiv.org/abs/2604.26668); titles only, not read). I did not check whether Paleka et al.'s consistency-enforcing post-processing improved Brier.
5. **How to evaluate "the right abstraction."** Natural abstractions has theorems (Telephone, generalized KPD) but little evidence that they apply in finite settings. Chan, Lang & Jenner 2023 ([LessWrong](https://www.lesswrong.com/posts/gvzW46Z3BsaZsLc25/natural-abstractions-key-claims-theorems-and-critiques-1)) list the gaps. Vafa et al. show that probes and next-token accuracy do not certify a world model.
6. **A learned VOC for forecasting.** No work learns "which related question, or which hypothetical outcome, to condition on next" for an LLM forecaster. The FRI conditional-trees report ([2024](https://forecastingresearch.org/research/ai-conditional-trees)) generates high-VOI questions by hand, with humans in the loop.
7. **Arbitrary order can hurt.** "The Flexibility Trap" (Ni et al. 2026, [arXiv 2601.15165](https://arxiv.org/abs/2601.15165)) finds arbitrary order can *narrow* what RL can extract; AR order gives higher pass@k [secondary]. No theory says when flexible order helps and when it hurts.

---

## C. Experiments: learned abstraction plus learned sampling order (cheap, within Kelvin's rules)

The rules that apply throughout:
- outcomes only, with the exact posterior used for scoring only;
- no hand-coded noise model or ship-shape program;
- no hierarchy in the architecture;
- any added "levels" exist only as *questions about outcomes* (as they do in real forecasting) or emerge inside the network.

### E1. Hierarchical ShipHunt with aggregate questions (the "who wins vs county turnout" analogue)
- **Generator:** a hidden *fleet type* per episode (for example, type A = two long ships, type B = four short ships, with different noise near ships). The fleet type is revealed at episode end as an ordinary outcome, the way "candidate Y drops out" resolves. It is never an input to the forecaster.
- **Forecaster:** one query-conditioned network. The input is the evidence plus a query. A query is a cell, an aggregate over a cell set (row count, "any ship in quadrant", total ship cells), or the fleet-type outcome. Aggregates are just more outcome questions, so there is no architectural hierarchy.
- **Reveal-chain TD extended:** the chain can reveal aggregate answers as well as cells.
- **Arms:**
  - coarse-first (aggregates and fleet type first, then cells);
  - fine-first;
  - random;
  - learned (see E2).
- **Metrics:**
  - single-cell KL to the exact posterior (as in round 1);
  - KL on aggregates;
  - the coherence gap between the forecast of an aggregate and the aggregate implied by the cell forecasts. This is a MinT-style residual that is label-free.
- **Predictions:**
  - Kim et al.'s Latents-and-Observations analysis and Sclocchi's phase transition predict that coarse-first lowers early-stage target error once the latent is easy to predict.
  - Korchinski predicts that TD self-targets learn the fleet-type latent with fewer boards than Monte Carlo targets.

### E2. Learned reveal order in reveal-chain TD (a meta-policy, about one GPU-hour)
- **Baselines:**
  - random order (the current champion);
  - easy-first: reveal the cell where the model is most confident, using the top-probability margin as in Kim et al. [memory];
  - hard-first / max-entropy-first (myopic VOI);
  - loss-profiling order search (Sato et al. 2025, [arXiv 2506.23875](https://arxiv.org/abs/2506.23875): rank orders by early training loss).
- **Learned arm:** a small policy head trained in the Mohamud et al. style (reweight the TD loss by the policy over the next reveal) or with GRPO, where the reward is the reduction in validation outcome log-loss.
- **Key comparison:** the order used *for training targets* vs the order used *at test or dream time*, so the effects can be separated (gap B1).
- **Diagnostic:** plot the per-step bias and variance of the target against λ, to test the "order = adaptive λ" hypothesis.

### E3. Dream order for the round-2 world model
- **Setup:** sample dreamed episodes from the generic joint world model under these orders:
  - random any-order;
  - confidence-first (Kim; MaskGIT, [arXiv 2202.04200](https://arxiv.org/abs/2202.04200));
  - a learned planner (P2, [arXiv 2502.03540](https://arxiv.org/abs/2502.03540); DDPD, [arXiv 2410.06264](https://arxiv.org/abs/2410.06264));
  - self-distilled specialized order (OAR, [arXiv 2504.17069](https://arxiv.org/abs/2504.17069)).
- **Metrics:**
  - dream fidelity, scored with data only: held-out log-likelihood of real boards, the distribution of ship count and run lengths against held-out real boards, and the rate of contiguous ships, as a statistic computed rather than built into the model;
  - downstream forecaster KL after training on the dreams.
- **Prediction:** ship placement is constraint-like (Sudoku-like), so adaptive order should give a large gain in dream coherence.

### E4. Emergent abstraction probe and order-coherence diagnostic (almost free; reuses the E1 checkpoints)
- **Probe:** a linear probe for fleet type and total ship count in the residual stream, at 100, 300, 1k, 3k and 10k boards.
- **Test:** whether the high-level latent becomes decodable at a data threshold (Cagnetta & Wyart 2024, [arXiv 2406.00048](https://arxiv.org/abs/2406.00048)), and whether it does so earlier with TD/self-targets than with Monte Carlo targets (Korchinski).
- **Order-invariance diagnostic:** compute log p(board) under two random orders (Young et al.'s inconsistency). This turns "coherent joint beliefs" into a scalar.

### E5. Real-world transfer: order of conditioning for an LLM forecaster (small; uses existing question clusters)
- **Data:** linked resolved question clusters (nominee to general winner to state or turnout questions; ForecastBench, [arXiv 2409.19839](https://arxiv.org/abs/2409.19839), or Metaculus/Polymarket groups).
- **Arms:**
  - direct forecasts;
  - aggregate-first elicitation, then details conditioned on it;
  - details-first elicitation, then the aggregate;
  - scenario sampling: sample K high-level outcomes from the model's own forecast, forecast the details conditionally, then average;
  - the model's own VOI chooses which linked question to condition on first.
- **Scoring:** Brier score plus the arbitrage-consistency metric of Paleka et al. MinT-style reconciliation is a post-hoc baseline.
- **Rules:** no structure is hand-coded beyond the question links that already exist in the data.
- **Purpose:** this is the test of insight 7's open causal question.

---

## D. Literature by topic (verified citations)

### 1. Order of sampling and generation

**Any-order and learned-order autoregression**
- Uria, Murray & Larochelle 2014. Order-agnostic NADE: one model trained over all orderings, with ensembles over orders. [arXiv 1310.1757](https://arxiv.org/abs/1310.1757).
- Yang et al. 2019, XLNet. Permutation language modeling. [arXiv 1906.08237](https://arxiv.org/abs/1906.08237).
- Stern et al. 2019, Insertion Transformer. Generation by insertion, with flexible orders such as balanced-binary-tree order [memory]. [arXiv 1902.03249](https://arxiv.org/abs/1902.03249).
- Gu, Liu & Cho 2019. Insertion-based decoding with an order inferred automatically as a latent variable. [arXiv 1902.01370](https://arxiv.org/abs/1902.01370).
- Li et al. 2021. Non-monotonic AR orderings discovered by variational inference: an order *encoder* is trained jointly with the decoder [memory]. [arXiv 2110.15797](https://arxiv.org/abs/2110.15797).
- Hoogeboom et al. 2022, Autoregressive Diffusion Models (ICLR). Order-agnostic AR as absorbing diffusion. [arXiv 2110.02037](https://arxiv.org/abs/2110.02037).
- Shih, Sadigh & Ermon 2022, "AO-ARMs the right way" (NeurIPS).
  - Any-order models define the same joint in many redundant ways, which wastes capacity [secondary].
  - The fix trains on a smaller set of conditionals.
  - [arXiv 2205.13554](https://arxiv.org/abs/2205.13554).
- Pannatier et al. 2024, σ-GPT. Random-order AR with double positional encoding, plus rejection-sampling burst generation [memory]. [arXiv 2404.09562](https://arxiv.org/abs/2404.09562).
- Wang et al. 2025, Learning-Order ARMs (LO-ARM). A trainable order policy for molecular graphs. [arXiv 2503.05979](https://arxiv.org/abs/2503.05979).
- Guo & Ermon 2025, any-subset AR models. Any-subset speculative decoding samples parallel tokens from the *correct* joint [abstract]. [arXiv 2504.20456](https://arxiv.org/abs/2504.20456).
- Sato et al. 2025. Learning-friendly orders found by loss profiling.
  - Random orders are nearly unlearnable; found orders reach about 100%.
  - The method rediscovers reverse-digit multiplication [abstract].
  - [arXiv 2506.23875](https://arxiv.org/abs/2506.23875).
- Biswal, Vuffray & Lokhov 2026. In AR Ising models, an order derived from the learned graph beats naive orders [abstract]. [arXiv 2602.20394](https://arxiv.org/abs/2602.20394).
- Bachmann & Nagarajan 2024, "The pitfalls of next-token prediction."
  - Teacher forcing can fail to *learn* a planning task (path-star graph).
  - A multi-token "teacherless" objective helps [abstract].
  - Lesson: the training order or factorization decides learnability, not only inference.
  - [arXiv 2403.06963](https://arxiv.org/abs/2403.06963).

**Masked diffusion and the choice of what to unmask**
- Lou et al. 2024, SEDD. [arXiv 2310.16834](https://arxiv.org/abs/2310.16834).
- Sahoo et al. 2024, MDLM. [arXiv 2406.07524](https://arxiv.org/abs/2406.07524).
- Ou et al. 2024, RADD. Absorbing diffusion models clean-data conditionals. [arXiv 2406.03736](https://arxiv.org/abs/2406.03736).
- Zheng et al. 2024. MDMs are time-agnostic masked models; categorical-sampling precision had inflated earlier results [memory]. [arXiv 2409.02908](https://arxiv.org/abs/2409.02908).
- Nie et al. 2025, LLaDA. An 8B masked-diffusion LM that uses low-confidence remasking [memory]. [arXiv 2502.09992](https://arxiv.org/abs/2502.09992).
- Chang et al. 2022, MaskGIT. The confidence-first unmasking schedule. [arXiv 2202.04200](https://arxiv.org/abs/2202.04200).
- Liu et al. 2024, DDPD. A separate *planner* chooses which position to denoise; the denoiser fills it in. [arXiv 2410.06264](https://arxiv.org/abs/2410.06264).
- Peng et al. 2025, P2 path planning. A planner chooses which positions to unmask or remask. [arXiv 2502.03540](https://arxiv.org/abs/2502.03540).
- Ye et al. 2024, "Beyond Autoregression." On "subgoal imbalance," diffusion beats AR on Sudoku/Countdown-type planning [memory]. [arXiv 2410.14157](https://arxiv.org/abs/2410.14157).
- Kim et al. 2025, "Train for the Worst, Plan for the Best" (ICML 2025; PMLR [v267](https://proceedings.mlr.press/v267/kim25ah.html); outstanding paper per secondary source). Insight 1. Its "Latents-and-Observations" distributions, where latent tokens determine the observations, are the closest formal model of Kelvin's "sample the latent first." [arXiv 2502.06768](https://arxiv.org/abs/2502.06768).
- Garg et al. 2025, "MDMs are secretly learned-order ARMs." [arXiv 2511.19152](https://arxiv.org/abs/2511.19152).
- Jazbec et al. 2025 (Apple), learned unmasking policies with RL. [arXiv 2512.09106](https://arxiv.org/abs/2512.09106); code at [github.com/apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm) [secondary].
- Chen et al. 2025, dUltra. [arXiv 2512.21446](https://arxiv.org/abs/2512.21446).
- Asano et al. 2026, Where-to-Unmask. [arXiv 2602.09501](https://arxiv.org/abs/2602.09501).
- Mohamud et al. 2026, Adaptive Order Policies. [arXiv 2606.00295](https://arxiv.org/abs/2606.00295).
- Raajesh et al. 2026, mask-aware policy gradients: the unmasking order belongs inside the RL likelihood. [arXiv 2607.15200](https://arxiv.org/abs/2607.15200).
- Pramanik et al. 2025, OAR: self-distill specialized orders from the any-order model's own confidence [abstract]. [arXiv 2504.17069](https://arxiv.org/abs/2504.17069).
- Counterpoints:
  - Ni et al. 2026, "The Flexibility Trap." [arXiv 2601.15165](https://arxiv.org/abs/2601.15165).
  - Yang et al. 2025: any-order adds little expressivity over a masked ARM [secondary]. [arXiv 2510.06190](https://arxiv.org/abs/2510.06190).
- Young et al. 2022: inconsistent conditionals in masked LMs. [arXiv 2301.00068](https://arxiv.org/abs/2301.00068).
- Liu et al. 2024: marginalization models. [arXiv 2310.12920](https://arxiv.org/abs/2310.12920).

**Coarse-to-fine**
- Reed et al. 2017, multiscale PixelCNN. [arXiv 1703.03664](https://arxiv.org/abs/1703.03664).
- Menick & Kalchbrenner 2018, Subscale Pixel Networks. [arXiv 1812.01608](https://arxiv.org/abs/1812.01608).
- Razavi et al. 2019, VQ-VAE-2: a hierarchical prior with the top level sampled first. [arXiv 1906.00446](https://arxiv.org/abs/1906.00446).
- Tian et al. 2024, VAR next-scale prediction (NeurIPS 2024). [arXiv 2404.02905](https://arxiv.org/abs/2404.02905).
- Rombach et al. 2022, latent diffusion. [arXiv 2112.10752](https://arxiv.org/abs/2112.10752).
- Rissanen et al. 2022, inverse heat dissipation: generation as explicit coarse-to-fine in frequency. [arXiv 2206.13397](https://arxiv.org/abs/2206.13397).
- Sclocchi et al. 2024/2025 (insight 3). [arXiv 2402.16991](https://arxiv.org/abs/2402.16991), [arXiv 2410.13770](https://arxiv.org/abs/2410.13770).
- Chen et al. 2024, Diffusion Forcing: per-token noise levels, i.e. a per-variable "how committed" level for world models and planning. [arXiv 2407.01392](https://arxiv.org/abs/2407.01392).
- Note: VAR, VQ-VAE-2 and the multiscale models hard-code the levels. Diffusion and masked diffusion get them implicitly. Only the latter fits Kelvin's rule.

### 2. Emergent abstraction without explicit architecture

**World states and latents decodable from flat training**
- Li et al. 2023, Othello-GPT. [arXiv 2210.13382](https://arxiv.org/abs/2210.13382).
- Nanda et al. 2023, linear representations in the Othello model. [arXiv 2309.00941](https://arxiv.org/abs/2309.00941).
- Allen-Zhu & Li, Physics of LMs Part 1: GPT learns deep CFGs, and hidden states encode non-terminals [memory]. [arXiv 2305.13673](https://arxiv.org/abs/2305.13673).
- Gurnee & Tegmark 2023, space and time. [arXiv 2310.02207](https://arxiv.org/abs/2310.02207).
- Park, Choe & Veitch 2024, hierarchical concepts are orthogonal in representation space [memory]. [arXiv 2406.01506](https://arxiv.org/abs/2406.01506).
- Shai et al. 2024, belief-state geometry. [arXiv 2405.15943](https://arxiv.org/abs/2405.15943).
- Huh et al. 2024, Platonic representation hypothesis: weak evidence of universality. [arXiv 2405.07987](https://arxiv.org/abs/2405.07987).

**Hierarchy learned as data grows**
- Cagnetta et al. 2024, Random Hierarchy Model (PRX). Deep networks need about n_c·m^L samples, the point where representations become invariant to synonyms [secondary]. [arXiv 2307.02129](https://arxiv.org/abs/2307.02129).
- Cagnetta & Wyart 2024: next-token prediction acquires the hierarchy level by level as data grows. [arXiv 2406.00048](https://arxiv.org/abs/2406.00048).
- Favero et al. 2025, the same for diffusion. [arXiv 2502.12089](https://arxiv.org/abs/2502.12089).
- Ren et al. 2026, provable learning of RHMs. [arXiv 2601.19756](https://arxiv.org/abs/2601.19756).
- Korchinski et al. 2026, insight 4. [arXiv 2605.27734](https://arxiv.org/abs/2605.27734).

**Planning ahead inside flat models**
- Pal et al. 2023, Future Lens. [arXiv 2311.04897](https://arxiv.org/abs/2311.04897).
- Wu et al. 2024: mostly "breadcrumbs" rather than deliberate pre-caching at small scale [memory]. [arXiv 2404.00859](https://arxiv.org/abs/2404.00859).
- Lindsey et al. 2025, "On the biology of an LLM": rhyme planning [memory]. [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

**Prediction in latent space**
- LeCun 2022, "A Path Towards Autonomous Machine Intelligence" (H-JEPA). [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf).
- Assran et al. 2023, I-JEPA. [arXiv 2301.08243](https://arxiv.org/abs/2301.08243).
- Assran et al. 2025, V-JEPA 2. Over 1M hours of video, then an action-conditioned world model from <62 h of robot video, with zero-shot planning [abstract]. [arXiv 2506.09985](https://arxiv.org/abs/2506.09985).
- LCM team 2024, Large Concept Models: AR in a sentence-embedding space. Concept = sentence is an *explicit* abstraction level, so it does not fit Kelvin's rule [abstract]. [arXiv 2412.08821](https://arxiv.org/abs/2412.08821).

**Latent thoughts: a generic way to "sample a high-level latent first"**
- Wang et al. 2023, planning tokens. [arXiv 2310.05707](https://arxiv.org/abs/2310.05707).
- Hao et al. 2024, Coconut. [arXiv 2412.06769](https://arxiv.org/abs/2412.06769).
- Ruan et al. 2025, latent thoughts. [arXiv 2503.18866](https://arxiv.org/abs/2503.18866).
- Phan et al. 2023, TRICE: the CoT is treated as a latent variable trained from answer correctness only, which is outcomes-only latent learning. [arXiv 2312.02179](https://arxiv.org/abs/2312.02179).

### 3. Abstraction in world models and planning

**Temporal abstraction**
- Sutton, Precup & Singh 1999, options. [doi:10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1).
- Bacon et al. 2017, option-critic. [arXiv 1609.05140](https://arxiv.org/abs/1609.05140).
- Hafner et al. 2022, Director: a manager chooses latent goals for a worker inside a learned world model. [arXiv 2206.04114](https://arxiv.org/abs/2206.04114).
- Hafner et al. 2023, DreamerV3. [arXiv 2301.04104](https://arxiv.org/abs/2301.04104).
- Gumbsch et al. 2024, THICK (ICLR). The high level emerges from sparsely changing latent contexts and predicts only context changes. [OpenReview TjCDNssXKU](https://openreview.net/forum?id=TjCDNssXKU).
- Mattes et al. 2023, Hieros. [arXiv 2310.05167](https://arxiv.org/abs/2310.05167).
- Saxena et al. 2021, Clockwork VAE: fixed tick rates, i.e. an explicit architecture. [arXiv 2102.09532](https://arxiv.org/abs/2102.09532).
- Ha & Schmidhuber 2018, World Models. [arXiv 1803.10122](https://arxiv.org/abs/1803.10122).
- **Gregor et al. 2019, TD-VAE:** the belief at time t1 is trained to predict the state at a randomly chosen later t2 ("jumpy" prediction) [memory]. This is the closest classic relative of reveal-chain TD; the reveal count plays the role of time. [arXiv 1806.03107](https://arxiv.org/abs/1806.03107).

**State abstraction**
- Li, Walsh & Littman 2006: a hierarchy of abstraction types, from model-irrelevance (bisimulation) to π*-irrelevance; finer types preserve more. [PDF](http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf).
- Givan, Dean & Greig 2003, bisimulation. [doi:10.1016/S0004-3702(02)00376-4](https://doi.org/10.1016/S0004-3702(02)00376-4).
- Zhang et al. 2021, deep bisimulation for control. [arXiv 2006.10742](https://arxiv.org/abs/2006.10742).
- Tishby, Pereira & Bialek 1999/2000, information bottleneck. [arXiv physics/0004057](https://arxiv.org/abs/physics/0004057).

**Causal abstraction**
- Rubenstein et al. 2017. [arXiv 1707.00819](https://arxiv.org/abs/1707.00819).
- Beckers & Halpern 2019. [arXiv 1812.03789](https://arxiv.org/abs/1812.03789).
- Geiger et al. 2021, causal abstractions of neural networks. [arXiv 2106.02997](https://arxiv.org/abs/2106.02997).
- Geiger et al. 2023/2025, theoretical foundation for mechanistic interpretability. [arXiv 2301.04709](https://arxiv.org/abs/2301.04709).
- Use for Kelvin (our hypothesis): an interchange-intervention test of whether the forecaster's internal fleet-type direction *causally* mediates the cell forecasts. This is stronger evidence than a probe.

**Natural abstractions**
- Wentworth, "Testing the Natural Abstraction Hypothesis." [LessWrong](https://www.lesswrong.com/posts/cy3BhHrGinZCp3LXE/testing-the-natural-abstraction-hypothesis-project-intro).
- Chan, Lang & Jenner 2023, critique. [LessWrong](https://www.lesswrong.com/posts/gvzW46Z3BsaZsLc25/natural-abstractions-key-claims-theorems-and-critiques-1).
  - Central claim: abstractions are the information that is redundant, or conserved over distance.
  - Critique: the theorems are infinite-limit results, and the universality evidence is mixed [secondary].
  - The redundancy idea matches ShipHunt directly: fleet type is the information redundantly carried by many cells.

### 4. Abstraction in probabilistic forecasting

**Hierarchical reconciliation**
- Hyndman et al. 2011. [doi:10.1016/j.csda.2011.03.006](https://doi.org/10.1016/j.csda.2011.03.006).
- Wickramasuriya, Athanasopoulos & Hyndman 2019, MinT (JASA). [doi:10.1080/01621459.2018.1448825](https://doi.org/10.1080/01621459.2018.1448825).
- Panagiotelis et al. 2021, geometric view. [doi:10.1016/j.ijforecast.2020.06.004](https://doi.org/10.1016/j.ijforecast.2020.06.004).
- Panagiotelis et al. 2023, probabilistic reconciliation with score optimisation (EJOR). [doi:10.1016/j.ejor.2022.07.040](https://doi.org/10.1016/j.ejor.2022.07.040).
- Athanasopoulos et al. 2024, review (IJF 40(2)). [doi:10.1016/j.ijforecast.2023.10.010](https://doi.org/10.1016/j.ijforecast.2023.10.010).
- Lesson: coherence across aggregation levels is cheap to impose and provably helps for distance-based losses. It does not create an abstraction; it only makes the given levels agree.

**Scenario and correlated forecasting**
- Heidemanns, Gelman & Morris 2020, the Economist model (HDSR). [doi:10.1162/99608f92.fc62f1e1](https://doi.org/10.1162/99608f92.fc62f1e1).
  - State trends are correlated, so a national swing moves every state. This is effectively "sample the high-level swing, then the state deviations."
  - Gelman et al. (JDM 2020) criticized FiveThirtyEight's state correlations as too low [secondary].
- Han et al. 2025, scenario analysis via sequential meta-posterior sampling. [arXiv 2510.19133](https://arxiv.org/abs/2510.19133) (title only).

**Coherence and accuracy in human forecasting**
- Wang, Kulkarni, Poor & Osherson 2011. [doi:10.1287/deca.1110.0206](https://doi.org/10.1287/deca.1110.0206).
- Karvetski et al. 2013. [doi:10.1287/deca.2013.0279](https://doi.org/10.1287/deca.2013.0279).
- Both find that coherentizing or coherence-weighting improves accuracy [secondary].

**LLM forecasters**
- Halawi et al. 2024. [arXiv 2402.18563](https://arxiv.org/abs/2402.18563).
- ForecastBench. [arXiv 2409.19839](https://arxiv.org/abs/2409.19839).
- Paleka et al. 2024, consistency checks: an arbitrage metric that correlates with Brier score. [arXiv 2412.18544](https://arxiv.org/abs/2412.18544).
- Bosse et al. 2026, subquestion decomposition. [arXiv 2601.22444](https://arxiv.org/abs/2601.22444).
- Feng et al. 2024, BIRD: the LLM proposes factors, and a Bayesian network combines them. [arXiv 2404.12494](https://arxiv.org/abs/2404.12494) (title only).
- Turtel et al. 2025: outcome-ranked self-play with DPO, which is outcomes-only. [arXiv 2502.05253](https://arxiv.org/abs/2502.05253).
- FRI conditional trees 2024: VOI-driven question generation. [link](https://forecastingresearch.org/research/ai-conditional-trees).

### 5. Rational metareasoning

- Russell & Wefald 1991. The value of computation is the expected gain in decision quality minus the cost of time, with myopic estimates. [doi:10.1016/0004-3702(91)90015-C](https://doi.org/10.1016/0004-3702(91)90015-C).
- Hay, Russell, Tolpin & Shimony 2012: computation selection as an MDP. [arXiv 1207.5879](https://arxiv.org/abs/1207.5879).
- Callaway et al. 2018, BMPS. The meta-policy approximates VOC from myopic-VOI and perfect-information features [memory]. [arXiv 1711.06892](https://arxiv.org/abs/1711.06892).
- Related, amortized "what to query next" policies trained on simulated (dreamed) data:
  - Foster et al. 2021, Deep Adaptive Design: a policy trained offline on simulations from a model, which is the "dreams train the meta-policy" pattern. [arXiv 2103.02438](https://arxiv.org/abs/2103.02438).
  - Ma et al. 2019, EDDI: information-reward feature acquisition. [arXiv 1809.11142](https://arxiv.org/abs/1809.11142).
- GFlowNets (Bengio et al. 2021): the construction order is the learned policy itself. [arXiv 2106.04399](https://arxiv.org/abs/2106.04399).
- The 2025-26 dLLM unmasking policies (section 1) are, in effect, learned metareasoning policies.
- **Synthesis for ShipHunt** (our hypothesis): VOC theory says to reveal next whatever most changes the decision-relevant forecast per unit cost. For TD *training*, the analogous criterion is the reveal that most reduces the expected error of the bootstrapped target. That quantity is learnable from outcomes (E2).

---

## E. How this maps to Kelvin's two systems

- **Reveal-chain TD (round 1)** is any-order conditional modeling (as in Uria, Shih and the MDMs), trained by self-distillation along random reveal chains. Its relatives:
  - TD-VAE (jumpy belief-to-belief targets);
  - the "learn from your own latents" theory, applied in output space rather than latent space (Korchinski).

  It inherits both the hard-subproblem cost of random order (Kim) and the cross-order inconsistency risk (Young, Shih). Its bootstrapping is the probable source of its data efficiency (our hypothesis, consistent with Korchinski and Sutton 1988).
- **The round-2 dreaming world model** is a generic joint model sampled in some order. The literature says the dream order is the cheapest lever: adaptive or learned order beats random by large margins on constraint-like data. The world model need not change.
- **The real-world analogue:** levels of abstraction come for free as *questions at different levels* (winner vs county turnout). The order of sampling or conditioning is a VOC meta-policy over those questions. Coherence across levels is a label-free signal with theoretical support (MinT) and correlational LLM evidence (Paleka). It is not yet shown to improve LLM accuracy causally.
