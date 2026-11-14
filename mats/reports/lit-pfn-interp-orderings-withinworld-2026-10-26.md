# PFN interpretability, ordering consistency, within-world compute: verified literature

2026-10-26. This is literature research for the MATS plan to train small PFN-style transformers from scratch on synthetic information-aggregation worlds.

**How this was checked.** I downloaded every arXiv paper below as a PDF on 2026-10-26 and extracted the text with pypdf. Titles, authors, dates and venue comments come from the arXiv API. TabPFN v2 was read from the Nature HTML. The `tabpfn-extensions` code was read at commit `f3dac23`. Every quote is copied from that text. I only re-inserted spaces that pypdf drops between words (for example "datasetD" became "dataset D"), and "…" marks a cut. Locators are "Abstract", a § (section) number where the PDF shows one, or "p.N" (page of the arXiv PDF). I left out any paper I could not open. I also left out claims from earlier outbox notes that I did not re-check today. "[ours]" marks my own inference, not a claim from a paper.

---

## Q1. How PFN-style and in-context Bayesian learners aggregate context: mechanism and probing

### A. Small transformers whose exact posterior is known ("wind tunnels")

**1. Agarwal, Dalal & Misra, "The Bayesian Geometry of Transformer Attention" (Paper I), arXiv 2512.22471 (v5).** https://arxiv.org/abs/2512.22471
- Finding: 2.7M-parameter transformers match closed-form posteriors on bijection elimination and HMM filtering. Mamba, LSTMs and MLPs each fail on different "inference primitives". Ablating one Layer-0 head is uniquely destructive.
- Quotes:
  - Abstract: "small transformers reproduce exact Bayesian posteriors for filtering and hypothesis elimination with 10−3–10−4 bit accuracy, while capacity-matched MLPs fail by orders of magnitude."
  - Abstract: "Geometric diagnostics reveal orthogonal key bases, low-dimensional value manifolds parameterized by posterior entropy"
  - p.10, §4.1: "Headwise ablations (Figure 6) identify a single Layer 0 hypothesis-frame head whose removal is uniquely destructive"
  - p.8, architecture: "learned absolute positional embeddings"
- Relevance: this is the closest template for a mechanistic study with a known posterior. These are causal sequence models with positional embeddings, though, not set-PFNs.

**2. Agarwal, Dalal & Misra, "Gradient Dynamics of Attention: How Cross-Entropy Sculpts Bayesian Manifolds" (Paper II), arXiv 2512.22473.** https://arxiv.org/abs/2512.22473
- Finding: a first-order analysis of how cross-entropy training shapes attention. It reads attention weights as soft responsibilities (an E-step) and value vectors as prototypes (an M-step).
- Quote, Abstract: "attention weights implement an E-step (soft responsibilities), while values implement an M-step (responsibility-weighted prototype updates)."
- Relevance: a candidate account of how a PFN's aggregation circuit forms during training. It is an analysis, not a measurement on PFNs.

**3. Dalal, Misra & Parekh, "Bayesian Wind Tunnels for Model Selection", arXiv 2607.19379.** https://arxiv.org/abs/2607.19379
- Finding: a 2.8M transformer tracks the exact posterior over hypothesis classes. It fails when the discriminating statistic needs arithmetic on opaque symbols, and this failure persists at 316M parameters.
- Quote, Abstract: "when the discriminative statistic requires arithmetic … model selection succeeds with integer tokens but fails completely with opaque symbols, and this boundary persists under 112× scaling (2.8M to 316M parameters)."
- Relevance: Bayesian model selection in context can be checked exactly. Its failures are set by whether the needed statistic is computable from the inputs, not by scale.

**4. Aswadi, Ma & Wei, "What does a Bayes-filtered transformer believe? A predictive Monte Carlo approach", arXiv 2607.17060.** https://arxiv.org/abs/2607.17060
- Finding: predictive resampling ("predictive Monte Carlo") used as an interpretability tool. It recovers the trained model's implicit prior and posterior over the latent task from next-token generation alone.
- Quote, Abstract: "This prediction-space comparison is fragile: different posteriors can share the same posterior-mean predictions. We use predictive Monte Carlo (PMC) as a general interpretability tool for any BFT: using only next-token generation, PMC returns an approximation to the implicit prior and posterior over the latent task"
- Relevance: a behavioural, latent-space readout of "what the model believes". It also bridges to the Q2 martingale tools.

### B. TabPFN and tabular-foundation-model internals

**5. Biloš, Wilson, Schneider & Nevmyvaka, "A Mechanistic Study of Tabular Foundation Models", arXiv 2605.21288.** https://arxiv.org/abs/2605.21288
- Finding: TabPFNv2 classifies with an attention-weighted vote over context labels at layer 9, confirmed causally. TabICLv2 uses a class-prototype readout instead. Approximate invariances trace to specific positional parameters.
- Quotes:
  - §2.1: "TabPFNv2 keeps mean probe accuracy near chance through most of the layers, then jumps sharply between L8 and L9, from 0.47 to 0.78"
  - §3.1: "Read off the attention weights at L9 and take a weighted vote over context labels. The vote tracks the model's predicted probabilities at mean Pearson r=0.89"
  - §3.1: "Replacing the L9 attention pattern with a uniform one collapses accuracy from 0.87 to 0.49 (the majority-class baseline)"
  - §3.1: "The rule therefore acts as a learned similarity over the context set, not a metric kNN on the final features."
  - Abstract: "Each model's permutation invariances trace to specific positional parameters whose removal preserves accuracy and makes approximate invariance exact."
- Relevance: this is the strongest existing mechanism result for a real PFN. Aggregation works as a learned kernel vote, not as an explicit sufficient statistic.

**6. Rubachev, Kotelnikov, Kartashev & Babenko, "On Finetuning Tabular Foundation Models", arXiv 2506.08982.** https://arxiv.org/abs/2506.08982
- Finding: fine-tuning works by making last-layer query–key similarity track target similarity. This fits a retrieval-style aggregation mechanism.
- Quote, Abstract: "after gradient-based adaptation, the dot products of the query-representations of test objects and the key-representations of in-context training objects more accurately reflect their target similarity."
- Relevance: interpretability that also bears on Q3. Adaptation sharpens the aggregation kernel.

**7. Gupta, Sethi & Kumar, "TabPFN Through The Looking Glass", arXiv 2601.08181.** https://arxiv.org/abs/2601.08181
- Finding: linear probes on TabPFN v2 recover the coefficients of z=αx+βy and intermediate arithmetic terms from middle layers.
- Quotes:
  - p.2: "Linear coefficients and intermediate arithmetic terms are often linearly decodable in the middle layers"
  - p.3, Fig. 1 caption: "The probe R2 sees a sharp increase at layer 6 and drops off at the last layer"
- Relevance: weak evidence (a short workshop-style paper) that parametric task quantities are present in the representations.

**8. Gupta, Kumar, Mandal & Deshpande, "Where Computation Lives Inside TabPFN", arXiv 2606.12917 (FMSD workshop at ICML 2026).** https://arxiv.org/abs/2606.12917
- Finding: activation patching on TabPFN-2.5's feature-wise attention heads finds one dominant head. Steering vectors do not transfer.
- Quotes, Abstract:
  - "one head's causal necessity dominates that of the others by 2–5× at peak layer"
  - "TabPFN's in-context learning mechanism … encodes task structure through context-dependent attention rather than the stable parametric directions that make steering tractable in language models."
- Relevance: the paper's diagnosis of why steering fails is not firmly established. It covers only two synthetic regression tasks.

**9. Ye, Liu & Chao, "A Closer Look at TabPFN v2", arXiv 2502.17361.** https://arxiv.org/abs/2502.17361
- Quote, Abstract: "TabPFN v2 can infer attribute relationships even when provided with randomized attribute token inputs"
- Relevance: feature identity is inferred in context. This matters if worlds have exchangeable signal sources.

### C. In-context regression mechanism papers (GPT-style, not PFN-style)

**10. Garg, Tsipras, Liang & Valiant 2022, arXiv 2208.01066.** https://arxiv.org/abs/2208.01066
- Quotes:
  - Abstract: "standard Transformers can be trained from scratch to perform in-context learning of linear functions … with performance comparable to the optimal least squares estimator."
  - p.4: "We use a decoder-only Transformer architecture … from the GPT-2 family"
- Relevance: behaviour-level match to known estimators, not mechanism. The model is causal with positional encoding.

**11. Akyürek, Schuurmans, Andreas, Ma & Zhou, ICLR 2023, arXiv 2211.15661.** https://arxiv.org/abs/2211.15661
- Quote, Abstract: "trained in-context learners closely match the predictors computed by gradient descent, ridge regression, and exact least-squares regression, transitioning between different predictors as transformer depth and dataset noise vary, and converging to Bayesian estimators for large widths and depths … learners' late layers non-linearly encode weight vectors and moment matrices."
- Relevance: probing for sufficient statistics (XᵀY, w_OLS) is the template for probing a PFN.

**12. von Oswald et al. 2022, arXiv 2212.07677.** https://arxiv.org/abs/2212.07677
- Quote, Abstract: "equivalence of data transformations induced by 1) a single linear self-attention layer and by 2) gradient descent (GD) on a regression loss … the weights found by optimization match the construction."
- Relevance: an exact mechanism, but only for linear-attention-only models.

**13. Fu, Chen, Jia & Sharan, NeurIPS 2024, arXiv 2310.17086.** https://arxiv.org/abs/2310.17086
- Quote, Abstract: "predictions from successive Transformer layers closely match different iterations of Newton's Method linearly, with each middle layer roughly computing 3 iterations"
- Relevance: a layer-by-layer estimator correspondence, competing with the GD story.

**14. Bai, Chen, Wang, Xiong & Mei 2023, "Transformers as Statisticians", arXiv 2306.04637.** https://arxiv.org/abs/2306.04637
- Quote, Abstract: "A single transformer can adaptively select different base ICL algorithms"
- Relevance: a construction (existence proof) for in-context algorithm selection, not a measurement on trained models.

**15. Raventós, Paul, Chen & Ganguli 2023, arXiv 2306.15063.** https://arxiv.org/abs/2306.15063
- Quote, Abstract: "Below this threshold, the pretrained transformer cannot solve unseen regression tasks, instead behaving like a Bayesian estimator with the non-diverse pretraining task distribution as the prior. Beyond this threshold, the transformer significantly outperforms this estimator; its behavior aligns with that of ridge regression"
- Relevance: which estimator a from-scratch model implements depends on world diversity. This bears directly on the design of the world prior.

**16. Panwar, Ahuja & Goyal, ICLR 2024, arXiv 2306.04891.** https://arxiv.org/abs/2306.04891
- Quotes, Abstract:
  - "Where Bayesian inference is tractable, we find evidence that high-capacity transformers mimic the Bayesian predictor."
  - "transformers can learn to generalize to new function classes that were not seen during pretraining. This involves deviation from the Bayesian predictor."

**17. Hill, Eyre & Creager, "Transformers Don't In-Context Learn Least Squares Regression", arXiv 2507.09440 (ICML 2025 workshop).** https://arxiv.org/abs/2507.09440
- Quote, Abstract: "transformers trained for ICL fail to generalize after shifts in the prompt distribution, a behaviour that is inconsistent with the notion of transformers implementing algorithms such as OLS."
- Relevance: a counterweight. Matching an estimator in distribution does not show the model implements it.

**18. Shai et al. 2024, "Transformers represent belief state geometry in their residual stream", arXiv 2405.15943.** https://arxiv.org/abs/2405.15943
- Quote, Abstract: "belief states are linearly represented in the residual stream of transformers, even in cases where the predicted belief state geometry has highly nontrivial fractal structure."
- Relevance: the posterior itself, not just the prediction, can be read out linearly. This is shown for HMM next-token models, not set-PFNs.

### D. PFN amortization quality

**19. Hülskamp, Kobialka, Sommer, Nagler & Rügamer, "How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?", arXiv 2610.13764.** https://arxiv.org/abs/2610.13764
- Quote, Abstract: "PFNs generally recover the targeted PPD very well. Nevertheless, we find that parts of the residual amortization error persist, especially in rare settings under the prior pretraining distribution, even when scaling the model size, which manifests in disagreement between independently pretrained PFNs"
- Relevance: amortization error can be measured against a reference posterior predictive distribution (PPD), and ensembles over pretraining runs reduce it.

**20. Nagler, "Statistical Foundations of Prior-Data Fitted Networks", ICML 2023, arXiv 2305.11097.** https://arxiv.org/abs/2305.11097
- Quote, Abstract: "A predictor's variance vanishes if its sensitivity to individual training samples does and the bias vanishes only if it is appropriately localized around the test feature. The transformer architecture used in current PFN implementations ensures only the former."

### What is known (Q1)
- Behaviour: small transformers can match exact posteriors to about 1e-3 bits (items 1, 3), and GPT-style in-context regressors match ridge, least squares or Bayes estimators (items 10, 11, 15, 16).
- Mechanism: linear-attention models implement GD (item 12). In real TabPFN v2, the readout is a late-layer learned-similarity vote (item 5), which fine-tuning sharpens (item 6).
- Representations: posteriors are linearly present in HMM next-token models (item 18). The implicit posterior can be recovered behaviourally by predictive Monte Carlo (item 4).

### Open gaps (Q1)
1. No one has done a mechanistic study of a set-PFN (no positional encoding, test rows attending only to context) on a prior whose posterior is in closed form. The wind tunnels use causal models with absolute positional embeddings (item 1, p.8). The TabPFN studies (items 5, 7, 8) have no reference posterior. So no paper says which heads in a PFN compute which sufficient statistic, or where in a PFN the posterior error enters.
2. "Vote/kernel readout" versus "explicit sufficient statistic" is unresolved for PFNs. Item 5 finds a similarity vote, while item 7 decodes linear coefficients in middle layers. No paper varies the prior (for example conjugate versus non-conjugate, or few versus many latent parameters) to test when a PFN switches between the two. Item 15 does this only for GPT-style linear regression.
3. No paper checks whether a probe-decoded latent posterior agrees with the model's own predictive. Items 4 and 18 each give one readout (behavioural and linear respectively). Nobody has compared the two inside one model, which would be an internal coherence test for PFNs.

---

## Q2. How PFNs are trained and queried; joint predictions; consistency across orderings

### A. Confirmed training and query mechanics

**21. Müller, Hollmann, Pineda Arango, Grabocka & Hutter, "Transformers Can Do Bayesian Inference", ICLR 2022, arXiv 2112.10510.** https://arxiv.org/abs/2112.10510
- **Set input, no positional encoding.**
  - §4, p.5: "a Transformer encoder without positional encodings, which makes it invariant to permutations in the dataset D."
  - App. E.1, p.18: "we remove positional encodings and feed (x,y) pairs together to the Transformer as a sum of their encodings"
- **Query rows see only the context.** App. E.1, p.18: "we use an attention mask that allows attention from every input example to every other input example and allows attention from every query points to every input example."
- **Riemann (bar) output head.**
  - §4, p.5: "a discretized continuous distribution that we call Riemann Distribution. PDFs of this distribution are bar plots"
  - §4, p.5: buckets chosen "such that they all have the same probability in prior-data"
- **Single forward pass.** Abstract: "PFNs make probabilistic predictions for arbitrary other data points in a single forward propagation"
- **Training objective.** p.4: "We train this model by cross-entropy over samples drawn from the prior."
- **Ablation of invariance.** App. E.3, p.18: "the effect of making our architecture permutation invariant only slowly increases with sequence length."
- Code: https://github.com/automl/TransformersCanDoBayesianInference. Not re-read today.

**22. Hollmann, Müller, Eggensperger & Hutter, TabPFN (v1), ICLR 2023, arXiv 2207.01848.** https://arxiv.org/abs/2207.01848
- §3, p.3–4: "predictions that ensemble 32 forward passes of datasets modified by a power transformation (applied with probability 0.5) and rotating the indices of feature columns and class labels"
- Relevance: order-averaging over feature and label permutations was part of the default from v1 on.

**23. Hollmann et al., "Accurate predictions on small data with a tabular foundation model" (TabPFN v2), Nature 637 (2025).** https://www.nature.com/articles/s41586-024-08328-6
- **Two-way attention, test rows attend only to train rows.** Methods, "Details on the neural architecture":
  - "Each layer first has an attention over features, followed by an attention over samples"
  - "our inter-sample attention does not allow the test samples to attend to each other but only to the training data."
- **Not fully permutation invariant, so ensembles shuffle features.** Methods, "Inference details": "As our models are not fully permutation invariant, for each ensemble member, we shuffle the feature order, approximating order invariance"
- **Joint density over features, autoregressive, averaged over orders.** Methods, "Density estimation":
  - "The feature order of the joint density factorization influences the estimated densities. To reduce variance from this source, we apply a permutation sampling approximation of Janossy Pooling at inference time, in which we average the outputs of N_j feature permutations, with N_j = 24 in our experiments"
  - The factorization is ∏_j q_θ(x_j | x_<j, D_:,<j) · q_θ(y | x, D). Methods: "we condition the prediction of the first feature x_1, on a feature with random noise"
- **Generation.** Methods, "Synthetic data generation": "by simply following the factorization in equation (5) and sampling each feature step by step."
- **Training loss.** Methods, "Training details": "The training loss of any PFN is the cross-entropy between the targets of held-out samples of synthetic datasets and the model prediction."
- Relevance: this is the official recipe for joint and generative use. It is an unsupervised "mode" built from feature-by-feature conditionals of the supervised model, not a separate head.

**24. PriorLabs `tabpfn-extensions`, `src/tabpfn_extensions/unsupervised/unsupervised.py` (commit f3dac23).** https://github.com/PriorLabs/tabpfn-extensions
- **Density.** `outliers()` docstring: "Calculate outlier scores as the log of the arithmetic mean (AM) of the densities across the permutations used to approximate the chain rule." Default `n_permutations=10`.
- **Generation.** `generate_synthetic_data()` docstring: "Samples are generated feature by feature in a single pass, with each feature conditioned on previously generated features." Default `n_permutations=3`.
- **What the code does** (my reading, not a quote): features are generated in column order. At each column, it averages the predictive distributions obtained under random orderings of the conditioning columns, using `average_bar_distributions_into_this` or the mean of class probabilities, and then samples.
- Relevance: density is averaged over full chain-rule orders, but generation keeps one fixed column order. These are two different combiners.

### B. Joint predictions from set-based models, and their order dependence

**25. Bruinsma et al., "Autoregressive Conditional Neural Processes", ICLR 2023, arXiv 2303.14468.** https://arxiv.org/abs/2303.14468
- Quotes:
  - Abstract: "we autoregressively define a joint predictive distribution using the chain rule of probability"
  - p.4: "AR CNPs give up the fundamental property of consistency … permuting x and introducing or marginalising target points can change the distribution. This violates the conditions of the Kolmogorov extension theorem"
- Relevance: the AR-from-marginals recipe and its loss of consistency are stated explicitly.

**26. Hassan et al., "Efficient Autoregressive Inference for Transformer Probabilistic Models", ICLR 2026, arXiv 2510.09477.** https://arxiv.org/abs/2510.09477
- Quotes:
  - p.3, fn. 2: "Eq. (2) imposes a specific factorization order. While fixing the order can be a valid modeling choice under certain circumstances, this breaks permutation invariance. In cases where permutation invariance is required, a Monte Carlo approximation can be obtained by averaging over multiple target orderings."
  - p.6: "autoregressive predictive density estimates are order-dependent; to recover approximate permutation invariance, we average the predictive densities over multiple buffer orderings (Murphy et al., 2019)."
  - App. H.3: "increasing the number of permutations reduces the RMSE of the estimated joint density relative to the true joint"
  - App. H.1: a decoder transformer that reads the context as a sequence "performs substantially worse than our method in the GP task"
- Relevance: the only PFN/TNP paper I found that reports error against a true joint as a function of the number of orderings.

**27. Tugnoli, De Lorenzo, Virgolin et al., "Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure", UAI 2026, arXiv 2603.10254.** https://arxiv.org/abs/2603.10254
- Quotes:
  - p.2: "Although the model averages predictions over random permutations of the conditioning set, this mitigates but does not eliminate order sensitivity: during generation, each variable can only condition on features that appear earlier in the sequence."
  - p.2, contribution 1: "TabPFN synthetic data quality depends heavily on the order of the features in the input data … with this sensitivity persisting even at large training sizes."
- Relevance: measured order dependence in TabPFN generation. Their fix chooses the order from a causal DAG.

### C. Martingale and exchangeability checks

**28. Fong, Holmes & Walker, "Martingale posterior distributions", JRSSB 2023, arXiv 2103.15671.** https://arxiv.org/abs/2103.15671
- Quote, Abstract: "the Bayesian would thus assign a predictive distribution on the missing Y_{n+1:∞} conditional on Y_{1:n}, which then induces a distribution on the parameter … this distribution can be sampled through a computational scheme we name predictive resampling."

**29. Falck, Wang & Holmes, "Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective", ICML 2024, arXiv 2406.00793.** https://arxiv.org/abs/2406.00793
- Quotes:
  - Abstract: "we provide evidence for violations of the martingale property, and deviations from a Bayesian scaling behaviour of uncertainty"
  - §2.2, p.4: "Exchangeability guarantees the invariance of predictions to the ordering of the observations Z_{1:n}, but also with respect to the order of future imputations"
  - App., p.15: "As nonexchangeable models may demonstrate different behaviour on different permutations of the same dataset … we permute the observations when generating each sample path"
- Relevance: gives ready-made test statistics. They were applied to LLMs, not PFNs.

**30. Nagler & Rügamer, "Uncertainty Quantification for PFNs using Martingale Posteriors", arXiv 2505.11325.** https://arxiv.org/abs/2505.11325
- Quote, §3, p.4: "We propose to use the PPD implied by the PFN only as a starting point for the MP sampling scheme … iteratively update it using a nonparametric forward sampling scheme that ensures the martingale property … without requiring the PFN to provide coherent posterior updates"
- Relevance: the authors avoid the PFN's own updates because, citing Falck et al., transformers "substantially deviate from the martingale property".

**31. Ng, Fong, Frazier, Knoblauch & Wei, "TabMGP: Martingale Posterior with TabPFN", ICML 2026, arXiv 2510.25154.** https://arxiv.org/abs/2510.25154
- Quotes:
  - p.2: "Even when high-capacity predictive rules like TabPFN do not strictly satisfy formal sufficient conditions such as the martingale property, the resulting posteriors remain empirically stable"
  - App., p.13: "the predictive rule induced by TabPFN does not strictly adhere to the a.c.i.d. condition as N increases. However, at these scales (N >1000), any measurable drift falls below the numerical resolution"
- Relevance: the only direct martingale-type diagnostic of a PFN's own predictive resampling that I found.

**32. Lee, Yun, Nam, Fong & Lee, "Martingale Posterior Neural Processes", ICLR 2023, arXiv 2304.09431.** https://arxiv.org/abs/2304.09431
- Quote, Abstract: "we equip a NP with a predictive distribution implicitly defined with neural networks and use the corresponding martingale posteriors as the source of uncertainty."

**33. Ye & Namkoong, "Exchangeable Sequence Models Quantify Uncertainty Over Latent Concepts", arXiv 2408.03307.** https://arxiv.org/abs/2408.03307
- Quotes:
  - Abstract: "epistemic uncertainty over latent environments is captured by variation in predicted future observations"
  - p.3: "we compare data augmentation …, loss-based regularizers …, and causal masking schemes … and find the direct modification to the causal masks is the most effective."

**34. Mittal, Li, Yen, Guetta & Namkoong, "Architectural and Inferential Inductive Biases For Exchangeable Sequence Modeling", arXiv 2503.01215.** https://arxiv.org/abs/2503.01215
- Quotes, Abstract:
  - "a fundamental limitation of the prevalent single-step generation approach: inability to distinguish between epistemic and aleatoric uncertainty"
  - "recently proposed Transformer architectures for exchangeable sequences … cannot guarantee exchangeability despite introducing significant computational overhead."

**35. Chlon et al., "LLMs are Bayesian, In Expectation, Not in Realization", arXiv 2507.11768 (v3).** https://arxiv.org/abs/2507.11768
- Status: a contested preprint. The title printed in the PDF differs slightly from the arXiv metadata title.
- Quote, Abstract: "the expected regret of a single admissible ordering decomposes into the regret of the order-averaged predictor plus an order-averaging gain. Exchangeability violations are therefore not binary refutations; they are priced by log loss."
- Relevance: gives a log-loss accounting of what order-averaging buys.

### Not found
I found no PFN paper that studies "reasoning tokens". The PFN papers above all predict in one forward pass. One exception is TabPFN-3's closed "Thinking mode" (see Q3, item 46), whose mechanism is not disclosed.

### Open gaps (Q2)
1. **Order spread as a measure of distance from Bayes is unstudied.**
   - TabPFN v2 averages over 24 feature orders without reporting the spread.
   - Item 26 reports RMSE against the true joint as a function of the number of orders, on one task.
   - Item 27 measures order sensitivity through downstream generation quality.
   - [ours] No one has related the order-variance of a PFN's joint log-likelihood to its true KL from the exact joint posterior predictive, across context sizes and numbers of targets. In our worlds the exact joint is computable, so this can be done.
2. **The choice of combiner has not been analysed.**
   - TabPFN density uses the arithmetic mean of chain-rule densities over orders (item 24).
   - TabPFN generation averages per-step conditionals over conditioning-set orders, then samples in a fixed column order (item 24).
   - Hassan averages densities over buffer orders (item 26).
   - [ours] Nobody has checked which of these combiners yields an exchangeable or Kolmogorov-consistent joint, or how far each is from the Bayes joint.
3. **There is no martingale or exchangeability test for multi-target PFN joints, and no training-time fix inside PFNs.**
   - Martingale diagnostics exist for single-target predictive resampling with TabPFN (item 31: small but nonzero drift) and for LLMs (item 29).
   - Exchangeability regularizers exist for sequence models (item 33).
   - [ours] Nobody has applied the c.i.d. or martingale test to a PFN's AR joint over several targets of one world, or trained a PFN with an order-consistency or martingale penalty and measured the effect on joint KL.

---

## Q3. Using more test-time compute to learn more about one world

### A. Test-time training and fine-tuning on the observed part of the same task

**36. Sun et al., "Test-Time Training with Self-Supervision…", ICML 2020, arXiv 1909.13231.** https://arxiv.org/abs/1909.13231
- Quote, Abstract: "We turn a single unlabeled test sample into a self-supervised learning problem, on which we update the model parameters before making a prediction."

**37. Akyürek et al., "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning", arXiv 2411.07279.** https://arxiv.org/abs/2411.07279
- Quotes:
  - §3.1, p.3: "Leave-one-out tasks … For each pair (x_j,y_j), we exclude it from the set of demonstrations and treat it as a "test" example within the newly formed synthetic task"
  - Abstract: "performing TTT with in-context examples yields up to 6× higher accuracy compared to fine-tuned baselines"
  - p.3: the final prediction comes from "an augmented inference strategy based on self-consistency"
- Relevance: the leave-one-out recipe maps one-to-one onto PFN context/query splits within one world.

**38. Gozeten, Ildiz, Zhang, Soltanolkotabi, Mondelli & Oymak, "Test-Time Training Provably Improves Transformers as In-Context Learners", ICML 2025, arXiv 2503.11842.** https://arxiv.org/abs/2503.11842
- Quotes:
  - Abstract: "we train a transformer model on the in-context demonstrations provided in the test prompt … linear transformers when the update rule is a single gradient step"
  - Abstract: "TTT significantly reduces the required sample size for tabular classification (3 to 5 times fewer)" (TabPFN)
- Relevance: the only theory of TTT for in-context learning, plus a TabPFN experiment.

**39. Kuwataka & Suzuki, "Test time training enhances in-context learning of nonlinear functions", arXiv 2509.25741.** https://arxiv.org/abs/2509.25741
- Quote, Abstract: "TTT enables the single-layer transformers to adapt to both the feature vector and the link function, which vary across tasks. This creates a sharp contrast with ICL alone, which is theoretically difficult to adapt to shifts in the link function."

**40. den Breejen, Bae, Cha & Yun, "Fine-tuned In-Context Learning Transformers are Excellent Tabular Data Classifiers", arXiv 2405.13396.** https://arxiv.org/abs/2405.13396
- Quotes:
  - p.5: "Every gradient descent step, we randomly draw a 80/20 support and query split from the training dataset."
  - p.2: "Fine-tuned ICL-transformers can create complex decision boundaries"
- Relevance: within-world training by splitting one dataset into context and query sets.

**41. Rubachev et al. 2025 (item 6 above).**
- Quote, Abstract: "we managed to finetune TabPFNv2 on datasets with up to 50K objects, observing performance improvements on almost all tasks … on datasets with gradual temporal shifts and rich feature sets, TabPFNv2 is less stable"

**42. Thomas et al., "Retrieval & Fine-Tuning for In-Context Tabular Models" (LoCalPFN), arXiv 2406.05207.** https://arxiv.org/abs/2406.05207
- Quotes:
  - Fig. 1 caption, p.2: "TabPFN – even when using the entire training data as context – underfits and cannot classify patterns such as three pairs of concentric circles"
  - Abstract: "adapt the transformer to a local subset of the data by collecting nearest neighbours, and then perform task-specific fine-tuning with this retrieved set of neighbours in context."

**43. Feuer et al., "TuneTables", NeurIPS 2024, arXiv 2402.11137.** https://arxiv.org/abs/2402.11137
- Quote, Abstract: "a parameter-efficient fine-tuning strategy for PFNs that compresses large datasets into a smaller learned context … optimizing fewer than 5% of TabPFN's parameters."

**44. Xu et al., "Mixture of In-Context Prompters for Tabular PFNs" (MixturePFN), arXiv 2405.16156.** https://arxiv.org/abs/2405.16156
- Quote, Abstract: "extends nearest-neighbor sampling … and uses bootstrapping to finetune said model on the inference-time dataset."

**45. Liu & Ye, "TabPFN Unleashed" (BETA), arXiv 2502.02527.** https://arxiv.org/abs/2502.02527
- Quote, Abstract: "most approaches focus either on reducing bias or variance, often neglecting the need to address the other side"

### B. Larger worlds: context extension, subsampling and ensembling

**46. Grinsztajn et al., "TabPFN-3: Technical Report", arXiv 2605.13986.** https://arxiv.org/abs/2605.13986
- Quotes:
  - p.3: "we scaled beyond a hundred thousand rows to one million rows"
  - §2.6: "TabPFN-3-Plus (Thinking) applies additional inference-time computation on top of TabPFN-3-Plus … while only relying on TabPFN, without using LLMs, real data, internet search, or any other model."
- Relevance: a commercial claim that test-time compute pays off for PFNs. The mechanism is not disclosed.

**47. Qu, Holzmüller, Varoquaux & Le Morvan, "TabICL", ICML 2025, arXiv 2502.05564.** https://arxiv.org/abs/2502.05564
- Quote, Abstract: "pretrained on synthetic datasets with up to 60K samples and capable of handling 500K samples … On 53 datasets with over 10K samples, TabICL surpasses both TabPFNv2 and CatBoost"

**48. Feuer, Hegde & Cohen, "Scaling TabPFN: Sketching and Feature Selection", NeurIPS 2023 TRL workshop, arXiv 2311.10609.** https://arxiv.org/abs/2311.10609
- Quote, p.3: "In most cases, random sample selection is sufficient for optimal performance."

**49. Ye, Liu & Chao (item 9).**
- Quote, p.2: "these limitations can be effectively addressed through carefully designed post-hoc divide-and-conquer strategies, reminiscent of test-time scaling techniques developed for large language models"

**50. Hülskamp et al. (item 19).**
- Quote, Abstract: Bayesian model averages over PFN weights "can consistently reduce this uncertainty and improve predictions up to full-size TabICLv2 models"
- Relevance: more compute spent on ensembles over pretraining seeds, not on the world itself.

### C. In-context self-training and in-context versus gradient adaptation

**51. Li, Chang, Kara et al., "When and How Unlabeled Data Provably Improve In-Context Learning", arXiv 2506.15329.** https://arxiv.org/abs/2506.15329
- Quotes:
  - Abstract: "multilayer or looped transformers can effectively leverage unlabeled data by implicitly constructing estimators … connections to Expectation Maximization, an iterative pseudo-labeling algorithm … we propose looping off-the-shelf tabular foundation models"
  - Algorithm 1 (p.12): "LoopTabFM: Looping Tabular FM with Soft Pseudo-labels and Risk-aware Updates"

**52. Mosbach et al., "Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation", Findings of ACL 2023, arXiv 2305.16938.** https://arxiv.org/abs/2305.16938
- Quote, Abstract: "fine-tuned language models can in fact generalize well out-of-domain. We find that both approaches generalize similarly; they exhibit large variation"

### D. Within-world adaptation when the one world is large (sequence-model side)

**53. Krause, Kahembwe, Murray & Renals, "Dynamic Evaluation of Neural Sequence Models", arXiv 1709.07432.** https://arxiv.org/abs/1709.07432
- Quote, Abstract: "Models are adapted to recent history via a gradient descent based mechanism, causing them to assign higher probabilities to re-occurring sequential patterns."

**54. Hardt & Sun, "Test-Time Training on Nearest Neighbors for Large Language Models", ICLR 2024, arXiv 2305.18466.** https://arxiv.org/abs/2305.18466
- Quote, Abstract: "retrieving and training on as few as 20 neighbors, each for only one gradient iteration, drastically improves performance"

**55. Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", arXiv 2407.04620.** https://arxiv.org/abs/2407.04620
- Quote, Abstract: "make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences"
- Relevance: an architecture where within-world gradient learning is the inference mechanism itself.

### What is known (Q3)
- Gradient adaptation on the observed part of one dataset reliably helps PFNs on real data (items 38, 40–45). It reduces underfitting of complex boundaries (items 40, 42) and works by sharpening the query–key retrieval kernel (item 6).
- Retrieval and subsampled-context ensembles extend PFNs to large worlds (items 42, 44, 47–49).
- In-context pseudo-labelling helps semi-supervised TabPFN (item 51).
- All of these results are on real data, measured by accuracy or AUC. None is measured against an exact posterior.

### Open gaps (Q3)
1. **No one has tested TTT or fine-tuning of a PFN against the exact Bayes predictive of one world.**
   - [ours] If the PFN were exactly Bayes for the prior that generated the world, fine-tuning on that world's own context/query splits could only add error. It would reuse the same data twice and overfit.
   - [ours] So any gain measures either amortization error (item 19) or prior mismatch (items 38, 39 frame it as "alignment").
   - Nobody has separated the two in a controlled setting: an in-prior world versus an out-of-prior world, measuring KL to the exact posterior predictive and calibration, not accuracy.
2. **There is no compute-matched comparison of the ways to spend test-time compute on one large world.** The candidates are:
   - (a) a longer context;
   - (b) an ensemble over subsampled or retrieved contexts;
   - (c) order and feature-permutation ensembles;
   - (d) TTT or fine-tuning with leave-one-out or 80/20 splits;
   - (e) in-context pseudo-label looping.

   Each paper tests one or two of these on real data. Nobody has put them on one FLOP axis against the Bayes optimum. TabPFN-3's "Thinking" result (item 46) cannot be reproduced because its method is not disclosed.
3. **The case that self-generated data adds no information has not been tested on PFNs.**
   - [ours] Under exact Bayes, conditioning on the model's own predictive samples, whether pseudo-labels (item 51) or predictive resampling (items 28, 31), leaves expected predictions unchanged. That is the martingale property.
   - [ours] So the accuracy gain of looping or self-training on a synthetic world with a known posterior should equal the model's martingale violation (amortization error). No paper measures both quantities on the same model.
   - This links Q2 gap 3 to Q3. It is a precise, cheap experiment in our setting.

---

**Excluded because they were not verified today:**
- Kim et al. 2025 (adaptive decoding order) and Prabhudesai 2025, both cited in earlier outbox notes.
- The Hollmann et al. TabPFN repository README.
- arXiv 2610.13764's comparison with MCMC beyond its abstract.
