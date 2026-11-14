# L1 cluster B: attribution and pruning (worker 2)

Read 2026-09-27 22:10-22:30Z. All 9 IDs were fetched from arxiv.org (abs page + PDF) and match
the plan's titles: `verify-B.json` (title, authors, date, pages, PDF sha256). Quoted formulas and
sentences: `quotes-B.txt`. Two extra IDs verified by title only: 1611.06440, 1810.02340.
No experiments were run in this cluster, so there is no PREREG.
"Ideas 1-6" refers to CONTEXT.md. "Books" means wealth-net's `books.py` (unit income a·dR/da,
with R the output revenue settled at s = y − p; the whole backward pass is one attribution flow).

## Per paper (5 lines: built / overlap / did not do / consequence for us / source)

**Movement pruning** (Sanh, Wolf, Rush 2020, https://arxiv.org/abs/2005.07683)
- Built: a learned score S per weight, with mask TopV(S) and a straight-through estimator. This
  gives S(T) = −α Σ_t (∂L/∂W)(t) W(t) (their Eq. 3), which accumulates signed first-order movement
  over fine-tuning; there is also a soft variant with threshold τ and a sigmoid penalty on S.
- Overlap: summed over a unit's weights, S is our cumulative in-sample books with cotangent ∂L
  (idea 6's "wealth = cumulative revenue"), and it is used to gate execution (idea 6's "gate by
  wealth").
- Not done: per-unit or neuron-level structured wealth, a held-out or prequential settlement,
  multiplicative wealth, rebirth, family splits or a regime shift. Pruning is one-way and targets
  BERT fine-tuning only.
- Consequence: the M8 arm "gate by exp(η·cumulative income)" with in-sample tangent books is
  soft movement pruning with a different squashing function. Only the currency (prequential,
  LMSR, EMA) and re-entry (floor α, rebirth) are new.
- Source: quotes-B.txt "2005.07683 lines 199-218".

**Taylor importance** (Molchanov et al. 2019, https://arxiv.org/abs/1906.10771)
- Built: I_m = (Σ_{s∈S_m} g_s w_s)², equal to (∂E/∂z_m)² for a unit gate z = 1 (their Eq. 9), and
  averaged over minibatches with an EMA (momentum) between iterative pruning steps.
  Spearman >93% against a greedy leave-one-out oracle on ImageNet ResNets.
- Overlap: the per-step gate gradient is exactly the books' unit income a·dR/da, and the
  EMA-averaged score is M3's EMA currency.
- Not done: the signed version, which it reports to be *worse*. "Squared loss as a measure for
  contribution leads to better correlations with the oracle and better accuracy when compared to
  signed difference." It also found that signed OBD destabilises pruning. No wealth or
  settlement semantics.
- Consequence: M2 must include squared-EMA Taylor, (Σ g·w)², as a predictor (the plan lists
  |a·g| accumulated, which is close but not the same). A signed-wealth win is surprising given
  this result and is the thing to test.
- Source: quotes-B.txt "1906.10771" blocks. Earlier activation-level version: 1611.06440
  (verified title "Pruning Convolutional Neural Networks for Resource Efficient Inference").

**Integrated Gradients** (Sundararajan, Taly, Yan 2017, https://arxiv.org/abs/1703.01365)
- Built: path integral of gradients from a baseline x′ to x. It satisfies Sensitivity,
  Implementation Invariance, Completeness (sum = F(x) − F(x′)) and Symmetry; path methods
  correspond to Aumann-Shapley cost sharing.
- Overlap: idea 2's "Shapley-like" wealth and T1 lemma 2 (exact LMSR = log-score gain from the
  prior) have the same shape: completeness from a named baseline. The 16-step LMSR path in M1(c)
  is IG over the logit with baseline ℓ0 = base rate.
- Not done: attribution to parameters or units over time, or any market or settlement reading.
  Attribution is per input only.
- Consequence: the "exact LMSR from a prior" book = IG on the log-score along the logit path.
  It is known math, so cite it rather than claim it.
- Source: quotes-B.txt "1703.01365" blocks.

**Conductance** (Dhamdhere, Sundararajan, Yan 2018, https://arxiv.org/abs/1805.12233)
- Built: IG decomposed through a hidden unit via the chain rule. Conductances of any layer sum to
  F(x) − F(x′) (layerwise conservation), and conductance satisfies linearity. It shows that
  gradient×activation (our books) overshoots and can have the wrong sign in saturated or
  thresholded units.
- Overlap: idea 6's "consumers pay suppliers, any complete cut sums to R" is conductance's layer
  completeness. Books are its one-point (non-integrated) approximation, so they conserve only to
  first order (books.py docstring).
- Not done: summing over training (wealth), gating or pruning. Evaluation used ablation scores,
  which it notes "artifactually favor gradient*activation" because single ablation is a discrete
  gradient.
- Consequence: M2 is right to include IG/conductance. When M2 uses one-at-a-time ablation as the
  oracle, apply the caveat that the oracle is biased toward a·g books.
- Source: quotes-B.txt "1805.12233" blocks.

**AttnLRP** (Achtibat et al. 2024, https://arxiv.org/abs/2402.05602)
- Built: LRP rules for transformers.
  - softmax: R_i = x_i(R_i − s_i Σ_j R_j), where the hidden bias absorbs the error.
  - bilinear products such as A·V and gates: uniform split R/N.
  - LayerNorm/RMSNorm: the scale is treated as a constant (identity rule).
  - linear layers: ε-rules.
  One backward pass, evaluated on LLaMA-2, Mixtral and ViT.
- Overlap: the premises "books leak at attention/LN" and "gates pay twice" are the problems
  AttnLRP's rules address. The bilinear 1/N rule removes gate double-payment. wealth-net's
  "LN denominators detached" is AttnLRP's normalisation rule.
- Not done: accumulation over training, currency, settlement against labels. Relevance is about
  explanation, not P&L.
- Consequence: M1(d) "try AttnLRP-rule books" is a direct port, not a novelty. Present the
  conservation fix as borrowed.
- Source: quotes-B.txt "2402.05602" blocks.

**Neuron Shapley** (Ghorbani, Zou 2020, https://arxiv.org/abs/2002.09815)
- Built: Shapley values of filters for a performance metric V(S), with non-members zeroed on a
  validation set and no retraining. Uses truncated Monte Carlo plus a multi-armed-bandit (TMAB)
  to find the top-k. Removing 30 top filters destroys Inception-v3; removing negative-Shapley
  filters repairs bias and adversarial vulnerability.
- Overlap: idea 2 (wealth = marginal information, Shapley-like), and the ablation test in idea 7
  ("remove the most negative-wealth units"). "Remove the most negative units to repair the model"
  is their headline use.
- Not done: online or cumulative accounting, cheap per-step estimates (it costs many forward
  passes) or market semantics. The value is held-out and static, at one checkpoint.
- Consequence: M2's "sampled Shapley (64 permutations)" is this method. The wealth-net
  "remove negative-wealth units" ablation result will read as a cheap proxy for Neuron Shapley.
  The comparison worth reporting is cost versus Spearman against it.
- Source: quotes-B.txt "2002.09815" blocks.

**Hydra effect** (McGrath et al. 2023, https://arxiv.org/abs/2307.15771)
- Built: in Chinchilla-7B, ablating one attention layer makes downstream layers compensate
  (self-repair), and late MLPs downregulate the max-likelihood token. Direct-effect (unembedding)
  and total-effect (resample-ablation) importance disagree in most layers.
- Overlap: books measure a direct or first-order effect, while M2's oracle measures the total
  ablation effect. Hydra predicts systematic disagreement, and more of it for units whose
  downstream can substitute. This bears on "a convex pool cannot add independent evidence" and on
  coalition additivity.
- Not done: small nets, training dynamics or pruning.
- Consequence: in M2, report the coalition additivity gap separately for heads versus MLP
  neurons, and use mean or resample ablation as well as zero ablation. A books-vs-ablation gap is
  not by itself evidence against wealth.
- Source: quotes-B.txt "2307.15771 lines 207-230".

**LCA, Loss Change Allocation** (Lan, Liu, Zhou, Yosinski 2019, https://arxiv.org/abs/1909.01440)
- Built: ΔL over training split per parameter as Σ_t ∇L_train(θ_t)_i · Δθ_{t,i}, with
  Runge-Kutta (1,4,1) correction and conservation to total ΔL. It is summable over neurons and
  layers. Findings: about 50% of parameters help per step, some whole layers hurt, and learning
  is synchronised across layers.
- Overlap: this is closest to idea 6's *time* side. LCA is cumulative "wealth" per parameter,
  summable to any big trader (partial sums), with exact conservation. It uses the displacement
  Δθ rather than θ (removal Taylor).
- Not done: held-out or prequential settlement (it uses training-set loss), unit-level gating,
  and use for pruning or routing.
- Consequence: add LCA as a currency arm in M3 or a predictor in M2. It is the "movement" dual of
  books, cheap on the S0 toy, and it answers "income vs contribution to learning". The "≈50%
  help" finding is a null to compare against wealth sign-churn.
- Source: quotes-B.txt "1909.01440" blocks.

**Lottery ticket hypothesis** (Frankle, Carbin 2018, https://arxiv.org/abs/1803.03635)
- Built: iterative magnitude pruning with a reset to θ0. Winning tickets of 10-20% size train to
  equal or better accuracy on MNIST/CIFAR, while random reinit fails.
- Overlap: M9(a)'s "lottery-ticket link" to K/Q (many random traders, a few pay). Idea 6's
  "birth/bankruptcy" is a one-shot, then reset version of this.
- Not done: any wealth or attribution criterion (magnitude only), online selection, or market
  semantics.
- Consequence: a cheap arm is "tickets chosen by final wealth vs by magnitude, retrained from
  θ0". It is not planned. No planned arm duplicates LTH.
- Source: quotes-B.txt "1803.03635 lines 104-116".

## Rows for NOVELTY.md (claim → nearest prior art → delta)

| our claim | nearest prior art (cluster B) | delta that survives |
|---|---|---|
| unit income a·dR/da, accumulated = wealth | movement pruning S = −Σ g·W (2005.07683); Taylor gate (Σgw)² EMA (1906.10771) | only the settlement (prequential/LMSR vs in-sample loss) and the multiplicative or Kelly reading |
| books conserve across any cut (fractal firm) | conductance layer completeness (1805.12233); LRP/AttnLRP conservation (2402.05602); LCA conservation over time (1909.01440) | combining cut-conservation with time-accumulation per unit; none of these do both |
| exact LMSR from a named prior = log-score gain | IG completeness / Aumann-Shapley (1703.01365) | a restatement; cite it |
| wealth ≈ marginal information (Shapley-like) | Neuron Shapley (2002.09815) | cheap, online, per-step, versus expensive, static and held-out |
| remove negative-wealth units repairs the model | Neuron Shapley "remove culprit filters" | the online, cumulative version only |
| gate execution by wealth | soft movement pruning (threshold on accumulated movement) | exp-weights with floor, exit price, rebirth, regime shift |

## Flags for NOVELTY-FLAGS.md (planned arms prior art already ran)

1. **M8 "gate by exp(η·income)" in in-sample tangent currency** is soft movement pruning
   (2005.07683). Keep it only as a baseline labelled "movement". The novel arm needs M3's
   prequential or EMA currency, or re-entry.
2. **M1(a) "regress wealth on the movement score"** is expected to give a high R² by
   construction: same formula at the unit level. Treat that as a check of the identity rather
   than a finding.
3. **M2 predictor "accumulated |a·g|"**: add the published variant, (Σ_{unit} g·w)² per step with
   EMA (1906.10771). It already reported that squared beats signed against a leave-one-out
   oracle.
4. **M1(d) "AttnLRP-rule books"** is a port of 2402.05602. Name it that way.
5. **M2 sampled Shapley** is Neuron Shapley. Its TMAB top-k trick can cut cost if 64 permutations
   are too few.
6. **M2 oracle**: one-at-a-time ablation favours gradient×activation (1805.12233 §5.1), and
   ablation ≠ direct effect due to self-repair (2307.15771). Add mean or resample ablation, and
   report heads and neurons separately.
7. **Missing arm**: LCA (1909.01440) as a currency or predictor. It is cheap and is the nearest
   "wealth over training" prior art.

## Recollections not verified from these PDFs
- (unverified) Mozer & Smolensky 1989 "skeletonization" and LeCun 1990 OBD are the origins of
  gate-gradient and second-order importance; OBD is cited inside 1906.10771.
- (unverified) SNIP (1810.02340, title verified) scores |g·w| at init. It is the at-init
  analogue of one-step books.
