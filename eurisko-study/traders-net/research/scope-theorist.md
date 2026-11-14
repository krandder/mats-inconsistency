# Scope: theorist lens on "traders inside a neural network"

Lens: logical induction (LI), online learning (Hedge, EG, universal portfolios,
specialists), prediction-market theory (LMSR, cost-function markets, Kelly
markets), mechanism design. Scoping only: what the larger study should
investigate, not answers. Written 2026-09-27 by a scoping subagent.
Inputs read: `../CONTEXT.md`, `../../logical-soup/REPORT-SOUP.md` §1-3,
`../../wealth-net/{README.md,books.py,train.py}` and
`../../wealth-net/results/summary.json` (checked 2026-09-27T21:50Z, steps
7500-7600, 2 seeds; there is no `RESULTS.md` yet).
URLs marked (verified) were confirmed by web search in this session; the rest
are recollections marked (unverified).

## 0. The one finding that should reshape the study

**The wealth-net books are, almost exactly, the movement-pruning score, and
under plain SGD they are weight-norm growth.**

- For a unit that is positively homogeneous of degree 1 in its outgoing
  weights v (every MLP neuron, every head's value/output channel, every output
  readout), revenue per example is a·∂R/∂a = v·∂R/∂v, and with s = y−p
  detached, ∂R/∂θ = −∂L/∂θ. So **income_t = −θ_unit·∇L_t** (a theorem, Euler's
  identity; the chain rule gives the same number measured on the incoming side
  of a ReLU neuron, which is the Du-Hu-Lee layer-balancing invariant).
- Cumulative wealth = −Σ_t θ_t·∇L_t, which is exactly the movement-pruning
  importance score S = −Σ_t (∂L/∂W)·W of Sanh, Wolf, Rush 2020,
  https://arxiv.org/abs/2005.07683 (verified), and its one-step version is the
  first-order Taylor pruning criterion (Molchanov et al. 2016,
  https://arxiv.org/abs/1611.06440 (unverified)).
- Under gradient flow θ̇ = −η∇L this is (‖θ_T‖² − ‖θ_0‖²)/(2η): a unit's wealth
  is its squared-norm growth (Du, Hu, Lee 2018,
  https://arxiv.org/abs/1806.00900 (verified), prove the layer-difference
  version). With Adam (what `train.py` uses) it is a preconditioned relative;
  with weight decay λ, stationary income is λ‖θ‖² > 0 for every survivor.
- Consequence 1: at any stationary point of the training loss, every
  homogeneous unit's income is zero and total R is zero. Positive R means the
  model is still growing its logits (under-confident on the data the books are
  kept on). On separable/memorisable data, cross-entropy drives norms to
  infinity (implicit-bias results, Soudry et al. 2018,
  https://arxiv.org/abs/1710.10345 (unverified)), so wealth accrues forever.
- Consequence 2, already visible in the data: the books are kept on training
  batches that are re-sampled from a fixed 80% split. In seed 1 at step 7500
  the pi family has the **largest** readout wealth (507, of 8 families) and the
  **worst** held-out BCE (3.53 vs 0.70 for the logistic baseline): the books
  pay memorisation. Wealth as kept is an in-sample quantity; LI, Hedge and
  Kelly all settle **prequentially** (on outcomes not yet trained on).

What the study must therefore do before any wealth-gating: (a) show what the
books measure beyond Δ‖θ‖² and movement pruning, and (b) move settlement to
held-out, never-trained-on outcomes. Section 5, experiments E1-E2.

## 1. Claims in CONTEXT.md: theorem, false, open

| # | Claim | Status | Note |
|---|---|---|---|
| 1a | Kelly traders + wealth updated by P&L = Bayesian model averaging | **Theorem** (binary, price-taking, full Kelly) | Beygelzimer, Langford, Pennock 2012, https://arxiv.org/abs/1201.6655 (verified): price = wealth-weighted mean of beliefs, log regret vs best trader ≤ ln(1/w_i). Fractional Kelly = tempered/shrunk Bayes (same paper, from memory). Price is a **linear pool in probability space**, not log-odds. |
| 1b | LMSR = no-regret learning | **Theorem, but about the maker, not the traders** | Chen & Vaughan 2010, https://arxiv.org/abs/1003.0034 (verified): cost-function markets = FTRL over *outcomes* with trades as losses; bound is the maker's worst-case loss b·ln n. It is not "Hedge over traders". The study should keep the two Hedges apart: Hedge-over-outcomes (maker) and Hedge-over-traders (Kelly wealth). |
| 1c | Price-reading traders break BMA and that is where LI lives | **Half right** | Price-dependent traders break the closed form, but LI does not rely on wealth dynamics at all: Garrabrant's LIA picks prices by a fixed point that makes a budgeted, weighted combination of traders unable to profit (https://arxiv.org/abs/1609.03543 (unverified id, confident)). "Where LI lives" is the fixed-point price + budgets + enumeration, not price-reading per se. |
| 2 | Demand ∝ q_i − price with q_i = update(price, LR_i) is self-limiting; wealth = marginal information (Shapley-like) | **Self-limiting: false for redundant traders. Shapley: false as stated; true for one arrival order** | If k traders hold the same evidence, each applies the LR to the same price: naive-Bayes double counting (product of experts), not self-limiting. It is limited only when the price already contains the evidence *and* the trader can tell (needs price-reading). Under a sequential LMSR, trader i's P&L = log-score of price after i minus before i (Hanson's market scoring rule, telescoping) = marginal contribution **in that order**; Shapley needs the average over orders (random arrival). The boosting reading is right: forward-stagewise additive logistic regression (Friedman, Hastie, Tibshirani 2000, https://projecteuclid.org/euclid.aos/1016218223 (unverified)). |
| 3a | Softmax = LMSR over exclusive outcomes | **Theorem** | LMSR price = softmax(q/b); log-sum-exp is the convex conjugate of negative entropy (Abernethy, Chen, Vaughan, https://arxiv.org/abs/1011.1941 (unverified)). |
| 3b | CE gradient p − y = settlement, so backprop pays demand × (outcome − price) | **True to first order only; the difference matters** | Backprop marks every position at the **final** price: z_i(y−p). A real LMSR charges C(q+z)−C(q), so exact total P&L from a prior is log p(y) − log p0(y) (positive in expectation for a good model). The linearised total z(y−p) has **zero expectation under calibration**, so a calibrated network has zero total income. The gap is the Bregman divergence of log-sum-exp. Exact, efficient split of the log-score gain: Aumann-Shapley path integral = integrated gradients (Sundararajan et al. 2017, https://arxiv.org/abs/1703.01365 (unverified)). |
| 3c | Residual stream is a sequential market | **False as a description of backprop; true as a design option** | Backprop settles all layers at one clearing price: a call market (batch auction). A sequential market would settle layer k along the price path p_{k−1}→p_k (logit lens prices). Which one to use is a design choice with different credit (sequential = order-dependent marginal; call = Euler share). Worth testing, see E3. |
| 4 | Unit = anything with an additively separable logit contribution | **Definitional; fine** | Exactness of books needs degree-1 homogeneity; attention softmax and LayerNorm break it (the README's residual). |
| 5a | Fixed point ℓ* = (ℓ0 + Σk_iℓ_i)/(1+Σk_i) | **Theorem (trivial), but a step backwards** | This is a log-linear pool with weights summing to 1: it can never be more extreme than its most extreme input, so it **cannot aggregate independent evidence** (the known need to extremise, Satopää et al. 2014, https://doi.org/10.1016/j.ijforecast.2013.09.009 (unverified)). The plain logit sum (product of experts) can. Storkey 2011 / Storkey, Millin, Geras 2012, https://arxiv.org/abs/1206.6443 (verified): log-utility agents → linear (mixture) pool; exponential-utility agents → product pool; isoelastic → α-mixtures. So "logit sum vs Kelly market" is a choice of trader utility, which is a cleaner framing than convex normalisation. |
| 5b | Eisenberg-Gale program for Kelly traders | **Theorem** (log utilities + linear payoffs = Fisher market) | Eisenberg-Gale 1959 (unverified citation). For binary questions it reduces to the BLP wealth-weighted mean, no program needed. |
| 5c | BASE layers are a market router | **True, and the field has gone further** | BASE: linear assignment (Lewis et al. 2021, https://arxiv.org/abs/2103.16716 (unverified)). DeepSeek's loss-free balancing adds a per-expert bias updated from observed load (Wang et al. 2024, https://arxiv.org/abs/2408.15664 (verified)): that bias is a **capacity price updated by tâtonnement / dual ascent**. The load-balancing aux loss (Switch, https://arxiv.org/abs/2101.03961 (unverified)) is a penalty method for the same constraint. |
| 6a | Money flows backward like gradients; partial ownership automatic; big-trader wealth = sum | **True for the Euler split (theorem for homogeneous graphs)**; not new | Holland's bucket brigade (classifier systems, 1985 (unverified)), Baum's Hayek machine (1999 (unverified)), and conservation-based attribution (LRP, DeepLIFT, conductance https://arxiv.org/abs/1805.12233 (unverified)). Under Euler, an intermediate homogeneous unit's net (received − paid to suppliers) is exactly zero: all value lands on inputs and constants, so "value added" needs a non-homogeneous step or a compute charge to be nonzero. |
| 6b | Gate execution by wealth | **Open**; with the current books it is movement pruning | See §0. The open part is multiplicative (Hedge) gating on prequential income, which is not what pruning does. |
| soup | "Per-trader compute is not the lever; population density is" | **Overstated as a general law** | 5000 epochs, one DSL, 3 seeds. A lottery model predicts it: P(some trader hits a pattern) ≈ 1−(1−π)^{K/Q}; fuel changes π only if the hit needs more instructions. Test the lottery model's quantitative prediction before calling it the bitter-lesson lever (Q-T4). |

## 2. What could be proven about a wealth-gated network

Target statements, easiest first. Each is a candidate lemma for the study.

1. **Books identity (theorem, hours).** For degree-1-homogeneous units:
   income = −θ·∇L; gradient-flow wealth = Δ‖θ‖²/2η; zero at stationary points;
   λ‖θ‖² with weight decay. Also: z(y−p) has zero mean under calibration.
2. **Hedge-gated mixture (theorem, known).** If units are experts in
   probability space, the gate is w_i ∝ exp(cumulative prequential log-score),
   and the output is the gated mixture: log-loss regret ≤ ln N vs the best
   unit (Vovk aggregating algorithm; mixable, η=1). With logit-space gating
   (output = Σ w_i ℓ_i), it is exponentiated gradient on the simplex (Kivinen
   & Warmuth 1997 (unverified)): O(√(T ln N)) vs the best convex combination;
   log-loss is exp-concave so ONS/universal-portfolio rates O(N log T) are
   available (Cover 1991 (unverified); Hazan et al. 2007 (unverified)).
3. **Units that learn = shifting / sleeping experts.** Units change while
   being weighted, and MoE experts fire only on routed tokens. The right
   frames with regret bounds: Fixed Share (Herbster & Warmuth 1998
   (unverified)) for birth/death and non-stationarity; **specialists / sleeping
   experts** (Freund, Schapire, Singer, Warmuth 1997 (unverified)) for units
   that abstain. Specialists give a principled wealth update for a routed
   expert: only awake experts' weights move, renormalised within the awake set.
4. **Top-k routing is bandit feedback (claim to check, days).** An unrouted
   expert gets no gradient and no settlement on that token, so router
   training is a partial-information problem. Exp3-style importance weighting
   (Auer et al. 2002 (unverified)) or a uniform exploration floor are the
   principled replacements for the load-balancing aux loss, and the rich-get-
   richer collapse is the known failure of greedy exploitation under bandit
   feedback. Theorem candidate: with importance-weighted credit, router regret
   is O(√(T N ln N)) against the best fixed routing; without it, collapse is
   possible. Prior art must be searched (MoE-as-bandit papers likely exist).
5. **Capacity prices (theorem, known form).** Routing tokens to experts with
   capacity c is an assignment LP; the DeepSeek bias is its dual variable
   under subgradient ascent; Sinkhorn routing is its entropic version. A
   wealth-budgeted version (tokens have budgets, experts sell capacity) is a
   Fisher market solvable by Eisenberg-Gale / proportional response dynamics
   (unverified: Wu & Zhang 2007; Zhang 2011). Provable: market-clearing,
   envy-freeness, and "load balance without an aux gradient".
6. **Fractal books (theorem, hours).** A DAG of Kelly funds with **fixed**
   ownership fractions equals a flat mixture of leaves with path-product
   weights: no extra power. Power comes only from **rebalancing** at inner
   nodes (constant-rebalanced portfolios beat buy-and-hold; Cover). The exact
   recursive-Bayes version of "big trader = mixture of small traders, split or
   not" is Context Tree Weighting (Willems, Shtarkov, Tjalkens 1995
   (unverified)); tree-structured expert pruning (Helmbold & Schapire 1997
   (unverified)). Partial ownership in a DAG = mixture over paths; conservation
   holds by construction.
7. **Local profit-maximisation = backprop (theorem, hours).** A **price-taking**
   unit maximising its own linearised P&L z_i(y−p) with p held fixed has
   gradient (y−p)∂z_i/∂θ_i = −∂L/∂θ_i: backprop is the competitive
   equilibrium of price-taking units. A **price-making** unit (accounting for
   its effect on p) shades its position (Cournot), so strategic units do not
   minimise the global loss. Mechanism-design question: which settlement rule
   makes strategic units' Nash equilibrium the loss minimiser? Chang et al.
   2020 solve the RL analogue with a cloned Vickrey auction,
   https://arxiv.org/abs/2007.02382 (verified).
8. **Market selection / survival (theorem, known).** Blume & Easley 2006,
   https://www.econometricsociety.org/publications/econometrica/2006/07/01/if-youre-so-smart-why-arent-you-rich-belief-selection-complete
   (verified): with log-utility in complete markets the most accurate beliefs
   survive; among Bayesians, the smaller parameter space wins. Predicts the
   soup's Gini 0.75-0.82 is the healthy regime, not a pathology, *if* the rich
   are the accurate; wealth-concentration is a failure only when the rich also
   control who learns (MoE collapse).

## 3. The finite-population analogue of the LI criterion

LI criterion: for every efficiently computable trader T, T's plausible
holdings are bounded above or unbounded below (no exploitation). Three
finite versions, from weakest to strongest; the study should pick one and
report it as a curve, not a pass/fail.

1. **Class-C exploitability curve.** Fix a finite reference class C of
   price-taking, budgeted shadow traders (the soup already has six). Define
   E_C(t) = max_{T∈C} cumulative P&L_T at market prices, prequential. The
   market is (C, f)-inductive if E_C(t) ≤ f(t) with f sublinear (o(t)); report
   the slope. Choosing C encodes the desiderata: C = binned sign bets on
   price buckets gives calibration (Foster-Vohra); C = coherence arbitrageurs
   gives approximate coherence; C = pattern-k gives pattern learning; C = a
   function class H times price buckets gives **multicalibration / outcome
   indistinguishability**, the closest existing theory (Hébert-Johnson et al.
   2018 (unverified); online multicalibration, Gupta et al.
   https://arxiv.org/abs/2101.01739 (unverified); OI, Dwork et al.
   https://arxiv.org/abs/2011.13426 (unverified)). Deterministic fixed-point
   calibration (Foster & Hart forecast hedging; Kakade & Foster 2004
   (unverified)) is the closest cousin of LI's price-selection trick.
2. **Invasion test (evolutionary-stability form).** For T ∈ C, inject T with
   wealth ε into the resident population and measure its log-growth rate at
   resident prices (invasion fitness). Criterion: every T ∈ C has invasion
   fitness ≤ 0 eventually, and its total take is O(log 1/ε). This is the
   right finite analogue because LI's enumeration with weights 2^{−k} *is*
   "every trader is present at small wealth"; a finite population is a sample
   of that enumeration, and the invasion test checks the missing ones. The
   soup's Δ-shadow slopes are invasion fitness estimates already.
3. **Neural version: probe exploitability.** For a network, T ∈ C is a small
   trader reading (features, price) and emitting a logit correction, trained
   online on held-out outcomes; its cumulative settlement is the network's
   exploitability by C. It is a prequential boosting test: if a class-C probe
   earns, the network is not C-inductive. C scales with probe compute, giving
   a "frontier of exploitability vs trader compute" curve, the neural twin of
   the soup's frontier.

Open theoretical question: does a finite wealth-dynamic market (no fixed
point, no enumeration) achieve (1) for any interesting C, given that C's
members must evolve? The soup says not for the arbitrageur in 5000 epochs.
Candidate result: sublinear E_C requires either (a) C ⊆ population with
positive wealth (then Hedge bound), or (b) prices set by a fixed point over
the population (LI), or (c) a birth process that samples C with positive rate
(then Fixed-Share-like bound, rate depending on the birth probability of the
exploiter). (c) turns the soup's frontier into a birth-rate question.

## 4. Prior art to check (priority order)

1. Movement pruning (Sanh, Wolf, Rush 2020), https://arxiv.org/abs/2005.07683 (verified): same score as the books.
2. Storkey 2011 "Machine learning markets" and Storkey, Millin, Geras 2012, https://arxiv.org/abs/1206.6443 (verified): utility choice determines pooling; wealth updates implement Bayes or mixture-weight updates.
3. Beygelzimer, Langford, Pennock 2012, https://arxiv.org/abs/1201.6655 (verified).
4. Chen & Vaughan 2010, https://arxiv.org/abs/1003.0034 (verified); Abernethy, Chen, Vaughan, https://arxiv.org/abs/1011.1941 (unverified).
5. Garrabrant et al., Logical Induction, https://arxiv.org/abs/1609.03543 (unverified id).
6. Loss-free balancing, https://arxiv.org/abs/2408.15664 (verified); BASE https://arxiv.org/abs/2103.16716, Expert Choice https://arxiv.org/abs/2202.09368, Switch https://arxiv.org/abs/2101.03961 (unverified).
7. Chang et al. 2020, cloned Vickrey society, https://arxiv.org/abs/2007.02382 (verified); survey "Market-based Architectures in RL and Beyond", https://arxiv.org/abs/2503.05828 (seen in search, unread).
8. Du, Hu, Lee 2018, https://arxiv.org/abs/1806.00900 (verified).
9. Blume & Easley 2006, https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2006.00691.x (verified).
10. Specialists / sleeping experts (Freund, Schapire, Singer, Warmuth 1997), Fixed Share (Herbster & Warmuth 1998), Exp3 (Auer et al. 2002), EG (Kivinen & Warmuth 1997), Universal portfolios (Cover 1991), CTW (Willems et al. 1995): all (unverified), standard.
11. Online multicalibration https://arxiv.org/abs/2101.01739 and outcome indistinguishability https://arxiv.org/abs/2011.13426 (unverified).
12. Bucket brigade (Holland 1985), Hayek machine (Baum 1999), integrated gradients https://arxiv.org/abs/1703.01365, conductance https://arxiv.org/abs/1805.12233, Neuron Shapley https://arxiv.org/abs/2002.09815 (unverified).

## 5. Experiments (cheap first)

- **E1 Books vs norm growth (hours).** On the existing wealth-net checkpoints
  and a rerun with plain SGD (no Adam, no decay): regress per-unit wealth on
  Δ‖θ_out‖²/2η and on the movement-pruning score. Prediction: R² ≈ 1 under SGD,
  high under Adam. If so, any wealth-based ablation result is a pruning result
  and must be compared to movement and magnitude pruning.
- **E2 Prequential books (hours).** Keep the books on fresh held-out outcomes
  (next chunk of the stream, before training on it), and add exact LMSR
  settlement (log-score gain via integrated gradients from a zero-logit
  baseline) alongside the linearised one. Prediction: pi's wealth turns
  negative; the per-family wealth ranks match held-out BCE improvement over
  the logistic baseline.
- **E3 Call vs sequential settlement (days).** Pay each layer along the logit-
  lens price path instead of at the final price; compare stability of unit
  wealth ranks and ablation predictiveness.
- **E4 Hedge-gated MLP (days).** Gate each MLP neuron's output by
  w_i ∝ exp(η · prequential income), keep gradient training of the weights
  untouched; sweep η, add a Fixed-Share floor α. Measure held-out loss, the
  class-C exploitability curve, and wealth Gini. Compare to magnitude/movement
  pruning at matched active-unit count.
- **E5 Market router for a small MoE (days).** Replace the aux loss with (a)
  DeepSeek bias (price) — the baseline to beat, (b) specialist-Hedge wealth per
  expert, (c) Exp3-style importance-weighted credit for unrouted experts.
  Measure collapse (expert-usage entropy over time), held-out loss, and
  whether wealth predicts ablation damage.
- **E6 Exploitability harness (days).** Shared across soup and wealth-net: a
  fixed class C (calibration bins, coherence arbitrageur for A∧B/A∨B, pattern-k,
  small MLP probes at 3 compute sizes) run as price-taking shadow traders on
  held-out streams; report E_C(t) slopes. This is the metric every other
  experiment reports.
- **E7 Lottery model of K/Q (hours).** Fit 1−(1−π_f)^{K/Q} per family to the
  soup's Brier table; if it fits, K/Q is "tickets per question" and predicts
  the K/Q needed for LFSR/CA before spending GPU on it.
