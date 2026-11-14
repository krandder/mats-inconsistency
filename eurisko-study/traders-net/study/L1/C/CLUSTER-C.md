# L1 cluster C: routing, logical induction, online learning

Worker 3 of 3 (cluster reader C), 2026-09-27 ~22:15Z. No experiments were run, so there is no PREREG.md.
Evidence: `C/verify-C.json` (ID, title, authors and date from each paper's own arXiv abs page, 14/14 verified),
`C/quotes-C.txt` (verbatim excerpts behind every mechanism claim below). Full texts were read from
arxiv.org/html (LI from its PDF) and cached in /tmp/traders-net/L1C/, which is not synced.
"Ideas 1-6" are the numbered ideas in `../../CONTEXT.md`.

Each entry has five lines: **built** / **overlap** / **did not** / **so for our arms** / **ID check**.

## Routing and MoE

**Wang et al. 2024, Auxiliary-Loss-Free Load Balancing** https://arxiv.org/abs/2408.15664
- Built: a per-expert bias b_i added to routing scores *only for the top-K choice*, updated
  `b_i += u*sign(mean load - load_i)` after each batch (Alg. 1). No gradient and no aux loss. Tested at 1B-3B, 200B tokens.
- Overlap: this is already a per-expert **price on quantity**, set by tatonnement (idea 5's "MoE balancing as a market").
  The CONTEXT "DeepSeek bias is a capacity price" premise is this paper.
- Did not: it prices load, not quality. The bias is not in the gate weight, it holds no wealth or P&L, has no
  per-family prices, and has no bankruptcy or birth. The step size u is a hyperparameter (u=1e-3).
- So: this is M7's main control. "Wealth replaces load balancing" has to beat it on loss per active FLOP. Beating the
  aux loss alone is not a result.
- ID: verified, the title matches (verify-C.json).

**Han & Zhong 2025, theory of ALF-LB** https://arxiv.org/abs/2512.03915
- Built: casts DeepSeek's bias as a single-shot primal-dual (dual-ascent) method for a token-expert
  **assignment problem**, proves Lagrangian monotonicity, approximate balance and log regret in an online model,
  and runs 216 1B DeepSeekMoE models over step-size schedules.
- Overlap: it proves that "the bias is a Lagrange multiplier / price", so idea 5's BASE and assignment reading is
  already a theorem.
- Did not: no wealth, no quality-based payment, no per-expert budgets. Its own grid finds a tradeoff with no free lunch:
  u/sqrt(n) gives the lowest validation loss and the worst imbalance, and the aux loss the reverse (quotes-C.txt).
- So: M7's claim "no tuned coefficient" is weak unless it beats *all* the u schedules. This paper shows that even the
  loss-free price needed a u search (u in {1e-4..10}). NOVELTY-FLAG: pricing load via duals is done.
- ID: verified.

**Vyas 2025, MoB: Mixture of Bidders** https://arxiv.org/abs/2512.10969
- Built: replaces the gate with a per-batch second-price (VCG) auction among **4 CNN experts**. The bid is
  α·loss(E_i(x),y) + β·λ·ΣFisher (EWC). The winner trains on the batch. Split-MNIST, 5 seeds, 88.8% vs 19.5% for a gated MoE.
- Overlap: idea 5 (market routing) and "experts as bidders" in name. Continual learning ties to M3/M8's regime shift.
- Did not: payments are computed but never accumulated: **no wealth, no budget, no bankruptcy**. The bid uses the
  **label y**, and the paper does not say how it routes at test time without labels (a leakage risk; our reading of
  §3.3.1). Routing is one expert per batch, not per token, and only at toy scale. There is no load-balancing baseline.
- So: MoB is not a wealth mechanism. It is "route by lowest current loss". Any M7 arm that routes by prequential
  loss (test before train) is the honest, label-free version of MoB and should cite it.
- ID: verified (arXiv date 2025-11-30).

**Liu et al. 2026, Routing-Free MoE** https://arxiv.org/abs/2604.00801
- Built: each expert decides its own activation (an internal low-rank gate norm minus a learned per-expert bias,
  thresholded by a global θ). No router, softmax or top-K. Expert and token balance are interpolated by μ, and the aux
  coefficient is adapted multiplicatively: `λ_{t+1} = λ_t(1+η)^{sign(ρ_t-ρ*)}`. Up to 0.8B, OpenWebText.
- Overlap: "experts decide for themselves whether to trade" is close to Kelvin's decentralised traders. The
  per-expert bias b_i is a self-set participation threshold.
- Did not: no money. Balance still comes from an auxiliary loss, with one adaptive global price λ and a step η.
  No per-expert P&L, no delayed settlement.
- So: this is a second non-economic baseline for M7 ("no hand-tuned coefficient" is already claimed here through
  adaptive λ). A wealth arm must beat it or tie it with fewer knobs.
- ID: verified.

**Mouzouni 2026, Three Phases of Expert Routing** https://arxiv.org/abs/2604.04230
- Built: models token routing as a congestion game (mean-field). The single-type equilibrium **is temperature-scaled
  softmax**. It fits γ_eff over OLMoE / OpenMoE checkpoints and finds surge, then stabilisation, then relaxation.
- Overlap: idea 3's "softmax = equilibrium price" claim, on the routing side. Softmax gating is already a
  congestion-priced equilibrium.
- Did not: it is purely observational, with no interventions (its own "Limitations"). The game does not beat softmax
  at prediction (L1 0.199 vs 0.200). There is no wealth.
- So: in M7 we should log a per-checkpoint γ_eff-style balance/quality slope for each arm. This is cheap, and it lets
  us say whether the wealth arms move the phase trajectory. Measuring on OLMoE stays out of scope, as in the plan.
- ID: verified.

**He 2024, PEER: Mixture of A Million Experts** https://arxiv.org/abs/2407.04153
- Built: about 1M **single-neuron experts** e_i(x)=σ(u_i·x)v_i, retrieved per token by product keys (8 heads × top-16),
  with query BatchNorm for usage. It beats dense FFW and coarse MoE at iso-FLOP on C4. Usage is about 100% at 1M
  experts, and unevenness is reported.
- Overlap: this is exactly idea 4's unit (neuron = trader) at Kelvin's "thousands of traders sharing core computation"
  scale. Retrieval by product key is the shared compute.
- Did not: no aux balance loss. Balance comes from BN, with no prices and no wealth. It does not report per-expert value.
- So: this is the reference architecture if Kelvin wants "thousands of traders". Books on PEER's singleton experts are
  the natural substrate beyond our small nets (not run: out of budget). Cite it in D1.
- ID: verified.

**Liu, Gao, Chen 2023, SparseMixer** https://arxiv.org/abs/2310.00811
- Built: a second-order (midpoint ODE) gradient estimate for the router terms that sparse top-K training drops. Tested
  on Switch; convergence up to 2× faster. Still uses Switch's aux loss at 0.01.
- Overlap: it is the non-economic answer to **censored feedback**: unselected experts and routes get no signal. That
  is the same problem wealth-gated execution (M8) and priced routing (M7) create.
- Did not: no prices, no wealth. It is about routing gradients only.
- So: M3/M8 "wealth of unexecuted units" needs a counterfactual estimate. SparseMixer or GRIN-style estimators are the
  control. A wealth-gated arm that starves units of signal should be compared to one using this estimator. (The
  scope file's "unverified id" tag is now resolved: the ID is verified.)
- ID: verified.

## Logical induction, markets as learners, selection

**Garrabrant, Benson-Tilsen, Critch, Soares, Taylor 2016, Logical Induction** https://arxiv.org/abs/1609.03543
- Built: the LI criterion (no efficiently computable trader exploits the prices) and LIA. LIA enumerates all e.c.
  traders, wraps each with **Budgeter** (it zeroes a trader once it could fall below −b in any plausible world),
  sums them with fixed weights over trader index k and budget b in **TradingFirm**, and sets prices by a **Brouwer
  fixed point** (MarketMaker) so that the firm makes no trade.
- Overlap: idea 5's fixed point, "bounded bets" in idea 3, and Q3 (M5) exploitability all trace here. Jessica is a co-author.
- Did not: **no multiplicative wealth reallocation**. A trader's influence is prior weight × budget tier, not
  accumulated profit, and budget-busting traders are cut off rather than shrunk. There is no learning of trader
  internals, no shared computation among traders, and no runtime or convergence rate (§5.5; it proves that some
  rates are uncomputable).
- So: this is the main delta for our claim. We *learn* the traders by SGD, weight them by *realised wealth*, and share
  their computation. M5's fixed trader class plus E_C(t) is the finite, computable shadow of the LI criterion and
  should be stated as such.
- ID: verified (PDF, 131 pp).

**Beygelzimer, Langford, Pennock 2012, Kelly bettors** https://arxiv.org/abs/1201.6655
- Built: price-taking Kelly traders in a binary market. The price is the **wealth-weighted mean belief**, wealth
  updates by Bayes' law, and the price has bounded log regret to the best trader. Fractional Kelly equals full Kelly
  with a belief mixed toward the market, and the price then tracks a time-discounted frequency. It also learns the
  Kelly fraction.
- Overlap: idea 1 (settled theorem). Fractional Kelly **is** idea 2's "correct the market, shrunk by confidence" and
  Kelvin's "traders know they can only improve here and there".
- Did not: fixed beliefs (no learning traders, no features), one event at a time, and no price-reading traders beyond
  the fractional mix.
- So: K1's Kelly pool with fixed beliefs is BMA. Its behaviour is known, so K1 is a demo, not a finding. The humble
  layer in M6 should cite fractional Kelly as its prior art: "anchored" is fractional Kelly.
- ID: verified.

**Chen & Vaughan 2010, prediction markets via no-regret** https://arxiv.org/abs/1003.0034
- Built: cost-function markets correspond to FTRL, with the cost function as the regulariser. The market maker's
  bounded loss gives O(√T) regret. LMSR corresponds to Hedge/entropy.
- Overlap: idea 1's "LMSR = no-regret", and idea 3's "softmax = LMSR" (LMSR over exclusive outcomes is softmax of
  cumulative shares).
- Did not: the "experts" in this learning problem are **outcomes**, not traders. It says nothing about trader
  wealth or selection.
- So: T1's lemma 2 ("exact LMSR settlement = log-score gain") is a corollary to cite, not a claim to make.
- ID: verified.

**Gupta, Jung, Noarov, Pai, Roth 2021, Online Multivalid Learning** https://arxiv.org/abs/2101.01739
- Built: efficient online algorithms whose mean, moment and interval predictions are calibrated conditionally on
  every group G in an arbitrary intersecting collection, against adversarial sequences.
- Overlap: Q3/M5. Multicalibration w.r.t. a class G amounts to no profitable bet by any "trader" that bets on
  group-indicator × calibration-bucket (our reading, unverified as a formal equivalence). It is a polynomial-time,
  finite-class counterpart of LI's criterion.
- Did not: no market, no wealth, no traders as learned units, no deductive coherence (no logical relations between
  questions).
- So: M5 needs this as a **baseline**. The online multicalibration forecaster is the non-market way to be
  unexploitable by a fixed class, so "a net of traders is unexploitable by class C" must be compared to it at equal C.
  NOVELTY-FLAG for Q3.
- ID: verified.

**Blume & Easley 2006, "If You're so Smart, why Aren't You Rich?"** https://doi.org/10.1111/j.1468-0262.2006.00691.x (Econometrica 74(4):929-966)
- Built: in Pareto-optimal (complete-market) allocations, whether a trader survives is decided only by discount
  factors and beliefs (relative-entropy accuracy). The market selects Bayesians with the truth in their support, and
  prefers smaller parameter spaces. The market selection hypothesis can fail in incomplete markets.
- Overlap: K1/Q8 directly. Complete market plus equal discounting gives monoculture of the most accurate belief.
  Coexistence needs exact ties or market incompleteness.
- Did not: no bounded or computable traders, no learning of features, no network.
- So: K1's "expected: monoculture without niches" is this theorem. K1 can only surprise via incompleteness (per-niche
  markets, i.e. separate questions), which is what should make the "coexistence" arm meaningful. A 2013 Econometrica
  comment https://doi.org/10.3982/ecta10455 reportedly gives a counterexample to its Theorem 8 (search summary only,
  unverified; author Massari, unverified).
- ID: DOI and metadata verified via Crossref. The abstract comes from the publisher and search summaries; the full text was not read.

**Hub wiki `consistency-ft.md`, Exploit LP** https://github.com/futarchy-fi/hub/blob/main/wiki/consistency-ft.md
- Built: `Exploit(c,S)` = max t s.t. t ≤ Σ α_i (p_i − w_i) for every consistent world w, α ∈ [−1,1]^n, solved with
  `scipy.optimize.linprog` (code: /home/agent/workspaces/consistency-ft-next-run-20260919/metrics.py,
  `exploit()`). It is the guaranteed profit of an L∞-bounded arbitrageur on one check tuple (Negation, And, Or, Cond, ...).
- Overlap: M5's "coherence before resolution" on D-prove questions (A∨¬A, A∧¬A, (A∧B)→A) is exactly this check family.
- Did not: static and single-check. No trader class over time, no E_C(t), no delayed resolution. The wiki records
  that for single-constraint checks it equals the plain violation metric (bead hub-t8k9c.5 close note).
- So: **M5 should import `exploit()` rather than re-derive it** and report the Exploit LP on D-prove bundles next to
  E_C(t). NOVELTY-FLAG: the static coherence metric already exists in the fleet.
- ID: read locally, /home/agent/data/repos/hub/wiki/consistency-ft.md lines 430-490 and 1690-1702.

## For the synthesizer: what cluster C already covers

Candidate NOVELTY-FLAGS (planned arms that prior art has already run or settled):
1. M7 "a price replaces the load-balancing loss": done as a per-expert load price (2408.15664), proven as primal-dual
   (2512.03915), and done again with an adaptive global λ (2604.00801). The only open part is a **quality/wealth**
   price, and a result there must beat all three plus the u-schedule grid.
2. M7 "auction router": MoB (2512.10969) did it at 4 experts with label-using bids and no wealth. Not a threat to
   novelty, but it must be cited, and its label leak should not be repeated.
3. K1 "monoculture vs coexistence": Blume-Easley settles the complete-market case. BLP settles price = wealth-weighted
   belief and Bayes. K1 is a demo unless it has incompleteness or learning traders.
4. M6 "anchored/humble traders": this is fractional Kelly (BLP §6-7). Cite it; the new part is only the fixed point
   plus SGD-learned beliefs.
5. M5 "unexploitable by a fixed class": online multicalibration (2101.01739) is the non-market baseline, and the
   static Exploit LP is already in the fleet (consistency-ft `metrics.py`).
6. M3/M8 censored feedback: SparseMixer (2310.00811) is the gradient-estimator control.

What cluster C does **not** contain (candidate delta for the one-liner): no paper here keeps **multiplicative,
P&L-driven wealth for units that are learned by SGD and share computation**, or uses that wealth to gate or route.
LI has fixed-weight budgeted traders, BLP and Blume-Easley have fixed-belief traders, the MoE papers price load and
not value, and MoB computes payments and throws them away. That gap is real. Cluster A (Balduzzi, Storkey, Economy of
Minds) must be checked before it is claimed.
