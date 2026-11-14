# Microstructure and information-aggregation priors for the PFN program (2026-10-28)

For: Kelvin (MATS), via opus-mats. Requested in Kelvin's words: "the task of a trader that has to decide how to price something based on private information ... a bunch of traders with a bunch of different signals, and some signals are truly private and some signals are shared (and some others may even be public and interfere)".

Scope: a literature map (section 1), three proposed prior worlds (section 2), a ranking and a novelty check (section 3). This is a design proposal. No transformer has been trained on any of it.

Status tags (what I actually checked today):
- **R**: I read the relevant section of the primary text today.
- **A**: abstract or publisher/repository page, as relayed by the web-search tool today. This is not the PDF.
- **S**: a secondary description today (a later paper, course notes, a survey).
- **M**: from memory, not checked today.
- **C**: I computed or verified it numerically today. The scripts are in `work/microstructure-priors/` (`lop_check.py`, `worlds_check.py`; numpy, CPU).
- **I**: my own inference. No source makes the claim.

Primary texts read today: Lambert-Ostrovsky-Panov (NBER WP 20516, sections 1-7); Vives, *Information and Learning in Markets*, ch. 4 §4.2 (the PDF text extraction is garbled but the formulas are legible); Allen-Morris-Shin 2006 (RFS pp. 723-726). Everything else is at the level of abstracts or secondary sources, as tagged.

---

## 0. Summary

- **One fact makes this tractable.** Lambert, Ostrovsky and Panov (Econometrica 2018; LOP below) give a closed-form, unique linear equilibrium of the one-shot Kyle market for an arbitrary jointly normal information structure. The structure can include multi-dimensional signals, signals shared across traders, signals about other traders' errors, a market maker with its own signals, and liquidity demand correlated with value. I implemented the formula and checked it [C]:
  - it reproduces Kyle (1985), static Holden-Subrahmanyam and LOP Example 2;
  - on 200 random structures it satisfies both equilibrium conditions (market-maker Bayes consistency and trader best responses) to machine precision;
  - it costs about 0.1-0.45 ms per episode in unbatched numpy for up to 32 traders.
- **The beauty-contest equilibrium is just as cheap.** This is the Morris-Shin / Angeletos-Pavan / Myatt-Wallace family of pricing games, where a quote is judged against both the fundamental and other traders' quotes. For any finite Gaussian structure the equilibrium is a single linear solve, and it is unique whenever the coordination weight r < 1. It reproduces the Morris-Shin public-signal weight α/(α+β(1−r)) exactly [C].
- **Proposal.** Use one shared prior over information structures (sources with shared "sender" noise, private "receiver" noise, truly private sources, public signals, and "interfering" signals). Put three market mechanisms on top of it, each with an exact per-episode target:
  - **(A) beauty-contest pricing.** Recommended first.
  - **(B) Kyle-LOP batch auction with strategic insiders.**
  - **(C) sequential dealer market with predecessors of unknown type.**
- **What would be new** (section 3). Every learning-in-markets paper I found trains agents on one fixed game. I found none that amortises across a prior over information structures, compares against the exact per-episode equilibrium, or looks at how a network computes higher-order beliefs or strategic shading. This claim is bounded by my searches, which are listed in section 3.

---

## 1. Map of the literature

Notation: v is the asset value (fundamental), u is noise or liquidity demand, CARA-normal means exponential utility with Gaussian variables, REE means rational-expectations equilibrium, MM means market maker.

### 1a. Competitive rational-expectations models (prices as noisy public aggregates)

| Paper | Information structure | What is priced or traded | Bayes / equilibrium object | Closed form? | Key behaviour | Tag |
|---|---|---|---|---|---|---|
| Grossman 1976, JF 31(2) | n trader types; private signals v+ε_i with independent noise in the signals; no supply noise | one risky asset, competitive | Price is a function of the sufficient statistic of all signals (fully revealing REE) | yes (CARA-normal) | Price aggregates everything, so nobody needs their own signal. This is the "how does information get into the price" paradox. | A/S |
| Grossman-Stiglitz 1980, AER 70(3) | informed (observe a payoff component) vs uninformed; noisy supply; costly information | one risky asset | Partially revealing REE; fraction informed is endogenous | yes | No equilibrium in the noiseless limit: an "equilibrium degree of disequilibrium" | A (opening text) |
| Hellwig 1980, JET 22(3) | many traders, private signals with diverse precisions; noisy supply | one risky asset | Linear REE; the price is informationally equivalent to z = v + u/a, with a = ρτ_ε. Continuum case: price precision τ_p = τ_v + (ρτ_ε)²τ_u, with ρ = risk tolerance | yes | Partial revelation; full revelation as supply noise vanishes; addresses the "schizophrenia" of price-taking | formula R (Vives ch. 4: τ = τ_θ + τ_u(aμ)², a = ρτ_ε); paper M |
| Diamond-Verrecchia 1981, JFE 9(3) | private signals; noise from random endowments | one risky asset | Partially revealing REE | yes | Partial aggregation without exogenous noise traders | A (bibliographic) / M (endowment detail) |
| Admati 1985, Econometrica 53(3) | many assets, private signals, supply noise in each | vector of prices | Multi-asset noisy REE | yes (matrix) | A price can fall in its own payoff; Giffen assets; a high price can be bad news; uncertainty in other assets' supply blocks full revelation | A |
| Breon-Drish 2015, REStud 82(3) | as in GS/Hellwig, but payoff conditional on signal in an exponential family | one risky asset | Continuous-price REE; existence, and uniqueness among continuous equilibria | non-normal closed-form examples | Learning amplifies shocks; price drift; skewness effects | A/S |
| Lou, Parsa, Ray, Li, Wang 2019, JET 183 | continuum of traders in finitely many **information groups**; each trader sees the group signal plus own noise; arbitrary correlation across groups | one risky asset | Linear REE (existence and regularity) | **fixed point**, not closed form | Price aggregates perfectly as noise trade vanishes | A |
| Rostek-Weretka 2012, Econometrica 80(2) | double auction; heterogeneous correlation of **values** across traders | one good | Symmetric linear equilibrium under "equicommonality" | yes | Efficient only if pairwise value correlations are all equal; price informativeness can **fall** with market size | A; equicommonality definition R (via LOP §2) |
| Vives 2011, Econometrica 79(6) | n sellers; costs with common and private components; private signals | supply schedules | Privately revealing Bayesian supply-function equilibrium | yes (affine information) | Higher cost correlation gives steeper schedules and more market power; shading O(1/n) | A |
| Vives 2008 (book) | unified CARA-normal / affine information structures | — | — | — | The textbook treatment of the above | ch. 4 R (partial) |

### 1b. Strategic informed trading (Kyle family)

| Paper | Information structure | Priced or traded | Object | Closed form? | Key behaviour | Tag |
|---|---|---|---|---|---|---|
| Kyle 1985, Econometrica | one insider knows v; noise traders u; risk-neutral MM sees total order flow | market orders; MM sets p = E[v \| flow] | Static: x = β(v−p₀), β = σ_u/√Σ₀; p = p₀ + λ(x+u), λ = √Σ₀/(2σ_u); half of Σ₀ revealed | yes | The insider shades; information is revealed gradually in the dynamic version | C (formulas reproduced via LOP) |
| Kyle 1989, REStud 56(3) | several informed traders with private signals; uninformed speculators; noise | demand **functions** (Nash) | Linear equilibrium in demand schedules | yes | Each trader internalises price impact (no schizophrenia). Prices less informative than competitive; no full revelation as noise → 0 or under risk neutrality | A |
| Holden-Subrahmanyam 1992, JF 47(1) | N identically informed insiders, multi-period | market orders | Static: λ = √(NΣ₀)/((N+1)σ_u); Var(v \| p) = Σ₀/(N+1) | yes | Competition makes them aggressive: information is revealed fast, immediately in the continuous limit | A; static formula C |
| Foster-Viswanathan 1996, JF 51(4) | multiple insiders with **correlated** noisy signals (symmetric) | multi-period market orders | Linear equilibrium; traders learn others' signals from order flow ("forecasting the forecasts of others") | recursive | The initial correlation of signals drives profits and price informativeness | A |
| Back-Cao-Willard 2000, JF 55(5) | continuous-time FV | — | Unique linear equilibrium for imperfectly correlated signals | yes (ODE) | Early "rat race", then endogenous disagreement; the market approaches illiquidity near the end; later efficiency can be lower than with a monopolist | A (partial); non-existence under perfect correlation M |
| **Lambert-Ostrovsky-Panov 2018**, Econometrica 86(4) | **arbitrary** joint normal of (v; θ₁…θ_n multi-dim; θ_M of the MM; u). Includes shared, nested and technical signals and u correlated with anything | one-shot market orders; MM sees D = Σd_i + u and θ_M | **Unique** linear equilibrium. Depth γ is the positive root of one quadratic; α = γA_v − A_u; A_v = Λ⁻¹(Σ_θv − …); Λ = Σ_diag + Σ_θθ − Σ_θM Σ_MM⁻¹ Σ_Mθ | **yes** | Ex. 1: a trader can trade **against** an informative signal (threshold ρ₁ = 2ρ₂ + 1/ρ₂). Ex. 2: a "technical" trader who sees only another trader's noise trades against it. MM-observed ≠ public. Thm 2: large markets aggregate everything if Cov(u,v \| ·) ≥ 0, everything except u if ≤ 0. Thm 3: "abundant" information enters the price as if the MM had observed it | **R + C** |
| Bernhardt-Miao 2004; Colla-Mele 2010 | asymmetric (BM) or circle-correlated (CM) signals | multi-period | Necessary/sufficient conditions (BM); symmetric equilibrium (CM) | partly | — | S (via LOP §2) |

One identity follows from the LOP formula [I]. In the case with no MM signal and independent u, A_v = (Σ_θθ + Σ_diag)⁻¹ Σ_θv. Compare the Bayes regression of v on all signals, Σ_θθ⁻¹ Σ_θv. The equilibrium direction is that Bayes regression with each trader's own-signal covariance counted twice. Strategic shading is therefore a structured "discount on your own information", computed jointly with the overlaps between traders. This connects LOP directly to the program's dedup question.

### 1c. Sequential trading, dealers and cascades

| Paper | Information structure | Priced or traded | Object | Closed form? | Key behaviour | Tag |
|---|---|---|---|---|---|---|
| Glosten-Milgrom 1985, JFE 14 | binary v; traders arrive one at a time; a fraction are informed | unit buys/sells at bid/ask | ask = E[v \| buy], bid = E[v \| sell]; prices a martingale | yes (recursion) | Adverse selection alone creates a spread; market breakdown when informed traders are too many | A (first sentences) / S |
| Easley-O'Hara 1992; EKOP 1996, JF 51 | **event uncertainty** (probability α), informed arrival μ, uninformed ε | buys / sells / no-trades | Bayesian MM over (event, direction); PIN = αμ/(αμ+2ε) | yes | A no-trade carries information (it suggests no event) | S; PIN formula M |
| Bikhchandani-Hirshleifer-Welch 1992, JPE 100(5); Banerjee 1992, QJE 107(3) | conditionally i.i.d. private binary signals; discrete actions; fixed price | binary actions | Public belief from the action history | yes | A cascade starts once public evidence outweighs any single signal; it is fragile | A |
| Avery-Zemsky 1998, AER 88(4) | value uncertainty, plus **event** uncertainty, plus **composition** uncertainty (unknown share of well-informed traders) | GM-style prices | Bayesian MM over the multi-dimensional state | yes (finite states) | Value uncertainty alone: the price adjusts and there is no herding. Event uncertainty: herding with little price effect. Composition uncertainty: herding with significant short-run mispricing | A (partial) / S |
| Park-Sabourian 2011, Econometrica | GM with general signal structures | — | — | — | Herding vs contrarian behaviour depends on whether signals are U-shaped or hill-shaped | M |
| Hanson 2003/2007, LMSR | any | Arrow securities | C(q) = b·log Σ e^{q_i/b}; prices are a softmax; a myopic trader moves the price to its belief | yes | Modular combinatorial markets (#P-hard in general) | M; hardness V (earlier sweep) |
| Chen, Dimitrov, Sami, Reeves, Pennock, Hanson, Fortnow, Gonen 2010, Algorithmica 58 | conditionally independent vs unconditionally independent signals | LMSR | PBE | — | Conditionally independent signals: truthful betting is a PBE and the unique WPBE. Unconditionally independent: bluffing; no full revelation in finite time; a discounted MSR fixes it | A |
| Ostrovsky 2012, Econometrica 80(6) | arbitrary finite partitions | MSR / Kyle-type, dynamic | Every equilibrium | — | Separable securities aggregate; non-separable ones (XOR) may not | V (earlier sweep) |
| Chen-Waggoner 2016 | substitutes vs complements | MSR | — | — | Substitutes are revealed at once; complements late | V (earlier sweep) |
| Eyster-Rabin-Vayanos 2019, JF 74(1) | CARA-normal REE with "cursed" traders who neglect what prices reveal | — | Cursed equilibrium | yes | More volume; volume grows with market size | A |
| Eyster-Rabin 2010 (naive herding, "BRTNI") | rich signals; agents treat predecessors' actions as reflecting only private information | — | — | — | Early signals are double counted; confident herds on wrong actions | M |

### 1d. Coordination motives and public information (beauty contests)

| Paper | Information structure | Priced or traded | Object | Closed form? | Key behaviour | Tag |
|---|---|---|---|---|---|---|
| Morris-Shin 2002, AER 92(5) | public y (precision α), private x_i (precision β); payoff weight r on matching the average action | action / quote a_i | a_i = [αy + β(1−r)x_i] / [α + β(1−r)] | yes | Public information is **overweighted** relative to Bayes (α/(α+β)). More precise public information can lower welfare iff β/α > 1/((2r−1)(1−r)), which needs r > 1/2 | S (secondary formula); weight C; welfare condition S |
| Allen-Morris-Shin 2006, RFS 19(3) | short-lived CARA traders; public prior y, private x_i; noisy supply | sequence of prices | p_t = Ē_t p_{t+1}; average k-th order expectation Ē^k θ = (1−γ^k)y + γ^k θ with γ = β/(α+β) | yes | The law of iterated expectations fails for average belief, so the price is biased toward the public signal relative to the average expectation of the fundamental | **R** (eq. 1, 3 and the Markov-chain form) |
| Angeletos-Pavan 2007, Econometrica 75(4) | linear-quadratic games, general complementarity or substitutability, public and private signals | actions | Linear equilibrium vs an efficiency benchmark | yes | Complementarity raises the weight on public information (volatility). Substitutability raises the weight on private information (dispersion). The social value of information depends on the inefficiency in how it is used | A |
| Myatt-Wallace 2012, REStud 79(1) | each signal has **accuracy** (sender noise, common to all who use it) and **clarity** (receiver noise, private) | beauty-contest actions | Unique linear equilibrium with endogenous attention | yes | Attention goes to the clearest signals even if they are inaccurate. More complementarity means fewer signals used and more public information acquired | A; sender/receiver decomposition M |
| Bergemann-Morris (BCE); Bergemann-Heumann-Morris 2021, RAND 52(1) | **all** information structures | demand functions vs Cournot | Set of equilibrium outcomes over all information structures | moments | With demand-function competition, any market power can arise near complete information; price volatility stays below the variance of the aggregate shock | A |
| Goldstein-Yang 2015, JF 70(4) | two fundamentals; different traders informed about different ones | one asset | Linear REE | yes | Complementarities in trading and information acquisition; possible multiplicity | A (bibliographic) / M |

### 1e. What this literature gives the program

1. **Gaussian closed forms for exactly the private / shared / public mix Kelvin describes.**
   - LOP: strategic trading, arbitrary structure, unique.
   - Morris-Shin generalised: coordination, a linear solve, unique for r < 1 [C].
   - Hellwig: competitive; closed form only when noise is purely private. Lou et al. need a fixed point when noise is shared.
2. **Named, sharp behaviours that a non-strategic Bayes aggregator does not show.** Each comes with a formula:
   - overweighting of common information (Morris-Shin, AMS);
   - trading against your own signal (LOP Ex. 1-2);
   - competition-driven aggressiveness (Holden-Subrahmanyam);
   - "abundant ≈ MM-observed" (LOP Thm 3);
   - non-monotone informativeness in N (Rostek-Weretka);
   - herding under composition uncertainty (Avery-Zemsky);
   - naive double counting (Eyster-Rabin).
3. **Named shortcuts to use as reference predictors.**
   - "Quotes are posteriors" (the r = 0 Bayes action).
   - Cursed: ignore what prices reveal.
   - Naive / BRTNI: treat the price as independent of your own information.
   - Non-strategic demand: trade on E[v | x_i].
   - Level-k truncation of higher-order beliefs.

---

## 2. Proposed prior worlds

### 2.0 Shared structure prior ("source-listener prior")

One generative prior over Gaussian information structures. Each mechanism (A, B, C) sits on top of it. Per episode:

- **Fundamental and shared sources.**
  - Fundamental v ~ N(0,1).
  - Shared sources s = 1..S, with S ~ U{2..8}. Source s carries sender noise η_s ~ N(0, σ_s²), with σ_s² ∈ {0.1, 0.5, 2} (accuracy level, **hidden**).
- **Public signals.** P ~ U{0..2} public signals y_k = v + ζ_k, with Var ζ_k ∈ {0.25, 1, 4}. Everyone sees them, including the MM in world B.
- **Traders and what they hear.**
  - N traders, log-uniform on [4, 64]. Source popularity π ~ Dir(0.5·1_S).
  - Trader i listens to m_i ~ U{1..3} sources drawn by π and reads x_is = v + η_s + ε_is. The receiver-noise variance ∈ {0.1, 1} is the clarity.
  - With probability 1/2 the trader also has a truly private source x_i0 = v + ξ_i, with Var ξ_i ∈ {0.25, 1, 4}.
  - The Dirichlet popularity produces both abundant sources (heard by many) and scarce ones (heard by one).
- **"Interfering" signals, each with probability about 0.2:**
  - (a) a **public relay** of a shared source (y = v + η_s + small noise). Public information then overlaps private readings, so traders must dedup against it.
  - (b) a **technical** signal: a trader sees η_s + noise, which is information about another source's error (LOP Ex. 2).
  - (c) world B only: a trader sees u + noise (Rochet-Vila; LOP §5).
  - (d) world B only: u correlated with v. The sign decides which LOP Thm 2 limit applies.
- **What is visible and what is hidden.**
  - Visible on tokens: trader id and source ids (random embeddings per episode, as in W2), the listening graph, public values, and each trader's own readings where the role allows.
  - Hidden: the noise levels, plus the mechanism parameters (r in A, σ_u in B, predecessor type in C).
- **Two context modes.**
  - "Described": graph shown, levels hidden, no history.
  - "History": R ∈ {0..32} resolved past rounds with the same structure and fresh v. These show public data (quotes or prices or order flow, public signals, realised v), so the hidden levels can be inferred in context.

All variables are a sparse linear map of independent Gaussians (x = Az), so every structure is a valid LOP / Morris-Shin input.

- **Special cases.** Grossman/Hellwig (only private noise), Morris-Shin (one public signal plus private), Myatt-Wallace (sender plus receiver noise), Foster-Viswanathan (one source heard by all plus private) and LOP Ex. 1-2 (technical signals) are all special cases.
- **Exact Bayes with hidden levels.** It is a mixture over the discrete hypotheses H (levels × mechanism parameter), with one equilibrium solve per hypothesis. This is the same pattern as W2's 2^8 noise hypotheses. |H| should stay at or below about 10³; for example, make per-source accuracy binary when S is large.
- **Training needs no mixture.** Every role below is trained on a proper loss over sampled outcomes: the agent's own payoff, its profit, or NLL of v. The minimiser is then the Bayes object [I, standard PFN argument]. The mixture is needed only for evaluation sets.

A caveat that applies to all three worlds: in the source models, the information structure is common knowledge among the players. Here the other traders know it and the transformer may have to infer it. The target is therefore the Bayes best response of a small newcomer to equilibrium play by informed incumbents. It is not an equilibrium of a game in which everyone is uncertain. That is exact and well defined, but it should be stated in any write-up.

### 2.1 World A: beauty-contest pricing with private, shared and public sources (recommended first)

**Lineage.** Morris-Shin 2002, Allen-Morris-Shin 2006, Angeletos-Pavan 2007, Myatt-Wallace 2012. Keynes's beauty contest: a quote is judged both against the fundamental and against what others quote. This is also the AMS reading of short-horizon traders, where the price is the average expectation of the next price.

**Mechanism.** Trader i quotes a_i to minimise (1−r)(a_i − v)² + r(a_i − ā_{−i})², where ā_{−i} is the mean quote of the others. Hidden r ∈ {0, 0.3, 0.6, 0.85}, plus held-out values. With linear strategies a_i = w_iᵀx_i, the equilibrium is

  w_i = (1−r)·c_i + (r/(N−1))·Σ_ii⁻¹ Σ_{j≠i} Cov(x_i, x_j) w_j,  with c_i = Σ_ii⁻¹ Cov(x_i, v).

This is one linear solve of size K = Σ_i dim(x_i) [C].

- **Uniqueness [I, sketch].** Conditional expectation is an L² contraction, so the best-response map has modulus at most r.
- **Level-k convergence.** Iterating from the Bayes action converges geometrically. In the Morris-Shin case the rate is r·β/(α+β) [C: with α=1, β=3, r=0.6 the public-signal weight goes 0.25, 0.3625, 0.4131, 0.4359, … → 0.4545, ratio 0.45 per step].

**Tasks.**
- **A1, the quoter (the trader pricing on private information).**
  - One output token per trader, so the supervision is dense.
  - Input: the trader's own readings, the listening graph, public signals, and optionally the history rounds.
  - Training loss: the trader's own payoff, computed with realised v and realised ā_{−i}. Its minimiser is the Bayes best response given context.
  - Evaluation target: the exact equilibrium quote, or the H-mixture of best responses when levels or r are hidden.
- **A2, the price reader / aggregator.**
  - Input: all quotes a_1..a_N, plus the graph and public signals.
  - Output: the posterior of v (Gaussian NLL).
  - Target: Gaussian linear given H; a mixture over H.

**In-context inference that a prior-level shortcut cannot do.**
- A1: the equilibrium weight on a source depends on how many other traders hear it and on what else they hear. Own signals alone do not determine it; only the listening graph plus higher-order reasoning does. r and source accuracies must come from the history rounds: how tightly past quotes clustered around public signals and around each other, relative to the realised v.
- A2: every quote leans toward public and abundant sources by an amount that depends on r and on the graph. Averaging quotes double counts that common component. Bayes inverts the lean and dedups shared sources.

**Crux tests (matched pairs in the style of W2 C4).**
1. **Commonality pairs (A1).**
   - Setup: the focal trader has identical readings and sources in both members. The other traders are rewired so that the focal trader's source s is heard by 1 trader in one member and by many in the other.
   - Prediction: the Bayes posterior (r=0 action) is identical across the pair. At r > 0 the equilibrium puts more weight on s in the "many" member, by the amount the linear solve gives.
   - Shortcuts that fail: "quote = posterior" and "fixed overweight by source type" both predict no difference.
2. **r sweep on the Morris-Shin sub-structure (A1).** The public weight should follow α/(α+β(1−r)), including at held-out r. If r is hidden, the model must read it from history: identical current-round inputs, with histories implying different r, should give different quotes.
3. **Depth and level-k (A1, mechanistic) [I, hypothesis].**
   - Hypothesis: if the network computes higher-order expectations iteratively, then the weights decoded at layer ℓ track the level-ℓ iterate w⁽ℓ⁾ = b + M w⁽ℓ⁻¹⁾. Shallow models should then underweight common information by about ρ(M)^L times the initial gap, a closed-form truncation bias that grows with r.
   - Alternative: the network could learn a direct solve, which would not show this layer-by-layer pattern. The test separates the two.
   - Human analogue: lab subjects underweight the public signal relative to Morris-Shin, consistent with level-k (Shapiro-Shi) [S].
4. **Lean-inversion pairs (A2).** Same quotes; histories that imply different r. Bayes reads the quotes back differently, away from the public signal more strongly at high r. Mean-quote and rescaled-mean-quote pools do not change.

**Reference gaps on the draft prior** [C; 300 episodes, N ≤ 32, structure known; MSE of v after reading all quotes]:

| r | Bayes | mean quote | rescaled mean quote |
|---|---|---|---|
| 0 | 0.046 | 0.124 | 0.102 |
| 0.3 | 0.054 | 0.146 | 0.112 |
| 0.6 | 0.055 | 0.208 | 0.147 |
| 0.85 | 0.082 | 0.337 | 0.215 |

The naive pools are 2-4× worse than Bayes, and the gap grows with r. So the world separates the shortcuts. The r=0 row already includes shared-source dedup and public double counting, which is the existing program's question.

**Tractability.**
- About 2.5-4.6 ms per episode in unbatched numpy for N ≤ 32, dominated by building the K×K covariance [C].
- On GPU in batches it should be well under the training step time [I]. K stays at or below about 250 at N = 64.
- Evaluation with hidden levels costs |H| solves per episode, batched.

### 2.2 World B: Kyle-LOP batch auction with strategic insiders

**Lineage.** Kyle 1985 → Holden-Subrahmanyam 1992 → Foster-Viswanathan 1996 / Back-Cao-Willard 2000 → Lambert-Ostrovsky-Panov 2018. Rostek-Weretka 2012 and Vives 2011 give the schedule-competition counterparts.

**Mechanism.**
- Each trader submits d_i = α_iᵀx_i.
- Liquidity demand u ~ N(0, σ_u²), with σ_u² ∈ {0.25, 1, 4} hidden, and optionally correlated with v.
- The MM sees the public signals and D = Σd_i + u, and sets P = β_Mᵀy + β_D·D.
- Equilibrium from the LOP closed form [R, C].

**Tasks.**
- **B1, the insider (dense, one token per trader).**
  - Training loss: realised profit, d(v − P_{−i}) − β_D·d². The minimiser is E[v − P_{−i} | ctx] / (2E[β_D | ctx]), the Bayes-optimal demand given context [I]. When the structure is known, it equals α_iᵀx_i.
  - A second head outputs the trader's valuation E[v | x_i, ctx]. The gap between "what I believe" and "what I trade" is the strategic part, and it can be probed separately.
- **B2, an exchange-side price reader.** It sees the disaggregated orders d_1..d_N. The traders do not know this observer exists, so LOP play is unchanged. It outputs the posterior of v; exact and Gaussian.
  - The plain MM role is not proposed: it sees only D (and y), so its task is a 1-2 feature in-context regression.

**In-context inference.**
- α_i depends on everyone's information through Λ. Who shares sources with whom sets each trader's shading.
- σ_u has to come from history: the spread of order flow not explained by v.
- Technical traders (signals about other sources' errors) must be recognised from the graph.
- For B2, an order is a strategically shaded and possibly sign-flipped function of the signals, so reading it as a noisy report of the trader's posterior is wrong in a computable way.

**Crux tests.**
1. **LOP Example 1 threshold.**
   - Setup: two traders with a common noise driver, θ₁ = v + ρ₁ξ and θ₂ = v + ρ₂ξ.
   - Prediction: α ∝ (1 + 2ρ₂² − ρ₁ρ₂, 1 + 2ρ₁² − ρ₁ρ₂) [R]. Trader 1 stops trading at ρ₁ = 2ρ₂ + 1/ρ₂ and trades against its signal beyond that. Sweeping ρ₁, the model's sign change should land at this threshold.
   - Shortcut that fails: a "trade on your posterior" agent never flips.
2. **LOP Example 2.** The technical trader (sees only ξ) has demand in ratio (2, −1) to the fundamental trader [R, C].
3. **Competition pairs (Holden-Subrahmanyam).**
   - Setup: the focal trader and its source are held fixed; the number of other traders on that source varies from 0 to k.
   - Prediction: with identical insiders, per-trader intensity is σ_u/√(NΣ₀), and the closed form gives the general case.
   - Shortcut that fails: non-strategic demand is constant across the pair.
4. **Scaling (LOP Thm 2-3).**
   - Prediction: as an abundant group grows beyond the training N, the price converges to the alternative market in which the MM observes the abundant signal.
   - Sign condition: with Cov(u,v) < 0, the price stops using u in the limit.
   - Rostek-Weretka warns that informativeness need not rise with N, so scaling is scored against the per-N exact target. It must not be assumed to improve with N.
5. **Strategic dedup [I].** Test whether the model's implied weights follow Λ⁻¹Σ_θv (the equilibrium) or Σ_θθ⁻¹Σ_θv (Bayes regression), on structures where the two differ.

**Reference statistics on the draft prior** [C, 300 episodes, independent u]:
- Strategic weight sign opposite to the trader's own Bayes weight:
  - 9.9% of non-public signal components;
  - at least one such component in 80% of episodes;
  - 100% of public components. They are traded against because the MM already prices them, which is the expected "trade relative to the price" effect.
- Mean correlation between a trader's strategic and Bayes linear forms: 0.61.
- So strategic demand is far from "trade on your posterior" across most of the prior.

**Tractability.**
- 0.3-0.45 ms per episode in unbatched numpy at N ≤ 32 (one K×K solve plus a scalar quadratic) [C].
- Equilibrium conditions verified to about 1e-15 on random structures [C].
- Profit-based training has higher target variance than payoff MSE in world A. If it stalls, the exact α_i is available as a dense regression target [I].

### 2.3 World C: sequential dealer market with overlapping information and predecessors of unknown type

**Lineage.** Glosten-Milgrom 1985; Hanson's LMSR; Chen et al. 2010; Ostrovsky 2012; BHW / Banerjee 1992; Avery-Zemsky 1998; Easley-O'Hara / EKOP; Eyster-Rabin 2010 and Eyster-Rabin-Vayanos 2019.

**Mechanism (Gaussian version).**
- Traders arrive in random order. Each sees the price path and its own readings, and moves the price to its posterior (LMSR-style myopic reporting).
- Shared noise breaks conditional independence, so the Chen et al. truthfulness result does not cover this case. Myopic reporting is an assumption here and is labelled as such [I].
- Predecessor processing type is hidden, one per episode: Bayesian (knows the structure); naive (treats the price as independent of its own information, i.e. double counts); or cursed (ignores the price).
- Given the type, every price is linear in the signals, so the target is exact Gaussian generalised least squares (GLS), mixed over 3 × |levels| hypotheses.

**Tasks.**
- **C1, the next trader or dealer.** It sees its own readings, the price path with trader and source tags, and optionally history rounds. It outputs the posterior of v.
- **C2, discrete Glosten-Milgrom / Avery-Zemsky variant.**
  - v ∈ {0, 1}, with event uncertainty adding v = ½.
  - Composition uncertainty: the share of informed traders and their accuracy come from a small grid.
  - Trades are buy / sell / no-trade at a Bayesian MM's bid/ask.
  - Target: the MM's posterior over (v, composition), by exact forward recursion with cost O(T·|V|·|C|).

**In-context inference.**
- Is my source already in the price (did an earlier trader share it)?
- How did my predecessors process information? With naive predecessors the last price over-counts early signals, so Bayes needs the whole path, not only the last price.
- In C2: composition, read from trade and no-trade patterns.

**Crux tests.**
1. **Own-source-in-price pairs.**
   - Setup: same readings and same last price; in one member my source appeared earlier, in the other it did not.
   - Prediction: Bayes discounts my shared component when it is already in the price.
   - Shortcut that fails: "independence" double counts.
   - On the natural distribution this gap is moderate: KL(Bayes ‖ independence) has median 0.047 nats, p90 0.087, and is above 0.05 in 47% of episodes [C]. That is comparable to W2's prov_naive gap, so constructed pairs are needed for power.
2. **Predecessor-type pairs.** Identical current path and readings, with histories implying Bayesian vs naive predecessors, should give different weights on the path.
3. **Herding (C2).** On sequences that end in a herd under composition uncertainty, compare against naive imbalance counting. Avery-Zemsky predict short-run mispricing.

**Tractability.** GLS through per-source sufficient statistics, O(T·k); the discrete recursion is trivial. This is the cheapest of the three worlds.

---

## 3. Ranking, risks and novelty

### Ranking

1. **World A (beauty-contest pricing).**
   - It is the closest match to Kelvin's wording: a trader pricing on private information, with private, shared and public signals, and public information that interferes. The "interference" is the best-known result in this literature: public information crowds out private information and is overweighted (Morris-Shin, AMS, Angeletos-Pavan, Myatt-Wallace).
   - The target is one unique linear solve, and the trader's payoff is the training loss.
   - One knob, r, turns the strategic content off (r = 0 is pure Bayes aggregation, i.e. the current program) and on, within the same world and the same structure prior.
   - It carries the sharpest mechanistic hypothesis: higher-order beliefs computed layer by layer, with closed-form level-k iterates to probe against.
   - Main risk: in the "described" mode A1 is amortised equilibrium computation rather than structure inference. The fix is hidden levels and hidden r with history rounds, and the commonality pairs isolate the inference.
2. **World B (Kyle-LOP).**
   - It has the strongest Econometrica pedigree for "informationally complex" markets: a unique closed form for arbitrary Gaussian structures.
   - Its behaviours are discontinuous with closed-form thresholds (sign flips), which make the cleanest cruxes.
   - It connects to dedup through Λ.
   - It ranks second because the price-setter role is trivial, because the trader's output is a quantity rather than a price, and because profit-based training is noisier.
   - It is the natural second world, and it reuses the same prior and code.
3. **World C (sequential dealer).**
   - It is closest to W2/W2c. Its dedup crux (own source in price) is the same capability that W2c failed: C7 dedup about 0. It will likely inherit that difficulty.
   - Its economic novelty is predecessor-type inference and the discrete herding variant.
   - Treat it as the bridge from W2, not as the headline.

A later stage could put a mechanism-id token on the shared prior and train A, B and C together. That tests whether the network separates "what I know" from "how others use what they know" [I]. I would not start there.

### Risks and limits to state up front

- **Non-Gaussian structure is mostly out of reach.**
  - Hellwig-type REE with shared noise (Lou et al.) needs a fixed point whose uniqueness I have not seen proven. Do not use it as an exact target without a per-episode numerical check.
  - Breon-Drish gives non-normal closed forms, but only in narrow classes.
  - XOR-type non-separable structures (Ostrovsky) should only be a negative control.
- **Prices and orders are scalar by design.** Here the market interface is the object of study. This does not replace the rich-message aggregation line. It gives the economically grounded benchmark of what scalar prices can and cannot carry (LOP Thm 2-3, Ostrovsky).
- **Mixture size.** Exact evaluation under hidden levels needs |H| equilibrium solves per episode. Keep |H| at or below about 10³ and batch on GPU.

### What would be novel, and what exists

**Existing work on learning in markets** (all A or S today):
- Friedrich-Teichmann 2020 (arXiv:2006.13889): deep agents converge to the Kyle single-period equilibrium, including some non-Gaussian value distributions. One fixed game.
- Colliard-Foucault-Lovo (RFS 2026): Q-learning market makers in a Glosten-Milgrom/Kyle-like setting learn adverse-selection pricing but keep a supra-competitive markup.
- Dou-Goldstein-Ji (NBER w34054, 2025): reinforcement-learning informed speculators in a Kyle framework collude by under-reacting to information; prices become less informative.
- Ellis-Piccione-Zhang (arXiv:2607.13670, July 2026): Chow-Liu-tree "machine learning" traders in a Hellwig market. The price fails to aggregate what the algorithm extracts.
- LLM agents in experimental asset markets (Henning et al. arXiv:2502.15800; del Rio-Chanona et al. arXiv:2505.07457; arXiv:2604.18602; arXiv:2604.18373): behavioural, mostly bubbles and feedback markets. None was about aggregating private information.
- Deep equilibrium solvers in economics (deep equilibrium nets; "Equilibrium World Models", arXiv:2606.23463) solve one model at a time.
- On the ML side, the in-context-learning-as-Bayes and PFN literature has no strategic or equilibrium targets. Bayesian-CFR-type solvers are not amortised.

**Novel, as far as my searches found:**
- **(i) Amortisation across a prior over information structures**, where each episode is a different Kyle or beauty-contest game, with an exact per-episode equilibrium or Bayes target.
- **(ii) Matched-pair tests of in-context structure inference in strategic settings**: commonality pairs, competition pairs, predecessor-type pairs.
- **(iii) Mechanistic questions with closed-form intermediate objects.** Are higher-order beliefs computed as iterated average expectations (level-k by depth)? Is strategic shading computed as the Λ-modified regression?
- **(iv) For economists:** a trained network as a bounded-rationality trader whose deviations from equilibrium can be measured per information structure and compared with level-k, cursed and naive benchmarks.

**Search limits.**
- I searched for transformers or neural networks learning Kyle / Glosten-Milgrom / Bayes-Nash equilibria in context; for deep learning of noisy-REE models; and for LLM agents in private-information markets.
- I did not do a systematic citation chase of Friedrich-Teichmann or Colliard et al.
- I did not search the computer-science literature on learned prediction-market agents beyond Chen et al.
- "Not found" means not found by these searches.

---

## Sources

Primary texts read: [LOP NBER WP 20516](https://www.nber.org/papers/w20516); [Vives ch. 4](https://blog.iese.edu/xvives/files/2011/09/Chapter-4-XV-InformationLearning-in-Markets2.pdf); [Allen-Morris-Shin 2006](https://economics.mit.edu/sites/default/files/publications/morris-beautycontestsanditeratedexpectations.pdf).

Abstract or secondary pages:
- Classic market models:
  - [LOP, Econometrica](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA12635)
  - [Foster-Viswanathan 1996](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7915)
  - [Back-Cao-Willard 2000](https://www.semanticscholar.org/paper/Imperfect-Competition-among-Informed-Traders-Back-Cao/00ce29fbbe9987caa2c18bf9af97ec785be188da)
  - [Holden-Subrahmanyam 1992](https://econpapers.repec.org/RePEc:bla:jfinan:v:47:y:1992:i:1:p:247-70)
  - [Admati 1985](https://www.gsb.stanford.edu/faculty-research/publications/noisy-rational-expectations-equilibrium-multi-asset-securities-markets)
  - [Kyle 1989](https://www.semanticscholar.org/paper/Informed-Speculation-with-Imperfect-Competition-Kyle/d6ed3c588f73ac00212c24de6e906e1ea0cf8419)
  - [Glosten-Milgrom 1985](https://www.semanticscholar.org/paper/Bid,-ask-and-transaction-prices-in-a-specialist-Glosten-Milgrom/5827ca4a5ac97e717fb5768f313079e813cebe86)
  - [Grossman-Stiglitz 1980](https://www.aeaweb.org/aer/top20/70.3.393-408.pdf)
  - [Grossman 1976](https://ideas.repec.org/a/bla/jfinan/v31y1976i2p573-85.html)
  - [Diamond-Verrecchia 1981](https://ideas.repec.org/a/eee/jfinec/v9y1981i3p221-235.html)
  - [Hellwig 1980 (citing paper)](https://www.econtheory.org/ojs/index.php/te/article/viewFile/20210161/29623/853)
- REE extensions and market structure:
  - [Rostek-Weretka 2012](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA9573)
  - [Vives 2011](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA8126)
  - [Breon-Drish 2015](https://www.ssrn.com/abstract=2038080)
  - [Lou et al. 2019](https://ideas.repec.org/a/eee/jetheo/v183y2019icp594-624.html)
  - [Eyster-Rabin-Vayanos 2019](https://www.ssrn.com/abstract=3117986)
  - [Goldstein-Yang 2015](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12226)
  - [Bergemann-Heumann-Morris](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3276348)
- Coordination and public information:
  - [Morris-Shin 2002 (SNB WP restating the formula)](https://www.snb.ch/public/asset/it/www-snb-ch/publications/research/working-papers/2017/working_paper_2017_17/publications0_it/working_paper_2017_17.n.pdf)
  - [Shapiro-Shi level-k](https://www.economics.utoronto.ca/xianwenshi/Levelk.pdf)
  - [Angeletos-Pavan 2007](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2007.00783.x)
  - [Myatt-Wallace 2012](https://ideas.repec.org/a/oup/restud/v79y2012i1p340-374.html)
- Herding, cascades and dealer markets:
  - [Avery-Zemsky 1998](https://dash.harvard.edu/entities/publication/d4123d33-86d6-4914-8cbd-157fca2b0c8f)
  - [BHW 1992](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1286306)
  - [Banerjee 1992](https://www.jstor.org/stable/2118364)
  - [EKOP / PIN](https://journal.r-project.org/articles/RJ-2013-008/)
  - [Chen et al. 2010](https://dash.harvard.edu/entities/publication/73120378-9635-6bd4-e053-0100007fdf3b)
- Learning agents and solvers:
  - [Friedrich-Teichmann 2020](https://arxiv.org/abs/2006.13889)
  - [Colliard-Foucault-Lovo](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4252858)
  - [Dou-Goldstein-Ji](https://www.nber.org/papers/w34054)
  - [Ellis-Piccione-Zhang 2026](https://arxiv.org/abs/2607.13670)
  - [Equilibrium World Models](https://arxiv.org/abs/2606.23463)
  - [LLM market experiments](https://arxiv.org/pdf/2505.07457)
