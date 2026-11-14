# Scope: an economist's reading of "traders inside a neural network"

Lens: firms and markets (Coase, Alchian-Demsetz, Holmström, Hayek machine,
market-based RL, national accounts). Scoping only: what the larger study should
investigate, not answers. 2026-09-27. Inputs read: traders-net/CONTEXT.md,
logical-soup/REPORT-SOUP.md sections 1-3, wealth-net/README.md, books.py,
train.py and results/summary.json at checked_at 2026-09-27T21:50:46Z
(wealth-net/RESULTS.md does not exist yet). Citations marked (unverified) were
not checked in this session; the four marked (checked) were confirmed by web
search today.

## 0. Summary in five lines

1. The wealth-net books record **gross sales on a tangent**, z·(y−p). That is a
   learning signal. It is not value added, and at any stationary point of
   training it is exactly zero. Wealth should be booked as **value added along a
   path**: log-score gain over an explicit prior, split by Aumann-Shapley.
2. Backprop is a **Groves mechanism**: every parameter is paid the whole
   marginal effect, so payments over-exhaust output by the depth factor. A
   conserved (budget-balanced) wealth ledger cannot also keep every unit's
   selfish update equal to the global gradient (Holmström 1982;
   Green-Laffont). Such a ledger needs a budget breaker, the market maker, and
   the study should measure the efficiency cost of that choice.
3. Dropping the Harberger tax is safe while no asset is contested (a fixed
   architecture with books only). Dropping **every sink** is not: wealth goes
   stale and nominal wealth inflates. wealth-net already shows this: wealth
   rank correlation over the second half is 0.98 while neuron income rank
   correlation is −0.27. The minimal sink is depreciation (demurrage) or a
   metered compute price. Neither of these is Harberger.
4. MoE load balancing is already fiscal policy. The auxiliary loss is a
   redistributive tax/subsidy. DeepSeek's auxiliary-loss-free bias is a
   Walrasian price on expert capacity. Router collapse is either an efficient
   Kelly concentration or an Arthur-style lock-in, and that question can be
   tested.
5. Minimal economy with interesting dynamics: Kelly wealth, visible prices,
   one exogenous inflow (information measured in nats), one sink (compute at a
   nats-per-FLOP price), entry and exit, niches (many questions), and a regime
   shift. Blume-Easley predicts a monoculture without niches or
   nonstationarity.

## 1. Accounting identities the study should fix before measuring anything

**A. Settlement: tangent vs integral.** An LMSR trader who moves log-odds from
ℓ0 to ℓ0+z earns the cost-function payoff ∫(y − σ(ℓ0+t))dt over [0,z], which
equals the log-score gain log p(y|ℓ0+z) − log p(y|ℓ0). wealth-net pays the
tangent z·(y−p) instead. For a linear readout z = w·h + b, the first-order
condition of the loss gives Σ_examples (y−p)·h = 0 and Σ(y−p) = 0, so the
population tangent revenue satisfies R = w·Σ(y−p)h + bΣ(y−p) = 0 at every
stationary point. Tangent wealth therefore measures cumulative disequilibrium
(learning), not ongoing value. The same result holds unit by unit: for a unit
entering homogeneously, revenue = −∂L/∂log(scale of that unit). Proposed
books: expenditure side = cumulative log-score gain over a named prior (base
rate or the previous layer's price), in nats. This quantity is bounded and
equals the maker's loss.

**B. National accounts: gross sales vs value added.** At any complete cut,
settlement by expenditure (the output) = payments to primary factors (income)
= Σ value added. Units with constant returns (ReLU neurons, linear maps,
bias counted as a supplier) pass through exactly what they receive. Their
value added (profit) is zero by Euler's theorem. Under the current books all
value lands on the primary factors: input embeddings, question-id embeddings
and biases. "Wealth = cumulative revenue" is **gross sales**, which is why a
big trader's wealth is a plain sum only within a layer (README) and double
counts across layers. For Kelvin's "big trader = sum of its parts" to hold for
*any* set, including non-hierarchical ones, the ledger must book value added.
That needs a choice of what the factors are.

**C. Product exhaustion (Wicksteed/Euler).** Marginal-product payments
exhaust output only under degree-1 homogeneity. Two failures:
- *Parameters as factors.* A bias-free ReLU net of depth L is homogeneous of
  degree L in its weights, so Σθ·∂z/∂θ = L·z. Paying every parameter its
  marginal product overpays by the depth factor. This is increasing returns,
  and it is the Groves structure in point 2 of the summary.
- *Non-homogeneous consumers.* Attention softmax, LayerNorm and GELU break
  the sum. wealth-net's residual at those cuts is the textbook adding-up
  problem, and someone must be its residual claimant (Alchian-Demsetz).

Aumann-Shapley cost sharing, i.e. integrated gradients from a baseline, meets
the efficiency (adding-up) axiom exactly and agrees with identity A when the
baseline is the prior. It is the natural exact ledger.

**D. Ownership consolidation.** Supplier payments are trade, not equity.
Ownership means a residual claim plus control. With an ownership matrix Θ
(Θ_ij = share of j owned by i, each column summing to ≤ 1, the rest held
outside), consolidated value is V = W_own + ΘV, so V = (I−Θ)⁻¹W_own. This is
the Leontief inverse; it is well defined when the spectral radius of Θ is
below 1, and it handles cyclic cross-holdings. Total economy wealth = Σ
outside-held equity, not Σ V (the market-cap double counting problem).
Backprop does not produce Θ. Something has to: who funded a birth, who holds
a contested slot.

**E. The SGD norm identity: wealth may already be the parameter norm.** Plain
SGD gives ‖w_{t+1}‖² = ‖w_t‖² − 2η w·∇L + η²‖∇L‖². For a unit's outgoing weight
group, tangent revenue is −w·∇L, so cumulative revenue ≈ (‖w_T‖² − ‖w_0‖²)/2η.
In ReLU nets ‖w_in‖² − ‖w_out‖² is conserved under gradient flow (Du, Hu, Lee
2018, https://arxiv.org/abs/1806.00900, unverified), so either side works. With
L2 weight decay λ, stationary revenue = λ‖w‖²: **weight decay is a holding
tax, and the stationary state is the zero-profit condition** (income = rent on
capital). wealth-net uses Adam without weight decay, so the identity holds
only approximately there; this is testable. If it holds, "gate execution by
wealth" = gate by norm, and the smallest modification is weight normalization
with a log-gain (https://arxiv.org/abs/1602.07868, unverified). Gradient steps
on log-gain are then multiplicative (Hedge/Kelly) updates of each unit's
budget.

**F. Hulten vs Baqaee-Farhi.** A unit's first-order revenue share is its Domar
weight. Hulten's theorem says Domar weights give the aggregate effect of
*small* shocks (Hulten 1978, https://doi.org/10.2307/2297272, unverified).
Removing a unit is a *large* shock, and its effect depends on elasticities of
substitution (redundancy) at second order (Baqaee & Farhi 2019,
https://doi.org/10.3982/ECTA15202, unverified). The running ablation test
("remove the most negative-wealth units") is really a Hulten test. Expect it to
fail where units substitute for each other, and read the failure as a
substitutability measurement, not as proof the books are wrong.

## 2. What breaks without a tax (and what does not)

| Problem | Needs Harberger? | Minimal remedy | Evidence / test |
|---|---|---|---|
| Stale stocks and entrenchment: early windfalls dominate | no | depreciation at rate δ (demurrage) | wealth-net: wealth rank corr 0.98 vs neuron income rank corr −0.27 in the 2nd half |
| Nominal inflation: R is booked from nowhere, total wealth grows monotonically | no | name the maker (prior); report shares W_i/ΣW | identity A |
| Unpriced commons: residual-stream bandwidth, interference | no, Pigouvian | price on capacity or a second-order charge | first-order books miss externalities; they sit in Hessian cross terms |
| Contested slots: expert capacity, a fixed K of units | **yes**, or an auction | Vickrey/assignment auction for the slot (BASE, Chang et al.) | only once births or replacements compete for fixed slots |
| Team production and budget balance | no, but needs a budget breaker | the maker as residual claimant; accept the efficiency loss | Holmström 1982 |
| Rich lineages spread one demand everywhere (soup rule 30 at 0.26-0.29) | no | per-question budgets (niche accounts); wealth buys share, not reach | soup §3.2 |

Framing for Kelvin: "no Harberger now" is consistent with a
fixed-architecture books-only phase, because no asset is contested yet.
Harberger (or an auction) becomes necessary the moment units compete for a
fixed set of slots. MoE already spends a redistributive tax (the auxiliary
loss); the principled replacement is a price, not no policy.

## 3. Compute as wealth: what should wealth buy?

The soup answers this. 64× more fuel per trader moved Brier by about one seed
standard deviation, while traders per question moved periodic from 0.20 to
0.08 (§3.1, §3.6). Wealth should buy **market share** (copies, routed tokens,
top-k slots, lottery tickets per question), not per-trader depth. "Compute =
wealth" also needs an **exchange rate** in nats of log-score per FLOP. This is
the central free parameter of the economy and should be swept. If returns to
wealth are increasing (wealth buys compute, which buys more wealth), the
economy tips to winner-take-all. Coexistence needs diminishing returns per
question, which is also what Blume-Easley require for survival of
non-best traders.

## 4. Minimal economy with interesting dynamics (proposed spec)

Ingredients, each necessary for a stated reason:
1. **Kelly traders with visible prices.** Price = wealth-weighted pooling
   (parimutuel = Eisenberg-Gale program; Storkey et al. 2012 alpha-mixtures,
   https://arxiv.org/abs/1206.6443, checked). Traders emit corrections to the
   visible price (idea 2).
2. **One inflow:** the maker's loss = log-score gain over the prior (nats).
3. **One sink:** compute at price c per participated question. No tax.
4. **Entry and exit:** split above a threshold, die below a floor. Births may
   be co-funded, which gives partial ownership by funding (identity D).
5. **Niches:** many questions and families (soup streams). Blume-Easley
   predicts monoculture in a stationary complete market; niches make the
   market effectively incomplete.
6. **Nonstationarity:** a regime shift halfway (swap which families are
   periodic). Without it the economy converges to a static rich set.

Two learning regimes on the same economy: (a) selfish, each trader ascends its
own log-wealth; (b) cooperative, backprop on the pooled loss. The gap between
them is the measured cost of budget balance.

## 5. Questions (ranked)

See the structured summary; the most important are identity A/E audits on
existing checkpoints (hours), the budget-balance impossibility (theory, days),
market selection vs niches (experiment, days), and router collapse as lock-in
(experiment, days).

## 6. Prior art to check

- Relevance conservation, i.e. money flowing backward with conservation: LRP,
  Bach et al. 2015, https://doi.org/10.1371/journal.pone.0130140 (unverified).
  The LRP-ε rule's ε is literally a sink. DeepLIFT summation-to-delta,
  https://arxiv.org/abs/1704.02685 (unverified). Integrated gradients =
  Aumann-Shapley, https://arxiv.org/abs/1703.01365 (unverified).
- Neuron Shapley, https://arxiv.org/abs/2002.09815 (unverified): baseline for
  the ablation test.
- Balduzzi, Cortical prediction markets, https://arxiv.org/abs/1401.1465
  (checked): neurons as scoring-rule agents that backpropagate incentives.
- Storkey, Machine learning markets (AISTATS 2011) and Storkey, Millin, Geras
  2012, https://arxiv.org/abs/1206.6443 (checked): wealth updates implement
  Bayesian or mixture updates depending on payoff structure.
- Chang, Kaushik, Weinberg, Griffiths, Levine 2020, cloned Vickrey society,
  https://arxiv.org/abs/2007.02382 (checked). Survey: Market-based
  architectures in RL and beyond, https://arxiv.org/abs/2503.05828 (found in
  search, not read).
- Baum, Hayek machine / "Toward a model of mind as a laissez-faire economy of
  idiots" 1996 (unverified; see the survey above). Holland, bucket brigade
  (1985, unverified). Kwee, Hutter, Schmidhuber 2001, market-based RL,
  https://arxiv.org/abs/cs/0105025 (unverified). Schmidhuber 1989, neural
  bucket brigade with conserved "weight substance" (unverified).
- Wang et al. 2024, auxiliary-loss-free load balancing,
  https://arxiv.org/abs/2408.15664 (checked); primal-dual analysis of it,
  https://arxiv.org/abs/2512.03915 (found in search, not read). Expert
  Choice, https://arxiv.org/abs/2202.09368 (unverified). BASE layers,
  https://arxiv.org/abs/2103.16716 (unverified). Fine-grained MoE scaling
  (Coase's firm-size question in ML), https://arxiv.org/abs/2402.07871
  (unverified).
- Economics: Coase 1937, https://doi.org/10.1111/j.1468-0335.1937.tb00002.x;
  Alchian & Demsetz 1972, AER 62(5); Alchian 1950, "Uncertainty, evolution,
  and economic theory", JPE; Holmström 1982, Moral hazard in teams,
  https://doi.org/10.2307/3003457; Blume & Easley 2006, Econometrica 74(4),
  "If you're so smart, why aren't you rich?"; Sandroni 2000, Econometrica;
  Eisenberg & Gale 1959, pari-mutuel consensus, Ann. Math. Stat.; Arthur
  1989, lock-in, Economic Journal; Green & Laffont 1979 (Groves budget
  imbalance); Arrow-Hurwicz and Scarf 1960 (tâtonnement stability); Birnbaum,
  Devanur, Xiao 2011, proportional response dynamics (all unverified).

## 7. Ideas in CONTEXT.md I think are wrong or overstated

See the structured summary (risks_or_wrong_ideas).
