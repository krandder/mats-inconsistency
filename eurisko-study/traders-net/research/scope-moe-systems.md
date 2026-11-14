# Scope: wealth instead of the router or the balancing loss (MoE systems lens)

Scoping notes for the traders-net study, 2026-09-27. Lens: someone who has
trained MoE models and fought router collapse, aux losses, capacity factors and
token dropping. Scoping only: the questions and experiments a larger workflow
should run, with costs. Links that I opened this session are given as plain
URLs. Items marked "(unverified)" are from memory.

Inputs read: ../CONTEXT.md; ../../logical-soup/REPORT-SOUP.md sections 1-3.
../../wealth-net/RESULTS.md does not exist yet. I peeked at
../../wealth-net/results/summary.json, checked at 2026-09-27T21:50Z, seed 1,
step 7500: head wealth Gini 0.78; about 10% of neurons are "parasites";
wealth rank correlation across the second half of training is 0.97-0.99.
Wealth ranks lock in early. That matters for any scheme that gates on wealth.

## 0. Three observations that shape the study

**A. A softmax router already runs replicator dynamics on per-token P&L.**
Take the top-k gate over the selected set S: h = sum_j g_j E_j(x) and
g = softmax(z_S). Let r_j = <E_j(x), -dL/dh>, the revenue expert j would earn
at unit gate. Then

    -dL/dz_i = g_i (r_i - sum_j g_j r_j)      for i in S,   0 otherwise.

So the router logit is a context-dependent log-wealth, and each step moves it
by the expert's P&L relative to the gate-weighted market. That is Hedge, or
replicator dynamics. (The algebra is a two-line derivation; the larger study
should check it numerically in E0.) What makes routers collapse is not the
lack of a market. It is **censored feedback**: an expert outside the top k
gets zero revenue signal. This is Hedge run with bandit feedback and no
exploration, so the rich get richer. In this reading the Switch/GShard aux
loss N·sum f_i P_i is a crude exploration bonus plus a hardware quota. The
principled replacements are (i) exploration with importance-weighted revenue,
as in EXP3, and (ii) a congestion price for capacity. These are two separate
jobs.

**B. The balancing loss does two jobs, and wealth can take over only one of them.**
- Hardware balance (capacity, stragglers under expert parallelism). This
  needs a price that clears supply. DeepSeek's loss-free balancing already is
  one: a per-expert bias moved by load deficit, which is tâtonnement
  (https://arxiv.org/abs/2408.15664). It has since been analysed as a
  primal-dual method for the assignment problem
  (https://arxiv.org/abs/2512.03915). "Price Equilibrium Routing" does the
  same thing with explicit prices
  (https://link.springer.com/chapter/10.1007/978-981-95-4972-6_1).
- Anti-collapse and specialisation. Wealth fits here: a P&L-based budget,
  plus bankruptcy and birth.
So "replace the balancing loss with wealth" is only half right. Keep a price
for capacity and use wealth for the budget.

**C. Global wealth repeats the soup's Rule-30 failure.** In the soup, lineages
that got rich on biased families put the same demand on pseudo-random ones and
priced them below chance. A context-free log W_i bias on the gate does exactly
this: a globally rich expert gets routed to tokens where it has no edge.
Wealth has to be kept per market (per token cluster, domain, or family), or
it has to stay inside the context-dependent router logit, which is where A
says it already lives. The study should measure this failure, not assume it
away.

## 1. Candidate smallest modifications (ordered by code delta)

| id | change to a standard top-k MoE | lines | what "wealth" means |
|---|---|---|---|
| M0 | books only: log r_i per token per active expert (free from the existing backward), cumulative per expert and per family | ~30 | cumulative revenue = log-wealth |
| M1 | drop the aux loss; gate logit += beta·logW_i (EMA of revenue, per expert, optionally per cluster); keep a loss-free congestion bias for capacity | ~20 | Hedge prior on the gate |
| M2 | expert-choice routing with capacity_i ∝ W_i instead of equal capacity ("compute = wealth" literally) | ~5 on top of EC | FLOPs each expert may buy |
| M3 | exploration: each token also runs one sampled non-selected expert at zero gate (a shadow trade); importance-weighted revenue credited to it; no aux loss | ~40, +1/k FLOPs | fixes censored feedback |
| M4 | bankruptcy and birth: when W_i < threshold, re-initialise expert i as a noisy clone of a rich expert and split the wealth (the soup's replication rule) | ~30 | selection, not gradient |
| M5 | PEER-style (https://arxiv.org/abs/2407.04153): 16k-262k single-neuron experts, add beta·logW to product-key scores; bankruptcy re-inits one row | ~40 on a PEER impl | "thousands of traders" regime |

Mechanisms to hold constant or ablate: the router z-loss (ST-MoE, bounded
bets / logit clipping as the budgeter, https://arxiv.org/abs/2202.08906
(unverified id)), capacity factor and token dropping (dropped token = unfilled
order), shared experts (DeepSeekMoE shared plus fine-grained experts,
https://arxiv.org/abs/2401.06066 (unverified id)). The shared expert is the
"shared core computation" Kelvin wants. The routed experts are corrections on
top of it, which is idea 2's "traders only improve here and there" already
built into a production architecture.

## 2. An accounting problem specific to MoE: the gate doubles revenue

The gate multiplies the expert output, and gate·expert is homogeneous of
degree 2 in (g, E). With a·dR/da books, the router side gets g·dR/dg = r and
the expert side gets sum E·dR/dE = r. Together that is 2r, while the residual
stream paid r (Euler's theorem). Conservation therefore breaks by a factor of
2 at every gate, and not as a small residual the way attention softmax does in
wealth-net. The books need a convention before any MoE result is read:
(a) the router is a broker and is paid nothing; (b) Euler split, half each;
(c) Shapley over the pair. Choose (a) for M0-M2 and report (b) as a
sensitivity check. The same doubling hits GLU/SwiGLU MLPs (gate·up product)
and attention (QK product). wealth-net's ReLU MLP avoids it by construction.
This is worth a paragraph in the study, because every modern FFN is gated.

## 3. Substrates and costs

1. **Soup streams** (wealth-net's streams.py). Families are known markets,
   which gives ground truth for specialisation (MI between expert and family)
   and for the Rule-30-style cross-market leak. Model: wealth-net's 2-layer
   d=64 transformer with the MLP swapped for an 8-64-expert MoE. Minutes per
   run on one L4 or on CPU. The grid of 6 routers × 3 seeds × 2 expert counts
   is about 36 runs, a few GPU-hours: under $5 at spot L4 prices
   (unverified price; the soup's whole GPU bill was $9.38).
2. **Language modelling at nanoGPT scale.** 6-8 layers, d=384-512, 16-64
   experts top-2, 50-150M total params, 1-2B tokens of FineWeb-edu or
   TinyStories. About 2-6 L4-hours per run, or 1-2 A100-hours. Grid of 5
   routers × 3 seeds = 15 runs, roughly $30-80. This scale is where the known
   pathologies (collapse, dead experts, loss spikes) reproduce. Small-scale
   routing testbeds exist:
   https://arxiv.org/abs/2604.07030 (MoE Routing Testbed, found by search,
   not read).
3. **An existing trained MoE, measurement only.** OLMoE-1B-7B has open
   weights and checkpoints (https://arxiv.org/abs/2409.02060 (unverified id)).
   Run M0 books over about 10M tokens with forward and backward, no update:
   about 1-2 A100-hours. The mid-training checkpoints show whether wealth
   concentration comes before router collapse or specialisation.

Engineering note: M2 (unequal capacity) and M5 need dropless or grouped-GEMM
kernels such as MegaBlocks (https://arxiv.org/abs/2211.15622 (unverified
id)). On a single GPU at small scale, a plain loop over experts is fine, and
throughput is not the metric.

## 4. Metrics every run reports

Validation loss at matched active FLOPs. Load CV and max/mean load. Dropped
tokens. Dead experts (under 0.1% of tokens). Expert-family mutual information
(soup substrate). Cross-market leak: the loss change on families where an
expert has negative revenue but is routed. Wealth Gini over time and
wealth-rank lock-in step. Router calibration: does g_i rank-predict realised
r_i? Ablation predictivity: does deleting an expert raise loss in proportion
to its wealth? This extends wealth-net's parasite test. Loss spikes and
router-logit magnitude.

## 5. Prior art to check (market-based routing is not new; the wealth/Kelly angle is thinner)

- Loss-free balancing, a price by another name: https://arxiv.org/abs/2408.15664 ; theory https://arxiv.org/abs/2512.03915
- Congestion-game view of routing (softmax as a mean-field equilibrium): https://arxiv.org/abs/2604.04230
- Mixture of Bidders (VCG auctions replace the gate, continual learning, small scale): https://arxiv.org/abs/2512.10969
- Price Equilibrium Routing: https://link.springer.com/chapter/10.1007/978-981-95-4972-6_1
- Routing-free MoE (experts decide their own activation): https://arxiv.org/abs/2604.00801
- ReMoE, ReLU routing with an adaptive L1 budget: https://arxiv.org/abs/2412.14711
- Expert Choice: https://arxiv.org/abs/2202.09368 ; BASE layers (assignment with equal quotas, Bertsekas auction algorithm) https://arxiv.org/abs/2103.16716 (unverified id)
- PEER, a million experts: https://arxiv.org/abs/2407.04153 ; fine-grained scaling https://arxiv.org/abs/2402.07871
- Soft MoE https://arxiv.org/abs/2308.00951 (unverified id); Mixture-of-Depths https://arxiv.org/abs/2404.02258 (unverified id); ACT https://arxiv.org/abs/1603.08983 (unverified id); PonderNet https://arxiv.org/abs/2107.05407 (unverified id)
- Gradient estimates for unselected experts (SparseMixer / GRIN): https://arxiv.org/abs/2310.00811 (unverified id). These are the non-economic answer to censored feedback, and M3 must beat them.
- Market-based learners: Baum's Hayek machine https://link.springer.com/article/10.1023/A:1007593124513 ; Kwee, Hutter, Schmidhuber https://arxiv.org/abs/cs/0105025 ; Chang et al., cloned Vickrey society https://arxiv.org/abs/2007.02382 ; Sudhir and Tran-Thanh, market architectures generalise NNs, and they cite logical induction: https://arxiv.org/abs/2503.05828 ; Economy of Minds (2026) https://arxiv.org/abs/2606.02859. Holland's bucket brigade is the ancestor of idea 6's backward money flow (unverified cite).
- Gated Linear Networks, where every neuron predicts the target and learns locally by geometric mixing, i.e. logit-sum pooling with local credit: https://arxiv.org/abs/1910.01526
- Fisher markets / Eisenberg-Gale with proportional-response dynamics (Wu and Zhang 2007, unverified): the budgeted version of BASE, and the natural home for M2.

## 6. Where CONTEXT.md is wrong or overstated

- "BASE layers are a market (assignment) router" (idea 5). BASE solves a
  linear assignment with equal quotas. It has no budgets, no wealth and no
  prices that anyone pays. It is at most an equal-income assignment market. A
  wealth version needs unequal quotas (M2) or budgets (Fisher market).
- "backprop already pays each contributor demand × (outcome − price)"
  (idea 3). This is exact at the logit only. Inside the network the payment
  is first-order attribution. It fails to conserve at every multiplicative
  gate (off by a factor of 2, section 2), at LayerNorm (wealth-net had to
  freeze the denominators) and at attention softmax. In MoE and GLU models
  the gates are everywhere.
- "The missing piece is wealth" (ideas 3 and 5, applied to MoE). Per
  observation A, the router logit already is a Hedge log-wealth. The missing
  piece is counterfactual revenue for experts that did not trade. Adding a
  separate global wealth term mostly adds a second, context-free copy of the
  same signal, with the Rule-30 leak.
- "replace the router or the balancing loss with wealth." The balancing loss
  partly exists for hardware capacity under expert parallelism. Wealth-
  proportional compute (M2) makes that problem worse at scale. Keep a
  congestion price.
- Wealth-gating plus early rank lock-in is a collapse risk. wealth-net's
  wealth ranks correlate 0.97-0.99 across the second half, and the soup's
  Gini is 0.75-0.82. Any wealth-gated scheme needs bankruptcy and birth (M4)
  or exploration (M3), or it will reproduce router collapse on purpose.

## 7. A prediction the soup makes for MoE

The soup found that density per question (K/Q) is the lever and per-trader
compute (fuel) is not. The MoE analogue: active experts per token and
granularity matter more than expert size. That agrees with the fine-grained
scaling law (https://arxiv.org/abs/2402.07871) and with PEER. The study can
test it directly: at fixed active FLOPs, sweep (expert count, top-k, expert
width) on the soup substrate and see whether the Brier-vs-density curve
collapses the way the soup's did.
