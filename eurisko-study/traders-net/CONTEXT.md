# Context: traders inside neural networks (Kelvin + Jessica Taylor, 2026-09-27)

Written by eurisko (scientific lead) from the conversation with Kelvin. Kelvin
and Jessica Taylor (co-author of the logical induction paper) are thinking about
representing "thousands of traders" inside one neural network.

## Where this comes from

- The logical-induction soup (see ../logical-soup/REPORT-SOUP.md): evolving
  fuel-bounded trader programs priced by an LMSR market over synthetic binary
  questions, one Bend2 program, wealth as the only currency. Result: Brier
  collapses on traders per question; cheap patterns get priced out at high
  density; coherence arbitrage never evolved; LFSR/CA/pi never priced.
- Kelvin's wishes, verbatim where possible: "thousands of traders be represented
  inside a similar neural network. Maybe something inspired by mixture of experts
  models, but more aggressive. I like the compute = wealth framing, and I wish
  the traders could share core computation somehow." "traders shouldn't think
  they are smarter than the market; the traders should know that they can only
  improve on the market here-and-there." "a fractal kind of thing; where you
  have a big-trader being composed of many small traders, recursively so, but
  not necessarily hierarchical. If hierarchical, the big trader wealth is the
  sum of its constituent traders; if not hierarchical, then the small trader
  can be partially owned by different big traders." "the smallest modification
  or interpretation of existing MoE or neural networks that would help us reason
  in terms of wealth of different traders at different scales." "we don't want
  any Harberger tax now, what we may want is to get a small network and see how
  the wealth works or evolves over time." Interested in MoE routing failures and
  the unprincipled load-balancing losses. Wants suggestions, experiments and
  insights he might like.

## Ideas on the table (eurisko's, to be checked, extended or refuted)

1. Kelly-betting traders with wealth updated by P&L = Bayesian model averaging
   (Beygelzimer, Langford, Pennock 2012; Cover); LMSR = no-regret learning (Chen
   and Vaughan 2010). Price-reading traders (edge relative to the market) break
   this, and that is where logical-induction behaviour lives.
2. Traders should emit a correction to the market's log-odds sized by
   calibrated confidence (q_i = update(price, LR_i), demand ∝ q_i − price):
   self-limiting, wealth = marginal information (Shapley-like), boosting reading.
3. At the softmax layer: logit = Σ contributions; softmax = LMSR over exclusive
   outcomes; cross-entropy gradient p − y = market settlement, so backprop already
   pays each contributor demand × (outcome − price). The residual stream is a
   sequential market (later layers read the running price). Missing: wealth
   (multiplicative budgets), bankruptcy/birth, price-visible demands within a
   layer, bounded bets (logit clipping as the budgeter).
4. Unit = anything with an additively separable logit contribution: output
   weights (per question), MLP neurons or SAE features (all questions at once),
   heads, layers. Deeper weights are "the trader's brain".
5. Confidence-weighted pooling with a fixed point: with linear edge traders,
   ℓ* = (ℓ0 + Σ k_i ℓ_i)/(1 + Σ k_i), closed form; otherwise Newton + implicit
   differentiation (DEQ) or Eisenberg-Gale convex program for Kelly traders.
   Differences from a plain logit sum: convex normalisation, weights from
   realised P&L not gradient, explicit prior. Nearest failures: capsule routing
   by agreement, MoE router collapse with load-balancing losses; BASE layers are
   a market (assignment) router.
6. Fractal books: backprop as the accounting of a fractal firm. Money flows
   backward like gradients: a consumer pays each supplier its first-order
   attribution share of its own revenue; wealth = cumulative revenue (− optional
   compute cost). Partial ownership automatic; a big trader is any set of units
   with wealth = sum (internal transfers cancel). Smallest modification: keep
   the books on an existing model; then gate execution by wealth; (Harberger
   self-assessment deferred at Kelvin's request).
7. Running now: wealth-net, a small transformer on the soup's streams with the
   books kept every 100 steps over training, plus an ablation test (remove the
   most negative-wealth units vs random/low-gradient; held-out loss). Partial
   results: ../wealth-net/RESULTS.md and http://[ip-removed]:8765/wealth-net/.

## What Kelvin asked for now

First a scoping workflow: figure out what to study and which experiments to
run. Then a separate, larger workflow executes that plan.

## Addendum (eurisko, 2026-09-27 23:40Z): candidate settlement rule for attention and gates ("three rules")

Wealth-net found the books balance above attention (residual 4e-6 with frozen attention) but
over-credit units below live attention by 67-110%: gradient × input double-pays the value path
and the routing path of a bilinear-plus-softmax node. Candidate exact rule, to be tested by M1
against integrated gradients and conductance, with exact conservation as the first check:
1. Linear (degree-1 homogeneous) nodes: Euler split x_i·∂R/∂x_i (exact).
2. Bilinear products (scores q·k, output Σ a_j v_j): Shapley split of a product with zero
   baseline, half to each factor (exact, symmetric).
3. Softmax = LMSR sub-market. With u_j the revenue attributable to value j given the weights
   (linear, exact) and prior π_j (uniform 1/n, or a learned/previous-step prior):
   R = Σ_j π_j·u_j + Σ_j (a_j − π_j)·u_j exactly. Values are paid at the prior price; key j
   earns (a_j − π_j)·u_j, i.e. demand × (outcome − price) settlement for the routing trader.
   The same rule prices an MoE gate (experts = values, router logits = bids), giving a wealth
   router with no load-balancing loss.
LayerNorm is degree-0 homogeneous (Euler gives 0): treat σ as a fixed exchange rate per step
(detach it), as wealth-net does. Kelvin's question for M1/T1: is this the most theoretically
elegant conservative money flow, and does it change the pruning and lock-in results?

## Retarget (Kelvin, 2026-09-28 00:25Z, confirmed)

Wealth-net result: backprop credit (gradient × activation, accumulated or decayed) and market
settlement disagree in sign on exactly the units that matter: a trader whose signal is already
in the price earned for putting it there; a unit whose signal is already absorbed gets zero or
negative gradient because the model over-relies on it. Kelvin: "Yes, this is the primary
problem." Consequence for this study: units in a dense network cannot be traders by bookkeeping
alone; a trader must be a designed object that commits to a prediction (or a log-odds
correction with confidence) on a defined question, at a price, with a bounded stake, and is
settled against that question's outcome relative to the price at the time. Attention heads and
MoE gates qualify (internal outcome: which position/expert was useful); generic neurons do not.

## Addendum (2026-09-28 01:00Z): exact settlement works; block-local architecture; price of anarchy

- Wealth-net: exact leave-one-out log-score settlement at the output (s_i = log p(y) − log p_{-i}(y))
  tracks the single-neuron ablation oracle at Spearman +0.58 (gradient wealth +0.04, gradient norm
  +0.17), 73% of the most important neurons in its top 10%, best pruning criterion; +0.87 for the last
  layer, +0.3 for the first (direct path only). Sampled Shapley ~0. Files: ../wealth-net/RESULTS.md
  "Exact settlement". Residual writes are LMSR trades (b=1): log p_{≤ℓ}(y) − log p_{<ℓ}(y) telescopes
  to the total log score; within a layer, integrated gradients along the softmax path is the
  conservative split.
- Kelvin (confirmed): drop the "consumers pay suppliers" money flow; a trader must in expectation earn
  more than it pays. Study Baum & Durdanovic's Hayek machine. Focus on the local-settlement learning
  algorithms most competitive with backprop at scale (block-local training: Belilovsky 2019, Nøkland &
  Eidnes 2019, Siddiqui 2023; then predictive coding, forward-forward, equilibrium propagation), and the
  continuum: a trader is a block with backprop inside and market settlement at its boundary.
- Theory item for T1 (Lemma 9): "price of anarchy" of internal competition: the loss gap between
  block-local settlement training and end-to-end backprop, and its dependence on block size; and the
  fixed-point statement: market dynamics and gradient descent share fixed points in price, not in the
  wealth distribution.
- M6 becomes the architecture test: end-to-end vs block-local vs block-local + wealth-scaled units with
  leave-one-out settlement (first run started on wealth-net, 01:00Z); measure loss gap, oracle agreement,
  Gini, over-reliance.
