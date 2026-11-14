# Scope: the experimentalist's plan (traders-net)

Lens: 16-CPU box, spot L4s, PyTorch, the existing `../wealth-net` code
(2-layer pre-LN transformer, d=64, Q=64 question tokens, window 32, books
every step, snapshot every 100). Written 2026-09-27 by a scoping subagent.
This is a proposal for the larger study, not results.

## 0. What the running wealth-net already tells us (checked 2026-09-27 ~21:55Z)

Check: `results/train-s{1,2,3}/snapshots/*.json` field `ev_bce` (per family,
order periodic, lfsr16, ca30, ca110, prime, pi, and, or) and
`results/summary.json` (checked_at 21:50:46Z, steps ~7500-8900 of 60000).

1. **The net overfits, and the in-sample books pay for it.** Held-out BCE on
   pi-parity rises monotonically in all 3 seeds: 0.70 at step 100, 0.77 at
   1600, ~2.0 at 3100, 3.6-3.7 at ~8800 (ln 2 = 0.69 is "no edge"). CA30 also
   drifts above ln 2 (0.70 -> 0.73-0.75) and prime degrades in s1/s2. Yet the
   output readouts' cumulative wealth split by family is largest for pi
   (s1: +507, vs 193 periodic, 161 LFSR). The data are 24k positions, the
   first 80% trained on for ~400 epochs by 60k steps of batch 128. So
   **wealth as currently booked is in-sample revenue and rewards
   memorisation.** In the soup every bet was settled on an outcome the trader
   had not seen; here most settlements are on replayed examples. This is the
   first thing to fix, before any pruning or gating claim.
   Also possible leakage in the pi family: questions are pi-parity at offsets
   0-2000, so one question's eval window overlaps another's train window
   (streams.py lines ~94-120). The net is confidently wrong, so it is not
   exploiting the overlap, but it confounds any "pi is unlearnable" reading.
2. **The net prices what the soup never did.** LFSR-16 held-out BCE 0.001 and
   CA110 0.005-0.02 (logistic baseline 0.70 and 0.24-0.42); the soup left both
   at Brier 0.25 in every cell. CA30 stays at chance in both. So
   backprop-shared computation (Kelvin's "share core computation") already
   buys the families that evolution-without-sharing missed. Keep this as the
   headline contrast; every trader-style change must not lose it.
3. **Cumulative wealth freezes early.** Rank correlation of wealth vs 1000
   steps earlier is 0.97-0.99 at every granularity, while the rank correlation
   of consecutive income windows for MLP neurons is negative (-0.27). Wealth
   is dominated by the high-loss first few hundred steps. A cumulative sum is
   a poor signal of a unit's current value; the currency definition itself
   is an experimental variable (E2).
4. **Books balance.** Frozen-attention conservation residual 1e-6 relative,
   sum rules 1e-6, live-attention residual at the inner cuts ~0.05% of R.
   Implementation is trustworthy; the open questions are about meaning.
5. **"Parasites" exist in the in-sample books:** 49-72 MLP neurons (10-14%)
   with negative wealth, carrying 11-16% of activations; 13-18 of 64 output
   readouts negative. The ablation test named in CONTEXT.md is not in
   `train.py` yet (grep finds no ablation/prune code); E1 builds it.

## 1. Dependency order

```
E0 prequential books + oracle  --->  E1 prune-by-wealth  ---> E6 bankrupt/reinit (plasticity)
          |                     \->  E2 currency definition --^         |
          |                                   |                         v
          \----------------------------> E3 market layer (closed-form pool) -> E4 K/Q sweep
                                              |                                 \-> E7 price-reading 2nd round (coherence)
                                              \-> E5 wealth-gated MoE, no balancing loss
```
E0 is a hard prerequisite: every later experiment uses held-out (prequential)
books. E1 and E2 share the same oracle. E3 and E5 can run in parallel once E2
fixes what "wealth" means. E6, E4, E7 are conditional on earlier results.

Budget reference: current train runs are ~96 ms/step on an L4 with books
every step (3 runs in parallel), so 60k steps ~1.6 h. Books every 10 steps
cuts that ~5x (estimate, unverified). L4 spot ~$0.25-0.35/h (unverified).
All seeds x3 unless stated; report mean ± sd; pre-register the stop rules
below before running.

## 2. Experiments

### E0. Prequential books and a leave-one-out oracle (prerequisite)
- **Change.** (a) Streams: generate positions on the fly (all families except
  pi are cheap generators; for pi, compute more digits or drop pi and use a
  seeded PRNG-parity family as the "no edge" control, with no shared offsets
  across questions). Single pass, test-then-train: every batch is booked
  *before* the optimiser step sees it, exactly the market ordering. (b) Keep a
  second arm with the current multi-epoch regime (small T) as the overfitting
  stress test. (c) In both, book a fixed held-out batch every 100 steps
  ("held-out wealth"), next to the in-sample books.
- **Oracle.** At checkpoints (steps 500, 2k, 8k, final) compute exact
  leave-one-out held-out ΔBCE for every MLP neuron (512), head (8),
  readout (64) and component (6): ~600 forward passes over 2k eval examples,
  minutes on CPU. Plus the per-unit first-order prediction −revenue, to
  measure how well the books' first-order number predicts the true ablation
  effect.
- **Metrics.** Spearman(in-sample wealth, held-out wealth); Spearman(each
  wealth, LOO ΔBCE); for pi-feeding readouts, sign of held-out wealth.
- **Controls.** Random-init books (already exist), frozen-weight books.
- **Stop / decide.** If in-sample vs held-out rank corr ≥ 0.9 in the
  prequential arm, cheap in-sample books are fine there; in the multi-epoch
  arm expect it to fail and use held-out books everywhere downstream.
- **Cost.** ~0.5 day code, <2 L4-hours.

### E1. Pruning by wealth (the test CONTEXT.md promised)
- **Question.** Is signed wealth a better removal signal than what pruning
  already uses? Removing a negative-wealth unit should, to first order,
  *lower* held-out loss.
- **Arms (remove k units, k = 1, 2, 4, ..., 128 neurons; heads and readouts
  separately).** (1) most negative held-out wealth, (2) most negative
  in-sample wealth, (3) most negative recent income (EMA, half-life from E2),
  (4) lowest accumulated |a·g| (Molchanov Taylor-FO pruning: the same
  quantity without sign; this is the baseline that matters),
  (5) lowest loss-gradient norm, (6) lowest activation frequency,
  (7) random, (8) most *positive* wealth (reverse control, should hurt most),
  (9) LOO oracle greedy (upper bound).
- **Metrics.** Held-out BCE vs k per family; area under the degradation
  curve; fraction of arm-(1) removals that improve loss; recovery after 200
  fine-tune steps.
- **What would excite.** Arm (1) improves held-out loss for small k (true
  parasites) and beats arm (4) at every k; arm (2) fails in the multi-epoch
  regime exactly on pi-feeding units (in-sample wealth mislabels memorisers
  as rich).
- **Stop.** If arm (1) ≤ arm (4) within seed noise at all k in both regimes,
  write it up as "wealth = signed Taylor importance, no pruning gain" and do
  not build more on pruning. E6 then only runs if E2 finds a recent-income
  signal that beats it.
- **Cost.** CPU, from E0 checkpoints: ~0.5 day including code.

### E2. What should "wealth" be? (currency definition, books-only)
- **Candidates.** Cumulative revenue (current); EMA income with half-lives
  100/1k/10k steps; Kelly-multiplicative wealth w ← w·(1 + f·r/|scale|)
  (log-wealth = sum of log-returns); fixed-share (Herbster-Warmuth style
  redistribution of a fraction α of all wealth each window, the principled
  cure for lock-in); first-order revenue vs LOO revenue.
- **Metric.** For each candidate at step t, Spearman with LOO ΔBCE at
  t + 1k and t + 5k (future value, not current). Also stability (rank corr
  across seeds of family-split wealth for the same functional role, e.g.
  the head that implements the periodic lookup).
- **Controls.** Shuffled-label books (revenue computed with random y):
  every candidate should give ~zero signal.
- **Stop.** Pick the best candidate by future-LOO correlation; if none beats
  plain recent |a·g|, the gating experiments (E5, E6) use |a·g| and the
  wealth framing is accounting-only.
- **Cost.** Hours; reuse E0 snapshots (add per-step income logging).

### E3. A market layer at the output with the closed-form confidence pool
- **Design.** Per question, K trader heads (K = 16 to start) read the final
  residual stream and the market's running price ℓ0 (a price-reading
  trader). ℓ0 = the logistic baseline's logit, detached (the "market" the
  traders must beat). Each trader emits log-odds ℓ_i and confidence k_i ≥ 0.
  Price ℓ* = (ℓ0 + Σ k_i ℓ_i)/(1 + Σ k_i).
- **Arms.** (a) current plain readout (logit sum); (b) plain residual:
  ℓ* = ℓ0 + Σ Δ_i (additive, "extremizing"); (c) pool with k_i learned by
  gradient; (d) pool with k_i = g(wealth_i), wealth updated from prequential
  P&L (E2's definition), no gradient into k; (e) Barbu-Lay artificial
  prediction market (budgets bet, wealth updated multiplicatively).
- **Stress tests.** Multi-epoch regime (does the pool keep pi at ≤ ln 2,
  i.e. "traders know they can only improve on the market here-and-there"?);
  inject 50% noise traders at step N (does (d) starve them while (c) does
  not?); an "independent evidence" family where the label is the AND of
  two independent partial cues seen by different traders (the convex pool
  cannot extremize past its most confident trader; arm (b) can).
- **Metrics.** Held-out BCE per family, ECE, max over unlearnable families
  of (BCE − ln 2) (the overconfidence penalty), wealth Gini, fraction of
  traders with k_i ≈ 0.
- **Excite.** (d) matches (a) on learnable families (keeps LFSR and CA110),
  keeps pi and CA30 at ≤ ln 2 + 0.01 in the multi-epoch regime where (a) goes
  to 3.7, and ignores injected noise traders within ~500 steps.
- **Stop.** If (c)/(d) are ≥ (a) + 1% BCE on learnable families in 3/3
  seeds and give no overconfidence protection beyond what early stopping or
  weight decay gives (add those as controls), drop the pool.
- **Cost.** ~1 day code, ~4 L4-hours.

### E4. Does the soup's K/Q law exist in a gradient-trained market? (conditional on E3)
- Sweep traders per question K/Q ∈ {1, 4, 16, 64} at fixed total parameters
  (smaller traders when more of them) and at fixed per-trader size.
- **Excite.** A collapse on K/Q rather than K or Q, as in the soup; or its
  clean absence (then the soup's law was about evolutionary search, not
  markets). **Stop.** 2 points of the sweep with no trend within noise.
- **Cost.** ~6 L4-hours.

### E5. Wealth-gated MoE with no balancing loss
- **Design.** Replace each block's MLP by E experts (E = 16, 64, 256 tiny
  experts; parameters matched to the dense MLP; top-k per question token).
  Tokens are questions, so families give ground truth for specialisation.
  "Compute = wealth": each expert pays a compute price c per token it
  processes and earns its booked revenue (E2 definition, held-out or
  prequential). An expert fires on a token only if its bid (router score)
  clears the batch's clearing price and it can pay from wealth; bankrupt
  experts cannot buy tokens and are re-seeded from a small fixed-share
  subsidy. No auxiliary loss.
- **Arms.** (1) dense MLP; (2) top-k, no balancing (expected collapse);
  (3) Switch/GShard auxiliary loss; (4) DeepSeek loss-free bias balancing
  (bias nudged by load: this is already a tâtonnement price, the honest
  competitor); (5) expert-choice routing; (6) wealth-gated with budgets;
  (7) wealth-gated with budgets but no clearing price (pure rich-get-richer,
  to show what the price does).
- **Metrics.** Held-out BCE per active FLOP; max expert load share; dead
  experts; load entropy; mutual information expert x family; seed stability.
- **Excite.** (6) matches (3)/(4) in loss, has ≤ 5% dead experts and higher
  expert-family MI, with no tuned loss coefficient. **Also exciting as a
  negative:** (7) collapses and (6) does not, showing the balancing term is
  a price, not a loss.
- **Stop.** Collapse (max load share > 50%) in ≥ 2/3 seeds of (6), or
  loss > 2% worse than (4) at every E.
- **Cost.** ~1.5 days code, ~10 L4-hours.

### E6. Bankruptcy and rebirth as plasticity (conditional on E1/E2)
- **Design.** Nonstationary streams: every 10k steps, a random quarter of
  questions switch family. Units whose wealth (E2 definition) stays negative
  over a window are re-initialised with a small subsidy (birth) instead of
  removed.
- **Arms.** No reinit; random reinit at the same rate; continual backprop
  (utility-based reinit, Dohare et al.); wealth-based reinit.
- **Metric.** Held-out BCE in the 2k steps after each switch; long-run
  plasticity (loss on the 5th switch vs the 1st).
- **Excite.** Wealth-based reinit ≥ continual backprop. **Stop.** No gain
  over random reinit in 3 seeds.
- **Cost.** ~6 L4-hours.

### E7. A price-reading second round for coherence (conditional on E3)
- The soup never evolved the coherence arbitrageur (A∨B unpriced, AND
  violations 0.01-0.09). Measure first: coherence violations
  max(0, P(A∧B) − min(P(A),P(B))) and the OR analogue in the current net
  (cheap, on existing weights). Then add a second market round in which
  traders read *all* round-1 prices (attention over price tokens) and emit
  corrections through the same pool.
- **Excite.** Violations fall to ~0 and OR-family BCE drops toward the
  AND-family level. **Stop.** Violations already ~0 in the plain net (then
  shared computation solved it and the round adds nothing).
- **Cost.** Hours to measure; ~1 day for the second round.

## 3. What to measure everywhere
- Held-out BCE per family, with ln 2 and the logistic baseline as reference
  lines; ECE; overconfidence on unlearnable families.
- Books conservation residual (a regression test for every architecture
  change: new modules must book to the same total).
- Wealth Gini, turnover, number of bankrupt units, fraction of activations
  carried by negative-wealth units.
- For every positive claim: 3 seeds, and the shuffled-label control.

## 4. Prior art the larger study must compare against
- Molchanov et al., Taylor-expansion pruning (a·g importance, i.e. unsigned
  per-unit revenue): https://arxiv.org/abs/1611.06440 and
  https://arxiv.org/abs/1906.10771
- Ghorbani & Zou, Neuron Shapley (removing negative-contribution neurons):
  https://arxiv.org/abs/2002.09815
- Integrated gradients / DeepLIFT / LRP (conservation of attribution):
  https://arxiv.org/abs/1703.01365 , https://arxiv.org/abs/1704.02685
- Barbu & Lay, artificial prediction markets for classifier aggregation:
  https://arxiv.org/abs/1102.1465 ; Storkey, machine learning markets
  https://proceedings.mlr.press/v15/storkey11a.html (URL unverified)
- Beygelzimer, Langford & Pennock, Kelly bettors: https://arxiv.org/abs/1201.6655
- Chen & Wortman Vaughan, markets as no-regret learning: https://arxiv.org/abs/1003.0034
- Garrabrant et al., Logical Induction: https://arxiv.org/abs/1609.03543
- Herbster & Warmuth, tracking the best expert (fixed share)
  https://link.springer.com/article/10.1023/A:1007424614876 (URL unverified)
- MoE routing: Shazeer https://arxiv.org/abs/1701.06538 ; Switch
  https://arxiv.org/abs/2101.03961 ; BASE layers https://arxiv.org/abs/2103.16716 ;
  expert choice https://arxiv.org/abs/2202.09368 ; DeepSeek loss-free
  balancing https://arxiv.org/abs/2408.15664 ; hash layers
  https://arxiv.org/abs/2106.04426 ; PEER, a million tiny experts sharing
  computation via product keys https://arxiv.org/abs/2407.04202
- Dohare et al., loss of plasticity / continual backprop:
  https://arxiv.org/abs/2306.13812
- Logit-pool aggregation and extremizing: Satopää et al. 2014
  https://doi.org/10.1016/j.ijforecast.2013.09.009 (unverified)
- Baum's Hayek machine / Holland's bucket brigade (economies of agents with
  wealth and credit passed backward): recollection, (unverified);
  Baum & Durdanovic 2000 https://doi.org/10.1162/106365600568130 (unverified)
- DEQ (implicit fixed-point layers): https://arxiv.org/abs/1909.01377

## 5. Ideas in CONTEXT.md I think are wrong or overstated
1. "Wealth = marginal information (Shapley-like)". a·dR/da is first-order
   Taylor attribution, which equals Shapley only for additive units. It
   ignores saturation and interactions; E0 measures the gap against LOO.
   And it is Molchanov's pruning score with the sign kept: the novelty is
   the sign, the family split and the conservation, not the quantity.
2. Books kept on training batches are not a market. In the market, bets
   settle on unseen outcomes; with replayed data the books pay memorisation
   (pi: largest in-sample wealth, held-out BCE 3.7). All wealth claims need
   prequential or held-out books.
3. Cumulative wealth as the currency locks in early winners (rank corr 0.98
   while income decorrelates). Kelly-multiplicative wealth has the same
   property. Something like fixed share (a redistribution) or discounting is
   needed; this is a design choice, not a detail.
4. The closed-form pool ℓ* = (ℓ0 + Σ k_i ℓ_i)/(1 + Σ k_i) is a convex
   combination of log-odds: it can never be more extreme than its most
   extreme trader, so it cannot add independent evidence (Bayes adds
   log-likelihood ratios). "Not smarter than the market" is bought at the
   cost of under-confidence when edges are independent; E3's
   independent-evidence family tests this, and the additive arm (b) is the
   control.
5. "Wealth-gated MoE without a balancing loss" does not get balance for
   free. Wealth compounding is rich-get-richer, the same dynamic as router
   collapse. Balance must come from a price on compute (a clearing price or
   capacity auction). DeepSeek's loss-free bias and BASE's assignment are
   already such prices. The honest claim to test is "a price derived from
   books can replace a tuned loss coefficient", not "no balancing term".
6. The soup's K/Q law may be a property of evolutionary search (lottery
   tickets per question), not of markets; gradient-trained traders share
   gradients and may not show it. That is E4's question.
