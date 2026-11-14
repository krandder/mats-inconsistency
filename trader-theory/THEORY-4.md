# Round 4: stop

trader-theory, 30 Sep 2026. Sources: `market-resnet/RESULTS.md` §17, `puzzles/P18-detached-mazes/RESULTS.md`, `P17-breadth-depth/RESULTS.md`, `P19-fast-reader/RESULTS.md`, `followups-batch2/item1-costnet/RESULTS.md`, the catalogue. No new references; every paper named below was verified in rounds 1 to 3.

## What is now dead

Nineteen pre-registered experiments, zero gains. Each line below names the number that killed it.

1. **The market over depth.** The trader payoff's gradient is truncated BPTT(1) with per-step, covariance and VOC losses (cosine 1.0000000). Random training order plus per-trader optimizers carry the 100× result; the exact twin without a ledger overlaps the economy (529/600 against 557/600). The published DT-Recall block beats both (589 to 600/600). Kelvin's growing-liquidity arms bound the logits as designed but cost accuracy (0.34 and 0.71 at 100×); the ±10 logit cap that works (600/600) works the same without a market (598/600). The economics attribution is retired by the runners' own rule.
2. **Random-order detached deep supervision as a general lever** (round 3's survivor). P18 killed it on mazes: 1.7% against 8.1% for DT-Recall progressive at 59×59, one-sided bound [-11.1, -1.7], with 69% of large mazes locked in period-4 cycles. It was a prefix-parity effect.
3. **Learned local costs plus an exact solver.** Replicates (25.05% [23.6, 26.5], 20 seeds) but the released DT-Recall checkpoint solves 100% of the same 59×59 mazes in-house; Bansal et al. report 97.3%. Worse by 75 points, and the method is Vlastelica et al. 2020 unchanged.
4. **The breadth-by-depth market** (P17). Beats EqR's own selection rules by 6 points, ties plain restart-until-valid (98.54 against 98.49, one puzzle), a single trader equals the pool, and the market learned a fixed rule ("restart, don't deepen"). Checker gain plus budget pooling; no pricing gain.
5. **Cheap pricing of candidate variables** (P19). The fast reader ranks intermediates at tau 0.576 against a 0.70 bar; the first-order price is within 0.05 of it; a fully trained reader reaches 0.674 and the wrapper 0.866. The meta-trained reader (P16) never met its training precondition, so it is inconclusive, not alive.
6. **Everything from rounds 1 to 3:** pay rules as training signals (Wolpert and Tumer's aligned payoff is leave-one-out, which is backprop), latent pricing and discovery, joint markets and installed arbitrageurs, value-of-computation gains, the wealth cap, and hindsight settlement as a new learner (it is EM; only its exact-inference forms train).

**A process fault that ran through the kills:** every in-house baseline that mattered was under-trained against the released model (P18's DT-Recall at 8% where the released checkpoint scores 100%; CostNet at width 32; P16 below its precondition; P5). Nothing in the programme was ever compared against the strongest baseline at the strongest baseline's budget and won.

## What survives

- **Three diagnostics**, none new: unit-level Shapley pay ranks units at 0.97 (Neuron Shapley); the arbitrageur's price detects wrong mazes at AUROC about 0.86 within a size; the hindsight KL flags mis-encoded intermediates at 0.86 to 0.90. Worth keeping as tools.
- **Four equivalence results**, which are the programme's real theoretical output: the gradient identity (market pay over depth = detached deep supervision); aligned pay rule = difference utility = backprop; hindsight settlement = generalised EM; the LMSR cost function = the Ortega–Braun free energy. They say why every mechanism reduced to something known before the data did.
- **The catalogue itself.** Nineteen pre-registered, independently verified null results with kill rules and a spend under $150 is a rare artefact. Written up as "internal prediction markets do not train neural networks", with the equivalences as the explanation, it is the one output with value outside the fleet.

## Odds

A theoretical advance from the economy framing: about 2% (was 5%). Any market-derived tool beating the strongest known baseline on any task in scope: about 10%. The marginal value of another mechanism experiment is now near zero: every candidate mechanism has either a proved equivalence to a known method or a pre-registered kill.

## Recommendation: stop funding mechanism experiments

Kelvin's standing offer to spawn as many experiments as we want does not change this. Spending is justified by expected information, and the last seven experiments moved the odds by a point or two each while confirming what the equivalences already said. Two bounded items, and nothing else:

1. **Write-up, no GPU.** The negative-results note above, from the catalogue and the verifier reports, plus a hub wiki page carrying the four equivalences and the kill table so no future agent re-derives them. Kill rule: none needed; it is documentation.
2. **P16 to its pre-registered training budget, only if Kelvin wants a *valuation tool*** (a reader that values a newly listed bit), not a learning mechanism. Its precondition failed from under-training, so its null is not clean. Kill rule as pre-registered. If he does not want the tool, skip it.

**What would reopen the programme:** a written proof, before any run, of a property a proposed mechanism has that gradient descent on the same architecture lacks, with no equivalence to a known method. Until someone writes that proof, no GPU for markets.
