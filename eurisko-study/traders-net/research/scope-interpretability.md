# Scope: wealth accounting through a mechanistic-interpretability lens

Scoping notes for the traders-net study (2026-09-27). The lens is mechanistic
interpretability. I only propose what to study; I answer nothing here. Inputs:
`../CONTEXT.md`, `../../logical-soup/REPORT-SOUP.md` sections 1-3, and
`../../wealth-net/{README.md,books.py,train.py,results/summary.json}`
(`wealth-net/RESULTS.md` does not exist yet; I read `summary.json`, checked at
2026-09-27T21:50Z, 3 seeds at step ~7500). Recollections not re-checked in this
session are marked (unverified). The URLs are the canonical ones as I remember
them; I did not fetch them here.

## 0. What wealth-net actually computes, in interp vocabulary

- Per example, unit revenue = `a * dR/da`, with `R = sum_q z_q * (y_q - p_q)`
  and `s = y - p` detached. Because `dBCE/dz = p - y` exactly, this equals
  `-a * dL/da`. It is **gradient x input** (first-order Taylor attribution
  against a zero baseline). At the readout it is **direct logit attribution
  (DLA) weighted by the settlement**. Internally it is **attribution patching
  (AtP) with zero ablation**, summed over examples and training steps.
- The books freeze LayerNorm denominators. This is the convention used by DLA
  practice, by conservative LRP for transformers, and by attribution-graph
  "replacement models". The frozen-attention variant is the AttnLRP/Ali et al.
  "detach softmax" rule (see prior art). It is a known propagation rule, not
  just an implementation check.
- Wealth = the cumulative sum over training of per-step, per-batch income,
  computed on the pre-update weights (Adam, lr 1e-3, no weight decay).

## 1. Is "backprop as attribution flow of settlement" sound accounting?

Short version: it is exactly the **Euler / product-exhaustion** accounting from
economics. Factors paid their marginal product exhaust output only under
constant returns to scale, which here means degree-1 homogeneity. It is sound
wherever the graph is degree-1 homogeneous and leaks everywhere else. The
leaks are specific and measurable:

1. **Biases and constants.** ReLU + affine is not homogeneous. The books book
   biases as suppliers, which is the right fix. Question to study: what share
   of R do constants take over training? A large, growing share means the
   market mostly pays "rent" to fixed factors.
2. **Attention softmax (degree 0 in a score shift) and bilinear QK (degree 2).**
   Q and K suppliers of a bilinear score are paid ~2x the score's revenue by
   Euler, and the softmax pays nothing for a common shift. The summary's
   cumulative conservation residual, `(R - cut - biases below)/|R|`, is
   **-0.64 / -0.45 / -0.45** at the three earliest cuts in seed 1 (s2 similar,
   s3 -0.48/-0.31). It is ~0 only at the two last cuts. So the "fractal firm"
   books currently fail to balance by about half at the embedding cut. This is
   the first thing the study must explain: decompose the residual by source
   (QK bilinearity, softmax, LN convention).
3. **The LN convention changes whose money it is.** The books' derivatives are
   not the true loss derivatives. Any claim "negative wealth ⇒ removing it
   lowers loss" must be checked against true ablation, not against books.
4. **Saturation and redundancy (OR-gates).** When `p≈y`, `s≈0` and every
   contributor earns ~0, even if removing all of them would be catastrophic.
   Two redundant units each earn their own marginal, yet ablating one does
   nothing (backup name movers; the Hydra effect). This is the classic AtP
   false-negative case (Kramár et al. 2024).
5. **Interactions (AND-gates, K-composition).** For `f = ab`, both factors are
   paid `ab` (sum `2ab`). First-order accounting double-pays conjunctions and
   under-pays disjunctions. The soup's coherence failure (A∧B priced, A∨B not)
   has a structural analogue here worth testing: do the A∨B-paying units look
   under-paid relative to their ablation value?
6. **Baseline = zero.** Zero-ablation is off-distribution for residual
   components and heads. The market-flavoured baseline is the **mean** (or
   "optimal ablation" constant). Then revenue = `(a - ā)·g`, whose expectation
   is `Cov(a, g)`: payment only for information beyond the average. That is
   Kelvin's "traders only improve on the market here and there", and it looks
   like the selection term of the Price equation (connection unverified). The
   zero baseline instead pays a base-rate rent `ā·ḡ`.
7. **Time aggregation.** A unit's identity is stable (same weights) but its
   function drifts. Cumulative wealth mixes payments for different jobs.

### A theoretical warning that should be checked first (hours)

For any unit whose output has a free trainable scale (an MLP neuron's `fc2`
column, a head's `W_O` slice, a readout row), the batch revenue is an exact
identity:

`sum_b a_i * dL/da_i = <w_out_i, dL/dw_out_i>`

(exact with the true gradient; approximate under the LN-frozen books). So
income is minus the directional derivative of the loss along "scale this unit
up". At a stationary point of training its expectation is **zero** for every
such unit (weight decay λ would pin it at `λ||w_out||²`). This is the
zero-economic-profit condition of a competitive equilibrium. Under plain SGD,
cumulative wealth ≈ `(||w_out||²_T - ||w_out||²_0)/(2η)` minus a noise term.
Adam distorts this, but the directional-derivative reading still holds.

Consequences to test:
- Wealth may be mostly a record of **norm growth in the early transient**,
  i.e. close to magnitude pruning, a strong but known baseline. Test the
  Spearman correlation of wealth with Δ||w_out||², and repeat under SGD.
- The summary already hints at this. Rank correlation of *income* between
  windows in the second half is 0.09 (heads) and -0.27 (neurons). The 0.98
  *wealth* rank correlation is an artifact of cumulative sums, since adjacent
  cumulative snapshots share most of their terms. The study should report
  stability only between disjoint windows.
- The only non-trivial steady-state signal is then on held-out data or under
  distribution shift. That is where "who adds value" differs from "who is at
  equilibrium". The soup streams are stationary, so the study needs a
  shift/new-family protocol (see E5).

## 2. Where first-order attribution breaks, and the principled fixes

| fix | what it buys | prior art |
|---|---|---|
| Path-integrated books (**conductance**) | exact conservation at every cut, including through softmax and LN, because the fundamental theorem of calculus holds along one common input path | Dhamdhere et al. 2018; IG |
| **Aumann-Shapley** cost sharing | the unique path rule that satisfies the cost-sharing axioms; this is the economically principled "fractal firm" | Billera & Heath 1982; Friedman & Moulin 1999 |
| Conservative LRP rules (AttnLRP) | cheap (1 pass), conservation by design, explicit rules for softmax, bilinear (uniform 50/50 split) and LN | Ali et al. 2022; Achtibat et al. 2024 |
| DeepLIFT multipliers | reference-based, obey a chain rule, and sum to delta | Shrikumar et al. 2017 |
| Sampled Shapley over units | handles interactions and redundancy; expensive | Neuron Shapley (Ghorbani & Zou 2020) |
| AtP* / EAP-IG | fixes attention-saturation false negatives; edge-level payments | Kramár et al. 2024; Hanna et al. 2024 |

The key design question for the study is which of these keeps Kelvin's
properties together: additivity across scales (big trader = sum),
partial ownership, and single-backward-pass cost. Conductance and AttnLRP
keep conservation; Shapley keeps interactions but not cheapness; plain
grad×input keeps cheapness only.

## 3. What would make "wealth" a better unit score, and how we would know

What could make it better than existing scores, as testable claims:
- (a) **signed and additive across granularities**: neuron → head/layer →
  arbitrary coalitions, with transfers cancelling. |a·g| (Taylor pruning) is
  not additive, and Shapley is additive but not cheap.
- (b) **settled on loss, not logit**: it should pay calibrating/hedging units.
  DLA calls copy-suppression and negative name-mover heads "negative", yet they
  improve loss. Wealth should rank them positive.
- (c) **integrated over data and time**: less noise than a single-batch AtP, if
  the stationarity warning above does not make it a norm tracker.
- (d) **per-family split**: specialization comes for free.

How we would know (pre-register these before looking):
1. **Counterfactual validity**: Spearman(wealth, true held-out Δloss on
   ablation) at each granularity, vs grad×input snapshot, AtP*, IG/conductance,
   |a·g| accumulated (Molchanov), weight norm, and sampled Shapley. Use both
   zero and mean ablation.
2. **Additivity**: for random and structured coalitions, compare the sum of
   member wealth with the coalition's joint ablation Δloss. The gap is the
   interaction mass that first-order misses.
3. **Pruning frontier**: loss vs fraction removed, by ranking method.
4. **Forecasting**: does wealth at 10% of training predict final ablation value
   better than the magnitude or gradient-norm score at the same time?
5. **Ground-truth recovery** on models with known circuits (section 4).
6. **The decisive null**: if wealth is no better than `||w_out||` or than one
   AtP pass on a held-out batch, the accounting is still useful as a
   *language*, but not as a score. Say so.

## 4. Known circuits that could validate a wealth ranking

- **Inside wealth-net itself (cheapest):** tokens are *questions*, so attention
  is cross-question reading. The A∧B / A∨B compound families require heads
  whose attention goes from the compound token to its component tokens. The
  prediction to test: compound-family-funded head wealth should concentrate on
  heads with that pattern. Planted families with known algorithms (lag-XOR,
  period-p copy) give neuron-level ground truth.
- **Tracr-compiled / InterpBench models**: circuits known exactly, so we can
  measure precision/recall of top-wealth units.
- **Induction heads** (2-layer attn-only; Olsson et al. 2022). A test of the
  degree-2 K-composition overpay, and a phase change in *time* that wealth
  must show: the induction bump.
- **IOI in GPT-2 small** (Wang et al. 2022). Name movers (positive), backup
  name movers (redundancy / saturation test), negative name movers and
  **copy suppression L10H7** (McDougall et al. 2023): a sign test for
  loss-settled wealth vs DLA.
- **Modular-addition grokking** (Nanda et al. 2023). The known phases are
  memorization, circuit formation and cleanup. Wealth over time should move
  from memorizing neurons to key-frequency neurons. This is the best test of
  wealth *dynamics*, and it has weight decay, which makes the λ||w||²
  equilibrium prediction checkable.
- **Toy models of superposition** (Elhage et al. 2022): known features spread
  over fewer neurons. This tests whether features, not neurons, are the right
  traders.

## 5. Superposition: who is the trader?

A neuron is a firm jointly owned by the features it hosts. At a residual cut,
an SAE/transcoder decomposition `x = Σ f_i d_i + err` is linear, so revenue
splits exactly across features plus an "error node" trader (as in attribution
graphs). That is Kelvin's partial ownership made literal. Questions: is
feature-level wealth more heavy-tailed and more stable than neuron-level
wealth? Do "parasite neurons" (9.6-11.6% of neurons here; `parasite_neuron_frac`)
host mixed positive and negative features, i.e. cross-subsidy? Tracking over
training needs features with a stable identity across checkpoints
(crosscoders).

## 6. Prior art to check (URLs as recalled; not fetched this session)

- Elhage et al. 2021, A Mathematical Framework for Transformer Circuits (DLA): https://transformer-circuits.pub/2021/framework/index.html
- Nanda, Attribution patching: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- Kramár et al. 2024, AtP*: https://arxiv.org/abs/2403.00745
- Syed et al. 2023, Edge attribution patching: https://arxiv.org/abs/2310.10348
- Hanna et al. 2024, EAP-IG / faithfulness: https://arxiv.org/abs/2403.17806
- Conmy et al. 2023, ACDC: https://arxiv.org/abs/2304.14997
- Ameisen et al. 2025, Circuit tracing / attribution graphs (frozen attention+LN, error nodes): https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Marks et al. 2024, Sparse feature circuits: https://arxiv.org/abs/2403.19647
- Sundararajan et al. 2017, Integrated Gradients: https://arxiv.org/abs/1703.01365
- Dhamdhere et al. 2018, Conductance: https://arxiv.org/abs/1805.12233
- Shrikumar et al. 2017, DeepLIFT: https://arxiv.org/abs/1704.02685
- Bach et al. 2015, LRP: https://doi.org/10.1371/journal.pone.0130140
- Ali et al. 2022, conservative propagation in Transformers: https://arxiv.org/abs/2202.07304
- Achtibat et al. 2024, AttnLRP: https://arxiv.org/abs/2402.05602
- Sundararajan & Najmi 2020, The many Shapley values: https://arxiv.org/abs/1908.08474
- Ghorbani & Zou 2020, Neuron Shapley: https://arxiv.org/abs/2002.09815
- Sundararajan et al. 2020, Shapley-Taylor interaction index: https://arxiv.org/abs/1902.05622
- Billera & Heath 1982, Aumann-Shapley cost allocation axioms (Math. of OR) (unverified URL; search title)
- Friedman & Moulin 1999, Three methods to share joint costs or surplus (JET) (unverified URL)
- Lan et al. 2019, Loss Change Allocation (about half of parameters "hurt" per step): https://arxiv.org/abs/1909.01440
- Molchanov et al. 2019, Taylor importance for pruning (accumulated first-order): https://arxiv.org/abs/1906.10771
- Lee et al. 2019, SNIP: https://arxiv.org/abs/1810.02340
- Pruthi et al. 2020, TracIn (sums over checkpoints): https://arxiv.org/abs/2002.08484
- McGrath et al. 2023, Hydra effect / self-repair: https://arxiv.org/abs/2307.15771
- Li & Janson 2024, Optimal ablation: https://arxiv.org/abs/2409.09951
- Zhang & Nanda 2023, Best practices of activation patching: https://arxiv.org/abs/2309.16042
- Olsson et al. 2022, Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Wang et al. 2022, IOI: https://arxiv.org/abs/2211.00593
- McDougall et al. 2023, Copy suppression: https://arxiv.org/abs/2310.04625
- Nanda et al. 2023, Grokking progress measures: https://arxiv.org/abs/2301.05217
- Elhage et al. 2022, Toy models of superposition: https://transformer-circuits.pub/2022/toy_model/index.html
- Lindner et al. 2023, Tracr: https://arxiv.org/abs/2301.05062
- Gupta et al. 2024, InterpBench: https://arxiv.org/abs/2407.14494
- Lindsey et al. 2024, Sparse crosscoders: https://transformer-circuits.pub/2024/crosscoders/index.html
- Wang et al. 2024, Aux-loss-free MoE balancing (per-expert bias as a price): https://arxiv.org/abs/2408.15664

## 7. Experiments (ordered by cost)

- **E1 norm identity (hours).** On existing wealth-net snapshots, correlate
  per-neuron/head wealth with Δ||w_out||² and weight norm; rerun one seed with
  SGD and one with weight decay. Exciting outcome: wealth carries signal beyond
  norm (partial correlation with ablation Δloss after controlling for norm).
- **E2 residual decomposition (hours).** Split the -0.45 to -0.64 cut
  residual into QK-bilinear, softmax and LN parts. Add a conductance
  (20-step path) books variant and an AttnLRP-rule variant. Exciting: one
  cheap rule closes the books to <5% and ranks units as well as conductance
  does.
- **E3 validity bake-off (1-2 days).** Wealth vs AtP*, conductance, |a·g|,
  norm and sampled Shapley against true held-out ablation (zero and mean) and
  coalition ablation, per granularity. Exciting: wealth wins on coalitions
  (additivity), not just singletons.
- **E4 mean-baseline ("market") books (hours).** Revenue `(a-ā)·g`. Exciting:
  a more stable income ranking between disjoint windows and a better ablation
  match than zero-baseline books.
- **E5 shift test (1 day).** Introduce a new family mid-training. Exciting:
  income reallocates to the units that later prove necessary for it, earlier
  than gradient-norm or AtP flags them.
- **E6 known circuits (days).** Compound-family heads inside wealth-net;
  tracr/InterpBench; IOI copy-suppression sign test in GPT-2 small;
  grokking wealth trajectories. Exciting: loss-settled wealth ranks
  copy-suppression positive while DLA ranks it negative, and grokking wealth
  shows the cleanup transfer.
- **E7 feature-level books (days).** SAE/transcoder features at one cut, with
  an error-node trader. Exciting: parasite neurons turn out to be mixed
  firms with clearly positive and negative feature owners.
