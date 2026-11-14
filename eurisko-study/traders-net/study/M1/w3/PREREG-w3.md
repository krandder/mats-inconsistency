# PREREG M1 worker 3, item (c): books_v2 (written 2026-09-27 23:22Z, before any run)

**Code.** `M1/books_v2.py` (three books: prequential `tangent`, exact `lmsr` from the running base-rate
prior l0 with a 16-step midpoint path, `meanbase` (a − ā)·g; extras `meanbase_lmsr`, `absag`).
Unit tests `w3/test_books_v2.py` (frozen-attention conservation < 1e-8 relative, readout LMSR book =
exact log-score gain, masks, purity). Model: S0 `model.Net` Q=64, W=32, d=64, H=4, L=2 (wealth-net
size), Adam lr 1e-3. Stream: S0 `streams_ext.make_streams(heldout=0, regime_every=0)` (prng replaces pi),
`to_array_stream`. Family groups for the books: {prng} and {rest} (cost: a full 9-family split is
9x slower on this box; per-family numbers are not claimed).

**Arms (per seed 1, 2, 3).**
- `P` prequential single pass, T=32032 (n_pos 32000), B=16, 2000 steps. For every block: books with
  pre-update weights on the block (`pre` = prequential), one Adam step, books again on the same block
  with post-update weights (`post` = in-sample by one step), then prior and means observe the block.
- `ME` multi-epoch stress: T=16032, train positions [0, 12800), 40 epochs of B=128 (4000 steps);
  every 2nd step books on 32 examples of the training batch (`in` = in-sample, pre-update weights, as
  wealth-net v1) and on 32 random held-out positions [12800, 16000) with the same weights (`ho` =
  never trained, the prequential analogue). Prior = train-split base rate (fixed); means from `in`.
- Controls: `frozen` = final P weights, no updates, books on every 4th P block; `random` = fresh init,
  no updates, same blocks.

**Metrics.** Per book (tangent, lmsr, meanbase), per granularity (out, head, neuron):
1. Spearman of cumulative wealth: `ME in` vs `ME ho` (primary), and `P pre` vs `P post`.
2. PRNG wealth sign: total prng-group income per book and granularity, and the fraction of neurons
   whose prng income is > 0; in `ME in`, `ME ho`, `P pre`.
3. Cross-book Spearman (tangent vs lmsr vs meanbase) and vs `absag`, on `P pre`.
4. Diagnostics: mean relative conservation residual (live attention), 16-step path vs closed form,
   gain − readout LMSR book.
3 seeds, mean ± sd (sample sd).

**Decision rule (the M1 stop rule, applied per book).** If Spearman(ME in, ME ho) has seed mean ≥ 0.9
at every granularity, in-sample (cheap) books are acceptable for that book; otherwise later arms use
prequential books. Expected: < 0.9 for neurons (memorisation pays), prng income > 0 in `ME in` and < 0
in `ME ho` / `P pre`; lmsr readout income ≈ information gain > 0 on learnable families.

**Stop rule for this item.** One pass of each arm per seed; no retuning after seeing results. If a seed
exceeds 90 min wall time, cut ME to the epochs done and report that.
