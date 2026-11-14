# M6 w5 PREREG: committed traders (RETARGET item), base regime only

Written 2026-09-28 before the first run. Worker 5 of 5 (claim in `../STATUS.md`).

**Question.** Units become traders only by emitting a committed correction. If the 16 output
trader heads of M6's market layer *commit* (each posts its own price p_i = sigma(l0 + c_i) on a
question and is trained and settled on that price), does wealth start to mean importance, where
for uncommitted heads (a plain additive sum) gradient credit does not (CURRENCY.lock, M1)?

**Substrate.** `S0/substrate`: `streams_ext.make_streams(Q=64, T=20000, seed, heldout=8,
regime_every=0)` (static stream; 72 questions; held-out periodic family released at T/2),
single-pass prequential (test at trade time, then one Adam step on the same block, B=4),
trunk = S0 `model.Net` body (d=48, H=4, L=2), final LN -> residual y[q].
Market price l0 = per-question online logistic (`model.Logistic`, Adam 1e-2), detached.
Heads: 16 per question, c_{i,q} = r_{i,q} . y_q + b_{i,q} (per-(head, question) weights;
a (head, question) cell is one trader, 16 x 72 = 1152 traders per run). Adam 3e-3.

**Arms** (3 seeds each: 1, 2, 3; all arms share the stream and trunk init per seed):
- `U` uncommitted additive (= M6 arm (b)): l = l0 + sum_i c_i, trained end to end on market BCE.
- `Cm` committed, equal stakes: each head trained on its own BCE(l0 + c_i, y) (block-local at
  the head boundary; the trunk gets the mean of the heads' gradients); market l = l0 + mean_i c_i.
- `Cw` committed, wealth stakes: as Cm, but market l = l0 + sum_i w_{i,q} c_{i,q},
  w_{.,q} = softmax_i(K_{i,q}) with K = cumulative committed settlement (below) booked before
  the step (Bayesian stake update, eta = 1).
- `L0` reference: l0 alone.

**Wealth measures per (head, question)**, all prequential at trade time, accumulated over the
first 75% of positions (the "book window"):
- G gradient credit: sum -c_i * dL_market/dc_i (the per-unit book CURRENCY.lock rejects);
- S leave-one-out market settlement: sum log p(y) - log p_{-i}(y) (head removed from the pool;
  Cw renormalises its stakes);
- K committed settlement (CURRENCY.lock rule): sum log p_i(y) - log p0(y), p_i = sigma(l0 + c_i).
  For U it is computed but the head never posted that price (counterfactual commitment).

**Oracle.** Delta_{i,q} = mean BCE on the last 25% of positions (the "future window") of the
market with head i removed minus the full market, at trade time, no retraining.

**Metrics.** (1) Spearman over the 1152 traders of G, S, K (book window) vs Delta (future
window); (2) sign agreement: among traders with Delta > 0 (useful), fraction with positive
measure; (3) Gini of K over the 1152 traders (`panel.gini`, signed)
with the S0 shuffled-label null z (`panel.gini_null`, 20 permutations of labels within each question,
predictions held fixed); (4) performance: prequential BCE
on the future window, learnable families (periodic, lfsr16, ca110, prime, and, or,
periodic_ho) mean, and noise excess max(BCE - ln 2) over prng and ca30.

**Hypotheses.** H1: in U, Spearman(G, Delta) < 0.3 and sign agreement of G < 0.7 (replicates the
sign problem at the output layer). H2: in Cw, Spearman(K, Delta) >= Spearman(G, Delta in U) + 0.2
in 3/3 seeds (commitment changes what wealth means). H3: Cw learnable BCE within 1% of U.

**Stop rule (base arms survive or not).** Following the M6 table rule: a committed arm (Cm, Cw)
with learnable BCE >= U + 1% in 3/3 seeds is dropped from the stress pass. If H2 also fails in
3/3 seeds, the commit item is reported as "commitment costs accuracy and does not change wealth"
and stopped. No stress tests in this pass. CPU only (9 runs, at most 3 concurrent).

## Amendment 1 (2026-09-28 ~06:50Z, after the first 12 runs, before any Cs run)

Observed in v1 (`results/v1/*.json`): committed heads trained block-locally on the same target
collapse to clones (mean pairwise corr of head corrections 0.99), so each head's leave-one-out
value is ~1e-8 and every Spearman against it is noise. Also the pooled Spearman over 1152 traders
mixes questions (heads matter more on learnable questions). Changes, applied to every arm by a
rerun (v2, same seeds):
- New arm `Cs` committed *sequential*: head i trades at the price left by heads < i,
  p_{<i} = sigma(l0 + sum_{j<i} c_j) (detached for training), posts p_{<=i}, is trained on its
  own loss -log p_{<=i}(y) and settled K_i = log p_{<=i}(y) - log p_{<i}(y) (exact LMSR at the
  trade price; sum_i K_i = log p(y) - log p0(y) exactly). Market l = l0 + sum_i c_i.
  For Cs, K is its own settlement, not the stand-alone one.
- Metrics added: within-question Spearman (mean over questions of Spearman across the 16 heads),
  a stand-alone oracle Delta_alone = BCE(l0) - BCE(l0 + c_i) on the future window, and per-trader
  arrays saved as `results/<arm>-s<seed>.npz`.
- H2 is judged on Cs (and Cw) with the within-question Spearman. Stop rule unchanged.
