# M2 worker 1 prereg: the ablation oracle (`oracle.py`)

Written 2026-09-28 ~02:00Z, before any run. Item 1 of M2 (see ../STATUS.md).

**What.** Ground truth for M2: the exact change in held-out BCE when a unit or a set of units is
removed from a trained wealth-net model. Predictors (worker 2) and metrics (worker 3) are scored
against it; this item scores nothing itself.

**Models (read-only).** wealth-net `train-s{1,2,3}`, checkpoints 8000, 20000, 30000 and final
(31100/31200/31100): `../../../wealth-net/results/weights/train-sS-stepK.pt` and
`../../../wealth-net/results/train-sS.pt`. These are the 4 checkpoints `ablate.py` used.

**Held-out data.** The same fresh segment as wealth-net `ablate.py`: `make_questions(64, T+4096, seed)`,
windows predicting bits[T .. T+4095], T=24000, never used in training or in-training eval. All
4096 positions (no subsample; decay.py's 512-position oracle was noisy). Halves A = even positions,
B = odd positions, for split-half reliability.

**Units (arms).** 512 MLP neurons (unit id `N{l}.{i}`, flat index l*256+i = wealth-net snapshot
order), 8 heads (`H{l}.{h}`, flat l*4+h), 64 per-question readouts (`R{q}`: the logit z_q).

**Ablations.**
- zero: neuron activation := 0; head output hv := 0; readout logit z_q := 0 (p = 1/2).
- mean: replaced by its mean over 1024 training positions (the positions `ablate.py` draws:
  randint(0, 0.8(T-W)) with seed=seed), per (question, unit) (per question, head and channel for heads;
  per question for readouts).
- Δloss = BCE(ablated) − BCE(intact), mean over positions × questions; positive = the unit helps.
  Also per family, per half (A/B), and Δ Brier.

**Coalitions (per seed × checkpoint).** 200 random: 100 neuron sets (size log-uniform in 2..64),
40 head sets (size 2..6), 30 neuron sets within one layer (size 2..32), 30 mixed neuron+head
sets (size 3..24). 50 structured, built from the single-unit zero oracle of that checkpoint:
top-k neurons by Δloss (k = 2,4,8,16,32), the 15 pairs among the top-6 neurons (pairs hypothesis
of wealth-net obs. 1), all neurons of each layer, the heads of each layer, all 8 heads, the top-8
neurons per family by that family's Δloss (8), bottom-k neurons (k = 8, 32), top-8/16 per layer
(4), the most negative wealth-net all-time wealth neurons at 1% and 5% (2), the readouts of each
family (8), the top-2 heads (1). Each coalition gets zero and mean Δloss and the additivity gap
Δ(S) − Σ_{i∈S} Δ(i).

**Seeds.** 3 (wealth-net seeds 1-3); coalition sampling RNG = seed*1000+step.

**Checks (pass/fail, reported in ORACLE.md).**
1. Intact BCE at each checkpoint equals wealth-net `ablation.csv` frac=0 rows within 1e-4 (same
   segment, same weights); ablation.csv is read, not recomputed.
2. Incremental evaluation (cached residuals for layer-1 units) equals a full masked forward to
   1e-5 on 8 random units per granularity.
3. Split-half reliability: Spearman(Δ_A, Δ_B) over neurons, per checkpoint; mean ± sd over seeds.

**Stop rule.** If split-half Spearman over neurons is < 0.5 at the final checkpoint in 2/3 seeds,
the single-neuron oracle is flagged too noisy for fine rank metrics and worker 3 is told to use
|Δ| > 2·(half-difference sd) units only. No other stopping; this is a measurement.

**Compute.** CPU only (M2 cap), 3 jobs (one per seed) × ≤3 threads; bulky intermediates in
/tmp/traders-net/M2/w1/; only CSV/NPZ/JSON under study/M2/w1/results.

**Amendment 1 (2026-09-28 03:00Z, before any oracle output existed).** The box runs at load ~80 on 16
cores, so each 1-thread job gets ~20% of a core (~45-50 CPU-min per checkpoint with coalitions).
Coalitions (both modes) are run at the final and 20k checkpoints; 30k and 8k get singles only
(`--no-coalitions`) unless the load drops. Coalitions use half A (even positions, 2048) and are
compared with the singles' half-A Δ, so the additivity gap uses the same data on both sides.
