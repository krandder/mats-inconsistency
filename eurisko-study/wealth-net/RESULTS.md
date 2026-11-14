# wealth-net: results

Page: http://[ip-removed]:8765/wealth-net/ (file: `results/index.html`, copy at
`/home/agent/memo-serve/wealth-net/index.html`). Numbers below are read from
`results/summary.json` and `results/ablation.csv`, checked at 2026-09-27 22:40Z.

## What ran

- 64 questions, 8 families (periodic, LFSR-16, CA rule 30, CA rule 110, primality, pi-digit
  parity, A and B, A or B), window 32, stream length 24000 per seed; train on the first 80%,
  evaluate on the last 20%.
- Transformer: 2 layers, d=64, 4 heads per layer (8 heads), 512 MLP neurons, one token per
  question. Adam 1e-3, batch 128.
- 3 seeds on one spot L4 GPU (in parallel), 50-minute wall-clock cap, about 31k steps each
  (the 60k-step target was not reached; the time cap bound first).
- Books every step, snapshots every 100 steps, weights every 2k steps (`results/weights/`).
- Controls, 500 steps each: frozen (the final trained weights, no updates) and random init
  (no training).
- Ablation (Kelvin's test) on a fresh segment: positions 24000-28095 of the same streams,
  never seen in training. It used checkpoints at 8k, 20k and 30k steps plus the final one;
  5 criteria; removal fractions 0, 1, 2, 5, 10, 20 and 40%. 840 rows.
- GPU cost: about $1.02 at the list-price ceiling ($0.87 for 1.00 h on the second VM, plus about
  10 minutes on a first VM that was preempted). Both VMs are gone: `gcloud compute instances
  list --filter=name~wealth` returns 0 items. The DELETE line is in `logical-soup/GPU-LOG.md`.

## Numbers (final step; the three values are seeds 1 / 2 / 3)

| | s1 | s2 | s3 |
|---|---|---|---|
| step | 31100 | 31200 | 31100 |
| train loss (BCE) | 0.103 | 0.137 | 0.136 |
| total revenue R (sum of s times z) | 2482 | 2083 | 2412 |
| neurons with negative wealth (of 512) | 188 | 211 | 191 |
| their share of neuron activation | 0.39 | 0.44 | 0.40 |
| heads with negative wealth (of 8) | 7 | 7 | 8 |
| parasite neurons (negative in 90% of 2nd-half snapshots) | 121 | 160 | 140 |
| parasites' activation share | 0.26 | 0.34 | 0.30 |
| dead neurons | 0 | 0 | 0 |
| Gini of positive wealth: neurons / heads / outputs | 0.66 / 0.88 / 0.38 | 0.69 / 0.88 / 0.43 | 0.68 / 0 (all negative) / 0.37 |
| wealth rank corr vs 1000 steps earlier (neurons) | 0.996 | 0.997 | 0.997 |
| income rank corr, window vs previous window (neurons) | -0.29 | -0.29 | -0.24 |
| Spearman(wealth, loss-grad norm), neurons | -0.43 | -0.45 | -0.23 |
| cumulative conservation residual, cut 0 / cut 1 / cuts 3-4 (relative to R) | -0.67 / -0.52 / <1e-5 | -1.10 / -0.66 / <1e-5 | -0.78 / -0.36 / <1e-5 |
| frozen-attention check, max relative residual | 1e-6 | 2e-6 | 4e-6 |
| sum rule (heads + bias = attn component; neurons + bias = MLP component), max | 1.8e-6 | 1.6e-6 | 1.5e-6 |

Held-out BCE per family (in-training eval split, mean of 3 seeds), net vs logistic baseline:

| family | periodic | lfsr16 | ca30 | ca110 | prime | pi | and | or |
|---|---|---|---|---|---|---|---|---|
| net | 0.000 | 0.002 | 0.96 | 0.03 | 0.49 | 4.76 | 0.51 | 0.92 |
| logistic | 0.000 | 0.70 | 0.70 | 0.41 | 0.20 | 0.70 | 0.33 | 0.58 |

Which families pay the output units (cumulative, s1 / s2 / s3): ca30 is the only net loser
(-508 / -376 / -386). By component, the question-id embedding holds most of the wealth
(5679 / 6301 / 5095), the attention layers are negative, and the window embedding is negative.

## Ablation: does deleting the poorest traders help?

Fresh-segment BCE at the final checkpoint, mean of 3 seeds (intact: BCE 1.051, Brier 0.150).
Random is the mean of 5 draws.

| neurons removed | 1% | 2% | 5% | 10% | 20% | 40% |
|---|---|---|---|---|---|---|
| (a) most negative wealth | 1.300 | 1.426 | 1.528 | 1.680 | 1.873 | 2.487 |
| (b) most negative recent revenue (last 2k steps) | 1.291 | 1.371 | 1.559 | 1.636 | 2.015 | 3.387 |
| (c) random | 1.074 | 1.114 | 1.274 | 1.574 | 1.922 | 2.356 |
| (d) smallest loss-gradient norm | 1.061 | 1.074 | 1.181 | 1.231 | 1.457 | 2.326 |
| (e) smallest mean activation | 1.059 | 1.106 | 1.206 | 1.275 | 1.471 | 1.811 |

- The claim fails for neurons. Removing the poorest 1% (5 neurons) raises BCE by 0.25; removing
  5 random neurons raises it by 0.02. The damage lands on LFSR: its BCE goes from 0.002 to 1.48 at
  1% by wealth, and it is still 0.014 at 1% random. OR goes from 0.94 to 1.41. Excluding pi,
  BCE goes 0.43 to 0.72 (wealth) against 0.44 (random).
- By seed, at the final checkpoint: at 1-5% the poorest neurons cost more than random in seeds 1
  and 3 (seed 1 at 1%: 1.76 against 1.09). In seed 2 they are within 0.01 of random up to 2%.
  At 10-20% the order flips in seeds 2 and 3, where wealth beats random (seed 3 at 10%: 1.68
  against 1.91), but it never beats the smallest-gradient order. The 8k, 20k and 30k checkpoints
  show the same small-fraction pattern (seed means at 2%: 0.94/1.18/1.34 by wealth against
  0.77/1.10/1.26 random).
- Heads: with 8 heads, 1-5% removes nothing (it rounds to 0), 10% removes 1 head and 20% removes 1-2.
  At 20% the poorest heads cost less than random (1.115 against 1.124) but more than the
  smallest-gradient heads (1.058). Too few units to call it.
- Brier gives the same ordering (neurons at 5%: wealth 0.180, random 0.161, gradnorm 0.163).
- The cheapest units to remove are the smallest-gradient and smallest-activation ones, not the
  poorest.
- Curves are in `results/ablation.csv`, one row per seed, checkpoint, granularity, criterion and
  fraction, with per-family BCE and Brier.

## Five observations

1. **Surprising. The poorest neurons are load-bearing, not parasites.** A neuron's revenue is
   a first-order estimate of how much loss it saves. Negative wealth should mean that removing
   it lowers the loss. Instead the most negative neurons are the ones LFSR depends on. The
   first-order books disagree with the actual ablation, and by a wide margin. The likely reason
   (untested) is that these neurons act in tight nonlinear pairs: each pays a small price
   alone, but the pair carries the answer.
2. **Surprising. Wealth is locked in, while income is anti-persistent.** Wealth rank barely
   moves over 1000 steps (0.99+). The neuron income rank in one 100-step window correlates
   negatively with the previous window (-0.24 to -0.29). The frozen control has stable income
   (+0.95), so the churn comes from learning itself: a neuron that earned last window is
   pushed by the update to overshoot the next one.
3. **Expected, larger than expected. The books balance exactly except across attention.**
   Above the attention layers every cut balances to 1e-5, and with attention patterns frozen
   the whole network balances to 4e-6. With live attention, the units below the attention
   layers are credited 67-110% more revenue than the network earns in total. Softmax is not
   homogeneous, so input-times-gradient credit is not conserved through it. Summing wealth
   across the attention boundary therefore double-counts.
4. **Surprising. Rich neurons get small training gradients.** Spearman between cumulative
   wealth and cumulative loss-gradient norm is negative for neurons (-0.23 to -0.45) and for
   output units. The optimizer works hardest on the units the market considers poor. This fits
   observation 1: gradient norm and wealth do not measure the same thing.
5. **Expected. The market charges for unlearnable questions.** CA rule 30 is the only family
   the net cannot learn at this size. It is the one family whose settlements drain the output
   units, and the window embedding loses money on it. The network also overfits pi
   (memorised windows: held-out BCE 4.8 against 0.69 for a coin flip), which the books show as
   positive in-sample revenue. The books reward overfitting as long as it pays on training data.

Also noted: all 8 heads are poor in seed 3 and 7 of 8 in seeds 1 and 2, while the question-id
embedding holds most of the wealth. The random-init control loses 6000-8000 in 500 steps, with
more than half its compute in parasites.

## Unverified

- Why negative-wealth neurons matter for LFSR (the pairs hypothesis in observation 1): no
  pairwise ablation was run.
- Whether the attention residual is fully explained by softmax non-homogeneity. LayerNorm is
  frozen in the books (std detached) but live in the model and in the ablation; its share of
  the gap is not measured.
- Dollar cost: the GPU-LOG figure is an on-demand list-price ceiling. The Billing API is
  disabled, so the actual spot charge is unknown (lower).
- Only 3 seeds, one model size, about 31k steps. Nothing here is tested for scale.
- The fresh segment reuses the same questions at later positions, not new questions, because
  the question-id embedding is tied to each question.
- Pi and CA30 dominate mean BCE. The ablation conclusions hold with pi excluded (checked at the
  final checkpoint only).

## Decayed wealth (follow-up, 2026-09-27 23:33-23:50Z, CPU only, no training)

Kelvin asked: all-time wealth is history; is recent income the currency that matters? `decay.py`
turns each neuron's income per 100 steps (from the snapshots) into decayed wealth
W_t = λW_{t-1} + income_t, with half-lives 100, 300, 1000, 3000 and 10000 steps, and into income
over the last 100, 500 and 2000 steps. The oracle is the real change in held-out BCE when one
neuron is zeroed, for all 512 neurons at the final checkpoint (positive = the neuron helps).
The held-out segment is the one above, subsampled to fit the CPU budget: every 8th position for
the oracle, every 4th for the curves. Data: `results/decay_ablation_s*.csv`,
`results/decay_spearman_s*.csv`, `results/decay_oracle_s*.npy`. Total run time 16 minutes.

Spearman between each currency and the oracle (final checkpoint), and the share of neurons with
a negative currency:

| currency | s1 | s2 | s3 | mean | share negative (s1/s2/s3) |
|---|---|---|---|---|---|
| all-time wealth | +0.026 | -0.005 | +0.108 | +0.043 | 0.37 / 0.41 / 0.37 |
| half-life 100 | -0.008 | -0.051 | +0.008 | -0.017 | 0.53 / 0.51 / 0.52 |
| half-life 300 | -0.024 | -0.037 | +0.042 | -0.006 | 0.55 / 0.57 / 0.52 |
| half-life 1000 | -0.012 | -0.040 | +0.054 | +0.001 | 0.57 / 0.56 / 0.52 |
| half-life 3000 | -0.014 | -0.047 | +0.060 | -0.000 | 0.53 / 0.56 / 0.52 |
| half-life 10000 | -0.007 | -0.032 | +0.075 | +0.012 | 0.48 / 0.50 / 0.44 |
| income, last 100 | +0.014 | -0.063 | +0.013 | -0.012 | 0.51 / 0.47 / 0.51 |
| income, last 500 | -0.050 | -0.013 | +0.007 | -0.019 | 0.55 / 0.57 / 0.49 |
| income, last 2000 | +0.002 | -0.030 | +0.059 | +0.010 | 0.55 / 0.57 / 0.54 |
| loss-gradient norm (control) | +0.232 | +0.180 | +0.093 | +0.168 | - |

Best half-life: 10000 steps (mean +0.012). No decayed or windowed currency beats all-time
wealth (+0.043), and all of them are within noise of zero. Removing one neuron alone lowers
held-out BCE for 104 / 76 / 118 neurons.

Held-out BCE increase when the lowest 1% / 5% / 20% of neurons by each currency are removed (mean
of 3 seeds):

- Final checkpoint:
  - all-time wealth: +0.25 / +0.50 / +0.82
  - half-life 100: +0.16 / +0.53 / +1.11
  - half-life 10000: +0.26 / +0.55 / +0.82
  - last 500 steps: +0.24 / +0.40 / +1.04
  - random: +0.02 / +0.26 / +0.76
  - smallest gradient norm: +0.01 / +0.13 / +0.39
- Step 20000:
  - all-time wealth: +0.20 / +0.33 / +0.67
  - half-life 100: +0.69 / +1.19 / +1.47
  - last 100 steps: +0.75 / +1.03 / +1.67
  - random: +0.02 / +0.14 / +0.69

Three observations:

1. **Surprising. Decay does not help; no currency predicts which neurons matter.** Across all 512
   neurons, every currency's rank against the true single-neuron ablation has a Spearman
   correlation within ±0.11 of zero. The best half-life, 10000 steps, is the one closest to no
   decay. Plain loss-gradient norm does better (+0.17).
2. **Expected in hindsight. The books find the important neurons but get the sign wrong.** A handful
   of neurons carry most of the value (single-neuron ablation raises BCE by up to 0.6-2.0; the
   median is 0.004). Every currency puts them at the extremes, but at either extreme: for example,
   seed 1's top four sit at percentiles 33, 100, 3 and 0 of all-time wealth. So removing the
   lowest-currency neurons is always worse than removing random ones, whatever the half-life.
3. **Surprising. Short memories are worse mid-training.** At step 20000, removing the lowest 1% by
   income over the last 100 steps, or by half-life 100, costs +0.69 to +0.75 BCE, three to four times
   what all-time wealth costs (+0.20). Short-window income swings sign from window to window
   (observation 2 above), so it mostly measures noise. By the final checkpoint the gap has mostly
   closed.

Unverified here: the oracle uses a 512-position subsample, so for small effects it is noisy.
The median dloss is 0.004, and the per-neuron sampling error is not measured. Interactions
between neurons (removing several at once) are not in the oracle.

## Exact settlement (follow-up, 2026-09-28 00:30-00:40Z, CPU only, no training)

Gradient credit is replaced by the market's own settlement at the output (`settle.py`). A neuron's
direct logit contribution c_i is its output vector read through the final LayerNorm (scale held
fixed) and the readout. It matches a real direct-path removal to 7e-6, and needs no extra forward
pass. Each neuron earns the log-score gain it caused:

- **Leave-one-out:** log p(y) − log p_{−i}(y).
- **Layer by layer:** each MLP layer earns the logit-lens gain across it. That is split inside the
  layer by leave-one-out, with the leftover spread equally, so the layer's neurons sum to its
  earnings (checked to 0.006 out of about 2e5).
- **Sampled Shapley:** 8 random neuron orders per example over the 512 direct contributions. It
  sums to the total gain within 0.005.

Each is summed over the held-out segment (the labels the oracle is measured on, so an upper bound)
and over 4096 training positions (what a market would have seen). Data:
`results/settle_spearman_s*.csv`, `results/settle_ablation_s*.csv`,
`results/settle_currencies_s*_step*.npz`.

Spearman against the single-neuron ablation oracle, final checkpoint:

| settlement | data | s1 | s2 | s3 | mean | layer 0 only | layer 1 only | oracle top 5% in its top 10% | in its bottom 10% | share negative |
|---|---|---|---|---|---|---|---|---|---|---|
| leave-one-out | held-out | +0.595 | +0.593 | +0.539 | **+0.576** | +0.31 | +0.87 | 0.73 | 0.03 | 0.10 |
| layer by layer | held-out | +0.552 | +0.548 | +0.139 | +0.413 | +0.16 | +0.87 | 0.74 | 0.01 | 0.63 |
| sampled Shapley | held-out | -0.035 | -0.025 | +0.048 | -0.004 | +0.04 | -0.00 | 0.19 | 0.26 | 0.50 |
| leave-one-out | training | +0.443 | +0.430 | +0.379 | +0.417 | +0.27 | +0.71 | 0.69 | 0.01 | 0.00 |
| layer by layer | training | +0.468 | +0.463 | +0.044 | +0.325 | +0.14 | +0.71 | 0.56 | 0.03 | 0.56 |
| sampled Shapley | training | +0.005 | +0.099 | +0.065 | +0.057 | +0.00 | +0.05 | 0.22 | 0.31 | 0.35 |
| (earlier) all-time gradient wealth | training | +0.026 | -0.005 | +0.108 | +0.043 | | | | | 0.38 |
| (earlier) loss-gradient norm | training | +0.232 | +0.180 | +0.093 | +0.168 | | | | | |

Held-out BCE increase when the lowest 1% / 5% / 20% are removed, final checkpoint, mean of 3 seeds:

| currency | 1% | 5% | 20% |
|---|---|---|---|
| leave-one-out, held-out data | −0.007 | +0.065 | +0.345 |
| leave-one-out, training data | +0.015 | +0.108 | +0.363 |
| layer by layer, held-out data | +0.051 | +0.240 | +0.596 |
| sampled Shapley, held-out data | +0.825 | +1.848 | +2.566 |
| smallest gradient norm | +0.009 | +0.126 | +0.392 |
| random | +0.023 | +0.262 | +0.760 |
| all-time gradient wealth | +0.245 | +0.500 | +0.821 |

At step 20000 the ordering is the same: leave-one-out on held-out data gives −0.007 / +0.035 /
+0.262.

Three observations:

1. **Surprising, and good news for the idea. Exact leave-one-out settlement works where gradient
   credit did not.**
   - Its Spearman correlation with the true ablation effect is +0.58 on held-out data, and +0.42
     on training data alone. Gradient wealth scored +0.04 and gradient norm +0.17.
   - It ranks 73% of the most important neurons in its top 10%, and almost none (1-3%) in its
     bottom 10%.
   - Pruning by it is the best criterion tested, at or below gradient norm at every fraction
     except 10% at the final checkpoint (+0.18 against +0.17).
     Removing its lowest 1% slightly improves held-out BCE (−0.007 at both checkpoints).
   - The bet-sized, nonlinear settlement is what fixes the sign problem of gradient credit.
2. **Expected. It is exact for the last layer and weaker for the first.** For layer-1 neurons,
   whose direct path to the output is their whole effect, leave-one-out settlement tracks the
   oracle at +0.87 (held-out) and +0.71 (training). For layer-0 neurons it is +0.27 to +0.31,
   because the direct path leaves out their effect through layer 1. Layer-by-layer settlement adds
   noise at layer 0 without fixing that; seed 3 falls to +0.14.
3. **Surprising. Shapley, the "fair" split, is useless here and harmful for pruning.**
   - Spearman is about 0, and pruning its lowest 1% costs +0.8 BCE, the worst criterion of all.
   - Hypothesis (not measured): the 512 direct contributions are large and cancel each other.
     Random partial coalitions then sit far from the real logit, where the log score saturates.
     The biggest neurons collect large, sign-random marginals: |Shapley| correlates +0.16 to
     +0.24 with the oracle.
   - Settlement priced at the actual market state (leave-one-out) is the useful one.

Unverified here:
- The held-out currencies are summed over the same labels the oracle uses; the training-data
  columns are the fair comparison.
- The layer-0 gap is attributed to the missing indirect path, not measured.
- Only 8 Shapley orders per example.
- The oracle is from a 512-position subsample.

## Block-local settlement (follow-up, 2026-09-28)

Traders as blocks. Each MLP block trains on its own loss (the cross-entropy of the logit lens after
that block) with the incoming residual detached, so no gradient crosses a block boundary. Arms:
A end-to-end (final readout only), B block-local, C block-local plus a per-block market (outputs scaled
by N·w, w updated from leave-one-out settlement at the block's own price, eta 1, floor 0.1, births every
200 steps). Leaky sweep: arm B with a share alpha of the gradient crossing each boundary (alpha 0 = B;
alpha 1 = end-to-end on the sum of block losses, not the same as A). 30000 steps, 3 seeds each, two spot L4
VMs (cost ceiling $0.38 + $0.47, both deleted, GPU-LOG.md). Code: `blocklocal.py`, `gpu-bl/run_bl.sh`;
data: `results/blocklocal/`. Settlement is on 4096 training positions at the arm's own price; the oracle is
single-neuron ablation on the fresh held-out segment. Mean of 3 seeds.

| measure | A end-to-end | B block-local | C block-local + market | C-random | C-wealth only |
|---|---|---|---|---|---|
| held-out BCE | 1.042 | 0.859 | 0.299 | 0.335 | 0.306 |
| price of anarchy (BCE − A) | 0 | −0.183 | −0.743 | −0.708 | −0.736 |
| Spearman settlement vs oracle | +0.395 | +0.857 | +0.186 | −0.151 | +0.272 |
| Spearman wealth vs oracle | – | – | +0.107 | – | +0.268 |
| Gini of settlement (positive part) | 0.728 | 0.705 | 0.920 | 0.912 | 0.987 |
| Gini of wealth | – | – | 0.693 | – | 0.898 |
| share negative settlement | 0.003 | 0.020 | 0.225 | 0.165 | 0.333 |
| units alive / births | 512 / 0 | 512 / 0 | 359 / 12794 | 359 / 12493 | 512 / 0 |
| max single-unit damage | 1.312 | 0.527 | 0.017 | 0.092 | 0.768 |
| remove top 1% (5 units) | +1.987 | +0.876 | +0.013 | +0.095 | +0.828 |
| remove random 1% | +0.036 | +0.055 | +0.000 | −0.000 | +0.139 |

Held-out BCE by family: periodic 0.000 / 0.000 / 0.000, lfsr16 0.002 / 0.001 / 0.002, ca30 0.948 / 0.751 / 0.702,
ca110 0.030 / 0.017 / 0.021, prime 0.567 / 0.241 / 0.220, pi 5.377 / 5.023 / 0.744, and 0.491 / 0.264 / 0.273,
or 0.922 / 0.572 / 0.430 (A / B / C). Controls (C-random / C-wealth only): ca30 0.700 / 0.710, prime 0.209 / 0.220,
pi 1.116 / 0.830, and 0.229 / 0.255, or 0.399 / 0.408; periodic, lfsr16 and ca110 at most 0.022.

Controls for C (added after the first run, one more spot L4, cost ceiling $0.30, deleted): C-random skips and
rebirths the same number of units per step as C did (schedule read from C's log for the same seed) but picks them
uniformly at random, with no wealth scaling; C-wealth only keeps the wealth scaling but never skips or rebirths.

Leaky boundary (B with leak alpha; gap is BCE − A):

| alpha | held-out BCE | gap | Spearman vs oracle | Gini | share negative | max single damage | remove top 1% |
|---|---|---|---|---|---|---|---|
| 0 (= B) | 0.859 | −0.183 | +0.857 | 0.705 | 0.020 | 0.527 | +0.876 |
| 0.1 | 0.930 | −0.112 | +0.865 | 0.716 | 0.027 | 1.066 | +1.488 |
| 0.3 | 1.022 | −0.020 | +0.860 | 0.709 | 0.002 | 1.070 | +1.389 |
| 0.6 | 1.054 | +0.012 | +0.849 | 0.713 | 0.001 | 0.975 | +1.462 |
| 1.0 | 1.099 | +0.057 | +0.810 | 0.713 | 0.002 | 0.945 | +1.827 |

Three observations:

1. On this benchmark the price of anarchy is negative: cutting gradients at block boundaries lowers held-out BCE
   (B −0.18, C −0.74), and BCE rises steadily as the boundary leaks (alpha 0 → 1: 0.86 → 1.10). The gain comes from
   the families nobody can predict. On the learnable ones (periodic, lfsr16, ca110) all arms are within 0.02.
   On pi, ca30, prime, and and or, arm A memorises training windows and is confidently wrong on fresh data (pi 5.4
   nats against 0.69 for a coin flip). That repeats the earlier finding: intact A = 1.042 here, 1.051 in the
   first run. So this measures regularisation against memorisation, not a cost of cooperation on learnable
   structure. The market arm has the highest training loss of the three and is nearly at chance on pi (0.744).
2. Block-local prices make settlement honest. Leave-one-out settlement at the block's own lens agrees with the
   ablation oracle at Spearman +0.86 in B (at every alpha up to 0.6), against +0.40 for end-to-end A. When each
   block is paid for its own prediction, what a unit earns there is what the network loses without it. The market
   (C) breaks that agreement (+0.19 for settlement, +0.11 for wealth): its units are near-interchangeable
   (max single damage 0.017), so the oracle is mostly noise to rank against.
3. The market removes over-reliance (though observation 4 shows random skipping does too), at the price of a violent ecology. In C, removing the top 1% of units costs
   0.013 nats (A: 1.99, B: 0.88), and the most important single unit costs 0.017 (A: 1.31). But 30% of units
   are skipped at any time, there are about 12800 births per seed (about 0.4 per step), 22% of units have
   negative settlement and settlement Gini is 0.92. Any gradient leak at all (alpha 0.1) doubles the maximum
   single-unit damage of B (0.53 → 1.07).

4. Controls: on 3 seeds, settlement adds little beyond dropout-like skipping and rebirth. Both controls
   get most of C's held-out gain (C 0.299, C-wealth only 0.306, C-random 0.335), and C's edge over C-random is
   almost all pi (0.74 against 1.12). Robustness comes from skipping and rebirth, not from choosing units by
   settlement. Removing the top 1% costs 0.007 / 0.010 / 0.023 in C by seed and 0.263 / 0.020 / 0.0003 in
   C-random, so C-random is as robust in two seeds of three and its mean (0.095) is one bad seed. Wealth scaling
   without skipping regularises but leaves strong over-reliance (0.67 to 0.93). What settlement plausibly buys
   here is a small BCE edge on pi and fewer bad seeds; 3 seeds cannot tell those apart from noise.

Unverified: whether the ordering survives on a benchmark without unpredictable families, or with early
stopping for A. Whether C's small edge over C-random (pi, and the one
bad C-random seed) holds up with more seeds.
