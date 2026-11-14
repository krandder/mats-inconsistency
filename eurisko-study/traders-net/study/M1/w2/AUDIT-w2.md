# M1 (b): wealth-net books under plain SGD and SGD + weight decay (worker 2)

Checked at 2026-09-28 01:07Z. Pre-registration `PREREG-w2.md` (amendment 1 = the eta=0.3 arm,
exploratory, added after the main runs). All numbers: mean ± sample sd over seeds 1-3, final step
10000, from `gpu/analysis/summary.csv` (main, eta=3) and `gpu-eta03/analysis/summary.csv`
(amendment); time courses and the squared-gradient correlations from `*/analysis/extra_summary.csv`.
Figure: `gpu/analysis/wealth_vs_normgrowth.png` (seed 1, readouts and neurons, adam/sgd/sgdwd).
GPU: G1 jobs `m1-b-sgd-adam` ($0.75) and `m1-b-eta03` ($0.52), audited in `study/G1/COST.md`, 0 VMs left.
Rerun: `G1/infra/submit.sh --ws M1 --src M1/w2/code ...` (run.sh = main, run_small.sh = amendment), then
`python3 code/analyze_w2.py <runs> <out>; python3 code/analyze_extra.py <runs> <out>` (needs pandas).

Setup: wealth-net unchanged (same streams incl. pi, model, `books.account` every step, in-sample
minibatches), only the optimiser swapped. Per unit: W = books wealth; T = -Q Σ θ·g (true loss gradient,
= signed movement score × Q); N = Q Δ|θ|²/(2η); rent = Qλ Σ|θ|²; second = (Qη/2) Σ|g+λθ|².
Units: 64 readouts, 512 neurons (θ = fc2 column, "neuron"; or fc1 row + bias, "neuron_in"), 8 heads.

## 1. The identity holds exactly; its second-order term does not vanish at practical step sizes

Under SGD and SGD+wd, T = N + rent − second per unit to float precision: max relative error
≤ 3e-6 in every run (`ident_err_rel`). So "income = norm growth" (T1 lemma 1) is exact only after
subtracting (Qη/2)Σ|g|², the accumulated squared-gradient energy, and adding the weight-decay rent.

Share of that second-order term, Σ|second| / Σ|T|:

| eta | readouts | neurons (out side) | neurons (in side) | heads |
|---|---|---|---|---|
| 3 (pilot choice) | 0.63 ± 0.02 | 2.87 ± 0.12 | 3.85 ± 0.14 | 1.86 ± 0.07 |
| 0.3 (amendment) | 0.034 ± 0.003 | 0.73 ± 0.11 | 0.34 ± 0.06 | 0.34 ± 0.11 |

At eta=3 (largest non-diverging of {0.3, 1, 3}; `gpu/choice.txt`, pilot `gpu/pilot/*.log`) hidden
units' norms grow mainly from gradient noise, not from income: R²(T, N) = 0.17 ± 0.05 for neurons.
Pre-registered prediction "sgd R²(T,N) ≥ 0.99" FAILED at eta=3 (readouts 0.915 ± 0.008) and at
eta=0.3 held only for readouts (1.000) and neuron in-side (0.93 ± 0.05; out-side 0.81 ± 0.15).

## 2. Books wealth vs norm growth (primary metric, R² with intercept)

| arm | readouts R²(W,N) | neurons R²(W,N) | heads R²(W,N) | eval BCE |
|---|---|---|---|---|
| adam 1e-3 | 0.49 ± 0.03 | 0.35 ± 0.04 | 0.40 ± 0.44 | 0.70 ± 0.03 |
| sgd eta=3 | 0.915 ± 0.008 | 0.03 ± 0.03 | 0.84 ± 0.03 (Spearman −0.82 ± 0.10) | 0.47 ± 0.08 |
| sgdwd eta=3, λ=3.3e-5 | 0.84 ± 0.03 (N+rent: 0.93 ± 0.01) | 0.10 ± 0.01 | 0.42 ± 0.32 | 0.41 ± 0.02 |
| sgd eta=0.3 (expl.) | 1.000 ± 0.000 | 0.68 ± 0.10 | 0.31 ± 0.44 | 0.33 ± 0.02 |
| sgdwd eta=0.3, λ=3.3e-4 (expl.) | 0.72 ± 0.04 (N+rent: 1.000) | 0.45 ± 0.04 (N+rent: 0.54 ± 0.03) | 0.06 ± 0.08 | 0.31 ± 0.02 |

- **Readouts**: wealth is exactly movement (R²(W,T) = 1, |W−T|/|T| ~1e-8: no LN between rw and z),
  and under small-step SGD exactly norm growth; with weight decay exactly norm growth + rent
  (R² 0.72 → 1.000). Prediction "sgdwd R²(W,N) < R²(W,N+rent)" held for readouts in both etas.
- **Adam residual**: under Adam R²(T,N) is 0.49 ± 0.03 (readouts), 0.66 ± 0.02 (neurons); the SGD
  identity error is 1e3-5e3× |T| (it is not an identity there). Prediction "adam R²(W,N) ≥ 0.1 below
  sgd on neurons" held only at eta=0.3 (0.35 vs 0.68); at eta=3 SGD neurons are worse (0.03).
- **Neurons and heads**: books wealth differs from movement by the frozen-LN convention:
  Σ|W−T|/Σ|T| = 0.75 ± 0.08 (sgd eta=3), 2.58 ± 0.24 (sgd eta=0.3), 1.43 ± 0.15 (adam);
  R²(W,T) = 0.47 ± 0.29, 0.58 ± 0.22, 0.53 ± 0.14. So for hidden units the book is neither norm
  growth nor movement; the LN term is as large as the income itself (worker 4 splits it by source).

## 3. What wealth tracks instead: squared-gradient energy (a loss)

Spearman(W, second) for readouts: −0.89 ± 0.04 (sgd eta=3), −0.85 ± 0.02 (sgd eta=0.3),
−0.57 ± 0.08 (adam; there with Σ|g|² as the proxy). Readouts on hard questions (large, noisy loss
gradients: pi, ca30) grow in norm from noise and lose wealth. This explains wealth-net's negative
Spearman(wealth, grad norm) (−0.43/−0.45/−0.23, `wealth-net/RESULTS.md`). Heads under SGD eta=3:
all 8 lose wealth in seed 1 (`frac_W_neg` 1.0 ± 0) while their norms grow; wealth anti-correlates
with norm (Spearman(W, |θ|²) −0.82 ± 0.10).

## 4. Consequences for CURRENCY.lock (input to the lead)

- A norm-growth proxy for wealth is only valid for readouts and only at small SGD steps; it must
  subtract (Qη/2)Σ|g|² and add rent under wd. Under Adam (wealth-net) it does not hold (R² ≤ 0.5).
- For readouts, books = movement exactly, under every optimiser: the readout currency is signed
  movement pruning (premise confirmed, no LN leak).
- For neurons/heads the frozen-LN book departs from movement by 75-260% of |T|: a currency that
  claims "income = −θ·∇L" should use the live-gradient tangent T, or state the LN convention.

## Limits

wealth-net streams (pi = shifted copies; in-sample books) kept deliberately for comparability with
M1(a); 10k steps, not 31k; one eta per SGD arm per job (eta=0.3 arm is post hoc, exploratory);
per-unit sums are over all families.
