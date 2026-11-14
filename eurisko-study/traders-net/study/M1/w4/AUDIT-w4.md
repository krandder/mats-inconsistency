# M1 (d): where the books leak, and AttnLRP-rule books (worker 4)

Checked at 2026-09-28 00:15Z. Pre-registration: `PREREG.md` (written before the first run).
Code: `leak.py` (per-checkpoint books under 4 rule sets, per-op defect split, exact single-unit
zero ablation), `analyze.py` (CSV/JSON/PNG). Raw: `results/s{1,2,3}-step{8000,16000,24000,30000}.json`.
Tables: `resid.csv`, `ops.csv`, `ablation_spearman.csv`, `rule_vs_rule.csv`, `summary.json`; figure `leak.png`.
Substrate: wealth-net train weights (Adam, 3 seeds), read-only. Books on the held-out tail
(2397 examples); ablation on every second of those (1199). Numbers are mean ± sd over the 3 seeds
per checkpoint; "pooled" = mean ± sd over the 12 (seed, checkpoint) pairs.
Rerun: `cd M1/w4 && PYTHONPATH=<wealth-net>/.pylib python3 leak.py --seeds 1 && ... && python3 analyze.py` (~40 min per seed at load 40-55).

## Rules compared
| rule | LN denominator | attention pattern A | bilinear Q.K^T, A.V |
|---|---|---|---|
| `live` | live | live | plain gradient |
| `v1` (wealth-net books.py) | detached | live | plain gradient |
| `cp` (CP-LRP, Ali et al. 2022, https://arxiv.org/abs/2202.07304) | detached | detached | plain gradient |
| `attnlrp` (Achtibat et al. 2024, https://arxiv.org/abs/2402.05602) | detached (Prop. 3.4 identity rule) | live | uniform 1/2 rule (Props. 3.2-3.3); softmax Prop. 3.1 = grad x input on the modified gradient |

Rules verified against the arXiv HTML of 2402.05602 on 2026-09-27 (softmax rule R_i = x_i (R_i - s_i sum_j R_j);
matmul uniform rule; LN identity rule; the paper accepts the softmax conservation violation as a "hidden bias").

## 1. The split closes exactly
Local defect of an op = relevance out - relevance in (biases, LN shifts counted as inputs). Every other op
(linear, ReLU, residual add, mean-centering) is exactly conservative, so the defects downstream of each
cut must equal that cut's residual. They do: max |closure| / |R| = 7.7e-8 over all 48 (rule, seed, ckpt)
(`resid.csv`, column `closure_max`). The split below is therefore an exact accounting, not an estimate.

## 2. Residual at the input cut (cut 0), relative to |R| (`summary.json` `resid_cut0`)
| rule | 8k | 16k | 24k | 30k |
|---|---|---|---|---|
| live | -1.142 ± 0.090 | -1.073 ± 0.076 | -1.097 ± 0.094 | -1.048 ± 0.073 |
| v1 | +0.125 ± 0.010 | +0.283 ± 0.067 | +0.377 ± 0.068 | +0.416 ± 0.110 |
| cp | 0 (< 1e-8) | 0 | 0 | 0 |
| attnlrp | +0.001 ± 0.001 | -0.007 ± 0.014 | -0.016 ± 0.016 | -0.018 ± 0.018 |

## 3. Which op leaks (pooled local defect / |R|, `ops.csv`, `summary.json` `ops`, `ops_absshare`)
- **live:** LN is 94.8 ± 2.4% of the total |defect|. The final LN alone absorbs all of R
  (lnf -1.000 ± 0.003): a scale-invariant LN makes grad x input upstream of it sum to ~0 (Euler, degree 0),
  so plain gradient x input books do not conserve at all. The inner LNs push relevance around
  (L0.ln2 +0.14, L1.ln2 -0.26, L1.ln1 +0.08). QK/softmax/AV are 2-3% each.
- **v1:** no LN defect (detached); the leak is QK 45 ± 5% and AV 49 ± 1% of |defect|, softmax 5 ± 5%.
  AV and QK are degree-2 products: the pattern path hands the inputs a second copy of the value relevance.
  Per layer: L1.AV +0.109 ± 0.056, L1.QK +0.093 ± 0.041, L0.AV +0.063 ± 0.036, L0.QK +0.057 ± 0.028.
  The v1 residual grows with training (0.125 -> 0.416) as attention patterns sharpen.
- **attnlrp:** the only leak is softmax (100% by construction), -0.008 ± 0.011 (L1) of |R|; total <= 2%.
- **cp:** exact to float error (total |defect| / |R| ~ 5e-9); the "shares" for cp in `summary.json` are shares of float noise.

## 4. Do the rules change who is rich? (`rule_vs_rule.csv`, pooled Spearman of per-unit income)
| pair | neurons (512) | heads (8) |
|---|---|---|
| v1 ~ cp | 0.987 ± 0.006 | 0.937 ± 0.071 |
| v1 ~ attnlrp | 0.985 ± 0.007 | 0.944 ± 0.051 |
| cp ~ attnlrp | 0.993 ± 0.004 | 0.984 ± 0.021 |
| live ~ v1 | 0.788 ± 0.053 | 0.137 ± 0.366 |
Fixing the attention leak (cp, attnlrp) barely reorders units; detaching LN reorders them a lot,
heads most. Layer-1 neurons are identical under v1/cp/attnlrp (only lnf lies downstream).

## 5. Which book tracks held-out zero ablation? (`ablation_spearman.csv`; local sanity check only, M2 owns the oracle)
Spearman(unit income on held-out data, exact single-unit zero-ablation ΔBCE on held-out data), pooled 12 pairs:
| rule | neurons all | neurons L0 | neurons L1 | heads (n=8) |
|---|---|---|---|---|
| live | 0.382 ± 0.101 | 0.686 ± 0.065 | 0.135 ± 0.159 | 0.635 ± 0.259 |
| v1 | 0.230 ± 0.116 | 0.389 ± 0.081 | 0.101 ± 0.159 | -0.163 ± 0.263 |
| cp | 0.226 ± 0.121 | 0.384 ± 0.090 | 0.101 ± 0.159 | -0.181 ± 0.288 |
| attnlrp | 0.232 ± 0.120 | 0.397 ± 0.088 | 0.101 ± 0.159 | -0.147 ± 0.284 |
Plain grad x input (live) is the first-order Taylor estimate of the zero-ablation effect, and it wins
at every checkpoint (per-step values in `summary.json`, figure `leak.png` right). Every conservative rule
buys conservation with the LN detach, and the LN detach is what costs faithfulness to ablation
(neurons L0 0.69 -> 0.39; heads 0.64 -> -0.16, with only 8 heads per net).

## 6. Consequence for CURRENCY.lock (input for the lead, worker 1)
- If the currency must conserve (wealth of a coalition = sum of members, internal transfers cancel), use
  **`cp`** (LN and attention pattern detached): exact to 1e-8, cheapest (one plain backward, ~40% faster
  than attnlrp in the smoke test: 4.1 s vs 6.4 s per 512 examples), and ranks units like v1/attnlrp (>= 0.98 neurons).
  AttnLRP gives no ranking benefit over cp here and leaves a <= 2% softmax leak.
- The v1 leak is not LN (the docstring's worry) but QK + AV double counting, and it grows to ~40% of |R|
  by 30k steps, so v1 cut sums are not trustworthy late in training.
- The price of conservation is the LN convention: conserving books track single-unit ablation worse than
  plain grad x input. M2 should carry `live` grad x input as a separate predictor, and any "wealth beats
  movement pruning" claim must say which LN convention it used.

## Limits
Held-out eval only (not the prequential books of item c); zero ablation, single units, 1199 examples;
Adam-trained wealth-net weights only (item b's SGD runs not audited); head Spearman has n=8 per net.
Per-unit AttnLRP with epsilon=0 on linear layers (biases counted as units, so the epsilon rule's bias
absorption is booked to bias accounts instead of dropped).
