# M1 audit: which currency

Checked at 2026-09-28 01:55Z. All numbers are mean ± sample sd over seeds 1-3 unless marked, and each
names its file (paths relative to `traders-net/study/M1/`). Pre-registrations: `PREREG.md`, `lead/PREREG-lead.md`.

## Decision (Gate A, after the 00:25Z RETARGET)

`CURRENCY.lock`: **no bookkeeping currency for generic neurons; committed-trader settlement only.**
Units that post a price on a question (readouts, explicit traders) are paid the exact LMSR log-score
gain `log p_after(y) - log p_trade(y)`, booked at the trade-time price and settled prequentially
(`currency.committed_settlement`, smoke test `python3 currency.py`). Hidden neurons and heads get no
wealth. The worker-1 lock (cp/lmsr books for every unit, 00:30Z) is kept at
`lead/superseded/CURRENCY.lock.v1-2026-09-28T0030Z`. `books_v2.py` (sha in the lock) stays as the
diagnostic book; `w3/test_books_v2.py` 7/7 pass. For M2, live grad × input is the importance
predictor, and it is reported as a predictor, never as wealth.

## (a) Sign relation, provable toy — `lead/toy_sign.py`, `lead/results/toy_summary.json`

Setup: one unit enters the logit linearly (z = rest + w·a). Gradient credit is C(w) = −w·L'(w) and
ablation value is V(w) = L(0) − L(w).
- Lemma: V ≥ C for all w, because L is convex (tangent line below the chord).
- At the optimum w* = β, C = 0 while V = I(X;Y) = 0.1109 nats.
- On (β, w0) = (1, 2.1786), C < 0 < V. All 23/23 grid points in that interval disagree in sign
  (`lead/results/toy_closed_form.csv`).
- Under SGD, accumulated credit = Δ|w|²/(2η) − (η/2)Σ|g|², so at stationarity credit drifts
  negative whether or not the unit is useful. The identity holds to 3.6e-13.

Simulation: 2 redundant useful units plus 1 noise unit, 20k steps, n = 6 redundant units
(`lead/results/toy_sim.csv`).
- Adam: credit −75.7 ± 1.9 (100% negative), LOO settlement +467.8 ± 6.0, ablation dBCE +0.024 ± 0.004.
- SGD: credit −50.5 ± 0.1, settlement +487.5 ± 7.6, ablation dBCE +0.026 ± 0.006.
- Noise unit: settlement −25 (SGD) and −37 to −41 (Adam).

So the absorbed unit loses gradient credit while its committed settlement earns.

## (b) Sign census on wealth-net checkpoints — `lead/sign_census.py`, `lead/results/sign_census_summary.csv`

Fraction of oracle-important units with negative credit. Important means the top 5% of neurons
(26 of 512) or the top 2 of 8 heads by ablation dBCE. The base rate over all units is in brackets.

| source / currency | neurons top 5% | heads top 2 |
|---|---|---|
| wealth-net final, all-time books | 0.256 ± 0.059 (0.384) | n/a |
| wealth-net final, decayed hl1000 / win2000 | 0.564 ± 0.059 (0.549 / 0.555) | n/a |
| w4 ckpts, all-time books | 0.144 ± 0.092 (0.305) | 0.625 ± 0.331 (0.906) |
| held-out, cp, tangent | 0.619 ± 0.087 (0.728) | 1.00 (1.00) |
| held-out, cp, lmsr | 0.462 ± 0.117 (0.619) | 0.25 ± 0.33 (0.71) |
| held-out, live grad × input | 0.381 ± 0.100 (0.485) | 0.208 ± 0.144 (0.615) |
| LOO settlement, train / held-out | 0.000 / 0.026 ± 0.022 | n/a |

Gradient books put 14-62% of important neurons at negative credit, below the base rate but far
from zero. LOO settlement puts about 0%. Decayed books are worse than chance (0.56 against a 0.55
base rate).

## (c) Three-rules settlement for attention/gates — `lead/three_rules.py`, `lead/analyze_three.py`, `lead/results/three_agg.csv`

Rules:
- LayerNorm denominators are detached, with Euler for linear nodes.
- S = q·k/√dh is split half/half.
- Softmax is treated as an LMSR market: values are paid at the prior 1/Q, and routing gets
  (a − π)·u along a 16-step integrated-gradients path.

Runs: 12 checkpoints × 3 seeds, settled with tangent and lmsr.

Results (tangent / lmsr):
- **Conservation:** max residual 1.8e-4 / 5.1e-4 of |R| at 16 steps, and 1.1e-5 / 3.2e-5 at 64
  steps. This is quadrature error only; the pre-registered stop rule (≤ 1e-3) passes.
- **Routing share:** routing carries 0.92 / 0.90 of |head wealth|.
- **Negative heads:** every head's routing is negative under tangent (overconfidence on held-out
  data). Under lmsr, 0.72 ± 0.14 of heads have negative routing.
- **Faithfulness to ablation (Spearman):** heads three −0.15 / 0.52 against cp −0.20 / 0.52 and
  live 0.61 ± 0.21. Neurons three 0.24 / 0.29 against cp 0.23 / 0.29 and live 0.39 ± 0.08.
- **Three vs cp:** rank agreement for neurons is 0.994 / 0.988 and for heads 0.97 / 0.97.
- **Lock-in:** rank stability 8k→30k is unchanged: heads three 0.45 / 0.65 against cp 0.60 / 0.64,
  with the oracle at 0.90 (`lead/results/three_lockin_*.csv`).
- **Cross-check:** my cp recompute equals w4's re-aggregated cp. Rank correlation is 1.000 for
  heads and neurons, and the scale ratio is 1.0000000 (`three_agg.csv`, cp_vs_w4fixcp_*).

Verdict: the three rules conserve and split heads into routing and value. They add no ranking,
faithfulness or lock-in over cp, so they are a diagnostic, not a currency.

## Worker items

| item | file | result |
|---|---|---|
| (a) wealth vs norm growth | w1/AUDIT-w1.md | books = −Q θ·∇L under frozen LN (rel err 3e-7). Under Adam, norm growth explains R² 0.19 ± 0.15 of neuron wealth |
| (b) SGD / SGD+wd | w2/AUDIT-w2.md | T = N + rent − (Qη/2)Σ\|g\|² exact (≤ 3e-6). Readouts: W = T. At η=3 the second-order term is 2.87 ± 0.12 × \|T\| for neurons. The readout wealth-to-gradient-energy Spearman is −0.89 ± 0.04. GPU $1.27, 0 VMs |
| (c) books_v2 + stop rule | w3/AUDIT-w3.md | in-sample vs held-out Spearman (out/head/neuron): tangent 0.59/−0.50/−0.19, lmsr 0.92/0.98/0.98. PRNG income flips sign |
| (d) leak split / AttnLRP | w4/AUDIT-w4.md | the v1 leak is QK 45% + AV 49%. cp is exact. Live grad × input tracks ablation best |

## Bug found and fixed

`w4/leak.py books()` divided the per-unit head and neuron arrays by B twice. R, the residuals and
closure were unaffected. I re-aggregated the data (`lead/w4_fix.py` → `lead/results/w4fix-*.json`)
without editing the worker files. Agreement with w4's raw files, minimum over 12 checkpoints × 4 rules:
- Spearman: heads 0.976, neurons 0.996.
- Sign agreement: heads 0.875, neurons 0.969.

w4's own `ablation_spearman.csv` and `rule_vs_rule.csv` still carry this weighting. The census in
(b) and the comparison in (c) use the re-aggregated values. The same bug was in `lead/three_rules.py`;
it is fixed there (chunk invariance to 4e-16) and every run was redone. `books_v2.py` is clean: it
takes gradients of z with per-example cotangents, then divides by B once.

## Limits and unverified

- Only 8 heads per net, so the top-2 head fractions are coarse.
- wealth-net oracle importances come from a subsample of eval rows.
- Held-out log-score gain against the base-rate prior is negative (−16.8 ± 1.9): the nets are
  overconfident on held-out data, which drives the tangent-settled heads negative.
- The lmsr in-sample vs held-out test (w3) ran under v1, not cp. The cp-vs-v1 rank agreement is
  0.976, so the result likely carries over (unverified).
- w1's init reconstruction is unverified.
