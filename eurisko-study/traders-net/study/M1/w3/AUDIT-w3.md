# M1 item (c): books_v2 and the in-sample vs prequential audit (worker 3)

2026-09-28 00:45Z. Code `M1/books_v2.py`; tests `w3/test_books_v2.py` (7 pass: frozen-attention
conservation < 1e-8 relative, readout LMSR book = exact log-score gain, masks, purity). Runs `w3/run_w3.py`
(prereg `w3/PREREG-w3.md`, no deviations), summary `w3/analyze_w3.py` -> `w3/results/summary-w3.csv`
(mean, sd, n=3, per-seed values), per-seed rows `w3/results/rows-w3.csv`, cumulative books
`w3/results/{P_pre,P_post,P_frozen,P_random,ME_in,ME_ho}-s{1,2,3}.npz`, run logs `w3/results/run-s*.json`.
Model: wealth-net size (Q=64, d=64, L=2, H=4), Adam 1e-3, S0 streams with prng (no pi). Family split
is {prng} vs {rest} only. Every number below is `mean ± sd` over seeds 1-3 from `summary-w3.csv`.

**Books.** `tangent` c·(y − σ(z)) at trade price; `lmsr` c·(y − p̄), p̄ the 16-step midpoint average
price from the running base-rate prior ℓ0 (readout book = exact log-score gain; maker account
−ℓ0(y − p̄)); `meanbase` (a − ā)·g_tangent; `meanbase_lmsr` (a − ā)·g_lmsr. c = a·dz/da on the frozen-LN
surrogate (same as v1; see w4 for the leak).

## Stop-rule metric: Spearman of in-sample vs held-out cumulative wealth (ME, 40 epochs)
Memorisation is real: final prng BCE in-sample 0.582 ± 0.009 vs held-out 0.793 ± 0.010.

| book | out | head | neuron | ≥ 0.9 everywhere? |
|---|---|---|---|---|
| tangent | 0.59 ± 0.07 | −0.50 ± 0.25 | −0.19 ± 0.05 | no |
| meanbase | 0.56 ± 0.12 | −0.22 ± 0.20 | −0.17 ± 0.05 | no |
| lmsr | 0.92 ± 0.03 | 0.98 ± 0.01 | 0.98 ± 0.00 | yes (ranks) |
| meanbase_lmsr | 0.92 ± 0.03 | 0.94 ± 0.03 | 0.98 ± 0.00 | yes (ranks) |

## PRNG wealth sign (income per step, prng group)
| book, granularity | ME in-sample | ME held-out | P prequential |
|---|---|---|---|
| tangent, neuron | −0.004 ± 0.011 | −0.74 ± 0.07 | −0.115 ± 0.014 |
| lmsr, readout (= log-score gain) | +0.257 ± 0.017 | −0.287 ± 0.029 | −0.053 ± 0.005 |
| lmsr, neuron | +0.350 ± 0.039 | −0.387 ± 0.040 | −0.059 ± 0.007 |
| lmsr, neurons with prng income > 0 | 0.87 ± 0.02 | 0.26 ± 0.01 | |
| meanbase_lmsr, heads with prng income > 0 | 1.00 ± 0 | 0.00 ± 0 | |

Prequential prng income is negative for every book, granularity and seed (P arm and ME held-out).
In-sample exact-LMSR books pay memorisation (+0.26 per step, 87-100% of units positive); in-sample
tangent books show ~0 (a calibration residual, T1 lemma 1.4), so they hide it instead of paying it.

## Reading
1. Tangent and mean-baseline ranks from in-sample books are unrelated or anti-related to held-out
   ranks at heads and neurons: for those books, prequential settlement is required.
2. Exact-LMSR ranks pass the 0.9 rule because the learnable families' gain (rest 17.6 ± 0.8 per step
   in-sample, 15.5 ± 1.0 held-out) swamps prng (±0.3), but the prng sign flips. The rule on ranks alone
   would keep cheap books; the sign test says no. Recommendation for CURRENCY.lock: **prequential
   settlement everywhere; unit book = lmsr (or meanbase_lmsr) from the running base-rate prior**, with
   tangent kept as the zero-mean calibration diagnostic.
3. Books disagree: prequential Spearman lmsr vs tangent at neurons 0.27 ± 0.09, heads 0.67 ± 0.25;
   lmsr vs meanbase_lmsr neurons 0.86 ± 0.05. Tangent books vs accumulated |a·g|: neurons 0.09 ± 0.10;
   lmsr vs |a·g| 0.55 ± 0.06.
4. Controls (P arm): books vs random-init books ≈ 0 at neurons for all books (|ρ| ≤ 0.02). lmsr vs
   frozen-final-weights books 0.87 ± 0.05 (neurons): lmsr ranks mostly reflect final structure;
   tangent vs frozen 0.05 ± 0.17.
5. Checks (live attention): 16-step path vs closed form p̄ max |Δ| 3.3e-4 (P) to 5.3e-4 (ME); gain −
   readout LMSR book 0.022 ± 0.004 per step vs gain 10.9 ± 1.3 (0.2%). First-order conservation is
   not exact with live attention/LN: |residual|/|R| 0.35 ± 0.07 (lmsr, P); for tangent the ratio is
   4.1 ± 1.2 because R_tangent ≈ 0 near calibration (the ratio is meaningless there). Neuron lmsr
   income (15.3 ± 1.7) exceeds the readout gain (10.8 ± 1.3): over-credit by the leak w4 measures.
   Maker account −ℓ0(y − p̄) is small and positive (0.51 ± 0.04 per step, P).

**Limits.** Only {prng, rest} family split; one model size; Adam only (SGD is worker 2's item);
books use c = a·dz/da (first-order), not a hidden-unit IG path; not checked against ablation (M2).
