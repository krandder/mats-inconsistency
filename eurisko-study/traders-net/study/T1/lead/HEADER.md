# THEORY (T1): lemmas 1-7 for traders-net

Assembled 2026-09-27 from three parts written in parallel, then merged and checked by the T1 lead.
Each lemma is tagged theorem, known (cited) or conjecture, and each has a numerical check on an S0 toy.
- Part 1 (lemmas 1-3: books identity, exact LMSR = IG, trunk cost sharing and FLOP budgets): `THEORY-part1.md`, `w1/`
- Part 2 (lemmas 4-5: delayed settlement, finite-population LI): `THEORY-part2.md`, `w2/`
- Part 3 (lemmas 6-7: fund of funds, top-k router as broker): `THEORY-part3.md`, `w3/`

The part files are the sources. Below the lead summary, this file is their concatenation. To regenerate it:
`cd study/T1 && cat lead/HEADER.md THEORY-part1.md THEORY-part2.md THEORY-part3.md > THEORY.md`.
All paths are relative to `study/T1/`. No GPU was used (CPU only, torch 2.14 cpu), so the GPU cost was $0.

## Lead summary (2026-09-27 ~22:46Z)

**How to check this page.**
- `python3 lead/check_claims.py --out lead/claims_check.json` re-reads the output files and asserts the 57
  headline numbers below and in the parts. The last run passed 57 of 57 (`lead/claims_check.json`).
- `python3 lead/compare_rerun.py /tmp/traders-net/T1/rerun --out lead/rerun_compare.json` compares every numeric
  leaf of the committed outputs with a clean rerun of all scripts (see "Reproduction" below).

**The seven lemmas.** Tags use the parts' own labels; seeds are 3 unless stated, as mean ± sd.

| lemma | tag | what it says | key check (file) | verdict |
|---|---|---|---|---|
| 1 books identity | theorem (1.1 known Euler) | books income a·dR/da = −θ·∇L. Under SGD it is norm growth, under weight decay rent; at calibration its mean is 0 | Euler identity rel err 8.8e-16 (`w1/out/L1a.json`). SGD per-step identity 1.8e-13 (`w1/out/L1b.json`). Books vs true income R² 0.32 (SGD) and 0.14 (Adam) on neurons (`w1/out/L1b.json`). With λ=1e-2, Spearman(income, ‖θ‖²) 0.9993 (`w1/out/L1c.json`). In-sample books pay 100% of noise features; prequential books charge 97.5% of them (`w1/out/L1d.json`) | holds; the books are **not** the true income under Adam or LN |
| 2 exact LMSR = IG | known (2.1) + theorem (2.2-2.4) | exact settlement = log-score gain = tangent + KL; its IG split has a closed form; the mean from the base rate is I(X;Y) | payoff − tangent − KL max err 1.1e-14. Seed 1: E[exact] 0.2645 vs I 0.2656 nats. Tangent vs IG shares: relative L1 0.72; arrival-order sd/share 0.95 (`w1/out/L2.json`) | holds |
| 3 trunk cost sharing | theorem + known (Aumann-Shapley); 3.4 conjecture | Euler cost shares balance the budget; the equal split of the trunk subsidises crowding | concave-trunk marginal revenue covers 0.707 of cost (`w1/out/L3.json`). E8 fine-grained: 240,704 active FLOPs/token and 59,664 per routed token (`w1/out/flops_M7.json`). K1 break-even 5.7e-4 nats/FLOP (`w1/out/flops_K1.json`) | holds; 3.4 is untested |
| 4 delayed settlement | theorem (4.1-4.5); known (4.7 LI); 4.6 statement only | book at the trade price, mark in between, settle at reveal. Re-pricing at reveal books hindsight | telescoping 3.3e-14 (`w2/results/summary.json`). Hindsight gap on periodic 0.0051 (D=64) and 0.046 (D=512) at lr 0.05 (`w2/results/summary_delay.csv`). Stale/‖W_trade‖ −0.747 and −1.197 (`w2/results/summary.json`). Self-resolution pays KL: prng D=0 income 0.056 vs KL 0.055 (`w2/results/summary_delay.csv`) | holds. Prereg 4b fails; it passes only in the exploratory arm |
| 5 finite-population LI | theorem (5.1-5.5, some known); 5.6 conjecture | members are bounded by ln(1/ε); price readers need fixed-point clearing; delay costs (D+1); births bound E_C at about φ·n/ρ | member E 6.90739 ≤ 6.907755 (`w2/results/summary.json`). Reader at one-pass clearing 48.4 (seed 1). a-D64: 447.81 ≤ 449.00. E_C periodic 1239 → 472 with births, diag 4535 → 1065 (`w2/results/summary_li.csv`) | bounds hold; plateau fails on periodic (ratio 0.29) and passes on diag (0.18) |
| 6 fund of funds | theorem (6.1-6.3); nested Fixed Share conjecture | buy-and-hold hierarchy = flat Bayes mixture; constant rebalancing = flat CRP; only rebalancing adds power | DAG vs flat: 8.9e-16; CRP DAG vs flat CRP: 9.1e-13. Toy CRP +0.087 vs leaf −0.424 nats/step. On ca110, EG beats the best leaf by 0.0132 (`w3/out/summary.json`, `w3/out/L6_growth.csv`) | holds |
| 7 router as broker | theorem (7.1-7.6); 7.7 known | router gradient g_i(r_i − R) is zero-sum; top-1 and unnormalised gates lock in under censoring, top-2 renormalised explores; Exp3 shadow credit is unbiased; router = commission-free broker | closed form vs autograd 8.6e-16. Top-1 censored finds the best 6.7% of the time with regret 0.378; with Exp3, 52% and 0.023. Censored top-2 gradient rel err 0.43. Broker residual 2.6e-18 (`w3/out/summary.json`) | holds; the Exp3 arm misses its prereg prediction "near 1" |

**Deviations from preregistration and failures (all disclosed in the parts; collected here).**
- w2 4b failed as preregistered: the lr 0.05 learner is unstable for D ≥ 64. It passes only in the exploratory
  scaled-step arm (addendum in `w2/PREREG.md`, dated 22:17:59Z, before that arm was run).
- w2 5.4 plateau: fails on periodic (slope ratio 0.29 against < 0.2) and passes on diag (0.18) (`w2/results/summary_li.csv`).
- w2 P5.4-d on periodic holds only vacuously. Pattern-4's E is negative (slopes −0.114 → −0.116), so
  there is nothing to bound.
- w2 used 60 bisection iterations, not the preregistered 50 (stated in part 2).
- w3 C7e predicted Exp3 would find the best expert "near 1" of the time; it did so 52 ± 4% of the time. Its low
  regret (0.023 ± 0.002) shows the misses are near-ties, which the regret alone supports.
- w3 C7b: sigmoid gates are not zero-sum (|Σu| 3.3e-4 to 7.9e-4, on a scale of 4.4e-4 to 8.1e-4). The criterion was refined before the run.
- w3 C7d used 1000 × 256 draws, against the preregistered 2e5.
- w1 started from a copy of the wealth-net files before S0 existed. `model.py` and `streams.py` are byte-identical
  to `study/S0/substrate/` (sha 14128c2a… for streams.py), so no rerun was needed.
- Lead corrections (22:45Z) to part 2, section 4.3:
  - The lr 0.05 hindsight gap had been quoted with the scaled-step values; it is now 0.0051 ± 0.00006 and
    0.046 ± 0.0008.
  - "|gap| ≤ 1.5e-4 on lfsr16/prng" had to be widened to ≤ 4.9e-4 (lfsr16 at D=64).
  - The part 2 check time is ~22:37Z, not ~22:45Z.

**What M1, M5, M7 and K1 should take from this.**
- **M1 (currency).**
  - Settle exactly from a named prior, using the IG closed form with the 16-step midpoint rule.
  - Require prequential settlement in `CURRENCY.lock`: in-sample books pay memorisation (lemma 1.4).
  - Do not call books "true income" under Adam or LayerNorm (lemma 1.2).
  - Cache books at trade time under delay (4.3); self-resolved labels pay KL, not accuracy (4.5).
- **M5 (LI metrics).**
  - Report E_C both at the clearing price and as traded.
  - Use fixed-point clearing whenever price readers are present.
  - Under delay, use D+1 interleaved books.
- **M7 (MoE).**
  - Use cost convention (A), consumers pay, with (B) as the sensitivity arm.
  - The FLOP tables in `w1/out/flops_M7.json` come from `w1/costshare.py`.
  - The router is a broker (7.6).
  - Expect lock-in only for top-1 or non-renormalised gates.
  - Normalise expert wealth by routed-token count.
- **K1.**
  - Break-even prices are in `w1/out/flops_K1.json`.
  - Births must be targeted: uniform births from a big pool wait n/ρ steps.
  - Use the diag family as the cheap detector of one-pass clearing.

**Reproduction.**
- A clean copy of w1/w2/w3 was rerun under `/tmp/traders-net/T1/rerun` (single-thread CPU, same seeds).
- Every script was rerun except `check_L1 b` (11 min; its numbers are covered by `lead/check_claims.py`).
- `lead/rerun_compare.json` lists the per-file results. Result at 22:46Z: 23 output files and 33185 numeric values were compared; all are bit-identical (max relative difference 0, nothing missing).

**Not verified.**
- Items marked "(unverified)" in the parts' prior-art lists (recollections of papers not opened here).
- The "only if" direction of conjecture 5.6.
- Conjecture 3.4 (crowding) and the nested Fixed Share conjecture (part 3).
- 4.6 (LM next-token and RL), which is argument only and was not run.
- The LI toy learner is per-question logistic, not wealth-net.

