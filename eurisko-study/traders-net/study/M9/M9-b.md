# M9(b): does the net price LFSR and CA110, and what does it cost against the soup? (worker 2)

2026-09-28 09:05Z. Prereg: `b/PREREG.md`. All numbers come from `b/final_brier_summary.csv`,
`b/compute_to_brier_summary.csv` (raw per seed: `b/*_raw.csv`) and the plot `b/flops_to_brier.png`.
Rerun: `PYTHONPATH=<wealth-net>/.pylib python3 b/analyze_b.py`. This is a re-analysis only; no new
training. Inputs were read without changes: `wealth-net/results/train-s{1,2,3}/snapshots`,
`/tmp/traders-net/S0/w1/P{4,16}-s{1,2,3}.npz` (S0 prequential trade-time logits) and
`logical-soup/runs/phase3/*/metrics.csv`.

## 1. Confirmation: the net prices LFSR16 and CA110 (at 31.1k steps, not 60k)
Wealth-net never reached 60k steps: the time cap stopped it at 31.1k-31.2k (`wealth-net/RESULTS.md`).
The numbers below are from the last checkpoint. Brier values are mean ± sd over 3 seeds; "priced"
means Brier ≤ 0.1.

| arm | LFSR16 | CA110 | logistic baseline LFSR / CA110 |
|---|---|---|---|
| wealth-net held-out (multi-epoch, B=128) | 0.0004 ± 0.0002 (3/3) | 0.0046 ± 0.0019 (3/3) | 0.252 / 0.132 |
| S0 prequential B=4 (tail 51.2k-64k) | 0.167 ± 0.145 (**1/3**) | 0.0040 ± 0.0019 (3/3) | 0.260 / 0.121 |
| S0 prequential B=16 | 0.035 ± 0.060 (2/3) | 0.0041 ± 0.0027 (3/3) | 0.255 / 0.119 |
| soup, all 7 phase-3 cells | 0.250-0.263 (0/21) | 0.249-0.256 (0/21) | n/a |

- **Confirmed for wealth-net:** 3/3 seeds on both families. The worst rolling-1000 value over the last 5k
  steps is ≤ 0.0063 (`brier_max_last5k` in `b/final_brier_raw.csv`), so the curves had plateaued and
  60k steps would not change the verdict.
- **Two caveats.** (i) Under prequential training LFSR is a seed-dependent phase transition: it was
  learnt in 1/3 seeds at B=4 and 2/3 at B=16, while CA110 was learnt in 3/3. (ii) S0 flagged that at
  T ≈ 64k, held-out LFSR contexts repeat across questions (`S0/SUBSTRATE-w1.md`). The prequential
  numbers are still honest (every label is scored before it is trained on), but part of the late LFSR
  success may be recall of the cycle rather than the recurrence.
- **Same information as the soup.** The soup's traders see a 32-outcome packed history
  (`logical-soup/src/questions.bend`, "ring of 1..32 outcomes") from the same generators
  (`wealth-net/streams.py` header). The soup's CA110 failure is therefore not a lack of information:
  a per-question *linear* model already gets CA110 to 0.12-0.13. LFSR needs a nonlinear XOR of the
  history, and the logistic model stays at 0.25 on it.

## 2. Compute and data to reach Brier 0.1 (log10 per question, mean ± sd, k/3 seeds)
Net compute = 3 × forward FLOPs (S0 `panel.net_forward_flops` formula) per training step, divided by
Q = 64. Soup compute = `cum_ops` (virtual instructions, 1 VI = 1 op), divided by its Q. Net and soup
curves are 50-outcome rolling prequential Brier; wealth-net uses its held-out eval every 100 steps.

| family | net P4 FLOPs/q | net P4 outcomes/q | best soup VI/q (cell) | soup outcomes/q | soup total VI/q (max) |
|---|---|---|---|---|---|
| periodic | 7.67 ± 0.06 (3/3) | 1.84 ± 0.06 | 7.31 ± 0.13 (k14-f8, 3/3) | 3.47 ± 0.13 | 7.54 |
| prime | 7.70 ± 0.04 (3/3) | 1.87 ± 0.03 | 6.17 ± 0.08 (k12-f8, 3/3) | 2.92 ± 0.10 | 7.54 |
| CA110 | 9.00 ± 0.04 (3/3) | 3.16 ± 0.04 | never (0/21) | never within 3.70 | 7.54 |
| LFSR16 | 10.03 (1/3); P16 10.53 ± 0.07 (3/3) | 4.19 (1/3); P16 4.68 ± 0.07 | never (0/21) | never within 3.70 | 7.54 |

Wealth-net (B=128) reaches 0.1 on periodic, prime and CA110 by its first eval (≤ 10^9.95 FLOPs/q),
and on LFSR at 10^10.83 ± 0.11. A "sustained" variant (a later amendment; the 500 outcomes after the
crossing must also average ≤ 0.1) gives the same net numbers. For the soup it drops prime in k14-f10
and k16-f12 to 0/3, and k12-f8 periodic to 1/3 (`sust_*` columns).

Reading:
1. **Compute gap on the recurrences: the net needs 1.5-3 decades more compute per question than the
   soup ever spent.** The net prices CA110 at 10^9.0 FLOPs/q and LFSR at 10^10.0-10.5. The largest soup
   budget is 10^7.54 VI/q (k14-f8). Even under the generous assumption 1 VI = 1 FLOP, no phase-3 soup
   cell came within 1.5 decades of the compute at which the net learnt CA110.
2. **Data gap on LFSR only.** The net learns CA110 after 10^3.16-3.45 outcomes per question (about
   1.4k-2.8k), which is inside the soup's 5000 epochs (10^3.70). It needs 10^4.2-4.7 outcomes for LFSR
   (15k-50k), 3-10 times more than any soup run observed. The soup's CA110 failure is a search and
   compute failure. Its LFSR failure is also data-limited: with this history and budget, the net would
   not have priced LFSR in 5000 epochs either.
3. **On the easy families the soup is cheaper per op.** The soup reaches 0.1 on primality at
   10^6.2 VI/q against the net's 10^7.7 FLOPs/q, and on periodic at 10^7.3 against 10^7.7. The net's
   figures there are floors set by its size: one prequential position already costs about 10^5.85
   FLOPs/q, and it crosses 0.1 within about 70 positions, while the soup needs 10^2.9-3.5 epochs.
   The net is data-efficient and compute-hungry; the soup is the reverse. A compute-matched comparison
   needs a smaller net, which is the M9(c) sweep territory.
4. **Wall clock (context, not a metric).** A full 64k-position P4 run took 246 ± 11 s on one CPU
   thread (`S0/w1/SUMMARY-w1.csv`, wall_s). A 5000-epoch soup run took 2.1k-11.8k s
   (`logical-soup/runs/phase3/summary.md`, wall_s). The soup's VIs are about 10^3-10^4 times more
   expensive in hardware time than the net's FLOPs (unverified ratio, computed from those two wall
   times and the op counts above), so the 1 VI = 1 op assumption flatters the soup.
5. **Wealth-net's multi-epoch arm overfits primality.** Held-out prime Brier rises again late in
   training in 2/3 seeds, to 0.099 and 0.104 (curve in `b/flops_to_brier.png`). The prequential arms
   stay at 0.065 ± 0.000.

## 3. Deviations and open points
- "At 60k" was replaced by the final checkpoint (31.1k), as the prereg stated beforehand.
- The sustained-crossing metric was added after the first output (amendment 1); the primary metric is
  unchanged and both are reported.
- The net's FLOPs exclude the books pass and the logistic baseline. The soup's VIs exclude host-side
  market, settlement and reproduction work.
- Not run: a prequential net at the soup's compute budget (10^6-10^7.5 FLOPs/q, a d ≈ 8-16 model).
  That would be the direct compute-matched test and belongs with M9(c).
