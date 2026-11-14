# M9(b) pre-registration (worker 2), written 2026-09-28 before any analysis output

Item: "Confirm that wealth-net prices LFSR and CA110 at 60k, then plot FLOPs-to-Brier for the net
against the soup's virtual instructions (assumption: 1 VI = 1 op)."

Known before writing: wealth-net stopped at ~31.1k steps (time cap; RESULTS.md), so "at 60k" is
read as "at the final checkpoint"; no 60k checkpoint exists and wealth-net is read-only.
No new training runs. Pure re-analysis of existing logs (CPU, 1 job).

## Arms (all existing data, 3 seeds each)
- WN: wealth-net train-s{1,2,3} (multi-epoch, Adam 1e-3, B=128, held-out = last 20% of a 24k stream),
  held-out Brier per family every 100 steps (`ev_brier` in snapshots).
- P4, P16: S0 worker-1 prequential single-pass runs of the same Net on wealth-net streams
  (n_pos 64000, B=4/16 positions per step), trade-time logits in /tmp/traders-net/S0/w1/*.npz.
- ME: S0 multi-epoch stress (curves only as available).
- SOUP: logical-soup runs/phase3 cells k12-f8,k12-f10,k12-f12,k14-f8,k14-f10,k14-f12,k16-f12
  (3 seeds each), per-epoch market Brier per family, `cum_ops` (virtual instructions).
- Control: per-question logistic baseline (zb in the npz; `ev_brier_log` in snapshots).

## Metrics
1. "Priced" = family Brier <= 0.1 (base-rate trader ~0.25 on LFSR/CA110). Confirm claim holds iff
   WN final held-out Brier <= 0.1 for LFSR16 and CA110 in >= 2/3 seeds. Also report P4/P16 tail
   (positions 51200-64000) from S0's SUMMARY-w1.csv.
2. Compute-to-Brier: first cumulative compute at which the family Brier (net prequential: rolling
   mean over 50 positions; soup: rolling mean over 50 epochs; WN: held-out eval point) is <= 0.1;
   also thresholds 0.2 and 0.05. Never reached = censored at the run's total compute (">").
   Net compute = 3 x panel.net_forward_flops(B,Q=64,W=32) per step (fwd+bwd; books pass and
   baseline excluded). Soup compute = cum_ops (1 VI = 1 op; market/settlement/reproduction
   host work excluded).
3. Data-to-Brier: same, counted in outcomes observed per question (positions / epochs).
Report mean +- sd over seeds of log10 compute where all 3 seeds reach it, else k/3 reached.

## Stop rule
Analysis only; stop when the table and plot exist. If the soup reaches no threshold <= 0.1 on a
family, report the ratio as a lower bound only.
