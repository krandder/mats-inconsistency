# K1 worker 2 PREREG: mid-run shift on/off + compute-price c sweep

Written 2026-09-28 ~02:05Z, before any run. Engine: `K1/econ.py` (worker1, ENGINE READY 01:59Z), imported read-only.
Driver: `K1/w2/w2run.py` (adds shift-tracking metrics through the `setup`/`extra_metrics` hooks and a
Stream wrapper; it does not change settlement). Outputs: `K1/runs/w2-<arm>-s<seed>/{metrics.csv,summary.json}`
(V1 wide schema), aggregate `K1/w2/results/*.csv|json`.

## Common settings
T = 20000 steps, batch B = 16, subsidy = 0.01 (maker stakes 1% of initial total wealth per step), stake 1.0,
split_at 4, die_at 0.01, engine defaults otherwise. Seeds 1, 2, 3. Every run carries the lockstep
beliefs-shuffled null (same stream, same population, beliefs permuted among traders on each question).
Compute price is given relative to break-even c* = subsidy x (pairs per trader per step)^-1:
niches on, c* = 0.01 x Q/B = 0.04; niches off, c* = 0.01/B = 6.25e-4 (above c* the sink exceeds the
maximum subsidy per trade on average). Baseline c/c* = 0.0025 (= worker1's default c = 1e-4 with niches on).

## Arms
Shift (shift_at = 10000, shift_frac = 0.25: a random quarter of questions redraw their family/table):
- S-on10k: N = 10k, Q = 64, niches on, shift on; S-off10k: same, shift off.
- S-on1k-nn: N = 1k, Q = 64, niches off, shift on; S-off1k-nn: same, shift off.
c sweep (no shift), c/c* in {0, 0.0025, 0.1, 0.3, 1, 3}:
- C-10k: N = 10k, Q = 64, niches on (c/c* = 0.0025 is S-off10k, reused).
- C-1k-nn: N = 1k, Q = 64, niches off (c/c* = 0.0025 is S-off1k-nn, reused).
Total 12 + 30 = 42 runs, less 6 reused = 36 runs; CPU only (about 5-10 min per run).

## Metrics (mean +- sd over 3 seeds)
- survivors: `alive` at the end; `eff_clades`, `top_clade_share` (monoculture = top clade >= 0.9 of wealth).
- Gini of trader wealth vs the shuffled-beliefs null: gini_traders - gini_null_traders at the end.
- market quality: tail Brier (last 10% of logs), regret = Brier - oracle Brier (true P), and the same for the null.
- shift only: per-window Brier regret on shifted questions minus on unshifted ones (excess regret);
  peak excess after the shift; recovery time = steps after the shift until excess regret stays <= 10% of the
  peak for 5 consecutive log windows; share of wealth in shifted niches held by traders born after the shift
  (niches on); Spearman of clade wealth at 9900 vs 20000 (lock-in). Shift-off arms get the same metrics on a
  placebo question set (same size, independent RNG).

## Predictions (written before running)
P1. Coexistence: with niches on, no arm with c/c* <= 0.3 ends in monoculture; eff_clades >> 1 and Gini above
    the null. Niches off (1k): closer to monoculture (top clade share larger than niches-on).
P2. Shift: excess regret spikes on shifted questions and recovers to <= 10% of the peak within 3000 steps with niches on;
    most post-shift wealth in shifted niches is held by traders born after the shift (> 50%).
P3. c sweep: survivors fall monotonically in c; at c/c* >= 1 the trader population collapses (alive < 10% of N)
    and Brier reverts toward the maker-only price; Brier is best at an intermediate c (0.1-0.3 c*) because the
    sink removes noise traders. Gini gap over the null grows with c until collapse.

## Stop rule
- K1 rule: monoculture (top_clade_share >= 0.9) in every one of my arms at the end -> report Blume-Easley and stop.
- Wall cap: 6 h from the first run; at most 3 of my jobs concurrently (fleet cap 12 CPU jobs, check `uptime`).
- A run whose ledger residual (cons_resid_ledger) exceeds 1e-6 is invalid and reported as such.

## Amendment 1 (02:05Z, before any run): CPU budget
K1 has no GPU cap (G1 caps.tsv: CPU only) and the box is loaded (load ~50 on 16 cores, ~12 study jobs), so I run
2 jobs at a time. Shift arms keep T = 20000. The c sweep uses T = 10000 (no shift); its c/c* = 0.0025 point is read
from the S-off runs' first 10000 steps (identical stream, population and dynamics up to t = 10000; the placebo
set uses a separate RNG), with the tail window = steps 9000-10000 in both.

## Amendment 2 (02:15Z, after the first two shift runs, before any c-sweep run): add c/c* in {0.01, 0.03}
Observed in w2-S-on10k-s1 (metrics.csv): the maker's realised loss (trader inflow, R_total) is only ~3% of the maker's
stake (66k of 2M staked over 20k steps), i.e. ~1.3e-3 per (trader, question) trade with niches on, so the nominal c*
(stake-based) overstates break-even by ~30x. The pari-mutuel maker loses only when the price overshoots the truth
(its expected P&L is sub*(1 - 0.5*(p_true/p + (1-p_true)/(1-p))), negative outside the interval between 0.5 and
p_true); the shuffled null, with worse prices, receives ~2.7x more inflow. To bracket the real transition I add
c/c* in {0.01, 0.03} for both populations (12 more runs, T = 10000). Predictions P3 unchanged, except that the
collapse is now predicted at c/c* ~ 0.03 (niches on) rather than 1.
