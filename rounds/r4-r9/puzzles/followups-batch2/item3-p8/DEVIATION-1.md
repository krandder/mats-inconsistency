# Deviation 1: scheduling change during the run (written 2026-09-30T00:23:56Z, before any price or truth number was read)

What was seen when this was decided: queue.log timings, G1 gate lines (all 5 PASS), and py-spy stack samples of the
running jobs (stage index and step counter only). No price file, truth file or G2/G3 gate line had been read.

Why: the pre-registered time model was wrong. py-spy at 00:17-00:20Z showed a price replicate runs 9 stages of about
2,000 steps at about 2.7-2.9 steps/s with 31 jobs on 32 vCPUs, i.e. about 115 min per replicate (PREREG assumed about
67). Under the frozen driver, seed 2's replicates r5-r7 could start only when the truth jobs ended (about 00:25-00:40Z)
and would end about 02:35Z, after the 02:20Z deadline, so only seeds 0-1 would be complete: status Problem. Seed 4's
prices cannot finish inside the $5 item cap in any schedule.

Change (driver2.py, sha256 9d0608d5e9fe761cde9f01cd39ece4ade78c823e24854d55a537aab9cb9b1282; frozen driver.py unchanged):
- The running driver is replaced by driver2, which adopts the jobs already running (no job is lost or relaunched).
- Prices and R7b only for seeds 0-3 (PRICE_SEEDS = 4). Seed 4's B2 and truth still run and its G1/G2 gates still apply.
- Workers 31 -> 34, so seed 2's r5-r7 start at once instead of waiting for truth slots (brief 2-job oversubscription).
- R7b replicates start only after every price merge (primary work first; R7b is secondary).
- Deadline 02:20Z -> 02:35:00Z (epoch 1790735700). Spend at the ceiling if it runs to the deadline plus 5 min for sync and
  delete: about $4.70, under the $5 item cap.
- Gates, bars, rules, truth, analysis code and the contingency (complete seeds in seed order, >= 3 or Problem) are
  unchanged. Power with 3 seeds is 0.37 and with 4 seeds 0.72 at sigma 0.041, delta 0.10 (power3.py).

Queue line: `driver2 $PY driver2.py 5 8 4 34 1790735700 4`.
