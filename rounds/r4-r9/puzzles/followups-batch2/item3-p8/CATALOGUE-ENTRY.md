## FU-B2-3: With 4× the data (N = 4096) and a gain-over-base truth, does charging learnability (R4, or R8 with λ fixed at 0.0063) rank P8's 32 maze intermediates better than importance alone (R1)?

Status: Problem (a pre-registered gate failed; verified by an independent recomputation)

Setup: P8's maze task and 32 items, price stream and truth-train bank at N = 4096 (C2's baselines: B2 0.95, B3 0.99). Truth = gain over base of the deployed Y net (bits/maze), mean of 2 training replicates. Prices = mean of 8 prequential replicates. Primary: d = τ_b(R) − τ_b(R1) for R4 and R8, mean over 5 seeds (power calculation) with 97.5% t-CIs. Gates G1 (B2 ≥ 0.90), G2 (truth A/A ≥ 0.80), G3 (price noise ratio ≤ 0.10, random/null checks) enforced by the driver. PREREG frozen at 23:19Z on 09-29; CPU only.

Result: No verdict. Gate G3 on seed 2 gave a price noise ratio of 0.1058, above the 0.10 bar, and the run stopped itself at 01:48Z. Only seeds 0-1 pass every gate, against the minimum of 3.
- Gates:
  - G1 passed on all 5 seeds (B2 held-out accuracy 0.94-0.98).
  - G2 passed on all 5 (truth A/A Spearman 0.97-0.99).
  - G3 passed on seeds 0-1 (0.073, 0.057) and failed on seed 2. The failure is marginal: the frozen rule uses the sample sd (ddof = 1); with ddof = 0 it would be 0.099.
- Descriptive only (seeds 0-2, including the failed seed; not a verdict):
  - R4 − R1 = −0.515 [−0.611, −0.419] and R8 − R1 = −0.008 [−0.060, +0.044] (97.5% CIs).
  - τ_b is R1 0.770, R4 0.255 and R8 0.762.
  - R4 stays about 0.5 τ below R1 in each of the 3 priced seeds, as at N = 1024. On the 2 gate-passing seeds alone the CIs are wide (R4 − R1 −0.508 [−1.124, +0.108]).
  - Noisy-copy check: nz05 − g2 = −3.96 [−6.97, −0.95] bits/maze (the truth still favours the noisy copy of Y); on the 2 gate-passing seeds −3.59 [−16.7, +9.5], a tie.
  - Exploratory: P8's N = 1024 wrapper prices (R7b) rank the N = 4096 truth at τ 0.94 [0.89, 1.00]. R7b was not run at N = 4096.
- Deviation (DEVIATION-1): mid-run, the driver was swapped to price seeds 0-3 only and the deadline moved from 02:20Z to 02:35Z, because replicates took longer than the time model. It was decided before any price or truth number was read (the G1 results had been seen). It cut the planned seeds from 5 to 4 but changed no gate, metric, bar or analysis code.
- To get a verdict: rerun with KREP = 12 (P8's count). The observed ratios (0.057-0.106 at 8 replicates) scale to about 0.05-0.09 at 12.
- Cost $3.71 at the ceiling rate (3.09 VM-hours on c2d-highcpu-32 at $1.20/h). The VMs and disks are deleted, and gcloud confirms it.

Files: [followups-batch2/item3-p8/RESULTS.md](followups-batch2/item3-p8/RESULTS.md), `followups-batch2/item3-p8/{PREREG.md,DEVIATION-1.md,code/analyze3.py,vmout/GATE_FAILED.json,vmout/item3_summary.json,vmout/STATUS.txt}`, `GPU-LEDGER.md` (batch-2 follow-ups section)
