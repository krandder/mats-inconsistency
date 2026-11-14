# Track-E Exploratory Results Ready: `expshiphuntkv` (Unified Compute/Data/Memory Key-Query VoI & Confiability Attention Head)

- **Commit:** `e02a9e8` (runner commit `c9673ab`)
- **Runner:** `benchmarks/explore_shiphunt_unified_kv_confiability.py` (`sha256: 654e566788c6a91459baa55636b11cbaa2e518de1b2c021e82cbefb69ce98219`)
- **Results JSON:** `results/explore/expshiphuntkv/explore_shiphunt_unified_kv_confiability.json`
- **Summary README:** `results/explore/expshiphuntkv/README.md` (`EXPLORATORY` per `PROTOCOL.md` §10.1)
- **Hardware & Cost:** 3 seeds (`101, 102, 103`) on NVIDIA L4 GPU (`soup-gpu-expshiphuntkv-1020191745-31887`, `$0.17`, VM auto-deleted).
- **Exact Bayesian Oracle Verification (Rule 9.6):** Enumerated all $H=802$ valid hidden ship configurations on `ShipHunt-6x6`; `max_posterior_abs_diff = 2.66e-15`, `max_voi_abs_diff = 3.27e-13`.
- **Also completed in Track E today:**
  1. `results/explore/expaomppl/` (`4db68bb`, `$0.31`): Search-augmented Any-Order Perplexity across $B \in \{0, 4, 16, 32\}$ on Sudoku-Extreme dev & N-Queens dev.
  2. `results/explore/explowdatavar/` (`82f99d1`, `$0.91`): Low-data ($N \in \{32, 128, 512\}$) Two-Parameter Split ($\theta_{\text{real}}$ vs. $\theta_{\text{amp}}$) & Unified $\text{VoI} - H$ ordering on stiff coin flips.
  3. `results/explore/exptrustamp/` (`d76e851`, `$0.26`): Confiability $\theta_{\text{trust}}$ inside $\text{params}_{\text{real}}$ ("search into reality") vs. self-gamed inside $\text{params}_{\text{search}}$ + $\text{params}_{\text{amp}}$ distillation.
- **Note on `scripts/preflight_check.py` (`56e12f7`):** Hardened to pass `M0` (`exit=0`) and catch all `9/9` mutations (`M1`–`M7`, `M3b`, `M6b` all `exit=1`) in `/tmp/r28b-review/mut/run_mut.py`.
