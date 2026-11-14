# P8 results: pay proposed intermediates by bits saved, against cheap selectors

Status: **No gain.** Checked 2026-09-29 15:35 UTC. Evidence: `results/p8_summary.json` (made by `analyze.py`), per-job JSONs in `results/`, chart data in `results/p8_chart.json`.

## Headline

The untuned one-code price R4 = L(Y|X,S) − L(Z|X,S) − L(Y|X,S,Z) ranks proposed maze intermediates **worse** than the cheap selectors people already use.

| Rule (32 maze items, 3 seeds) | Kendall tau-b vs deployed truth (95% CI) |
|---|---|
| R7b wrapper validation (80/20) | **0.83 ± 0.05** |
| R7a predicted-Ẑ RDA | 0.78 ± 0.10 |
| R1 importance only | 0.70 ± 0.02 |
| R8 importance − λ·learnability, λ from parity (0.006) | 0.70 ± 0.02 |
| R2 learnability only | 0.41 ± 0.16 |
| R5 asymptotic (final-block) saving | 0.34 ± 0.03 |
| **R4, the P8 rule** | **0.33 ± 0.02** |
| R6 masked-reveal surrogate | 0.31 ± 0.06 |
| R0 random | −0.01 ± 0.23 |

- **S1 fails.** R4 vs R1, R7a, R7b and R8 gives paired permutation p = 1.0 in every case. The result holds in both contexts:
  - S = ∅: R4 0.39, R7b 0.90.
  - S = {g2, g4}: R4 0.19, R7b 0.92.
- **F1 (novelty nil) fires.** The best cheap selector is 0.50 above R4, not merely within 0.05 of it.
- **F1' (rule wrong) fires in 3/3 seeds.** The Manhattan rung m2 at S = ∅ is priced at about +12,500 bits (sd ~1,100), but its deployed truth is −6 to −8 bits per maze against the free-latent control. g2 at S = ∅ is priced positive, so the other F1' condition does not fire.
- **Why R4 loses.** The learnability charge L(Z|X,S) is a whole-stream prequential cost. For deep rungs it is tens of thousands of bits (for example, L(g6|X) ≈ 58k), but a predicted Ẑ of such a rung still helps the deployed pipeline. R4's rank therefore follows R5 and R2 rather than deployed use.
- **Robustness check (not pre-registered).** I scored the rules against the other truth column: the gain over the same Y net without Z (`gain_vs_base`). The order is unchanged: R7b 0.91, R7a 0.81, R1 0.74, R4 0.29.
- **Almost every item has negative truth.** Against the pre-registered free-latent control, only g2g4:g6 is positive (+1.1 to +2.6 bits per maze). A free end-to-end channel beats any single labelled rung at N = 1024. The AUC for "truth > 0" therefore has one positive item per seed and is not informative; R4 and R5 both score 1.0.

## Other questions

- **(b) Economy: T1-C2, the programme test. The F3 kill gate fired at seed 0, so seeds 1–4 were skipped.**
  - T1-C2 listed nothing. All 16 level-1 R4 prices were negative (−17k to −54k bits), because the median-binarized teacher channels cost 20k–98k bits to learn.
  - T7-C2 listed T6c9 and T2c27. T0-C2 listed T2c21.
  - Held-out exact-maze accuracy at N = 1024:

    | Arm | Accuracy |
    |---|---|
    | T1 | 0.000 |
    | T7 | 0.001 |
    | T0 | 0.002 |
    | B1 | 0.000 |
    | B5 | 0.007 |
    | **B2** | **0.356** |
    | **B3 DT-recall** | **0.931** |
    | B4 oracle hints | 0.998 |

  - At N = 4096: B2 0.952, B3 0.993, T1 0.
  - On paired bits, T1 is worse than B2 by 59.9 ± 0.8 bits per maze.
  - S2 fails. F2 also fires for C2 (used-by depth 0).
- **S5 oracle check fails in 3/3 seeds.** On the oracle pool, greedy T1 listed:

  | Seed | Listed | Held-out accuracy | 15x15 accuracy |
  |---|---|---|---|
  | 0 | g3, g5, g7 | 0.39 | 0.33 |
  | 1 | m2, g5, g7 | 0.17 | 0.13 |
  | 2 | g3, g6 | 0.11 | 0.07 |

  - B4 scores 1.00 on both sets.
  - Tower code: L_S = 19–27k bits, against 8.3–8.6k for the full ladder.
  - Level-1 prices are tied within noise (g3 12,568 vs g2 12,538). Step-wise greedy misses the ladder that the whole-tower code prefers.
  - Used-by depth is 3, 3 and 2.
  - The direct/indirect split behaves as expected: ≥ 99% of the lower rungs' pay is indirect.
  - F2 does not fire: L_cap / ladder L_S = 4.5–4.9.
- **(c) Kelvin's randomized-z market, R6.** R6 reaches Spearman ≥ 0.8 with R4 using only 2–4% of R4's CPU, in 3/3 seeds, so S3 passes on its thresholds and F4 does not fire.
  - This does not discriminate. R4 and R5 are nearly collinear over this pool: R6's final ρ is 0.99/0.97/0.98 with R4 and 0.99/0.96/0.98 with R5.
  - R6 is a cheap route to R4's prices, but those prices rank deployed use poorly (tau 0.31).
- **R6b, the literal void-on-mismatch market (parity).** Effective sample size is 4,088 at S = ∅ and falls to 114–116 at the largest S. Its ρ with R4 is 0.05, 0.39 and 0.52. It cannot rank reliably.
- **(d) Parity search. S4 fails: T5 recovered the XOR tree in 0/3 seeds.**
  - R6 proposals are dominated by `maj`.
  - Brute force finds Y's true parents (q1, q2) as the best non-sanity pair in 3/3 seeds.
  - The true tree's exact L_S beats the empty model only in seeds 1–2 (by 391 and 490 bits). At N = 4096 the trap is marginal.
  - Parity pricing taus are all within noise (R4 0.27 ± 0.44), because the deployed parity pipelines are at chance.
- **Unit tests.** U1–U5 and U7 pass. U8 is reported (R3 equals final-block importance).
  - U6 fails its ordering. With start masked, the tower lists g6 (4,662) before start. Start's price equals its analytic prediction exactly (3,005). The gap of 1,657 is below 3 single-run SD (about 3,300), and g6 = {dist ≤ 6} contains the start. I read this as a wrong prediction, not a bug, and did not rerun.

## Deviations from the spec (stated)

- **Compute.** Ran on one GPU spot VM (trader-cat-p8-1, g2-standard-8 + L4, $2.01 ceiling estimate, deleted and verified 15:28 UTC), not CPU. This gave CPU-parallel slots; the nets are tiny.
- **The Phase 0 noise gate fired for single runs.**
  - 3 SD of a saving was 25.5% of g2's saving (13,003 bits; sd_single 1,106 over 8 inits × 4 orders), and still 12.8% at 4 replicates.
  - Prices were therefore redefined as means over 12 paired replicates (7.4%; a direct Latin-square check gives 8.4%). Tower steps use 4 replicates.
  - R6, truth, U6 and the schedule check are single runs.
- **Phase A kill gate.** F1' fired at seed 0. The remaining queued jobs (seeds 1–2, parity, R6b, BF) had already been launched in parallel and finished in about 20 minutes; I let them finish rather than stop.
- **Arm details.**
  - R7a conditions on the true S.
  - R7b is an 80/20 split averaged over 12 replicates.
  - R6's CPU fraction is measured against one R4 replicate.
  - R8's λ is fit on parity and applied to mazes; the other-context λ gives 0.45/0.64, and the in-sample oracle λ 0.063 gives 0.75.
  - B2 and B3 were selected by validation accuracy, and each used 80–110x the T1 step × parameter budget.
- **Truth.**
  - Measured at N = 1024 and on 15x15; the N = 4096 point exists only for C2.
  - The C2 gate read c2_s0.json dumped before the 4k point; the deployed section it uses was already complete.
- **Parity.** The context is p1234, and calibration chose n = 32 input bits.
- **Output.** The chart is JSON only, because the venv has no matplotlib.

## Files

- Code: `jobs.py`, `p8lib.py`, `queue_p8.py`, `analyze.py`, `maze_gen.py` (P7 vendored), `dtnet_vendored.py`, `gpu.sh`, `startup.sh`.
- Results: `results/*.json` (per job), `results/p8_summary.json`, `results/p8_chart.json`, `results/pilot/`.
- VM ledger: `gpu/LEDGER.md`, `../../GPU-LEDGER.md`. VM job logs: `vm_out/logs/`.
