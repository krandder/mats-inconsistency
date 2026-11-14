# E3 RESULTS: separate goal-blind forecaster vs shared-adapter forecaster under best-of-n and RL pressure

Opus 5.5 subagent for eurisko, 2026-10-03 ~17:40Z. Spec: ../../GOAL.md (E3); hawk: ../../hawks/E3.md (RESHAPE, adopted);
pre-registration and append-only deviations: PREREG.md. Fixed forecast format only (no FF row). All paths are relative to
this directory. Every number below is from `results/e3_tables.json` (row = arm, key as named) or the CSV twin
`results/e3_table.csv`, computed by `e3_analyze.py` (per run: `runs/sS/ARM/e3_metrics.json`) and `e3_aggregate.py`.
Values are mean ± sd over training seeds (P, RAPc, RAPf: 2 seeds s0, s1; M0g, B0s: 3 seeds s0-s2).

## Status in one paragraph

Ran at reduced scale. GPU: 5 training VMs (Qwen3-0.6B, ShipHunt "hard", 80 SFT steps, 35-61 RL steps set by the time
budget), $7.8 at the rate ceiling (GPU-LOG.md, lir-e3-* DELETE lines). The programme hit the $50 launch stop at 13:49Z, so
roles seed 2 was not trained and no eval VM ran. The 64-candidate evaluation was finished on CPU with fewer states (c100 pairs:
60 states; c000 references: 120 states; probe sets 30 to 200 states). Checkpoints are 0% and 100% of RL only (25% and 50% were
dropped). Main finding: at this scale the separate goal-blind forecaster (arm P) collapsed to a near-constant P(success)
= 0.650 (probe correlation with exact Bayes 0.026 ± 0.004), so the main claim could not be tested. The instrument does detect
exploitation when reward passes through a forecast. Its two pre-registered delta metrics miss it, however, and a post-hoc
reference-calibrated gap catches it.

## Table 1. Selection pressure: frozen proposer (warm start, c000), forecaster c000, best-of-n by subset estimator

| arm | gold n=1 | gold n=64 | oracle n=64 | raw gap n=64 | excess gap n=64 (pre-reg) | excess xfit n=64 (post-hoc) | ITP gold n=64 | probe Brier vs Bayes | probe corr(f, Bayes) | distinct plan sets /64 |
|---|---|---|---|---|---|---|---|---|---|---|
| P (separate, goal-blind) | 0.663 ± 0.012 | 0.668 ± 0.033 | 0.732 ± 0.006 | -0.018 ± 0.033 | 0.019 ± 0.019 | 0.022 ± 0.001 | 0.663 ± 0.012 | 0.131 ± 0.000 | 0.026 ± 0.004 | 15.0 ± 0.3 |
| M0g (shared adapter, goal visible) | 0.629 ± 0.046 | 0.666 ± 0.038 | 0.719 ± 0.025 | -0.053 ± 0.019 | 0.013 ± 0.027 | 0.020 ± 0.018 | 0.637 ± 0.052 | 0.099 ± 0.024 | 0.163 ± 0.064 | 16.7 ± 1.6 |
| B0s (fused B0, hardsyco) | 0.634 ± 0.065 | 0.676 ± 0.042 | 0.726 ± 0.026 | -0.026 ± 0.039 | -0.001 ± 0.006 | 0.000 ± 0.007 | 0.646 ± 0.055 | 0.118 ± 0.011 | 0.155 ± 0.091 | 16.0 ± 2.4 |

Keys: `c000:c000/{gold,oracle,gap,excess_gap}/n{1,64}`, `c000:c000/posthoc_excess_xfit/n64`, `c000:c000/itp_gold/n64`,
`probe_c000/{brier_vs_bayes,corr_f_bayes}`, `c000:c000/distinct_sets_per64`. RAPc and RAPf share P's warm start (same row).
Reading: the warm proposer is already good (gold at n=1 is 0.63-0.66), but the forecasters rank candidates weakly. P's
best-of-64 gains +0.005 over n=1 against an oracle gain of +0.069. The fused forecasters rank better (M0g +0.037,
B0s +0.042). Excess gap is small and positive (0.00-0.02): this is the tail miscalibration that remains after isotonic
recalibration. ITP (Huang et al. 2025) with β tuned on calibration states never beat BoN here (β picked = 0.005, the smallest
on the grid). The Smith-Winkler shrinkage gap at n=64 is -0.016 ± 0.027 (P), -0.041 ± 0.013 (M0g), -0.040 ± 0.016 (B0s)
(`c000:c000/sw_gap/n64`).

## Table 2. Learning pressure: proposer at 100% of RL vs the frozen-proposer null, n=64

| arm (reward) | forecaster | gold n=64 | Δgold vs c000 | raw gap | Δ raw gap | excess gap (pre-reg) | Δ excess (pre-reg) | ref-calibrated gap (iso from c000:c000, post-hoc) | distinct sets /64 | probe Brier vs Bayes (c100 forecaster) |
|---|---|---|---|---|---|---|---|---|---|---|
| P (env) | co-trained | 0.645 ± 0.023 | -0.023 ± 0.010 | 0.005 ± 0.023 | +0.023 ± 0.010 | 0.071 ± 0.023 | +0.051 ± 0.004 | - | 5.8 ± 0.7 | 0.122 ± 0.000 |
| P (env) | frozen c000 | 0.643 ± 0.021 | -0.025 ± 0.012 | 0.008 ± 0.021 | +0.025 ± 0.012 | 0.044 ± 0.008 | +0.025 ± 0.011 | 0.040 ± 0.002 | 5.8 ± 0.7 | 0.131 (c000) |
| RAPc (+ctl: reward = own forecast) | co-trained | 0.497 ± 0.111 | -0.171 ± 0.078 | -0.110 ± 0.022 | -0.092 ± 0.011 | -0.002 ± 0.006 | -0.021 ± 0.013 | - | 10.0 ± 9.8 | 0.035 ± 0.001 |
| RAPf (+ctl: reward = frozen forecast) | frozen | 0.395 ± 0.084 | -0.273 ± 0.051 | 0.255 ± 0.084 | +0.273 ± 0.051 | 0.021 ± 0.021 | +0.002 ± 0.040 | 0.301 ± 0.061 | 1.7 ± 0.9 | 0.131 ± 0.000 |
| M0g (env, shared, goal visible) | co-trained | 0.643 ± 0.026 | -0.023 ± 0.028 | -0.009 ± 0.147 | +0.045 ± 0.149 | 0.080 ± 0.085 | +0.067 ± 0.080 | - | 15.8 ± 7.2 | 0.107 ± 0.100 |
| B0s (+ctl: approval reward, fused) | co-trained | 0.644 ± 0.012 | -0.032 ± 0.045 | -0.023 ± 0.227 | +0.003 ± 0.265 | 0.065 ± 0.015 | +0.067 ± 0.019 | - | 10.9 ± 2.2 | 0.137 ± 0.105 |

Keys: `c100:{c100,c000}/{gold,gap,excess_gap}/n64`, `.../delta_{gold,gap,excess_gap}/n64`,
`c100:c000/excess_ref_gap/n64` (P), `c100:c100/excess_ref_gap/n64` (RAPf), `c100:c100/distinct_sets_per64`,
`probe_c100/brier_vs_bayes`. The gold deficit (oracle - gold) does not move: Δ = -0.024 ± 0.018 (P), +0.006 ± 0.014 (RAPc),
-0.037 ± 0.065 (RAPf), +0.008 ± 0.017 (M0g), +0.006 ± 0.028 (B0s) (`c100:c100/delta_gold_deficit/n64`), because RL also
narrows the candidate pool and lowers the oracle.

Forecast at commit during RL (prequential, `runs/sS/ARM/steps.jsonl`, first vs last 10 RL steps,
`train/fc_committed_signed_vs_realised/delta`): P -0.025 ± 0.066; RAPc -0.336 ± 0.023 (the forecaster corrects to
pessimism); RAPf +0.127 ± 0.091 (frozen forecaster over-optimistic on the drifted proposer); M0g +0.068 ± 0.048; B0s
+0.003 ± 0.097. B0s approval-reward signature at eval: the c100 forecaster's mean stated P(success) on the probe set is
0.747 and 0.750 in s0, s1 (c000: 0.631 ± 0.017). That is decile 7, exactly the approval threshold (claimed success ≥ 0.7
pays w_conf). In s2 it is 0.314 (`probe_c100/mean_f` per_seed). Forecast minus Bayes over all candidates goes from
-0.012/-0.040/+0.082 to +0.104/+0.096/-0.268 (`c{000,100}:c{000,100}/forecast_minus_bayes_all` per_seed).
All 14 training runs have mask_violations 0 (`runs/sS/ARM/summary.json`; `mask_violations` row).

Other required measurements:
- Realised check at n=1: realised minus Bayes is -0.068 ± 0.016 (P c000) (`c000:c000/n1_realised_minus_bayes`). Every arm shares
  this, because all arms use the same eval states. On those 120 states it is -0.023 ± 0.014, and on 1500 fresh states it is
  +0.004 ± 0.004 (one-off check, command output in this session). It is state-sampling noise, not an oracle bias.
- Duplicates: 15-17 distinct plan sets per 64 draws at c000, 1.7-16 at c100 (Tables 1-2). Dedup-n results are under
  `dedup_gold/n16` and `dedup_gap/n16`.
- Entropy strata (exogenous n=16, raw gap): low-entropy third -0.089 ± 0.037 and high-entropy third +0.053 ± 0.004 (P c000,
  `c000:c000/entropy_{low,high}/gap16`). The forecaster is pessimistic where the posterior is sharp and optimistic where it is
  diffuse, as a constant forecaster must be. Policy-chosen n: not measurable (K=3 fixed in the harness).
- SE of excess gap at n=64 per seed: 0.012 ± 0.003 (P c000, 60 test states) and 0.017 ± 0.002 (P c100, 30 test states)
  (`*/se_excess_gap_n64`). Pooled over 2 seeds this is about 0.009-0.012. The hawk's ≤ 0.01 target is met only for the c000
  references. These SEs omit calibration-fit noise (see PREREG 15:55Z).
- Gap vs probe Brier: `results/e3_gap_vs_brier.png`. Curves gold/oracle/excess/raw vs n: `results/e3_curves.png`.

## Pre-registered verdicts

- Instrument check (PREREG): it required RAPf Δ gold_deficit > 0.02 or Δ excess_gap > 0.02 in all seeds, or B0s
  commit-forecast inflation > 0.05 in ≥ 2/3 seeds. **Not met.** RAPf Δ excess per seed was -0.027 and +0.030; Δ deficit was
  -0.084 and +0.009 (`c100:c100/delta_*` per_seed). B0s inflation per seed was +0.049, -0.109, +0.069. By the pre-registered
  rule the main-arm result is therefore uninterpretable.
- Post-hoc (labelled): the exploitation is large but sits where the pre-registered metrics do not look. RAPf gold fell by
  0.273 ± 0.051, the proposer collapsed to 1.7 plan sets per 64, and the frozen forecaster's gap calibrated on c000 is
  0.301 ± 0.061. The own-pair isotonic fit absorbs the shift (excess 0.021) and the oracle collapses with the pool (deficit
  flat). B0s stated the approval threshold (0.75) in 2/3 seeds.
- Main arm P: Δ excess +0.051 ± 0.004 is above the 0.02 refute line, but with a constant forecaster this is mostly
  calibration-split noise (post-hoc cross-fit: 0.022 → 0.054, still +0.032). Δ gold is -0.023 ± 0.010, and the forecaster
  never left 0.650. **The claim is not tested: there was no informative forecaster to keep or lose tail calibration.**
- P vs M0g (separation): at c100 they are equal on gold (0.645 vs 0.643) and excess (0.071 ± 0.023 vs 0.080 ± 0.085). At warm
  start the shared adapter learned a more informative forecaster (corr 0.163 vs 0.026, Brier vs Bayes 0.099 vs 0.131). At
  this compute, separation cost forecaster quality, and it bought no measurable protection.

## The hawk's objection, answered

1. "Raw gap measures calibration, not exploitation": **conceded, with numbers.** RAPc's raw gap went *negative*
   (-0.110 ± 0.022) while its gold collapsed (Δ -0.171 ± 0.078). P's raw gap stays near 0 only because its constant
   forecast (0.650) happens to equal the warm proposer's mean Bayes value (0.663). Raw gap tells you nothing about decisions here.
2. "Fooling has no gradient path in the settled design": **supported.** With realised-outcome reward (P), gold moved -0.023
   and the frozen-forecaster reference gap was 0.040 ± 0.002. With reward passing through a forecast it collapsed: RAPf
   Δgold -0.273, RAPc -0.171. B0s on approval converged on stating the threshold in 2/3 seeds. Co-training the forecaster on
   realised outcomes (RAPc vs RAPf) fixed the forecaster: probe Brier vs Bayes fell to 0.035 ± 0.001 and the commit forecast
   went from +0.091 to -0.244. It did not save the decisions: gold was 0.497 vs 0.395.
3. Reshape items not executable here: no gating switch exists (separate passes; "all" not runnable); FCIN is canonical by
   construction (no raw-text channel); FF is not built; policy-chosen n does not exist; prompted-base RAP row not run;
   25%/50% checkpoints dropped for compute (PREREG deviations 13:10Z-15:55Z).

## Prior work named (URLs as fetched by the hawk 2026-10-03, hawks/E3.md sec. 2/7; Huang re-fetched by me, PDF read)
Gao, Schulman, Hilton 2022 https://arxiv.org/abs/2210.10760 ; Snell et al. 2024 https://arxiv.org/abs/2408.03314 ;
Huang et al. 2025 https://arxiv.org/abs/2503.21878 (Alg. 2-4 used for ITP; /tmp/lir-agent/e3/huang.txt) ;
Hao et al. 2023 (RAP) https://arxiv.org/abs/2305.14992 ; Yao et al. 2023 (ToT) https://arxiv.org/abs/2305.10601 ;
Smith & Winkler 2006 doi 10.1287/mnsc.1050.0451 (metadata verified by the hawk; content unverified).

## What would make E3 informative next time
The forecaster must be informative before RL. Options: more forecast-only warm-up, or a larger model. The target is a probe
correlation with Bayes well above 0.2. Evaluation needs one L4 per seed (about 1 h at 120 states). The headline metrics should
be gold and the frozen-reference calibrated gap; per-pair isotonic excess hides shift.
