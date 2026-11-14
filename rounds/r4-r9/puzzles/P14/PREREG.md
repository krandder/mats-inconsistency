# P14 pre-registration (frozen before any eval seed runs)

Frozen 2026-09-29T22:03:46Z. The SHA256 of this file is in `PREREG.sha256` and in `vm/LEDGER.md` (line FREEZE). Nothing below changes after the freeze; later departures go in RESULTS.md under 'Departures after the freeze'.

The design is the P14 spec (S1-S5, F1-F6, the attribution family and the secondary statistics, unchanged). This file fixes the values tuning chose, the matched budget, the analysis details the spec left open, and every deviation from the spec found before the freeze.

## Frozen values

- Grace G = 500 steps (gate g7: median steps to half the asymptotic VoI of a newly planted g2 slot = 88, 82, 93 on seeds 100-102; asymptote about 31.5 nats/maze).
- Budget B = V's counted FLOPs over 6000 steps = 6000 x 321221376 = 1.9273e+12 FLOPs (matmul FLOPs from the analytic counter plus the element-wise flip-value adds; gate g5 checks the matmul count against torch's counter within 1.3%).
- 50%-flip fallback: not used (V pilot on the VM: 0.113 s/step on one thread; tuning runs under 28 concurrent jobs stay near 0.25 s/step; threshold 0.5).
- Steps to B (the FLOP multipliers; eval arms train until their counter reaches B, ORACLE and planted arms run 6000 steps):

| arm | counted FLOPs/step | steps to B | multiplier vs V |
|---|---|---|---|
| N0 | 7.581e+07 | 25423 | 4.24 |
| ORACLE | 4.197e+07 | 6000 | 1.00 |
| BP-L0-cm | 1.311e+08 | 14707 | 2.45 |
| BP-L1-cm | 1.311e+08 | 14707 | 2.45 |
| DisARM-L1-cm | 1.523e+08 | 12657 | 2.11 |
| N-sel | 2.437e+08 | 7909 | 1.32 |
| STG-L1-cm | 1.311e+08 | 14707 | 2.45 |
| BP | 1.311e+08 | 14707 | 2.45 |
| BP-D | 2.989e+08 | 6448 | 1.07 |
| BP-L1-D-cm | 2.989e+08 | 6448 | 1.07 |
| V-local | 2.799e+08 | 6886 | 1.15 |
| P-LOCO | 4.607e+08 | 6000 | 1.00 |
| V-W | 1.533e+08 | 12569 | 2.09 |
| V-noD | 1.533e+08 | 12569 | 2.09 |
| V-L1 | 1.533e+08 | 12569 | 2.09 |
| V | 3.212e+08 | 6001 | 1.00 |
| P-W | 1.739e+08 | 6000 | 1.00 |

- Tuned values (label-only rule on seeds 100-102, validation set 777; score = best live bit's probe log loss; constraint = greedy-bit output log loss within 0.02 of the no-penalty counterpart):

| arm | parameter | value | tuning score | output log loss | counterpart log loss | tolerance binds | bracket extensions | still at edge |
|---|---|---|---|---|---|---|---|---|
| STG | tau | 0.5 | 0.0074 | 0.0007 | - | never binds | 1 | - |
| V | fee | 1 | 0.0264 | 0.0240 | 0.0051 | never binds | 0 | - |
| BP-D | fee | 1 | 0.0129 | 0.0043 | 0.0012 | never binds | 0 | - |
| BP-L1-cm | lam | 0.01 | 0.0189 | 0.0001 | 0.0012 | yes | 0 | - |
| V-L1 | lam | 0.001 | 0.0273 | 0.0036 | 0.0051 | yes | 1 | - |
| BP-L0-cm | lam | 3 | 0.0022 | 0.0013 | 0.0012 | yes | 0 | - |
| DisARM-L1-cm | lam | 0.01 | 0.0416 | 0.0040 | 0.0022 | yes | 0 | - |
| BP-L1-D-cm | fee | 0.3 | 0.0142 | 0.0018 | 0.0012 | never binds | 1 | - |
| STG-L1-cm | lam | 0.000111 | 0.0066 | 0.0020 | 0.0007 | yes | 2 | low |

  Bracketing: when a choice sat at a grid edge, the grid was extended on that side in 3x steps (at most twice). Both extension values ran together; the selection admits the second only if the first was chosen, which gives the same result as extending one step at a time. 'still at edge' marks a choice at the edge of the grid it was chosen from after the extensions.
  Stage 2 (BP-L1-D-cm fee; STG-L1-cm lambda) ran for every BP-L1-cm lambda and STG tau value in parallel with stage 1; the selection uses only the runs at BP-L1-cm's lambda* and STG's tau*, as the spec's sequential order would.
  Shared settings: BP-L1-D-cm uses BP-L1-cm's lambda*; STG-L1-cm uses STG's tau*; V-W, V-local, N-sel, P-LOCO and P-W use V's c*.
- **BP\* = BP-L0-cm** (lowest tuning score among BP-L1-cm, BP-L0-cm, BP-L1-D-cm, DisARM-L1-cm, STG-L1-cm; ties to BP-L1-cm). Its secondary seeds are extended to 20.
- **Endpoint switch: FIRED** (BP\* SUFF1 on 3/3 tuning seeds with gap_1 < 0.005). S1 is therefore the one-sided log-rank test on FLOPs to first SUFF1 (12 checkpoints, censored at B).
- T-dist (seeds 100-101, joint-probe score): V fee=3, BP-D fee=0.0333, BP-L1-cm lam=0.01, BP-L0-cm lam=0.1
- Gap floor for a run with no live bit (entropy >= 0.01 on open cells): the walls-only probe gap on the eval set 4102, 0.5614 nats/cell (tuning: walls-only probe loss on 777, 0.5594). Such a run scores as carrying no information and fails SUFF1.
- L* (g2-oracle probe loss) = 0.00102 on 4102, 0.00104 on 777.

## Analysis details fixed now (the spec left them open)

- Primary family {S1, S2}, Holm at 0.05. S1: V vs BP*; S2: V vs BP-D; one-sided (V lower gap_1). Paired Wilcoxon on seed pairs is a sensitivity analysis only.
- Attribution family (Holm over 3), two-sided van Elteren tests (stratified Wilcoxon rank-sum, weights 1/(N+1), normal approximation with ties): estimator (LEG vs ST) stratified by listing in factorial A {BP, BP-D, V-noD, V}; listing (on vs off) stratified by estimator in A; penalty type (slot fee + listing vs L1) stratified by estimator in factorial B {BP-L1-cm, V-L1, BP-D, V}. Interactions are descriptive (difference of median differences).
- Medians get 10,000-draw percentile bootstrap 95% intervals; rates get Wilson intervals; rate differences get Newcombe hybrid-score intervals and one-sided Fisher exact tests (secondary, unadjusted); means get t intervals.
- S3: paired t upper 95% bound on the seed-paired difference of greedy-bit output log loss (V minus BP* on 9x9; V minus ORACLE on 21x21), each <= 0.02.
- Planted criteria (b) and (c): a planted slot counts as live while it is listed (the wall copy is constant on open cells, so the entropy-filtered live set would pass (c) trivially). The entropy-filtered reading is reported alongside.
- Planted Spearman: over the 6 planted slots, the final VoI EMA (P-LOCO) or wealth (P-W) against the slot's joint-probe contribution = Shapley value of the probe log-loss reduction over the 5 analytic codes (g2 and its duplicate share; the seed-specific random slot is assigned 0), median over seeds.
- Cross-seed AMI (arithmetic normalisation, exact expected MI) between the joint live-bit partitions of open cells on probe-test (300 mazes, seed 4102), median over seed pairs.
- Collapse: no slot live with bit entropy >= 0.01 at the end.
- F6: the point estimate of V's advantage over BP* (median gap_1 difference, or SUFF1 rate difference) is positive under greedy bits and <= 0 under 8 sampled bits.
- Code identity: NMI = I/sqrt(H(b)H(g)); the delisting-quality table uses I/min(H(b),H(g)).

## Deviations from the spec found before the freeze

1. Probe optimiser. The specified probe (400 full-batch Adam steps at 3e-3) failed gate g6: the exact g2 code reached only probe acc 0.976, loss 0.058. The probe is minibatch Adam, 2000 steps of 4096 open-cell rows, lr 1e-2 with cosine decay (g2 on 4102: loss 0.0010, acc 1.0).
2. Probe input centring. With the new probe, complements failed g6 (~g2 gap 0.043) because zero padding distinguishes a code from its complement at the grid border. Each bit channel is now mean-centred (probe-train mean) before zero padding; g6 then passes for all four sufficient codes and their complements (largest gap: ~ring 0.0099), and g1 and g0 stay insufficient (0.090, 0.253).
3. Gate g2 rescored. A reader fed g1 alone reaches 9x9 accuracy 0.983, above the spec's 0.97 bar (which came from g1's exact-rule accuracy, 0.92). The gate is rescored on the sufficiency criterion used everywhere else: its log loss is 0.079 above the g2-fed reader's (> 0.02), so g1 is insufficient. Both verdicts are reported.
4. FLOP accounting. The counter counts matmul FLOPs analytically plus element-wise flip-value adds; torch's counter over-counts grouped-conv backward, so g5 compares against a custom mapping (all arms within 1.3%).
5. Tuning constraint uses the arm's greedy-bit output log loss on 777 (greedy is the primary readout).
6. A tuning run with no live bit is scored by the walls-only probe loss on 777 (the spec did not say).
7. T-dist tunes V, BP-D, BP-L1-cm and their counterparts; if BP* is not BP-L1-cm, BP* is tuned on T-dist by the same rule on seeds 100-101 before its T-dist eval.
8. Speculative runs. To save wall time, stage 2 ran for every lambda / tau, and bracketing extensions for V (fee 0.1, 0.0333), V-L1, BP-L1-cm, STG and STG-L1-cm (lambda 3.33e-4, 1.11e-4; tau 0.167, 0.0556) were launched before the stage-1 choices were final. The selection above admits only the runs the sequential procedure would have used (column 'used' in results/hp_select_*.json); the unused runs are kept and not read. Two V extension runs were stopped once V's choice (fee 1, interior) was known.
9. Tuning VM trader-b3-p14-1 was spot-preempted at 20:52Z; the queue resumed on trader-b3-p14-1-r1 from the last verified sync (.partial checkpoints hold the model, optimiser, market, data stream and RNG state).
