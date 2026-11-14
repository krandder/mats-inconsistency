# P13 results: market ResNet on mazes

**Question.** With the belief-logit runaway fixed, does the trader economy (market ResNet, ported from 1D prefix parity to
Bansal/Schwarzschild mazes, trained on 9x9 only) beat Bansal et al. 2022 DT-recall at 99x99 (11x) at equal training compute
and equal test FLOPs, and is any gain from the economics?

**Answer: no, and there is no gain to attribute.** Verdict: F1 holds (pre-registered falsifier).

- At 99x99 every Trader2D model evaluated there scores 0.00 exact (9 of 9: 4 ECON-fix seeds, 4 ECON-nocap seeds, the
  ECON-F0 pilot).
  DT-recall retrained at the same 3.0 L4-h scores 0.21 on average (seeds 0.00 / 0.83 / 0.00 / 0.00; 3 of 4 seeds trained).
  ECON-fix minus DTR-eq at 99, equal test FLOPs: -0.21, Welch 95% CI [-0.87, 0.45] (the CI includes 0 only because
  DT-recall's seeds vary so much; the hierarchical bootstrap CI is [-0.63, 0.00]). The released Bansal checkpoint (M-ref)
  scores 0.94 on the same 100 mazes.
- The Trader2D arms learn 9x9 (val 0.93-0.99 in 7 of 8 seeds) but do not extrapolate: 33x33 at most 0.04, 59x59 at most 0.02.
  DT-recall at equal compute is ahead in the mean at every size (e19 0.70 vs 0.24, e33 0.55 vs 0.015, e59 0.32 vs 0.005;
  Welch CIs include 0 because of DT-recall's seed spread, hierarchical-bootstrap CIs exclude 0 at 33 and 59).
- The economy never binds. In every ECON-fix seed the wealth cap never scales a trade (s < 1 on 0% of trade-pixels after the
  first 10% of training), there are 0 respawns, and the poorest trader ends with 503-559 from W0 = 20. The ECON-active ladder
  (W0 = 0.5; Lindahl trader-funded markets) found no setting that qualified (s < 1 on at least 10% of trade-pixels and
  v9 >= 0.9): W0 = 0.5 bound at 0.5 h (17% of trade-pixels, 104 respawns) while it had learned little (v9 0.30), and by
  1.0 h, with v9 0.86, it no longer bound (0%, no new respawns). The cap binds only while the network is still bad at the
  task; once predictions are good, the traders get rich. So S3 is "economics inactive, untestable", and ECON-fix
  and ECON-nocap (the same code with the cap off) are indistinguishable at every size (e19 0.23 vs 0.28 at t_eq, CI of the
  difference [-0.31, 0.22]; 0.00 vs 0.00 at 99). The economy reduces to local per-step losses.
- The runaway fix (F-LB, belief logits bounded at 10) works mechanically: max |b| at 99 is 10 instead of 3,634 for the
  unfixed ECON-F0 pilot, and the log-|b| growth slope drops from 0.82 to 0.07. It does not improve calibration: raw log loss
  at 99 is 2.16 nats per open pixel vs 2.20 for F0, and F0 after a post-hoc temperature (0.66) beats it (F2 holds: the runaway
  is a calibration problem that post-hoc methods solve). After post-hoc control, every Trader2D model is near chance
  (0.58-0.68 nats vs ln 2 = 0.69) at 99.
- BPTT-dscov (one seed, exploratory): full backprop through the same 30 steps with the same hints also learns 9x9 (v9 0.99) and does not extrapolate (v33 0.00, m59 0.00). So detaching gradients is not what stops the Trader2D arms from extrapolating; the objective and hints are. Its test-set evals were dropped by the $38 spend line.

Status for the programme: on mazes at this scale the economy is, at best, a valuation/control tool, not an extrapolation
mechanism (the sentence the PREREG says to add to THEORY.md if F1 holds; THEORY.md is outside this puzzle's directory, so
it is proposed here, not written there).

## Design in brief

Frozen in PREREG.md before any GPU spend; every later change is a timed addendum there.

- Training data: 9x9 mazes (P0 split). Test sets: e19, e33, e59, e99 (easy-to-hard generator; never used for selection);
  dev sets v9, v33, m13, m59 (used for pilot selection, J option and post-hoc controls).
- Compute: every trained arm gets B = 3.0 L4-h (ECON 9,900 steps, ECON-nocap 9,856, BPTT-dscov 11,762, DTR-eq 30,456).
  Test compute: exact accuracy at equal test FLOPs t_eq (primary) and at T = 12n.
- Arms: ECON-fix (4 seeds), ECON-nocap (4), DTR-eq (4; faithful retrain from P0 train.py), M-ref (released checkpoint,
  eval only), ECON-F0 (1.0 L4-h pilot, flagged), BPTT-dscov (1 seed, the one extra run; full backprop through D = 30 with the
  same hints).
- Fix selection: 7 candidate fixes x {ECON, ECON-nocap} piloted at 0.5 and 1.0 L4-h; F-LB (logit bound 10) selected by
  mean rank (tie rule used). Learnability gate passed (4 candidates reach v9 >= 0.9 and m13 >= 0.5 at 1.0 h).
- M-ref gate passed: 1.00 at 13, 1.00 at 59 (dev).
- Statistics: Welch t 95% CIs over seed means (primary, intent-to-treat), hierarchical bootstrap and exact permutation p
  alongside. "Trained" means v9 >= 0.95 and v33 >= 0.5.

## Results

Exact accuracy (fraction of mazes solved exactly), first 100 test mazes of each size, J option chosen on m59 dev.

| model | arm | J | v9 | v33 | e19 | e33 | e59 | e99 @12n | e99 log loss raw / after post-hoc control @t_eq | trained |
|---|---|---|---|---|---|---|---|---|---|---|
| m_dtr_s0 | DTR-eq | none | 1.00 | 0.51 | 0.95 | 0.55 | 0.07 | 0.00 | 2.40 / 0.54 | True |
| m_dtr_s1 | DTR-eq | none | 1.00 | 0.95 | 0.99 | 0.97 | 0.94 | 0.83 | 0.03 / 0.03 | True |
| m_dtr_s2 | DTR-eq | none | 0.99 | 0.61 | 0.85 | 0.69 | 0.26 | 0.00 | 0.16 / 0.12 | True |
| m_dtr_s3 | DTR-eq | none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.62 / 0.62 | False |
| m_econ_s0 | ECON-fix | none | 0.93 | 0.02 | 0.36 | 0.02 | 0.00 | 0.00 | 2.03 / 0.67 | False |
| m_econ_s1 | ECON-fix | none | 0.97 | 0.00 | 0.16 | 0.00 | 0.00 | 0.00 | 1.59 / 0.68 | False |
| m_econ_s2 | ECON-fix | none | 0.95 | 0.00 | 0.28 | 0.00 | 0.00 | 0.00 | 2.44 / 0.60 | False |
| m_econ_s3 | ECON-fix | none | 0.98 | 0.04 | 0.15 | 0.04 | 0.02 | 0.00 | 2.57 / 0.60 | False |
| m_nocap_s0 | ECON-nocap | scale | 0.74 | 0.01 | 0.02 | 0.04 | 0.00 | 0.00 | 1.34 / 0.58 | False |
| m_nocap_s1 | ECON-nocap | scale | 0.99 | 0.00 | 0.33 | 0.01 | 0.00 | 0.00 | 1.58 / 0.61 | False |
| m_nocap_s2 | ECON-nocap | scale | 0.96 | 0.03 | 0.39 | 0.03 | 0.00 | 0.00 | 1.33 / 0.60 | False |
| m_nocap_s3 | ECON-nocap | scale | 0.99 | 0.00 | 0.38 | 0.03 | 0.00 | 0.00 | 2.31 / 0.58 | False |
| m_bds_s0 | BPTT-dscov | none | 0.99 | 0.00 | - | - | - | - | - / - | False |
| mref | M-ref | none | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.94 | 0.03 / 0.05 | True |
| pil_econ_F0 | ECON-F0 | none | 0.82 | 0.02 | - | - | - | 0.00 | 2.20 / 0.66 | False |

Per arm (mean over seeds, 95% Welch CI; e19-e59 at 12n, e99 at equal test FLOPs):

| arm | n | e19 | e33 | e59 | e99 (mean, 95% Welch CI) |
|---|---|---|---|---|---|
| DTR-eq | 4 | 0.698 [-0.05, 1.44] | 0.552 [-0.10, 1.20] | 0.318 [-0.37, 1.00] | 0.207 [-0.45, 0.87] |
| ECON-fix | 4 | 0.238 [0.08, 0.40] | 0.015 [-0.02, 0.05] | 0.005 [-0.01, 0.02] | 0.000 [0.00, 0.00] |
| ECON-nocap | 4 | 0.280 [0.00, 0.56] | 0.028 [0.01, 0.05] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| BPTT-dscov | 1 | - | - | - | - |
| M-ref | 1 | 1.000 | 1.000 | 0.990 | 0.940 |
| ECON-F0 | 1 | - | - | - | 0.000 |

Comparisons (A minus B; e19-e59 at t_eq, e99 at t_eq and 12n):

| comparison | mean A | mean B | diff | Welch 95% CI | hier. bootstrap 95% CI | perm p |
|---|---|---|---|---|---|---|
| ECON-fix - DTR-eq, e19 @ t_eq | 0.233 | 0.698 | -0.465 | [-1.19, 0.26] | [-0.78, 0.01] | 0.14 |
| ECON-fix - DTR-eq, e33 @ t_eq | 0.015 | 0.552 | -0.537 | [-1.19, 0.11] | [-0.85, -0.15] | 0.14 |
| ECON-fix - DTR-eq, e59 @ t_eq | 0.005 | 0.318 | -0.312 | [-1.00, 0.37] | [-0.72, -0.02] | 0.14 |
| ECON-fix - DTR-eq, e99 @ t_eq | 0.000 | 0.207 | -0.207 | [-0.87, 0.45] | [-0.63, 0.00] | 1.00 |
| ECON-fix - DTR-eq, e99 @ 12n | 0.000 | 0.207 | -0.207 | [-0.87, 0.45] | [-0.62, 0.00] | 1.00 |
| ECON-fix - ECON-nocap, e19 @ t_eq | 0.233 | 0.280 | -0.048 | [-0.31, 0.22] | [-0.21, 0.14] | 0.71 |
| ECON-fix - ECON-nocap, e33 @ t_eq | 0.015 | 0.025 | -0.010 | [-0.04, 0.02] | [-0.04, 0.02] | 0.51 |
| ECON-fix - ECON-nocap, e59 @ t_eq | 0.005 | 0.003 | +0.003 | [-0.01, 0.02] | [-0.01, 0.02] | 1.00 |
| ECON-fix - ECON-nocap, e99 @ t_eq | 0.000 | 0.000 | +0.000 | [0.00, 0.00] | [0.00, 0.00] | 1.00 |
| ECON-fix - ECON-nocap, e99 @ 12n | 0.000 | 0.000 | +0.000 | [0.00, 0.00] | [0.00, 0.00] | 1.00 |
| ECON-fix - M-ref, e19 @ t_eq | 0.233 | 1.000 | -0.767 | [-0.92, -0.61] | [-0.86, -0.68] | - |
| ECON-fix - M-ref, e33 @ t_eq | 0.015 | 1.000 | -0.985 | [-1.02, -0.95] | [-1.00, -0.96] | - |
| ECON-fix - M-ref, e59 @ t_eq | 0.005 | 0.990 | -0.985 | [-1.00, -0.97] | [-1.00, -0.96] | - |
| ECON-fix - M-ref, e99 @ t_eq | 0.000 | 0.940 | -0.940 | [-0.94, -0.94] | [-0.98, -0.89] | - |
| ECON-fix - M-ref, e99 @ 12n | 0.000 | 0.940 | -0.940 | [-0.94, -0.94] | [-0.98, -0.89] | - |
| ECON-fix - DTR-eq(trained seeds), e99 @ t_eq | 0.000 | 0.277 | -0.277 | [-1.47, 0.91] | [-0.82, 0.00] | 0.43 |

DT-recall-eq on all 200 e99 mazes (the PREREG set): 0.010 / 0.825 / 0.005 / 0.000 by seed (mean 0.21); M-ref 0.91 at 12n
and 1.00 best-over-t up to 20n.

## Pre-registered tiers

| tier | result |
|---|---|
| S1 runaway fixed | not met. Only the post-hoc leg counts for F-LB: ECON-fix raw ll99 2.16 vs F0 after control 0.66. Label "slowed" (slope upper CI 0.35 > 0.1). F0 reference is the 1.0 L4-h pilot (flagged). F2 holds. |
| S2 beats Bansal at equal compute | not met. CI of ECON-fix - DTR-eq at 99 includes 0 ([-0.87, 0.45]; trained seeds only [-1.47, 0.91]); ECON-fix is 94 pp below M-ref; raw ll 2.16 vs DTR-eq 0.80. |
| **F1** | **holds** (CI includes 0, and ECON-fix is more than 2 pp below M-ref). |
| S3 gain is economics | economics inactive, untestable (cap never binds, no respawn, in every seed); the ladder found no active economy ("economics could not be made to bind"). F3 note: the economy reduces to local losses on mazes. |
| S3b / F4 detaching helps | not run (BPTT-dscov's test-set evals were dropped by the $38 spend line). On dev sets only: BPTT-dscov v33 0.00 vs ECON-nocap v33 0.00-0.03, which looks consistent with F4 (detaching does not hurt), but it is one seed and exploratory. |
| S4 beats M-ref outright | not met (ECON-fix 0.00 vs M-ref 0.94 at 99). |
| Gates | M-ref gate passed; DTR-eq trained 3 of 4 (at least 2, so "Bansal does not train reliably" is not declared). |

## Simplifications and changes (all timed in PREREG.md)

1. Test sets for the Trader2D arms: e19, e33, e59 and e99 use the first 100 mazes of each set (02:35Z spend plan);
   DT-recall-eq and M-ref ran on all mazes and are compared on the same first 100 (all-maze values in results.json).
2. No 201 stretch, no optional arms (DTR-pair, DTR-eqflop), no BPTT-progcov, no ECON-F0 main run, no ECON-active
   (the ladder found nothing to run), BPTT-dscov one seed instead of two (02:35Z cut order, then extended).
3. S1 compares ECON-fix with the 1.0 L4-h ECON-F0 pilot (one seed, one third of the budget), flagged.
4. No 20n extension for the Trader2D median-dev seed; Trader2D arms are scored at 12n (12.1n for J = cap).
5. ECON-active ladder: A2 (Lindahl bounty) ran the same 1,650 steps as A1, about 1.5 L4-h rather than 0.5 (it is about 3x
   slower per step); the 03:20Z rule extended a binding candidate to 1.0 h when none qualified at 0.5 h.
6. BPTT-dscov s0: its Phase-4 evals (e19-e99) were dropped by the $38 spend line after the 08:02Z spot wave, and its ev59
   keeps only J = none (09:08Z price check). S3b is therefore not run; BPTT-dscov is compared on dev sets only.
7. One job per L4 (k = 1; packing measured at 2.19x slowdown), 100 GB boot disks, spot VMs with resume from checkpoint after
   preemption (training checkpoints every 300 s, eval checkpoints every 300 s, sha256-verified sync every ~3-5 min).
8. Cost is estimated at the prescribed $0.35 per VM-hour, not billed (see Spend).

## Spend and teardown

- Estimated $37.52 of the $40 cap at the prescribed $0.35 per VM-hour: 107.2 VM-hours, 90 spot VMs created and 90 deleted, including several relaunches after spot preemption waves (07:03Z, 08:02-08:06Z).
- At the published g2-standard-4 spot price (about $0.424/h in us-central1, plus about $0.014/h for the disk) the same VM-hours cost about $47, above the cap. The Cloud Billing API is not enabled in the project, so no billed figure was read.
- Teardown: `gcloud compute instances list` and `gcloud compute disks list --filter='name~trader-b3-p13'` were empty at 10:30:19Z on 2026-09-30. The controller exited at 10:33:02Z and the watchdog was stopped at 10:33Z. No GPU resources are left.

## Files

- PREREG.md: the frozen plan plus every timed addendum (cuts, projections, ladder, price check).
- results/results.json and results/chart.json: main-phase results (analyze.py); results/results_gate.json and
  chart_gate.json: pilots, gate and DT-recall detail (analyze_gate.py); pilot_selection_*.json, ladder_selection_*.json.
- results/vm/<slot>/out/: synced eval summaries (.json), per-maze arrays (.npz), training logs and SHA256SUMS.remote.
- make_tables.py: prints the three result tables above from results/results.json.
- Training resume checkpoints (*.ckpt.pt) and DT-recall partial files were moved to the trash after the run to keep /srv small (final weights *.pt are kept); SHA256SUMS.remote still lists them.
- Code: econ_maze.py (Trader2D economy, ECON-nocap and BPTT modes), dtr_maze.py (DT-recall retrain), eval_maze.py
  (resumable evaluation), analyze.py, analyze_gate.py, pilot_select.py, ladder_select.py, tests_cpu.py (unit tests a-h,
  results/unit_tests/), gpu/ (controller, launcher, runner, queue planners).
