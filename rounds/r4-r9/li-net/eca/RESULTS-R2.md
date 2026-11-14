# E-CA round 2: a depth-bottlenecked net, an anchored pool, restricted-source keys and variance-gated VoI: results

eurisko study, 2026-10-15 (overnight, round 2). Written by an Opus 5.5 subagent for eurisko. Follow-ups proposed by gemini
after round 1 (`/home/agent/workspaces/mats/outbox/overnight-followups-round-1.md`, ranks 2 and 3), assigned by opus-mats.
Pre-registration: `PREREG-R2.md` (drafted 04:29-04:30Z; its pre-run save is timestamped 04:30:36Z, 15-23 s after the
five launch commands and before any job ran or any result existed; there is no earlier copy, so the audit marks the timing
check FAIL, M1; erratum in `PREREG-R2-ERRATA.md`, the file itself is kept as it was when the runs started). Round 1: `RESULTS.md`, `AUDIT.md`.
Round-2 audit: `AUDIT-R2.md` (all four verdicts re-derived; its findings H1, M1-M2 and L1-L5 are applied below, Sec. 9).
Numbers: `results/r2/summary.json` and `results/r2/tables.md`, re-derivable from `results/raw/r2a/` and `results/raw/r2b/`.

## 1. TL;DR

- **R2-D2a (depth gap with a working one-step law, 1 layer): FAIL for both rules.** One layer opens a large one-pass gap
  (validation KL rule 110: F 0.34, B1 0.45, B2 0.26; rule 90: F 0.10, B1 0.36, B2 0.049 nats) but also fails the one-step
  check (S1 KL 0.037 and 0.077 against a 0.025 bar, vs about 0.008 at 4 layers): the network no longer computes the local
  rule reliably.
- **R2-D2b (stiff-first search wins under the bottleneck): FAIL for both rules.** Best case, rule 110 under sw_mix at M = 16:
  stiffness-first chains cut KL by 24% on forward queries (bar 25%) and beat random order there; on bridges they cut 21-29%
  with CIs including 0 and no edge over random order. Under the plain mixture they gain nothing.
- **R2-D3a (key separation): PASS as pre-registered, but by construction.** The rebuilt key's cosine between forward and
  backward chains is -0.10 / 0.07 (B1 / B2; round 1: 0.95), redundant views keep 0.91, drift ratio 0.05. The audit showed the
  separation comes from restricting the fingerprint to the newly materialised cells, which are disjoint for the two chains:
  the same key over all revealed cells gives 0.73 / 0.81 (it would fail), and two disjoint same-direction chains also score
  about 0.08. So the key separates disjoint supports, not directions.
- **R2-D3b (variance-gated VoI): FAIL.** VoI over uncertain cells beats VoI over stiff cells on B1 (M = 8, mixture: 0.099 ->
  0.068, cut 0.31 [0.15, 0.47]) and beats one pass by 25-31% there, short of the 35% bar, and does not beat stiff-gated VoI
  on B2. Random order remains the best selector on bridges.
- **Secondary, an anchored mixture repairs round 1's pooling:** on the 4-layer rule-110 models, sw_mix (anchored at the
  one-pass belief) improves on one pass at M = 16 for every selector and family except VoI-uncertain on F (14 of 20 cuts
  with paired CIs above 0, bridges -20% to -46%; on F only confidence-first's CI is above 0), where round 1's pool made
  beliefs worse. The fitted stiffness exponent is 0 in 89 of 90 of these fits, so this is an anchored uniform mixture, not
  stiffness weighting. Added after the results were read; secondary.
- **Cost:** $2.88 for round 2 (5 L4 VMs; ceiling rate), $19.90 for the night so far; zero eca VMs left.

## 2. What ran

- **R2a (depth bottleneck):** the round-1 setup with a **1-layer** network (rules 110 and 90, plain masked training T1,
  3 seeds each), P2-Stiff sampled at tau = 5, chains as in round 1 (K = 16, M up to 16), plus the new **sw_mix** pool:
  mu = (lam P0 + sum_r w_r q_r) / (lam + sum_r w_r), w_r = beta_int(Y | S_M)^eta, anchored at the one-pass claim P0 and
  fitted (lam, eta) on validation outcomes. 2 L4 VMs (3 runs each). Training stopped on patience after 11-29 minutes.
- **R2b (keys and VoI, rule 110, 4 layers, T1, 3 seeds):** key head rebuilt without the target's own hidden state: a
  LayerNormed fingerprint of where the target attends, restricted to the cells a chain materialised (gemini's root cause 1:
  h_i dominated round 1's key 20,000:1 in variance). New selector **voiu**: VoI among the 6 most uncertain cells q(1 - q)
  instead of the 6 stiffest (`voi`, rerun on the same models). 3 L4 VMs (one run each).
- Checkpoints (`model.pt`) for all 9 runs are kept under `results/raw/r2a/*/` and `results/raw/r2b/*/` for later rounds.
- Cost: see Sec. 6.

## 3. R2a: a 1-layer network (depth bottleneck)

IMG: results/r2/kl-vs-M-r2a-r110.png

IMG: results/r2/kl-vs-M-r2a-r90.png

- **The gap is now large, but the one-step law is not learned either.** One-pass KL on validation: rule 110 F 0.34, B1 0.45,
  B2 0.26 nats; rule 90 F 0.10, B1 0.36, B2 0.049 (4-layer round 1: 3-seed means 0.006-0.080). But the one-step check S1 (the query's three
  parents known) is 0.037 (rule 110) and 0.077 (rule 90), above the 0.025 bar (4 layers: 3-seed means 0.008): one layer cannot even compute the local rule
  reliably, rule 90's XOR least of all. **R2-D2a FAILS for both rules** (on S1; the long-range half passes).
- **R2-D2b FAILS for both rules.** Rule 110, under sw_mix at M = 16: stiffness-first chains cut KL by 24% on F (bar 25%;
  they beat random order by 0.07 nats, CI above 0), 29% on B1 and 21% on B2 (CIs include 0, and random order does as well);
  under the plain mixture stiffness-first and confidence-first chains gain nothing (F +3%, B1 +7%, B2 -22%). Rule 90:
  stiffness-first and confidence-first chains gain at most 11% (CIs include 0); random order with the mixture gains 24% on B2
  (CI above 0) and nothing elsewhere. Random order is the best or tied-best selector on every
  bridge family, as in round 1.
- Reading (an interpretation): with a 1-layer network the conditionals that search chains through are themselves less
  accurate (S1 KL 0.04-0.08 vs about 0.008 at 4 layers), which limits what routing through them can gain. Stiffness still
  ranks pairs by information gain (Spearman 0.82 / 0.88, AUC 0.88 / 0.86), and on rule-110 F under sw_mix stiffness-first
  does beat random order (KL difference 0.066 [0.049, 0.082] at M = 16).
- The fitted sw_mix weights use stiffness mildly for rule 110 (eta = 0.5 in all 9 rule-110 fits at M = 16, lam 6-56) and
  little for rule 90 (eta = 0 in 6 of 9, 0.5-1 in the other 3).

## 4. R2b: keys without h_i, and VoI over uncertain cells (rule 110, 4 layers)

IMG: results/r2/kl-vs-M-r2b-r110.png

- **R2-D3a PASSES as pre-registered, but the pass is mostly by construction (audit H1).** cos(k_fwd, k_bwd) = -0.10 on B1
  and 0.07 on B2 (round 1: 0.95 and 0.95; bar <= 0.35), redundant views 0.91 (bar >= 0.85), drift ratio 0.052 at m = 5
  (0.149 at m = 2; bar < 0.20); every seed passes on its own. But the v2 key changed three things at once (no h_i, a
  fingerprint restricted to the newly materialised cells S minus E_0, a LayerNorm), and forward and backward chains
  materialise disjoint cells, so their restricted fingerprints have disjoint supports. The audit recomputed the same trained
  key over all revealed cells: 0.73 (B1) and 0.81 (B2), which would fail; and two disjoint chains in the SAME direction also
  score about 0.08. The key separates disjoint supports, not directions, and dropping h_i alone does not fix it. The
  redundant-view half is computed over all of S, so the two halves of the rule use different supports.
- The forward + backward pool beats both single chains on B1 (0.089 vs 0.118 / 0.127) and B2 (0.090 vs 0.120 / 0.137), but
  round 1's keys (cosine 0.95) gave the same ordering, so this is not an effect of the new keys (audit L1).
- **R2-D3b FAILS.** VoI over the 6 most uncertain cells is better than VoI over the 6 stiffest on B1 under the mixture
  (M = 8: 0.099 -> 0.068, cut 0.31 [0.15, 0.47], meeting the 30% bar) and beats one pass by 25% [12%, 38%] there, short of
  the 35% bar; under sw_mix it beats one pass by 31% [22%, 40%] but improves on stiff-gated VoI by only 19% (CI includes 0);
  on B2 it does not beat stiff-gated VoI (cuts 0.16 and 0.00). Random order (B1 0.047 at M = 8 under the mixture) is still
  better than either VoI.
- Note: the parameter-free attention-vector cosine between forward and backward chains is 0 by construction (their new
  cells are disjoint); the learned key inherits most of that separation.

## 5. The anchored pool (sw_mix), all selectors (secondary readout)

The round-1 stiffness-weighted pool made one-pass beliefs worse (rule-110 F 0.025 -> 0.21 at M = 0). The anchored pool
cannot do that (at M = 0 it equals P0), and on the 4-layer rule-110 models of R2b it turns every selector into a gain over
one pass at M = 16 on every family except VoI-uncertain on F (-2%); 14 of the 20 selector x family cuts have paired CIs
above 0 (on F only confidence-first's): e.g. F: stiffness-first -43% (0.025 -> 0.014, CI includes 0), random -33% (CI
includes 0); B1: random -46%, VoI-uncertain -32%, stiffness-first -24%; B2: random -40%, stiffness
-32%; R: -13% to -26%. Under the plain mixture most of these are losses (Sec. 10 table). This readout was added at 05:25Z
after the round-2 results were read, so it is secondary. On these 4-layer models the fitted stiffness exponent is 0 in 89
of 90 fits (lam 10-32 at M = 16, i.e. 38-66% of the weight on P0), so the gain comes from an anchored uniform mixture of
the chains, not from stiffness weighting. The round-1 pool also differed in being log-linear and anchored at the
network's prior rather than P0, so no readout isolates which difference matters most (audit M2).

## 6. Cost and clean-up

- Round 2: **$2.88** at the $0.87/h ceiling for 5 on-demand L4 VMs (round-2 cap $25); night total **$19.90**
  for E-CA (`GPU-LOG.md`, rebuilt by `gpu/ledger.py`). One R2a VM landed in us-west1-a after stockouts in us-central1.
- Live eca VMs at the final check: none. MATS cluster: not used.

## 7. Deviations

- R2b's selector set adds stiff and conf (cheap) to the voi/voiu/rand that the rules need; R2a's adds no VoI arm (not
  needed by R2-D2).
- Round-2 runs are T1 only, as specified; P2-Stiff tau = 5 in R2a only (R2b keeps tau = 1 for comparability with round 1).
- `ecanet.Net` now builds an extra (unused unless --key-mode v2) keyW2 layer; every shared parameter initialises
  identically, but the global RNG stream after initialisation (dropout masks) differs, so round-2 runs are new trainings.
- The sw_mix-vs-P0 table (Sec. 5) and its figure rows are a secondary readout added after the results were read.
- The rule-90 1-layer runs vary widely across seeds: seed 2 learned the one-step rule (S1 0.021, validation B1 0.05), seeds
  0-1 did not (S1 0.10-0.11, B1 0.51), so the rule-90 CIs are wide.

## 8. Files

`PREREG-R2.md`; `analyze_r2.py`; `results/r2/summary.json`, `results/r2/tables.md`, `results/r2/*.png`; raw per-run
files and checkpoints `results/raw/r2a/r{rule}-T1L1-s{seed}/` and `results/raw/r2b/r110-T1-s{seed}/` (`result.json`,
`test-estimates.csv`, `d1-pairs.csv`, `train-history.csv`, `model.pt`). Code: `ecanet.py` (v2 key), `ecarun.py` (`--layers`,
`--key-mode`, `--tau`, `--selectors`, `--save-model`, selector `voiu`, pool `sw`).

## 9. Audit (`AUDIT-R2.md`)

An independent pass (Opus 5.5, 05:26-05:56Z) re-derived all four verdicts and every table line from the raw files with its
own code (0 mismatches), confirmed that the code diff contains only the flagged features with training unchanged, that
sw_mix and the v2 key are fitted on outcomes only, that sw at M = 0 equals P0, and the GPU total. It found: the key
separation is by construction (H1, applied in Sec. 4 and the TL;DR); PREREG-R2's pre-run save came 15-23 s after the
launch commands (M1, stated in the header and in `PREREG-R2-ERRATA.md`; the 05:49Z header edit was reverted at 06:02Z); sw_mix on the 4-layer runs is an anchored uniform mixture (M2, applied in Sec. 5);
and five small wording items (L1-L5, applied in Sec. 3-7 and, for L4, in the TL;DR at 06:02Z). This section and the
corrections were written after the audit (05:49-06:02Z).

## 10. Full tables


### r2a-r110 (1 layer, T1, seeds [0, 1, 2])
R2-D2a gate (validation P0 KL): F 0.3390 [0.3091, 0.3690]; B1 0.4540 [0.2668, 0.6412]; B2 0.2594 [0.0513, 0.4674]; R 0.1000 [0.0367, 0.1634]; S1 0.0370 [0.0097, 0.0643] -> **FAIL** (3 of F/B1/B2 >= 0.05; S1 <= 0.025: False)
Test P0 KL (1,000 chain instances): F 0.3346 [0.3110, 0.3582]; B1 0.4464 [0.2556, 0.6372]; B2 0.2637 [0.0772, 0.4503]; R 0.1082 [0.0455, 0.1709]
R2-D2b (qualifying families, test P0 KL >= 0.05: ['F', 'B1', 'B2']) -> **FAIL**

| F | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.3238 [0.2259, 0.4217] | 0.034 [-0.208, 0.276] | FAIL | 0.0226 [-0.0145, 0.0597] | -0.0061 [-0.0941, 0.0819] | FAIL |
| stiff|sw | 0.2557 [0.1885, 0.3229] | 0.237 [0.087, 0.387] | FAIL | 0.0378 [0.0226, 0.0531] | 0.0658 [0.0494, 0.0822] | PASS |
| conf|mix | 0.3648 [0.1672, 0.5624] | -0.088 [-0.627, 0.451] | FAIL | 0.0118 [-0.0079, 0.0314] | -0.0471 [-0.2365, 0.1422] | FAIL |
| conf|sw | 0.2689 [0.1909, 0.3468] | 0.198 [0.019, 0.377] | FAIL | 0.0324 [0.0155, 0.0492] | 0.0526 [0.0140, 0.0913] | PASS |
Family F: **FAIL**

| B1 | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.4080 [0.3040, 0.5120] | 0.066 [-0.411, 0.543] | FAIL | -0.0390 [-0.1497, 0.0717] | -0.0631 [-0.3213, 0.1950] | FAIL |
| stiff|sw | 0.3140 [0.2322, 0.3959] | 0.288 [0.046, 0.530] | FAIL | -0.0105 [-0.0862, 0.0653] | 0.0080 [-0.1711, 0.1871] | FAIL |
| conf|mix | 0.4319 [0.2462, 0.6175] | 0.007 [-0.642, 0.657] | FAIL | -0.0490 [-0.2036, 0.1056] | -0.0870 [-0.4215, 0.2475] | FAIL |
| conf|sw | 0.3229 [0.2358, 0.4099] | 0.268 [0.030, 0.507] | FAIL | -0.0126 [-0.1018, 0.0767] | -0.0008 [-0.1736, 0.1720] | FAIL |
Family B1: **FAIL**

| B2 | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.2950 [0.1621, 0.4279] | -0.222 [-1.576, 1.133] | FAIL | -0.0458 [-0.2299, 0.1382] | -0.0795 [-0.3939, 0.2350] | FAIL |
| stiff|sw | 0.1995 [0.1693, 0.2298] | 0.208 [-0.252, 0.669] | FAIL | -0.0079 [-0.1032, 0.0874] | 0.0008 [-0.1358, 0.1374] | FAIL |
| conf|mix | 0.3197 [0.2031, 0.4363] | -0.318 [-1.704, 1.069] | FAIL | -0.0633 [-0.2521, 0.1256] | -0.1042 [-0.4024, 0.1940] | FAIL |
| conf|sw | 0.2052 [0.1514, 0.2590] | 0.194 [-0.178, 0.565] | FAIL | -0.0145 [-0.0971, 0.0680] | -0.0049 [-0.1179, 0.1082] | FAIL |
Family B2: **FAIL**

KL vs M (3-seed means):

| family | selector, pool | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff, mix | 0.3346 | 0.3315 | 0.3272 | 0.3135 | 0.3019 | 0.3238 | 0.3346 |
| F | stiff, sw | 0.3346 | 0.3315 | 0.3278 | 0.3156 | 0.2945 | 0.2557 | 0.3346 |
| F | conf, mix | 0.3346 | 0.3297 | 0.3242 | 0.3128 | 0.3128 | 0.3648 | 0.3346 |
| F | conf, sw | 0.3346 | 0.3299 | 0.3253 | 0.3153 | 0.3000 | 0.2689 | 0.3346 |
| F | rand, mix | 0.3346 | 0.3335 | 0.3328 | 0.3288 | 0.3245 | 0.3177 | 0.3346 |
| F | rand, sw | 0.3346 | 0.3315 | 0.3320 | 0.3301 | 0.3323 | 0.3215 | 0.3346 |
| B1 | stiff, mix | 0.4464 | 0.4218 | 0.4093 | 0.3988 | 0.4111 | 0.4080 | 0.4464 |
| B1 | stiff, sw | 0.4464 | 0.4214 | 0.4091 | 0.3899 | 0.3642 | 0.3140 | 0.4464 |
| B1 | conf, mix | 0.4464 | 0.4212 | 0.4055 | 0.3946 | 0.4212 | 0.4319 | 0.4464 |
| B1 | conf, sw | 0.4464 | 0.4208 | 0.4044 | 0.3886 | 0.3663 | 0.3229 | 0.4464 |
| B1 | rand, mix | 0.4464 | 0.4306 | 0.4192 | 0.3973 | 0.3722 | 0.3448 | 0.4464 |
| B1 | rand, sw | 0.4464 | 0.4213 | 0.4070 | 0.3881 | 0.3537 | 0.3220 | 0.4464 |
| B2 | stiff, mix | 0.2637 | 0.2604 | 0.2564 | 0.2563 | 0.2668 | 0.2950 | 0.2637 |
| B2 | stiff, sw | 0.2637 | 0.2601 | 0.2533 | 0.2481 | 0.2283 | 0.1995 | 0.2637 |
| B2 | conf, mix | 0.2637 | 0.2613 | 0.2585 | 0.2615 | 0.2842 | 0.3197 | 0.2637 |
| B2 | conf, sw | 0.2637 | 0.2606 | 0.2566 | 0.2501 | 0.2350 | 0.2052 | 0.2637 |
| B2 | rand, mix | 0.2637 | 0.2548 | 0.2480 | 0.2365 | 0.2209 | 0.2155 | 0.2637 |
| B2 | rand, sw | 0.2637 | 0.2551 | 0.2499 | 0.2379 | 0.2204 | 0.2003 | 0.2637 |
| R | stiff, mix | 0.1082 | 0.1080 | 0.1073 | 0.1047 | 0.1028 | 0.1095 | 0.1082 |
| R | stiff, sw | 0.1082 | 0.1082 | 0.1070 | 0.1027 | 0.0958 | 0.0833 | 0.1082 |
| R | conf, mix | 0.1082 | 0.1072 | 0.1064 | 0.1033 | 0.1072 | 0.1138 | 0.1082 |
| R | conf, sw | 0.1082 | 0.1082 | 0.1078 | 0.1007 | 0.0966 | 0.0838 | 0.1082 |
| R | rand, mix | 0.1082 | 0.1003 | 0.0952 | 0.0872 | 0.0809 | 0.0801 | 0.1082 |
| R | rand, sw | 0.1082 | 0.1007 | 0.0958 | 0.0881 | 0.0828 | 0.0775 | 0.1082 |

D1 at 1 layer: Spearman 0.8226 [0.7832, 0.8619], AUC 0.8802 [0.8466, 0.9139]

### r2a-r90 (1 layer, T1, seeds [0, 1, 2])
R2-D2a gate (validation P0 KL): F 0.1046 [-0.0808, 0.2901]; B1 0.3594 [-0.3072, 1.0259]; B2 0.0493 [0.0356, 0.0630]; R 0.0398 [-0.0136, 0.0932]; S1 0.0765 [-0.0431, 0.1961] -> **FAIL** (2 of F/B1/B2 >= 0.05; S1 <= 0.025: False)
Test P0 KL (1,000 chain instances): F 0.1023 [-0.0804, 0.2850]; B1 0.3610 [-0.3012, 1.0232]; B2 0.0582 [0.0346, 0.0817]; R 0.0365 [-0.0043, 0.0773]
R2-D2b (qualifying families, test P0 KL >= 0.05: ['F', 'B1', 'B2']) -> **FAIL**

| F | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.1349 [-0.0314, 0.3012] | -0.907 [-3.951, 2.138] | FAIL | -0.0006 [-0.0141, 0.0130] | -0.0099 [-0.0319, 0.0120] | FAIL |
| stiff|sw | 0.1045 [-0.0789, 0.2878] | -0.048 [-0.185, 0.090] | FAIL | -0.0032 [-0.0076, 0.0013] | -0.0018 [-0.0050, 0.0014] | FAIL |
| conf|mix | 0.1566 [-0.0355, 0.3487] | -1.221 [-4.797, 2.354] | FAIL | -0.0050 [-0.0184, 0.0084] | -0.0317 [-0.0365, -0.0269] | FAIL |
| conf|sw | 0.1070 [-0.0712, 0.2853] | -0.149 [-0.685, 0.386] | FAIL | -0.0046 [-0.0054, -0.0039] | -0.0044 [-0.0109, 0.0022] | FAIL |
Family F: **FAIL**

| B1 | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.3781 [-0.2794, 1.0355] | -0.142 [-0.618, 0.335] | FAIL | 0.0047 [-0.0385, 0.0479] | 0.0045 [-0.0182, 0.0271] | FAIL |
| stiff|sw | 0.3569 [-0.3316, 1.0454] | 0.099 [-0.346, 0.544] | FAIL | -0.0016 [-0.0084, 0.0052] | -0.0029 [-0.0077, 0.0020] | FAIL |
| conf|mix | 0.3811 [-0.2732, 1.0354] | -0.173 [-0.766, 0.420] | FAIL | 0.0036 [-0.0436, 0.0507] | 0.0014 [-0.0238, 0.0266] | FAIL |
| conf|sw | 0.3577 [-0.3283, 1.0437] | 0.087 [-0.306, 0.479] | FAIL | -0.0017 [-0.0098, 0.0063] | -0.0036 [-0.0114, 0.0042] | FAIL |
Family B1: **FAIL**

| B2 | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |
|---|---|---|---|---|---|---|
| stiff|mix | 0.0612 [0.0237, 0.0988] | -0.042 [-0.287, 0.203] | FAIL | -0.0102 [-0.0255, 0.0052] | -0.0169 [-0.0394, 0.0056] | FAIL |
| stiff|sw | 0.0540 [0.0237, 0.0843] | 0.078 [-0.092, 0.249] | FAIL | -0.0022 [-0.0086, 0.0042] | -0.0013 [-0.0117, 0.0092] | FAIL |
| conf|mix | 0.0626 [0.0156, 0.1095] | -0.060 [-0.445, 0.324] | FAIL | -0.0105 [-0.0282, 0.0073] | -0.0183 [-0.0445, 0.0079] | FAIL |
| conf|sw | 0.0544 [0.0224, 0.0864] | 0.072 [-0.110, 0.255] | FAIL | -0.0022 [-0.0102, 0.0059] | -0.0017 [-0.0143, 0.0110] | FAIL |
Family B2: **FAIL**

KL vs M (3-seed means):

| family | selector, pool | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff, mix | 0.1023 | 0.1037 | 0.1041 | 0.1075 | 0.1122 | 0.1349 | 0.1023 |
| F | stiff, sw | 0.1023 | 0.1028 | 0.1028 | 0.1053 | 0.1058 | 0.1045 | 0.1023 |
| F | conf, mix | 0.1023 | 0.1040 | 0.1045 | 0.1074 | 0.1166 | 0.1566 | 0.1023 |
| F | conf, sw | 0.1023 | 0.1032 | 0.1034 | 0.1042 | 0.1072 | 0.1070 | 0.1023 |
| F | rand, mix | 0.1023 | 0.1036 | 0.1047 | 0.1064 | 0.1116 | 0.1249 | 0.1023 |
| F | rand, sw | 0.1023 | 0.1029 | 0.1027 | 0.1025 | 0.1026 | 0.1027 | 0.1023 |
| B1 | stiff, mix | 0.3610 | 0.3593 | 0.3586 | 0.3596 | 0.3633 | 0.3781 | 0.3610 |
| B1 | stiff, sw | 0.3610 | 0.3594 | 0.3584 | 0.3580 | 0.3557 | 0.3569 | 0.3610 |
| B1 | conf, mix | 0.3610 | 0.3596 | 0.3607 | 0.3599 | 0.3644 | 0.3811 | 0.3610 |
| B1 | conf, sw | 0.3610 | 0.3594 | 0.3599 | 0.3579 | 0.3558 | 0.3577 | 0.3610 |
| B1 | rand, mix | 0.3610 | 0.3597 | 0.3586 | 0.3589 | 0.3680 | 0.3825 | 0.3610 |
| B1 | rand, sw | 0.3610 | 0.3567 | 0.3553 | 0.3545 | 0.3541 | 0.3541 | 0.3610 |
| B2 | stiff, mix | 0.0582 | 0.0569 | 0.0567 | 0.0561 | 0.0569 | 0.0612 | 0.0582 |
| B2 | stiff, sw | 0.0582 | 0.0574 | 0.0573 | 0.0566 | 0.0565 | 0.0540 | 0.0582 |
| B2 | conf, mix | 0.0582 | 0.0574 | 0.0573 | 0.0568 | 0.0572 | 0.0626 | 0.0582 |
| B2 | conf, sw | 0.0582 | 0.0575 | 0.0574 | 0.0569 | 0.0564 | 0.0544 | 0.0582 |
| B2 | rand, mix | 0.0582 | 0.0555 | 0.0535 | 0.0494 | 0.0467 | 0.0443 | 0.0582 |
| B2 | rand, sw | 0.0582 | 0.0566 | 0.0563 | 0.0548 | 0.0543 | 0.0528 | 0.0582 |
| R | stiff, mix | 0.0365 | 0.0364 | 0.0360 | 0.0361 | 0.0378 | 0.0403 | 0.0365 |
| R | stiff, sw | 0.0365 | 0.0363 | 0.0362 | 0.0360 | 0.0358 | 0.0318 | 0.0365 |
| R | conf, mix | 0.0365 | 0.0368 | 0.0362 | 0.0361 | 0.0372 | 0.0412 | 0.0365 |
| R | conf, sw | 0.0365 | 0.0368 | 0.0363 | 0.0360 | 0.0354 | 0.0321 | 0.0365 |
| R | rand, mix | 0.0365 | 0.0345 | 0.0340 | 0.0344 | 0.0359 | 0.0406 | 0.0365 |
| R | rand, sw | 0.0365 | 0.0376 | 0.0368 | 0.0350 | 0.0343 | 0.0346 | 0.0365 |

D1 at 1 layer: Spearman 0.8829 [0.8212, 0.9447], AUC 0.8554 [0.8137, 0.8971]

### r2b-r110 (4 layers, T1, v2 keys, seeds [0, 1, 2])
R2-D3a: cos(k_fwd, k_bwd) B1 -0.0957 [-0.7060, 0.5146], B2 0.0658 [-0.5689, 0.7005] (bar <= 0.35); redundant-view key cos 0.9077 [0.8021, 1.0134] (bar >= 0.85); drift ratio m=5 0.0517 [0.0293, 0.0741] (bar < 0.20), m=2 0.1494 [0.1036, 0.1951]; attention-vector cosines: fwd/bwd B1 0.0000 [0.0000, 0.0000], B2 0.0000 [0.0000, 0.0000] (disjoint supports, 0 by construction), redundant views 0.7625 [0.5485, 0.9765] -> **PASS**
D3b pools with v2 keys (KL, dd pool): B1|both 0.0892 [0.0530, 0.1255]; B1|fwd 0.1179 [0.0625, 0.1732]; B1|bwd 0.1267 [0.1128, 0.1406]; B2|both 0.0897 [0.0689, 0.1106]; B2|fwd 0.1204 [0.0955, 0.1452]; B2|bwd 0.1367 [0.1132, 0.1603]

R2-D3b (voiu vs voi at M = 8, same models) -> **FAIL**

| family, pool | KL P0 | KL voi M=8 | KL voiu M=8 | cut vs voi | cut vs P0 | rule | round-1 voi M=8 (other models) |
|---|---|---|---|---|---|---|---|
| B1|mix | 0.0902 [0.0680, 0.1123] | 0.0985 [0.0635, 0.1335] | 0.0676 [0.0492, 0.0860] | 0.310 [0.146, 0.474] | 0.250 [0.116, 0.384] | FAIL | 0.1058 [0.0925, 0.1192] |
| B1|sw | 0.0902 [0.0680, 0.1123] | 0.0763 [0.0667, 0.0860] | 0.0621 [0.0399, 0.0844] | 0.187 [-0.062, 0.435] | 0.313 [0.224, 0.402] | FAIL | - |
| B2|mix | 0.0632 [0.0314, 0.0950] | 0.0757 [0.0373, 0.1140] | 0.0620 [0.0464, 0.0775] | 0.164 [-0.159, 0.488] | 0.006 [-0.232, 0.244] | FAIL | 0.0745 [0.0447, 0.1043] |
| B2|sw | 0.0632 [0.0314, 0.0950] | 0.0488 [0.0225, 0.0750] | 0.0482 [0.0288, 0.0676] | 0.004 [-0.129, 0.137] | 0.233 [0.160, 0.307] | FAIL | - |

KL vs M (3-seed means):

| family | selector, pool | M=0 | M=1 | M=2 | M=4 | M=8 | M=16 | P0 |
|---|---|---|---|---|---|---|---|---|
| F | stiff, mix | 0.0251 | 0.0230 | 0.0217 | 0.0203 | 0.0230 | 0.0517 | 0.0251 |
| F | stiff, sw | 0.0251 | 0.0233 | 0.0224 | 0.0209 | 0.0172 | 0.0139 | 0.0251 |
| F | voi, mix | 0.0251 | 0.0193 | 0.0175 | 0.0234 | 0.0519 | 0.1002 | 0.0251 |
| F | voi, sw | 0.0251 | 0.0206 | 0.0181 | 0.0157 | 0.0138 | 0.0138 | 0.0251 |
| F | voiu, mix | 0.0251 | 0.0244 | 0.0322 | 0.0447 | 0.0584 | 0.0699 | 0.0251 |
| F | voiu, sw | 0.0251 | 0.0220 | 0.0221 | 0.0242 | 0.0251 | 0.0258 | 0.0251 |
| F | conf, mix | 0.0251 | 0.0233 | 0.0222 | 0.0214 | 0.0245 | 0.0544 | 0.0251 |
| F | conf, sw | 0.0251 | 0.0236 | 0.0230 | 0.0215 | 0.0176 | 0.0148 | 0.0251 |
| F | rand, mix | 0.0251 | 0.0184 | 0.0156 | 0.0175 | 0.0291 | 0.0484 | 0.0251 |
| F | rand, sw | 0.0251 | 0.0184 | 0.0155 | 0.0151 | 0.0148 | 0.0164 | 0.0251 |
| B1 | stiff, mix | 0.0902 | 0.0908 | 0.0899 | 0.0880 | 0.0885 | 0.0851 | 0.0902 |
| B1 | stiff, sw | 0.0902 | 0.0902 | 0.0891 | 0.0865 | 0.0811 | 0.0683 | 0.0902 |
| B1 | voi, mix | 0.0902 | 0.0887 | 0.0912 | 0.0967 | 0.0985 | 0.1078 | 0.0902 |
| B1 | voi, sw | 0.0902 | 0.0866 | 0.0850 | 0.0821 | 0.0763 | 0.0655 | 0.0902 |
| B1 | voiu, mix | 0.0902 | 0.0665 | 0.0656 | 0.0672 | 0.0676 | 0.0668 | 0.0902 |
| B1 | voiu, sw | 0.0902 | 0.0647 | 0.0643 | 0.0628 | 0.0621 | 0.0615 | 0.0902 |
| B1 | conf, mix | 0.0902 | 0.0910 | 0.0899 | 0.0891 | 0.0931 | 0.0994 | 0.0902 |
| B1 | conf, sw | 0.0902 | 0.0908 | 0.0891 | 0.0866 | 0.0823 | 0.0722 | 0.0902 |
| B1 | rand, mix | 0.0902 | 0.0751 | 0.0655 | 0.0552 | 0.0468 | 0.0454 | 0.0902 |
| B1 | rand, sw | 0.0902 | 0.0751 | 0.0657 | 0.0575 | 0.0523 | 0.0490 | 0.0902 |
| B2 | stiff, mix | 0.0632 | 0.0616 | 0.0605 | 0.0573 | 0.0561 | 0.0634 | 0.0632 |
| B2 | stiff, sw | 0.0632 | 0.0617 | 0.0598 | 0.0568 | 0.0507 | 0.0425 | 0.0632 |
| B2 | voi, mix | 0.0632 | 0.0611 | 0.0635 | 0.0692 | 0.0757 | 0.1180 | 0.0632 |
| B2 | voi, sw | 0.0632 | 0.0571 | 0.0551 | 0.0529 | 0.0488 | 0.0457 | 0.0632 |
| B2 | voiu, mix | 0.0632 | 0.0568 | 0.0570 | 0.0604 | 0.0620 | 0.0630 | 0.0632 |
| B2 | voiu, sw | 0.0632 | 0.0493 | 0.0483 | 0.0479 | 0.0482 | 0.0481 | 0.0632 |
| B2 | conf, mix | 0.0632 | 0.0621 | 0.0620 | 0.0590 | 0.0616 | 0.0768 | 0.0632 |
| B2 | conf, sw | 0.0632 | 0.0623 | 0.0609 | 0.0571 | 0.0512 | 0.0435 | 0.0632 |
| B2 | rand, mix | 0.0632 | 0.0519 | 0.0452 | 0.0394 | 0.0376 | 0.0412 | 0.0632 |
| B2 | rand, sw | 0.0632 | 0.0519 | 0.0453 | 0.0403 | 0.0379 | 0.0376 | 0.0632 |
| R | stiff, mix | 0.0174 | 0.0173 | 0.0177 | 0.0174 | 0.0199 | 0.0231 | 0.0174 |
| R | stiff, sw | 0.0174 | 0.0171 | 0.0171 | 0.0161 | 0.0154 | 0.0135 | 0.0174 |
| R | voi, mix | 0.0174 | 0.0190 | 0.0213 | 0.0235 | 0.0257 | 0.0346 | 0.0174 |
| R | voi, sw | 0.0174 | 0.0164 | 0.0163 | 0.0157 | 0.0145 | 0.0128 | 0.0174 |
| R | voiu, mix | 0.0174 | 0.0211 | 0.0224 | 0.0254 | 0.0285 | 0.0299 | 0.0174 |
| R | voiu, sw | 0.0174 | 0.0157 | 0.0148 | 0.0150 | 0.0150 | 0.0151 | 0.0174 |
| R | conf, mix | 0.0174 | 0.0173 | 0.0177 | 0.0185 | 0.0222 | 0.0286 | 0.0174 |
| R | conf, sw | 0.0174 | 0.0175 | 0.0170 | 0.0165 | 0.0155 | 0.0141 | 0.0174 |
| R | rand, mix | 0.0174 | 0.0155 | 0.0144 | 0.0146 | 0.0170 | 0.0212 | 0.0174 |
| R | rand, sw | 0.0174 | 0.0155 | 0.0144 | 0.0140 | 0.0133 | 0.0133 | 0.0174 |

### Secondary: each selector vs P0 at M = 16 under sw_mix and the plain mixture (cut = 1 - KL/KL(P0), 3-seed mean, paired 95% CI)

| run set | family | stiff sw | stiff mix | voi sw | voi mix | voiu sw | voiu mix | conf sw | conf mix | rand sw | rand mix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r2a-r110 | F | +0.24* | +0.03 | - | - | - | - | +0.20* | -0.09 | +0.04 | +0.05 |
| r2a-r110 | B1 | +0.29 | +0.07 | - | - | - | - | +0.27 | +0.01 | +0.29* | +0.23* |
| r2a-r110 | B2 | +0.21 | -0.22 | - | - | - | - | +0.19 | -0.32 | +0.24* | +0.19* |
| r2a-r110 | R | +0.22 | -0.05 | - | - | - | - | +0.22 | -0.09 | +0.28* | +0.26* |
| r2a-r90 | F | -0.05 | -0.91 | - | - | - | - | -0.15 | -1.22 | -0.01 | -0.50 |
| r2a-r90 | B1 | +0.10 | -0.14 | - | - | - | - | +0.09 | -0.17 | +0.13 | -0.12 |
| r2a-r90 | B2 | +0.08 | -0.04 | - | - | - | - | +0.07 | -0.06 | +0.09* | +0.24* |
| r2a-r90 | R | +0.11 | -0.20 | - | - | - | - | +0.10 | -0.23 | +0.03 | -0.15 |
| r2b-r110 | F | +0.43 | -1.09 | +0.43 | -3.07 | -0.02 | -1.85 | +0.40* | -1.21 | +0.33 | -0.99 |
| r2b-r110 | B1 | +0.24* | +0.05 | +0.27* | -0.20 | +0.32* | +0.26* | +0.20* | -0.10 | +0.46* | +0.50* |
| r2b-r110 | B2 | +0.32* | -0.02 | +0.27* | -0.91 | +0.24* | -0.01 | +0.30 | -0.23 | +0.40* | +0.33 |
| r2b-r110 | R | +0.23* | -0.32 | +0.26* | -1.01 | +0.13 | -0.73 | +0.19* | -0.64 | +0.24* | -0.21 |

(* = paired 95% CI of KL(P0) - KL(arm) above 0.)

