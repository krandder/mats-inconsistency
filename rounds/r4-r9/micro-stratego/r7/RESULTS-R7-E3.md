# RESULTS-R7-E3: 6x6 Stratego, pure salience keys, trust gating and active low-trust replacement on mixed-reliability bags (E-MS-TRUST-SEARCH)

Executor: an Opus 5.5 subagent of eurisko, 2026-10-17, for opus-mats (agent mail #3717). Spec:
`/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md`, section 2 (rules E3-D0 to E3-D2). Pre-registration:
`r7/PREREG-R7-E3.md`, sha256 `f3868522...` (`r7/PREREG-R7-E3.sha256`), hashed 2026-10-17T09:40:03Z, before the only
launch (09:40:26Z, `r7/gpu/launches.log`). A first hash at 09:39:45Z was superseded before launch because its header
wrongly said no amendments file existed (`r7/PREREG-R7-E3-AMENDMENTS.md`). The round-7 amendments file
(`outbox/experiments-round-7-amendments.md`, 09:24Z) changes R7-E2 and R7-E1 only; it was checked before launch and
before scoring. Paths below are relative to `/home/agent/workspaces/mats/work/micro-stratego/`. The R5-E2 primary
checkpoints were reused unchanged (no retraining; parameter hashes checked at the start and end of every seed's run).

## TL;DR

- **Pre-registered result: the gate E3-D0 PASSED, E3-D1 PASSED, E3-D2 FAILED (its ranking half passed, its pooling half
  failed).**
- **E3-D0 (gate) passes, reproducing R6-E3 exactly.** Sixteen marginal window rollouts cut KL to the exact posterior by
  48.2% [46.9, 49.5] against the 4-plies-old anchor (0.185 -> 0.096 nats per hidden slot; bar 40%). As in R6-E3, the
  net's own one-pass belief at the current turn is closer still (0.047).
- **E3-D1 passes, through the bag-centred key K3c, not through pure salience.** Pure salience keys K3 reach a
  within-minus-across-flank cosine gap of 0.234 [0.215, 0.254], no better than R6-E3's K4 (0.228) and below the 0.28
  bar. Bag-centred K3c reaches 0.661 [0.642, 0.679]. Most of that gap comes from centring: in a bag that leans 85/15 to
  one flank, centring pushes the minority flank's keys away from the majority's (across-flank cosine -0.133). With K3c
  keys, attn_L3's belief divergence between left- and right-preferring search is 10.8x lower than scaled_mix's and 10.2x
  lower than ataraxos_mmd's (bar 5x), and its KL to the exact posterior is lower than scaled_mix's, ataraxos_mmd's and
  kc_dedup's (all intervals > 0). The key head hardly matters for that: attn_L3 with K3 gives the same JSD and KL.
- **E3-D2 (1) passes: the new trust head ranks rollout errors above the pre-registered bars.** On mixed bags (8 clean + 8 exploratory rollouts per
  action), Spearman(beta, -KL(Q* || q_r)) is 0.347 [0.275, 0.420] overall and 0.369 [0.310, 0.429] within an action's
  16 rollouts. For stiffness -H it is -0.395 and -0.551. The trust-minus-stiffness gap is 0.743 overall (bar 0.55) and
  0.250 within entropy strata (bar 0.20).
- **E3-D2 (2) fails: no trust rule pools better than the plain mean.** All three trust rules cut action-value MSE by
  about 63% against fixed stiffness weighting (gamma 2; MSE 0.0388), but none beats the uniform mean (0.01366):
  trust_gate 0.01394 (difference -0.0003 [-0.0008, 0.0002]), trust_pool 0.01412 (-0.0005 [-0.0011, 0.0001]) and
  trust_resample 0.01431, significantly worse (-0.0006 [-0.0011, -0.0002]).
- **With the exploratory tier built here (cone types drawn at temperature 2.5 plus one type swap, plies at temperature
  2.0; prereg section 6 items 1-3), the exploratory rollouts were not hallucination traps.** The spec's other
  exploratory option (a cone drawn uniformly over hidden slots) and a clean tier at temperature 0.3 were not run. They were
  about as accurate as clean ones (squared value error 0.0476 vs 0.0453; KL 0.363 vs 0.361) and made FEWER low-entropy
  claims (9.4% vs 14.2% with H < 0.15). The trust head, trained on realised results, correctly gave the two tiers almost
  the same trust (beta 3.88 vs 3.95), so trust_resample pruned exploratory and clean rollouts about equally (50.4%
  exploratory). Even dropping the exploratory tier with perfect knowledge of the tier (clean8, a reference) is worse
  than keeping it (MSE 0.0155 vs 0.0137).
- **Post hoc:** terminal-like claims (H < 0.15) have 2.4-3.3x the squared error of the others, and trust ranks them
  low. A rollout with H >= 0.15 has the higher beta with probability 0.67 [0.59, 0.74], and terminal-like rollouts are
  21% of trust_resample's pruned rollouts against 12% of its initial batch (auditor's numbers from d2_rollouts.csv,
  `r7/AUDIT-R7-E3.md`). But most of the pooled error is bias that all rollouts share (about 0.0115 of the uniform
  pool's 0.0137, including about 0.0022 of noise in Q* itself). Unequal weights add about 0.0004 of noise by a
  fixed-weight estimate, about the size of trust_pool's extra MSE (0.0005 [-0.0001, 0.0011]).
- Cost: **$0.27** at the $0.87/h list-price ceiling (one on-demand L4, 0.31 h; ceiling $1.50). No r7e3 VM or disk left.

## Verdict table (from `r7/report/verdicts.csv`; 3 seeds; mean [95% seed-t interval, df = 2])

| rule | verdict | numbers (seeds 101, 102, 103; mean [95% CI]) |
|---|---|---|
| E3-D0 (0) anchor KL >= 0.12 | pass | 0.1859, 0.1856, 0.1841; 0.1852 [0.1828, 0.1876] |
| E3-D0 (1) battle accuracy >= 0.99 | pass | 0.9999, 1.0000, 0.9999; 0.9999 [0.9998, 1.0001] |
| E3-D0 (1) 7-known NLL <= 0.01 | pass | 0.0001, 0.0002, 0.0001; 0.0001 [0.0000, 0.0002] |
| E3-D0 (2) 16 marginal rollouts cut KL vs the t-4 anchor >= 40% | pass | 0.4805, 0.4779, 0.4880; 0.4821 [0.4690, 0.4952] |
| **E3-D0 gate** | **PASS** | one pass at t (reported): 0.0470, 0.0475, 0.0455 |
| E3-D1 (1) K3 key gap >= 0.28, CI low > 0.25 | fail | 0.2258, 0.2409, 0.2362; 0.2343 [0.2151, 0.2535] |
| E3-D1 (1) K3c key gap >= 0.28, CI low > 0.25 | pass | 0.6521, 0.6659, 0.6637; 0.6606 [0.6421, 0.6790] |
| E3-D1 (1) K3 or K3c | pass | key head for (2): K3c, the head that passes (1) |
| E3-D1 (2) JSD ratio scaled_mix / attn_L3 >= 5 | pass | 11.26, 8.32, 12.92; 10.83; difference 0.00211 [0.00186, 0.00236] |
| E3-D1 (2) JSD ratio ataraxos_mmd / attn_L3 >= 5 | pass | 9.97, 7.20, 13.31; 10.16; difference 0.00194 [0.00122, 0.00266] |
| E3-D1 (2) KL scaled_mix - attn_L3 > 0 | pass | 0.0071 [0.0064, 0.0078] |
| E3-D1 (2) KL ataraxos_mmd - attn_L3 > 0 | pass | 0.0206 [0.0138, 0.0273] |
| E3-D1 (2) KL kc_dedup - attn_L3 > 0 | pass | 0.0181 [0.0099, 0.0264] |
| **E3-D1** | **PASS** | (1) and all five of (2) |
| E3-D2 (1a) Spearman(beta, -KL) overall >= 0.30 | pass | 0.3276, 0.3807, 0.3337; 0.3473 [0.2751, 0.4195] |
| E3-D2 (1b) same within (s, a) pairs >= 0.25 | pass | 0.3617, 0.3959, 0.3498; 0.3691 [0.3096, 0.4287] |
| E3-D2 (1c) minus Spearman(-H, -KL) overall >= 0.55 | pass | 0.7380, 0.7801, 0.7093; 0.7425 [0.6541, 0.8309] |
| E3-D2 (1d) same within 10 equal-count entropy strata >= 0.20 | pass | 0.2323, 0.2628, 0.2540; 0.2497 [0.2107, 0.2887] |
| E3-D2 (2) trust_gate: MSE cut vs stiff_pool >= 20% | pass | 0.6235, 0.5976, 0.6968; 0.6393; MSE difference 0.0248 [0.0180, 0.0316] |
| E3-D2 (2) trust_gate: MSE below unif_pool | **fail** | -0.0005, -0.0002, -0.0002; -0.0003 [-0.0008, 0.0002] |
| E3-D2 (2) trust_pool: MSE cut vs stiff_pool >= 20% | pass | 0.6233, 0.5837, 0.6961; 0.6344; MSE difference 0.0246 [0.0172, 0.0320] |
| E3-D2 (2) trust_pool: MSE below unif_pool | **fail** | -0.0005, -0.0007, -0.0002; -0.0005 [-0.0011, 0.0001] |
| E3-D2 (2) trust_resample: MSE cut vs stiff_pool >= 20% | pass | 0.6156, 0.5831, 0.6902; 0.6296; MSE difference 0.0244 [0.0173, 0.0316] |
| E3-D2 (2) trust_resample: MSE below unif_pool | **fail** | -0.0008, -0.0007, -0.0004; -0.0006 [-0.0011, -0.0002] (worse) |
| **E3-D2** | **FAIL** | (1) passes; (2) fails: no trust rule beats unif_pool |

## What this means

1. **The gate is the R6-E3 gate, rerun.** Same function, same positions, same seeds: every gate number equals R6-E3's
   exactly (`r7/runs/s*/eval/d0.json` vs `r6/runs/s*/eval/d0.json`). The stale-anchor caveat from R6-E3 stands: the
   net's own current belief (KL 0.047 on the calibration positions, `d0.json` cal.kl_p0) is better than any
   window-updated stale belief (0.096 with 16 rollouts, 0.078 with every hidden slot scored).
2. **Pure salience did not fix the keys; centring did.** K3 keys are a linear map of the 8-slot salience alone. Their
   gap (0.234) is the same as K4's (0.228): within-flank cosine 0.869, across 0.635 (`r7/report/tables.md`). Presumably
   left- and right-flank cones often touch the same hidden slots, so raw salience vectors stay similar across flanks
   (not measured here). K3c
   adds the bag-centred salience. In an 85/15 bag the bag mean is close to the majority flank's salience, so the minority
   flank's centred vectors point away from it, and the across-flank cosine turns negative (-0.133). The within-flank
   cosine also falls (0.526), probably because the majority's centred vectors are small residuals. The rule asks for a gap and
   K3c delivers one, but it is a property of centring within a skewed bag, not evidence that the keys learned which
   rollouts are duplicates. The aggregators barely care: kc_dedup's KL is 0.1188 with K3 and 0.1196 with K3c, attn_L3's
   is 0.1016 and 0.1014 (`d1.json`).
3. **attn_L3 stays policy-independent and accurate, as in R6-E3.** With K3c keys its JSD between left- and
   right-preferring search is 0.00022, against 0.00233 for scaled_mix and 0.00216 for ataraxos_mmd; its KL to the exact
   posterior is 0.1014, against 0.1086 (scaled_mix), 0.1220 (ataraxos_mmd), 0.1196 (kc_dedup) and 0.1004 for the
   exact per-slot union call (`d1.json`). As in R6-E3, attn_L3 was trained toward the union-call reference, which the
   shared base cover makes policy-independent; it learned to imitate it.
4. **The trust head ranks errors, but pooling by it does not help.** Trained on realised calibration results with two
   consistency features and no tier label, the new head ranks rollouts by their KL to Q* with Spearman 0.347 overall and
   0.369 within an action's bag (`d2.json`). The R5-E2 head, used in R6-E3, reaches 0.213 and 0.225 on the same
   rollouts. Stiffness ranks them backwards (-0.395, -0.551): the most confident claims are the worst ones. But every
   trust rule pooled slightly worse than the plain mean. The bottom-tail gate kept the effective sample high (13.9 of
   16) and still lost 0.0003. Pruning and replacing lost 0.0006, a significant loss.
5. **Why: the "unreliable" tier was not unreliable, and the errors that matter are shared.** The exploratory rollouts
   (cone types drawn at temperature 2.5, one type swap, plies at temperature 2.0) were almost as accurate as clean ones
   (`d2.json` tier_clean / tier_explore). Tempering made them less confident, not more: fewer flag-capture endings
   (7.4% vs 11.7%), fewer claims with H < 0.15 (9.4% vs 14.2%). They did break the piece counts more often (3.7% vs
   0.02%) and played moves the policy head itself rates below 1e-3 (13% vs 0.4% of moves). On realised results their
   excess surprise was no higher than the clean tier's (1.057 vs 1.077, `trust.json`), so the trust head had no reason to
   distrust them. The rule's mechanism, trust pruning the hallucination traps of an exploratory tier, had nothing to act
   on here.
6. **Stiffness weighting is harmful, again.** Fixed (1 - H/log 3)^2 weights concentrate on confident rollouts
   (effective sample 8.0 of 16) and nearly triple the MSE (0.0388 vs 0.0137). Fitted on calibration results, the
   stiffness exponent was 0 (uniform weights) in every seed, as in R6-E3.

## Setup as run

- Checkpoints: the R5-E2 primary models (4 layers, d = 128, 1.0M parameters per seed) in
  `/tmp/r5e2/ckpt/s{101,102,103}/primary/`, unchanged; parameter hashes checked at the start and end (`eval/done.json`).
- Fitted in this run on the CALIBRATION split only (no oracle): scaled_mix, ataraxos_mmd, the K3 and K3c key heads (each
  jointly with its own kc_dedup plus the key-alignment loss), one attn_L3 per head (1,500 two-flank roots, model-only
  references, 1,500 Adam steps per fit, as in R6-E3); the new theta_trust (4,000 calibration roots x 16 rollouts of the
  move really played = 64,000 rollouts; target = excess surprise against the realised result; inputs = r5's trust
  features + cone_lp + ply_lp, no tier label); the pooling hyperparameters (1,000 further calibration roots, log loss
  against the realised result). Selected: alpha 0.5 in every seed; (q, tau_g) = (0.35, 0.5), (0.25, 0.5), (0.35, 0.5);
  stiffness gamma 0 in every seed (`eval/trust.json`).
- E3-D0: R6-E3's gate function, unchanged, on the same 1,000 calibration and test D0 positions.
- E3-D1: R6-E3's D1 protocol on the 500 two-flank test positions (4 shared base-cover rollouts + 20 rollouts skewed
  85/15 per bag).
- E3-D2: the 240 Q* test positions x every legal move = 2,361 (s, a) pairs per seed; per pair 16 clean and 8 exploratory
  rollouts (56,664 rollouts per seed). Mixed-reliability bag = clean 1-8 + exploratory 1-8.
- Compute: one on-demand L4 ran the three seeds in parallel (`r7/gpu/`; 796 s per seed); CSV/JSON-only sync back.

## Descriptive arms and diagnostics (`r7/report/tables.md`, `r7/runs/s*/eval/d2.json`; not rules)

| quantity (3-seed mean [95% CI]) | value |
|---|---|
| MSE unif_pool (mixed bag, 16) | 0.01366 [0.00972, 0.01760] |
| MSE stiff_pool, gamma 2 | 0.03875 [0.03514, 0.04235] |
| MSE trust_gate / trust_pool / trust_resample | 0.01394 / 0.01412 / 0.01431 |
| MSE trust_pool with the R5-E2 head (R6-E3's arm) | 0.01761 [0.01270, 0.02252] |
| MSE clean8 (tier label used: reference only) / explore8 | 0.01549 / 0.01862 |
| MSE clean16 (homogeneous clean bag) | 0.01328 [0.00932, 0.01723] |
| MSE trust_trim12 on clean16, new head / R5-E2 head | 0.01503 / 0.01485 |
| MSE trust_pool on clean16 (alpha 0.5) | 0.01370 |
| effective sample: trust_pool / trust_gate / stiff_pool (of 16) | 13.87 / 13.86 / 8.05 |
| share of exploratory rollouts among trust_resample's 4 pruned | 0.504 [0.481, 0.526] |
| delusional moves (R6-E3 definition) | 0 for every rule except stiff_pool: 1 of 240 positions in seeds 102 and 103 |
| per-rollout KL to Q*: clean / exploratory | 0.361 / 0.363 |
| per-rollout squared value error: clean / exploratory | 0.0453 / 0.0476 |
| share with H < 0.15: clean / exploratory | 0.142 / 0.094 |
| flag-capture endings: clean / exploratory | 0.117 / 0.074 |
| piece-count violations: clean / exploratory | 0.0002 / 0.037 |
| moves with untempered probability < 1e-3: clean / exploratory | 0.004 / 0.130 per move |
| cone_lp / ply_lp: clean | -1.01 / -2.19 |
| cone_lp / ply_lp: exploratory | -3.53 / -3.68 |
| mean beta (new head): clean / exploratory | 3.95 / 3.88 |
| Spearman(beta, -KL), R5-E2 head: overall / within pairs | 0.213 / 0.225 |
| calibration log loss: unif / trust_pool / trust_gate / trust_resample / stiff (gamma 2) / clean8 | 0.6267 / 0.6253 / 0.6261 / 0.6280 / 0.6996 / 0.6344 |
| key cosine within / across: K3 | 0.869 / 0.635 |
| key cosine within / across: K3c | 0.526 / -0.133 |
| calibration key gap: K3 / K3c | 0.249 / 0.674 |
| D1 KL: union / attn_L3 (K3c) / attn_L3 (K3) / scaled_mix / kc_dedup (K3c) / kc_dedup (K3) / mix / ataraxos_mmd | 0.1004 / 0.1014 / 0.1016 / 0.1086 / 0.1196 / 0.1188 / 0.1192 / 0.1220 |

## Post hoc diagnostics (after the verdicts were seen; NOT rule tests; `r7/expl_r7d2.py` -> `r7/report/expl_d2.json`)

- Terminal-like claims (H < 0.15) are the bad ones: squared value error 0.090 vs 0.038 for the other clean rollouts and
  0.128 vs 0.039 for the exploratory ones (2.4x and 3.3x; the spec expected about 4x).
- The trust head also ranks squared value error, the quantity the pooled MSE is built from: Spearman(beta, -squared
  error) 0.251 [0.189, 0.313] overall and 0.272 [0.254, 0.290] within pairs; for -H it is -0.181 and -0.368.
- The uniform pool's MSE (0.0137) splits into rollout noise (within-pair variance / 16 = 0.0022) and the rest (0.0115):
  bias of the depth-4 rollouts against Q* that all rollouts share (mean signed error -0.021), plus Q*'s own noise from
  64 playouts. Q*'s playout noise is about 0.0022 of it (plug-in variance / 64, mean over the 2,361 pairs; auditor's
  number), so the rollouts' shared bias is about 0.0093 of the uniform pool's 0.0137. Reweighting cannot remove shared
  bias. trust_pool's unequal weights add 0.00042 of noise [0.00027, 0.00056], which is about the size of its extra MSE
  (0.00046 [-0.00013, 0.00105]). The noise estimate treats the weights as fixed, although beta depends on each
  rollout's own e and H. The gain from down-weighting the bad rollouts is close to nothing here.
- With perfect knowledge of the tier, dropping the exploratory half (clean8) costs more than it saves (0.0155 vs 0.0137):
  in this testbed the exploratory rollouts carry nearly as much information as clean ones.

## Figures

- `r7/report/e7_d1.png`: key gaps (R6-E3's K4, K3, K3c) and D1 JSD / KL per pooling rule (data `r7/report/figures.csv`).
- `r7/report/e7_d2.png`: D2 action-value MSE per pooling rule on mixed bags, with clean-only as a reference.

## Compute (GPU-LOG; `r7/GPU-LOG-R7-E3.md`, rebuilt by `r7/gpu/ledger.py` from the shared logical-soup GPU-LOG.md)

| VM | provisioning | created | deleted | cost (ceiling) |
|---|---|---|---|---|
| soup-gpu-r7e3-main-1017094026-8653 | STANDARD (on-demand) L4, us-central1-a | 09:40Z | 09:59Z | $0.27 (0.31 h x $0.87/h) |

Total **$0.27** of the $1.50 R7-E3 ceiling (group r7 cap $6.50, shared with R7-E1 and R7-E2). The $0.87/h rate is the
unverified on-demand list-price ceiling (Billing API disabled). gcloud instances and disks list (project GCP_PROJECT,
name~soup-gpu-r7e3-) showed nothing left after the run.

## Deviations log (UTC; full text in `r7/PREREG-R7-E3-AMENDMENTS.md`)

- 09:39:45Z first prereg hash superseded before any launch (header sentence about the amendments file corrected);
  re-hashed 09:40:03Z.
- 09:40:26Z launch; the driver with the R6-E3 launch fix verified at 09:46:34Z that all three seeds were running. No
  crash, no rerun. Pipeline finished 09:58:22Z; VM deleted about 09:59Z.
- Before scoring: no new amendment for R7-E3.
- Interpretations declared before launch (prereg section 6): tempered + swapped cone for the exploratory tier; policy
  temperature 2.0 over the whole policy head; clean tier = R6-E3 setting; stiffness weights (1 - H/log 3)^2; trust
  consistency features without the tier label; stratified 6 + 6 initial batch for trust_resample; "that key head".
- The pre-hash CPU smoke used 12 D2 and 48 D1 test positions (disclosed in the prereg, section 6, item 11).
- Correction to prereg section 6 item 11(c) (from the audit): besides 48 calibration and 48 test D0 positions for the
  gate KL, the smoke's D0 part also computed, for the seed-101 checkpoint, battle accuracy on all 11,034 test battles
  (0.99991), the 7-known-slot NLL (9.4e-05) and the one-pass belief KL (0.0484) on all test D0 positions. These are the
  inherited R6-E3 gate numbers. They are deterministic and identical to the published R6-E3 values, so the smoke showed
  no new test information.

## Self-audit and independent audit

`r7/report_r7e3.py` re-derived 117 numbers from the per-item CSVs (D0 means, every D1 JSD/KL and key gap, the D2
Spearmans, the within-pair Spearman and every pool's MSE); the largest difference from the JSON summaries is 1.9e-5
(`r7/report/selfcheck.csv`). Independent audit (`r7/AUDIT-R7-E3.md`, a separate read-only pass): supported with corrections. Every verdict was
reproduced from the per-item CSVs (largest difference on a rule statistic 9.2e-5, from 6-digit rounding; no verdict is
near its bar), and the D0 KLs and both rollout tiers were recomputed independently from the seed-101 checkpoint on CPU
(anchor and route-B KL means within 4e-6 nats; tier statistics within sampling error). Its six corrections are
wording and disclosure fixes (C1-C6) and are applied above; none changes a verdict.

## Files

- Pre-registration: `r7/PREREG-R7-E3.md`, `r7/PREREG-R7-E3.sha256`, `r7/PREREG-R7-E3-AMENDMENTS.md`.
- Code: `r7/r7lib.py`, `r7/run_r7e3.py`, `r7/report_r7e3.py`, `r7/expl_r7d2.py`, `r7/gpu/` (stage, run_vm, drive, launch,
  ledger, launches.log); imports `r6/e3lib.py`, `r6/run_e3.py` and `r5/` unchanged.
- Results: `r7/runs/s{101,102,103}/eval/*.json|*.csv`; `r7/report/` (verdicts.csv, tables.md, summary.json,
  selfcheck.csv, expl_d2.json, figures.csv, e7_d1.png, e7_d2.png); `r7/GPU-LOG-R7-E3.md`.
- Page: `/home/agent/memo-serve/lir-agent/r7-e3.html`.
