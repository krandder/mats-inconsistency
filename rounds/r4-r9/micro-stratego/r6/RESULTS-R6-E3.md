# RESULTS-R6-E3: 6x6 Stratego marginal window rollouts, centred+salience keys, trust-weighted pooling (E-MS-MARGINAL)

Executor: an Opus 5.5 subagent of eurisko, 2026-10-17, for opus-mats (agent mail #3704). Spec:
`/home/agent/workspaces/mats/outbox/experiments-round-6-spec.md`, section 3 (rules E3-D0 to E3-D2). Pre-registration:
`r6/PREREG-R6-E3.md`, sha256 `5b45ec96...` (`r6/PREREG-R6-E3.sha256`), hashed 2026-10-17T06:34:20Z, before the only
launch (06:35:51Z, `r6/gpu/launches.log`). No `outbox/experiments-round-6-amendments*.md` existed before launch or before
scoring (checked 06:34Z, 06:35Z, 06:54Z). Operational events: `r6/PREREG-R6-E3-AMENDMENTS.md`. Paths below are relative
to `/home/agent/workspaces/mats/work/micro-stratego/`. The R5-E2 primary checkpoints were reused unchanged (no retraining;
parameter hashes checked before and after every seed's run).

## TL;DR

- **Pre-registered result: the gate E3-D0 PASSED; E3-D1 FAILED on its key rule only; E3-D2 FAILED on pooling.**
- **E3-D0 (gate) passes.** Scoring each cone slot marginally against the real last 4 plies removes R5-E2's veto damage.
  Sixteen marginal cone rollouts cut KL to the exact posterior by 48.2% [46.9, 49.5] against the 4-plies-old anchor
  (0.185 -> 0.096 nats per hidden slot; bar 40%). In R5-E2 the neighbour-conditioned rollouts made beliefs worse
  (0.047 -> 0.539). **But this is a gain over a stale belief only:** theta_real's own one-pass belief at the current turn
  sits at 0.047, closer to the posterior than the pooled rollouts (0.096) and than scoring every hidden slot (0.078).
- **E3-D1 fails, on the key gap.** K4 keys (bag-centred fingerprint + 8-slot salience) no longer collapse. Cosine is
  0.870 within flanks vs 0.643 across, a gap of 0.228 [0.211, 0.244] (R5-E2: 0.004). That is just under the 0.25 bar.
  Every policy-independence and accuracy comparison passes. attn_L3's belief JSD between left- and right-preferring
  search is 9.7x below ataraxos_mmd's and 10.5x below scaled_mix's. Its KL to the exact posterior is lower than
  scaled_mix's, ataraxos_mmd's and kc_dedup's (all intervals > 0). attn_L3 was trained toward the exact per-slot
  deduplication (the calibration reference), which the shared base cover makes policy-independent; it learned to
  imitate it.
- **E3-D2 fails.** The trust head again ranks rollout errors well: Spearman +0.247 [0.175, 0.319] vs -0.417 for
  entropy; difference +0.664 (bar 0.45); within entropy strata +0.197 (bar 0.15). But weighting by trust made action
  values worse, not better: MSE 0.0171 vs 0.0134 for entropy weighting, a 26.8% increase [14.6%, 39.0%] (bar: a 20% cut).
  The fitted entropy weighting chose gamma = 0, i.e. uniform weights, in every seed. No pooling rule ever chose a
  delusional move (0 of 240 decisions per rule in every seed), so the delusional-rate comparison is at floor and fails as written.
- **Post hoc:** within one action's 16 rollouts, trust ranks the rollouts correctly (Spearman +0.25 vs -0.58 for
  entropy). But its weights cut the effective sample to 10 of 16, which explains about half of the extra error. Of the
  uniform pool's 0.0134 MSE, about 0.0022 is rollout noise and about 0.0022 is noise in Q* itself (64 playouts per
  action); about 0.009 is bias of the depth-4 rollouts against Q*. Trust and entropy reweighting did not remove it; no
  other reweighting was tested.
- Cost: **$0.26** at the $0.87/h list-price ceiling (one on-demand L4, 0.30 h; ceiling $1.80). No r6e3 VM or disk left.

## Verdict table (from `r6/report/verdicts.csv`; 3 seeds; mean [95% seed-t interval, df = 2])

| rule | verdict | numbers (seeds 101, 102, 103; mean [95% CI]) |
|---|---|---|
| E3-D0 (0) anchor KL >= 0.12 | pass | 0.1859, 0.1856, 0.1841; 0.1852 [0.1828, 0.1876] (calibration) |
| E3-D0 (1) battle accuracy >= 0.99, 7-known NLL <= 0.01 | pass | 0.9999 [0.9998, 1.0001]; 0.0001 [0.0000, 0.0002] |
| E3-D0 (2) 16 marginal rollouts cut KL vs anchor >= 40% | pass | 0.4805, 0.4779, 0.4880; 0.4821 [0.4690, 0.4952] |
| **E3-D0 gate** | **PASS** | all three parts |
| E3-D1 (1) key cosine within minus across >= 0.25 | **fail** | 0.2202, 0.2334, 0.2291; 0.2276 [0.2109, 0.2443] (per position the mean of the two policies' gaps, then over positions; the pre-registration's order gives 0.2274 [0.2107, 0.2441]; bags with rollouts on one flank only are skipped: 3, 5 and 3 per seed) |
| E3-D1 (2) JSD ratio ataraxos_mmd / attn_L3 >= 2.5 | pass | 9.50, 7.17, 12.46; 9.71; difference 0.00196 [0.00123, 0.00270] |
| E3-D1 (2) JSD ratio scaled_mix / attn_L3 >= 2.5 | pass | 10.84, 8.00, 12.69; 10.51; difference 0.00215 [0.00170, 0.00260] |
| E3-D1 (2) KL: scaled_mix - attn_L3 > 0 | pass | 0.0071 [0.0060, 0.0082] |
| E3-D1 (2) KL: ataraxos_mmd - attn_L3 > 0 | pass | 0.0206 [0.0143, 0.0269] |
| E3-D1 (2) KL: kc_dedup - attn_L3 > 0 | pass | 0.0171 [0.0100, 0.0242] |
| **E3-D1** | **FAIL** | (1) fails by 0.022; all five parts of (2) pass |
| E3-D2 (1) Spearman(beta, -KL) >= 0.20 | pass | 0.2482, 0.2751, 0.2173; 0.2469 [0.1749, 0.3188] |
| E3-D2 (1) minus Spearman(-H, -KL) >= 0.45 | pass | 0.6809, 0.6909, 0.6189; 0.6636 [0.5667, 0.7605] |
| E3-D2 (1) same within 10 equal-count strata >= 0.15 | pass | 0.2290, 0.2038, 0.1568; 0.1965 [0.1055, 0.2875] |
| E3-D2 (2) MSE cut trust_pool vs stiff_pool >= 20% | **fail** | -0.3135, -0.2747, -0.2159; -0.2680 [-0.3902, -0.1459] (MSE 0.0134 -> 0.0171); MSE difference stiff - trust -0.0037 [-0.0064, -0.0009] |
| E3-D2 (2) delusional rate trust < stiff | **fail (at floor)** | 0 delusional choices for every pool in every seed |
| **E3-D2** | **FAIL** | (1) passes; (2) fails |

## What this means

- **The marginal fix works as a repair, not as a gain over one pass.** Without the neighbour conditioning, the window
  messages no longer veto true piece types. Sixteen rollouts then cover 79% of hidden slots and close about half of the
  gap between a 4-plies-old belief and the posterior. The spec framed this as updating a stale belief cache, and
  measured that way the gate passes clearly. Measured against what the agent already has, theta_real's own belief at the
  current turn (KL 0.047, `r6/runs/s10x/eval/d0.json`), the stale anchor plus window evidence stays worse than that
  belief (0.078 at best, scoring every hidden slot, 1.7x worse, as the R5-E2 post hoc diagnostic found). Whether the
  window adds anything to the current belief was not tested.
- **Policy independence comes from deduplication, and the support floor makes it reachable.** Every rollout that covers
  a slot carries the same marginal message. With a shared 4-rollout base cover, the exact per-slot deduplication (the
  "union call") is identical under left- and right-preferring search: JSD 0.0000, KL 0.100 (`r6/report/tables.md`).
  attn_L3, fitted only on calibration bags against a model-only reference, nearly reproduces it (JSD 0.0002, KL 0.102).
  That was its training target. The calibration reference is the union call over the base cover plus 128 reference
  rollouts. The 4 greedy base-cover rollouts already reach every hidden slot that any cone can reach (499 of 500 D1 test
  positions), so that reference equals the union call of every training bag (calibration belief loss of the union call
  1e-10 to 1e-6, `r6/runs/s10x/eval/fit.json`). D1 (2) shows that attn_L3 learns to imitate exact deduplication when
  trained toward it, not that it finds deduplication on its own (`r6/AUDIT-R6-E3.md`). The baselines are about 5-10x
  more policy-sensitive (3-seed mean JSD ratio to attn_L3: kc_dedup 4.8x, mix 7.4x, ataraxos_mmd 9.7x, scaled_mix
  10.5x; 3.9x to 12.7x across seeds; JSD 0.0011-0.0024) and 0.007-0.021 nats less accurate. This is the round-6 synthesis's point, that support and frequency must be separated, now on the
  Stratego board.
- **Keys: the collapse is fixed but the bar is not met.** Centring the fingerprint on its bag and adding the 8-slot
  salience gives a 0.228 gap against R5-E2's 0.004. Cones still overlap heavily across flanks (R5-E2 post hoc: Jaccard
  0.56), which limits how different a left key and a right key can be. attn_L3 did not need a larger gap to deduplicate.
- **Trust ranks rollouts but does not make a better average.** Trust's ranking signal replicated with fresh rollouts
  at the agents' setting: +0.25 overall, and within the 16 rollouts of one action as well. As a pooling weight it hurts.
  The pooled estimates' error is mostly bias of the depth-4 rollouts against Q* (plus noise in the 64-playout Q*
  itself). Trust and entropy reweighting did not remove it (Q* lies inside the range of the 16 claims for 89% of pairs,
  so the data alone do not show that no weighting could), and concentrating weight on fewer rollouts adds noise. Entropy weighting was no better: the
  calibration fit set gamma to 0.

Options we can see, each with its assumption. None has been run or chosen:

- Use trust to choose WHICH rollouts to run or to stop early, not to weight a fixed set. This assumes the ranking can
  pick rollouts with less bias rather than only less noise.
- Score the window against the current one-pass belief instead of the stale anchor, i.e. as a check on it. This assumes
  the policy head carries evidence that the one-pass belief misses, which nothing here has shown.
- Raise the key gap without new supervision: drop the shared-fingerprint part, or use salience alone. This assumes the
  0.25 bar matters for deduplication; here attn_L3 deduplicated at 0.228.

## Setup as run

- Checkpoints: R5-E2's `/tmp/r5e2/ckpt/s{101,102,103}/primary/` (theta_real 4 layers d 128, 1,001,644 parameters;
  slot- and rollout-trust heads; theta_amp), hash-checked. Data: R5-E2's `/tmp/r5e2/data/`, plus `oracle_d1_test.npz` (P*_full
  for the 500 D1 positions, computed before the hash for scoring only).
- Marginal window messages (`r6/e3lib.py`): anchor = theta_real's 0-step belief at t0 = max(t - 4, t mod 2); for every
  hidden slot s and type k, the log-likelihood of the opponent's real moves in the window under theta_real's policy head
  with Z_s = k and every other hidden slot MASK; mu_s = softmax(log P_anchor,s + LL_s).
- Rollouts: r5 value rollouts (stiff_voi cone sample, depth 4), each carrying the messages of its cone (3 nearest hidden
  slots), theta_trust's beta and an attention fingerprint. K4 keys: normalise(W [LayerNorm(fingerprint - bag mean) ||
  8-slot salience]), fitted with kc_dedup's calibration loss + the salience-alignment loss.
- Fitting (calibration split, model-only references): 1,500 two-flank roots, base cover 4 + pools 32 + 32, reference
  64 + 64, 1,500 Adam steps per rule; stiff_pool's gamma on 1,000 calibration positions against realised results
  (chosen gamma: 0, 0, 0; `r6/runs/s10x/eval/fit.json`).
- Scoring: gate on 1,000 calibration positions (and the same on 1,000 test positions, reported); D1 on 500 two-flank test
  positions (bags: 4 shared base-cover rollouts + 20 skewed, mean 17.0 Left under U_Left and 3.0 under U_Right); D2 on 240
  Q* test positions x every legal action x 16 rollouts (37,776 rollouts per seed).

## Rule tables (`r6/report_e3.py` -> `r6/report/tables.md`, `r6/report/verdicts.csv`)

| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass |
|---|---|---|---|---|---|---|
| E3-D0 (0) | anchor KL(P*_full ‖ P_anchor) per hidden slot, calibration | 0.1859, 0.1856, 0.1841 | 0.1852 | [0.1828, 0.1876] | >= 0.12 | True |
| E3-D0 (1) | battle accuracy, test | 0.9999, 1.0000, 0.9999 | 0.9999 | [0.9998, 1.0001] | >= 0.99 | True |
| E3-D0 (1) | 7-known-slot NLL (nats), test | 0.0001, 0.0002, 0.0001 | 0.0001 | [0.0000, 0.0002] | <= 0.01 | True |
| E3-D0 (2) | KL cut of 16 marginal rollouts vs anchor, calibration | 0.4805, 0.4779, 0.4880 | 0.4821 | [0.4690, 0.4952] | >= 0.40 | True |
| E3-D1 (1) | K4 key cosine within minus across flanks | 0.2202, 0.2334, 0.2291 | 0.2276 | [0.2109, 0.2443] | >= 0.25, CI > 0 | False |
| E3-D1 (2) | belief JSD ratio ataraxos_mmd / attn_L3 | 9.4967, 7.1707, 12.4577 | 9.7084 | (diff [0.0012, 0.0027]) | >= 2.5 | True |
| E3-D1 (2) | belief JSD ratio scaled_mix / attn_L3 | 10.8389, 8.0027, 12.6862 | 10.5093 | (diff [0.0017, 0.0026]) | >= 2.5 | True |
| E3-D1 (2) | KL scaled_mix - attn_L3 | 0.0072, 0.0066, 0.0075 | 0.0071 | [0.0060, 0.0082] | > 0 | True |
| E3-D1 (2) | KL ataraxos_mmd - attn_L3 | 0.0215, 0.0225, 0.0177 | 0.0206 | [0.0143, 0.0269] | > 0 | True |
| E3-D1 (2) | KL kc_dedup - attn_L3 | 0.0173, 0.0198, 0.0141 | 0.0171 | [0.0100, 0.0242] | > 0 | True |
| E3-D2 (1) | Spearman(beta, -KL(Q* ‖ q_r)) | 0.2482, 0.2751, 0.2173 | 0.2469 | [0.1749, 0.3188] | >= 0.20 | True |
| E3-D2 (1) | Spearman(beta) - Spearman(-H) | 0.6809, 0.6909, 0.6189 | 0.6636 | [0.5667, 0.7605] | >= 0.45 | True |
| E3-D2 (1) | same within 10 equal-count strata | 0.2290, 0.2038, 0.1568 | 0.1965 | [0.1055, 0.2875] | >= 0.15 | True |
| E3-D2 (2) | relative MSE reduction trust vs stiff | -0.3135, -0.2747, -0.2159 | -0.2680 | [-0.3902, -0.1459] | >= 0.20 | False |
| E3-D2 (2) | delusional rate stiff - trust | 0, 0, 0 | 0 | [0, 0] | > 0 | False |

Secondary numbers (3-seed means, `r6/report/tables.md`):

| quantity | value |
|---|---|
| gate, calibration: KL anchor / 16 rollouts / every hidden slot scored / one pass at t | 0.185 / 0.096 / 0.078 / 0.047 |
| gate, test: KL anchor / 16 rollouts (cut 44.5%) / every slot / one pass at t | 0.176 / 0.098 / 0.078 / 0.048 |
| share of hidden slots covered by the 16 rollouts | 0.787 |
| gate, scaled_mix with 16 rollouts (reported, not a rule): KL cut vs anchor | 40.7% [40.2, 41.3] calibration (KL 0.110); 37.6% [37.1, 38.2] test (`r6/runs/s10x/eval/d0.json`, reduction_scaled16_vs_anchor) |
| D1 KL to P*_full: union call / mix / scaled_mix / ataraxos_mmd / kc_dedup / attn_L3 | 0.100 / 0.119 / 0.109 / 0.122 / 0.119 / 0.102 |
| D1 JSD: union call / mix / scaled_mix / ataraxos_mmd / kc_dedup / attn_L3 | 0.0000 / 0.0017 / 0.0024 / 0.0022 / 0.0011 / 0.0002 |
| D1 KL of the anchor / one pass at t / every slot scored | 0.179 / 0.049 / 0.080 |
| D2 MSE unif / stiff (gamma 0) / trust | 0.0134 / 0.0134 / 0.0171 |
| D2 share of rollouts with H(q_r) < 0.15 (near-terminal claims) | 0.142 |

## Post hoc diagnostics (after the verdicts were seen; NOT rule tests)

From the synced CSVs only (`r6/expl_d2.py` -> `r6/report/expl_d2.json`), 3-seed means [95% CI]:

- Within one (position, action) pair, the 16 rollouts share one Q*. Spearman(beta, -KL) within pairs is +0.25 [0.15,
  0.35], against -0.58 [-0.61, -0.55] for -H. Between pairs (pair means) it is +0.17 [-0.01, 0.34]. Trust ranks the
  rollouts of the same action in the right order.
- Trust weights keep an effective sample size of 10.2 [9.5, 11.0] of 16 (largest single weight 0.18 on average).
- Shrinkage is not the explanation. Trusted rollouts make somewhat less extreme claims (within-pair Spearman of beta
  with |e_r - 0.5| is -0.22 [-0.37, -0.08]). But trust-pooled estimates are not systematically closer to 0.5 (mean
  distance 0.190 vs 0.192 uniform; Q*: 0.179).
- Variance explains about half of the extra error. Treating a pair's rollouts as i.i.d., the noise part of the squared
  error is 0.0022 for the uniform pool and 0.0041 for trust_pool. The extra noise is 0.0019 [0.0015, 0.0023] of the
  extra MSE of 0.0037 [0.0009, 0.0064]. Of the uniform pool's 0.0134, about 0.0022 is the sampling noise of Q* (64
  playouts) and about 0.009 is bias of the rollouts against Q* (`r6/AUDIT-R6-E3.md`, auditor observations).

## Figures

- `r6/report/e3_d0_d1.png`: gate KLs (anchor, 16 rollouts, every slot, one pass) and D1 JSD / KL per aggregator; data
  in `r6/report/figures.csv`.
- `r6/report/e3_d2.png`: D2 MSE per pooling rule; data in `r6/report/figures.csv`.

## Compute (GPU-LOG; `r6/GPU-LOG-R6-E3.md`, rebuilt by `r6/gpu/ledger.py` from the shared logical-soup GPU-LOG.md)

| VM | run | provisioning | created | deleted | cost (ceiling) |
|---|---|---|---|---|---|
| soup-gpu-r6e3-main-1017063551-4510 | all three seeds | on-demand (STANDARD) | 06:37Z | 06:54Z | $0.26 |
| **total** | | | | | **$0.26** of the $1.80 ceiling |

Group r6, us-central1-a, g2-standard-8 + 1 L4, launched through `logical-soup/infra/gpu-run.sh` (GPU_RUN_TIMEOUT 75m,
worst-case reservation $1.61). After deletion, `gcloud compute instances list` and `disks list` (name~soup-gpu-r6e3-)
show nothing (`r6/GPU-LOG-R6-E3.md`). Wall time: VM up 06:37Z, deps 3 min, the three seeds 06:44-06:54Z in parallel (fits
about 7.5 min, then D0 15 s, D1 55 s, D2 40 s per seed). CPU: the D1 oracle 3 s; the CPU smoke about 7.5 min (06:24:58-06:32:24Z).

## Deviations log (UTC; full text in `r6/PREREG-R6-E3-AMENDMENTS.md`)

- Declared before the hash (`r6/PREREG-R6-E3.md` section 6): union-call pooling for the gate; the anchor condition as a
  gate part; greedy shared base cover; Binomial skew from independent pools; scaled_mix fitted against the model-only
  reference; K4 fitted via kc_dedup + L_key; D2 on one rollout set at the agents' setting; gamma fitted on calibration;
  primary checkpoints only; one VM; the pre-hash CPU crash smoke on small subsets (its numbers were seen; nothing changed).
- 06:40:42Z the driver's launch command did not start the run (an output redirect into /root ran without sudo); found
  06:43Z; started by hand 06:44:14Z with the redirect inside sudo; the corrected launcher is `r6/gpu/drive_v2.sh`
  (`r6/gpu/drive.sh` keeps its original inode but was rewritten in place at 06:44:50Z, then restored by reversing the
  one-line edit; no copy of the launched bytes exists, so this cannot be verified, but the driver's output after its
  polling loop in `/tmp/r6e3/launch-r6e3-main.log` is clean, consistent with unchanged bytes). No experiment setting
  changed.
- Also disclosed (audit): the pre-hash smoke showed a key gap of 0.054 and trust_pool 29% worse than stiff_pool on 12
  positions; no code changed after it (e3lib.py 06:22:57Z, run_e3.py 06:24:34Z, smoke 06:24:58-06:32:24Z).
  `r6/report_e3.py` was written after the hash and its one table label was changed at 06:37:17Z after a dry run on the
  smoke output (`||` to `‖`); no rule logic changed.
- After the verdicts: the post hoc diagnostics above (`r6/expl_d2.py`), labelled; no rule was re-evaluated.

## Self-audit and independent audit

- `r6/report_e3.py` re-derives 66 numbers from the per-item CSVs; largest difference 2.6e-6 (`r6/report/selfcheck.csv`).
  The CSVs round to 6 significant digits. Recomputed from the CSVs with average ranks, Spearman(-H, -KL) moves by at
  most 3.5e-7 and Spearman(beta, -KL) by at most 2.6e-6 (`r6/AUDIT-R6-E3.md`).
- Each run asserted that theta_real's parameter hash equals R5-E2's recorded `param_hash_final` before and after
  (`r6/runs/s10x/eval/done.json`).
- `r6/AUDIT-R6-E3.md` is a separate pass by an independent read-only agent. Verdict: **supported with corrections**.
  It reproduced all 18 verdict rows from the per-item CSVs with its own code, recomputed the anchor KL and the marginal
  window posterior from the checkpoints on 1,000 calibration and 1,000 test positions for all three seeds (per position
  at most 8.6e-4 apart; its own random 16-rollout draw gives a 48.1% cut), found no oracle in any fitted quantity,
  confirmed the pre-registration hash, timing and byte-exact rules, and confirmed $0.26 and no r6e3 VM or disk left. Its
  corrections (wording, two missing numbers, the driver-file note) are applied above; none changes a verdict.

## Files

- `r6/PREREG-R6-E3.md`, `.sha256`, `-AMENDMENTS.md`; code `r6/e3lib.py`, `r6/run_e3.py`, `r6/report_e3.py`,
  `r6/oracle_d1.py`, `r6/expl_d2.py`, `r6/site_add_entry.py`, `r6/gpu/` (`stage.sh`, `run_vm.sh`, `drive.sh`,
  `drive_v2.sh`, `launch.sh`, `ledger.py`, `launches.log`); reused `r5/` modules unchanged.
- Results: `r6/runs/s101|s102|s103/eval/` (`fit.json`, `d0.json`, `d0_gate_{cal,test}.csv`, `d0_battles.csv`,
  `d0_nll7.csv`, `d1.json`, `d1_positions.csv`, `d2.json`, `d2_actions.csv`, `d2_rollouts.csv`, `done.json`);
  `r6/report/` (`verdicts.csv`, `tables.md`, `summary.json`, `selfcheck.csv`, `figures.csv`, `e3_d0_d1.png`, `e3_d2.png`,
  `expl_d2.json`); ledger `r6/GPU-LOG-R6-E3.md`.
