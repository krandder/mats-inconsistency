# R6-E1 (E-INFO-V3): key and attention variants, evidence-floor policy independence and key-space PUCT: results

eurisko study, 2026-10-17 (06:15-08:40Z). Written by an Opus 5.5 subagent for eurisko. Assignment: opus-mats, agent-mail #3704
(group `r6`; this experiment's ceiling $2.50; 3 seeds; seed-t 95% CIs; independent audit; exact oracles for evaluation only).
Spec: gemini, `/home/agent/workspaces/mats/outbox/experiments-round-6-spec.md`, section 1. No amendment file appeared before
launch or before scoring.
- Pre-registration: `PREREG-R6-E1.md`, registered 06:27:53Z, 13 s before the first r6e1 launch (06:28:06Z in the launch
  log; GCE insert 06:28:17Z; the first r6e1 line in `logical-soup/GPU-LOG.md`, line 2064, is stamped 06:28Z). Its
  deviations log is in section 5.
- Pre-spend ambiguity mail: #3708, 06:18:47Z.
- Every number below names its source file; paths are relative to `li-net/r6e1/`. Audit: `AUDIT-R6-E1.md` (section 9).

Round 5 found three problems: raw keys that barely tracked overlap (Pearson 0.375,
`li-net/r5e1/results/v2/analysis/results.json`); a policy-independence test confounded by the cells only one search read;
and a mutual-support weight just under its bar. This round tests the fixes gemini proposed for each. As in R5-E1's amended
design, a "read" is a bounded-context call that shows the world model the real sonar readings of one patch, not a
hypothetical sample.

## 1. TL;DR

**3 of 5 pre-registered rules pass; the two failures each miss on one clause.** Verdicts: `results/main/analysis/verdicts.md`
and `results.json`; 3-seed mean [seed-t 95% CI].

| rule | verdict | decisive numbers |
|---|---|---|
| E1-D0 gate | **PASS** | world model Pearson with exact Bayes 0.9994 [0.9994, 0.9995] (bar 0.99); on P-FullSupport, scaled_mix cuts KL vs the root belief by 63.2% [62.1, 64.2] (bar 40%) |
| E1-D1 keys track graded overlap | **PASS** | K4: Pearson 0.863 [0.858, 0.867], gap 0.629, cosine monotone in overlap (Spearman 1.00). K3: 0.880 [0.876, 0.884], gap 0.967, Spearman 1.00 (bars 0.60 with CI > 0.50, gap 0.50, Spearman 0.94) |
| E1-D2.1 full-support policy independence | **PASS** | attn_L3-K4's JSD between the two preferences is 5.8x / 6.5x / 2.8x lower than scaled_mix / ataraxos_mmd / kc_dedup (bar 2.5x; every CI > 0), and its KL is below each of those three baselines' (all paired CIs > 0); attn_L3-K3's KL is lower still (0.033 vs 0.034; lowest of all pools in seeds 101 and 102) |
| E1-D2.2 active key-space PUCT | **FAIL** (coverage clause) | coverage 0.9494 [0.9471, 0.9517] vs bar 0.95 (open loop: 0.828); the other three clauses pass: JSD 2.00x lower [1.91, 2.09] (bar 1.8x), KL 38% lower (bar 20%), wave-1 preferred share 0.82 (bar 0.75; the 4 patches straddling rows 2-3 count for both halves, as registered; reads wholly in the preferred half: 0.67) |
| E1-D3 three-regime weight law | **FAIL** (one clause) | weight law holds: W(16)/W(1) = 1.08 at tau 0.3 (bar <= 1.25); W(8)/W(1) = 3.54 [3.28, 3.79] and W(16)/W(1) = 4.52 at tau 2.5 (bars 2.20, 2.80); monotone in noise in every seed; MSE cut vs ataraxos_mmd at tau 0.3 38.5% (bar 35%); but MSE cut vs kc_dedup at tau 2.5 is 10.3% [7.7, 13.0] (bar 15%) |

- **Keys: the belief-shift footprint fixes them.**
  - The K3 salience key (where the read moved the beliefs) correlates 0.88 with true overlap. Its cosine climbs from 0.01
    for disjoint reads to 0.96 for the same patch, and across-sector reads are orthogonal.
  - K4 (centred hidden state + salience) gives 0.86.
  - Bag-centring alone (K2) separates the same patch from other patches (gap 0.58) but does not grade small overlaps
    (Pearson 0.40).
  - The raw key K1 reproduces round 5 (0.38 here, 0.375 there).
- **Policy independence, with the support gap removed: yes for attn_L3.**
  - When both searches read every cell once and then 18 skewed clean reads, the two merged beliefs differ by JSD 0.0078.
    For the fitted one-step pools the figure is 0.022-0.051.
  - attn_L3's KL is 0.034, against 0.074-0.120 for those pools (the one-call union of the same clean readings is at 0.002).
  - Interpretation: this is consistent with round 5's failure coming from the support gap and the yardstick, not from
    attn_L3's deduplication. With full support even round-5-style raw keys (attn_L3-K1) give JSD 0.0091, 5.0x below
    scaled_mix. Round 5's protocol was not rerun.
- **Active key-space PUCT closes most of the coverage gap.**
  - With 4 waves of 6 reads it covers 94.9% of cells, against 82.8% for open-loop 85/15 search.
  - It halves attn_L3's policy dependence (JSD 0.0261 to 0.0131) and cuts KL by 38% (0.080 to 0.050).
  - It keeps 82% of wave 1 in the preferred half (the 4 patches straddling rows 2-3 count for both halves, as registered;
    reads wholly in the preferred half: 67% [64, 69]).
  - It misses the 95% coverage bar by 0.06 percentage points.
- **The weight law holds on the 3-seed means in all three regimes.**
  - Sixteen identical clean reads weigh 1.08x one (CI [0.63, 1.52]; seed 102 is at 1.246, just under the 1.25 bar).
  - Independent reads at the two noise tiers earn more weight as they accumulate: W(8)/W(1) = 2.00 at tau 1.5 and 3.54 at
    tau 2.5, ordered in every seed.
  - theta_trust grades the tiers: mean beta on the calibration reads 3.47 / 2.18 / 1.52 (3-seed means; `results.json`
    `fit.<seed>.beta_by_tau`).
  - What fails is attn_L3's margin over kc_dedup (10% vs 15%). kc_dedup itself uses the new K4 keys; whether the
    better keys narrowed the gap was not tested.
- **looped_lin_T4** (2,789 parameters, 1/39 of attn_L3-K4) is about as accurate as kc_dedup but has 2.0x attn_L3's KL on
  P-FullSupport (secondary bar: 1.15x). Its keys are as good as K4's (Pearson 0.885).
- **Cost: $1.10** at the list-price ceiling ($0.28 for a spot VM preempted 15 minutes into training, $0.82 for the on-demand
  rerun), of the $2.50 ceiling. Zero r6e1 VMs at 07:49Z and at 08:28:06Z.

## 2. What ran

- **Questions (spec 1.1).**
  1. Do keys built from bag-centred hidden states and/or the read's own belief-shift footprint track graded patch overlap, where
     R5-E1's raw keys did not (Pearson 0.375)?
  2. Is a looped linear-attention pool (`looped_lin_T4`) a cheap substitute for attn_L3?
  3. With the support gap removed, does attn_L3 make two oppositely steered searches agree?
     - **P-FullSupport:** both searches read all 6 partition tiles, then 18 skewed reads.
     - **P-ActivePUCT:** a 4-wave key-space PUCT search covers the board by itself.
  4. Does the Hellwig / Clemen-Winkler weight law hold across three noise tiers?
- **Environment and world model** (as R5-E1 Part A v2; `li-net/r5e1/partA2.py`, imported unchanged):
  - 6x6 board, ships 4/3/2, a commander's note, and a dense sonar field (one reading per cell, flip 0.15).
  - The root belief sees the note + 2 anchor readings. A read is one theta_real call on the note, the anchors and the 6
    readings of one 2x3 / 3x2 patch. It is a local evidence read, not a hypothetical sample.
  - Read tiers: tau 0.3 clean; 1.5 and 2.5 re-flip each patch reading with probability 0.127 and 0.28 (effective flips 0.239
    and 0.346).
  - theta_real v2 (669,633 parameters) was retrained exactly as in R5-E1 (40,000 steps), because no checkpoint had been kept.
  - Exact Bayes over 49,504 ship patterns, given all 36 readings and the note, is used for scoring only.
- **Learned pieces** (calibration split, 3,200 fit boards). Learned pools and fitted scalars: target = the model's own
  unskewed K = 128 union-call belief, never exact Bayes.
  - theta_trust: real excess surprise of each calibration read against that board's true occupancy (no exact Bayes).
  - attn_L3 with each of four key heads:
    - K1 raw: LN(hbar);
    - K2 bag-centred hbar;
    - K3 the salience footprint s = |Delta l| / ||Delta l||;
    - K4 [LN(centred hbar) || s] plus a salience-alignment loss.
  - looped_lin_T4: 4 Richardson steps sharing one K4 key head and one attention matrix G, each with its own learned step
    size eta_t and diagonal preconditioner P_t (as registered); 2,789 parameters.
  - Fitted scalars: ataraxos_mmd eta, kc_dedup u (frozen K4 keys), scaled_mix alpha (l0 + alpha sum Delta l).
- **Tests** (500 test boards, the same as R5-E1 v2):
  - P-FullSupport;
  - P-ActivePUCT: open_U (85/15 open loop) vs puct_key_U (4 waves of 6, preference-weighted variance divided by learned key
    crowding against earlier waves; candidate keys come from prospective reads with readings imputed from the current belief);
  - key geometry on the open_U and unskewed bags;
  - controlled bags with N same-patch reads at three tiers.
- **Seeds** 101, 102, 103 on one L4 (`r6e1-main2`, the on-demand rerun after `r6e1-main` was preempted). Statistics:
  3-seed means with seed-t 95% intervals (t = 4.303).

## 3. E1-D0: the gate (PASS)

Source: `results/main/analysis/results.json` (`verdicts.E1-D0`, `stats`).
- **World model.** theta_real's one-pass belief on the standard root (6 readings + note) correlates 0.9994 [0.9994, 0.9995]
  with exact Bayes on the same evidence (0.9994 / 0.9995 / 0.9994).
- **P-FullSupport.**
  - The root belief (note + 2 anchors) is at KL 0.280 nats per cell from the full-board posterior.
  - scaled_mix (l0 + alpha sum_r Delta l_r, alpha = 0.22, per seed 0.226 / 0.218 / 0.223 in `results.json` `fit`, fitted
    to the model's own reference) brings it to 0.103: a cut of 63.2% [62.1, 64.2].
  - For reference, the one-call union of the same clean readings is at 0.0022.

## 4. E1-D1: keys (PASS)

All within-bag pairs of the open_U NW and SE bags and an unskewed bag, 3 x 276 pairs x 500 boards per seed. Source:
`results.json` (`stats.keys`), figure `results/main/analysis/keys.png` (data `keys.csv`).

| key | Pearson with overlap/6 | same-patch cos | across-sector cos | gap | mean cosine at 0 / 1 / 2 / 3 / 4 / 6 shared cells | Spearman |
|---|---|---|---|---|---|---|
| K1 raw (R5-E1's) | 0.382 | 0.950 | 0.756 | 0.194 | 0.74, 0.71, 0.77, 0.80, 0.87, 0.95 | 0.94 |
| K2 bag-centred | 0.403 | 0.793 | 0.213 | 0.580 | 0.18, 0.14, 0.21, 0.28, 0.45, 0.79 | 0.94 |
| **K3 salience** | **0.880 [0.876, 0.884]** | 0.955 | -0.012 | **0.967 [0.936, 0.999]** | 0.01, 0.27, 0.50, 0.67, 0.77, 0.96 | **1.00** |
| **K4 centred + salience** | **0.863 [0.858, 0.867]** | 0.943 | 0.314 | **0.629 [0.627, 0.632]** | 0.31, 0.42, 0.54, 0.66, 0.75, 0.94 | **1.00** |
| K4 head of looped_lin_T4 | 0.885 | 0.942 | 0.321 | 0.621 | 0.31, 0.42, 0.55, 0.67, 0.76, 0.94 | 1.00 |

- K3 and K4 each meet both clauses in every seed (per-seed Pearson 0.880 / 0.881 / 0.878 and 0.863 / 0.861 / 0.865).
- Why it works (interpretation): s_r = |Delta l_r| / ||Delta l_r|| is where the read moved the beliefs. Two reads of
  overlapping patches move overlapping cells, so their footprints overlap more as the reads share more cells (K3 cosine
  0.01 / 0.27 / 0.50 / 0.67 / 0.77 / 0.96 at 0 / 1 / 2 / 3 / 4 / 6 shared cells), with no patch index ever given.
- Round 5's exploratory finding is confirmed directly: removing the shared component (K2) fixes the same-vs-across gap but
  not the grading. The footprint is what grades the overlap.

## 5. E1-D2: policy independence (D2.1 PASS, D2.2 FAIL)

Source: `results.json` (`verdicts.E1-D2.1`, `E1-D2.2`, `stats`), figure `results/main/analysis/policy_independence.png`
(data `policy_independence.csv`).

**P-FullSupport** (each search: the 6 tiles that partition the board, then 18 clean reads, 90% in its preferred half). KL to
exact Bayes and JSD between the U_NW and U_SE merged beliefs, 3-seed means:

| pool | KL | JSD(U_NW, U_SE) | JSD ratio to attn_L3-K4 [CI] | KL minus attn_L3-K4 [CI] |
|---|---|---|---|---|
| mix | 0.197 | 0.0038 | (no rule) | |
| scaled_mix | 0.103 | 0.0453 | 5.82 [5.15, 6.49] | 0.0695 [0.0658, 0.0732] |
| ataraxos_mmd | 0.120 | 0.0509 | 6.54 [5.81, 7.26] | 0.0863 [0.0830, 0.0897] |
| kc_dedup (K4 keys) | 0.074 | 0.0221 | 2.85 [2.48, 3.21] | 0.0401 [0.0297, 0.0506] |
| looped_lin_T4 | 0.069 | 0.0219 | (secondary) | KL ratio 2.04 [1.84, 2.24] |
| attn_L3-K1 / K2 / K3 | 0.037 / 0.037 / 0.033 | 0.0091 / 0.0096 / 0.0069 | | |
| **attn_L3-K4** | **0.034** | **0.0078** | | |
| union call (clean readings, one call) | 0.002 | 0.0000 | | |

- **E1-D2.1 PASS.** Against each of scaled_mix, ataraxos_mmd and kc_dedup: the JSD ratio is above 2.5 (every per-seed ratio
  too: 5.67-6.13, 6.25-6.84 and 2.68-2.96), and both paired CIs (JSD and KL) exclude 0.
  - Interpretation: even here, where both searches see every cell, the summing pools let the about 16 NW reads outvote
    the about 2 SE reads; attn_L3 is much less affected (JSD 0.0078 vs 0.022-0.051).
  - mix's JSD (0.0038) is lower than attn_L3's only because mix barely uses the reads (KL 0.197). That is the dilution
    round 5 identified, and why the rule compares against scaled_mix.
- **Secondary, looped_lin_T4: fails.** It has 2,789 parameters (attn_L3-K4: 108,228), but its KL is 2.04x attn_L3's (bar
  1.15x). It ends level with kc_dedup.
- The key variant matters little for accuracy here: attn_L3 with any key is at KL 0.033-0.037.

**P-ActivePUCT** (reads 50/50 clean and tau 1.5):

| | open_U (85/15 open loop) | puct_key_U (4 waves x 6) |
|---|---|---|
| coverage (share of cells read) | 0.828 [0.819, 0.836] | **0.9494 [0.9471, 0.9517]** (0.9484 / 0.9496 / 0.9502) |
| attn_L3-K4 JSD(U_NW, U_SE) | 0.0261 | 0.0131 (ratio 2.00 [1.91, 2.09]; difference 0.0131 [0.0126, 0.0135]) |
| attn_L3-K4 KL | 0.0803 | 0.0496 (cut 38.2% [35.7, 40.7]) |
| union call JSD / KL (descriptive) | 0.0179 / 0.042 | 0.0041 / 0.011 |
| preferred-half share | 0.85 by construction | wave 1 under U_NW: 0.824 [0.815, 0.833] (the 4 patches straddling rows 2-3 count for both halves, as registered; wholly in the preferred half: 0.666 [0.642, 0.691]) |

- **E1-D2.2 FAIL**, on coverage only. puct_key_U reads 94.94% of cells, short of the 95% bar by 0.06 percentage points.
  Seed 101 is at 0.9484; seeds 102 and 103 reach 0.9496 and 0.9502. The JSD, KL and wave-1 clauses all pass.
- In 4 waves, the PUCT score (belief variance divided by key crowding) moves the search to unread cells (coverage +12
  points); which term does it was not tested. The union call's JSD (how differently the two read sets inform the model)
  falls 4.4x, and attn_L3's falls 2.0x. After PUCT, attn_L3's JSD (0.0131) is about 3x the union call's (0.0041; per
  seed 3.19 / 3.24 / 3.13; `summary.json` `ap_puct_jsd_*`).

## 6. E1-D3: the three-regime weight law (FAIL on one clause)

Controlled bags: N reads of one patch at each tier, plus 8 shared background reads. attn_L3-K4 cluster weight W(N), 3-seed
means. Source: `results.json` (`verdicts.E1-D3`), figure `results/main/analysis/controlled_bags.png` (data
`controlled_bags.csv`).

| tier | W(1) | W(2) | W(4) | W(8) | W(16) | W(8)/W(1) | W(16)/W(1) |
|---|---|---|---|---|---|---|---|
| tau 0.3 (identical clean reads) | 0.86 | 0.96 | 0.93 | 0.90 | 0.92 | 1.05 | **1.08** [0.63, 1.52] (1.09 / 1.25 / 0.89) |
| tau 1.5 (effective flip 0.239) | 0.56 | 0.80 | 0.99 | 1.12 | 1.23 | 2.00 | 2.19 |
| tau 2.5 (effective flip 0.346) | 0.36 | 0.58 | 0.89 | 1.26 | 1.61 | **3.54** [3.28, 3.79] | **4.52** [3.95, 5.09] |

- **Clause 1 (clean, deduplication): passes.** W(16)/W(1) = 1.08 (bar <= 1.25). attn_L3's MSE is 38.5% [36.4, 40.7] below
  ataraxos_mmd's (bar 35%).
- **Clause 2 (high noise, mutual support): fails on its MSE part.** W(8)/W(1) = 3.54 with CI lower bound 3.28 (bars 2.20 and
  1.90) and W(16)/W(1) = 4.52 (bar 2.80) pass. But attn_L3's MSE is only 10.3% [7.7, 13.0] below kc_dedup's (bar 15%). It is
  23% below ataraxos_mmd's and 47% below mix's.
- **Clause 3 (monotone in noise): passes in every seed.** W(8)/W(1) is 1.05 < 2.00 < 3.54 on average.
- So the weight law itself, deduplicate when reads are identical and add support when they are independent and noisy, now
  holds on the 3-seed means at all three tiers; at tau 0.3, W(16)/W(1) = 1.08 has CI [0.63, 1.52] and seed 102 is at
  1.246, just under the 1.25 bar. What fails is a margin over a one-step pool that shares the same improved keys.

## 7. Deviations and what ran when

- **06:15Z** assignment (#3704). **06:18:47Z** pre-spend ambiguity mail (#3708), with every resolution later restated in
  the pre-registration:
  - no cached theta_real, so it was retrained;
  - scaled_mix's alpha fitted to the model's own reference, not to exact Bayes;
  - looped_lin_T4's update rule;
  - the K3 / K4 details;
  - the tau mix;
  - the PUCT selection, crowding window and candidate keys;
  - rule scope.
- **06:27:53Z** pre-registration (`PREREG-R6-E1.md`, sha256 8f0a3ec5...b65c; the registered copy is in
  `results/PREREG-R6-E1-as-registered.md`). It discloses the two CPU code-path smokes; there was no GPU smoke.
- **06:28Z** launch `r6e1-main` (spot). **06:47:12Z** GCE preempted it 15 minutes into theta_real training; no result existed.
  Cost $0.28, recorded by a COST-AUDIT line (deviation 2).
- **06:51Z** identical rerun `r6e1-main2` on an on-demand VM. The remote budget was 90 instead of 100 minutes, to keep the
  worst case under the ceiling; the run needed 52. It finished **07:47Z** with every seed rc 0 and no resume needed
  (`results/main/s*/rc.json`).
- Deviation 1: the registered text gives the ambiguity mail's time as 06:16Z; agent-mail records 06:18:47Z.
- Deviation 3, logged at 08:28Z (in `PREREG-R6-E1.md` too) after the audit found it: **07:48:57Z, `analyze6.py` was
  edited after the raw results existed (plot only).** Panel 3 of `policy_independence.png` now shows the union-call JSD
  instead of KL/10. The registered script (sha256 2f76562b..., kept as `results/analyze6-as-registered.py`) reproduces
  `results.json`, `verdicts.md`, the three CSVs and the other two PNGs byte-for-byte (audit check 1). No other analysis
  code changed after the results existed.
- No amendment file existed at launch (06:28Z) or at scoring (07:48Z). Nothing was retuned after any test number.
- Exact Bayes was used only for test metrics, the gate and the monitor curves. PUCT chooses its reads from prospective reads
  whose readings are imputed from its own belief; unit test `test_prospective_reads_never_use_real_readings`.

## 8. Compute and GPU-LOG

L4 VMs (g2-standard-8) through `logical-soup/infra/gpu-run.sh`, group `r6` (cap $12 shared with R6-E3), at the $0.87/h
list-price ceiling (real spend on spot is lower). Ledger: `python3 ledger6.py --markdown`.

| VM | zone | provisioning | created (UTC) | deleted (UTC) | cost USD (ceiling) | basis |
|---|---|---|---|---|---|---|
| soup-gpu-r6e1-main-1017062806-8653 | us-central1-a | SPOT | 2026-10-17T06:28Z | 2026-10-17T06:51Z | 0.28 | DELETE/COST-AUDIT line |
| soup-gpu-r6e1-main2-1017065125-1754 | us-central1-a | STANDARD | 2026-10-17T06:51Z | 2026-10-17T07:48Z | 0.82 | DELETE/COST-AUDIT line |

**R6-E1 total (ceiling): $1.10** of its $2.50 ceiling (2 VM entries).

Note: the spot VM's "deleted 06:51Z" is the time of its COST-AUDIT line; GCE preempted and deleted it at 06:47:12Z.

- Per seed (3 seeds sharing one L4): theta_real 40,000 steps in 2,553-2,559 s; calibration fit (pools, reference,
  theta_trust, five learned pools, three fitted scalars) 499-517 s; tests 39-43 s; total 3,106-3,114 s
  (`results.json`, `compute`).
- Zero r6e1 VMs at the end: `gcloud --project=GCP_PROJECT compute instances list --filter='name~r6e1'` returned nothing at
  07:49Z and again at 08:28:06Z (a control listing in the same project showed an r6e2 VM, so the query works).

## 9. Audit

Independent audit by a Board investigator subagent (Claude Opus 5.5), 07:50-08:25Z, of this file as it stood before the
corrections below (sha256 7259317b...): `AUDIT-R6-E1.md`, with its scripts and outputs in `audit/`. It did not run the
experiment and changed nothing in it.

- **Verdict: sound, publishable after corrections.** It re-derived every rule from `raw.json` with its own numpy code
  (`audit/rederive.py`, importing none of this experiment's code) and decided all five the same way: E1-D0, E1-D1 and
  E1-D2.1 PASS; E1-D2.2 and E1-D3 FAIL.
- **Numbers.** 462 seed-level values and 105 three-seed statistics were compared. The largest difference, 5.8e-5 (seed 101
  P-FullSupport KL of ataraxos_mmd), lies inside the interval the 5-decimal rounding of the saved beliefs allows; everything
  else agrees within 1.4e-5. Its own exact Bayes matches the saved oracle to 5.0e-6, and the test boards are R5-E1 v2's.
- **Confirmed.** Registration came before the first launch; the staged code is hash-identical to the registered files and
  passes the 6 unit tests; exact Bayes is used only for scoring, the gate and monitor curves; PUCT never uses an unread
  cell's real reading (a flip test on all 4 waves, with a positive control); the parameter counts (2,789 and 108,228); the
  exactly zero union-call JSD; $1.10 by the ledger and $1.09 from GCE operation times; no r6e1 VM at 07:54:53Z, 08:19Z
  or 08:24Z.
- **Corrections applied after the audit** (none changes a verdict):
  - Integrity: the plot-only `analyze6.py` edit made after the results existed is now deviation 3 (section 7 and
    `PREREG-R6-E1.md`); the registered script is kept as `results/analyze6-as-registered.py`; section 11 says the text
    logs are temporary.
  - 14 of 337 hand-checked numbers were wrong at their printed precision (10 rounded twice, 4 seed-101 values printed
    without a seed label). They are fixed: the K4 Pearson CI high bound (0.867), the kc_dedup JSD ratio (2.8x), beta, alpha,
    three key cosines, kc_dedup's JSD (0.0221) and attn_L3-K2's KL (0.037).
  - Wording: section 10 no longer says PUCT reaches 95% coverage or that the remaining disagreement is evidence one search
    never saw. "Learned key crowding redirects", "holds clearly", "KL is lowest", "weight-tied", the run name and
    theta_trust's target are corrected, the wave-1 counting convention is stated, and three mechanism sentences are marked
    as interpretation.
  - Not in the audit's list: I also removed "its crowding deduplicates well" (section 1), which no number here supports,
    and softened section 10's opening from "now holds" to "is mostly supported".
- **Other readings of three clauses** (sensitivity only; the registered definitions govern and no verdict changes):
  - counting the 2 root anchor cells as covered gives PUCT coverage 0.9531 (would pass);
  - counting only reads wholly in the NW half gives a wave-1 share of 0.666 (would fail);
  - the MSE cut vs kc_dedup at tau 2.5 is 0.104 as the mean of per-N cuts and 0.069 on A* cells (fail), and 0.157 at
    N = 16 alone (would pass); the MSE cut vs ataraxos_mmd at tau 0.3 as the mean of per-N cuts is 0.353 (bar 0.35).
- **Not checkable by the auditor:**
  - the clause-1 Pearson and some secondary statistics, which `raw.json` does not keep (taken from `summary.json`);
  - that the training stream is disjoint from the test stream (by construction only);
  - that no amendment file existed at 06:28Z and 07:48Z (it could check only that none exists now);
  - the agent-mail times;
  - actual billing (the Billing API is disabled).

## 10. What this means, and what could come next (not run)

- **The space-of-information picture is mostly supported in this testbed** (3 of 5 rules pass). Three things carry it:
  - Keys built from where a read moved the belief (no patch index) track graded overlap (0.86-0.88).
  - A 3-layer attention pool over such reads gives nearly the same belief whichever way the search was skewed, once both
    searches see the same evidence (JSD 0.008, 2.8-6.5x below the one-step pools), and attn_L3 (with any key) is the most
    accurate pool family.
  - The same pool deduplicates identical reads and adds support for independent noisy ones, with the support growing with
    noise.
- **What is left.**
  - Under open-ended search, policy independence is limited by what each search reads. Key-space PUCT reaches 94.9%
    coverage in 4 waves (bar 95%, missed). After it, attn_L3's JSD (0.0131) is about 3x the clean union call's on the same
    read cells (0.0041), so evidence one search never saw is not the main remaining source of disagreement.
  - attn_L3's margin over a one-step pool that shares the improved keys is smaller at high noise (10%).
  - A cheap looped linear-attention pool is not yet a substitute: KL 2.0x attn_L3's.
- **Possible next steps** (for gemini's next round; none was run):
  1. PUCT with 5 waves, or a coverage-aware stopping rule, to test whether the remaining 5% matters.
  2. kc_dedup with K1 keys next to K4, to measure how much of attn_L3's round-5 margin was really the keys.
  3. A looped_lin variant with more steps, or a separate key head or G for its last step.

## 11. Files

- Code: `e1v3.py` (protocols, key variants, looped_lin_T4, PUCT), `analyze6.py`, `test_r6e1.py` (6 tests), `ledger6.py`,
  `stage6.sh`, `remote6.sh`, `driver6.sh`, `launch6.sh`; imported unchanged from `li-net/r5e1/`: `env6.py`, `wm.py`,
  `rollouts.py`, `run.py`, `partA2.py`; publishing: `render_page6.py`, `site_patch6.py`, `site-entry.json`.
- Pre-registration: `PREREG-R6-E1.md` (+ `.sha256`, `results/PREREG-R6-E1-as-registered.md`); staged hashes per launch:
  `results/stage-manifests.txt`.
- Raw results (JSON only came back from the VM): `results/main/s{101,102,103}/{summary.json, raw.json, rc.json}`. Derived:
  `results/main/analysis/{results.json, verdicts.md, keys.csv, keys.png, policy_independence.csv, policy_independence.png,
  controlled_bags.csv, controlled_bags.png}`. The registered analysis script: `results/analyze6-as-registered.py`
  (deviation 3). Text logs: `/tmp/r6e1/logs/`, temporary and not kept (the assignment allows only CSV/JSON/PNG back from
  the VM).
- Audit: `AUDIT-R6-E1.md`, `audit/`. Page: `/home/agent/memo-serve/lir-agent/r6-e1.html`. GPU ledger: `GPU-LOG.md`.
