# Does attention deduplicate? Set-style PFN, depth 1–8 (opus-mats worker, 2026-10-27)

Kelvin (10-26): "I somehow expected attention to be responsible for much of the deduplication; this will show."

## What survives (2026-10-28, after review 2)

- **Every trained model deduplicates partly from within-episode evidence. None does it fully.** In distribution it
  removes 0.25–0.70 of an exact copy; Bayes removes all of it. A reader with no within-episode comparison removes
  0.01 (section 0, table A).
- **Without keys, the cue is mostly the exact tie of the copy's query reading.** Removing the tie halves the effect;
  identity vectors and labelled-value equality barely matter (table B). Depth 1 does this as well as or better than
  depths 2–8 (0.36 vs 0.25–0.28), so depth adds nothing here.
- **With keys, the cue is the shared group key, and depth helps.** A copy with a new key is barely discounted
  (0.02–0.12). Depth 1 removes 0.49 of a copy, depths 2–8 0.62–0.70.
- **On shared errors, deeper models with keys remove 85–88% of the error left by a reader that compares query values
  only (R2); without keys, 36–44%; depth 1 without keys, 3%.** Most of the improvement over count-all is plain
  shrinkage (60–79%), which any reader learns, so that normalisation understates the in-context part (table C).
- **The no-keys ceiling is largely weak signal (Track A2, end of report).** With copies made common and repeated,
  no-keys deduplication rises: depth 1 0.36 → 0.61, depth 2 0.26 → 0.61, depth 4 0.25 → 0.77. Depths 2 and 4 start
  to use source identity, and depth 4 beats depth 2 in every seed.
- **Withdrawn:** the v4 claims "depth 1 without keys is prior-level only (prediction confirmed)", "depths 2–8 dedup
  about halfway by labelled value equality", "69% at depth 1 with keys" and "shared errors mostly prior-level". The
  v4 copy-vs-independent test kept an exact query tie in the "independent" row, and a tie is copy evidence. The
  preregistered prediction "depth 1 ≈ prior level" **failed**: depth 1 beats R1 on every split and removes 0.36 of a
  copy (R1: 0.01).

## Answer (v5, 2026-10-28)

The question was whether the deduplication is prior-level (shrinkage and row discounting learned from the training
distribution) or in-context (this episode's evidence that two rows are one source, or share an error). Measured
in distribution:

| | depth 1 | depths 2–8 |
|---|---|---|
| copy removed in context, no keys | 0.36 [0.24, 0.47] | 0.25–0.28 |
| copy removed in context, keys | 0.49 [0.45, 0.52] | 0.62–0.70 |
| main cue, no keys | exact query tie | exact query tie |
| main cue, keys | group key | group key |
| R2's shared-error loss removed, no keys | 0.03 | 0.36–0.44 |
| R2's shared-error loss removed, keys | 0.60 | 0.85–0.88 |

- **Without keys, depth helps shared errors but not copies.** Depth 1 handles shared errors only at the level of a
  query-value comparison (R2). Depths 2–8 use labelled residuals for another 36–44%. On copies, depth 1 does as well
  as or better than depths 2–8, all relying on the exact query tie.
- **Keys make both problems easier, and depth then helps both.**
- **Where it happens.** MLPs act per token, but the query's MLP acts on the attention-pooled vector. So depth 1 is a
  "pool then nonlinearity" reader, and it can detect a tie among pooled readings. Saying that depth 1 does only
  prior-level work in its readout was wrong.
- **Convergence:** 15/24 runs converged by the prereg rule; depth 2 and depth 8 without keys, and depth 8 with keys,
  are 1/3 each. Converged and unconverged seeds of the same arm agree (seed ranges in table A).
- **Statistics:** CIs are from a bootstrap over episodes with the seeds fixed; seed ranges are shown separately. The
  three-seed spread is small next to the gaps between arms that the headlines rely on.

<details><summary>Superseded headlines (v4, 2026-10-27 ~07:30Z; prior-level vs in-context split on the tied test, withdrawn after review 2)</summary>

Two different things lower a model's error compared with counting every reading:
- **prior-level discounting**: shrinkage, and down-weighting extra rows, learned from the training distribution. A
  readout can do this without comparing tokens within the episode.
- **in-context aggregation**: using this episode's evidence that two rows are one source, or share an error.

The cleanest test adds one row to an episode in one of two forms: a true copy of a source, or an independent new
source with the same query reading (table A). A reader without within-episode comparison treats the two identically.

- **Depth 1 without keys is prior-level only.** A copy pulls the prediction 0.049, an independent source 0.054: it
  barely tells them apart (9%). On shared-error and singleton episodes its error sits at the level of R2, Bayes on
  the query values alone with no identities or labels (0.029 vs 0.030; 0.027 vs 0.023–0.029). Prediction confirmed.
- **Depths 2–8 without keys deduplicate in context, about halfway.** A copy pulls 0.046–0.048, an independent source
  0.087–0.093, so the models discount about half of a copy beyond the prior (44–49%). They use value equality across
  the labelled questions, not the source identity vector: a copy with a new identity behaves exactly like a copy.
  Depth beyond 2 adds nothing.
- **With keys, in-context deduplication is strong at every depth** (69% at depth 1, 80–83% at depths 2–8). The copy
  shares its source's group key, which is direct in-context evidence.
- **Shared errors are mostly prior-level.** Of each model's MSE gain over count-all on shared-error episodes, 60–79%
  is reproduced by R1, the best additive pool with no comparisons, and 75–99% by R2. What is left in context:
  - without keys: depth 1 0.01, depths 2–8 0.11–0.14 beyond R2;
  - with keys: depth 1 0.19, depths 2–8 0.24–0.25 beyond R2.
- **No model deduplicates fully.** Bayes ignores a copy (pull 0); the best models still give it a pull of 0.026.
- **Attention is necessarily the carrier,** since MLPs act per token. The in-context part rides on the group keys or
  on comparing labelled values across rows. At depth 1 without keys, readout attention does only prior-level work.

Caveat for table A: the independent-source variant gives an episode 7 distinct sources, which never occurs in
training. The duplicate-excess version (table C) is confounded: R1, being additive, cannot express softmax
normalisation, which is itself a prior-level discount. Table C credits depth 1 with 0.68 "in-context" on duplicates;
table A shows that this is prior-level row discounting. Table A is the test to trust.

</details>

<details><summary>Superseded headlines (v3, 2026-10-27 06:30Z; before the prior-level/in-context split)</summary>

- **Attention does all of the deduplication there is, and the architecture forces that.** MLPs act per token and
  there is no positional encoding, so only attention can see that two rows are the same. The testable question is
  which attention. The answer depends on whether the partition is visible (keys).
- **At depth 1, readout attention alone gets a large part of the discount.** With one copy (m = 2, the only copy
  count seen in training), the pull toward the duplicated value is 0.049 without keys and 0.040 with keys. For
  reference: true duplicates 0, count-all 0.121, an extra independent reading 0.176. At depth 1 a copy's attention
  score equals the original's, so the discount comes from softmax normalisation. The trained query puts more mass on
  the duplicated source's whole row, so copies take mass from each other: copy-mass ratio 1.50, against 1.72
  untrained.
- **Without keys, depth adds nothing on duplicates, and deeper readouts never lower the copies' scores.**
  - The pull at m = 2 is 0.046–0.048 at depths 2, 4 and 8, against 0.049 at depth 1. The duplicate MSE excess is
    0.56–0.64 of count-all's at depths 2–8, against 0.45 at depth 1.
  - The readout mass on the copies is what identical scores give, within 0–6% at depths 2–4 and within 10% at depth 8.
  - Deeper models still use context-mixed state. Patching after layer 0 at m = 2 raises the pull from 0.046–0.048 to
    0.06–0.07, whether the copies' state or the *rest* of the context is replaced. Replacing both gives 0.076–0.094; a
    random patch of the same norm does nothing.
  - That state is spread over the whole context, not held in the copies, and its net effect leaves the pull no lower
    than at depth 1.
- **With keys (group identity visible), depth helps and the readout scores do target copies.**
  - The pull at m = 2 falls from 0.040 at depth 1 to 0.026–0.031 at depths 2–8. The duplicate excess falls from 0.44
    to 0.18–0.27 of count-all's.
  - At layers ≥ 1 the scores remove 20–77% of the copies' extra mass: depth 2 0.41, depth 4 0.35–0.58, depth 8 up to
    0.75–0.77 in layers 2–3.
  - The copies' own state matters more than the rest of the context: at m = 2, patching the copies gives a pull of
    0.067–0.076, the rest 0.043–0.056.
  - On shared-error episodes, per-source readout attention tracks the GLS weights: corr 0.86 at depth 2 (last layer)
    and 0.78 at layer 2 of depth 4 (0.59 at its last layer), against ≈ 0 untrained.
- **No model deduplicates fully.** Every trained model keeps a pull of 0.026–0.049 at m = 2. That is 21–40% of
  count-all's pull, or about 25–50% of the reviewer's no-dedup baseline measured on the models themselves (≈ 0.10).
  The prereg duplicate level (L1, ≥ 75% of the NLL gap closed) is reached by no arm; the best is 0.72, at depths 4 and
  8 with keys.
- **Shared errors are handled at every depth, depth 1 included.** Depth 1 closes 88% (no keys) and 95% (keys) of the
  gap between iid-distinct and Bayes on shared-error episodes, and the MSE to the Bayes mean agrees. Predictions 1
  and 3 failed.

</details>

## Status, caveats (read first)
- All 8 arms (depth 1/2/4/8 × keys 0/1, 3 seeds each) are trained to 40k steps; 15/24 converged by the prereg rule.
  - Depths 1 and 2 come from pfndedup-1026c.
  - Depth 8 without keys comes from pfndedup-1027c.
  - Depth 4 (both arms) and depth 8 with keys come from the coordinator's on-demand relaunch, pfndedup-1027e/f/g. It
    replaces the 1026c depth-4 models, which had timed out at 10k–35k steps. Those old models stay in runs_1026c/ and
    are not reported.
- The prereg L1 test (NLL, duplicates vs count-all) is weak: count-all's own gap there is only 0.074 nats, and the
  model NLL mixes deduplication with variance calibration. The mean-based duplicate excess and the copy sweep below
  measure deduplication directly.
- The copy sweep goes to m = 8, but training never had more than 2 rows per source (k ≤ 3 *distinct* sources
  copied once). m ≥ 3 is out of distribution; m = 2 is the in-distribution number.
- Copy-sweep, attention and patching results are 3 seeds per arm. Probes and ablation use 3 seeds at depths 1–4;
  at depth 8 the ablation is seed 0 only, and depth 8 is missing from the probe table. Every probe is retrained per
  layer (ridge, episode-level split, inner-CV λ), with untrained (same init seed) and shuffled-target baselines.
- **Cost: about $10.9, against a $6 cap.**
  - My runs 1026/1026b/1026c: $3.71.
  - 1027a–d: 3.6 VM-hours of spot, logged at the on-demand ceiling of $0.87/h, so $3.1. Real spot billing is lower;
    I could not check it because the Billing API is disabled.
  - The coordinator's 1027e/f/g: $4.04.
  - All VMs are deleted: `gcloud compute instances list --filter=name~pfndedup` returned nothing at 06:00Z.
- The adversarial reviewer saw only the timed-out depth-4 models and the 6k-step depth-8 models. The converged models
  agree with the revised headlines without keys. With keys they strengthen the "readout scores target copies" claim
  and change "depth does not help duplicates" to "depth helps with keys only".

- Section 0 was rebuilt after review 2 (outbox/pfn-dedup-review-2-2026-10-28.md). The in-distribution test
  (split_indist.py) follows the reviewer's method in an independent implementation, on different episodes (seed
  78000 vs 77000). Seed-mean results agree with the reviewer's within the CIs: no keys depth 1 0.36 vs 0.44,
  depths 2–8 0.25–0.28 vs 0.26–0.32; keys 0.49 / 0.62–0.70 vs 0.49 / 0.64–0.73. R2's ratio is unstable
  (100-episode blocks give 2 − rho from −0.43 to 0.33), so R2 is not a precise reference for table A.

## 0. Prior-level vs in-context split (rebuilt 2026-10-28 after review 2)

R1 is the best additive pool, mu = a_n + sum_r g_n(o_r): it knows the row count and each query token's value and never compares tokens. It is the prior-level reference. R2 is exact Bayes on the query readings alone (no identities, no labelled residuals). It compares query values, so it is *not* prior-level: R1 → R2 is the query-value-comparison tier. Both are fitted per row count, so they know how many rows there are, though not which ones are copies.

### A. In-distribution test: how much of an exact copy does each model remove using within-episode evidence?

600 episodes with 6 singleton sources plus one exact copy of source s0 (seed 78000). rho = (d mu / d o* for s0, moving both its rows) / (d mu / d o* for another source s1 in the same episode). A reader that cannot tell which rows are copies has rho = 2; full deduplication has rho = 1. In-context dedup = 2 − rho. Mean over 3 seeds; 95% CI from a bootstrap over episodes (seeds fixed); seed range in brackets. conv = seeds meeting the preregistered convergence rule.

References: R1 (additive, no comparison) 2 − rho = 0.01; R2 (Bayes on query readings, no identities or labels) 0.11; Bayes 1.00.

| depth | keys | conv | in-context dedup 2 − rho | 95% CI | seed range | copy pull | untied independent pull (Bayes 0.121) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3/3 | 0.36 | [0.24, 0.47] | 0.30..0.40 | 0.059 | 0.105 |
| 1 | 1 | 2/3 | 0.49 | [0.45, 0.52] | 0.40..0.57 | 0.061 | 0.117 |
| 2 | 0 | 1/3 | 0.26 | [0.19, 0.32] | 0.23..0.29 | 0.080 | 0.117 |
| 2 | 1 | 2/3 | 0.66 | [0.64, 0.67] | 0.64..0.67 | 0.042 | 0.121 |
| 4 | 0 | 3/3 | 0.25 | [0.19, 0.31] | 0.23..0.29 | 0.086 | 0.119 |
| 4 | 1 | 2/3 | 0.70 | [0.68, 0.72] | 0.66..0.75 | 0.036 | 0.120 |
| 8 | 0 | 1/3 | 0.28 | [0.21, 0.34] | 0.26..0.29 | 0.081 | 0.118 |
| 8 | 1 | 1/3 | 0.62 | [0.60, 0.64] | 0.55..0.67 | 0.047 | 0.119 |

### B. Which cue does the deduplication use? Remove one cue from the copy at a time (out of distribution)

In-context dedup 2 − rho for an exact copy and for copies with one cue removed: the exact query tie (copy's query reading offset by delta ~ N(0, 0.5)); equality of the 8 labelled readings (fresh values); identity vector and group key (new ones). A large drop means that cue is used. These copies never occur in training.

| depth | keys | exact copy | no query tie | no labelled equality | new identity and key |
|---|---|---|---|---|---|
| R1 | | 0.01 | -0.00 | 0.01 | 0.01 |
| R2 | | 0.11 | -0.03 | 0.11 | 0.11 |
| 1 | 0 | 0.36 | 0.16 | 0.35 | 0.37 |
| 1 | 1 | 0.49 | 0.43 | 0.49 | 0.12 |
| 2 | 0 | 0.26 | 0.13 | 0.21 | 0.26 |
| 2 | 1 | 0.66 | 0.64 | 0.66 | 0.02 |
| 4 | 0 | 0.25 | 0.10 | 0.21 | 0.24 |
| 4 | 1 | 0.70 | 0.64 | 0.70 | 0.08 |
| 8 | 0 | 0.28 | 0.12 | 0.25 | 0.27 |
| 8 | 1 | 0.62 | 0.57 | 0.62 | 0.09 |

### C. Shared errors in three tiers (preregistered split, corrected reading)

MSE of the predictive mean to the arm's Bayes mean on shared-error episodes without duplicates (n = 3915), mean over 3 seeds. Tiers: prior (R1, no comparison), query-value comparison (R1 → R2), labelled/identity/key evidence (beyond R2). Two normalisations: the preregistered share of the model's gain over count-all, and the share of R's remaining error that the model removes. R1 and R2 are fitted per row count n (here n = 6 always).

| depth | keys | conv | MSE model | count-all | R1 | R2 | gain over count-all: prior / query-value / beyond R2 | R1's error removed | R2's error removed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3/3 | 0.0289 | 0.1134 | 0.0467 | 0.0298 | 0.79 / 0.20 / 0.01 | 0.38 | 0.03 |
| 1 | 1 | 2/3 | 0.0127 | 0.1168 | 0.0495 | 0.0320 | 0.65 / 0.17 / 0.19 | 0.74 | 0.60 |
| 2 | 0 | 1/3 | 0.0191 | 0.1134 | 0.0467 | 0.0298 | 0.71 / 0.18 / 0.11 | 0.59 | 0.36 |
| 2 | 1 | 2/3 | 0.0047 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.24 | 0.91 | 0.85 |
| 4 | 0 | 3/3 | 0.0166 | 0.1134 | 0.0467 | 0.0298 | 0.69 / 0.17 / 0.14 | 0.64 | 0.44 |
| 4 | 1 | 2/3 | 0.0048 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.24 | 0.90 | 0.85 |
| 8 | 0 | 1/3 | 0.0171 | 0.1134 | 0.0467 | 0.0298 | 0.69 / 0.17 / 0.13 | 0.63 | 0.43 |
| 8 | 1 | 1/3 | 0.0039 | 0.1168 | 0.0495 | 0.0320 | 0.60 / 0.16 / 0.25 | 0.92 | 0.88 |

<details><summary>Withdrawn 2026-10-28: v4 section-0 tables (the 'independent' row kept an exact query tie with o_s0, which is copy evidence)</summary>


R1 is the best additive pool, mu = a_n + sum_r g_n(o_r): it knows the row count and each query token's own value, never compares tokens, and is fitted on 300k fresh training episodes. R2 is exact Bayes given only the query readings, with no identities and no labelled residuals, under a duplicate-blind model in which all rows are distinct sources. A tie is scored like any two readings 0 apart, so R2 cannot use exact value equality as copy identity, though it does compare values. R2eq drops exact duplicate values first, so it leaks copy identity; it is shown only as a bracket. Code: split.py, split_copy.py, split_tables.py.

### A. The decisive test: one extra row that is a copy vs an independent source with the same query reading

Pull toward the duplicated reading o_s0 when one row is added to a 6-source episode (600 episodes; 3 seeds). (i) true copy; (ii) same values on all 9 questions, new identity; (iv) new independent source (fresh labelled readings) with the same query reading. A reference with no within-episode comparison gives the same pull for all three: R1 0.076, R2 0.061. Bayes: copy 0, independent 0.176. In-context dedup = 1 − pull(i)/pull(iv): 0 = the model cannot tell a copy from an independent source.

| depth | keys | (i) copy | (ii) new identity, same values | (iv) independent source | in-context dedup 1 − (i)/(iv) |
|---|---|---|---|---|---|
| 1 | 0 | 0.049 | 0.050 | 0.054 | 0.09 [0.07..0.10] |
| 1 | 1 | 0.040 | 0.040 | 0.128 | 0.69 [0.63..0.73] |
| 2 | 0 | 0.048 | 0.048 | 0.087 | 0.44 [0.41..0.50] |
| 2 | 1 | 0.030 | 0.031 | 0.160 | 0.81 [0.81..0.82] |
| 4 | 0 | 0.047 | 0.048 | 0.093 | 0.49 [0.47..0.52] |
| 4 | 1 | 0.026 | 0.026 | 0.154 | 0.83 [0.81..0.86] |
| 8 | 0 | 0.046 | 0.047 | 0.090 | 0.49 [0.44..0.54] |
| 8 | 1 | 0.031 | 0.031 | 0.157 | 0.80 [0.78..0.82] |

### B. Preregistered split: share of each model's MSE improvement over count-all that is beyond R1 / R2

MSE of the predictive mean to the arm's Bayes mean, mean over 3 seeds. In-context share beyond R = (E_R − E_model)/(E_count-all − E_model); prior-level share = 1 − that. Shares only where the model improves on count-all by > 0.005 (on dup_singletons and singletons_nodup count-all's iid assumption is nearly exact, so 'improvement over count-all' is ill-posed there; see C).


keys = 0. References: dup_singletons: count-all 0.0191, R1 0.0445, R2 0.0370; shared_nodup: count-all 0.1134, R1 0.0467, R2 0.0298; singletons_nodup: count-all 0.0007, R1 0.0294, R2 0.0234

| depth | MSE shared_nodup | in-context beyond R1 | beyond R2 | MSE singletons_nodup (R1, R2 above) | MSE dup_singletons |
|---|---|---|---|---|---|
| 1 | 0.0289 | 0.21 | 0.01 | 0.0266 | 0.0350 |
| 2 | 0.0191 | 0.29 | 0.11 | 0.0073 | 0.0190 |
| 4 | 0.0166 | 0.31 | 0.14 | 0.0040 | 0.0146 |
| 8 | 0.0171 | 0.31 | 0.13 | 0.0049 | 0.0151 |

keys = 1. References: dup_singletons: count-all 0.0180, R1 0.0457, R2 0.0380; shared_nodup: count-all 0.1168, R1 0.0495, R2 0.0320; singletons_nodup: count-all 0.0000, R1 0.0302, R2 0.0244

| depth | MSE shared_nodup | in-context beyond R1 | beyond R2 | MSE singletons_nodup (R1, R2 above) | MSE dup_singletons |
|---|---|---|---|---|---|
| 1 | 0.0127 | 0.35 | 0.19 | 0.0077 | 0.0156 |
| 2 | 0.0047 | 0.40 | 0.24 | 0.0031 | 0.0065 |
| 4 | 0.0048 | 0.40 | 0.24 | 0.0029 | 0.0062 |
| 8 | 0.0039 | 0.40 | 0.25 | 0.0021 | 0.0068 |

### C. Duplicate excess (post-hoc, PREREG addendum): MSE(dup_singletons) − MSE(singletons_nodup)

References (no keys): count-all 0.0184, R1 0.0152, R2 0.0136, R2eq (drops exact equal values: leaks copy identity) 0.0017, Bayes 0. In-context share = (excess_R1 − excess_model)/(excess_count-all − excess_model).

| depth | keys | model excess | in-context share beyond R1 |
|---|---|---|---|
| 1 | 0 | 0.0084 | 0.68 |
| 1 | 1 | 0.0079 | 0.75 |
| 2 | 0 | 0.0117 | 0.52 |
| 2 | 1 | 0.0033 | 0.83 |
| 4 | 0 | 0.0106 | 0.59 |
| 4 | 1 | 0.0033 | 0.83 |
| 8 | 0 | 0.0103 | 0.60 |
| 8 | 1 | 0.0048 | 0.81 |


</details>

## 1. Accuracy by depth and redundancy type (NLL gap to the arm's exact Bayes, nats, mean ± se over seeds)
Splits: duplicate episodes with all-singleton partitions; all-singleton episodes without duplicates; largest group of
size 2, 3, ≥ 4; any shared group. The test set has 8000 episodes. Bayes is the exact posterior over 203 partitions
(no keys) or GLS with the true partition (keys).

## keys = 0

| depth | conv | all | dup_singletons | singletons_nodup | group2 | group3 | group4plus | shared_any |
|---|---|---|---|---|---|---|---|---|
| 1 | 3/3 | 0.0685 ± 0.0015 | 0.0989 ± 0.0066 | 0.0876 ± 0.0043 | 0.0369 ± 0.0048 | 0.0398 ± 0.0019 | 0.0703 ± 0.0024 | 0.0586 ± 0.0005 |
| 2 | 1/3 | 0.0339 ± 0.0053 | 0.0474 ± 0.0090 | 0.0238 ± 0.0091 | 0.0231 ± 0.0013 | 0.0393 ± 0.0013 | 0.0356 ± 0.0068 | 0.0353 ± 0.0044 |
| 4 | 3/3 | 0.0254 ± 0.0011 | 0.0309 ± 0.0043 | 0.0124 ± 0.0019 | 0.0228 ± 0.0012 | 0.0361 ± 0.0020 | 0.0265 ± 0.0010 | 0.0287 ± 0.0008 |
| 8 | 1/3 | 0.0276 ± 0.0008 | 0.0370 ± 0.0016 | 0.0180 ± 0.0032 | 0.0197 ± 0.0028 | 0.0340 ± 0.0017 | 0.0290 ± 0.0021 | 0.0294 ± 0.0012 |
| ref (a) count-all | | 0.4391 | 0.0739 | -0.0000 | 0.1041 | 0.2710 | 0.8615 | 0.6234 |
| ref (b) iid-distinct | | 0.3515 | -0.0047 | -0.0000 | 0.0549 | 0.1907 | 0.7190 | 0.5073 |
| ref (c) GLS-oracle | | -0.0046 | -0.0047 | -0.0000 | -0.0132 | -0.0102 | -0.0031 | -0.0060 |

## keys = 1

| depth | conv | all | dup_singletons | singletons_nodup | group2 | group3 | group4plus | shared_any |
|---|---|---|---|---|---|---|---|---|
| 1 | 2/3 | 0.0263 ± 0.0008 | 0.0465 ± 0.0044 | 0.0218 ± 0.0024 | 0.0295 ± 0.0044 | 0.0266 ± 0.0018 | 0.0238 ± 0.0012 | 0.0251 ± 0.0015 |
| 2 | 2/3 | 0.0124 ± 0.0005 | 0.0255 ± 0.0021 | 0.0070 ± 0.0014 | 0.0156 ± 0.0044 | 0.0137 ± 0.0025 | 0.0112 ± 0.0005 | 0.0123 ± 0.0006 |
| 4 | 2/3 | 0.0118 ± 0.0008 | 0.0220 ± 0.0027 | 0.0102 ± 0.0011 | 0.0185 ± 0.0025 | 0.0146 ± 0.0002 | 0.0083 ± 0.0010 | 0.0110 ± 0.0005 |
| 8 | 1/3 | 0.0089 ± 0.0001 | 0.0217 ± 0.0045 | 0.0013 ± 0.0010 | 0.0153 ± 0.0017 | 0.0111 ± 0.0011 | 0.0081 ± 0.0013 | 0.0096 ± 0.0007 |
| ref (a) count-all | | 0.4437 | 0.0787 | -0.0000 | 0.1173 | 0.2812 | 0.8646 | 0.6294 |
| ref (b) iid-distinct | | 0.3561 | -0.0000 | 0.0000 | 0.0681 | 0.2009 | 0.7220 | 0.5134 |

## Levels (prereg): fraction of reference gap closed, mean over seeds

| depth | keys | L1 dup (vs a) | L2 singletons gap | L3 shared (vs b) |
|---|---|---|---|---|
| 1 | 0 | -0.34 | 0.0876 | 0.88 ✓ |
| 1 | 1 | 0.41 | 0.0218 | 0.95 ✓ |
| 2 | 0 | 0.36 | 0.0238 | 0.93 ✓ |
| 2 | 1 | 0.68 | 0.0070 ✓ | 0.98 ✓ |
| 4 | 0 | 0.58 | 0.0124 ✓ | 0.94 ✓ |
| 4 | 1 | 0.72 | 0.0102 ✓ | 0.98 ✓ |
| 8 | 0 | 0.50 | 0.0180 ✓ | 0.94 ✓ |
| 8 | 1 | 0.72 | 0.0013 ✓ | 0.98 ✓ |

### Context-to-context ablation

Context-to-context ablation, mean over 3 seeds (depth 8: seed 0 only). Gap on the 2000-episode mechanism set with
context tokens attending only to themselves at one layer. Ablating the last layer is a no-op, so it is the baseline.

| depth | keys | ablate layer 0, 1, … (last = no-op baseline) |
|---|---|---|
| 2 | 0 | 0.226, 0.038 |
| 2 | 1 | 0.176, 0.013 |
| 4 | 0 | 0.223, 0.214, 0.155, 0.033 |
| 4 | 1 | 0.170, 0.044, 0.023, 0.013 |
| 8 | 0 | 0.259, 0.173, 0.105, 0.048, 0.048, 0.029, 0.029, 0.029 |
| 8 | 1 | 0.141, 0.019, 0.025, 0.015, 0.018, 0.014, 0.020, 0.013 |

Context mixing in the first layers carries most of the gain over depth 1, on all episode types (these gaps are over
all test episodes, most of which have no duplicates).

## 2. Mechanism: the copy sweep and baselines (coordinator's eval-only addition)
600 episodes without duplicates (seed 40000). One random source s0 is copied to m = 1..8 rows (same identity, same
values on all 9 questions); nothing else changes. Code: mech2.py, mech3.py, analysis2.py; tables by make_tables.py.

### Predictive mean vs the Bayes mean (MSE; separates deduplication from variance calibration)

Duplicate excess = MSE on duplicate episodes with all-singleton partitions minus MSE on all-singleton episodes without duplicates. Count-all (a) has excess ≈ 0.018; Bayes has 0. Fraction = model excess / (a) excess.

| depth | keys | n seeds | conv | MSE singletons, no dup | MSE dup | duplicate excess | fraction of (a) | MSE shared_any (a: 0.126, b: 0.116) | untrained MSE shared_any |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | 3/3 | 0.027 ± 0.003 | 0.035 ± 0.002 | 0.008 ± 0.001 | 0.45 | 0.030 ± 0.000 | 0.85 |
| 1 | 1 | 3 | 2/3 | 0.008 ± 0.001 | 0.016 ± 0.000 | 0.008 ± 0.001 | 0.44 | 0.013 ± 0.001 | 0.83 |
| 2 | 0 | 3 | 1/3 | 0.007 ± 0.002 | 0.019 ± 0.002 | 0.012 ± 0.000 | 0.64 | 0.020 ± 0.002 | 0.96 |
| 2 | 1 | 3 | 2/3 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.19 | 0.005 ± 0.000 | 1.44 |
| 4 | 0 | 3 | 3/3 | 0.004 ± 0.001 | 0.015 ± 0.001 | 0.011 ± 0.000 | 0.58 | 0.017 ± 0.000 | 0.93 |
| 4 | 1 | 3 | 2/3 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.18 | 0.005 ± 0.001 | 1.02 |
| 8 | 0 | 3 | 1/3 | 0.005 ± 0.001 | 0.015 ± 0.001 | 0.010 ± 0.000 | 0.56 | 0.017 ± 0.000 | 0.91 |
| 8 | 1 | 3 | 1/3 | 0.002 ± 0.000 | 0.007 ± 0.001 | 0.005 ± 0.001 | 0.27 | 0.004 ± 0.000 | 1.08 |

### Probes (retrained ridge per layer) and attention, trained vs untrained (same init seed); mean over seeds

| depth | keys | n | group-size R² by layer, trained | untrained | query → Bayes-mean R², trained | untrained | corr(query attn per source, GLS weight) by layer, trained | untrained |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 3 | -0.00, 0.23 | -0.00, -0.01 | -0.03, 0.95 | -0.03, 0.16 | 0.20 | 0.00 |
| 1 | 1 | 3 | -0.02, 0.58 | -0.02, 0.04 | -0.03, 0.98 | -0.03, 0.13 | 0.35 | 0.03 |
| 2 | 0 | 3 | -0.00, 0.24, 0.61 | -0.00, -0.01, -0.01 | -0.03, 0.93, 0.97 | -0.03, 0.16, 0.17 | 0.09, 0.24 | 0.00, -0.02 |
| 2 | 1 | 3 | -0.02, 0.75, 0.83 | -0.02, 0.04, 0.12 | -0.03, 0.93, 0.99 | -0.03, 0.13, 0.13 | 0.21, 0.86 | 0.03, 0.01 |
| 4 | 0 | 3 | -0.00, 0.25, 0.59, 0.68, 0.69 | -0.00, -0.01, -0.01, -0.02, -0.02 | -0.03, 0.91, 0.96, 0.97, 0.98 | -0.03, 0.16, 0.17, 0.19, 0.19 | 0.09, 0.13, 0.15, 0.30 | 0.00, -0.02, -0.01, 0.02 |
| 4 | 1 | 3 | -0.02, 0.74, 0.85, 0.86, 0.86 | -0.02, 0.04, 0.12, 0.16, 0.19 | -0.03, 0.92, 0.96, 0.99, 0.99 | -0.03, 0.13, 0.13, 0.12, 0.14 | 0.13, 0.58, 0.78, 0.59 | 0.03, 0.01, -0.00, 0.03 |

### Copy sweep: one source's row copied to m rows (m = 1..8; training only ever had m ≤ 2)

Pull = OLS slope of (μ_m − μ_1) on (o_dup − μ_1). True duplicates (Bayes): 0 for every m.

| model | n | m=2 | m=3 | m=4 | m=6 | m=8 |
|---|---|---|---|---|---|---|
| Bayes, true duplicates | | 0 | 0 | 0 | 0 | 0 |
| GLS + independent copies | | 0.176 | 0.296 | 0.384 | 0.506 | 0.587 |
| count-all (a) | | 0.121 | 0.216 | 0.293 | 0.408 | 0.491 |
| L1 keys 0 | 3 | 0.049 | 0.083 | 0.106 | 0.134 | 0.150 |  (untrained m=8: 0.003)
| L1 keys 1 | 3 | 0.040 | 0.062 | 0.075 | 0.090 | 0.096 |  (untrained m=8: 0.005)
| L2 keys 0 | 3 | 0.048 | 0.076 | 0.087 | 0.081 | 0.066 |  (untrained m=8: 0.003)
| L2 keys 1 | 3 | 0.030 | 0.048 | 0.063 | 0.084 | 0.101 |  (untrained m=8: 0.021)
| L4 keys 0 | 3 | 0.047 | 0.074 | 0.082 | 0.061 | 0.034 |  (untrained m=8: 0.011)
| L4 keys 1 | 3 | 0.026 | 0.038 | 0.047 | 0.060 | 0.071 |  (untrained m=8: 0.008)
| L8 keys 0 | 3 | 0.046 | 0.074 | 0.085 | 0.076 | 0.058 |  (untrained m=8: 0.004)
| L8 keys 1 | 3 | 0.031 | 0.044 | 0.053 | 0.068 | 0.081 |  (untrained m=8: 0.004)

Log-precision shift (log σ_1 − log σ_m; positive = more confident). Bayes duplicates 0; independent copies m=2: 0.104, m=4: 0.260, m=8: 0.473.

| model | m=2 | m=4 | m=8 |
|---|---|---|---|
| L1 keys 0 | -0.003 | -0.019 | -0.048 |
| L1 keys 1 | -0.007 | -0.030 | -0.073 |
| L2 keys 0 | -0.009 | -0.050 | -0.118 |
| L2 keys 1 | -0.005 | -0.023 | -0.059 |
| L4 keys 0 | -0.010 | -0.055 | -0.131 |
| L4 keys 1 | -0.006 | -0.027 | -0.058 |
| L8 keys 0 | -0.008 | -0.045 | -0.106 |
| L8 keys 1 | -0.007 | -0.031 | -0.068 |

### Query attention mass on the copies' query-question tokens (sum over copies), last layer

Perfect deduplication by readout attention: total constant in m (per-copy mass ∝ 1/m). Ratio = total(m=2)/total(m=1); per-head ratios in brackets.

| model | total m=1 | m=2 | m=4 | m=8 | ratio m2/m1 (heads) | untrained ratio | min head ratio over all layers |
|---|---|---|---|---|---|---|---|
| L1 keys 0 | 0.119 | 0.179 | 0.253 | 0.334 | 1.50 (1.34, 1.70, 1.45, 1.59) | 1.72 | 1.34 |
| L1 keys 1 | 0.151 | 0.230 | 0.329 | 0.437 | 1.52 (1.45, 1.54, 1.53, 1.60) | 1.71 | 1.45 |
| L2 keys 0 | 0.114 | 0.165 | 0.228 | 0.296 | 1.45 (1.48, 1.36, 1.74, 1.35) | 1.72 | 1.35 |
| L2 keys 1 | 0.124 | 0.170 | 0.226 | 0.288 | 1.37 (1.41, 1.50, 1.15, 1.25) | 1.72 | 1.15 |
| L4 keys 0 | 0.101 | 0.162 | 0.236 | 0.295 | 1.60 (1.66, 1.48, 1.55, 1.65) | 1.72 | 1.21 |
| L4 keys 1 | 0.111 | 0.151 | 0.198 | 0.252 | 1.36 (1.27, 1.48, 1.35, 1.40) | 1.72 | 1.07 |
| L8 keys 0 | 0.099 | 0.167 | 0.252 | 0.328 | 1.70 (1.71, 1.70, 1.71, 1.67) | 1.72 | 1.30 |
| L8 keys 1 | 0.144 | 0.220 | 0.312 | 0.420 | 1.53 (1.55, 1.50, 1.51, 1.57) | 1.72 | 1.00 |

### Is the copy count linearly decodable from the copies? (retrained ridge, R² for log m, by layer; 0 = input)

| model | trained, copies' states | untrained | shuffled target | trained, within-episode h_m − h_1 | untrained within |
|---|---|---|---|---|---|
| L1 keys 0 | -0.00, 0.03 | -0.00, -0.00 | -0.00, -0.00 | 0.00, 0.15 | 0.00, 0.08 |
| L1 keys 1 | -0.00, 0.02 | -0.00, -0.01 | -0.00, -0.01 | 0.00, 0.09 | 0.00, 0.07 |
| L2 keys 0 | -0.00, 0.03, 0.15 | -0.00, -0.00, 0.01 | -0.00, -0.01, -0.01 | 0.00, 0.24, 0.42 | 0.00, 0.08, 0.13 |
| L2 keys 1 | -0.00, 0.20, 0.27 | -0.00, -0.01, -0.02 | -0.00, -0.01, -0.01 | 0.00, 0.10, 0.26 | 0.00, 0.07, 0.04 |
| L4 keys 0 | -0.00, 0.03, 0.17, 0.26, 0.28 | -0.00, -0.00, 0.01, 0.02, 0.02 | -0.00, -0.00, -0.00, -0.01, -0.01 | 0.00, 0.14, 0.37, 0.45, 0.45 | 0.00, 0.08, 0.13, 0.17, 0.16 |
| L4 keys 1 | -0.00, 0.20, 0.27, 0.28, 0.28 | -0.00, -0.01, -0.02, -0.01, -0.01 | -0.00, -0.01, -0.01, -0.01, -0.01 | 0.00, 0.11, 0.29, 0.29, 0.29 | 0.00, 0.07, 0.04, 0.02, 0.05 |
| L8 keys 0 | -0.00, 0.03, 0.13, 0.19, 0.22, 0.23, 0.23, 0.23, 0.23 | -0.00, -0.00, 0.01, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00 | -0.00, -0.00, -0.01, -0.00, -0.01, -0.01, -0.01, -0.01, -0.01 | 0.00, 0.14, 0.32, 0.40, 0.43, 0.43, 0.43, 0.43, 0.43 | 0.00, 0.08, 0.13, 0.17, 0.16, 0.16, 0.16, 0.17, 0.16 |
| L8 keys 1 | -0.00, 0.20, 0.27, 0.28, 0.28, 0.27, 0.27, 0.28, 0.27 | -0.00, -0.01, -0.02, -0.01, -0.01, 0.00, -0.00, -0.01, -0.01 | -0.00, -0.01, -0.01, -0.00, -0.01, -0.01, -0.01, -0.01, -0.01 | 0.00, 0.08, 0.28, 0.31, 0.31, 0.31, 0.32, 0.32, 0.34 | 0.00, 0.07, 0.04, 0.02, 0.05, 0.05, 0.01, 0.02, 0.01 |

### Intervention: m = 4 copies, copies' residual after layer l replaced by the m = 1 state of the original token (superseded by the controls table below)

If the copies' rewritten state carries the discount, the pull rises toward the independent-copies value (0.384) when patched early. Patching after the last layer is a no-op control.

| model | unpatched pull m=4 | patched after layer 0, 1, ... |
|---|---|---|
| L1 keys 0 | 0.106 | 0.11 |
| L1 keys 1 | 0.075 | 0.08 |
| L2 keys 0 | 0.087 | 0.16, 0.09 |
| L2 keys 1 | 0.063 | 0.17, 0.06 |
| L4 keys 0 | 0.082 | 0.16, 0.14, 0.11, 0.08 |
| L4 keys 1 | 0.047 | 0.16, 0.14, 0.09, 0.05 |
| L8 keys 0 | 0.085 | 0.16, 0.14, 0.12, 0.12, 0.11, 0.11, 0.09, 0.08 |
| L8 keys 1 | 0.053 | 0.16, 0.16, 0.13, 0.11, 0.11, 0.10, 0.10, 0.05 |

### Do the query's attention scores deduplicate copies beyond softmax arithmetic?

Pooled over heads per layer: (arith − observed) / (arith − q1) for the total mass on the copies, where arith = m·q1/(1+(m−1)·P1) is what identical scores give and q1 is what perfect score-based deduplication gives. 0 = no deduplication in the scores, 1 = full. Mean over seeds (range in brackets). Depth 1 is 0 by construction.

| model | m | layer 0, 1, ... trained | untrained (max abs) |
|---|---|---|---|
| L1 keys 0 | 2 | 0.00 [-0.00..0.00] | 0.00 |
| L1 keys 0 | 4 | 0.00 [-0.00..0.00] | 0.00 |
| L1 keys 1 | 2 | 0.00 [0.00..0.00] | 0.00 |
| L1 keys 1 | 4 | -0.00 [-0.00..0.00] | 0.00 |
| L2 keys 0 | 2 | 0.00 [0.00..0.00], 0.00 [0.00..0.00] | 0.00 |
| L2 keys 0 | 4 | -0.00 [-0.00..0.00], 0.01 [0.00..0.01] | 0.00 |
| L2 keys 1 | 2 | 0.00 [0.00..0.00], 0.41 [0.31..0.46] | 0.00 |
| L2 keys 1 | 4 | -0.00 [-0.00..-0.00], 0.43 [0.34..0.48] | 0.01 |
| L4 keys 0 | 2 | -0.00 [-0.00..-0.00], -0.00 [-0.02..0.03], -0.02 [-0.08..0.01], 0.02 [-0.00..0.03] | 0.00 |
| L4 keys 0 | 4 | -0.00 [-0.00..0.00], 0.01 [-0.03..0.07], -0.03 [-0.11..0.01], 0.06 [-0.00..0.11] | 0.00 |
| L4 keys 1 | 2 | -0.00 [-0.00..0.00], 0.44 [0.27..0.67], 0.58 [0.52..0.66], 0.35 [0.33..0.37] | 0.00 |
| L4 keys 1 | 4 | -0.00 [-0.00..0.00], 0.43 [0.25..0.67], 0.66 [0.61..0.73], 0.39 [0.37..0.40] | 0.01 |
| L8 keys 0 | 2 | -0.00 [-0.00..0.00], -0.00 [-0.02..0.02], -0.06 [-0.06..-0.06], -0.01 [-0.05..0.04], -0.01 [-0.04..0.02], 0.06 [0.01..0.14], -0.00 [-0.01..0.01], 0.04 [0.00..0.11] | 0.00 |
| L8 keys 0 | 4 | 0.00 [-0.00..0.00], -0.00 [-0.02..0.03], -0.09 [-0.12..-0.07], 0.03 [-0.08..0.11], 0.03 [-0.06..0.13], 0.08 [0.05..0.11], 0.05 [-0.00..0.13], 0.10 [0.04..0.19] | 0.00 |
| L8 keys 1 | 2 | -0.00 [-0.00..0.00], 0.24 [-0.06..0.47], 0.75 [0.40..0.94], 0.77 [0.55..1.00], 0.36 [0.26..0.54], 0.20 [0.09..0.31], 0.25 [0.21..0.29], 0.26 [0.24..0.29] | 0.00 |
| L8 keys 1 | 4 | -0.00 [-0.00..0.00], 0.29 [0.06..0.52], 0.86 [0.45..1.09], 0.93 [0.64..1.25], 0.42 [0.29..0.65], 0.24 [0.15..0.35], 0.28 [0.24..0.34], 0.31 [0.27..0.36] | 0.01 |

### Patching controls after layer 0 (reviewer-requested): which tokens' m=1 state removes the discount?

Pull toward the duplicated value; mean over seeds. 'random' = copies perturbed by a random vector of the same norm as the copies patch. Independent-copies reference: m=2 0.176, m=4 0.384; true duplicates 0.

| model | m | n | unpatched | copies ← m=1 | rest ← m=1 | all ← m=1 | random |
|---|---|---|---|---|---|---|---|
| L2 keys 0 | 2 | 3 | 0.048 | 0.068 | 0.074 | 0.094 | 0.048 |
| L2 keys 0 | 4 | 3 | 0.087 | 0.162 | 0.139 | 0.217 | 0.088 |
| L2 keys 1 | 2 | 3 | 0.030 | 0.076 | 0.056 | 0.102 | 0.026 |
| L2 keys 1 | 4 | 3 | 0.063 | 0.175 | 0.117 | 0.233 | 0.053 |
| L4 keys 0 | 2 | 3 | 0.047 | 0.063 | 0.068 | 0.085 | 0.047 |
| L4 keys 0 | 4 | 3 | 0.082 | 0.155 | 0.105 | 0.197 | 0.084 |
| L4 keys 1 | 2 | 3 | 0.026 | 0.067 | 0.043 | 0.084 | 0.026 |
| L4 keys 1 | 4 | 3 | 0.047 | 0.156 | 0.087 | 0.194 | 0.052 |
| L8 keys 0 | 2 | 3 | 0.046 | 0.062 | 0.059 | 0.076 | 0.046 |
| L8 keys 0 | 4 | 3 | 0.085 | 0.157 | 0.092 | 0.177 | 0.086 |
| L8 keys 1 | 2 | 3 | 0.031 | 0.067 | 0.047 | 0.083 | 0.030 |
| L8 keys 1 | 4 | 3 | 0.053 | 0.161 | 0.093 | 0.196 | 0.055 |

How to read the attention tables: in a set model, an exact copy has the same key as the original at layer 0. At depth 1
the copies' total mass is therefore exactly m·q1/(1+(m−1)·P1), and I checked it to 3 decimals per head. The
"beyond softmax arithmetic" metric scores only changes in the *scores*. It scores the normalisation discount as 0,
although the trained depth-1 model learned that discount by concentrating mass on rows (ratio 1.50 vs 1.72 untrained;
reviewer finding 1). Readout attention, normalisation included, is what deduplicates at depth 1.

The patching-controls table, added after the review, replaces the original copies-only patch result. Patching the rest
of the context removes about as much discount as patching the copies (no keys), so the mixed state that carries the
discount is not held in the copies. A random patch of the same norm does nothing.

The count-probe R² pools rows in proportion to m, so m ≥ 3 dominates it. It shows count information above the
untrained and shuffled baselines, not an in-distribution count.

For m > 2 (out of distribution), the no-keys depth-2/4/8 pull peaks at m = 4 and then falls; the others keep rising
sublinearly. The predictive std *rises* slightly with m in every trained model (negative log-precision shift).
Independent readings would make it fall, and true duplicates would leave it unchanged.

## 3. Predictions (PREREG) scored
1. Depth 1 behaves like count-all: **false.** On shared errors, depth 1 closes 88% (no keys) and 95% (keys) of the
   iid-distinct→Bayes gap. On duplicates it keeps about 45% of count-all's excess, not 100%.
2. Depth ≥ 2 reaches L1 and L2: **L1 false at every depth** (best: 0.72 of the gap closed, at depths 4 and 8 with
   keys). **L2 true** for every arm at depth ≥ 2 except depth 2 without keys (0.024).
3. Without keys, L3 needs more depth than with keys: **false.** L3 is met at depth 1 in both arms.
4. With keys, L3 at depth 2: **true** (already at depth 1).
5. Kelvin's hypothesis, "attention does much of the deduplication": **true, and forced by the architecture.** In its
   prereg form, "final-layer query attention per source tracks the GLS weights":
   - **with keys, true**: corr 0.86 at depth 2 and 0.59 at depth 4 (0.78 at its layer 2); at layers ≥ 1 the scores
     deduplicate copies by 20–77%, and depth lowers the pull from 0.040 to 0.026–0.031;
   - **without keys, weak**: corr ≤ 0.31, and the scores do not single out copies. The depth-1 discount is readout
     softmax normalisation. Deeper models add mixed state across the whole context, with no net gain on duplicates.
6. mech2 amendment, "depth ≥ 2 keeps the pull near 0": **false** (0.03–0.05 at m = 2). The amendment's test, "constant
   total mass would mean readout deduplication", does not apply: no model's total readout mass on the copies is
   constant in m.

7. Split amendment, "depth 1 ≈ prior level (MSE close to R1's)": **failed.** Depth 1 beats R1 on every split
   (shared 0.029 vs 0.047, duplicates 0.035 vs 0.045, singletons 0.027 vs 0.029). In distribution it removes 0.36 of
   a copy; R1 removes 0.01. v4 scored this as confirmed, using R2 and the tied test; that scoring is withdrawn.
   "Deeper models and keys add in-context aggregation, mostly on shared errors": **partly.** Without keys, depth
   adds it on shared errors only (R2's loss removed 0.03 → 0.36–0.44). With keys, depth adds it on both.

## Files
- /home/agent/workspaces/mats/work/pfn-dedup/: PREREG.md (with amendments), world.py and test_world.py (4 exact-reference tests pass),
  pfn.py, mech2.py, mech3.py, analysis2.py, make_tables.py, summarize.py, split.py, split_copy.py, split_tables.py
  (runs_final/split.json, split_copy.json, split_tables.md).
- runs_final/: summary.md, analysis2.json, mech2/3/4.json and tables.md for the reported models. It holds symlinks:
  - depths 1 and 2 to runs_1026c/;
  - depth 8 without keys to runs_l8/ (from pfndedup-1027c);
  - depth 4 (both arms) and depth 8 with keys to runs_efg/ (from pfndedup-1027e/f/g).
  The timed-out depth-4 and 6k-step depth-8 models stay in runs_1026c/ and are not reported. merge_json.py,
  splice_*.py and head_v3.md rebuild the report.
- Post-hoc analyses (mech3, mech4, analysis2) are labelled as such in PREREG.md.
- Code sha256 (first 16 hex): world 4de70e602f93eae1, pfn 6c18676ccaa1cc85, mech2 c392addaea02b87b,
  mech3 fa0d267348158200, analysis2 376ca3b6d95c9d40.

## Adversarial review (one independent general-purpose agent; given the code, data and claims, not my conclusions)
The reviewer checked these as sound: references (a)–(d) and the 203-partition Bayes, the NLL gaps, masks and indexing,
the depth-1 softmax identity (holds per head), the pull metric and its references (count-all pull = (m−1)/(6.25+m)),
the probe protocol and baselines, and the patch indexing. Findings, most severe first, with what I changed:

1. **Broke a headline.** The draft said "the discount is in the token contents, not the readout" and called attention
   "context mixing that computes the count". But depth 1, where no mixed state reaches the query, gets the same
   m = 2 pull as depths 2–8. The readout path alone produces the whole in-distribution discount. The "beyond softmax"
   metric scores learned normalisation as "no dedup" by definition. Reviewer's counterfactual on L1_k0_s0 at m = 2
   (each copy given the original's weight, no renormalisation): pull 0.055, against 0.039 observed.
   **Accepted; headline rewritten.**
2. **Broke the patching inference.** Patching the *non-copy* tokens with their m = 1 states raises the pull about as
   much as patching the copies, and a random patch of the same norm does nothing. My only control, patching after the
   last layer, was a trivial no-op. **Accepted.** I ran the controls on all depth ≥ 2 models and seeds (mech4.py, table
   above). No keys: copies and rest give similar effects, so the state is spread over the context. Keys: the copies'
   state matters more (m = 2, final models: 0.067–0.076 vs 0.043–0.056).
3. **Weakened the depth comparison.** Depth 8 had 6k steps; depth 4 is 0/6 converged; training had m ≤ 2 only. A flat
   curve over depth cannot separate capacity from optimisation; only depth 1 vs 2 is firm. **Accepted, and fixed after
   the review:** every arm is now trained to 40k steps (15/24 converged). The comparison now shows depth helps with
   keys and not without. The m ≤ 2 limit stands.
4. **Weakened "25–40% of count-all".** Count-all's per-reading weight (0.138) exceeds the models' (0.105–0.116) and
   Bayes' (0.114). Against the models' own no-dedup baseline (≈ 0.10), the residual is about 30–50%. **Accepted; both
   given.**
5. **Design.** Without positional encoding the MLPs act per token, so "attention is responsible" is guaranteed; only
   "which attention" is testable. Copies are identical at every layer, so a copy's score can never drop relative to
   the original's; only row-level effects can. The count-probe R² is dominated by m ≥ 3, which is out of distribution.
   **Accepted; stated.**
6. **Minor.** Corr 0.86 with GLS weights is on shared-error episodes, not copies. The last layer at depth 4 with keys
   gave 0.40 on the timed-out models and gives 0.59 on the converged ones; both are now reported. Untrained pull and MSE baselines are near-constant outputs, uninformative for
   those metrics (they remain informative for probes and attention). L3 NLL closure partly reflects variance
   calibration; the MSE results back it up.

## Adversarial review 2 (2026-10-28; independent subagent; full text: outbox/pfn-dedup-review-2-2026-10-28.md)
Critical: v4 table A's 'independent' row copied o_s0's query reading exactly, and in training an exact tie
means a copy. The reviewer's in-distribution test reversed the no-keys headline. Major: R2 is not prior-level
by the prereg's own definition, and shares anchored on count-all overstate 'mostly prior-level'. Moderate:
the cues were not isolated; the references know the row count. Minor: CIs and convergence were not shown.
**All accepted.**
- Section 0 is rebuilt on an in-distribution test, with the cues isolated (table B).
- Three tiers and both normalisations are reported (table C), with CIs and convergence columns.
- Prediction 7 is rescored as failed.
- The v4 headlines and tables are kept collapsed and marked superseded or withdrawn.
Table B answers the reviewer's open question: without keys, depth 1's discount comes mainly from the exact
query tie, not from identity vectors.

## Track A2 (2026-10-28): is the no-keys ceiling a weak-signal effect? Retrained with costly copies

**Short answer: yes, mostly.** I retrained no-keys depths 1, 2 and 4 (3 seeds, 40k steps, everything else unchanged)
in a world where copies are common and repeated. With probability 0.8 an episode copies 1–3 sources, each 1–3 times
(up to 4 rows per source). In-context deduplication then rises, deeper models start to use source identity, and depth
helps beyond 2. Depth 1 also improves, so it is not prior-level in either world. Predictions are in PREREG.md
(amendment A2); the evaluation was switched to the in-distribution test after review 2, before any A2 model existed.

All evaluations use the same original-world episodes for both worlds. Controls are the existing original-world
models. The variant models' own test NLL gap to Bayes is 0.076–0.079 at depth 1, 0.046–0.054 at depth 2 and
0.029–0.033 at depth 4 (count-all: 0.319 on duplicate episodes).

Per-seed one-copy deduplication (2 − rho):

| depth | control | costly world |
|---|---|---|
| 1 | 0.30–0.40 | 0.60–0.62 |
| 2 | 0.23–0.29 | 0.60–0.61 |
| 4 | 0.23–0.29 | 0.73–0.83 |

Every depth-4 seed exceeds every depth-2 seed. The bootstrap CIs below are wide because per-episode weight ratios
are heavy-tailed.

### A2-1. In-context deduplication, in distribution (original-world episodes; 95% bootstrap CI over episodes)

One copy: 2 − rho (0 = no within-episode dedup, 1 = full). Four rows (s0 plus 3 copies; in distribution only for the costly world): (4 − rho)/3. Cue removal as in section 0 table B (out of distribution for both worlds).

| depth | world | conv | one copy | four rows | no query tie | no labelled equality | new identity |
|---|---|---|---|---|---|---|---|
| 1 | original (control) | 3/3 | 0.36 [0.24, 0.48] | 0.49 [0.42, 0.54] | 0.16 | 0.35 | 0.37 |
| 1 | costly | 3/3 | 0.61 [0.44, 0.74] | 0.71 [0.64, 0.77] | 0.20 | 0.60 | 0.60 |
| 2 | original (control) | 1/3 | 0.26 [0.19, 0.32] | 0.45 [0.40, 0.51] | 0.13 | 0.21 | 0.26 |
| 2 | costly | 1/3 | 0.61 [0.51, 0.70] | 0.73 [0.67, 0.77] | 0.29 | 0.57 | 0.44 |
| 4 | original (control) | 3/3 | 0.25 [0.19, 0.31] | 0.45 [0.40, 0.50] | 0.10 | 0.21 | 0.24 |
| 4 | costly | 1/3 | 0.77 [0.50, 0.96] | 0.84 [0.73, 0.90] | 0.06 | 0.74 | 0.44 |

### A2-2. Tied copy-vs-independent test (withdrawn as a headline after review 2; kept for the preregistered P1 continuity): 1 − pull(copy)/pull(independent row with the same query reading)

| depth | original (control) | costly |
|---|---|---|
| 1 | 0.09 | 0.13 |
| 2 | 0.44 | 0.62 |
| 4 | 0.49 | 0.86 |

### A2-3. Copy sweep: pull toward the copied reading vs number of rows m (mech2; Bayes 0 for every m)

| depth | world | m=2 | m=3 | m=4 | m=8 |
|---|---|---|---|---|---|
| 1 | original (control) | 0.049 | 0.083 | 0.106 | 0.150 |
| 2 | original (control) | 0.048 | 0.076 | 0.087 | 0.066 |
| 4 | original (control) | 0.047 | 0.074 | 0.082 | 0.034 |
| 1 | costly | 0.029 | 0.051 | 0.068 | 0.105 |
| 2 | costly | 0.024 | 0.039 | 0.047 | 0.046 |
| 4 | costly | 0.010 | 0.016 | 0.016 | -0.018 |

Scored against the A2 predictions as amended before any A2 result:

| prediction | result | verdict |
|---|---|---|
| P1: depth ≥ 2 in-distribution deduplication ≥ 0.55 | 0.61 (depth 2), 0.77 (depth 4) | **true** |
| P1, original form on the tied test (≥ 0.70) | depth 4 0.86 true; depth 2 0.62 false | — |
| P2: identity is used (the new-identity copy loses ≥ 0.10) | depth 2 −0.17, depth 4 −0.33; depth 1 −0.01 | **true at depth ≥ 2** |
| P3: depth 4 beats depth 2 by ≥ 0.10, or its m = 4 pull is ≤ 2/3 of depth 2's | +0.16 (one copy), +0.11 (four rows); m = 4 pull 0.016 vs 0.047 | **true** |
| P4: m = 4 pull ≤ 0.05 at depth ≥ 2 | 0.047, 0.016 | **true** |
| Depth 1 stays prior-level (≤ 0.2) in both worlds | 0.36 control, 0.61 costly | **false** |

- **Cues.** In the costly world, depth 4 relies on the exact query tie: without the tie it falls from 0.77 to 0.06. It
  also uses identity (0.77 → 0.44). Depth 2 uses both (tie 0.61 → 0.29, identity 0.61 → 0.44). Depth 1 still uses
  only the tie.
- **Caveats.**
  - Convergence: depth 1 3/3, depth 2 1/3, depth 4 1/3.
  - The costly world also changes the row count (up to 15 rows), so "more copy signal" is not the only change.
  - Cue removal is out of distribution.
  - Even at depth 4 a single copy still keeps about 0.2 of its weight; Bayes removes all of it.
- **Cost:** 3 on-demand L4 VMs, 1.61 + 1.63 + 1.69 h, $4.29 at $0.87/h (cap $15). All pfndedup VMs deleted:
  `gcloud compute instances list --filter=name~pfndedup` returned nothing at 03:30Z. Running total for this report:
  about $15.2 ($10.9 + $4.29).
- **Files:** world.py (PFN_WORLD=costly), launch.sh (passes PFN_WORLD), runs_a2/ (checkpoints, split_indist.json,
  split_copy.json, mech2.json, a2_tables.md), runs_a2ctl/ (controls), a2_tables.py, a2_code_sha256.txt.
