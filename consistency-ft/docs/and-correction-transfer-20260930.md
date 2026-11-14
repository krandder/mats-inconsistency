# Does the AND-correction prompt transfer to the Manifold pair set? (2026-09-30)

Status: **done**. It was pre-registered 2026-09-30 ~14:53Z (commit 1a04618), before any call, and run 14:54–15:01Z.
The primary failed: the paragraph does not transfer (C − M0 = +0.016 [+0.001, +0.032]). The rest of the text above Results is as pre-registered.

## Question

On the weekly world model, one frozen paragraph appended to the joint ("A and B") prompt cut Qwen3-32B's joint
log loss below the product of its own single answers. It did so on the 6 test weeks (C − M0 = −0.046
[−0.058, −0.035]) and on 5 held-out train-split weeks (−0.062 [−0.078, −0.046]); see
`docs/weekly-worldmodel-20260927.md`. Does the same paragraph, unedited, help on a different dataset: the
Manifold pair test set of `docs/coherence-projection-20260924.md` and `docs/strength-scaling-20260930.md`?

## Design

**Pairs and prompts.** All 1,066 test pairs of the pair set (`/home/agent/data/jev-disjoint-v1-filtered-20260919`),
one form per call, with the v7b_explicit wording (`scripts/prompt_negation_fix.py`, `rows_for("test",
"v7b_explicit")`). All 12 forms are labelled on this split.

**Arm C (new calls).** The four AND forms, (P) AND (Q), (P) AND (NOT Q), (NOT P) AND (Q) and (NOT P) AND (NOT Q):
4,264 calls. Each prompt is the v7b_explicit prompt, with the frozen correction paragraph inserted as its own
paragraph before the answer line. The insertion code is the one used on the weekly world model. The paragraph
is the `CORRECTION` constant of `scripts/weekly_worldmodel_arms.py`, sha256
`31e151649af70f4372baa6a8b15085c4b7e781c2b6595f27dc8260ba9ce4875c`, and the script asserts the hash:

> Note: forecasters tend to rate statements like this one too high. The statement needs every question above
> to resolve YES, so its probability cannot be higher than that of the least likely question, and unless the
> questions are closely linked it is close to the product of their separate probabilities, which is usually
> much lower than any one of them.

The model is `qwen3-32b` on DashScope intl, with thinking off, temperature 0, max_tokens 20, the v6 parser, and
one retry with `RETRY_REMINDER`. The settings are those of the existing answers. Any served model other than
qwen3-32b stops the run: there is no fallback. Up to 12 workers.

**Wording caveat (known before the run, not fixed, because the paragraph is frozen).** The paragraph says "needs
every question above to resolve YES". That is literally true only for (P) AND (Q). The other three AND forms
need one or both questions to resolve NO, and for them the right upper bound is the smaller of the two
*literal* probabilities, not of the questions. (P) AND (Q) is therefore reported separately, as the form where
the wording fits.

**Comparison (no new calls).** The existing uncorrected Qwen3-32B v7b_explicit answers for all 12 forms (U):
`/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl`, 12,792 answers, served
qwen3-32b only, 0 missing. The script checks that, before the paragraph is inserted, all 4,264 AND-form
prompts are byte-identical to the logged U prompts (4,264/4,264 at pre-registration). The singles (P, NOT P,
Q, NOT Q) are not asked again. Their existing answers serve both arms.

**Metric.** Outcome log loss with p clipped to [0.01, 0.99]. A missing answer counts as 0.5.

- **Primary: joint.** The mean log loss over the four AND forms of arm C, compared with M0. M0 is the product
  of the model's own answers to the two literal forms in the conjunction (for example p(P)·p(NOT Q) for
  (P) AND (NOT Q)). The result is C − M0, with a paired bootstrap over pairs (2,000 resamples, seed 20260930).
  **The paragraph transfers if the 95% CI upper bound is < 0.**
- **Secondary, pre-registered and reported whatever they show:**
  - C − U: the corrected joints against the uncorrected joints.
  - U − M0.
  - C − M0c, where M0c takes 1 − p(A) instead of the answer to NOT A.
  - Each AND form separately; (P) AND (Q) is the form whose wording fits.
  - The share of AND answers above the smaller of their two literal answers, for U and C.
  - Mean stated p against the YES rate.
- **Singles.** Log loss over P, NOT P, Q and NOT Q. The answers are unchanged, so this is context, not a test.
- **Coherence projection.** Fit the kl projection of `scripts/coherence_projection.py` twice. The first fit
  uses all 12 existing answers (U). The second uses the 8 existing non-AND answers plus the 4 corrected AND
  answers (C). Read all 12 forms back from each fit. The comparison is the all-12 log loss of the C projection
  minus that of the U projection, with a paired bootstrap over pairs. **Projection of the corrected answers
  beats projection of the uncorrected ones if that CI's upper bound is < 0.** Reported the same way: the
  literals, AND forms and OR forms after projection, and the C projection against raw C.

The secondaries are not corrected for multiplicity. They describe the result; they do not rescue a failed
primary.

**Sizing (power, from the existing answers only).** On these 1,066 pairs, the per-pair AND-form difference
U − M0 has SD 0.259. With n = 1,066 the standard error is 0.0079. A paired test therefore has 80% power
(two-sided α = 0.05) to detect a true C − M0 of 0.022 nats.

The weekly effects were −0.046 and −0.062. The only other pair data are the 300-pair subset (inside this set) and
the validation split (used to design v7b). Neither adds held-out pairs, so the full test set is the sample.

**Baseline already in hand (existing answers, no new calls).**

| | AND-form log loss (4 forms) |
|---|---|
| U, uncorrected joints | 0.5652 |
| M0, product of literal answers | 0.5479 |
| M0c, product using 1 − p | 0.5288 |
| singles (P, NOT P, Q, NOT Q) | 0.6696 |

U − M0 = +0.017 [+0.002, +0.033]: the uncorrected joints are already worse than the product, as on the
weekly world model. The YES rate over the AND forms is exactly 0.25, because exactly one of the four is true.
The mean stated U is 0.344 (the four sum to 1.38 on average), and the mean M0 is 0.313.

**Cost.** Mean prompt 518 tokens plus about 75 for the paragraph, 7 output tokens per call. At list price
($0.7/M in, $2.8/M out), 4,264 calls cost about $1.9. The cap is $15, the run stops at the cap or at
18:15Z, and cost goes into `BUDGET-24H-20260929.md`.

**Code and outputs.** `scripts/and_correction_transfer.py` (`check`, `run`, `score`). The log goes to
`/home/agent/data/and-correction-transfer-20260930/log.jsonl` and the scores to `summary.json` in the same
directory, with a copy at `docs/and-correction-transfer-20260930-summary.json`.

## Results

Run 14:54–15:01Z. 4,264/4,264 corrected calls, served qwen3-32b only, 0 missing, 0 retries, $2.05 at list
price. Scores: `docs/and-correction-transfer-20260930-summary.json`
(`python3 scripts/and_correction_transfer.py score` re-derives them from the two logs).

**Primary: the paragraph does not transfer.** C − M0 = **+0.016 [+0.001, +0.032]** over the four AND forms. The
corrected joints are still worse than the product of the model's own literal answers, and no better than the
uncorrected joints: C − U = −0.001 [−0.014, +0.011].

| AND-form log loss | C (corrected) | U (uncorrected) | M0 (product) | C − M0 | C − U |
|---|---|---|---|---|---|
| all four (primary) | 0.5642 | 0.5652 | 0.5479 | +0.016 [+0.001, +0.032] | −0.001 [−0.014, +0.011] |
| (P) AND (Q), wording fits | 0.3754 | 0.4366 | 0.4007 | −0.025 [−0.046, −0.005] | −0.061 [−0.082, −0.041] |
| (P) AND (NOT Q) | 0.4744 | 0.5006 | 0.5212 | −0.047 [−0.080, −0.013] | −0.026 [−0.053, +0.003] |
| (NOT P) AND (Q) | 0.5206 | 0.5526 | 0.5318 | −0.011 [−0.045, +0.023] | −0.032 [−0.067, +0.004] |
| (NOT P) AND (NOT Q) | 0.8864 | 0.7711 | 0.7380 | +0.148 [+0.095, +0.199] | +0.115 [+0.061, +0.167] |

Other secondaries:

- C − M0c = +0.035 [+0.020, +0.052].
- The share of AND answers above the smaller of their two literal answers falls from 23.5% (U) to 4.6% (C).
- Mean stated p on the AND forms is 0.180 for C, 0.344 for U and 0.313 for M0, against a YES rate of 0.250. The
  four corrected AND answers sum to 0.72 on average, where U summed to 1.38.
- The singles are unchanged: log loss 0.6696.

**Coherence projection: the corrected answers project worse, not better.** The comparison is the kl projection,
all-12 log loss, with C − U and a paired bootstrap over pairs.

| after projection | U | C | C − U |
|---|---|---|---|
| all 12 (pre-registered test) | 0.5256 | 0.5311 | +0.0055 [+0.0017, +0.0090] |
| literals | 0.5965 | 0.6010 | +0.0045 [+0.0003, +0.0083] |
| AND forms | 0.4902 | 0.4962 | +0.0060 [+0.0022, +0.0096] |

Projection still helps a lot within each arm: raw all-12 is 0.5943 for U and 0.5939 for C. The OR rows equal the
AND rows after projection, because each OR form is the negation of an AND form and projected answers are coherent.

**Reading (post hoc, descriptive).** The paragraph pushes every AND answer down, which helps on the three forms
that are usually false: YES rates 0.14, 0.19 and 0.20, against mean U answers of 0.25, 0.27 and 0.33. It hurts
badly on (NOT P) AND (NOT Q), which is true in 47.5% of pairs because most Manifold questions resolve NO: the
mean answer drops from 0.53 to 0.28. On the weekly world model every joint was an all-YES conjunction of
questions with low YES rates, so "much lower than any one of them" pointed the right way. Here it does not for
the both-NO form. The one form whose wording fits, (P) AND (Q), does improve against both the product and the
uncorrected answers. The paragraph encodes the bias of one dataset, not a general rule, and the frozen wording
does not transfer as a whole.

## Deviations

None. The code, prompts, metric and decision rules are as pre-registered in commit 1a04618.
