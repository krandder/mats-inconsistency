# Per-answer confidence weighting for the coherence projection (2026-09-27)

Kelvin's question (27 Sep): the consistency adjustment weights all 12 answers equally, but the model may
have a strong prior on some answers and be guessing on others. Can we get a per-answer confidence from
the LLM, weight the answers by it in the projection, and show that accuracy improves?

**Answer: no, not with any per-answer signal we tried.** Several signals do predict which answers the
projection will move most (Spearman up to +0.29), but none of them improves log loss when used as weights.
Only a **per-form** weight improves log loss: the mean sampling variance of each of the 12 forms, which
mostly down-weights NOT P and NOT Q. It gains −0.0056 nats per answer [−0.0084, −0.0028] on 1,066 pairs.
The earlier per-form reliability weighting (docs/weighted-projection-20260924.md) was estimated from labels.
This one is label-free, from sampling alone, and does help. Adding per-answer weights on top of it does not.

Model: Qwen3-32B, DashScope, `enable_thinking` off, v6_direct prompt family, one form per call.
Method:
- Projection: `batch_projection.project(T, W, "kl")`, where W is 1/(v + τ). LSL is reported too.
- τ is cross-fitted over 5 folds by pair (seed 20260924) and picked by all-12 log loss. τ = ∞ gives equal weights.
- CIs: paired bootstrap 95%, 2,000 resamples over pairs (clusters for the structured pairs).
- Log loss: nats per answer over all 12 forms.

## Signals

| Signal | Source | Cost |
|---|---|---|
| sampling variance | var of logit p over k=5 samples at T=0.7 (s0..s4, Sep 24 run) | free (existing) |
| wording variance | var over 3 paraphrases (300-pair subset) | free (existing) |
| digit variance | var of the implied probability under the token logprobs of its two digits (T=0 rerun, top-5) | paid |
| first-digit entropy | entropy of the first-digit distribution | paid |
| range width | "Range: a-b", a 90% interval asked on a second answer line; sd = Δlogit/3.29 | paid |
| knowledge | "Knowledge: k" (1–5) on a second line; 6 − E[k] from the logprobs (the stated k was almost always 3) | paid |
| P(YES) | same question as YES/NO; YES/(YES+NO) token mass; Bernoulli var, and disagreement with the stated p | paid |

Step 2 made 14,400 calls on the 300-pair subset (4 variants × 12 forms), with 100% parsed.

## Which signals predict error (Spearman ρ)

Fit error is |logit p − logit projected|. Outcome error is the answer's own log loss. "Partial" controls for
extremeness |logit p|.

| Signal (pairs) | Fit: pooled / within-form / partial | Outcome: pooled / partial |
|---|---|---|
| sampling (1,066) | +0.11 / +0.10 / +0.10 | +0.02 / +0.06 |
| sampling (300) | +0.13 / +0.11 / +0.11 | +0.01 / +0.07 |
| wording (300) | +0.12 / +0.14 / +0.07 | −0.12 / −0.01 |
| sampling + wording (300) | +0.20 / +0.22 / +0.13 | −0.12 / +0.04 |
| digit variance (300) | **+0.28 / +0.27 / +0.21** | −0.14 / +0.09 |
| first-digit entropy (300) | +0.01 / −0.02 / +0.12 | **+0.30 / +0.14** |
| range width (300) | +0.18 / +0.16 / +0.03 | −0.31 / +0.01 |
| 6 − E[knowledge] (300) | −0.18 / −0.15 / −0.16 | −0.01 / −0.07 |
| P(YES) Bernoulli var (300) | +0.14 / +0.07 / +0.22 | +0.24 / +0.12 |
| P(YES) vs stated disagreement (300) | −0.08 | +0.14 |

The signals are largely proxies for extremeness: an answer near 0.5 is both "uncertain" and the one
the projection moves most. Digit variance correlates +0.39 with extremeness and range width +0.61. After
controlling for extremeness, the correlations with outcome error are at most about 0.14. Stated knowledge
points the wrong way.

## Log loss, equal vs weighted (cross-fitted, all 12 forms, KL)

| Point / weights | Pairs | Equal | Weighted | Δ [95% CI] |
|---|---|---|---|---|
| t0, per-answer sampling | 1,066 | 0.5410 | 0.5404 | −0.0006 [−0.0022, +0.0011] |
| t0, **per-form mean sampling** (control) | 1,066 | 0.5410 | 0.5354 | **−0.0056 [−0.0084, −0.0028]** |
| t0, form × per-answer vs form | 1,066 | 0.5354 | 0.5350 | −0.0004 [−0.0018, +0.0010] |
| s0 weighted by s1..s4 (separate samples) | 1,066 | 0.5424 | 0.5428 | +0.0004 [−0.0014, +0.0022] |
| mean of 5, weighted by the same 5 | 1,066 | 0.5378 | 0.5380 | +0.0002 [−0.0001, +0.0005] |
| mean of s0..s2, weighted by s3,s4 | 1,066 | 0.5385 | 0.5383 | −0.0002 [−0.0007, +0.0002] |
| t0, sampling | 300 | 0.5437 | 0.5446 | +0.0010 [−0.0013, +0.0037] |
| t0, wording | 300 | 0.5437 | 0.5437 | 0 (τ picked equal) |
| t0, sampling + wording | 300 | 0.5437 | 0.5434 | −0.0003 [−0.0023, +0.0017] |
| lp, digit variance | 300 | 0.5375 | 0.5434 | **+0.0058 [+0.0001, +0.0119]** (worse) |
| lp, first-digit entropy | 300 | 0.5375 | 0.5370 | −0.0006 [−0.0053, +0.0048] |
| lp, P(YES) var | 300 | 0.5375 | 0.5371 | −0.0005 [−0.0034, +0.0024] |
| lp, P(YES) disagreement | 300 | 0.5375 | 0.5375 | 0 (τ picked equal) |
| lp, sampling | 300 | 0.5375 | 0.5387 | +0.0012 [−0.0002, +0.0029] |
| lp, per-form mean sampling | 300 | 0.5375 | 0.5330 | −0.0046 [−0.0094, +0.0005] |
| lp, form × digit vs form | 300 | 0.5330 | 0.5403 | +0.0073 [+0.0018, +0.0134] (worse) |
| range, range width | 300 | 0.5480 | 0.5480 | 0.0000 [−0.0023, +0.0023] |
| range, form × width | 300 | 0.5480 | 0.5463 | −0.0017 [−0.0063, +0.0033] |
| know, 6 − E[k] | 300 | 0.5509 | 0.5511 | +0.0002 [−0.0006, +0.0010] |
| know, form × 6 − E[k] | 300 | 0.5509 | 0.5446 | −0.0063 [−0.0110, −0.0014] (the form part) |

LSL gives the same picture. On 1,066 pairs:
- per-form control −0.0042 [−0.0069, −0.0012];
- per-answer sampling +0.0003 [−0.0017, +0.0027].

On the 300-pair subset, digit variance is +0.0059 [+0.0013, +0.0118]. Full tables are in the chart JSON.

A likely reason the per-answer weights fail (an interpretation; not tested separately): the KL projection
already moves near-0.5 answers the most. The per-answer signals are mostly extremeness again, so they add
little that is new. After controlling for extremeness, they barely predict outcome error (partial ρ ≤ 0.14).
The per-form weight, by contrast, captures a stable fact: the model's NOT P and NOT Q answers are noisy
(sampling variance about 0.52 vs about 0.1 for the other forms).

## Side findings

- Asking for the confidence line changes the probability. Temperature 0 is not reproducible across days:
  the lp rerun matches the Sep 24 t0 answer on only 61.6% of forms (logit r 0.92), and that is the noise floor.
  - Adding the range line: logit r 0.85 with the Sep 24 t0 answer (lp: 0.92), +0.018 mean shift. Raw all-12 log loss 0.631 vs 0.619 for lp, +0.012 [−0.000, +0.024].
  - Adding the knowledge line: r 0.84 with t0. Log loss 0.628, +0.008 [−0.004, +0.021].
  - Both are slightly worse, not significantly.
- The mean of the digit distribution is a better raw forecast than the stated number:
  0.596 vs 0.619, −0.024 [−0.031, −0.017]. This gain disappears after projection: +0.0016 [−0.0033, +0.0063].
- P(YES) from YES/NO token mass is a poor forecast: raw 0.974. After projection it is +0.093 [+0.061, +0.125] worse than lp.

## Structured pairs (known relations)

The structured pairs are from docs/structured-pairs-20260924.md. They have known implication ladders,
date ladders and exclusive options.

Metrics:
- "Recovery" is the hybrid's share of the known relation's joint log loss gain over market independence.
  The hybrid is market marginals + the projection's log odds ratio.
- "Dependence captured" is the median share of the known dependence reproduced by the projection.

The weights use two signals:
- form: the external per-form mean sampling variance from the 1,066-pair test.
- digit: per-answer digit variance, from a new T=0 logprob rerun of the 32B question-only prompts (12,924 calls).

τ is cross-fitted, picked either by model all-12 log loss (_m) or by hybrid joint log loss (_h).
CIs are clustered.

| Job (pairs) | Recovery eq → weighted | Hybrid joint Δ vs eq [95% CI] | Projection joint Δ vs eq |
|---|---|---|---|
| 235B P (170) | 0.459 → 0.489 (form) | −0.0013 [−0.0028, +0.0001] | −0.0049 [−0.0086, −0.0010] |
| 235B Q (1,077) | 0.570 → 0.572 (form) | −0.0001 [−0.0004, +0.0003] | +0.0002 [−0.0019, +0.0023] |
| 32B P (1,077) | 0.408 → 0.408 (form; τ picked eq) | 0.0000 | 0.0000 |
| 32B Q (1,077) | 0.488 → 0.493 (form) | −0.0002 [−0.0005, +0.0001] | −0.0006 [−0.0023, +0.0012] |
| 32B Q lp rerun, form | 0.453 → 0.457 | −0.0001 [−0.0003, +0.0000] | +0.0001 [−0.0007, +0.0008] |
| 32B Q lp rerun, digit | 0.453 → 0.461 | −0.0003 [−0.0008, +0.0002] | −0.0016 [−0.0035, +0.0002] |
| 32B Q lp rerun, form × digit | 0.453 → 0.460 | −0.0003 [−0.0007, +0.0002] | −0.0011 [−0.0029, +0.0008] |

Relation respect (median dependence captured, 32B Q lp rerun, eq → digit _m):
- threshold 0.31 → 0.33;
- date 0.20 → 0.20;
- exclusive options 0.67 → 0.70.

With τ picked by hybrid loss (_h), form × digit reaches:
- exclusive options 0.81 and threshold 0.39;
- but joint log loss is no better: −0.0007 [−0.0023, +0.0011].

In short, confidence weighting nudges recovery up by 0.00–0.03 in every job, and every hybrid CI includes 0.
The only CI that excludes 0 is the projection itself on 235B P with form weights (−0.0049).
There, the hybrid's gain share rises 0.459 → 0.489, but the hybrid Δ CI still touches 0.

## Cost and files

- DashScope: Step 2 $1.254; smoke tests $0.007; structured logprob rerun $1.197. Total $2.46 of the $8 cap.
- Code:
  - `scripts/confidence_weighting.py`: Steps 1–2 analysis.
  - `scripts/run_confidence_signals.py`: paid calls.
  - `scripts/confidence_weighting_structured.py`: Step 3.
  - `scripts/tables_confidence_weighting.py`: chart JSON.
  - `scripts/precutoff_client.py` now returns streamed token logprobs.
- Data: `/home/agent/data/confidence-weighting-20260927/`
  - step1.json, step2-only.json, structured2.json;
  - raw call logs in step2-old/ and step2-structured/.
- Chart JSON: `docs/confidence-weighting-20260927-charts.json`.
- Rerun:
  - `python3 scripts/confidence_weighting.py --out .../step1.json`
  - `... --step2 .../step2-old --step2-only --out .../step2-only.json`
  - `python3 scripts/confidence_weighting_structured.py --out .../structured2.json --lp-log .../step2-structured/log.jsonl`
