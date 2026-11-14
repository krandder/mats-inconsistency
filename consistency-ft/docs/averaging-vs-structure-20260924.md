# Averaging vs logical structure, and the nature of the noise (Qwen3-32B, 2026-09-24)

Question (Kelvin): how much of the gain from "adjust the 12 answers to be consistent" comes
from the real logical structure, and how much from just asking more times and averaging
(uncorrelated errors)? And, more broadly: what is the noise in these answers made of, and is
it decorrelated across forms?

**Answer: structure.** On all 1,066 test pairs, with one call per form, projecting onto a
coherent distribution lowers all-12 log loss by 0.081 [0.076, 0.086]. Averaging five calls
per form lowers it by 0.017 [0.016, 0.019], and averaging on top of the projection adds only
0.003 more. The same projection with the wrong relations, or structure-free shrinkage of the
same size, is 0.05 to 0.06 worse than the true structure. Sampling noise is 3% of the Brier
score. The incoherence that projection removes is a systematic error (AND forms too high,
OR forms too low), not noise that averaging can remove.

The 300-pair subset (with paraphrases) is reported in detail below. The replication on all
1,066 pairs (t0 plus 5 samples, no paraphrases) comes first.

## All 1,066 test pairs (the headline)

| arm [calls/pair] | all-12 LL | observed LL | AND/OR LL | Brier | msv |
|---|---|---|---|---|---|
| vLLM baseline t=0 [12] | 0.6114 | 0.6155 | 0.5421 | 0.2063 | 0.0217 |
| A0 DashScope t=0 [12] | 0.6222 | 0.6167 | 0.5582 | 0.2092 | 0.0243 |
| A one t=0.7 sample [12] | 0.6210 | 0.6117 | 0.5579 | 0.2090 | 0.0242 |
| B mean of 5, probability [60] | 0.6039 | 0.5959 | 0.5453 | 0.2039 | 0.0197 |
| B mean of 5, log-odds [60] | 0.6063 | 0.5980 | 0.5473 | 0.2044 | 0.0204 |
| C projected, kl [12] | 0.5405 | 0.5469 | 0.5039 | 0.1808 | 0 |
| C projected, ls_logit [12] | 0.5405 | 0.5456 | 0.5044 | 0.1807 | 0 |
| C projected, ls_prob [12] | 0.5459 | 0.5503 | 0.5094 | 0.1816 | 0 |
| D mean of 5 (log-odds), then kl [60] | 0.5378 | 0.5442 | 0.5011 | 0.1798 | 0 |
| D mean of 5 (log-odds), then ls_logit [60] | **0.5374** | 0.5429 | 0.5012 | 0.1796 | 0 |
| E_group: wrong relations, levels kept [12] | 0.5895 | 0.5896 | 0.5450 | 0.2014 | 0.0034 |
| E_full: wrong relations, full shuffle [12] | 0.7885 | 0.7801 | 0.7789 | 0.2880 | 0.0472 |
| E_shrink: size-matched shrinkage [12] | 0.6051 | 0.6011 | 0.5742 | 0.2078 | 0.0066 |
| F_lit: literal complements only [12] | 0.5936 | 0.5937 | 0.5579 | 0.1999 | 0.0122 |
| F_comp: literal + De Morgan complements [12] | 0.5688 | 0.5593 | 0.5206 | 0.1920 | 0.0037 |

Gain curves (all-12 LL, k = 1..5):

| arm | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| B probability | .6210 | .6105 | .6068 | .6050 | .6039 |
| B log-odds | .6210 | .6117 | .6087 | .6072 | .6063 |
| D log-odds + kl | .5405 | .5388 | .5382 | .5379 | .5378 |

Paired bootstrap, 1,066 pairs (negative = first arm better):

| comparison | Δ all-12 LL | Δ observed LL |
|---|---|---|
| B(5, probability) vs A: averaging | −0.0171 [−0.0187, −0.0158] | −0.016 [−0.019, −0.013] |
| C kl vs A: structure at 12 calls | **−0.0805 [−0.0856, −0.0757]** | −0.065 [−0.090, −0.041] |
| D(5) kl vs B(5), log-odds: structure at 60 calls | −0.0685 [−0.0730, −0.0640] | −0.054 [−0.079, −0.030] |
| C kl (12 calls) vs B(5, log-odds) (60 calls) | −0.0657 [−0.0703, −0.0613] | −0.051 [−0.077, −0.027] |
| D(5) kl vs C kl: averaging after projection | −0.0027 [−0.0031, −0.0024] | −0.003 [−0.003, −0.002] |
| E_group vs C: wrong vs true relations | **+0.0490 [+0.0420, +0.0564]** | +0.043 [+0.030, +0.056] |
| E_shrink vs C | +0.0646 [+0.0593, +0.0706] | +0.054 [+0.036, +0.074] |
| E_full vs C | +0.2480 [+0.2382, +0.2585] | +0.233 [+0.204, +0.263] |
| E_group vs A | −0.0315 [−0.0408, −0.0219] | −0.022 [−0.050, +0.004] |
| E_shrink vs A | −0.0159 [−0.0235, −0.0081] | −0.011 [−0.031, +0.008] |
| F_lit vs A | −0.0274 [−0.0297, −0.0252] | −0.018 [−0.033, −0.004] |
| F_comp vs A | −0.0522 [−0.0560, −0.0487] | −0.052 [−0.073, −0.033] |
| C vs F_comp: the non-complement rules | −0.0282 [−0.0304, −0.0261] | −0.012 [−0.025, −0.001] |
| A (t=0.7) vs A0 (t=0) | −0.0012 [−0.0054, +0.0029] | −0.005 [−0.018, +0.007] |
| A0 DashScope vs vLLM baseline | +0.0108 [+0.0044, +0.0174] | +0.001 [−0.022, +0.025] |

The agreement of A0 with vLLM over 12,792 forms:

- same answer 54.7%
- within 0.05: 60.3%
- mean |Δp| 0.068
- r(log-odds) 0.87
- DashScope +0.021 higher on average

The noise decomposition on 1,066 pairs (Brier units) matches the subset:

- d shared error 0.180 (86%)
- c incoherence 0.022 (11%)
- a sampling 0.0064 (3%)
- predicted affine-projection Brier 0.1819, measured 0.1819
- log-odds: a 0.21, c 0.71, between-pair spread 1.31; ICC 0.86

For the correlations:

- Sampling-noise |r| across forms is 0.014.
- Residual excess over the independent-error null is 0.14.
- The first PC carries 26% of the residual variance.
- AND residuals are +0.046 to +0.089 and OR residuals −0.008 to −0.050.

Cost: the whole run was 87,552 calls, $7.16 at list price, with 0 retries and 0 missing
answers. The subset (including paraphrases) cost $2.68 of that.

## Setup

- Model: `qwen3-32b` on DashScope intl (Model Studio, Singapore), thinking off, streaming.
  Every call reported `qwen3-32b` as the served model; there was no substitution.
- Data: the old 1,066-pair test set `/home/agent/data/jev-disjoint-v1-filtered-20260919/test`,
  question only. We first ran a fixed random subset of 300 pairs, then extended the run
  to all 1,066 pairs (t0 and s0..s4 only). The subset is
  `random.Random(20260924).sample(sorted(pair_ids), 300)`, with the ids in
  `/home/agent/data/decomposition-32b-20260924/subset.json`.
- Prompts: the same as the vLLM Qwen3-32B baseline (`training/verbalized_baseline.py`
  v6_direct @37578ec), with one form per call as in `scripts/run_precutoff_perform.py`
  (@4541343): max_tokens 20, the same parser, and one retry. No call needed a retry, and
  no answer was missing.
- Samples per (pair, form):
  - `t0`: temperature 0, the baseline protocol. The vLLM baseline was also temperature 0
    (greedy), with its retry at temperature 0.
  - `s0..s4`: temperature 0.7, top_p 0.8, top_k 20 (the Qwen3 non-thinking defaults). Each
    call has its own seed (sha256 of query and sample). DashScope is deterministic for a
    given seed, and `n>1` in one request returned identical choices, so we did not use it.
  - `w1..w3`: one temperature-0.7 sample for each of three fixed paraphrases of the
    statement. The rest of the prompt is unchanged.
    - w1 reorders the conditions (compounds give B first; literals become "The outcome of
      Question A is YES.").
    - w2 is a plain sentence ("Question A resolves YES and Question B resolves NO.",
      "... or ... (or both).", "Question A will resolve YES.").
    - w3 is the negated-complement / De Morgan rewrite ("It is false that at least one of
      the following is true: (1) A resolves NO; (2) B resolves YES.", "It is false that
      Question A resolves NO.").
- Scoring: `score_verbalized_baseline` conventions (clip 0.01, missing imputed at 0.5).
  - All-12 log loss (LL), observed-form LL, AND/OR LL (the 8 compound forms), and Brier.
  - Coherence: the mean squared violation of `relation_penalties` (msv), and
    |sum of the 4 AND atoms − 1|.
  - CIs: a paired bootstrap over pairs (2,000 resamples, seed 20260924, ratio of sums).
- Code: `scripts/run_decomposition_sampling.py` (runner) and
  `scripts/analyze_decomposition.py` (arms, noise decomposition, correlations). The
  projection is `scripts/coherence_projection.fit` @6d4fc01.
- Full numbers: `docs/averaging-vs-structure-20260924.json`, which is chart-ready: arms,
  k-curves, CIs, the noise decomposition and the 12×12 correlation matrices.

## Sanity check: DashScope vs the vLLM baseline (A0 vs vLLM, same 300 pairs, same prompts)

Serving differs noticeably between the two.

| measure | value |
|---|---|
| same answer | 53.5% of forms |
| within 0.05 | 59.7% |
| mean abs diff | 0.067 |
| Pearson r (log-odds / probability) | 0.87 / 0.89 |
| mean signed diff (DashScope − vLLM) | +0.021 |
| all-12 LL: DashScope t0 vs vLLM | 0.626 vs 0.609: +0.017 [+0.004, +0.030] |

A temperature-0.7 single sample (A) scores the same as temperature 0 (A0): −0.003 [−0.011, +0.005].
All arms below use DashScope answers only, so they are comparable with each other. Compare
them with the vLLM baseline only with this offset in mind.

## The 300-pair subset in detail (with paraphrases)

### Arms (300 pairs; calls per pair in brackets)

| arm | all-12 LL | observed LL | AND/OR LL | Brier | msv | \|Σatoms−1\| |
|---|---|---|---|---|---|---|
| vLLM baseline t=0 (reference) [12] | 0.6094 | 0.5967 | 0.5403 | 0.2061 | 0.0218 | 0.255 |
| A0 t=0 [12] | 0.6264 | 0.5991 | 0.5646 | 0.2111 | 0.0254 | 0.354 |
| A one t=0.7 sample [12] | 0.6235 | 0.5867 | 0.5634 | 0.2105 | 0.0256 | 0.347 |
| B mean of 5, probability [60] | 0.6059 | 0.5697 | 0.5510 | 0.2050 | 0.0208 | 0.336 |
| B mean of 5, log-odds [60] | 0.6079 | 0.5718 | 0.5523 | 0.2054 | 0.0215 | 0.333 |
| C projected, kl [12] | 0.5410 | 0.5400 | 0.5045 | 0.1808 | 0 | 0 |
| C projected, ls_logit [12] | **0.5388** | 0.5363 | 0.5028 | 0.1799 | 0 | 0 |
| C projected, ls_prob [12] | 0.5466 | 0.5438 | 0.5112 | 0.1818 | 0 | 0 |
| D mean of 5 (log-odds), then kl [60] | 0.5377 | 0.5364 | 0.5013 | 0.1796 | 0 | 0 |
| D mean of 5 (log-odds), then ls_logit [60] | **0.5353** | 0.5332 | 0.4992 | 0.1787 | 0 | 0 |
| E_group: wrong relations, levels kept, kl [12] | 0.5917 | 0.5835 | 0.5502 | 0.2021 | 0.0034 | 0 |
| E_full: wrong relations, full shuffle, kl [12] | 0.7859 | 0.7649 | 0.7753 | 0.2871 | 0.0473 | 0.950 |
| E_shrink: size-matched log-odds shrinkage [12] | 0.6066 | 0.5884 | 0.5777 | 0.2086 | 0.0068 | 0.613 |
| F_lit: only P+¬P=1, Q+¬Q=1 [12] | 0.5941 | 0.5807 | 0.5634 | 0.2005 | 0.0126 | 0.347 |
| F_comp: literal + 4 De Morgan complements [12] | 0.5699 | 0.5535 | 0.5271 | 0.1926 | 0.0040 | 0.261 |
| W mean of 4 wordings (log-odds) [48] | 0.6052 | 0.5692 | 0.5643 | 0.2057 | 0.0166 | 0.406 |
| W then kl [48] | 0.5475 | 0.5429 | 0.5130 | 0.1833 | 0 | 0 |

Notes on the table:

- A, B(k<5) and C average the per-pair loss over every choice of sample (5 singles, all
  k-subsets), which removes the luck of which sample was drawn.
- All three fits give the same ordering. The rest of the text uses kl.

Gain curves (all-12 LL, k = 1..5):

| arm | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| B probability | .6235 | .6125 | .6087 | .6069 | .6059 |
| B log-odds | .6235 | .6136 | .6105 | .6088 | .6079 |
| D probability + kl | .5410 | .5390 | .5384 | .5381 | .5379 |
| D log-odds + kl | .5410 | .5390 | .5382 | .5379 | .5377 |

### Paired bootstrap, all-12 LL difference (95% CI); negative = first arm better

| comparison | Δ all-12 LL | Δ observed LL |
|---|---|---|
| B(5, probability) vs A: averaging, 60 vs 12 calls | −0.0177 [−0.0203, −0.0153] | −0.017 [−0.023, −0.012] |
| C kl vs A: structure, equal cost (12 calls) | **−0.0826 [−0.0930, −0.0731]** | −0.047 [−0.098, +0.001] |
| C ls_logit vs A | −0.0847 [−0.0944, −0.0758] | −0.050 [−0.101, −0.003] |
| D(5, log-odds) kl vs B(5, log-odds): structure, equal cost (60 calls) | −0.0702 [−0.0798, −0.0613] | −0.035 [−0.085, +0.012] |
| C kl (12 calls) vs B(5, log-odds) (60 calls) | −0.0669 [−0.0761, −0.0583] | −0.032 [−0.082, +0.014] |
| D(5) kl vs C kl: what averaging adds after projection | −0.0033 [−0.0042, −0.0024] | −0.004 [−0.005, −0.002] |
| E_group vs C: wrong vs true relations | **+0.0508 [+0.0362, +0.0655]** | +0.044 [+0.017, +0.068] |
| E_full vs C | +0.2449 [+0.2258, +0.2650] | +0.225 [+0.174, +0.278] |
| E_shrink vs C | +0.0657 [+0.0538, +0.0769] | +0.048 [+0.010, +0.084] |
| E_group vs A | −0.0318 [−0.0502, −0.0130] | −0.003 [−0.056, +0.048] |
| E_shrink vs A | −0.0169 [−0.0325, −0.0012] | +0.002 [−0.036, +0.036] |
| F_lit vs A | −0.0295 [−0.0339, −0.0252] | −0.006 [−0.038, +0.021] |
| F_comp vs A | −0.0537 [−0.0617, −0.0466] | −0.033 [−0.070, +0.003] |
| C (full structure) vs F_comp | −0.0289 [−0.0335, −0.0248] | −0.014 [−0.038, +0.011] |
| W(4 wordings) vs B(4 same prompt), equal cost (48 calls) | −0.0037 [−0.0127, +0.0058] | −0.003 [−0.035, +0.026] |
| W then kl vs D(4) kl | +0.0096 [+0.0019, +0.0170] | +0.006 [−0.007, +0.018] |

The observed-form CIs are wide because each pair has only one observed form (300 labels).

### The control arm E: what we chose and why

Every pair has the same 12 forms, so "relations from another pair" is the same structure;
it would not be a control. Instead, E keeps everything about the projection except the
true relations:

- Each stated value is moved to another form's slot by a random derangement (5 per pair,
  fixed seed).
- The same kl fit runs on the moved values.
- Each form reads back the implied value of the slot its value went to.

This pools the same 12 numbers into the same 3 degrees of freedom, with the same shrinkage
machinery, but the ties are wrong.

- **E_group** permutes only within literals, within ANDs and within ORs. Each value keeps
  its operator's typical level, and only which forms are logically tied is broken. It keeps
  some coincidental true relations (for example, a derangement that swaps P and ¬P still
  pairs them), so it is a conservative control.
- **E_full** permutes across all 12. This also breaks the levels, so it measures level
  damage more than logic.
- Permutations that are only a relabeling of the atoms (for example, P↔¬P and Q↔¬Q
  everywhere) keep the true structure and are excluded.
- **E_shrink** is a label-free, structure-free alternative: one global log-odds shrink
  factor (λ = 0.44), chosen so that the mean |Δp| equals the projection's (0.123).

E_group recovers 0.032 of C's 0.083, and size-matched shrinkage recovers 0.017, about what
averaging gets. The true structure is worth 0.05 to 0.07 more than either (CIs above).

Among the true rules, the literal complements give 0.030, the De Morgan complements a
further 0.024, and the rest (marginals, P = P∧Q + P∧¬Q and so on) a further 0.029.

## What the noise is made of

This is a probability-space (Brier) decomposition per form, over all 300 pairs. It uses
the affine coherent subspace {Aq : Σq = 1}, which contains the truth vector:

single-sample error = **d** shared (coherent) error + **c** incoherence + **a** sampling
noise.

- c and d are measured on the expected answer μ, de-biased for the noise left in the
  5-sample mean.
- b is the variance across the 4 wordings (the original plus w1..w3, one sample each),
  minus a.
- The predictions below reproduce the measured affine-projection Brier (0.1820 predicted,
  0.1820 measured).

| component | Brier units | share of single-sample Brier | removed by averaging k=5 | removed by projection (1 sample) |
|---|---|---|---|---|
| d shared error of the coherent answer | 0.1803 | 86% | none | none |
| c incoherence (systematic, per form) | 0.0233 | 11% | none | all |
| a sampling noise | 0.0069 | 3% | 80% (0.0055) | 75% (0.0052, the part outside the 3-d subspace) |
| total (measured single-sample Brier) | 0.2105 | | B5 measured 0.2050 | C affine measured 0.1820 |
| b wording variance (not in the fixed-prompt error as variance) | 0.0214 (0.0109 without w3) | | | |

The same decomposition in log-odds (variances):

- a sampling = 0.22
- b wording = 0.64
- c incoherence (ls_logit residual) = 0.74
- between-pair spread of the mean answer = 1.36

So 86% of a single sample's log-odds variance is fixed by the prompt (ICC 0.86).
Sampling noise is largest on negated literals: the variance for ¬P is 0.018 against 0.004
for P (in log-odds, 0.60 against 0.10).

**Wording.** Wording variance is 3× the sampling variance: 0.016 is pair-specific and
0.005 is a fixed shift per template. Averaging over wordings still did not help:

- W(4) is equal to B(4).
- Incoherence fell only from 0.023 to 0.021, and the shared error rose from 0.180 to 0.184.

The reason is that the paraphrases are not noise around one answer; they are harder to
read:

- w1 matches the original (c 0.023, Brier 0.210).
- w2 is worse (c 0.031, Brier 0.222).
- The negated w3 doubles the incoherence (c 0.049, Brier 0.268; all-12 LL 0.774).

**Which component each fix removes.**

- Averaging only kills a, and a is 3% of the error.
- Projection kills c and three quarters of a (0.0285 of Brier, 14%).
- Neither touches d, which is 86%: the model's coherent belief is itself wrong or unsharp.
- Once the answers are projected, averaging can only remove the quarter of a that lies
  inside the coherent subspace (0.0017). That is why D5 − C is only −0.003 LL.

## Correlation structure (12×12 across pairs; matrices in the JSON)

- **Sampling noise is decorrelated across forms:** the mean |r| between forms is 0.025.
  The calls are independent, so this is expected, and it confirms there is no hidden
  shared state.
- **Incoherence is not decorrelated.** The residuals μ − Πμ are compared with the
  pattern expected if each form's error were independent with equal variance (the
  normalized (I−Π)):
  - Mean |excess r| is 0.16.
  - The first principal component carries 29% of the residual variance.
- **Signed residuals are systematic:**
  - Every AND form is above the coherent answer: +0.05, +0.055, +0.059, and +0.093 for
    ¬P∧¬Q.
  - Every OR form is below it: −0.018 to −0.051.
  - This compresses compounds toward the middle: conjunctions are overstated and
    disjunctions understated.
- **The largest co-movements above the null:**

  | forms | r | null r |
  |---|---|---|
  | P∧Q with P∨¬Q | +0.22 | −0.17 |
  | P with P∧¬Q | +0.17 | −0.17 |
  | Q with P∧Q | +0.17 | −0.17 |
  | ¬Q with P∧Q | +0.49 | +0.17 |
  | ¬P with P∧Q | +0.46 | +0.17 |

  So when the model overstates a literal, it also overstates the conjunctions that contain
  it, and even those that contain its negation. The pattern looks like a "yes-level" for
  the pair that leaks into every form.
- **Errors against the truth** are dominated by the coherent part d:
  - P vs ¬P: r = −0.79.
  - P∧Q vs ¬P∨¬Q: r = −0.91.
  - P vs P∧Q and vs P∨Q: r ≈ +0.55.
  This is the structure a coherent predictor's errors must have.

## Cost

- Subset: 32,400 calls (300 pairs × 12 forms × 9 samples), $2.68 at the Model Studio
  Singapore list price for qwen3-32b ($0.16/M input, $0.64/M output), 0 retries, 0 missing.
- Extension to all 1,066 pairs: a further 55,152 calls (t0 + s0..s4). The total is 87,552
  calls, $7.16, within the $15 cap.
- Calls ran with 12 workers at about 6–9 calls/s. The limit is DashScope's per-key request
  rate (HTTP 429 `limit_requests`), handled by backoff.
