# Does the projection gain shrink as the model gets stronger? (2026-09-30)

Status: **pre-registered** 2026-09-30 07:01Z (commit 1bc2270), before any gpt-6-luna or claude-opus-5-5 call. The Results
section is filled in after the runs; everything above it stays as written (any change is listed under
"Deviations" with its reason).

## Question

Coherence projection (fit one distribution over the four atoms of a pair to the 12 stated forms, read all 12
back) cut all-12 log loss on the Manifold pair test set from 0.611 to 0.541 for Qwen3-32B and from 0.553 to
0.516 for Qwen3-235B (v6 wording). The MiniBench backtest with Opus 5.5 found the consistency arm about equal
to the plain arm on real questions (C − B = −0.30 [−1.92, +1.41], `minibench/backtest/bt-scale/RESULTS.md`).
If a frontier model gains about nothing from projection **on the same pairs**, consistency mainly repairs
noise that strong models do not make. If it still gains, the MiniBench null is about that question set or
setup.

## Design

**Pairs.** The 300 test pairs of `docs/averaging-vs-structure-20260924.md`
(`/home/agent/data/decomposition-32b-20260924/subset.json`, drawn with
`random.Random(20260924).sample(sorted(test pair ids), 300)`). Qwen3-32B and Qwen3-235B one-form-per-call
answers with the v7b_explicit wording already exist for all of them. The new arms run the pairs in that
sample order, so a budget stop leaves a random prefix.

**Why 300 (power).** On these pairs the 235B per-pair all-12 projection gain has SD 0.040 and the
per-pair difference of two models' gains (32B − 235B) has SD 0.071. With n = 300 a paired test has 80%
power (two-sided α = 0.05) to detect a single-model gain of 0.0065 nats and a difference in gains of 0.0115
nats. The "No gain" bar below (95% CI lower bound ≥ −0.01) is met with 90% power at n ≈ 171 if the true gain
is 0, so 300 leaves room for a frontier model with a larger spread. Both Qwen gains on this subset are
≥ 0.035, far above these limits.

**Prompt.** Exactly the v7b_explicit messages of `scripts/prompt_negation_fix.py` (`rows_for("test",
"v7b_explicit")`): the system line "You are a careful forecaster. You estimate probabilities of future
events." and the v7b user prompt (question texts and resolution criteria as in the Qwen runs; no search, no
tools, no extra evidence). The one change: the trailing " /no_think" (a Qwen-only switch) is removed; the
script asserts it was there. One form per call, 12 calls per pair, 3,600 calls per model. Parser:
`verbalized_baseline.parse_probability`; an unparseable reply gets the protocol's one retry with
`RETRY_REMINDER` (for gpt-6-luna as a second turn after the model's reply; for Opus appended to the prompt,
because headless `claude -p` has no multi-turn without a session). Still unparseable = missing.

**Models and routes (no silent fallback).**
- Qwen3-32B: existing log `/home/agent/data/prompt-negation-fix-20260929/log-test-v7b_explicit.jsonl`
  (DashScope, thinking off, t0).
- Qwen3-235B-A22B: existing log
  `/home/agent/data/consistency-distill-v2-20260929/teacher235/log-test-v7b_explicit.jsonl` (DashScope,
  thinking off, t0).
- **gpt-6-luna** via the fleet Codex route: the codex-any proxy (`127.0.0.1:7810`, Responses API, streamed,
  `store: false`, no tools). Reasoning effort: the first of `none`, `minimal`, `low` the route accepts in the
  smoke test (lowest available, matching Qwen with thinking off). The response's `model` must be
  `gpt-6-luna` (or a dated snapshot of it); anything else stops the arm. If the direct Responses call does
  not work, the fallback *transport* is `codex exec` with the same model pin (as in
  `scripts/run_luna_ceiling.py`); never another model.
- **claude-opus-5-5** via Prime (`127.0.0.1:7800`), headless `claude -p` with no tools, no MCP, no hooks,
  no session, thinking off (`alwaysThinkingEnabled: false`), `--effort low`, `CLAUDE_CODE_NO_MODEL_FALLBACK=1`,
  tag `token-proxy-managed:calibrated-ai` (the MiniBench bot's call pattern). The result's `modelUsage` must be
  exactly `["claude-opus-5-5"]`; anything else stops the arm.
- A stopped arm writes `STOP-<model>` in the output directory and is reported as such; it is not rerun on
  another model.

**Smoke test.** One validation pair (12 calls) per model: effort level, served model, parsing and cost per
pair. If 300 pairs would exceed the arm's cap, the arm runs the largest prefix of the sample order that fits
(fewer than 250 pairs is a Problem, below).

**Budget.** $60 at list prices for everything (Kelvin's cap). Script caps: gpt-6-luna $8, Opus $48, and a
hard stop at $58 summed over all logs in `/home/agent/data/strength-scaling-20260930/` (smoke tests
included). Luna is costed from token usage at $0.10 / $0.01 / $0.50 per M input / cached input / output
tokens; Opus from the CLI's `total_cost_usd` (list-price equivalent). Spend is logged in
`calibrated-ai/BUDGET-24H-20260929.md`.

## Analysis (fixed now)

Script: `scripts/strength_scaling.py analyze`. The analysis set is the subset pairs on which every model has
all 12 forms logged (a missing answer counts as logged: it is imputed 0.5 for scoring and left out of the
fit, as in all earlier runs).

- **Raw**: each form's stated probability. **Projected**: the kl coherence projection of
  `scripts/coherence_projection.py` per pair (a pair with no stated form at all passes through unprojected).
- **Primary metric**: all-12 log loss, probabilities clipped to [0.01, 0.99]; per pair, the mean over its 12
  forms; per model, the mean over pairs. **Gain** = projected − raw (negative = projection helps).
- **CIs**: paired bootstrap over pairs, 2,000 resamples, seed 20260930, the same resample indices for every
  model and contrast; 95% percentile intervals.
- **Also reported** (not used for bars): literals and compounds separately, Brier, the negation coherence of
  the raw answers (corr of p(A) with p(not A) over A = P, Q; share of identical numbers; share with
  |p(A) + p(not A) − 1| > 0.2; mean sum), the per-model missing count, and the contrasts between every pair of
  models in raw log loss and in gain.
- **Model strength** = raw all-12 log loss on the same pairs.

**Bars per frontier model** (all-12 kl gain, 95% CI):
- **No gain**: CI lower bound ≥ −0.01 (the gain is at most 0.01 nats; checked first).
- **Gain**: otherwise, CI upper bound < 0.
- **Inconclusive**: otherwise.
- **Problem** (overrides the above): P1 the model is unavailable or substituted and the arm stopped; P2 more
  than 5% of its forms missing after the retry; P3 fewer than 250 pairs complete for all four models; P4 the
  independent verifier's recomputation differs from a headline number by more than 0.001.

**Scaling reading** (per frontier model m, against 235B): "shrinks" if m's raw log loss is significantly
below 235B's (CI of m − 235B raw < 0) and m's gain is significantly smaller in size (CI of gain_m − gain_235B
> 0); "does not shrink" if m is significantly stronger but the gain difference CI includes 0 or is < 0; no
reading if m is not significantly stronger than 235B.

**What each outcome means for Kelvin's question.** Opus "No gain": consistency mainly repairs noise a
frontier model does not have; the MiniBench null is expected. Opus "Gain": the frontier model is still
incoherent on these pairs, so the MiniBench null is about its question set or setup (e.g. questions without
the structure projection needs, or the arm's implementation).

**Contamination (secondary, pre-registered).** The test questions closed May–Dec 2025, inside Opus 5.5's
training data (cutoff June 2026) and probably gpt-6-luna's. A model that recalls outcomes answers literals
near 0 or 1, is already coherent, and gains nothing, which would make "No gain" a recall artifact rather than
a strength effect. Two checks:
- **Recall flag**: share of literal answers (P, not P, Q, not Q) at ≤ 0.05 or ≥ 0.95, and their accuracy.
  Flag if the share is ≥ 25% and the accuracy ≥ 95%.
- **Uncertain pairs** (label-free selection): for each frontier model m, the pairs where m's raw P and Q are
  both in [0.10, 0.90]; all four models' gains on that set (reported if it has ≥ 30 pairs). If m's gain on its
  uncertain pairs is still ~0 while the Qwen models gain there, "No gain" is not only recall.

**Continuity.** The v6_direct numbers Kelvin quoted (32B vLLM per-form, 235B per-form) are recomputed on the
same pairs for reference; no new calls.

## Outputs

`docs/strength-scaling-20260930.json` (all numbers), `docs/strength-scaling-20260930-chart.json` (one point per
model: raw log loss, projected log loss, gain with CI, negation corr with CI, bar). Logs:
`/home/agent/data/strength-scaling-20260930/` (not committed).

## Results

Run 07:03–07:55Z 2026-09-30. All 300 pairs complete for all four models; 0 missing forms after the protocol
(no unparseable replies at all; 5 Luna transport failures rerun, see Deviations); served models only
`gpt-6-luna` and `claude-opus-5-5`, no arm stopped. No Problem condition (P1–P3) holds; P4 is below.
Numbers: `docs/strength-scaling-20260930.json`; chart: `docs/strength-scaling-20260930-chart.json`.
Check: `python3 scripts/strength_scaling.py analyze --out ... --chart ...` reproduces them from the logs.

**Headline (same 300 test pairs, v7b_explicit, one form per call; all-12 log loss; 95% paired bootstrap CI)**

| model | raw | kl-projected | gain (proj − raw) | negation corr p(A) vs p(not A) | bar |
|---|---|---|---|---|---|
| Qwen3-32B (t0, no think) | 0.600 | 0.526 | −0.074 [−0.083, −0.066] | −0.54 [−0.62, −0.45] | (reference) |
| Qwen3-235B-A22B (t0, no think) | 0.547 | 0.511 | −0.036 [−0.041, −0.031] | −0.77 [−0.82, −0.72] | (reference) |
| gpt-6-luna (effort none) | 0.529 | 0.459 | −0.071 [−0.085, −0.056] | −0.80 [−0.85, −0.74] | **Gain** |
| claude-opus-5-5 (effort low) | 0.368 | 0.362 | −0.005 [−0.008, −0.003] | −0.98 [−0.985, −0.970] | **No gain** |

Opus's gain is statistically nonzero but under the 0.01 bar (the CI lower bound is −0.008), so the bar is
"No gain". Contrasts (paired): Opus − 235B raw −0.180 [−0.211, −0.148], gain +0.031 [+0.025, +0.036]:
Opus is much stronger and its gain is smaller, so the pre-registered scaling reading for Opus is
**"shrinks"**. Luna − 235B raw −0.018 [−0.048, +0.013] (not significantly stronger), gain −0.035
[−0.049, −0.021] (Luna gains *more*): no scaling reading for Luna. 235B − 32B gain +0.038 [+0.031, +0.046].

**Contamination.** The recall flag is on for Opus: 34% of its literal answers are ≤ 0.05 or ≥ 0.95, and
98.5% of those are right (Luna 29% at 89%, 235B 9% at 90%, 32B 13% at 82%). Part of Opus's strength here is
recall of 2025 outcomes. The label-free secondary check says the "No gain" is not only recall: on the 127
pairs where Opus's own P and Q are both in [0.10, 0.90], Opus still gains only −0.005 [−0.007, −0.003]
(raw 0.551), while on the same pairs 32B gains −0.078, 235B −0.038 and Luna −0.039. On Luna's 138 uncertain
pairs Luna gains −0.028 [−0.033, −0.023] and Opus −0.005.

**Where the gain comes from (exploratory, not pre-registered).** Gain tracks coherence, not raw accuracy.
Luna at effort none is about as accurate as 235B but breaks compound bounds (e.g. p(P and Q) more than 0.1
above min(p(P), p(Q))) on 16% of checks, vs 13% for 32B, 5.7% for 235B and 2.2% for Opus; its compound
gain is −0.075 vs literal −0.062. Opus is nearly coherent on negations (corr −0.98, mean p(A)+p(not A)
0.991) and compounds.

**Continuity.** The v6_direct numbers on the same 300 pairs: 32B 0.609 → gain −0.068, 235B 0.553 → gain
−0.036 (Kelvin's full-set figures were 0.611 → 0.541 and 0.553 → 0.516), so v7b does not change the Qwen
gains much.

**Answer.** On the same pairs the frontier Claude model gains about nothing from projection (−0.005 nats,
vs −0.036 for 235B and −0.074 for 32B), also on the pairs it is unsure about. So for Opus, consistency
mainly repairs noise it does not make, and the MiniBench null (C − B ≈ 0) is what this predicts, not a sign
that the question set or setup hid a gain. The gain does not fall with raw accuracy as such: gpt-6-luna with
no reasoning is as accurate as 235B and gains twice as much, because it is incoherent on compound forms.
Projection pays for incoherent answerers (small models, frontier models run without reasoning), not for a
coherent frontier model. Limits: frontier arms ran at default sampling temperature (Qwen at t0), one effort
level each (Luna none, Opus low), and Opus recalls many 2025 outcomes.

**Cost.** $31.00 at list price (Luna $0.20, Opus $30.80) of the $60 cap; ledger rows 07:32 and 07:56 in
`calibrated-ai/BUDGET-24H-20260929.md`.

**Verification (P4).** An independent verifier (a separate agent that wrote its own scoring, sharing only
`fit`/`FORM_ATOMS`) recomputed 150 numbers from the raw logs at ~08:05Z: raw, projected, gain and CI,
negation corr, recall stats, the six contrasts and the uncertain-pair gains. The largest difference was
1.7e-9, under the 0.001 tolerance, so P4 does not apply. It also confirmed 3,600 rows per model, no
duplicates, labels agreeing across models, only the requested served models, and new-arm spend of $30.99
in the logs. The JSON's "No gain (small but significant)" is the pre-registered bar "No gain" plus a note
that the CI excludes 0.

## Deviations

- Recorded after the smoke tests, before the main runs (07:02Z, commit cbdcd3a): the Qwen answers were sampled at
  temperature 0; the frontier arms run at the provider's default sampling, because the Luna route rejects
  `temperature` ("Unsupported parameter", checked with one call) and headless `claude -p` has no temperature
  setting. Sampling noise can add incoherence that projection removes, so a frontier "Gain" is partly
  noise the model shows in normal use (as in MiniBench, which also ran at the default); a "No gain" is
  robust to it. Smoke: gpt-6-luna accepted effort `none` (served `gpt-6-luna`, $0.0008 per pair, 300 pairs
  ≈ $0.24); Opus served `claude-opus-5-5` at `--effort low`, $0.105 per pair, 300 pairs ≈ $31.4, so both
  arms run all 300 pairs.
- gpt-6-luna: 5 of 3,600 calls failed in transport after 6 tries (proxy HTTP 409 `duplicate_cross_account`,
  no answer returned). They were rerun once at 07:32Z (the answer was never seen, so this does not select on
  the outcome); all 5 then answered. The failed records are kept in `transport-failed-gpt-6-luna.jsonl`.
