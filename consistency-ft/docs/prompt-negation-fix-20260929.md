# Negation-explicit prompt v7b_explicit vs v6_direct (Qwen3-32B, 2026-09-29)

In the question-only v6_direct prompt, "A" and "not A" differed by one word at the end:
"Statement to assess: Question A resolves YES." versus "... resolves NO.". That line came right
below "This proposition is true if the market resolves YES.". Kelvin found the prompts confusing,
and approved a fix on 29 September.

The new prompt describes each question neutrally and restates the proposition inside the
statement. It writes the negation out in capitals and says when the statement is true.
v6_direct is unchanged: the prompts of all v1 to v6 variants hash the same before and after this
change (sha256 over every validation and test prompt).

**Result, on the 1,066 test pairs (2,132 A/not-A events), v7b_explicit vs v6_direct (the same DashScope
server, temperature 0).** Differences are v7b minus v6, with 95% paired bootstrap CIs over pairs.

| metric | v6_direct (DashScope t0) | v6_direct (vLLM) | **v7b_explicit** | v7b − v6 DashScope |
|---|---|---|---|---|
| corr(p(A), p(not A)) (ideal −1) | −0.184 | −0.182 | **−0.611** | −0.427 [−0.483, −0.371] |
| same number for A and not A | 19.8% | 24.2% | **13.0%** | −6.9 pts [−9.0, −4.8] |
| \|p(A) + p(not A) − 1\| > 0.2 | 37.2% | 36.5% | **26.6%** | −10.6 pts [−13.1, −8.0] |
| mean p(A) + p(not A) | 0.997 | 0.952 | 1.116 | |
| all-12 log loss, raw | 0.6222 | 0.6114 | **0.5943** | −0.028 [−0.038, −0.017] |
| literals (P, not P, Q, not Q), raw | 0.7501 | 0.7499 | **0.6696** | −0.081 [−0.099, −0.060] |
| NOT P / NOT Q, raw | 0.833 / 0.861 | 0.848 / 0.861 | **0.676 / 0.707** | −0.157 / −0.155 |
| P / Q, raw | 0.636 / 0.670 | 0.619 / 0.671 | 0.634 / 0.661 | −0.002 / −0.009 (n.s.) |
| compounds (8 forms), raw | 0.5582 | 0.5421 | 0.5566 | −0.002 [−0.013, +0.010] |
| all-12 log loss, projected (kl) | 0.5410 | 0.5409 | **0.5256** | −0.015 [−0.023, −0.006] |

Against the vLLM v6 answers the picture is the same. The all-12 raw log loss is −0.017 [−0.028, −0.006]
and the projected one −0.015 [−0.024, −0.006]. The raw compounds are +0.015 [+0.004, +0.025] worse
than vLLM, but vLLM v6 was also better than DashScope v6 on the compounds.

What changed:

- **Most of the raw gain is on the negated literals.** NOT P and NOT Q log loss drops by 0.16.
  P and Q do not change. The compound forms are no better raw, but they become better after
  projection, because the projected atoms now use cleaner marginals.
- **Coherence improves, but it is not solved.** 13% of events still get the same number for A and
  not A. The model now leans toward "not A": the mean p(A) + p(not A) is 1.12, up from 1.00. On validation
  the lean was on the multiple-choice options: mean p(not A) was 0.71 and mean p(A) 0.40.
- **Validation did not predict the log-loss gain.** On the 422 validation pairs the raw log loss
  did not move (below). Only coherence moved there. We chose the wording on coherence, with log
  loss "no worse". The test run was done once, with the chosen wording.

Cost: $2.63 of $5 in all (validation $1.51, test $1.12). There were 33,048 answers: 24,608 API calls, plus 8,440 identical
prompts reused from an earlier log. There were 0 retries and 0 missing answers.
Every call reported `qwen3-32b` as the served model.

## Validation (422 pairs, 844 events), and the two iterations

| arm | corr | same number | \|sum−1\|>0.2 | mean sum | LL raw all-12 | LL raw literals | LL projected |
|---|---|---|---|---|---|---|---|
| v6_direct, vLLM (09-23) | −0.498 | 19.1% | 27.1% | 0.992 | 0.6467 | 0.7418 | 0.5881 |
| v6_direct, DashScope t0 (this run) | −0.314 | 19.9% | 38.6% | 1.029 | 0.6646 | 0.7651 | 0.5926 |
| v7_explicit (first wording) | −0.515 | 17.3% | 26.5% | 1.107 | 0.6608 | 0.7813 | 0.5926 |
| **v7b_explicit** (iteration 1, chosen) | **−0.654** | **12.3%** | **20.0%** | 1.105 | 0.6610 | 0.7822 | 0.5953 |
| v7c_explicit (iteration 2) | −0.653 | 13.7% | 25.1% | 1.132 | 0.6592 | 0.7767 | 0.5928 |

On validation, v7b minus v6 DashScope is:

- corr −0.340 [−0.440, −0.244]
- same number −7.6 pts [−10.7, −4.5]
- |sum−1|>0.2 −18.6 pts [−23.1, −14.3]
- all-12 raw log loss −0.004 [−0.025, +0.019]
- projected log loss +0.003 [−0.014, +0.022]

The three wordings:

- **v7_explicit** describes each question neutrally, as "(yes/no)" or "(multiple choice; each
  option either happens or does not)" plus "Option under consideration". The "This proposition is
  true if the market resolves YES" line is gone. The statement quotes the question or option and
  writes NOT in capitals, e.g. `The option "X" of Question A will NOT happen (it resolves NO).`
  Joint statements use "BOTH of these are true: (1) ...; AND (2) ..." and "AT LEAST ONE of these is
  true (possibly both): (1) ...; OR (2) ...".
- **v7b_explicit** is v7 plus, on negated literals only, "(a NEGATION)" before the statement and
  "This statement is TRUE if X does NOT happen, and FALSE if X happens." after it.
- **v7c_explicit** is v7b plus the matching TRUE/FALSE sentence on positive literals. It had the
  same correlation as v7b, but more sums off by more than 0.2, so we kept v7b.

Choosing the declarative form: turning each question into a declarative sentence with a template
("Will X beat Y?" → "X will NOT beat Y") is not safe on this data. 79% of the propositions are
multiple-choice options with free-form text ("Burnley beat Liverpool (Sun Sep 14, 14:00)", "3 (or
more) come-from-behind wins"), and the yes/no titles are not uniform ("Grokipedia v1 before Nov
30?"). So v7 quotes the question or option and says "The answer to Question A (...) is NOT yes: it
is NO" or "The option "..." will NOT happen".

Criteria: `[image]`, `[video]`, `[embed]`, `[link]` and `[tweet]` placeholders are removed. When
nothing is left, or the criteria are empty, the prompt says "Resolution criteria: not available
(see the question)."

- On the test split, 289 of 2,132 propositions had an inline `[image]`.
- 102 had empty criteria: v6 printed no criteria line for them.
- One was `[image]` only: v6 printed `Resolution criteria:\n[image]`.

The as-of line, the answer instruction, the parser, max_tokens 20 and the single retry are
unchanged.

## Example prompts: A and not A, old and new

These examples use validation-pair-00004, Question A, a yes/no market. All four prompts share the
system message "You are a careful forecaster. You estimate probabilities of future events."

### Old (v6_direct), A

```
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A
Question: Will Oklahoma Defeat Alabama on December 19th, 2025?
This proposition is true if the market resolves YES.
Closes in the week of 2025-12-15.
Resolution criteria:
The market will resolve to YES if Oklahoma’s team defeats Alabama’s team in that game, as determined by the official final score (including any overtime, if played). It will resolve to NO if Alabama wins, or if the game ends in a tie/draw with no winner officially recorded.

Statement to assess: Question A resolves YES.
What is the probability that this statement is true?

Reply with exactly one line of this form and nothing else:
Probability: 0.xx
(a single number between 0.01 and 0.99). /no_think
```

### Old (v6_direct), not A

This is the same prompt with the last word of the statement changed: `Statement to assess: Question A resolves NO.`

```
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A
Question: Will Oklahoma Defeat Alabama on December 19th, 2025?
This proposition is true if the market resolves YES.
Closes in the week of 2025-12-15.
Resolution criteria:
The market will resolve to YES if Oklahoma’s team defeats Alabama’s team in that game, as determined by the official final score (including any overtime, if played). It will resolve to NO if Alabama wins, or if the game ends in a tie/draw with no winner officially recorded.

Statement to assess: Question A resolves NO.
What is the probability that this statement is true?

Reply with exactly one line of this form and nothing else:
Probability: 0.xx
(a single number between 0.01 and 0.99). /no_think
```

### New (v7b_explicit), A

```
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A (yes/no): Will Oklahoma Defeat Alabama on December 19th, 2025?
Closes in the week of 2025-12-15.
Resolution criteria:
The market will resolve to YES if Oklahoma’s team defeats Alabama’s team in that game, as determined by the official final score (including any overtime, if played). It will resolve to NO if Alabama wins, or if the game ends in a tie/draw with no winner officially recorded.

Statement: The answer to Question A ("Will Oklahoma Defeat Alabama on December 19th, 2025?") is YES.
What is the probability that this statement is true?

Reply with exactly one line of this form and nothing else:
Probability: 0.xx
(a single number between 0.01 and 0.99). /no_think
```

### New (v7b_explicit), not A

```
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A (yes/no): Will Oklahoma Defeat Alabama on December 19th, 2025?
Closes in the week of 2025-12-15.
Resolution criteria:
The market will resolve to YES if Oklahoma’s team defeats Alabama’s team in that game, as determined by the official final score (including any overtime, if played). It will resolve to NO if Alabama wins, or if the game ends in a tie/draw with no winner officially recorded.

Statement: (a NEGATION) The answer to Question A ("Will Oklahoma Defeat Alabama on December 19th, 2025?") is NOT yes: it is NO. This statement is TRUE if the answer is NO, and FALSE if the answer is YES.
What is the probability that this statement is true?

Reply with exactly one line of this form and nothing else:
Probability: 0.xx
(a single number between 0.01 and 0.99). /no_think
```

For a multiple-choice option with no usable criteria (validation-pair-00086), the new not-A statement
reads:

```
Question A (multiple choice; each option either happens or does not): Pluribus season 1 props
Option under consideration: It is revealed that the hivemind intentionally kills-off members specifically because their brains are unsuitable (e.g. damaged, ill, etc)
Closes in the week of 2025-12-22.
Resolution criteria: not available (see the question).

Statement: (a NEGATION) The option "It is revealed that ... (e.g. damaged, ill, etc)" of Question A will NOT happen (it resolves NO). This statement is TRUE if "It is revealed that ..." does NOT happen, and FALSE if "It is revealed that ..." happens.
```

In the real prompt, the option text is written out in full wherever "..." appears above.

More examples, including joint forms, are in `example-prompts.json` in the data directory.

## Experiments that used the old wording (to re-check)

Each of these used the v6 question block ("This proposition is true if the market resolves YES")
and "Question X resolves YES/NO" statements. Some of them read the v6 answers directly.

- **Averaging vs structure** (`docs/averaging-vs-structure-20260924.md`,
  `scripts/run_decomposition_sampling.py`) and **coherence projection**
  (`docs/coherence-projection-20260924.md`): the size of the projection gain. The v6 raw baseline is
  now 0.028 worse than v7b, and the projection gain shrinks from 0.081 to 0.069 on DashScope.
- **Balanced triangles** (`docs/balanced-triangles-20260927.md`, `scripts/balanced_triangles.py`,
  `worldmodel_clusters.messages`): signed literals "Question k resolves YES/NO". These are the most
  exposed to this bug.
- **Weekly world model** (`docs/weekly-worldmodel-20260927.md`, `scripts/weekly_worldmodel.py`) and
  **example week** (`scripts/example_week*.py`): singles, conditionals and ANDs, all with v6 blocks.
  The "AND bias" finding should be re-checked.
- **Confidence weighting** (`docs/confidence-weighting-20260927.md`,
  `scripts/run_confidence_signals.py`, `confidence_weighting*.py`): the v6_direct prompt plus logprob,
  range and know lines.
- **Eliciting weights** (`docs/eliciting-weights-20260927.md`, `scripts/run_eliciting_weights.py`,
  `eliciting_weights*.py`): v6 answers as the quick point.
- **World-model clusters, global and projection** (`worldmodel_clusters.py`, `worldmodel_global.py`,
  `worldmodel_projection.py`, 09-24): the same question blocks.
- **Structured pairs** (`structured_pairs.py`), **active structure** (`active_structure_collect.py`),
  **market-context zero-shot** (`market_context_zeroshot.py`), **Luna ceiling** (`run_luna_ceiling.py`)
  and **order matrix** (`order_matrix.py`, which reads the vLLM v6 answers): all use
  `vb.question_block` or `plain_statement`.

## Files

- Prompt: `training/verbalized_baseline.py`. It adds the variants `v7_explicit`, `v7b_explicit` (the
  chosen one) and `v7c_explicit`, plus `question_block_v7`, `explicit_clause` and
  `explicit_statement`. The v1 to v6 variants are unchanged.
- The test is `test_v7_explicit_statements` in `test_verbalized_baseline.py`.
- Runner and analysis: `scripts/prompt_negation_fix.py`:
  - `run`: DashScope `qwen3-32b` with `enable_thinking: false` and temperature 0. It uses at most 8
    workers, a $5 cap over all logs, and the key from `secret get [removed]`. A
    temperature-0 prompt identical to an earlier logged one is reused, not re-sent.
  - `analyze`: coherence, per-form log loss (raw and kl-projected with
    `scripts/coherence_projection.project`), and a paired bootstrap over pairs (2,000 resamples
    for coherence, 1,000 for log loss).
- Data in `/home/agent/data/prompt-negation-fix-20260929/`:
  - `log-<split>-<variant>.jsonl`: every answer
  - `report-validation-v7*.json` and `report-test-v7b_explicit.json`: all metrics, per form, with CIs
  - `example-prompts.json`
- The DashScope v6 test answers are the t0 sample of `/home/agent/data/decomposition-32b-20260924/log.jsonl`.
  The vLLM v6 answers are in `/home/agent/data/qwen32b-baseline-20260923/remote-out/`.

Reproduce the test comparison:

```
python3 scripts/prompt_negation_fix.py run --split test --variant v7b_explicit --workers 8
python3 scripts/prompt_negation_fix.py analyze --split test --variant v7b_explicit \
  --against v6_direct_dashscope_t0=/home/agent/data/decomposition-32b-20260924/log.jsonl:p:t0 \
            v6_direct_vllm=/home/agent/data/qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl \
  --report /home/agent/data/prompt-negation-fix-20260929/report-test-v7b_explicit.json
```
