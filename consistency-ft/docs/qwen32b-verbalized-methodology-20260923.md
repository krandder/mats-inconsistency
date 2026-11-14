# Qwen3-32B zero-shot verbalized baseline: methodology (for review before rental)

Status 2026-09-23: code written, nothing rented. Code: `training/verbalized_baseline.py`,
`test_verbalized_baseline.py` on branch `plan/next-run-20260919`
(`/home/agent/workspaces/consistency-ft-next-run-20260919`). Model: `Qwen/Qwen3-32B`
revision `9216db5781bf21249d130ec9da846c4624c16137`, bf16 (FP8 only if bf16 does not load; would be stated).
Data: `jev-disjoint-v1-filtered-20260919`, validation 422 pairs (5,064 queries), test 1,066 pairs (12,792 queries).

## 1. Three rendered validation prompts (pair `validation-pair-01683`, variant v2, exact model input)

Chat template: the Qwen3-32B `tokenizer_config.json` template at the pinned revision, rendered with
`apply_chat_template(add_generation_prompt=True, enable_thinking=False)`. Non-thinking is enforced twice:
the hard switch (`enable_thinking=False`, which pre-fills an empty `<think>\n\n</think>` block) and the
`/no_think` soft switch at the end of the user message. The rendered string below is what vLLM receives
(offline `LLM.generate` on pre-rendered text, so no server-side template drift). Literal forms show only
their own question; compound forms show both (variants v4/v5 show both for every form).

### Literal P (A is a multiple-choice option market)
```
<|im_start|>system
You are a careful forecaster. You estimate probabilities of future events.<|im_end|>
<|im_start|>user
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A
Question: New Zealand vs West Indies 3rd Test 2025
Option: Follow on enforced
This proposition is true if the market resolves YES for this option.
Closes in the week of 2025-12-22.
Resolution criteria:
https://manifold.markets/post/something-crazy-happens-test-match?r=Um9kZHk

Statement to assess: Question A resolves YES.
What is the probability that this statement is true?

Think briefly (at most five sentences), then end your reply with a final line of exactly this form:
Probability: 0.xx
(a single number between 0.01 and 0.99, nothing after it). /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### NOT Q (binary market)
```
<|im_start|>system
You are a careful forecaster. You estimate probabilities of future events.<|im_end|>
<|im_start|>user
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question B
Question: QB Tanner McKee starts at least 3 NFL regular season games in '25 season?
This proposition is true if the market resolves YES.
Closes in the week of 2025-12-15.
Resolution criteria:
Includes possibility of being traded and starting for another team.

Includes all regular games through week 18, which is scheduled for Jan 4.

Statement to assess: Question B resolves NO.
What is the probability that this statement is true?

Think briefly (at most five sentences), then end your reply with a final line of exactly this form:
Probability: 0.xx
(a single number between 0.01 and 0.99, nothing after it). /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### (P) AND (NOT Q)
```
<|im_start|>system
You are a careful forecaster. You estimate probabilities of future events.<|im_end|>
<|im_start|>user
Assume today is 2025-12-08. Use only information available up to that date; do not use knowledge of later events.

Question A
Question: New Zealand vs West Indies 3rd Test 2025
Option: Follow on enforced
This proposition is true if the market resolves YES for this option.
Closes in the week of 2025-12-22.
Resolution criteria:
https://manifold.markets/post/something-crazy-happens-test-match?r=Um9kZHk

Question B
Question: QB Tanner McKee starts at least 3 NFL regular season games in '25 season?
This proposition is true if the market resolves YES.
Closes in the week of 2025-12-15.
Resolution criteria:
Includes possibility of being traded and starting for another team.

Includes all regular games through week 18, which is scheduled for Jan 4.

Statement to assess: Both of the following are true: (1) Question A resolves YES; (2) Question B resolves NO.
What is the probability that this statement is true?

Think briefly (at most five sentences), then end your reply with a final line of exactly this form:
Probability: 0.xx
(a single number between 0.01 and 0.99, nothing after it). /no_think<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## 2. Text fields: included, excluded, why

Own renderer (`question_block`); `training/supervised_pairs.py` is not modified.

| Field | Used | Why |
|---|---|---|
| `parent_question` | yes | the question |
| `answer_text` | **yes**, as `Option:` line | For multiple-choice markets (id contains `:`; 1,678/2,132 test, 762/844 validation propositions) it is the option name; without it the proposition is undefined. Checked: option names carry no resolution markers. **Defect note:** `supervised_pairs.proposition_text` drops it, so the 20-Sep Qwen3-0.6B arms (A1–A5) saw undefined propositions for ~79% of test rows. |
| `criteria_text` | yes, cleaned, capped at 1,500 chars + ` [...]` | same cap as the 0.6B arms |
| creator edits inside `criteria_text` | **removed** | Paragraphs starting `Update 20..`, `Edit:`/`Edit ...`, `EDIT` are dropped; an `Update YYYY-MM-DD` or `Edit Mon DD:` marker inside a paragraph (e.g. after `[image]`, `[tweet]`, `[link preview]`) is cut to the paragraph end. Done before the cap. Propositions affected: validation 378/844, test 489/2,132 (train 1,081/4,328). Residual after stripping: validation 0; test 1 harmless inline `[edit: a portion of]`. `historical_text_verified` is False for every row, so criteria text may still contain other post-hoc wording we cannot detect. |
| `closing_week` | yes | per question ("Closes in the week of ...") and for the as-of line |
| as-of line | yes | "Assume today is <earlier closing_week of the pair minus 7 days>. Use only information available up to that date; do not use knowledge of later events." One date per pair so all 12 forms share it. Questions close 2025-05..12, after Qwen3's training cutoff, so this mainly sets tense. |
| `url`, `family_id`, ids, `historical_text_verified` | no | not forecast content |
| labels, `hidden_query_ids`, `observed_form` | no | never in a prompt |

## 3. Output contract and parser

Contract (reasoning variants): at most five sentences, then final line `Probability: 0.xx`, a single
number in [0.01, 0.99]. v6: only that line.

Parser (`parse_probability`): strip closed `<think>...</think>` blocks; if an unclosed `<think>` remains -> missing;
take the **last** match of

```
probability\s*\**\s*(?:[:=]|is)\s*\**\s*(?:~|≈|about|approximately|approx\.?)?\s*(?P<num>\d*\.?\d+)(?![\d.]*[A-Za-z_])(?P<frac>\s*/\s*\d+)?\s*(?P<pct>%|percent)?(?P<tail>\s*(?:-|–|to)\s*\d*\.?\d+\s*%?)?
```
(case-insensitive). A range -> missing; a fraction -> missing; `%`/`percent` divides by 100; a bare number in
(1, 100] is read as a percent; anything else outside [0,1] -> missing. The parser never returns a default.
For scoring, parsed values are clipped to [0.01, 0.99] (the range the prompt asks for); unclipped scores are
also reported.

Adversarial tests (all pass, `test_verbalized_baseline.py`):

| Input | Parsed | Reason |
|---|---|---|
| `'Short reasoning.\nProbability: 0.73'` | 0.73 | ok |
| `'Probability: 73%'` | 0.73 | ok |
| `'0.73'` | None | no_probability_line |
| `'p=.7'` | None | no_probability_line |
| `'Probability: 60-70%'` | None | range |
| `'I cannot estimate this.'` | None | no_probability_line |
| `'Probability: 0.4\nOn reflection...\nProbability: 0.35'` | 0.35 | ok |
| `'**Probability:** 0.2 - given the base rate'` | 0.2 | ok |
| `'<think>Probability: 0.9</think>\nProbability: 0.15'` | 0.15 | ok |
| `'<think>long reasoning Probability: 0.9 ...'` | None | unclosed_think |
| `'Probability: 1/3'` | None | fraction |
| `'Probability: 150'` | None | out_of_range |
| `'The probability is about 5 percent.'` | 0.05 | ok |
| `'Probability: .05.'` | 0.05 | ok |
| `'Probability: 0.xx'` | None | no_probability_line |
| `'Probability: 0'` | 0.0 | ok |

## 4. Failure policy

1. Unparseable first answer -> one retry, temperature 0: the conversation plus the model's own reply plus
   the user turn "Your reply did not end with a valid line of the form 'Probability: 0.xx'. Reply with exactly one line, 'Probability: ' followed by a single number between 0.01 and 0.99, and nothing else." (max 20 tokens).
2. Still unparseable -> recorded as missing with the raw text and the parser reason. Never imputed silently.
3. Reported: missing rate overall, per form, per split, first-try vs after-retry.
4. Every score is reported twice: missing excluded, and missing imputed at 0.5. Consistency: pairs with any
   missing form excluded (count reported) and 0.5-imputed.
5. A variant with > 5% missing on validation after retry is disqualified; if the chosen variant exceeds 5%, abort
   before test and report.

## 5. Sampling and serving

vLLM offline engine, 1xH100 80GB, bf16, `max_model_len` 4096 (longest rendered validation prompt 1,209 tokens
over all variants, median 947–1,079), `gpu_memory_utilization` 0.92, prefix caching on (the question blocks
precede the statement, so the 12 forms of a pair share a prefix).
Main run: temperature 0 (greedy), `max_tokens` 400 (v6: 20), one sample per query; a truncated reply is
unparseable and goes through the retry path. Thinking ablation (if budget allows): chosen variant, first 50
validation pairs (600 queries), `enable_thinking=True`, no `/no_think`, Qwen's recommended thinking sampling
(temperature 0.6, top_p 0.95, top_k 20; greedy is discouraged for thinking because it loops), seed 20260923,
`max_tokens` 4096 with `max_model_len` 6144; compared with the non-thinking run on the same 600 queries.

## 6. Prompt variants (validation only)

All share the as-of line, question blocks and output contract. Train-split literal-P base rates used by v3:
binary 0.445, option 0.392 (train only).

| Variant | System | Statement | Extra |
|---|---|---|---|
| v1_plain_symbolic | careful forecaster | `(A) AND (NOT B)` symbols with a legend | - |
| v2_plain_language | careful forecaster | "Both of the following are true: (1) Question A resolves YES; (2) Question B resolves NO." / "At least one of ..." / "Question A resolves NO." | - |
| v3_superforecaster_baserate | superforecaster, calibrated, never 0/1 | plain | train base-rate sentence (binary 44%, options 39% resolve YES) |
| v4_logic_steps | careful forecaster | plain, both questions always shown | estimate A and B first, then combine (NOT = 1-p, AND <= min, OR >= max and <= sum, independence if unrelated) |
| v5_pair_context | careful forecaster | plain, both questions always shown | - |
| v6_direct | careful forecaster | plain | answer line only, no reasoning |

v1 vs v2 is the symbols-vs-plain-language test. Every variant runs on all 5,064 validation queries.
Selection: lowest validation log loss over all 12 forms with missing imputed at 0.5, among variants with
<= 5% missing. Test is scored once with the chosen variant. Optional post-hoc Platt scaling
(a*logit(p)+b) fitted on the chosen variant's validation predictions, applied to test; raw and calibrated
both reported. Full rendered user prompts of every variant for the AND example:
`/home/agent/data/qwen32b-baseline-20260923/example-user-prompts-all-variants.json`.

## 7. Metrics and comparison

Test: log loss, Brier, accuracy, ECE (10 bins), reliability-bin dump, per-form breakdown, consistency =
mean over pairs of `relation_penalties(...)['total']` (mean squared violation on the named forms), for
(a) all 12 forms (full labels) and (b) the observed-form subset (1,066 queries) for comparison with constant
0.5 (0.693), Boolean prior (0.607), per-form train base rate (0.578), and the 20-Sep 0.6B arms (A4 0.666/0.634,
A5 λ=1 0.642/0.614, A2 0.598, A3 0.595; note their `answer_text` defect). The three prior baselines are also
recomputed on all 12 forms.

## Cost plan

One H100 ~$2–2.5/h. Estimated 3–5 h total (6 variants x 5,064 validation + 12,792 test + ablation, greedy,
short outputs, prefix caching) -> ~$8–13, under the $30 / 8 h cap; watchdog stops the instance at 8 h, $30 or
30 min lease silence.
