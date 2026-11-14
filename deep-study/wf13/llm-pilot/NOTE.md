# Three biased rollout policies did not move the score-weighted answer; a helpful one moved the distribution (wf13/llm-pilot)

Model: Claude Fable 5.2 (`claude-fable-5-2[1m]`). 2026-11-10, 16:26-19:20Z. One L40 job (57877) and CPU analysis.
Words follow `wf9/FRAME.md` section 5. GUARANTEE = proved or true by construction; FINDING = measured; "post hoc" =
not in `PREREG.md` or its addendum. Registered bin: MINOR, not MAJOR; three of my registered expectations failed.
Computed: 8 writers x 3 aggregations x 2 families x 7 K (all in `out/main/verdict.txt`), plus the extension grid.

## 0. In plain words

The context first. You said the rollout policy may be anything (alien, RL-trained, biased), because the simple world
model scores only the finished demonstration and the search (the amplification over many rollouts) aims at
demonstrations in proportion to that score. This is the smallest test of that on a real language model: a frozen
Qwen3-1.7B as the simple world model; a demonstration is one line of worked steps ending in an answer; its score is
the model's own probability of writing that line. Two tasks (arithmetic, variable tracking), 150 held-out problems each.

1. Thinking helps. The model's direct answer is right 10% / 24% of the time; a vote over 64 of its own step-by-step
   rollouts 99% / 92%. A model 2.3 times larger answering directly: 9% / 25%.
2. Then I swapped the writer for a biased one: the same model fine-tuned for two minutes to make one particular
   mistake in the last step about half the time. A plain vote follows it: 57% / 35%; more rollouts do not help.
3. Count each distinct demonstration once, weighted by the world model's score: the same rollouts give 99% / 93% from
   16 rollouts on, and the answer distribution returns to within 0.03 of where the model's own rollouts put it. Same
   for two other biased writers (one prefers plausible wrong answers, one is biased by its prompt). None moved the answer.
4. The textbook correction (weight each rollout by score over the writer's own probability) also works, but is
   noisier, stalls for one writer, and has to believe the writer's own number. I had registered that the simpler
   "each distinct demonstration once" would do worse. It did better everywhere.
5. The limit showed up on the helpful side. A writer tuned on correct solutions writes only the right demonstration,
   never the model's own mistakes; the amplified distribution then says 99% where the target says 87% at 64 rollouts,
   and the gap is still there at 256. A finite search returns the score's distribution over what the writer chose to
   show. A writer that hides the best demonstration would use the same gap in the bad direction; I did not build one.
6. What catches this, from the score alone: the scores of the distinct demonstrations found add up to a share of the
   whole, which gives an interval for the target's probability of each answer whoever wrote the rollouts (the bound
   is proved; I added it after seeing item 5, so its table is post hoc). At 16 rollouts it was 0.14-0.22 wide and
   certified the answer on 90-98% of problems, biased writers included; helpful writer on variables: 0.31, 79%.
7. Smallest trust head (a calibration fitted on 300 real problem-answer pairs): on top of the vote it breaks when the
   writer changes (error 0.04 becomes 0.35); on top of the score-weighted aggregation it carries over. It improved
   nothing, because this model's belief after thinking is already calibrated on these tasks.
8. Not shown: truth. The target is the model's belief after thinking; on variable tracking its top answer is wrong on
   8% of problems and a faithful aggregation reproduces that. Repairing it needs real data on which kinds of step to
   trust; not built. I don't know whether any of this survives long free-form demonstrations: here four to six
   distinct lines carry 86% of the score, and everything cheap rests on that.

## 1. What was built (your design, FRAME section 5, at the size of one small language model)

| Your object | Here |
|---|---|
| Simple world model M | Frozen `Qwen/Qwen3-1.7B-Base` (a plain next-token model, no chat tuning) with one fixed prompt per family: a header and four solved examples. |
| Demonstration (the terminal state) | One line: the worked steps in their order, ending in " answer X". Example: ` c=c*3=1*3=3; a=a-c=9-3=6; c=c+a=3+6=9; c=c-b=9-9=0; answer 0`. |
| Score | M's probability of writing exactly that line and then stopping (the product of its token probabilities), times a fixed format check (right number of steps of the right shape; it never checks content). Target T = the score, normalised over the lines that pass the format check. The target's answer distribution is T summed over the lines with the same answer. |
| Rollout policy (writer) | Whatever writes lines. Its prompt, its temperature, its LoRA weights are the alien part: none of it is submitted. |
| Search = amplification | An aggregation of K rollouts into an answer distribution: (a) plain vote; (b) each distinct demonstration found counted once, weight = M's score; (c) each rollout weighted by M's score divided by the writer's own probability of it, self-normalised. (c) reads one number from the search, the writer's probability; (b) reads nothing from it. |
| Trust head | Smallest version only (section 6). Distillation back into the rollout policy: not run. |

Two task families with exact answers from a generator (`code/tasks.py`): **arithmetic chain** (`arithL`: a three-digit
start and three operations, values up to 9999; `223; -177; +994; +456`) and **variable tracking** (`vars`: three
variables, four assignments; each step copies the assignment, substitutes the current values and computes; M's typical
mistake is a stale value). Sizes were chosen on pilot seed 7 so that M's direct answer is poor and its step-by-step
answer much better. 150 test problems per family (seed 200), 150 calibration problems (seed 300), none among the 1,500
tuning problems per family (seed 100), the few-shot examples or the pilot problems (overlap 0). Memorising scores 0.

Writers, 64 rollouts each per test problem, all sampled from the full distribution (no top-k, no top-p):

| | Writer | What it is |
|---|---|---|
| W1 | M itself | M's prompt, temperature 1. By construction its rollouts are draws from T. |
| W2a, W2b, W2c | M, changed | Temperature 0.7; temperature 1.3 (half its lines fail the format); another header and two other examples. |
| W3 | biased, tuned | LoRA of M (rank 16, 2 epochs, 135 s), no examples in its prompt, tuned on: per tuning problem one line with a particular error in the last step (arithmetic: last digit becomes 7; variables: the last step returns the old value) and one of M's own rollouts. It writes the erroneous line about half the time. |
| W3s | biased, selective | LoRA (104 s) tuned on M's own rollouts, keeping every wrong-answer rollout and 10% of the right ones: a policy trained to prefer demonstrations that M finds plausible and that end wrong. |
| W3p | biased by prompt | M with its four examples replaced by their erroneous versions. |
| W4 | helpful | LoRA (68 s) tuned on the generator's correct lines. It writes the correct line almost every time. |

The reference for the target is the vote of 256 further rollouts of M per problem (not shared with W1); its first 64
are a "second M", whose movement from W1 is the noise floor. Distances are total variation (0 = same, 1 = disjoint).

Engineering the numbers rest on. (i) The K rollouts of one problem are packed into one sequence behind the prompt with
a 4-D attention mask (the prompt is processed once); M's score comes from the ordinary forward pass. The two paths
agree to 8e-5 in float32 (`logs/prof_f32.log`); the runs use bfloat16, where they differ by rounding (mean 0.023 nats,
`RESULT-work.json: sanity`). (ii) A rollout ends at the first token containing a newline or a special token; stopping
is charged the total probability of that set. (iii) Compute per problem for 64 rollouts: 3,100-4,900 token positions;
M's scoring pass over the distinct valid lines adds 50-330 (1.5-8%; `RESULT-posthoc.json: compute`). So (b) and (c) at
K cost about what the vote costs at K; the registered comparison "(b), (c) at K against the vote at 2K" is conservative.

## 2. The registered result: bin MINOR (one threshold missed by 0.008); every number at K = 64, 150 problems per family

`PREREG.md` was hashed at 16:59:07Z, before any run on test or calibration problems. `code/verdict.py` applies the
thresholds mechanically to `RESULT-work.json` (`verdict.txt`, `VERDICT.json`). All findings. "Excess movement" =
movement from W1 minus the noise floor.

| Hypothesis (threshold) | Arithmetic chain | Variable tracking | Verdict |
|---|---|---|---|
| H1 thinking helps (vote of M's rollouts beats M's direct answer by >= 0.30) | 0.987 against 0.100 | 0.920 against 0.240 | holds |
| H2 the plain vote follows the biased writer W3 (excess movement >= 0.20; accuracy down >= 0.15) | 0.401; accuracy 0.567 | 0.478; accuracy 0.353 | holds |
| Writers that "bite" (vote's excess movement >= 0.10) | W3 0.401, W3s 0.186, W3p 0.158 | W3 0.478, W3s 0.111, W3p 0.322 | all three, both families |
| H3 (b) and (c) do not follow (excess movement <= 0.10 and <= a third of the vote's) | (b) <= 0.005; (c) <= 0.008 | (b) <= 0.017; (c) <= 0.024 | major |
| H4 (c) returns to the target at <= 4x the rollouts (distance <= M's own vote at K = 16: 0.077 / 0.077) | W3 0.073, W3s 0.057, W3p 0.061 | W3 **0.085**, W3s 0.052, W3p 0.056 | major in arithmetic; in variables W3 misses by 0.008 (intervals overlap: 0.067-0.106 against 0.064-0.092); minor holds (bar at K = 4: 0.115) |
| H5 accuracy of (b), (c) within 0.03 of M's own vote | 0.993, 0.993 / 0.993, 0.993 / 0.993, 0.980 | 0.933, 0.927 / 0.927, 0.920 / 0.927, 0.913 | major |
| Registered expectation: (b) will NOT return to the target, even from W1 | (b) from W1 0.048; from W3, W3s, W3p 0.053, 0.048, 0.047 | 0.041; 0.064, 0.040, 0.042 | **my expectation was wrong**: (b) is the closest of the three from every one of these writers |

Why I was wrong about (b). I expected M's distribution over demonstrations to have a long tail that K rollouts miss.
Here it does not: 64 rollouts of M find 4 to 6 distinct valid lines carrying 0.86-0.87 of M's whole probability (0.91
is all the valid mass). Given those, (b) uses exact scores: its noise floor is 0.004-0.014 against 0.06 for the vote.

## 3. Accuracy against the exact answers as a function of K (Figure 1: `figs/fig1_accuracy.png`)

Each cell: plain vote / (b) score-weighted, distinct / (c) importance-weighted. At K = 1 the three coincide.
M's direct answer (one pass, no steps, greedy): 0.100 (arithmetic), 0.240 (variables). A network 2.3 times larger
answering in one pass with the same prompt (Qwen3-4B, `out/main/direct_4b.json`): 0.087 and 0.253.

| Family | Writer | K = 1 | K = 4 | K = 16 | K = 64 |
|---|---|---|---|---|---|
| arithmetic | W1, M itself | 0.855 | 0.981 / 0.990 / 0.979 | 0.988 / 0.993 / 0.988 | 0.987 / 0.993 / 0.987 |
| | W3, biased LoRA | 0.529 | 0.471 / 0.918 / 0.904 | 0.528 / 0.993 / 0.972 | **0.567** / 0.993 / 0.993 |
| | W3s, selective LoRA | 0.717 | 0.912 / 0.978 / 0.973 | 0.972 / 0.993 / 0.988 | 0.980 / 0.993 / 0.993 |
| | W3p, biased prompt | 0.677 | 0.903 / 0.976 / 0.964 | 0.987 / 0.993 / 0.983 | 0.993 / 0.993 / 0.980 |
| | W4, helpful LoRA | 0.978 | 0.988 / 0.993 / 0.983 | 0.993 / 0.993 / 0.985 | 0.993 / 0.993 / 0.973 |
| variables | W1, M itself | 0.791 | 0.896 / 0.913 / 0.896 | 0.912 / 0.920 / 0.910 | 0.920 / 0.927 / 0.920 |
| | W3, biased LoRA | 0.433 | 0.441 / 0.857 / 0.843 | 0.423 / 0.945 / 0.927 | **0.353** / 0.933 / 0.927 |
| | W3s, selective LoRA | 0.747 | 0.847 / 0.908 / 0.889 | 0.887 / 0.930 / 0.912 | 0.893 / 0.927 / 0.920 |
| | W3p, biased prompt | 0.455 | 0.663 / 0.807 / 0.813 | 0.870 / 0.922 / 0.905 | 0.913 / 0.927 / 0.913 |
| | W4, helpful LoRA | 0.993 | 0.993 / 0.995 / 0.995 | 0.993 / 0.998 / 0.993 | 0.993 / 1.000 / 0.987 |

Reading. With W3 the plain vote is stuck near one half in arithmetic and gets worse with more rollouts in variables
(0.433 at K = 1, 0.353 at K = 64; interval 0.28-0.43): more rollouts only make the writer's preference clearer.
(b) and (c) from the same rollouts reach M's own level by K = 16. Intervals at K = 64, variables: M's own vote
0.873-0.960; W3 with (b) 0.887-0.967; with (c) 0.880-0.967. (c) is a little noisier than (b) everywhere, and from W4
it falls slightly as K grows: a rare line that M likes and the writer almost never writes arrives with a very large
weight. W2a-c are in `verdict.txt`: the hot writer moves the vote by 0.12 / 0.06 over the floor, (b) and (c) not at all.

## 4. How far the answer distribution moves when the writer is swapped, and whether it returns (Figure 2)

Distance from the aggregated answer distribution to the target (the vote of 256 rollouts of M); same cell layout.

| Family | Writer | K = 4 | K = 16 | K = 64 |
|---|---|---|---|---|
| arithmetic | W1, M itself | 0.097 / 0.061 / 0.098 | 0.077 / 0.053 / 0.078 | 0.059 / 0.048 / 0.060 |
| | W3, biased LoRA | 0.476 / 0.133 / 0.147 | 0.468 / 0.060 / 0.084 | 0.464 / 0.053 / 0.073 |
| | W3s, selective LoRA | 0.276 / 0.071 / 0.099 | 0.257 / 0.054 / 0.070 | 0.248 / 0.048 / 0.057 |
| | W3p, biased prompt | 0.254 / 0.076 / 0.111 | 0.230 / 0.055 / 0.076 | 0.219 / 0.047 / 0.061 |
| | W4, helpful LoRA | 0.069 / 0.064 / 0.070 | 0.064 / 0.060 / 0.068 | 0.062 / 0.055 / 0.067 |
| variables | W1, M itself | 0.115 / 0.077 / 0.115 | 0.077 / 0.052 / 0.079 | 0.051 / 0.041 / 0.053 |
| | W3, biased LoRA | 0.553 / 0.199 / 0.209 | 0.543 / 0.084 / 0.105 | 0.537 / 0.064 / 0.085 |
| | W3s, selective LoRA | 0.216 / 0.097 / 0.125 | 0.186 / 0.053 / 0.074 | 0.170 / 0.040 / 0.052 |
| | W3p, biased prompt | 0.430 / 0.195 / 0.204 | 0.401 / 0.062 / 0.088 | 0.381 / 0.042 / 0.056 |
| | W4, helpful LoRA | **0.134 / 0.133 / 0.134** | **0.133 / 0.131 / 0.133** | **0.133 / 0.130 / 0.132** |

Movement from W1's distribution under the same aggregation, W3, at K = 4, 16, 64 (noise floor in brackets):
arithmetic: vote 0.477, 0.470, 0.465 (0.104, 0.085, 0.064); (b) 0.085, 0.012, 0.009 (0.013, 0.007, 0.004); (c) 0.148,
0.087, 0.073 (0.105, 0.087, 0.065). Variables: vote 0.564, 0.547, 0.539 (0.116, 0.081, 0.061); (b) 0.170, 0.054, 0.031
(0.037, 0.015, 0.014); (c) 0.218, 0.112, 0.086 (0.117, 0.083, 0.062). Mass on the writer's preferred wrong answer,
variables, K = 64: vote 0.493; (b) 0.007; (c) 0.010; M's own 0.010.

Reading. The vote's distance does not shrink with K for any biased writer: it converges to the writer's own answer
distribution. (b) and (c) from the same rollouts converge to M's. Under (b) the bias of W3 costs about 4x the rollouts
at small K (W3 at K = 16 is where M's own writer is at K = 4); at K = 64 a gap of 0.005 (arithmetic) and 0.023
(variables) is left. Even W3s, which moves the vote's distribution by 0.17-0.25 while hardly changing its top answer,
is brought back. The exception is the helpful writer on variables: every aggregation stays 0.13 from the target at
every K. It is closer to the truth than the target is, and it is not the target (section 5).

One problem, concretely (post hoc, `RESULT-posthoc.json: example`): `a=9 b=9 c=1; c=c*3; a=a-c; c=c+a; c=c-b; c?`,
right answer 0. M's score is 0.575 on the correct line and 0.274 on one that takes the stale `a` in step 3
(`c=c+a=3+9=12; c=c-b=12-9=3; answer 3`). M's own 64 rollouts: vote 0.61 / 0.34 on answers 0 / 3; (b) 0.67 / 0.32;
(c) 0.55 / 0.39. W3: 29 of 64 rollouts are `... c=c-b=9-9=9; answer 9` (its own probability 0.48, M's score 0.0001).
Vote: answer 9 with 0.48. (b): answer 0 with 1.00. (c): 0 with 0.998. The planted answer is gone; so is M's own second
line, which W3 never wrote. W4 wrote the correct line 64 times: every aggregation says 0 with 1.00.

## 5. What the correction costs, and what bounds it

**The divergence between writers** (test problems, K = 64; `RESULT-work.json: div`). "M's lines under the writer" =
mean over M's own valid rollouts of log(M's score / writer's probability), in nats; "writer's lines under M" = the
same the other way round, on the writer's rollouts. Cells are arithmetic / variables.

| Writer (arith / vars) | Valid lines | Distinct valid in 64 | M's lines under the writer | Writer's lines under M | Effective share of (c) | Score found by (b) |
|---|---|---|---|---|---|---|
| W1, M itself | 0.91 | 4.1 / 5.5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.91 / 0.90 | 0.87 / 0.86 |
| W2a, T = 0.7 | 0.99 / 0.98 | 1.6 / 2.2 | 0.04 / 0.06 | 0.13 / 0.13 | 0.90 / 0.82 | 0.87 / 0.85 |
| W2b, T = 1.3 | 0.46 / 0.50 | 6.7 / 8.0 | 0.81 / 0.70 | -0.51 / -0.43 | 0.40 / 0.44 | 0.86 / 0.85 |
| W2c, other prompt | 0.83 / 0.85 | 4.9 / 6.6 | 0.12 / 0.37 | -0.03 / 0.28 | 0.80 / 0.67 | 0.87 / 0.84 |
| W3, biased LoRA | 1.00 / 0.99 | 4.2 / 6.7 | 0.62 / 1.09 | 3.63 / 4.62 | 0.50 / 0.38 | 0.86 / 0.82 |
| W3s, selective LoRA | 0.98 / 0.99 | 16.2 / 11.4 | 0.14 / 0.22 | 0.51 / 0.62 | 0.77 / 0.72 | 0.87 / 0.85 |
| W3p, biased prompt | 0.89 / 0.84 | 13.4 / 27.5 | 0.19 / 0.74 | 0.39 / 2.29 | 0.74 / 0.41 | 0.87 / 0.85 |
| W4, helpful LoRA | 1.00 / 1.00 | 1.5 / 1.04 | 0.25 / **2.75** | 0.15 / 0.68 | 0.95 / 0.98 | 0.86 / **0.69** |

- What makes correction cheap is the first divergence, not the second. W3's own lines are 3.6-4.6 nats less likely
  under M (so the score throws them out), but M's lines are only 0.6-1.1 nats less likely under W3 (about log 2: half
  of what W3 writes is M-like). The ordering of (c)'s distance by the first divergence is only rough (rank correlation
  0.29 in arithmetic, where all distances are inside the noise; 0.64 in variables): H6 is not supported as a rule.
- The helpful writer is the far one: 2.75 nats on variables, because 24% of M's own rollouts there are lines W4 almost
  never writes (M's stale-value mistakes). (c)'s effective share does not see this (0.98: W4 writes one line, so all
  weights are equal). The share of the score found does: 0.69 against 0.86.
- K-extension (`PREREG-ADDENDUM.md`, registered before any test result was read; 32 problems per family, up to 256
  rollouts, target from 1,024 rollouts of M; Figure 3, `ext_table.txt`). E3 held: the biased writers' vote stays put
  (W3: 0.43 and 0.54 at K = 256). Under (b) every biased writer ends within 0.01 of M's own writer (0.029-0.045
  against 0.038 and 0.031 at 256), except W3 on variables (0.058; it passes M's K = 16 level at K = 128). Under (c),
  W3s and W3p track M's own writer (they pass that level at K = 16-32, M itself at 16); W3 passes it at K = 16 in
  arithmetic and then stalls (0.066-0.069 from K = 32 to 256, M's own 0.043), and on variables has not passed it at
  256 (0.077 against 0.074): **my expectation E1 was wrong for W3**. W4 on variables stays at 0.145 / 0.154, as expected.

**A bound from the score alone, valid for any writer (post hoc; three lines of proof, then checked).** Let F be the
distinct valid demonstrations found, m_F their total score, R_F(a) the part of it with answer a, and Z <= 1 the total
score of all valid demonstrations. The target is T(a) = R(a)/Z with R_F(a) <= R(a) <= R_F(a) + Z - m_F. Hence
R_F(a) <= T(a) <= 1 - (m_F - R_F(a)); the width is 1 - m_F; the distance of (b) from the target is at most
1 - m_F/Z; and an answer with R_F(a) > 1/2 is the target's majority answer whoever wrote the rollouts. Nothing in it
uses the writer's own numbers. Checked on the test rollouts (`RESULT-posthoc.json: interval, found`):

| K = 16 (K = 64) | W1 | W3 | W3s | W3p | W4 |
|---|---|---|---|---|---|
| arithmetic: interval width | 0.14 (0.13) | 0.15 (0.14) | 0.14 (0.13) | 0.14 (0.13) | 0.15 (0.14) |
| arithmetic: answers certified | 98% (98%) | 98% (98%) | 98% (98%) | 98% (98%) | 98% (98%) |
| variables: interval width | 0.16 (0.14) | 0.22 (0.18) | 0.17 (0.15) | 0.18 (0.15) | 0.31 (0.31) |
| variables: answers certified | 95% (95%) | 90% (93%) | 93% (93%) | 92% (95%) | 79% (79%) |

Every certified answer was the reference's top answer (all 48 writer-by-K cells), and the reference probability
left the interval by more than 0.03 in at most 0.3% of cases (the reference has its own noise). Certified answers were
right 100% of the time in arithmetic and 94-97% in variables: the certificate is about the target, not the truth.
This is the idea the checker of `wf11/searcher-proof` left untested ("let the alien searcher choose where the audit
looks; its errors then cost width, never correctness") at language-model size: the biased writers cost width at K = 4
(0.35 against 0.20 on variables) and little by K = 64. It works because the score is concentrated (section 7).

## 6. The smallest trust head (H7, exploratory): what real answers add here, and what they cannot add

A logistic map from the aggregated top-answer probability (and the family) to "the top answer is right", fitted on
the 300 calibration problems' real answers with W1's rollouts (K = 16), tested on the 300 test problems
(`code/trust.py`, `out/main/RESULT-trust.json`). Brier score (lower is better), raw probability -> after the head:

| Head on top of | W1 (as fitted) | swapped to W3 | to W3s | to W3p |
|---|---|---|---|---|
| (a) vote | 0.036 -> 0.037 | 0.264 -> **0.345** (accuracy 0.48, stated confidence 0.79) | 0.087 -> 0.057 | 0.129 -> 0.072 |
| (b) score-weighted | 0.033 -> 0.033 | 0.028 -> 0.025 | 0.035 -> 0.035 | 0.035 -> 0.034 |
| (c) importance-weighted | 0.037 -> 0.038 | 0.040 -> 0.042 | 0.038 -> 0.039 | 0.040 -> 0.042 |
| (c) plus search statistics, fitted on all writers | 0.038 | 0.044 | 0.037 | 0.039 |

- Expected and found: a trust head on top of the vote is only as good as the writer it was fitted with (0.037 becomes
  0.345 when the writer is swapped); on top of (b) or (c) it carries over to the three biased writers (within 0.01).
- Expected and NOT found: the head does not improve W1's Brier score. M's belief after thinking is already close to
  calibrated here: where the target gives its top answer 0.5-0.7 it is right 62% of the time (16 problems), 0.7-0.9:
  91% (34), 0.9-0.99: 99% (230); calibration error 0.037.
- A head that also reads search statistics (the vote's probability, whether vote and (c) agree, the effective share,
  the score found, the share of valid lines), fitted on all eight writers, is no better than the plain (c) probability
  (0.038 against 0.037): "trust in the search effort" had nothing left to learn beyond the score-based aggregation.
- What a reality-trained trust head would have to add, and this pilot cannot supply: (1) truth where M's belief is
  wrong. The target's top answer is wrong on 8% of the variable problems; (b) and (c) reproduce that from M's own and
  the biased writers (0.91-0.93), and 300 real answers could not single those problems out from the answer
  distribution alone. It would have to read the demonstration (your "trust in kinds of demonstration": here, steps
  that fetch a variable's value are the unreliable kind, steps that add are not). Not built. (2) Trust in a writer's
  own reported probability, which (c) takes on faith. (3) What to do when little of the score is found: the bound above.

## 7. Guarantee or finding; caveats; what I do not know

- **Guarantees** (by construction or proved): W1's vote is an unbiased estimate of the target; (c) is consistent for
  any writer that can write every valid line; (b) is exact once all the score is found; the interval of section 5.
  **Everything else is a finding**: 150 problems per family, one model, one tuning run per writer, one sampling seed.
- **The target is M's belief after thinking, not the truth.** The pilot shows that for the three biased writers built
  the score-weighted answer did not depend on the writer; it does not show that it is right. Where M's belief is right
  the answer is right (99% in arithmetic); where it is wrong (8% of the variable problems) the aggregation is wrong too.
- **Why it was this easy.** A demonstration here is 45-49 tokens in a rigid format, and M's probability is
  concentrated: 4 to 6 distinct lines per 64 rollouts carry 0.86 of it. (b), the certificate and the cheap correction
  rest on that. For free-form reasoning of hundreds of tokens the score of one text is negligible, the share found is
  near zero and the interval says nothing; (c) still applies, with weights whose spread grows with length. Not tested.
- **Writers I did not build.** (i) One that searches for wrong lines that M scores highly: an attack on the score,
  not on the vote; (b) and (c) would stay faithful to M, so it tests how wrong M's own belief is (W3s is a mild
  version). (ii) One that hides M's best line and shows a plausible second-best: (b) and (c) then return the
  second-best, and only the share of the score found reveals it; the helpful writer is this mechanism in the harmless
  direction. (iii) One that misreports its own probability to (c). (b) and the interval need nothing from the writer.
- **The helpful writer** makes the amplified answer better than the target (0.99-1.00 against 0.92 on variables) and
  therefore different from it. Under your requirement (the same amplified prediction whatever the search) that is a
  miss at the compute tried; under "be right" it is a gain. The design has to say which it wants.
- W3, W4 and W3s were tuned on text written or labelled with the generator's exact answers: the writers' power is
  given by construction. No exact answer enters M, the score or any aggregation. Runs are in bfloat16 (rounding 0.023
  nats on average, 8e-5 in float32). "Effective share" counts invalid rollouts as zeros, hence 0.91 for W1.
- **Departures from the pre-registration.** None in the analysis of the K = 64 run. Added before any result was read:
  the K-extension (`PREREG-ADDENDUM.md` says ~17:42Z; the hash file records 17:38:12Z). Added after the results and
  labelled post hoc: the worked example, the compute table, the bound and interval (its threshold 1/2, K = 16 and the
  0.03 tolerance were set once, after seeing the data), the rank correlation. Figure 3 was planned as divergence
  against distance; that result is not clean, so it shows the extension. Three registered expectations of mine were
  wrong: (b) would not return to the target; the trust head would improve W1's Brier score; E1 for W3.
- **Against FRAME.** The pilot works in your reading of section 5: target in proportion to the score; the writer's
  internals are forgotten at submission, and the one search-side number (c) reads is the kind of "augmented submitted
  state" you allowed; the trust head is fitted on real answers only. Of your consistency requirement it tests "the
  amplified prediction is the same for different rollout policies", not the query-order form (amplified P(X) then
  P(Y given X) against P(Y) then P(X given Y)). Decisions (Q4) are untouched. Under the older reading "the search
  returns the top one" (Q3 b) only the accuracy rows apply; they hold.
- **I do not know:** whether this survives longer, free-form demonstrations; what an attack on the score does;
  whether distilling (b)'s output back into the rollout policy keeps the helpful writer's blind spot.

## 8. Files, compute, job

- One Slurm job, id **57877**, one L40 on the MATS cluster, 16:35:43Z-19:01:27Z (2 h 26 min), no queue wait. Inside
  it: sizing pilots (seed 7), M's rollouts on the tuning problems (9 min), three LoRA tunings (135, 104, 68 s), the
  test run (54 min, 230,400 rollouts), the calibration run (18 min, 76,800), Qwen3-4B's direct answers (30 s), the
  K-extension (32 min, 147,456). No other GPU, no paid API, no gated model. Analysis on CPU at nice 19.
- Cluster: `/cluster/home/user/wave13/llm-pilot/` (`code2/` = the code as run, hashes as in `PREREG.sha256`;
  `lora/` = the three adapters). Here, under `/home/agent/workspaces/mats/work/deep-study/wf13/llm-pilot/`:
  `PREREG.md`, `PREREG-ADDENDUM.md`, `PREREG.sha256`; `code/` (`tasks.py`, `lm.py`, `run_main.py`, `train_lora.py`,
  `analyze.py` registered analysis, `verdict.py`, `ext_table.py`, `trust.py`, `posthoc.py`, `selfcheck.py`, `figs.py`,
  pilots, `config.json`); `out/main/` (`test.jsonl`, `calib.jsonl`, `ext.jsonl`: every rollout with the writer's
  log-probability and M's score; `RESULT-*.json`, `VERDICT.json`, `verdict.txt`, `ext_table.txt`, `direct_4b.json`);
  `out/pilot*.json`; `logs/`; `figs/` (validated five-colour set; line style and marker repeat the writer).
- Self-check (`code/selfcheck.py`, fresh code): 24 headline numbers recomputed from the stored rollouts, 0 mismatches;
  the exact answers re-derived from the problem text by a separate evaluator, 0 of 600 differ. To re-derive every
  number (CPU, about a minute): `python3 code/analyze.py out/main`, `code/verdict.py out/main`,
  `code/analyze.py out/main ext`, `code/ext_table.py out/main`, `code/trust.py out/main`, `code/posthoc.py out/main`.
