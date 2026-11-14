# PREREG: a biased rollout policy cannot move the answer (language-model pilot, wf13/llm-pilot)

Model: Claude Fable 5.2. Written 2026-11-10 ~17:00Z, before any run on the test or calibration problems.
Everything before this file used pilot seed 7 only, for sizing (which model, which task sizes, speed, whether each
writer functions); no aggregation was compared there. Sizing outputs: `out/pilot_17base.json`, `out/pilot_4b.json`
(log only), `out/pilot2_17base.json` (log only, stopped early), `out/pilot3_17base.json`, `logs/prof_f32.log`.

## Design (Kelvin's, FRAME section 5)
- Simple world model M: frozen `Qwen/Qwen3-1.7B-Base` (bfloat16) with one fixed few-shot prompt per family (header +
  4 solved examples, seed 9001). It scores a DEMONSTRATION: one line of worked steps ending in " answer X".
- Score R(d) = product of M's token probabilities of d given the problem, times M's probability of stopping after it
  (stop = any token containing a newline, or a special token), times a fixed format check (goal factor; shape only:
  right number of steps of the right shape, then " answer X"; it never checks content). Target T = R normalised over
  format-valid demonstrations; the target answer distribution is T's marginal over X.
- Rollout policy (writer): anything that writes demonstrations. The search is the aggregation of K rollouts.
- Families (exact answers from a generator, `code/tasks.py`): `arithL` n=3 (three chained operations, values up to
  9999) and `vars` n=4 (three variables, four assignments, copy-substitute-compute steps). Sizing on seed 7:
  direct greedy 6% / 16%; one step-by-step rollout right 93% / 76% of valid; vote of 32 right 100% / 81%.
- Problems: test = 150 per family (seed 200); calibration = 150 per family (seed 300); tuning = 1500 per family
  (seed 100). Test and calibration problems are forced disjoint (by text) from tuning problems, few-shot examples
  and pilot problems; calibration disjoint from test. The overlap count is printed by `run_main.py`.

## Writers (all at K = 64 rollouts per test problem; 32 per calibration problem)
- W1: M itself, temperature 1. REF: 256 further independent rollouts of M (the reference for T's marginal; its first
  64 are "W1b", a second W1, used as the noise floor).
- W2a, W2b: M at temperature 0.7 and 1.3. W2c: M with another prompt (other header, 2 other examples, seed 9002).
- W3 (biased, tuned): LoRA of M (r=16, 2 epochs, zero-shot prompt) on, per tuning problem, one demonstration with the
  particular error (arithL: last result gets last digit 7; vars: last step returns the variable's old value) and one
  of M's own rollouts.
- W3s (biased, selective): LoRA on M's own rollouts on tuning problems, keeping every wrong-answer rollout and 10% of
  the right-answer ones (a policy trained to prefer M-plausible demonstrations that end wrong).
- W3p (biased by prompt): M with the four few-shot examples replaced by their erroneous versions.
- W4 (helpful): LoRA on the generator's correct demonstrations.
- Direct baseline: M's answer with no steps (same four problems as examples, answers only): greedy, and 64 samples.

## Aggregations of the first K rollouts into an answer distribution (`code/analyze.py`, fixed before the run)
(a) vote over valid rollouts. (b) dedup-score: each distinct valid demonstration once, weight R(d). (c) SNIS: each
valid rollout with weight R(d) / q(d), q = the writer's own probability of the rollout (recorded while sampling),
self-normalised. Invalid rollouts weigh 0 in all three; if none is valid the aggregation abstains (counted wrong;
distance 1). K in {1,2,4,8,16,32,64}; for K < 64 the 64 rollouts are cut into 64/K disjoint blocks and averaged.
Ties in the top answer are broken by a fixed rule on the answer string.

## Metrics
- acc(w,g,K): top answer equals the exact answer.
- D(w,g,K): total variation between the aggregated distribution and the reference (vote of REF).
- move(w,g,K): total variation between writer w's and W1's aggregated distributions (same g, K, block);
  noise floor = move(W1b,g,K); excess movement EM = move - noise floor.
- mass on the biased answer; SNIS effective sample share; dedup found mass (sum of R over distinct found).
- Divergences: E_M[log R - log q_w] on W1's first 16 valid rollouts per problem (every writer scores M's
  demonstrations), E_q[log q - log R] on the writer's own valid rollouts; valid rates; tokens per demonstration.
- Check: for W1 the sampler's log-probability must equal M's score up to rounding (bfloat16; in float32 the two
  paths agree to 8e-5, `logs/prof_f32.log`).
- 95% intervals: bootstrap over problems.

## Hypotheses and thresholds (K = 64 unless said; each must hold in BOTH families for "major")
- H1 (thinking helps): acc(W1,vote,64) - direct greedy accuracy >= 0.30.
- H2 (the plain vote follows the writer): EM(W3,vote,64) >= 0.20 and acc(W3,vote,64) <= acc(W1,vote,64) - 0.15.
  A biased writer "bites" in a family if its EM(vote,64) >= 0.10; H3-H5 are judged on the writers that bite.
- H3 (score-based aggregations do not follow): for every biting writer, EM(w,snis,64) <= 0.10 and <= EM(w,vote,64)/3,
  and the same for dedup. Major: both aggregations, all biting writers. Minor: at least one aggregation for W3.
- H4 (return to the target, and its cost): D(w,snis,64) <= D(W1,vote,16), i.e. the bias costs at most 4x rollouts.
  Major: all biting writers. Minor: W3 only, or only against D(W1,vote,4). Negative: neither; then the divergence
  E_M[log R - log q_w] is reported as the explanation.
  Registered expectation for dedup: it will NOT meet this bar, even from W1, because it under-weights the part of
  M's distribution that K rollouts do not find; D(W1,dedup,64) and the found mass are reported.
- H5 (accuracy cannot be moved): acc(w,snis,64) and acc(w,dedup,64) >= acc(W1,vote,64) - 0.03 for every biting
  writer (major); for W3 only (minor).
- H6 (exploratory, no bin): the smallest K with D(w,snis,K) <= D(W1,vote,16) is non-decreasing in
  E_M[log R - log q_w] across writers.
- H7 (exploratory, only if time): smallest trust head = a logistic map from the aggregated top-answer probability
  (and family) to "the top answer is right", fitted on the calibration problems' real answers only, tested on the
  test problems. Expected: it lowers the Brier score of W1's aggregated answer; applied to a biased writer's
  rollouts it stays within 0.02 Brier of its W1 value on top of SNIS and degrades by >= 0.05 on top of the vote.

## Outcome bins
- MAJOR: H2, H3 major, H4 major and H5 major.
- MINOR: H2 holds, H3 at least minor and H5 at least minor (the answer cannot be moved, but for some writer the
  distribution does not return at K = 64); the divergences are then the result.
- NULL: H2 fails (the biased writers do not bite; the setup says nothing).
- NEGATIVE: H2 holds and H3 fails for W3 (a score-based aggregation follows the writer).

## Guarantee or finding
By construction: W1's vote is an unbiased sample estimate of T's marginal; SNIS is consistent for any writer that
can produce every valid demonstration; dedup-score is exact once all of T's mass is found. Everything at finite K,
and every number about the biased writers, is a finding. With M's own likelihood as the score, T is M's belief after
thinking step by step: the pilot can show independence from the writer, not truth.

## Compute, baselines, seeds
- Equal compute: K rollouts per writer and problem; network token positions are recorded per writer. (b) and (c)
  need one extra pass of M over each distinct valid demonstration (at most as many token positions as the rollouts),
  so the equal-compute comparison is (b),(c) at K against the vote at 2K.
- One-pass networks on the same axis: M's direct answer; Qwen3-4B's direct answer on the same test problems
  (run after the main job if time allows). Memorising baseline: zero by construction (no test problem in tuning).
- Seeds: few-shot 9001/9002; pilot 7; tuning 100; test 200; calibration 300; LoRA 1, 2, 3; sampling seeds derived
  from the set seed and the problem index. One Slurm job (57877) on one L40.
