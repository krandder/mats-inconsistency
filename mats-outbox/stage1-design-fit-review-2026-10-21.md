# Stage-1 design-fit review: does the work test what Kelvin asked for?

Reviewer: board investigator, model Opus 5.5 (claude-opus-5-5[1m]). Read-only. Checked 2026-10-21 00:41-00:50Z.

All repository line numbers refer to `origin/main` at 773586a of `futarchy-fi/inconsistency-search`, exported read-only with `git archive` to `scratch/stage1-review/src/`. The shared checkout `/home/agent/workspaces/inconsistency-search` is one commit behind origin (`git status -sb`, 00:43Z).

## Kelvin's stated bar (his words, the yardstick for every finding)

- K1 "The output (value) needs to be a high-dimensional." (research-orientation-2026-10-20-addendum.md:37)
- K2 "it needs to converge near the full best bayes for all kinds of information structures we can think of." / "Until it does, we know something is wrong about our setup." (addendum:50-51)
- K3 "we need to be able to learn XOR like information structures ... Our aggregation mechanism cannot be worse than a transofrmer with self-attention that can read all the info sequentially." (addendum:54)
- K4 Gemini must "never let us water-it-down to something easier to validate" (addendum:42)
- K5 "the idea is to use as much of what is already known as possible." (agg-reuse-plan:3)
- K6 attention should find "what queries are needed to improve the consistency" (addendum:30)
- K7 "We want the agents to report much more than that, obviously." (rca-scalar-reports:59)

## Measurements behind the counts (re-runnable)

- `classify_runs.py` gives `auto_labels.tsv`. It is an automatic rule over the code each run executed: it calls `agent_inputs(` means rich, it imports aggbench but not that means scalar, it passes `alignment="unaligned"`, it calls `make_env("Z…")`, it references `full_context_transformer`.
  - I hand-labelled all 37 run directories from their READMEs and runners.
  - Interface agreement is 36/37. The one disagreement is L10: the rule says rich, but by hand its value is a scalar forecast and only the key is learned.
  - Agreement on alignment, zoo use and transformer bar is 37/37.
- Totals over 33 aggbench run directories, from the rule:
  - 20 scalar-report runs and 13 rich-message runs;
  - all 13 rich runs used aligned inputs; 0 used unaligned inputs;
  - 0 used any zoo environment, including XOR;
  - 0 compared with the full-context transformer.
  - By distinct experiment: 13 experiments ran on aggbench (gemini: highdim, scalar, kylevoi; opus: L1-L10). Only 3 had vector values (gemini highdim, gemini kylevoi, L2), plus L10 with a learned key only. All of them were aligned.
- `visibility_floor.py` (run at 00:47Z, 2000 dev episodes) computes the exact posterior given what agents can actually see at keys `none`, as regret to aggbench's `bayes`:
  - E2: 0.0391 [0.0278, 0.0506];
  - E3-sybil: 0.0391 [0.0278, 0.0506];
  - Z6 XOR: 0.1492 [0.1297, 0.1690], against a prior-only regret of 0.2466.

## Findings

### Blocks the research question

**B1. Kelvin's bar has not been measured. The harness README promises a comparison with a component that is not in the harness.**
- **Violates:** K3.
- **Evidence:**
  - aggbench/README.md:10-13 says headline results "are compared with `full_context_transformer`".
  - A grep of origin/main aggbench and of all results finds the name only in that README line.
  - The implementation exists only on the unmerged branch `aggzoo`, commit ea8e3af (aggbench/transformer.py).
  - Its training run is in progress: `ps`, about 00:46Z, shows `python3 -m aggbench.report --transformer --write`, PIDs 731246-731248, about 55 CPU-minutes each. No `baselines_transformer.json` exists yet.
  - 0 of 37 run directories reference the bar.
  - As written on that branch, the bar is trained with "the lanes' budget (20k train episodes, 3000 AdamW steps)". Kelvin's bar is about what a transformer can learn, not what it learns on a small budget. A weakly trained transformer is a bar that hand-structured aggregators pass easily.
- **Fix:**
  - Merge the transformer reference and finish its run.
  - Add a data-scaling curve to convergence (PFN-style; the simulators are free), and treat the converged transformer as the bar.
  - Give it exactly the same per-agent inputs as the encoders.
  - Until then, no result should say it "meets" or "approaches" the design.

**B2. XOR and the rest of the structure zoo have never been run with any learned aggregator.**
- **Violates:** K2, K3.
- **Evidence:**
  - 0 of 37 run directories call `make_env("Z…")`.
  - Z6 appears only in the baseline table, where every scalar baseline equals the prior: aggbench/README.md table, Z6 and Z6-parity3 rows.
  - The held-out structures (zoo.HELDOUT) have never been evaluated.
- **Fix:** run the transformer bar and every candidate aggregator on Z1-Z7, unaligned. For the generality claim, train one model across the open zoo and evaluate it once on the held-out families.

**B3. At the headline condition, the Bayes ceiling uses information that no agent has. "Keys" mix the agent's knowledge of itself with provenance for the aggregator.**
- **Violates:** K2. The gap to `bayes` cannot close by construction, so failures would be wrongly blamed on the aggregator.
- **Evidence:**
  - `agent_inputs(visibility)` gives each agent's encoder the same key the aggregator would get (envs.py:70-93). At `none` that key is empty (envs.py:383 for E2/E3, :275 for E1, zoo.py base `episode` "key.none").
  - The agent's own report, and `bayes`, use the piece strengths mu (envs.py:372-374).
  - A rich encoder at keys `none` sees only the raw x (envs.py:381). In E1 it sees s_i without A_i, B_i, sigma_i. In Z6 it sees one bit without knowing which bit or how accurate it is.
  - The measured floor at keys `none` (above) is 0.039 nats in E2. That is worse than the scalar `oracle_clearing` ceiling (0.017) and close to `extremized_log_pool` (0.045). In Z6 the floor is 0.149, 60% of the gap between the prior and Bayes.
  - The README's headline rule (aggbench/README.md:208: unaligned, keys `none`/`noisy`) therefore cannot test whether the aggregator gets close to Bayes, and "beat `oracle_clearing`" is impossible there for raw-observation messages.
- **Fix:**
  - Always give the agent's encoder its self-knowledge: its own likelihood model, meaning the mu of its pieces; A_i, B_i and sigma_i; which bit it reads and with what accuracy.
  - Let visibility control only metadata that goes to the aggregator.
  - Add a visibility-matched exact ceiling (Bayes given everything the encoders can see) and report regret to both ceilings.

**B4. Every rich-message result uses aligned slots and keys that are the true structure. These are giveaways the README itself demotes to diagnostics. One of them is still recorded as "Decided".**
- **Violates:** K1, K2, K4.
- **Evidence:**
  - All 13 rich run directories are aligned; the default is `alignment="aligned"` (envs.py:70).
  - Full keys are the oracle structure: E2 coverage bits and coverage x mu (envs.py:376); E1 A, B, log sigma (envs.py:268); Z6 bit one-hot and accuracy (zoo.py ParityEnv.make_keys).
  - Gemini's own audit found that a model with 0 parameters reaches 0.0000 regret on E2, E3-sybil and E3-collude at full keys (expagghighdim/README.md:5).
  - Yet agg_highdim_summary/README.md:5 still grades "Gain", and line 13 says "Decided: Adopt Slot-Equivariant Key-Value Messages + Gram-Confiability Clearing (d_v = K) as the stage-1 reference architecture". expagghighdim/README.md:13 keeps the same "Decided" line.
  - With d_v = K, the "message" is the aligned per-piece vector of evidence itself.
  - Even unaligned, a piece seen by two agents is the same float (envs.py:381; README:209-211), so a hash-and-deduplicate rule reaches Bayes without learning anything about overlap.
- **Fix:**
  - Retract the "Decided" lines and the "Gain" grade.
  - Rerun unaligned.
  - Add private reading noise per agent for shared pieces, so that overlap has to be inferred statistically. The Gaussian Bayes ceiling stays exact.

### Distorts conclusions

**D1. Harness defaults and rules still describe the scalar design, so the first RCA fix ("make rich messages the default API", rca:86) has not been done.**
- **Violates:** K1, K7.
- **Evidence:**
  - README:38-39: "An aggregator gets only `batch.view(visibility)`: reports, keys".
  - Rule 2 (README:170-171): "Aggregators see `batch.view(...)` only ... `obs` never reach `predict`". Read literally, this forbids the rich-message pipeline.
  - The learned-aggregator interface is `.fit(view)`/`.predict(view)` (README:85, controls.py:153), and the example uses `tr.view("full")` (README:80).
  - README:185: "Experiments **may** ... train per-agent encoders".
  - Rule 4 (README:173-174) compares results only with two scalar baselines.
  - The shared checkout's README:3-5 still says "Every agent reports a VALUE (its exact local posterior)".
- **Fix:**
  - Make rich messages and unaligned inputs the default.
  - Rewrite rule 2 as "the aggregator sees messages; encoders see only their own agent's inputs".
  - Change rule 4 to require comparisons with `bayes`, the visibility-matched ceiling and the transformer bar.
  - Pull the shared checkout.

**D2. The literature-reuse integration, in progress and not committed, repeats the scalar-report failure, including for the transformer bar.**
- **Violates:** K3, K5, K7.
- **Evidence:**
  - /tmp/aggext/aggbench/external/learned.py:33-42: the tokens for DeepSets, Set Transformer and TarMAC are the scalar report plus the key.
  - tabpfn_adapter.py:7-11 calls its report-based encoding "the 'transformer that reads all the info' reference". It uses TabPFN v2 with `n_estimators=1` and at most 1000 context rows.
  - agg-reuse-plan:99 names the PFN "also Kelvin's full-context transformer bar".
  - File times are 00:25-00:33Z, before the RCA file (00:40:12Z).
- **Fix:** build tokens from `agent_inputs` (raw observation plus self-knowledge, unaligned). Label any version that uses reports as "scalar-report baseline". Do not call TabPFN on reports the bar.

**D3. "Attention" is often only a weighted pool of scalar reports, and the conclusions are then stated about attention in general.**
- **Violates:** K3.
- **Evidence:**
  - L3 README:15: arms map keys only to per-agent weights, with readout l0 + Σα_i(l_i − l0).
  - L8 README:12: the same form of readout. L8 README:20-23: "Neither size nor data fixes this".
  - L6 README:1: "understanding what attention must be". L6 README:8: "softmax attention ... loses more as N grows".
  - L7 README:12: trust weights per source.
  - L4: g_j(r_j − l0).
  - Literature map:74 gives the normalization argument for convex pools.
  - A pool with one scalar weight per source cannot express XOR, because every Z6 report equals the prior, or E1's correlated noise (L3 README:37; L6 README:16). These results say nothing about a transformer with vector values and an MLP, which is Kelvin's bar.
- **Fix:** rename these arms "scalar-weight readouts" and retest each claim about attention against the transformer bar.

**D4. Some successes follow by construction from a hand-coded solver fed oracle keys, and the required controls cannot detect this.**
- **Violates:** K4, K2.
- **Evidence:**
  - L1 README:21-22: "With phi_ik = sqrt(2) mu_k c_ik this *is* oracle_clearing ... only the keys are learned".
  - L8 README:14 ("With full keys it can represent `oracle_clearing`") feeds its headline "transfers almost perfectly" (README:20).
  - L1 README:1 promises "without being told who knows what", but its keys are the coverage bits, which are exactly who knows what.
  - Rule 5 (README:175-176) requires only random-weight and untrained controls. These sit near prior-only regret and always fail, so they cannot reveal success by construction.
  - The 0-parameter plug-in control was found only by gemini's manual audit.
- **Fix:** make a control mandatory that plugs the true structure or keys into the hand-coded part with no learned parameters, and credit learning only with the improvement over it.

**D5. The RCA's relabel list is incomplete, and two copies of the same run disagree on its grade.**
- **Violates:** K7.
- **Evidence:**
  - The RCA (rca:75, :88) relabels L1 and L3-L8.
  - L9 is also scalar: its tokens are "[report, key]" (L9 README:79).
  - Gemini's 11 agg_scalar_* directories and expaggscalar are scalar.
  - agg_scalar_summary/README.md:5 says "Overall Status: ● Gain"; the same run as expaggscalar/README.md:5 says "Informative".
  - In all, 20 run directories are scalar.
- **Fix:** relabel L9 and all of gemini's scalar directories, and make the two copies agree.

**D6. Generality is claimed from one family per run.**
- **Violates:** K2.
- **Evidence:**
  - All 13 aggbench experiments ran only on E1-E4. E3 is E2 plus copies or distortions (envs.py:551-558).
  - Models are trained for one environment and tested on the same one, except L5 "pooled" and L7, which mix E2/E3 variants.
  - No model has been tested across structures.
- **Fix:** train PFN-style across the zoo (reuse plan, row 1) and evaluate on held-out structures.

**D7. L9's conclusion is assumed by its environment.**
- **Violates:** K4.
- **Evidence:**
  - L9 README:8: "Rollouts behave like information gathering with correlated sources."
  - In E4 the rollouts are defined as linear-Gaussian sources with hand-set error parameters (envs.py:19-26, 499-516). No model is rolled out.
- **Fix:** state the result as "under this rollout-error model", and test with real rollouts of a trained any-order model.

**D8. The query/VoI question cannot be tested in these environments.**
- **Violates:** K6.
- **Evidence:** L4 README:461: with keys free and the structure known, the optimal policy is non-adaptive, so "queries that depend on the reports have nothing to gain".
- **Fix:** use environments where the value of a query depends on content (Z7 regimes, Z5, Z6, Z2), with costly keys and unaligned inputs.

**D9. The transformer baselines used to justify hand-built structure are strawmen.**
- **Violates:** K3.
- **Evidence:**
  - agg_scalar_summary/README.md:8 cites "Standard token attention (`aggbench.examples.smoke_attention` ...) got 0.241". That example is "SMOKE ONLY ... a few hundred steps" (examples/smoke_attention.py:1-3, steps=300, width 32).
  - `naive_mha` was trained on 2000 episodes (agg_highdim_summary:10).
- **Fix:** compare with the trained transformer bar.

**D10. "Collusion" is honest duplication, not collusion.**
- **Evidence:**
  - envs.py:360-361 and :372-373: colluders take the union of their coverage and honestly report the pooled posterior. In rich mode they are simply honest agents with high overlap.
  - The biased agent's raw observations are honest; transformer.py on the aggzoo branch says so itself.
- **Fix:** add a strategic misreporting variant before claiming robustness to collusion.

**D11. The kylevoi README misdescribes the environments.**
- **Evidence:**
  - expaggkylevoi/README.md:23 says E1 has "noise precision τ_i = 1/(1 − ‖k_i‖²)". In fact E1 is s_i = A_i θ + B_i η + ε_i (envs.py:3-7, 252-265).
  - The same line says "agent 0 has an additive bias". In fact the bias is temp 3 plus shift 1.5, at a shuffled position (envs.py:291-293, 367-370).
  - Line 33 is headed "E2 (Non-Redundant Pieces)", but coverage overlaps (envs.py:354-357).
- **Fix:** correct the README against envs.py.

**D12. The literature map pre-excuses failures.**
- **Evidence:** agg-literature-map:76 says "'keys need provenance' is the expected result, not a failure". The impossibility results it relies on concern forecast-only (scalar) aggregation. With rich messages, Bayes is reachable; the aligned 0-parameter skeleton reaches 0 regret.
- **Fix:** limit the statement to scalar reports.

### Minor

- **M1. Rich messages are only a pass-through here.** Toy observations are tiny: one symbol in the zoo, at most 6 pieces in E2, 2 scalars in E1. A "rich message" just forwards the raw input. No environment tests what an agent should say under limited bandwidth, which is the stage-2 question of keys describing "what kind of information they have" (addendum:34).
- **M2. The transformer bar on the aggzoo branch gets an extra input.** Its tokens include the report, a local posterior computed with the true mu. At keys `none` the bar therefore sees information that raw-observation encoders lack. Match the inputs.
- [removed]
- **M4. Log timestamps run ahead of the clock.** OPUS-MATS-CONTEXT.md logs the RCA at "~01:20Z", but the RCA file's mtime is 00:40:12Z and `date -u` read 00:41Z. Times in this log are not reliable evidence.

## What is genuinely aligned and trustworthy

- **Exact ceilings with independent checks:** brute-force oracle checks to 1e-15 (README baseline footer), mutation tests, Monte Carlo checks of the zoo tables, calibration checks (README:227-231).
- **The locality check** (messages.py:11-27) correctly enforces that no agent's message depends on another agent's input.
- **Test split and harness hygiene:** the test split is locked (envs.py:189-190) and dirty harnesses are rejected.
- **The README is candid about weaknesses:** it names the aligned-slot giveaway and the identical-float giveaway (README:208-211), and it demotes aligned and full-key runs to diagnostics.
- **The zoo's coverage:** it has exactly the structures Kelvin means: XOR, common cause, hypothesis vs evidence, sub-questions and regimes, with held-out families reserved.
- **The transformer bar on the aggzoo branch** is the right object, with a "matched bandwidth" rule that attributes the gap (transformer.py docstring, README diff in ea8e3af). It is running now.
- **Honest self-audits:** gemini's 0-parameter skeleton (expagghighdim:5, kylevoi:18-20); L4's statement that adaptivity cannot help (L4:461); L2's distillation diagnostic and its plain admission that E1 fails (L2:89-94); L1's "keys describe information, not honesty".
- **The E1 failures are a real signal that the setup is wrong (K2):** learned aggregators do not reach Bayes; L2 is at 1.21 nats and L1 at 0.354.
- **The reuse plan's first row (a PFN trained over a prior on structures)** is the right recipe and coincides with Kelvin's bar.

## What I could not check

- Whether /experiments or the leaderboard now label stage-1 lanes as scalar. I found no page generator that references them in /home/agent/workspaces/mats, /tmp/om or site/experiments-page.
- The results of the transformer bar run; it was still running at about 00:46Z.
- The visibility-matched floor for E1 and E4, and for keys `noisy`. I computed it only for E2, E3-sybil and Z6 at keys `none`.
- Whether the shared-float giveaway was actually exploited in any run. All runs were aligned, so it was moot.
- No GitHub API was used (0 REST calls); everything was read from local git objects.

## Files

- /home/agent/workspaces/fable-strategy/scratch/stage1-review/classify_runs.py
- /home/agent/workspaces/fable-strategy/scratch/stage1-review/auto_labels.tsv
- /home/agent/workspaces/fable-strategy/scratch/stage1-review/visibility_floor.py
- /home/agent/workspaces/fable-strategy/scratch/stage1-review/src/ (origin/main snapshot)
- /home/agent/workspaces/fable-strategy/scratch/stage1-review/runners/
