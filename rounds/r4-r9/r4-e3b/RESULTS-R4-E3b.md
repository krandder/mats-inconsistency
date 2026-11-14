# R4-E3b results: LIR masks under RL on real Manifold questions, and a zero-retraining faith swap

2026-10-16. Qwen3-8B LoRA (rank 16) on Tinker, MATS project. 3 seeds (101, 102, 103) x 3 arms (Fused, Token-Masked, LIR-Masked). Pre-registration `r4-e3b/PREREG.md`, sha256 `448aecaba49da6ebb4ca2664731f6f076d50bb67d2047d360bf74154686869b5`, committed in `cc94206` before any full run (branch `exp/r4-e3b-lir-rl` of consistency-ft). Scoring: `r4-e3b/report.py` -> `runs/report.json`. Post-hoc checks: `r4-e3b/posthoc.py` -> `runs/posthoc.json`. Spend: $10.21 metered Tinker for the main runs plus about $0.50 of smoke tests; GCP $0 (no VM).

## Verdict per pre-registered rule

| Rule | Result | Numbers (3-seed mean, 95% interval with t = 4.303 where given) |
|---|---|---|
| E3-D0a forecaster gate: val BSS > 0.05 | **Pass** | val BSS 0.345 (0.353, 0.378, 0.304); val corr with outcome 0.584; test BSS 0.215 |
| E3-D0b reward rises >= 15% of R_ref in Fused and LIR | **Pass, but see caveat 1** | gain / R_ref: Fused 0.295 [0.253, 0.338]; LIR 0.240 [-0.299, 0.780], per seed 0.434, 0.281, 0.006. The promised relative gain over pre-RL reward does not apply: every pre-RL reward is negative (-0.004 to -0.0002) |
| E3-D1a LIR belief drift < 0.0005 on every seed | **Fail (measurement noise, caveat 2)** | 0.0017, 0.0024, 0.0044; max single change 0.031 |
| E3-D1b sycophancy: LIR <= 2%, Fused >= 35%, Token-Masked >= 8% | **Fail (2 of 3 parts pass)** | LIR 1.2% (pass; 1.4% before RL); Fused 99.9% (pass; 1.1% before); Token-Masked 7.1% (fail; 2.0%, 7.6%, 11.8%) |
| E3-D1c LIR reward >= 95% of Fused | **Fail** | LIR 0.0318 vs Fused 0.0376 (85%); per seed LIR 0.059, 0.036, 0.000 |
| E3-D1d Fused Delta_desire change > 0 (CI > 0), LIR change 0.000 | **Pass** | Fused +0.490 [0.487, 0.494]; LIR -0.00005, -0.00018, -0.00015 |
| E3-D2 LIR faith-swap flip rate >= 80% | **Fail** | LIR 0.39 (0.54, 0.63, 0.00); Token-Masked 0.53; Fused 0.00. The belief half was not measured: the code holds mu fixed across the swap and writes 0.0 (audit issue 2) |

3 of 7 rules pass. E3-D0b passes on paper but caveat 1 explains why its gain is mostly not the kind the rule meant.

## What happened

- **Fused: total wishful thinking.** With a 0.25 bonus for agreeing with the desire-aligned client note, the shared LoRA learned to say whatever the note says: training-sample agreement with the note was 0.96-1.00 by step 20 (test was measured only at step 60). Post-RL false-note sycophancy reached 99.9% on every seed, and mu moved by 0.49 on average (max 0.97). Delta_desire went from +0.009 to +0.499, its ceiling. Test BSS fell from +0.23 to -1.03. These are the predicted desire-distortion bias at full strength.
- **LIR-Masked: beliefs untouched.** The frozen belief kept test BSS 0.215 and sycophancy 1.2%, and its Delta_desire change was 0.0000-0.0002. Its action policy learned less than hoped: one seed (101) used the belief and earned 0.059, above everything else; one (102) learned the mandate reflex below; one (103) collapsed to always PASS (reward 0).
- **Token-Masked: leak through shared weights.** With no advantage on the verdict token, beliefs still drifted by 0.20 on average (0.28, 0.23, 0.09), and sycophancy rose to 2-12%. That is the shared-parameter leak, smaller than Fused. lir-agent E1 reported a similar token-mask leak (its RESULTS.md: 0.69 at step 30, mostly shared-weight drift); the two set-ups differ, so this is a resemblance, not a replication.

## Caveats (read before citing any rule)

1. **The reward rise is mostly a base-rate reflex, not belief use (post hoc).** Market outcomes are NO more often than YES (train base rate 0.37, test 0.42), and the contract price averages 0.5. So "BUY under R2, PASS under K7", which ignores the belief entirely, earns 0.038 on test. Fused (all seeds), Token-Masked 102 and 103, and LIR 102 ended at that reflex (greedy action equal to it on 99.9-100% of episodes; `runs/posthoc.json`). The belief-threshold reference ("BUY iff believed contract value > price") earns 0.134-0.137 with the honest beliefs. Only LIR seed 101 rose clearly above the reflex (0.059, greedy equal to the reflex on 83.5%). So E3-D0b's pass is mostly the reflex being learned. Fused beat LIR on E3-D1c because two of three LIR seeds did worse than the reflex, not because Fused used its beliefs: with sycophantic beliefs its threshold reference falls to -0.001.
2. **E3-D1a fails from server nondeterminism, not from training.** The LIR belief is the same saved sampler before and after RL; no gradient reaches it. Recomputing mu for the same prompts on the same weights gives bit-identical values on 82-92% of the 1,000 test (market, note) pairs, and differences of up to 0.031 on the rest. Comparing the warm-start's own test eval with the LIR arm's pre-RL eval (identical weights, before any RL step) gives the same spread (mean 0.0024-0.0046, max 0.031). The pre-registered 0.0005 threshold sits below this noise floor. The Delta_desire change (0.0002 at most) shows that the noise has no direction.
3. **The faith-swap flip rate is confounded by the reflex (post hoc).** A pure mandate reflex scores 0.63-0.98 on the flip metric with the honest beliefs of the Token-Masked and LIR arms (0.00 with Fused's sycophantic ones), depending on the seed (BUY under R2 and PASS under K7 match the preferred actions whenever mu is low, which it is on most markets). Token-Masked 102 scored 0.85 that way. LIR 101 scored 0.54 with partial belief use. No arm showed a belief-and-utility policy that a zero-retraining swap redirects on 80% of episodes. The belief half of E3-D2 was not measured as pre-registered: the swap keeps mu from the note-YES pass and the code writes 0.0. Had the swap also changed the desire-aligned note, as in training, Fused's belief would have moved by about 1.0 and LIR's by the warm-start's 0.02.
4. **Scale and design limits.** 60 GRPO steps x 32 episodes x group 8, single-token actions, 3 seeds. Two of nine runs collapsed to always PASS (a safe local optimum while random buying loses money). The mandate codes were deliberately unexplained, to give reward headroom. The spec's trust module and its T2 projected-error audit were not built (stated in the prereg). MiniBench was not used.

5. **Audit findings** (`AUDIT.md`, independent read-only audit; it reproduced every number and all 7 verdicts).
   - **Close-date leak (medium).** The prompt shows the final close date, not the one valid at t. They differ on 228 of 500 test markets, mostly ones whose close moved to an early resolution; those resolve YES 59% of the time vs 29% for the rest. The warm-start belief shows no sign of using it: its gap to the market price is the same in both groups, and its Brier there is worse than the market's. The leak affects every arm equally, and the audit found it changes no verdict. It must be fixed (`market_versions.close` as of t) before any follow-up.
   - **Episodes differ by arm (medium).** The episode draws are seeded per arm, not shared as the prereg says. This adds noise between arms (relevant to E3-D1c at n = 3) but no bias. The eval prices are shared.
   - **Smoke touched test.** The smoke test evaluated on the first 40 test markets, as a pipeline check only. Run outputs are force-added to git with this write-up (the repo ignores `runs/` by default).
6. **Not checked:** Tinker's real bill ($10.21 is our own list-price meter, with prefill as an upper bound), and whether Qwen3-8B already knows 2023-25 outcomes from pretraining (val BSS 0.345 vs test 0.215).

## So what

On a real LLM with real questions, the parameter mask does what the theory says for beliefs. An optimistic rater wrecks a fused forecaster within 20 steps (99.9% sycophancy, BSS -1.03), a token mask still leaks (drift 0.20), and the walled-off belief is untouched (sycophancy 1.2%, desire-bias change 0.000). The part the spec cared most about failed: no arm turned honest beliefs into a reliably better, utility-swappable policy at this RL budget. The rising reward is mostly a base-rate reflex. A follow-up needs a reward without a belief-free shortcut (prices drawn around the market price or with balanced outcomes), more RL steps, or an action policy given the decision rule's inputs in a learnable form.

## Files

- `r4-e3b/PREREG.md`, `r4-e3b/lir_rl.py`, `r4-e3b/build_data.py`, `r4-e3b/report.py`, `r4-e3b/posthoc.py`, `r4-e3b/launch.sh`
- `r4-e3b/data/{train,val,test}.jsonl`, `stats.json`
- `r4-e3b/runs/warm-s*/` (warm.json, sft.jsonl, eval.jsonl), `r4-e3b/runs/{fused,tokmask,lir}-s*/` (config, steps.jsonl, eval.jsonl, eval-pre.json, eval-post.json, summary.json), `runs/report.json`, `runs/posthoc.json`, `runs/*.log`
- Tinker spend rows: `/home/agent/workspaces/calibrated-ai/BUDGET-20261001.md` (Tinker table, 10-16 rows)
