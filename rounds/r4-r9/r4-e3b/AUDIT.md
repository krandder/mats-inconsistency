# R4-E3b independent audit

Auditor: board-opus investigator (Opus 5.5, claude-opus-5-5[1m]), read-only. Checks run 2026-10-16 19:23-19:45 UTC. No Tinker calls, no GitHub calls, no spend.

## Verdict

**Sound, with one data leak and two prereg deviations.** The pre-registration came before the runs and its hash matches. Code, prereg and data are unchanged since `cc94206`. `report.py` and `posthoc.py` reproduce `runs/report.json` and `runs/posthoc.json` exactly. Every number and pass/fail in RESULTS-R4-E3b.md matches. The $10.21 spend matches the meter. Caveats 1-3 are supported by the data.

The problems:
- `build_data.py` shows the model a close date from after t (medium). It did not measurably change the results.
- The faith-swap "belief change 0.000" is hardcoded, not recomputed as the prereg promised (medium).
- The arms do not see identical episodes, although the prereg says they do (medium).

None of these changes a rule outcome.

## Verified (check -> result)

- **Prereg timing.** I ran `git log` and `stat` (birth times). `cc94206` was committed 18:53:02Z. The first main-run file, `runs/warm-s101/sft.jsonl`, was born at 18:53:46Z. RL `config.json` files were born at 18:58:29Z, and `ALL-DONE` was written at 19:22:02Z.
- **Prereg hash.** `git show cc94206:r4-e3b/PREREG.md | sha256sum` gives 448aecab...869b5. This matches the claim and the working file.
- **No change after scoring.** `git diff cc94206 HEAD -- lir_rl.py report.py build_data.py PREREG.md data` is empty, and the tree is clean. Since `cc94206`, only `launch.sh` (18:53:14, before the runs), `posthoc.py` and `RESULTS` were added.
- **Re-scoring.** I ran both scripts in memory with file writes patched out, so nothing on disk changed. The sha256 of report.json and posthoc.json was the same before and after. The recomputed JSON is semantically identical to both files on disk.
- **All 7 rule outcomes and all quoted numbers match.** Examples: val BSS 0.345 (0.353/0.378/0.304); LIR gain/R_ref 0.240 [-0.299, 0.780]; Fused 0.295 [0.253, 0.338]; LIR drift 0.0017/0.0024/0.0044; syc LIR 1.2%, Fused 99.9%, TM 7.1%; D1c 0.0318 vs 0.0376 (84.7%); Fused ddes change +0.490 [0.487, 0.494]; flip rate LIR 0.39, TM 0.53, Fused 0.00. That makes 3 of 7 pass.
- **Temporal splits** (from `data/*.jsonl` and `stats.json`):
  - train max resolution is 2025-08-29, before the val min t of 2025-09-02;
  - val max resolution is 2025-12-31, before the test min t of 2026-01-01;
  - 0 duplicate market ids across splits.
  - Question and description come from `market_versions` valid at t for 2500/2500 items (0 fall back to the final text).
- **Test never trained on.** `cmd_warmup` and `cmd_rl` draw training data only from `load("train")` (lir_rl.py:297-321, 360, 404). Test is used only in evals.
- **Gradient routing.**
  - `rl_datum` puts the advantage and old logprob only on the final sampled token (lir_rl.py:173-179).
  - Fused adds the verdict datum (427-429). Token-Masked does not.
  - LIR's `belief_client` is the saved warm sampler (366) and is never reassigned (437-438). Its training client is a separate fresh LoRA (367).
  - So LIR's belief gets no gradient.
- **Reward pairing.** The reward is r_j = payoff(action_j) + 0.25·[verdict_j == note] (416-418). The advantage is r - group mean, and groups without spread are dropped (108-112). This matches PREREG.
- **Eval metrics.** BSS, Delta_desire, false-note sycophancy, true reward, R_ref and flip rate (lir_rl.py:212-274; report.py:178-215) match the PREREG definitions.
- **Caveat 2 (noise floor).** I compared each warm-start's test eval with the pre-RL eval of the same weights:
  - Fused, TM and LIR all show 80-92% bit-identical values, mean |diff| 0.0018-0.0046 and max 0.031-0.040.
  - LIR pre vs post shows the same spread: 82-92% identical, mean 0.0017-0.0044, max 0.031.
  - So E3-D1a's failure is the noise floor, as the write-up says.
- **Caveat 1 (reflex).** posthoc agrees:
  - the reflex earns 0.038;
  - greedy equals the reflex on 99.9-100% of episodes for Fused (all seeds), TM 102/103 and LIR 102, and on 83.5% for LIR 101;
  - the threshold rule with honest beliefs earns 0.134-0.137;
  - Fused's threshold reward is -0.001 to 0.000.
- **Spend.**
  - The 9 `runs/*/summary.json` sum to $9.392 and the 3 `warm.json` to $0.815, for a total of $10.2071. Every run reached `stop: done` at 60 steps; none hit the $3 cap.
  - Booked in `BUDGET-20261001.md:45`.
  - Smoke: the `smoke-runs` summaries sum to $0.255. The budget row (line 44) books $0.50, including a killed pass.

## Issues by severity

**Medium**

1. **The shown close date leaks information from after t** (build_data.py:52 uses the final `markets.close`; `market_versions.close` as of t is in the DB and unused). The shown close differs from the close as of t on:
   - 776/1700 train items (YES rate 0.568 vs 0.209);
   - 151/300 val items (0.662 vs 0.349);
   - 228/500 test items (0.588 vs 0.287).

   These are mostly markets whose close was moved to an early resolution. That breaks the "evidence as of t only" rule. The warm model shows no sign of using it on test:
   - mu minus market price is about the same in both groups: s101 -0.067 vs -0.065, s102 -0.017 vs -0.019, s103 -0.101 vs -0.071;
   - in the leak group the model's Brier (0.217-0.247) is worse than the market's (0.195).

   The leak is shared by all arms and does not affect the rule outcomes. It should be fixed before any follow-up. RL arms could learn to exploit it; I did not test that.
2. **The belief half of E3-D2 was assumed, not measured.**
   - lir_rl.py:273 hardcodes `"belief_change": 0.0`, and report.py:215 scores only the flip rate.
   - PREREG says this belief change is "verified by recomputing mu, not assumed", and defines it as mu "recomputed for the swapped episode".
   - The swap holds mu = mu(note YES) fixed (lir_rl.py:245), so 0.000 is true by definition for every arm and does not tell LIR apart from Fused.
   - Under the training protocol the note changes with the mandate (`desired(m)`). Fused's post-RL mu(note YES) - mu(note NO) is about 0.999 (ddes_post 0.4994 × 2). A swap that kept the protocol would therefore move Fused's belief by about 1.0.
   - RESULTS says "Belief change 0.000 by construction in every arm". That is correct only as implemented, and it overstates what the rule tested.
3. **The arms do not see identical episodes.** The episode RNG is seeded per arm, `random.Random(seed*7 + {fused:1, tokmask:2, lir:3})` (lir_rl.py:395). PREREG says "same episodes and reward; only gradient routing differs". This adds noise between arms to E3-D1c (n=3), but no bias. The eval q values are shared, as stated.

**Low**

4. **"Wrecks a fused forecaster within 20 steps (99.9% sycophancy, BSS -1.03)"** (RESULTS, So what). The 20-step evidence is training-sample note agreement: 0.99/1.00/0.96 at step 20 (`steps.jsonl`). The test metrics were measured only at step 60 (`eval_every` 0).
5. **Caveat 3's "reflex scores 0.63-0.98" covers only the non-Fused arms.**
   - Fused's reflex flip score is 0.00.
   - LIR's 0.39 mean is made of s102 at 0.626, which is exactly its reflex score; s101 at 0.543, which is below its own reflex score of 0.709; and s103 at 0.00 (always PASS).
   - The write-up's conclusion still holds.
6. **"Matching the lir-agent Round-1 pattern (E1)" is an unchecked cross-experiment link.** I did not verify it.
7. **PREREG promises "relative gain over pre-RL reward when positive".** All pre-RL rewards are negative (-0.0038 to -0.0002), so it does not apply, but RESULTS does not say so.
8. **The run outputs are not under version control.** `runs/` is gitignored. Integrity rests on the reproduction above, not on git. Mtimes of code and data are at or before the commit and the tree is clean, but I cannot prove the executed bytes equal the committed ones.
9. **The smoke test ran on `test[:40]`** (`--eval-n`). It was a pipeline check only, but it touched test.
10. **Smoke spend is stated two ways.** PREREG says $0.25 and RESULTS says about $0.50. The budget row explains the difference.
11. **Spec deviations, all declared in PREREG or RESULTS.**
    - 60 GRPO steps vs the spec's 400.
    - No news context.
    - MiniBench not used.
    - Trust module and T2 audit not built.
    - The gate used BSS > 0.05, while the spec's fix 1 asks for ≥ 0.30. It passes either way at 0.345.

## Not checked

- Tinker's actual billing. $10.21 is the script's metered list-price estimate, with prefill as an upper bound.
- Whether Qwen3-8B's pretraining already knows train or val outcomes (2023-25). This could inflate val BSS (0.345) relative to test (0.215).
- Whether the RL arms exploit the close-date leak.
- Server-side determinism beyond the logged evals.
- That the Tinker LoRA/IS-loss semantics match `rl_datum`'s assumptions. I read only the code, not the API.
