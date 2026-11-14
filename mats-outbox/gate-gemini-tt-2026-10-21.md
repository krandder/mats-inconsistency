# Design-fit gate: gemini trader-transformer results (605faec, e5039a6, c6a9d5e), 2026-10-21
Reviewer: adversarial design-fit subagent for opus-mats. Checked at about 04:30Z on origin/main c6a9d5e, against Agent Mail #3858.
Verdict: **EXPLORATORY baseline. Do not report as a finding.** The trader-transformer design is implemented faithfully, but "beats the bar" mostly comes from giving the trader arms a better input encoder than the bar. No env is below 0.01 nats, and no XOR-type env was run.

## 1. Gate (a)-(d)
- (a) Token: one token per trader, built from that trader's own agent_inputs. **Holds.** The winning deepsets and grid cells are [present, obs, own cov, own cov*mu] with no report (trader_transformer.py:31-33, 113-135; runner L238-253). Caveat: the linear arms and the bar also get the scalar report (`report_logit/4`, transformer.py:50,55). It is an extra feature next to the rich inputs, not a replacement for them.
- (b) Layer: a pre-norm block (attention + MLP, residual on both; trader_transformer.py:239-247). **Holds** for the kv arms. In `attn_last`, layers 0-2 are MLP-only, so the arm is a set encoder followed by one attention block. `recurrent` replaces attention with a GRU (the CEO design), which is labelled.
- (c) What crosses between traders: width-128 layer states through causal attention (64 per piece cell in the grid arms). Rich, not scalar. **Holds.**
- (d) Regret is reported against exact Bayes and the bar. **Partly holds:**
  - **No XOR-type env was run** (Z5, Z6 and the XOR zoo env are missing).
  - Z8 Bayes knows provenance, which cannot be identified at `none`. The none-floor is unknown: none_floor_logit returns None for Z8 (zoo.py:798-812).
  - **Z8 prefix Bayes is wrong.** sequential.py treats Z8 as E2: it averages the noisy reads, while Z8 Bayes divides their sum by n_k + tau^2. Measured on 500 Z8 dev episodes: max |prefix_final - bayes| = 3.02 log-odds, against 1e-15 on E2, E3-sybil and E3-biased. So the Z8 prefix-KL table and the Z8 distill arms are invalid. In the JSON, the Z8 final prefix KL (0.0421) does not match the Z8 KL (0.0303). opus-mats fixed this only on the nearbayes branch (7379c5d).
- Unaligned: yes. But in E2 the inputs identify each piece: x_k is shared exactly and mu_k is continuous. E2 at `none` therefore behaves like `full`: V_vis_full_deepsets 0.0435 vs none 0.0447. Z8 is the real hidden-provenance test.

## 2. Fairness. Verdict: partly.
- Same for every arm: the 20k training pool, the 2000 dev episodes, 2000 steps, batch 512, lr 1e-3, wd 1e-2, early stopping on 10% of train with patience 800, and the same validation split (runner L337-338 and L144-146; transformer.py:193-196).
- **The trader arms get a better encoder than the bar.** The winners use slot-invariant deepsets or piece tokens. The bar gets flat linear tokens over 12 unaligned slots. With the same linear tokens the trader-transformer does not beat the bar: A_attn_all 0.0841, reperm 0.0557, bar 0.0615 (E2).
- The bar here is the secondary pool regime at width 128 with 4 layers. The headline bar is the fresh regime at width 64 with 3 layers (aggbench/README.md:269-276).
- Seeds: the 3 seeds change only the model init and trader order. The train pool and the dev set are drawn once (runner L337-338), so the ±SD leaves out data variance. E3-sybil grid rankfree has 2 seeds (grid README:38) but is labelled 3 seeds. E3-sybil is missing from summary.json (`partial: true`).

## 3. E3-biased equals E2 (0.0365±0.0053, 0.0430±0.0062). Verdict: env artefact, not a copy error.
- With the same seeds, E3-biased and E2 episodes are byte-identical in obs, own model, truth and Bayes. Only the reports differ (checked on 2000 train and 2000 dev episodes). The biased agent's raw obs is honest (transformer.py:20 already warns about this).
- The deepsets arms never read reports, so they see identical inputs and give identical numbers. The baselines read the biased report and get worse (extremized_log_pool 0.0763, bar 0.0729), so the "win" on E3-biased is only that.
- No E3-biased or abc E3-sybil artifact is committed.

## 4. Leakage. Verdict: holds, with two must-fixes.
- predict never sees prefix Bayes (cdv.prefix_bayes is set to None). Early stopping uses train, never dev. No arm sees the outcome or keys beyond its visibility.
- **Distill arms are graded "● Gain (Oracle Distill)"** (abc README tables). They must be labelled oracle ceilings, not Gain.
- The headline is the best of 29 arms picked on the same dev set, with test untouched. Treat it as winner-selected.

## 5. Rank-free grid. Verdict: legal, but not a test of the 3-D idea.
- The mask lets a cell see every cell of traders r' <= r at every layer (runner L102-111). That respects the causal trader order and "each path passes a layer once" (standard per-layer attention, no feedback).
- Within a trader, attention is bidirectional, so the token axis is dropped. At the last trader it is a full-context transformer over piece tokens, made causal across traders: the bar with better tokens.
- Its readout also differs: 0.5 × the last cell + 0.5 × the mean over present cells (L113-117). The other grid arms read cell T-1, usually an empty padding cell, so mask and readout are confounded.
- Kelvin's 3-D axis needs ordered tokens within a rollout. Unaligned E2/Z8 pieces have no order.

## 6. Fresh episodes. Verdict: cannot verify, and not comparable.
- The code forces n_train = steps×batch with one pass and no reuse (runner L702-703, L173-177), so "no reuse" holds as coded. It also forces reperm on and early stopping off for all arms, which collapses the (A) reperm and no_es contrasts.
- 102,400 episodes implies 200 steps at batch 512: 10x fewer steps than the pool runs and 15x fewer episodes than the 0d94ef1 bar (3000×512 = 1.536M).
- The claim 0.0615 → 0.0409 is one seed (the pool seed-0 value was 0.0553), from an uncommitted, still-running run (tt-abc-fresh-0d9).

## 7. Labels. Verdict: violations.
- "● Gain" appears in both Status lines (abc README:6, grid README:6) and throughout both tables.
- Grid README:39: "Finding V5 Confirmed Across All Environments".
- Grid README:41: "completely solves". Mail #3858: "Finding V4 Confirmed".
- Both READMEs have "Decided:" lines.
- The abc README links result_E2.json and result_Z8.json, which do not exist; the files are at tt_abc_*/result.json.
- harness git_commit is "unknown".

## 8. Distance from near-exact Bayes
- **No env is below 0.01 nats.**
  - E2: best regret 0.0365 (KL 0.0322), rank-free grid 0.0373, oracle distill 0.0250.
  - Z8: best 0.0391, and nothing beats extremized_log_pool (0.0344). The mail's "beats ext_log_pool on Z8" is false.
  - E3-sybil: 0.038-0.045.
  - Z1 (0.024) and Z1-blind (0.024) appear only in the mail.
- **Most likely cause: data. Optimisation is second; capacity and the env floor are unlikely.**
  - On E2 the floor is 0 (content identifies the pieces).
  - Every architecture lands at 0.036-0.045: the 200k grid, deepsets at 830k, recurrent at 1.0M, attn_last at 630k. That points to one shared bottleneck, not capacity.
  - Early stopping finds the best weights at step 100-200 for the linear arms and 300-1500 for deepsets (summary.json logs), and without early stopping the model memorizes the pool (0.94).
  - Oracle labels on the same pool cut regret to 0.025.
  - Even one trader's posterior, a plain sum of 2·mu·x over its cells, which a deepsets encoder can represent exactly, leaves prefix KL 0.0062 at t=1.
- Z8 also has an unknown none-floor. But with provenance given (V_vis_full_deepsets, floor 0) it still sits at 0.039, so Z8 is limited by learning too.
- The test is opus-mats's nearbayes lane (4M pools, 60k steps; no results committed yet). Do not duplicate it.

## Must-fix for gemini
1. Relabel both READMEs and the mail as "EXPLORATORY baseline": no Gain, Confirmed, Decided or "completely solves". Grade the distill arms as oracle ceilings.
2. Rerun the bar with the same encoder as the winners: deepsets tokens, or a bidirectional transformer over piece tokens, both without the report. Compare at matched regime, against the fresh 3000×512 headline bar as well.
3. Drop E3-biased from the trader-arm claims, or give the biased agent biased raw inputs. Withdraw "beats ext_log_pool on Z8".
4. Do not use Z8 prefix-KL or Z8 distill results until sequential.py has the Z8 Bayes from 7379c5d. Add a test that prefix_final equals bayes for every env.
5. Commit the E3-sybil 3rd seed, the abc E3-sybil and E3-biased runs, the fresh run and its args. Record git_commit. Fix the artifact links. Resample the train pool per seed, or say that it is fixed.
6. Isolate the rank-free effect: rank-free mask with the cell T-1 readout, and grid_full with the pooled readout.
7. Run the XOR-type envs (Z5, Z6, the XOR zoo env) before any Axis (A)-(D) claim, as the overnight plan requires.
