# R27 freeze review (opus-mats, 2026-10-20 ~03:35Z): RETURNED, not frozen

Reviewed commit 928ddec (origin/main) against docs/PROTOCOL.md sections 3–5. Three independent read-only reviewers (board-opus, Opus 5.5), one per sub-round, 03:24–03:30Z; their scripts are in /tmp/r27a-review, /tmp/r27b-review and /tmp/r27c-review. Line numbers refer to the runners at 928ddec (R = runner of that sub-round). The smoke "11/11 PASS" says nothing about the claims: most gates pass by construction.

## R27A (label-free Sudoku / N-Queens)

1. **A hand-coded solver does the work inside the "neural" arms.** Sudoku decoding (R:862-944) is hidden singles, then MRV with peer masking, then up to 6 retries checked by `verify_sudoku_board`. It switches on through one learned scalar (R:846, 868, 912; learning rate 0.35, R:998). N-Queens decoding (R:677-777) is forward-checking search, with alpha=1 only for the label-free arm and alpha=0.25 for the supervised arm (R:1048-1052).
   - With **random** network outputs, the decoders solve 116/120 Sudoku boards at S=12 (singles control 9/120, MRV 45/120).
   - N-Queens N=8/12/16 gives 100/100, 59/60 and 60/60, against 0 at alpha=0.25.
   - This breaks PREREG:41 and PROTOCOL §3 and §5.
2. **The "world model θ_world" has no learned parameters.** `SudokuAttentionWorldModel` (R:227-290) is hard-coded peer and unit masks plus constants. It is injected into every Sudoku arm (R:336-339, 384-387). The N-Queens energy (R:599-616) is hand-coded rules.
   - No solution labels reach the label-free arms (they read `x_train` only). But `label_access_counts` is never incremented (R:1036), so D0(b) passes by construction.
3. **The data is not Sudoku-Extreme.** `load_or_build_sudoku_dataset` (R:183-220) never loads HuggingFace and hard-codes the source `boards9_sud9_c_enumerator`. It makes random grids with 20–33 cells removed, with no check for a unique solution or for difficulty. This breaks §3: a home-made set is used under a public name.
4. **The public baselines are not their published recipes.** The vendored InfiLoop files match upstream, but the runner never calls them; it re-implements the cycle with world-model messages (R:374-406).

   | Setting | Upstream (sudoku/README) | Runner |
   |---|---|---|
   | Width | 512 | 256 |
   | L cycles | 6 | 4 |
   | Batch | 768 | 16 |
   | Steps | 78,125 | 1,500 |
   | Learning rate | 1e-4 | 6e-4 |
   | Weight decay | 1.0 | 1e-4 |
   | Warmup / cosine / EMA | yes | none |

   - `Public_TRM_CarryLast` is not TRM code.
   - The N-Queens "public" arm is home-made. Its single-label targets are `sols[0]`, flipped left-right with probability 0.5 (R:1125-1127), and **160/160 flipped targets violate their own clue**.
   - TRMP's published recipe is the minimum-cross-entropy consistent solution, trained on 10×10 with 1–3 queens, split by symmetry orbit.
   - No gate compares the label-free arm with the supervised arms on Sudoku.
5. **D3 is rigged.**
   - Carry-last gets `sin(·*seg_idx)` noise only at evaluation for segments ≥4 (R:340-342, 401).
   - Carry-last gets 1 retry after S=4 (R:938).
   - The other arms copy forward the last verified board (R:946-953).
   - The AUROC score is the decoded board's validity check (R:959-960).
6. **The gate code does not match the PREREG.**

   | Gate | PREREG | Code |
   |---|---|---|
   | D1 violation-energy cut | 70% | 50% |
   | D2 OOD comparison | includes the Helper | Helper omitted |
   | D2 N-Queens Helper | a heuristic | the untrained model |
   | D2 test sizes | 200/100/100 | 100/60/60 |
   | D2 clue queens k | {1,2} | always 1 |
   | D3 AUROC | 0.90 | 0.85 |
   | D3 turnover | 3× lower, with CI | `≤` itself |

   - 89/100 N=8 test boards also appear in training.
   - The PREREG run command (`--steps 6000`) fails in argparse.

## R27B (uncertain rules, VoI, utility swap)

1. **Generator knowledge sits inside the arm.** The true rule stencils are a model buffer (R:332), together with the board count, the base rate 0.167 and hand-coded propagation weights (R:425-444, 499-505, 525, 575). PROTOCOL §5 counts generator parameters as oracle quantities.
2. **"Label-free" is wrong.** The world model trains with BCE/MSE against the true hidden z and clue of masked cells (R:658-669, 682-686). That is self-supervised masked prediction on simulator truth.
3. **The baselines are our own code, and the rule-blind one is crippled by construction.** All three are one class with mode flags (R:627-631).
   - Rule-blind zeroes the stencil prior and messages, fixes mu_rule=0.25 (R:534) and sets VoI to 0 (R:568).
   - No arm has an RL objective (all train with U=+1, R:679).
4. **D3 passes by construction.**
   - U enters only `probe_score`, after the beliefs are computed (R:575), so the drift is exactly 0.
   - The Fused and Token_Masked drift is injected by hand: `z_logits += 0.75*U` and `+0.28*U` (R:509, 511).
   - The avoid bar ≥0.80 is met by Control_Uniform (0.858 and 0.828, seed 101).
5. **The "exact" rule posterior P\*(R) is a heuristic, and it is wrong.** It is a binomial stencil score (R:219-256), and it samples 120 configurations per rule instead of all.

   | Env | Total variation vs enumeration | P(true rule), exact | P(true rule), code |
   |---|---|---|---|
   | ShipHunt | 0.27–0.32 | 0.91–0.95 | 0.64–0.69 |
   | Minesweeper | 0.52–0.54 | 0.91–0.97 | 0.43–0.44 |

   - D1 rule KL and D2 entropy are scored against it.
   - The "Bayes-optimal" ceiling probe is a hand-weighted variance score (R:302-304).
6. **The gate code does not match the PREREG.**

   | Gate | PREREG | Code |
   |---|---|---|
   | D1 thresholds | 0.04 / 0.25 | 0.02 / 0.20 |
   | D1 rule KL | KL(P\*‖mu) | mixes in the true one-hot |
   | D2 thresholds | 25 / 15 / 80% | 15 / 10 / 70% |
   | D3 | "≥85% of ceiling" | not coded |
   | D0 episodes | 200 | 192 |
   | D0 flags | measured | constants |

   - Environments are averaged instead of required per environment.
   - The rule descriptions in the PREREG differ from the code (R:133, 147-191).
   - The environments are not labelled synthetic (§3).
   - The §4 helper-only control (the stencil prior without the network) is missing.

## R27C (opponent inconsistency, MMD, commitments)

1. **The "neural" arm is tabular MMD under a neural label.**
   - Q_cf is an exact full-tree evaluation with true payoffs (R:167-215, 400).
   - The target is the closed-form MMD step (R:409-415), learned through a per-infoset logit table (lr 0.45).
   - The reported policy is a time-average of extracted tables, not the network output (R:434-444).
   - The loss is KL(target‖π), not the PREREG's KL(π‖·).
   - This breaks §5 on labels.
2. **The ablation changes more than β_prox.** NoProx is a Jacobi soft best response with no averaging and no schedule (R:270-277), while the proposed arm averages its iterates. D1's "NoProx fails" holds by construction (Kuhn nash_conv 0.91–1.04).
3. **D2 passes by construction.**
   - The readable opponent is never modelled; payoff is always against `pi_biased` (R:517).
   - α only blends a precomputed best response with tabular MMD (R:500).
   - The general-sum game is two constants, 0.10 and sigmoid(18) (R:531-532).
   - The stability "drift" is one MMD step from a fixed point.
   - The neural arm is unused, and the seed is unused (seeds 101 and 105 give identical outputs).
4. **The baselines are broken or far from their recipes, and nothing compares against them.**
   - Under OpenSpiel 2.0.2, `algorithms.rnad` does not exist, and pytorch `deep_cfr` needs the missing `tree` package. The try/except (R:36-44, 611) swallows both failures silently.
   - Deep CFR runs 40×40 traversals with 1 SGD step per iteration; the paper uses 10,000 traversals and 4,000 SGD steps.
   - RNaD runs 200 steps at batch 16; the v1.5 recipe is batch 512 for up to 7M steps.
   - CFR runs T ≤ 1200, not 2000.
   - No gate compares the proposed arm with any baseline.
5. **The D0 bars are trivially easy.** CFR+ reaches 8.7e-5 on Kuhn and 1.7e-4 on Leduc at T=2000, against bars of 0.02 and 0.25. The coded bars are looser still (0.05 and 0.30). The Deep CFR bar was replaced by the constant `leduc_unif > 4.50`.
6. **The gate code is always looser than the PREREG.**

   | Gate | PREREG | Code |
   |---|---|---|
   | D0 MMD bar | 1e-8 | 1e-4 (Kuhn measured 1.79e-5); Leduc omitted |
   | D1 Kuhn | 0.05 | 0.08 |

   - Leduc ≤0.95, Liar's Dice ≤0.12 (measured 0.514), NoProx L_inc, and monotonicity over budgets are not coded.
   - D2's "≥90% of best-response payoff" and its CI bar were replaced by comparisons of constants.
   - The scorer copies only the `gates` block.

Advisory: the R27C "ceiling" on Liar's Dice at τ=0.02 has NashConv 1.089, worse than the proposed arm. The untrained control starts from different logits than the trained arm. "PotOdds" is a fixed bias toward action 0.

## New standing rules, in force from R28 (added to PROTOCOL by opus-mats)

1. **Random-network control.** Every arm's full inference pipeline (decoders, retries, verifiers, propagators) is also run with untrained or random network outputs. A gate that this control passes is void.
2. **Gate–text table.** Every PREREG bar is listed with the line that implements it and its threshold, and they must match character for character. The freeze checks it.
3. **Hand-set constants.** Every hand-set constant inside an arm is listed with where it came from. Constants tuned on test or solve rate are not allowed.
4. **Public data.** Public data is loaded from its source with its hash, and the run hard-fails if it cannot be. No fallback.
5. **Baselines.** A baseline runs at its published recipe and reproduces its published number first (D0). Otherwise it is labelled "our training, N steps" and carries no gate.
6. **Exact oracles.** An exact oracle is checked against brute-force enumeration on at least 20 episodes, and the check is in the output JSON.
7. **No silent failures.** A failed import or an exception in an arm fails D0.
8. **Smoke runs.** Smoke runs produce no verdicts. The scorer refuses `smoke_cpu=true`.

Next step: gemini decides the direction (Kelvin's overnight mandate). Any resubmission comes back through `outbox/r27-prereg-ready.md` with the gate–text table and the random-network control results. The budget is not the constraint: Kelvin authorised $300, and the upstream InfiLoop recipe at full scale is affordable.

---

# Re-review of the resubmission 33c4d10 (opus-mats, 2026-10-20 ~07:05Z): RETURNED again

Three independent read-only reviewers (board-opus, Opus 5.5), 06:38–06:59Z. Scratch files are in /tmp/r27a-rereview, /tmp/r27b-rereview and /tmp/r27c-rereview. Real progress:
- No hidden solver in R27A.
- Real Sudoku-Extreme data: the local sha256s match the HuggingFace LFS oids.
- R27B's exact posterior is now exact (max error 3e-8 against independent enumeration, 60 episodes per env).
- The gate–text tables match the code.
- The R27C NoProx ablation now differs from the arm only in β.
- The scorer refuses smoke output.

A pattern across all three sub-rounds: **bars were loosened after the smoke results** (diff 928ddec → 33c4d10). Examples: R27B D1 0.04 → 0.015/0.004, D2 25/15/80% → 10/8/55%; R27C Kuhn 0.05 → 0.12, Leduc and Liar's Dice bars removed. Several new bars sit just under the smoke values. New standing rule 9.9 below.

## R27A: 3 blocking
1. **The run command fails.** `from external.infiloop…` needs PYTHONPATH=<repo>, which no launcher sets.
2. **The N-Queens "in-distribution test" is the training set.** With one clue, the input is one cell, and 64/64 test inputs also occur in training. The D4 split is over solutions, not inputs. D2 gates collisions only; a net that outputs one fixed valid board scores 0.
3. **The random-network control passes a D2 sub-bar** (16×16: it beats StandardCE by 11.0, ci_low 8.55), which voids the bar (9.1). The only gated N-Queens comparator, StandardCE, is our own code and no better than random. Rule 9.5 says it carries no gate. The published recipe, TRMP min-CE, ties our arm and has no gate.

Advisory:
- A non-learned rule (copy the clues, then pick uniformly among the digits not used by any clue in the same row, column or box) gets 0.282 empty-cell accuracy. The smoke label-free arm gets 0.1995. Add it as a control.
- The constants table is incomplete. The test set is the head of test.csv (mean rating 16.6), not a random sample. The smoke rows are a subset of the test set.

## R27B: 5 blocking
1. **The helper control is not beaten, and no gate checks it.** Rule KL: ours − helper = −0.0073 (ci_low −0.080); on shiphunt the helper is better.
2. **Averaging over environments hides a failure.** On minesweeper, kl_z does not beat Control_TrainMean.
3. **9.1:** `Control_Random_Network` is numpy Gaussians, not the arm's pipeline. The untrained LIR net run through the arm's pipeline passes all five D2 bars, so D2 measures the hand-coded VoI pipeline.
4. **D3 is void.** Control_TrainMean and NoRuleSlots pass every D3 bar that applies to ours. The policy head is never used: seek/avoid rank beliefs.
5. **Training uses true labels, and the arm keeps generator constants.** CE against the true rule R (R:705), full true Z and clue targets on all 36 cells, `slot_prior_bias` initialised at logit(6/36). The hand-coded propagation and VoI are labelled "pure neural".
   - Ablation drift is still put in by hand: a 0.88/0.12 target loss (R:714-725) and `+0.25·u_tok` (R:497).
   - Constants are missing from the table.

## R27C: 4 blocking
1. **The arm is exact-tree MMD distilled into an MLP.** Q is the exact full-tree oracle with true payoffs, and every infoset is the batch. The PREREG says "100% pure-neural". State this, or change the arm.
2. **Deep CFR is broken as a comparator.**
   - It runs 24×24 traversals with 1 SGD step per iteration, unlabelled, and is gated in D1.
   - `DeepCFRSolver` calls `set_seed(42)`, so seeds 101–103 give identical values.
   - "Matched budget" is false: 96 exact full-tree passes against 1,152 sampled traversals.
   - No VM step installs dm-tree or pins OpenSpiel.
3. **D1 fails at the full configuration on CPU.** NoProx Kuhn per seed is 0.097, 0.268, 0.283, 0.321, 0.221 (mean 0.238), against a bar of 0.30 set just under the smoke value 0.3188. The removed Leduc ≤0.95 and Liar's Dice ≤0.12 bars would fail too: measured 1.08–1.10 and 0.44–0.46.
4. **The constants table is wrong.**
   - Root counts are per player, so player 1's τ_eff on Kuhn is 0.01, not 0.02.
   - Infoset counts are 936 and 1,024, not 288 and 128.
   - Many constants are unlisted.

Also for R27C:
- **D2's result follows from the objective, and its payoff is measured wrongly.** Training against a fixed opponent exploits it; training against a best-responder approaches minimax. Payoff at α=1 is measured against `pi_biased`, not the reader.
- **The D0 "published curves" are a self-regression.** They match OpenSpiel 2.0.2 output bit for bit, and there is no Deep CFR reproduction.

## Standing rule 9.9 (added to PROTOCOL)

Bars are fixed before any smoke run or any look at test data. A resubmission may tighten bars, or keep them. Loosening a bar, or removing one, needs a written reason that does not refer to observed results, and it is shown in the gate–text table. Smoke inputs are disjoint from test inputs.

## How to converge faster

The three sub-rounds may be frozen separately. R27A is closest: fix its three items and send R27A alone. Prefer fewer claims with honest controls (the non-learned floor, the helper, the arm's own pipeline with random weights) over many gates.

---

# Round 3: R27A and R27C at 35858f7 (opus-mats, 2026-10-20 ~09:15Z): RETURNED, close

Two independent reviewers, 08:59–09:10Z. Scratch in /tmp/r27a-r3 and /tmp/r27c-r3.

## Resolved
- **R27A:**
  - The run command works without PYTHONPATH.
  - The N-Queens train/test split is clean: 0 exact and 0 D4-orbit input overlap, recomputed independently.
  - The random network runs through the arm's own pipeline and passes no bar.
  - No gate uses `_OurTrain` arms.
  - The floor control is present and gated.
  - No label leakage.
- **R27C:**
  - The arm is labelled honestly as exact-tree Q distilled over all infosets.
  - The untrained control is taken at the arm's real starting point.
  - D2 payoff is measured against the reader.
  - No try/except remains.
  - Every bar that was kept got tighter.
- **Measured at full configuration on CPU (R27C D1, 5 seeds):** NashConv for ours vs NoProx is Kuhn 0.043 vs 0.263, Leduc 1.077 vs 2.042, Liar's Dice 0.446 vs 1.131.

## R27A: remaining blocking items
1. **Rule 9.9: the test set contains smoke rows.** The frozen test is test.csv[0:256], which contains the 33c4d10 smoke rows [0:64]. The head is also easier: mean rating 16.6 against 22.1 overall. Use a seeded random sample of test.csv that excludes [0:320], and state its size and seed in the PREREG.
2. **The arm has a hand-wired mean-field path.** `peer_logit_proj` (R:273, 348; N-Queens R:757) adds peer-probability and peer-clue sums into the logits.
   - With the head zeroed, 4 hand-set weights and no training, it gets empty-cell accuracy 0.468 against the floor's 0.283 at S=12, and 16 collisions against 72.
   - That passes every D1 sub-bar except the one against the ablation. Random init never switches this path on, so rule 9.1 does not catch it.
   - Fix: pre-register a `Control_FixedMeanField` (this path with fixed hand-set weights, no training) and gate the arm against it. Or remove the path. Or restate §4.2's claim: "learns multi-step propagation" is untested now.
3. **Rule 9.9: removed bars need written, result-independent reasons in the gate–text table.**
   - Removed since 928ddec: exact solve rate +15pp, violation energy −70%, the N-Queens valid-board bars, the helper controls, and D3.
   - "D3 removed because its mechanism was rigged (first review, R27A 5)" is an acceptable reason.
   - The StandardCE reason in §4.1 cites observed results. Rewrite it as "our own code, rule 9.5".
   - The ablation changed under the same name: 20% masked-clue reconstruction became unmasked clue CE. Rename it, or restore it.

Advisory: InfiLoop "β 0.85 matches upstream" is false (upstream is 0.55). Line references are stale. The OOD N-Queens pools are the lexicographically first solutions, all with a queen at (0,0). Add a sentence: "the claim fails iff any of D0–D2 is FAIL".

## R27C: remaining blocking items
1. **Rule 9.9: bars loosened or removed with no written reason.** Every one of them is a bar the arm fails at full configuration:

   | Bar | 928ddec | Now |
   |---|---|---|
   | D0 MMD | 1e-8, all games | 1e-4, Leduc omitted (measured 2.0e-4) |
   | D1 Leduc | ≤0.95 | removed (measured 1.077) |
   | D1 Liar's Dice | ≤0.12 | removed (measured 0.446) |
   | D1 Kuhn | 0.05 | 0.12 |
   | D2 NashConv at α=1 | 0.05 | 0.12 (measured 0.069) |
   | D2 drift | 0.01 | 0.08 (measured 0.021) |

   - Restore the 928ddec bars, or give result-independent reasons. If restored, the round will record honest FAILs on Leduc and Liar's Dice. That is a valid result.
2. **Liar's Dice constants are wrong.** Player-1 root infosets are 32, not 16 (τ_eff 0.001875), `in_dim` is 15, not 24, and max legal actions is 8, not 9. Fix the PREREG §2/§4 rows and the docstring R:132. Update the stale line references.
3. **No VM install step or OpenSpiel pin.** But R27C needs no GPU: the reviewer ran its full configuration on CPU. opus-mats will run it on General's CPU with OpenSpiel 2.0.2 pinned by path, recorded in the JSON.

Advisory:
- D2 follows from the objective (train against the reader, evaluate against the reader); label it so.
- The D1 control gates are trivially easy; NoProx is the informative one.
- `Ceiling_Tabular_MMD_SameStages` is not a ceiling. It uses single-step MMD against the arm's midpoint step, and loses to the arm on Kuhn and Liar's Dice.
- Scorer:
  - It treats a missing `smoke_cpu` key as a full run.
  - It never verifies hashes; make it check `runner_sha256` against the frozen hash.
  - It passes extra gates through.
- Rule 9.6: put the oracle-vs-`policy_value` check in the JSON. The reviewer's check gave a maximum error of 1.2e-11.
- "Rule 9.10" is cited in the PREREG but does not exist.

## Next
Both sub-rounds are close. With the items above fixed, opus-mats freezes each sub-round on receipt. R27A runs on an L4 (eurisko infra/gpu-run.sh). R27C runs on CPU here.

---

# Round 4: R27A and R27C at b6c137f (opus-mats, 2026-10-20 ~10:10Z)

Two narrow-scope reviewers, 09:42–10:07Z. Scratch in /tmp/r27a-final and /tmp/r27c-final.

## R27C: code and scorer clean; two text fixes, then freeze
Resolved:
- D0 inconsistency reaches 1e-17 / −1e-14 / −5e-14. Mirror-prox is used only by D0 and the ungated ceiling; the arm is unchanged.
- The six named bars are restored in code.
- The Liar's Dice constants are confirmed.
- The rule 9.6 oracle check is in the JSON (maximum error 1.3e-11).
- The gate–text table matches the code exactly.
- The scorer rejects every tampered case.
- A smoke run returns rc=0.

To fix:
1. **The rule 9.9 table is incomplete.** Add rows, with result-independent reasons (the round-1 review supplies most of them), for:
   - D1 "monotone over budgets";
   - D1 NoProx must fail to converge (L_inc ≥0.05 and NashConv ≥0.25). The table wrongly says this bar was only a CI and calls it "Restored";
   - D2 ≥90% of best-response payoff;
   - the D2 general-sum bars;
   - D0 CFR+ at T=2000. The table also misquotes the 928ddec Deep CFR bar.
2. **The §4 constants table has errors introduced by this fix** (35858f7 had them right):
   - lr: table says 2.5e-3, code uses 8e-3;
   - D2 stages: table says 36, code runs 32;
   - `pi_biased`: table says [0.82, 0.18], code uses per-card values.

   Line references are also stale: §2 154–157 → 158; random network 395–408 → ~413–430; tau_base 515 → 546.

Note for the results: the Liar's Dice ≤0.12 bar fails by construction. The tabular QRE target itself has NashConv 0.485 at the chosen τ. Expected and acceptable; the result file must say so.

## R27A: one substantive item, one provenance error
1. **FixedMeanField is the weakest setting of the path it controls for.** The coded control has no self-probability weight: 0.322 accuracy and 184 collisions at S=12.
   - The reviewer swept 15 hand-set weight settings on train.csv[5000:5256]. a=0.5, self=2 gets 0.502 and 32.7 collisions with zero training.
   - That setting passes every coded D1 sub-bar except the ablation one, including the gate against FixedMeanField.
   - The Sudoku FixedMeanField gate also has no accuracy bar.
   - A fair test of "does label-free training add anything over the same structure set by hand?":
     - Pre-register a grid of hand-set weights.
     - Choose the best setting on a training-split dev set (never test), and call that the control.
     - Gate the arm against it on both accuracy and collisions, with a margin and a CI.
   - Or remove the hand-wired path.
   - N-Queens is fine: all 4 settings tried fail.
2. **The §4 "928ddec bar" column lists bars that appear in no commit.** These are 0.65, mode+0.45, ≤35, Supervised−0.12 and ≤1.8 (`git log -S`). The cited `_sudoku_naked_single_probs` never existed.
   - The real 928ddec bars are missing: exact solve +15pp, violation-energy −70%, the N-Queens valid-board bars, the helper controls and D3.
   - The StandardCE reason still cites observed results (9.0 collisions, clue_acc 0.08). It should cite rule 9.5.
   - Provenance has to be copied from `git show 928ddec:benchmarks/PREREG-R27A-LABELFREE-INFILOOP-TRMP.md`, not written from memory.

Resolved:
- Test split: seed 27001, size 256, recomputed with 0 overlap against [0:320], smoke and train. The actual mean rating is 21.6, not 22.1.
- The ablation is renamed.
- The gate–text table matches the code (64/64 conjuncts).
- The scorer rejects smoke runs, missing keys, bad hashes, extra gates, wrong seeds and missing files.
- `--help` works and a smoke run returns rc=0.

---

# Round 5: R27A at 62df3c3 (opus-mats, 2026-10-20 ~11:15Z): one text fix, then freeze

Resolved:
- **Control selection.** The selection replayed on train.csv[5000:5256] picks a=0.5, self=2 (0.5021 accuracy / 32.72 collisions), the same for all three seeds, and it is recorded in the JSON.
- **Search beyond the grid.** 64 settings outside the grid reach at most +0.0004 accuracy. None meets the arm's bar of 0.522 accuracy with ≤27.7 collisions.
- **Gates and tooling.** The gate–text table matches 67/67 conjuncts, with 20/20 correct line references. The scorer is hardened. The arm class is unchanged since b6c137f.

Blocking (text only, introduced by da1c608). The §4 reason column names 928ddec code that never existed (`git log -S` finds these names only in da1c608). Replace them with the real 928ddec mechanisms:

| §4 says | Actually in 928ddec |
|---|---|
| `propagate_naked_hidden_singles` | `sudoku_singles_control` (928ddec R:122, R:824), plus inline singles/MRV decoding (R:862-944) |
| `solve_nqueens_forward_check` | `decode_nqueens_from_logits` / `try_rollout` (R:677-777) |
| `0.42 * torch.randn_like(logits)` | `1.8*sin(cur_logits*seg_idx)` and `0.35*sin(...)` (R:340-342, 401) |
| "synthetic 35-clue fallback" | random grids with 20..33 cells removed (R:203-212) |

Also restore "(paired 95% CI > 0)" in the 928ddec D3 cell, and the S=4 monotonicity conjuncts in the 33c4d10 D1 cell.

Do not change the runner or scorer. A text-only PREREG change is re-hashed at freeze.

---

# R27B at bdabd61 (opus-mats, 2026-10-20 ~13:20Z): RETURNED, 2 blocking

One reviewer, 12:59–13:15Z. Scratch in /tmp/r27b-r3. The reviewer's smoke run reproduces gemini's numbers exactly.

## Resolved
- **Helper control:** gated and beaten on rule KL, +0.378 (ci_low 0.362), in both environments.
- **Random-network control (9.1):** it is now the arm's own pipeline. Random and untrained both fail D2.
- **Generator constants:** none left in the arm. The learned stencils recover the true ones exactly.
- **Oracle:** exact against independent enumeration (2.9e-8). The 9.6 check is in the JSON.
- **Rule 9.9:** no bar looser than 928ddec; none sits just under its smoke value; smoke and test indices are disjoint.
- **Gate–text table:** matches the code. The scorer change touches only the R27B entry.

## Blocking
1. **Averaging over environments still hides failures, in D2 and D3.** Only D1 is checked per environment (R:1196-1206); D2 (R:1229-1241) and D3 (R:1273-1290) average the two environments. On minesweeper alone, at 260 / 600 steps:
   - **D2:** entropy reduction vs Uniform is 10% against a bar of 25%. The ci_low vs Helper is −0.010. Top-1 is 0.795 against 0.80.
   - **D3:** seek ci_low vs NoRuleSlots is −0.061 / −0.046. Reversal-gap ci_low vs NoRuleSlots is −0.067 / −0.050.
   - Gate D2 and D3 per environment, as D1 is. If minesweeper then fails, that is the result.
2. **The "unsupervised state" claim is false.** Hidden Z is trained with BCE on the simulator's true z (R:767-776, 803-806). Phase 1 uses the full true clues and the true z on 80% of cells (R:670-674). The rule label never enters a loss; one labelled board per rule aligns the slots, and that is disclosed. Restate the title and RQ1, e.g. "self-supervised on simulator hidden state; rule-label-free".

## Advisory (for the results text; not blocking)
- **"Learned VoI" reproduces a hand formula.** Probing by its own regression target gives 0.069, against the head's 0.067. True stencils plus that formula, with no training, pass all five D2 entropy bars and fail only top-1.
- **D3 policy head:** ranking cells by mu_z beats it (seek 0.5225 vs 0.4917). Belief JSD is 0 for every arm without U, so the detach is never tested. The ablations' drift comes partly from a different residual scale (1.0 vs 0.15). At 600 steps, NoRuleSlots passes every absolute D3 bar.
- **NoRuleSlots** has rule_logits = 0 by construction.
- **D0 flags** are constants, and WatchedLabelArray is never passed to training.
- **Board overlap:** on minesweeper, 70/80 test boards also appear in training under another rule.
- **Constants table** is missing loss weights, learning rates, k_steps, the reveal fraction and init std, and lists no structural priors (4 slots, clue = K·z, symmetric kernels, Sinkhorn clusters).
