# P13 PREREG: market ResNet on mazes (frozen before any GPU spend)

Title: Market ResNet on mazes with the logit runaway fixed: does the trader economy beat Bansal DT-recall at 11x maze size at equal compute, and is the gain from the economics?

Structure: the v2 spec (verbatim, from the workflow's /tmp/p13_spec.json, revised 19:56Z 29 Sep), then A (implementation
choices frozen with it), B (deviations), C (readings with times), D (citation checks), E (unit tests at freeze).

## Spec: question

Port the phase-2 market ResNet economy (covariance markets on dyadic pair-equality targets, plus VOC markets) from 1D prefix parity to Bansal/Schwarzschild mazes. Train on 9x9 only and add a pilot-selected fix for the belief-logit runaway: 1D phase 2 scored 93% exact at L3200 but had a log loss of 501 nats/bit.

Two questions:
1. At equal training compute and equal test FLOPs, does the port beat Bansal et al. 2022 DT-recall on exact accuracy at 99x99 (11x; 201x201 as a stretch)? DT-recall is represented twice: a retrain at equal compute and the released full-recipe checkpoint.
2. Is any gain due to the economics, meaning wealth-scaled trade caps and respawns that are verified to bind, rather than to detached per-step training on the same hints? This is measured against the same code with the wealth cap switched off (ECON-nocap), and against full backprop with the same hints (BPTT-dscov, BPTT-progcov).

## Spec: prior_work

Revised at 19:56Z on 29 Sep after two critiques. arXiv IDs marked (m) are from memory and must be verified in Phase 0 before PREREG.md is frozen.

(1) Task and bar
- Schwarzschild et al. 2021 (arXiv 2106.04537) and the easy-to-hard-data maze generator (arXiv 2108.06011).
- Bansal et al. 2022, DT-recall with progressive loss (arXiv 2202.05826): alpha 0.01, width 128, max_iters 30, 9x9 training. The recipe is 50 epochs x 800 steps = 40k steps at batch 50 on P0 train.py's 40k/10k split. v1 wrongly said 50k steps.
- M-ref is the released checkpoint. On our pipeline:
  - 13x13: 1.00. 59x59: 0.996 at 12n on 500 mazes.
  - 99 dev (50 mazes): 0.12, 0.46, 0.64, 0.74 and 0.94 exact at t = 300, 500, 700, 1000 and 1188 (12n), still rising.
- Retraining is fragile:
  - P0: a w=64 retrain sat at chance (loss 0.69315) for 15k steps. It reached 0.996 on 9x9 at 30k but only 0.086 on 33x33.
  - P0 timing: 118.5 s per 1k steps. The same 15k-step job took 1119 s on one VM and 5056 s on another.

(2) Maze failure modes
- Knutson et al. 2024 (arXiv 2410.03020): loops, percolation and dead-end shifts, and limit cycles. This matches P0, where M-ref ends in 2-cycles on 411 of 500 33x33 mazes.
- Anil et al. 2022 (arXiv 2211.09961): path-independent equilibrium models.
- Bear et al. 2024, DT-L (arXiv 2410.23451, vendored in P0/P6/P11): the published stability fix for runaway iterates in deep-thinking nets.

(3) Detached per-iteration training without economics, the nearest mechanism
- HRM (arXiv 2506.21734) and TRM (arXiv 2510.04871): deep supervision on detached state, a one-step gradient, and learned halting on 30x30 mazes, all in distribution.
- JFB (arXiv 2103.12803).
- RAFT (Teed and Deng 2020, arXiv 2003.12039 (m)): detached per-iteration refinement with a sequence loss.
- CLRS hints (Velickovic et al. 2022, arXiv 2205.15659 (m)): supervised algorithm traces.

(4) Economic credit assignment and market makers
- Chang et al. 2020 (arXiv 2007.02382); Baum's Hayek machine (1999); Kwee, Hutter and Schmidhuber 2001.
- Markets as learners: Abernethy and Frongillo 2011; Storkey 2011.
- Hanson 2003 LMSR. A logit clip is a bounded price range, not LMSR bounded loss, so F-LB is only loosely related to it.
- Othman et al. 2013 LS-LMSR grows liquidity with outstanding shares, not with depth. F-TR2's depth-growing liquidity is only loosely inspired by it.
- Chen et al. 2014 decision markets: the eps settlement used here.
- Guo et al. 2017 temperature scaling (arXiv 1706.04599 (m)): the post-hoc calibration control.
- Li, Yumer and Ramanan 2020, budgeted training (arXiv 1905.04753 (m)): compressing an LR schedule to a step budget.

(5) Internal work read for this revision (read at 19:50-19:56Z)
- market-resnet RESULTS section 13: in every house-funded cov run the four traders end rich and equal (2,770-2,950 from W0 = 20). The cap never binds, there are 0 respawns, and nocap scores 0.92 against 0.93 at L3200.
- Section 8: at 10x, backprop given the same cov targets (bptt_cov_dscov) scores 0.98 against 1.00; at 100x it scores 0.00-0.28 by seed.
- market-resnet phase 3, renamed from v1's "P3 1D ablation", which clashed with puzzles/P3. It runs on trader-mresnet-10..15.
  - It includes the belief-regularizer arms liq t, liq sqrt, floor 0.01, l2b 1e-3, bcap 10 and dbcap 1.
  - At 19:50Z those runs were still training. Their logs show wealth of 1,500-2,300 from W0 = 20 for all four traders, so the cap again does not bind. Final results must be read at PREREG freeze if available.
- P11 (JOB_DONE 19:42Z, results/summary.json): DT-recall 2D, w=32, 6k steps, trained on 9x9 and tested at 17/29/59. Verdict FALSIFIED (F2): belief budgets lose to known fixes.
  - At the three endpoints, M-budget scores 0.71/0.21/0.01 against 0.79 (DT-L) and 0.58/0.44 (R-prog).
  - The LMSR budget beats the TV budget.
  - The per-step cap (eps 0.3/1.0) and clip (c 0.3/1.0) saturate on 69-93% of pixel-steps.
  - R-prog (Bansal's progressive loss) failed to train in 2 of 5 seeds (loss stuck at 0.693). Its trained seeds score 0.99/0.97/0.74.
- P6 (1D, JOB_DONE, analysis pending): EMA-clone and PBT controls that remove the ledger. Read at PREREG freeze.
- P0 (maze bar), P1 (halting headroom 1-6%, so VOC halting is secondary) and P7.
- THEORY.md K8 and the programme falsifier. The falsifier compares against deep supervision at 1000x depth; P13 reaches about 40x depth at 99 and 80x at 201. P13 is evidence toward the falsifier, not the test itself.

(6) Mechanism found while reading econ2.py (it goes into PREREG with a prediction)
- In 1D, pair_truth at offset 2^j equals 1 - XOR(x over i+1..i+2^j). That is the level-j value of a Hillis-Steele (1986) parallel-prefix scan.
- Trader.forward reads c_{j-1} at offsets 0 and +2^{j-1}. Those are exactly the two terms of eq_j(i) = XNOR(eq_{j-1}(i), eq_{j-1}(i+2^{j-1})). So phase 2 was hint-supervised algorithm-trace learning on a scan-shaped architecture with no fixed scale.
- For binary path labels the XNOR composition still holds in 2D. But the level-0 base case, whether adjacent pixels agree on path membership, is not a local function of the input on mazes; in 1D it is (x_{i+1} = 0). On mazes the hints give the composition step but not the hard part.
- A second runaway mechanism: the level sum grows with J = ceil(log2 side), which is 5 at 24 px and 8 at 204 px, and it is fed through saturating tanh(b/4) inputs.

## Spec: spec

VERSION AND LOCATION
- v2, revised at 19:56Z on 29 Sep. The change map is at the end.
- All code goes in /home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/.
- Never write in market-resnet/ (phase 3 is running there) or in any other puzzle's directory; copy files instead.
  - Copy econ2.py to P13/econ2_1d_snapshot.py and record its sha256.
  - Copy P0 train.py to P13/dtr_maze.py.
- Write PREREG.md (this spec, frozen) before any GPU spend. Before freezing it:
  - read P11's summary (done), plus market-resnet phase 3's regularizer results and P6's results if they exist at that time, and record them with the time read;
  - verify every arXiv ID marked (m).

DATA
- Training data: P0 frozen maze_train9.npz (50k e2h mazes, checked against the manifest sha256), using P0 train.py's split of 40k train and 10k held-out 9x9 val. The split is identical for every arm. Image side is 24 px.
- Test sets (P1 data.py MAZE_SPLITS eval indices):
  - 19: 200 of 600 (44 px)
  - 33: 300 of 600 (72 px)
  - 59: 200 of 500 (124 px)
  - 99: all 200 (204 px; P1 make_mazes_upstream.py)
- Stretch, 201 (408 px): 30 mazes from make_mazes_upstream.py with seed 13001, sha256 frozen before any model is evaluated. If generation takes more than 30 CPU-min, use P0's m201 instead, flagged as a different generator.
- Dev sets, for selection only and never scored: 9x9 val, P1 13 dev (200), P0 v33, and P1 59 dev (100).
- Exact match uses the open-pixel mask (P0 check.py). Log loss is in nats per open pixel.

PRE-REGISTERED PREDICTION (from mechanism note (6) and the 1D evidence)
- The economy will be much weaker relative to Bansal on mazes than in 1D. P(S2) is about 0.2.
- At W0 = 20 the wealth cap will bind on fewer than 1% of trades, so ECON-fix will be about equal to ECON-nocap.
- Report the per-level cov-market log loss at 59 and 99, to show whether the dyadic targets are learnable on mazes at all.

2D PORT (econ_maze.py; one rollout function shared by the econ and bptt paths)
Helpers:
- shift2d;
- J = ceil(log2(side));
- pair_truth2d: equality targets under the open-pixel mask at offsets (2^j, 0) and (0, 2^j), so c has 2J channels.

Trader2D:
- An RGB embedding, recalled every step.
- A dilation-1 3x3 read of [embedding, tanh(b/4)]. Flood fill needs b at dilation 1.
- Per level j, a 3x3 read at dilation 2^j of [embedding, tanh(b/4), c_j, c_{j-1}].
- The MLP is shared across levels.
- Outputs: db (summed over j, as in 1D), dc_j, dv(3) and the gain bid.
- Belief markets, pay and every loss are masked to open pixels.
- N = 4. Width w is chosen so that per-step FLOPs at 24 px are within +-25% of DT-recall w=128 (flops.py). Report parameter counts.

ECON settings:
- Traders read detached state and are paid LS(b_t) - LS(b_{t-1}), the LMSR log score.
- W0 = 20, frac 0.5, wealth_cap on, rent 1e-3, respawns.
- Cov markets with lam_c = 1.
- VOC markets with k = 5, 15 and 30, lam_v 0.3, settle eps 0.1. All k are at most D, so each market has a distinct training target. v1's k = 50 and 500 gave identical targets at D = 30.
- Batched train_step, not --online: econ2 asserts that online is not combined with VOC or with any regularizer, and D = 30 at 24 px fits in memory.
- --train_pick random for every Trader2D arm (the 1D econ_cov_voc headline; market-resnet phase 3's econ_cov_voc_rr trains round robin). Round robin at test.
- D = 30, lr 1e-3, batch 50.
Settings for every arm:
- TF32 on in training (P0 train.py's default) and off at test.
- The final checkpoint is always used (no best-val).
Economy diagnostics, logged per 1k iterations:
- the fraction of trade-pixels with wealth scale s < 1;
- minimum wealth and wealth shares;
- respawns and entries.

FIX SEMANTICS (identical in the econ and bptt paths)
Trained fixes:
- bcap c: b <- clamp(b, +-c) after every step (torch.clamp, no gradient past the bound).
- dbcap d: db <- d * tanh(db/d).
- liq sqrt: as econ2 --liq, inside the rollout.
- l2b lam:
  - econ: a fee lam*(b_t^2 - b_{t-1}^2) per bit, charged to the mover;
  - bptt: a loss term lam*(b_t^2 - sg(b_{t-1})^2) per step.
- floor eps: a fixed-share forecast (1-eps)p + eps/2 in every log score.
- insq (the trained part of v1's F-IN): the extra input sign(b)*log1p|b|.

Post-hoc test-time options (no training cost):
- J option, every Trader2D arm. Choose one of: none; scale the level sum by J_train/J; cap J_test = J_train = 5. Selected per arm on 59 dev at 12n: highest exact, ties to fewer FLOPs.
- Calibration, every arm including DT-recall, fit on 59 dev:
  - a logit clip with c in {4, 8, 16};
  - temperature scaling.
  - "Best post-hoc" is whichever gives the lowest dev log loss.

RUNAWAY-FIX PILOT (pre-registered selection)
Candidates, each with one setting; the settings match market-resnet phase 3 so the results transfer:
- F0: none.
- F-TR1: dbcap 1.
- F-TR2: liq sqrt.
- F-LB: bcap 10.
- F-HF: l2b 1e-3.
- F-FL: floor 0.01.
- F-IN: insq.

Procedure:
- Run each candidate on ECON and on ECON-nocap: 14 runs, 1 seed each, 0.5 L4-h each.
- Evaluate on 9x9 val, 13 dev, v33 and 59 dev, all at T = 12n.
- Pool per arm: 9x9 val >= 0.9, and v33 exact within 0.02 of that arm's best candidate.
- Rank each candidate by v33 log loss after its own best post-hoc control, so the rank measures what a fix does beyond clipping and temperature scaling.
- Select the lowest mean rank over the two arms. If mean ranks are within 0.5, pick the higher mean 59-dev exact.
- If F0 wins, there is no training-time fix: ECON-fix = ECON-F0 + best post-hoc control, F2 is recorded, and the ECON-F0 main run merges into ECON-fix.
- Extension: if fewer than 3 candidates per arm reach 9x9 val >= 0.9 at 0.5 h, extend all 14 runs to 1.0 L4-h before selecting.
- Learnability gate: the selected fix must reach 9x9 val >= 0.9 and 13 dev >= 0.5 at 1.0 L4-h (extend it if needed) in ECON or ECON-nocap. Otherwise stop, report the F0-gate outcome, and run no main phase.
- P11 prior (recorded in PREREG): caps and clips on DT-recall mazes saturated 69-93% of pixel-steps and cost extrapolation accuracy. Expect the accuracy guard to remove F-TR1 and F-LB.

ECON-ACTIVE LADDER (stage 2B, after fix selection)
- If ECON-fix's pilot has s < 1 on at least 10% of trade-pixels in the last 25% of the pilot, ECON-fix is itself active: skip the ladder and ECON-active.
- Otherwise run two candidates with the selected fix, 0.5 L4-h each:
  - A1: W0 = 0.5;
  - A2: --bounty lindahl (trader-funded cov markets).
- Select the candidate with s < 1 on at least 10% of trade-pixels and 9x9 val >= 0.9. If both qualify, take the higher v33 exact.
- If neither qualifies, ECON-active is not run, and S3 is recorded as "economics could not be made to bind".

EQUAL COMPUTE
- B_ref is the Bansal recipe of 40k steps at batch 50, timed from 300 DTR-eq steps on a clean L4.
- B = min(B_ref, max(3.0 L4-h, the time DTR-eq needs to reach 25k steps)), capped at 3.5 L4-h. If B > 3.0, the BPTT arms drop to 1 seed each.
- Each arm's budget is B x its Phase-1 clean-VM throughput, turned into a step count, and every run trains to its step count. This is preemption-proof and immune to per-VM speed differences.
- DTR-eq's schedule is compressed to its step budget: warmup is 20% of the steps (the recipe warms up for 10 of 50 epochs), with constant LR afterwards (the recipe's milestone at epoch 100 is never reached).
- Report training FLOPs per arm. If any Trader2D arm differs from DTR-eq by more than 2x at equal time, add 1 seed of DTR-eqflop.

TEST PROTOCOL
- Every model runs to T = 12n: 1188 steps at 99, 2412 at 201.
- Curves are recorded every 10 steps, as float16:
  - exact accuracy;
  - log loss, raw and after the best post-hoc control;
  - max and mean |b|;
  - per-level cov log loss;
  - cycles of period p <= 8 in the thresholded mask (this covers 2-cycles and the 4-trader round robin).
- Equal test FLOPs: for each arm and size, t_eq is the step at which cumulative FLOPs equal M-ref's at 12n. Trader2D's cost depends on J and includes the dev-chosen J option.
- 20n extension at 99: M-ref, the median-dev ECON-fix seed and the median-dev DTR-eq seed.
- 201 stretch: M-ref and the median-dev seed (by 59-dev exact) of ECON-fix, ECON-nocap and DTR-eq. These seeds are fixed in advance; the best seed is never chosen.
- Secondary: VOC-halted exact accuracy and steps against the KL-lag-2 rule on the same net.

STATISTICS
- Primary endpoint: exact accuracy at n = 99 at equal test FLOPs (t_eq).
- Secondary endpoints: 12n (equal steps); best-over-t up to 20n; n = 59; n = 201; log loss.
- Primary CI: a 95% Welch t-interval over per-seed means (maze-averaged).
- Secondary checks: a hierarchical bootstrap (seed, then maze; 10k resamples) and an exact permutation p over seed means (70 splits at 4 vs 4). Per-seed tables are always shown.
- Intent-to-treat: every seed counts, including collapsed seeds. Results on gate-passing seeds only are reported alongside.
- No other endpoint is chosen after the fact.

VALIDITY GATES (applied symmetrically)
- M-ref must score 1.00 at 13 and at least 0.99 at 59 on our pipeline; otherwise the run is invalid.
- A seed of any trained arm is "trained" if it scores 9x9 val >= 0.95 and v33 >= 0.5. Trained status is reported for every arm.
- If fewer than 2 DTR-eq seeds pass:
  - S2's equal-compute leg reads "Bansal's recipe does not train reliably at this compute (k of 4)";
  - S2 is judged against M-ref only;
  - a claim of beating Bansal then requires S4.

CPU UNIT TESTS (on General, before GPU)
- (a) shift2d, pair_truth2d and nlevels on toy grids, plus the XNOR composition on a toy maze.
- (b) ECON with --wealth_cap 0 against the bptt path (--tbptt 1 --ds 1 --covsup 1 --vocsup 1 --bidsup 1 --ptopt 1 --train_pick random). The parameter updates after clipping and Adam must be identical over 3 iterations, including the v and bid heads, within 1e-5. This shows that with the cap off, market settlement equals per-step local loss.
- (c) ECON with W0 = 1e9 against --wealth_cap 0: identical trajectories and updates.
- (d) For each fix flag, the econ and bptt paths give identical b, c and v over 30 steps from identical weights.
- (e) Resume:
  - on CPU, bit-identical over 50 steps, covering the model, every optimizer, the ledger, respawns, the RNGs and the data position;
  - on GPU, state checksums must match at resume and the loss must agree within 1e-4.
- (f) M-ref through P0 dtnet.py scores 1.00 on 20 13x13 mazes.
- (g) flops.py per step for every arm at 24, 124, 204 and 408 px, with each J option.
- (h) dtr_maze.py with warmup_frac 0.2 and 40k steps matches P0 train.py for the first 200 steps (same seed), within 1e-6.

FLEET NOTE (Phase 0; no edits in market-resnet/)
Send market-resnet phase 3's owner a short note. Their bptt_cov_dscov_tb1_rand has two confounds:
- untrained-v drift: dv gets zero gradient under tbptt 1 without vocsup;
- a joint optimizer where the econ arm uses per-trader optimizers.

CHANGE MAP (v1 to v2)
Adopted:
- S3 comparator is ECON-nocap, in the same code (critique 1 fix 1, critique 2 fix 2).
- Economy-activity logging and the "inactive, untestable" rule, plus the ECON-active ladder.
- --online dropped; train_pick stated for every arm.
- Fix semantics specified under backprop, with test (d).
- Plain final-loss BPTT replaced by BPTT-dscov and BPTT-progcov.
- The Hillis-Steele note and its prediction.
- P11, phase 3 and P6 read and cited.
- Fix ranking done after each candidate's best post-hoc control, and run on both ECON and ECON-nocap.
- F-IN split into a trained part and post-hoc parts; J cap added.
- floor added as a candidate.
- DTR-eq built from P0 train.py and trained to a step budget.
- B raised to 3.0-3.5 L4-h.
- DTR-eq v33 gate; failure of DTR-eq is a pre-declared outcome.
- Primary endpoint at equal test FLOPs; S4 also checked against M-ref best-over-t to 20n.
- Welch t-interval as the primary CI; median-dev seeds pre-assigned.
- Learnability gate raised to 13 dev >= 0.5.
- Citations corrected; all 200 n=99 mazes used; the "13 dev" split named.
- Cycle detection covers periods up to 8; k windows at most D.
- GPU resume tested by checksum; quota checked.

Not adopted, because of the $40 cap:
- 5 seeds and a $60 cap. v2 uses 4 seeds for ECON-fix, ECON-nocap and DTR-eq, with an exact permutation test.
- B = 4.0 h. B is capped at 3.5 h.
- LOCAL as a separate BPTT-class GPU arm. It is kept as unit test (b).

## Spec: arms

- ECON-fix (the primary economy, as posed): Trader2D economy with cov markets and VOC markets (k 5/15/30), W0 20, wealth cap, ledger, respawns and LMSR log-score pay on detached state; batched train_step, random pick in training, round robin at test; pilot-selected runaway fix. 4 seeds x B.
- ECON-nocap (detached local training; the S3 comparator): same code and fix with --wealth_cap 0 (s = 1, no respawns, ledger logged only). Unit test (b) shows it is exactly per-step local-loss training. 4 seeds x B.
- ECON-active (conditional; economics forced to bind): runs only if ECON-fix's cap binds on fewer than 10% of trade-pixels in the pilot. Ladder-selected W0 = 0.5, or lindahl trader-funded cov markets; same fix. 2 seeds x B.
- ECON-F0: ECON-fix without a training-time fix. Shows the runaway and is the S1 reference (raw and after its best post-hoc clip or temperature). 1 seed x B; merged into ECON-fix if F0 is selected.
- DTR-eq (Bansal at equal compute, faithful retrain): a copy of P0 train.py (dtr_maze.py), DT-recall w=128, progressive loss alpha 0.01, max_iters 30, batch 50, Adam lr 1e-3 wd 2e-4 with recur params at lr/30, warmup 20% of its step budget, open-pixel loss, same 9x9 split, final checkpoint. 4 seeds.
- BPTT-dscov (tests detaching): Trader2D, full backprop through D=30, loss on every step's belief plus cov and VOC targets (--ds 1 --covsup 1 --vocsup 1 --ptopt 1 --train_pick random), same fix. The 2D analogue of 1D bptt_cov_dscov. 2 seeds x B.
- BPTT-progcov (Bansal's training rule on the trader architecture with hints): Trader2D with progressive loss alpha 0.01, covsup, vocsup, ptopt and random pick, same fix. 2 seeds x B.
- M-ref (Bansal's full recipe, more compute): released DT-recall checkpoint (P0 src/dt_net.pth), eval only, with the same sets, caps and post-hoc calibration controls.
- Optional, only if spend after the main phase is under $30: DTR-pair (DT-recall w=128 plus a pair_truth2d auxiliary head, 1 seed; puts the hints on Bansal's architecture). DTR-eqflop (1 seed), only if the FLOP-mismatch rule fires.

## Spec: metrics

- Exact-match accuracy (open-pixel mask) at n=99 at equal test FLOPs t_eq (primary). Also at T=12n; at n=19/33/59; at n=201 (stretch); and best-over-t up to 20n for M-ref and the pre-assigned median-dev seeds.
- Log loss in nats per open pixel at t_eq and 12n (mean and median over mazes), raw and after the best post-hoc control (clip c in {4,8,16} or temperature, fit on 59 dev), for every arm including DT-recall.
- Runaway diagnostics: max and mean |b| against t, and the slope of log mean|b| against log t over [n, 12n].
- Economy activity per seed, every 1k iterations: fraction of trade-pixels with wealth scale s<1, minimum wealth, wealth shares, respawns and entries.
- Per-level cov-market log loss at 59 and 99 (are the dyadic targets learnable on mazes).
- Calibration: ECE (15 bins, open pixels) and AUROC of pixel-error detection from |b|.
- Accuracy against cumulative test FLOPs, with t_eq per arm and size, and the dev-chosen J option per arm.
- Training compute per arm: steps, L4-seconds, FLOPs and parameter counts.
- Seed validity: 9x9 val and v33 exact per seed, and trained/failed counts per arm.
- Limit-cycle rate: fraction of mazes in a cycle of period 8 or less, at t_eq and 12n.
- Secondary halting: VOC-halted exact accuracy and mean steps against the KL-lag-2 rule on the same net.
- Welch t 95% CIs (primary), hierarchical-bootstrap CIs and exact permutation p for ECON-fix minus DTR-eq, ECON-nocap, BPTT-dscov, BPTT-progcov and M-ref; for ECON-active minus ECON-nocap; and for ECON-nocap minus BPTT-dscov.
- Spend: VM-hours, $ at $0.35/h, preemptions, and wall-clock time per phase.

## Spec: success

Pre-registered, in tiers. All tiers use eval sets never used for selection.
- Primary endpoint: exact accuracy at n=99 at equal test FLOPs.
- CIs are Welch t over seed means, intent-to-treat. Hierarchical bootstrap and gate-passing-only results are reported alongside.

S1, runaway fixed. All of:
- ECON-fix's 99 log loss is at most 0.1x ECON-F0's raw log loss;
- it is also below ECON-F0's log loss after F0's best post-hoc control (clip or temperature, fit on 59 dev);
- ECON-fix's exact accuracy is at least ECON-F0's minus 2 pp.
Labelling:
- "Fixed" if the upper CI of the slope of log mean|b| against log t over [n, 12n] is at most 0.1; otherwise "slowed".
- If F-LB or F-FL is selected, only the post-hoc comparison counts toward S1.

S2, beats Bansal at equal compute. All of:
- the CI of ECON-fix minus DTR-eq at 99 is above 0, both over all seeds and, when at least 2 DTR-eq seeds pass the gate, over the passing seeds;
- ECON-fix is at least M-ref minus 2 pp;
- ECON-fix's raw log loss is at most DTR-eq's.
The win is credited to "detached training plus dyadic pair hints plus trader architecture", not to markets, unless S3 also holds.

S3, the gain is economics. The active ECON arm is ECON-fix if its cap binds, else ECON-active. Both of:
- in every seed, it has s<1 on at least 1% of trade-pixels after the first 10% of training, or at least 1 respawn;
- the CI of the active arm minus ECON-nocap at 99 is above 0.
If the economy is inactive in every seed, S3 is reported as "economics inactive, untestable", whatever the CI says.

S3b, detaching helps: the CI of ECON-nocap minus BPTT-dscov at 99 is above 0.

S4, beats the published Bansal model outright. Both of:
- the CI of ECON-fix minus M-ref at 99 at equal test FLOPs is above 0;
- ECON-fix at 12n beats M-ref's best-over-t up to 20n.
Otherwise any lead is recorded as "faster per step", not as beating Bansal.

Gates (they invalidate or qualify the run; they do not score it):
- M-ref must score 1.00 at 13 and at least 0.99 at 59; otherwise the run is invalid.
- Trained status is reported for every seed of every arm.
- If fewer than 2 DTR-eq seeds are trained, the outcome "Bansal does not train reliably at this compute" is pre-declared, and "beats Bansal" requires S4.

## Spec: falsify

- **F1: the economy does not beat Bansal at the largest sizes (as posed).** This holds if, at n = 99 at equal test FLOPs, the CI of ECON-fix − DTR-eq includes or lies below 0, or ECON-fix is more than 2 pp below M-ref.
  - This is evidence at 11x size and about 40x depth. It is not THEORY.md's programme falsifier, which compares against deep supervision at 1000x depth.
  - If F1 holds, add to THEORY.md: on mazes at this scale, the economy is a valuation/control tool, not an extrapolation mechanism.
- **F2: no training-time fix beats F0 plus its best post-hoc control** (clip or temperature scaling) on v33 and 99 log loss while staying within 2 pp exact accuracy. The runaway is then a calibration problem that post-hoc methods solve.
- **F3: the economics adds nothing beyond detached local training.** The economy is active by the S3 rule, but the CI of the active arm − ECON-nocap at 99 includes 0 or is negative. Any gain then belongs to the RAFT/HRM/TRM line, and this is the result that speaks to the programme falsifier. If the economy never binds, this is recorded as "the economy reduces to local losses on mazes".
- **F4: detaching adds nothing either.** BPTT-dscov ≥ ECON-nocap at 99. Any win then belongs to the hints plus the trader architecture.
- **F0-gate: the port cannot learn mazes.** No candidate reaches 9x9 val ≥ 0.9 and 13 dev ≥ 0.5 at 1.0 L4-h. The run then stops before the main phase and reports this.

All thresholds, B, sizes, caps, seeds, the median-dev seed rule and both selection rules are frozen in PREREG.md before any GPU spend.

## Spec: compute_plan

Window: 19:17Z 29 Sep to 19:17Z 30 Sep. This revision was marked at 19:56Z on 29 Sep. Hard stop: every P13 VM deleted by 17:17Z on 30 Sep. Cap: $40.

Phase 0 (CPU on General, $0, until about 23:30Z)
- Write econ_maze.py, dtr_maze.py (a copy of P0 train.py, adding --warmup_frac and a step budget) and eval_maze.py.
- Run unit tests (a) to (h).
- Read phase 3 and P6 results if available, and verify the (m) citations.
- Write PREREG.md.
- Send the fleet note to the phase 3 owner.
- Append "P13 v2 approved, cap $40" to calibrated-ai/BUDGET-24H-20260929.md.

Phase 1 (1 VM, about 0.7 VM-h, until about 00:15Z)
- 300-step s/iter at 24 px for every arm, including DTR-eq w=128 (this sets B_ref, B and the step budgets).
- flops.py at every size and J option.
- A 50-step eval throughput test at 204 px (M-ref and Trader2D).
- A GPU kill-and-resume checksum test.
- The L4 spot quota check: regional PREEMPTIBLE_NVIDIA_L4_GPUS and NVIDIA_L4_GPUS plus global GPUS_ALL_REGIONS, minus market-resnet phase 3's 6 VMs and the distillation VMs. P13 uses up to 10 VMs; if fewer than 8 are free, the wall-time cut below applies early.

Phase 2A (about 8.4 VM-h, until about 01:45Z)
- 14 pilots at 0.5 L4-h, plus dev eval.
- Contingency: extending all pilots to 1.0 h adds about 8.4 VM-h.

Phase 2B (about 1.2 VM-h, until about 02:30Z)
- The ECON-active ladder, only if needed.

Phase 3 (about 57 VM-h at B = 3.0, on 8-10 VMs in 2-3 rounds, until about 08:30-11:30Z)
- 19 runs: ECON-fix 4, ECON-nocap 4, DTR-eq 4, ECON-active 2, BPTT-dscov 2, BPTT-progcov 2, ECON-F0 1.
- If B > 3.0, the BPTT arms drop to 1 seed each, which keeps this phase under 60 VM-h.

Phase 4 (about 17.6 VM-h, interleaved on free VMs, done by about 13:30Z)
- 21 models at 19/33/59/99 (about 12.6).
- J-option dev selection (about 2).
- 20n extensions (about 1).
- The 201 stretch for 4 models (about 2).

Spend
- About 85 VM-h, plus 10% overhead, comes to about 93 VM-h: about $33 at $0.35/h. That rate is g2-standard-4 + 1x L4 spot + a 50 GB disk; the ledger observed $0.32/h. With the pilot contingency it is about 102 VM-h, or $36.
- The cap is enforced on the spot-rate estimate, as market-resnet does, and reconciled against GCP billing afterwards.

VMs
- Named trader-b3-p13-<k>; recreates use -r<N>. GCP project GCP_PROJECT.
- g2-standard-4 with 1x L4, SPOT, --max-run-duration 6h, --instance-termination-action DELETE, boot disk auto-delete, 50 GB.
- Multi-zone and multi-region fallback over the 13 L4 zones used in P0.
- Launchers are copies (not edits) of consistency-ft-next-run-20260919/scripts/gpu/, P1 gpu.sh and the market-resnet run_p3.sh/watchdog3.sh pattern:
  - a jobs.txt queue ordered by priority, with .done markers;
  - resume from .ckpt.pt;
  - apt timers masked.

Checkpoints and sync
- Checkpoint every 10 min, atomically: model, every trader optimizer, wealth ledger, respawns, step, train-seconds, CPU and CUDA RNG, and data position.
- Sync to General trader-economy/puzzles/P13/results/vm/<vm>/ every 10 min or less, with sha256 verification.
- After a preemption, the job resumes on a new VM from the last synced checkpoint. Runs train to step budgets, so preemptions do not change compute.

Watchdog (5-min loop)
- It sums the hours of every trader-b3-p13-* VM, deleted ones included, at $0.35/h and deletes everything at $40 (about 114 VM-h).
- It deletes any VM at age 5.8 h or after 20 min idle.

Cut order (for spend or wall time)
- Cut first: the 201 stretch, then the optional arms, then BPTT-progcov seed 2, then BPTT-dscov seed 2, then the ECON-F0 main run (S1 then uses the F0 pilot, flagged), then ECON-active seed 2.
- Never cut ECON-fix, ECON-nocap, DTR-eq (4 seeds each) or M-ref at 99.
- Wall-time cut at 11:00Z on 30 Sep: any job not started that cannot finish, including its eval, by 15:30Z is dropped in cut order. That leaves about 1.7 h of slack before the 17:17Z hard stop.

Records
- Every VM gets a row in trader-economy/GPU-LEDGER.md (P13 section), and each spend line goes to BUDGET-24H-20260929.md.
- A fleet-now change line about every 2 h and at each phase boundary.

Disk (/srv is near full)
- Keep only the final .pt files (about 100 MB), float16 curves every 10 steps, and 5 example trajectories per size.
- Scratch goes through trash-put; never rm -rf outside /tmp.

Cleanup
- After the final verified sync, delete VMs and disks.
- Confirm that `gcloud compute instances list` and `gcloud compute disks list --filter='name~trader-b3-p13'` are both empty, and log the check time.

Other rules
- No paid LLM APIs.
- No AI attribution in any commit.
- No edits under market-resnet/ or any other puzzle directory.

## A. Implementation choices frozen with the spec (P13 agent, 29 Sep)

These are fixed before any GPU spend. Where the implementation departs from the spec text, the departure is listed in B.

A1. Data and sets (frozen/manifest.json, file sha256 ea22733d...; P0 frozen/manifest.json)
- Train: P0 maze_train9.npz (sha256 4cfd5b7e...), P0 split random_split 40000/10000 seed 42; 24 px; identical for every arm.
- Dev (selection only, never scored): v9 = 1000 mazes of the 10k val split (the "9x9 val"); m13 = e2h test_13 [0:200] (the
  "13 dev"); v33 = e2h test_33 [500:600]; m59 = e2h test_59 [0:100] (the "59 dev"); d99 = upstream 99 chunk [0:50].
- Test (scored): e19 = test_19 [200:400]; e33 = test_33 [400:500]+[600:800] (excludes v33); e59 = test_59 [500:700];
  e99 = all 200 P1 99 eval mazes; e201 = 30 mazes, make_mazes_upstream seed 13001 (43.6 CPU-s, same generator; sha256 of
  code/sol/sp_len in frozen/manifest.json, frozen 20:12Z before any model was evaluated).

A2. Architecture and compute (flops.py; results/flops.json; unit test (g))
- Trader2D width w = 224, embedding width 32, N = 4 traders; about 206k parameters per trader (0.82M total).
  Per-step MACs per pixel at 24 px (J = 5) are 0.994x DT-recall w = 128 (inside the +-25% rule).
- t_eq at 99 (equal test MACs to M-ref at 12n = 1188): J option none 774, scale 774, cap 1195. At 201: 1407 / 1407 / 2427.
- DT-recall w = 128 (dtr_maze.py = P0 train.py + --warmup_frac + step budget + --time_steps).

A3. Seeds (fixed now)
- Pilots: seed 100. ECON-fix, ECON-nocap, DTR-eq: seeds 0, 1, 2, 3. ECON-active, BPTT-dscov, BPTT-progcov: seeds 0, 1
  (seed 0 only if B > 3.0 h). ECON-F0: seed 0. Optional DTR-pair / DTR-eqflop: seed 0.
- Median-dev seed rule (20n extension and 201 stretch): sort an arm's 4 seeds by m59 exact at 12n, descending, ties to the
  lower seed id; the median-dev seed is the one in position 2 (upper median). The best seed is never chosen.

A4. Budgets
- Clean throughput = Phase-1 300-step timing on one L4 with no other job (after 20 warm-up steps).
- B_ref = 40,000 x DTR-eq clean s/step. B = min(B_ref, max(3.0 h, 25,000 x DTR-eq clean s/step)), then min(B, 3.5 h).
- Steps per arm = floor(B / arm clean s/step); pilots: floor(0.5 h / arm clean s/step) (1.0 h if extended).
  Fixes with extra cost (insq, liq) use their own timing; other fixes use the plain ECON / ECON-nocap timing.
- Jobs may be packed 2-3 per VM (Phase-1 packed timing decides); packing changes wall time only, never the step count.
- Trader2D arms: constant lr 1e-3 (econ2 default; no schedule in the spec). DTR-eq: warmup 20% of its steps, then constant.

A5. Pilot selection details
- Every pilot is evaluated with J option none at T = 12n on v9, m13, v33 and m59.
- Best post-hoc control per candidate: among raw, clip c in {4, 8, 16} and temperature tau in {2^(k/2), k = -2..16}, the one
  with the lowest m59 log loss at 12n; its v33 log loss (mean over mazes, nats per open pixel) is the ranking score.
- Pool per arm (ECON, ECON-nocap): v9 exact >= 0.9 and v33 exact >= (arm's best candidate v33 exact) - 0.02.
  Candidates outside an arm's pool take rank (pool size + 1) in that arm. Mean rank over the two arms; lowest wins; if the two
  lowest mean ranks are within 0.5, the higher mean m59 exact wins.
- s < 1 share for the ladder rule: the econ diagnostic frac_capped_pix averaged over the log records in the last 25% of the
  pilot's iterations.

A6. Evaluation
- T = 12n for every model; records every 10 steps plus t_eq and 12n; TF32 off; final checkpoint.
- J option per Trader2D arm chosen on m59 at 12n (highest seed-mean exact; ties to fewer MACs), then applied to every seed
  of that arm.
- Post-hoc calibration per model: fit on m59 at 12n (lowest log loss among raw, 3 clips, 19 temperatures), applied at every
  size and step.
- Test-time Trader2D trades use round robin; in econ arms with the wealth cap, each trade is scaled by s computed from the
  trader's final training wealth (as econ2's evaluation does); ECON-nocap trades unscaled.

A7. Statistics
- Per-seed mean exact over mazes; Welch t 95% CI on the difference of arm means (df by Welch-Satterthwaite).
- Hierarchical bootstrap: 10,000 resamples, seeds with replacement within arm, then mazes with replacement within seed.
- Exact permutation p: all C(8,4) = 70 splits for 4 vs 4 (two-sided, on the difference of seed means); C(6,2) = 15 for 4 vs 2.

## B. Deviations and simplifications (stated in advance)

1. Trader2D reads c_{j-1} at dilation 2^(j-1) (the Hillis-Steele term), and its base read includes tanh(b/4); both as in
   econ_maze.py and tested in (a).
2. Slot rent and VOC settle eps are logged but inactive (no slots in the 2D port; settlement is the per-step local rule).
3. Lindahl-funded cov markets (ladder A2) use the same fixed transforms as the house-funded markets.
4. BPTT arms use gradient checkpointing (identical numerics, extra recompute counted in FLOPs).
5. The pilot uses J option none only (the J option is selected for the main arms after training).
6. The 20n extension and VOC-halting secondary are run only if wall time allows under the cut order; example trajectories
   are skipped.
7. Unit test (b) is run at width 8, D = 6, batch 3 in float64 on CPU (General is shared); (d) at width 8 with the output
   head scaled x20 so every fix binds; (e) CPU resume uses 50 iterations for ECON, 20 for BPTT, 30 steps for DT-recall.
8. GPU resume is judged against the GPU nondeterminism floor: resumed vs uninterrupted must agree within max(1e-4,
   2x the gap between two uninterrupted runs), plus an exact checksum of the reloaded state.
9. Sync directories are per slot (results/vm/trader-b3-p13-K/), shared by the slot's -rN recreates.
10. The 13 dev set is e2h test_13 [0:200] and the 59 dev set is e2h test_59 [0:100] (P0 frozen sets m13, m59).

## C. Readings recorded at freeze (read by the P13 agent, times UTC 29 Sep)

- market-resnet phase 3 (1D prefix parity; acc at 100x / NLL), read 20:32-20:33Z: econ_cov_voc 0.933 / 501; floor 0.642 / 0.71;
  l2b 0.558 / 0.10; bcap 1.000 / 5.9e-5 (best); dbcap 0.417 / 221; liq t 0.342 / 12.6; liq sqrt 0.708 / 91.9;
  bptt_cov_dscov_tb1_voc_rand 0.792; tb1_voc_rep_rand 0.892; tb1_rand 0.775; bptt_cov_progcov 0.812; dtr_full 0.983;
  dtr_eqflop 1.000; dtr_eqtime 0.975. In 1D, bcap 10 fixed the runaway without losing accuracy; the maze pilot decides here.
- P6 (1D; EMA-clone and PBT controls without the ledger), read 20:34Z: JOB_DONE, analysis pending; EC (economy) and F (fixed)
  both reach L256 at 5000 iterations with exact about 0.79-1.0; R0 0.81 and 0.09.
- P11 summary (from the spec, read 19:42Z by its author): FALSIFIED (F2); caps/clips saturate 69-93% of pixel-steps.
- L4 quota at 20:43Z: PREEMPTIBLE_NVIDIA_L4_GPUS 16 per region; in use us-central1 6, us-east1 5, us-west1 2, us-west4 2, others 0.
  No global GPUS_ALL_REGIONS quota listed for the project. At least 8 P13 VMs fit.

## D. Citations marked (m), verified about 20:35Z on arxiv.org

- 2003.12039: Teed and Deng, RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. Correct.
- 2205.15659: Velickovic et al., The CLRS Algorithmic Reasoning Benchmark. Correct.
- 1706.04599: Guo, Pleiss, Sun and Weinberger, On Calibration of Modern Neural Networks. Correct.
- 1905.04753: Li, Yumer and Ramanan, Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints. Correct.

## E. Unit tests (results/unit_tests/<t>.json)

Status at freeze: (a) pass, (b) pass (max param diff 3.5e-17 per iteration; v/bid heads 0 and 8.7e-19), (c) pass (diff 0.0),
(d) pass (every fix: econ vs bptt rollout, settlement vs rollout, eval step vs rollout all 0.0; binding caps included),
(g) pass (ratio 0.994). (e), (f), (h) and the non-vacuity reruns of (b), (c) were still running at freeze; their results are
appended below the freeze line, and a failure stops GPU work until fixed.

---
FREEZE: frozen at 2026-09-29T20:46:05Z; sha256 of everything above this line: a19d99e7de9c4fe6c7300cecf555738ea4abefb7b64b81525ae748c6c9108d63. Nothing above is edited after this point.

## Addenda after freeze (results only, with times)

- 20:47:55Z unit test (e) PASS: ECON bit-identical over 50 iterations killed at 23 (1 respawn in the run), BPTT-progcov
  bit-identical over 20 killed at 9, DT-recall bit-identical over 30 steps killed mid-run (CPU, 925 s).
- 20:49Z deviation 11 (infrastructure): boot disk 100 GB, not 50 GB; the image refuses a smaller disk. Cost is inside the
  $0.35/h estimate (the watchdog cap is unchanged).
- 20:54Z operations: launch.sh's "runner up" check matched its own ssh command line (pgrep -f), so trader-b3-p13-1 came up
  without its runner for 4 min; fixed with pgrep -fx and the runner started by hand. No job was affected.
- 20:58Z code (no effect on any step count or number): econ_maze.py gains --final_ckpt 1, which keeps a resumable checkpoint
  at the end of a run. Pilots use it so the pre-registered 1.0 h extension resumes instead of retraining.
- 21:05Z unit test (f) PASS: M-ref on our pipeline scores 1.00 exact on m13[0:20] at 12n, raw log loss 4.8e-4 (CPU, 1492 s).
  Non-vacuity reruns of (b) and (c) PASS: the 3 Adam updates move every parameter group (max move 3.0e-3 overall, 3.0e-3 on
  the v rows, 3.0e-3 on the bid head) while ECON-nocap vs bptt stays within 3.5e-17 per iteration (v head 0.0, bid 8.7e-19)
  and (c) within 0.0. (h) is still running.
- 21:04Z operations: ctl.sh now also deletes a VM at the next 3-min round once its runner wrote RUNNER_DONE and its queue has
  no pending job (sha256-verified sync first), instead of waiting 10 min idle. Saves VM time only.
- 21:32Z unit test (h) PASS: our DT-recall trainer matches the P0 reference train.py step for step over 200 steps (max
  parameter difference 0.0, CPU, 3583 s). All unit tests (a)-(h) now pass.
- 21:40Z operations: ctl.sh keeps at most MAXVMS=10 P13 VMs up at once and launches slots in slots.list order.
- 21:45Z Phase 1 (trader-b3-p13-1, L4): solo s/1k steps econ 1090.8, nocap 1095.7, insq 1088.0, liq sqrt 1097.3, bptt-dscov
  918.2, bptt-progcov 1252.4, dtr 354.6 (peak memory 1.3-5.6 GB). Two ECON runs packed on one L4 run at 2386-2390 s/1k each
  (slowdown 2.19), so packing costs more VM time than running solo; every VM runs k=1 job at a time. The econ x3 timing was
  stopped on purpose once x2 showed packing loses (its error entry in phase1.json is ignored). DT-recall at 0.3546 s/step
  gives B_ref = 3.94 h and B = 3.0 h (the cap), so BPTT arms keep 2 seeds. Budgets: gpu/budgets.json (main steps econ 9900,
  nocap 9856, bptt-dscov 11762, bptt-progcov 8623, dtr 30456; pilots econ 1650, nocap 1642 = 0.5 h each).
- 21:53Z pilots: 14 pilots (7 fixes x ECON/ECON-nocap, seed 100) on 7 VMs (trader-b3-p13-2..8), 2 pilots per VM run in
  sequence with k=1, each followed by its dev eval (v9, m13, v33, m59 at 12n, J option none); the M-ref gate eval rides on
  -8. DT-recall-eq and the M-ref evals do not depend on the pilot selection and are queued as soon as Phase 1 finishes.
- 21:54Z logging (no effect on any step, update or random number): trader runs log the economy diagnostics
  (frac_capped_pix, mean s, wealth shares) every 100 iterations instead of every 1000. The counters accumulate every
  iteration either way; at 1000 a 1650-iteration pilot would have had one record in its last 25%, which the ladder rule
  (frac_capped over the last 25%) and the activity analysis need. Applies to pilots and main runs.
- 22:03Z operations: trader-b3-p13-1 was spot-preempted during the dtr_x2 timing. Solo, econ_x2 and econ_x3 had synced
  (out/phase1.json, which set the budgets above). The dtr+econ line was read from the live log over ssh before the
  preemption and is kept in gpu/phase1_recovered.json: DT-recall 727.4 s/1k (x2.05 of solo), ECON 1359.6 s/1k (x1.25),
  wall 499 s. dtr_x2 and bptt+econ were not measured. Deviation 12: DT-recall-eq runs k=1 per VM (the pre-registered rule
  needs dtr_x2; sharing an L4 already halves DT-recall's speed, and a 3 h run at x2 would pass the 5.2 h wall limit), and
  BPTT arms also run k=1 (like ECON). Parts 3-4 of Phase 1 (eval throughput at 204 px, GPU kill-and-resume test) run as
  job phase1b (gpu/phase1b.py, a copy of those parts) at the end of trader-b3-p13-8's queue, alone on its L4, instead of
  relaunching slot 1.
- 22:10Z DT-recall-eq seeds 0-3 queued on trader-b3-p13-9..12 (30456 steps = 3.0 h each, k=1), each followed by ev59,
  evT and ev99 (20n, which also records 12n); the M-ref evals ride on -12. With 7 pilot VMs up, -12 waits for a free
  VM under the 10-VM limit.
- 22:32-22:37Z operations: spot preemption wave in us-central1-a took trader-b3-p13-2, -5, -6, -7 and -8; ctl.sh relaunched
  each as -r1 from its last sha256-verified sync (checkpoints every 10 min, sync every ~5 min), losing at most one
  checkpoint interval per job. From 22:40Z mkvm.sh rotates the 10 US L4 zones by slot number so slots spread over zones
  (operations only).
- 23:03-23:17Z operations: a second spot wave (us-central1-a, -c, us-east1-d) took 9 VMs, some within minutes of creation;
  the three DT-recall-eq VMs in us-central1-b survived both waves. 23:24Z: ctl.sh allows 16 VMs per slot (was 8) and
  tries us-central1-b, europe-west4-b, europe-west1-b and europe-west4-a first (Europe is off-peak), then the rotation.
  23:26Z: at most 12 P13 VMs at once (was 10); the total VM-hours do not change, only the wall time. Slot 2 (4 VMs lost)
  keeps pil_econ_F0; pil_nocap_LB and its eval move to new slot trader-b3-p13-13; ev_mref_gate and phase1b move from -8 to
  new slot -14. Main-phase and extension trader jobs checkpoint every 300 s (was 600 s; no effect on any update).
- 23:28Z Phase 2A extension queued before the 0.5 h rule is evaluated (operations): six of the fourteen 0.5 h evals are in
  and all have v9 exact < 0.9 (best 0.837), so the extension to 1.0 h is very likely; queuing it now keeps each pilot VM
  running instead of deleting and relaunching it. The 0.5 h results are unaffected, and the extension results are used
  only if the pre-registered rule calls for them (else they are reported as unused). Because an extension resumes into
  the same --out, whose training log is cumulative, pilot_select.py --tag 0p5 reads that log truncated at the pilot's
  0.5 h iteration count (from its queue line); the truncated log equals the 0.5 h log.
- 23:36Z operations: eval_maze.py resumes after a preemption. Every 300 s (after a batch) it saves the finished sets and the
  current set's per-maze arrays and next batch offset to OUT.evck.npz (atomic); a rerun with the same arguments continues
  from there. Batches are independent, so the output equals an uninterrupted run: CPU test with kills after 1-2 batches,
  including across a set boundary, gave identical npz arrays and identical summaries (apart from wall seconds) for M-ref
  and for a Trader2D pilot (tests in /tmp, not kept). Installed on General and pushed to every live VM at 23:36Z; jobs
  already running keep the old code.
- 23:45Z decision before the gate outcome is known (operations and reporting): DT-recall-eq seeds 0-3 and the M-ref evals
  started at 22:10Z, alongside the pilots, to save wall time. If the learnability gate fails (F0-gate), no Trader2D main
  phase runs, but these jobs (about 40% done, about $3 left) finish and are reported as context: DT-recall-eq trained
  status (k of 4), its exact accuracy at 19/33/59/99 with Welch 95% CIs, and M-ref at the same sizes. With F0-gate there
  is no S2 comparison. An equal-time view on 9x9 only is reported as exploratory (not pre-registered): pilot v9 exact at
  0.5 and 1.0 L4-h against DT-recall-eq's training-log val9_acc30 at the same wall seconds (different test T: 12n vs 30).
  analyze_gate.py writes results/results_gate.json and chart_gate.json for this case; analyze.py stays the main-phase path.
- 00:33Z result of the pre-registered 0.5 h rule (pilot_select.py --tag 0p5, results/pilot_selection_0p5.json): no
  candidate in either arm reaches 9x9 val 0.9 (best 0.854), so all 14 extend to 1.0 L4-h; the extension queued at 23:28Z
  is used. 00:35Z operations: at most 13 P13 VMs at once (was 12) so slot 14 (ev_mref_gate, phase1b) starts now.
- 00:44-01:25Z operations: ev_mref_gate passed (M-ref exact 1.00 at 13 and 1.00 at 59). phase1b: eval at 204 px costs
  0.0085 s per maze-step for M-ref and 0.041 for Trader2D on an L4; the GPU kill-and-resume test passed (state checksums
  match; resumed losses within the GPU nondeterminism floor). The three M-ref evals move from slot 12 (behind DT-recall-eq
  seed 3, delayed by a preemption) to new slot 15 so they finish about 1.3 h sooner; same jobs, same arguments.
- 02:32Z result of the pre-registered 1.0 h rule (pilot_select.py --tag 1p0, results/pilot_selection_1p0.json): pools
  ECON {HF}, ECON-nocap {LB, HF, IN}; mean ranks LB 1.5 and HF 1.5, so the tie rule applies (m59 exact 0 for both; LB
  is earlier in FIXES): F-LB is selected. The learnability gate passes (ECON-nocap LB: 9x9 val 0.945, 13 dev 0.675), so
  the main phase runs. ECON-fix (LB) has frac_capped 0.0 over its last 25%, so the ECON-active ladder runs (A1, A2 on slot
  16, as coded). Because F-LB is selected, only the post-hoc comparison counts toward S1 (as pre-registered).
- 02:35Z spend plan, written before any main-phase result exists (spend cut and eval simplifications). Measured costs
  (phase1b and the pilot evals) put Trader2D evaluation far above the Phase 4 estimate: e99 on all 200 mazes to 12n is
  about 2.7 L4-h per Trader2D model (0.041 s per maze-step at 204 px, GPU-bound), m59 about 0.25 h. The full plan would
  cost about $50 against the $40 cap (which deletes every VM). At 02:30Z the ledger is $16.35; DT-recall-eq and M-ref
  need about $1.8 more and the ladder about $0.6. Estimated cost per Trader2D main run with the simplifications below:
  3.0 h training + about 2.4 h of evals + 0.2 h overhead = about 5.6 VM-h, $2.0. Decisions:
  1. The pre-registered cut order is applied now, in full: no 201 stretch, no optional arms (DTR-pair, DTR-eqflop), no
     BPTT-progcov seed 2, no BPTT-dscov seed 2, no ECON-F0 main run (S1 uses the ECON-F0 1.0 h pilot, flagged), no
     ECON-active seed 2.
  2. The order is extended (new, same direction as the pre-registered list): next BPTT-progcov seed 1, then BPTT-dscov
     seed 1. ECON-fix 4 and ECON-nocap 4 are queued now (slots 17-24, k=1: Phase-1 packing gives 2.19x per job at k=2).
     One more run is queued when the ladder decides: ECON-active seed 1 (s0) if a ladder candidate qualifies, else
     BPTT-dscov seed 1 (s0). A second one (the next survivor in reverse cut order) is queued only if, when all 8 core
     runs have finished training, the ledger plus the remaining queued work (at the measured per-job costs) plus $2.0 is
     at most $37, and it can finish, eval included, by 15:30Z. Single-seed arms have no seed-level CI: S3 or S3b is then
     reported as the single seed's value against ECON-nocap's 4-seed 95% interval, flagged as not the pre-registered test.
  3. Trader2D test evals (plan_evals.py): ev99 and evT use the first 100 mazes of each test set (e99 100 of 200, e19 100
     of 200, e33 100 of 300, e59 100 of 200); T = 12n (mult 12, or 12.1 when the arm's J option is cap, because cap's
     t_eq is 1.006 x 12n and would otherwise be clamped); no 20n extension for ECON-fix's median-dev seed. DT-recall-eq and
     M-ref (already queued on all mazes, 20n at 99) are compared with the Trader2D arms on the same first 100 mazes
     (paired primary), and their all-maze values are reported alongside. Per-seed binomial noise on 100 mazes is at most
     0.05 and is inside the pre-registered seed-level Welch interval. J-option selection (m59, 100 mazes, 3 options) and
     the dev evals are unchanged.
  4. If a preemption wave pushes the projection above $38, the next cut is the second extra run, then the first extra
     run; ECON-fix, ECON-nocap, DT-recall-eq and M-ref at 99 are never cut.
- 02:40Z S1 with the ECON-F0 main run cut: "S1 uses the F0 pilot" is implemented as ev99_pil_econ_F0, the 1.0 L4-h
  ECON-F0 pilot checkpoint (seed 100) on the same first 100 e99 mazes, J option none, T = 12n (queued last on slot 16).
  S1 compares ECON-fix's 4-seed mean with this single model, flagged (different budget: 1.0 h against 3.0 h, one seed).
- 03:20Z ladder extension rule, written before any ladder eval exists (training diagnostics seen: at 900 of 1650 steps
  lad_A1 (W0 0.5) has frac_capped 0.24 and 81 respawns; lad_A2 (Lindahl bounty) runs about 3x slower per step and has
  frac_capped 0.0 at step 300). At 0.5 L4-h no pilot of any fix reached 9x9 val 0.9 (Phase 2A), so the ladder's v9 leg at
  0.5 h can reject a candidate for a reason unrelated to the economy. Rule: if no ladder candidate qualifies at 0.5 h,
  every candidate whose cap binds (frac_capped >= 0.10 over its last 25%) is extended to 1.0 L4-h (3300 steps, resumed
  into the same --out, as in Phase 2A) and the ladder rule is re-applied to the 1.0 h results (ladder_select.py --tag
  1p0; the 0.5 h selection reads the training log truncated at 1650 steps). The lad_A1 extension is queued now on slot
  16 after ev_lad_A1 (before ev99_pil_econ_F0), so the VM does not idle; it is used only if the rule calls for it. A
  lad_A2 extension is queued only if lad_A2's cap binds by step 1650. The ECON-active main run (one seed, s0) uses the
  qualifying candidate's setting; if none qualifies at 1.0 h, the extra run is BPTT-dscov s0 (02:35Z plan).
- 05:42Z second-extra check (02:35Z plan, item 2), made when all 8 core runs finished training (05:37-05:41Z), before any
  main-phase eval exists: ledger $27.14; remaining queued work at measured costs, $0.35 per VM-h: ev59 on the 8 core VMs
  (about 0.85 h each, $2.4), slot 16 (lad_A2 end, ev_lad_A2, ev_lad_A1_x, ev99_pil_econ_F0, about 2.0 h, $0.7), Trader2D
  evT and ev99 for 8 models (about 1.8 VM-h each with boot, $5.0), the first extra run (3.0 h training + about 2.5 h of
  evals, $2.0): $37.2. Plus $2.0 is $39.2 > $37, so no second extra run is queued. The first extra run stays (projection
  $37.2 < $38, item 4). Also recorded: A2 (Lindahl bounty) runs 1650 steps like A1, which is about 1.5 L4-h of its own
  solo throughput, not 0.5 h (it is about 3x slower per step); its 0.5 h selection uses the 1650-step checkpoint as coded.
- 06:06Z ladder decision from the training logs, before ev_lad_A2 and ev_lad_A1_x exist (their v9 cannot change it, since
  a candidate needs frac_capped >= 0.10 over its last 25% AND 9x9 val >= 0.9): at 0.5 h, lad_A1 binds (0.17) but has 9x9 val
  0.295 (ev_lad_A1), and lad_A2 has frac_capped 0.0 (5 respawns), so nothing qualifies at 0.5 h; only lad_A1 binds, so only
  lad_A1 is extended (no lad_A2 extension); at 1.0 h lad_A1's frac_capped over its last 25% (steps 2475-3299) is 0.0 (its
  traders grew rich; 104 respawns in total), so nothing qualifies at 1.0 h either. Outcome: ECON-active is not run, S3 is
  recorded as "economics could not be made to bind", and the extra run is BPTT-dscov s0 (plan_main.py trader --only
  m_bds_s0, 11762 steps = 3.0 L4-h, single seed, flagged), queued now. ladder_select.py (--tag 0p5, then --tag 1p0) is
  run for the record when the two evals sync. The 8 main ECON-fix runs also never bind (frac_capped 0.0 at every log
  record after 10% of training, 0 respawns, minimum wealth above 500 against W0 = 20), so ECON-fix is inactive in every
  seed and S3 reads "economics inactive, untestable".
- 06:43Z ladder_select.py for the record (results/ladder_selection_0p5.json, _1p0.json): 0.5 h lad_A1 v9 0.295, frac 0.174;
  lad_A2 v9 0.699, frac 0.0; 1.0 h lad_A1 v9 0.864, frac 0.0; no candidate qualifies, as recorded at 06:06Z. 06:34Z Phase 4:
  plan_evals.py --vms 8 --par 1 queued evT and ev99 (first 100 mazes) for the 8 core runs on slots 26-33; J options by the
  A6 rule on m59: ECON-fix none (seed means none 0.0075, scale 0.005, cap 0.005), ECON-nocap scale (0.0, 0.0025, 0.0).
- 07:12Z projection check (02:35Z plan, item 4) after a spot wave in us-west1-b (07:03-07:05Z) took all 9 VMs there
  (slots 25-33; relaunched as -r1/-r2 in us-central1-b at 07:10Z, resuming from their last synced checkpoints): ledger
  $31.74 + evT/ev99 on 8 VMs (about 1.45 h each left, $4.1) + slot 16 ($0.2) + BPTT-dscov s0 training and ev59 (about
  3.4 h, $1.2) + its evT/ev99 ($0.6) = $37.9, not above $38, so nothing is cut. If a later wave pushes it above $38, the
  BPTT-dscov s0 run (the only extra) is cut.
- 07:45Z projection refresh, no new wave: measured eval speed (46 s per e99 maze; slots 32-33 restarted 7 min later
  than assumed) gives ledger $33.25 + 8 eval VMs (about 9.0 VM-h, $3.2) + slot 16 ($0.1) + BPTT-dscov s0 training and
  ev59 (3.1 h, $1.1) + its evT/ev99 ($0.6) = $38.2. The item-4 trigger (a wave) is not met, so nothing is cut now, but
  BPTT-dscov s0's Phase-4 evals are decided when its ev59 has synced, before queuing, against the same $38 line:
  ledger + $0.63 <= $38: evT and ev99; else ledger + $0.50 <= $38: ev99 only (S3b is at 99; evT is descriptive);
  else no Phase-4 evals (S3b not run; ECON-nocap vs BPTT-dscov reported on v9/v33/m59 from ev59, exploratory).
- 08:15Z item-4 check after a second spot wave (08:02-08:06Z: slots 25-33, all nine; relaunched as -r2/-r3 in
  us-central1-a/-c and us-east1-d by 08:12Z, resuming from synced checkpoints: evals at 81/100 or 72/100 e99 mazes,
  BPTT-dscov s0 near it 4500): ledger $34.73 + 8 eval VMs (6.3 VM-h, $2.2) + BPTT-dscov s0 training and ev59 (2.75 h,
  $1.0) = $37.9; with its evT/ev99 ($0.63) $38.5, above $38. The smallest cut in the pre-registered direction that
  restores the line is BPTT-dscov s0's Phase-4 evals, so they are dropped unless the 07:45Z check passes when its ev59
  has synced; its training and ev59 continue (ECON-nocap vs BPTT-dscov then on v9/v33/m59, exploratory; S3b not run).
  A further wave that pushes $37.9 above $38 cuts the rest of the BPTT-dscov s0 run.
- 09:08Z price check and ev59 trim. The watchdog rate ($0.35 per VM-h) is below the published g2-standard-4 spot
  price with the L4 (gcloud-compute.com, read 09:06Z: $0.4241/h in us-central1, us-west1 and us-east1, $0.3759 us-east4,
  $0.4529 europe-west1, $0.4457 europe-west4; plus about $0.014/h for the 100 GB disk). The Cloud Billing catalog API is
  not enabled in this project, so billed cost is not checked. At about $0.438/h the ledger's 105.8 VM-h so far are about
  $46, above the $40 cap in real money; the $35-$40 figures in this file are at the prescribed $0.35 rate. Discretionary
  spend is therefore trimmed: BPTT-dscov s0's ev59 keeps only J = none (v9, v33, m59); the scale and cap m59 evals, which
  only choose J for the dropped Phase-4 evals, are removed from its queue (saves about 0.5 VM-h). Its training continues
  (about 1.1 h left) because the ECON-nocap vs BPTT-dscov comparison on v9/v33/m59 is the only direct read on whether
  detaching, rather than the economy, limits the Trader2D arms (exploratory, one seed).
- 10:30Z BPTT-dscov s0 Phase-4 decision (07:45Z rule): ev59 synced 10:29Z with ledger $37.52; $37.52 + $0.63 and
  $37.52 + $0.50 are both above $38, so no Phase-4 evals (S3b not run). BPTT-dscov s0 dev results (J = none): v9 0.988,
  v33 0.00. Teardown: slot 25-r3 deleted 10:29:55Z after a sha256-verified sync; gcloud instances and disks lists for
  name~trader-b3-p13 in GCP_PROJECT both empty at 10:30:19Z; 90 VMs created, 90 deleted (gpu/vms.tsv, gpu/deleted.tsv);
  gpu/STOP set, ctl.sh exited 10:33:02Z (CTL_STOP), watchdog.sh stopped 10:33Z. Final P13 ledger $37.52 at $0.35/h
  (107.2 VM-h); about $47 at the published spot price plus disk.
