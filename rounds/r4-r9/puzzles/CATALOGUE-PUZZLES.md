# Trader-economy puzzle catalogue

One entry per puzzle experiment (P0-P12), appended by the agent that ran it; the format follows `../CATALOGUE.md`. Append only.

## P7: Can hindsight-posterior settlement train chains of latent markets without backprop?

Status: Informative

Setup:
- **Traders and settlement.** Traders hold discrete latents (K = 4, 2 units per site) in chains of depth 1, 2 and 4. Settlement is local: the hindsight posterior q(z | x, y) ∝ p(z | x)·P(y | z) settles each trader's market by cross-entropy, and no gradient crosses a trader.
- **Puzzles.**
  - Mazes: y = reachable from the start within 4 moves. Train on 9x9; test on 9x9, 15x15 and 21x21.
  - Prefix sums mod 3: y = [s == 0], with every d-th output revealed (d = 1, 2, 4). Train on length 16; test up to length 256.
- **Arms.**
  - A: end-to-end straight-through Gumbel.
  - B: literal q-form settlement.
  - Bl: likelihood-message settlement (added).
  - Bx: exact forward-backward settlement (added, prefix sums only).
  - C: random fixed encoders.
  - D and Dl: B or Bl plus a rate cost and different input views.
- **Runs.** 3 seeds, 95% t-intervals; CPU spot VMs.

Result:
- **Likelihood messages train deep chains locally.** Bl reaches maze log loss 0.059 ± 0.022 at depth 2 in 2000 steps, and 0.006 ± 0.014 in 6000 steps. Matched A is at 0.026 and 0.002. At depth 4 over 6000 steps, Bl reaches 0.040 ± 0.065 against 0.010 ± 0.032 for A (A is lower in all 3 seeds; the intervals overlap). Bl's intermediate latents align with the ground-truth reachability radius: nmi 0.66–0.80, about half of it beyond y. A reaches 0.78–0.96.
- **The literal q-form collapses.** At depth 4, B's intermediate markets hold 1 constant state (log loss 0.500 against 0.082 at depth 1).
- **The rate cost collapses D and Dl.** They sit at the marginal (0.619) at depth 2 and above. The ablations blame the rate cost (λ = 0.01 is still bad); views alone cost about 0.06 nats.
- **Prefix sums mod 3.** Only Bx learns the counter, and it generalises to length 256 in 2/3, 1/3 and 0/3 seeds at d = 1, 2 and 4. A, including a tuning sweep, B and Bl all stay at the base rate (0/3).
- **KL(q‖p) as value of computation.** It predicts where depth pays less well than label-free predictive entropy (Spearman 0.64 against 0.90). It does flag intermediate encoding errors (AUROC 0.86–0.90).
- **Cost.** Bl is about 10x slower per step than A. The whole study ran on CPU spot VMs for at most $1.72 (no GPU), and all VMs and disks are deleted.

Files: `puzzles/P7/RESULTS.md`, `puzzles/P7/{maze,prefix,common,analyze}.py`, `puzzles/P7/results/summary.json`, `puzzles/P7/charts/*.json`

## P12: Does in-loop flow BP (a stop-gradient arbitrageur that corrects a DT maze solver's edge prices) give more exact shortest paths than DT, DT with feedback and post-hoc decoders on weighted mazes 3-7x wider than training?

Status: No gain

Setup:
- **Task.** Weighted mazes: train on 100k 9x9 mazes; val at n = 25; test at n = 9, 15, 25, 33, 49, 65 and two shift sets (extra open walls, p = 0.1 / 0.5, n = 33).
- **DTxARB.** A DTNet with feedback. After each step, max-sum BP over 3-state edge variables (51-configuration cell factors) turns the proposer's logits u into prices b = u + sg(5 tanh(delta / 5)).
- **Baselines.**
  - DT, DT+fb.
  - Post-hoc on DT: DT+dec (min-cost path on -log sigma(u) + c) and DT+ARB(test) (flow BP at test time). Both tuned per seed on val.
  - ARB-kappa (flow BP with one cost on every edge).
  - CostNet+BB (learned cell costs plus an exact shortest-path solver).
- **Training.** w = 64 (the 90% 9x9 gate failed: 60%), 12k steps, 4 seeds for the main arms.
- **Primary cell.** n in {33, 49, 65}, L > 21.
- **Compute.** Spot L4 VMs, $4.88 estimated at the rate ceiling.

Result:
- **Falsified on both counts.**
  - Primary exact rate: DTxARB 2.5% [-0.5, 5.5] against 3.1% for the per-seed best baseline (DT+dec or DT+ARB(test)). Paired difference -0.6 points [-3.5, 2.3]; hierarchical bootstrap [-2.6, 1.1].
  - DTxARB outputs a valid path at n = 49 in only 1.5% of mazes (the bar was 95%).
- **In distribution the arbitrageur helps DT.** At n = 9, DTxARB is exact in 65.8% against 56.8% for DT and valid in 94% against 66%. But post-hoc decoding of plain DT does more: 73.4% exact, with no training change.
- **The arbitrageur's prices know when they are wrong.** Summed semantic loss detects wrong mazes in the primary cell with AUROC 0.92 (DT+fb: 0.59). But the prices cannot repair a proposer whose logits carry no path information past n = 33.
- **What does extrapolate: learned local costs plus an exact solver.** This is the CostNet+BB reference arm (1 seed, evaluated from its step-9000 checkpoint). It is exact in 27.0% of the primary cell and in 40.5% / 18.0% / 9.0% of mazes at n = 33 / 49 / 65, where DTxARB gets 4.6% / 0.5% / 0.2%. Given the true costs, flow BP is exact in 99% (A0-flow). For the trader question: pay for local, size-invariant signals and aggregate them with an exact clearing rule, rather than asking one recurrent trader to carry the global state.
- **Bottleneck.** The proposer's size extrapolation, not the consistency layer. Ablations (noFB, bp, DT+pen) were not completed: two replacement spot VMs were preempted within 7 minutes.

Files: `puzzles/P12/RESULTS.md`, `puzzles/P12/results/summary.json`, `puzzles/P12/charts/p12_charts.json`, `puzzles/P12/{flow,nets,train,evaluate,analyze}.py`, `GPU-LEDGER.md` (P12 section)

## P0: Before testing any market idea, how strong is a small DT-Recall solver, does one-step credit lose to full credit, does the market-resnet arm B gain survive a progressive-loss control, and how much room is there for smarter halting?

Status: Informative (verified; verifier agrees)

Result: The ruler is set and it leaves little headroom. The prefix-sum bar P1 is 8/8 at 0.9998 ± 0.0006 over 8 seeds (fL512@300). One-step credit P4 is 8/8 at 0.9948 ± 0.0068 (Fisher p = 1.0, median gap 0 pp), so on prefix sums credit horizon does not matter at ceiling. The best halting rule comes within 1 pp of the oracle at 1.01x steps: halting headroom is about 1-5% (falsifier 4 fires). Mazes need width 128: M-ref scores 1.0 on 33x33@300 and 59x59@3000, while w=64 scores 0.0 (15k steps) and 0.086 (30k).
- Verifier corrections:
  - The width result confounds width with schedule: M1-long was stopped at 30k while its probe was still rising.
  - The KL eps 3e-4 was picked on test. The validation pick, 1e-4, gives 0.994 at 1.045x.
  - "Arm B's gain is a loss artefact" is unsupported. MR-P ran 10k iterations against 3k for the pilots, and market-resnet's own equal-compute grid gives B 1.000 vs P 0.64 ± 0.82. L48 was omitted (MR-P 0.75 vs B 1.0).
- The maze credit test is inconclusive: M4 − M-prog = +0.0 / −31.6 / +0.2 pp, with every maze arm under-trained.
- The DT-L reimplementation (P-L, 0/2) is a bug, not a finding.
- Cost $6.19.

Files: [P0/RESULTS.md](P0/RESULTS.md)

## P1: Do heads that predict the value of thinking further, at several horizons, halt a frozen recurrent net earlier than simple stopping rules and a one-step gate?

Status: No gain (verified; verifier agrees)

Result: The VOC heads (R7) save 1.1% [0.0, 2.1] (L64) and 1.6% [0.0, 2.6] (L128) of steps over the one-step gate R4c; the pre-registered bar was 20%. R4c is already within about 2% of the oracle (L64: 14.4 steps vs oracle 14.1; R7 14.2 [13.0, 15.5]). Label-free, a simple consistency probe (R12TT) beats R9 by 16 pp at size 59 and 6 pp at size 99, and R9 − R7TT = −0.10 [−0.18, −0.02] at 0.5x budget (size 59).
- Validity:
  - The capacity gate failed on prefix sums (Spearman 0.46-0.63), so E-P1-1 is gate-invalid.
  - The maze capacity gate also failed (0.31-0.48). The runner did not disclose this, so E-P1-3 is gate-invalid too.
  - Maze headroom was 0.048-0.059 against a bar of 0.30.
- With FLOPs counted, R7 is only 3.7% (L64) and 6.6% (L128) under the fixed envelope.
- Cost $4.90.

Files: [P1/RESULTS.md](P1/RESULTS.md), [P1/PREREG.md](P1/PREREG.md)

## P2: Can an exactly priced quadtree market over a latent "which route" variable carry the choice between tied maze routes better than fixed canonical labels or hard-EM labels?

Status: Problem (verified; verifier agrees). If it were valid, the reading would be No gain.

Result: On n=13 tie mazes, M-tree VSP is 46.1 [29.9, 62.2] against hard-EM U-min at 64.4 [61.1, 67.7]. M-tree − U-min = −18.3 [−34.8, −1.9], and the DiD is −26.2 [−37.5, −14.9]. The latent carries I/H = 0.32 [0.22, 0.42] of the route choice (target 0.8).
- Why Problem:
  - The toy M-tile error was 6.2%, over the 5% limit, and this failure was already known before the $4.2 GPU run started.
  - The floor rule passed only at n=13.
  - The trap test covered 26 of 30 cells.
  - Every arm scores about 11% or less at n ≥ 25.
- Sensitivity run (n in {13, 19}): −14.9 [−32.0, 2.2]. The runner's −1.8 is stale.
- Cost $4.48.

Files: [P2/RESULTS.md](P2/RESULTS.md)

## P3: If latent traders are paid Shapley-type shares of the output gain, do they pick the factors the output naturally depends on rather than an arbitrary rotation?

Status: No gain (verified; verifier agrees)

Result: AS-IG pay aligned with the natural factors in 3/20 runs (Wilson 0.05-0.36), the same as additive pay (3/20): difference 0.00 [−0.23, 0.23], Holm p = 1.0. Under AS-IG the codes collapse to constants (Bayes gap 6.28 [4.75, 7.81] nats).
- The tabular market maker makes constant codes absorbing, which is a trainability confound.
- An exploration diagnostic, not pre-registered, gave Shapley 13/20 once. It did not replicate (6/20 vs 5/20, +0.05 [−0.22, 0.31]).
- Maze oracle M0, settled at $0 with no GPU stage:
  - The default is wrong on 95.8% of percolated mazes.
  - BFS earns −22% of VOI at c=6 before backtracking with a shared reader, and about +5% with per-round readers.
- Cost $1.14.

Files: [P3/RESULTS.md](P3/RESULTS.md)

## P4: Does Aumann-Shapley attribution rank latents the way exact Shapley does at lower cost, and does a market on the top-ranked latent help depth extrapolation?

Status: No gain (verified; verifier agrees)

Result: AS-SOFT reaches Spearman 0.924 [0.908, 0.938] against exact Shapley (top-1 correct 95%). Sampled Shapley at matched cost does better: 0.981 [0.976, 0.986]. AS − Shapley = −0.058 (90% CI [−0.071, −0.045]), so the equivalence test fails. AS does beat leave-one-flip on R2 AUC (0.953 vs 0.832, +0.12 [0.045, 0.198]).
- The fine-tune gives AS-mkt − F0 = −0.34 [−0.68, −0.00]. The permutation p is 0.066 and one subject drives the effect, so read it as no gain, not as harm.
- S3 is invalid: F0's seed SD is 11-32 points.
- The cost check passes 10/10 on R2 codes and 1/4 on market-resnet.
- Cost $1.16.

Files: [P4/RESULTS.md](P4/RESULTS.md)

## P5: Can an online, posted-price market for private per-cell memory slots learn maze path parity when the answer is only checked at the last step?

Status: Problem (verified; verifier agrees)

Result: The backbone gate failed, so the market arms never ran. On path parity, A1 scores 0.528 [0.497, 0.559] and A2 0.484 [0.453, 0.515], against a 0.90 bar and a validation base rate of 0.545. Accuracy is 1.0 at depth 1 and chance from depth 2 on. The dense exact-maze check for A1 is 0.000 (bar 0.95).
- The spec curriculum lengthened paths by halving the short ones, so the curriculum itself was wrong.
- Out of spec, an incremental curriculum (A1w) reaches 0.603 but is at chance beyond depth about 10.
- One seed. The phase-0 tests pass.
- Cost $0.93.

Files: [P5/RESULTS.md](P5/RESULTS.md)

## P6: Under a growing-depth curriculum, does letting traders go bankrupt by wealth and be replaced by clones beat a fixed population, matched random churn and cloning by EMA utility?

Status: Problem (unfinished, about 30% of the plan; verified from training logs; verifier agrees)

Result: This result is UNVERIFIED by the planned ladder evals. Both wealth entry (EC) and the fixed population (F) reach the L=256 cap. EC needs 125 more iterations to get there (mean 1025 vs 900; paired 95% [−50, 375]; with seed 4 added, +100 [−40, 320]), so there is no speed-up. EC had 81-87 exits, so the treatment was delivered. R0 reached 128, 32 and 256: the runner omitted seed 1, which hit 256 at iteration 2300.
- Why Problem:
  - There are no ladder evals, so L*_4 is unmeasured.
  - The DT-L reference is stuck at exact 0.0 and is probably broken.
  - There is no RESULTS.md.
- Resuming via `P6/run_split.sh` would cost about $4.5.
- Cost $1.92.

Files: no RESULTS.md; [P6/results/train](P6/results/train), `GPU-LEDGER.md` (P6 section)

Dropped designs (P0-P6 batch): none.


## P8: Should a hidden variable be paid by the output bits it saves minus the bits it costs to learn, and does that price pick useful maze intermediates better than importance alone?

Status: No gain (verified; both verifiers agree)

Result: Charging learnability made the ranking worse. Against deployed truth (32 maze intermediates, 3 seeds), Kendall tau-b is 0.33 ± 0.02 (95% CI) for the one-code price R4, 0.70 ± 0.02 for importance alone (R1) and 0.83 ± 0.05 for wrapper validation (R7b). Paired permutation p = 1.0 for R4 against each rule.
- Why R4 loses: the learnability charge L(Z|X,S) is a whole-stream cost of tens of thousands of bits for deep rungs, so R4 follows learnability (R2 0.41, R5 0.34) rather than deployed use. The best exchange rate in "importance - λ·learnability" is near 0.06, not 1.
- The binding problem is proposing the intermediates. The non-oracle pool (C2) listed nothing: all 16 level-1 prices were -17k to -54k bits, so gate F3 fired and seeds 1-4 were skipped. The oracle tower missed the BFS ladder (held-out 0.39 / 0.17 / 0.11 vs 1.00). On parity, the tower recovered the tree in 0/3 seeds; brute force did in 3/3.
- Kelvin's randomized-z market (R6) matches R4 with 2-4% of R4's CPU (final Spearman 0.97-0.99), but it matches R5 just as well, and R4's prices are the bad ones. The literal void-on-mismatch market (R6b) loses effective samples (4,088 down to about 115) and cannot rank (Spearman 0.05-0.52 with R4).
- Verifier caveats:
  - R7a and R7b are partly circular, so the clean comparison is R1 0.70 vs R4 0.33.
  - The free-latent control makes 31/32 items negative, and the truth metric rewards a noisy copy (nz05 -3.73 vs g2 -6.71 bits/maze).
  - The shortcut failure F1' (m2 priced at +12k bits, truth -6.5 to -8.4) is weak: m2's gain over base (+14.3) is close to g2's (+15.7).
  - On items with R4 > 0, R4 roughly ties R1 and picks the true top item in 3/3 seeds.
  - The tower check (U6) is a tautology. Earlier single-run files were reused without disclosure.
- Cost $2.01.

Files: [P8/RESULTS.md](P8/RESULTS.md)

## P9: If we randomly clamp a hidden variable and void the trades that bet on the other value (futarchy on z), do the payouts measure how much it matters, and does paying latents that way train better ones?

Status: Informative (verified; both verifiers agree)

Result: It measures correctly only on exogenous variables, and only after a retune. On the net's own latents it measures nothing useful and does not help training. On exogenous doors (P9a, retuned, 3 seeds), KL to truth is 0.016 [0.006, 0.027], VOI Spearman 0.911 and sign accuracy 1.0. On the net's own binary latents (P9b-E, 9x9), clamp pay finds must-pass cells with AUROC 0.452 against 0.763 for a free local heuristic: difference -0.311 [-0.345, -0.277]. At 15x15 it is -0.276 [-0.349, -0.214].
- Theory (D1/D2): clamp-and-void equals reveal-true plus importance-weighting variance. On the net's own latents it reduces to self-reveal teacher forcing.
- The pre-registered P9a recipe missed every bar (KL 0.272), and the retune saw the test numbers. Equivalence to plain reveal-true is not shown (TOST +0.0024, 90% CI [-0.0079, 0.0126]).
- Training with clamp pay (P9b-T, 33x33 exact at T = 150): N4a 3.2% vs N1c 0.8%, +2.4 [-3.7, 8.5] points; vs N0 -9.5 [-28.7, 9.8]. This is a null, and a weak one, because 33x33 is at the floor.
- Problems:
  - The A/A check failed: seed 0 differs by -0.097 against a bar of 0.02, and a same-seed rerun differs by 0.041.
  - The 25x25 gate reached 0.079 and seed σ is 12.9 points, so the test is underpowered.
  - The gate seeds were the retune's selection runs.
  - Allocation at 33x33 is at the floor (+1.6 [-5.9, 9.2]).
- Verifiers: every net-internal estimator (BS, LW, FCI, AtP) is near chance against the heuristic. The right claim is "clamp pay does not recover task importance".
- Cost $2.2.

Files: [P9/RESULTS.md](P9/RESULTS.md)

## P10: If masked input cells are treated as hidden variables, can a value head pick which cell to reveal next, and can an auction with fees keep only the learned events that pay their way?

Status: Informative (verified; both verifiers agree)

Result: No on both parts.
- Part A (Sudoku stalls): the value head's within-state rank correlation with the realised gain is 0.026 [-0.096, 0.149], below simply counting masked peers (0.185). The difference is -0.159 [-0.178, -0.138] against a +0.1 bar, so gate G1 failed and branching was never built.
- Part B (learned events M2, 6x6, 5 seeds): the planted positive was de-listed in 5/5 seeds and every learned slot collapsed to a constant (NMI 0). Auction minus the reg2 control is +0.104 [-0.104, 0.313] nats/cell.
- Post hoc, a fixed cell bit [y_14 <= 3] survived the auction in 4/4 seeds, with royalty G +0.050 [0.033, 0.067] against a fee of 0.005. The verifiers found that royalty measures decoder reliance, not information value: the bit made the NLL bound worse in 3/4 seeds (+0.092), and it is a coarse copy of an output cell.
- Readouts:
  - Void-rule importance weights have 47x the variance of the signal with the greedy proposal (about -49% Huber bias), and 4.1x with a mixture proposal.
  - ARDM joint NLL is 1.96 / 1.27 / 1.03 / 1.016 at 8 / 16 / 32 / 48 passes, against an entropy of 1.007. A known method reaches the minimal-Bayes frontier.
- Caveats:
  - 9/15 encoders are already constant at init (about 51% by simulation), so the collapse may be an init or straight-through artifact rather than the auction.
  - The G1 target is noisy (rho(G, I*) = -0.009).
  - No bank atom was a true positive.
- Cost $1.26.

Files: [P10/RESULTS.md](P10/RESULTS.md)

## P11: Does capping each pixel's wealth with an LMSR worst-case-loss budget act as a depth trust region, so a recurrent maze solver can think longer without overthinking, better than a total-variation cap and known fixes?

Status: Informative (verified; both verifiers agree). The runner's verdict is "falsified (F2)", which is the mixed outcome "LMSR beats TV, but budgets lose to known fixes". A strict gate reading would make it inconclusive.

Result: The LMSR shape beats the TV shape only by hurting less. Budgets lose to known fixes. Exact-maze accuracy, mean of E1-E3 over 5 seeds:
- M-budget beats M-TV by +7.9 [+5.0, +10.9] points in 5/5 seeds.
- M-budget ties its uncapped twin M-B: -1.6 [-9.0, +5.9].
- M-budget loses to DT-L by -13.4 [-28.9, +2.1] and to R-prog (trained seeds) by -58.0 [-70.5, -45.5].
- Arm means: R-prog 53.9 (89.8 on its 3 trained seeds), DT-L 44.4, M-B 32.6, M-budget 31.0, M-TV 23.1.
- As test-time trackers on R-prog, LMSR beats TV by +8.2 [+1.5, +15.0], but both score below the plain readout (39.2 and 31.0 vs 53.9).
- Late corrections: 3.5% vs 0.4%, +3.1 [+0.7, +5.5].
- 98-99% of on-path pixels exhaust the budget, so it works as a hard stop, not a trust region.
- Problems:
  - The sensitivity gate failed: there was no overthinking to fix (R-final seed 2 scored 100/100/100), and 3 of 8 recall seeds never trained.
  - Both tuning grids stopped at their top edge, so the LMSR-over-TV result may be a grid artifact.
  - M-B also plateaus by 3x, so the headline mechanism is overstated.
  - R-prog seed 3 had non-finite pixels.
  - The cost-cap rate was switched mid-run.
- Cost $6.80 at $0.45/h, or $10.74 at the $0.71/h ceiling, over the $8 cap.

Files: [P11/RESULTS.md](P11/RESULTS.md)

## P12 (verification note): the entry above stands as No gain (verified; both verifiers agree)

Corrections to the entry above:
- DTxARB seed 0 diverged. Without it the paired difference is about +0.25 points, still null. Training was unstable in every arm.
- The design's minimum detectable effect was misstated. The true 80%-power MDE is about 22-25 points.
- The arbitrageur ran τ = 0.3 sum-product BP, not max-sum. DTxARB had 0 extra channels, not 8.
- CostNet+BB was trained at w = 32, not w = 64. It is still one seed.
- AUROC 0.92 is inflated by maze size: within n = 33 it is about 0.86, and maze size alone gives 0.70-0.76.
- A0-flow (flow BP is exact given true costs) was shown only at n <= 15.
- The L > 21 filter removes nothing.
- The gate failed (DT 60% vs 90%).

Files: [P12/RESULTS.md](P12/RESULTS.md)

Dropped designs (P8-P12 batch): none.

## P17: At equal compute (NFE), does a market that prices each next restart or extra depth per puzzle, from traders paid on checkable signals (constraint validity, residual), beat EqR's smallest-residual selection and majority vote on Sudoku-Extreme?

Status: Checker gain, not a market gain

Setup:
- **Solver.** The released EqR Sudoku-Extreme checkpoint (Huang, Geng and Kolter, ICML 2026; locuslab/EqR-model; code and data Apache-2.0, the model card names no licence).
  - Path noise 0.01, set by PREREG amendment 1 with the paper's Section 5.2, Tables 12-13 and the repo defaults cited, before any market or test data.
  - Init std 1.0, lambda 0.95.
- **Data.** Test set locuslab/EqR-data (422,786 puzzles); val = 1,024 sampled puzzles, test = 2,048.
- **Bank.** 128 restarts x a staircase of depths (8,192 NFE per puzzle); every policy is replayed from it with exact NFE accounting.
- **Market (M-chk).** Actions are SPAWN a restart or EXTEND a candidate 2x.
  - Prices come from Kelly-weighted pools of logistic traders on checkable signals.
  - Clearing is an exact greedy value-density rule under a pooled budget (256 x b per block).
- **Baselines at equal NFE.** SR, MV, fixed depth, fixed restarts, random split, SR+chk, SEQ-chk (restart until valid), RT.
- **Kill rule (b = 64).** The paired-bootstrap CI lower bound must be > 0 against both SR and MV. A secondary test against the best verifier-aware simple rule (SEQ-chk, chosen on val) decides "market gain" vs "checker gain".

Result:
- **The kill rule passes.** At b = 64 on 2,048 test puzzles, M-chk scores 98.54% against SR 92.72% and MV 92.38%. The paired differences are +5.81 [+4.83, +6.84] and +6.15 [+5.13, +7.23].
- **The secondary test fails.** Restart-until-valid (SEQ-chk, depth 16) scores 98.49%, so the market's edge is +0.05 [0.00, +0.15]: 1 puzzle won, 0 lost. No mazes.
- **The market learned "restart, don't deepen".** 93% of its actions were spawns, and the extends (13% of NFE) produced one valid grid.
  - Without pooling, its choices are identical to restart-until-valid, puzzle by puzzle, at every budget.
  - The gain is the checker (+0.7 over SR) plus block-level budget pooling (+5.1), not pricing.
  - The single-trader and wealth-pool versions tie.
- **Verifier-free (reported only).** M-res beats SR and MV at b = 32-256, but not its simple residual-threshold rule RT (-0.05 at b = 64, -1.56 at b >= 128).
- **Headroom.** Depth alone reaches 97.4% (noise 0.01) or 99.0% (noise 0.5) at 1,024 steps. Budgets 16-128, where restarts matter, are included.
  - Above b = 128 the bank is the limit: 98.68% of test puzzles have some restart that becomes valid within its cap.
- **Cost.** $10.76 of the $30 cap at the ceiling (15.15 VM-h, 7 spot preemptions resumed). All VMs and disks were deleted, and gcloud list confirms it.

Files: [P17-breadth-depth/RESULTS.md](P17-breadth-depth/RESULTS.md), `P17-breadth-depth/charts/p17_charts.json`, `P17-breadth-depth/results/summary.json`, `P17-breadth-depth/PREREG.md`

## P14: Does value-priced event publishing (LEG training of publisher slots plus a LOCO listing market) find a sufficient single-bit maze code more often, or with fewer FLOPs, than the best of five tuned sparse-backprop baselines at matched FLOPs?

Status: No gain

Setup:
- **Task.** T-reach on 9x9 mazes: Y[c] = [dist(start, c) <= 4]. The sufficient single-bit codes are known (g2 = [dist <= 2], d02, ring, d12 and complements; gate g6).
- **V.** k = 4 publisher slots trained by the local-expectation gradient (LEG, exact flip values through the reader). A listing market sits on top: prequential LOCO value of information against a fee, grace 500 steps, one delisting per check, respawn with reader-row and Adam reset.
- **Baselines.**
  - Tuned sparse backprop: ST+L1, ST+L0 hard-concrete gates, ST+L1+listing, DisARM+L1, ST-Gumbel+L1.
  - BP* = BP-L0-cm, chosen and frozen before eval.
- **Attribution.** Two 2x2 factorials (estimator x listing; estimator x penalty type), 20 seeds per cell.
- **Money and valuation arms.** Wealth survival (V-W), planted analytic slots under LOCO and under wealth (P-LOCO, P-W), and floors (N-sel, N0).
- **Pre-registration.** Every arm trained to V's counted-FLOP budget, 1.93e12. PREREG frozen 22:03Z, sha256 001137d1...; 290 eval runs.
- **Compute.** CPU spot VMs.

Result:
- **Falsified: V is clearly worse.**
  - V ends with a sufficient single bit (SUFF1) in 3/20 seeds, against 17/20 for BP-L0-cm. Difference -0.70 [-0.84, -0.40].
  - Median gap_1 is 0.042 against 0.0005.
  - S1 (log-rank on FLOPs to SUFF1) and S2 (against the same listing market with straight-through training, 8/20) both fail.
  - F1, F2, F3 (9x9 non-inferiority) and F5 fire; F4 (collapse, 0/20) and F6 do not.
- **The only significant attribution runs against LEG.** The estimator main effect is significant (van Elteren z = +3.94, Holm p = 0.0002). The listing (p = 0.085, Holm 0.17) and penalty-type (p = 0.92) effects are not.
- **Mechanism: churn with resets.** V delists about 16 slots per 6000-step run. Each respawn resets a reader row, so V's output log loss is 0.040 against 0.003 without listing. The same churn barely hurts straight-through.
- **The regulator values planted slots well (S4 passes).** P-LOCO keeps exactly one g2 copy in 10/10 seeds; its VoI tracks the Shapley value with Spearman 0.70. But on trained slots it made harmful delistings in 9/20 V runs.
- **Money is not a substitute (S5 fails).** P-W kills the worthless slots but keeps redundant ones, so it fails minimality in 10/10. The report therefore says "LEG plus LOCO pruning", not "economy".
- **T-dist (descriptive).** No arm found a thermometer code. V's joint codes are insufficient in 6/10 seeds; the backprop arms reach joint-probe accuracy 0.997-0.998.
- **Scope.** P14 cannot bear on the Bansal (recall plus progressive loss) programme falsifier: it has no depth or extrapolation axis, and its 21x21 test only checks translation equivariance.
- **Cost.** $7.69 at the ceiling rate (6.4 VM-h; one spot preemption, resumed). All VMs and disks are deleted, and gcloud list confirms it.

Files: [P14/RESULTS.md](P14/RESULTS.md), `P14/PREREG.md`, `P14/results/summary.json`, `P14/results/summary_tdist.json`, `P14/charts/*.json`, `P14/{p14,stats14,tdist_stats}.py`, `GPU-LEDGER.md` (P14 section)

## FU-B2-1: Does P12's CostNet+BB lead (learned local cell costs + an exact blackbox-differentiated solver, 27% exact on 3-7x wider mazes, one seed) hold over many seeds, and does it match the official DT-Recall on the Bansal et al. easy-to-hard mazes?

Status: Informative (verified; verifier agrees). 1a replicates; 1b is worse than the released DT-Recall.

Setup:
- **Prior work, named.** CostNet+BB is Vlastelica et al., "Differentiation of Blackbox Combinatorial Solvers", ICLR 2020 (arXiv 1912.02175). The "local traders price cells, exact clearing" reading is a relabelling of that method: no market mechanism was built or tested.
- **1a.** P12's recipe verbatim (w = 32, 12k steps, λ = 20) over 32 seeds, a number set by a power calculation written before the run. Primary: pooled exact on n = 33/49/65 (400 mazes). Gates G1 (oracle ≥ 0.99) and G2 (median 9×9 ≥ 0.60) are applied automatically.
- **1b.** Official maze_data_train_9 and the e2h_maze test sets, 20 seeds, official exact rule. Two fixes were chosen on train/val before any test:
  - perturbed costs are clamped at 1e-3, not 0;
  - costs are normalised per maze.
- **1b primary.** e2h_maze_59 against the released DT-Recall checkpoint as evaluated by dt-baseline, with ±5-point bars.
- **Pre-registration.** PREREG sha256s were frozen at 23:14Z (1a) and 23:24Z (1b).

Result:
- **1a replicates. P12's 27% was typical.**
  - Primary exact is 28.66% [28.19, 29.14] (SD 1.3), against 1.25% for uniform costs and 100% for the oracle.
  - 26 of 32 seeds reach at least 27% at step 9000.
  - The only DT it beats is P12's, which failed its validity gate (labelled bar).
  - Exact still falls with size: 78.9 / 41.2 / 20.7 / 11.7% at n = 9 / 33 / 49 / 65.
- **1b: worse (pre-registered primary).** On 59×59, CostNet+BB solves **25.05% [23.56, 26.53]** exactly; the released DT-Recall checkpoint, evaluated in-house by dt-baseline (lane trader-dt-p1, T = 1000), solves **100%** (peak at iteration 726, final also 100%). The upper bound is far below the DT − 5 bar of 95. The DT validity gate passed (13×13: 100%).
  - Exact by set, CostNet+BB against DT: 99.84% [99.77, 99.91] against 100% on 13×13, 58.7% [55.5, 62.0] against 100% on 33×33, and 2.0% on 201×201 (DT eval not landed at 02:50Z on 09-30).
  - Paper context: Bansal et al. 2022 report DT-Recall at 97.3% on 59×59 (Table 4) and 74% on 201×201 (text); the in-house reproduction is at or above the paper.
- **Why it fails at size (exploratory).**
  - The net prices walls at about 200–360× open cells (per-seed medians). Once the true detour is longer than that, cutting through a wall is cheaper.
  - Exact on 59×59 falls from 100% for paths under 100 blocks to 0% above 500.
  - An exact solver does not extrapolate "by construction" when walls are in the priced graph.
- **Defect found.** P12's blackbox backward clamps perturbed costs at 0, which makes the solver's walk-back cycle. It failed on up to 7 of 640 backward solves in P12's own run (`clamp0_diag.txt`). 1a keeps the bug; 1b fixes it.
- **Cost.** $0.68 at the ceiling (c2d-highcpu-56 spot, 0.33 h). The VM and disk are deleted, and gcloud confirms it.

Files: [followups-batch2/item1-costnet/RESULTS.md](followups-batch2/item1-costnet/RESULTS.md), `followups-batch2/item1-costnet/{PREREG.md,PREREG-1b.md,analyze1.py,results1.json,why1b_exploratory.json,prefreeze_pilots/}`, `dt-baseline/results/vm/trader-dt-p1/eval/released_maze/`, `GPU-LEDGER.md` (batch-2 follow-ups section)

## FU-B2-2: Do P10's learned events still collapse to constants with a non-degenerate encoder init, a rate warm-up, or both?

Status: No gain (pre-registered; all gates passed; verified, verifier agrees)

Setup: P10's M2 learned-event arm (S6 6x6 puzzles, 3 learned encoders + 5 reference slots, 20000 steps, CPU), rerun with a centred (data-dependent) encoder init, a 4000-step rate warm-up, or both; cvae arm on 10 seeds (power calculation), auction arm on 5. Primary: share of learned slots non-constant on test at the end, paired sign-flip test, Holm over 3. Gates G0 (centred init non-constant), G1 (A/A with P10), G2 (reg2 NLL) enforced by the driver. PREREG frozen at 23:11Z on 09-29.

Result: Yes, they still collapse. The collapse is not an init or rate artifact.
- Primary (the cvae arm, 10 paired seeds): the share of the 3 learned slots that are non-constant on test at the end.
  - Original init: 0/30.
  - Centred init: 0/30.
  - Rate warm-up (no rate gradient into the encoder for 4000 steps): 1/30, diff +0.033 [-0.042, 0.109].
  - Both fixes: 1/30, the same diff.
  - Holm p = 1.0 for all three fixes. The 95% upper bounds on the per-slot survival rate are 0.095 for 0/30 and 0.149 for 1/30.
- The centred init does its job: 30/30 slots are non-constant at step 0, against 9/15 constant with P10's init. The slots are then constant again within 50-100 steps (median). Slots that started non-constant went constant before any rate gradient in 10/10 both-fixes seeds, 5/5 auction_both seeds and 6/10 warm-only seeds (the other 4 were constant from step 0). So the straight-through NLL gradient alone is enough to cause the collapse.
- The one cvae survivor (seed 0, slot 0, in both warm arms) is the non-cell atom `bl_row(0,4,0)` exactly (NMI 1.00), with G 0.083 / 0.060 against a fee of 0.005.
- Secondary, the auction with both fixes (5 seeds): 0.73 [0.39, 1.08] of learned slots are non-constant at the end, with G > fee in 5/5 seeds. The original auction has 0.00. Post hoc, only slots spawned at step 14000 survive (10/15, against 0/15 for slots spawned at steps 0 and 7000). Those slots had a low cosine LR (4.1e-4, falling to 0), a trained decoder and only 6000 steps of age.
- The A/A replication of P10 (seeds 0-4, CPU instead of GPU) shows no clear difference from P10 (not an equivalence test). The NLL differences are cvae -0.020 [-0.170, 0.129], auction -0.007 [-0.102, 0.087] and reg2 +0.001 [-0.097, 0.100].
- Caveats:
  - The pre-registered "aged" auction secondary is undefined (n = 0). Every auction_both slot failed its check at 7000 and again at 14000 and was respawned, so none was old enough.
  - The step-14000 survival is post hoc and confounded, so it is not evidence that any fix works.
  - G can exceed the fee for a constant slot (a resampling artifact), so "G > fee" alone overstates usefulness.
  - NLL is underpowered (n = 5-10).
  - Untested next step: a low encoder LR from step 0, or a damped straight-through gradient, with every slot scored after a full rate phase.
- Cost $3.02 at the ceiling rate (2.51 VM-hours on a c2d-highcpu-32 at $1.20/h). The VM and disk were deleted; the instances and disks lists showed 0 items at 01:28Z on 09-30.

Files: [followups-batch2/item2-p10/RESULTS.md](followups-batch2/item2-p10/RESULTS.md), `followups-batch2/item2-p10/{PREREG.md,analyze.py,posthoc.py,results/summary.txt,results/posthoc.txt}`, `GPU-LEDGER.md` (batch-2 follow-ups section)

## FU-B2-3: With 4× the data (N = 4096) and a gain-over-base truth, does charging learnability (R4, or R8 with λ fixed at 0.0063) rank P8's 32 maze intermediates better than importance alone (R1)?

Status: Problem (a pre-registered gate failed; verified by an independent recomputation)

Setup: P8's maze task and 32 items, price stream and truth-train bank at N = 4096 (C2's baselines: B2 0.95, B3 0.99). Truth = gain over base of the deployed Y net (bits/maze), mean of 2 training replicates. Prices = mean of 8 prequential replicates. Primary: d = τ_b(R) − τ_b(R1) for R4 and R8, mean over 5 seeds (power calculation) with 97.5% t-CIs. Gates G1 (B2 ≥ 0.90), G2 (truth A/A ≥ 0.80), G3 (price noise ratio ≤ 0.10, random/null checks) enforced by the driver. PREREG frozen at 23:19Z on 09-29; CPU only.

Result: No verdict. Gate G3 on seed 2 gave a price noise ratio of 0.1058, above the 0.10 bar, and the run stopped itself at 01:48Z. Only seeds 0-1 pass every gate, against the minimum of 3.
- Gates:
  - G1 passed on all 5 seeds (B2 held-out accuracy 0.94-0.98).
  - G2 passed on all 5 (truth A/A Spearman 0.97-0.99).
  - G3 passed on seeds 0-1 (0.073, 0.057) and failed on seed 2. The failure is marginal: the frozen rule uses the sample sd (ddof = 1); with ddof = 0 it would be 0.099.
- Descriptive only (seeds 0-2, including the failed seed; not a verdict):
  - R4 − R1 = −0.515 [−0.611, −0.419] and R8 − R1 = −0.008 [−0.060, +0.044] (97.5% CIs).
  - τ_b is R1 0.770, R4 0.255 and R8 0.762.
  - R4 stays about 0.5 τ below R1 in each of the 3 priced seeds, as at N = 1024. On the 2 gate-passing seeds alone the CIs are wide (R4 − R1 −0.508 [−1.124, +0.108]).
  - Noisy-copy check: nz05 − g2 = −3.96 [−6.97, −0.95] bits/maze (the truth still favours the noisy copy of Y); on the 2 gate-passing seeds −3.59 [−16.7, +9.5], a tie.
  - Exploratory: P8's N = 1024 wrapper prices (R7b) rank the N = 4096 truth at τ 0.94 [0.89, 1.00]. R7b was not run at N = 4096.
- Deviation (DEVIATION-1): mid-run, the driver was swapped to price seeds 0-3 only and the deadline moved from 02:20Z to 02:35Z, because replicates took longer than the time model. It was decided before any price or truth number was read (the G1 results had been seen). It cut the planned seeds from 5 to 4 but changed no gate, metric, bar or analysis code.
- To get a verdict: rerun with KREP = 12 (P8's count). The observed ratios (0.057-0.106 at 8 replicates) scale to about 0.05-0.09 at 12.
- Cost $3.71 at the ceiling rate (3.09 VM-hours on c2d-highcpu-32 at $1.20/h). The VMs and disks are deleted, and gcloud confirms it.

Files: [followups-batch2/item3-p8/RESULTS.md](followups-batch2/item3-p8/RESULTS.md), `followups-batch2/item3-p8/{PREREG.md,DEVIATION-1.md,code/analyze3.py,vmout/GATE_FAILED.json,vmout/item3_summary.json,vmout/STATUS.txt}`, `GPU-LEDGER.md` (batch-2 follow-ups section)

## P19: Does a fast reader's prequential gain rank candidate maze intermediates as well as the wrapper (Kendall tau ≥ 0.70) at ≤ 5% of its compute, and better than the first-order price against the current reader?

Status: Killed (K1 and K2) (verified)

Setup:
- **Task.** THEORY-3 §3 option 3 and its "Cheapest decisive experiment", run on P8's maze ladder (11×11, N = 1024, contexts empty and g2g4). There are 44 ranked items: P8's pool plus XOR-type candidates (x2, x4, x6 = g_k XOR key; xb4), pg4 and fg4.
- **Truth.** The deployed gain over base (P8/item3 truthrep), 2 inits × 2 replicates.
- **Rules.**
  - FR: the fast reader. A warm-start copy of the current reader with a zero-initialised channel for z; prequential, 64 chunks, Adam 3e-3, 2 stream orders.
  - FO: a first-order ridge-logistic offset on the current reader's logit.
  - R4 (P8's price), R1 (importance) and R7b (P8's wrapper).
- **Pre-registration.** PREREG frozen by sha256 at 01:30Z. LR and (K, T) were fixed by a truth-free pilot on seed 99. Test seeds 0 to 4.
- **Kill rules.** K1: tau_FR < 0.70 or FR compute > 5% of one wrapper replicate. K2: tau_FR − tau_FO < 0.05.
- **Deviations.** Three, all dated and none touching the verdict inputs or a kill rule (the first lowers the comparator replicates): fewer comparator replicates for cost, a JSON bug fix, and a spot preemption rerun on demand.

Result:
- **Killed, K1 and K2.**
  - FR tau is 0.576 ± 0.051 at 0.040 of a wrapper replicate.
  - FO is 0.530 ± 0.040 at 0.024.
  - FR − FO paired is +0.046 ± 0.035: real, but under the 0.05 margin.
  - Gates G1 to G3 passed on 5/5 seeds; truth A/A Spearman was 0.91 to 0.95.
- **Comparators.** R7b 0.866 ± 0.037 with 4 replicates (0.836 with 1). Importance R1 0.674 ± 0.033, at about 27× FR's compute. P8's R4 0.293 ± 0.036.
- **XOR items.** FO prices all six at about 0, as THEORY-3 predicts. FR does not fix this: it gives about 0 or less on x4, x6 and g2g4:x6, whose truth is 11 to 14 bits per maze (g4 is 21). Mean rank error: R7b 2.5, FR 9.5, FO 11.5.
- **FR is not invariant to relabeling.** fg4 = 1 − g4 gets 0.53× g4's FR, while FO is exactly invariant and the truth nearly so (gap −0.04).
- **More FR orders do not help** (1, 2 and 4 orders give 0.573 to 0.577). Longer stream prefixes do: T = 16, 32 and 64 give 0.505, 0.549 and 0.577, but T = 64 is the whole stream.
- **Exploratory: FR, FO and R1 share their misrankings; R7b does not.** On g3, g6, reach and nz, the three true-z rules agree with each other. A hypothesis follows: the gap comes from pricing the true z while the truth deploys ẑ, not from a slow reader. The next test is FR or FO on ẑ.
- **Cost.** $2.85 at the ceiling (2.38 VM-h; one spot preemption). Both VMs and disks are deleted, and gcloud list confirms it.

Files: [P19-fast-reader/RESULTS.md](P19-fast-reader/RESULTS.md), `P19-fast-reader/PREREG.md`, `P19-fast-reader/DEVIATIONS.md`, `P19-fast-reader/results/p19_summary.json`, `P19-fast-reader/results/p19_chart.json`, `GPU-LEDGER.md` (P19 section)

## P16: Can a meta-trained in-context reader value a newly listed bit (its prequential gain V^preq_k) better than prior-free Bayes, CI betting, SGD readers, direct CMI regression and TabPFN v2, on bit types it never saw?

Status: Problem (inconclusive, undertrained: the pre-registered precondition was not met)

Setup:
- **Scope.** P16 tests a valuation tool. It tests no economic mechanism: a PASS would have supported a valuation tool only. P16 does not bear on the Bansal falsifier.
- **Episodes.** Each episode has a context of B (|B| = 2, 6 or 12) and y, plus one listed bit b; the value is read at k = 4 to 128 rows.
  - 20 generator types (8 held out).
  - Test: 6000 episodes. The primary slice is 1200 held-out positives against 1200 held-out negatives.
- **Arms.**
  - FR: an axial transformer (1.46M parameters), meta-trained to 1e16 FLOPs, 3 seeds. Also FR-small (1e15) and FR-direct (CMI regression, 2 seeds).
  - C1 (KT counts), C4 (tuned sparse-parent BMA, forward gain + reverse CI term), C5 (CI betting), S1 and S2 (SGD readers).
  - E1: TabPFN v2 on a paired subsample.
  - L-train and L-all: library matchers.
  - An adversarial slice of about 95 null candidates per episode.
- **Pre-registration.** PREREG frozen by sha256, with Addendum A (gate) and Addendum B (exploratory compute run).
  - Precondition: FR converged and ID AUROC at k = 16 >= 0.80.
  - PASS: (a) AUROC(FR) − max(C1, C4, C5, S1, S2, E1) >= 0.05 on the held-out primary slice at k = 16, plus (b) and (c), with falsifiers F1-F4 and F6 silent.
- **Simplifications** (stated in PREREG):
  - Dropped C3, S3, FR-fact and CRT debiasing, so F5 is untestable.
  - F_meta 1e16; 6000 test episodes, not 18k.
  - E1 on a subsample; a reduced adversarial slice; stochastic baselines at one seed each.
- **Compute.** L4 spot VMs, 13 in all, 10 preempted; $9.82 estimated at the rate ceiling (cap $15).

Result:
- **Precondition fails.** FR converged (validation NLL improved 0.29-0.35% over the last 20% of steps), but its ID AUROC at k = 16 is 0.711 (seeds 0.719, 0.719, 0.696), under the 0.80 bar. FR-small is 0.067 lower, so the run is compute-limited. The pre-registered verdict is "inconclusive (undertrained)".
- **Read past the precondition, FR loses to C4.**
  - Held-out primary AUROC at k = 16: FR 0.705, C4 0.745, C5 0.649, C1 0.625, S2 0.592, S1 0.571, FR-direct 0.560. E1 is 0.599 on its subsample.
  - (a): FR − C4 = −0.040 [−0.061, −0.020], negative for every seed; (b) and (c) fail too.
  - F1 (prior-free Bayes suffices) fires.
  - F4 clause 1 fires: FR's FPR at 80% TPR on held-out nulls is 0.63 (C4 0.54).
  - F2, F3 and F6 do not fire. F6 is weak: FR-direct had not converged.
  - At higher k the two are level: primary k = 64 is 0.884 against 0.885.
- **FR's value is a price in the right units; C4's is not.**
  - Regressed on the realised next-16-row gain, FR's V has slope 1.05-1.10 (R² 0.41-0.45); C4's has slope 0.27.
  - Ranked by that realised gain, FR's primary AUROC is 0.775 against C4's 0.711.
- **Winner's curse.** On null bits, FR's top pick among about 95 candidates settles at −6.14 nats [−9.55, −3.77], against −0.83 for the average candidate. C4's pick settles at −0.95.
- **Exploratory (post hoc).**
  - **4x compute.** FR seed 0 at 4e16 FLOPs reaches only ID 0.732 and primary 0.715, about +0.022 per decade after +0.075 from 1e15 to 1e16.
  - **C4's edge comes from its reverse CI term.** C4's forward gain alone scores 0.660 on primary, below FR's 0.705; the reverse term alone scores 0.738.
  - **FR adds information.** Rank-averaging FR with C4's reverse term gives 0.758 at k = 16 and 0.908 at k = 64 (C4: 0.745 and 0.885).
  - Hypothesis for a new PREREG: a reader that also amortizes the reverse direction.
- **Cost.** FR and C4 each take 0.061 s per episode; TabPFN about 36 s. All VMs and disks are deleted, and the gcloud lists confirm it. Final weights are archived at `/home/agent/data/archive/p16-weights/`.

Files: `puzzles/P16/RESULTS.md`, `puzzles/P16/PREREG.md`, `puzzles/P16/{gen,oracle,reader,train,baselines,larms,e1,eval,adv,analyze,analyze_final,analyze_x}.py`, `puzzles/P16/results/summary.json`, `puzzles/P16/results/explore_compute.json`, `puzzles/P16/charts/*.json`, `GPU-LEDGER.md` (P16 section)

## P18: On mazes, does training each recurrent step on its own loss from a detached state, with blocks drawn in random order, extrapolate from 9x9 to 59x59 better than DT-Recall with the progressive loss at equal compute?

Status: Falsified (verified; verifier agrees, corrections applied)

Setup:
- **Source.** THEORY-3 §5 item 1. On prefix sums (market-resnet §17.2), detached one-step training in random order (`tb1_rand`) extrapolated 100x at 482/600, against 71/600 in round-robin order. Full BPTT in random order reached 449/600, so order, not detachment, carried it; DT-Recall scored 600/600 at equal FLOPs.
- **Arms.** Width 128, trained on official 9x9 mazes with the official maze recipe (no gradient clipping), each run at 1.7 L4-hours:
  - A1: K = 4 untied DT-Recall blocks, detached per-step loss, random block order.
  - A2: as A1, with round-robin order.
  - A3: as A1, with full BPTT.
  - A4: DT-Recall with the progressive loss, alpha 0, 0.5 and 1. A4\* is picked on dev m59; it was alpha 1.
  - A5 (K = 1): not run, by the spend rule.
- **Metric and test.** Exact maze accuracy at T = 12n on held-out official test slices e19, e33 and e59 (200, 300 and 200 mazes). The primary is a Welch one-sided test, 5 seeds against 5.
- **Pre-registration.** PREREG frozen 01:23:48Z (sha256 in `PREREG.sha256`); unit tests a-f passed; M-ref gate passed.

Result:
- **Falsified: A1 is clearly worse.** On 59x59, A1 solves 0.017 of mazes and A4\* (alpha 1) 0.081. d59 = -0.064, one-sided 95% bounds [-0.111, -0.017]; d33 = -0.149.
- **By size, A1 against A4\*:** 0.841 / 0.301 / 0.017 against 0.885 / 0.450 / 0.081 on 19 / 33 / 59. Both trained 5/5.
- **Order helps only at moderate size.** A1 - A2 is +0.25 [0.08, 0.41] on 19x19, and about 0 on 59x59.
- **Detachment helps trainability, not extrapolation.** Full BPTT through four untied blocks did not train in 1.7 h (0/3, v9 0.53).
- **Alpha 0 and 0.5 were unstable** without clipping: 1/3 and 0/3 seeds trained.
- **Mechanism.** On 59x59 the four untied blocks lock into a period-4 cycle, the test-time round-robin period, on 69% of mazes. DT-Recall reaches a fixed point on 62% and cycles on 2.5%.
- **Winner also trains from detached states.** Alpha 1 is the progressive term alone, so the gap is tied weights and random-length gradient windows against untied blocks and one-step windows.
- **Scope.**
  - The budget is about 36-40% of the official 50-epoch recipe in TF32-equivalent hours. That recipe (dt-baseline, 40,000 steps, ~7.1 L4-h in FP32) gets 0.915 on e59, and the released checkpoint 0.995.
  - A4\* got about 13% more counted FLOPs than A1, 2.7x the optimizer steps and about 3% more realised wall time.
  - Other test-time block orders are untested.
- **Cost.** $36.40 at the ceiling, of $40 (61 spot L4 VMs: 23 slots plus 38 relaunches from checkpoints after spot preemptions). All VMs and disks are deleted; gcloud lists are empty (07:34Z).

Files: [P18-detached-mazes/RESULTS.md](P18-detached-mazes/RESULTS.md), `P18-detached-mazes/PREREG.md`, `P18-detached-mazes/results/{analysis,references}.json`, `P18-detached-mazes/charts/*.json`, `P18-detached-mazes/{train18,eval18,analyze18,refs18}.py`, `GPU-LEDGER.md` (P18 section)

## P13: With the belief-logit runaway fixed, does the trader economy (market ResNet ported to mazes, trained on 9x9 only) beat Bansal et al. DT-recall at 99x99 at equal training compute and equal test FLOPs, and is any gain from the economics?

Status: No gain (verified; both verifiers agree). F1 holds (pre-registered falsifier); S1 is not met; S3 is untestable because the economy never binds.

Setup:
- **Arms.**
  - ECON-fix: Trader2D with belief logits bounded at 10 (F-LB) and the LMSR wealth cap, W0 = 20. 4 seeds.
  - ECON-nocap: the same code with the cap off. 4 seeds.
  - ECON-F0: the unfixed pilot.
  - ECON-active ladder: W0 = 0.5, Lindahl trader-funded markets.
  - DTR-eq: DT-recall retrained at the same 3.0 L4-h; 3 of 4 seeds trained.
  - M-ref: the released Bansal checkpoint.
- **Task.** Train on 9x9; test at 19, 33, 59 and 99 at equal test FLOPs.
- **Cost.** Estimated at $0.35 per VM-h: $37.52 of the $40 cap (107.2 VM-h, 90 spot VMs). At the published spot price it is about $47, over the internal cap but well under the $500 budget. Nothing was billed. The ledger misses 6 short VMs (about 0.4-1 VM-h).

Result:
- **Headline.** At 99x99 every Trader2D model scores 0.00 exact (9 of 9). DTR-eq scores 0.207 (seeds 0 / 0.83 / 0 / 0); M-ref scores 0.94.
  - ECON-fix minus DTR-eq: -0.21, Welch 95% CI [-0.87, 0.45].
  - Permutation p is 0.14.
- **No extrapolation.** The Trader2D arms learn 9x9 (v9 0.93-0.99 in 7 of 8 seeds). They score at most 0.04 at 33x33 and 0.02 at 59x59.
- **The economy never binds.**
  - After the first 10% of training the cap scales 0% of trade-pixels, and there are 0 respawns.
  - ECON-fix and ECON-nocap are indistinguishable at every size, so the economy reduces to local per-step losses.
  - The active ladder bound only while the network was bad: at W0 = 0.5 it bound on 17% of trade-pixels at v9 0.30, and on 0% by v9 0.86.
- **The runaway fix works mechanically but does not improve calibration.** max |b| is 10 against 3,634. Log loss at 99 is 2.16 nats against 2.20, and F0 with a post-hoc temperature beats it (F2 holds).
- **Verifier corrections.**
  - "Poorest trader ends with 503-559" is the minimum after the first 10% of training. The final minimum wealth is about 10.2k-11.2k.
  - The bootstrap CIs listed under 12n are the t_eq values; the 12n CIs are [-0.86, -0.15] and [-0.72, -0.03].
  - The seed bootstrap is anti-conservative. The Welch CI includes 0 at every size.
  - Training FLOPs: ECON used about 25 PMACs against about 84 for DTR-eq (0.30x). The pre-registered rule for a >2x gap (the DTR-eqflop arm) fired. That arm was cut, and the cut was not disclosed.
  - "Detaching is not what stops it" overreaches. No arm varies the hints with the architecture held fixed, BPTT-dscov was one seed on dev sets only, and S3b was not run.

Files: [P13/RESULTS.md](P13/RESULTS.md), `P13/PREREG.md`, `P13/results/`, `P13/{econ_maze,dtr_maze,eval_maze,analyze}.py`

## P15: On a tied discrete cellular automaton at depth 8-32, does likelihood-message settlement (local EM, no cross-level gradient) keep up with backprop through depth in distribution, and does it extrapolate better by recovering the exact rule?

Status: No gain (verified; both verifiers agree).
- S1 and S2 are false; F1, F2 and F3 fire.
- The maze task is void because G3 fails.
- Comparisons against the retrained DT-recall (P-*) are void because G5 fails as scored.

Setup:
- **Arms.**
  - Bl: likelihood-message settlement, with variants Bl-ac, Bl-local and Bl-shuffle.
  - A: backprop through depth, with variants A-1step, A-ds and A-soft-prog.
  - O: the oracle.
  - P-*: DT-recall.
- **Tasks.**
  - Prefix parity (mod 2) at D8, D16 and D32, with 5-10 seeds.
  - Mazes at D16 and D32, with 5 seeds.
- **Cost.** $9.26 spot ($15.31 at the ceiling), 17.81 VM-h.

Result:
- **Keeps up at D16, not at D32.**
  - Prefix D16: Bl is exact in 10/10 seeds, and so is A.
  - Prefix D32: Bl scores 0.734 [0.50, 0.97] and is exact in 1/5 seeds. A scores 0.798 and is exact in 4/5. Fisher p is about 0.10.
- **The gap grows in log loss.** Delta log loss (Bl minus the best A arm) is 0.103, 0.394 and 0.406 nats per cell at D8, D16 and D32; the bar was <= 0.123. This is a log-loss statement (sampled latent entropy), not an accuracy one.
- **Extrapolation is a tie.** At 1024 bits Bl scores 1.000 against A's 1.000, 10/10 seeds each. Bl recovers the exact rule in 0/10 seeds (A 5/10, O 5/5), yet it still solves 1024 bits.
- **Mazes (void).** Bl is 0/5 in distribution. At 59x59, Bl scores 0.000 against 0.128 for A-ds.
- **Verifier corrections.**
  - "Never recovers the rule, so it extrapolates no better" is misleading: Bl solves 1024 bits in 10/10 seeds with 0/10 rule recovery.
  - The G3 maze text is wrong. The true rule is a fixed point. The failures are that corridor flips are invisible to settlement and that deadend_keep is pushed away.
  - Bl memory at D1024 was measured at batch 8, since batch 32 ran out of memory: 11,012 MiB. That is about 93x checkpointed BPTT at equal batch, not 20x.
  - G5 decay was x0.1, not x0.01.
  - The maze test sets were 200/50/50, not 200/100/100.
  - P15 tests EM, not the market: nothing in it is priced or traded.

Files: [P15/RESULTS.md](P15/RESULTS.md), `P15/results/analysis.json`, `P15/{p15,analyze,gates,p15_dtrecall}.py`

## P14 (verification note): the entry above stands as No gain (verified; both verifiers agree)

Corrections to the entry above:
- SHA256SUMS.vm covers only the odd-seed half of the eval files (145 of 290 JSONs).
- S3's 0.052 is the two-sided bound. The one-sided bound is 0.0495, still above 0.02.
- The log-rank test with actual FLOPs gives z = -4.88.
- The V fee grid saturates: there are 18-19 delistings at every fee of 1 and above. So the market was tested only in its maximum-churn regime.
- V-noD, at 4/20, still shows that LEG falls short without the churn.
- The listing-by-estimator interaction is untested.
- The equal-steps view is robust: BP* 15/20 against V 3/20.

Files: [P14/RESULTS.md](P14/RESULTS.md)

## P16 (verification note): the entry above stands as Problem (inconclusive, undertrained; verified; both verifiers agree)

Corrections to the entry above:
- **"FR's value is a price in the right units; C4's is not" is an artifact.**
  - C4's slope of 0.27 comes from its composite score (forward gain plus reverse CI term).
  - C4's forward gain alone has slope 1.18 (R² 0.36).
- **FAILED_STEPS has 13 entries, not 1.** All of them were re-run.
- **"Converged" partly reflects the cosine LR reaching its floor.**
- **F4 clause 1 is not discriminative:** every arm fails it.
- **C4 was under-tuned.** Its reverse term alone scored 0.783 on the tune set, so the negative (a) result (FR - C4 = -0.040 [-0.061, -0.020]) is conservative.
- **F1 wording.** FR is below C4, not "within 0.02".

Files: [P16/RESULTS.md](P16/RESULTS.md)

Dropped designs (batch 3, P13-P16): none.
