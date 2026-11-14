# ShipHunt round 2: learning a world model from N episodes (wm2): results

eurisko study, 2026-10-11. Written by an Opus 5.5 subagent for eurisko. Spec: `/home/agent/workspaces/mats/outbox/shiphunt-worldmodel-spec-2026-10-11.md` (Kelvin: "Sounds good to me, let us make new experiments to try to find the optimal way to learn the world-model as we discussed."). Review: `SPEC-REVIEW.md`. Pre-registration and deviations: `PREREG.md`. Experiment log with hypotheses written before each wave: `LOOP.md`. Builds on round 1 (`../smallnet/`). Paths are relative to `/home/agent/workspaces/eurisko-study/li-net/wm2/` unless absolute.

**Status: done.** 437 loop run files and 225 final forecaster runs (plus their 50 world models); the champion was selected on validation outcomes and scored once on TEST (22:31Z). T2 answered on TEST. Zero wm2 VMs at the end. GPU total $79.44.

## 1. TL;DR

- **Champion** (selected on validation outcomes only, from 13 forecaster arms on 4 world models): a world model whose board network is trained by reveal-chain TD, used to make dreams on which rbtd8 trains, half dreamed and half real episodes per batch (`WMr-D-rbtd8`). It is real-world-valid and L0: everything it learns comes from the realised episodes and its own outputs.
- **On TEST (5 seeds) the pre-registered rule "beats rbtd8 at every N" is not met.** The champion is better than rbtd8 at small N only: KL 0.54x / 0.49x / 0.85x rbtd8's at 100 / 300 / 1000 boards, with paired 95% intervals below 0. At 3000 boards it is 0.78x and at 10k 1.07x, both intervals including 0: a tie, within run-to-run noise (rbtd8 itself moved 15% at 10k between rounds with the same code and boards). Against k0 it wins at every N (0.21-0.46x), and it reaches k0's 10k-board KL with **0.15 of the boards** (CI 0.10-0.25), against 0.25 for rbtd8.
- **How to learn the world model.** (1) Split early stopping: the board model and the probe model, two networks for the same joint distribution, each stopped on its own validation likelihood. This was the biggest single fix (2.0 nats better board model at 100 boards). (2) Train the board model with TD targets (its own conditional after more cells are revealed) instead of 0/1 cells: within 0.1 nats of the optimum from 1000 boards, and 95% of its dreamed boards are legal ships. (3) Three-model ensembles add little.
- **How to use it.** Dreams alone are best at small N (0.44x rbtd8 at 100 boards with the TD world model) but worse at 10k (1.5x): the forecaster converges to the world model's own posterior, whose small residual error then dominates. rbtd8 on dreams plus real episodes keeps most of the small-N gain and matches rbtd8 at large N. A shared network (world model = forecaster) beats a separate one on the joint world model at four of five N, but the split world model beats both.
- **T2: yes, the world model learns that where a probe lands is evidence.** Its probe model learns the selection rule from the episodes alone: 1.86-1.93 of the 1.82 nats it carries, from 1000 boards on, and 0.00 in T1, where there is none. Its dreams carry the effect to the forecaster. On TEST, both of its forecasters beat a Bayesian that ignores where probes land (15.2e-3) and beat their board-blind twins at every N (paired 95% intervals below 0). The board-blind dream-only twin stays at the ignore-locations level (14.6-16.5e-3).
- **Arm 4 (flagged; Kelvin has not ruled):** the world model's own posterior as target is the strongest split-model arm at N <= 1000 on validation (0.42-0.78x rbtd8) but worse at 3k-10k. It was never eligible and was not run on TEST.
- **Cost.** The champion spends 1.2-4x rbtd8's training compute per board on TEST (48-171x k0's on validation); counting early-stopping evaluation too, about 2-8x rbtd8's (7.3x at 100 boards on the final runs; `AUDIT.md`). GPU total **$79.44** at the on-demand ceiling rate (91 VM-hours; cap $100). Zero wm2 VMs at the end.

## 2. What ran

- **Tasks.** T1 is round 1's ShipHunt "hard", unchanged (harness `lir-agent/harness/env/shiphunt.py`): a 5x5 board, ships of length 3, 2 and 2 (11088 placements, 7 occupied cells each), 6 random probe cells whose readings flip with probability 0.2; forecasts after 0, 2, 4 and 6 probes. Round 1's cached boards and posteriors are reused byte for byte (TEST and validation md5 as in the round-1 audit). T2 keeps the same boards, but probes depend on the board (pre-registered in SPEC-REVIEW C.13 before any model ran): with probability 0.7 a probe lands uniformly on an unprobed cell that is occupied or edge-adjacent to an occupied cell, otherwise uniformly on an unprobed cell. T2's exact posterior includes that selection likelihood (`t2data.py`, checked against simulation); a Bayesian that ignores where probes land scores 15.1e-3 nats per cell on validation and 15.2e-3 on TEST.
- **Data.** N = 100, 300, 1000, 3000, 10000 training episodes (a board and its one probe trajectory), nested, the same boards for every arm of a seed. 3000 validation episodes (early stopping and selection, outcomes only) and 5000 TEST episodes (scored once at the end). Exact posteriors are used only to score.
- **World model (WM).** The round-1 cell-token transformer (d 128, 4 layers, 2-D relative attention bias) plus a probe-order embedding and two per-cell heads (next probe cell, reading). It models a whole episode: the board cell by cell in a random order (any-order autoregressive), then the 6 probes in their observed order, each (cell, reading) given the full board and the earlier probes. It is trained on the N real episodes only, with the 8 board symmetries; nothing about ships, flip rates or how probes are chosen is built in. Variants (section 5): joint (one network, one early stop), split (the same joint distribution as a board network and a probe network, each early-stopped on its own part of the validation NLL), TD-trained (split, board network trained by round 1's reveal-chain TD instead of maximum likelihood), and an ensemble of three split WMs.
- **Dreams.** Sample a board from the WM cell by cell (25 passes), then 6 probes (cell among unprobed cells, then the reading). Forecasters train on the dreamed evidence after 0, 2, 4, 6 probes with the dreamed board's 0/1 cells as target (or rbtd8's targets on the dreamed board). Dreams come in buffers of 4096, each drawn about 4 times with a fresh symmetry.
- **Forecasters.** The round-1 network (577k parameters). k0 and rbtd8 run through round 1's own `smallnet.py`, unchanged. The dream arms use a loop that reproduces `smallnet.py` bit for bit when given real data only (checked on CPU: identical loss, validation and final KL, 0.026718, for k0 at N = 100; `/tmp/wm2/scratch/sn-k0.json`, `fc-rk0.json`).
- **Protocol, identical for every phase of every arm.** Round 1's per-N learning rate, weight EMA and evaluation cadence; patience 25 evaluations at N <= 1000 and 40 at N >= 3000; early stopping on all 3000 validation episodes' outcomes (forecasters: per-cell log-loss against the true cells; WMs: episode NLL); a common cap of 40000 steps at every N.
- **Selection, validation outcomes only.** S_out = mean over the 5 N of the seed-paired outcome log-loss difference to rbtd8, divided by rbtd8's own log-loss gain over k0 at that N (0 = rbtd8, -1 = beyond rbtd8 by as much as rbtd8 is beyond k0). Eligible: real-world-valid, L0, not flagged, full sweep with 3 seeds. KL to the posterior is reported, never used to select or stop.
- **Loop.** 3 waves plus 2 requeues, 3 seeds x 5 N each. T1: 4 WM variants, 13 forecaster arms on them, 2 baselines. T2: 2 WM variants (L0 and board-blind), 2 forecaster arms, 2 baselines. 437 run files (296 forecaster or baseline runs, 141 WM runs), of which 18 are RQ2 reruns of wave-2 configurations. Hypotheses were written into LOOP.md before each launch, and each launch log records the sha256 prefix of LOOP.md at that moment (`/tmp/wm2/logs/launch-*.log`, `loop_sha=`).

## 3. TEST: the champion against rbtd8 and k0 (T1, 5 seeds, scored once)

The champion was fixed on validation at 18:39Z and confirmed at 19:35Z (LOOP.md, "Selection"). Every final run was retrained from scratch with the same protocol plus `--test`, seeds 0-4: `F-WMr-D-rbtd8` (its world model retrained per seed and N), `F-B-rbtd8`, `F-B-k0`. None stopped at a rental deadline; 2 stopped at the 40000-step cap (one champion run and one rbtd8 run at N = 10000). TEST was read once, by `final2.py` at 22:31Z. Sources: `results/test-t1-final.json`, `results/test-t1-final.md`; raw `/tmp/wm2/runs/F/t1/`.

KL x 1e3 on TEST (mean over seeds +- 95% t-interval half-width; 5 seeds each):

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| WMr-D-rbtd8 (champion) | 2.675 +- 0.936 | 0.874 +- 0.165 | 0.496 +- 0.068 | 0.294 +- 0.092 | 0.186 +- 0.049 |
| B-rbtd8 (round-1 champion) | 4.863 +- 0.921 | 1.780 +- 0.406 | 0.585 +- 0.077 | 0.379 +- 0.122 | 0.172 +- 0.018 |
| B-k0 | 7.092 +- 0.615 | 4.221 +- 0.617 | 1.491 +- 0.195 | 0.756 +- 0.068 | 0.406 +- 0.112 |

Paired differences (champion minus base, KL x 1e3; same seed = same training boards; 95% t-interval), geometric-mean ratio, and the same comparison on outcome log-loss (x1e-4 nats per cell, no posterior involved):

| comparison | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| vs rbtd8, KL | **-2.19 +- 0.65 (x0.54)** | **-0.91 +- 0.37 (x0.49)** | **-0.090 +- 0.058 (x0.85)** | -0.086 +- 0.130 (x0.78) | +0.015 +- 0.064 (x1.07) |
| vs rbtd8, log-loss | -22.0 +- 6.2 | -10.0 +- 4.1 | -1.18 +- 0.67 | -0.99 +- 1.44 | +0.37 +- 0.71 |
| vs k0, KL | -4.42 +- 1.18 (x0.36) | -3.35 +- 0.47 (x0.21) | -1.00 +- 0.22 (x0.33) | -0.46 +- 0.13 (x0.38) | -0.22 +- 0.11 (x0.46) |

**The pre-registered decision rule on TEST** (`results/test-t1-final.json`, "decision"):
- Beats rbtd8 at every N: **no, the rule is not met.** The paired intervals are below 0 at N = 100, 300 and 1000, not at 3000 (interval includes 0) or 10000 (no difference). The outcome log-loss says the same at every N.
- Beats k0 at every N: **yes**. Boards needed to reach k0's 10k-board KL: **0.15 of 10k** (seed bootstrap 0.10-0.25); this round's rbtd8 needs 0.25 (0.15-0.39), round 1's 0.28 (0.20-0.37). Boards needed to reach rbtd8's 10k-board KL: not reached within the sweep (the champion's 10k KL is 0.186 against rbtd8's 0.172).

IMG: results/test-t1-curve_kl_vs_boards.png

Where the gain sits (TEST, KL x1e3 per stage at N = 1000; stage = probes seen 0 / 2 / 4 / 6): champion 0.063 / 0.310 / 0.617 / 0.993, rbtd8 0.073 / 0.369 / 0.733 / 1.166, k0 0.137 / 1.009 / 1.879 / 2.938. The champion's edge over rbtd8 is spread over all stages, about 15% at each.

Consistency with round 1: this round's rbtd8 on TEST (4.86 / 1.78 / 0.59 / 0.38 / 0.17) and k0 (7.09 / 4.22 / 1.49 / 0.76 / 0.41) match round 1's (rbtd8 4.94 / 1.86 / 0.64 / 0.40 / 0.20; k0 7.12 / 4.22 / 1.46 / 0.75 / 0.41) within seed noise at N <= 3000, on the same boards and the same code. At 10k this round's rbtd8 is 15% lower than round 1's (paired over the same seeds -0.031 +- 0.028e-3), which makes rbtd8 a slightly harder baseline here. That run-to-run shift is twice the champion's 10k difference from rbtd8, so at 10k the two are within noise.

Why the loop's small validation edge at 3k-10k did not hold: the retrained final runs show the same tie on their own validation data as on TEST (per seed, validation and TEST differences agree to 0.01e-3), while the loop's runs of the same seeds had shown -0.04e-3 at 10k. So that edge was run-to-run noise, not a TEST artefact (`AUDIT.md`, row 28).

Compute on TEST (FLOPs per training board to the best checkpoint, all phases): champion 5.0e12 / 2.8e12 / 1.8e12 / 1.2e12 / 5.8e11; rbtd8 1.5e12 / 7.1e11 / 7.0e11 / 4.4e11 / 4.9e11; k0 3.3e10 / 3.0e10 / 2.6e10 / 2.4e10 / 2.1e10. The champion uses 1.2-3.9x rbtd8's compute per board.

## 4. T2: does the world model learn that where a probe lands is evidence?

T2 keeps T1's boards but makes probes board-dependent (pre-registered rule: with probability 0.7 a probe lands on an unprobed cell that is occupied or edge-adjacent to an occupied cell). A Bayesian that ignores where probes land scores 15.1e-3 nats per cell against the selection-aware posterior on validation (15.2e-3 on TEST), more than every T2 learner's KL except at 100 boards, so the effect is large at every N.

**The L0 world model learns the selection rule from the episodes.** Its probe model is never told how probes are chosen; it sees the board and learns p(next cell | board, earlier probes) by maximum likelihood. Location information = mean over validation episodes of log p(probe cells | the true board) minus the same under another episode's board; the exact value under the true rule is 1.82 nats per episode (0 in T1). Source: `results/t2-val-wm.md`, `results/t1-val-wm.md`.

| N | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|
| L0 split WM on T2 (3 seeds, mean +- sd) | 0.31 +- 0.06 | 1.13 +- 0.85 | 1.82 +- 0.14 | 1.87 +- 0.03 | 1.85 +- 0.04 |
| the same WM with a board-blind probe model (T2, not L0) | 0 | 0 | 0 | 0 | 0 |
| L0 split WM on T1 (truth 0) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Its dreams carry the effect into the forecaster (VALIDATION, 3 seeds; KL x1e3 to the selection-aware posterior; in brackets the same forecaster's KL to the selection-unaware posterior).** Source: `results/t2-val-summary.md`.

| arm | N = 100 / 300 / 1k / 3k / 10k |
|---|---|
| L0 split WM, dream-only (WMs-D-only) | 13.99 / 6.80 / 1.85 / 1.05 / 0.97 (6.6 / 9.8 / 14.7 / 15.9 / 15.9) |
| board-blind WM, dream-only (WMsb-D-only, not L0) | 16.84 / 15.12 / 14.89 / 14.64 / 14.58 (4.3 / 1.7 / 0.99 / 0.81 / 0.74) |
| rbtd8 (sees probe cells as inputs, learns from 0/1 outcomes) | 10.82 / 4.60 / 2.03 / 1.08 / 0.55 |
| k0 | 15.34 / 8.53 / 4.11 / 2.30 / 1.12 |
| exact posterior that ignores where probes land | 15.13 at every N |

The board-blind twin converges to the selection-unaware posterior (its distance to it falls from 4.3 to 0.74e-3), while the L0 WM's forecaster moves away from it (6.6 to 15.9e-3) and towards the truth. The direct learners (k0, rbtd8) also pick the effect up from their inputs and 0/1 outcomes: k0 needs about 300 boards to beat the ignore-locations Bayesian; rbtd8 already beats it at 100, and beats the WM's dream-only forecaster at 100, 300 and 10000 boards. So my pre-registered hypothesis E13 (the direct learners need more boards than the WM pipeline) is refuted for rbtd8. The T2 question is about the world model, and the pre-registered test compares the WM pipeline with its board-blind twin and the ignore-locations Bayesian, not with rbtd8.

IMG: results/t2-val-curve_kl_vs_boards.png

**TEST (5 seeds, scored once).** The pre-registered T2 final ran the champion's pipeline (TD-trained world model + rbtd8 on dreams and real episodes) and the dream-only forecaster, each on the L0 world model and on the same world model with a board-blind probe network (one board network trained per seed and N, two probe networks; PREREG deviation 17:55Z). Sources: `results/test-t2-final.json`, `results/test-t2-final.md`, `results/test-t2-locinfo.json`; raw `/tmp/wm2/runs/F/t2/`.

KL x 1e3 to the selection-aware posterior on TEST (mean +- 95% half-width, 5 seeds):

| arm | N=100 | N=300 | N=1000 | N=3000 | N=10000 |
|---|---|---|---|---|---|
| L0 WM + rbtd8 on dreams and real (WMr-D-rbtd8) | 9.82 +- 1.33 | 3.56 +- 2.18 | 1.08 +- 0.25 | 0.62 +- 0.12 | 0.50 +- 0.12 |
| board-blind twin (WMrb-D-rbtd8, not L0) | 10.76 +- 0.91 | 7.23 +- 0.56 | 5.70 +- 0.24 | 5.21 +- 0.42 | 4.79 +- 0.11 |
| L0 WM, dream-only (WMr-D-only) | 12.04 +- 0.79 | 5.15 +- 4.42 | 1.62 +- 0.21 | 1.00 +- 0.03 | 0.83 +- 0.19 |
| board-blind twin, dream-only (WMrb-D-only, not L0) | 16.49 +- 2.32 | 14.83 +- 0.88 | 14.75 +- 0.50 | 14.90 +- 0.30 | 14.59 +- 0.18 |
| rbtd8 | 11.66 +- 2.76 | 4.73 +- 0.91 | 2.04 +- 0.19 | 1.05 +- 0.09 | 0.57 +- 0.06 |
| k0 | 15.70 +- 2.07 | 8.46 +- 1.75 | 4.07 +- 0.35 | 2.29 +- 0.24 | 1.12 +- 0.08 |
| exact posterior that ignores where probes land | 15.18 | 15.18 | 15.18 | 15.18 | 15.18 |

The pre-registered T2 rule (PREREG, claim 3), applied as written:
- **(a) The L0 world model's probe model carries information about the board at every N:** 0.31 +- 0.05, 1.39 +- 0.82, 1.93 +- 0.13, 1.90 +- 0.01, 1.86 +- 0.03 nats per episode (final T2 world models, 5 seeds, 95% half-width; every interval above 0; exact 1.82). The board-blind twin: 0.
- **(b)(i) Below the ignore-locations Bayesian at every N:** rbtd8-on-dreams pipeline -5.4 / -11.6 / -14.1 / -14.6 / -14.7e-3; dream-only -3.1 / -10.0 / -13.6 / -14.2 / -14.3e-3; every 95% interval below 0.
- **(b)(ii) Below its board-blind twin at every N** (paired by seed): rbtd8-on-dreams -0.95 +- 0.67, -3.7 +- 1.7, -4.6 +- 0.3, -4.6 +- 0.5, -4.3 +- 0.1e-3 (x0.91 / 0.46 / 0.19 / 0.12 / 0.10); dream-only -4.5 +- 2.0, -9.7 +- 4.5, -13.1 +- 0.6, -13.9 +- 0.3, -13.8 +- 0.3e-3 (x0.73 / 0.30 / 0.11 / 0.07 / 0.06). Every interval below 0.
- **Verdict: yes at every N.** Where a probe lands counts as evidence for the L0 world model, and its dreams pass that on to the forecaster. Of the three parts, (a) and (b)(ii) are the ones specific to the world model: (b)(i) is also met by rbtd8 trained on real episodes alone and by the board-blind pipeline that mixes in real episodes, because both see probe cells and real outcomes (`AUDIT.md`, row 29).

IMG: results/test-t2-curve_kl_vs_boards.png

Two further readings:
- **A world model that cannot learn the selection effect drags the forecaster towards ignoring it.** The board-blind dream-only twin sits at the ignore-locations level at every N. With half real episodes (board-blind + rbtd8), the real half teaches some of the effect, but the blind dreams still hold it at 4.8e-3 at 10k boards, 8x worse than rbtd8 on real episodes alone.
- **On T2 too, the champion's pipeline beats rbtd8 in the middle of the range:** 0.52x at 1000 boards and 0.59x at 3000 (intervals below 0), no clear difference at 100, 300 or 10000. It reaches rbtd8's 10k-board KL with 0.50 of the boards (bootstrap 0.28 to not reached), and k0's with 0.10.

## 5. How to learn the world model (VALIDATION, T1)

Four WM variants trained on the same N episodes (3 seeds; validation NLL per episode in nats; floors: board log 11088 = 9.31, readings 6 x H(0.2) = 3.00, probe cells 18.66 in T1). "Dreams valid" is the share of dreamed boards that are legal placements (evaluation-only: the WM never sees the placement list). Source: `results/t1-val-wm.md`; raw `/tmp/wm2/runs/{W1,W2,RQ2}/t1/WM*/`.

| WM | how it is trained | board NLL at N = 100 / 300 / 1k / 3k / 10k | dreams valid | reading NLL at N = 100 / 1k / 10k |
|---|---|---|---|---|
| joint (WMc) | one network, maximum likelihood of the whole episode, one early stop | 13.13 / 10.94 / 10.26 / 9.85 / 9.62 | 2 / 28 / 49 / 68 / 81% | 3.14 / 3.06 / 3.02 |
| split (WMs) | the same distribution as two networks (board, probes), each early-stopped on its own NLL | 11.13 / 10.30 / 9.80 / 9.60 / 9.60 | 25 / 49 / 72 / 83 / 83% | 3.07 / 3.02 / 3.01 |
| ensemble (WMe) | three split WMs (init seeds 0-2); each dream comes from one member in turn | members 11.11-11.23 / 10.18-10.30 / 9.67-9.80 / 9.58-9.62 / 9.60-9.65 | 23-25 / 49-55 / 72-79 / 82-84 / 80-83% | 3.06-3.08 / 3.02 / 3.01 |
| TD-trained (WMr) | split, board network trained by reveal-chain TD (round 1's rbtd, 8 levels, empty evidence) | **10.13 / 9.61 / 9.42 / 9.38 / 9.38** | **63 / 85 / 95 / 97 / 97%** | 3.06 / 3.02 / 3.01 |

- **Separate early stopping is the first fix.** In the joint WM the probe model (which learns fast and then overfits) sets the stopping point, so the board model stops 2-9x earlier than it would on its own at N <= 3000 (best step 317 vs 2773 at N = 100) and the reading model stays short of the true flip rate (3.14 vs 3.00 nats at N = 100). Splitting the same distribution into two networks fixes both: 2.0 nats less board NLL at N = 100.
- **TD targets make a much better board prior.** Training each board conditional toward the model's own conditional after more cells are revealed (lower-variance targets than the 0/1 cells, as in round 1) cuts the gap to the optimum from 1.8 to 0.8 nats at N = 100 and from 0.5 to 0.1 nats at N = 1000; 95% of its dreams at N = 1000 are legal boards, against 72%. The price: 8-12x more WM compute per board.
- **Three WMs from different initialisations** have the same quality as one; any ensemble gain can only show through the averaged dreams (section 6).
- **The probe model learns what is true and nothing else.** In T1, where probe cells do not depend on the board, every WM's location information is 0.00 +- 0.01 nats per episode. In T2 the same architecture learns 1.85 of the 1.82 nats the selection rule carries (section 4).
- **The common 40000-step cap binds for the split WMs' board network at N >= 3000** (best steps 36.6k-40k in every seed and member), so large-N WMs are cap-limited. No T1 baseline loop run reached the cap.

## 6. How to use the world model (VALIDATION, T1)

All 21 T1 experiments, 3 seeds each. KL x1e3 at N = 100 / 300 / 1k / 3k / 10k with the seed-paired KL ratio to rbtd8 in brackets; dLL = seed-paired outcome log-loss difference to rbtd8 (x1e4 nats per cell; negative = better); S_out is the selection score. Source: `results/t1-val-summary.md` / `.json` (from `analyze2.py`), raw `/tmp/wm2/runs/{W1,W2,W3,RQ2}/t1/`.

| tag | what it is | eligible | KL x1e3 (ratio vs rbtd8) | dLL vs rbtd8 x1e4 | S_out |
|---|---|---|---|---|---|
| **WMr-D-rbtd8** | TD-trained WM; rbtd8 on 50% dreams + 50% real | yes | 2.39 (0.49) / 0.99 (0.55) / 0.46 (0.74) / 0.27 (0.76) / 0.17 (0.84) | -25.2 / -7.3 / -2.2 / -0.8 / -0.4 | **-0.447** |
| WMs-D-rbtd8-rho0.75 | split WM; rbtd8 on 75% dreams | partial (one 10k seed lost to a spot preemption) | 3.25 (0.66) / 0.80 (0.46) / 0.60 (0.97) / 0.32 (0.90) / 0.27 (1.32) | -16.1 / -8.9 / -1.8 / -0.7 / -0.2 | -0.393* |
| WMe-D-rbtd8 | 3-WM ensemble; rbtd8 on 50% dreams | yes (its 10k runs stopped at a rental deadline) | 3.33 (0.68) / 1.00 (0.57) / 0.52 (0.83) / 0.28 (0.83) / 0.21 (1.04) | -16.5 / -7.5 / -2.1 / -0.7 / -0.2 | -0.338 |
| WMs-D-rbtd8 | split WM; rbtd8 on 50% dreams (arm 5) | yes | 3.39 (0.69) / 1.08 (0.62) / 0.52 (0.83) / 0.31 (0.91) / 0.21 (1.04) | -16.0 / -6.8 / -2.2 / -0.6 / -0.2 | -0.320 |
| WMr-D-only | TD-trained WM; dreams only | yes | 2.15 (0.44) / 1.08 (0.62) / 0.59 (0.94) / 0.35 (1.01) / 0.31 (1.53) | -27.5 / -6.8 / -1.4 / -0.1 / +0.7 | -0.285 |
| WMs-D-rbtd8-only | split WM; rbtd8 on dreams only | yes | 3.95 (0.81) / 1.09 (0.62) / 0.62 (1.00) / 0.32 (0.92) / 0.30 (1.43) | -10.3 / -7.1 / -1.7 / -0.6 / +0.3 | -0.206 |
| WMs-D-post | split WM; WM's own posterior as target (arm 4) | **FLAGGED** | 2.61 (0.53) / 0.75 (0.42) / 0.49 (0.78) / 0.51 (1.41) / 0.28 (1.36) | -22.0 / -9.3 / -2.7 / +1.9 / +0.8 | (-0.153) |
| WMs-D-only | split WM; dreams only (arm 3) | yes | 3.35 (0.68) / 1.16 (0.66) / 0.63 (1.00) / 0.48 (1.39) / 0.34 (1.69) | -16.4 / -6.6 / -1.3 / +0.8 / +0.5 | -0.133 |
| WMe-D-only | 3-WM ensemble; dreams only | yes | 4.38 (0.85) / 0.95 (0.55) / 0.48 (0.77) / 0.39 (1.12) / 0.36 (1.76) | -5.6 / -7.9 / -2.7 / +0.0 / +0.7 | -0.113 |
| WMs-D-mix88 | split WM; k0 on 7/8 dreams | yes | 3.48 (0.70) / 1.19 (0.68) / 0.63 (1.02) / 0.49 (1.36) / 0.37 (1.81) | -15.0 / -6.1 / -1.6 / +0.7 / +0.8 | -0.109 |
| B-rbtd8 | round-1 champion | baseline | 4.91 / 1.75 / 0.62 / 0.35 / 0.20 | 0 | 0 |
| WMs-D-mix50 | split WM; k0 on half dreams | yes | 6.05 (1.25) / 1.60 (0.92) / 0.73 (1.18) / 0.44 (1.27) / 0.30 (1.50) | +8.9 / -2.0 / -0.2 / +0.4 / +0.5 | 0.145 |
| WMc-S-only | joint WM shared with the forecaster (arm 6) | yes | 8.15 (1.69) / 2.04 (1.16) / 0.89 (1.42) / 0.50 (1.46) / 0.32 (1.61) | +30.8 / +2.7 / +1.6 / +1.7 / +1.1 | 0.593 |
| WMc-D-only | joint WM; dreams only (arm 2 + 3) | yes | 8.86 (1.83) / 2.27 (1.29) / 0.88 (1.39) / 0.54 (1.57) / 0.45 (2.20) | +38.4 / +4.8 / +1.5 / +1.8 / +1.9 | 0.770 |
| B-k0 | plain k0 | baseline | 7.03 (1.45) / 4.72 (2.71) / 1.50 (2.42) / 0.73 (2.15) / 0.45 (2.27) | +20.5 / +28.9 / +7.0 / +3.2 / +1.8 | 1.000 |

The k0-rbtd8 gap that scales S_out is 20.5 / 28.9 / 7.0 / 3.2 / 1.8 x1e-4 nats per cell. The validation baselines are close to round 1's (rbtd8 4.91 / 1.75 / 0.62 / 0.35 / 0.20 here, 5.00 / 1.98 / 0.71 / 0.42 / 0.22 in round 1's loop, which early-stopped on 1000 instead of 3000 validation boards).

IMG: results/t1-val-curve_kl_vs_boards.png

What the loop shows:
- **Dream-only training wins at small N and loses at large N.** Its fixed point is the WM's own posterior, so it is only as good as the WM. The TD-trained WM's dream-only forecaster is 0.44x rbtd8 at 100 boards but 1.5x at 10k: at 10k boards even a WM within 0.07 nats of the optimum (97% legal dreams) costs more than rbtd8 gains from 10k real boards.
- **Keeping real episodes, with rbtd's targets, holds the gain at every N.** rbtd8 on half-dreamed batches is below rbtd8 on outcomes at all five N for all three WMs tried (split, ensemble, TD-trained), as point estimates: with 3 seeds only the N = 100 interval excludes 0 on validation (the champion's paired log-loss difference x1e4: -25.2 +- 17.8 / -7.3 +- 9.5 / -2.2 +- 3.2 / -0.8 +- 2.0 / -0.4 +- 0.9; `AUDIT.md`, row 6). The champion's lead over the runner-up also comes from N = 100; at N >= 300 the three rbtd8-on-dreams arms are within noise of each other. With k0's 0/1 targets instead (D-mix50), the real half overfits at 100 boards and the mix is worse than rbtd8; with rbtd's own-forecast targets the real half helps. Dreams only under rbtd (D-rbtd8-only) falls back at 10k, and 75% dreams is stronger at 300 boards but weaker at 10k than 50%.
- **A better board prior helps both forecasters.** From the split WM to the TD-trained one, at N = 100, rbtd8-on-dreams goes from 3.39 to 2.39 and dream-only from 3.35 to 2.15 x1e-3. The ensemble does not help consistently (3.33 and 4.38).
- **Shared vs separate (arm 6).** On the joint WM, the shared network (the WM itself trained to forecast from its own dreams while keeping its WM loss) beats a separate dream-only forecaster at four of five N (8.15 vs 8.86 at 100, 0.32 vs 0.45 at 10k; equal at 1000, 0.89 vs 0.88). But both sit far behind the split-WM arms, because the joint WM is the weak part, and a split WM has no single network to share. Sharing helps a little; splitting the WM's early stopping helps far more.
- **Arm 4 (flagged)** is in section 8.

## 7. Compute per training board (VALIDATION, T1)

FLOPs per training board to the best checkpoint, every phase counted (WM training, dream sampling, importance sampling, forecaster training; forward + backward = 3x forward; evaluation excluded for every arm), and the multiple of k0 at the same N. Source: `results/t1-val-summary.json` (`flops_per_board_mean`, `flops_per_board_total_mean`).

| arm | N = 100 / 300 / 1k / 3k / 10k | x k0 |
|---|---|---|
| B-k0 | 3.3e10 / 2.9e10 / 2.4e10 / 2.6e10 / 1.4e10 | 1 |
| B-rbtd8 | 1.5e12 / 7.3e11 / 7.0e11 / 5.3e11 / 3.2e11 | 47 / 25 / 30 / 20 / 23 |
| WMr-D-rbtd8 (champion) | 5.7e12 / 2.8e12 / 2.1e12 / 1.3e12 / 6.5e11 | 171 / 97 / 89 / 48 / 48 |
| WMs-D-rbtd8 | 3.5e12 / 1.5e12 / 8.7e11 / 9.1e11 / 4.2e11 | 106 / 50 / 37 / 35 / 31 |
| WMr-D-only | 2.7e12 / 1.6e12 / 1.2e12 / 7.1e11 / 2.4e11 | 83 / 54 / 51 / 27 / 17 |
| WMs-D-only | 8.8e11 / 3.2e11 / 2.7e11 / 1.9e11 / 1.2e11 | 27 / 11 / 11 / 7 / 9 |
| WMs-D-post (flagged) | 1.4e13 / 4.8e12 / 1.6e12 / 5.6e11 / 2.0e11 | 434 / 165 / 69 / 21 / 14 |

The champion spends 2-4x rbtd8's training compute per board to its best checkpoint (the world model is 27-52% of it, dreams 2-3%), and 48-171x k0's. Training to the end (including the steps after the best checkpoint) adds 11-25% for the champion, 30-40% for rbtd8 and 2.1-2.6x for k0. **Not counted above, for any arm: early-stopping evaluation**, which is large for world models (the board network's validation NLL costs 25 passes x 3000 episodes per evaluation). With it, the audit's all-in figure for the champion is 7.8 / 6.3 / 3.7 / 2.4 / 1.9x rbtd8 per board at N = 100 / 300 / 1k / 3k / 10k (`AUDIT.md`, row 20).

## 8. Arm 4 (FLAGGED: Kelvin has not ruled): the WM's own posterior as target

The forecaster trains on a fixed pool of 8192 dreams whose targets are the WM's own posterior marginals for the dreamed evidence, estimated by self-normalised importance sampling over 1024 of the WM's own prior samples, weighted by the WM's own likelihood of the probe cells and readings (split WM). Effective sample size about 620 / 380 / 230 at 2 / 4 / 6 probes; the sampling cost 9-36 min per run on a shared L4. It uses only the model's own outputs, so it follows the letter of the rule; Kelvin has not ruled on it, so it was never eligible as champion and was not run on TEST.

Validation KL x1e3: 2.61 / 0.75 / 0.49 / 0.51 / 0.28 (0.53 / 0.42 / 0.78 / 1.41 / 1.36x rbtd8); outcome dLL -22.0 / -9.3 / -2.7 / +1.9 / +0.8 x1e-4. With the same WM, its lower-variance targets beat 0/1 dream targets (WMs-D-only 3.35 / 1.16 / 0.63 / 0.48 / 0.34) at N <= 1000, by the most of any split-WM arm at 300. At 3k-10k it is worse, likely because of the fixed pool (8192 dreams) and the importance sampler's bias at 1024 samples. S_out would be -0.153.

IMG: results/arm4-t1-val-curve_kl_vs_boards.png

## 9. Real-world validity of every arm

| arm | real-world-valid | L0 | reason |
|---|---|---|---|
| B-k0 | yes | yes | per-cell log score against the realised board |
| B-rbtd8 | yes | yes | targets are the model's own forecasts given more of the same board's true cells |
| world models (joint, split, TD-trained, ensemble) | yes | yes | trained only on realised episodes (maximum likelihood, or TD on the model's own conditionals); the probe model sees the board and has to learn whether probe cells depend on it |
| D-only, D-mix50, D-mix88 | yes | yes | trained on the WM's own samples (and real episodes); targets are dreamed or real boards |
| D-rbtd8 and its ratio variants (arm 5), including the champion | yes | yes | rbtd8's targets on real and dreamed episodes: the model's own forecasts and realised or dreamed boards |
| S-only (arm 6, shared) | yes | yes | the WM network itself, trained on its own dreams plus its WM loss on real episodes |
| D-post (arm 4) | by the letter yes | yes | targets are the WM's own posterior (importance sampling over its own prior samples); **Kelvin has not ruled: flagged, never champion** |
| board-blind WM (T2 ablation) | yes | **no** | its probe model cannot see the board, which hard-codes "probe cells carry no information" |

Every run also takes its own decisions from real-world signals only: early stopping on the validation episodes' outcomes (or the WM's likelihood of them), never on the posterior; the champion was chosen by the outcome-only S_out. Exact posteriors and the placement list appear only in scoring and in evaluation-only diagnostics (dream legality). Shared with round 1 and equal across arms, but not real-world: the 8 board symmetries and the 2-D attention bias are task knowledge, and every arm early-stops on 3000 validation episodes with known outcomes (30x the 100-board training set).

## 10. Deviations from the plan (full log with times: PREREG.md, "Deviations log")

1. **Split WM added before wave 1** (smoke: the joint WM's single early stop leaves its reading model short; LOOP.md 14:03Z).
2. **Two bugs fixed on running VMs before any affected job started**: a probe-model reference registered as a submodule (would have made joint-WM forecasters recurse; 14:15Z) and memory settings after one CUDA out-of-memory death (15:09Z). Both were patched by sed in the frozen copies, checked by grep; the lost bundle was rerun unchanged (RQ1).
3. **Deadline stops.** With 7 jobs per L4, TD-trained WMs plus rbtd8-on-dreams at N = 10000 did not fit a 220-minute rental. The rule (written 17:21Z, before those runs ended): deadline-stopped runs are replaced by reruns on lightly loaded rentals where they exist (RQ2: WMr-D-rbtd8 and WMr-D-only at N = 3000 and 10000); otherwise they count as is, marked. WMe-D-rbtd8's three N = 10000 runs count as deadline-stopped (not rerun: budget, and it could not have overtaken the champion).
4. **Spot and stockouts.** Two wave-3 rentals ran on spot L4s because the on-demand L4 quota (16) was full; one was preempted, losing WMs-D-rbtd8-rho0.75 at N = 10000 seed 1. Contrary to the 16:00Z log entry, that job was not requeued (it could not change the champion; budget), so the arm is reported as partial. Several final rentals waited out on-demand stockouts.
5. **Final part 1 for T2** put all 50 baseline jobs on one rental; at 17:41Z its queue was drained (new jobs exit at once, running jobs untouched) and the 34 unstarted jobs ran unchanged on a new rental (F1b).
6. **T2 final twins share one board network** (`wm_twin=1`): the board-blind flag changes only the probe network, and in the T2 loop the twins' board networks were identical to 3 decimals (17:55Z, before the champion was fixed).
7. **Champion fixed before the last RQ2 reruns landed** (18:39Z), with a bound showing they could not change it; recomputed at 19:35Z with the reruns: unchanged (S_out -0.428 -> -0.447).
8. **Budget mechanics.** One final rental was refused by gpu-up's group check, which counts every open rental at its full 4.5-hour reservation ($96.13 + $3.89 > $100 while actual spend was $54); wm2's launcher threshold on spend-plus-reservations went from $95 to $99 and the rental was relaunched once RQ2's reservations were released. The guard's $95 on actual spend was never changed. Infrastructure: an opt-in `GPU_STARTUP=minimal` was added to logical-soup's gpu-up.sh (the full startup spent 17 minutes on unused clang/LLVM downloads), and eurisko raised the project-wide cap in `lib.sh` from $200 to $260 so that wm2 had its full $100.
9. **The common 40000-step cap binds** mainly for the world-model arms: every split WM's board network at N >= 3000, and a few forecaster runs at N = 10000. No T1 baseline loop run reached it; two T2 rbtd8 runs at N = 10000 did. The cap can only understate the WM arms at large N.

## 11. Cost and clean-up

- **GPU total: $79.44** for wm2 (per VM, the DELETE or COST-AUDIT `cost_usd` in `logical-soup/GPU-LOG.md`, the on-demand ceiling rate x age; spot is billed lower). Computed by `gpu/cost.py` into `results/gpu-cost.json`; `spend_usd wm2` gives the same $79.44, with no open reservations.
- By machine: 32 on-demand L4 rentals, 86.5 VM-hours, $75.29; 2 spot L4 rentals, 4.6 VM-hours, $4.00 (one preempted); 5 failed creates (stockouts, quota), $0.15. 91 VM-hours in all.
- By phase: smoke $0.97; wave 1 $17.34 (7 rentals); wave 2 $12.44 (4); wave 3 $6.41 (3 plus 2 failed creates); requeues $5.85 (RQ1 $0.30, RQ2 $5.55); final baselines $8.01 (F1 $7.12, F1b $0.89); final part 2 $28.42 (11 rentals plus failed creates).
- Caps: wm2's group cap $100 was never reached; the guard (`guard-wm2.sh`, deletes wm2 VMs older than 250 min and the youngest above $95 of actual spend) never fired. eurisko raised the project-wide cap from $200 to $260 at 15:50Z so wm2 had its full $100; the logical-soup project total is now $209.74.
- Clean-up: `gcloud compute instances list --filter='name~soup-gpu-wm2'` returned 0 at 22:32:38Z and 22:32:40Z. The guard (eurisko's, pid 2032576) is still running and does nothing while no wm2 VM exists (stop it with `kill 2032576` when no more wm2 rentals are planned).

## 12. Files

- Plan and log: `SPEC-REVIEW.md` (review, 25 points), `PREREG.md` (pre-registration, pricing, deviations log), `LOOP.md` (hypotheses written before each wave, observations, the selection), `AUDIT.md` (independent skeptic audit).
- Code: `t2data.py` (T2 episodes and exact selection-aware posteriors), `wm2.py` (world models, dreams, importance sampling, forecaster loop), `job.py` (one job: round-1 `smallnet.py` for baselines, `wm2.py` otherwise), `mkjobs2.py` / `mkfinal2.py` (job files), `analyze2.py` (tables, the S_out selection, curves), `final2.py` (the one TEST analysis), `build_tables.py`, `render.sh`, `gpu/` (launcher, driver, VM-side ready-queue `remote_sched.py`, cost audits), `cost_model.json`.
- Results: `results/t1-val-*` and `results/t2-val-*` (runs CSV, summary JSON/MD, WM table, curves), `results/arm4-t1-val-*` (flagged arm), `results/test-t1-final.*` and `results/test-t2-final.*` (TEST), `results/gpu-cost.json`. Raw run JSONs: `/tmp/wm2/runs/<wave>/<task>/<tag>/` (waves S = smoke, W1-W3, RQ2, F = final).
- Rendered page: `/home/agent/memo-serve/lir-agent/wm2.html`.
