# wm2 autoresearch loop: experiment log

One entry per experiment, written BEFORE it runs (hypothesis, configuration, what result would change the plan), then
its result (VALIDATION only; TEST is read once at the end). Times are UTC. Each launch records the sha256 prefix of
this file in `/tmp/wm2/logs/launch-*.log` (`loop_sha=`), so the order "hypothesis, then runs" can be checked.
Numbers come from `results/*-summary.json` (`python3 analyze2.py`); raw runs in `/tmp/wm2/runs/<wave>/<task>/<tag>/`.
Selection rule and protocol: PREREG.md. Review: SPEC-REVIEW.md.

## Smoke (written 2026-10-11 13:11Z, before launch; one on-demand L4, seed 0, validation only; not part of the loop)
Purpose: timings for pricing and a functional check of every arm on a GPU. Jobs: `jobs/smoke.txt` (WMc at N = 300, 1000,
10000 (25-min cap) on T1 and N = 300 on T2; k0 at N = 1000 and rbtd8 at N = 300; forecasters D-only, D-mix50, D-post,
S-only, D-rbtd8, D-init on the N = 300 WM, D-only and D-post at N = 1000, D-only on the T2 WM).
Expectations: the WM's location head learns about 0 information about the board in T1 and clearly positive information in
T2; dreamed boards are mostly valid placements at N = 1000; dream-only forecasting at N = 300 lands between k0 (about
4.5e-3) and rbtd8 (about 2.0e-3), since the WM's prior from 300 boards is rough. No loop decision rests on the smoke.

## Wave 1 (written 2026-10-11 13:33Z, before launch): baselines, the generic joint WM, and every way the spec uses it

All at N = 100, 300, 1000, 3000, 10000, seeds 0-2 (paired boards across arms; every forecaster arm of one (N, seed) uses
the same WM), protocol of PREREG.md. Task T1 unless marked T2.

| # | tag | hypothesis (what I expect, and why) | what would change the plan |
|---|---|---|---|
| E01 | B-k0 | round-1 k0 under the final protocol: about 7.0 / 4.2 / 1.5 / 0.75 / 0.41e-3 (round-1 TEST numbers) | - |
| E02 | B-rbtd8 | round-1 champion: about 4.9 / 1.9 / 0.64 / 0.40 / 0.20e-3 | if it differs from round 1 by more than seed noise, check the harness before anything else |
| E03 | WMc | the generic joint WM (no forecaster). Expect: location information about 0 in T1 (it learns that probes ignore the board), readings at about 3.0 nats per episode (6 x H(0.2)), board NLL falling from about 11 nats at N = 100 towards log 11088 = 9.31 at 10k; dreamed boards mostly valid placements from N = 1000 on | if the location head learns a spurious board link at small N (location info > 0.1 nats), fix the WM (split early stopping) before trusting dream arms |
| E04 | WMc-D-only | a fresh forecaster trained only on dreams converges to the WM's own posterior. The WM learns the prior from fully observed boards and a simple reading model, so generative training should beat k0 at every N; it beats rbtd8 where the WM's prior is good (N >= 1000) and may trail it at 100-300, where 100-300 boards give a rough prior | if it beats rbtd8 everywhere, the WM is the way and wave 2 improves the WM; if it loses everywhere, the WM's posterior (diagnostic) tells whether the WM or the dream training is short |
| E05 | WMc-D-mix50 | half real, half dreams: the real half anchors the forecaster on true outcomes but overfits at small N; expect between k0 and D-only at small N, close to D-only at large N | the dream ratio as a knob |
| E06 | WMc-D-mix88 | 7/8 dreams: expect close to D-only | as E05 |
| E07 | WMc-D-rbtd8 | rbtd8 on 50% dreams + 50% real (arm 5): rbtd's low-variance targets plus unlimited dreamed episodes; expect the best wave-1 arm, beating rbtd8 at every N | if it wins, wave 2 varies its dream ratio and the WM under it |
| E08 | WMc-S-only | shared network (arm 6): the WM itself learns to forecast from dreams while keeping its WM loss; its features already encode ship structure, so expect a faster start than D-only and the same end point, possibly slightly better at small N | shared vs separate answered |
| E09 | WMc-D-post | arm 4, FLAGGED, never champion: the WM's own importance-sampling posterior as target on a fixed pool of 8192 dreams. Lower-variance targets: expect better than D-only at small N; the fixed pool may limit it at large N | reported separately |

### Smoke result and wave-1 change (written 2026-10-11 14:03Z, before the wave-1 launch)
Smoke (seed 0, validation, `/tmp/wm2/runs/S/`): at N = 300 the joint WM (WMc) early-stops at step 1040, when its board
model starts to overfit; its reading model is then still short of the truth (validation reading NLL 3.16 nats per
episode against 3.00 for the true 0.2 flip rate), and only 25% of its dreamed boards are valid placements. Forecasters on
it: D-only 2.00, D-mix50 2.05, S-only 1.83, D-init 2.19 x1e-3, against rbtd8 1.51 (k0 is about 4.5). The dream arms
lose mostly at the later probe stages (stage-6 KL 3.97 vs rbtd8 2.82), which is what an under-trained reading model
does. The T1 WM learns no spurious location link (0.002 nats per episode); the T2 WM already carries 1.16 of the 1.82
nats of selection information at N = 300. The WM's posterior diagnostic cost 13 min per WM and uses one fixed cell order
per placement, so it is off in the loop.
Change: wave 1 adds the split WM **WMs** (board model and probe model as two networks of the same size, each
early-stopped on its own part of the validation NLL; still the same joint distribution, nothing hand-coded), and moves
the dream ratio, arm 5 and arm 4 onto it. Hypothesis E10: WMs' reading NLL reaches about 3.00 and its dream arms close
much of the stage-6 gap, so WMs-D-only beats WMc-D-only at every N. Dream reuse: each dream is now drawn about 4 times
(fresh symmetry each time) and dreams are sampled under bf16, to cut the cost of dream arms about 3x.

Wave 1 as launched (T1: 4 VMs, T2: 3 VMs; job files `jobs/W1-t1-vm*.txt`, `jobs/W1-t2-vm*.txt`):
- T1: B-k0, B-rbtd8; WMc with D-only, S-only; WMs with D-only, D-mix50, D-mix88, D-rbtd8, D-post (flagged).
- T2 (written before any T2 loop run): B-k0, B-rbtd8; WMs with D-only; **WMsb** = WMs with a board-blind location head (not
  L0; the ablation) with D-only. Hypotheses: E11 WMs's location information on T2 rises with N towards 1.82 nats; E12
  T2-WMs-D-only has lower KL to the selection-aware posterior than T2-WMsb-D-only at every N and than the
  selection-unaware exact posterior (15.1e-3) from N = 300 on; E13 the baselines k0 and rbtd8 also learn the selection
  effect implicitly (they see probe cells as inputs) but need more boards than the WM pipeline to beat the unaware
  posterior.

## Wave 2 (written 2026-10-11 14:58Z, before launch): better world models under the two main forecasters

Signal (wave-1 world models, T1 validation, 3 seeds at N <= 300, `/tmp/wm2/scratch/an/t1-val-wm.md`): the split WM is far
better than the joint WM. Board NLL per episode 11.25 vs 13.13 at N = 100, 10.30 vs 10.94 at 300, 9.87 vs 10.26 at 1000
(optimum log 11088 = 9.31); readings 3.02-3.08 vs 3.06-3.15 (optimum 3.00); dreamed boards valid 23 / 49 / 67% vs 2 / 28 /
49%. Its board network early-stops 3-7x later once the probe model no longer shares its stopping point. No forecaster
result yet. Smoke (1 seed, joint WM): rbtd8 on half-dreamed batches beat rbtd8 at N = 300 (1.16 vs 1.51e-3).
So the board prior is the lever; wave 2 tries two generic ways to learn a better one, each under D-only (to read the WM
through its own posterior) and D-rbtd8 (the strongest forecaster so far). T1 only, 5 N x 3 seeds, 4 VMs.

| # | tag | change | hypothesis | decision it informs |
|---|---|---|---|---|
| E14 | WMr | split WM whose board network is trained with reveal-chain TD (round-1 rbtd, 8 levels, at the empty-evidence stage) instead of maximum likelihood on random cut points | TD targets lower the variance of the board conditionals, as in round 1, so the prior is closer to the truth at small N (board NLL below WMs at N <= 1000) and its dreams are better | the WM's training objective |
| E15 | WMr-D-only, WMr-D-rbtd8 | the two forecasters on WMr | below WMs-D-only / WMs-D-rbtd8 at N <= 1000 if E14 holds | as E14 |
| E16 | WMe | an ensemble of 3 split WMs (same boards, init seeds 0-2); each dream comes from one member in turn, so the dream distribution is their average | averaging three models' distributions cannot raise the KL to the truth (convexity) and smooths each model's idiosyncratic mistakes; expect better dreams at small N, at 3x WM compute | ensembles as a compute-for-data knob |
| E17 | WMe-D-only, WMe-D-rbtd8 | the two forecasters on WMe | below the single-WM versions at N <= 1000 | as E16 |

## Final phase, part 1 (written 2026-10-11 14:59Z, before launch): the two baselines on TEST, both tasks, 5 seeds
F-B-k0 and F-B-rbtd8 are in the final comparison whatever the champion is, and rbtd8 at N = 10000 is among the slowest
runs, so they start now to save wall-clock (as round 1 did). Same protocol as the loop (PREREG), seeds 0-4, all 5 N,
`--test`, tags `F-B-k0`, `F-B-rbtd8`, wave `F`, tasks t1 and t2 (100 runs, 3 VMs). Their TEST numbers are not read until
the champion is fixed on validation: the loop's analysis (`analyze2.py`) reads only the `val` field, and `final2.py`, the
only script that reads `test`, runs once after the selection entry below.

## Requeue 1 (written 2026-10-11 15:39Z): the wave-1 bundle lost to CUDA OOM, rerun unchanged
T1, WMs, N = 100, seed 2 (the WM and its 5 forecasters D-only, D-mix50, D-mix88, D-rbtd8, D-post): same configuration and
output paths as wave 1 (`jobs/RQ1-t1-vm0.txt`), one rental. No new hypothesis.

## Wave 3 (written 2026-10-11 15:53Z, before launch): the dream ratio of the strongest forecaster
Signal (T1 validation, partial): rbtd8 trained on half-dreamed batches from the split WM (WMs-D-rbtd8) is below rbtd8 at
N = 300 and 1000 on outcome log-loss (1-2 seeds: -8.5 and -2.7 x1e-4 nats per cell; KL 1.01 vs 1.75 and 0.51 vs 0.62e-3);
dream-only k0 on the same WM (WMs-D-only) is close to rbtd8 at 1000-3000 (0.57 vs 0.62, 0.37 vs 0.35e-3). The project cap
was raised by eurisko (lib.sh BUDGET_USD 260), so wm2 has its full $100 cap. T1 only, 5 N x 3 seeds, new WMs per bundle.

| # | tag | change | hypothesis | decision it informs |
|---|---|---|---|---|
| E18 | WMs-D-rbtd8-rho0.75 | rbtd8 with 75% dreamed + 25% real episodes per batch | more dreams help where the WM is good (N >= 1000) and hurt little at small N because rbtd's own targets already smooth the real half; expect at least as good as rho 0.5 | the dream ratio of arm 5 |
| E19 | WMs-D-rbtd8-only | rbtd8 on dreams only (rho 1) | separates the two ingredients: if it matches D-rbtd8 (rho 0.5), real episodes add nothing once rbtd trains on dreams; if it is worse, the real half corrects the WM's errors; expect worse at N <= 300 (rough WM) and close at large N; better than D-only everywhere (rbtd's lower-variance targets) | whether the forecaster needs real episodes at all |

### Observation 16:14Z (validation, partial)
- T2: the L0 split WM's location head learns the selection rule: location information 0.31 / 1.13 / 1.82 / 1.87 / 1.85 nats
  per episode at N = 100 / 300 / 1k / 3k / 10k (exact 1.82; T1 WMs: 0.00 +- 0.01). Its dream-only forecaster reaches KL
  14.0 / 6.8 / 1.85 / 1.05 / 1.04e-3 against the selection-aware posterior; the board-blind twin stays at 16.8 / 15.1 /
  14.9 / 14.6 / 14.6e-3, the level of the exact selection-unaware posterior (15.1e-3). E11 and E12 hold so far.
- The split WM's board network reaches the common 40000-step cap at N >= 3000 (best steps 38.9k-39.8k in every member
  and seed), so WM arms at 3k-10k are cap-limited (as round 1's 30k runs were). The protocol stays as pre-registered; this
  is reported as a limitation, and the obvious follow-up is a larger cap for every arm.

## Requeue 2 (written 2026-10-11 17:22Z, before launch): WMr with rbtd8-on-dreams at N = 3000 and 10000 on lightly loaded rentals
Same configuration as wave 2 (E14/E15), seeds 0-2, one seed per rental, 4 jobs at a time, to replace runs that will stop
at wave 2's rental deadline (PREREG deviation 17:21Z). No new hypothesis.
Added 17:29Z: the same three rentals also rerun WMr dream-only (WMr-D-only) at N = 3000 and 10000 on their own RQ2 WMs
(a second ready-queue per rental, `/tmp/wm2/scratch/rq2-extra.sh`), because wave 2's N = 10000 dream-only runs on WMr
started late and may also stop at the rental deadline. Same rule: they replace only deadline-stopped originals.

## Selection (2026-10-11 18:39Z, VALIDATION OUTCOMES ONLY, `analyze2.py --task t1 --waves W1,W2,W3,RQ2`)
Champion by the pre-registered rule: **WMr-D-rbtd8** = a split world model whose board network is trained by reveal-chain
TD (WMr), with rbtd8 as the forecaster trained on 50% dreamed + 50% real episodes. S_out -0.428; its normalised terms at
N = 100 / 300 / 1k / 3k / 10k are -1.23 / -0.25 / -0.32 / -0.26 / -0.09 (outcome log-loss below rbtd8 at every N).
Next eligible: WMe-D-rbtd8 -0.338, WMs-D-rbtd8 -0.320, WMr-D-only -0.285. Not eligible: WMs-D-rbtd8-rho0.75 (-0.393 on
2 of 3 seeds at N = 10000; seed 1 was lost to a spot preemption), WMs-D-post (arm 4, flagged; -0.153).
Pending at this time: the RQ2 reruns of WMr-D-rbtd8's four deadline-stopped runs (N = 10000 seeds 0-2, N = 3000 seed 2).
They cannot plausibly change the choice: to fall behind WMe-D-rbtd8, WMr-D-rbtd8's N = 10000 term would have to rise
from -0.09 (on runs truncated by the deadline, which can only understate it) to above +0.37. The rule is recomputed when
RQ2 lands and the result is written below. WMe-D-rbtd8's three N = 10000 runs also stopped at the deadline and are not
rerun (budget): for WMe-D-rbtd8 to overtake, its N = 10000 term would need to go from -0.10 to about -0.55, far beyond any
arm's N = 10000 result (best -0.13).
Final part 2 launched next: F-WMr-D-rbtd8 on T1, and on T2 the twin WMr (L0 and board-blind probe networks) with
rbtd8-on-dreams and dream-only forecasters, seeds 0-4, all 5 N, `--test` (`mkfinal2.py --champ WMr-D-rbtd8`).

### Selection recomputed with the RQ2 reruns (2026-10-11 19:35Z, VALIDATION OUTCOMES ONLY)
RQ2's reruns replace WMr-D-rbtd8's four deadline-stopped runs (N = 10000 seeds 0-2 and N = 3000 seed 2; all reruns
stopped by patience except N = 10000 seed 0, which reached the 40000-step cap at best step 32600). The rule's output is
unchanged: **WMr-D-rbtd8**, S_out -0.447 (normalised terms -1.23 / -0.25 / -0.32 / -0.24 / -0.20), validation KL 2.39 /
0.99 / 0.46 / 0.27 / 0.17 x 1e-3 at N = 100 / 300 / 1k / 3k / 10k = 0.49 / 0.55 / 0.74 / 0.76 / 0.84 x rbtd8's. Runner-up
WMe-D-rbtd8 -0.338. The final part 2 already running is therefore the right one.

## Final TEST analysis (2026-10-11 22:32Z): run once, after all 225 final runs were home
All final runs complete (T1: F-B-k0, F-B-rbtd8, F-WMr-D-rbtd8, 25 each; T2: F-B-k0, F-B-rbtd8, F-WMr-D-rbtd8,
F-WMrb-D-rbtd8, F-WMr-D-only, F-WMrb-D-only, 25 each). None stopped at a rental deadline; 9 stopped at the 40000-step cap.
Zero wm2 VMs (gcloud, 22:31Z). `final2.py` is run now, once per task.
Result (`results/test-t1-final.json`, `results/test-t2-final.json`, read once at 22:31Z):
- T1: WMr-D-rbtd8 vs rbtd8 on TEST: KL x0.54 / 0.49 / 0.85 / 0.78 / 1.07 at N = 100 / 300 / 1k / 3k / 10k; paired 95% intervals
  below 0 at 100-1000, including 0 at 3000 and 10000. Decision rule confirmed at 100-1000 only. vs k0: below at every N;
  k0's 10k-board KL reached with 0.15 of the boards (0.10-0.25).
- T2: yes at every N: location information > 0 (0.31-1.93 nats), both L0 pipelines below the ignore-locations posterior
  (15.18e-3) and below their board-blind twins, every paired 95% interval below 0.
Hypotheses checked against the result: E07/E15 (rbtd8 on dreams beats rbtd8) held at small N, not at 3k-10k; E10 (split WM
better than joint) held at every N; E14 (TD-trained board model better) held strongly; E16 (ensembles) gave little; E11-E13
(T2) held.
