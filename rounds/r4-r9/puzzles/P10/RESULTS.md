# P10 results: masked cells as hidden variables (value-priced branching, learned events)

Run on 2026-09-29 on one spot L4 VM (trader-cat-p10-1, 12:35Z to 14:21Z, about $1.26 at the on-demand ceiling (spot bills less)). The VM is deleted; the GPU-LEDGER row has the check.

## Verdict

- **Part A (value-priced branching): F1.** Gate G1 fails, and the failure is robust. The head's within-state rank correlation with the realised commit gain is 0.026. The best free proxy, masked-peer count, reaches 0.185. The difference is -0.159 with a 95% CI of [-0.178, -0.138]. A post-hoc head trained on the within-state centred gain also fails (-0.187). Per the pre-registration, S9 and branch.py were not built, and $0 went to part A GPU work.
- **Part B (learned events): inconclusive by the pre-registration (F3b). The post-hoc controls locate the failure.**
  - The bank-chosen planted atom was not a positive. The atom-bank decoder ignored single atoms, so no atom had a positive held-out gain; even the cell atom [y_7 = 3] gained only 1.2e-4 nats/cell.
  - The learned events collapsed to constant events in all 45 learned-slot trainings (3 slots x 5 seeds x auction, grace 5000 and cvae): NMI 0 with every atom, G = -rate or 0. The rate term makes a constant the cheapest event before the decoder has learned to read any slot. This is posterior collapse.
  - The machinery itself works. A post-hoc positive control made the planted slot a fixed-cell bit [y_14 <= 3]. The decoder learned to read it in 3 of 4 seeds, with G = +0.051 nats/cell [-0.014, 0.117]. That is 10x the fee, and the gain is about 5x the bit's rate. This is V-information: one hidden bit is worth far more than its code length to a 2-pass model. In the auction (grace 5000, 4 seeds, post-hoc) the same bit survived in 4/4 seeds with G = +0.050 [0.033, 0.067], while cell-bit, copy-hash, random and trivial were de-listed at step 5000 in 4/4. So the auction keeps a slot that pays and drops those that do not; what failed is learning the events, not pricing them.
- **Status: Informative.** Part A's null is clean. Part B found no important-and-learnable *learned* events, but it does not show that none exist: the learned events collapse before the decoder can use them.

## Part A: gate G1 (S4 4x4, exact posteriors, 3 seeds x 3000 held-out states)

The table gives the within-state tie-aware Spearman with the realised commit gain G (wrong commits included; paired no-commit baseline, T_r = 1). Seed means are shown with t 95% intervals.

| score | rho vs realised G | rho vs exact I*(z_c; z_q given X) |
|---|---|---|
| value head (pre-registered) | 0.026 [-0.096, 0.149] | -0.55 |
| masked-peer count | **0.185** [0.037, 0.334] | -0.39 |
| 1 - p_max | 0.092 | 0.48 |
| entropy | 0.089 | 0.73 |
| MRV count | 0.007 | -0.28 |
| linear fit of the 4 proxies | 0.175 | -0.49 |
| centred head (post-hoc, separate B2 retrain) | -0.009 [-0.065, 0.048] | n/a |

- Head minus best proxy: -0.159, hierarchical bootstrap (seeds x states) [-0.178, -0.138]. The bar was +0.1, so G1 fails.
- The partial Spearman given entropy is 0.059 [0.051, 0.066]. It is above 0, but that alone is not enough to pass.
- Why the gain is hard to price: realised G is dominated by whether the argmax commit matches the sampled true grid in multi-solution states (mean -3 to -4.5 nats, sd 6-9). Its within-state rank correlation with the exact information I* is -0.009. The head does fit G on pooled data (Pearson 0.54), but only through between-state level; within-state it learns almost nothing (train 0.02-0.11). The "price" of a masked cell here is its uncertainty plus peer count, which matches F1.

## Readouts (known results, no bar)

- Void rule (the eps = 0.1 importance-sampled estimator against clamp-to-truth, exact on S4):
  - eps-greedy proposal: variance 47x the signal variance; the Huber-clipped estimate is biased by -0.43 to -0.53 nats (about 49%).
  - mixture proposal (0.9 p + 0.1 uniform): 4.1x variance, bias under 1%.
  - Oscillating z works only if the branch proposal follows the model; greedy oscillation is unusable.
- Joint NLL against passes (S4, exact H(Y|X) = 1.007 nats per state; see charts/s4_joint_nll.json):
  - one-shot: 4.58 at 4 passes;
  - B3b tau 0.999: 1.12 at 7.6 passes;
  - B3a top-1: 1.12 at 12.7 passes;
  - ARDM DP schedule: 1.96 / 1.47 / 1.27 / 1.03 / 1.016 at 8 / 12 / 16 / 32 / 48 passes.
  - The model fit is 0.456 against Bayes 0.452 nats per masked cell, so the "minimal Bayes under compute" frontier is reached at about 32-48 passes.

## Part B: M2 learned events (S6 6x6 unique puzzles, T = 1, 5 seeds, fee 0.005 nats/cell)

The pilot chose T = 1: B2-reg at 2 passes has 0.674 nats/cell, well above 0.05. At T = 2 (4 passes) it is 0.279.

| criterion | result |
|---|---|
| (i) planted survives 5/5 | **no**: de-listed in 5/5 at step 1000 (F3b) |
| (ii) learned slot G above fee in >= 4/5 | no: 0/5; learned G = -0.0035 (= -rate), constant events |
| (iii) end-to-end gain over B2-reg (reg2 control) | not passed: auction minus reg2 = +0.104 nats/cell, seed-t [-0.104, 0.313], hierarchical bootstrap [-0.026, 0.238]; against the pre-registered same-constant reg it is +0.057 [-0.135, 0.250]. The test events are constant, so no part of this gap comes from events |
| (iv) copy-hash / trivial / random de-listed >= 4/5 | yes, 5/5 each (but everything was de-listed) |
| (v) surviving learned slot NMI >= 0.5 with a non-cell atom | no: max NMI 0.00 (live), 0.00 (dead), 0.014 (init) |
| (vi) auction surviving G >= M2-cvae | no: -0.0107 vs 0.0000 (cvae prunes every constant slot) |

- Held-out royalty G_k per slot (nats/cell, 5 seeds): learned -0.0035, planted -0.0070, cell-bit -0.020, copy-hash -0.020, random -0.020, trivial -0.063.
  - Every reference slot's decoder gain is about 0.
  - The decoder never learned to read any slot within the 1000-step grace, and reading them was still not learned at 5000 steps (the grace-5000 diagnostic below). So every slot pays its rate and nothing more.
- Grace-5000 diagnostic (post-hoc): every reference slot was de-listed at the first check after grace (step 5000) in 5/5 seeds; decoder gains stayed about 1e-5, learned slots were still constant (G = 0), with 9 respawns per seed instead of 57. Test NLL 0.422 against reg2 0.550: +0.128, seed-t [-0.059, 0.315], hierarchical bootstrap [0.013, 0.243], so not passed on the pre-registered seed-t interval. The events are constant at test, so any real gap would come from training dynamics. One unverified guess: respawned slots act as random auxiliary targets that help optimisation (auction arms: 10/10 seeds below 0.6 nats/cell; cvae, reg and reg2: 9/15).
- Control fix (post-hoc): the pre-registered B2-reg used the same learned constants in both passes. M2's slot tokens differ between the prior pass and the decoder pass, so the shared block can tell which pass it is on; B2-reg cannot. reg2 is M2's own decoder with every slot de-listed, which gives the same phase signal. reg minus reg2 = -0.047 [-0.195, 0.101] nats/cell, so the phase signal made no difference on average. Both controls are in results/.
- Seed variance is large in every arm: per-cell NLL spans 0.31-0.66 (reg) and 0.37-0.57 (auction). A few seeds "click". Five seeds are the minimum for any M2 claim.

## Deviations (simplified; all stated)

- Hardware: the host load was about 96 on 16 cores, so S4, S6 and M2 ran on the GPU VM rather than CPU. Steps: S4 10k, S6 20k.
- Part A: S9 and branch.py were not built, because G1 failed (F1).
- The G1 target is the paired-baseline commit gain. The centred head and reg2 are post-hoc and labelled as such in the JSON.
- The planted atom was chosen by the bank's prior-resampled G (the spec asked for leave-one-out gain in the pilot). No atom had a positive gain.
- Grace (1000 steps) is my choice; the spec does not set it. Grace 5000 was run post-hoc.
- The fixed-cell positive controls (cvae_fix14, auction_g5000_fix14) are post-hoc and use 4 seeds, not 5.
- S6 has only 49 canonical solution classes under the allowed symmetries, so the class split gives 20374 train / 1245 val / 2381 test puzzles.

## Kelvin's questions, as far as P10 answers them

- **Private signals vs each variable as its own paid trader:** pricing a variable by its realised usefulness (part A) was no better than counting its masked peers. A per-variable price adds nothing over free proxies here.
- **Important vs learnable:** with a rate price, straight-through learned events went straight to the "trivial" corner (constant, rate 0, gain 0) in every seed; the copy corner was never reached. But a hand-made fixed-cell bit earned about 10x the fee and survived the auction in 4/4 seeds: a bounded decoder pays for a bit far above its code length (V-information). So "important" is priced correctly; the open problem is making events learnable before the rate term collapses them.
- **Oscillating z with void branches:** unbiased, but its variance is 4x (model-following proposal) to 47x (greedy) the signal. It is a readout, not a training signal.
- **Tower backwards to a minimal Bayes model:** on 4x4, the ARDM schedule reaches H(Y|X) within 0.01 nats at 48 passes. That is the known result, and nothing deeper was found.

## Files

- Code: grids.py, models.py, train.py, train_s6.py, auction.py, stats.py, eval.py, queue_p10.py, gpu.sh, startup.sh.
- Results: results/g1.json, s4_readouts.json, s6_T_choice.json, m2_summary.json (reg2 control), m2_summary_regsame.json (pre-registered same-constant control), m2_summary_g5000.json, m2_summary_fix14.json (auction positive control), m2_arms_posthoc.json (test NLL of every arm, cvae positive control); raw per-run JSON in vm_out/results/.
- Charts: charts/s4_g1.json, s4_joint_nll.json, m2_royalties*.json, m2_auction_traj*.json, m2_arm_nll.json.
