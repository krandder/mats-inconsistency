# Market ResNet phase 2: audit of the headline

Date: 2026-09-29. Sources: four independent adversarial audits (two judged the claim refuted as stated, two judged it not refuted, all four with severity major or minor), plus a read of the phase-3 results that had landed by 19:45Z (results/vm/trader-mresnet-10/p3/*.json).

Claim audited (RESULTS.md section 16, CATALOGUE E-MR-5/6/12): a purely economic network (traders trained on their own market payoffs, no backprop across depth) scores 0.93 at 100x the training length, while the best backprop-through-depth arm scores 0.12 even with the same covariance targets. At 10x the covariance targets explain most of the gap (0.98 vs 1.00). The economy trains at depth 10,000 in 4.2 GB while backprop runs out of memory at 1,000. VOC halting beats random halting by 0.13 to 0.24. Belief logits run away.

**Verdict: the numbers are real. What the claim says causes them is not supported.**

## 1. What reproduces

All four audits recomputed these from the raw per-seed JSON (results/vm/*/p2/*_s[0-9].json). Checkpoint hashes match across VMs, and CPU re-evals of small cases match.

| quantity | value | note |
|---|---|---|
| economy (econ_cov_voc), L3200, 3200 steps | 1.000 / 0.900 / 0.900, mean 0.93 +- 0.14 | 40 test strings, the same 40 in every seed |
| bptt_cov_dscov, same point | 0.000 / 0.075 / 0.275, mean 0.12 +- 0.35 | the economy's worst seed beats dscov's best seed; 112/120 vs 14/120 |
| L320, 320 steps | economy 1.00 (x3), dscov 0.970 / 0.965 / 1.000 | |
| L32000, 3200 steps | economy 1 of 8 in each seed; every BPTT arm 0 | only 8 strings |
| memory (memdepth.json) | online economy 4.20 GB at D=L=10000 (one step, 1948 s); BPTT out of memory at 1000 (21.4 GB peak) | see objection 3 |
| VOC halting minus random-same-histogram | 0.13 to 0.24 (per seed as low as 0.08) | see objection 5 |
| belief log loss, L3200 | 501 nats/bit (seeds 0.0, 575, 929) | accuracy scores only the sign, and all values are finite, so this does not inflate accuracy |

## 2. Strongest objections, ranked

1. **The "economic" training is detached one-step backprop.** A trader's pay is soft_score(b + s*db) - soft_score(b) (plus the same on c), with b recorded under no_grad (econ2.py train_step, about lines 261-318). Its gradient is the gradient of per-step log loss plus covariance loss on a state detached at every step: truncated BPTT with k=1 and deep supervision. Two audits checked this on CPU: cosine 1.000000 and max abs difference 5e-6 against a plain supervised detached gradient (no VOC; cap off or not binding). The only economic term that touches gradients is the wealth scale s, and it never binds: final wealth is 2,787 to 2,948 from W0 = 20, and nocap scores 0.92. So "purely economic" and "no backprop across depth" name the wrong mechanism. This is the one-step-gradient ingredient of HRM and Bansal's progressive loss, which THEORY.md K8 already names.
2. **The comparison has three uncontrolled differences besides detachment.** (a) The economy trains in random trader order; BPTT trains round-robin (both are tested round-robin). (b) Every reported economy has voc=1, so its v stream is paid on VOC targets; in every BPTT arm, dscov included, v is a free latent. "Same covariance targets" is true, but "same targets" is not. (c) The economy uses one Adam optimizer and gradient clipping per trader; BPTT uses one global optimizer and clip. **The early phase-3 results (seed 0 only, 40 strings) already show order matters a lot.** The same economy trained round-robin scores 0.325 at L3200 (0.905 at L320). Trained in random order but without VOC, it scores 0.80. So random training order carries much of the gap, and VOC is not required.
3. **The memory sentence compares a different model.** The 4.2 GB is one step of train_step_online, which asserts voc=0. That configuration was never trained to completion or evaluated for accuracy. The procedure that produced 0.93 (batched train_step, voc=1) also runs out of memory at D=1000 (14.6 GB), just like BPTT. memdepth sets L=D, so depth and length are mixed. Memory that stays flat with depth is generic to any truncated or detached update. No TBPTT(1) or gradient-checkpointing arm was run.
4. **The BPTT baselines are weak.** Bansal et al. 2022 report about 96-97% at 16x the training length with plain or progressive BPTT (width 400, 150 epochs, tuned schedule). Here every BPTT arm without covariance targets gets 0.00 at 10x (width 96, 3000 iterations, no tuning on either side). The progressive-loss arm used alpha=1 only, with no deep supervision or covariance targets.
5. **The halting result is against a weak baseline and mislabelled.** The test uses mixed lengths 4 to 64 (prefix_batch(n, 4, 64)), not "L64". Against fixed depth at the same mean steps it is within noise. Plain entropy-threshold halting, with no market at all, matches VOC within 0.02 at every cost (one audit's CPU check: 3 seeds x 200). By construction VOC = sigmoid(v0) x belief entropy / 5.
6. **"At 10x the covariance targets explain most of the gap" depends on the depth picked.** It holds at D=320. At shallower matched depths the economy leads by far more: 0.88 vs 0.50 at D=24, 0.98 vs 0.84 at D=64.
7. **The runaway-logit explanation is wrong.** The runaway is real, but section 16's cause ("once a trader is rich enough to push it") does not fit: the cap never binds, nocap runs away too (1,277 nats/bit), and BPTT runs away more (5,900 to 11,000). The cause is unbounded additive logit updates past the trained regime.
8. **The test sets are small**: 40 strings at L3200 and 8 at L32000. The seed-level gap survives, but the +-intervals leave out sampling error over strings. The task is only prefix-sum parity, with a hand-built dyadic (Kogge-Stone style) module layout that both arms share.

## 3. Minimal honest claim

On prefix-sum parity, with weight-tied dyadic-offset modules and segment-parity auxiliary targets, training each recurrent step on its own loss from a detached incoming state, in random module order, reaches 0.93 exact-string accuracy at 100x the training length (3 seeds, 40 strings). Full BPTT through the 24 training steps with the same per-step and covariance losses, trained in fixed order, reaches 0.12. At 10x both work (1.00 vs 0.98). The market payoff is exactly this detached per-step objective (the gradients are equal). Wealth, cap, bids and auction add nothing or hurt. Early phase-3 evidence points to random training order as a large share of the effect. Both arms become extremely overconfident past the training length.

## 4. Status labels now

- Headline as worded ("purely economic network beats backprop"): **not supported**. The numbers are verified. The attribution to economics is refuted by the gradient equivalence, and the rest of the attribution is unisolated.
- E-MR-5: Gain -> **Informative** (a real detached-vs-full-BPTT effect on one task, confounded with training order and VOC, and of a known kind).
- E-MR-12: Gain -> **Informative** (a known property of truncated or detached updates, measured on a configuration with no accuracy result).
- E-MR-6: Gain -> **Informative** (confidence halting works; no market-specific value shown).
- RESULTS.md section 16 and the TL;DR should state what the payoff is mathematically and drop the wealth explanation of the runaway.

## 5. Experiments that settle it

Protocol for every arm: 3 seeds, 3000 iterations, batch 64, D=24; test with reeval.py (fresh set, n=200 at L3200, depths 320/1600/3200) plus L320.

Already queued or running in phase 3 (trader-mresnet-10..15; only seed 0 of the first two economy arms has landed):

| arm | question it answers |
|---|---|
| bptt_cov_dscov_tb1_rand | Does detached BPTT(1) + ds + cov in random order reach about 0.9? If yes, the economy adds nothing beyond relabelling. |
| bptt_cov_dscov_tb1 (round-robin) | Detachment alone, without the order effect |
| bptt_cov_dscov_rand (full BPTT, random order) | If this extrapolates, the cause is order randomization, not locality |
| econ_cov (voc=0), econ_cov_voc_rr | Separate VOC and order inside the economy (seed 0: 0.80 and 0.325) |
| bptt_cov_progcov | Bansal progressive loss + covariance targets |
| dtr_full / dtr_eqtime / dtr_eqflop | A published-strength DT-Recall baseline |

Still missing, needed to close it:
1. bptt_cov_dscov with v trained on the same VOC targets (or v frozen), in random order.
2. Bansal progcov with alpha swept over {0, 0.5, 1}; the queued arm is alpha=1 only.
3. memdepth at fixed L=32 with depth varied, for TBPTT(1), checkpointed BPTT, the online economy and the batched voc=1 economy. Also train the online (voc=0) economy and report its accuracy, so the memory row belongs to a model that works.
4. Halting at a single length of 64, with an entropy-threshold baseline and led by the fixed-depth comparison.

Decision rule: if tb1_rand, with or without VOC-supervised v, reaches about the economy's L3200 accuracy on the fresh 200-string set, report the result as "detached per-step training in random order extrapolates" and retire the economics attribution. If only the economy reaches it, name the remaining difference (per-trader optimizer and clipping) and test it directly.
