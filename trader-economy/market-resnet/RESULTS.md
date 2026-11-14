# Market ResNet: results

Status: done. Phases 1 and 2: nine VMs, trader-mresnet-1..9, deleted by 2026-09-29 18:46 UTC. Phase 3 (the ablation, the Bansal baseline, the regularizers and the audit's controls, section 17): sixteen VMs, trader-mresnet-10..25, each deleted after a sha256-verified sync. Check: `gcloud compute instances list` and `gcloud compute disks list` (project GCP_PROJECT, filter name ~ ^trader-mresnet-) both empty, checked at 2026-09-30 00:47 UTC.

**Audit (2026-09-29, `../market-resnet-AUDIT.md`).** The phase-2 market payoff is exactly detached one-step backprop with a loss at every step (truncated BPTT, k = 1, plus deep supervision): the gradients are equal (per-trader cosine 1.0000000, `eqv4.py`, `results/eqv4.txt`, also with the VOC and bid heads). Sections 8, 9 and 10 were relabelled from Gain to Informative and the headline reworded to the minimal claim below; section 16's explanation of the runaway was corrected.

## TL;DR

- **Phase 1 (backprop through depth, per-step market pay).** Paying every step its log-score change extrapolates to 4 times the training length in every seed (L128: 1.000), where the final-loss net (0.11) and progressive loss on the same block (0.64; Bansal et al. 2022's training method on our block, not their DT-recall architecture, which extrapolates to 100x: section 17.3) do not; deep supervision does as well, so the gain is from scoring every step, not from the market form.
- **Priced halting meets the bar out of distribution only.** On mixed lengths it reaches 1.000 at 17.7 steps where random halting at the same steps gets 0.54 and PonderNet 0.75 to 0.96; in the training range PonderNet ties. A real training cost (0.01 to 0.2) works as a price but adds nothing over pricing a cost-0 model after training.
- **Settlement.** Settling only the steps that ran (the decision-market bug) halts too early (0.95 to 0.97); counterfactual or epsilon-continuation settlement (Chen et al. 2014) fixes it. Aumann-Shapley credit fails its completeness check (Problem).
- **Phase 2, minimal honest claim (after the audit).** On prefix sums, training each recurrent step on its own loss from a detached incoming state, in random module order, extrapolates to 100 times the training length: 0.93 exact-string accuracy at L3200 (3 seeds, 40 strings) against 0.12 for full BPTT through the 24 training steps with the same per-step and covariance losses, trained in fixed order; at 10 times both work (1.00 against 0.98). The market payoff is a relabelling of that detached per-step objective; wealth, cap, bids and auction add nothing or hurt. Both arms become extremely overconfident past the training length. The 4.2 GB at depth 10,000 is from the online voc=0 update (section 9); trained in phase 3, that update scores 0.88 at 100x, and flat memory in depth holds for any streaming one-step update, market or not (section 17.6).
- **Economic halting and funding.** VOC-priced halting beats random halting at matched steps by 0.13 to 0.24 (mixed lengths 4 to 64), but VOC is sigmoid(v0) x belief entropy / 5 and a plain entropy threshold matches it (audit). Problems: auction bids are 23 times too high, so auction order and bid halting fail; private slots with rent go bankrupt; voluntary funding of the covariance markets free-rides (Lindahl pricing fixes it); belief logits run away past the training length.
- **Phase 3 (after the audit, section 17): the economics attribution is retired.** Random training order is the main ingredient: the same economy trained in round-robin order scores 0.40 at 100x, and full BPTT trained in random order 0.79 (0.12 in round-robin order). On 200 fresh strings per seed at L3200, BPTT truncated to one step in random order gets 482/600 [0.77, 0.83], and with per-trader Adam and clipping (the economy's exact gradient twin) 529/600 [0.85, 0.91], overlapping the economy's 557/600 [0.90, 0.95]. The strongest fair baseline, Bansal et al. 2022 DT-recall with progressive loss, beats the economy at 100x (589/600 [0.97, 0.99]; 600/600 [0.99, 1.00] at the economy's training FLOPs), so the 0.93 against 0.12 gap is not a Gain. The logit runaway costs about 5 points: clamping the logits at evaluation only lifts the economy to 0.98. A logit cap of +-10 in training gives 1.00 at 100x with log loss 0.000, and the same cap without a market 0.99. The epsilon floor, the L2 fee and Kelvin's liquidity that grows with depth (b_t = t, sqrt t) cut the log loss at 100x (0.1 to 92 nats per bit against 501) but lose accuracy there (0.34 to 0.71); the trade-size cap fails both ways. VOC halting equals an entropy threshold at L64, and both beat fixed depth read at matched steps.
- Two cost-bookkeeping bugs were found and fixed; neither changed a reported result. Spend: est. $16.72 of the $20 cap (phases 1 and 2 $4.26, phase 3 $12.46; phase 3 is logged in Kelvin's 24-hour budget, `../../BUDGET-24H-20260929.md`).

## Setup

Task: prefix sums of a bit string (output bit i = parity of bits 0..i), as in Schwarzschild et al. 2021.
Training uses lengths 8 to 32; tests go to 64 and 128 (phase 1) and to 32000 (phase 2).
Accuracy is exact-string accuracy: every output bit right.

**Phase 1** is one recurrent residual block (width 128) applied T times, with three streams: x (the input, re-injected every step), b (the belief logits, one per output bit) and h (scratch).
Each step's change to b is a trade in a log-score market: the step is paid the change in log score it causes (Hanson 2003), in nats per example.
All phase-1 arms train by backprop through depth, T = 24 steps.

| arm | training signal |
|---|---|
| A | final-step log loss only (fixed steps) |
| B | final loss + every step paid its own log-score change (per-step market scoring rule, earlier belief held fixed) |
| Bds | final loss + the log loss of every step's belief (deep supervision, Lee et al. 2014) |
| C | B + a gain head that predicts the next step's pay; at test time an example halts when predicted gain <= cost (priced halting) |
| Ctt | C with halting during training (cost 0.05 after 20% of training); halted steps are settled on their counterfactual pay |
| Ctt_real | Ctt settled on the realized pay, which is 0 after a halt (self-fulfilling; kept as the buggy comparison) |
| Ctt_exec | Ctt settled only on steps that ran (the decision-market bug: no data after a halt) |
| Ctt_eps | Ctt with epsilon-continuation: 10% of halted examples keep running, and their steps settle with weight 1/eps (Chen et al. 2014) |
| D, D_lp0.2 | PonderNet (Banino et al. 2021), prior 1/12 and 0.2, standing in for ACT (Graves 2016) |
| E | random halting with the same step histogram as the priced rule (E_perm) or geometric (E_geom); post hoc, no training |
| P | Bansal et al. 2022 progressive loss (the extrapolation bar) |

Seeds: 3 for every arm (A, B, Bds, C, Ctt, Ctt_real, Ctt_exec, Ctt_eps, Ctt_eps at training costs 0.01 and 0.2, D, D_lp0.2, P); 10000 training iterations, batch 128, lengths 8-32 (`train.py`, `run_grid*.sh`).
Numbers are means with 95% t-intervals over seeds; (n=1) marks single runs.
Test sets: id_8_32 (lengths 8-32, the training range), mix_8_64 (lengths 8-64, half of them unseen), L64 and L128, each run to a step cap of twice the longest length (128 steps for mix_8_64 and L64, 256 for L128).

**Phase 2** removes backprop across depth: see its section.

## Phase 1 findings

### 1. Extrapolation to longer inputs: Gain

Exact accuracy at the step cap:

| arm | L64 | L128 |
|---|---|---|
| A fixed steps | 0.63 ± 0.80 (seeds 0.50, 1.00, 0.40) | 0.11 ± 0.39 |
| B per-step market | 1.000 | 1.000 |
| Bds deep supervision | 1.000 | 1.000 |
| C priced halting | 1.000 | 1.000 |
| Ctt train-time halting | 1.000 | 0.999 |
| D PonderNet | 0.007 | 0.000 |
| D_lp0.2 PonderNet | 0.003 | 0.001 |
| P Bansal et al. 2022 | 0.99 ± 0.03 | 0.64 ± 0.82 (seeds 0.78, 0.87, 0.26) |

Paying every step for its own improvement (B, C) extrapolates to 4x the training length in every seed; the final-loss-only model (A) does not, and the Bansal et al. 2022 bar (P) holds at 2x but not at 4x.
The success bar "extrapolates at least as well as A" is met, and C also beats P at L128 (1.000 vs 0.64, 3 of 3 seeds).
Deep supervision (Bds) does as well, so this gain comes from scoring every step, not from the market form of the score.
PonderNet collapses when run past its halting point (its best accuracy at L64 is 0.33 ± 1.40 and 0.80 ± 0.77 at some earlier step), the "overthinking" failure Bansal et al. describe.

### 2. Monotone improvement: Informative

Fraction of examples whose log score never drops by more than 0.001 nats at any step up to the cap:
L64: B 1.00, Bds 1.00, C 1.00 (A 0.45, D 0.00).
L128: B 0.50 (seeds 0.02, 0.55, 0.94), Bds 0.75, C 0.66 (seeds 0.00, 1.00, 0.99).
At 4x the training length, some seeds make small losing trades (C seed 0: 0.84 nats of negative pay per example in 256 steps) but accuracy stays at 1.0.
Per-step payment does not guarantee monotone improvement far out of distribution.

### 3. Priced halting at equal compute: Gain out of distribution, no gain in distribution

Operating point = the cheapest cost whose accuracy is within 0.005 of the best; every other policy is read at the same mean number of steps.

| test set | C steps | C acc | random, same histogram | geometric random | A fixed | own fixed depth | PonderNet D | PonderNet D_lp0.2 |
|---|---|---|---|---|---|---|---|---|
| id_8_32 | 9.9 | 0.999 | 0.58 | 0.46 | 0.44 | 0.58 | 1.000 | 1.000 |
| mix_8_64 | 17.7 | 1.000 | 0.54 | 0.43 | 0.45 | 0.54 | 0.75 ± 0.33 | 0.96 ± 0.06 |
| L64 | 32.2 | 0.999 | 0.99 | 0.39 | 0.13 | 0.99 | 0.10 ± 0.42 | 0.50 ± 0.14 |

In the training range PonderNet matches priced halting: it reaches C's accuracy in 10.2 ± 0.7 steps against C's 9.9.
On mixed lengths with unseen ones, priced halting reaches 1.000 at 17.7 steps where random halting with the same step histogram gets 0.54 and PonderNet 0.75 to 0.96.
At a single length (L64) per-example halting has nothing to adapt to: random halting with C's histogram and C's own fixed depth are as good (0.99); the gain there is only that C's gain head still picks the right depth for an unseen length and PonderNet's does not.
Ctt (train-time halting at cost 0.05) gives the same frontier as C (mix_8_64: 17.8 steps, 0.999), and so does Ctt_eps (17.7 steps, 0.998).
PonderNet here is read off its own accuracy-vs-mean-steps curve (all halting thresholds q), so it gets its best threshold for each compute level.

### 4. Nonzero training cost (Ctt_eps trained and halted at 0.01, 0.05, 0.2): Informative

Priced halting with a real cost in training: Ctt_eps trains with halting at cost c (after 20% of training, epsilon-continued, 1/eps settlement) and is tested halting at the same c.
Every other policy is read at the same mean steps; C is the cost-0 model priced after training at the same c.

| test set | cost c | steps | acc | random, same histogram | PonderNet D | PonderNet D_lp0.2 | A fixed | C priced after training (steps, acc) |
|---|---|---|---|---|---|---|---|---|
| mix_8_64 | 0.01 | 26.9 ± 31.1 | 1.000 | 0.58 | 0.77 | 0.97 | 0.67 | 34.7, 1.000 |
| mix_8_64 | 0.05 | 21.0 ± 7.5 | 1.000 | 0.57 | 0.77 | 0.98 | 0.55 | 20.4, 1.000 |
| mix_8_64 | 0.2 | 17.8 ± 0.2 | 1.000 | 0.54 | 0.75 | 0.96 | 0.46 | 17.9, 1.000 |
| L64 | 0.01 | 69.6 ± 97.4 | 1.000 | 1.00 | 0.25 | 0.71 | 0.55 | 93.3, 1.000 |
| L64 | 0.05 | 56.7 ± 23.4 | 1.000 | 1.00 | 0.25 | 0.75 | 0.57 | 61.6, 1.000 |
| L64 | 0.2 | 33.1 ± 0.7 | 0.986 ± 0.059 | 0.98 | 0.18 | 0.68 | 0.15 | 34.7, 1.000 |

The price works as a price: a higher cost buys fewer steps, and accuracy stays at 1.000 except at L64 and cost 0.2 (0.986; one seed at 0.96).
At equal compute, priced halting beats random halting with the same step histogram by 0.42 to 0.46 on mixed lengths, PonderNet (prior 1/12) by 0.23 to 0.25, and PonderNet (prior 0.2) by 0.02 to 0.04.
Training with the cost adds nothing over pricing a cost-0 model after training: same accuracy at similar steps.
Extrapolation holds with a training cost: L128 at the cap is 0.97 ± 0.13 (c = 0.01; one seed 0.91), 1.000 (c = 0.05) and 1.000 (c = 0.2).

### 5. Settlement of halted steps (the decision-market bug): Problem for executed-only settlement; fixed by counterfactual or epsilon settlement

All trained with halting at cost 0.05; tested on L64 (unseen length) at halting cost c.
Post-halt gain = log score the example would still have earned after its halt (nats per example; 0 means the halt was right).

| settlement | c = 0.05: steps, acc, post-halt gain | c = 0.2: steps, acc, post-halt gain |
|---|---|---|
| Ctt: counterfactual pay | 34.5, 1.000, 0.000 | 33.2, 0.998, 0.004 |
| Ctt_eps: epsilon-continuation, 1/eps weights | 56.7, 1.000, 0.000 | 33.3, 1.000, 0.000 |
| Ctt_real: realized pay (self-fulfilling) | 34.7, 1.000, 0.000 | 33.3, 0.999, 0.001 |
| Ctt_exec: only steps that ran | 32.7, 0.970 ± 0.055, 0.040 | 32.2, 0.948 ± 0.094, 0.090 |

Settling only the steps that ran halts too early at unseen lengths: the gain head never sees what happens after a halt, so a wrong early stop is never corrected (Chen et al. 2014).
Counterfactual settlement and epsilon-continuation both fix it; epsilon-continuation is more conservative at low cost (56.7 vs 34.5 steps at c = 0.05) with the same frontier (section 3).
The self-fulfilling realized-pay version did not hurt here (3 seeds): the gain head learns before halting starts at 20% of training, and on this task a halted example rarely had gain left.

### 6. Halting calibration: Informative

Correlation of the gain head's prediction with the realized pay, over all steps to the cap: C 0.91 ± 0.11 on mix_8_64, 0.91 ± 0.06 on L64, 0.73 ± 0.37 on L128.
Within the training depth (24 steps) at L64 it is 0.35 ± 0.39: at an unseen length the head is well calibrated late (where it matters for stopping) and poorly early.

### 7. Aumann-Shapley credit: Problem

Integrated-gradients credit (Sundararajan et al. 2017) of each stream's writes to later payments: the h stream gets 2-4% of the credit (A 0.044, B 0.021, C 0.031), and a step's own b write gets 2.7 to 3.1 times the total, so other components carry large negative credit.
The completeness check fails: at the worst step the credits miss the payment they explain by 0.46 ± 1.25 (C) and 0.75 ± 0.82 (B) nats, against 0.05 for A.
With 64 integration points the path integral is not accurate for B and C, so these credit numbers are not reliable; they are reported, not used.

## Phase 2 findings

**Setup.** `econ2.py` has no backprop across depth: each trader's pay gradient is backprop through one step from a detached state, the same as a per-step loss (audit, section 8).
Four traders, each a small per-position network with its own weights and its own Adam optimizer, shared across levels j so that it applies at any length: at level j a trader reads x, b and stream 3 at offsets 0 and +-2^j.
A trader reads detached state and trains only on its own profit: log-score earnings on every market it moves, minus the compute cost, minus slot rent, slot payments or bounty charges where those are on.
Wealth ledger: each trader starts at W0 = 20, a trade is scaled down so its worst-case loss is at most half the trader's wealth, and a trader at wealth <= 0 is replaced by a fresh one (a respawn).
Stream 3 is one of: `cov`, covariance markets (at level j, "is y_i = y_(i+2^j)?", settled on the realized outcome); `slots`, private banks whose owner pays writers their Aumann-Shapley share of its earnings and pays rent; `none`.
VOC markets (k = 5, 50, 500) price the share of each position's log loss that the next k steps remove; halting stops an example when VOC <= cost.
The BPTT comparison uses identical traders and streams with c and v as free latents, trained end to end through depth: `bptt_cov` on the final-step log loss, `bptt_cov_prog` with the Bansal et al. 2022 progressive loss, `bptt_cov_ds` on every step's log loss (deep supervision, which separates per-step signal from locality), and `bptt_cov_dscov` on every step's log loss plus the covariance targets on c at every step (the economy's covariance markets pay on these targets; in the other BPTT arms c is a free latent).
Training: 3000 iterations, batch 64, D = 24 steps, lengths 8-32; the economy in random trader order, every BPTT arm in round-robin order (the code's default for bptt; this line said random for both before the audit found the confound).
Test: round-robin order (a fixed rotation, no bids) and auction order (the highest bid acts), at L = 32, 320, 3200, 32000 with 1000, 200, 40 and 8 examples.
Seeds: 3 for econ_cov_voc, bptt_cov, bptt_cov_ds and bptt_cov_dscov, 2 for the rest (`run_p2.sh`, `run_p2b_vm6.sh`, `run_p2post.sh`, `run_p2cov.sh`).
Equal compute: at D = 24 the economy trains in 657 s against 772 +- 189 s for bptt_cov (bptt_cov_ds 694 +- 12 s, bptt_cov_dscov 754 +- 97 s), same iterations, batch and data, peak memory 0.28 against 0.29 GB.

### 8. Depth and length extrapolation without backprop across depth: Informative (was Gain): real at 100 times the training length, but the training is detached per-step backprop, not economics

Exact-string accuracy, round-robin order for the economy:

| run | L32, D=24 | L32, D=64 | L320, D=320 | L3200, D=3200 | L32000, D=3200 |
|---|---|---|---|---|---|
| econ_cov_voc | 1.00 | 1.00 | 1.00 | 0.93 +- 0.14 | 0.12 (1 of 8, every seed) |
| econ_cov_voc_nocap | 1.00 | 1.00 | 1.00 | 0.92 +- 0.32 | - |
| econ_cov_voc_grow | 1.00 | 1.00 | 1.00 | 0.99 +- 0.16 | - |
| econ_cov_voc_bountylindahl | 1.00 | 1.00 | 1.00 | 1.00 | - |
| bptt_cov (final loss) | 0.97 +- 0.13 | 0.63 +- 1.25 | 0.00 | 0.00 | 0.00 |
| bptt_cov_prog (Bansal) | 0.86 +- 1.29 | 0.76 +- 1.04 | 0.00 | 0.00 | 0.00 |
| bptt_cov_ds (every step's loss) | 0.99 +- 0.03 | 0.65 +- 0.56 | 0.00 | 0.00 | 0.00 |
| bptt_cov_dscov (every step's loss + covariance targets) | 1.00 | 1.00 | 0.98 +- 0.05 | 0.12 +- 0.35 (seeds 0.00, 0.08, 0.28) | 0.00 |
| econ_none | 0.00 | 0.00 | 0.00 | 0.00 | - |
| bptt_none | 0.28 +- 0.51 | 0.07 +- 0.79 | 0.00 | 0.00 | - |

The economy keeps improving with depth far past its training depth of 24: at L3200 it scores 0.62 at 64 steps, 0.78 at 320, 0.88 at 1600 and 0.93 at 3200 (133 times the training depth, 100 times the training length).
At L32000 it reaches 0.12 at 3200 steps, one of 8 examples in each seed; more depth there was not run, so 1000 times the training length is untested.
The same traders trained end to end on the final loss fail at L320 at every depth and lose accuracy past their training depth even at L32 (0.97 at 24 steps, 0.63 at 64).
Two controls separate what the economy has from what backprop through depth lacks (the phase-1 backprop arms with per-step signal, B, Bds and C, did extrapolate to 4 times the length, section 1).
Per-step signal alone is not enough: bptt_cov_ds, trained on every step's log loss down to 1e-5 nats per bit, still loses accuracy past its training depth at L32 (0.99 at 24 steps, 0.65 at 64) and scores 0.00 at L320 and beyond at every depth.
The covariance targets explain most of the gain at 10 times the length at D = 320 (at shallower matched depths the economy leads by more: 0.88 against 0.50 at D = 24, 0.98 against 0.84 at D = 64; audit): the economy's stream 3 is paid on "is y_i = y_(i+2^j)?", while in bptt_cov and bptt_cov_ds c is a free latent.
Given the same targets at every step, backprop through depth (bptt_cov_dscov) holds 1.00 at L32 and 64 steps and reaches 0.98 at L320 and 320 steps (economy 1.00).
At 100 times the length only the economy works: bptt_cov_dscov scores 0.12 at L3200 (0.00, 0.08 and 0.28 by seed) with its belief logits exploding (log loss 5,900 to 11,000 nats per bit at 3200 steps), against the economy's 0.93.
What makes the difference at L3200 is not the market: the pay gradient equals the gradient of each step's log loss plus covariance loss from a detached incoming state (truncated BPTT with k = 1 and deep supervision; `eqv4.py`), and the wealth scale never binds (nocap: 0.92). Three differences from bptt_cov_dscov remain besides detachment: random against round-robin training order, VOC targets on v for the economy only, and per-trader Adam and clipping. Section 17 tests each.
Caveat: the economy's belief logits run away at unseen lengths. Mean log loss is 501 nats per bit at L3200 and 3200 steps even though 93% of examples are exactly right, so the wrong bits are held with extreme confidence (section 16).

### 9. Memory of training at depth (the infeasibility claim): Informative (was Gain): a property of any detached update, measured on a configuration with no accuracy result

`memdepth.py` measures the peak GPU memory and time of one training step at depth D = L (batch 16, one L4 with 24 GB) for the economy's online update (each step's trades settle and update as they happen, so nothing is kept across steps), its batched update (keeps the whole rollout, no graph) and BPTT through the same traders.

| D = L | econ online: GB, s | econ batched: GB | BPTT: GB |
|---|---|---|---|
| 32 | 0.006, 0.8 | 0.07 | 0.09 |
| 320 | 0.09, 2.5 | 7.4 | 15.9 |
| 1000 | 0.30, 13 | out of memory | out of memory |
| 3200 | 1.16, 163 | - | - |
| 10000 | 4.20, 1948 | - | - |

BPTT through the traders cannot train at depth 1000 on this GPU; the online economy runs a training step at depth 10000 in 4.2 GB, and its memory grows only with the length of the input (here L = D), not with depth.
Audit: the online rows are `train_step_online` with voc=0, a configuration never trained to completion or evaluated for accuracy; the batched voc=1 economy behind the 0.93 runs out of memory at D = 1000 like BPTT; and memdepth.py sets L = D, which mixes depth with length. Flat memory in depth is generic to truncated or detached updates. Section 17 repeats the measurement at fixed L = 32 with truncated (k = 1) and checkpointed BPTT.
Limits: gradient checkpointing (Chen et al. 2016) would let BPTT go deeper at extra compute, and truncated BPTT would too at the cost of long-range credit; both were run in phase 3 at fixed L = 32 (section 17.6).
The economy was trained only at depth 24; the improvement past that depth in section 8 is at test time.
So the runs show that training at depth 10000 is feasible for the economy, not that training there improves it further.

### 10. Priced halting in the economy (VOC markets): Informative (was Gain): beats random halting and fixed depth at matched steps, matched by an entropy threshold

Halting test: mixed lengths 4 to 64 (`prefix_batch(n, 4, 64)`; this line said L64 before the audit), 500 examples, round-robin order, step cap 96; an example halts when its VOC <= c.
Audit: by construction VOC = sigmoid(v0) x belief entropy / 5, and a plain entropy threshold on the belief, with no market, matches VOC halting within 0.02 at every cost (3 seeds x 200, CPU). Section 17 reruns the test at L64 only with an entropy-threshold baseline.
Correction (phase 3, section 17.7): the fixed-depth columns below round the mean steps up (for example 7 steps for a mean of 6.6, `eval_halting` in `econ2.py`), which gives fixed depth extra compute. Read at exactly the matched mean steps (linear interpolation, `halt64.py`), fixed depth scores 0.652 +- 0.047 at c = 0.01 against 0.859 for VOC halting, so VOC and entropy halting both beat it.
The cost-0 run is econ_cov_voc (3 seeds) halted post hoc at c; the trained runs train with cost c and halting at c, settled with epsilon-continuation (eps = 0.1, 1/eps weights), 2 seeds each.
Random and fixed-depth columns are read at the same mean steps as the row.

| c | cost-0 run: steps, acc | random at matched steps | fixed depth at matched mean | trained at c: steps, acc | random | fixed depth |
|---|---|---|---|---|---|---|
| 0.001 | 9.6 +- 2.2, 0.994 | 0.836 | 0.985 | 14.0 +- 58, 0.994 | 0.863 | 0.992 |
| 0.003 | 7.6 +- 0.6, 0.980 | 0.743 | 0.876 +- 0.169 | 8.1 +- 5.0, 0.975 | 0.763 | 0.957 |
| 0.01 | 6.6 +- 0.2, 0.859 | 0.634 | 0.753 +- 0.129 | 6.9 +- 5.1, 0.859 | 0.668 | 0.629 |

Priced halting beats random halting at matched compute by 0.13 to 0.24 at every cost.
It beats fixed depth at the same mean steps in every row, but the fixed-depth intervals overlap it.
Training with the cost gives the same accuracy at slightly more steps than pricing the cost-0 run post hoc, as in phase 1 (section 4).
There is no ACT or PonderNet arm in phase 2: both need backprop across depth; the phase-1 comparison with them is section 3.
The VOC prices are calibrated: correlation with the realized value 0.82, mean predicted 0.0065 against realized 0.0067 nats per bit per step.
The auction bids are not: correlation 0.20, mean predicted 0.126 against realized 0.0055, 23 times too high, so bid halting almost never fires (85 to 90 of 96 steps at every cost) and gains nothing over random.
A likely cause, not tested, is the winner's curse (Capen, Clapp & Campbell 1971): the auction picks the highest of four noisy bids.
Cost 0.01 hurts depth extrapolation in one seed: at L320 seed 1 falls from 1.00 at 64 steps to 0.51 at 320 and 0.00 at 640; seed 0 stays at 1.00.

### 11. Settlement of halted steps in the economy: Informative, no harm from the self-fulfilling version

Both runs halt during training at cost 0.002.
`haltreal` settles halted steps on their realized value (the self-fulfilling version); `halteps` settles with epsilon-continuation.
At c = 0.001 haltreal takes 7.8 +- 2.6 steps at accuracy 1.000, halteps 14.1 +- 56 steps at 0.992; in round-robin order both stay at 0.99 or better at L320 to 640 steps.
As in phase 1 (Ctt_real), the self-fulfilling version showed no harm here; the harm needs settlement on executed steps only (phase 1, Ctt_exec).

### 12. Auction order at depth: Problem

With the highest bid choosing who acts, accuracy is fine at the training length (0.99 at L32, 32 steps) but erratic at L320: per seed 0.99, 0.02 and 0.66 at 320 steps in econ_cov_voc, and 0.00 in both seeds of the halteps, bountylindahl and all six priced runs.
Round-robin order, which ignores the bids, extrapolates in every cov run.
The cause is the miscalibrated bids of section 10.

### 13. Wealth: cap, no cap, entry and bankruptcy: Informative

In every house-funded cov run all four traders end rich and equal: final wealth 2,770 to 2,950 from a start of 20 (the house subsidizes the market maker), largest share 0.25.
So the wealth cap never binds, and removing it (nocap) changes nothing measurable (L3200 0.92 against 0.93).
With a new trader every 500 iterations (grow, 5 entrants), later entrants earn less (final wealth 2,009 for an original trader down to 79 for the last entrant), the largest share is 0.18, and accuracy is unchanged (L3200 0.99 +- 0.16).
No cov run had a respawn except bountylindahl seed 0 (2).

### 14. Private slots with Aumann-Shapley payments and rent: Problem

Training log loss ends at 0.641 nats per bit (chance is 0.693) with zero accuracy everywhere; each run has 71 respawns, and traders end with wealth 1 to 15, below their starting 20.
Rent and payments bankrupt traders faster than they learn, so no useful stream 3 forms.

### 15. Bounties for stream 3 (phase 2b): voluntary funding Problem, Lindahl pricing Gain

When traders, not the house, must pay for the covariance markets:
- **vol** (each trader chooses per episode whether to fund, trained on its own profit): funders pay 0.125 per episode, most cov markets go unsettled, and accuracy falls to 0.98 at L32, 0.21 at L320 and 0.00 at L3200. This is the free-rider problem of public goods (Samuelson 1954).
- **lindahl** (markets always open, each trader charged in proportion to its Aumann-Shapley credit of stream 3 in its own earnings; Foley 1970): traders pay 2.23 per episode, end with wealth 930 to 1,510, and reach 1.00 at L3200 from 1600 steps, as good as the house-funded economy.

### 16. The risks: runaway confidence observed; stall only without stream 3; no divergence

- **Bubbles.** Wealth shows none (largest share at most 0.32 in any cov run). Prices show two: auction bids are 23 times too high (section 10), and belief logits run away past the training length (mean log loss 501 nats per bit at L3200 and 3200 steps, 8,429 at L32000, while in distribution it is 0.0000). The cause is unbounded additive logit updates past the trained regime, not wealth (corrected after the audit; this line blamed rich traders): the wealth cap never binds, nocap runs away too (1,277 nats per bit), bptt_cov's logits run away the same way (801 at L3200), and bptt_cov_dscov's more (5,900 to 11,000). Section 17 tests regularizers, including liquidity that grows with depth.
- **Fixed-point stall.** The b-only economy (econ_none) stalls: training loss stays at 0.25 to 0.31 nats per bit through the last 2000 iterations with zero accuracy. The slots economy stalls near chance. The cov economy does not stall: accuracy keeps rising with depth to 3200 steps.
- **TD divergence.** None: VOC and covariance markets settle on realized outcomes, not on later prices, so there is no bootstrapped target. After iteration 1000 no house- or Lindahl-funded cov run (19 runs) logged a training loss above 0.1 nats per bit, against 4 and 7 such spikes in the two bptt_cov_prog seeds and 3 and 4 in the voluntary-bounty runs.

### 17. Phase 3 (after the audit): what carries the 100x result, the strongest backprop baseline, and regularizers for the runaway

**Status: Informative.** The 100x result is detached per-step training in random order plus per-trader optimizers; no economic ingredient adds accuracy. The strongest fair backprop baseline, Bansal et al. 2022 DT-recall with progressive loss, beats the economy at 100x (589/600 [0.97, 0.99] against 557/600 [0.90, 0.95] on 200 fresh strings per seed). A logit cap of +-10 keeps the 100x accuracy and makes log loss sane (1.00, 0.000 nats per bit), but the same cap without a market does as well (0.99). Kelvin's liquidity that grows with depth bounds the logits as designed but costs accuracy at long lengths within 10x-100x steps. The runaway itself costs accuracy: clamping the logits at evaluation only lifts the economy from 0.93 to 0.98 on the fresh strings (17.5).

**Metric.** Exact-string accuracy: a string counts only if every output bit's sign is right (L3200 = 3,200 bits). One task, prefix-sum parity. Two test sets:
- the phase-2 sets (`eval_depths` in `econ2.py`, seed 555 + L): 1000, 200 and 40 strings per seed at L32, L320 and L3200; +- is a 95% t-interval over 3 seeds;
- a fresh re-evaluation set (`reeval.py`, seed 777 + L, disjoint): 200 strings per seed at L3200, depths 320, 1600 and 3200, round-robin order; [a, b] is a 95% Wilson interval over the strings pooled over seeds (k of n exactly right).
Log loss is the mean over bits of the model's own forecast, nats per bit (for the floor arm, the floored forecast; `nll_raw` ignores the floor).
Every arm: 3 seeds, 3000 iterations, batch 64, D = 24, lengths 8 to 32, as in phase 2 (`econ2.py`, `dtrecall.py`; job lists `mkjobs3.sh`, `mkregjobs3.sh`, `mkjobs4.sh`; runner `run_p3.sh`).

**17.1 The payoff is a loss (gradient identity).** For each trader, the gradient of its market pay equals the gradient of BPTT truncated to one step with a loss at every step, covariance targets on c, VOC targets on v and the bid head, in random trader order: cosine 1.0000000 for all four traders, max abs difference at most 3.2e-6, with VOC markets on and off (`eqv4.py`, `results/eqv4.txt`). The only economic term left is the wealth scale, which never binds. So `bptt_cov_dscov_tb1_voc_rep_rand` (that loss, plus one Adam optimizer and gradient clip per trader, as the economy has) is the economy's exact twin without the ledger.

**17.2 Ablation: why the economy works at 100x.**

| arm | what it is | 10x: L320 | 100x: L3200 | fresh L3200: k/n exactly right [Wilson 95%] | log loss 100x (fresh), nats per bit |
|---|---|---|---|---|---|
| `econ_cov_voc` | economy, random order (phase 2) | 1.00 | 0.93 +- 0.14 | 557/600 [0.90, 0.95] | 341 |
| `econ_cov` | economy without VOC markets | 0.98 +- 0.05 | 0.87 +- 0.19 | 506/600 [0.81, 0.87] | 1,040 |
| `econ_cov_voc_rr` | economy, round-robin order in training | 0.96 +- 0.12 | 0.40 +- 0.27 | 235/600 [0.35, 0.43] | 7,178 |
| `bptt_cov_dscov` | BPTT full graph, round-robin (phase 2) | 0.98 +- 0.05 | 0.12 +- 0.35 | 56/600 [0.07, 0.12] | 7,439 |
| `bptt_cov_dscov_rand` | BPTT full graph, random order | 1.00 | 0.79 +- 0.47 | 449/600 [0.71, 0.78] | 1,790 |
| `bptt_cov_dscov_tb1` | BPTT truncated to 1 step, round-robin | 0.52 +- 1.20 | 0.07 +- 0.32 | 71/600 [0.09, 0.15] | 5,409 |
| `bptt_cov_dscov_tb1_rand` | BPTT truncated to 1 step, random order | 1.00 | 0.80 +- 0.11 | 482/600 [0.77, 0.83] | 1,330 |
| `bptt_cov_dscov_voc_rand` | BPTT full graph, random order, v on the VOC targets | 0.67 +- 1.43 | 0.42 +- 0.96 | 269/600 [0.41, 0.49] | 4.5 |
| `bptt_cov_dscov_tb1_voc_rand` | BPTT truncated to 1 step, random order, v on the VOC targets | 1.00 | 0.79 +- 0.53 | 497/600 [0.80, 0.86] | 636 |
| `bptt_cov_dscov_tb1_voc_rep_rand` | the economy's gradient as BPTT(1): + per-trader Adam/clip, bid head | 1.00 | 0.89 +- 0.04 | 529/600 [0.85, 0.91] | 416 |
| `econ_cov_online` | economy, online update, no VOC (the 4.2 GB configuration) | 1.00 | 0.88 +- 0.20 | 502/600 [0.80, 0.86] | 1,347 |
| `bptt_cov_progcov_a0` | BPTT loss at depth 24 only + covariance targets (alpha 0) | 0.84 +- 0.52 | 0.14 +- 0.32 | 104/600 [0.15, 0.21] | 2,180 |
| `bptt_cov_progcov_a05` | BPTT progressive loss alpha 0.5 + covariance targets | 0.98 +- 0.09 | 0.80 +- 0.65 | 487/600 [0.78, 0.84] | 115 |
| `bptt_cov_progcov` | BPTT progressive loss alpha 1 + covariance targets | 1.00 +- 0.01 | 0.80 +- 0.11 | 513/600 [0.82, 0.88] | 525 |
| `dtr_eqflop` | DT-recall width 96, equal FLOPs (3090 it) | 1.00 +- 0.01 | 1.00 | 600/600 [0.99, 1.00] | 0.000 |
| `dtr_eqtime` | DT-recall width 400, equal time (700 s) | 0.99 +- 0.04 | 0.98 +- 0.04 | 574/600 [0.94, 0.97] | undefined (overflow) |
| `dtr_full` | DT-recall width 400, published schedule (12000 it) | 0.99 +- 0.01 | 0.98 +- 0.04 | 589/600 [0.97, 0.99] | undefined (overflow) |

- **Training order is the main ingredient.** The same economy trained in round-robin order scores 0.40 +- 0.27 at 100x (235/600 [0.35, 0.43] fresh) against 0.93 in random order; full BPTT trained in random order scores 0.79 +- 0.47 against 0.12 in round-robin order.
- **Detachment alone fails.** BPTT truncated to one step in round-robin order scores 0.07 +- 0.32 at 100x (and 0.52 at 10x); in random order 0.80 +- 0.11 (482/600 [0.77, 0.83] fresh).
- **VOC targets add little.** The economy without VOC markets scores 0.87 +- 0.19 (506/600 [0.81, 0.87] fresh, per seed 0.74, 0.83, 0.96, against the economy's 557/600 [0.90, 0.95]); truncated BPTT with v trained on the VOC targets 0.79 +- 0.53, the same as without (0.80). Full BPTT with VOC targets in random order is worse (0.42 +- 0.96).
- **Per-trader optimizers close most of the rest.** The twin with per-trader Adam and clipping scores 0.89 +- 0.04 (529/600 [0.85, 0.91] fresh), overlapping the economy's 557/600 [0.90, 0.95].
- **Audit decision rule.** Truncated BPTT in random order reaches 482/600 [0.77, 0.83] on the fresh strings against the economy's 557/600 [0.90, 0.95]; it falls short on the pooled strings, and so does the variant with v trained on the VOC targets (497/600 [0.80, 0.86]). Per the rule, the remaining difference was named and tested: the per-trader Adam optimizers and gradient clipping. With them (the exact twin of 17.1) truncated BPTT reaches 529/600 [0.85, 0.91], overlapping the economy's interval; per seed 0.87, 0.87, 0.91 against 0.99, 0.92, 0.88. Because the twin's gradients equal the economy's, what is left is run-to-run variation, not a mechanism. The economics attribution is retired. The minimal claim of the TL;DR stands, restated: detached per-step training in random order extrapolates; per-trader optimizers add a little; the market adds nothing measurable.
- **The online economy works.** The online update (voc = 0, every trade backpropagated at once, the configuration behind section 9's 4.2 GB) was trained for the first time: 0.88 +- 0.20 at 100x (502/600 [0.80, 0.86] fresh). Its memory at fixed length is in 17.6.

**17.3 The Bansal baselines.** Two versions:
- **Progressive loss on our own block** (`--prog 1`, with covariance targets; n steps without gradient, then k steps with gradient from the detached state; alpha mixes (1 - alpha) x the loss at depth 24 with alpha x the progressive loss): alpha 0 scores 0.14 +- 0.32 at 100x, alpha 0.5 0.80 +- 0.65, alpha 1 0.80 +- 0.11 (104/600 [0.15, 0.21], 487/600 [0.78, 0.84] and 513/600 [0.82, 0.88] fresh). Progressive loss reaches the level of truncated BPTT in random order, as expected (it too trains from varied, detached starting states). Phase 1's "Bansal bar 0.64" was this kind of arm, on phase 1's block.
- **DT-recall, the published architecture** (`dtrecall.py`: Bansal et al. 2022's recurrent block with input recall, width-3 convolutions, progressive loss alpha 1, max 30 iterations in training), in three budgets: the published width 400 and schedule (12,000 iterations; `dtr_full`), width 400 for the economy's training time (700 s; `dtr_eqtime`), and width 96 for the economy's training FLOPs (3,090 iterations; `dtr_eqflop`). At 100x it scores 0.98 +- 0.04, 0.98 +- 0.04 and 1.00, and on the fresh strings 589/600 [0.97, 0.99], 574/600 [0.94, 0.97] and 600/600 [0.99, 1.00], above the economy (557/600 [0.90, 0.95]). Its log loss is small at 100x (0.000 to 0.061 nats per bit on the phase-2 sets) but not immune to blow-ups: dtr_full's mean at 10x is 2.3 nats per bit, and on the fresh strings seed 1's state overflowed on 3 of its 200 strings (about 1,600 wrong bits each; dtr_eqtime seed 1 likewise), so their mean log loss there is undefined.
- **The difference is the architecture's reach, not the training.** DT-recall's block reaches about 5 positions per iteration, so it needs about L / 5 steps: at L3200 it scores 0.00 at 320 steps and 0.98 at 1600. The traders read offsets +-2^j (a hand-built dyadic layout), so the economy gets 0.78 at 320 steps (its random-order backprop twin 0.73). At matched small depth the dyadic layout wins; at the depth each needs, DT-recall wins.
- **The 0.93 against 0.12 gap does not survive the strongest fair baseline** (the coordinator's condition for calling it a Gain): the fair comparison is DT-recall at 0.98 to 1.00, not full BPTT in round-robin order at 0.12.

**17.4 Regularizers for the belief-logit runaway (Kelvin: "We probably need some stronger regularization").**

| arm | what it is | 10x: L320 | 100x: L3200 | log loss 10x | log loss 100x | fresh L3200: k/n [Wilson 95%] | fresh log loss | L320 at 32,000 steps |
|---|---|---|---|---|---|---|---|---|
| `econ_cov_voc` | unregularized (phase 2) | 1.00 | 0.93 +- 0.14 | 0.000 | 501 | 557/600 [0.90, 0.95] | 341 | 1.00 |
| `econ_cov_voc_floor` | eps floor 0.01 | 1.00 +- 0.01 | 0.64 +- 0.29 | 0.007 | 0.707 | 347/600 [0.54, 0.62] | 0.849 | 1.00 |
| `econ_cov_voc_l2b` | L2 fee on belief logits 1e-3 | 1.00 +- 0.01 | 0.56 +- 1.15 | 0.019 | 0.104 | 328/600 [0.51, 0.59] | 0.100 | 1.00 |
| `econ_cov_voc_bcap` | logit cap +-10 (bounded liquidity) | 1.00 | 1.00 | 0.000 | 0.000 | 600/600 [0.99, 1.00] | 0.000 | 1.00 |
| `econ_cov_voc_dbcap` | trade-size cap 1 (tanh) | 0.97 +- 0.08 | 0.42 +- 0.90 | 0.162 | 221 | 277/600 [0.42, 0.50] | 166 | 0.67 +- 1.43 |
| `econ_cov_voc_liqt` | liquidity b_t = t | 0.89 +- 0.19 | 0.34 +- 0.26 | 0.608 | 12.6 | 214/600 [0.32, 0.40] | 12.1 | 1.00 |
| `econ_cov_voc_liqsqrt` | liquidity b_t = sqrt(t) | 1.00 +- 0.01 | 0.71 +- 0.18 | 0.001 | 91.9 | 454/600 [0.72, 0.79] | 64.2 | 1.00 |
| `bptt_cov_dscov_tb1_voc_rand` | no market: the economy's BPTT(1) twin, uncapped | 1.00 | 0.79 +- 0.53 | 0.000 | 948 | 497/600 [0.80, 0.86] | 636 | 0.99 +- 0.03 |
| `bptt_cov_dscov_tb1_voc_bcap_rand` | no market: the BPTT(1) twin with the same cap +-10 | 1.00 | 0.99 +- 0.04 | 0.000 | 0.000 | 598/600 [0.99, 1.00] | 0.000 | 1.00 |
| `bptt_cov_dscov_bcap` | no market: full BPTT, round-robin, with the cap +-10 | 0.96 +- 0.15 | 0.30 +- 1.29 | 0.000 | 0.025 | 152/600 [0.22, 0.29] | 0.036 | 0.96 +- 0.13 |
| `dtr_full` | Bansal DT-recall, published schedule | 0.99 +- 0.01 | 0.98 +- 0.04 | 2.3 | 0.000 | 589/600 [0.97, 0.99] | undefined (overflow) | 0.99 +- 0.06 |
| `dtr_eqflop` | Bansal DT-recall, equal FLOPs | 1.00 +- 0.01 | 1.00 | 0.000 | 0.000 | 600/600 [0.99, 1.00] | 0.000 | 1.00 |

- **A logit cap (bounded liquidity, +-10) is the one that works:** 1.00 at 100x with log loss 0.000 (600/600 [0.99, 1.00] fresh). **It is not economic:** the same cap on the economy's backprop twin, with no market, scores 0.99 +- 0.04 (598/600 [0.99, 1.00] fresh). On full BPTT in round-robin order the cap raises 100x accuracy from 0.12 to only 0.30 +- 1.29 (152/600 [0.22, 0.29] fresh), so it does not replace random order.
- **Epsilon floor 0.01** (fixed share, p' = 0.99 p + 0.005): the forecast's log loss is bounded (0.707 nats per bit; fresh 0.849), but the logits underneath still run away (raw log loss 4,825), and accuracy falls to 0.64 +- 0.29 (347/600 [0.54, 0.62] fresh).
- **L2 fee on belief logits (1e-3):** log loss 0.104 (fresh 0.100), accuracy 0.56 +- 1.15 (328/600 [0.51, 0.59] fresh): sane log loss, lower accuracy.
- **Trade-size cap (db -> tanh(db), a trust region per step):** fails both ways: 0.42 +- 0.90 and 221 nats per bit (277/600 [0.42, 0.50] fresh).
- **Liquidity growing with depth (Kelvin's arms).** The belief market at step t has liquidity b_t = t or sqrt(t) (b_1 = 1), positions carry over, the opening price relaxes toward the middle, and each step settles at its own liquidity. As designed, the logit becomes a running average of trades and cannot run away: log loss at 100x is 12.6 (b_t = t) and 91.9 (sqrt t) nats per bit against 501, and the wrong bits' median |logit| is 41.5 against 17,112 for the economy. But accuracy falls: b_t = t scores 0.89 +- 0.19 at 10x and 0.34 +- 0.26 at 100x (214/600 [0.32, 0.40] fresh); sqrt(t) 1.00 and 0.71 +- 0.18 (454/600 [0.72, 0.79] fresh). Neither beats the unregularized economy's accuracy or the cap's log loss.
- **Early information is diluted at long lengths, not in the training range.** At L32 the liquidity arms match the economy at every early depth (b_t = t: 0.58 at 6 steps against 0.53, 0.98 at 8 against 0.99). At L320 b_t = t trails from 16 steps on (0.55 against 0.79 at 16 steps, 0.89 against 1.00 at 320); at L3200 it scores 0.25 at 320 steps against the economy's 0.78. A possible reason, not tested: with a 1/t step size, the late trades that finish a long string move the price little, which is Kelvin's "later traders must trade more to move the price"; the traders, trained at 24 steps, may not have learned to trade that much more. More steps make up for it at L320: b_t = t scores 0.84 at 320 steps, 0.97 at 3,200 and 1.00 at 32,000 (50 strings per seed). At L3200 no run went past 3,200 steps.
- **Extreme depths.** The epsilon floor, the L2 fee, the logit cap and both liquidity arms hold 1.00 at L32 out to 24,000 steps (1,000 times the training depth) and at L320 at 32,000 steps, with log loss 0.000 to 0.021 there: none of them degrades late. The trade-size cap does: one seed of three falls from 1.00 at L32 with 24 steps to 0.59 at 240 steps and 0.00 at 2,400 and 24,000 steps, and at L320 from 0.82 at 320 steps to 0.00 at 3,200 and 32,000 (mean 0.67 at both extremes; log loss 1,509 and 1,068 nats per bit). DT-recall scores 0.99 to 1.00 at the same depths; full BPTT in round-robin order 1.00 at L32 and 0.95 at L320 (log loss 4.7 there).

**17.5 Does the runaway cost accuracy at 100x?** Yes. `reeval.py --clamp 20` reruns the fresh L3200 strings with b, c and v clamped to +-20 after every step, at evaluation only (no retraining; the traders read tanh(z / 4), so values past about 20 look the same to them). The economy rises from 557/600 [0.90, 0.95] to 586/600 [0.96, 0.99] exactly right, and its log loss falls from 341 to 0.095 nats per bit. Full BPTT in round-robin order rises from 56/600 [0.07, 0.12] to 288/600 [0.44, 0.52] (log loss 7,439 to 0.193; per seed 0.00, 0.08, 0.20 to 1.00, 0.08, 0.37). So the runaway feeds back and costs accuracy at 100x: about 5 points for the economy; for full BPTT, all of its failure in seed 0 (0.00 to 0.995), part of it in seed 2, none in seed 1. A likely mechanism, not tested: a wrong bit at |logit| in the tens of thousands (the median over wrong bits is 17,112 for the economy) cannot be flipped by later trades of normal size, while a clamped one can. Training with the cap removes the problem (17.4). The clamp does not change the ranking: with it, full BPTT in round-robin order stays far below the economy.

**17.6 Memory at fixed length (the audit's objection to section 9).** One training step at L = 32, batch 16, one L4 (`memdepth4.py`, alone on the GPU), peak GB:

| mode | D=24 | D=240 | D=2400 | D=24000 |
|---|---|---|---|---|
| bptt | 0.069 GB, 0.9 s | 0.672 GB, 1.7 s | 6.721 GB, 17.9 s | out of memory (20.7 GB) |
| bptt_ckpt | 0.006 GB, 0.3 s | 0.013 GB, 2.7 s | 0.105 GB, 26.9 s | 1.045 GB, 280.1 s |
| bptt_tb1 | 0.069 GB, 0.1 s | 0.668 GB, 1.1 s | 6.676 GB, 11.8 s | out of memory (20.5 GB) |
| tb1_stream | 0.007 GB, 0.1 s | 0.007 GB, 1.2 s | 0.007 GB, 12.0 s | 0.007 GB, 119.8 s |
| econ_batched | 0.049 GB, 0.3 s | 0.315 GB, 0.9 s | 3.045 GB, 8.3 s | out of memory (22.2 GB) |
| econ_online | 0.006 GB, 0.2 s | 0.006 GB, 1.5 s | 0.006 GB, 14.9 s | 0.006 GB, 150.5 s |

Flat memory with depth belongs to any update that backpropagates one step at a time and keeps nothing across steps: truncated BPTT with a backward at every step (0.007 GB at every depth) and the online economy (0.006 GB) are the same. Truncated BPTT with one backward at the end, and the batched economy behind the 0.93 (it settles every trade at the end of the episode), grow like full BPTT and run out of memory at 24,000 steps (22 GB). Checkpointed BPTT stays at 1.0 GB at 24,000 steps (1.5 times full BPTT's time at 2,400 steps: 26.9 s against 17.9 s). With the online economy now trained (0.88 at 100x, 17.2), the flat-memory row belongs to a model that works; the property is generic to streaming one-step updates, not to markets (E-MR-12 stays Informative).

**17.7 Halting at a single length of 64 (the audit's objection to section 10).** 500 strings per seed of length 64, 3 seeds of `econ_cov_voc`, step cap 96 (`halt64.py`); every policy read at the VOC rule's mean steps:

Every string of length 64:

| cost | steps (VOC) | VOC halting | entropy threshold | fixed depth | random (same histogram) | cap 96 |
|---|---|---|---|---|---|---|
| 0.0001 | 9.0 +- 0.6 | 0.981 +- 0.067 | 0.995 +- 0.019 | 0.856 +- 0.164 | 0.795 +- 0.194 | 1.000 |
| 0.0003 | 8.9 +- 0.6 | 0.974 +- 0.078 | 0.984 +- 0.039 | 0.823 +- 0.146 | 0.778 +- 0.164 | 1.000 |
| 0.001 | 8.7 +- 0.5 | 0.950 +- 0.108 | 0.959 +- 0.092 | 0.780 +- 0.141 | 0.738 +- 0.168 | 1.000 |
| 0.003 | 8.4 +- 0.5 | 0.871 +- 0.131 | 0.866 +- 0.168 | 0.717 +- 0.143 | 0.674 +- 0.188 | 1.000 |
| 0.01 | 7.9 +- 0.3 | 0.540 +- 0.079 | 0.539 +- 0.084 | 0.551 +- 0.078 | 0.528 +- 0.047 | 1.000 |
| 0.03 | 7.0 +- 0.0 | 0.120 +- 0.050 | 0.120 +- 0.050 | 0.171 +- 0.046 | 0.195 +- 0.035 | 1.000 |

Mixed lengths 4 to 64 (section 10's set, same script, same fixed-depth reading):

| cost | steps (VOC) | VOC halting | entropy threshold | fixed depth | random (same histogram) | cap 96 |
|---|---|---|---|---|---|---|
| 0.0001 | 20.2 +- 4.9 | 0.999 +- 0.006 | 1.000 | 1.000 | 0.960 +- 0.043 | 1.000 |
| 0.0003 | 15.0 +- 3.7 | 0.999 +- 0.006 | 1.000 | 1.000 | 0.933 +- 0.083 | 1.000 |
| 0.001 | 9.6 +- 2.2 | 0.994 +- 0.009 | 0.996 | 0.988 +- 0.027 | 0.831 +- 0.083 | 1.000 |
| 0.003 | 7.6 +- 0.6 | 0.980 +- 0.013 | 0.981 +- 0.013 | 0.854 +- 0.140 | 0.745 +- 0.061 | 1.000 |
| 0.01 | 6.6 +- 0.2 | 0.859 +- 0.042 | 0.858 +- 0.044 | 0.652 +- 0.047 | 0.625 +- 0.050 | 1.000 |
| 0.03 | 6.0 +- 0.1 | 0.531 +- 0.086 | 0.531 +- 0.086 | 0.474 +- 0.045 | 0.517 +- 0.069 | 1.000 |

VOC halting and an entropy threshold on the belief (no market) are equal within the intervals at every cost (0.981 against 0.995 at 9.0 steps; 0.871 against 0.866 at 8.4). Both beat fixed depth by 0.12 to 0.17 at costs up to 0.003 (fixed depth 0.856 at 9.0 steps). On section 10's mixed-length set the same script reproduces section 10's VOC numbers, and there too both beat fixed depth read at exactly the matched mean steps (0.859 and 0.858 against 0.652 +- 0.047 at cost 0.01). Section 10's fixed-depth column rounded the mean steps up (for example 7 steps for a mean of 6.6, `econ2.py` `eval_halting`), which gave fixed depth extra compute and made it look within noise. So confidence halting helps; the market adds nothing over the belief's own entropy (E-MR-6 stays Informative).

**What changed in the plan, and why.** The n = 200 L3200 re-evaluation takes about 43 minutes of a shared L4 per block (about 14 VM-minutes), so the clamp-20 block ran only for the economy and full BPTT (the two arms of the runaway question), and the extreme-depth blocks only for the arms whose depth behaviour is reported (the economy, full BPTT, its twin, the regularizer and liquidity arms, the cap controls and DT-recall). Spot preemptions (about 22 VM relaunches between 21:30 and 23:50 UTC) moved queued jobs between VMs; a moved job wrote a "moved to" marker, and the analysis takes each file from the highest-numbered VM. From 22:55 UTC `reeval.py` saved each finished chunk of 50 strings, so a relaunched VM resumed inside a block (tested: an interrupted and resumed block equals an uninterrupted one bit for bit).

## Two bugs in the compute-cost bookkeeping (found 2026-09-29, fixed)

1. **Cost counted twice in the halting decision (reviewer's report): real, fixed, no reported result affected.**
   `econ.py` (the first phase-2 pilot) and `econ2.py` trained each trader's bid on its net profit, `pay - cost`, and then halted when `bid <= cost`.
   That subtracts the cost twice: a step whose expected gross earnings lie between the cost and twice the cost was wrongly skipped.
   The fix trains the bid on earnings before the compute cost, `pay`, so the halting rule `bid > cost` counts the cost once.
   No reported result changed: every phase-2 run before the fix used `--cost 0` (checked in the saved `args` of every result file in `results/vm/*/p2/`), where both versions are identical.
   `econ.py` produced no reported results.
   Phase 1 (`mresnet.py`) does not have this bug: its gain head is trained on gross payments and halting fires when `gain <= halt_cost`.
2. **Train-time halting in `econ2.py` charged and settled halted steps (found while adding the cost sweep): real, fixed before any run used it.**
   With training-time halting on, a step whose example had halted was still charged the compute cost, and its bid was regressed to its realized pay, which is 0 after a halt.
   The second part is the self-fulfilling decision-market bug again: a low bid stops the example, the stop makes the pay 0, and the 0 confirms the low bid.
   The fix charges the cost only for steps that ran.
   With `--settle eps` it also settles bids only on steps that ran, weighting the epsilon-continued ones after a halt by 1/eps (Chen et al. 2014).
   `--settle real` keeps the self-fulfilling version, for comparison.
   The fix went into `econ2.py` on all VMs before any train-time-halting run started; runs without train-time halting are unchanged.

A consequence of bug 1's note: before these runs, every phase-2 run used `--cost 0`, so the priced-halting economics (halt when expected gain <= cost) had not been tested with a real cost in training.
The cost sweeps (sections 4 and 10) test it.

## Compute and cleanup

- Nine spot VMs, `trader-mresnet-1` to `-9`, each g2-standard-4 with one L4 and max-run set (12 h for -1 to -6, 3 h for -7 to -9): 13.3 VM-hours, est. $4.26 of the $20 cap at $0.32 per hour (`../GPU-LEDGER.md`, one row per VM).
- `watchdog.sh` synced `~/results` and `~/p2` from every VM every 5 minutes and summed the spend (delete-all at $18; never reached). It was stopped at 18:46 UTC after the last VM was deleted.
- Spot preemptions: -1 and -2 (12:31), -4 (16:32), -7 (16:39, 2 minutes after boot) and -8 (18:02). Each was deleted by GCP with its disk, so no sha256 check before deletion was possible; what the last sync held is kept, and what it missed was rerun: -1 and -2 on -3 and -4, the three bptt_cov_ds runs from -4 on -8, and bptt_cov_dscov seed 2 plus the L3200 and L32000 evals of seeds 0 and 1 (from their checkpoints) on -9.
- -3, -5, -6 and -9 were deleted by me after a sha256-verified sync (79, 44, 42 and 14 files, 0 mismatches) and confirmed gone with `gcloud compute instances list` and `gcloud compute disks list`.
- bptt_cov_s0's L32000 eval ran out of GPU memory while four jobs shared one GPU on -4; it was rerun alone from its checkpoint on -8 (no retraining).
- Not run: mazes (the brief allowed prefix sums and/or mazes; halting on mazes is in `../puzzles/P1`) and more than 3200 steps at L32000 or L3200. Gradient-checkpointed and truncated BPTT were run in phase 3 (section 17).
- Phase 3: sixteen spot VMs, `trader-mresnet-10` to `-25`, each g2-standard-4 with one L4 and max-run set: 8 h with stop on preemption for -10 to -15 (deliberate, to resume in place), 4 h with termination DELETE for -16 to -25 and for every VM recreated after a preemption. 38.9 VM-hours, est. $12.46 at $0.32 per hour (`../GPU-LEDGER.md`, one row per VM).
- `watchdog3.sh` synced `~/p3` from every VM every 5 minutes and summed the spend of every trader-mresnet-* VM, phases 1 and 2 included (delete-all at $19.2, raised to $21.2 at 22:58 UTC when preemptions pushed the estimate up; never reached). `p3ctl.sh`, then `p4ctl.sh`, recreated a VM that a spot preemption had removed (22 relaunches by `p4ctl.sh`, 21:30 to 23:50 UTC, besides a few earlier manual ones noted in the ledger) from its last synced state, and deleted each finished VM only after a sha256 check of every remote file against the local copy (0 mismatches), confirmed with the gcloud instances and disks lists.
- Two supervisor bugs, both fixed: `reeval.py` restarted a preempted L3200 block from its start until 22:55 UTC, when it began saving each 50-string chunk (an interrupted and resumed block equals an uninterrupted one bit for bit); and `p4ctl.sh` read its VM list on stdin, which ssh and `launch4.sh` consumed, so each round stopped after one action (fixed 23:45 UTC; it delayed one relaunch by about 30 minutes).
- All results are in `results/vm/<vm>/` (541 MB with phase 3); when a file exists on several VMs, `analyze_mr.py` uses the later VM's copy.

## Files

- Phase 1: `mresnet.py` (model, streams, losses, data), `train.py`, `evaluate.py` (trajectories to the step cap; halting policies applied post hoc), `run_grid.sh`, `run_grid2.sh`, `run_grid2b.sh`, `run_evalrest.sh`.
- Phase 2: `econ2.py` (the trader economy and its BPTT control), `memdepth.py` (section 9), `run_p2.sh`, `run_p2b.sh`, `run_p2b_vm6.sh`, `run_p2post.sh`, `run_p2cov.sh`. `econ.py` is the first phase-2 pilot, superseded by `econ2.py`, with no reported results.
- Analysis: `analyze_mr.py` writes `results/summary.json` and the charts in `charts/` (`p1_*.json`, `p2_*.json`; each has title, axis labels and series).
- Phase 3: `eqv4.py` (the gradient identity, `results/eqv4.txt`), `econ2.py` (new flags: training order, `--tbptt`, `--vocsup`, per-trader optimizers, `--online`, `--floor`, `--l2b`, `--bcap`, `--dbcap`, `--liq`), `dtrecall.py` (Bansal et al. 2022 DT-recall), `reeval.py` (the fresh 200-string re-evaluation, the clamp, extreme depths, chunked resume), `halt64.py`, `memdepth4.py`, `flops.py`, `clampwatch.py`; job lists `mkjobs3.sh`, `mkregjobs3.sh`, `mkjobs4.sh`; runner `run_p3.sh`, `smoke_p3_vm.sh`.
- Phase-3 analysis: `analyze_p3.py` writes `results/p3_summary.json` and `charts/p3_*.json`.
- Compute: `watchdog.sh`, `watchdog3.sh`, `launch3.sh`, `launch4.sh`, `mkvm4.sh`, `p3ctl.sh`, `p4ctl.sh` (logs in `results/`), `../GPU-LEDGER.md`.

## References (each link checked 2026-09-29)

- Hanson 2003, logarithmic market scoring rule: https://doi.org/10.1023/A:1022058209073
- Chen, Kash, Ruberry & Shnayder 2014, decision markets with good incentives (epsilon-continuation, 1/eps settlement): https://doi.org/10.1145/2556271
- Fissler & Ziegel 2016, higher-order elicitability: https://doi.org/10.1214/16-AOS1439
- Garrabrant et al. 2016, logical induction: https://arxiv.org/abs/1609.03543
- Sutton 1988, temporal-difference learning: https://doi.org/10.1007/BF00115009
- Schwarzschild et al. 2021, recurrent nets that extrapolate (prefix sums task): https://arxiv.org/abs/2106.04537
- Bansal et al. 2022, end-to-end algorithm synthesis (recall + progressive loss): https://arxiv.org/abs/2202.05826
- Graves 2016, adaptive computation time: https://arxiv.org/abs/1603.08983
- Banino, Balaguer & Blundell 2021, PonderNet: https://arxiv.org/abs/2107.05407
- Lee et al. 2014, deeply-supervised nets: https://arxiv.org/abs/1409.5185
- Sundararajan, Taly & Yan 2017, integrated gradients (Aumann-Shapley): https://arxiv.org/abs/1703.01365
- Samuelson 1954, the pure theory of public expenditure (free riding on public goods): https://doi.org/10.2307/1925895
- Foley 1970, Lindahl's solution and the core of an economy with public goods: https://doi.org/10.2307/1909241
- Capen, Clapp & Campbell 1971, competitive bidding in high-risk situations (winner's curse): https://doi.org/10.2118/2993-PA
- Chen, Xu, Zhang & Guestrin 2016, training deep nets with sublinear memory cost (gradient checkpointing): https://arxiv.org/abs/1604.06174
