# PREREG: untangled (tree / hierarchical) nets against dense transformers on the PFN worlds (2026-10-29)

Written before any training run (only smoke tests of 250-300 steps have run, to check the code). Later changes go to
the Amendments section at the end and are hashed in `prereg_hashes.txt`.

Kelvin's question: is tangledness necessary, or can a "scale-free" structure (information combines simply at every
scale; dimension may grow with scale) learn as well? Source proposal: `outbox/lit-tangledness-architectures-2026-10-29.md`
(Q5). Code: `sfworld.py` (worlds, exact Bayes), `sfmodels.py` (nets), `sftrain.py` (train / fine-tune / eval),
`audit.py`, `rank_post.py` (pre-training audits), `test_sfworld.py` (tests: grid Bayes equals tiny-pfn's enumeration at
N = 3 and brute-force enumeration over (partition, variance config) for coupled worlds, to 1e-6; calibration at N = 24).

## 1. Worlds (all share the tiny-pfn prior: y_q ~ N(0,1), Q = 5 questions (4 labelled, 1 query), v_s uniform on {0.1, 0.5, 2})

| id | world | knobs | exact Bayes | what Bayes needs |
|---|---|---|---|---|
| W0 | indep | N in {3, 6, 12, 24} | product of per-source likelihoods on an adaptive y grid | per-source (o*, ss): additive over sources |
| W1 | groups (known) | groups of 4, coupling rho = 0.5, N in {4, 8, 16, 24}; group ids shown to dense nets | product of per-group likelihoods (81 variance configs each) | additive over groups, not sources |
| W2 | pairs (unknown) | singles and pairs, p(partition) prop. to 1, rho = 0.9, N in {4, 6, 8} | DP over subsets | comparisons between sources (which ones share an error) |

Pre-training audit (`audit/value_table.json`, 3000 episodes each): coupling-blind Bayes (sources treated as
independent) loses 0.34 / 0.40 / 0.43 / 0.39 nats in W1 at N = 4 / 8 / 16 / 24 (rho 0.5) and 0.03 / 0.14 / 0.47 nats
in W2 at rho 0.5 / 0.8 / 0.95 (N = 4; N = 6 similar). Reliability-blind equal weighting loses 0.36-0.51 nats in W0.

Bayes-sized additive messages (`audit/rank_post.json`): projecting each source's log-likelihood (a function of y) onto
the top r prior-weighted singular functions and summing gives mean KL to exact Bayes <= 0.01 at
r = 6-8 (N = 3), 8 (N = 6), 10-12 (N = 12), 12-16 (N = 24) (constructive, not a lower bound). The multiset of N
per-source statistics (2 numbers each) also bounds the need at about 2N for small N.

## 2. Architectures (`sfmodels.py`; mixture head K = 4 as in phase B; about 10-20k parameters each)

- **tree**: per-source MLP (9 inputs: the source's 4 labelled readings, the 4 labels, its query reading; 2 hidden
  layers of 64) -> m-dim message -> SUM over sources -> readout MLP (2 x 64). No cross-source interaction. m in
  {1, 2, 4, 8, 16, 32} (W2 also 128).
- **hier** (W1): per-source -> m1 -> sum inside the known group (inner = sum) or one attention block over the group's
  sources then sum (inner = attn) -> group MLP -> m2 -> sum over groups -> readout.
- **set**: dense transformer over one token per source (same 9 features; + group id in W1) and a query token
  (d 32, 2 layers, 2 heads, PFN masks). **set-nocross**: source tokens attend only to themselves (a tree whose
  aggregation is the query token's softmax attention instead of a sum).
- **tok**: phase-B token transformer, one token per (source, question) with random 16-d source and question ids
  (+ group id in W1) and a query token; d 32, 2 layers, 2 heads.

## 3. Training and measurement

AdamW lr 1e-3, weight decay 0.01, warmup 1000, batch 256, 40,000 steps (linear cooldown over the last 10,000);
fresh episodes every step; 3 seeds in W0 (2 in W1, W2). Val gap (4096 fixed episodes, exact Bayes) at fixed steps
250 ... 40,000; best checkpoint kept. Test: 4096 fixed episodes, KL(Bayes || net) by quadrature on the Bayes grid,
and the NLL gap. Runs on one on-demand L4 VM (packed, 8 at a time); cap $10 for this lane, logged in `GPU-LOG.md`.

Metrics: (i) test KL (seed mean, seed range); (ii) sample efficiency: first eval step at which val gap <= 0.05 and
<= 0.02 (episodes = 256 x step), and val gap at steps 4000 and 10000; (iii) stability: seed range of final KL,
failures (final KL > 2 x the arm's median), loss spikes (100-step train-loss mean rising > 0.1 nats after step 2000).

## 4. Experiments 1 and 3 (this batch, from scratch)

**E1-W0**: tree (6 m x 4 N x 3 seeds), set, set-nocross, tok (4 N x 3 seeds each).
**E1-W1**: tree m 32 (flat, group-blind), hier sum (m1 32; m2 4, 16), hier attn (m1 16; m2 4, 16), set + gid, tok + gid;
4 N x 2 seeds.
**E1-W2**: tree m 8, 32, 128; set; tok; 3 N x 2 seeds.
**E3 (routing vs content)**, tok with 3 layers, W0 at N = 3 and 12, 2 seeds:
A `dense:all x3`; B `dense:self, uniform:src, uniform:all` (true routing graph, fixed uniform combiners, a per-token
step first); C `dense:self, dense:src, dense:all` (content attention restricted to the routing graph);
D `idroute:all x3` (learned routing from ids only, values from content); B' `dense:self, uniform:src, dense:all`.
W1 at N = 8 and 16 (gid), 4 layers, 2 seeds: A4 `dense:all x4`; G4 `dense:self, uniform:src, dense:grp, uniform:all`;
G4u `dense:self, uniform:src, uniform:grp, uniform:all`.

## 5. Experiment 2 (after E1, same VM): bits across cuts in trained tok nets

Bases: E1 tok nets, W0 N = 3, 12, 24 (seeds 0, 1), W1 N = 8 (gid, seeds 0, 1). Fine-tune 10,000 steps (lr 3e-4,
cooldown over the last 3000) with a bottleneck on every layer at one cut; rank-k keys and values on cut edges
(PCA-initialised from the trained net), or the edges removed (k = 0). Control: the same fine-tune without a cut.
Cuts: `xsrc` (edges between different sources' tokens) k in {0, 1, 2, 4}; `toq` (edges into the query token)
k in {1, 2, 4, 8}; `labout` (edges out of a source's labelled-question tokens to anything else) k in {0, 1, 2, 4};
W1 also `xgrp` (edges between groups) and `xsrc`, k in {0, 2}. Read-out: test KL minus the control's, against k.
Bayes's needs: W0 `xsrc` 0; `labout` 1 number per source (its sum of squared errors); `toq` r (additive) or 2
(arbitrary aggregator) numbers per source. W1 `xgrp` 0, `xsrc` > 0.

## 6. Predictions (frozen here)

- P1 (W0, does untangled match?): the best tree with m >= 8 has test KL within 0.01 nats of the best dense net
  (min over set, tok) at N = 3 and 6, and within 0.02 at N = 12 and 24, or below it.
- P2 (W0, message size): the smallest m whose seed-mean KL is within 0.005 of the m = 32 tree grows with N and is
  <= 16 at N = 24 (sublinear in N; Bayes audit: about 6-8, 8, 10-12, 12-16).
- P3 (W0, sample efficiency): the tree (m = 16) reaches val gap <= 0.05 in fewer steps than tok at every N.
- P4 (W0, stability): tree (m = 16) seed range of final KL <= tok's at every N.
- P5 (W0, scale): tok's test KL increases from N = 3 to N = 24 by more than the tree's (m = 16) does.
- P6 (W1): flat tree KL >= 0.8 x the coupling-blind gap at every N; hier attn (m2 = 16) KL within 0.03 nats of
  the best dense net at every N, or below it.
- P7 (W2): set recovers >= 50% of the coupling value (1 - KL_net / KL_coupling-blind) at every N; tree's recovered
  fraction at m = 32 is lower at N = 8 than at N = 4, and tree at m = 8 recovers <= 20% at every N.
- P8 (E3): B (routing graph + fixed combiners) within 0.02 nats of A at N = 3 and below A at N = 12; D (id-only
  routing) within 0.02 nats of A at N = 3.
- P9 (E2, W0): `xsrc` k = 0 costs <= 0.01 nats at every N; `labout` k = 1 costs <= 0.01; `toq` k = 2 costs <= 0.02
  at N = 3 and the k needed for <= 0.01 cost is non-decreasing in N.
- P10 (E2, W1): `xgrp` k = 0 costs <= 0.02 nats; `xsrc` k = 0 costs >= 0.5 x the coupling-blind gap if the base
  net captured the coupling (its KL < 0.5 x coupling-blind gap).

## 7. What counts (Kelvin's question)

For "untangled suffices here": P1 and P6 hold (the tree / told hierarchy matches dense nets where Bayes's
structure allows it), with comparable or better sample efficiency (P3).
Against: (a) at W0 the tree needs m far above the Bayes audit (> 2 x) or never matches dense nets (learning, not
representation); (b) W2 (comparisons needed): the tree cannot recover the coupling value at any m we try while dense
nets can (tangling buys input-dependent routing); (c) E2 shows dense nets rely on cross-source communication that
Bayes does not need (their solution is tangled even where the task is not).
Every result is reported, including failed predictions. Claims wait for a GPT-6 review.

## Amendment 1 (2026-10-29 19:33Z per hash log; heading said 19:40Z; before any result): infrastructure, and E4 (World FA routing)

Infrastructure: training draws episodes on the GPU (`sfgpu.py`, tested equal in law to the numpy sampler by
`test_sfgpu.py`) and the VM runs CUDA MPS; nothing else in sections 1-6 changes. Two earlier VM attempts produced no
results (sfx-b1 stopped for speed after 9 min; sfx-b2 failed at import), total $0.28.

E4 (`e4fa.py`; `fa_core_copy.py` is an unchanged copy of work/world-fa/fa_core.py, sha256 8d9ac741...). World FA
prior f with one anti-correlated pair (rho 0.8, one negative loading), N = 4, T = 32, report tokens (133 per
episode); history value (history-blind Bayes NLL minus Bayes NLL) 0.25 nats (val, 2048) / 0.22 (test, 4096).
Nets: sfmodels.TokTF, d 64, 4 layers, 4 heads, 20,000 steps, batch 256, lr 5e-4 (warmup 1000, cooldown 5000),
2 seeds. Arms (per-layer attention):
  flat `dense:all x4`; flat1 `dense:self, dense:all x3`;
  told `dense:self, dense:src, dense:all, dense:all` (layer 2 attends only within the same trader's tokens);
  toldu `dense:self, uniform:src, dense:all, dense:all` (fixed uniform pooling within the trader);
  learned `dense:self, idroute:all, dense:all, dense:all` (layer-2 routing from the id features only).
Read-out: fraction of the history value recovered on the test set, best-val checkpoint.
Predictions: P11 flat and flat1 recover <= 10% (both seeds). P12 told recovers >= 30% in both seeds.
P13 learned recovers >= 30% in at least one seed. toldu: no prediction (the pooled column loses round alignment
unless the per-token step encodes it).

## Amendment 2 (2026-10-29 19:43Z per hash log; heading said 19:55Z; before the E1 nets exist): E5, deduction vs sampling for tree nets

Code `est5.py` (developed on a 4000-step local tree, `/tmp/sfx_dev_tree_m2`; dev numbers in `e5dev/dev.jsonl` are
not results). Targets: the E1 trees on W0 at N = 12, seed 0, m = 2 and m = 16; the E1 tok net (N = 12, s0) as the
dense contrast (sampling only: it has no per-source factorisation to exploit).
Method. Given the shared variables y (4 labels, y*), sources are iid, so the tree's root law is an N-fold convolution
of one source's message law. One source: 3 variance levels x 2^13 scrambled-Sobol points per level (tail-covering,
importance-weighted); shared variables: 2^10 such points; m = 2: exact FFT convolution on a per-y grid (G = 1200),
readout on every cell with mass; any m: Gaussian root (CLT) with the exact per-source moments.
Estimands: E[NLL_net], E[(mean_net - mean_Bayes)^2] (via E[mean^2] - 2 E[mean y*] + E[mean_Bayes^2], the last a
world constant by Monte Carlo with exact Bayes), P(NLL_net > L), L in {4, 6, 8, 10, 12, 14, 16, 20}.
Reference: plain Monte Carlo with 8M episodes. Equal-time Monte Carlo: M matched to deduction's wall clock.
Predictions: P14 m = 2 exact deduction within 0.003 nats of the reference on E[NLL] and within 25% on P(NLL > L)
wherever the reference has >= 20 events. P15 the Gaussian root underestimates P(NLL > 8) by >= 2x (m = 2 and m = 16).
P16 at equal wall clock, Monte Carlo is more accurate than deduction for E[NLL]; for L = 16 and 20 deduction gives
estimates stable within 30% under refinement (G 1600, 2^14 points per level) where equal-time Monte Carlo sees no
event.

## Amendment 3 (2026-10-29 19:45Z per hash log; heading said 20:15Z; before any result): rounds and larger N (from the theory note)

The theory note (`outbox/scale-free-theory-2026-10-29.md`, section 5.3, 6.4 P1) predicts that rounds matter more
than hierarchy shape: a two-pass tree (pass 1 sends a few numbers per source, the root broadcasts a pilot estimate,
pass 2 sends a few numbers relative to it) matches a one-pass tree that needs many more numbers, once N >= 64.
Adopted measurements: this rounds arm, M1 (interaction profile across source-aligned vs source-splitting cuts) on
the E1 dense nets, and P5 (= E5).
New arch `tree2` (sfmodels.TwoPassTree): pass 1 m1 = 3 numbers per source, summed -> pilot (mean, log-sd; auxiliary
NLL weight 0.3); pass 2 per-source encoder sees its 9 features, the pilot and its query reading standardised by the
pilot -> m2 numbers, summed -> readout (+ pilot). Same training recipe as E1; 2 seeds.
Arms (W0): tree m in {4, 8, 16, 32} at N in {64, 256}; tree2 m2 in {2, 4, 8} at N in {3, 24, 64, 256};
set (d 32, 2 layers) at N in {64, 256}. Val/test sets 4096 episodes each at N = 64, 256 (exact grid Bayes).
Predictions: P17 at N in {64, 256} the best tree2 with m2 <= 8 has test KL <= the one-pass tree's at m = 32
(seed means); at N = 3 tree2 (m2 = 8) is within 0.01 nats of the one-pass m = 8 tree.
P18 (scale) the one-pass tree's smallest m within 0.005 of its m = 32 KL does not decrease from N = 24 to N = 256.
P19 the set net's KL at N = 256 exceeds the best tree's KL at N = 256.
M1 (interaction profile; after E1): for each E1 dense net (set, tok at N = 3, 12) and tree, Delta = L(a,b) - L(a',b)
- L(a,b') + L(a',b'), L = log p(y1 | x) - log p(y2 | x) at y1, y2 = Bayes mean -/+ one Bayes sd; a, a' (b, b') two
draws of the cut's (the rest's) errors (fresh source: new variance level and errors; labels, ids and y* kept).
Cuts: one source vs the rest; two of one source's labelled questions vs the rest. Share = Var(Delta) /
(Var(main effect of a) + Var(main effect of b)). Bayes: 0 for the source cut (exactly), > 0 for the splitting cut.
Prediction P20: for the dense nets the source-cut share is below the splitting-cut share and below 0.1.

## Amendment 4 (2026-10-29 20:01Z per hash log; heading said 20:05Z; before any result): compute changes

On VM sfx-b3 (asia-southeast1-a) the ten E4 runs saturated the GPU (FP32, about 340 ms per step for every job).
At 20:01Z: the E4 runs there were stopped (no result kept) and moved to a second VM (sfx-b4, with the Amendment 3
arms); the trainer on sfx-b3 was patched so that jobs started after 20:01Z train with TF32 matmuls (evaluation,
validation and test, stays in full FP32). The two jobs running at the time (e1w0_tok_N24_s0, s1) train in FP32.
The working copies of sftrain.py and e4fa.py carry the same TF32-for-training change.

## Amendment 5 (2026-10-29 20:04Z per hash log; heading said 20:10Z; before any result): batch 1024, 10,000 steps; work split over two VMs

Per-step overhead (12 processes on 8 vCPUs) caps every job at about 45-50 ms per step, so 40,000-step runs would take
about 10 h. New recipe for every E1 / E3 / Amendment-3 run: the same 10.24M training episodes as batches of 1024
for 10,000 steps (warmup 250, linear cooldown over the last 2,500), lr 1e-3; E2 fine-tunes: 2,500 steps at batch
1024 (lr 3e-4, cooldown 750) = the same 2.56M episodes as planned. Sample efficiency is reported in episodes. E4 keeps
its recipe (batch 256, 20,000 steps). The 12 E1 token-net runs (W0) that had started at batch 256 on sfx-b3 are
stopped (no result kept) and rerun on sfx-b4 with the new recipe, together with E2 (which fine-tunes them), E4 and
the Amendment-3 arms; sfx-b3 runs the rest of E1 and E3. Thresholds and predictions are unchanged.

## Amendment 6 (2026-10-29 20:25Z per hash log; heading said 20:30Z; before any E1-E4 result): adopt theory note sections 6.3 and 6.5

Coordinator's request (20:20Z): adopt the final theory note's (`outbox/scale-free-theory-2026-10-29.md`) section 6.3
(M1-M7) and 6.5 (hierarchy vs dense; routing given vs learned; one vs two rounds), with their predictions and
refutation criteria, and make sure learned routing is exercised. What we run, mapped to their items:

- Token order. Our token nets have no positional encoding, so they are equivariant to the order of context tokens and
  must route by the random per-episode ids; shuffling the order per episode cannot change their outputs. We verify
  this numerically on trained nets (outputs before vs after a per-episode permutation, max abs difference) and use
  per-episode permutation wherever an analysis compares patterns across episodes (M3). Learned routing is exercised
  in tok (dense, learned), E3 idroute (learned, id-only), and oracle routing in tree / set / E3 B, C (routing graph).
- M1 (`m1prof.py`): also the deviation share (Delta of the net's log-odds minus Bayes's). Theory prediction: nets
  without a per-source nonlinearity before mixing show less within-source interaction than Bayes (0.039); refuted if
  within-source shares of such nets are at or above Bayes's. Our nets: tok (2 layers), set, setnc, tree; N = 3, 12.
- M2 (routing classes, label fit) on tok nets N = 3, 12 (seeds 0, 1) and the E3 A nets: clusters of attention rows
  (TV <= 0.05, 0.1, 0.2), per-key column share, label R^2 of the pair-specific part, and in the last layer the
  correlation of the query's log mass on each source's query reading with log E[1/v | residuals]. Theory prediction:
  few classes in layer 1, labels explain most of it; query mass tracks log precision. Refuted if classes are near the
  number of rows with labels explaining little.
- M3 (routing substitutions) on tok N = 3 (seeds 0, 1), after a per-episode permutation: replace layer-l logits by
  (ii) the pair-class mean of the logits, (iii) another episode's logits, (iv) the episode-averaged pattern; KL to the
  net and to Bayes. Theory prediction: (iii), (iv) worse than (ii)... refuted if (iv) is as good as the net.
- M4 = new GPU jobs on sfx-b4 phase 2: rank-k keys and values on every edge out of a source's tokens (cut `both`), in
  both layers, base frozen, only the maps trained (2,500 steps, lr 3e-3), k in {1, 2, 4, 8}, tok N = 3, 12, 24, seeds 0, 1.
  Read-out: smallest k with KL(net || bottlenecked net) <= 0.01. Theory prediction: 2-4 for nets near Bayes; refuted
  if k >> 4. (Our E2 cuts fine-tune the whole net; M4 freezes it.)
- M5 (sink and posterior variance) on tok N = 3, 12: regress the query token's attention to itself in the last layer
  on the net's and Bayes's predictive variance. Prediction R^2 >= 0.5 in some nets.
- M6 (effective rounds) on tok N = 3, 12: sensitivity of source s's query-question token after layer 1 to other
  sources' errors relative to its own (finite differences over error redraws). Descriptive.
- M7 on E4 nets: own/other attention mass per layer (flat vs told vs learned). Prediction (theirs): flat arms show one
  routing class (own/other about 1).
- Section 6.5 rounds: capacity-matched one-pass trees (h 96, m = 7 and 11, matching tree2's 3 + 4 and 3 + 8 numbers
  and about 20k parameters) at N = 3, 24, 64, 256 (sfx-b4 phase 2). Their P1: the two-pass tree reaches at small k
  the KL a one-pass tree reaches only with larger k once N >= 64; similar at N = 3; refuted if no better at matched
  size for N >= 256. Read-outs add the 95th percentile of per-episode test KL and the smallest k reaching KL <= 0.01
  and 0.001 (95th percentile).
- Their P4 (routing) is read from E3 (A dense learned routing vs C dense oracle routing graph) and E4; their P5 is E5.
Not run: their W2 with G groups of hidden membership beyond our unknown-pairs world; a tree with learned token-to-source
clustering (with exact shared ids it equals oracle routing, as the note says).

## Amendment 7 (2026-10-29 20:55Z, before any E4 or set-N256 result): compute cuts on sfx-b4

The set net at N = 256 (257 tokens, batch 1024) needs several GB of GPU memory per process; sharing the GPU with E4
it caused a CUDA out-of-memory failure (e4_flat1_s1) and slowed every E4 run to about 1 s per step (20,000 steps would
take about 5.5 h). Changes: the set arm at N = 256 is dropped (P19 is not tested); the running E4 runs are stopped
(none had passed step 1000; nothing kept) and rerun in sfx-b4's phase 2 as `e4r_*` with 8,000 steps (cooldown 2,000;
batch 256, lr 5e-4, same nets and arms, 2 seeds). E4's predictions P11-P13 are read on these runs.

## Amendment 8 (2026-10-29 20:57Z per hash log; heading said 21:00Z; before any set / tree / hier result): third VM

sfx-b3 had finished about 30 of its 226 jobs after 55 min. Its 168 light jobs (every set, setnc, setg, tree and hier
run of E1, none started) are moved to a third VM, sfx-b5, with the same code and recipe (the working-copy trainer:
TF32 training, batch 1024, 10,000 steps); sfx-b3 skips them through marker files (`eval_test_model.json` containing
`{"skipped": ...}`, no meta.json, so no analysis reads them). Nothing else changes.

## Amendment 9 (2026-10-29 21:04Z per hash log; heading said 21:06Z; before any E2 / M4 / matched-tree result): phase-2 split

To finish sooner, 92 of sfx-b4's phase-2 jobs (M4: 24; capacity-matched trees: 16; E2 fine-tunes of the N = 3 and
N = 12 base nets: 52) run as sfx-b5's phase 2. Their base nets (e1w0_tok_N{3,12,24}_s{0,1}, as trained on sfx-b4)
were copied to sfx-b5; sfx-b4 skips these jobs through marker files. sfx-b4 keeps E4r and the N = 24 E2 fine-tunes.

## Amendment 10 (written 22:08Z, before any capacity-matched or ablation result): two-pass ablation

The GPT-6 review (results/review1.md) notes that the two-pass tree differs from the one-pass tree in three ways
besides the second round: about twice the parameters, an auxiliary NLL on the pilot, and a standardised query
feature. The capacity-matched one-pass trees (Amendment 6) address the first. Added: two-pass trees with the
auxiliary loss switched off (`--aux 0`), m2 in {4, 8}, N in {24, 64}, seeds 0, 1 (8 jobs on sfx-b5's phase 2).
Prediction P21: without the auxiliary loss the two-pass tree's KL stays within 25% of the two-pass tree with it,
and below the capacity-matched one-pass tree, at N = 24 and 64. If the capacity-matched one-pass tree reaches the
two-pass tree's KL, "rounds are the lever" is withdrawn.

## Amendment 11 (written 22:47Z, after seeing E1 W0; labelled exploratory): budget check at N = 12

Question: is the one-pass tree's 1.5-2x gap to the dense set net (W0, N = 12: 0.014 vs 0.0076 at 10.24M episodes) a
learning-speed effect that closes with more data, or a limit of the one-pass form? Runs (CPU, seed 0, same recipe but
40,000 steps at batch 1024 = 40.96M episodes, cooldown over the last 10,000): tree m = 32, two-pass tree 3 + 8, set.
Prediction (made after seeing the 10.24M results, so exploratory): the one-pass tree stays above 1.3x the set net's KL
at 4x the budget.
Addendum (23:46Z, before their results): seed 1 of the 4x tree m 32 and two-pass 3 + 8 runs, to check the one-seed result (the set net at 4x stays one seed).
