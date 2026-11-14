## W3 discrete chain world — C2 pilot (written 2026-10-28 ~01:55Z, before any W3 training)

(Same text as the W3 section of PREREG.md; kept as a separate file because PREREG.md is shared and was rewritten
by another lane at 01:59Z.)

Code: world3_chains.py (world, exact Bayes, references; tests test_world3_chains.py), looped.py (weight-tied
LoopedSetPFN on harness Blocks; test_looped.py), harness.py (snapshot shipped with the run; sha256 in the run dir).

World (defaults in world3_chains.DEFAULT): M = 8 disjoint chains x n = 12 nodes per episode, node ids drawn without
replacement from V = 128; latent bits iid Bern(1/2). Messages: UNARY reports (flip rate p, reliability c = 1-2p
shown on the token, c in {0.6, 0.8, 0.9}), XOR between adjacent nodes (token order random). Every chain is drawn
by the same procedure around a virtual query at distance l' from its nearest unary report; further reports with
prob 0.25 per node beyond l'. Information matching: a chain is exact (XOR c = 1) w.p. 1/3, else XOR
c_e = c_path^(1/max(l',1)), c_path in {0.7, 0.9}, so the nearest-report path has the same reliability at every l.
Majority messages: implemented (exact), OFF in this pilot. Policy 'std': distractor l' ~ U{0..4}; 'far' (held out):
distractor l' ~ U{0..8}. Sequence length ~125-130 tokens at every l (only l changes between train and test).
Exact Bayes: pair-state forward-backward on the query chain (tested against enumeration and against a
tokens-only enumeration oracle, i.e. it is a function of the visible messages; policy ignorable).
References: prior, unary-only (reports on q itself), nearest-report-only, Bayes.
Information available, KL(Bayes||prior) (20k eps, std): l=0 0.49, 1 0.39, 2 0.39, 3 0.38, 4 0.36, 5 0.33, 6 0.31,
7 0.30, 8 0.29 nats. Gap nearest-only -> Bayes: 0.14, 0.10, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04, 0.03.

Arms (harness SetPFN without sink: pre-norm, no positions, context tokens attend to context only, query reads;
width 128, 4 heads, MLP 4x; target 'cat'; lr 5e-4, warmup 1000, cosine, AdamW wd 0.01, bs 256, 50,000 steps,
sampled-bit NLL, best.pt by val NLL on the training distribution):
  U2, U4, U6   untied depth 2, 4, 6
  LP2x2, LP2x3 weight-tied 2-layer block x 2, x 3 iterations, input injection (rounds 4, 6; LP2x1 == U2)
Train l ~ U{0..4}; seeds 0, 1, 2 (pilot = screen; crux claims later need >= 10 seeds).
Eval: harness evalset, l in {0..8} (harness N = l), policies std and far, 2000 episodes per cell, seed 7.
Looped arms also at test-time iterations 1..6 (rounds 2..12).

Metrics: KL(Bayes||model) per l (seed mean, se over seeds); fraction of available information captured
1 - KL/KL(Bayes||prior); reach = largest l with seed-mean KL < 0.05 at every l' <= l (std policy).

Pre-registered readings:
- R1 (C2 refutation for W3): U2 seed-mean KL < 0.05 at l = 6 (std, distractors present) => C2 refuted for this
  world. Otherwise not refuted (the pilot cannot confirm C2; >= 10 seeds needed).
- R2 (round-limited signature): reach increases with rounds. "One rung per round": reach(U4)-reach(U2) and
  reach(U6)-reach(U4) each in [1, 3] (about one rung per added layer). Pointer-doubling alternative: a depth-L
  net reaches l >= 2^(L-1) (e.g. U4 reaches l = 8). Report which, or neither.
- R3 (looped ~ untied at equal rounds): on l <= 4, |KL(LP2xT) - KL(U(2T))| <= 0.02 nats or within 2 se, T = 2, 3.
- R4 (test-time iterations): LP2x3 run at 4..6 iterations lowers KL at l in {5..8} vs 3 iterations (seed mean,
  every seed) => useful test-time computation. This does not establish an expressivity bound (Codex r3).
- Training validity: if U6 does not reach KL < 0.05 on l <= 2 (in distribution), the pilot is a training failure
  (C5-type) and is reported as such, not read for R1-R4.

### W3 amendment A1 (2026-10-28 ~02:40Z; before any W3 result beyond step-2000 validation NLL)
A first launch (2 VMs, 9 and 6 concurrent jobs) was stopped after ~25 min: one job hit CUDA OOM and steps took
~0.2 s, so 50k steps would not finish inside the VM timeout. Seen before stopping: validation NLL at step 2000 only
(all arms 0.692-0.700, near the prior level); no evaluation results. Changes, applied to every arm alike:
bf16 autocast (looped.py AmpWrap; logits cast to fp32) and TF32 matmuls; 40,000 steps instead of 50,000; one VM
per seed with the 5 arms of that seed in parallel. Everything else, including R1-R4, is unchanged.

### W3 amendment A2 (2026-10-28 ~03:25Z): plateau, learnability screen
Observed (validation NLL only, no evaluation): at 10k-18k steps all 15 runs (5 arms x 3 seeds) sit at val gap
~0.30 nats to Bayes, i.e. the l = 0 share solved and nothing at l >= 1 (all-arm plateau; identical across depth).
Seeds 1 and 2 were stopped; seed 0 (all 5 arms) continues to 40k as the no-curriculum reference.
Screen (one VM, seed 0, 20k steps, train only, validation curves; disclosed as a screen, not a result):
U2, U4, U6, LP2x3 with an l-curriculum (world cfg curriculum = 3000: training l ~ U{0..k} after 3000k steps,
full range U{0..4} from step 12k; the validation set keeps the full range), plus U4 with curriculum and exact
chains only (p_exact = 1). If the curriculum arms leave the plateau, the pilot (5 arms x 3 seeds, A1 settings)
is rerun with the curriculum and the same R1-R4; the plain-training result is reported alongside as a C5-type
finding.

### W3 amendment A3 (2026-10-28 ~03:45Z): l-curriculum did not help; correlated-latent anneal screen
Screen A2 (validation only) stopped at 5k-11k steps: every arm still at the l = 0 plateau, including U2 at
stage l <= 3 and U4 with exact chains only (so XOR noise is not the obstacle). Reading: with iid uniform bits every
message on the path is individually independent of x_q; only their parity is informative (a sparse-parity
gradient barrier, C5-type), so a curriculum on l alone does not give a first-order signal.
Screen 2 (one VM, seed 0, 20k steps, train only): training latents are a Markov chain with
P(x_{i+1} != x_i) = flip rising linearly from 0.05 to 0.5 over the first A steps (cfg anneal = A; A = 10000, and
5000 for one arm), then iid (the target world). Validation and evaluation always use the target world, so the
oracle, references and R1-R4 are unchanged. Arms: U2, U4, U6, LP2x3 (A = 10k), U4 (A = 5k).
If the anneal arms leave the plateau, the pilot (5 arms x 3 seeds, 40k steps, A1 settings) is run with
anneal = 10000 (or 5000 if only that arm escapes) for every arm.

### W3 amendment A4 (2026-10-28 ~04:55Z): anneal failed; plain seed-0 pilot evaluated; diagnostic screen 3
Plain pilot seed 0 (A1 settings, 40k steps, all 5 arms) finished and was evaluated: every arm is at the prior at
every l >= 1 (KL equal to the available information) and at the unary-only level at l = 0 (KL 0.13-0.14), i.e.
it reads only reports on the query node. Pre-registered training-validity check fails; R1-R4 are not read.
Anneal screen (A3): all arms still on the same plateau at 15k-20k steps. Local CPU checks (width 64, bs 64,
<= 4.5k steps; not results): dense queries (w3_dense.py, K = 16 extra queries) and M = 1 or 2 chains also stayed on
the plateau.
Screen 3 (one VM, seed 0, U4, 20k steps, train only) decomposes the barrier; each arm changes one thing, and each
variant world has its own exact oracle for validation (the Markov-prior oracle is tested against enumeration):
U4eq: equality world (flip = 0, chains constant; pure multi-hop retrieval, no parity);
U4sg: extra signed-evidence feature (2r-1)c; U4lr: lr 1e-3; U4M2: M = 2 chains; U4eqM2: equality and M = 2.

### W3 amendment A5 (2026-10-28 11:25Z): diagnosis; C2 pilot moved to the equality world
Screen 3 at 20k steps (val gap to Bayes): XOR world stays on the plateau with lr 1e-3 (0.300), the signed feature
(0.299) and even M = 2 chains (0.322). The equality world (flip = 0, same messages and noise, latent constant along
a chain) is learnable: M = 2 reaches 0.057; M = 8 is still improving (0.46 at 5k, 0.35 at 10k, 0.26 at 20k).
Reading: the barrier is the parity along the path (individually uninformative XOR bits), not retrieval or
distractors. The XOR world is reported as a C5-type blocker (plain training, l-curriculum, latent anneal, lr,
signed feature, dense queries locally: all stuck).
C2 pilot now runs on the equality world (cfg flip = 0; M = 8, n = 12, everything else as in A1), 50,000 steps,
5 arms x 3 seeds (one VM per seed). R1-R4 and the training-validity check are unchanged and read on this world.
What it tests: multi-hop retrieval/connectivity over M = 8 chains with random ids (the posterior is the sum of the
log-likelihood ratios of all reports reachable from q), not discrete parity propagation.
Information per l (equality world, KL(Bayes||prior), 20k eps): 0.60, 0.59, 0.56, 0.53, 0.50, 0.46, 0.44, 0.42,
0.40 for l = 0..8; nearest-only gap 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.07, 0.05.

### W3 result note (2026-10-28 13:20Z, after the A5 pilot; not an amendment)
The A5 pilot fails the training-validity check (U6 KL 0.20-0.30 on l <= 2). The models solve noisy chains partly
and flatly in l, and exact chains hardly at all: consistent with a shortcut where the per-chain XOR reliability
c_path^(1/l') labels the chain. Any rerun must draw XOR reliabilities per edge from a common set.
Report: outbox/world3-chains-pilot-2026-10-28.md.
