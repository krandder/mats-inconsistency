# W1 junta pilot: preregistration (2026-10-28, before any training)

Code: `junta.py` (world, exact Bayes, references), `harness.py` (world-agnostic PFN harness), `diag_junta.py`
(constructed-context diagnostics), `test_junta.py`, `run_pilot.sh`. SHA-256 in `code_sha256.txt`.
Spec source: `../world-critique/DESIGN-v1.md` (W1) plus the coordinator's round-2/3 and literature amendments.

## World (final W1 spec)
- Items x in {0,1}^16. Hidden R uniform over the 120 coordinate pairs. Cell values g(c) ~ N(0,1) (4 cells).
- Item value v_x = g(x_R) + b_x, b_x ~ N(0, tau2); pilot both tau2 = 0 (plain junta) and tau2 = 0.25.
- Fresh reading y ~ N(v_x, sigma2), sigma2 uniform on {0.1, 0.5} per episode.
- Provenance: each fresh reading carries a random event-id vector in R^8. Token t is, with per-episode probability
  p_copy, an exact copy of a uniformly chosen earlier token (same item, y and id). p_copy = 0 with prob 1/3, else
  U(0, 0.6). Hidden-ancestry relays are dropped (their exact oracle needs a sum over partitions; and the
  fresh-vs-copy odds would then involve the policy's item probabilities, so Bayes would stop being policy-free).
- Policies, all ignorable (visible bits, recorded observations, independent randomness only): uniform; Zipf(a),
  a ~ U(0.8, 1.5), ranks hashed to items; pattern (prob U(0.5, 0.95) of matching a random bit pattern on 2-6 random
  coordinates chosen independently of R); repeat (prob U(0.5, 0.95) of one of K ~ U{1..16} items); UCB on running
  item means (eps = 0.2). Train on the first four; UCB is held out. Every structural count is randomised per
  episode (copy rate, Zipf exponent, pattern rate and width, repeat pool and rate), and the training N is log-uniform
  in [8, 64].
- Query: with prob 0.5 a uniformly chosen distinct context item, else a uniform item. Target v_q.
- Tokens: 16 item bits (+-1), y, event id (8), query flag. The query token is last; no positional encoding.

## Exact Bayes and references
Bayes: collapse tokens by event id; per item n_x, mean and within-item SS; for each (R, sigma2), item means are
N(g_c, tau2 + sigma2/n_x) given g_c and the within SS informs sigma2; g_c is integrated analytically; the result is a
mixture of 240 Gaussians for v_q. It is verified by tests against dense joint-Gaussian conditioning (to 1e-9),
against ABC on the generative spec, by reductions (no copies: copy-naive = Bayes; copies removed: unchanged;
one hypothesis: oracle = Bayes), by replay ignorability (same policy randomness and observed values under a
different hidden world give identical items, provenance and query), and by calibration of Bayes z-scores under
every policy.
References: `prior` N(0, 1+tau2); `r0` fixed shrinkage of the query's raw readings (copy-blind; mean n*ybar/(n+k),
var a + b/(n+k)), fitted on training episodes (prior-level); `shortcut` Gaussian regression of the target on the
query's counts only (no values), fitted on training episodes; `item` item-only Bayes (pools repeats of one item,
no sharing across items); `count` copy-naive Bayes (copies treated as fresh); `oracle` Bayes with R known;
`bayes` full.
Abilities: similarity = fraction of the item->Bayes NLL gain the model achieves; dedup = fraction of the
count->Bayes gain on episodes with copies; decomposition = (NLL_r0 - NLL_ref) / (NLL_r0 - NLL_model) per reference.

## Pilot arms and evaluation
24 arms: tau2 {0, 0.25} x depth {1, 2, 4} x attention sink {0, 1} x seed {0, 1}; width 64, 4 heads, MLP 4x,
4-component Gaussian-mixture head, AdamW lr 5e-4 (1k warmup, cosine to 10%), batch 256, 40k steps, best-val
checkpoint (val = 2048 training-distribution episodes). The sink is a learned key every token sees (Barbero et al.
2024 ablation; crossed with depth 1 vs 2+).
Eval: N in {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} x all five policies; 1000 episodes per cell up to N=512,
then 500/250/125. Metrics: NLL gap to Bayes (mean, se), KL(Bayes || model) by quadrature, gap of every reference,
bias and log variance ratio of the model predictive vs Bayes, splits (query observed / unobserved, copies / none).
Diagnostics (`diag_junta.py`, 100 constructed bases each): C1 contamination (M in {0,16,64,512,4048} readings of
one dominant item at Hamming distance 8 from the query, in another cell; fresh vs copy variants; matched-length
balanced control; Bayes and R-known-Bayes contamination simulated as anchors; query attention mass on the dominant
item vs its share); D1 R-evidence pairs (same items and value multiset, values permuted among non-query reports);
D2 k copies vs k agreeing fresh readings of the query's single reading (the arXiv 2610.16801 control; k in
{1,4,16,64}); D4 whole context x k as copies (Bayes unchanged) vs as agreeing fresh readings (Bayes sharper);
ridge probes of token states for log item count (fresh, raw) and log true-cell count.
C1 rule (DESIGN-v1): refuted if at any M where the matched-length control has KL < 0.05 the model's KL on the
fresh-dominant context exceeds the control's by > 0.10, or its KL grows > 0.03 per doubling of M. If the control
is never below 0.05 the rule is vacuous and is reported as such.

## Predictions (before training)
P1 Similarity. At N=64 in distribution, depth 4 achieves >= 50% of the item->Bayes gain and depth 2 >= 25%, for
   both tau2; depth 1 < 25% (it stays near item/prior level, as in the tiny-world runs).
P2 Dedup. At N=64 on copy episodes, depth >= 2 achieves >= 70% of the count->Bayes gain (ids make copies easy).
P3 Length. Gap to Bayes grows with N past the training range for every arm: gap(N=1024) > 2 x gap(N=64), and at
   N=4096 at least one policy has the model worse than `item`. The transversal crux is refuted for this
   architecture (gap grows).
P4 Held-out policy. At N <= 64 the UCB gap is within 0.05 nats of the mean in-distribution gap.
P5 Sink. Sink arms have smaller gap growth with N than no-sink arms of the same depth (by >= 20% at N=1024).
P6 C1. Without a sink, layer-1 query attention mass on the dominant item tracks its share M/(48+M) within 0.1;
   model contamination exceeds the Bayes anchor at M >= 512 for both fresh and copy variants; the rule is vacuous
   or refuted (control KL above 0.05 at large M).
P7 D4. Without a sink, x k copies leave the model exactly unchanged (token multiset scaled; a check, not a
   finding). For x k agreeing fresh readings the model gets < 50% of the Bayes log-variance drop at k = 8.
P8 D2. Depth >= 2: |log variance change| < 0.1 for k <= 16 copies; for agreeing fresh readings >= 50% of the
   Bayes log-variance drop at k = 4 and < 50% at k = 64.
P9 D1. Depth 4 at N=64: slope of model shift on Bayes shift >= 0.5; lower at N=256.
P10 Probes. Log item count (fresh) is linearly decodable (R^2 > 0.5) after layer 1 in depth >= 2 arms.

## Related work to cite
arXiv 2610.16801 (production PFNs over-count repeated measurements under an IID-rows prior; proposes a
copies-vs-fresh hierarchical prior, which this world implements; D2 is their missing control). Barbero et al.
2024 (softmax without a sink sees only proportions; D4 and the sink arms). Golkar et al. (fixed training counts
become biases; hence randomised counts).


# W3 discrete chain world (lane: W3 builder; full text also in PREREG-W3.md)

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

## W2 market world — pilot (written 2026-10-28 ~02:15Z, before any W2 training)

Code: world2_market.py (generator, exact Bayes, references, C4/C7 constructions; tests test_world2_market.py,
23 passing), w2_summarize.py, run_w2.sh, harness.py (snapshot shipped with the run; sha256 in the run dir).
Spec: ../world-critique/DESIGN-v1.md (W2, C4, C7) and the r3 critiques.

World. K items v ~ N(0,1); S sources, bias b_s ~ N(0, 0.5^2); known noise sd ~ logU[0.15, 1] shown on tokens,
except U ~ U{0..min(8,S)} sources whose sd is SIG_L = 0.2 or SIG_H = 1.2 (prior 1/2, token shows only "unknown").
Quote types: SPOT v_k + b + eps; DIFF v_k - v_j + eps (bias-free: an additive bias cancels in a spread); BASKET
mean of 2-4 items + b + eps. Relays: a quote re-quoting an earlier measurement (root event id and origin source on
the token) through the relayer's own bias, y = a.v + c b_relayer + eps_root + N(0, 0.1^2); the original may be
visible or hidden (measured outside the context). Ids (items R^64, sources R^32, events R^32) are random per episode.
Policy (ignorable): per-episode Dirichlet weights over sources and items (concentration logU[0.2, 5]), type mix,
relay rate U[0, 0.3], hidden-root rate 0.4, uniform root among earlier events. K = N / U[6,12], S = N / U[6,16]
(both >= 2), so K and S grow with N. Query: an item with no SPOT quote w.p. 1/2, else uniform; target v_q.
Held-out policies (still ignorable): conc (concentration logU[0.03, 0.1]), relay (relay rate U[0.5, 0.6]).
Exact Bayes: mixture over <= 2^8 noise hypotheses of linear-Gaussian solves (latents v, b, hidden-root noises;
relay of a visible root -> difference to the root). Tested against an independent covariance-form brute force
(all relay cases), Monte Carlo calibration (PIT), an ignorable value-adaptive policy (calibrated) and a value-leaking
policy (detected).

References: prior; freq_only (unknown sources at prior-mean variance: no in-context noise inference; weights depend
only on counts and known sigmas); bias_naive (biases as independent per-quote noise); prov_naive (relays as
independent quotes); noise_oracle (true sigmas; cheating bound); bayes.
Reference gaps over Bayes measured before training (nats, 400 episodes, N=128, dir): freq_only 0.106 (unk split
0.122), bias_naive 0.485, prov_naive 0.023 (relay policy: 0.075), noise_oracle -0.002, prior 1.097.

Diagnostic sets (harness policies c4, c7):
- C4 frequency-matched calibration pairs: q has exactly one SPOT from unknown-noise A and one from unknown-noise B;
  A and B each quote other items exactly 12 times; known calibrators measure the other items. Member 0: A low,
  B high; member 1: swapped; identical structure, ids, query readings and noise draws. Kept iff
  KL(Bayes_0 || Bayes_1) >= 0.3 nats (pre-registered effect size). Measured: median 4.6 nats; KL(Bayes||freq_only)
  median 0.61, min 0.36 (>= 0.1 on 100% of episodes, so freq_only is distinct from Bayes).
- C7: diamond (one root SPOT, paths of length 2 to q) vs k-anchor (k root SPOTs) for k = 2 (D/T) and 4 (D4/T4);
  provenance pairs DR/DI (visible root + relay vs + independent quote), HR/HI and HR4/HI4 (2 or 4 relays of a hidden
  root vs independent quotes, variance matched). Within a pair: same token count, sources, path lengths, edge noise.
  Measured KL(Bayes || naive) median at N=96: D 0.08, D4 0.37, DR 0.05, HR 0.05 (HR4 ~0.3).

Arms (harness SetPFN, no sink: pre-norm, no positions, context attends to context, query reads; 4 heads, MLP 4x,
mix 4; lr 5e-4, warmup 1000, cosine, AdamW wd 0.01, bs 256, 30,000 steps, best.pt by val NLL on training dist):
  L2w128, L4w128, L4w64; seeds 0, 1 (pilot = screen; crux claims need >= 10 seeds). Train N log-uniform [8, 128].
Eval: dir/conc/relay at N in {64, 128, 256, 512} (1000 eps), 1024 (200), 2048 (100); C4 at N=96 (200 pairs) and
512 (50 pairs); C7 at N=96 (50 sets) and 512 (10 sets).

Metrics: excess NLL over Bayes and KL(Bayes||model) per cell and split (all, unk, relay, hidden_relay, q_noSPOT);
gap-to-Bayes of every reference; ability fraction (weak-ref NLL - model NLL) / (weak-ref NLL - Bayes NLL) for
freq_only (unk split), prov_naive (relay split), bias_naive, prior.

Pre-registered readings:
- R1 (C4, in-context source inference): per arm-seed at N=96, closure 1 - KL(B||model)/KL(B||freq_only) >= 0.5 on
  >= 80% of C4 episodes => passes; otherwise "no in-context source inference" for that model (DESIGN C4).
- R2 (C7 dedup): dedup fraction (model log-sd-ratio - naive) / (Bayes - naive) on D4/T4 and HR4/HI4 (N=96):
  >= 0.5 => deduplicates; <= 0.2 => double-counts; between => partial.
- R3 (scaling, transversal crux): slope of KL(Bayes||model) per doubling of N over N = 128..2048 (dir):
  <= +0.01 nats/doubling => scales; > +0.01 => gap grows (refuted for that arm).
- R4 (does the world discriminate shortcuts): yes iff, at N=128 (dir), the pre-training reference gaps for
  freq_only (unk split), bias_naive and the C4/C7 naive references are each >= 0.05 nats, AND the trained arms
  separate from at least one shortcut reference (ability fraction differs from that reference's 0 by > 2 se, or C4
  closure distribution differs from 0). prov_naive on the natural distribution is small (0.02-0.08); provenance is
  judged on the C7 sets.
- Training validity: if no arm gets within 0.10 nats of Bayes on dir N=128, the pilot is a training failure and is
  reported as such, not read for R1-R3.

Planned variant (coordinator, not in this pilot): graded overlap (Gaussian partial-information model): quotes as
sums over subsets of primitive measurements, overlap observed first, latent overlap level later.

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

## W2b amendment (written 2026-10-28 ~04:50Z, after the W2 pilot, before any W2b training)

Pilot outcome: training failure (no arm within 0.10 nats of Bayes at N=128; val gap 0.20-0.26 and still falling;
data generation ~170 ms/step was the bottleneck). Goal of W2b: a model that learns W2, so C4/C7 become informative.

Changes (world2_market.py, its evalsets and the references are unchanged; eval reuses the pilot evalsets):
- Fast path w2fast.py: batch generator, tokens and exact Bayes (per-item mixture moments and the query mixture)
  in torch on the GPU. Tested equal to the numpy path (tokens via pack(), Bayes via predictive()) on the same
  episodes; the generator matches the world2_market training distribution (same parameters; tested on summary
  statistics and Bayes NLL).
- Curriculum (fractions of total steps): 0-10% N in [8,24], K<=4, S<=4, U<=1, no relays; 10-25% N<=48, K<=8, S<=6,
  U<=4, relay rate <= 0.15; 25-40% N<=96, full K/S rule, U<=8, relay <= 0.3; 40-100% the pilot training
  distribution (N log-uniform [8,128]). Eval and best.pt use the full training distribution only.
- Steps: 150k target, capped by a per-VM time budget (steps fixed after a 500-step speed probe; logged).
  bf16 autocast; AdamW lr 5e-4, warmup 2000, cosine to 10%; bs 256; width 128, 4 heads, MLP 4x, mix 4.
- Arms (seed 0 each; screen):
  Q4, Q6   query-only NLL (control), depth 4 / 6
  D4, D6   dense: query NLL + mean over every item of CE(Bayes item posterior moments || model item Gaussian), via
           one extra query token per item (query tokens are never keys, so the query prediction function and the
           architecture are identical to Q; harness eval applies unchanged). This is the C5 supervision-density arm.
  SQ4, SD4 stepping stone (fewer moving parts): SPOT+DIFF only, all noise levels known, no relays, biases on;
           query-only / dense, depth 4. Read only through learning curves on its own distribution (not C4/C7).
- Learning curves every 5k steps: val gap (training distribution), gap at N=128 (dir), C4 N=96 (fraction with
  closure >= 0.5, median mean-shift fraction), C7 N=96 dedup fraction (D4/T4, HR4/HI4).

Readings: the pilot readings R1-R3 and the training-validity rule apply unchanged to Q4/Q6/D4/D6 (final eval with
best.pt). Added: R5 (C5 supervision density) D arm vs Q arm at equal depth: final val gap lower by > 0.05 nats and
the C4/C7 readings no worse => dense supervision helps; within 0.02 => no effect. Single seed: screen only.

## W2c amendment (written 2026-10-28 ~10:15Z, before any W2c training) — in-context dedup

W2b outcome: C4 learned at N=96 (80-94%), C7 dedup ~0 at N=96/512; gap grows with N. Goal: in-context dedup.
Hard deadline 14:30Z for results, so the arms are warm-started (deviation from from-scratch training).

Changes:
- Training mix "rich" (ignorable policy change only; generative model, oracle and eval sets unchanged): each episode
  is shared-rich with prob 0.6: relay rate U[0.3, 0.6], hidden-root rate 0.4, root choice concentrated on early
  events (pick = floor(count * u^4), so several relays per root), and with prob 1/2 sparse anchors (SPOT weight
  x0.15, so most items are reached through shared-root DIFF paths). Other episodes: the W2 training policy.
  Measured before training (below): share of episodes with relays or shared-root DIFF paths, and prov_naive gap.
- N_train log-uniform up to 512, drawn per batch (episodes in [n/2, n]); dense targets (D arm) only on batches with
  n <= 128 (cost); the query loss on every batch.
- Arms (seed 0): Q6-rich (from W2b Q6 best.pt), D6-rich (from W2b D6 best.pt), Q6-old (from W2b Q6 best.pt, W2
  training policy, same N range and steps budget: the control). No curriculum (warm start). Training stops at a
  fixed wall-clock deadline (13:10Z); steps are set from a speed probe; lr 3e-4, warmup 500, cosine.
- C7 eval sets are constructed graphs (c7_set), never produced by the training generator (held out).

Eval: harness eval on the pilot evalsets (gap vs N to 2048, C4 N=96/512, C7 N=96/512). Cue check on C7 (N=96 and
512), model only, relay pairs DR/DI, HR/HI, HR4/HI4: (a) no-prov: relay tokens get fresh unique event ids;
(b) no-flag: relay flag zeroed and origin-source block replaced by the quoting source; (c) both; (d) value ties:
in HI4 the four independent quotes are given one identical value (their mean) — does the model discount ties?
Dedup fraction recomputed under each.

Readings: R2 (C7) as before (>= 0.5 dedups; <= 0.2 does not), now for Q6-rich and D6-rich vs Q6-old.
R6 (cue): dedup that survives (b) but collapses under (a) => relies on provenance ids; survives (a) but collapses
under (b) => relies on the relay flag / origin fields; survives both but not (c) => uses either; drop under (d)
of the HI4 sd => ties are used as a copy cue. Single seed: screen.
- (added before launch) All W2c arms are evaluated with the final weights (model.pt), not best.pt: best.pt is chosen
  on the old-mix N<=128 validation set and could select the warm-start weights.

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
