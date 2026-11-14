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
