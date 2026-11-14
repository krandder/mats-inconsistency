# PREREG: W2 market world, mechanism of in-context source calibration (C4) — 2026-10-28 ~10:50Z

Written before any probe, attention statistic or patch is computed on a trained model. Coordinator: opus-mats.
Question: the W2b models pass C4 at N=96 (closure >= 0.5 on 80-94% of pairs) but not C7. HOW do they do C4?
Method template: outbox/tiny-pfn-interp-plan-2026-10-27.md (locate -> intervene -> replace); this prereg covers
locate (probes, attention) and intervene (patching). Replacement is out of scope (hypothesis only).
World and models are read only (../worlds/, runs_w2b_1028/combined/); code here imports world2_market, w2fast,
harness and modifies nothing there. W2c (another agent) files and VMs are not touched.

## Networks
- D arm: D4_s0/best.pt (best C4 at N=96: 94%; depth 4, width 128, 4 heads; dense per-item supervision in training).
- Q arm: Q6_s0/best.pt (best Q C4: 84%; depth 6).
- Untrained control (U) for each: same architecture built right after torch.manual_seed(0) (the init the seed-0
  training run started from).
- Single training seed per arm exists. Claims are per network; "both arms" = holds in D4 and in Q6.

## Data
- PROBE set: natural W2 training-distribution episodes (w2fast.gen, stage_cfg(1.0), policy dir), N = 96 fixed,
  E = 2000, torch seed 4242. Episode split fixed: episodes 0-1499 probe-train, 1500-1999 held-out.
- C4 set: the pilot evalset C4 cell at N=96 (es_c4_96: 400 episodes = 200 frequency-matched pairs), regenerated
  from its seed exactly as w2_train.Curves does.
- Exact Bayes: extended copy of w2fast.bayes (bayes_ext) returning, per episode, mixture moments of every latent
  (items v and source biases b), the hypothesis weights, the unknown-source posterior P(sigma = high), and the
  linear Bayes weights beta_t on each quote for the query mean (sum_t beta_t y_t = Bayes mean, mixture of the
  per-hypothesis linear maps; relays of visible roots mapped back to token values). Checked: item moments equal
  w2fast.bayes to 1e-8; sum_t beta_t y_t equals the Bayes mean to 1e-6.

## Sites
Residual stream after the embedding (L0) and after each block (L1..Ld), raw (not layer-normed); additionally the
attention-only partial site inside each block (after attention, before MLP; "Lk.a"). A re-implemented forward
(manual attention so full attention matrices are available) must reproduce harness predict() to < 1e-4.
Note fixed in advance: context tokens never see the query token (keys = context only), so context states after
the last block are not read by anything; and in layer d only the query token's update matters.

## Targets (exact Bayes on the full context) and token types
Token types: CTX = all context tokens; UNK = context tokens whose quoting source has unknown noise; SPOTB = SPOT and
BASKET tokens (bias-bearing); SPOT = SPOT tokens; QT = query token.
Per-source targets, assigned to every token by its QUOTING source (samples stacked over tokens, all tokens of an
episode on the same side of the split):
- bmean: posterior mean of b_s (CTX).
- pH: posterior P(sigma_s = SIG_H) (UNK only; known sources excluded).
- logprec: log effective precision of a source's SPOT quote, -log(E[sigma_s^2 | D] + Var(b_s | D)) (CTX and UNK;
  on CTX it is partly given by the token's log-sigma feature, so UNK is the informative set).
Per-token (item) targets:
- vmean: posterior mean of the token's item functional a.v (SPOT: v_k; BASKET: mean; DIFF: spread) (CTX).
- resid: y_t - E[a.v | D] (the quote's residual against the posterior consensus = estimated c*b_s + eps) (SPOTB).
- resid2: resid^2 (SPOTB, UNK subset reported too).
- qmean / qlogvar: Bayes predictive mean and log variance of v_q at QT.
Raw-input controls reported alongside (not baselines for the rule): R^2 of the same probe on the token's raw input
features X (equivalent to L0 up to the linear map; shows what is trivially on the token).

## Probe
Ridge with intercept on standardised features, lambda in logspace(-3, 4, 15) by 5-fold episode-grouped CV on the
probe-train episodes, refit, held-out R^2 = 1 - SSE/SST. Per episode at most 48 tokens per token type are used
(fixed random subset, seed 0) to bound cost; QT has one sample per episode.
Baselines, same pipeline: (U) the untrained network, same site; (S) shuffled targets (probe-train targets permuted
across probe-train episodes, one fixed permutation seed 0), evaluated on true held-out targets.

## Decision rules
- POSITIVE (target, type, site, network): held-out R^2 >= 0.50 and R^2 - max(R^2_U, R^2_S) >= 0.10. STRONG: R^2 >= 0.80.
- "First appears" = earliest site (L0, L1.a, L1, L2.a, ...) that is POSITIVE.

## Attention (PROBE set held-out episodes, and C4 set)
A1. Query attention vs Bayes weights: per layer and head (and the head mean), Pearson r over (episode, context
    token) pairs between QT attention a_t and |beta_t| (beta normalised by sum |beta| within the episode).
    "Tracks Bayes weights" = r >= 0.50 and r - |r_U| >= 0.20 for some layer/head. Reported per layer.
A2. Source-identity matching: for each context token i (as attention query), mean per-key attention to OTHER tokens
    of the same quoting source divided by mean per-key attention to tokens of other sources (episode-mean of the
    per-token ratio of means; tokens with no same-source key skipped). "Source-structured" layer/head: ratio >= 2
    and untrained ratio in [0.8, 1.25]. Reported also restricted to UNK query tokens.
A3. Overlap graph: same, for keys from OTHER sources whose item set intersects the query token's item set vs keys
    from other sources with disjoint items. "Overlap-structured": ratio >= 2, untrained ratio in [0.8, 1.25].
A4. C4 paired attention: tokens Aq, Bq (the A and B quotes on the query item; identical inputs in both members).
    Per layer/head, QT attention log-ratio d = log a(Aq) - log a(Bq); between members, delta_d = d0 - d1
    (member 0: A low noise). Bayes: delta_beta = log|beta_A| - log|beta_B| difference. Report sign agreement
    (fraction of pairs with sign(delta_d) = sign(delta_beta)) and Pearson r. "QT reweights by reliability" at
    a layer/head = sign agreement >= 0.80 (untrained: by construction 0 difference -> report).

## Intervention (C4 set, both networks)
Patch residual states from member j into member i (both directions, 200 pairs x 2) at the site after block l
(l = 0 .. d-1; l = 0 is the embedding), for token groups:
  QQ   = {Aq, Bq} (the query-item quotes; identical inputs across members)
  EV   = the 2 x 12 other-item quotes of A and B (the only tokens whose inputs differ)
  SRC  = QQ + EV
  QT   = the query token
  CTX  = all context tokens
Flip fraction f = (m_patched - m_i) / (m_j - m_i) on the model predictive mean (pairs with |m_j - m_i| < 0.05
excluded; count reported); median over pairs and directions. Also the variance flip on log sd.
Reading: "flip location" for a group = earliest site with median f >= 0.5. Expected by construction: EV at L0 and
CTX at every site give f = 1 (sanity). Calibration evidence has LEFT the EV tokens by site l when EV-patch f at l
is <= 0.2; it has ARRIVED at QQ by site l when QQ-patch f >= 0.5.

## Hypothesis (step 4)
Written in the report as a hypothesis only, for replacement tests later; it must name layer(s), token group(s) and
an algorithmic description, and list which numbers above support or contradict it.

## Compute
CPU only, OMP_NUM_THREADS=1, <= 4 processes; no GPU, no spending. Outputs: out/*.json, report
outbox/w2-market-mechanism-2026-10-28.md.

## Amendment 1 (2026-10-28 ~10:55Z; after the C4 attention/patching results of both nets, before running these)
Seen: in both nets, calibration evidence leaves the A/B other-item quotes (EV) in block 3 and arrives at the
query-item quotes Aq/Bq in block 3 (EV-patch f 0.93-0.99 at L2 -> 0.01-0.02 at L3; QQ-patch 0.02-0.07 -> 0.94-0.97);
the query token reads QQ in block 4 (D4: last block; Q6: QT-patch 0.00 at L3 -> 0.79 at L4). Block-3 heads with very
high own-source attention from Aq exist in both nets. Added tests (C4 set, trained nets; untrained not applicable):
(i)  Edge ablation in block 3: for attention-query rows Aq and Bq, zero the attention to own-source other quotes
     (Aq -> AEV, Bq -> BEV), renormalise; all heads. Control: zero 12 random other keys per row (excluding own-source,
     self, Aq, Bq; fixed seed). Also: ablation in block 2 instead of 3 (same edges), and in blocks 2+3.
     Metric: shift fraction = median over pairs of (m0 - m1)_edited / (m0 - m1)_model; C4 closure (fraction of
     episodes with 1 - KL(B||model)/KL(B||freq_only) >= 0.5), as in w2_train.Curves.
     Reading: the Aq<-AEV edge in block 3 is CAUSAL for calibration if shift fraction <= 0.3 while the control >= 0.8.
(ii) Read-out mechanism in the read block (D4: block 4; Q6: block 4): set QT's attention to Aq and Bq, every head,
     to their mean (sum preserved). Reading: weighting is carried by the attention PATTERN if shift fraction <= 0.3,
     by the VALUES (precision-scaled content on Aq/Bq) if >= 0.7, mixed otherwise. Complement: swap the Aq and Bq
     value vectors' sources... (not run; noted only).
(iii) Probe transfer: ridge probes fitted on the PROBE set (trained net, same pipeline) for pH / logprec (UNK tokens)
     and resid2 (SPOTB tokens), applied to C4 Aq/Bq (pH, logprec of own source) and AEV/BEV (resid2) per site;
     report decoded-vs-Bayes Pearson r on C4 and the member-0 minus member-1 mean of decoded pH(A).

## Amendment 2 (2026-10-28 ~11:00Z; after locate, attention and Amendment-1 results; before running these)
Seen: block-1 heads attend to other sources' quotes on the same items (pooled overlap ratio 4-6); item consensus and
the quote residual are decodable after block-1 attention, resid^2 after the block-1 MLP. Added (C4 set, trained):
(iv) Overlap-edge ablation: for attention-query rows = EV tokens (A's and B's 24 other-item quotes), zero the
     attention to other-source tokens whose item set intersects the row token's items, renormalise; in block 1,
     block 2, blocks 1+2. Control: zero an equal number of random other-source non-overlapping keys per row (fixed
     seed). Reading as in (i): the overlap edge is CAUSAL for calibration if shift fraction <= 0.3, control >= 0.8.
(v)  D4 read-out equalisation in blocks 3+4 and block 3 alone (Q6 had blocks 4, 5, 4+5); same reading as (ii).
