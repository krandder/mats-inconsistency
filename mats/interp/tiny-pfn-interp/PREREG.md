# PREREG: tiny-PFN interpretability step 1, "Locate" (2026-10-27 07:50Z)

Written before any probe or attention statistic is computed on a trained model. Plan:
outbox/tiny-pfn-interp-plan-2026-10-27.md. Models and world: ../tiny-pfn/ (read only; world.py and pfn.py imported
from there, nothing modified). Steps 2-3 (intervene, replace) are out of scope.

## Networks
- Main: w32_h2_L2 + warm-started K=4 mixture head, seeds 0-4 (runs/w32_h2_L2_mix4_ws_s*/model.pt), R^16 test set.
- Small: w16_h2_L2 at R^4 + warm-started mixture head, seeds 2 and 4 (testset_ds4.npz).
- Plateau contrast: w32_h2_L2_mix4_s3 (cold start, stuck on the query-only plateau, gap 0.144).
- Untrained control for every network: same architecture, built after torch.manual_seed(seed) (the init the
  seed's original training run started from; the body init is identical for Gaussian and mixture heads).

## Data
The fixed test set (n = 16000, seed 20000). Episode-level split, fixed: episodes 0-11999 probe-train,
12000-15999 held-out. No other data.

## Sites and token types
Sites (residual stream, raw, not layer-normed): emb (after the input linear map), attn1, mlp1 (after block 1),
attn2, mlp2 (after block 2). A forward pass re-implemented from pfn.Block that captures attn outputs; checked to
reproduce the model's head output to < 1e-5.
Token types: LAB = labelled-question tokens (s, q<4), 12 per episode; QQ = query-question tokens (s, q=4), 3 per
episode; QT = the query token. Per-source targets are probed at LAB and QQ with "own source" targets: the sample
for token (s, q) gets source s's value; samples are stacked over sources (and over q for LAB), all tokens of an
episode on the same side of the split. Note fixed in advance: context tokens after attn2/mlp2 are never read by the
output (nothing attends after layer 2), so those sites are reported but cannot be on the computational path.

## Targets (exact, from the test set / world.py)
Intermediate (1): r = o_sq - y_q of the token itself (LAB only); r^2 (LAB only); SS_s = sum of the source's 4
labelled r^2; log SS_s.
Intermediate (2): p_s = labelled-only posterior over {0.1, 0.5, 2.0} for source s (3 components, each probed;
reported as the mean R^2 over the 3); E_s = expected precision sum_j p_sj / v_j. Secondary: the same two
computed from the full Bayes posterior (marginal of w_bayes, includes the query-reading coupling).
Intermediate (3): beta_s = Bayes weight on source s's query reading, sum_c w_c (1/v_cs) / PREC_c (so that the
Bayes mean = sum_s beta_s o_s*); probed at LAB and QQ. At QT (no source index possible): entropy of the 27-way
Bayes weights (secondary).
Intermediate (4): Bayes predictive mean and log variance (and variance) at QT; also at QQ (secondary).
Calibration target (secondary): the model's own predictive mean at QT (how linearly readable the output is).

## Probe
Ridge regression with intercept on standardised features (train mean/sd), lambda from 10^-3 ... 10^4 (15 values,
log-spaced) by 5-fold episode-grouped CV on the probe-train episodes (max mean CV R^2), refit on all probe-train
episodes, held-out R^2 = 1 - SSE/SST (SST around the held-out mean).
Baselines, same pipeline: (U) the untrained network of the same seed; (S) shuffled targets: probe-train targets
permuted across probe-train episodes (one fixed permutation, seed 0; tokens of an episode keep their within-episode
structure), evaluated on the true held-out targets.

## Decision rules (fixed now)
- A probe is POSITIVE for a network if held-out R^2 >= 0.50 and R^2 - max(R^2_U, R^2_S) >= 0.10.
- Main model: a (target, token type, site) is POSITIVE if positive in >= 4 of 5 seeds. "First appears" = the
  earliest site in the order emb, attn1, mlp1, attn2, mlp2 that is POSITIVE. R^2 >= 0.80 is reported as "strong".
- Small model: positive in both seeds. Plateau network: reported descriptively with the same rule.
- Attention, per layer and head: mean attention mass by (from type, to category); categories for LAB/QQ sources:
  self, own-source LAB, own-source QQ, other-source LAB, other-source QQ; for QT: self, LAB, QQ. "Source-structured"
  = per-token mass to own-source tokens / per-token mass to other-source tokens (same token kind) >= 2, in >= 4/5
  seeds, with the untrained ratio in [0.8, 1.25].
- Query-token attention vs Bayes precision weights: Pearson r over (episode, source) pairs, held-out episodes,
  between the QT's attention mass on source s (a: all 5 tokens of s; b: the QQ token of s only) and beta_s.
  "Tracks Bayes weights" = r >= 0.50 and r - |r_U| >= 0.20 in >= 4/5 seeds. Also reported against E_s.
- Within-source attention vs residual size (secondary): Pearson r between the attention weight from a QQ/LAB token
  to an own-source LAB token and that token's r^2.

## Symmetry checks
(i) Token-permutation equivariance (sanity, no positional encoding): permute the source blocks of tokens; every
site's activations must permute accordingly, max abs difference < 1e-4; head output unchanged.
(ii) Source relabelling: permute which identity vector goes with which source's readings (readings fixed in place).
The probes fit on unmodified episodes, applied to the relabelled episodes, must decode source s's own value (not
the value of the source whose identity vector the token now carries): pass if held-out R^2 on SS_s / E_s at the
first POSITIVE site drops by <= 0.05. Output: report mean |change| of the predictive mean, and, as a secondary
diagnostic for step 4, the sd of the predictive mean across a fresh redraw of all identity vectors, compared with the
sd of (model mean - Bayes mean).

## Outputs
probe_results.json, attn_results.json, symmetry_results.json, tables in the report
outbox/tiny-pfn-interp-step1-2026-10-27.md. The algorithm hypothesis there is a hypothesis for steps 2-3.
CPU only, OMP_NUM_THREADS=1, <= 6 processes, no GPU, no spending.

## Amendment 1: steps 2-4 (2026-10-27 08:02Z; written before any intervention, replacement or ID experiment)
Coordinator request after the step-1 report. Same networks (main s0-4, small s2/s4, plateau s3 descriptive), same
test set; step-1 results have been seen (they motivate the hypotheses tested here). Fitting (means for ablation,
reconstruction regressions) uses probe-train episodes 0-11999; all scores on held-out episodes 12000-15999 unless
stated. Code: steps.py (one process per network), CPU only, <= 6 processes.

### Step 2, intervene
Pairs: for each held-out episode A, one source s ~ U{0,1,2} and a level j ~ U{0,1,2} (rng seed 1); episode B = A
with source s's 4 labelled readings redrawn as y_q + N(0, VLEV[j]) (same y, same query readings, same identity
vectors). Bayes shift = Bayes(B) - Bayes(A) of the predictive mean and of the log predictive variance (exact,
world.references). Model shifts of the moment-matched mean and log variance, from:
 full swap (run B), (a) patch A's QQ_s residual state after MLP1 with B's, (b) patch the layer-1 attention row of
 A's QQ_s token (both heads) with B's (values stay A's), (c, extra) patch A's 4 LAB tokens of s after MLP1 with B's.
Controls: (R) random patch: add to A's QQ_s state after MLP1 a random direction (fixed rng) with the norm of that
pair's (B - A) difference; (U) the untrained network, same patches.
Metrics: OLS slope and R^2 of model shift on Bayes shift (mean; log variance), per network; also R^2 of the patched
shift on the model's own full-swap shift. Rule: a patch "carries the Bayes shift" if slope in [0.5, 1.5] and
R^2 >= 0.5 for the mean shift, while (R) and (U) have R^2 < 0.1; main model: in >= 4/5 seeds.

### Step 3, replace. Score = (NLL_ablated - NLL_replaced) / (NLL_ablated - NLL_model), held-out mean NLL.
3a-attn: layer-1 attention rows of the 3 QQ tokens, both heads := uniform over the source's own 4 LAB tokens.
  Ablation: the layer-1 attention output (after the output projection) at QQ tokens := its train mean.
  Secondary: only the source-pooling heads of that seed (step-1 ratio >= 2), ablating only those heads' contribution.
3a-mlp: MLP1 output at the 3 QQ tokens := c + M [log SS_s, E_s] (true values; c, M by least squares on train QQ
  tokens). Ablation: MLP1 output at QQ := train mean. Secondary: regressors [log SS_s, E_s, p_s].
3a-both: both of the above together (ablation: both ablated).
3b: layer-2 attention row of the query token, both heads: its mass on the 3 QQ tokens is redistributed across
  sources proportional to the Bayes beta_s (total QQ mass, LAB mass and self mass unchanged). Ablation: the same
  mass split uniformly across sources (= the mean pattern, by source symmetry). Secondary: LAB mass also
  redistributed across sources prop. to beta_s (within-source shape kept), ablation uniform for both.
3-all: 3a-both + 3b (ablation: all ablated).
Reading: score >= 0.8 "replacement captures the component"; 0.5-0.8 partial; < 0.5 not captured. Reported with the
denominators (if NLL_ablated - NLL_model < 0.01 nats the score is flagged as unstable).

### Step 4, identity-vector noise (all 16000 test episodes; no fitting)
Original model: NLL gap, KL(Bayes||model) by kl_grid, sd(model mean - Bayes mean).
(a) Constant identity vectors: source ids = a fixed orthonormal set scaled to norm sqrt(ds)/4 (the typical training
  norm); question ids likewise (R^16: orthonormal; R^4: 5 vectors = a rotated regular simplex, pairwise cos -1/4).
  Same set in every episode; 3 different random frames (seeds 0-2), each reported.
(b) Average over K = 16 independent identity draws per episode: predictive = equal-weight mixture of the 16
  predictives (64 components); NLL gap, KL, mean error.
Reported: fraction of KL and of mean-error variance removed relative to the original. Rule: ID noise "is a major
part of the deviation" if (b) removes >= 25% of KL in >= 4/5 main seeds.

## Amendment 1b (2026-10-27 08:20Z; POST HOC, after seeing small_s2's pre-registered step-3 scores, which were
## negative for every 3a replacement; no other network's step-2/3/4 result seen)
Pre-registered 3a results stand and are reported. Added variants, labelled post hoc in the report:
- ph_attn_tilt_vs_flat: in layer 1, from each QQ token, both heads: keep the model's total mass on the own source's
  4 LAB tokens (and all other mass); within those 4, replace the distribution by softmax(c1_h r + c2_h r^2) of the
  key residuals, c fitted per head by least squares on centred log-weights (train). Ablation = uniform within the 4
  (the mean pattern by exchangeability of questions). Tests the step-1 "residual-tilted pooling" claim.
- ph_mlp_subspace: MLP1 output at QQ tokens: only its component in the 2-dim subspace spanned by the encoding
  directions of [log SS_s, E_s] (least-squares fit mo ~ c + z M on train) is replaced by c + z_true M (projected);
  ablation = the same with z at its train mean. The rest of the MLP1 output is kept.
- ph_tilt_subspace_3b: the two above plus 3b together (ablation: all three ablated).

## Amendment 2: B1, fresh-seed confirmation (2026-10-28 01:15Z; before any fresh-seed training)
Overnight plan Track B1 (coordinator; Kelvin approved the plan). Seen before writing: all step 1-4 results on seeds
0-4, small s2/s4, plateau s3.
Training (cloud VM, g2-standard-8, CPU only, 5 processes, OMP_NUM_THREADS=1; tiny-pfn world.py/pfn.py unchanged):
seeds 5-9 of w32_h2_L2, the exact warm-start recipe of tiny-pfn Amendments 2-3: Gaussian head, phase 1 fixed at
160k steps + 10k cooldown (`--phase1_fixed 160000`), then the K=4 mixture head warm-started from that checkpoint
(`--mixk 4 --init_from ...`, standard 40k/80k rule + cooldown), best-validation checkpoints, eval on the same test set
(regenerated on the VM by `pfn.py testset`; must match the local one: sha256 prefix of X b1ca6c6f2bb16aff, of w_bayes
5c86272722d6f6e3, else the run is invalid).
Analysis: locate.py and steps.py UNCHANGED (path made configurable only) on seeds 5-9 (tags main_s5..main_s9).
H1 (layer-1 rule), c-fitting procedure fixed as in steps.py: per layer-1 head, least squares of the within-source-
centred log attention weights from each QQ token to its own source's 4 labelled tokens on the centred r and r^2 of
those tokens, probe-train episodes 0-11999; replacement keeps each head's own-labelled mass and reshapes within it
by softmax(c1 r + c2 r^2), both heads, all three QQ tokens; ablation = uniform within (ph_attn_tilt_vs_flat).
Per seed: PASS if score >= 0.8 and NLL_replaced - NLL_model <= 0.002 nats (held-out). H1 confirmed if PASS in >= 4/5.
H2 (combined replacement, ph_tilt_subspace_3b, unchanged): PASS if score >= 0.8; confirmed if >= 4/5. Its NLL cost
relative to the model's own gap to Bayes is reported.
Seeds with test gap > 0.05 (plateau) are reported, excluded from H1/H2, and the threshold becomes "all but one of the
remaining seeds". Everything else (steps 1-4 numbers on fresh seeds) is descriptive replication.

## Amendment 3: B2, budget of the model's KL to Bayes (2026-10-28 01:15Z; before any B2 computation)
Networks: main seeds 0-4 (existing checkpoints). Fits on probe-train episodes 0-11999; every number on held-out
12000-15999. KL(Bayes || model) per episode by pfn.kl_grid (as in eval). Code: budget.py.
Part 1, nuisance symmetrisation (Bayes is exactly invariant to each; checked numerically on the modified episodes).
 N_id: resample all identity vectors. N_rot: replace each source's labelled residual vector r_s (4 values) by
 sqrt(SS_s) u, u uniform on the 3-sphere, o_sq = y_q + r'_sq. N_y: resample the 4 labelled y_q ~ N(0,1), keep r
 (o_sq = y'_q + r_sq). For every subset of {N_id, N_rot, N_y}: the predictive = equal-weight mixture of the
 network's predictives on K = 16 independent draws (rng fixed); KL of that mixture. KL_sys = KL with all three
 averaged = the part of the error that is a deterministic function of the Bayes-sufficient statistics
 (SS_1..3, query readings); "nuisance part" = KL_model - KL_sys, split by Shapley values over the 3 nuisances.
Part 2, layer-2 query-token weights vs Bayes (descriptive): per head, the query token's normalised weights over the 3
 QQ tokens w_hs; regress centred log w_hs on centred log beta_s (full Bayes) and, separately, on centred log beta_lab_s
 (labelled-only Bayes weights); slope and R^2; residual mean by quintile of query spread (max - min of o*) and of
 query coupling KL(w_bayes || w_lab). Replacement (a): within each head's QQ mass, w_hs := softmax_s(a_h log beta_s),
 a_h fitted on train (least squares, centred); (a_lab): the same with beta_lab. Measured: held-out NLL and KL vs model.
Part 3, MLP1 at the QQ tokens: output replaced by g(z), g a 2-hidden-layer MLP (64 GELU units each, Adam lr 3e-3,
 3000 full-batch steps, MSE, standardised inputs), fitted on train QQ tokens to the model's own MLP1 output, with
 (b_suff) z = (log SS_s, o_s*), and (b_faith) z = (log SS_s, mean r_s, mean r_s^3, o_s*, mean labelled y).
 Measured: held-out fit R^2, NLL and KL vs model.
Part 4, budget: KL_model; KL_sys (Part 1); KL with (a) / (a_lab) / (b_suff) / (b_faith) applied alone and with
 N-all averaging on top. A mechanism is called a "named share" if idealising it lowers KL by >= 10% of KL_model in
 >= 4/5 seeds. Mechanisms whose idealisation raises KL are reported as "not an error source by this test"
 (downstream is calibrated to them). Whatever no replacement removes is reported as unexplained.
Part 5 (ARC estimate) will be pre-registered separately before it is run.

## Amendment 3b (2026-10-28 02:03Z; POST HOC, after seeing main_s4's B2 results only)
Seen: on s4 the nuisance part is ~29% of KL (ids 20%, residual direction 9%, labelled y 0%); every mechanism
idealisation of Part 4 RAISES KL (beta^a weights 0.0071 -> 0.024, while the mean MSE drops 25%). Added, labelled
post hoc, for all 5 seeds:
- a_beta_H0 / a_beta_H1: the beta^a replacement in one layer-2 head only.
- Output-moment split of KL_model and of KL_sys: KL after shifting every mixture component so that the predictive
  mean equals the Bayes mean (mean part = drop), then also rescaling around it so that the variance equals the Bayes
  variance (variance part = further drop); the rest = shape part.
- K = 4 head floor: per episode, the best 4-component Gaussian mixture to the Bayes mixture (weighted EM on a
  1201-point grid, 300 iterations, init from the 4 heaviest Bayes components), KL(Bayes || fit); compared with the
  shape part.
- The mean-error share attributable to the layer-2 weights' deviation from beta^a = the fraction of mean MSE
  removed by a_beta (all heads / per head), with and without nuisance averaging.

## Amendment 3c (2026-10-28 03:48Z; POST HOC, after the B2 tables)
Seen: b_suff/b_faith replacements are catastrophic in seeds 0, 2, 3 (KL ~0.19) but not in 1, 4. Added: b_cross,
the same MLP1-at-QQ replacement with z = (log SS_s, o_s*, mean and |difference| of the other two query readings, the
other two log SS sorted) (budget_b2.py), all 5 seeds, held-out NLL/KL vs model. Tests whether the QQ code in the
one-pooling-head seeds also carries the other sources query readings.

## Amendment 4: B2 Part 5, ARC-style estimate of the expected excess loss (2026-10-28 03:49Z; before any Part-5 computation)
Target per main seed s (0-4): G_s = E_episode[KL(Bayes || model_s)] (= expected NLL gap). Pool: 100,000 fresh episodes
(pfn.fixed_set seed 30000, R^16 tokens); truth = pool mean of per-episode KL (kl_grid). Cost unit = one network forward
(one episode); Bayes-side quantities need no network and are reported separately (measured wall-time ratio).
Features (Bayes-side, network-free), chosen from the B2 budget: phi_mech = [1, spread^2/var_b (spread = max-min of the
query readings; carrier of the ID / residual-direction noise and mean parts), between-component variance of the Bayes
mixture / var_b (shape part), query coupling KL(w_bayes || w_lab) (layer-2 weight deviation)]; phi_generic = [1, var_b,
log var_b] (control: features with no mechanistic motivation).
For target seed s, theta fitted by least squares of KL on phi over the OTHER four seeds (pool episodes; target seed
never used); c_s = phi theta on the target pool.
Estimators at N network calls (N = 2^2 ... 2^13; 4000 resamples with replacement from the pool): F flat = 0; S sampling =
mean KL of N episodes; M0 mechanistic zero-call = pool mean of c_s (no network calls on s); M1 = M0 + mean of (KL - c_s)
over N episodes (control variate); G1 = the same with phi_generic. MSE against the truth.
Reported: N*_flat = smallest N where MSE_S < MSE_F; MSE of M0 and the N range where it beats S; variance ratio
Var(KL)/Var(KL - c) for M1 and G1. Criterion (per seed): M0 "beats sampling past the flat crossover" if MSE_M0 <
MSE_S at N = 2 N*_flat; M1 does if MSE_M1 < MSE_S at every N >= N*_flat. "Mechanistic features add value" if M1's
variance ratio exceeds G1's by >= 20% in >= 4/5 seeds. Code: arc.py.

## Amendment 5: layer-2 head-count test (2026-10-28 04:41Z; before any training of these arms)
Coordinator request after B2. Question: is the shared systematic part of the KL (KL_sys = held-out KL after averaging the
predictive over identity vectors, residual direction and labelled y; 57% of the main seeds' KL) a two-head layer-2 readout
bottleneck, or capacity? Arms (heads/pfn2.py = tiny-pfn pfn.py + per-layer heads/head dim + CUDA; layer 1 unchanged, 2 heads
x 16): base = w32_h2_L2 (layer 2: 2 heads x 16); h3 = layer 2 with 3 heads x 16; h4 = 4 heads x 16; w64 = layer 2 with 2
heads x 32 (same attention parameters as h4: capacity control for head count); L3 = three layers of 2 heads x 16 (extra-layer
control). Seeds 10-14 for every arm (25 runs). Recipe unchanged: Gaussian head, phase 1 fixed 160k + 10k cooldown, then the
K=4 mixture head warm-started from it (standard rule + cooldown). All arms incl. base trained on the GPU of one on-demand L4
VM (8 parallel processes) so that device effects are common to all arms; the CPU-trained seeds 0-4 are a secondary reference.
Analysis: heads/budget_lite.py = budget.py Part 1 (K = 16; none, id, rot, all) + output-moment split, held-out episodes.
Seeds with test NLL gap > 0.05 are excluded (plateau), reported. Metric per arm: mean KL_sys over kept seeds; R_arm = 1 -
KL_sys(arm) / KL_sys(base).
Predictions/thresholds: (i) two-head readout bottleneck supported if R_h4 >= 0.30, R_h3 >= 0.15, R_h4 >= R_h3, and h4 < base
by one-sided Welch t over seeds, p < 0.05. (ii) Head count (not capacity) if (i) and both R_w64 < 0.5 R_h4 and R_L3 < 0.5
R_h4. (iii) Capacity, not head count, if R_w64 >= 0.75 R_h4 or R_L3 >= 0.75 R_h4 (with R_h4 >= 0.15). (iv) If R_h4 < 0.15:
head count is not the limit; then R_w64 and R_L3 alone say whether capacity is. Otherwise: ambiguous. Also reported: raw KL,
nuisance part, moment split, per-episode correlation of KL_sys with the main seeds' mean KL_sys (same systematic pattern?).
Budget: cap $4; one VM, GPU_RUN_TIMEOUT 4 h (worst case ~$3.75); job queue seed-major (base, h4, h3, w64, L3) so a timeout
loses the controls of the last seeds first; deletion verified in GPU-LOG.

## Amendment 6: what the third layer computes (2026-10-28 08:50Z; before any analysis of the L3 models' internals)
Seen: Amendment 5 results (L3 cuts KL_sys by 46%; same per-episode pattern). Networks: L3 seeds 10-14 (heads/hruns) and,
as comparison, base 2-layer GPU seeds 10, 12, 13, 14 (s11 is on the plateau). CPU, <= 4 processes. Code: heads/l3.py.
Fits on episodes 0-11999; scores on held-out 12000-15999. Sites: emb, attn_l, mlp_l for every layer l.
(1) Probes (locate.py ridge pipeline, untrained-same-seed and shuffled baselines, same POSITIVE rule: R^2 >= 0.5 and >= 0.10
above both, in >= 4/5 L3 seeds / >= 3/4 base seeds). Targets: at QQ and LAB tokens (own source): E_lab, E_bayes, p_bayes
(full-Bayes per-source posterior over the 3 levels given labelled AND query readings), and the query-coupling parts
dE = E_bayes - E_lab and dp = p_bayes - p_lab (mean R^2 over 3 levels); beta. At QT: dmean = mean_bayes - mean_a,
dmean_coup = mean_bayes - mean_lab, log var_bayes.
(2) Readout weights: for each layer's QT attention heads, R^2 of centred log weights over the 3 QQ tokens on centred log beta
vs on centred log beta_lab (slope fitted on train), as in Amendment 3 Part 2.
(3) Replacements (held-out KL by kl_grid, overall and in the top quintile of query coupling KL(w_bayes||w_lab)):
A_dE_l: at the 3 QQ tokens after block l (l = 1, 2 for L3; l = 1 for base), remove the linearly decodable coupling: with b, c
the least-squares fit dE ~ b.h + c on train QQ tokens, h' = h - (b.h + c) b / |b|^2 (decoded coupling set to 0 = "no query
coupling"). A_rand_l: the same per-token shift norm along a random unit direction (fixed rng). A_attn3 / A_mlp3 / A_L3:
mean-ablate (train mean) layer 3's attention output / MLP output / both at QT. A_skipQQ2: for L3, the QQ tokens' layer-2
update (attn2 + mlp2 output) mean-ablated, so layer 3 reads QQ states as of after layer 1.
Rule: "the coupling carried into layer 3 is load-bearing" if A_dE_2 raises the L3 model's held-out KL by >= 50% of the KL
gap between base and L3 (mean over seeds; gap = KL_base - KL_L3 on held-out) while A_rand_2 raises it by < 25% of that gap,
in >= 4/5 seeds. Gain map (descriptive): share of the base-minus-L3 reduction of per-episode KL_sys (budget_lite) in each
quintile of coupling, between-component variance and query spread.
